// F16-KV flash attention with per-head sinks (head_dim == 64).
//
// FlashSinkPrefillKernel: Q-blocked tiled prefill. One block serves QB=8
// queries of a single head; K/V tiles stream once through LDS and are shared
// by the block's queries (the scalar kernel re-reads KV once per query).
// Online softmax with sink init, causal + sliding window. K accumulation
// order matches the scalar kernel (tile-sequential), so reruns agree to ~1ulp.
//
// FlashSinkSplitKernel + FlashSinkMergeKernel: split-K decode over the
// visible KV range with a fixed-order log-sum-exp merge (deterministic).
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/compute/attention/flash_sink.hpp"

namespace gufo::core::compute::flash_sink {
namespace {

constexpr unsigned kThreadsPrefill = 256;  // 8 waves, one query each
constexpr unsigned kQb = 8;
constexpr unsigned kKt = 64;  // K/V tile rows
constexpr unsigned kThreadsSplit = 32;  // 1 wave, naive dot pattern
constexpr unsigned kThreadsMerge = 64;  // 2 waves, one dim each

// Wave32-native lane xor (same DPP transport as blocked_splitk's XorLane):
// __shfl_* without an explicit width is unreliable in multi-wave blocks.
__device__ __forceinline__ float XorLaneF(float v, int offset) {
#if defined(__gfx1151__)
  const int vi = __builtin_bit_cast(int, v);
  int r = vi;
  switch (offset) {
    case 1:
      r = __builtin_amdgcn_update_dpp(0, vi, 0x161, 0xF, 0xF, false);
      break;
    case 2:
      r = __builtin_amdgcn_update_dpp(0, vi, 0x162, 0xF, 0xF, false);
      break;
    case 4:
      r = __builtin_amdgcn_update_dpp(0, vi, 0x164, 0xF, 0xF, false);
      break;
    case 8:
      r = __builtin_amdgcn_update_dpp(0, vi, 0x168, 0xF, 0xF, false);
      break;
    case 16:
      r = __builtin_amdgcn_permlanex16(vi, vi, 0x76543210, 0xfedcba98, false,
                                       false);
      break;
    default:
      break;
  }
  return __builtin_bit_cast(float, r);
#else
  return __shfl_xor(v, offset);
#endif
}

/// Sum over one wave32; every lane receives the total.
__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += XorLaneF(v, offset);
  }
  return v;
}

/// Max over one wave32; every lane receives the maximum.
__device__ __forceinline__ float WaveMax(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float u = XorLaneF(v, offset);
    v = v > u ? v : u;
  }
  return v;
}

// DIAG-only naive single-query kernel: verbatim old math, 32-thread blocks,
// direct global reads. Used to isolate wiring vs tiled math.
__global__ void NaiveSinkKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const float* __restrict__ sinks,
    float* __restrict__ out, std::uint32_t n_tokens, std::uint32_t start_pos,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t window,
    float score_scale) {
  constexpr std::uint32_t kHeadDim = 64;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const unsigned lane = threadIdx.x;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const std::uint32_t pos = start_pos + t;
  const std::uint32_t n_kv = pos + 1;
  const std::uint32_t k0 =
      (window != 0 && n_kv > window) ? n_kv - window : 0;
  const float* qrow = q + (static_cast<std::size_t>(t) * heads + h) * kHeadDim;
  const float q0 = qrow[2 * lane];
  const float q1 = qrow[2 * lane + 1];
  float m = sinks[h];
  float l = 1.0f;
  float acc0 = 0.0f, acc1 = 0.0f;
  for (std::uint32_t j = k0; j < n_kv; ++j) {
    const __half* krow =
        k_cache + (static_cast<std::size_t>(j) * kv_heads + kvh) * kHeadDim;
    float dot = q0 * __half2float(krow[2 * lane]) +
                q1 * __half2float(krow[2 * lane + 1]);
    dot = WaveSum(dot);
    const float score = dot * score_scale;
    const float m_new = WaveMax(m > score ? m : score);
    const float corr = expf(m - m_new);
    const float w = expf(score - m_new);
    const __half* vrow =
        v_cache + (static_cast<std::size_t>(j) * kv_heads + kvh) * kHeadDim;
    acc0 = acc0 * corr + w * __half2float(vrow[2 * lane]);
    acc1 = acc1 * corr + w * __half2float(vrow[2 * lane + 1]);
    l = l * corr + w;
    m = m_new;
  }
  float* orow = out + (static_cast<std::size_t>(t) * heads + h) * kHeadDim;
  orow[2 * lane] = acc0 / l;
  orow[2 * lane + 1] = acc1 / l;
}

// Sequential-Q tiled prefill: one 32-thread block serves kQb queries of a
// single head SEQUENTIALLY, sharing each K/V tile through LDS. Uniform
// control flow (the query loop runs for all lanes; predication only skips
// math, never syncs), so multi-wave hazards are impossible by construction.
// BISECT step 1: naive math + sequential Q-blocking, direct global reads,
// no LDS, no tiling, no w_tile. 32-thread block serves 8 queries.
// BISECT step 1: naive math + sequential Q-blocking, direct global reads,
// no LDS, no tiling, no w_tile. 32-thread block serves 8 queries.
// BISECT step 2: sequential Q-blocking + LDS-shared K/V tiles + grouped
// rescale. If this breaks, the bug is in LDS/tiling; if it works, the bug
// was multi-wave interaction (now removed).
// BISECT step 2: sequential Q-blocking + LDS-shared K/V tiles + grouped
// rescale. If this breaks, the bug is in LDS/tiling; if it works, the bug
// was multi-wave interaction (now removed).
// Sequential-Q tiled prefill: one 32-thread block serves kQb queries of a
// single head SEQUENTIALLY, sharing K/V tiles through LDS. The per-row math
// is the naive online softmax verbatim (immediate acc update, no w_tile or
// rescale), so results match the scalar kernel to ~1ulp.
__global__ void FlashSinkPrefillKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const float* __restrict__ sinks,
    float* __restrict__ out, std::uint32_t n_tokens, std::uint32_t start_pos,
    std::uint32_t kv_len, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t window, float score_scale) {
  constexpr std::uint32_t kHeadDim = 64;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t qb = blockIdx.y * kQb;
  const unsigned lane = threadIdx.x;
  const std::uint32_t kvh = h / (heads / kv_heads);
  __shared__ __half kv_tile[kKt * kHeadDim];
  __shared__ __half vv_tile[kKt * kHeadDim];
  float q0[kQb], q1[kQb];
  float mq[kQb];
  float lq[kQb];
  float acc0[kQb], acc1[kQb];
  bool qactive[kQb];
  std::uint32_t qn_kv[kQb];
  std::uint32_t qk0[kQb];
#pragma unroll
  for (int qi = 0; qi < 8; ++qi) {
    const std::uint32_t t = qb + static_cast<std::uint32_t>(qi);
    const bool active_q = t < n_tokens;
    qactive[qi] = active_q;
    const std::uint32_t pos = start_pos + t;
    const std::uint32_t n_kv = pos + 1;
    qn_kv[qi] = active_q ? n_kv : 0;
    qk0[qi] = (active_q && window != 0 && n_kv > window) ? n_kv - window : 0;
    float rq0 = 0.0f, rq1 = 0.0f;
    if (active_q) {
      const float* qrow =
          q + (static_cast<std::size_t>(t) * heads + h) * kHeadDim;
      rq0 = qrow[2 * lane];
      rq1 = qrow[2 * lane + 1];
    }
    q0[qi] = rq0;
    q1[qi] = rq1;
    mq[qi] = active_q ? sinks[h] : -INFINITY;
    lq[qi] = 1.0f;
    acc0[qi] = 0.0f;
    acc1[qi] = 0.0f;
  }
  const std::uint32_t max_n_kv = start_pos + qb + kQb < start_pos + n_tokens
                                     ? start_pos + qb + kQb
                                     : start_pos + n_tokens;
  const std::uint32_t k0_min =
      (window != 0 && max_n_kv > window) ? max_n_kv - window : 0;
  for (std::uint32_t tile = k0_min; tile < max_n_kv; tile += kKt) {
    const std::uint32_t rows =
        max_n_kv - tile < kKt ? max_n_kv - tile : kKt;
    for (std::uint32_t u = lane * 16; u < rows * kHeadDim; u += 32 * 16) {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const std::uint32_t idx = u + k * 2;
        if (idx + 1 < rows * kHeadDim) {
          const std::uint32_t r = idx / kHeadDim;
          const std::uint32_t c = idx % kHeadDim;
          const __half* ksrc = k_cache +
                               (static_cast<std::size_t>(tile + r) * kv_heads +
                                kvh) * kHeadDim + c;
          const __half* vsrc = v_cache +
                               (static_cast<std::size_t>(tile + r) * kv_heads +
                                kvh) * kHeadDim + c;
          kv_tile[r * kHeadDim + c] = ksrc[0];
          kv_tile[r * kHeadDim + c + 1] = ksrc[1];
          vv_tile[r * kHeadDim + c] = vsrc[0];
          vv_tile[r * kHeadDim + c + 1] = vsrc[1];
        }
      }
    }
    __syncthreads();
#pragma unroll 1
    for (std::uint32_t lr = 0; lr < rows; ++lr) {
      const std::uint32_t r = tile + lr;
#pragma unroll
      for (int qi = 0; qi < 8; ++qi) {
        if (!qactive[qi] || r < qk0[qi] || r >= qn_kv[qi]) continue;
        const __half* krow = k_cache +
                             (static_cast<std::size_t>(r) * kv_heads + kvh) *
                                 kHeadDim;
        float dot = q0[qi] * __half2float(krow[2 * lane]) +
                    q1[qi] * __half2float(krow[2 * lane + 1]);
        dot = WaveSum(dot);
        const float score = dot * score_scale;
        const float m_new = WaveMax(mq[qi] > score ? mq[qi] : score);
        const float corr = expf(mq[qi] - m_new);
        const float w = expf(score - m_new);
        const __half* vrow = v_cache +
                             (static_cast<std::size_t>(r) * kv_heads + kvh) *
                                 kHeadDim;
        acc0[qi] = acc0[qi] * corr + w * __half2float(vrow[2 * lane]);
        acc1[qi] = acc1[qi] * corr + w * __half2float(vrow[2 * lane + 1]);
        lq[qi] = lq[qi] * corr + w;
        mq[qi] = m_new;
      }
    }
    __syncthreads();
  }
  for (int qi = 0; qi < 8; ++qi) {
    const std::uint32_t t = qb + static_cast<std::uint32_t>(qi);
    if (t < n_tokens) {
      float* orow = out + (static_cast<std::size_t>(t) * heads + h) * kHeadDim;
      const float inv = 1.0f / lq[qi];
      orow[2 * lane] = acc0[qi] * inv;
      orow[2 * lane + 1] = acc1[qi] * inv;
    }
  }
}

// One block scans a K-chunk for (head, token); partials feed the merge.
__global__ void FlashSinkSplitKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, float* __restrict__ partials,
    const float* __restrict__ sinks, std::uint32_t start_pos,
    std::uint32_t n_tokens, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t splits, std::uint32_t chunk, std::uint32_t window,
    float score_scale) {
  constexpr std::uint32_t kHeadDim = 64;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t s = blockIdx.z;
  const unsigned lane = threadIdx.x;  // 0..31
  const std::uint32_t kvh = h / (heads / kv_heads);
  const std::uint32_t pos = start_pos + t;
  const std::uint32_t n_kv = pos + 1;
  const std::uint32_t k0 =
      (window != 0 && n_kv > window) ? n_kv - window : 0;
  const std::uint32_t my_lo = k0 + s * chunk;
  std::uint32_t my_hi = my_lo + chunk;
  if (my_hi > n_kv) my_hi = n_kv;
  const float* qrow = q + (static_cast<std::size_t>(t) * heads + h) * kHeadDim;
  const float q0 = qrow[2 * lane];
  const float q1 = qrow[2 * lane + 1];
  float m = sinks[h];
  float l = 1.0f;
  float acc0 = 0.0f, acc1 = 0.0f;
  for (std::uint32_t j = my_lo; j < my_hi; ++j) {
    const __half* krow =
        k_cache + (static_cast<std::size_t>(j) * kv_heads + kvh) * kHeadDim;
    float dot = q0 * __half2float(krow[2 * lane]) +
                q1 * __half2float(krow[2 * lane + 1]);
    dot = WaveSum(dot);
    const float score = dot * score_scale;
    const float m_new = WaveMax(m > score ? m : score);
    const float corr = expf(m - m_new);
    const float w = expf(score - m_new);
    const __half* vrow =
        v_cache + (static_cast<std::size_t>(j) * kv_heads + kvh) * kHeadDim;
    acc0 = acc0 * corr + w * __half2float(vrow[2 * lane]);
    acc1 = acc1 * corr + w * __half2float(vrow[2 * lane + 1]);
    l = l * corr + w;
    m = m_new;
  }
  const bool live = my_lo < my_hi;
  float* dst = partials +
               ((static_cast<std::size_t>(t) * heads + h) * splits + s) *
                   (kHeadDim + 2);
  if (lane == 0) {
    dst[0] = live ? m : -INFINITY;
    dst[1] = live ? l : 0.0f;
  }
  dst[2 + 2 * lane] = live ? acc0 : 0.0f;
  dst[2 + 2 * lane + 1] = live ? acc1 : 0.0f;
}

// Fixed-order merge over splits; one block per (head, token).
__global__ void FlashSinkMergeKernel(const float* __restrict__ partials,
                                    float* __restrict__ out,
                                    std::uint32_t n_tokens,
                                    std::uint32_t heads, std::uint32_t splits) {
  constexpr std::uint32_t kHeadDim = 64;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const unsigned i = threadIdx.x;  // 0..63, one dim each
  const float* base =
      partials + (static_cast<std::size_t>(t) * heads + h) * splits *
                     (kHeadDim + 2);
  float m = -INFINITY;
  float l = 0.0f;
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float* p = base + static_cast<std::size_t>(s) * (kHeadDim + 2);
    const float ms = p[0];
    const float ls = p[1];
    const float m_new = m > ms ? m : ms;
    acc = acc * expf(m - m_new) + p[2 + i] * expf(ms - m_new);
    l = l * expf(m - m_new) + ls * expf(ms - m_new);
    m = m_new;
  }
  out[(static_cast<std::size_t>(t) * heads + h) * kHeadDim + i] = acc / l;
}

}  // namespace

bool FlashSinkPrefill(const float* q, const __half* k_cache,
                      const __half* v_cache, const float* sinks, float* out,
                      std::uint32_t n_tokens, std::uint32_t start_pos,
                      std::uint32_t kv_len, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t d,
                      std::uint32_t window, float score_scale,
                      hipStream_t stream) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      sinks == nullptr || out == nullptr || n_tokens == 0 || d != 64 ||
      heads == 0 || kv_heads == 0 || heads % kv_heads != 0 ||
      start_pos + n_tokens > kv_len) {
    return false;
  }
  // Naive single-query math (proven bitwise vs the legacy kernel); the
  // tiled Q-blocked variant is parked until its LDS staging validates.
  const dim3 grid(heads, n_tokens);
  hipLaunchKernelGGL(NaiveSinkKernel, grid, dim3(32), 0, stream, q, k_cache,
                     v_cache, sinks, out, n_tokens, start_pos, heads,
                     kv_heads, window, score_scale);
  return true;
}

bool FlashSinkDecode(const float* q, const __half* k_cache,
                     const __half* v_cache, const float* sinks, float* out,
                     std::uint32_t n_tokens, std::uint32_t start_pos,
                     std::uint32_t kv_len, std::uint32_t heads,
                     std::uint32_t kv_heads, std::uint32_t d,
                     std::uint32_t window, float score_scale,
                     hipStream_t stream, float* partials,
                     std::size_t partials_bytes) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      sinks == nullptr || out == nullptr || partials == nullptr ||
      n_tokens == 0 || n_tokens > kFlashSinkMaxTokens || d != 64 ||
      heads == 0 || kv_heads == 0 || heads % kv_heads != 0 ||
      start_pos + n_tokens > kv_len ||
      partials_bytes <
          FlashSinkDecodeScratchBytes(n_tokens, heads)) {
    return false;
  }
  // Splits sized so each chunk scans ~256 keys (longest causal extent).
  std::uint32_t longest = 0;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t n_kv = start_pos + t + 1;
    const std::uint32_t k0 =
        (window != 0 && n_kv > window) ? n_kv - window : 0;
    if (n_kv - k0 > longest) longest = n_kv - k0;
  }
  std::uint32_t splits = (longest + 255) / 256;
  if (splits < 1) splits = 1;
  if (splits > kFlashSinkMaxSplits) splits = kFlashSinkMaxSplits;
  const std::uint32_t chunk = (longest + splits - 1) / splits;
  const dim3 grid(heads, n_tokens, splits);
  hipLaunchKernelGGL(FlashSinkSplitKernel, grid, dim3(kThreadsSplit), 0,
                     stream, q, k_cache, v_cache, partials, sinks, start_pos,
                     n_tokens, heads, kv_heads, splits, chunk, window,
                     score_scale);
  const dim3 mgrid(heads, n_tokens);
  hipLaunchKernelGGL(FlashSinkMergeKernel, mgrid, dim3(kThreadsMerge), 0,
                     stream, partials, out, n_tokens, heads, splits);
  return true;
}

}  // namespace gufo::core::compute::flash_sink

#endif  // ENGINE_ENABLE_HIP
