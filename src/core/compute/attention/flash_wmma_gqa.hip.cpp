// WMMA flash-attention prefill for head_dim 64 or 128, any GQA ratio that is even, F16 KV, causal with an
// optional sliding window and optional per-head attention sinks.
//
// The data flow is the proven one of fused_fa_gqa128 / flash_f16_dense (LDS-staged Q, K/V streaming in
// 32-key tiles, LoadFrag fragments, LDS-mediated S/P exchange, F32 online softmax, F16 P, F32 O accumulate on
// the 16x16x16 F16 WMMA), generalized so one model's prefill attention does not need its own kernel:
//   * heads / kv_heads are runtime parameters; a block serves one PAIR of query heads (both inside the same KV
//     group, so heads/kv_heads must be even), grid.y runs over head pairs;
//   * K/V are read from [position][kv_head][head_dim] F16 planes (the layout the engines keep or stage);
//   * a block only walks the key tiles its 16 queries can see: up to its last query (causal) and, with a
//     window, from the first key the window still reaches (the original swept every block to the end);
//   * optional sliding window: key kc is visible to query position qp iff kc <= qp and kc + window > qp.
// Math contract: QK scale folded into the F16 Q tile; F32 accumulate in both WMMA passes; online softmax
// identical to a full-row pass up to summation order. Callers with blocked-Q8 KV dequantize the visible range
// into F16 planes first (core blocked_kv::DequantizeKvRows).
#if defined(ENGINE_ENABLE_HIP)
#include "src/core/compute/attention/flash_wmma_gqa.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gufo::core::compute::flash_wmma_gqa {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// A fragment is 16 contiguous halves = 32 bytes (two 16-byte loads); every fragment base here is 16-byte aligned.
__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kRows = 16;         // queries per block (one WMMA row block)
constexpr std::uint32_t kKeys = 64;         // keys per K/V tile
constexpr std::uint32_t kKeyBlocks = kKeys / 16;   // 4 key blocks per tile
constexpr std::uint32_t kVtStride = kKeys + 8;     // V^T tile row stride
constexpr std::uint32_t kPStride = kKeys + 8;      // per-wave P row stride

// One wave = one query head x 16 queries; a block = kWaves heads of one KV group sharing the K/V tile.
// Softmax runs in registers on the WMMA accumulator layout (lane holds S[row = 2i + half][key = lane & 15]),
// only P round-trips through a per-wave LDS tile (C layout -> A layout). Per 64-key tile and wave: 32 WMMA for
// S = Q K^T, 32 for O += P V; three block barriers per tile.
template <std::uint32_t kHeadDim, std::uint32_t kWaves>
__global__ __launch_bounds__(32 * kWaves) void FlashWmmaKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    const float* __restrict__ sinks, float* __restrict__ out, std::uint32_t start_pos, std::uint32_t n_tokens, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t window, float scale) {
  constexpr std::uint32_t kThreads = 32 * kWaves;
  constexpr std::uint32_t kKSteps = kHeadDim / 16;    // WMMA k-steps for QK^T
  constexpr std::uint32_t kDimTiles = kHeadDim / 16;  // output dim tiles for PV
  constexpr std::uint32_t kKStride = kHeadDim + 8;    // K tile row stride (halves): 16B-aligned fragment reads
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;
  const std::uint32_t head = blockIdx.y * kWaves + wave;
  const std::uint32_t qpg = heads / kv_heads;
  const std::uint32_t kv_head = (blockIdx.y * kWaves) / qpg;  // all kWaves heads share one KV head
  const std::uint32_t query_start = blockIdx.x * kRows;
  const std::size_t kv_width = static_cast<std::size_t>(kv_heads) * kHeadDim;

  __shared__ __half k_lds[kKeys][kKStride];
  __shared__ __half vt_lds[kHeadDim][kVtStride];
  __shared__ __half p_lds[kWaves][kRows][kPStride];

  // Q fragments for this wave's head: lane owns query row `sub`; F32 -> F16 with the score scale folded in.
  v16h q_frag[kKSteps];
  {
    const std::uint32_t qr = query_start + sub;
    const float* qrow = q + (static_cast<std::size_t>(qr) * heads + head) * kHeadDim;
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKSteps; ++ks) {
#pragma unroll
      for (std::uint32_t e = 0; e < 16; ++e) {
        q_frag[ks][e] = static_cast<_Float16>(qr < n_tokens ? qrow[ks * 16u + e] * scale : 0.0F);
      }
    }
  }

  v8f o_acc[kDimTiles] = {};
  float m_row[8], l_row[8];
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    // The sink is one extra softmax term (value zero): max starts at its raw logit, the denominator at 1 (held by
    // one of the 16 lanes of the row, since the per-lane partial sums are added up in the epilogue).
    m_row[i] = sinks != nullptr ? sinks[head] : -INFINITY;
    l_row[i] = (sinks != nullptr && sub == 0) ? 1.0F : 0.0F;
  }

  // Key range this block can see: keys <= its last query (causal), and with a window only from the first key
  // its first query still reaches.
  const std::uint32_t n_keys_all = start_pos + n_tokens;
  std::uint32_t key_hi = start_pos + query_start + kRows;
  if (key_hi > n_keys_all) key_hi = n_keys_all;
  const std::uint32_t first_q_pos = start_pos + query_start;
  const std::uint32_t key_lo = (window != 0 && first_q_pos + 1 > window) ? first_q_pos + 1 - window : 0;
  const std::uint32_t kb_begin = key_lo / kKeys;
  const std::uint32_t kb_end = (key_hi + kKeys - 1u) / kKeys;

  for (std::uint32_t kb = kb_begin; kb < kb_end; ++kb) {
    const std::uint32_t key_start = kb * kKeys;
    __syncthreads();  // previous tile's K/V^T/P reads are done
    // Cooperative tile load: 16 B per thread per step; K row-major, V transposed into vt_lds[dim][key].
    for (std::uint32_t i = tid; i < kKeys * (kHeadDim / 8); i += kThreads) {
      const std::uint32_t r = i / (kHeadDim / 8);
      const std::uint32_t c8 = (i % (kHeadDim / 8)) * 8u;
      const std::uint32_t kr = key_start + r;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (kr < key_hi) {
        const std::size_t off = static_cast<std::size_t>(kr) * kv_width + static_cast<std::size_t>(kv_head) * kHeadDim + c8;
        kv = *reinterpret_cast<const uint4*>(k_cache + off);
        vv = *reinterpret_cast<const uint4*>(v_cache + off);
      }
      *reinterpret_cast<uint4*>(&k_lds[r][c8]) = kv;
      const __half* vh = reinterpret_cast<const __half*>(&vv);
#pragma unroll
      for (std::uint32_t e = 0; e < 8; ++e) vt_lds[c8 + e][r] = vh[e];
    }
    __syncthreads();

    // S = Q K^T for 4 key blocks (16 keys each).
    v8f s_acc[kKeyBlocks] = {};
#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKSteps; ++ks) {
        const v16h kf = LoadFrag(&k_lds[kbk * 16u + sub][ks * 16u]);
        s_acc[kbk] = Wmma(q_frag[ks], kf, s_acc[kbk]);
      }
    }

    // Online softmax in registers: lane holds rows 2i + half_id (i = 0..7), key column `sub` of every block.
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      const std::uint32_t row = 2u * i + half_id;
      const std::uint32_t qrel = query_start + row;
      const bool row_live = qrel < n_tokens;
      const std::uint32_t qpos = start_pos + qrel;
      float sv[kKeyBlocks];
      float part_max = -INFINITY;
#pragma unroll
      for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
        const std::uint32_t kc = key_start + kbk * 16u + sub;
        const bool valid = row_live && kc < key_hi && kc <= qpos && (window == 0 || kc + window > qpos);
        sv[kbk] = valid ? s_acc[kbk][i] : -INFINITY;
        part_max = fmaxf(part_max, sv[kbk]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < 16; off <<= 1) part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      const float m_new = fmaxf(m_row[i], part_max);
      const float alpha = isfinite(m_row[i]) ? __expf(m_row[i] - m_new) : 0.0F;
      float psum = 0.0F;
#pragma unroll
      for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
        const float pw = isfinite(sv[kbk]) ? __expf(sv[kbk] - m_new) : 0.0F;
        psum += pw;
        p_lds[wave][row][kbk * 16u + sub] = __half(pw);
      }
      l_row[i] = l_row[i] * alpha + psum;  // per-lane partial; reduced across the 16 lanes at the end
      m_row[i] = m_new;
#pragma unroll
      for (std::uint32_t dt = 0; dt < kDimTiles; ++dt) o_acc[dt][i] *= alpha;
    }
    __syncthreads();  // P tiles visible to every lane of their wave

    // O += P V
#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
      const v16h pf = LoadFrag(&p_lds[wave][sub][kbk * 16u]);
#pragma unroll
      for (std::uint32_t dt = 0; dt < kDimTiles; ++dt) {
        const v16h vf = LoadFrag(&vt_lds[dt * 16u + sub][kbk * 16u]);
        o_acc[dt] = Wmma(pf, vf, o_acc[dt]);
      }
    }
  }

  // Epilogue: reduce the per-lane partial row sums over the 16 lanes of each row, normalize, write.
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    float l = l_row[i];
#pragma unroll
    for (std::uint32_t off = 1; off < 16; off <<= 1) l += __shfl_xor(l, off);
    const std::uint32_t qrel = query_start + 2u * i + half_id;
    if (qrel >= n_tokens) continue;
    const float inv = l > 0.0F ? 1.0F / l : 0.0F;
    float* orow = out + (static_cast<std::size_t>(qrel) * heads + head) * kHeadDim;
#pragma unroll
    for (std::uint32_t dt = 0; dt < kDimTiles; ++dt) orow[dt * 16u + sub] = o_acc[dt][i] * inv;
  }
}

template <std::uint32_t kHeadDim>
static void Launch(const float* q, const __half* k, const __half* v, const float* sinks, float* out,
                   std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t heads, std::uint32_t kv_heads,
                   std::uint32_t window, float scale, hipStream_t stream) {
  const std::uint32_t qpg = heads / kv_heads;
  // Waves per block = query heads sharing one K/V tile (they must lie in the same KV group).
  if (qpg % 4 == 0) {
    const dim3 grid((n_tokens + kRows - 1) / kRows, heads / 4);
    hipLaunchKernelGGL((FlashWmmaKernel<kHeadDim, 4>), grid, dim3(128), 0, stream, q, k, v, sinks, out, start_pos,
                       n_tokens, heads, kv_heads, window, scale);
  } else {
    const dim3 grid((n_tokens + kRows - 1) / kRows, heads / 2);
    hipLaunchKernelGGL((FlashWmmaKernel<kHeadDim, 2>), grid, dim3(64), 0, stream, q, k, v, sinks, out, start_pos,
                       n_tokens, heads, kv_heads, window, scale);
  }
}

}  // namespace

bool FlashWmmaPrefill(const float* q, const __half* k_cache, const __half* v_cache, const float* sinks, float* out,
                      std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t heads, std::uint32_t kv_heads,
                      std::uint32_t head_dim, std::uint32_t window, float score_scale, hipStream_t stream) {
  if (!IsFlashWmmaSupported(heads, kv_heads, head_dim) || q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      out == nullptr || n_tokens == 0) {
    return false;
  }
  if (head_dim == 128)
    Launch<128>(q, k_cache, v_cache, sinks, out, n_tokens, start_pos, heads, kv_heads, window, score_scale, stream);
  else
    Launch<64>(q, k_cache, v_cache, sinks, out, n_tokens, start_pos, heads, kv_heads, window, score_scale, stream);
  return true;
}

}  // namespace gufo::core::compute::flash_wmma_gqa

#endif  // defined(ENGINE_ENABLE_HIP)
