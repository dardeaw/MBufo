#include "src/models/inkling/kernels/kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace mbufo::models::inkling::kernels {
namespace {

// ---------------------------------------------------------------------------
// Host-side NVFP4 conversion (unchanged skeleton helper).
// ---------------------------------------------------------------------------

// Doubled E2M1 codebook (matches the kvalues_mxfp4 convention: the UE4M3
// scale helper folds the compensating 0.5, so no extra halving here).
// ref: OCP microscaling formats (E2M1: {0,0.5,1,1.5,2,3,4,6} x sign).
constexpr float kE2M1Doubled[16] = {
    0.0F,   1.0F,  2.0F,  3.0F,  4.0F,   6.0F,   8.0F,   12.0F,
    0.0F,  -1.0F, -2.0F, -3.0F, -4.0F,  -6.0F,  -8.0F,  -12.0F,
};

// UE4M3: unsigned, 4 exponent bits (bias 7), 3 mantissa bits.
// Returns value * 0.5 to match the doubled E2M1 codebook above.
float Ue4M3ToFp32(std::uint8_t x) {
  if (x == 0 || x == 0x7F) {
    return 0.0F;
  }
  const int exp = (x >> 3) & 0xF;
  const int man = x & 0x7;
  float raw = 0.0F;
  if (exp == 0) {
    raw = ldexpf(static_cast<float>(man), -9);
  } else {
    raw = ldexpf(1.0F + static_cast<float>(man) / 8.0F, exp - 7);
  }
  return raw * 0.5F;
}

std::uint16_t F32ToF16Bits(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::int32_t exp = static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 112;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;
  if ((bits & 0x7FFFFFFFU) == 0) return static_cast<std::uint16_t>(sign);
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7BFFU);
  if (exp <= 0) return static_cast<std::uint16_t>(sign);
  const std::uint32_t half =
      sign | (static_cast<std::uint32_t>(exp) << 10U) | (mantissa >> 13U);
  const std::uint32_t round_bit = (mantissa >> 12U) & 1U;
  const std::uint32_t sticky = mantissa & 0xFFFU;
  if (round_bit != 0 && (sticky != 0 || (half & 1U) != 0)) {
    return static_cast<std::uint16_t>(half + 1U);
  }
  return static_cast<std::uint16_t>(half);
}

constexpr std::size_t kNvfp4BlockElems = 64;
constexpr std::size_t kNvfp4SubElems = 16;
constexpr std::size_t kNvfp4BlockBytes = 4 + kNvfp4BlockElems / 2;

// ---------------------------------------------------------------------------
// Device kernels.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0F / (1.0F + expf(-x));
}

// Block-wide sum over hidden (mirrors the laguna RMSNorm reduction shape so
// the numerics stay in the same family: per-thread strided sum, 32-lane
// shuffle fold, 8-way shared combine).
__device__ float BlockHiddenSum(float v) {
  for (int off = 16; off > 0; off /= 2) v += __shfl_down(v, off);
  __shared__ float s_sum[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) s_sum[warp] = v;
  __syncthreads();
  float total = 0.0F;
  if (warp == 0) {
    total = (lane < static_cast<int>(blockDim.x >> 5)) ? s_sum[lane] : 0.0F;
    for (int off = 16; off > 0; off /= 2) total += __shfl_down(total, off);
  }
  // Broadcast the warp-0 total to every thread: only warp 0 computes it
  // above, so returning `total` directly leaves warps 1+ with 0 (which made
  // every RMS/clip norm past lane 31 read rsqrt(0 + eps)).
  if (threadIdx.x == 0) s_sum[0] = total;
  __syncthreads();
  return s_sum[0];
}

__global__ void RmsNormGammaKernel(const float* x, const float* w, float* out,
                                   std::uint32_t hidden, float eps) {
  const std::uint32_t tok = blockIdx.x;
  const float* row = x + static_cast<std::size_t>(tok) * hidden;
  float* dst = out + static_cast<std::size_t>(tok) * hidden;
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = row[i];
    sum += v * v;
  }
  const float inv =
      rsqrtf(BlockHiddenSum(sum) / static_cast<float>(hidden) + eps);
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst[i] = row[i] * inv * w[i];
  }
}

__global__ void RmsNormNoGammaKernel(const float* x, float* out,
                                     std::uint32_t hidden, float eps) {
  const std::uint32_t tok = blockIdx.x;
  const float* row = x + static_cast<std::size_t>(tok) * hidden;
  float* dst = out + static_cast<std::size_t>(tok) * hidden;
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = row[i];
    sum += v * v;
  }
  const float inv =
      rsqrtf(BlockHiddenSum(sum) / static_cast<float>(hidden) + eps);
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst[i] = row[i] * inv;
  }
}

__global__ void AddResidualKernel(float* a, const float* b,
                                  std::size_t count) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) a[idx] += b[idx];
}

__global__ void StoreKvF16Kernel(const float* k_in, const float* v_in,
                                 __half* k_cache, __half* v_cache,
                                 std::uint32_t start_pos,
                                 std::uint32_t n_kv_heads,
                                 std::uint32_t head_dim) {
  const std::uint32_t tok = blockIdx.x;
  const std::size_t kv_stride = static_cast<std::size_t>(n_kv_heads) * head_dim;
  const std::size_t src_off = static_cast<std::size_t>(tok) * kv_stride;
  const std::size_t dst_off =
      static_cast<std::size_t>(start_pos + tok) * kv_stride;
  for (std::size_t i = threadIdx.x; i < kv_stride; i += blockDim.x) {
    k_cache[dst_off + i] = __float2half(k_in[src_off + i]);
    v_cache[dst_off + i] = __float2half(v_in[src_off + i]);
  }
}

// Static short-conv: exactly one thread owns each dim element and walks the
// tokens in order (blocks are dim-disjoint, no inter-block sync is needed).
// In-place safe (callers alias x == out): each element is read once, then
// overwritten by its sole owner thread. state[i*3+k] holds the input from
// k+1 steps ago for dim element i.

__global__ void StaticSconv1TokKernel(const float* __restrict__ x,
                                      float* __restrict__ state,
                                      const float* __restrict__ taps,
                                      float* __restrict__ out,
                                      std::uint32_t dim, float norm) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  const float t0 = taps[i * 4 + 3];
  const float t1 = taps[i * 4 + 2];
  const float t2 = taps[i * 4 + 1];
  const float t3 = taps[i * 4];
  const float s0 = state[i * 3];
  const float s1 = state[i * 3 + 1];
  const float s2 = state[i * 3 + 2];
  const float cur = x[i];
  const float conv = t0 * cur + t1 * s0 + t2 * s1 + t3 * s2;
  out[i] = (conv + cur) * norm;
  state[i * 3] = cur;
  state[i * 3 + 1] = s0;
  state[i * 3 + 2] = s1;
}

__global__ void StaticSconvParKernel(const float* __restrict__ src,
                                     const float* __restrict__ state,
                                     const float* __restrict__ taps,
                                     float* __restrict__ out,
                                     std::uint32_t n_tokens, std::uint32_t dim,
                                     float norm) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (i >= dim || t >= n_tokens) return;
  const float t0 = taps[i * 4 + 3];
  const float t1 = taps[i * 4 + 2];
  const float t2 = taps[i * 4 + 1];
  const float t3 = taps[i * 4];
  const float cur = src[static_cast<std::size_t>(t) * dim + i];
  const float x1 = (t >= 1U) ? src[static_cast<std::size_t>(t - 1U) * dim + i] : state[i * 3];
  const float x2 = (t >= 2U) ? src[static_cast<std::size_t>(t - 2U) * dim + i] : state[i * 3 + (1U - t)];
  const float x3 = (t >= 3U) ? src[static_cast<std::size_t>(t - 3U) * dim + i] : state[i * 3 + (2U - t)];
  const float conv = t0 * cur + t1 * x1 + t2 * x2 + t3 * x3;
  out[static_cast<std::size_t>(t) * dim + i] = (conv + cur) * norm;
}

__global__ void DynamicSconv1TokKernel(const float* __restrict__ x,
                                       float* __restrict__ state,
                                       const float* __restrict__ taps,
                                       const float* __restrict__ delta,
                                       float* __restrict__ out,
                                       std::uint32_t dim,
                                       std::uint32_t head_dim, float norm) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  const std::uint32_t head = i / head_dim;
  const float* dl = delta + head * 4;
  const float* tp = taps + static_cast<std::size_t>(i) * 4;
  const float cur = x[i];
  const float s0 = state[i * 3];
  const float s1 = state[i * 3 + 1];
  const float s2 = state[i * 3 + 2];
  const float conv = (tp[3] + dl[0]) * cur + (tp[2] + dl[1]) * s0 +
                     (tp[1] + dl[2]) * s1 + (tp[0] + dl[3]) * s2;
  out[i] = (conv + cur) * norm;
  state[i * 3] = cur;
  state[i * 3 + 1] = s0;
  state[i * 3 + 2] = s1;
}

__global__ void DynamicSconvParKernel(const float* __restrict__ src,
                                      const float* __restrict__ state,
                                      const float* __restrict__ taps,
                                      const float* __restrict__ delta,
                                      float* __restrict__ out,
                                      std::uint32_t n_tokens, std::uint32_t dim,
                                      std::uint32_t head_dim, float norm) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (i >= dim || t >= n_tokens) return;
  const std::uint32_t head = i / head_dim;
  const float* dl =
      delta + (static_cast<std::size_t>(t) * (dim / head_dim) + head) * 4;
  const float* tp = taps + static_cast<std::size_t>(i) * 4;
  const float cur = src[static_cast<std::size_t>(t) * dim + i];
  const float x1 = (t >= 1U) ? src[static_cast<std::size_t>(t - 1U) * dim + i] : state[i * 3];
  const float x2 = (t >= 2U) ? src[static_cast<std::size_t>(t - 2U) * dim + i] : state[i * 3 + (1U - t)];
  const float x3 = (t >= 3U) ? src[static_cast<std::size_t>(t - 3U) * dim + i] : state[i * 3 + (2U - t)];
  const float conv = (tp[3] + dl[0]) * cur + (tp[2] + dl[1]) * x1 +
                     (tp[1] + dl[2]) * x2 + (tp[0] + dl[3]) * x3;
  out[static_cast<std::size_t>(t) * dim + i] = (conv + cur) * norm;
}

__global__ void UpdateSconvStateKernel(const float* __restrict__ src,
                                       float* __restrict__ state,
                                       std::uint32_t n_tokens,
                                       std::uint32_t dim) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  const float old0 = state[i * 3];
  const float old1 = state[i * 3 + 1];
  const float s0 = src[static_cast<std::size_t>(n_tokens - 1U) * dim + i];
  const float s1 = (n_tokens >= 2U) ? src[static_cast<std::size_t>(n_tokens - 2U) * dim + i] : old0;
  const float s2 = (n_tokens >= 3U) ? src[static_cast<std::size_t>(n_tokens - 3U) * dim + i]
                                    : (n_tokens == 2U ? old0 : old1);
  state[i * 3] = s0;
  state[i * 3 + 1] = s1;
  state[i * 3 + 2] = s2;
}

__global__ void SaveSconvPrefixesKernel(const float* __restrict__ src,
                                        const float* __restrict__ old_state,
                                        float* __restrict__ pfx_out,
                                        std::size_t pfx_stride,
                                        std::uint32_t n_tokens,
                                        std::uint32_t dim) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (i >= dim || t >= n_tokens) return;
  const float old0 = old_state[i * 3];
  const float old1 = old_state[i * 3 + 1];
  const float s0 = src[static_cast<std::size_t>(t) * dim + i];
  const float s1 = (t >= 1U) ? src[static_cast<std::size_t>(t - 1U) * dim + i] : old0;
  const float s2 = (t >= 2U) ? src[static_cast<std::size_t>(t - 2U) * dim + i]
                             : (t == 1U ? old0 : old1);
  float* dst = pfx_out + static_cast<std::size_t>(t) * pfx_stride;
  dst[i * 3] = s0;
  dst[i * 3 + 1] = s1;
  dst[i * 3 + 2] = s2;
}

__global__ void StaticSconvKernel(const float* x, float* state,
                                  const float* taps, float* out,
                                  std::uint32_t n_tokens, std::uint32_t dim,
                                  std::uint32_t dim_off, float norm) {
  const std::uint32_t i = dim_off + threadIdx.x;
  if (i >= dim) return;
  // GGUF sconv taps are dim-major [dim, 4] (bit-exact vs HF torch
  // [dim,1,4] flat). Reference nn.Conv1d(K=4, padding=3) yields
  // out[t] = W3*x_t + W2*x_{t-1} + W1*x_{t-2} + W0*x_{t-3}: weight
  // index 0 taps the oldest lag, index 3 the current token (the
  // dynamic class documents the opposite convention for its own
  // static_weight, which the dynamic kernel keeps).
  const float t0 = taps[i * 4 + 3];
  const float t1 = taps[i * 4 + 2];
  const float t2 = taps[i * 4 + 1];
  const float t3 = taps[i * 4];
  float s0 = state[i * 3];
  float s1 = state[i * 3 + 1];
  float s2 = state[i * 3 + 2];
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const float cur = x[static_cast<std::size_t>(t) * dim + i];
    const float conv = t0 * cur + t1 * s0 + t2 * s1 + t3 * s2;
    out[static_cast<std::size_t>(t) * dim + i] = (conv + cur) * norm;
    s2 = s1;
    s1 = s0;
    s0 = cur;
  }
  state[i * 3] = s0;
  state[i * 3 + 1] = s1;
  state[i * 3 + 2] = s2;
}

// Dynamic short-conv: one thread per dim element (same ownership rule as the
// static form, so the in-place x == out aliasing is race-free).
// tap_eff[head, j] = taps[j][dim] + delta[t, head, j].
__global__ void DynamicSconvKernel(const float* x, float* state,
                                   const float* taps, const float* delta,
                                   float* out, std::uint32_t n_tokens,
                                   std::uint32_t dim, std::uint32_t dim_off,
                                   std::uint32_t head_dim, float norm) {
  const std::uint32_t i = dim_off + threadIdx.x;
  if (i >= dim) return;
  const std::uint32_t head = i / head_dim;
  float s0 = state[i * 3];
  float s1 = state[i * 3 + 1];
  float s2 = state[i * 3 + 2];
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const float* dl =
        delta + (static_cast<std::size_t>(t) * (dim / head_dim) + head) * 4;
    const float cur = x[static_cast<std::size_t>(t) * dim + i];
    // GGUF dynamic taps are oldest-first in memory (same file convention as
    // the static sconv taps): taps[i*4+3] weights the current token.
    // Forensics (least-squares vs transformers): effective tap[lag] must
    // equal torch static_weight[lag]; delta dl[lag] already aligns.
    const float* tp = taps + static_cast<std::size_t>(i) * 4;
    const float conv = (tp[3] + dl[0]) * cur + (tp[2] + dl[1]) * s0 +
                       (tp[1] + dl[2]) * s1 + (tp[0] + dl[3]) * s2;
    out[static_cast<std::size_t>(t) * dim + i] = (conv + cur) * norm;
    s2 = s1;
    s1 = s0;
    s0 = cur;
  }
  state[i * 3] = s0;
  state[i * 3 + 1] = s1;
  state[i * 3 + 2] = s2;
}

__global__ void QScaleClipKernel(float* x, const float* scale,
                                 std::uint32_t n_heads,
                                 std::uint32_t head_dim, float clip) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  float* row =
      x + (static_cast<std::size_t>(tok) * n_heads + head) * head_dim;
  const float* sc = scale + static_cast<std::size_t>(head) * head_dim;
  for (std::uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    row[d] *= sc[d];
  }
  __syncthreads();
  float sum = 0.0F;
  for (std::uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    sum += row[d] * row[d];
  }
  const float total = BlockHiddenSum(sum);
  if (total > clip * clip) {
    const float f = clip / sqrtf(total);
    for (std::uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      row[d] *= f;
    }
  }
}

__global__ void L2ClipKernel(float* x, std::uint32_t n_heads,
                             std::uint32_t head_dim, float clip) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  float* row =
      x + (static_cast<std::size_t>(tok) * n_heads + head) * head_dim;
  float sum = 0.0F;
  for (std::uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    sum += row[d] * row[d];
  }
  const float total = BlockHiddenSum(sum);
  if (total > clip * clip) {
    const float f = clip / sqrtf(total);
    for (std::uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      row[d] *= f;
    }
  }
}

// rel layout per token: [n_heads x 16]; clip layout: [n_heads x 16]
// (head-major, matching TensorRef [rows=heads, cols=16]).
__global__ void RelClipTauKernel(float* r, const float* clip, const float* tau,
                                 std::uint32_t n_heads, float clip_val) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  float* row = r + (static_cast<std::size_t>(tok) * n_heads + head) * 16;
  const float* cl = clip + static_cast<std::size_t>(head) * 16;
  for (std::uint32_t e = threadIdx.x; e < 16; e += blockDim.x) {
    row[e] *= cl[e];
  }
  __syncthreads();
  float sum = 0.0F;
  for (std::uint32_t e = threadIdx.x; e < 16; e += blockDim.x) {
    sum += row[e] * row[e];
  }
  const float total = BlockHiddenSum(sum);
  float f = tau[tok];
  if (total > clip_val * clip_val) {
    f *= clip_val / sqrtf(total);
  }
  for (std::uint32_t e = threadIdx.x; e < 16; e += blockDim.x) {
    row[e] *= f;
  }
}

// Fused causal attention with per-token sinks and relative-bias gather.
// Mirrors the laguna scalar F16 kernel structure (per-(token, head) block,
// online softmax with warp-folded dot), extended with:
//   * score += <rel[t,h,:], bank[:, dist]> for 0 <= dist < extent (else 0)
//   * online softmax seeded with the sink logit (value 0): m = sink, l = 1.



using InklingV16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using InklingV8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using InklingV8u = __attribute__((__vector_size__(8 * sizeof(unsigned)))) unsigned;

__device__ __forceinline__ InklingV8f InklingWmma(InklingV16h a, InklingV16h b, InklingV8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ InklingV16h InklingLoadFrag(const __half* p) {
  union {
    InklingV16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

__device__ __forceinline__ unsigned InklingPackHalves(float a, float b) {
  return __builtin_bit_cast(unsigned, __floats2half2_rn(a, b));
}

template <std::uint32_t kHeadDim, std::uint32_t kHeadWaves, std::uint32_t kQBlocks>
__global__ __launch_bounds__(32 * kHeadWaves * kQBlocks) void InklingAttnWmmaPrefillKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const float* __restrict__ sinks,
    const float* __restrict__ rel, const float* __restrict__ rel_bank,
    float* __restrict__ out, std::uint32_t start_pos, std::uint32_t n_tokens,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t window,
    std::uint32_t rel_extent, float scale) {
  constexpr std::uint32_t kRows = 16;
  constexpr std::uint32_t kKeys = 64;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  constexpr std::uint32_t kWaves = kHeadWaves * kQBlocks;
  constexpr std::uint32_t kThreads = 32 * kWaves;
  constexpr std::uint32_t kKSteps = kHeadDim / 16;
  constexpr std::uint32_t kDimTiles = kHeadDim / 16;
  constexpr std::uint32_t kKStride = kHeadDim + 8;

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;
  const std::uint32_t head = blockIdx.y * kHeadWaves + (wave % kHeadWaves);
  const std::uint32_t qblk = wave / kHeadWaves;
  const std::uint32_t qpg = heads / kv_heads;
  const std::uint32_t kv_head = (blockIdx.y * kHeadWaves) / qpg;
  const std::uint32_t block_q0 = blockIdx.x * kQBlocks * kRows;
  const std::uint32_t query_start = block_q0 + qblk * kRows;
  const std::size_t kv_width = static_cast<std::size_t>(kv_heads) * kHeadDim;

  __shared__ __half k_lds[kKeys][kKStride];
  __shared__ __half vt_lds[kHeadDim][kVtStride];

  const std::uint32_t qrel = query_start + sub;
  const bool row_live = qrel < n_tokens;
  const std::uint32_t qpos = start_pos + qrel;

  InklingV16h q_frag[kKSteps];
  {
    const float* qrow = q + (static_cast<std::size_t>(row_live ? qrel : 0) * heads + head) * kHeadDim;
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKSteps; ++ks) {
#pragma unroll
      for (std::uint32_t e = 0; e < 16; ++e) {
        q_frag[ks][e] = static_cast<_Float16>(row_live ? qrow[ks * 16u + e] * scale : 0.0F);
      }
    }
  }

  float rel_reg[16];
  const bool has_rel = (rel != nullptr && rel_bank != nullptr && row_live);
  if (has_rel) {
    const float* rrow = rel + (static_cast<std::size_t>(qrel) * heads + head) * 16u;
#pragma unroll
    for (int e = 0; e < 16; ++e) rel_reg[e] = rrow[e];
  }

  InklingV8f o_acc[kDimTiles] = {};
  float m_run = (sinks != nullptr && row_live) ? sinks[static_cast<std::size_t>(qrel) * heads + head] : -INFINITY;
  float l_run = (sinks != nullptr && row_live && half_id == 0) ? 1.0F : 0.0F;

  const std::uint32_t n_keys_all = start_pos + n_tokens;
  std::uint32_t key_hi = start_pos + block_q0 + kQBlocks * kRows;
  if (key_hi > n_keys_all) key_hi = n_keys_all;
  const std::uint32_t first_q_pos = start_pos + block_q0;
  const std::uint32_t key_lo = (window != 0 && first_q_pos + 1 > window) ? first_q_pos + 1 - window : 0;
  const std::uint32_t kb_begin = key_lo / kKeys;
  const std::uint32_t kb_end = (key_hi + kKeys - 1u) / kKeys;

  for (std::uint32_t kb = kb_begin; kb < kb_end; ++kb) {
    const std::uint32_t key_start = kb * kKeys;
    __syncthreads();
    for (std::uint32_t i = tid; i < kKeys * (kHeadDim / 8); i += kThreads) {
      const std::uint32_t r = i / (kHeadDim / 8);
      const std::uint32_t c8 = (i % (kHeadDim / 8)) * 8u;
      const std::uint32_t kr = key_start + r;
      uint4 kv = make_uint4(0, 0, 0, 0);
      if (kr < key_hi)
        kv = *reinterpret_cast<const uint4*>(k_cache + static_cast<std::size_t>(kr) * kv_width +
                                             static_cast<std::size_t>(kv_head) * kHeadDim + c8);
      *reinterpret_cast<uint4*>(&k_lds[r][c8]) = kv;
    }
    for (std::uint32_t u = tid; u < (kKeys / 8) * (kHeadDim / 8); u += kThreads) {
      const std::uint32_t kg = u % (kKeys / 8);
      const std::uint32_t c8 = (u / (kKeys / 8)) * 8u;
      unsigned w[8][4];
#pragma unroll
      for (std::uint32_t j = 0; j < 8; ++j) {
        const std::uint32_t kr = key_start + kg * 8u + j;
        uint4 vv = make_uint4(0, 0, 0, 0);
        if (kr < key_hi)
          vv = *reinterpret_cast<const uint4*>(v_cache + static_cast<std::size_t>(kr) * kv_width +
                                               static_cast<std::size_t>(kv_head) * kHeadDim + c8);
        w[j][0] = vv.x; w[j][1] = vv.y; w[j][2] = vv.z; w[j][3] = vv.w;
      }
      const std::uint32_t col = (kg >> 1) * 16u + (kg & 1u) * 4u;
#pragma unroll
      for (std::uint32_t e = 0; e < 8; ++e) {
        const std::uint32_t sh = (e & 1u) * 16u;
        const std::uint32_t wi = e >> 1;
        auto H = [&](std::uint32_t j) { return (w[j][wi] >> sh) & 0xffffu; };
        const uint2 ev = make_uint2(H(0) | (H(2) << 16), H(4) | (H(6) << 16));
        const uint2 od = make_uint2(H(1) | (H(3) << 16), H(5) | (H(7) << 16));
        *reinterpret_cast<uint2*>(&vt_lds[c8 + e][col]) = ev;
        *reinterpret_cast<uint2*>(&vt_lds[c8 + e][col + 8u]) = od;
      }
    }
    __syncthreads();

    InklingV8f s_acc[kKeyBlocks] = {};
#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKSteps; ++ks) {
        const InklingV16h kf = InklingLoadFrag(&k_lds[kbk * 16u + sub][ks * 16u]);
        s_acc[kbk] = InklingWmma(kf, q_frag[ks], s_acc[kbk]);
      }
    }

    float part_max = -INFINITY;
#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t kc = key_start + kbk * 16u + 2u * i + half_id;
        const bool valid = row_live && kc < key_hi && kc <= qpos && (window == 0 || kc + window > qpos);
        float sc = s_acc[kbk][i];
        if (valid && has_rel) {
          const std::uint32_t dist = qpos - kc;
          if (dist < rel_extent) {
            float rb = 0.0F;
#pragma unroll
            for (int e = 0; e < 16; ++e) {
              rb += rel_reg[e] * rel_bank[static_cast<std::size_t>(dist) * 16U + e];
            }
            sc += rb;
          }
        }
        sc = valid ? sc : -INFINITY;
        s_acc[kbk][i] = sc;
        part_max = fmaxf(part_max, sc);
      }
    }
    part_max = fmaxf(part_max, __shfl_xor(part_max, 16));
    const float m_new = fmaxf(m_run, part_max);
    const float alpha = isfinite(m_run) ? __expf(m_run - m_new) : 0.0F;
    m_run = m_new;
    float psum = 0.0F;
    unsigned own[kKeyBlocks][4];
#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
      float pw[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        pw[i] = isfinite(s_acc[kbk][i]) ? __expf(s_acc[kbk][i] - m_new) : 0.0F;
        psum += pw[i];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < 4; ++t) own[kbk][t] = InklingPackHalves(pw[2 * t], pw[2 * t + 1]);
    }
    l_run = l_run * alpha + psum;
#pragma unroll
    for (std::uint32_t dt = 0; dt < kDimTiles; ++dt) o_acc[dt] *= alpha;

#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
      InklingV8u bu;
#pragma unroll
      for (std::uint32_t t = 0; t < 4; ++t) {
        const unsigned other = static_cast<unsigned>(__shfl_xor(static_cast<int>(own[kbk][t]), 16));
        bu[t] = half_id == 0 ? own[kbk][t] : other;
        bu[4 + t] = half_id == 0 ? other : own[kbk][t];
      }
      const InklingV16h pb = __builtin_bit_cast(InklingV16h, bu);
#pragma unroll
      for (std::uint32_t dt = 0; dt < kDimTiles; ++dt) {
        const InklingV16h vf = InklingLoadFrag(&vt_lds[dt * 16u + sub][kbk * 16u]);
        o_acc[dt] = InklingWmma(vf, pb, o_acc[dt]);
      }
    }
  }

  const float l_tot = l_run + __shfl_xor(l_run, 16);
  const float inv = l_tot > 0.0F ? 1.0F / l_tot : 0.0F;
  if (row_live) {
    float* orow = out + (static_cast<std::size_t>(qrel) * heads + head) * kHeadDim;
#pragma unroll
    for (std::uint32_t dt = 0; dt < kDimTiles; ++dt)
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) orow[dt * 16u + 2u * i + half_id] = o_acc[dt][i] * inv;
  }
}

__global__ void __launch_bounds__(384) InklingAttnPrefillTiledKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const float* __restrict__ sinks,
    const float* __restrict__ rel, const float* __restrict__ rel_bank,
    float* __restrict__ out, std::uint32_t start_pos, std::uint32_t n_tokens,
    std::uint32_t window, std::uint32_t n_q_heads, std::uint32_t n_kv_heads,
    std::uint32_t rel_extent, float scale) {
  constexpr std::uint32_t kBKV = 32U;
  __shared__ uint2 s_k[kBKV][32];
  __shared__ uint2 s_v[kBKV][32];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t warp = tid >> 5;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wt = warp / 3U;  // 0..3
  const std::uint32_t wq = warp % 3U;  // 0..2
  const std::uint32_t tok = blockIdx.x * 4U + wt;
  const std::uint32_t kvh = blockIdx.y;
  const std::uint32_t qh = kvh * 3U + wq;

  const std::uint32_t tok_first = blockIdx.x * 4U;
  const std::uint32_t tok_last = min(tok_first + 3U, n_tokens - 1U);
  const std::uint32_t pos_first = start_pos + tok_first;
  const std::uint32_t pos_last = start_pos + tok_last;
  const std::uint32_t lo_blk =
      (window > 0 && pos_first + 1U > window) ? (pos_first + 1U - window) : 0U;

  const bool active = (tok < n_tokens);
  const std::uint32_t cur_pos = start_pos + tok;
  const std::uint32_t lo =
      (window > 0 && cur_pos + 1U > window) ? (cur_pos + 1U - window) : 0U;

  float4 q4 = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  float rv = 0.0F;
  float m = -1e30F;
  float l = 0.0F;
  if (active) {
    const float* q_vec =
        q + (static_cast<std::size_t>(tok) * n_q_heads + qh) * 128U;
    q4 = *reinterpret_cast<const float4*>(q_vec + lane * 4U);
    if (rel != nullptr && lane < 16U) {
      rv = rel[(static_cast<std::size_t>(tok) * n_q_heads + qh) * 16U + lane];
    }
    if (sinks != nullptr) {
      m = sinks[static_cast<std::size_t>(tok) * n_q_heads + qh];
      l = 1.0F;
    }
  }
  float4 acc = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  const std::size_t kv_stride = static_cast<std::size_t>(n_kv_heads) * 128U;
  const std::size_t kv_head_off = static_cast<std::size_t>(kvh) * 128U;

  for (std::uint32_t p_base = lo_blk; p_base <= pos_last; p_base += kBKV) {
    const std::uint32_t tile_n = min(kBKV, pos_last - p_base + 1U);
    // 384 threads load up to 16 * 32 = 512 uint2s for K and V.
    for (std::uint32_t idx = tid; idx < tile_n * 32U; idx += 384U) {
      const std::uint32_t j = idx >> 5;
      const std::uint32_t ln = idx & 31U;
      const std::size_t off =
          static_cast<std::size_t>(p_base + j) * kv_stride + kv_head_off +
          ln * 4U;
      s_k[j][ln] = *reinterpret_cast<const uint2*>(k_cache + off);
      s_v[j][ln] = *reinterpret_cast<const uint2*>(v_cache + off);
    }
    __syncthreads();

    if (active) {
      for (std::uint32_t j = 0; j < tile_n; ++j) {
        const std::uint32_t p = p_base + j;
        if (p < lo || p > cur_pos) continue;
        const uint2 k_raw = s_k[j][lane];
        const __half2* kh = reinterpret_cast<const __half2*>(&k_raw);
        const float2 k01 = __half22float2(kh[0]);
        const float2 k23 = __half22float2(kh[1]);
        float dot = q4.x * k01.x + q4.y * k01.y + q4.z * k23.x + q4.w * k23.y;
        for (int off = 16; off > 0; off /= 2) dot += __shfl_xor(dot, off);
        float score = dot * scale;
        if (rel != nullptr && rel_bank != nullptr) {
          const std::uint32_t dist = cur_pos - p;
          if (dist < rel_extent) {
            float rb = (lane < 16U)
                           ? (rv * rel_bank[static_cast<std::size_t>(dist) * 16U + lane])
                           : 0.0F;
            for (int off = 8; off > 0; off /= 2) rb += __shfl_xor(rb, off);
            score += __shfl(rb, 0);
          }
        }
        const float m_new = fmaxf(m, score);
        const float alpha = __expf(m - m_new);
        const float w = __expf(score - m_new);
        const uint2 v_raw = s_v[j][lane];
        const __half2* vh = reinterpret_cast<const __half2*>(&v_raw);
        const float2 v01 = __half22float2(vh[0]);
        const float2 v23 = __half22float2(vh[1]);
        acc.x = acc.x * alpha + w * v01.x;
        acc.y = acc.y * alpha + w * v01.y;
        acc.z = acc.z * alpha + w * v23.x;
        acc.w = acc.w * alpha + w * v23.y;
        l = l * alpha + w;
        m = m_new;
      }
    }
    __syncthreads();
  }

  if (active) {
    const float inv_l = (l > 0.0F) ? (1.0F / l) : 0.0F;
    float4 out4 = make_float4(acc.x * inv_l, acc.y * inv_l, acc.z * inv_l,
                              acc.w * inv_l);
    *reinterpret_cast<float4*>(
        out + (static_cast<std::size_t>(tok) * n_q_heads + qh) * 128U +
        lane * 4U) = out4;
  }
}

__global__ void __launch_bounds__(256) InklingAttnWave32Kernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const float* __restrict__ sinks,
    const float* __restrict__ rel, const float* __restrict__ rel_bank,
    float* __restrict__ out, std::uint32_t start_pos,
    std::uint32_t min_pos_base, std::uint32_t window, std::uint32_t n_q_heads,
    std::uint32_t n_kv_heads, std::uint32_t rel_extent, float scale) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t qh = blockIdx.y;
  const std::uint32_t wave = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t kv_group = n_q_heads / n_kv_heads;
  const std::uint32_t kvh = qh / kv_group;
  const std::uint32_t cur_pos = start_pos + tok;
  const std::uint32_t min_pos =
      (window > 0 && cur_pos + 1 > window) ? (cur_pos + 1 - window) : 0U;
  const std::uint32_t lo = min_pos > min_pos_base ? min_pos : min_pos_base;

  const float* q_vec =
      q + (static_cast<std::size_t>(tok) * n_q_heads + qh) * 128U;
  const float4 q4 = *reinterpret_cast<const float4*>(q_vec + lane * 4U);

  const float* rel_vec =
      (rel != nullptr)
          ? rel + (static_cast<std::size_t>(tok) * n_q_heads + qh) * 16U
          : nullptr;
  const float rv = (rel_vec != nullptr && lane < 16U) ? rel_vec[lane] : 0.0F;
  const std::size_t kv_stride = static_cast<std::size_t>(n_kv_heads) * 128U;
  const std::size_t kv_head_off = static_cast<std::size_t>(kvh) * 128U + lane * 4U;

  float m = (wave == 0U && sinks != nullptr)
                ? sinks[static_cast<std::size_t>(tok) * n_q_heads + qh]
                : -1e30F;
  float l = (wave == 0U && sinks != nullptr) ? 1.0F : 0.0F;
  float4 acc = make_float4(0.0F, 0.0F, 0.0F, 0.0F);

  for (std::uint32_t p = lo + wave; p <= cur_pos; p += 8U) {
    const std::size_t kv_off =
        static_cast<std::size_t>(p) * kv_stride + kv_head_off;
    const uint2 k_raw = *reinterpret_cast<const uint2*>(k_cache + kv_off);
    const __half2* kh = reinterpret_cast<const __half2*>(&k_raw);
    const float2 k01 = __half22float2(kh[0]);
    const float2 k23 = __half22float2(kh[1]);
    float dot = q4.x * k01.x + q4.y * k01.y + q4.z * k23.x + q4.w * k23.y;
    for (int off = 16; off > 0; off /= 2) {
      dot += __shfl_xor(dot, off);
    }
    float score = dot * scale;
    if (rel_vec != nullptr && rel_bank != nullptr) {
      const std::uint32_t dist = cur_pos - p;
      if (dist < rel_extent) {
        float rb = (lane < 16U) ? (rv * rel_bank[static_cast<std::size_t>(dist) * 16U + lane]) : 0.0F;
        for (int off = 8; off > 0; off /= 2) {
          rb += __shfl_xor(rb, off);
        }
        score += __shfl(rb, 0);
      }
    }
    const float m_new = fmaxf(m, score);
    const float alpha = __expf(m - m_new);
    const float w = __expf(score - m_new);
    const uint2 v_raw = *reinterpret_cast<const uint2*>(v_cache + kv_off);
    const __half2* vh = reinterpret_cast<const __half2*>(&v_raw);
    const float2 v01 = __half22float2(vh[0]);
    const float2 v23 = __half22float2(vh[1]);
    acc.x = acc.x * alpha + w * v01.x;
    acc.y = acc.y * alpha + w * v01.y;
    acc.z = acc.z * alpha + w * v23.x;
    acc.w = acc.w * alpha + w * v23.y;
    l = l * alpha + w;
    m = m_new;
  }

  __shared__ float smem_m[8];
  __shared__ float smem_l[8];
  __shared__ float smem_acc[8][128];
  if (lane == 0U) {
    smem_m[wave] = m;
    smem_l[wave] = l;
  }
  *reinterpret_cast<float4*>(&smem_acc[wave][lane * 4U]) = acc;
  __syncthreads();

  if (wave == 0U) {
    float m_all = -1e30F;
    for (int w = 0; w < 8; ++w) {
      if (smem_l[w] > 0.0F) m_all = fmaxf(m_all, smem_m[w]);
    }
    float l_tot = 0.0F;
    float4 acc_tot = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    for (int w = 0; w < 8; ++w) {
      const float lw = smem_l[w];
      if (lw > 0.0F) {
        const float sw = __expf(smem_m[w] - m_all);
        l_tot += sw * lw;
        const float4 aw = *reinterpret_cast<const float4*>(&smem_acc[w][lane * 4U]);
        acc_tot.x += sw * aw.x;
        acc_tot.y += sw * aw.y;
        acc_tot.z += sw * aw.z;
        acc_tot.w += sw * aw.w;
      }
    }
    const float inv_l = (l_tot > 0.0F) ? (1.0F / l_tot) : 0.0F;
    float4 out4 = make_float4(acc_tot.x * inv_l, acc_tot.y * inv_l,
                              acc_tot.z * inv_l, acc_tot.w * inv_l);
    *reinterpret_cast<float4*>(
        out + (static_cast<std::size_t>(tok) * n_q_heads + qh) * 128U + lane * 4U) = out4;
  }
}


__global__ void __launch_bounds__(256) InklingAttnSplitKKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const float* __restrict__ sinks,
    const float* __restrict__ rel, const float* __restrict__ rel_bank,
    float* __restrict__ split_buf, std::uint32_t start_pos,
    const std::uint32_t* __restrict__ d_pos, std::uint32_t window,
    std::uint32_t n_q_heads, std::uint32_t n_kv_heads,
    std::uint32_t rel_extent, float scale, std::uint32_t n_splits) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t qh = blockIdx.y;
  const std::uint32_t sp = blockIdx.z;
  const std::uint32_t wave = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t kv_group = n_q_heads / n_kv_heads;
  const std::uint32_t kvh = qh / kv_group;
  const std::uint32_t cur_pos = (d_pos != nullptr ? *d_pos : start_pos) + tok;
  const std::uint32_t lo =
      (window > 0U && cur_pos + 1U > window) ? (cur_pos + 1U - window) : 0U;
  const std::uint32_t total_kv = (cur_pos >= lo) ? (cur_pos - lo + 1U) : 0U;
  const std::uint32_t chunk = (total_kv + n_splits - 1U) / n_splits;
  const std::uint32_t p_beg = lo + sp * chunk;
  const std::uint32_t p_end =
      (p_beg < cur_pos + 1U)
          ? ((p_beg + chunk < cur_pos + 1U) ? (p_beg + chunk) : (cur_pos + 1U))
          : p_beg;

  const float* q_vec =
      q + (static_cast<std::size_t>(tok) * n_q_heads + qh) * 128U;
  const float4 q4 = *reinterpret_cast<const float4*>(q_vec + lane * 4U);

  const float* rel_vec =
      (rel != nullptr)
          ? rel + (static_cast<std::size_t>(tok) * n_q_heads + qh) * 16U
          : nullptr;
  const float rv = (rel_vec != nullptr && lane < 16U) ? rel_vec[lane] : 0.0F;
  const std::size_t kv_stride = static_cast<std::size_t>(n_kv_heads) * 128U;
  const std::size_t kv_head_off = static_cast<std::size_t>(kvh) * 128U + lane * 4U;

  float m = (sp == 0U && wave == 0U && sinks != nullptr)
                ? sinks[static_cast<std::size_t>(tok) * n_q_heads + qh]
                : -1e30F;
  float l = (sp == 0U && wave == 0U && sinks != nullptr) ? 1.0F : 0.0F;
  float4 acc = make_float4(0.0F, 0.0F, 0.0F, 0.0F);

  for (std::uint32_t p = p_beg + wave; p < p_end; p += 8U) {
    const std::size_t kv_off =
        static_cast<std::size_t>(p) * kv_stride + kv_head_off;
    const uint2 k_raw = *reinterpret_cast<const uint2*>(k_cache + kv_off);
    const __half2* kh = reinterpret_cast<const __half2*>(&k_raw);
    const float2 k01 = __half22float2(kh[0]);
    const float2 k23 = __half22float2(kh[1]);
    float dot = q4.x * k01.x + q4.y * k01.y + q4.z * k23.x + q4.w * k23.y;
    for (int off = 16; off > 0; off /= 2) dot += __shfl_xor(dot, off);
    float score = dot * scale;
    if (rel_vec != nullptr && rel_bank != nullptr) {
      const std::uint32_t dist = cur_pos - p;
      if (dist < rel_extent) {
        float rb = (lane < 16U) ? (rv * rel_bank[static_cast<std::size_t>(dist) * 16U + lane]) : 0.0F;
        for (int off = 8; off > 0; off /= 2) rb += __shfl_xor(rb, off);
        score += __shfl(rb, 0);
      }
    }
    const float m_new = fmaxf(m, score);
    const float alpha = __expf(m - m_new);
    const float w = __expf(score - m_new);
    const uint2 v_raw = *reinterpret_cast<const uint2*>(v_cache + kv_off);
    const __half2* vh = reinterpret_cast<const __half2*>(&v_raw);
    const float2 v01 = __half22float2(vh[0]);
    const float2 v23 = __half22float2(vh[1]);
    acc.x = acc.x * alpha + w * v01.x;
    acc.y = acc.y * alpha + w * v01.y;
    acc.z = acc.z * alpha + w * v23.x;
    acc.w = acc.w * alpha + w * v23.y;
    l = l * alpha + w;
    m = m_new;
  }

  __shared__ float smem_m[8];
  __shared__ float smem_l[8];
  __shared__ float smem_acc[8][128];
  if (lane == 0U) {
    smem_m[wave] = m;
    smem_l[wave] = l;
  }
  *reinterpret_cast<float4*>(&smem_acc[wave][lane * 4U]) = acc;
  __syncthreads();

  if (wave == 0U) {
    float m_all = -1e30F;
    for (int w = 0; w < 8; ++w) {
      if (smem_l[w] > 0.0F) m_all = fmaxf(m_all, smem_m[w]);
    }
    float l_tot = 0.0F;
    float4 acc_tot = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    for (int w = 0; w < 8; ++w) {
      const float lw = smem_l[w];
      if (lw > 0.0F) {
        const float sw = __expf(smem_m[w] - m_all);
        l_tot += sw * lw;
        const float4 aw = *reinterpret_cast<const float4*>(&smem_acc[w][lane * 4U]);
        acc_tot.x += sw * aw.x;
        acc_tot.y += sw * aw.y;
        acc_tot.z += sw * aw.z;
        acc_tot.w += sw * aw.w;
      }
    }
    float* dst = split_buf + ((static_cast<std::size_t>(tok) * n_q_heads + qh) * n_splits + sp) * 130U;
    if (lane == 0U) {
      dst[0] = m_all;
      dst[1] = l_tot;
    }
    *reinterpret_cast<float4*>(dst + 2U + lane * 4U) = acc_tot;
  }
}

__global__ void __launch_bounds__(32) InklingAttnSplitKMergeKernel(
    const float* __restrict__ split_buf, float* __restrict__ out,
    std::uint32_t n_q_heads, std::uint32_t n_splits) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t qh = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  float m_all = -1e30F;
  float l_tot = 0.0F;
  float4 acc_tot = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  for (std::uint32_t sp = 0; sp < n_splits; ++sp) {
    const float* src = split_buf + ((static_cast<std::size_t>(tok) * n_q_heads + qh) * n_splits + sp) * 130U;
    const float ms = src[0];
    const float ls = src[1];
    if (ls <= 0.0F) continue;
    const float4 as = *reinterpret_cast<const float4*>(src + 2U + lane * 4U);
    if (l_tot <= 0.0F) {
      m_all = ms;
      l_tot = ls;
      acc_tot = as;
    } else {
      const float m_new = fmaxf(m_all, ms);
      const float a_old = __expf(m_all - m_new);
      const float a_new = __expf(ms - m_new);
      l_tot = l_tot * a_old + ls * a_new;
      acc_tot.x = acc_tot.x * a_old + as.x * a_new;
      acc_tot.y = acc_tot.y * a_old + as.y * a_new;
      acc_tot.z = acc_tot.z * a_old + as.z * a_new;
      acc_tot.w = acc_tot.w * a_old + as.w * a_new;
      m_all = m_new;
    }
  }
  const float inv_l = (l_tot > 0.0F) ? (1.0F / l_tot) : 0.0F;
  *reinterpret_cast<float4*>(
      out + (static_cast<std::size_t>(tok) * n_q_heads + qh) * 128U + lane * 4U) =
      make_float4(acc_tot.x * inv_l, acc_tot.y * inv_l,
                acc_tot.z * inv_l, acc_tot.w * inv_l);
}

__global__ void InklingAttnKernel(
    const float* q, const __half* k_cache, const __half* v_cache,
    const float* sinks, const float* rel, const float* rel_bank, float* out,
    std::uint32_t start_pos, std::uint32_t n_tokens,
    std::uint32_t min_pos_base, std::uint32_t window, std::uint32_t n_q_heads,
    std::uint32_t n_kv_heads, std::uint32_t head_dim,
    std::uint32_t rel_extent, float scale) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t qh = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  if (d >= head_dim) return;

  const std::uint32_t kv_group = n_q_heads / n_kv_heads;
  const std::uint32_t kvh = qh / kv_group;
  const std::uint32_t cur_pos = start_pos + tok;
  const std::uint32_t min_pos =
      (window > 0 && cur_pos + 1 > window) ? (cur_pos + 1 - window) : 0U;
  const std::uint32_t lo = min_pos > min_pos_base ? min_pos : min_pos_base;

  const float* q_vec =
      q + (static_cast<std::size_t>(tok) * n_q_heads + qh) * head_dim;
  const float q_val = q_vec[d];
  const float* rel_vec =
      (rel != nullptr)
          ? rel + (static_cast<std::size_t>(tok) * n_q_heads + qh) * 16
          : nullptr;
  const std::size_t kv_stride = static_cast<std::size_t>(n_kv_heads) * head_dim;

  float m = (sinks != nullptr)
                ? sinks[static_cast<std::size_t>(tok) * n_q_heads + qh]
                : -1e30F;
  float l = (sinks != nullptr) ? 1.0F : 0.0F;
  float acc = 0.0F;

  for (std::uint32_t p = lo; p <= cur_pos; ++p) {
    const std::size_t kv_off =
        static_cast<std::size_t>(p) * kv_stride +
        static_cast<std::size_t>(kvh) * head_dim;
    float dot = q_val * __half2float(k_cache[kv_off + d]);
    __shared__ float s_dot[4];
    for (int off = 16; off > 0; off /= 2) dot += __shfl_down(dot, off);
    const int lane = d & 31;
    const int warp = d >> 5;
    if (lane == 0) s_dot[warp] = dot;
    __syncthreads();
    if (warp == 0) {
      float t = (lane < 4) ? s_dot[lane] : 0.0F;
      for (int off = 16; off > 0; off /= 2) t += __shfl_down(t, off);
      if (lane == 0) s_dot[0] = t * scale;
    }
    __syncthreads();
    float score = s_dot[0];
    __syncthreads();
    if (rel_vec != nullptr && rel_bank != nullptr) {
      const std::uint32_t dist = cur_pos - p;
      if (dist < rel_extent) {
        const float* bank_row = rel_bank + static_cast<std::size_t>(dist) * 16U;
        float rb = 0.0F;
        for (std::uint32_t e = 0; e < 16; ++e) {
          rb += rel_vec[e] * bank_row[e];
        }
        score += rb;
      }
    }
    __syncthreads();
    const float m_new = fmaxf(m, score);
    const float alpha = __expf(m - m_new);
    const float w = __expf(score - m_new);
    acc = acc * alpha + w * __half2float(v_cache[kv_off + d]);
    l = l * alpha + w;
    m = m_new;
  }

  out[(static_cast<std::size_t>(tok) * n_q_heads + qh) * head_dim + d] =
      (l > 0.0F) ? (acc / l) : 0.0F;
  (void)n_tokens;
}

__global__ void SigmoidGateMulKernel(float* attn_out, const float* gate,
                                     std::uint32_t n_heads,
                                     std::uint32_t head_dim) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  if (d >= head_dim) return;
  // Elementwise gate (matches GatedOutputProjection: attn * sigmoid(gate)
  // over the full q_dim, not one scalar per head).
  const std::size_t idx =
      (static_cast<std::size_t>(tok) * n_heads + head) * head_dim + d;
  attn_out[idx] *= SigmoidF(gate[idx]);
}

__global__ void SigmoidRouterTop8Kernel(const float* logits, const float* bias,
                                        std::int32_t* ids, float* weights,
                                        std::uint32_t n_experts,
                                        std::uint32_t top_k, float route_scale,
                                        float eps) {
  const std::uint32_t tok = blockIdx.x;
  if (threadIdx.x != 0) return;

  const float* row = logits + static_cast<std::size_t>(tok) * n_experts;
  std::int32_t* out_ids = ids + static_cast<std::size_t>(tok) * top_k;
  float* out_w = weights + static_cast<std::size_t>(tok) * top_k;

  constexpr int kMaxK = 8;
  float best_sel[kMaxK];
  float best_prob[kMaxK];
  int best_idx[kMaxK];
  for (std::uint32_t k = 0; k < top_k; ++k) {
    best_sel[k] = -1e30F;
    best_prob[k] = 0.0F;
    best_idx[k] = -1;
  }

  for (std::uint32_t e = 0; e < n_experts; ++e) {
    const float p = SigmoidF(row[e]);
    const float b = (bias != nullptr) ? bias[e] : 0.0F;
    const float sel = p + b;
    if (sel > best_sel[top_k - 1]) {
      best_sel[top_k - 1] = sel;
      best_prob[top_k - 1] = p;
      best_idx[top_k - 1] = static_cast<int>(e);
      for (int j = static_cast<int>(top_k) - 1; j > 0; --j) {
        if (best_sel[j] > best_sel[j - 1]) {
          float ts = best_sel[j - 1];
          best_sel[j - 1] = best_sel[j];
          best_sel[j] = ts;
          float tp = best_prob[j - 1];
          best_prob[j - 1] = best_prob[j];
          best_prob[j] = tp;
          int ti = best_idx[j - 1];
          best_idx[j - 1] = best_idx[j];
          best_idx[j] = ti;
        } else {
          break;
        }
      }
    }
  }

  float sum = 0.0F;
  for (std::uint32_t k = 0; k < top_k; ++k) sum += best_prob[k];
  const float inv = 1.0F / (sum + eps);
  for (std::uint32_t k = 0; k < top_k; ++k) {
    out_ids[k] = best_idx[k];
    out_w[k] = best_prob[k] * inv * route_scale;
  }
}

__global__ void SituFuseHalfToF16Kernel(const __half* __restrict__ gateup,
                                        __half* __restrict__ act16,
                                        std::uint32_t ff) {
  const std::size_t row = blockIdx.x;
  const __half* gu = gateup + row * (2ULL * ff);
  __half* dst = act16 + row * ff;
  if ((ff & 7U) == 0U) {
    const std::uint32_t n8 = ff >> 3;
    const uint4* g8 = reinterpret_cast<const uint4*>(gu);
    const uint4* u8 = reinterpret_cast<const uint4*>(gu + ff);
    uint4* d8 = reinterpret_cast<uint4*>(dst);
    for (std::uint32_t idx = threadIdx.x; idx < n8; idx += blockDim.x) {
      const uint4 gv = g8[idx];
      const uint4 uv = u8[idx];
      const __half2* gh = reinterpret_cast<const __half2*>(&gv);
      const __half2* uh = reinterpret_cast<const __half2*>(&uv);
      uint4 outv;
      __half2* oh = reinterpret_cast<__half2*>(&outv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 gf = __half22float2(gh[j]);
        const float2 uf = __half22float2(uh[j]);
        const float r0 = (4.0F * tanhf(gf.x * 0.25F)) * SigmoidF(gf.x) * (25.0F * tanhf(uf.x * 0.04F));
        const float r1 = (4.0F * tanhf(gf.y * 0.25F)) * SigmoidF(gf.y) * (25.0F * tanhf(uf.y * 0.04F));
        oh[j] = __floats2half2_rn(r0, r1);
      }
      d8[idx] = outv;
    }
    return;
  }
  for (std::uint32_t i = threadIdx.x; i < ff; i += blockDim.x) {
    const float g = __half2float(gu[i]);
    const float u = __half2float(gu[ff + i]);
    const float gs = 4.0F * tanhf(g * 0.25F);
    const float us = 25.0F * tanhf(u * 0.04F);
    dst[i] = __float2half(gs * SigmoidF(g) * us);
  }
}

__global__ void MoeEpilogueHalfKernel(const __half* __restrict__ expert_out,
                                      const float* __restrict__ weights,
                                      float* __restrict__ out, std::uint32_t k,
                                      std::uint32_t dim) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= dim) return;
  const std::size_t base_slot = static_cast<std::size_t>(tok) * k;
  float sum = 0.0F;
  for (std::uint32_t i = 0; i < k; ++i) {
    sum += weights[base_slot + i] *
           __half2float(expert_out[(base_slot + i) * dim + c]);
  }
  out[static_cast<std::size_t>(tok) * dim + c] = sum;
}

__global__ void SituFuseToF16Kernel(const float* __restrict__ gateup,
                                    __half* __restrict__ act16,
                                    std::uint32_t ff) {
  const std::size_t row = blockIdx.x;
  const float* gu = gateup + row * (2ULL * ff);
  __half* dst = act16 + row * ff;
  for (std::uint32_t i = threadIdx.x; i < ff; i += blockDim.x) {
    const float g = gu[i];
    const float u = gu[ff + i];
    const float gs = 4.0F * tanhf(g * 0.25F);
    const float us = 25.0F * tanhf(u * 0.04F);
    dst[i] = __float2half(gs * SigmoidF(g) * us);
  }
}

__global__ void SituFuseKernel(const float* gateup, float* act,
                               std::uint32_t ff) {
  const std::size_t row = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < ff; i += blockDim.x) {
    const float g = gateup[row * (2 * ff) + i];
    const float u = gateup[row * (2 * ff) + ff + i];
    const float gs = 4.0F * tanhf(g * 0.25F);
    const float us = 25.0F * tanhf(u * 0.04F);
    act[row * ff + i] = gs * SigmoidF(g) * us;
  }
}

__global__ void GatherRowsKernel(const float* x, const std::int32_t* list,
                                 float* out, std::uint32_t batch,
                                 std::uint32_t hidden) {
  const std::uint32_t r = blockIdx.x;
  if (r >= batch) return;
  const std::int32_t src = list[r];
  const float* src_row = x + static_cast<std::size_t>(src) * hidden;
  float* dst_row = out + static_cast<std::size_t>(r) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst_row[i] = (src >= 0) ? src_row[i] : 0.0F;
  }
}

__global__ void ScatterAddWeightedKernel(const float* x, const float* weights,
                                          const std::int32_t* list, float* out,
                                          std::uint32_t batch,
                                          std::uint32_t hidden) {
  const std::uint32_t r = blockIdx.x;
  if (r >= batch) return;
  const std::int32_t dst = list[r];
  if (dst < 0) return;
  const float w = weights[r];
  const float* src_row = x + static_cast<std::size_t>(r) * hidden;
  float* dst_row = out + static_cast<std::size_t>(dst) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    // Fixed slot order per expert loop on the host keeps this deterministic.
    dst_row[i] += w * src_row[i];
  }
}

// Multi-query attention over a small state bank (message / residual paths).
// Keys and values are RMS-normalized in passing (matching the reference,
// which attends over normalized states for both).
__global__ void MultiQueryResAttnKernel(const float* states,
                                        std::uint32_t n_states,
                                        std::uint32_t state_stride,
                                        const float* queries,
                                        std::uint32_t n_queries, float* out,
                                        std::uint32_t hidden, float scale,
                                        float eps) {
  const std::uint32_t tok = blockIdx.x;
  const float* tok_states =
      states + static_cast<std::size_t>(tok) * state_stride;
  float* tok_out = out + static_cast<std::size_t>(tok) * hidden;
  __shared__ float s_tmp[16];
  __shared__ float s_w[24 * 16];
  if (n_states > 16) n_states = 16;
  if (n_queries > 24) n_queries = 24;

  // Step 1: per-state RMS norms.
  for (std::uint32_t s = 0; s < n_states; ++s) {
    const float* st = tok_states + static_cast<std::size_t>(s) * hidden;
    float sum = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
      sum += st[i] * st[i];
    }
    s_tmp[s] = rsqrtf(BlockHiddenSum(sum) / static_cast<float>(hidden) + eps);
  }
  __syncthreads();

  // Step 2: scores[q][s] = <query_q, norm(state_s)> * scale.
  for (std::uint32_t q = 0; q < n_queries; ++q) {
    const float* qq = queries + static_cast<std::size_t>(q) * hidden;
    for (std::uint32_t s = 0; s < n_states; ++s) {
      const float* st = tok_states + static_cast<std::size_t>(s) * hidden;
      const float inv = s_tmp[s];
      float dot = 0.0F;
      for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        dot += qq[i] * st[i] * inv;
      }
      s_w[q * 16 + s] = BlockHiddenSum(dot) * scale;
    }
  }
  __syncthreads();

  // Step 3: softmax over states per query (tiny; one thread is enough).
  if (threadIdx.x == 0) {
    for (std::uint32_t q = 0; q < n_queries; ++q) {
      float m = -1e30F;
      for (std::uint32_t s = 0; s < n_states; ++s) {
        const float v = s_w[q * 16 + s];
        if (v > m) m = v;
      }
      float l = 0.0F;
      for (std::uint32_t s = 0; s < n_states; ++s) {
        const float w = expf(s_w[q * 16 + s] - m);
        s_w[q * 16 + s] = w;
        l += w;
      }
      const float inv = (l > 0.0F) ? (1.0F / l) : 0.0F;
      for (std::uint32_t s = 0; s < n_states; ++s) s_w[q * 16 + s] *= inv;
    }
  }
  __syncthreads();

  // Step 4: mean over queries of the weighted *normalized* states
  // (reference BlockAttentionResidual/MessageAttentionResidual attend over
  // RMS-normalized states for both keys and values; s_tmp holds per-state
  // inverse RMS norms from Step 1).
  const float qinv =
      (n_queries > 0) ? (1.0F / static_cast<float>(n_queries)) : 0.0F;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    float acc = 0.0F;
    for (std::uint32_t q = 0; q < n_queries; ++q) {
      float wsum = 0.0F;
      for (std::uint32_t s = 0; s < n_states; ++s) {
        const float* st = tok_states + static_cast<std::size_t>(s) * hidden;
        wsum += s_w[q * 16 + s] * st[i] * s_tmp[s];
      }
      acc += wsum;
    }
    tok_out[i] = acc * qinv;
  }
}

__global__ void __launch_bounds__(256) FuseGateProjApplyWmmaPrefillKernel(
    const float* __restrict__ h, const float* __restrict__ msg,
    const float* __restrict__ g64, const __half* __restrict__ forget_w,
    const __half* __restrict__ gain_w, float* __restrict__ out,
    std::uint32_t n_tokens, std::uint32_t hidden) {
  __shared__ __half s_g[32][72];
  __shared__ __half s_fw[64][72];
  __shared__ __half s_gw[64][72];

  const std::uint32_t tok_base = blockIdx.x * 32U;
  const std::uint32_t h_base = blockIdx.y * 64U;
  const std::uint32_t tid = threadIdx.x;

  // Load g64[32][64] -> s_g (32 * 64 = 2048 elems; 256 threads * 8 = 2048)
  for (std::uint32_t idx = tid; idx < 32U * 16U; idx += 256U) {
    const std::uint32_t tr = idx >> 4;
    const std::uint32_t kc = (idx & 15U) << 2;
    const std::uint32_t tok = tok_base + tr;
    if (tok < n_tokens) {
      const float4 gv = *reinterpret_cast<const float4*>(
          g64 + static_cast<std::size_t>(tok) * 64U + kc);
      s_g[tr][kc + 0] = __float2half(gv.x);
      s_g[tr][kc + 1] = __float2half(gv.y);
      s_g[tr][kc + 2] = __float2half(gv.z);
      s_g[tr][kc + 3] = __float2half(gv.w);
    } else {
      s_g[tr][kc + 0] = __float2half(0.0F);
      s_g[tr][kc + 1] = __float2half(0.0F);
      s_g[tr][kc + 2] = __float2half(0.0F);
      s_g[tr][kc + 3] = __float2half(0.0F);
    }
  }

  // Load forget_w[64][64] and gain_w[64][64] (64 * 8 uint4 = 512 uint4)
  for (std::uint32_t idx = tid; idx < 64U * 8U; idx += 256U) {
    const std::uint32_t hr = idx >> 3;
    const std::uint32_t kc = (idx & 7U) << 3;
    const std::uint32_t hi = h_base + hr;
    if (hi < hidden) {
      *reinterpret_cast<uint4*>(&s_fw[hr][kc]) = *reinterpret_cast<const uint4*>(
          forget_w + static_cast<std::size_t>(hi) * 64U + kc);
      *reinterpret_cast<uint4*>(&s_gw[hr][kc]) = *reinterpret_cast<const uint4*>(
          gain_w + static_cast<std::size_t>(hi) * 64U + kc);
    } else {
      *reinterpret_cast<uint4*>(&s_fw[hr][kc]) = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(&s_gw[hr][kc]) = make_uint4(0, 0, 0, 0);
    }
  }
  __syncthreads();

  const std::uint32_t wave = tid >> 5;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4;
  const std::uint32_t wt = wave & 1U;
  const std::uint32_t wh = wave >> 1;

  InklingV8f fg_acc = {};
  InklingV8f mg_acc = {};
#pragma unroll
  for (std::uint32_t k = 0; k < 4U; ++k) {
    const InklingV16h g_frag = InklingLoadFrag(&s_g[wt * 16U + sub][k * 16U]);
    const InklingV16h fw_frag = InklingLoadFrag(&s_fw[wh * 16U + sub][k * 16U]);
    const InklingV16h gw_frag = InklingLoadFrag(&s_gw[wh * 16U + sub][k * 16U]);
    fg_acc = InklingWmma(g_frag, fw_frag, fg_acc);
    mg_acc = InklingWmma(g_frag, gw_frag, mg_acc);
  }

  const std::uint32_t hi = h_base + wh * 16U + sub;
  if (hi < hidden) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const std::uint32_t tok = tok_base + wt * 16U + static_cast<std::uint32_t>(e) * 2U + half_id;
      if (tok < n_tokens) {
        const std::size_t off = static_cast<std::size_t>(tok) * hidden + hi;
        out[off] = (1.0F - tanhf(fg_acc[e])) * h[off] + tanhf(mg_acc[e]) * msg[off];
      }
    }
  }
}

__global__ void __launch_bounds__(256) FuseGateProjApplyWave64Kernel(
    const float* __restrict__ h, const float* __restrict__ msg,
    const float* __restrict__ g64, const __half* __restrict__ forget_w,
    const __half* __restrict__ gain_w, float* __restrict__ out,
    std::uint32_t hidden) {
  __shared__ float2 sg2[32];
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t wave = tid >> 5;
  const std::uint32_t lane = tid & 31U;
  if (tid < 32U) {
    sg2[tid] = *reinterpret_cast<const float2*>(
        g64 + static_cast<std::size_t>(tok) * 64U + tid * 2U);
  }
  __syncthreads();

  const std::uint32_t i = blockIdx.y * 8U + wave;
  if (i >= hidden) return;

  const float2 g01 = sg2[lane];
  const __half2 f01 = reinterpret_cast<const __half2*>(
      forget_w + static_cast<std::size_t>(i) * 64U)[lane];
  const __half2 m01 = reinterpret_cast<const __half2*>(
      gain_w + static_cast<std::size_t>(i) * 64U)[lane];
  const float2 ff = __half22float2(f01);
  const float2 mm = __half22float2(m01);
  float fg = ff.x * g01.x + ff.y * g01.y;
  float mg = mm.x * g01.x + mm.y * g01.y;
  for (int off = 16; off > 0; off /= 2) {
    fg += __shfl_xor(fg, off);
    mg += __shfl_xor(mg, off);
  }
  if (lane == 0U) {
    const std::size_t off = static_cast<std::size_t>(tok) * hidden + i;
    out[off] = (1.0F - tanhf(fg)) * h[off] + tanhf(mg) * msg[off];
  }
}

__global__ void FuseGateProjApplyKernel(
    const float* __restrict__ h, const float* __restrict__ msg,
    const float* __restrict__ g64, const __half* __restrict__ forget_w,
    const __half* __restrict__ gain_w, float* __restrict__ out,
    std::uint32_t hidden, std::uint32_t rank) {
  __shared__ float sg[64];
  const std::uint32_t tok = blockIdx.x;
  if (threadIdx.x < rank) {
    sg[threadIdx.x] = g64[static_cast<std::size_t>(tok) * rank + threadIdx.x];
  }
  __syncthreads();
  const std::uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= hidden) return;
  const __half* fw = forget_w + static_cast<std::size_t>(i) * rank;
  const __half* gw = gain_w + static_cast<std::size_t>(i) * rank;
  float fg = 0.0F;
  float mg = 0.0F;
  for (std::uint32_t r = 0; r < rank; ++r) {
    const float g = sg[r];
    fg += __half2float(fw[r]) * g;
    mg += __half2float(gw[r]) * g;
  }
  const std::size_t off = static_cast<std::size_t>(tok) * hidden + i;
  out[off] = (1.0F - tanhf(fg)) * h[off] + tanhf(mg) * msg[off];
}


__global__ void MultiQueryResAttn4Kernel(
    const float* __restrict__ states, std::uint32_t n_states,
    std::uint32_t state_stride, const float* __restrict__ q0,
    const float* __restrict__ q1, const float* __restrict__ q2,
    const float* __restrict__ q3, float* __restrict__ out0,
    float* __restrict__ out1, float* __restrict__ out2,
    float* __restrict__ out3, std::uint32_t hidden, float scale) {
  const std::uint32_t tok = blockIdx.x;
  const float* tok_states = states + static_cast<std::size_t>(tok) * state_stride;
  __shared__ float s_inv[16];
  __shared__ float s_w[4][16];

  for (std::uint32_t s = 0; s < n_states; ++s) {
    const float* st = tok_states + static_cast<std::size_t>(s) * hidden;
    float ss = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
      const float v = st[i];
      ss += v * v;
    }
    const float inv = rsqrtf(BlockHiddenSum(ss) / static_cast<float>(hidden) + 1e-6F);
    if (threadIdx.x == 0) s_inv[s] = inv;
    float d0 = 0.0F, d1 = 0.0F, d2 = 0.0F, d3 = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
      const float k = st[i] * inv;
      d0 += q0[i] * k;
      d1 += q1[i] * k;
      d2 += q2[i] * k;
      d3 += q3[i] * k;
    }
    d0 = BlockHiddenSum(d0) * scale;
    d1 = BlockHiddenSum(d1) * scale;
    d2 = BlockHiddenSum(d2) * scale;
    d3 = BlockHiddenSum(d3) * scale;
    if (threadIdx.x == 0) {
      s_w[0][s] = d0;
      s_w[1][s] = d1;
      s_w[2][s] = d2;
      s_w[3][s] = d3;
    }
    __syncthreads();
  }
  if (threadIdx.x < 4U) {
    const std::uint32_t q = threadIdx.x;
    float mx = -1e30F;
    for (std::uint32_t s = 0; s < n_states; ++s) mx = fmaxf(mx, s_w[q][s]);
    float l = 0.0F;
    for (std::uint32_t s = 0; s < n_states; ++s) {
      s_w[q][s] = __expf(s_w[q][s] - mx);
      l += s_w[q][s];
    }
    const float inv_l = (l > 0.0F) ? (1.0F / l) : 0.0F;
    for (std::uint32_t s = 0; s < n_states; ++s) s_w[q][s] *= inv_l;
  }
  __syncthreads();

  const std::size_t out_off = static_cast<std::size_t>(tok) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;
    for (std::uint32_t s = 0; s < n_states; ++s) {
      const float v = tok_states[static_cast<std::size_t>(s) * hidden + i] * s_inv[s];
      a0 += s_w[0][s] * v;
      a1 += s_w[1][s] * v;
      a2 += s_w[2][s] * v;
      a3 += s_w[3][s] * v;
    }
    out0[out_off + i] = a0;
    out1[out_off + i] = a1;
    out2[out_off + i] = a2;
    out3[out_off + i] = a3;
  }
}

__global__ void GateFuseApplyKernel(const float* h, const float* msg,
                                      const float* fg_raw, const float* mg_raw,
                                      float* out, std::uint32_t hidden) {
  const std::uint32_t tok = blockIdx.x;
  const float* hr = h + static_cast<std::size_t>(tok) * hidden;
  const float* mr = msg + static_cast<std::size_t>(tok) * hidden;
  const float* fr = fg_raw + static_cast<std::size_t>(tok) * hidden;
  const float* gr = mg_raw + static_cast<std::size_t>(tok) * hidden;
  float* dst = out + static_cast<std::size_t>(tok) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    // Reference Inkling2LoopMessageGate: BOTH gates are tanh
    // (message_gate = tanh(message_gate_proj(...)), NOT sigmoid).
    const float fg = tanhf(fr[i]);
    const float mg = tanhf(gr[i]);
    dst[i] = (1.0F - fg) * hr[i] + mg * mr[i];
  }
}

__global__ void __launch_bounds__(256) FusedPairNormDownKernel(
    const float* __restrict__ h, const float* __restrict__ msg,
    const __half* __restrict__ down_w, float* __restrict__ out,
    std::uint32_t hs, float eps) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t wave = tid >> 5;
  const std::uint32_t lane = tid & 31U;

  float ss_h = 0.0F;
  float ss_m = 0.0F;
  for (std::uint32_t i = tid; i < hs; i += 256U) {
    const float vh = h[i];
    const float vm = msg[i];
    ss_h += vh * vh;
    ss_m += vm * vm;
  }
  for (int off = 16; off > 0; off /= 2) {
    ss_h += __shfl_xor(ss_h, off);
    ss_m += __shfl_xor(ss_m, off);
  }
  __shared__ float sm_h[8];
  __shared__ float sm_m[8];
  if (lane == 0U) {
    sm_h[wave] = ss_h;
    sm_m[wave] = ss_m;
  }
  __syncthreads();
  if (tid == 0U) {
    float th = 0.0F;
    float tm = 0.0F;
    for (int w = 0; w < 8; ++w) {
      th += sm_h[w];
      tm += sm_m[w];
    }
    sm_h[0] = rsqrtf(th / static_cast<float>(hs) + eps);
    sm_m[0] = rsqrtf(tm / static_cast<float>(hs) + eps);
  }
  __syncthreads();

  const float inv_h = sm_h[0];
  const float inv_m = sm_m[0];
  const std::size_t k = static_cast<std::size_t>(hs) * 2U;
  const __half* w_row = down_w + static_cast<std::size_t>(row) * k;
  const __half* w_row_m = w_row + hs;

  float acc = 0.0F;
  // Vectorized 8-half loads (uint4 = 16 bytes = 8 __half) when hs is multiple of 8
  const std::uint32_t hs8 = hs / 8U;
  const float4* h4 = reinterpret_cast<const float4*>(h);
  const float4* m4 = reinterpret_cast<const float4*>(msg);
  const uint4* wh8 = reinterpret_cast<const uint4*>(w_row);
  const uint4* wm8 = reinterpret_cast<const uint4*>(w_row_m);
  for (std::uint32_t idx = tid; idx < hs8; idx += 256U) {
    const float4 hv0 = h4[idx * 2U];
    const float4 hv1 = h4[idx * 2U + 1U];
    const uint4 wv = wh8[idx];
    const __half2* wh2 = reinterpret_cast<const __half2*>(&wv);
    const float2 w0 = __half22float2(wh2[0]);
    const float2 w1 = __half22float2(wh2[1]);
    const float2 w2 = __half22float2(wh2[2]);
    const float2 w3 = __half22float2(wh2[3]);
    acc += (hv0.x * w0.x + hv0.y * w0.y + hv0.z * w1.x + hv0.w * w1.y +
            hv1.x * w2.x + hv1.y * w2.y + hv1.z * w3.x + hv1.w * w3.y) * inv_h;

    const float4 mv0 = m4[idx * 2U];
    const float4 mv1 = m4[idx * 2U + 1U];
    const uint4 wmv = wm8[idx];
    const __half2* wm2 = reinterpret_cast<const __half2*>(&wmv);
    const float2 u0 = __half22float2(wm2[0]);
    const float2 u1 = __half22float2(wm2[1]);
    const float2 u2 = __half22float2(wm2[2]);
    const float2 u3 = __half22float2(wm2[3]);
    acc += (mv0.x * u0.x + mv0.y * u0.y + mv0.z * u1.x + mv0.w * u1.y +
            mv1.x * u2.x + mv1.y * u2.y + mv1.z * u3.x + mv1.w * u3.y) * inv_m;
  }
  for (int off = 16; off > 0; off /= 2) {
    acc += __shfl_xor(acc, off);
  }
  if (lane == 0U) sm_h[wave] = acc;
  __syncthreads();
  if (tid == 0U) {
    float tot = 0.0F;
    for (int w = 0; w < 8; ++w) tot += sm_h[w];
    out[row] = tot;
  }
}

__global__ void PairNormConcatKernel(const float* h, const float* msg,
                                     float* pair, std::uint32_t hidden,
                                     float eps) {
  const std::uint32_t tok = blockIdx.x;
  const float* hr = h + static_cast<std::size_t>(tok) * hidden;
  const float* mr = msg + static_cast<std::size_t>(tok) * hidden;
  float* dst = pair + static_cast<std::size_t>(tok) * hidden * 2;
  float sh = 0.0F;
  float sm = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    sh += hr[i] * hr[i];
    sm += mr[i] * mr[i];
  }
  const float ih =
      rsqrtf(BlockHiddenSum(sh) / static_cast<float>(hidden) + eps);
  const float im =
      rsqrtf(BlockHiddenSum(sm) / static_cast<float>(hidden) + eps);
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst[i] = hr[i] * ih;
    dst[hidden + i] = mr[i] * im;
  }
}

__global__ void ScaleRowsKernel(float* x, const float* scale,
                                std::uint32_t row_elems) {
  const std::uint32_t tok = blockIdx.x;
  const float s = scale[tok];
  float* row = x + static_cast<std::size_t>(tok) * row_elems;
  for (std::uint32_t i = threadIdx.x; i < row_elems; i += blockDim.x) {
    row[i] *= s;
  }
}

__global__ void BuildRoutedTilesKernel(const std::uint32_t* __restrict__ counts,
                                       std::uint32_t n_experts,
                                       std::uint32_t tile_rows,
                                       std::int32_t* __restrict__ tiles,
                                       std::uint32_t* __restrict__ n_tiles) {
  const std::uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_experts) return;
  const std::uint32_t c = counts[e];
  const std::uint32_t chunks = (c + tile_rows - 1U) / tile_rows;
  for (std::uint32_t j = 0; j < chunks; ++j) {
    const std::uint32_t idx = atomicAdd(n_tiles, 1U);
    tiles[idx] = static_cast<std::int32_t>(e | (j << 16));
  }
}

__global__ void SnapPushKernel(const float* h, float* bank,
                               std::uint32_t hidden, std::uint32_t stride,
                               std::uint32_t slot) {
  const std::uint32_t tok = blockIdx.x;
  const float* src = h + static_cast<std::size_t>(tok) * hidden;
  float* dst = bank + static_cast<std::size_t>(tok) * stride +
               static_cast<std::size_t>(slot) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst[i] = src[i];
  }
}

}  // namespace

bool ConvertNvfp4TensorToF16Host(const void* nvfp4,
                                 const float* expert_scales_or_null,
                                 std::uint16_t* f16_out, std::size_t n_elems,
                                 std::uint32_t n_experts) {
  if (nvfp4 == nullptr || f16_out == nullptr || n_experts == 0) return false;
  if (n_elems == 0 || n_elems % (static_cast<std::size_t>(n_experts) *
                                 kNvfp4BlockElems) != 0) {
    return false;
  }
  const auto* src = static_cast<const std::uint8_t*>(nvfp4);
  const std::size_t expert_elems = n_elems / n_experts;
  const std::size_t expert_blocks = expert_elems / kNvfp4BlockElems;
  const std::size_t expert_bytes = expert_blocks * kNvfp4BlockBytes;

  for (std::uint32_t expert = 0; expert < n_experts; ++expert) {
    const std::uint8_t* e_src = src + expert * expert_bytes;
    std::uint16_t* e_dst = f16_out + expert * expert_elems;
    const float outer = (expert_scales_or_null != nullptr)
                            ? expert_scales_or_null[expert]
                            : 1.0F;
    for (std::size_t b = 0; b < expert_blocks; ++b) {
      const std::uint8_t* blk = e_src + b * kNvfp4BlockBytes;
      const std::uint8_t* scales = blk;                 // 4 UE4M3 bytes
      const std::uint8_t* quants = blk + 4;             // 32 packed bytes
      std::uint16_t* dst = e_dst + b * kNvfp4BlockElems;
      for (std::size_t s = 0; s < 4; ++s) {
        const float d = Ue4M3ToFp32(scales[s]) * outer;
        const std::uint8_t* q = quants + s * (kNvfp4SubElems / 2);
        float tmp[kNvfp4SubElems];
        for (std::size_t j = 0; j < kNvfp4SubElems / 2; ++j) {
          const std::uint8_t byte = q[j];
          tmp[j] = d * kE2M1Doubled[byte & 0xFU];
          tmp[j + kNvfp4SubElems / 2] = d * kE2M1Doubled[byte >> 4U];
        }
        for (std::size_t j = 0; j < kNvfp4SubElems; ++j) {
          dst[s * kNvfp4SubElems + j] = F32ToF16Bits(tmp[j]);
        }
      }
    }
  }
  return true;
}

bool ConvertNvfp4TensorToF32Host(const void* nvfp4,
                                 const float* expert_scales_or_null,
                                 float* f32_out, std::size_t n_elems,
                                 std::uint32_t n_experts) {
  if (nvfp4 == nullptr || f32_out == nullptr || n_experts == 0) return false;
  if (n_elems == 0 || n_elems % (static_cast<std::size_t>(n_experts) *
                                 kNvfp4BlockElems) != 0) {
    return false;
  }
  const auto* src = static_cast<const std::uint8_t*>(nvfp4);
  const std::size_t expert_elems = n_elems / n_experts;
  const std::size_t expert_blocks = expert_elems / kNvfp4BlockElems;
  const std::size_t expert_bytes = expert_blocks * kNvfp4BlockBytes;

  for (std::uint32_t expert = 0; expert < n_experts; ++expert) {
    const std::uint8_t* e_src = src + expert * expert_bytes;
    float* e_dst = f32_out + expert * expert_elems;
    const float outer = (expert_scales_or_null != nullptr)
                            ? expert_scales_or_null[expert]
                            : 1.0F;
    for (std::size_t b = 0; b < expert_blocks; ++b) {
      const std::uint8_t* blk = e_src + b * kNvfp4BlockBytes;
      const std::uint8_t* scales = blk;                 // 4 UE4M3 bytes
      const std::uint8_t* quants = blk + 4;             // 32 packed bytes
      float* dst = e_dst + b * kNvfp4BlockElems;
      for (std::size_t s = 0; s < 4; ++s) {
        const float d = Ue4M3ToFp32(scales[s]) * outer;
        const std::uint8_t* q = quants + s * (kNvfp4SubElems / 2);
        for (std::size_t j = 0; j < kNvfp4SubElems / 2; ++j) {
          const std::uint8_t byte = q[j];
          dst[s * kNvfp4SubElems + j] = d * kE2M1Doubled[byte & 0xFU];
          dst[s * kNvfp4SubElems + j + kNvfp4SubElems / 2] =
              d * kE2M1Doubled[byte >> 4U];
        }
      }
    }
  }
  return true;
}

const char* KernelsVersion() { return "inkling-fwd-v2-sconv1t"; }

void RmsNormGamma(const float* x, const float* weight, float* out,
                  std::uint32_t n_tokens, std::uint32_t hidden, float eps,
                  hipStream_t stream) {
  if (n_tokens == 0 || x == nullptr || out == nullptr) return;
  hipLaunchKernelGGL(RmsNormGammaKernel, dim3(n_tokens), dim3(256), 0, stream,
                     x, weight, out, hidden, eps);
}

void RmsNormNoGamma(const float* x, float* out, std::uint32_t n_tokens,
                    std::uint32_t hidden, float eps, hipStream_t stream) {
  if (n_tokens == 0 || x == nullptr || out == nullptr) return;
  hipLaunchKernelGGL(RmsNormNoGammaKernel, dim3(n_tokens), dim3(256), 0, stream,
                     x, out, hidden, eps);
}

void AddResidual(float* a, const float* b, std::size_t count,
                 hipStream_t stream) {
  if (count == 0 || a == nullptr || b == nullptr) return;
  const std::size_t blocks = (count + 255) / 256;
  hipLaunchKernelGGL(AddResidualKernel, dim3(blocks), dim3(256), 0, stream, a,
                     b, count);
}

void StaticSconvForward(const float* x, float* state, const float* taps,
                        float* out, std::uint32_t n_tokens, std::uint32_t dim,
                        float norm, hipStream_t stream, float* scratch,
                        float* pfx_out, std::size_t pfx_stride) {
  if (n_tokens == 0 || x == nullptr || out == nullptr) return;
  const std::uint32_t blocks = (dim + 255) / 256;
  if (n_tokens == 1U) {
    if (pfx_out != nullptr) {
      hipLaunchKernelGGL(SaveSconvPrefixesKernel, dim3(blocks, 1U), dim3(256),
                         0, stream, x, state, pfx_out, pfx_stride, 1U, dim);
    }
    hipLaunchKernelGGL(StaticSconv1TokKernel, dim3(blocks), dim3(256), 0,
                       stream, x, state, taps, out, dim, norm);
    return;
  }
  const float* src = x;
  if (x == out && scratch != nullptr) {
    (void)hipMemcpyAsync(scratch, x,
                         static_cast<std::size_t>(n_tokens) * dim * sizeof(float),
                         hipMemcpyDeviceToDevice, stream);
    src = scratch;
  }
  if (src != out) {
    hipLaunchKernelGGL(StaticSconvParKernel, dim3(blocks, n_tokens), dim3(256),
                       0, stream, src, state, taps, out, n_tokens, dim, norm);
    if (pfx_out != nullptr) {
      hipLaunchKernelGGL(SaveSconvPrefixesKernel, dim3(blocks, n_tokens),
                         dim3(256), 0, stream, src, state, pfx_out, pfx_stride,
                         n_tokens, dim);
    }
    hipLaunchKernelGGL(UpdateSconvStateKernel, dim3(blocks), dim3(256), 0,
                       stream, src, state, n_tokens, dim);
    return;
  }
  for (std::uint32_t b = 0; b < blocks; ++b) {
    hipLaunchKernelGGL(StaticSconvKernel, dim3(1), dim3(256), 0, stream, x,
                       state, taps, out, n_tokens, dim, b * 256, norm);
  }
}

void DynamicSconvForward(const float* x, float* state, const float* taps,
                         const float* delta, float* out,
                         std::uint32_t n_tokens, std::uint32_t dim,
                         std::uint32_t n_heads, float norm,
                         hipStream_t stream, float* scratch,
                         float* pfx_out, std::size_t pfx_stride) {
  if (n_tokens == 0 || x == nullptr || out == nullptr) return;
  if (n_heads == 0 || dim % n_heads != 0) return;
  const std::uint32_t head_dim = dim / n_heads;
  const std::uint32_t blocks = (dim + 255) / 256;
  if (n_tokens == 1U) {
    if (pfx_out != nullptr) {
      hipLaunchKernelGGL(SaveSconvPrefixesKernel, dim3(blocks, 1U), dim3(256),
                         0, stream, x, state, pfx_out, pfx_stride, 1U, dim);
    }
    hipLaunchKernelGGL(DynamicSconv1TokKernel, dim3(blocks), dim3(256), 0,
                       stream, x, state, taps, delta, out, dim, head_dim, norm);
    return;
  }
  const float* src = x;
  if (x == out && scratch != nullptr) {
    (void)hipMemcpyAsync(scratch, x,
                         static_cast<std::size_t>(n_tokens) * dim * sizeof(float),
                         hipMemcpyDeviceToDevice, stream);
    src = scratch;
  }
  if (src != out) {
    hipLaunchKernelGGL(DynamicSconvParKernel, dim3(blocks, n_tokens), dim3(256),
                       0, stream, src, state, taps, delta, out, n_tokens, dim,
                       head_dim, norm);
    if (pfx_out != nullptr) {
      hipLaunchKernelGGL(SaveSconvPrefixesKernel, dim3(blocks, n_tokens),
                         dim3(256), 0, stream, src, state, pfx_out, pfx_stride,
                         n_tokens, dim);
    }
    hipLaunchKernelGGL(UpdateSconvStateKernel, dim3(blocks), dim3(256), 0,
                       stream, src, state, n_tokens, dim);
    return;
  }
  for (std::uint32_t b = 0; b < blocks; ++b) {
    hipLaunchKernelGGL(DynamicSconvKernel, dim3(1), dim3(256), 0, stream, x,
                       state, taps, delta, out, n_tokens, dim, b * 256,
                       head_dim, norm);
  }
}

void QScaleClip(float* x, const float* scale, std::uint32_t n_tokens,
                std::uint32_t n_heads, std::uint32_t head_dim, float clip,
                hipStream_t stream) {
  if (n_tokens == 0 || x == nullptr || scale == nullptr) return;
  hipLaunchKernelGGL(QScaleClipKernel, dim3(n_tokens, n_heads), dim3(128), 0,
                     stream, x, scale, n_heads, head_dim, clip);
}

void L2Clip(float* x, std::uint32_t n_tokens, std::uint32_t n_heads,
            std::uint32_t head_dim, float clip, hipStream_t stream) {
  if (n_tokens == 0 || x == nullptr) return;
  hipLaunchKernelGGL(L2ClipKernel, dim3(n_tokens, n_heads), dim3(128), 0,
                     stream, x, n_heads, head_dim, clip);
}


__global__ void FillLogScaleTauKernel(float* __restrict__ tau,
                                      std::uint32_t n_tokens,
                                      std::uint32_t start_pos,
                                      float inv_floor, float alpha,
                                      bool is_global) {
  const std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_tokens) return;
  if (!is_global) {
    tau[t] = 1.0F;
    return;
  }
  const float eff = static_cast<float>(start_pos + t + 1U) * inv_floor;
  tau[t] = 1.0F + alpha * logf(eff > 1.0F ? eff : 1.0F);
}

void FillLogScaleTau(float* tau, std::uint32_t n_tokens,
                     std::uint32_t start_pos, std::uint32_t floor_val,
                     float alpha, bool is_global, hipStream_t stream) {
  if (n_tokens == 0 || tau == nullptr) return;
  const float inv_floor = 1.0F / static_cast<float>(floor_val);
  hipLaunchKernelGGL(FillLogScaleTauKernel, dim3((n_tokens + 255U) / 256U),
                     dim3(256), 0, stream, tau, n_tokens, start_pos, inv_floor,
                     alpha, is_global);
}

void RelClipTau(float* r, const float* clip, const float* tau,
                std::uint32_t n_tokens, std::uint32_t n_heads, float clip_val,
                hipStream_t stream) {
  if (n_tokens == 0 || r == nullptr || clip == nullptr || tau == nullptr) {
    return;
  }
  hipLaunchKernelGGL(RelClipTauKernel, dim3(n_tokens, n_heads), dim3(32), 0,
                     stream, r, clip, tau, n_heads, clip_val);
}

void AttentionDecode(const float* q, const __half* k_cache,
                     const __half* v_cache, const float* sinks,
                     const float* rel, const float* rel_bank, float* out,
                     std::uint32_t start_pos, std::uint32_t kv_len,
                     std::uint32_t window, std::uint32_t n_heads,
                     std::uint32_t n_kv_heads, std::uint32_t head_dim,
                     std::uint32_t rel_extent, float scale,
                     hipStream_t stream, float* splitk_scratch) {
  (void)kv_len;
  if (head_dim == 128U && splitk_scratch != nullptr && start_pos >= 512U && n_heads <= 32U) {
    const std::uint32_t n_splits = (window > 0U && window <= 2048U) ? 8U : 32U;
    hipLaunchKernelGGL(InklingAttnSplitKKernel, dim3(1U, n_heads, n_splits),
                       dim3(256), 0, stream, q, k_cache, v_cache, sinks, rel,
                       rel_bank, splitk_scratch, start_pos, nullptr, window,
                       n_heads, n_kv_heads, rel_extent, scale, n_splits);
    hipLaunchKernelGGL(InklingAttnSplitKMergeKernel, dim3(1U, n_heads),
                       dim3(32), 0, stream, splitk_scratch, out, n_heads,
                       n_splits);
    return;
  }
  if (head_dim == 128U) {
    hipLaunchKernelGGL(InklingAttnWave32Kernel, dim3(1, n_heads), dim3(256), 0,
                       stream, q, k_cache, v_cache, sinks, rel, rel_bank, out,
                       start_pos, 0, window, n_heads, n_kv_heads, rel_extent,
                       scale);
    return;
  }
  hipLaunchKernelGGL(InklingAttnKernel, dim3(1, n_heads), dim3(head_dim), 0,
                     stream, q, k_cache, v_cache, sinks, rel, rel_bank, out,
                     start_pos, 1, 0, window, n_heads, n_kv_heads, head_dim,
                     rel_extent, scale);
}

void AttentionPrefill(const float* q, const __half* k_cache,
                      const __half* v_cache, const float* sinks,
                      const float* rel, const float* rel_bank, float* out,
                      std::uint32_t n_tokens, std::uint32_t start_pos,
                      std::uint32_t kv_len, std::uint32_t window,
                      std::uint32_t n_heads, std::uint32_t n_kv_heads,
                      std::uint32_t head_dim, std::uint32_t rel_extent,
                      float scale, hipStream_t stream, float* splitk_scratch) {
  if (n_tokens == 0) return;
  (void)kv_len;
  if (head_dim == 128U && splitk_scratch != nullptr && n_tokens <= 8U &&
      n_heads <= 32U && start_pos >= 512U) {
    const std::uint32_t n_splits = (window > 0U && window <= 2048U) ? 8U : 32U;
    hipLaunchKernelGGL(InklingAttnSplitKKernel, dim3(n_tokens, n_heads, n_splits),
                       dim3(256), 0, stream, q, k_cache, v_cache, sinks, rel,
                       rel_bank, splitk_scratch, start_pos, nullptr, window,
                       n_heads, n_kv_heads, rel_extent, scale, n_splits);
    hipLaunchKernelGGL(InklingAttnSplitKMergeKernel, dim3(n_tokens, n_heads),
                       dim3(32), 0, stream, splitk_scratch, out, n_heads,
                       n_splits);
    return;
  }
  if (head_dim == 128U && n_heads == 24U && n_kv_heads == 8U && n_tokens >= 16U) {
    hipLaunchKernelGGL((InklingAttnWmmaPrefillKernel<128U, 3U, 2U>),
                       dim3((n_tokens + 31U) / 32U, 8U), dim3(192), 0, stream,
                       q, k_cache, v_cache, sinks, rel, rel_bank, out,
                       start_pos, n_tokens, n_heads, n_kv_heads, window,
                       rel_extent, scale);
    return;
  }
  if (head_dim == 128U && n_heads == 24U && n_kv_heads == 8U && n_tokens >= 4U) {
    hipLaunchKernelGGL(InklingAttnPrefillTiledKernel,
                       dim3((n_tokens + 3U) / 4U, 8U), dim3(384), 0, stream, q,
                       k_cache, v_cache, sinks, rel, rel_bank, out, start_pos,
                       n_tokens, window, n_heads, n_kv_heads, rel_extent,
                       scale);
    return;
  }
  if (head_dim == 128U) {
    hipLaunchKernelGGL(InklingAttnWave32Kernel, dim3(n_tokens, n_heads),
                       dim3(256), 0, stream, q, k_cache, v_cache, sinks, rel,
                       rel_bank, out, start_pos, 0, window, n_heads, n_kv_heads,
                       rel_extent, scale);
    return;
  }
  hipLaunchKernelGGL(InklingAttnKernel, dim3(n_tokens, n_heads),
                     dim3(head_dim), 0, stream, q, k_cache, v_cache, sinks,
                     rel, rel_bank, out, start_pos, n_tokens, 0, window,
                     n_heads, n_kv_heads, head_dim, rel_extent, scale);
}

void SigmoidGateMul(float* attn_out, const float* gate, std::uint32_t n_tokens,
                    std::uint32_t n_heads, std::uint32_t head_dim,
                    hipStream_t stream) {
  if (n_tokens == 0 || attn_out == nullptr || gate == nullptr) return;
  hipLaunchKernelGGL(SigmoidGateMulKernel, dim3(n_tokens, n_heads),
                     dim3(head_dim), 0, stream, attn_out, gate, n_heads,
                     head_dim);
}

void ScaleRows(float* x, const float* scale, std::uint32_t n_tokens,
               std::uint32_t row_elems, hipStream_t stream) {
  if (n_tokens == 0 || x == nullptr || scale == nullptr || row_elems == 0) {
    return;
  }
  hipLaunchKernelGGL(ScaleRowsKernel, dim3(n_tokens), dim3(256), 0, stream, x,
                     scale, row_elems);
}

void SnapPush(const float* h, float* bank, std::uint32_t n_tokens,
              std::uint32_t hidden, std::uint32_t stride, std::uint32_t slot,
              hipStream_t stream) {
  if (n_tokens == 0 || h == nullptr || bank == nullptr) return;
  hipLaunchKernelGGL(SnapPushKernel, dim3(n_tokens), dim3(256), 0, stream, h,
                     bank, hidden, stride, slot);
}

void StoreKvF16(const float* k_in, const float* v_in, __half* k_cache,
                __half* v_cache, std::uint32_t n_tokens,
                std::uint32_t start_pos, std::uint32_t n_kv_heads,
                std::uint32_t head_dim, hipStream_t stream) {
  if (n_tokens == 0 || k_in == nullptr || v_in == nullptr) return;
  hipLaunchKernelGGL(StoreKvF16Kernel, dim3(n_tokens), dim3(256), 0, stream,
                     k_in, v_in, k_cache, v_cache, start_pos, n_kv_heads,
                     head_dim);
}

void SigmoidRouterTop8(const float* logits, const float* bias,
                       std::int32_t* ids, float* weights,
                       std::uint32_t n_tokens, std::uint32_t n_experts,
                       float route_scale, hipStream_t stream) {
  if (n_tokens == 0 || logits == nullptr || ids == nullptr ||
      weights == nullptr) {
    return;
  }
  hipLaunchKernelGGL(SigmoidRouterTop8Kernel, dim3(n_tokens), dim3(1), 0,
                     stream, logits, bias, ids, weights, n_experts, 8,
                     route_scale, 1e-20F);
}

void MoeEpilogueHalf(const __half* expert_out, const float* weights,
                     float* out, std::uint32_t n_tokens, std::uint32_t k,
                     std::uint32_t dim, hipStream_t stream) {
  if (n_tokens == 0 || dim == 0 || k == 0) return;
  hipLaunchKernelGGL(MoeEpilogueHalfKernel,
                     dim3(n_tokens, (dim + 255U) / 256U), dim3(256), 0, stream,
                     expert_out, weights, out, k, dim);
}

void SituFuseHalfToF16(const __half* gateup, __half* act16, std::uint32_t rows,
                       std::uint32_t ff, hipStream_t stream) {
  if (rows == 0 || ff == 0) return;
  const std::uint32_t threads = ff < 256 ? ff : 256;
  hipLaunchKernelGGL(SituFuseHalfToF16Kernel, dim3(rows), dim3(threads), 0,
                     stream, gateup, act16, ff);
}

void SituFuseToF16(const float* gateup, __half* act16, std::uint32_t rows,
                   std::uint32_t ff, hipStream_t stream) {
  if (rows == 0 || gateup == nullptr || act16 == nullptr) return;
  const std::uint32_t block = std::min<std::uint32_t>(256, ff);
  hipLaunchKernelGGL(SituFuseToF16Kernel, dim3(rows), dim3(block), 0, stream,
                     gateup, act16, ff);
}

void SituFuse(const float* gateup, float* act, std::uint32_t rows,
              std::uint32_t ff, hipStream_t stream) {
  if (rows == 0 || gateup == nullptr || act == nullptr || ff == 0) return;
  const std::uint32_t block = ff > 1024 ? 1024 : ff;
  hipLaunchKernelGGL(SituFuseKernel, dim3(rows), dim3(block), 0, stream,
                     gateup, act, ff);
}

void GatherRows(const float* x, const std::int32_t* list, float* out,
                std::uint32_t batch, std::uint32_t hidden,
                hipStream_t stream) {
  if (batch == 0 || x == nullptr || out == nullptr) return;
  const std::uint32_t block = hidden > 256 ? 256 : hidden;
  hipLaunchKernelGGL(GatherRowsKernel, dim3(batch), dim3(block), 0, stream, x,
                     list, out, batch, hidden);
}

void ScatterAddWeighted(const float* x, const float* weights,
                        const std::int32_t* list, float* out,
                        std::uint32_t batch, std::uint32_t hidden,
                        hipStream_t stream) {
  if (batch == 0 || x == nullptr || out == nullptr) return;
  const std::uint32_t block = hidden > 256 ? 256 : hidden;
  hipLaunchKernelGGL(ScatterAddWeightedKernel, dim3(batch), dim3(block), 0,
                     stream, x, weights, list, out, batch, hidden);
}

void MultiQueryResAttn4(const float* states, std::uint32_t n_states,
                        std::uint32_t state_stride, const float* q0,
                        const float* q1, const float* q2, const float* q3,
                        float* out0, float* out1, float* out2, float* out3,
                        std::uint32_t n_tokens, std::uint32_t hidden,
                        float scale, hipStream_t stream) {
  if (n_tokens == 0 || n_states == 0 || n_states > 16U) return;
  hipLaunchKernelGGL(MultiQueryResAttn4Kernel, dim3(n_tokens), dim3(256), 0,
                     stream, states, n_states, state_stride, q0, q1, q2, q3,
                     out0, out1, out2, out3, hidden, scale);
}

void MultiQueryResAttn(const float* states, std::uint32_t n_states,
                       std::uint32_t state_stride, const float* queries,
                       std::uint32_t n_queries, float* out,
                       std::uint32_t n_tokens, std::uint32_t hidden,
                       float scale, hipStream_t stream) {
  if (n_tokens == 0 || n_states == 0 || n_queries == 0 || states == nullptr ||
      queries == nullptr || out == nullptr) {
    return;
  }
  if (n_states > 16 || n_queries > 24) return;
  hipLaunchKernelGGL(MultiQueryResAttnKernel, dim3(n_tokens), dim3(256), 0,
                     stream, states, n_states, state_stride, queries,
                     n_queries, out, hidden, scale, 1e-6F);
}

void FuseGateProjApply(const float* h, const float* msg, const float* g64,
                       const __half* forget_w, const __half* gain_w, float* out,
                       std::uint32_t n_tokens, std::uint32_t hidden,
                       std::uint32_t rank, hipStream_t stream) {
  if (n_tokens == 0 || hidden == 0 || rank == 0 || rank > 64U) return;
  if (rank == 64U && n_tokens > 8U) {
    hipLaunchKernelGGL(FuseGateProjApplyWmmaPrefillKernel,
                       dim3((n_tokens + 31U) / 32U, (hidden + 63U) / 64U),
                       dim3(256), 0, stream, h, msg, g64, forget_w, gain_w,
                       out, n_tokens, hidden);
    return;
  }
  if (rank == 64U) {
    hipLaunchKernelGGL(FuseGateProjApplyWave64Kernel,
                       dim3(n_tokens, (hidden + 7U) / 8U), dim3(256), 0,
                       stream, h, msg, g64, forget_w, gain_w, out, hidden);
    return;
  }
  hipLaunchKernelGGL(FuseGateProjApplyKernel,
                     dim3(n_tokens, (hidden + 255U) / 256U), dim3(256), 0,
                     stream, h, msg, g64, forget_w, gain_w, out, hidden, rank);
}

void GateFuseApply(const float* h, const float* msg, const float* fg_raw,
                   const float* mg_raw, float* out, std::uint32_t n_tokens,
                   std::uint32_t hidden, hipStream_t stream) {
  if (n_tokens == 0 || h == nullptr || msg == nullptr || out == nullptr) {
    return;
  }
  hipLaunchKernelGGL(GateFuseApplyKernel, dim3(n_tokens), dim3(256), 0, stream,
                     h, msg, fg_raw, mg_raw, out, hidden);
}

void FusedPairNormDown(const float* h, const float* msg, const __half* down_w,
                       float* out, std::uint32_t hs, std::uint32_t rank,
                       float eps, hipStream_t stream) {
  if (rank == 0 || hs == 0 || (hs % 8U) != 0) return;
  hipLaunchKernelGGL(FusedPairNormDownKernel, dim3(rank), dim3(256), 0, stream,
                     h, msg, down_w, out, hs, eps);
}

void PairNormConcat(const float* h, const float* msg, float* pair,
                    std::uint32_t n_tokens, std::uint32_t hidden, float eps,
                    hipStream_t stream) {
  if (n_tokens == 0 || h == nullptr || msg == nullptr || pair == nullptr) {
    return;
  }
  hipLaunchKernelGGL(PairNormConcatKernel, dim3(n_tokens), dim3(256), 0, stream,
                     h, msg, pair, hidden, eps);
}

void BuildRoutedTiles(const std::uint32_t* counts, std::uint32_t n_experts,
                      std::uint32_t tile_rows, std::int32_t* tiles,
                      std::uint32_t* n_tiles, hipStream_t stream) {
  (void)hipMemsetAsync(n_tiles, 0, sizeof(std::uint32_t), stream);
  const unsigned blocks = static_cast<unsigned>((n_experts + 255U) / 256U);
  hipLaunchKernelGGL(BuildRoutedTilesKernel, dim3(blocks), dim3(256), 0,
                     stream, counts, n_experts, tile_rows, tiles, n_tiles);
}


__global__ void FillLogScaleTauByPosKernel(float* __restrict__ tau,
                                           const std::uint32_t* __restrict__ d_pos,
                                           float inv_floor, float alpha,
                                           bool is_global) {
  if (!is_global) {
    tau[0] = 1.0F;
    return;
  }
  const float eff = static_cast<float>(*d_pos + 1U) * inv_floor;
  tau[0] = 1.0F + alpha * logf(eff > 1.0F ? eff : 1.0F);
}

__global__ void StoreKvF16ByPosKernel(const float* __restrict__ k_in,
                                      const float* __restrict__ v_in,
                                      __half* __restrict__ k_cache,
                                      __half* __restrict__ v_cache,
                                      const std::uint32_t* __restrict__ d_pos,
                                      std::uint32_t kv_stride) {
  const std::size_t dst_off = static_cast<std::size_t>(*d_pos) * kv_stride;
  for (std::size_t i = threadIdx.x; i < kv_stride; i += blockDim.x) {
    k_cache[dst_off + i] = __float2half(k_in[i]);
    v_cache[dst_off + i] = __float2half(v_in[i]);
  }
}

__global__ void CopyKvSlotByPosKernel(__half* __restrict__ cache,
                                      __half* __restrict__ backup,
                                      const std::uint32_t* __restrict__ d_pos,
                                      std::uint32_t kv_dim, bool to_backup) {
  const std::size_t off = static_cast<std::size_t>(*d_pos) * kv_dim;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kv_dim;
       i += gridDim.x * blockDim.x) {
    if (to_backup) {
      backup[i] = cache[off + i];
    } else {
      cache[off + i] = backup[i];
    }
  }
}

__global__ void __launch_bounds__(256) InklingAttnWave32ByPosKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const float* __restrict__ sinks,
    const float* __restrict__ rel, const float* __restrict__ rel_bank,
    float* __restrict__ out, const std::uint32_t* __restrict__ d_pos,
    std::uint32_t window, std::uint32_t n_q_heads, std::uint32_t n_kv_heads,
    std::uint32_t rel_extent, float scale) {
  const std::uint32_t qh = blockIdx.y;
  const std::uint32_t wave = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t kv_group = n_q_heads / n_kv_heads;
  const std::uint32_t kvh = qh / kv_group;
  const std::uint32_t cur_pos = *d_pos;
  const std::uint32_t lo =
      (window > 0 && cur_pos + 1U > window) ? (cur_pos + 1U - window) : 0U;

  const float* q_vec = q + static_cast<std::size_t>(qh) * 128U;
  const float4 q4 = *reinterpret_cast<const float4*>(q_vec + lane * 4U);
  const float* rel_vec =
      (rel != nullptr) ? (rel + static_cast<std::size_t>(qh) * 16U) : nullptr;
  const float rv = (rel_vec != nullptr && lane < 16U) ? rel_vec[lane] : 0.0F;
  const std::size_t kv_stride = static_cast<std::size_t>(n_kv_heads) * 128U;
  const std::size_t kv_head_off = static_cast<std::size_t>(kvh) * 128U + lane * 4U;

  float m = (wave == 0U && sinks != nullptr) ? sinks[qh] : -1e30F;
  float l = (wave == 0U && sinks != nullptr) ? 1.0F : 0.0F;
  float4 acc = make_float4(0.0F, 0.0F, 0.0F, 0.0F);

  for (std::uint32_t p = lo + wave; p <= cur_pos; p += 8U) {
    const std::size_t kv_off =
        static_cast<std::size_t>(p) * kv_stride + kv_head_off;
    const uint2 k_raw = *reinterpret_cast<const uint2*>(k_cache + kv_off);
    const __half2* kh = reinterpret_cast<const __half2*>(&k_raw);
    const float2 k01 = __half22float2(kh[0]);
    const float2 k23 = __half22float2(kh[1]);
    float dot = q4.x * k01.x + q4.y * k01.y + q4.z * k23.x + q4.w * k23.y;
    for (int off = 16; off > 0; off /= 2) dot += __shfl_xor(dot, off);
    float score = dot * scale;
    if (rel_vec != nullptr && rel_bank != nullptr) {
      const std::uint32_t dist = cur_pos - p;
      if (dist < rel_extent) {
        float rb = (lane < 16U) ? (rv * rel_bank[static_cast<std::size_t>(dist) * 16U + lane]) : 0.0F;
        for (int off = 8; off > 0; off /= 2) rb += __shfl_xor(rb, off);
        score += __shfl(rb, 0);
      }
    }
    const float m_new = fmaxf(m, score);
    const float alpha = __expf(m - m_new);
    const float w = __expf(score - m_new);
    const uint2 v_raw = *reinterpret_cast<const uint2*>(v_cache + kv_off);
    const __half2* vh = reinterpret_cast<const __half2*>(&v_raw);
    const float2 v01 = __half22float2(vh[0]);
    const float2 v23 = __half22float2(vh[1]);
    acc.x = acc.x * alpha + w * v01.x;
    acc.y = acc.y * alpha + w * v01.y;
    acc.z = acc.z * alpha + w * v23.x;
    acc.w = acc.w * alpha + w * v23.y;
    l = l * alpha + w;
    m = m_new;
  }

  __shared__ float smem_m[8];
  __shared__ float smem_l[8];
  __shared__ float smem_acc[8][128];
  if (lane == 0U) {
    smem_m[wave] = m;
    smem_l[wave] = l;
  }
  *reinterpret_cast<float4*>(&smem_acc[wave][lane * 4U]) = acc;
  __syncthreads();

  if (wave == 0U) {
    float m_all = -1e30F;
    for (int w = 0; w < 8; ++w) {
      if (smem_l[w] > 0.0F) m_all = fmaxf(m_all, smem_m[w]);
    }
    float l_tot = 0.0F;
    float4 acc_tot = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    for (int w = 0; w < 8; ++w) {
      const float lw = smem_l[w];
      if (lw > 0.0F) {
        const float sw = __expf(smem_m[w] - m_all);
        l_tot += sw * lw;
        const float4 aw = *reinterpret_cast<const float4*>(&smem_acc[w][lane * 4U]);
        acc_tot.x += sw * aw.x;
        acc_tot.y += sw * aw.y;
        acc_tot.z += sw * aw.z;
        acc_tot.w += sw * aw.w;
      }
    }
    const float inv_l = (l_tot > 0.0F) ? (1.0F / l_tot) : 0.0F;
    float4 out4 = make_float4(acc_tot.x * inv_l, acc_tot.y * inv_l,
                              acc_tot.z * inv_l, acc_tot.w * inv_l);
    *reinterpret_cast<float4*>(out + static_cast<std::size_t>(qh) * 128U + lane * 4U) = out4;
  }
}

void FillLogScaleTauByPos(float* tau, const std::uint32_t* d_pos,
                          std::uint32_t floor_val, float alpha, bool is_global,
                          hipStream_t stream) {
  hipLaunchKernelGGL(FillLogScaleTauByPosKernel, dim3(1), dim3(1), 0, stream,
                     tau, d_pos, 1.0F / static_cast<float>(floor_val), alpha,
                     is_global);
}

void StoreKvF16ByPos(const float* k_in, const float* v_in, __half* k_cache,
                     __half* v_cache, const std::uint32_t* d_pos,
                     std::uint32_t n_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream) {
  const std::uint32_t kv_stride = n_kv_heads * head_dim;
  hipLaunchKernelGGL(StoreKvF16ByPosKernel, dim3(1), dim3(256), 0, stream,
                     k_in, v_in, k_cache, v_cache, d_pos, kv_stride);
}

void AttentionDecodeByPos(const float* q, const __half* k_cache,
                          const __half* v_cache, const float* sinks,
                          const float* rel, const float* rel_bank, float* out,
                          const std::uint32_t* d_pos, std::uint32_t window,
                          std::uint32_t n_heads, std::uint32_t n_kv_heads,
                          std::uint32_t head_dim, std::uint32_t rel_extent,
                          float scale, hipStream_t stream, float* splitk_scratch) {
  if (head_dim == 128U && splitk_scratch != nullptr && n_heads <= 32U) {
    const std::uint32_t n_splits = (window > 0U && window <= 2048U) ? 8U : 32U;
    hipLaunchKernelGGL(InklingAttnSplitKKernel, dim3(1U, n_heads, n_splits),
                       dim3(256), 0, stream, q, k_cache, v_cache, sinks, rel,
                       rel_bank, splitk_scratch, 0U, d_pos, window,
                       n_heads, n_kv_heads, rel_extent, scale, n_splits);
    hipLaunchKernelGGL(InklingAttnSplitKMergeKernel, dim3(1U, n_heads),
                       dim3(32), 0, stream, splitk_scratch, out, n_heads,
                       n_splits);
    return;
  }
  hipLaunchKernelGGL(InklingAttnWave32ByPosKernel, dim3(1, n_heads), dim3(256),
                     0, stream, q, k_cache, v_cache, sinks, rel, rel_bank, out,
                     d_pos, window, n_heads, n_kv_heads, rel_extent, scale);
}

void CopyKvSlotByPos(__half* cache, __half* backup, const std::uint32_t* d_pos,
                     std::uint32_t kv_dim, bool to_backup, hipStream_t stream) {
  hipLaunchKernelGGL(CopyKvSlotByPosKernel, dim3((kv_dim + 255U) / 256U),
                     dim3(256), 0, stream, cache, backup, d_pos, kv_dim,
                     to_backup);
}

}  // namespace mbufo::models::inkling::kernels
