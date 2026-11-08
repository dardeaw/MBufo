// Llama dense-decoder kernels: simple, obviously-correct implementations.
// Prefill throughput comes from the shared core (DenseF16Gemm / Split-K);
// these kernels handle the elementwise and narrow-bandwidth work.
#include "src/models/llama/kernels/kernels.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace gufo::models::glimmer::kernels {
namespace {

// Q8_0 block (GGUF grid): f16 scale first, then 32 int8 codes. (This is
// NOT the shared blocked-KV grid, which trails the scale at +32.)
__device__ __forceinline__ float DequantQ8(const std::uint8_t* blk,
                                           std::uint32_t e) {
  const float scale = __half2float(*reinterpret_cast<const __half*>(blk));
  return static_cast<float>(static_cast<std::int8_t>(blk[2 + e])) * scale;
}

__global__ void EmbedKernel(const void* __restrict__ table,
                            core::GgmlType type,
                            const std::int32_t* __restrict__ ids,
                            float* __restrict__ out, std::uint32_t hidden) {
  const std::uint32_t t = blockIdx.x;
  // 2-D grid (token, wave): each block owns 256 dims of the row.
  const std::uint32_t i = blockIdx.y * 256u + threadIdx.x;
  if (i >= hidden)
    return;
  const std::int32_t id = ids[t];
  float v = 0.0f;
  if (type == core::GgmlType::kQ8_0) {
    const auto* tab = static_cast<const std::uint8_t*>(table);
    const std::size_t row_bytes = (hidden / 32) * 34;
    const std::uint8_t* row =
        tab + static_cast<std::size_t>(id) * row_bytes;
    const std::uint32_t blk = i / 32;
    v = DequantQ8(row + static_cast<std::size_t>(blk) * 34, i % 32);
  } else if (type == core::GgmlType::kF16) {
    const auto* tab = static_cast<const __half*>(table);
    v = __half2float(tab[static_cast<std::size_t>(id) * hidden + i]);
  } else {
    const auto* tab = static_cast<const float*>(table);
    v = tab[static_cast<std::size_t>(id) * hidden + i];
  }
  out[static_cast<std::size_t>(t) * hidden + i] = v;
}

__global__ void RmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ gamma,
                              float* __restrict__ out, std::uint32_t dim,
                              float eps) {
  const std::uint32_t t = blockIdx.x;
  const float* row = x + static_cast<std::size_t>(t) * dim;
  float* orow = out + static_cast<std::size_t>(t) * dim;
  __shared__ float red[32];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5;
  float v = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float x0 = row[i];
    v += x0 * x0;
  }
  v += __shfl_down_sync(0xffffffffull, v, 16);
  v += __shfl_down_sync(0xffffffffull, v, 8);
  v += __shfl_down_sync(0xffffffffull, v, 4);
  v += __shfl_down_sync(0xffffffffull, v, 2);
  v += __shfl_down_sync(0xffffffffull, v, 1);
  if (lane == 0)
    red[warp] = v;
  __syncthreads();
  float total = 0.0f;
  if (threadIdx.x < 8)
    total = red[threadIdx.x];
  total += __shfl_down_sync(0xffffffffull, total, 4);
  total += __shfl_down_sync(0xffffffffull, total, 2);
  total += __shfl_down_sync(0xffffffffull, total, 1);
  // Broadcasts never cross wavefronts: share lane 0's sum through LDS.
  __shared__ float inv_s;
  if (threadIdx.x == 0)
    inv_s = rsqrtf(total / static_cast<float>(dim) + eps);
  __syncthreads();
  const float inv = inv_s;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    orow[i] = row[i] * inv * gamma[i];
  }
}

__global__ void RmsNormAddKernel(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 float* __restrict__ residual,
                                 std::uint32_t dim, float eps) {
  const std::uint32_t t = blockIdx.x;
  const float* row = x + static_cast<std::size_t>(t) * dim;
  float* rrow = residual + static_cast<std::size_t>(t) * dim;
  __shared__ float red[32];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5;
  float v = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float x0 = row[i];
    v += x0 * x0;
  }
  v += __shfl_down_sync(0xffffffffull, v, 16);
  v += __shfl_down_sync(0xffffffffull, v, 8);
  v += __shfl_down_sync(0xffffffffull, v, 4);
  v += __shfl_down_sync(0xffffffffull, v, 2);
  v += __shfl_down_sync(0xffffffffull, v, 1);
  if (lane == 0)
    red[warp] = v;
  __syncthreads();
  float total = 0.0f;
  if (threadIdx.x < 8)
    total = red[threadIdx.x];
  total += __shfl_down_sync(0xffffffffull, total, 4);
  total += __shfl_down_sync(0xffffffffull, total, 2);
  total += __shfl_down_sync(0xffffffffull, total, 1);
  // Broadcasts never cross wavefronts: share lane 0's sum through LDS.
  __shared__ float inv_s;
  if (threadIdx.x == 0)
    inv_s = rsqrtf(total / static_cast<float>(dim) + eps);
  __syncthreads();
  const float inv = inv_s;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    rrow[i] += row[i] * inv * gamma[i];
  }
}

// Consecutive-pair rotation (GGML_ROPE_TYPE_NORMAL): pair (2i, 2i+1)
// rotates by pos * theta^(-2i/head_dim). muse-glimmer maps to
// LLAMA_ROPE_TYPE_NORM (not NeoX), so the halves must NOT be paired as
// (i, i+d/2); that mismatch collapses attention into echo.
__global__ void RopeKernel(float* __restrict__ x, std::uint32_t heads,
                           std::uint32_t head_dim, std::uint32_t start_pos,
                           float theta) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t i = threadIdx.x;  // 0 .. head_dim/2 (pair index)
  if (i >= head_dim / 2)
    return;
  float* row =
      x + (static_cast<std::size_t>(t) * heads + h) * head_dim;
  const float pos = static_cast<float>(start_pos + t);
  const float inv_freq =
      1.0f / powf(theta, (2.0f * static_cast<float>(i)) /
                             static_cast<float>(head_dim));
  const float ang = pos * inv_freq;
  const float c = cosf(ang);
  const float s = sinf(ang);
  const float a = row[2u * i];
  const float b = row[2u * i + 1u];
  row[2u * i] = a * c - b * s;
  row[2u * i + 1u] = a * s + b * c;
}

// NeoX split-half rotation (LLAMA_ROPE_TYPE_NEOX, dflash draft): pair
// (i, i+head_dim/2) rotates by pos * theta^(-2i/head_dim). Same
// frequencies as RopeKernel, different pairing; that pairing is what the
// draft was trained with, and consecutive-pair rotation here collapses
// its block-diffusion acceptance.
__global__ void RopeNeoxKernel(float* __restrict__ x,
                               std::uint32_t heads,
                               std::uint32_t head_dim,
                               std::uint32_t start_pos, float theta) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t i = threadIdx.x;  // 0 .. head_dim/2 (pair index)
  if (i >= head_dim / 2)
    return;
  float* row =
      x + (static_cast<std::size_t>(t) * heads + h) * head_dim;
  const float pos = static_cast<float>(start_pos + t);
  const float inv_freq =
      1.0f / powf(theta, (2.0f * static_cast<float>(i)) /
                              static_cast<float>(head_dim));
  const float ang = pos * inv_freq;
  const float c = cosf(ang);
  const float s = sinf(ang);
  const std::uint32_t j = head_dim / 2u;
  const float a = row[i];
  const float b = row[i + j];
  row[i] = a * c - b * s;
  row[i + j] = a * s + b * c;
}

__global__ void SiluMulKernel(const float* __restrict__ gate,
                              const float* __restrict__ up,
                              float* __restrict__ out, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (i >= count)
    return;
  const float g = gate[i];
  out[i] = (g / (1.0f + expf(-g))) * up[i];
}

__global__ void F32ToF16Kernel(const float* __restrict__ in,
                               __half* __restrict__ out, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (i >= count)
    return;
  out[i] = __float2half(in[i]);
}

__global__ void StoreKvF16Kernel(const float* __restrict__ src,
                                 __half* __restrict__ cache,
                                 std::uint32_t dim, std::uint32_t start_pos) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * 256u + threadIdx.x;
  if (i >= dim)
    return;
  cache[(static_cast<std::size_t>(start_pos + t)) * dim + i] =
      __float2half(src[static_cast<std::size_t>(t) * dim + i]);
}

// One wavefront (32 threads) per (head, query token); each lane owns
// d/32 dims (d == 128 production). The QK dot replays over the lane's dims
// and folds within the wavefront, so no __syncthreads is needed; the
// project-wide wave32 contract applies (same as the shared blocked
// kernels). Online softmax is lane-0-driven with per-key broadcasts that
// stay inside the wavefront. Deterministic: fixed key order, one state.
__global__ void AttentionF16Kernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, float* __restrict__ out,
    std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t d, float score_scale,
    std::uint32_t window) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;  // 0 .. 31
  const std::uint32_t kvh = h / (heads / kv_heads);
  const std::uint32_t pos = start_pos + t;
  const std::uint32_t n_kv = pos + 1;
  // Sliding window: keys at or below (pos - window) are cut (llama.cpp SWA
  // semantics: visible extent is (pos+1-window, pos]). Window 0 = dense.
  const std::uint32_t k_beg =
      (window != 0 && n_kv > window) ? n_kv - window : 0;
  // Dims per lane (d == 32/64/128 covered; narrower shapes idle upper lanes
  // but keep them in the shuffle for uniformity).
  const std::uint32_t dpl = d <= 32u ? 1u : d / 32u;
  const bool active = (lane * dpl < d);
  const float* qrow = q + (static_cast<std::size_t>(t) * heads + h) * d;
  float m = -1e30f;
  float l = 0.0f;
  // Per-lane value accumulators (up to 4 dims for d == 128).
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (std::uint32_t j = k_beg; j < n_kv; ++j) {
    const __half* krow =
        k_cache + (static_cast<std::size_t>(j) * kv_heads + kvh) * d;
    float dot = 0.0f;
    if (active) {
      for (std::uint32_t u = 0; u < dpl; ++u) {
        const std::uint32_t e = lane * dpl + u;
        dot += (e < d) ? qrow[e] * __half2float(krow[e]) : 0.0f;
      }
    }
    dot += __shfl_down_sync(0xffffffffull, dot, 16);
    dot += __shfl_down_sync(0xffffffffull, dot, 8);
    dot += __shfl_down_sync(0xffffffffull, dot, 4);
    dot += __shfl_down_sync(0xffffffffull, dot, 2);
    dot += __shfl_down_sync(0xffffffffull, dot, 1);
    // Lane 0's dot is complete; share it wavefront-wide (no sync needed).
    const float full = __shfl_sync(0xffffffffull, dot, 0);
    const float score = full * score_scale;
    float m_new = m;
    if (lane == 0) {
      m_new = m > score ? m : score;
    }
    const float msh = __shfl_sync(0xffffffffull, m_new, 0);
    const float corr = (lane == 0 && m == -1e30f) ? 0.0f : expf(m - msh);
    const float wgt = expf(score - msh);
    if (active) {
      const __half* vrow =
          v_cache + (static_cast<std::size_t>(j) * kv_heads + kvh) * d;
      const std::uint32_t e0 = lane * dpl;
      const float vs0 =
          (dpl > 0 && e0 + 0 < d) ? __half2float(vrow[e0 + 0]) : 0.0f;
      const float vs1 =
          (dpl > 1 && e0 + 1 < d) ? __half2float(vrow[e0 + 1]) : 0.0f;
      const float vs2 =
          (dpl > 2 && e0 + 2 < d) ? __half2float(vrow[e0 + 2]) : 0.0f;
      const float vs3 =
          (dpl > 3 && e0 + 3 < d) ? __half2float(vrow[e0 + 3]) : 0.0f;
      acc0 = acc0 * corr + wgt * vs0;
      acc1 = acc1 * corr + wgt * vs1;
      acc2 = acc2 * corr + wgt * vs2;
      acc3 = acc3 * corr + wgt * vs3;
    }
    m = msh;
    if (lane == 0) {
      l = l * corr + wgt;
    }
  }
  const float mf = __shfl_sync(0xffffffffull, m, 0);
  const float lf = __shfl_sync(0xffffffffull, l, 0);
  if (active) {
    float* orow = out + (static_cast<std::size_t>(t) * heads + h) * d;
    // Online states on lane 0 are exact; other lanes replayed the same
    // broadcasts, so their (m, l) match lane 0's bit-for-bit.
    const float inv = 1.0f / lf;
    const std::uint32_t e0 = lane * dpl;
    const float fix = expf(m - mf) * inv;
    if (dpl > 0 && e0 + 0 < d)
      orow[e0 + 0] = acc0 * fix;
    if (dpl > 1 && e0 + 1 < d)
      orow[e0 + 1] = acc1 * fix;
    if (dpl > 2 && e0 + 2 < d)
      orow[e0 + 2] = acc2 * fix;
    if (dpl > 3 && e0 + 3 < d)
      orow[e0 + 3] = acc3 * fix;
  }
}

}  // namespace

void Embed(const void* table, core::GgmlType type, const std::int32_t* ids,
           float* out, std::uint32_t n_tokens, std::uint32_t hidden,
           hipStream_t stream) {
  if (n_tokens == 0)
    return;
  // 2-D grid (token, 256-dim wave) covers any hidden width.
  const std::uint32_t waves = (hidden + 255) / 256;
  hipLaunchKernelGGL(EmbedKernel, dim3(n_tokens, waves), dim3(256), 0, stream,
                     table, type, ids, out, hidden);
}

void RmsNorm(const float* x, const float* gamma, float* out,
             std::uint32_t n_tokens, std::uint32_t dim, float eps,
             hipStream_t stream) {
  if (n_tokens == 0)
    return;
  hipLaunchKernelGGL(RmsNormKernel, dim3(n_tokens), dim3(256), 0, stream, x,
                     gamma, out, dim, eps);
}

__global__ void ResidualAddKernel(const float* __restrict__ x,
                                   float* __restrict__ residual,
                                   std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (i >= count)
    return;
  residual[i] += x[i];
}

__global__ void RmsNormPlainKernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    std::uint32_t dim, float eps) {
  const std::uint32_t t = blockIdx.x;
  const float* row = x + static_cast<std::size_t>(t) * dim;
  float* orow = out + static_cast<std::size_t>(t) * dim;
  __shared__ float red[32];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5;
  float v = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float x0 = row[i];
    v += x0 * x0;
  }
  v += __shfl_down_sync(0xffffffffull, v, 16);
  v += __shfl_down_sync(0xffffffffull, v, 8);
  v += __shfl_down_sync(0xffffffffull, v, 4);
  v += __shfl_down_sync(0xffffffffull, v, 2);
  v += __shfl_down_sync(0xffffffffull, v, 1);
  if (lane == 0)
    red[warp] = v;
  __syncthreads();
  float total = 0.0f;
  if (threadIdx.x < 8)
    total = red[threadIdx.x];
  total += __shfl_down_sync(0xffffffffull, total, 4);
  total += __shfl_down_sync(0xffffffffull, total, 2);
  total += __shfl_down_sync(0xffffffffull, total, 1);
  __shared__ float inv_s;
  if (threadIdx.x == 0)
    inv_s = rsqrtf(total / static_cast<float>(dim) + eps);
  __syncthreads();
  const float inv = inv_s;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    orow[i] = row[i] * inv;
  }
}

void RmsNormPlain(const float* x, float* out, std::uint32_t n_tokens,
                  std::uint32_t dim, float eps, hipStream_t stream) {
  if (n_tokens == 0)
    return;
  hipLaunchKernelGGL(RmsNormPlainKernel, dim3(n_tokens), dim3(256), 0, stream,
                     x, out, dim, eps);
}

void RmsNormAdd(const float* x, const float* gamma, float* residual,
                std::uint32_t n_tokens, std::uint32_t dim, float eps,
                hipStream_t stream) {
  if (n_tokens == 0)
    return;
  hipLaunchKernelGGL(RmsNormAddKernel, dim3(n_tokens), dim3(256), 0, stream,
                     x, gamma, residual, dim, eps);
}

void ResidualAdd(const float* x, float* residual, std::size_t count,
                 hipStream_t stream) {
  if (count == 0)
    return;
  const std::uint32_t bs = 256;
  hipLaunchKernelGGL(ResidualAddKernel,
                     dim3(static_cast<std::uint32_t>((count + bs - 1) / bs)),
                     dim3(bs), 0, stream, x, residual, count);
}

void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t head_dim, std::uint32_t start_pos, float theta,
          hipStream_t stream) {
  if (n_tokens == 0)
    return;
  hipLaunchKernelGGL(RopeKernel, dim3(n_tokens, heads), dim3(64), 0, stream,
                     x, heads, head_dim, start_pos, theta);
}

void RopeNeox(float* x, std::uint32_t n_tokens, std::uint32_t heads,
              std::uint32_t head_dim, std::uint32_t start_pos, float theta,
              hipStream_t stream) {
  if (n_tokens == 0)
    return;
  hipLaunchKernelGGL(RopeNeoxKernel, dim3(n_tokens, heads), dim3(64), 0,
                     stream, x, heads, head_dim, start_pos, theta);
}

void SiluMul(const float* gate, const float* up, float* out, std::size_t count,
             hipStream_t stream) {
  if (count == 0)
    return;
  const std::uint32_t bs = 256;
  hipLaunchKernelGGL(SiluMulKernel,
                     dim3(static_cast<std::uint32_t>((count + bs - 1) / bs)),
                     dim3(bs), 0, stream, gate, up, out, count);
}

void F32ToF16(const float* in, __half* out, std::size_t count,
              hipStream_t stream) {
  if (count == 0)
    return;
  const std::uint32_t bs = 256;
  hipLaunchKernelGGL(F32ToF16Kernel,
                     dim3(static_cast<std::uint32_t>((count + bs - 1) / bs)),
                     dim3(bs), 0, stream, in, out, count);
}

void StoreKvF16(const float* src, __half* cache, std::uint32_t n_tokens,
                std::uint32_t dim, std::uint32_t start_pos,
                hipStream_t stream) {
  if (n_tokens == 0)
    return;
  const std::uint32_t waves = (dim + 255) / 256;
  hipLaunchKernelGGL(StoreKvF16Kernel, dim3(n_tokens, waves), dim3(256), 0,
                     stream, src, cache, dim, start_pos);
}

void AttentionF16(const float* q, const __half* k_cache,
                  const __half* v_cache, float* out, std::uint32_t n_tokens,
                  std::uint32_t start_pos, std::uint32_t heads,
                  std::uint32_t kv_heads, std::uint32_t d, hipStream_t stream,
                  float score_scale, std::uint32_t window) {
  if (n_tokens == 0)
    return;
  // One 32-thread wavefront per (head, token); see the kernel contract.
  hipLaunchKernelGGL(AttentionF16Kernel, dim3(heads, n_tokens), dim3(32), 0,
                     stream, q, k_cache, v_cache, out, n_tokens, start_pos,
                     heads, kv_heads, d, score_scale, window);
}

__global__ void SigmoidMulKernel(const float* __restrict__ gate,
                                const float* __restrict__ x,
                                float* __restrict__ out, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (i >= count)
    return;
  const float g = gate[i];
  out[i] = (1.0f / (1.0f + expf(-g))) * x[i];
}

void SigmoidMul(const float* gate, const float* x, float* out,
                std::size_t count, hipStream_t stream) {
  if (count == 0)
    return;
  const std::uint32_t bs = 256;
  hipLaunchKernelGGL(SigmoidMulKernel,
                     dim3(static_cast<std::uint32_t>((count + bs - 1) / bs)),
                     dim3(bs), 0, stream, gate, x, out, count);
}

__global__ void SoftCapKernel(float* x, float cap, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (i >= count)
    return;
  const float v = x[i] / cap;
  // tanh via exp (no libm dependency issues on device).
  const float e2 = expf(2.0f * (v > 0 ? v : -v));
  const float t = (e2 - 1.0f) / (e2 + 1.0f);
  x[i] = (v < 0.0f ? -t : t) * cap;
}

void SoftCap(float* x, float cap, std::size_t count, hipStream_t stream) {
  if (count == 0 || cap <= 0.0f)
    return;
  const std::uint32_t bs = 256;
  hipLaunchKernelGGL(SoftCapKernel,
                     dim3(static_cast<std::uint32_t>((count + bs - 1) / bs)),
                     dim3(bs), 0, stream, x, cap, count);
}

__global__ void ScaleKernel(float* x, float s, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (i >= count)
    return;
  x[i] *= s;
}

void ScaleInplace(float* x, float s, std::size_t count, hipStream_t stream) {
  if (count == 0)
    return;
  const std::uint32_t bs = 256;
  hipLaunchKernelGGL(ScaleKernel,
                     dim3(static_cast<std::uint32_t>((count + bs - 1) / bs)),
                     dim3(bs), 0, stream, x, s, count);
}

}  // namespace gufo::models::glimmer::kernels
