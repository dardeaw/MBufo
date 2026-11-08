// GPT-OSS kernels: simple, obviously-correct implementations mirroring the
// llama model-local kernel structure. NeoX YaRN RoPE, sink attention with
// an optional sliding window, and the SwiGLU-OAI expert activation follow
// the llama.cpp openai-moe math exactly (f32, same operation order).
#include "src/models/gptoss/kernels/kernels.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace gufo::models::gptoss::kernels {
namespace {

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
  __shared__ float inv_s;
  if (threadIdx.x == 0)
    inv_s = rsqrtf(total / static_cast<float>(dim) + eps);
  __syncthreads();
  const float inv = inv_s;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    orow[i] = row[i] * inv * gamma[i];
  }
}

__global__ void RopeYarnKernel(float* __restrict__ x, std::uint32_t heads,
                               std::uint32_t head_dim, std::uint32_t start_pos,
                               float theta, float yarn_low, float yarn_high,
                               float yarn_inv_factor, float yarn_mscale) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t i = threadIdx.x;  // 0 .. head_dim/2
  if (i >= head_dim / 2)
    return;
  float* row =
      x + (static_cast<std::size_t>(t) * heads + h) * head_dim;
  const float pos = static_cast<float>(start_pos + t);
  float mult = 1.0f;
  if (yarn_inv_factor != 1.0f) {
    const float width = yarn_high - yarn_low;
    const float span = width > 0.001f ? width : 0.001f;
    float ramp = (static_cast<float>(i) - yarn_low) / span;
    ramp = ramp < 0.0f ? 0.0f : (ramp > 1.0f ? 1.0f : ramp);
    mult = 1.0f - ramp * (1.0f - yarn_inv_factor);
  }
  const float inv_freq =
      mult /
      powf(theta, (2.0f * static_cast<float>(i)) /
                      static_cast<float>(head_dim));
  const float ang = pos * inv_freq;
  const float c = cosf(ang) * yarn_mscale;
  const float s = sinf(ang) * yarn_mscale;
  const float a = row[i];
  const float b = row[i + head_dim / 2];
  row[i] = a * c - b * s;
  row[i + head_dim / 2] = a * s + b * c;
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

// One wavefront (32 threads) per (head, query token); d <= 128 covered with
// d/32 dims per lane. Online softmax starts from the per-head sink mass
// (m = sink, l = 1) so the sink participates exactly as in llama.cpp.
// A nonzero window masks keys older than (pos - window + 1) for causal SWA.
__global__ void AttentionF16SinkKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const float* __restrict__ sinks,
    float* __restrict__ out, std::uint32_t n_tokens, std::uint32_t start_pos,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d,
    std::uint32_t window, float score_scale) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;  // 0 .. 31
  const std::uint32_t kvh = h / (heads / kv_heads);
  const std::uint32_t pos = start_pos + t;
  const std::uint32_t n_kv = pos + 1;
  const std::uint32_t k0 =
      (window != 0 && n_kv > window) ? n_kv - window : 0;
  const std::uint32_t dpl = d <= 32u ? 1u : d / 32u;
  const bool active = (lane * dpl < d);
  const float* qrow = q + (static_cast<std::size_t>(t) * heads + h) * d;
  float m = sinks[h];
  float l = 1.0f;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (std::uint32_t j = k0; j < n_kv; ++j) {
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
    const float full = __shfl_sync(0xffffffffull, dot, 0);
    const float score = full * score_scale;
    float m_new = m;
    if (lane == 0) {
      m_new = m > score ? m : score;
    }
    const float msh = __shfl_sync(0xffffffffull, m_new, 0);
    const float corr = (lane == 0 && l == 0.0f) ? 0.0f : expf(m - msh);
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

__global__ void SwigluOaiBiasKernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    const std::int32_t* __restrict__ ids, const float* __restrict__ gate_b,
    const float* __restrict__ up_b, float* __restrict__ out,
    std::uint32_t slots, std::uint32_t ff) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(slots) * ff;
  if (i >= count)
    return;
  const std::uint32_t s = static_cast<std::uint32_t>(i / ff);
  const std::uint32_t j = static_cast<std::uint32_t>(i % ff);
  const std::int32_t e = ids[s];
  if (e < 0) {
    out[i] = 0.0f;
    return;
  }
  const std::size_t boff = static_cast<std::size_t>(e) * ff + j;
  const float g = gate[i] + gate_b[boff];
  const float u = up[i] + up_b[boff];
  // ggml_swiglu_oai, alpha=1.702, limit=7.
  const float x = g < 7.0f ? g : 7.0f;
  const float y = u < -7.0f ? -7.0f : (u > 7.0f ? 7.0f : u);
  const float gated = x / (1.0f + expf(-1.702f * x));
  out[i] = gated * (y + 1.0f);
}

__global__ void DownBiasAddKernel(float* __restrict__ out,
                                  const std::int32_t* __restrict__ ids,
                                  const float* __restrict__ down_b,
                                  std::uint32_t slots, std::uint32_t hidden) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(slots) * hidden;
  if (i >= count)
    return;
  const std::uint32_t s = static_cast<std::uint32_t>(i / hidden);
  const std::uint32_t j = static_cast<std::uint32_t>(i % hidden);
  const std::int32_t e = ids[s];
  if (e < 0)
    return;
  out[i] += down_b[static_cast<std::size_t>(e) * hidden + j];
}

__global__ void AddRowBiasKernel(float* __restrict__ out,
                                 const float* __restrict__ bias,
                                 std::uint32_t cols) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const std::uint32_t t = static_cast<std::uint32_t>(
      i / cols);
  const std::uint32_t j = static_cast<std::uint32_t>(i % cols);
  out[i] += bias[j];
  (void)t;
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

}  // namespace

void Embed(const void* table, core::GgmlType type, const std::int32_t* ids,
           float* out, std::uint32_t n_tokens, std::uint32_t hidden,
           hipStream_t stream) {
  if (n_tokens == 0)
    return;
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

void ResidualAdd(const float* x, float* residual, std::size_t count,
                 hipStream_t stream) {
  if (count == 0)
    return;
  const std::uint32_t bs = 256;
  hipLaunchKernelGGL(ResidualAddKernel,
                     dim3(static_cast<std::uint32_t>((count + bs - 1) / bs)),
                     dim3(bs), 0, stream, x, residual, count);
}

void AddRowBias(float* out, const float* bias, std::uint32_t n_rows,
                std::uint32_t cols, hipStream_t stream) {
  if (n_rows == 0 || cols == 0)
    return;
  const std::size_t count = static_cast<std::size_t>(n_rows) * cols;
  const std::uint32_t bs = 256;
  hipLaunchKernelGGL(AddRowBiasKernel,
                     dim3(static_cast<std::uint32_t>((count + bs - 1) / bs)),
                     dim3(bs), 0, stream, out, bias, cols);
}

void RopeYarn(float* x, std::uint32_t n_tokens, std::uint32_t heads,
              std::uint32_t head_dim, std::uint32_t start_pos, float theta,
              float yarn_low, float yarn_high, float yarn_inv_factor,
              float yarn_mscale, hipStream_t stream) {
  if (n_tokens == 0)
    return;
  hipLaunchKernelGGL(RopeYarnKernel, dim3(n_tokens, heads), dim3(64), 0,
                     stream, x, heads, head_dim, start_pos, theta, yarn_low,
                     yarn_high, yarn_inv_factor, yarn_mscale);
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

void AttentionF16Sink(const float* q, const __half* k_cache,
                      const __half* v_cache, const float* sinks, float* out,
                      std::uint32_t n_tokens, std::uint32_t start_pos,
                      std::uint32_t heads, std::uint32_t kv_heads,
                      std::uint32_t d, std::uint32_t window,
                      hipStream_t stream, float score_scale) {
  if (n_tokens == 0)
    return;
  hipLaunchKernelGGL(AttentionF16SinkKernel, dim3(heads, n_tokens), dim3(32),
                     0, stream, q, k_cache, v_cache, sinks, out, n_tokens,
                     start_pos, heads, kv_heads, d, window, score_scale);
}

void SwigluOaiBias(const float* gate, const float* up,
                   const std::int32_t* ids, const float* gate_b,
                   const float* up_b, float* out, std::uint32_t slots,
                   std::uint32_t ff, hipStream_t stream) {
  if (slots == 0 || ff == 0)
    return;
  const std::size_t count = static_cast<std::size_t>(slots) * ff;
  const std::uint32_t bs = 256;
  hipLaunchKernelGGL(SwigluOaiBiasKernel,
                     dim3(static_cast<std::uint32_t>((count + bs - 1) / bs)),
                     dim3(bs), 0, stream, gate, up, ids, gate_b, up_b, out,
                     slots, ff);
}

void DownBiasAdd(float* out, const std::int32_t* ids, const float* down_b,
                 std::uint32_t slots, std::uint32_t hidden,
                 hipStream_t stream) {
  if (slots == 0 || hidden == 0)
    return;
  const std::size_t count = static_cast<std::size_t>(slots) * hidden;
  const std::uint32_t bs = 256;
  hipLaunchKernelGGL(DownBiasAddKernel,
                     dim3(static_cast<std::uint32_t>((count + bs - 1) / bs)),
                     dim3(bs), 0, stream, out, ids, down_b, slots, hidden);
}

__global__ void MoeCompactSplitBiasActKernel(
    const float* __restrict__ gateup, const std::int32_t* __restrict__ rows_token,
    const std::int32_t* __restrict__ rows_expert,
    const float* __restrict__ gate_b, const float* __restrict__ up_b,
    __half* __restrict__ act, std::uint32_t rows, std::uint32_t ff) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(rows) * ff;
  if (i >= count)
    return;
  const std::uint32_t c = static_cast<std::uint32_t>(i / ff);
  const std::uint32_t j = static_cast<std::uint32_t>(i % ff);
  if (rows_token[c] < 0) {
    act[i] = __float2half(0.0F);
    return;
  }
  const std::int32_t e = rows_expert[c];
  const std::size_t boff = static_cast<std::size_t>(e) * ff + j;
  const float g = gateup[static_cast<std::size_t>(c) * 2 * ff + j] + gate_b[boff];
  const float u =
      gateup[static_cast<std::size_t>(c) * 2 * ff + ff + j] + up_b[boff];
  // ggml_swiglu_oai, alpha=1.702, limit=7 (same as SwigluOaiBiasKernel).
  const float x = g < 7.0F ? g : 7.0F;
  const float y = u < -7.0F ? -7.0F : (u > 7.0F ? 7.0F : u);
  const float gated = x / (1.0F + expf(-1.702F * x));
  act[i] = __float2half(gated * (y + 1.0F));
}

void MoeCompactSplitBiasAct(const float* gateup,
                            const std::int32_t* rows_token,
                            const std::int32_t* rows_expert,
                            const float* gate_b, const float* up_b,
                            __half* act, std::uint32_t rows, std::uint32_t ff,
                            hipStream_t stream) {
  if (rows == 0 || ff == 0)
    return;
  const std::size_t count = static_cast<std::size_t>(rows) * ff;
  const std::uint32_t bs = 256;
  hipLaunchKernelGGL(MoeCompactSplitBiasActKernel,
                     dim3(static_cast<std::uint32_t>((count + bs - 1) / bs)),
                     dim3(bs), 0, stream, gateup, rows_token, rows_expert,
                     gate_b, up_b, act, rows, ff);
}

}  // namespace gufo::models::gptoss::kernels
