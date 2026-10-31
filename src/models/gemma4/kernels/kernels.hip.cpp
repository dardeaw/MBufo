#include "src/models/gemma4/kernels/kernels.hpp"
#include "src/models/gemma4/weights.hpp"

#include <cmath>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>

namespace gufo::models::gemma4 {
namespace kernels {
namespace {
// NOTE: host wrappers for kernels defined in this file live after the
// anonymous namespace closes at the `}  // namespace` marker.

constexpr std::uint32_t kThreads = 256;

[[nodiscard]] inline std::uint32_t Blocks(std::size_t n) {
  const std::size_t b = (n + kThreads - 1) / kThreads;
  return static_cast<std::uint32_t>(b);
}

[[nodiscard]] __device__ __forceinline__ float GeluTanhOne(float x) {
  // Exact tanh approximation used by Gemma (0.7978845608 = sqrt(2/pi)).
  const float inner = 0.7978845608F * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + tanhf(inner));
}

__global__ void GeluTanhSwigluKernel(float* gate, const float* up,
                                     std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       i < count; i += stride) {
    gate[i] = GeluTanhOne(gate[i]) * up[i];
  }
}

__global__ void GeluTanhKernel(float* x, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       i < count; i += stride) {
    x[i] = GeluTanhOne(x[i]);
  }
}

__global__ void SoftCapKernel(float* x, float cap, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       i < count; i += stride) {
    x[i] = tanhf(x[i] / cap) * cap;
  }
}

__global__ void AddInplaceKernel(float* out, const float* add,
                                 std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       i < count; i += stride) {
    out[i] += add[i];
  }
}

__global__ void ScaleInplaceKernel(float* out, float scale,
                                   std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       i < count; i += stride) {
    out[i] *= scale;
  }
}

__global__ void LayerScaleKernel(float* out, float scale, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       i < count; i += stride) {
    out[i] *= scale;
  }
}

__global__ void RmsNormKernel(const float* x, const float* gamma, float* out,
                              std::uint32_t group_dim, std::uint32_t groups,
                              float eps) {
  const std::uint32_t row = blockIdx.x / groups;
  const std::uint32_t group = blockIdx.x % groups;
  const float* src = x + static_cast<std::size_t>(row) * group_dim * groups +
                     static_cast<std::size_t>(group) * group_dim;
  const float* g = gamma == nullptr
                       ? nullptr
                       : gamma + static_cast<std::size_t>(group) * group_dim;
  float* dst = out + static_cast<std::size_t>(row) * group_dim * groups +
               static_cast<std::size_t>(group) * group_dim;
  __shared__ float shared[kThreads / 32];
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
    const float v = src[i];
    sum += v * v;
  }
  // Block reduce.
  for (std::uint32_t off = 16; off > 0; off >>= 1) {
    sum += __shfl_down_sync(0xffffffffull, sum, off);
  }
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : 0.0F;
    for (std::uint32_t off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffull, v, off);
    }
    if (threadIdx.x == 0) shared[0] = v;
  }
  __syncthreads();
  const float scale = rsqrtf(shared[0] / static_cast<float>(group_dim) + eps);
  for (std::uint32_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
    dst[i] = src[i] * scale * (g != nullptr ? g[i] : 1.0F);
  }
}

__global__ void RmsNormMulKernel(const float* x, const float* gamma,
                                 const float* mul, float* out,
                                 std::uint32_t dim, float eps) {
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * dim;
  const float* src = x + base;
  float* dst = out + base;
  __shared__ float shared[kThreads / 32];
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = src[i];
    sum += v * v;
  }
  for (std::uint32_t off = 16; off > 0; off >>= 1) {
    sum += __shfl_down_sync(0xffffffffull, sum, off);
  }
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : 0.0F;
    for (std::uint32_t off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffull, v, off);
    }
    if (threadIdx.x == 0) shared[0] = v;
  }
  __syncthreads();
  const float scale = rsqrtf(shared[0] / static_cast<float>(dim) + eps);
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float g = gamma != nullptr ? gamma[i] : 1.0F;
    dst[i] = src[i] * scale * g * mul[i];
  }
}





// Fused decode GEMV: y[m] = dot(dequant(W row m), x), one block per row.
// Avoids the DequantSlab + F32ToF16 + hipBLAS round-trips for n == 1.
__global__ void GemvQKernel(const void* w, int type, std::uint64_t row_bytes,
                           std::uint64_t cols, const float* x, float* y) {
  const std::uint32_t row = blockIdx.x;
  const auto* base = static_cast<const std::uint8_t*>(w) + (std::size_t)row * row_bytes;
  float sum = 0.0F;
  if (type == 0) {  // F32
    const auto* r = reinterpret_cast<const float*>(base);
    for (std::uint64_t i = threadIdx.x; i < cols; i += blockDim.x)
      sum += r[i] * x[i];
  } else if (type == 1) {  // F16
    const auto* r = reinterpret_cast<const __half*>(base);
    for (std::uint64_t i = threadIdx.x; i < cols; i += blockDim.x)
      sum += __half2float(r[i]) * x[i];
  } else if (type == 8) {  // Q8_0
    for (std::uint64_t b = threadIdx.x; b < cols / 32; b += blockDim.x) {
      const std::uint8_t* blk = base + (std::size_t)34 * b;
      const float d = __half2float(*reinterpret_cast<const __half*>(blk));
      const auto* qs = reinterpret_cast<const std::int8_t*>(blk + 2);
      const std::size_t off = (std::size_t)b * 32;
      #pragma unroll
      for (int q = 0; q < 32; ++q)
        sum += d * static_cast<float>(qs[q]) * x[off + q];
    }
  } else {  // Q4_0 (type 2): lo nibble = q, hi nibble = q+16
    for (std::uint64_t b = threadIdx.x; b < cols / 32; b += blockDim.x) {
      const std::uint8_t* blk = base + (std::size_t)18 * b;
      const float d = __half2float(*reinterpret_cast<const __half*>(blk));
      const std::uint8_t* qs = blk + 2;
      const std::size_t off = (std::size_t)b * 32;
      #pragma unroll
      for (int q = 0; q < 16; ++q) {
        const float lo = d * (static_cast<float>(qs[q] & 0xF) - 8.0F);
        const float hi = d * (static_cast<float>(qs[q] >> 4) - 8.0F);
        sum += lo * x[off + q] + hi * x[off + q + 16];
      }
    }
  }
  __shared__ float red[256];
  red[threadIdx.x] = sum;
  __syncthreads();
  for (std::uint32_t s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) y[row] = red[0];
}

__global__ void GemvF16Kernel(const __half* w, std::uint64_t cols,
                             const float* x, float* y) {
  const std::uint32_t row = blockIdx.x;
  const __half* r = w + (std::size_t)row * cols;
  float sum = 0.0F;
  for (std::uint64_t i = threadIdx.x; i < cols; i += blockDim.x)
    sum += __half2float(r[i]) * x[i];
  __shared__ float red[256];
  red[threadIdx.x] = sum;
  __syncthreads();
  for (std::uint32_t s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) y[row] = red[0];
}


__global__ void F32ToF16Kernel(const float* in, __half* out,
                               std::size_t count) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    out[i] = __float2half_rn(in[i]);
  }
}



__global__ void RopeKernel(float* x, std::uint32_t heads, std::uint32_t d,
                           std::uint32_t rotary_dim, std::uint32_t position,
                           float theta) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t my_pos = position + t;
  float* base = x + static_cast<std::size_t>(t) * heads * d;
  const std::uint32_t half = rotary_dim / 2;
  for (std::uint32_t h = 0; h < heads; ++h) {
    float* row = base + static_cast<std::size_t>(h) * d;
    for (std::uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
      const float freq =
          1.0F / powf(theta, static_cast<float>(i) / static_cast<float>(half));
      const float angle = static_cast<float>(my_pos) * freq;
      float s, c;
      sincosf(angle, &s, &c);
      const float x0 = row[i];
      const float x1 = row[i + half];
      row[i] = x0 * c - x1 * s;
      row[i + half] = x0 * s + x1 * c;
    }
  }
}

__global__ void StoreKvKernel(const float* src, __half* cache,
                              std::uint32_t row_dim, std::uint32_t position) {
  const std::uint32_t t = blockIdx.x;
  const float* in = src + static_cast<std::size_t>(t) * row_dim;
  __half* out = cache + static_cast<std::size_t>(position + t) * row_dim;
  for (std::uint32_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
    out[i] = __float2half(in[i]);
  }
}

// GQA attention: one block per (token, kv-head) computes ALL q-heads of the
// group while streaming each K/V row once. The per-q-head AttentionKernel
// re-reads the same KV rows heads/kv_heads times (8x on full layers with
// kv_heads=2). Math per (t,h) is identical to AttentionKernel (same fp32
// online-softmax recurrence), so results match bit-for-bit.
template <std::uint32_t MaxD>
__global__ void AttentionGqaKernel(
    const float* q, const __half* k_cache, const __half* v_cache, float* out,
    std::uint32_t start_pos, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t d, std::uint32_t kv_stride, std::uint32_t window,
    std::uint32_t kv_len) {
  constexpr std::uint32_t MaxQ = 8;
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t kv_h = blockIdx.y;
  const std::uint32_t qpg = heads / kv_heads;
  const std::uint32_t pos = start_pos + t;
  const float* qbase = q + static_cast<std::size_t>(t) * heads * d +
                       static_cast<std::size_t>(kv_h) * qpg * d;
  __shared__ float qsm[MaxQ][MaxD];
  for (std::uint32_t qq = 0; qq < qpg; ++qq) {
    const float* qrow = qbase + static_cast<std::size_t>(qq) * d;
    for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x)
      qsm[qq][i] = qrow[i];
  }
  __syncthreads();
  const std::uint32_t lo =
      window > 0 && pos + 1 > window ? pos + 1 - window : 0;
  __shared__ float mls[MaxQ][2];
  __shared__ float acc[MaxQ][MaxD];
  for (std::uint32_t qq = 0; qq < qpg; ++qq) {
    if (threadIdx.x < 2)
      mls[qq][threadIdx.x] = threadIdx.x == 0 ? -INFINITY : 0.0F;
    for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x)
      acc[qq][i] = 0.0F;
  }
  __syncthreads();
  __shared__ float red[8][MaxQ];
  __shared__ float cw[MaxQ][2];
  for (std::uint32_t pp = lo; pp <= pos && pp < kv_len; ++pp) {
    const __half* krow =
        k_cache + static_cast<std::size_t>(pp) * kv_stride + kv_h * d;
    float dot[MaxQ];
    for (std::uint32_t qq = 0; qq < qpg; ++qq) dot[qq] = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
      const float kv = __half2float(krow[i]);
      for (std::uint32_t qq = 0; qq < qpg; ++qq)
        dot[qq] += qsm[qq][i] * kv;
    }
    for (std::uint32_t off = 16; off > 0; off >>= 1)
      for (std::uint32_t qq = 0; qq < qpg; ++qq)
        dot[qq] += __shfl_down_sync(0xffffffffull, dot[qq], off);
    if ((threadIdx.x & 31) == 0) {
      const std::uint32_t w = threadIdx.x / 32;
      for (std::uint32_t qq = 0; qq < qpg; ++qq) red[w][qq] = dot[qq];
    }
    __syncthreads();
    if (threadIdx.x < MaxQ) {
      const std::uint32_t qq = threadIdx.x;
      if (qq < qpg) {
        float sc = 0.0F;
        for (std::uint32_t w = 0; w < blockDim.x / 32; ++w) sc += red[w][qq];
        const float m_new = fmaxf(mls[qq][0], sc);
        const float corr = __expf(mls[qq][0] - m_new);
        const float wgt = __expf(sc - m_new);
        mls[qq][0] = m_new;
        mls[qq][1] = mls[qq][1] * corr + wgt;
        cw[qq][0] = corr;
        cw[qq][1] = wgt;
      }
    }
    __syncthreads();
    const __half* vrow =
        v_cache + static_cast<std::size_t>(pp) * kv_stride + kv_h * d;
    for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
      const float vv = __half2float(vrow[i]);
      for (std::uint32_t qq = 0; qq < qpg; ++qq)
        acc[qq][i] = acc[qq][i] * cw[qq][0] + cw[qq][1] * vv;
    }
    __syncthreads();
  }
  for (std::uint32_t qq = 0; qq < qpg; ++qq) {
    float* orow = out + (static_cast<std::size_t>(t) * heads +
                         static_cast<std::size_t>(kv_h) * qpg + qq) *
                            d;
    const float linv = 1.0F / mls[qq][1];
    for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x)
      orow[i] = acc[qq][i] * linv;
  }
}

__global__ void RouterInputKernel(const float* attn_out,
                                  const float* router_scale, float* tmp,
                                  std::uint32_t hidden, float eps) {
  const std::uint32_t t = blockIdx.x;
  const float* src = attn_out + static_cast<std::size_t>(t) * hidden;
  float* dst = tmp + static_cast<std::size_t>(t) * hidden;
  __shared__ float shared[kThreads / 32];
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = src[i];
    sum += v * v;
  }
  for (std::uint32_t off = 16; off > 0; off >>= 1) {
    sum += __shfl_down_sync(0xffffffffull, sum, off);
  }
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : 0.0F;
    for (std::uint32_t off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffull, v, off);
    }
    if (threadIdx.x == 0) shared[0] = v;
  }
  __syncthreads();
  const float inv = rsqrtf(shared[0] / static_cast<float>(hidden) + eps) *
                    rsqrtf(static_cast<float>(hidden));
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst[i] = src[i] * inv * router_scale[i];
  }
}

__global__ void MoeEpilogueKernel(const float* expert_out,
                                  const float* weights, float* out,
                                  std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  float* dst = out + static_cast<std::size_t>(t) * dim;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t s = 0; s < k; ++s) {
      sum += weights[static_cast<std::size_t>(t) * k + s] *
             expert_out[(static_cast<std::size_t>(t) * k + s) * dim + i];
    }
    dst[i] = sum;
  }
}

}  // namespace

__global__ void EmbedKernel(const std::uint8_t* __restrict__ table,
                            int type, const std::int32_t* __restrict__ tokens,
                            float* __restrict__ res, std::uint32_t hidden) {
  const std::uint32_t t = blockIdx.x;
  const std::size_t row_bytes =
      type == 0 ? static_cast<std::size_t>(hidden) * 4        // F32
                : type == 1 ? static_cast<std::size_t>(hidden) * 2   // F16
                : type == 8 ? static_cast<std::size_t>(hidden) / 32 * 34  // Q8_0
                : static_cast<std::size_t>(hidden) / 32 * 18;        // Q4_0
  const std::uint8_t* row = table + row_bytes * tokens[t];
  const std::size_t base = static_cast<std::size_t>(t) * hidden;
  if (type == 0) {
    const auto* src = reinterpret_cast<const float*>(row);
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
      res[base + i] = src[i];
    }
    return;
  }
  if (type == 1) {
    const auto* src = reinterpret_cast<const __half*>(row);
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
      res[base + i] = __half2float(src[i]);
    }
    return;
  }
  if (type == 8) {  // Q8_0: f16 scale (2B) + 32 int8
    for (std::uint32_t b = threadIdx.x; b < hidden / 32; b += blockDim.x) {
      const std::uint8_t* blk = row + static_cast<std::size_t>(34) * b;
      const float d = __half2float(*reinterpret_cast<const __half*>(blk));
      const std::size_t off = base + static_cast<std::size_t>(b) * 32;
      for (int q = 0; q < 32; ++q) {
        res[off + q] = d * static_cast<float>(static_cast<std::int8_t>(blk[2 + q]));
      }
    }
    return;
  }
  for (std::uint32_t b = threadIdx.x; b < hidden / 32; b += blockDim.x) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(18) * b;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const std::uint8_t* qs = blk + 2;
    const std::size_t off = base + static_cast<std::size_t>(b) * 32;
    for (int q = 0; q < 16; ++q) {
      const int lo = qs[q] & 0xF;
      const int hi = qs[q] >> 4;
      res[off + q] = d * static_cast<float>(lo - 8);
      res[off + q + 16] = d * static_cast<float>(hi - 8);
    }
  }
}

__global__ void DebugRowKernel(const std::uint8_t* table,
                               std::uint32_t row_bytes, std::uint32_t token,
                               float* out, std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = static_cast<float>(table[row_bytes * token + i]);
}

void DebugRow(const void* table, std::uint32_t row_bytes,
              std::uint32_t token, float* d_out, std::uint32_t n,
              hipStream_t stream) {
  hipLaunchKernelGGL(DebugRowKernel, dim3((n + 255) / 256), dim3(256), 0,
                     stream, static_cast<const std::uint8_t*>(table),
                     row_bytes, token, d_out, n);
}

static std::int32_t* g_d_tokens = nullptr;
static std::size_t g_d_tokens_cap = 0;

__global__ void EmbedScaleKernel(const std::uint8_t* __restrict__ table,
                                 int type, const std::int32_t* __restrict__ tokens,
                                 float* __restrict__ res, std::uint32_t hidden,
                                 float scale) {
  const std::uint32_t t = blockIdx.x;
  const std::size_t row_bytes =
      type == 0 ? static_cast<std::size_t>(hidden) * 4
                : type == 1 ? static_cast<std::size_t>(hidden) * 2
                : type == 8 ? static_cast<std::size_t>(hidden) / 32 * 34
                            : static_cast<std::size_t>(hidden) / 32 * 18;
  const std::uint8_t* row = table + row_bytes * tokens[t];
  const std::size_t base = static_cast<std::size_t>(t) * hidden;
  if (type == 0) {
    const auto* src = reinterpret_cast<const float*>(row);
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
      res[base + i] = src[i] * scale;
    return;
  }
  if (type == 1) {
    const auto* src = reinterpret_cast<const __half*>(row);
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x)
      res[base + i] = __half2float(src[i]) * scale;
    return;
  }
  if (type == 8) {
    for (std::uint32_t b = threadIdx.x; b < hidden / 32; b += blockDim.x) {
      const std::uint8_t* blk = row + static_cast<std::size_t>(34) * b;
      const float d = __half2float(*reinterpret_cast<const __half*>(blk)) * scale;
      const std::size_t off = base + static_cast<std::size_t>(b) * 32;
      for (int q = 0; q < 32; ++q)
        res[off + q] = d * static_cast<float>(static_cast<std::int8_t>(blk[2 + q]));
    }
    return;
  }
  for (std::uint32_t b = threadIdx.x; b < hidden / 32; b += blockDim.x) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(18) * b;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk)) * scale;
    const std::uint8_t* qs = blk + 2;
    const std::size_t off = base + static_cast<std::size_t>(b) * 32;
    for (int q = 0; q < 16; ++q) {
      const int lo = qs[q] & 0xF;
      const int hi = qs[q] >> 4;
      res[off + q] = d * static_cast<float>(lo - 8);
      res[off + q + 16] = d * static_cast<float>(hi - 8);
    }
  }
}

void EmbedScale(const void* table, core::GgmlType type,
                const std::int32_t* tokens, float* res, std::uint32_t n_tokens,
                std::uint32_t hidden, float scale, hipStream_t stream) {
  UploadTokens(tokens, n_tokens, stream);
  hipLaunchKernelGGL(EmbedScaleKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, static_cast<const std::uint8_t*>(table),
                     static_cast<int>(type), g_d_tokens, res, hidden, scale);
}

void UploadTokens(const std::int32_t* tokens, std::uint32_t n_tokens,
                  hipStream_t stream) {
  // Persistent device token staging: malloc/free per call races the async
  // kernels that read the ids and churns the allocator. Grow-only.
  if (static_cast<std::size_t>(n_tokens) > g_d_tokens_cap) {
    void* grown = nullptr;
    if (hipMalloc(&grown, sizeof(std::int32_t) * n_tokens) != hipSuccess)
      return;
    g_d_tokens = static_cast<std::int32_t*>(grown);
    g_d_tokens_cap = n_tokens;
  }
  hipMemcpyAsync(g_d_tokens, tokens, sizeof(std::int32_t) * n_tokens,
                 hipMemcpyHostToDevice, stream);
}

std::int32_t* DeviceTokens() { return g_d_tokens; }

void Embed(const void* table, core::GgmlType type,
           const std::int32_t* tokens, float* res, std::uint32_t n_tokens,
           std::uint32_t hidden, hipStream_t stream) {
  UploadTokens(tokens, n_tokens, stream);
  hipLaunchKernelGGL(EmbedKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     static_cast<const std::uint8_t*>(table),
                     static_cast<int>(type), g_d_tokens, res, hidden);
}

bool CopyHostToDevice(void* dst, const void* host, std::size_t bytes) {
  return hipMemcpy(dst, host, bytes, hipMemcpyHostToDevice) == hipSuccess;
}

void DownloadMirror(void* host, const void* device, std::size_t bytes) {
  hipMemcpy(host, device, bytes, hipMemcpyDeviceToHost);
}

void* UploadMirror(const void* host, std::size_t bytes) {
  void* d = nullptr;
  if (hipMalloc(&d, bytes + 64) != hipSuccess)
    return nullptr;
  if (hipMemcpy(d, host, bytes, hipMemcpyHostToDevice) != hipSuccess) {
    hipFree(d);
    return nullptr;
  }
  return d;
}

void RmsNormMul(const float* x, const float* gamma, const float* mul,
                float* out, std::uint32_t n_rows, std::uint32_t dim, float eps,
                hipStream_t stream) {
  hipLaunchKernelGGL(RmsNormMulKernel, dim3(n_rows), dim3(kThreads), 0, stream,
                     x, gamma, mul, out, dim, eps);
}

void F32ToF16(const float* in, __half* out, std::size_t count,
              hipStream_t stream) {
  hipLaunchKernelGGL(F32ToF16Kernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, in, out, count);
}

hipblasStatus_t F16GemmStatus(const __half* w, const __half* x, float* y,
             std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
             hipStream_t stream) {
  // Row-major y[n][m] = x[n][k] * w[m][k]^T maps to the column-major
  // gemm(y_cm[m][n] = w_cm[k][m]^T * x_cm[k][n]).
  static hipblasHandle_t handle = nullptr;
  if (handle == nullptr) {
    hipblasCreate(&handle);
  }
  hipblasSetStream(handle, stream);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const hipblasStatus_t st = hipblasGemmEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(m),
      static_cast<int>(n_tokens), static_cast<int>(k), &alpha, w,
      HIPBLAS_R_16F, static_cast<int>(k), x, HIPBLAS_R_16F,
      static_cast<int>(k), &beta, y, HIPBLAS_R_32F, static_cast<int>(m),
      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  return st;
}

void GeluTanhSwiglu(float* gate, const float* up, std::size_t count,
                    hipStream_t stream) {
  if (count == 0)
    return;
  hipLaunchKernelGGL(GeluTanhSwigluKernel, dim3(Blocks(count)),
                     dim3(kThreads), 0, stream, gate, up, count);
}

// Compact MoE prefill epilogue: act[i] = f16(silu(gate[i]) * up[i]).
// Fuses GeluTanhSwiglu + F32ToF16 into one pass; the F32 silu and multiply
// happen on the same inputs and the single rounding to F16 matches the old
// float->float->half chain bit for bit.
__global__ void SwigluF16Kernel(const float* __restrict__ gate,
                                const float* __restrict__ up,
                                __half* __restrict__ act,
                                std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       i < count; i += stride) {
    act[i] = __float2half_rn(GeluTanhOne(gate[i]) * up[i]);
  }
}

void SwigluF16(const float* gate, const float* up, __half* act,
               std::size_t count, hipStream_t stream) {
  if (count == 0)
    return;
  hipLaunchKernelGGL(SwigluF16Kernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, gate, up, act, count);
}

// Fused split+silu for the Q4-direct grouped MoE prefill.
__global__ void SwigluSplitF16Kernel(const float* __restrict__ combined,
                                     __half* __restrict__ act,
                                     std::uint32_t ff, std::size_t rows) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t count = rows * ff;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       i < count; i += stride) {
    const std::size_t c = i / ff;
    const std::size_t j = i % ff;
    const float g = combined[c * (2 * ff) + j];
    const float u = combined[c * (2 * ff) + ff + j];
    act[i] = __float2half_rn(GeluTanhOne(g) * u);
  }
}

void SwigluSplitF16(const float* combined, __half* act, std::size_t rows,
                    std::uint32_t ff, hipStream_t stream) {
  if (rows == 0 || ff == 0)
    return;
  const std::size_t count = rows * ff;
  hipLaunchKernelGGL(SwigluSplitF16Kernel, dim3(Blocks(count)),
                     dim3(kThreads), 0, stream, combined, act, ff, rows);
}

void GeluTanh(float* x, std::size_t count, hipStream_t stream) {
  if (count == 0)
    return;
  hipLaunchKernelGGL(GeluTanhKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, count);
}

void SoftCap(float* x, float cap, std::size_t count, hipStream_t stream) {
  if (count == 0 || cap <= 0.0F)
    return;
  hipLaunchKernelGGL(SoftCapKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, cap, count);
}

void AddInplace(float* out, const float* add, std::size_t count,
                hipStream_t stream) {
  if (count == 0)
    return;
  hipLaunchKernelGGL(AddInplaceKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, out, add, count);
}

void ScaleInplace(float* out, float scale, std::size_t count,
                  hipStream_t stream) {
  if (count == 0)
    return;
  hipLaunchKernelGGL(ScaleInplaceKernel, dim3(Blocks(count)), dim3(kThreads),
                     0, stream, out, scale, count);
}

__global__ void MulInplaceKernel(float* out, const float* mul,
                                 std::size_t count, std::size_t mul_count) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    out[i] *= mul[i % mul_count];
  }
}

void MulInplace(float* out, const float* mul, std::size_t count,
                std::size_t mul_count, hipStream_t stream) {
  hipLaunchKernelGGL(MulInplaceKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, out, mul, count, mul_count);
}

__global__ void RopeFrequenciesKernel(float* x, std::uint32_t heads,
                                      std::uint32_t d,
                                      std::uint32_t rotary_dim,
                                      std::uint32_t position, float theta,
                                      const float* freq_factors) {
  const std::uint32_t t = blockIdx.x;
  // Per-token position: token t sits at absolute position `position` + t.
  const std::uint32_t my_pos = position + t;
  float* base = x + static_cast<std::size_t>(t) * heads * d;
  const std::uint32_t half = rotary_dim / 2;
  for (std::uint32_t h = 0; h < heads; ++h) {
    float* row = base + static_cast<std::size_t>(h) * d;
    for (std::uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
      const float inv = 1.0F / powf(theta, static_cast<float>(2 * i) /
                                                 static_cast<float>(rotary_dim));
      // ggml rope_yarn semantics: theta/ff, so 1e30 factor freezes the pair.
      const float freq = inv / freq_factors[i];
      const float angle = static_cast<float>(my_pos) * freq;
      float c, s;
      sincosf(angle, &s, &c);
      const float x0 = row[i];
      const float x1 = row[i + half];
      row[i] = x0 * c - x1 * s;
      row[i + half] = x0 * s + x1 * c;
    }
  }
}

void RopeFrequencies(float* x, std::uint32_t n_tokens, std::uint32_t heads,
                     std::uint32_t d, std::uint32_t rotary_dim,
                     const std::uint32_t* start_pos, float theta,
                     const float* freq_factors, hipStream_t stream) {
  hipLaunchKernelGGL(RopeFrequenciesKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, x, heads, d, rotary_dim, *start_pos, theta,
                     freq_factors);
}

void RopeFrequenciesMulti(float* x, std::uint32_t n_tokens,
                          std::uint32_t heads, std::uint32_t d,
                          std::uint32_t rotary_dim, std::uint32_t start_pos,
                          float theta, const float* freq_factors,
                          hipStream_t stream) {
  hipLaunchKernelGGL(RopeFrequenciesKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, x, heads, d, rotary_dim, start_pos, theta,
                     freq_factors);
}

void LayerScale(float* out, float scale, std::size_t count,
                hipStream_t stream) {
  if (count == 0)
    return;
  hipLaunchKernelGGL(LayerScaleKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, out, scale, count);
}

void RmsNorm(const float* x, const float* gamma, float* out,
             std::uint32_t n_rows, std::uint32_t dim, float eps,
             hipStream_t stream) {
  // RmsNormKernel reads and writes the same row in place; a staging buffer
  // is required when x == out to avoid a read/write race across threads.
  static float* stage = nullptr;
  static std::size_t stage_elems = 0;
  const std::size_t need = static_cast<std::size_t>(n_rows) * dim;
  if (x == out && need > stage_elems) {
    // Never free the old stage: a previously launched async RmsNorm kernel
    // may still be reading it on this stream. Grow-only; the largest need
    // (kMaxChunk rows x hidden) is reached early and reused thereafter.
    void* grown = nullptr;
    if (hipMalloc(&grown, need * sizeof(float)) == hipSuccess) {
      stage = static_cast<float*>(grown);
      stage_elems = need;
    }
  }
  if (x == out && stage != nullptr) {
    hipMemcpyAsync(stage, x, need * sizeof(float), hipMemcpyDeviceToDevice,
                   stream);
    hipLaunchKernelGGL(RmsNormKernel, dim3(n_rows), dim3(kThreads), 0, stream,
                       stage, gamma, out, dim, 1, eps);
    return;
  }
  hipLaunchKernelGGL(RmsNormKernel, dim3(n_rows), dim3(kThreads), 0, stream, x,
                     gamma, out, dim, 1, eps);
}

void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          std::uint32_t start_pos, float theta, hipStream_t stream) {
  hipLaunchKernelGGL(RopeKernel, dim3(n_tokens), dim3(kThreads), 0, stream, x,
                     heads, d, rotary_dim, start_pos, theta);
}

void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, std::uint32_t start_pos,
             hipStream_t stream) {
  hipLaunchKernelGGL(StoreKvKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     src, cache, row_dim, start_pos);
}

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               float* out, std::uint32_t n_tokens,
               std::uint32_t start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t d,
               std::uint32_t kv_stride, std::uint32_t window,
               hipStream_t stream) {
  const std::uint32_t kv_len = start_pos + n_tokens;
  // WMMA flash prefill (d=256 sliding-window and d=512 full layers); the
  // scalar GQA kernel is the fallback when the WMMA launch is unsupported.
  // SWA window clipping for the WMMA path only: dense full-scan wastes
  // ~7.8x KV reads at 8k context. Shift k/v base forward by lo and reduce
  // start_pos by lo; the window itself is unchanged and the relative mask
  // comparison is translation-invariant. Scalar fallback keeps raw values.
  const __half* k2 = k_cache;
  const __half* v2 = v_cache;
  std::uint32_t sp2 = start_pos;
  if (window != 0 && start_pos + 1 > window && getenv("GEMMA4_NO_SWA_SHIFT") == nullptr) {
    std::uint32_t lo = (start_pos + 1 - window) & ~3u;
    if (lo != 0 && lo <= start_pos) {
      const std::size_t off = static_cast<std::size_t>(lo) * kv_stride;
      k2 += off;
      v2 += off;
      sp2 -= lo;
    }
  }
  if (getenv("GEMMA4_ATTN_NO_WMMA") == nullptr) {
    if (d <= 256) {
      if (wmma::AttentionWmma(q, k2, v2, out, n_tokens, sp2, d,
                              kv_stride, window, stream))
        return;
    } else if (wmma512::AttentionWmma512(q, k2, v2, out, n_tokens,
                                         sp2, d, kv_stride, window,
                                         stream)) {
      return;
    }
  }
  // Scalar GQA fallback: streams KV once per kv-head group.
  if (d <= 256) {
    hipLaunchKernelGGL((AttentionGqaKernel<256>), dim3(n_tokens, kv_heads),
                       dim3(kThreads), 0, stream, q, k_cache, v_cache, out,
                       start_pos, heads, kv_heads, d, kv_stride, window,
                       kv_len);
    return;
  }
  hipLaunchKernelGGL((AttentionGqaKernel<512>), dim3(n_tokens, kv_heads),
                     dim3(kThreads), 0, stream, q, k_cache, v_cache, out,
                     start_pos, heads, kv_heads, d, kv_stride, window,
                     kv_len);
  // (the per-q-head scalar kernel was removed: GQA covers it)

}

void RouterInput(const float* attn_out, const float* router_scale, float* tmp,
                 std::uint32_t n_tokens, std::uint32_t hidden, float eps,
                 hipStream_t stream) {
  hipLaunchKernelGGL(RouterInputKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, attn_out, router_scale, tmp, hidden, eps);
}

// Gather the rows routed to one expert into a compact batch:
// dst[i] = src[rows_token[i]] for i in [begin,end); padding rows (token -1)
// are zero-filled so the GEMM output stays finite (the scatter skips them).
__global__ void CompactGatherKernel(const float* src, float* dst,
                                    const std::int32_t* rows,
                                    std::uint32_t begin, std::uint32_t end,
                                    std::uint32_t dim) {
  const std::uint32_t i = blockIdx.x;
  if (i < begin || i >= end) return;
  const std::int32_t tok = rows[i];
  const float* s = src + static_cast<std::size_t>(tok < 0 ? 0 : tok) * dim;
  float* d = dst + static_cast<std::size_t>(i) * dim;
  for (std::uint32_t j = threadIdx.x; j < dim; j += blockDim.x)
    d[j] = tok < 0 ? 0.0F : s[j];
}

void CompactGather(const float* src, float* dst, const std::int32_t* rows,
                   std::uint32_t begin, std::uint32_t end, std::uint32_t dim,
                   hipStream_t stream) {
  if (end <= begin) return;
  hipLaunchKernelGGL(CompactGatherKernel, dim3(end), dim3(kThreads), 0, stream,
                     src, dst, rows, begin, end, dim);
}

// Gather + convert in one pass: dst_f16[i] = f16(src[rows_token[i]]),
// padding rows (token -1) zero-filled. The compact MoE path feeds the WMMA
// GEMM directly, skipping the separate F32->F16 staging pass.
__global__ void CompactGatherF16Kernel(const float* src, __half* dst,
                                       const std::int32_t* rows,
                                       std::uint32_t begin, std::uint32_t end,
                                       std::uint32_t dim) {
  const std::uint32_t i = blockIdx.x;
  if (i < begin || i >= end) return;
  const std::int32_t tok = rows[i];
  const float* s = src + static_cast<std::size_t>(tok < 0 ? 0 : tok) * dim;
  __half* d = dst + static_cast<std::size_t>(i) * dim;
  for (std::uint32_t j = threadIdx.x; j < dim; j += blockDim.x)
    d[j] = tok < 0 ? __float2half_rn(0.0F) : __float2half_rn(s[j]);
}

void CompactGatherF16(const float* src, __half* dst, const std::int32_t* rows,
                      std::uint32_t begin, std::uint32_t end,
                      std::uint32_t dim, hipStream_t stream) {
  if (end <= begin) return;
  hipLaunchKernelGGL(CompactGatherF16Kernel, dim3(end), dim3(kThreads), 0,
                     stream, src, dst, rows, begin, end, dim);
}

// Weighted routed accumulate: out[t] += w(t,e) * expert_down[t] for one
// expert, where w(t,e) sums this expert's routing weights over the token's
// slots. Replaces the per-(expert,token) ScaleInplace + AddInplace pair -- the
// dense MoE fallback issued those two launches 512*128 times per layer, which
// showed up as the dominant elementwise cost. Same expression and same expert
// order, so results are bit-identical.
__global__ void MoeRoutedAddKernel(const float* expert_down,
                                   const std::int32_t* ids, const float* wts,
                                   std::uint32_t expert, float* out,
                                   std::uint32_t used, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  float w = 0.0F;
  for (std::uint32_t s = 0; s < used; ++s) {
    const std::size_t i = static_cast<std::size_t>(t) * used + s;
    if (ids[i] == static_cast<std::int32_t>(expert)) w += wts[i];
  }
  if (w == 0.0F) return;  // uniform across the block
  const float* src = expert_down + static_cast<std::size_t>(t) * dim;
  float* dst = out + static_cast<std::size_t>(t) * dim;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
    dst[i] += w * src[i];
}

// Scatter variant of the routed accumulate: out[rows_token[i]] +=
// w(slot) * down[i] over one expert's compact rows. Padding rows are -1 and
// skipped. w comes from the routing slot, so weights match the old path.
__global__ void MoeScatterAddKernel(const float* down,
                                    const std::int32_t* rows_token,
                                    const std::int32_t* rows_slot,
                                    const float* wts, std::uint32_t begin,
                                    std::uint32_t end, float* out,
                                    std::uint32_t dim) {
  const std::uint32_t i = blockIdx.x;
  if (i < begin || i >= end) return;
  const std::int32_t tok = rows_token[i];
  if (tok < 0) return;
  const float w = wts[rows_slot[i]];
  if (w == 0.0F) return;
  // down is the per-expert compact output (row 0 == begin), so index it with
  // the offset within this expert's range, not the global compact index.
  const float* src = down + static_cast<std::size_t>(i - begin) * dim;
  float* dst = out + static_cast<std::size_t>(tok) * dim;
  for (std::uint32_t j = threadIdx.x; j < dim; j += blockDim.x)
    dst[j] += w * src[j];
}

void MoeScatterAdd(const float* down, const std::int32_t* rows_token,
                   const std::int32_t* rows_slot, const float* wts,
                   std::uint32_t begin, std::uint32_t end, float* out,
                   std::uint32_t dim, hipStream_t stream) {
  if (end <= begin) return;
  hipLaunchKernelGGL(MoeScatterAddKernel, dim3(end), dim3(kThreads), 0, stream,
                     down, rows_token, rows_slot, wts, begin, end, out, dim);
}

// Decode-time fused MoE gate/up: one block per (slot, output row). Expert
// comes from the DEVICE ids array (negative = skip). 16-byte vector weight
// loads (one uint4 = one 32-block's nibble payload); fp32 x stays scalar with
// the same per-element products and low-before-high order as GemvQKernel.
__global__ void MoeVecGateUpKernel(const void* __restrict__ gate_up,
                                   std::uint64_t row_bytes,
                                   std::uint64_t expert_stride_bytes,
                                   const float* __restrict__ x,
                                   const std::int32_t* __restrict__ ids,
                                   float* __restrict__ gout,
                                   float* __restrict__ uout,
                                   std::uint32_t used, std::uint32_t ff,
                                   std::uint32_t hidden) {
  const std::uint32_t row = blockIdx.x;  // token*used + slot
  const std::uint32_t r = blockIdx.y;    // 0..2*ff-1 (first ff = gate rows)
  const std::int32_t e = ids[row];
  if (e < 0) return;
  const bool is_up = r >= ff;
  const std::uint32_t out_row = is_up ? r - ff : r;
  const auto* wrow = static_cast<const std::uint8_t*>(gate_up) +
                     static_cast<std::size_t>(e) * expert_stride_bytes +
                     static_cast<std::size_t>(r) * row_bytes;
  const float* xr = x + static_cast<std::size_t>(row / used) * hidden;
  float sum = 0.0F;
  const std::uint64_t nblk = hidden / 32;
  for (std::uint64_t b = threadIdx.x; b < nblk;
       b += static_cast<std::uint64_t>(blockDim.x)) {
    const float d =
        __half2float(*reinterpret_cast<const __half*>(wrow + b * 18));
    const uint4 q = *reinterpret_cast<const uint4*>(wrow + b * 18 + 2);
    const float* xv = xr + b * 32;
    const std::uint32_t qw[4] = {q.x, q.y, q.z, q.w};
    #pragma unroll
    for (int q4 = 0; q4 < 4; ++q4) {
      const std::uint32_t byte0 = qw[q4] & 0xFF;
      const std::uint32_t byte1 = (qw[q4] >> 8) & 0xFF;
      const std::uint32_t byte2 = (qw[q4] >> 16) & 0xFF;
      const std::uint32_t byte3 = (qw[q4] >> 24) & 0xFF;
      const float lo0 = d * (static_cast<float>(byte0 & 0xF) - 8.0F);
      const float hi0 = d * (static_cast<float>(byte0 >> 4) - 8.0F);
      const float lo1 = d * (static_cast<float>(byte1 & 0xF) - 8.0F);
      const float hi1 = d * (static_cast<float>(byte1 >> 4) - 8.0F);
      const float lo2 = d * (static_cast<float>(byte2 & 0xF) - 8.0F);
      const float hi2 = d * (static_cast<float>(byte2 >> 4) - 8.0F);
      const float lo3 = d * (static_cast<float>(byte3 & 0xF) - 8.0F);
      const float hi3 = d * (static_cast<float>(byte3 >> 4) - 8.0F);
      sum += lo0 * xv[q4 * 4 + 0];
      sum += hi0 * xv[q4 * 4 + 16];
      sum += lo1 * xv[q4 * 4 + 1];
      sum += hi1 * xv[q4 * 4 + 17];
      sum += lo2 * xv[q4 * 4 + 2];
      sum += hi2 * xv[q4 * 4 + 18];
      sum += lo3 * xv[q4 * 4 + 3];
      sum += hi3 * xv[q4 * 4 + 19];
    }
  }
  for (std::uint32_t off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffffull, sum, off);
  __shared__ float shared[kThreads / 32];
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : 0.0F;
    for (std::uint32_t off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffull, v, off);
    if (threadIdx.x == 0)
      (is_up ? uout : gout)[static_cast<std::size_t>(row) * ff + out_row] = v;
  }
}

void MoeVecGateUp(const void* gate_up_dev, std::uint64_t row_bytes,
                  std::uint64_t expert_stride_bytes, const float* x,
                  const std::int32_t* ids, float* gout, float* uout,
                  std::uint32_t slots, std::uint32_t used, std::uint32_t ff,
                  std::uint32_t hidden, hipStream_t stream) {
  if (slots == 0) return;
  hipLaunchKernelGGL(MoeVecGateUpKernel, dim3(slots, 2 * ff), dim3(kThreads),
                     0, stream, gate_up_dev, row_bytes, expert_stride_bytes,
                     x, ids, gout, uout, used, ff, hidden);
}

// Decode routed-MoE epilogue: out[t] += sum_s wts[t*used+s] * dscale[ids[..]]
// * slot_down[t*used+s]. One block per token; replaces the per-expert
// ScaleInplace + MoeRoutedAdd pair of the old loop.
__global__ void MoeSlotEpilogueKernel(const float* __restrict__ slot_down,
                                      const std::int32_t* __restrict__ ids,
                                      const float* __restrict__ wts,
                                      const float* __restrict__ dscale,
                                      float* __restrict__ out,
                                      std::uint32_t used,
                                      std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  float* dst = out + static_cast<std::size_t>(t) * dim;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t s = 0; s < used; ++s) {
      const std::uint32_t row = t * used + s;
      const std::int32_t e = ids[row];
      if (e < 0) continue;
      const float w = wts[row];
      if (w == 0.0F) continue;
      const float scale = dscale != nullptr ? dscale[e] : 1.0F;
      sum += w * scale *
             slot_down[static_cast<std::size_t>(row) * dim + i];
    }
    dst[i] += sum;
  }
}

void MoeSlotEpilogue(const float* slot_down, const std::int32_t* ids,
                     const float* wts, const float* dscale, float* out,
                     std::uint32_t n_tokens, std::uint32_t used,
                     std::uint32_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(MoeSlotEpilogueKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, slot_down, ids, wts, dscale, out, used, dim);
}

// Fused Q4_0 down GEMV for the decode MoE: expert row selected per slot from
// the DEVICE ids array (negative -> output 0). uint4 vector weight loads,
// same per-element math as GemvQKernel.
__global__ void GemvQSlotKernel(const void* __restrict__ down,
                                std::uint64_t row_bytes,
                                std::uint64_t expert_stride_bytes,
                                const float* __restrict__ act,
                                const std::int32_t* __restrict__ ids,
                                float* __restrict__ out,
                                std::uint32_t hidden, std::uint32_t ff) {
  const std::uint32_t slot = blockIdx.x;
  const std::uint32_t h = blockIdx.y;
  const std::size_t orow = static_cast<std::size_t>(slot) * hidden + h;
  const std::int32_t e = ids[slot];
  if (e < 0) {
    if (threadIdx.x == 0) out[orow] = 0.0F;
    return;
  }
  const auto* wrow = static_cast<const std::uint8_t*>(down) +
                     static_cast<std::size_t>(e) * expert_stride_bytes +
                     static_cast<std::size_t>(h) * row_bytes;
  const float* ar = act + static_cast<std::size_t>(slot) * ff;
  float sum = 0.0F;
  const std::uint64_t nblk = ff / 32;
  for (std::uint64_t b = threadIdx.x; b < nblk;
       b += static_cast<std::uint64_t>(blockDim.x)) {
    const float d =
        __half2float(*reinterpret_cast<const __half*>(wrow + b * 18));
    const uint4 q = *reinterpret_cast<const uint4*>(wrow + b * 18 + 2);
    const float* xv = ar + b * 32;
    const std::uint32_t qw[4] = {q.x, q.y, q.z, q.w};
    #pragma unroll
    for (int q4 = 0; q4 < 4; ++q4) {
      const std::uint32_t byte0 = qw[q4] & 0xFF;
      const std::uint32_t byte1 = (qw[q4] >> 8) & 0xFF;
      const std::uint32_t byte2 = (qw[q4] >> 16) & 0xFF;
      const std::uint32_t byte3 = (qw[q4] >> 24) & 0xFF;
      const float lo0 = d * (static_cast<float>(byte0 & 0xF) - 8.0F);
      const float hi0 = d * (static_cast<float>(byte0 >> 4) - 8.0F);
      const float lo1 = d * (static_cast<float>(byte1 & 0xF) - 8.0F);
      const float hi1 = d * (static_cast<float>(byte1 >> 4) - 8.0F);
      const float lo2 = d * (static_cast<float>(byte2 & 0xF) - 8.0F);
      const float hi2 = d * (static_cast<float>(byte2 >> 4) - 8.0F);
      const float lo3 = d * (static_cast<float>(byte3 & 0xF) - 8.0F);
      const float hi3 = d * (static_cast<float>(byte3 >> 4) - 8.0F);
      sum += lo0 * xv[q4 * 4 + 0];
      sum += hi0 * xv[q4 * 4 + 16];
      sum += lo1 * xv[q4 * 4 + 1];
      sum += hi1 * xv[q4 * 4 + 17];
      sum += lo2 * xv[q4 * 4 + 2];
      sum += hi2 * xv[q4 * 4 + 18];
      sum += lo3 * xv[q4 * 4 + 3];
      sum += hi3 * xv[q4 * 4 + 19];
    }
  }
  for (std::uint32_t off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffffull, sum, off);
  __shared__ float shared[kThreads / 32];
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : 0.0F;
    for (std::uint32_t off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffull, v, off);
    if (threadIdx.x == 0) out[orow] = v;
  }
}

void GemvQSlot(const void* down_dev, std::uint64_t row_bytes,
               std::uint64_t expert_stride_bytes, const float* act,
               const std::int32_t* ids, float* out, std::uint32_t slots,
               std::uint32_t hidden, std::uint32_t ff, hipStream_t stream) {
  hipLaunchKernelGGL(GemvQSlotKernel, dim3(slots, hidden), dim3(kThreads), 0,
                     stream, down_dev, row_bytes, expert_stride_bytes, act,
                     ids, out, hidden, ff);
}

void MoeRoutedAdd(const float* expert_down, const std::int32_t* ids,
                  const float* wts, std::uint32_t expert, float* out,
                  std::uint32_t n_tokens, std::uint32_t used,
                  std::uint32_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(MoeRoutedAddKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, expert_down, ids, wts, expert, out, used, dim);
}

void MoeEpilogue(const float* expert_out, const float* weights, float* out,
                 std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                 hipStream_t stream) {
  hipLaunchKernelGGL(MoeEpilogueKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, expert_out, weights, out, k, dim);
}


/// Dequantize one row of a GGML quant tensor into an F16 staging row.
/// Supports F32/F16/BF16/Q4_0/Q8_0 (the gemma4 dense formats).
__global__ void DequantRowKernel(const void* w, int type, std::uint64_t cols,
                                 std::uint64_t row_bytes, __half* out) {
  const std::uint64_t row = blockIdx.x;
  const auto* base = static_cast<const std::uint8_t*>(w) + row * row_bytes;
  __half* out_row = out + row * cols;
  for (std::uint64_t i = threadIdx.x; i < cols;
       i += static_cast<std::uint64_t>(blockDim.x)) {
    float v = 0.0F;
    switch (type) {
      case 0:  // F32
        v = reinterpret_cast<const float*>(base)[i];
        break;
      case 1:  // F16
        v = __half2float(reinterpret_cast<const __half*>(base)[i]);
        break;
      case 30: {  // BF16
        const auto bits = reinterpret_cast<const std::uint16_t*>(base)[i];
        v = __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
        break;
      }
      case 2: {  // Q4_0: lo nibbles = first half, hi nibbles = second half
        const std::uint64_t blk = i / 32;
        const std::uint64_t off = i % 32;
        const auto* b =
            reinterpret_cast<const std::uint8_t*>(base + blk * 18);
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const std::uint8_t byte = b[2 + (off % 16)];
        const std::int32_t q =
            ((off < 16 ? (byte & 0xF) : (byte >> 4)) - 8);
        v = d * static_cast<float>(q);
        break;
      }
      case 8: {  // Q8_0: blocks of 32, 2-byte fp16 scale + 32 int8
        const std::uint64_t blk = i / 32;
        const std::uint64_t off = i % 32;
        const auto* b =
            reinterpret_cast<const std::uint8_t*>(base + blk * 34);
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const auto q = static_cast<std::int8_t>(b[2 + off]);
        v = d * static_cast<float>(q);
        break;
      }
      default:
        v = 0.0F;
    }
    out_row[i] = __float2half_rn(v);
  }
}


/// Lossless Q4_0 -> Q8_0 requant: per 32-element block, d8 = d4/16 (f16),
/// q8 = q4*16 (int8). Exact because q4 in [-8,7] => q8 in [-128,112].
/// Reads `rows` rows of `cols` elements (cols % 32 == 0).
// Q4_0 weight row -> F16, fused straight into the GEMM's weight buffer so the
// compact MoE path skips the separate Q8 dequant launch (and its re-read).
// Layout matches the F16 GEMM contract: row-major [cols] halves.
__global__ void DequantQ40ToF16Kernel(const void* __restrict__ w,
                                      std::uint64_t cols,
                                      std::uint64_t row_bytes,
                                      __half* __restrict__ out) {
  const std::uint64_t row = blockIdx.x;
  const auto* base = static_cast<const std::uint8_t*>(w) + row * row_bytes;
  __half* out_row = out + row * cols;
  const std::uint64_t nblk = cols / 32;
  for (std::uint64_t b = threadIdx.x; b < nblk;
       b += static_cast<std::uint64_t>(blockDim.x)) {
    const auto* q4 = reinterpret_cast<const std::uint8_t*>(base + b * 18);
    const float d4 = __half2float(*reinterpret_cast<const __half*>(q4));
    const std::uint8_t* qs = q4 + 2;
    __half* dst = out_row + b * 32;
    for (int off = 0; off < 16; ++off) {
      const std::uint8_t byte = qs[off];
      const int lo = (byte & 0xF) - 8;
      const int hi = (byte >> 4) - 8;
      dst[off] = __float2half_rn(static_cast<float>(lo) * d4);
      dst[off + 16] = __float2half_rn(static_cast<float>(hi) * d4);
    }
  }
}

__global__ void DequantQ40ToQ8Kernel(const void* __restrict__ w,
                                     std::uint64_t cols,
                                     std::uint64_t row_bytes,
                                     std::uint8_t* __restrict__ out) {
  const std::uint64_t row = blockIdx.x;
  const auto* base = static_cast<const std::uint8_t*>(w) + row * row_bytes;
  std::uint8_t* out_row = out + row * (cols / 32) * 34;
  const std::uint64_t nblk = cols / 32;
  for (std::uint64_t b = threadIdx.x; b < nblk;
       b += static_cast<std::uint64_t>(blockDim.x)) {
    const auto* q4 = reinterpret_cast<const std::uint8_t*>(base + b * 18);
    const float d4 = __half2float(*reinterpret_cast<const __half*>(q4));
    // Lossless: d8 = d4/16, q8 = q4*16 (q4 in [-8,7] -> q8 in [-128,112]).
    const __half d8 = __float2half_rn(d4 * 0.0625F);
    *reinterpret_cast<__half*>(out_row + b * 34) = d8;
    std::int8_t* q8 = reinterpret_cast<std::int8_t*>(out_row + b * 34 + 2);
    const std::uint8_t* qs = q4 + 2;
    for (int off = 0; off < 16; ++off) {
      const std::uint8_t byte = qs[off];
      const int lo = (byte & 0xF) - 8;
      const int hi = (byte >> 4) - 8;
      q8[off]      = static_cast<std::int8_t>(lo * 16);
      q8[off + 16] = static_cast<std::int8_t>(hi * 16);
    }
  }
}


void DequantSlab(const TensorRef& t, std::uint64_t rows, __half* dst,
                 hipStream_t stream) {
  const std::uint64_t row_bytes = t.RowBytes();
  const std::uint32_t blocks = static_cast<std::uint32_t>(rows);
  const void* src = t.device != nullptr ? t.device : t.data;
  hipLaunchKernelGGL(DequantRowKernel, dim3(blocks), dim3(256), 0, stream,
                     src, static_cast<int>(t.type), t.cols, row_bytes, dst);
}

void DequantSlabQ8(const TensorRef& t, std::uint64_t rows, std::uint8_t* dst,
                   hipStream_t stream) {
  const std::uint64_t row_bytes = t.RowBytes();
  const void* src = t.device != nullptr ? t.device : t.data;
  hipLaunchKernelGGL(DequantQ40ToQ8Kernel, dim3(static_cast<std::uint32_t>(rows)),
                     dim3(256), 0, stream, src, t.cols, row_bytes, dst);
}

void DequantSlabF16(const TensorRef& t, std::uint64_t rows, __half* dst,
                    hipStream_t stream) {
  const std::uint64_t row_bytes = t.RowBytes();
  const void* src = t.device != nullptr ? t.device : t.data;
  hipLaunchKernelGGL(DequantQ40ToF16Kernel,
                     dim3(static_cast<std::uint32_t>(rows)), dim3(256), 0,
                     stream, src, t.cols, row_bytes, dst);
}


void GemvFused(const void* wdev, int type, std::uint64_t row_bytes,
               std::uint64_t cols, const float* x, float* y, std::uint32_t m,
               hipStream_t stream) {
  if (m == 0) return;
  hipLaunchKernelGGL(GemvQKernel, dim3(m), dim3(kThreads), 0, stream, wdev,
                     type, row_bytes, cols, x, y);
}

void GemvF16Fused(const __half* wdev, std::uint64_t cols, const float* x,
                  float* y, std::uint32_t m, hipStream_t stream) {
  if (m == 0) return;
  hipLaunchKernelGGL(GemvF16Kernel, dim3(m), dim3(kThreads), 0, stream, wdev,
                     cols, x, y);
}

}  // namespace kernels
}  // namespace gufo::models::gemma4
