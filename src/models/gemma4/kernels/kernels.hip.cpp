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
  for (std::uint32_t off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffffull, sum, off);
  __shared__ float shared[kThreads / 32];
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : 0.0F;
    for (std::uint32_t off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffull, v, off);
    if (threadIdx.x == 0) y[row] = v;
  }
}

__global__ void GemvF16Kernel(const __half* w, std::uint64_t cols,
                             const float* x, float* y) {
  const std::uint32_t row = blockIdx.x;
  const __half* r = w + (std::size_t)row * cols;
  float sum = 0.0F;
  for (std::uint64_t i = threadIdx.x; i < cols; i += blockDim.x)
    sum += __half2float(r[i]) * x[i];
  for (std::uint32_t off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffffull, sum, off);
  __shared__ float shared[kThreads / 32];
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : 0.0F;
    for (std::uint32_t off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffull, v, off);
    if (threadIdx.x == 0) y[row] = v;
  }
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

/// One block per (token, head). Walks the KV extent in tiles of 32 keys,
/// online-softmax in registers (d <= 512 staged in LDS).
template <std::uint32_t MaxD>
__global__ void AttentionKernel(const float* q, const __half* k_cache,
                                const __half* v_cache, float* out,
                                std::uint32_t start_pos, std::uint32_t heads,
                                std::uint32_t kv_heads, std::uint32_t d,
                                std::uint32_t kv_stride, std::uint32_t window,
                                std::uint32_t kv_len) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t kv_h = h / (heads / kv_heads);
  const std::uint32_t pos = start_pos + t;
  const float* qrow = q + (static_cast<std::size_t>(t) * heads + h) * d;
  __shared__ float qsm[MaxD];
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
    qsm[i] = qrow[i];
  }
  __syncthreads();
  // llama.cpp gemma4: f_attention_scale = 1.0 (no 1/sqrt(d) scaling).
  const float scale = 1.0F;
  const std::uint32_t lo =
      window > 0 && pos + 1 > window ? pos + 1 - window : 0;
  float m = -INFINITY;
  float l = 0.0F;
  __shared__ float acc[MaxD];
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) acc[i] = 0.0F;
  __syncthreads();
  for (std::uint32_t p = lo; p <= pos && p < kv_len; ++p) {
    const __half* krow =
        k_cache + static_cast<std::size_t>(p) * kv_stride + kv_h * d;
    float dot = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
      dot += qsm[i] * __half2float(krow[i]);
    }
    for (std::uint32_t off = 16; off > 0; off >>= 1) {
      dot += __shfl_down_sync(0xffffffffull, dot, off);
    }
    __shared__ float warp[8];
    if ((threadIdx.x & 31) == 0) warp[threadIdx.x / 32] = dot;
    __syncthreads();
    if (threadIdx.x == 0) {
      float s = 0.0F;
      for (std::uint32_t w = 0; w < blockDim.x / 32; ++w) s += warp[w];
      warp[0] = s;
    }
    __syncthreads();
    const float score = warp[0] * scale;
    const float m_new = fmaxf(m, score);
    const float corr = __expf(m - m_new);
    const float w = __expf(score - m_new);
    m = m_new;
    l = l * corr + w;
    const __half* vrow =
        v_cache + static_cast<std::size_t>(p) * kv_stride + kv_h * d;
    for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
      acc[i] = acc[i] * corr + w * __half2float(vrow[i]);
    }
    __syncthreads();
  }
  float* orow = out + (static_cast<std::size_t>(t) * heads + h) * d;
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
    orow[i] = acc[i] / l;
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

void Embed(const void* table, core::GgmlType type,
           const std::int32_t* tokens, float* res, std::uint32_t n_tokens,
           std::uint32_t hidden, hipStream_t stream) {
  std::int32_t* d_tokens = nullptr;
  hipMalloc(&d_tokens, sizeof(std::int32_t) * n_tokens);
  hipMemcpyAsync(d_tokens, tokens, sizeof(std::int32_t) * n_tokens,
                 hipMemcpyHostToDevice, stream);
  hipLaunchKernelGGL(EmbedKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     static_cast<const std::uint8_t*>(table),
                     static_cast<int>(type), d_tokens, res, hidden);
  hipFree(d_tokens);
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
    if (stage)
      hipFree(stage);
    hipMalloc(&stage, need * sizeof(float));
    stage_elems = need;
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
  if (d <= 256) {
    hipLaunchKernelGGL((AttentionKernel<256>), dim3(n_tokens, heads),
                       dim3(kThreads), 0, stream, q, k_cache, v_cache, out,
                       start_pos, heads, kv_heads, d, kv_stride, window,
                       kv_len);
    return;
  }
  hipLaunchKernelGGL((AttentionKernel<512>), dim3(n_tokens, heads),
                     dim3(kThreads), 0, stream, q, k_cache, v_cache, out,
                     start_pos, heads, kv_heads, d, kv_stride, window,
                     kv_len);
}

void RouterInput(const float* attn_out, const float* router_scale, float* tmp,
                 std::uint32_t n_tokens, std::uint32_t hidden, float eps,
                 hipStream_t stream) {
  hipLaunchKernelGGL(RouterInputKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, attn_out, router_scale, tmp, hidden, eps);
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

void DequantSlab(const TensorRef& t, std::uint64_t rows, __half* dst,
                 hipStream_t stream) {
  const std::uint64_t row_bytes = t.RowBytes();
  const std::uint32_t blocks = static_cast<std::uint32_t>(rows);
  const void* src = t.device != nullptr ? t.device : t.data;
  hipLaunchKernelGGL(DequantRowKernel, dim3(blocks), dim3(256), 0, stream,
                     src, static_cast<int>(t.type), t.cols, row_bytes, dst);
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
