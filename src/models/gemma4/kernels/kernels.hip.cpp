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

__device__ __forceinline__ int XorLaneI(int v, int offset) {
#if defined(__gfx1151__)
  switch (offset) {
    case 1:
      return __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, false);
    case 2:
      return __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, false);
    case 4:
      return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, false);
    case 8:
      return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, false);
    case 16:
      return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false,
                                          false);
  }
#endif
  return __shfl_xor(v, offset);
}

__device__ __forceinline__ float XorLaneF(float v, int offset) {
  return __builtin_bit_cast(float, XorLaneI(__builtin_bit_cast(int, v), offset));
}

__device__ __forceinline__ float WaveSum(float v) {
  v += XorLaneF(v, 16);
  v += XorLaneF(v, 8);
  v += XorLaneF(v, 4);
  v += XorLaneF(v, 2);
  v += XorLaneF(v, 1);
  return v;
}

__global__ void RmsNormAddKernel(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 float* __restrict__ residual_inout,
                                 std::uint32_t dim, float eps,
                                 float post_scale) {
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * dim;
  const float* src = x + base;
  float* dst = residual_inout + base;
  __shared__ float shared[kThreads / 32];
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = src[i];
    sum += v * v;
  }
  sum = WaveSum(sum);
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? shared[threadIdx.x] : 0.0F;
    v = WaveSum(v);
    if (threadIdx.x == 0) shared[0] = v;
  }
  __syncthreads();
  const float scale = rsqrtf(shared[0] / static_cast<float>(dim) + eps);
  if (post_scale == 1.0F) {
    for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
      const float g = gamma != nullptr ? gamma[i] : 1.0F;
      dst[i] += src[i] * scale * g;
    }
  } else {
    for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
      const float g = gamma != nullptr ? gamma[i] : 1.0F;
      dst[i] = (dst[i] + src[i] * scale * g) * post_scale;
    }
  }
}

// Compute partial dot product for one wave lane over [k0, k0 + tile_k).
__device__ __forceinline__ float WaveRowTileDot(
    const std::uint8_t* __restrict__ wrow, int type, std::uint32_t k0,
    std::uint32_t tile_k, std::uint32_t lane, const float* __restrict__ sx) {
  float sum = 0.0F;
  if (type == 2) {  // Q4_0: 8 blocks (256 weights) per 32-lane wave step
    const std::uint32_t nblk = tile_k >> 5;
    const std::uint32_t b_grp = lane >> 2;
    const std::uint32_t sub = lane & 3u;
    const std::uint8_t* wbase = wrow + static_cast<std::size_t>(k0 >> 5) * 18;
    for (std::uint32_t b0 = 0; b0 < nblk; b0 += 8) {
      const std::uint32_t b = b0 + b_grp;
      if (b < nblk) {
        const auto* bp =
            reinterpret_cast<const std::uint16_t*>(wbase + b * 18u);
        const float d = __half2float(*reinterpret_cast<const __half*>(bp));
        const std::uint32_t q01 = bp[1 + 2 * sub];
        const std::uint32_t q23 = bp[2 + 2 * sub];
        const float4 xlo =
            *reinterpret_cast<const float4*>(sx + b * 32u + 4u * sub);
        const float4 xhi =
            *reinterpret_cast<const float4*>(sx + b * 32u + 16u + 4u * sub);
        float acc = 0.0F;
        acc += static_cast<float>(static_cast<int>(q01 & 0xFu) - 8) * xlo.x;
        acc += static_cast<float>(static_cast<int>((q01 >> 4) & 0xFu) - 8) * xhi.x;
        acc += static_cast<float>(static_cast<int>((q01 >> 8) & 0xFu) - 8) * xlo.y;
        acc += static_cast<float>(static_cast<int>((q01 >> 12) & 0xFu) - 8) * xhi.y;
        acc += static_cast<float>(static_cast<int>(q23 & 0xFu) - 8) * xlo.z;
        acc += static_cast<float>(static_cast<int>((q23 >> 4) & 0xFu) - 8) * xhi.z;
        acc += static_cast<float>(static_cast<int>((q23 >> 8) & 0xFu) - 8) * xlo.w;
        acc += static_cast<float>(static_cast<int>((q23 >> 12) & 0xFu) - 8) * xhi.w;
        sum += d * acc;
      }
    }
  } else if (type == 8) {  // Q8_0: 4 blocks (128 weights) per 32-lane wave step
    const std::uint32_t nblk = tile_k >> 5;
    const std::uint32_t b_grp = lane >> 3;
    const std::uint32_t sub = lane & 7u;
    const std::uint8_t* wbase = wrow + static_cast<std::size_t>(k0 >> 5) * 34;
    for (std::uint32_t b0 = 0; b0 < nblk; b0 += 4) {
      const std::uint32_t b = b0 + b_grp;
      if (b < nblk) {
        const auto* bp =
            reinterpret_cast<const std::uint16_t*>(wbase + b * 34u);
        const float d = __half2float(*reinterpret_cast<const __half*>(bp));
        const std::uint16_t q01 = bp[1 + 2 * sub];
        const std::uint16_t q23 = bp[2 + 2 * sub];
        const float4 xv =
            *reinterpret_cast<const float4*>(sx + b * 32u + 4u * sub);
        float acc = 0.0F;
        acc += static_cast<float>(static_cast<std::int8_t>(q01 & 0xFFu)) * xv.x;
        acc += static_cast<float>(static_cast<std::int8_t>(q01 >> 8)) * xv.y;
        acc += static_cast<float>(static_cast<std::int8_t>(q23 & 0xFFu)) * xv.z;
        acc += static_cast<float>(static_cast<std::int8_t>(q23 >> 8)) * xv.w;
        sum += d * acc;
      }
    }
  } else if (type == 1) {  // F16
    const auto* r = reinterpret_cast<const __half*>(wrow) + k0;
    for (std::uint32_t j = lane * 4u; j < tile_k; j += 128u) {
      const uint2 u = *reinterpret_cast<const uint2*>(r + j);
      const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
      const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
      const float4 xv = *reinterpret_cast<const float4*>(sx + j);
      sum += f0.x * xv.x + f0.y * xv.y + f1.x * xv.z + f1.y * xv.w;
    }
  } else if (type == 30) {  // BF16
    const auto* r = reinterpret_cast<const std::uint16_t*>(wrow) + k0;
    for (std::uint32_t j = lane * 4u; j < tile_k; j += 128u) {
      const uint2 u = *reinterpret_cast<const uint2*>(r + j);
      const float w0 = __uint_as_float((u.x & 0xFFFFu) << 16);
      const float w1 = __uint_as_float(u.x & 0xFFFF0000u);
      const float w2 = __uint_as_float((u.y & 0xFFFFu) << 16);
      const float w3 = __uint_as_float(u.y & 0xFFFF0000u);
      const float4 xv = *reinterpret_cast<const float4*>(sx + j);
      sum += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
    }
  } else {  // F32
    const auto* r = reinterpret_cast<const float*>(wrow) + k0;
    for (std::uint32_t j = lane * 4u; j < tile_k; j += 128u) {
      const float4 wv = *reinterpret_cast<const float4*>(r + j);
      const float4 xv = *reinterpret_cast<const float4*>(sx + j);
      sum += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
  }
  return sum;
}

// Wave32 single-token GEMV: 8 warps = 8 output rows per 256-thread block.
struct alignas(8) BlockQ8_1 {
  float d;
  float s;
  std::int32_t qs[8];
};
static_assert(sizeof(BlockQ8_1) == 40, "BlockQ8_1 size mismatch");

__device__ __forceinline__ int Dot4U4S8(int u4_packed, int s8_packed, int acc) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
  return __builtin_amdgcn_sudot4(true, u4_packed, true, s8_packed, acc, false);
#elif defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx942__) || defined(__gfx1030__)
  return __builtin_amdgcn_sdot4(u4_packed, s8_packed, acc, false);
#else
  const auto* pa = reinterpret_cast<const std::int8_t*>(&u4_packed);
  const auto* pb = reinterpret_cast<const std::int8_t*>(&s8_packed);
  return acc + pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2] + pa[3] * pb[3];
#endif
}

__device__ __forceinline__ int Dot4S8S8(int a_packed, int b_packed, int acc) {
  return Dot4U4S8(a_packed, b_packed, acc);
}

__global__ void __launch_bounds__(256)
    QuantizeQ8_1Kernel(const float* __restrict__ x,
                       BlockQ8_1* __restrict__ out,
                       std::uint32_t n_blocks) {
  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t blk = blockIdx.x * 8u + warp_id;
  if (blk >= n_blocks) return;

  const float v = x[static_cast<std::size_t>(blk) * 32u + lane];
  float amax = fabsf(v);
  amax = fmaxf(amax, XorLaneF(amax, 16));
  amax = fmaxf(amax, XorLaneF(amax, 8));
  amax = fmaxf(amax, XorLaneF(amax, 4));
  amax = fmaxf(amax, XorLaneF(amax, 2));
  amax = fmaxf(amax, XorLaneF(amax, 1));

  const float d = amax * (1.0F / 127.0F);
  const float id = (amax > 0.0F) ? (127.0F / amax) : 0.0F;
  const int q = __float2int_rn(v * id);

  const int q_x1 = XorLaneI(q, 1);
  const int q_x2 = XorLaneI(q, 2);
  const int q_x3 = XorLaneI(q_x1, 2);
  const int packed = (q & 0xFF) | ((q_x1 & 0xFF) << 8) |
                     ((q_x2 & 0xFF) << 16) | ((q_x3 & 0xFF) << 24);

  int qsum = q;
  qsum += XorLaneI(qsum, 16);
  qsum += XorLaneI(qsum, 8);
  qsum += XorLaneI(qsum, 4);
  qsum += XorLaneI(qsum, 2);
  qsum += XorLaneI(qsum, 1);

  if ((lane & 3u) == 0u) {
    out[blk].qs[lane >> 2] = packed;
  }
  if (lane == 0u) {
    out[blk].d = d;
    out[blk].s = d * static_cast<float>(qsum);
  }
}

static BlockQ8_1* g_q8_1_scratch = nullptr;
static std::size_t g_q8_1_scratch_cap = 0;

static BlockQ8_1* QuantizeToQ8Scratch(const float* x, std::size_t n_blocks,
                                      hipStream_t stream) {
  if (n_blocks == 0) return nullptr;
  if (n_blocks > g_q8_1_scratch_cap) {
    const std::size_t alloc_blks = n_blocks < 4096u ? 4096u : n_blocks * 2u;
    void* p = nullptr;
    if (hipMalloc(&p, alloc_blks * sizeof(BlockQ8_1)) != hipSuccess) {
      return nullptr;
    }
    if (g_q8_1_scratch != nullptr) (void)hipFree(g_q8_1_scratch);
    g_q8_1_scratch = static_cast<BlockQ8_1*>(p);
    g_q8_1_scratch_cap = alloc_blks;
  }
  const std::uint32_t nb = static_cast<std::uint32_t>(n_blocks);
  hipLaunchKernelGGL(QuantizeQ8_1Kernel, dim3((nb + 7u) / 8u), dim3(256), 0,
                     stream, x, g_q8_1_scratch, nb);
  return g_q8_1_scratch;
}

__global__ void __launch_bounds__(256)
    GemvWave32BatchedQ8Kernel(const std::uint8_t* __restrict__ w,
                              std::uint64_t row_bytes, std::uint32_t total_blk,
                              std::uint32_t tile_blk,
                              const BlockQ8_1* __restrict__ x_q8,
                              float* __restrict__ y, std::uint32_t n_tokens,
                              std::uint32_t m, float softcap) {
  extern __shared__ __align__(8) unsigned char smem_gemv_q8[];
  auto* sx = reinterpret_cast<BlockQ8_1*>(smem_gemv_q8);

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t b_grp = lane >> 2;
  const std::uint32_t sub = lane & 3u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  const std::uint8_t* wrow =
      row < m ? (w + static_cast<std::size_t>(row) * row_bytes) : nullptr;

  float sum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  for (std::uint32_t b_off = 0; b_off < total_blk; b_off += tile_blk) {
    const std::uint32_t tk_blk =
        (total_blk - b_off) < tile_blk ? (total_blk - b_off) : tile_blk;
    const std::uint32_t words2 = tk_blk * 5u;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const auto* src2 = reinterpret_cast<const uint2*>(
          x_q8 + static_cast<std::size_t>(t) * total_blk + b_off);
      auto* dst2 = reinterpret_cast<uint2*>(sx + t * tile_blk);
      for (std::uint32_t i = threadIdx.x; i < words2; i += 256u) {
        dst2[i] = src2[i];
      }
    }
    __syncthreads();

    if (row < m) {
      const std::uint8_t* wbase = wrow + static_cast<std::size_t>(b_off) * 18u;
      for (std::uint32_t b0 = 0; b0 < tk_blk; b0 += 8u) {
        const std::uint32_t b = b0 + b_grp;
        if (b < tk_blk) {
          const auto* bp =
              reinterpret_cast<const std::uint16_t*>(wbase + b * 18u);
          const float d4 = __half2float(*reinterpret_cast<const __half*>(bp));
          const int v32 = static_cast<int>(bp[1u + 2u * sub]) |
                          (static_cast<int>(bp[2u + 2u * sub]) << 16);
          const int vi0 = v32 & 0x0F0F0F0F;
          const int vi1 = (v32 >> 4) & 0x0F0F0F0F;
          for (std::uint32_t t = 0; t < n_tokens; ++t) {
            const BlockQ8_1& xblk = sx[t * tile_blk + b];
            int dot = Dot4U4S8(vi0, xblk.qs[sub], 0);
            dot = Dot4U4S8(vi1, xblk.qs[4u + sub], dot);
            sum[t] += d4 * (xblk.d * static_cast<float>(dot) - 2.0F * xblk.s);
          }
        }
      }
    }
    if (b_off + tile_blk < total_blk) __syncthreads();
  }

  if (row < m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      float v = WaveSum(sum[t]);
      if (lane == 0u) {
        if (softcap > 0.0F) v = tanhf(v / softcap) * softcap;
        y[static_cast<std::size_t>(t) * m + row] = v;
      }
    }
  }
}

// Q8_0 weights (34-byte blocks, e.g. tied 262k x 2816 token_embd / output head)
// x BlockQ8_1 activations using hardware sdot4. Reads weights ONCE across all
// n_tokens (eliminating the 5x 784 MB LM-head re-read in MTP verify).
__global__ void __launch_bounds__(256)
    GemvWave32BatchedQ8_0Kernel(const std::uint8_t* __restrict__ w,
                                std::uint64_t row_bytes,
                                std::uint32_t total_blk, std::uint32_t tile_blk,
                                const BlockQ8_1* __restrict__ x_q8,
                                float* __restrict__ y, std::uint32_t n_tokens,
                                std::uint32_t m, float softcap) {
  extern __shared__ __align__(8) unsigned char smem_gemv_q80[];
  auto* sx = reinterpret_cast<BlockQ8_1*>(smem_gemv_q80);

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t b_grp = lane >> 3;  // 0..3 (4 blocks per wave step)
  const std::uint32_t sub = lane & 7u;    // 0..7 (4 int8s per lane = 32B/block)
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  const std::uint8_t* wrow =
      row < m ? (w + static_cast<std::size_t>(row) * row_bytes) : nullptr;

  float sum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  for (std::uint32_t b_off = 0; b_off < total_blk; b_off += tile_blk) {
    const std::uint32_t tk_blk =
        (total_blk - b_off) < tile_blk ? (total_blk - b_off) : tile_blk;
    const std::uint32_t words2 = tk_blk * 5u;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const auto* src2 = reinterpret_cast<const uint2*>(
          x_q8 + static_cast<std::size_t>(t) * total_blk + b_off);
      auto* dst2 = reinterpret_cast<uint2*>(sx + t * tile_blk);
      for (std::uint32_t i = threadIdx.x; i < words2; i += 256u) {
        dst2[i] = src2[i];
      }
    }
    __syncthreads();

    if (row < m) {
      const std::uint8_t* wbase = wrow + static_cast<std::size_t>(b_off) * 34u;
      for (std::uint32_t b0 = 0; b0 < tk_blk; b0 += 4u) {
        const std::uint32_t b = b0 + b_grp;
        if (b < tk_blk) {
          const auto* bp =
              reinterpret_cast<const std::uint16_t*>(wbase + b * 34u);
          const float d8 = __half2float(*reinterpret_cast<const __half*>(bp));
          const int w32 = static_cast<int>(bp[1u + 2u * sub]) |
                          (static_cast<int>(bp[2u + 2u * sub]) << 16);
          for (std::uint32_t t = 0; t < n_tokens; ++t) {
            const BlockQ8_1& xblk = sx[t * tile_blk + b];
            const int dot = Dot4S8S8(w32, xblk.qs[sub], 0);
            sum[t] += (d8 * xblk.d) * static_cast<float>(dot);
          }
        }
      }
    }
    if (b_off + tile_blk < total_blk) __syncthreads();
  }

  if (row < m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      float v = WaveSum(sum[t]);
      if (lane == 0u) {
        if (softcap > 0.0F) v = tanhf(v / softcap) * softcap;
        y[static_cast<std::size_t>(t) * m + row] = v;
      }
    }
  }
}

__global__ void __launch_bounds__(256)
    GemvGateUpSwigluBatchedQ8Kernel(const std::uint8_t* __restrict__ w_gate,
                                    const std::uint8_t* __restrict__ w_up,
                                    std::uint64_t row_bytes,
                                    std::uint32_t total_blk,
                                    std::uint32_t tile_blk,
                                    const BlockQ8_1* __restrict__ x_q8,
                                    float* __restrict__ out_act,
                                    std::uint32_t n_tokens, std::uint32_t m) {
  extern __shared__ __align__(8) unsigned char smem_gemv_q8[];
  auto* sx = reinterpret_cast<BlockQ8_1*>(smem_gemv_q8);

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t b_grp = lane >> 2;
  const std::uint32_t sub = lane & 3u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  const std::size_t off = static_cast<std::size_t>(row) * row_bytes;

  float sg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float su[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (std::uint32_t b_off = 0; b_off < total_blk; b_off += tile_blk) {
    const std::uint32_t tk_blk =
        (total_blk - b_off) < tile_blk ? (total_blk - b_off) : tile_blk;
    const std::uint32_t words2 = tk_blk * 5u;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const auto* src2 = reinterpret_cast<const uint2*>(
          x_q8 + static_cast<std::size_t>(t) * total_blk + b_off);
      auto* dst2 = reinterpret_cast<uint2*>(sx + t * tile_blk);
      for (std::uint32_t i = threadIdx.x; i < words2; i += 256u) {
        dst2[i] = src2[i];
      }
    }
    __syncthreads();

    if (row < m) {
      const std::uint8_t* gbase =
          w_gate + off + static_cast<std::size_t>(b_off) * 18u;
      const std::uint8_t* ubase =
          w_up + off + static_cast<std::size_t>(b_off) * 18u;
      for (std::uint32_t b0 = 0; b0 < tk_blk; b0 += 8u) {
        const std::uint32_t b = b0 + b_grp;
        if (b < tk_blk) {
          const auto* gbp =
              reinterpret_cast<const std::uint16_t*>(gbase + b * 18u);
          const float gd4 = __half2float(*reinterpret_cast<const __half*>(gbp));
          const int gv32 = static_cast<int>(gbp[1u + 2u * sub]) |
                           (static_cast<int>(gbp[2u + 2u * sub]) << 16);
          const int gvi0 = gv32 & 0x0F0F0F0F;
          const int gvi1 = (gv32 >> 4) & 0x0F0F0F0F;

          const auto* ubp =
              reinterpret_cast<const std::uint16_t*>(ubase + b * 18u);
          const float ud4 = __half2float(*reinterpret_cast<const __half*>(ubp));
          const int uv32 = static_cast<int>(ubp[1u + 2u * sub]) |
                           (static_cast<int>(ubp[2u + 2u * sub]) << 16);
          const int uvi0 = uv32 & 0x0F0F0F0F;
          const int uvi1 = (uv32 >> 4) & 0x0F0F0F0F;

          for (std::uint32_t t = 0; t < n_tokens; ++t) {
            const BlockQ8_1& xblk = sx[t * tile_blk + b];
            const int xlo = xblk.qs[sub];
            const int xhi = xblk.qs[4u + sub];
            int gdot = Dot4U4S8(gvi0, xlo, 0);
            gdot = Dot4U4S8(gvi1, xhi, gdot);
            int udot = Dot4U4S8(uvi0, xlo, 0);
            udot = Dot4U4S8(uvi1, xhi, udot);
            sg[t] += gd4 * (xblk.d * static_cast<float>(gdot) - 2.0F * xblk.s);
            su[t] += ud4 * (xblk.d * static_cast<float>(udot) - 2.0F * xblk.s);
          }
        }
      }
    }
    if (b_off + tile_blk < total_blk) __syncthreads();
  }

  if (row < m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float g = WaveSum(sg[t]);
      const float u = WaveSum(su[t]);
      if (lane == 0u) {
        out_act[static_cast<std::size_t>(t) * m + row] = GeluTanhOne(g) * u;
      }
    }
  }
}

__global__ void __launch_bounds__(256)
    GemvWave32SingleKernel(const std::uint8_t* __restrict__ w, int type,
                           std::uint64_t row_bytes, std::uint32_t cols,
                           const float* __restrict__ x, float* __restrict__ y,
                           std::uint32_t m, float softcap) {
  extern __shared__ float sx[];
  for (std::uint32_t i = threadIdx.x * 4u; i < cols; i += 256u * 4u) {
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
  }
  __syncthreads();

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  if (row >= m) return;

  const std::uint8_t* wrow = w + static_cast<std::size_t>(row) * row_bytes;
  float sum = WaveRowTileDot(wrow, type, 0u, cols, lane, sx);
  sum = WaveSum(sum);
  if (lane == 0) {
    if (softcap > 0.0F) sum = tanhf(sum / softcap) * softcap;
    y[row] = sum;
  }
}

// Wave32 narrow-batch GEMM (n_tokens in 2..8): reads weights ONCE across all
// n_tokens by tiling activations in 32 KB LDS (8 x 1024 floats).
__global__ void __launch_bounds__(256)
    GemvWave32BatchedKernel(const std::uint8_t* __restrict__ w, int type,
                            std::uint64_t row_bytes, std::uint32_t cols,
                            const float* __restrict__ x, float* __restrict__ y,
                            std::uint32_t n_tokens, std::uint32_t m,
                            float softcap) {
  constexpr std::uint32_t kTileK = 1024;
  __shared__ float sx[8][kTileK];

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  const std::uint8_t* wrow =
      row < m ? (w + static_cast<std::size_t>(row) * row_bytes) : nullptr;

  float sum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  for (std::uint32_t k0 = 0; k0 < cols; k0 += kTileK) {
    const std::uint32_t tk = (cols - k0) < kTileK ? (cols - k0) : kTileK;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float* xt = x + static_cast<std::size_t>(t) * cols + k0;
      for (std::uint32_t i = threadIdx.x * 4u; i < tk; i += 256u * 4u) {
        *reinterpret_cast<float4*>(&sx[t][i]) =
            *reinterpret_cast<const float4*>(xt + i);
      }
    }
    __syncthreads();

    if (row < m) {
      if (type == 2) {
        const std::uint32_t nblk = tk >> 5;
        const std::uint32_t b_grp = lane >> 2;
        const std::uint32_t sub = lane & 3u;
        const std::uint8_t* wbase =
            wrow + static_cast<std::size_t>(k0 >> 5) * 18;
        for (std::uint32_t b0 = 0; b0 < nblk; b0 += 8) {
          const std::uint32_t b = b0 + b_grp;
          if (b < nblk) {
            const auto* bp =
                reinterpret_cast<const std::uint16_t*>(wbase + b * 18u);
            const float d = __half2float(*reinterpret_cast<const __half*>(bp));
            const std::uint32_t q01 = bp[1 + 2 * sub];
            const std::uint32_t q23 = bp[2 + 2 * sub];
            const float wl0 = static_cast<float>(static_cast<int>(q01 & 0xFu) - 8);
            const float wh0 = static_cast<float>(static_cast<int>((q01 >> 4) & 0xFu) - 8);
            const float wl1 = static_cast<float>(static_cast<int>((q01 >> 8) & 0xFu) - 8);
            const float wh1 = static_cast<float>(static_cast<int>((q01 >> 12) & 0xFu) - 8);
            const float wl2 = static_cast<float>(static_cast<int>(q23 & 0xFu) - 8);
            const float wh2 = static_cast<float>(static_cast<int>((q23 >> 4) & 0xFu) - 8);
            const float wl3 = static_cast<float>(static_cast<int>((q23 >> 8) & 0xFu) - 8);
            const float wh3 = static_cast<float>(static_cast<int>((q23 >> 12) & 0xFu) - 8);
            for (std::uint32_t t = 0; t < n_tokens; ++t) {
              const float4 xlo = *reinterpret_cast<const float4*>(
                  &sx[t][b * 32u + 4u * sub]);
              const float4 xhi = *reinterpret_cast<const float4*>(
                  &sx[t][b * 32u + 16u + 4u * sub]);
              const float acc = wl0 * xlo.x + wh0 * xhi.x + wl1 * xlo.y +
                                wh1 * xhi.y + wl2 * xlo.z + wh2 * xhi.z +
                                wl3 * xlo.w + wh3 * xhi.w;
              sum[t] += d * acc;
            }
          }
        }
      } else if (type == 0) {
        const auto* r = reinterpret_cast<const float*>(wrow) + k0;
        for (std::uint32_t j = lane * 4u; j < tk; j += 128u) {
          const float4 wv = *reinterpret_cast<const float4*>(r + j);
          for (std::uint32_t t = 0; t < n_tokens; ++t) {
            const float4 xv = *reinterpret_cast<const float4*>(&sx[t][j]);
            sum[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
          }
        }
      } else {
        for (std::uint32_t t = 0; t < n_tokens; ++t) {
          sum[t] += WaveRowTileDot(wrow, type, k0, tk, lane, sx[t]);
        }
      }
    }
    if (k0 + kTileK < cols) __syncthreads();
  }

  if (row < m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      float v = WaveSum(sum[t]);
      if (lane == 0) {
        if (softcap > 0.0F) v = tanhf(v / softcap) * softcap;
        y[static_cast<std::size_t>(t) * m + row] = v;
      }
    }
  }
}

__global__ void __launch_bounds__(256)
    GemvGateUpSwigluSingleKernel(const std::uint8_t* __restrict__ w_gate,
                                 const std::uint8_t* __restrict__ w_up,
                                 int type, std::uint64_t row_bytes,
                                 std::uint32_t cols,
                                 const float* __restrict__ x,
                                 float* __restrict__ out_act,
                                 std::uint32_t m) {
  extern __shared__ float sx[];
  for (std::uint32_t i = threadIdx.x * 4u; i < cols; i += 256u * 4u) {
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
  }
  __syncthreads();

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  if (row >= m) return;

  const std::size_t off = static_cast<std::size_t>(row) * row_bytes;
  float g = WaveRowTileDot(w_gate + off, type, 0u, cols, lane, sx);
  float u = WaveRowTileDot(w_up + off, type, 0u, cols, lane, sx);
  g = WaveSum(g);
  u = WaveSum(u);
  if (lane == 0) {
    out_act[row] = GeluTanhOne(g) * u;
  }
}

__global__ void __launch_bounds__(256)
    GemvGateUpSwigluBatchedKernel(const std::uint8_t* __restrict__ w_gate,
                                  const std::uint8_t* __restrict__ w_up,
                                  int type, std::uint64_t row_bytes,
                                  std::uint32_t cols,
                                  const float* __restrict__ x,
                                  float* __restrict__ out_act,
                                  std::uint32_t n_tokens, std::uint32_t m) {
  constexpr std::uint32_t kTileK = 1024;
  __shared__ float sx[8][kTileK];

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  const std::size_t off = static_cast<std::size_t>(row) * row_bytes;

  float sg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float su[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (std::uint32_t k0 = 0; k0 < cols; k0 += kTileK) {
    const std::uint32_t tk = (cols - k0) < kTileK ? (cols - k0) : kTileK;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float* xt = x + static_cast<std::size_t>(t) * cols + k0;
      for (std::uint32_t i = threadIdx.x * 4u; i < tk; i += 256u * 4u) {
        *reinterpret_cast<float4*>(&sx[t][i]) =
            *reinterpret_cast<const float4*>(xt + i);
      }
    }
    __syncthreads();
    if (row < m) {
      if (type == 2) {
        const std::uint32_t nblk = tk >> 5;
        const std::uint32_t b_grp = lane >> 2;
        const std::uint32_t sub = lane & 3u;
        const std::size_t boff = static_cast<std::size_t>(k0 >> 5) * 18u;
        const std::uint8_t* gbase = w_gate + off + boff;
        const std::uint8_t* ubase = w_up + off + boff;
        for (std::uint32_t b0 = 0; b0 < nblk; b0 += 8u) {
          const std::uint32_t b = b0 + b_grp;
          if (b < nblk) {
            const auto* gbp =
                reinterpret_cast<const std::uint16_t*>(gbase + b * 18u);
            const float gd = __half2float(*reinterpret_cast<const __half*>(gbp));
            const std::uint32_t gq01 = gbp[1 + 2 * sub];
            const std::uint32_t gq23 = gbp[2 + 2 * sub];
            const float gwl0 = static_cast<float>(static_cast<int>(gq01 & 0xFu) - 8);
            const float gwh0 = static_cast<float>(static_cast<int>((gq01 >> 4) & 0xFu) - 8);
            const float gwl1 = static_cast<float>(static_cast<int>((gq01 >> 8) & 0xFu) - 8);
            const float gwh1 = static_cast<float>(static_cast<int>((gq01 >> 12) & 0xFu) - 8);
            const float gwl2 = static_cast<float>(static_cast<int>(gq23 & 0xFu) - 8);
            const float gwh2 = static_cast<float>(static_cast<int>((gq23 >> 4) & 0xFu) - 8);
            const float gwl3 = static_cast<float>(static_cast<int>((gq23 >> 8) & 0xFu) - 8);
            const float gwh3 = static_cast<float>(static_cast<int>((gq23 >> 12) & 0xFu) - 8);

            const auto* ubp =
                reinterpret_cast<const std::uint16_t*>(ubase + b * 18u);
            const float ud = __half2float(*reinterpret_cast<const __half*>(ubp));
            const std::uint32_t uq01 = ubp[1 + 2 * sub];
            const std::uint32_t uq23 = ubp[2 + 2 * sub];
            const float uwl0 = static_cast<float>(static_cast<int>(uq01 & 0xFu) - 8);
            const float uwh0 = static_cast<float>(static_cast<int>((uq01 >> 4) & 0xFu) - 8);
            const float uwl1 = static_cast<float>(static_cast<int>((uq01 >> 8) & 0xFu) - 8);
            const float uwh1 = static_cast<float>(static_cast<int>((uq01 >> 12) & 0xFu) - 8);
            const float uwl2 = static_cast<float>(static_cast<int>(uq23 & 0xFu) - 8);
            const float uwh2 = static_cast<float>(static_cast<int>((uq23 >> 4) & 0xFu) - 8);
            const float uwl3 = static_cast<float>(static_cast<int>((uq23 >> 8) & 0xFu) - 8);
            const float uwh3 = static_cast<float>(static_cast<int>((uq23 >> 12) & 0xFu) - 8);

            for (std::uint32_t t = 0; t < n_tokens; ++t) {
              const float4 xlo = *reinterpret_cast<const float4*>(
                  &sx[t][b * 32u + 4u * sub]);
              const float4 xhi = *reinterpret_cast<const float4*>(
                  &sx[t][b * 32u + 16u + 4u * sub]);
              sg[t] += gd * (gwl0 * xlo.x + gwh0 * xhi.x + gwl1 * xlo.y +
                             gwh1 * xhi.y + gwl2 * xlo.z + gwh2 * xhi.z +
                             gwl3 * xlo.w + gwh3 * xhi.w);
              su[t] += ud * (uwl0 * xlo.x + uwh0 * xhi.x + uwl1 * xlo.y +
                             uwh1 * xhi.y + uwl2 * xlo.z + uwh2 * xhi.z +
                             uwl3 * xlo.w + uwh3 * xhi.w);
            }
          }
        }
      } else {
        for (std::uint32_t t = 0; t < n_tokens; ++t) {
          sg[t] += WaveRowTileDot(w_gate + off, type, k0, tk, lane, sx[t]);
          su[t] += WaveRowTileDot(w_up + off, type, k0, tk, lane, sx[t]);
        }
      }
    }
    if (k0 + kTileK < cols) __syncthreads();
  }
  if (row < m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float g = WaveSum(sg[t]);
      const float u = WaveSum(su[t]);
      if (lane == 0) {
        out_act[static_cast<std::size_t>(t) * m + row] = GeluTanhOne(g) * u;
      }
    }
  }
}

// Split-K Flash-Decoding GQA kernel for narrow batches (n_tokens <= 8) and MTP
// cross-attention. 128 threads = 4 Wave32 warps per block; zero barriers inside
// the KV loop, coalesced uint2 (4x half) KV reads, exact online softmax merge.
template <std::uint32_t D, std::uint32_t QPG>
__global__ void __launch_bounds__(128) AttentionSplitKKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, float* __restrict__ out,
    float* __restrict__ partial_ml, float* __restrict__ partial_acc,
    std::uint32_t query_pos, std::uint32_t kv_len, std::uint32_t heads,
    std::uint32_t kv_stride, std::uint32_t window, std::uint32_t splits) {
  constexpr std::uint32_t kVec4s = D / 128u;
  const std::uint32_t split_id = blockIdx.x;
  const std::uint32_t kv_h = blockIdx.y;
  const std::uint32_t t = blockIdx.z;

  const std::uint32_t pos_q = query_pos + t;
  const std::uint32_t lo =
      (window > 0u && pos_q + 1u > window) ? (pos_q + 1u - window) : 0u;
  const std::uint32_t hi = (pos_q + 1u < kv_len) ? (pos_q + 1u) : kv_len;

  if (lo >= hi) {
    if (splits == 1u) {
      for (std::uint32_t qq = 0; qq < QPG; ++qq) {
        float* orow = out + (static_cast<std::size_t>(t) * heads +
                             kv_h * QPG + qq) * D;
        for (std::uint32_t i = threadIdx.x; i < D; i += 128u) orow[i] = 0.0F;
      }
    } else if (threadIdx.x < QPG) {
      const std::size_t idx =
          ((static_cast<std::size_t>(t) * heads + kv_h * QPG + threadIdx.x) *
               splits +
           split_id) * 2u;
      partial_ml[idx] = -INFINITY;
      partial_ml[idx + 1] = 0.0F;
    }
    return;
  }

  const std::uint32_t span = hi - lo;
  const std::uint32_t chunk = (span + splits - 1u) / splits;
  const std::uint32_t s_beg = lo + split_id * chunk;
  const std::uint32_t s_end = (s_beg + chunk < hi) ? (s_beg + chunk) : hi;
  if (s_beg >= hi) {
    if (splits > 1u && threadIdx.x < QPG) {
      const std::size_t idx =
          ((static_cast<std::size_t>(t) * heads + kv_h * QPG + threadIdx.x) *
               splits +
           split_id) * 2u;
      partial_ml[idx] = -INFINITY;
      partial_ml[idx + 1] = 0.0F;
    }
    return;
  }

  const std::uint32_t wave_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;

  __shared__ union {
    float4 q_lds[QPG][kVec4s][32];
    float4 w_acc[4][kVec4s][32];
  } smem;
  __shared__ float w_m[4][QPG];
  __shared__ float w_l[4][QPG];

  const float* qbase = q + (static_cast<std::size_t>(t) * heads + kv_h * QPG) * D;
  for (std::uint32_t idx = threadIdx.x; idx < QPG * kVec4s * 32u; idx += 128u) {
    const std::uint32_t qq = idx / (kVec4s * 32u);
    const std::uint32_t rem = idx % (kVec4s * 32u);
    const std::uint32_t v = rem >> 5;
    const std::uint32_t ln = rem & 31u;
    smem.q_lds[qq][v][ln] = *reinterpret_cast<const float4*>(
        qbase + static_cast<std::size_t>(qq) * D + v * 128u + ln * 4u);
  }
  __syncthreads();

  float m_reg[QPG];
  float l_reg[QPG];
  float4 acc_reg[QPG][kVec4s];
  #pragma unroll
  for (std::uint32_t qq = 0; qq < QPG; ++qq) {
    m_reg[qq] = -INFINITY;
    l_reg[qq] = 0.0F;
    #pragma unroll
    for (std::uint32_t v = 0; v < kVec4s; ++v) {
      acc_reg[qq][v] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    }
  }

  for (std::uint32_t pp = s_beg + wave_id; pp < s_end; pp += 4u) {
    const __half* krow =
        k_cache + static_cast<std::size_t>(pp) * kv_stride + kv_h * D;
    float4 kf[kVec4s];
    #pragma unroll
    for (std::uint32_t v = 0; v < kVec4s; ++v) {
      const uint2 u =
          *reinterpret_cast<const uint2*>(krow + v * 128u + lane * 4u);
      const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
      const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
      kf[v] = make_float4(f0.x, f0.y, f1.x, f1.y);
    }

    float dot[QPG];
    #pragma unroll
    for (std::uint32_t qq = 0; qq < QPG; ++qq) {
      float s = 0.0F;
      #pragma unroll
      for (std::uint32_t v = 0; v < kVec4s; ++v) {
        const float4 qf = smem.q_lds[qq][v][lane];
        s += qf.x * kf[v].x + qf.y * kf[v].y + qf.z * kf[v].z + qf.w * kf[v].w;
      }
      dot[qq] = WaveSum(s);
    }

    const __half* vrow =
        v_cache + static_cast<std::size_t>(pp) * kv_stride + kv_h * D;
    float4 vf[kVec4s];
    #pragma unroll
    for (std::uint32_t v = 0; v < kVec4s; ++v) {
      const uint2 u =
          *reinterpret_cast<const uint2*>(vrow + v * 128u + lane * 4u);
      const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
      const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
      vf[v] = make_float4(f0.x, f0.y, f1.x, f1.y);
    }

    #pragma unroll
    for (std::uint32_t qq = 0; qq < QPG; ++qq) {
      const float sc = dot[qq];
      const float m_new = fmaxf(m_reg[qq], sc);
      const float corr =
          (m_reg[qq] == -INFINITY) ? 0.0F : __expf(m_reg[qq] - m_new);
      const float wgt = __expf(sc - m_new);
      m_reg[qq] = m_new;
      l_reg[qq] = l_reg[qq] * corr + wgt;
      #pragma unroll
      for (std::uint32_t v = 0; v < kVec4s; ++v) {
        acc_reg[qq][v].x = acc_reg[qq][v].x * corr + wgt * vf[v].x;
        acc_reg[qq][v].y = acc_reg[qq][v].y * corr + wgt * vf[v].y;
        acc_reg[qq][v].z = acc_reg[qq][v].z * corr + wgt * vf[v].z;
        acc_reg[qq][v].w = acc_reg[qq][v].w * corr + wgt * vf[v].w;
      }
    }
  }

  __syncthreads();
  if (lane == 0) {
    #pragma unroll
    for (std::uint32_t qq = 0; qq < QPG; ++qq) {
      w_m[wave_id][qq] = m_reg[qq];
      w_l[wave_id][qq] = l_reg[qq];
    }
  }

  for (std::uint32_t qq = 0; qq < QPG; ++qq) {
    #pragma unroll
    for (std::uint32_t v = 0; v < kVec4s; ++v) {
      smem.w_acc[wave_id][v][lane] = acc_reg[qq][v];
    }
    __syncthreads();

    if (wave_id == 0) {
      float m_blk = -INFINITY;
      #pragma unroll
      for (int w = 0; w < 4; ++w) {
        if (w_l[w][qq] > 0.0F) m_blk = fmaxf(m_blk, w_m[w][qq]);
      }
      float cw[4];
      float l_blk = 0.0F;
      #pragma unroll
      for (int w = 0; w < 4; ++w) {
        cw[w] = (w_l[w][qq] > 0.0F && m_blk > -INFINITY)
                    ? __expf(w_m[w][qq] - m_blk)
                    : 0.0F;
        l_blk += w_l[w][qq] * cw[w];
      }
      const std::size_t h_idx =
          static_cast<std::size_t>(t) * heads + kv_h * QPG + qq;
      if (splits == 1u) {
        const float inv_l = l_blk > 0.0F ? (1.0F / l_blk) : 0.0F;
        float* orow = out + h_idx * D;
        #pragma unroll
        for (std::uint32_t v = 0; v < kVec4s; ++v) {
          float4 ov = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
          #pragma unroll
          for (int w = 0; w < 4; ++w) {
            const float4 av = smem.w_acc[w][v][lane];
            ov.x += cw[w] * av.x;
            ov.y += cw[w] * av.y;
            ov.z += cw[w] * av.z;
            ov.w += cw[w] * av.w;
          }
          ov.x *= inv_l;
          ov.y *= inv_l;
          ov.z *= inv_l;
          ov.w *= inv_l;
          *reinterpret_cast<float4*>(orow + v * 128u + lane * 4u) = ov;
        }
      } else {
        const std::size_t s_idx = h_idx * splits + split_id;
        if (lane == 0) {
          partial_ml[s_idx * 2u] = m_blk;
          partial_ml[s_idx * 2u + 1u] = l_blk;
        }
        float* prow = partial_acc + s_idx * D;
        #pragma unroll
        for (std::uint32_t v = 0; v < kVec4s; ++v) {
          float4 ov = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
          #pragma unroll
          for (int w = 0; w < 4; ++w) {
            const float4 av = smem.w_acc[w][v][lane];
            ov.x += cw[w] * av.x;
            ov.y += cw[w] * av.y;
            ov.z += cw[w] * av.z;
            ov.w += cw[w] * av.w;
          }
          *reinterpret_cast<float4*>(prow + v * 128u + lane * 4u) = ov;
        }
      }
    }
    if (qq + 1u < QPG) __syncthreads();
  }
}

// Multi-token causal Split-K Flash-Decoding GQA kernel for MTP verification
// (2 <= n_tokens <= MAX_TOK). Sweeps K/V cache ONCE across all n_tokens instead
// of n_tokens times, with compile-time unrolled register accumulators per token.
template <std::uint32_t D, std::uint32_t QPG_SUB, std::uint32_t MAX_TOK>
__global__ void __launch_bounds__(256) AttentionSplitKMultiKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, float* __restrict__ out,
    float* __restrict__ partial_ml, float* __restrict__ partial_acc,
    std::uint32_t n_tokens, std::uint32_t query_pos, std::uint32_t kv_len,
    std::uint32_t heads, std::uint32_t kv_stride, std::uint32_t window,
    std::uint32_t splits) {
  constexpr std::uint32_t kVec4s = D / 128u;
  constexpr std::uint32_t kStreams = 8u / QPG_SUB;
  static_assert(QPG_SUB * kStreams == 8u, "8 waves per 256-thread block");

  const std::uint32_t q_sub = blockIdx.x;
  const std::uint32_t split_id = blockIdx.y;
  const std::uint32_t kv_h = blockIdx.z;
  const std::uint32_t head_base = (kv_h * gridDim.x + q_sub) * QPG_SUB;

  const std::uint32_t pos_last = query_pos + n_tokens - 1u;
  const std::uint32_t lo_all =
      (window > 0u && query_pos + 1u > window) ? (query_pos + 1u - window) : 0u;
  const std::uint32_t hi_all =
      (pos_last + 1u < kv_len) ? (pos_last + 1u) : kv_len;

  if (lo_all >= hi_all) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      if (splits == 1u) {
        for (std::uint32_t qq = 0; qq < QPG_SUB; ++qq) {
          float* orow =
              out + (static_cast<std::size_t>(t) * heads + head_base + qq) * D;
          for (std::uint32_t i = threadIdx.x; i < D; i += 256u) orow[i] = 0.0F;
        }
      } else if (threadIdx.x < QPG_SUB) {
        const std::size_t idx =
            ((static_cast<std::size_t>(t) * heads + head_base + threadIdx.x) *
                 splits +
             split_id) *
            2u;
        partial_ml[idx] = -INFINITY;
        partial_ml[idx + 1u] = 0.0F;
      }
    }
    return;
  }

  const std::uint32_t span = hi_all - lo_all;
  const std::uint32_t chunk = (span + splits - 1u) / splits;
  const std::uint32_t s_beg = lo_all + split_id * chunk;
  const std::uint32_t s_end = (s_beg + chunk < hi_all) ? (s_beg + chunk) : hi_all;
  if (s_beg >= hi_all) {
    if (splits > 1u && threadIdx.x < QPG_SUB) {
      for (std::uint32_t t = 0; t < n_tokens; ++t) {
        const std::size_t idx =
            ((static_cast<std::size_t>(t) * heads + head_base + threadIdx.x) *
                 splits +
             split_id) *
            2u;
        partial_ml[idx] = -INFINITY;
        partial_ml[idx + 1u] = 0.0F;
      }
    }
    return;
  }

  const std::uint32_t wave_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t pw = wave_id / QPG_SUB;
  const std::uint32_t qq = wave_id % QPG_SUB;

  __shared__ union {
    float4 q_lds[MAX_TOK][QPG_SUB][kVec4s][32];
    float4 w_acc[QPG_SUB][kStreams][kVec4s][32];
  } smem;
  __shared__ float w_m[QPG_SUB][kStreams];
  __shared__ float w_l[QPG_SUB][kStreams];

  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const float* qbase =
        q + (static_cast<std::size_t>(t) * heads + head_base) * D;
    for (std::uint32_t idx = threadIdx.x; idx < QPG_SUB * kVec4s * 32u;
         idx += 256u) {
      const std::uint32_t qh = idx / (kVec4s * 32u);
      const std::uint32_t rem = idx % (kVec4s * 32u);
      const std::uint32_t v = rem >> 5;
      const std::uint32_t ln = rem & 31u;
      smem.q_lds[t][qh][v][ln] = *reinterpret_cast<const float4*>(
          qbase + static_cast<std::size_t>(qh) * D + v * 128u + ln * 4u);
    }
  }
  __syncthreads();

  float m_reg[MAX_TOK];
  float l_reg[MAX_TOK];
  float4 acc_reg[MAX_TOK][kVec4s];
  #pragma unroll
  for (std::uint32_t t = 0; t < MAX_TOK; ++t) {
    m_reg[t] = -INFINITY;
    l_reg[t] = 0.0F;
    #pragma unroll
    for (std::uint32_t v = 0; v < kVec4s; ++v) {
      acc_reg[t][v] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    }
  }

  const bool all_valid =
      (s_end <= query_pos + 1u) &&
      (window == 0u || s_beg + window > pos_last);

  for (std::uint32_t pp = s_beg + pw; pp < s_end; pp += kStreams) {
    const __half* krow =
        k_cache + static_cast<std::size_t>(pp) * kv_stride + kv_h * D;
    float4 kf[kVec4s];
    #pragma unroll
    for (std::uint32_t v = 0; v < kVec4s; ++v) {
      const uint2 u =
          *reinterpret_cast<const uint2*>(krow + v * 128u + lane * 4u);
      const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
      const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
      kf[v] = make_float4(f0.x, f0.y, f1.x, f1.y);
    }

    const __half* vrow =
        v_cache + static_cast<std::size_t>(pp) * kv_stride + kv_h * D;
    float4 vf[kVec4s];
    #pragma unroll
    for (std::uint32_t v = 0; v < kVec4s; ++v) {
      const uint2 u =
          *reinterpret_cast<const uint2*>(vrow + v * 128u + lane * 4u);
      const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
      const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
      vf[v] = make_float4(f0.x, f0.y, f1.x, f1.y);
    }

    #pragma unroll
    for (std::uint32_t t = 0; t < MAX_TOK; ++t) {
      if (t < n_tokens) {
        const std::uint32_t pos_t = query_pos + t;
        if (all_valid ||
            (pp <= pos_t && (window == 0u || pp + window > pos_t))) {
          float s = 0.0F;
          #pragma unroll
          for (std::uint32_t v = 0; v < kVec4s; ++v) {
            const float4 qf = smem.q_lds[t][qq][v][lane];
            s += qf.x * kf[v].x + qf.y * kf[v].y + qf.z * kf[v].z +
                 qf.w * kf[v].w;
          }
          const float sc = WaveSum(s);
          const float m_new = fmaxf(m_reg[t], sc);
          const float corr =
              (m_reg[t] == -INFINITY) ? 0.0F : __expf(m_reg[t] - m_new);
          const float wgt = __expf(sc - m_new);
          m_reg[t] = m_new;
          l_reg[t] = l_reg[t] * corr + wgt;
          #pragma unroll
          for (std::uint32_t v = 0; v < kVec4s; ++v) {
            acc_reg[t][v].x = acc_reg[t][v].x * corr + wgt * vf[v].x;
            acc_reg[t][v].y = acc_reg[t][v].y * corr + wgt * vf[v].y;
            acc_reg[t][v].z = acc_reg[t][v].z * corr + wgt * vf[v].z;
            acc_reg[t][v].w = acc_reg[t][v].w * corr + wgt * vf[v].w;
          }
        }
      }
    }
  }

  __syncthreads();
  #pragma unroll
  for (std::uint32_t t = 0; t < MAX_TOK; ++t) {
    if (t < n_tokens) {
      if (lane == 0u) {
        w_m[qq][pw] = m_reg[t];
        w_l[qq][pw] = l_reg[t];
      }
      #pragma unroll
      for (std::uint32_t v = 0; v < kVec4s; ++v) {
        smem.w_acc[qq][pw][v][lane] = acc_reg[t][v];
      }
      __syncthreads();

      if (wave_id < QPG_SUB) {
        const std::uint32_t hq = wave_id;
        float m_blk = -INFINITY;
        #pragma unroll
        for (std::uint32_t w = 0; w < kStreams; ++w) {
          if (w_l[hq][w] > 0.0F) m_blk = fmaxf(m_blk, w_m[hq][w]);
        }
        float cw[kStreams];
        float l_blk = 0.0F;
        #pragma unroll
        for (std::uint32_t w = 0; w < kStreams; ++w) {
          cw[w] = (w_l[hq][w] > 0.0F && m_blk > -INFINITY)
                      ? __expf(w_m[hq][w] - m_blk)
                      : 0.0F;
          l_blk += w_l[hq][w] * cw[w];
        }
        const std::size_t h_idx =
            static_cast<std::size_t>(t) * heads + head_base + hq;
        if (splits == 1u) {
          const float inv_l = l_blk > 0.0F ? (1.0F / l_blk) : 0.0F;
          float* orow = out + h_idx * D;
          #pragma unroll
          for (std::uint32_t v = 0; v < kVec4s; ++v) {
            float4 ov = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            #pragma unroll
            for (std::uint32_t w = 0; w < kStreams; ++w) {
              const float4 av = smem.w_acc[hq][w][v][lane];
              ov.x += cw[w] * av.x;
              ov.y += cw[w] * av.y;
              ov.z += cw[w] * av.z;
              ov.w += cw[w] * av.w;
            }
            ov.x *= inv_l;
            ov.y *= inv_l;
            ov.z *= inv_l;
            ov.w *= inv_l;
            *reinterpret_cast<float4*>(orow + v * 128u + lane * 4u) = ov;
          }
        } else {
          const std::size_t s_idx = h_idx * splits + split_id;
          if (lane == 0u) {
            partial_ml[s_idx * 2u] = m_blk;
            partial_ml[s_idx * 2u + 1u] = l_blk;
          }
          float* prow = partial_acc + s_idx * D;
          #pragma unroll
          for (std::uint32_t v = 0; v < kVec4s; ++v) {
            float4 ov = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            #pragma unroll
            for (std::uint32_t w = 0; w < kStreams; ++w) {
              const float4 av = smem.w_acc[hq][w][v][lane];
              ov.x += cw[w] * av.x;
              ov.y += cw[w] * av.y;
              ov.z += cw[w] * av.z;
              ov.w += cw[w] * av.w;
            }
            *reinterpret_cast<float4*>(prow + v * 128u + lane * 4u) = ov;
          }
        }
      }
      if (t + 1u < n_tokens) __syncthreads();
    }
  }
}

__global__ void __launch_bounds__(128) AttentionSplitKMergeKernel(
    const float* __restrict__ partial_ml, const float* __restrict__ partial_acc,
    float* __restrict__ out, std::uint32_t heads, std::uint32_t d,
    std::uint32_t splits) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::size_t h_idx = static_cast<std::size_t>(t) * heads + h;

  __shared__ float cs[64];
  __shared__ float inv_l_sh;
  if (threadIdx.x == 0) {
    float m_max = -INFINITY;
    for (std::uint32_t s = 0; s < splits; ++s) {
      const float ms = partial_ml[(h_idx * splits + s) * 2u];
      const float ls = partial_ml[(h_idx * splits + s) * 2u + 1u];
      if (ls > 0.0F) m_max = fmaxf(m_max, ms);
    }
    float l_tot = 0.0F;
    for (std::uint32_t s = 0; s < splits; ++s) {
      const float ms = partial_ml[(h_idx * splits + s) * 2u];
      const float ls = partial_ml[(h_idx * splits + s) * 2u + 1u];
      const float c =
          (ls > 0.0F && m_max > -INFINITY) ? __expf(ms - m_max) : 0.0F;
      cs[s] = c;
      l_tot += ls * c;
    }
    inv_l_sh = l_tot > 0.0F ? (1.0F / l_tot) : 0.0F;
  }
  __syncthreads();

  const float inv_l = inv_l_sh;
  float* orow = out + h_idx * d;
  for (std::uint32_t i = threadIdx.x * 4u; i < d; i += 128u * 4u) {
    float4 acc = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    for (std::uint32_t s = 0; s < splits; ++s) {
      const float c = cs[s];
      if (c == 0.0F) continue;
      const float4 pv = *reinterpret_cast<const float4*>(
          partial_acc + (h_idx * splits + s) * d + i);
      acc.x += c * pv.x;
      acc.y += c * pv.y;
      acc.z += c * pv.z;
      acc.w += c * pv.w;
    }
    acc.x *= inv_l;
    acc.y *= inv_l;
    acc.z *= inv_l;
    acc.w *= inv_l;
    *reinterpret_cast<float4*>(orow + i) = acc;
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

void EmbedScaleDevice(const void* table, core::GgmlType type,
                      const std::int32_t* d_tokens, float* res,
                      std::uint32_t n_tokens, std::uint32_t hidden, float scale,
                      hipStream_t stream) {
  if (n_tokens == 0) return;
  hipLaunchKernelGGL(EmbedScaleKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, static_cast<const std::uint8_t*>(table),
                     static_cast<int>(type), d_tokens, res, hidden, scale);
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
  if (n_rows == 0 || dim == 0) return;
  hipLaunchKernelGGL(RmsNormKernel, dim3(n_rows), dim3(kThreads), 0, stream, x,
                     gamma, out, dim, 1, eps);
}

void RmsNormAdd(const float* x, const float* gamma, float* residual_inout,
                std::uint32_t n_rows, std::uint32_t dim, float eps,
                float post_scale, hipStream_t stream) {
  if (n_rows == 0 || dim == 0) return;
  hipLaunchKernelGGL(RmsNormAddKernel, dim3(n_rows), dim3(kThreads), 0, stream,
                     x, gamma, residual_inout, dim, eps, post_scale);
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

void AttentionCross(const float* q, const __half* k_cache,
                    const __half* v_cache, float* out, std::uint32_t n_tokens,
                    std::uint32_t query_pos, std::uint32_t kv_len,
                    std::uint32_t heads, std::uint32_t kv_heads,
                    std::uint32_t d, std::uint32_t kv_stride,
                    std::uint32_t window, hipStream_t stream) {
  if (n_tokens == 0) return;
  const bool splitk_shape =
      (n_tokens <= 8u) && (getenv("GEMMA4_NO_SPLITK_ATTN") == nullptr) &&
      ((d == 256u && heads == 16u && kv_heads == 8u) ||
       (d == 512u && heads == 16u && kv_heads == 2u));
  if (splitk_shape) {
    const std::uint32_t pos_last = query_pos + n_tokens - 1u;
    const std::uint32_t hi_max =
        (pos_last + 1u < kv_len) ? (pos_last + 1u) : kv_len;
    const std::uint32_t lo_min =
        (window > 0u && query_pos + 1u > window) ? (query_pos + 1u - window) : 0u;
    const std::uint32_t span = (hi_max > lo_min) ? (hi_max - lo_min) : 1u;
    const bool multi_splitk =
        (n_tokens >= 2u && n_tokens <= 6u) &&
        (getenv("GEMMA4_NO_MULTI_SPLITK") == nullptr);
    const std::uint32_t min_tile =
        multi_splitk ? ((d == 256u) ? 64u : 32u)
                     : ((kv_heads * n_tokens >= 8u) ? 128u : 64u);
    std::uint32_t splits = (span + min_tile - 1u) / min_tile;
    if (splits < 1u) splits = 1u;
    if (splits > 64u) splits = 64u;

    static float* s_partial_ml = nullptr;
    static float* s_partial_acc = nullptr;
    if (splits > 1u && (s_partial_ml == nullptr || s_partial_acc == nullptr)) {
      constexpr std::size_t kMaxMl = 8ull * 16ull * 64ull * 2ull;
      constexpr std::size_t kMaxAcc = 8ull * 16ull * 64ull * 512ull;
      void* p_ml = nullptr;
      void* p_acc = nullptr;
      if (hipMalloc(&p_ml, kMaxMl * sizeof(float)) == hipSuccess &&
          hipMalloc(&p_acc, kMaxAcc * sizeof(float)) == hipSuccess) {
        s_partial_ml = static_cast<float*>(p_ml);
        s_partial_acc = static_cast<float*>(p_acc);
      } else {
        if (p_ml != nullptr) (void)hipFree(p_ml);
        if (p_acc != nullptr) (void)hipFree(p_acc);
        splits = 1u;
      }
    }

    if (multi_splitk) {
      if (d == 256u) {
        const dim3 mgrid(1u, splits, kv_heads);
        hipLaunchKernelGGL((AttentionSplitKMultiKernel<256, 2, 6>), mgrid,
                           dim3(256), 0, stream, q, k_cache, v_cache, out,
                           s_partial_ml, s_partial_acc, n_tokens, query_pos,
                           kv_len, heads, kv_stride, window, splits);
      } else {
        const dim3 mgrid(2u, splits, kv_heads);
        hipLaunchKernelGGL((AttentionSplitKMultiKernel<512, 4, 6>), mgrid,
                           dim3(256), 0, stream, q, k_cache, v_cache, out,
                           s_partial_ml, s_partial_acc, n_tokens, query_pos,
                           kv_len, heads, kv_stride, window, splits);
      }
    } else {
      const dim3 grid(splits, kv_heads, n_tokens);
      if (d == 256u) {
        hipLaunchKernelGGL((AttentionSplitKKernel<256, 2>), grid, dim3(128), 0,
                           stream, q, k_cache, v_cache, out, s_partial_ml,
                           s_partial_acc, query_pos, kv_len, heads, kv_stride,
                           window, splits);
      } else {
        hipLaunchKernelGGL((AttentionSplitKKernel<512, 8>), grid, dim3(128), 0,
                           stream, q, k_cache, v_cache, out, s_partial_ml,
                           s_partial_acc, query_pos, kv_len, heads, kv_stride,
                           window, splits);
      }
    }
    if (splits > 1u) {
      hipLaunchKernelGGL(AttentionSplitKMergeKernel, dim3(heads, n_tokens),
                         dim3(128), 0, stream, s_partial_ml, s_partial_acc, out,
                         heads, d, splits);
    }
    return;
  }

  if (d <= 256u) {
    hipLaunchKernelGGL((AttentionGqaKernel<256>), dim3(n_tokens, kv_heads),
                       dim3(kThreads), 0, stream, q, k_cache, v_cache, out,
                       query_pos, heads, kv_heads, d, kv_stride, window,
                       kv_len);
    return;
  }
  hipLaunchKernelGGL((AttentionGqaKernel<512>), dim3(n_tokens, kv_heads),
                     dim3(kThreads), 0, stream, q, k_cache, v_cache, out,
                     query_pos, heads, kv_heads, d, kv_stride, window,
                     kv_len);
}

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               float* out, std::uint32_t n_tokens,
               std::uint32_t start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t d,
               std::uint32_t kv_stride, std::uint32_t window,
               hipStream_t stream) {
  const std::uint32_t kv_len = start_pos + n_tokens;
  if (n_tokens <= 8u && getenv("GEMMA4_NO_SPLITK_ATTN") == nullptr) {
    AttentionCross(q, k_cache, v_cache, out, n_tokens, start_pos, kv_len, heads,
                   kv_heads, d, kv_stride, window, stream);
    return;
  }
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

__global__ void __launch_bounds__(256) MoeVecGateUpSwigluKernel(
    const void* __restrict__ gate_up, std::uint64_t row_bytes,
    std::uint64_t expert_stride_bytes, const float* __restrict__ x,
    const std::int32_t* __restrict__ ids, float* __restrict__ act_out,
    std::uint32_t used, std::uint32_t ff, std::uint32_t hidden) {
  const std::uint32_t slot = blockIdx.x;
  const std::int32_t e = ids[slot];
  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t out_row = blockIdx.y * 8u + warp_id;
  if (e < 0) {
    if (out_row < ff && lane == 0) {
      act_out[static_cast<std::size_t>(slot) * ff + out_row] = 0.0F;
    }
    return;
  }
  extern __shared__ float sx[];
  const float* xr = x + static_cast<std::size_t>(slot / used) * hidden;
  for (std::uint32_t i = threadIdx.x * 4u; i < hidden; i += 256u * 4u) {
    *reinterpret_cast<float4*>(sx + i) =
        *reinterpret_cast<const float4*>(xr + i);
  }
  __syncthreads();
  if (out_row >= ff) return;

  const auto* ebase = static_cast<const std::uint8_t*>(gate_up) +
                      static_cast<std::size_t>(e) * expert_stride_bytes;
  const auto* grow = ebase + static_cast<std::size_t>(out_row) * row_bytes;
  const auto* urow =
      ebase + static_cast<std::size_t>(ff + out_row) * row_bytes;
  float g = WaveRowTileDot(grow, 2, 0u, hidden, lane, sx);
  float u = WaveRowTileDot(urow, 2, 0u, hidden, lane, sx);
  g = WaveSum(g);
  u = WaveSum(u);
  if (lane == 0) {
    act_out[static_cast<std::size_t>(slot) * ff + out_row] =
        GeluTanhOne(g) * u;
  }
}

namespace {

static std::uint32_t* g_moe_u_cnt = nullptr;
static std::int32_t* g_moe_u_exp = nullptr;
static std::uint32_t* g_moe_u_len = nullptr;
static std::uint32_t* g_moe_u_slots = nullptr;

bool EnsureMoeUniqueBuffers() {
  if (g_moe_u_cnt != nullptr) return true;
  void *p_cnt = nullptr, *p_exp = nullptr, *p_len = nullptr, *p_slots = nullptr;
  if (hipMalloc(&p_cnt, sizeof(std::uint32_t)) != hipSuccess ||
      hipMalloc(&p_exp, 64 * sizeof(std::int32_t)) != hipSuccess ||
      hipMalloc(&p_len, 64 * sizeof(std::uint32_t)) != hipSuccess ||
      hipMalloc(&p_slots, 64 * 8 * sizeof(std::uint32_t)) != hipSuccess) {
    if (p_cnt) (void)hipFree(p_cnt);
    if (p_exp) (void)hipFree(p_exp);
    if (p_len) (void)hipFree(p_len);
    if (p_slots) (void)hipFree(p_slots);
    return false;
  }
  g_moe_u_cnt = static_cast<std::uint32_t*>(p_cnt);
  g_moe_u_exp = static_cast<std::int32_t*>(p_exp);
  g_moe_u_len = static_cast<std::uint32_t*>(p_len);
  g_moe_u_slots = static_cast<std::uint32_t*>(p_slots);
  return true;
}

__global__ void __launch_bounds__(256) MoeBuildUniqueBucketsKernel(
    const std::int32_t* __restrict__ ids, std::uint32_t slots,
    std::uint32_t* __restrict__ u_cnt, std::int32_t* __restrict__ u_exp,
    std::uint32_t* __restrict__ u_len, std::uint32_t* __restrict__ u_slots) {
  __shared__ std::uint32_t s_cnt[256];
  __shared__ std::uint32_t s_list[256][8];
  __shared__ std::uint32_t s_uniq;

  const std::uint32_t tid = threadIdx.x;
  s_cnt[tid] = 0u;
  if (tid == 0u) s_uniq = 0u;
  __syncthreads();

  if (tid < slots) {
    const std::int32_t e = ids[tid];
    if (e >= 0 && e < 256) {
      const std::uint32_t pos = atomicAdd(&s_cnt[e], 1u);
      if (pos < 8u) s_list[e][pos] = tid;
    }
  }
  __syncthreads();

  const std::uint32_t c = s_cnt[tid];
  if (c > 0u) {
    const std::uint32_t u = atomicAdd(&s_uniq, 1u);
    if (u < 64u) {
      u_exp[u] = static_cast<std::int32_t>(tid);
      const std::uint32_t cl = c < 8u ? c : 8u;
      u_len[u] = cl;
      for (std::uint32_t k = 0; k < cl; ++k) {
        u_slots[u * 8u + k] = s_list[tid][k];
      }
    }
  }
  __syncthreads();
  if (tid == 0u) *u_cnt = s_uniq < 64u ? s_uniq : 64u;
}

__global__ void __launch_bounds__(256) MoeGroupedGateUpSwigluKernel(
    const std::uint8_t* __restrict__ gate_up, std::uint64_t row_bytes,
    std::uint64_t expert_stride_bytes, const float* __restrict__ x,
    const std::uint32_t* __restrict__ u_cnt,
    const std::int32_t* __restrict__ u_exp,
    const std::uint32_t* __restrict__ u_len,
    const std::uint32_t* __restrict__ u_slots, float* __restrict__ act_out,
    std::uint32_t used, std::uint32_t ff, std::uint32_t hidden) {
  const std::uint32_t u = blockIdx.y;
  if (u >= *u_cnt) return;

  constexpr std::uint32_t kTileK = 1024;
  __shared__ float sx[8][kTileK];

  const std::int32_t e = u_exp[u];
  const std::uint32_t cnt = u_len[u];
  const std::uint32_t* slots_u = u_slots + u * 8u;

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t out_row = blockIdx.x * 8u + warp_id;

  const std::uint8_t* ebase =
      gate_up + static_cast<std::size_t>(e) * expert_stride_bytes;
  const std::uint8_t* grow =
      out_row < ff ? (ebase + static_cast<std::size_t>(out_row) * row_bytes)
                   : nullptr;
  const std::uint8_t* urow =
      out_row < ff
          ? (ebase + static_cast<std::size_t>(ff + out_row) * row_bytes)
          : nullptr;

  float sg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float su[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  for (std::uint32_t k0 = 0; k0 < hidden; k0 += kTileK) {
    const std::uint32_t tk = (hidden - k0) < kTileK ? (hidden - k0) : kTileK;
    for (std::uint32_t m = 0; m < cnt; ++m) {
      const std::uint32_t tok = slots_u[m] / used;
      const float* xt = x + static_cast<std::size_t>(tok) * hidden + k0;
      for (std::uint32_t i = threadIdx.x * 4u; i < tk; i += 256u * 4u) {
        *reinterpret_cast<float4*>(&sx[m][i]) =
            *reinterpret_cast<const float4*>(xt + i);
      }
    }
    __syncthreads();

    if (out_row < ff) {
      const std::uint32_t nblk = tk >> 5;
      const std::uint32_t b_grp = lane >> 2;
      const std::uint32_t sub = lane & 3u;
      const std::size_t boff = static_cast<std::size_t>(k0 >> 5) * 18u;
      const std::uint8_t* gbase = grow + boff;
      const std::uint8_t* ubase = urow + boff;
      for (std::uint32_t b0 = 0; b0 < nblk; b0 += 8u) {
        const std::uint32_t b = b0 + b_grp;
        if (b < nblk) {
          const auto* gbp =
              reinterpret_cast<const std::uint16_t*>(gbase + b * 18u);
          const float gd = __half2float(*reinterpret_cast<const __half*>(gbp));
          const std::uint32_t gq01 = gbp[1 + 2 * sub];
          const std::uint32_t gq23 = gbp[2 + 2 * sub];
          const float gwl0 = static_cast<float>(static_cast<int>(gq01 & 0xFu) - 8);
          const float gwh0 = static_cast<float>(static_cast<int>((gq01 >> 4) & 0xFu) - 8);
          const float gwl1 = static_cast<float>(static_cast<int>((gq01 >> 8) & 0xFu) - 8);
          const float gwh1 = static_cast<float>(static_cast<int>((gq01 >> 12) & 0xFu) - 8);
          const float gwl2 = static_cast<float>(static_cast<int>(gq23 & 0xFu) - 8);
          const float gwh2 = static_cast<float>(static_cast<int>((gq23 >> 4) & 0xFu) - 8);
          const float gwl3 = static_cast<float>(static_cast<int>((gq23 >> 8) & 0xFu) - 8);
          const float gwh3 = static_cast<float>(static_cast<int>((gq23 >> 12) & 0xFu) - 8);

          const auto* ubp =
              reinterpret_cast<const std::uint16_t*>(ubase + b * 18u);
          const float ud = __half2float(*reinterpret_cast<const __half*>(ubp));
          const std::uint32_t uq01 = ubp[1 + 2 * sub];
          const std::uint32_t uq23 = ubp[2 + 2 * sub];
          const float uwl0 = static_cast<float>(static_cast<int>(uq01 & 0xFu) - 8);
          const float uwh0 = static_cast<float>(static_cast<int>((uq01 >> 4) & 0xFu) - 8);
          const float uwl1 = static_cast<float>(static_cast<int>((uq01 >> 8) & 0xFu) - 8);
          const float uwh1 = static_cast<float>(static_cast<int>((uq01 >> 12) & 0xFu) - 8);
          const float uwl2 = static_cast<float>(static_cast<int>(uq23 & 0xFu) - 8);
          const float uwh2 = static_cast<float>(static_cast<int>((uq23 >> 4) & 0xFu) - 8);
          const float uwl3 = static_cast<float>(static_cast<int>((uq23 >> 8) & 0xFu) - 8);
          const float uwh3 = static_cast<float>(static_cast<int>((uq23 >> 12) & 0xFu) - 8);

          for (std::uint32_t m = 0; m < cnt; ++m) {
            const float4 xlo = *reinterpret_cast<const float4*>(
                &sx[m][b * 32u + 4u * sub]);
            const float4 xhi = *reinterpret_cast<const float4*>(
                &sx[m][b * 32u + 16u + 4u * sub]);
            sg[m] += gd * (gwl0 * xlo.x + gwh0 * xhi.x + gwl1 * xlo.y +
                           gwh1 * xhi.y + gwl2 * xlo.z + gwh2 * xhi.z +
                           gwl3 * xlo.w + gwh3 * xhi.w);
            su[m] += ud * (uwl0 * xlo.x + uwh0 * xhi.x + uwl1 * xlo.y +
                           uwh1 * xhi.y + uwl2 * xlo.z + uwh2 * xhi.z +
                           uwl3 * xlo.w + uwh3 * xhi.w);
          }
        }
      }
    }
    if (k0 + kTileK < hidden) __syncthreads();
  }

  if (out_row < ff) {
    for (std::uint32_t m = 0; m < cnt; ++m) {
      const float g = WaveSum(sg[m]);
      const float up = WaveSum(su[m]);
      if (lane == 0u) {
        act_out[static_cast<std::size_t>(slots_u[m]) * ff + out_row] =
            GeluTanhOne(g) * up;
      }
    }
  }
}

__global__ void __launch_bounds__(256) MoeGroupedDownKernel(
    const std::uint8_t* __restrict__ down, std::uint64_t row_bytes,
    std::uint64_t expert_stride_bytes, const float* __restrict__ act,
    const std::uint32_t* __restrict__ u_cnt,
    const std::int32_t* __restrict__ u_exp,
    const std::uint32_t* __restrict__ u_len,
    const std::uint32_t* __restrict__ u_slots, float* __restrict__ out,
    std::uint32_t hidden, std::uint32_t ff) {
  const std::uint32_t u = blockIdx.y;
  if (u >= *u_cnt) return;

  constexpr std::uint32_t kTileK = 1024;
  __shared__ float sa[8][kTileK];

  const std::int32_t e = u_exp[u];
  const std::uint32_t cnt = u_len[u];
  const std::uint32_t* slots_u = u_slots + u * 8u;

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t h = blockIdx.x * 8u + warp_id;

  const std::uint8_t* wrow =
      h < hidden ? (down + static_cast<std::size_t>(e) * expert_stride_bytes +
                    static_cast<std::size_t>(h) * row_bytes)
                 : nullptr;

  float sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (std::uint32_t k0 = 0; k0 < ff; k0 += kTileK) {
    const std::uint32_t tk = (ff - k0) < kTileK ? (ff - k0) : kTileK;
    for (std::uint32_t m = 0; m < cnt; ++m) {
      const float* ar = act + static_cast<std::size_t>(slots_u[m]) * ff + k0;
      for (std::uint32_t i = threadIdx.x * 4u; i < tk; i += 256u * 4u) {
        *reinterpret_cast<float4*>(&sa[m][i]) =
            *reinterpret_cast<const float4*>(ar + i);
      }
    }
    __syncthreads();

    if (h < hidden) {
      const std::uint32_t nblk = tk >> 5;
      const std::uint32_t b_grp = lane >> 2;
      const std::uint32_t sub = lane & 3u;
      const std::uint8_t* wbase =
          wrow + static_cast<std::size_t>(k0 >> 5) * 18u;
      for (std::uint32_t b0 = 0; b0 < nblk; b0 += 8u) {
        const std::uint32_t b = b0 + b_grp;
        if (b < nblk) {
          const auto* bp =
              reinterpret_cast<const std::uint16_t*>(wbase + b * 18u);
          const float d = __half2float(*reinterpret_cast<const __half*>(bp));
          const std::uint32_t q01 = bp[1 + 2 * sub];
          const std::uint32_t q23 = bp[2 + 2 * sub];
          const float wl0 = static_cast<float>(static_cast<int>(q01 & 0xFu) - 8);
          const float wh0 = static_cast<float>(static_cast<int>((q01 >> 4) & 0xFu) - 8);
          const float wl1 = static_cast<float>(static_cast<int>((q01 >> 8) & 0xFu) - 8);
          const float wh1 = static_cast<float>(static_cast<int>((q01 >> 12) & 0xFu) - 8);
          const float wl2 = static_cast<float>(static_cast<int>(q23 & 0xFu) - 8);
          const float wh2 = static_cast<float>(static_cast<int>((q23 >> 4) & 0xFu) - 8);
          const float wl3 = static_cast<float>(static_cast<int>((q23 >> 8) & 0xFu) - 8);
          const float wh3 = static_cast<float>(static_cast<int>((q23 >> 12) & 0xFu) - 8);
          for (std::uint32_t m = 0; m < cnt; ++m) {
            const float4 xlo = *reinterpret_cast<const float4*>(
                &sa[m][b * 32u + 4u * sub]);
            const float4 xhi = *reinterpret_cast<const float4*>(
                &sa[m][b * 32u + 16u + 4u * sub]);
            sum[m] += d * (wl0 * xlo.x + wh0 * xhi.x + wl1 * xlo.y +
                           wh1 * xhi.y + wl2 * xlo.z + wh2 * xhi.z +
                           wl3 * xlo.w + wh3 * xhi.w);
          }
        }
      }
    }
    if (k0 + kTileK < ff) __syncthreads();
  }

  if (h < hidden) {
    for (std::uint32_t m = 0; m < cnt; ++m) {
      const float v = WaveSum(sum[m]);
      if (lane == 0u) {
        out[static_cast<std::size_t>(slots_u[m]) * hidden + h] = v;
      }
    }
  }
}

__global__ void __launch_bounds__(256) MoeGroupedGateUpSwigluQ8Kernel(
    const std::uint8_t* __restrict__ gate_up, std::uint64_t row_bytes,
    std::uint64_t expert_stride_bytes, const BlockQ8_1* __restrict__ x_q8,
    const std::uint32_t* __restrict__ u_cnt,
    const std::int32_t* __restrict__ u_exp,
    const std::uint32_t* __restrict__ u_len,
    const std::uint32_t* __restrict__ u_slots, float* __restrict__ act_out,
    std::uint32_t used, std::uint32_t ff, std::uint32_t total_blk,
    std::uint32_t tile_blk, std::uint32_t max_m) {
  const std::uint32_t u = blockIdx.y;
  if (u >= *u_cnt) return;

  extern __shared__ __align__(8) unsigned char smem_moe_gu_q8[];
  auto* sx = reinterpret_cast<BlockQ8_1*>(smem_moe_gu_q8);

  const std::int32_t e = u_exp[u];
  const std::uint32_t cnt_raw = u_len[u];
  const std::uint32_t cnt = cnt_raw < max_m ? cnt_raw : max_m;
  const std::uint32_t* slots_u = u_slots + u * 8u;

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t b_grp = lane >> 2;
  const std::uint32_t sub = lane & 3u;
  const std::uint32_t out_row = blockIdx.x * 8u + warp_id;

  const std::uint8_t* ebase =
      gate_up + static_cast<std::size_t>(e) * expert_stride_bytes;
  const std::uint8_t* grow =
      out_row < ff ? (ebase + static_cast<std::size_t>(out_row) * row_bytes)
                   : nullptr;
  const std::uint8_t* urow =
      out_row < ff
          ? (ebase + static_cast<std::size_t>(ff + out_row) * row_bytes)
          : nullptr;

  float sg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float su[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  for (std::uint32_t b_off = 0; b_off < total_blk; b_off += tile_blk) {
    const std::uint32_t tk_blk =
        (total_blk - b_off) < tile_blk ? (total_blk - b_off) : tile_blk;
    const std::uint32_t words2 = tk_blk * 5u;
    for (std::uint32_t m = 0; m < cnt; ++m) {
      const std::uint32_t tok = slots_u[m] / used;
      const auto* src2 = reinterpret_cast<const uint2*>(
          x_q8 + static_cast<std::size_t>(tok) * total_blk + b_off);
      auto* dst2 = reinterpret_cast<uint2*>(sx + m * tile_blk);
      for (std::uint32_t i = threadIdx.x; i < words2; i += 256u) {
        dst2[i] = src2[i];
      }
    }
    __syncthreads();

    if (out_row < ff) {
      const std::size_t boff = static_cast<std::size_t>(b_off) * 18u;
      const std::uint8_t* gbase = grow + boff;
      const std::uint8_t* ubase = urow + boff;
      for (std::uint32_t b0 = 0; b0 < tk_blk; b0 += 8u) {
        const std::uint32_t b = b0 + b_grp;
        if (b < tk_blk) {
          const auto* gbp =
              reinterpret_cast<const std::uint16_t*>(gbase + b * 18u);
          const float gd4 = __half2float(*reinterpret_cast<const __half*>(gbp));
          const int gv32 = static_cast<int>(gbp[1u + 2u * sub]) |
                           (static_cast<int>(gbp[2u + 2u * sub]) << 16);
          const int gvi0 = gv32 & 0x0F0F0F0F;
          const int gvi1 = (gv32 >> 4) & 0x0F0F0F0F;

          const auto* ubp =
              reinterpret_cast<const std::uint16_t*>(ubase + b * 18u);
          const float ud4 = __half2float(*reinterpret_cast<const __half*>(ubp));
          const int uv32 = static_cast<int>(ubp[1u + 2u * sub]) |
                           (static_cast<int>(ubp[2u + 2u * sub]) << 16);
          const int uvi0 = uv32 & 0x0F0F0F0F;
          const int uvi1 = (uv32 >> 4) & 0x0F0F0F0F;

          for (std::uint32_t m = 0; m < cnt; ++m) {
            const BlockQ8_1& xblk = sx[m * tile_blk + b];
            const int xlo = xblk.qs[sub];
            const int xhi = xblk.qs[4u + sub];
            int gdot = Dot4U4S8(gvi0, xlo, 0);
            gdot = Dot4U4S8(gvi1, xhi, gdot);
            int udot = Dot4U4S8(uvi0, xlo, 0);
            udot = Dot4U4S8(uvi1, xhi, udot);
            sg[m] += gd4 * (xblk.d * static_cast<float>(gdot) - 2.0F * xblk.s);
            su[m] += ud4 * (xblk.d * static_cast<float>(udot) - 2.0F * xblk.s);
          }
        }
      }
    }
    if (b_off + tile_blk < total_blk) __syncthreads();
  }

  if (out_row < ff) {
    for (std::uint32_t m = 0; m < cnt; ++m) {
      const float g = WaveSum(sg[m]);
      const float up = WaveSum(su[m]);
      if (lane == 0u) {
        act_out[static_cast<std::size_t>(slots_u[m]) * ff + out_row] =
            GeluTanhOne(g) * up;
      }
    }
  }
}

__global__ void __launch_bounds__(256) MoeGroupedDownQ8Kernel(
    const std::uint8_t* __restrict__ down, std::uint64_t row_bytes,
    std::uint64_t expert_stride_bytes, const BlockQ8_1* __restrict__ act_q8,
    const std::uint32_t* __restrict__ u_cnt,
    const std::int32_t* __restrict__ u_exp,
    const std::uint32_t* __restrict__ u_len,
    const std::uint32_t* __restrict__ u_slots, float* __restrict__ out,
    std::uint32_t hidden, std::uint32_t total_blk, std::uint32_t tile_blk,
    std::uint32_t max_m) {
  const std::uint32_t u = blockIdx.y;
  if (u >= *u_cnt) return;

  extern __shared__ __align__(8) unsigned char smem_moe_dn_q8[];
  auto* sa = reinterpret_cast<BlockQ8_1*>(smem_moe_dn_q8);

  const std::int32_t e = u_exp[u];
  const std::uint32_t cnt_raw = u_len[u];
  const std::uint32_t cnt = cnt_raw < max_m ? cnt_raw : max_m;
  const std::uint32_t* slots_u = u_slots + u * 8u;

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t b_grp = lane >> 2;
  const std::uint32_t sub = lane & 3u;
  const std::uint32_t h = blockIdx.x * 8u + warp_id;

  const std::uint8_t* wrow =
      h < hidden ? (down + static_cast<std::size_t>(e) * expert_stride_bytes +
                    static_cast<std::size_t>(h) * row_bytes)
                 : nullptr;

  float sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (std::uint32_t b_off = 0; b_off < total_blk; b_off += tile_blk) {
    const std::uint32_t tk_blk =
        (total_blk - b_off) < tile_blk ? (total_blk - b_off) : tile_blk;
    const std::uint32_t words2 = tk_blk * 5u;
    for (std::uint32_t m = 0; m < cnt; ++m) {
      const auto* src2 = reinterpret_cast<const uint2*>(
          act_q8 + static_cast<std::size_t>(slots_u[m]) * total_blk + b_off);
      auto* dst2 = reinterpret_cast<uint2*>(sa + m * tile_blk);
      for (std::uint32_t i = threadIdx.x; i < words2; i += 256u) {
        dst2[i] = src2[i];
      }
    }
    __syncthreads();

    if (h < hidden) {
      const std::uint8_t* wbase = wrow + static_cast<std::size_t>(b_off) * 18u;
      for (std::uint32_t b0 = 0; b0 < tk_blk; b0 += 8u) {
        const std::uint32_t b = b0 + b_grp;
        if (b < tk_blk) {
          const auto* bp =
              reinterpret_cast<const std::uint16_t*>(wbase + b * 18u);
          const float d4 = __half2float(*reinterpret_cast<const __half*>(bp));
          const int v32 = static_cast<int>(bp[1u + 2u * sub]) |
                          (static_cast<int>(bp[2u + 2u * sub]) << 16);
          const int vi0 = v32 & 0x0F0F0F0F;
          const int vi1 = (v32 >> 4) & 0x0F0F0F0F;
          for (std::uint32_t m = 0; m < cnt; ++m) {
            const BlockQ8_1& ablk = sa[m * tile_blk + b];
            int dot = Dot4U4S8(vi0, ablk.qs[sub], 0);
            dot = Dot4U4S8(vi1, ablk.qs[4u + sub], dot);
            sum[m] += d4 * (ablk.d * static_cast<float>(dot) - 2.0F * ablk.s);
          }
        }
      }
    }
    if (b_off + tile_blk < total_blk) __syncthreads();
  }

  if (h < hidden) {
    for (std::uint32_t m = 0; m < cnt; ++m) {
      const float v = WaveSum(sum[m]);
      if (lane == 0u) {
        out[static_cast<std::size_t>(slots_u[m]) * hidden + h] = v;
      }
    }
  }
}

static bool g_moe_grouped_active = false;
static std::uint32_t g_moe_grouped_ntok = 8u;

}  // namespace

void MoeVecGateUpSwiglu(const void* gate_up_dev, std::uint64_t row_bytes,
                        std::uint64_t expert_stride_bytes, const float* x,
                        const std::int32_t* ids, float* act_out,
                        std::uint32_t slots, std::uint32_t used,
                        std::uint32_t ff, std::uint32_t hidden,
                        hipStream_t stream) {
  g_moe_grouped_active = false;
  if (slots == 0 || ff == 0) return;
  if (slots > used && slots <= 64u && (hidden % 32u) == 0u &&
      (ff % 32u) == 0u && getenv("GEMMA4_NO_GROUPED_VERIFY_MOE") == nullptr &&
      EnsureMoeUniqueBuffers()) {
    hipLaunchKernelGGL(MoeBuildUniqueBucketsKernel, dim3(1), dim3(256), 0,
                       stream, ids, slots, g_moe_u_cnt, g_moe_u_exp,
                       g_moe_u_len, g_moe_u_slots);
    const std::uint32_t n_tokens = slots / used;
    const std::uint32_t max_m = n_tokens < 8u ? n_tokens : 8u;
    const std::uint32_t total_blk = hidden >> 5;
    const std::uint32_t tile_blk = total_blk < 128u ? total_blk : 128u;
    if (getenv("GEMMA4_NO_Q8_BATCH") == nullptr) {
      if (const BlockQ8_1* x_q8 = QuantizeToQ8Scratch(
              x, static_cast<std::size_t>(n_tokens) * total_blk, stream)) {
        const std::size_t smem =
            static_cast<std::size_t>(max_m) * tile_blk * sizeof(BlockQ8_1);
        hipLaunchKernelGGL(
            MoeGroupedGateUpSwigluQ8Kernel, dim3((ff + 7u) / 8u, slots),
            dim3(256), smem, stream,
            static_cast<const std::uint8_t*>(gate_up_dev), row_bytes,
            expert_stride_bytes, x_q8, g_moe_u_cnt, g_moe_u_exp, g_moe_u_len,
            g_moe_u_slots, act_out, used, ff, total_blk, tile_blk, max_m);
        g_moe_grouped_active = true;
        g_moe_grouped_ntok = max_m;
        return;
      }
    }
    hipLaunchKernelGGL(
        MoeGroupedGateUpSwigluKernel, dim3((ff + 7u) / 8u, slots), dim3(256),
        0, stream, static_cast<const std::uint8_t*>(gate_up_dev), row_bytes,
        expert_stride_bytes, x, g_moe_u_cnt, g_moe_u_exp, g_moe_u_len,
        g_moe_u_slots, act_out, used, ff, hidden);
    g_moe_grouped_active = true;
    g_moe_grouped_ntok = max_m;
    return;
  }
  hipLaunchKernelGGL(MoeVecGateUpSwigluKernel, dim3(slots, (ff + 7u) / 8u),
                     dim3(256), hidden * sizeof(float), stream, gate_up_dev,
                     row_bytes, expert_stride_bytes, x, ids, act_out, used, ff,
                     hidden);
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

__global__ void MoeSlotEpilogueOverwriteKernel(
    const float* __restrict__ slot_down, const std::int32_t* __restrict__ ids,
    const float* __restrict__ wts, const float* __restrict__ dscale,
    float* __restrict__ out, std::uint32_t used, std::uint32_t dim) {
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
    dst[i] = sum;
  }
}

void MoeSlotEpilogue(const float* slot_down, const std::int32_t* ids,
                     const float* wts, const float* dscale, float* out,
                     std::uint32_t n_tokens, std::uint32_t used,
                     std::uint32_t dim, hipStream_t stream) {
  if (n_tokens == 0) return;
  hipLaunchKernelGGL(MoeSlotEpilogueKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, slot_down, ids, wts, dscale, out, used, dim);
}

void MoeSlotEpilogueOverwrite(const float* slot_down, const std::int32_t* ids,
                              const float* wts, const float* dscale, float* out,
                              std::uint32_t n_tokens, std::uint32_t used,
                              std::uint32_t dim, hipStream_t stream) {
  if (n_tokens == 0) return;
  hipLaunchKernelGGL(MoeSlotEpilogueOverwriteKernel, dim3(n_tokens),
                     dim3(kThreads), 0, stream, slot_down, ids, wts, dscale, out,
                     used, dim);
}

// Fused Q4_0 down GEMV for the decode MoE: expert row selected per slot from
// the DEVICE ids array (negative -> output 0).
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
    const auto* bp = reinterpret_cast<const std::uint16_t*>(wrow + b * 18);
    const float d = __half2float(*reinterpret_cast<const __half*>(bp));
    const float* xv = ar + b * 32;
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      const std::uint32_t q01 = bp[1 + 2 * sub];
      const std::uint32_t q23 = bp[2 + 2 * sub];
      sum += d * static_cast<float>(static_cast<int>(q01 & 0xFu) - 8) * xv[4 * sub + 0];
      sum += d * static_cast<float>(static_cast<int>((q01 >> 4) & 0xFu) - 8) * xv[4 * sub + 16];
      sum += d * static_cast<float>(static_cast<int>((q01 >> 8) & 0xFu) - 8) * xv[4 * sub + 1];
      sum += d * static_cast<float>(static_cast<int>((q01 >> 12) & 0xFu) - 8) * xv[4 * sub + 17];
      sum += d * static_cast<float>(static_cast<int>(q23 & 0xFu) - 8) * xv[4 * sub + 2];
      sum += d * static_cast<float>(static_cast<int>((q23 >> 4) & 0xFu) - 8) * xv[4 * sub + 18];
      sum += d * static_cast<float>(static_cast<int>((q23 >> 8) & 0xFu) - 8) * xv[4 * sub + 3];
      sum += d * static_cast<float>(static_cast<int>((q23 >> 12) & 0xFu) - 8) * xv[4 * sub + 19];
    }
  }
  sum = WaveSum(sum);
  __shared__ float shared[kThreads / 32];
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : 0.0F;
    v = WaveSum(v);
    if (threadIdx.x == 0) out[orow] = v;
  }
}

__global__ void __launch_bounds__(256) GemvQSlotWave32Kernel(
    const void* __restrict__ down, std::uint64_t row_bytes,
    std::uint64_t expert_stride_bytes, const float* __restrict__ act,
    const std::int32_t* __restrict__ ids, float* __restrict__ out,
    std::uint32_t hidden, std::uint32_t ff) {
  const std::uint32_t slot = blockIdx.x;
  const std::int32_t e = ids[slot];
  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t h = blockIdx.y * 8u + warp_id;
  if (e < 0) {
    if (h < hidden && lane == 0) {
      out[static_cast<std::size_t>(slot) * hidden + h] = 0.0F;
    }
    return;
  }
  extern __shared__ float sa[];
  const float* ar = act + static_cast<std::size_t>(slot) * ff;
  for (std::uint32_t i = threadIdx.x * 4u; i < ff; i += 256u * 4u) {
    *reinterpret_cast<float4*>(sa + i) =
        *reinterpret_cast<const float4*>(ar + i);
  }
  __syncthreads();
  if (h >= hidden) return;

  const auto* wrow = static_cast<const std::uint8_t*>(down) +
                     static_cast<std::size_t>(e) * expert_stride_bytes +
                     static_cast<std::size_t>(h) * row_bytes;
  float sum = WaveRowTileDot(wrow, 2, 0u, ff, lane, sa);
  sum = WaveSum(sum);
  if (lane == 0) {
    out[static_cast<std::size_t>(slot) * hidden + h] = sum;
  }
}

void GemvQSlot(const void* down_dev, std::uint64_t row_bytes,
               std::uint64_t expert_stride_bytes, const float* act,
               const std::int32_t* ids, float* out, std::uint32_t slots,
               std::uint32_t hidden, std::uint32_t ff, hipStream_t stream) {
  if (slots == 0 || hidden == 0) return;
  if (g_moe_grouped_active && slots <= 64u && (ff % 32u) == 0u &&
      getenv("GEMMA4_NO_WAVE32_GEMV") == nullptr) {
    g_moe_grouped_active = false;
    const std::uint32_t max_m = g_moe_grouped_ntok < 8u ? g_moe_grouped_ntok : 8u;
    const std::uint32_t total_blk = ff >> 5;
    const std::uint32_t tile_blk = total_blk < 128u ? total_blk : 128u;
    if (getenv("GEMMA4_NO_Q8_BATCH") == nullptr) {
      if (const BlockQ8_1* act_q8 = QuantizeToQ8Scratch(
              act, static_cast<std::size_t>(slots) * total_blk, stream)) {
        const std::size_t smem =
            static_cast<std::size_t>(max_m) * tile_blk * sizeof(BlockQ8_1);
        hipLaunchKernelGGL(
            MoeGroupedDownQ8Kernel, dim3((hidden + 7u) / 8u, slots), dim3(256),
            smem, stream, static_cast<const std::uint8_t*>(down_dev), row_bytes,
            expert_stride_bytes, act_q8, g_moe_u_cnt, g_moe_u_exp, g_moe_u_len,
            g_moe_u_slots, out, hidden, total_blk, tile_blk, max_m);
        return;
      }
    }
    hipLaunchKernelGGL(
        MoeGroupedDownKernel, dim3((hidden + 7u) / 8u, slots), dim3(256), 0,
        stream, static_cast<const std::uint8_t*>(down_dev), row_bytes,
        expert_stride_bytes, act, g_moe_u_cnt, g_moe_u_exp, g_moe_u_len,
        g_moe_u_slots, out, hidden, ff);
    return;
  }
  g_moe_grouped_active = false;
  if ((ff % 32u) == 0u && ff <= 8192u &&
      getenv("GEMMA4_NO_WAVE32_GEMV") == nullptr) {
    hipLaunchKernelGGL(GemvQSlotWave32Kernel, dim3(slots, (hidden + 7u) / 8u),
                       dim3(256), ff * sizeof(float), stream, down_dev,
                       row_bytes, expert_stride_bytes, act, ids, out, hidden,
                       ff);
    return;
  }
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

void GemvBatched(const void* wdev, int type, std::uint64_t row_bytes,
                 std::uint64_t cols, const float* x, float* y,
                 std::uint32_t n_tokens, std::uint32_t m, float softcap,
                 hipStream_t stream) {
  if (m == 0 || n_tokens == 0) return;
  const bool type_ok =
      (type == 0 || type == 1 || type == 2 || type == 8 || type == 30);
  if (type_ok && (cols % 32ull) == 0ull && n_tokens <= 8u &&
      getenv("GEMMA4_NO_WAVE32_GEMV") == nullptr) {
    const dim3 grid((m + 7u) / 8u);
    const dim3 block(256);
    const bool q8_batch_ok = (getenv("GEMMA4_NO_Q8_BATCH") == nullptr);
    const bool q8_single = (getenv("GEMMA4_Q8_SINGLE") != nullptr);
    if (type == 8 && q8_batch_ok &&
        (n_tokens > 1u || getenv("GEMMA4_NO_Q8_SINGLE_HEAD") == nullptr)) {
      const std::uint32_t total_blk = static_cast<std::uint32_t>(cols >> 5);
      const std::uint32_t tile_blk = total_blk < 128u ? total_blk : 128u;
      if (const BlockQ8_1* x_q8 = QuantizeToQ8Scratch(
              x, static_cast<std::size_t>(n_tokens) * total_blk, stream)) {
        const std::size_t smem =
            static_cast<std::size_t>(n_tokens) * tile_blk * sizeof(BlockQ8_1);
        hipLaunchKernelGGL(
            GemvWave32BatchedQ8_0Kernel, grid, block, smem, stream,
            static_cast<const std::uint8_t*>(wdev), row_bytes, total_blk,
            tile_blk, x_q8, y, n_tokens, m, softcap);
        return;
      }
    }
    if (n_tokens == 1u && cols <= 12288ull &&
        !(type == 2 && q8_batch_ok && (q8_single || m >= 65536u))) {
      hipLaunchKernelGGL(
          GemvWave32SingleKernel, grid, block,
          static_cast<std::size_t>(cols) * sizeof(float), stream,
          static_cast<const std::uint8_t*>(wdev), type, row_bytes,
          static_cast<std::uint32_t>(cols), x, y, m, softcap);
      return;
    }
    if (type == 2 && q8_batch_ok) {
      const std::uint32_t total_blk = static_cast<std::uint32_t>(cols >> 5);
      const std::uint32_t tile_blk = total_blk < 128u ? total_blk : 128u;
      if (const BlockQ8_1* x_q8 = QuantizeToQ8Scratch(
              x, static_cast<std::size_t>(n_tokens) * total_blk, stream)) {
        const std::size_t smem =
            static_cast<std::size_t>(n_tokens) * tile_blk * sizeof(BlockQ8_1);
        hipLaunchKernelGGL(
            GemvWave32BatchedQ8Kernel, grid, block, smem, stream,
            static_cast<const std::uint8_t*>(wdev), row_bytes, total_blk,
            tile_blk, x_q8, y, n_tokens, m, softcap);
        return;
      }
    }
    hipLaunchKernelGGL(
        GemvWave32BatchedKernel, grid, block, 0, stream,
        static_cast<const std::uint8_t*>(wdev), type, row_bytes,
        static_cast<std::uint32_t>(cols), x, y, n_tokens, m, softcap);
    return;
  }
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    hipLaunchKernelGGL(GemvQKernel, dim3(m), dim3(kThreads), 0, stream, wdev,
                       type, row_bytes, cols,
                       x + static_cast<std::size_t>(t) * cols,
                       y + static_cast<std::size_t>(t) * m);
  }
  if (softcap > 0.0F) {
    SoftCap(y, softcap, static_cast<std::size_t>(n_tokens) * m, stream);
  }
}

void GemvGateUpSwiglu(const void* w_gate, const void* w_up, int type,
                      std::uint64_t row_bytes, std::uint64_t cols,
                      const float* x, float* out_act, std::uint32_t n_tokens,
                      std::uint32_t m, hipStream_t stream) {
  if (m == 0 || n_tokens == 0) return;
  const bool type_ok =
      (type == 0 || type == 1 || type == 2 || type == 8 || type == 30);
  if (type_ok && (cols % 32ull) == 0ull && n_tokens <= 8u &&
      getenv("GEMMA4_NO_WAVE32_GEMV") == nullptr) {
    const dim3 grid((m + 7u) / 8u);
    const dim3 block(256);
    const bool q8_batch_ok = (getenv("GEMMA4_NO_Q8_BATCH") == nullptr);
    const bool q8_single = (getenv("GEMMA4_Q8_SINGLE") != nullptr);
    if (n_tokens == 1u && cols <= 12288ull &&
        !(type == 2 && q8_batch_ok && q8_single)) {
      hipLaunchKernelGGL(
          GemvGateUpSwigluSingleKernel, grid, block,
          static_cast<std::size_t>(cols) * sizeof(float), stream,
          static_cast<const std::uint8_t*>(w_gate),
          static_cast<const std::uint8_t*>(w_up), type, row_bytes,
          static_cast<std::uint32_t>(cols), x, out_act, m);
      return;
    }
    if (type == 2 && q8_batch_ok) {
      const std::uint32_t total_blk = static_cast<std::uint32_t>(cols >> 5);
      const std::uint32_t tile_blk = total_blk < 128u ? total_blk : 128u;
      if (const BlockQ8_1* x_q8 = QuantizeToQ8Scratch(
              x, static_cast<std::size_t>(n_tokens) * total_blk, stream)) {
        const std::size_t smem =
            static_cast<std::size_t>(n_tokens) * tile_blk * sizeof(BlockQ8_1);
        hipLaunchKernelGGL(
            GemvGateUpSwigluBatchedQ8Kernel, grid, block, smem, stream,
            static_cast<const std::uint8_t*>(w_gate),
            static_cast<const std::uint8_t*>(w_up), row_bytes, total_blk,
            tile_blk, x_q8, out_act, n_tokens, m);
        return;
      }
    }
    hipLaunchKernelGGL(
        GemvGateUpSwigluBatchedKernel, grid, block, 0, stream,
        static_cast<const std::uint8_t*>(w_gate),
        static_cast<const std::uint8_t*>(w_up), type, row_bytes,
        static_cast<std::uint32_t>(cols), x, out_act, n_tokens, m);
    return;
  }
  // Fallback should not be reached on Gemma 4 shapes, but keep safe.
}

void GemvFused(const void* wdev, int type, std::uint64_t row_bytes,
               std::uint64_t cols, const float* x, float* y, std::uint32_t m,
               hipStream_t stream) {
  GemvBatched(wdev, type, row_bytes, cols, x, y, 1u, m, 0.0F, stream);
}

void GemvF16Fused(const __half* wdev, std::uint64_t cols, const float* x,
                  float* y, std::uint32_t m, hipStream_t stream) {
  GemvBatched(wdev, 1, cols * sizeof(__half), cols, x, y, 1u, m, 0.0F, stream);
}

namespace {

__device__ __forceinline__ float BlockReduceSum256(float sum, float* s_red) {
  sum = WaveSum(sum);
  if ((threadIdx.x & 31u) == 0u) s_red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32u) {
    float v = (threadIdx.x < 8u) ? s_red[threadIdx.x] : 0.0F;
    v = WaveSum(v);
    if (threadIdx.x == 0u) s_red[0] = v;
  }
  __syncthreads();
  const float total = s_red[0];
  __syncthreads();
  return total;
}

__global__ void __launch_bounds__(256) GemvQkvSingleKernel(
    const std::uint8_t* __restrict__ w_q, int type_q, std::uint64_t rb_q,
    std::uint32_t qdim, const std::uint8_t* __restrict__ w_k, int type_k,
    std::uint64_t rb_k, const std::uint8_t* __restrict__ w_v, int type_v,
    std::uint64_t rb_v, std::uint32_t kvdim, std::uint32_t total_m,
    std::uint32_t cols, const float* __restrict__ x,
    float* __restrict__ q_out, float* __restrict__ k_out,
    float* __restrict__ v_out) {
  extern __shared__ float sx[];
  for (std::uint32_t i = threadIdx.x * 4u; i < cols; i += 256u * 4u) {
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
  }
  __syncthreads();

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  if (row >= total_m) return;

  const std::uint8_t* wrow = nullptr;
  int tp = 2;
  float* dst = nullptr;
  if (row < qdim) {
    wrow = w_q + static_cast<std::size_t>(row) * rb_q;
    tp = type_q;
    dst = q_out + row;
  } else if (row < qdim + kvdim) {
    const std::uint32_t lr = row - qdim;
    wrow = w_k + static_cast<std::size_t>(lr) * rb_k;
    tp = type_k;
    dst = k_out + lr;
  } else {
    const std::uint32_t lr = row - qdim - kvdim;
    wrow = w_v + static_cast<std::size_t>(lr) * rb_v;
    tp = type_v;
    dst = v_out + lr;
  }

  float sum = WaveRowTileDot(wrow, tp, 0u, cols, lane, sx);
  sum = WaveSum(sum);
  if (lane == 0u) *dst = sum;
}

__global__ void __launch_bounds__(256) GemvQkvBatchedKernel(
    const std::uint8_t* __restrict__ w_q, int type_q, std::uint64_t rb_q,
    std::uint32_t qdim, const std::uint8_t* __restrict__ w_k, int type_k,
    std::uint64_t rb_k, const std::uint8_t* __restrict__ w_v, int type_v,
    std::uint64_t rb_v, std::uint32_t kvdim, std::uint32_t total_m,
    std::uint32_t cols, const float* __restrict__ x,
    float* __restrict__ q_out, float* __restrict__ k_out,
    float* __restrict__ v_out, std::uint32_t n_tokens) {
  constexpr std::uint32_t kTileK = 1024;
  __shared__ float sx[8][kTileK];

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;

  const std::uint8_t* wrow = nullptr;
  int tp = 2;
  float* out_base = nullptr;
  std::uint32_t out_stride = 0;
  std::uint32_t local_row = 0;
  if (row < qdim) {
    local_row = row;
    wrow = w_q + static_cast<std::size_t>(local_row) * rb_q;
    tp = type_q;
    out_base = q_out;
    out_stride = qdim;
  } else if (row < qdim + kvdim) {
    local_row = row - qdim;
    wrow = w_k + static_cast<std::size_t>(local_row) * rb_k;
    tp = type_k;
    out_base = k_out;
    out_stride = kvdim;
  } else if (row < total_m) {
    local_row = row - qdim - kvdim;
    wrow = w_v + static_cast<std::size_t>(local_row) * rb_v;
    tp = type_v;
    out_base = v_out;
    out_stride = kvdim;
  }

  float sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (std::uint32_t k0 = 0; k0 < cols; k0 += kTileK) {
    const std::uint32_t tk = (cols - k0) < kTileK ? (cols - k0) : kTileK;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float* xt = x + static_cast<std::size_t>(t) * cols + k0;
      for (std::uint32_t i = threadIdx.x * 4u; i < tk; i += 256u * 4u) {
        *reinterpret_cast<float4*>(&sx[t][i]) =
            *reinterpret_cast<const float4*>(xt + i);
      }
    }
    __syncthreads();
    if (row < total_m) {
      if (tp == 2) {
        const std::uint32_t nblk = tk >> 5;
        const std::uint32_t b_grp = lane >> 2;
        const std::uint32_t sub = lane & 3u;
        const std::uint8_t* wbase =
            wrow + static_cast<std::size_t>(k0 >> 5) * 18;
        for (std::uint32_t b0 = 0; b0 < nblk; b0 += 8) {
          const std::uint32_t b = b0 + b_grp;
          if (b < nblk) {
            const auto* bp =
                reinterpret_cast<const std::uint16_t*>(wbase + b * 18u);
            const float d = __half2float(*reinterpret_cast<const __half*>(bp));
            const std::uint32_t q01 = bp[1 + 2 * sub];
            const std::uint32_t q23 = bp[2 + 2 * sub];
            const float wl0 = static_cast<float>(static_cast<int>(q01 & 0xFu) - 8);
            const float wh0 = static_cast<float>(static_cast<int>((q01 >> 4) & 0xFu) - 8);
            const float wl1 = static_cast<float>(static_cast<int>((q01 >> 8) & 0xFu) - 8);
            const float wh1 = static_cast<float>(static_cast<int>((q01 >> 12) & 0xFu) - 8);
            const float wl2 = static_cast<float>(static_cast<int>(q23 & 0xFu) - 8);
            const float wh2 = static_cast<float>(static_cast<int>((q23 >> 4) & 0xFu) - 8);
            const float wl3 = static_cast<float>(static_cast<int>((q23 >> 8) & 0xFu) - 8);
            const float wh3 = static_cast<float>(static_cast<int>((q23 >> 12) & 0xFu) - 8);
            for (std::uint32_t t = 0; t < n_tokens; ++t) {
              const float4 xlo = *reinterpret_cast<const float4*>(
                  &sx[t][b * 32u + 4u * sub]);
              const float4 xhi = *reinterpret_cast<const float4*>(
                  &sx[t][b * 32u + 16u + 4u * sub]);
              const float acc = wl0 * xlo.x + wh0 * xhi.x + wl1 * xlo.y +
                                wh1 * xhi.y + wl2 * xlo.z + wh2 * xhi.z +
                                wl3 * xlo.w + wh3 * xhi.w;
              sum[t] += d * acc;
            }
          }
        }
      } else {
        for (std::uint32_t t = 0; t < n_tokens; ++t) {
          sum[t] += WaveRowTileDot(wrow, tp, k0, tk, lane, sx[t]);
        }
      }
    }
    if (k0 + kTileK < cols) __syncthreads();
  }

  if (row < total_m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float v = WaveSum(sum[t]);
      if (lane == 0u) {
        out_base[static_cast<std::size_t>(t) * out_stride + local_row] = v;
      }
    }
  }
}

__global__ void __launch_bounds__(256) GemvQkvBatchedQ8Kernel(
    const std::uint8_t* __restrict__ w_q, std::uint64_t rb_q,
    std::uint32_t qdim, const std::uint8_t* __restrict__ w_k,
    std::uint64_t rb_k, const std::uint8_t* __restrict__ w_v,
    std::uint64_t rb_v, std::uint32_t kvdim, std::uint32_t total_m,
    std::uint32_t total_blk, std::uint32_t tile_blk,
    const BlockQ8_1* __restrict__ x_q8, float* __restrict__ q_out,
    float* __restrict__ k_out, float* __restrict__ v_out,
    std::uint32_t n_tokens) {
  extern __shared__ __align__(8) unsigned char smem_qkv_q8[];
  auto* sx = reinterpret_cast<BlockQ8_1*>(smem_qkv_q8);

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t b_grp = lane >> 2;
  const std::uint32_t sub = lane & 3u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;

  const std::uint8_t* wrow = nullptr;
  float* out_base = nullptr;
  std::uint32_t out_stride = 0;
  std::uint32_t local_row = 0;
  if (row < qdim) {
    local_row = row;
    wrow = w_q + static_cast<std::size_t>(local_row) * rb_q;
    out_base = q_out;
    out_stride = qdim;
  } else if (row < qdim + kvdim) {
    local_row = row - qdim;
    wrow = w_k + static_cast<std::size_t>(local_row) * rb_k;
    out_base = k_out;
    out_stride = kvdim;
  } else if (row < total_m) {
    local_row = row - qdim - kvdim;
    wrow = w_v + static_cast<std::size_t>(local_row) * rb_v;
    out_base = v_out;
    out_stride = kvdim;
  }

  float sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (std::uint32_t b_off = 0; b_off < total_blk; b_off += tile_blk) {
    const std::uint32_t tk_blk =
        (total_blk - b_off) < tile_blk ? (total_blk - b_off) : tile_blk;
    const std::uint32_t words2 = tk_blk * 5u;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const auto* src2 = reinterpret_cast<const uint2*>(
          x_q8 + static_cast<std::size_t>(t) * total_blk + b_off);
      auto* dst2 = reinterpret_cast<uint2*>(sx + t * tile_blk);
      for (std::uint32_t i = threadIdx.x; i < words2; i += 256u) {
        dst2[i] = src2[i];
      }
    }
    __syncthreads();

    if (row < total_m) {
      const std::uint8_t* wbase = wrow + static_cast<std::size_t>(b_off) * 18u;
      for (std::uint32_t b0 = 0; b0 < tk_blk; b0 += 8u) {
        const std::uint32_t b = b0 + b_grp;
        if (b < tk_blk) {
          const auto* bp =
              reinterpret_cast<const std::uint16_t*>(wbase + b * 18u);
          const float d4 = __half2float(*reinterpret_cast<const __half*>(bp));
          const int v32 = static_cast<int>(bp[1u + 2u * sub]) |
                          (static_cast<int>(bp[2u + 2u * sub]) << 16);
          const int vi0 = v32 & 0x0F0F0F0F;
          const int vi1 = (v32 >> 4) & 0x0F0F0F0F;
          for (std::uint32_t t = 0; t < n_tokens; ++t) {
            const BlockQ8_1& xblk = sx[t * tile_blk + b];
            int dot = Dot4U4S8(vi0, xblk.qs[sub], 0);
            dot = Dot4U4S8(vi1, xblk.qs[4u + sub], dot);
            sum[t] += d4 * (xblk.d * static_cast<float>(dot) - 2.0F * xblk.s);
          }
        }
      }
    }
    if (b_off + tile_blk < total_blk) __syncthreads();
  }

  if (row < total_m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float v = WaveSum(sum[t]);
      if (lane == 0u) {
        out_base[static_cast<std::size_t>(t) * out_stride + local_row] = v;
      }
    }
  }
}

__global__ void __launch_bounds__(256) PrepAttnQkvKernel(
    float* __restrict__ qbuf, const float* __restrict__ kbuf,
    const float* __restrict__ vbuf, const float* __restrict__ q_norm,
    const float* __restrict__ k_norm, const float* __restrict__ v_norm,
    __half* __restrict__ kv_k, __half* __restrict__ kv_v,
    std::uint32_t num_heads, std::uint32_t kv_heads, bool has_v_proj,
    std::uint32_t head_dim, std::uint32_t rotary_dim, std::uint32_t start_pos,
    float theta, const float* __restrict__ freq_factors, float eps) {
  __shared__ float s_red[8];
  __shared__ float sk[512];

  const std::uint32_t g = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t half = rotary_dim >> 1;
  const std::uint32_t my_pos = start_pos + t;

  if (g < num_heads) {
    float* row =
        qbuf + (static_cast<std::size_t>(t) * num_heads + g) * head_dim;
    float sum = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < head_dim; i += 256u) {
      const float v = row[i];
      sum += v * v;
    }
    const float scale =
        rsqrtf(BlockReduceSum256(sum, s_red) / static_cast<float>(head_dim) + eps);
    if (threadIdx.x < half) {
      const std::uint32_t i = threadIdx.x;
      const float x0 = row[i] * scale * (q_norm != nullptr ? q_norm[i] : 1.0F);
      const float x1 = row[i + half] * scale *
                       (q_norm != nullptr ? q_norm[i + half] : 1.0F);
      float freq;
      if (freq_factors != nullptr) {
        const float inv =
            1.0F / powf(theta, static_cast<float>(2u * i) /
                                   static_cast<float>(rotary_dim));
        freq = inv / freq_factors[i];
      } else {
        freq = 1.0F / powf(theta, static_cast<float>(i) / static_cast<float>(half));
      }
      float s, c;
      sincosf(static_cast<float>(my_pos) * freq, &s, &c);
      row[i] = x0 * c - x1 * s;
      row[i + half] = x0 * s + x1 * c;
    }
    for (std::uint32_t i = rotary_dim + threadIdx.x; i < head_dim; i += 256u) {
      row[i] = row[i] * scale * (q_norm != nullptr ? q_norm[i] : 1.0F);
    }
    return;
  }

  const std::uint32_t kvdim = kv_heads * head_dim;
  if (g < num_heads + kv_heads) {
    const std::uint32_t kh = g - num_heads;
    const float* k_row =
        kbuf + (static_cast<std::size_t>(t) * kv_heads + kh) * head_dim;
    __half* k_dst =
        kv_k + static_cast<std::size_t>(my_pos) * kvdim + kh * head_dim;
    float sum_k = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < head_dim; i += 256u) {
      const float v = k_row[i];
      sum_k += v * v;
    }
    const float scale_k =
        rsqrtf(BlockReduceSum256(sum_k, s_red) / static_cast<float>(head_dim) + eps);
    if (has_v_proj) {
      if (threadIdx.x < half) {
        const std::uint32_t i = threadIdx.x;
        const float x0 =
            k_row[i] * scale_k * (k_norm != nullptr ? k_norm[i] : 1.0F);
        const float x1 = k_row[i + half] * scale_k *
                         (k_norm != nullptr ? k_norm[i + half] : 1.0F);
        float freq;
        if (freq_factors != nullptr) {
          const float inv =
              1.0F / powf(theta, static_cast<float>(2u * i) /
                                     static_cast<float>(rotary_dim));
          freq = inv / freq_factors[i];
        } else {
          freq = 1.0F /
                 powf(theta, static_cast<float>(i) / static_cast<float>(half));
        }
        float s, c;
        sincosf(static_cast<float>(my_pos) * freq, &s, &c);
        k_dst[i] = __float2half(x0 * c - x1 * s);
        k_dst[i + half] = __float2half(x0 * s + x1 * c);
      }
      for (std::uint32_t i = rotary_dim + threadIdx.x; i < head_dim; i += 256u) {
        k_dst[i] = __float2half(
            k_row[i] * scale_k * (k_norm != nullptr ? k_norm[i] : 1.0F));
      }
    } else {
      __half* v_dst =
          kv_v + static_cast<std::size_t>(my_pos) * kvdim + kh * head_dim;
      float sum_v = 0.0F;
      for (std::uint32_t i = threadIdx.x; i < head_dim; i += 256u) {
        const float kn =
            k_row[i] * scale_k * (k_norm != nullptr ? k_norm[i] : 1.0F);
        sk[i] = kn;
        sum_v += kn * kn;
      }
      const float scale_v =
          rsqrtf(BlockReduceSum256(sum_v, s_red) / static_cast<float>(head_dim) +
                 eps);
      for (std::uint32_t i = threadIdx.x; i < head_dim; i += 256u) {
        v_dst[i] = __float2half(
            sk[i] * scale_v * (v_norm != nullptr ? v_norm[i] : 1.0F));
      }
      if (threadIdx.x < half) {
        const std::uint32_t i = threadIdx.x;
        const float x0 = sk[i];
        const float x1 = sk[i + half];
        float freq;
        if (freq_factors != nullptr) {
          const float inv =
              1.0F / powf(theta, static_cast<float>(2u * i) /
                                     static_cast<float>(rotary_dim));
          freq = inv / freq_factors[i];
        } else {
          freq = 1.0F /
                 powf(theta, static_cast<float>(i) / static_cast<float>(half));
        }
        float s, c;
        sincosf(static_cast<float>(my_pos) * freq, &s, &c);
        k_dst[i] = __float2half(x0 * c - x1 * s);
        k_dst[i + half] = __float2half(x0 * s + x1 * c);
      }
      for (std::uint32_t i = rotary_dim + threadIdx.x; i < head_dim; i += 256u) {
        k_dst[i] = __float2half(sk[i]);
      }
    }
    return;
  }

  const std::uint32_t vh = g - num_heads - kv_heads;
  const float* v_row =
      vbuf + (static_cast<std::size_t>(t) * kv_heads + vh) * head_dim;
  __half* v_dst =
      kv_v + static_cast<std::size_t>(my_pos) * kvdim + vh * head_dim;
  float sum_v = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += 256u) {
    const float v = v_row[i];
    sum_v += v * v;
  }
  const float scale_v =
      rsqrtf(BlockReduceSum256(sum_v, s_red) / static_cast<float>(head_dim) + eps);
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += 256u) {
    v_dst[i] =
        __float2half(v_row[i] * scale_v * (v_norm != nullptr ? v_norm[i] : 1.0F));
  }
}

__global__ void __launch_bounds__(256) FfnTripleNormKernel(
    const float* __restrict__ x, const float* __restrict__ gamma_ffn,
    const float* __restrict__ gamma_pre2,
    const float* __restrict__ router_scale, float* __restrict__ out_ffn,
    float* __restrict__ out_pre2, float* __restrict__ out_router,
    std::uint32_t dim, float eps) {
  __shared__ float s_red[8];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * dim;
  const float* src = x + base;
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += 256u) {
    const float v = src[i];
    sum += v * v;
  }
  const float scale =
      rsqrtf(BlockReduceSum256(sum, s_red) / static_cast<float>(dim) + eps);
  const float inv_router = scale * rsqrtf(static_cast<float>(dim));
  float* d_ffn = out_ffn + base;
  float* d_pre2 = out_pre2 + base;
  float* d_rtr = out_router + base;
  for (std::uint32_t i = threadIdx.x; i < dim; i += 256u) {
    const float xi = src[i];
    const float nx = xi * scale;
    d_ffn[i] = nx * (gamma_ffn != nullptr ? gamma_ffn[i] : 1.0F);
    d_pre2[i] = nx * (gamma_pre2 != nullptr ? gamma_pre2[i] : 1.0F);
    d_rtr[i] = xi * inv_router * router_scale[i];
  }
}

__global__ void __launch_bounds__(256) MoeCombinePostNormResidualKernel(
    const float* __restrict__ slot_down, const std::int32_t* __restrict__ ids,
    const float* __restrict__ wts, const float* __restrict__ dscale,
    const float* __restrict__ shared_down,
    const float* __restrict__ gamma_post1,
    const float* __restrict__ gamma_post2, const float* __restrict__ gamma_post,
    float* __restrict__ residual_inout, std::uint32_t used, std::uint32_t dim,
    float eps, float post_scale) {
  extern __shared__ float s_vec[];
  __shared__ float s_ws[32];
  __shared__ float s_red1[8];
  __shared__ float s_red2[8];

  const std::uint32_t t = blockIdx.x;
  const std::size_t base = static_cast<std::size_t>(t) * dim;
  if (threadIdx.x < used && threadIdx.x < 32u) {
    const std::uint32_t s = threadIdx.x;
    const std::uint32_t r = t * used + s;
    const std::int32_t e = ids[r];
    if (e >= 0) {
      const float ds = (dscale != nullptr) ? dscale[e] : 1.0F;
      s_ws[s] = wts[r] * ds;
    } else {
      s_ws[s] = 0.0F;
    }
  }
  __syncthreads();

  const float* sh_row = shared_down + base;
  const float* sd_base =
      slot_down + static_cast<std::size_t>(t) * used * dim;
  float sq_routed = 0.0F;
  float sq_shared = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += 256u) {
    float rsum = 0.0F;
    for (std::uint32_t s = 0; s < used; ++s) {
      const float ws = s_ws[s];
      if (ws != 0.0F) {
        rsum += ws * sd_base[static_cast<std::size_t>(s) * dim + i];
      }
    }
    s_vec[i] = rsum;
    sq_routed += rsum * rsum;
    const float sv = sh_row[i];
    sq_shared += sv * sv;
  }

  sq_routed = WaveSum(sq_routed);
  sq_shared = WaveSum(sq_shared);
  if ((threadIdx.x & 31u) == 0u) {
    s_red1[threadIdx.x >> 5] = sq_routed;
    s_red2[threadIdx.x >> 5] = sq_shared;
  }
  __syncthreads();
  if (threadIdx.x < 32u) {
    float v1 = (threadIdx.x < 8u) ? s_red1[threadIdx.x] : 0.0F;
    float v2 = (threadIdx.x < 8u) ? s_red2[threadIdx.x] : 0.0F;
    v1 = WaveSum(v1);
    v2 = WaveSum(v2);
    if (threadIdx.x == 0u) {
      s_red1[0] = v1;
      s_red2[0] = v2;
    }
  }
  __syncthreads();
  const float s1_tot = s_red1[0];
  const float s2_tot = s_red2[0];
  __syncthreads();

  const float scale_routed =
      rsqrtf(s1_tot / static_cast<float>(dim) + eps);
  const float scale_shared =
      rsqrtf(s2_tot / static_cast<float>(dim) + eps);

  float sq_comb = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += 256u) {
    const float c =
        sh_row[i] * scale_shared * (gamma_post1 != nullptr ? gamma_post1[i] : 1.0F) +
        s_vec[i] * scale_routed * (gamma_post2 != nullptr ? gamma_post2[i] : 1.0F);
    s_vec[i] = c;
    sq_comb += c * c;
  }
  const float scale_post =
      rsqrtf(BlockReduceSum256(sq_comb, s_red1) / static_cast<float>(dim) + eps);

  float* dst = residual_inout + base;
  if (post_scale == 1.0F) {
    for (std::uint32_t i = threadIdx.x; i < dim; i += 256u) {
      dst[i] += s_vec[i] * scale_post *
                (gamma_post != nullptr ? gamma_post[i] : 1.0F);
    }
  } else {
    for (std::uint32_t i = threadIdx.x; i < dim; i += 256u) {
      dst[i] = (dst[i] + s_vec[i] * scale_post *
                             (gamma_post != nullptr ? gamma_post[i] : 1.0F)) *
               post_scale;
    }
  }
}

__global__ void __launch_bounds__(256) BuildRoutedTilesKernel(
    const std::int32_t* __restrict__ routed_bounds,
    std::int32_t* __restrict__ q4_tiles, std::uint32_t* __restrict__ d_meta,
    std::uint32_t num_experts, std::uint32_t tile_rows) {
  __shared__ std::uint32_t s_cnt[512];
  __shared__ std::uint32_t s_off[512];
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lim = num_experts < 512u ? num_experts : 512u;
  for (std::uint32_t e = tid; e < lim; e += 256u) {
    const std::int32_t b0 = routed_bounds[e];
    const std::int32_t b1 = routed_bounds[e + 1u];
    const std::uint32_t rows_e =
        (b1 > b0) ? static_cast<std::uint32_t>(b1 - b0) : 0u;
    s_cnt[e] = (rows_e + tile_rows - 1u) / tile_rows;
  }
  __syncthreads();
  if (tid == 0u) {
    std::uint32_t acc = 0u;
    for (std::uint32_t e = 0u; e < lim; ++e) {
      s_off[e] = acc;
      acc += s_cnt[e];
    }
    d_meta[0] = acc;
    const std::int32_t total_rows = routed_bounds[num_experts];
    d_meta[1] = total_rows > 0 ? static_cast<std::uint32_t>(total_rows) : 0u;
  }
  __syncthreads();
  for (std::uint32_t e = tid; e < lim; e += 256u) {
    const std::uint32_t n_t = s_cnt[e];
    const std::uint32_t base = s_off[e];
    for (std::uint32_t j = 0u; j < n_t; ++j) {
      q4_tiles[base + j] = static_cast<std::int32_t>(e | (j << 16));
    }
  }
}

}  // namespace

void GemvQkv(const void* w_q, int type_q, std::uint64_t rb_q,
             std::uint32_t qdim, const void* w_k, int type_k,
             std::uint64_t rb_k, const void* w_v, int type_v,
             std::uint64_t rb_v, std::uint32_t kvdim, std::uint64_t cols,
             const float* x, float* q_out, float* k_out, float* v_out,
             std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == 0) return;
  const std::uint32_t total_m =
      qdim + kvdim + (w_v != nullptr ? kvdim : 0u);
  if ((cols % 32ull) == 0ull && n_tokens <= 8u &&
      getenv("GEMMA4_NO_WAVE32_GEMV") == nullptr) {
    const dim3 grid((total_m + 7u) / 8u);
    const dim3 block(256);
    const bool all_q4 =
        (type_q == 2 && type_k == 2 && (w_v == nullptr || type_v == 2));
    const bool q8_batch_ok = (getenv("GEMMA4_NO_Q8_BATCH") == nullptr);
    const bool q8_single = (getenv("GEMMA4_Q8_SINGLE") != nullptr);
    if (n_tokens == 1u && cols <= 12288ull &&
        !(all_q4 && q8_batch_ok && (q8_single || total_m >= 65536u))) {
      hipLaunchKernelGGL(
          GemvQkvSingleKernel, grid, block,
          static_cast<std::size_t>(cols) * sizeof(float), stream,
          static_cast<const std::uint8_t*>(w_q), type_q, rb_q, qdim,
          static_cast<const std::uint8_t*>(w_k), type_k, rb_k,
          static_cast<const std::uint8_t*>(w_v), type_v, rb_v, kvdim, total_m,
          static_cast<std::uint32_t>(cols), x, q_out, k_out, v_out);
      return;
    }
    if (all_q4 && q8_batch_ok) {
      const std::uint32_t total_blk = static_cast<std::uint32_t>(cols >> 5);
      const std::uint32_t tile_blk = total_blk < 128u ? total_blk : 128u;
      if (const BlockQ8_1* x_q8 = QuantizeToQ8Scratch(
              x, static_cast<std::size_t>(n_tokens) * total_blk, stream)) {
        const std::size_t smem =
            static_cast<std::size_t>(n_tokens) * tile_blk * sizeof(BlockQ8_1);
        hipLaunchKernelGGL(
            GemvQkvBatchedQ8Kernel, grid, block, smem, stream,
            static_cast<const std::uint8_t*>(w_q), rb_q, qdim,
            static_cast<const std::uint8_t*>(w_k), rb_k,
            static_cast<const std::uint8_t*>(w_v), rb_v, kvdim, total_m,
            total_blk, tile_blk, x_q8, q_out, k_out, v_out, n_tokens);
        return;
      }
    }
    hipLaunchKernelGGL(
        GemvQkvBatchedKernel, grid, block, 0, stream,
        static_cast<const std::uint8_t*>(w_q), type_q, rb_q, qdim,
        static_cast<const std::uint8_t*>(w_k), type_k, rb_k,
        static_cast<const std::uint8_t*>(w_v), type_v, rb_v, kvdim, total_m,
        static_cast<std::uint32_t>(cols), x, q_out, k_out, v_out, n_tokens);
    return;
  }
  GemvBatched(w_q, type_q, rb_q, cols, x, q_out, n_tokens, qdim, 0.0F, stream);
  GemvBatched(w_k, type_k, rb_k, cols, x, k_out, n_tokens, kvdim, 0.0F, stream);
  if (w_v != nullptr) {
    GemvBatched(w_v, type_v, rb_v, cols, x, v_out, n_tokens, kvdim, 0.0F,
                stream);
  }
}

void PrepAttnQkv(float* qbuf, const float* kbuf, const float* vbuf,
                 const float* q_norm, const float* k_norm, const float* v_norm,
                 __half* kv_k, __half* kv_v, std::uint32_t n_tokens,
                 std::uint32_t num_heads, std::uint32_t kv_heads,
                 bool has_kv, bool has_v_proj, std::uint32_t head_dim,
                 std::uint32_t rotary_dim, std::uint32_t start_pos,
                 float theta, const float* freq_factors, float eps,
                 hipStream_t stream) {
  if (n_tokens == 0) return;
  const std::uint32_t total_groups =
      num_heads + (has_kv ? (has_v_proj ? 2u * kv_heads : kv_heads) : 0u);
  hipLaunchKernelGGL(PrepAttnQkvKernel, dim3(total_groups, n_tokens), dim3(256),
                     0, stream, qbuf, kbuf, vbuf, q_norm, k_norm, v_norm, kv_k,
                     kv_v, num_heads, kv_heads, has_v_proj, head_dim,
                     rotary_dim, start_pos, theta, freq_factors, eps);
}

void FfnTripleNorm(const float* x, const float* gamma_ffn,
                   const float* gamma_pre2, const float* router_scale,
                   float* out_ffn, float* out_pre2, float* out_router,
                   std::uint32_t n_rows, std::uint32_t dim, float eps,
                   hipStream_t stream) {
  if (n_rows == 0 || dim == 0) return;
  hipLaunchKernelGGL(FfnTripleNormKernel, dim3(n_rows), dim3(256), 0, stream, x,
                     gamma_ffn, gamma_pre2, router_scale, out_ffn, out_pre2,
                     out_router, dim, eps);
}

void MoeCombinePostNormResidual(
    const float* slot_down, const std::int32_t* ids, const float* wts,
    const float* dscale, const float* shared_down, const float* gamma_post1,
    const float* gamma_post2, const float* gamma_post, float* residual_inout,
    std::uint32_t n_tokens, std::uint32_t used, std::uint32_t dim, float eps,
    float post_scale, hipStream_t stream) {
  if (n_tokens == 0 || dim == 0) return;
  hipLaunchKernelGGL(
      MoeCombinePostNormResidualKernel, dim3(n_tokens), dim3(256),
      static_cast<std::size_t>(dim) * sizeof(float), stream, slot_down, ids,
      wts, dscale, shared_down, gamma_post1, gamma_post2, gamma_post,
      residual_inout, used, dim, eps, post_scale);
}

void BuildRoutedTiles(const std::int32_t* routed_bounds,
                      std::int32_t* q4_tiles, std::uint32_t* d_meta,
                      std::uint32_t num_experts, std::uint32_t tile_rows,
                      hipStream_t stream) {
  if (num_experts == 0 || tile_rows == 0) return;
  hipLaunchKernelGGL(BuildRoutedTilesKernel, dim3(1), dim3(256), 0, stream,
                     routed_bounds, q4_tiles, d_meta, num_experts, tile_rows);
}

}  // namespace kernels
}  // namespace gufo::models::gemma4
