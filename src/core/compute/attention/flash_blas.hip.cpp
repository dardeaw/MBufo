// rocBLAS strided-batched flash attention (head_dim == 64).
// QK^T and P*V ride tensor-core strided-batched GEMMs (F32, proven working);
// a fused online-softmax kernel handles causal masking, sliding windows,
// sinks and the cross-tile reduction. K/V stay F16 in cache, expanded+
// converted to per-head dense F32 tiles on the fly (the pointer-array
// batched GEMM faults on this ROCm build; uniform-stride strided is used).
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/compute/attention/flash_blas.hpp"

namespace gufo::core::compute::flash_blas {
namespace {

// Wave32-native lane xor (same DPP transport as blocked_splitk).
__device__ __forceinline__ float XorLaneF(float v, int offset) {
#if defined(__gfx1151__)
  const int vi = __builtin_bit_cast(int, v);
  int r = vi;
  switch (offset) {
    case 1: r = __builtin_amdgcn_update_dpp(0, vi, 0x161, 0xF, 0xF, false); break;
    case 2: r = __builtin_amdgcn_update_dpp(0, vi, 0x162, 0xF, 0xF, false); break;
    case 4: r = __builtin_amdgcn_update_dpp(0, vi, 0x164, 0xF, 0xF, false); break;
    case 8: r = __builtin_amdgcn_update_dpp(0, vi, 0x168, 0xF, 0xF, false); break;
    case 16: r = __builtin_amdgcn_permlanex16(vi, vi, 0x76543210, 0xfedcba98, false, false); break;
    default: break;
  }
  return __builtin_bit_cast(float, r);
#else
  return __shfl_xor(v, offset);
#endif
}

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += XorLaneF(v, offset);
  return v;
}

__device__ __forceinline__ float WaveMax(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float u = XorLaneF(v, offset);
    v = v > u ? v : u;
  }
  return v;
}

constexpr unsigned kThreadsTranspose = 256;

// F32 [n][heads][d] token-major -> F32 [heads][n][d] head-major.
__global__ void TransposeQKernel(const float* __restrict__ q, float* __restrict__ qh,
                                std::uint32_t n_tokens, std::uint32_t heads) {
  constexpr std::uint32_t kHeadDim = 64;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(n_tokens) * heads * kHeadDim;
  if (i >= count) return;
  const std::uint32_t d = static_cast<std::uint32_t>(i % kHeadDim);
  const std::uint32_t h = static_cast<std::uint32_t>((i / kHeadDim) % heads);
  const std::uint32_t t = static_cast<std::uint32_t>(i / (kHeadDim * heads));
  qh[(static_cast<std::size_t>(h) * n_tokens + t) * kHeadDim + d] =
      q[(static_cast<std::size_t>(t) * heads + h) * kHeadDim + d];
}

// Expand + F16->F32: cache rows [k0,k0+rows) for ALL kv_heads ->
// dense [heads][Kt][d] (Kt-padded stride). Each head reads its kv_head.
__global__ void ExpandConvertKernel(const __half* __restrict__ cache,
                                   float* __restrict__ out,
                                   std::uint32_t k0, std::uint32_t rows,
                                   std::uint32_t kv_heads, std::uint32_t heads,
                                   std::uint32_t heads_per_kv) {
  constexpr std::uint32_t kHeadDim = 64;
  constexpr std::uint32_t kKt = kFlashBlasKt;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(heads) * rows * kHeadDim;
  if (i >= count) return;
  const std::uint32_t c = static_cast<std::uint32_t>(i % kHeadDim);
  const std::uint32_t r = static_cast<std::uint32_t>((i / kHeadDim) % rows);
  const std::uint32_t h = static_cast<std::uint32_t>(i / (kHeadDim * rows));
  const std::uint32_t g = h / heads_per_kv;
  out[(static_cast<std::size_t>(h) * kKt + r) * kHeadDim + c] =
      __half2float(cache[(static_cast<std::size_t>(k0 + r) * kv_heads + g) * kHeadDim + c]);
}

__global__ void InitMLKernel(const float* __restrict__ sinks, float* __restrict__ m,
                            float* __restrict__ l, std::uint32_t n_tokens,
                            std::uint32_t heads) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (t >= n_tokens) return;
  m[(static_cast<std::size_t>(h) * n_tokens + t)] = sinks[h];
  l[(static_cast<std::size_t>(h) * n_tokens + t)] = 1.0f;
}

// Online softmax over one Kt-score tile + acc rescale, 32 threads per
// (head, token): parallel max (WaveMax) + parallel exp/sum (WaveSum).
// All lanes participate uniformly (no divergence); masking via -inf/zero.
__global__ void SoftmaxTileKernel(float* __restrict__ scores, float* __restrict__ acc,
                                  float* __restrict__ m, float* __restrict__ l,
                                  std::uint32_t n_tokens, std::uint32_t k0,
                                  std::uint32_t kt, std::uint32_t start_pos,
                                  std::uint32_t window, std::uint32_t heads) {
  constexpr std::uint32_t kHeadDim = 64;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const unsigned lane = threadIdx.x;
  if (t >= n_tokens || h >= heads) return;
  const std::uint32_t pos = start_pos + t;
  const std::uint32_t n_kv = pos + 1;
  const std::uint32_t k0h = (window != 0 && n_kv > window) ? n_kv - window : 0;
  float* prow = scores + (static_cast<std::size_t>(h) * n_tokens + t) * kFlashBlasKt;
  float* arow = acc + (static_cast<std::size_t>(h) * n_tokens + t) * kHeadDim;
  float m_old = m[(static_cast<std::size_t>(h) * n_tokens + t)];
  float l_old = l[(static_cast<std::size_t>(h) * n_tokens + t)];
  // Pass 1: parallel max over valid rows.
  float m_part = -INFINITY;
  for (std::uint32_t k = lane; k < kt; k += 32) {
    const std::uint32_t r = k0 + k;
    if (r < k0h || r >= n_kv) continue;
    const float s = prow[k];
    if (s > m_part) m_part = s;
  }
  const float m_tile = WaveMax(m_part);
  float m_new = m_old > m_tile ? m_old : m_tile;
  const float rescale_old = expf(m_old - m_new);
  // Pass 2: exp-normalize, rescale acc, accumulate l (parallel).
  float l_part = 0.0f;
  for (std::uint32_t k = lane; k < kt; k += 32) {
    const std::uint32_t r = k0 + k;
    float w = 0.0f;
    if (r >= k0h && r < n_kv) {
      w = expf(prow[k] - m_new);
      prow[k] = w;
      l_part += w;
    } else {
      prow[k] = 0.0f;
    }
  }
  l_part = WaveSum(l_part);
  // Rescale acc (64 dims, 2 per lane) + finalize m/l.
  if (lane < 32) {
    arow[2 * lane] *= rescale_old;
    arow[2 * lane + 1] *= rescale_old;
  }
  if (lane == 0) {
    m[(static_cast<std::size_t>(h) * n_tokens + t)] = m_new;
    l[(static_cast<std::size_t>(h) * n_tokens + t)] = l_old * rescale_old + l_part;
  }
}

// Finalize: out[t][h][d] = acc[h][t][d] / l[h][t] (token-major out).
__global__ void FinalizeKernel(const float* __restrict__ acc,
                              const float* __restrict__ l, float* __restrict__ out,
                              std::uint32_t n_tokens, std::uint32_t heads) {
  constexpr std::uint32_t kHeadDim = 64;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(n_tokens) * heads * kHeadDim;
  if (i >= count) return;
  const std::uint32_t d = static_cast<std::uint32_t>(i % kHeadDim);
  const std::uint32_t h = static_cast<std::uint32_t>((i / kHeadDim) % heads);
  const std::uint32_t t = static_cast<std::uint32_t>(i / (kHeadDim * heads));
  const float denom = l[(static_cast<std::size_t>(h) * n_tokens + t)];
  out[(static_cast<std::size_t>(t) * heads + h) * kHeadDim + d] =
      acc[(static_cast<std::size_t>(h) * n_tokens + t) * kHeadDim + d] / denom;
}

#define HIPBLAS_CHECK(cmd)                             \
  do {                                                 \
    if ((cmd) != HIPBLAS_STATUS_SUCCESS) return false; \
  } while (0)

}  // namespace

bool FlashBlasPrefill(const float* q, const __half* k_cache,
                      const __half* v_cache, const float* sinks, float* out,
                      std::uint32_t n_tokens, std::uint32_t start_pos,
                      std::uint32_t kv_len, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t d,
                      std::uint32_t window, float score_scale,
                      hipStream_t stream, hipblasHandle_t handle,
                      void* scratch, std::size_t scratch_bytes) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      sinks == nullptr || out == nullptr || handle == nullptr ||
      scratch == nullptr || n_tokens == 0 || n_tokens > kFlashBlasMaxTok ||
      d != 64 || heads == 0 || heads > 128 || kv_heads == 0 ||
      heads % kv_heads != 0 || start_pos + n_tokens > kv_len ||
      scratch_bytes < FlashBlasScratchBytes(n_tokens)) {
    return false;
  }
  HIPBLAS_CHECK(hipblasSetStream(handle, stream));
  constexpr std::uint32_t kHeadDim = 64;
  constexpr std::uint32_t kKt = kFlashBlasKt;
  char* mem = static_cast<char*>(scratch);
  float* qh = reinterpret_cast<float*>(mem);
  mem += static_cast<std::size_t>(n_tokens) * heads * kHeadDim * sizeof(float);
  float* scores = reinterpret_cast<float*>(mem);
  mem += static_cast<std::size_t>(heads) * n_tokens * kKt * sizeof(float);
  float* acc = reinterpret_cast<float*>(mem);
  mem += static_cast<std::size_t>(heads) * n_tokens * kHeadDim * sizeof(float);
  float* mdev = reinterpret_cast<float*>(mem);
  mem += static_cast<std::size_t>(heads) * n_tokens * sizeof(float);
  float* ldev = reinterpret_cast<float*>(mem);
  mem += static_cast<std::size_t>(heads) * n_tokens * sizeof(float);
  float* k_tile = reinterpret_cast<float*>(mem);
  mem += static_cast<std::size_t>(heads) * kKt * kHeadDim * sizeof(float);
  float* v_tile = reinterpret_cast<float*>(mem);

  const float one = 1.0f;
  const float zero = 0.0f;
  const std::uint32_t kv_total = start_pos + n_tokens;
  const std::uint32_t heads_per_kv = heads / kv_heads;
  // 1. Transpose Q (F32->F32, layout only) + m/l init + acc zero.
  {
    const std::size_t count =
        static_cast<std::size_t>(n_tokens) * heads * kHeadDim;
    hipLaunchKernelGGL(TransposeQKernel,
                       dim3(static_cast<unsigned>((count + 255) / 256)),
                       dim3(256), 0, stream, q, qh, n_tokens, heads);
    const dim3 grid(heads, n_tokens);
    hipLaunchKernelGGL(InitMLKernel, grid, dim3(1), 0, stream, sinks, mdev,
                       ldev, n_tokens, heads);
    HIPBLAS_CHECK(hipMemsetAsync(
        acc, 0, static_cast<std::size_t>(heads) * n_tokens * kHeadDim * sizeof(float), stream));
  }
  // 2. Per K-tile: expand+convert K/V -> QK^T (strided, beta=0) ->
  //    softmax+rescale -> PV (strided, beta=1).
  for (std::uint32_t k0 = 0; k0 < kv_total; k0 += kKt) {
    std::uint32_t kt = kv_total - k0;
    if (kt > kKt) kt = kKt;
    {
      const std::size_t count =
          static_cast<std::size_t>(heads) * kt * kHeadDim;
      hipLaunchKernelGGL(ExpandConvertKernel,
                         dim3(static_cast<unsigned>((count + 255) / 256)),
                         dim3(256), 0, stream, k_cache, k_tile, k0, kt,
                         kv_heads, heads, heads_per_kv);
      hipLaunchKernelGGL(ExpandConvertKernel,
                         dim3(static_cast<unsigned>((count + 255) / 256)),
                         dim3(256), 0, stream, v_cache, v_tile, k0, kt,
                         kv_heads, heads, heads_per_kv);
    }
    HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(kt),
        static_cast<int>(n_tokens), static_cast<int>(kHeadDim), &score_scale,
        k_tile, HIP_R_32F, static_cast<int>(kHeadDim),
        static_cast<hipblasStride>(kKt * kHeadDim), qh, HIP_R_32F,
        static_cast<int>(kHeadDim),
        static_cast<hipblasStride>(n_tokens * kHeadDim), &zero, scores,
        HIP_R_32F, static_cast<int>(kKt),
        static_cast<hipblasStride>(n_tokens * kKt), static_cast<int>(heads),
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    {
      const dim3 grid(heads, n_tokens);
      hipLaunchKernelGGL(SoftmaxTileKernel, grid, dim3(32), 0, stream, scores,
                         acc, mdev, ldev, n_tokens, k0, kt, start_pos, window,
                         heads);
    }
    HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
        handle, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(kHeadDim),
        static_cast<int>(n_tokens), static_cast<int>(kt), &one, v_tile,
        HIP_R_32F, static_cast<int>(kHeadDim),
        static_cast<hipblasStride>(kKt * kHeadDim), scores, HIP_R_32F,
        static_cast<int>(kKt),
        static_cast<hipblasStride>(n_tokens * kKt), &one, acc, HIP_R_32F,
        static_cast<int>(kHeadDim),
        static_cast<hipblasStride>(n_tokens * kHeadDim), static_cast<int>(heads),
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  }
  // 3. Finalize (normalize + transpose to token-major).
  {
    const std::size_t count =
        static_cast<std::size_t>(n_tokens) * heads * kHeadDim;
    hipLaunchKernelGGL(FinalizeKernel,
                       dim3(static_cast<unsigned>((count + 255) / 256)),
                       dim3(256), 0, stream, acc, ldev, out, n_tokens, heads);
  }
  return true;
}

}  // namespace gufo::core::compute::flash_blas

#endif  // ENGINE_ENABLE_HIP
