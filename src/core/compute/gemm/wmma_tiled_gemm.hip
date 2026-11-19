#ifdef ENGINE_ENABLE_HIP

// Shared prefill GEMM for weights that are already F16 in the TILED layout (see gemm.hpp, kTiledRows):
//   Y[t][r] = sum_k X[t][k] * W[r][k]      X F16 row-major [batch][k], Y F32 [batch][m].
// Why a tiled weight layout: with the row-major weight (row stride 13-40 KB) every K step of a block touches 128-256
// different rows for only 64 bytes each, and the memory system serves that at ~12-20 TFLOPS even when everything is
// in cache. Tiled storage makes each block's K step a contiguous 8 KB read and the same kernel reaches ~37-39 TFLOPS
// (WMMA-only ceiling on gfx1151 measured at ~54, no-memory version of this kernel ~40).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include <type_traits>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/quant/quant_traits.hpp"

namespace gufo::core::compute::gemm {
namespace quant = ::gufo::core::quant;
namespace {

typedef _Float16 v16h_t __attribute__((ext_vector_type(16)));
typedef float v8f_t __attribute__((ext_vector_type(8)));
typedef unsigned v4u_t __attribute__((ext_vector_type(4)));
typedef unsigned v8u_t __attribute__((ext_vector_type(8)));

// 16-byte chunks of a 64-byte LDS row are xor-swizzled by (row >> 1) & 3 so a 16-lane fragment read is conflict-free.
__device__ __forceinline__ v16h_t TiledLoadFrag(const __half* base, int row, int ks) {
  const int sw = (row >> 1) & 3;
  const v4u_t lo = *reinterpret_cast<const v4u_t*>(base + row * 32 + (((2 * ks) ^ sw) * 8));
  const v4u_t hi = *reinterpret_cast<const v4u_t*>(base + row * 32 + (((2 * ks + 1) ^ sw) * 8));
  const v8u_t v = {lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
  return __builtin_bit_cast(v16h_t, v);
}

// Block = 8 waves; tile BM weight rows x BN tokens; K stepped 32 at a time through a double-buffered LDS stage
// (global loads for step k+1 are issued before the WMMAs of step k and committed after them: one barrier per step).
template <class F, int BM, int BN, int WM, int WN>
__launch_bounds__(256) __global__
    void TiledGemmKernel(const void* __restrict__ w_raw, const __half* __restrict__ x, float* __restrict__ y,
                         int batch, int m, int k, std::size_t row_bytes) {
  constexpr bool kFused = false;
  const __half* w = static_cast<const __half*>(w_raw);
  static_assert(WM * WN == 8);
  constexpr int WRT = BM / WM / 16;
  constexpr int WTT = BN / WN / 16;
  constexpr int A_CHUNKS = BM * 4;
  constexpr int B_CHUNKS = BN * 4;
  static_assert(A_CHUNKS % 256 == 0 && B_CHUNKS % 256 == 0);
  constexpr int A_PER = A_CHUNKS / 256;
  constexpr int B_PER = B_CHUNKS / 256;
  __shared__ __attribute__((aligned(16))) __half lds[2][(BM + BN) * 32];

  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, sub = lane & 15, half_id = lane >> 4;
  const int wr = wave / WN, wt = wave % WN;
  const int r0 = blockIdx.y * BM, t0 = blockIdx.x * BN;
  const int nk = k / 32;

  v4u_t a_reg[A_PER], b_reg[B_PER];
  const __half* a_src[A_PER];
  const std::uint8_t* a_row[A_PER];
  const __half* b_src[B_PER];
#pragma unroll
  for (int p = 0; p < A_PER; ++p) {
    const int idx = p * 256 + tid;
    const int row = min(r0 + (idx >> 2), m - 1);
    if constexpr (kFused) {
      a_row[p] = static_cast<const std::uint8_t*>(w_raw) + static_cast<std::size_t>(row) * row_bytes;
      a_src[p] = nullptr;
    } else {
      a_src[p] = w + (static_cast<std::size_t>(row >> 7) * nk * 128 + (row & 127)) * 32 + (idx & 3) * 8;
      a_row[p] = nullptr;
    }
  }
#pragma unroll
  for (int p = 0; p < B_PER; ++p) {
    const int idx = p * 256 + tid;
    b_src[p] = x + static_cast<std::size_t>(min(t0 + (idx >> 2), batch - 1)) * k + (idx & 3) * 8;
  }

#define TG_GLOAD(KB)                                                                                         \
  _Pragma("unroll") for (int p = 0; p < A_PER; ++p) {                                                        \
    if constexpr (kFused) {                                                                                  \
      const std::uint32_t c0 = static_cast<std::uint32_t>(KB) * 32u + (((p * 256 + tid) & 3) << 3);          \
      float vals[8];                                                                                         \
      _Pragma("unroll") for (int e = 0; e < 8; ++e) vals[e] = F::Element(a_row[p], c0 + e);                  \
      v4u_t packed;                                                                                          \
      _Pragma("unroll") for (int e = 0; e < 4; ++e)                                                          \
          packed[e] = __builtin_bit_cast(unsigned, __floats2half2_rn(vals[2 * e], vals[2 * e + 1]));         \
      a_reg[p] = packed;                                                                                     \
    } else {                                                                                                 \
      a_reg[p] = *reinterpret_cast<const v4u_t*>(a_src[p] + static_cast<std::size_t>(KB) * 4096);            \
    }                                                                                                        \
  }                                                                                                          \
  _Pragma("unroll") for (int p = 0; p < B_PER; ++p) b_reg[p] =                                              \
      *reinterpret_cast<const v4u_t*>(b_src[p] + static_cast<std::size_t>(KB) * 32);
#define TG_SSTORE(BUF)                                                                                       \
  {                                                                                                          \
    __half* sa = lds[BUF];                                                                                   \
    __half* sb = lds[BUF] + BM * 32;                                                                         \
    _Pragma("unroll") for (int p = 0; p < A_PER; ++p) {                                                      \
      const int idx = p * 256 + tid;                                                                         \
      *reinterpret_cast<v4u_t*>(sa + (idx >> 2) * 32 + (((idx & 3) ^ (((idx >> 2) >> 1) & 3)) * 8)) =        \
          a_reg[p];                                                                                          \
    }                                                                                                        \
    _Pragma("unroll") for (int p = 0; p < B_PER; ++p) {                                                      \
      const int idx = p * 256 + tid;                                                                         \
      *reinterpret_cast<v4u_t*>(sb + (idx >> 2) * 32 + (((idx & 3) ^ (((idx >> 2) >> 1) & 3)) * 8)) =        \
          b_reg[p];                                                                                          \
    }                                                                                                        \
  }

  v8f_t acc[WRT][WTT];
#pragma unroll
  for (int i = 0; i < WRT; ++i)
#pragma unroll
    for (int j = 0; j < WTT; ++j) acc[i][j] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};

  TG_GLOAD(0)
  TG_SSTORE(0)
  __syncthreads();
  for (int kb = 0; kb < nk; ++kb) {
    const int cur = kb & 1;
    TG_GLOAD(min(kb + 1, nk - 1))  // the last step re-reads a valid stage and discards it
    const __half* sa = lds[cur];
    const __half* sb = lds[cur] + BM * 32;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      v16h_t af[WRT], bf[WTT];
#pragma unroll
      for (int i = 0; i < WRT; ++i) af[i] = TiledLoadFrag(sa, wr * WRT * 16 + i * 16 + sub, ks);
#pragma unroll
      for (int j = 0; j < WTT; ++j) bf[j] = TiledLoadFrag(sb, wt * WTT * 16 + j * 16 + sub, ks);
#pragma unroll
      for (int i = 0; i < WRT; ++i)
#pragma unroll
        for (int j = 0; j < WTT; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[i], bf[j], acc[i][j]);
    }
    TG_SSTORE(cur ^ 1)
    __syncthreads();
  }
#undef TG_GLOAD
#undef TG_SSTORE
  // C layout: lane holds S[row = 2l + half_id][col = sub] of each 16x16 tile (row = weight row, col = token).
#pragma unroll
  for (int i = 0; i < WRT; ++i)
#pragma unroll
    for (int j = 0; j < WTT; ++j) {
      const int tok = t0 + wt * WTT * 16 + j * 16 + sub;
      if (tok < batch) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const int row = r0 + wr * WRT * 16 + i * 16 + 2 * l + half_id;
          if (row < m) y[static_cast<std::size_t>(tok) * m + row] = acc[i][j][l];
        }
      }
    }
}

}  // namespace

namespace {
template <class F>
bool LaunchTiled(const void* w, const __half* x, float* y, std::size_t batch, std::size_t m, std::size_t k,
                 std::size_t row_bytes, hipStream_t stream) {
  if (m % 256 == 0) {
    hipLaunchKernelGGL((TiledGemmKernel<F, 256, 128, 4, 2>),
                       dim3(static_cast<unsigned>((batch + 127) / 128), static_cast<unsigned>(m / 256)), dim3(256), 0,
                       stream, w, x, y, static_cast<int>(batch), static_cast<int>(m), static_cast<int>(k), row_bytes);
  } else {
    hipLaunchKernelGGL((TiledGemmKernel<F, 128, 128, 2, 4>),
                       dim3(static_cast<unsigned>((batch + 127) / 128), static_cast<unsigned>(m / 128)), dim3(256), 0,
                       stream, w, x, y, static_cast<int>(batch), static_cast<int>(m), static_cast<int>(k), row_bytes);
  }
  return true;
}
}  // namespace

bool TiledF16Gemm(const __half* w_tiled, const __half* x, float* y, std::size_t batch, std::size_t m, std::size_t k,
                  hipStream_t stream) {
  if (w_tiled == nullptr || x == nullptr || y == nullptr || batch == 0 || m == 0 || k == 0) return false;
  if (m % kTiledRows != 0 || k % 32 != 0 || batch > 0x7fffffffu || m > 0x7fffffffu) return false;
  return LaunchTiled<void>(w_tiled, x, y, batch, m, k, 0, stream);
}

}  // namespace gufo::core::compute::gemm

#endif  // ENGINE_ENABLE_HIP