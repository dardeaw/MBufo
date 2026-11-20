#if defined(ENGINE_ENABLE_HIP)
// Multi-token (2..32) decode/verify GEMM for K-quant weights: Y[t][r] = sum_k W[r][k] * X[t][k].
//
// Why: the single-token block GEMV (LaunchQ8KBlockGEMV) is run once per token by GemvDecode, so a speculative-decode
// verify of 20 tokens re-reads every weight 20 times; anything above 8 tokens fell back to "dequantize the whole matrix
// into F16, then GEMM" (>0.5 s per verify on a 30B model). This kernel reads each weight byte ONCE for all tokens:
// the packed weights are dequantized in registers straight into the WMMA A fragments (16 rows x 16 k per fragment),
// the F32 activations are converted to F16 B fragments (up to 2 tiles of 16 tokens), accumulation is one F32 chain in
// K order per output. Formats: Q4_K and Q5_K (the K-quants a 4-bit dense model contains); other formats return false
// and keep their existing route.
//
// The format-specific part (loading a lane's share of a superblock, decoding a 16-element sub-block into packed F16 pairs)
// is RowDot<F>::LoadSuper / DecodeSub16 (core/quant/row_dot_traits.hpp, concept TileDecodeFormat), shared with the other
// row-dot kernels; this file owns the WMMA tiling, the LDS activation pipeline and the output mapping.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/quant/quant_traits.hpp"
#include "src/core/quant/row_dot_traits.hpp"

namespace mbufo::core::compute::gemm {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using v8u = __attribute__((__vector_size__(8 * sizeof(unsigned)))) unsigned;

using ::mbufo::core::quant::rowdot_detail::PackHalves;  // activation staging; also used by the format decoders

__device__ __forceinline__ unsigned SwapHalves(unsigned v) {  // value of the same lane position in the other 16-lane half
  return static_cast<unsigned>(__builtin_amdgcn_permlanex16(static_cast<int>(v), static_cast<int>(v), 0x76543210u,
                                                            0xfedcba98u, false, false));
}

// One block = 4 waves = 64 weight rows; every wave owns 16 rows. The F16 activation tile of the current 256-wide K
// superblock lives in LDS (double buffered, shared by the four waves), the packed weights of the next superblock are
// prefetched into registers while the current one is consumed. Per 32-element sub-block the two 16-lane halves decode
// one 16-element A fragment each and swap them, so no dequantization work is duplicated.
template <class R, int kTokTiles>
__global__ __launch_bounds__(128) void QuantGemvMultiKernel(const std::uint8_t* __restrict__ w,
                                                            std::size_t row_bytes, const float* __restrict__ x,
                                                            float* __restrict__ y, std::uint32_t n_tokens,
                                                            std::uint32_t m, std::uint32_t k) {
  constexpr std::uint32_t kXStride = 264;  // halves per LDS token row (256 + 8 pad): 16-byte aligned, bank friendly
  constexpr std::uint32_t kXIter = kTokTiles * 16 * 64 / 128;  // float4 per thread per superblock
  __shared__ __attribute__((aligned(16))) __half xs[2][kTokTiles * 16][kXStride];
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;
  const std::uint32_t row0 = (blockIdx.x * 4u + wave) * 16u;
  const bool wave_live = row0 < m;  // dead waves still take part in the barriers (rows are clamped, stores skipped)
  const std::uint8_t* wrow = w + static_cast<std::size_t>(wave_live ? row0 + sub : sub) * row_bytes;
  const std::uint32_t nsb = k / 256u;

  // The activation tile of the next superblock is staged in four parts (one per g step): part p is loaded at the start of
  // step p and converted + stored to LDS at the end of step p + 1, so only ~2 parts of float4 are live at a time.
  constexpr std::uint32_t kXPart = kXIter / 4;
  float4 xr[2][kXPart];
  const auto load_x_part = [&](std::uint32_t sb, std::uint32_t part, float4 (&dst)[kXPart]) {
#pragma unroll
    for (std::uint32_t q = 0; q < kXPart; ++q) {
      const std::uint32_t i = tid + (part * kXPart + q) * 128u;
      const std::uint32_t t = i >> 6, f4 = i & 63u;
      dst[q] = t < n_tokens ? *reinterpret_cast<const float4*>(x + static_cast<std::size_t>(t) * k + sb * 256u + f4 * 4u)
                            : make_float4(0.f, 0.f, 0.f, 0.f);
    }
  };
  const auto store_x_part = [&](std::uint32_t buf, std::uint32_t part, const float4 (&src)[kXPart]) {
#pragma unroll
    for (std::uint32_t q = 0; q < kXPart; ++q) {
      const std::uint32_t i = tid + (part * kXPart + q) * 128u;
      const std::uint32_t t = i >> 6, f4 = i & 63u;
      *reinterpret_cast<uint2*>(&xs[buf][t][f4 * 4u]) = make_uint2(PackHalves(src[q].x, src[q].y), PackHalves(src[q].z, src[q].w));
    }
  };
  // Packed weights of one superblock for THIS lane (format-defined registers).
  using Wt = typename R::TileRegs;
  const auto load_w = [&](std::uint32_t sb, Wt& o) { R::LoadSuper(wrow + static_cast<std::size_t>(sb) * R::kBlockBytes, half_id, o); };

  v8f acc[kTokTiles] = {};
  Wt wc, wn;
#pragma unroll
  for (std::uint32_t part = 0; part < 4; ++part) {
    load_x_part(0, part, xr[0]);
    store_x_part(0, part, xr[0]);
  }
  load_w(0, wc);
  for (std::uint32_t sb = 0; sb < nsb; ++sb) {
    const std::uint32_t cur = sb & 1u;
    if (sb + 1 < nsb) load_w(sb + 1, wn);  // next superblock's weights: issued before the matrix work of this one
    __syncthreads();  // xs[cur] (stored at the end of the previous step) is visible to every wave
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      // This lane decodes the 16 elements l = 16 * half_id .. +15 of group g.
      if (sb + 1 < nsb) load_x_part(sb + 1, g, xr[g & 1]);  // part g of the next activation tile
#pragma unroll
      for (int hi = 0; hi < 2; ++hi) {
        const int j = 2 * g + hi;
        unsigned own[8];
        R::DecodeSub16(wc, j, own);
        // Half 0 decoded k-step hh = 0, half 1 decoded hh = 1: swap so both halves hold both fragments.
        v8u a0, a1;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const unsigned other = SwapHalves(own[e]);
          a0[e] = half_id == 0 ? own[e] : other;
          a1[e] = half_id == 0 ? other : own[e];
        }
        const v16h af0 = __builtin_bit_cast(v16h, a0);
        const v16h af1 = __builtin_bit_cast(v16h, a1);
#pragma unroll
        for (int tt = 0; tt < kTokTiles; ++tt) {
          const __half* bp = &xs[cur][tt * 16 + sub][j * 32];
          union { v16h f; uint4 u[2]; } b0, b1;
          b0.u[0] = *reinterpret_cast<const uint4*>(bp);
          b0.u[1] = *reinterpret_cast<const uint4*>(bp + 8);
          b1.u[0] = *reinterpret_cast<const uint4*>(bp + 16);
          b1.u[1] = *reinterpret_cast<const uint4*>(bp + 24);
          acc[tt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af0, b0.f, acc[tt]);
          acc[tt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af1, b1.f, acc[tt]);
        }
      }
      // the other buffer was last read during the previous step, which every wave has finished
      if (sb + 1 < nsb && g >= 1) store_x_part(cur ^ 1u, g - 1, xr[(g - 1) & 1]);
    }
    if (sb + 1 < nsb) {
      store_x_part(cur ^ 1u, 3, xr[1]);
      wc = wn;
    }
  }
  if (!wave_live) return;
  // C[row = 2i + half][token = sub]
#pragma unroll
  for (int tt = 0; tt < kTokTiles; ++tt) {
    const std::uint32_t t = tt * 16u + sub;
    if (t >= n_tokens) continue;
    float* yt = y + static_cast<std::size_t>(t) * m + row0;
#pragma unroll
    for (int i = 0; i < 8; ++i) yt[2 * i + half_id] = acc[tt][i];
  }
}

template <class R>
bool Launch(const void* w, std::size_t row_bytes, const float* x, float* y, std::size_t n, std::size_t m,
            std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((m / 16 + 3) / 4));
  if (n <= 16)
    hipLaunchKernelGGL((QuantGemvMultiKernel<R, 1>), grid, dim3(128), 0, stream,
                       static_cast<const std::uint8_t*>(w), row_bytes, x, y, static_cast<std::uint32_t>(n),
                       static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(k));
  else
    hipLaunchKernelGGL((QuantGemvMultiKernel<R, 2>), grid, dim3(128), 0, stream,
                       static_cast<const std::uint8_t*>(w), row_bytes, x, y, static_cast<std::uint32_t>(n),
                       static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(k));
  return true;
}

}  // namespace

bool QuantGemvMulti(const void* w, int type, const float* x, float* y, std::size_t n_tokens, std::size_t m,
                    std::size_t k, hipStream_t stream) {
  if (w == nullptr || x == nullptr || y == nullptr || n_tokens == 0 || n_tokens > kQuantGemvMultiMaxTokens ||
      m == 0 || m % 16 != 0 || k == 0 || k % 256 != 0 || reinterpret_cast<std::uintptr_t>(w) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(x) % 16 != 0)
    return false;
  namespace q = ::mbufo::core::quant;
  bool launched = false;
  (void)q::RowFormats::Dispatch(type, [&](auto fm) {
    using R = decltype(fm);
    if constexpr (q::TileDecodeFormat<R>) launched = Launch<R>(w, R::RowBytes(k), x, y, n_tokens, m, k, stream);
  });
  return launched;
}

}  // namespace mbufo::core::compute::gemm

#endif  // ENGINE_ENABLE_HIP