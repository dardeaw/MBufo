#if defined(ENGINE_ENABLE_HIP)
// Single-token (decode) GEMV for K-quant weights (Q4_K, Q5_K): y[r] = sum_k W[r][k] * x[k], F32 activations.
//
// Memory-system shape: the existing block GEMV (Q8KBlockGEMVKernel_2Rows) gives every lane one 16-element sub-block per
// step -- 8 weight bytes against 64 activation bytes -- and loads the activations again for its second row, so the
// L1/L2 request stream is ~8-16x the weight bytes. Here each wave owns kRows rows and loads each 16-element slice of x
// once, reusing it for all of them; scales/bits are decoded with word-wide operations like the existing decoder.
// Arithmetic per row: sum over sub-blocks of (d*sc)*dot(q, x) - (dmin*mn)*sum(x), F32 -- the same as the existing path,
// so results agree to float rounding.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/quant/quant_traits.hpp"

namespace gufo::core::compute::gemm {
namespace {

constexpr int kRows = 2;  // rows per wave: measured best (224-230 GB/s); 4 and 8 rows spill the memory pipeline

template <int kType>
struct KGeom;
template <>
struct KGeom<12> {
  static constexpr unsigned kBlockBytes = 144, kQsOffset = 16;
  static constexpr bool kHigh = false;
};
template <>
struct KGeom<13> {
  static constexpr unsigned kBlockBytes = 176, kQsOffset = 48;
  static constexpr bool kHigh = true;
};

__device__ __forceinline__ unsigned Byte(unsigned w, int i) { return (w >> (8 * i)) & 0xFFu; }

template <int kType, int kRows>
__global__ __launch_bounds__(128) void QuantGemvSingleKernel(const std::uint8_t* __restrict__ w, std::size_t row_bytes,
                                                             const float* __restrict__ x, float* __restrict__ y,
                                                             std::uint32_t m, std::uint32_t k) {
  using G = KGeom<kType>;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t wave = threadIdx.x >> 5u;
  const std::uint32_t row0 = (blockIdx.x * 4u + wave) * kRows;
  if (row0 >= m) return;
  const std::uint8_t* rp[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) rp[r] = w + static_cast<std::size_t>(row0 + r < m ? row0 + r : row0) * row_bytes;
  float sum[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) sum[r] = 0.0F;
  const std::uint32_t nsub = k / 16u;
  for (std::uint32_t s = lane; s < nsub; s += 32u) {
    const std::uint32_t blk = s >> 4, sub16 = s & 15u;
    const std::uint32_t sb32 = sub16 >> 1, half = sub16 & 1u;
    const unsigned shift = 4u * (sb32 & 1u);
    const float4* xb = reinterpret_cast<const float4*>(x + static_cast<std::size_t>(s) * 16u);
    const float4 x0 = xb[0], x1 = xb[1], x2 = xb[2], x3 = xb[3];
    const float xsum = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w) + (x2.x + x2.y + x2.z + x2.w) +
                       (x3.x + x3.y + x3.z + x3.w);
    const float xv[16] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w, x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w};
    const std::uint32_t qs_off = G::kQsOffset + (sb32 >> 1) * 32u + half * 16u;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const std::uint8_t* b = rp[r] + static_cast<std::size_t>(blk) * G::kBlockBytes;
      const uint4 hdr = *reinterpret_cast<const uint4*>(b);  // d, dmin, scales[12]
      const uint4 qv = *reinterpret_cast<const uint4*>(b + qs_off);
      uint4 hv = make_uint4(0, 0, 0, 0);
      if constexpr (G::kHigh) hv = *reinterpret_cast<const uint4*>(b + 16 + half * 16);
      const unsigned sw[3] = {hdr.y, hdr.z, hdr.w};
      // get_scale_min_k4 (sb32 is uniform per iteration only per lane: use the generic extraction)
      const unsigned j = sb32;
      const unsigned lo = j & 3u;
      const unsigned wsel = j >> 2;  // 0: j < 4, 1: j >= 4
      unsigned sc, mn;
      if (j < 4) {
        sc = Byte(sw[0], lo) & 63u;
        mn = Byte(sw[1], lo) & 63u;
      } else {
        const unsigned b8 = Byte(sw[2], lo), b0 = Byte(sw[0], lo), b4 = Byte(sw[1], lo);
        sc = (b8 & 0xFu) | ((b0 >> 6) << 4);
        mn = (b8 >> 4) | ((b4 >> 6) << 4);
        (void)wsel;
      }
      const float d = __half2float(__builtin_bit_cast(__half, static_cast<std::uint16_t>(hdr.x & 0xFFFFu)));
      const float dmin = __half2float(__builtin_bit_cast(__half, static_cast<std::uint16_t>(hdr.x >> 16)));
      const unsigned qw[4] = {qv.x, qv.y, qv.z, qv.w};
      const unsigned hw[4] = {hv.x, hv.y, hv.z, hv.w};
      float dot = 0.0F;
#pragma unroll
      for (int wi = 0; wi < 4; ++wi) {
        unsigned q4 = (qw[wi] >> shift) & 0x0F0F0F0Fu;
        if constexpr (G::kHigh) q4 |= ((hw[wi] >> sb32) & 0x01010101u) << 4;
#pragma unroll
        for (int bi = 0; bi < 4; ++bi) dot += static_cast<float>((q4 >> (8 * bi)) & 0xFFu) * xv[wi * 4 + bi];
      }
      sum[r] += (d * static_cast<float>(sc)) * dot - (dmin * static_cast<float>(mn)) * xsum;
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    float v = sum[r];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
    if (lane == 0 && row0 + r < m) y[row0 + r] = v;
  }
}

template <int kType, int kRows>
void LaunchSingle(const void* w, std::size_t row_bytes, const float* x, float* y, std::size_t m, std::size_t k,
                  hipStream_t stream) {
  const std::size_t waves = (m + kRows - 1) / kRows;
  hipLaunchKernelGGL((QuantGemvSingleKernel<kType, kRows>), dim3(static_cast<unsigned>((waves + 3) / 4)), dim3(128), 0,
                     stream, static_cast<const std::uint8_t*>(w), row_bytes, x, y, static_cast<std::uint32_t>(m),
                     static_cast<std::uint32_t>(k));
}

}  // namespace

bool QuantGemvSingle(const void* w, int type, const float* x, float* y, std::size_t m, std::size_t k,
                     hipStream_t stream) {
  if (w == nullptr || x == nullptr || y == nullptr || m == 0 || k == 0 || k % 256 != 0 ||
      reinterpret_cast<std::uintptr_t>(w) % 16 != 0 || reinterpret_cast<std::uintptr_t>(x) % 16 != 0)
    return false;
  namespace q = ::gufo::core::quant;
  bool launched = false;
  (void)q::AllFormats::Dispatch(type, [&](auto fm) {
    using F = decltype(fm);
    if constexpr (F::kId == q::FormatId::kQ4_K) {
      LaunchSingle<12, kRows>(w, F::RowBytes(k), x, y, m, k, stream);
      launched = true;
    } else if constexpr (F::kId == q::FormatId::kQ5_K) {
      LaunchSingle<13, kRows>(w, F::RowBytes(k), x, y, m, k, stream);
      launched = true;
    }
  });
  return launched;
}

}  // namespace gufo::core::compute::gemm

#endif  // ENGINE_ENABLE_HIP