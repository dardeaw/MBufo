#ifndef MBUFO_CORE_COMPUTE_GEMM_DEVICE_CUH_
#define MBUFO_CORE_COMPUTE_GEMM_DEVICE_CUH_

// Device-side helpers shared by the core GEMM TUs (verbatim from gemma4).
// Included only by .hip files, never by host TUs.
#include <hip/hip_fp16.h>

#include <cstdint>

#include "src/core/quant/int_dot.hpp"
#include "src/core/quant/row_dot_traits.hpp"

namespace mbufo::core::compute::gemm {
namespace device {
// SwiGLU-OAI (gpt-oss): min(g, 7) / (1 + exp(-1.702 min(g, 7))) * (clamp(u, -7, 7) + 1).
[[nodiscard]] __device__ __forceinline__ float SwigluOaiOne(float g, float u) {
  const float x = g < 7.0F ? g : 7.0F;
  const float y = u < -7.0F ? -7.0F : (u > 7.0F ? 7.0F : u);
  return (x / (1.0F + expf(-1.702F * x))) * (y + 1.0F);
}

[[nodiscard]] __device__ __forceinline__ float GeluTanhOne(float x) {
  // Exact tanh approximation used by Gemma (0.7978845608 = sqrt(2/pi)).
  const float inner = 0.7978845608F * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + tanhf(inner));
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

__device__ __forceinline__ float WaveRowTileDot(
    const std::uint8_t* __restrict__ wrow, int type, std::uint32_t k0,
    std::uint32_t tile_k, std::uint32_t lane, const float* __restrict__ sx) {
  // One definition per format: core/quant/row_dot_traits.hpp (lane geometry). Types other than the four
  // below are F32, as before.
  namespace q = ::mbufo::core::quant;
  switch (type) {
    case 2:  // Q4_0: 8 blocks (256 weights) per 32-lane wave step
      return q::RowDot<q::FormatId::kQ4_0>::LaneDot(wrow + static_cast<std::size_t>(k0 >> 5) * 18, sx, tile_k, lane);
    case 8:  // Q8_0: 4 blocks (128 weights) per 32-lane wave step
      return q::RowDot<q::FormatId::kQ8_0>::LaneDot(wrow + static_cast<std::size_t>(k0 >> 5) * 34, sx, tile_k, lane);
    case 1:  // F16
      return q::RowDot<q::FormatId::kF16>::LaneDot(wrow + static_cast<std::size_t>(k0) * 2, sx, tile_k, lane);
    case 30:  // BF16
      return q::RowDot<q::FormatId::kBF16>::LaneDot(wrow + static_cast<std::size_t>(k0) * 2, sx, tile_k, lane);
    default:  // F32
      return q::RowDot<q::FormatId::kF32>::LaneDot(wrow + static_cast<std::size_t>(k0) * 4, sx, tile_k, lane);
  }
}

using ::mbufo::core::quant::intdot::Dot4U4S8;
using ::mbufo::core::quant::intdot::Dot4S8S8;
}  // namespace device
}  // namespace mbufo::core::compute::gemm

#endif  // MBUFO_CORE_COMPUTE_GEMM_DEVICE_CUH_
