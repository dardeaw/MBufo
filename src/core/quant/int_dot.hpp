#ifndef MBUFO_CORE_QUANT_INT_DOT_HPP_
#define MBUFO_CORE_QUANT_INT_DOT_HPP_

// 4 x int8 dot-accumulate helpers shared by every quantized-activation kernel (Q8_1 / Q8_K). Moved verbatim from
// compute/gemm/gemm_device.cuh, which keeps its device:: names as forwarding aliases. HIP translation units only.
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

namespace mbufo::core::quant::intdot {

// unsigned-4-bit x signed-8-bit (the Q4 nibble codes are unsigned, the activations signed)
__device__ __forceinline__ int Dot4U4S8(int u4_packed, int s8_packed, int acc) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
  return __builtin_amdgcn_sudot4(true, u4_packed, true, s8_packed, acc, false);
#elif defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx942__) || defined(__gfx1030__)
  return __builtin_amdgcn_sdot4(u4_packed, s8_packed, acc, false);
#else
  const auto* pa = reinterpret_cast<const signed char*>(&u4_packed);
  const auto* pb = reinterpret_cast<const signed char*>(&s8_packed);
  return acc + pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2] + pa[3] * pb[3];
#endif
}

__device__ __forceinline__ int Dot4S8S8(int a_packed, int b_packed, int acc) { return Dot4U4S8(a_packed, b_packed, acc); }

}  // namespace mbufo::core::quant::intdot

#endif  // ENGINE_ENABLE_HIP
#endif  // MBUFO_CORE_QUANT_INT_DOT_HPP_
