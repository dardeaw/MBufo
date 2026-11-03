#ifndef GUFO_CORE_COMPUTE_GEMM_DEVICE_CUH_
#define GUFO_CORE_COMPUTE_GEMM_DEVICE_CUH_

// Device-side helpers shared by the core GEMM TUs (verbatim from gemma4).
// Included only by .hip files, never by host TUs.
#include <hip/hip_fp16.h>

#include <cstdint>

namespace gufo::core::compute::gemm {
namespace device {
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
}  // namespace device
}  // namespace gufo::core::compute::gemm

#endif  // GUFO_CORE_COMPUTE_GEMM_DEVICE_CUH_
