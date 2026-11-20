// Decode MoE, lane geometry: one wave per weight row, lane partials reduced across the wave (gemma4; gptoss
// joins when its MXFP4 row is ported). The sibling of moe_decode_kq.hip (thread geometry, qwen35moe).
//
// Kernels are templates over `RowDot<F>` (src/core/quant/row_dot_traits.hpp, LaneRowFormat) and a gate
// activation, so a new format or activation is a trait / enum value, not a copy of the kernel. The body is the
// gemma4 vec path verbatim with the hard-coded Q4_0 tile dot replaced by `F::LaneDot`, so results are
// bit-identical to MoeVecGateUpSwiglu / GemvQSlot for Q4_0.
//
// Merged gate/up layout: each expert holds 2 * ff rows (gate rows [0, ff), up rows [ff, 2 ff)) of `row_bytes`
// bytes, experts `expert_stride_bytes` apart. Masked slots (negative expert id) write zeros, so a consumer can
// sum all slots without checking (gemma semantics; the thread-geometry path leaves them untouched).
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/compute/gemm/gemm_device.cuh"
#include "src/core/quant/row_dot_traits.hpp"

namespace mbufo::core::compute::gemm {
namespace {
namespace quant = ::mbufo::core::quant;

template <MoeGateAct kAct>
__device__ __forceinline__ float GateActivation(float g, float u) {
  if constexpr (kAct == MoeGateAct::kGeluTanhMul) {
    return device::GeluTanhOne(g) * u;
  } else {
    return (g * (1.0F / (1.0F + expf(-g)))) * u;
  }
}

template <class F, MoeGateAct kAct>
__global__ void __launch_bounds__(256) MoeWaveGateUpKernel(
    const void* __restrict__ gate_up, std::uint64_t row_bytes, std::uint64_t expert_stride_bytes,
    const float* __restrict__ x, const std::int32_t* __restrict__ ids, float* __restrict__ act_out,
    std::uint32_t used, std::uint32_t ff, std::uint32_t hidden) {
  const std::uint32_t slot = blockIdx.x;
  const std::int32_t e = ids[slot];
  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t out_row = blockIdx.y * 8u + warp_id;
  if (e < 0) {
    if (out_row < ff && lane == 0) act_out[static_cast<std::size_t>(slot) * ff + out_row] = 0.0F;
    return;
  }
  extern __shared__ float sx[];
  const float* xr = x + static_cast<std::size_t>(slot / used) * hidden;
  for (std::uint32_t i = threadIdx.x * 4u; i < hidden; i += 256u * 4u) {
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(xr + i);
  }
  __syncthreads();
  if (out_row >= ff) return;

  const auto* ebase = static_cast<const std::uint8_t*>(gate_up) + static_cast<std::size_t>(e) * expert_stride_bytes;
  const auto* grow = ebase + static_cast<std::size_t>(out_row) * row_bytes;
  const auto* urow = ebase + static_cast<std::size_t>(ff + out_row) * row_bytes;
  float g = F::LaneDot(grow, sx, hidden, lane);
  float u = F::LaneDot(urow, sx, hidden, lane);
  g = device::WaveSum(g);
  u = device::WaveSum(u);
  if (lane == 0) act_out[static_cast<std::size_t>(slot) * ff + out_row] = GateActivation<kAct>(g, u);
}

template <class F>
__global__ void __launch_bounds__(256) MoeWaveDownKernel(
    const void* __restrict__ down, std::uint64_t row_bytes, std::uint64_t expert_stride_bytes,
    const float* __restrict__ act, const std::int32_t* __restrict__ ids, float* __restrict__ out,
    std::uint32_t hidden, std::uint32_t ff) {
  const std::uint32_t slot = blockIdx.x;
  const std::int32_t e = ids[slot];
  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t h = blockIdx.y * 8u + warp_id;
  if (e < 0) {
    if (h < hidden && lane == 0) out[static_cast<std::size_t>(slot) * hidden + h] = 0.0F;
    return;
  }
  extern __shared__ float sa[];
  const float* ar = act + static_cast<std::size_t>(slot) * ff;
  for (std::uint32_t i = threadIdx.x * 4u; i < ff; i += 256u * 4u) {
    *reinterpret_cast<float4*>(sa + i) = *reinterpret_cast<const float4*>(ar + i);
  }
  __syncthreads();
  if (h >= hidden) return;

  const auto* wrow = static_cast<const std::uint8_t*>(down) + static_cast<std::size_t>(e) * expert_stride_bytes +
                     static_cast<std::size_t>(h) * row_bytes;
  float sum = F::LaneDot(wrow, sa, ff, lane);
  sum = device::WaveSum(sum);
  if (lane == 0) out[static_cast<std::size_t>(slot) * hidden + h] = sum;
}

}  // namespace

bool LaunchMoeWaveGateUp(const void* gate_up, WeightType type, std::uint64_t row_bytes,
                         std::uint64_t expert_stride_bytes, const float* x, const std::int32_t* ids, float* act_out,
                         std::uint32_t slots, std::uint32_t used, std::uint32_t ff, std::uint32_t hidden,
                         MoeGateAct act, hipStream_t stream) {
  if (gate_up == nullptr || x == nullptr || ids == nullptr || act_out == nullptr) return false;
  if (slots == 0 || used == 0 || slots % used != 0 || ff == 0 || hidden == 0) return false;
  if (hidden % 4 != 0 || hidden > 16384) return false;  // float4 staging, LDS budget
  bool launched = false;
  (void)quant::LaneRowFormats::Dispatch(static_cast<int>(type), [&](auto fm) {
    using F = decltype(fm);
    if (!F::ColsOk(hidden) || row_bytes != F::RowBytes(hidden)) return;
    const dim3 grid(slots, (ff + 7u) / 8u);
    const auto go = [&](auto act_c) {
      hipLaunchKernelGGL((MoeWaveGateUpKernel<F, decltype(act_c)::value>), grid, dim3(256), hidden * sizeof(float),
                         stream, gate_up, row_bytes, expert_stride_bytes, x, ids, act_out, used, ff, hidden);
    };
    if (act == MoeGateAct::kGeluTanhMul) {
      go(std::integral_constant<MoeGateAct, MoeGateAct::kGeluTanhMul>{});
    } else {
      go(std::integral_constant<MoeGateAct, MoeGateAct::kSiluMul>{});
    }
    launched = true;
  });
  return launched;
}

bool LaunchMoeWaveDown(const void* down, WeightType type, std::uint64_t row_bytes, std::uint64_t expert_stride_bytes,
                       const float* act, const std::int32_t* ids, float* out, std::uint32_t slots,
                       std::uint32_t hidden, std::uint32_t ff, hipStream_t stream) {
  if (down == nullptr || act == nullptr || ids == nullptr || out == nullptr) return false;
  if (slots == 0 || hidden == 0 || ff == 0) return false;
  if (ff % 4 != 0 || ff > 16384) return false;
  bool launched = false;
  (void)quant::LaneRowFormats::Dispatch(static_cast<int>(type), [&](auto fm) {
    using F = decltype(fm);
    if (!F::ColsOk(ff) || row_bytes != F::RowBytes(ff)) return;
    hipLaunchKernelGGL((MoeWaveDownKernel<F>), dim3(slots, (hidden + 7u) / 8u), dim3(256), ff * sizeof(float), stream,
                       down, row_bytes, expert_stride_bytes, act, ids, out, hidden, ff);
    launched = true;
  });
  return launched;
}

}  // namespace mbufo::core::compute::gemm

#endif  // defined(ENGINE_ENABLE_HIP)
