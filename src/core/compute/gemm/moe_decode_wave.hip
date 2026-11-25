// Decode MoE, lane geometry: one wave per weight row, lane partials reduced across the wave (gemma4 GeLU, gpt-oss SwiGLU-OAI with per-expert biases). The sibling of moe_decode_kq.hip (thread geometry, qwen35moe).
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
  } else if constexpr (kAct == MoeGateAct::kSwigluOaiBias) {
    return device::SwigluOaiOne(g, u);
  } else {
    return (g * (1.0F / (1.0F + expf(-g)))) * u;
  }
}

template <class F, MoeGateAct kAct>
__global__ void __launch_bounds__(256) MoeWaveGateUpKernel(
    const void* __restrict__ gate_up, std::uint64_t row_bytes, std::uint64_t expert_stride_bytes,
    const float* __restrict__ x, const std::int32_t* __restrict__ ids, float* __restrict__ act_out,
    std::uint32_t used, std::uint32_t ff, std::uint32_t hidden, const float* __restrict__ gate_b,
    const float* __restrict__ up_b) {
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
  if (lane == 0) {
    if constexpr (kAct == MoeGateAct::kSwigluOaiBias) {
      const std::size_t boff = static_cast<std::size_t>(e) * ff + out_row;
      g += gate_b[boff];
      u += up_b[boff];
    }
    act_out[static_cast<std::size_t>(slot) * ff + out_row] = GateActivation<kAct>(g, u);
  }
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


struct MoeBlockQ8_1 {
  __half2 ds;
  int qs[8];
};
static_assert(sizeof(MoeBlockQ8_1) == 36, "MoeBlockQ8_1 must match block_q8_1");

struct MoeLdsQ8_1 {
  float d;
  float s;
  int qs[8];
};

template <class F, MoeGateAct kAct>
__global__ void __launch_bounds__(256) MoeWaveGateUpQ8Kernel(
    const void* __restrict__ gate_up, std::uint64_t row_bytes, std::uint64_t expert_stride_bytes,
    const MoeBlockQ8_1* __restrict__ x_q8, std::size_t q8_stride_blocks, const std::int32_t* __restrict__ ids,
    float* __restrict__ act_out, std::uint32_t used, std::uint32_t ff, std::uint32_t hidden,
    const float* __restrict__ gate_b, const float* __restrict__ up_b) {
  const std::uint32_t slot = blockIdx.x;
  const std::int32_t e = ids[slot];
  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t base_row = blockIdx.y * 16u + warp_id * 2u;
  if (e < 0) {
    if (lane == 0) {
      if (base_row < ff) act_out[static_cast<std::size_t>(slot) * ff + base_row] = 0.0F;
      if (base_row + 1u < ff) act_out[static_cast<std::size_t>(slot) * ff + base_row + 1u] = 0.0F;
    }
    return;
  }
  extern __shared__ __align__(8) unsigned char smem_q8[];
  auto* sx = reinterpret_cast<MoeLdsQ8_1*>(smem_q8);
  const std::uint32_t nb = hidden >> 5;
  const MoeBlockQ8_1* xr = x_q8 + static_cast<std::size_t>(slot / used) * q8_stride_blocks;
  for (std::uint32_t b = threadIdx.x; b < nb; b += 256u) {
    const MoeBlockQ8_1 blk = xr[b];
    const float2 ds2 = __half22float2(blk.ds);
    sx[b].d = ds2.x;
    sx[b].s = ds2.y;
#pragma unroll
    for (int i = 0; i < 8; ++i) sx[b].qs[i] = blk.qs[i];
  }
  __syncthreads();

  const std::uint32_t b_grp = lane >> F::kQLaneShift;
  const std::uint32_t sub = lane & F::kQLaneMask;
  const auto* ebase = static_cast<const std::uint8_t*>(gate_up) + static_cast<std::size_t>(e) * expert_stride_bytes;
#pragma unroll
  for (std::uint32_t r = 0; r < 2u; ++r) {
    const std::uint32_t out_row = base_row + r;
    if (out_row >= ff) break;
    const auto* grow = ebase + static_cast<std::size_t>(out_row) * row_bytes;
    const auto* urow = ebase + static_cast<std::size_t>(ff + out_row) * row_bytes;
    float g = 0.0F;
    float u = 0.0F;
    for (std::uint32_t b0 = 0; b0 < nb; b0 += F::kQBlocksPerStep) {
      const std::uint32_t b = b0 + b_grp;
      if (b < nb) {
        typename F::QLane gl, ul;
        F::LoadQLane(grow + b * F::kBlockBytes, sub, gl);
        F::LoadQLane(urow + b * F::kBlockBytes, sub, ul);
        const MoeLdsQ8_1& xblk = sx[b];
        g += F::QTail(gl, xblk, F::QIntDot(gl, xblk, sub));
        u += F::QTail(ul, xblk, F::QIntDot(ul, xblk, sub));
      }
    }
    g = device::WaveSum(g);
    u = device::WaveSum(u);
    if (lane == 0) {
      if constexpr (kAct == MoeGateAct::kSwigluOaiBias) {
        const std::size_t boff = static_cast<std::size_t>(e) * ff + out_row;
        g += gate_b[boff];
        u += up_b[boff];
      }
      act_out[static_cast<std::size_t>(slot) * ff + out_row] = GateActivation<kAct>(g, u);
    }
  }
}

template <class F>
__global__ void __launch_bounds__(256) MoeWaveDownQ8Kernel(
    const void* __restrict__ down, std::uint64_t row_bytes, std::uint64_t expert_stride_bytes,
    const MoeBlockQ8_1* __restrict__ act_q8, std::size_t q8_stride_blocks, const std::int32_t* __restrict__ ids,
    float* __restrict__ out, std::uint32_t hidden, std::uint32_t ff) {
  const std::uint32_t slot = blockIdx.x;
  const std::int32_t e = ids[slot];
  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t base_h = blockIdx.y * 32u + warp_id * 4u;
  if (e < 0) {
    if (lane == 0) {
      for (std::uint32_t r = 0; r < 4u; ++r) {
        if (base_h + r < hidden) out[static_cast<std::size_t>(slot) * hidden + base_h + r] = 0.0F;
      }
    }
    return;
  }
  extern __shared__ __align__(8) unsigned char smem_dn_q8[];
  auto* sa = reinterpret_cast<MoeLdsQ8_1*>(smem_dn_q8);
  const std::uint32_t nb = ff >> 5;
  const MoeBlockQ8_1* ar = act_q8 + static_cast<std::size_t>(slot) * q8_stride_blocks;
  for (std::uint32_t b = threadIdx.x; b < nb; b += 256u) {
    const MoeBlockQ8_1 blk = ar[b];
    const float2 ds2 = __half22float2(blk.ds);
    sa[b].d = ds2.x;
    sa[b].s = ds2.y;
#pragma unroll
    for (int i = 0; i < 8; ++i) sa[b].qs[i] = blk.qs[i];
  }
  __syncthreads();

  const std::uint32_t b_grp = lane >> F::kQLaneShift;
  const std::uint32_t sub = lane & F::kQLaneMask;
  const auto* ebase = static_cast<const std::uint8_t*>(down) + static_cast<std::size_t>(e) * expert_stride_bytes;
#pragma unroll
  for (std::uint32_t r = 0; r < 4u; ++r) {
    const std::uint32_t h = base_h + r;
    if (h >= hidden) break;
    const auto* wrow = ebase + static_cast<std::size_t>(h) * row_bytes;
    float sum = 0.0F;
    for (std::uint32_t b0 = 0; b0 < nb; b0 += F::kQBlocksPerStep) {
      const std::uint32_t b = b0 + b_grp;
      if (b < nb) {
        typename F::QLane wl;
        F::LoadQLane(wrow + b * F::kBlockBytes, sub, wl);
        const MoeLdsQ8_1& ablk = sa[b];
        sum += F::QTail(wl, ablk, F::QIntDot(wl, ablk, sub));
      }
    }
    sum = device::WaveSum(sum);
    if (lane == 0) out[static_cast<std::size_t>(slot) * hidden + h] = sum;
  }
}

}  // namespace


bool LaunchMoeWaveGateUpQ8(const void* gate_up, WeightType type, std::uint64_t row_bytes,
                           std::uint64_t expert_stride_bytes, const void* x_q8, std::size_t q8_stride_blocks,
                           const std::int32_t* ids, float* act_out, std::uint32_t slots, std::uint32_t used,
                           std::uint32_t ff, std::uint32_t hidden, MoeGateAct act, hipStream_t stream,
                           const float* gate_b, const float* up_b) {
  if (gate_up == nullptr || x_q8 == nullptr || ids == nullptr || act_out == nullptr) return false;
  if (slots == 0 || used == 0 || slots % used != 0 || ff == 0 || hidden == 0) return false;
  if ((hidden % 32u) != 0 || hidden > 16384u || q8_stride_blocks < hidden / 32u) return false;
  if (act == MoeGateAct::kSwigluOaiBias && (gate_b == nullptr || up_b == nullptr)) return false;
  bool launched = false;
  (void)quant::LaneRowFormats::Dispatch(static_cast<int>(type), [&](auto fm) {
    using F = decltype(fm);
    if constexpr (quant::QActLaneFormat<F>) {
      if (!F::ColsOk(hidden) || row_bytes != F::RowBytes(hidden)) return;
      const dim3 grid(slots, (ff + 15u) / 16u);
      const std::size_t smem = (hidden >> 5) * sizeof(MoeLdsQ8_1);
      const auto go = [&](auto act_c) {
        hipLaunchKernelGGL((MoeWaveGateUpQ8Kernel<F, decltype(act_c)::value>), grid, dim3(256), smem,
                           stream, gate_up, row_bytes, expert_stride_bytes,
                           static_cast<const MoeBlockQ8_1*>(x_q8), q8_stride_blocks, ids, act_out,
                           used, ff, hidden, gate_b, up_b);
      };
      if (act == MoeGateAct::kGeluTanhMul) {
        go(std::integral_constant<MoeGateAct, MoeGateAct::kGeluTanhMul>{});
      } else if (act == MoeGateAct::kSwigluOaiBias) {
        go(std::integral_constant<MoeGateAct, MoeGateAct::kSwigluOaiBias>{});
      } else if (act == MoeGateAct::kSituSoftcap) {
        go(std::integral_constant<MoeGateAct, MoeGateAct::kSituSoftcap>{});
      } else {
        go(std::integral_constant<MoeGateAct, MoeGateAct::kSiluMul>{});
      }
      launched = true;
    }
  });
  return launched;
}

bool LaunchMoeWaveDownQ8(const void* down, WeightType type, std::uint64_t row_bytes,
                         std::uint64_t expert_stride_bytes, const void* act_q8, std::size_t q8_stride_blocks,
                         const std::int32_t* ids, float* out, std::uint32_t slots, std::uint32_t hidden,
                         std::uint32_t ff, hipStream_t stream) {
  if (down == nullptr || act_q8 == nullptr || ids == nullptr || out == nullptr) return false;
  if (slots == 0 || hidden == 0 || ff == 0) return false;
  if ((ff % 32u) != 0 || ff > 16384u || q8_stride_blocks < ff / 32u) return false;
  bool launched = false;
  (void)quant::LaneRowFormats::Dispatch(static_cast<int>(type), [&](auto fm) {
    using F = decltype(fm);
    if constexpr (quant::QActLaneFormat<F>) {
      if (!F::ColsOk(ff) || row_bytes != F::RowBytes(ff)) return;
      const std::size_t smem = (ff >> 5) * sizeof(MoeLdsQ8_1);
      hipLaunchKernelGGL((MoeWaveDownQ8Kernel<F>), dim3(slots, (hidden + 31u) / 32u), dim3(256), smem, stream,
                         down, row_bytes, expert_stride_bytes, static_cast<const MoeBlockQ8_1*>(act_q8),
                         q8_stride_blocks, ids, out, hidden, ff);
      launched = true;
    }
  });
  return launched;
}

bool LaunchMoeWaveGateUp(const void* gate_up, WeightType type, std::uint64_t row_bytes,
                         std::uint64_t expert_stride_bytes, const float* x, const std::int32_t* ids, float* act_out,
                         std::uint32_t slots, std::uint32_t used, std::uint32_t ff, std::uint32_t hidden,
                         MoeGateAct act, hipStream_t stream, const float* gate_b, const float* up_b) {
  if (gate_up == nullptr || x == nullptr || ids == nullptr || act_out == nullptr) return false;
  if (slots == 0 || used == 0 || slots % used != 0 || ff == 0 || hidden == 0) return false;
  if (hidden % 4 != 0 || hidden > 16384) return false;  // float4 staging, LDS budget
  if (act == MoeGateAct::kSwigluOaiBias && (gate_b == nullptr || up_b == nullptr)) return false;
  bool launched = false;
  (void)quant::LaneRowFormats::Dispatch(static_cast<int>(type), [&](auto fm) {
    using F = decltype(fm);
    if (!F::ColsOk(hidden) || row_bytes != F::RowBytes(hidden)) return;
    const dim3 grid(slots, (ff + 7u) / 8u);
    const auto go = [&](auto act_c) {
      hipLaunchKernelGGL((MoeWaveGateUpKernel<F, decltype(act_c)::value>), grid, dim3(256), hidden * sizeof(float),
                         stream, gate_up, row_bytes, expert_stride_bytes, x, ids, act_out, used, ff, hidden, gate_b, up_b);
    };
    if (act == MoeGateAct::kGeluTanhMul) {
      go(std::integral_constant<MoeGateAct, MoeGateAct::kGeluTanhMul>{});
    } else if (act == MoeGateAct::kSwigluOaiBias) {
      go(std::integral_constant<MoeGateAct, MoeGateAct::kSwigluOaiBias>{});
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
