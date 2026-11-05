#if defined(ENGINE_ENABLE_HIP)
// Unified fused MoE GEMV (Phase B, knife 2): gemma4 grouped/vec fusion, 27B
// fused SwiGLU, flash-next MoE epilogue. Mechanical merge only; see gemm.hpp.
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/compute/gemm/gemm_device.cuh"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::core::compute::gemm {
using device::Dot4S8S8;
using device::Dot4U4S8;
using device::GeluTanhOne;
using device::WaveRowTileDot;
using device::WaveSum;
using device::XorLaneF;
using device::XorLaneI;

namespace {
// Decode MoE routing workspace (was TU-local in the gemma4 backend).


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

static bool g_moe_grouped_active = false;
static std::uint32_t g_moe_grouped_ntok = 8u;

}  // namespace

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
      if (const BlockQ8_1* x_q8 = detail::QuantizeToQ8Scratch(
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


// ---- native MXFP4 vec MoE (gpt-oss experts, GGML type 39) ----
// OCP microscaling: 32 e2m1 values sharing one e8m0 scale (1 scale byte +
// 16 value bytes = 17 bytes/block). Low nibbles give elements 0..15, high
// nibbles 16..31. The codebook is 2xE2M1 with a halved scale, matching
// llama.cpp (kvalues_mxfp4 x GGML_E8M0_TO_FP32_HALF) bit-for-bit.
__constant__ std::int8_t kMxfp4VecValues[16] = {0, 1, 2,  3,  4,  6,  8,  12,
                                                0, -1, -2, -3, -4, -6, -8, -12};

// One MXFP4 block (17 bytes) dotted with 32 F32 activations.
__device__ __forceinline__ float Mxfp4BlockDot(
    const std::uint8_t* __restrict__ blk, const float* __restrict__ xv) {
  const float d = exp2f(static_cast<float>(blk[0]) - 128.0F);
  float sum = 0.0F;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const std::uint8_t b = blk[1 + j];
    sum += static_cast<float>(kMxfp4VecValues[b & 0x0F]) * xv[j];
    sum += static_cast<float>(kMxfp4VecValues[b >> 4]) * xv[j + 16];
  }
  return d * sum;
}

__global__ void MoeVecGateUpMxfp4Kernel(
    const void* __restrict__ gate_up, std::uint64_t row_bytes,
    std::uint64_t expert_stride_bytes, const float* __restrict__ x,
    const std::int32_t* __restrict__ ids, float* __restrict__ gout,
    float* __restrict__ uout, std::uint32_t used, std::uint32_t ff,
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
    sum += Mxfp4BlockDot(wrow + b * 17, xr + b * 32);
  }
  for (std::uint32_t off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffffffffffffull, sum, off);
  __shared__ float shared[kThreads / 32];
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : 0.0F;
    for (std::uint32_t off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffffffffffull, v, off);
    if (threadIdx.x == 0)
      (is_up ? uout : gout)[static_cast<std::size_t>(row) * ff + out_row] = v;
  }
}

__global__ void GemvQSlotMxfp4Kernel(
    const void* __restrict__ down, std::uint64_t row_bytes,
    std::uint64_t expert_stride_bytes, const float* __restrict__ act,
    const std::int32_t* __restrict__ ids, float* __restrict__ out,
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
    sum += Mxfp4BlockDot(wrow + b * 17, ar + b * 32);
  }
  __shared__ float red[256];
  red[threadIdx.x] = sum;
  __syncthreads();
  for (std::uint32_t s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[orow] = red[0];
}

void MoeVecGateUpMxfp4(const void* gate_up_dev, std::uint64_t row_bytes,
                       std::uint64_t expert_stride_bytes, const float* x,
                       const std::int32_t* ids, float* gout, float* uout,
                       std::uint32_t slots, std::uint32_t used,
                       std::uint32_t ff, std::uint32_t hidden,
                       hipStream_t stream) {
  if (slots == 0) return;
  hipLaunchKernelGGL(MoeVecGateUpMxfp4Kernel, dim3(slots, 2 * ff),
                     dim3(kThreads), 0, stream, gate_up_dev, row_bytes,
                     expert_stride_bytes, x, ids, gout, uout, used, ff,
                     hidden);
}

void GemvQSlotMxfp4(const void* down_dev, std::uint64_t row_bytes,
                    std::uint64_t expert_stride_bytes, const float* act,
                    const std::int32_t* ids, float* out, std::uint32_t slots,
                    std::uint32_t hidden, std::uint32_t ff,
                    hipStream_t stream) {
  if (slots == 0 || hidden == 0) return;
  hipLaunchKernelGGL(GemvQSlotMxfp4Kernel, dim3(slots, hidden), dim3(kThreads),
                     0, stream, down_dev, row_bytes, expert_stride_bytes,
                     act, ids, out, hidden, ff);
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
      if (const BlockQ8_1* act_q8 = detail::QuantizeToQ8Scratch(
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

// ---- 27B fused gate/up + silu SwiGLU (from qwen swiglu.hip) ----

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedSwiGLUGEMVKernel_1Row(
        const hip_bfloat16* __restrict__ gate_w,
        const hip_bfloat16* __restrict__ up_w, const float* __restrict__ x,
        float* __restrict__ out, std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const auto* g_row = gate_w + (m * K);
  const auto* u_row = up_w + (m * K);
  const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
  const auto* u_vec = reinterpret_cast<const uint4*>(u_row);
  const std::size_t K_vec = K / 8;

  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    const uint4 gv = g_vec[vi];
    const uint4 uv = u_vec[vi];
    const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
    const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);

    dot_gate += (static_cast<float>(g_vals[0]) * x0.x) +
                (static_cast<float>(g_vals[1]) * x0.y) +
                (static_cast<float>(g_vals[2]) * x0.z) +
                (static_cast<float>(g_vals[3]) * x0.w) +
                (static_cast<float>(g_vals[4]) * x1.x) +
                (static_cast<float>(g_vals[5]) * x1.y) +
                (static_cast<float>(g_vals[6]) * x1.z) +
                (static_cast<float>(g_vals[7]) * x1.w);

    dot_up += (static_cast<float>(u_vals[0]) * x0.x) +
              (static_cast<float>(u_vals[1]) * x0.y) +
              (static_cast<float>(u_vals[2]) * x0.z) +
              (static_cast<float>(u_vals[3]) * x0.w) +
              (static_cast<float>(u_vals[4]) * x1.x) +
              (static_cast<float>(u_vals[5]) * x1.y) +
              (static_cast<float>(u_vals[6]) * x1.z) +
              (static_cast<float>(u_vals[7]) * x1.w);
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float xv = x[k];
    dot_gate += static_cast<float>(g_row[k]) * xv;
    dot_up += static_cast<float>(u_row[k]) * xv;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up += __shfl_xor(dot_up, offset);
  }

  if (lane_id == 0) {
    const float sig_g = 1.0F / (1.0F + expf(-dot_gate));
    out[m] = (dot_gate * sig_g) * dot_up;
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedSwiGLUGEMVKernel_2Rows(
        const hip_bfloat16* __restrict__ gate_w,
        const hip_bfloat16* __restrict__ up_w, const float* __restrict__ x,
        float* __restrict__ out, std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t row_pair = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = row_pair * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  const auto* g_row0 = gate_w + (m0 * K);
  const auto* u_row0 = up_w + (m0 * K);
  const auto* g_row1 = (m1 < M) ? gate_w + (m1 * K) : nullptr;
  const auto* u_row1 = (m1 < M) ? up_w + (m1 * K) : nullptr;

  const auto* g_vec0 = reinterpret_cast<const uint4*>(g_row0);
  const auto* u_vec0 = reinterpret_cast<const uint4*>(u_row0);
  const auto* g_vec1 = reinterpret_cast<const uint4*>(g_row1);
  const auto* u_vec1 = reinterpret_cast<const uint4*>(u_row1);
  const std::size_t K_vec = K / 8;

  float dot_g0 = 0.0F, dot_u0 = 0.0F;
  float dot_g1 = 0.0F, dot_u1 = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    const uint4 gv0 = g_vec0[vi];
    const uint4 uv0 = u_vec0[vi];
    const auto* g_vals0 = reinterpret_cast<const hip_bfloat16*>(&gv0);
    const auto* u_vals0 = reinterpret_cast<const hip_bfloat16*>(&uv0);

    dot_g0 += (static_cast<float>(g_vals0[0]) * x0.x) +
              (static_cast<float>(g_vals0[1]) * x0.y) +
              (static_cast<float>(g_vals0[2]) * x0.z) +
              (static_cast<float>(g_vals0[3]) * x0.w) +
              (static_cast<float>(g_vals0[4]) * x1.x) +
              (static_cast<float>(g_vals0[5]) * x1.y) +
              (static_cast<float>(g_vals0[6]) * x1.z) +
              (static_cast<float>(g_vals0[7]) * x1.w);

    dot_u0 += (static_cast<float>(u_vals0[0]) * x0.x) +
              (static_cast<float>(u_vals0[1]) * x0.y) +
              (static_cast<float>(u_vals0[2]) * x0.z) +
              (static_cast<float>(u_vals0[3]) * x0.w) +
              (static_cast<float>(u_vals0[4]) * x1.x) +
              (static_cast<float>(u_vals0[5]) * x1.y) +
              (static_cast<float>(u_vals0[6]) * x1.z) +
              (static_cast<float>(u_vals0[7]) * x1.w);

    if (g_row1 != nullptr) {
      const uint4 gv1 = g_vec1[vi];
      const uint4 uv1 = u_vec1[vi];
      const auto* g_vals1 = reinterpret_cast<const hip_bfloat16*>(&gv1);
      const auto* u_vals1 = reinterpret_cast<const hip_bfloat16*>(&uv1);

      dot_g1 += (static_cast<float>(g_vals1[0]) * x0.x) +
                (static_cast<float>(g_vals1[1]) * x0.y) +
                (static_cast<float>(g_vals1[2]) * x0.z) +
                (static_cast<float>(g_vals1[3]) * x0.w) +
                (static_cast<float>(g_vals1[4]) * x1.x) +
                (static_cast<float>(g_vals1[5]) * x1.y) +
                (static_cast<float>(g_vals1[6]) * x1.z) +
                (static_cast<float>(g_vals1[7]) * x1.w);

      dot_u1 += (static_cast<float>(u_vals1[0]) * x0.x) +
                (static_cast<float>(u_vals1[1]) * x0.y) +
                (static_cast<float>(u_vals1[2]) * x0.z) +
                (static_cast<float>(u_vals1[3]) * x0.w) +
                (static_cast<float>(u_vals1[4]) * x1.x) +
                (static_cast<float>(u_vals1[5]) * x1.y) +
                (static_cast<float>(u_vals1[6]) * x1.z) +
                (static_cast<float>(u_vals1[7]) * x1.w);
    }
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float xv = x[k];
    dot_g0 += static_cast<float>(g_row0[k]) * xv;
    dot_u0 += static_cast<float>(u_row0[k]) * xv;
    if (g_row1 != nullptr) {
      dot_g1 += static_cast<float>(g_row1[k]) * xv;
      dot_u1 += static_cast<float>(u_row1[k]) * xv;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_g0 += __shfl_xor(dot_g0, offset);
    dot_u0 += __shfl_xor(dot_u0, offset);
    dot_g1 += __shfl_xor(dot_g1, offset);
    dot_u1 += __shfl_xor(dot_u1, offset);
  }

  if (lane_id == 0) {
    const float sig_g0 = 1.0F / (1.0F + expf(-dot_g0));
    out[m0] = (dot_g0 * sig_g0) * dot_u0;
    if (m1 < M) {
      const float sig_g1 = 1.0F / (1.0F + expf(-dot_g1));
      out[m1] = (dot_g1 * sig_g1) * dot_u1;
    }
  }
}

__global__ void FastFusedSwiGLUGEMVBlockKernel(const void* __restrict__ gate_w,
                                               core::GgmlType gate_type,
                                               const void* __restrict__ up_w,
                                               core::GgmlType up_type,
                                               const float* __restrict__ x,
                                               float* __restrict__ out,
                                               std::size_t M, std::size_t K) {
  const std::size_t m = blockIdx.x;
  if (m >= M)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  if (gate_type == core::GgmlType::kBF16 && up_type == core::GgmlType::kBF16) {
    const auto* g_row = static_cast<const hip_bfloat16*>(gate_w) + (m * K);
    const auto* u_row = static_cast<const hip_bfloat16*>(up_w) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
    const auto* u_vec = reinterpret_cast<const uint4*>(u_row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 gv = g_vec[vi];
      const uint4 uv = u_vec[vi];
      const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
      const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

      dot_gate += (static_cast<float>(g_vals[0]) * x0.x) +
                  (static_cast<float>(g_vals[1]) * x0.y) +
                  (static_cast<float>(g_vals[2]) * x0.z) +
                  (static_cast<float>(g_vals[3]) * x0.w) +
                  (static_cast<float>(g_vals[4]) * x1.x) +
                  (static_cast<float>(g_vals[5]) * x1.y) +
                  (static_cast<float>(g_vals[6]) * x1.z) +
                  (static_cast<float>(g_vals[7]) * x1.w);

      dot_up += (static_cast<float>(u_vals[0]) * x0.x) +
                (static_cast<float>(u_vals[1]) * x0.y) +
                (static_cast<float>(u_vals[2]) * x0.z) +
                (static_cast<float>(u_vals[3]) * x0.w) +
                (static_cast<float>(u_vals[4]) * x1.x) +
                (static_cast<float>(u_vals[5]) * x1.y) +
                (static_cast<float>(u_vals[6]) * x1.z) +
                (static_cast<float>(u_vals[7]) * x1.w);
    }
  } else {
    // Per-weight fallback for the non-(both-bf16) route: each of gate/up is
    // computed independently — Q8 (integer dot) or bf16 — based on its own
    // flags. A weight that is neither (legacy fp32) keeps its initialized
    // 0.0 dot so the block reduction collapses to 0, matching the pre-fallback
    // (HEAD) semantics. For Q8 the full row dot is computed in thread 0 only;
    // every other lane contributes 0 so the block reduction below collapses to
    // exactly the row dot (not multiplied by block/thread count).
    if (gate_type == core::GgmlType::kBF16) {
      const auto* g_row = static_cast<const hip_bfloat16*>(gate_w) + (m * K);
      const std::size_t K_vec = K / 8;
      const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
      for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
        const uint4 gv = g_vec[vi];
        const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
        const std::size_t k = vi * 8;
        const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
        const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);
        dot_gate += (static_cast<float>(g_vals[0]) * x0.x) +
                    (static_cast<float>(g_vals[1]) * x0.y) +
                    (static_cast<float>(g_vals[2]) * x0.z) +
                    (static_cast<float>(g_vals[3]) * x0.w) +
                    (static_cast<float>(g_vals[4]) * x1.x) +
                    (static_cast<float>(g_vals[5]) * x1.y) +
                    (static_cast<float>(g_vals[6]) * x1.z) +
                    (static_cast<float>(g_vals[7]) * x1.w);
      }
    } else if (gate_type != core::GgmlType::kF32) {
      // Quantized weight (Q8_K now; Q8_0/Q5_K/Q6_K wired later via
      // ::gufo::hip::QuantWarpBlockDot). All lanes cooperate; lane 0 returns this warp's
      // partial row-dot and the block reduction below combines the warps.
      dot_gate = ::gufo::hip::QuantWarpBlockDot(gate_type, gate_w, m, x, K, lane_id, warp_id,
                                   block_threads >> 5u);
    }
    // else (kF32 / legacy fp32) keeps dot_gate == 0.0, matching HEAD semantics.

    if (up_type == core::GgmlType::kBF16) {
      const auto* u_row = static_cast<const hip_bfloat16*>(up_w) + (m * K);
      const std::size_t K_vec = K / 8;
      const auto* u_vec = reinterpret_cast<const uint4*>(u_row);
      for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
        const uint4 uv = u_vec[vi];
        const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);
        const std::size_t k = vi * 8;
        const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
        const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);
        dot_up += (static_cast<float>(u_vals[0]) * x0.x) +
                  (static_cast<float>(u_vals[1]) * x0.y) +
                  (static_cast<float>(u_vals[2]) * x0.z) +
                  (static_cast<float>(u_vals[3]) * x0.w) +
                  (static_cast<float>(u_vals[4]) * x1.x) +
                  (static_cast<float>(u_vals[5]) * x1.y) +
                  (static_cast<float>(u_vals[6]) * x1.z) +
                  (static_cast<float>(u_vals[7]) * x1.w);
      }
    } else if (up_type != core::GgmlType::kF32) {
      // Quantized weight (Q8_K now; Q8_0/Q5_K/Q6_K wired later via
      // ::gufo::hip::QuantWarpBlockDot). All lanes cooperate; lane 0 returns this warp's
      // partial row-dot and the block reduction below combines the warps.
      dot_up = ::gufo::hip::QuantWarpBlockDot(up_type, up_w, m, x, K, lane_id, warp_id,
                                 block_threads >> 5u);
    }
    // else (kF32 / legacy fp32) keeps dot_up == 0.0, matching HEAD semantics.
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up += __shfl_xor(dot_up, offset);
  }

  __shared__ float s_gate_sums[16];
  __shared__ float s_up_sums[16];
  if (lane_id == 0) {
    s_gate_sums[warp_id] = dot_gate;
    s_up_sums[warp_id] = dot_up;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_gate = (lane_id < num_warps) ? s_gate_sums[lane_id] : 0.0F;
    float block_up = (lane_id < num_warps) ? s_up_sums[lane_id] : 0.0F;
    for (int offset = static_cast<int>(num_warps / 2); offset > 0;
         offset >>= 1) {
      block_gate += __shfl_xor(block_gate, offset);
      block_up += __shfl_xor(block_up, offset);
    }
    if (lane_id == 0) {
      const float sig_g = 1.0F / (1.0F + expf(-block_gate));
      out[m] = (block_gate * sig_g) * block_up;
    }
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedQuantSwiGLUGEMVKernel_1Row(const void* __restrict__ gate_w,
                                               core::GgmlType gate_type,
                                               const void* __restrict__ up_w,
                                               core::GgmlType up_type,
                                               const float* __restrict__ x,
                                               float* __restrict__ out,
                                               std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const float dot_gate =
      ::gufo::hip::QuantWarpBlockDot(gate_type, gate_w, m, x, K, lane_id, 0, 1);
  const float dot_up = ::gufo::hip::QuantWarpBlockDot(up_type, up_w, m, x, K, lane_id, 0, 1);

  if (lane_id == 0) {
    const float sig_g = 1.0F / (1.0F + expf(-dot_gate));
    out[m] = (dot_gate * sig_g) * dot_up;
  }
}

template<std::uint32_t WavesPerBlock,
         core::GgmlType StaticType = core::GgmlType::kF32,
         core::GgmlType StaticUpType = StaticType>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedQuantSwiGLUGEMVKernel_2Rows(const void* __restrict__ gate_w,
                                                core::GgmlType gate_type,
                                                const void* __restrict__ up_w,
                                                core::GgmlType up_type,
                                                const float* __restrict__ x,
                                                float* __restrict__ out,
                                                std::size_t M, std::size_t K) {
  if constexpr (StaticType != core::GgmlType::kF32) {
    gate_type = StaticType;
  }
  if constexpr (StaticUpType != core::GgmlType::kF32) {
    up_type = StaticUpType;
  }
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t pair_idx = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = pair_idx * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  if (gate_type == core::GgmlType::kQ8_0 && up_type == core::GgmlType::kQ8_0) {
    const std::size_t num_blocks = K / ::gufo::hip::kQ8_0BlockSize;
    const auto* g0_blocks =
        static_cast<const ::gufo::hip::Q8_0Block*>(gate_w) + (m0 * num_blocks);
    const auto* u0_blocks =
        static_cast<const ::gufo::hip::Q8_0Block*>(up_w) + (m0 * num_blocks);
    const auto* g1_blocks =
        (m1 < M) ? (static_cast<const ::gufo::hip::Q8_0Block*>(gate_w) + (m1 * num_blocks))
                 : nullptr;
    const auto* u1_blocks =
        (m1 < M) ? (static_cast<const ::gufo::hip::Q8_0Block*>(up_w) + (m1 * num_blocks))
                 : nullptr;

    float sum_g0 = 0.0F, sum_u0 = 0.0F;
    float sum_g1 = 0.0F, sum_u1 = 0.0F;

    for (std::size_t b = lane_id; b < num_blocks; b += 32) {
      const float* xb = x + (b * ::gufo::hip::kQ8_0BlockSize);
      const ::gufo::hip::Q8_0Block& gb0 = g0_blocks[b];
      const ::gufo::hip::Q8_0Block& ub0 = u0_blocks[b];
      const float dg0 = __half2float(gb0.d);
      const float du0 = __half2float(ub0.d);

      float blk_g0 = 0.0F, blk_u0 = 0.0F;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        blk_g0 += static_cast<float>(gb0.qs[i]) * xb[i];
        blk_u0 += static_cast<float>(ub0.qs[i]) * xb[i];
      }
      sum_g0 += dg0 * blk_g0;
      sum_u0 += du0 * blk_u0;

      if (g1_blocks != nullptr && u1_blocks != nullptr) {
        const ::gufo::hip::Q8_0Block& gb1 = g1_blocks[b];
        const ::gufo::hip::Q8_0Block& ub1 = u1_blocks[b];
        const float dg1 = __half2float(gb1.d);
        const float du1 = __half2float(ub1.d);
        float blk_g1 = 0.0F, blk_u1 = 0.0F;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          blk_g1 += static_cast<float>(gb1.qs[i]) * xb[i];
          blk_u1 += static_cast<float>(ub1.qs[i]) * xb[i];
        }
        sum_g1 += dg1 * blk_g1;
        sum_u1 += du1 * blk_u1;
      }
    }

    for (int off = 16; off > 0; off >>= 1) {
      sum_g0 += __shfl_xor(sum_g0, off);
      sum_u0 += __shfl_xor(sum_u0, off);
      sum_g1 += __shfl_xor(sum_g1, off);
      sum_u1 += __shfl_xor(sum_u1, off);
    }

    if (lane_id == 0) {
      const float sig_g0 = 1.0F / (1.0F + expf(-sum_g0));
      out[m0] = (sum_g0 * sig_g0) * sum_u0;
      if (m1 < M) {
        const float sig_g1 = 1.0F / (1.0F + expf(-sum_g1));
        out[m1] = (sum_g1 * sig_g1) * sum_u1;
      }
    }
    return;
  }

  const float dot_g0 =
      ::gufo::hip::QuantWarpBlockDot(gate_type, gate_w, m0, x, K, lane_id, 0, 1);
  const float dot_u0 =
      ::gufo::hip::QuantWarpBlockDot(up_type, up_w, m0, x, K, lane_id, 0, 1);
  const float dot_g1 =
      (m1 < M) ? ::gufo::hip::QuantWarpBlockDot(gate_type, gate_w, m1, x, K, lane_id, 0, 1)
               : 0.0F;
  const float dot_u1 =
      (m1 < M) ? ::gufo::hip::QuantWarpBlockDot(up_type, up_w, m1, x, K, lane_id, 0, 1)
               : 0.0F;

  if (lane_id == 0) {
    const float sig_g0 = 1.0F / (1.0F + expf(-dot_g0));
    out[m0] = (dot_g0 * sig_g0) * dot_u0;
    if (m1 < M) {
      const float sig_g1 = 1.0F / (1.0F + expf(-dot_g1));
      out[m1] = (dot_g1 * sig_g1) * dot_u1;
    }
  }
}

void LaunchFusedSwiGLUGEMV(const void* gate_w, core::GgmlType gate_type,
                           const void* up_w, core::GgmlType up_type,
                           const float* x, float* out,
                           std::size_t intermediate_size,
                           std::size_t hidden_size, hipStream_t stream) {
  if (gate_type == core::GgmlType::kBF16 && up_type == core::GgmlType::kBF16 &&
      (hidden_size % 8 == 0)) {
    constexpr std::uint32_t kWavesPerBlock = 4;
    constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;

    if (intermediate_size >= 2 && (intermediate_size % 2 == 0)) {
      const std::size_t num_pairs = intermediate_size / 2;
      const std::size_t grid =
          (num_pairs + kWavesPerBlock - 1) / kWavesPerBlock;
      Wave32FusedSwiGLUGEMVKernel_2Rows<kWavesPerBlock>
          <<<grid, kBlockThreads, 0, stream>>>(
              static_cast<const hip_bfloat16*>(gate_w),
              static_cast<const hip_bfloat16*>(up_w), x, out, intermediate_size,
              hidden_size);
      return;
    }

    const std::size_t grid =
        (intermediate_size + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32FusedSwiGLUGEMVKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(
            static_cast<const hip_bfloat16*>(gate_w),
            static_cast<const hip_bfloat16*>(up_w), x, out, intermediate_size,
            hidden_size);
    return;
  }

  if (gate_type != core::GgmlType::kF32 && up_type != core::GgmlType::kF32) {
    constexpr std::uint32_t kWavesPerBlock = 4;
    constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;

    if (intermediate_size >= 2 && (intermediate_size % 2 == 0)) {
      const std::size_t num_pairs = intermediate_size / 2;
      const std::size_t grid =
          (num_pairs + kWavesPerBlock - 1) / kWavesPerBlock;
      const auto launch =
          [&]<core::GgmlType StaticType,
              core::GgmlType StaticUpType = StaticType>() {
        Wave32FusedQuantSwiGLUGEMVKernel_2Rows<kWavesPerBlock, StaticType,
                                               StaticUpType>
            <<<grid, kBlockThreads, 0, stream>>>(
                gate_w, gate_type, up_w, up_type, x, out, intermediate_size,
                hidden_size);
      };
      if (gate_type == up_type) {
        switch (gate_type) {
          case core::GgmlType::kQ4_K:
            launch.template operator()<core::GgmlType::kQ4_K>();
            return;
          case core::GgmlType::kQ5_K:
            launch.template operator()<core::GgmlType::kQ5_K>();
            return;
          case core::GgmlType::kIQ4_XS:
            launch.template operator()<core::GgmlType::kIQ4_XS>();
            return;
          default:
            break;
        }
      }
      // Mixed-format FFNs benefit from specializing each projection while
      // keeping the same dot products and SwiGLU arithmetic.
      if (gate_type == core::GgmlType::kQ4_K) {
        if (up_type == core::GgmlType::kQ5_K) {
          launch.template operator()<core::GgmlType::kQ4_K,
                                     core::GgmlType::kQ5_K>();
          return;
        }
        if (up_type == core::GgmlType::kIQ4_XS) {
          launch.template operator()<core::GgmlType::kQ4_K,
                                     core::GgmlType::kIQ4_XS>();
          return;
        }
      }
      if (gate_type == core::GgmlType::kIQ4_XS) {
        if (up_type == core::GgmlType::kQ4_K) {
          launch.template operator()<core::GgmlType::kIQ4_XS,
                                     core::GgmlType::kQ4_K>();
          return;
        }
        if (up_type == core::GgmlType::kQ5_K) {
          launch.template operator()<core::GgmlType::kIQ4_XS,
                                     core::GgmlType::kQ5_K>();
          return;
        }
      }
      launch.template operator()<core::GgmlType::kF32>();
      return;
    }

    const std::size_t grid =
        (intermediate_size + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32FusedQuantSwiGLUGEMVKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(gate_w, gate_type, up_w, up_type,
                                             x, out, intermediate_size,
                                             hidden_size);
    return;
  }

  FastFusedSwiGLUGEMVBlockKernel<<<intermediate_size, 512, 0, stream>>>(
      gate_w, gate_type, up_w, up_type, x, out, intermediate_size, hidden_size);
}

// ---- flash-next MoE epilogue (HC geometry, fn_compat) ----
namespace fn_inner {
__device__ __forceinline__ float Materialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
}  // namespace fn_inner
using fn_inner::Materialize;
using fn_inner::SigmoidF;

namespace fn_compat {

__global__ void MoeEpilogueKernel(const float* expert_out, const float* weights,
                                  const float* shared_out, const float* gate,
                                  std::uint32_t gate_stride, float* out,
                                  std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= dim) {
    return;
  }
  constexpr std::uint32_t kMaxSlots = 16;
  if (k <= kMaxSlots) {
    const float* rows = expert_out + static_cast<std::size_t>(t) * k * dim + i;
    float w[kMaxSlots];
    float v[kMaxSlots];
#pragma unroll
    for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
      w[s] = s < k ? weights[t * k + s] : 0.0f;
      v[s] = s < k ? rows[static_cast<std::size_t>(s) * dim] : 0.0f;
    }
    const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
    const float sh = shared_out[idx];
    const float g = gate[static_cast<std::size_t>(t) * gate_stride];
    float acc = Materialize(0.0f);
#pragma unroll
    for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
      if (s < k) {
        acc = Materialize(acc + w[s] * v[s]);
      }
    }
    out[idx] = acc + SigmoidF(g) * sh;
    return;
  }
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < k; ++s) {
    acc += weights[t * k + s] *
           expert_out[(static_cast<std::size_t>(t) * k + s) * dim + i];
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  out[idx] = acc + SigmoidF(gate[static_cast<std::size_t>(t) * gate_stride]) *
                       shared_out[idx];
}

void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(MoeEpilogueKernel, dim3(n_tokens, Blocks(dim)),
                     dim3(kThreads), 0, stream, expert_out, weights, shared,
                     gate, gate_stride, out, k, dim);
}

}  // namespace fn_compat

}  // namespace gufo::core::compute::gemm

#endif  // defined(ENGINE_ENABLE_HIP)
