#if defined(ENGINE_ENABLE_HIP)
// Unified decode GEMV (Phase B, knife 1): gemma4 Wave32 sdot4 base, 27B block
// shapes, flash-next SmallGemm. Mechanical merge only; see gemm.hpp.
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
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::core::compute::gemm {


using device::Dot4S8S8;
using device::Dot4U4S8;
using device::GeluTanhOne;
using device::WaveRowTileDot;
using device::WaveSum;
using device::XorLaneF;
using device::XorLaneI;

__global__ void __launch_bounds__(256)
    QuantizeQ8_1Kernel(const float* __restrict__ x,
                       BlockQ8_1* __restrict__ out,
                       std::uint32_t n_blocks) {
  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t blk = blockIdx.x * 8u + warp_id;
  if (blk >= n_blocks) return;

  const float v = x[static_cast<std::size_t>(blk) * 32u + lane];
  float amax = fabsf(v);
  amax = fmaxf(amax, XorLaneF(amax, 16));
  amax = fmaxf(amax, XorLaneF(amax, 8));
  amax = fmaxf(amax, XorLaneF(amax, 4));
  amax = fmaxf(amax, XorLaneF(amax, 2));
  amax = fmaxf(amax, XorLaneF(amax, 1));

  const float d = amax * (1.0F / 127.0F);
  const float id = (amax > 0.0F) ? (127.0F / amax) : 0.0F;
  const int q = __float2int_rn(v * id);

  const int q_x1 = XorLaneI(q, 1);
  const int q_x2 = XorLaneI(q, 2);
  const int q_x3 = XorLaneI(q_x1, 2);
  const int packed = (q & 0xFF) | ((q_x1 & 0xFF) << 8) |
                     ((q_x2 & 0xFF) << 16) | ((q_x3 & 0xFF) << 24);

  int qsum = q;
  qsum += XorLaneI(qsum, 16);
  qsum += XorLaneI(qsum, 8);
  qsum += XorLaneI(qsum, 4);
  qsum += XorLaneI(qsum, 2);
  qsum += XorLaneI(qsum, 1);

  if ((lane & 3u) == 0u) {
    out[blk].qs[lane >> 2] = packed;
  }
  if (lane == 0u) {
    out[blk].d = d;
    out[blk].s = d * static_cast<float>(qsum);
  }
}

namespace detail {
BlockQ8_1* g_gemv_q8_scratch = nullptr;
std::size_t g_gemv_q8_scratch_cap = 0;
}  // namespace detail

namespace detail {
BlockQ8_1* GemvQ8Scratch() { return g_gemv_q8_scratch; }
std::size_t GemvQ8ScratchCap() { return g_gemv_q8_scratch_cap; }
BlockQ8_1* QuantizeToQ8Scratch(const float* x, std::size_t n_blocks,
                                      hipStream_t stream) {
  if (n_blocks == 0) return nullptr;
  if (n_blocks > detail::g_gemv_q8_scratch_cap) {
    const std::size_t alloc_blks = n_blocks < 4096u ? 4096u : n_blocks * 2u;
    void* p = nullptr;
    if (hipMalloc(&p, alloc_blks * sizeof(BlockQ8_1)) != hipSuccess) {
      return nullptr;
    }
    if (detail::g_gemv_q8_scratch != nullptr) (void)hipFree(detail::g_gemv_q8_scratch);
    detail::g_gemv_q8_scratch = static_cast<BlockQ8_1*>(p);
    detail::g_gemv_q8_scratch_cap = alloc_blks;
  }
  const std::uint32_t nb = static_cast<std::uint32_t>(n_blocks);
  hipLaunchKernelGGL(QuantizeQ8_1Kernel, dim3((nb + 7u) / 8u), dim3(256), 0,
                     stream, x, detail::g_gemv_q8_scratch, nb);
  return detail::g_gemv_q8_scratch;
}
}  // namespace detail

__global__ void __launch_bounds__(256)
    GemvWave32BatchedQ8Kernel(const std::uint8_t* __restrict__ w,
                              std::uint64_t row_bytes, std::uint32_t total_blk,
                              std::uint32_t tile_blk,
                              const BlockQ8_1* __restrict__ x_q8,
                              float* __restrict__ y, std::uint32_t n_tokens,
                              std::uint32_t m, float softcap) {
  extern __shared__ __align__(8) unsigned char smem_gemv_q8[];
  auto* sx = reinterpret_cast<BlockQ8_1*>(smem_gemv_q8);

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t b_grp = lane >> 2;
  const std::uint32_t sub = lane & 3u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  const std::uint8_t* wrow =
      row < m ? (w + static_cast<std::size_t>(row) * row_bytes) : nullptr;

  float sum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  for (std::uint32_t b_off = 0; b_off < total_blk; b_off += tile_blk) {
    const std::uint32_t tk_blk =
        (total_blk - b_off) < tile_blk ? (total_blk - b_off) : tile_blk;
    const std::uint32_t words2 = tk_blk * 5u;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const auto* src2 = reinterpret_cast<const uint2*>(
          x_q8 + static_cast<std::size_t>(t) * total_blk + b_off);
      auto* dst2 = reinterpret_cast<uint2*>(sx + t * tile_blk);
      for (std::uint32_t i = threadIdx.x; i < words2; i += 256u) {
        dst2[i] = src2[i];
      }
    }
    __syncthreads();

    if (row < m) {
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
          for (std::uint32_t t = 0; t < n_tokens; ++t) {
            const BlockQ8_1& xblk = sx[t * tile_blk + b];
            int dot = Dot4U4S8(vi0, xblk.qs[sub], 0);
            dot = Dot4U4S8(vi1, xblk.qs[4u + sub], dot);
            sum[t] += d4 * (xblk.d * static_cast<float>(dot) - 2.0F * xblk.s);
          }
        }
      }
    }
    if (b_off + tile_blk < total_blk) __syncthreads();
  }

  if (row < m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      float v = WaveSum(sum[t]);
      if (lane == 0u) {
        if (softcap > 0.0F) v = tanhf(v / softcap) * softcap;
        y[static_cast<std::size_t>(t) * m + row] = v;
      }
    }
  }
}

__global__ void __launch_bounds__(256)
    GemvWave32BatchedQ8_0Kernel(const std::uint8_t* __restrict__ w,
                                std::uint64_t row_bytes,
                                std::uint32_t total_blk, std::uint32_t tile_blk,
                                const BlockQ8_1* __restrict__ x_q8,
                                float* __restrict__ y, std::uint32_t n_tokens,
                                std::uint32_t m, float softcap) {
  extern __shared__ __align__(8) unsigned char smem_gemv_q80[];
  auto* sx = reinterpret_cast<BlockQ8_1*>(smem_gemv_q80);

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t b_grp = lane >> 3;  // 0..3 (4 blocks per wave step)
  const std::uint32_t sub = lane & 7u;    // 0..7 (4 int8s per lane = 32B/block)
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  const std::uint8_t* wrow =
      row < m ? (w + static_cast<std::size_t>(row) * row_bytes) : nullptr;

  float sum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  for (std::uint32_t b_off = 0; b_off < total_blk; b_off += tile_blk) {
    const std::uint32_t tk_blk =
        (total_blk - b_off) < tile_blk ? (total_blk - b_off) : tile_blk;
    const std::uint32_t words2 = tk_blk * 5u;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const auto* src2 = reinterpret_cast<const uint2*>(
          x_q8 + static_cast<std::size_t>(t) * total_blk + b_off);
      auto* dst2 = reinterpret_cast<uint2*>(sx + t * tile_blk);
      for (std::uint32_t i = threadIdx.x; i < words2; i += 256u) {
        dst2[i] = src2[i];
      }
    }
    __syncthreads();

    if (row < m) {
      const std::uint8_t* wbase = wrow + static_cast<std::size_t>(b_off) * 34u;
      for (std::uint32_t b0 = 0; b0 < tk_blk; b0 += 4u) {
        const std::uint32_t b = b0 + b_grp;
        if (b < tk_blk) {
          const auto* bp =
              reinterpret_cast<const std::uint16_t*>(wbase + b * 34u);
          const float d8 = __half2float(*reinterpret_cast<const __half*>(bp));
          const int w32 = static_cast<int>(bp[1u + 2u * sub]) |
                          (static_cast<int>(bp[2u + 2u * sub]) << 16);
          for (std::uint32_t t = 0; t < n_tokens; ++t) {
            const BlockQ8_1& xblk = sx[t * tile_blk + b];
            const int dot = Dot4S8S8(w32, xblk.qs[sub], 0);
            sum[t] += (d8 * xblk.d) * static_cast<float>(dot);
          }
        }
      }
    }
    if (b_off + tile_blk < total_blk) __syncthreads();
  }

  if (row < m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      float v = WaveSum(sum[t]);
      if (lane == 0u) {
        if (softcap > 0.0F) v = tanhf(v / softcap) * softcap;
        y[static_cast<std::size_t>(t) * m + row] = v;
      }
    }
  }
}

__global__ void __launch_bounds__(256)
    GemvGateUpSwigluBatchedQ8Kernel(const std::uint8_t* __restrict__ w_gate,
                                    const std::uint8_t* __restrict__ w_up,
                                    std::uint64_t row_bytes,
                                    std::uint32_t total_blk,
                                    std::uint32_t tile_blk,
                                    const BlockQ8_1* __restrict__ x_q8,
                                    float* __restrict__ out_act,
                                    std::uint32_t n_tokens, std::uint32_t m) {
  extern __shared__ __align__(8) unsigned char smem_gemv_q8[];
  auto* sx = reinterpret_cast<BlockQ8_1*>(smem_gemv_q8);

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t b_grp = lane >> 2;
  const std::uint32_t sub = lane & 3u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  const std::size_t off = static_cast<std::size_t>(row) * row_bytes;

  float sg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float su[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (std::uint32_t b_off = 0; b_off < total_blk; b_off += tile_blk) {
    const std::uint32_t tk_blk =
        (total_blk - b_off) < tile_blk ? (total_blk - b_off) : tile_blk;
    const std::uint32_t words2 = tk_blk * 5u;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const auto* src2 = reinterpret_cast<const uint2*>(
          x_q8 + static_cast<std::size_t>(t) * total_blk + b_off);
      auto* dst2 = reinterpret_cast<uint2*>(sx + t * tile_blk);
      for (std::uint32_t i = threadIdx.x; i < words2; i += 256u) {
        dst2[i] = src2[i];
      }
    }
    __syncthreads();

    if (row < m) {
      const std::uint8_t* gbase =
          w_gate + off + static_cast<std::size_t>(b_off) * 18u;
      const std::uint8_t* ubase =
          w_up + off + static_cast<std::size_t>(b_off) * 18u;
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

          for (std::uint32_t t = 0; t < n_tokens; ++t) {
            const BlockQ8_1& xblk = sx[t * tile_blk + b];
            const int xlo = xblk.qs[sub];
            const int xhi = xblk.qs[4u + sub];
            int gdot = Dot4U4S8(gvi0, xlo, 0);
            gdot = Dot4U4S8(gvi1, xhi, gdot);
            int udot = Dot4U4S8(uvi0, xlo, 0);
            udot = Dot4U4S8(uvi1, xhi, udot);
            sg[t] += gd4 * (xblk.d * static_cast<float>(gdot) - 2.0F * xblk.s);
            su[t] += ud4 * (xblk.d * static_cast<float>(udot) - 2.0F * xblk.s);
          }
        }
      }
    }
    if (b_off + tile_blk < total_blk) __syncthreads();
  }

  if (row < m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float g = WaveSum(sg[t]);
      const float u = WaveSum(su[t]);
      if (lane == 0u) {
        out_act[static_cast<std::size_t>(t) * m + row] = GeluTanhOne(g) * u;
      }
    }
  }
}

__global__ void __launch_bounds__(256)
    GemvWave32SingleKernel(const std::uint8_t* __restrict__ w, int type,
                           std::uint64_t row_bytes, std::uint32_t cols,
                           const float* __restrict__ x, float* __restrict__ y,
                           std::uint32_t m, float softcap) {
  extern __shared__ float sx[];
  for (std::uint32_t i = threadIdx.x * 4u; i < cols; i += 256u * 4u) {
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
  }
  __syncthreads();

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  if (row >= m) return;

  const std::uint8_t* wrow = w + static_cast<std::size_t>(row) * row_bytes;
  float sum = WaveRowTileDot(wrow, type, 0u, cols, lane, sx);
  sum = WaveSum(sum);
  if (lane == 0) {
    if (softcap > 0.0F) sum = tanhf(sum / softcap) * softcap;
    y[row] = sum;
  }
}

__global__ void __launch_bounds__(256)
    GemvWave32BatchedKernel(const std::uint8_t* __restrict__ w, int type,
                            std::uint64_t row_bytes, std::uint32_t cols,
                            const float* __restrict__ x, float* __restrict__ y,
                            std::uint32_t n_tokens, std::uint32_t m,
                            float softcap) {
  constexpr std::uint32_t kTileK = 1024;
  __shared__ float sx[8][kTileK];

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  const std::uint8_t* wrow =
      row < m ? (w + static_cast<std::size_t>(row) * row_bytes) : nullptr;

  float sum[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  for (std::uint32_t k0 = 0; k0 < cols; k0 += kTileK) {
    const std::uint32_t tk = (cols - k0) < kTileK ? (cols - k0) : kTileK;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float* xt = x + static_cast<std::size_t>(t) * cols + k0;
      for (std::uint32_t i = threadIdx.x * 4u; i < tk; i += 256u * 4u) {
        *reinterpret_cast<float4*>(&sx[t][i]) =
            *reinterpret_cast<const float4*>(xt + i);
      }
    }
    __syncthreads();

    if (row < m) {
      if (type == 2) {
        const std::uint32_t nblk = tk >> 5;
        const std::uint32_t b_grp = lane >> 2;
        const std::uint32_t sub = lane & 3u;
        const std::uint8_t* wbase =
            wrow + static_cast<std::size_t>(k0 >> 5) * 18;
        for (std::uint32_t b0 = 0; b0 < nblk; b0 += 8) {
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
            for (std::uint32_t t = 0; t < n_tokens; ++t) {
              const float4 xlo = *reinterpret_cast<const float4*>(
                  &sx[t][b * 32u + 4u * sub]);
              const float4 xhi = *reinterpret_cast<const float4*>(
                  &sx[t][b * 32u + 16u + 4u * sub]);
              const float acc = wl0 * xlo.x + wh0 * xhi.x + wl1 * xlo.y +
                                wh1 * xhi.y + wl2 * xlo.z + wh2 * xhi.z +
                                wl3 * xlo.w + wh3 * xhi.w;
              sum[t] += d * acc;
            }
          }
        }
      } else if (type == 0) {
        const auto* r = reinterpret_cast<const float*>(wrow) + k0;
        for (std::uint32_t j = lane * 4u; j < tk; j += 128u) {
          const float4 wv = *reinterpret_cast<const float4*>(r + j);
          for (std::uint32_t t = 0; t < n_tokens; ++t) {
            const float4 xv = *reinterpret_cast<const float4*>(&sx[t][j]);
            sum[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
          }
        }
      } else {
        for (std::uint32_t t = 0; t < n_tokens; ++t) {
          sum[t] += WaveRowTileDot(wrow, type, k0, tk, lane, sx[t]);
        }
      }
    }
    if (k0 + kTileK < cols) __syncthreads();
  }

  if (row < m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      float v = WaveSum(sum[t]);
      if (lane == 0) {
        if (softcap > 0.0F) v = tanhf(v / softcap) * softcap;
        y[static_cast<std::size_t>(t) * m + row] = v;
      }
    }
  }
}

__global__ void __launch_bounds__(256)
    GemvGateUpSwigluSingleKernel(const std::uint8_t* __restrict__ w_gate,
                                 const std::uint8_t* __restrict__ w_up,
                                 int type, std::uint64_t row_bytes,
                                 std::uint32_t cols,
                                 const float* __restrict__ x,
                                 float* __restrict__ out_act,
                                 std::uint32_t m) {
  extern __shared__ float sx[];
  for (std::uint32_t i = threadIdx.x * 4u; i < cols; i += 256u * 4u) {
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
  }
  __syncthreads();

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  if (row >= m) return;

  const std::size_t off = static_cast<std::size_t>(row) * row_bytes;
  float g = WaveRowTileDot(w_gate + off, type, 0u, cols, lane, sx);
  float u = WaveRowTileDot(w_up + off, type, 0u, cols, lane, sx);
  g = WaveSum(g);
  u = WaveSum(u);
  if (lane == 0) {
    out_act[row] = GeluTanhOne(g) * u;
  }
}

__global__ void __launch_bounds__(256)
    GemvGateUpSwigluBatchedKernel(const std::uint8_t* __restrict__ w_gate,
                                  const std::uint8_t* __restrict__ w_up,
                                  int type, std::uint64_t row_bytes,
                                  std::uint32_t cols,
                                  const float* __restrict__ x,
                                  float* __restrict__ out_act,
                                  std::uint32_t n_tokens, std::uint32_t m) {
  constexpr std::uint32_t kTileK = 1024;
  __shared__ float sx[8][kTileK];

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  const std::size_t off = static_cast<std::size_t>(row) * row_bytes;

  float sg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float su[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (std::uint32_t k0 = 0; k0 < cols; k0 += kTileK) {
    const std::uint32_t tk = (cols - k0) < kTileK ? (cols - k0) : kTileK;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float* xt = x + static_cast<std::size_t>(t) * cols + k0;
      for (std::uint32_t i = threadIdx.x * 4u; i < tk; i += 256u * 4u) {
        *reinterpret_cast<float4*>(&sx[t][i]) =
            *reinterpret_cast<const float4*>(xt + i);
      }
    }
    __syncthreads();
    if (row < m) {
      if (type == 2) {
        const std::uint32_t nblk = tk >> 5;
        const std::uint32_t b_grp = lane >> 2;
        const std::uint32_t sub = lane & 3u;
        const std::size_t boff = static_cast<std::size_t>(k0 >> 5) * 18u;
        const std::uint8_t* gbase = w_gate + off + boff;
        const std::uint8_t* ubase = w_up + off + boff;
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

            for (std::uint32_t t = 0; t < n_tokens; ++t) {
              const float4 xlo = *reinterpret_cast<const float4*>(
                  &sx[t][b * 32u + 4u * sub]);
              const float4 xhi = *reinterpret_cast<const float4*>(
                  &sx[t][b * 32u + 16u + 4u * sub]);
              sg[t] += gd * (gwl0 * xlo.x + gwh0 * xhi.x + gwl1 * xlo.y +
                             gwh1 * xhi.y + gwl2 * xlo.z + gwh2 * xhi.z +
                             gwl3 * xlo.w + gwh3 * xhi.w);
              su[t] += ud * (uwl0 * xlo.x + uwh0 * xhi.x + uwl1 * xlo.y +
                             uwh1 * xhi.y + uwl2 * xlo.z + uwh2 * xhi.z +
                             uwl3 * xlo.w + uwh3 * xhi.w);
            }
          }
        }
      } else {
        for (std::uint32_t t = 0; t < n_tokens; ++t) {
          sg[t] += WaveRowTileDot(w_gate + off, type, k0, tk, lane, sx[t]);
          su[t] += WaveRowTileDot(w_up + off, type, k0, tk, lane, sx[t]);
        }
      }
    }
    if (k0 + kTileK < cols) __syncthreads();
  }
  if (row < m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float g = WaveSum(sg[t]);
      const float u = WaveSum(su[t]);
      if (lane == 0) {
        out_act[static_cast<std::size_t>(t) * m + row] = GeluTanhOne(g) * u;
      }
    }
  }
}

__global__ void GemvQKernel(const void* w, int type, std::uint64_t row_bytes,
                           std::uint64_t cols, const float* x, float* y) {
  const std::uint32_t row = blockIdx.x;
  const auto* base = static_cast<const std::uint8_t*>(w) + (std::size_t)row * row_bytes;
  float sum = 0.0F;
  if (type == 0) {  // F32
    const auto* r = reinterpret_cast<const float*>(base);
    for (std::uint64_t i = threadIdx.x; i < cols; i += blockDim.x)
      sum += r[i] * x[i];
  } else if (type == 1) {  // F16
    const auto* r = reinterpret_cast<const __half*>(base);
    for (std::uint64_t i = threadIdx.x; i < cols; i += blockDim.x)
      sum += __half2float(r[i]) * x[i];
  } else if (type == 8) {  // Q8_0
    for (std::uint64_t b = threadIdx.x; b < cols / 32; b += blockDim.x) {
      const std::uint8_t* blk = base + (std::size_t)34 * b;
      const float d = __half2float(*reinterpret_cast<const __half*>(blk));
      const auto* qs = reinterpret_cast<const std::int8_t*>(blk + 2);
      const std::size_t off = (std::size_t)b * 32;
      #pragma unroll
      for (int q = 0; q < 32; ++q)
        sum += d * static_cast<float>(qs[q]) * x[off + q];
    }
  } else {  // Q4_0 (type 2): lo nibble = q, hi nibble = q+16
    for (std::uint64_t b = threadIdx.x; b < cols / 32; b += blockDim.x) {
      const std::uint8_t* blk = base + (std::size_t)18 * b;
      const float d = __half2float(*reinterpret_cast<const __half*>(blk));
      const std::uint8_t* qs = blk + 2;
      const std::size_t off = (std::size_t)b * 32;
      #pragma unroll
      for (int q = 0; q < 16; ++q) {
        const float lo = d * (static_cast<float>(qs[q] & 0xF) - 8.0F);
        const float hi = d * (static_cast<float>(qs[q] >> 4) - 8.0F);
        sum += lo * x[off + q] + hi * x[off + q + 16];
      }
    }
  }
  __shared__ float red[256];
  red[threadIdx.x] = sum;
  __syncthreads();
  for (std::uint32_t s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) y[row] = red[0];
}

__global__ void SoftCapKernel(float* x, float cap, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       i < count; i += stride) {
    x[i] = tanhf(x[i] / cap) * cap;
  }
}

void SoftCap(float* x, float cap, std::size_t count, hipStream_t stream) {
  if (count == 0 || cap <= 0.0F)
    return;
  hipLaunchKernelGGL(SoftCapKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, cap, count);
}

void GemvBatched(const void* wdev, int type, std::uint64_t row_bytes,
                 std::uint64_t cols, const float* x, float* y,
                 std::uint32_t n_tokens, std::uint32_t m, float softcap,
                 hipStream_t stream) {
  if (m == 0 || n_tokens == 0) return;
  const bool type_ok =
      (type == 0 || type == 1 || type == 2 || type == 8 || type == 30);
  if (type_ok && (cols % 32ull) == 0ull && n_tokens <= 8u &&
      getenv("GEMMA4_NO_WAVE32_GEMV") == nullptr) {
    const dim3 grid((m + 7u) / 8u);
    const dim3 block(256);
    const bool q8_batch_ok = (getenv("GEMMA4_NO_Q8_BATCH") == nullptr);
    const bool q8_single = (getenv("GEMMA4_Q8_SINGLE") != nullptr);
    if (type == 8 && q8_batch_ok &&
        (n_tokens > 1u || getenv("GEMMA4_NO_Q8_SINGLE_HEAD") == nullptr)) {
      const std::uint32_t total_blk = static_cast<std::uint32_t>(cols >> 5);
      const std::uint32_t tile_blk = total_blk < 128u ? total_blk : 128u;
      if (const BlockQ8_1* x_q8 = detail::QuantizeToQ8Scratch(
              x, static_cast<std::size_t>(n_tokens) * total_blk, stream)) {
        const std::size_t smem =
            static_cast<std::size_t>(n_tokens) * tile_blk * sizeof(BlockQ8_1);
        hipLaunchKernelGGL(
            GemvWave32BatchedQ8_0Kernel, grid, block, smem, stream,
            static_cast<const std::uint8_t*>(wdev), row_bytes, total_blk,
            tile_blk, x_q8, y, n_tokens, m, softcap);
        return;
      }
    }
    if (n_tokens == 1u && cols <= 12288ull &&
        !(type == 2 && q8_batch_ok && (q8_single || m >= 65536u))) {
      hipLaunchKernelGGL(
          GemvWave32SingleKernel, grid, block,
          static_cast<std::size_t>(cols) * sizeof(float), stream,
          static_cast<const std::uint8_t*>(wdev), type, row_bytes,
          static_cast<std::uint32_t>(cols), x, y, m, softcap);
      return;
    }
    if (type == 2 && q8_batch_ok) {
      const std::uint32_t total_blk = static_cast<std::uint32_t>(cols >> 5);
      const std::uint32_t tile_blk = total_blk < 128u ? total_blk : 128u;
      if (const BlockQ8_1* x_q8 = detail::QuantizeToQ8Scratch(
              x, static_cast<std::size_t>(n_tokens) * total_blk, stream)) {
        const std::size_t smem =
            static_cast<std::size_t>(n_tokens) * tile_blk * sizeof(BlockQ8_1);
        hipLaunchKernelGGL(
            GemvWave32BatchedQ8Kernel, grid, block, smem, stream,
            static_cast<const std::uint8_t*>(wdev), row_bytes, total_blk,
            tile_blk, x_q8, y, n_tokens, m, softcap);
        return;
      }
    }
    hipLaunchKernelGGL(
        GemvWave32BatchedKernel, grid, block, 0, stream,
        static_cast<const std::uint8_t*>(wdev), type, row_bytes,
        static_cast<std::uint32_t>(cols), x, y, n_tokens, m, softcap);
    return;
  }
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    hipLaunchKernelGGL(GemvQKernel, dim3(m), dim3(kThreads), 0, stream, wdev,
                       type, row_bytes, cols,
                       x + static_cast<std::size_t>(t) * cols,
                       y + static_cast<std::size_t>(t) * m);
  }
  if (softcap > 0.0F) {
    SoftCap(y, softcap, static_cast<std::size_t>(n_tokens) * m, stream);
  }
}

void GemvGateUpSwiglu(const void* w_gate, const void* w_up, int type,
                      std::uint64_t row_bytes, std::uint64_t cols,
                      const float* x, float* out_act, std::uint32_t n_tokens,
                      std::uint32_t m, hipStream_t stream) {
  if (m == 0 || n_tokens == 0) return;
  const bool type_ok =
      (type == 0 || type == 1 || type == 2 || type == 8 || type == 30);
  if (type_ok && (cols % 32ull) == 0ull && n_tokens <= 8u &&
      getenv("GEMMA4_NO_WAVE32_GEMV") == nullptr) {
    const dim3 grid((m + 7u) / 8u);
    const dim3 block(256);
    const bool q8_batch_ok = (getenv("GEMMA4_NO_Q8_BATCH") == nullptr);
    const bool q8_single = (getenv("GEMMA4_Q8_SINGLE") != nullptr);
    if (n_tokens == 1u && cols <= 12288ull &&
        !(type == 2 && q8_batch_ok && q8_single)) {
      hipLaunchKernelGGL(
          GemvGateUpSwigluSingleKernel, grid, block,
          static_cast<std::size_t>(cols) * sizeof(float), stream,
          static_cast<const std::uint8_t*>(w_gate),
          static_cast<const std::uint8_t*>(w_up), type, row_bytes,
          static_cast<std::uint32_t>(cols), x, out_act, m);
      return;
    }
    if (type == 2 && q8_batch_ok) {
      const std::uint32_t total_blk = static_cast<std::uint32_t>(cols >> 5);
      const std::uint32_t tile_blk = total_blk < 128u ? total_blk : 128u;
      if (const BlockQ8_1* x_q8 = detail::QuantizeToQ8Scratch(
              x, static_cast<std::size_t>(n_tokens) * total_blk, stream)) {
        const std::size_t smem =
            static_cast<std::size_t>(n_tokens) * tile_blk * sizeof(BlockQ8_1);
        hipLaunchKernelGGL(
            GemvGateUpSwigluBatchedQ8Kernel, grid, block, smem, stream,
            static_cast<const std::uint8_t*>(w_gate),
            static_cast<const std::uint8_t*>(w_up), row_bytes, total_blk,
            tile_blk, x_q8, out_act, n_tokens, m);
        return;
      }
    }
    hipLaunchKernelGGL(
        GemvGateUpSwigluBatchedKernel, grid, block, 0, stream,
        static_cast<const std::uint8_t*>(w_gate),
        static_cast<const std::uint8_t*>(w_up), type, row_bytes,
        static_cast<std::uint32_t>(cols), x, out_act, n_tokens, m);
    return;
  }
  // Fallback should not be reached on Gemma 4 shapes, but keep safe.
}

void GemvFused(const void* wdev, int type, std::uint64_t row_bytes,
               std::uint64_t cols, const float* x, float* y, std::uint32_t m,
               hipStream_t stream) {
  GemvBatched(wdev, type, row_bytes, cols, x, y, 1u, m, 0.0F, stream);
}

void GemvF16Fused(const __half* wdev, std::uint64_t cols, const float* x,
                  float* y, std::uint32_t m, hipStream_t stream) {
  GemvBatched(wdev, 1, cols * sizeof(__half), cols, x, y, 1u, m, 0.0F, stream);
}

__global__ void __launch_bounds__(256) GemvQkvSingleKernel(
    const std::uint8_t* __restrict__ w_q, int type_q, std::uint64_t rb_q,
    std::uint32_t qdim, const std::uint8_t* __restrict__ w_k, int type_k,
    std::uint64_t rb_k, const std::uint8_t* __restrict__ w_v, int type_v,
    std::uint64_t rb_v, std::uint32_t kvdim, std::uint32_t total_m,
    std::uint32_t cols, const float* __restrict__ x,
    float* __restrict__ q_out, float* __restrict__ k_out,
    float* __restrict__ v_out) {
  extern __shared__ float sx[];
  for (std::uint32_t i = threadIdx.x * 4u; i < cols; i += 256u * 4u) {
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
  }
  __syncthreads();

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;
  if (row >= total_m) return;

  const std::uint8_t* wrow = nullptr;
  int tp = 2;
  float* dst = nullptr;
  if (row < qdim) {
    wrow = w_q + static_cast<std::size_t>(row) * rb_q;
    tp = type_q;
    dst = q_out + row;
  } else if (row < qdim + kvdim) {
    const std::uint32_t lr = row - qdim;
    wrow = w_k + static_cast<std::size_t>(lr) * rb_k;
    tp = type_k;
    dst = k_out + lr;
  } else {
    const std::uint32_t lr = row - qdim - kvdim;
    wrow = w_v + static_cast<std::size_t>(lr) * rb_v;
    tp = type_v;
    dst = v_out + lr;
  }

  float sum = WaveRowTileDot(wrow, tp, 0u, cols, lane, sx);
  sum = WaveSum(sum);
  if (lane == 0u) *dst = sum;
}

__global__ void __launch_bounds__(256) GemvQkvBatchedKernel(
    const std::uint8_t* __restrict__ w_q, int type_q, std::uint64_t rb_q,
    std::uint32_t qdim, const std::uint8_t* __restrict__ w_k, int type_k,
    std::uint64_t rb_k, const std::uint8_t* __restrict__ w_v, int type_v,
    std::uint64_t rb_v, std::uint32_t kvdim, std::uint32_t total_m,
    std::uint32_t cols, const float* __restrict__ x,
    float* __restrict__ q_out, float* __restrict__ k_out,
    float* __restrict__ v_out, std::uint32_t n_tokens) {
  constexpr std::uint32_t kTileK = 1024;
  __shared__ float sx[8][kTileK];

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;

  const std::uint8_t* wrow = nullptr;
  int tp = 2;
  float* out_base = nullptr;
  std::uint32_t out_stride = 0;
  std::uint32_t local_row = 0;
  if (row < qdim) {
    local_row = row;
    wrow = w_q + static_cast<std::size_t>(local_row) * rb_q;
    tp = type_q;
    out_base = q_out;
    out_stride = qdim;
  } else if (row < qdim + kvdim) {
    local_row = row - qdim;
    wrow = w_k + static_cast<std::size_t>(local_row) * rb_k;
    tp = type_k;
    out_base = k_out;
    out_stride = kvdim;
  } else if (row < total_m) {
    local_row = row - qdim - kvdim;
    wrow = w_v + static_cast<std::size_t>(local_row) * rb_v;
    tp = type_v;
    out_base = v_out;
    out_stride = kvdim;
  }

  float sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (std::uint32_t k0 = 0; k0 < cols; k0 += kTileK) {
    const std::uint32_t tk = (cols - k0) < kTileK ? (cols - k0) : kTileK;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float* xt = x + static_cast<std::size_t>(t) * cols + k0;
      for (std::uint32_t i = threadIdx.x * 4u; i < tk; i += 256u * 4u) {
        *reinterpret_cast<float4*>(&sx[t][i]) =
            *reinterpret_cast<const float4*>(xt + i);
      }
    }
    __syncthreads();
    if (row < total_m) {
      if (tp == 2) {
        const std::uint32_t nblk = tk >> 5;
        const std::uint32_t b_grp = lane >> 2;
        const std::uint32_t sub = lane & 3u;
        const std::uint8_t* wbase =
            wrow + static_cast<std::size_t>(k0 >> 5) * 18;
        for (std::uint32_t b0 = 0; b0 < nblk; b0 += 8) {
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
            for (std::uint32_t t = 0; t < n_tokens; ++t) {
              const float4 xlo = *reinterpret_cast<const float4*>(
                  &sx[t][b * 32u + 4u * sub]);
              const float4 xhi = *reinterpret_cast<const float4*>(
                  &sx[t][b * 32u + 16u + 4u * sub]);
              const float acc = wl0 * xlo.x + wh0 * xhi.x + wl1 * xlo.y +
                                wh1 * xhi.y + wl2 * xlo.z + wh2 * xhi.z +
                                wl3 * xlo.w + wh3 * xhi.w;
              sum[t] += d * acc;
            }
          }
        }
      } else {
        for (std::uint32_t t = 0; t < n_tokens; ++t) {
          sum[t] += WaveRowTileDot(wrow, tp, k0, tk, lane, sx[t]);
        }
      }
    }
    if (k0 + kTileK < cols) __syncthreads();
  }

  if (row < total_m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float v = WaveSum(sum[t]);
      if (lane == 0u) {
        out_base[static_cast<std::size_t>(t) * out_stride + local_row] = v;
      }
    }
  }
}

__global__ void __launch_bounds__(256) GemvQkvBatchedQ8Kernel(
    const std::uint8_t* __restrict__ w_q, std::uint64_t rb_q,
    std::uint32_t qdim, const std::uint8_t* __restrict__ w_k,
    std::uint64_t rb_k, const std::uint8_t* __restrict__ w_v,
    std::uint64_t rb_v, std::uint32_t kvdim, std::uint32_t total_m,
    std::uint32_t total_blk, std::uint32_t tile_blk,
    const BlockQ8_1* __restrict__ x_q8, float* __restrict__ q_out,
    float* __restrict__ k_out, float* __restrict__ v_out,
    std::uint32_t n_tokens) {
  extern __shared__ __align__(8) unsigned char smem_qkv_q8[];
  auto* sx = reinterpret_cast<BlockQ8_1*>(smem_qkv_q8);

  const std::uint32_t warp_id = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t b_grp = lane >> 2;
  const std::uint32_t sub = lane & 3u;
  const std::uint32_t row = blockIdx.x * 8u + warp_id;

  const std::uint8_t* wrow = nullptr;
  float* out_base = nullptr;
  std::uint32_t out_stride = 0;
  std::uint32_t local_row = 0;
  if (row < qdim) {
    local_row = row;
    wrow = w_q + static_cast<std::size_t>(local_row) * rb_q;
    out_base = q_out;
    out_stride = qdim;
  } else if (row < qdim + kvdim) {
    local_row = row - qdim;
    wrow = w_k + static_cast<std::size_t>(local_row) * rb_k;
    out_base = k_out;
    out_stride = kvdim;
  } else if (row < total_m) {
    local_row = row - qdim - kvdim;
    wrow = w_v + static_cast<std::size_t>(local_row) * rb_v;
    out_base = v_out;
    out_stride = kvdim;
  }

  float sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (std::uint32_t b_off = 0; b_off < total_blk; b_off += tile_blk) {
    const std::uint32_t tk_blk =
        (total_blk - b_off) < tile_blk ? (total_blk - b_off) : tile_blk;
    const std::uint32_t words2 = tk_blk * 5u;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const auto* src2 = reinterpret_cast<const uint2*>(
          x_q8 + static_cast<std::size_t>(t) * total_blk + b_off);
      auto* dst2 = reinterpret_cast<uint2*>(sx + t * tile_blk);
      for (std::uint32_t i = threadIdx.x; i < words2; i += 256u) {
        dst2[i] = src2[i];
      }
    }
    __syncthreads();

    if (row < total_m) {
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
          for (std::uint32_t t = 0; t < n_tokens; ++t) {
            const BlockQ8_1& xblk = sx[t * tile_blk + b];
            int dot = Dot4U4S8(vi0, xblk.qs[sub], 0);
            dot = Dot4U4S8(vi1, xblk.qs[4u + sub], dot);
            sum[t] += d4 * (xblk.d * static_cast<float>(dot) - 2.0F * xblk.s);
          }
        }
      }
    }
    if (b_off + tile_blk < total_blk) __syncthreads();
  }

  if (row < total_m) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const float v = WaveSum(sum[t]);
      if (lane == 0u) {
        out_base[static_cast<std::size_t>(t) * out_stride + local_row] = v;
      }
    }
  }
}

void GemvQkv(const void* w_q, int type_q, std::uint64_t rb_q,
             std::uint32_t qdim, const void* w_k, int type_k,
             std::uint64_t rb_k, const void* w_v, int type_v,
             std::uint64_t rb_v, std::uint32_t kvdim, std::uint64_t cols,
             const float* x, float* q_out, float* k_out, float* v_out,
             std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == 0) return;
  const std::uint32_t total_m =
      qdim + kvdim + (w_v != nullptr ? kvdim : 0u);
  if ((cols % 32ull) == 0ull && n_tokens <= 8u &&
      getenv("GEMMA4_NO_WAVE32_GEMV") == nullptr) {
    const dim3 grid((total_m + 7u) / 8u);
    const dim3 block(256);
    const bool all_q4 =
        (type_q == 2 && type_k == 2 && (w_v == nullptr || type_v == 2));
    const bool q8_batch_ok = (getenv("GEMMA4_NO_Q8_BATCH") == nullptr);
    const bool q8_single = (getenv("GEMMA4_Q8_SINGLE") != nullptr);
    if (n_tokens == 1u && cols <= 12288ull &&
        !(all_q4 && q8_batch_ok && (q8_single || total_m >= 65536u))) {
      hipLaunchKernelGGL(
          GemvQkvSingleKernel, grid, block,
          static_cast<std::size_t>(cols) * sizeof(float), stream,
          static_cast<const std::uint8_t*>(w_q), type_q, rb_q, qdim,
          static_cast<const std::uint8_t*>(w_k), type_k, rb_k,
          static_cast<const std::uint8_t*>(w_v), type_v, rb_v, kvdim, total_m,
          static_cast<std::uint32_t>(cols), x, q_out, k_out, v_out);
      return;
    }
    if (all_q4 && q8_batch_ok) {
      const std::uint32_t total_blk = static_cast<std::uint32_t>(cols >> 5);
      const std::uint32_t tile_blk = total_blk < 128u ? total_blk : 128u;
      if (const BlockQ8_1* x_q8 = detail::QuantizeToQ8Scratch(
              x, static_cast<std::size_t>(n_tokens) * total_blk, stream)) {
        const std::size_t smem =
            static_cast<std::size_t>(n_tokens) * tile_blk * sizeof(BlockQ8_1);
        hipLaunchKernelGGL(
            GemvQkvBatchedQ8Kernel, grid, block, smem, stream,
            static_cast<const std::uint8_t*>(w_q), rb_q, qdim,
            static_cast<const std::uint8_t*>(w_k), rb_k,
            static_cast<const std::uint8_t*>(w_v), rb_v, kvdim, total_m,
            total_blk, tile_blk, x_q8, q_out, k_out, v_out, n_tokens);
        return;
      }
    }
    hipLaunchKernelGGL(
        GemvQkvBatchedKernel, grid, block, 0, stream,
        static_cast<const std::uint8_t*>(w_q), type_q, rb_q, qdim,
        static_cast<const std::uint8_t*>(w_k), type_k, rb_k,
        static_cast<const std::uint8_t*>(w_v), type_v, rb_v, kvdim, total_m,
        static_cast<std::uint32_t>(cols), x, q_out, k_out, v_out, n_tokens);
    return;
  }
  GemvBatched(w_q, type_q, rb_q, cols, x, q_out, n_tokens, qdim, 0.0F, stream);
  GemvBatched(w_k, type_k, rb_k, cols, x, k_out, n_tokens, kvdim, 0.0F, stream);
  if (w_v != nullptr) {
    GemvBatched(w_v, type_v, rb_v, cols, x, v_out, n_tokens, kvdim, 0.0F,
                stream);
  }
}

// ---- 27B block-GEMV shapes (from src/models/qwen/hip/kernels/gemv_quant.hip) ----

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q8KBlockGEMVKernel(const void* __restrict__ A, core::GgmlType type,
                            const float* __restrict__ x, float* __restrict__ y,
                            std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const float sumf = ::gufo::hip::QuantWarpBlockDot(type, A, m, x, K, lane_id, 0, 1);
  if (lane_id == 0) {
    y[m] = sumf;
  }
}

template<std::uint32_t WavesPerBlock,
         core::GgmlType StaticType = core::GgmlType::kF32>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q8KBlockGEMVKernel_2Rows(const void* __restrict__ A,
                                  core::GgmlType type,
                                  const float* __restrict__ x,
                                  float* __restrict__ y, std::size_t M,
                                  std::size_t K) {
  // F32 denotes runtime format dispatch here; the common packed formats
  // specialize this same arithmetic body to remove per-block type branches.
  if constexpr (StaticType != core::GgmlType::kF32) {
    type = StaticType;
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

  if (type == core::GgmlType::kQ8_0) {
    const std::size_t num_blocks = K / ::gufo::hip::kQ8_0BlockSize;
    const auto* row0_blocks =
        static_cast<const ::gufo::hip::Q8_0Block*>(A) + (m0 * num_blocks);
    const auto* row1_blocks =
        (m1 < M) ? (static_cast<const ::gufo::hip::Q8_0Block*>(A) + (m1 * num_blocks))
                 : nullptr;
    float sum0 = 0.0F;
    float sum1 = 0.0F;

    for (std::size_t b = lane_id; b < num_blocks; b += 32) {
      const float* xb = x + (b * ::gufo::hip::kQ8_0BlockSize);
      const ::gufo::hip::Q8_0Block& wblk0 = row0_blocks[b];
      const float d0 = __half2float(wblk0.d);
      float blk_dot0 = 0.0F;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        blk_dot0 += static_cast<float>(wblk0.qs[i]) * xb[i];
      }
      sum0 += d0 * blk_dot0;

      if (row1_blocks != nullptr) {
        const ::gufo::hip::Q8_0Block& wblk1 = row1_blocks[b];
        const float d1 = __half2float(wblk1.d);
        float blk_dot1 = 0.0F;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          blk_dot1 += static_cast<float>(wblk1.qs[i]) * xb[i];
        }
        sum1 += d1 * blk_dot1;
      }
    }

    for (int off = 16; off > 0; off >>= 1) {
      sum0 += __shfl_xor(sum0, off);
      sum1 += __shfl_xor(sum1, off);
    }

    if (lane_id == 0) {
      y[m0] = sum0;
      if (m1 < M) {
        y[m1] = sum1;
      }
    }
    return;
  }

  if (type == core::GgmlType::kQ8_K) {
    const std::size_t num_blocks = K / ::gufo::hip::kQ8KBlockSize;
    const auto* row0_blocks =
        static_cast<const ::gufo::hip::Q8KBlock*>(A) + (m0 * num_blocks);
    const auto* row1_blocks =
        (m1 < M) ? (static_cast<const ::gufo::hip::Q8KBlock*>(A) + (m1 * num_blocks))
                 : nullptr;
    float sum0 = 0.0F;
    float sum1 = 0.0F;

    for (std::size_t b = 0; b < num_blocks; ++b) {
      const ::gufo::hip::Q8KBlock& wblk0 = row0_blocks[b];
      const float d_w0 = wblk0.d;
      const ::gufo::hip::Q8KBlock* wblk1 =
          (row1_blocks != nullptr) ? &row1_blocks[b] : nullptr;
      const float d_w1 = (wblk1 != nullptr) ? wblk1->d : 0.0F;
      const float* xb = x + (b * ::gufo::hip::kQ8KBlockSize);
      float dot0 = 0.0F;
      float dot1 = 0.0F;
#pragma unroll
      for (std::size_t k = 0; k < 8; ++k) {
        const std::size_t idx = lane_id + (32 * k);
        const float xv = xb[idx];
        dot0 += static_cast<float>(wblk0.qs[idx]) * xv;
        if (wblk1 != nullptr) {
          dot1 += static_cast<float>(wblk1->qs[idx]) * xv;
        }
      }
      sum0 += d_w0 * dot0;
      sum1 += d_w1 * dot1;
    }

    for (int off = 16; off > 0; off >>= 1) {
      sum0 += __shfl_xor(sum0, off);
      sum1 += __shfl_xor(sum1, off);
    }

    if (lane_id == 0) {
      y[m0] = sum0;
      if (m1 < M) {
        y[m1] = sum1;
      }
    }
    return;
  }

  const float s0 = ::gufo::hip::QuantWarpBlockDot(type, A, m0, x, K, lane_id, 0, 1);
  const float s1 =
      (m1 < M) ? ::gufo::hip::QuantWarpBlockDot(type, A, m1, x, K, lane_id, 0, 1) : 0.0F;
  if (lane_id == 0) {
    y[m0] = s0;
    if (m1 < M) {
      y[m1] = s1;
    }
  }
}

// opt-c1xx-q8k-gemv: launches the Q8_K / Q8_0 block GEMV.
void LaunchQ8KBlockGEMV(const void* A, core::GgmlType type, const float* x,
                        float* y, std::size_t M, std::size_t K,
                        hipStream_t stream) {
  constexpr std::uint32_t kWavesPerBlock = 4;
  constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;

  if (M >= 2 && (M % 2 == 0)) {
    const std::size_t num_pairs = M / 2;
    const std::size_t grid = (num_pairs + kWavesPerBlock - 1U) / kWavesPerBlock;
    const auto launch = [&]<core::GgmlType StaticType>() {
      Q8KBlockGEMVKernel_2Rows<kWavesPerBlock, StaticType>
          <<<grid, kBlockThreads, 0, stream>>>(A, type, x, y, M, K);
    };
    switch (type) {
      case core::GgmlType::kQ4_K:
        launch.template operator()<core::GgmlType::kQ4_K>();
        break;
      case core::GgmlType::kQ5_K:
        launch.template operator()<core::GgmlType::kQ5_K>();
        break;
      case core::GgmlType::kIQ4_XS:
        launch.template operator()<core::GgmlType::kIQ4_XS>();
        break;
      default:
        launch.template operator()<core::GgmlType::kF32>();
        break;
    }
    return;
  }

  const std::size_t grid = (M + kWavesPerBlock - 1U) / kWavesPerBlock;
  Q8KBlockGEMVKernel<kWavesPerBlock>
      <<<grid, kBlockThreads, 0, stream>>>(A, type, x, y, M, K);
}

// ---- flash-next decode SmallGemm (from qwen38_flash_next kernels) ----

namespace fn_compat {
namespace {

__device__ __forceinline__ float Materialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}

__device__ __forceinline__ int XorLane(int v, int offset) {
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

__device__ __forceinline__ unsigned XorLane(unsigned v, int offset) {
  return static_cast<unsigned>(XorLane(static_cast<int>(v), offset));
}

__device__ __forceinline__ float XorLane(float v, int offset) {
  return __builtin_bit_cast(float, XorLane(__builtin_bit_cast(int, v), offset));
}

__device__ __forceinline__ float Bf16ToF32(std::uint16_t h) {
  return __uint_as_float(static_cast<std::uint32_t>(h) << 16);
}

__device__ __forceinline__ float RowElement(const void* row, int type,
                                            std::uint32_t i) {
  switch (type) {
    case 0:
      return static_cast<const float*>(row)[i];
    case 30:
      return Bf16ToF32(static_cast<const std::uint16_t*>(row)[i]);
    case 1:
      return __half2float(static_cast<const __half*>(row)[i]);
    case 8: {
      const auto* blk = static_cast<const std::uint8_t*>(row) + (i / 32) * 34;
      const __half d = *reinterpret_cast<const __half*>(blk);
      const auto q = static_cast<const std::int8_t*>(
          static_cast<const void*>(blk + 2))[i % 32];
      return __half2float(d) * static_cast<float>(q);
    }
  }
  return 0.0f;
}

__device__ __forceinline__ std::size_t RowBytes(int type,
                                                std::uint32_t k) {
  switch (type) {
    case 0:
      return static_cast<std::size_t>(k) * 4;
    case 30:
    case 1:
      return static_cast<std::size_t>(k) * 2;
    case 8:
      return static_cast<std::size_t>(k / 32) * 34;
  }
  return 0;
}

}  // namespace

constexpr unsigned kSmallGemmRows = 4;

template<int type, unsigned tokens, bool grouped = false,
         bool kWhole = false>
__global__ void SmallGemmKernel(const void* w, const float* x, float* out,
                                std::uint32_t m, std::uint32_t k) {
  if constexpr (grouped) {
    x += std::size_t{blockIdx.y} * tokens * k;
    out += std::size_t{blockIdx.y} * tokens * m;
  }
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= m) {
    return;
  }
  const auto* wrow =
      static_cast<const std::uint8_t*>(w) + RowBytes(type, k) * row;
  // A lane's steps are loaded kSteps at a time so their reads overlap (one
  // step at a time leaves a decode-sized projection latency-bound); the
  // steps still accumulate one after another, in order.
  constexpr unsigned kSteps = kWhole || tokens <= 2 ? 4 : 2;
  const std::uint32_t stride = warpSize * 4;
  float acc[tokens] = {};
  for (std::uint32_t i_base = lane * 4; i_base < k; i_base += stride * kSteps) {
    float wv[kSteps][4];
    float xv[kSteps][tokens][4];
#pragma unroll
    for (unsigned st = 0; st < kSteps; ++st) {
      const std::uint32_t i0 = i_base + st * stride;
      if constexpr (kWhole) {
        // Rows of whole passes (k a multiple of stride * kSteps): every step
        // is in range, so the four elements arrive as one load: 16 bytes for
        // F32, 8 bytes for F16/BF16 (widened exactly, as RowElement does).
        if constexpr (type == 0) {
          const float4 w4 = reinterpret_cast<const float4*>(
              static_cast<const void*>(wrow))[i0 / 4];
          wv[st][0] = w4.x;
          wv[st][1] = w4.y;
          wv[st][2] = w4.z;
          wv[st][3] = w4.w;
        } else {
          static_assert(type == 1 || type == 30);
          const ushort4 h4 = reinterpret_cast<const ushort4*>(
              static_cast<const void*>(wrow))[i0 / 4];
          if constexpr (type == 30) {
            wv[st][0] = Bf16ToF32(h4.x);
            wv[st][1] = Bf16ToF32(h4.y);
            wv[st][2] = Bf16ToF32(h4.z);
            wv[st][3] = Bf16ToF32(h4.w);
          } else {
            wv[st][0] = __half2float(__ushort_as_half(h4.x));
            wv[st][1] = __half2float(__ushort_as_half(h4.y));
            wv[st][2] = __half2float(__ushort_as_half(h4.z));
            wv[st][3] = __half2float(__ushort_as_half(h4.w));
          }
        }
#pragma unroll
        for (unsigned j = 0; j < tokens; ++j) {
          const float4 x4 = reinterpret_cast<const float4*>(
              x + static_cast<std::size_t>(j) * k)[i0 / 4];
          xv[st][j][0] = x4.x;
          xv[st][j][1] = x4.y;
          xv[st][j][2] = x4.z;
          xv[st][j][3] = x4.w;
        }
        continue;
      }
#pragma unroll
      for (unsigned r = 0; r < 4; ++r) {
        wv[st][r] = i0 + r < k ? RowElement(wrow, type, i0 + r) : 0.0f;
      }
#pragma unroll
      for (unsigned j = 0; j < tokens; ++j) {
        const float* xr = x + static_cast<std::size_t>(j) * k + i0;
#pragma unroll
        for (unsigned r = 0; r < 4; ++r) {
          xv[st][j][r] = i0 + r < k ? xr[r] : 0.0f;
        }
      }
    }
#pragma unroll
    for (unsigned st = 0; st < kSteps; ++st) {
      const std::uint32_t i0 = i_base + st * stride;
      if (!kWhole && i0 >= k) {
        break;
      }
#pragma unroll
      for (unsigned j = 0; j < tokens; ++j) {
        if constexpr (kWhole) {
          // The bounds-checked path compiles acc + dot to one FMA chain
          // from acc in element order; spelled out so it cannot reassociate.
          float a = acc[j];
#pragma unroll
          for (unsigned r = 0; r < 4; ++r) {
            a = __builtin_fmaf(wv[st][r], xv[st][j][r], a);
          }
          acc[j] = Materialize(a);
          continue;
        }
        float dot = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < 4; ++r) {
          dot += wv[st][r] * xv[st][j][r];
        }
        acc[j] = Materialize(acc[j] + dot);
      }
    }
  }
#pragma unroll
  for (unsigned j = 0; j < tokens; ++j) {
    const float total = WaveSum(acc[j]);
    if (lane == 0) {
      out[static_cast<std::size_t>(j) * m + row] = total;
    }
  }
}

template<int type, unsigned tokens>
void LaunchSmallGemm(const void* w, const float* x, float* out, std::uint32_t m,
                     std::uint32_t k, hipStream_t stream) {
  if constexpr (type != 8) {
    // Rows of whole four-step passes (K 2560: router and GDN alpha/beta in
    // F16, indexer in BF16) take the unchecked vector loads, four steps at a
    // time at any width: the same products in the same order.
    const bool aligned = (reinterpret_cast<std::uintptr_t>(w) |
                          reinterpret_cast<std::uintptr_t>(x)) %
                             16 ==
                         0;
    if (aligned && k % (32 * 4 * 4) == 0) {
      hipLaunchKernelGGL((SmallGemmKernel<type, tokens, false, true>),
                         dim3((m + kSmallGemmRows - 1) / kSmallGemmRows),
                         dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
      return;
    }
  }
  hipLaunchKernelGGL((SmallGemmKernel<type, tokens>),
                     dim3((m + kSmallGemmRows - 1) / kSmallGemmRows),
                     dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
}

template<int type>
void SmallGemmForType(const void* w, const float* x, float* out,
                      std::uint32_t tokens, std::uint32_t m, std::uint32_t k,
                      hipStream_t stream) {
  if (tokens > 8) {
    const auto groups = tokens / 8;
    hipLaunchKernelGGL((SmallGemmKernel<type, 8, true>),
                       dim3((m + kSmallGemmRows - 1) / kSmallGemmRows, groups),
                       dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
    const auto consumed = groups * 8;
    x += std::size_t{consumed} * k;
    out += std::size_t{consumed} * m;
    tokens -= consumed;
    if (tokens == 0)
      return;
  }
  switch (tokens) {
    case 1:
      return LaunchSmallGemm<type, 1>(w, x, out, m, k, stream);
    case 2:
      return LaunchSmallGemm<type, 2>(w, x, out, m, k, stream);
    case 3:
      return LaunchSmallGemm<type, 3>(w, x, out, m, k, stream);
    case 4:
      return LaunchSmallGemm<type, 4>(w, x, out, m, k, stream);
    case 5:
      return LaunchSmallGemm<type, 5>(w, x, out, m, k, stream);
    case 6:
      return LaunchSmallGemm<type, 6>(w, x, out, m, k, stream);
    case 7:
      return LaunchSmallGemm<type, 7>(w, x, out, m, k, stream);
    case 8:
      return LaunchSmallGemm<type, 8>(w, x, out, m, k, stream);
    default:
      throw std::logic_error("small projection requires 1-8 token rows");
  }
}

void SmallGemm(const void* w, int type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream) {
  switch (type) {
    case 0:
      return SmallGemmForType<0>(w, x, out, n_tokens, m, k,
                                                stream);
    case 30:
      return SmallGemmForType<30>(w, x, out, n_tokens, m, k,
                                                 stream);
    case 1:
      return SmallGemmForType<1>(w, x, out, n_tokens, m, k,
                                                stream);
    default:
      throw std::logic_error("unsupported small projection format");
  }
}
}  // namespace fn_compat

}  // namespace gufo::core::compute::gemm

#endif  // defined(ENGINE_ENABLE_HIP)
