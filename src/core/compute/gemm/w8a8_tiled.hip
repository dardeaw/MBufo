// Phase C-1 (GGUF通用化.md v2 §4): shared-core home for the quantized-GEMM
// launchers gemma4 previously reached through
// models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp. Every body below is
// a verbatim copy (namespace relocated to gufo::core::compute::gemm only);
// flash-next keeps its own originals untouched until Phase C-2 rewires it.
// Numeric behavior is identical by construction (same kernels, same launch).
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/compute/gemm/gemm.hpp"
#include <type_traits>

namespace gufo::core::compute::gemm {
namespace {

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}


/// Keeps fast-math from folding a stored value's arithmetic into what
/// follows: the quantization must see exactly the float that was stored.
__device__ __forceinline__ float Materialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}


/// four selections overlap far less than 32 (measured at 16k depth: 846
/// versus 2,478 selected blocks against 512 per query).
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

__device__ __forceinline__ int32x8_t WmmaI8(int32x4_t a, int32x4_t b,
                                            int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}


/// Lane i receives lane (i ^ offset)'s value, as __shfl_xor(v, offset) does
/// across a wave32. On gfx1151 it is a DPP row_xmask (offsets below 16) or
/// v_permlanex16 (16): a register move instead of an LDS crossbar round trip
/// (ds_bpermute). Only the transport differs, so reductions stay bitwise;
/// every lane must be active, as the shuffle's callers already require.
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


constexpr std::size_t kQ8ActTileTokens = 16;
constexpr std::size_t kQ8ActTileBytes = 576;
constexpr std::size_t kQ8ActScaleOffset = 512;
__device__ __forceinline__ std::int8_t* Q8ActTile(void* base,
                                                  std::size_t num_blocks,
                                                  std::size_t tt,
                                                  std::size_t kb) {
  return static_cast<std::int8_t*>(base) +
         (((tt * num_blocks) + kb) * kQ8ActTileBytes);
}


//
//   tile(tt, kb) = [b0: 16 tokens x 16 bytes]   offset   0
//                  [b1: 16 tokens x 16 bytes]   offset 256
//                  [16 fp32 token scales    ]   offset 512
//
// with tt = token / 16 and kb the 32-element K block, 576 bytes per tile.
struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");
/// W8A8TailKernel limits: tokens past the last full tile and K width.
constexpr std::size_t kW8A8TailMaxTokens = 16;
constexpr std::size_t kW8A8TailMaxK = 10240;


/// Four elements per lane, eight lanes per 32-wide K block, four blocks per
/// wave: per-block absmax scale, four codes stored as one word straight
/// into the fragment order the GEMM stages from.
__global__ void QuantizeQ8TiledVec4Kernel(const float* __restrict__ x,
                                          void* __restrict__ y,
                                          std::size_t batch, std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) >> 3u;
  const std::uint32_t lane8 = threadIdx.x & 7u;
  if (b_idx >= batch * num_blocks) {
    return;
  }
  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const float4 v = *reinterpret_cast<const float4*>(x + (tok * k) + (blk * 32) +
                                                    (lane8 * 4));
  float max_abs =
      fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
  for (int off = 4; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, XorLane(max_abs, off));
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q0 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.x * id))));
  const auto q1 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.y * id))));
  const auto q2 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.z * id))));
  const auto q3 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.w * id))));
  const std::size_t tt = tok / kQ8ActTileTokens;
  const std::size_t tl = tok % kQ8ActTileTokens;
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, blk);
  const std::uint32_t pos = lane8 * 4;  // 0, 4, ..., 28
  *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                    (pos & 15u)) =
      q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  if (lane8 == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

/// One wave per (token, 32-wide K block): per-block absmax scale, codes
/// written straight into the fragment order the GEMM stages from.
__global__ void QuantizeQ8TiledKernel(const float* __restrict__ x,
                                      void* __restrict__ y, std::size_t batch,
                                      std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;
  if (b_idx >= batch * num_blocks) {
    return;
  }
  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const float val = x[(tok * k) + (blk * 32) + lane_id];
  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, XorLane(max_abs, off));
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q = static_cast<std::int8_t>(roundf(val * id));
  const std::size_t tt = tok / kQ8ActTileTokens;
  const std::size_t tl = tok % kQ8ActTileTokens;
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, blk);
  const std::size_t half = lane_id >> 4u;
  const std::size_t pos = lane_id & 15u;
  tile[(half * 256) + (tl * 16) + pos] = q;
  if (lane_id == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}


// Preserve the separate scale/SiLU kernel's F32 intermediates before narrowing.
// Without these compiler boundaries, fused F16 output can round differently.
__device__ __forceinline__ float HcScaledSilu(float value) {
  float scaled = value * 0.25F;
  asm volatile("" : "+v"(scaled));
  float result = SiluF(scaled);
  asm volatile("" : "+v"(result));
  return result;
}


/// Two-dimensionally blocked W8A8 WMMA GEMM: block = BM rows x BN tokens, BK
/// 32-element K blocks per LDS stage, waves = WM row groups x WN token groups.
/// Out-of-range rows and K blocks are clamped and their scale zeroed, so
/// they contribute exactly zero without divergence. y is [batch][m].
template<int BM, int BN, int BK, int WM, int WN, bool kHcDown = false>
__launch_bounds__(256) __global__ void W8A8BlockedWmmaGEMMKernel(
    const void* __restrict__ w, const void* __restrict__ x_blocks,
    std::conditional_t<kHcDown, __half, float>* __restrict__ y,
    std::size_t batch, std::size_t m, std::size_t k) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(BN / 16 <= 8,
                "the activation stage assigns one wave per token subtile");
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;

  // One LDS plane: codes and scales of both operands; the epilogue reuses
  // the first 16 KB to transpose 16 x 32 result tiles per wave.
  constexpr int kABytes = BK * kRowTiles * 32 * 16;
  constexpr int kDwBytes = BK * kRowTiles * 16 * 4;
  constexpr int kBBytes = BK * kTokTiles * 32 * 16;
  constexpr int kDxBytes = BK * kTokTiles * 16 * 4;
  constexpr int kLdsBytes = kABytes + kDwBytes + kBBytes + kDxBytes;
  static_assert(kLdsBytes >= 8 * 512 * 4, "epilogue transposes 16 KB");
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_a = reinterpret_cast<int32x4_t(*)[kRowTiles][32]>(lds);
  auto* s_dw = reinterpret_cast<float (*)[kRowTiles][16]>(lds + kABytes);
  auto* s_b =
      reinterpret_cast<int32x4_t(*)[kTokTiles][32]>(lds + kABytes + kDwBytes);
  auto* s_dx = reinterpret_cast<float (*)[kTokTiles][16]>(lds + kABytes +
                                                          kDwBytes + kBBytes);

  const auto* w_blocks = static_cast<const Q8_0Block*>(w);
  const std::size_t num_blocks = k / 32;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / kQ8ActTileTokens;

  float acc[kWaveRowTiles][kWaveTokTiles][8];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  // Staging registers for the next K stage, fetched one stage ahead so the
  // scattered 34-byte block reads overlap the WMMA work.
  constexpr int kPrefetch = (BM * BK) / 256;
  int32x4_t r_q0[kPrefetch];
  int32x4_t r_q1[kPrefetch];
  float r_dw[kPrefetch];
  int32x4_t r_b[BK];
  float r_dx[BK];
  const int b_tile = wave_id;  // one wave per token subtile

  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const Q8_0Block* w_row[kPrefetch];
  float row_live[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int r = static_cast<int>(r_block) + (((p * 256) + tid) / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = w_blocks + (static_cast<std::size_t>(r_clamped) * num_blocks);
    row_live[p] = (r < m_i) ? 1.0F : 0.0F;
  }
  const auto* b_base = static_cast<const std::int8_t*>(x_blocks) +
                       ((tt_block + static_cast<std::size_t>(b_tile)) *
                        num_blocks * kQ8ActTileBytes);
  const bool b_live = b_tile < kTokTiles;

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const Q8_0Block& blk = w_row[p][kb_clamped];
      __builtin_memcpy(&r_q0[p], blk.qs + 0, 16);
      __builtin_memcpy(&r_q1[p], blk.qs + 16, 16);
      r_dw[p] = __half2float(blk.d) * ((kb < num_kb) ? row_live[p] : 0.0F);
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile =
            b_base + (static_cast<std::size_t>(kb_clamped) * kQ8ActTileBytes);
        r_b[i] = reinterpret_cast<const int32x4_t*>(tile)[lane_id];
        r_dx[i] = ((kb < num_kb) ? 1.0F : 0.0F) *
                  reinterpret_cast<const float*>(
                      tile + kQ8ActScaleOffset)[lane_id & 15];
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = r_dw[p];
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane_id] = r_b[i];
        if (lane_id < 16) {
          s_dx[i][b_tile][lane_id] = r_dx[i];
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }

#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[kWaveRowTiles];
      int32x4_t a1[kWaveRowTiles];
      float dw[kWaveRowTiles][8];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][half_id * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(half_id * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
      }
      // Token-tile operands are read one tile at a time to keep the live
      // register set small enough for the BK=2 stage.
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = WmmaI8(a0[i], b0, c);
          c = WmmaI8(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            // The order fast-math compiled (dw * dx) * c to, written out
            // so W8A8TailKernel can repeat it: (dx * c), then one FMA.
            acc[i][j][l] = __fmaf_rn(__fmul_rn(dx, static_cast<float>(c[l])),
                                     dw[i][l], acc[i][j][l]);
          }
        }
      }
#if __clang_major__ >= 23
      // LLVM 23 hoists the next K block's operands above this one's WMMAs,
      // which spills registers on gfx1151. ROCm 7.2.3 (LLVM 22) does not, and
      // the fence costs it about 3%.
      __builtin_amdgcn_sched_barrier(0);
#endif
    }
    __syncthreads();
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  __syncthreads();
  constexpr unsigned kOutputStride = kHcDown ? 36 : 32;
  static_assert(8 * 16 * kOutputStride * sizeof(float) <= sizeof(lds));
  float* tile_scratch =
      reinterpret_cast<float*>(lds) + wave_id * 16 * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      // scratch[token][row] over 16 tokens x 32 rows.
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * kOutputStride) + (2 * l) + half_id] =
            acc[i][j][l];
        tile_scratch[(sub_lane * kOutputStride) + 16 + (2 * l) + half_id] =
            acc[i + 1][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          r_block +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          t_block +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
      // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
      const int tok_l = lane_id >> 1;
      const int row_l = (lane_id & 1) * 16;
      const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
      const auto* src = reinterpret_cast<const float4*>(
          tile_scratch + (tok_l * kOutputStride) + row_l);
      if (tok < batch && r0 + 32 <= m) {
        if constexpr (kHcDown) {
          auto* dst = y + tok * m + r0 + static_cast<std::size_t>(row_l);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float4 v = src[q];
            *reinterpret_cast<__half2*>(dst + 4 * q) =
                __floats2half2_rn(HcScaledSilu(v.x), HcScaledSilu(v.y));
            *reinterpret_cast<__half2*>(dst + 4 * q + 2) =
                __floats2half2_rn(HcScaledSilu(v.z), HcScaledSilu(v.w));
          }
        } else {
          auto* dst = reinterpret_cast<float4*>(
              y + (tok * m) + r0 + static_cast<std::size_t>(row_l));
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            dst[q] = src[q];
          }
        }
      } else if (tok < batch) {
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
          if (r < m) {
            const float value =
                tile_scratch[(tok_l * kOutputStride) + row_l + q];
            if constexpr (kHcDown)
              y[(tok * m) + r] = __float2half_rn(HcScaledSilu(value));
            else
              y[(tok * m) + r] = value;
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}


/// Tokens [first, batch) of a W8A8 GEMM with W8A8BlockedWmmaGEMMKernel's
/// arithmetic, one wave per output: lanes form the exact integer dot and
/// dx * c of every 32-wide K block, then the K blocks' FMAs run in order
/// through lane reads. A few tokens past the last full 128-token tile
/// otherwise cost the tiled kernel a second wave of blocks.
template<bool kHcDown>
__launch_bounds__(256) __global__ void W8A8TailKernel(
    const void* __restrict__ w, const void* __restrict__ x_blocks,
    std::conditional_t<kHcDown, __half, float>* __restrict__ y,
    std::size_t first, std::size_t m, std::size_t k) {
  constexpr int kMaxSteps = kW8A8TailMaxK / 32 / 32;
  const std::size_t row =
      (static_cast<std::size_t>(blockIdx.x) * 8) + (threadIdx.x >> 5u);
  const std::size_t tok = first + blockIdx.y;
  const int lane = static_cast<int>(threadIdx.x & 31u);
  if (row >= m) {
    return;
  }
  const int num_kb = static_cast<int>(k / 32);
  const auto* w_row = static_cast<const Q8_0Block*>(w) + (row * num_kb);
  const std::size_t tl = tok % kQ8ActTileTokens;
  float t[kMaxSteps];
  float dw[kMaxSteps];
#pragma unroll
  for (int s = 0; s < kMaxSteps; ++s) {
    const int kb = (s * 32) + lane;
    t[s] = 0.0F;
    dw[s] = 0.0F;
    if (kb < num_kb) {
      const Q8_0Block& blk = w_row[kb];
      int qs[8];
      __builtin_memcpy(qs, blk.qs, sizeof(qs));
      const auto* tile =
          static_cast<const std::int8_t*>(x_blocks) +
          ((((tok / kQ8ActTileTokens) * num_kb) + kb) * kQ8ActTileBytes);
      const auto lo = *reinterpret_cast<const int4*>(tile + (tl * 16));
      const auto hi = *reinterpret_cast<const int4*>(tile + 256 + (tl * 16));
      const int xs[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
      int c = 0;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        c = __builtin_amdgcn_sudot4(true, qs[q], true, xs[q], c, false);
      }
      const float dx = *reinterpret_cast<const float*>(
          tile + kQ8ActScaleOffset + (tl * sizeof(float)));
      t[s] = __fmul_rn(dx, static_cast<float>(c));
      dw[s] = __half2float(blk.d);
    }
  }
  float acc = Materialize(0.0F);
#pragma unroll
  for (int s = 0; s < kMaxSteps; ++s) {
#pragma unroll
    for (int l = 0; l < 32; ++l) {
      if ((s * 32) + l < num_kb) {
        const float ts = __builtin_bit_cast(
            float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, t[s]), l));
        const float ws = __builtin_bit_cast(
            float,
            __builtin_amdgcn_readlane(__builtin_bit_cast(int, dw[s]), l));
        acc = __fmaf_rn(ts, ws, acc);
      }
    }
  }
  if (lane == 0) {
    if constexpr (kHcDown) {
      y[(tok * m) + row] = __float2half_rn(HcScaledSilu(acc));
    } else {
      y[(tok * m) + row] = acc;
    }
  }
}


/// Splits a 128-token-tiled W8A8 launch: the tail past the last full tile
/// goes to W8A8TailKernel when it is short. Returns the tiled token count.
std::size_t W8A8TailSplit(std::size_t batch, std::size_t k) {
  const std::size_t tail = batch % 128;
  return batch > 128 && tail != 0 && tail <= kW8A8TailMaxTokens &&
                 k <= kW8A8TailMaxK
             ? batch - tail
             : batch;
}


template<bool kHcDown>
void LaunchW8A8Tail(const void* w, const void* x_tiled,
                    std::conditional_t<kHcDown, __half, float>* out,
                    std::size_t first, std::size_t batch, std::size_t m,
                    std::size_t k, hipStream_t stream) {
  if (first < batch) {
    hipLaunchKernelGGL(W8A8TailKernel<kHcDown>,
                       dim3(static_cast<unsigned>((m + 7) / 8),
                            static_cast<unsigned>(batch - first)),
                       dim3(kThreads), 0, stream, w, x_tiled, out, first, m, k);
  }
}

}  // namespace

std::size_t Q8TiledBytes(std::size_t batch, std::size_t k) {
  // The GEMM stages whole 128-token macro tiles, so the buffer is sized to
  // the batch rounded up to one (the padding tiles are never read as
  // results, only staged).
  constexpr std::size_t kMacroTokens = 128;
  const std::size_t padded =
      (batch + kMacroTokens - 1) / kMacroTokens * kMacroTokens;
  return (padded / kQ8ActTileTokens) * (k / 32) * kQ8ActTileBytes;
}

void QuantizeQ8Tiled(const float* x, void* out, std::size_t batch,
                     std::size_t k, hipStream_t stream) {
  const std::size_t blocks = batch * (k / 32);
  if (batch * k >= 4096) {
    const std::size_t per_block = kThreads / 8;
    hipLaunchKernelGGL(QuantizeQ8TiledVec4Kernel,
                       dim3((blocks + per_block - 1) / per_block),
                       dim3(kThreads), 0, stream, x, out, batch, k);
    return;
  }
  const std::size_t waves = kThreads / 32;
  hipLaunchKernelGGL(QuantizeQ8TiledKernel, dim3((blocks + waves - 1) / waves),
                     dim3(kThreads), 0, stream, x, out, batch, k);
}


// Legacy W8A8 dispatcher: the Wave64 shape runs the shared-core
// Wave64 binary (identical instantiation); every other shape keeps the
// original blocked/tail launch geometry.
bool W8A8Gemm(const void* w, const void* x_tiled, float* out, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  // Qwen's wave64 matrix kernel with four row groups improves the model's
  // large output projections while preserving every K32 accumulator update.
  if (batch >= 1024 && m == 2560 && k == 6144) {
    W8A8GemmWave64Core(w, x_tiled, out, batch, m, k, stream);
    return true;
  }
  // A 128-token macro tile is the throughput configuration; short chunks
  // would leave most of it idle and take the 64-token variant. A narrow
  // projection (the 320-row mixer down over K = 10240) gets 64-row tiles so
  // it still fills the device.
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const std::size_t tiled = W8A8TailSplit(batch, k);
    const dim3 grid(static_cast<unsigned int>((tiled + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kNarrowBM, kBN, 4, 2, 4>),
                       grid, dim3(kThreads), 0, stream, w, x_tiled, out, tiled,
                       m, k);
    LaunchW8A8Tail<false>(w, x_tiled, out, tiled, batch, m, k, stream);
  } else if (batch >= 96) {
    constexpr int kBN = 128;
    const std::size_t tiled = W8A8TailSplit(batch, k);
    const dim3 grid(static_cast<unsigned int>((tiled + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x_tiled, out, tiled, m, k);
    LaunchW8A8Tail<false>(w, x_tiled, out, tiled, batch, m, k, stream);
  } else {
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kBM, kBN, 4, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x_tiled, out, batch, m, k);
  }
  return true;
}

}  // namespace gufo::core::compute::gemm

#endif  // ENGINE_ENABLE_HIP
