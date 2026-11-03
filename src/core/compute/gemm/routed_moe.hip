// Phase C-1 (GGUF通用化.md v2 §4): shared-core home for the quantized-GEMM
// launchers gemma4 previously reached through
// models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp. Every body below is
// a verbatim copy (namespace relocated to gufo::core::compute::gemm only);
// flash-next keeps its own originals untouched until Phase C-2 rewires it.
// Numeric behavior is identical by construction (same kernels, same launch).
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/compute/gemm/gemm.hpp"

namespace gufo::core::compute::gemm {
namespace {

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}


using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
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


__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}


/// Block-wide sum over kThreads threads; every thread receives the total.
__device__ float BlockSum(float v, float* shared) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += XorLane(v, offset);
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total += shared[w];
  }
  return total;
}

__device__ float BlockMax(float v, float* shared) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, XorLane(v, offset));
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = -INFINITY;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total = fmaxf(total, shared[w]);
  }
  return total;
}


template<unsigned MaxExperts>
__global__ void RouterTopKKernel(const float* logits, std::uint32_t stride,
                                 std::int32_t* ids, float* weights,
                                 std::uint32_t n_experts, std::uint32_t k) {
  __shared__ float probs[1024];
  __shared__ float shared[32];
  __shared__ std::uint32_t chosen[32];
  __shared__ float chosen_p[32];
  const std::uint32_t t = blockIdx.x;
  const float* src = logits + static_cast<std::size_t>(t) * stride;
  float local_max = -INFINITY;
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    local_max = fmaxf(local_max, src[e]);
  }
  const float max_logit = BlockMax(local_max, shared);
  float local_sum = 0.0f;
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    probs[e] = __expf(src[e] - max_logit);
    local_sum += probs[e];
  }
  const float denom = BlockSum(local_sum, shared);
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    probs[e] /= denom;
  }
  __syncthreads();

  if (threadIdx.x >= 32)
    return;
  constexpr unsigned Items = MaxExperts / 32;
  float values[Items];
#pragma unroll
  for (unsigned j = 0; j < Items; ++j) {
    unsigned e = threadIdx.x + j * 32;
    values[j] = e < n_experts ? probs[e] : -1.0f;
  }
  for (unsigned slot = 0; slot < k; ++slot) {
    float best = -1.0f;
    unsigned index = 0xffffffffu;
#pragma unroll
    for (unsigned j = 0; j < Items; ++j) {
      unsigned e = threadIdx.x + j * 32;
      if (values[j] > best) {
        best = values[j];
        index = e;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other = XorLane(best, offset);
      unsigned oi = XorLane(index, offset);
      if (other > best || (other == best && oi < index)) {
        best = other;
        index = oi;
      }
    }
    if (threadIdx.x == 0) {
      chosen[slot] = index;
      chosen_p[slot] = best;
    }
#pragma unroll
    for (unsigned j = 0; j < Items; ++j) {
      if (threadIdx.x + j * 32 == index)
        values[j] = -1.0f;
    }
  }
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      sum += chosen_p[slot];
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      ids[t * k + slot] = static_cast<std::int32_t>(chosen[slot]);
      weights[t * k + slot] = chosen_p[slot] / sum;
    }
  }
}


// Routed expert GEMMs: assignment rows are compacted by expert with every
// bucket padded to a 16-row tile (`pad_bounds`), so a token tile never
// straddles experts; `rows_out` maps a compact row to its (token, slot)
// output row or -1 for padding.
constexpr std::size_t kRoutedTileTokens = 16;


struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5_1Block {
  __half d;
  __half m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q5_1Block) == 24, "block_q5_1 must be 24 bytes");

struct Q4_1Block {
  __half d;
  __half m;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q4_1Block) == 20, "block_q4_1 must be 20 bytes");


struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");


constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

/// block_q5_K: the Q4_K header, 32 high-bit bytes (bit s of byte j is the
/// fifth bit of element j of K block s), then the Q4_K nibble layout.
constexpr std::size_t kQ5KBlockBytes = 176;


template<WeightType kType>
__device__ __forceinline__ std::size_t RoutedF16RowBytes(std::size_t k) {
  return kType == WeightType::kQ4_K   ? (k / 256) * sizeof(Q4KBlock)
         : kType == WeightType::kQ5_K ? (k / 256) * kQ5KBlockBytes
         : kType == WeightType::kQ5_1 ? (k / 32) * sizeof(Q5_1Block)
         : kType == WeightType::kQ4_0_ROCMFP4_FAST
             ? (k / 32) * 17
         : kType == WeightType::kQ4_0 ? (k / 32) * 18
                                       : (k / 32) * sizeof(Q8_0Block);
}


/// Bit `s` of each of the four bytes of `w`, packed into bits 0-3.
__device__ __forceinline__ std::uint32_t GatherBit(std::uint32_t w, int s) {
  // 0x01020408 moves byte b's bit to bit 24 + b.
  return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}

/// Four packed 5-bit codes: `nib` holds the 4-bit parts one per byte, `bits`
/// bits j..j+3 of the Q5_1 high-bit word, spread to bit 4 of each byte.
__device__ __forceinline__ std::uint32_t SpreadHighBits(std::uint32_t bits) {
  // 0x00204081 = 1 + 2^7 + 2^14 + 2^21: bit b of `bits` lands at 8b, every
  // cross term falls off the 0x01010101 mask.
  return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}


/// Four halves from four code bytes: 1024 + q as F16, minus `magic` (1024,
/// or 1152 for a signed byte carried as q + 128), then the affine.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 magic, __half2 scale2,
                                              __half2 bias2, __half2& lo,
                                              __half2& hi) {
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}


// ROCmFP4-FAST: 16-entry E2M1-derived codebook {0,1,2,3,4,6,8,10,0,-1,...}
// with one unsigned E4M3 scale per 32-value block. The routed F16 path expands
// each 4-bit code to its codebook byte and then reuses the Q8_0 dequant (the
// 1152 magic), so FAST rides the same fp16 WMMA core as the production experts.
__constant__ int8_t kFastCodebook[16] = {0, 1,  2,  3,  4,  6,  8,  10,
                                         0, -1, -2, -3, -4, -6, -8, -10};

__device__ __forceinline__ float FastScaleUe4m3(std::uint8_t e) {
  if (e > 0x7e)
    return 0.0f;
  const std::uint32_t exp = e >> 3;
  const std::uint32_t mant = e & 7u;
  if (exp == 0)
    return static_cast<float>(mant) * 0x1p-10f;
  return __uint_as_float(((exp + 119u) << 23) | (mant << 20));
}


// Two code bytes -> their codebook bytes via the perm LUT, matching
// get_int_from_table_16 in the mmq path.
__device__ __forceinline__ int2 FastTable16(std::uint32_t q4) {
  const std::uint32_t* values =
      reinterpret_cast<const std::uint32_t*>(kFastCodebook);
  const std::uint32_t q_even = q4;
  const std::uint32_t q_odd = (q4 >> 4);
  const std::uint32_t ve_low =
      __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707U);
  const std::uint32_t vo_low =
      __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707U);
  const std::uint32_t ve_high =
      __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707U);
  const std::uint32_t vo_high =
      __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707U);
  const std::uint32_t mask_even = 0x03020100U | ((q_even & 0x08080808U) >> 1);
  const std::uint32_t mask_odd = 0x03020100U | ((q_odd & 0x08080808U) >> 1);
  return make_int2(__builtin_amdgcn_perm(ve_high, ve_low, mask_even),
                   __builtin_amdgcn_perm(vo_high, vo_low, mask_odd));
}


template<WeightType kType, int BM, int BN, int BK, bool kPair = false>
__launch_bounds__(256) __global__
    void RoutedF16GEMMKernel(const void* __restrict__ w,
                             const __half* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ pad_bounds,
                             const std::int32_t* __restrict__ rows_in,
                             const std::int32_t* __restrict__ rows_out,
                             const float* __restrict__ swiglu_gate,
                             float* __restrict__ out,
                             __half* __restrict__ out_half, std::size_t m,
                             std::size_t k, const void* __restrict__ w_up) {
  static_assert(BM == 128 || BM == 256, "eight waves, 16-row tiles");
  static_assert(!kPair || BM == 128);
  static_assert(BN % 16 == 0 && BN / 16 <= 8);
  static_assert(BK == 2, "one stage is one 32-byte Q4_K nibble group");
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = BM / 128;  // 16-row tiles per wave
  constexpr bool kQ5 = kType == WeightType::kQ5_1;
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  constexpr bool kQ8 = kType == WeightType::kQ8_0;
  constexpr bool kFast = kType == WeightType::kQ4_0_ROCMFP4_FAST;
  constexpr bool kQ40 = kType == WeightType::kQ4_0;
  constexpr bool kQ8Like = kQ8 || kFast || kQ40;
  constexpr bool kKQuant = kType == WeightType::kQ4_K || kQ5K;
  // 16-byte code chunks per row and stage: Q4_K's nibble pair and Q5_1's
  // two nibble blocks are two, Q8_0's two byte blocks are four.
  constexpr int kChunks = kQ8Like ? 2 * BK : BK;

  // Separate projections stage codes and affine parameters in LDS, with
  // nearby rows permuted across the bank groups. Paired waves keep them in
  // registers: each half owns the same rows' first or second K16 fragment.
  // Both paths share the activation plane,
  // [kb][16-element quarter][token][16 B] so a fragment read is 256
  // contiguous bytes; the epilogue reuses it all.
  // Q5_1 rows stage their blocks as stored: a row's BK blocks are 16-byte
  // aligned, so whole 16-byte chunks load coalesced and each wave reads its
  // rows' scale, high bits and codes from LDS.
  constexpr int kQ5Bytes = BK * 24;  // one row's blocks per stage
  constexpr int kCodeBytes = kPair ? 0
                             : kQ5 ? BM * kQ5Bytes
                                   : BM * kChunks * 16;
  constexpr int kHighBytes = !kPair && kQ5K ? BK * BM * 4 : 0;
  constexpr int kScaleBytes = kPair || kQ5 ? 0 : BK * BM * 4;
  // One slot of padding per activation quarter plane: the eight chunks of
  // a token then land on eight bank groups when they are written.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // The epilogue transposes one 16x16 tile per wave through the same
  // bytes (8 KB), which the narrow tile's stages do not reach.
  constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes + kActBytes;
  constexpr int kEpilogueBytes = kPair ? BN * (BM / 2 + 2) * 2 : 8 * 1024;
  constexpr int kLdsBytes =
      kStageBytes > kEpilogueBytes ? kStageBytes : kEpilogueBytes;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_high = reinterpret_cast<std::uint32_t*>(lds + kCodeBytes);
  auto* s_scale =
      reinterpret_cast<std::uint32_t*>(lds + kCodeBytes + kHighBytes);
  auto* s_act =
      reinterpret_cast<uint4*>(lds + kCodeBytes + kHighBytes + kScaleBytes);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = RoutedF16RowBytes<kType>(k);
  const auto* w_expert = static_cast<const std::uint8_t*>(w) +
                         static_cast<std::size_t>(expert) * m * row_bytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  constexpr int kRows = kPair ? BM / 2 : BM;
  const int r_block = static_cast<int>(blockIdx.x) * kRows;

  // A paired wave owns eight gate rows and their matching eight up rows.
  // Its halves load the two 16-byte chunks of each row's nibble group;
  // each chunk carries one K16 fragment in both K32 blocks. Separate
  // projections use row=tid/2+128u and chunk=tid%2 instead.
  const int f_c = kPair ? half_id : tid & 1;
  const std::uint8_t* f_ptr[kWaveRowTiles];
  bool f_live[kWaveRowTiles];
  uint4 f_header[kWaveRowTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r =
        r_block + (kPair ? wave_id * 8 + sub_lane % 8 : (tid >> 1) + (u * 128));
    f_live[u] = r < m_i;
    const std::uint8_t* weights = w_expert;
    if constexpr (kPair) {
      if ((sub_lane & 8) != 0) {
        weights = static_cast<const std::uint8_t*>(w_up) +
                  static_cast<std::size_t>(expert) * m * row_bytes;
      }
    }
    f_ptr[u] = weights +
               static_cast<std::size_t>(f_live[u] ? r : (m_i - 1)) * row_bytes;
    f_header[u] = make_uint4(0u, 0u, 0u, 0u);
  }
  // The next stage's weights and activations, fetched one stage ahead. The
  // (scale, bias) pair is derived from the raw header word only when the
  // stage is committed, so nothing waits on the loads before the compute.
  uint4 f_codes[kWaveRowTiles];
  // Paired gate/up tiles reuse the full quantized block over four stages.
  // Keep its remaining codes in registers alongside the cached header.
  uint4 code_cache[kWaveRowTiles][4];
  uint4 f_codes_hi[kWaveRowTiles];  ///< Q8_0: the block's second 16 codes
  uint4 f_qh[kWaveRowTiles][2];     ///< Q5_K: the superblock's high bits
  std::uint32_t f_high[kWaveRowTiles];
  std::uint32_t f_dm[kWaveRowTiles];  ///< Q8_0: d
  int f_sb32[kWaveRowTiles];          ///< Q4_K: the K block in its superblock
  constexpr int kActFetch = BN <= 64 ? 2 : 4;
  uint4 a_data[kActFetch];
  // Q5_1 chunk c is part c % kQ5Parts of row c / kQ5Parts.
  constexpr int kQ5Parts = kQ5Bytes / 16;
  constexpr int kQ5Chunks = BM * kQ5Parts;
  constexpr int kQ5Fetch = kQ5 ? (kQ5Chunks + 255) / 256 : 1;
  static_assert(!kQ5 || kQ5Bytes % 16 == 0);
  const std::uint8_t* q5_src[kQ5Fetch];
  uint4 f_q5[kQ5Fetch];
  if constexpr (kQ5) {
#pragma unroll
    for (int i = 0; i < kQ5Fetch; ++i) {
      const int chunk = tid + (i * 256);
      const int r = r_block + (chunk / kQ5Parts);
      q5_src[i] = chunk < kQ5Chunks
                      ? w_expert +
                            (static_cast<std::size_t>(r < m_i ? r : m_i - 1) *
                             row_bytes) +
                            ((chunk % kQ5Parts) * 16)
                      : nullptr;
    }
  }

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; each thread fetches consecutive 256-chunk
  // strides, up to four for a 128-token tile.
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256);
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src = (chunk < kActChunks && c_row < bucket_rows)
                                 ? rows_in[bucket_begin + c_row]
                                 : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    // s_act[(kb * 4 + quarter) * kActStride + t]
    a_slot[i] = chunk < kActChunks ? (sub * kActStride) + t : -1;
  }

  const auto swizzle = [](int row, int c) {
    return (row * kChunks) +
           (c ^ (kChunks == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1)));
  };

  const auto fetch_stage = [&](int kb0, int code_group) {
    if constexpr (kQ5) {
#pragma unroll
      for (int i = 0; i < kQ5Fetch; ++i) {
        f_q5[i] = q5_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(q5_src[i] + (kb0 * 24))
                      : make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int u = 0; u < (kQ5 ? 0 : kWaveRowTiles); ++u) {
      if constexpr (kQ8) {
        // block_q8_0 is 34 bytes, so the code loads are 2-byte aligned.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 34);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        __builtin_memcpy(&f_codes[u], blk + 2, 16);
        __builtin_memcpy(&f_codes_hi[u], blk + 18, 16);
      } else if constexpr (kQ40) {
        // GGML Q4_0 block is 18 bytes: F16 scale + 16 nibble bytes (low
        // nibbles are elements 0-15, high nibbles 16-31, value = nibble-8).
        // Lossless Q8_0 presentation (same math as DequantQ40ToQ8Kernel):
        // d8 = d/16, q8 = (q-8)*16, so d8*q8 == d*(q-8) exactly.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 18);
        const float d =
            __half2float(*reinterpret_cast<const __half*>(blk));
        f_dm[u] = __builtin_bit_cast(
            std::uint32_t, __floats2half2_rn(d * 0.0625F, 0.0f));
        uint4 q;
        __builtin_memcpy(&q, blk + 2, 16);
        std::uint8_t qs[16];
        __builtin_memcpy(qs, &q, 16);
        std::int8_t c0[16], c1[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          c0[j] = static_cast<std::int8_t>(
              (static_cast<int>(qs[j] & 0xF) - 8) * 16);
          c1[j] = static_cast<std::int8_t>(
              (static_cast<int>(qs[j] >> 4) - 8) * 16);
        }
        __builtin_memcpy(&f_codes[u], c0, 16);
        __builtin_memcpy(&f_codes_hi[u], c1, 16);
      } else if constexpr (kFast) {
        // block_rocmfp4_fast is 17 bytes: 16 packed 4-bit codes then one
        // UE4M3 scale. Expand the codes to codebook bytes (low nibbles are
        // elements 0-15, high nibbles 16-31) and present them as Q8_0-shaped
        // signed bytes so the shared dequant path takes over.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 17);
        const float sc = FastScaleUe4m3(blk[16]);
        f_dm[u] =
            __builtin_bit_cast(std::uint32_t, __floats2half2_rn(sc, 0.0f));
        uint4 q;
        __builtin_memcpy(&q, blk, 16);
        const int2 r0 = FastTable16(q.x);
        const int2 r1 = FastTable16(q.y);
        const int2 r2 = FastTable16(q.z);
        const int2 r3 = FastTable16(q.w);
        f_codes[u] = make_uint4(r0.x, r1.x, r2.x, r3.x);
        f_codes_hi[u] = make_uint4(r0.y, r1.y, r2.y, r3.y);
      } else {
        constexpr int kBlockChunks = kQ5K ? 11 : 9;
        constexpr int kCodeChunk = kQ5K ? 3 : 1;
        const int block = kb0 / 8;
        const auto* blk =
            reinterpret_cast<const uint4*>(f_ptr[u]) + (block * kBlockChunks);
        // The K sweep enters a new superblock every eight Q8-sized blocks.
        if (kPair ? code_group == 0 : kb0 % 8 == 0) {
          f_header[u] = blk[0];
          if constexpr (kPair) {
#pragma unroll
            for (int group = 0; group < 4; ++group)
              code_cache[u][group] = blk[kCodeChunk + group * 2 + f_c];
          }
          if constexpr (kQ5K) {
            f_qh[u][0] = blk[1];
            f_qh[u][1] = blk[2];
          }
        }
        const int sb32 = (kPair ? code_group * 2 : kb0 % 8) + f_c;
        if constexpr (kPair) {
          // Each unrolled stage supplies a fixed index; no cache selector
          // or addressable private array remains in the compiled loop.
          f_codes[u] = code_cache[u][code_group];
        } else {
          f_codes[u] = blk[kCodeChunk + (sb32 / 2) * 2 + f_c];
        }
        f_sb32[u] = sb32;
        if constexpr (kQ5K) {
          // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
          const std::uint32_t qh[8] = {f_qh[u][0].x, f_qh[u][0].y, f_qh[u][0].z,
                                       f_qh[u][0].w, f_qh[u][1].x, f_qh[u][1].y,
                                       f_qh[u][1].z, f_qh[u][1].w};
          std::uint32_t high = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            high |= GatherBit(qh[i], sb32) << (4 * i);
          }
          f_high[u] = high;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      a_data[i] = a_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32))
                      : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  // Keep this stage's codes and affine/high-bit parameters live while the
  // next fetch can replace its quantization header and code-cache group.
  uint4 own_codes[kWaveRowTiles];
  std::uint32_t own_scale[kWaveRowTiles];
  std::uint32_t own_high[kWaveRowTiles];
  const auto commit_stage = [&]() {
    if constexpr (kQ5) {
#pragma unroll
      for (int i = 0; i < kQ5Fetch; ++i) {
        if (q5_src[i] != nullptr)
          s_codes[tid + (i * 256)] = f_q5[i];
      }
    }
#pragma unroll
    for (int u = 0; u < (kQ5 ? 0 : kWaveRowTiles); ++u) {
      const int row = (tid >> 1) + (u * 128);
      std::uint32_t scale_bias = 0;
      if constexpr (kPair) {
        own_codes[u] = f_codes[u];
        if constexpr (kQ5K)
          own_high[u] = f_high[u];
      } else if constexpr (kQ8Like) {
        s_codes[swizzle(row, 2 * f_c)] = f_codes[u];
        s_codes[swizzle(row, (2 * f_c) + 1)] = f_codes_hi[u];
        scale_bias = f_live[u] ? f_dm[u] : 0U;  // half2 (d, 0)
      } else {
        s_codes[swizzle(row, f_c)] = f_codes[u];
      }
      if constexpr (kQ5K && !kPair) {
        s_high[(f_c * BM) + row] = f_high[u];
      }
      if constexpr (kQ8Like) {
      } else {
        const int sb32 = f_sb32[u];
        // The first four scale/min bytes live in y/z. Their remaining
        // low nibbles share w; the high two bits reuse those same y/z bytes.
        // Decode the fixed words directly, avoiding header-member selection.
        const uint4 h = f_header[u];
        const unsigned shift = 8U * (static_cast<unsigned>(sb32) & 3U);
        const std::uint32_t scale_byte = h.y >> shift;
        const std::uint32_t min_byte = h.z >> shift;
        std::uint32_t sc;
        std::uint32_t mn;
        if (sb32 < 4) {
          sc = scale_byte & 0x3FU;
          mn = min_byte & 0x3FU;
        } else {
          const std::uint32_t packed = h.w >> shift;
          sc = (packed & 0x0FU) | ((scale_byte & 0xC0U) >> 2U);
          mn = ((packed >> 4U) & 0x0FU) | ((min_byte & 0xC0U) >> 2U);
        }
        const __half2 dm = __builtin_bit_cast(__half2, f_header[u].x);
        const float scale =
            f_live[u] ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
        const float offset =
            f_live[u] ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
        scale_bias = __builtin_bit_cast(std::uint32_t,
                                        __floats2half2_rn(scale, -offset));
      }
      if constexpr (kPair)
        own_scale[u] = scale_bias;
      else
        s_scale[(f_c * BM) + row] = scale_bias;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_slot[i] >= 0) {
        s_act[a_slot[i]] = a_data[i];
      }
    }
  };

  v8f acc[kWaveRowTiles][kTokTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      acc[u][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  const __half2 magic = __floats2half2_rn(kQ8Like ? -1152.0F : -1024.0F,
                                          kQ8Like ? -1152.0F : -1024.0F);
  const auto compute_stage = [&]() {
    uint4 raw[kWaveRowTiles][BK];
    if constexpr (!kQ8Like && !kQ5 && !kPair) {
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
#pragma unroll
        for (int c = 0; c < BK; ++c) {
          raw[u][c] = s_codes[swizzle(row, c)];
        }
      }
    }
    std::uint32_t peer_scale[kWaveRowTiles];
    std::uint32_t peer_high[kWaveRowTiles];
    if constexpr (kPair) {
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        peer_scale[u] = XorLane(own_scale[u], 16);
        if constexpr (kQ5K)
          peer_high[u] = XorLane(own_high[u], 16);
      }
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
      if constexpr (kPair) {
        // Decode each K16 fragment once, then duplicate its packed F16 bits
        // across the wave halves for WMMA. No weight data passes through LDS.
        using v8u = __attribute__((__vector_size__(32))) std::uint32_t;
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          const __half2 sb = __builtin_bit_cast(
              __half2, half_id == kb ? own_scale[u] : peer_scale[u]);
          const __half2 scale2 = __low2half2(sb);
          const __half2 bias2 = __high2half2(sb);
          const uint4 codes = own_codes[u];
          const std::uint32_t words[4] = {codes.x, codes.y, codes.z, codes.w};
          const std::uint32_t high =
              kQ5K ? ((half_id == kb ? own_high[u] : peer_high[u]) >>
                      (16U * half_id))
                   : 0U;
          __half2 h[8];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            std::uint32_t nib = (words[i] >> (4U * kb)) & 0x0F0F0F0FU;
            if constexpr (kQ5K)
              nib |= SpreadHighBits((high >> (4 * i)) & 0xFU);
            CodesToHalves(nib, magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
          }
          v16h own;
          __builtin_memcpy(&own, &h[0], 32);
          const v8u packed = __builtin_bit_cast(v8u, own);
          v8u peer;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            peer[i] = XorLane(packed[i], 16);
          a_lo[u] = __builtin_bit_cast(v16h, half_id == 0 ? packed : peer);
          a_hi[u] = __builtin_bit_cast(v16h, half_id == 1 ? packed : peer);
        }
      } else {
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          const int row = (wave_id * 16) + (u * 128) + sub_lane;
          // Q5_1 block kb of the row: d | m, the high-bit word, 16 codes.
          uint2 q5_head{};
          uint2 q5_lo{};
          uint2 q5_hi{};
          std::uint32_t scale_bias;
          if constexpr (kQ5) {
            const std::uint8_t* blk = lds + (row * kQ5Bytes) + (kb * 24);
            q5_head = *reinterpret_cast<const uint2*>(blk);
            q5_lo = *reinterpret_cast<const uint2*>(blk + 8);
            q5_hi = *reinterpret_cast<const uint2*>(blk + 16);
            scale_bias = r_block + row < m_i ? q5_head.x : 0U;
          } else {
            scale_bias = s_scale[(kb * BM) + row];
          }
          const __half2 sb = __builtin_bit_cast(__half2, scale_bias);
          const __half2 scale2 = __low2half2(sb);
          const __half2 bias2 = __high2half2(sb);
          std::uint32_t nib[8];
          if constexpr (kQ8Like) {
            // Q8_0: the block's 32 signed bytes are chunks 2 kb and 2 kb + 1;
            // flipping the sign bit carries q + 128, which the 1152 magic
            // takes back out.
            const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
            const uint4 c1 = s_codes[swizzle(row, (2 * kb) + 1)];
            const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w,
                                            c1.x, c1.y, c1.z, c1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              nib[i] = words[i] ^ 0x80808080U;
            }
          } else if constexpr (kQ5) {
            // Q5_1: elements 0-15 take the low nibbles of the 16 code
            // bytes, 16-31 the high, plus bit j of the high-bit word.
            const std::uint32_t high = q5_head.y;
            const std::uint32_t words[4] = {q5_lo.x, q5_lo.y, q5_hi.x, q5_hi.y};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              nib[i] = (words[i] & 0x0F0F0F0FU) |
                       SpreadHighBits((high >> (4 * i)) & 0xFU);
              nib[4 + i] = ((words[i] >> 4U) & 0x0F0F0F0FU) |
                           SpreadHighBits((high >> (16 + 4 * i)) & 0xFU);
            }
          } else {
            // Q4_K / Q5_K: elements 0-15 of K block kb0 + kb are the low
            // (kb = 0) or high (kb = 1) nibbles of chunk 0, elements 16-31
            // of chunk 1; Q5_K adds bit j of the staged high-bit word.
            const unsigned shift = 4U * static_cast<unsigned>(kb);
            const std::uint32_t words[8] = {
                raw[u][0].x, raw[u][0].y, raw[u][0].z, raw[u][0].w,
                raw[u][1].x, raw[u][1].y, raw[u][1].z, raw[u][1].w};
            const std::uint32_t high = kQ5K ? s_high[(kb * BM) + row] : 0U;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              nib[i] = (words[i] >> shift) & 0x0F0F0F0FU;
              if constexpr (kQ5K) {
                nib[i] |= SpreadHighBits((high >> (4 * i)) & 0xFU);
              }
            }
          }
          __half2 h[16];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            CodesToHalves(nib[i], magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
          }
          __builtin_memcpy(&a_lo[u], &h[0], 32);
          __builtin_memcpy(&a_hi[u], &h[8], 32);
        }
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if constexpr (kPair || ((kQ5 || kQ8Like) && BN >= 48)) {
          // Keep one token tile's LDS fragments live at a time. Hoisting
          // all eight tiles spills registers and defeats the wider tile's
          // reuse of each weight decode. This is a compiler barrier only.
          asm volatile("" ::: "memory");
        }
        // A short expert bucket has no output in the remaining token
        // tiles, so omit their WMMA work.
        if constexpr ((kPair || ((kQ5 || kQ8Like) && BN >= 48)) &&
                      kTokTiles > 1) {
          if (j >= live_tok_tiles)
            continue;
        }
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          b[q] = frag[q * kActStride];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          acc[u][j] = Wmma(a_lo[u], b_lo, acc[u][j]);
          acc[u][j] = Wmma(a_hi[u], b_hi, acc[u][j]);
        }
      }
    }
  };

  const auto run_stage = [&](int kb0, int next_group) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK, next_group);
    }
    compute_stage();
    __syncthreads();
  };
  fetch_stage(0, 0);
  if constexpr (kPair) {
    // K-quant launchers require complete 256-element blocks. Expose their
    // four stages so cached code selection and scale shifts are constant.
    for (int kb0 = 0; kb0 < num_kb; kb0 += 8) {
      run_stage(kb0, 1);
      run_stage(kb0 + 2, 2);
      run_stage(kb0 + 4, 3);
      run_stage(kb0 + 6, 0);
    }
  } else {
    for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
      run_stage(kb0, 0);
    }
  }

  if constexpr (kPair) {
    // A tile has eight gate rows followed by their matching up rows. Each
    // lane's two projections stay in registers through SwiGLU. Stage all
    // F16 results once so stores cover the full 64-column output row; two
    // padding halves distribute the producer writes across LDS banks.
    constexpr int kPairStride = kRows + 2;
    static_assert(BN * kPairStride * sizeof(__half) <= kLdsBytes);
    __half* scratch = reinterpret_cast<__half*>(lds);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      if (j >= live_tok_tiles)
        continue;
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        float product = acc[0][j][l + 4] * acc[0][j][l];
        asm volatile("" : "+v"(product));
        float value = product * SigmoidF(acc[0][j][l]);
        asm volatile("" : "+v"(value));
        scratch[(j * 16 + sub_lane) * kPairStride + wave_id * 8 + 2 * l +
                half_id] = __float2half(value);
      }
    }
    __syncthreads();
#pragma unroll
    for (int unit = 0; unit < BN * kRows / (2 * 256); ++unit) {
      const int flat = (unit * 256 + tid) * 2;
      const int t = t_local + flat / kRows;
      const int r = r_block + flat % kRows;
      if (t < bucket_rows && r < m_i) {
        const std::int32_t dst = rows_out[bucket_begin + t];
        if (dst >= 0) {
          const auto offset = static_cast<std::size_t>(dst) * m + r;
          const int src = (flat / kRows) * kPairStride + flat % kRows;
          if (m % 2 == 0 && r + 1 < m_i) {
            *reinterpret_cast<__half2*>(out_half + offset) =
                *reinterpret_cast<const __half2*>(scratch + src);
          } else {
            out_half[offset] = scratch[src];
            if (r + 1 < m_i)
              out_half[offset + 1] = scratch[src + 1];
          }
        }
      }
    }
    return;
  }

  // One wave writes a complete 128-byte line of F16 output. Padding the
  // shared row by two floats also makes the accumulator scatter conflict-free.
  // Narrow buckets keep the lighter wave-local epilogue below.
  if constexpr (BN >= 48) {
    static_assert(kLdsBytes >= 16 * (BM + 2) * sizeof(float));
    if (out_half != nullptr) {
      constexpr unsigned stride = BM + 2;
      float* scratch = reinterpret_cast<float*>(lds);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
          for (int l = 0; l < 8; ++l)
            scratch[sub_lane * stride + wave_id * 16 + u * 128 + 2 * l +
                    half_id] = acc[u][j][l];
        }
        __syncthreads();
#pragma unroll
        for (int round = 0; round < 16 * BM / (256 * 2); ++round) {
          const unsigned flat = (round * 256 + tid) * 2;
          const unsigned tr = flat / BM, row = flat % BM;
          const unsigned t = t_local + j * 16 + tr, r = r_block + row;
          if (t < unsigned(bucket_rows) && r < m) {
            const int dst = rows_out[bucket_begin + t];
            if (dst >= 0) {
              float2 v =
                  *reinterpret_cast<const float2*>(scratch + tr * stride + row);
              const size_t o = size_t(dst) * m + r;
              if (swiglu_gate != nullptr) {
                v.x *= SiluF(swiglu_gate[o]);
                if (r + 1 < m)
                  v.y *= SiluF(swiglu_gate[o + 1]);
              }
              if (m % 2 == 0 && r + 1 < m)
                *reinterpret_cast<__half2*>(out_half + o) =
                    __floats2half2_rn(v.x, v.y);
              else {
                out_half[o] = __float2half(v.x);
                if (r + 1 < m)
                  out_half[o + 1] = __float2half(v.y);
              }
            }
          }
        }
        __syncthreads();
      }
      return;
    }
  }
  // Transpose each 16x16 tile through LDS, then scatter the 16 rows of each
  // token to its output row.
  float* tile_scratch = reinterpret_cast<float*>(lds) + (wave_id * 256);
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r0 = r_block + (wave_id * 16) + (u * 128);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[u][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const int t0 = t_local + (j * 16);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = (s * 32) + lane_id;
        const int t = t0 + (flat >> 4);
        const int r = r0 + (flat & 15);
        if (t < bucket_rows && r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const std::size_t o = (static_cast<std::size_t>(dst) * m) +
                                  static_cast<std::size_t>(r);
            float v = tile_scratch[flat];
            if (out_half != nullptr) {
              out_half[o] = __float2half(
                  swiglu_gate != nullptr ? v * SiluF(swiglu_gate[o]) : v);
            } else {
              out[o] = v;
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}


/// Compacts the routed assignments by expert with 16-row padded buckets.
/// One block: exclusive scan of the padded counts into pad_bounds[0..E].
__global__ void RoutedPadBoundsKernel(const std::uint32_t* __restrict__ counts,
                                      std::int32_t* __restrict__ pad_bounds,
                                      std::int32_t* __restrict__ cursors,
                                      std::int32_t* __restrict__ rows_token,
                                      std::int32_t* __restrict__ rows_slot,
                                      std::uint32_t n_experts) {
  __shared__ std::int32_t padded[1024];
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    padded[e] = static_cast<std::int32_t>((counts[e] + 15u) / 16u * 16u);
    cursors[e] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      pad_bounds[e] = running;
      running += padded[e];
    }
    pad_bounds[n_experts] = running;
  }
  __syncthreads();
  // Scatter overwrites every occupied row. Only the padding can retain a
  // previous batch's assignments; clear it while the bucket bounds are hot.
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    for (std::uint32_t row = counts[e];
         row < static_cast<std::uint32_t>(padded[e]); ++row) {
      const auto at = pad_bounds[e] + row;
      rows_token[at] = -1;
      rows_slot[at] = -1;
    }
  }
}

/// rows_token[c] / rows_slot[c] for every routed (token, slot); the order
/// inside a bucket is whatever the atomics produce, which changes nothing:
/// every output row is computed from its own inputs only.
__global__ void RoutedScatterKernel(const std::int32_t* __restrict__ ids,
                                    const std::int32_t* __restrict__ pad_bounds,
                                    std::int32_t* __restrict__ cursors,
                                    std::int32_t* __restrict__ rows_token,
                                    std::int32_t* __restrict__ rows_slot,
                                    std::uint32_t slots, std::uint32_t k) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const std::int32_t e = ids[slot];
  if (e < 0) {
    return;
  }
  const std::int32_t c = pad_bounds[e] + atomicAdd(&cursors[e], 1);
  rows_token[c] = static_cast<std::int32_t>(slot / k);
  rows_slot[c] = static_cast<std::int32_t>(slot);
}


template<int BN>
bool LaunchRoutedF16(const void* w, WeightType type, const __half* x,
                     const std::int32_t* tiles, std::uint32_t n_tiles,
                     const std::int32_t* pad_bounds,
                     const std::int32_t* rows_in, const std::int32_t* rows_out,
                     const float* swiglu_gate, float* out, __half* out_half,
                     std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr int kBM = 128;
  constexpr int kBK = 2;
  const dim3 grid(static_cast<unsigned int>((m + kBM - 1) / kBM), n_tiles);
  switch (type) {
    case WeightType::kQ4_K:
      if constexpr (BN > 48) {
        return false;
      } else {
        hipLaunchKernelGGL(
            (RoutedF16GEMMKernel<WeightType::kQ4_K, kBM, BN, kBK>), grid,
            dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
            rows_out, swiglu_gate, out, out_half, m, k, nullptr);
        return true;
      }
    case WeightType::kQ5_1:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ5_1, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, swiglu_gate, out,
                         out_half, m, k, nullptr);
      return true;
    case WeightType::kQ8_0:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ8_0, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, swiglu_gate, out,
                         out_half, m, k, nullptr);
      return true;
    case WeightType::kQ4_0:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ4_0, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, swiglu_gate, out,
                         out_half, m, k, nullptr);
      return true;
    case WeightType::kQ4_0_ROCMFP4_FAST:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ4_0_ROCMFP4_FAST, kBM, BN, kBK>),
          grid, dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
          rows_out, swiglu_gate, out, out_half, m, k, nullptr);
      return true;
    case WeightType::kQ5_K:
      if constexpr (BN > 48) {
        return false;
      } else {
        hipLaunchKernelGGL(
            (RoutedF16GEMMKernel<WeightType::kQ5_K, kBM, BN, kBK>), grid,
            dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
            rows_out, swiglu_gate, out, out_half, m, k, nullptr);
        return true;
      }
    default:
      return false;
  }
}


__global__ void ExpertCountsKernel(const std::int32_t* ids,
                                   std::uint32_t* counts, std::size_t slots) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < slots && ids[i] >= 0) {
    atomicAdd(counts + ids[i], 1u);
  }
}

}  // namespace

std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts) {
  return slots + (n_experts * (kRoutedTileTokens - 1));
}

void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream) {
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  hipLaunchKernelGGL(RoutedPadBoundsKernel, dim3(1), dim3(1024), 0, stream,
                     counts, pad_bounds, cursors, rows_token, rows_slot,
                     n_experts);
  hipLaunchKernelGGL(RoutedScatterKernel, dim3(Blocks(slots)), dim3(kThreads),
                     0, stream, ids, pad_bounds, cursors, rows_token, rows_slot,
                     static_cast<std::uint32_t>(slots), k);
}


bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream) {
  const std::size_t block_elems =
      (type == WeightType::kQ4_K || type == WeightType::kQ5_K) ? 256 : 64;
  if (m == 0 || k == 0 || k % block_elems != 0 || n_tiles == 0 ||
      (out_half == nullptr) == (out == nullptr)) {
    return false;
  }
  switch (tile_rows) {
    case 16:
      return LaunchRoutedF16<16>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 48:
      return LaunchRoutedF16<48>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 64:
      return LaunchRoutedF16<64>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    default:
      return false;
  }
}


void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream) {
  if (n_experts <= 512) {
    hipLaunchKernelGGL((RouterTopKKernel<512>), dim3(n_tokens), dim3(kThreads),
                       0, stream, logits, stride, ids, weights, n_experts, k);
  } else {
    hipLaunchKernelGGL((RouterTopKKernel<1024>), dim3(n_tokens), dim3(kThreads),
                       0, stream, logits, stride, ids, weights, n_experts, k);
  }
}


void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream) {
  (void)hipMemsetAsync(counts, 0, n_experts * sizeof(std::uint32_t), stream);
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  hipLaunchKernelGGL(ExpertCountsKernel, dim3(Blocks(slots)), dim3(kThreads), 0,
                     stream, ids, counts, slots);
}

}  // namespace gufo::core::compute::gemm

#endif  // ENGINE_ENABLE_HIP
