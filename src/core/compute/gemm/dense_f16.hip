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

// Rope declarations for the template's attention epilogue (kAttention only;
// core instantiates kAttention=false, so these are never ODR-used here).
namespace qwen {
namespace vision {
struct DeviceRope;
__device__ inline float RopePosition(const DeviceRope* rope,
                                     std::uint32_t physical,
                                     std::uint32_t pair);
}  // namespace vision
}  // namespace qwen

// YarnRope + pair multiplier, verbatim from
// models/qwen38_flash_next/rope_scaling.hpp (plain-data rope params).
/// Launch constants shared by the CPU oracle and every HIP rope site.
struct YarnRope {
  std::uint32_t enabled{0};
  float low{0.0F};         ///< First ramped rotary pair.
  float high{0.0F};        ///< Pair from which frequencies are divided.
  float inv_factor{1.0F};  ///< 1 / factor.
  float mscale{1.0F};      ///< Multiplies cos and sin.
};

__host__ __device__ inline float YarnPairMultiplier(const YarnRope& yarn,
                                                std::uint32_t pair) noexcept {
  const float width = yarn.high - yarn.low;
  const float span = width > 0.001F ? width : 0.001F;
  float ramp = (static_cast<float>(pair) - yarn.low) / span;
  ramp = ramp < 0.0F ? 0.0F : (ramp > 1.0F ? 1.0F : ramp);
  return 1.0F - ramp * (1.0F - yarn.inv_factor);
}

__device__ __forceinline__ float YarnFrequency(float freq, std::uint32_t pair,
                                               const YarnRope& yarn) {
  float scaled = freq * YarnPairMultiplier(yarn, pair);
  asm volatile("" : "+v"(scaled));
  return scaled;
}

// YaRN attention factor on cos and sin, opaque for the same reason: the
// sites rotate with different expressions (explicit FMA or fast-math
// contraction), which must all see the same rounded scaled values.
__device__ __forceinline__ void YarnScale(float& sine, float& cosine,
                                          const YarnRope& yarn) {
  sine *= yarn.mscale;
  cosine *= yarn.mscale;
  asm volatile("" : "+v"(sine), "+v"(cosine));
}

// Keep each head's original eight-wave reduction. Heads in a block share
// rotary angles; normalization statistics remain independent.

// Optional output layout for the stacked QKV projection. The projection's
// 256-row tile covers one query, gate, key or value head.
struct AttentionProjectionOutput {
  const float* q_gamma;
  const float* k_gamma;
  float* query;
  float* gate;
  __half* keys;
  __half* values;
  const std::uint32_t* position;
  float theta;
  float eps;
  const qwen::vision::DeviceRope* rope;
  YarnRope yarn;
  std::uint8_t* keys_q8{nullptr};    ///< Q8_0 blocks (fused Q8 path)
  std::uint8_t* values_q8{nullptr};  ///< Q8_0 blocks (fused Q8 path)
};

namespace {

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float SoftplusF(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
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


/// registers while cutting the number of blocks and inject partials by almost
/// four for the model's 2,560-wide, four-stream rows.
__device__ __forceinline__ float4 Load4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ float4 Load4(const __half* p) {
  const auto packed = *reinterpret_cast<const __half2*>(p);
  const auto packed_hi = *reinterpret_cast<const __half2*>(p + 2);
  const float2 lo = __half22float2(packed);
  const float2 hi = __half22float2(packed_hi);
  return float4{lo.x, lo.y, hi.x, hi.y};
}


using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// BF16 fragments carry the same sixteen 16-bit lanes.
using v16bf = __attribute__((ext_vector_type(16))) __bf16;

__device__ __forceinline__ v8f WmmaBf16(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
      __builtin_bit_cast(v16bf, a), __builtin_bit_cast(v16bf, b), c);
}


constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

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


/// Sum over one wave; every lane receives the total. The wave-per-row
/// kernels split rows across lanes (gfx1151 runs wave32).
__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += XorLane(v, offset);
  }
  return v;
}


__device__ __forceinline__ float SsmConv4Value(float4 w, float x0, float x1,
                                               float x2, float x3) {
  float acc = __fmaf_rn(w.x, x0, __fmul_rn(w.y, x1));
  acc = __fmaf_rn(w.z, x2, acc);
  acc = __fmaf_rn(w.w, x3, acc);
  return SiluF(acc);
}

constexpr unsigned kSsmProjectionTileTokens = 32;


template<int BM, int BN, int BK, int WM, int WN, int kRowGroup = 1,
         bool kHcMix = false, bool kSsmConv = false, bool kAttention = false,
         bool kHalfWeights = false, bool kBf16 = false, bool kStrided = false,
         bool kQ8Out = false, bool kYarn = false>
__launch_bounds__(256) __global__ void DenseF16GEMMKernel(
    const void* __restrict__ w, const __half* __restrict__ x,
    float* __restrict__ y, std::size_t batch, std::size_t m, std::size_t k,
    const __half* xn = nullptr, __half* mixed_half = nullptr,
    void* mixed_q8 = nullptr, const float* conv_w = nullptr,
    float* conv_out = nullptr, AttentionProjectionOutput attention = {},
    std::uint32_t checkpoint_tokens = 0, std::size_t input_stride = 0) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(!(kBf16 && kHalfWeights));
  // 16-bit weight rows staged as they are: F16, or BF16 with BF16
  // activations.
  constexpr bool kRawWeights = kHalfWeights || kBf16;
  constexpr bool kRegisterWeights = kSsmConv;
  static_assert(!kRegisterWeights ||
                (BM == 256 && BK == 2 && WM == 8 && WN == 1));
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  // Weight and activation K blocks staged per thread; a unit past the
  // stage's block count is idle.
  constexpr int kAUnits = BM * BK;
  constexpr int kBUnits = BN * BK;
  constexpr int kAPer = (kAUnits + 255) / 256;
  constexpr int kBPer = (kBUnits + 255) / 256;

  // One K block of one row or token is four 16-byte chunks; the chunks of
  // nearby rows are permuted so a fragment read (one row per lane, 64-byte
  // stride) covers all bank groups.
  constexpr int kLdsChunks = BK * ((kRegisterWeights ? 0 : BM) + BN) * 4;
  constexpr int kTransposeChunks = 8 *
                                   (kSsmConv ? kSsmProjectionTileTokens : 16) *
                                   36 * sizeof(float) / sizeof(uint4);
  constexpr int kLdsStorage =
      kLdsChunks < kTransposeChunks ? kTransposeChunks : kLdsChunks;
  __shared__ __attribute__((aligned(16))) uint4 s_lds[kLdsStorage];
  auto* s_a = reinterpret_cast<uint4(*)[BM][4]>(s_lds);
  auto* s_b = reinterpret_cast<uint4(*)[BN][4]>(
      s_lds + (kRegisterWeights ? 0 : BK * BM * 4));
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };

  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const auto* w_bytes = static_cast<const std::uint8_t*>(w);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;
  // Group row tiles for the narrow HC up projection. The tile grid's row
  // count is divisible by kRowGroup; every dot product keeps its K order.
  const unsigned row_group = blockIdx.y / kRowGroup;
  const unsigned within = (blockIdx.y % kRowGroup) * gridDim.x + blockIdx.x;
  const int r_block = (row_group * kRowGroup + within % kRowGroup) * BM;
  const int t_block = (within / kRowGroup) * BN;
  const auto token_of = [&](int local) { return t_block + local; };

  // SSM thread tid owns row tid and K block p of the stage. Other paths
  // use row (p*256+tid)/BK and K block (p*256+tid)%BK; rows past m read
  // the last row with a zero scale.
  const std::uint8_t* a_ptr[kAPer];
  bool a_live[kAPer];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = (p * 256) + tid;
    const int r = r_block + (kRegisterWeights ? tid : idx / BK);
    a_live[p] = idx < kAUnits && r < m_i;
    // Interleave the four gate streams within the tile, without repacking
    // weights. Each group of four rows produces one mixed hidden element.
    const int weight_row = kHcMix ? (r % 4) * (m_i / 4) + r / 4 : r;
    a_ptr[p] = w_bytes +
               static_cast<std::size_t>(a_live[p] ? weight_row : (m_i - 1)) *
                   static_cast<std::size_t>(num_kb) * (kRawWeights ? 64 : 34);
  }
  const __half* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = (p * 256) + tid;
    const int t = token_of(idx / BK);
    b_ptr[p] =
        idx < kBUnits && t < static_cast<int>(batch)
            ? x + (static_cast<std::size_t>(t) * (kStrided ? input_stride : k))
            : nullptr;
  }
  uint4 a_codes[kAPer][2];
  uint4 a_half[kRawWeights ? kAPer : 1][4];
  std::uint32_t a_d[kAPer];
  uint4 b_data[kBPer][4];

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int kb = kb0 + (kRegisterWeights ? p : (((p * 256) + tid) % BK));
      const bool live = a_live[p] && kb < num_kb;
      if constexpr (kRawWeights) {
        const auto* src = reinterpret_cast<const uint4*>(
            a_ptr[p] + static_cast<std::size_t>(kb) * 64);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          a_half[p][c] = live ? src[c] : make_uint4(0u, 0u, 0u, 0u);
      } else {
        const std::uint8_t* blk =
            a_ptr[p] +
            (static_cast<std::size_t>(live ? kb : (num_kb - 1)) * 34);
        a_d[p] = live ? *reinterpret_cast<const std::uint16_t*>(blk) : 0U;
        __builtin_memcpy(&a_codes[p][0], blk + 2, 16);
        __builtin_memcpy(&a_codes[p][1], blk + 18, 16);
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      if (b_ptr[p] != nullptr && kb < num_kb) {
        const auto* src = reinterpret_cast<const uint4*>(b_ptr[p] + (kb * 32));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = src[c];
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  const __half2 magic = __floats2half2_rn(-1152.0F, -1152.0F);
  const __half2 zero2 = __floats2half2_rn(0.0F, 0.0F);
  // Preserve this stage's rounded fragments before prefetch overwrites
  // the raw Q8 codes. Each SSM wave owns both of its 16-row output tiles.
  v16h own_a_lo[BK];
  v16h own_a_hi[BK];
  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kAUnits) {
        break;
      }
      const int row = idx / BK;
      const int kk = idx % BK;
      if constexpr (kRawWeights) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          s_a[kk][row][swizzle(row, c)] = a_half[p][c];
      } else {
        // Q8_0 codes are signed; flipping the sign bit carries q + 128, which
        // the 1152 magic takes back out.
        const std::uint32_t words[8] = {
            a_codes[p][0].x, a_codes[p][0].y, a_codes[p][0].z, a_codes[p][0].w,
            a_codes[p][1].x, a_codes[p][1].y, a_codes[p][1].z, a_codes[p][1].w};
        const __half2 scale2 = __half2half2(
            __builtin_bit_cast(__half, static_cast<std::uint16_t>(a_d[p])));
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          CodesToHalves(words[i] ^ 0x80808080U, magic, scale2, zero2, h[2 * i],
                        h[2 * i + 1]);
        }
        if constexpr (kRegisterWeights) {
          __builtin_memcpy(&own_a_lo[p], &h[0], 32);
          __builtin_memcpy(&own_a_hi[p], &h[8], 32);
        } else {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            uint4 v;
            __builtin_memcpy(&v, &h[4 * c], 16);
            s_a[kk][row][swizzle(row, c)] = v;
          }
        }
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kBUnits) {
        break;
      }
      const int t = idx / BK;
      const int kk = idx % BK;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        s_b[kk][t][swizzle(t, c)] = b_data[p][c];
      }
    }
  };

  v8f acc[kWaveRowTiles][kWaveTokTiles];
  // Separate K16 chains reduce FP32 accumulation error for the sensitive
  // unquantized router/gate projections, with the same order in every chunk.
  v8f acc_high[kHalfWeights ? kWaveRowTiles : 1]
              [kHalfWeights ? kWaveTokTiles : 1]{};
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      acc[i][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  // Keep full tiles free of per-K bounds branches. The wide SSM paths can
  // omit whole 16-token subtiles in their final partial tile while retaining
  // the original WMMA order for every live output. One launch preserves
  // weight reuse between neighboring token tiles.
  const auto accumulate = [&]<bool partial>() {
    fetch_stage(0);
    for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
      commit_stage();
      __syncthreads();
      if (kb0 + BK < num_kb) {
        fetch_stage(kb0 + BK);
      }
      // Keeps the next stage's loads ahead of the matrix work: scheduled
      // freely, the compiler sinks them below it and the commit waits on
      // the full memory latency.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int kb = 0; kb < BK; ++kb) {
        v16h a_lo[kWaveRowTiles];
        v16h a_hi[kWaveRowTiles];
        if constexpr (kRegisterWeights) {
          using v8u = __attribute__((__vector_size__(32))) std::uint32_t;
          const v8u own_lo = __builtin_bit_cast(v8u, own_a_lo[kb]);
          const v8u own_hi = __builtin_bit_cast(v8u, own_a_hi[kb]);
          v8u peer_lo;
          v8u peer_hi;
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            peer_lo[q] = XorLane(own_lo[q], 16);
            peer_hi[q] = XorLane(own_hi[q], 16);
          }
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
            a_lo[i] = __builtin_bit_cast(v16h, half_id == i ? own_lo : peer_lo);
            a_hi[i] = __builtin_bit_cast(v16h, half_id == i ? own_hi : peer_hi);
          }
        } else {
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
            const int row = (((wave_row * kWaveRowTiles) + i) * 16) + sub_lane;
            uint4 c[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              c[q] = s_a[kb][row][swizzle(row, q)];
            }
            __builtin_memcpy(&a_lo[i], &c[0], 32);
            __builtin_memcpy(&a_hi[i], &c[2], 32);
          }
        }
#pragma unroll
        for (int j = 0; j < kWaveTokTiles; ++j) {
          if constexpr (partial) {
            if (token_of(((wave_tok * kWaveTokTiles) + j) * 16) >= batch)
              continue;
          }
          const int t = (((wave_tok * kWaveTokTiles) + j) * 16) + sub_lane;
          uint4 c[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            c[q] = s_b[kb][t][swizzle(t, q)];
          }
          v16h b_lo;
          v16h b_hi;
          __builtin_memcpy(&b_lo, &c[0], 32);
          __builtin_memcpy(&b_hi, &c[2], 32);
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
            if constexpr (kBf16) {
              // One K16 chain in K order for every token.
              acc[i][j] = WmmaBf16(a_lo[i], b_lo, acc[i][j]);
              acc[i][j] = WmmaBf16(a_hi[i], b_hi, acc[i][j]);
            } else if constexpr (kHalfWeights) {
              acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
              acc_high[i][j] = Wmma(a_hi[i], b_hi, acc_high[i][j]);
            } else {
              acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
              acc[i][j] = Wmma(a_hi[i], b_hi, acc[i][j]);
            }
          }
        }
      }
      __syncthreads();
    }
  };
  if constexpr (kSsmConv || kStrided) {
    if (t_block + BN <= batch)
      accumulate.template operator()<false>();
    else
      accumulate.template operator()<true>();
  } else {
    accumulate.template operator()<false>();
  }

  if constexpr (kHalfWeights) {
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j)
        acc[i][j] += acc_high[i][j];
    }
  }

  if constexpr (kAttention) {
    static_assert(BM == 256 && BN == 128 && WM == 8 && WN == 1);
    static_assert(!kHcMix && !kSsmConv);
    constexpr unsigned stride = 36, dim = 256, width = 6144, kvwidth = 512;
    float* scratch = reinterpret_cast<float*>(s_lds);
    float* tile = scratch + wave_id * 16 * stride;
    const unsigned projection_head = r_block / 256;
    const bool query = projection_head < 48 && (projection_head % 2) == 0;
    const bool key = projection_head >= 48 && projection_head < 50;
    const bool gate = projection_head < 48 && (projection_head % 2) == 1;
    const unsigned head =
        projection_head < 48 ? projection_head / 2 : projection_head % 2;
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile[sub_lane * stride + 2 * l + half_id] = acc[0][j][l];
        tile[sub_lane * stride + 16 + 2 * l + half_id] = acc[1][j][l];
      }
      __syncthreads();
#pragma unroll
      for (unsigned phase = 0; phase < 2; ++phase) {
        // One wave handles a token, emulating its original eight-wave norm.
        const unsigned tok_local = wave_id * 2 + phase;
        const std::size_t tok = t_block + j * 16 + tok_local;
        if (tok < batch) {
          float v[8];
#pragma unroll
          for (unsigned c = 0; c < 8; ++c)
            v[c] = scratch[(c * 16 + tok_local) * stride + lane_id];
          if (query || key) {
            float total = 0.0F;
#pragma unroll
            for (unsigned c = 0; c < 8; ++c) {
              // Match the separate norm: round each square and each wave sum
              // before accumulating the eight partials in their original order.
              float sq = v[c] * v[c];
              asm volatile("" : "+v"(sq));
              float ss = WaveSum(sq);
              asm volatile("" : "+v"(ss));
              total += ss;
              asm volatile("" : "+v"(total));
            }
            const float scale = rsqrtf(total / 256.0F + attention.eps);
            const float* gamma = query ? attention.q_gamma : attention.k_gamma;
#pragma unroll
            for (unsigned c = 0; c < 8; ++c) {
              v[c] = v[c] * scale;
              asm volatile("" : "+v"(v[c]));
              v[c] = v[c] * gamma[c * 32 + lane_id];
              asm volatile("" : "+v"(v[c]));
            }
            float freq = powf(attention.theta,
                              -2.0F * static_cast<float>(lane_id) / 64.0F);
            if constexpr (kYarn)
              freq = YarnFrequency(freq, lane_id, attention.yarn);
            float sn = 0.0F, cs = 0.0F;
            sincosf(qwen::vision::RopePosition(
                        attention.rope, *attention.position + tok, lane_id) *
                        freq,
                    &sn, &cs);
            if constexpr (kYarn)
              YarnScale(sn, cs, attention.yarn);
            const float lo = __fmaf_rn(v[0], cs, -__fmul_rn(v[1], sn));
            const float hi = __fmaf_rn(v[0], sn, __fmul_rn(v[1], cs));
            v[0] = lo;
            v[1] = hi;
          }
#pragma unroll
          for (unsigned c = 0; c < 8; ++c) {
            const unsigned col = c * 32 + lane_id;
            if (query)
              attention.query[tok * width + head * dim + col] = v[c];
            else if (gate)
              attention.gate[tok * width + head * dim + col] = v[c];
            else if (key) {
              // Compile-time Q8/F16 split: the F16 arm is the untouched
              // pre-Q8 body, so its codegen (VGPR/occupancy) cannot drift
              // from main. Only AttentionQ8Gemm instantiates kQ8Out.
              if constexpr (kQ8Out) {
                if (attention.keys_q8 != nullptr) {
                  // Fused Q8_0: the wave's lanes hold block c (lane l = elem
                  // l), so the per-block max is a wave-local shuffle reduction.
                  float amax = fabsf(v[c]);
                  amax = fmaxf(amax, XorLane(amax, 16));
                  amax = fmaxf(amax, XorLane(amax, 8));
                  amax = fmaxf(amax, XorLane(amax, 4));
                  amax = fmaxf(amax, XorLane(amax, 2));
                  amax = fmaxf(amax, XorLane(amax, 1));
                  const float dd = amax > 0.0F ? amax / 127.0F : 1.0F;
                  int qi = static_cast<int>(rintf(v[c] / dd));
                  qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
                  std::uint8_t* blk =
                      attention.keys_q8 +
                      (std::size_t(*attention.position + tok) * 16 + head * 8 +
                       c) *
                          34;
                  blk[lane_id] = static_cast<std::uint8_t>(qi);
                  if (lane_id == 0)
                    *reinterpret_cast<__half*>(blk + 32) = __float2half(dd);
                }
              } else {
                attention
                    .keys[std::size_t(*attention.position + tok) * kvwidth +
                          head * dim + col] = __float2half_rn(v[c]);
              }
            } else {
              // Compile-time Q8/F16 split: the F16 arm is the untouched
              // pre-Q8 body, so its codegen (VGPR/occupancy) cannot drift
              // from main. Only AttentionQ8Gemm instantiates kQ8Out.
              if constexpr (kQ8Out) {
                if (attention.values_q8 != nullptr) {
                  // Fused Q8_0: the wave's lanes hold block c (lane l = elem
                  // l), so the per-block max is a wave-local shuffle reduction.
                  float amax = fabsf(v[c]);
                  amax = fmaxf(amax, XorLane(amax, 16));
                  amax = fmaxf(amax, XorLane(amax, 8));
                  amax = fmaxf(amax, XorLane(amax, 4));
                  amax = fmaxf(amax, XorLane(amax, 2));
                  amax = fmaxf(amax, XorLane(amax, 1));
                  const float dd = amax > 0.0F ? amax / 127.0F : 1.0F;
                  int qi = static_cast<int>(rintf(v[c] / dd));
                  qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
                  std::uint8_t* blk =
                      attention.values_q8 +
                      (std::size_t(*attention.position + tok) * 16 + head * 8 +
                       c) *
                          34;
                  blk[lane_id] = static_cast<std::uint8_t>(qi);
                  if (lane_id == 0)
                    *reinterpret_cast<__half*>(blk + 32) = __float2half(dd);
                }
              } else {
                attention
                    .values[std::size_t(*attention.position + tok) * kvwidth +
                            head * dim + col] = __float2half_rn(v[c]);
              }
            }
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  if constexpr (kHcMix) {
    static_assert(BM == 256 && BN == 128 && BK == 1 && WM == 4 && WN == 2);
    constexpr unsigned kHiddenTile = BM / 4;
    constexpr unsigned kStreamStride = kHiddenTile + 1;
    constexpr unsigned kPlane = 16 * kStreamStride;
    static_assert(4 * kPlane * sizeof(float) <= sizeof(s_lds));
    float* gates = reinterpret_cast<float*>(s_lds);
    const std::size_t hidden = m / 4;
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int token_group = 0; token_group < WN; ++token_group) {
        if (wave_tok == token_group) {
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
              const unsigned row =
                  (wave_row * kWaveRowTiles + i) * 16 + 2 * l + half_id;
              gates[(row % 4) * kPlane + sub_lane * kStreamStride + row / 4] =
                  acc[i][j][l];
            }
          }
        }
        __syncthreads();
        // A half-wave writes all 64 hidden values of one token. The stream
        // planes are padded to avoid the gate transpose's LDS bank conflicts.
        const unsigned token_in_tile = tid / (kHiddenTile / 4);
        const unsigned h_local = (tid % (kHiddenTile / 4)) * 4;
        const std::size_t token =
            t_block + (token_group * kWaveTokTiles + j) * 16 + token_in_tile;
        const std::size_t h = r_block / 4 + h_local;
        if (token < batch) {
          float4 value{0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
          for (unsigned stream = 0; stream < 4; ++stream) {
            const float* g = gates + stream * kPlane +
                             token_in_tile * kStreamStride + h_local;
            const float4 v = Load4(xn + token * m + stream * hidden + h);
            // Preserve the separate mixer's F32 FMA rounding.
            value.x = __fmaf_rn(v.x, SigmoidF(g[0]), value.x);
            value.y = __fmaf_rn(v.y, SigmoidF(g[1]), value.y);
            value.z = __fmaf_rn(v.z, SigmoidF(g[2]), value.z);
            value.w = __fmaf_rn(v.w, SigmoidF(g[3]), value.w);
          }
          value.x *= 0.25F;
          value.y *= 0.25F;
          value.z *= 0.25F;
          value.w *= 0.25F;
          if (y != nullptr)
            *reinterpret_cast<float4*>(y + token * hidden + h) = value;
          if (mixed_half != nullptr) {
            *reinterpret_cast<__half2*>(mixed_half + token * hidden + h) =
                __floats2half2_rn(value.x, value.y);
            *reinterpret_cast<__half2*>(mixed_half + token * hidden + h + 2) =
                __floats2half2_rn(value.z, value.w);
          }
          if (mixed_q8 != nullptr) {
            float max_abs = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                  fmaxf(fabsf(value.z), fabsf(value.w)));
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
              max_abs = fmaxf(max_abs, XorLane(max_abs, off));
            }
            const float d = max_abs / 127.0F;
            const float id = d != 0.0F ? 1.0F / d : 0.0F;
            const auto q0 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.x * id))));
            const auto q1 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.y * id))));
            const auto q2 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.z * id))));
            const auto q3 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.w * id))));
            const unsigned pos = h % 32;
            auto* tile = Q8ActTile(mixed_q8, hidden / 32,
                                   token / kQ8ActTileTokens, h / 32);
            const std::size_t tl = token % kQ8ActTileTokens;
            *reinterpret_cast<unsigned*>(tile + (pos >> 4) * 256 + tl * 16 +
                                         (pos & 15)) =
                q0 | q1 << 8 | q2 << 16 | q3 << 24;
            if (pos == 0) {
              *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                        tl * sizeof(float)) = d;
            }
          }
        }
        __syncthreads();
      }
    }
    return;
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  // Four padding floats reduce scatter bank conflicts and retain float4
  // alignment for both output stores and the fused convolution's reads.
  constexpr unsigned kOutputStride = 36;
  // Pair token tiles for SSM: half as many convolutions cross a tile edge.
  // The larger transpose still fits the projection's existing LDS allocation.
  constexpr unsigned kOutputTokens = kSsmConv ? kSsmProjectionTileTokens : 16;
  constexpr unsigned kOutputGroups = kOutputTokens / 16;
  static_assert(8 * kOutputTokens * kOutputStride * sizeof(float) <=
                sizeof(s_lds));
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  float* tile_scratch =
      reinterpret_cast<float*>(s_lds) + wave_id * kOutputTokens * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; j += kOutputGroups) {
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // scratch[token][row], with 32 output rows per token.
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + (2 * l) +
                       half_id] = acc[i][j + group][l];
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + 16 +
                       (2 * l) + half_id] = acc[i + 1][j + group][l];
        }
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          static_cast<std::size_t>(r_block) +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 = static_cast<std::size_t>(
          token_of(((wave_tok * kWaveTokTiles) + j) * 16));
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
        const int tok_l = (lane_id >> 1) + group * 16;
        const int row_l = (lane_id & 1) * 16;
        const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
        const auto* src = reinterpret_cast<const float4*>(
            tile_scratch + (tok_l * kOutputStride) + row_l);
        if constexpr (kSsmConv) {
          static_assert(BM == 256 && BN == 128 && BK == 2 && WM == 8 &&
                        WN == 1);
          static_assert(!kHcMix);
          constexpr std::uint32_t channels = 10240;
          if (tok < batch) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              const unsigned row = r0 + row_l + v * 4;
              if (row >= m)
                continue;
              const float4 current = src[v];
              if (row >= channels || tok_l < 3 || tok_l >= kOutputTokens - 3 ||
                  tok + 3 >= batch ||
                  (tok < checkpoint_tokens && tok + 3 >= checkpoint_tokens)) {
                *reinterpret_cast<float4*>(y + tok * m + row) = current;
              }
              if (row < channels && tok_l >= 3) {
                const float4 x0 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 3) * kOutputStride + row_l + v * 4);
                const float4 x1 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 2) * kOutputStride + row_l + v * 4);
                const float4 x2 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 1) * kOutputStride + row_l + v * 4);
                const float4 w0 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 0) * 4);
                const float4 w1 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 1) * 4);
                const float4 w2 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 2) * 4);
                const float4 w3 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 3) * 4);
                const float4 value{
                    SsmConv4Value(w0, x0.x, x1.x, x2.x, current.x),
                    SsmConv4Value(w1, x0.y, x1.y, x2.y, current.y),
                    SsmConv4Value(w2, x0.z, x1.z, x2.z, current.z),
                    SsmConv4Value(w3, x0.w, x1.w, x2.w, current.w)};
                *reinterpret_cast<float4*>(conv_out + tok * channels + row) =
                    value;
              }
            }
          }
        } else {
          if (tok < batch && r0 + 32 <= m && ((tok * m) % 4 == 0)) {
            auto* dst = reinterpret_cast<float4*>(
                y + (tok * m) + r0 + static_cast<std::size_t>(row_l));
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              dst[q] = src[q];
            }
          } else if (tok < batch) {
#pragma unroll
            for (int q = 0; q < 16; ++q) {
              const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
              if (r < m) {
                y[(tok * m) + r] =
                    tile_scratch[(tok_l * kOutputStride) + row_l + q];
              }
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

}  // namespace

namespace {

__global__ void ConvertF32ToF16Kernel(const float* __restrict__ in,
                                      __half* __restrict__ out,
                                      std::size_t n) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) *
                              static_cast<std::size_t>(blockDim.x) +
                          threadIdx.x;
  if (idx < n) {
    out[idx] = __float2half(in[idx]);
  }
}

}  // namespace

bool ConvertF32ToF16(const float* in, __half* out, std::size_t n,
                     hipStream_t stream) {
  if (n == 0) {
    return true;
  }
  if (in == nullptr || out == nullptr) {
    return false;
  }
  constexpr unsigned kBlockSize = 256;
  const unsigned grid =
      static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize);
  hipLaunchKernelGGL(ConvertF32ToF16Kernel, dim3(grid), dim3(kBlockSize), 0,
                     stream, in, out, n);
  return true;
}

bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream) {
  if (m == 0 || batch == 0 || k == 0 || k % 32 != 0)
    return false;
  // Wide batches reuse each staged fragment across twice the rows and
  // tokens (router 0.61 to 0.52 ms, GDN alpha/beta 0.14 to 0.09 ms at 4,096
  // tokens). Every output keeps its two K16 chains in K order.
  if (batch >= 1024) {
    hipLaunchKernelGGL(
        (DenseF16GEMMKernel<128, 128, 2, 2, 4, 1, false, false, false, true>),
        dim3((batch + 127) / 128, (m + 127) / 128), dim3(kThreads), 0, stream,
        w, x, out, batch, m, k);
    return true;
  }
  hipLaunchKernelGGL(
      (DenseF16GEMMKernel<64, 64, 2, 2, 4, 1, false, false, false, true>),
      dim3((batch + 63) / 64, (m + 63) / 64), dim3(kThreads), 0, stream, w, x,
      out, batch, m, k);
  return true;
}

bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream,
                  std::size_t input_stride) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  if (input_stride != 0 && input_stride != k) {
    if (batch < 1024 || m != 2560 || k != 6144 || input_stride < k ||
        input_stride % 8 != 0)
      return false;
    hipLaunchKernelGGL((DenseF16GEMMKernel<256, 128, 2, 8, 1, 5, false, false,
                                           false, false, false, true>),
                       dim3((batch + 127) / 128, (m + 255) / 256),
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k,
                       nullptr, nullptr, nullptr, nullptr, nullptr,
                       AttentionProjectionOutput{}, 0, input_stride);
    return true;
  }
  // The same tile plan as the W8A8 route: a 128-token macro tile for wide
  // batches, 64-row tiles for narrow projections, 64 tokens below 96.
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kNarrowBM, kBN, 2, 2, 4>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  } else if (batch >= 96) {
    // 64 x 64 wave tiles: half the LDS fragment bytes per matrix product
    // of the 32 x 64 tile (the F16 fragments are twice the int8 ones).
    constexpr int kWideBM = 256;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kWideBM - 1) / kWideBM));
    if (m == 10240 && k == 320 && batch >= 1024) {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2, 8>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else if (batch >= 1024 && m == 2560 && k == 6144) {
      // The output projection's activation is 50 MB at 4,096 tokens. In
      // grid order the resident blocks each walk a different token tile of
      // it; walking five row tiles of a token tile first shares each
      // activation tile between them (5.7 to 3.4 ms). Only the launch order
      // changes.
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 2, 8, 1, 5>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else if (batch >= 1024 && (m == 16384 || m == 13312) && k == 2560) {
      // Eight row groups reuse each weight fragment across all token tiles
      // and keep fewer weight fragments live. K accumulation is unchanged.
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 2, 8, 1>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else if (batch >= 1024 && m == 2560) {
      // Shared-expert down: pairs of row tiles per token tile (-11 %).
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2, 2>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    }
  } else {
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kBM, kBN, 4, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  }
  return true;
}

}  // namespace gufo::core::compute::gemm

#endif  // ENGINE_ENABLE_HIP
