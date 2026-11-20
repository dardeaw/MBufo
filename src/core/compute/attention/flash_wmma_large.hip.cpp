// WMMA flash-attention prefill for LARGE head dimensions (256 and 512), any GQA ratio (MHA is ratio 1), F16 KV, causal with
// an optional sliding window. Same S^T = K Q^T register-resident FlashAttention-2 form as flash_wmma_gqa (head_dim 64/128;
// see there for the layout), organised for the register and LDS budget of wide heads:
//
//   * a head row of d = 256 / 512 does not fit one wave's registers (the Q fragments plus the O accumulator alone need
//     2 * d / 2 VGPRs), so every (head, 16-query block) "group" is served by kSlices = d / 128 waves, each owning 1/kSlices
//     of the head dimension: its slice of the QK reduction (partial S) and its slice of the output O;
//   * the partial S tiles go through LDS (one barrier per key tile), every wave of a group sums them in slice order (so all
//     waves hold bit-identical S), runs the online softmax redundantly (a few hundred FLOP against ~100k for the matrix
//     work) and applies PV to its own dim slice;
//   * the K / V^T tiles in LDS are shared by every group of the block: kHeadWaves heads of one KV group x kQBlocks
//     consecutive 16-query blocks.
// A key tile is 32 keys at d = 256 and 16 keys at d = 512 (LDS budget).
// Math contract (same as flash_wmma_gqa): the QK scale is folded into the F16 Q fragments, F32 accumulation in both WMMA
// passes, online softmax identical to a full-row pass up to summation order, P rounded to F16 for the PV product.
#if defined(ENGINE_ENABLE_HIP)
#include "src/core/compute/attention/flash_wmma_large.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mbufo::core::compute::flash_wmma_large {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using v8u = __attribute__((__vector_size__(8 * sizeof(unsigned)))) unsigned;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

__device__ __forceinline__ unsigned PackHalves(float a, float b) {
  return __builtin_bit_cast(unsigned, __floats2half2_rn(a, b));
}

constexpr std::uint32_t kRows = 16;  // queries per group (one WMMA column block)

// One 8-element chunk (c8 .. c8 + 7, c8 a multiple of 8) of a blocked-Q8 key / value row of one KV head, dequantized to
// four packed half2 words: block b = c8 / 32 holds 32 int8 codes followed by an F16 scale (34 bytes, 2-byte aligned).
__device__ __forceinline__ uint4 DequantChunk8(const std::uint8_t* head_row, std::uint32_t c8) {
  const std::uint8_t* blk = head_row + static_cast<std::size_t>(c8 / 32u) * 34u;
  const float sc = __half2float(*reinterpret_cast<const __half*>(blk + 32));
  const unsigned short* p = reinterpret_cast<const unsigned short*>(blk + (c8 % 32u));
  unsigned wd[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned u = p[i];
    wd[i] = PackHalves(static_cast<float>(static_cast<std::int8_t>(u & 0xFFu)) * sc,
                       static_cast<float>(static_cast<std::int8_t>(u >> 8)) * sc);
  }
  return make_uint4(wd[0], wd[1], wd[2], wd[3]);
}

template <std::uint32_t kHeadDim, std::uint32_t kHeadWaves, std::uint32_t kQBlocks, bool kQ8>
__global__ __launch_bounds__(32 * (kHeadDim / 128) * kHeadWaves * kQBlocks) void FlashWmmaLargeKernel(
    const float* __restrict__ q, const void* __restrict__ k_raw, const void* __restrict__ v_raw,
    float* __restrict__ out, std::uint32_t start_pos, std::uint32_t n_tokens, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t window, float scale) {
  constexpr std::uint32_t kSlices = kHeadDim / 128;                 // waves per group, each owning 128 dims
  constexpr std::uint32_t kGroups = kHeadWaves * kQBlocks;          // (head, query block) pairs of the block
  constexpr std::uint32_t kWaves = kGroups * kSlices;
  constexpr std::uint32_t kThreads = 32 * kWaves;
  constexpr std::uint32_t kKeys = kHeadDim == 256 ? 32 : 16;        // keys per K/V tile
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kKSteps = kHeadDim / 16;                  // WMMA k-steps of the full QK reduction
  constexpr std::uint32_t kSliceKSteps = kKSteps / kSlices;         // this wave's share (8)
  constexpr std::uint32_t kSliceDimTiles = (kHeadDim / 16) / kSlices;  // output dim tiles of this wave (8)
  constexpr std::uint32_t kKStride = kHeadDim + 8;                  // K tile row stride (halves)
  constexpr std::uint32_t kVtStride = kKeys + 8;                    // V^T tile row stride (halves)
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;
  const std::uint32_t slice = wave % kSlices;
  const std::uint32_t group = wave / kSlices;
  const std::uint32_t head = blockIdx.y * kHeadWaves + (group % kHeadWaves);
  const std::uint32_t qblk = group / kHeadWaves;
  const std::uint32_t qpg = heads / kv_heads;
  const std::uint32_t kv_head = (blockIdx.y * kHeadWaves) / qpg;  // all kHeadWaves heads share one KV head
  const std::uint32_t block_q0 = blockIdx.x * kQBlocks * kRows;
  const std::uint32_t query_start = block_q0 + qblk * kRows;
  const std::size_t kv_width = static_cast<std::size_t>(kv_heads) * kHeadDim;
  const __half* k_cache = static_cast<const __half*>(k_raw);
  const __half* v_cache = static_cast<const __half*>(v_raw);
  const std::uint8_t* k8 = static_cast<const std::uint8_t*>(k_raw);
  const std::uint8_t* v8 = static_cast<const std::uint8_t*>(v_raw);
  const std::size_t q8_row_bytes = static_cast<std::size_t>(kv_heads) * (kHeadDim / 32) * 34;  // blocked-Q8 key row
  const std::size_t q8_head_off = static_cast<std::size_t>(kv_head) * (kHeadDim / 32) * 34;

  __shared__ __half k_lds[kKeys][kKStride];
  __shared__ __half vt_lds[kHeadDim][kVtStride];
  __shared__ float s_lds[kWaves][kKeyBlocks][32][8];  // partial S of every wave

  const std::uint32_t qrel = query_start + sub;
  const bool row_live = qrel < n_tokens;
  const std::uint32_t qpos = start_pos + qrel;

  // Q fragments of this wave's K-steps (B operand of S^T = K Q^T); the score scale is folded in.
  v16h q_frag[kSliceKSteps];
  {
    const float* qrow = q + (static_cast<std::size_t>(row_live ? qrel : 0) * heads + head) * kHeadDim;
#pragma unroll
    for (std::uint32_t ks = 0; ks < kSliceKSteps; ++ks) {
#pragma unroll
      for (std::uint32_t e = 0; e < 16; ++e) {
        q_frag[ks][e] = static_cast<_Float16>(row_live ? qrow[(slice * kSliceKSteps + ks) * 16u + e] * scale : 0.0F);
      }
    }
  }

  v8f o_acc[kSliceDimTiles] = {};
  float m_run = -INFINITY;
  float l_run = 0.0F;

  const std::uint32_t n_keys_all = start_pos + n_tokens;
  std::uint32_t key_hi = start_pos + block_q0 + kQBlocks * kRows;
  if (key_hi > n_keys_all) key_hi = n_keys_all;
  const std::uint32_t first_q_pos = start_pos + block_q0;
  const std::uint32_t key_lo = (window != 0 && first_q_pos + 1 > window) ? first_q_pos + 1 - window : 0;
  const std::uint32_t kb_begin = key_lo / kKeys;
  const std::uint32_t kb_end = (key_hi + kKeys - 1u) / kKeys;

  for (std::uint32_t kb = kb_begin; kb < kb_end; ++kb) {
    const std::uint32_t key_start = kb * kKeys;
    __syncthreads();  // previous tile's K / V^T / partial-S reads are done
    // Cooperative tile load: K row-major (16 B per thread per step); V transposed through registers (8 keys x 8 dims blocks)
    // into vt_lds[dim][pos(key)] with the even-keys-first permutation inside each 16-key group (see flash_wmma_gqa).
    for (std::uint32_t i = tid; i < kKeys * (kHeadDim / 8); i += kThreads) {
      const std::uint32_t r = i / (kHeadDim / 8);
      const std::uint32_t c8 = (i % (kHeadDim / 8)) * 8u;
      const std::uint32_t kr = key_start + r;
      uint4 kv = make_uint4(0, 0, 0, 0);
      if (kr < key_hi) {
        if constexpr (kQ8) {
          kv = DequantChunk8(k8 + static_cast<std::size_t>(kr) * q8_row_bytes + q8_head_off, c8);
        } else {
          kv = *reinterpret_cast<const uint4*>(k_cache + static_cast<std::size_t>(kr) * kv_width +
                                               static_cast<std::size_t>(kv_head) * kHeadDim + c8);
        }
      }
      *reinterpret_cast<uint4*>(&k_lds[r][c8]) = kv;
    }
    for (std::uint32_t u = tid; u < (kKeys / 8) * (kHeadDim / 8); u += kThreads) {
      const std::uint32_t kg = u % (kKeys / 8);
      const std::uint32_t c8 = (u / (kKeys / 8)) * 8u;
      unsigned w[8][4];
#pragma unroll
      for (std::uint32_t j = 0; j < 8; ++j) {
        const std::uint32_t kr = key_start + kg * 8u + j;
        uint4 vv = make_uint4(0, 0, 0, 0);
        if (kr < key_hi) {
          if constexpr (kQ8) {
            vv = DequantChunk8(v8 + static_cast<std::size_t>(kr) * q8_row_bytes + q8_head_off, c8);
          } else {
            vv = *reinterpret_cast<const uint4*>(v_cache + static_cast<std::size_t>(kr) * kv_width +
                                                 static_cast<std::size_t>(kv_head) * kHeadDim + c8);
          }
        }
        w[j][0] = vv.x; w[j][1] = vv.y; w[j][2] = vv.z; w[j][3] = vv.w;
      }
      const std::uint32_t col = (kg >> 1) * 16u + (kg & 1u) * 4u;
#pragma unroll
      for (std::uint32_t e = 0; e < 8; ++e) {
        const std::uint32_t sh = (e & 1u) * 16u;
        const std::uint32_t wi = e >> 1;
        auto H = [&](std::uint32_t j) { return (w[j][wi] >> sh) & 0xffffu; };
        const uint2 ev = make_uint2(H(0) | (H(2) << 16), H(4) | (H(6) << 16));
        const uint2 od = make_uint2(H(1) | (H(3) << 16), H(5) | (H(7) << 16));
        *reinterpret_cast<uint2*>(&vt_lds[c8 + e][col]) = ev;
        *reinterpret_cast<uint2*>(&vt_lds[c8 + e][col + 8u]) = od;
      }
    }
    __syncthreads();

    // Partial S^T over this wave's slice of the head dimension: C[key = 2i + half][query = sub].
    v8f s_part[kKeyBlocks] = {};
#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceKSteps; ++ks) {
        const v16h kf = LoadFrag(&k_lds[kbk * 16u + sub][(slice * kSliceKSteps + ks) * 16u]);
        s_part[kbk] = Wmma(kf, q_frag[ks], s_part[kbk]);
      }
    }
    if constexpr (kSlices > 1) {
#pragma unroll
      for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk)
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) s_lds[wave][kbk][lane][i] = s_part[kbk][i];
      __syncthreads();
    }
    // Full S: the slices of this group summed in slice order (every wave of the group computes the same value).
    v8f s_acc[kKeyBlocks];
#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
      if constexpr (kSlices > 1) {
        v8f acc = {};
#pragma unroll
        for (std::uint32_t sl = 0; sl < kSlices; ++sl) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) acc[i] += s_lds[group * kSlices + sl][kbk][lane][i];
        }
        s_acc[kbk] = acc;
      } else {
        s_acc[kbk] = s_part[kbk];
      }
    }

    // Mask + online softmax (this lane's scores belong to ONE query; its partner lane holds the other keys).
    const std::uint32_t wave_qpos_min = start_pos + query_start;
    const bool full = key_start + kKeys <= key_hi && key_start + kKeys - 1u <= wave_qpos_min &&
                      (window == 0 || key_start + window > wave_qpos_min + kRows - 1u);
    float part_max = -INFINITY;
    if (!full) {
#pragma unroll
      for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk)
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          const std::uint32_t kc = key_start + kbk * 16u + 2u * i + half_id;
          const bool valid = row_live && kc < key_hi && kc <= qpos && (window == 0 || kc + window > qpos);
          s_acc[kbk][i] = valid ? s_acc[kbk][i] : -INFINITY;
        }
    }
#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk)
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) part_max = fmaxf(part_max, s_acc[kbk][i]);
    part_max = fmaxf(part_max, __shfl_xor(part_max, 16));
    const float m_new = fmaxf(m_run, part_max);
    const float alpha = isfinite(m_run) ? __expf(m_run - m_new) : 0.0F;
    m_run = m_new;
    float psum = 0.0F;
    unsigned own[kKeyBlocks][4];
#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
      float pw[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        pw[i] = isfinite(s_acc[kbk][i]) ? __expf(s_acc[kbk][i] - m_new) : 0.0F;
        psum += pw[i];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < 4; ++t) own[kbk][t] = PackHalves(pw[2 * t], pw[2 * t + 1]);
    }
    l_run = l_run * alpha + psum;
#pragma unroll
    for (std::uint32_t dt = 0; dt < kSliceDimTiles; ++dt) o_acc[dt] *= alpha;

    // O^T += V^T P^T for this wave's dim slice: B operand per key block = [even keys | odd keys] of this lane's query.
#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
      v8u bu;
#pragma unroll
      for (std::uint32_t t = 0; t < 4; ++t) {
        const unsigned other = static_cast<unsigned>(__shfl_xor(static_cast<int>(own[kbk][t]), 16));
        bu[t] = half_id == 0 ? own[kbk][t] : other;
        bu[4 + t] = half_id == 0 ? other : own[kbk][t];
      }
      const v16h pb = __builtin_bit_cast(v16h, bu);
#pragma unroll
      for (std::uint32_t dt = 0; dt < kSliceDimTiles; ++dt) {
        const v16h vf = LoadFrag(&vt_lds[(slice * kSliceDimTiles + dt) * 16u + sub][kbk * 16u]);
        o_acc[dt] = Wmma(vf, pb, o_acc[dt]);
      }
    }
  }

  const float l_tot = l_run + __shfl_xor(l_run, 16);
  const float inv = l_tot > 0.0F ? 1.0F / l_tot : 0.0F;
  if (row_live) {
    float* orow = out + (static_cast<std::size_t>(qrel) * heads + head) * kHeadDim;
#pragma unroll
    for (std::uint32_t dt = 0; dt < kSliceDimTiles; ++dt)
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) orow[(slice * kSliceDimTiles + dt) * 16u + 2u * i + half_id] = o_acc[dt][i] * inv;
  }
}

template <std::uint32_t kHeadDim, std::uint32_t kHeadWaves, std::uint32_t kQBlocks, bool kQ8>
static void LaunchOne(const float* q, const void* k, const void* v, float* out, std::uint32_t n_tokens,
                      std::uint32_t start_pos, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t window,
                      float scale, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 32 * (kHeadDim / 128) * kHeadWaves * kQBlocks;
  const dim3 grid((n_tokens + kRows * kQBlocks - 1) / (kRows * kQBlocks), heads / kHeadWaves);
  hipLaunchKernelGGL((FlashWmmaLargeKernel<kHeadDim, kHeadWaves, kQBlocks, kQ8>), grid, dim3(kThreads), 0, stream, q, k, v, out,
                     start_pos, n_tokens, heads, kv_heads, window, scale);
}

template <std::uint32_t kHeadDim, bool kQ8>
static void Launch(const float* q, const void* k, const void* v, float* out, std::uint32_t n_tokens,
                   std::uint32_t start_pos, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t window, float scale,
                   hipStream_t stream) {
  const std::uint32_t qpg = heads / kv_heads;
  // 4 groups per block at d = 256 (8 waves), 2 at d = 512 (8 waves): heads of one KV group first, then query blocks.
  constexpr std::uint32_t kGroupsPerBlock = kHeadDim == 256 ? 4 : 2;
  if constexpr (kGroupsPerBlock == 4) {
    if (qpg % 4 == 0) LaunchOne<kHeadDim, 4, 1, kQ8>(q, k, v, out, n_tokens, start_pos, heads, kv_heads, window, scale, stream);
    else if (qpg % 2 == 0) LaunchOne<kHeadDim, 2, 2, kQ8>(q, k, v, out, n_tokens, start_pos, heads, kv_heads, window, scale, stream);
    else LaunchOne<kHeadDim, 1, 4, kQ8>(q, k, v, out, n_tokens, start_pos, heads, kv_heads, window, scale, stream);
  } else {
    if (qpg % 2 == 0) LaunchOne<kHeadDim, 2, 1, kQ8>(q, k, v, out, n_tokens, start_pos, heads, kv_heads, window, scale, stream);
    else LaunchOne<kHeadDim, 1, 2, kQ8>(q, k, v, out, n_tokens, start_pos, heads, kv_heads, window, scale, stream);
  }
}

}  // namespace

bool FlashWmmaLargePrefill(const float* q, const __half* k_cache, const __half* v_cache, float* out, std::uint32_t n_tokens,
                           std::uint32_t start_pos, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t head_dim,
                           std::uint32_t window, float score_scale, hipStream_t stream) {
  if (!IsFlashWmmaLargeSupported(heads, kv_heads, head_dim) || q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      out == nullptr || n_tokens == 0) {
    return false;
  }
  if (head_dim == 256)
    Launch<256, false>(q, k_cache, v_cache, out, n_tokens, start_pos, heads, kv_heads, window, score_scale, stream);
  else
    Launch<512, false>(q, k_cache, v_cache, out, n_tokens, start_pos, heads, kv_heads, window, score_scale, stream);
  return true;
}

bool FlashWmmaLargePrefillQ8(const float* q, const std::uint8_t* k_blocks, const std::uint8_t* v_blocks, float* out,
                             std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t heads, std::uint32_t kv_heads,
                             std::uint32_t head_dim, std::uint32_t window, float score_scale, hipStream_t stream) {
  if (!IsFlashWmmaLargeSupported(heads, kv_heads, head_dim) || q == nullptr || k_blocks == nullptr || v_blocks == nullptr ||
      out == nullptr || n_tokens == 0) {
    return false;
  }
  if (head_dim == 256)
    Launch<256, true>(q, k_blocks, v_blocks, out, n_tokens, start_pos, heads, kv_heads, window, score_scale, stream);
  else
    Launch<512, true>(q, k_blocks, v_blocks, out, n_tokens, start_pos, heads, kv_heads, window, score_scale, stream);
  return true;
}

}  // namespace mbufo::core::compute::flash_wmma_large
#endif  // ENGINE_ENABLE_HIP
