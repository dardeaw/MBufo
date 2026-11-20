// WMMA flash-attention prefill for head_dim 64 or 128, any even GQA ratio, F16 KV, causal with an optional sliding
// window and optional per-head attention sinks.
//
// v2 data flow ("S transposed", the register-resident FlashAttention-2 form for the RDNA WMMA layout):
//   S^T[key][query] = K Q^T   (A operand = K tile fragments, B operand = Q fragments)
// puts each lane in charge of ONE query column (two lanes per query, holding complementary keys), so
//   * the online-softmax state (max m, denominator l) is one scalar per lane and the row max needs a single
//     cross-half exchange instead of four shuffles per row;
//   * the probabilities are already in registers in (nearly) the layout the PV product wants as its B operand: one
//     16-lane-half swap turns "keys of my parity" into all 16 keys of a block, so P never goes through LDS (the old
//     version round-tripped P through LDS with two extra barriers per tile);
//   O^T[dim][query] = V^T P^T   (A operand = V^T tile fragments, B operand = the P registers).
// Key order inside a 16-key WMMA block is a free permutation as long as both operands use it: B holds the block's
// even keys first, then the odd keys, and V^T is stored in LDS with the same column permutation.
// Block = kWaves waves = kWaves query heads of one KV group sharing the K/V tile; 16 queries per wave.
// Math contract: QK scale folded into the F16 Q fragments; F32 accumulate in both WMMA passes; online softmax
// identical to a full-row pass up to summation order; P rounded to F16 for the PV product.
#if defined(ENGINE_ENABLE_HIP)
#include "src/core/compute/attention/flash_wmma_gqa.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mbufo::core::compute::flash_wmma_gqa {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using v8u = __attribute__((__vector_size__(8 * sizeof(unsigned)))) unsigned;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// A fragment is 16 contiguous halves = 32 bytes (two 16-byte loads); every fragment base here is 16-byte aligned.
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

constexpr std::uint32_t kRows = 16;               // queries per wave (one WMMA column block)
constexpr std::uint32_t kKeys = 64;               // keys per K/V tile
constexpr std::uint32_t kKeyBlocks = kKeys / 16;  // 4 key blocks per tile
constexpr std::uint32_t kVtStride = kKeys + 8;    // V^T tile row stride (halves)
constexpr std::uint32_t QB = 2;  // 16-query blocks per block: 2x the queries share every K/V tile load

template <std::uint32_t kHeadDim, std::uint32_t kHeadWaves, std::uint32_t kQBlocks>
__global__ __launch_bounds__(32 * kHeadWaves * kQBlocks) void FlashWmmaKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    const float* __restrict__ sinks, float* __restrict__ out, std::uint32_t start_pos, std::uint32_t n_tokens,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t window, float scale) {
  constexpr std::uint32_t kWaves = kHeadWaves * kQBlocks;  // waves: kHeadWaves heads x kQBlocks query blocks
  constexpr std::uint32_t kThreads = 32 * kWaves;
  constexpr std::uint32_t kKSteps = kHeadDim / 16;    // WMMA k-steps for QK^T
  constexpr std::uint32_t kDimTiles = kHeadDim / 16;  // output dim tiles for PV
  constexpr std::uint32_t kKStride = kHeadDim + 8;    // K tile row stride (halves): 16B-aligned fragment reads
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;
  const std::uint32_t head = blockIdx.y * kHeadWaves + (wave % kHeadWaves);
  const std::uint32_t qblk = wave / kHeadWaves;
  const std::uint32_t qpg = heads / kv_heads;
  const std::uint32_t kv_head = (blockIdx.y * kHeadWaves) / qpg;  // all kHeadWaves heads share one KV head
  const std::uint32_t block_q0 = blockIdx.x * kQBlocks * kRows;  // first query of the block
  const std::uint32_t query_start = block_q0 + qblk * kRows;     // first query of this wave
  const std::size_t kv_width = static_cast<std::size_t>(kv_heads) * kHeadDim;

  __shared__ __half k_lds[kKeys][kKStride];
  __shared__ __half vt_lds[kHeadDim][kVtStride];

  // This lane's query column.
  const std::uint32_t qrel = query_start + sub;
  const bool row_live = qrel < n_tokens;
  const std::uint32_t qpos = start_pos + qrel;

  // Q fragments (B operand of S^T = K Q^T): lane owns query `sub`; F32 -> F16 with the score scale folded in.
  v16h q_frag[kKSteps];
  {
    const float* qrow = q + (static_cast<std::size_t>(row_live ? qrel : 0) * heads + head) * kHeadDim;
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKSteps; ++ks) {
#pragma unroll
      for (std::uint32_t e = 0; e < 16; ++e) {
        q_frag[ks][e] = static_cast<_Float16>(row_live ? qrow[ks * 16u + e] * scale : 0.0F);
      }
    }
  }

  v8f o_acc[kDimTiles] = {};  // O^T[dim = 2i + half within tile][query = sub]
  // The sink is one extra softmax term (value zero): max starts at its raw logit, the denominator at 1 (held by
  // one lane of the pair; the two partial sums are added in the epilogue).
  float m_run = sinks != nullptr ? sinks[head] : -INFINITY;
  float l_run = (sinks != nullptr && half_id == 0) ? 1.0F : 0.0F;

  // Key range this block can see: keys <= its last query (causal), and with a window only from the first key
  // its first query still reaches.
  const std::uint32_t n_keys_all = start_pos + n_tokens;
  std::uint32_t key_hi = start_pos + block_q0 + kQBlocks * kRows;
  if (key_hi > n_keys_all) key_hi = n_keys_all;
  const std::uint32_t first_q_pos = start_pos + block_q0;
  const std::uint32_t key_lo = (window != 0 && first_q_pos + 1 > window) ? first_q_pos + 1 - window : 0;
  const std::uint32_t kb_begin = key_lo / kKeys;
  const std::uint32_t kb_end = (key_hi + kKeys - 1u) / kKeys;

  for (std::uint32_t kb = kb_begin; kb < kb_end; ++kb) {
    const std::uint32_t key_start = kb * kKeys;
    __syncthreads();  // previous tile's K/V^T reads are done
    // Cooperative tile load. K: 16 B per thread per step, row-major. V: each thread owns 8 consecutive keys of one
    // 8-dim chunk, transposes the 8x8 half block in registers and writes vt_lds[dim][pos(key)] with 8-byte stores
    // (pos = (r & 1) * 8 + ((r & 15) >> 1) inside each 16-key group: even keys first, then odd keys).
    for (std::uint32_t i = tid; i < kKeys * (kHeadDim / 8); i += kThreads) {
      const std::uint32_t r = i / (kHeadDim / 8);
      const std::uint32_t c8 = (i % (kHeadDim / 8)) * 8u;
      const std::uint32_t kr = key_start + r;
      uint4 kv = make_uint4(0, 0, 0, 0);
      if (kr < key_hi)
        kv = *reinterpret_cast<const uint4*>(k_cache + static_cast<std::size_t>(kr) * kv_width +
                                             static_cast<std::size_t>(kv_head) * kHeadDim + c8);
      *reinterpret_cast<uint4*>(&k_lds[r][c8]) = kv;
    }
    for (std::uint32_t u = tid; u < (kKeys / 8) * (kHeadDim / 8); u += kThreads) {
      const std::uint32_t kg = u % (kKeys / 8);      // key group of 8 (fastest: lanes read 8 rows x 4 chunks)
      const std::uint32_t c8 = (u / (kKeys / 8)) * 8u;
      unsigned w[8][4];
#pragma unroll
      for (std::uint32_t j = 0; j < 8; ++j) {
        const std::uint32_t kr = key_start + kg * 8u + j;
        uint4 vv = make_uint4(0, 0, 0, 0);
        if (kr < key_hi)
          vv = *reinterpret_cast<const uint4*>(v_cache + static_cast<std::size_t>(kr) * kv_width +
                                               static_cast<std::size_t>(kv_head) * kHeadDim + c8);
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

    // S^T = K Q^T for 4 key blocks (16 keys each): C[key = 2i + half][query = sub].
    v8f s_acc[kKeyBlocks] = {};
#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKSteps; ++ks) {
        const v16h kf = LoadFrag(&k_lds[kbk * 16u + sub][ks * 16u]);
        s_acc[kbk] = Wmma(kf, q_frag[ks], s_acc[kbk]);
      }
    }

    // Mask + online softmax: this lane's 32 scores belong to ONE query; its partner lane (lane ^ 16) holds the other
    // keys of the same query.
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
    for (std::uint32_t dt = 0; dt < kDimTiles; ++dt) o_acc[dt] *= alpha;

    // O^T += V^T P^T: B operand per key block = [even keys 0..14 | odd keys 1..15] of this lane's query.
#pragma unroll
    for (std::uint32_t kbk = 0; kbk < kKeyBlocks; ++kbk) {
      v8u bu;
#pragma unroll
      for (std::uint32_t t = 0; t < 4; ++t) {
        const unsigned other = static_cast<unsigned>(__shfl_xor(static_cast<int>(own[kbk][t]), 16));
        bu[t] = half_id == 0 ? own[kbk][t] : other;       // even keys
        bu[4 + t] = half_id == 0 ? other : own[kbk][t];   // odd keys
      }
      const v16h pb = __builtin_bit_cast(v16h, bu);
#pragma unroll
      for (std::uint32_t dt = 0; dt < kDimTiles; ++dt) {
        const v16h vf = LoadFrag(&vt_lds[dt * 16u + sub][kbk * 16u]);
        o_acc[dt] = Wmma(vf, pb, o_acc[dt]);
      }
    }
  }

  // Epilogue: add the partner lane's partial denominator, normalize, write O[query][head][dim].
  const float l_tot = l_run + __shfl_xor(l_run, 16);
  const float inv = l_tot > 0.0F ? 1.0F / l_tot : 0.0F;
  if (row_live) {
    float* orow = out + (static_cast<std::size_t>(qrel) * heads + head) * kHeadDim;
#pragma unroll
    for (std::uint32_t dt = 0; dt < kDimTiles; ++dt)
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) orow[dt * 16u + 2u * i + half_id] = o_acc[dt][i] * inv;
  }
}

template <std::uint32_t kHeadDim>
static void Launch(const float* q, const __half* k, const __half* v, const float* sinks, float* out,
                   std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t heads, std::uint32_t kv_heads,
                   std::uint32_t window, float scale, hipStream_t stream) {
  const std::uint32_t qpg = heads / kv_heads;
  // A block = kHeadWaves query heads of one KV group x kQBlocks consecutive 16-query blocks, all sharing one K/V tile.
  if (qpg % 4 == 0) {
    const dim3 grid((n_tokens + kRows * QB - 1) / (kRows * QB), heads / 4);
    hipLaunchKernelGGL((FlashWmmaKernel<kHeadDim, 4, QB>), grid, dim3(32 * 4 * QB), 0, stream, q, k, v, sinks, out,
                       start_pos, n_tokens, heads, kv_heads, window, scale);
  } else {
    const dim3 grid((n_tokens + kRows * QB - 1) / (kRows * QB), heads / 2);
    hipLaunchKernelGGL((FlashWmmaKernel<kHeadDim, 2, QB>), grid, dim3(32 * 2 * QB), 0, stream, q, k, v, sinks, out,
                       start_pos, n_tokens, heads, kv_heads, window, scale);
  }
}

}  // namespace

bool FlashWmmaPrefill(const float* q, const __half* k_cache, const __half* v_cache, const float* sinks, float* out,
                      std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t heads, std::uint32_t kv_heads,
                      std::uint32_t head_dim, std::uint32_t window, float score_scale, hipStream_t stream) {
  if (!IsFlashWmmaSupported(heads, kv_heads, head_dim) || q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      out == nullptr || n_tokens == 0) {
    return false;
  }
  if (head_dim == 128)
    Launch<128>(q, k_cache, v_cache, sinks, out, n_tokens, start_pos, heads, kv_heads, window, score_scale, stream);
  else
    Launch<64>(q, k_cache, v_cache, sinks, out, n_tokens, start_pos, heads, kv_heads, window, score_scale, stream);
  return true;
}

}  // namespace mbufo::core::compute::flash_wmma_gqa

#endif  // defined(ENGINE_ENABLE_HIP)