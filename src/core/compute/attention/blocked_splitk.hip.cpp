// Blocked-Q8 Split-K decode attention (d == 256 SWA / d == 128 dense).
//
// d == 128 reuses the same 256-thread block: lanes i >= kHeadDim are
// predicated off (all 8 waves still participate in WaveSum so the shuffle
// reductions stay uniform), and the merge kernel early-returns them.
//
// The scalar AttentionBlocked path sweeps the whole visible KV extent from a
// single block per (head, token): 16 blocks each scanning up to the full
// context, which leaves most CUs idle on long-context decode. This kernel
// copies the F16 AttentionSplitKKernel split philosophy instead: the visible
// KV range is cut into K chunks, one block per (head, token, chunk) computes
// online-softmax partials (m, l, acc[d]) over its chunk with in-register
// int8 x scale dequant from the shared 34-byte blocked rows, and a merge
// kernel folds the partials in fixed chunk order.
//
// Cross-chunk folding reassociates the online-softmax rescaling, so scores
// may differ ~1ulp from the scalar single pass; the fold order is fixed so
// reruns are bitwise stable. The dequant math itself is identical (same
// codes/scales), so greedy decoding stays token-identical.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdlib>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/compute/attention/blocked_splitk.hpp"

namespace mbufo::core::compute::blocked_splitk {
namespace {

// Wave-shuffle helpers copied verbatim from blocked_kv.hip.cpp (same wave32
// DPP transport as the scalar kernel).
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

__device__ __forceinline__ float XorLane(float v, int offset) {
  return __builtin_bit_cast(float, XorLane(__builtin_bit_cast(int, v), offset));
}

/// Sum over one wave; every lane receives the total (wave32 on gfx1151).
__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += XorLane(v, offset);
  }
  return v;
}

/// Block-wide sum over kThreads threads; every thread receives the total.

constexpr unsigned kThreads = 256;

// Element decode comes from the unified codec trait (blocked_kv_codec.hpp)
// via Codec::Decode; do not reintroduce a TU-local CodecTraits.

// One (head, token, split) partial scan over keys [s*chunk, ...), clamped to
// the query's own causal extent (*start_pos + t + 1). Thread i owns dim i.
// Keys stream in batches of kBatch: per-key dots use a wave shuffle
// (WaveSum, no barrier) plus one block-wide exchange per batch, so a chunk
// costs two __syncthreads per kBatch keys instead of two per key. The
// per-batch fold order is fixed, so reruns are deterministic.
template <typename Codec, std::uint32_t kHeadDim>
__global__ void AttentionBlockedSplitKKernel(
    const float* __restrict__ q,
    const typename Codec::Elem* __restrict__ k_cache,
    const typename Codec::Elem* __restrict__ v_cache,
    float* __restrict__ partials, const std::uint32_t* __restrict__ start_pos,
    std::uint32_t heads, std::uint32_t kv_heads, float score_scale,
    std::uint32_t splits, std::uint32_t chunk, std::uint32_t key_block_bytes,
    std::uint32_t value_block_bytes) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t s = blockIdx.z;
  const std::uint32_t i = threadIdx.x;
  const std::uint32_t lane = i & 31u;
  const std::uint32_t wave = i >> 5;
  const std::uint32_t pos = *start_pos + t;
  const std::uint32_t n_kv = pos + 1;
  const std::uint32_t kvh = h / (heads / kv_heads);
  constexpr std::uint32_t kBatch = 8;
  __shared__ float qs[kHeadDim];
  __shared__ float wsum[kBatch][8];
  if (i < kHeadDim) {
    qs[i] = q[(static_cast<std::size_t>(t) * heads + h) * kHeadDim + i];
  }
  __syncthreads();
  const float qi = (i < kHeadDim) ? qs[i] : 0.0f;
  float m = -INFINITY;
  float l = 0.0f;
  float acc = 0.0f;
  // Blocked rows: (kv_heads*d/32) blocks. Key rows use `key_block_bytes`
  // and value rows `value_block_bytes` (34 = 8-bit codes, 18 = packed
  // 4-bit nibbles for the MBUFO_KV_Q4/V4 grids).
  const std::size_t k_stride =
      static_cast<std::size_t>(kv_heads) * kHeadDim / 32 * key_block_bytes;
  const std::size_t v_stride =
      static_cast<std::size_t>(kv_heads) * kHeadDim / 32 * value_block_bytes;
  const std::size_t k_head_off =
      static_cast<std::size_t>(kvh) * (kHeadDim / 32) * key_block_bytes;
  const std::size_t v_head_off =
      static_cast<std::size_t>(kvh) * (kHeadDim / 32) * value_block_bytes;
  // Dim i lives in 32-elem block i/32 at byte offset i%32; the f16 scale
  // trails the block (+32 for 8-bit codes, +16 for packed nibbles) and is
  // broadcast-read the same for every lane of the block.
  const std::uint32_t lane_block = i / 32;
  const std::uint32_t lane_off = i % 32;
  const std::uint32_t k_beg = s * chunk;
  std::uint32_t k_end = k_beg + chunk;
  if (k_end > n_kv) {
    k_end = n_kv;
  }
  for (std::uint32_t j0 = k_beg; j0 < k_end; j0 += kBatch) {
    float vv[kBatch];
#pragma unroll
    for (std::uint32_t b = 0; b < kBatch; ++b) {
      const std::uint32_t j = j0 + b;
      float kval = 0.0f;
      float vval = 0.0f;
      if (j < k_end && i < kHeadDim) {
        if (key_block_bytes == 34) {
          const typename Codec::Elem* kblk =
              k_cache + static_cast<std::size_t>(j) * k_stride + k_head_off +
              static_cast<std::size_t>(lane_block) * key_block_bytes;
          const float sk = __half2float(
              *reinterpret_cast<const __half*>(kblk + key_block_bytes - 2));
          kval = Codec::Decode(kblk[lane_off]) * sk;
        } else {
          // 18-byte 4-bit keys: element 2e in the low nibble, 2e+1 in the
          // high; sign-extended exactly like the scalar unpack path.
          const std::uint8_t* kblk =
              reinterpret_cast<const std::uint8_t*>(k_cache) +
              static_cast<std::size_t>(j) * k_stride + k_head_off +
              static_cast<std::size_t>(lane_block) * key_block_bytes;
          const float sk = __half2float(
              *reinterpret_cast<const __half*>(kblk + key_block_bytes - 2));
          const std::uint8_t byte = kblk[lane_off / 2];
          const int q = (lane_off % 2 == 0) ? (byte & 0xFu) : (byte >> 4);
          kval = static_cast<float>(q - (q & 8 ? 16 : 0)) * sk;
        }
        if (value_block_bytes == 34) {
          const typename Codec::Elem* vblk =
              v_cache + static_cast<std::size_t>(j) * v_stride + v_head_off +
              static_cast<std::size_t>(lane_block) * value_block_bytes;
          const float sv = __half2float(
              *reinterpret_cast<const __half*>(vblk + value_block_bytes - 2));
          vval = Codec::Decode(vblk[lane_off]) * sv;
        } else {
          const std::uint8_t* vblk =
              reinterpret_cast<const std::uint8_t*>(v_cache) +
              static_cast<std::size_t>(j) * v_stride + v_head_off +
              static_cast<std::size_t>(lane_block) * value_block_bytes;
          const float sv = __half2float(
              *reinterpret_cast<const __half*>(vblk + value_block_bytes - 2));
          const std::uint8_t byte = vblk[lane_off / 2];
          const int q = (lane_off % 2 == 0) ? (byte & 0xFu) : (byte >> 4);
          vval = static_cast<float>(q - (q & 8 ? 16 : 0)) * sv;
        }
      }
      vv[b] = vval;
      // One wave holds 32 consecutive dims; WaveSum needs no barrier, and
      // every lane participates uniformly (shuffle requirement).
      const float ws = WaveSum(qi * kval);
      if (lane == 0) {
        wsum[b][wave] = ws;
      }
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t b = 0; b < kBatch; ++b) {
      const std::uint32_t j = j0 + b;
      if (j >= k_end) {
        break;
      }
      float dot = wsum[b][0];
#pragma unroll
      for (std::uint32_t w = 1; w < 8; ++w) {
        dot += wsum[b][w];
      }
      const float score = dot * score_scale;
      const float m_new = fmaxf(m, score);
      const float corr = m == -INFINITY ? 0.0f : __expf(m - m_new);
      const float wgt = __expf(score - m_new);
      m = m_new;
      l = l * corr + wgt;
      acc = acc * corr + wgt * vv[b];
    }
    __syncthreads();
  }
  // Empty chunks (early queries under a late split) stay at m=-inf/l=0 and
  // contribute nothing in the merge.
  float* part = partials + (static_cast<std::size_t>(t) * heads + h) * splits *
                                (kHeadDim + 2) +
                static_cast<std::size_t>(s) * (kHeadDim + 2);
  if (i == 0) {
    part[0] = m;
    part[1] = l;
  }
  if (i < kHeadDim) {
    part[2 + i] = acc;
  }
}

/// Fold the split partials of one row in fixed chunk order (deterministic
/// across runs): the usual log-sum-exp rescale over (m, l, acc[d]).
template <std::uint32_t kHeadDim, bool kRowChunk = false>
__global__ void AttentionBlockedSplitKMergeKernel(
    const float* __restrict__ partials, float* __restrict__ out,
    std::uint32_t heads, std::uint32_t splits,
    const std::uint32_t* __restrict__ start_pos = nullptr, std::uint32_t chunk_base = 0) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t i = threadIdx.x;
  // No __syncthreads below: per-thread fold, safe to early-return the
  // idle upper lanes of a d == 128 launch.
  if (i >= kHeadDim) {
    return;
  }
  const float* base = partials + (static_cast<std::size_t>(t) * heads + h) *
                                     splits * (kHeadDim + 2);
  // kRowChunk: `splits` is only the partial stride; this row folds its own split count.
  std::uint32_t n_fold = splits;
  if constexpr (kRowChunk) n_fold = RowChunkSplits(*start_pos + t + 1u, chunk_base);
  float m = -INFINITY;
  for (std::uint32_t s = 0; s < n_fold; ++s) {
    m = fmaxf(m, base[s * (kHeadDim + 2)]);
  }
  float l = 0.0f;
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < n_fold; ++s) {
    const float* part = base + s * (kHeadDim + 2);
    const float ms = part[0];
    const float scale = ms == -INFINITY ? 0.0f : __expf(ms - m);
    l += part[1] * scale;
    acc += part[2 + i] * scale;
  }
  out[(static_cast<std::size_t>(t) * heads + h) * kHeadDim + i] = acc / l;
}


__device__ __forceinline__ float WaveMax(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, XorLane(v, offset));
  }
  return v;
}

// ---------------------------------------------------------------------------
// GQA-grouped split-K decode. One single-wave block per (kv head, token,
// split) serves ALL q heads of the group (kQ == heads / kv_heads), so every K/V
// block is read once per group instead of once per q head (16x for Glimmer's
// 32 heads / 2 kv heads), and the keys are scored lane-per-key in 32-key tiles:
// one wave max / wave sum per head per tile instead of one per key.
//   score pass : lane j decodes its key row (block by block) and dots it with
//                the group's q rows held in LDS (broadcast reads);
//   softmax    : per head, wave max/sum over the tile, rescale accumulators,
//                weights go to LDS;
//   PV pass    : each lane owns kD / 32 dims (half2-aligned pairs 2*lane+64*u)
//                and walks the tile's keys with the LDS weights.
// Partials use the same layout as the legacy kernel, so the merge is shared.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void DecodeKeyBlock32(const std::uint8_t* blk,
                                                 std::uint32_t block_bytes,
                                                 float* out32) {
  if (block_bytes == 34) {
    const float sc = blocked_kv::LoadScaleF16(blk + 32);
    const unsigned short* p = reinterpret_cast<const unsigned short*>(blk);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const unsigned u = p[i];
      out32[2 * i] = static_cast<float>(static_cast<std::int8_t>(u & 0xFFu)) * sc;
      out32[2 * i + 1] =
          static_cast<float>(static_cast<std::int8_t>(u >> 8)) * sc;
    }
  } else {
    const float sc = blocked_kv::LoadScaleF16(blk + 16);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const unsigned byte = blk[i];
      const int lo = static_cast<int>(byte & 0xFu);
      const int hi = static_cast<int>(byte >> 4);
      out32[2 * i] = static_cast<float>(lo - ((lo & 8) ? 16 : 0)) * sc;
      out32[2 * i + 1] = static_cast<float>(hi - ((hi & 8) ? 16 : 0)) * sc;
    }
  }
}

// Elements p and p + 1 (p even) of one value row.
__device__ __forceinline__ float2 DecodeValuePair(const std::uint8_t* row,
                                                  std::uint32_t block_bytes,
                                                  std::uint32_t p) {
  const std::uint8_t* blk = row + static_cast<std::size_t>(p >> 5) * block_bytes;
  const std::uint32_t off = p & 31u;
  if (block_bytes == 34) {
    const float sc = blocked_kv::LoadScaleF16(blk + 32);
    const unsigned u = *reinterpret_cast<const unsigned short*>(blk + off);
    return make_float2(
        static_cast<float>(static_cast<std::int8_t>(u & 0xFFu)) * sc,
        static_cast<float>(static_cast<std::int8_t>(u >> 8)) * sc);
  }
  const float sc = blocked_kv::LoadScaleF16(blk + 16);
  const unsigned byte = blk[off >> 1];
  const int lo = static_cast<int>(byte & 0xFu);
  const int hi = static_cast<int>(byte >> 4);
  return make_float2(static_cast<float>(lo - ((lo & 8) ? 16 : 0)) * sc,
                     static_cast<float>(hi - ((hi & 8) ? 16 : 0)) * sc);
}

template <std::uint32_t kD, std::uint32_t kQ, bool kRowChunk = false>
__launch_bounds__(32) __global__ void AttentionBlockedSplitKGqaKernel(
    const float* __restrict__ q, const std::uint8_t* __restrict__ k_cache,
    const std::uint8_t* __restrict__ v_cache, float* __restrict__ partials,
    const std::uint32_t* __restrict__ start_pos, std::uint32_t heads,
    std::uint32_t kv_heads, float score_scale, std::uint32_t splits,
    std::uint32_t chunk, std::uint32_t key_block_bytes,
    std::uint32_t value_block_bytes) {
  constexpr std::uint32_t kB = kD / 32;  // 32-element blocks per head row
  constexpr std::uint32_t kU = kD / 64;  // dim pairs per lane in the PV pass
  const std::uint32_t kvh = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t s = blockIdx.z;
  const std::uint32_t lane = threadIdx.x;  // 0..31, one wave
  const std::uint32_t pos = *start_pos + t;
  const std::uint32_t n_kv = pos + 1;
  // kRowChunk: `chunk` is the base of this row's own chunk size (RowChunkKeys), `splits` the partial stride; blocks past the
  // row's last split have nothing to do and write nothing (the row's merge folds only its own splits).
  if constexpr (kRowChunk) chunk = RowChunkKeys(n_kv, chunk);
  const std::uint32_t k_beg = s * chunk;
  if constexpr (kRowChunk) {
    if (k_beg >= n_kv) return;
  }
  std::uint32_t k_end = k_beg + chunk;
  if (k_end > n_kv) k_end = n_kv;
  const std::size_t k_stride =
      static_cast<std::size_t>(kv_heads) * kB * key_block_bytes;
  const std::size_t v_stride =
      static_cast<std::size_t>(kv_heads) * kB * value_block_bytes;
  const std::size_t k_head_off = static_cast<std::size_t>(kvh) * kB * key_block_bytes;
  const std::size_t v_head_off = static_cast<std::size_t>(kvh) * kB * value_block_bytes;

  __shared__ float qsm[kQ][kD];
  __shared__ float wsm[kQ][32];
#pragma unroll
  for (std::uint32_t hh = 0; hh < kQ; ++hh) {
    const float* qrow =
        q + (static_cast<std::size_t>(t) * heads + kvh * kQ + hh) * kD;
    for (std::uint32_t i = lane; i < kD; i += 32) qsm[hh][i] = qrow[i];
  }
  __syncthreads();

  float m[kQ], l[kQ];
  float2 acc[kQ][kU];
#pragma unroll
  for (std::uint32_t hh = 0; hh < kQ; ++hh) {
    m[hh] = -INFINITY;
    l[hh] = 0.0f;
#pragma unroll
    for (std::uint32_t u = 0; u < kU; ++u) acc[hh][u] = make_float2(0.0f, 0.0f);
  }

  for (std::uint32_t base = k_beg; base < k_end; base += 32) {
    const std::uint32_t j = base + lane;
    const bool valid = j < k_end;
    float score[kQ];
#pragma unroll
    for (std::uint32_t hh = 0; hh < kQ; ++hh) score[hh] = -INFINITY;
    if (valid) {
      float dot[kQ];
#pragma unroll
      for (std::uint32_t hh = 0; hh < kQ; ++hh) dot[hh] = 0.0f;
      const std::uint8_t* krow = k_cache + static_cast<std::size_t>(j) * k_stride + k_head_off;
#pragma unroll 1  // full unrolling hoists every LDS read and spills to scratch
      for (std::uint32_t b = 0; b < kB; ++b) {
        float kv[32];
        DecodeKeyBlock32(krow + static_cast<std::size_t>(b) * key_block_bytes,
                         key_block_bytes, kv);
#pragma unroll
        for (std::uint32_t hh = 0; hh < kQ; ++hh) {
          float a = dot[hh];
#pragma unroll
          for (int e = 0; e < 32; e += 4) {
            const float4 qq = *reinterpret_cast<const float4*>(&qsm[hh][b * 32 + e]);
            a = fmaf(qq.x, kv[e], a);
            a = fmaf(qq.y, kv[e + 1], a);
            a = fmaf(qq.z, kv[e + 2], a);
            a = fmaf(qq.w, kv[e + 3], a);
          }
          dot[hh] = a;
        }
      }
#pragma unroll
      for (std::uint32_t hh = 0; hh < kQ; ++hh) score[hh] = dot[hh] * score_scale;
    }
#pragma unroll
    for (std::uint32_t hh = 0; hh < kQ; ++hh) {
      const float m_tile = WaveMax(score[hh]);
      const float m_new = m[hh] > m_tile ? m[hh] : m_tile;
      const float corr = (m[hh] == -INFINITY) ? 0.0f : expf(m[hh] - m_new);
      const float w = valid ? expf(score[hh] - m_new) : 0.0f;
      l[hh] = l[hh] * corr + WaveSum(w);
#pragma unroll
      for (std::uint32_t u = 0; u < kU; ++u) {
        acc[hh][u].x *= corr;
        acc[hh][u].y *= corr;
      }
      wsm[hh][lane] = w;
      m[hh] = m_new;
    }
    __syncthreads();
    const std::uint32_t cnt = k_end - base < 32u ? k_end - base : 32u;
    for (std::uint32_t jj = 0; jj < cnt; ++jj) {
      const std::uint8_t* vrow =
          v_cache + static_cast<std::size_t>(base + jj) * v_stride + v_head_off;
      float2 vv[kU];
#pragma unroll
      for (std::uint32_t u = 0; u < kU; ++u)
        vv[u] = DecodeValuePair(vrow, value_block_bytes, 2 * lane + 64 * u);
#pragma unroll
      for (std::uint32_t hh = 0; hh < kQ; ++hh) {
        const float wj = wsm[hh][jj];
#pragma unroll
        for (std::uint32_t u = 0; u < kU; ++u) {
          acc[hh][u].x = fmaf(wj, vv[u].x, acc[hh][u].x);
          acc[hh][u].y = fmaf(wj, vv[u].y, acc[hh][u].y);
        }
      }
    }
    __syncthreads();
  }
  const bool live = k_beg < k_end;
#pragma unroll
  for (std::uint32_t hh = 0; hh < kQ; ++hh) {
    float* part = partials +
                  ((static_cast<std::size_t>(t) * heads + kvh * kQ + hh) * splits + s) *
                      (kD + 2);
    if (lane == 0) {
      part[0] = live ? m[hh] : -INFINITY;
      part[1] = live ? l[hh] : 0.0f;
    }
#pragma unroll
    for (std::uint32_t u = 0; u < kU; ++u) {
      part[2 + 64 * u + 2 * lane] = live ? acc[hh][u].x : 0.0f;
      part[2 + 64 * u + 2 * lane + 1] = live ? acc[hh][u].y : 0.0f;
    }
  }
}

template <std::uint32_t kD, std::uint32_t kQ, bool kRowChunk = false>
void LaunchGqa(const float* q, const std::uint8_t* k_cache,
               const std::uint8_t* v_cache, float* partials,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t n_tokens, float score_scale,
               std::uint32_t splits, std::uint32_t chunk,
               std::uint32_t key_block_bytes, std::uint32_t value_block_bytes,
               hipStream_t stream) {
  hipLaunchKernelGGL((AttentionBlockedSplitKGqaKernel<kD, kQ, kRowChunk>),
                     dim3(kv_heads, n_tokens, splits), dim3(32), 0, stream, q,
                     k_cache, v_cache, partials, start_pos, heads, kv_heads,
                     score_scale, splits, chunk, key_block_bytes,
                     value_block_bytes);
}


// ---------------------------------------------------------------------------
// Row-chunk decode kernel, multi-wave (d == 256, 34-byte Q8 blocks, GQA group kQ <= 8).
// Same math as the GQA kernel above -- bit-identical scores (the dot of a key row with a query row runs over the dims in the
// same order on the same dequantized values) -- but organised for latency hiding instead of one wave per chunk:
//  * a block is kWaves waves of one (kv head, token, split); the chunk's 32-key tiles are dealt round-robin to the waves, so
//    four times as many waves are in flight and each carries a short dependent chain;
//  * a key row (272 bytes, 16-byte aligned) is read as 17 vector loads instead of 128 two-byte loads;
//  * the PV pass loads four value rows before it consumes them.
// Each wave keeps its own online-softmax state; wave 0 folds the waves in fixed order (LDS) before the block writes its
// partial, so the result is deterministic and depends only on the row's chunk size (RowChunkKeys).
// ---------------------------------------------------------------------------
constexpr std::uint32_t kMwWaves = 4;

// Element `e` (0..31) of block `b` of a 272-byte row held in 68 dwords: block b starts at byte 34 * b.
template <std::uint32_t b, std::uint32_t e, std::size_t N>
__device__ __forceinline__ float RowCode(const unsigned (&w)[N], float sc) {
  constexpr std::uint32_t byte = 34 * b + e;
  const int v = static_cast<std::int8_t>((w[byte >> 2] >> (8 * (byte & 3))) & 0xFFu);
  return static_cast<float>(v) * sc;
}
template <std::uint32_t b, std::size_t N>
__device__ __forceinline__ float RowScale(const unsigned (&w)[N]) {
  constexpr std::uint32_t byte = 34 * b + 32;
  return __half2float(__builtin_bit_cast(__half, static_cast<std::uint16_t>((w[byte >> 2] >> (8 * (byte & 3))) & 0xFFFFu)));
}

template <std::uint32_t kQ, std::uint32_t b, std::size_t N, std::uint32_t kQD = 256>
__device__ __forceinline__ void DotBlock(const unsigned (&w)[N], const float (*qsm)[kQD], float (&dot)[kQ]) {
  const float sc = RowScale<b>(w);
  float kv[32];
  kv[0] = RowCode<b, 0>(w, sc);   kv[1] = RowCode<b, 1>(w, sc);   kv[2] = RowCode<b, 2>(w, sc);   kv[3] = RowCode<b, 3>(w, sc);
  kv[4] = RowCode<b, 4>(w, sc);   kv[5] = RowCode<b, 5>(w, sc);   kv[6] = RowCode<b, 6>(w, sc);   kv[7] = RowCode<b, 7>(w, sc);
  kv[8] = RowCode<b, 8>(w, sc);   kv[9] = RowCode<b, 9>(w, sc);   kv[10] = RowCode<b, 10>(w, sc); kv[11] = RowCode<b, 11>(w, sc);
  kv[12] = RowCode<b, 12>(w, sc); kv[13] = RowCode<b, 13>(w, sc); kv[14] = RowCode<b, 14>(w, sc); kv[15] = RowCode<b, 15>(w, sc);
  kv[16] = RowCode<b, 16>(w, sc); kv[17] = RowCode<b, 17>(w, sc); kv[18] = RowCode<b, 18>(w, sc); kv[19] = RowCode<b, 19>(w, sc);
  kv[20] = RowCode<b, 20>(w, sc); kv[21] = RowCode<b, 21>(w, sc); kv[22] = RowCode<b, 22>(w, sc); kv[23] = RowCode<b, 23>(w, sc);
  kv[24] = RowCode<b, 24>(w, sc); kv[25] = RowCode<b, 25>(w, sc); kv[26] = RowCode<b, 26>(w, sc); kv[27] = RowCode<b, 27>(w, sc);
  kv[28] = RowCode<b, 28>(w, sc); kv[29] = RowCode<b, 29>(w, sc); kv[30] = RowCode<b, 30>(w, sc); kv[31] = RowCode<b, 31>(w, sc);
#pragma unroll
  for (std::uint32_t hh = 0; hh < kQ; ++hh) {
    float a = dot[hh];
#pragma unroll
    for (int e = 0; e < 32; e += 4) {
      const float4 qq = *reinterpret_cast<const float4*>(&qsm[hh][b * 32 + e]);
      a = fmaf(qq.x, kv[e], a);
      a = fmaf(qq.y, kv[e + 1], a);
      a = fmaf(qq.z, kv[e + 2], a);
      a = fmaf(qq.w, kv[e + 3], a);
    }
    dot[hh] = a;
  }
}

template <std::uint32_t kQ>
__device__ __forceinline__ void DotRow(const std::uint8_t* krow, const float (*qsm)[256], float (&dot)[kQ]) {
  unsigned w[68];
#pragma unroll
  for (std::uint32_t i = 0; i < 17; ++i) {
    const uint4 v = *reinterpret_cast<const uint4*>(krow + 16 * i);
    w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w;
  }
  DotBlock<kQ, 0>(w, qsm, dot); DotBlock<kQ, 1>(w, qsm, dot); DotBlock<kQ, 2>(w, qsm, dot); DotBlock<kQ, 3>(w, qsm, dot);
  DotBlock<kQ, 4>(w, qsm, dot); DotBlock<kQ, 5>(w, qsm, dot); DotBlock<kQ, 6>(w, qsm, dot); DotBlock<kQ, 7>(w, qsm, dot);
}

template <std::uint32_t kQ>
__launch_bounds__(32 * kMwWaves) __global__ void AttentionBlockedSplitKRowChunkMwKernel(
    const float* __restrict__ q, const std::uint8_t* __restrict__ k_cache, const std::uint8_t* __restrict__ v_cache,
    float* __restrict__ partials, const std::uint32_t* __restrict__ start_pos, std::uint32_t heads, std::uint32_t kv_heads,
    float score_scale, std::uint32_t splits, std::uint32_t chunk_base) {
  constexpr std::uint32_t kD = 256, kU = 4;
  const std::uint32_t kvh = blockIdx.x, t = blockIdx.y, s = blockIdx.z;
  const std::uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
  const std::uint32_t n_kv = *start_pos + t + 1u;
  const std::uint32_t chunk = RowChunkKeys(n_kv, chunk_base);
  const std::uint32_t k_beg = s * chunk;
  if (k_beg >= n_kv) return;  // uniform over the block
  std::uint32_t k_end = k_beg + chunk;
  if (k_end > n_kv) k_end = n_kv;
  constexpr std::uint32_t kBlk = 34, kRowBlocks = 8;
  const std::size_t k_stride = static_cast<std::size_t>(kv_heads) * kRowBlocks * kBlk;
  const std::size_t head_off = static_cast<std::size_t>(kvh) * kRowBlocks * kBlk;

  __shared__ __attribute__((aligned(16))) float qsm[kQ][kD];
  __shared__ float wsm[kMwWaves][kQ][32];
  __shared__ __attribute__((aligned(16))) float mrg[kQ][kD + 2];  // one wave's state at a time (LDS caps resident blocks)
  for (std::uint32_t i = tid; i < kQ * kD; i += 32 * kMwWaves) {
    const std::uint32_t hh = i / kD, d = i % kD;
    qsm[hh][d] = q[(static_cast<std::size_t>(t) * heads + kvh * kQ + hh) * kD + d];
  }
  __syncthreads();

  float m[kQ], l[kQ];
  float2 acc[kQ][kU];
#pragma unroll
  for (std::uint32_t hh = 0; hh < kQ; ++hh) {
    m[hh] = -INFINITY;
    l[hh] = 0.0f;
#pragma unroll
    for (std::uint32_t u = 0; u < kU; ++u) acc[hh][u] = make_float2(0.0f, 0.0f);
  }

  const std::uint32_t n_tiles = (k_end - k_beg + 31u) / 32u;
  for (std::uint32_t it = 0; it * kMwWaves < n_tiles; ++it) {
    const std::uint32_t tile = it * kMwWaves + wave;
    const std::uint32_t base = k_beg + tile * 32u;
    const bool tile_live = tile < n_tiles;
    const std::uint32_t j = base + lane;
    const bool valid = tile_live && j < k_end;
    float score[kQ];
#pragma unroll
    for (std::uint32_t hh = 0; hh < kQ; ++hh) score[hh] = -INFINITY;
    if (valid) {
      float dot[kQ];
#pragma unroll
      for (std::uint32_t hh = 0; hh < kQ; ++hh) dot[hh] = 0.0f;
      DotRow<kQ>(k_cache + static_cast<std::size_t>(j) * k_stride + head_off, qsm, dot);
#pragma unroll
      for (std::uint32_t hh = 0; hh < kQ; ++hh) score[hh] = dot[hh] * score_scale;
    }
#pragma unroll
    for (std::uint32_t hh = 0; hh < kQ; ++hh) {
      const float m_tile = WaveMax(score[hh]);
      const float m_new = m[hh] > m_tile ? m[hh] : m_tile;
      const float corr = (m[hh] == -INFINITY) ? 0.0f : expf(m[hh] - m_new);
      const float w = valid ? expf(score[hh] - m_new) : 0.0f;
      l[hh] = l[hh] * corr + WaveSum(w);
#pragma unroll
      for (std::uint32_t u = 0; u < kU; ++u) {
        acc[hh][u].x *= corr;
        acc[hh][u].y *= corr;
      }
      wsm[wave][hh][lane] = w;
      m[hh] = m_new;
    }
    __syncthreads();  // weights of this wave's tile are visible to all its lanes
    const std::uint32_t cnt = !tile_live ? 0u : (k_end - base < 32u ? k_end - base : 32u);
    for (std::uint32_t jj = 0; jj < cnt; jj += 4) {
      float2 vv[4][kU];
#pragma unroll
      for (std::uint32_t k4 = 0; k4 < 4; ++k4) {
        if (jj + k4 < cnt) {
          const std::uint8_t* vrow = v_cache + static_cast<std::size_t>(base + jj + k4) * k_stride + head_off;
#pragma unroll
          for (std::uint32_t u = 0; u < kU; ++u) vv[k4][u] = DecodeValuePair(vrow, kBlk, 2 * lane + 64 * u);
        }
      }
#pragma unroll
      for (std::uint32_t k4 = 0; k4 < 4; ++k4) {
        if (jj + k4 < cnt) {
#pragma unroll
          for (std::uint32_t hh = 0; hh < kQ; ++hh) {
            const float wj = wsm[wave][hh][jj + k4];
#pragma unroll
            for (std::uint32_t u = 0; u < kU; ++u) {
              acc[hh][u].x = fmaf(wj, vv[k4][u].x, acc[hh][u].x);
              acc[hh][u].y = fmaf(wj, vv[k4][u].y, acc[hh][u].y);
            }
          }
        }
      }
    }
    __syncthreads();  // wsm is rewritten by the next tile
  }

  // Fold the waves in fixed order: wave 1, 2, 3 each hand their state to wave 0 through one LDS buffer.
  for (std::uint32_t w2 = 1; w2 < kMwWaves; ++w2) {
    if (wave == w2) {
#pragma unroll
      for (std::uint32_t hh = 0; hh < kQ; ++hh) {
        if (lane == 0) {
          mrg[hh][0] = m[hh];
          mrg[hh][1] = l[hh];
        }
#pragma unroll
        for (std::uint32_t u = 0; u < kU; ++u) {
          mrg[hh][2 + 64 * u + 2 * lane] = acc[hh][u].x;
          mrg[hh][2 + 64 * u + 2 * lane + 1] = acc[hh][u].y;
        }
      }
    }
    __syncthreads();
    if (wave == 0) {
#pragma unroll
      for (std::uint32_t hh = 0; hh < kQ; ++hh) {
        const float m2 = mrg[hh][0], l2 = mrg[hh][1];
        const float m_new = m[hh] > m2 ? m[hh] : m2;
        const float c1 = (m[hh] == -INFINITY) ? 0.0f : expf(m[hh] - m_new);
        const float c2 = (m2 == -INFINITY) ? 0.0f : expf(m2 - m_new);
        l[hh] = l[hh] * c1 + l2 * c2;
#pragma unroll
        for (std::uint32_t u = 0; u < kU; ++u) {
          const float ax = mrg[hh][2 + 64 * u + 2 * lane], ay = mrg[hh][2 + 64 * u + 2 * lane + 1];
          acc[hh][u].x = acc[hh][u].x * c1 + ax * c2;
          acc[hh][u].y = acc[hh][u].y * c1 + ay * c2;
        }
        m[hh] = m_new;
      }
    }
    __syncthreads();
  }
  if (wave != 0) return;
#pragma unroll
  for (std::uint32_t hh = 0; hh < kQ; ++hh) {
    float* part = partials + ((static_cast<std::size_t>(t) * heads + kvh * kQ + hh) * splits + s) * (kD + 2);
    if (lane == 0) {
      part[0] = m[hh];
      part[1] = l[hh];
    }
#pragma unroll
    for (std::uint32_t u = 0; u < kU; ++u) {
      part[2 + 64 * u + 2 * lane] = acc[hh][u].x;
      part[2 + 64 * u + 2 * lane + 1] = acc[hh][u].y;
    }
  }
}

template <std::uint32_t kQ>
void LaunchRowChunkMw(const float* q, const std::uint8_t* k_cache, const std::uint8_t* v_cache, float* partials,
                      const std::uint32_t* start_pos, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t n_tokens,
                      float score_scale, std::uint32_t splits, std::uint32_t chunk_base, hipStream_t stream) {
  hipLaunchKernelGGL((AttentionBlockedSplitKRowChunkMwKernel<kQ>), dim3(kv_heads, n_tokens, splits), dim3(32 * kMwWaves), 0,
                     stream, q, k_cache, v_cache, partials, start_pos, heads, kv_heads, score_scale, splits, chunk_base);
}


// ---------------------------------------------------------------------------
// Row-chunk decode with an attention sink and an optional sliding window (gpt-oss: d == 64, 8 query heads per kv head; also
// usable at d == 128), reading the blocked Q8 cache directly. The per-step dequantization of the whole cache into F16 staging
// plus the F16 flash-sink kernel moved ~3x the bytes of this read.
// Same structure as AttentionBlockedSplitKRowChunkMwKernel: a row's split depends only on the length of its visible span
// (n_kv - window start), the span's 32-key tiles are dealt to 4 waves, lane-per-key scores, per-wave online softmax folded in
// fixed order through LDS. The sink is one extra softmax term per (token, head) carried by wave 0 of split 0.
// ---------------------------------------------------------------------------
template <std::uint32_t kD, std::uint32_t kQ>
__launch_bounds__(32 * kMwWaves) __global__ void AttentionBlockedSinkRowChunkKernel(
    const float* __restrict__ q, const std::uint8_t* __restrict__ k_cache, const std::uint8_t* __restrict__ v_cache,
    const float* __restrict__ sinks, float* __restrict__ partials, std::uint32_t start_pos, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t window, float score_scale, std::uint32_t splits, std::uint32_t chunk_base) {
  constexpr std::uint32_t kU = kD / 64, kNB = kD / 32, kBlk = 34;
  constexpr std::uint32_t kDw = kNB * kBlk / 4;
  static_assert(kNB * kBlk % 4 == 0, "a key row of one kv head must be a whole number of dwords");
  const std::uint32_t kvh = blockIdx.x, t = blockIdx.y, s = blockIdx.z;
  const std::uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
  const std::uint32_t n_kv = start_pos + t + 1u;
  const std::uint32_t k_lo = (window != 0 && n_kv > window) ? n_kv - window : 0u;
  const std::uint32_t span = n_kv - k_lo;
  const std::uint32_t chunk = RowChunkKeys(span, chunk_base);
  const std::uint32_t rel = s * chunk;
  if (rel >= span) return;  // uniform over the block
  const std::uint32_t k_beg = k_lo + rel;
  std::uint32_t k_end = k_beg + chunk;
  if (k_end > n_kv) k_end = n_kv;
  const std::size_t k_stride = static_cast<std::size_t>(kv_heads) * kNB * kBlk;
  const std::size_t head_off = static_cast<std::size_t>(kvh) * kNB * kBlk;
  const std::size_t h_base = static_cast<std::size_t>(t) * heads + kvh * kQ;

  __shared__ __attribute__((aligned(16))) float qsm[kQ][kD];
  __shared__ float wsm[kMwWaves][kQ][32];
  __shared__ __attribute__((aligned(16))) float mrg[kQ][kD + 2];
  for (std::uint32_t i = tid; i < kQ * kD; i += 32 * kMwWaves) {
    const std::uint32_t hh = i / kD, d = i % kD;
    qsm[hh][d] = q[(h_base + hh) * kD + d];
  }
  __syncthreads();

  float m[kQ], l[kQ];
  float2 acc[kQ][kU];
  const bool carries_sink = (s == 0) && (wave == 0) && (sinks != nullptr);
#pragma unroll
  for (std::uint32_t hh = 0; hh < kQ; ++hh) {
    m[hh] = carries_sink ? sinks[kvh * kQ + hh] : -INFINITY;
    l[hh] = carries_sink ? 1.0F : 0.0F;
#pragma unroll
    for (std::uint32_t u = 0; u < kU; ++u) acc[hh][u] = make_float2(0.0F, 0.0F);
  }

  const std::uint32_t n_tiles = (k_end - k_beg + 31u) / 32u;
  for (std::uint32_t it = 0; it * kMwWaves < n_tiles; ++it) {
    const std::uint32_t tile = it * kMwWaves + wave;
    const std::uint32_t base = k_beg + tile * 32u;
    const bool tile_live = tile < n_tiles;
    const std::uint32_t j = base + lane;
    const bool valid = tile_live && j < k_end;
    float score[kQ];
#pragma unroll
    for (std::uint32_t hh = 0; hh < kQ; ++hh) score[hh] = -INFINITY;
    if (valid) {
      float dot[kQ];
#pragma unroll
      for (std::uint32_t hh = 0; hh < kQ; ++hh) dot[hh] = 0.0F;
      const unsigned* krow = reinterpret_cast<const unsigned*>(k_cache + static_cast<std::size_t>(j) * k_stride + head_off);
      unsigned w[kDw];
#pragma unroll
      for (std::uint32_t i = 0; i < kDw; ++i) w[i] = krow[i];
      DotBlock<kQ, 0, kDw, kD>(w, qsm, dot);
      DotBlock<kQ, 1, kDw, kD>(w, qsm, dot);
      if constexpr (kNB > 2) {
        DotBlock<kQ, 2, kDw, kD>(w, qsm, dot);
        DotBlock<kQ, 3, kDw, kD>(w, qsm, dot);
      }
#pragma unroll
      for (std::uint32_t hh = 0; hh < kQ; ++hh) score[hh] = dot[hh] * score_scale;
    }
#pragma unroll
    for (std::uint32_t hh = 0; hh < kQ; ++hh) {
      const float m_tile = WaveMax(score[hh]);
      const float m_new = m[hh] > m_tile ? m[hh] : m_tile;
      const float corr = (m[hh] == -INFINITY) ? 0.0F : expf(m[hh] - m_new);
      const float w = valid ? expf(score[hh] - m_new) : 0.0F;
      l[hh] = l[hh] * corr + WaveSum(w);
#pragma unroll
      for (std::uint32_t u = 0; u < kU; ++u) {
        acc[hh][u].x *= corr;
        acc[hh][u].y *= corr;
      }
      wsm[wave][hh][lane] = w;
      m[hh] = m_new;
    }
    __syncthreads();
    const std::uint32_t cnt = !tile_live ? 0u : (k_end - base < 32u ? k_end - base : 32u);
    for (std::uint32_t jj = 0; jj < cnt; jj += 4) {
      float2 vv[4][kU];
#pragma unroll
      for (std::uint32_t k4 = 0; k4 < 4; ++k4) {
        if (jj + k4 < cnt) {
          const std::uint8_t* vrow = v_cache + static_cast<std::size_t>(base + jj + k4) * k_stride + head_off;
#pragma unroll
          for (std::uint32_t u = 0; u < kU; ++u) vv[k4][u] = DecodeValuePair(vrow, kBlk, 2 * lane + 64 * u);
        }
      }
#pragma unroll
      for (std::uint32_t k4 = 0; k4 < 4; ++k4) {
        if (jj + k4 < cnt) {
#pragma unroll
          for (std::uint32_t hh = 0; hh < kQ; ++hh) {
            const float wj = wsm[wave][hh][jj + k4];
#pragma unroll
            for (std::uint32_t u = 0; u < kU; ++u) {
              acc[hh][u].x = fmaf(wj, vv[k4][u].x, acc[hh][u].x);
              acc[hh][u].y = fmaf(wj, vv[k4][u].y, acc[hh][u].y);
            }
          }
        }
      }
    }
    __syncthreads();  // wsm is rewritten by the next tile
  }

  for (std::uint32_t w2 = 1; w2 < kMwWaves; ++w2) {
    if (wave == w2) {
#pragma unroll
      for (std::uint32_t hh = 0; hh < kQ; ++hh) {
        if (lane == 0) {
          mrg[hh][0] = m[hh];
          mrg[hh][1] = l[hh];
        }
#pragma unroll
        for (std::uint32_t u = 0; u < kU; ++u) {
          mrg[hh][2 + 64 * u + 2 * lane] = acc[hh][u].x;
          mrg[hh][2 + 64 * u + 2 * lane + 1] = acc[hh][u].y;
        }
      }
    }
    __syncthreads();
    if (wave == 0) {
#pragma unroll
      for (std::uint32_t hh = 0; hh < kQ; ++hh) {
        const float m2 = mrg[hh][0], l2 = mrg[hh][1];
        const float m_new = m[hh] > m2 ? m[hh] : m2;
        const float c1 = (m[hh] == -INFINITY) ? 0.0F : expf(m[hh] - m_new);
        const float c2 = (m2 == -INFINITY) ? 0.0F : expf(m2 - m_new);
        l[hh] = l[hh] * c1 + l2 * c2;
#pragma unroll
        for (std::uint32_t u = 0; u < kU; ++u) {
          const float ax = mrg[hh][2 + 64 * u + 2 * lane], ay = mrg[hh][2 + 64 * u + 2 * lane + 1];
          acc[hh][u].x = acc[hh][u].x * c1 + ax * c2;
          acc[hh][u].y = acc[hh][u].y * c1 + ay * c2;
        }
        m[hh] = m_new;
      }
    }
    __syncthreads();
  }
  if (wave != 0) return;
#pragma unroll
  for (std::uint32_t hh = 0; hh < kQ; ++hh) {
    float* part = partials + ((h_base + hh) * splits + s) * (kD + 2);
    if (lane == 0) {
      part[0] = m[hh];
      part[1] = l[hh];
    }
#pragma unroll
    for (std::uint32_t u = 0; u < kU; ++u) {
      part[2 + 64 * u + 2 * lane] = acc[hh][u].x;
      part[2 + 64 * u + 2 * lane + 1] = acc[hh][u].y;
    }
  }
}

// Fold the splits of one row (its own count, from the length of its visible span) in fixed order.
template <std::uint32_t kD>
__global__ void AttentionBlockedSinkMergeKernel(const float* __restrict__ partials, float* __restrict__ out,
                                                std::uint32_t heads, std::uint32_t splits, std::uint32_t start_pos,
                                                std::uint32_t window, std::uint32_t chunk_base) {
  const std::uint32_t h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
  if (i >= kD) return;
  const std::uint32_t n_kv = start_pos + t + 1u;
  const std::uint32_t span = n_kv - ((window != 0 && n_kv > window) ? n_kv - window : 0u);
  const std::uint32_t n_fold = RowChunkSplits(span, chunk_base);
  const float* base = partials + (static_cast<std::size_t>(t) * heads + h) * splits * (kD + 2);
  float m = -INFINITY, l = 0.0F, acc = 0.0F;
  for (std::uint32_t s = 0; s < n_fold; ++s) {
    const float* p = base + static_cast<std::size_t>(s) * (kD + 2);
    const float ms = p[0], ls = p[1];
    if (ms == -INFINITY) continue;
    const float m_new = m > ms ? m : ms;
    const float c0 = (m == -INFINITY) ? 0.0F : expf(m - m_new);
    const float c1 = expf(ms - m_new);
    acc = acc * c0 + p[2 + i] * c1;
    l = l * c0 + ls * c1;
    m = m_new;
  }
  out[(static_cast<std::size_t>(t) * heads + h) * kD + i] = acc / l;
}

// Supported groups: kD == 128 -> heads/kv_heads in {1,2,4,8,16}; kD == 256 ->
// {1,2,4,8} (register budget). Returns false for anything else.
template <std::uint32_t kD>
bool DispatchGqa(const float* q, const std::uint8_t* k_cache,
                 const std::uint8_t* v_cache, float* partials,
                 const std::uint32_t* start_pos, std::uint32_t heads,
                 std::uint32_t kv_heads, std::uint32_t n_tokens,
                 float score_scale, std::uint32_t splits, std::uint32_t chunk,
                 std::uint32_t key_block_bytes, std::uint32_t value_block_bytes,
                 hipStream_t stream) {
  const std::uint32_t qpg = heads / kv_heads;
#define MBUFO_GQA_CASE(Q)                                                        \
  case Q:                                                                       \
    LaunchGqa<kD, Q>(q, k_cache, v_cache, partials, start_pos, heads, kv_heads, \
                     n_tokens, score_scale, splits, chunk, key_block_bytes,     \
                     value_block_bytes, stream);                                \
    return true;
  switch (qpg) {
    MBUFO_GQA_CASE(1)
    MBUFO_GQA_CASE(2)
    MBUFO_GQA_CASE(4)
    MBUFO_GQA_CASE(8)
    case 16:
      if constexpr (kD == 128) {
        LaunchGqa<kD, 16>(q, k_cache, v_cache, partials, start_pos, heads,
                          kv_heads, n_tokens, score_scale, splits, chunk,
                          key_block_bytes, value_block_bytes, stream);
        return true;
      }
      return false;
    default:
      return false;
  }
#undef MBUFO_GQA_CASE
}

}  // namespace

// Split-K decode over blocked Q8 caches (d == 256 SWA, heads <= 16;
// d == 128 dense, heads <= 32). The split
// sizing copies the F16 Split-K heuristic (enough keys per chunk to amortize
// the merge, 64-way cap); one chunk falls back to the scalar path.
template <typename Codec, std::uint32_t kHeadDim>
bool AttentionBlockedSplitK(const float* q, const std::uint8_t* k_cache,
                            const std::uint8_t* v_cache, float* out,
                            std::uint32_t n_tokens,
                            const std::uint32_t* start_pos,
                            std::uint32_t kv_len, std::uint32_t heads,
                            std::uint32_t kv_heads, std::uint32_t d,
                            hipStream_t stream, float score_scale,
                            std::uint32_t key_block_bytes,
                            std::uint32_t value_block_bytes, float* partials,
                            std::size_t partials_bytes) {
  static_assert(kHeadDim == 256 || kHeadDim == 128,
                "blocked Split-K decode is d=128/256 only");
  const std::uint32_t kMaxHeads = (kHeadDim == 256) ? 16u : 32u;
  if (kHeadDim != d || heads == 0 || kv_heads == 0 || heads > kMaxHeads ||
      heads % kv_heads != 0 || n_tokens == 0 ||
      n_tokens > kBlockedSplitKMaxTokens || q == nullptr ||
      k_cache == nullptr || v_cache == nullptr || out == nullptr ||
      start_pos == nullptr || kv_len < kBlockedSplitKMinKv ||
      (key_block_bytes != 34 && key_block_bytes != 18) ||
      (value_block_bytes != 34 && value_block_bytes != 18)) {
    return false;
  }
  // GQA-grouped lane-per-key kernel (default); MBUFO_BLOCKED_SPLITK_LEGACY=1
  // restores the per-q-head kernel below for A/B. Groups the grouped kernel does
  // not cover (heads / kv_heads outside {1,2,4,8,16}, {1,2,4,8} for d == 256)
  // fall through to the legacy kernel.
  const std::uint32_t qpg = heads / kv_heads;
  const bool gqa_ok = std::getenv("MBUFO_BLOCKED_SPLITK_LEGACY") == nullptr &&
                      (qpg == 1 || qpg == 2 || qpg == 4 || qpg == 8 ||
                       (qpg == 16 && kHeadDim == 128));
  // Legacy sizing: ~128 keys per chunk at full GQA occupancy; grouped kernel:
  // one 32-key tile per chunk. 64-way cap either way.
  const std::uint32_t min_tile =
      gqa_ok ? 32u : ((kv_heads * n_tokens >= 8u) ? 128u : 64u);
  std::uint32_t splits = (kv_len + min_tile - 1u) / min_tile;
  if (splits < (gqa_ok ? 1u : 2u)) {
    return false;
  }
  if (splits > kBlockedSplitKMaxSplits) {
    splits = kBlockedSplitKMaxSplits;
  }
  // Partials for the per-chunk online-softmax state (MaxTokens x 16 heads x
  // 64 splits x (d+2)) come from the caller arena (§4.2: no statics inside
  // operators); short scratch refuses cleanly. Launches on one stream are
  // serial, so one buffer serves the whole batch.
  if (partials == nullptr ||
      partials_bytes < AttentionBlockedSplitKScratchBytes()) {
    return false;
  }
  const std::uint32_t chunk = (kv_len + splits - 1u) / splits;
  if (gqa_ok &&
      DispatchGqa<kHeadDim>(q, k_cache, v_cache, partials, start_pos, heads,
                            kv_heads, n_tokens, score_scale, splits, chunk,
                            key_block_bytes, value_block_bytes, stream)) {
    hipLaunchKernelGGL((AttentionBlockedSplitKMergeKernel<kHeadDim>),
                       dim3(heads, n_tokens), dim3(kThreads), 0, stream,
                       partials, out, heads, splits);
    return true;
  }
  hipLaunchKernelGGL((AttentionBlockedSplitKKernel<Codec, kHeadDim>),
                     dim3(heads, n_tokens, splits), dim3(kThreads), 0, stream,
                     q, k_cache, v_cache, partials, start_pos, heads, kv_heads,
                     score_scale, splits, chunk, key_block_bytes,
                     value_block_bytes);
  hipLaunchKernelGGL((AttentionBlockedSplitKMergeKernel<kHeadDim>),
                     dim3(heads, n_tokens), dim3(kThreads), 0, stream,
                     partials, out, heads, splits);
  return true;
}

template <typename Codec, std::uint32_t kHeadDim>
bool AttentionBlockedSplitKRowChunk(const float* q, const std::uint8_t* k_cache, const std::uint8_t* v_cache, float* out,
                                    std::uint32_t n_tokens, const std::uint32_t* start_pos, std::uint32_t kv_len,
                                    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d, hipStream_t stream,
                                    float score_scale, std::uint32_t key_block_bytes, std::uint32_t value_block_bytes,
                                    float* partials, std::size_t partials_bytes, std::uint32_t chunk_base) {
  static_assert(kHeadDim == 256 || kHeadDim == 128, "blocked Split-K decode is d=128/256 only");
  // Row-chunk mode has no register-budget limit on the head count (the multi-wave kernel keeps one group of <= 8 query
  // heads per block), so d == 256 models with more than 16 heads (Qwen3.8-27B: 24 heads / 4 kv heads) are served as well.
  const std::uint32_t kMaxHeads = 32u;
  if (kHeadDim != d || heads == 0 || kv_heads == 0 || heads > kMaxHeads || heads % kv_heads != 0 || n_tokens == 0 ||
      n_tokens > kBlockedSplitKMaxTokens || q == nullptr || k_cache == nullptr || v_cache == nullptr || out == nullptr ||
      start_pos == nullptr || kv_len == 0 || chunk_base < 32 || chunk_base % 32 != 0 || partials == nullptr ||
      (key_block_bytes != 34 && key_block_bytes != 18) || (value_block_bytes != 34 && value_block_bytes != 18)) {
    return false;
  }
  const std::uint32_t qpg = heads / kv_heads;
  const std::uint32_t splits = RowChunkGridSplits(kv_len, n_tokens, chunk_base);
  if (splits > 256 || partials_bytes < RowChunkPartialBytes(n_tokens, heads, splits, kHeadDim)) return false;
  bool launched = true;
  // d == 256 with 34-byte blocks: the multi-wave kernel (MBUFO_SPLITK_ROWCHUNK_V1=1 keeps the one-wave GQA kernel for A/B).
  const bool mw = kHeadDim == 256 && key_block_bytes == 34 && value_block_bytes == 34 &&
                  std::getenv("MBUFO_SPLITK_ROWCHUNK_V1") == nullptr;
  if (mw) {
#define MBUFO_MW_CASE(Q)                                                                                        \
  case Q:                                                                                                       \
    LaunchRowChunkMw<Q>(q, k_cache, v_cache, partials, start_pos, heads, kv_heads, n_tokens, score_scale, splits, \
                        chunk_base, stream);                                                                    \
    break;
    switch (qpg) {
      MBUFO_MW_CASE(1)
      MBUFO_MW_CASE(2)
      MBUFO_MW_CASE(4)
      MBUFO_MW_CASE(6)
      MBUFO_MW_CASE(8)
      default:
        launched = false;
    }
#undef MBUFO_MW_CASE
    if (!launched) return false;
    hipLaunchKernelGGL((AttentionBlockedSplitKMergeKernel<kHeadDim, true>), dim3(heads, n_tokens), dim3(kThreads), 0,
                       stream, partials, out, heads, splits, start_pos, chunk_base);
    return true;
  }
#define MBUFO_RC_CASE(Q)                                                                                            \
  case Q:                                                                                                          \
    LaunchGqa<kHeadDim, Q, true>(q, k_cache, v_cache, partials, start_pos, heads, kv_heads, n_tokens, score_scale, \
                                 splits, chunk_base, key_block_bytes, value_block_bytes, stream);                  \
    break;
  switch (qpg) {
    MBUFO_RC_CASE(1)
    MBUFO_RC_CASE(2)
    MBUFO_RC_CASE(4)
    MBUFO_RC_CASE(8)
    default:
      launched = false;
  }
#undef MBUFO_RC_CASE
  if (!launched) return false;
  hipLaunchKernelGGL((AttentionBlockedSplitKMergeKernel<kHeadDim, true>), dim3(heads, n_tokens), dim3(kThreads), 0, stream,
                     partials, out, heads, splits, start_pos, chunk_base);
  return true;
}

template bool AttentionBlockedSplitKRowChunk<KvCodecQ8, 256>(
    const float*, const std::uint8_t*, const std::uint8_t*, float*, std::uint32_t, const std::uint32_t*, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, hipStream_t, float, std::uint32_t, std::uint32_t, float*, std::size_t,
    std::uint32_t);
template bool AttentionBlockedSplitKRowChunk<KvCodecQ8, 128>(
    const float*, const std::uint8_t*, const std::uint8_t*, float*, std::uint32_t, const std::uint32_t*, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, hipStream_t, float, std::uint32_t, std::uint32_t, float*, std::size_t,
    std::uint32_t);


template <typename Codec, std::uint32_t kHeadDim>
bool AttentionBlockedSinkRowChunk(const float* q, const std::uint8_t* k_cache, const std::uint8_t* v_cache, const float* sinks,
                                  float* out, std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t heads,
                                  std::uint32_t kv_heads, std::uint32_t d, std::uint32_t window, hipStream_t stream,
                                  float score_scale, float* partials, std::size_t partials_bytes, std::uint32_t chunk_base) {
  static_assert(kHeadDim == 64 || kHeadDim == 128, "sink row-chunk decode is d=64/128 only");
  if (kHeadDim != d || heads == 0 || kv_heads == 0 || heads % kv_heads != 0 || n_tokens == 0 ||
      n_tokens > kBlockedSplitKMaxTokens || q == nullptr || k_cache == nullptr || v_cache == nullptr || out == nullptr ||
      partials == nullptr || chunk_base < 32 || chunk_base % 32 != 0) {
    return false;
  }
  const std::uint32_t qpg = heads / kv_heads;
  if (qpg != 1 && qpg != 2 && qpg != 4 && qpg != 8) return false;
  // Grid depth: the largest split count over the rows start_pos + 1 .. start_pos + n_tokens.
  std::uint32_t splits = 1;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t n_kv = start_pos + t + 1u;
    const std::uint32_t span = n_kv - ((window != 0 && n_kv > window) ? n_kv - window : 0u);
    const std::uint32_t sp = RowChunkSplits(span, chunk_base);
    if (sp > splits) splits = sp;
  }
  if (splits > 256 || partials_bytes < RowChunkPartialBytes(n_tokens, heads, splits, kHeadDim)) return false;
#define MBUFO_SINK_CASE(Q)                                                                                                \
  case Q:                                                                                                                \
    hipLaunchKernelGGL((AttentionBlockedSinkRowChunkKernel<kHeadDim, Q>), dim3(kv_heads, n_tokens, splits),            \
                       dim3(32 * kMwWaves), 0, stream, q, k_cache, v_cache, sinks, partials, start_pos, heads, kv_heads, \
                       window, score_scale, splits, chunk_base);                                                         \
    break;
  switch (qpg) {
    MBUFO_SINK_CASE(1)
    MBUFO_SINK_CASE(2)
    MBUFO_SINK_CASE(4)
    MBUFO_SINK_CASE(8)
    default:
      return false;
  }
#undef MBUFO_SINK_CASE
  hipLaunchKernelGGL((AttentionBlockedSinkMergeKernel<kHeadDim>), dim3(heads, n_tokens), dim3(kHeadDim), 0, stream, partials,
                     out, heads, splits, start_pos, window, chunk_base);
  return true;
}

template bool AttentionBlockedSinkRowChunk<KvCodecQ8, 64>(const float*, const std::uint8_t*, const std::uint8_t*,
                                                          const float*, float*, std::uint32_t, std::uint32_t, std::uint32_t,
                                                          std::uint32_t, std::uint32_t, std::uint32_t, hipStream_t, float,
                                                          float*, std::size_t, std::uint32_t);
template bool AttentionBlockedSinkRowChunk<KvCodecQ8, 128>(const float*, const std::uint8_t*, const std::uint8_t*,
                                                           const float*, float*, std::uint32_t, std::uint32_t, std::uint32_t,
                                                           std::uint32_t, std::uint32_t, std::uint32_t, hipStream_t, float,
                                                           float*, std::size_t, std::uint32_t);

template bool AttentionBlockedSplitK<KvCodecQ8, 256>(
    const float*, const std::uint8_t*, const std::uint8_t*, float*,
    std::uint32_t, const std::uint32_t*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, hipStream_t, float, std::uint32_t,
    std::uint32_t, float*, std::size_t);

template bool AttentionBlockedSplitK<KvCodecQ8, 128>(
    const float*, const std::uint8_t*, const std::uint8_t*, float*,
    std::uint32_t, const std::uint32_t*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, hipStream_t, float, std::uint32_t,
    std::uint32_t, float*, std::size_t);

}  // namespace mbufo::core::compute::blocked_splitk
#endif  // ENGINE_ENABLE_HIP
