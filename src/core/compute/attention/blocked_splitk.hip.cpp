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

namespace gufo::core::compute::blocked_splitk {
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

// Device-side decode traits for the blocked KV codecs (the header stays
// host-safe, so the bodies live here). Q8 is the symmetric INT8 grid over
// amax/127, bit-compatible with blocked_kv::KvCodecQ8.
template <typename Codec>
struct CodecTraits;
template <>
struct CodecTraits<KvCodecQ8> {
  static __device__ __forceinline__ float Decode(std::uint8_t e) {
    return static_cast<float>(static_cast<std::int8_t>(e));
  }
};

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
  // 4-bit nibbles for the GUFO_KV_Q4/V4 grids).
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
          kval = CodecTraits<Codec>::Decode(kblk[lane_off]) * sk;
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
          vval = CodecTraits<Codec>::Decode(vblk[lane_off]) * sv;
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
template <std::uint32_t kHeadDim>
__global__ void AttentionBlockedSplitKMergeKernel(
    const float* __restrict__ partials, float* __restrict__ out,
    std::uint32_t heads, std::uint32_t splits) {
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
  float m = -INFINITY;
  for (std::uint32_t s = 0; s < splits; ++s) {
    m = fmaxf(m, base[s * (kHeadDim + 2)]);
  }
  float l = 0.0f;
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < splits; ++s) {
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
    const float sc = __half2float(*reinterpret_cast<const __half*>(blk + 32));
    const unsigned short* p = reinterpret_cast<const unsigned short*>(blk);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const unsigned u = p[i];
      out32[2 * i] = static_cast<float>(static_cast<std::int8_t>(u & 0xFFu)) * sc;
      out32[2 * i + 1] =
          static_cast<float>(static_cast<std::int8_t>(u >> 8)) * sc;
    }
  } else {
    const float sc = __half2float(*reinterpret_cast<const __half*>(blk + 16));
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
    const float sc = __half2float(*reinterpret_cast<const __half*>(blk + 32));
    const unsigned u = *reinterpret_cast<const unsigned short*>(blk + off);
    return make_float2(
        static_cast<float>(static_cast<std::int8_t>(u & 0xFFu)) * sc,
        static_cast<float>(static_cast<std::int8_t>(u >> 8)) * sc);
  }
  const float sc = __half2float(*reinterpret_cast<const __half*>(blk + 16));
  const unsigned byte = blk[off >> 1];
  const int lo = static_cast<int>(byte & 0xFu);
  const int hi = static_cast<int>(byte >> 4);
  return make_float2(static_cast<float>(lo - ((lo & 8) ? 16 : 0)) * sc,
                     static_cast<float>(hi - ((hi & 8) ? 16 : 0)) * sc);
}

template <std::uint32_t kD, std::uint32_t kQ>
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
  const std::uint32_t k_beg = s * chunk;
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

template <std::uint32_t kD, std::uint32_t kQ>
void LaunchGqa(const float* q, const std::uint8_t* k_cache,
               const std::uint8_t* v_cache, float* partials,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t n_tokens, float score_scale,
               std::uint32_t splits, std::uint32_t chunk,
               std::uint32_t key_block_bytes, std::uint32_t value_block_bytes,
               hipStream_t stream) {
  hipLaunchKernelGGL((AttentionBlockedSplitKGqaKernel<kD, kQ>),
                     dim3(kv_heads, n_tokens, splits), dim3(32), 0, stream, q,
                     k_cache, v_cache, partials, start_pos, heads, kv_heads,
                     score_scale, splits, chunk, key_block_bytes,
                     value_block_bytes);
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
#define GUFO_GQA_CASE(Q)                                                        \
  case Q:                                                                       \
    LaunchGqa<kD, Q>(q, k_cache, v_cache, partials, start_pos, heads, kv_heads, \
                     n_tokens, score_scale, splits, chunk, key_block_bytes,     \
                     value_block_bytes, stream);                                \
    return true;
  switch (qpg) {
    GUFO_GQA_CASE(1)
    GUFO_GQA_CASE(2)
    GUFO_GQA_CASE(4)
    GUFO_GQA_CASE(8)
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
#undef GUFO_GQA_CASE
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
  // GQA-grouped lane-per-key kernel (default); GUFO_BLOCKED_SPLITK_LEGACY=1
  // restores the per-q-head kernel below for A/B. Groups the grouped kernel does
  // not cover (heads / kv_heads outside {1,2,4,8,16}, {1,2,4,8} for d == 256)
  // fall through to the legacy kernel.
  const std::uint32_t qpg = heads / kv_heads;
  const bool gqa_ok = std::getenv("GUFO_BLOCKED_SPLITK_LEGACY") == nullptr &&
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

}  // namespace gufo::core::compute::blocked_splitk
#endif  // ENGINE_ENABLE_HIP
