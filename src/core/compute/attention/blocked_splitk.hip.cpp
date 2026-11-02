// Blocked-Q8 Split-K decode attention (SWA d == 256 layers).
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
  qs[i] = q[(static_cast<std::size_t>(t) * heads + h) * kHeadDim + i];
  __syncthreads();
  const float qi = qs[i];
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
      if (j < k_end) {
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
  part[2 + i] = acc;
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

}  // namespace

// Split-K decode over blocked Q8 caches (SWA geometry only). The split
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
                            std::uint32_t value_block_bytes) {
  static_assert(kHeadDim == 256, "blocked Split-K decode is d=256 (SWA) only");
  if (kHeadDim != 256 || d != kHeadDim || heads == 0 || kv_heads == 0 ||
      heads > 16 || heads % kv_heads != 0 || n_tokens == 0 ||
      n_tokens > kBlockedSplitKMaxTokens || q == nullptr ||
      k_cache == nullptr || v_cache == nullptr || out == nullptr ||
      start_pos == nullptr || kv_len < kBlockedSplitKMinKv ||
      (key_block_bytes != 34 && key_block_bytes != 18) ||
      (value_block_bytes != 34 && value_block_bytes != 18)) {
    return false;
  }
  // Same split sizing as the F16 narrow-batch path: ~128 keys per chunk at
  // full GQA occupancy, 64-way cap.
  const std::uint32_t min_tile = (kv_heads * n_tokens >= 8u) ? 128u : 64u;
  std::uint32_t splits = (kv_len + min_tile - 1u) / min_tile;
  if (splits < 2u) {
    return false;
  }
  if (splits > kBlockedSplitKMaxSplits) {
    splits = kBlockedSplitKMaxSplits;
  }
  // Scratch for the gated maxima (MaxTokens x 16 heads x 64 splits x (d+2)),
  // allocated once like the F16 Split-K statics; failure falls back cleanly.
  static float* s_partial = nullptr;
  if (s_partial == nullptr) {
    constexpr std::size_t kMaxPartials = 8ull * 16ull * 64ull * (256ull + 2ull);
    void* p = nullptr;
    if (hipMalloc(&p, kMaxPartials * sizeof(float)) != hipSuccess) {
      return false;
    }
    s_partial = static_cast<float*>(p);
  }
  const std::uint32_t chunk = (kv_len + splits - 1u) / splits;
  hipLaunchKernelGGL((AttentionBlockedSplitKKernel<Codec, kHeadDim>),
                     dim3(heads, n_tokens, splits), dim3(kThreads), 0, stream,
                     q, k_cache, v_cache, s_partial, start_pos, heads, kv_heads,
                     score_scale, splits, chunk, key_block_bytes,
                     value_block_bytes);
  hipLaunchKernelGGL((AttentionBlockedSplitKMergeKernel<kHeadDim>),
                     dim3(heads, n_tokens), dim3(kThreads), 0, stream,
                     s_partial, out, heads, splits);
  return true;
}

template bool AttentionBlockedSplitK<KvCodecQ8, 256>(
    const float*, const std::uint8_t*, const std::uint8_t*, float*,
    std::uint32_t, const std::uint32_t*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, hipStream_t, float, std::uint32_t,
    std::uint32_t);

}  // namespace gufo::core::compute::blocked_splitk
#endif  // ENGINE_ENABLE_HIP
