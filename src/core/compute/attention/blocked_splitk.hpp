#ifndef MBUFO_CORE_COMPUTE_ATTENTION_BLOCKED_SPLITK_HPP_
#define MBUFO_CORE_COMPUTE_ATTENTION_BLOCKED_SPLITK_HPP_

// Blocked-Q8 Split-K decode attention for d == 256 (SWA) and d == 128
// (dense) layers: the same
// shared 34-byte blocked KV caches as the scalar direct read (32x int8 codes
// + f16 per-32-block scale, dequantized in registers), but the visible KV
// range is cut into K chunks scanned by parallel blocks with per-chunk
// online-softmax partials and a final deterministic log-sum-exp merge — the
// split philosophy of gemma4's F16 AttentionSplitK(Multi)Kernel, retargeted
// at the narrow-batch decode case where WMMA tiles stay empty and the scalar
// single-block sweep leaves most CUs idle.
//
// SWA window semantics are inherited from the caller: the engine applies the
// same pointer-shift/rebase as the scalar AttentionBlocked path before
// calling, so this kernel always sees dense-causal keys [0, *start_pos].
//
// This header is host-safe (includable from plain C++ TUs): it carries only
// layout tags, tuning constants and the launcher declaration. The unified
// element codec lives in blocked_kv_codec.hpp.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/compute/attention/blocked_kv_codec.hpp"

#include <cstddef>
#include <cstdint>

namespace mbufo::core::compute::blocked_splitk {

// Single definition lives in blocked_kv_codec.hpp (same layout and codec
// math as every other consumer); alias kept so existing template arguments
// (blocked_splitk::KvCodecQ8) keep compiling.
using KvCodecQ8 = blocked_kv::KvCodecQ8;

/// Upper token bound of the Split-K decode path (mirrors the F16 Split-K
/// narrow-batch shape: decode and other single-digit forwards). Longer
/// forwards stay on WMMA (prefill) or the scalar direct read.
inline constexpr std::uint32_t kBlockedSplitKMaxTokens = 8;

/// Minimum visible KV length that pays off: shorter contexts stay on the
/// scalar single-block scan, whose launch + merge overhead wins below this.
/// (1024 for the legacy per-q-head kernel; the GQA-grouped kernel wins from ~256.)
inline constexpr std::uint32_t kBlockedSplitKMinKv = 256;

/// Maximum KV splits (matches the F16 Split-K cap).
inline constexpr std::uint32_t kBlockedSplitKMaxSplits = 64;

/// Device scratch for the split-K partials: MaxTokens x 32 heads x
/// 64 splits x (256+2) floats, which also covers 32 heads x (128+2)
/// (32*130 < 32*258). The launcher takes it from the caller arena
/// (§4.2 Slot-concurrency iron rule: no `static` buffers inside operators);
/// null or short scratch refuses cleanly so the caller falls back.
[[nodiscard]] inline constexpr std::size_t
AttentionBlockedSplitKScratchBytes() noexcept {
  return 8ull * 32ull * 64ull * (256ull + 2ull) * sizeof(float);
}

/// Split-K decode over blocked Q8 caches (d == 256 SWA, heads <= 16;
/// d == 128 dense, heads <= 32). `k_cache` /
/// `v_cache` and `start_pos` must already carry the SWA window
/// pointer-shift/rebase (identical to the AttentionBlocked call: keys are
/// dense-causal [0, *start_pos]). `kv_len` is the host-visible key count
/// (rebase position + n_tokens) used to size the split; the kernel clamps
/// each chunk to its own device-side `*start_pos + t + 1`. `score_scale`
/// multiplies the QK scores (gemma4 passes 1.0, its f_attention_scale).
/// `key_block_bytes` / `value_block_bytes` select each side's block layout
/// (34 = 8-bit codes, 18 = packed 4-bit codes for the MBUFO_KV_Q4/V4 grids).
/// `partials` is caller-owned device scratch of at least
/// AttentionBlockedSplitKScratchBytes() (§4.2). Instantiates <KvCodecQ8, 256>
/// and <KvCodecQ8, 128>; other shapes return false so the caller falls back
/// to the scalar direct read. Never launches an empty grid. The merge folds
/// chunks in fixed order, so reruns are deterministic.
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
                            std::size_t partials_bytes);

/// Row-chunk Split-K decode (GQA-grouped kernel only: heads / kv_heads in {1,2,4,8}). The split of one query row is a
/// function of that row's own length alone: chunk = max(chunk_base, ceil(n_kv / 32) rounded up to 32) keys per split,
/// ceil(n_kv / chunk) splits (about 32 once n_kv > 32 * chunk_base, at most 32 + 1), with n_kv = *start_pos + t + 1 read on the device. A row's result therefore does
/// not depend on the batch it is computed in, on `kv_len`, or on the grid: a speculative-verify row equals the same row
/// decoded on its own, and one graph captured at a short length replays correctly while every row still fits in one chunk
/// (n_kv <= chunk_base). `kv_len` is the host-visible length of the LAST row (start + n_tokens); `partials` needs
/// n_tokens x heads x splits x (d + 2) floats (RowChunkPartialBytes). No minimum length (a short row is one chunk).
inline constexpr std::uint32_t kRowChunkTargetSplits = 32;  // measured optimum at 6K..30K (bench: tools/bench/attn_check)
[[nodiscard]] inline constexpr std::uint32_t RowChunkKeys(std::uint32_t n_kv, std::uint32_t chunk_base) noexcept {
  const std::uint32_t c = ((n_kv + kRowChunkTargetSplits - 1u) / kRowChunkTargetSplits + 31u) / 32u * 32u;
  return c > chunk_base ? c : chunk_base;
}
[[nodiscard]] inline constexpr std::uint32_t RowChunkSplits(std::uint32_t n_kv, std::uint32_t chunk_base) noexcept {
  const std::uint32_t c = RowChunkKeys(n_kv, chunk_base);
  return (n_kv + c - 1u) / c;
}
/// Largest split count over the rows `kv_len - n_tokens + 1 .. kv_len` (the grid depth of one launch).
[[nodiscard]] inline constexpr std::uint32_t RowChunkGridSplits(std::uint32_t kv_len, std::uint32_t n_tokens,
                                                                 std::uint32_t chunk_base) noexcept {
  std::uint32_t z = 1;
  for (std::uint32_t t = 0; t < n_tokens && t < kv_len; ++t) {
    const std::uint32_t sp = RowChunkSplits(kv_len - t, chunk_base);
    if (sp > z) z = sp;
  }
  return z;
}
[[nodiscard]] inline constexpr std::size_t RowChunkPartialBytes(std::uint32_t n_tokens, std::uint32_t heads,
                                                                 std::uint32_t splits, std::uint32_t d) noexcept {
  return static_cast<std::size_t>(n_tokens) * heads * splits * (d + 2u) * sizeof(float);
}
template <typename Codec, std::uint32_t kHeadDim>
bool AttentionBlockedSplitKRowChunk(const float* q, const std::uint8_t* k_cache, const std::uint8_t* v_cache, float* out,
                                    std::uint32_t n_tokens, const std::uint32_t* start_pos, std::uint32_t kv_len,
                                    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d, hipStream_t stream,
                                    float score_scale, std::uint32_t key_block_bytes, std::uint32_t value_block_bytes,
                                    float* partials, std::size_t partials_bytes, std::uint32_t chunk_base);

/// Row-chunk decode with an attention sink and an optional sliding window, reading the blocked Q8 cache directly (gpt-oss:
/// d == 64, 64 heads / 8 kv heads, window 128 on half of the layers; d == 128 also instantiated). `sinks` is [heads] F32 and
/// may be null (no sink term); the row's visible span is [max(0, n_kv - window), n_kv) with n_kv = start_pos + t + 1 (window
/// 0 = dense causal) and its split depends only on the span length (RowChunkKeys), so rows do not depend on the batch.
/// `partials` needs RowChunkPartialBytes(n_tokens, heads, splits, d) bytes. Returns false (nothing launched) outside the
/// contract (heads / kv_heads in {1,2,4,8}, n_tokens <= 8).
template <typename Codec, std::uint32_t kHeadDim>
bool AttentionBlockedSinkRowChunk(const float* q, const std::uint8_t* k_cache, const std::uint8_t* v_cache, const float* sinks,
                                  float* out, std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t heads,
                                  std::uint32_t kv_heads, std::uint32_t d, std::uint32_t window, hipStream_t stream,
                                  float score_scale, float* partials, std::size_t partials_bytes, std::uint32_t chunk_base);

}  // namespace mbufo::core::compute::blocked_splitk

#endif  // MBUFO_CORE_COMPUTE_ATTENTION_BLOCKED_SPLITK_HPP_
