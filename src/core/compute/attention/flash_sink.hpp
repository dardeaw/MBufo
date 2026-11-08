#ifndef GUFO_CORE_COMPUTE_ATTENTION_FLASH_SINK_HPP_
#define GUFO_CORE_COMPUTE_ATTENTION_FLASH_SINK_HPP_

// F16-KV flash attention with per-head sinks for head_dim == 64 (gpt-oss):
//   * FlashSinkPrefill: Q-blocked tiled prefill. Each block serves QB=8
//     queries of one head; K/V tiles stream once through LDS and are shared
//     by the block's queries (the scalar kernel re-reads KV per query).
//     Causal + optional sliding window + sink-init online softmax.
//   * FlashSinkDecode: split-K decode over the visible KV range with a fixed
//     order log-sum-exp merge (deterministic; ~1ulp vs single pass).
//
// Contract: d == 64, F16 KV planes [kv_len][kv_heads][d], F32 Q rows,
// F32 per-head sinks, GQA (heads % kv_heads == 0). Out-of-contract shapes,
// null pointers, or short scratch refuse cleanly (caller keeps its kernel).
// Prefill needs no scratch. Decode scratch sizing:
// FlashSinkDecodeScratchBytes(max_tokens, heads).
//
// This header is host-safe (includable from plain C++ TUs).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::core::compute::flash_sink {

/// Queries per prefill block.
inline constexpr std::uint32_t kFlashSinkQb = 8;
/// K/V tile rows per stage.
inline constexpr std::uint32_t kFlashSinkKt = 64;
/// Maximum decode splits.
inline constexpr std::uint32_t kFlashSinkMaxSplits = 32;
/// Maximum decode tokens per call (n == 1 in practice).
inline constexpr std::uint32_t kFlashSinkMaxTokens = 8;

/// Decode scratch: max_tokens x heads x splits x (d + 2) floats.
[[nodiscard]] inline constexpr std::size_t FlashSinkDecodeScratchBytes(
    std::uint32_t max_tokens, std::uint32_t heads) noexcept {
  return static_cast<std::size_t>(max_tokens) * heads *
         kFlashSinkMaxSplits * (64u + 2u) * sizeof(float);
}

/// Tiled prefill over F16 KV. q is [n][heads][d] F32, k/v caches are
/// [kv_len][kv_heads][d] F16, sinks [heads] F32, out [n][heads][d] F32.
/// start_pos is the global position of q row 0; kv_len is the host-visible
/// cache length (>= start_pos + n). window = 0 for dense causal.
bool FlashSinkPrefill(const float* q, const __half* k_cache,
                      const __half* v_cache, const float* sinks, float* out,
                      std::uint32_t n_tokens, std::uint32_t start_pos,
                      std::uint32_t kv_len, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t d,
                      std::uint32_t window, float score_scale,
                      hipStream_t stream);

/// Split-K decode over F16 KV with the same math. partials is caller-owned
/// device scratch of at least FlashSinkDecodeScratchBytes(n_tokens, heads).
bool FlashSinkDecode(const float* q, const __half* k_cache,
                     const __half* v_cache, const float* sinks, float* out,
                     std::uint32_t n_tokens, std::uint32_t start_pos,
                     std::uint32_t kv_len, std::uint32_t heads,
                     std::uint32_t kv_heads, std::uint32_t d,
                     std::uint32_t window, float score_scale,
                     hipStream_t stream, float* partials,
                     std::size_t partials_bytes);

}  // namespace gufo::core::compute::flash_sink

#endif  // GUFO_CORE_COMPUTE_ATTENTION_FLASH_SINK_HPP_
