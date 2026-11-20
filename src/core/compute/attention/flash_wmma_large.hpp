#ifndef MBUFO_CORE_COMPUTE_ATTENTION_FLASH_WMMA_LARGE_HPP_
#define MBUFO_CORE_COMPUTE_ATTENTION_FLASH_WMMA_LARGE_HPP_

// WMMA (16x16x16 F16, F32 accumulate) flash-attention prefill for head_dim 256 or 512 with ANY heads / kv_heads ratio
// (1 = MHA, 2.. = GQA), F16 KV planes, causal masking and an optional sliding window. Companion of flash_wmma_gqa (head_dim
// 64 / 128); see flash_wmma_large.hip.cpp for how wide heads are split over waves.
//
// Layout: q F32 [n_tokens][heads][head_dim]; k_cache / v_cache F16 [kv_len][kv_heads][head_dim] holding every key from
// position 0 (kv_len >= start_pos + n_tokens; entries past that are never read); out F32 like q. Query t sits at global
// position start_pos + t. `score_scale` multiplies QK (Gemma 4: 1.0, Qwen: 1/sqrt(head_dim)). window == 0 is dense causal;
// otherwise key kc is visible to a query at position qp iff kc <= qp and kc + window > qp.
//
// Host-safe header: launcher declaration only.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace mbufo::core::compute::flash_wmma_large {

/// A block serves kHeadWaves heads of one KV group (or consecutive query blocks when the group is smaller), so any
/// heads % kv_heads == 0 works; head_dim must be 256 or 512.
[[nodiscard]] constexpr bool IsFlashWmmaLargeSupported(std::uint32_t heads, std::uint32_t kv_heads,
                                                       std::uint32_t head_dim) noexcept {
  return (head_dim == 256 || head_dim == 512) && heads != 0 && kv_heads != 0 && heads % kv_heads == 0;
}

/// Returns false (nothing launched) on unsupported geometry, null pointers or n_tokens == 0; the caller keeps its existing
/// path. Stream-ordered.
[[nodiscard]] bool FlashWmmaLargePrefill(const float* q, const __half* k_cache, const __half* v_cache, float* out,
                                         std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t heads,
                                         std::uint32_t kv_heads, std::uint32_t head_dim, std::uint32_t window,
                                         float score_scale, hipStream_t stream);

/// Same kernel over the shared blocked-Q8 KV layout ([kv_len][kv_heads][head_dim / 32] blocks of 32 int8 codes + F16 scale,
/// 34 bytes; the codes are dequantized to F16 while the tile is loaded). `start_pos` is the HOST position of query row 0 and
/// the pointers address key 0 (no sliding-window pointer shift: `window` is applied exactly).
[[nodiscard]] bool FlashWmmaLargePrefillQ8(const float* q, const std::uint8_t* k_blocks, const std::uint8_t* v_blocks,
                                           float* out, std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t heads,
                                           std::uint32_t kv_heads, std::uint32_t head_dim, std::uint32_t window,
                                           float score_scale, hipStream_t stream);

}  // namespace mbufo::core::compute::flash_wmma_large

#endif  // MBUFO_CORE_COMPUTE_ATTENTION_FLASH_WMMA_LARGE_HPP_
