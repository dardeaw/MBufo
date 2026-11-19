#ifndef GUFO_CORE_COMPUTE_ATTENTION_FLASH_WMMA_GQA128_HPP_
#define GUFO_CORE_COMPUTE_ATTENTION_FLASH_WMMA_GQA128_HPP_

// WMMA (16x16x16 F16, F32 accumulate) flash-attention prefill for head_dim == 128 with any even GQA ratio,
// F16 KV planes and causal masking with an optional sliding window. See flash_wmma_gqa128.hip.cpp.
//
// Layout: q F32 [n_tokens][heads][128]; k_cache / v_cache F16 [kv_len][kv_heads][128] holding every key from
// position 0 (kv_len >= start_pos + n_tokens; entries past that are never read); out F32 [n_tokens][heads][128].
// Query t sits at global position start_pos + t. `score_scale` multiplies QK (typically 1/sqrt(128)).
// window == 0 is dense causal; otherwise key kc is visible to a query at position qp iff kc <= qp and
// kc + window > qp. Callers with blocked-Q8 KV dequantize the visible range into F16 planes first.
//
// Host-safe header: launcher declaration only.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::core::compute::flash_wmma_gqa128 {

/// A block serves a PAIR of query heads inside one KV group, so heads / kv_heads must be even.
[[nodiscard]] constexpr bool IsFlashWmmaGqa128Supported(std::uint32_t heads, std::uint32_t kv_heads,
                                                        std::uint32_t head_dim) noexcept {
  return head_dim == 128 && heads != 0 && kv_heads != 0 && heads % kv_heads == 0 && (heads / kv_heads) % 2 == 0;
}

/// Returns false (nothing launched) on unsupported geometry, null pointers or n_tokens == 0; the caller keeps
/// its existing path. Stream-ordered.
[[nodiscard]] bool FlashWmmaGqa128Prefill(const float* q, const __half* k_cache, const __half* v_cache,
                                          float* out, std::uint32_t n_tokens, std::uint32_t start_pos,
                                          std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t head_dim,
                                          std::uint32_t window, float score_scale, hipStream_t stream);

}  // namespace gufo::core::compute::flash_wmma_gqa128

#endif  // GUFO_CORE_COMPUTE_ATTENTION_FLASH_WMMA_GQA128_HPP_
