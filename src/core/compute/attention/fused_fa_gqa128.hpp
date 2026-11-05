#ifndef GUFO_CORE_COMPUTE_ATTENTION_FUSED_FA_GQA128_HPP_
#define GUFO_CORE_COMPUTE_ATTENTION_FUSED_FA_GQA128_HPP_

// Fused GQA flash-attention prefill for the (16Q, 8KV, d=128) dense shape
// (e.g. Qwen3-Embedding-0.6B): online softmax with KV tiling -- the Q tile
// stays resident while K/V stream through in blocks -- F16 KV planes, F32
// accumulate. Lives in core (not model-side); the attention dispatcher
// routes this exact geometry here, everything else falls back to legacy.
//
// Geometry served: heads == 16, kv_heads == 8, head_dim == 128,
// dense-causal, any batch >= 1 with start_pos + batch <= max_context.
// Q/K/V are FP32 rows; K/V also land in the canonical F16 cache planes
// (packed unless skip_kv_write, mirroring the tiled prefill contract).
// `gate`, when non-null, applies the SiLU gate multiply; dense decoders
// pass nullptr and skip it. Stream-ordered.
//
// This header is host-safe: launcher declaration only.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::core::compute::fused_fa_gqa128 {

/// Exact geometry this kernel serves. The launcher returns false otherwise
/// and the caller keeps its existing fallback chain (never partially run).
[[nodiscard]] constexpr bool IsFusedFaGqa128Supported(
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t q_len = 1, std::uint32_t kv_len = 1) noexcept {
  return heads == 16 && kv_heads == 8 && (heads % kv_heads) == 0 &&
         head_dim == 128 && q_len != 0 && kv_len != 0 && kv_len >= q_len;
}

/// Forward pass. `scale` multiplies QK (typically 1/sqrt(head_dim)).
/// Returns false (no launch) on unsupported geometry or null pointers.
[[nodiscard]] bool LaunchFusedFaGqa128(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream, bool skip_kv_write = false);

}  // namespace gufo::core::compute::fused_fa_gqa128

#endif  // GUFO_CORE_COMPUTE_ATTENTION_FUSED_FA_GQA128_HPP_
