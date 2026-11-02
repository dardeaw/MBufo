#ifndef GUFO_CORE_COMPUTE_ATTENTION_BLOCKED_KV_HPP_
#define GUFO_CORE_COMPUTE_ATTENTION_BLOCKED_KV_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/vision/rope.hpp"

/// Shared blocked-KV layout, element codecs and launchers, extracted verbatim
/// from the flash-next ROCm backend (see
/// src/models/qwen38_flash_next/kernels/rocm/kernels.hpp). The flash-next
/// header re-exports every symbol below, so existing callers keep compiling
/// with bitwise-identical behavior.
namespace gufo::core::compute::blocked_kv {

/// KV cache element format of the Flash-Next attention caches.
///
/// kF16 stores plain half rows of `AttentionKvDim()`. kQ8 and kFp8 store the
/// same blocked layout -- 32 elements followed by an f16 block scale, 34 bytes
/// -- and differ only in the element codec. Because the blocked layouts are
/// byte-compatible, allocation, prefill staging, attention addressing and
/// snapshots are shared; only encode/decode and the quantization range differ.
enum class KvFormat : std::uint32_t { kF16 = 0, kQ8 = 1, kFp8 = 2 };

/// Bytes of one 32-element blocked block. Keys always use the 8-bit layout
/// (32 codec bytes + an f16 scale). Values may pack 32 signed 4-bit codes
/// ahead of the f16 scale (16 bytes + 2) when the 4-bit value grid is on.
inline constexpr std::uint32_t kKvKeyBlockBytes = 34;
inline constexpr std::uint32_t kKvKey4BitBlockBytes = 18;
inline constexpr std::uint32_t kKvValue4BitBlockBytes = 18;

/// Bytes of one blocked row of `d` elements at `block_bytes` per 32-element
/// block; `d` need not be a multiple of 32.
[[nodiscard]] constexpr std::size_t KvBlockedRowBytes(
    std::uint32_t d, std::uint32_t block_bytes) noexcept {
  return ((static_cast<std::size_t>(d) + 31) / 32) * block_bytes;
}

/// True when `format` stores the shared blocked layout.
[[nodiscard]] constexpr bool KvBlocked(KvFormat format) noexcept {
  return format != KvFormat::kF16;
}

/// Element codec of a blocked layout: symmetric INT8 (amax/127) or OCP E4M3
/// (amax/448). Both store a per-32-block f16 scale.
enum class KvCodec { kQ8, kFp8 };

/// Element codec for a blocked `format`. Only call with `KvBlocked(format)`.
[[nodiscard]] constexpr KvCodec KvCodecOf(KvFormat format) noexcept {
  return format == KvFormat::kFp8 ? KvCodec::kFp8 : KvCodec::kQ8;
}

/// Blocked twin of PrepareAttention: identical math, K and V rows written as
/// `codec` blocks. K and V rows use `k_range` / `v_range` for their block
/// scale. `key_block_bytes` / `value_block_bytes` select each side's block
/// layout (34 = 8-bit codes, 18 = packed 4-bit codes). Plain Dense paths
/// quantize them separately via QuantizeKvRows over packed rows.
bool PrepareAttentionKv(
    const float* packed, std::uint32_t stride, const float* q_gamma,
    const float* k_gamma, float* q, float* gate, std::uint8_t* k_cache,
    std::uint8_t* v_cache, KvCodec codec, float k_range, float v_range,
    std::uint32_t key_block_bytes, std::uint32_t value_block_bytes,
    std::uint32_t n_tokens, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t d, std::uint32_t rotary_dim, const std::uint32_t* start_pos,
    float theta, float eps, hipStream_t stream,
    const gufo::models::qwen::vision::DeviceRope* rope = nullptr, bool prefill = false);

/// Symmetric quantize of float rows into `codec` blocks (see .hip): 32 elements
/// plus a per-32-block f16 scale. `codec` selects INT8 or E4M3 elements.
/// `range` is the quantization numerator (127 for INT8, 448 for E4M3); value
/// caches may pass a smaller range to study narrower value precision.
/// `block_bytes` is 34 for the 8-bit layout and 18 for packed 4-bit values.
void QuantizeKvRows(const float* src, std::uint32_t src_tok_stride,
                    std::uint32_t src_head_stride, std::uint8_t* dst,
                    std::uint32_t kv_width, std::uint32_t kv_heads,
                    std::uint32_t d, std::uint32_t n_tokens, KvCodec codec,
                    float range, std::uint32_t block_bytes,
                    const std::uint32_t* start_pos, hipStream_t stream);

/// Dual-buffer blocked -> F16 (one launch for K+V staging). Keys decode from
/// the 34-byte layout; `value_block_bytes` selects the value block layout
/// (34 = 8-bit codes, 18 = packed 4-bit codes), so `codec` selects how each
/// element decodes.
bool DequantizeKvRows(const std::uint8_t* src_k, __half* dst_k,
                      const std::uint8_t* src_v, __half* dst_v,
                      std::uint32_t n_rows, std::uint32_t row_elems,
                      std::uint32_t key_block_bytes,
                      std::uint32_t value_block_bytes, KvCodec codec,
                      hipStream_t stream);

/// Blocked twin of Attention: `codec` blocked caches, fused dequant on read.
/// `key_block_bytes` / `value_block_bytes` select each side's layout
/// (34 = 8-bit codes, 18 = packed 4-bit codes). `score_scale` multiplies the
/// QK scores; 0 selects the legacy 1/sqrt(d) (flash-next behavior).
/// Gemma 4 (f_attention_scale = 1.0) passes 1.0 explicitly.
void AttentionKv(const float* q, const std::uint8_t* k_cache,
                 const std::uint8_t* v_cache, const std::uint32_t* mask,
                 std::uint32_t mask_words, float* out, float* partials,
                 std::uint32_t splits, std::uint32_t n_tokens,
                 const std::uint32_t* start_pos, std::uint32_t heads,
                 std::uint32_t kv_heads, std::uint32_t d, std::uint32_t ratio,
                 std::uint32_t key_block_bytes, std::uint32_t value_block_bytes,
                 KvCodec codec, hipStream_t stream, float score_scale = 0.0F);

}  // namespace gufo::core::compute::blocked_kv

#endif  // GUFO_CORE_COMPUTE_ATTENTION_BLOCKED_KV_HPP_
