#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/blocked_q8.hpp"

#include <algorithm>
#include <stdexcept>

#include "src/core/compute/attention/blocked_kv.hpp"
#include "src/core/compute/attention/blocked_splitk.hpp"

namespace mbufo::hip {

bool ValidateQwenBlockedQ8Config(const core::ModelConfig& config,
                                 std::string* error) {
  if (!config.IsValidQwen() && !config.IsValidQwenMoe()) {
    if (error != nullptr)
      *error = "blocked-Q8 attention requires a valid Qwen3.5/3.8 config";
    return false;
  }
  if (config.FullAttentionLayerCount() == 0) {
    if (error != nullptr)
      *error = "blocked-Q8 attention requires attention layers";
    return false;
  }
  if (config.head_dim != 256) {
    if (error != nullptr)
      *error = "blocked-Q8 attention requires head_dim 256, found " +
               std::to_string(config.head_dim);
    return false;
  }
  if (config.num_key_value_heads == 0) {
    if (error != nullptr)
      *error = "blocked-Q8 attention requires nonzero kv heads";
    return false;
  }
  return true;
}

void WriteQwenBlockedKv(const float* k_rows, const float* v_rows,
                        std::uint8_t* blocked_k, std::uint8_t* blocked_v,
                        std::uint32_t n_tokens, std::uint32_t kv_heads,
                        std::uint32_t head_dim, const std::uint32_t* d_pos,
                        hipStream_t stream) {
  namespace blocked = core::compute::blocked_kv;
  if (k_rows == nullptr || v_rows == nullptr || blocked_k == nullptr ||
      blocked_v == nullptr || d_pos == nullptr) {
    throw std::invalid_argument(
        "blocked-Q8 KV write received a null buffer");
  }
  if (n_tokens == 0 || kv_heads == 0 || head_dim == 0) {
    throw std::invalid_argument("blocked-Q8 KV write received an empty shape");
  }
  const std::uint32_t kv_width = kv_heads * head_dim;
  blocked::QuantizeKvRows(k_rows, kv_width, head_dim, blocked_k, kv_width,
                          kv_heads, head_dim, n_tokens, blocked::KvCodec::kQ8,
                          kQwenBlockedQ8Range, kQwenBlockedQ8BlockBytes, d_pos,
                          stream);
  blocked::QuantizeKvRows(v_rows, kv_width, head_dim, blocked_v, kv_width,
                          kv_heads, head_dim, n_tokens, blocked::KvCodec::kQ8,
                          kQwenBlockedQ8Range, kQwenBlockedQ8BlockBytes, d_pos,
                          stream);
}

void ReadQwenBlockedAttention(const float* q, const std::uint8_t* blocked_k,
                              const std::uint8_t* blocked_v, const float* gate,
                              float* out, std::uint32_t n_tokens,
                              const std::uint32_t* d_pos, std::uint32_t heads,
                              std::uint32_t kv_heads, std::uint32_t head_dim,
                              hipStream_t stream, std::uint32_t kv_len,
                              float* splitk_partials,
                              std::size_t splitk_partial_bytes) {
  namespace blocked = core::compute::blocked_kv;
  if (q == nullptr || blocked_k == nullptr || blocked_v == nullptr ||
      out == nullptr || d_pos == nullptr) {
    throw std::invalid_argument(
        "blocked-Q8 attention read received a null buffer");
  }
  if (n_tokens == 0 || heads == 0 || kv_heads == 0 || head_dim == 0 ||
      (heads % kv_heads) != 0) {
    throw std::invalid_argument(
        "blocked-Q8 attention read received an invalid shape");
  }
  // The shared direct-read kernel is d == 256-only and returns silently
  // otherwise; fail loudly here instead of emitting an unnormalized row.
  if (head_dim != 256) {
    throw std::invalid_argument(
        "blocked-Q8 attention read requires head_dim 256");
  }
  // Dense causal prefix, like the F16 decode kernels: query t at absolute
  // position *d_pos + t attends keys [0, *d_pos + t]. The scale defaults to
  // 1/sqrt(d), matching the F16 decode kernels (score_scale 0 selects it).
  bool done = false;
  if (kv_len != 0 && splitk_partials != nullptr && head_dim == 256 && heads <= 32 && kv_len >= n_tokens) {
    // Row-chunk Split-K (core blocked_splitk): a row's split depends only on its own length, so the rows of a verify batch
    // equal the same rows decoded one at a time, and `kv_len` (the host-visible length of the LAST row) only sizes the grid.
    // Batches wider than the kernel's 8 rows run in groups. 1/sqrt(256) is exactly representable, so the score scale is the
    // value the single-block scan derives.
    done = true;
    for (std::uint32_t g = 0; g < n_tokens && done; g += 8) {
      const std::uint32_t cnt = std::min<std::uint32_t>(8, n_tokens - g);
      done = core::compute::blocked_splitk::AttentionBlockedSplitKRowChunk<core::compute::blocked_splitk::KvCodecQ8, 256>(
          q + static_cast<std::size_t>(g) * heads * head_dim, blocked_k, blocked_v,
          out + static_cast<std::size_t>(g) * heads * head_dim, cnt, d_pos + g, kv_len - (n_tokens - (g + cnt)), heads,
          kv_heads, head_dim, stream, 0.0625F, kQwenBlockedQ8BlockBytes, kQwenBlockedQ8BlockBytes, splitk_partials,
          splitk_partial_bytes, kQwenSplitKChunkBase);
    }
  }
  if (!done)
    blocked::AttentionKv(q, blocked_k, blocked_v, nullptr, 0, out, nullptr, 1,
                         n_tokens, d_pos, heads, kv_heads, head_dim, 1,
                         kQwenBlockedQ8BlockBytes, kQwenBlockedQ8BlockBytes,
                         blocked::KvCodec::kQ8, stream);
  if (gate != nullptr) {
    LaunchSigmoidGateMultiply(
        out, gate,
        static_cast<std::size_t>(n_tokens) * heads * head_dim, stream);
  }
}

}  // namespace mbufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
