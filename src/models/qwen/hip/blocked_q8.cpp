#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/blocked_q8.hpp"

#include <stdexcept>

#include "src/core/compute/attention/blocked_kv.hpp"

namespace gufo::hip {

bool ValidateQwenBlockedQ8Config(const core::ModelConfig& config,
                                 std::string* error) {
  if (!config.IsValidQwen()) {
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
                              hipStream_t stream) {
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

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
