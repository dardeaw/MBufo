#ifndef GUFO_CORE_MODEL_CONFIG_HPP_
#define GUFO_CORE_MODEL_CONFIG_HPP_

#include <cstdint>
#include <string>
#include <string_view>

namespace gufo::core {

/// Unified architectural configuration parsed dynamically from GGUF metadata or
/// JSON.
struct ModelConfig {
  std::string architecture{"qwen35"};
  std::string model_name{"qwen3.5-4b-text"};
  std::uint32_t num_layers{36};  ///< Main transformer layers, excluding MTP.
  std::uint32_t hidden_size{2560};
  std::uint32_t intermediate_size{9728};  ///< 17408 for Qwen3.8-27B.
  std::uint32_t num_attention_heads{16};
  std::uint32_t num_key_value_heads{4};
  std::uint32_t head_dim{256};
  std::uint32_t vocab_size{248320};  ///< 248320 for Qwen3.5/3.8
  std::uint32_t context_length{32768};
  std::uint32_t full_attention_interval{
      4};  ///< 3 linear attention + 1 full attention
  std::uint32_t mtp_num_layers{1};
  std::uint32_t ssm_conv_kernel{4};
  std::uint32_t ssm_state_size{128};
  std::uint32_t ssm_group_count{16};
  std::uint32_t ssm_time_step_rank{32};
  std::uint32_t ssm_inner_size{4096};
  std::uint32_t rotary_dim{64};
  float rope_theta{10000000.0F};
  float rope_scale{1.0F};
  // mRoPE dimension sections (qwen35moe text: [11, 11, 10, 0]); all zeros
  // selects the legacy single-section rotary path.
  std::uint32_t rope_sections[4]{0, 0, 0, 0};
  // Routed-MoE geometry (qwen35moe: 256 experts, top-8, expert_ff 512,
  // shared expert 512). Zero expert_count selects the dense FFN path.
  std::uint32_t expert_count{0};
  std::uint32_t expert_used_count{0};
  std::uint32_t expert_ffn_size{0};
  std::uint32_t shared_expert_ffn_size{0};
  bool is_text_only{true};

  [[nodiscard]] constexpr std::uint32_t AttentionSize() const noexcept {
    return num_attention_heads * head_dim;
  }

  [[nodiscard]] constexpr std::uint32_t FullAttentionLayerCount()
      const noexcept {
    return full_attention_interval == 0 ? 0
                                        : num_layers / full_attention_interval;
  }

  [[nodiscard]] constexpr std::uint32_t SsmLayerCount() const noexcept {
    return num_layers - FullAttentionLayerCount();
  }

  /// Dense state index for a recurrent transformer layer.
  [[nodiscard]] constexpr std::uint32_t SsmLayerIndex(
      std::uint32_t layer) const noexcept {
    return full_attention_interval == 0
               ? layer
               : layer - layer / full_attention_interval;
  }

  [[nodiscard]] constexpr std::uint32_t SsmValueSize() const noexcept {
    return ssm_time_step_rank == 0 ? 0 : ssm_inner_size / ssm_time_step_rank;
  }

  [[nodiscard]] constexpr std::uint32_t SsmQkvSize() const noexcept {
    return (2 * ssm_group_count * ssm_state_size) + ssm_inner_size;
  }

  /// Returns true if this configuration conforms to the Qwen3.5/3.8 repeating
  /// block structure.
  [[nodiscard]] constexpr bool IsValidQwen() const noexcept {
    return head_dim > 0 && num_layers > 0 && hidden_size > 0 &&
           intermediate_size > 0 && num_attention_heads > 0 &&
           num_key_value_heads > 0 && vocab_size > 0 &&
           full_attention_interval == 4 &&
           (num_layers % full_attention_interval) == 0 && rotary_dim > 0 &&
           rotary_dim <= head_dim && (rotary_dim % 2) == 0 &&
           ssm_conv_kernel == 4 && ssm_state_size == 128 &&
           ssm_group_count > 0 && ssm_time_step_rank > 0 &&
           ssm_inner_size > 0 && (ssm_inner_size % ssm_time_step_rank) == 0 &&
           SsmValueSize() == 128 && is_text_only;
  }

  /// Returns true for a dense full-attention decoder (e.g. Qwen3-Embedding,
  /// arch "qwen3"): every layer is causal attention, the hybrid interval
  /// collapses to 1, and there is no recurrent state. The shared RMSNorm /
  /// RoPE / SwiGLU prefill kernels serve these layers unchanged; only the
  /// hybrid-only geometry gates need to admit the config.
  [[nodiscard]] constexpr bool IsDenseQwenAttention() const noexcept {
    return head_dim > 0 && num_layers > 0 && hidden_size > 0 &&
           intermediate_size > 0 && num_attention_heads > 0 &&
           num_key_value_heads > 0 && vocab_size > 0 &&
           full_attention_interval == 1 && rotary_dim > 0 &&
           rotary_dim <= head_dim && (rotary_dim % 2) == 0 && is_text_only;
  }

  /// True when the FFN block is a routed MoE (experts + shared expert).
  [[nodiscard]] constexpr bool IsMoE() const noexcept {
    return expert_count > 0;
  }

  /// MoE twin of IsValidQwen: the same hybrid attention/SSM block rules,
  /// plus routed-expert geometry. Dense `intermediate_size` is unused on
  /// MoE layers (no feed_forward_length key in qwen35moe GGUFs).
  [[nodiscard]] constexpr bool IsValidQwenMoe() const noexcept {
    return head_dim > 0 && num_layers > 0 && hidden_size > 0 &&
           num_attention_heads > 0 && num_key_value_heads > 0 &&
           vocab_size > 0 && full_attention_interval == 4 &&
           (num_layers % full_attention_interval) == 0 && rotary_dim > 0 &&
           rotary_dim <= head_dim && (rotary_dim % 2) == 0 &&
           ssm_conv_kernel == 4 && ssm_state_size == 128 &&
           ssm_group_count > 0 && ssm_time_step_rank > 0 &&
           ssm_inner_size > 0 && (ssm_inner_size % ssm_time_step_rank) == 0 &&
           SsmValueSize() == 128 && is_text_only && expert_count > 0 &&
           expert_count <= 1024 && expert_used_count > 0 &&
           expert_used_count <= expert_count && expert_used_count <= 32 &&
           expert_ffn_size > 0 && shared_expert_ffn_size > 0;
  }
};

}  // namespace gufo::core

#endif  // GUFO_CORE_MODEL_CONFIG_HPP_
