#ifndef GUFO_MODELS_GLIMMER_CONFIG_HPP_
#define GUFO_MODELS_GLIMMER_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glimmer {

/// Architecture parameters of a `muse-glimmer` GGUF artifact (Muse Glimmer
/// 30B: dense GQA decoder with SWA layers, QK-norm, attention output gate,
/// post-attention/FFN norms, final-logit softcapping). Every value is read
/// from the file.
struct Config {
  std::uint32_t num_layers{0};        ///< muse-glimmer.block_count, 52.
  std::uint32_t hidden_size{0};       ///< muse-glimmer.embedding_length, 6656.
  std::uint32_t intermediate_size{0}; ///< muse-glimmer.feed_forward_length.
  std::uint32_t num_heads{0};         ///< muse-glimmer.attention.head_count, 32.
  std::uint32_t num_kv_heads{0};      ///< muse-glimmer.attention.head_count_kv, 2.
  std::uint32_t head_dim{0};          ///< muse-glimmer.attention.key_length, 128.
  std::uint32_t vocab_size{0};       ///< 202048 (from token_embd shape).
  std::uint32_t context_length{0};    ///< muse-glimmer.context_length, 131072.
  float rope_theta{500000.0F};        ///< muse-glimmer.rope.freq_base.
  float rope_theta_swa{500000.0F};    ///< muse-glimmer.rope.freq_base_swa (opt).
  float rms_eps{1e-5F};               ///< muse-glimmer.attention.layer_norm_rms_epsilon.
  float post_norm_eps{1e-5F};         ///< post-attention/FFN norm eps.
  float logit_scale{0.196116F};       ///< muse-glimmer.logit_scale.
  float final_logit_softcapping{20.0F};  ///< muse-glimmer.final_logit_softcapping.

  std::uint32_t sliding_window{0};    ///< muse-glimmer.attention.sliding_window.
  std::uint32_t sliding_period{0};    ///< muse-glimmer.attention.sliding_window_pattern.

  /// SWA membership: layers with (il % period) < period - 1 use the sliding
  /// window (and RoPE); the every-period-th layer is dense full attention
  /// without RoPE (llama.cpp set_swa_pattern semantics, dense_first=false).
  [[nodiscard]] bool LayerIsSwa(std::uint32_t layer) const noexcept {
    return sliding_period == 0 || (layer % sliding_period) < sliding_period - 1;
  }

  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::glimmer

#endif  // GUFO_MODELS_GLIMMER_CONFIG_HPP_
