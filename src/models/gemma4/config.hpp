#ifndef GUFO_MODELS_GEMMA4_CONFIG_HPP_
#define GUFO_MODELS_GEMMA4_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4 {

/// Architecture parameters of a `gemma4` GGUF artifact (Gemma 4 26B A4B).
/// Every value is read from the file.
struct Config {
  std::uint32_t num_layers{0};      ///< block_count, 30.
  std::uint32_t hidden_size{0};     ///< embedding_length, 2816.
  std::uint32_t vocab_size{0};      ///< 262144.
  std::uint32_t context_length{0};  ///< 262144.
  float rms_eps{1e-6F};

  std::uint32_t num_heads{0};       ///< attention.head_count, 16.
  std::vector<std::uint32_t> num_kv_heads;  ///< per-layer head_count_kv.
  std::uint32_t head_dim_full{0};   ///< attention.key_length, 512.
  std::uint32_t head_dim_swa{0};    ///< attention.key_length_swa, 256.
  std::uint32_t rotary_full{0};     ///< rope.dimension_count, 512.
  std::uint32_t rotary_swa{0};      ///< rope.dimension_count_swa, 256.
  float rope_theta_full{0.0F};      ///< rope.freq_base, 1e6.
  float rope_theta_swa{0.0F};       ///< rope.freq_base_swa, 1e4.

  std::uint32_t sliding_window{0};  ///< attention.sliding_window, 1024.
  std::uint32_t shared_kv_layers{0};  ///< attention.shared_kv_layers, 0 = every layer owns KV.
  std::vector<bool> is_swa;         ///< per-layer sliding-window flag.
  std::vector<std::uint32_t> head_dim_per_layer;  ///< from weight shapes.

  std::uint32_t num_experts{0};     ///< expert_count, 128.
  std::uint32_t experts_used{0};    ///< expert_used_count, 8.
  std::uint32_t expert_ff{0};       ///< expert_feed_forward_length, 704.
  std::uint32_t shared_ff{0};       ///< feed_forward_length, 2112.

  float final_logit_softcapping{0.0F};  ///< 30.0; 0 disables.

  [[nodiscard]] bool LayerIsSwa(std::uint32_t layer) const noexcept {
    return layer < is_swa.size() && is_swa[layer];
  }
  /// Layers that own a KV cache (llama.cpp has_kv: head_count_kv > 0).
  [[nodiscard]] bool LayerHasKv(std::uint32_t layer) const noexcept {
    return KvHeads(layer) > 0;
  }
  // Per-layer head dimensions are taken from the ACTUAL weight shapes
  // (attn_q rows / num_heads), not from the GGUF sliding_window_pattern,
  // whose polarity is inconsistent across Gemma 4 exports.
  [[nodiscard]] std::uint32_t HeadDim(std::uint32_t layer) const noexcept {
    return layer < head_dim_per_layer.size()
               ? head_dim_per_layer[layer]
               : head_dim_full;
  }
  [[nodiscard]] std::uint32_t RotaryDim(std::uint32_t layer) const noexcept {
    return HeadDim(layer);
  }
  [[nodiscard]] float RopeTheta(std::uint32_t layer) const noexcept {
    return LayerIsSwa(layer) ? rope_theta_swa : rope_theta_full;
  }
  [[nodiscard]] std::uint32_t KvHeads(std::uint32_t layer) const noexcept {
    return layer < num_kv_heads.size() ? num_kv_heads[layer] : 0;
  }
  /// Layers whose attention writes its own KV (non-SWA layers with
  /// head_count_kv > 0). SWA layers reuse the previous KV-producing layer.
  [[nodiscard]] std::uint32_t KvProducingLayer(std::uint32_t layer) const;

  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_CONFIG_HPP_
