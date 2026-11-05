#ifndef GUFO_MODELS_GPTOSS_CONFIG_HPP_
#define GUFO_MODELS_GPTOSS_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gptoss {

// Architecture parameters of a `gpt-oss` GGUF artifact (OpenAI GPT-OSS MoE:
// 20B = 24 layers, hidden 2880, 64 q-heads x dim 64, 8 kv-heads, 32 experts
// top-4, SwiGLU-OAI activation, attention sinks, alternating SWA-128 on even
// layers, YaRN rope). Every value is read from the file; the runtime converts
// MXFP4 experts to Q4_0 and dense K-quants to Q8_0 at load (host) and runs
// the shared core vec-MoE / narrow / blocked-KV paths.
struct Config {
  std::uint32_t num_layers{0};        ///< gpt-oss.block_count, 24.
  std::uint32_t hidden_size{0};       ///< gpt-oss.embedding_length, 2880.
  std::uint32_t intermediate_size{0}; ///< == expert_ff for gpt-oss, 2880.
  std::uint32_t num_heads{0};         ///< gpt-oss.attention.head_count, 64.
  std::uint32_t num_kv_heads{0};      ///< gpt-oss.attention.head_count_kv, 8.
  std::uint32_t head_dim{0};          ///< key/value_length, 64.
  std::uint32_t vocab_size{0};        ///< from token_embd.shape, 201088.
  std::uint32_t context_length{0};    ///< gpt-oss.context_length, 131072.
  std::uint32_t num_experts{0};       ///< gpt-oss.expert_count, 32.
  std::uint32_t experts_used{0};      ///< gpt-oss.expert_used_count, 4.
  std::uint32_t expert_ff{0};         ///< gpt-oss.expert_feed_forward_length.
  std::uint32_t sliding_window{0};    ///< gpt-oss.attention.sliding_window.
  float rope_theta{150000.0F};        ///< gpt-oss.rope.freq_base.
  float rope_yarn_factor{32.0F};      ///< rope.scaling.factor.
  float rms_eps{1e-5F};               ///< attention.layer_norm_rms_epsilon.
  std::uint32_t yarn_orig_ctx{4096};  ///< rope.scaling.original_context_length.

  /// YaRN correction range for one rotary pair index (HF linear ramp).
  struct Yarn {
    float low{0.0F};
    float high{0.0F};
    float inv_factor{1.0F};
    float mscale{1.0F};
  };
  Yarn yarn;

  /// True for the sliding-window layers. llama.cpp openai-moe uses swa period
  /// 2 with dense-first=false: even layers are SWA, odd layers dense.
  [[nodiscard]] bool IsSwaLayer(std::uint32_t layer) const noexcept {
    return sliding_window != 0 && (layer % 2) == 0;
  }
  [[nodiscard]] std::uint32_t LayerWindow(std::uint32_t layer) const noexcept {
    return IsSwaLayer(layer) ? sliding_window : 0;
  }

  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::gptoss

#endif  // GUFO_MODELS_GPTOSS_CONFIG_HPP_
