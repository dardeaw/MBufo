#ifndef MBUFO_MODELS_LAGUNA_CONFIG_HPP_
#define MBUFO_MODELS_LAGUNA_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace mbufo::models::laguna {

/// Parsed hyperparameters for `general.architecture == "laguna"` GGUF artifacts
/// (Laguna-XS-2.1 33B-A3B and Laguna-S-2.1 118B-A8B).
struct LagunaConfig {
  std::string model_name;
  std::uint32_t vocab_size{100352};
  std::uint32_t hidden_size{2048};
  std::uint32_t intermediate_size{8192};
  std::uint32_t num_layers{40};
  std::uint32_t num_kv_heads{8};
  std::uint32_t head_dim{128};
  std::uint32_t max_q_heads{64};
  std::vector<std::uint32_t> q_heads_per_layer;
  std::vector<bool> is_swa_layer;

  std::uint32_t context_length{262144};
  std::uint32_t sliding_window{512};
  float rms_norm_eps{1e-6F};

  // Dual RoPE: Global (YaRN, partial rotary_dim=64) vs SWA (standard, rotary_dim=128).
  float rope_theta{500000.0F};
  std::uint32_t rotary_dim{64};
  float yarn_factor{1.0F};
  std::uint32_t yarn_orig_ctx{8192};
  float yarn_beta_fast{64.0F};
  float yarn_beta_slow{1.0F};
  float yarn_attn_factor{1.0F};

  float rope_theta_swa{10000.0F};
  std::uint32_t rotary_dim_swa{128};

  // Leading dense layers + Sigmoid-routed MoE + Shared Expert.
  std::uint32_t leading_dense_blocks{1};
  std::uint32_t expert_count{256};
  std::uint32_t expert_used_count{8};
  std::uint32_t expert_ffn_size{512};
  std::uint32_t shared_expert_ffn_size{512};
  bool expert_weights_norm{true};
  float expert_weights_scale{2.5F};
  std::uint32_t expert_gating_func{2};  // 2 = sigmoid

  [[nodiscard]] static std::optional<LagunaConfig> FromGguf(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);
};

}  // namespace mbufo::models::laguna

#endif  // MBUFO_MODELS_LAGUNA_CONFIG_HPP_
