#include "src/models/laguna/config.hpp"

#include <algorithm>
#include <cmath>
#include <variant>

namespace mbufo::models::laguna {
namespace {

bool Refuse(std::string* error_msg, const std::string& msg) {
  if (error_msg != nullptr) *error_msg = msg;
  return false;
}

}  // namespace

std::optional<LagunaConfig> LagunaConfig::FromGguf(const core::GgufReader& gguf,
                                                   std::string* error_msg) {
  if (gguf.GetMetadataString("general.architecture") != "laguna") {
    Refuse(error_msg, "laguna: expected general.architecture == 'laguna'");
    return std::nullopt;
  }

  LagunaConfig cfg;
  cfg.model_name =
      std::string(gguf.GetMetadataString("general.name").value_or("Laguna-2.1"));

  auto req_u32 = [&](const char* key, std::uint32_t* out) -> bool {
    auto v = gguf.GetMetadataUint32(key);
    if (!v) return Refuse(error_msg, std::string("laguna: missing ") + key);
    *out = *v;
    return true;
  };

  if (!req_u32("laguna.block_count", &cfg.num_layers) ||
      !req_u32("laguna.embedding_length", &cfg.hidden_size) ||
      !req_u32("laguna.attention.head_count_kv", &cfg.num_kv_heads) ||
      !req_u32("laguna.attention.key_length", &cfg.head_dim)) {
    return std::nullopt;
  }

  cfg.intermediate_size =
      gguf.GetMetadataUint32("laguna.feed_forward_length").value_or(cfg.hidden_size * 4);
  cfg.vocab_size = gguf.GetMetadataUint32("laguna.vocab_size").value_or(100352);
  cfg.context_length = gguf.GetMetadataUint32("laguna.context_length").value_or(262144);
  cfg.sliding_window = gguf.GetMetadataUint32("laguna.attention.sliding_window").value_or(512);
  cfg.rms_norm_eps = gguf.GetMetadataFloat32("laguna.attention.layer_norm_rms_epsilon").value_or(1e-6F);

  cfg.rope_theta = gguf.GetMetadataFloat32("laguna.rope.freq_base").value_or(500000.0F);
  cfg.rotary_dim = gguf.GetMetadataUint32("laguna.rope.dimension_count").value_or(64);
  cfg.yarn_factor = gguf.GetMetadataFloat32("laguna.rope.scaling.factor").value_or(1.0F);
  cfg.yarn_orig_ctx = gguf.GetMetadataUint32("laguna.rope.scaling.original_context_length").value_or(8192);
  cfg.yarn_beta_fast = gguf.GetMetadataFloat32("laguna.rope.scaling.yarn_beta_fast").value_or(64.0F);
  cfg.yarn_beta_slow = gguf.GetMetadataFloat32("laguna.rope.scaling.yarn_beta_slow").value_or(1.0F);
  const float raw_attn =
      gguf.GetMetadataFloat32("laguna.rope.scaling.yarn_attn_factor").value_or(1.0F);
  if (cfg.yarn_factor > 1.0F && std::abs(raw_attn - 1.0F) < 1e-4F) {
    cfg.yarn_attn_factor = 0.1F * std::log(cfg.yarn_factor) + 1.0F;
  } else {
    cfg.yarn_attn_factor = raw_attn;
  }

  cfg.rope_theta_swa = gguf.GetMetadataFloat32("laguna.rope.freq_base_swa").value_or(10000.0F);
  cfg.rotary_dim_swa = gguf.GetMetadataUint32("laguna.rope.dimension_count_swa").value_or(cfg.head_dim);

  cfg.leading_dense_blocks = gguf.GetMetadataUint32("laguna.leading_dense_block_count").value_or(1);
  cfg.expert_count = gguf.GetMetadataUint32("laguna.expert_count").value_or(256);
  cfg.expert_used_count = gguf.GetMetadataUint32("laguna.expert_used_count").value_or(8);
  cfg.expert_ffn_size = gguf.GetMetadataUint32("laguna.expert_feed_forward_length").value_or(512);
  cfg.shared_expert_ffn_size =
      gguf.GetMetadataUint32("laguna.expert_shared_feed_forward_length").value_or(cfg.expert_ffn_size);
  cfg.expert_weights_norm = gguf.GetMetadataBool("laguna.expert_weights_norm").value_or(true);
  cfg.expert_weights_scale = gguf.GetMetadataFloat32("laguna.expert_weights_scale").value_or(2.5F);
  cfg.expert_gating_func = gguf.GetMetadataUint32("laguna.expert_gating_func").value_or(2);

  // Per-layer Q head counts (Global 48 vs SWA 64/72).
  cfg.q_heads_per_layer.clear();
  if (const auto* meta = gguf.FindMetadata("laguna.attention.head_count")) {
    if (const auto* u64s = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      for (auto v : *u64s) cfg.q_heads_per_layer.push_back(static_cast<std::uint32_t>(v));
    } else if (const auto* i64s = std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *i64s) cfg.q_heads_per_layer.push_back(static_cast<std::uint32_t>(v));
    } else if (const auto* scalar = std::get_if<std::uint64_t>(&meta->value)) {
      cfg.q_heads_per_layer.assign(cfg.num_layers, static_cast<std::uint32_t>(*scalar));
    }
  }
  if (cfg.q_heads_per_layer.size() != cfg.num_layers) {
    Refuse(error_msg, "laguna: laguna.attention.head_count length mismatch");
    return std::nullopt;
  }

  cfg.max_q_heads = 0;
  cfg.is_swa_layer.resize(cfg.num_layers, false);
  for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
    cfg.max_q_heads = std::max(cfg.max_q_heads, cfg.q_heads_per_layer[l]);
    // Laguna uses 1 Global : 3 SWA pattern (layers 0, 4, 8, ... are full_attention).
    cfg.is_swa_layer[l] = (cfg.sliding_window > 0) && ((l % 4) != 0);
  }

  return cfg;
}

}  // namespace mbufo::models::laguna
