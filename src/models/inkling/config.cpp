#include "src/models/inkling/config.hpp"

#include <variant>

namespace mbufo::models::inkling {
namespace {

bool Refuse(std::string* error_msg, const std::string& msg) {
  if (error_msg != nullptr) *error_msg = msg;
  return false;
}

}  // namespace

std::optional<InklingConfig> InklingConfig::FromGguf(
    const core::GgufReader& gguf, std::string* error_msg) {
  if (gguf.GetMetadataString("general.architecture") != "inkling2") {
    Refuse(error_msg, "inkling: expected general.architecture == 'inkling2'");
    return std::nullopt;
  }

  InklingConfig cfg;
  cfg.model_name = std::string(
      gguf.GetMetadataString("general.name").value_or("Inkling-2-Mini"));

  auto req_u32 = [&](const char* key, std::uint32_t* out) -> bool {
    auto v = gguf.GetMetadataUint32(key);
    if (!v) return Refuse(error_msg, std::string("inkling: missing ") + key);
    *out = *v;
    return true;
  };

  std::uint32_t block_count = 0;
  if (!req_u32("inkling2.block_count", &block_count) ||
      !req_u32("inkling2.embedding_length", &cfg.hidden_size) ||
      !req_u32("inkling2.attention.key_length", &cfg.head_dim) ||
      !req_u32("inkling2.expert_count", &cfg.expert_count) ||
      !req_u32("inkling2.expert_used_count", &cfg.expert_used_count)) {
    return std::nullopt;
  }
  cfg.block_count = block_count;

  // Array-or-scalar metadata (writers pick either form).
  auto read_u32_array = [&](const char* key,
                            std::vector<std::uint32_t>* out) -> bool {
    out->clear();
    if (const auto* meta = gguf.FindMetadata(key)) {
      if (const auto* u64s =
              std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
        for (auto v : *u64s) out->push_back(static_cast<std::uint32_t>(v));
        return true;
      }
      if (const auto* i64s =
              std::get_if<std::vector<std::int64_t>>(&meta->value)) {
        for (auto v : *i64s) out->push_back(static_cast<std::uint32_t>(v));
        return true;
      }
    }
    if (auto scalar = gguf.GetMetadataUint32(key)) {
      out->push_back(*scalar);
      return true;
    }
    return false;
  };

  std::vector<std::uint32_t> q_heads, kv_heads, swa_pattern;
  if (!read_u32_array("inkling2.attention.head_count", &q_heads) ||
      q_heads.empty()) {
    Refuse(error_msg, "inkling: missing inkling2.attention.head_count");
    return std::nullopt;
  }
  if (!read_u32_array("inkling2.attention.head_count_kv", &kv_heads) ||
      kv_heads.empty()) {
    Refuse(error_msg, "inkling: missing inkling2.attention.head_count_kv");
    return std::nullopt;
  }
  cfg.num_q_heads = q_heads[0];
  cfg.num_kv_heads = kv_heads[0];
  if (cfg.num_q_heads == 0 || cfg.num_kv_heads == 0 ||
      cfg.num_q_heads % cfg.num_kv_heads != 0) {
    Refuse(error_msg, "inkling: bad q/kv head counts");
    return std::nullopt;
  }

  cfg.value_dim =
      gguf.GetMetadataUint32("inkling2.attention.value_length").value_or(cfg.head_dim);
  cfg.rel_dim =
      gguf.GetMetadataUint32("inkling2.attention.relative_length").value_or(16);
  cfg.rel_extent =
      gguf.GetMetadataUint32("inkling2.attention.relative_extent").value_or(1024);

  cfg.vocab_size =
      gguf.GetMetadataUint32("inkling2.vocab_size").value_or(201024);
  cfg.vocab_unpadded = gguf.GetMetadataUint32("inkling2.vocab_size_unpadded")
                           .value_or(cfg.vocab_size);
  cfg.context_length =
      gguf.GetMetadataUint32("inkling2.context_length").value_or(1048576);
  cfg.sliding_window =
      gguf.GetMetadataUint32("inkling2.attention.sliding_window").value_or(1024);
  cfg.rms_norm_eps = gguf.GetMetadataFloat32(
                             "inkling2.attention.layer_norm_rms_epsilon")
                         .value_or(1e-6F);
  cfg.rope_dim =
      gguf.GetMetadataUint32("inkling2.rope.dimension_count").value_or(0);
  cfg.sconv_width =
      gguf.GetMetadataUint32("inkling2.shortconv.l_cache").value_or(4);

  cfg.num_loops = gguf.GetMetadataUint32("inkling2.num_loops").value_or(3);
  cfg.loop_core_start =
      gguf.GetMetadataUint32("inkling2.loop.core_start").value_or(6);
  cfg.loop_core_end =
      gguf.GetMetadataUint32("inkling2.loop.core_end").value_or(18);
  cfg.loop_gate_rank =
      gguf.GetMetadataUint32("inkling2.loop.gate_rank").value_or(64);
  cfg.attn_res_block =
      gguf.GetMetadataUint32("inkling2.attn_res.block_size").value_or(6);

  cfg.intermediate_size =
      gguf.GetMetadataUint32("inkling2.feed_forward_length").value_or(1152);
  cfg.expert_ffn_size =
      gguf.GetMetadataUint32("inkling2.expert_feed_forward_length")
          .value_or(cfg.intermediate_size);
  cfg.expert_weights_scale =
      gguf.GetMetadataFloat32("inkling2.expert_weights_scale").value_or(8.0F);

  cfg.situ_beta =
      gguf.GetMetadataFloat32("inkling2.activation.situ_beta").value_or(4.0F);
  cfg.situ_linear_beta =
      gguf.GetMetadataFloat32("inkling2.activation.situ_linear_beta")
          .value_or(25.0F);
  cfg.final_logit_softcap =
      gguf.GetMetadataFloat32("inkling2.final_logit_softcapping").value_or(30.0F);
  cfg.attn_qk_clip =
      gguf.GetMetadataFloat32("inkling2.attention.qk_clip").value_or(22.627417F);
  cfg.attn_rel_clip =
      gguf.GetMetadataFloat32("inkling2.attention.relative_clip").value_or(8.0F);
  cfg.log_scale_floor =
      gguf.GetMetadataUint32("inkling2.attention.log_scale_floor").value_or(8192);
  cfg.log_scale_alpha =
      gguf.GetMetadataFloat32("inkling2.attention.log_scale_alpha").value_or(0.1F);

  // Unrolled schedule = physical layers + loop-core repeats.
  const std::uint32_t core_span = (cfg.loop_core_end > cfg.loop_core_start)
                                      ? (cfg.loop_core_end - cfg.loop_core_start)
                                      : 0;
  const std::uint32_t repeats =
      (cfg.num_loops > 0) ? (cfg.num_loops - 1) : 0;
  std::uint32_t physical = cfg.block_count;
  if (core_span > 0 && repeats > 0 && cfg.block_count > core_span * repeats) {
    physical = cfg.block_count - core_span * repeats;
  }
  if (physical == 0 || physical > 128) {
    Refuse(error_msg, "inkling: bad derived physical layer count");
    return std::nullopt;
  }
  cfg.num_layers = physical;

  // Sliding-window pattern: False marks the SWA boundary layers
  // (blk.5/11/17/23 mod 24); True marks full-attention layers.
  if (!read_u32_array("inkling2.attention.sliding_window_pattern",
                      &swa_pattern) ||
      swa_pattern.size() < cfg.num_layers) {
    Refuse(error_msg, "inkling: missing inkling2.attention.sliding_window_pattern");
    return std::nullopt;
  }
  cfg.is_swa_layer.assign(cfg.num_layers, false);
  for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
    cfg.is_swa_layer[l] = (swa_pattern[l] == 0);
  }

  return cfg;
}

}  // namespace mbufo::models::inkling