#include "src/models/laguna/weights.hpp"

#include <string>

namespace mbufo::models::laguna {
namespace {

bool BindTensor(const core::GgufReader& reader, const std::string& name,
                std::uint64_t expected_cols, std::uint64_t expected_rows,
                std::uint64_t expected_depth, TensorRef* out,
                std::string* error_msg) {
  const auto* info = reader.FindTensor(name);
  if (info == nullptr) {
    if (error_msg != nullptr) *error_msg = "laguna: missing tensor " + name;
    return false;
  }
  const std::uint64_t cols = !info->dimensions.empty() ? info->dimensions[0] : 1;
  const std::uint64_t rows = info->dimensions.size() >= 2 ? info->dimensions[1] : 1;
  const std::uint64_t depth = info->dimensions.size() >= 3 ? info->dimensions[2] : 1;
  if (cols != expected_cols || rows != expected_rows || depth != expected_depth) {
    if (error_msg != nullptr) {
      *error_msg = "laguna: unexpected shape on tensor " + name;
    }
    return false;
  }
  out->data = info->data;
  out->type = info->type;
  out->size_bytes = info->size_bytes;
  out->rows = rows;
  out->cols = cols;
  out->depth = depth;
  return true;
}

}  // namespace

std::optional<ModelWeights> ModelWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  auto cfg = LagunaConfig::FromGguf(reader, error_msg);
  if (!cfg) return std::nullopt;

  ModelWeights w;
  w.config = std::move(*cfg);

  const std::uint64_t h = w.config.hidden_size;
  const std::uint64_t vocab = w.config.vocab_size;
  const std::uint64_t kv_dim =
      static_cast<std::uint64_t>(w.config.num_kv_heads) * w.config.head_dim;

  if (!BindTensor(reader, "token_embd.weight", h, vocab, 1, &w.token_embd, error_msg) ||
      !BindTensor(reader, "output_norm.weight", h, 1, 1, &w.output_norm, error_msg) ||
      !BindTensor(reader, "output.weight", h, vocab, 1, &w.output, error_msg)) {
    return std::nullopt;
  }

  w.layers.resize(w.config.num_layers);
  for (std::uint32_t l = 0; l < w.config.num_layers; ++l) {
    auto& lw = w.layers[l];
    lw.q_heads = w.config.q_heads_per_layer[l];
    lw.is_swa = w.config.is_swa_layer[l];
    lw.is_moe = (l >= w.config.leading_dense_blocks);

    const std::string p = "blk." + std::to_string(l) + ".";
    const std::uint64_t q_dim = static_cast<std::uint64_t>(lw.q_heads) * w.config.head_dim;

    if (!BindTensor(reader, p + "attn_norm.weight", h, 1, 1, &lw.attn_norm, error_msg) ||
        !BindTensor(reader, p + "attn_q.weight", h, q_dim, 1, &lw.wq, error_msg) ||
        !BindTensor(reader, p + "attn_k.weight", h, kv_dim, 1, &lw.wk, error_msg) ||
        !BindTensor(reader, p + "attn_v.weight", h, kv_dim, 1, &lw.wv, error_msg) ||
        !BindTensor(reader, p + "attn_output.weight", q_dim, h, 1, &lw.wo, error_msg) ||
        !BindTensor(reader, p + "attn_q_norm.weight", w.config.head_dim, 1, 1, &lw.attn_q_norm, error_msg) ||
        !BindTensor(reader, p + "attn_k_norm.weight", w.config.head_dim, 1, 1, &lw.attn_k_norm, error_msg) ||
        !BindTensor(reader, p + "attn_gate.weight", h, lw.q_heads, 1, &lw.attn_gate, error_msg) ||
        !BindTensor(reader, p + "ffn_norm.weight", h, 1, 1, &lw.ffn_norm, error_msg)) {
      return std::nullopt;
    }

    if (!lw.is_moe) {
      const std::uint64_t ff = w.config.intermediate_size;
      if (!BindTensor(reader, p + "ffn_gate.weight", h, ff, 1, &lw.ffn_gate, error_msg) ||
          !BindTensor(reader, p + "ffn_up.weight", h, ff, 1, &lw.ffn_up, error_msg) ||
          !BindTensor(reader, p + "ffn_down.weight", ff, h, 1, &lw.ffn_down, error_msg)) {
        return std::nullopt;
      }
    } else {
      const std::uint64_t e = w.config.expert_count;
      const std::uint64_t eff = w.config.expert_ffn_size;
      const std::uint64_t sff = w.config.shared_expert_ffn_size;
      if (!BindTensor(reader, p + "ffn_gate_inp.weight", h, e, 1, &lw.ffn_gate_inp, error_msg) ||
          !BindTensor(reader, p + "ffn_gate_exps.weight", h, eff, e, &lw.ffn_gate_exps, error_msg) ||
          !BindTensor(reader, p + "ffn_up_exps.weight", h, eff, e, &lw.ffn_up_exps, error_msg) ||
          !BindTensor(reader, p + "ffn_down_exps.weight", eff, h, e, &lw.ffn_down_exps, error_msg) ||
          !BindTensor(reader, p + "ffn_gate_shexp.weight", h, sff, 1, &lw.ffn_gate_shexp, error_msg) ||
          !BindTensor(reader, p + "ffn_up_shexp.weight", h, sff, 1, &lw.ffn_up_shexp, error_msg) ||
          !BindTensor(reader, p + "ffn_down_shexp.weight", sff, h, 1, &lw.ffn_down_shexp, error_msg)) {
        return std::nullopt;
      }
      // Optional load-balancing bias (e_score_correction_bias)
      if (reader.FindTensor(p + "exp_probs_b.bias") != nullptr) {
        if (!BindTensor(reader, p + "exp_probs_b.bias", e, 1, 1, &lw.exp_probs_b, error_msg)) {
          return std::nullopt;
        }
      }
    }
  }

  return w;
}

}  // namespace mbufo::models::laguna
