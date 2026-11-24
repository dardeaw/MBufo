#include "src/models/inkling/weights.hpp"

#include <string>

namespace mbufo::models::inkling {
namespace {

// GGUF stores tensors as [n0,n1,n2] with n0 innermost (column-major reversal
// of C row-major). Laguna template binds cols=n0 (K, inner/fastest), rows=n1
// (M, outer), depth=n2 and calls GEMM with m=rows(M), k=cols(K) so that the
// row-major view is [M,K] with K contiguous. Keep the identical convention
// here; FillRef records the expected (cols,rows,depth) like laguna so that
// RowBytes()/ExpertStrideBytes() always match the m/k passed to
// GemvDecode/GemvBatched/QuantGemmPrefill even if the file carries trailing
// singleton dims.
bool ShapeMatches(const core::GgufTensorInfo& info, std::uint64_t cols,
                  std::uint64_t rows, std::uint64_t depth) {
  const std::uint64_t c = !info.dimensions.empty() ? info.dimensions[0] : 1;
  const std::uint64_t r = info.dimensions.size() >= 2 ? info.dimensions[1] : 1;
  const std::uint64_t d = info.dimensions.size() >= 3 ? info.dimensions[2] : 1;
  return c == cols && r == rows && d == depth;
}

void FillRef(const core::GgufTensorInfo& info, std::uint64_t cols,
             std::uint64_t rows, std::uint64_t depth, TensorRef* out) {
  out->data = info.data;
  out->type = info.type;
  out->size_bytes = info.size_bytes;
  out->rows = rows;
  out->cols = cols;
  out->depth = depth;
}

// Required tensor: missing name or wrong shape fails the load.
bool BindTensor(const core::GgufReader& reader, const std::string& name,
                std::uint64_t cols, std::uint64_t rows, std::uint64_t depth,
                TensorRef* out, std::string* error_msg) {
  const auto* info = reader.FindTensor(name);
  if (info == nullptr) {
    if (error_msg != nullptr) *error_msg = "inkling: missing tensor " + name;
    return false;
  }
  if (!ShapeMatches(*info, cols, rows, depth)) {
    if (error_msg != nullptr) {
      *error_msg = "inkling: unexpected shape on tensor " + name;
    }
    return false;
  }
  FillRef(*info, cols, rows, depth, out);
  return true;
}

// Optional tensor: absent names bind empty; present names must match.
bool BindOptional(const core::GgufReader& reader, const std::string& name,
                  std::uint64_t cols, std::uint64_t rows, std::uint64_t depth,
                  TensorRef* out, std::string* error_msg) {
  const auto* info = reader.FindTensor(name);
  if (info == nullptr) {
    *out = TensorRef{};
    return true;
  }
  return BindTensor(reader, name, cols, rows, depth, out, error_msg);
}

}  // namespace

std::optional<ModelWeights> ModelWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  auto cfg = InklingConfig::FromGguf(reader, error_msg);
  if (!cfg) return std::nullopt;

  ModelWeights w;
  w.config = std::move(*cfg);

  const std::uint64_t h = w.config.hidden_size;
  const std::uint64_t vocab = w.config.vocab_size;
  const std::uint64_t q_dim =
      static_cast<std::uint64_t>(w.config.num_q_heads) * w.config.head_dim;
  const std::uint64_t kv_dim =
      static_cast<std::uint64_t>(w.config.num_kv_heads) * w.config.head_dim;
  const std::uint64_t e = w.config.expert_count;
  const std::uint64_t eff = w.config.expert_ffn_size;
  const std::uint64_t fused = eff * 2;

  auto req = [&](const std::string& name, std::uint64_t cols,
                 std::uint64_t rows, std::uint64_t depth,
                 TensorRef* out) -> bool {
    return BindTensor(reader, name, cols, rows, depth, out, error_msg);
  };
  auto opt = [&](const std::string& name, std::uint64_t cols,
                 std::uint64_t rows, std::uint64_t depth,
                 TensorRef* out) -> bool {
    return BindOptional(reader, name, cols, rows, depth, out, error_msg);
  };

  if (!req("token_embd.weight", h, vocab, 1, &w.token_embd) ||
      !req("token_embd_norm.weight", h, 1, 1, &w.token_embd_norm) ||
      !req("output.weight", h, vocab, 1, &w.output) ||
      !req("output_norm.weight", h, 1, 1, &w.output_norm) ||
      !req("loop_kv_res_query.weight", h, 24, 1, &w.loop_kv_res_query) ||
      !req("loop_res_query.weight", h, 24, 1, &w.loop_res_query) ||
      !req("output_res_query.weight", h, 1, 1, &w.output_res_query)) {
    return std::nullopt;
  }

  w.layers.resize(w.config.num_layers);
  for (std::uint32_t l = 0; l < w.config.num_layers; ++l) {
    auto& lw = w.layers[l];
    lw.q_heads = w.config.num_q_heads;
    lw.is_swa = w.config.is_swa_layer[l];
    lw.is_loop_core =
        (l >= w.config.loop_core_start && l < w.config.loop_core_end);
    lw.is_nvfp4 = (l < 20);

    const std::string p = "blk." + std::to_string(l) + ".";

    if (!req(p + "attn_q.weight", h, q_dim, 1, &lw.wq) ||
        !req(p + "attn_k.weight", h, kv_dim, 1, &lw.wk) ||
        !req(p + "attn_v.weight", h, kv_dim, 1, &lw.wv) ||
        !req(p + "attn_output.weight", q_dim, h, 1, &lw.wo) ||
        !req(p + "attn_gate.weight", h, q_dim, 1, &lw.attn_gate) ||
        !req(p + "attn_rel.weight", h,
             static_cast<std::uint64_t>(w.config.num_q_heads) *
                 w.config.rel_dim,
             1, &lw.attn_rel) ||
        !req(p + "attn_rel_bias.weight", w.config.rel_dim, kv_dim, 1,
             &lw.attn_rel_bias) ||
        !req(p + "attn_rel_clip.weight", w.config.rel_dim,
             w.config.num_q_heads, 1, &lw.attn_rel_clip) ||
        !req(p + "attn_q_clip.weight", w.config.head_dim,
             w.config.num_q_heads, 1, &lw.attn_q_clip) ||
        !req(p + "attn_sink_proj.weight", h, w.config.num_q_heads, 1,
             &lw.attn_sink_proj) ||
        !req(p + "attn_k_conv.weight", w.config.sconv_width, kv_dim, 1,
             &lw.attn_k_conv) ||
        !req(p + "attn_v_conv.weight", w.config.sconv_width, kv_dim, 1,
             &lw.attn_v_conv) ||
        !req(p + "attn_post_conv.weight", w.config.sconv_width, h, 1,
             &lw.attn_post_conv) ||
        !req(p + "ffn_norm.weight", h, 1, 1, &lw.ffn_norm) ||
        !req(p + "ffn_post_conv.weight", w.config.sconv_width, h, 1,
             &lw.ffn_post_conv) ||
        !req(p + "ffn_res_query.weight", h, 1, 1, &lw.ffn_res_query) ||
        !req(p + "ffn_gate_inp.weight", h, e, 1, &lw.ffn_gate_inp) ||
        !req(p + "exp_probs_b.bias", e, 1, 1, &lw.exp_probs_b) ||
        !req(p + "ffn_gate_up_exps.weight", h, fused, e,
             &lw.ffn_gate_up_exps) ||
        !req(p + "ffn_down_exps.weight", eff, h, e, &lw.ffn_down_exps)) {
      return std::nullopt;
    }

    // SWA boundary layers swap attn_norm for q/k/v input norms plus
    // KV conv projections; other layers carry attn_norm.
    if (lw.is_swa) {
      if (!req(p + "attn_q_inp_norm.weight", h, 1, 1, &lw.attn_q_inp_norm) ||
          !req(p + "attn_k_inp_norm.weight", h, 1, 1, &lw.attn_k_inp_norm) ||
          !req(p + "attn_v_inp_norm.weight", h, 1, 1, &lw.attn_v_inp_norm) ||
          !req(p + "attn_k_conv_proj.weight", h, 32, 1,
               &lw.attn_k_conv_proj) ||
          !req(p + "attn_v_conv_proj.weight", h, 32, 1,
               &lw.attn_v_conv_proj)) {
        return std::nullopt;
      }
    } else {
      if (!req(p + "attn_norm.weight", h, 1, 1, &lw.attn_norm)) {
        return std::nullopt;
      }
    }

    // attn_res_query is absent on blk.0, [2304,3] on SWA layers.
    if (lw.is_swa) {
      if (!BindOptional(reader, p + "attn_res_query.weight", h, 3, 1,
                        &lw.attn_res_query, error_msg)) {
        return std::nullopt;
      }
    } else {
      if (!opt(p + "attn_res_query.weight", h, 1, 1, &lw.attn_res_query)) {
        return std::nullopt;
      }
    }

    // NVFP4 expert scales (F32 per expert) exist only on blk.0-19.
    if (!opt(p + "ffn_gate_up_exps.scale", e, 1, 1, &lw.ffn_gate_up_scale) ||
        !opt(p + "ffn_down_exps.scale", e, 1, 1, &lw.ffn_down_scale)) {
      return std::nullopt;
    }

    // Loop-core message tensors (physical blk.6-17 only).
    const std::uint64_t rank = w.config.loop_gate_rank;
    const std::uint64_t pair = h * 2;
    LoopTensors* loops[4] = {&lw.loop_attn, &lw.loop_k, &lw.loop_mlp,
                             &lw.loop_v};
    const char* names[4] = {"loop_attn_", "loop_k_", "loop_mlp_", "loop_v_"};
    for (int i = 0; i < 4; ++i) {
      if (!opt(p + names[i] + std::string("down.weight"), pair, rank, 1,
               &loops[i]->down) ||
          !opt(p + names[i] + std::string("forget.weight"), rank, h, 1,
               &loops[i]->forget) ||
          !opt(p + names[i] + std::string("gain.weight"), rank, h, 1,
               &loops[i]->gain)) {
        return std::nullopt;
      }
    }
  }

  return w;
}

}  // namespace mbufo::models::inkling
