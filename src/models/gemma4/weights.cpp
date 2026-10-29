#include "src/models/gemma4/weights.hpp"

#include <string>
#include <unordered_set>

namespace gufo::models::gemma4 {
namespace {

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  TensorRef Get(const std::string& name, std::uint64_t cols,
                std::uint64_t rows = 1, std::uint64_t experts = 1) {
    TensorRef ref;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      return ref;
    }
    if (info->dimensions.size() < 1 || info->dimensions[0] != cols ||
        (info->dimensions.size() > 1 && info->dimensions[1] != rows) ||
        (info->dimensions.size() > 2 && info->dimensions[2] != experts)) {
      Fail("tensor " + name + " has unexpected dimensions (want [" +
           std::to_string(cols) + ", " + std::to_string(rows) +
           (experts > 1 ? ", " + std::to_string(experts) : "") + "] got [" +
           [&] {
             std::string t;
             for (auto d : info->dimensions) {
               if (!t.empty()) t += ", ";
               t += std::to_string(d);
             }
             return t;
           }() + "])");
      return ref;
    }
    ref.data = info->data;
    ref.size_bytes = info->size_bytes;
    ref.type = info->type;
    ref.cols = cols;
    ref.rows = rows;
    ref.experts = experts;
    return ref;
  }
};

[[nodiscard]] bool AllowedType(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kF32:
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
    case core::GgmlType::kQ4_0:
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
      return true;
    default:
      return false;
  }
}

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  switch (type) {
    case core::GgmlType::kF32:
      return static_cast<std::size_t>(cols) * 4;
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
      return static_cast<std::size_t>(cols) * 2;
    case core::GgmlType::kQ4_0:
      return static_cast<std::size_t>(cols / 32) * 18;
    case core::GgmlType::kQ8_0:
      return static_cast<std::size_t>(cols / 32) * 34;
    case core::GgmlType::kQ4_K:
      return static_cast<std::size_t>(cols / 256) * 144;
    case core::GgmlType::kQ5_K:
      return static_cast<std::size_t>(cols / 256) * 176;
    case core::GgmlType::kQ6_K:
      return static_cast<std::size_t>(cols / 256) * 210;
    default:
      return 0;
  }
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(reader, error_msg);
  if (!config.has_value())
    return std::nullopt;

  Binder b{reader, error_msg};
  ModelWeights w;
  w.config = *config;
  const std::uint64_t hidden = config->hidden_size;

  // Vocab size from the embedding tensor.
  const auto* embd_info = reader.FindTensor("token_embd.weight");
  if (embd_info == nullptr || embd_info->dimensions.size() < 2) {
    if (error_msg != nullptr) {
      *error_msg = "missing token_embd.weight";
    }
    return std::nullopt;
  }
  const std::uint64_t vocab = embd_info->dimensions[1];
  w.config.vocab_size = static_cast<std::uint32_t>(vocab);

  w.token_embd = b.Get("token_embd.weight", hidden, vocab);
  w.output_norm = b.Get("output_norm.weight", hidden);
  w.output = b.Get("output.weight", hidden, vocab);
  if (w.output.empty()) {
    w.output = w.token_embd;  // tied LM head
  }

  // Vision sidecar tensors live in the mmproj file; the text artifact must
  // not carry them.
  for (const auto& info : reader.GetTensors()) {
    if (info.name.starts_with("v.") || info.name.starts_with("mm.")) {
      b.Fail("gemma4 text artifact carries vision tensors; use the mmproj "
             "sidecar");
      break;
    }
  }

  TensorRef rope_freqs;
  w.layers.resize(config->num_layers);
  for (std::uint32_t il = 0; il < config->num_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto& l = w.layers[il];
    l.swa = config->LayerIsSwa(il);
    const std::uint32_t head_dim = config->HeadDim(il);
    const std::uint32_t kv_heads = config->KvHeads(il);
    l.has_kv = kv_heads > 0;
    l.has_out_scale =
        reader.FindTensor(p + "layer_output_scale.weight") != nullptr;

    l.attn_norm = b.Get(p + "attn_norm.weight", hidden);
    std::fprintf(stderr, "[gemma4] bind L%u swa=%d head_dim=%u wq want %llu\n",
                 il, (int)l.swa, head_dim,
                 (unsigned long long)(config->num_heads) * head_dim);
    l.wq = b.Get(p + "attn_q.weight", hidden,
                 static_cast<std::uint64_t>(config->num_heads) * head_dim);
    l.wo = b.Get(p + "attn_output.weight",
                 static_cast<std::uint64_t>(config->num_heads) * head_dim,
                 hidden);
    l.attn_q_norm = b.Get(p + "attn_q_norm.weight", head_dim);
    l.attn_post_norm = b.Get(p + "post_attention_norm.weight", hidden);
    if (l.has_out_scale) {
      l.out_scale = b.Get(p + "layer_output_scale.weight", 1);
    }
    if (!l.swa) {
      // full_attention layers use GGUF rope_freqs for proportional rope.
      l.rope_freqs = b.Get("rope_freqs.weight", 256);
      if (l.rope_freqs.empty()) {
        l.rope_freqs = rope_freqs;  // synthesised fallback below
      } else if (rope_freqs.empty()) {
        rope_freqs = l.rope_freqs;
      }
    }
    if (l.has_kv) {
      l.wk = b.Get(p + "attn_k.weight", hidden,
                   static_cast<std::uint64_t>(kv_heads) * head_dim);
      l.wv = b.Get(p + "attn_v.weight", hidden,
                   static_cast<std::uint64_t>(kv_heads) * head_dim);
      l.has_v_proj = !l.wv.empty();
      l.attn_k_norm = b.Get(p + "attn_k_norm.weight", head_dim);
      l.attn_v_norm = b.Get(p + "attn_v_norm.weight", head_dim);
    }

    l.ffn_norm = b.Get(p + "ffn_norm.weight", hidden);
    l.ffn_gate = b.Get(p + "ffn_gate.weight", hidden, config->shared_ff);
    l.ffn_up = b.Get(p + "ffn_up.weight", hidden, config->shared_ff);
    l.ffn_down = b.Get(p + "ffn_down.weight", config->shared_ff, hidden);
    l.ffn_post_norm = b.Get(p + "post_ffw_norm.weight", hidden);

    l.router = b.Get(p + "ffn_gate_inp.weight", hidden, config->num_experts);
    if (l.IsMoe()) {
      l.router_scale = b.Get(p + "ffn_gate_inp.scale", hidden);
      l.ffn_pre_norm_2 = b.Get(p + "pre_ffw_norm_2.weight", hidden);
      l.ffn_post_norm_1 = b.Get(p + "post_ffw_norm_1.weight", hidden);
      l.ffn_post_norm_2 = b.Get(p + "post_ffw_norm_2.weight", hidden);
      l.gate_up_exps =
          b.Get(p + "ffn_gate_up_exps.weight", hidden,
                static_cast<std::uint64_t>(config->expert_ff) * 2,
                config->num_experts);
      if (l.gate_up_exps.empty()) {
        l.gate_exps = b.Get(p + "ffn_gate_exps.weight", hidden,
                            config->expert_ff, config->num_experts);
        l.up_exps = b.Get(p + "ffn_up_exps.weight", hidden, config->expert_ff,
                          config->num_experts);
      }
      l.down_exps = b.Get(p + "ffn_down_exps.weight", config->expert_ff,
                          hidden, config->num_experts);
      l.down_exps_scale = b.Get(p + "ffn_down_exps.scale",
                                config->num_experts);
    }
  }

  // Validate the required set and quant formats.
  static const std::unordered_set<core::GgmlType> kNormTypes{
      core::GgmlType::kF32};
  const auto require = [&](const TensorRef& t, const std::string& name,
                           bool norm = false) {
    if (t.empty()) {
      b.Fail("missing required tensor " + name);
      return;
    }
    if (!AllowedType(t.type)) {
      b.Fail("unsupported quant type for " + name);
      return;
    }
    if (norm && !kNormTypes.contains(t.type)) {
      b.Fail("norm tensor must be F32: " + name);
    }
  };
  require(w.token_embd, "token_embd.weight");
  require(w.output_norm, "output_norm.weight", true);
  require(w.output, "output.weight");
  for (std::uint32_t il = 0; il < config->num_layers && b.ok; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const auto& l = w.layers[il];
    require(l.attn_norm, p + "attn_norm.weight", true);
    require(l.wq, p + "attn_q.weight");
    require(l.wo, p + "attn_output.weight");
    require(l.attn_q_norm, p + "attn_q_norm.weight", true);
    require(l.attn_post_norm, p + "post_attention_norm.weight", true);
    if (l.has_kv) {
      require(l.wk, p + "attn_k.weight");
      require(l.attn_k_norm, p + "attn_k_norm.weight", true);
    }
    require(l.ffn_norm, p + "ffn_norm.weight", true);
    require(l.ffn_gate, p + "ffn_gate.weight");
    require(l.ffn_up, p + "ffn_up.weight");
    require(l.ffn_down, p + "ffn_down.weight");
    require(l.ffn_post_norm, p + "post_ffw_norm.weight", true);
    if (l.IsMoe()) {
      require(l.router, p + "ffn_gate_inp.weight");
      require(l.router_scale, p + "ffn_gate_inp.scale", true);
      require(l.ffn_pre_norm_2, p + "pre_ffw_norm_2.weight", true);
      require(l.ffn_post_norm_1, p + "post_ffw_norm_1.weight", true);
      require(l.ffn_post_norm_2, p + "post_ffw_norm_2.weight", true);
      if (l.gate_up_exps.empty()) {
        require(l.gate_exps, p + "ffn_gate_exps.weight");
        require(l.up_exps, p + "ffn_up_exps.weight");
      }
      require(l.down_exps, p + "ffn_down_exps.weight");
    }
  }
  if (!b.ok)
    return std::nullopt;
  return w;
}

}  // namespace gufo::models::gemma4
