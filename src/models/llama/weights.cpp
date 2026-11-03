#include "src/models/llama/weights.hpp"

#include <string>
#include <unordered_set>

namespace gufo::models::llama {
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
                std::uint64_t rows = 1) {
    TensorRef ref;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      return ref;
    }
    if (info->dimensions.size() < 1 || info->dimensions[0] != cols ||
        (info->dimensions.size() > 1 && info->dimensions[1] != rows)) {
      Fail("tensor " + name + " has unexpected dimensions");
      return ref;
    }
    ref.data = info->data;
    ref.size_bytes = info->size_bytes;
    ref.type = info->type;
    ref.cols = cols;
    ref.rows = rows;
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
  const std::uint64_t vocab = config->vocab_size;

  w.token_embd = b.Get("token_embd.weight", hidden, vocab);
  w.output_norm = b.Get("output_norm.weight", hidden);
  w.output = b.Get("output.weight", hidden, vocab);
  if (w.output.empty()) {
    w.output = w.token_embd;  // Tied LM head.
  }

  w.layers.resize(config->num_layers);
  const std::uint64_t qdim =
      static_cast<std::uint64_t>(config->num_heads) * config->head_dim;
  const std::uint64_t kvdim =
      static_cast<std::uint64_t>(config->num_kv_heads) * config->head_dim;
  for (std::uint32_t il = 0; il < config->num_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto& l = w.layers[il];
    l.attn_norm = b.Get(p + "attn_norm.weight", hidden);
    l.wq = b.Get(p + "attn_q.weight", hidden, qdim);
    l.wk = b.Get(p + "attn_k.weight", hidden, kvdim);
    l.wv = b.Get(p + "attn_v.weight", hidden, kvdim);
    l.wo = b.Get(p + "attn_output.weight", qdim, hidden);
    l.ffn_norm = b.Get(p + "ffn_norm.weight", hidden);
    l.ffn_gate = b.Get(p + "ffn_gate.weight", hidden, config->intermediate_size);
    l.ffn_up = b.Get(p + "ffn_up.weight", hidden, config->intermediate_size);
    l.ffn_down =
        b.Get(p + "ffn_down.weight", config->intermediate_size, hidden);
  }

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
    if (norm && t.type != core::GgmlType::kF32) {
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
    require(l.wk, p + "attn_k.weight");
    require(l.wv, p + "attn_v.weight");
    require(l.wo, p + "attn_output.weight");
    require(l.ffn_norm, p + "ffn_norm.weight", true);
    require(l.ffn_gate, p + "ffn_gate.weight");
    require(l.ffn_up, p + "ffn_up.weight");
    require(l.ffn_down, p + "ffn_down.weight");
  }
  if (!b.ok)
    return std::nullopt;
  return w;
}

}  // namespace gufo::models::llama
