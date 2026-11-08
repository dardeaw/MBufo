#include "src/models/llama/config.hpp"

namespace gufo::models::llama {
namespace {

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key) {
    const auto value = gguf.GetMetadataUint32(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF integer " + std::string(key));
      return 0;
    }
    return *value;
  }

  std::uint32_t U32Opt(std::string_view key, std::uint32_t fallback) {
    return gguf.GetMetadataUint32(key).value_or(fallback);
  }
};

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != "llama") {
    r.Fail("GGUF architecture is not llama");
    return std::nullopt;
  }
  Config c;
  c.num_layers = r.U32("llama.block_count");
  c.hidden_size = r.U32("llama.embedding_length");
  c.intermediate_size = r.U32("llama.feed_forward_length");
  c.context_length =
      r.U32Opt("llama.context_length", 131072);
  c.num_heads = r.U32("llama.attention.head_count");
  c.num_kv_heads = r.U32("llama.attention.head_count_kv");
  c.head_dim = r.U32Opt("llama.attention.key_length", 0);
  if (c.head_dim == 0 && c.num_heads > 0) {
    c.head_dim = c.hidden_size / c.num_heads;
  }
  if (const auto theta = gguf.GetMetadataFloat32("llama.rope.freq_base")) {
    c.rope_theta = *theta;
  }
  if (const auto eps =
          gguf.GetMetadataFloat32("llama.attention.layer_norm_rms_epsilon")) {
    c.rms_eps = *eps;
  }
  const auto* embd = gguf.FindTensor("token_embd.weight");
  if (embd == nullptr || embd->dimensions.size() < 2 || c.hidden_size == 0 ||
      embd->dimensions[0] != c.hidden_size) {
    r.Fail("token_embd.weight shape does not match llama.embedding_length");
    return std::nullopt;
  }
  c.vocab_size = static_cast<std::uint32_t>(embd->dimensions[1]);
  if (c.num_layers == 0 || c.hidden_size == 0 || c.num_heads == 0 ||
      c.num_kv_heads == 0 || c.head_dim == 0 || c.vocab_size == 0 ||
      (c.hidden_size % c.num_heads) != 0 ||
      (c.hidden_size / c.num_heads) != c.head_dim ||
      (c.num_heads % c.num_kv_heads) != 0) {
    r.Fail("llama artifact has inconsistent dimensions");
    return std::nullopt;
  }
  if (!r.ok)
    return std::nullopt;
  return c;
}

}  // namespace gufo::models::llama
