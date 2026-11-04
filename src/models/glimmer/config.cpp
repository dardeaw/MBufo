#include "src/models/glimmer/config.hpp"

namespace gufo::models::glimmer {
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
  if (gguf.GetMetadataString("general.architecture") != "muse-glimmer") {
    r.Fail("GGUF architecture is not muse-glimmer");
    return std::nullopt;
  }
  Config c;
  c.num_layers = r.U32("muse-glimmer.block_count");
  c.hidden_size = r.U32("muse-glimmer.embedding_length");
  c.intermediate_size = r.U32("muse-glimmer.feed_forward_length");
  c.context_length = r.U32Opt("muse-glimmer.context_length", 131072);
  c.num_heads = r.U32("muse-glimmer.attention.head_count");
  c.num_kv_heads = r.U32("muse-glimmer.attention.head_count_kv");
  c.head_dim = r.U32Opt("muse-glimmer.attention.key_length", 0);
  if (c.head_dim == 0 && c.num_heads > 0) {
    c.head_dim = c.hidden_size / c.num_heads;
  }
  if (const auto theta = gguf.GetMetadataFloat32("muse-glimmer.rope.freq_base")) {
    c.rope_theta = *theta;
  }
  if (const auto theta_swa =
          gguf.GetMetadataFloat32("muse-glimmer.rope.freq_base_swa")) {
    c.rope_theta_swa = *theta_swa;
  }
  if (const auto eps = gguf.GetMetadataFloat32(
          "muse-glimmer.attention.layer_norm_rms_epsilon")) {
    c.rms_eps = *eps;
  }
  c.post_norm_eps = c.rms_eps;
  if (const auto s = gguf.GetMetadataFloat32("muse-glimmer.logit_scale")) {
    c.logit_scale = *s;
  }
  if (const auto cap =
          gguf.GetMetadataFloat32("muse-glimmer.final_logit_softcapping")) {
    c.final_logit_softcapping = *cap;
  }
  c.sliding_window =
      r.U32Opt("muse-glimmer.attention.sliding_window", 0);
  c.sliding_period =
      r.U32Opt("muse-glimmer.attention.sliding_window_pattern", 0);
  const auto* embd = gguf.FindTensor("token_embd.weight");
  if (embd == nullptr || embd->dimensions.size() < 2 || c.hidden_size == 0 ||
      embd->dimensions[0] != c.hidden_size) {
    r.Fail("token_embd.weight shape does not match muse-glimmer.embedding_length");
    return std::nullopt;
  }
  c.vocab_size = static_cast<std::uint32_t>(embd->dimensions[1]);
  // Glimmer-30B is not hidden == heads * head_dim (6656 vs 32*128=4096:
  // attention out-projection is qdim->hidden, not hidden->hidden). Only
  // validate GQA divisibility and positive dims.
  if (c.num_layers == 0 || c.hidden_size == 0 || c.num_heads == 0 ||
      c.num_kv_heads == 0 || c.head_dim == 0 || c.vocab_size == 0 ||
      (c.num_heads % c.num_kv_heads) != 0) {
    r.Fail("muse-glimmer artifact has inconsistent dimensions");
    return std::nullopt;
  }
  if (!r.ok)
    return std::nullopt;
  return c;
}

}  // namespace gufo::models::glimmer
