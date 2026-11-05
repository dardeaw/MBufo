#include "src/models/gptoss/config.hpp"

#include <cmath>
#include <numbers>

namespace gufo::models::gptoss {
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
  if (gguf.GetMetadataString("general.architecture") != "gpt-oss") {
    r.Fail("GGUF architecture is not gpt-oss");
    return std::nullopt;
  }
  Config c;
  c.num_layers = r.U32("gpt-oss.block_count");
  c.hidden_size = r.U32("gpt-oss.embedding_length");
  c.num_heads = r.U32("gpt-oss.attention.head_count");
  c.num_kv_heads = r.U32("gpt-oss.attention.head_count_kv");
  c.head_dim = r.U32("gpt-oss.attention.key_length");
  const std::uint32_t vlen = r.U32("gpt-oss.attention.value_length");
  if (vlen != 0 && vlen != c.head_dim) {
    r.Fail("gpt-oss value_length differs from key_length");
    return std::nullopt;
  }
  c.context_length = r.U32Opt("gpt-oss.context_length", 131072);
  c.num_experts = r.U32("gpt-oss.expert_count");
  c.experts_used = r.U32("gpt-oss.expert_used_count");
  c.expert_ff = r.U32("gpt-oss.expert_feed_forward_length");
  c.intermediate_size = c.expert_ff;
  c.sliding_window = r.U32Opt("gpt-oss.attention.sliding_window", 0);
  if (const auto theta = gguf.GetMetadataFloat32("gpt-oss.rope.freq_base")) {
    c.rope_theta = *theta;
  }
  if (const auto factor = gguf.GetMetadataFloat32("gpt-oss.rope.scaling.factor")) {
    c.rope_yarn_factor = *factor;
  }
  if (const auto orig =
          gguf.GetMetadataUint32("gpt-oss.rope.scaling.original_context_length")) {
    c.yarn_orig_ctx = *orig;
  }
  if (const auto eps = gguf.GetMetadataFloat32(
          "gpt-oss.attention.layer_norm_rms_epsilon")) {
    c.rms_eps = *eps;
  }
  const auto* embd = gguf.FindTensor("token_embd.weight");
  if (embd == nullptr || embd->dimensions.size() < 2 || c.hidden_size == 0 ||
      embd->dimensions[0] != c.hidden_size) {
    r.Fail("token_embd.weight shape does not match gpt-oss.embedding_length");
    return std::nullopt;
  }
  c.vocab_size = static_cast<std::uint32_t>(embd->dimensions[1]);
  if (c.num_layers == 0 || c.hidden_size == 0 || c.num_heads == 0 ||
      c.num_kv_heads == 0 || c.head_dim == 0 || c.vocab_size == 0 ||
      c.num_experts == 0 || c.experts_used == 0 ||
      c.experts_used > c.num_experts || c.expert_ff == 0 ||
      (c.num_heads % c.num_kv_heads) != 0 ||
      static_cast<std::uint64_t>(c.num_heads) * c.head_dim >
          static_cast<std::uint64_t>(c.hidden_size) * 2) {
    r.Fail("gpt-oss artifact has inconsistent dimensions");
    return std::nullopt;
  }
  // YaRN correction range (HF linear ramp over rotary pairs, beta fast 32 /
  // slow 1, same defaults as qwen38_flash_next::MakeYarnRope). A factor of 1
  // disables the ramp (multiplier 1, mscale 1).
  if (c.rope_yarn_factor != 1.0F && c.head_dim > 0 && c.rope_theta > 0.0F &&
      c.yarn_orig_ctx > 0) {
    const double rotary = static_cast<double>(c.head_dim);
    const double theta = static_cast<double>(c.rope_theta);
    const auto dim = [&](double rotations) {
      return rotary * std::log(static_cast<double>(c.yarn_orig_ctx) /
                               (rotations * 2.0 * std::numbers::pi)) /
             (2.0 * std::log(theta));
    };
    double low = std::max(std::floor(dim(32.0)), 0.0);
    double high =
        std::min(std::ceil(dim(1.0)), static_cast<double>(c.head_dim - 1));
    if (high < low) {
      low = high = static_cast<double>(c.head_dim);
    }
    c.yarn.low = static_cast<float>(low);
    c.yarn.high = static_cast<float>(high);
    c.yarn.inv_factor = 1.0F / c.rope_yarn_factor;
    c.yarn.mscale = static_cast<float>(
        0.1 * std::log(static_cast<double>(c.rope_yarn_factor)) + 1.0);
  }
  if (!r.ok)
    return std::nullopt;
  return c;
}

}  // namespace gufo::models::gptoss
