// BgeM3Config extraction from GGUF metadata (see config.hpp for semantics).
#include "src/models/bge_m3/config.hpp"

#include "src/core/pooling_norm.hpp"

namespace gufo::models::bge_m3 {
namespace {

constexpr std::uint32_t kClsPoolingType = 2;

}  // namespace

std::optional<BgeM3Config> ExtractBgeM3Config(const core::GgufReader& reader,
                                              std::string* error_msg) {
  auto refuse = [&](const std::string& message) -> std::optional<BgeM3Config> {
    if (error_msg != nullptr) *error_msg = message;
    return std::nullopt;
  };
  const auto arch = reader.GetMetadataString("general.architecture");
  if (!arch.has_value() || *arch != "bert") {
    return refuse("bge-m3 requires general.architecture == \"bert\"");
  }
  BgeM3Config config;
  constexpr std::string_view kPrefix = "bert.";
  if (auto v = reader.GetMetadataUint32(std::string(kPrefix) + "block_count")) {
    config.num_layers = *v;
  }
  if (auto v =
          reader.GetMetadataUint32(std::string(kPrefix) + "embedding_length")) {
    config.hidden_size = *v;
  }
  if (auto v = reader.GetMetadataUint32(std::string(kPrefix) +
                                        "feed_forward_length")) {
    config.intermediate_size = *v;
  }
  if (auto v = reader.GetMetadataUint32(std::string(kPrefix) +
                                        "attention.head_count")) {
    config.num_attention_heads = *v;
  }
  if (auto v =
          reader.GetMetadataUint32(std::string(kPrefix) + "context_length")) {
    config.max_position_embeddings = *v;
  }
  if (auto v = reader.GetMetadataFloat32(
          std::string(kPrefix) + "attention.layer_norm_epsilon")) {
    config.layer_norm_eps = *v;
  }
  // Pooling is a declared artifact property, never a guess: the bge-m3
  // sentence-transformers layout pools CLS, which is GGUF pooling_type 2.
  // The shared core mapping (PoolingStrategyFromGgufType) is the single
  // source of truth; anything but CLS refuses here because this model
  // directory only implements the CLS semantic.
  const auto pooling =
      reader.GetMetadataUint32(std::string(kPrefix) + "pooling_type");
  if (!pooling.has_value()) {
    return refuse("bert.pooling_type is missing; refusing to guess pooling");
  }
  if (core::embedding::PoolingStrategyFromGgufType(*pooling) !=
      core::embedding::PoolingStrategy::kCls) {
    return refuse("bge-m3 supports only CLS pooling (bert.pooling_type 2)");
  }
  config.pooling_type = *pooling;
  if (auto v = reader.GetMetadataUint32("tokenizer.ggml.bos_token_id")) {
    config.cls_token_id = *v;
  }
  if (auto v = reader.GetMetadataUint32("tokenizer.ggml.eos_token_id")) {
    config.sep_token_id = *v;
  }
  if (auto v = reader.GetMetadataUint32("tokenizer.ggml.padding_token_id")) {
    config.pad_token_id = *v;
  }
  if (auto v = reader.GetMetadataUint32("tokenizer.ggml.unknown_token_id")) {
    config.unk_token_id = *v;
  }
  if (auto name = reader.GetMetadataString("general.name")) {
    config.model_name = std::string(*name);
  }
  if (const auto* token_embd = reader.FindTensor("token_embd.weight");
      token_embd != nullptr && config.hidden_size > 0) {
    const std::size_t elements = token_embd->ElementCount();
    if ((elements % config.hidden_size) == 0) {
      config.vocab_size =
          static_cast<std::uint32_t>(elements / config.hidden_size);
    }
  }
  if (!IsBgeM3Config(config)) {
    return refuse("bert geometry failed bge-m3 structural validation");
  }
  return config;
}

}  // namespace gufo::models::bge_m3
