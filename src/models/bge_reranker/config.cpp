// BgeRerankerConfig extraction from GGUF metadata (see config.hpp).
#include "src/models/bge_reranker/config.hpp"

#include <limits>

namespace gufo::models::bge_reranker {

std::optional<BgeRerankerConfig> ExtractBgeRerankerConfig(
    const core::GgufReader& reader, std::string* error_msg) {
  auto refuse =
      [&](const std::string& message) -> std::optional<BgeRerankerConfig> {
    if (error_msg != nullptr)
      *error_msg = message;
    return std::nullopt;
  };
  const auto arch = reader.GetMetadataString("general.architecture");
  if (!arch.has_value() || *arch != "bert") {
    return refuse("bge-reranker requires general.architecture == \"bert\"");
  }
  BgeRerankerConfig config;
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
  if (auto v = reader.GetMetadataFloat32(std::string(kPrefix) +
                                         "attention.layer_norm_epsilon")) {
    config.layer_norm_eps = *v;
  }
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
  // Vocabulary width comes from the word table (the only ground truth when
  // metadata is sparse): rows = elements / hidden.
  if (const auto* token_embd = reader.FindTensor("token_embd.weight");
      token_embd != nullptr && config.hidden_size > 0 &&
      token_embd->ElementCount() % config.hidden_size == 0) {
    const std::size_t rows = token_embd->ElementCount() / config.hidden_size;
    if (rows > 0 && rows <= static_cast<std::size_t>(
                                std::numeric_limits<std::uint32_t>::max())) {
      config.vocab_size = static_cast<std::uint32_t>(rows);
    }
  }
  if (!IsBgeRerankerConfig(config)) {
    return refuse("bge-reranker geometry failed structural validation");
  }
  return config;
}

}  // namespace gufo::models::bge_reranker
