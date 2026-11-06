// bge-reranker-v2-m3 (XLM-RoBERTa sequence-classification) configuration.
//
// Same 24-layer XLM-R encoder geometry as bge-m3 (hidden 1024, 24 layers,
// 16 heads, vocab 250002), plus the Roberta classification head
// (dense+tanh+out-projection to one logit). Unlike the embedding artifact,
// a reranker GGUF declares NO pooling metadata -- relevance always reads
// the CLS row by construction -- so extraction here never consults
// `bert.pooling_type` and refuses nothing on its absence.
#ifndef GUFO_MODELS_BGE_RERANKER_CONFIG_HPP_
#define GUFO_MODELS_BGE_RERANKER_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::bge_reranker {

// Single-prefill leg window: full bidirectional attention is O(n^2)
// transient. 2048 matches the encoder base and covers the retrieval-pair
// regime; longer pairs are refused explicitly, never truncated silently.
inline constexpr std::uint32_t kBgeRerankerMaxTokens = 2048;

/// bge-reranker-v2-m3 architectural configuration (XLM-R encoder + head).
struct BgeRerankerConfig {
  std::uint32_t num_layers{24};
  std::uint32_t hidden_size{1024};
  std::uint32_t intermediate_size{4096};
  std::uint32_t num_attention_heads{16};
  std::uint32_t vocab_size{250002};
  std::uint32_t max_position_embeddings{8192};
  float layer_norm_eps{1e-05F};
  std::string model_name{"bge-reranker-v2-m3"};
  /// Tokenizer control ids (XLM-R layout).
  std::uint32_t cls_token_id{0};  ///< `<s>`
  std::uint32_t sep_token_id{2};  ///< `</s>`
  std::uint32_t pad_token_id{1};  ///< `<pad>`
  std::uint32_t unk_token_id{3};  ///< `<unk>`
};

/// Structural validation for the XLM-R encoder layout. False means "not a
/// bge-reranker-shaped artifact": the loader refuses instead of guessing.
[[nodiscard]] inline bool IsBgeRerankerConfig(
    const BgeRerankerConfig& config) noexcept {
  return config.num_layers > 0 && config.hidden_size > 0 &&
         config.intermediate_size > 0 && config.num_attention_heads > 0 &&
         (config.hidden_size % config.num_attention_heads) == 0 &&
         config.vocab_size > 0 && config.max_position_embeddings > 0 &&
         config.layer_norm_eps > 0.0F;
}

/// Derives a BgeRerankerConfig from GGUF metadata. Returns nullopt (with
/// *error_msg set when non-null) when the artifact is not
/// `general.architecture == "bert"`, when a dimension is missing, or when
/// the geometry fails IsBgeRerankerConfig.
[[nodiscard]] std::optional<BgeRerankerConfig> ExtractBgeRerankerConfig(
    const core::GgufReader& reader, std::string* error_msg = nullptr);

}  // namespace gufo::models::bge_reranker

#endif  // GUFO_MODELS_BGE_RERANKER_CONFIG_HPP_
