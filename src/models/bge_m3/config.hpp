// bge-m3 (XLM-RoBERTa encoder) model configuration.
//
// Parsed from a `general.architecture == "bert"` GGUF artifact (e.g.
// gpustack/bge-m3-GGUF/bge-m3-FP16.gguf). The generic
// GgufReader::ExtractModelConfig already reads every `bert.*` hyperparameter;
// this header only fixes the BERT-specific semantics the generic struct
// cannot know:
//
// * attention is multi-head (one KV head per Q head); GGUF carries no
//   `bert.attention.head_count_kv`, so it is set to num_attention_heads here
//   rather than inheriting the Qwen hybrid default.
// * there is no causal-interval concept: full_attention_interval is forced to
//   0 ("not a causal decoder"), which also keeps IsValidQwen() and
//   IsDenseQwenAttention() false so no Qwen path can claim this artifact.
// * pooling comes from `bert.pooling_type` (2 == CLS, the only mode bge-m3
//   artifacts declare); anything else refuses explicitly via the shared
//   core::embedding::PoolingStrategyFromGgufType mapping.
// * the leg-1 window caps single-prefill inputs (N^2 attention transient);
//   longer inputs are refused explicitly, never truncated silently.
#ifndef GUFO_MODELS_BGE_M3_CONFIG_HPP_
#define GUFO_MODELS_BGE_M3_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"

namespace gufo::models::bge_m3 {

/// Single-prefill leg-1 window: full bidirectional attention is O(n^2)
/// transient. 2048 keeps the worst-case attention matrix at 16 MiB/layer in
/// F32 while covering the retrieval-query regime; longer inputs are refused
/// explicitly (bge-m3 trains to 8192, which a later leg can enable).
inline constexpr std::uint32_t kBgeM3Leg1MaxTokens = 2048;

/// bge-m3 architectural configuration (XLM-RoBERTa, encoder-only).
struct BgeM3Config {
  std::uint32_t num_layers{24};
  std::uint32_t hidden_size{1024};
  std::uint32_t intermediate_size{4096};
  std::uint32_t num_attention_heads{16};
  std::uint32_t vocab_size{250002};
  std::uint32_t max_position_embeddings{8192};
  float layer_norm_eps{1e-5F};
  /// GGUF `bert.pooling_type` (bge-m3 artifacts declare 2 == CLS).
  std::uint32_t pooling_type{2};
  std::string model_name{"bge-m3"};
  /// Tokenizer control ids (from `tokenizer.ggml.*`, XLM-R layout).
  std::uint32_t cls_token_id{0};  ///< `<s>`
  std::uint32_t sep_token_id{2};  ///< `</s>`
  std::uint32_t pad_token_id{1};  ///< `<pad>`
  std::uint32_t unk_token_id{3};  ///< `<unk>`
};

/// Structural validation for the XLM-R encoder layout. False means "not a
/// bge-m3-shaped artifact": the loader refuses instead of guessing.
[[nodiscard]] inline bool IsBgeM3Config(const BgeM3Config& config) noexcept {
  return config.num_layers > 0 && config.hidden_size > 0 &&
         config.intermediate_size > 0 && config.num_attention_heads > 0 &&
         (config.hidden_size % config.num_attention_heads) == 0 &&
         config.vocab_size > 0 && config.max_position_embeddings > 0 &&
         config.layer_norm_eps > 0.0F;
}

/// Derives a BgeM3Config from GGUF metadata. Returns nullopt (with *error_msg
/// set when non-null) when the artifact is not `general.architecture ==
/// "bert"`, when the pooling declaration is missing/not-CLS, or when the
/// geometry fails IsBgeM3Config.
[[nodiscard]] std::optional<BgeM3Config> ExtractBgeM3Config(
    const core::GgufReader& reader, std::string* error_msg = nullptr);

/// Mirrors a BgeM3Config into the shared ModelConfig for callers that only
/// need dimensions. Deliberately marks full_attention_interval = 0 so the
/// Qwen validators (IsValidQwen / IsDenseQwenAttention) stay false.
[[nodiscard]] inline core::ModelConfig ToModelConfig(
    const BgeM3Config& config) {
  core::ModelConfig out;
  out.architecture = "bert";
  out.model_name = config.model_name;
  out.num_layers = config.num_layers;
  out.hidden_size = config.hidden_size;
  out.intermediate_size = config.intermediate_size;
  out.num_attention_heads = config.num_attention_heads;
  out.num_key_value_heads = config.num_attention_heads;  // MHA, not GQA
  out.head_dim = config.hidden_size / config.num_attention_heads;
  out.vocab_size = config.vocab_size;
  out.context_length = config.max_position_embeddings;
  out.full_attention_interval = 0;  // encoder: no causal-interval concept
  out.is_text_only = true;
  return out;
}

}  // namespace gufo::models::bge_m3

#endif  // GUFO_MODELS_BGE_M3_CONFIG_HPP_
