// bge-reranker-v2-m3 classification-head weight table.
//
// Encoder tensors reuse the bge_m3 base verbatim (bound by
// bge_m3::EngineModel itself, never here): this file only binds the four
// Roberta classification-head tensors from the same GGUF artifact:
//
//   cls.weight         [hidden, hidden]  F32/F16/BF16 dense projection
//   cls.bias           [hidden]          F32/F16/BF16
//   cls.output.weight  [hidden]          F32/F16/BF16 ([1, hidden] flattened)
//   cls.output.bias     [1]               scalar logit bias
//
// Precision policy mirrors the encoder base: only exact F32/F16/BF16
// storage is accepted. Quantized head tensors are refused explicitly (no
// silent dequant); the served artifacts are F16.
#ifndef GUFO_MODELS_BGE_RERANKER_WEIGHTS_HPP_
#define GUFO_MODELS_BGE_RERANKER_WEIGHTS_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/bge_m3/weights.hpp"

namespace gufo::models::bge_reranker {

// Host F32 mirror of the classification head (one exact upcast per tensor
// at model load; CPU and scoring both consume these).
struct BgeRerankerHeadF32 {
  std::uint32_t hidden_size{0};
  std::vector<float> dense_w;  // [hidden, hidden] row-major
  std::vector<float> dense_b;  // [hidden]
  std::vector<float> out_w;    // [hidden]
  float out_b{0.0F};
  [[nodiscard]] bool empty() const noexcept { return dense_w.empty(); }
};

/// Binds the four head tensors from `reader` and upcasts them to F32.
/// Returns nullopt with *error_msg set on any missing tensor, unsupported
/// storage type, or shape mismatch against `hidden_size`.
[[nodiscard]] std::optional<BgeRerankerHeadF32> LoadRerankHead(
    const core::GgufReader& reader, std::uint32_t hidden_size,
    std::string* error_msg = nullptr);

}  // namespace gufo::models::bge_reranker

#endif  // GUFO_MODELS_BGE_RERANKER_WEIGHTS_HPP_
