#ifndef MBUFO_MODELS_LAGUNA_WEIGHTS_HPP_
#define MBUFO_MODELS_LAGUNA_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/laguna/config.hpp"

namespace mbufo::models::laguna {

struct TensorRef {
  const void* data{nullptr};
  void* device{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::size_t size_bytes{0};
  std::uint64_t rows{0};
  std::uint64_t cols{0};
  std::uint64_t depth{1};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t RowBytes() const noexcept {
    if (rows == 0 || depth == 0) return 0;
    return size_bytes / (rows * depth);
  }
  [[nodiscard]] std::uint64_t ExpertStrideBytes() const noexcept {
    if (depth == 0) return 0;
    return size_bytes / depth;
  }
};

struct LayerWeights {
  std::uint32_t q_heads{48};
  bool is_swa{false};
  bool is_moe{false};

  TensorRef attn_norm;
  TensorRef wq;
  TensorRef wk;
  TensorRef wv;
  TensorRef wo;
  TensorRef attn_q_norm;
  TensorRef attn_k_norm;
  TensorRef attn_gate;

  TensorRef ffn_norm;
  // Dense (layer < leading_dense_blocks)
  TensorRef ffn_gate;
  TensorRef ffn_up;
  TensorRef ffn_down;

  // MoE (layer >= leading_dense_blocks)
  TensorRef ffn_gate_inp;
  TensorRef exp_probs_b;
  TensorRef ffn_gate_exps;
  TensorRef ffn_up_exps;
  TensorRef ffn_down_exps;
  TensorRef ffn_gate_shexp;
  TensorRef ffn_up_shexp;
  TensorRef ffn_down_shexp;
};

struct ModelWeights {
  LagunaConfig config;
  TensorRef token_embd;
  TensorRef output_norm;
  TensorRef output;
  std::vector<LayerWeights> layers;

  [[nodiscard]] static std::optional<ModelWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace mbufo::models::laguna

#endif  // MBUFO_MODELS_LAGUNA_WEIGHTS_HPP_
