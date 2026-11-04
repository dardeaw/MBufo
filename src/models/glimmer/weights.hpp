#ifndef GUFO_MODELS_GLIMMER_WEIGHTS_HPP_
#define GUFO_MODELS_GLIMMER_WEIGHTS_HPP_

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glimmer/config.hpp"

namespace gufo::models::glimmer {

/// Non-owning view of one GGUF tensor (same convention as the llama runtime:
/// cols = ne[0] contiguous reduction, rows = ne[1]).
struct TensorRef {
  const void* data{nullptr};
  const void* device{nullptr};
  std::size_t size_bytes{0};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::size_t RowBytes() const noexcept;
};

struct LayerWeights {
  TensorRef attn_norm;  ///< [hidden]
  TensorRef wq;         ///< [hidden -> heads * head_dim]
  TensorRef wk;         ///< [hidden -> kv_heads * head_dim]
  TensorRef wv;         ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_q_norm;  ///< [head_dim] QK-norm gamma
  TensorRef attn_k_norm;  ///< [head_dim] QK-norm gamma
  TensorRef gate;         ///< [hidden -> heads * head_dim] attention output gate
  TensorRef wo;           ///< [heads * head_dim -> hidden]
  TensorRef attn_post_norm;  ///< [hidden] post-attention norm gamma
  TensorRef ffn_norm;     ///< [hidden]
  TensorRef ffn_gate;     ///< [hidden -> intermediate]
  TensorRef ffn_up;       ///< [hidden -> intermediate]
  TensorRef ffn_down;     ///< [intermediate -> hidden]
  TensorRef ffn_post_norm;  ///< [hidden] post-FFN norm gamma
};

struct ModelWeights {
  Config config;
  std::vector<LayerWeights> layers;
  TensorRef token_embd;  ///< [hidden -> vocab]
  TensorRef output_norm; ///< [hidden]
  TensorRef output;      ///< [hidden -> vocab], tied to token_embd when empty

  /// Binds and validates the artifact. Tensor payloads stay mapped.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::glimmer

#endif  // GUFO_MODELS_GLIMMER_WEIGHTS_HPP_
