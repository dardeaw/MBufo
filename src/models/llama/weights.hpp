#ifndef GUFO_MODELS_LLAMA_WEIGHTS_HPP_
#define GUFO_MODELS_LLAMA_WEIGHTS_HPP_

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/llama/config.hpp"

namespace gufo::models::llama {

/// Non-owning view of one GGUF tensor: cols = ne[0] contiguous reduction,
/// rows = ne[1]. `device` is the engine-owned device mirror (the GGUF mmap
/// is host-only); null until Model::Load mirrors it.
struct TensorRef {
  const void* data{nullptr};
  const void* device{nullptr};
  std::size_t size_bytes{0};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
};

struct LayerWeights {
  TensorRef attn_norm;  ///< [hidden]
  TensorRef wq;         ///< [hidden -> heads * head_dim]
  TensorRef wk;         ///< [hidden -> kv_heads * head_dim]
  TensorRef wv;         ///< [hidden -> kv_heads * head_dim]
  TensorRef wo;         ///< [heads * head_dim -> hidden]
  TensorRef ffn_norm;   ///< [hidden]
  TensorRef ffn_gate;   ///< [hidden -> intermediate]
  TensorRef ffn_up;     ///< [hidden -> intermediate]
  TensorRef ffn_down;   ///< [intermediate -> hidden]
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

}  // namespace gufo::models::llama

#endif  // GUFO_MODELS_LLAMA_WEIGHTS_HPP_
