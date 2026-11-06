// bge-m3 weight table bound from a `bert` GGUF artifact.
//
// Tensor roles (llama.cpp bert naming, verified against
// gpustack/bge-m3-GGUF/bge-m3-FP16.gguf):
//   token_embd.weight      [vocab, hidden]  F16  word embeddings
//   position_embd.weight   [max_pos, hidden] F32 learned absolute positions
//   token_types.weight     [n_types, hidden] F32 (type_vocab 1; row 0 used)
//   token_embd_norm.*      [hidden] F32 embedding LayerNorm weight+bias
//   blk.N.attn_{q,k,v,output}.{weight F16, bias F32}
//   blk.N.attn_output_norm.{weight,bias} F32 (post-attention LayerNorm)
//   blk.N.ffn_up.{weight F16 [hidden, inter], bias F32}  (intermediate)
//   blk.N.ffn_down.{weight F16 [inter, hidden], bias F32} (output)
//   blk.N.layer_output_norm.{weight,bias} F32 (post-FFN LayerNorm)
//
// Storage policy: dense F32/F16/BF16 everywhere; the six projection tables
// plus token_embd additionally accept the shared prefill quant set
// (Q4_K/Q5_K/Q6_K/Q8_0, Q4 leg). Anything else refuses explicitly (no silent
// dequant). There is no pooler (unused for CLS pooling),
// no lm_head (encoder-only, never decodes), no KV (no decode at all).
#ifndef GUFO_MODELS_BGE_M3_WEIGHTS_HPP_
#define GUFO_MODELS_BGE_M3_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/bge_m3/config.hpp"

namespace gufo::models::bge_m3 {

/// Non-owning reference to one mapped weight tensor. Storage is F32, F16,
/// or BF16 only; GetFloat upcasts exactly (one rounding for F16/BF16).
struct BgeM3TensorRef {
  const void* data = nullptr;
  core::GgmlType type = core::GgmlType::kF32;
  std::size_t num_elements = 0;
  // Raw GGUF byte size (K-quant device uploads use this, not num_elements).
  std::size_t size_bytes = 0;

  [[nodiscard]] bool empty() const noexcept {
    return data == nullptr || num_elements == 0;
  }

  [[nodiscard]] float GetFloat(std::size_t index) const noexcept {
    if (type == core::GgmlType::kF32) {
      return static_cast<const float*>(data)[index];
    }
    if (type == core::GgmlType::kF16) {
      return gufo::quant::Fp16ToFloat(
          static_cast<const std::uint16_t*>(data)[index]);
    }
    // BF16: shift into the high half of an F32.
    const auto u16 = static_cast<const std::uint16_t*>(data)[index];
    const std::uint32_t u32 = static_cast<std::uint32_t>(u16) << 16;
    float out = 0.0F;
    __builtin_memcpy(&out, &u32, sizeof(float));
    return out;
  }

  [[nodiscard]] std::span<const float> AsFloatSpan() const noexcept {
    if (type == core::GgmlType::kF32 && data != nullptr) {
      return {static_cast<const float*>(data), num_elements};
    }
    return {};
  }
};

/// One encoder layer's weights (16 tensors, all verified present by the
/// loader; see the header comment for roles).
struct BgeM3LayerWeights {
  BgeM3TensorRef attn_q_weight;
  BgeM3TensorRef attn_q_bias;
  BgeM3TensorRef attn_k_weight;
  BgeM3TensorRef attn_k_bias;
  BgeM3TensorRef attn_v_weight;
  BgeM3TensorRef attn_v_bias;
  BgeM3TensorRef attn_output_weight;
  BgeM3TensorRef attn_output_bias;
  BgeM3TensorRef attn_output_norm_weight;
  BgeM3TensorRef attn_output_norm_bias;
  BgeM3TensorRef ffn_up_weight;
  BgeM3TensorRef ffn_up_bias;
  BgeM3TensorRef ffn_down_weight;
  BgeM3TensorRef ffn_down_bias;
  BgeM3TensorRef layer_output_norm_weight;
  BgeM3TensorRef layer_output_norm_bias;
};

/// Full bge-m3 weight table mapped from GGUF storage (non-owning; the caller
/// keeps the GgufReader alive).
struct BgeM3Weights {
  BgeM3Config config;
  BgeM3TensorRef token_embd;
  BgeM3TensorRef position_embd;
  BgeM3TensorRef token_types;
  BgeM3TensorRef token_embd_norm_weight;
  BgeM3TensorRef token_embd_norm_bias;
  std::vector<BgeM3LayerWeights> layers;

  [[nodiscard]] static std::optional<BgeM3Weights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr,
      BgeM3ConfigOptions options = {});
};

/// Quant-aware bulk upcast of one tensor to F32 (load-time only).
/// F32 copies; F16/BF16 upcast exactly; K-quants (Q4_K/Q5_K/Q6_K/Q8_0)
/// dequantize row by row with K width `row_elems` (num_elements must be a
/// multiple of row_elems, row_elems a multiple of the quant block).
/// Anything else refuses. LoadFromGguf validates geometry first, so UpcastAll
/// treats failure as unreachable (empty vector, fails loudly downstream).
[[nodiscard]] bool UpcastTensorToF32(const BgeM3TensorRef& ref,
                                     std::vector<float>& out,
                                     std::size_t row_elems,
                                     std::string* error_msg = nullptr);

}  // namespace gufo::models::bge_m3

#endif  // GUFO_MODELS_BGE_M3_WEIGHTS_HPP_
