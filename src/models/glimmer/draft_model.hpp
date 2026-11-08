#ifndef GUFO_MODELS_GLIMMER_DRAFT_MODEL_HPP_
#define GUFO_MODELS_GLIMMER_DRAFT_MODEL_HPP_

// DFlash draft artifact description: `dflash` GGUF (EAGLE-style feature
// draft, e.g. 5 layers x hidden 6656, QK-norm, no attention gate, no
// post-norms, all-SWA). The draft consumes fused TARGET hidden states —
// `fc.weight` maps the concatenated tapped-layer features
// [n_taps * hidden] to the model width — and predicts with the TARGET's
// output head, so the artifact carries no embedding and no vocabulary
// head of its own (vocab comes from the target at backend bind time).
// Q4_K/Q6_K weights are dequantized to F32 once at load (the draft is
// tiny); KV lives on the shared blocked-Q8 grid.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glimmer::draft {

struct DraftConfig {
  std::uint32_t num_layers{0};
  std::uint32_t hidden_size{0};
  std::uint32_t intermediate_size{0};
  std::uint32_t num_heads{0};
  std::uint32_t num_kv_heads{0};
  std::uint32_t head_dim{0};
  // No vocabulary in the artifact (the draft predicts with the target
  // head); the backend fills this from the target model.
  std::uint32_t vocab_size{0};
  std::uint32_t context_length{0};
  float rope_theta{500000.0F};
  float rms_eps{1e-5F};
  std::uint32_t sliding_window{2048};
  // GGUF layer-INPUT numbering (1-based); the backend normalizes these to
  // zero-based tapped layer outputs (27B DFlash2 convention).
  std::vector<std::uint32_t> target_layer_ids;
  // Block-diffusion mask token (`tokenizer.ggml.mask_token_id` in the
  // artifact): proposals decode [id_last, mask * k] in one block.
  std::uint32_t mask_token_id{0};

  [[nodiscard]] static std::optional<DraftConfig> FromGguf(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);
};

struct DraftTensorRef {
  // Device-resident F32 mirror (dequantized at load when the artifact is
  // quantized; byte-copied when already F32). Matrices follow the GGUF
  // [K, M] convention (reduction dim contiguous): GEMM callers pass
  // m = rows(out-dim), k = cols(in-dim) with the matching config sizes.
  const void* device{nullptr};
  std::uint64_t cols{0};  ///< file dims[0] (reduction dim)
  std::uint64_t rows{0};  ///< file dims[1] (output dim)
  [[nodiscard]] bool empty() const noexcept { return device == nullptr; }
};

struct DraftLayerWeights {
  DraftTensorRef attn_norm;
  DraftTensorRef wq;
  DraftTensorRef wk;
  DraftTensorRef wv;
  DraftTensorRef attn_q_norm;
  DraftTensorRef attn_k_norm;
  DraftTensorRef wo;
  DraftTensorRef ffn_norm;
  DraftTensorRef ffn_gate;
  DraftTensorRef ffn_up;
  DraftTensorRef ffn_down;
};

struct DraftModelWeights {
  DraftConfig config;
  std::vector<DraftLayerWeights> layers;
  /// Feature-fusion projection [n_taps * hidden -> hidden] (file order).
  DraftTensorRef fc;
  /// RMSNorm over the fused features (encoder norm).
  DraftTensorRef enc_norm;
  DraftTensorRef output_norm;

  [[nodiscard]] static std::optional<DraftModelWeights> Bind(
      const core::GgufReader& reader,
      std::vector<void*>& owned /* device mirrors live here */,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::glimmer::draft

#endif  // GUFO_MODELS_GLIMMER_DRAFT_MODEL_HPP_
