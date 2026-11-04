#ifndef GUFO_MODELS_GLIMMER_DRAFT_MODEL_HPP_
#define GUFO_MODELS_GLIMMER_DRAFT_MODEL_HPP_

// DFlash draft artifact description: `dflash` GGUF (depth-pruned dense
// decoder, e.g. 5 layers x hidden 6656, QK-norm, no attention gate, no
// post-norms, all-SWA). Q4_K/Q6_K weights are dequantized to F16 once at
// load (the draft is tiny); KV lives on the shared blocked-Q8 grid.
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
  std::uint32_t vocab_size{0};
  std::uint32_t context_length{0};
  float rope_theta{500000.0F};
  float rms_eps{1e-5F};
  std::uint32_t sliding_window{2048};
  std::vector<std::uint32_t> target_layer_ids;

  [[nodiscard]] static std::optional<DraftConfig> FromGguf(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);
};

struct DraftTensorRef {
  // Device-resident F16 mirror (dequantized at load when the artifact is
  // quantized; byte-copied when already F16).
  const void* device{nullptr};
  std::uint64_t cols{0};
  std::uint64_t rows{0};
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
  DraftTensorRef token_embd;
  DraftTensorRef output_norm;
  DraftTensorRef output;

  [[nodiscard]] static std::optional<DraftModelWeights> Bind(
      const core::GgufReader& reader,
      std::vector<void*>& owned /* device mirrors live here */,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::glimmer::draft

#endif  // GUFO_MODELS_GLIMMER_DRAFT_MODEL_HPP_
