#ifndef MBUFO_MODELS_INKLING_WEIGHTS_HPP_
#define MBUFO_MODELS_INKLING_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/inkling/config.hpp"

namespace mbufo::models::inkling {

// GGUF type code for NVFP4 (4 E2M1 blocks + E4M3 scales, QK=64). The shared
// core GgmlType vocabulary has no NVFP4 entry and core carries no NVFP4
// dequant: inkling converts these tensors to Q8_0 once at load time
// (host dequant to F32, then core stock QuantizeQ8_0; see
// kernels::ConvertNvfp4TensorToF32Host), so no core change is needed.
inline constexpr int kGgmlTypeNvfp4Raw = 40;

[[nodiscard]] inline bool IsNvfp4(core::GgmlType t) noexcept {
  return static_cast<int>(t) == kGgmlTypeNvfp4Raw;
}

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

struct LoopTensors {
  TensorRef down;    // [4608, 64] down-projection to gate rank
  TensorRef forget;  // [64, 2304] forget gate
  TensorRef gain;    // [64, 2304] message gain
};

struct LayerWeights {
  std::uint32_t q_heads{24};
  bool is_swa{false};
  bool is_loop_core{false};  // physical blk.6-17
  bool is_nvfp4{false};      // blk.0-19 (blk.20-23 experts are Q8_0)

  // Attention core (SwiGLU-style gate, rel-pos, sconv, per-head sink).
  TensorRef attn_norm;  // absent on SWA layers (replaced by *_inp_norm)
  TensorRef wq;
  TensorRef wk;
  TensorRef wv;
  TensorRef wo;
  TensorRef attn_gate;
  TensorRef attn_rel;
  TensorRef attn_rel_bias;
  TensorRef attn_rel_clip;
  TensorRef attn_q_clip;
  TensorRef attn_sink_proj;  // BF16 in GGUF, converted to F16 at load
  TensorRef attn_k_conv;
  TensorRef attn_v_conv;
  TensorRef attn_post_conv;
  // SWA-boundary layers only.
  TensorRef attn_q_inp_norm;
  TensorRef attn_k_inp_norm;
  TensorRef attn_v_inp_norm;
  TensorRef attn_k_conv_proj;  // BF16 in GGUF, converted to F16 at load
  TensorRef attn_v_conv_proj;  // BF16 in GGUF, converted to F16 at load
  // Absent on blk.0; [2304,3] on SWA layers, [2304] elsewhere.
  TensorRef attn_res_query;

  // FFN / message path.
  TensorRef ffn_norm;
  TensorRef ffn_post_conv;
  TensorRef ffn_res_query;

  // MoE routing + experts (fused gate+up).
  TensorRef ffn_gate_inp;
  TensorRef exp_probs_b;
  TensorRef ffn_gate_up_exps;
  TensorRef ffn_gate_up_scale;  // F32[experts]; absent on Q8_0 layers
  TensorRef ffn_down_exps;
  TensorRef ffn_down_scale;  // F32[experts]; absent on Q8_0 layers

  // Loop-core message tensors (physical blk.6-17 only).
  LoopTensors loop_attn;
  LoopTensors loop_k;
  LoopTensors loop_mlp;
  LoopTensors loop_v;
};

struct ModelWeights {
  InklingConfig config;
  TensorRef token_embd;
  TensorRef token_embd_norm;
  TensorRef output;
  TensorRef output_norm;
  TensorRef loop_kv_res_query;
  TensorRef loop_res_query;
  TensorRef output_res_query;
  std::vector<LayerWeights> layers;

  [[nodiscard]] static std::optional<ModelWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace mbufo::models::inkling

#endif  // MBUFO_MODELS_INKLING_WEIGHTS_HPP_