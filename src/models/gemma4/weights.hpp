#ifndef GUFO_MODELS_GEMMA4_WEIGHTS_HPP_
#define GUFO_MODELS_GEMMA4_WEIGHTS_HPP_

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/config.hpp"

namespace gufo::models::gemma4 {

/// Non-owning view of one GGUF tensor (same convention as the Flash-Next
/// runtime: cols = ne[0] contiguous reduction, rows = ne[1], experts = ne[2]).
struct TensorRef {
  const void* data{nullptr};
  /// Device mirror of `data` (the GGUF mmap is host-only; kernels read this).
  const void* device{nullptr};
  /// True when `device` mirrors the host-mapped GGUF payload.
  bool host_mapped{false};
  std::size_t size_bytes{0};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Base address of one stacked expert matrix.
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
  /// Device base address of one stacked expert matrix.
  [[nodiscard]] const std::uint8_t* DeviceExpert(
      std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(device) + RowBytes() * rows * e;
  }
};

struct LayerWeights {
  bool swa{false};              ///< Sliding-window (KV-reusing) layer.
  bool has_kv{false};           ///< Layer owns wk/wv/attn_k_norm tensors.
  bool has_v_proj{false};       ///< wv present (else V reuses K).
  bool has_out_scale{false};    ///< per-layer output scalar present.

  TensorRef attn_norm;          ///< [hidden]
  TensorRef wq;                 ///< [hidden -> heads * head_dim]
  TensorRef wk;                 ///< [hidden -> kv_heads * head_dim]
  TensorRef wv;                 ///< [hidden -> kv_heads * head_dim]
  TensorRef wo;                 ///< [heads * head_dim -> hidden]
  TensorRef attn_q_norm;        ///< [head_dim]
  TensorRef attn_k_norm;        ///< [head_dim]
  TensorRef attn_post_norm;     ///< [hidden] blk.N.post_attention_norm
  TensorRef out_scale;          ///< [1] f32
  TensorRef attn_v_norm;        ///< [head_dim] f32 gamma (V RMSNorm)
  TensorRef rope_freqs;         ///< [head_dim_full/2] f32 (full-attn layers)

  TensorRef ffn_norm;           ///< [hidden]
  TensorRef ffn_gate;           ///< [hidden -> shared_ff]
  TensorRef ffn_up;             ///< [hidden -> shared_ff]
  TensorRef ffn_down;           ///< [shared_ff -> hidden]
  TensorRef ffn_post_norm;      ///< [hidden] blk.N.post_ffw_norm

  TensorRef router;             ///< [hidden -> experts] (ffn_gate_inp)
  TensorRef router_scale;       ///< [hidden] f32 (ffn_gate_inp.scale)
  TensorRef ffn_pre_norm_2;     ///< [hidden] blk.N.pre_ffw_norm_2
  TensorRef ffn_post_norm_1;    ///< [hidden] blk.N.post_ffw_norm_1
  TensorRef ffn_post_norm_2;    ///< [hidden] blk.N.post_ffw_norm_2
  TensorRef gate_exps;          ///< [hidden -> expert_ff] x experts
  TensorRef up_exps;            ///< [hidden -> expert_ff] x experts
  TensorRef gate_up_exps;       ///< [hidden -> 2*expert_ff] x experts (merged)
  TensorRef down_exps;          ///< [expert_ff -> hidden] x experts
  TensorRef down_exps_scale;    ///< [experts] f32 (blk.N.ffn_down_exps.scale)

  [[nodiscard]] bool IsMoe() const noexcept { return !router.empty(); }
};

struct ModelWeights {
  Config config;
  std::vector<LayerWeights> layers;
  TensorRef token_embd;   ///< [hidden -> vocab]
  TensorRef output_norm;  ///< [hidden]
  TensorRef output;       ///< [hidden -> vocab], tied to token_embd when empty

  /// KV-producing layer for each attention layer (SWA layers reuse the
  /// preceding full-attention layer's cache).
  [[nodiscard]] std::uint32_t KvLayer(std::uint32_t layer) const noexcept {
    return config.KvProducingLayer(layer);
  }

  /// Binds and validates the artifact. Tensor payloads stay mapped.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_WEIGHTS_HPP_
