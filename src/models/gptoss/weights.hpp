#ifndef GUFO_MODELS_GPTOSS_WEIGHTS_HPP_
#define GUFO_MODELS_GPTOSS_WEIGHTS_HPP_

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gptoss/config.hpp"

namespace gufo::models::gptoss {

// Non-owning view of one GGUF tensor (same convention as gemma4: cols =
// ne[0] contiguous reduction, rows = ne[1], experts = ne[2]). Converted
// tensors (dense Q8_0 arenas) point into ModelWeights-owned arenas;
// `device` is the engine-owned device mirror (null until Model::Load).
struct TensorRef {
  const void* data{nullptr};
  const void* device{nullptr};
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
  TensorRef attn_norm;      ///< [hidden] F32
  TensorRef wq;             ///< [hidden -> 4096] Q8_0 (converted)
  TensorRef wk;             ///< [hidden -> 512] Q8_0 (converted)
  TensorRef wv;             ///< [hidden -> 512] Q8_0 (converted)
  TensorRef wo;             ///< [4096 -> hidden] Q8_0 (converted)
  TensorRef wo_b;           ///< [hidden] F32 attention out bias
  TensorRef attn_sinks;     ///< [64] F32 attention sinks
  TensorRef post_norm;      ///< [hidden] F32 post_attention_norm
  TensorRef router;         ///< [hidden -> 32] F32 router (ffn_gate_inp)
  TensorRef router_b;       ///< [32] F32 router bias
  TensorRef gate_b;         ///< [2880 x 32] F32 expert gate bias
  TensorRef up_b;           ///< [2880 x 32] F32 expert up bias
  TensorRef down_b;         ///< [2880 x 32] F32 expert down bias
  // Merged gate+up expert slab [2*expert_ff][hidden] x experts, native
  // MXFP4 (memcpy-merged, no conversion): first expert_ff rows are gate.
  TensorRef gate_up_exps;
  TensorRef down_exps;      ///< [hidden][expert_ff] x experts, native MXFP4
};

struct ModelWeights {
  Config config;
  std::vector<LayerWeights> layers;
  TensorRef token_embd;   ///< [hidden -> vocab] Q8_0 (converted)
  TensorRef output_norm;  ///< [hidden] F32
  TensorRef output;       ///< [hidden -> vocab] Q8_0, tied to token_embd

  /// Binds and validates the artifact. Dense K-quants convert to Q8_0 on
  /// the host; experts stay native MXFP4. Converted payloads are owned
  /// here (arenas) and stay alive for the model lifetime.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Allocates a conversion arena owned by this weights object.
  std::vector<std::uint8_t>* NewArena(std::size_t bytes) {
    if (arenas_.empty()) arenas_.reserve(512);
    arenas_.emplace_back(bytes);
    return &arenas_.back();
  }

 private:
  std::vector<std::vector<std::uint8_t>> arenas_;
};

}  // namespace gufo::models::gptoss

#endif  // GUFO_MODELS_GPTOSS_WEIGHTS_HPP_
