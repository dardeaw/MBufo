#pragma once
// Shared Qwen3-4B text-encoder NPU path (P3).
//
// FLUX.2-klein and Z-Image-Turbo ship byte-identical Qwen3-4B text encoders
// (H=2560/F=9728/Q=32x128/KV=8x128, 36 layers, verified by weight hash), so
// one adapter serves both DiTs. The encoder is prefill-only: Encode() runs
// the full prompt through all 36 layers on NPU silicon and returns the
// final-normed [T, hidden] sequence for cross-attention conditioning.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/qwen3_decoder_npu.hpp"

namespace mbufo::core::npu {

struct TeDecodeConfig {
  std::string layers_prefix = "model.layers";
  std::string norm_key = "model.norm.weight";
  std::string embed_key = "model.embed_tokens.weight";
  std::size_t layers = 36;
  std::size_t hidden = 2560;
  std::size_t inter = 9728;
  std::size_t n_heads = 32;
  std::size_t n_kv_heads = 8;
  std::size_t head_dim = 128;
  float rms_eps = 1.0e-6F;
  float rope_theta = 1000000.0F;
  std::size_t max_seq = 512;
};

class TeNpuDecoder {
 public:
  TeNpuDecoder() = default;

  [[nodiscard]] bool Open(const std::string& model_dir,
                          NpuQuantGemmEngine* engine, NpuQuantMode mode,
                          const TeDecodeConfig& config = {},
                          std::string* error = nullptr);

  /// Full-sequence encode: [T, hidden] final-normed conditioning rows.
  [[nodiscard]] bool Encode(std::span<const float> prompt_hidden,
                            std::size_t T, std::vector<float>* all_hidden,
                            double* npu_ms = nullptr,
                            std::string* error = nullptr);

  // Prefill/DecodeStep exposed for the decode-consistency gate (same
  // prefill-vs-incremental math check as P1/P2a on TE weights).
  [[nodiscard]] bool Prefill(std::span<const float> prompt_hidden,
                             std::size_t T, std::vector<float>* last_hidden,
                             double* npu_ms = nullptr,
                             std::string* error = nullptr);
  [[nodiscard]] bool DecodeStep(std::span<const float> embed_row,
                                std::vector<float>* next_hidden,
                                double* npu_ms = nullptr,
                                std::string* error = nullptr);

  [[nodiscard]] std::size_t pos() const noexcept { return dec_.pos(); }
  [[nodiscard]] const TeDecodeConfig& config() const noexcept { return cfg_; }

 private:
  Qwen3NpuDecoder dec_;
  TeDecodeConfig cfg_;
};

}  // namespace mbufo::core::npu
