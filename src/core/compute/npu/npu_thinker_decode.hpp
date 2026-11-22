#pragma once
// ASR thinker NPU-only prefill + KV-cache decode (P1b).
//
// Thin model adapter over the shared Qwen3NpuDecoder core
// (qwen3_decoder_npu.hpp): same servable path (NPU GEMM only, streamed
// weights, GqaKvCache, greedy top-1 over the tied embed head block), same
// config defaults (H=2048, F=6144, Q=16x128, KV=8x128, 28 layers).

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/qwen3_decoder_npu.hpp"

namespace mbufo::core::npu {

struct ThinkerDecodeConfig {
  std::string layers_prefix = "thinker.model.layers";
  std::string norm_name = "thinker.model.norm.weight";
  std::string embed_name = "thinker.model.embed_tokens.weight";
  std::size_t layers = 28;
  std::size_t hidden = 2048;
  std::size_t inter = 6144;
  std::size_t n_heads = 16;
  std::size_t n_kv_heads = 8;
  std::size_t head_dim = 128;
  float rms_eps = 1.0e-6F;
  float rope_theta = 1000000.0F;
  std::size_t vocab_block_rows = 2048;
  std::size_t max_seq = 512;
};

class ThinkerNpuDecoder {
 public:
  ThinkerNpuDecoder() = default;

  [[nodiscard]] bool Open(const std::string& model_dir,
                          NpuQuantGemmEngine* engine, NpuQuantMode mode,
                          const ThinkerDecodeConfig& config = {},
                          std::string* error = nullptr);

  /// Prefill `T` prompt hidden rows [T, hidden]; fills KV cache, sets pos=T,
  /// returns final-layer hidden of the last prompt row.
  [[nodiscard]] bool Prefill(std::span<const float> prompt_hidden,
                             std::size_t T, std::vector<float>* last_hidden,
                             double* npu_ms = nullptr,
                             std::string* error = nullptr);

  /// Single decode step at absolute position pos(): embed row -> next hidden.
  [[nodiscard]] bool DecodeStep(std::span<const float> embed_row,
                                std::vector<float>* next_hidden,
                                double* npu_ms = nullptr,
                                std::string* error = nullptr);

  /// LM-head top-1 over the tied embed head block.
  [[nodiscard]] bool LogitsTop1(std::span<const float> hidden,
                                std::uint32_t* top1,
                                double* npu_ms = nullptr,
                                std::string* error = nullptr);

  [[nodiscard]] std::size_t pos() const noexcept { return dec_.pos(); }
  [[nodiscard]] const ThinkerDecodeConfig& config() const noexcept {
    return cfg_;
  }

 private:
  Qwen3NpuDecoder dec_;
  ThinkerDecodeConfig cfg_;
};

}  // namespace mbufo::core::npu
