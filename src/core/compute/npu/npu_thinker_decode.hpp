#pragma once
// ASR thinker NPU-only prefill + KV-cache decode (P1b).
//
// Unlike the parity shell (RunQwen3AsrQuantE2E: synthetic inputs, dual
// NPU+FP32 path, full-sequence only), this is the servable path:
//
//   - NPU GEMM only (no FP32 reference matmuls at serve time).
//   - Weights stream per projection via NpuShardReader (peak RAM << 2 GiB).
//   - K/V live in P0 GqaKvCache; decode attends one query row over cache.
//   - Logits + greedy top-1 come from the tied embed head block, exactly as
//     the parity shell does ([0, 2048) rows x 2048 cols).
//   - Sampling policy comes from P0 shared_sampling (greedy + EOS today).
//
// Config defaults match Qwen3-ASR-1.7B thinker: H=2048, F=6144, Q=16x128,
// KV=8x128, 28 layers, rms_eps=1e-6, rope_theta=1e6, qk_norm on,
// no layer_scale, causal on.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/kv_cache/gqa_kv_cache.hpp"
#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/npu_shard_reader.hpp"

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

  [[nodiscard]] std::size_t pos() const noexcept { return pos_; }
  [[nodiscard]] const ThinkerDecodeConfig& config() const noexcept {
    return cfg_;
  }

 private:
  [[nodiscard]] bool RunLayers(std::vector<float>* h, std::size_t T,
                              std::size_t pos_base, bool append_kv,
                              double* npu_ms, std::string* error);
  [[nodiscard]] bool LinearNpu(const std::string& weight_name,
                              std::span<const float> x, std::size_t M,
                              std::size_t N, std::size_t K,
                              std::vector<float>* y, double* npu_ms,
                              std::string* error);

  NpuQuantGemmEngine* engine_ = nullptr;
  NpuQuantMode mode_ = NpuQuantMode::kQ8Bfp16;
  ThinkerDecodeConfig cfg_;
  NpuShardReader reader_;
  kv::GqaKvCache kv_;
  std::vector<float> final_norm_w_;
  std::size_t pos_ = 0;
};

}  // namespace mbufo::core::npu
