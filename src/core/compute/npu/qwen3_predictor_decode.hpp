#pragma once
// Qwen3-TTS code predictor NPU-only (P2b).
//
// 5-layer Qwen3 (H=1024/F=3072/Q=16x128/KV=8x128) that expands one
// first-codebook token into the remaining 15 codec groups. Flow mirrors
// TalkerHipRuntime::PredictFrame exactly:
//
//   row0 = small_to_mtp_projection(talker_hidden[2048])      -> [1024]
//   row1 = small_to_mtp_projection(codec_embed_group0[c0])    -> [1024]
//   prefill [row0, row1] at pos 0, head0 top-1 -> code1
//   for h = 1..14: row = proj(predictor_embed[h-1][code_h]) -> [1024]
//                   decode at pos h+1, head_h top-1 -> code_{h+1}
//
// codes out = {c0, code1, ..., code15} (16 groups). Host-side projection
// (1024x2048 matvec, <=2 rows) is negligible next to 5 NPU layers.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/qwen3_decoder_npu.hpp"

namespace mbufo::core::npu {

struct PredictorDecodeConfig {
  std::string layers_prefix = "talker.code_predictor.model.layers";
  std::string norm_key = "talker.code_predictor.model.norm.weight";
  std::string proj_key = "talker.code_predictor.small_to_mtp_projection.weight";
  std::string proj_bias_key = "talker.code_predictor.small_to_mtp_projection.bias";
  std::string group0_embed_key = "talker.model.codec_embedding.weight";
  std::string embed_prefix = "talker.code_predictor.model.codec_embedding.";
  std::string heads_prefix = "talker.code_predictor.lm_head.";
  std::size_t layers = 5;
  std::size_t hidden = 1024;
  std::size_t inter = 3072;
  std::size_t n_heads = 16;
  std::size_t n_kv_heads = 8;
  std::size_t head_dim = 128;
  float rms_eps = 1.0e-6F;
  float rope_theta = 1000000.0F;
  std::size_t vocab = 2048;       // per-group codebook
  std::size_t group0_vocab = 3072;
  std::size_t talker_hidden = 2048;
  std::size_t num_heads = 15;     // code groups 1..15
  std::size_t max_seq = 32;
};

class PredictorNpuDecoder {
 public:
  PredictorNpuDecoder() = default;

  [[nodiscard]] bool Open(const std::string& model_dir,
                          NpuQuantGemmEngine* engine, NpuQuantMode mode,
                          const PredictorDecodeConfig& config = {},
                          std::string* error = nullptr);

  /// Project n 2048-dim rows through small_to_mtp_projection (+bias).
  [[nodiscard]] bool ProjectInput(std::span<const float> rows_2048,
                                  std::size_t n, std::vector<float>* out_1024,
                                  std::string* error = nullptr) const;

  [[nodiscard]] bool Prefill(std::span<const float> rows_1024, std::size_t n,
                             std::vector<float>* last_hidden,
                             double* npu_ms = nullptr,
                             std::string* error = nullptr);

  [[nodiscard]] bool DecodeStep(std::span<const float> row_1024,
                                std::vector<float>* next_hidden,
                                double* npu_ms = nullptr,
                                std::string* error = nullptr);

  [[nodiscard]] bool HeadLogits(std::size_t head, std::span<const float> hidden,
                                std::vector<float>* logits,
                                double* npu_ms = nullptr,
                                std::string* error = nullptr);

  [[nodiscard]] bool HeadTop1(std::size_t head, std::span<const float> hidden,
                              std::uint32_t* top1,
                              double* npu_ms = nullptr,
                              std::string* error = nullptr);

  /// Full 16-group frame expansion. Optional head-1 logit trace for gating.
  [[nodiscard]] bool PredictFrame(std::span<const float> talker_hidden,
                                  std::uint32_t first_code,
                                  std::vector<std::uint32_t>* codes,
                                  std::vector<float>* head1_logits = nullptr,
                                  double* npu_ms = nullptr,
                                  std::string* error = nullptr);

  [[nodiscard]] std::size_t pos() const noexcept { return dec_.pos(); }
  [[nodiscard]] const PredictorDecodeConfig& config() const noexcept {
    return cfg_;
  }
  /// Predictor group embedding tables, index g = 0..14 backing code groups
  /// 1..15. The chained prediction loop looks up tables 0..13 (group 0 is
  /// seeded from the talker codec table); talker frame-input building sums
  /// all 15. Mirrors TalkerHipRuntime::embedding_tables[1..15].
  [[nodiscard]] const std::vector<std::vector<float>>& group_embeds()
      const noexcept {
    return group_embeds_;
  }
  /// Group-0 codec table [group0_vocab, talker_hidden] for frame building.
  [[nodiscard]] std::span<const float> group0_embed() const noexcept {
    return group0_embed_;
  }

 private:
  Qwen3NpuDecoder dec_;
  PredictorDecodeConfig cfg_;
  std::vector<float> proj_w_;  // [1024, 2048]
  std::vector<float> proj_b_;  // [1024]
  std::vector<float> group0_embed_;  // [3072, 2048]
  std::vector<std::vector<float>> group_embeds_;  // 15 x [2048, 2048]
  std::vector<std::vector<float>> heads_;         // 15 x [2048, 1024]
};

}  // namespace mbufo::core::npu
