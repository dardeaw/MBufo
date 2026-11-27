#pragma once
// Qwen3-TTS talker NPU-only prefill + KV-cache decode (P2a).
//
// Same 28-layer Qwen3 shape as the ASR thinker (H=2048/F=6144/Q=16x128/
// KV=8x128), different weights: prefix "talker.model.layers", final norm
// "talker.model.norm", codec LM head "talker.codec_head" [3072, 2048].
// Prompt embeddings (text_projection + prompt builder) stay on the existing
// HIP path; this file owns the autoregressive decoder core: prefill prompt
// embeddings, decode one codec-frame embedding per step, codec logits.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/qwen3_decoder_npu.hpp"

namespace mbufo::core::npu {

struct TalkerDecodeConfig {
  std::string layers_prefix = "talker.model.layers";
  std::string norm_key = "talker.model.norm.weight";
  std::string codec_head_key = "talker.codec_head.weight";
  std::string codec_embed_key = "talker.model.codec_embedding.weight";
  std::size_t layers = 28;
  std::size_t hidden = 2048;
  std::size_t inter = 6144;
  std::size_t n_heads = 16;
  std::size_t n_kv_heads = 8;
  std::size_t head_dim = 128;
  float rms_eps = 1.0e-6F;
  float rope_theta = 1000000.0F;
  std::size_t codec_vocab = 3072;
  std::size_t num_code_groups = 16;
  std::size_t max_seq = 4096;
};

class TalkerNpuDecoder {
 public:
  TalkerNpuDecoder() = default;

  [[nodiscard]] bool Open(const std::string& model_dir,
                          NpuQuantGemmEngine* engine, NpuQuantMode mode,
                          const TalkerDecodeConfig& config = {},
                          std::string* error = nullptr);

  [[nodiscard]] bool Prefill(std::span<const float> prompt_hidden,
                             std::size_t T, std::vector<float>* last_hidden,
                             double* npu_ms = nullptr,
                             std::string* error = nullptr);

  [[nodiscard]] bool DecodeStep(std::span<const float> frame_embedding,
                                std::vector<float>* next_hidden,
                                double* npu_ms = nullptr,
                                std::string* error = nullptr);

  /// Full [codec_vocab] logit row for one hidden state.
  [[nodiscard]] bool CodecLogits(std::span<const float> hidden,
                                 std::vector<float>* logits,
                                 double* npu_ms = nullptr,
                                 std::string* error = nullptr);

  [[nodiscard]] bool CodecTop1(std::span<const float> hidden,
                               std::uint32_t* top1,
                               double* npu_ms = nullptr,
                               std::string* error = nullptr);

  [[nodiscard]] std::size_t pos() const noexcept { return dec_.pos(); }
  [[nodiscard]] const TalkerDecodeConfig& config() const noexcept {
    return cfg_;
  }
  [[nodiscard]] const std::vector<float>& codec_head() const noexcept {
    return codec_head_;
  }

 private:
  Qwen3NpuDecoder dec_;
  TalkerDecodeConfig cfg_;
  std::vector<float> codec_head_;  // [codec_vocab, hidden] f32
};

/// Frame input embedding: text_embedding + codec_embed[c0] + sum of the
/// predictor code-group embeddings (tables owned by the predictor).
/// Mirrors TalkerHipRuntime::PrepareCodeFrameEmbedding.
[[nodiscard]] inline bool BuildTalkerFrameInput(
    std::span<const std::uint32_t> codes, std::span<const float> text_embedding,
    std::span<const float> codec_embed_group0,
    const std::vector<std::vector<float>>& predictor_group_embeds,
    std::size_t num_code_groups, std::size_t hidden,
    std::vector<float>* out, std::string* error = nullptr) {
  if (codes.size() != num_code_groups ||
      text_embedding.size() != hidden ||
      codec_embed_group0.size() < 3072 * hidden ||
      predictor_group_embeds.size() + 1 != num_code_groups) {
    if (error != nullptr) *error = "bad talker frame input";
    return false;
  }
  out->assign(text_embedding.begin(), text_embedding.end());
  auto add_row = [&](const float* table, std::uint32_t id, std::size_t vocab) {
    if (id >= vocab) return false;
    const float* row = table + static_cast<std::size_t>(id) * hidden;
    for (std::size_t i = 0; i < hidden; ++i) (*out)[i] += row[i];
    return true;
  };
  if (!add_row(codec_embed_group0.data(), codes[0], 3072)) {
    if (error != nullptr) *error = "group0 code out of range";
    return false;
  }
  for (std::size_t g = 1; g < num_code_groups; ++g) {
    // Predictor group (g-1) vocab is 2048.
    if (!add_row(predictor_group_embeds[g - 1].data(), codes[g], 2048)) {
      if (error != nullptr) *error = "predictor code out of range";
      return false;
    }
  }
  return true;
}

}  // namespace mbufo::core::npu
