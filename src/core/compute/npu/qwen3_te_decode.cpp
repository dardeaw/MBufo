// Shared Qwen3-4B text-encoder NPU adapter implementation.

#include "src/core/compute/npu/qwen3_te_decode.hpp"

namespace mbufo::core::npu {

namespace {
Qwen3NpuDecoderConfig ToGeneric(const TeDecodeConfig& c) {
  Qwen3NpuDecoderConfig g;
  g.layers_prefix = c.layers_prefix;
  g.norm_key = c.norm_key;
  g.layers = c.layers;
  g.hidden = c.hidden;
  g.inter = c.inter;
  g.n_heads = c.n_heads;
  g.n_kv_heads = c.n_kv_heads;
  g.head_dim = c.head_dim;
  g.rms_eps = c.rms_eps;
  g.rope_theta = c.rope_theta;
  g.max_seq = c.max_seq;
  return g;
}
}  // namespace

bool TeNpuDecoder::Open(const std::string& model_dir,
                        NpuQuantGemmEngine* engine, NpuQuantMode mode,
                        const TeDecodeConfig& config, std::string* error) {
  cfg_ = config;
  return dec_.Open(model_dir, engine, mode, ToGeneric(config), error);
}

bool TeNpuDecoder::Encode(std::span<const float> prompt_hidden, std::size_t T,
                          std::vector<float>* all_hidden, double* npu_ms,
                          std::string* error) {
  std::vector<float> raw;
  if (!dec_.PrefillFull(prompt_hidden, T, &raw, npu_ms, error)) return false;
  all_hidden->resize(T * cfg_.hidden);
  std::vector<float> normed;
  for (std::size_t t = 0; t < T; ++t) {
    dec_.NormLastHidden(
        std::span<const float>(raw.data() + t * cfg_.hidden, cfg_.hidden),
        &normed);
    std::copy(normed.begin(), normed.end(),
              all_hidden->data() + t * cfg_.hidden);
  }
  return true;
}

bool TeNpuDecoder::Prefill(std::span<const float> prompt_hidden, std::size_t T,
                            std::vector<float>* last_hidden, double* npu_ms,
                            std::string* error) {
  return dec_.Prefill(prompt_hidden, T, last_hidden, npu_ms, error);
}

bool TeNpuDecoder::DecodeStep(std::span<const float> embed_row,
                              std::vector<float>* next_hidden, double* npu_ms,
                              std::string* error) {
  return dec_.DecodeStep(embed_row, next_hidden, npu_ms, error);
}

}  // namespace mbufo::core::npu
