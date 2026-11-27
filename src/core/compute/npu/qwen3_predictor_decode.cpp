// Qwen3-TTS code predictor NPU-only implementation.

#include "src/core/compute/npu/qwen3_predictor_decode.hpp"

#include <algorithm>

namespace mbufo::core::npu {

namespace {
Qwen3NpuDecoderConfig ToGeneric(const PredictorDecodeConfig& c) {
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

bool PredictorNpuDecoder::Open(const std::string& model_dir,
                               NpuQuantGemmEngine* engine, NpuQuantMode mode,
                               const PredictorDecodeConfig& config,
                               std::string* error) {
  cfg_ = config;
  if (!dec_.Open(model_dir, engine, mode, ToGeneric(config), error)) {
    return false;
  }
  if (!dec_.ReadF32(cfg_.proj_key, cfg_.hidden * cfg_.talker_hidden, &proj_w_,
                     error) ||
      !dec_.ReadF32(cfg_.proj_bias_key, cfg_.hidden, &proj_b_, error) ||
      !dec_.ReadF32(cfg_.group0_embed_key,
                     cfg_.group0_vocab * cfg_.talker_hidden, &group0_embed_,
                     error)) {
    return false;
  }
  group_embeds_.assign(cfg_.num_heads, {});
  heads_.assign(cfg_.num_heads, {});
  for (std::size_t g = 0; g < cfg_.num_heads; ++g) {
    const std::string gn = std::to_string(g);
    if (!dec_.ReadF32(cfg_.embed_prefix + gn + ".weight",
                       cfg_.vocab * cfg_.talker_hidden, &group_embeds_[g],
                       error) ||
        !dec_.ReadF32(cfg_.heads_prefix + gn + ".weight",
                       cfg_.vocab * cfg_.hidden, &heads_[g], error)) {
      return false;
    }
  }
  return true;
}

bool PredictorNpuDecoder::ProjectInput(std::span<const float> rows_2048,
                                       std::size_t n,
                                       std::vector<float>* out_1024,
                                       std::string* error) const {
  if (n == 0 || rows_2048.size() < n * cfg_.talker_hidden) {
    if (error != nullptr) *error = "bad predictor projection input";
    return false;
  }
  out_1024->assign(n * cfg_.hidden, 0.0F);
  for (std::size_t r = 0; r < n; ++r) {
    const float* x = rows_2048.data() + r * cfg_.talker_hidden;
    float* y = out_1024->data() + r * cfg_.hidden;
    for (std::size_t o = 0; o < cfg_.hidden; ++o) {
      const float* w = proj_w_.data() + o * cfg_.talker_hidden;
      double acc = proj_b_[o];
      for (std::size_t i = 0; i < cfg_.talker_hidden; ++i) {
        acc += static_cast<double>(w[i]) * static_cast<double>(x[i]);
      }
      y[o] = static_cast<float>(acc);
    }
  }
  return true;
}

bool PredictorNpuDecoder::Prefill(std::span<const float> rows_1024,
                                  std::size_t n,
                                  std::vector<float>* last_hidden,
                                  double* npu_ms, std::string* error) {
  return dec_.Prefill(rows_1024, n, last_hidden, npu_ms, error);
}

bool PredictorNpuDecoder::DecodeStep(std::span<const float> row_1024,
                                     std::vector<float>* next_hidden,
                                     double* npu_ms, std::string* error) {
  return dec_.DecodeStep(row_1024, next_hidden, npu_ms, error);
}

bool PredictorNpuDecoder::HeadLogits(std::size_t head,
                                     std::span<const float> hidden,
                                     std::vector<float>* logits, double* npu_ms,
                                     std::string* error) {
  if (head >= heads_.size()) {
    if (error != nullptr) *error = "predictor head out of range";
    return false;
  }
  std::vector<float> normed;
  dec_.NormLastHidden(hidden, &normed);
  return dec_.HeadLogits(normed, heads_[head], cfg_.vocab, logits, npu_ms,
                         error);
}

bool PredictorNpuDecoder::HeadTop1(std::size_t head,
                                   std::span<const float> hidden,
                                   std::uint32_t* top1, double* npu_ms,
                                   std::string* error) {
  if (head >= heads_.size()) {
    if (error != nullptr) *error = "predictor head out of range";
    return false;
  }
  return dec_.LogitsTop1(hidden, heads_[head], cfg_.vocab, top1, npu_ms,
                         error);
}

bool PredictorNpuDecoder::PredictFrame(std::span<const float> talker_hidden,
                                       std::uint32_t first_code,
                                       std::vector<std::uint32_t>* codes,
                                       std::vector<float>* head1_logits,
                                       double* npu_ms, std::string* error) {
  if (codes == nullptr || talker_hidden.size() < cfg_.talker_hidden ||
      first_code >= cfg_.group0_vocab) {
    if (error != nullptr) *error = "bad predictor frame input";
    return false;
  }
  // Seed rows: talker hidden + group-0 codec embedding, both projected.
  std::vector<float> seed(2 * cfg_.talker_hidden);
  std::copy(talker_hidden.begin(), talker_hidden.begin() + cfg_.talker_hidden,
            seed.begin());
  const float* g0 =
      group0_embed_.data() + static_cast<std::size_t>(first_code) *
                                 cfg_.talker_hidden;
  std::copy(g0, g0 + cfg_.talker_hidden,
            seed.begin() + cfg_.talker_hidden);
  std::vector<float> rows0;
  if (!ProjectInput(seed, 2, &rows0, error)) return false;

  std::vector<float> hidden;
  if (!dec_.Prefill(rows0, 2, &hidden, npu_ms, error)) return false;
  codes->clear();
  codes->reserve(cfg_.num_heads + 1);
  codes->push_back(first_code);
  std::uint32_t code = 0;
  if (!HeadTop1(0, hidden, &code, npu_ms, error)) return false;
  codes->push_back(code);
  for (std::size_t h = 1; h < heads_.size(); ++h) {
    const float* emb =
        group_embeds_[h - 1].data() + static_cast<std::size_t>(code) *
                                          cfg_.talker_hidden;
    std::vector<float> row2048(emb, emb + cfg_.talker_hidden);
    std::vector<float> row1024;
    if (!ProjectInput(row2048, 1, &row1024, error)) return false;
    if (!dec_.DecodeStep(row1024, &hidden, npu_ms, error)) return false;
    if (h == 1 && head1_logits != nullptr) {
      if (!HeadLogits(1, hidden, head1_logits, npu_ms, error)) return false;
    }
    if (!HeadTop1(h, hidden, &code, npu_ms, error)) return false;
    codes->push_back(code);
  }
  return true;
}

}  // namespace mbufo::core::npu
