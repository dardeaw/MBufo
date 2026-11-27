// Qwen3-TTS talker NPU-only adapter implementation.

#include "src/core/compute/npu/qwen3_talker_decode.hpp"

namespace mbufo::core::npu {

namespace {
Qwen3NpuDecoderConfig ToGeneric(const TalkerDecodeConfig& c) {
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

bool TalkerNpuDecoder::Open(const std::string& model_dir,
                            NpuQuantGemmEngine* engine, NpuQuantMode mode,
                            const TalkerDecodeConfig& config,
                            std::string* error) {
  cfg_ = config;
  if (!dec_.Open(model_dir, engine, mode, ToGeneric(config), error)) {
    return false;
  }
  return dec_.ReadF32(cfg_.codec_head_key,
                      cfg_.codec_vocab * cfg_.hidden, &codec_head_, error);
}

bool TalkerNpuDecoder::Prefill(std::span<const float> prompt_hidden,
                                std::size_t T, std::vector<float>* last_hidden,
                                double* npu_ms, std::string* error) {
  return dec_.Prefill(prompt_hidden, T, last_hidden, npu_ms, error);
}

bool TalkerNpuDecoder::DecodeStep(std::span<const float> frame_embedding,
                                  std::vector<float>* next_hidden,
                                  double* npu_ms, std::string* error) {
  return dec_.DecodeStep(frame_embedding, next_hidden, npu_ms, error);
}

bool TalkerNpuDecoder::CodecLogits(std::span<const float> hidden,
                                   std::vector<float>* logits, double* npu_ms,
                                   std::string* error) {
  std::vector<float> normed;
  dec_.NormLastHidden(hidden, &normed);
  return dec_.HeadLogits(normed, codec_head_, cfg_.codec_vocab, logits, npu_ms,
                         error);
}

bool TalkerNpuDecoder::CodecTop1(std::span<const float> hidden,
                                 std::uint32_t* top1, double* npu_ms,
                                 std::string* error) {
  return dec_.LogitsTop1(hidden, codec_head_, cfg_.codec_vocab, top1, npu_ms,
                         error);
}

}  // namespace mbufo::core::npu
