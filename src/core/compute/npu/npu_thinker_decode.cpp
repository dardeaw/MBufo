// ASR thinker NPU-only adapter implementation (thin wrapper over the
// shared Qwen3NpuDecoder core).

#include "src/core/compute/npu/npu_thinker_decode.hpp"

namespace mbufo::core::npu {

namespace {
Qwen3NpuDecoderConfig ToGeneric(const ThinkerDecodeConfig& c) {
  Qwen3NpuDecoderConfig g;
  g.layers_prefix = c.layers_prefix;
  g.norm_key = c.norm_name;
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

bool ThinkerNpuDecoder::Open(const std::string& model_dir,
                             NpuQuantGemmEngine* engine, NpuQuantMode mode,
                             const ThinkerDecodeConfig& config,
                             std::string* error) {
  cfg_ = config;
  return dec_.Open(model_dir, engine, mode, ToGeneric(config), error);
}

bool ThinkerNpuDecoder::Prefill(std::span<const float> prompt_hidden,
                                std::size_t T, std::vector<float>* last_hidden,
                                double* npu_ms, std::string* error) {
  return dec_.Prefill(prompt_hidden, T, last_hidden, npu_ms, error);
}

bool ThinkerNpuDecoder::DecodeStep(std::span<const float> embed_row,
                                   std::vector<float>* next_hidden,
                                   double* npu_ms, std::string* error) {
  return dec_.DecodeStep(embed_row, next_hidden, npu_ms, error);
}

bool ThinkerNpuDecoder::LogitsTop1(std::span<const float> hidden,
                                   std::uint32_t* top1, double* npu_ms,
                                   std::string* error) {
  std::vector<float> head;
  if (!dec_.ReadRowSliceF32(cfg_.embed_name, 0, cfg_.vocab_block_rows,
                            cfg_.hidden, &head, error)) {
    return false;
  }
  return dec_.LogitsTop1(hidden, head, cfg_.vocab_block_rows, top1, npu_ms,
                         error);
}

}  // namespace mbufo::core::npu
