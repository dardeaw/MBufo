// P2a: talker NPU-only prefill + 1-step KV decode consistency gate.
//
// Path A (servable): Prefill(T=4) then DecodeStep(5th row) through the KV
// cache. Path B (reference): Prefill(T=5) full sequence. Both run ALL 28
// talker layers on NPU silicon via TalkerNpuDecoder.
// Gate: cosine(codec_logits_A, codec_logits_B) >= Q8 0.998 / Q4 0.965 +
// top-1 agreement over the 3072-entry codec head.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/qwen3_talker_decode.hpp"

namespace {

float Cosine(std::span<const float> a, std::span<const float> b) {
  assert(a.size() == b.size() && !a.empty());
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

void TestTalkerConsistency(mbufo::core::npu::NpuQuantGemmEngine& engine,
                           mbufo::core::npu::NpuQuantMode mode) {
  using mbufo::core::npu::NpuQuantModeName;
  using mbufo::core::npu::TalkerNpuDecoder;
  const std::string model_dir = "/mnt/nvme-data/mbufo-models/qwen3-tts-1.7b-base";
  constexpr std::size_t kHidden = 2048;
  std::string err;

  // Synthetic prompt embeddings (small uniform noise, same for both paths).
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-0.1F, 0.1F);
  std::vector<float> rows(5 * kHidden);
  for (float& v : rows) v = dist(rng);

  TalkerNpuDecoder dec_a;
  assert(dec_a.Open(model_dir, &engine, mode, {}, &err));
  double ms_a = 0.0;
  std::vector<float> h4;
  assert(dec_a.Prefill(rows, 4, &h4, &ms_a, &err));
  assert(dec_a.pos() == 4);
  std::vector<float> row5(rows.begin() + 4 * kHidden, rows.end());
  std::vector<float> h5_inc;
  assert(dec_a.DecodeStep(row5, &h5_inc, &ms_a, &err));
  assert(dec_a.pos() == 5);
  std::vector<float> logits_inc;
  assert(dec_a.CodecLogits(h5_inc, &logits_inc, &ms_a, &err));
  std::uint32_t top1_inc = 0;
  assert(dec_a.CodecTop1(h5_inc, &top1_inc, &ms_a, &err));

  TalkerNpuDecoder dec_b;
  assert(dec_b.Open(model_dir, &engine, mode, {}, &err));
  double ms_b = 0.0;
  std::vector<float> h5_full;
  assert(dec_b.Prefill(rows, 5, &h5_full, &ms_b, &err));
  std::vector<float> logits_full;
  assert(dec_b.CodecLogits(h5_full, &logits_full, &ms_b, &err));
  std::uint32_t top1_full = 0;
  assert(dec_b.CodecTop1(h5_full, &top1_full, &ms_b, &err));

  const float cos = Cosine(logits_inc, logits_full);
  std::printf(
      "[talker-decode 28L] mode=%s npu_ms=%.2f+%.2f cos=%.6f top1_inc=%u "
      "top1_full=%u match=%d\n",
      NpuQuantModeName(mode), ms_a, ms_b, cos, top1_inc, top1_full,
      (top1_inc == top1_full) ? 1 : 0);
  std::fflush(stdout);
  const float min_cos =
      (mode == mbufo::core::npu::NpuQuantMode::kQ8Bfp16) ? 0.998F : 0.965F;
  assert(cos >= min_cos);
  assert(top1_inc == top1_full);
  assert(top1_inc < 3072);
}

}  // namespace

int main() {
  mbufo::core::npu::NpuQuantGemmEngine engine;
  std::string err;
  if (!engine.Init("/mnt/nvme-data/mbufo-models/npu-kernels/quant_gemm",
                   &err)) {
    std::fprintf(stderr, "NpuQuantGemmEngine::Init failed: %s\n", err.c_str());
    return 1;
  }
  TestTalkerConsistency(engine, mbufo::core::npu::NpuQuantMode::kQ8Bfp16);
  TestTalkerConsistency(engine, mbufo::core::npu::NpuQuantMode::kQ4Q4nx);
  std::puts("TALKER_DECODE_CONSISTENCY_PASS");
  return 0;
}
