// P2b: predictor NPU-only incremental-vs-full-prefill gate + determinism.
//
// Path A (servable): prefill [r0, r1] at pos 0/1, then DecodeStep(r2) at
// pos 2, head-1 logits + top-1. Path B (reference): prefill [r0, r1, r2]
// full sequence, head-1 logits + top-1. Both run ALL 5 predictor layers on
// NPU silicon. Gate: cosine >= Q8 0.998 + top-1 agreement.
// Plus: full PredictFrame runs twice with identical codes (determinism).

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/qwen3_predictor_decode.hpp"

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

void TestPredictor(mbufo::core::npu::NpuQuantGemmEngine& engine,
                   mbufo::core::npu::NpuQuantMode mode) {
  using mbufo::core::npu::NpuQuantModeName;
  using mbufo::core::npu::PredictorNpuDecoder;
  const std::string model_dir = "/mnt/nvme-data/mbufo-models/qwen3-tts-1.7b-base";
  std::string err;

  std::mt19937 rng(777);
  std::uniform_real_distribution<float> dist(-0.1F, 0.1F);
  std::vector<float> talker_hidden(2048);
  for (float& v : talker_hidden) v = dist(rng);
  constexpr std::uint32_t kFirstCode = 7;

  // Reference full frame (also seeds r2 from its own code1).
  PredictorNpuDecoder ref;
  assert(ref.Open(model_dir, &engine, mode, {}, &err));
  double ms_ref = 0.0;
  std::vector<std::uint32_t> frame;
  assert(ref.PredictFrame(talker_hidden, kFirstCode, &frame, nullptr, &ms_ref,
                          &err));
  assert(frame.size() == 16 && frame[0] == kFirstCode);
  for (std::uint32_t c : frame) assert(c < 2048);

  // Determinism: second run must match exactly.
  PredictorNpuDecoder rep;
  assert(rep.Open(model_dir, &engine, mode, {}, &err));
  std::vector<std::uint32_t> frame2;
  assert(rep.PredictFrame(talker_hidden, kFirstCode, &frame2, nullptr, nullptr,
                          &err));
  assert(frame == frame2);

  // Incremental-vs-full gate on head 1. r2 = embed table0[code1] projected.
  PredictorNpuDecoder dec_a;
  assert(dec_a.Open(model_dir, &engine, mode, {}, &err));
  double ms_a = 0.0;
  // Build seed rows via dec_a's own projection of synthetic group-0 rows.
  std::vector<float> g0rows(2 * 2048);
  for (float& v : g0rows) v = dist(rng);
  std::vector<float> rows01;
  assert(dec_a.ProjectInput(g0rows, 2, &rows01, &err));
  std::vector<float> h1;
  assert(dec_a.Prefill(rows01, 2, &h1, &ms_a, &err));
  std::vector<float> r2raw(2048);
  for (float& v : r2raw) v = dist(rng);
  std::vector<float> r2;
  assert(dec_a.ProjectInput(r2raw, 1, &r2, &err));
  std::vector<float> h2_inc;
  assert(dec_a.DecodeStep(r2, &h2_inc, &ms_a, &err));
  std::vector<float> logits_inc;
  assert(dec_a.HeadLogits(1, h2_inc, &logits_inc, &ms_a, &err));
  std::uint32_t top1_inc = 0;
  assert(dec_a.HeadTop1(1, h2_inc, &top1_inc, &ms_a, &err));

  PredictorNpuDecoder dec_b;
  assert(dec_b.Open(model_dir, &engine, mode, {}, &err));
  double ms_b = 0.0;
  std::vector<float> rows012(rows01);
  rows012.insert(rows012.end(), r2.begin(), r2.end());
  std::vector<float> h2_full;
  assert(dec_b.Prefill(rows012, 3, &h2_full, &ms_b, &err));
  std::vector<float> logits_full;
  assert(dec_b.HeadLogits(1, h2_full, &logits_full, &ms_b, &err));
  std::uint32_t top1_full = 0;
  assert(dec_b.HeadTop1(1, h2_full, &top1_full, &ms_b, &err));

  const float cos = Cosine(logits_inc, logits_full);
  std::printf(
      "[predictor 5L] mode=%s npu_ms=%.2f+%.2f frame_ms=%.2f cos=%.6f "
      "top1_inc=%u top1_full=%u match=%d deter=%d\n",
      NpuQuantModeName(mode), ms_a, ms_b, ms_ref, cos, top1_inc, top1_full,
      (top1_inc == top1_full) ? 1 : 0, (frame == frame2) ? 1 : 0);
  std::fflush(stdout);
  const float min_cos =
      (mode == mbufo::core::npu::NpuQuantMode::kQ8Bfp16) ? 0.998F : 0.965F;
  assert(cos >= min_cos);
  assert(top1_inc == top1_full);
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
  TestPredictor(engine, mbufo::core::npu::NpuQuantMode::kQ8Bfp16);
  std::puts("PREDICTOR_DECODE_CONSISTENCY_PASS");
  return 0;
}
