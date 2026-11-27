// P2c: TTS NPU frame-loop tests (host policy + silicon determinism).
//
// 1. SelectFirstCodeNpu masking: EOS wins at step>=2, is masked at step<2,
//    nonfinite logits are rejected.
// 2. Generate determinism on silicon: two identical runs emit identical
//    codec frames, all ids in range.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/qwen3_tts_frame.hpp"

namespace {

void TestSamplerPolicy() {
  using mbufo::core::npu::SelectFirstCodeNpu;
  using mbufo::core::npu::TtsSampling;
  using mbufo::core::npu::TtsTokenScore;
  TtsSampling s;
  s.sample = false;
  std::vector<float> logits(3072, -1.0e30F);
  logits[100] = 5.0F;
  logits[2150] = 10.0F;  // EOS
  std::vector<std::uint8_t> gen(3072, 0);
  std::vector<TtsTokenScore> scratch;
  // Step 0: EOS masked despite higher score.
  assert(SelectFirstCodeNpu(logits, gen, 0, s, nullptr, &scratch) == 100);
  // Step 5: EOS eligible and wins.
  assert(SelectFirstCodeNpu(logits, gen, 5, s, nullptr, &scratch) == 2150);
  // Nonfinite rejected: +inf at 300 must not win over finite 100.
  logits[300] = std::numeric_limits<float>::infinity();
  assert(SelectFirstCodeNpu(logits, gen, 0, s, nullptr, &scratch) == 100);
  // Repetition penalty: generated 100 penalized, 101 (equal base) wins.
  logits[101] = 5.0F;
  gen[100] = 1;
  s.repetition_penalty = 1.05F;
  assert(SelectFirstCodeNpu(logits, gen, 0, s, nullptr, &scratch) == 101);
  std::puts("[tts-frame] sampler policy OK");
}

void TestFrameLoop(mbufo::core::npu::NpuQuantGemmEngine& engine,
                   mbufo::core::npu::NpuQuantMode mode) {
  using mbufo::core::npu::NpuQuantModeName;
  using mbufo::core::npu::PredictorNpuDecoder;
  using mbufo::core::npu::TalkerNpuDecoder;
  using mbufo::core::npu::TtsNpuFrameLoop;
  using mbufo::core::npu::TtsSampling;
  const std::string model_dir = "/mnt/nvme-data/mbufo-models/qwen3-tts-1.7b-base";
  std::string err;

  auto run = [&](std::vector<std::uint32_t>* codes, std::size_t* frames,
                 bool* eos, double* ms) {
    TalkerNpuDecoder talker;
    assert(talker.Open(model_dir, &engine, mode, {}, &err));
    PredictorNpuDecoder predictor;
    assert(predictor.Open(model_dir, &engine, mode, {}, &err));
    TtsNpuFrameLoop loop;
    loop.Bind(&talker, &predictor);
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-0.1F, 0.1F);
    std::vector<float> prompt(4 * 2048);
    for (float& v : prompt) v = dist(rng);
    std::vector<float> pad(2048, 0.0F);
    TtsSampling sampling;
    sampling.sample = false;  // greedy: deterministic
    return loop.Generate(prompt, 4, {}, 0, pad, 2, sampling, codes, frames,
                         eos, ms, &err);
  };

  std::vector<std::uint32_t> codes_a, codes_b;
  std::size_t frames_a = 0, frames_b = 0;
  bool eos_a = false, eos_b = false;
  double ms_a = 0.0, ms_b = 0.0;
  assert(run(&codes_a, &frames_a, &eos_a, &ms_a));
  assert(run(&codes_b, &frames_b, &eos_b, &ms_b));
  assert(codes_a == codes_b);
  assert(frames_a == codes_a.size() / 16);
  assert(frames_a <= 2);
  for (std::size_t f = 0; f < frames_a; ++f) {
    assert(codes_a[f * 16] < 3072);
    for (std::size_t g = 1; g < 16; ++g) assert(codes_a[f * 16 + g] < 2048);
  }
  std::printf("[tts-frame] mode=%s frames=%zu eos=%d npu_ms=%.2f deter=1\n",
              NpuQuantModeName(mode), frames_a, eos_a ? 1 : 0, ms_a);
  std::fflush(stdout);
}

}  // namespace

int main() {
  TestSamplerPolicy();
  mbufo::core::npu::NpuQuantGemmEngine engine;
  std::string err;
  if (!engine.Init("/mnt/nvme-data/mbufo-models/npu-kernels/quant_gemm",
                   &err)) {
    std::fprintf(stderr, "NpuQuantGemmEngine::Init failed: %s\n", err.c_str());
    return 1;
  }
  TestFrameLoop(engine, mbufo::core::npu::NpuQuantMode::kQ8Bfp16);
  std::puts("TTS_FRAME_LOOP_PASS");
  return 0;
}
