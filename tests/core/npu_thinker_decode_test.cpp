// P1c: thinker NPU-only prefill + 1-step KV decode consistency gate.
//
// Path A (servable): Prefill(T=4) then DecodeStep(5th row) through the KV
// cache. Path B (reference): Prefill(T=5) full sequence. Both run ALL 28
// thinker layers on NPU silicon via ThinkerNpuDecoder (no FP32 dual path).
// Gate: cosine(A, B) >= Q8 0.998 + top-1 agreement.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/npu_shard_reader.hpp"
#include "src/core/compute/npu/npu_thinker_decode.hpp"

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

void TestDecodeConsistency(mbufo::core::npu::NpuQuantGemmEngine& engine,
                           mbufo::core::npu::NpuQuantMode mode) {
  using mbufo::core::npu::NpuQuantModeName;
  using mbufo::core::npu::NpuShardReader;
  using mbufo::core::npu::ThinkerNpuDecoder;
  const std::string model_dir = "/mnt/nvme-data/mbufo-models/qwen3-asr-1.7b";
  const std::vector<std::uint32_t> ids = {1, 2, 3, 4, 5};
  std::string err;

  NpuShardReader reader;
  assert(NpuShardReader::OpenDir(model_dir, &reader, &err));
  std::vector<float> rows;
  assert(reader.GatherRowsF32("thinker.model.embed_tokens.weight", ids, 2048,
                              &rows, &err));

  // Path A: prefill 4 + decode 1.
  ThinkerNpuDecoder dec_a;
  assert(dec_a.Open(model_dir, &engine, mode, {}, &err));
  double ms_a = 0.0;
  std::vector<float> h4;
  assert(dec_a.Prefill(rows, 4, &h4, &ms_a, &err));
  assert(dec_a.pos() == 4);
  std::vector<float> row5(rows.begin() + 4 * 2048, rows.end());
  std::vector<float> h5_inc;
  assert(dec_a.DecodeStep(row5, &h5_inc, &ms_a, &err));
  assert(dec_a.pos() == 5);
  std::uint32_t top1_inc = 0;
  assert(dec_a.LogitsTop1(h5_inc, &top1_inc, &ms_a, &err));

  // Path B: full-sequence prefill 5.
  ThinkerNpuDecoder dec_b;
  assert(dec_b.Open(model_dir, &engine, mode, {}, &err));
  double ms_b = 0.0;
  std::vector<float> h5_full;
  assert(dec_b.Prefill(rows, 5, &h5_full, &ms_b, &err));
  std::uint32_t top1_full = 0;
  assert(dec_b.LogitsTop1(h5_full, &top1_full, &ms_b, &err));

  const float cos = Cosine(h5_inc, h5_full);
  std::printf(
      "[thinker-decode 28L] mode=%s npu_ms=%.2f+%.2f cos=%.6f top1_inc=%u "
      "top1_full=%u match=%d\n",
      NpuQuantModeName(mode), ms_a, ms_b, cos, top1_inc, top1_full,
      (top1_inc == top1_full) ? 1 : 0);
  std::fflush(stdout);
  const float min_cos =
      (mode == mbufo::core::npu::NpuQuantMode::kQ8Bfp16) ? 0.998F : 0.965F;
  assert(cos >= min_cos);
  assert(top1_inc == top1_full);
  assert(top1_inc < 2048);
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
  TestDecodeConsistency(engine, mbufo::core::npu::NpuQuantMode::kQ8Bfp16);
  std::puts("THINKER_DECODE_CONSISTENCY_PASS");
  return 0;
}
