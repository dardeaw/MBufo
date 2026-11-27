// P3: shared Qwen3-4B TE NPU gate (weights identical for FLUX.2-klein and
// Z-Image-Turbo, verified by hash).
//
// Path A (incremental math check): Prefill(T=4) then DecodeStep(5th row).
// Path B (reference): Prefill(T=5). Both run ALL 36 TE layers on NPU.
// Gate: cosine(A, B) >= Q8 0.998.
// Plus: Encode() determinism + shape/finite check on the real TE dir.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/npu_shard_reader.hpp"
#include "src/core/compute/npu/qwen3_te_decode.hpp"

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

void TestTe(mbufo::core::npu::NpuQuantGemmEngine& engine,
            mbufo::core::npu::NpuQuantMode mode) {
  using mbufo::core::npu::NpuQuantModeName;
  using mbufo::core::npu::NpuShardReader;
  using mbufo::core::npu::TeNpuDecoder;
  const std::string model_dir =
      "/mnt/nvme-data/mbufo-models/flux2-klein-4b/text_encoder";
  constexpr std::size_t kHidden = 2560;
  const std::vector<std::uint32_t> ids = {1, 2, 3, 4, 5};
  std::string err;

  NpuShardReader reader;
  assert(NpuShardReader::OpenDir(model_dir, &reader, &err));
  std::vector<float> rows;
  assert(reader.GatherRowsF32("model.embed_tokens.weight", ids, kHidden,
                              &rows, &err));

  TeNpuDecoder dec_a;
  assert(dec_a.Open(model_dir, &engine, mode, {}, &err));
  double ms_a = 0.0;
  std::vector<float> h4;
  assert(dec_a.Prefill(rows, 4, &h4, &ms_a, &err));
  std::vector<float> row5(rows.begin() + 4 * kHidden, rows.end());
  std::vector<float> h5_inc;
  assert(dec_a.DecodeStep(row5, &h5_inc, &ms_a, &err));

  TeNpuDecoder dec_b;
  assert(dec_b.Open(model_dir, &engine, mode, {}, &err));
  double ms_b = 0.0;
  std::vector<float> h5_full;
  assert(dec_b.Prefill(rows, 5, &h5_full, &ms_b, &err));

  const float cos = Cosine(h5_inc, h5_full);
  std::printf("[te-decode 36L] mode=%s npu_ms=%.2f+%.2f cos=%.6f\n",
              NpuQuantModeName(mode), ms_a, ms_b, cos);
  std::fflush(stdout);
  const float min_cos =
      (mode == mbufo::core::npu::NpuQuantMode::kQ8Bfp16) ? 0.998F : 0.965F;
  assert(cos >= min_cos);

  // Encode path: full sequence, deterministic, finite, nonzero.
  TeNpuDecoder enc1, enc2;
  assert(enc1.Open(model_dir, &engine, mode, {}, &err));
  assert(enc2.Open(model_dir, &engine, mode, {}, &err));
  std::vector<float> seq1, seq2;
  assert(enc1.Encode(rows, 5, &seq1, &ms_b, &err));
  assert(enc2.Encode(rows, 5, &seq2, nullptr, &err));
  assert(seq1.size() == 5 * kHidden && seq1 == seq2);
  double energy = 0.0;
  for (float v : seq1) {
    assert(std::isfinite(v));
    energy += static_cast<double>(v) * v;
  }
  assert(energy > 1.0);
  std::printf("[te-encode 36L] mode=%s seq=%zu deter=1 energy=%.1f\n",
              NpuQuantModeName(mode), seq1.size() / kHidden, energy);
  std::fflush(stdout);
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
  TestTe(engine, mbufo::core::npu::NpuQuantMode::kQ8Bfp16);
  std::puts("TE_DECODE_CONSISTENCY_PASS");
  return 0;
}
