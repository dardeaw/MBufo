// P5: Z-Image-Turbo DiT NPU gates.
//
// Gate 1 (plumbing): one context_refiner block on tiny inputs, NPU path vs
// an independent host-FP32 reimplementation sharing only the norm/rope
// helpers. Cosine >= 0.998 (Q8).
// Gate 2 (E2E determinism): full 2-refiner + 30-main x 8-step Euler run on
// the shell's 9-token prompt + 8x8 latent grid. Two identical runs must
// match exactly; final latent finite + nonzero. Q8 only (Q4 is banned for Z-Image: q8-vs-q4 cos 0.975).

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/dit_npu_common.hpp"
#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/npu_shard_reader.hpp"
#include "src/core/compute/npu/qwen3_te_decode.hpp"
#include "src/core/compute/npu/zimage_dit_npu.hpp"

namespace {

using mbufo::core::npu::NpuShardReader;
using mbufo::core::npu::dit::CosineSimHost;
using mbufo::core::npu::dit::FluxRopeTables;
using mbufo::core::npu::dit::JointAttnTables;
using mbufo::core::npu::dit::MatMulRefHost;
using mbufo::core::npu::dit::RmsNormRows;
using mbufo::core::npu::dit::ZImageDitNpu;

// Independent host-FP32 context_refiner.0 (mirrors ZImageDitNpu::Block).
void RefRefinerBlock0(NpuShardReader* st, std::vector<float>* h,
                      std::size_t seq, std::span<const float> rope_cos,
                      std::span<const float> rope_sin) {
  constexpr std::size_t D = 3840, F = 10240;
  std::string err;
  std::vector<float> w;
  auto rd = [&](const std::string& n, std::size_t n_, std::vector<float>* y) {
    assert(st->ReadF32(n, n_, y, &err));
  };
  std::vector<float> an1, an2, fn1, fn2, nq, nk;
  rd("context_refiner.0.attention_norm1.weight", D, &an1);
  rd("context_refiner.0.attention_norm2.weight", D, &an2);
  rd("context_refiner.0.ffn_norm1.weight", D, &fn1);
  rd("context_refiner.0.ffn_norm2.weight", D, &fn2);
  rd("context_refiner.0.attention.norm_q.weight", 128, &nq);
  rd("context_refiner.0.attention.norm_k.weight", 128, &nk);
  std::vector<float> x, q, k, v;
  RmsNormRows(h->data(), an1.data(), seq, D, 1e-5F, &x);
  rd("context_refiner.0.attention.to_q.weight", D * D, &w);
  MatMulRefHost(x.data(), seq, w.data(), D, D, nullptr, &q);
  rd("context_refiner.0.attention.to_k.weight", D * D, &w);
  MatMulRefHost(x.data(), seq, w.data(), D, D, nullptr, &k);
  rd("context_refiner.0.attention.to_v.weight", D * D, &w);
  MatMulRefHost(x.data(), seq, w.data(), D, D, nullptr, &v);
  std::vector<float> ctx, o, on;
  std::vector<float> qn(seq * D), kn(seq * D);
  for (std::size_t t = 0; t < seq; ++t)
    for (std::size_t hh = 0; hh < 30; ++hh) {
      std::copy(nq.begin(), nq.end(), qn.begin() + t * D + hh * 128);
      std::copy(nk.begin(), nk.end(), kn.begin() + t * D + hh * 128);
    }
  JointAttnTables(q, k, v, qn, kn, seq, seq, 30, 30, 128, rope_cos, rope_sin,
                  rope_cos, rope_sin, 1e-5F, &ctx);
  rd("context_refiner.0.attention.to_out.0.weight", D * D, &w);
  MatMulRefHost(ctx.data(), seq, w.data(), D, D, nullptr, &o);
  RmsNormRows(o.data(), an2.data(), seq, D, 1e-5F, &on);
  for (std::size_t i = 0; i < seq * D; ++i)
    (*h)[i] += on[i];
  RmsNormRows(h->data(), fn1.data(), seq, D, 1e-5F, &x);
  std::vector<float> w1, w3;
  rd("context_refiner.0.feed_forward.w1.weight", F * D, &w);
  MatMulRefHost(x.data(), seq, w.data(), F, D, nullptr, &w1);
  rd("context_refiner.0.feed_forward.w3.weight", F * D, &w);
  MatMulRefHost(x.data(), seq, w.data(), F, D, nullptr, &w3);
  for (std::size_t i = 0; i < seq * F; ++i)
    w1[i] = (w1[i] / (1.0F + std::exp(-w1[i]))) * w3[i];
  std::vector<float> w2, fn;
  rd("context_refiner.0.feed_forward.w2.weight", D * F, &w);
  MatMulRefHost(w1.data(), seq, w.data(), D, F, nullptr, &w2);
  RmsNormRows(w2.data(), fn2.data(), seq, D, 1e-5F, &fn);
  for (std::size_t i = 0; i < seq * D; ++i)
    (*h)[i] += fn[i];
}

void TestPlumbing(mbufo::core::npu::NpuQuantGemmEngine& engine) {
  using mbufo::core::npu::NpuQuantMode;
  const std::string dir =
      "/mnt/nvme-data/mbufo-models/z-image-turbo/transformer";
  std::string err;
  ZImageDitNpu dit;
  assert(dit.Open(dir, &engine, NpuQuantMode::kQ8Bfp16, {}, &err));

  std::mt19937 rng(33);
  std::uniform_real_distribution<float> dist(-0.1F, 0.1F);
  constexpr std::size_t S = 4, D = 3840;
  std::vector<float> h(S * D);
  for (float& x : h)
    x = dist(rng);
  std::vector<float> h_n = h;
  double ms = 0.0;
  // Shared tables (plumbing gate only checks NPU-vs-host consistency).
  std::vector<std::int32_t> pids(S * 4u, 0);
  for (std::size_t t = 0; t < S; ++t)
    pids[t * 4u] = static_cast<std::int32_t>(t + 1u);
  const std::size_t zaxes[3] = {32, 48, 48};
  std::vector<float> rcos, rsin;
  FluxRopeTables(pids, S, zaxes, 3u, 256.0, &rcos, &rsin);
  assert(
      dit.Block("context_refiner.0", &h_n, S, nullptr, rcos, rsin, &ms, &err));
  NpuShardReader st;
  assert(NpuShardReader::OpenDir(dir, &st, &err));
  RefRefinerBlock0(&st, &h, S, rcos, rsin);
  const float c = CosineSimHost(h_n, h);
  std::printf("[zimg-plumbing refiner] npu_ms=%.2f cos=%.6f\n", ms, c);
  std::fflush(stdout);
  assert(c >= 0.998F);
}

std::vector<float> ReadF32File(const char* path, std::size_t expect,
                               std::string* err) {
  std::vector<float> out;
  if (path == nullptr)
    return out;
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.good())
    return out;
  ifs.seekg(0, std::ios::end);
  const std::size_t bytes = static_cast<std::size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  if (bytes != expect * sizeof(float))
    return out;
  out.resize(expect);
  ifs.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(bytes));
  if (!ifs.good())
    out.clear();
  return out;
}

void WriteF32File(const char* path, std::span<const float> v) {
  std::ofstream ofs(path, std::ios::binary);
  assert(ofs.good());
  ofs.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
  assert(ofs.good());
}

std::vector<float> RunFull(mbufo::core::npu::NpuQuantGemmEngine& engine,
                           mbufo::core::npu::NpuQuantMode mode, double* ms,
                           const std::vector<std::uint32_t>& ids,
                           std::span<const float> z0,
                           std::span<const float> sigmas, std::size_t n_ref,
                           std::size_t n_main, std::vector<float>* tap34_out) {
  using mbufo::core::npu::NpuShardReader;
  using mbufo::core::npu::TeNpuDecoder;
  std::string err;
  TeNpuDecoder te;
  assert(te.Open("/mnt/nvme-data/mbufo-models/z-image-turbo/text_encoder",
                 &engine, mode, {}, &err));
  NpuShardReader tr;
  assert(NpuShardReader::OpenDir(
      "/mnt/nvme-data/mbufo-models/z-image-turbo/text_encoder", &tr, &err));
  std::vector<float> erows;
  assert(
      tr.GatherRowsF32("model.embed_tokens.weight", ids, 2560, &erows, &err));
  // Pipeline uses hidden_states[-2] == post-layer 34 (36 layers 0..35).
  const std::vector<std::size_t> tap34 = {34};
  std::vector<std::vector<float>> taps;
  std::vector<float> te_final;
  assert(te.EncodeTaps(erows, ids.size(), tap34, &taps, &te_final, ms, &err));
  assert(taps.size() == 1);
  if (tap34_out != nullptr)
    *tap34_out = taps[0];

  ZImageDitNpu dit;
  assert(dit.Open("/mnt/nvme-data/mbufo-models/z-image-turbo/transformer",
                  &engine, mode, {}, &err));
  std::vector<float> cap;
  assert(dit.CapEmbed(taps[0], ids.size(), &cap, ms, &err));
  // Context refinement happens inside FullStep (on padded cap).

  constexpr std::size_t Si = 64;
  std::vector<float> z(Si * 64);
  if (!z0.empty()) {
    assert(z0.size() == z.size());
    std::copy(z0.begin(), z0.end(), z.begin());
  } else {
    for (std::size_t i = 0; i < Si * 64; ++i)
      z[i] = 0.25F * std::sin(0.041F * static_cast<float>(i + 3u));
  }
  const std::size_t steps = sigmas.size() - 1;
  for (std::size_t step = 0; step < steps; ++step) {
    const float dt = sigmas[step] - sigmas[step + 1];
    if (!dit.FullStep(&z, Si, cap, ids.size(), sigmas[step], dt, n_ref, n_main,
                      ms, &err)) {
      std::fprintf(stderr, "ZFullStep %zu failed: %s\n", step, err.c_str());
      assert(false);
    }
  }
  return z;
}

void TestFull(mbufo::core::npu::NpuQuantGemmEngine& engine,
              const std::vector<std::uint32_t>& ids, const char* dump_prefix) {
  using mbufo::core::npu::NpuQuantMode;
  std::vector<float> sigmas;
  for (int step = 0; step <= 8; ++step) {
    sigmas.push_back(1.0F - 0.125F * static_cast<float>(step));
  }
  std::vector<float> z0;
  if (const char* zp = std::getenv("NPU_LOAD_Z0")) {
    std::string err;
    z0 = ReadF32File(zp, 64 * 64, &err);
    assert(!z0.empty());
  }
  if (const char* sp = std::getenv("NPU_SIGMAS")) {
    std::ifstream ifs(sp);
    assert(ifs.good());
    sigmas.clear();
    float v = 0;
    while (ifs >> v)
      sigmas.push_back(v);
    assert(sigmas.size() >= 2);
  }
  double ms8 = 0.0, ms8b = 0.0;
  std::vector<float> tap34;
  std::vector<float> z8 =
      RunFull(engine, NpuQuantMode::kQ8Bfp16, &ms8, ids, z0, sigmas, 2, 30,
              dump_prefix != nullptr ? &tap34 : nullptr);
  std::vector<float> z8b = RunFull(engine, NpuQuantMode::kQ8Bfp16, &ms8b, ids,
                                   z0, sigmas, 2, 30, nullptr);
  assert(z8 == z8b);
  double energy = 0.0;
  for (float v : z8) {
    assert(std::isfinite(v));
    energy += (double)v * v;
  }
  assert(energy > 1.0);
  std::printf("[zimg-full 2ref+30main x8] q8_ms=%.1f deter=1 energy=%.1f\n",
              ms8, energy);
  std::fflush(stdout);
  if (dump_prefix != nullptr) {
    const std::string pre = dump_prefix;
    WriteF32File((pre + "_z.f32").c_str(), z8);
    WriteF32File((pre + "_tap34.f32").c_str(), tap34);
    std::printf("[zimg-dump] wrote z(%zu) + tap34(%zu) to %s_*\n", z8.size(),
                tap34.size(), dump_prefix);
    std::fflush(stdout);
  }
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
  TestPlumbing(engine);
  std::vector<std::uint32_t> ids = {151644, 872,  198, 32,    6977,
                                    17822,  4128, 304, 151645};
  if (const char* ip = std::getenv("NPU_DUMP_IDS")) {
    std::ifstream ifs(ip);
    assert(ifs.good());
    ids.clear();
    std::uint32_t v = 0;
    while (ifs >> v)
      ids.push_back(v);
    assert(!ids.empty());
    std::printf("[zimg-dump] %zu ids from %s\n", ids.size(), ip);
    std::fflush(stdout);
  }
  TestFull(engine, ids, std::getenv("NPU_DUMP_LATENT"));
  std::puts("ZIMAGE_DIT_PASS");
  return 0;
}
