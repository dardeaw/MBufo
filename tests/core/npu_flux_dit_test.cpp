// P4: FLUX.2-klein DiT NPU gates.
//
// Gate 1 (plumbing): one double block + one single block on tiny inputs,
// NPU path vs an independent host-FP32 reimplementation sharing only the
// norm/rope/activation helpers. Cosine >= 0.998 (Q8).
// Gate 2 (E2E determinism): full 5-double + 20-single x 4-step Euler run on
// the shell's 9-token prompt + 8x8 latent grid. Two identical runs must
// match exactly; final latent finite + nonzero. Q8-vs-Q4 final cosine >= 0.99.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/dit_npu_common.hpp"
#include "src/core/compute/npu/flux_dit_npu.hpp"
#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/npu_shard_reader.hpp"
#include "src/core/compute/npu/qwen3_te_decode.hpp"

namespace {

using mbufo::core::npu::NpuShardReader;
using mbufo::core::npu::dit::CosineSimHost;
using mbufo::core::npu::dit::FluxDitNpu;
using mbufo::core::npu::dit::MatMulRefHost;
using mbufo::core::npu::dit::RmsNormRows;
using mbufo::core::npu::dit::RopeAttn;
using mbufo::core::npu::dit::SiluInPlace;

// Independent host-FP32 double block 0 (mirrors FluxDitNpu::DoubleBlock).
void RefDoubleBlock0(NpuShardReader* st, std::vector<float>* img,
                     std::vector<float>* txt, std::size_t Si, std::size_t St,
                     std::span<const float> mod_img,
                     std::span<const float> mod_txt) {
  constexpr std::size_t D = 3072;
  std::string err;
  std::vector<float> w, o;
  auto lin = [&](const std::string& n, std::span<const float> x, std::size_t M,
                 std::size_t N, std::size_t K, std::vector<float>* y) {
    assert(st->ReadF32(n, N * K, &w, &err));
    MatMulRefHost(x.data(), M, w.data(), N, K, nullptr, y);
  };
  std::vector<float> nq, nk, naq, nak;
  assert(st->ReadF32("transformer_blocks.0.attn.norm_q.weight", 128, &nq, &err));
  assert(st->ReadF32("transformer_blocks.0.attn.norm_k.weight", 128, &nk, &err));
  assert(st->ReadF32("transformer_blocks.0.attn.norm_added_q.weight", 128, &naq, &err));
  assert(st->ReadF32("transformer_blocks.0.attn.norm_added_k.weight", 128, &nak, &err));
  std::vector<float> in_img, in_txt;
  RmsNormRows(img->data(), nullptr, Si, D, 1e-6F, &in_img);
  RmsNormRows(txt->data(), nullptr, St, D, 1e-6F, &in_txt);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      in_img[t * D + c] = in_img[t * D + c] * (1.0F + mod_img[D + c]) + mod_img[c];
  for (std::size_t t = 0; t < St; ++t)
    for (std::size_t c = 0; c < D; ++c)
      in_txt[t * D + c] = in_txt[t * D + c] * (1.0F + mod_txt[D + c]) + mod_txt[c];
  std::vector<float> qi, ki, vi, qt, kt, vt;
  lin("transformer_blocks.0.attn.to_q.weight", in_img, Si, D, D, &qi);
  lin("transformer_blocks.0.attn.to_k.weight", in_img, Si, D, D, &ki);
  lin("transformer_blocks.0.attn.to_v.weight", in_img, Si, D, D, &vi);
  lin("transformer_blocks.0.attn.add_q_proj.weight", in_txt, St, D, D, &qt);
  lin("transformer_blocks.0.attn.add_k_proj.weight", in_txt, St, D, D, &kt);
  lin("transformer_blocks.0.attn.add_v_proj.weight", in_txt, St, D, D, &vt);
  std::vector<float> ctx_i, ctx_t;
  RopeAttn(qi, ki, vi, nq, nk, Si, 24, 24, 128, 2000.0F, 1e-6F, false, &ctx_i);
  RopeAttn(qt, kt, vt, naq, nak, St, 24, 24, 128, 2000.0F, 1e-6F, false, &ctx_t);
  std::vector<float> oi, ot;
  lin("transformer_blocks.0.attn.to_out.0.weight", ctx_i, Si, D, D, &oi);
  lin("transformer_blocks.0.attn.to_add_out.weight", ctx_t, St, D, D, &ot);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*img)[t * D + c] += std::tanh(mod_img[2 * D + c]) * oi[t * D + c];
  for (std::size_t t = 0; t < St; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*txt)[t * D + c] += std::tanh(mod_txt[2 * D + c]) * ot[t * D + c];
  std::vector<float> ffi, ffi_in;
  RmsNormRows(img->data(), nullptr, Si, D, 1e-6F, &ffi);
  lin("transformer_blocks.0.ff.linear_in.weight", ffi, Si, 6 * D, D, &ffi_in);
  std::vector<float> act(Si * 3 * D), out;
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < 3 * D; ++c)
      act[t * 3 * D + c] =
          (ffi_in[t * 6 * D + c] / (1.0F + std::exp(-ffi_in[t * 6 * D + c]))) *
          ffi_in[t * 6 * D + 3 * D + c];
  lin("transformer_blocks.0.ff.linear_out.weight", act, Si, D, 3 * D, &out);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*img)[t * D + c] += std::tanh(mod_img[5 * D + c]) * out[t * D + c];
}

// Independent host-FP32 single block 0.
void RefSingleBlock0(NpuShardReader* st, std::vector<float>* img,
                     std::size_t Si, std::span<const float> mod_sgl) {
  constexpr std::size_t D = 3072;
  std::string err;
  std::vector<float> w;
  std::vector<float> nq, nk;
  assert(st->ReadF32("single_transformer_blocks.0.attn.norm_q.weight", 128, &nq, &err));
  assert(st->ReadF32("single_transformer_blocks.0.attn.norm_k.weight", 128, &nk, &err));
  std::vector<float> s_in, qkv;
  RmsNormRows(img->data(), nullptr, Si, D, 1e-6F, &s_in);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      s_in[t * D + c] = s_in[t * D + c] * (1.0F + mod_sgl[D + c]) + mod_sgl[c];
  assert(st->ReadF32("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
                     9 * D * D, &w, &err));
  MatMulRefHost(s_in.data(), Si, w.data(), 9 * D, D, nullptr, &qkv);
  std::vector<float> q(Si * D), k(Si * D), v(Si * D);
  for (std::size_t t = 0; t < Si; ++t) {
    const float* r = qkv.data() + t * 9 * D;
    std::copy(r, r + D, q.data() + t * D);
    std::copy(r + D, r + 2 * D, k.data() + t * D);
    std::copy(r + 2 * D, r + 3 * D, v.data() + t * D);
  }
  std::vector<float> ctx;
  RopeAttn(q, k, v, nq, nk, Si, 24, 24, 128, 2000.0F, 1e-6F, false, &ctx);
  std::vector<float> cat(Si * 4 * D), sout;
  for (std::size_t t = 0; t < Si; ++t) {
    const float* r = qkv.data() + t * 9 * D;
    std::copy(ctx.data() + t * D, ctx.data() + (t + 1) * D,
              cat.data() + t * 4 * D);
    for (std::size_t c = 0; c < 3 * D; ++c)
      cat[t * 4 * D + D + c] =
          (r[3 * D + c] / (1.0F + std::exp(-r[3 * D + c]))) * r[6 * D + c];
  }
  assert(st->ReadF32("single_transformer_blocks.0.attn.to_out.weight",
                     4 * D * D, &w, &err));
  MatMulRefHost(cat.data(), Si, w.data(), D, 4 * D, nullptr, &sout);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*img)[t * D + c] += std::tanh(mod_sgl[2 * D + c]) * sout[t * D + c];
}

void TestPlumbing(mbufo::core::npu::NpuQuantGemmEngine& engine) {
  using mbufo::core::npu::NpuQuantMode;
  const std::string dir = "/mnt/nvme-data/mbufo-models/flux2-klein-4b/transformer";
  std::string err;
  FluxDitNpu dit;
  assert(dit.Open(dir, &engine, NpuQuantMode::kQ8Bfp16, {}, &err));
  NpuShardReader st;
  assert(NpuShardReader::OpenDir(dir, &st, &err));

  std::mt19937 rng(21);
  std::uniform_real_distribution<float> dist(-0.1F, 0.1F);
  constexpr std::size_t Si = 4, St = 3, D = 3072;
  std::vector<float> img(Si * D), txt(St * D);
  for (float& x : img) x = dist(rng);
  for (float& x : txt) x = dist(rng);
  std::vector<float> img_n = img, txt_n = txt;

  std::vector<float> temb;
  double ms = 0.0;
  assert(dit.TimestepEmbed(1.0F, &temb, &ms, &err));
  mbufo::core::npu::dit::FluxModulations mod;
  assert(dit.Modulations(temb, &mod, &ms, &err));

  assert(dit.DenoiseBlocks(&img_n, &txt_n, Si, St, mod, 1, 1, &ms, &err));
  RefDoubleBlock0(&st, &img, &txt, Si, St, mod.img, mod.txt);
  RefSingleBlock0(&st, &img, Si, mod.sgl);

  const float ci = CosineSimHost(img_n, img);
  const float ct = CosineSimHost(txt_n, txt);
  std::printf("[flux-plumbing dbl+sgl] npu_ms=%.2f cos_img=%.6f cos_txt=%.6f\n",
              ms, ci, ct);
  std::fflush(stdout);
  assert(ci >= 0.998F && ct >= 0.998F);
}

std::vector<float> RunFull(mbufo::core::npu::NpuQuantGemmEngine& engine,
                           mbufo::core::npu::NpuQuantMode mode, double* ms) {
  using mbufo::core::npu::NpuShardReader;
  using mbufo::core::npu::TeNpuDecoder;
  std::string err;
  // TE taps (shell prompt ids, shell tap layers 0/18/final).
  TeNpuDecoder te;
  assert(te.Open("/mnt/nvme-data/mbufo-models/flux2-klein-4b/text_encoder",
                 &engine, mode, {}, &err));
  NpuShardReader tr;
  assert(NpuShardReader::OpenDir(
      "/mnt/nvme-data/mbufo-models/flux2-klein-4b/text_encoder", &tr, &err));
  const std::vector<std::uint32_t> ids = {151644, 872, 198,  32, 6977,
                                          17822,  4128, 304, 151645};
  std::vector<float> erows;
  assert(tr.GatherRowsF32("model.embed_tokens.weight", ids, 2560, &erows,
                          &err));
  const std::vector<std::size_t> tap_layers = {0, 18};
  std::vector<std::vector<float>> taps;
  std::vector<float> final_raw;
  assert(te.EncodeTaps(erows, ids.size(), tap_layers, &taps, &final_raw, ms,
                       &err));
  assert(taps.size() == 2);
  // cond7680 = [tap0, tap1, final].
  std::vector<float> cond(ids.size() * 7680);
  for (std::size_t t = 0; t < ids.size(); ++t) {
    std::copy(taps[0].data() + t * 2560, taps[0].data() + (t + 1) * 2560,
              cond.data() + t * 7680);
    std::copy(taps[1].data() + t * 2560, taps[1].data() + (t + 1) * 2560,
              cond.data() + t * 7680 + 2560);
    std::copy(final_raw.data() + t * 2560, final_raw.data() + (t + 1) * 2560,
              cond.data() + t * 7680 + 5120);
  }
  FluxDitNpu dit;
  assert(dit.Open("/mnt/nvme-data/mbufo-models/flux2-klein-4b/transformer",
                  &engine, mode, {}, &err));
  std::vector<float> txt;
  assert(dit.EncodeContext(cond, ids.size(), &txt, ms, &err));

  constexpr std::size_t Si = 64;
  std::vector<float> z(Si * 128);
  for (std::size_t i = 0; i < Si * 128; ++i)
    z[i] = 0.25F * std::cos(0.037F * static_cast<float>(i + 1u));
  const float sigmas[5] = {1.0F, 0.75F, 0.50F, 0.25F, 0.0F};
  for (int step = 0; step < 4; ++step) {
    if (!dit.FullStep(&z, Si, txt, ids.size(), sigmas[step],
                      sigmas[step] - sigmas[step + 1], 5, 20, ms, &err)) {
      std::fprintf(stderr, "FullStep %d failed: %s\n", step, err.c_str());
      assert(false);
    }
  }
  return z;
}

void TestFull(mbufo::core::npu::NpuQuantGemmEngine& engine) {
  using mbufo::core::npu::NpuQuantMode;
  double ms8 = 0.0, ms8b = 0.0, ms4 = 0.0;
  std::vector<float> z8 = RunFull(engine, NpuQuantMode::kQ8Bfp16, &ms8);
  std::vector<float> z8b = RunFull(engine, NpuQuantMode::kQ8Bfp16, &ms8b);
  assert(z8 == z8b);
  double energy = 0.0;
  for (float v : z8) {
    assert(std::isfinite(v));
    energy += (double)v * v;
  }
  assert(energy > 1.0);
  std::vector<float> z4 = RunFull(engine, NpuQuantMode::kQ4Q4nx, &ms4);
  const float c84 = CosineSimHost(z8, z4);
  std::printf(
      "[flux-full 5dbl+20sgl x4] q8_ms=%.1f deter=1 energy=%.1f q8q4_cos=%.6f "
      "q4_ms=%.1f\n",
      ms8, energy, c84, ms4);
  std::fflush(stdout);
  assert(c84 >= 0.99F);
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
  TestFull(engine);
  std::puts("FLUX_DIT_PASS");
  return 0;
}
