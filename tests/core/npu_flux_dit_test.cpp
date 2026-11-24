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
#include <cstdlib>
#include <fstream>
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
using mbufo::core::npu::dit::JointAttnTables;
using mbufo::core::npu::dit::LayerNormRows;
using mbufo::core::npu::dit::MatMulRefHost;
using mbufo::core::npu::dit::RmsNormRows;
using mbufo::core::npu::dit::RopeAttn;
using mbufo::core::npu::dit::SiluInPlace;

// Independent host-FP32 TRUE-port double block (mirrors FluxDitNpu::DoubleBlock).
void RefDoubleBlock0(NpuShardReader* st, std::vector<float>* img,
                     std::vector<float>* txt, std::size_t Si, std::size_t St,
                     std::span<const float> mod_img,
                     std::span<const float> mod_txt,
                     std::span<const float> rope_cos,
                     std::span<const float> rope_sin) {
  constexpr std::size_t D = 3072;
  const std::size_t S = St + Si;
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
  const float *ish = mod_img.data(), *isc = mod_img.data() + D,
              *igate = mod_img.data() + 2 * D, *fsh = mod_img.data() + 3 * D,
              *fsc = mod_img.data() + 4 * D, *fgate = mod_img.data() + 5 * D;
  const float *tsh = mod_txt.data(), *tsc = mod_txt.data() + D,
              *tgate = mod_txt.data() + 2 * D, *tfsh = mod_txt.data() + 3 * D,
              *tfsc = mod_txt.data() + 4 * D, *tfgate = mod_txt.data() + 5 * D;
  std::vector<float> in_img, in_txt;
  LayerNormRows(img->data(), Si, D, 1e-6F, &in_img);
  LayerNormRows(txt->data(), St, D, 1e-6F, &in_txt);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      in_img[t * D + c] = in_img[t * D + c] * (1.0F + isc[c]) + ish[c];
  for (std::size_t t = 0; t < St; ++t)
    for (std::size_t c = 0; c < D; ++c)
      in_txt[t * D + c] = in_txt[t * D + c] * (1.0F + tsc[c]) + tsh[c];
  std::vector<float> qi, ki, vi, qt, kt, vt;
  lin("transformer_blocks.0.attn.to_q.weight", in_img, Si, D, D, &qi);
  lin("transformer_blocks.0.attn.to_k.weight", in_img, Si, D, D, &ki);
  lin("transformer_blocks.0.attn.to_v.weight", in_img, Si, D, D, &vi);
  lin("transformer_blocks.0.attn.add_q_proj.weight", in_txt, St, D, D, &qt);
  lin("transformer_blocks.0.attn.add_k_proj.weight", in_txt, St, D, D, &kt);
  lin("transformer_blocks.0.attn.add_v_proj.weight", in_txt, St, D, D, &vt);
  std::vector<float> q(S * D), k(S * D), v(S * D);
  std::copy(qt.begin(), qt.end(), q.begin());
  std::copy(qi.begin(), qi.end(), q.begin() + St * D);
  std::copy(kt.begin(), kt.end(), k.begin());
  std::copy(ki.begin(), ki.end(), k.begin() + St * D);
  std::copy(vt.begin(), vt.end(), v.begin());
  std::copy(vi.begin(), vi.end(), v.begin() + St * D);
  std::vector<float> qn(S * D), kn(S * D);
  for (std::size_t t = 0; t < St; ++t)
    for (std::size_t h = 0; h < 24; ++h) {
      std::copy(naq.begin(), naq.end(), qn.begin() + t * D + h * 128);
      std::copy(nak.begin(), nak.end(), kn.begin() + t * D + h * 128);
    }
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t h = 0; h < 24; ++h) {
      std::copy(nq.begin(), nq.end(), qn.begin() + (St + t) * D + h * 128);
      std::copy(nk.begin(), nk.end(), kn.begin() + (St + t) * D + h * 128);
    }
  std::vector<float> ctx;
  JointAttnTables(q, k, v, qn, kn, S, S, 24, 24, 128, rope_cos, rope_sin,
                  rope_cos, rope_sin, 1e-6F, &ctx);
  std::vector<float> ctx_t(ctx.begin(), ctx.begin() + St * D);
  std::vector<float> ctx_i(ctx.begin() + St * D, ctx.end());
  std::vector<float> oi, ot;
  lin("transformer_blocks.0.attn.to_out.0.weight", ctx_i, Si, D, D, &oi);
  lin("transformer_blocks.0.attn.to_add_out.weight", ctx_t, St, D, D, &ot);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*img)[t * D + c] += igate[c] * oi[t * D + c];
  for (std::size_t t = 0; t < St; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*txt)[t * D + c] += tgate[c] * ot[t * D + c];
  auto ffn = [&](const std::string& pre, std::vector<float>* h, std::size_t Sn,
                 const float* sh, const float* sc, const float* g) {
    std::vector<float> x;
    LayerNormRows(h->data(), Sn, D, 1e-6F, &x);
    for (std::size_t t = 0; t < Sn; ++t)
      for (std::size_t c = 0; c < D; ++c)
        x[t * D + c] = x[t * D + c] * (1.0F + sc[c]) + sh[c];
    std::vector<float> fi, act(Sn * 3 * D), fo;
    lin(pre + ".linear_in.weight", x, Sn, 6 * D, D, &fi);
    for (std::size_t t = 0; t < Sn; ++t)
      for (std::size_t c = 0; c < 3 * D; ++c)
        act[t * 3 * D + c] =
            (fi[t * 6 * D + c] / (1.0F + std::exp(-fi[t * 6 * D + c]))) *
            fi[t * 6 * D + 3 * D + c];
    lin(pre + ".linear_out.weight", act, Sn, D, 3 * D, &fo);
    for (std::size_t t = 0; t < Sn; ++t)
      for (std::size_t c = 0; c < D; ++c) (*h)[t * D + c] += g[c] * fo[t * D + c];
  };
  ffn("transformer_blocks.0.ff", img, Si, fsh, fsc, fgate);
  ffn("transformer_blocks.0.ff_context", txt, St, tfsh, tfsc, tfgate);
}

// Independent host-FP32 TRUE-port single block on joint [txt;img].
void RefSingleBlock0(NpuShardReader* st, std::vector<float>* joint,
                     std::size_t Sj, std::span<const float> mod_sgl,
                     std::span<const float> rope_cos,
                     std::span<const float> rope_sin) {
  constexpr std::size_t D = 3072;
  std::string err;
  std::vector<float> w;
  std::vector<float> nq, nk;
  assert(st->ReadF32("single_transformer_blocks.0.attn.norm_q.weight", 128, &nq, &err));
  assert(st->ReadF32("single_transformer_blocks.0.attn.norm_k.weight", 128, &nk, &err));
  std::vector<float> s_in, qkv;
  LayerNormRows(joint->data(), Sj, D, 1e-6F, &s_in);
  for (std::size_t t = 0; t < Sj; ++t)
    for (std::size_t c = 0; c < D; ++c)
      s_in[t * D + c] =
          s_in[t * D + c] * (1.0F + mod_sgl[D + c]) + mod_sgl[c];
  assert(st->ReadF32("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
                     9 * D * D, &w, &err));
  MatMulRefHost(s_in.data(), Sj, w.data(), 9 * D, D, nullptr, &qkv);
  std::vector<float> q(Sj * D), k(Sj * D), v(Sj * D);
  for (std::size_t t = 0; t < Sj; ++t) {
    const float* r = qkv.data() + t * 9 * D;
    std::copy(r, r + D, q.data() + t * D);
    std::copy(r + D, r + 2 * D, k.data() + t * D);
    std::copy(r + 2 * D, r + 3 * D, v.data() + t * D);
  }
  std::vector<float> qn(Sj * D), kn(Sj * D);
  for (std::size_t t = 0; t < Sj; ++t)
    for (std::size_t h = 0; h < 24; ++h) {
      std::copy(nq.begin(), nq.end(), qn.begin() + t * D + h * 128);
      std::copy(nk.begin(), nk.end(), kn.begin() + t * D + h * 128);
    }
  std::vector<float> ctx;
  JointAttnTables(q, k, v, qn, kn, Sj, Sj, 24, 24, 128, rope_cos, rope_sin,
                  rope_cos, rope_sin, 1e-6F, &ctx);
  std::vector<float> cat(Sj * 4 * D), sout;
  for (std::size_t t = 0; t < Sj; ++t) {
    const float* r = qkv.data() + t * 9 * D;
    std::copy(ctx.data() + t * D, ctx.data() + (t + 1) * D,
              cat.data() + t * 4 * D);
    // SwiGLU: gate=r[9216:18432], up=r[18432:27648].
    for (std::size_t c = 0; c < 3 * D; ++c) {
      const float g = r[3 * D + c];
      cat[t * 4 * D + D + c] = (g / (1.0F + std::exp(-g))) * r[6 * D + c];
    }
  }
  assert(st->ReadF32("single_transformer_blocks.0.attn.to_out.weight",
                     4 * D * D, &w, &err));
  MatMulRefHost(cat.data(), Sj, w.data(), D, 4 * D, nullptr, &sout);
  for (std::size_t t = 0; t < Sj; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*joint)[t * D + c] += mod_sgl[2 * D + c] * sout[t * D + c];
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
  std::vector<float> rope_cos, rope_sin;
  dit.BuildRopeTables(St, Si, &rope_cos, &rope_sin);

  assert(dit.DenoiseBlocks(&img_n, &txt_n, Si, St, mod, rope_cos, rope_sin,
                           1, 1, &ms, &err));
  RefDoubleBlock0(&st, &img, &txt, Si, St, mod.img, mod.txt, rope_cos,
                  rope_sin);
  std::vector<float> joint((St + Si) * D);
  std::copy(txt.begin(), txt.end(), joint.begin());
  std::copy(img.begin(), img.end(), joint.begin() + St * D);
  RefSingleBlock0(&st, &joint, St + Si, mod.sgl, rope_cos, rope_sin);
  std::vector<float> img_ref(joint.begin() + St * D, joint.end());
  std::vector<float> txt_ref(joint.begin(), joint.begin() + St * D);

  const float ci = CosineSimHost(img_n, img_ref);
  const float ct = CosineSimHost(txt_n, txt_ref);
  std::printf("[flux-plumbing dbl+sgl] npu_ms=%.2f cos_img=%.6f cos_txt=%.6f\n",
              ms, ci, ct);
  std::fflush(stdout);
  assert(ci >= 0.997F && ct >= 0.997F);
}

std::vector<float> ReadF32File(const char* path, std::size_t expect,
                                std::string* err) {
  std::vector<float> out;
  if (path == nullptr) return out;
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.good()) return out;
  ifs.seekg(0, std::ios::end);
  const std::size_t bytes = static_cast<std::size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  if (bytes != expect * sizeof(float)) return out;
  out.resize(expect);
  ifs.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(bytes));
  if (!ifs.good()) out.clear();
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
                           const std::vector<std::size_t>& tap_layers,
                           std::span<const float> z0,
                           std::span<const float> sigmas, std::size_t n_double,
                           std::size_t n_single,
                           std::vector<std::vector<float>>* taps_out) {
  using mbufo::core::npu::NpuShardReader;
  using mbufo::core::npu::TeNpuDecoder;
  std::string err;
  // TE taps (shell tap layers 0/18/final).
  TeNpuDecoder te;
  assert(te.Open("/mnt/nvme-data/mbufo-models/flux2-klein-4b/text_encoder",
                 &engine, mode, {}, &err));
  NpuShardReader tr;
  assert(NpuShardReader::OpenDir(
      "/mnt/nvme-data/mbufo-models/flux2-klein-4b/text_encoder", &tr, &err));
  std::vector<float> erows;
  assert(tr.GatherRowsF32("model.embed_tokens.weight", ids, 2560, &erows,
                          &err));
  const std::vector<std::size_t> tap_layers_v = tap_layers;
  std::vector<std::vector<float>> taps;
  std::vector<float> final_raw;
  // Padding-key mask mirrors the pipeline's causal+padding mask: only the
  // first NPU_REAL_TOKENS rows are real (rest are pads).
  std::vector<std::uint8_t> key_keep;
  const std::vector<std::uint8_t>* key_keep_ptr = nullptr;
  if (const char* rt = std::getenv("NPU_REAL_TOKENS")) {
    const std::size_t n_real = static_cast<std::size_t>(std::stoul(rt));
    key_keep.assign(ids.size(), 0);
    for (std::size_t i = 0; i < n_real && i < ids.size(); ++i) key_keep[i] = 1;
    key_keep_ptr = &key_keep;
  }
  assert(te.EncodeTaps(erows, ids.size(), tap_layers_v, &taps, &final_raw, ms,
                       &err, key_keep_ptr));
  assert(taps.size() == tap_layers.size());
  // cond7680 = [tap(9), tap(18), tap(27)] == pipeline hs[10,19,28].
  assert(taps.size() == 3);
  std::vector<float> cond(ids.size() * 7680);
  for (std::size_t t = 0; t < ids.size(); ++t) {
    std::copy(taps[0].data() + t * 2560, taps[0].data() + (t + 1) * 2560,
              cond.data() + t * 7680);
    std::copy(taps[1].data() + t * 2560, taps[1].data() + (t + 1) * 2560,
              cond.data() + t * 7680 + 2560);
    std::copy(taps[2].data() + t * 2560, taps[2].data() + (t + 1) * 2560,
              cond.data() + t * 7680 + 5120);
  }
  FluxDitNpu dit;
  assert(dit.Open("/mnt/nvme-data/mbufo-models/flux2-klein-4b/transformer",
                  &engine, mode, {}, &err));
  std::vector<float> txt;
  assert(dit.EncodeContext(cond, ids.size(), &txt, ms, &err));

  constexpr std::size_t Si = 64;
  std::vector<float> z(Si * 128);
  if (!z0.empty()) {
    assert(z0.size() == z.size());
    std::copy(z0.begin(), z0.end(), z.begin());
  } else {
    for (std::size_t i = 0; i < Si * 128; ++i)
      z[i] = 0.25F * std::cos(0.037F * static_cast<float>(i + 1u));
  }
  const std::size_t steps = sigmas.size() - 1;
  for (std::size_t step = 0; step < steps; ++step) {
    if (!dit.FullStep(&z, Si, txt, ids.size(), sigmas[step],
                      sigmas[step] - sigmas[step + 1], n_double, n_single, ms,
                      &err)) {
      std::fprintf(stderr, "FullStep %zu failed: %s\n", step, err.c_str());
      assert(false);
    }
  }
  if (taps_out != nullptr) *taps_out = taps;
  return z;
}

void TestFull(mbufo::core::npu::NpuQuantGemmEngine& engine,
              const std::vector<std::uint32_t>& ids, const char* dump_prefix) {
  using mbufo::core::npu::NpuQuantMode;
  // Pipeline-faithful taps: HF hidden_states[0]=embed, so hidden_states[9,18,27]
  // == post-layer {9,18,27} (0-indexed).
  const std::vector<std::size_t> taps = {9, 18, 27};
  const float kSigmas[5] = {1.0F, 0.75F, 0.50F, 0.25F, 0.0F};
  std::vector<float> sigmas(kSigmas, kSigmas + 5);
  std::vector<float> z0;
  if (const char* zp = std::getenv("NPU_LOAD_Z0")) {
    std::string err;
    z0 = ReadF32File(zp, 64 * 128, &err);
    assert(!z0.empty());
  }
  if (const char* sp = std::getenv("NPU_SIGMAS")) {
    std::ifstream ifs(sp);
    assert(ifs.good());
    sigmas.clear();
    float v = 0;
    while (ifs >> v) sigmas.push_back(v);
    assert(sigmas.size() >= 2);
  }
  double ms8 = 0.0, ms8b = 0.0, ms4 = 0.0;
  std::vector<std::vector<float>> taps8;
  std::vector<float> z8 =
      RunFull(engine, NpuQuantMode::kQ8Bfp16, &ms8, ids, taps, z0, sigmas, 5,
              20, dump_prefix != nullptr ? &taps8 : nullptr);
  std::vector<float> z8b =
      RunFull(engine, NpuQuantMode::kQ8Bfp16, &ms8b, ids, taps, z0, sigmas, 5,
              20, nullptr);
  assert(z8 == z8b);
  double energy = 0.0;
  for (float v : z8) {
    assert(std::isfinite(v));
    energy += (double)v * v;
  }
  assert(energy > 1.0);
  std::vector<float> z4 = RunFull(engine, NpuQuantMode::kQ4Q4nx, &ms4, ids,
                                  taps, z0, sigmas, 5, 20, nullptr);
  const float c84 = CosineSimHost(z8, z4);
  std::printf(
      "[flux-full 5dbl+20sgl x4] q8_ms=%.1f deter=1 energy=%.1f q8q4_cos=%.6f "
      "q4_ms=%.1f\n",
      ms8, energy, c84, ms4);
  std::fflush(stdout);
  if (dump_prefix != nullptr) {
    const std::string pre = dump_prefix;
    WriteF32File((pre + "_z.f32").c_str(), z8);
    for (std::size_t i = 0; i < taps8.size(); ++i) {
      WriteF32File((pre + "_tap" + std::to_string(i) + ".f32").c_str(),
                   taps8[i]);
    }
    std::printf("[flux-dump] wrote z(%zu) + %zu taps to %s_*\n", z8.size(),
                taps8.size(), dump_prefix);
    std::fflush(stdout);
  }
  // q8-vs-q4 drift on klein (measured 0.8526, 2026-11-24); E2E-vs-pipe bar is 0.94 (measured 0.9485).
  assert(c84 >= 0.80F);
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
  std::vector<std::uint32_t> ids = {151644, 872, 198,  32, 6977,
                                    17822,  4128, 304, 151645};
  if (const char* ip = std::getenv("NPU_DUMP_IDS")) {
    std::ifstream ifs(ip);
    assert(ifs.good());
    ids.clear();
    std::uint32_t v = 0;
    while (ifs >> v) ids.push_back(v);
    assert(!ids.empty());
    std::printf("[flux-dump] %zu ids from %s\n", ids.size(), ip);
    std::fflush(stdout);
  }
  TestFull(engine, ids, std::getenv("NPU_DUMP_LATENT"));
  std::puts("FLUX_DIT_PASS");
  return 0;
}
