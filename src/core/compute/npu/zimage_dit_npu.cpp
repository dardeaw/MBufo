// Z-Image-Turbo DiT TRUE PORT of the diffusers pipeline math (P8b).
// Deltas vs the v1 shell port:
// - complex-cis 3-axis RoPE (axes 32/48/48, theta 256) via adjacent-pair
//   tables (same rotation math as the complex path)
// - cap padded to 32 with learned cap_pad_token; real cap t=1..St,
//   pad rows [0,0,0]; img t=33
// - joint [img(64); cap(32)]; per-token rope tables in joint order
// - FinalLayer: LayerNorm (no affine, eps 1e-6) + (1 + scale)
// Unchanged: RMSNorms, mod chunk(4) layout, tanh gates, SwiGLU FFN,
// no-mod refiner branch, t_scale=1000, [x, cap] order.

#include "src/core/compute/npu/zimage_dit_npu.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>

namespace mbufo::core::npu::dit {

bool ZImageDitNpu::Open(const std::string& transformer_dir,
                        NpuQuantGemmEngine* engine, NpuQuantMode mode,
                        const ZImageDitConfig& config, std::string* error) {
  if (engine == nullptr || !engine->ready()) {
    if (error != nullptr) *error = "NPU engine not ready";
    return false;
  }
  if (!NpuShardReader::OpenDir(transformer_dir, &reader_, error)) return false;
  engine_ = engine;
  mode_ = mode;
  cfg_ = config;
  lin_.Bind(&reader_, engine_, mode_);
  return true;
}

bool ZImageDitNpu::CapEmbed(std::span<const float> te_tap34,
                            std::size_t tokens, std::vector<float>* cap,
                            double* npu_ms, std::string* error) {
  std::vector<float> norm_w, normed;
  if (!reader_.ReadF32("cap_embedder.0.weight", 2560, &norm_w, error)) {
    return false;
  }
  RmsNormRows(te_tap34.data(), norm_w.data(), tokens, 2560, cfg_.eps,
              &normed);
  return lin_.Run("cap_embedder.1.weight", "cap_embedder.1.bias", normed,
                  tokens, cfg_.d, 2560, cap, npu_ms, error);
}

void ZImageDitNpu::PadCap(std::span<const float> cap_in, std::size_t tokens,
                           std::size_t hidden, std::size_t padded,
                           std::span<const float> pad_token,
                           std::vector<float>* cap_out) {
  cap_out->assign(padded * hidden, 0.0F);
  std::copy(cap_in.begin(), cap_in.begin() + tokens * hidden,
            cap_out->begin());
  for (std::size_t t = tokens; t < padded; ++t) {
    std::copy(pad_token.begin(), pad_token.begin() + hidden,
              cap_out->begin() + t * hidden);
  }
}

bool ZImageDitNpu::RefineContext(std::vector<float>* cap, std::size_t tokens,
                                 std::size_t n_ref,
                                 std::span<const float> rope_cos,
                                 std::span<const float> rope_sin,
                                 double* npu_ms, std::string* error) {
  for (std::size_t bi = 0; bi < n_ref; ++bi) {
    if (!Block("context_refiner." + std::to_string(bi), cap, tokens, nullptr,
               rope_cos, rope_sin, npu_ms, error)) {
      return false;
    }
  }
  return true;
}

bool ZImageDitNpu::TEmbed(float sigma, std::vector<float>* temb,
                          double* npu_ms, std::string* error) {
  std::vector<float> t_sin, tm0;
  TimestepSinusoid(sigma, &t_sin);
  if (!lin_.Run("t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias", t_sin, 1,
                1024, 256, &tm0, npu_ms, error)) {
    return false;
  }
  SiluInPlace(tm0);
  return lin_.Run("t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias", tm0, 1,
                  256, 1024, temb, npu_ms, error);
}

bool ZImageDitNpu::XEmbed(std::span<const float> z, std::size_t patches,
                          std::vector<float>* img, double* npu_ms,
                          std::string* error) {
  return lin_.Run("all_x_embedder.2-1.weight", "all_x_embedder.2-1.bias", z,
                  patches, cfg_.d, 64, img, npu_ms, error);
}

void ZImageDitNpu::BuildRopeTables(std::size_t cap_tokens,
                                    std::size_t cap_real, std::size_t Si,
                                    std::vector<float>* cos_t,
                                    std::vector<float>* sin_t) const {
  // Joint order [img; cap]: img rows (t=33,h,w), real cap rows (t=1..St),
  // pad cap rows [0,0,0] (pipeline _pad_with_ids parity).
  const std::size_t S = Si + cap_tokens;
  std::vector<std::int32_t> ids(S * 4u, 0);
  for (std::size_t t = 0; t < Si; ++t) {
    std::int32_t* row = ids.data() + t * 4u;
    row[0] = 33;
    row[1] = static_cast<std::int32_t>(t / 8u);
    row[2] = static_cast<std::int32_t>(t % 8u);
  }
  for (std::size_t t = 0; t < cap_real && t < cap_tokens; ++t) {
    std::int32_t* row = ids.data() + (Si + t) * 4u;
    row[0] = static_cast<std::int32_t>(t + 1u);
  }
  // Pad rows (t >= cap_real) stay [0,0,0] (ids zero-initialized).
  FluxRopeTables(ids, S, cfg_.rope_axes, 3u, (double)cfg_.rope_theta, cos_t,
                 sin_t);
}

bool ZImageDitNpu::Block(const std::string& prefix, std::vector<float>* h,
                         std::size_t seq, const float* mod,
                         std::span<const float> rope_cos,
                         std::span<const float> rope_sin, double* npu_ms,
                         std::string* error) {
  const std::size_t D = cfg_.d;
  const std::size_t F = cfg_.ffn;
  std::vector<float> an1_w, an2_w, fn1_w, fn2_w, nq_w, nk_w;
  if (!reader_.ReadF32(prefix + ".attention_norm1.weight", D, &an1_w, error) ||
      !reader_.ReadF32(prefix + ".attention_norm2.weight", D, &an2_w, error) ||
      !reader_.ReadF32(prefix + ".ffn_norm1.weight", D, &fn1_w, error) ||
      !reader_.ReadF32(prefix + ".ffn_norm2.weight", D, &fn2_w, error) ||
      !reader_.ReadF32(prefix + ".attention.norm_q.weight", cfg_.head_dim,
                       &nq_w, error) ||
      !reader_.ReadF32(prefix + ".attention.norm_k.weight", cfg_.head_dim,
                       &nk_w, error)) {
    return false;
  }
  std::vector<float> x;
  RmsNormRows(h->data(), an1_w.data(), seq, D, cfg_.eps, &x);
  if (mod != nullptr) {
    for (std::size_t t = 0; t < seq; ++t)
      for (std::size_t c = 0; c < D; ++c) x[t * D + c] *= (1.0F + mod[c]);
  }
  std::vector<float> q, k, v;
  if (!lin_.Run(prefix + ".attention.to_q.weight", "", x, seq, D, D, &q,
                npu_ms, error) ||
      !lin_.Run(prefix + ".attention.to_k.weight", "", x, seq, D, D, &k,
                npu_ms, error) ||
      !lin_.Run(prefix + ".attention.to_v.weight", "", x, seq, D, D, &v,
                npu_ms, error)) {
    return false;
  }
  std::vector<float> qn(seq * D), kn(seq * D);
  for (std::size_t t = 0; t < seq; ++t) {
    for (std::size_t hh = 0; hh < cfg_.heads; ++hh) {
      std::copy(nq_w.begin(), nq_w.end(), qn.begin() + t * D + hh * cfg_.head_dim);
      std::copy(nk_w.begin(), nk_w.end(), kn.begin() + t * D + hh * cfg_.head_dim);
    }
  }
  std::vector<float> ctx, o, on;
  JointAttnTables(q, k, v, qn, kn, seq, seq, cfg_.heads, cfg_.heads,
                  cfg_.head_dim, rope_cos, rope_sin, rope_cos, rope_sin,
                  cfg_.eps, &ctx);
  if (!lin_.Run(prefix + ".attention.to_out.0.weight", "", ctx, seq, D, D, &o,
                npu_ms, error)) {
    return false;
  }
  RmsNormRows(o.data(), an2_w.data(), seq, D, cfg_.eps, &on);
  for (std::size_t t = 0; t < seq; ++t)
    for (std::size_t c = 0; c < D; ++c) {
      const float g = (mod != nullptr) ? std::tanh(mod[D + c]) : 1.0F;
      (*h)[t * D + c] += g * on[t * D + c];
    }

  RmsNormRows(h->data(), fn1_w.data(), seq, D, cfg_.eps, &x);
  if (mod != nullptr) {
    for (std::size_t t = 0; t < seq; ++t)
      for (std::size_t c = 0; c < D; ++c) x[t * D + c] *= (1.0F + mod[2 * D + c]);
  }
  std::vector<float> w1, w3;
  if (!lin_.Run(prefix + ".feed_forward.w1.weight", "", x, seq, F, D, &w1,
                npu_ms, error) ||
      !lin_.Run(prefix + ".feed_forward.w3.weight", "", x, seq, F, D, &w3,
                npu_ms, error)) {
    return false;
  }
  for (std::size_t i = 0; i < seq * F; ++i) {
    w1[i] = (w1[i] / (1.0F + std::exp(-w1[i]))) * w3[i];
  }
  std::vector<float> w2, fn;
  if (!lin_.Run(prefix + ".feed_forward.w2.weight", "", w1, seq, D, F, &w2,
                npu_ms, error)) {
    return false;
  }
  RmsNormRows(w2.data(), fn2_w.data(), seq, D, cfg_.eps, &fn);
  for (std::size_t t = 0; t < seq; ++t)
    for (std::size_t c = 0; c < D; ++c) {
      const float g = (mod != nullptr) ? std::tanh(mod[3 * D + c]) : 1.0F;
      (*h)[t * D + c] += g * fn[t * D + c];
    }
  return true;
}

bool ZImageDitNpu::FullStep(std::vector<float>* z, std::size_t Si,
                            std::span<const float> cap, std::size_t St,
                            float sigma, float dt, std::size_t n_ref,
                            std::size_t n_main, double* npu_ms,
                            std::string* error) {
  const std::size_t D = cfg_.d;
  std::vector<float> temb;
  if (!TEmbed(1.0F - sigma, &temb, npu_ms, error)) return false;
  std::vector<float> img;
  if (!XEmbed(*z, Si, &img, npu_ms, error)) return false;
  std::vector<float> pad_tok;
  if (!reader_.ReadF32("cap_pad_token", D, &pad_tok, error)) return false;
  std::vector<float> cap32;
  PadCap(cap, St, D, cfg_.cap_pad, pad_tok, &cap32);
  if (const char* dp = std::getenv("NPU_DUMP_MID")) {
    const std::string pre = dp;
    std::ofstream ofs(pre + "_cap32.f32", std::ios::binary);
    if (ofs.good()) {
      ofs.write(reinterpret_cast<const char*>(cap32.data()),
                static_cast<std::streamsize>(cap32.size() * sizeof(float)));
    }
  }
  std::vector<float> rope_cos, rope_sin;
  BuildRopeTables(cfg_.cap_pad, St, Si, &rope_cos, &rope_sin);
  // Refiner rope = cap rows of the joint tables.
  const std::size_t img_rows = Si * 128;
  std::vector<float> cap_cos(rope_cos.begin() + img_rows, rope_cos.end());
  std::vector<float> cap_sin(rope_sin.begin() + img_rows, rope_sin.end());
  if (!RefineContext(&cap32, cfg_.cap_pad, n_ref, cap_cos, cap_sin, npu_ms,
                     error)) {
    return false;
  }

  for (std::size_t bi = 0; bi < n_ref; ++bi) {
    std::vector<float> mod;
    if (!lin_.Run("noise_refiner." + std::to_string(bi) +
                      ".adaLN_modulation.0.weight",
                  "noise_refiner." + std::to_string(bi) +
                      ".adaLN_modulation.0.bias",
                  temb, 1, 4 * D, 256, &mod, npu_ms, error)) {
      return false;
    }
    // Noise-refiner rope = img rows of the joint tables.
    std::vector<float> img_cos(rope_cos.begin(), rope_cos.begin() + img_rows);
    std::vector<float> img_sin(rope_sin.begin(), rope_sin.begin() + img_rows);
    if (!Block("noise_refiner." + std::to_string(bi), &img, Si, mod.data(),
               img_cos, img_sin, npu_ms, error)) {
      return false;
    }
  }

  const std::size_t Sall = Si + cfg_.cap_pad;
  std::vector<float> all(Sall * D);
  std::copy(img.begin(), img.end(), all.begin());
  std::copy(cap32.begin(), cap32.end(), all.begin() + Si * D);
  for (std::size_t bi = 0; bi < n_main; ++bi) {
    std::vector<float> mod;
    if (!lin_.Run("layers." + std::to_string(bi) + ".adaLN_modulation.0.weight",
                  "layers." + std::to_string(bi) + ".adaLN_modulation.0.bias",
                  temb, 1, 4 * D, 256, &mod, npu_ms, error)) {
      return false;
    }
    if (!Block("layers." + std::to_string(bi), &all, Sall, mod.data(),
               rope_cos, rope_sin, npu_ms, error)) {
      return false;
    }
  }

  std::vector<float> ts = temb;
  SiluInPlace(ts);
  std::vector<float> fmod;
  if (!lin_.Run("all_final_layer.2-1.adaLN_modulation.1.weight",
                "all_final_layer.2-1.adaLN_modulation.1.bias", ts, 1, D, 256,
                &fmod, npu_ms, error)) {
    return false;
  }
  std::vector<float> img_out(Si * D);
  std::copy(all.begin(), all.begin() + Si * D, img_out.begin());
  // FinalLayer: LayerNorm (no affine, eps 1e-6) + (1 + scale).
  std::vector<float> fn;
  LayerNormRows(img_out.data(), Si, D, cfg_.final_eps, &fn);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c) fn[t * D + c] *= (1.0F + fmod[c]);
  std::vector<float> v;
  if (!lin_.Run("all_final_layer.2-1.linear.weight",
                "all_final_layer.2-1.linear.bias", fn, Si, 64, D, &v, npu_ms,
                error)) {
    return false;
  }
  if (const char* dp = std::getenv("NPU_DUMP_MID")) {
    const std::string pre = dp;
    auto dump = [&](const char* n, const std::vector<float>& x) {
      std::ofstream ofs(pre + "_" + n + ".f32", std::ios::binary);
      if (ofs.good()) {
        ofs.write(reinterpret_cast<const char*>(x.data()),
                  static_cast<std::streamsize>(x.size() * sizeof(float)));
      }
    };
    dump("temb", temb);
    dump("img0", img);
    dump("v", v);
  }
  for (std::size_t i = 0; i < Si * 64; ++i) (*z)[i] += dt * v[i];  // pipeline net: z+=|dt|*out
  return true;
}

}  // namespace mbufo::core::npu::dit
