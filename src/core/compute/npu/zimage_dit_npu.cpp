// Z-Image-Turbo DiT NPU implementation. Exact port of the parity shell's
// block math (weighted RMSNorms, scale/gate mod layout, SwiGLU FFN,
// concat-stream main layers, silu-gated final head); NPU GEMMs, host rest.

#include "src/core/compute/npu/zimage_dit_npu.hpp"

#include <cmath>

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

bool ZImageDitNpu::CapEmbed(std::span<const float> te_final,
                            std::size_t tokens, std::vector<float>* cap,
                            double* npu_ms, std::string* error) {
  std::vector<float> norm_w, normed;
  if (!reader_.ReadF32("cap_embedder.0.weight", 2560, &norm_w, error)) {
    return false;
  }
  RmsNormRows(te_final.data(), norm_w.data(), tokens, 2560, cfg_.eps,
              &normed);
  return lin_.Run("cap_embedder.1.weight", "cap_embedder.1.bias", normed,
                  tokens, cfg_.d, 2560, cap, npu_ms, error);
}

bool ZImageDitNpu::RefineContext(std::vector<float>* cap, std::size_t tokens,
                                 std::size_t n_ref, double* npu_ms,
                                 std::string* error) {
  for (std::size_t bi = 0; bi < n_ref; ++bi) {
    if (!Block("context_refiner." + std::to_string(bi), cap, tokens, nullptr,
               npu_ms, error)) {
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

bool ZImageDitNpu::Block(const std::string& prefix, std::vector<float>* h,
                         std::size_t seq, const float* mod, double* npu_ms,
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
  std::vector<float> ctx, o, on;
  RopeAttn(q, k, v, nq_w, nk_w, seq, cfg_.heads, cfg_.heads, cfg_.head_dim,
           cfg_.rope_theta, cfg_.eps, false, &ctx);
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
  if (!TEmbed(sigma, &temb, npu_ms, error)) return false;
  std::vector<float> img;
  if (!XEmbed(*z, Si, &img, npu_ms, error)) return false;

  for (std::size_t bi = 0; bi < n_ref; ++bi) {
    std::vector<float> mod;
    if (!lin_.Run("noise_refiner." + std::to_string(bi) +
                      ".adaLN_modulation.0.weight",
                  "noise_refiner." + std::to_string(bi) +
                      ".adaLN_modulation.0.bias",
                  temb, 1, 4 * D, 256, &mod, npu_ms, error)) {
      return false;
    }
    if (!Block("noise_refiner." + std::to_string(bi), &img, Si, mod.data(),
               npu_ms, error)) {
      return false;
    }
  }

  const std::size_t Sall = Si + St;
  std::vector<float> all(Sall * D);
  std::copy(img.begin(), img.end(), all.begin());
  std::copy(cap.begin(), cap.end(), all.begin() + Si * D);
  for (std::size_t bi = 0; bi < n_main; ++bi) {
    std::vector<float> mod;
    if (!lin_.Run("layers." + std::to_string(bi) + ".adaLN_modulation.0.weight",
                  "layers." + std::to_string(bi) + ".adaLN_modulation.0.bias",
                  temb, 1, 4 * D, 256, &mod, npu_ms, error)) {
      return false;
    }
    if (!Block("layers." + std::to_string(bi), &all, Sall, mod.data(), npu_ms,
               error)) {
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
  std::vector<float> fn;
  RmsNormRows(img_out.data(), nullptr, Si, D, cfg_.eps, &fn);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c) fn[t * D + c] *= (1.0F + fmod[c]);
  std::vector<float> v;
  if (!lin_.Run("all_final_layer.2-1.linear.weight",
                "all_final_layer.2-1.linear.bias", fn, Si, 64, D, &v, npu_ms,
                error)) {
    return false;
  }
  for (std::size_t i = 0; i < Si * 64; ++i) (*z)[i] -= dt * v[i];
  return true;
}

}  // namespace mbufo::core::npu::dit
