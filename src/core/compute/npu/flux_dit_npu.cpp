// FLUX.2-klein DiT NPU implementation. Exact port of the parity shell's
// per-step math (modulation layout, gate indices, GeGLU/SwiGLU splits,
// Euler update); every Linear is NPU, every norm/activation/softmax is host.

#include "src/core/compute/npu/flux_dit_npu.hpp"

#include <cmath>

namespace mbufo::core::npu::dit {

bool FluxDitNpu::Open(const std::string& transformer_dir,
                      NpuQuantGemmEngine* engine, NpuQuantMode mode,
                      const FluxDitConfig& config, std::string* error) {
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

bool FluxDitNpu::TimestepEmbed(float sigma, std::vector<float>* temb,
                               double* npu_ms, std::string* error) {
  std::vector<float> t_sin;
  TimestepSinusoid(sigma, &t_sin);
  std::vector<float> t1;
  if (!lin_.Run("time_guidance_embed.timestep_embedder.linear_1.weight", "",
                t_sin, 1, cfg_.d, 256, &t1, npu_ms, error)) {
    return false;
  }
  SiluInPlace(t1);
  if (!lin_.Run("time_guidance_embed.timestep_embedder.linear_2.weight", "",
                t1, 1, cfg_.d, cfg_.d, temb, npu_ms, error)) {
    return false;
  }
  SiluInPlace(*temb);
  return true;
}

bool FluxDitNpu::Modulations(std::span<const float> temb, FluxModulations* mod,
                             double* npu_ms, std::string* error) {
  if (temb.size() < cfg_.d) {
    if (error != nullptr) *error = "bad temb";
    return false;
  }
  return lin_.Run("double_stream_modulation_img.linear.weight", "", temb, 1,
                  6 * cfg_.d, cfg_.d, &mod->img, npu_ms, error) &&
         lin_.Run("double_stream_modulation_txt.linear.weight", "", temb, 1,
                  6 * cfg_.d, cfg_.d, &mod->txt, npu_ms, error) &&
         lin_.Run("single_stream_modulation.linear.weight", "", temb, 1,
                  3 * cfg_.d, cfg_.d, &mod->sgl, npu_ms, error);
}

bool FluxDitNpu::EncodeContext(std::span<const float> cond7680,
                               std::size_t tokens, std::vector<float>* txt,
                               double* npu_ms, std::string* error) {
  return lin_.Run("context_embedder.weight", "", cond7680, tokens, cfg_.d,
                  7680, txt, npu_ms, error);
}

bool FluxDitNpu::XEmbed(std::span<const float> z, std::size_t patches,
                        std::vector<float>* img, double* npu_ms,
                        std::string* error) {
  return lin_.Run("x_embedder.weight", "", z, patches, cfg_.d, 128, img,
                  npu_ms, error);
}

bool FluxDitNpu::DoubleBlock(std::vector<float>* img, std::vector<float>* txt,
                             std::size_t Si, std::size_t St, std::size_t bi,
                             std::span<const float> mod_img,
                             std::span<const float> mod_txt, double* npu_ms,
                             std::string* error) {
  const std::size_t D = cfg_.d;
  const std::string p = "transformer_blocks." + std::to_string(bi) + ".";
  std::vector<float> nq_w, nk_w, naq_w, nak_w;
  if (!reader_.ReadF32(p + "attn.norm_q.weight", cfg_.head_dim, &nq_w, error) ||
      !reader_.ReadF32(p + "attn.norm_k.weight", cfg_.head_dim, &nk_w, error) ||
      !reader_.ReadF32(p + "attn.norm_added_q.weight", cfg_.head_dim, &naq_w,
                       error) ||
      !reader_.ReadF32(p + "attn.norm_added_k.weight", cfg_.head_dim, &nak_w,
                       error)) {
    return false;
  }
  std::vector<float> in_img, in_txt;
  RmsNormRows(img->data(), nullptr, Si, D, cfg_.eps, &in_img);
  RmsNormRows(txt->data(), nullptr, St, D, cfg_.eps, &in_txt);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      in_img[t * D + c] =
          in_img[t * D + c] * (1.0F + mod_img[D + c]) + mod_img[c];
  for (std::size_t t = 0; t < St; ++t)
    for (std::size_t c = 0; c < D; ++c)
      in_txt[t * D + c] =
          in_txt[t * D + c] * (1.0F + mod_txt[D + c]) + mod_txt[c];

  std::vector<float> qi, ki, vi, qt, kt, vt;
  if (!lin_.Run(p + "attn.to_q.weight", "", in_img, Si, D, D, &qi, npu_ms, error) ||
      !lin_.Run(p + "attn.to_k.weight", "", in_img, Si, D, D, &ki, npu_ms, error) ||
      !lin_.Run(p + "attn.to_v.weight", "", in_img, Si, D, D, &vi, npu_ms, error) ||
      !lin_.Run(p + "attn.add_q_proj.weight", "", in_txt, St, D, D, &qt, npu_ms, error) ||
      !lin_.Run(p + "attn.add_k_proj.weight", "", in_txt, St, D, D, &kt, npu_ms, error) ||
      !lin_.Run(p + "attn.add_v_proj.weight", "", in_txt, St, D, D, &vt, npu_ms, error)) {
    return false;
  }
  std::vector<float> ctx_i, ctx_t;
  RopeAttn(qi, ki, vi, nq_w, nk_w, Si, cfg_.heads, cfg_.heads, cfg_.head_dim,
           cfg_.rope_theta, cfg_.eps, false, &ctx_i);
  RopeAttn(qt, kt, vt, naq_w, nak_w, St, cfg_.heads, cfg_.heads, cfg_.head_dim,
           cfg_.rope_theta, cfg_.eps, false, &ctx_t);
  std::vector<float> oi, ot;
  if (!lin_.Run(p + "attn.to_out.0.weight", "", ctx_i, Si, D, D, &oi, npu_ms, error) ||
      !lin_.Run(p + "attn.to_add_out.weight", "", ctx_t, St, D, D, &ot, npu_ms, error)) {
    return false;
  }
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*img)[t * D + c] += std::tanh(mod_img[2 * D + c]) * oi[t * D + c];
  for (std::size_t t = 0; t < St; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*txt)[t * D + c] += std::tanh(mod_txt[2 * D + c]) * ot[t * D + c];

  // Image FFN: linear_in [6D, D] -> SwiGLU halves -> linear_out [D, 3D],
  // output gated by tanh(mod[5D]).
  std::vector<float> ffi;
  RmsNormRows(img->data(), nullptr, Si, D, cfg_.eps, &ffi);
  std::vector<float> ffi_in, ffi_act(Si * 3 * D), ffi_out;
  if (!lin_.Run(p + "ff.linear_in.weight", "", ffi, Si, 6 * D, D, &ffi_in,
                npu_ms, error)) {
    return false;
  }
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < 3 * D; ++c)
      ffi_act[t * 3 * D + c] =
          (ffi_in[t * 6 * D + c] / (1.0F + std::exp(-ffi_in[t * 6 * D + c]))) *
          ffi_in[t * 6 * D + 3 * D + c];
  if (!lin_.Run(p + "ff.linear_out.weight", "", ffi_act, Si, D, 3 * D,
                &ffi_out, npu_ms, error)) {
    return false;
  }
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*img)[t * D + c] += std::tanh(mod_img[5 * D + c]) * ffi_out[t * D + c];
  // NOTE: the parity shell does not apply the text FFN (ff_context.*);
  // matched exactly here. Revisit against diffusers during quality validation.
  return true;
}

bool FluxDitNpu::SingleBlock(std::vector<float>* img, std::size_t Si,
                             std::size_t bi, std::span<const float> mod_sgl,
                             double* npu_ms, std::string* error) {
  const std::size_t D = cfg_.d;
  const std::string p =
      "single_transformer_blocks." + std::to_string(bi) + ".attn.";
  std::vector<float> nq_w, nk_w;
  if (!reader_.ReadF32(p + "norm_q.weight", cfg_.head_dim, &nq_w, error) ||
      !reader_.ReadF32(p + "norm_k.weight", cfg_.head_dim, &nk_w, error)) {
    return false;
  }
  std::vector<float> s_in;
  RmsNormRows(img->data(), nullptr, Si, D, cfg_.eps, &s_in);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      s_in[t * D + c] =
          s_in[t * D + c] * (1.0F + mod_sgl[D + c]) + mod_sgl[c];
  std::vector<float> qkv_mlp;
  if (!lin_.Run(p + "to_qkv_mlp_proj.weight", "", s_in, Si, 9 * D, D,
                &qkv_mlp, npu_ms, error)) {
    return false;
  }
  std::vector<float> q(Si * D), k(Si * D), v(Si * D);
  for (std::size_t t = 0; t < Si; ++t) {
    const float* r = qkv_mlp.data() + t * 9 * D;
    std::copy(r, r + D, q.data() + t * D);
    std::copy(r + D, r + 2 * D, k.data() + t * D);
    std::copy(r + 2 * D, r + 3 * D, v.data() + t * D);
  }
  std::vector<float> ctx;
  RopeAttn(q, k, v, nq_w, nk_w, Si, cfg_.heads, cfg_.heads, cfg_.head_dim,
           cfg_.rope_theta, cfg_.eps, false, &ctx);
  std::vector<float> cat(Si * 4 * D);
  for (std::size_t t = 0; t < Si; ++t) {
    const float* r = qkv_mlp.data() + t * 9 * D;
    std::copy(ctx.data() + t * D, ctx.data() + (t + 1) * D,
              cat.data() + t * 4 * D);
    for (std::size_t c = 0; c < 3 * D; ++c)
      cat[t * 4 * D + D + c] =
          (r[3 * D + c] / (1.0F + std::exp(-r[3 * D + c]))) * r[6 * D + c];
  }
  std::vector<float> sout;
  if (!lin_.Run(p + "to_out.weight", "", cat, Si, D, 4 * D, &sout, npu_ms,
                error)) {
    return false;
  }
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*img)[t * D + c] += std::tanh(mod_sgl[2 * D + c]) * sout[t * D + c];
  return true;
}

bool FluxDitNpu::DenoiseBlocks(std::vector<float>* img, std::vector<float>* txt,
                               std::size_t Si, std::size_t St,
                               const FluxModulations& mod, std::size_t n_double,
                               std::size_t n_single, double* npu_ms,
                               std::string* error) {
  for (std::size_t bi = 0; bi < n_double; ++bi) {
    if (!DoubleBlock(img, txt, Si, St, bi, mod.img, mod.txt, npu_ms, error)) {
      return false;
    }
  }
  for (std::size_t bi = 0; bi < n_single; ++bi) {
    if (!SingleBlock(img, Si, bi, mod.sgl, npu_ms, error)) return false;
  }
  return true;
}

bool FluxDitNpu::VelocityHead(std::span<const float> img, std::size_t Si,
                             std::span<const float> temb, std::vector<float>* v,
                             double* npu_ms, std::string* error) {
  const std::size_t D = cfg_.d;
  std::vector<float> nout_mod;
  if (!lin_.Run("norm_out.linear.weight", "", temb, 1, 2 * D, D, &nout_mod,
                npu_ms, error)) {
    return false;
  }
  std::vector<float> n_img;
  RmsNormRows(img.data(), nullptr, Si, D, cfg_.eps, &n_img);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      n_img[t * D + c] =
          n_img[t * D + c] * (1.0F + nout_mod[D + c]) + nout_mod[c];
  return lin_.Run("proj_out.weight", "", n_img, Si, 128, D, v, npu_ms, error);
}

bool FluxDitNpu::FullStep(std::vector<float>* z, std::size_t Si,
                          std::span<const float> txt, std::size_t St,
                          float sigma, float dt, std::size_t n_double,
                          std::size_t n_single, double* npu_ms,
                          std::string* error) {
  std::vector<float> temb;
  if (!TimestepEmbed(sigma, &temb, npu_ms, error)) return false;
  FluxModulations mod;
  if (!Modulations(temb, &mod, npu_ms, error)) return false;
  std::vector<float> img;
  if (!XEmbed(*z, Si, &img, npu_ms, error)) return false;
  std::vector<float> txt_w(txt.begin(), txt.end());
  if (!DenoiseBlocks(&img, &txt_w, Si, St, mod, n_double, n_single, npu_ms,
                     error)) {
    return false;
  }
  std::vector<float> v;
  if (!VelocityHead(img, Si, temb, &v, npu_ms, error)) return false;
  for (std::size_t i = 0; i < Si * 128; ++i) (*z)[i] -= dt * v[i];
  return true;
}

}  // namespace mbufo::core::npu::dit
