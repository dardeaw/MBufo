// FLUX.2-klein DiT TRUE PORT of the diffusers pipeline math (P8a).
//
// Deltas vs the v1 shell-faithful port: LayerNorm (no affine), joint
// [txt;img] attention in every block, txt FFN, SiLU(temb) before modulation
// linears with [shift,scale,gate]x2 split, 4-axis adjacent-pair RoPE, raw
// gates, single-block plain-SiLU MLP, AdaLayerNormContinuous norm_out.

#include "src/core/compute/npu/flux_dit_npu.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>

namespace mbufo::core::npu::dit {

bool FluxDitNpu::Open(const std::string& transformer_dir,
                      NpuQuantGemmEngine* engine, NpuQuantMode mode,
                      const FluxDitConfig& config, std::string* error) {
  if (engine == nullptr || !engine->ready()) {
    if (error != nullptr)
      *error = "NPU engine not ready";
    return false;
  }
  if (!NpuShardReader::OpenDir(transformer_dir, &reader_, error))
    return false;
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
  // NOTE: no trailing SiLU (TimestepEmbedding post_act=None).
  return lin_.Run("time_guidance_embed.timestep_embedder.linear_2.weight", "",
                  t1, 1, cfg_.d, cfg_.d, temb, npu_ms, error);
}

bool FluxDitNpu::Modulations(std::span<const float> temb, FluxModulations* mod,
                             double* npu_ms, std::string* error) {
  if (temb.size() < cfg_.d) {
    if (error != nullptr)
      *error = "bad temb";
    return false;
  }
  // Flux2Modulation.forward: SiLU(temb) BEFORE the linear.
  std::vector<float> act(temb.begin(), temb.end());
  SiluInPlace(act);
  return lin_.Run("double_stream_modulation_img.linear.weight", "", act, 1,
                  6 * cfg_.d, cfg_.d, &mod->img, npu_ms, error) &&
         lin_.Run("double_stream_modulation_txt.linear.weight", "", act, 1,
                  6 * cfg_.d, cfg_.d, &mod->txt, npu_ms, error) &&
         lin_.Run("single_stream_modulation.linear.weight", "", act, 1,
                  3 * cfg_.d, cfg_.d, &mod->sgl, npu_ms, error);
}

bool FluxDitNpu::EncodeContext(std::span<const float> cond7680,
                               std::size_t tokens, std::vector<float>* txt,
                               double* npu_ms, std::string* error) {
  return lin_.Run("context_embedder.weight", "", cond7680, tokens, cfg_.d, 7680,
                  txt, npu_ms, error);
}

bool FluxDitNpu::XEmbed(std::span<const float> z, std::size_t patches,
                        std::vector<float>* img, double* npu_ms,
                        std::string* error) {
  return lin_.Run("x_embedder.weight", "", z, patches, cfg_.d, 128, img, npu_ms,
                  error);
}

void FluxDitNpu::BuildRopeTables(std::size_t St, std::size_t Si,
                                 std::vector<float>* cos_t,
                                 std::vector<float>* sin_t) const {
  const std::size_t S = St + Si;
  std::vector<std::int32_t> ids(S * 4u, 0);
  for (std::size_t l = 0; l < St; ++l) {
    ids[l * 4u + 3u] = static_cast<std::int32_t>(l);
  }
  for (std::size_t t = 0; t < Si; ++t) {
    const std::size_t h = t / 8u;
    const std::size_t w = t % 8u;
    std::int32_t* row = ids.data() + (St + t) * 4u;
    row[1] = static_cast<std::int32_t>(h);
    row[2] = static_cast<std::int32_t>(w);
  }
  FluxRopeTables(ids, S, cfg_.rope_axes, 4u, (double)cfg_.rope_theta, cos_t,
                 sin_t);
}

bool FluxDitNpu::DoubleBlock(std::vector<float>* img, std::vector<float>* txt,
                             std::size_t Si, std::size_t St, std::size_t bi,
                             std::span<const float> mod_img,
                             std::span<const float> mod_txt,
                             std::span<const float> rope_cos,
                             std::span<const float> rope_sin, double* npu_ms,
                             std::string* error) {
  const std::size_t D = cfg_.d;
  const std::size_t S = St + Si;
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
  // mod split: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
  const float* ish = mod_img.data();
  const float* isc = mod_img.data() + D;
  const float* imgate = mod_img.data() + 2u * D;
  const float* fsh = mod_img.data() + 3u * D;
  const float* fsc = mod_img.data() + 4u * D;
  const float* fgate = mod_img.data() + 5u * D;
  const float* tsh = mod_txt.data();
  const float* tsc = mod_txt.data() + D;
  const float* tgate = mod_txt.data() + 2u * D;
  const float* tfsh = mod_txt.data() + 3u * D;
  const float* tfsc = mod_txt.data() + 4u * D;
  const float* tfgate = mod_txt.data() + 5u * D;

  std::vector<float> in_img, in_txt;
  LayerNormRows(img->data(), Si, D, cfg_.eps, &in_img);
  LayerNormRows(txt->data(), St, D, cfg_.eps, &in_txt);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      in_img[t * D + c] = in_img[t * D + c] * (1.0F + isc[c]) + ish[c];
  for (std::size_t t = 0; t < St; ++t)
    for (std::size_t c = 0; c < D; ++c)
      in_txt[t * D + c] = in_txt[t * D + c] * (1.0F + tsc[c]) + tsh[c];

  std::vector<float> qi, ki, vi, qt, kt, vt;
  if (!lin_.Run(p + "attn.to_q.weight", "", in_img, Si, D, D, &qi, npu_ms,
                error) ||
      !lin_.Run(p + "attn.to_k.weight", "", in_img, Si, D, D, &ki, npu_ms,
                error) ||
      !lin_.Run(p + "attn.to_v.weight", "", in_img, Si, D, D, &vi, npu_ms,
                error) ||
      !lin_.Run(p + "attn.add_q_proj.weight", "", in_txt, St, D, D, &qt, npu_ms,
                error) ||
      !lin_.Run(p + "attn.add_k_proj.weight", "", in_txt, St, D, D, &kt, npu_ms,
                error) ||
      !lin_.Run(p + "attn.add_v_proj.weight", "", in_txt, St, D, D, &vt, npu_ms,
                error)) {
    return false;
  }
  // Joint [txt; img] attention.
  std::vector<float> q(S * D), k(S * D), v(S * D);
  std::copy(qt.begin(), qt.end(), q.begin());
  std::copy(qi.begin(), qi.end(), q.begin() + St * D);
  std::copy(kt.begin(), kt.end(), k.begin());
  std::copy(ki.begin(), ki.end(), k.begin() + St * D);
  std::copy(vt.begin(), vt.end(), v.begin());
  std::copy(vi.begin(), vi.end(), v.begin() + St * D);
  std::vector<float> qn(S * D, 0.0F), kn(S * D, 0.0F);
  for (std::size_t t = 0; t < St; ++t) {
    std::copy(naq_w.begin(), naq_w.end(), qn.begin() + t * D);
    std::copy(nak_w.begin(), nak_w.end(), kn.begin() + t * D);
    for (std::size_t h = 1; h < cfg_.heads; ++h) {
      std::copy(naq_w.begin(), naq_w.end(),
                qn.begin() + t * D + h * cfg_.head_dim);
      std::copy(nak_w.begin(), nak_w.end(),
                kn.begin() + t * D + h * cfg_.head_dim);
    }
  }
  for (std::size_t t = 0; t < Si; ++t) {
    for (std::size_t h = 0; h < cfg_.heads; ++h) {
      std::copy(nq_w.begin(), nq_w.end(),
                qn.begin() + (St + t) * D + h * cfg_.head_dim);
      std::copy(nk_w.begin(), nk_w.end(),
                kn.begin() + (St + t) * D + h * cfg_.head_dim);
    }
  }
  std::vector<float> ctx;
  JointAttnTables(q, k, v, qn, kn, S, S, cfg_.heads, cfg_.heads, cfg_.head_dim,
                  rope_cos, rope_sin, rope_cos, rope_sin, cfg_.eps, &ctx);
  std::vector<float> oi, ot;
  std::vector<float> ctx_t(ctx.begin(), ctx.begin() + St * D);
  std::vector<float> ctx_i(ctx.begin() + St * D, ctx.end());
  if (!lin_.Run(p + "attn.to_out.0.weight", "", ctx_i, Si, D, D, &oi, npu_ms,
                error) ||
      !lin_.Run(p + "attn.to_add_out.weight", "", ctx_t, St, D, D, &ot, npu_ms,
                error)) {
    return false;
  }
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*img)[t * D + c] += imgate[c] * oi[t * D + c];
  for (std::size_t t = 0; t < St; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*txt)[t * D + c] += tgate[c] * ot[t * D + c];

  // Image FFN (SwiGLU).
  std::vector<float> ffi;
  LayerNormRows(img->data(), Si, D, cfg_.eps, &ffi);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      ffi[t * D + c] = ffi[t * D + c] * (1.0F + fsc[c]) + fsh[c];
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
  if (!lin_.Run(p + "ff.linear_out.weight", "", ffi_act, Si, D, 3 * D, &ffi_out,
                npu_ms, error)) {
    return false;
  }
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*img)[t * D + c] += fgate[c] * ffi_out[t * D + c];

  // Text FFN (ff_context).
  std::vector<float> fft;
  LayerNormRows(txt->data(), St, D, cfg_.eps, &fft);
  for (std::size_t t = 0; t < St; ++t)
    for (std::size_t c = 0; c < D; ++c)
      fft[t * D + c] = fft[t * D + c] * (1.0F + tfsc[c]) + tfsh[c];
  std::vector<float> fft_in, fft_act(St * 3 * D), fft_out;
  if (!lin_.Run(p + "ff_context.linear_in.weight", "", fft, St, 6 * D, D,
                &fft_in, npu_ms, error)) {
    return false;
  }
  for (std::size_t t = 0; t < St; ++t)
    for (std::size_t c = 0; c < 3 * D; ++c)
      fft_act[t * 3 * D + c] =
          (fft_in[t * 6 * D + c] / (1.0F + std::exp(-fft_in[t * 6 * D + c]))) *
          fft_in[t * 6 * D + 3 * D + c];
  if (!lin_.Run(p + "ff_context.linear_out.weight", "", fft_act, St, D, 3 * D,
                &fft_out, npu_ms, error)) {
    return false;
  }
  for (std::size_t t = 0; t < St; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*txt)[t * D + c] += tfgate[c] * fft_out[t * D + c];
  return true;
}

bool FluxDitNpu::SingleBlock(std::vector<float>* joint, std::size_t Sj,
                             std::size_t bi, std::span<const float> mod_sgl,
                             std::span<const float> rope_cos,
                             std::span<const float> rope_sin, double* npu_ms,
                             std::string* error) {
  const std::size_t D = cfg_.d;
  const std::string p =
      "single_transformer_blocks." + std::to_string(bi) + ".attn.";
  std::vector<float> nq_w, nk_w;
  if (!reader_.ReadF32(p + "norm_q.weight", cfg_.head_dim, &nq_w, error) ||
      !reader_.ReadF32(p + "norm_k.weight", cfg_.head_dim, &nk_w, error)) {
    return false;
  }
  const float* shift = mod_sgl.data();
  const float* scale = mod_sgl.data() + D;
  const float* gate = mod_sgl.data() + 2u * D;
  std::vector<float> s_in;
  LayerNormRows(joint->data(), Sj, D, cfg_.eps, &s_in);
  for (std::size_t t = 0; t < Sj; ++t)
    for (std::size_t c = 0; c < D; ++c)
      s_in[t * D + c] = s_in[t * D + c] * (1.0F + scale[c]) + shift[c];
  std::vector<float> qkv_mlp;
  if (!lin_.Run(p + "to_qkv_mlp_proj.weight", "", s_in, Sj, 9 * D, D, &qkv_mlp,
                npu_ms, error)) {
    return false;
  }
  std::vector<float> q(Sj * D), k(Sj * D), v(Sj * D);
  for (std::size_t t = 0; t < Sj; ++t) {
    const float* r = qkv_mlp.data() + t * 9 * D;
    std::copy(r, r + D, q.data() + t * D);
    std::copy(r + D, r + 2 * D, k.data() + t * D);
    std::copy(r + 2 * D, r + 3 * D, v.data() + t * D);
  }
  // qk-norm weights broadcast per head.
  std::vector<float> qn(Sj * D), kn(Sj * D);
  for (std::size_t t = 0; t < Sj; ++t) {
    for (std::size_t h = 0; h < cfg_.heads; ++h) {
      std::copy(nq_w.begin(), nq_w.end(),
                qn.begin() + t * D + h * cfg_.head_dim);
      std::copy(nk_w.begin(), nk_w.end(),
                kn.begin() + t * D + h * cfg_.head_dim);
    }
  }
  std::vector<float> ctx;
  JointAttnTables(q, k, v, qn, kn, Sj, Sj, cfg_.heads, cfg_.heads,
                  cfg_.head_dim, rope_cos, rope_sin, rope_cos, rope_sin,
                  cfg_.eps, &ctx);
  // MLP: SwiGLU over the fused halves (gate=r[9216:18432], up=r[18432:27648]).
  std::vector<float> cat(Sj * 4 * D);
  for (std::size_t t = 0; t < Sj; ++t) {
    const float* r = qkv_mlp.data() + t * 9 * D;
    std::copy(ctx.data() + t * D, ctx.data() + (t + 1) * D,
              cat.data() + t * 4 * D);
    for (std::size_t c = 0; c < 3 * D; ++c) {
      const float g = r[3 * D + c];
      cat[t * 4 * D + D + c] = (g / (1.0F + std::exp(-g))) * r[6 * D + c];
    }
  }
  std::vector<float> sout;
  if (!lin_.Run(p + "to_out.weight", "", cat, Sj, D, 4 * D, &sout, npu_ms,
                error)) {
    return false;
  }
  for (std::size_t t = 0; t < Sj; ++t)
    for (std::size_t c = 0; c < D; ++c)
      (*joint)[t * D + c] += gate[c] * sout[t * D + c];
  return true;
}

bool FluxDitNpu::DenoiseBlocks(std::vector<float>* img, std::vector<float>* txt,
                               std::size_t Si, std::size_t St,
                               const FluxModulations& mod,
                               const std::vector<float>& rope_cos,
                               const std::vector<float>& rope_sin,
                               std::size_t n_double, std::size_t n_single,
                               double* npu_ms, std::string* error) {
  for (std::size_t bi = 0; bi < n_double; ++bi) {
    if (!DoubleBlock(img, txt, Si, St, bi, mod.img, mod.txt, rope_cos, rope_sin,
                     npu_ms, error)) {
      return false;
    }
  }
  // Joint [txt; img] stream for single blocks.
  std::vector<float> joint((St + Si) * cfg_.d);
  std::copy(txt->begin(), txt->end(), joint.begin());
  std::copy(img->begin(), img->end(), joint.begin() + St * cfg_.d);
  for (std::size_t bi = 0; bi < n_single; ++bi) {
    if (!SingleBlock(&joint, St + Si, bi, mod.sgl, rope_cos, rope_sin, npu_ms,
                     error)) {
      return false;
    }
  }
  std::copy(joint.begin(), joint.begin() + St * cfg_.d, txt->begin());
  std::copy(joint.begin() + St * cfg_.d, joint.end(), img->begin());
  return true;
}

bool FluxDitNpu::VelocityHead(std::span<const float> img, std::size_t Si,
                              std::span<const float> temb,
                              std::vector<float>* v, double* npu_ms,
                              std::string* error) {
  const std::size_t D = cfg_.d;
  // AdaLayerNormContinuous: SiLU(temb) -> linear -> chunk(shift, scale).
  std::vector<float> act(temb.begin(), temb.end());
  SiluInPlace(act);
  std::vector<float> nout_mod;
  if (!lin_.Run("norm_out.linear.weight", "", act, 1, 2 * D, D, &nout_mod,
                npu_ms, error)) {
    return false;
  }
  // AdaLayerNormContinuous: chunk is (scale, shift), NOT (shift, scale).
  std::vector<float> n_img;
  LayerNormRows(img.data(), Si, D, cfg_.eps, &n_img);
  for (std::size_t t = 0; t < Si; ++t)
    for (std::size_t c = 0; c < D; ++c)
      n_img[t * D + c] =
          n_img[t * D + c] * (1.0F + nout_mod[c]) + nout_mod[D + c];
  return lin_.Run("proj_out.weight", "", n_img, Si, 128, D, v, npu_ms, error);
}

bool FluxDitNpu::FullStep(std::vector<float>* z, std::size_t Si,
                          std::span<const float> txt, std::size_t St,
                          float sigma, float dt, std::size_t n_double,
                          std::size_t n_single, double* npu_ms,
                          std::string* error) {
  std::vector<float> temb;
  if (!TimestepEmbed(sigma, &temb, npu_ms, error))
    return false;
  FluxModulations mod;
  if (!Modulations(temb, &mod, npu_ms, error))
    return false;
  std::vector<float> rope_cos, rope_sin;
  BuildRopeTables(St, Si, &rope_cos, &rope_sin);
  std::vector<float> img;
  if (!XEmbed(*z, Si, &img, npu_ms, error))
    return false;
  if (const char* dp0 = std::getenv("NPU_DUMP_MID")) {
    const std::string pre0 = dp0;
    std::ofstream oi(pre0 + "_in_img.f32", std::ios::binary);
    if (oi.good()) {
      oi.write(reinterpret_cast<const char*>(img.data()),
               static_cast<std::streamsize>(img.size() * sizeof(float)));
    }
    std::ofstream ot(pre0 + "_in_txt.f32", std::ios::binary);
    std::vector<float> txt_in(txt.begin(), txt.end());
    if (ot.good()) {
      ot.write(reinterpret_cast<const char*>(txt_in.data()),
               static_cast<std::streamsize>(txt_in.size() * sizeof(float)));
    }
  }
  std::vector<float> txt_w(txt.begin(), txt.end());
  if (!DenoiseBlocks(&img, &txt_w, Si, St, mod, rope_cos, rope_sin, n_double,
                     n_single, npu_ms, error)) {
    return false;
  }
  std::vector<float> v;
  if (!VelocityHead(img, Si, temb, &v, npu_ms, error))
    return false;
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
    dump("mod_img", mod.img);
    dump("rope_cos", rope_cos);
    dump("img", img);
    dump("txt", txt_w);
    dump("v", v);
  }
  for (std::size_t i = 0; i < Si * 128; ++i)
    (*z)[i] -= dt * v[i];
  return true;
}

}  // namespace mbufo::core::npu::dit
