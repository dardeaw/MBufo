#include "src/core/compute/npu/npu_embgemma2_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <utility>

#include "src/core/compute/npu/npu_pooling.hpp"

namespace mbufo::core::npu {
namespace {

bool ReadTensorBf16(const core::GgufReader& reader, const std::string& name,
                    std::size_t expected_elems, std::vector<std::uint16_t>* out,
                    std::string* error) {
  const auto* t = reader.FindTensor(name);
  if (t == nullptr || t->ElementCount() != expected_elems) {
    if (error != nullptr) *error = "missing or mis-shaped tensor: " + name;
    return false;
  }
  out->resize(expected_elems);
  if (t->type == core::GgmlType::kBF16) {
    std::memcpy(out->data(), t->data, expected_elems * sizeof(std::uint16_t));
    return true;
  }
  if (t->type == core::GgmlType::kF32) {
    const auto* src = static_cast<const float*>(t->data);
    for (std::size_t i = 0; i < expected_elems; ++i) {
      (*out)[i] = FloatToBf16Rne(src[i]);
    }
    return true;
  }
  if (error != nullptr) *error = "unsupported tensor type for BF16: " + name;
  return false;
}

bool ReadTensorF32(const core::GgufReader& reader, const std::string& name,
                   std::size_t expected_elems, std::vector<float>* out,
                   std::string* error) {
  const auto* t = reader.FindTensor(name);
  if (t == nullptr || t->ElementCount() != expected_elems) {
    if (error != nullptr) *error = "missing or mis-shaped tensor: " + name;
    return false;
  }
  out->resize(expected_elems);
  if (t->type == core::GgmlType::kF32) {
    std::memcpy(out->data(), t->data, expected_elems * sizeof(float));
    return true;
  }
  if (t->type == core::GgmlType::kBF16) {
    const auto* src = static_cast<const std::uint16_t*>(t->data);
    for (std::size_t i = 0; i < expected_elems; ++i) {
      (*out)[i] = Bf16BitsToFloat(src[i]);
    }
    return true;
  }
  if (error != nullptr) *error = "unsupported tensor type for F32: " + name;
  return false;
}

// Transpose [rows=out_dim, cols=in_dim] -> [in_dim, out_dim].
void TransposeBf16(std::span<const std::uint16_t> src, std::size_t rows,
                   std::size_t cols, std::vector<std::uint16_t>* dst) {
  dst->resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      (*dst)[c * rows + r] = src[r * cols + c];
    }
  }
}

std::unique_ptr<hrx::ext::bo> UploadBf16Bo(
    HrxSession& sess, std::span<const std::uint16_t> data, std::string* error) {
  auto bo = sess.AllocBo(data.size() * sizeof(std::uint16_t), error);
  if (!bo) return nullptr;
  std::memcpy(bo->map<void*>(), data.data(),
              data.size() * sizeof(std::uint16_t));
  bo->flush();
  return bo;
}

std::unique_ptr<hrx::ext::bo> MakeRopeLutBo(HrxSession& sess, std::size_t s,
                                            std::size_t d, float theta,
                                            std::string* error) {
  const std::size_t half = d / 2;
  std::vector<std::uint16_t> lut(s * d);
  for (std::size_t pos = 0; pos < s; ++pos) {
    for (std::size_t i = 0; i < half; ++i) {
      const double freq =
          std::pow(static_cast<double>(theta),
                   -2.0 * static_cast<double>(i) / static_cast<double>(d));
      const double ang = static_cast<double>(pos) * freq;
      lut[pos * d + 2 * i] = FloatToBf16Rne(static_cast<float>(std::cos(ang)));
      lut[pos * d + 2 * i + 1] =
          FloatToBf16Rne(static_cast<float>(std::sin(ang)));
    }
  }
  return UploadBf16Bo(sess, lut, error);
}

inline float GeluTanh(float x) noexcept {
  const float u = 0.7978845608028654F * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + std::tanh(u));
}

void RmsNormWeighted(std::span<const float> x, std::span<const float> w,
                     std::size_t rows, std::size_t cols, float pre_scale,
                     float eps, std::vector<float>* out) {
  out->resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x.data() + r * cols;
    float* yr = out->data() + r * cols;
    double ss = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      const double v = static_cast<double>(xr[c]) * pre_scale;
      ss += v * v;
    }
    const float inv = static_cast<float>(
        1.0 / std::sqrt(ss / static_cast<double>(cols) + eps));
    for (std::size_t c = 0; c < cols; ++c) {
      yr[c] = xr[c] * pre_scale * inv * w[c];
    }
  }
}

void ApplyQkvPostAndBidirectionalAttn(
    const float* q_raw, const float* k_raw, const float* v_raw,
    std::span<const float> q_norm_w, std::span<const float> k_norm_w,
    std::size_t n_valid, std::size_t n_heads, std::size_t n_kv_heads,
    std::size_t head_dim, float rope_theta, float rms_eps, bool is_full,
    std::size_t sliding_window, std::vector<float>* ctx_out) {
  const std::size_t qd = n_heads * head_dim;
  const std::size_t kd = n_kv_heads * head_dim;
  const std::size_t half = head_dim / 2u;

  std::vector<float> q_post(n_valid * qd);
  std::vector<float> k_post(n_valid * kd);
  std::vector<float> v_post(n_valid * kd);

  std::vector<float> cos_tab(n_valid * half);
  std::vector<float> sin_tab(n_valid * half);
  for (std::size_t pos = 0; pos < n_valid; ++pos) {
    for (std::size_t i = 0; i < half; ++i) {
      const double freq = std::pow(
          static_cast<double>(rope_theta),
          -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
      const double ang = static_cast<double>(pos) * freq;
      cos_tab[pos * half + i] = static_cast<float>(std::cos(ang));
      sin_tab[pos * half + i] = static_cast<float>(std::sin(ang));
    }
  }

  for (std::size_t t = 0; t < n_valid; ++t) {
    const float* cs = cos_tab.data() + t * half;
    const float* sn = sin_tab.data() + t * half;
    for (std::size_t h = 0; h < n_heads; ++h) {
      const float* qh = q_raw + t * qd + h * head_dim;
      float* qdst = q_post.data() + t * qd + h * head_dim;
      double ss = 0.0;
      for (std::size_t i = 0; i < head_dim; ++i) {
        ss += static_cast<double>(qh[i]) * static_cast<double>(qh[i]);
      }
      const float inv = static_cast<float>(
          1.0 / std::sqrt(ss / static_cast<double>(head_dim) + rms_eps));
      for (std::size_t i = 0; i < half; ++i) {
        const float x0 = qh[i] * inv * q_norm_w[i];
        const float x1 = qh[half + i] * inv * q_norm_w[half + i];
        qdst[i] = x0 * cs[i] - x1 * sn[i];
        qdst[half + i] = x1 * cs[i] + x0 * sn[i];
      }
    }
    for (std::size_t h = 0; h < n_kv_heads; ++h) {
      const float* kh = k_raw + t * kd + h * head_dim;
      const float* vh = v_raw + t * kd + h * head_dim;
      float* kdst = k_post.data() + t * kd + h * head_dim;
      float* vdst = v_post.data() + t * kd + h * head_dim;
      double ssk = 0.0, ssv = 0.0;
      for (std::size_t i = 0; i < head_dim; ++i) {
        ssk += static_cast<double>(kh[i]) * static_cast<double>(kh[i]);
        ssv += static_cast<double>(vh[i]) * static_cast<double>(vh[i]);
      }
      const float inv_k = static_cast<float>(
          1.0 / std::sqrt(ssk / static_cast<double>(head_dim) + rms_eps));
      const float inv_v = static_cast<float>(
          1.0 / std::sqrt(ssv / static_cast<double>(head_dim) + rms_eps));
      for (std::size_t i = 0; i < half; ++i) {
        const float x0 = kh[i] * inv_k * k_norm_w[i];
        const float x1 = kh[half + i] * inv_k * k_norm_w[half + i];
        kdst[i] = x0 * cs[i] - x1 * sn[i];
        kdst[half + i] = x1 * cs[i] + x0 * sn[i];
      }
      for (std::size_t i = 0; i < head_dim; ++i) {
        vdst[i] = vh[i] * inv_v;
      }
    }
  }

  ctx_out->assign(n_valid * qd, 0.0F);
  const std::size_t group = n_heads / n_kv_heads;
  std::vector<float> scores(n_valid);
  for (std::size_t h = 0; h < n_heads; ++h) {
    const std::size_t kvh = h / group;
    for (std::size_t tq = 0; tq < n_valid; ++tq) {
      const float* qvec = q_post.data() + tq * qd + h * head_dim;
      float max_s = -1.0e30F;
      for (std::size_t tk = 0; tk < n_valid; ++tk) {
        const std::size_t dist = (tq > tk) ? (tq - tk) : (tk - tq);
        if (!is_full && dist > sliding_window) {
          scores[tk] = -1.0e30F;
          continue;
        }
        const float* kvec = k_post.data() + tk * kd + kvh * head_dim;
        double dot = 0.0;
        for (std::size_t i = 0; i < head_dim; ++i) {
          dot += static_cast<double>(qvec[i]) * static_cast<double>(kvec[i]);
        }
        const float s = static_cast<float>(dot);
        scores[tk] = s;
        if (s > max_s) max_s = s;
      }
      double sum_exp = 0.0;
      for (std::size_t tk = 0; tk < n_valid; ++tk) {
        const float e =
            (scores[tk] <= -1.0e20F) ? 0.0F : std::exp(scores[tk] - max_s);
        scores[tk] = e;
        sum_exp += static_cast<double>(e);
      }
      const float inv_sum =
          (sum_exp > 0.0) ? static_cast<float>(1.0 / sum_exp) : 0.0F;
      float* cdst = ctx_out->data() + tq * qd + h * head_dim;
      for (std::size_t tk = 0; tk < n_valid; ++tk) {
        const float p = scores[tk] * inv_sum;
        if (p == 0.0F) continue;
        const float* vvec = v_post.data() + tk * kd + kvh * head_dim;
        for (std::size_t i = 0; i < head_dim; ++i) {
          cdst[i] += p * vvec[i];
        }
      }
    }
  }
}

void MatMulF32TransposedB(std::span<const float> a, std::span<const float> bt,
                          std::size_t m, std::size_t n, std::size_t k,
                          std::vector<float>* out) {
  out->assign(m * n, 0.0F);
#pragma omp parallel for schedule(static) if (m * n >= 4096)
  for (std::size_t r = 0; r < m; ++r) {
    const float* ar = a.data() + r * k;
    float* orow = out->data() + r * n;
    for (std::size_t c = 0; c < n; ++c) {
      const float* bcol = bt.data() + c * k;
      double acc = 0.0;
      for (std::size_t i = 0; i < k; ++i) {
        acc += static_cast<double>(ar[i]) * static_cast<double>(bcol[i]);
      }
      orow[c] = static_cast<float>(acc);
    }
  }
}

}  // namespace

bool NpuEmbGemma2Runtime::Init(const core::GgufReader& reader,
                               const models::embeddinggemma2::Config& cfg,
                               const std::string& kernel_dir,
                               std::string* error) {
  resident_ = false;
  quant_mode_ = NpuQuantMode::kBf16;
  total_quant_bytes_ = 0;
  cfg_ = cfg;
  kernel_dir_ = kernel_dir;

  if (cfg.layers != 24 || cfg.hidden != 512 || cfg.ffn != 2048 ||
      cfg.heads != 4 || cfg.head_dim_local != 256 || cfg.head_dim_full != 512 ||
      cfg.kv_local != 2 || cfg.kv_full != 1 || cfg.ple_dim != 512 ||
      cfg.layer_is_full.size() != 24) {
    if (error != nullptr) {
      *error = "NpuEmbGemma2Runtime: unsupported model geometry";
    }
    return false;
  }

  const auto* emb = reader.FindTensor("embed_tokens.weight");
  if (emb == nullptr || emb->type != core::GgmlType::kBF16 ||
      emb->ElementCount() % cfg.hidden != 0) {
    if (error != nullptr) *error = "embed_tokens.weight missing or invalid";
    return false;
  }
  embed_table_bf16_ = static_cast<const std::uint16_t*>(emb->data);
  vocab_size_ = emb->ElementCount() / cfg.hidden;

  session_ = HrxSession::Open(0, error);
  if (!session_) return false;

  const char* kNames[] = {
      "qkv_pre_local", "qnr_local", "attn_local", "o_post_local",
      "qkv_pre_full",  "qnr_full",  "attn_full",  "o_post_full",
      "ffn_pre_mm",    "ffn_geglu",
  };
  for (const char* k : kNames) {
    const std::string xcl =
        (std::filesystem::path(kernel_dir) / (std::string(k) + ".xclbin")).string();
    const std::string inst =
        (std::filesystem::path(kernel_dir) / (std::string(k) + ".insts.bin")).string();
    if (!session_->LoadIronKernel(k, xcl, inst, error)) return false;
  }

  std::vector<float> ple_proj_all;
  if (!ReadTensorF32(reader, "ple.per_layer_model_projection.weight",
                     24 * cfg.ple_dim * cfg.hidden, &ple_proj_all, error) ||
      !ReadTensorF32(reader, "ple.per_layer_projection_norm.weight", cfg.ple_dim,
                     &ple_norm_f32_, error) ||
      !ReadTensorF32(reader, "norm.weight", cfg.hidden, &final_norm_f32_, error) ||
      !ReadTensorF32(reader, "embedding_projection.weight",
                     cfg.out_dim * cfg.hidden, &out_proj_f32_, error)) {
    return false;
  }

  layers_.clear();
  layers_.resize(24);
  for (std::size_t li = 0; li < 24; ++li) {
    LayerResident& L = layers_[li];
    const bool full = cfg.layer_is_full[li] != 0;
    L.is_full = full;
    const std::size_t d = full ? cfg.head_dim_full : cfg.head_dim_local;
    const std::size_t kvh = full ? cfg.kv_full : cfg.kv_local;
    const std::size_t qd = cfg.heads * d;
    const std::size_t kd = kvh * d;
    const std::size_t H = cfg.hidden;
    const std::size_t F = cfg.ffn;
    const std::string p = "layers." + std::to_string(li) + ".";

    std::vector<std::uint16_t> tmp, tq, tk, tv, t_out, wqkv(H * (qd + 2 * kd));
    if (!ReadTensorBf16(reader, p + "input_layernorm.weight", H, &tmp, error) ||
        !(L.in_norm = UploadBf16Bo(*session_, tmp, error))) {
      return false;
    }
    if (!ReadTensorBf16(reader, p + "self_attn.q_proj.weight", qd * H, &tmp, error)) {
      return false;
    }
    TransposeBf16(tmp, qd, H, &tq);
    if (!ReadTensorBf16(reader, p + "self_attn.k_proj.weight", kd * H, &tmp, error)) {
      return false;
    }
    TransposeBf16(tmp, kd, H, &tk);
    if (!ReadTensorBf16(reader, p + "self_attn.v_proj.weight", kd * H, &tmp, error)) {
      return false;
    }
    TransposeBf16(tmp, kd, H, &tv);
    const std::size_t qkv_cols = qd + 2 * kd;
    for (std::size_t r = 0; r < H; ++r) {
      std::uint16_t* dst = wqkv.data() + r * qkv_cols;
      std::memcpy(dst, tq.data() + r * qd, qd * sizeof(std::uint16_t));
      std::memcpy(dst + qd, tk.data() + r * kd, kd * sizeof(std::uint16_t));
      std::memcpy(dst + qd + kd, tv.data() + r * kd, kd * sizeof(std::uint16_t));
    }
    if (!(L.w_qkv = UploadBf16Bo(*session_, wqkv, error))) return false;

    if (!ReadTensorBf16(reader, p + "self_attn.q_norm.weight", d, &tmp, error) ||
        !(L.q_norm = UploadBf16Bo(*session_, tmp, error)) ||
        !ReadTensorBf16(reader, p + "self_attn.k_norm.weight", d, &tmp, error) ||
        !(L.k_norm = UploadBf16Bo(*session_, tmp, error))) {
      return false;
    }
    if (!ReadTensorBf16(reader, p + "self_attn.o_proj.weight", H * qd, &tmp, error)) {
      return false;
    }
    TransposeBf16(tmp, H, qd, &t_out);
    if (!(L.w_o = UploadBf16Bo(*session_, t_out, error))) return false;

    if (!ReadTensorBf16(reader, p + "post_attention_layernorm.weight", H, &tmp, error) ||
        !(L.post_attn = UploadBf16Bo(*session_, tmp, error)) ||
        !ReadTensorBf16(reader, p + "pre_feedforward_layernorm.weight", H, &tmp, error) ||
        !(L.pre_ffn = UploadBf16Bo(*session_, tmp, error))) {
      return false;
    }

    if (!ReadTensorBf16(reader, p + "mlp.gate_proj.weight", F * H, &tmp, error)) {
      return false;
    }
    TransposeBf16(tmp, F, H, &t_out);
    if (!(L.w_gate = UploadBf16Bo(*session_, t_out, error))) return false;

    if (!ReadTensorBf16(reader, p + "mlp.up_proj.weight", F * H, &tmp, error)) {
      return false;
    }
    TransposeBf16(tmp, F, H, &t_out);
    if (!(L.w_up = UploadBf16Bo(*session_, t_out, error))) return false;

    if (!ReadTensorBf16(reader, p + "mlp.down_proj.weight", H * F, &tmp, error)) {
      return false;
    }
    TransposeBf16(tmp, H, F, &t_out);
    if (!(L.w_down = UploadBf16Bo(*session_, t_out, error))) return false;

    if (!ReadTensorBf16(reader, p + "post_feedforward_layernorm.weight", H, &tmp, error) ||
        !(L.post_ffn = UploadBf16Bo(*session_, tmp, error))) {
      return false;
    }

    const std::size_t ple_slice = cfg.ple_dim * H;
    L.ple_proj.assign(ple_proj_all.begin() + li * ple_slice,
                      ple_proj_all.begin() + (li + 1) * ple_slice);
    if (!ReadTensorF32(reader, p + "ple_block.per_layer_input_gate.weight",
                       cfg.ple_dim * H, &L.wpgate, error) ||
        !ReadTensorF32(reader, p + "ple_block.per_layer_projection.weight",
                       H * cfg.ple_dim, &L.wpproj, error) ||
        !ReadTensorF32(reader, p + "ple_block.post_per_layer_input_norm.weight",
                       H, &L.ple_post, error)) {
      return false;
    }
    const auto* sc = reader.FindTensor(p + "layer_scalar");
    if (sc == nullptr || sc->type != core::GgmlType::kF32 ||
        sc->ElementCount() != 1) {
      if (error != nullptr) *error = "missing layer_scalar at layer " + std::to_string(li);
      return false;
    }
    L.scalar = *static_cast<const float*>(sc->data);
  }

  lut_local_ = MakeRopeLutBo(*session_, kFixedSeqLen, 256, cfg.rope_local, error);
  lut_full_ = MakeRopeLutBo(*session_, kFixedSeqLen, 512, cfg.rope_full, error);
  mask_local_ = session_->AllocBo(kFixedSeqLen * kFixedSeqLen * 2, error);
  mask_full_ = session_->AllocBo(kFixedSeqLen * kFixedSeqLen * 2, error);
  h_a_ = session_->AllocBo(kFixedSeqLen * 512 * 2, error);
  h_b_ = session_->AllocBo(kFixedSeqLen * 512 * 2, error);
  qkv_raw_l_ = session_->AllocBo(kFixedSeqLen * 2048 * 2, error);
  qkv_raw_f_ = session_->AllocBo(kFixedSeqLen * 3072 * 2, error);
  q_l_ = session_->AllocBo(4 * kFixedSeqLen * 256 * 2, error);
  k_l_ = session_->AllocBo(2 * kFixedSeqLen * 256 * 2, error);
  v_l_ = session_->AllocBo(2 * kFixedSeqLen * 256 * 2, error);
  q_f_ = session_->AllocBo(4 * kFixedSeqLen * 512 * 2, error);
  k_f_ = session_->AllocBo(1 * kFixedSeqLen * 512 * 2, error);
  v_f_ = session_->AllocBo(1 * kFixedSeqLen * 512 * 2, error);
  ctx_l_ = session_->AllocBo(kFixedSeqLen * 1024 * 2, error);
  ctx_f_ = session_->AllocBo(kFixedSeqLen * 2048 * 2, error);
  gate_bo_ = session_->AllocBo(kFixedSeqLen * 2048 * 2, error);
  up_bo_ = session_->AllocBo(kFixedSeqLen * 2048 * 2, error);
  act_bo_ = session_->AllocBo(kFixedSeqLen * 2048 * 2, error);

  if (!lut_local_ || !lut_full_ || !mask_local_ || !mask_full_ || !h_a_ ||
      !h_b_ || !qkv_raw_l_ || !qkv_raw_f_ || !q_l_ || !k_l_ || !v_l_ ||
      !q_f_ || !k_f_ || !v_f_ || !ctx_l_ || !ctx_f_ || !gate_bo_ || !up_bo_ ||
      !act_bo_) {
    return false;
  }

  resident_ = true;
  return true;
}

bool NpuEmbGemma2Runtime::InitQuant(
    const core::GgufReader& reader, const models::embeddinggemma2::Config& cfg,
    NpuQuantMode mode, const std::string& quant_kernel_dir,
    std::string* error) {
  resident_ = false;
  if (mode == NpuQuantMode::kBf16) {
    return Init(reader, cfg, quant_kernel_dir, error);
  }
  quant_mode_ = mode;
  total_quant_bytes_ = 0;
  cfg_ = cfg;
  kernel_dir_ = quant_kernel_dir;

  if (cfg.layers != 24 || cfg.hidden != 512 || cfg.ffn != 2048 ||
      cfg.heads != 4 || cfg.head_dim_local != 256 || cfg.head_dim_full != 512 ||
      cfg.kv_local != 2 || cfg.kv_full != 1 || cfg.ple_dim != 512 ||
      cfg.layer_is_full.size() != 24) {
    if (error != nullptr) {
      *error = "NpuEmbGemma2Runtime::InitQuant: unsupported model geometry";
    }
    return false;
  }

  const auto* emb = reader.FindTensor("embed_tokens.weight");
  if (emb == nullptr || emb->type != core::GgmlType::kBF16 ||
      emb->ElementCount() % cfg.hidden != 0) {
    if (error != nullptr) *error = "embed_tokens.weight missing or invalid";
    return false;
  }
  embed_table_bf16_ = static_cast<const std::uint16_t*>(emb->data);
  vocab_size_ = emb->ElementCount() / cfg.hidden;

  session_.reset();
  quant_engine_ = std::make_unique<NpuQuantGemmEngine>();
  if (!quant_engine_->Init(quant_kernel_dir, error)) return false;

  std::vector<float> ple_proj_all;
  if (!ReadTensorF32(reader, "ple.per_layer_model_projection.weight",
                     24 * cfg.ple_dim * cfg.hidden, &ple_proj_all, error) ||
      !ReadTensorF32(reader, "ple.per_layer_projection_norm.weight", cfg.ple_dim,
                     &ple_norm_f32_, error) ||
      !ReadTensorF32(reader, "norm.weight", cfg.hidden, &final_norm_f32_, error) ||
      !ReadTensorF32(reader, "embedding_projection.weight",
                     cfg.out_dim * cfg.hidden, &out_proj_f32_, error)) {
    return false;
  }

  layers_.clear();
  layers_.resize(24);
  for (std::size_t li = 0; li < 24; ++li) {
    LayerResident& L = layers_[li];
    const bool full = cfg.layer_is_full[li] != 0;
    L.is_full = full;
    const std::size_t d = full ? cfg.head_dim_full : cfg.head_dim_local;
    const std::size_t kvh = full ? cfg.kv_full : cfg.kv_local;
    const std::size_t qd = cfg.heads * d;
    const std::size_t kd = kvh * d;
    const std::size_t H = cfg.hidden;
    const std::size_t F = cfg.ffn;
    const std::string p = "layers." + std::to_string(li) + ".";

    if (!ReadTensorF32(reader, p + "input_layernorm.weight", H, &L.in_norm_f32, error) ||
        !ReadTensorF32(reader, p + "self_attn.q_norm.weight", d, &L.q_norm_f32, error) ||
        !ReadTensorF32(reader, p + "self_attn.k_norm.weight", d, &L.k_norm_f32, error) ||
        !ReadTensorF32(reader, p + "post_attention_layernorm.weight", H, &L.post_attn_f32, error) ||
        !ReadTensorF32(reader, p + "pre_feedforward_layernorm.weight", H, &L.pre_ffn_f32, error) ||
        !ReadTensorF32(reader, p + "post_feedforward_layernorm.weight", H, &L.post_ffn_f32, error)) {
      return false;
    }

    auto pack_gguf_proj = [&](const std::string& name, std::size_t out_dim,
                              std::size_t in_dim, QuantWeightMatrix* qm) -> bool {
      const auto* t = reader.FindTensor(name);
      if (t == nullptr || t->ElementCount() != out_dim * in_dim) {
        if (error != nullptr) *error = "missing or mis-shaped tensor: " + name;
        return false;
      }
      if (t->type == core::GgmlType::kBF16) {
        if (!quant_engine_->PackMatrixBf16NK(
                static_cast<const std::uint16_t*>(t->data), out_dim, in_dim,
                mode, false, qm, error)) {
          return false;
        }
      } else {
        std::vector<float> tmp_f;
        if (!ReadTensorF32(reader, name, out_dim * in_dim, &tmp_f, error)) {
          return false;
        }
        if (!quant_engine_->PackMatrixNK(tmp_f.data(), out_dim, in_dim, mode,
                                         false, qm, error)) {
          return false;
        }
      }
      total_quant_bytes_ += qm->total_packed_bytes;
      return true;
    };

    if (!pack_gguf_proj(p + "self_attn.q_proj.weight", qd, H, &L.q_quant) ||
        !pack_gguf_proj(p + "self_attn.k_proj.weight", kd, H, &L.k_quant) ||
        !pack_gguf_proj(p + "self_attn.v_proj.weight", kd, H, &L.v_quant) ||
        !pack_gguf_proj(p + "self_attn.o_proj.weight", H, qd, &L.o_quant) ||
        !pack_gguf_proj(p + "mlp.gate_proj.weight", F, H, &L.gate_quant) ||
        !pack_gguf_proj(p + "mlp.up_proj.weight", F, H, &L.up_quant) ||
        !pack_gguf_proj(p + "mlp.down_proj.weight", H, F, &L.down_quant) ||
        !pack_gguf_proj(p + "ple_block.per_layer_input_gate.weight", cfg.ple_dim, H, &L.wpgate_quant) ||
        !pack_gguf_proj(p + "ple_block.per_layer_projection.weight", H, cfg.ple_dim, &L.wpproj_quant)) {
      return false;
    }

    const std::size_t ple_slice = cfg.ple_dim * H;
    L.ple_proj.assign(ple_proj_all.begin() + li * ple_slice,
                      ple_proj_all.begin() + (li + 1) * ple_slice);
    if (!quant_engine_->PackMatrixNK(L.ple_proj.data(), cfg.ple_dim, H, mode,
                                     false, &L.ple_proj_quant, error)) {
      return false;
    }
    total_quant_bytes_ += L.ple_proj_quant.total_packed_bytes;

    if (!ReadTensorF32(reader, p + "ple_block.post_per_layer_input_norm.weight",
                       H, &L.ple_post, error)) {
      return false;
    }
    const auto* sc = reader.FindTensor(p + "layer_scalar");
    if (sc == nullptr || sc->type != core::GgmlType::kF32 ||
        sc->ElementCount() != 1) {
      if (error != nullptr) *error = "missing layer_scalar at layer " + std::to_string(li);
      return false;
    }
    L.scalar = *static_cast<const float*>(sc->data);
  }

  resident_ = true;
  return true;
}

bool NpuEmbGemma2Runtime::ForwardQuant(std::span<const std::uint32_t> ids,
                                       std::vector<float>* out_embedding,
                                       double* prefill_ms, std::string* error) {
  if (!resident_ || !quant_engine_ || !quant_engine_->ready()) {
    if (error != nullptr) *error = "NpuEmbGemma2Runtime: quant engine not ready";
    return false;
  }
  const std::size_t n_valid = ids.size();
  if (n_valid == 0 || n_valid > 512) {
    if (error != nullptr) *error = "NpuEmbGemma2Runtime: seq_len out of range [1, 512]";
    return false;
  }
  for (std::uint32_t id : ids) {
    if (id >= vocab_size_) {
      if (error != nullptr) *error = "NpuEmbGemma2Runtime: token id out of vocab";
      return false;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  const std::size_t H = cfg_.hidden;
  const std::size_t F = cfg_.ffn;
  std::vector<float> emb_f32(n_valid * H);
  for (std::size_t t = 0; t < n_valid; ++t) {
    const std::uint16_t* row =
        embed_table_bf16_ + static_cast<std::size_t>(ids[t]) * H;
    for (std::size_t c = 0; c < H; ++c) {
      emb_f32[t * H + c] = Bf16BitsToFloat(row[c]) * cfg_.embed_scale;
    }
  }

  std::vector<float> h_cur = emb_f32;
  std::vector<float> x_norm, q_raw, k_raw, v_raw, ctx, o_raw, o_norm, h_mid;
  std::vector<float> ffn_in, gate_buf(n_valid * F), up_buf(n_valid * F),
      act_buf(n_valid * F), down_raw(n_valid * H), down_norm;
  std::vector<float> g_buf(n_valid * cfg_.ple_dim), pl_buf,
      ple_gate(n_valid * cfg_.ple_dim), act_ple(n_valid * cfg_.ple_dim),
      y_ple(n_valid * H), y_ple_norm;
  const float ple_pre = 1.0F / std::sqrt(static_cast<float>(H));

  for (std::size_t li = 0; li < 24; ++li) {
    const LayerResident& L = layers_[li];
    const bool full = L.is_full;
    const std::size_t d = full ? cfg_.head_dim_full : cfg_.head_dim_local;
    const std::size_t kvh = full ? cfg_.kv_full : cfg_.kv_local;
    const std::size_t qd = cfg_.heads * d;
    const std::size_t kd = kvh * d;
    const float theta = full ? cfg_.rope_full : cfg_.rope_local;

    RmsNormWeighted(h_cur, L.in_norm_f32, n_valid, H, 1.0F, cfg_.rms_eps,
                    &x_norm);
    q_raw.resize(n_valid * qd);
    k_raw.resize(n_valid * kd);
    v_raw.resize(n_valid * kd);
    if (!quant_engine_->Gemm(x_norm.data(), n_valid, L.q_quant, q_raw.data(),
                             nullptr, nullptr, error) ||
        !quant_engine_->Gemm(x_norm.data(), n_valid, L.k_quant, k_raw.data(),
                             nullptr, nullptr, error) ||
        !quant_engine_->Gemm(x_norm.data(), n_valid, L.v_quant, v_raw.data(),
                             nullptr, nullptr, error)) {
      return false;
    }

    ApplyQkvPostAndBidirectionalAttn(
        q_raw.data(), k_raw.data(), v_raw.data(), L.q_norm_f32, L.k_norm_f32,
        n_valid, cfg_.heads, kvh, d, theta, cfg_.rms_eps, full,
        cfg_.sliding_window, &ctx);

    o_raw.resize(n_valid * H);
    if (!quant_engine_->Gemm(ctx.data(), n_valid, L.o_quant, o_raw.data(),
                             nullptr, nullptr, error)) {
      return false;
    }
    RmsNormWeighted(o_raw, L.post_attn_f32, n_valid, H, 1.0F, cfg_.rms_eps,
                    &o_norm);
    h_mid.resize(n_valid * H);
    for (std::size_t i = 0; i < n_valid * H; ++i) {
      h_mid[i] = h_cur[i] + o_norm[i];
    }

    RmsNormWeighted(h_mid, L.pre_ffn_f32, n_valid, H, 1.0F, cfg_.rms_eps,
                    &ffn_in);
    if (!quant_engine_->Gemm(ffn_in.data(), n_valid, L.gate_quant,
                             gate_buf.data(), nullptr, nullptr, error) ||
        !quant_engine_->Gemm(ffn_in.data(), n_valid, L.up_quant, up_buf.data(),
                             nullptr, nullptr, error)) {
      return false;
    }
    for (std::size_t i = 0; i < n_valid * F; ++i) {
      act_buf[i] = GeluTanh(gate_buf[i]) * up_buf[i];
    }
    if (!quant_engine_->Gemm(act_buf.data(), n_valid, L.down_quant,
                             down_raw.data(), nullptr, nullptr, error)) {
      return false;
    }
    RmsNormWeighted(down_raw, L.post_ffn_f32, n_valid, H, 1.0F, cfg_.rms_eps,
                    &down_norm);
    for (std::size_t i = 0; i < n_valid * H; ++i) {
      h_cur[i] = h_mid[i] + down_norm[i];
    }

    // Per-layer input block on NPU2 + layer_scalar.
    if (!quant_engine_->Gemm(emb_f32.data(), n_valid, L.ple_proj_quant,
                             g_buf.data(), nullptr, nullptr, error)) {
      return false;
    }
    RmsNormWeighted(g_buf, ple_norm_f32_, n_valid, cfg_.ple_dim, ple_pre,
                    cfg_.rms_eps, &pl_buf);
    if (!quant_engine_->Gemm(h_cur.data(), n_valid, L.wpgate_quant,
                             ple_gate.data(), nullptr, nullptr, error)) {
      return false;
    }
    for (std::size_t i = 0; i < n_valid * cfg_.ple_dim; ++i) {
      act_ple[i] = GeluTanh(ple_gate[i]) * pl_buf[i];
    }
    if (!quant_engine_->Gemm(act_ple.data(), n_valid, L.wpproj_quant,
                             y_ple.data(), nullptr, nullptr, error)) {
      return false;
    }
    RmsNormWeighted(y_ple, L.ple_post, n_valid, H, 1.0F, cfg_.rms_eps,
                    &y_ple_norm);
    for (std::size_t i = 0; i < n_valid * H; ++i) {
      h_cur[i] = (h_cur[i] + y_ple_norm[i]) * L.scalar;
    }
  }

  double pre_norm = 0.0;
  if (!NpuEmbGemma2PostProcess(h_cur, n_valid, H, final_norm_f32_, cfg_.rms_eps,
                               out_proj_f32_, cfg_.out_dim, *out_embedding,
                               &pre_norm)) {
    if (error != nullptr) *error = "NpuEmbGemma2Runtime: post-processing failed";
    return false;
  }
  if (prefill_ms != nullptr) {
    *prefill_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  }
  return true;
}

bool NpuEmbGemma2Runtime::Forward(std::span<const std::uint32_t> ids,
                                  std::vector<float>* out_embedding,
                                  double* prefill_ms, std::string* error) {
  if (quant_mode_ != NpuQuantMode::kBf16) {
    return ForwardQuant(ids, out_embedding, prefill_ms, error);
  }
  if (!resident_ || !session_) {
    if (error != nullptr) *error = "NpuEmbGemma2Runtime: not initialized";
    return false;
  }
  const std::size_t n_valid = ids.size();
  if (n_valid == 0 || n_valid > kFixedSeqLen) {
    if (error != nullptr) {
      *error = "NpuEmbGemma2Runtime: seq_len out of range [1, 128]";
    }
    return false;
  }
  for (std::uint32_t id : ids) {
    if (id >= vocab_size_) {
      if (error != nullptr) *error = "NpuEmbGemma2Runtime: token id out of vocab";
      return false;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  const std::size_t H = 512;
  const std::size_t S = kFixedSeqLen;

  auto* h_ptr = h_a_->map<std::uint16_t*>();
  std::memset(h_ptr, 0, S * H * sizeof(std::uint16_t));
  std::vector<float> emb_f32(n_valid * H);
  for (std::size_t t = 0; t < n_valid; ++t) {
    const std::uint16_t* row =
        embed_table_bf16_ + static_cast<std::size_t>(ids[t]) * H;
    for (std::size_t c = 0; c < H; ++c) {
      const float v = Bf16BitsToFloat(row[c]) * cfg_.embed_scale;
      emb_f32[t * H + c] = v;
      h_ptr[t * H + c] = FloatToBf16Rne(v);
    }
  }
  h_a_->flush();

  const std::uint16_t bf_zero = FloatToBf16Rne(0.0F);
  const std::uint16_t bf_mask = FloatToBf16Rne(-1.0e4F);
  auto* ml = mask_local_->map<std::uint16_t*>();
  auto* mf = mask_full_->map<std::uint16_t*>();
  for (std::size_t i = 0; i < S; ++i) {
    for (std::size_t j = 0; j < S; ++j) {
      const bool valid = (i < n_valid && j < n_valid);
      mf[i * S + j] = valid ? bf_zero : bf_mask;
      const std::size_t dist = (i > j) ? (i - j) : (j - i);
      ml[i * S + j] =
          (valid && dist <= cfg_.sliding_window) ? bf_zero : bf_mask;
    }
  }
  mask_local_->flush();
  mask_full_->flush();

  const float ple_pre = 1.0F / std::sqrt(static_cast<float>(H));
  std::vector<float> g_buf, pl_buf, ple_gate, act_ple, y_ple, y_ple_norm;
  std::vector<float> h_final_f32(n_valid * H);

  for (std::size_t li = 0; li < 24; ++li) {
    const LayerResident& L = layers_[li];
    std::vector<std::pair<std::string, std::vector<const hrx::bo*>>> steps;
    steps.reserve(8);
    if (L.is_full) {
      steps.emplace_back("qkv_pre_full",
                         std::vector<const hrx::bo*>{h_a_.get(), L.in_norm.get(),
                                                     L.w_qkv.get(), qkv_raw_f_.get()});
      steps.emplace_back("qnr_full",
                         std::vector<const hrx::bo*>{
                             qkv_raw_f_.get(), L.q_norm.get(), L.k_norm.get(),
                             lut_full_.get(), q_f_.get(), k_f_.get(), v_f_.get()});
      steps.emplace_back("attn_full",
                         std::vector<const hrx::bo*>{q_f_.get(), k_f_.get(), v_f_.get(),
                                                     mask_full_.get(), ctx_f_.get()});
      steps.emplace_back("o_post_full",
                         std::vector<const hrx::bo*>{ctx_f_.get(), L.w_o.get(),
                                                     h_a_.get(), L.post_attn.get(),
                                                     h_b_.get()});
    } else {
      steps.emplace_back("qkv_pre_local",
                         std::vector<const hrx::bo*>{h_a_.get(), L.in_norm.get(),
                                                     L.w_qkv.get(), qkv_raw_l_.get()});
      steps.emplace_back("qnr_local",
                         std::vector<const hrx::bo*>{
                             qkv_raw_l_.get(), L.q_norm.get(), L.k_norm.get(),
                             lut_local_.get(), q_l_.get(), k_l_.get(), v_l_.get()});
      steps.emplace_back("attn_local",
                         std::vector<const hrx::bo*>{q_l_.get(), k_l_.get(), v_l_.get(),
                                                     mask_local_.get(), ctx_l_.get()});
      steps.emplace_back("o_post_local",
                         std::vector<const hrx::bo*>{ctx_f_.get(), L.w_o.get(),
                                                     h_a_.get(), L.post_attn.get(),
                                                     h_b_.get()});
      steps.back().second[0] = ctx_l_.get();
    }
    steps.emplace_back("ffn_pre_mm",
                       std::vector<const hrx::bo*>{h_b_.get(), L.pre_ffn.get(),
                                                   L.w_gate.get(), gate_bo_.get()});
    steps.emplace_back("ffn_pre_mm",
                       std::vector<const hrx::bo*>{h_b_.get(), L.pre_ffn.get(),
                                                   L.w_up.get(), up_bo_.get()});
    steps.emplace_back("ffn_geglu",
                       std::vector<const hrx::bo*>{gate_bo_.get(), up_bo_.get(),
                                                   act_bo_.get()});
    steps.emplace_back("o_post_full",
                       std::vector<const hrx::bo*>{act_bo_.get(), L.w_down.get(),
                                                   h_b_.get(), L.post_ffn.get(),
                                                   h_a_.get()});

    if (!session_->RunKernelChain(steps, error)) return false;

    h_a_->invalidate();
    const auto* out_bf16 = h_a_->map<const std::uint16_t*>();
    for (std::size_t i = 0; i < n_valid * H; ++i) {
      h_final_f32[i] = Bf16BitsToFloat(out_bf16[i]);
    }

    // Per-layer input block + layer_scalar.
    MatMulF32TransposedB(emb_f32, L.ple_proj, n_valid, cfg_.ple_dim, H, &g_buf);
    RmsNormWeighted(g_buf, ple_norm_f32_, n_valid, cfg_.ple_dim, ple_pre,
                    cfg_.rms_eps, &pl_buf);
    MatMulF32TransposedB(h_final_f32, L.wpgate, n_valid, cfg_.ple_dim, H,
                         &ple_gate);
    act_ple.resize(n_valid * cfg_.ple_dim);
    for (std::size_t i = 0; i < act_ple.size(); ++i) {
      act_ple[i] = GeluTanh(ple_gate[i]) * pl_buf[i];
    }
    MatMulF32TransposedB(act_ple, L.wpproj, n_valid, H, cfg_.ple_dim, &y_ple);
    RmsNormWeighted(y_ple, L.ple_post, n_valid, H, 1.0F, cfg_.rms_eps,
                    &y_ple_norm);
    for (std::size_t i = 0; i < n_valid * H; ++i) {
      h_final_f32[i] = (h_final_f32[i] + y_ple_norm[i]) * L.scalar;
    }

    if (li + 1 < 24) {
      auto* h_next = h_a_->map<std::uint16_t*>();
      std::memset(h_next, 0, S * H * sizeof(std::uint16_t));
      for (std::size_t i = 0; i < n_valid * H; ++i) {
        h_next[i] = FloatToBf16Rne(h_final_f32[i]);
      }
      h_a_->flush();
    }
  }

  double pre_norm = 0.0;
  if (!NpuEmbGemma2PostProcess(h_final_f32, n_valid, H, final_norm_f32_,
                               cfg_.rms_eps, out_proj_f32_, cfg_.out_dim,
                               *out_embedding, &pre_norm)) {
    if (error != nullptr) *error = "NpuEmbGemma2Runtime: post-processing failed";
    return false;
  }
  if (prefill_ms != nullptr) {
    *prefill_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  }
  return true;
}

}  // namespace mbufo::core::npu
