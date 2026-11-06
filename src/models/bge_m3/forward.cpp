// bge-m3 CPU forward implementation (see forward.hpp).
#include "src/models/bge_m3/forward.hpp"

#include <algorithm>
#include <cmath>

#include "src/core/compute/gelu/gelu.hpp"
#include "src/core/compute/layernorm/layernorm.hpp"

namespace gufo::models::bge_m3 {
namespace {

void UpcastInto(const BgeM3TensorRef& ref, std::vector<float>& out,
                std::size_t row_elems) {
  // Geometry was validated at load; failure here is unreachable. A refused
  // upcast yields an empty vector, which fails loudly at the first shape
  // check downstream (never silent wrong math).
  (void)UpcastTensorToF32(ref, out, row_elems, nullptr);
}

// out[t, m] = bias[m] + sum_k x[t, k] * w[m, k]; w row-major [m, k].
void GemmBias(const float* x, const float* w, const float* b, std::size_t t,
              std::size_t m, std::size_t k, float* out) noexcept {
#pragma omp parallel for schedule(static)
  for (std::size_t row = 0; row < t; ++row) {
    const float* xr = x + row * k;
    float* orow = out + row * m;
    for (std::size_t j = 0; j < m; ++j) {
      const float* wr = w + j * k;
      double acc = (b != nullptr) ? static_cast<double>(b[j]) : 0.0;
      for (std::size_t c = 0; c < k; ++c) {
        acc += static_cast<double>(xr[c]) * static_cast<double>(wr[c]);
      }
      orow[j] = static_cast<float>(acc);
    }
  }
}

}  // namespace

BgeM3F32Weights UpcastAll(const BgeM3Weights& weights) {
  BgeM3F32Weights out;
  out.config = weights.config;
  const std::size_t hidden = weights.config.hidden_size;
  const std::size_t inter = weights.config.intermediate_size;
  UpcastInto(weights.token_embd, out.token_embd, hidden);
  UpcastInto(weights.position_embd, out.position_embd, hidden);
  out.position_rows = hidden == 0 ? 0 : out.position_embd.size() / hidden;
  UpcastInto(weights.token_types, out.token_types, hidden);
  UpcastInto(weights.token_embd_norm_weight, out.emb_norm_w, hidden);
  UpcastInto(weights.token_embd_norm_bias, out.emb_norm_b, hidden);
  out.layers.resize(weights.layers.size());
  for (std::size_t i = 0; i < weights.layers.size(); ++i) {
    const auto& src = weights.layers[i];
    auto& dst = out.layers[i];
    UpcastInto(src.attn_q_weight, dst.q_w, hidden);
    UpcastInto(src.attn_q_bias, dst.q_b, hidden);
    UpcastInto(src.attn_k_weight, dst.k_w, hidden);
    UpcastInto(src.attn_k_bias, dst.k_b, hidden);
    UpcastInto(src.attn_v_weight, dst.v_w, hidden);
    UpcastInto(src.attn_v_bias, dst.v_b, hidden);
    UpcastInto(src.attn_output_weight, dst.out_w, hidden);
    UpcastInto(src.attn_output_bias, dst.out_b, hidden);
    UpcastInto(src.attn_output_norm_weight, dst.attn_norm_w, hidden);
    UpcastInto(src.attn_output_norm_bias, dst.attn_norm_b, hidden);
    UpcastInto(src.ffn_up_weight, dst.up_w, hidden);
    UpcastInto(src.ffn_up_bias, dst.up_b, inter);
    UpcastInto(src.ffn_down_weight, dst.down_w, inter);
    UpcastInto(src.ffn_down_bias, dst.down_b, hidden);
    UpcastInto(src.layer_output_norm_weight, dst.out_norm_w, hidden);
    UpcastInto(src.layer_output_norm_bias, dst.out_norm_b, hidden);
  }
  return out;
}

BgeM3Scratch BgeM3Scratch::Create(const BgeM3Config& config,
                                  std::size_t max_seq) {
  BgeM3Scratch scratch;
  scratch.max_seq = max_seq;
  const std::size_t h = config.hidden_size;
  const std::size_t inter = config.intermediate_size;
  scratch.hidden.assign(max_seq * h, 0.0F);
  scratch.normed.assign(max_seq * h, 0.0F);
  scratch.qkv.assign(max_seq * 3 * h, 0.0F);
  scratch.scores.assign(
      static_cast<std::size_t>(config.num_attention_heads) * max_seq * max_seq,
      0.0F);
  scratch.context.assign(max_seq * h, 0.0F);
  scratch.proj.assign(max_seq * h, 0.0F);
  scratch.ffn_inter.assign(max_seq * inter, 0.0F);
  scratch.ffn_out.assign(max_seq * h, 0.0F);
  return scratch;
}

bool CpuForward(const BgeM3F32Weights& weights,
                std::span<const std::uint32_t> ids, BgeM3Scratch& scratch,
                std::span<float> hidden_out) noexcept {
  const std::size_t n = ids.size();
  const std::size_t h = weights.config.hidden_size;
  const std::size_t heads = weights.config.num_attention_heads;
  const std::size_t inter = weights.config.intermediate_size;
  const std::size_t hd = heads == 0 ? 0 : h / heads;
  if (n == 0 || n > scratch.max_seq || h == 0 || heads == 0 || hd == 0) {
    return false;
  }
  if (hidden_out.size() != n * h) return false;
  if (weights.token_embd.size() < weights.config.vocab_size * h) return false;
  if (weights.position_rows < n) return false;
  const float eps = weights.config.layer_norm_eps;

  std::span<float> hidden(scratch.hidden.data(), n * h);
  // Embed: word + position + token_type[row 0], then embedding LayerNorm.
  for (std::size_t t = 0; t < n; ++t) {
    const std::size_t id = ids[t];
    if (id >= weights.config.vocab_size) return false;
    const float* wrow = weights.token_embd.data() + id * h;
    const float* prow = weights.position_embd.data() + t * h;
    const float* trow = weights.token_types.data();  // row 0
    float* hrow = hidden.data() + t * h;
    for (std::size_t d = 0; d < h; ++d) {
      hrow[d] = wrow[d] + prow[d] + trow[d];
    }
  }
  {
    std::span<float> normed(scratch.normed.data(), n * h);
    for (std::size_t t = 0; t < n; ++t) {
      if (!core::ln::LayerNorm(
              std::span<const float>(hidden.data() + t * h, h),
              std::span<const float>(weights.emb_norm_w.data(), h),
              std::span<const float>(weights.emb_norm_b.data(), h), eps,
              std::span<float>(normed.data() + t * h, h))) {
        return false;
      }
    }
    std::copy(normed.begin(), normed.begin() + n * h, hidden.begin());
  }

  const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
  std::span<float> normed(scratch.normed.data(), n * h);
  std::span<float> qkv(scratch.qkv.data(), n * 3 * h);

  for (const auto& layer : weights.layers) {
    // QKV projections from hidden (post-norm BERT order: no pre-norm).
    const float* xq = hidden.data();
    GemmBias(xq, layer.q_w.data(), layer.q_b.data(), n, h, h,
             qkv.data());
    GemmBias(xq, layer.k_w.data(), layer.k_b.data(), n, h, h,
             qkv.data() + n * h);
    GemmBias(xq, layer.v_w.data(), layer.v_b.data(), n, h, h,
             qkv.data() + 2 * n * h);
    const float* q = qkv.data();
    const float* k = qkv.data() + n * h;
    const float* v = qkv.data() + 2 * n * h;

    // Bidirectional full attention, per head.
#pragma omp parallel for schedule(static)
    for (std::size_t hh = 0; hh < heads; ++hh) {
      float* scores =
          scratch.scores.data() + hh * scratch.max_seq * scratch.max_seq;
      float* ctx = scratch.context.data();
      for (std::size_t i = 0; i < n; ++i) {
        const float* qi = q + i * h + hh * hd;
        double row_max = -1e300;
        for (std::size_t j = 0; j < n; ++j) {
          const float* kj = k + j * h + hh * hd;
          double dot = 0.0;
          for (std::size_t d = 0; d < hd; ++d) {
            dot += static_cast<double>(qi[d]) * static_cast<double>(kj[d]);
          }
          const double s = dot * scale;
          scores[i * n + j] = static_cast<float>(s);
          if (s > row_max) row_max = s;
        }
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
          const double e = std::exp(scores[i * n + j] - row_max);
          scores[i * n + j] = static_cast<float>(e);
          sum += e;
        }
        const float inv = static_cast<float>(1.0 / sum);
        for (std::size_t j = 0; j < n; ++j) scores[i * n + j] *= inv;
        float* ci = ctx + i * h + hh * hd;
        for (std::size_t d = 0; d < hd; ++d) ci[d] = 0.0F;
        for (std::size_t j = 0; j < n; ++j) {
          const float w = scores[i * n + j];
          const float* vj = v + j * h + hh * hd;
          for (std::size_t d = 0; d < hd; ++d) {
            ci[d] += w * vj[d];
          }
        }
      }
    }

    // Output projection + residual + post-attention LayerNorm.
    GemmBias(scratch.context.data(), layer.out_w.data(), layer.out_b.data(),
             n, h, h, scratch.proj.data());
    for (std::size_t t = 0; t < n; ++t) {
      float* hrow = hidden.data() + t * h;
      const float* prow = scratch.proj.data() + t * h;
      for (std::size_t d = 0; d < h; ++d) hrow[d] += prow[d];
      if (!core::ln::LayerNorm(std::span<const float>(hrow, h),
                         std::span<const float>(layer.attn_norm_w.data(), h),
                         std::span<const float>(layer.attn_norm_b.data(), h),
                         eps, std::span<float>(normed.data() + t * h, h))) {
        return false;
      }
    }
    // FFN: up + bias -> exact GELU -> down + bias -> residual -> LayerNorm.
    GemmBias(normed.data(), layer.up_w.data(), layer.up_b.data(), n, inter,
             h, scratch.ffn_inter.data());
    if (!core::gelu::ApplyGelu(
            std::span<const float>(scratch.ffn_inter.data(), n * inter),
            std::span<float>(scratch.ffn_inter.data(), n * inter),
            core::gelu::GeluMode::kExact)) {
      return false;
    }
    GemmBias(scratch.ffn_inter.data(), layer.down_w.data(),
             layer.down_b.data(), n, h, inter, scratch.ffn_out.data());
    {
      // BertOutput residual: the FFN output adds to the post-attention
      // LayerNorm state (normed), not to the pre-norm hidden buffer.
      std::span<float> out_rows(scratch.normed.data(), n * h);
      for (std::size_t t = 0; t < n; ++t) {
        float* hrow = scratch.normed.data() + t * h;
        const float* frow = scratch.ffn_out.data() + t * h;
        for (std::size_t d = 0; d < h; ++d) hrow[d] += frow[d];
        if (!core::ln::LayerNorm(std::span<const float>(hrow, h),
                           std::span<const float>(layer.out_norm_w.data(), h),
                           std::span<const float>(layer.out_norm_b.data(), h),
                           eps,
                           std::span<float>(out_rows.data() + t * h, h))) {
          return false;
        }
      }
      std::copy(out_rows.begin(), out_rows.begin() + n * h, hidden.begin());
    }
  }
  std::copy(hidden.begin(), hidden.begin() + n * h, hidden_out.begin());
  return true;
}

}  // namespace gufo::models::bge_m3
