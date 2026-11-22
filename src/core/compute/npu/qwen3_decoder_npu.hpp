#pragma once
// Shared Qwen3-style decoder core for NPU-only prefill + KV-cache decode.
//
// Thinker (ASR), talker (TTS) and code predictor (TTS) are the same Qwen3
// block math with different shapes/prefixes, so they all run through this
// one class. Model adapters (thinker/talker/predictor headers) own their
// embed tables, LM heads, projections and sampling; this file owns layers,
// KV cache, attention, norms and the NPU GEMM calls.
//
// Numerics mirror the parity shell exactly: RMSNorm, per-head QK-norm,
// RoPE at absolute positions, causal softmax, SiLU MLP. The decode path is
// gated by prefill(T)+1step vs prefill(T+1) cosine (>= 0.998), same as P1.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/kv_cache/gqa_kv_cache.hpp"
#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/npu_shard_reader.hpp"

namespace mbufo::core::npu {

struct Qwen3NpuDecoderConfig {
  std::string layers_prefix;  // e.g. "talker.model.layers"
  std::string norm_key;       // e.g. "talker.model.norm.weight"
  std::size_t layers = 28;
  std::size_t hidden = 2048;
  std::size_t inter = 6144;
  std::size_t n_heads = 16;
  std::size_t n_kv_heads = 8;
  std::size_t head_dim = 128;
  float rms_eps = 1.0e-6F;
  float rope_theta = 1000000.0F;
  std::size_t max_seq = 512;
};

namespace qwen3_detail {

inline void RmsNormRow(const float* x, const float* w, std::size_t dim,
                       float eps, float* y) {
  double ss = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    ss += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  }
  const float inv =
      static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(dim) + eps));
  for (std::size_t i = 0; i < dim; ++i) y[i] = x[i] * inv * w[i];
}

inline float SiluLocal(float x) noexcept { return x / (1.0F + std::exp(-x)); }

// Per-head QK-norm then RoPE at absolute position `pos`, in place over
// [n_vec, head_dim]. Mirrors the parity shell's per-position math.
inline void QkNormRope(float* t, const float* norm_w, std::size_t n_vec,
                       std::size_t head_dim, std::size_t pos, float rope_theta,
                       float rms_eps) {
  const std::size_t half = head_dim / 2u;
  for (std::size_t h = 0; h < n_vec; ++h) {
    float* v = t + h * head_dim;
    double ss = 0.0;
    for (std::size_t i = 0; i < head_dim; ++i) {
      ss += static_cast<double>(v[i]) * static_cast<double>(v[i]);
    }
    const float inv = static_cast<float>(
        1.0 / std::sqrt(ss / static_cast<double>(head_dim) + rms_eps));
    for (std::size_t i = 0; i < head_dim; ++i) v[i] = v[i] * inv * norm_w[i];
    for (std::size_t i = 0; i < half; ++i) {
      const double freq = std::pow(static_cast<double>(rope_theta),
                                   -2.0 * static_cast<double>(i) /
                                       static_cast<double>(head_dim));
      const double ang = static_cast<double>(pos) * freq;
      const float cs = static_cast<float>(std::cos(ang));
      const float sn = static_cast<float>(std::sin(ang));
      const float x0 = v[i];
      const float x1 = v[half + i];
      v[i] = x0 * cs - x1 * sn;
      v[half + i] = x1 * cs + x0 * sn;
    }
  }
}

// Single query row q[hd] over K/V [kv_len, hd] (already sliced to one kv
// head). Softmax scale 1/sqrt(hd), matching the parity shell.
inline void AttnOneQuery(const float* q, const float* K, const float* V,
                         std::size_t kv_len, std::size_t head_dim,
                         float* ctx) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> scores(kv_len);
  float max_s = -1.0e30F;
  for (std::size_t k = 0; k < kv_len; ++k) {
    const float* kvec = K + k * head_dim;
    double dot = 0.0;
    for (std::size_t i = 0; i < head_dim; ++i) {
      dot += static_cast<double>(q[i]) * static_cast<double>(kvec[i]);
    }
    const float s = static_cast<float>(dot) * scale;
    scores[k] = s;
    if (s > max_s) max_s = s;
  }
  double sum = 0.0;
  for (std::size_t k = 0; k < kv_len; ++k) {
    const float e = std::exp(scores[k] - max_s);
    scores[k] = e;
    sum += static_cast<double>(e);
  }
  const float inv = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0F;
  for (std::size_t i = 0; i < head_dim; ++i) ctx[i] = 0.0F;
  for (std::size_t k = 0; k < kv_len; ++k) {
    const float p = scores[k] * inv;
    const float* vvec = V + k * head_dim;
    for (std::size_t i = 0; i < head_dim; ++i) ctx[i] += p * vvec[i];
  }
}

}  // namespace qwen3_detail

class Qwen3NpuDecoder {
 public:
  Qwen3NpuDecoder() = default;

  [[nodiscard]] bool Open(const std::string& model_dir,
                          NpuQuantGemmEngine* engine, NpuQuantMode mode,
                          const Qwen3NpuDecoderConfig& config,
                          std::string* error = nullptr) {
    if (engine == nullptr || !engine->ready()) {
      if (error != nullptr) *error = "NPU engine not ready";
      return false;
    }
    if (!NpuShardReader::OpenDir(model_dir, &reader_, error)) return false;
    engine_ = engine;
    mode_ = mode;
    cfg_ = config;
    kv_ = kv::GqaKvCache(kv::GqaKvConfig{cfg_.layers, cfg_.n_kv_heads,
                                        cfg_.head_dim, cfg_.max_seq});
    if (!reader_.ReadF32(cfg_.norm_key, cfg_.hidden, &final_norm_w_, error)) {
      return false;
    }
    pos_ = 0;
    return true;
  }

  /// Prefill `T` hidden rows [T, hidden]; fills KV cache, sets pos=T,
  /// returns final-layer hidden of the last row.
  [[nodiscard]] bool Prefill(std::span<const float> hidden_rows, std::size_t T,
                             std::vector<float>* last_hidden,
                             double* npu_ms = nullptr,
                             std::string* error = nullptr) {
    if (T == 0 || hidden_rows.size() < T * cfg_.hidden) {
      if (error != nullptr) *error = "bad prefill input";
      return false;
    }
    if (T > cfg_.max_seq) {
      if (error != nullptr) *error = "prefill exceeds max_seq";
      return false;
    }
    std::vector<float> h(hidden_rows.begin(),
                         hidden_rows.begin() + T * cfg_.hidden);
    if (!RunLayers(&h, T, 0, true, npu_ms, error)) return false;
    pos_ = T;
    last_hidden->assign(h.end() - cfg_.hidden, h.end());
    return true;
  }

  /// Prefill variant returning the full final-layer sequence [T, hidden]
  /// (text-encoder serving needs every row, not just the last).
  [[nodiscard]] bool PrefillFull(std::span<const float> hidden_rows,
                                 std::size_t T, std::vector<float>* all_hidden,
                                 double* npu_ms = nullptr,
                                 std::string* error = nullptr) {
    if (T == 0 || hidden_rows.size() < T * cfg_.hidden) {
      if (error != nullptr) *error = "bad prefill input";
      return false;
    }
    if (T > cfg_.max_seq) {
      if (error != nullptr) *error = "prefill exceeds max_seq";
      return false;
    }
    all_hidden->assign(hidden_rows.begin(),
                       hidden_rows.begin() + T * cfg_.hidden);
    if (!RunLayers(all_hidden, T, 0, true, npu_ms, error)) return false;
    pos_ = T;
    return true;
  }

  /// Single decode step at absolute position pos().
  [[nodiscard]] bool DecodeStep(std::span<const float> hidden_row,
                                std::vector<float>* next_hidden,
                                double* npu_ms = nullptr,
                                std::string* error = nullptr) {
    if (hidden_row.size() < cfg_.hidden) {
      if (error != nullptr) *error = "bad decode input";
      return false;
    }
    if (pos_ >= cfg_.max_seq) {
      if (error != nullptr) *error = "decode exceeds max_seq";
      return false;
    }
    std::vector<float> h(hidden_row.begin(),
                         hidden_row.begin() + cfg_.hidden);
    if (!RunLayers(&h, 1, pos_, false, npu_ms, error)) return false;
    ++pos_;
    *next_hidden = std::move(h);
    return true;
  }

  /// Final-norm of one hidden row.
  void NormLastHidden(std::span<const float> hidden,
                      std::vector<float>* normed) const {
    normed->resize(cfg_.hidden);
    qwen3_detail::RmsNormRow(hidden.data(), final_norm_w_.data(), cfg_.hidden,
                             cfg_.rms_eps, normed->data());
  }

  /// Full logit row over a caller-owned [vocab, hidden] f32 head table.
  [[nodiscard]] bool HeadLogits(std::span<const float> normed_hidden,
                                std::span<const float> head_table,
                                std::size_t vocab,
                                std::vector<float>* logits, double* npu_ms,
                                std::string* error) {
    if (normed_hidden.size() < cfg_.hidden ||
        head_table.size() < vocab * cfg_.hidden) {
      if (error != nullptr) *error = "bad head input";
      return false;
    }
    QuantWeightMatrix qm;
    if (!engine_->PackMatrixNK(head_table.data(), vocab, cfg_.hidden, mode_,
                               false, &qm, error)) {
      return false;
    }
    logits->resize(vocab);
    if (!engine_->Gemm(normed_hidden.data(), 1, qm, logits->data(), nullptr,
                       npu_ms, error)) {
      return false;
    }
    return true;
  }

  /// Greedy top-1 over a caller-owned head table.
  [[nodiscard]] bool LogitsTop1(std::span<const float> hidden,
                                std::span<const float> head_table,
                                std::size_t vocab, std::uint32_t* top1,
                                double* npu_ms = nullptr,
                                std::string* error = nullptr) {
    std::vector<float> normed;
    NormLastHidden(hidden, &normed);
    std::vector<float> logits;
    if (!HeadLogits(normed, head_table, vocab, &logits, npu_ms, error)) {
      return false;
    }
    *top1 = static_cast<std::uint32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
    return true;
  }

  /// Raw weight read passthrough (embeds, heads, projections).
  [[nodiscard]] bool ReadF32(const std::string& name, std::size_t count,
                             std::vector<float>* out,
                             std::string* error = nullptr) {
    return reader_.ReadF32(name, count, out, error);
  }

  /// Row-slice read passthrough (embed head blocks, per-row gathers).
  [[nodiscard]] bool ReadRowSliceF32(const std::string& name,
                                     std::size_t start_row,
                                     std::size_t num_rows, std::size_t cols,
                                     std::vector<float>* out,
                                     std::string* error = nullptr) {
    return reader_.ReadRowSliceF32(name, start_row, num_rows, cols, out,
                                   error);
  }

  [[nodiscard]] std::size_t pos() const noexcept { return pos_; }
  [[nodiscard]] const Qwen3NpuDecoderConfig& config() const noexcept {
    return cfg_;
  }
  [[nodiscard]] std::size_t hidden_size() const noexcept {
    return cfg_.hidden;
  }

 private:
  [[nodiscard]] bool LinearNpu(const std::string& weight_name,
                               std::span<const float> x, std::size_t M,
                               std::size_t N, std::size_t K,
                               std::vector<float>* y, double* npu_ms,
                               std::string* error) {
    std::vector<float> w;
    if (!reader_.ReadF32(weight_name, N * K, &w, error)) return false;
    QuantWeightMatrix qm;
    if (!engine_->PackMatrixNK(w.data(), N, K, mode_, false, &qm, error)) {
      return false;
    }
    y->resize(M * N);
    if (!engine_->Gemm(x.data(), M, qm, y->data(), nullptr, npu_ms, error)) {
      return false;
    }
    return true;
  }

  [[nodiscard]] bool RunLayers(std::vector<float>* h, std::size_t T,
                              std::size_t pos_base, bool append_kv,
                              double* npu_ms, std::string* error) {
    const std::size_t H = cfg_.hidden;
    const std::size_t F = cfg_.inter;
    const std::size_t nh = cfg_.n_heads;
    const std::size_t nkv = cfg_.n_kv_heads;
    const std::size_t hd = cfg_.head_dim;
    const std::size_t qd = nh * hd;
    const std::size_t kd = nkv * hd;
    const std::size_t group = nh / nkv;

    std::vector<float> x(T * H);
    std::vector<float> q_all(T * qd), k_all(T * kd), v_all(T * kd);
    std::vector<float> qn_w, kn_w, in_w, post_w;
    std::vector<float> attn_out, gate, up, down, tmp;
    std::vector<float> kh_slice, vh_slice;
    // Per-layer rope'd K/V retained for cache commit after the layer loop.
    std::vector<std::vector<float>> pre_k(cfg_.layers), pre_v(cfg_.layers);

    for (std::size_t li = 0; li < cfg_.layers; ++li) {
      const std::string p = cfg_.layers_prefix + "." + std::to_string(li);
      if (!reader_.ReadF32(p + ".input_layernorm.weight", H, &in_w, error) ||
          !reader_.ReadF32(p + ".post_attention_layernorm.weight", H, &post_w,
                           error) ||
          !reader_.ReadF32(p + ".self_attn.q_norm.weight", hd, &qn_w, error) ||
          !reader_.ReadF32(p + ".self_attn.k_norm.weight", hd, &kn_w,
                           error)) {
        return false;
      }
      for (std::size_t t = 0; t < T; ++t) {
        qwen3_detail::RmsNormRow(h->data() + t * H, in_w.data(), H,
                                 cfg_.rms_eps, x.data() + t * H);
      }
      if (!LinearNpu(p + ".self_attn.q_proj.weight", x, T, qd, H, &q_all,
                     npu_ms, error) ||
          !LinearNpu(p + ".self_attn.k_proj.weight", x, T, kd, H, &k_all,
                     npu_ms, error) ||
          !LinearNpu(p + ".self_attn.v_proj.weight", x, T, kd, H, &v_all,
                     npu_ms, error)) {
        return false;
      }
      for (std::size_t t = 0; t < T; ++t) {
        qwen3_detail::QkNormRope(q_all.data() + t * qd, qn_w.data(), nh, hd,
                                 pos_base + t, cfg_.rope_theta, cfg_.rms_eps);
        qwen3_detail::QkNormRope(k_all.data() + t * kd, kn_w.data(), nkv, hd,
                                 pos_base + t, cfg_.rope_theta, cfg_.rms_eps);
      }
      attn_out.assign(T * qd, 0.0F);
      if (append_kv) {
        // Prefill: causal attention over the prompt rows.
        for (std::size_t t = 0; t < T; ++t) {
          for (std::size_t hh = 0; hh < nh; ++hh) {
            const std::size_t kvh = hh / group;
            kh_slice.resize((t + 1) * hd);
            vh_slice.resize((t + 1) * hd);
            for (std::size_t k = 0; k <= t; ++k) {
              const float* ksrc = k_all.data() + k * kd + kvh * hd;
              const float* vsrc = v_all.data() + k * kd + kvh * hd;
              std::copy(ksrc, ksrc + hd, kh_slice.data() + k * hd);
              std::copy(vsrc, vsrc + hd, vh_slice.data() + k * hd);
            }
            qwen3_detail::AttnOneQuery(q_all.data() + t * qd + hh * hd,
                                       kh_slice.data(), vh_slice.data(), t + 1,
                                       hd, attn_out.data() + t * qd + hh * hd);
          }
        }
      } else {
        // Decode: single query at absolute position pos_base over cache.
        const float* Kc = kv_.K(li);
        const float* Vc = kv_.V(li);
        const std::size_t filled = kv_.filled();
        for (std::size_t hh = 0; hh < nh; ++hh) {
          const std::size_t kvh = hh / group;
          kh_slice.resize((filled + 1) * hd);
          vh_slice.resize((filled + 1) * hd);
          for (std::size_t k = 0; k < filled; ++k) {
            const float* ksrc = Kc + k * kd + kvh * hd;
            const float* vsrc = Vc + k * kd + kvh * hd;
            std::copy(ksrc, ksrc + hd, kh_slice.data() + k * hd);
            std::copy(vsrc, vsrc + hd, vh_slice.data() + k * hd);
          }
          std::copy(k_all.data() + kvh * hd, k_all.data() + kvh * hd + hd,
                    kh_slice.data() + filled * hd);
          std::copy(v_all.data() + kvh * hd, v_all.data() + kvh * hd + hd,
                    vh_slice.data() + filled * hd);
          qwen3_detail::AttnOneQuery(q_all.data() + hh * hd, kh_slice.data(),
                                     vh_slice.data(), filled + 1, hd,
                                     attn_out.data() + hh * hd);
        }
      }
      pre_k[li] = k_all;
      pre_v[li] = v_all;
      if (!LinearNpu(p + ".self_attn.o_proj.weight", attn_out, T, H, qd, &tmp,
                     npu_ms, error)) {
        return false;
      }
      for (std::size_t i = 0; i < T * H; ++i) (*h)[i] += tmp[i];
      for (std::size_t t = 0; t < T; ++t) {
        qwen3_detail::RmsNormRow(h->data() + t * H, post_w.data(), H,
                                 cfg_.rms_eps, x.data() + t * H);
      }
      if (!LinearNpu(p + ".mlp.gate_proj.weight", x, T, F, H, &gate, npu_ms,
                     error) ||
          !LinearNpu(p + ".mlp.up_proj.weight", x, T, F, H, &up, npu_ms,
                     error)) {
        return false;
      }
      for (std::size_t i = 0; i < T * F; ++i) {
        gate[i] = qwen3_detail::SiluLocal(gate[i]) * up[i];
      }
      if (!LinearNpu(p + ".mlp.down_proj.weight", gate, T, H, F, &down, npu_ms,
                     error)) {
        return false;
      }
      for (std::size_t i = 0; i < T * H; ++i) (*h)[i] += down[i];
    }

    // Commit K/V to the cache in position order (fixes cross-layer staging).
    std::vector<float> kl(cfg_.layers * kd), vl(cfg_.layers * kd);
    const std::size_t n_new = append_kv ? T : 1;
    for (std::size_t t = 0; t < n_new; ++t) {
      for (std::size_t li = 0; li < cfg_.layers; ++li) {
        std::copy(pre_k[li].data() + t * kd, pre_k[li].data() + (t + 1) * kd,
                  kl.data() + li * kd);
        std::copy(pre_v[li].data() + t * kd, pre_v[li].data() + (t + 1) * kd,
                  vl.data() + li * kd);
      }
      kv_.Append(kl, vl);
    }
    return true;
  }

  NpuQuantGemmEngine* engine_ = nullptr;
  NpuQuantMode mode_ = NpuQuantMode::kQ8Bfp16;
  Qwen3NpuDecoderConfig cfg_;
  NpuShardReader reader_;
  kv::GqaKvCache kv_;
  std::vector<float> final_norm_w_;
  std::size_t pos_ = 0;
};

}  // namespace mbufo::core::npu
