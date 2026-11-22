#pragma once
// Shared DiT (diffusion transformer) NPU support: host-side math mirroring
// the parity shell bit-for-bit, plus streamed NPU GEMMs.
//
// The parity shells (RunFlux2KleinQuantE2E / RunZImageTurboQuantE2E) proved
// NPU-vs-FP32 equivalence per projection. The servable DiT ports in
// flux_dit_npu / zimage_dit_npu reuse these exact helpers so the only delta
// vs the shell is NPU-only GEMMs (no dual FP32 path at serve time).
// Convention: weights are [N, K] row-major f32 (streamed per projection);
// biases optional; all norms/activations/softmax stay host-FP32.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/npu_shard_reader.hpp"

namespace mbufo::core::npu::dit {

inline void RmsNormRows(const float* x, const float* w_or_null,
                        std::size_t rows, std::size_t cols, float eps,
                        std::vector<float>* out) {
  out->resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * cols;
    float* yr = out->data() + r * cols;
    double ss = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      ss += static_cast<double>(xr[c]) * static_cast<double>(xr[c]);
    }
    const float inv =
        static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(cols) + eps));
    for (std::size_t c = 0; c < cols; ++c) {
      yr[c] = xr[c] * inv * (w_or_null != nullptr ? w_or_null[c] : 1.0F);
    }
  }
}

inline void SiluInPlace(std::vector<float>& v) {
  for (float& x : v) x = x / (1.0F + std::exp(-x));
}

/// FLUX/Z-Image timestep sinusoid [256]: cos half then sin half,
/// freq = 10000^(-i/128) * sigma * 1000.
inline void TimestepSinusoid(float sigma, std::vector<float>* out) {
  out->assign(256, 0.0F);
  for (std::size_t i = 0; i < 128; ++i) {
    const float freq = std::pow(10000.0F, -static_cast<float>(i) / 128.0F) *
                       sigma * 1000.0F;
    (*out)[i] = std::cos(freq);
    (*out)[128 + i] = std::sin(freq);
  }
}

/// Naive host [M,N]x[N,K] reference (test plumbing gate only).
inline void MatMulRefHost(const float* A, std::size_t M, const float* W_nk,
                          std::size_t N, std::size_t K, const float* bias,
                          std::vector<float>* C) {
  C->resize(M * N);
  for (std::size_t m = 0; m < M; ++m) {
    const float* a_row = A + m * K;
    float* c_row = C->data() + m * N;
    for (std::size_t n = 0; n < N; ++n) {
      const float* w_row = W_nk + n * K;
      double acc = (bias != nullptr) ? static_cast<double>(bias[n]) : 0.0;
      for (std::size_t k = 0; k < K; ++k) {
        acc += static_cast<double>(a_row[k]) * static_cast<double>(w_row[k]);
      }
      c_row[n] = static_cast<float>(acc);
    }
  }
}

inline float CosineSimHost(std::span<const float> a, std::span<const float> b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  if (na <= 0.0 || nb <= 0.0) return 0.0F;
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

/// Exact port of the shell's ApplyGqaRopeAttn: per-head QK RMS-norm, 1D
/// interleaved RoPE at sequential positions, full (non-causal unless asked)
/// softmax at 1/sqrt(hd). Used by BOTH the NPU path and the test FP32 path.
inline void RopeAttn(const std::vector<float>& q_in,
                     const std::vector<float>& k_in,
                     const std::vector<float>& v_in,
                     std::span<const float> q_norm_w,
                     std::span<const float> k_norm_w, std::size_t seq_len,
                     std::size_t n_heads, std::size_t n_kv_heads,
                     std::size_t head_dim, float rope_theta, float rms_eps,
                     bool causal, std::vector<float>* ctx_out) {
  const std::size_t qd = n_heads * head_dim;
  const std::size_t kd = n_kv_heads * head_dim;
  const std::size_t half = head_dim / 2u;
  std::vector<float> q = q_in;
  std::vector<float> k = k_in;
  for (std::size_t t = 0; t < seq_len; ++t) {
    for (std::size_t h = 0; h < n_heads; ++h) {
      float* qh = q.data() + t * qd + h * head_dim;
      if (!q_norm_w.empty()) {
        double ss = 0.0;
        for (std::size_t i = 0; i < head_dim; ++i) ss += (double)qh[i] * qh[i];
        const float inv =
            static_cast<float>(1.0 / std::sqrt(ss / (double)head_dim + rms_eps));
        for (std::size_t i = 0; i < head_dim; ++i) qh[i] = qh[i] * inv * q_norm_w[i];
      }
      if (rope_theta > 0.0F) {
        for (std::size_t i = 0; i < half; ++i) {
          const double freq = std::pow((double)rope_theta,
                                       -2.0 * (double)i / (double)head_dim);
          const double ang = (double)t * freq;
          const float cs = (float)std::cos(ang), sn = (float)std::sin(ang);
          const float x0 = qh[i], x1 = qh[half + i];
          qh[i] = x0 * cs - x1 * sn;
          qh[half + i] = x1 * cs + x0 * sn;
        }
      }
    }
    for (std::size_t h = 0; h < n_kv_heads; ++h) {
      float* kh = k.data() + t * kd + h * head_dim;
      if (!k_norm_w.empty()) {
        double ss = 0.0;
        for (std::size_t i = 0; i < head_dim; ++i) ss += (double)kh[i] * kh[i];
        const float inv =
            static_cast<float>(1.0 / std::sqrt(ss / (double)head_dim + rms_eps));
        for (std::size_t i = 0; i < head_dim; ++i) kh[i] = kh[i] * inv * k_norm_w[i];
      }
      if (rope_theta > 0.0F) {
        for (std::size_t i = 0; i < half; ++i) {
          const double freq = std::pow((double)rope_theta,
                                       -2.0 * (double)i / (double)head_dim);
          const double ang = (double)t * freq;
          const float cs = (float)std::cos(ang), sn = (float)std::sin(ang);
          const float x0 = kh[i], x1 = kh[half + i];
          kh[i] = x0 * cs - x1 * sn;
          kh[half + i] = x1 * cs + x0 * sn;
        }
      }
    }
  }
  ctx_out->assign(seq_len * qd, 0.0F);
  const std::size_t group = n_heads / n_kv_heads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> scores(seq_len);
  for (std::size_t h = 0; h < n_heads; ++h) {
    const std::size_t kvh = h / group;
    for (std::size_t tq = 0; tq < seq_len; ++tq) {
      const float* qvec = q.data() + tq * qd + h * head_dim;
      const std::size_t tk_max = causal ? (tq + 1u) : seq_len;
      float max_s = -1.0e30F;
      for (std::size_t tk = 0; tk < tk_max; ++tk) {
        const float* kvec = k.data() + tk * kd + kvh * head_dim;
        double dot = 0.0;
        for (std::size_t i = 0; i < head_dim; ++i) dot += (double)qvec[i] * kvec[i];
        const float s = (float)dot * scale;
        scores[tk] = s;
        if (s > max_s) max_s = s;
      }
      double sum_exp = 0.0;
      for (std::size_t tk = 0; tk < tk_max; ++tk) {
        const float e = std::exp(scores[tk] - max_s);
        scores[tk] = e;
        sum_exp += (double)e;
      }
      const float inv_sum = (sum_exp > 0.0) ? (float)(1.0 / sum_exp) : 0.0F;
      float* cdst = ctx_out->data() + tq * qd + h * head_dim;
      for (std::size_t tk = 0; tk < tk_max; ++tk) {
        const float p = scores[tk] * inv_sum;
        const float* vvec = v_in.data() + tk * kd + kvh * head_dim;
        for (std::size_t i = 0; i < head_dim; ++i) cdst[i] += p * vvec[i];
      }
    }
  }
}

/// Streamed NPU linear: read f32 [N,K] (+ optional [N] bias), pack, GEMM.
class DitNpuLinear {
 public:
  DitNpuLinear() = default;
  void Bind(NpuShardReader* reader, NpuQuantGemmEngine* engine,
            NpuQuantMode mode) {
    reader_ = reader;
    engine_ = engine;
    mode_ = mode;
  }
  [[nodiscard]] bool Run(const std::string& weight_name,
                         const std::string& bias_name,
                         std::span<const float> x, std::size_t M,
                         std::size_t N, std::size_t K, std::vector<float>* y,
                         double* npu_ms, std::string* error) {
    std::vector<float> w, b;
    if (!reader_->ReadF32(weight_name, N * K, &w, error)) return false;
    const float* bp = nullptr;
    if (!bias_name.empty()) {
      if (!reader_->ReadF32(bias_name, N, &b, error)) return false;
      bp = b.data();
    }
    // The NPU kernel packs in 256-wide tiles; narrow projections (DiT
    // embedders/heads) fall back to host FP32, exactly as the parity shell
    // does for x_embedder/proj_out.
    if (N % 256u != 0u || K % 256u != 0u) {
      MatMulRefHost(x.data(), M, w.data(), N, K, bp, y);
      return true;
    }
    QuantWeightMatrix qm;
    if (!engine_->PackMatrixNK(w.data(), N, K, mode_, false, &qm, error)) {
      return false;
    }
    y->resize(M * N);
    return engine_->Gemm(x.data(), M, qm, y->data(), bp, npu_ms, error);
  }

 private:
  NpuShardReader* reader_ = nullptr;
  NpuQuantGemmEngine* engine_ = nullptr;
  NpuQuantMode mode_ = NpuQuantMode::kQ8Bfp16;
};

}  // namespace mbufo::core::npu::dit
