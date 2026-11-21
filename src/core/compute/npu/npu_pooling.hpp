#pragma once
// NPU pooling + L2 normalization adapter bridging bf16/fp32 device outputs to
// core::embedding (exact parity with src/core/pooling_norm.hpp and
// src/models/embeddinggemma2/hip/session.hip).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "src/core/compute/npu/npu_kernels.hpp"
#include "src/core/pooling_norm.hpp"

namespace mbufo::core::npu {

[[nodiscard]] inline float Bf16BitsToFloat(std::uint16_t bits) noexcept {
  const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16u;
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

[[nodiscard]] inline std::uint16_t FloatToBf16Rne(float v) noexcept {
  std::uint32_t u = 0;
  std::memcpy(&u, &v, sizeof(u));
  // Preserve NaN as quiet NaN.
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
    return static_cast<std::uint16_t>((u >> 16u) | 0x0040u);
  }
  const std::uint32_t rounding_bias = 0x7FFFu + ((u >> 16u) & 1u);
  return static_cast<std::uint16_t>((u + rounding_bias) >> 16u);
}

/// Upcasts NPU bf16 hidden states `[seq_len, hidden_dim]`, runs exact
/// core::embedding::ApplyPooling (kCls / kMean / kLastToken) and L2Normalize.
/// Returns true iff shape is valid and pre-normalization L2 norm > 0.
[[nodiscard]] inline bool NpuPoolAndNormalizeBf16(
    std::span<const std::uint16_t> hidden_bf16, std::size_t seq_len,
    std::size_t hidden_dim, core::embedding::PoolingStrategy strategy,
    std::vector<float>& out, double* pre_norm_out = nullptr) {
  if (seq_len == 0 || hidden_dim == 0 ||
      hidden_bf16.size() != seq_len * hidden_dim) {
    return false;
  }
  std::vector<float> hidden_f32(hidden_bf16.size());
  for (std::size_t i = 0; i < hidden_bf16.size(); ++i) {
    hidden_f32[i] = Bf16BitsToFloat(hidden_bf16[i]);
  }
  out.assign(hidden_dim, 0.0f);
  if (!core::embedding::ApplyPooling(hidden_f32, seq_len, hidden_dim, strategy,
                                     out)) {
    return false;
  }
  const double pre_norm = core::embedding::L2Normalize(out);
  if (pre_norm_out != nullptr) *pre_norm_out = pre_norm;
  return pre_norm > 0.0;
}

/// Exact EmbeddingGemma 2 post-transformer head:
///   final RMSNorm(eps, final_norm_w) -> MeanPool -> out_proj [out_dim, H] -> L2.
[[nodiscard]] inline bool NpuEmbGemma2PostProcess(
    std::span<const float> hidden_f32, std::size_t seq_len,
    std::size_t hidden_dim, std::span<const float> final_norm_w, float rms_eps,
    std::span<const float> out_proj_w, std::size_t out_dim,
    std::vector<float>& out, double* pre_norm_out = nullptr) {
  if (seq_len == 0 || hidden_dim == 0 || out_dim == 0 ||
      hidden_f32.size() != seq_len * hidden_dim ||
      final_norm_w.size() != hidden_dim ||
      out_proj_w.size() != out_dim * hidden_dim) {
    return false;
  }
  std::vector<float> x_vec(hidden_f32.begin(), hidden_f32.end());
  std::vector<float> w_vec(final_norm_w.begin(), final_norm_w.end());
  std::vector<float> normed;
  RefRmsNormRows(x_vec, &w_vec, seq_len, hidden_dim,
                 static_cast<double>(rms_eps), normed);

  std::vector<float> mean(hidden_dim, 0.0f);
  if (!core::embedding::ApplyPooling(
          normed, seq_len, hidden_dim,
          core::embedding::PoolingStrategy::kMean, mean)) {
    return false;
  }
  out.assign(out_dim, 0.0f);
  for (std::size_t o = 0; o < out_dim; ++o) {
    double acc = 0.0;
    const std::size_t row_off = o * hidden_dim;
    for (std::size_t j = 0; j < hidden_dim; ++j) {
      acc += static_cast<double>(out_proj_w[row_off + j]) *
             static_cast<double>(mean[j]);
    }
    out[o] = static_cast<float>(acc);
  }
  const double pre_norm = core::embedding::L2Normalize(out);
  if (pre_norm_out != nullptr) *pre_norm_out = pre_norm;
  return pre_norm > 0.0;
}

/// BF16 overload of EmbeddingGemma 2 post-processing (from NPU final hidden buffer).
[[nodiscard]] inline bool NpuEmbGemma2PostProcessBf16(
    std::span<const std::uint16_t> hidden_bf16, std::size_t seq_len,
    std::size_t hidden_dim, std::span<const std::uint16_t> final_norm_w_bf16,
    float rms_eps, std::span<const std::uint16_t> out_proj_w_bf16,
    std::size_t out_dim, std::vector<float>& out,
    double* pre_norm_out = nullptr) {
  if (seq_len == 0 || hidden_dim == 0 || out_dim == 0 ||
      hidden_bf16.size() != seq_len * hidden_dim ||
      final_norm_w_bf16.size() != hidden_dim ||
      out_proj_w_bf16.size() != out_dim * hidden_dim) {
    return false;
  }
  std::vector<float> h(hidden_bf16.size());
  std::vector<float> nw(final_norm_w_bf16.size());
  std::vector<float> pw(out_proj_w_bf16.size());
  for (std::size_t i = 0; i < h.size(); ++i) h[i] = Bf16BitsToFloat(hidden_bf16[i]);
  for (std::size_t i = 0; i < nw.size(); ++i) nw[i] = Bf16BitsToFloat(final_norm_w_bf16[i]);
  for (std::size_t i = 0; i < pw.size(); ++i) pw[i] = Bf16BitsToFloat(out_proj_w_bf16[i]);
  return NpuEmbGemma2PostProcess(h, seq_len, hidden_dim, nw, rms_eps, pw,
                                 out_dim, out, pre_norm_out);
}

}  // namespace mbufo::core::npu
