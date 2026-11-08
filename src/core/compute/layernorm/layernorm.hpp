// Traditional (biased-variance, with-bias) LayerNorm plus learned absolute
// positional addition for bidirectional encoder artifacts (XLM-R family).
//
// bge-m3 is XLM-RoBERTa-based, not Qwen-based: it needs the *traditional*
// LayerNorm (mean subtraction, optional bias, eps inside the root) rather
// than the RMSNorm in src/models/qwen/modules/norm.*. The math lives here so
// no model directory carries its own copy; both helpers are pure functions
// over caller-owned memory (no allocation, no device handle, no state).
//
// Precision note: mean/variance accumulate in double; the final scale is
// applied in float. This matches the HuggingFace FP32 reference bit-pattern
// for the tested magnitudes (see tests/core/layernorm_gelu_test.cpp).
#ifndef GUFO_CORE_COMPUTE_LAYERNORM_LAYERNORM_HPP_
#define GUFO_CORE_COMPUTE_LAYERNORM_LAYERNORM_HPP_

#include <cmath>
#include <cstddef>
#include <span>

namespace gufo::core::ln {

/// Traditional LayerNorm over one row:
///   out[i] = (x[i] - mean(x)) / sqrt(var(x) + eps) * weight[i]
///            + (bias.empty() ? 0 : bias[i])
/// Variance is the *biased* (population) variance, as in HuggingFace
/// BertLayerNorm / torch.nn.LayerNorm. `bias` may be empty (no bias term);
/// otherwise it must have exactly `dim` elements. `weight` must have `dim`
/// elements and `out` must alias `dim` elements (`x == out` is safe).
/// Returns false (leaving `out` untouched) on any shape mismatch or dim == 0.
[[nodiscard]] inline bool LayerNorm(std::span<const float> x,
                                    std::span<const float> weight,
                                    std::span<const float> bias, float eps,
                                    std::span<float> out) {
  const std::size_t dim = x.size();
  if (dim == 0 || weight.size() != dim || out.size() != dim) return false;
  if (!bias.empty() && bias.size() != dim) return false;
  double mean = 0.0;
  for (const float v : x) mean += static_cast<double>(v);
  mean /= static_cast<double>(dim);
  double var = 0.0;
  for (const float v : x) {
    const double d = static_cast<double>(v) - mean;
    var += d * d;
  }
  var /= static_cast<double>(dim);
  const float inv =
      static_cast<float>(1.0 / std::sqrt(var + static_cast<double>(eps)));
  const float mean_f = static_cast<float>(mean);
  if (bias.empty()) {
    for (std::size_t i = 0; i < dim; ++i) {
      out[i] = (x[i] - mean_f) * inv * weight[i];
    }
  } else {
    for (std::size_t i = 0; i < dim; ++i) {
      out[i] = (x[i] - mean_f) * inv * weight[i] + bias[i];
    }
  }
  return true;
}

/// Adds learned absolute position rows to a row-major [seq_len, dim] hidden
/// buffer in place: hidden[t, d] += pos_table[t * dim + d].
/// `pos_table` must hold at least `seq_len * dim` floats (extra rows, e.g. a
/// longer trained table, are ignored). Returns false on shape mismatch.
[[nodiscard]] inline bool AddPositional(std::span<float> hidden,
                                        std::span<const float> pos_table,
                                        std::size_t seq_len,
                                        std::size_t dim) {
  if (dim == 0 || seq_len == 0) return false;
  if (hidden.size() != seq_len * dim) return false;
  if (pos_table.size() < seq_len * dim) return false;
  for (std::size_t t = 0; t < seq_len; ++t) {
    float* row = hidden.data() + t * dim;
    const float* pos = pos_table.data() + t * dim;
    for (std::size_t d = 0; d < dim; ++d) row[d] += pos[d];
  }
  return true;
}

}  // namespace gufo::core::ln

#endif  // GUFO_CORE_COMPUTE_LAYERNORM_LAYERNORM_HPP_
