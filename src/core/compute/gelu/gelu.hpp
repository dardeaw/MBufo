// Gaussian Error Linear Unit for encoder FFNs (XLM-R family).
//
// Which variant a checkpoint wants is fixed by its HF config `hidden_act`:
//   * "gelu"     -> exact erf form 0.5*x*(1+erf(x/sqrt(2))). This is
//                   transformers ACT2FN["gelu"] and what bge-m3 (XLM-R,
//                   hidden_act "gelu") trains with. USE THIS for bge-m3.
//   * "gelu_new" -> tanh approximation 0.5*x*(1+tanh(sqrt(2/pi)*(x+
//                   0.044715*x^3))). This is BERT's original approximation
//                   and transformers ACT2FN["gelu_new"]. Provided here only
//                   for the documented对照 (tests/core/layernorm_gelu_test.cpp
//                   pins both against closed forms and the local torch check
//                   confirms exact-erf is the one matching bge-m3).
// Pure functions over caller memory; no allocation, no state.
#ifndef GUFO_CORE_COMPUTE_GELU_GELU_HPP_
#define GUFO_CORE_COMPUTE_GELU_GELU_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace gufo::core::gelu {

enum class GeluMode : std::uint8_t {
  /// Exact erf form; the bge-m3 / XLM-R `hidden_act: "gelu"` choice.
  kExact,
  /// BERT-original tanh approximation (transformers "gelu_new");对照 only.
  kTanh,
};

/// Exact erf GELU, computed in double and rounded once to float.
[[nodiscard]] inline float GeluExact(float x) {
  static constexpr double kSqrtHalf = 0.70710678118654752440;  // 1/sqrt(2)
  const double xd = static_cast<double>(x);
  return static_cast<float>(0.5 * xd * (1.0 + std::erf(xd * kSqrtHalf)));
}

/// Tanh-approximation GELU, computed in double and rounded once to float.
[[nodiscard]] inline float GeluTanh(float x) {
  static constexpr double kAlpha = 0.79788456080286535588;  // sqrt(2/pi)
  static constexpr double kBeta = 0.044715;
  const double xd = static_cast<double>(x);
  const double inner = kAlpha * (xd + kBeta * xd * xd * xd);
  return static_cast<float>(0.5 * xd * (1.0 + std::tanh(inner)));
}

/// Elementwise GELU over `x` into `out` (`x == out` is safe).
/// Returns false (leaving `out` untouched) on shape mismatch.
[[nodiscard]] inline bool ApplyGelu(std::span<const float> x,
                                    std::span<float> out, GeluMode mode) {
  if (x.size() != out.size()) return false;
  switch (mode) {
    case GeluMode::kExact:
      for (std::size_t i = 0; i < x.size(); ++i) out[i] = GeluExact(x[i]);
      return true;
    case GeluMode::kTanh:
      for (std::size_t i = 0; i < x.size(); ++i) out[i] = GeluTanh(x[i]);
      return true;
  }
  return false;
}

}  // namespace gufo::core::gelu

#endif  // GUFO_CORE_COMPUTE_GELU_GELU_HPP_
