// Shared sequence-to-vector reduction for embedding models.
//
// pooling_norm owns the model-agnostic math every embedding endpoint needs:
// picking one pooled row out of a prefill's final-layer hidden states and
// L2-normalizing it. Model code only declares *which* pooling the artifact
// wants (via its GGUF `pooling_type`); the reduction itself lives here so no
// model directory carries its own copy.
//
// Both helpers are pure functions over caller-owned memory: the caller passes
// the hidden buffer and the output scratch, so there is no allocation, no
// device handle, and no retained state. The CPU reference below is what the
// first serving leg uses; `src/core/compute/pooling/pooling_norm.hip` holds
// the device twin behind ENGINE_ENABLE_HIP.
#ifndef GUFO_CORE_POOLING_NORM_HPP_
#define GUFO_CORE_POOLING_NORM_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace gufo::core::embedding {

/// Sequence-to-vector reduction a model artifact declares.
enum class PoolingStrategy : std::uint8_t {
  /// Last non-padding row (causal decoders with EOS pooling).
  kLastToken,
  /// Mean over all rows (bidirectional encoders).
  kMean,
  /// First row (CLS-style encoders).
  kCls,
};

/// Maps a GGUF `<arch>.pooling_type` metadata value onto a strategy.
/// llama.cpp convention: 0 = unspecified/none, 1 = mean, 2 = CLS, 3 = last.
/// 0, unknown, and out-of-range values yield nullopt: the caller must refuse
/// the embedding explicitly rather than guess a reduction.
[[nodiscard]] inline std::optional<PoolingStrategy> PoolingStrategyFromGgufType(
    std::uint32_t pooling_type) {
  switch (pooling_type) {
    case 1:
      return PoolingStrategy::kMean;
    case 2:
      return PoolingStrategy::kCls;
    case 3:
      return PoolingStrategy::kLastToken;
    default:
      return std::nullopt;
  }
}

/// Reduces row-major `[token_count, dim]` final-layer hidden states to one
/// `dim` vector in caller scratch `out`. Returns false (and leaves `out`
/// untouched) when the shapes mismatch or `token_count` is zero.
[[nodiscard]] inline bool ApplyPooling(std::span<const float> hidden,
                                       std::size_t token_count,
                                       std::size_t dim,
                                       PoolingStrategy strategy,
                                       std::span<float> out) {
  if (dim == 0 || out.size() != dim) return false;
  if (token_count == 0 || hidden.size() != token_count * dim) return false;
  switch (strategy) {
    case PoolingStrategy::kLastToken: {
      const float* row = hidden.data() + (token_count - 1) * dim;
      for (std::size_t d = 0; d < dim; ++d) out[d] = row[d];
      return true;
    }
    case PoolingStrategy::kCls: {
      for (std::size_t d = 0; d < dim; ++d) out[d] = hidden[d];
      return true;
    }
    case PoolingStrategy::kMean: {
      for (std::size_t d = 0; d < dim; ++d) {
        double acc = 0.0;
        for (std::size_t t = 0; t < token_count; ++t)
          acc += static_cast<double>(hidden[t * dim + d]);
        out[d] = static_cast<float>(acc / static_cast<double>(token_count));
      }
      return true;
    }
  }
  return false;
}

/// Euclidean norm of a vector.
[[nodiscard]] inline double VectorNorm(std::span<const float> vec) {
  double acc = 0.0;
  for (const float x : vec)
    acc += static_cast<double>(x) * static_cast<double>(x);
  return std::sqrt(acc);
}

/// Scales `vec` to unit L2 norm in place and returns the pre-norm.
/// A zero (or non-finite) vector is left untouched and reports 0.0; callers
/// treat that as an explicit refusal, never as a silent zero embedding.
[[nodiscard]] inline double L2Normalize(std::span<float> vec) {
  const double norm = VectorNorm(vec);
  if (!(norm > 0.0)) return 0.0;
  const float inv = static_cast<float>(1.0 / norm);
  for (float& x : vec) x *= inv;
  return norm;
}

#if defined(ENGINE_ENABLE_HIP)
// Device twin of the helpers above
// (src/core/compute/pooling/pooling_norm.hip). Same contract; `stream` is a
// hipStream_t passed through as void* so this header stays HIP-free.
void PoolingReduceDevice(const float* hidden, std::size_t token_count,
                         std::size_t dim, PoolingStrategy strategy, float* out,
                         void* stream);
void L2NormalizeDevice(float* vec, std::size_t dim, double* norm_out,
                       void* stream);
#endif

}  // namespace gufo::core::embedding

#endif  // GUFO_CORE_POOLING_NORM_HPP_
