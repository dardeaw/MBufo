#include "src/testing/compare/logit_comparator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace mbufo::testing {

LogitCompareResult CompareLogits(std::span<const float> reference,
                                 std::span<const float> candidate, float atol,
                                 float rtol) {
  LogitCompareResult res;
  if (reference.size() != candidate.size()) {
    res.match = false;
    res.top1_match = false;
    res.details = "Size mismatch: ref=" + std::to_string(reference.size()) +
                  ", cand=" + std::to_string(candidate.size());
    return res;
  }

  if (reference.empty()) {
    return res;
  }

  double sum_abs_diff = 0.0;
  double sum_squared_diff = 0.0;
  double dot_product = 0.0;
  double reference_squared_norm = 0.0;
  double candidate_squared_norm = 0.0;
  float best_reference = -std::numeric_limits<float>::infinity();
  float best_candidate = -std::numeric_limits<float>::infinity();

  for (std::size_t i = 0; i < reference.size(); ++i) {
    const float ref = reference[i];
    const float cand = candidate[i];
    if (!std::isfinite(ref) || !std::isfinite(cand)) {
      res.match = false;
      res.finite = false;
      res.max_abs_diff = std::numeric_limits<float>::infinity();
      res.max_rel_diff = std::numeric_limits<float>::infinity();
      res.mean_abs_diff = std::numeric_limits<float>::infinity();
      res.root_mean_square_error = std::numeric_limits<float>::infinity();
      res.cosine_similarity = 0.0F;
      if (res.details.empty()) {
        res.first_mismatch_idx = i;
        res.details = "Non-finite logit at index " + std::to_string(i);
      }
      continue;
    }

    if (ref > best_reference) {
      best_reference = ref;
      res.reference_argmax = i;
    }
    if (cand > best_candidate) {
      best_candidate = cand;
      res.candidate_argmax = i;
    }

    const float abs_diff = std::abs(ref - cand);
    const float denom = std::max(std::abs(ref), std::abs(cand));
    const float rel_diff = (denom > 1e-7F) ? (abs_diff / denom) : abs_diff;

    res.max_abs_diff = std::max(res.max_abs_diff, abs_diff);
    res.max_rel_diff = std::max(res.max_rel_diff, rel_diff);
    sum_abs_diff += static_cast<double>(abs_diff);
    sum_squared_diff += static_cast<double>(abs_diff) * abs_diff;
    dot_product += static_cast<double>(ref) * cand;
    reference_squared_norm += static_cast<double>(ref) * ref;
    candidate_squared_norm += static_cast<double>(cand) * cand;

    const float tol = atol + (rtol * std::abs(ref));
    if (abs_diff > tol && res.match) {
      res.match = false;
      res.first_mismatch_idx = i;
      std::ostringstream ss;
      ss << "Mismatch at index " << i << ": ref=" << ref << ", cand=" << cand
         << ", abs_diff=" << abs_diff << " (tol=" << tol << ")";
      res.details = ss.str();
    }
  }

  res.top1_match = res.reference_argmax == res.candidate_argmax;
  if (res.finite) {
    const auto count = static_cast<double>(reference.size());
    res.mean_abs_diff = static_cast<float>(sum_abs_diff / count);
    res.root_mean_square_error =
        static_cast<float>(std::sqrt(sum_squared_diff / count));
    const double norm_product =
        std::sqrt(reference_squared_norm * candidate_squared_norm);
    if (norm_product > 0.0) {
      res.cosine_similarity = static_cast<float>(dot_product / norm_product);
    } else {
      res.cosine_similarity =
          reference_squared_norm == candidate_squared_norm ? 1.0F : 0.0F;
    }
  }

  return res;
}

}  // namespace mbufo::testing
