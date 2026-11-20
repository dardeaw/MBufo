#ifndef MBUFO_TESTING_COMPARE_LOGIT_COMPARATOR_HPP_
#define MBUFO_TESTING_COMPARE_LOGIT_COMPARATOR_HPP_

#include <cstddef>
#include <span>
#include <string>

namespace mbufo::testing {

struct LogitCompareResult {
  bool match{true};
  bool finite{true};
  bool top1_match{true};
  float max_abs_diff{0.0F};
  float max_rel_diff{0.0F};
  float mean_abs_diff{0.0F};
  float root_mean_square_error{0.0F};
  float cosine_similarity{1.0F};
  std::size_t first_mismatch_idx{0};
  std::size_t reference_argmax{0};
  std::size_t candidate_argmax{0};
  std::string details;
};

/// Compares candidate logits against reference logits with specified atol and
/// rtol tolerances.
LogitCompareResult CompareLogits(std::span<const float> reference,
                                 std::span<const float> candidate,
                                 float atol = 1e-3F, float rtol = 1e-3F);

}  // namespace mbufo::testing

#endif  // MBUFO_TESTING_COMPARE_LOGIT_COMPARATOR_HPP_
