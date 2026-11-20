#ifndef MBUFO_EVAL_REGRADE_HPP_
#define MBUFO_EVAL_REGRADE_HPP_

#include "src/eval/trace.hpp"

namespace mbufo::eval {

struct RegradeSummary {
  std::size_t total_cases{0};
  std::size_t passed_cases{0};
  std::size_t failed_cases{0};
  double pass_rate{0.0};
};

/// Regrades a captured evaluation trace without loading any model weights or
/// initializing hardware.
[[nodiscard]] RegradeSummary RegradeTrace(
    const EvalTraceReport& report) noexcept;

}  // namespace mbufo::eval

#endif  // MBUFO_EVAL_REGRADE_HPP_
