#include "src/eval/regrade.hpp"

namespace mbufo::eval {

RegradeSummary RegradeTrace(const EvalTraceReport& report) noexcept {
  RegradeSummary summary;
  summary.total_cases = report.cases.size();

  for (const auto& c : report.cases) {
    if ((!c.expected_answer.empty() &&
         c.extracted_answer == c.expected_answer) ||
        c.passed) {
      ++summary.passed_cases;
    } else {
      ++summary.failed_cases;
    }
  }

  if (summary.total_cases > 0) {
    summary.pass_rate = static_cast<double>(summary.passed_cases) /
                        static_cast<double>(summary.total_cases);
  }

  return summary;
}

}  // namespace mbufo::eval
