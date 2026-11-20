#include <cassert>
#include <iostream>

#include "src/eval/regrade.hpp"
#include "src/eval/trace.hpp"

void TestEvalRegradeAllPass() {
  mbufo::eval::EvalTraceReport report;
  report.model_id = "qwen35-4b";
  report.route_identity = "GPU_ONLY:gfx1151";
  report.suite_hash = "sha256:abcd1234";

  mbufo::eval::EvalCaseTrace c1;
  c1.case_id = "c1";
  c1.expected_answer = "Paris";
  c1.extracted_answer = "Paris";
  c1.passed = true;
  report.cases.push_back(c1);

  mbufo::eval::EvalCaseTrace c2;
  c2.case_id = "c2";
  c2.expected_answer = "Tokyo";
  c2.extracted_answer = "Tokyo";
  c2.passed = true;
  report.cases.push_back(c2);

  const auto summary = mbufo::eval::RegradeTrace(report);
  assert(summary.total_cases == 2);
  assert(summary.passed_cases == 2);
  assert(summary.failed_cases == 0);
  assert(summary.pass_rate == 1.0);
}

void TestEvalRegradePartialPass() {
  mbufo::eval::EvalTraceReport report;
  report.model_id = "qwen35-4b";

  mbufo::eval::EvalCaseTrace c1;
  c1.case_id = "c1";
  c1.expected_answer = "Paris";
  c1.extracted_answer = "London";
  c1.passed = false;
  report.cases.push_back(c1);

  mbufo::eval::EvalCaseTrace c2;
  c2.case_id = "c2";
  c2.expected_answer = "Tokyo";
  c2.extracted_answer = "Tokyo";
  c2.passed = true;
  report.cases.push_back(c2);

  const auto summary = mbufo::eval::RegradeTrace(report);
  assert(summary.total_cases == 2);
  assert(summary.passed_cases == 1);
  assert(summary.failed_cases == 1);
  assert(summary.pass_rate == 0.5);
}

int main() {
  TestEvalRegradeAllPass();
  TestEvalRegradePartialPass();
  std::cout << "All offline evaluation regrade tests passed.\n";
  return 0;
}
