#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

#include "src/eval/dataset.hpp"
#include "src/eval/extract.hpp"

#ifndef MBUFO_EVAL_TEST_DATA
#define MBUFO_EVAL_TEST_DATA \
  "tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json"
#endif

namespace {

mbufo::eval::EvalCase MultipleChoiceCase(std::string answer,
                                        std::size_t choices = 10) {
  mbufo::eval::EvalCase eval_case;
  eval_case.kind = mbufo::eval::AnswerKind::kMultipleChoice;
  eval_case.answer = std::move(answer);
  for (std::size_t index = 0; index < choices; ++index) {
    eval_case.choices.push_back(std::string(1, static_cast<char>('A' + index)));
  }
  return eval_case;
}

void TestPinnedDataset() {
  std::string error;
  const auto suite = mbufo::eval::LoadEvalSuite(
      std::filesystem::path(MBUFO_EVAL_TEST_DATA), &error);
  assert(suite.has_value());
  assert(error.empty());
  assert(suite->source_revision == "84cc882352757baf628a1776badf7cc54d584e28");
  assert(suite->source_blob == "7aed5d5c5b5cdc74d1b3aa0310161e2b437c1d08");
  assert(suite->cases.size() == 75);
  for (std::size_t index = 0; index < suite->cases.size(); ++index) {
    assert(suite->cases[index].ds4_index == index);
    assert(!suite->cases[index].source_record_sha256.empty());
    assert(!suite->cases[index].license.empty());
  }
  assert(suite->cases[0].id == "recNu3MXkvWUzHZr9");
  assert(suite->cases[0].answer == "B");
  assert(suite->cases[1].id == "001b51d76b4d422988f2c11f104a2c6c");
  assert(suite->cases[1].answer == "C");
  assert(suite->cases[2].id == "aime2025-01");
  assert(suite->cases[2].answer == "70");
  assert(suite->cases[3].id == "recoiTJPGUmzAkief");
  assert(suite->cases[3].answer == "C");

  const std::string prompt = mbufo::eval::BuildUserPrompt(suite->cases[2]);
  assert(prompt.starts_with(
      "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor "
      "of $97_b.$\n"));
  assert(prompt.ends_with("Answer: <integer>"));
}

void TestMultipleChoiceExtraction() {
  auto eval_case = MultipleChoiceCase("F");
  assert(mbufo::eval::GradeAnswer(
             eval_case,
             "</think>Answer: F\nThis answer is final; option H is tempting.")
             .passed);
  assert(mbufo::eval::GradeAnswer(eval_case, "</think>Answer: I think it is F")
             .passed);
  assert(mbufo::eval::GradeAnswer(eval_case, "</think>Answer: I'll go with F.")
             .passed);

  eval_case.answer = "D";
  assert(mbufo::eval::GradeAnswer(eval_case,
                                 "</think>Answer: It is not B, the answer is D")
             .passed);
  assert(mbufo::eval::GradeAnswer(eval_case,
                                 "</think>Answer: rules out C, leaving D")
             .passed);
  assert(mbufo::eval::GradeAnswer(eval_case, "</think>Answer: D, not B").passed);

  const auto ambiguous =
      mbufo::eval::ExtractAnswer(eval_case, "</think>Answer: B or D");
  assert(ambiguous.status == mbufo::eval::ExtractionStatus::kAmbiguous);
  const auto missing =
      mbufo::eval::ExtractAnswer(eval_case, "No final choice was produced.");
  assert(missing.status == mbufo::eval::ExtractionStatus::kNoAnswer);
}

void TestIntegerExtraction() {
  mbufo::eval::EvalCase eval_case;
  eval_case.kind = mbufo::eval::AnswerKind::kInteger;
  eval_case.answer = "293";
  assert(
      mbufo::eval::GradeAnswer(eval_case, "</think>Answer: m+n = 256+37 = 293")
          .passed);
  eval_case.answer = "82";
  assert(mbufo::eval::GradeAnswer(
             eval_case, "</think>Answer: 082\nThe value 2025 is just the year.")
             .passed);
  const auto ambiguous =
      mbufo::eval::ExtractAnswer(eval_case, "</think>Answer: 81 or 82");
  assert(ambiguous.status == mbufo::eval::ExtractionStatus::kAmbiguous);
  const auto missing = mbufo::eval::ExtractAnswer(eval_case, "No number given.");
  assert(missing.status == mbufo::eval::ExtractionStatus::kNoAnswer);
}

void TestLineSpecExtraction() {
  mbufo::eval::EvalCase eval_case;
  eval_case.kind = mbufo::eval::AnswerKind::kLineSpec;
  eval_case.answer = "9-10";
  assert(mbufo::eval::GradeAnswer(eval_case, "Answer: line 10").passed);
  assert(mbufo::eval::GradeAnswer(eval_case, "Answer: lines 9-10").passed);
  assert(!mbufo::eval::GradeAnswer(eval_case, "Answer: lines 9-11").passed);
  assert(mbufo::eval::ExtractAnswer(eval_case, "No localization.").status ==
         mbufo::eval::ExtractionStatus::kNoAnswer);
}

}  // namespace

int main() {
  TestPinnedDataset();
  TestMultipleChoiceExtraction();
  TestIntegerExtraction();
  TestLineSpecExtraction();
  std::cout << "All evaluation dataset and extractor tests passed.\n";
  return 0;
}
