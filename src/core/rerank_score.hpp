// Shared cross-encoder scoring math for rerank models.
//
// A reranker scores (query, document) pairs with one relevance value: the
// encoder runs a single full prefill over the concatenated pair and a small
// head maps the pooled state to a scalar. Both head shapes in service live
// here so no model directory carries its own copy:
//
// * sequence-classification head (bge-reranker-v2-m3): CLS row -> dense +
//   tanh -> single-logit out-projection -> sigmoid.
// * yes/no head (Qwen3-Reranker): last-token logits at the "yes"/"no"
//   vocabulary ids -> softmax -> P(yes).
//
// All helpers are pure functions over caller-owned memory: no allocation
// (except RankIndices' index vector), no device handle, no retained state.
// The HIP prefill twins stay in the model directories; only this scalar
// math is shared -- it runs on the host even for device prefills (a
// 1024-wide head is microseconds on CPU).
#ifndef GUFO_CORE_RERANK_SCORE_HPP_
#define GUFO_CORE_RERANK_SCORE_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

namespace gufo::core::rerank {

// Logistic sigmoid: maps a raw relevance logit to (0, 1).
[[nodiscard]] inline float Sigmoid(float logit) noexcept {
  return 1.0F / (1.0F + std::exp(-logit));
}

// P("yes") from the last-token logits at the yes/no vocabulary ids: the
// Qwen3-Reranker relevance score (softmax over the two candidates).
// A non-finite input yields NaN rather than a silent ranking.
[[nodiscard]] inline float YesNoScore(float logit_yes,
                                      float logit_no) noexcept {
  if (!(logit_yes > -std::numeric_limits<float>::infinity()) ||
      !(logit_yes < std::numeric_limits<float>::infinity()) ||
      !(logit_no > -std::numeric_limits<float>::infinity()) ||
      !(logit_no < std::numeric_limits<float>::infinity())) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const float max_logit = std::max(logit_yes, logit_no);
  const float exp_yes = std::exp(logit_yes - max_logit);
  const float exp_no = std::exp(logit_no - max_logit);
  return exp_yes / (exp_yes + exp_no);
}

// Roberta-style sequence-classification head logit:
//   tanh(cls * dense^T + dense_b) . out_w + out_b
// with dense stored row-major [hidden, hidden] and out_w a [hidden] vector
// (GGUF `cls.weight` / `cls.output.weight` layout -- the same row-major
// [out, in] convention as the encoder GEMMs). Shapes are preconditions:
// a mismatch yields NaN, never a silent score. Callers validate tensor
// shapes once at load; per-call dims are fixed.
[[nodiscard]] inline float SequenceHeadLogit(std::span<const float> cls,
                                             std::span<const float> dense_w,
                                             std::span<const float> dense_b,
                                             std::span<const float> out_w,
                                             float out_b) noexcept {
  const std::size_t hidden = cls.size();
  if (hidden == 0 || dense_w.size() != hidden * hidden ||
      dense_b.size() != hidden || out_w.size() != hidden) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  double logit = 0.0;
  for (std::size_t j = 0; j < hidden; ++j) {
    double acc = static_cast<double>(dense_b[j]);
    const float* row = dense_w.data() + j * hidden;
    for (std::size_t c = 0; c < hidden; ++c) {
      acc += static_cast<double>(cls[c]) * static_cast<double>(row[c]);
    }
    logit += std::tanh(acc) * static_cast<double>(out_w[j]);
  }
  return static_cast<float>(logit + static_cast<double>(out_b));
}

// Descending rank indices with stable ties: out[i] is the document index of
// the i-th most relevant score. top_n == 0 (or >= n) ranks everything.
[[nodiscard]] inline std::vector<std::size_t> RankIndices(
    std::span<const float> scores, std::size_t top_n = 0) {
  std::vector<std::size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(
      order.begin(), order.end(),
      [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
  if (top_n != 0 && top_n < order.size())
    order.resize(top_n);
  return order;
}

}  // namespace gufo::core::rerank

#endif  // GUFO_CORE_RERANK_SCORE_HPP_
