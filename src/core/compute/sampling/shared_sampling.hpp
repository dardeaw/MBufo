#pragma once
// P0 shared sampling (header-only, zero wiring).
//
// ASR thinker + TTS talker/predictor share one greedy/top-p/top-k path with
// per-model EOS sets. Image DiTs need no sampling (flow matching is
// deterministic given the schedule), so they ignore this header.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace mbufo::core::sampling {

struct SamplingParams {
  float temperature{0.0F};  ///< <=0 means greedy
  float top_p{1.0F};
  std::size_t top_k{0};  ///< 0 means full vocab
  std::vector<std::uint32_t> eos_ids{};
  std::uint64_t seed{0};
};

[[nodiscard]] inline std::uint32_t SampleLogits(std::span<const float> logits, const SamplingParams& p,
                                                std::mt19937* rng = nullptr) {
  const std::size_t n = logits.size();
  if (p.temperature <= 0.0F || n == 0) {
    return static_cast<std::uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  }
  std::vector<float> probs(n);
  float m = *std::max_element(logits.begin(), logits.end());
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) { probs[i] = std::exp((logits[i] - m) / p.temperature); sum += probs[i]; }
  for (auto& x : probs) x = static_cast<float>(x / sum);
  std::mt19937 local(p.seed);
  std::mt19937& r = (rng != nullptr) ? *rng : local;
  std::uniform_real_distribution<float> u(0.0F, 1.0F);
  float x = u(r), acc = 0.0F;
  for (std::size_t i = 0; i < n; ++i) { acc += probs[i]; if (x <= acc) return static_cast<std::uint32_t>(i); }
  return static_cast<std::uint32_t>(n - 1);
}

[[nodiscard]] inline bool IsEos(std::uint32_t id, const SamplingParams& p) noexcept {
  for (std::uint32_t e : p.eos_ids) { if (e == id) return true; }
  return false;
}

}  // namespace mbufo::core::sampling
