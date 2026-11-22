#pragma once
// Qwen3-TTS NPU frame loop (P2c): talker decode -> predictor expansion ->
// frame-embedding feedback, with HIP-identical first-code policy.
//
// Mirrors TalkerHipRuntime::Generate (talker_runtime.cpp ~1690-1745):
// prefill prompt embeddings, then per step SelectMainCode (vocab masking +
// repetition penalty + temperature/top-k/top-p), EOS break, PredictFrame,
// DecodeCodeFrame feedback with trailing-text rows then tts_pad.
// Prompt builders (text_projection, trailing_text, tts_pad) stay on the
// existing HIP path; this loop owns only the autoregressive core.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/qwen3_predictor_decode.hpp"
#include "src/core/compute/npu/qwen3_talker_decode.hpp"

namespace mbufo::core::npu {

/// Mirrors models::qwen3_tts::SamplingOptions (+ codec EOS id).
struct TtsSampling {
  bool sample{true};
  std::uint32_t seed{42};
  std::size_t top_k{50};
  float temperature{0.9F};
  float top_p{1.0F};
  float repetition_penalty{1.05F};
  std::uint32_t eos_id{2150};  // TalkerConfig::codec_eos_token_id
};

struct TtsTokenScore {
  std::uint32_t token;
  float score;
};

/// Temperature + top-k (boundary ties kept) + nucleus. Null RNG = greedy.
/// Nonfinite logits are rejected, never sampled.
[[nodiscard]] inline std::uint32_t SampleCodecNpu(
    std::vector<TtsTokenScore>& candidates, std::size_t top_k, float top_p,
    float temperature, std::mt19937* rng) {
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [](const TtsTokenScore& c) {
                       return !std::isfinite(c.score);
                     }),
      candidates.end());
  if (candidates.empty()) return 0;
  if (rng == nullptr || temperature <= 0.0F) {
    return std::max_element(candidates.begin(), candidates.end(),
                            [](const TtsTokenScore& a,
                               const TtsTokenScore& b) {
                              return a.score < b.score;
                            })
        ->token;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const TtsTokenScore& a, const TtsTokenScore& b) {
              return a.score > b.score;
            });
  std::size_t keep = candidates.size();
  if (top_k > 0 && top_k < keep) {
    // Keep boundary ties with the kth score.
    const float cutoff = candidates[top_k - 1].score;
    keep = top_k;
    while (keep < candidates.size() && candidates[keep].score == cutoff) {
      ++keep;
    }
  }
  // Softmax over the kept prefix.
  float m = candidates.front().score;
  double sum = 0.0;
  std::vector<double> probs(keep);
  for (std::size_t i = 0; i < keep; ++i) {
    const double p =
        std::exp((static_cast<double>(candidates[i].score) - m) / temperature);
    probs[i] = p;
    sum += p;
  }
  // Nucleus truncation.
  std::size_t nucleus = keep;
  if (top_p < 1.0F) {
    double acc = 0.0;
    nucleus = 0;
    for (std::size_t i = 0; i < keep; ++i) {
      acc += probs[i] / sum;
      ++nucleus;
      if (acc >= top_p) break;
    }
    if (nucleus == 0) nucleus = 1;
  }
  double nsum = 0.0;
  for (std::size_t i = 0; i < nucleus; ++i) nsum += probs[i];
  std::uniform_real_distribution<double> u(0.0, nsum);
  const double x = u(*rng);
  double acc = 0.0;
  for (std::size_t i = 0; i < nucleus; ++i) {
    acc += probs[i];
    if (x <= acc) return candidates[i].token;
  }
  return candidates[nucleus - 1].token;
}

/// HIP SelectMainCode equivalent: tokens >= 2048 are masked except EOS;
/// EOS is masked for the first two steps; generated ids get the repetition
/// penalty. generated_flags has codec_vocab entries.
[[nodiscard]] inline std::uint32_t SelectFirstCodeNpu(
    std::span<const float> logits, std::span<const std::uint8_t> generated,
    std::size_t step, const TtsSampling& sampling, std::mt19937* rng,
    std::vector<TtsTokenScore>* scratch) {
  std::vector<TtsTokenScore>& candidates = *scratch;
  candidates.clear();
  candidates.reserve(2049);
  for (std::uint32_t token = 0; token < logits.size(); ++token) {
    if ((token >= 2048 && token != sampling.eos_id) ||
        (step < 2 && token == sampling.eos_id)) {
      continue;
    }
    float score = logits[token];
    if (token < generated.size() && generated[token] != 0) {
      score = score < 0.0F ? score * sampling.repetition_penalty
                           : score / sampling.repetition_penalty;
    }
    candidates.push_back({token, score});
  }
  return SampleCodecNpu(candidates, sampling.top_k, sampling.top_p,
                        sampling.temperature, rng);
}

class TtsNpuFrameLoop {
 public:
  TtsNpuFrameLoop() = default;

  void Bind(TalkerNpuDecoder* talker, PredictorNpuDecoder* predictor) {
    talker_ = talker;
    predictor_ = predictor;
  }

  /// Autoregressive codec frames. prompt/trailing/tts_pad come from the
  /// existing HIP prompt builders. Returns frames emitted (EOS may stop
  /// early; eos_hit reports it).
  [[nodiscard]] bool Generate(std::span<const float> prompt_embeddings,
                              std::size_t prompt_tokens,
                              std::span<const float> trailing_text,
                              std::size_t trailing_rows,
                              std::span<const float> tts_pad,
                              std::size_t max_frames, TtsSampling sampling,
                              std::vector<std::uint32_t>* codes,
                              std::size_t* frames_out, bool* eos_hit,
                              double* npu_ms = nullptr,
                              std::string* error = nullptr) {
    if (talker_ == nullptr || predictor_ == nullptr || codes == nullptr ||
        prompt_tokens == 0 || tts_pad.size() < talker_->config().hidden) {
      if (error != nullptr) *error = "bad TTS frame loop input";
      return false;
    }
    const std::size_t H = talker_->config().hidden;
    const std::size_t groups = talker_->config().num_code_groups;
    std::mt19937 rng(sampling.seed);
    std::mt19937* rng_ptr = sampling.sample ? &rng : nullptr;

    std::vector<float> hidden;
    if (!talker_->Prefill(prompt_embeddings, prompt_tokens, &hidden, npu_ms,
                          error)) {
      return false;
    }
    codes->clear();
    codes->reserve(max_frames * groups);
    std::vector<std::uint8_t> generated(talker_->config().codec_vocab, 0);
    std::vector<TtsTokenScore> scratch;
    std::size_t frames = 0;
    bool eos = false;
    for (std::size_t step = 0; step < max_frames; ++step) {
      std::vector<float> logits;
      if (!talker_->CodecLogits(hidden, &logits, npu_ms, error)) return false;
      const std::uint32_t first = SelectFirstCodeNpu(
          logits, generated, step, sampling, rng_ptr, &scratch);
      if (first == sampling.eos_id) {
        eos = true;
        break;
      }
      if (first < generated.size()) generated[first] = 1;
      std::vector<std::uint32_t> frame;
      if (!predictor_->PredictFrame(hidden, first, &frame, nullptr, npu_ms,
                                    error)) {
        return false;
      }
      codes->insert(codes->end(), frame.begin(), frame.end());
      ++frames;
      if (step + 1 == max_frames) break;
      std::span<const float> text_emb =
          (step < trailing_rows &&
           trailing_text.size() >= (step + 1) * H)
              ? std::span<const float>(trailing_text.data() + step * H, H)
              : std::span<const float>(tts_pad.data(), H);
      std::vector<float> frame_in;
      if (!BuildTalkerFrameInput(frame, text_emb, predictor_->group0_embed(),
                                 predictor_->group_embeds(), groups, H,
                                 &frame_in, error)) {
        return false;
      }
      if (!talker_->DecodeStep(frame_in, &hidden, npu_ms, error)) return false;
    }
    if (frames_out != nullptr) *frames_out = frames;
    if (eos_hit != nullptr) *eos_hit = eos;
    return true;
  }

 private:
  TalkerNpuDecoder* talker_ = nullptr;
  PredictorNpuDecoder* predictor_ = nullptr;
};

}  // namespace mbufo::core::npu
