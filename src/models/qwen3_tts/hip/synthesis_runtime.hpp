#ifndef MBUFO_MODELS_QWEN3_TTS_HIP_SYNTHESIS_RUNTIME_HPP_
#define MBUFO_MODELS_QWEN3_TTS_HIP_SYNTHESIS_RUNTIME_HPP_

#include <cstddef>
#include <memory>
#include <string>

#include "src/models/qwen3_tts/config.hpp"
#include "src/models/qwen3_tts/synthesis.hpp"

namespace mbufo::models::qwen3_tts::hip {

/// End-to-end native HIP synthesis for one supported Qwen3-TTS 12Hz 1.7B
/// variant.
class SynthesisHipRuntime {
public:
  ~SynthesisHipRuntime();

  SynthesisHipRuntime(const SynthesisHipRuntime&) = delete;
  SynthesisHipRuntime& operator=(const SynthesisHipRuntime&) = delete;
  SynthesisHipRuntime(SynthesisHipRuntime&&) = delete;
  SynthesisHipRuntime& operator=(SynthesisHipRuntime&&) = delete;

  [[nodiscard]] static std::unique_ptr<SynthesisHipRuntime> Create(
      const std::string& model_root, std::size_t maximum_context_tokens,
      ModelVariant variant, std::string* error = nullptr);

  [[nodiscard]] bool Generate(const SynthesisRequest& request,
                              const CancellationCheck& is_cancelled,
                              SynthesisResult* result,
                              std::string* error = nullptr);

private:
  struct Impl;
  explicit SynthesisHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace mbufo::models::qwen3_tts::hip

#endif  // MBUFO_MODELS_QWEN3_TTS_HIP_SYNTHESIS_RUNTIME_HPP_
