#ifndef GUFO_CORE_ENGINE_ENGINE_BACKEND_HPP_
#define GUFO_CORE_ENGINE_ENGINE_BACKEND_HPP_

// L0 engine surface: the Load -> CreateSession -> Sync -> Evaluate/Forward
// -> Logits/Length contract shared by all five engines (llama, glimmer,
// gemma4, qwen3.8-flash-next, qwen3.8-27b).
//
// Deliberately NOT unified here (stays model-side):
//   - SSM state, vision sidecars, three-format batch layouts, and the 27B
//     session-less runner shape;
//   - draft-weight loading, acceptance policy, target verify loops.
// Virtual dispatch lives only in this host-orchestration layer; device
// kernels stay non-virtual.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::core::engine {

/// One conversation's decode context. Sync feeds a prompt, Evaluate appends
/// one token, Forward runs a span starting at a position; the logits of the
/// last token are kept for sampling.
class EngineSession {
 public:
  virtual ~EngineSession() = default;

  EngineSession(const EngineSession&) = delete;
  EngineSession& operator=(const EngineSession&) = delete;

  /// Reprocesses the prompt (engines may chunk internally).
  [[nodiscard]] virtual bool Sync(std::span<const std::int32_t> prompt,
                                  std::string* error_msg = nullptr) = 0;
  /// Appends one token and runs one forward step.
  [[nodiscard]] virtual bool Evaluate(std::int32_t token,
                                      std::string* error_msg = nullptr) = 0;
  /// Runs a forward over `tokens` starting at `start_pos`.
  [[nodiscard]] virtual bool Forward(std::span<const std::int32_t> tokens,
                                     std::uint32_t start_pos,
                                     std::string* error_msg = nullptr) = 0;
  /// Logits of the last committed token.
  [[nodiscard]] virtual std::span<const float> Logits() const noexcept = 0;
  /// Number of committed tokens in this session.
  [[nodiscard]] virtual std::uint32_t Length() const noexcept = 0;
  /// Greedy sample from the current logits (deterministic, no RNG needed).
  [[nodiscard]] virtual std::int32_t SampleGreedy() = 0;

 protected:
  EngineSession() = default;
};

/// Weight/table owner. Sessions are per-conversation views over one model.
class EngineModel {
 public:
  virtual ~EngineModel() = default;

  EngineModel(const EngineModel&) = delete;
  EngineModel& operator=(const EngineModel&) = delete;

  [[nodiscard]] virtual std::unique_ptr<EngineSession> CreateSession(
      std::string* error_msg = nullptr) = 0;

  [[nodiscard]] virtual std::vector<std::int32_t> Tokenize(
      std::string_view text, bool add_bos) const = 0;
  [[nodiscard]] virtual std::uint32_t VocabSize() const noexcept = 0;
  [[nodiscard]] virtual std::uint32_t MaxContext() const noexcept = 0;
  [[nodiscard]] virtual std::size_t ResidentBytes() const noexcept = 0;
  [[nodiscard]] virtual std::string ModelName() const = 0;

 protected:
  EngineModel() = default;
};

}  // namespace gufo::core::engine

#endif  // GUFO_CORE_ENGINE_ENGINE_BACKEND_HPP_
