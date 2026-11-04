#ifndef GUFO_MODELS_LLAMA_ENGINE_ADAPTER_HPP_
#define GUFO_MODELS_LLAMA_ENGINE_ADAPTER_HPP_

// Llama pilot adapter: exposes the existing llama::Model/Session through
// the L0/L1 engine abstraction (EngineModel / SpeculativeSession).
//
// The adapter only forwards; llama::Model/Session semantics are unchanged
// (no llama engine-body edits). Llama has no speculative machinery, so
// SupportsSpeculative() is false and DecodeStep is the plain greedy
// single-token fallback (sample current logits, Evaluate it).

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/engine/engine_batch.hpp"
#include "src/models/llama/engine.hpp"

namespace gufo::models::llama {

class LlamaEngineSession final : public core::engine::SpeculativeSession {
 public:
  LlamaEngineSession(std::shared_ptr<Model> model,
                     std::unique_ptr<Session> inner);

  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr) override;
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr) override;
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             std::uint32_t start_pos,
                             std::string* error_msg = nullptr) override;
  [[nodiscard]] std::span<const float> Logits() const noexcept override;
  [[nodiscard]] std::uint32_t Length() const noexcept override;
  [[nodiscard]] std::int32_t SampleGreedy() override;

  [[nodiscard]] bool SupportsSpeculative() const noexcept override {
    return false;
  }
  [[nodiscard]] bool DecodeStep(std::uint32_t max_draft_tokens,
                                core::engine::DecodeOutcome* outcome,
                                std::string* error_msg = nullptr) override;

  // L0-external llama pass-throughs (unchanged llama semantics).
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;

 private:
  std::shared_ptr<Model> model_;
  std::unique_ptr<Session> inner_;
};

class LlamaEngineModel final : public core::engine::EngineModel {
 public:
  [[nodiscard]] static std::shared_ptr<LlamaEngineModel> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);
  [[nodiscard]] static std::shared_ptr<LlamaEngineModel> Load(
      const std::string& model_path, std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<core::engine::EngineSession> CreateSession(
      std::string* error_msg = nullptr) override;

  [[nodiscard]] std::vector<std::int32_t> Tokenize(
      std::string_view text, bool add_bos) const override;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept override;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept override;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept override;
  [[nodiscard]] std::string ModelName() const override;

  // L0-external llama pass-throughs (unchanged llama semantics).
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] const std::shared_ptr<Model>& inner() const noexcept {
    return model_;
  }

 private:
  explicit LlamaEngineModel(std::shared_ptr<Model> model);

  std::shared_ptr<Model> model_;
};

}  // namespace gufo::models::llama

#endif  // GUFO_MODELS_LLAMA_ENGINE_ADAPTER_HPP_
