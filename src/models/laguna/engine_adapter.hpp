#ifndef MBUFO_MODELS_LAGUNA_ENGINE_ADAPTER_HPP_
#define MBUFO_MODELS_LAGUNA_ENGINE_ADAPTER_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/engine/attention_contract.hpp"
#include "src/core/engine/engine_batch.hpp"
#include "src/core/engine/moe_contract.hpp"
#include "src/models/laguna/engine.hpp"

namespace mbufo::models::laguna {

class LagunaEngineSession final : public core::engine::SpeculativeSession {
 public:
  LagunaEngineSession(std::shared_ptr<Model> model,
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

 private:
  std::shared_ptr<Model> model_;
  std::unique_ptr<Session> inner_;
};

class LagunaEngineModel final : public core::engine::EngineModel,
                                public core::engine::AttentionCapable,
                                public core::engine::MoeCapable {
 public:
  [[nodiscard]] static std::shared_ptr<LagunaEngineModel> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<core::engine::EngineSession> CreateSession(
      std::string* error_msg = nullptr) override;

  [[nodiscard]] std::vector<std::int32_t> Tokenize(
      std::string_view text, bool add_bos) const override;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept override;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept override;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept override;
  [[nodiscard]] std::string ModelName() const override;

  [[nodiscard]] core::engine::AttentionGeometry AttentionGeometryInfo() const override;
  [[nodiscard]] core::engine::AttentionKernelInfo AttentionKernelsInfo() const override;
  [[nodiscard]] core::engine::MoeGeometry MoeGeometryInfo() const override;
  [[nodiscard]] core::engine::MoeKernelInfo MoeKernelsInfo() const override;

 private:
  explicit LagunaEngineModel(std::shared_ptr<Model> model);
  std::shared_ptr<Model> model_;
};

}  // namespace mbufo::models::laguna

#endif  // MBUFO_MODELS_LAGUNA_ENGINE_ADAPTER_HPP_
