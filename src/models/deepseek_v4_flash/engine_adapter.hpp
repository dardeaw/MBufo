#ifndef MBUFO_MODELS_DEEPSEEK_V4_FLASH_ENGINE_ADAPTER_HPP_
#define MBUFO_MODELS_DEEPSEEK_V4_FLASH_ENGINE_ADAPTER_HPP_

// DeepSeek V4 Flash adapter: exposes deepseek_v4_flash::Model/Session through the L0/L1 engine abstraction
// (EngineModel / SpeculativeSession / SnapshotCapable) and registers the "deepseek_v4_flash" engine.
//
// The adapter only forwards; the model code, its kernels (which stay private, see UPSTREAM.md) and the serve path that
// drives Model/Session directly are unchanged. What it maps:
//   - Sync(prompt): the native Sync takes a full prompt but must be fed in PrefillCapacity()-sized growing prefixes
//     (the server does the same); the adapter loops over those and relies on the native prefix reuse.
//   - Forward(tokens, start_pos): DeepSeek has no "run a span at a position" entry. The adapter keeps the committed
//     token history, truncates it to start_pos, appends the tokens and Syncs the resulting prompt (native prefix
//     reuse makes this a delta prefill when start_pos == Length()).
//   - Logits(): the native logits live on the device; they are copied to a host cache lazily after every state change.
//   - SampleGreedy(): native argmax on the device logits.
//   - Tokenize(text, add_bos): native text tokenization; add_bos prepends the DeepSeek BOS token.
//   - ResidentBytes(): the GGUF is memory-mapped; the file size is reported.
//   - SnapshotCapable: forwards the native session snapshot (payload bytes, save, restore).
// Left model-side (outside L0, exposed as pass-throughs only): DSpark speculative cycles (DsparkEnabled /
// DsparkStep), chat templates and DSML tool calls. SupportsSpeculative() is false, so DecodeStep is the plain greedy
// single-token step.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/engine/attention_contract.hpp"
#include "src/core/engine/engine_batch.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"

namespace mbufo::models::deepseek_v4_flash {

class DeepSeekEngineSession final : public core::engine::SpeculativeSession, public core::engine::SnapshotCapable {
 public:
  DeepSeekEngineSession(std::shared_ptr<Model> model, std::unique_ptr<Session> inner);

  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt, std::string* error_msg = nullptr) override;
  [[nodiscard]] bool Evaluate(std::int32_t token, std::string* error_msg = nullptr) override;
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens, std::uint32_t start_pos,
                             std::string* error_msg = nullptr) override;
  [[nodiscard]] std::span<const float> Logits() const noexcept override;
  [[nodiscard]] std::uint32_t Length() const noexcept override;
  [[nodiscard]] std::int32_t SampleGreedy() override;

  [[nodiscard]] bool SupportsSpeculative() const noexcept override { return false; }
  [[nodiscard]] bool DecodeStep(std::uint32_t max_draft_tokens, core::engine::DecodeOutcome* outcome,
                                std::string* error_msg = nullptr) override;

  // SnapshotCapable
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override;
  [[nodiscard]] bool Snapshot(std::span<std::uint8_t> destination, std::string* error_msg = nullptr) const override;
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload, std::string* error_msg = nullptr) override;

  // Model-side pass-throughs (unchanged DeepSeek semantics).
  [[nodiscard]] bool DsparkEnabled() const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const;
  [[nodiscard]] std::string DecodeToken(std::int32_t token) const;
  [[nodiscard]] Session& inner() { return *inner_; }

 private:
  void InvalidateLogits() noexcept { logits_valid_ = false; }

  std::shared_ptr<Model> model_;
  std::unique_ptr<Session> inner_;
  std::vector<std::int32_t> history_;  ///< committed tokens (what Forward truncates / extends)
  mutable std::vector<float> logits_;
  mutable bool logits_valid_{false};
};

class DeepSeekEngineModel final : public core::engine::EngineModel, public core::engine::MoeCapable, public core::engine::AttentionCapable {
 public:
  [[nodiscard]] static std::shared_ptr<DeepSeekEngineModel> Load(const std::string& model_path,
                                                                 const ModelOptions& options,
                                                                 std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<core::engine::EngineSession> CreateSession(std::string* error_msg = nullptr) override;

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text, bool add_bos) const override;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept override;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept override;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept override;
  [[nodiscard]] std::string ModelName() const override;
  [[nodiscard]] core::engine::AttentionGeometry AttentionGeometryInfo() const override;
  [[nodiscard]] core::engine::AttentionKernelInfo AttentionKernelsInfo() const override;
  [[nodiscard]] core::engine::MoeGeometry MoeGeometryInfo() const override;
  [[nodiscard]] core::engine::MoeKernelInfo MoeKernelsInfo() const override;

  [[nodiscard]] std::string DecodeToken(std::int32_t token) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const;
  [[nodiscard]] const std::shared_ptr<Model>& inner() const noexcept { return model_; }

 private:
  DeepSeekEngineModel(std::shared_ptr<Model> model, std::size_t resident_bytes);

  std::shared_ptr<Model> model_;
  std::size_t resident_bytes_{0};
  core::engine::MoeGeometry moe_geometry_;
  core::engine::AttentionGeometry attention_geometry_;
};

/// The MoE contract declaration of a DeepSeek GGUF (also what mbufo serve checks when it loads one).
[[nodiscard]] core::engine::MoeGeometry DescribeMoeGeometry(const core::GgufReader& reader);
[[nodiscard]] core::engine::AttentionGeometry DescribeAttentionGeometry(const core::GgufReader& reader);
[[nodiscard]] core::engine::MoeKernelInfo DescribeMoeKernels();

}  // namespace mbufo::models::deepseek_v4_flash

#endif  // MBUFO_MODELS_DEEPSEEK_V4_FLASH_ENGINE_ADAPTER_HPP_