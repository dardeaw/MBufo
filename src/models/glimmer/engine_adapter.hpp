#ifndef GUFO_MODELS_GLIMMER_ENGINE_ADAPTER_HPP_
#define GUFO_MODELS_GLIMMER_ENGINE_ADAPTER_HPP_

// Glimmer adapter: exposes the existing glimmer::Model/Session through
// the L0/L1 engine abstraction (EngineModel / SpeculativeSession).
//
// The adapter only forwards; glimmer::Model/Session semantics are unchanged
// (no glimmer engine-body edits).
//
// Glimmer-specific notes:
//   - Session::Verify(tokens,start,vbest) COMMITS like Forward: it runs one
//     batched forward, writes KV rows at [start_pos, start_pos+n),
//     advances Length() by n, and reports the greedy token for every
//     position (speculative verification primitive; the accept path
//     overwrites with its own Forward calls). It is therefore NOT a
//     read-only peek and can NOT back DecodeStep's single-commit contract,
//     so DecodeStep is the plain greedy single-token fallback (sample
//     current logits, Evaluate it), identical in shape to the llama pilot.
//     (An earlier draft of this adapter assumed read-only Verify; the
//     registry pilot empirically disproves it: Length moves by k.)
//   - TapSink / EnableDraftTaps / CopyTappedRow are draft-side feature taps
//     for DFlash priming; they stay outside L0 and are exposed here as
//     pass-throughs only.
//   - SwaShiftedKV is an internal blocked-KV SWA pointer shift inside the
//     forward path (same convention as gemma4: advance the cache base past
//     rows that fell out of the window, rebase the position); nothing to
//     forward at the adapter boundary.
//   - Session::SpeculativeRound(IDraftBackend&) needs a caller-owned
//     glimmer-dflash draft backend (dynamic_cast-checked) with matching
//     EnableDraftTaps layers, which EngineLoadRequest cannot supply, so
//     SupportsSpeculative() is false; owners of a draft backend can still
//     reach the native round through SpeculativeRound()/inner() below.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/engine/engine_batch.hpp"
#include "src/models/glimmer/engine.hpp"

namespace gufo::speculative {
class IDraftBackend;
}  // namespace gufo::speculative

namespace gufo::models::glimmer {

class GlimmerEngineSession final : public core::engine::SpeculativeSession {
 public:
  GlimmerEngineSession(std::shared_ptr<Model> model,
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

  // No draft backend can be conjured from the L1 DecodeStep contract (the
  // native round needs a caller-owned glimmer-dflash backend, see above),
  // so this is the plain greedy single-token fallback.
  [[nodiscard]] bool SupportsSpeculative() const noexcept override {
    return false;
  }
  [[nodiscard]] bool DecodeStep(std::uint32_t max_draft_tokens,
                                core::engine::DecodeOutcome* outcome,
                                std::string* error_msg = nullptr) override;

  // L0-external glimmer pass-throughs (unchanged glimmer semantics).
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  // Committing batched verify: fills vbest with the greedy token of every
  // position and advances Length() by tokens.size() (Forward mechanics;
  // NOT a DecodeStep substitute, NOT a read-only peek).
  [[nodiscard]] bool Verify(std::span<const std::int32_t> tokens,
                            std::uint32_t start_pos,
                            std::vector<std::int32_t>& vbest,
                            std::string* error_msg = nullptr);
  // Draft-side tap staging (DFlash priming); outside L0.
  [[nodiscard]] bool EnableDraftTaps(std::span<const std::uint32_t> layers,
                                     std::string* error_msg = nullptr);
  [[nodiscard]] bool CopyTappedRow(std::span<float> host_out,
                                   std::string* error_msg = nullptr) const;
  // Native speculative round for callers that own a glimmer-dflash draft
  // backend (requires EnableDraftTaps with the backend's tap layers and
  // prefix.size() == Length()).
  [[nodiscard]] bool SpeculativeRound(speculative::IDraftBackend& draft,
                                      std::span<const std::int32_t> prefix,
                                      std::uint32_t n_draft,
                                      std::vector<std::int32_t>* out,
                                      std::string* error_msg = nullptr);

 private:
  std::shared_ptr<Model> model_;
  std::unique_ptr<Session> inner_;
};

class GlimmerEngineModel final : public core::engine::EngineModel {
 public:
  [[nodiscard]] static std::shared_ptr<GlimmerEngineModel> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);
  [[nodiscard]] static std::shared_ptr<GlimmerEngineModel> Load(
      const std::string& model_path, std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<core::engine::EngineSession> CreateSession(
      std::string* error_msg = nullptr) override;

  [[nodiscard]] std::vector<std::int32_t> Tokenize(
      std::string_view text, bool add_bos) const override;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept override;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept override;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept override;
  [[nodiscard]] std::string ModelName() const override;

  // L0-external glimmer pass-throughs (unchanged glimmer semantics).
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] const std::shared_ptr<Model>& inner() const noexcept {
    return model_;
  }

 private:
  explicit GlimmerEngineModel(std::shared_ptr<Model> model);

  std::shared_ptr<Model> model_;
};

}  // namespace gufo::models::glimmer

#endif  // GUFO_MODELS_GLIMMER_ENGINE_ADAPTER_HPP_
