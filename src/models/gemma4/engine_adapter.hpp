#ifndef GUFO_MODELS_GEMMA4_ENGINE_ADAPTER_HPP_
#define GUFO_MODELS_GEMMA4_ENGINE_ADAPTER_HPP_

// Gemma4 adapter: exposes the existing gemma4::Model/Session through
// the L0/L1 engine abstraction (EngineModel / SpeculativeSession).
//
// The adapter only forwards; gemma4::Model/Session semantics are unchanged
// (no gemma4 engine-body edits).
//
// Gemma4-specific notes:
//   - Tokenize(text, add_bos): the native Model::Tokenize(text) has no
//     add_bos switch (it Encodes with defaults, i.e. no BOS). The adapter
//     keeps that exact path for add_bos=false and honors add_bos=true via
//     the shared QwenTokenizer Encode options (same convention as llama).
//   - DecodeBatch signature differs from glimmer's: glimmer takes
//     BatchItem{session, per-item token span} (variable rows per item,
//     decode or prefill-chunk), while gemma4 takes parallel arrays
//     (span<Session*>, span<const int32_t>) with exactly one decode token
//     per session. The adapter static DecodeBatch below takes adapter
//     sessions and translates: size-match validation, same-model check,
//     inner unwrap, then the native (Session*, tokens) call. MTP-enabled
//     sessions stay refused by the native core (use Evaluate instead).
//   - Forward(tokens, start_pos) maps onto the native Forward with the
//     default all_logits=false; Sync is the native incremental-prefix Sync.
//   - MTP (EnableMtp / MtpEnabled / SpeculativeRound) is an MTP extension
//     point, not part of the L0 base: exposed here as pass-throughs only.
//     CommitMtpRound is private to gemma4::Session (reachable only from
//     inside SpeculativeRound), so the adapter cannot forward it without
//     touching the gemma4 engine body, which is out of scope.
//   - SupportsSpeculative() is false: the native SpeculativeRound needs an
//     MTP head attached via EnableMtp(drafts_path), and EngineLoadRequest
//     cannot supply a drafts file; the session-bound GemmaMtpDraftBackend
//     likewise builds only from a live MTP-enabled Session, not from a
//     registry request. Owners of a drafts file can still reach the native
//     round through SpeculativeRound()/EnableMtp() below. DecodeStep is
//     therefore the plain greedy single-token fallback (sample the current
//     logits, Evaluate it), identical in shape to the llama/glimmer pilots.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/engine/engine_batch.hpp"
#include "src/models/gemma4/engine.hpp"

namespace gufo::models::gemma4 {

class GemmaEngineSession final : public core::engine::SpeculativeSession {
 public:
  GemmaEngineSession(std::shared_ptr<Model> model,
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

  // No drafts file can be conjured from the L1 DecodeStep contract (the
  // native SpeculativeRound needs EnableMtp(drafts_path) first, see above),
  // so this is the plain greedy single-token fallback.
  [[nodiscard]] bool SupportsSpeculative() const noexcept override {
    return false;
  }
  [[nodiscard]] bool DecodeStep(std::uint32_t max_draft_tokens,
                                core::engine::DecodeOutcome* outcome,
                                std::string* error_msg = nullptr) override;

  // Batch bridge: adapter sessions in, native (Session*, tokens) out.
  // sessions.size() must equal tokens.size(); every session must be live
  // and share one model (the native core runs one mixed-batch launch).
  [[nodiscard]] static bool DecodeBatch(
      std::span<GemmaEngineSession*> sessions,
      std::span<const std::int32_t> tokens,
      std::string* error_msg = nullptr);

  // L0-external gemma4 pass-throughs (unchanged gemma4 semantics).
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  // MTP extension point (outside L0): attaches the draft head from a
  // drafts GGUF, then runs native speculative rounds through it.
  [[nodiscard]] bool EnableMtp(const std::string& drafts_path,
                               std::string* error_msg = nullptr);
  [[nodiscard]] bool MtpEnabled() const noexcept;
  [[nodiscard]] bool SpeculativeRound(std::int32_t last_token,
                                      std::uint32_t n_draft,
                                      std::vector<std::int32_t>* out,
                                      std::string* error_msg = nullptr);

 private:
  std::shared_ptr<Model> model_;
  std::unique_ptr<Session> inner_;
};

class GemmaEngineModel final : public core::engine::EngineModel {
 public:
  [[nodiscard]] static std::shared_ptr<GemmaEngineModel> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);
  [[nodiscard]] static std::shared_ptr<GemmaEngineModel> Load(
      const std::string& model_path, std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<core::engine::EngineSession> CreateSession(
      std::string* error_msg = nullptr) override;

  [[nodiscard]] std::vector<std::int32_t> Tokenize(
      std::string_view text, bool add_bos) const override;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept override;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept override;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept override;
  [[nodiscard]] std::string ModelName() const override;

  // L0-external gemma4 pass-throughs (unchanged gemma4 semantics).
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] const std::shared_ptr<Model>& inner() const noexcept {
    return model_;
  }

 private:
  explicit GemmaEngineModel(std::shared_ptr<Model> model);

  std::shared_ptr<Model> model_;
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_ENGINE_ADAPTER_HPP_
