#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_ADAPTER_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_ADAPTER_HPP_

// Flash-Next adapter: exposes the existing qwen38_flash_next::Model/Session
// through the L0/L1 engine abstraction (EngineModel / SpeculativeSession).
//
// The adapter only forwards; flash-next engine semantics are unchanged
// (no engine-body edits). Flash is the serve主力 with the thickest native
// surface of the four registry adapters, so every thick point is bridged
// inside this translation unit:
//
//   - Model::CreateSession(mode, max_context): L0 CreateSession() takes no
//     arguments, so the parameterless override builds an autoregressive
//     session at the model's full context. Registry requests carry no mode,
//     and autoregressive is the safe default (it never allocates
//     speculative rollback rows). Callers that arm MTP via
//     ModelOptions::mtp_model_path at Load should use the mode/max_context
//     overload below with kSpeculative, mirroring session_test.
//   - Session::Sync(prompt, err, next): the L0 override forwards with an
//     empty `next` span; the three-argument overload below exposes the
//     native n-gram prefetch hint.
//   - No native Forward/Length/SampleGreedy: Forward(tokens, start_pos) is
//     emulated with one Sync over prefix+tokens (exact for extension,
//     prefix-reuse on overlap, reset+refeed on divergence — all inside the
//     native Sync contract); Length() reads Position(); SampleGreedy()
//     samples the current logits through the session-owned greedy
//     SamplerState (default config, temperature 0: no RNG draw, no penalty
//     history dependence; Sync resets history, Evaluate/DecodeStep accept
//     committed tokens, matching native sampler discipline).
//   - DecodeStep(max_tokens, sampler, result, ...): the adapter owns one
//     greedy SamplerState per session and maps the L1 outcome onto the
//     native DecodeResult (token = last committed, tokens_committed =
//     result.tokens.size()). A zero budget means the model-configured
//     max_draft_tokens. stop_at_eos stays true with occupancy 1: L1 carries
//     neither, and both are accounting-only under greedy verification.
//     An immediate stop (empty native commit) is committed explicitly,
//     so the L1 step still advances by one stop token like the sibling
//     fallbacks.
//   - SyncThrough(prompt, boundary, checkpoint): pass-through returning the
//     native SessionSnapshot object (native constraints: boundary must
//     extend the frontier, prompt tail past it <= 8 tokens).
//   - Snapshots: the session derives SnapshotCapable. PayloadBytes reads
//     Session::SnapshotBytes(); Snapshot() materializes (kMaterialized, so
//     the payload is self-contained) and checks the exact size; the native
//     payload carries kSnapshotPayloadVersion (currently 20) plus the KV
//     format, so RestoreSnapshot refuses cross-version/cross-format
//     payloads without adapter-side parsing.
//   - MTP: pass-throughs only, never in the base — Model::HasMtp (armed at
//     Load via ModelOptions::mtp_model_path), the static
//     MtpConfidenceGate, ResetDraftPolicy, and Statistics. Two pieces stay
//     unreachable without an engine-body edit: Session::MtpEnabled() and
//     the draft_length_ controller are both private to Session, so the
//     adapter observes speculation through Statistics()
//     (cycles/drafted/accepted) and DecodeStep's tokens_committed instead.
//   - SupportsSpeculative() is true: unlike the gemma4/glimmer pilots (whose
//     speculative entries need a drafts file / draft backend the registry
//     request cannot supply, hence false + hand-rolled fallback), the L1
//     DecodeStep maps 1:1 onto the native draft-verify DecodeStep — the
//     same entry the engine body uses, degrading to ordinary single-token
//     decode when no MTP sidecar is armed. Multi-token commits additionally
//     require mtp_model_path at Load (EngineLoadRequest carries no such
//     field, so registry-loaded models run single-token steps through the
//     same native path).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/engine/engine_batch.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace gufo::models::qwen38_flash_next {

class FlashNextEngineSession final : public core::engine::SpeculativeSession,
                                     public core::engine::SnapshotCapable {
 public:
  FlashNextEngineSession(std::shared_ptr<Model> model,
                         std::unique_ptr<Session> inner,
                         std::size_t default_drafts);

  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr) override;
  // Native prefetch-hint overload: `next` names the tokens the following
  // Sync will append, so their n-gram rows can be read during this one; it
  // never changes the result.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::span<const std::int32_t> next,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr) override;
  // Emulated with one Sync over prefix+tokens (see file doc); exact for
  // extension, prefix-reuse on overlap, reset+refeed on divergence.
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             std::uint32_t start_pos,
                             std::string* error_msg = nullptr) override;
  [[nodiscard]] std::span<const float> Logits() const noexcept override;
  [[nodiscard]] std::uint32_t Length() const noexcept override;
  [[nodiscard]] std::int32_t SampleGreedy() override;
  // Committed token history (for tests reconstructing multi-token
  // DecodeStep commits through the L1 outcome, which names only the last
  // token).
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept;

  // True: DecodeStep below IS the native draft-verify entry (see file doc).
  [[nodiscard]] bool SupportsSpeculative() const noexcept override {
    return true;
  }
  [[nodiscard]] bool DecodeStep(std::uint32_t max_draft_tokens,
                                core::engine::DecodeOutcome* outcome,
                                std::string* error_msg = nullptr) override;

  // SnapshotCapable: versioned native payload (kSnapshotPayloadVersion).
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override;
  [[nodiscard]] bool Snapshot(std::span<std::uint8_t> destination,
                              std::string* error_msg = nullptr) const override;
  [[nodiscard]] bool RestoreSnapshot(
      std::span<const std::uint8_t> payload,
      std::string* error_msg = nullptr) override;

  // Batch bridges onto the native DecodeBatch/EvaluateBatch statics using
  // each session's owned sampler (keeps sampler history consistent with
  // the serial DecodeStep path above).
  struct DecodeRequest {
    FlashNextEngineSession* session;
    std::size_t max_tokens;
    Session::DecodeResult* result;
    bool stop_at_eos{true};
    Session::BatchOutcome* outcome{nullptr};
  };
  [[nodiscard]] static bool DecodeBatch(
      std::span<const DecodeRequest> requests,
      std::string* error_msg = nullptr);
  struct AdvanceRequest {
    FlashNextEngineSession* session;
    std::int32_t token;
    Session::BatchOutcome* outcome{nullptr};
  };
  [[nodiscard]] static bool EvaluateBatch(
      std::span<const AdvanceRequest> requests,
      std::string* error_msg = nullptr);

  // L0-external flash-next pass-throughs (unchanged engine semantics).
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool HasMtp() const noexcept;
  // Prefill through a stable text boundary, returning that boundary's
  // checkpoint while the live session reaches `prompt` (native constraints:
  // boundary must extend the frontier, prompt tail past it <= 8 tokens).
  [[nodiscard]] bool SyncThrough(std::span<const std::int32_t> prompt,
                                 std::uint32_t boundary,
                                 std::unique_ptr<SessionSnapshot>* checkpoint,
                                 std::string* error_msg = nullptr);
  void ResetDraftPolicy();
  [[nodiscard]] const Session::SpeculativeStats& Statistics() const noexcept;

 private:
  void ResetSamplerHistory(std::span<const std::int32_t> tokens);

  std::shared_ptr<Model> model_;
  std::unique_ptr<Session> inner_;
  sampling::SamplerState sampler_;
  std::size_t default_drafts_;
};

class FlashNextEngineModel final : public core::engine::EngineModel {
 public:
  [[nodiscard]] static std::shared_ptr<FlashNextEngineModel> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);
  [[nodiscard]] static std::shared_ptr<FlashNextEngineModel> Load(
      const std::string& model_path, std::string* error_msg = nullptr);

  // L0 default: autoregressive session at the model's full context (see
  // file doc). MTP-armed callers should use the overload below.
  [[nodiscard]] std::unique_ptr<core::engine::EngineSession> CreateSession(
      std::string* error_msg = nullptr) override;
  [[nodiscard]] std::unique_ptr<core::engine::EngineSession> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(
      std::string_view text, bool add_bos) const override;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept override;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept override;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept override;
  [[nodiscard]] std::string ModelName() const override;

  // L0-external flash-next pass-throughs (unchanged engine semantics).
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool HasMtp() const noexcept;
  [[nodiscard]] static float MtpConfidenceGate() noexcept;
  [[nodiscard]] const std::shared_ptr<Model>& inner() const noexcept {
    return model_;
  }
  [[nodiscard]] std::size_t default_drafts() const noexcept {
    return options_.max_draft_tokens;
  }

 private:
  explicit FlashNextEngineModel(std::shared_ptr<Model> model,
                                ModelOptions options);

  std::shared_ptr<Model> model_;
  ModelOptions options_;
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_ADAPTER_HPP_
