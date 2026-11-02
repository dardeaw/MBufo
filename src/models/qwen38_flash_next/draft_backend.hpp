#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DRAFT_BACKEND_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DRAFT_BACKEND_HPP_

// Flash-Next MTP draft backend: session-bound IDraftBackend adapter.
//
// The flash-next MTP head consumes target trunk residuals through the
// rocm::Executor draft block, so — like the gemma4 head — it can only run
// from a live Session and is registered as session-bound ("flash-next-mtp").
// The accept policy (VerifyDraft/VerifyMtpProposal, confidence gate, prompt
// lookup interleave) and the batched verify loop (DecodeBatchImpl) stay
// model-side; only the propose and rollback phases are exposed here:
//
//   ProposeChain(...)  -> Session::DraftCatchUp(anchor, propose=true):
//                         advances the draft block past the anchor and
//                         harvests the head's greedy next token. Wired into
//                         Session::PrepareDecode (verbatim call).
//   Propose(...)       -> greedy pure-MTP chain (catch-up + repeated
//                         MtpForward steps, no lookup interleave, stops at
//                         EOS) for external unified drivers. The engine's own
//                         hot path keeps its interleaved loop (lookup,
//                         subset head, confidence gate, sampled rejection,
//                         DraftCatchUpBatch) — see the deferred-work note in
//                         draft_backend.cpp.
//   Rollback(keep,..)  -> rocm::Executor::Rollback (keeps the first `keep`
//                         tokens of the speculative batch, optionally
//                         harvesting the frontier logits). Wired into
//                         Session::FinishDecode (verbatim call). The draft
//                         block rewind (MtpRewind) stays with the driver's
//                         commit, next to the stats it feeds.
//   AcceptFeedback     -> accepted/drafted counters for accepted/round
//                         reporting (the engine keeps its own Stats).
//
// Draft KV policy: the adapter operates on the session's existing executor
// KV (F16 or blocked-Q8 shared layer per QWEN_KV_QUANT); no new KV layout.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"

namespace gufo::models::qwen38_flash_next {

class Session;
struct MtpCandidateLogits;

class FlashNextMtpDraftBackend final : public speculative::IDraftBackend {
 public:
  explicit FlashNextMtpDraftBackend(Session& session) : session_(&session) {}

  /// Session-bound construction for the "flash-next-mtp" registry factory.
  /// Returns nullptr (with *error_msg set) when the session has no MTP head.
  [[nodiscard]] static std::unique_ptr<FlashNextMtpDraftBackend> Create(
      Session& session, std::string* error_msg);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "FlashNextMtpDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  /// Greedy pure-MTP chain from a session-consistent prefix: `prompt_tokens`
  /// must be the committed history plus the anchor as its last element, and
  /// `current_pos` must equal the committed length. Returns an empty
  /// proposal when MTP is off or the prefix mismatches (no error channel on
  /// this interface). Stops at EOS.
  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;

  [[nodiscard]] bool SupportsSampledProposals() const noexcept override {
    return false;
  }

  void AcceptFeedback(
      std::span<const tokenization::TokenId> accepted,
      tokenization::TokenId correction_token) override;

  void Reset() noexcept override;

  /// Native propose entry used by Session::PrepareDecode: advances the
  /// draft block past `anchor` with the head (`candidates` non-null selects
  /// the sampled compact-logits path). Thin forward to
  /// Session::DraftCatchUp(anchor, propose=true, ...).
  [[nodiscard]] bool ProposeChain(std::int32_t anchor,
                                  MtpCandidateLogits* candidates,
                                  bool draft_subset, std::string* error_msg);

  /// Native rollback entry used by Session::FinishDecode: keeps the first
  /// `keep` tokens of the speculative batch (`keep_logits_row` harvests the
  /// frontier logits, mirroring the FinishDecode call). Thin forward to
  /// rocm::Executor::Rollback.
  [[nodiscard]] bool Rollback(std::uint32_t keep, bool keep_logits_row,
                              std::string* error_msg);

  [[nodiscard]] std::uint64_t drafted() const noexcept { return drafted_; }
  [[nodiscard]] std::uint64_t accepted() const noexcept { return accepted_; }

 private:
  Session* session_;
  std::uint64_t drafted_{0};
  std::uint64_t accepted_{0};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DRAFT_BACKEND_HPP_
