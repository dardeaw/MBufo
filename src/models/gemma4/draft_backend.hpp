#ifndef GUFO_MODELS_GEMMA4_DRAFT_BACKEND_HPP_
#define GUFO_MODELS_GEMMA4_DRAFT_BACKEND_HPP_

// Gemma4 MTP draft backend: session-bound IDraftBackend adapter.
//
// The Gemma 4 MTP draft head (MtpDrafter, mtp.hpp) cross-attends the target
// session's live KV caches, so — unlike the 27B DFlash2 backend — it cannot
// be built from a GGUF path alone. It is constructed from a live Session
// whose MTP head is already attached (Session::EnableMtp), and registered
// under "gemma4-mtp" as a session-bound backend (see draft_registry.hpp).
//
// Lifecycle mapping onto IDraftBackend:
//   Propose(...)          -> ProposeDrafts(): runs the MTP Step chain and
//                            returns greedy draft tokens (temperature-0 path;
//                            sampled proposals are unsupported, matching the
//                            engine which only speculates greedily).
//   AcceptFeedback(...)   -> records accepted/drafted counters for
//                            accepted/round reporting. The engine commit
//                            itself (KV length adjust) stays in
//                            Session::SpeculativeRound via CommitMtpRound.
//   Rollback semantics    -> no-op by construction: the verify Forward
//                            writes KV rows [old .. old+n] that stay valid,
//                            and the commit only moves `length` back down.
//                            No target KV is ever discarded.
//   Snapshots             -> unsupported (Gemma4 sessions expose no
//                            continuation snapshots); base throws.
//   RequiresTargetHiddenStates() == true: every Step consumes the target
//                            hidden state of the preceding token.
//
// Draft KV policy: the adapter operates on the session's existing KV
// caches; no new KV layout is introduced.
//
// Session::SpeculativeRound is rewired to drive the round through this
// backend plus GemmaTargetVerify below, with every numeric op moved
// verbatim (zero behavior change): ProposeDrafts -> Verify -> greedy
// compare (accept policy stays model-side) -> CommitMtpRound.

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"

namespace gufo::models::gemma4 {

class Session;

/// Per-round timestamps shared by ProposeDrafts/Verify so the
/// GEMMA4_MTP_TIMING debug print keeps reporting draft vs verify latency.
struct MtpRoundTimes {
  std::chrono::steady_clock::time_point t_start{};
  std::chrono::steady_clock::time_point t_after_draft{};
};

/// Thin TargetVerify wrapper over the Gemma4 target verify path: one
/// batched target Forward over [cur, draft...] plus argmax, returning the
/// target's preferred tokens (vbest[0..n_draft]). The greedy accept
/// comparison stays with the caller (model-side accept policy).
struct GemmaTargetVerify {
  explicit GemmaTargetVerify(Session& session) : session_(&session) {}

  /// Verifies [cur, draft...] in a single target forward. `vbest` must hold
  /// at least 1 + draft.size() entries (clamped to 8 by the engine cap).
  /// `vtoks` receives the verified token block [cur, draft...].
  [[nodiscard]] bool Verify(std::int32_t cur,
                            const std::vector<std::int32_t>& draft,
                            std::uint32_t old_length, std::int32_t* vbest,
                            std::vector<std::int32_t>* vtoks,
                            std::string* error_msg,
                            const MtpRoundTimes* times = nullptr);

 private:
  Session* session_;
};

/// Session-bound MTP draft backend (greedy only).
class GemmaMtpDraftBackend final : public speculative::IDraftBackend {
 public:
  explicit GemmaMtpDraftBackend(Session& session,
                                std::uint32_t max_draft_tokens = 4)
      : session_(&session), max_draft_tokens_(max_draft_tokens) {}

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "GemmaMtpDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  /// Greedy propose from the session's committed state. `prompt_tokens` /
  /// `current_pos` are advisory: the session state is authoritative (the
  /// head reads the live target hidden of the current token). Returns an
  /// empty proposal when MTP is not enabled.
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

  /// Native-shape greedy propose used by Session::SpeculativeRound: drafts
  /// up to n_draft tokens after `cur` at KV position `old_length`. Moved
  /// verbatim from the draft loop of SpeculativeRound (same
  /// MtpDrafter::Step sequence). `times` feeds the GEMMA4_MTP_TIMING print.
  [[nodiscard]] bool ProposeDrafts(std::int32_t cur, std::uint32_t n_draft,
                                   std::uint32_t old_length,
                                   std::vector<std::int32_t>* draft,
                                   std::string* error_msg,
                                   MtpRoundTimes* times = nullptr);

  /// Records one driven round (used by Session::SpeculativeRound).
  void RecordRound(std::size_t drafted, std::size_t accepted) noexcept {
    drafted_ += drafted;
    accepted_ += accepted;
  }

  [[nodiscard]] std::uint64_t drafted() const noexcept { return drafted_; }
  [[nodiscard]] std::uint64_t accepted() const noexcept { return accepted_; }

 private:
  Session* session_;
  std::uint32_t max_draft_tokens_;
  std::uint64_t drafted_{0};
  std::uint64_t accepted_{0};
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_DRAFT_BACKEND_HPP_
