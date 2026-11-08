#ifndef GUFO_MODELS_GLIMMER_DRAFT_BACKEND_HPP_
#define GUFO_MODELS_GLIMMER_DRAFT_BACKEND_HPP_

// Glimmer DFlash draft backend: EAGLE-style target-bound feature draft.
//
// The muse-glimmer companion draft (dflash-kquant, depth-pruned dense
// decoder) consumes FUSED TARGET hidden states — `fc.weight` maps the
// concatenated tapped-layer features [n_taps * hidden] to the model width
// — and predicts with the TARGET's output head, so temperature-0 greedy
// proposals only. Draft KV lives on the shared blocked-Q8 grid (see
// draft_engine.hpp); no F16 mirror is kept.
//
// Lifecycle mapping onto IDraftBackend:
//   TargetHiddenLayerIds() -> zero-based tapped target layer outputs
//                             (artifact 1-based layer-input numbering
//                             normalized with the 27B N-1 convention).
//   PrimeTargetContext(...)   -> prompt-first priming: fuses every prompt
//                             row and runs the draft transformer over them,
//                             building the draft KV for positions [0, P).
//   AppendTargetContext(...)  -> commit path: rewinds past the last
//                             speculative chain, then encodes the accepted
//                             rows plus the bonus row (positions must extend
//                             the draft length exactly; re-encoding accepted
//                             rows from true features keeps the draft KV
//                             position-exact, no self-loop drift).
//   UpdateTargetHidden(...)   -> unused (base no-op): proposals sample
//                             from the current draft logits, so no per-step
//                             anchor injection is needed; history arrives
//                             via Prime/Append.
//   Propose(...)              -> samples the current draft head (position
//                             P), then chains self-loop rows
//                             (input = previous draft output hidden) for
//                             the remaining drafts. Empty when not primed,
//                             out of room, or max_tokens == 0.
//   AcceptFeedback(...)       -> records drafted/accepted counters only
//                             (positioning already happened in Append).
//   Rollback semantics        -> the Append rewind; no target KV touched.
//   Snapshots                 -> unsupported (draft sessions expose no
//                             continuation snapshots); base throws.
//   RequiresTargetHiddenStates() == true: every Propose consumes fused
//                             target features (prompt-first, then commits).

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"
#include "src/models/glimmer/draft_engine.hpp"
#include "src/models/glimmer/engine.hpp"

namespace gufo::models::glimmer {

/// Target-bound EAGLE-style DFlash draft backend (greedy only).
class GlimmerDflashDraftBackend final : public speculative::IDraftBackend {
 public:
  /// Creates the backend: loads the draft artifact, normalizes the tap
  /// list, binds the target output head and opens a draft session.
  /// Returns nullptr and sets *error_msg on failure.
  [[nodiscard]] static std::unique_ptr<GlimmerDflashDraftBackend> Create(
      std::shared_ptr<const Model> target,
      const std::string& draft_model_path, std::uint32_t max_context,
      std::uint32_t max_draft_tokens, std::string* error_msg = nullptr);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "GlimmerDflashDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] std::span<const std::uint32_t> TargetHiddenLayerIds()
      const noexcept override {
    return tap_layers_;
  }

  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;

  [[nodiscard]] bool AppendTargetContext(
      const speculative::DraftTargetContext& context,
      std::uint32_t position) override;

  /// Samples the current draft head (position Length) and chains
  /// self-loop rows for the rest. `prompt_tokens`/`current_pos` are
  /// advisory (position must equal the draft length); temperature-0 path
  /// only, sampled proposals are unsupported.
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

  [[nodiscard]] std::uint64_t drafted() const noexcept { return drafted_; }
  [[nodiscard]] std::uint64_t accepted() const noexcept { return accepted_; }
  [[nodiscard]] bool primed() const noexcept { return primed_; }

 private:
  GlimmerDflashDraftBackend(std::shared_ptr<draft::DraftModel> draft_model,
                            std::unique_ptr<draft::DraftSession> session,
                            std::vector<std::uint32_t> tap_layers,
                            std::uint32_t max_draft_tokens,
                            std::uint32_t feat_width)
      : draft_model_(std::move(draft_model)),
        draft_session_(std::move(session)),
        tap_layers_(std::move(tap_layers)),
        max_draft_tokens_(max_draft_tokens),
        feat_width_(feat_width) {}

  std::shared_ptr<draft::DraftModel> draft_model_;
  std::unique_ptr<draft::DraftSession> draft_session_;
  std::vector<std::uint32_t> tap_layers_;
  std::uint32_t max_draft_tokens_;
  std::uint32_t feat_width_;
  std::uint32_t propose_pos_{0};
  std::uint32_t proposed_{0};
  bool proposal_active_{false};
  bool primed_{false};
  std::uint64_t drafted_{0};
  std::uint64_t accepted_{0};
};

}  // namespace gufo::models::glimmer

#endif  // GUFO_MODELS_GLIMMER_DRAFT_BACKEND_HPP_
