#ifndef GUFO_MODELS_GLIMMER_DRAFT_ENGINE_HPP_
#define GUFO_MODELS_GLIMMER_DRAFT_ENGINE_HPP_

// DFlash draft runtime (muse-glimmer companion draft, e.g. dflash-kquant):
// an EAGLE-style feature draft. A depth-pruned dense decoder (5 layers,
// same width/dim as the target) consumes FUSED TARGET hidden states —
// `fc.weight` maps the concatenated tapped-layer features [n_taps*hidden]
// to the model width — and predicts with the TARGET's output head, so the
// artifact carries no embedding and no vocabulary of its own.
//
// Per-round flow (temperature-0, greedy only):
//   PrimeRows  prompt fused features (prompt-first) build the draft KV;
//   ExtendRow  fuses one anchor row, self-loop rows chain autoregressively;
//   ScoreLast  scores the last row through the target head for sampling.
// The backend drives propose/verify/commit through these primitives plus
// the shared blocked-Q8 draft KV (core blocked_kv); no F16 mirror is kept.
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/models/glimmer/draft_model.hpp"
#include "src/models/glimmer/weights.hpp"

namespace gufo::models::glimmer::draft {

struct DraftModelOptions {
  std::uint32_t max_context = 32768;
};

/// Target output head bound by the backend (non-owning views into the
/// target model, stable for the model lifetime).
struct TargetHead {
  TensorRef output;
  float logit_scale{1.0F};
  float final_logit_softcapping{0.0F};
  std::uint32_t vocab_size{0};
};

class DraftSession;

class DraftModel final : public std::enable_shared_from_this<DraftModel> {
 public:
  ~DraftModel();
  DraftModel(const DraftModel&) = delete;
  DraftModel& operator=(const DraftModel&) = delete;

  [[nodiscard]] static std::shared_ptr<DraftModel> Load(
      const std::string& model_path, const DraftModelOptions& options,
      const std::vector<std::uint32_t>& target_layer_ids,
      std::string* error_msg = nullptr);

  /// Binds the target output head (logits live in the target vocabulary).
  /// Must be called once before CreateSession.
  void SetTargetHead(TargetHead head) { target_head_ = head; }

  [[nodiscard]] std::unique_ptr<DraftSession> CreateSession(
      std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t VocabSize() const noexcept {
    return target_head_.vocab_size;
  }
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] const std::vector<std::uint32_t>& TargetLayerIds() const noexcept {
    return target_layer_ids_;
  }
  [[nodiscard]] const TargetHead& target_head() const noexcept {
    return target_head_;
  }

 private:
  DraftModel() = default;

  DraftModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<DraftModelWeights> weights_;
  TargetHead target_head_;
  std::vector<void*> device_allocations_;
  std::vector<std::uint32_t> target_layer_ids_;

  friend class DraftSession;
};

/// One draft conversation over fused target features (all host-side spans
/// are [n][n_taps * hidden] row-major feature grids).
class DraftSession final {
 public:
  ~DraftSession();
  DraftSession(const DraftSession&) = delete;
  DraftSession& operator=(const DraftSession&) = delete;

  /// Encodes prompt feature rows [start, start+n) into the draft KV
  /// (prompt-first priming). Loops internally in kDraftChunk blocks.
  [[nodiscard]] bool PrimeRows(std::span<const float> feats, std::uint32_t n,
                               std::uint32_t start_pos,
                               std::string* error_msg = nullptr);
  /// Fuses one host feature row into the device fused staging.
  [[nodiscard]] bool FuseRows(std::span<const float> feats, std::uint32_t n,
                              std::string* error_msg = nullptr);
  /// Device pointer to the fused staging (valid after FuseRows).
  [[nodiscard]] const float* FusedStaging() const noexcept;
  /// Forwards one fused/device row at the current length (no head).
  /// `d_row` may alias the last hidden row (self-loop).
  [[nodiscard]] bool ExtendRow(const float* d_row, std::string* error_msg = nullptr);
  /// Scores the last forwarded row through the target head (for sampling).
  [[nodiscard]] bool ScoreLast(std::string* error_msg = nullptr);
  /// Device pointer to the last hidden row (self-loop input).
  [[nodiscard]] const float* LastHiddenRow() const noexcept;
  /// Moves the length marker back (used by the backend to roll back
  /// unaccepted proposals). KV rows beyond the marker stay valid and are
  /// overwritten by later forwards.
  [[nodiscard]] bool Rewind(std::uint32_t new_length,
                            std::string* error_msg = nullptr);
  [[nodiscard]] std::int32_t SampleGreedy();
  [[nodiscard]] std::uint32_t Length() const noexcept;

 private:
  DraftSession(std::shared_ptr<DraftModel> model);

  static void ForwardLayer(DraftSession& s, std::uint32_t il,
                           std::uint32_t n, std::uint32_t start_pos,
                           hipStream_t stream);
  // Transformer stack over impl.hidden rows (no head scoring).
  static void ForwardFused(DraftSession& s, std::uint32_t n,
                           std::uint32_t start_pos, hipStream_t stream);

  std::shared_ptr<DraftModel> model_;
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class DraftModel;
};

}  // namespace gufo::models::glimmer::draft

#endif  // GUFO_MODELS_GLIMMER_DRAFT_ENGINE_HPP_
