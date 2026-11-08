#ifndef GUFO_MODELS_GLIMMER_DRAFT_ENGINE_HPP_
#define GUFO_MODELS_GLIMMER_DRAFT_ENGINE_HPP_

// DFlash draft runtime (muse-glimmer companion draft, e.g. dflash-kquant):
// a block-diffusion feature draft. A depth-pruned dense decoder (5 layers,
// same width/dim as the target) consumes FUSED TARGET hidden states —
// `fc.weight` maps the concatenated tapped-layer features [n_taps*hidden]
// to the model width — and predicts with the TARGET's output head, so the
// artifact carries no embedding and no vocabulary of its own.
//
// Reference (llama.cpp draft-dflash, 96.3% accept @ temp 0):
//   PrimeRows  prompt fused features (prompt-first) are encoded (fc + enc
//              norm) and INJECTED as the draft KV (K = qk-norm(wk(enc)) +
//              RoPE @ target pos, V = wv(enc)); the draft transformer never
//              runs over fused rows.
//   Propose    one NON-CAUSAL block decode over embedded
//              [id_last, <mask> * k] (target tok_embd, raw, no embd norm),
//              greedy argmax per mask position (llama top_k=10/p_min=0
//              equivalent at temp 0).
// The backend drives prime/append/propose through these primitives plus
// the shared blocked-Q8 draft KV; no F16 mirror is kept.
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

/// Target token embedding bound by the backend (non-owning view into the
/// target model; the draft embeds [id_last, mask...] blocks raw, without
/// the target's embd_norm, mirroring the reference decoder).
struct TargetEmbed {
  const void* device{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
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
  /// Binds the target token embedding (block input rows).
  /// Must be called once before CreateSession.
  void SetTargetEmbed(TargetEmbed embed) { target_embed_ = embed; }

  [[nodiscard]] std::unique_ptr<DraftSession> CreateSession(
      std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t VocabSize() const noexcept {
    return target_head_.vocab_size;
  }
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] std::uint32_t HiddenSize() const noexcept;
  [[nodiscard]] std::uint32_t MaskTokenId() const noexcept;
  [[nodiscard]] const std::vector<std::uint32_t>& TargetLayerIds() const noexcept {
    return target_layer_ids_;
  }
  [[nodiscard]] const TargetHead& target_head() const noexcept {
    return target_head_;
  }
  [[nodiscard]] const TargetEmbed& target_embed() const noexcept {
    return target_embed_;
  }

 private:
  DraftModel() = default;

  DraftModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<DraftModelWeights> weights_;
  TargetHead target_head_;
  TargetEmbed target_embed_;
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
  /// (encoder + KV injection at target positions; the draft transformer
  /// never runs over fused rows). Loops internally in kDraftChunk blocks.
  [[nodiscard]] bool PrimeRows(std::span<const float> feats, std::uint32_t n,
                               std::uint32_t start_pos,
                               std::string* error_msg = nullptr);
  /// Embeds one block of token ids (e.g. [id_last, mask*k]) with the
  /// target embedding (raw gather, no embd norm) into the hidden staging.
  [[nodiscard]] bool EmbedBlock(std::span<const std::int32_t> toks,
                                std::string* error_msg = nullptr);
  /// Forwards the staged block rows (from EmbedBlock) through the draft
  /// transformer with FULL visibility over [0, start_pos+n) (non-causal
  /// block diffusion; each query sees the whole prefix plus the block).
  [[nodiscard]] bool ForwardBlock(std::uint32_t n, std::uint32_t start_pos,
                                  std::string* error_msg = nullptr);
  /// Scores staged hidden row i through the target head (for sampling).
  [[nodiscard]] bool ScoreRow(std::uint32_t i, std::string* error_msg = nullptr);
  /// Moves the length marker back (used by the backend to roll back
  /// unaccepted proposals). KV rows beyond the marker stay valid and are
  /// overwritten by later forwards.
  [[nodiscard]] bool Rewind(std::uint32_t new_length,
                            std::string* error_msg = nullptr);
  [[nodiscard]] std::int32_t SampleGreedy();
  [[nodiscard]] std::uint32_t Length() const noexcept;

 private:
  DraftSession(std::shared_ptr<DraftModel> model);

  [[nodiscard]] bool FuseRows(std::span<const float> feats, std::uint32_t n,
                              std::string* error_msg = nullptr);
  /// Encodes the fused staging [0, n) as draft KV at [start_pos, ...).
  [[nodiscard]] bool InjectFused(std::uint32_t n, std::uint32_t start_pos,
                                 std::string* error_msg = nullptr);

  static void ForwardBlockLayer(DraftSession& s, std::uint32_t il,
                                std::uint32_t n, std::uint32_t start_pos,
                                hipStream_t stream);

  std::shared_ptr<DraftModel> model_;
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class DraftModel;
};

}  // namespace gufo::models::glimmer::draft

#endif  // GUFO_MODELS_GLIMMER_DRAFT_ENGINE_HPP_
