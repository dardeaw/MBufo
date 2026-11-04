#ifndef GUFO_MODELS_QWEN_ENGINE_ADAPTER_HPP_
#define GUFO_MODELS_QWEN_ENGINE_ADAPTER_HPP_

// Qwen3.8-27B adapter: exposes the session-less QwenGpuExecutor through the
// L0/L1 engine abstraction (EngineModel / SpeculativeSession).
//
// The adapter only forwards; the 27B engine body is untouched (no qwen/ or
// qwen/hip edits). The 27B runner is the semantic outlier of the five
// registry engines: there is no Session class — the executor itself carries
// the live KV/SSM frontier — so each L1 session owns one exclusive
// QwenGpuExecutor, which IS the session (see Lifetime below).
//
// Mapping (all inside this translation unit):
//   - Sync(prompt) -> Reset + ForwardPromptBatch. A strict extension of the
//     committed ledger appends only the tail at the current frontier (the
//     same incremental-prefix shape as GenerateFromPrefix); an identical
//     prompt is a no-op; anything else resets and refeeds the full prompt
//     through the native prefill contract.
//   - Evaluate(token) -> ForwardToken(token, Length()) + CopyLastLogits.
//     ForwardToken already argmaxes on device; the host logits mirror is
//     refreshed from the same row so SampleGreedy (host argmax) agrees
//     with the device prediction that the Generate loop would consume.
//   - Forward(tokens, start_pos) emulates the L0 span contract with one
//     Sync over ledger[0:start_pos] + tokens: exact for extension,
//     ledger-reuse on overlap, reset+refeed on divergence (same shape as
//     the flash-next adapter; 27B has no native Forward entry either).
//   - DecodeStep is the plain greedy single-token fallback (sample the
//     current logits, Evaluate it), identical in shape to the
//     llama/gemma4/glimmer fallbacks. Generate semantics collapse to one
//     committed token per step; callers needing a full loop drive DecodeStep
//     until EOS/max tokens.
//   - SaveState/RestoreState/ReplaySsmState -> SnapshotCapable. The 27B
//     SSM state is a single rolling buffer pair, not per-position rows:
//     re-running an old position against the current rolling state would
//     corrupt it (the KV rows are rewritable, the recurrent state is
//     cumulative), so Snapshot() stores only the adapter token ledger and
//     restore recomputes by Reset + full re-prefill — deterministic
//     kernels, hence bit-exact. Honest differences, stated plainly:
//       * SSM state is RECOMPUTED, never copied (SSM重算語義): unlike the
//         arena pin pair (single-slot, in-place, no caller-owned bytes)
//         and unlike ReplaySsmState (which replays the qkv/alpha/beta log
//         and is valid only for captured positions), a restored snapshot
//         depends on neither the pin nor the replay log.
//       * Restore TRUNCATES (截斷語義): the frontier rewinds to the ledger
//         boundary; positions beyond it are discarded, never merged.
//     Truncated, version-mismatched, or ledger-inconsistent payloads are
//     refused before any device work; a failed restore leaves the device
//     reset, so the caller must re-Sync before further L0 calls.
//   - Vision: HasVision()/ConfigureVision() pass-throughs only, never in
//     the base. The executor configures vision from a Prompt object plus an
//     Encoder, not from raw encoded bytes, so there is no meaningful
//     VisionCapable::AttachImage mapping; the adapter deliberately does not
//     derive VisionCapable (documented non-support of the L1 vision form).
//     ConfigureVision must precede the first Sync (native rope validation
//     runs against the session max context).
//   - MTP/speculative: the verify entries (ForwardVerificationChunk and
//     friends) are executor-level — they consume caller-supplied draft
//     spans at an explicit start_pos and settle through
//     CommitVerificationChunk — while L1 DecodeStep carries no drafts and
//     EngineLoadRequest carries no drafts artifact. SupportsSpeculative()
//     is therefore false (same structural reason as gemma4/glimmer), and
//     the chunk entries stay available as pass-throughs for external draft
//     producers (dflash/MTP backends drive them at executor level in the
//     speculative verifier).
//
// Lifetime: Qwen27BEngineModel shares one immutable QwenGpuModel (weights,
// tokenizer, vision encoder); each CreateSession builds one exclusive
// QwenGpuExecutor over it plus an adapter-side token ledger. Executors are
// never shared between sessions, and sessions are never pooled: closing a
// session destroys its executor (GPU arenas release with it).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/engine/engine_batch.hpp"
#include "src/models/qwen/hip/executor.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace gufo::models::qwen {

/// Load-time options for the 27B registry model. Zero/empty fields keep
/// engine defaults (full native context, f16 KV); "q8" selects the shared
/// blocked INT8 attention-KV path (refused at Load when the config cannot
/// use it, never silently).
struct Qwen27BModelOptions {
  std::uint32_t max_context{0};
  std::string kv_cache_quant;
};

class Qwen27BEngineSession final : public core::engine::SpeculativeSession,
                                   public core::engine::SnapshotCapable {
 public:
  Qwen27BEngineSession(
      std::shared_ptr<const hip::QwenGpuModel> model,
      std::unique_ptr<hip::QwenGpuExecutor> executor,
      std::uint32_t max_context);

  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr) override;
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr) override;
  // Emulated with one Sync over ledger[0:start_pos] + tokens (see file
  // doc): exact for extension, ledger-reuse on overlap, reset+refeed on
  // divergence.
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             std::uint32_t start_pos,
                             std::string* error_msg = nullptr) override;
  [[nodiscard]] std::span<const float> Logits() const noexcept override;
  [[nodiscard]] std::uint32_t Length() const noexcept override;
  [[nodiscard]] std::int32_t SampleGreedy() override;
  // Committed token history (prompt + decoded tokens), for tests
  // reconstructing multi-step runs through the L1 outcome (which names
  // only the last committed token per step).
  [[nodiscard]] std::span<const std::uint32_t> Tokens() const noexcept;

  // False: the native verify entry needs caller-supplied drafts at an
  // explicit start_pos (see file doc); L1 DecodeStep carries none, so this
  // is the plain greedy single-token fallback.
  [[nodiscard]] bool SupportsSpeculative() const noexcept override {
    return false;
  }
  [[nodiscard]] bool DecodeStep(std::uint32_t max_draft_tokens,
                                core::engine::DecodeOutcome* outcome,
                                std::string* error_msg = nullptr) override;

  // SnapshotCapable: framed token ledger (see file doc). Payload = magic
  // + version + valid_context + the committed token ledger; restore
  // rewinds by Reset + full re-prefill, so the SSM rolling state is
  // recomputed and the frontier truncates to the ledger boundary.
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override;
  [[nodiscard]] bool Snapshot(std::span<std::uint8_t> destination,
                              std::string* error_msg = nullptr) const override;
  [[nodiscard]] bool RestoreSnapshot(
      std::span<const std::uint8_t> payload,
      std::string* error_msg = nullptr) override;

  // Executor-level verify pass-throughs for external draft producers
  // (unchanged 27B semantics). They bypass the L0 token ledger: re-Sync
  // before resuming L0 calls after using them.
  [[nodiscard]] std::vector<std::uint32_t> ForwardVerificationChunk(
      std::span<const std::uint32_t> candidate_tokens, std::uint32_t start_pos,
      bool capture_logits, std::string* error_msg = nullptr);
  [[nodiscard]] bool CommitVerificationChunk(
      std::span<const std::uint32_t> committed_tokens, std::uint32_t start_pos,
      std::string* error_msg = nullptr);
  [[nodiscard]] bool FinishVerification(std::string* error_msg = nullptr);

  // L0-external 27B pass-throughs (unchanged engine semantics).
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool HasVision() const noexcept;
  // Attaches a vision prompt to the live executor (call before the first
  // Sync; native rope validation runs against the session max context).
  // Ledger restore re-prefills through the currently configured prompt,
  // so re-configure after a restore when the prompt changed.
  [[nodiscard]] bool ConfigureVision(
      std::shared_ptr<const vision::Prompt> prompt,
      std::shared_ptr<vision::Encoder> encoder,
      std::string* error_msg = nullptr);
  [[nodiscard]] const std::shared_ptr<const hip::QwenGpuModel>& inner_model()
      const noexcept {
    return model_;
  }
  [[nodiscard]] hip::QwenGpuExecutor* inner_executor() const noexcept {
    return executor_.get();
  }

 private:
  [[nodiscard]] bool EnsureOpen(std::string* error_msg) const;

  std::shared_ptr<const hip::QwenGpuModel> model_;
  // Mutable: Snapshot() capture is const on the session frontier (it never
  // advances Length) but the native capture allocates and syncs device
  // memory through non-const entries.
  mutable std::unique_ptr<hip::QwenGpuExecutor> executor_;
  std::uint32_t max_context_{0};
  // Adapter-side ledger: every token the executor frontier represents.
  std::vector<std::uint32_t> tokens_;
  // Host mirror of the last forward row (empty before the first Sync).
  std::vector<float> logits_;
};

class Qwen27BEngineModel final : public core::engine::EngineModel {
 public:
  [[nodiscard]] static std::shared_ptr<Qwen27BEngineModel> Load(
      const std::string& model_path, const Qwen27BModelOptions& options,
      std::string* error_msg = nullptr);
  [[nodiscard]] static std::shared_ptr<Qwen27BEngineModel> Load(
      const std::string& model_path, std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<core::engine::EngineSession> CreateSession(
      std::string* error_msg = nullptr) override;

  [[nodiscard]] std::vector<std::int32_t> Tokenize(
      std::string_view text, bool add_bos) const override;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept override;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept override;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept override;
  // No native name exists on QwenGpuModel; returns the load path.
  [[nodiscard]] std::string ModelName() const override;

  // L0-external 27B pass-throughs (unchanged engine semantics).
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool HasVision() const noexcept;
  [[nodiscard]] const std::shared_ptr<const hip::QwenGpuModel>& inner()
      const noexcept {
    return model_;
  }

 private:
  Qwen27BEngineModel(std::shared_ptr<const hip::QwenGpuModel> model,
                      std::string model_path, std::uint32_t max_context,
                      hip::QwenExecutionPolicy policy);

  std::shared_ptr<const hip::QwenGpuModel> model_;
  std::string model_path_;
  std::uint32_t max_context_{0};
  hip::QwenExecutionPolicy policy_;
};

}  // namespace gufo::models::qwen

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_ENGINE_ADAPTER_HPP_
