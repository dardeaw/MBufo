#ifndef GUFO_MODELS_GLIMMER_ENGINE_HPP_
#define GUFO_MODELS_GLIMMER_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/models/glimmer/weights.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::glimmer {

struct ModelOptions {
  std::uint32_t max_context = 32768;
  // KV cache format: "f16" (dense planes, default) or "q8" (shared blocked
  // grid, post-RoPE rows quantized straight in; SWA layers pre-shift past
  // the window, mirroring the llama q8 path).
  std::string kv_cache_quant = "f16";
};

class Session;

/// Minimal single-model runtime for `muse-glimmer` GGUF artifacts (dense
/// GQA decoder with SWA layers, QK-norm, attention output gate and
/// post-norms; e.g. Muse Glimmer 30B) on gfx1151. GEMM rides the shared core
/// (GemvBatched decode / DenseF16Gemm prefill); attention is a local dense
/// F16 kernel with SWA windowing, with the single-session and multi-session
/// (DecodeBatch) fast paths routed through the shared unified mixed-batch
/// launcher (refusal falls back to the legacy cores, F16 semantics intact).
class Model final : public std::enable_shared_from_this<Model> {
 public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(
      std::string_view text, bool add_bos) const;
  [[nodiscard]] std::string Decode(
      std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] const ModelWeights& weights() const noexcept {
    return *weights_;
  }
  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;

 private:
  Model() = default;

  // Caller-owned mixed-batch attention arena (§4.2): one buffer shared by
  // every unified attention launch, regrown when a wider batch needs more.
  // All glimmer launches run on stream 0 (same serial discipline as the
  // gemma4 mixed-batch path), so no locking is needed.
  [[nodiscard]] bool EnsureMixedScratch(std::size_t need,
                                        std::string* error_msg);

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  // Engine-owned device mirrors of every weight tensor (freed in ~Model).
  std::vector<void*> device_allocations_;
  void* mixed_scratch_{nullptr};
  std::size_t mixed_scratch_bytes_{0};

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt; Evaluate appends one
/// token; the logits of the last token are kept for sampling.
class Session final {
 public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// One session's rows inside a DecodeBatch launch: 1 token is a decode
  /// row, N tokens are prefill-chunk rows. Every item must share one model
  /// (hence one KV codec); the attention phase runs a single unified
  /// launch and falls back to the legacy per-session cores on refusal.
  struct BatchItem {
    Session* session{nullptr};
    std::span<const std::int32_t> tokens;
  };

  /// Runs one forward step for every item (variable rows per session) and
  /// leaves each session's logits on its last row. Refusal inside the
  /// unified attention launch falls back to the legacy per-session cores,
  /// so the result matches the serial per-session forwards token-for-token.
  [[nodiscard]] static bool DecodeBatch(std::span<BatchItem> items,
                                        std::string* error_msg = nullptr);

  /// Reprocesses the prompt in 512-token chunks. Append-only growth
  /// (the new prompt extends the previously synced prefix) forwards only
  /// the delta; anything else falls back to a full resync from 0.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  /// Appends one token and runs one forward step.
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  /// Runs a forward over `tokens` starting at `start_pos`.
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             std::uint32_t start_pos,
                             std::string* error_msg = nullptr,
                             bool all_logits = false);
  /// Batched verify: forward `tokens` and return the greedy token for EVERY
  /// position (used by speculative verification). `vbest` must hold `n`.
  [[nodiscard]] bool Verify(std::span<const std::int32_t> tokens,
                            std::uint32_t start_pos,
                            std::vector<std::int32_t>& vbest,
                            std::string* error_msg = nullptr);
  [[nodiscard]] std::span<const float> Logits() const noexcept;
  [[nodiscard]] std::uint32_t Length() const noexcept;
  [[nodiscard]] std::int32_t SampleGreedy();

 private:
  Session(std::shared_ptr<Model> model);

  static void ForwardLayerPre(Session& s, std::uint32_t il, std::uint32_t n,
                              std::uint32_t start_pos, hipStream_t stream);
  static void ForwardLayerAttn(Session& s, std::uint32_t il, std::uint32_t n,
                               std::uint32_t start_pos, hipStream_t stream,
                               bool try_unified);
  static void ForwardLayerPost(Session& s, std::uint32_t il, std::uint32_t n,
                               std::uint32_t start_pos, hipStream_t stream);
  // Blocked-KV SWA pointer shift: advances the cache base past rows that
  // fell out of the window and rebases the position, keeping dense-causal
  // semantics for the caller (gemma4 SwaShiftedKV convention).
  static void SwaShiftedKV(Session& s, std::uint32_t il, std::uint32_t window,
                           std::uint32_t start_pos, hipStream_t stream,
                           const std::uint8_t*& k2, const std::uint8_t*& v2,
                           const std::uint32_t*& sp, std::uint32_t& rebased);
  static void ForwardFinish(Session& s, std::uint32_t n,
                            std::uint32_t start_pos, hipStream_t stream,
                            bool all_logits = false);

  std::shared_ptr<Model> model_;
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class Model;
};

}  // namespace gufo::models::glimmer

#endif  // GUFO_MODELS_GLIMMER_ENGINE_HPP_
