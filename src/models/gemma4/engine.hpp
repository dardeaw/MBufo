#ifndef GUFO_MODELS_GEMMA4_ENGINE_HPP_
#define GUFO_MODELS_GEMMA4_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/weights.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::gemma4 {

struct ModelOptions {
  std::uint32_t max_context = 8192;
  std::uint32_t decode_concurrency = 1;
  // KV cache format: "f16" (default, legacy path) or "q8" (shared blocked
  // INT8 layout on SWA d == 256 layers; full d == 512 layers stay F16).
  std::string kv_cache_quant = "f16";
};

class Session;
class GemmaMtpDraftBackend;
struct GemmaTargetVerify;

/// Minimal single-model runtime for Gemma 4 26B A4B on gfx1151. Weights
/// stream from the mapped GGUF file through the Flash-Next mmq GEMM kernels;
/// KV caches live in device memory per session.
class Model final : public std::enable_shared_from_this<Model> {
public:
  struct Impl;
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
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

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  // Caller-owned arena for Session::DecodeBatch unified launches (§4.2:
  // allocated on first use, freed in ~Model).
  mutable void* mixed_scratch_{nullptr};
  mutable std::size_t mixed_scratch_bytes_{0};

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt; Evaluate appends one
/// token; the logits of the last token are kept for sampling.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Reprocesses the prompt; append-only growth over the previously synced
  /// prefix forwards only the delta (incremental prefix: O(n) instead of
  /// the v1 full O(n^2) replay). Anything else fully resyncs from 0.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  /// Appends one token and runs one forward step.
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  /// One decode step for a batch of sessions on the same model: per layer,
  /// every session runs QKV projection + KV store, one unified mixed-batch
  /// attention launch (gufo::core::compute::mixed_batch) covers the batch,
  /// then every session runs wo + FFN. Full d=512 layers stay F16; blocked
  /// SWA layers ride the unified blocked dispatch. Sessions with MTP enabled
  /// are refused (use Evaluate instead). Returns false on validation
  /// failure; a per-layer unified refusal falls back to the legacy
  /// single-session core internally, so output matches serial Evaluate.
  [[nodiscard]] static bool DecodeBatch(std::span<Session*> sessions,
                                        std::span<const std::int32_t> tokens,
                                        std::string* error_msg = nullptr);
  /// Runs a forward over `tokens` starting at `start_pos`.
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             std::uint32_t start_pos,
                             std::string* error_msg = nullptr,
                             bool all_logits = false);
  /// Attaches a Gemma 4 MTP draft head (speculative decoding for Evaluate).
  [[nodiscard]] bool EnableMtp(const std::string& drafts_path,
                               std::string* error_msg = nullptr);
  [[nodiscard]] bool MtpEnabled() const noexcept;
  /// One speculative round: drafts up to `n_draft` tokens after `last_token`,
  /// verifies them with the target in a single batched forward, keeps the
  /// longest matching prefix plus the target's correction token, rolls the KV
  /// back, and appends the accepted tokens to `out`.
  [[nodiscard]] bool SpeculativeRound(std::int32_t last_token,
                                      std::uint32_t n_draft,
                                      std::vector<std::int32_t>* out,
                                      std::string* error_msg = nullptr);
  /// Resolves the anchor token for an MTP round (pending correction or a
  /// fresh greedy sample) and snapshots its target hidden state, mirroring
  /// the preamble of SpeculativeRound. Used by the unified draft backend.
  [[nodiscard]] std::int32_t ResolveDraftAnchor() const;
  [[nodiscard]] std::span<const float> Logits() const noexcept;
  [[nodiscard]] std::uint32_t Length() const noexcept;
  [[nodiscard]] std::int32_t SampleGreedy() const;

private:
  Session(std::shared_ptr<Model> model);

  std::shared_ptr<Model> model_;
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // Single-layer phases factored out of Forward so DecodeBatch can run them
  // batch-wide with one unified attention launch in the middle.
  static void ForwardLayerPre(Session& s, std::uint32_t il, std::uint32_t n,
                              std::uint32_t start_pos, hipStream_t stream);
  static void ForwardLayerAttn(Session& s, std::uint32_t il, std::uint32_t n,
                               std::uint32_t start_pos, hipStream_t stream);
  static void ForwardLayerPost(Session& s, std::uint32_t il, std::uint32_t n,
                               std::uint32_t start_pos, hipStream_t stream);
  static void ForwardFinish(Session& s, std::uint32_t n,
                            std::uint32_t start_pos, bool all_logits,
                            hipStream_t stream);
  // SWA pointer-shift/rebase shared by the legacy core and DecodeBatch:
  // dense-causal (k2, v2, sp, rebased) for keys [0, rebased + n).
  static void SwaShiftedKV(Session& s, std::uint32_t kv_layer,
                            std::uint32_t window, std::uint32_t start_pos,
                            hipStream_t stream, const std::uint8_t*& k2,
                            const std::uint8_t*& v2, const std::uint32_t*& sp,
                            std::uint32_t& rebased);

  friend class Model;
  friend class GemmaMtpDraftBackend;
  friend struct GemmaTargetVerify;

  /// Commits a verified round: assigns [cur, accepted drafts], moves the KV
  /// length, and stages the correction token. Moved verbatim out of
  /// SpeculativeRound (the accept comparison itself stays in the round).
  [[nodiscard]] bool CommitMtpRound(const std::vector<std::int32_t>& vtoks,
                                    const std::int32_t* vbest,
                                    std::uint32_t accepted,
                                    std::uint32_t old_length,
                                    std::vector<std::int32_t>* out);
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_ENGINE_HPP_
