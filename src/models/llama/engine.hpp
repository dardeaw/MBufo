#ifndef MBUFO_MODELS_LLAMA_ENGINE_HPP_
#define MBUFO_MODELS_LLAMA_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/models/llama/weights.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace mbufo::models::llama {

struct ModelOptions {
  std::uint32_t max_context = 8192;
  // KV cache format: "f16" (default dense path) or "q8" (shared blocked
  // INT8 layout, post-RoPE per-32-block scale, served by the shared
  // attention dispatcher: Split-K-128 decode, scalar prefill).
  std::string kv_cache_quant = "f16";
};

class Session;

/// Minimal single-model runtime for `llama` GGUF artifacts (dense GQA
/// decoder, e.g. Llama-3.1-8B) on gfx1151. GEMM rides the shared core
/// (GemvBatched decode / DenseF16Gemm prefill); attention rides the shared
/// dispatcher for q8 KV and a local dense F16 kernel otherwise.
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
  /// True when every projection is Q8_0, so packed decode lanes take the same GEMV kernel as a single token.
  [[nodiscard]] bool PackedLanesExact() const noexcept;

 private:
  Model() = default;

  /// Packed row scratch shared by every session's lane step (one pass runs at a time under the device lock).
  struct LaneScratch {
    float* hidden{nullptr};
    float* normed{nullptr};
    float* q{nullptr};
    float* k{nullptr};
    float* v{nullptr};
    float* attn{nullptr};
    float* gate{nullptr};
    float* up{nullptr};
    float* ffn{nullptr};
    float* logits{nullptr};
  };
  [[nodiscard]] bool EnsureLaneScratch(std::string* error_msg);
  LaneScratch lane_;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  // Engine-owned device mirrors of every weight tensor (freed in ~Model).
  std::vector<void*> device_allocations_;

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt; Evaluate appends one
/// token; the logits of the last token are kept for sampling.
class Session final {
 public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Reprocesses the prompt in 512-token chunks.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  /// Appends one token and runs one forward step.
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  /// Runs a forward over `tokens` starting at `start_pos`.
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             std::uint32_t start_pos,
                             std::string* error_msg = nullptr);
  /// Forwards prompt[offset, prompt.size()) on top of the retained prefix (`offset` must equal Length(), or be 0 for a
  /// restart). Sync(prompt) == Extend(prompt, 0). A span of more than one chunk runs layer-major.
  [[nodiscard]] bool Extend(std::span<const std::int32_t> prompt, std::size_t offset,
                            std::string* error_msg = nullptr);
  [[nodiscard]] std::span<const float> Logits() const noexcept;
  [[nodiscard]] std::uint32_t Length() const noexcept;
  [[nodiscard]] std::int32_t SampleGreedy();

  /// Rows per canonical prefill piece (512 unless LLAMA_PREFILL_CHUNK overrides it).
  [[nodiscard]] static std::uint32_t PrefillGrain() noexcept;
  /// How far the engine may compute when the serving layer has consumed `reported` prompt tokens: the last grid
  /// boundary at or before it (everything once the prompt is complete). The remainder stays pending, so the pieces
  /// forwarded are the same for any scheduler budget and mixed and separate runs stay bit-identical.
  [[nodiscard]] static std::size_t PrefillTarget(std::size_t reported, std::size_t prompt_size) noexcept;

  static constexpr std::uint32_t kMaxLanes = 8;
  struct LaneItem {
    Session* session{nullptr};
    std::int32_t token{0};
  };
  /// One token for every lane in one layer-synchronous pass: the lanes share the projection weight reads (packed
  /// GemvBatched), each keeps its own rope / KV append / attention. Bit-identical to Evaluate on each lane. Refuses
  /// cleanly (false, `*refused` true, nothing mutated) for non-Q8_0 weights, >8 lanes, duplicates, exhausted context.
  [[nodiscard]] static bool StepBatch(std::span<const LaneItem> lanes, std::string* error_msg = nullptr,
                                      bool* refused = nullptr);

  /// One prefill chunk: prompt is the prefix through the END of the chunk, offset its first row. An empty chunk
  /// (offset == prompt.size()) steps only the lanes.
  struct ChunkItem {
    Session* session{nullptr};
    std::span<const std::int32_t> prompt;
    std::size_t offset{0};
  };
  /// StepBatch plus one session's prefill chunk (Extend) in the same call, validated up front. The chunk never
  /// shares a GEMM with the lanes, so both halves are bit-identical to running them separately. A false return with
  /// `*refused` set mutated nothing; otherwise `*lanes_advanced` tells whether the lanes already took their step.
  [[nodiscard]] static bool DecodeBatchWithChunk(std::span<const LaneItem> lanes, const ChunkItem& chunk,
                                                 std::string* error_msg = nullptr, bool* refused = nullptr,
                                                 bool* lanes_advanced = nullptr);

 private:
  Session(std::shared_ptr<Model> model);

  [[nodiscard]] static bool ValidateLanes(std::span<const LaneItem> lanes, const Session* chunk_session,
                                          std::string* error_msg, bool* refused);
  [[nodiscard]] static bool StepLanes(std::span<const LaneItem> lanes, std::string* error_msg);
  static void AttentionCore(Session& s, std::uint32_t il, std::uint32_t n, std::uint32_t start_pos,
                            hipStream_t stream);
  static void ForwardLayer(Session& s, std::uint32_t il, std::uint32_t n,
                           std::uint32_t start_pos, hipStream_t stream);
  static void ForwardFinish(Session& s, std::uint32_t n,
                            std::uint32_t start_pos, hipStream_t stream);

  std::shared_ptr<Model> model_;
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class Model;
};

}  // namespace mbufo::models::llama

#endif  // MBUFO_MODELS_LLAMA_ENGINE_HPP_
