// bge-reranker OOP adapter: EngineModel (shared encoder + head), EngineSession
// (single-sequence Score), and the model-kind registry.
//
// Rerank semantics, fixed for this model kind:
//   Score(ids) = full bidirectional encoder prefill over the concatenated
//   [CLS] query [SEP] doc [SEP] pair, CLS row (row 0) through the Roberta
//   classification head (dense + tanh + out-projection), sigmoid to (0, 1).
//   There is no decode, no KV cache, no speculation, and no lm_head --
//   encoder-only artifacts never generate.
//
// The encoder is the bge_m3 base verbatim: the model OWNS a
// bge_m3::EngineModel (CPU reference + HIP prefill twin) and only adds the
// four head tensors. Registry: model kinds self-register by name; serve and
// drivers resolve "bge-reranker" through ModelRegistry::Create.
#ifndef GUFO_MODELS_BGE_RERANKER_ENGINE_HPP_
#define GUFO_MODELS_BGE_RERANKER_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/models/bge_m3/engine.hpp"
#include "src/models/bge_m3/forward.hpp"
#include "src/models/bge_reranker/config.hpp"
#include "src/models/bge_reranker/weights.hpp"

namespace gufo::models::bge_reranker {

class EngineModel;

/// One rerank session: caller-owned scratch, synchronous single-sequence
/// Score. Sessions are not thread-safe; models are freely shared.
class EngineSession {
public:
  explicit EngineSession(const EngineModel* model);
  ~EngineSession();
  EngineSession(const EngineSession&) = delete;
  EngineSession& operator=(const EngineSession&) = delete;

  struct ScoreResult {
    float score{0.0F};  // sigmoid(head(CLS)), in (0, 1)
    float logit{0.0F};  // raw head logit (matches llama.cpp relevance scale)
    double prefill_ms{0.0};
  };

  /// Full pair prefill + head + sigmoid. Returns nullopt (with *error_msg
  /// set) when ids is empty/over-long or holds out-of-vocab ids. `use_hip`
  /// requests the device encoder path; without ENGINE_ENABLE_HIP (or when
  /// device init failed) the call runs the CPU reference instead.
  [[nodiscard]] std::optional<ScoreResult> Score(
      std::span<const std::uint32_t> ids, bool use_hip,
      std::string* error_msg = nullptr);

  /// Tokenizes one (query, document) pair as
  /// [CLS] query [SEP] document [SEP] through the encoder tokenizer.
  /// Returns nullopt when either side is empty after normalization or the
  /// wrapped pair exceeds `max_tokens`.
  [[nodiscard]] std::optional<std::vector<std::uint32_t>> EncodePair(
      std::string_view query, std::string_view document,
      std::string* error_msg = nullptr) const;

  /// Scores one (query, document) pair end to end.
  [[nodiscard]] std::optional<ScoreResult> ScorePair(
      std::string_view query, std::string_view document, bool use_hip,
      std::string* error_msg = nullptr);

  /// Pair-batching on one session: every document is prefilled through the
  /// same encoder (no reload, no session rebuild); only the head math runs
  /// per pair. Returns scores in input order with the summed prefill time.
  [[nodiscard]] std::optional<std::vector<float>> ScoreBatch(
      std::string_view query, const std::vector<std::string>& documents,
      bool use_hip, double* total_prefill_ms, std::size_t* prompt_tokens,
      std::string* error_msg = nullptr);

private:
  const EngineModel* model_;
  bge_m3::BgeM3Scratch scratch_;
};

/// Immutable, shareable bge-reranker model: the bge_m3 encoder plus the F32
/// classification-head mirror. The encoder owns the mapped GGUF storage, so
/// head references stay valid as long as the encoder outlives them (it is
/// declared first).
class EngineModel {
public:
  EngineModel(const EngineModel&) = delete;
  EngineModel& operator=(const EngineModel&) = delete;
  ~EngineModel();

  /// Loads a `bert` + classifier-head GGUF artifact (encoder + head).
  /// Returns nullptr with *error_msg set on any refusal.
  [[nodiscard]] static std::unique_ptr<EngineModel> Create(
      const std::string& model_path, std::string* error_msg = nullptr);

  [[nodiscard]] const BgeRerankerConfig& Config() const noexcept {
    return config_;
  }
  [[nodiscard]] const bge_m3::EngineModel& Encoder() const noexcept {
    return *encoder_;
  }
  [[nodiscard]] const BgeRerankerHeadF32& Head() const noexcept {
    return head_;
  }

private:
  EngineModel();
  std::unique_ptr<bge_m3::EngineModel> encoder_;
  BgeRerankerConfig config_;
  BgeRerankerHeadF32 head_;
};

/// Model-kind registry: name -> factory. bge_reranker registers
/// "bge-reranker" via a function-local static initializer (see engine.cpp).
class ModelRegistry {
public:
  using Factory = std::function<std::unique_ptr<EngineModel>(
      const std::string& model_path, std::string* error_msg)>;

  static ModelRegistry& Instance();
  /// Returns false when `name` is already registered (first wins).
  bool Register(const std::string& name, Factory factory);
  [[nodiscard]] std::unique_ptr<EngineModel> Create(
      const std::string& name, const std::string& model_path,
      std::string* error_msg = nullptr) const;

private:
  ModelRegistry() = default;
  mutable std::mutex mutex_;
  std::map<std::string, Factory> factories_;
};

}  // namespace gufo::models::bge_reranker

#endif  // GUFO_MODELS_BGE_RERANKER_ENGINE_HPP_
