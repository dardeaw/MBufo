// qwen-reranker OOP adapter: EngineModel (shared Qwen weights + yes/no ids),
// EngineSession (single-prefill Score), and the model-kind registry.
//
// Rerank semantics, fixed for this model kind:
//   Score(ids) = one causal prefill over the templated
//   [system judge | user <Instruct>/<Query>/<Document> | assistant <think>]
//   prompt, last-token logits at the "yes"/"no" vocabulary ids, softmax ->
//   P(yes). There is no decode beyond the single prefill, no KV retention,
//   and no sampling -- the predicted token is intentionally unused.
//
// The transformer is the Qwen dense path verbatim: the model OWNS (driver)
// or SHARES (serve, via Wrap) a hip::QwenGpuModel and only adds the scoring
// readout. Registry: model kinds self-register by name; drivers resolve
// "qwen-reranker" through ModelRegistry::Create.
#ifndef GUFO_MODELS_QWEN_RERANKER_ENGINE_HPP_
#define GUFO_MODELS_QWEN_RERANKER_ENGINE_HPP_

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

#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::qwen_reranker {

/// Default retrieval instruction (Qwen3-Reranker chat template default).
inline constexpr std::string_view kDefaultInstruction =
    "Given a web search query, retrieve relevant passages that answer the "
    "query";

/// Single-prefill leg window for rerank pairs. The artifact trains to
/// 40960, but pairs are short prompts; 4096 bounds one executor arena while
/// covering the retrieval regime. Longer pairs are refused explicitly.
inline constexpr std::uint32_t kQwenRerankerMaxTokens = 4096;

class EngineModel;

/// One rerank session: owns one cached executor, synchronous Score.
/// Sessions are not thread-safe (a mutex serializes use); models are freely
/// shared.
class EngineSession {
public:
  explicit EngineSession(const EngineModel* model);
  ~EngineSession();
  EngineSession(const EngineSession&) = delete;
  EngineSession& operator=(const EngineSession&) = delete;

  struct ScoreResult {
    float score{0.0F};  // P(yes): softmax over the yes/no last-token logits
    float logit_yes{0.0F};
    float logit_no{0.0F};
    double prefill_ms{0.0};
  };

  /// One prefill + yes/no readout. Returns nullopt (with *error_msg set)
  /// when ids is empty/over-long. The executor is Reset() to a cold state
  /// before every prefill so pairs vary freely in length and content.
  [[nodiscard]] std::optional<ScoreResult> Score(
      std::span<const tokenization::TokenId> ids,
      std::string* error_msg = nullptr);

  /// Renders one (query, document) pair through the Qwen3-Reranker chat
  /// template and scores it end to end.
  [[nodiscard]] std::optional<ScoreResult> ScorePair(
      std::string_view query, std::string_view document,
      std::string_view instruction, std::string* error_msg = nullptr);

  /// Pair-batching on one session: every pair is prefilled through the same
  /// cached executor (no reload, no session rebuild). Returns scores in
  /// input order with the summed prefill time.
  [[nodiscard]] std::optional<std::vector<float>> ScoreBatch(
      std::string_view query, const std::vector<std::string>& documents,
      std::string_view instruction, double* total_prefill_ms,
      std::size_t* prompt_tokens, std::string* error_msg = nullptr);

private:
  [[nodiscard]] bool EnsureExecutor(std::string* error_msg);
  const EngineModel* model_;
  std::mutex mutex_;
  std::unique_ptr<hip::QwenGpuExecutor> executor_;
  std::uint32_t executor_context_{0};
};

/// Immutable, shareable qwen-reranker model: Qwen dense weights plus the
/// resolved yes/no vocabulary ids and the default instruction.
class EngineModel {
public:
  EngineModel(const EngineModel&) = delete;
  EngineModel& operator=(const EngineModel&) = delete;
  ~EngineModel();

  /// Loads a `qwen3` GGUF artifact and resolves the yes/no scoring ids.
  /// Returns nullptr with *error_msg set on any refusal.
  [[nodiscard]] static std::unique_ptr<EngineModel> Create(
      const std::string& model_path, std::string* error_msg = nullptr);

  /// Shares an already-loaded Qwen model (serve path): no weight reload,
  /// only the yes/no readout is added.
  [[nodiscard]] static std::unique_ptr<EngineModel> Wrap(
      std::shared_ptr<const hip::QwenGpuModel> model,
      std::string* error_msg = nullptr);

  [[nodiscard]] const hip::QwenGpuModel& Qwen() const noexcept {
    return *model_;
  }
  [[nodiscard]] const std::shared_ptr<const hip::QwenGpuModel>& SharedModel()
      const noexcept {
    return model_;
  }
  [[nodiscard]] tokenization::TokenId YesId() const noexcept { return yes_id_; }
  [[nodiscard]] tokenization::TokenId NoId() const noexcept { return no_id_; }
  [[nodiscard]] const hip::QwenExecutionPolicy& Policy() const noexcept {
    return policy_;
  }
  void SetPolicy(hip::QwenExecutionPolicy policy) { policy_ = policy; }

  /// Renders the Qwen3-Reranker chat template for one pair. The framing is
  /// byte-exact with the artifact's chat_template.jinja; content never
  /// parses as control tokens (the tokenizer splits specials itself).
  [[nodiscard]] static std::optional<std::vector<tokenization::TokenId>>
  BuildRerankIds(const tokenization::QwenTokenizer& tokenizer,
                 std::string_view instruction, std::string_view query,
                 std::string_view document, std::string* error_msg = nullptr);

private:
  EngineModel();
  [[nodiscard]] static std::unique_ptr<EngineModel> FromModel(
      std::shared_ptr<const hip::QwenGpuModel> model, std::string* error_msg);
  std::shared_ptr<const hip::QwenGpuModel> model_;
  tokenization::TokenId yes_id_{tokenization::kInvalidTokenId};
  tokenization::TokenId no_id_{tokenization::kInvalidTokenId};
  hip::QwenExecutionPolicy policy_{hip::QwenExecutionPolicy::Production()};
};

/// Model-kind registry: name -> factory. qwen_reranker registers
/// "qwen-reranker" via a function-local static initializer (see engine.cpp).
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

}  // namespace gufo::models::qwen_reranker

#endif  // GUFO_MODELS_QWEN_RERANKER_ENGINE_HPP_
