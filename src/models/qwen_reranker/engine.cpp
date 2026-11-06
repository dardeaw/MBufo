// qwen-reranker EngineModel / EngineSession / ModelRegistry implementation.
#include "src/models/qwen_reranker/engine.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <mutex>
#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/core/rerank_score.hpp"

namespace gufo::models::qwen_reranker {

namespace {

// Qwen3-Reranker chat template (byte-exact with the artifact's
// chat_template.jinja: system judge instruction, user Instruct/Query/
// Document turns, assistant <think> opener -- the model emits "yes"/"no"
// as the first decoded token).
constexpr std::string_view kSystemFrameOpen = "<|im_start|>system\n";
constexpr std::string_view kSystemText =
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be "
    "\"yes\" or \"no\".";
constexpr std::string_view kUserFrameOpen =
    "<|im_end|>\n<|im_start|>user\n<Instruct>: ";
constexpr std::string_view kQueryLabel = "\n<Query>: ";
constexpr std::string_view kDocumentLabel = "\n<Document>: ";
constexpr std::string_view kAssistantFrameOpen =
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

}  // namespace

EngineModel::~EngineModel() = default;
EngineModel::EngineModel() = default;

std::unique_ptr<EngineModel> EngineModel::FromModel(
    std::shared_ptr<const hip::QwenGpuModel> model, std::string* error_msg) {
  auto refuse = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return std::unique_ptr<EngineModel>{};
  };
  if (model == nullptr) {
    return refuse("qwen-reranker needs a loaded Qwen model");
  }
  const auto& tokenizer = model->GetTokenizer();
  // The scoring ids are plain vocabulary tokens (no specials): Encode with
  // the default options and require exactly one id each, mirroring
  // convert_tokens_to_ids("yes")/("no") on the HF side.
  const auto yes_ids = tokenizer.Encode("yes");
  const auto no_ids = tokenizer.Encode("no");
  if (yes_ids.size() != 1) {
    return refuse("\"yes\" does not encode to one token in this vocabulary");
  }
  if (no_ids.size() != 1) {
    return refuse("\"no\" does not encode to one token in this vocabulary");
  }
  auto engine = std::unique_ptr<EngineModel>(new EngineModel());
  engine->model_ = std::move(model);
  engine->yes_id_ = yes_ids.front();
  engine->no_id_ = no_ids.front();
  return engine;
}

std::unique_ptr<EngineModel> EngineModel::Create(const std::string& model_path,
                                                 std::string* error_msg) {
  std::string load_error;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
  if (reader_owner == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to open GGUF: " + load_error;
    }
    return nullptr;
  }
  std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  auto model = hip::QwenGpuModel::CreateFromGguf(std::move(reader), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return FromModel(std::move(model), error_msg);
}

std::unique_ptr<EngineModel> EngineModel::Wrap(
    std::shared_ptr<const hip::QwenGpuModel> model, std::string* error_msg) {
  return FromModel(std::move(model), error_msg);
}

std::optional<std::vector<tokenization::TokenId>> EngineModel::BuildRerankIds(
    const tokenization::QwenTokenizer& tokenizer, std::string_view instruction,
    std::string_view query, std::string_view document, std::string* error_msg) {
  auto refuse = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return std::optional<std::vector<tokenization::TokenId>>{};
  };
  if (query.empty()) {
    return refuse("qwen-reranker pair needs a non-empty query");
  }
  if (document.empty()) {
    return refuse("qwen-reranker pair needs a non-empty document");
  }
  const std::string_view effective_instruction =
      instruction.empty() ? kDefaultInstruction : instruction;
  std::string prompt;
  prompt.reserve(kSystemFrameOpen.size() + kSystemText.size() +
                 kUserFrameOpen.size() + effective_instruction.size() +
                 kQueryLabel.size() + query.size() + kDocumentLabel.size() +
                 document.size() + kAssistantFrameOpen.size());
  prompt += kSystemFrameOpen;
  prompt += kSystemText;
  prompt += kUserFrameOpen;
  prompt += effective_instruction;
  prompt += kQueryLabel;
  prompt += query;
  prompt += kDocumentLabel;
  prompt += document;
  prompt += kAssistantFrameOpen;
  // Default options parse the framing as control tokens and encode content
  // as text -- the same split chat rendering uses.
  std::vector<tokenization::TokenId> ids = tokenizer.Encode(prompt);
  if (ids.empty()) {
    return refuse("qwen-reranker pair tokenizes to zero tokens");
  }
  if (ids.size() > kQwenRerankerMaxTokens) {
    return refuse("qwen-reranker pair of " + std::to_string(ids.size()) +
                  " tokens exceeds the " +
                  std::to_string(kQwenRerankerMaxTokens) + "-token window");
  }
  return ids;
}

EngineSession::EngineSession(const EngineModel* model) : model_(model) {}
EngineSession::~EngineSession() = default;

bool EngineSession::EnsureExecutor(std::string* error_msg) {
  const std::uint32_t want_context = std::min(
      model_->Qwen().GetConfig().context_length, kQwenRerankerMaxTokens);
  if (executor_ != nullptr && executor_context_ == want_context) {
    return true;
  }
  std::string exec_error;
  auto executor = hip::QwenGpuExecutor::Create(
      model_->SharedModel(), &exec_error, want_context, model_->Policy());
  if (executor == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "qwen-reranker executor init failed: " + exec_error;
    }
    return false;
  }
  executor_ = std::move(executor);
  executor_context_ = want_context;
  return true;
}

std::optional<EngineSession::ScoreResult> EngineSession::Score(
    std::span<const tokenization::TokenId> ids, std::string* error_msg) {
  auto refuse = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return std::optional<ScoreResult>{};
  };
  if (ids.empty()) {
    return refuse("qwen-reranker Score: empty id sequence");
  }
  if (ids.size() > kQwenRerankerMaxTokens) {
    return refuse("qwen-reranker Score: " + std::to_string(ids.size()) +
                  " tokens exceeds the leg window");
  }
  if (ids.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return refuse("qwen-reranker Score: id sequence too long");
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!EnsureExecutor(error_msg)) {
    return std::nullopt;
  }
  // Cold prefill state per pair (mirrors the embedding path): prompt
  // lengths and contents vary freely; compute_logits runs the lm_head so
  // the last row holds the yes/no readout.
  executor_->Reset();
  const auto start = std::chrono::steady_clock::now();
  try {
    (void)executor_->ForwardPromptBatch(ids, 0, true);
  } catch (const std::exception& exception) {
    return refuse(std::string("qwen-reranker prefill failed: ") +
                  exception.what());
  }
  const double prefill_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
  const std::span<const float> logits = executor_->CopyLastLogits();
  const std::size_t vocab = model_->Qwen().GetTokenizer().GetVocabSize();
  if (logits.size() < vocab || vocab == 0) {
    return refuse("qwen-reranker Score: logit capture size mismatch");
  }
  const auto yes = static_cast<std::size_t>(model_->YesId());
  const auto no = static_cast<std::size_t>(model_->NoId());
  if (yes >= vocab || no >= vocab) {
    return refuse("qwen-reranker Score: scoring id outside vocabulary");
  }
  ScoreResult result;
  result.logit_yes = logits[yes];
  result.logit_no = logits[no];
  result.score = core::rerank::YesNoScore(result.logit_yes, result.logit_no);
  if (!(result.score >= 0.0F) || !(result.score <= 1.0F)) {
    return refuse("qwen-reranker Score: non-finite yes/no readout refused");
  }
  result.prefill_ms = prefill_ms;
  return result;
}

std::optional<EngineSession::ScoreResult> EngineSession::ScorePair(
    std::string_view query, std::string_view document,
    std::string_view instruction, std::string* error_msg) {
  auto ids = EngineModel::BuildRerankIds(
      model_->Qwen().GetTokenizer(), instruction, query, document, error_msg);
  if (!ids.has_value()) {
    return std::nullopt;
  }
  return Score(*ids, error_msg);
}

std::optional<std::vector<float>> EngineSession::ScoreBatch(
    std::string_view query, const std::vector<std::string>& documents,
    std::string_view instruction, double* total_prefill_ms,
    std::size_t* prompt_tokens, std::string* error_msg) {
  auto refuse = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return std::optional<std::vector<float>>{};
  };
  if (documents.empty()) {
    return refuse("qwen-reranker ScoreBatch: no documents");
  }
  // Render every pair up front: one over-long pair fails the whole batch
  // before any prefill runs.
  std::vector<std::vector<tokenization::TokenId>> batch;
  batch.reserve(documents.size());
  std::size_t tokens = 0;
  for (const auto& document : documents) {
    auto ids = EngineModel::BuildRerankIds(
        model_->Qwen().GetTokenizer(), instruction, query, document, error_msg);
    if (!ids.has_value()) {
      return std::nullopt;
    }
    tokens += ids->size();
    batch.push_back(std::move(*ids));
  }
  std::vector<float> scores;
  scores.reserve(batch.size());
  double total_ms = 0.0;
  for (const auto& ids : batch) {
    auto result = Score(ids, error_msg);
    if (!result.has_value()) {
      return std::nullopt;
    }
    scores.push_back(result->score);
    total_ms += result->prefill_ms;
  }
  if (total_prefill_ms != nullptr)
    *total_prefill_ms = total_ms;
  if (prompt_tokens != nullptr)
    *prompt_tokens = tokens;
  return scores;
}

ModelRegistry& ModelRegistry::Instance() {
  static ModelRegistry instance;
  return instance;
}

namespace {
// "qwen-reranker" (the Qwen causal yes/no kind) self-registers.
const bool kQwenRerankerRegistered = ModelRegistry::Instance().Register(
    "qwen-reranker", [](const std::string& model_path, std::string* error_msg) {
      return EngineModel::Create(model_path, error_msg);
    });
}  // namespace

bool ModelRegistry::Register(const std::string& name, Factory factory) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (factories_.find(name) != factories_.end()) {
    return false;
  }
  factories_.emplace(std::move(name), std::move(factory));
  return true;
}

std::unique_ptr<EngineModel> ModelRegistry::Create(
    const std::string& name, const std::string& model_path,
    std::string* error_msg) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = factories_.find(name);
  if (it == factories_.end()) {
    if (error_msg != nullptr) {
      *error_msg = "unknown qwen-reranker model kind: " + name;
    }
    return nullptr;
  }
  return it->second(model_path, error_msg);
}

}  // namespace gufo::models::qwen_reranker
