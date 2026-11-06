// bge-reranker EngineModel / EngineSession / ModelRegistry implementation.
#include "src/models/bge_reranker/engine.hpp"

#include <chrono>
#include <map>
#include <mutex>
#include <utility>

#include "src/core/rerank_score.hpp"

namespace gufo::models::bge_reranker {

namespace {

// Head math over the CLS row, shared by the CPU and HIP prefill paths: the
// F32 head mirror lives in the model, so scoring is one row-major dense +
// tanh + dot product on already-host memory (microseconds at hidden 1024).
EngineSession::ScoreResult ScoreClsRow(std::span<const float> cls,
                                       const BgeRerankerHeadF32& head,
                                       double prefill_ms) {
  EngineSession::ScoreResult result;
  result.prefill_ms = prefill_ms;
  result.logit = core::rerank::SequenceHeadLogit(
      cls, head.dense_w, head.dense_b, head.out_w, head.out_b);
  result.score = core::rerank::Sigmoid(result.logit);
  return result;
}

}  // namespace

EngineModel::~EngineModel() = default;
EngineModel::EngineModel() = default;

std::unique_ptr<EngineModel> EngineModel::Create(const std::string& model_path,
                                                 std::string* error_msg) {
  auto refuse = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return std::unique_ptr<EngineModel>{};
  };
  // The encoder base binds every shared tensor; pooling is explicitly not
  // required (reranker GGUFs declare no pooling_type -- relevance always
  // reads CLS by construction).
  auto encoder = bge_m3::EngineModel::Create(
      model_path, error_msg,
      bge_m3::BgeM3ConfigOptions{.require_cls_pooling = false});
  if (encoder == nullptr) {
    return refuse(error_msg != nullptr ? *error_msg
                                       : "bge-reranker encoder load failed");
  }
  auto config = ExtractBgeRerankerConfig(encoder->GetReader(), error_msg);
  if (!config.has_value()) {
    return std::unique_ptr<EngineModel>{};
  }
  auto head =
      LoadRerankHead(encoder->GetReader(), config->hidden_size, error_msg);
  if (!head.has_value()) {
    return std::unique_ptr<EngineModel>{};
  }
  auto model = std::unique_ptr<EngineModel>(new EngineModel());
  model->encoder_ = std::move(encoder);
  model->config_ = *config;
  model->head_ = std::move(*head);
  return model;
}

EngineSession::EngineSession(const EngineModel* model)
    : model_(model),
      scratch_(bge_m3::BgeM3Scratch::Create(model->Encoder().Config(),
                                            kBgeRerankerMaxTokens)) {}

EngineSession::~EngineSession() = default;

std::optional<std::vector<std::uint32_t>> EngineSession::EncodePair(
    std::string_view query, std::string_view document,
    std::string* error_msg) const {
  auto refuse = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return std::optional<std::vector<std::uint32_t>>{};
  };
  const auto& tokenizer = model_->Encoder().Tokenizer();
  auto encoded_query = tokenizer.EncodeBody(query, kBgeRerankerMaxTokens);
  if (!encoded_query.has_value() || encoded_query->empty()) {
    return refuse("bge-reranker EncodePair: empty query after normalization");
  }
  auto encoded_doc = tokenizer.EncodeBody(document, kBgeRerankerMaxTokens);
  if (!encoded_doc.has_value() || encoded_doc->empty()) {
    return refuse(
        "bge-reranker EncodePair: empty document after normalization");
  }
  const auto& config = model_->Config();
  const std::size_t total =
      encoded_query->size() + encoded_doc->size() + 3;  // CLS + 2x SEP
  if (total > kBgeRerankerMaxTokens) {
    return refuse("bge-reranker EncodePair: pair of " + std::to_string(total) +
                  " tokens exceeds the " +
                  std::to_string(kBgeRerankerMaxTokens) + "-token window");
  }
  std::vector<std::uint32_t> ids;
  ids.reserve(total);
  ids.push_back(config.cls_token_id);
  ids.insert(ids.end(), encoded_query->begin(), encoded_query->end());
  ids.push_back(config.sep_token_id);
  ids.insert(ids.end(), encoded_doc->begin(), encoded_doc->end());
  ids.push_back(config.sep_token_id);
  return ids;
}

std::optional<EngineSession::ScoreResult> EngineSession::Score(
    std::span<const std::uint32_t> ids, bool use_hip, std::string* error_msg) {
  auto refuse = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return std::optional<ScoreResult>{};
  };
  if (ids.empty()) {
    return refuse("bge-reranker Score: empty id sequence");
  }
  if (ids.size() > kBgeRerankerMaxTokens) {
    return refuse("bge-reranker Score: " + std::to_string(ids.size()) +
                  " tokens exceeds the leg window");
  }
  const auto& encoder = model_->Encoder();
  const std::size_t hidden = model_->Head().hidden_size;
#ifdef ENGINE_ENABLE_HIP
  if (use_hip) {
    // Device encoder prefill (shared bge_m3 twin), CLS row back on host;
    // the head itself runs on host -- one 1024-wide dense is not worth a
    // device launch. Failure is an explicit refusal, never silent.
    auto cls = bge_m3::HipClsHidden(encoder, ids, error_msg);
    if (!cls.has_value()) {
      return std::nullopt;
    }
    if (cls->cls.size() != hidden) {
      return refuse("bge-reranker Score: CLS hidden size mismatch");
    }
    return ScoreClsRow(cls->cls, model_->Head(), cls->prefill_ms);
  }
#else
  (void)use_hip;
#endif
  const auto& f32 = encoder.F32();
  std::vector<float> last_hidden(ids.size() * hidden);
  const auto start = std::chrono::steady_clock::now();
  if (!bge_m3::CpuForward(f32, ids, scratch_, last_hidden)) {
    return refuse("bge-reranker Score: CPU forward shape/id failure");
  }
  const double prefill_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
  // CLS is row 0; token-type embeddings are all row 0 by construction
  // (type_vocab 1), matching the trained artifact exactly.
  return ScoreClsRow(std::span<const float>(last_hidden.data(), hidden),
                     model_->Head(), prefill_ms);
}

std::optional<EngineSession::ScoreResult> EngineSession::ScorePair(
    std::string_view query, std::string_view document, bool use_hip,
    std::string* error_msg) {
  auto ids = EncodePair(query, document, error_msg);
  if (!ids.has_value()) {
    return std::nullopt;
  }
  return Score(*ids, use_hip, error_msg);
}

std::optional<std::vector<float>> EngineSession::ScoreBatch(
    std::string_view query, const std::vector<std::string>& documents,
    bool use_hip, double* total_prefill_ms, std::size_t* prompt_tokens,
    std::string* error_msg) {
  auto refuse = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return std::optional<std::vector<float>>{};
  };
  if (documents.empty()) {
    return refuse("bge-reranker ScoreBatch: no documents");
  }
  // Tokenize every pair up front: one over-long pair fails the whole batch
  // before any prefill runs.
  std::vector<std::vector<std::uint32_t>> batch;
  batch.reserve(documents.size());
  std::size_t tokens = 0;
  for (const auto& document : documents) {
    auto ids = EncodePair(query, document, error_msg);
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
    auto result = Score(ids, use_hip, error_msg);
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
// "bge-reranker" (the XLM-R sequence-classification kind) self-registers.
const bool kBgeRerankerRegistered = ModelRegistry::Instance().Register(
    "bge-reranker", [](const std::string& model_path, std::string* error_msg) {
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
      *error_msg = "unknown bge-reranker model kind: " + name;
    }
    return nullptr;
  }
  return it->second(model_path, error_msg);
}

}  // namespace gufo::models::bge_reranker
