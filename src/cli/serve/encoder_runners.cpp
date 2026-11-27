// Encoder runner implementations (see encoder_runner.hpp). The bodies are the
// former InferenceBackend::Embed / Rerank branches, one class per model kind.
#include "src/cli/serve/encoder_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/pooling_norm.hpp"
#include "src/models/bge_m3/engine.hpp"
#include "src/models/bge_reranker/engine.hpp"
#include "src/models/embeddinggemma2/engine.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen_reranker/engine.hpp"
#ifdef MBUFO_ENABLE_NPU
#include "src/core/compute/npu/npu_graph_models.hpp"
#endif
#endif

namespace mbufo::server {

std::optional<std::vector<EmbeddingResult>> EmbeddingRunner::SerialEmbedBatch(
    const std::vector<std::string>& texts, std::string* error,
    const std::function<std::optional<EmbeddingResult>(const std::string&,
                                                       std::string*)>& one) {
  std::vector<EmbeddingResult> results;
  results.reserve(texts.size());
  for (const auto& text : texts) {
    auto result = one(text, error);
    if (!result.has_value()) return std::nullopt;
    results.push_back(std::move(*result));
  }
  return results;
}

#if defined(ENGINE_ENABLE_HIP)
namespace {

std::optional<std::vector<EmbeddingResult>> Refuse(std::string* error,
                                                   const std::string& message) {
  if (error != nullptr) *error = message;
  return std::nullopt;
}

// ---------------------------------------------------------------- Gemma 2
class Gemma2EmbeddingRunner final : public EmbeddingRunner {
 public:
  explicit Gemma2EmbeddingRunner(
      std::unique_ptr<models::embeddinggemma2::EngineModel> model)
      : model_(std::move(model)) {}
  std::string_view Kind() const noexcept override { return "embeddinggemma2"; }
  bool NativeBatch() const noexcept override { return true; }

  std::optional<std::vector<EmbeddingResult>> EmbedBatch(
      const std::vector<std::string>& texts, std::string* error) override {
    std::vector<std::vector<std::uint32_t>> ids;
    ids.reserve(texts.size());
    for (const auto& text : texts) {
      ids.push_back(model_->Encode(text));
      if (ids.back().size() <= 2)
        return Refuse(error, "embedding input tokenizes to zero tokens");
      if (ids.back().size() > models::embeddinggemma2::kMaxTokens)
        return Refuse(error, "EmbeddingGemma 2 input exceeds the context window");
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    auto batch = model_->EmbedBatch(ids, error);
    if (!batch.has_value()) return std::nullopt;
    std::vector<EmbeddingResult> results(batch->size());
    for (std::size_t i = 0; i < batch->size(); ++i) {
      results[i].values = std::move((*batch)[i].values);
      results[i].prompt_tokens = ids[i].size();
      results[i].prefill_ms = (*batch)[i].prefill_ms;
    }
    return results;
  }

 private:
  std::mutex mutex_;
  std::unique_ptr<models::embeddinggemma2::EngineModel> model_;
};

// ---------------------------------------------------------------- bge-m3
class BgeM3EmbeddingRunner final : public EmbeddingRunner {
 public:
  BgeM3EmbeddingRunner(
      std::unique_ptr<models::bge_m3::EngineModel> model, bool serial_only,
      std::function<void(models::bge_m3::EngineSession&)> configure)
      : model_(std::move(model)),
        serial_only_(serial_only),
        configure_(std::move(configure)) {}
  std::string_view Kind() const noexcept override { return "bge-m3"; }
  bool NativeBatch() const noexcept override { return !serial_only_; }

  std::optional<std::vector<EmbeddingResult>> EmbedBatch(
      const std::vector<std::string>& texts, std::string* error) override {
    std::vector<std::vector<std::uint32_t>> ids;
    ids.reserve(texts.size());
    for (const auto& text : texts) {
      auto encoded = model_->Tokenizer().Encode(
          text, models::bge_m3::kBgeM3HipMaxTokens);
      if (!encoded.has_value())
        return Refuse(error,
                      "bge-m3 tokenize failed or input exceeds context window");
      ids.push_back(std::move(*encoded));
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    if (serial_only_) return SerialSync(ids, error);
    return Batched(ids, error);
  }

 private:
  models::bge_m3::EngineSession& Session() {
    if (session_ == nullptr) {
      session_ = std::make_unique<models::bge_m3::EngineSession>(model_.get());
      if (configure_) configure_(*session_);
    }
    return *session_;
  }

  // NPU lane (per-sequence) and any batch the device path refuses.
  std::optional<std::vector<EmbeddingResult>> SerialSync(
      const std::vector<std::vector<std::uint32_t>>& ids, std::string* error) {
    std::vector<EmbeddingResult> results;
    results.reserve(ids.size());
    for (const auto& one : ids) {
      auto sync = Session().Sync(one, true, error);
      if (!sync.has_value()) return std::nullopt;
      EmbeddingResult result;
      result.values = std::move(sync->values);
      result.prompt_tokens = one.size();
      result.prefill_ms = sync->prefill_ms;
      results.push_back(std::move(result));
    }
    return results;
  }

  // Length-sorted chunks whose padded shape fits the device arena (the same
  // grouping as bge-reranker ScoreBatch); the CLS row of each sequence is
  // L2-normalized, which is exactly what the single-sequence Sync returns.
  std::optional<std::vector<EmbeddingResult>> Batched(
      const std::vector<std::vector<std::uint32_t>>& ids, std::string* error) {
    std::vector<std::size_t> order(ids.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return ids[a].size() < ids[b].size();
    });
    std::vector<EmbeddingResult> results(ids.size());
    std::size_t pos = 0;
    while (pos < order.size()) {
      std::size_t end = pos;
      std::size_t chunk_lmax = 0;
      while (end < order.size()) {
        const std::size_t l = std::max(chunk_lmax, ids[order[end]].size());
        const std::size_t count = end - pos + 1;
        if (count > 128 ||
            models::bge_m3::BgeBatchBpad(count) * models::bge_m3::BgeBatchLpad(l) >
                models::bge_m3::kBgeM3HipMaxRows) {
          break;
        }
        chunk_lmax = l;
        ++end;
      }
      if (end == pos) end = pos + 1;
      std::vector<std::vector<std::uint32_t>> chunk;
      chunk.reserve(end - pos);
      std::size_t chunk_tokens = 0;
      for (std::size_t i = pos; i < end; ++i) {
        chunk.push_back(ids[order[i]]);
        chunk_tokens += ids[order[i]].size();
      }
      auto cls = models::bge_m3::HipClsHiddenBatch(
          *model_, chunk, model_->Config().pad_token_id, error);
      if (!cls.has_value()) return std::nullopt;
      if (cls->cls.size() != chunk.size())
        return Refuse(error, "bge-m3 batch CLS count mismatch");
      for (std::size_t i = 0; i < chunk.size(); ++i) {
        EmbeddingResult& result = results[order[pos + i]];
        result.values = std::move(cls->cls[i]);
        if (!(core::embedding::L2Normalize(result.values) > 0.0))
          return Refuse(error, "bge-m3 zero-norm embedding refused");
        result.prompt_tokens = chunk[i].size();
        result.prefill_ms = cls->prefill_ms * static_cast<double>(chunk[i].size()) /
                            static_cast<double>(chunk_tokens);
      }
      pos = end;
    }
    return results;
  }

  std::mutex mutex_;
  std::unique_ptr<models::bge_m3::EngineModel> model_;
  std::unique_ptr<models::bge_m3::EngineSession> session_;
  bool serial_only_;
  std::function<void(models::bge_m3::EngineSession&)> configure_;
};

// ------------------------------------------------------------ Qwen3-Embedding
class QwenEmbeddingRunner final : public EmbeddingRunner {
 public:
  QwenEmbeddingRunner(std::shared_ptr<const hip::QwenGpuModel> model,
                      hip::QwenExecutionPolicy policy,
                      std::uint32_t embed_context, std::string model_id,
                      TokenizeFn tokenize)
      : model_(std::move(model)),
        policy_(policy),
        embed_context_(embed_context),
        model_id_(std::move(model_id)),
        tokenize_(std::move(tokenize)) {}
  std::string_view Kind() const noexcept override { return "qwen-embedding"; }
  // Fused only where ForwardPromptSegments serves the model (dense head_dim
  // 128 Qwen, F16 KV); otherwise the explicit serial fallback below.
  bool NativeBatch() const noexcept override {
    return hip::QwenGpuExecutor::SupportsPromptSegments(model_->GetConfig(),
                                                        policy_);
  }

  std::optional<std::vector<EmbeddingResult>> EmbedBatch(
      const std::vector<std::string>& texts, std::string* error) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (texts.size() < 2 || !NativeBatch()) {
      return SerialEmbedBatch(
          texts, error, [this](const std::string& text, std::string* err) {
            return EmbedOne(text, err);
          });
    }
    std::vector<std::vector<std::uint32_t>> all_ids;
    all_ids.reserve(texts.size());
    for (const auto& text : texts) {
      auto ids = PrepareIds(text, error);
      if (!ids.has_value()) return std::nullopt;
      all_ids.push_back(std::move(*ids));
    }
    const auto pooling = ResolvePooling(error);
    if (!pooling.has_value()) return std::nullopt;
    if (!EnsureExecutor(error)) return std::nullopt;
    auto& executor = *executor_;
    const std::size_t row_budget = executor.GetMaxPromptBatch();
    std::vector<EmbeddingResult> results(texts.size());
    // Waves of sequences whose rows fit one device batch (arena rows); every
    // wave is a single fused forward.
    std::size_t first = 0;
    while (first < all_ids.size()) {
      std::size_t last = first;
      std::size_t rows = 0;
      while (last < all_ids.size() &&
             (last == first || rows + all_ids[last].size() <= row_budget)) {
        rows += all_ids[last].size();
        ++last;
      }
      if (!RunWave(all_ids, first, last, rows, *pooling, results, error))
        return std::nullopt;
      first = last;
    }
    return results;
  }

 private:
  std::optional<core::embedding::PoolingStrategy> ResolvePooling(
      std::string* error) const {
    // The artifact declares its reduction; an artifact without pooling_type
    // (chat checkpoints) is refused explicitly instead of guessed.
    const std::string arch = model_->GetConfig().architecture;
    const std::uint32_t pooling_type =
        model_->GetReader().GetMetadataUint32(arch + ".pooling_type").value_or(0);
    const auto pooling =
        core::embedding::PoolingStrategyFromGgufType(pooling_type);
    if (!pooling.has_value() && error != nullptr) {
      *error = "model '" + model_id_ +
               "' declares no GGUF pooling_type; embeddings refused";
    }
    return pooling;
  }

  std::optional<std::vector<std::uint32_t>> PrepareIds(const std::string& text,
                                                       std::string* error) const {
    auto refuse = [&](const std::string& message) {
      if (error != nullptr) *error = message;
      return std::optional<std::vector<std::uint32_t>>{};
    };
    std::vector<std::uint32_t> ids;
    try {
      ids = tokenize_(text);
    } catch (const std::exception& exception) {
      return refuse(std::string("embedding tokenize failed: ") + exception.what());
    }
    if (ids.empty()) return refuse("embedding input tokenizes to zero tokens");
    // EOS pooling (GGUF pooling_type 3 = last token): the artifact was trained
    // with an EOS terminator, so one is appended unless the text already ends
    // with it. This mirrors the HuggingFace tokenize path exactly.
    {
      const auto eos =
          static_cast<std::uint32_t>(model_->GetTokenizer().GetEosTokenId());
      if (eos == tokenization::kInvalidTokenId)
        return refuse("embedding model has no EOS token");
      if (ids.back() != eos) ids.push_back(eos);
    }
    if (ids.size() > embed_context_) {
      return refuse("embedding input of " + std::to_string(ids.size()) +
                    " tokens exceeds the 4096-token leg-1 window");
    }
    return ids;
  }

  bool EnsureExecutor(std::string* error) {
    if (executor_ != nullptr) return true;
    std::string exec_error;
    executor_ = hip::QwenGpuExecutor::Create(model_, &exec_error,
                                             embed_context_, policy_);
    if (executor_ == nullptr && error != nullptr)
      *error = "embedding executor init failed: " + exec_error;
    return executor_ != nullptr;
  }

  // Final RMSNorm (host, exact F32) + L2 on a pooled vector, in place.
  bool FinishVector(std::vector<float>& values, std::string* error) const {
    const std::size_t dim = values.size();
    std::vector<float> normed(dim);
    models::ForwardRMSNorm(values, model_->GetWeights().output_norm, 1e-6F,
                           normed);
    values.swap(normed);
    if (!(core::embedding::L2Normalize(values) > 0.0)) {
      if (error != nullptr) *error = "embedding has zero norm; refused";
      return false;
    }
    return true;
  }

  // One fused forward over sequences [first, last): concatenated rows, shared
  // projections, per-sequence attention, then per-sequence host pooling.
  bool RunWave(const std::vector<std::vector<std::uint32_t>>& all_ids,
               std::size_t first, std::size_t last, std::size_t rows,
               core::embedding::PoolingStrategy pooling,
               std::vector<EmbeddingResult>& results, std::string* error) {
    auto& executor = *executor_;
    const std::size_t dim = model_->GetConfig().hidden_size;
    std::vector<tokenization::TokenId> tokens;
    std::vector<std::uint32_t> lengths;
    tokens.reserve(rows);
    for (std::size_t i = first; i < last; ++i) {
      tokens.insert(tokens.end(), all_ids[i].begin(), all_ids[i].end());
      lengths.push_back(static_cast<std::uint32_t>(all_ids[i].size()));
    }
    executor.Reset();
    executor.SetPromptHiddenCapture(true);
    const auto start = std::chrono::steady_clock::now();
    try {
      executor.ForwardPromptSegments(tokens, lengths);
    } catch (const std::exception& exception) {
      if (error != nullptr)
        *error = std::string("embedding batch prefill failed: ") + exception.what();
      return false;
    }
    const double wave_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start).count();
    const std::span<const float> hidden = executor.GetPromptHiddenStates();
    if (dim == 0 || hidden.size() != rows * dim) {
      if (error != nullptr) *error = "embedding hidden capture size mismatch";
      return false;
    }
    std::size_t row = 0;
    for (std::size_t i = first; i < last; ++i) {
      const std::size_t length = all_ids[i].size();
      EmbeddingResult& result = results[i];
      result.values.assign(dim, 0.0F);
      result.prompt_tokens = length;
      result.prefill_ms = wave_ms * static_cast<double>(length) /
                          static_cast<double>(rows);
      if (!core::embedding::ApplyPooling(hidden.subspan(row * dim, length * dim),
                                         length, dim, pooling, result.values)) {
        if (error != nullptr) *error = "embedding pooling shape mismatch";
        return false;
      }
      if (!FinishVector(result.values, error)) return false;
      row += length;
    }
    return true;
  }

  std::optional<EmbeddingResult> EmbedOne(const std::string& text,
                                          std::string* error) {
    auto refuse = [&](const std::string& message) {
      if (error != nullptr) *error = message;
      return std::optional<EmbeddingResult>{};
    };
    const auto pooling = ResolvePooling(error);
    if (!pooling.has_value()) return std::nullopt;
    auto prepared = PrepareIds(text, error);
    if (!prepared.has_value()) return std::nullopt;
    const std::vector<std::uint32_t> ids = std::move(*prepared);
    if (!EnsureExecutor(error)) return std::nullopt;
    auto& executor = *executor_;
    executor.Reset();
    const std::size_t dim = model_->GetConfig().hidden_size;
    executor.SetPromptHiddenCapture(true);
    const auto start = std::chrono::steady_clock::now();
    try {
      // compute_logits=false skips the lm_head matmul; hidden states are what
      // the pooling reads. An embedding request never decodes.
      std::vector<tokenization::TokenId> token_ids(ids.begin(), ids.end());
      (void)executor.ForwardPromptBatch(token_ids, 0, false);
    } catch (const std::exception& exception) {
      return refuse(std::string("embedding prefill failed: ") + exception.what());
    }
    const double prefill_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start).count();
    const std::span<const float> hidden = executor.GetPromptHiddenStates();
    if (hidden.size() != ids.size() * dim || dim == 0)
      return refuse("embedding hidden capture size mismatch");
    EmbeddingResult result;
    result.values.resize(dim);
    result.prompt_tokens = ids.size();
    result.prefill_ms = prefill_ms;
    if (!core::embedding::ApplyPooling(hidden, ids.size(), dim, *pooling,
                                       result.values))
      return refuse("embedding pooling shape mismatch");
    // The hidden capture is the last block output (pre-final-norm); the
    // artifact pools the post-norm state, so run the model's output RMSNorm on
    // the host. Dim-sized with exact F32 weights, bit-deterministic.
    {
      std::vector<float> normed(dim);
      models::ForwardRMSNorm(result.values, model_->GetWeights().output_norm,
                             1e-6F, normed);
      result.values.swap(normed);
    }
    if (!(core::embedding::L2Normalize(result.values) > 0.0))
      return refuse("embedding has zero norm; refused");
    return result;
  }

  std::mutex mutex_;
  std::shared_ptr<const hip::QwenGpuModel> model_;
  hip::QwenExecutionPolicy policy_;
  std::uint32_t embed_context_;
  std::string model_id_;
  TokenizeFn tokenize_;
  std::unique_ptr<hip::QwenGpuExecutor> executor_;
};

// ---------------------------------------------------------------- rerankers
class BgeRerankRunner final : public RerankRunner {
 public:
  BgeRerankRunner(std::unique_ptr<models::bge_reranker::EngineModel> model,
                  std::function<void(models::bge_reranker::EngineSession&)> configure)
      : model_(std::move(model)), configure_(std::move(configure)) {}
  std::string_view Kind() const noexcept override { return "bge-reranker"; }
  bool NativeBatch() const noexcept override { return true; }

  std::optional<RerankResult> RerankBatch(const std::string& query,
                                          const std::vector<std::string>& documents,
                                          std::string* error) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (session_ == nullptr) {
      session_ = std::make_unique<models::bge_reranker::EngineSession>(model_.get());
      if (configure_) configure_(*session_);
    }
    double total_ms = 0.0;
    std::size_t prompt_tokens = 0;
    auto scores = session_->ScoreBatch(query, documents, true, &total_ms,
                                       &prompt_tokens, error);
    if (!scores.has_value()) return std::nullopt;
    RerankResult result;
    result.scores = std::move(*scores);
    result.prompt_tokens = prompt_tokens;
    result.prefill_ms = total_ms;
    return result;
  }

 private:
  std::mutex mutex_;
  std::unique_ptr<models::bge_reranker::EngineModel> model_;
  std::unique_ptr<models::bge_reranker::EngineSession> session_;
  std::function<void(models::bge_reranker::EngineSession&)> configure_;
};

class QwenRerankRunner final : public RerankRunner {
 public:
  explicit QwenRerankRunner(std::unique_ptr<models::qwen_reranker::EngineModel> model)
      : model_(std::move(model)),
        session_(std::make_unique<models::qwen_reranker::EngineSession>(model_.get())) {}
  std::string_view Kind() const noexcept override { return "qwen-reranker"; }
  // Pairs share the query framing; documents run in lockstep groups of up to
  // eight pooled executors (ForwardPromptBatchMulti), which shares attention
  // launches and overlaps streams but does NOT fuse the GEMMs across documents
  // (measured only ~6% faster), so this stays declared non-native.
  bool NativeBatch() const noexcept override { return false; }

  std::optional<RerankResult> RerankBatch(const std::string& query,
                                          const std::vector<std::string>& documents,
                                          std::string* error) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    double total_ms = 0.0;
    std::size_t prompt_tokens = 0;
    auto scores = session_->ScoreBatch(query, documents, std::string_view{},
                                       &total_ms, &prompt_tokens, error);
    if (!scores.has_value()) return std::nullopt;
    RerankResult result;
    result.scores = std::move(*scores);
    result.prompt_tokens = prompt_tokens;
    result.prefill_ms = total_ms;
    return result;
  }

 private:
  std::mutex mutex_;
  std::unique_ptr<models::qwen_reranker::EngineModel> model_;
  std::unique_ptr<models::qwen_reranker::EngineSession> session_;
};

#ifdef MBUFO_ENABLE_NPU
// ------------------------------------------------------- NPU Qwen3 encoders
class NpuQwenEmbeddingRunner final : public EmbeddingRunner {
 public:
  explicit NpuQwenEmbeddingRunner(std::unique_ptr<core::npu::NpuQwen3Encoder> enc)
      : enc_(std::move(enc)) {}
  std::string_view Kind() const noexcept override { return "npu-qwen-embedding"; }
  bool NativeBatch() const noexcept override { return false; }
  std::optional<std::vector<EmbeddingResult>> EmbedBatch(
      const std::vector<std::string>& texts, std::string* error) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    return SerialEmbedBatch(
        texts, error,
        [this](const std::string& text, std::string* err)
            -> std::optional<EmbeddingResult> {
          std::vector<float> values;
          std::size_t toks = 0;
          double ms = 0.0;
          if (!enc_->Embed(text, &values, &toks, &ms, err)) return std::nullopt;
          EmbeddingResult result;
          result.values = std::move(values);
          result.prompt_tokens = toks;
          result.prefill_ms = ms;
          return result;
        });
  }

 private:
  std::mutex mutex_;
  std::unique_ptr<core::npu::NpuQwen3Encoder> enc_;
};

class NpuQwenRerankRunner final : public RerankRunner {
 public:
  explicit NpuQwenRerankRunner(std::unique_ptr<core::npu::NpuQwen3Encoder> enc)
      : enc_(std::move(enc)) {}
  std::string_view Kind() const noexcept override { return "npu-qwen-reranker"; }
  bool NativeBatch() const noexcept override { return false; }
  std::optional<RerankResult> RerankBatch(const std::string& query,
                                          const std::vector<std::string>& documents,
                                          std::string* error) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    RerankResult result;
    result.scores.reserve(documents.size());
    for (const auto& doc : documents) {
      float score = 0.0F;
      std::size_t toks = 0;
      double ms = 0.0;
      if (!enc_->ScorePair(query, doc, "", &score, &toks, &ms, error))
        return std::nullopt;
      result.scores.push_back(score);
      result.prompt_tokens += toks;
      result.prefill_ms += ms;
    }
    return result;
  }

 private:
  std::mutex mutex_;
  std::unique_ptr<core::npu::NpuQwen3Encoder> enc_;
};
#endif

}  // namespace

std::unique_ptr<EmbeddingRunner> MakeEmbeddingGemma2Runner(
    std::unique_ptr<models::embeddinggemma2::EngineModel> model) {
  return std::make_unique<Gemma2EmbeddingRunner>(std::move(model));
}

std::unique_ptr<EmbeddingRunner> MakeBgeM3EmbeddingRunner(
    std::unique_ptr<models::bge_m3::EngineModel> model, bool serial_only,
    std::function<void(models::bge_m3::EngineSession&)> configure) {
  return std::make_unique<BgeM3EmbeddingRunner>(std::move(model), serial_only,
                                                std::move(configure));
}

std::unique_ptr<EmbeddingRunner> MakeQwenEmbeddingRunner(
    std::shared_ptr<const hip::QwenGpuModel> model,
    hip::QwenExecutionPolicy policy, std::uint32_t embed_context,
    std::string model_id, TokenizeFn tokenize) {
  return std::make_unique<QwenEmbeddingRunner>(
      std::move(model), policy, embed_context, std::move(model_id),
      std::move(tokenize));
}

std::unique_ptr<RerankRunner> MakeBgeRerankRunner(
    std::unique_ptr<models::bge_reranker::EngineModel> model,
    std::function<void(models::bge_reranker::EngineSession&)> configure) {
  return std::make_unique<BgeRerankRunner>(std::move(model), std::move(configure));
}

std::unique_ptr<RerankRunner> MakeQwenRerankRunner(
    std::shared_ptr<const hip::QwenGpuModel> model,
    hip::QwenExecutionPolicy policy, std::string* error) {
  std::string wrap_error;
  auto wrapped = models::qwen_reranker::EngineModel::Wrap(std::move(model), &wrap_error);
  if (wrapped == nullptr) {
    if (error != nullptr) *error = "qwen-reranker wrap failed: " + wrap_error;
    return nullptr;
  }
  wrapped->SetPolicy(policy);
  return std::make_unique<QwenRerankRunner>(std::move(wrapped));
}
#ifdef MBUFO_ENABLE_NPU
std::unique_ptr<EmbeddingRunner> MakeNpuQwenEmbeddingRunner(
    std::unique_ptr<core::npu::NpuQwen3Encoder> encoder) {
  return std::make_unique<NpuQwenEmbeddingRunner>(std::move(encoder));
}
std::unique_ptr<RerankRunner> MakeNpuQwenRerankRunner(
    std::unique_ptr<core::npu::NpuQwen3Encoder> encoder) {
  return std::make_unique<NpuQwenRerankRunner>(std::move(encoder));
}
#endif
#endif  // ENGINE_ENABLE_HIP

}  // namespace mbufo::server
