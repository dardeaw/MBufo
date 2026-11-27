// Encoder-side runner interfaces: the embedding and reranking counterparts of
// TextModelRunner. A model kind that serves /v1/embeddings or /v1/reranking
// implements one of these; the batch entry point is pure virtual so a new
// encoder cannot compile until it has chosen between a native batched forward
// and an explicit serial fallback (SerialEmbedBatch). The serve layer only ever
// calls the batch form; a single text or a single query is a batch of one.
#ifndef MBUFO_CLI_SERVE_ENCODER_RUNNER_HPP_
#define MBUFO_CLI_SERVE_ENCODER_RUNNER_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"

namespace mbufo::models::bge_m3 {
class EngineModel;
class EngineSession;
}  // namespace mbufo::models::bge_m3
namespace mbufo::models::bge_reranker {
class EngineModel;
class EngineSession;
}  // namespace mbufo::models::bge_reranker
namespace mbufo::models::embeddinggemma2 {
class EngineModel;
}
namespace mbufo::hip {
class QwenGpuModel;
struct QwenExecutionPolicy;
}  // namespace mbufo::hip

namespace mbufo::server {

class EmbeddingRunner {
 public:
  virtual ~EmbeddingRunner() = default;
  [[nodiscard]] virtual std::string_view Kind() const noexcept = 0;
  /// True when one call fuses the texts into shared device work; false when
  /// the runner declared SerialEmbedBatch.
  [[nodiscard]] virtual bool NativeBatch() const noexcept = 0;
  /// One L2-normalized vector per text, in input order. All-or-nothing:
  /// nullopt always carries a human-readable reason in `error`.
  [[nodiscard]] virtual std::optional<std::vector<EmbeddingResult>> EmbedBatch(
      const std::vector<std::string>& texts, std::string* error) = 0;

 protected:
  /// Explicit fallback for runners with no batched forward.
  [[nodiscard]] static std::optional<std::vector<EmbeddingResult>>
  SerialEmbedBatch(
      const std::vector<std::string>& texts, std::string* error,
      const std::function<std::optional<EmbeddingResult>(const std::string&,
                                                         std::string*)>& one);
};

class RerankRunner {
 public:
  virtual ~RerankRunner() = default;
  [[nodiscard]] virtual std::string_view Kind() const noexcept = 0;
  [[nodiscard]] virtual bool NativeBatch() const noexcept = 0;
  /// One score per document, in input order (one query against N documents is
  /// the batch). nullopt carries a reason in `error`.
  [[nodiscard]] virtual std::optional<RerankResult> RerankBatch(
      const std::string& query, const std::vector<std::string>& documents,
      std::string* error) = 0;
};

#if defined(ENGINE_ENABLE_HIP)
using TokenizeFn = std::function<std::vector<std::uint32_t>(const std::string&)>;

/// EmbeddingGemma 2: one fused device forward over all texts.
[[nodiscard]] std::unique_ptr<EmbeddingRunner> MakeEmbeddingGemma2Runner(
    std::unique_ptr<models::embeddinggemma2::EngineModel> model);

/// bge-m3: padded multi-sequence encoder forward on the device. With
/// `serial_only` (NPU lane, which is per-sequence) it declares the serial
/// fallback. `configure` runs on the lazily created session.
[[nodiscard]] std::unique_ptr<EmbeddingRunner> MakeBgeM3EmbeddingRunner(
    std::unique_ptr<models::bge_m3::EngineModel> model, bool serial_only,
    std::function<void(models::bge_m3::EngineSession&)> configure);

/// Qwen3-Embedding on the shared Qwen weights: one prefill per text with
/// hidden capture (no cross-session hidden batching yet, so it declares the
/// serial fallback).
[[nodiscard]] std::unique_ptr<EmbeddingRunner> MakeQwenEmbeddingRunner(
    std::shared_ptr<const hip::QwenGpuModel> model,
    hip::QwenExecutionPolicy policy, std::uint32_t embed_context,
    std::string model_id, TokenizeFn tokenize);

[[nodiscard]] std::unique_ptr<RerankRunner> MakeBgeRerankRunner(
    std::unique_ptr<models::bge_reranker::EngineModel> model,
    std::function<void(models::bge_reranker::EngineSession&)> configure);

[[nodiscard]] std::unique_ptr<RerankRunner> MakeQwenRerankRunner(
    std::shared_ptr<const hip::QwenGpuModel> model,
    hip::QwenExecutionPolicy policy, std::string* error);
#endif

}  // namespace mbufo::server

#endif  // MBUFO_CLI_SERVE_ENCODER_RUNNER_HPP_
