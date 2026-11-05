#ifndef GUFO_MODELS_GPTOSS_ENGINE_HPP_
#define GUFO_MODELS_GPTOSS_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/models/gptoss/weights.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::gptoss {

struct ModelOptions {
  std::uint32_t max_context = 8192;
  // KV cache format: "f16" (default dense path) or "q8" (shared blocked
  // INT8 layout, post-RoPE per-32-block scale; stored with the shared
  // blocked_kv::QuantizeKvRows and served back through the shared
  // blocked_kv::DequantizeKvRows into the exact sink/window kernel).
  std::string kv_cache_quant = "f16";
};

class Session;

/// Minimal single-model runtime for `gpt-oss` GGUF artifacts (GPT-OSS MoE:
/// 20B probe; 120B shares the code path). GEMM rides the shared core
/// (narrow-prequant decode / DenseF16Gemm prefill); the MoE decode rides the
/// shared vec path (MoeVecGateUp / GemvQSlot / MoeSlotEpilogueOverwrite)
/// with a model-local SwiGLU-OAI activation; attention is a model-local
/// dense F16 kernel (sinks + sliding window) over f16 or dequantized-q8 KV.
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

  /// Reprocesses the prompt in 128-token chunks.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  /// Appends one token and runs one forward step.
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  /// Runs a forward over `tokens` starting at `start_pos`.
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             std::uint32_t start_pos,
                             std::string* error_msg = nullptr);
  [[nodiscard]] std::span<const float> Logits() const noexcept;
  [[nodiscard]] std::uint32_t Length() const noexcept;
  [[nodiscard]] std::int32_t SampleGreedy();

 private:
  Session(std::shared_ptr<Model> model);

  static void ForwardLayer(Session& s, std::uint32_t il, std::uint32_t n,
                           std::uint32_t start_pos, hipStream_t stream);
  static void ForwardFinish(Session& s, std::uint32_t n,
                            std::uint32_t start_pos, hipStream_t stream);

  std::shared_ptr<Model> model_;
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class Model;
};

}  // namespace gufo::models::gptoss

#endif  // GUFO_MODELS_GPTOSS_ENGINE_HPP_
