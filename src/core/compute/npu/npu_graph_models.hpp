#pragma once
// Model bindings for the NPU op-graph executor (npu_op_graph.hpp):
//   * NpuGraphBgeRuntime  - bge-m3 / bge-reranker-v2-m3 encoder (24L XLM-R)
//   * NpuQwen3Encoder     - Qwen3-Embedding-0.6B / Qwen3-Reranker-0.6B (28L)
// Compiled only when MBUFO_ENABLE_NPU=ON.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/compute/npu/npu_op_graph.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/bge_m3/forward.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace mbufo::core::npu {

/// bge-m3 / bge-reranker encoder as a declarative graph on the Q8 engine.
/// Drop-in twin of NpuBgeRuntime (same ForwardCls contract).
class NpuGraphBgeRuntime {
 public:
  /// `weights` must outlive the runtime (zero-copy views into its F32 table).
  [[nodiscard]] bool Init(const models::bge_m3::BgeM3F32Weights& weights,
                          graph::LinearBackend backend, std::string* error);
  [[nodiscard]] bool ForwardCls(std::span<const std::uint32_t> ids,
                                std::vector<float>* cls_out,
                                double* prefill_ms, std::string* error);
  [[nodiscard]] bool resident() const noexcept { return exec_.prepared(); }
  [[nodiscard]] const graph::RunStats& last_stats() const noexcept {
    return stats_;
  }

 private:
  graph::WeightStore store_;
  graph::GraphExecutor exec_;
  graph::RunStats stats_;
};

/// Qwen3 dense encoder (causal GQA, QK-norm, RoPE, SwiGLU) from a Q8_0/F16/
/// BF16/F32 GGUF, in embedding (last-token pool + L2) or reranker (yes/no
/// logit) role.
class NpuQwen3Encoder {
 public:
  enum class Role : std::uint8_t { kEmbedding, kReranker };

  [[nodiscard]] static std::unique_ptr<NpuQwen3Encoder> Load(
      const std::string& gguf_path, Role role, graph::LinearBackend backend,
      std::string* error);

  /// Tokenizes `text`, appends EOS when missing (artifact convention), runs
  /// the graph and returns the L2-normalized embedding.
  [[nodiscard]] bool Embed(std::string_view text, std::vector<float>* out,
                           std::size_t* tokens, double* ms,
                           std::string* error);
  [[nodiscard]] bool EmbedIds(std::span<const std::uint32_t> ids,
                              std::vector<float>* out, double* ms,
                              std::string* error);

  /// Renders the Qwen3-Reranker template for (query, document) and returns
  /// P(yes) (softmax over the yes/no logits at the last token).
  [[nodiscard]] bool ScorePair(std::string_view query,
                               std::string_view document,
                               std::string_view instruction, float* score,
                               std::size_t* tokens, double* ms,
                               std::string* error);
  [[nodiscard]] bool ScoreIds(std::span<const std::uint32_t> ids, float* score,
                              float* logit_yes, float* logit_no, double* ms,
                              std::string* error);

  [[nodiscard]] Role role() const noexcept { return role_; }
  [[nodiscard]] std::size_t hidden() const noexcept { return cfg_.hidden; }
  [[nodiscard]] std::size_t max_tokens() const noexcept { return 4096; }
  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const graph::RunStats& last_stats() const noexcept {
    return stats_;
  }

 private:
  NpuQwen3Encoder() = default;
  Role role_{Role::kEmbedding};
  graph::Qwen3GraphConfig cfg_;
  std::unique_ptr<core::GgufReader> reader_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  graph::WeightStore store_;
  graph::GraphExecutor exec_;
  graph::RunStats stats_;
  std::uint32_t yes_id_{0};
  std::uint32_t no_id_{0};
};

}  // namespace mbufo::core::npu
