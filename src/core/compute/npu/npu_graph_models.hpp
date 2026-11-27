#pragma once
// Model bindings for the NPU op-graph executor (npu_op_graph.hpp):
//   * NpuGraphBgeRuntime  - bge-m3 / bge-reranker-v2-m3 encoder (24L XLM-R)
//   * NpuQwen3Encoder     - Qwen3-Embedding-0.6B / Qwen3-Reranker-0.6B (28L)
//   * NpuGraphEmbGemma2   - EmbeddingGemma 2 (24L Gemma-style, sliding/full,
//   PLE)
// Compiled only when MBUFO_ENABLE_NPU=ON.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/compute/npu/npu_op_graph.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/bge_m3/forward.hpp"
#include "src/models/embeddinggemma2/engine.hpp"
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
                                std::vector<float>* cls_out, double* prefill_ms,
                                std::string* error);
  [[nodiscard]] bool resident() const noexcept { return exec_.prepared(); }
  [[nodiscard]] const graph::RunStats& last_stats() const noexcept {
    return stats_;
  }

private:
  graph::WeightStore store_;
  graph::GraphExecutor exec_;
  graph::RunStats stats_;
};

/// Which EmbeddingGemma 2 NPU runtime to use. Graph (Q8 on the shared engine)
/// is the default: measured cos >= 0.9992 vs the CPU/GPU reference, faster than
/// the dedicated runtime, inputs up to 512 tokens instead of 128.
/// Only an explicit MBUFO_NPU_QUANT other than q8 (q4) keeps the dedicated
/// runtime, because the graph has no Q4 path.
[[nodiscard]] inline bool EmbGemma2UseGraph() {
  const char* q = std::getenv("MBUFO_NPU_QUANT");
  if (q != nullptr && q[0] != '\0') {
    const std::string qs(q);
    return qs == "q8" || qs == "Q8";
  }
  return true;
}

/// EmbeddingGemma 2 as a declarative graph (npu_op_graph.hpp
/// BuildEmbGemma2Graph) on the shared Q8 engine. Same output contract as
/// NpuEmbGemma2Runtime::Forward: L2-normalized `out_dim` embedding.
class NpuGraphEmbGemma2 {
public:
  /// Host attention has no fixed tile, so unlike the 128-row BF16 kernel chain
  /// any length up to this bound runs.
  static constexpr std::size_t kMaxTokens = 512;

  /// `reader` (GGUF mmap) must outlive the runtime: weights are zero-copy
  /// views.
  [[nodiscard]] bool Init(const core::GgufReader& reader,
                          const models::embeddinggemma2::Config& cfg,
                          graph::LinearBackend backend, std::string* error);
  [[nodiscard]] bool Forward(std::span<const std::uint32_t> ids,
                             std::vector<float>* out, double* ms,
                             std::string* error);
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
                           std::size_t* tokens, double* ms, std::string* error);
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
