#ifndef MBUFO_MODELS_LAGUNA_ENGINE_HPP_
#define MBUFO_MODELS_LAGUNA_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/models/laguna/weights.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace mbufo::models::laguna {

struct ModelOptions {
  std::uint32_t max_context = 32768;
  std::string kv_cache_quant = "f16";
};

class Session;

/// HIP runtime for `laguna` GGUF models (Laguna-XS-2.1 33B-A3B and Laguna-S-2.1 118B-A8B)
/// on gfx1151. Supports variable per-layer Q head counts, dual Global(YaRN)/SWA(512)
/// RoPE+attention, per-head softplus attention gating, leading dense block + sigmoid-bias
/// top-k MoE via shared core::compute::gemm launchers.
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
 public:
  struct RegisteredRegion {
    const void* host_data{nullptr};
    void* device_data{nullptr};
    void* host_copy{nullptr};
    std::size_t size{0};
  };
 private:
 private:
  std::vector<RegisteredRegion> registered_regions_;
  std::vector<void*> device_allocations_;

  friend class Session;
};

class Session final {
 public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             std::uint32_t start_pos,
                             std::string* error_msg = nullptr);
  [[nodiscard]] std::span<const float> Logits() const noexcept;
  [[nodiscard]] std::uint32_t Length() const noexcept;
  [[nodiscard]] std::int32_t SampleGreedy();

  /// Rows one prefill step may carry (Forward splits longer inputs into steps of this size).
  [[nodiscard]] static constexpr std::size_t MaxChunkTokens() noexcept { return 512; }

  /// Advance every lane by one token and, optionally, run one prefill chunk, all enqueued back to back
  /// on one stream with one host synchronisation. Every lane keeps the exact single-session decode
  /// core (n == 1 kernels) and the chunk keeps the exact prefill core, so the result is bit-identical
  /// to Evaluate per lane plus Sync of the chunk. A weight-sharing fused GEMM would not be, and
  /// separate streams would race on the process-global decode scratch (MoE unique buffers, Q8
  /// activation scratch), hence the single in-order stream.
  ///
  /// `chunk` may be null (plain batched advance). `prefix` is the full prompt prefix the chunk session
  /// must hold after the call and `offset` the tokens it already holds; prefix.size() - offset <=
  /// MaxChunkTokens(). Preconditions are validated before any state is touched: a failed
  /// validation sets *refused (when non-null) and leaves every session unchanged. Any other failure
  /// is mid-pass and the sessions involved must be treated as dirty.
  [[nodiscard]] static bool AdvanceBatchWithChunk(
      std::span<Session* const> lanes, std::span<const std::int32_t> tokens,
      Session* chunk, std::span<const std::int32_t> prefix, std::size_t offset,
      std::string* error_msg = nullptr, bool* refused = nullptr);

 private:
  explicit Session(std::shared_ptr<Model> model);
  [[nodiscard]] bool EnqueueForward(std::span<const std::int32_t> tokens,
                                    std::uint32_t start_pos, hipStream_t stream,
                                    std::string* error_msg);
  [[nodiscard]] bool EnqueueLogitsCopy(hipStream_t stream, std::string* error_msg);
  [[nodiscard]] bool Init(std::string* error_msg);
  [[nodiscard]] bool ForwardTokens(std::span<const std::int32_t> tokens,
                                   std::uint32_t start_pos,
                                   std::string* error_msg);

  std::shared_ptr<Model> model_;
  hipStream_t stream_{nullptr};

  // Per-layer F16 KV cache [max_context, n_kv_heads, head_dim]
  std::vector<void*> k_cache_;
  std::vector<void*> v_cache_;

  // Activation & MoE scratch buffers
  std::int32_t* d_tokens_{nullptr};
  std::int32_t* d_argmax_{nullptr};
  float* d_hidden_{nullptr};
  float* d_normed_{nullptr};
  float* d_q_{nullptr};
  float* d_k_{nullptr};
  float* d_v_{nullptr};
  float* d_gate_logits_{nullptr};
  float* d_attn_out_{nullptr};
  float* d_proj_out_{nullptr};
  float* d_mlp_gate_{nullptr};
  float* d_mlp_up_{nullptr};
  float* d_mlp_out_{nullptr};
  float* d_shared_out_{nullptr};
  float* d_router_logits_{nullptr};
  std::int32_t* d_exp_ids_{nullptr};
  float* d_exp_weights_{nullptr};
  float* d_act_scratch_{nullptr};
  float* d_down_e_{nullptr};
  __half* d_x16_{nullptr};
  __half* d_routed_act16_{nullptr};
  std::uint32_t* d_exp_counts_{nullptr};
  std::int32_t* d_pad_bounds_{nullptr};
  std::int32_t* d_cursors_{nullptr};
  std::int32_t* d_rows_token_{nullptr};
  std::int32_t* d_rows_slot_{nullptr};
  std::int32_t* d_tiles_{nullptr};
  std::uint32_t* d_n_tiles_{nullptr};
  __half* d_wdq_scratch_{nullptr};
  float* d_logits_{nullptr};
  std::size_t routed_compact_rows_{0};
  std::size_t tile_cap_{0};
  std::size_t wdq_scratch_bytes_{0};
  float* d_attn_partials_{nullptr};
  std::size_t attn_partials_bytes_{0};
  void* d_preq_scratch_{nullptr};
  std::size_t preq_scratch_bytes_{0};

  std::vector<float> host_logits_;
  std::vector<std::int32_t> history_;

  friend class Model;
};

}  // namespace mbufo::models::laguna

#endif  // MBUFO_MODELS_LAGUNA_ENGINE_HPP_
