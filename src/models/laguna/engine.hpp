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

 private:
  explicit Session(std::shared_ptr<Model> model);
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
  float* d_logits_{nullptr};
  std::size_t routed_compact_rows_{0};

  std::vector<float> host_logits_;
  std::vector<std::int32_t> history_;

  friend class Model;
};

}  // namespace mbufo::models::laguna

#endif  // MBUFO_MODELS_LAGUNA_ENGINE_HPP_
