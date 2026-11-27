#pragma once
// Declarative op-graph executor for encoder-style NPU models.
//
// A model is a flat list of `Op`s over named row-major float buffers
// ([rows, cols]); `Linear` runs on XDNA2 through the shared
// `NpuQuantGemmEngine` (Q8 bfp16ebs8 path, weights resident in BOs), every
// other op runs on the host. The same executor serves bidirectional BERT
// encoders (bge-m3 / bge-reranker) and causal GQA Qwen3 encoders
// (Qwen3-Embedding / Qwen3-Reranker); only the graph description and the
// weight table differ.
//
// Compiled only when MBUFO_ENABLE_NPU=ON.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/compute/npu/npu_quant_gemm.hpp"

namespace mbufo::core::npu::graph {

// ---------------------------------------------------------------- weights --

enum class TensorKind : std::uint8_t { kF32, kBf16, kF16, kQ8_0 };

/// Read-only weight tensor, row-major [rows, cols] (GGUF `[ne0=cols, ne1=rows]`
/// order). Non-owning over GGUF mmap / an external F32 table, or owning.
struct Tensor {
  TensorKind kind{TensorKind::kF32};
  std::size_t rows{1};
  std::size_t cols{0};
  const void* data{nullptr};
  std::vector<float> owned;  ///< backs `data` when the tensor owns its floats

  [[nodiscard]] static Tensor ViewF32(const float* p, std::size_t rows,
                                      std::size_t cols);
  [[nodiscard]] static Tensor Owned(std::vector<float> v, std::size_t rows,
                                    std::size_t cols);
  [[nodiscard]] static Tensor View(TensorKind kind, const void* p,
                                   std::size_t rows, std::size_t cols);

  /// Dequantize / convert row `r` into `dst[cols]`.
  void ReadRow(std::size_t r, float* dst) const;
  /// Dequantize the whole tensor into `out` (resized to rows*cols).
  void ReadAll(std::vector<float>* out) const;
};

class WeightStore {
 public:
  void Put(std::string name, Tensor t) { map_[std::move(name)] = std::move(t); }
  [[nodiscard]] const Tensor* Find(const std::string& name) const {
    const auto it = map_.find(name);
    return it == map_.end() ? nullptr : &it->second;
  }

 private:
  std::unordered_map<std::string, Tensor> map_;
};

// ------------------------------------------------------------------- ops ---

enum class OpKind : std::uint8_t {
  kEmbed,        ///< out = table[ids] (+ pos[t]) (+ type_row) , * scale
  kLayerNorm,    ///< out = LN(in) * w + b
  kRmsNorm,      ///< out = rms(in) * w
  kLinear,       ///< out = in @ W^T (+ bias)   -- NPU engine GEMM
  kAttention,    ///< out = softmax(QK^T/sqrt(d)) V (GQA, causal|bidir, QK-norm, RoPE)
  kGelu,         ///< out = gelu(in)  (exact erf)
  kSwiGlu,       ///< out = silu(in) * in2
  kResidual,     ///< out = in + in2
  kPool,         ///< out[1, cols] = cls | mean | last row of in
  kL2Norm,       ///< out = in / ||in||  (per row)
  kRowDot,       ///< out[1, k] = dot(in[last row], W.row(ids[j]))  (lm-head probe)
};

enum class PoolMode : std::uint8_t { kCls, kMean, kLast };

struct Op {
  OpKind kind{OpKind::kResidual};
  std::string out;
  std::string in;
  std::string in2;  ///< second input (Residual/SwiGlu up/Attention K)
  std::string in3;  ///< Attention V
  std::string w;    ///< weight / table / norm scale
  std::string b;    ///< bias / norm shift / Embed position table
  std::string w2;   ///< Embed token-type table / Attention q_norm
  std::string w3;   ///< Attention k_norm
  float eps{1e-5F};
  float scale{1.0F};
  // Attention
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  bool causal{false};
  float rope_theta{0.0F};  ///< 0 = no RoPE; NEOX split-half rotation otherwise
  // Pool
  PoolMode pool{PoolMode::kCls};
  // RowDot
  std::vector<std::uint32_t> ids;
};

struct Graph {
  std::string name;
  std::vector<Op> ops;
  std::string output;  ///< buffer returned by Run()
};

// --------------------------------------------------------------- builders --

struct BertGraphConfig {
  std::size_t layers{24};
  std::size_t hidden{1024};
  std::size_t heads{16};
  std::size_t inter{4096};
  float eps{1e-5F};
};

/// Post-norm BERT/XLM-R encoder, CLS output: weight names
/// `tok`, `pos`, `type`, `emb_ln.w/b`, `L{i}.{q,k,v,o,up,down}.{w,b}`,
/// `L{i}.attn_ln.w/b`, `L{i}.out_ln.w/b`. Output buffer = `cls` [1, hidden].
[[nodiscard]] Graph BuildBertClsGraph(const BertGraphConfig& cfg);

struct Qwen3GraphConfig {
  std::size_t layers{28};
  std::size_t hidden{1024};
  std::size_t heads{16};
  std::size_t kv_heads{8};
  std::size_t head_dim{128};
  std::size_t inter{3072};
  float eps{1e-6F};
  float rope_theta{1000000.0F};
};

/// Pre-norm causal GQA Qwen3 (QK-norm + RoPE + SwiGLU), last-token pooling.
/// Weight names `tok`, `out_norm.w`, `L{i}.attn_norm.w`, `L{i}.{q,k,v,o}.w`,
/// `L{i}.q_norm.w`, `L{i}.k_norm.w`, `L{i}.ffn_norm.w`, `L{i}.{gate,up,down}.w`.
/// Embedding variant: output `emb` [1, hidden] (final-norm + L2).
[[nodiscard]] Graph BuildQwen3EmbeddingGraph(const Qwen3GraphConfig& cfg);
/// Reranker variant: output `logits` [1, ids.size()] (final-norm, lm-head rows
/// of `ids`, tied to `tok`).
[[nodiscard]] Graph BuildQwen3RerankGraph(const Qwen3GraphConfig& cfg,
                                          std::vector<std::uint32_t> probe_ids);

// --------------------------------------------------------------- executor --

struct RunStats {
  double total_ms{0.0};
  double npu_ms{0.0};      ///< time inside NpuQuantGemmEngine kernel chains
  double linear_ms{0.0};   ///< wall time of all Linear ops (pack + NPU + unpack)
  double attention_ms{0.0};
  double other_host_ms{0.0};
};

enum class LinearBackend : std::uint8_t {
  kNpuQ8,   ///< NPU engine, Q8 bfp16ebs8 resident BOs
  kHostF32  ///< exact host fp32 reference (no NPU; for regression checks)
};

class GraphExecutor {
 public:
  GraphExecutor() = default;
  GraphExecutor(const GraphExecutor&) = delete;
  GraphExecutor& operator=(const GraphExecutor&) = delete;

  /// Binds graph + weights. `weights` must outlive the executor. For
  /// kNpuQ8 every Linear weight is packed into resident BOs through `engine`
  /// (shared; must be Init()ed). Validates every referenced weight.
  [[nodiscard]] bool Prepare(Graph graph, const WeightStore* weights,
                             std::shared_ptr<NpuQuantGemmEngine> engine,
                             LinearBackend backend, std::string* error);

  /// Runs the graph for token `ids`. Writes the output buffer row-major to
  /// `out` (rows*cols floats; rows==1 for the pooled outputs).
  [[nodiscard]] bool Run(std::span<const std::uint32_t> ids,
                         std::vector<float>* out, RunStats* stats,
                         std::string* error);

  [[nodiscard]] bool prepared() const noexcept { return prepared_; }
  [[nodiscard]] const Graph& graph() const noexcept { return graph_; }

 private:
  struct Buf {
    std::vector<float> v;
    std::size_t rows{0};
    std::size_t cols{0};
  };
  bool RunOp(const Op& op, std::span<const std::uint32_t> ids,
             RunStats* stats, std::string* error);
  Buf* Get(const std::string& name) {
    const auto it = bufs_.find(name);
    return it == bufs_.end() ? nullptr : &it->second;
  }

  bool prepared_{false};
  Graph graph_;
  const WeightStore* weights_{nullptr};
  std::shared_ptr<NpuQuantGemmEngine> engine_;
  LinearBackend backend_{LinearBackend::kNpuQ8};
  std::unordered_map<std::string, QuantWeightMatrix> packed_;
  std::unordered_map<std::string, std::vector<float>> hostw_;  // kHostF32 only
  std::unordered_map<std::string, std::vector<float>> bias_;  // resolved f32
  std::unordered_map<std::string, std::vector<float>> small_;  // norm scales etc.
  std::unordered_map<std::string, Buf> bufs_;
};

/// Process-wide shared quant GEMM engine (lazy; one HRX session for every
/// graph model so BO memory is paid once). Returns nullptr + error on failure.
[[nodiscard]] std::shared_ptr<NpuQuantGemmEngine> SharedQuantEngine(
    std::string* error);
/// Drops the process-wide engine reference (models still holding it keep it).
void ReleaseSharedQuantEngine();

}  // namespace mbufo::core::npu::graph
