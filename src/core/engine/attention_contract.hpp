#ifndef MBUFO_CORE_ENGINE_ATTENTION_CONTRACT_HPP_
#define MBUFO_CORE_ENGINE_ATTENTION_CONTRACT_HPP_

// Attention engine contract: what every engine must declare about the attention it runs.
//
// Why: attention kernels are shared infrastructure (row-chunk split-K decode, WMMA flash prefill, blocked-Q8 KV), but each
// engine used to carry its own idea of its head geometry, KV layout and kernels, so nothing could say which engines a core
// kernel improvement reaches, which engines silently ship a private kernel, or whether an engine keeps the property the
// speculative-decoding verifier depends on: a query row's output must not depend on the other rows batched with it
// ("row invariance", bitwise: row r of an n-row call == the same row run alone).
//
// The contract does NOT force one kernel shape on every architecture. A model whose attention the core kernels cannot serve
// (MLA, linear / recurrent attention, an upstream port) declares kModelPrivate with a mandatory reason; a model that claims
// the core kernels must declare softmax GQA geometry the core kernels accept and must be row invariant. The registry refuses
// to create an engine whose model does not implement AttentionCapable or whose declaration does not validate
// (EngineBackendInfo::attention = false opts out, for engines that have no attention at all), and tools/bench/attn_contract_check
// runs the declared core kernels against the row-invariance and accuracy requirements.

#include <cstdint>
#include <string>
#include <vector>

namespace mbufo::core::engine {

enum class AttentionKind {
  kSoftmaxGqa,       ///< softmax attention over a KV cache (MHA / GQA / MQA, optional sliding window and sink)
  kMla,              ///< multi-head latent attention (compressed KV)
  kLinearRecurrent,  ///< linear / state-space / gated-delta recurrent mixing instead of a KV cache
};

enum class AttentionKvFormat {
  kF16,         ///< F16 [kv_len][kv_heads][head_dim] planes
  kBlockedQ8,   ///< shared blocked-Q8 layout (34-byte blocks: 32 int8 + F16 scale)
  kEither,      ///< chosen at load by the KV-quant option; both layouts are served
  kModelSpecific,
};

/// One distinct attention shape and how many layers use it.
struct AttentionGroup {
  AttentionKind kind{AttentionKind::kSoftmaxGqa};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t window{0};  ///< sliding window in keys, 0 = full causal
  bool sink{false};         ///< learned per-head softmax sink
  AttentionKvFormat kv_format{AttentionKvFormat::kEither};
  std::uint32_t layers{0};

  [[nodiscard]] bool SameShape(const AttentionGroup& o) const noexcept {
    return kind == o.kind && heads == o.heads && kv_heads == o.kv_heads && head_dim == o.head_dim &&
           window == o.window && sink == o.sink && kv_format == o.kv_format;
  }
};

struct AttentionGeometry {
  std::vector<AttentionGroup> groups;

  /// Records one layer, merging it into an existing group of the same shape.
  void AddLayer(AttentionGroup layer) {
    layer.layers = 1;
    for (auto& g : groups) {
      if (g.SameShape(layer)) {
        ++g.layers;
        return;
      }
    }
    groups.push_back(layer);
  }
  [[nodiscard]] std::uint32_t TotalLayers() const noexcept {
    std::uint32_t n = 0;
    for (const auto& g : groups) n += g.layers;
    return n;
  }
};

enum class AttentionKernelOrigin {
  kCore,          ///< the default path runs core attention kernels: a core improvement reaches this engine
  kModelPrivate,  ///< the engine's own kernels; private_reason must say why the core kernels cannot serve it
};

/// One execution path (decode or prefill) of the engine's attention.
struct AttentionPathInfo {
  AttentionKernelOrigin origin{AttentionKernelOrigin::kModelPrivate};
  std::string kernel;          ///< human-readable kernel family
  std::string private_reason;  ///< required for kModelPrivate
  std::string exceptions;      ///< optional: sub-paths that are NOT on the declared origin (e.g. F16 layers)
  /// Bitwise: a row's output does not depend on the other rows of the batch (verification batches == single-token decode).
  /// REQUIRED of a core decode path (speculative verification depends on it); informational for prefill, where a row of an
  /// n-row call may differ from the same row run alone (the core WMMA flash kernels do: tile shape follows n).
  bool row_invariant{false};
};

struct AttentionKernelInfo {
  AttentionPathInfo decode;   ///< 1..few query rows per call, long KV
  AttentionPathInfo prefill;  ///< many query rows per call
};

/// Implemented by the EngineModel of every engine that runs attention.
class AttentionCapable {
 public:
  virtual ~AttentionCapable() = default;
  [[nodiscard]] virtual AttentionGeometry AttentionGeometryInfo() const = 0;
  [[nodiscard]] virtual AttentionKernelInfo AttentionKernelsInfo() const = 0;
};

struct AttentionContractReport {
  std::vector<std::string> problems;
  [[nodiscard]] bool ok() const noexcept { return problems.empty(); }
};

/// Structural validation (no GPU): geometry coherence, private paths carry a reason, core paths only claim softmax GQA
/// shapes the core kernels accept (heads % kv_heads == 0, head_dim % 32 == 0 for blocked Q8, no sink on non-softmax kinds)
/// and (decode) are row invariant.
[[nodiscard]] AttentionContractReport ValidateAttentionContract(const AttentionGeometry& geometry,
                                                                const AttentionKernelInfo& kernels);

[[nodiscard]] const char* AttentionKindName(AttentionKind kind) noexcept;
[[nodiscard]] const char* AttentionKernelOriginName(AttentionKernelOrigin origin) noexcept;
[[nodiscard]] const char* AttentionKvFormatName(AttentionKvFormat format) noexcept;

}  // namespace mbufo::core::engine

#endif  // MBUFO_CORE_ENGINE_ATTENTION_CONTRACT_HPP_