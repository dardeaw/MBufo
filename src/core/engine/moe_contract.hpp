#ifndef MBUFO_CORE_ENGINE_MOE_CONTRACT_HPP_
#define MBUFO_CORE_ENGINE_MOE_CONTRACT_HPP_

// MoE engine contract: what an engine that serves a mixture-of-experts model must declare about it.
//
// Why: every MoE engine used to carry its own idea of its geometry, expert formats and decode kernels, so nothing could say
// which engines a core kernel improvement reaches, whether a new MoE model was wired into the shared row-dot layer
// (core/quant/row_dot_traits.hpp) or silently shipped its own kernels, or whether its expert byte traffic matched what
// the geometry implies. An engine registered with `EngineBackendInfo::moe = true` must hand out a model that implements
// MoeCapable; the registry refuses to create it otherwise, and refuses a MoeCapable model registered as dense. The declaration
// is checked structurally by ValidateMoeContract (no GPU needed) and, in tools/bench/moe_contract_check, against the row-dot
// format registry for every engine that claims core kernels.
//
// Dense variants of an MoE-capable engine (e.g. a dense Gemma 4) report moe_layers == 0.

#include <cstdint>
#include <string>
#include <vector>

namespace mbufo::core::engine {

/// One distinct set of expert weight formats and how many MoE layers use it (type codes are core::GgmlType values).
struct MoeFormatUse {
  int gate_type{-1};
  int up_type{-1};  ///< equals gate_type when gate and up are one merged tensor
  int down_type{-1};
  std::uint32_t layers{0};
};

struct MoeGeometry {
  std::uint32_t moe_layers{0};  ///< layers whose FFN is a routed MoE (0 = dense variant)
  std::uint32_t experts{0};
  std::uint32_t top_k{0};
  std::uint32_t hidden{0};
  std::uint32_t expert_ff{0};
  std::uint32_t shared_ff{0};   ///< always-on shared / parallel dense FFN width, 0 when absent
  std::vector<MoeFormatUse> formats;

  /// Records one layer's formats, merging it into an existing identical entry.
  void AddLayer(int gate_type, int up_type, int down_type) {
    ++moe_layers;
    for (auto& f : formats) {
      if (f.gate_type == gate_type && f.up_type == up_type && f.down_type == down_type) {
        ++f.layers;
        return;
      }
    }
    formats.push_back({gate_type, up_type, down_type, 1});
  }
};

/// Where an engine's decode expert kernels come from.
enum class MoeKernelOrigin {
  kCoreRowDot,    ///< expert dots go through the core row-dot layer (RowDot<F>): a core kernel change reaches this engine
  kModelPrivate,  ///< the engine's own kernels; private_reason must say why they cannot use the core layer
};

struct MoeKernelInfo {
  MoeKernelOrigin origin{MoeKernelOrigin::kModelPrivate};
  std::string gate_up_kernel;   ///< human-readable kernel family for the gate/up (+ activation) stage
  std::string down_kernel;      ///< ... for the down stage
  std::string private_reason;   ///< required for kModelPrivate
  bool masks_inactive_slots{false};  ///< negative expert ids / padding slots never touch memory or leave garbage
  bool graph_capture_safe{false};    ///< decode MoE launches have no host sync or data-dependent launch
};

/// Implemented by the EngineModel of every MoE-capable engine.
class MoeCapable {
 public:
  virtual ~MoeCapable() = default;
  [[nodiscard]] virtual MoeGeometry MoeGeometryInfo() const = 0;
  [[nodiscard]] virtual MoeKernelInfo MoeKernelsInfo() const = 0;
};

struct MoeContractReport {
  std::vector<std::string> problems;
  [[nodiscard]] bool ok() const noexcept { return problems.empty(); }
};

/// Structural validation: geometry coherence, every expert format sized for its row width, formats cover exactly the
/// declared MoE layers, kernel declaration complete (private kernels carry a reason).
[[nodiscard]] MoeContractReport ValidateMoeContract(const MoeGeometry& geometry, const MoeKernelInfo& kernels);

/// Routed-expert bytes one decoded token reads, implied by the geometry and formats (top_k experts of every MoE layer:
/// gate + up rows of expert_ff x hidden plus down rows of hidden x expert_ff). 0 for dense variants or unknown formats.
[[nodiscard]] std::uint64_t ExpectedRoutedBytesPerToken(const MoeGeometry& geometry);

[[nodiscard]] const char* MoeKernelOriginName(MoeKernelOrigin origin) noexcept;

}  // namespace mbufo::core::engine

#endif  // MBUFO_CORE_ENGINE_MOE_CONTRACT_HPP_
