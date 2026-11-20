#include "src/core/engine/moe_contract.hpp"

#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace mbufo::core::engine {
namespace {

std::size_t RowBytes(int type, std::uint64_t cols) {
  if (type < 0) return 0;
  return ::mbufo::quant::QuantizedRowBytes(static_cast<core::GgmlType>(type), static_cast<std::size_t>(cols));
}

}  // namespace

const char* MoeKernelOriginName(MoeKernelOrigin origin) noexcept {
  return origin == MoeKernelOrigin::kCoreRowDot ? "core-row-dot" : "model-private";
}

std::uint64_t ExpectedRoutedBytesPerToken(const MoeGeometry& g) {
  if (g.moe_layers == 0 || g.top_k == 0) return 0;
  std::uint64_t total = 0;
  for (const auto& f : g.formats) {
    const std::size_t gate = RowBytes(f.gate_type, g.hidden), up = RowBytes(f.up_type, g.hidden),
                      down = RowBytes(f.down_type, g.expert_ff);
    if (gate == 0 || up == 0 || down == 0) return 0;
    const std::uint64_t per_expert = static_cast<std::uint64_t>(g.expert_ff) * (gate + up) +
                                     static_cast<std::uint64_t>(g.hidden) * down;
    total += static_cast<std::uint64_t>(f.layers) * g.top_k * per_expert;
  }
  return total;
}

MoeContractReport ValidateMoeContract(const MoeGeometry& g, const MoeKernelInfo& k) {
  MoeContractReport r;
  auto bad = [&](std::string msg) { r.problems.push_back(std::move(msg)); };
  if (g.moe_layers == 0) {  // dense variant of an MoE-capable engine: nothing may claim experts
    if (g.experts != 0 || g.top_k != 0 || !g.formats.empty())
      bad("moe_layers is 0 but experts / top_k / formats are set");
    return r;
  }
  if (g.experts == 0) bad("experts is 0 although moe_layers > 0");
  if (g.top_k == 0 || g.top_k > g.experts) bad("top_k must be in [1, experts]");
  if (g.hidden == 0 || g.expert_ff == 0) bad("hidden and expert_ff must be set");
  if (g.formats.empty()) bad("no expert formats declared");
  std::uint64_t counted = 0;
  for (const auto& f : g.formats) {
    counted += f.layers;
    if (f.layers == 0) bad("a format entry covers zero layers");
    if (RowBytes(f.gate_type, g.hidden) == 0) bad("gate type " + std::to_string(f.gate_type) + " is not sized for hidden " + std::to_string(g.hidden));
    if (RowBytes(f.up_type, g.hidden) == 0) bad("up type " + std::to_string(f.up_type) + " is not sized for hidden " + std::to_string(g.hidden));
    if (RowBytes(f.down_type, g.expert_ff) == 0) bad("down type " + std::to_string(f.down_type) + " is not sized for expert_ff " + std::to_string(g.expert_ff));
  }
  if (counted != g.moe_layers) bad("format entries cover " + std::to_string(counted) + " layers, geometry declares " + std::to_string(g.moe_layers));
  if (k.gate_up_kernel.empty() || k.down_kernel.empty()) bad("kernel families must be named");
  if (k.origin == MoeKernelOrigin::kModelPrivate && k.private_reason.empty()) bad("model-private kernels need a private_reason");
  if (k.origin == MoeKernelOrigin::kCoreRowDot && !k.private_reason.empty()) bad("core row-dot kernels must not carry a private_reason");
  return r;
}

}  // namespace mbufo::core::engine
