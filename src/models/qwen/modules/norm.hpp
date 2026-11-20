#ifndef MBUFO_MODELS_QWEN_MODULES_NORM_HPP_
#define MBUFO_MODELS_QWEN_MODULES_NORM_HPP_

#include <span>

#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"

namespace mbufo::models::qwen {

/// Layer norm (attn pre-norm, ffn pre-norm, final output norm):
/// out = (x / rms(x) + eps) * weight.
///
/// CPU reproduces `ForwardRMSNorm`; HIP launches the existing RMSNorm kernel.
/// In-place is safe (`x == out`) on the CPU reference path.
void NormForward(const CpuModuleContext& ctx, const NormLayerView& view,
                 std::span<const float> x, std::span<float> out) noexcept;
void NormForward(const HipModuleContext& ctx, const NormLayerView& view,
                 std::span<const float> x, std::span<float> out) noexcept;

}  // namespace mbufo::models::qwen

#endif  // MBUFO_MODELS_QWEN_MODULES_NORM_HPP_
