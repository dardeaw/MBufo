#ifndef GUFO_MODELS_QWEN_MODULES_MOE_HPP_
#define GUFO_MODELS_QWEN_MODULES_MOE_HPP_

#include <cstddef>
#include <span>

#include "src/core/model_config.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::models::qwen {

/// Routed-MoE FFN (qwen35moe), CPU reference backend:
///   out = sum_s w_s * (down_e(silu(gate_e(x)) .* up_e(x)))
///       + sigmoid(gate_inp_shexp . x) * (down_shexp(silu(gate_shexp(x)) .*
///         up_shexp(x)))
/// with a softmax router, top-k selection and renormalized weights. Matches
/// the shared RouterTopK kernel plus llama build_moe_ffn (SOFTMAX, norm_w).
/// x/out are [batch, hidden_size] row-major; batch is derived from sizes.
void MoeForwardCpu(const core::ModelConfig& config,
                   const QwenLayerWeights& layer, std::span<const float> x,
                   std::span<float> out);

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_MODULES_MOE_HPP_
