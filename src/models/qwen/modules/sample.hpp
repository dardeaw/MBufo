#ifndef MBUFO_MODELS_QWEN_MODULES_SAMPLE_HPP_
#define MBUFO_MODELS_QWEN_MODULES_SAMPLE_HPP_

#include <cstdint>
#include <span>

#include "src/models/qwen/modules/module_ctx.hpp"

namespace mbufo::models::qwen {

/// Greedy argmax over a logit distribution (sampling-policy module seam).
std::uint32_t SampleForward(const CpuModuleContext& ctx,
                            std::span<const float> logits) noexcept;

}  // namespace mbufo::models::qwen

#endif  // MBUFO_MODELS_QWEN_MODULES_SAMPLE_HPP_
