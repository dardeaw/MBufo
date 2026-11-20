#ifndef MBUFO_MODELS_QWEN_HIP_OPS_HPP_
#define MBUFO_MODELS_QWEN_HIP_OPS_HPP_

// Compatibility umbrella. New call sites should include the narrow operation
// family they use.
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/ops/token.hpp"

#endif  // MBUFO_MODELS_QWEN_HIP_OPS_HPP_
