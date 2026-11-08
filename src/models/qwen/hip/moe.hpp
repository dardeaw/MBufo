#ifndef GUFO_MODELS_QWEN_HIP_MOE_HPP_
#define GUFO_MODELS_QWEN_HIP_MOE_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::hip {

// Routed-MoE forward for qwen35moe layers (HIP): router SGEMM, shared
// RouterTopK, RoutedCompact, device-built tiles, RoutedF16Gemm gate/up/down,
// Q8_0 shared expert (narrow decode / batched prefill), and the shared
// fn_compat::MoeEpilogue combine. Prefill and decode share this path;
// residual add stays with the caller.
bool MoeForwardHip(const core::ModelConfig& config,
                   const models::QwenLayerWeights& layer, const float* x,
                   const void* x_bf16, float* out, std::uint32_t n_tokens,
                   const QwenMoeScratch& scratch, hipStream_t stream,
                   hipblasHandle_t blas, void* preq_scratch,
                   std::size_t preq_bytes, std::string* error_msg);

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_MOE_HPP_
