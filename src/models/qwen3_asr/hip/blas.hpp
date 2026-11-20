#ifndef MBUFO_MODELS_QWEN3_ASR_HIP_BLAS_HPP_
#define MBUFO_MODELS_QWEN3_ASR_HIP_BLAS_HPP_

#include <cstddef>
#include <memory>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace mbufo::models::qwen3_asr::hip {

/// Multi-token BF16 projection: `Y[batch, m] = X_bf16[batch, k] *
/// A_bf16[m, k]^T` with a float32 result.
///
/// This is the model-private replacement for the shared qwen wrapper, so that
/// src/models/qwen3_asr depends on no other model's sources. Every weight in
/// this checkpoint is BF16, so there is no format switch.
void LaunchGemmBf16(hipblasHandle_t handle, const void* a_bf16,
                    const void* x_bf16, float* y, std::size_t batch_size,
                    std::size_t m, std::size_t k, hipStream_t stream);

/// hipBLASLt plans for the same operation, cached per (batch, m, k).
///
/// rocBLAS has no WMMA kernel for a transposed BF16 operand with a float32
/// output and falls back to a scalar Tensile kernel at 3-5 TFLOPS, so hipBLASLt
/// is tried first. Unlike the shared qwen implementation this class carries no
/// plan database and no tuning: it selects an algorithm by a policy measured on
/// this model's own shapes (tools/bench/asr_prefill_gemm_bench.hip).
class GemmLt {
public:
  GemmLt();
  ~GemmLt();
  GemmLt(const GemmLt&) = delete;
  GemmLt& operator=(const GemmLt&) = delete;
  GemmLt(GemmLt&&) noexcept;
  GemmLt& operator=(GemmLt&&) noexcept;

  /// Returns false when no usable plan exists, in which case the caller must
  /// fall back to LaunchGemmBf16().
  [[nodiscard]] bool Run(const void* a_bf16, const void* x_bf16, float* y,
                         std::size_t batch_size, std::size_t m, std::size_t k,
                         hipStream_t stream);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mbufo::models::qwen3_asr::hip
#endif

#endif  // MBUFO_MODELS_QWEN3_ASR_HIP_BLAS_HPP_
