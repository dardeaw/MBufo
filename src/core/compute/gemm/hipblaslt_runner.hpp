#ifndef GUFO_CORE_COMPUTE_GEMM_HIPBLASLT_RUNNER_HPP_
#define GUFO_CORE_COMPUTE_GEMM_HIPBLASLT_RUNNER_HPP_

// Shared hipBLASLt plan-cache runner for dense F16/BF16 prefill projections
// (Phase B-3, staged). Mechanical搬入 of flash-next BlasLt (qwen38_flash_next
// kernels/rocm/blaslt.{hpp,cpp}): the plan cache, exact-shape keys, and the
// workspace-free solution selection are unchanged.
//
// One relocation adaptation: the two model-owned fallback kernels for shapes
// whose library kernel is position-variant (flash-next OwnF16/OwnBf16 quirks)
// are injected as hooks. With no hook set the runner declines those shapes
// (returns false) so the caller keeps its legacy path; it never silently
// reroutes them through the library.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace gufo::core::compute::gemm {

// Fallback for an own-kernel shape: (weights, input, out, m, n, k).
// input is the model-owned staging buffer (F16 or BF16 rows).
using HipBlasLtOwnHook = bool (*)(const void* weights, const void* input,
                                   float* out, int m, int n, int k,
                                   hipStream_t stream, std::string* error);

/// Dense 16-bit projections with F32 accumulation and output. Plans depend
/// only on tensor geometry and the pinned HIP library, never runtime timing.
class HipBlasLtRunner {
 public:
  ~HipBlasLtRunner();
  HipBlasLtRunner(const HipBlasLtRunner&) = delete;
  HipBlasLtRunner& operator=(const HipBlasLtRunner&) = delete;

  [[nodiscard]] static std::unique_ptr<HipBlasLtRunner> Create(
      hipStream_t stream, std::string* error_msg,
      HipBlasLtOwnHook own_f16 = nullptr,
      HipBlasLtOwnHook own_bf16 = nullptr);

  /// Prepare descriptors and load the solution library without launching a
  /// GEMM or reading model data. Gemm uses the same exact-shape plan.
  /// Own-kernel shapes resolve only when their hook is set.
  [[nodiscard]] bool Prepare(hipDataType type, int m, int n, int k,
                             std::string* error_msg);
  /// Row-major out[n][m] = input[n][k] * weights[m][k]^T.
  [[nodiscard]] bool Gemm(const void* weights, const void* input, float* out,
                          hipDataType type, int m, int n, int k,
                          std::string* error_msg);
  /// Gemm through the library kernel even where Gemm runs its own kernel
  /// (benchmarks compare the two).
  [[nodiscard]] bool LibraryGemm(const void* weights, const void* input,
                                 float* out, hipDataType type, int m, int n,
                                 int k, std::string* error_msg);
  /// Whether Gemm runs a model-owned kernel for this shape (hook required).
  [[nodiscard]] static bool UsesOwnKernel(hipDataType type, int m, int k);

 private:
  HipBlasLtRunner() = default;

  struct Plan;
  std::unique_ptr<Plan> MakePlan(hipDataType type, int m, int n, int k,
                                 std::string* error_msg) const;
  Plan* FindPlan(hipDataType type, int m, int n, int k, std::string* error_msg);
  bool Run(Plan& plan, const void* weights, const void* input, float* out,
           std::string* error_msg);

  hipblasLtHandle_t handle_{nullptr};
  hipStream_t stream_{nullptr};
  HipBlasLtOwnHook own_f16_{nullptr};
  HipBlasLtOwnHook own_bf16_{nullptr};
  std::map<std::array<int, 4>, std::unique_ptr<Plan>> plans_;
};

}  // namespace gufo::core::compute::gemm

#endif  // GUFO_CORE_COMPUTE_GEMM_HIPBLASLT_RUNNER_HPP_
