#include "src/models/qwen/modules/quant_gemm.hpp"

#include <cstdlib>

#include "src/models/qwen/forward.hpp"  // TensorGEMV

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/ops/gemm.hpp"
#endif

namespace gufo::models::qwen {

void QuantGemm(const CpuModuleContext&, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept {
  TensorGEMV(A, x, M, K, y);
}

#if defined(ENGINE_ENABLE_HIP)
void QuantGemm(const HipModuleContext& ctx, const QwenTensorRef& A,
               std::span<const float> x, std::size_t M, std::size_t K,
               std::span<float> y) noexcept {
  // Phase C-3: triage lives in ::gufo::hip::LaunchGEMV now (shared
  // dispatcher); this wrapper only validates its buffers.
  if (A.empty() || x.size() < K || y.size() < M || A.num_elements < M * K) {
    std::abort();
  }
  ::gufo::hip::LaunchGEMV(A.data, A.type, x.data(), y.data(), M, K,
                          static_cast<hipStream_t>(ctx.Stream()),
                          ctx.PreqScratch(), ctx.PreqBytes());
}
#endif

}  // namespace gufo::models::qwen
