#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/blocked_q8.hpp"

namespace gufo::hip {

// Elementwise SiLU gate for the blocked-Q8 decode read: the shared
// direct-read kernel has no gate input, while the F16 decode kernels fuse
// out *= sigmoid(gate) into the output row. Same scalar arithmetic as the
// F16 scalar path, so the gating itself is bit-identical; only the
// dequantized attention values differ (last-bit level).
__global__ void SigmoidGateMultiplyKernel(float* __restrict__ out,
                                         const float* __restrict__ gate,
                                         std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] *= 1.0F / (1.0F + expf(-gate[i]));
  }
}

void LaunchSigmoidGateMultiply(float* out, const float* gate, std::size_t n,
                               hipStream_t stream) {
  if (out == nullptr || gate == nullptr || n == 0) {
    return;
  }
  constexpr std::uint32_t kBlock = 256;
  const std::uint64_t grid64 =
      (static_cast<std::uint64_t>(n) + kBlock - 1) / kBlock;
  // Decode rows are heads * head_dim (6144 for 27B); far below the grid limit.
  if (grid64 == 0 || grid64 > static_cast<std::uint64_t>(0xFFFFFFFFU)) {
    return;
  }
  hipLaunchKernelGGL(SigmoidGateMultiplyKernel,
                     dim3(static_cast<std::uint32_t>(grid64)), dim3(kBlock), 0,
                     stream, out, gate, n);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
