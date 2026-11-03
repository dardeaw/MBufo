// Device twin of src/core/pooling_norm.hpp: pooled-row gather plus L2
// normalization for the batched embedding path. Leg 1 serves embeddings
// through the host helpers; these kernels are compile-covered here so the
// future batch lane reuses the same contract without forking the math.
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cstddef>

#include "src/core/pooling_norm.hpp"

namespace gufo::core::embedding {
namespace {

// One thread per output dim, grid-strided; `strategy` is the numeric value
// of PoolingStrategy (0 = last, 1 = mean, 2 = CLS).
__global__ void pool_kernel(const float* hidden, std::size_t token_count,
                            std::size_t dim, int strategy, float* out) {
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t d = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       d < dim; d += step) {
    float value = 0.0f;
    if (strategy == 2) {
      value = hidden[d];
    } else if (strategy == 0) {
      value = hidden[(token_count - 1) * dim + d];
    } else {
      double acc = 0.0;
      for (std::size_t t = 0; t < token_count; ++t)
        acc += static_cast<double>(hidden[t * dim + d]);
      value = static_cast<float>(acc / static_cast<double>(token_count));
    }
    out[d] = value;
  }
}

// Single-block Euclidean norm plus in-place normalization. `norm_out` is
// optional host-or-device storage for the pre-norm.
__global__ void norm_kernel(float* vec, std::size_t dim, double* norm_out) {
  __shared__ double partial[256];
  double acc = 0.0;
  for (std::size_t d = threadIdx.x; d < dim;
       d += static_cast<std::size_t>(blockDim.x)) {
    const double x = static_cast<double>(vec[d]);
    acc += x * x;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const double norm = sqrt(partial[0]);
  if (threadIdx.x == 0 && norm_out != nullptr) *norm_out = norm;
  __syncthreads();
  if (norm > 0.0) {
    const float inv = static_cast<float>(1.0 / norm);
    for (std::size_t d = threadIdx.x; d < dim;
         d += static_cast<std::size_t>(blockDim.x))
      vec[d] *= inv;
  }
}

}  // namespace

void PoolingReduceDevice(const float* hidden, std::size_t token_count,
                         std::size_t dim, PoolingStrategy strategy, float* out,
                         void* stream) {
  if (hidden == nullptr || out == nullptr || token_count == 0 || dim == 0)
    return;
  int code = 0;
  switch (strategy) {
    case PoolingStrategy::kLastToken:
      code = 0;
      break;
    case PoolingStrategy::kMean:
      code = 1;
      break;
    case PoolingStrategy::kCls:
      code = 2;
      break;
  }
  const unsigned blocks =
      static_cast<unsigned>((dim + 255) / 256 > 64 ? 64 : (dim + 255) / 256);
  hipStream_t queue = static_cast<hipStream_t>(stream);
  hipLaunchKernelGGL(pool_kernel, dim3(blocks), dim3(256), 0, queue, hidden,
                     token_count, dim, code, out);
}

void L2NormalizeDevice(float* vec, std::size_t dim, double* norm_out,
                       void* stream) {
  if (vec == nullptr || dim == 0) return;
  hipStream_t queue = static_cast<hipStream_t>(stream);
  hipLaunchKernelGGL(norm_kernel, dim3(1), dim3(256), 0, queue, vec, dim,
                     norm_out);
}

}  // namespace gufo::core::embedding

#endif  // defined(ENGINE_ENABLE_HIP)
