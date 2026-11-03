// Phase C-1 (GGUF v2 section 4): shared-core home for the quantized-GEMM
// launchers gemma4 previously reached through
// models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp. Every body below is
// a verbatim copy (namespace relocated to gufo::core::compute::gemm only);
// flash-next keeps its own originals untouched until Phase C-2 rewires it.
// Numeric behavior is identical by construction (same kernels, same launch).
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/compute/gemm/gemm.hpp"

namespace gufo::core::compute::gemm {
namespace {

/// Lane i receives lane (i ^ offset)'s value, as __shfl_xor(v, offset) does
/// across a wave32. On gfx1151 it is a DPP row_xmask (offsets below 16) or
/// v_permlanex16 (16): a register move instead of an LDS crossbar round trip
/// (ds_bpermute). Only the transport differs, so reductions stay bitwise;
/// every lane must be active, as the shuffle's callers already require.
__device__ __forceinline__ int XorLane(int v, int offset) {
#if defined(__gfx1151__)
  switch (offset) {
    case 1:
      return __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, false);
    case 2:
      return __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, false);
    case 4:
      return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, false);
    case 8:
      return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, false);
    case 16:
      return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false,
                                          false);
  }
#endif
  return __shfl_xor(v, offset);
}

__device__ __forceinline__ unsigned XorLane(unsigned v, int offset) {
  return static_cast<unsigned>(XorLane(static_cast<int>(v), offset));
}

__device__ __forceinline__ float XorLane(float v, int offset) {
  return __builtin_bit_cast(float, XorLane(__builtin_bit_cast(int, v), offset));
}


__device__ __forceinline__ ArgmaxCandidate BetterCandidate(ArgmaxCandidate a,
                                                           ArgmaxCandidate b) {
  return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}

__device__ ArgmaxCandidate ArgmaxBlock(ArgmaxCandidate best,
                                       ArgmaxCandidate* shared) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned wave = threadIdx.x >> 5u;
  for (int offset = 16; offset > 0; offset >>= 1) {
    best = BetterCandidate(
        best, {XorLane(best.value, offset), XorLane(best.index, offset)});
  }
  if (lane == 0) {
    shared[wave] = best;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (unsigned w = 1; w < kThreads / 32; ++w) {
      best = BetterCandidate(best, shared[w]);
    }
  }
  return best;
}

__global__ void ArgmaxPartialKernel(const float* logits,
                                    ArgmaxCandidate* partial,
                                    std::uint32_t vocab) {
  __shared__ ArgmaxCandidate shared[kThreads / 32];
  const std::uint32_t t = blockIdx.y;
  const float* src = logits + static_cast<std::size_t>(t) * vocab;
  ArgmaxCandidate best{-INFINITY, INT32_MAX};
  for (std::uint32_t i = blockIdx.x * kThreads + threadIdx.x; i < vocab;
       i += kArgmaxParts * kThreads) {
    best = BetterCandidate(best, {src[i], static_cast<std::int32_t>(i)});
  }
  best = ArgmaxBlock(best, shared);
  if (threadIdx.x == 0) {
    partial[t * kArgmaxParts + blockIdx.x] = best;
  }
}

__global__ void ArgmaxFinishKernel(const float* logits,
                                   const ArgmaxCandidate* partial,
                                   std::int32_t* out, std::uint32_t vocab) {
  __shared__ ArgmaxCandidate shared[kThreads / 32];
  const std::uint32_t t = blockIdx.x;
  ArgmaxCandidate best{-INFINITY, INT32_MAX};
  if (threadIdx.x < kArgmaxParts) {
    best = partial[t * kArgmaxParts + threadIdx.x];
  }
  best = ArgmaxBlock(best, shared);
  if (threadIdx.x == 0) {
    // std::max_element keeps element zero when it is NaN; later NaNs
    // never replace a finite candidate.
    out[t] =
        isnan(logits[static_cast<std::size_t>(t) * vocab]) ? 0 : best.index;
  }
}

}  // namespace

void Argmax(const float* logits, ArgmaxCandidate* scratch, std::int32_t* out,
            std::uint32_t n_tokens, std::uint32_t vocab, hipStream_t stream) {
  hipLaunchKernelGGL(ArgmaxPartialKernel, dim3(kArgmaxParts, n_tokens),
                     dim3(kThreads), 0, stream, logits, scratch, vocab);
  hipLaunchKernelGGL(ArgmaxFinishKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, logits, scratch, out, vocab);
}

}  // namespace gufo::core::compute::gemm

#endif  // ENGINE_ENABLE_HIP
