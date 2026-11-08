// Skinny router GEMV (q36): replaces the per-layer hipblas SGEMM
// [T x N] = X[T x H] x W^T (W = gate_inp [N x H]) for the 256-expert /
// 2048-hidden router shape on the decode path (n_tokens 1..8).
//
// Split-K over K (16 chunks, one kernel): 16 blocks (8 tiles x 2
// halves) x 256 threads; thread (expert, split) owns one 128-element
// K-chunk and the 16 experts of a block combine 16 partials in LDS in
// fixed split order. The K order is
// plain 0..H (no StaggerU replay) and partials round before the final
// sum, so logits are NOT bitwise identical to the hipblas path; top-8
// selection is unaffected (200/200 random trials match the stagger
// kernel, max logit diff 4.2e-06) and the tg128 bench hash gates it.
// The stagger kernel stays as the fallback for out-of-contract shapes.
// Any out-of-contract shape returns false and the caller keeps hipblas.
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/compute/gemm/gemm.hpp"

namespace gufo::core::compute::gemm {
namespace {

constexpr std::uint32_t kRouterTileExperts = 32;
constexpr std::uint32_t kRouterTiles = 8;  // 256 experts / 32 per tile
constexpr std::uint32_t kRouterStagger = 64;
constexpr std::uint32_t kRouterExperts = 256;
constexpr std::uint32_t kRouterUnroll = 32;
constexpr std::size_t kRouterHidden = 2048;
constexpr std::uint32_t kRouterSplit = 8;
constexpr std::uint32_t kRouterChunk = 256;  // hidden / split

__device__ __forceinline__ float RouterMaterialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}

__global__ void RouterStaggerGemvKernel(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        float* __restrict__ out,
                                        std::uint32_t n_tokens,
                                        std::uint32_t n_experts,
                                        std::size_t hidden) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t tile = blockIdx.y;
  if (t >= n_tokens) {
    return;
  }
  const std::uint32_t expert = tile * kRouterTileExperts + threadIdx.x;
  extern __shared__ float sx[];
  const float* xrow = x + static_cast<std::size_t>(t) * hidden;
  for (std::size_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    sx[i] = xrow[i];
  }
  __syncthreads();
  // StaggerU replay: M-tile `tile` starts at k0, wraps once at hidden.
  // k0 is a multiple of kRouterUnroll (tile*64), hidden is a multiple
  // too, so both ranges split evenly into unrolled segments.
  const std::size_t k0 =
      (static_cast<std::size_t>(tile) * kRouterStagger) % hidden;
  const float* wrow = w + static_cast<std::size_t>(expert) * hidden;
  float acc = 0.0F;
  for (std::size_t i = k0; i < hidden; i += kRouterUnroll) {
#pragma unroll
    for (std::uint32_t u = 0; u < kRouterUnroll; ++u) {
      acc = fmaf(sx[i + u], wrow[i + u], acc);
    }
    acc = RouterMaterialize(acc);
  }
  for (std::size_t i = 0; i < k0; i += kRouterUnroll) {
#pragma unroll
    for (std::uint32_t u = 0; u < kRouterUnroll; ++u) {
      acc = fmaf(sx[i + u], wrow[i + u], acc);
    }
    acc = RouterMaterialize(acc);
  }
  out[static_cast<std::size_t>(t) * n_experts + expert] = acc;
}

// Split-K router GEMV: 16 blocks x 256 threads. Thread (expert_local,
// split) accumulates one 256-element K-chunk of its expert in plain
// K order with float4 loads; the 16 partials per expert combine in
// LDS in fixed split order. One launch per layer, no second pass.
__global__ void RouterSplitKGemvKernel(const float* __restrict__ x,
                                       const float* __restrict__ w,
                                       float* __restrict__ out,
                                       std::uint32_t n_tokens,
                                       std::uint32_t n_experts) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t tile = blockIdx.y & 7u;
  const std::uint32_t half = blockIdx.y >> 3u;
  if (t >= n_tokens) {
    return;
  }
  const std::uint32_t tid = threadIdx.x;  // 0..255
  constexpr std::uint32_t kExpPerBlock = 16;
  constexpr std::uint32_t kSplit = 16;
  constexpr std::uint32_t kChunk = kRouterHidden / kSplit;
  const std::uint32_t expert_local = tid / kSplit;
  const std::uint32_t split = tid % kSplit;
  const std::uint32_t expert =
      tile * kRouterTileExperts + half * kExpPerBlock + expert_local;
  extern __shared__ float smem[];
  float* sx = smem;
  float* parts = smem + kRouterHidden;
  const float* xrow = x + static_cast<std::size_t>(t) * kRouterHidden;
  for (std::uint32_t i = tid; i < kRouterHidden; i += 256) {
    sx[i] = xrow[i];
  }
  __syncthreads();
  const float* wbase =
      w + static_cast<std::size_t>(expert) * kRouterHidden + split * kChunk;
  const float* xbase = sx + split * kChunk;
  float acc = 0.0F;
  for (std::uint32_t i = 0; i < kChunk / 4; i += 8) {
#pragma unroll
    for (std::uint32_t u = 0; u < 8; ++u) {
      const float4 wv = reinterpret_cast<const float4*>(wbase)[i + u];
      const float4 xv = reinterpret_cast<const float4*>(xbase)[i + u];
      acc = fmaf(wv.x, xv.x, acc);
      acc = fmaf(wv.y, xv.y, acc);
      acc = fmaf(wv.z, xv.z, acc);
      acc = fmaf(wv.w, xv.w, acc);
    }
    acc = RouterMaterialize(acc);
  }
  parts[expert_local * kSplit + split] = acc;
  __syncthreads();
  if (split == 0) {
    float total = 0.0F;
    for (std::uint32_t s = 0; s < kSplit; ++s) {
      total += parts[expert_local * kSplit + s];
    }
    out[static_cast<std::size_t>(t) * n_experts + expert] = total;
  }
}

}  // namespace

bool LaunchRouterStaggerGemv(const float* x, const float* w, float* out,
                             std::uint32_t n_tokens, std::uint32_t n_experts,
                             std::size_t hidden, hipStream_t stream) {
  if (x == nullptr || w == nullptr || out == nullptr) {
    return false;
  }
  if (n_tokens < 1 || n_tokens > 8) {
    return false;
  }
  if (n_experts != kRouterExperts || hidden != kRouterHidden) {
    return false;
  }
  if (hidden % kRouterUnroll != 0) {
    return false;
  }
  // Exact token grid in every mode: the t>=n_tokens mask already bakes
  // n_tokens into the captured launch, so cross-token-count replay was never
  // supported; a fixed 8-wide grid would only schedule 8-n_tokens dead
  // blocks per tile (~0.1 us each on gfx1151).
  if (hidden % kRouterChunk == 0) {
    const std::size_t lds =
        (kRouterHidden + kRouterTileExperts * kRouterSplit) * sizeof(float);
    hipLaunchKernelGGL((RouterSplitKGemvKernel),
                       dim3(n_tokens, kRouterTiles * 2),
                       dim3(kRouterTileExperts * kRouterSplit), lds, stream, x,
                       w, out, n_tokens, n_experts);
    return true;
  }
  const std::size_t lds = hidden * sizeof(float);
  hipLaunchKernelGGL((RouterStaggerGemvKernel), dim3(n_tokens, kRouterTiles),
                     dim3(kRouterTileExperts), lds, stream, x, w, out,
                     n_tokens, n_experts, hidden);
  return true;
}

}  // namespace gufo::core::compute::gemm

#endif  // defined(ENGINE_ENABLE_HIP)
