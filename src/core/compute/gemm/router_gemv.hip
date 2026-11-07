// Order-preserving skinny router GEMV (q36): replaces the per-layer hipblas
// SGEMM [T x N] = X[T x H] x W^T (W = gate_inp [N x H]) for the 256-expert /
// 2048-hidden router shape on the decode path (n_tokens 1..8).
//
// hipBLAS order forensics (gfx1151, rocprofv3 kernel
// Cijk_Alik_Bljk_SB_MT32x32x8_..._TT2_2_..._WG16_16_1, Tensile MAC_F32
// inner loop): the per-output accumulation is a sequential FMA chain, but
// StaggerU rotates each 32-row M-tile's K start by tile*64 elements with a
// single wrap: output (t, n) accumulates k = k0, k0+1, ..., H-1, 0, ...,
// k0-1 where k0 = ((n / 32) * 64) % H. Verified bitwise against
// hipblasSgemm(OP_T, OP_N, 256 x T x 2048) for T = 1..8 over multiple
// seeds (9216 outputs, zero mismatches); rows 0..31 are the k0 = 0 case.
//
// This kernel replays exactly that rotation with one FMA chain per output.
// A compiler barrier materializes the accumulator once per unrolled
// segment: the TU builds with -ffast-math, which would otherwise
// reassociate the chain into multiple accumulators. Within a segment the
// chain is strictly sequential, so the K order is unchanged; the unroll
// only lets independent loads overlap in the memory pipeline (a barrier
// per element exposes full HBM latency on every step and costs ~5x).
// Router logits — and hence greedy top-8 — are therefore bitwise
// identical to the hipblas path at a fraction of the launch cost. Any
// out-of-contract shape returns false and the caller keeps hipblas.
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
  // Capture/eager grid split (same idiom as LaunchDecodeMoeKQuant):
  // captured graphs keep the fixed 8-token x 8-tile grid with the
  // t>=n_tokens mask; eager launches use the exact token grid.
  // hipStreamIsCapturing is a host-side query, legal during capture.
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(stream, &capture_status) != hipSuccess) {
    return false;
  }
  const std::uint32_t grid_t =
      capture_status == hipStreamCaptureStatusNone ? n_tokens : 8;
  const std::size_t lds = hidden * sizeof(float);
  hipLaunchKernelGGL((RouterStaggerGemvKernel), dim3(grid_t, kRouterTiles),
                     dim3(kRouterTileExperts), lds, stream, x, w, out,
                     n_tokens, n_experts, hidden);
  return true;
}

}  // namespace gufo::core::compute::gemm

#endif  // defined(ENGINE_ENABLE_HIP)
