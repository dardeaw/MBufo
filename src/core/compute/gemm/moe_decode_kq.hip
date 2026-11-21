// Decode MoE (q36 / qwen35moe): packed-weight x SwiGLU x single/small batch (n <= 8) x top-k sparse.
//
// Why this exists: the routed WMMA core (routed_moe.hip) is built for large
// batches. On single-token decode it still stages full 128-row x 48-token
// tiles through LDS, runs every K sweep per (expert, bucket-chunk) tile and
// repeats the whole tile per slot. Capture forbids host sync, so the grid
// stays fixed. This path instead streams each selected expert's gate/up/down
// rows exactly once with scalar dequant + F32 dot in CPU-reference element
// order. No compaction, no F16 narrowing.
//
// Two launches, many blocks. The first version used one 256-thread block per
// live slot for everything (8 blocks for a top-8 token: ~8 of 40 CUs busy,
// ~100 GB/s of ~226). Now:
//   1. gate/up: grid (expert_ff / kRows, slots); each thread owns whole rows,
//      so the per-row reduction order is exactly the old kernel's, and the
//      SwiGLU activations go to a small F32 scratch [slots x expert_ff];
//   2. down: grid (hidden / kRows, slots) reads that slot's activations
//      through LDS and writes the slot-ordered down vector.
// Per-row results are bit-identical to the single-kernel version because the
// split only changes which block computes a row, never the order inside it.
//
// Format interface: the kernels are templates over `RowDot<F>` (see
// src/core/quant/row_dot_traits.hpp); the launcher dispatches the gate/up
// and down type codes through `ThreadRowFormats` independently, so a new
// format is one trait specialization, not a new kernel or a new launch
// combination, and an unlisted type is refused instead of decoded wrong.
//
// Thread geometry stays shuffle-free (per-row cross-lane reduces cost more
// than the coalescing gain on this GPU); the input vector is broadcast from
// LDS (conflict-free) and weights stream once per row.
//
// Masked blocks (slot >= live slots, negative expert id) return before
// touching LDS, so a fixed capture grid is safe; the mask check stays in the
// kernels for every launch mode.
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/quant/row_dot_traits.hpp"

namespace mbufo::core::compute::gemm {
namespace {

namespace quant = ::mbufo::core::quant;

// Slots cap: 8 tokens x top-8 slots.
constexpr std::uint32_t kSlotsCap = 64;

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0F / (1.0F + expf(-x));
}

// Gate/up rows of one (slot, row tile) through SwiGLU into the activation scratch.
template <class G, std::uint32_t kRows>
__launch_bounds__(kRows) __global__ void MoeDecodeGateUpKernel(
    const void* __restrict__ gate_w, const void* __restrict__ up_w, const float* __restrict__ x,
    const std::int32_t* __restrict__ ids, float* __restrict__ act, std::uint32_t n_tokens,
    std::uint32_t top_k, std::size_t hidden, std::size_t expert_ff) {
  const std::uint32_t slot = blockIdx.y;
  if (slot >= n_tokens * top_k) return;
  const std::int32_t expert = ids[slot];
  if (expert < 0) return;
  const std::uint32_t token = slot / top_k;

  extern __shared__ float smem[];  // [hidden] input row, broadcast read
  const float* xrow = x + static_cast<std::size_t>(token) * hidden;
  for (std::size_t i = threadIdx.x; i < hidden; i += blockDim.x) smem[i] = xrow[i];
  __syncthreads();

  const std::size_t r = static_cast<std::size_t>(blockIdx.x) * kRows + threadIdx.x;
  if (r >= expert_ff) return;
  const std::size_t rb = G::RowBytes(hidden);
  const std::size_t base = static_cast<std::size_t>(expert) * expert_ff * rb + r * rb;
  const float g = G::DotRow(static_cast<const std::uint8_t*>(gate_w) + base, smem, hidden);
  const float u = G::DotRow(static_cast<const std::uint8_t*>(up_w) + base, smem, hidden);
  act[static_cast<std::size_t>(slot) * expert_ff + r] = (g * SigmoidF(g)) * u;
}

// Down rows of one (slot, row tile): slot-ordered F32 output for the shared combine.
template <class D, std::uint32_t kRows>
__launch_bounds__(kRows) __global__ void MoeDecodeDownKernel(
    const void* __restrict__ down_w, const float* __restrict__ act, const std::int32_t* __restrict__ ids,
    float* __restrict__ down_e, std::uint32_t n_tokens, std::uint32_t top_k, std::size_t hidden,
    std::size_t expert_ff) {
  const std::uint32_t slot = blockIdx.y;
  if (slot >= n_tokens * top_k) return;
  const std::int32_t expert = ids[slot];
  if (expert < 0) return;

  extern __shared__ float smem[];  // [expert_ff] this slot's activations
  const float* arow = act + static_cast<std::size_t>(slot) * expert_ff;
  for (std::size_t i = threadIdx.x; i < expert_ff; i += blockDim.x) smem[i] = arow[i];
  __syncthreads();

  const std::size_t h = static_cast<std::size_t>(blockIdx.x) * kRows + threadIdx.x;
  if (h >= hidden) return;
  const std::size_t rb = D::RowBytes(expert_ff);
  const auto* row = static_cast<const std::uint8_t*>(down_w) + static_cast<std::size_t>(expert) * hidden * rb + h * rb;
  down_e[static_cast<std::size_t>(slot) * hidden + h] = D::DotRow(row, smem, expert_ff);
}

template <class G, class D, std::uint32_t kRows>
void LaunchPair(const void* gate_w, const void* up_w, const void* down_w, const float* x, const std::int32_t* ids,
                float* act_scratch, float* down_e, std::uint32_t n_tokens, std::uint32_t top_k, std::size_t hidden,
                std::size_t expert_ff, hipStream_t stream) {
  const std::uint32_t slots = n_tokens * top_k;
  const dim3 gate_grid(static_cast<std::uint32_t>((expert_ff + kRows - 1) / kRows), slots);
  const dim3 down_grid(static_cast<std::uint32_t>((hidden + kRows - 1) / kRows), slots);
  hipLaunchKernelGGL((MoeDecodeGateUpKernel<G, kRows>), gate_grid, dim3(kRows), hidden * sizeof(float), stream, gate_w,
                     up_w, x, ids, act_scratch, n_tokens, top_k, hidden, expert_ff);
  hipLaunchKernelGGL((MoeDecodeDownKernel<D, kRows>), down_grid, dim3(kRows), expert_ff * sizeof(float), stream,
                     down_w, act_scratch, ids, down_e, n_tokens, top_k, hidden, expert_ff);
}

// Rows per block for a launch. Measured on gfx1151 at qwen35moe shapes (hidden 2048, expert ff 512, top-8,
// 1..8 tokens, rotating expert ids): 256 is best or within a few percent of best for every format pair and
// slot count once weights are read with 16-byte loads, so there is no slot-count dependence to encode.
std::uint32_t AutoRows(std::uint32_t /*slots*/) { return 256u; }

}  // namespace

bool LaunchDecodeMoeKQuant(const void* gate_w, WeightType gate_type, const void* up_w, const void* down_w,
                           WeightType down_type, const float* x, const std::int32_t* ids, float* act_scratch,
                           float* down_e, std::uint32_t n_tokens, std::uint32_t top_k, std::size_t hidden,
                           std::size_t expert_ff, hipStream_t stream, std::uint32_t rows_hint) {
  if (gate_w == nullptr || up_w == nullptr || down_w == nullptr || x == nullptr || ids == nullptr ||
      act_scratch == nullptr || down_e == nullptr) {
    return false;
  }
  if (n_tokens < 1 || n_tokens > 8 || top_k == 0 || top_k > 16) return false;
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * top_k;
  if (slots > kSlotsCap) return false;
  if (hidden == 0 || hidden % 256 != 0 || expert_ff == 0 || expert_ff % 256 != 0) return false;
  if (hidden + expert_ff > 16384) return false;

  const std::uint32_t rows = rows_hint != 0 ? rows_hint : AutoRows(static_cast<std::uint32_t>(slots));
  if (rows != 32 && rows != 64 && rows != 128 && rows != 256) return false;
  // Both type codes must be listed and fit before anything is launched; the pair is then dispatched
  // through the trait list (gate/up and down independently, so the combinations are not enumerated here).
  bool launched = false;
  (void)quant::ThreadRowFormats::Dispatch(static_cast<int>(gate_type), [&](auto gm) {
    using G = decltype(gm);
    if (!G::ColsOk(hidden)) return;
    (void)quant::ThreadRowFormats::Dispatch(static_cast<int>(down_type), [&](auto dm) {
      using D = decltype(dm);
      if (!D::ColsOk(expert_ff)) return;
      const auto go = [&](auto rows_c) {
        LaunchPair<G, D, decltype(rows_c)::value>(gate_w, up_w, down_w, x, ids, act_scratch, down_e, n_tokens, top_k,
                                                  hidden, expert_ff, stream);
      };
      switch (rows) {
        case 32: go(std::integral_constant<std::uint32_t, 32>{}); break;
        case 64: go(std::integral_constant<std::uint32_t, 64>{}); break;
        case 128: go(std::integral_constant<std::uint32_t, 128>{}); break;
        default: go(std::integral_constant<std::uint32_t, 256>{}); break;
      }
      launched = true;
    });
  });
  return launched;
}

// Fused shared-gate + combine epilogue for the 35B decode MoE path (this
// file only; the shared fn_compat::MoeEpilogue is untouched). Replaces the
// per-layer hipblas glogit SGEMM (a 1 x hidden dot that costs a full small-
// SGEMM launch) plus the shared epilogue with one launch: the block dots
// the shared gate vector against the token row, then every thread combines
// one output element as out = sum_s w_s * down_e + sigmoid(glogit) *
// shared. The combine mirrors fn_compat::MoeEpilogueKernel exactly
// (k <= 16 branch, same Materialize barriers); only the gate source differs
// (parallel F32 dot instead of hipblas SGEMM: rounding-level only).
__device__ __forceinline__ float EpilogueMaterialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}

__launch_bounds__(256) __global__ void DecodeMoeGlogitEpilogueKernel(
    const float* __restrict__ expert_out, const float* __restrict__ weights,
    const float* __restrict__ shared_out, const float* __restrict__ shexp_vec,
    const float* __restrict__ x, float* __restrict__ out, std::uint32_t top_k,
    std::size_t hidden) {
  const std::uint32_t t = blockIdx.x;
  const std::size_t i = static_cast<std::size_t>(blockIdx.y) * blockDim.x +
                        threadIdx.x;
  // Shared-gate dot, parallel across the block (the single-thread form is
  // latency-bound; this is a few microseconds). Partial dots in registers,
  // one block-wide reduction through LDS.
  __shared__ float s_part[kThreads];
  __shared__ float s_gw;
  {
    const float* xrow = x + static_cast<std::size_t>(t) * hidden;
    float partial = 0.0F;
    for (std::size_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      partial += shexp_vec[h] * xrow[h];
    }
    s_part[threadIdx.x] = partial;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float glogit = 0.0F;
    for (std::uint32_t u = 0; u < kThreads; ++u) {
      glogit += s_part[u];
    }
    s_gw = 1.0F / (1.0F + expf(-glogit));
  }
  __syncthreads();
  if (i >= hidden) {
    return;
  }
  // Matches fn_compat::MoeEpilogueKernel (k <= 16 branch) with gate_stride 1.
  constexpr std::uint32_t kMaxSlots = 16;
  const float* rows =
      expert_out + static_cast<std::size_t>(t) * top_k * hidden + i;
  float w[kMaxSlots];
  float v[kMaxSlots];
#pragma unroll
  for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
    w[s] = s < top_k ? weights[t * top_k + s] : 0.0F;
    v[s] = s < top_k ? rows[static_cast<std::size_t>(s) * hidden] : 0.0F;
  }
  const std::size_t idx = static_cast<std::size_t>(t) * hidden + i;
  const float sh = shared_out[idx];
  float acc = EpilogueMaterialize(0.0F);
#pragma unroll
  for (std::uint32_t s = 0; s < kMaxSlots; ++s) {
    if (s < top_k) {
      acc = EpilogueMaterialize(acc + w[s] * v[s]);
    }
  }
  out[idx] = acc + s_gw * sh;
}

bool LaunchDecodeMoeEpilogue(const float* expert_out, const float* weights,
                             const float* shared_out,
                             const float* shexp_vec, const float* x,
                             float* out, std::uint32_t n_tokens,
                             std::uint32_t top_k, std::size_t hidden,
                             hipStream_t stream) {
  if (expert_out == nullptr || weights == nullptr || shared_out == nullptr ||
      shexp_vec == nullptr || x == nullptr || out == nullptr) {
    return false;
  }
  if (n_tokens < 1 || n_tokens > 8 || top_k == 0 || top_k > 16) {
    return false;
  }
  if (hidden == 0 || hidden % 256 != 0) {
    return false;
  }
  hipLaunchKernelGGL(DecodeMoeGlogitEpilogueKernel,
                     dim3(n_tokens, Blocks(hidden)), dim3(kThreads), 0,
                     stream, expert_out, weights, shared_out, shexp_vec, x,
                     out, top_k, hidden);
  return true;
}

}  // namespace mbufo::core::compute::gemm

#endif  // defined(ENGINE_ENABLE_HIP)
