// Lightweight decode MoE kernel (q36): K-quant (Q4_K/Q5_K, plus the
// load-converted Q8_0 down) x SwiGLU x single/small batch (n <= 8) x top-k
// sparse.
//
// Why this exists: the routed WMMA core (routed_moe.hip) is built for large
// batches. On single-token decode it still stages full 128-row x 48-token
// tiles through LDS, runs every K sweep per (expert, bucket-chunk) tile and
// repeats the whole tile per slot (two tiles per slot at BN = 48). Capture
// forbids host sync, so the grid stays fixed and every slot redundantly
// recomputes. This kernel instead assigns one thread block per live slot
// and streams each selected expert's gate/up/down rows exactly once with
// scalar K-quant dequant + F32 dot in CPU-reference element order. No
// compaction, no F16 narrowing, no gate/up intermediates: gate/up fuse
// through SwiGLU in registers/LDS and only the slot-ordered down vectors
// are written for the shared combine.
//
// Thread geometry (deliberately shuffle-free): each thread owns whole rows,
// so there is no cross-lane reduction anywhere in the hot loop. Finer
// warp-cooperative variants were tried and lost: per-row shuffle reduces
// cost more than the improved coalescing saves on this GPU, and staging
// the input in LDS with per-lane chunk offsets causes 32-way bank
// conflicts. The input row is broadcast from LDS (conflict-free); weights
// stream once per row.
//
// Grid policy (single kernel binary, decided by the launcher): captured
// graphs use the fixed 64-slot grid with a slot>=live mask (no host sync,
// no data-dependent launch); eager launches use the exact live-slot grid.
// Masked blocks return before touching LDS. The mask check stays in the
// kernel for both modes so a miscount can never corrupt memory.
//
// Numerics: dequant matches the CPU reference element values and
// accumulation order (see src/core/quant/ggml_dequant.cpp Q4Value/Q5Value),
// so deviations from the routed F16-WMMA path are rounding-only. Greedy
// token equality is verified in the q36 acceptance runs.
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/compute/gemm/gemm.hpp"

namespace gufo::core::compute::gemm {
namespace {

// Fixed grid for captured graphs: 8 tokens x top-8 slots. Eager launches
// use the exact live-slot count instead (same kernel, mask never fires).
constexpr std::uint32_t kSlotsCap = 64;
// Block width reuses ::gufo::core::compute::gemm::kThreads (256).

struct DkQ4KBlock {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(DkQ4KBlock) == 144, "block_q4_K must be 144 bytes");

struct DkQ5KBlock {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};
static_assert(sizeof(DkQ5KBlock) == 176, "block_q5_K must be 176 bytes");

struct DkQ80Block {
  std::uint16_t d;
  std::int8_t qs[32];
};
static_assert(sizeof(DkQ80Block) == 34, "block_q8_0 must be 34 bytes");

__device__ __forceinline__ float F16BitsToF32(std::uint16_t bits) {
  return __half2float(__builtin_bit_cast(__half, bits));
}

// Mirrors GetQ4ScaleMin in src/core/quant/ggml_dequant.cpp.
__device__ __forceinline__ void Q4KScaleMin(const std::uint8_t* scales, int j,
                                           std::uint8_t& sc,
                                           std::uint8_t& mn) {
  if (j < 4) {
    sc = scales[j] & 0x3FU;
    mn = scales[j + 4] & 0x3FU;
  } else {
    sc = static_cast<std::uint8_t>((scales[j + 4] & 0x0FU) |
                                   ((scales[j - 4] >> 6U) << 4U));
    mn = static_cast<std::uint8_t>(((scales[j + 4] >> 4U) & 0x0FU) |
                                   ((scales[j] >> 6U) << 4U));
  }
}

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0F / (1.0F + expf(-x));
}

template <WeightType kType>
__device__ __forceinline__ float QuantsDot(const std::uint8_t* row,
                                           const float* vec, std::size_t n);

// Q4_K: per 256-block, 8 sub-blocks of 32 elements. Sub-block j reads the
// pair row (j / 2) bytes, low nibbles for even j, high for odd. Scalar
// element loop in slot order (matches the CPU reference exactly).
template <>
__device__ __forceinline__ float QuantsDot<WeightType::kQ4_K>(
    const std::uint8_t* row, const float* vec, std::size_t n) {
  float acc = 0.0F;
  const std::size_t nb = n / 256;
  for (std::size_t b = 0; b < nb; ++b) {
    const auto* blk =
        reinterpret_cast<const DkQ4KBlock*>(row + b * sizeof(DkQ4KBlock));
    const float d = F16BitsToF32(blk->d);
    const float dmin = F16BitsToF32(blk->dmin);
    const float* xv = vec + b * 256;
    for (int j = 0; j < 8; ++j) {
      std::uint8_t sc = 0;
      std::uint8_t mn = 0;
      Q4KScaleMin(blk->scales, j, sc, mn);
      const float s = d * static_cast<float>(sc);
      const float m = dmin * static_cast<float>(mn);
      const std::uint8_t* qb = blk->qs + (j / 2) * 32;
      const bool hi = (j & 1) != 0;
      const float* xw = xv + j * 32;
      for (int l = 0; l < 32; ++l) {
        const std::uint8_t q = hi ? static_cast<std::uint8_t>(qb[l] >> 4U)
                                  : static_cast<std::uint8_t>(qb[l] & 0x0FU);
        acc += (s * static_cast<float>(q) - m) * xw[l];
      }
    }
  }
  return acc;
}

// Q5_K: Q4_K layout plus 32 high-bit bytes (bit s of qh[lane] is the fifth
// bit of that lane's element in sub-group s). Scalar element loop in slot
// order (matches the CPU reference exactly).
template <>
__device__ __forceinline__ float QuantsDot<WeightType::kQ5_K>(
    const std::uint8_t* row, const float* vec, std::size_t n) {
  float acc = 0.0F;
  const std::size_t nb = n / 256;
  for (std::size_t b = 0; b < nb; ++b) {
    const auto* blk =
        reinterpret_cast<const DkQ5KBlock*>(row + b * sizeof(DkQ5KBlock));
    const float d = F16BitsToF32(blk->d);
    const float dmin = F16BitsToF32(blk->dmin);
    const float* xv = vec + b * 256;
    for (int gg = 0; gg < 4; ++gg) {
      for (int half = 0; half < 2; ++half) {
        const int bit = 2 * gg + half;
        std::uint8_t sc = 0;
        std::uint8_t mn = 0;
        Q4KScaleMin(blk->scales, bit, sc, mn);
        const float s = d * static_cast<float>(sc);
        const float m = dmin * static_cast<float>(mn);
        const std::uint8_t* qb = blk->qs + gg * 32;
        const float* xw = xv + gg * 64 + half * 32;
        for (int lane = 0; lane < 32; ++lane) {
          const std::uint8_t q4 = half != 0
                                      ? static_cast<std::uint8_t>(qb[lane] >> 4U)
                                      : static_cast<std::uint8_t>(qb[lane] &
                                                                  0x0FU);
          const std::uint8_t q = static_cast<std::uint8_t>(
              q4 + ((((blk->qh[lane] >> bit) & 1U) != 0U) ? 16U : 0U));
          acc += (s * static_cast<float>(q) - m) * xw[lane];
        }
      }
    }
  }
  return acc;
}

// Q8_0: fp16 scale + 32 int8 codes; no offset term.
template <>
__device__ __forceinline__ float QuantsDot<WeightType::kQ8_0>(
    const std::uint8_t* row, const float* vec, std::size_t n) {
  float acc = 0.0F;
  const std::size_t nb = n / 32;
  for (std::size_t b = 0; b < nb; ++b) {
    const auto* blk =
        reinterpret_cast<const DkQ80Block*>(row + b * sizeof(DkQ80Block));
    const float d = F16BitsToF32(blk->d);
    const float* xw = vec + b * 32;
    for (int l = 0; l < 32; ++l) {
      acc += d * static_cast<float>(blk->qs[l]) * xw[l];
    }
  }
  return acc;
}

template <WeightType kType>
__device__ __forceinline__ std::size_t KQuantRowBytes(std::size_t n) {
  return kType == WeightType::kQ4_K   ? (n / 256) * sizeof(DkQ4KBlock)
         : kType == WeightType::kQ5_K ? (n / 256) * sizeof(DkQ5KBlock)
                                      : (n / 32) * sizeof(DkQ80Block);
}

// One block per slot: stream the selected expert's gate/up rows through
// SwiGLU into LDS, then the down rows into the slot-ordered F32 output.
// Masked blocks (slot >= live_slots, or a negative expert id) return before
// touching LDS, so the fixed capture grid is safe.
template <WeightType kGate, WeightType kDown>
__launch_bounds__(256) __global__ void DecodeMoeKQuantKernel(
    const void* __restrict__ gate_w, const void* __restrict__ up_w,
    const void* __restrict__ down_w, const float* __restrict__ x,
    const std::int32_t* __restrict__ ids, float* __restrict__ down_e,
    std::uint32_t n_tokens, std::uint32_t top_k, std::size_t hidden,
    std::size_t expert_ff) {
  const std::uint32_t slot = blockIdx.x;
  const std::uint32_t live_slots = n_tokens * top_k;
  if (slot >= live_slots) {
    return;
  }
  const std::int32_t expert = ids[slot];
  if (expert < 0) {
    return;
  }
  const std::uint32_t token = slot / top_k;

  extern __shared__ float smem[];
  float* sx = smem;               // [hidden] input row, broadcast read
  float* sact = smem + hidden;    // [expert_ff] SwiGLU activations
  const float* xrow = x + static_cast<std::size_t>(token) * hidden;
  for (std::size_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    sx[i] = xrow[i];
  }
  __syncthreads();

  const std::size_t gate_rb = KQuantRowBytes<kGate>(hidden);
  const auto* gate_e = static_cast<const std::uint8_t*>(gate_w) +
                       static_cast<std::size_t>(expert) * expert_ff * gate_rb;
  const auto* up_e = static_cast<const std::uint8_t*>(up_w) +
                     static_cast<std::size_t>(expert) * expert_ff * gate_rb;
  for (std::size_t r = threadIdx.x; r < expert_ff; r += blockDim.x) {
    const float g = QuantsDot<kGate>(gate_e + r * gate_rb, sx, hidden);
    const float u = QuantsDot<kGate>(up_e + r * gate_rb, sx, hidden);
    sact[r] = (g * SigmoidF(g)) * u;
  }
  __syncthreads();

  const std::size_t down_rb = KQuantRowBytes<kDown>(expert_ff);
  const auto* down_base = static_cast<const std::uint8_t*>(down_w) +
                          static_cast<std::size_t>(expert) * hidden * down_rb;
  float* orow = down_e + static_cast<std::size_t>(slot) * hidden;
  for (std::size_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    orow[h] = QuantsDot<kDown>(down_base + h * down_rb, sact, expert_ff);
  }
}

}  // namespace

bool LaunchDecodeMoeKQuant(const void* gate_w, WeightType gate_type,
                           const void* up_w, const void* down_w,
                           WeightType down_type, const float* x,
                           const std::int32_t* ids, float* down_e,
                           std::uint32_t n_tokens, std::uint32_t top_k,
                           std::size_t hidden, std::size_t expert_ff,
                           hipStream_t stream) {
  if (gate_w == nullptr || up_w == nullptr || down_w == nullptr ||
      x == nullptr || ids == nullptr || down_e == nullptr) {
    return false;
  }
  if (n_tokens < 1 || n_tokens > 8 || top_k == 0 || top_k > 8) {
    return false;
  }
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * top_k;
  if (slots > kSlotsCap) {
    return false;
  }
  if (hidden == 0 || hidden % 256 != 0 || expert_ff == 0 ||
      expert_ff % 256 != 0) {
    return false;
  }
  if (hidden + expert_ff > 16384) {
    return false;
  }
  const bool gate_ok = gate_type == WeightType::kQ4_K ||
                       gate_type == WeightType::kQ5_K;
  const bool down_ok = down_type == WeightType::kQ5_K ||
                       down_type == WeightType::kQ8_0;
  if (!gate_ok || !down_ok) {
    return false;
  }
  // Exact live-slot grid in every mode: the slot>=live mask already bakes
  // the live count into the captured launch, so cross-batch replay was never
  // supported; a fixed 64-slot grid would only schedule 64-live dead blocks
  // (~0.1 us each on gfx1151).
  const std::size_t lds = (hidden + expert_ff) * sizeof(float);
  const bool is_q4k = gate_type == WeightType::kQ4_K;
  const bool down_q80 = down_type == WeightType::kQ8_0;
  if (is_q4k && !down_q80) {
    hipLaunchKernelGGL(
        (DecodeMoeKQuantKernel<WeightType::kQ4_K, WeightType::kQ5_K>),
        dim3(static_cast<std::uint32_t>(slots)), dim3(kThreads), lds, stream, gate_w, up_w, down_w, x,
        ids, down_e, n_tokens, top_k, hidden, expert_ff);
  } else if (is_q4k && down_q80) {
    hipLaunchKernelGGL(
        (DecodeMoeKQuantKernel<WeightType::kQ4_K, WeightType::kQ8_0>),
        dim3(static_cast<std::uint32_t>(slots)), dim3(kThreads), lds, stream, gate_w, up_w, down_w, x,
        ids, down_e, n_tokens, top_k, hidden, expert_ff);
  } else if (!is_q4k && !down_q80) {
    hipLaunchKernelGGL(
        (DecodeMoeKQuantKernel<WeightType::kQ5_K, WeightType::kQ5_K>),
        dim3(static_cast<std::uint32_t>(slots)), dim3(kThreads), lds, stream, gate_w, up_w, down_w, x,
        ids, down_e, n_tokens, top_k, hidden, expert_ff);
  } else {
    hipLaunchKernelGGL(
        (DecodeMoeKQuantKernel<WeightType::kQ5_K, WeightType::kQ8_0>),
        dim3(static_cast<std::uint32_t>(slots)), dim3(kThreads), lds, stream, gate_w, up_w, down_w, x,
        ids, down_e, n_tokens, top_k, hidden, expert_ff);
  }
  return true;
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

}  // namespace gufo::core::compute::gemm

#endif  // defined(ENGINE_ENABLE_HIP)
