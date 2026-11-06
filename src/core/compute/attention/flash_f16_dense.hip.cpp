// Dense bidirectional F16-WMMA flash attention, d == 64 MHA (see the .hpp
// for the contract and the dispatcher note).
//
// Data flow mirrors the proven qwen dense WMMA kernel 1:1 (LDS-staged Q,
// K/V streaming, LoadFrag fragments, LDS-mediated S/P exchange, F32 online
// softmax, F16 P, F32 O accumulate), retiled: d=256->64 (K-steps 16->4,
// split 2+2 across the s_kh halves), GQA/causal/mask/gate/int8 removed,
// query rows 32/block, 32 keys/sweep step, one head/block, one O-tile/wave.
//
// gfx1151 notes: wave32 __shfl_xor reductions (no DPP asm needed for
// correctness); LDS strides keep every LoadFrag 16B-aligned (q/k stride
// 72 halves = 144 B, v/p stride 40 halves = 80 B); K/V staging is
// thread-coarsened x8 with float4-friendly row-major reads. No prefetch
// double-buffering in v1 (correctness first); the sweep is still
// scores-traffic-free, which is the whole point on GTT-backed memory.
#if defined(ENGINE_ENABLE_HIP)
#include "src/core/compute/attention/flash_f16_dense.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gufo::core::compute::flash_f16 {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kRows = 32;
constexpr std::uint32_t kKeys = 32;
constexpr std::uint32_t kHeadDim = 64;
constexpr std::uint32_t kRowBlocks = 2;
constexpr std::uint32_t kKeyBlocks = 2;
constexpr std::uint32_t kSTiles = 4;
constexpr std::uint32_t kKSteps = 4;
constexpr std::uint32_t kKStepsPerWave = 2;
constexpr std::uint32_t kSoftmaxLanes = 8;
constexpr std::uint32_t kPerLane = 4;
constexpr std::uint32_t kQStride = kHeadDim + 8;
constexpr std::uint32_t kKStride = kHeadDim + 8;
constexpr std::uint32_t kVtStride = kKeys + 8;
constexpr std::uint32_t kScoreStride = 20;
constexpr std::uint32_t kPStride = kKeys + 8;

__launch_bounds__(256, 2) __global__ void FlashF16DenseKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ out, std::uint32_t n,
    float scale) {
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t query_start = blockIdx.x * kRows;

  const float* qh =
      q + static_cast<std::size_t>(head) * n * kHeadDim;
  const float* kh =
      k + static_cast<std::size_t>(head) * n * kHeadDim;
  const float* vh =
      v + static_cast<std::size_t>(head) * n * kHeadDim;
  float* oh = out + static_cast<std::size_t>(head) * n * kHeadDim;

  __shared__ __half q_lds[kRows][kQStride];
  __shared__ __half k_lds[kKeys][kKStride];
  __shared__ __half v_lds[kHeadDim][kVtStride];
  __shared__ float s_lds[2][kSTiles][16][kScoreStride];
  __shared__ __half p_lds[kRows][kPStride];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  // Stage Q once (F32 -> F16, scale folded).
  for (std::uint32_t i = tid; i < kRows * kHeadDim; i += 256u) {
    const std::uint32_t r = i / kHeadDim;
    const std::uint32_t c = i % kHeadDim;
    const std::uint32_t qr = query_start + r;
    const float val =
        (qr < n) ? qh[static_cast<std::size_t>(qr) * kHeadDim + c] * scale
                 : 0.0F;
    q_lds[r][c] = __half(val);
  }
  __syncthreads();

  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
#pragma unroll
  for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
    const std::uint32_t d0 = (s_kh * kKStepsPerWave + ks) * 16u;
    q_frag[ks] = LoadFrag(&q_lds[s_rb * 16u + sub][d0]);
  }

  // One O tile per wave: (row-block, dim-tile) bijection over 8 waves.
  const std::uint32_t rb_o = (wave >> 2u) & 1u;
  const std::uint32_t dim_o = wave & 3u;
  v8f o_acc = {};
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t n_key_tiles = (n + kKeys - 1u) / kKeys;
  for (std::uint32_t kb = 0; kb < n_key_tiles; ++kb) {
    const std::uint32_t key_start = kb * kKeys;
    // Stage K tile.
    for (std::uint32_t i = tid; i < kKeys * kHeadDim; i += 256u) {
      const std::uint32_t r = i / kHeadDim;
      const std::uint32_t c = i % kHeadDim;
      const std::uint32_t kr = key_start + r;
      const float val =
          (kr < n) ? kh[static_cast<std::size_t>(kr) * kHeadDim + c] : 0.0F;
      k_lds[r][c] = __half(val);
    }
    // Stage V tile transposed (lane owns key tid%32, dims (tid/32)*8+i).
    {
      const std::uint32_t v_key = tid % kKeys;
      const std::uint32_t v_slice = (tid / kKeys) * 8u;
      const std::uint32_t kr = key_start + v_key;
#pragma unroll
      for (std::uint32_t i = 0; i < 8u; ++i) {
        const float val =
            (kr < n)
                ? vh[static_cast<std::size_t>(kr) * kHeadDim + v_slice + i]
                : 0.0F;
        v_lds[v_slice + i][v_key] = __half(val);
      }
    }
    __syncthreads();

    // S = Q K^T for this wave's tile.
    v8f s_acc = {};
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = (s_kh * kKStepsPerWave + ks) * 16u;
      const v16h k_frag = LoadFrag(&k_lds[(s_kb * 16u + sub)][d0]);
      s_acc = Wmma(q_frag[ks], k_frag, s_acc);
    }
#pragma unroll
    for (std::uint32_t i = 0; i < 8u; ++i) {
      s_lds[s_kh][s_tile][(2u * i) + half_id][sub] = s_acc[i];
    }
    __syncthreads();

    // Online softmax over the full 32-key tile (bidirectional: no mask).
    {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      const std::uint32_t seg = tid % kSoftmaxLanes;
      const std::uint32_t qr = query_start + rg;
      const bool row_live = qr < n;
      const std::uint32_t rb = rg / 16u;
      const std::uint32_t row = rg % 16u;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = seg * kPerLane + m;
        const std::uint32_t kc = key_start + col;
        const bool valid = row_live && kc < n;
        const std::uint32_t tile = (col / 16u) * kRowBlocks + rb;
        vals[m] =
            valid ? (s_lds[0][tile][row][col % 16u] +
                     s_lds[1][tile][row][col % 16u])
                  : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      }
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w =
            isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][seg * kPerLane + m] = __half(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += __shfl_xor(part_sum, off);
      }
      running_max = next_max;
      running_sum = running_sum * prior_scale + part_sum;
      if (seg == 0) row_scale[rg] = prior_scale;
    }
    __syncthreads();

    // Rescale O accumulator.
    {
      float sc[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8u; ++i) {
        sc[i] = row_scale[rb_o * 16u + 2u * i + half_id];
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8u; ++i) {
        o_acc[i] *= sc[i];
      }
    }

    // O += P V.
#pragma unroll
    for (std::uint32_t kbb = 0; kbb < kKeyBlocks; ++kbb) {
      const v16h p_frag = LoadFrag(&p_lds[(rb_o * 16u) + sub][kbb * 16u]);
      const v16h v_frag =
          LoadFrag(&v_lds[(dim_o * 16u) + sub][kbb * 16u]);
      o_acc = Wmma(p_frag, v_frag, o_acc);
    }
    // Separate this step's PV/V reads from the next step's K/V staging
    // writes (all cross-thread LDS traffic needs the barrier on both sides).
    __syncthreads();
  }

  if (tid % kSoftmaxLanes == 0) {
    row_sum[tid / kSoftmaxLanes] = running_sum;
  }
  __syncthreads();

  // Epilogue: normalize + write this wave's 16x16 O tile.
#pragma unroll
  for (std::uint32_t i = 0; i < 8u; ++i) {
    const std::uint32_t row = 2u * i + half_id;
    const std::uint32_t qr = query_start + rb_o * 16u + row;
    if (qr >= n) continue;
    const float denominator = row_sum[rb_o * 16u + row];
    const std::size_t off =
        static_cast<std::size_t>(qr) * kHeadDim + dim_o * 16u + sub;
    oh[off] = (denominator > 0.0F) ? (o_acc[i] / denominator) : 0.0F;
  }
}

}  // namespace

bool FlashF16DenseMha(const float* q, const float* k, const float* v,
                      float* out, std::uint32_t n, std::uint32_t heads,
                      std::uint32_t head_dim, float scale,
                      hipStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      n == 0 || heads == 0 || head_dim != kHeadDim) {
    return false;
  }
  const dim3 grid((n + kRows - 1u) / kRows, heads);
  hipLaunchKernelGGL(FlashF16DenseKernel, grid, dim3(256), 0, stream, q, k, v,
                     out, n, scale);
  return true;
}

}  // namespace gufo::core::compute::flash_f16

#endif  // defined(ENGINE_ENABLE_HIP)
