// Narrow Q8_0 decode GEMV core (extracted verbatim from flash-next MMQ).
//
// Origin: src/models/qwen38_flash_next/kernels/rocm/mmq/mmvq.hip.cpp,
// `mul_mat_vec_q8` (one wave per output row, up to eight prequantized inputs).
// The kernel body below is byte-identical to the source; only the surrounding
// model-specific scaffolding (GGML type enums, row-padding constants, gate
// fusion, HC/matrix specializations, dispatch switch) was replaced by the
// model-agnostic contract in this header comment and `DecideNarrowDecodeRoute`
// (src/core/compute/gemm/gemm_dispatcher.hpp).
//
// Contract (route engages only if ALL hold; otherwise the caller keeps its
// legacy path -- never force-fed):
//   * weights: row-major Q8_0 blocks, 34 bytes each (half scale + 32 int8),
//     K columns, K > 0, K % 32 == 0. No padding assumption on the weight side.
//   * activations: prequantized Q8_1 rows, 36 bytes per block (half2 scale
//     pair + 32 int8), same K blocking as the weights. Row stride is an
//     explicit parameter (`input_stride_blocks`, in blocks, >= K / 32); the
//     producer's padding policy (e.g. flash's 512-element row pad) is NOT a
//     core assumption.
//   * tokens N (rows in flight): 1..8, ungated. The fused gate/SwiGLU variant,
//     the HC 320x10240 geometry, the K==2560/K==320 matrix tiles and the
//     N > 8 batch routes stay model-local (different arithmetic or geometry).
//   * M (output rows): arbitrary, grid = M waves; one wave owns one row.
//   * output: float, token-major, out[t * M + row].
//   * single-row latency specialization excluded: N == 1 && M <= 64 &&
//     K % 5120 == 0 uses a different accumulation order in the model kernel,
//     so it is out of contract even though the shapes look covered.
// Numerics: dp4a integer dots, one float scale product per block, warp-shuffle
// reductions -- identical order to the source kernel.
//
// Reuse status: 27B and gemma4 decode GEMVs consume F32 activations (27B
// `LaunchQ8KBlockGEMV`, gemma4 `GemvBatched`), so they fail the prequantized-
// Q8_1 leg of this contract; their M/K shapes and N == 1 width would otherwise
// fit. Reuse is proven at the kernel level (standalone numeric check on 27B
// projection shapes); full-model integration needs a prequantized-activation
// producer first.
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/compute/gemm/gemm.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(__gfx900__) || defined(__gfx906__)
#define NARROW_Q8_GCN5
#endif

#if defined(__gfx803__)
#define NARROW_Q8_GCN4
#endif

#if defined(__gfx950__)
#define NARROW_Q8_CDNA4
#endif

#if defined(__gfx942__)
#define NARROW_Q8_CDNA3
#endif

#if defined(__gfx90a__)
#define NARROW_Q8_CDNA2
#endif

#if defined(__gfx908__)
#define NARROW_Q8_CDNA1
#endif

#if defined(NARROW_Q8_CDNA4) || defined(NARROW_Q8_CDNA3) || \
    defined(NARROW_Q8_CDNA2) || defined(NARROW_Q8_CDNA1)
#define NARROW_Q8_CDNA
#endif

#if defined(__GFX12__)
#define NARROW_Q8_RDNA4
#endif

#if defined(__GFX11__)
#define NARROW_Q8_RDNA3
#endif

#if defined(__gfx1150__) || defined(__gfx1151__)
#define NARROW_Q8_RDNA3_5
#endif

#if defined(NARROW_Q8_RDNA3) && !defined(NARROW_Q8_RDNA3_5)
#define NARROW_Q8_RDNA3_0
#endif

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
    defined(__gfx1036__) || defined(__gfx1037__)
#define NARROW_Q8_RDNA2
#endif

#if defined(__gfx1010__) || defined(__gfx1012__)
#define NARROW_Q8_RDNA1
#endif

#define NARROW_Q8_WARP_SIZE 32

namespace gufo::core::compute::gemm::narrow_q8 {

using std::min;

// ---- Quantized block layouts (GGML Q8_0 / Q8_1, 32 elements) ----
#define NARROW_Q8_QK8_0 32
#define NARROW_Q8_QK8_1 32
#define NARROW_Q8_QI8_0 (NARROW_Q8_QK8_0 / 4)
#define NARROW_Q8_VDR_Q8_0_Q8_1 2

struct NarrowQ80WeightBlock {
  __half d;                      // delta (scale)
  int8_t qs[NARROW_Q8_QK8_0];    // quants
};

struct NarrowQ81ActBlock {
  __half2 ds;                    // delta + sum, packed
  int8_t qs[NARROW_Q8_QK8_1];    // quants
};

// Aliases keep the moved kernel body (and its helpers) byte-identical to the
// source, which names these types block_q8_0 / block_q8_1.
using block_q8_0 = NarrowQ80WeightBlock;
using block_q8_1 = NarrowQ81ActBlock;
#define QK8_0 NARROW_Q8_QK8_0
#define QK8_1 NARROW_Q8_QK8_1
#define QI8_0 NARROW_Q8_QI8_0
#define VDR_Q8_0_Q8_1_MMVQ NARROW_Q8_VDR_Q8_0_Q8_1
#define WARP_SIZE NARROW_Q8_WARP_SIZE

static_assert(sizeof(block_q8_0) == sizeof(__half) + 32,
              "narrow-q8: Q8_0 weight block must be 34 bytes");
static_assert(sizeof(block_q8_1) == sizeof(__half2) + 32,
              "narrow-q8: Q8_1 activation block must be 36 bytes");
static_assert(NARROW_Q8_QK8_0 == 32 && NARROW_Q8_QK8_1 == 32,
              "narrow-q8: blocking factor is 32");
static_assert(NARROW_Q8_VDR_Q8_0_Q8_1 == 2, "narrow-q8: VDR is 2");

// ---- Verbatim device helpers (source: mmq vecdotq.hpp / common.hpp) ----

static __device__ __forceinline__ int get_int_b2(const void* x,
                                                 const int& i32) {
  const uint16_t* x16 =
      (const uint16_t*)x;  // assume at least 2 byte alignment

  int x32 = x16[2 * i32 + 0] << 0;
  x32 |= x16[2 * i32 + 1] << 16;

  return x32;
}

static __device__ __forceinline__ int get_int_b4(const void* x,
                                                 const int& i32) {
  return ((const int*)x)[i32];  // assume at least 4 byte alignment
}

// Lane i receives lane (i ^ offset)'s value, as __shfl_xor does for a full
// wave. On gfx1151 this is a DPP row_xmask (offsets below 16) or
// v_permlanex16 (16): a register move, not a round trip through the LDS
// crossbar (ds_bpermute). Only the transport differs, so sums stay bitwise.
template <int offset>
static __device__ __forceinline__ int xor_lane(int x) {
#if defined(__gfx1151__)
  if constexpr (offset < 16) {
    return __builtin_amdgcn_update_dpp(0, x, 0x160 | offset, 0xF, 0xF, false);
  } else if constexpr (offset == 16) {
    return __builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false,
                                       false);
  }
#endif
  return __shfl_xor(x, offset, WARP_SIZE);
}

template <int offset>
static __device__ __forceinline__ float xor_lane(float x) {
  return __builtin_bit_cast(float, xor_lane<offset>(__builtin_bit_cast(int, x)));
}

// The butterfly in the shuffle loop's order (width/2 down to 1).
template <int width, typename T>
static __device__ __forceinline__ T xor_reduce_sum(T x) {
  static_assert(width <= 32, "xor_lane covers offsets up to 16");
  if constexpr (width >= 32) x += xor_lane<16>(x);
  if constexpr (width >= 16) x += xor_lane<8>(x);
  if constexpr (width >= 8) x += xor_lane<4>(x);
  if constexpr (width >= 4) x += xor_lane<2>(x);
  if constexpr (width >= 2) x += xor_lane<1>(x);
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
  return xor_reduce_sum<width>(x);
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
  return xor_reduce_sum<width>(x);
}

// Reduce-scatter levels for n sums: the first levels of the wave butterfly
// each halve the sums a lane carries.
constexpr int ScatterLevels(int n) {
  return n <= 1 ? 0 : n <= 2 ? 1 : n <= 4 ? 2 : 3;
}

// One butterfly level that keeps half of `count` sums: the lane on the
// `offset` side keeps the upper half and sends the lower, its partner the
// reverse, so each kept sum adds the partner's copy of the same sum.
template <int offset, int count>
static __device__ __forceinline__ void scatter_level(float* s, int lane) {
  const bool upper = (lane & offset) != 0;
#pragma unroll
  for (int i = 0; i < count / 2; ++i) {
    const float keep = upper ? s[count / 2 + i] : s[i];
    const float send = upper ? s[i] : s[count / 2 + i];
    s[i] = keep + xor_lane<offset>(send);
  }
}

// warp_reduce_sum<32> over n sums at once: the same pairs (16, 8, 4, 2, 1)
// and operands as the full butterfly, which leaves every lane with identical
// bits, but lane l only finishes sum l >> (5 - ScatterLevels(n)).
template <int n>
static __device__ __forceinline__ float scatter_reduce_sum(const float (&v)[n],
                                                           int lane) {
  constexpr int levels = ScatterLevels(n);
  float s[1 << levels];
#pragma unroll
  for (int j = 0; j < (1 << levels); ++j) s[j] = j < n ? v[j] : 0.0f;
  if constexpr (levels == 3) {
    scatter_level<16, 8>(s, lane);
    scatter_level<8, 4>(s, lane);
    scatter_level<4, 2>(s, lane);
  } else if constexpr (levels == 2) {
    scatter_level<16, 4>(s, lane);
    scatter_level<8, 2>(s, lane);
    s[0] += xor_lane<4>(s[0]);
  } else if constexpr (levels == 1) {
    scatter_level<16, 2>(s, lane);
    s[0] += xor_lane<8>(s[0]);
    s[0] += xor_lane<4>(s[0]);
  } else {
    s[0] += xor_lane<16>(s[0]);
    s[0] += xor_lane<8>(s[0]);
    s[0] += xor_lane<4>(s[0]);
  }
  s[0] += xor_lane<2>(s[0]);
  s[0] += xor_lane<1>(s[0]);
  return s[0];
}

typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ int ggml_hip_dp4a(const int a, const int b,
                                                   int c) {
#if defined(NARROW_Q8_CDNA) || defined(NARROW_Q8_RDNA2) || \
    defined(__gfx906__)
  c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(NARROW_Q8_RDNA3) || defined(NARROW_Q8_RDNA4)
  c = __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(NARROW_Q8_RDNA1) || defined(__gfx900__)
  int tmp1;
  int tmp2;
  asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
      : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
      : "v"(a), "v"(b));
#else
  const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
  const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
  c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
  return c;
}

template <typename T, int vdr>
static __device__ __forceinline__ T vec_dot_q8_0_q8_1_impl(
    const int* v, const int* u, const T& d8_0, const T& d8_1) {
  int sumi = 0;

#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    // SIMD dot product of quantized values
    sumi = ggml_hip_dp4a(v[i], u[i], sumi);
  }

  return d8_0 * d8_1 * ((T)sumi);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
    const int& kbx, const int& iqs) {
  const block_q8_0* bq8_0 = (const block_q8_0*)vbq + kbx;

  int v[VDR_Q8_0_Q8_1_MMVQ];
  int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
  for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_b2(bq8_0->qs, iqs + i);
    u[i] = get_int_b4(bq8_1->qs, iqs + i);
  }

  return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(
      v, u, bq8_0->d, __low2half(bq8_1->ds));
}

__device__ __forceinline__ float ggml_hip_op_silu_single(float x) {
  return x / (1.0f + expf(-x));
}

// ---- Verbatim narrow decode kernel (source: mmq mmvq.hip.cpp) ----
//
// Each wave handles up to eight dense inputs for one weight row on gfx1151.
// `scatter` reduces the inputs' sums together and stores them from one lane
// each (same bits as reducing each sum on every lane and storing from lane 0).
// clang-format off: body below is byte-identical to the source
template<int ncols_dst, bool has_gate, int token_waves = 1, bool ragged = false,
         bool scatter = false>
__launch_bounds__(32 * token_waves, 1) static __global__
    void mul_mat_vec_q8(const void* __restrict__ weights,
                        const void* __restrict__ gate,
                        const block_q8_1* __restrict__ input,
                        float* __restrict__ output, const uint32_t ncols_x,
                        const uint32_t nrows_x, const uint32_t stride_col_y,
                        const uint32_t valid_tokens = ncols_dst) {
  constexpr int qi = QI8_0;
  constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
  constexpr int blocks_per_iter = vdr * 32 / qi;
  const int lane = token_waves > 1 ? threadIdx.x % 32 : threadIdx.x;
  const int first_token = token_waves > 1 ? threadIdx.x / 32 * ncols_dst : 0;
  const int row = blockIdx.x;
  const int blocks_per_row = ncols_x / QK8_0;
  const int row_offset = row * blocks_per_row;
  const int kqs = vdr * (lane % (qi / vdr));
  float sum[ncols_dst] = {};
  float gate_sum[ncols_dst] = {};
  for (int kbx = lane / (qi / vdr); kbx < blocks_per_row;
       kbx += blocks_per_iter) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      // The last wave may contain fewer than eight inputs. Its inactive
      // columns reuse a valid row; their results are never stored.
      const int token =
          ragged ? min(first_token + j, int(valid_tokens) - 1) : first_token + j;
      sum[j] += vec_dot_q8_0_q8_1(
          weights, input + token * stride_col_y + kbx,
          row_offset + kbx, kqs);
      if constexpr (has_gate) {
        gate_sum[j] += vec_dot_q8_0_q8_1(gate, input + j * stride_col_y + kbx,
                                         row_offset + kbx, kqs);
      }
    }
  }
  if constexpr (scatter) {
    static_assert(token_waves == 1 && !ragged);
    constexpr int shift = 5 - ScatterLevels(ncols_dst);
    // Both reductions run on the full wave; only the stores are per lane.
    float value = scatter_reduce_sum(sum, lane);
    float gate_value = 0.0f;
    if constexpr (has_gate)
      gate_value = scatter_reduce_sum(gate_sum, lane);
    const int token = lane >> shift;
    if ((lane & ((1 << shift) - 1)) == 0 && token < ncols_dst) {
      if constexpr (has_gate)
        value *= ggml_hip_op_silu_single(gate_value);
      output[token * nrows_x + row] = value;
    }
    return;
  }
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
    sum[j] = warp_reduce_sum<32>(sum[j]);
    if constexpr (has_gate)
      gate_sum[j] = warp_reduce_sum<32>(gate_sum[j]);
    if (lane == 0 && (token_waves == 1 || first_token + j < valid_tokens)) {
      float value = sum[j];
      if constexpr (has_gate)
        value *= ggml_hip_op_silu_single(gate_sum[j]);
      output[(first_token + j) * nrows_x + row] = value;
    }
  }
}
// clang-format on


}  // namespace gufo::core::compute::gemm::narrow_q8

// ---- Model-agnostic launcher (ungated narrow decode only) ----
//
// Mirrors the source dispatcher's ungated 1..8-token arms exactly: N == 1
// takes the plain kernel, N > 1 takes the scatter variant (same bits as the
// plain reduction, cheaper stores). The single-row latency specialization
// (M <= 64, K % 5120 == 0), the fused gate, HC/matrix geometries and N > 8
// batch tiles are out of contract -- the caller keeps its legacy path there.
namespace gufo::core::compute::gemm {

bool LaunchNarrowQ80Decode(const void* weights, const void* input,
                                  float* output, std::size_t k,
                                  std::size_t rows, std::size_t tokens,
                                  std::size_t input_stride_blocks,
                                  hipStream_t stream) {
  namespace nq = narrow_q8;
  if (weights == nullptr || input == nullptr || output == nullptr) return false;
  if (tokens < 1 || tokens > 8) return false;
  if (k == 0 || k % 32 != 0) return false;
  if (rows == 0) return false;
  if (input_stride_blocks < k / 32) return false;
  const auto* q8_input = static_cast<const nq::block_q8_1*>(input);
  const auto ncols = static_cast<std::uint32_t>(k);
  const auto nrows = static_cast<std::uint32_t>(rows);
  const auto stride = static_cast<std::uint32_t>(input_stride_blocks);
  const auto grid = static_cast<unsigned>(rows);
  switch (tokens) {
#define NARROW_Q8_LAUNCH(N)                                                   \
  case N:                                                                     \
    nq::mul_mat_vec_q8<N, false>                                              \
        <<<grid, 32, 0, stream>>>(weights, nullptr, q8_input, output, ncols,   \
                                  nrows, stride);                             \
    break
    NARROW_Q8_LAUNCH(1);
#undef NARROW_Q8_LAUNCH
#define NARROW_Q8_LAUNCH_SCATTER(N)                                           \
  case N:                                                                     \
    nq::mul_mat_vec_q8<N, false, 1, false, true>                              \
        <<<grid, 32, 0, stream>>>(weights, nullptr, q8_input, output, ncols,   \
                                  nrows, stride);                             \
    break
    NARROW_Q8_LAUNCH_SCATTER(2);
    NARROW_Q8_LAUNCH_SCATTER(3);
    NARROW_Q8_LAUNCH_SCATTER(4);
    NARROW_Q8_LAUNCH_SCATTER(5);
    NARROW_Q8_LAUNCH_SCATTER(6);
    NARROW_Q8_LAUNCH_SCATTER(7);
    NARROW_Q8_LAUNCH_SCATTER(8);
#undef NARROW_Q8_LAUNCH_SCATTER
    default:
      return false;
  }
  return true;
}

// ---- Fused gated narrow decode (shared-expert gate/up + SwiGLU) ----
//
// Single-token decode only: one PrequantizeQ8_1 pass over x, then the
// has_gate narrow core reads the up weights (weights) and gate weights
// (gate) and writes silu(gate*x)*(up*x) in the same element order as the
// gate-GEMV + up-GEMV + SwiGLU trio. Replaces five launches (two
// quantizes, two GEMVs, one SwiGLU) with two on the shared-expert path.
// Contract mirrors DecideNarrowDecodePrequantRoute plus tokens == 1;
// anything else returns false and the caller keeps the trio bit-identical.
bool LaunchNarrowQ80GatedPrequant(const void* up_w, const void* gate_w,
                                 const float* x_f32, float* output,
                                 std::size_t k, std::size_t rows, void* scratch,
                                 std::size_t scratch_bytes,
                                 hipStream_t stream) {
  namespace nq = narrow_q8;
  if (up_w == nullptr || gate_w == nullptr || x_f32 == nullptr ||
      output == nullptr)
    return false;
  if (k == 0 || k % 32 != 0 || rows == 0) return false;
  if (rows > static_cast<std::size_t>(0xFFFFFFFFu)) return false;
  if (scratch == nullptr || scratch_bytes < NarrowQ80PrequantBytes(1, k))
    return false;
  if (PrequantizeQ8_1(x_f32, scratch, 1, k, stream) != 0) return false;
  const auto* q8_input = static_cast<const nq::block_q8_1*>(scratch);
  const auto ncols = static_cast<std::uint32_t>(k);
  const auto nrows = static_cast<std::uint32_t>(rows);
  const auto stride =
      static_cast<std::uint32_t>(NarrowQ80PrequantStrideBlocks(k));
  const auto grid = static_cast<unsigned>(rows);
  nq::mul_mat_vec_q8<1, true>
      <<<grid, 32, 0, stream>>>(up_w, gate_w, q8_input, output, ncols, nrows,
                                stride);
  return true;
}

// ---- Fused QKV narrow decode (Q8_0 x prequantized-Q8_1, ungated) ----
namespace narrow_q8 {

template <int ncols_dst, bool scatter = false>
__launch_bounds__(32, 1) __global__
void mul_mat_vec_q8_qkv(const void* __restrict__ wq,
                        const void* __restrict__ wk,
                        const void* __restrict__ wv,
                        const block_q8_1* __restrict__ input,
                        float* __restrict__ out_q, float* __restrict__ out_k,
                        float* __restrict__ out_v, const uint32_t ncols_x,
                        const uint32_t qrows, const uint32_t kvrows,
                        const uint32_t stride_col_y) {
  constexpr int qi = QI8_0;
  constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
  constexpr int blocks_per_iter = vdr * 32 / qi;
  const int lane = threadIdx.x;
  const uint32_t row = blockIdx.x;
  const void* weights = wq;
  float* output = out_q;
  uint32_t rrow = row;
  uint32_t nrows_out = qrows;
  if (row >= qrows + kvrows) {
    weights = wv;
    output = out_v;
    rrow = row - qrows - kvrows;
    nrows_out = kvrows;
  } else if (row >= qrows) {
    weights = wk;
    output = out_k;
    rrow = row - qrows;
    nrows_out = kvrows;
  }
  const int blocks_per_row = static_cast<int>(ncols_x / QK8_0);
  const int row_offset = static_cast<int>(rrow) * blocks_per_row;
  const int kqs = vdr * (lane % (qi / vdr));
  float sum[ncols_dst] = {};
  for (int kbx = lane / (qi / vdr); kbx < blocks_per_row;
       kbx += blocks_per_iter) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      sum[j] += vec_dot_q8_0_q8_1(
          weights, input + static_cast<std::size_t>(j) * stride_col_y + kbx,
          row_offset + kbx, kqs);
    }
  }
  if constexpr (scatter) {
    constexpr int shift = 5 - ScatterLevels(ncols_dst);
    const float value = scatter_reduce_sum(sum, lane);
    const int token = lane >> shift;
    if ((lane & ((1 << shift) - 1)) == 0 && token < ncols_dst) {
      output[static_cast<std::size_t>(token) * nrows_out + rrow] = value;
    }
    return;
  }
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
    sum[j] = warp_reduce_sum<32>(sum[j]);
    if (lane == 0) {
      output[static_cast<std::size_t>(j) * nrows_out + rrow] = sum[j];
    }
  }
}

}  // namespace narrow_q8

bool LaunchNarrowQ80QkvDecode(const void* wq, const void* wk, const void* wv,
                             const void* input, float* out_q, float* out_k,
                             float* out_v, std::size_t k, std::size_t qrows,
                             std::size_t kvrows, std::size_t tokens,
                             std::size_t input_stride_blocks,
                             hipStream_t stream) {
  namespace nq = narrow_q8;
  if (wq == nullptr || wk == nullptr || wv == nullptr || input == nullptr ||
      out_q == nullptr || out_k == nullptr || out_v == nullptr)
    return false;
  if (tokens < 1 || tokens > 8) return false;
  if (k == 0 || k % 32 != 0) return false;
  if (qrows == 0 || kvrows == 0) return false;
  if (input_stride_blocks < k / 32) return false;
  const std::size_t total_rows = qrows + 2 * kvrows;
  if (total_rows == 0 ||
      total_rows > static_cast<std::size_t>(0xFFFFFFFFu))
    return false;
  if (tokens == 1 && total_rows <= 64 && k % 5120 == 0) return false;
  if (total_rows == 320 && k == 10240) return false;
  const auto* q8_input = static_cast<const nq::block_q8_1*>(input);
  const auto ncols = static_cast<std::uint32_t>(k);
  const auto nqrows = static_cast<std::uint32_t>(qrows);
  const auto nkvrows = static_cast<std::uint32_t>(kvrows);
  const auto stride = static_cast<std::uint32_t>(input_stride_blocks);
  const auto grid = static_cast<unsigned>(total_rows);
  switch (tokens) {
#define NARROW_Q8_QKV_LAUNCH(N)                                              \
  case N:                                                                    \
    nq::mul_mat_vec_q8_qkv<N, false>                                          \
        <<<grid, 32, 0, stream>>>(wq, wk, wv, q8_input, out_q, out_k, out_v,  \
                                  ncols, nqrows, nkvrows, stride);            \
    break
    NARROW_Q8_QKV_LAUNCH(1);
#undef NARROW_Q8_QKV_LAUNCH
#define NARROW_Q8_QKV_LAUNCH_SCATTER(N)                                      \
  case N:                                                                    \
    nq::mul_mat_vec_q8_qkv<N, true>                                           \
        <<<grid, 32, 0, stream>>>(wq, wk, wv, q8_input, out_q, out_k, out_v,  \
                                  ncols, nqrows, nkvrows, stride);            \
    break
    NARROW_Q8_QKV_LAUNCH_SCATTER(2);
    NARROW_Q8_QKV_LAUNCH_SCATTER(3);
    NARROW_Q8_QKV_LAUNCH_SCATTER(4);
    NARROW_Q8_QKV_LAUNCH_SCATTER(5);
    NARROW_Q8_QKV_LAUNCH_SCATTER(6);
    NARROW_Q8_QKV_LAUNCH_SCATTER(7);
    NARROW_Q8_QKV_LAUNCH_SCATTER(8);
#undef NARROW_Q8_QKV_LAUNCH_SCATTER
    default:
      return false;
  }
  return true;
}

// ---- Narrow decode with fused row-bias + residual add (Q8_0, ungated) ----
// Same per-row arithmetic as mul_mat_vec_q8; the store folds the epilogue
// that callers otherwise run as two extra launches (AddRowBias then
// ResidualAdd): inout[token*nrows+row] += value + bias[row]. The addition
// order matches the unfused sequence bit-for-bit ((dot + bias) + resid),
// so engaging this path never changes numerics. Same contract as
// LaunchNarrowQ80Decode; out of contract the launcher returns false and
// the caller keeps its three-launch sequence.
namespace narrow_q8 {

template <int ncols_dst, bool scatter = false>
__launch_bounds__(32, 1) __global__
void mul_mat_vec_q8_rowadd(const void* __restrict__ weights,
                           const block_q8_1* __restrict__ input,
                           const float* __restrict__ bias,
                           float* __restrict__ inout, const uint32_t ncols_x,
                           const uint32_t nrows_x,
                           const uint32_t stride_col_y) {
  constexpr int qi = QI8_0;
  constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
  constexpr int blocks_per_iter = vdr * 32 / qi;
  const int lane = threadIdx.x;
  const uint32_t row = blockIdx.x;
  const int blocks_per_row = static_cast<int>(ncols_x / QK8_0);
  const int row_offset = static_cast<int>(row) * blocks_per_row;
  const int kqs = vdr * (lane % (qi / vdr));
  float sum[ncols_dst] = {};
  for (int kbx = lane / (qi / vdr); kbx < blocks_per_row;
       kbx += blocks_per_iter) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      sum[j] += vec_dot_q8_0_q8_1(
          weights, input + static_cast<std::size_t>(j) * stride_col_y + kbx,
          row_offset + kbx, kqs);
    }
  }
  if constexpr (scatter) {
    constexpr int shift = 5 - ScatterLevels(ncols_dst);
    const float value = scatter_reduce_sum(sum, lane);
    const int token = lane >> shift;
    if ((lane & ((1 << shift) - 1)) == 0 && token < ncols_dst) {
      float* dst = inout + static_cast<std::size_t>(token) * nrows_x + row;
      *dst += value + bias[row];
    }
    return;
  }
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
    sum[j] = warp_reduce_sum<32>(sum[j]);
    if (lane == 0) {
      float* dst = inout + static_cast<std::size_t>(j) * nrows_x + row;
      *dst += sum[j] + bias[row];
    }
  }
}

}  // namespace narrow_q8

// Prequant helpers live in quantize_q8_1.hip (same static lib); forward
// declarations keep this TU free of the model-facing gemm.hpp. Already
// inside namespace gufo::core::compute::gemm here.
std::size_t PreqQ81RowPad(std::size_t k);
std::size_t PreqQ81Bytes(std::size_t n, std::size_t k);
int PrequantizeQ8_1(const float* x_f32, void* x_q8, std::size_t n,
                    std::size_t k, hipStream_t stream);
std::size_t NarrowQ80PrequantScratchBytes(std::size_t max_k);
std::size_t NarrowQ80PrequantBytes(std::size_t tokens, std::size_t k);
std::size_t NarrowQ80PrequantStrideBlocks(std::size_t k);

bool LaunchNarrowQ80PrequantRowAdd(const void* weights, const float* x_f32,
                                   const float* bias, float* inout,
                                   std::size_t k, std::size_t rows,
                                   std::size_t tokens, void* scratch,
                                   std::size_t scratch_bytes,
                                   hipStream_t stream) {
  namespace nq = narrow_q8;
  if (weights == nullptr || x_f32 == nullptr || bias == nullptr ||
      inout == nullptr)
    return false;
  if (tokens < 1 || tokens > 8 || k == 0 || k % 32 != 0 || rows == 0)
    return false;
  if (scratch == nullptr || scratch_bytes < NarrowQ80PrequantBytes(tokens, k))
    return false;
  if (tokens == 1 && rows <= 64 && k % 5120 == 0) return false;
  if (rows == 320 && k == 10240) return false;
  if (PrequantizeQ8_1(x_f32, scratch, tokens, k, stream) != 0) return false;
  const auto* q8_input = static_cast<const nq::block_q8_1*>(scratch);
  const auto ncols = static_cast<std::uint32_t>(k);
  const auto nrows = static_cast<std::uint32_t>(rows);
  const auto stride =
      static_cast<std::uint32_t>(NarrowQ80PrequantStrideBlocks(k));
  const auto grid = static_cast<unsigned>(rows);
  switch (tokens) {
#define NARROW_Q8_ROWADD_LAUNCH(N)                                           \
  case N:                                                                    \
    nq::mul_mat_vec_q8_rowadd<N, false>                                       \
        <<<grid, 32, 0, stream>>>(weights, q8_input, bias, inout, ncols,      \
                                  nrows, stride);                            \
    break
    NARROW_Q8_ROWADD_LAUNCH(1);
#undef NARROW_Q8_ROWADD_LAUNCH
#define NARROW_Q8_ROWADD_LAUNCH_SCATTER(N)                                   \
  case N:                                                                    \
    nq::mul_mat_vec_q8_rowadd<N, true>                                        \
        <<<grid, 32, 0, stream>>>(weights, q8_input, bias, inout, ncols,      \
                                  nrows, stride);                            \
    break
    NARROW_Q8_ROWADD_LAUNCH_SCATTER(2);
    NARROW_Q8_ROWADD_LAUNCH_SCATTER(3);
    NARROW_Q8_ROWADD_LAUNCH_SCATTER(4);
    NARROW_Q8_ROWADD_LAUNCH_SCATTER(5);
    NARROW_Q8_ROWADD_LAUNCH_SCATTER(6);
    NARROW_Q8_ROWADD_LAUNCH_SCATTER(7);
    NARROW_Q8_ROWADD_LAUNCH_SCATTER(8);
#undef NARROW_Q8_ROWADD_LAUNCH_SCATTER
    default:
      return false;
  }
  return true;
}

}  // namespace gufo::core::compute::gemm

#undef QK8_0
#undef QK8_1
#undef QI8_0
#undef VDR_Q8_0_Q8_1_MMVQ
#undef WARP_SIZE
