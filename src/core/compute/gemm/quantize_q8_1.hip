// Shared Q8_1 prequantize core (verbatim from flash-next MMQ).
//
// Origin: src/models/qwen38_flash_next/kernels/rocm/mmq/quantize.hip.cpp,
// `quantize_q8_1` (+ `quantize_row_q8_1_hip` launcher) behind
// `qfn_mmq_quantize_q8_1` (qfn_mmq.h / qfn_mmq.hip.cpp).
// The kernel body below is byte-identical to the source; only the surrounding
// model-specific scaffolding (GGML type enums, ggml-common/fastdiv helpers,
// MATRIX_ROW_PADDING import, qfn_mmq namespace, by-expert/MMQ interleaved
// variants) was replaced by the model-agnostic contract in this file and
// `DecideNarrowDecodePrequantRoute`
// (src/core/compute/gemm/gemm_dispatcher.hpp).
//
// Contract (row-major F32 in -> canonical block_q8_1 out, same as flash):
//   * K > 0, K % 32 == 0; N >= 1.
//   * out row width = pad(K, 512) elements (= pad(K,512)/32 blocks, 36B each),
//     zero-filled tail when K is not a multiple of 512. This matches
//     qfn_mmq_q8_1_bytes(N, K) and the narrow-Q8 core's stride_blocks leg
//     (>= K/32); the producer's 512-element pad is NOT a core assumption
//     beyond the byte-size function.
//   * one thread per element (256-thread blocks, grid = (padK+255)/256 x N),
//     amax via warp_reduce_max<32>, d = amax/127, q = round(x/d),
//     sum = d*sum(q) stored as half2(ds) -- identical order to source.
// Numerics: dp4a-free quantize; bitwise identical to qfn_mmq_quantize_q8_1
// for contiguous [N, K] inputs.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace gufo::core::compute::gemm::preq {

constexpr int kPreqQK8_1 = 32;
constexpr int kPreqBlockSize = 256;
constexpr std::size_t kPreqRowPad = 512;

struct PreqQ81Block {
  __half2 ds;
  int8_t qs[kPreqQK8_1];
};

static_assert(sizeof(PreqQ81Block) == sizeof(__half2) + 32,
              "preq-q81: Q8_1 activation block must be 36 bytes");
static_assert(kPreqQK8_1 == 32, "preq-q81: blocking factor is 32");

// ---- Verbatim device helpers (source: mmq common.hpp) ----

template <int width = 32>
static __device__ __forceinline__ float preq_warp_reduce_max(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = fmaxf(x, __shfl_xor(x, offset, width));
  }
  return x;
}

template <int width = 32>
static __device__ __forceinline__ int preq_warp_reduce_sum(int x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor(x, offset, width);
  }
  return x;
}

// ---- Verbatim row-quantize kernel (source: mmq quantize.hip.cpp) ----
// Contiguous-rows specialization of the source `quantize_q8_1`: the generic
// (ne00/s01/s02/s03/ne0/ne1/ne2) striding collapses to N rows x K cols
// (ne00=K, s01=K, ne0=padK, ne1=N, single channel), which is exactly how
// `qfn_mmq_quantize_q8_1` calls it. Body is byte-identical otherwise.
__launch_bounds__(kPreqBlockSize, 1) static __global__ void quantize_q8_1_row(
    const float* __restrict__ x, void* __restrict__ vy, int64_t ne00,
    int64_t s01, int64_t ne0, int ne1) {
  const int64_t i0 = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

  if (i0 >= ne0) {
    return;
  }

  const int64_t i1 = blockIdx.y;

  const int64_t i_cont = (int64_t)i1 * ne0 + i0;

  PreqQ81Block* y = (PreqQ81Block*)vy;

  const int64_t ib = i_cont / kPreqQK8_1;   // block index
  const int64_t iqs = i_cont % kPreqQK8_1;  // quant index

  const float xi = i0 < ne00 ? x[i1 * s01 + i0] : 0.0f;
  float amax = fabsf(xi);

  amax = preq_warp_reduce_max<kPreqQK8_1>(amax);

  const float d = amax / 127.0f;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  // block_q8_1's sum is d*sum(qs), the sum of the values the dot products
  // consume, as ggml's CPU quantize_row_q8_1 stores it. The sum of the
  // unquantized x would leave the Q4_1/Q5_1 min term and the Q4_K/Q5_K MMQ
  // mins with m*sum(x - d*q), a rounding bias shared by every output row.
  const int sumq = preq_warp_reduce_sum<kPreqQK8_1>((int)q);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds = make_half2(d, d * sumq);
}

}  // namespace gufo::core::compute::gemm::preq

// ---- Model-agnostic launcher (mirrors qfn_mmq_quantize_q8_1) ----
namespace gufo::core::compute::gemm {

std::size_t PreqQ81RowPad(std::size_t k) {
  constexpr std::size_t pad = preq::kPreqRowPad;
  return (k + pad - 1) / pad * pad;
}

std::size_t PreqQ81Bytes(std::size_t n, std::size_t k) {
  return n * PreqQ81RowPad(k) * sizeof(preq::PreqQ81Block) / preq::kPreqQK8_1;
}

int PrequantizeQ8_1(const float* x_f32, void* x_q8, std::size_t n,
                    std::size_t k, hipStream_t stream) {
  if (x_f32 == nullptr || x_q8 == nullptr || n == 0 || k == 0 || k % 32 != 0) {
    return -1;
  }
  const int64_t pad = static_cast<int64_t>(PreqQ81RowPad(k));
  const int64_t kk = static_cast<int64_t>(k);
  const int nn = static_cast<int>(n);
  const int64_t block_num_x =
      (pad + preq::kPreqBlockSize - 1) / preq::kPreqBlockSize;
  const dim3 num_blocks(static_cast<unsigned>(block_num_x),
                        static_cast<unsigned>(nn), 1);
  const dim3 block_size(preq::kPreqBlockSize, 1, 1);
  preq::quantize_q8_1_row<<<num_blocks, block_size, 0, stream>>>(x_f32, x_q8,
                                                                 kk, kk, pad, nn);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

// Caller-owned prequant scratch (§4.2: no `static` buffers inside
// operators). Fixed 8-token ceiling: decode never stages more than 8 rows,
// so one buffer sized for the widest decode projection input row serves the
// whole batch (launches on one stream are serial). Allocated once by the
// caller outside graph capture; capture only uses it, never grows it.
std::size_t NarrowQ80PrequantScratchBytes(std::size_t max_k) {
  const std::size_t k = (max_k + 31) / 32 * 32;
  return PreqQ81Bytes(8, k);
}

std::size_t NarrowQ80PrequantBytes(std::size_t tokens, std::size_t k) {
  return PreqQ81Bytes(tokens, k);
}

std::size_t NarrowQ80PrequantStrideBlocks(std::size_t k) {
  return PreqQ81RowPad(k) / preq::kPreqQK8_1;
}

// Shared quantize -> narrow-GEMV for F32 activations (27B / gemma4 decode).
// `scratch` is caller-owned (§4.2); null or short scratch refuses cleanly
// and the caller keeps its legacy path.
bool LaunchNarrowQ80Decode(const void* weights, const void* input,
                           float* output, std::size_t k, std::size_t rows,
                           std::size_t tokens,
                           std::size_t input_stride_blocks,
                           hipStream_t stream);

bool LaunchNarrowQ80Prequant(const void* weights, const float* x_f32,
                             float* output, std::size_t k, std::size_t rows,
                             std::size_t tokens, void* scratch,
                             std::size_t scratch_bytes, hipStream_t stream) {
  if (weights == nullptr || x_f32 == nullptr || output == nullptr) return false;
  if (tokens < 1 || tokens > 8 || k == 0 || k % 32 != 0 || rows == 0)
    return false;
  if (scratch == nullptr || scratch_bytes < NarrowQ80PrequantBytes(tokens, k))
    return false;
  if (PrequantizeQ8_1(x_f32, scratch, tokens, k, stream) != 0) return false;
  const std::size_t stride = NarrowQ80PrequantStrideBlocks(k);
  return LaunchNarrowQ80Decode(weights, scratch, output, k, rows, tokens,
                               stride, stream);
}

}  // namespace gufo::core::compute::gemm
