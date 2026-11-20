// Narrow single-token decode GEMVs for native Q5_0 / Q4_K weights against
// prequantized Q8_1 activations (the same activation block the Q8_0 narrow
// kernel consumes: half2 ds = (d, d * sum(q)) + 32 int8).
//
// Why: models such as gpt-oss ship dense attention weights as Q5_0 / Q4_K but
// the Q8_0-only narrow path forces a load-time conversion to Q8_0, which reads
// 1.4-1.9x more bytes per token on a memory-bound decode. These kernels read the
// native bits.
//
// Q5_0 uses a planar repack (qs plane, qh plane, d plane; see
// NarrowQ50PlanarBytes) so each lane issues one aligned 16 B load for a
// 32-weight block instead of unaligned 22 B block reads. Q4_K's 144 B block is
// already 16 B aligned and is read in place.
//
// One wave (32 lanes) owns one output row; 4 rows per 128-thread block.
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/compute/gemm/gemm.hpp"

namespace mbufo::core::compute::gemm {
namespace {

struct XBlock {  // block_q8_1
  __half2 ds;
  std::int8_t qs[32];
};
static_assert(sizeof(XBlock) == 36, "Q8_1 block is 36 bytes");

// Signed 4x int8 dot-accumulate. gfx11 (RDNA3/3.5) exposes it as the mixed-sign
// sudot4 with both operands marked signed (same choice as the narrow Q8_0 kernel).
__device__ __forceinline__ int Dp4a(int a, int b, int c) {
#if defined(__GFX11__) || defined(__GFX12__)
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
  return __builtin_amdgcn_sdot4(a, b, c, false);
#endif
}

// Spread the low 4 bits of t into bit 4 of each of 4 bytes.
__device__ __forceinline__ std::uint32_t Spread4(std::uint32_t t) {
  return ((t & 1u) << 4) | ((t & 2u) << 11) | ((t & 4u) << 18) |
         ((t & 8u) << 25);
}

__device__ __forceinline__ float WaveSum32(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
  return v;
}

__device__ __forceinline__ void ScaleMinK4(int j, const std::uint8_t* q,
                                           float* sc, float* mn) {
  if (j < 4) {
    *sc = static_cast<float>(q[j] & 63);
    *mn = static_cast<float>(q[j + 4] & 63);
  } else {
    *sc = static_cast<float>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *mn = static_cast<float>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

__global__ void NarrowQ50PlanarKernel(const std::uint8_t* __restrict__ qs_plane,
                                      const std::uint32_t* __restrict__ qh_plane,
                                      const __half* __restrict__ d_plane,
                                      const XBlock* __restrict__ x,
                                      float* __restrict__ y,
                                      const float* __restrict__ bias, int nb,
                                      int rows) {
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + wave;
  if (row >= rows) return;
  const std::size_t base = static_cast<std::size_t>(row) * nb;
  float sum = 0.0f;
  for (int b = lane; b < nb; b += 32) {
    const uint4 q = reinterpret_cast<const uint4*>(qs_plane)[base + b];
    const std::uint32_t qh = qh_plane[base + b];
    const float dw = __half2float(d_plane[base + b]);
    const XBlock& xb = x[b];
    const int* xs = reinterpret_cast<const int*>(xb.qs);
    const std::uint32_t qw[4] = {q.x, q.y, q.z, q.w};
    int sumi = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      // element j (< 16) = low nibble of qs[j] | qh bit j; element 16 + j =
      // high nibble of qs[j] | qh bit 16 + j (ggml Q5_0 layout); the -16
      // offset is folded in below through the activation sum.
      const std::uint32_t lo =
          (qw[w] & 0x0F0F0F0Fu) | Spread4((qh >> (4 * w)) & 0xFu);
      const std::uint32_t hi =
          ((qw[w] >> 4) & 0x0F0F0F0Fu) | Spread4((qh >> (16 + 4 * w)) & 0xFu);
      sumi = Dp4a(static_cast<int>(lo), xs[w], sumi);
      sumi = Dp4a(static_cast<int>(hi), xs[4 + w], sumi);
    }
    const float2 ds = __half22float2(xb.ds);
    sum += dw * (ds.x * static_cast<float>(sumi) - 16.0f * ds.y);
  }
  sum = WaveSum32(sum);
  if (lane == 0) y[row] = sum + (bias != nullptr ? bias[row] : 0.0f);
}

__global__ void NarrowQ4KRowAddKernel(const std::uint8_t* __restrict__ w,
                                      const XBlock* __restrict__ x,
                                      const float* __restrict__ bias,
                                      float* __restrict__ inout, int nblk,
                                      int rows) {
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + wave;
  if (row >= rows) return;
  const std::uint8_t* rowp = w + static_cast<std::size_t>(row) * nblk * 144;
  float sum = 0.0f;
  const int items = nblk * 4;  // (256-block, 64-element group) pairs
  for (int p = lane; p < items; p += 32) {
    const int b = p >> 2;
    const int g = p & 3;
    const std::uint8_t* blk = rowp + static_cast<std::size_t>(b) * 144;
    const uint4 hdr = *reinterpret_cast<const uint4*>(blk);
    const uint4* qp = reinterpret_cast<const uint4*>(blk + 16 + g * 32);
    const uint4 q0 = qp[0];
    const uint4 q1 = qp[1];
    const float d4 = __half2float(__ushort_as_half(
        static_cast<unsigned short>(hdr.x & 0xFFFFu)));
    const float dmin4 =
        __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));
    const std::uint8_t* sc = reinterpret_cast<const std::uint8_t*>(&hdr) + 4;
    float s0, m0, s1, m1;
    ScaleMinK4(2 * g, sc, &s0, &m0);
    ScaleMinK4(2 * g + 1, sc, &s1, &m1);
    const XBlock& xl = x[b * 8 + 2 * g];
    const XBlock& xh = x[b * 8 + 2 * g + 1];
    const int* xls = reinterpret_cast<const int*>(xl.qs);
    const int* xhs = reinterpret_cast<const int*>(xh.qs);
    const std::uint32_t qa[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    int slo = 0, shi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      slo = Dp4a(static_cast<int>(qa[i] & 0x0F0F0F0Fu), xls[i], slo);
      shi = Dp4a(static_cast<int>((qa[i] >> 4) & 0x0F0F0F0Fu), xhs[i], shi);
    }
    const float2 dl = __half22float2(xl.ds);
    const float2 dh = __half22float2(xh.ds);
    sum += d4 * (s0 * dl.x * static_cast<float>(slo) +
                 s1 * dh.x * static_cast<float>(shi)) -
           dmin4 * (m0 * dl.y + m1 * dh.y);
  }
  sum = WaveSum32(sum);
  if (lane == 0) inout[row] += sum + bias[row];
}

}  // namespace

std::size_t NarrowQ50PlanarBytes(std::size_t rows, std::size_t cols) {
  return rows * (cols / 32) * (16 + 4 + 2);
}

bool LaunchNarrowQ50Planar(const void* planar_w, const void* x_q8_1, float* y,
                           const float* bias, std::size_t k, std::size_t rows,
                           hipStream_t stream) {
  if (planar_w == nullptr || x_q8_1 == nullptr || y == nullptr || k == 0 ||
      k % 32 != 0 || rows == 0)
    return false;
  const std::size_t nb = k / 32;
  const auto* qs = static_cast<const std::uint8_t*>(planar_w);
  const auto* qh = reinterpret_cast<const std::uint32_t*>(qs + rows * nb * 16);
  const auto* d = reinterpret_cast<const __half*>(qs + rows * nb * (16 + 4));
  hipLaunchKernelGGL(NarrowQ50PlanarKernel,
                     dim3(static_cast<unsigned>((rows + 3) / 4)), dim3(128), 0,
                     stream, qs, qh, d, static_cast<const XBlock*>(x_q8_1), y,
                     bias, static_cast<int>(nb), static_cast<int>(rows));
  return true;
}

bool LaunchNarrowQ4KRowAdd(const void* w, const void* x_q8_1, const float* bias,
                           float* inout, std::size_t k, std::size_t rows,
                           hipStream_t stream) {
  if (w == nullptr || x_q8_1 == nullptr || bias == nullptr || inout == nullptr ||
      k == 0 || k % 256 != 0 || rows == 0)
    return false;
  hipLaunchKernelGGL(NarrowQ4KRowAddKernel,
                     dim3(static_cast<unsigned>((rows + 3) / 4)), dim3(128), 0,
                     stream, static_cast<const std::uint8_t*>(w),
                     static_cast<const XBlock*>(x_q8_1), bias, inout,
                     static_cast<int>(k / 256), static_cast<int>(rows));
  return true;
}

}  // namespace mbufo::core::compute::gemm

#endif  // ENGINE_ENABLE_HIP
