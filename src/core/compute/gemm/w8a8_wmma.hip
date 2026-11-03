// Shared W8A8 prefill GEMM with host-side variable-length batch handling
// (Phase B-3, GGUF通用化.md v2 §3.2).
//
// Provenance (mechanical搬入, no algorithm change):
//  * The kernel template WKQuantA8BlockedWmmaGEMMKernel is included from
//    src/models/qwen/hip/kernels/prefill_quant_gemm.hpp — the same header
//    flash-next's w8a8_wave64.hip includes. Not one line of the template is
//    touched here; the instantiation below uses the identical arguments as
//    flash-next W8A8GemmWave64 (<128,128,2,4,1,kQ8_0,64>), so the aligned
//    prefix runs the original kernel as the original binary (SASS逐字相同).
//  * The scalar tail kernel is the <false> (F32 out) branch of
//    W8A8TailKernel from qwen38_flash_next/.../kernels.hip.cpp, renamed.
//    Same tiled-activation reads, same sudot4 + readlane reduction.
//  * W8A8GemmVarlen applies the §3.2铁律 on the host: main = batch & ~15
//    via the Wave64 launch, tail (if any) via the scalar kernel. No
//    in-kernel mask is added anywhere. Uncovered shapes return false and
//    the caller keeps its legacy route bit-identical.
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::core::compute::gemm {
namespace detail {

// W8A8 tiled-activation layout, verbatim from the kernel family that writes
// and reads it (flash-next kernels.hip.cpp): 16 tokens x one 32-wide K
// block, codes in fragment order (two 256-byte halves of 16 tokens x 16
// codes) followed by the sixteen per-token scales.
constexpr std::size_t kQ8ActTileTokens = 16;
constexpr std::size_t kQ8ActTileBytes = 576;
constexpr std::size_t kQ8ActScaleOffset = 512;
__device__ __forceinline__ const std::int8_t* Q8ActTile(const void* base,
                                                       std::size_t num_blocks,
                                                       std::size_t tt,
                                                       std::size_t kb) {
  return static_cast<const std::int8_t*>(base) +
         (((tt * num_blocks) + kb) * kQ8ActTileBytes);
}

// Keeps fast-math from folding a stored value's arithmetic into what follows.
__device__ __forceinline__ float Materialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}

}  // namespace detail

// Native wave64 launch, verbatim geometry of flash-next W8A8GemmWave64:
// one 128x128 macro tile per block, 256 threads, no tail handling here.
void W8A8GemmWave64Core(const void* w, const void* x_tiled, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 127) / 128),
                  static_cast<unsigned>((m + 127) / 128));
  gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 128, 2, 4, 1,
                                            core::GgmlType::kQ8_0, 64>
      <<<grid, 256, 0, stream>>>(w, x_tiled, out, batch, m, k);
}

hipError_t PrepareW8A8GemmWave64Core() {
  hipFuncAttributes attributes{};
  return hipFuncGetAttributes(
      &attributes, reinterpret_cast<const void*>(
                       gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<
                           128, 128, 2, 4, 1, core::GgmlType::kQ8_0, 64>));
}

// Scalar tail for rows [first, batch): one wave per output, lanes form the
// exact integer dot and dx * c of every 32-wide K block, then the K blocks'
// FMAs run in order through lane reads. Covers tail <= 16, k <= 10240.
__launch_bounds__(256) __global__
    void W8A8TailScalarKernel(const void* __restrict__ w,
                              const void* __restrict__ x_blocks,
                              float* __restrict__ y, std::size_t first,
                              std::size_t m, std::size_t k) {
  constexpr int kMaxSteps = 10240 / 32 / 32;
  const std::size_t row =
      (static_cast<std::size_t>(blockIdx.x) * 8) + (threadIdx.x >> 5u);
  const std::size_t tok = first + blockIdx.y;
  const int lane = static_cast<int>(threadIdx.x & 31u);
  if (row >= m) {
    return;
  }
  const int num_kb = static_cast<int>(k / 32);
  const auto* w_row =
      static_cast<const gufo::hip::Q8_0Block*>(w) + (row * num_kb);
  const std::size_t tl = tok % detail::kQ8ActTileTokens;
  float t[kMaxSteps];
  float dw[kMaxSteps];
#pragma unroll
  for (int s = 0; s < kMaxSteps; ++s) {
    const int kb = (s * 32) + lane;
    t[s] = 0.0F;
    dw[s] = 0.0F;
    if (kb < num_kb) {
      const gufo::hip::Q8_0Block& blk = w_row[kb];
      int qs[8];
      __builtin_memcpy(qs, blk.qs, sizeof(qs));
      const auto* tile = detail::Q8ActTile(
          x_blocks, static_cast<std::size_t>(num_kb),
          tok / detail::kQ8ActTileTokens, static_cast<std::size_t>(kb));
      const auto lo = *reinterpret_cast<const int4*>(tile + (tl * 16));
      const auto hi = *reinterpret_cast<const int4*>(tile + 256 + (tl * 16));
      const int xs[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
      int c = 0;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        c = __builtin_amdgcn_sudot4(true, qs[q], true, xs[q], c, false);
      }
      const float dx = *reinterpret_cast<const float*>(
          tile + detail::kQ8ActScaleOffset + (tl * sizeof(float)));
      t[s] = __fmul_rn(dx, static_cast<float>(c));
      dw[s] = __half2float(blk.d);
    }
  }
  float acc = detail::Materialize(0.0F);
#pragma unroll
  for (int s = 0; s < kMaxSteps; ++s) {
#pragma unroll
    for (int l = 0; l < 32; ++l) {
      if ((s * 32) + l < num_kb) {
        const float ts = __builtin_bit_cast(
            float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, t[s]), l));
        const float ws = __builtin_bit_cast(
            float,
            __builtin_amdgcn_readlane(__builtin_bit_cast(int, dw[s]), l));
        acc = __fmaf_rn(ts, ws, acc);
      }
    }
  }
  if (lane == 0) {
    y[(tok * m) + row] = acc;
  }
}

bool W8A8GemmVarlen(const void* w, const void* x_tiled, float* out,
                    std::size_t batch, std::size_t m, std::size_t k,
                    hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || (k % 32 != 0)) {
    return false;
  }
  if (!W8A8VarlenCovers(m, k) || k > kW8A8VarlenMaxK) {
    return false;
  }
  std::size_t main = 0;
  std::size_t tail = 0;
  W8A8SplitBatch(batch, &main, &tail);
  if (main != 0) {
    W8A8GemmWave64Core(w, x_tiled, out, main, m, k, stream);
  }
  if (tail != 0) {
    hipLaunchKernelGGL(W8A8TailScalarKernel,
                       dim3(static_cast<unsigned>((m + 7) / 8),
                            static_cast<unsigned>(tail)),
                       dim3(kThreads), 0, stream, w, x_tiled, out, main, m, k);
  }
  return true;
}

}  // namespace gufo::core::compute::gemm

#endif  // defined(ENGINE_ENABLE_HIP)
