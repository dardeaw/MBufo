// Shared quantized-row decoding for model backends: embedding gather and
// dequantize-to-F16 of packed GGML rows.
//
// Built on core::quant::Traits (src/core/quant/quant_traits.hpp): one element
// decoder per format, one dispatch list. Every model used to carry its own
// EmbedKernel that understood only a few formats (Q8_0/F16, then "anything else is
// F32"), so a GGUF whose token_embd.weight is Q4_K/Q5_K (UD-Q4_K_XL files) was read
// as floats and faulted. A format that is not in quant::AllFormats is refused
// (false), never decoded as another format.
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/quant/quant_traits.hpp"

namespace gufo::core::compute::gemm {
namespace {

namespace q = ::gufo::core::quant;

template <class F>
__global__ void EmbedGatherKernel(const std::uint8_t* __restrict__ table, std::size_t row_bytes,
                                  const std::int32_t* __restrict__ ids, float* __restrict__ out,
                                  std::uint32_t hidden) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * 256u + threadIdx.x;
  if (i >= hidden) return;
  const std::uint8_t* row = table + static_cast<std::size_t>(ids[t]) * row_bytes;
  out[static_cast<std::size_t>(t) * hidden + i] = F::Element(row, i);
}

template <class F>
__global__ void DequantRowsKernel(const std::uint8_t* __restrict__ w, std::size_t row_bytes,
                                  __half* __restrict__ out, std::size_t row0, std::size_t cols) {
  const std::size_t r = row0 + blockIdx.y;
  const std::size_t c = static_cast<std::size_t>(blockIdx.x) * 256u + threadIdx.x;
  if (c >= cols) return;
  out[r * cols + c] = __float2half(F::Element(w + r * row_bytes, static_cast<std::uint32_t>(c)));
}

}  // namespace

std::size_t DequantRowBytes(int type, std::size_t cols) {
  std::size_t bytes = 0;
  (void)q::AllFormats::Dispatch(type, [&](auto fm) {
    using F = decltype(fm);
    if (F::ColsOk(cols)) bytes = F::RowBytes(cols);
  });
  return bytes;
}

bool EmbedGather(const void* table, int type, const std::int32_t* ids, float* out, std::size_t n_tokens,
                 std::size_t hidden, hipStream_t stream) {
  if (table == nullptr || ids == nullptr || out == nullptr) return false;
  if (n_tokens == 0) return true;
  if (n_tokens > 65535) return false;
  bool launched = false;
  const bool known = q::AllFormats::Dispatch(type, [&](auto fm) {
    using F = decltype(fm);
    if (!F::ColsOk(hidden)) return;
    const unsigned waves = static_cast<unsigned>((hidden + 255) / 256);
    hipLaunchKernelGGL((EmbedGatherKernel<F>), dim3(static_cast<unsigned>(n_tokens), waves), dim3(256), 0,
                       stream, static_cast<const std::uint8_t*>(table), F::RowBytes(hidden), ids, out,
                       static_cast<std::uint32_t>(hidden));
    launched = true;
  });
  return known && launched;
}

bool DequantRowsF16(const void* w, int type, __half* out, std::size_t rows, std::size_t cols,
                    hipStream_t stream) {
  if (w == nullptr || out == nullptr) return false;
  if (rows == 0 || cols == 0) return true;
  bool launched = false;
  const bool known = q::AllFormats::Dispatch(type, [&](auto fm) {
    using F = decltype(fm);
    if (!F::ColsOk(cols)) return;
    const std::size_t row_bytes = F::RowBytes(cols);
    const unsigned col_blocks = static_cast<unsigned>((cols + 255) / 256);
    for (std::size_t row0 = 0; row0 < rows; row0 += 65535) {
      const std::size_t n = rows - row0 < 65535 ? rows - row0 : 65535;
      hipLaunchKernelGGL((DequantRowsKernel<F>), dim3(col_blocks, static_cast<unsigned>(n)), dim3(256), 0,
                         stream, static_cast<const std::uint8_t*>(w), row_bytes, out, row0, cols);
    }
    launched = true;
  });
  return known && launched;
}

}  // namespace gufo::core::compute::gemm

#endif  // ENGINE_ENABLE_HIP
