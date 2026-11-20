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

namespace mbufo::core::compute::gemm {
namespace {

namespace q = ::mbufo::core::quant;

// kScaled multiplies the decoded element by `scale` (embedding scaling, e.g. Gemma's sqrt(hidden)); the
// unscaled instance does no extra multiply so its output stays bit-identical to the plain decode.
// `streams` > 1 writes every decoded element to that many consecutive output rows (out[(t*streams+s)*hidden+i]);
// flash-next's residual keeps one copy of the embedding per stream.
template <class F, bool kScaled>
__global__ void EmbedGatherKernel(const std::uint8_t* __restrict__ table, std::size_t row_bytes,
                                  const std::int32_t* __restrict__ ids, float* __restrict__ out,
                                  std::uint32_t hidden, float scale, std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * 256u + threadIdx.x;
  if (i >= hidden) return;
  const std::uint8_t* row = table + static_cast<std::size_t>(ids[t]) * row_bytes;
  const float v = kScaled ? q::ScaledElementOf<F>(row, i, scale) : F::Element(row, i);
  for (std::uint32_t s = 0; s < streams; ++s)
    out[(static_cast<std::size_t>(t) * streams + s) * hidden + i] = v;
}

// One token passed by value (no device id buffer needed).
template <class F>
__global__ void EmbedGatherOneKernel(const std::uint8_t* __restrict__ table, std::size_t row_bytes,
                                     std::uint32_t token, float* __restrict__ out, std::uint32_t hidden) {
  const std::uint32_t i = blockIdx.x * 256u + threadIdx.x;
  if (i >= hidden) return;
  out[i] = F::Element(table + static_cast<std::size_t>(token) * row_bytes, i);
}

template <bool kScaled>
bool EmbedGatherImpl(const void* table, int type, const std::int32_t* ids, float* out, std::size_t n_tokens,
                     std::size_t hidden, float scale, std::uint32_t streams, hipStream_t stream) {
  if (table == nullptr || ids == nullptr || out == nullptr || streams == 0) return false;
  if (n_tokens == 0) return true;
  if (n_tokens > 65535) return false;
  bool launched = false;
  const bool known = q::AllFormats::Dispatch(type, [&](auto fm) {
    using F = decltype(fm);
    if (!F::ColsOk(hidden)) return;
    const unsigned waves = static_cast<unsigned>((hidden + 255) / 256);
    hipLaunchKernelGGL((EmbedGatherKernel<F, kScaled>), dim3(static_cast<unsigned>(n_tokens), waves), dim3(256),
                       0, stream, static_cast<const std::uint8_t*>(table), F::RowBytes(hidden), ids, out,
                       static_cast<std::uint32_t>(hidden), scale, streams);
    launched = true;
  });
  return known && launched;
}

template <class F>
__global__ void DequantRowsKernel(const std::uint8_t* __restrict__ w, std::size_t row_bytes,
                                  __half* __restrict__ out, std::size_t row0, std::size_t cols) {
  const std::size_t r = row0 + blockIdx.y;
  const std::size_t c = static_cast<std::size_t>(blockIdx.x) * 256u + threadIdx.x;
  if (c >= cols) return;
  out[r * cols + c] = __float2half(F::Element(w + r * row_bytes, static_cast<std::uint32_t>(c)));
}

// One thread decodes 8 consecutive elements of one row (one 16-byte output chunk) and stores them with a single
// 16-byte write; consecutive threads write consecutive chunks of the tiled layout, so the whole store stream is
// coalesced (the packed input is small: ~0.56 byte/element for Q4_K).
template <class F>
__global__ void DequantRowsTiledKernel(const std::uint8_t* __restrict__ w, std::size_t row_bytes,
                                       __half* __restrict__ out, std::size_t cols, std::size_t n_chunks) {
  const std::size_t o = static_cast<std::size_t>(blockIdx.x) * 256u + threadIdx.x;
  if (o >= n_chunks) return;
  // o = (((row_tile * kblocks + kb) * 128 + row_in_tile) * 4 + chunk_in_k32)
  const std::size_t cc = o & 3u, rin = (o >> 2) & 127u, t = o >> 9;
  const std::size_t kblocks = cols >> 5;
  const std::size_t kb = t % kblocks, rt = t / kblocks;
  const std::size_t r = rt * 128 + rin;
  const std::uint32_t c0 = static_cast<std::uint32_t>(kb * 32 + cc * 8);
  const std::uint8_t* row = w + r * row_bytes;
  float v[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) v[e] = F::Element(row, c0 + static_cast<std::uint32_t>(e));
  __half2 h[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) h[e] = __floats2half2_rn(v[2 * e], v[2 * e + 1]);
  *reinterpret_cast<uint4*>(out + o * 8) = *reinterpret_cast<const uint4*>(h);
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
  return EmbedGatherImpl<false>(table, type, ids, out, n_tokens, hidden, 1.0f, 1, stream);
}

bool EmbedGatherScaled(const void* table, int type, const std::int32_t* ids, float* out, std::size_t n_tokens,
                       std::size_t hidden, float scale, hipStream_t stream) {
  return EmbedGatherImpl<true>(table, type, ids, out, n_tokens, hidden, scale, 1, stream);
}

bool EmbedGatherStreams(const void* table, int type, const std::int32_t* ids, float* out, std::size_t n_tokens,
                        std::size_t hidden, std::uint32_t streams, hipStream_t stream) {
  return EmbedGatherImpl<false>(table, type, ids, out, n_tokens, hidden, 1.0f, streams, stream);
}

bool EmbedGatherOne(const void* table, int type, std::uint32_t token, float* out, std::size_t hidden,
                    hipStream_t stream) {
  if (table == nullptr || out == nullptr) return false;
  bool launched = false;
  const bool known = q::AllFormats::Dispatch(type, [&](auto fm) {
    using F = decltype(fm);
    if (!F::ColsOk(hidden)) return;
    hipLaunchKernelGGL((EmbedGatherOneKernel<F>), dim3(static_cast<unsigned>((hidden + 255) / 256)), dim3(256), 0,
                       stream, static_cast<const std::uint8_t*>(table), F::RowBytes(hidden), token, out,
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

bool DequantRowsF16Tiled(const void* w, int type, __half* out, std::size_t rows, std::size_t cols,
                         hipStream_t stream) {
  if (w == nullptr || out == nullptr) return false;
  if (rows == 0 || cols == 0) return true;
  if (rows % kTiledRows != 0 || cols % 32 != 0) return false;
  bool launched = false;
  const bool known = q::AllFormats::Dispatch(type, [&](auto fm) {
    using F = decltype(fm);
    if (!F::ColsOk(cols)) return;
    const std::size_t row_bytes = F::RowBytes(cols);
    const std::size_t n_chunks = rows * cols / 8;
    hipLaunchKernelGGL((DequantRowsTiledKernel<F>), dim3(static_cast<unsigned>((n_chunks + 255) / 256)), dim3(256), 0,
                       stream, static_cast<const std::uint8_t*>(w), row_bytes, out, cols, n_chunks);
    launched = true;
  });
  return known && launched;
}

}  // namespace mbufo::core::compute::gemm

#endif  // ENGINE_ENABLE_HIP
