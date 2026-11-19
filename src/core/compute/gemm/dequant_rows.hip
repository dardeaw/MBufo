// Shared quantized-row decoding for model backends: embedding gather and
// dequantize-to-F16 of packed GGML rows.
//
// Every model used to carry its own EmbedKernel that understood only a few
// formats (Q8_0/F16, then "anything else is F32"), so a GGUF whose
// token_embd.weight is Q4_K/Q5_K (UD-Q4_K_XL files) was read as floats and
// faulted. The element decoders below follow the ggml block layouts
// (dequantize_row_q4_0/q8_0/q4_K/q5_K/q6_K) so one implementation serves all
// of them. `DequantRowsF16` is the prefill fallback for packed weights that
// have no WMMA/MMQ kernel yet: dequantize into a dense F16 scratch, then use
// the dense F16 GEMM.
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "src/core/compute/gemm/gemm.hpp"

namespace gufo::core::compute::gemm {
namespace {

constexpr int kBlk32Q4_0 = 18;
constexpr int kBlk32Q8_0 = 34;
constexpr int kBlk256Q4K = 144;
constexpr int kBlk256Q5K = 176;
constexpr int kBlk256Q6K = 210;

__host__ __device__ inline std::size_t RowBytesOf(int type, std::size_t cols) {
  switch (type) {
    case GUFO_GEMM_F32:
      return cols * 4;
    case GUFO_GEMM_F16:
    case GUFO_GEMM_BF16:
      return cols * 2;
    case GUFO_GEMM_Q4_0:
      return (cols / 32) * kBlk32Q4_0;
    case GUFO_GEMM_Q8_0:
      return (cols / 32) * kBlk32Q8_0;
    case GUFO_GEMM_Q4_K:
      return (cols / 256) * kBlk256Q4K;
    case GUFO_GEMM_Q5_K:
      return (cols / 256) * kBlk256Q5K;
    case GUFO_GEMM_Q6_K:
      return (cols / 256) * kBlk256Q6K;
    default:
      return 0;
  }
}

__host__ inline bool ColsOk(int type, std::size_t cols) {
  switch (type) {
    case GUFO_GEMM_Q4_0:
    case GUFO_GEMM_Q8_0:
      return cols % 32 == 0;
    case GUFO_GEMM_Q4_K:
    case GUFO_GEMM_Q5_K:
    case GUFO_GEMM_Q6_K:
      return cols % 256 == 0;
    default:
      return RowBytesOf(type, cols) != 0;
  }
}

__device__ __forceinline__ float HalfAt(const std::uint8_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}

__device__ __forceinline__ void ScaleMinK4(std::uint32_t j,
                                           const std::uint8_t* q, float* sc,
                                           float* mn) {
  if (j < 4) {
    *sc = static_cast<float>(q[j] & 63);
    *mn = static_cast<float>(q[j + 4] & 63);
  } else {
    *sc = static_cast<float>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *mn = static_cast<float>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

// Element `i` of a packed row (`row` points at the row start). The type is a
// template parameter so every kernel instance decodes exactly one format.
template <int kType>
__device__ __forceinline__ float RowElementAt(const std::uint8_t* row,
                                              std::uint32_t i) {
  switch (kType) {
    case GUFO_GEMM_F32:
      return reinterpret_cast<const float*>(row)[i];
    case GUFO_GEMM_F16:
      return __half2float(reinterpret_cast<const __half*>(row)[i]);
    case GUFO_GEMM_BF16:
      return __uint_as_float(
          static_cast<std::uint32_t>(
              reinterpret_cast<const std::uint16_t*>(row)[i])
          << 16);
    case GUFO_GEMM_Q4_0: {
      const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * kBlk32Q4_0;
      const std::uint32_t e = i % 32;
      const std::uint8_t b = blk[2 + (e & 15u)];
      const int q = (e < 16u ? (b & 0xF) : (b >> 4)) - 8;
      return static_cast<float>(q) * HalfAt(blk);
    }
    case GUFO_GEMM_Q8_0: {
      const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * kBlk32Q8_0;
      return static_cast<float>(static_cast<std::int8_t>(blk[2 + (i % 32)])) *
             HalfAt(blk);
    }
    case GUFO_GEMM_Q4_K: {
      const std::uint8_t* blk = row + static_cast<std::size_t>(i / 256) * kBlk256Q4K;
      const std::uint32_t e = i % 256;
      const std::uint32_t g = e / 64u, r = e % 64u, l = r % 32u;
      const std::uint32_t hi = r >= 32u ? 1u : 0u;
      float sc, mn;
      ScaleMinK4(2u * g + hi, blk + 4, &sc, &mn);
      const std::uint8_t b = blk[16 + g * 32u + l];
      const float q = static_cast<float>(hi ? (b >> 4) : (b & 0xF));
      return HalfAt(blk) * sc * q - HalfAt(blk + 2) * mn;
    }
    case GUFO_GEMM_Q5_K: {
      const std::uint8_t* blk = row + static_cast<std::size_t>(i / 256) * kBlk256Q5K;
      const std::uint32_t e = i % 256;
      const std::uint32_t g = e / 64u, r = e % 64u, l = r % 32u;
      const std::uint32_t hi = r >= 32u ? 1u : 0u;
      const std::uint32_t j = 2u * g + hi;
      float sc, mn;
      ScaleMinK4(j, blk + 4, &sc, &mn);
      const std::uint8_t b = blk[48 + g * 32u + l];
      const float q = static_cast<float>((hi ? (b >> 4) : (b & 0xF)) +
                                         (((blk[16 + l] >> j) & 1) ? 16 : 0));
      return HalfAt(blk) * sc * q - HalfAt(blk + 2) * mn;
    }
    case GUFO_GEMM_Q6_K: {
      const std::uint8_t* blk = row + static_cast<std::size_t>(i / 256) * kBlk256Q6K;
      const std::uint32_t e = i % 256;
      const std::uint32_t n = e / 128u, r = e % 128u, l = r % 32u, q = r / 32u;
      const std::uint8_t lo = blk[n * 64u + l + (q & 1u) * 32u];
      const std::uint8_t hb = blk[128 + n * 32u + l];
      const int v = ((q >= 2u ? (lo >> 4) : (lo & 0xF)) |
                     (((hb >> (2u * q)) & 3) << 4)) -
                    32;
      const float sc = static_cast<float>(
          static_cast<std::int8_t>(blk[192 + n * 8u + l / 16u + 2u * q]));
      return HalfAt(blk + 208) * sc * static_cast<float>(v);
    }
    default:
      return 0.0f;
  }
}

template <int kType>
__global__ void EmbedGatherKernel(const std::uint8_t* __restrict__ table,
                                  std::size_t row_bytes,
                                  const std::int32_t* __restrict__ ids,
                                  float* __restrict__ out,
                                  std::uint32_t hidden) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * 256u + threadIdx.x;
  if (i >= hidden)
    return;
  const std::uint8_t* row =
      table + static_cast<std::size_t>(ids[t]) * row_bytes;
  out[static_cast<std::size_t>(t) * hidden + i] = RowElementAt<kType>(row, i);
}

template <int kType>
__global__ void DequantRowsKernel(const std::uint8_t* __restrict__ w,
                                  std::size_t row_bytes,
                                  __half* __restrict__ out,
                                  std::size_t row0, std::size_t cols) {
  const std::size_t r = row0 + blockIdx.y;
  const std::size_t c = static_cast<std::size_t>(blockIdx.x) * 256u + threadIdx.x;
  if (c >= cols)
    return;
  out[r * cols + c] = __float2half(
      RowElementAt<kType>(w + r * row_bytes, static_cast<std::uint32_t>(c)));
}

// Calls f(std::integral_constant<int, T>{}) for the runtime type code; false when
// the type has no decoder.
template <class F>
bool DispatchType(int type, F&& f) {
  switch (type) {
    case GUFO_GEMM_F32:  f(std::integral_constant<int, GUFO_GEMM_F32>{});  return true;
    case GUFO_GEMM_F16:  f(std::integral_constant<int, GUFO_GEMM_F16>{});  return true;
    case GUFO_GEMM_BF16: f(std::integral_constant<int, GUFO_GEMM_BF16>{}); return true;
    case GUFO_GEMM_Q4_0: f(std::integral_constant<int, GUFO_GEMM_Q4_0>{}); return true;
    case GUFO_GEMM_Q8_0: f(std::integral_constant<int, GUFO_GEMM_Q8_0>{}); return true;
    case GUFO_GEMM_Q4_K: f(std::integral_constant<int, GUFO_GEMM_Q4_K>{}); return true;
    case GUFO_GEMM_Q5_K: f(std::integral_constant<int, GUFO_GEMM_Q5_K>{}); return true;
    case GUFO_GEMM_Q6_K: f(std::integral_constant<int, GUFO_GEMM_Q6_K>{}); return true;
    default: return false;
  }
}

}  // namespace

std::size_t DequantRowBytes(int type, std::size_t cols) {
  return ColsOk(type, cols) ? RowBytesOf(type, cols) : 0;
}

bool EmbedGather(const void* table, int type, const std::int32_t* ids,
                 float* out, std::size_t n_tokens, std::size_t hidden,
                 hipStream_t stream) {
  if (table == nullptr || ids == nullptr || out == nullptr)
    return false;
  if (n_tokens == 0)
    return true;
  if (!ColsOk(type, hidden) || n_tokens > 65535)
    return false;
  const std::size_t row_bytes = RowBytesOf(type, hidden);
  const unsigned waves = static_cast<unsigned>((hidden + 255) / 256);
  return DispatchType(type, [&](auto tc) {
    constexpr int kT = decltype(tc)::value;
    hipLaunchKernelGGL((EmbedGatherKernel<kT>),
                       dim3(static_cast<unsigned>(n_tokens), waves), dim3(256),
                       0, stream, static_cast<const std::uint8_t*>(table),
                       row_bytes, ids, out,
                       static_cast<std::uint32_t>(hidden));
  });
}

bool DequantRowsF16(const void* w, int type, __half* out, std::size_t rows,
                    std::size_t cols, hipStream_t stream) {
  if (w == nullptr || out == nullptr)
    return false;
  if (rows == 0 || cols == 0)
    return true;
  if (!ColsOk(type, cols))
    return false;
  const std::size_t row_bytes = RowBytesOf(type, cols);
  const unsigned col_blocks = static_cast<unsigned>((cols + 255) / 256);
  return DispatchType(type, [&](auto tc) {
    constexpr int kT = decltype(tc)::value;
    for (std::size_t row0 = 0; row0 < rows; row0 += 65535) {
      const std::size_t n = rows - row0 < 65535 ? rows - row0 : 65535;
      hipLaunchKernelGGL((DequantRowsKernel<kT>),
                         dim3(col_blocks, static_cast<unsigned>(n)), dim3(256),
                         0, stream, static_cast<const std::uint8_t*>(w),
                         row_bytes, out, row0, cols);
    }
  });
}

}  // namespace gufo::core::compute::gemm

#endif  // ENGINE_ENABLE_HIP
