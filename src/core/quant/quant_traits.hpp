#ifndef MBUFO_CORE_QUANT_QUANT_TRAITS_HPP_
#define MBUFO_CORE_QUANT_QUANT_TRAITS_HPP_

// One traits type per packed GGML weight format, checked by a concept, plus the
// single list every type dispatch in core goes through.
//
// Why: the kernels used to switch on a raw type code with an implicit "everything
// else is Q4_0 / Q8_0" default (GemvBatched), or take no type at all and silently
// assume Q8_0 (DenseF16Gemm). A K-quant weight then read garbage or faulted. Here a
// format is only usable if its Traits satisfy `QuantFormat` (geometry, element
// decode, and how its decode GEMV and prefill GEMM are provided). Adding a format
// means writing a Traits specialization and listing it in `AllFormats`; the
// static_asserts and the exhaustive `switch` over DecodePath / PrefillPath (no
// default, -Wswitch) refuse to compile until every operation is accounted for.
// A type code that is not in the list is reported as unsupported at run time
// ([[nodiscard]] status), never decoded as some other format.
//
// HIP translation units only (uses __half and __host__ __device__).
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/quant/iq2_tables.hpp"
#include "src/core/quant/quant_status.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>

#define MBUFO_QT_FN __host__ __device__ __forceinline__

namespace mbufo::core::quant {

/// GGML type codes (same values as core::GgmlType / the MBUFO_GEMM_* codes).
enum class FormatId : int {
  kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ4_1 = 3, kQ5_0 = 6, kQ5_1 = 7, kQ8_0 = 8, kQ2_K = 10, kQ4_K = 12, kQ5_K = 13, kIQ2_XXS = 16,
  kQ6_K = 14, kQ3_K = 11, kIQ4_NL = 20, kIQ4_XS = 23, kBF16 = 30, kMXFP4 = 39, kQ4_0_ROCMFP4 = 100, kQ4_0_ROCMFP4_FAST = 101, kQ3_0_ROCMFPX = 104,
};

/// How single-token (1..8 rows) decode GEMV is provided for a format.
enum class DecodePath {
  kNarrowQ8,   ///< prequantized-Q8_1 narrow kernel (Q8_0), legacy GEMV as the fallback
  kBlockGemv,  ///< K-quant block GEMV on F32 activations (LaunchQ8KBlockGEMV)
  kLegacyGemv, ///< wave32 GEMV (GemvBatched): F32/F16/BF16/Q4_0
  kNone,       ///< no decode kernel: the caller dequantizes and uses the dense path
};

/// How batched (prefill) GEMM is provided for a format.
enum class PrefillPath {
  kDenseQ8,         ///< shared dense F16 GEMM that decodes Q8_0 blocks in LDS
  kPassthroughF16,  ///< weights already F16: unquantized F16 GEMM
  kDequantF16,      ///< dequantize into an F16 scratch, then the unquantized F16 GEMM
};

template <FormatId> struct Traits;  // primary template undefined: unlisted format = compile error

namespace detail {
MBUFO_QT_FN float HalfAt(const std::uint8_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}
MBUFO_QT_FN void ScaleMinK4(int j, const std::uint8_t* q, float* sc, float* mn) {
  if (j < 4) {
    *sc = static_cast<float>(q[j] & 63);
    *mn = static_cast<float>(q[j + 4] & 63);
  } else {
    *sc = static_cast<float>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *mn = static_cast<float>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}
}  // namespace detail

template <> struct Traits<FormatId::kF32> {
  static constexpr FormatId kId = FormatId::kF32;
  static constexpr unsigned kBlockElems = 1, kBlockBytes = 4;
  static constexpr DecodePath kDecode = DecodePath::kLegacyGemv;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols * 4; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    return reinterpret_cast<const float*>(row)[i];
  }
};

template <> struct Traits<FormatId::kF16> {
  static constexpr FormatId kId = FormatId::kF16;
  static constexpr unsigned kBlockElems = 1, kBlockBytes = 2;
  static constexpr DecodePath kDecode = DecodePath::kLegacyGemv;
  static constexpr PrefillPath kPrefill = PrefillPath::kPassthroughF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols * 2; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    return __half2float(reinterpret_cast<const __half*>(row)[i]);
  }
};

template <> struct Traits<FormatId::kBF16> {
  static constexpr FormatId kId = FormatId::kBF16;
  static constexpr unsigned kBlockElems = 1, kBlockBytes = 2;
  static constexpr DecodePath kDecode = DecodePath::kLegacyGemv;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols * 2; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    return __builtin_bit_cast(float, static_cast<std::uint32_t>(reinterpret_cast<const std::uint16_t*>(row)[i]) << 16);
  }
};

template <> struct Traits<FormatId::kQ4_0> {
  static constexpr FormatId kId = FormatId::kQ4_0;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 18;
  static constexpr DecodePath kDecode = DecodePath::kLegacyGemv;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 32 * 18; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 32 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 18;
    const std::uint32_t e = i % 32;
    const std::uint8_t b = blk[2 + (e & 15u)];
    const int q = (e < 16u ? (b & 0xF) : (b >> 4)) - 8;
    return static_cast<float>(q) * detail::HalfAt(blk);
  }
  /// Element * scale with the scale folded into the block scale first, the order embedding-scaling callers
  /// (Gemma's sqrt(hidden)) used before the shared decode, so their output stays bit-identical.
  MBUFO_QT_FN static float ScaledElement(const std::uint8_t* row, std::uint32_t i, float scale) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 18;
    const std::uint32_t e = i % 32;
    const std::uint8_t b = blk[2 + (e & 15u)];
    const int q = (e < 16u ? (b & 0xF) : (b >> 4)) - 8;
    const float d = detail::HalfAt(blk) * scale;
    return d * static_cast<float>(q);
  }
};

template <> struct Traits<FormatId::kQ8_0> {
  static constexpr FormatId kId = FormatId::kQ8_0;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 34;
  static constexpr DecodePath kDecode = DecodePath::kNarrowQ8;
  static constexpr PrefillPath kPrefill = PrefillPath::kDenseQ8;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 32 * 34; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 32 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 34;
    return static_cast<float>(static_cast<std::int8_t>(blk[2 + (i % 32)])) * detail::HalfAt(blk);
  }
  /// See Traits<kQ4_0>::ScaledElement.
  MBUFO_QT_FN static float ScaledElement(const std::uint8_t* row, std::uint32_t i, float scale) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 34;
    const float d = detail::HalfAt(blk) * scale;
    return d * static_cast<float>(static_cast<std::int8_t>(blk[2 + (i % 32)]));
  }
};

template <> struct Traits<FormatId::kQ4_K> {
  static constexpr FormatId kId = FormatId::kQ4_K;
  static constexpr unsigned kBlockElems = 256, kBlockBytes = 144;
  static constexpr DecodePath kDecode = DecodePath::kBlockGemv;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 256 * 144; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 256 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 256) * 144;
    const std::uint32_t e = i % 256;
    const std::uint32_t g = e / 64u, r = e % 64u, l = r % 32u;
    const std::uint32_t hi = r >= 32u ? 1u : 0u;
    float sc, mn;
    detail::ScaleMinK4(static_cast<int>(2u * g + hi), blk + 4, &sc, &mn);
    const std::uint8_t b = blk[16 + g * 32u + l];
    const float q = static_cast<float>(hi ? (b >> 4) : (b & 0xF));
    return detail::HalfAt(blk) * sc * q - detail::HalfAt(blk + 2) * mn;
  }
};

template <> struct Traits<FormatId::kQ5_K> {
  static constexpr FormatId kId = FormatId::kQ5_K;
  static constexpr unsigned kBlockElems = 256, kBlockBytes = 176;
  static constexpr DecodePath kDecode = DecodePath::kBlockGemv;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 256 * 176; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 256 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 256) * 176;
    const std::uint32_t e = i % 256;
    const std::uint32_t g = e / 64u, r = e % 64u, l = r % 32u;
    const std::uint32_t hi = r >= 32u ? 1u : 0u;
    const std::uint32_t j = 2u * g + hi;
    float sc, mn;
    detail::ScaleMinK4(static_cast<int>(j), blk + 4, &sc, &mn);
    const std::uint8_t b = blk[48 + g * 32u + l];
    const float q = static_cast<float>((hi ? (b >> 4) : (b & 0xF)) + (((blk[16 + l] >> j) & 1) ? 16 : 0));
    return detail::HalfAt(blk) * sc * q - detail::HalfAt(blk + 2) * mn;
  }
};

template <> struct Traits<FormatId::kQ6_K> {
  static constexpr FormatId kId = FormatId::kQ6_K;
  static constexpr unsigned kBlockElems = 256, kBlockBytes = 210;
  static constexpr DecodePath kDecode = DecodePath::kNone;  // no block GEMV: dequantize + dense
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 256 * 210; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 256 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 256) * 210;
    const std::uint32_t e = i % 256;
    const std::uint32_t n = e / 128u, r = e % 128u, l = r % 32u, q = r / 32u;
    const std::uint8_t lo = blk[n * 64u + l + (q & 1u) * 32u];
    const std::uint8_t hb = blk[128 + n * 32u + l];
    const int v = ((q >= 2u ? (lo >> 4) : (lo & 0xF)) | (((hb >> (2u * q)) & 3) << 4)) - 32;
    const float sc = static_cast<float>(static_cast<std::int8_t>(blk[192 + n * 8u + l / 16u + 2u * q]));
    return detail::HalfAt(blk + 208) * sc * static_cast<float>(v);
  }
};

// ---- formats added so every packed format a shipped engine reads is registered (decode through the
// shared row-dot layer, dequantize + dense GEMM for prefill; no block-GEMV decode kernel of their own) ----

namespace detail {
MBUFO_QT_FN std::uint32_t U32At(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
// ROCmFP / unsigned E4M3 block scale (core/quant/ggml_dequant.cpp Ue4m3ScaleToFloat).
MBUFO_QT_FN float Ue4m3Scale(std::uint8_t e) {
  if (e > 0x7EU) return 0.0F;
  const int exp = e >> 3;
  const int mant = e & 0x07;
  if (exp == 0) return static_cast<float>(mant) * 0x1p-10f;
  return static_cast<float>(8 + mant) * ldexpf(1.0F, exp - 11);
}
MBUFO_QT_FN float RocmFp4Code(std::uint32_t code) {
  constexpr int kCodebook[16] = {0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10};
  return static_cast<float>(kCodebook[code & 15u]);
}
}  // namespace detail

/// Q4_1: fp16 scale + fp16 min + 16 nibble bytes; value = d * q + m.
template <> struct Traits<FormatId::kQ4_1> {
  static constexpr FormatId kId = FormatId::kQ4_1;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 20;
  static constexpr DecodePath kDecode = DecodePath::kNone;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 32 * 20; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 32 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 20;
    const std::uint32_t e = i % 32;
    const std::uint8_t b = blk[4 + (e & 15u)];
    const float q = static_cast<float>(e < 16u ? (b & 0xF) : (b >> 4));
    return detail::HalfAt(blk) * q + detail::HalfAt(blk + 2);
  }
};

/// Q5_0: fp16 scale + 4 high-bit bytes + 16 nibble bytes; value = d * (q5 - 16).
template <> struct Traits<FormatId::kQ5_0> {
  static constexpr FormatId kId = FormatId::kQ5_0;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 22;
  static constexpr DecodePath kDecode = DecodePath::kNone;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 32 * 22; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 32 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 22;
    const std::uint32_t e = i % 32, j = e & 15u;
    const std::uint32_t qh = detail::U32At(blk + 2);
    const std::uint8_t b = blk[6 + j];
    const std::uint32_t q5 = (e < 16u ? (b & 0xFu) : (b >> 4)) | (((qh >> e) & 1u) << 4);
    return detail::HalfAt(blk) * static_cast<float>(static_cast<int>(q5) - 16);
  }
};

/// Q5_1: fp16 scale + fp16 min + 4 high-bit bytes + 16 nibble bytes; value = d * q5 + m.
template <> struct Traits<FormatId::kQ5_1> {
  static constexpr FormatId kId = FormatId::kQ5_1;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 24;
  static constexpr DecodePath kDecode = DecodePath::kNone;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 32 * 24; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 32 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 24;
    const std::uint32_t e = i % 32, j = e & 15u;
    const std::uint32_t qh = detail::U32At(blk + 4);
    const std::uint8_t b = blk[8 + j];
    const std::uint32_t q5 = (e < 16u ? (b & 0xFu) : (b >> 4)) | (((qh >> e) & 1u) << 4);
    return detail::HalfAt(blk) * static_cast<float>(q5) + detail::HalfAt(blk + 2);
  }
};

/// Q2_K: 16 sub-block scale bytes (low nibble scale, high nibble min), 64 bytes of 2-bit codes, fp16 d and dmin.
template <> struct Traits<FormatId::kQ2_K> {
  static constexpr FormatId kId = FormatId::kQ2_K;
  static constexpr unsigned kBlockElems = 256, kBlockBytes = 84;
  static constexpr DecodePath kDecode = DecodePath::kNone;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 256 * 84; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 256 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 256) * 84;
    const std::uint32_t e = i % 256;
    const std::uint32_t n = e / 128u, r = e % 128u, j = r / 32u, l = r % 32u;  // n: 128-half, j: shift step
    const std::uint32_t second = l / 16u, lo = l % 16u;
    const std::uint8_t sc = blk[n * 8u + 2u * j + second];
    const std::uint8_t q = blk[16 + n * 32u + second * 16u + lo];
    const float dl = detail::HalfAt(blk + 80) * static_cast<float>(sc & 0xFu);
    const float ml = detail::HalfAt(blk + 82) * static_cast<float>(sc >> 4);
    return dl * static_cast<float>((q >> (2u * j)) & 3u) - ml;
  }
};

/// Q3_K: 32 bytes of high bits, 64 bytes of 2-bit codes, 12 bytes of packed 6-bit scales (16 sub-blocks of 16), fp16 d.
/// value = d * (scale - 32) * ((code) - (high bit set ? 0 : 4)).
template <> struct Traits<FormatId::kQ3_K> {
  static constexpr FormatId kId = FormatId::kQ3_K;
  static constexpr unsigned kBlockElems = 256, kBlockBytes = 110;
  static constexpr DecodePath kDecode = DecodePath::kNone;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 256 * 110; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 256 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 256) * 110;
    const std::uint32_t e = i % 256;
    const std::uint32_t n = e / 128u, r = e % 128u, j = r / 32u, l32 = r % 32u;
    const std::uint32_t second = l32 / 16u, l = l32 % 16u;
    const std::uint32_t k = n * 8u + j * 2u + second;  // sub-block 0..15
    const std::uint8_t* sc = blk + 96;
    const std::uint32_t low = k < 8u ? (sc[k] & 0xFu) : (sc[k - 8u] >> 4);
    const std::uint32_t high = (sc[8u + (k & 3u)] >> (2u * (k / 4u))) & 3u;
    const float dl = detail::HalfAt(blk + 108) * static_cast<float>(static_cast<int>(low | (high << 4)) - 32);
    const std::uint32_t q = (blk[32 + n * 32u + second * 16u + l] >> (2u * j)) & 3u;
    const bool hb = (blk[second * 16u + l] & (1u << (n * 4u + j))) != 0;
    return dl * static_cast<float>(static_cast<int>(q) - (hb ? 0 : 4));
  }
};

namespace detail {
// Non-linear 4-bit codebook shared by IQ4_NL and IQ4_XS.
MBUFO_QT_FN float Iq4Value(std::uint32_t nibble) {
  constexpr float kValues[16] = {-127.0F, -104.0F, -83.0F, -65.0F, -49.0F, -35.0F, -22.0F, -10.0F,
                                 1.0F,    13.0F,   25.0F,  38.0F,  53.0F,  69.0F,  89.0F,  113.0F};
  return kValues[nibble & 15u];
}
}  // namespace detail

/// IQ4_NL: fp16 d + 16 bytes of 4-bit indices into the non-linear codebook, 18 bytes per 32 (low nibbles first).
template <> struct Traits<FormatId::kIQ4_NL> {
  static constexpr FormatId kId = FormatId::kIQ4_NL;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 18;
  static constexpr DecodePath kDecode = DecodePath::kNone;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 32 * 18; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 32 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 18;
    const std::uint32_t e = i % 32;
    const std::uint8_t b = blk[2 + (e & 15u)];
    return detail::HalfAt(blk) * detail::Iq4Value(e < 16u ? (b & 0xFu) : (b >> 4));
  }
};

/// IQ4_XS: fp16 d, 16-bit high scale bits, 4 bytes of low scale nibbles (8 sub-blocks of 32), 128 bytes of 4-bit indices,
/// 136 bytes per 256. value = d * (scale - 32) * codebook[nibble].
template <> struct Traits<FormatId::kIQ4_XS> {
  static constexpr FormatId kId = FormatId::kIQ4_XS;
  static constexpr unsigned kBlockElems = 256, kBlockBytes = 136;
  static constexpr DecodePath kDecode = DecodePath::kNone;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 256 * 136; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 256 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 256) * 136;
    const std::uint32_t e = i % 256;
    const std::uint32_t ib = e / 32u, j = e % 32u;
    const std::uint32_t scales_h = static_cast<std::uint32_t>(blk[2]) | (static_cast<std::uint32_t>(blk[3]) << 8);
    const std::uint32_t ls = ((blk[4 + ib / 2u] >> (4u * (ib % 2u))) & 0xFu) | (((scales_h >> (2u * ib)) & 3u) << 4);
    const float dl = detail::HalfAt(blk) * static_cast<float>(static_cast<int>(ls) - 32);
    const std::uint8_t b = blk[8 + ib * 16u + (j & 15u)];
    return dl * detail::Iq4Value(j < 16u ? (b & 0xFu) : (b >> 4));
  }
};

/// MXFP4 (OCP microscaling, GGML type 39): one e8m0 scale byte + 16 e2m1 nibble bytes (low nibbles first).
/// The codebook is halved ({0,.5,1,1.5,2,3,4,6}) against scale 2^(e-127), which is exactly the engines'
/// integer codebook {0,1,2,3,4,6,8,12} against 2^(e-128).
template <> struct Traits<FormatId::kMXFP4> {
  static constexpr FormatId kId = FormatId::kMXFP4;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 17;
  static constexpr DecodePath kDecode = DecodePath::kNone;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 32 * 17; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 32 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    constexpr float kCodebook[16] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
                                     0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 17;
    const std::uint32_t e = i % 32;
    const std::uint8_t b = blk[1 + (e & 15u)];
    return kCodebook[e < 16u ? (b & 0xFu) : (b >> 4)] * exp2f(static_cast<float>(static_cast<int>(blk[0]) - 127));
  }
};

/// ROCmFP4: AMD E2M1-derived codebook with dual UE4M3 scales (low half / high half), 18 bytes per 32.
template <> struct Traits<FormatId::kQ4_0_ROCMFP4> {
  static constexpr FormatId kId = FormatId::kQ4_0_ROCMFP4;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 18;
  static constexpr DecodePath kDecode = DecodePath::kNone;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 32 * 18; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 32 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 18;
    const std::uint32_t e = i % 32;
    const std::uint8_t b = blk[e & 15u];
    const std::uint32_t hi = e >= 16u ? 1u : 0u;
    return detail::RocmFp4Code(hi ? (b >> 4) : (b & 0xFu)) * detail::Ue4m3Scale(blk[16 + hi]);
  }
};

/// ROCmFP4 FAST: same nibbles, one UE4M3 scale per 32 elements, 17 bytes.
template <> struct Traits<FormatId::kQ4_0_ROCMFP4_FAST> {
  static constexpr FormatId kId = FormatId::kQ4_0_ROCMFP4_FAST;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 17;
  static constexpr DecodePath kDecode = DecodePath::kNone;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 32 * 17; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 32 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 17;
    const std::uint32_t e = i % 32;
    const std::uint8_t b = blk[e & 15u];
    return detail::RocmFp4Code(e >= 16u ? (b >> 4) : (b & 0xFu)) * detail::Ue4m3Scale(blk[16]);
  }
};

/// ROCmFP3: 3-bit codes {0,1,2,4,-0,-1,-2,-4}, eight per 3 bytes little-endian, dual UE4M3 scales, 14 bytes per 32.
template <> struct Traits<FormatId::kQ3_0_ROCMFPX> {
  static constexpr FormatId kId = FormatId::kQ3_0_ROCMFPX;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 14;
  static constexpr DecodePath kDecode = DecodePath::kNone;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 32 * 14; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 32 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 14;
    const std::uint32_t e = i % 32, g = e / 8u, k = e % 8u;
    const std::uint8_t* src = blk + g * 3u;
    const std::uint32_t bitpos = 3u * k, byte = bitpos / 8u, sh = bitpos % 8u;
    std::uint32_t code = static_cast<std::uint32_t>(src[byte]) >> sh;
    if (sh > 5u) code |= static_cast<std::uint32_t>(src[byte + 1u]) << (8u - sh);
    code &= 7u;
    constexpr int kMag[4] = {0, 1, 2, 4};
    const float mag = static_cast<float>(kMag[code & 3u]);
    return ((code & 4u) ? -mag : mag) * detail::Ue4m3Scale(blk[12 + (e >= 16u ? 1u : 0u)]);
  }
};

/// IQ2_XXS: fp16 scale + 32 uint16 (8 sub-blocks of 4 grid-index bytes plus a 32-bit word of four 7-bit sign
/// selectors and a 4-bit scale); every group of 8 weights is a row of the 256-entry grid with signs from the
/// 128-entry table. Same arithmetic order as ggml dequantize_row_iq2_xxs.
template <> struct Traits<FormatId::kIQ2_XXS> {
  static constexpr FormatId kId = FormatId::kIQ2_XXS;
  static constexpr unsigned kBlockElems = 256, kBlockBytes = 66;
  static constexpr DecodePath kDecode = DecodePath::kNone;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 256 * 66; }
  MBUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 256 == 0; }
  MBUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 256) * 66;
    const std::uint32_t e = i % 256, ib32 = e / 32u, r = e % 32u, l = r / 8u, j = r % 8u;
    const std::uint8_t* qs = blk + 2 + ib32 * 8u;
    const std::uint32_t aux1 = detail::U32At(qs + 4);
    const float db = detail::HalfAt(blk) * (0.5F + static_cast<float>(aux1 >> 28)) * 0.25F;
    const std::uint64_t grid = detail::kIq2xxsGrid[qs[l]];
    const float g = static_cast<float>(static_cast<std::uint32_t>((grid >> (8u * j)) & 0xFFu));
    const bool neg = (detail::kIq2Signs[(aux1 >> (7u * l)) & 127u] & (1u << j)) != 0u;
    return db * g * (neg ? -1.0F : 1.0F);
  }
};

/// What every format must provide. Adding a format to AllFormats without these fails to compile.
template <class T>
concept QuantFormat = requires(const std::uint8_t* row, std::size_t n, std::uint32_t i) {
  { T::kId } -> std::convertible_to<FormatId>;
  { T::kBlockElems } -> std::convertible_to<unsigned>;
  { T::kBlockBytes } -> std::convertible_to<unsigned>;
  { T::kDecode } -> std::convertible_to<DecodePath>;
  { T::kPrefill } -> std::convertible_to<PrefillPath>;
  { T::RowBytes(n) } -> std::same_as<std::size_t>;
  { T::ColsOk(n) } -> std::same_as<bool>;
  { T::Element(row, i) } -> std::same_as<float>;
};

/// Element(row, i) * scale, using the format's own ScaledElement when it declares one (so the scale is
/// folded in the same order its callers always used) and Element * scale otherwise.
template <class F>
MBUFO_QT_FN float ScaledElementOf(const std::uint8_t* row, std::uint32_t i, float scale) {
  if constexpr (requires { { F::ScaledElement(row, i, scale) } -> std::same_as<float>; })
    return F::ScaledElement(row, i, scale);
  else
    return F::Element(row, i) * scale;
}

template <class... Fs>
struct FormatList {
  static constexpr bool kAllValid = (QuantFormat<Fs> && ...);
  static constexpr bool kDistinctIds = [] {
    const int ids[] = {static_cast<int>(Fs::kId)...};
    for (std::size_t a = 0; a < sizeof...(Fs); ++a)
      for (std::size_t b = a + 1; b < sizeof...(Fs); ++b)
        if (ids[a] == ids[b]) return false;
    return true;
  }();
  /// Calls f(Traits{}) for the format with this type code. Returns false (f not called) when the
  /// code is not in the list: there is deliberately no default format.
  template <class F>
  static bool Dispatch(int type, F&& f) {
    return (TryOne<Fs>(type, f) || ...);
  }

 private:
  template <class Fm, class F>
  static bool TryOne(int type, F& f) {
    if (type != static_cast<int>(Fm::kId)) return false;
    f(Fm{});
    return true;
  }
};

using AllFormats =
    FormatList<Traits<FormatId::kF32>, Traits<FormatId::kF16>, Traits<FormatId::kBF16>, Traits<FormatId::kQ4_0>,
               Traits<FormatId::kQ4_1>, Traits<FormatId::kQ5_0>, Traits<FormatId::kQ5_1>, Traits<FormatId::kQ8_0>,
               Traits<FormatId::kQ2_K>, Traits<FormatId::kQ3_K>, Traits<FormatId::kIQ4_NL>, Traits<FormatId::kIQ4_XS>,
               Traits<FormatId::kIQ2_XXS>, Traits<FormatId::kQ4_K>, Traits<FormatId::kQ5_K>, Traits<FormatId::kQ6_K>,
               Traits<FormatId::kMXFP4>, Traits<FormatId::kQ4_0_ROCMFP4>, Traits<FormatId::kQ4_0_ROCMFP4_FAST>,
               Traits<FormatId::kQ3_0_ROCMFPX>>;
static_assert(AllFormats::kAllValid, "every listed format must satisfy QuantFormat");
static_assert(AllFormats::kDistinctIds, "format type codes must be unique");

}  // namespace mbufo::core::quant

#endif  // ENGINE_ENABLE_HIP
#endif  // MBUFO_CORE_QUANT_QUANT_TRAITS_HPP_
