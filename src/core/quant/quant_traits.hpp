#ifndef GUFO_CORE_QUANT_QUANT_TRAITS_HPP_
#define GUFO_CORE_QUANT_QUANT_TRAITS_HPP_

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

#include "src/core/quant/quant_status.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>

#define GUFO_QT_FN __host__ __device__ __forceinline__

namespace gufo::core::quant {

/// GGML type codes (same values as core::GgmlType / the GUFO_GEMM_* codes).
enum class FormatId : int {
  kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ8_0 = 8, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14, kBF16 = 30,
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
GUFO_QT_FN float HalfAt(const std::uint8_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}
GUFO_QT_FN void ScaleMinK4(int j, const std::uint8_t* q, float* sc, float* mn) {
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
  GUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols * 4; }
  GUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0; }
  GUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    return reinterpret_cast<const float*>(row)[i];
  }
};

template <> struct Traits<FormatId::kF16> {
  static constexpr FormatId kId = FormatId::kF16;
  static constexpr unsigned kBlockElems = 1, kBlockBytes = 2;
  static constexpr DecodePath kDecode = DecodePath::kLegacyGemv;
  static constexpr PrefillPath kPrefill = PrefillPath::kPassthroughF16;
  GUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols * 2; }
  GUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0; }
  GUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    return __half2float(reinterpret_cast<const __half*>(row)[i]);
  }
};

template <> struct Traits<FormatId::kBF16> {
  static constexpr FormatId kId = FormatId::kBF16;
  static constexpr unsigned kBlockElems = 1, kBlockBytes = 2;
  static constexpr DecodePath kDecode = DecodePath::kLegacyGemv;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  GUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols * 2; }
  GUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0; }
  GUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    return __builtin_bit_cast(float, static_cast<std::uint32_t>(reinterpret_cast<const std::uint16_t*>(row)[i]) << 16);
  }
};

template <> struct Traits<FormatId::kQ4_0> {
  static constexpr FormatId kId = FormatId::kQ4_0;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 18;
  static constexpr DecodePath kDecode = DecodePath::kLegacyGemv;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  GUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 32 * 18; }
  GUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 32 == 0; }
  GUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 18;
    const std::uint32_t e = i % 32;
    const std::uint8_t b = blk[2 + (e & 15u)];
    const int q = (e < 16u ? (b & 0xF) : (b >> 4)) - 8;
    return static_cast<float>(q) * detail::HalfAt(blk);
  }
};

template <> struct Traits<FormatId::kQ8_0> {
  static constexpr FormatId kId = FormatId::kQ8_0;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 34;
  static constexpr DecodePath kDecode = DecodePath::kNarrowQ8;
  static constexpr PrefillPath kPrefill = PrefillPath::kDenseQ8;
  GUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 32 * 34; }
  GUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 32 == 0; }
  GUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
    const std::uint8_t* blk = row + static_cast<std::size_t>(i / 32) * 34;
    return static_cast<float>(static_cast<std::int8_t>(blk[2 + (i % 32)])) * detail::HalfAt(blk);
  }
};

template <> struct Traits<FormatId::kQ4_K> {
  static constexpr FormatId kId = FormatId::kQ4_K;
  static constexpr unsigned kBlockElems = 256, kBlockBytes = 144;
  static constexpr DecodePath kDecode = DecodePath::kBlockGemv;
  static constexpr PrefillPath kPrefill = PrefillPath::kDequantF16;
  GUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 256 * 144; }
  GUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 256 == 0; }
  GUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
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
  GUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 256 * 176; }
  GUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 256 == 0; }
  GUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
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
  GUFO_QT_FN static std::size_t RowBytes(std::size_t cols) { return cols / 256 * 210; }
  GUFO_QT_FN static bool ColsOk(std::size_t cols) { return cols != 0 && cols % 256 == 0; }
  GUFO_QT_FN static float Element(const std::uint8_t* row, std::uint32_t i) {
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
               Traits<FormatId::kQ8_0>, Traits<FormatId::kQ4_K>, Traits<FormatId::kQ5_K>, Traits<FormatId::kQ6_K>>;
static_assert(AllFormats::kAllValid, "every listed format must satisfy QuantFormat");
static_assert(AllFormats::kDistinctIds, "format type codes must be unique");

}  // namespace gufo::core::quant

#endif  // ENGINE_ENABLE_HIP
#endif  // GUFO_CORE_QUANT_QUANT_TRAITS_HPP_
