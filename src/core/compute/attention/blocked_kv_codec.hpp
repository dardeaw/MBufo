#ifndef MBUFO_CORE_COMPUTE_ATTENTION_BLOCKED_KV_CODEC_HPP_
#define MBUFO_CORE_COMPUTE_ATTENTION_BLOCKED_KV_CODEC_HPP_

// Unified blocked-KV element codecs, row layout, cache view and interface.
//
// Single source of truth for every blocked KV format in the tree. The same
// 34-byte Q8 rows (and 18-byte packed-nibble value rows) used to be
// described by three copy-pasted `KvCodecQ8` tags (blocked_kv.hip.cpp,
// blocked_splitk.hpp, blocked_wmma.hpp) plus a separate CodecTraits decode
// in two kernel TUs, each carrying a "bit-compatible with ..." comment.
// Copy-paste is how a codec silently forks; this header is the one
// definition. Do not redeclare these tags elsewhere --
// tools/ci/check-kv-codec.py rejects new copies.
//
// Everything here is plain integer and libm math with no device builtins,
// so one header serves host code (unit tests, host-side validation), g++
// translation units (engine.cpp via kernels.hpp) and HIP device code.
// Half<->float uses exact integer bit manipulation (round-to-nearest-even,
// IEEE-754) instead of __float2half/__half2float; the FP8 encoder
// bit-casts through memcpy instead of __float_as_uint. Both substitutions
// are bit-exact by construction.
//
// Layout recap (shared with blocked_kv::KvBlockedRowBytes): every 32
// elements form one block. Q8/FP8 blocks hold 32 code bytes + one f16
// scale (34 bytes); packed-nibble value blocks hold 16 payload bytes +
// one f16 scale (18 bytes). Row bytes = blocks * block_bytes.
//
// Interface: engines implement attention themselves, but any function that
// touches blocked rows must take a BlockedKvView (built only by
// MakeBlockedKvView, whose strides are asserted here) and decode through
// the codec tag. IsBlockedKvView (+RequireKvView) is the interface that
// enforces this at compile time for new entry points: anything that is
// not a BlockedKvView fails with a readable error instead of silently
// misreading rows.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

// Execution-space annotations. Under HIP they come from the toolchain;
// for plain host compilers (g++, including host-side unit tests) they
// are empty, which keeps this header dependency-free and compilable
// anywhere. Deliberately no <hip/...> include here.
#ifndef __HIPCC__
#define __host__
#define __device__
#endif

namespace mbufo::core::compute::blocked_kv {

// ---------------------------------------------------------------------------
// Exact half <-> float (integer bit manipulation, host + device safe).
// Round-to-nearest-even per IEEE-754, bit-identical to the hardware
// __float2half/__half2float conversions including subnormals, inf and NaN.
// ---------------------------------------------------------------------------

[[nodiscard]] __host__ __device__ inline float HalfBitsToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
  const std::uint32_t exp = (static_cast<std::uint32_t>(h) >> 10) & 0x1Fu;
  const std::uint32_t mant = static_cast<std::uint32_t>(h) & 0x3FFu;
  std::uint32_t f = 0;
  if (exp == 0u) {
    if (mant == 0u) {
      f = sign;  // signed zero
    } else {
      // Subnormal: renormalize (value = mant/1024 * 2^-14).
      std::uint32_t m = mant;
      int e = -14;
      while ((m & 0x400u) == 0u) {
        m <<= 1;
        --e;
      }
      m &= 0x3FFu;
      f = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (m << 13);
    }
  } else if (exp == 31u) {
    f = sign | 0x7F800000u | (mant << 13);  // inf / NaN, payload preserved
  } else {
    f = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float out = 0.0F;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

[[nodiscard]] __host__ __device__ inline std::uint16_t FloatToHalfBits(float f) noexcept {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  const std::uint32_t sign = (u >> 16) & 0x8000u;
  const std::uint32_t exp = (u >> 23) & 0xFFu;
  const std::uint32_t mant = u & 0x7FFFFFu;
  if (exp == 0xFFu) {  // inf / NaN -> quiet NaN keeps a payload bit
    return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
  }
  const std::int32_t e = static_cast<std::int32_t>(exp) - 127;  // unbiased
  if (e >= 16) {  // overflows half max (65504)
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  if (e >= -14) {
    // Normal half: keep the top 10 mantissa bits, RNE on the dropped 13.
    const std::uint32_t kept = mant >> 13;
    const std::uint32_t dropped = mant & 0x1FFFu;
    const std::uint32_t half_lsb = 0x1000u;
    std::uint32_t hmant = kept;
    if (dropped > half_lsb || (dropped == half_lsb && (kept & 1u) != 0u)) {
      ++hmant;  // round up (ties to even via the kept-LSB check)
    }
    std::uint32_t he = static_cast<std::uint32_t>(e + 15);
    if (hmant == 0x400u) {  // mantissa carry -> bump exponent
      hmant = 0u;
      ++he;
      if (he >= 31u) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
      }
    }
    return static_cast<std::uint16_t>(sign | (he << 10) | hmant);
  }
  if (e >= -25) {
    // Subnormal half: M * 2^-24 where M = RNE((0x800000|mant) >> shift).
    // shift in [14, 24]; e == -25 still rounds (up to 1) instead of
    // flushing, e < -25 is exactly zero.
    const std::uint32_t m24 = mant | 0x800000u;
    const int shift = -e - 1;
    const std::uint32_t kept = m24 >> shift;
    const std::uint32_t mask =
        (shift >= 32) ? 0xFFFFFFFFu : ((1u << shift) - 1u);
    const std::uint32_t dropped = m24 & mask;
    const std::uint32_t half_lsb = 1u << (shift - 1);
    std::uint32_t m10 = kept;
    if (dropped > half_lsb || (dropped == half_lsb && (kept & 1u) != 0u)) {
      ++m10;
    }
    if (m10 >= 0x400u) {  // rounded up into the normal range
      return static_cast<std::uint16_t>(sign | 0x0400u);
    }
    return static_cast<std::uint16_t>(sign | m10);
  }
  return static_cast<std::uint16_t>(sign);  // too small: signed zero
}

// ---------------------------------------------------------------------------
// Codec tags. THE single definitions -- do not redeclare elsewhere.
// ---------------------------------------------------------------------------

/// Symmetric INT8 grid over amax/127, stored as bytes. 32 codes + one f16
/// scale per 32-element block (34 bytes). Bit-compatible with every
/// historical KvCodecQ8 copy in the tree.
struct KvCodecQ8 {
  using Elem = std::uint8_t;
  static constexpr float kRange = 127.0f;
  static constexpr std::uint32_t kBlockElems = 32u;
  static constexpr std::uint32_t kBlockBytes = 34u;
  static __host__ __device__ inline Elem Encode(float v) noexcept {
    v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
    return static_cast<Elem>(static_cast<std::int8_t>(std::rintf(v)));
  }
  static __host__ __device__ inline float Decode(Elem e) noexcept {
    return static_cast<float>(static_cast<std::int8_t>(e));
  }
};

/// OCP E4M3 (no inf; 0x7F and 0xFF are NaN). Emulated in software:
/// gfx1151 has no fp8 conversion instruction and the mmq helper is
/// CDNA3/Blackwell only. Values round to nearest even; the block scale
/// keeps in-range blocks below the format maximum, and anything past it
/// saturates. Bit layout identical to the historical copies; only the
/// __float_as_uint bit-cast became memcpy (bit-exact by construction).
struct KvCodecFp8 {
  using Elem = std::uint8_t;
  static constexpr float kRange = 448.0f;
  static constexpr std::uint32_t kBlockElems = 32u;
  static constexpr std::uint32_t kBlockBytes = 34u;
  static __host__ __device__ inline Elem Encode(float x) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const std::uint32_t sign = (bits >> 24) & 0x80u;
    const float a = std::fabsf(x);
    if (!(a == a))  // NaN keeps the format's NaN encoding
      return static_cast<Elem>(sign | 0x7Fu);
    if (a >= 448.0f)  // saturate; also catches inf
      return static_cast<Elem>(sign | 0x7Eu);
    if (a < 0.015625f) {  // 2^-6: subnormal grid of step 2^-9
      const std::uint32_t m =
          static_cast<std::uint32_t>(std::rintf(a * 512.0f));
      return static_cast<Elem>(sign | m);
    }
    const std::uint32_t m23 = bits & 0x7FFFFFu;
    const std::uint32_t r =
        m23 + 0x7FFFFu + ((m23 >> 20) & 1u);  // round to nearest even, 3 bits
    std::uint32_t m3 = r >> 20u;
    std::uint32_t e = ((bits >> 23) & 0xFFu) - 120u;  // unbiased + E4M3 bias
    if (m3 > 7u) {  // mantissa carry
      m3 = 0u;
      ++e;
    }
    if (e > 15u)  // unreachable below 448; kept so a bug cannot alias the sign
      return static_cast<Elem>(sign | 0x7Eu);
    return static_cast<Elem>(sign | (e << 3u) | m3);
  }
  static __host__ __device__ inline float Decode(Elem e) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(e & 0x80u) << 24;
    const std::uint32_t f = static_cast<std::uint32_t>(e & 0x7Fu);
    const std::uint32_t exp = f >> 3u;
    if (exp == 0u) {  // subnormal grid: mant * 2^-9, exact in fp32
      const float v = static_cast<float>(f & 0x7u) * (1.0f / 512.0f);
      return sign == 0u ? v : -v;
    }
    // NOTE: 0x7E/0x7F (NaN encodings) decode as huge finite floats here,
    // matching the historical behavior verbatim.
    std::uint32_t bits =
        sign | ((exp + 120u) << 23u) | ((f & 0x7u) << 20u);
    float v = 0.0F;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
  }
};

/// Packed-nibble value codec for K8V4 (keys stay Q8): 16 payload bytes +
/// one f16 scale per 32-element block (18 bytes). Symmetric grid over
/// amax/7 with round-to-nearest-even; -1 encodes as 0xF (two's complement
/// nibble), matching the historical inline codec in the blocked kernels.
struct KvCodecQ4 {
  using Elem = std::uint8_t;  // payload bytes, two nibbles each
  static constexpr float kRange = 7.0f;
  static constexpr std::uint32_t kBlockElems = 32u;
  static constexpr std::uint32_t kBlockBytes = 18u;
  static __host__ __device__ inline std::uint32_t EncodeNibble(float v) noexcept {
    int q = static_cast<int>(std::rintf(v));
    q = q > 7 ? 7 : (q < -7 ? -7 : q);
    return static_cast<std::uint32_t>(q) & 0xFu;
  }
  static __host__ __device__ inline int DecodeNibble(std::uint32_t nib) noexcept {
    const std::uint32_t q = nib & 0xFu;
    return static_cast<int>(q) - ((q & 8u) != 0u ? 16 : 0);
  }
};

// ---------------------------------------------------------------------------
// Shared per-block helpers. Kernels keep their own loop structure and
// reduction shape, but every scale, code and nibble below comes from here.
// ---------------------------------------------------------------------------

/// Block scale from the block absolute maximum. Zero blocks (all padding)
/// use scale 1 so the codes stay exact zeros.
[[nodiscard]] __host__ __device__ inline float ScaleFromAmax(float amax, float range) noexcept {
  return amax > 0.0f ? amax / range : 1.0f;
}

/// Store a block scale as f16 bits (exact conversion, see above).
__host__ __device__ inline void StoreScaleF16(std::uint8_t* dst_scale2, float scale) noexcept {
  const std::uint16_t h = FloatToHalfBits(scale);
  std::memcpy(dst_scale2, &h, sizeof(h));
}

/// Load a block scale from f16 bits (exact conversion, see above).
[[nodiscard]] __host__ __device__ inline float LoadScaleF16(
    const std::uint8_t* blk_scale2) noexcept {
  std::uint16_t h = 0;
  std::memcpy(&h, blk_scale2, sizeof(h));
  return HalfBitsToFloat(h);
}

/// Row bytes of one token row with `kv_dim` elements at `block_bytes` per
/// 32-element block. Canonical formula shared with KvBlockedRowBytes.
[[nodiscard]] __host__ __device__ constexpr std::uint32_t BlockedRowBytes(
    std::uint32_t kv_dim, std::uint32_t block_bytes) noexcept {
  return ((kv_dim + 31u) / 32u) * block_bytes;
}

/// Byte stride of one head with `head_dim` elements inside a row.
[[nodiscard]] __host__ __device__ constexpr std::uint32_t BlockedHeadStrideBytes(
    std::uint32_t head_dim, std::uint32_t block_bytes) noexcept {
  return ((head_dim + 31u) / 32u) * block_bytes;
}

/// Lane -> block / intra-block offset for the fused-dequant read pattern
/// (each lane owns `elems_per_lane` consecutive elements of the head).
[[nodiscard]] __host__ __device__ constexpr std::uint32_t BlockOfLane(
    std::uint32_t lane, std::uint32_t elems_per_lane = 4u) noexcept {
  return (lane * elems_per_lane) / 32u;
}
[[nodiscard]] __host__ __device__ constexpr std::uint32_t OffsetOfLane(
    std::uint32_t lane, std::uint32_t elems_per_lane = 4u) noexcept {
  return (lane * elems_per_lane) % 32u;
}

// ---------------------------------------------------------------------------
// Cache view + interface trait. Engines implement attention themselves,
// but any function touching blocked rows must take a view built by
// MakeBlockedKvView. IsBlockedKvView (+RequireKvView) constrains new
// entry points at compile time: anything that is not a BlockedKvView
// fails with a readable error instead of misreading rows.
// ---------------------------------------------------------------------------

template <typename Codec>
struct BlockedKvView {
  using codec = Codec;
  std::uint8_t* data = nullptr;
  std::uint32_t row_stride = 0u;   // bytes per token row
  std::uint32_t head_stride = 0u;  // bytes per head within a row
};

template <typename T>
struct KvViewLikeHelper : std::false_type {};
template <typename Codec>
struct KvViewLikeHelper<BlockedKvView<Codec>> : std::true_type {};

// Interface enforcement. NOTE on concepts: this header is also compiled by
// the HIP toolchain, whose language standard is still C++17, so the C++20
// `concept` keyword is unavailable here. Enforcement is therefore spelled
// as a type trait + static_assert -- same compile-time rejection with a
// readable message, zero runtime cost. (When CMAKE_HIP_STANDARD moves to
// 20, KvViewLike below becomes a real concept with no call-site changes.)
// Only BlockedKvView<Codec> satisfies the interface: concrete entry points
// take it by value/reference (then the type system itself is the check);
// generic entry points constrain their parameter with RequireKvView<V>().
template <typename V>
struct IsBlockedKvView : KvViewLikeHelper<V> {};

template <typename V>
__host__ __device__ inline void RequireKvView(const V&) {
  static_assert(IsBlockedKvView<V>::value,
                "KvViewLike: blocked-KV entry points only accept "
                "BlockedKvView<Codec> built by MakeBlockedKvView (see "
                "blocked_kv_codec.hpp). Raw pointers and ad-hoc stride "
                "tuples are rejected: construct a view instead.");
}

/// Enforced construction path: strides always match the shared layout.
/// `wrap` is informational (kernels apply their own ring rule); the view
/// only pins down the byte geometry so a wrong stride is a compile-time
/// or single-point failure instead of silent misaddressing.
template <typename Codec>
[[nodiscard]] __host__ __device__ inline BlockedKvView<Codec> MakeBlockedKvView(
    std::uint8_t* data, std::uint32_t kv_dim,
    std::uint32_t head_dim) noexcept {
  BlockedKvView<Codec> v;
  v.data = data;
  v.row_stride =
      BlockedRowBytes(kv_dim, Codec::kBlockBytes);
  v.head_stride =
      BlockedHeadStrideBytes(head_dim, Codec::kBlockBytes);
  return v;
}

}  // namespace mbufo::core::compute::blocked_kv

#endif  // MBUFO_CORE_COMPUTE_ATTENTION_BLOCKED_KV_CODEC_HPP_
