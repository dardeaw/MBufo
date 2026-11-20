#ifndef MBUFO_CORE_QUANT_ROW_DOT_TRAITS_HPP_
#define MBUFO_CORE_QUANT_ROW_DOT_TRAITS_HPP_

// Row-dot primitives of the packed weight formats: the quantized inner product every GEMV-like kernel needs,
// kept apart from anything MoE-specific so dense and MoE kernels share one definition per format.
//
// A `RowDot<FormatId>` states how one weight row is dotted with an F32 vector, in one or both reduction
// geometries (concepts check which): DotRow(row, vec, n) -- one thread owns the row and accumulates in
// ascending element order; LaneDot(row, vec, n, lane) -- one wave owns the row, each lane returns its partial and
// the caller reduces across the wave. Both use the CPU-reference dequant values (src/core/quant/ggml_dequant.cpp).
// Kernels (MoE expert GEMV in compute/gemm/moe_decode_*.hip, dense wave32 GEMV through device::WaveRowTileDot)
// depend only on this layer; expert selection, gate activation, layouts, masking and epilogues live above it.
// Adding a format = one RowDot specialization listed in the geometry's list; it must agree with `Traits<kId>`
// on geometry (static_asserted below), and a type code that is not listed is refused at run time, never decoded
// as another format.
//
// HIP translation units only.
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <concepts>
#include <cstddef>
#include <cstdint>

#include "src/core/quant/int_dot.hpp"
#include "src/core/quant/quant_traits.hpp"

namespace mbufo::core::quant {

namespace rowdot_detail {

struct Q4KBlock {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5KBlock {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q5KBlock) == 176, "block_q5_K must be 176 bytes");

struct Q80Block {
  std::uint16_t d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q80Block) == 34, "block_q8_0 must be 34 bytes");
MBUFO_QT_FN float F16Bits(std::uint16_t bits) { return __half2float(__builtin_bit_cast(__half, bits)); }

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));

// MXFP4 integer codebook {0,1,2,3,4,6,8,12} with the top bit as sign, in constant memory (one copy per translation unit).
static __device__ __constant__ std::int8_t kMxfp4Codes[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// A 16-byte-aligned packed block (kBytes % 16 == 0) loaded with wide vector loads instead of per-byte ones;
// the value is a plain copy, so the arithmetic order of whoever reads it is unchanged.
template <class Block>
struct alignas(16) WideBlock {
  static_assert(sizeof(Block) % 16 == 0, "wide loads need a 16-byte multiple");
  v4u_t v[sizeof(Block) / 16];
  MBUFO_QT_FN static WideBlock Load(const std::uint8_t* p) {
    WideBlock w;
    const auto* s = reinterpret_cast<const v4u_t*>(p);
#pragma unroll
    for (unsigned i = 0; i < sizeof(Block) / 16; ++i) w.v[i] = s[i];
    return w;
  }
  MBUFO_QT_FN const Block& get() const { return *reinterpret_cast<const Block*>(v); }
};

// Mirrors GetQ4ScaleMin in src/core/quant/ggml_dequant.cpp.
MBUFO_QT_FN void ScaleMin(const std::uint8_t* scales, int j, std::uint8_t& sc, std::uint8_t& mn) {
  if (j < 4) {
    sc = scales[j] & 0x3FU;
    mn = scales[j + 4] & 0x3FU;
  } else {
    sc = static_cast<std::uint8_t>((scales[j + 4] & 0x0FU) | ((scales[j - 4] >> 6U) << 4U));
    mn = static_cast<std::uint8_t>(((scales[j + 4] >> 4U) & 0x0FU) | ((scales[j] >> 6U) << 4U));
  }
}


// ---- K-quant (Q4_K / Q5_K) register tile decode shared by the WMMA kernels (QuantGemvMulti). Code moved verbatim from
// that kernel so its results stay bit-identical; blocks must be 16-byte aligned (144 / 176 byte blocks). Device-only.
// (QuantGemvSingle keeps its own kernel: moving its hot loop into a function changed the compiler's floating-point
// contraction for Q5_K under every formulation tried, and nothing else needs that loop.)
__device__ __forceinline__ unsigned PackHalves(float a, float b) { return __builtin_bit_cast(unsigned, __floats2half2_rn(a, b)); }

// get_scale_min_k4 for sub-block J (compile time) of the 12 packed scale bytes held in three words.
template <int J>
__device__ __forceinline__ void ScaleMinJ(const unsigned (&sw)[3], float& sc, float& mn) {
  const auto byte = [&](int i) { return (sw[i >> 2] >> (8 * (i & 3))) & 0xFFu; };
  if constexpr (J < 4) {
    sc = static_cast<float>(byte(J) & 63u);
    mn = static_cast<float>(byte(J + 4) & 63u);
  } else {
    sc = static_cast<float>((byte(J + 4) & 0xFu) | ((byte(J - 4) >> 6) << 4));
    mn = static_cast<float>((byte(J + 4) >> 4) | ((byte(J) >> 6) << 4));
  }
}

template <unsigned kBlockBytesV, unsigned kQsOffsetV, bool kHighV>
struct KQuantOps {
  static constexpr unsigned kQsOffset = kQsOffsetV;
  static constexpr bool kHigh = kHighV;

  // Register tile decode for the WMMA kernels: this lane's share of one 256-element superblock (header, its half of every
  // group's qs, its half of qh) and the 16 elements of one 32-element sub-block j as eight packed F16 pairs.
  struct TileRegs {
    uint4 hdr, qs[4], qh;
  };
  __device__ __forceinline__ static void LoadSuper(const std::uint8_t* blk, std::uint32_t half_id, TileRegs& o) {
    o.hdr = *reinterpret_cast<const uint4*>(blk);
    if constexpr (kHighV)
      o.qh = *reinterpret_cast<const uint4*>(blk + 16 + 16 * half_id);
    else
      o.qh = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int g = 0; g < 4; ++g) o.qs[g] = *reinterpret_cast<const uint4*>(blk + kQsOffsetV + g * 32 + 16 * half_id);
  }
  // j is the sub-block (0..7), compile-time after unrolling.
  __device__ __forceinline__ static void DecodeSub16(const TileRegs& r, int j, unsigned (&own)[8]) {
    const float d = __half2float(__builtin_bit_cast(__half, static_cast<std::uint16_t>(r.hdr.x & 0xFFFFu)));
    const float dmin = __half2float(__builtin_bit_cast(__half, static_cast<std::uint16_t>(r.hdr.x >> 16)));
    const unsigned sw[3] = {r.hdr.y, r.hdr.z, r.hdr.w};
    const int g = j >> 1, hi = j & 1;
    const uint4 qs = r.qs[g];
    const uint4 hv = r.qh;
    float sc, mn;
    switch (j) {
      case 0: ScaleMinJ<0>(sw, sc, mn); break;
      case 1: ScaleMinJ<1>(sw, sc, mn); break;
      case 2: ScaleMinJ<2>(sw, sc, mn); break;
      case 3: ScaleMinJ<3>(sw, sc, mn); break;
      case 4: ScaleMinJ<4>(sw, sc, mn); break;
      case 5: ScaleMinJ<5>(sw, sc, mn); break;
      case 6: ScaleMinJ<6>(sw, sc, mn); break;
      default: ScaleMinJ<7>(sw, sc, mn); break;
    }
    const float ds = d * sc;
    const float dm = dmin * mn;
    const unsigned qw[4] = {qs.x, qs.y, qs.z, qs.w};
    const unsigned hw[4] = {hv.x, hv.y, hv.z, hv.w};
#pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
      float v[4];
#pragma unroll
      for (int bi = 0; bi < 4; ++bi) {
        unsigned q = (qw[wi] >> (8 * bi + 4 * hi)) & 0xFu;
        if constexpr (kHighV) q += ((hw[wi] >> (8 * bi + j)) & 1u) << 4;
        v[bi] = ds * static_cast<float>(q) - dm;
      }
      own[2 * wi] = PackHalves(v[0], v[1]);
      own[2 * wi + 1] = PackHalves(v[2], v[3]);
    }
  }
};

// ---- Q6_K register tile decode for the WMMA kernels (QuantGemvMulti). A Q6_K block is 210 bytes, so blocks are only 2-byte
// aligned: a 16-byte chunk is read as four dwords (funnel-shifted when the address is 2 mod 4) instead of one vector load.
// Element 32 * j + l of a block (sub-block j, l = 0..31) is d * scale[8 * (j / 4) + l / 16 + 2 * (j % 4)] * (q - 32) with the
// 6-bit q assembled from ql[64 * (j / 4) + 32 * ((j % 4) & 1) + l] (low nibble for (j % 4) < 2, high otherwise) and two bits of
// qh[32 * (j / 4) + l]; the multiplication order is the one of Traits<Q6_K>::Element, so the F16 fragments hold exactly
// F16(Element). Device-only.
typedef unsigned v4u_a4 __attribute__((ext_vector_type(4), aligned(4)));
__device__ __forceinline__ uint4 Load16Align2(const std::uint8_t* p) {
  const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p);
  if ((a & 2u) == 0u) {
    const v4u_a4 v = *reinterpret_cast<const v4u_a4*>(p);
    return make_uint4(v.x, v.y, v.z, v.w);
  }
  const unsigned* q = reinterpret_cast<const unsigned*>(a - 2u);
  const v4u_a4 v = *reinterpret_cast<const v4u_a4*>(q);
  const unsigned e = q[4];
  return make_uint4((v.x >> 16) | (v.y << 16), (v.y >> 16) | (v.z << 16), (v.z >> 16) | (v.w << 16), (v.w >> 16) | (e << 16));
}

struct Q6KOps {
  // This lane's share of one superblock: 4 ql chunks (half n, 32-byte plane c), 2 qh chunks (half n), the 16 scales already
  // shifted so that scale(n, q) is byte (2 * (q & 1)) of word 2 * n + (q >> 1), and the F16 d.
  struct TileRegs {
    uint4 ql[4], qh[2];
    unsigned sc[4];
    unsigned d;
  };
  __device__ __forceinline__ static void LoadSuper(const std::uint8_t* blk, std::uint32_t half_id, TileRegs& o) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int c = 0; c < 2; ++c) o.ql[2 * n + c] = Load16Align2(blk + n * 64 + c * 32 + 16 * half_id);
      o.qh[n] = Load16Align2(blk + 128 + n * 32 + 16 * half_id);
    }
    const uint4 s = Load16Align2(blk + 192);
    const unsigned sh = 8u * half_id;
    o.sc[0] = s.x >> sh;
    o.sc[1] = s.y >> sh;
    o.sc[2] = s.z >> sh;
    o.sc[3] = s.w >> sh;
    o.d = *reinterpret_cast<const std::uint16_t*>(blk + 208);
  }
  __device__ __forceinline__ static void DecodeSub16(const TileRegs& r, int j, unsigned (&own)[8]) {
    const int n = j >> 2, q = j & 3;
    const float d = __half2float(__builtin_bit_cast(__half, static_cast<std::uint16_t>(r.d)));
    const float sc = static_cast<float>(static_cast<std::int8_t>((r.sc[2 * n + (q >> 1)] >> (16 * (q & 1))) & 0xFFu));
    const float ds = d * sc;
    const uint4 qs = r.ql[2 * n + (q & 1)];
    const uint4 hv = r.qh[n];
    const unsigned qw[4] = {qs.x, qs.y, qs.z, qs.w};
    const unsigned hw[4] = {hv.x, hv.y, hv.z, hv.w};
#pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
      float v[4];
#pragma unroll
      for (int bi = 0; bi < 4; ++bi) {
        const unsigned lo = (qw[wi] >> (8 * bi + 4 * (q >> 1))) & 0xFu;
        const unsigned hb = (hw[wi] >> (8 * bi + 2 * q)) & 3u;
        v[bi] = ds * static_cast<float>(static_cast<int>((hb << 4) | lo) - 32);
      }
      own[2 * wi] = PackHalves(v[0], v[1]);
      own[2 * wi + 1] = PackHalves(v[2], v[3]);
    }
  }
};

}  // namespace rowdot_detail

/// Generic row dot for any registered format, derived from its QuantFormat traits' Element(): DotRow accumulates the
/// elements in ascending order (CPU-reference order); LaneDot gives each lane whole blocks (block b goes to lane
/// b % 32, elements ascending inside a block). Formats with a tuned kernel specialize RowDot (inheriting this for
/// whichever geometry they do not specialize), so every format in AllFormats is usable by every row-dot kernel.
template <FormatId Id>
struct RowDotGeneric {
  using Tr = Traits<Id>;
  static constexpr FormatId kId = Id;
  static constexpr unsigned kBlockElems = Tr::kBlockElems, kBlockBytes = Tr::kBlockBytes;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t n) { return Tr::RowBytes(n); }
  MBUFO_QT_FN static bool ColsOk(std::size_t n) { return Tr::ColsOk(n); }
  MBUFO_QT_FN static float DotRow(const std::uint8_t* row, const float* vec, std::size_t n) {
    float acc = 0.0F;
    for (std::size_t i = 0; i < n; ++i) acc += Tr::Element(row, static_cast<std::uint32_t>(i)) * vec[i];
    return acc;
  }
  MBUFO_QT_FN static float LaneDot(const std::uint8_t* row, const float* vec, std::size_t n, std::uint32_t lane) {
    float sum = 0.0F;
    const std::size_t nb = n / kBlockElems;
    for (std::size_t b = lane; b < nb; b += 32) {
      float acc = 0.0F;
      const std::size_t base = b * kBlockElems;
      for (unsigned e = 0; e < kBlockElems; ++e)
        acc += Tr::Element(row, static_cast<std::uint32_t>(base + e)) * vec[base + e];
      sum += acc;
    }
    return sum;
  }
};

template <FormatId Id>
struct RowDot : RowDotGeneric<Id> {};  // generic unless specialized below

// Q4_K: per 256-block, 8 sub-blocks of 32 elements; sub-block j reads byte-row (j / 2), low nibbles for even
// j, high for odd. Scalar element loop in slot order.
template <>
struct RowDot<FormatId::kQ4_K> : RowDotGeneric<FormatId::kQ4_K> {
  using Ops = rowdot_detail::KQuantOps<144, 16, false>;
  using TileRegs = typename Ops::TileRegs;
  static constexpr unsigned kSuperElems = 256;
  // Register tile decode (WMMA kernels).
  __device__ __forceinline__ static void LoadSuper(const std::uint8_t* blk, std::uint32_t half_id, TileRegs& o) {
    Ops::LoadSuper(blk, half_id, o);
  }
  __device__ __forceinline__ static void DecodeSub16(const TileRegs& r, int j, unsigned (&own)[8]) { Ops::DecodeSub16(r, j, own); }
  static constexpr FormatId kId = FormatId::kQ4_K;
  static constexpr unsigned kBlockElems = 256, kBlockBytes = 144;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t n) { return n / 256 * sizeof(rowdot_detail::Q4KBlock); }
  MBUFO_QT_FN static bool ColsOk(std::size_t n) { return n != 0 && n % 256 == 0; }
  MBUFO_QT_FN static float DotRow(const std::uint8_t* row, const float* vec, std::size_t n) {
    float acc = 0.0F;
    const std::size_t nb = n / 256;
    for (std::size_t b = 0; b < nb; ++b) {
      const auto wide = rowdot_detail::WideBlock<rowdot_detail::Q4KBlock>::Load(row + b * sizeof(rowdot_detail::Q4KBlock));
      const rowdot_detail::Q4KBlock* blk = &wide.get();
      const float d = rowdot_detail::F16Bits(blk->d);
      const float dmin = rowdot_detail::F16Bits(blk->dmin);
      const float* xv = vec + b * 256;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        std::uint8_t sc = 0;
        std::uint8_t mn = 0;
        rowdot_detail::ScaleMin(blk->scales, j, sc, mn);
        const float s = d * static_cast<float>(sc);
        const float m = dmin * static_cast<float>(mn);
        const std::uint8_t* qb = blk->qs + (j / 2) * 32;
        const bool hi = (j & 1) != 0;
        const float* xw = xv + j * 32;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
          const std::uint8_t q = hi ? static_cast<std::uint8_t>(qb[l] >> 4U) : static_cast<std::uint8_t>(qb[l] & 0x0FU);
          acc += (s * static_cast<float>(q) - m) * xw[l];
        }
      }
    }
    return acc;
  }
};

// Q5_K: Q4_K layout plus 32 high-bit bytes (bit s of qh[lane] is the fifth bit of that lane's element in
// sub-group s).
template <>
struct RowDot<FormatId::kQ5_K> : RowDotGeneric<FormatId::kQ5_K> {
  using Ops = rowdot_detail::KQuantOps<176, 48, true>;
  using TileRegs = typename Ops::TileRegs;
  static constexpr unsigned kSuperElems = 256;
  // Register tile decode (WMMA kernels).
  __device__ __forceinline__ static void LoadSuper(const std::uint8_t* blk, std::uint32_t half_id, TileRegs& o) {
    Ops::LoadSuper(blk, half_id, o);
  }
  __device__ __forceinline__ static void DecodeSub16(const TileRegs& r, int j, unsigned (&own)[8]) { Ops::DecodeSub16(r, j, own); }
  static constexpr FormatId kId = FormatId::kQ5_K;
  static constexpr unsigned kBlockElems = 256, kBlockBytes = 176;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t n) { return n / 256 * sizeof(rowdot_detail::Q5KBlock); }
  MBUFO_QT_FN static bool ColsOk(std::size_t n) { return n != 0 && n % 256 == 0; }
  MBUFO_QT_FN static float DotRow(const std::uint8_t* row, const float* vec, std::size_t n) {
    float acc = 0.0F;
    const std::size_t nb = n / 256;
    for (std::size_t b = 0; b < nb; ++b) {
      const auto wide = rowdot_detail::WideBlock<rowdot_detail::Q5KBlock>::Load(row + b * sizeof(rowdot_detail::Q5KBlock));
      const rowdot_detail::Q5KBlock* blk = &wide.get();
      const float d = rowdot_detail::F16Bits(blk->d);
      const float dmin = rowdot_detail::F16Bits(blk->dmin);
      const float* xv = vec + b * 256;
#pragma unroll
      for (int gg = 0; gg < 4; ++gg) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int bit = 2 * gg + half;
          std::uint8_t sc = 0;
          std::uint8_t mn = 0;
          rowdot_detail::ScaleMin(blk->scales, bit, sc, mn);
          const float s = d * static_cast<float>(sc);
          const float m = dmin * static_cast<float>(mn);
          const std::uint8_t* qb = blk->qs + gg * 32;
          const float* xw = xv + gg * 64 + half * 32;
#pragma unroll
          for (int lane = 0; lane < 32; ++lane) {
            const std::uint8_t q4 =
                half != 0 ? static_cast<std::uint8_t>(qb[lane] >> 4U) : static_cast<std::uint8_t>(qb[lane] & 0x0FU);
            const std::uint8_t q = static_cast<std::uint8_t>(q4 + ((((blk->qh[lane] >> bit) & 1U) != 0U) ? 16U : 0U));
            acc += (s * static_cast<float>(q) - m) * xw[lane];
          }
        }
      }
    }
    return acc;
  }
};

// Q6_K: tile decode for the WMMA multi-token kernel only (the format keeps Traits<Q6_K>::kDecode == kNone, so the shared
// GemvDecode / QuantGemmPrefill routes of the engines that read Q6_K are unchanged; a caller opts in by calling QuantGemvMulti).
template <>
struct RowDot<FormatId::kQ6_K> : RowDotGeneric<FormatId::kQ6_K> {
  using Ops = rowdot_detail::Q6KOps;
  using TileRegs = typename Ops::TileRegs;
  static constexpr unsigned kSuperElems = 256;
  __device__ __forceinline__ static void LoadSuper(const std::uint8_t* blk, std::uint32_t half_id, TileRegs& o) {
    Ops::LoadSuper(blk, half_id, o);
  }
  __device__ __forceinline__ static void DecodeSub16(const TileRegs& r, int j, unsigned (&own)[8]) { Ops::DecodeSub16(r, j, own); }
};

// Q8_0: fp16 scale + 32 int8 codes; no offset term. Scalar loads: a 34-byte block is not 16-byte aligned,
// and a wide-load variant over 8-block groups measured no better (the Q8_0 down path is only used by the
// layers whose Q6_K down weights were converted at load).
template <>
struct RowDot<FormatId::kQ8_0> : RowDotGeneric<FormatId::kQ8_0> {
  // ---- quantized-activation family: 4 blocks per wave step, 8 lanes per block (4 int8 each); signed x signed sdot4,
  // scale product folded first. See RowDot<Q4_0> for the shape of the interface. Device-only.
  static constexpr unsigned kQLaneShift = 3, kQLaneMask = 7, kQBlocksPerStep = 4;
  struct QLane {
    float d;
    int w32;
  };
  __device__ __forceinline__ static void LoadQLane(const std::uint8_t* blk, std::uint32_t sub, QLane& o) {
    const auto* bp = reinterpret_cast<const std::uint16_t*>(blk);
    o.d = __half2float(*reinterpret_cast<const __half*>(bp));
    o.w32 = static_cast<int>(bp[1u + 2u * sub]) | (static_cast<int>(bp[2u + 2u * sub]) << 16);
  }
  template <class XB>
  __device__ __forceinline__ static int QIntDot(const QLane& w, const XB& x, std::uint32_t sub) {
    return intdot::Dot4S8S8(w.w32, x.qs[sub], 0);
  }
  template <class XB>
  __device__ __forceinline__ static float QTail(const QLane& w, const XB& x, int dot) {
    return (w.d * x.d) * static_cast<float>(dot);
  }
  static constexpr FormatId kId = FormatId::kQ8_0;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 34;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t n) { return n / 32 * sizeof(rowdot_detail::Q80Block); }
  MBUFO_QT_FN static bool ColsOk(std::size_t n) { return n != 0 && n % 32 == 0; }
  MBUFO_QT_FN static float DotRow(const std::uint8_t* row, const float* vec, std::size_t n) {
    float acc = 0.0F;
    const std::size_t nb = n / 32;
    for (std::size_t b = 0; b < nb; ++b) {
      const auto* blk = reinterpret_cast<const rowdot_detail::Q80Block*>(row + b * sizeof(rowdot_detail::Q80Block));
      const float d = rowdot_detail::F16Bits(blk->d);
      const float* xw = vec + b * 32;
      for (int l = 0; l < 32; ++l) acc += d * static_cast<float>(blk->qs[l]) * xw[l];
    }
    return acc;
  }
  // Lane geometry (one wave per row): 4 blocks (128 weights) per 32-lane step, lane partial returned; the
  // caller reduces across the wave. Same code as device::WaveRowTileDot's Q8_0 branch (k0 = 0), so a kernel
  // using it is bit-identical to the gemma4/gptoss wave kernels. `vec` must be 16-byte aligned (LDS).
  MBUFO_QT_FN static float LaneDot(const std::uint8_t* row, const float* vec, std::size_t n, std::uint32_t lane) {
    float sum = 0.0F;
    const std::uint32_t nblk = static_cast<std::uint32_t>(n) >> 5;
    const std::uint32_t b_grp = lane >> 3;
    const std::uint32_t sub = lane & 7u;
    for (std::uint32_t b0 = 0; b0 < nblk; b0 += 4) {
      const std::uint32_t b = b0 + b_grp;
      if (b < nblk) {
        const auto* bp = reinterpret_cast<const std::uint16_t*>(row + b * 34u);
        const float d = __half2float(*reinterpret_cast<const __half*>(bp));
        const std::uint16_t q01 = bp[1 + 2 * sub];
        const std::uint16_t q23 = bp[2 + 2 * sub];
        const float4 xv = *reinterpret_cast<const float4*>(vec + b * 32u + 4u * sub);
        float acc = 0.0F;
        acc += static_cast<float>(static_cast<std::int8_t>(q01 & 0xFFu)) * xv.x;
        acc += static_cast<float>(static_cast<std::int8_t>(q01 >> 8)) * xv.y;
        acc += static_cast<float>(static_cast<std::int8_t>(q23 & 0xFFu)) * xv.z;
        acc += static_cast<float>(static_cast<std::int8_t>(q23 >> 8)) * xv.w;
        sum += d * acc;
      }
    }
    return sum;
  }
};

// Q4_0: fp16 scale + 16 bytes of 4-bit codes (value = nibble - 8), 18 bytes per 32 elements. Wave-geometry
// only (gemma4): 8 blocks (256 weights) per 32-lane step, 4 lanes per block; same code as
// device::WaveRowTileDot's Q4_0 branch (k0 = 0).
template <>
struct RowDot<FormatId::kQ4_0> : RowDotGeneric<FormatId::kQ4_0> {
  // ---- quantized-activation family (Q8_1 blocks: d, s = d * sum(q), 32 int8 as 8 packed ints). One wave step covers 8 blocks,
  // 4 lanes per block; a lane's share is the scale and two packed words of 4-bit codes. Integer dot with the sdot4 helper;
  // the float tail (offset 8 is folded through the activation sum s, a quarter per lane: 2 * s) is the form every Q4_0 x Q8_1
  // kernel always used. Device-only.
  static constexpr unsigned kQLaneShift = 2, kQLaneMask = 3, kQBlocksPerStep = 8;
  struct QLane {
    float d;
    int vi0, vi1;
  };
  __device__ __forceinline__ static void LoadQLane(const std::uint8_t* blk, std::uint32_t sub, QLane& o) {
    const auto* bp = reinterpret_cast<const std::uint16_t*>(blk);
    o.d = __half2float(*reinterpret_cast<const __half*>(bp));
    const int v32 = static_cast<int>(bp[1u + 2u * sub]) | (static_cast<int>(bp[2u + 2u * sub]) << 16);
    o.vi0 = v32 & 0x0F0F0F0F;
    o.vi1 = (v32 >> 4) & 0x0F0F0F0F;
  }
  template <class XB>
  __device__ __forceinline__ static int QIntDot(const QLane& w, const XB& x, std::uint32_t sub) {
    int dot = intdot::Dot4U4S8(w.vi0, x.qs[sub], 0);
    dot = intdot::Dot4U4S8(w.vi1, x.qs[4u + sub], dot);
    return dot;
  }
  template <class XB>
  __device__ __forceinline__ static float QTail(const QLane& w, const XB& x, int dot) {
    return w.d * (x.d * static_cast<float>(dot) - 2.0F * x.s);
  }
  // ---- primitives of the multi-token / legacy fallback kernels (kept bit-for-bit; device-only) ----
  // A lane's share of one block under the wave geometry: the scale and the eight signed weights it owns, in the order
  // lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3 (low nibble then high nibble of each byte pair).
  struct LaneBlock {
    float d;
    float w[8];
  };
  __device__ __forceinline__ static void LoadLaneBlock(const std::uint8_t* blk, std::uint32_t sub, LaneBlock& o) {
    const auto* bp = reinterpret_cast<const std::uint16_t*>(blk);
    o.d = __half2float(*reinterpret_cast<const __half*>(bp));
    const std::uint32_t q01 = bp[1 + 2 * sub];
    const std::uint32_t q23 = bp[2 + 2 * sub];
    o.w[0] = static_cast<float>(static_cast<int>(q01 & 0xFu) - 8);
    o.w[1] = static_cast<float>(static_cast<int>((q01 >> 4) & 0xFu) - 8);
    o.w[2] = static_cast<float>(static_cast<int>((q01 >> 8) & 0xFu) - 8);
    o.w[3] = static_cast<float>(static_cast<int>((q01 >> 12) & 0xFu) - 8);
    o.w[4] = static_cast<float>(static_cast<int>(q23 & 0xFu) - 8);
    o.w[5] = static_cast<float>(static_cast<int>((q23 >> 4) & 0xFu) - 8);
    o.w[6] = static_cast<float>(static_cast<int>((q23 >> 8) & 0xFu) - 8);
    o.w[7] = static_cast<float>(static_cast<int>((q23 >> 12) & 0xFu) - 8);
  }
  // The eight products as ONE expression (the multi-token kernels share the decoded block across tokens and write it this
  // way; LaneDot above accumulates them one statement at a time, a different contraction), unscaled: callers do sum += d * acc.
  __device__ __forceinline__ static float LaneBlockExpr(const LaneBlock& b, float4 xlo, float4 xhi) {
    return b.w[0] * xlo.x + b.w[1] * xhi.x + b.w[2] * xlo.y + b.w[3] * xhi.y + b.w[4] * xlo.z + b.w[5] * xhi.z +
           b.w[6] * xlo.w + b.w[7] * xhi.w;
  }
  static constexpr FormatId kId = FormatId::kQ4_0;
  static constexpr unsigned kBlockElems = 32, kBlockBytes = 18;
  MBUFO_QT_FN static std::size_t RowBytes(std::size_t n) { return n / 32 * 18; }
  MBUFO_QT_FN static bool ColsOk(std::size_t n) { return n != 0 && n % 32 == 0; }
  MBUFO_QT_FN static float LaneDot(const std::uint8_t* row, const float* vec, std::size_t n, std::uint32_t lane) {
    float sum = 0.0F;
    const std::uint32_t nblk = static_cast<std::uint32_t>(n) >> 5;
    const std::uint32_t b_grp = lane >> 2;
    const std::uint32_t sub = lane & 3u;
    for (std::uint32_t b0 = 0; b0 < nblk; b0 += 8) {
      const std::uint32_t b = b0 + b_grp;
      if (b < nblk) {
        const auto* bp = reinterpret_cast<const std::uint16_t*>(row + b * 18u);
        const float d = __half2float(*reinterpret_cast<const __half*>(bp));
        const std::uint32_t q01 = bp[1 + 2 * sub];
        const std::uint32_t q23 = bp[2 + 2 * sub];
        const float4 xlo = *reinterpret_cast<const float4*>(vec + b * 32u + 4u * sub);
        const float4 xhi = *reinterpret_cast<const float4*>(vec + b * 32u + 16u + 4u * sub);
        float acc = 0.0F;
        acc += static_cast<float>(static_cast<int>(q01 & 0xFu) - 8) * xlo.x;
        acc += static_cast<float>(static_cast<int>((q01 >> 4) & 0xFu) - 8) * xhi.x;
        acc += static_cast<float>(static_cast<int>((q01 >> 8) & 0xFu) - 8) * xlo.y;
        acc += static_cast<float>(static_cast<int>((q01 >> 12) & 0xFu) - 8) * xhi.y;
        acc += static_cast<float>(static_cast<int>(q23 & 0xFu) - 8) * xlo.z;
        acc += static_cast<float>(static_cast<int>((q23 >> 4) & 0xFu) - 8) * xhi.z;
        acc += static_cast<float>(static_cast<int>((q23 >> 8) & 0xFu) - 8) * xlo.w;
        acc += static_cast<float>(static_cast<int>((q23 >> 12) & 0xFu) - 8) * xhi.w;
        sum += d * acc;
      }
    }
    return sum;
  }
};

// MXFP4 (OCP microscaling, 17 bytes per 32: one e8m0 scale byte + 16 e2m1 nibble bytes). Engines work with the integer
// codebook {0,1,2,3,4,6,8,12} against 2^(e-128), the exact double of the halved codebook against 2^(e-127) that
// Traits::Element uses. Two per-block accumulation orders exist because the engines grew two: BlockDot (two separate
// adds per byte pair: lo * x[j], then hi * x[j+16]) and BlockDotPairedAcc (the gate/up pair of the fused gpt-oss decode
// kernel: one combined lo*x0 + hi*x1 term per byte pair, scale folded per block). Both are kept so every kernel's
// numerics stay bit-identical to what it always computed; LaneDot strides blocks across lanes with BlockDot.
template <>
struct RowDot<FormatId::kMXFP4> : RowDotGeneric<FormatId::kMXFP4> {
  // Integer codebook from constant memory, exactly as the gpt-oss kernels always read it. (A local constexpr table cost 14-28%
  // on the fused kernels and an arithmetic decode changed their floating-point contraction, so neither is used.) The MXFP4 block
  // functions are device-only for that reason; they are never called from host code.
  __device__ __forceinline__ static float Code(std::uint32_t nibble) {
    return static_cast<float>(rowdot_detail::kMxfp4Codes[nibble & 15u]);
  }
  __device__ __forceinline__ static float BlockDot(const std::uint8_t* blk, const float* xv) {
    const float d = exp2f(static_cast<float>(blk[0]) - 128.0F);
    float sum = 0.0F;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const std::uint8_t b = blk[1 + j];
      sum += Code(b & 0x0Fu) * xv[j];
      sum += Code(b >> 4) * xv[j + 16];
    }
    return d * sum;
  }
  __device__ __forceinline__ static void BlockDotPairedAcc(const std::uint8_t* gblk, const std::uint8_t* ublk, const float* xv, float* sg,
                                           float* su) {
    const float dg = exp2f(static_cast<float>(gblk[0]) - 128.0F);
    const float du = exp2f(static_cast<float>(ublk[0]) - 128.0F);
    float tg = 0.0F, tu = 0.0F;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const std::uint8_t gb = gblk[1 + j];
      const std::uint8_t ub = ublk[1 + j];
      const float x0 = xv[j];
      const float x1 = xv[j + 16];
      tg += Code(gb & 0x0Fu) * x0 + Code(gb >> 4) * x1;
      tu += Code(ub & 0x0Fu) * x0 + Code(ub >> 4) * x1;
    }
    *sg += dg * tg;
    *su += du * tu;
  }
  __device__ __forceinline__ static float LaneDot(const std::uint8_t* row, const float* vec, std::size_t n, std::uint32_t lane) {
    float sum = 0.0F;
    const std::size_t nb = n / 32;
    for (std::size_t b = lane; b < nb; b += 32) sum += BlockDot(row + b * 17, vec + b * 32);
    return sum;
  }
};

// F32 / F16 / BF16: 4 contiguous elements per lane per 128-element step, lane partial returned (the
// float4-per-lane code device::WaveRowTileDot always had; DotRow stays generic).
template <>
struct RowDot<FormatId::kF32> : RowDotGeneric<FormatId::kF32> {
  // The four-term expression of one float4 step as the multi-token kernels write it (one expression).
  __device__ __forceinline__ static float Dot4Expr(float4 wv, float4 xv) {
    return wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
  }
  MBUFO_QT_FN static float LaneDot(const std::uint8_t* row, const float* vec, std::size_t n, std::uint32_t lane) {
    float sum = 0.0F;
    const auto* r = reinterpret_cast<const float*>(row);
    for (std::uint32_t j = lane * 4u; j < n; j += 128u) {
      const float4 wv = *reinterpret_cast<const float4*>(r + j);
      const float4 xv = *reinterpret_cast<const float4*>(vec + j);
      sum += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
    return sum;
  }
};
template <>
struct RowDot<FormatId::kF16> : RowDotGeneric<FormatId::kF16> {
  MBUFO_QT_FN static float LaneDot(const std::uint8_t* row, const float* vec, std::size_t n, std::uint32_t lane) {
    float sum = 0.0F;
    const auto* r = reinterpret_cast<const __half*>(row);
    for (std::uint32_t j = lane * 4u; j < n; j += 128u) {
      const uint2 u = *reinterpret_cast<const uint2*>(r + j);
      const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
      const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
      const float4 xv = *reinterpret_cast<const float4*>(vec + j);
      sum += f0.x * xv.x + f0.y * xv.y + f1.x * xv.z + f1.y * xv.w;
    }
    return sum;
  }
};
template <>
struct RowDot<FormatId::kBF16> : RowDotGeneric<FormatId::kBF16> {
  MBUFO_QT_FN static float LaneDot(const std::uint8_t* row, const float* vec, std::size_t n, std::uint32_t lane) {
    float sum = 0.0F;
    const auto* r = reinterpret_cast<const std::uint16_t*>(row);
    for (std::uint32_t j = lane * 4u; j < n; j += 128u) {
      const uint2 u = *reinterpret_cast<const uint2*>(r + j);
      const float w0 = __builtin_bit_cast(float, (u.x & 0xFFFFu) << 16);
      const float w1 = __builtin_bit_cast(float, u.x & 0xFFFF0000u);
      const float w2 = __builtin_bit_cast(float, (u.y & 0xFFFFu) << 16);
      const float w3 = __builtin_bit_cast(float, u.y & 0xFFFF0000u);
      const float4 xv = *reinterpret_cast<const float4*>(vec + j);
      sum += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
    }
    return sum;
  }
};

/// Geometry every MoE row format declares, whatever reduction geometry its kernels use.
template <class T>
concept RowGeometry = requires(std::size_t n) {
  { T::kId } -> std::convertible_to<FormatId>;
  { T::kBlockElems } -> std::convertible_to<unsigned>;
  { T::kBlockBytes } -> std::convertible_to<unsigned>;
  { T::RowBytes(n) } -> std::same_as<std::size_t>;
  { T::ColsOk(n) } -> std::same_as<bool>;
};

/// Thread geometry (each thread owns whole rows, sequential accumulation; qwen35moe). Adding a format to
/// ThreadRowFormats without DotRow fails to compile.
template <class T>
concept ThreadRowFormat = RowGeometry<T> && requires(const std::uint8_t* row, const float* vec, std::size_t n) {
  { T::DotRow(row, vec, n) } -> std::same_as<float>;
};

/// Lane geometry (one wave per row, lane partials reduced by the caller; gemma4, gptoss).
template <class T>
concept LaneRowFormat =
    RowGeometry<T> && requires(const std::uint8_t* row, const float* vec, std::size_t n, std::uint32_t lane) {
      { T::LaneDot(row, vec, n, lane) } -> std::same_as<float>;
    };

/// A format whose superblock can be decoded straight into WMMA A-fragment registers (QuantGemvMulti): a per-lane register
/// bundle, a loader, and a 16-element sub-block decoder producing packed F16 pairs.
template <class T>
concept TileDecodeFormat = RowGeometry<T> && T::kSuperElems == 256 &&
    requires(const std::uint8_t* blk, std::uint32_t half_id, typename T::TileRegs& regs, unsigned (&own)[8]) {
  T::LoadSuper(blk, half_id, regs);
  T::DecodeSub16(regs, 0, own);
};

/// A format with quantized-activation (Q8_1) lane kernels: a lane's decoded share of a block, the integer dot with one
/// activation block, and the float tail that scales it.
struct ProbeQ8Block {
  float d, s;
  int qs[8];
};
template <class T>
concept QActLaneFormat = RowGeometry<T> && requires(const std::uint8_t* blk, std::uint32_t sub, typename T::QLane& w, const ProbeQ8Block& x, int dot) {
  T::kQLaneShift;
  T::kQLaneMask;
  T::kQBlocksPerStep;
  T::LoadQLane(blk, sub, w);
  { T::QIntDot(w, x, sub) } -> std::same_as<int>;
  { T::QTail(w, x, dot) } -> std::same_as<float>;
};

/// A RowDot must describe the same geometry as the format's QuantFormat traits.
template <class M>
concept RowMatchesTraits = RowGeometry<M> && QuantFormat<Traits<M::kId>> &&
                           M::kBlockElems == Traits<M::kId>::kBlockElems &&
                           M::kBlockBytes == Traits<M::kId>::kBlockBytes;

/// Every format registered in quant::AllFormats, as a RowDot. A format added to AllFormats (it needs Element) is
/// usable by every row-dot kernel in both geometries with no further step; a type code that is not registered is
/// refused at run time, never decoded as another format.
template <class List>
struct RowFormatsOf;
template <class... Fs>
struct RowFormatsOf<FormatList<Fs...>> {
  static constexpr bool kAllValid = ((ThreadRowFormat<RowDot<Fs::kId>> && LaneRowFormat<RowDot<Fs::kId>>) && ...);
  static constexpr bool kMatchTraits = (RowMatchesTraits<RowDot<Fs::kId>> && ...);
  /// Calls f(RowDot{}) for the format with this type code; false (f not called) when it is not registered.
  template <class F>
  static bool Dispatch(int type, F&& f) {
    return AllFormats::Dispatch(type, [&](auto tr) { f(RowDot<decltype(tr)::kId>{}); });
  }
};
using RowFormats = RowFormatsOf<AllFormats>;
static_assert(RowFormats::kAllValid, "every registered format must have both row-dot geometries");
static_assert(RowFormats::kMatchTraits, "RowDot geometry must match the format's Traits");

/// Kernel-facing names: the geometry a kernel uses; both resolve to the same registry.
using ThreadRowFormats = RowFormats;
using LaneRowFormats = RowFormats;

}  // namespace mbufo::core::quant

#endif  // ENGINE_ENABLE_HIP
#endif  // MBUFO_CORE_QUANT_ROW_DOT_TRAITS_HPP_
