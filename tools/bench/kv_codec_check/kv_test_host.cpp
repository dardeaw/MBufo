// kv_codec_check: the shared blocked-KV codec, pinned.
//
// Host part (no GPU): the trait math in blocked_kv_codec.hpp must agree
// element-for-element with independently written oracles (double-precision
// reference implementations, not copies of the trait code), and half<->float
// must round-trip all 65536 patterns plus random values against a
// frexp/ldexp oracle.
// Device part (needs one ROCm GPU): the shared kernels that now call the
// trait (QuantizeKvRows / DequantizeKvRows / AttentionKv /
// AttentionBlockedSplitK) must reproduce a host FP32 reference within the
// format's error budget. Any future codec edit that changes behavior fails
// here before it can silently fork a model's numerics.
//
// Built by tools/bench/kv_codec_check/kv_ab.sh.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "src/core/compute/attention/blocked_kv_codec.hpp"

namespace bk = mbufo::core::compute::blocked_kv;

static int g_bad = 0;
#define CHECK(cond, ...)                                   \
  do {                                                    \
    if (!(cond)) {                                        \
      ++g_bad;                                            \
      std::printf("FAIL " __VA_ARGS__);                   \
      std::printf("\n");                                  \
    }                                                     \
  } while (0)

// ---------------- independent double-precision oracles ----------------

// Reference float->half (RNE) via frexp/ldexp, written without looking at
// the trait implementation: decompose, round the mantissa in double,
// recompose.
static std::uint16_t OracleFloatToHalfBits(double f) {
  const bool neg = std::signbit(f);
  double a = std::fabs(f);
  std::uint16_t sign = neg ? 0x8000u : 0u;
  if (f != f)
    return static_cast<std::uint16_t>(sign | 0x7E00u);
  if (a > 65504.0)
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  if (a < 0.5 * 0.000000059604644775390625)
    return sign;  // below half of the smallest subnormal (2^-25)
  if (a < 0.00006103515625) {
    // subnormal: round a/2^-24 to nearest even integer in [1, 1023]
    double m = a * 16777216.0;
    double r = std::rint(m);
    if (r >= 1024.0)  // rounded up into normal range
      return static_cast<std::uint16_t>(sign | 0x0400u);
    return static_cast<std::uint16_t>(sign | static_cast<std::uint16_t>(r));
  }
  int exp = 0;
  double m = std::frexp(a, &exp);  // a = m * 2^exp, m in [0.5, 1)
  // want (1 + f/1024) * 2^(exp-1) with RNE on f
  double f10 = (m * 2.0 - 1.0) * 1024.0;
  double r = std::rint(f10);
  int he = exp - 1 + 15;
  std::uint16_t hmant = static_cast<std::uint16_t>(r);
  if (hmant >= 1024u) {
    hmant = 0;
    ++he;
    if (he >= 31)
      return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint16_t>(he) << 10) | hmant);
}

static float OracleHalfToFloat(std::uint16_t h) {
  int sign = (h >> 15) & 1;
  int exp = (h >> 10) & 0x1F;
  int mant = h & 0x3FF;
  double v;
  if (exp == 0) {
    v = std::ldexp(static_cast<double>(mant), -24);
  } else if (exp == 31) {
    v = mant ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
  } else {
    v = std::ldexp(1024.0 + mant, exp - 25);
  }
  return static_cast<float>(sign ? -v : v);
}

// ---------------- host tests ----------------

static void TestHalfBits() {
  int bad = 0;
  for (std::uint32_t h = 0; h < 65536u; ++h) {
    float f = bk::HalfBitsToFloat(static_cast<std::uint16_t>(h));
    std::uint16_t rt = bk::FloatToHalfBits(f);
    // NaN payloads are canonicalized; everything else must round-trip.
    if (((h & 0x7C00u) == 0x7C00u) && (h & 0x03FFu)) {
      if ((rt & 0x7C00u) != 0x7C00u || (rt & 0x03FFu) == 0u)
        ++bad;
    } else if (rt != static_cast<std::uint16_t>(h)) {
      // -0.0 vs +0.0 are distinct bit patterns of the same value; the
      // trait preserves the sign, so require exactness.
      ++bad;
    }
  }
  std::mt19937_64 rng(1234);
  int orbad = 0;
  double worst = 0;
  for (int i = 0; i < 200000; ++i) {
    union {
      std::uint32_t u;
      float f;
    } c;
    // mix of normal, subnormal-edge, large and tiny magnitudes
    switch (i % 4) {
      case 0: c.u = static_cast<std::uint32_t>(rng()); break;
      case 1: c.u = static_cast<std::uint32_t>(rng() & 0x7FFFFF) | (static_cast<std::uint32_t>(rng() % 2) << 31); break;
      case 2: c.u = 0x33800000u + static_cast<std::uint32_t>(rng() % 0x7FFFFFu); break;
      default: c.f = static_cast<float>((rng() % 2000000 - 1000000) * 1e-6); break;
    }
    float f = c.f;
    std::uint16_t got = bk::FloatToHalfBits(f);
    std::uint16_t want = OracleFloatToHalfBits(static_cast<double>(f));
    bool both_nan = ((got & 0x7C00u) == 0x7C00u) && (got & 0x03FFu) && ((want & 0x7C00u) == 0x7C00u) &&
                     (want & 0x03FFu);
    if (!both_nan && got != want) {
      double err = std::fabs(static_cast<double>(got) - static_cast<double>(want));
      worst = worst > err ? worst : err;
      ++orbad;
    }
    // decode path vs oracle
    float df = bk::HalfBitsToFloat(got);
    float rf = OracleHalfToFloat(got);
    if (!(df != df && rf != rf) && df != rf)
      ++orbad;
  }
  std::printf("  half bits: roundtrip mismatches=%d oracle mismatches=%d %s\n", bad, orbad,
              (bad || orbad) ? "FAIL" : "ok");
  if (bad || orbad)
    ++g_bad;
  (void)worst;
}

static void TestQ8Codec() {
  std::mt19937 rng(777);
  int bad = 0;
  double worst = 0;
  for (int t = 0; t < 5000; ++t) {
    float vals[32];
    float amax = 0;
    for (int i = 0; i < 32; ++i) {
      vals[i] = static_cast<float>(static_cast<int>(rng() % 200000) - 100000) * 1e-5f *
                (1 + (t % 7) * 0.37f);
      amax = amax > std::fabs(vals[i]) ? amax : std::fabs(vals[i]);
    }
    float sc = amax > 0 ? amax / 127.0f : 1.0f;
    float tsc = bk::ScaleFromAmax(amax, bk::KvCodecQ8::kRange);
    if (tsc != sc)
      ++bad;
    for (int i = 0; i < 32; ++i) {
      float v = vals[i] / sc;
      v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
      int q = static_cast<int>(std::rint(v));
      std::uint8_t want = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
      std::uint8_t got = bk::KvCodecQ8::Encode(vals[i] / sc);
      if (got != want)
        ++bad;
      float dq = static_cast<float>(static_cast<std::int8_t>(got));
      if (bk::KvCodecQ8::Decode(got) != dq)
        ++bad;
      worst = std::fmax(worst, std::fabs(static_cast<double>(bk::KvCodecQ8::Decode(got) * tsc - vals[i])));
    }
  }
  double worst_amax = 0;
  // (recompute a global amax for a meaningful roundtrip bound)
  {
    std::mt19937 rng2(777);
    for (int t = 0; t < 5000; ++t)
      for (int i = 0; i < 32; ++i) {
        float v = static_cast<float>(static_cast<int>(rng2() % 200000) - 100000) * 1e-5f *
                  (1 + (t % 7) * 0.37f);
        worst_amax = worst_amax > std::fabs(static_cast<double>(v)) ? worst_amax : std::fabs(static_cast<double>(v));
      }
  }
  double bound = worst_amax / 127.0 / 2.0 + 1e-3;
  std::printf("  Q8 codec: mismatches=%d worst roundtrip err=%.5f (bound %.5f) %s\n", bad, worst, bound,
              (bad || worst > bound) ? "FAIL" : "ok");
  if (bad || worst > bound)
    ++g_bad;
}

static void TestNibble() {
  std::mt19937 rng(4242);
  int bad = 0;
  for (int t = 0; t < 20000; ++t) {
    float v = static_cast<float>(static_cast<int>(rng() % 20000) - 10000) * 1e-4f * 3.0f;
    float c = v > 7.0f ? 7.0f : (v < -7.0f ? -7.0f : v);
    int q = static_cast<int>(std::rint(static_cast<double>(c)));
    std::uint32_t want = static_cast<std::uint32_t>(q) & 0xFu;
    if (bk::KvCodecQ4::EncodeNibble(v) != want)
      ++bad;
    int dq = static_cast<int>(want) - ((want & 8u) ? 16 : 0);
    if (bk::KvCodecQ4::DecodeNibble(want) != dq)
      ++bad;
  }
  std::printf("  nibble codec: mismatches=%d %s\n", bad, bad ? "FAIL" : "ok");
  if (bad)
    ++g_bad;
}

static void TestFp8Codec() {
  std::mt19937_64 rng(999);
  int bad = 0;
  for (int t = 0; t < 50000; ++t) {
    union {
      std::uint32_t u;
      float f;
    } c;
    switch (t % 5) {
      case 0: c.u = static_cast<std::uint32_t>(rng()); break;
      case 1: c.f = static_cast<float>((rng() % 2000000 - 1000000) * 1e-4); break;
      case 2: c.f = static_cast<float>((rng() % 2000 - 1000) * 1e-3); break;
      case 3: c.u = 0x7F800000u | static_cast<std::uint32_t>(rng() & 0x7FFFFFu); break;
      default: c.u = static_cast<std::uint32_t>(rng() & 0xFFFFFFu); break;
    }
    float x = c.f;
    // independent oracle: E4M3 per the format spec (bias 7, RNE, NaN, subnormals)
    std::uint8_t want = 0;
    {
      double a = std::fabs(static_cast<double>(x));
      std::uint32_t sign = (std::signbit(static_cast<double>(x)) ? 0x80u : 0u);
      if (x != x) {
        want = static_cast<std::uint8_t>(sign | 0x7Fu);
      } else if (a >= 448.0) {
        want = static_cast<std::uint8_t>(sign | 0x7Eu);
      } else if (a < 0.015625) {
        want = static_cast<std::uint8_t>(sign | static_cast<std::uint32_t>(std::rint(a * 512.0)));
      } else {
        int exp = 0;
        double m = std::frexp(a, &exp);  // [0.5,1)
        // E4M3 normal: (1 + f/8) * 2^(exp-1), unbiased E4M3 exp = exp-1+7
        double f8 = (m * 2.0 - 1.0) * 8.0;
        double r = std::rint(f8);
        int e = exp - 1 + 7;
        int m3 = static_cast<int>(r);
        if (m3 >= 8) {
          m3 = 0;
          ++e;
        }
        if (e > 15)
          want = static_cast<std::uint8_t>(sign | 0x7Eu);
        else
          want = static_cast<std::uint8_t>(sign | (static_cast<std::uint32_t>(e) << 3) | static_cast<std::uint32_t>(m3));
      }
    }
    std::uint8_t got = bk::KvCodecFp8::Encode(x);
    if (got != want) {
      // NaN payloads may differ in low bits; both must be NaN encodings.
      bool both_nan_enc = ((got & 0x7Fu) == 0x7Fu || (got & 0x7Fu) == 0x7Eu) &&
                          ((want & 0x7Fu) == 0x7Fu || (want & 0x7Fu) == 0x7Eu) && (x != x);
      if (!both_nan_enc)
        ++bad;
    }
    // decode agreement: same bit manipulation as the trait (pins the
    // historical behavior, including 0x7E -> 448.0), plus hand-computed
    // known-answer checks below.
    {
      std::uint32_t u = got;
      std::uint32_t exp = (u & 0x7Fu) >> 3;
      float wantv;
      if (exp == 0) {
        wantv = static_cast<float>(u & 7u) * (1.0f / 512.0f);
        if ((u >> 7) & 1u)
          wantv = -wantv;
      } else {
        std::uint32_t bits = ((u >> 7) & 1u) << 31 | ((exp + 120u) << 23) | ((u & 7u) << 20);
        std::memcpy(&wantv, &bits, sizeof(wantv));
      }
      float gd = bk::KvCodecFp8::Decode(got);
      bool both_nan = (gd != gd) && (wantv != wantv);
      if (!both_nan && std::memcmp(&gd, &wantv, sizeof(float)) != 0)
        ++bad;
    }
  }
  std::printf("  Fp8 codec: mismatches=%d %s\n", bad, bad ? "FAIL" : "ok");
  if (bad)
    ++g_bad;
  // hand-computed known answers (independent of both trait and oracle)
  {
    struct Kat {
      std::uint8_t b;
      float v;
    };
    const Kat kats[] = {{0x00, 0.0f}, {0x08, 0.015625f}, {0x3C, 1.5f}, {0x40, 2.0f},
                        {0x7E, 448.0f}, {0xFE, -448.0f}, {0x38, 1.0f}};
    for (const auto& k : kats) {
      float got = bk::KvCodecFp8::Decode(k.b);
      std::uint32_t gb = 0, kb = 0;
      std::memcpy(&gb, &got, 4);
      std::memcpy(&kb, &k.v, 4);
      // -0.0 vs +0.0: compare magnitudes, then sign separately
      if (gb != kb) {
        std::printf("  Fp8 KAT 0x%02X: got %a want %a FAIL\n", k.b, static_cast<double>(got),
                    static_cast<double>(k.v));
        ++g_bad;
      }
    }
  }
}

static void TestView() {
  int bad = 0;
  auto v = bk::MakeBlockedKvView<bk::KvCodecQ8>(nullptr, 1024u, 128u);
  if (v.row_stride != 1088u) {
    std::printf("  view row_stride %u != 1088\n", v.row_stride);
    ++bad;
  }
  if (v.head_stride != 136u) {
    std::printf("  view head_stride %u != 136\n", v.head_stride);
    ++bad;
  }
  if (!bk::IsBlockedKvView<decltype(v)>::value) {
    std::printf("  IsBlockedKvView false for view\n");
    ++bad;
  }
  if (bk::IsBlockedKvView<std::uint8_t*>::value) {
    std::printf("  IsBlockedKvView true for raw pointer\n");
    ++bad;
  }
  if (bk::IsBlockedKvView<std::uint32_t>::value) {
    std::printf("  IsBlockedKvView true for uint32\n");
    ++bad;
  }
  auto v4 = bk::MakeBlockedKvView<bk::KvCodecQ4>(nullptr, 1024u, 128u);
  if (v4.row_stride != 576u) {
    std::printf("  Q4 view row_stride %u != 576\n", v4.row_stride);
    ++bad;
  }
  if (v4.head_stride != 72u) {
    std::printf("  Q4 view head_stride %u != 72\n", v4.head_stride);
    ++bad;
  }
  std::printf("  view/layout: %s\n", bad ? "FAIL" : "ok");
  if (bad)
    ++g_bad;
}

int main() {
  std::printf("kv_codec_check (trait-vs-oracle, host):\n");
  TestHalfBits();
  TestQ8Codec();
  TestNibble();
  TestFp8Codec();
  TestView();
  if (g_bad) {
    std::printf("HOST FAILED (%d)\n", g_bad);
    return 1;
  }
  std::printf("host OK\n");
  return 0;
}
