// Traits<F>::Element (the decode every gather / dequant / row-dot kernel shares) against the host oracles of
// quant/ggml_dequant for the K-quant and IQ4 formats, on random packed rows: they must agree element for element.
// Built by tools/bench/quant_traits_check/qt_ab.sh.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/quant_traits.hpp"

namespace q = mbufo::core::quant;

template <q::FormatId Id, class Oracle>
static int Check(const char* name, Oracle oracle, std::mt19937& rng) {
  using T = q::Traits<Id>;
  const std::size_t cols = 256 * 6;  // six super-blocks (also covers the 32-wide format)
  std::vector<std::uint8_t> row(T::RowBytes(cols));
  int bad = 0;
  double worst = 0;
  for (int trial = 0; trial < 200; ++trial) {
    for (auto& b : row) b = static_cast<std::uint8_t>(rng());
    // keep the fp16 scales finite and moderate: set the exponent bits of every d / dmin field
    for (std::size_t b = 0; b < cols / T::kBlockElems; ++b) {
      std::uint8_t* blk = row.data() + b * T::kBlockBytes;
      const std::size_t d_off = Id == q::FormatId::kQ3_K ? 108 : 0;  // IQ4_NL / IQ4_XS keep d first
      const std::uint16_t h = static_cast<std::uint16_t>(0x2800 | (rng() & 0x03FF));  // ~ 0.03 .. 0.06
      std::memcpy(blk + d_off, &h, 2);
      if (Id == q::FormatId::kQ4_K) std::memcpy(blk + 2, &h, 2);  // dmin
    }
    std::vector<float> ref(cols);
    oracle(row.data(), ref.data(), cols);
    for (std::size_t i = 0; i < cols; ++i) {
      const float got = T::Element(row.data(), static_cast<std::uint32_t>(i));
      const double err = std::fabs(static_cast<double>(got) - ref[i]);
      worst = std::fmax(worst, err);
      if (err > 1e-6 * std::fmax(1.0, std::fabs(static_cast<double>(ref[i])))) ++bad;
    }
  }
  std::printf("  %-7s Traits::Element vs host oracle: %d mismatches over 200 random rows (max |err| %.3g) %s\n", name, bad, worst,
              bad ? "FAIL" : "ok");
  return bad != 0;
}

int main() {
  std::mt19937 rng(5);
  int bad = 0;
  bad += Check<q::FormatId::kQ3_K>("Q3_K", mbufo::quant::DequantizeQ3_K, rng);
  bad += Check<q::FormatId::kIQ4_NL>("IQ4_NL", mbufo::quant::DequantizeIQ4_NL, rng);
  bad += Check<q::FormatId::kIQ4_XS>("IQ4_XS", mbufo::quant::DequantizeIQ4_XS, rng);
  bad += Check<q::FormatId::kQ4_K>("Q4_K", mbufo::quant::DequantizeQ4_K, rng);
  std::printf(bad ? "FAILED\n" : "all OK\n");
  return bad ? 1 : 0;
}
