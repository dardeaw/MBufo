// Decode MoE (core/compute/gemm/moe_decode_kq.hip, formats from core/quant/row_dot_traits.hpp) against an
// independent float64 host reference built from the QuantFormat traits' Element(), on random packed weights,
// with timing at qwen35moe shapes. Checks every format pair, every row-tile size, masked slots, and that
// unlisted types / bad widths are refused.
//   hipcc -O3 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I<repo> -x hip mdk_test.cpp \
//         <repo>/src/core/compute/gemm/moe_decode_kq.hip -o mdk_test && ./mdk_test
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/quant/row_dot_traits.hpp"
#include "src/core/quant/quant_traits.hpp"

namespace gg = mbufo::core::compute::gemm;
namespace quant = mbufo::core::quant;

#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

static std::size_t RowBytes(int t, std::size_t n) {
  std::size_t r = 0;
  (void)quant::AllFormats::Dispatch(t, [&](auto fm) { r = decltype(fm)::RowBytes(n); });
  return r;
}
static double Element(int t, const std::uint8_t* row, std::uint32_t i) {
  double v = 0;
  (void)quant::AllFormats::Dispatch(t, [&](auto fm) { v = decltype(fm)::Element(row, i); });
  return v;
}

template <class Fn>
static double Time(Fn fn, int iters = 200) {
  for (int i = 0; i < 20; ++i) fn();
  CK(hipDeviceSynchronize());
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  CK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
  float ms; CK(hipEventElapsedTime(&ms, a, b));
  return ms * 1000.0 / iters;
}

int main() {
  // trait-list sanity at run time too (the static_asserts already gate the build)
  int bad = 0;
  const std::size_t hidden = 2048, ff = 512, n_exp = 64;  // 64 experts keeps the host reference quick
  const std::uint32_t top_k = 8;
  struct Combo { int gate, down; } combos[] = {{12, 13}, {12, 8}, {13, 13}, {13, 8}};
  for (auto cb : combos)
    for (std::uint32_t nt : {1u, 3u, 8u}) {
      const std::size_t grb = RowBytes(cb.gate, hidden), drb = RowBytes(cb.down, ff);
      const std::size_t gbytes = n_exp * ff * grb, dbytes = n_exp * hidden * drb;
      std::vector<std::uint8_t> hg(gbytes), hu(gbytes), hd(dbytes);
      srand(21 + cb.gate * 7 + cb.down);
      for (auto& v : hg) v = (std::uint8_t)rand();
      for (auto& v : hu) v = (std::uint8_t)rand();
      for (auto& v : hd) v = (std::uint8_t)rand();
      auto fix = [&](std::vector<std::uint8_t>& w, std::size_t rows, std::size_t rb, int t) {  // small fp16 scales
        const std::uint16_t h = 0x1c00;
        const std::size_t bb = t == 12 ? 144 : t == 13 ? 176 : 34;
        for (std::size_t r = 0; r < rows; ++r)
          for (std::size_t b = 0; b < rb / bb; ++b) {
            std::uint8_t* p = w.data() + r * rb + b * bb;
            p[0] = h & 0xff; p[1] = h >> 8;
            if (t != 8) { p[2] = h & 0xff; p[3] = h >> 8; }
          }
      };
      fix(hg, n_exp * ff, grb, cb.gate); fix(hu, n_exp * ff, grb, cb.gate); fix(hd, n_exp * hidden, drb, cb.down);
      std::vector<float> hx((std::size_t)nt * hidden);
      for (auto& v : hx) v = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
      const std::size_t slots = (std::size_t)nt * top_k;
      std::vector<std::int32_t> hids(slots);
      for (auto& v : hids) v = rand() % n_exp;
      if (slots > 1) hids[1] = -1;  // masked slot: output untouched
      // host reference, float64
      std::vector<double> ref(slots * hidden, 0.0);
      std::vector<bool> live(slots, false);
      for (std::size_t s = 0; s < slots; ++s) {
        if (hids[s] < 0) continue;
        live[s] = true;
        const std::size_t tok = s / top_k;
        std::vector<double> act(ff);
        for (std::size_t r = 0; r < ff; ++r) {
          const std::uint8_t* gr = hg.data() + ((std::size_t)hids[s] * ff + r) * grb;
          const std::uint8_t* ur = hu.data() + ((std::size_t)hids[s] * ff + r) * grb;
          double g = 0, u = 0;
          for (std::size_t i = 0; i < hidden; ++i) { g += Element(cb.gate, gr, (std::uint32_t)i) * hx[tok * hidden + i]; u += Element(cb.gate, ur, (std::uint32_t)i) * hx[tok * hidden + i]; }
          act[r] = g / (1.0 + std::exp(-g)) * u;
        }
        for (std::size_t h = 0; h < hidden; ++h) {
          const std::uint8_t* dr = hd.data() + ((std::size_t)hids[s] * hidden + h) * drb;
          double d = 0;
          for (std::size_t i = 0; i < ff; ++i) d += Element(cb.down, dr, (std::uint32_t)i) * act[i];
          ref[s * hidden + h] = d;
        }
      }
      void *dg, *du, *dd; float *dx, *dact, *dout; std::int32_t* dids;
      CK(hipMalloc(&dg, gbytes)); CK(hipMalloc(&du, gbytes)); CK(hipMalloc(&dd, dbytes));
      CK(hipMemcpy(dg, hg.data(), gbytes, hipMemcpyHostToDevice)); CK(hipMemcpy(du, hu.data(), gbytes, hipMemcpyHostToDevice)); CK(hipMemcpy(dd, hd.data(), dbytes, hipMemcpyHostToDevice));
      CK(hipMalloc(&dx, hx.size() * 4)); CK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
      CK(hipMalloc(&dids, slots * 4)); CK(hipMemcpy(dids, hids.data(), slots * 4, hipMemcpyHostToDevice));
      CK(hipMalloc(&dact, slots * ff * 4)); CK(hipMalloc(&dout, slots * hidden * 4));
      const auto gt = (gg::WeightType)cb.gate, dt = (gg::WeightType)cb.down;
      std::vector<float> got(slots * hidden);
      for (std::uint32_t rh : {0u, 32u, 64u, 128u, 256u}) {
        const float sentinel = 12345.f;
        std::vector<float> fill(slots * hidden, sentinel);
        CK(hipMemcpy(dout, fill.data(), fill.size() * 4, hipMemcpyHostToDevice));
        if (!gg::LaunchDecodeMoeKQuant(dg, gt, du, dd, dt, dx, dids, dact, dout, nt, top_k, hidden, ff, 0, rh)) { std::printf("refused rows %u\n", rh); return 1; }
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost));
        double num = 0, den = 0; bool masked_ok = true;
        for (std::size_t s = 0; s < slots; ++s)
          for (std::size_t h = 0; h < hidden; ++h) {
            if (!live[s]) { masked_ok &= got[s * hidden + h] == sentinel; continue; }
            const double e = got[s * hidden + h] - ref[s * hidden + h];
            num += e * e; den += ref[s * hidden + h] * ref[s * hidden + h];
          }
        const double rel = std::sqrt(num / den);
        const bool ok = rel < 1e-5 && masked_ok;
        if (!ok) std::printf("  gate %d down %d tokens %u rows %u: rel-L2 %.3e masked %s FAIL\n", cb.gate, cb.down, nt, rh, rel, masked_ok ? "ok" : "TOUCHED");
        bad += !ok;
        if (rh == 0) std::printf("gate %d down %d tokens %u: rel-L2 vs float64 %.2e %s\n", cb.gate, cb.down, nt, rel, ok ? "OK" : "FAIL");
      }
      hipFree(dg); hipFree(du); hipFree(dd); hipFree(dx); hipFree(dids); hipFree(dact); hipFree(dout);
    }
  // timing at the real shape (256 experts), rotating expert ids so weights are not cache-resident
  for (auto cb : combos) {
    const std::size_t E = 256, grb = RowBytes(cb.gate, hidden), drb = RowBytes(cb.down, ff);
    void *dg, *du, *dd; float *dx, *dact, *dout; std::int32_t* dall;
    CK(hipMalloc(&dg, E * ff * grb)); CK(hipMalloc(&du, E * ff * grb)); CK(hipMalloc(&dd, E * hidden * drb));
    CK(hipMemset(dg, 1, E * ff * grb)); CK(hipMemset(du, 1, E * ff * grb)); CK(hipMemset(dd, 1, E * hidden * drb));
    CK(hipMalloc(&dx, 8 * hidden * 4)); CK(hipMemset(dx, 0, 8 * hidden * 4));
    CK(hipMalloc(&dact, 64 * ff * 4)); CK(hipMalloc(&dout, 64 * hidden * 4));
    std::vector<std::int32_t> hall(32 * 64);
    for (auto& v : hall) v = rand() % E;
    CK(hipMalloc(&dall, hall.size() * 4)); CK(hipMemcpy(dall, hall.data(), hall.size() * 4, hipMemcpyHostToDevice));
    for (std::uint32_t nt : {1u, 8u}) {
      const std::size_t slots = (std::size_t)nt * top_k;
      int it = 0;
      const double t = Time([&] { (void)gg::LaunchDecodeMoeKQuant(dg, (gg::WeightType)cb.gate, du, dd, (gg::WeightType)cb.down, dx, dall + (std::size_t)(it++ % 32) * 64, dact, dout, nt, top_k, hidden, ff, 0); });
      const double mb = (double)slots * (2.0 * ff * grb + (double)hidden * drb) / 1e6;
      std::printf("timing gate %d down %d tokens %u: %6.1f us  %5.1f GB/s\n", cb.gate, cb.down, nt, t, mb / t * 1e3);
    }
    hipFree(dg); hipFree(du); hipFree(dd); hipFree(dx); hipFree(dact); hipFree(dout); hipFree(dall);
  }
  // contract: unlisted type codes and bad widths are refused
  float* p; CK(hipMalloc(&p, 1 << 20));
  std::int32_t* ids; CK(hipMalloc(&ids, 64));
  const bool r1 = !gg::LaunchDecodeMoeKQuant(p, (gg::WeightType)99, p, p, (gg::WeightType)13, p, ids, p, p, 1, 8, 2048, 512, 0);
  const bool r2 = !gg::LaunchDecodeMoeKQuant(p, (gg::WeightType)98, p, p, (gg::WeightType)13, p, ids, p, p, 1, 8, 2048, 512, 0);
  const bool r3 = !gg::LaunchDecodeMoeKQuant(p, (gg::WeightType)12, p, p, (gg::WeightType)13, p, ids, p, p, 1, 8, 2000, 512, 0);
  const bool r4 = !gg::LaunchDecodeMoeKQuant(p, (gg::WeightType)12, p, p, (gg::WeightType)97, p, ids, p, p, 1, 8, 2048, 512, 0);
  const bool r5 = !gg::LaunchDecodeMoeKQuant(p, (gg::WeightType)12, p, p, (gg::WeightType)13, p, ids, p, p, 1, 8, 2048, 512, 0, 96);
  std::printf("refuse unregistered gate 99 %s, gate 98 %s, bad width %s, unregistered down 97 %s, bad rows hint %s\n", r1 ? "ok" : "FAIL", r2 ? "ok" : "FAIL", r3 ? "ok" : "FAIL", r4 ? "ok" : "FAIL", r5 ? "ok" : "FAIL");
  bad += !(r1 && r2 && r3 && r4 && r5);
  std::printf(bad ? "FAILED\n" : "all OK\n");
  return bad ? 1 : 0;
}
