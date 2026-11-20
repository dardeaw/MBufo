// Lane-geometry decode MoE (core/compute/gemm/moe_decode_wave.hip) vs the existing gemma4 kernels (bit-exact for
// Q4_0, GeLU-tanh, merged gate/up, masked slots) and vs a float64 reference (Q8_0, SiLU), with timing at gemma4
// shapes. Links the core gemm library (MoeVecGateUpSwiglu / GemvQSlot live in moe_fused.hip):
//   hipcc -O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I<repo> -x hip mdk_wave_test.cpp \
//         -x none <build>/.../libmbufo_core_gemm.a -o mdk_wave_test && ./mdk_wave_test
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
static void FixScales(std::vector<std::uint8_t>& w, std::size_t rows, std::size_t rb, int t) {
  const std::uint16_t h = 0x1c00;  // small fp16 block scale
  const std::size_t bb = t == 2 ? 18 : 34;
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t b = 0; b < rb / bb; ++b) {
      std::uint8_t* p = w.data() + r * rb + b * bb;
      p[0] = h & 0xff; p[1] = h >> 8;
    }
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
  int bad = 0;
  const std::uint32_t hidden = 2816, ff = 704, n_exp = 128, used = 8;
  // ---- Q4_0 / GeLU-tanh: bit-exact vs the existing kernels ----
  {
    const std::uint64_t grb = RowBytes(2, hidden), drb = RowBytes(2, ff);
    const std::uint64_t gstride = 2ull * ff * grb, dstride = (std::uint64_t)hidden * drb;
    std::vector<std::uint8_t> hg(n_exp * gstride), hd(n_exp * dstride);
    srand(5);
    for (auto& v : hg) v = (std::uint8_t)rand();
    for (auto& v : hd) v = (std::uint8_t)rand();
    FixScales(hg, n_exp * 2ull * ff, grb, 2); FixScales(hd, (std::size_t)n_exp * hidden, drb, 2);
    std::vector<float> hx(hidden);
    for (auto& v : hx) v = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
    void *dg, *dd; float *dx, *dact_old, *dact_new, *dout_old, *dout_new; std::int32_t* dids;
    CK(hipMalloc(&dg, hg.size())); CK(hipMalloc(&dd, hd.size()));
    CK(hipMemcpy(dg, hg.data(), hg.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dd, hd.data(), hd.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&dx, hidden * 4)); CK(hipMemcpy(dx, hx.data(), hidden * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dids, used * 4));
    CK(hipMalloc(&dact_old, used * ff * 4)); CK(hipMalloc(&dact_new, used * ff * 4));
    CK(hipMalloc(&dout_old, used * hidden * 4)); CK(hipMalloc(&dout_new, used * hidden * 4));
    for (int trial = 0; trial < 3; ++trial) {
      std::vector<std::int32_t> ids(used);
      for (auto& v : ids) v = rand() % n_exp;
      if (trial == 1) ids[2] = -1;  // masked slot: both paths write zeros
      CK(hipMemcpy(dids, ids.data(), used * 4, hipMemcpyHostToDevice));
      gg::MoeVecGateUpSwiglu(dg, grb, gstride, dx, dids, dact_old, used, used, ff, hidden, 0);
      gg::GemvQSlot(dd, drb, dstride, dact_old, dids, dout_old, used, hidden, ff, 0);
      const bool ok1 = gg::LaunchMoeWaveGateUp(dg, gg::WeightType::kQ4_0, grb, gstride, dx, dids, dact_new, used, used, ff, hidden, gg::MoeGateAct::kGeluTanhMul, 0);
      const bool ok2 = gg::LaunchMoeWaveDown(dd, gg::WeightType::kQ4_0, drb, dstride, dact_new, dids, dout_new, used, hidden, ff, 0);
      CK(hipDeviceSynchronize());
      std::vector<float> a(used * ff), b(used * ff), c(used * (std::size_t)hidden), d(used * (std::size_t)hidden);
      CK(hipMemcpy(a.data(), dact_old, a.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(b.data(), dact_new, b.size() * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(c.data(), dout_old, c.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(d.data(), dout_new, d.size() * 4, hipMemcpyDeviceToHost));
      const bool same = ok1 && ok2 && !std::memcmp(a.data(), b.data(), a.size() * 4) && !std::memcmp(c.data(), d.data(), c.size() * 4);
      std::printf("Q4_0 gemma shape trial %d%s: %s\n", trial, trial == 1 ? " (masked slot)" : "", same ? "bit-exact" : "MISMATCH");
      bad += !same;
    }
    // timing (rotating ids so experts differ)
    std::vector<std::int32_t> all(32 * used);
    for (auto& v : all) v = rand() % n_exp;
    std::int32_t* dall; CK(hipMalloc(&dall, all.size() * 4)); CK(hipMemcpy(dall, all.data(), all.size() * 4, hipMemcpyHostToDevice));
    int it = 0;
    const double told = Time([&] {
      const std::int32_t* ids = dall + (it++ % 32) * used;
      gg::MoeVecGateUpSwiglu(dg, grb, gstride, dx, ids, dact_old, used, used, ff, hidden, 0);
      gg::GemvQSlot(dd, drb, dstride, dact_old, ids, dout_old, used, hidden, ff, 0);
    });
    it = 0;
    const double tnew = Time([&] {
      const std::int32_t* ids = dall + (it++ % 32) * used;
      (void)gg::LaunchMoeWaveGateUp(dg, gg::WeightType::kQ4_0, grb, gstride, dx, ids, dact_new, used, used, ff, hidden, gg::MoeGateAct::kGeluTanhMul, 0);
      (void)gg::LaunchMoeWaveDown(dd, gg::WeightType::kQ4_0, drb, dstride, dact_new, ids, dout_new, used, hidden, ff, 0);
    });
    std::printf("timing Q4_0 gate_up+down, 1 token: existing %.1f us, framework %.1f us (x%.2f)\n", told, tnew, told / tnew);
    hipFree(dg); hipFree(dd); hipFree(dx); hipFree(dids); hipFree(dact_old); hipFree(dact_new); hipFree(dout_old); hipFree(dout_new); hipFree(dall);
  }
  for (int ft : {2, 8}) {
  // ---- every lane format, SiLU: float64 reference ----
  {
    const std::uint32_t h8 = 2048, f8 = 512, e8 = 16, u8 = 8;
    const std::uint64_t grb = RowBytes(ft, h8), drb = RowBytes(ft, f8);
    const std::uint64_t gstride = 2ull * f8 * grb, dstride = (std::uint64_t)h8 * drb;
    std::vector<std::uint8_t> hg(e8 * gstride), hd(e8 * dstride);
    srand(9);
    for (auto& v : hg) v = (std::uint8_t)rand();
    for (auto& v : hd) v = (std::uint8_t)rand();
    FixScales(hg, e8 * 2ull * f8, grb, ft); FixScales(hd, (std::size_t)e8 * h8, drb, ft);
    std::vector<float> hx(h8);
    for (auto& v : hx) v = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
    std::vector<std::int32_t> ids(u8);
    for (auto& v : ids) v = rand() % e8;
    ids[1] = -1;
    std::vector<double> ref(u8 * (std::size_t)h8, 0.0);
    for (std::uint32_t s = 0; s < u8; ++s) {
      if (ids[s] < 0) continue;
      std::vector<double> act(f8);
      for (std::uint32_t r = 0; r < f8; ++r) {
        const std::uint8_t* gr = hg.data() + (std::size_t)ids[s] * gstride + (std::size_t)r * grb;
        const std::uint8_t* ur = hg.data() + (std::size_t)ids[s] * gstride + (std::size_t)(f8 + r) * grb;
        double g = 0, u = 0;
        for (std::uint32_t i = 0; i < h8; ++i) { g += Element(ft, gr, i) * hx[i]; u += Element(ft, ur, i) * hx[i]; }
        act[r] = g / (1.0 + std::exp(-g)) * u;
      }
      for (std::uint32_t h = 0; h < h8; ++h) {
        const std::uint8_t* dr = hd.data() + (std::size_t)ids[s] * dstride + (std::size_t)h * drb;
        double d = 0;
        for (std::uint32_t i = 0; i < f8; ++i) d += Element(ft, dr, i) * act[i];
        ref[s * (std::size_t)h8 + h] = d;
      }
    }
    void *dg, *dd; float *dx, *dact, *dout; std::int32_t* dids;
    CK(hipMalloc(&dg, hg.size())); CK(hipMalloc(&dd, hd.size()));
    CK(hipMemcpy(dg, hg.data(), hg.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dd, hd.data(), hd.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&dx, h8 * 4)); CK(hipMemcpy(dx, hx.data(), h8 * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dids, u8 * 4)); CK(hipMemcpy(dids, ids.data(), u8 * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dact, u8 * f8 * 4)); CK(hipMalloc(&dout, u8 * (std::size_t)h8 * 4));
    const bool ok1 = gg::LaunchMoeWaveGateUp(dg, (gg::WeightType)ft, grb, gstride, dx, dids, dact, u8, u8, f8, h8, gg::MoeGateAct::kSiluMul, 0);
    const bool ok2 = gg::LaunchMoeWaveDown(dd, (gg::WeightType)ft, drb, dstride, dact, dids, dout, u8, h8, f8, 0);
    CK(hipDeviceSynchronize());
    std::vector<float> got(u8 * (std::size_t)h8);
    CK(hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost));
    double num = 0, den = 0; bool zero_ok = true;
    for (std::uint32_t s = 0; s < u8; ++s)
      for (std::uint32_t h = 0; h < h8; ++h) {
        const double g = got[s * (std::size_t)h8 + h], r = ref[s * (std::size_t)h8 + h];
        if (ids[s] < 0) { zero_ok &= g == 0.0; continue; }
        num += (g - r) * (g - r); den += r * r;
      }
    const double rel = std::sqrt(num / den);
    const bool ok = ok1 && ok2 && rel < 1e-5 && zero_ok;
    std::printf("type %d SiLU vs float64: rel-L2 %.2e, masked slot zero %s: %s\n", ft, rel, zero_ok ? "yes" : "NO", ok ? "OK" : "FAIL");
    bad += !ok;
    // refusal paths
    const bool r1 = !gg::LaunchMoeWaveGateUp(dg, (gg::WeightType)99, grb, gstride, dx, dids, dact, u8, u8, f8, h8, gg::MoeGateAct::kSiluMul, 0);
    const bool r2 = !gg::LaunchMoeWaveGateUp(dg, (gg::WeightType)ft, grb + 2, gstride, dx, dids, dact, u8, u8, f8, h8, gg::MoeGateAct::kSiluMul, 0);
    const bool r3 = !gg::LaunchMoeWaveDown(dd, (gg::WeightType)98, drb, dstride, dact, dids, dout, u8, h8, f8, 0);
    const bool r4 = !gg::LaunchMoeWaveGateUp(dg, (gg::WeightType)ft, grb, gstride, dx, dids, dact, 5, u8, f8, h8, gg::MoeGateAct::kSiluMul, 0);
    std::printf("refuse unregistered gate 99 %s, wrong row_bytes %s, unregistered down 98 %s, slots %% used != 0 %s\n", r1 ? "ok" : "FAIL", r2 ? "ok" : "FAIL", r3 ? "ok" : "FAIL", r4 ? "ok" : "FAIL");
    bad += !(r1 && r2 && r3 && r4);
  }
  }
  std::printf(bad ? "FAILED\n" : "all OK\n");
  return bad ? 1 : 0;
}
