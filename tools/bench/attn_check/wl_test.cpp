// flash_wmma_large (head_dim 256 / 512, any GQA ratio, window): accuracy against a double-precision host reference on random
// F16 data, and throughput. Built by tools/bench/attn_check/wl_ab.sh.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "src/core/compute/attention/flash_wmma_large.hpp"

namespace fw = mbufo::core::compute::flash_wmma_large;
#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

template <class Fn>
static double Time(Fn fn, int iters = 5) {
  for (int i = 0; i < 2; ++i) fn();
  CK(hipDeviceSynchronize());
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  CK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
  float ms; CK(hipEventElapsedTime(&ms, a, b));
  return ms * 1000.0 / iters;
}

int main() {
  std::mt19937 rng(21);
  int bad = 0;
  struct Case { std::uint32_t heads, kv, d, n, start, window; float scale; };
  const Case cases[] = {
      {8, 2, 512, 150, 0, 0, 1.0f},    {8, 2, 512, 150, 300, 0, 1.0f},  {16, 2, 512, 100, 40, 0, 1.0f},  {8, 1, 512, 130, 0, 0, 1.0f},
      {8, 2, 256, 150, 0, 0, 1.0f},    {8, 2, 256, 150, 300, 128, 1.0f}, {16, 2, 256, 100, 40, 0, 0.0625f}, {16, 8, 256, 100, 0, 0, 1.0f},
      {4, 4, 256, 70, 0, 0, 1.0f},     {24, 4, 256, 90, 10, 0, 0.0625f}, {8, 2, 256, 33, 5, 0, 1.0f},     {8, 2, 512, 17, 0, 0, 1.0f}};
  for (const auto& c : cases) {
    const std::uint32_t kvlen = c.start + c.n;
    std::vector<__half> hk(static_cast<std::size_t>(kvlen) * c.kv * c.d), hv(hk.size());
    for (auto& x : hk) x = __float2half(static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 900.0f);
    for (auto& x : hv) x = __float2half(static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 900.0f);
    std::vector<float> hq(static_cast<std::size_t>(c.n) * c.heads * c.d);
    for (auto& x : hq) x = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1500.0f;
    __half *dk, *dv; float *dq, *dout;
    CK(hipMalloc(&dk, hk.size() * 2)); CK(hipMalloc(&dv, hv.size() * 2)); CK(hipMalloc(&dq, hq.size() * 4)); CK(hipMalloc(&dout, hq.size() * 4));
    CK(hipMemcpy(dk, hk.data(), hk.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dv, hv.data(), hv.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice)); CK(hipMemset(dout, 0, hq.size() * 4));
    if (!fw::FlashWmmaLargePrefill(dq, dk, dv, dout, c.n, c.start, c.heads, c.kv, c.d, c.window, c.scale, 0)) { std::printf("  refused h%u kv%u d%u\n", c.heads, c.kv, c.d); ++bad; continue; }
    CK(hipDeviceSynchronize());
    std::vector<float> o(hq.size()); CK(hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost));
    double worst = 0, scale_ref = 0;
    const std::uint32_t qpg = c.heads / c.kv;
    for (std::uint32_t t = 0; t < c.n; t += (c.n > 64 ? 7 : 1))
      for (std::uint32_t h = 0; h < c.heads; ++h) {
        const std::uint32_t kvh = h / qpg, qpos = c.start + t;
        const std::uint32_t lo = (c.window != 0 && qpos + 1 > c.window) ? qpos + 1 - c.window : 0;
        std::vector<double> sc; double mx = -1e300;
        for (std::uint32_t j = lo; j <= qpos; ++j) {
          double a = 0;
          for (std::uint32_t i = 0; i < c.d; ++i) a += static_cast<double>(hq[(static_cast<std::size_t>(t) * c.heads + h) * c.d + i]) * __half2float(hk[(static_cast<std::size_t>(j) * c.kv + kvh) * c.d + i]);
          sc.push_back(a * c.scale); mx = std::max(mx, sc.back());
        }
        double z = 0; std::vector<double> acc(c.d, 0.0);
        for (std::uint32_t j = lo; j <= qpos; ++j) {
          const double w = std::exp(sc[j - lo] - mx); z += w;
          for (std::uint32_t i = 0; i < c.d; ++i) acc[i] += w * __half2float(hv[(static_cast<std::size_t>(j) * c.kv + kvh) * c.d + i]);
        }
        for (std::uint32_t i = 0; i < c.d; ++i) {
          const double r = acc[i] / z;
          worst = std::max(worst, std::fabs(r - o[(static_cast<std::size_t>(t) * c.heads + h) * c.d + i]));
          scale_ref = std::max(scale_ref, std::fabs(r));
        }
      }
    const bool ok = worst <= 4e-3 * std::max(scale_ref, 1e-3);
    std::printf("  h%-2u kv%u d%u n%-4u start%-4u window%-3u: max|err| %.3g (max|ref| %.3g) %s\n", c.heads, c.kv, c.d, c.n, c.start, c.window, worst, scale_ref, ok ? "ok" : "FAIL");
    if (!ok) ++bad;
    CK(hipFree(dk)); CK(hipFree(dv)); CK(hipFree(dq)); CK(hipFree(dout));
  }
  // blocked-Q8 variant: identical to the F16 kernel over the dequantized planes (bit for bit)
  for (const auto& c : {Case{8, 2, 256, 150, 300, 128, 1.0f}, Case{8, 2, 256, 90, 0, 0, 1.0f}, Case{8, 2, 512, 60, 20, 0, 1.0f}}) {
    const std::uint32_t kvlen = c.start + c.n, nb = c.d / 32;
    std::vector<std::uint8_t> bk(static_cast<std::size_t>(kvlen) * c.kv * nb * 34), bv(bk.size());
    std::vector<__half> fk(static_cast<std::size_t>(kvlen) * c.kv * c.d), fv(fk.size());
    auto fill = [&](std::vector<std::uint8_t>& blocks, std::vector<__half>& plane) {
      for (std::size_t b = 0; b < static_cast<std::size_t>(kvlen) * c.kv * nb; ++b) {
        std::uint8_t* p = blocks.data() + b * 34;
        const __half sc = __float2half(0.005f + 0.01f * static_cast<float>(rng() % 100) / 100.0f);
        std::memcpy(p + 32, &sc, 2);
        for (int i = 0; i < 32; ++i) {
          p[i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(rng() % 255) - 127));
          plane[b * 32 + i] = __float2half(static_cast<float>(static_cast<std::int8_t>(p[i])) * __half2float(sc));
        }
      }
    };
    fill(bk, fk); fill(bv, fv);
    std::vector<float> hq(static_cast<std::size_t>(c.n) * c.heads * c.d);
    for (auto& x : hq) x = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1500.0f;
    std::uint8_t *dbk, *dbv; __half *dfk, *dfv; float *dq, *o1, *o2;
    CK(hipMalloc(&dbk, bk.size())); CK(hipMalloc(&dbv, bv.size())); CK(hipMalloc(&dfk, fk.size() * 2)); CK(hipMalloc(&dfv, fv.size() * 2));
    CK(hipMalloc(&dq, hq.size() * 4)); CK(hipMalloc(&o1, hq.size() * 4)); CK(hipMalloc(&o2, hq.size() * 4));
    CK(hipMemcpy(dbk, bk.data(), bk.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dbv, bv.data(), bv.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dfk, fk.data(), fk.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dfv, fv.data(), fv.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
    const bool a = fw::FlashWmmaLargePrefill(dq, dfk, dfv, o1, c.n, c.start, c.heads, c.kv, c.d, c.window, c.scale, 0);
    const bool b = fw::FlashWmmaLargePrefillQ8(dq, dbk, dbv, o2, c.n, c.start, c.heads, c.kv, c.d, c.window, c.scale, 0);
    CK(hipDeviceSynchronize());
    std::vector<float> r1(hq.size()), r2(hq.size());
    CK(hipMemcpy(r1.data(), o1, r1.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(r2.data(), o2, r2.size() * 4, hipMemcpyDeviceToHost));
    const bool same = a && b && std::memcmp(r1.data(), r2.data(), r1.size() * 4) == 0;
    std::printf("  q8 blocks vs dequantized F16 planes: h%u kv%u d%u n%u start%u window%u: %s\n", c.heads, c.kv, c.d, c.n, c.start, c.window, same ? "bit-identical" : "DIFFERENT");
    if (!same) ++bad;
    CK(hipFree(dbk)); CK(hipFree(dbv)); CK(hipFree(dfk)); CK(hipFree(dfv)); CK(hipFree(dq)); CK(hipFree(o1)); CK(hipFree(o2));
  }
  // throughput
  struct T { std::uint32_t heads, kv, d, n; const char* name; } ts[] = {
      {8, 2, 512, 2048, "E4B global  (8/2, d512)"}, {16, 2, 512, 2048, "Gemma26B global (16/2, d512)"}, {16, 2, 256, 2048, "qwen35moe (16/2, d256)"},
      {24, 4, 256, 2048, "qwen 27B (24/4, d256)"},  {8, 2, 256, 2048, "E4B swa-as-dense (8/2, d256)"}};
  for (auto t : ts) {
    const std::size_t kvn = t.n;
    __half *dk, *dv; float *dq, *dout;
    CK(hipMalloc(&dk, kvn * t.kv * t.d * 2)); CK(hipMalloc(&dv, kvn * t.kv * t.d * 2)); CK(hipMalloc(&dq, kvn * t.heads * t.d * 4)); CK(hipMalloc(&dout, kvn * t.heads * t.d * 4));
    CK(hipMemset(dk, 0x3c, kvn * t.kv * t.d * 2)); CK(hipMemset(dv, 0x3c, kvn * t.kv * t.d * 2)); CK(hipMemset(dq, 0, kvn * t.heads * t.d * 4));
    const double us = Time([&] { (void)fw::FlashWmmaLargePrefill(dq, dk, dv, dout, t.n, 0, t.heads, t.kv, t.d, 0, 1.0f, 0); });
    const double flops = 4.0 * t.heads * t.d * (static_cast<double>(t.n) * t.n / 2.0);
    std::printf("  %-30s n=%u: %8.0f us  %5.1f TFLOPS (causal)\n", t.name, t.n, us, flops / us / 1e6);
    CK(hipFree(dk)); CK(hipFree(dv)); CK(hipFree(dq)); CK(hipFree(dout));
  }
  std::printf("%s\n", bad ? "FAILED" : "all OK");
  return bad ? 1 : 0;
}
