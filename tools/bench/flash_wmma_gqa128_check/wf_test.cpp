// FlashWmmaGqa128Prefill vs float64 reference (rel-L2) and, with a 2nd argument, timing at Glimmer's shape.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/core/compute/attention/flash_wmma_gqa128.hpp"

using namespace gufo::core::compute::flash_wmma_gqa128;

#define CK(x) do { hipError_t err__ = (x); if (err__ != hipSuccess) { std::fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(err__), __LINE__); return 2; } } while (0)

static std::vector<char> ReadAll(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
  const std::string dir = argv[1];
  const bool timing = argc > 2;
  std::ifstream meta(dir + "/meta.txt");
  std::string name; unsigned heads, kvh, n, start, window;
  int bad = 0, total = 0;
  while (meta >> name >> heads >> kvh >> n >> start >> window) {
    auto qb = ReadAll(dir + "/" + name + ".q"), kb = ReadAll(dir + "/" + name + ".k"),
         vb = ReadAll(dir + "/" + name + ".v"), rb = ReadAll(dir + "/" + name + ".ref");
    float *d_q, *d_o; __half *d_k, *d_v;
    CK(hipMalloc(&d_q, qb.size())); CK(hipMemcpy(d_q, qb.data(), qb.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&d_k, kb.size())); CK(hipMemcpy(d_k, kb.data(), kb.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&d_v, vb.size())); CK(hipMemcpy(d_v, vb.data(), vb.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&d_o, rb.size())); CK(hipMemset(d_o, 0xFF, rb.size()));
    const float scale = 1.0f / std::sqrt(128.0f);
    const bool ran = FlashWmmaGqa128Prefill(d_q, d_k, d_v, d_o, n, start, heads, kvh, 128, window, scale, 0);
    CK(hipDeviceSynchronize());
    std::vector<float> o(rb.size() / 4);
    CK(hipMemcpy(o.data(), d_o, rb.size(), hipMemcpyDeviceToHost));
    const float* ref = reinterpret_cast<const float*>(rb.data());
    double num = 0, den = 0, maxe = 0;
    for (size_t i = 0; i < o.size(); ++i) { const double e = o[i] - ref[i]; num += e * e; den += double(ref[i]) * ref[i]; if (std::fabs(e) > maxe) maxe = std::fabs(e); }
    const double rel = std::sqrt(num / den);
    const bool ok = ran && rel < 5e-3 && std::isfinite(rel);
    char tbuf[96] = "";
    if (timing && ok && n >= 512) {
      hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
      for (int i = 0; i < 5; ++i) (void)FlashWmmaGqa128Prefill(d_q, d_k, d_v, d_o, n, start, heads, kvh, 128, window, scale, 0);
      hipDeviceSynchronize();
      hipEventRecord(a);
      for (int i = 0; i < 20; ++i) (void)FlashWmmaGqa128Prefill(d_q, d_k, d_v, d_o, n, start, heads, kvh, 128, window, scale, 0);
      hipEventRecord(b); hipEventSynchronize(b);
      float ms; hipEventElapsedTime(&ms, a, b); ms /= 20;
      double keys = 0;
      for (unsigned t = 0; t < n; ++t) { const unsigned qp = start + t; const unsigned lo = window ? (qp + 1 > window ? qp + 1 - window : 0) : 0; keys += qp - lo + 1; }
      const double flops = 4.0 * heads * keys * 128.0;
      std::snprintf(tbuf, sizeof tbuf, "  %7.3f ms  %5.1f TFLOPS", ms, flops / (ms * 1e-3) / 1e12);
    }
    std::printf("%-4s heads=%2u kvh=%u n=%3u start=%4u window=%3u  rel-L2 %.2e max-abs %.2e  %s%s\n", name.c_str(), heads, kvh, n, start, window, rel, maxe, ok ? "OK" : "FAIL", tbuf);
    ++total; if (!ok) ++bad;
    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_o);
  }
  std::printf("%d/%d cases OK\n", total - bad, total);
  return bad ? 1 : 0;
}
