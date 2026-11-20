// QuantGemvSingle vs the existing block GEMV (LaunchQ8KBlockGEMV) at n = 1: agreement and weight bandwidth.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "src/core/compute/gemm/gemm.hpp"
using namespace mbufo::core::compute::gemm;
#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)
template <class Fn> static double Time(Fn fn, int iters = 40) {
  for (int i = 0; i < 5; ++i) fn();
  CK(hipDeviceSynchronize());
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) fn(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
  float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms / iters;
}
int main(int argc, char** argv) {
  const int type = argc > 1 ? std::atoi(argv[1]) : 12;
  const size_t bb = type == 12 ? 144 : 176;
  struct S { size_t m, k; const char* what; } shapes[] = {{19968, 6656, "ffn gate/up"}, {6656, 19968, "ffn down"}, {4096, 6656, "attn q/gate"}, {6656, 4096, "attn out"}, {256, 6656, "attn k/v"}, {202048, 6656, "output"}};
  int bad = 0;
  for (auto s : shapes) {
    const size_t rb = s.k / 256 * bb;
    std::vector<uint8_t> wq(s.m * rb); srand(5); for (auto& b : wq) b = (uint8_t)rand();
    for (size_t r = 0; r < s.m; ++r) for (size_t blk = 0; blk < s.k / 256; ++blk) { uint8_t* p = wq.data() + r * rb + blk * bb; p[0] = 0x1e; p[1] = 0x21; p[2] = 0x1e; p[3] = 0x21; }
    void* dw; CK(hipMalloc(&dw, wq.size())); CK(hipMemcpy(dw, wq.data(), wq.size(), hipMemcpyHostToDevice));
    std::vector<float> xh(s.k); for (auto& v : xh) v = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
    float *dx, *y1, *y2; CK(hipMalloc(&dx, s.k * 4)); CK(hipMemcpy(dx, xh.data(), s.k * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&y1, s.m * 4)); CK(hipMalloc(&y2, s.m * 4));
    LaunchQ8KBlockGEMV(dw, static_cast<mbufo::core::GgmlType>(type), dx, y1, s.m, s.k, 0);
    if (!QuantGemvSingle(dw, type, dx, y2, s.m, s.k, 0)) { std::printf("refused\n"); return 1; }
    CK(hipDeviceSynchronize());
    std::vector<float> a(s.m), b(s.m); CK(hipMemcpy(a.data(), y1, s.m * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(b.data(), y2, s.m * 4, hipMemcpyDeviceToHost));
    double num = 0, den = 0; for (size_t i = 0; i < s.m; ++i) { const double e = a[i] - b[i]; num += e * e; den += (double)a[i] * a[i]; }
    const double rel = std::sqrt(num / den);
    // cold-ish: rotate over 4 copies so the weights are not L2/MALL resident for the small shapes
    const double t_old = Time([&] { LaunchQ8KBlockGEMV(dw, static_cast<mbufo::core::GgmlType>(type), dx, y1, s.m, s.k, 0); });
    const double t_new = Time([&] { (void)QuantGemvSingle(dw, type, dx, y2, s.m, s.k, 0); });
    const bool ok = rel < 1e-4;
    std::printf("%-12s %6zu x %6zu  rel %.1e %s   old %7.1f us (%5.1f GB/s)   new %7.1f us (%5.1f GB/s)  x%.2f\n", s.what, s.m, s.k, rel, ok ? "OK" : "FAIL", t_old * 1e3, wq.size() / (t_old * 1e-3) / 1e9, t_new * 1e3, wq.size() / (t_new * 1e-3) / 1e9, t_old / t_new);
    bad += !ok;
    hipFree(dw); hipFree(dx); hipFree(y1); hipFree(y2);
  }
  std::printf(bad ? "FAILED\n" : "all OK\n");
  return bad ? 1 : 0;
}