// QuantGemvMulti (2..32 tokens, Q4_K / Q5_K) vs a float reference on the exact dequantized weights, plus timing against
// the per-token GEMV loop that GemvDecode used. usage: nw_test [type=12|13]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"

using namespace gufo::core::compute::gemm;
#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

__global__ void RefGemv(const __half* W, const float* X, float* Y, int n, int m, int k) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
  if (r >= m) return;
  double s = 0;
  for (int i = 0; i < k; ++i) s += (double)__half2float(W[(size_t)r * k + i]) * (double)X[(size_t)t * k + i];
  Y[(size_t)t * m + r] = (float)s;
}

template <class Fn> static double Time(Fn fn, int iters = 20) {
  for (int i = 0; i < 3; ++i) fn();
  CK(hipDeviceSynchronize());
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  CK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
  float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms / iters;
}

int main(int argc, char** argv) {
  const int type = argc > 1 ? std::atoi(argv[1]) : 12;
  const size_t bb = type == 12 ? 144 : 176;
  struct S { size_t m, k; const char* what; } shapes[] = {{19968, 6656, "ffn gate/up"}, {6656, 19968, "ffn down"}, {4096, 6656, "attn q/gate"},
                                                          {6656, 4096, "attn out"}, {256, 6656, "attn k/v"}, {202048, 6656, "output"}};
  int bad = 0;
  for (auto s : shapes) {
    const size_t row_bytes = s.k / 256 * bb;
    std::vector<uint8_t> wq(s.m * row_bytes);
    srand(5);
    for (auto& b : wq) b = (uint8_t)rand();
    const uint16_t h = 0x211e;
    for (size_t r = 0; r < s.m; ++r) for (size_t blk = 0; blk < s.k / 256; ++blk) { uint8_t* p = wq.data() + r * row_bytes + blk * bb; p[0] = h & 0xff; p[1] = h >> 8; p[2] = h & 0xff; p[3] = h >> 8; }
    void* dw; CK(hipMalloc(&dw, wq.size())); CK(hipMemcpy(dw, wq.data(), wq.size(), hipMemcpyHostToDevice));
    __half* dwh; CK(hipMalloc(&dwh, s.m * s.k * 2));
    if (!DequantRowsF16(dw, type, dwh, s.m, s.k, 0)) { std::printf("dequant failed\n"); return 1; }
    std::printf("%zu x %zu  %s  (%.1f MB packed)\n", s.m, s.k, s.what, wq.size() / 1e6);
    for (int n : {2, 4, 8, 16, 21, 32}) {
      std::vector<float> xh((size_t)n * s.k);
      for (auto& v : xh) v = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
      float *dx, *y, *yr, *yo; CK(hipMalloc(&dx, xh.size() * 4)); CK(hipMemcpy(dx, xh.data(), xh.size() * 4, hipMemcpyHostToDevice));
      CK(hipMalloc(&y, (size_t)n * s.m * 4)); CK(hipMalloc(&yr, (size_t)n * s.m * 4)); CK(hipMalloc(&yo, (size_t)n * s.m * 4));
      CK(hipMemset(y, 0xFF, (size_t)n * s.m * 4));
      if (!QuantGemvMulti(dw, type, dx, y, n, s.m, s.k, 0)) { std::printf("  n=%d launch refused\n", n); ++bad; continue; }
      CK(hipDeviceSynchronize());
      const int rt = n < 8 ? n : 8;
      RefGemv<<<dim3((s.m + 255) / 256, rt), 256>>>(dwh, dx, yr, n, (int)s.m, (int)s.k);
      CK(hipDeviceSynchronize());
      std::vector<float> a((size_t)rt * s.m), r(a.size());
      CK(hipMemcpy(a.data(), y, a.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(r.data(), yr, r.size() * 4, hipMemcpyDeviceToHost));
      double num = 0, den = 0; for (size_t i = 0; i < a.size(); ++i) { const double e = a[i] - r[i]; num += e * e; den += (double)r[i] * r[i]; }
      const double rel = std::sqrt(num / den);
      const double t_new = Time([&] { (void)QuantGemvMulti(dw, type, dx, y, n, s.m, s.k, 0); });
      const double t_old = Time([&] { for (int t = 0; t < n; ++t) LaunchQ8KBlockGEMV(dw, static_cast<gufo::core::GgmlType>(type), dx + (size_t)t * s.k, yo + (size_t)t * s.m, s.m, s.k, 0); }, 5);
      const bool ok = rel < 3e-3 && std::isfinite(rel);
      std::printf("  n=%2d  rel-L2 %.2e %s   new %7.3f ms (%5.1f GB/s)   per-token loop %7.3f ms   x%.1f\n", n, rel, ok ? "OK" : "FAIL", t_new, wq.size() / (t_new * 1e-3) / 1e9, t_old, t_old / t_new);
      bad += !ok;
      hipFree(dx); hipFree(y); hipFree(yr); hipFree(yo);
    }
    hipFree(dw); hipFree(dwh);
  }
  std::printf(bad ? "FAILED\n" : "all OK\n");
  return bad ? 1 : 0;
}