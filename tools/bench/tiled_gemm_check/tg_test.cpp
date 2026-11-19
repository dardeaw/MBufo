// Tiled / fused prefill GEMM vs the row-major reference path, on random packed weights, with timing at Glimmer shapes.
//   tg_test [type=12]   (12 = Q4_K, 13 = Q5_K, 14 = Q6_K)
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

__global__ void RefGemm(const __half* W, const __half* X, float* Y, int batch, int m, int k) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
  if (r >= m) return;
  float s = 0;
  for (int i = 0; i < k; ++i) s += __half2float(W[(size_t)r * k + i]) * __half2float(X[(size_t)t * k + i]);
  Y[(size_t)t * m + r] = s;
}

template <class Fn>
static double Time(Fn fn, int iters = 6) {
  for (int i = 0; i < 2; ++i) fn();
  CK(hipDeviceSynchronize());
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  CK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
  float ms; CK(hipEventElapsedTime(&ms, a, b));
  return ms / iters;
}

static double RelL2(const std::vector<float>& a, const std::vector<float>& r) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) { const double e = a[i] - r[i]; num += e * e; den += (double)r[i] * r[i]; }
  return std::sqrt(num / den);
}

int main(int argc, char** argv) {
  const int type = argc > 1 ? std::atoi(argv[1]) : 12;
  const size_t block_elems = 256, block_bytes = type == 12 ? 144 : type == 13 ? 176 : 210;
  struct S { size_t batch, m, k; const char* what; } shapes[] = {
      {2048, 19968, 6656, "ffn gate/up"}, {2048, 6656, 19968, "ffn down"}, {2048, 4096, 6656, "q proj"}, {2048, 6656, 4096, "wo"},
      {300, 4096, 6656, "q proj small batch"}};
  int bad = 0;
  for (auto s : shapes) {
    const size_t row_bytes = s.k / block_elems * block_bytes;
    std::vector<uint8_t> wq(s.m * row_bytes);
    srand(7);
    for (auto& b : wq) b = (uint8_t)rand();
    const uint16_t h = 0x211e;  // ~0.0106 half
    for (size_t r = 0; r < s.m; ++r)
      for (size_t blk = 0; blk < s.k / block_elems; ++blk) {
        uint8_t* p = wq.data() + r * row_bytes + blk * block_bytes;
        if (type == 14) { p[208] = h & 0xff; p[209] = h >> 8; } else { p[0] = h & 0xff; p[1] = h >> 8; p[2] = h & 0xff; p[3] = h >> 8; }
      }
    std::vector<__half> xh(s.batch * s.k);
    for (auto& v : xh) v = __float2half((rand() / (float)RAND_MAX - 0.5f) * 2.f);
    void* dw; __half *dx, *dwr, *dwt; float *y_ref, *y_a, *y_b;
    CK(hipMalloc(&dw, wq.size())); CK(hipMemcpy(dw, wq.data(), wq.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&dx, xh.size() * 2)); CK(hipMemcpy(dx, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
    CK(hipMalloc(&dwr, s.m * s.k * 2)); CK(hipMalloc(&dwt, s.m * s.k * 2));
    CK(hipMalloc(&y_ref, s.batch * s.m * 4)); CK(hipMalloc(&y_a, s.batch * s.m * 4)); CK(hipMalloc(&y_b, s.batch * s.m * 4));
    if (!DequantRowsF16(dw, type, dwr, s.m, s.k, 0)) { std::printf("dequant failed\n"); return 1; }
    const int ref_tokens = 48;
    RefGemm<<<dim3((s.m + 255) / 256, ref_tokens), 256>>>(dwr, dx, y_ref, ref_tokens, (int)s.m, (int)s.k);
    CK(hipDeviceSynchronize());
    std::vector<float> ref((size_t)ref_tokens * s.m), got(ref.size());
    CK(hipMemcpy(ref.data(), y_ref, ref.size() * 4, hipMemcpyDeviceToHost));

    auto old_path = [&] { (void)UnquantizedF16Gemm(dwr, dx, y_a, s.batch, s.m, s.k, 0); };
    auto tiled_path = [&] { (void)DequantRowsF16Tiled(dw, type, dwt, s.m, s.k, 0); (void)TiledF16Gemm(dwt, dx, y_b, s.batch, s.m, s.k, 0); };
    auto gemm_only = [&] { (void)TiledF16Gemm(dwt, dx, y_b, s.batch, s.m, s.k, 0); };
    auto dq_only = [&] { (void)DequantRowsF16Tiled(dw, type, dwt, s.m, s.k, 0); };
    auto dq_old = [&] { (void)DequantRowsF16(dw, type, dwr, s.m, s.k, 0); };

    std::printf("batch %zu  %zu x %zu  %s\n", s.batch, s.m, s.k, s.what);
    const double fl = 2.0 * s.batch * s.m * s.k;
    double t;
    t = Time(old_path);  std::printf("  old UnquantizedF16Gemm (row-major)   %8.3f ms %5.1f TFLOPS\n", t, fl / (t * 1e-3) / 1e12);
    t = Time(dq_old);    std::printf("  old DequantRowsF16                   %8.3f ms\n", t);
    t = Time(dq_only);   std::printf("  DequantRowsF16Tiled                  %8.3f ms\n", t);
    if (s.batch >= 256) {
      (void)DequantRowsF16Tiled(dw, type, dwt, s.m, s.k, 0);
      t = Time(gemm_only); std::printf("  TiledF16Gemm                         %8.3f ms %5.1f TFLOPS\n", t, fl / (t * 1e-3) / 1e12);
      CK(hipMemcpy(got.data(), y_b, got.size() * 4, hipMemcpyDeviceToHost));
      const double r1 = RelL2(got, ref); std::printf("    tiled  rel-L2 vs ref %.2e %s\n", r1, r1 < 1e-3 ? "OK" : "FAIL"); bad += r1 >= 1e-3;
      const double tt = Time(tiled_path);
      std::printf("  dequant-tiled + gemm total           %8.3f ms\n", tt);
    }
    hipFree(dw); hipFree(dx); hipFree(dwr); hipFree(dwt); hipFree(y_ref); hipFree(y_a); hipFree(y_b);
  }
  std::printf(bad ? "FAILED\n" : "all OK\n");
  return bad ? 1 : 0;
}
