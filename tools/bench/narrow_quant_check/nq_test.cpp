// Correctness + bandwidth check for LaunchNarrowQ50Planar / LaunchNarrowQ4KRowAdd.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"

using namespace gufo::core::compute::gemm;

#define CK(x) do { hipError_t err__ = (x); if (err__ != hipSuccess) { std::fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(err__), __LINE__); return 2; } } while (0)

static std::vector<char> ReadAll(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <class F>
static float TimeUs(F f, int iters) {
  hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
  for (int i = 0; i < 30; ++i) f();
  hipDeviceSynchronize();
  hipEventRecord(a);
  for (int i = 0; i < iters; ++i) f();
  hipEventRecord(b); hipEventSynchronize(b);
  float ms; hipEventElapsedTime(&ms, a, b);
  return ms * 1000.0f / iters;
}

int main(int argc, char** argv) {
  const std::string dir = argv[1];
  std::ifstream meta(dir + "/meta.txt");
  std::string name; int kind; size_t cols, rows;
  int bad = 0;
  while (meta >> name >> kind >> cols >> rows) {
    auto w = ReadAll(dir + "/" + name + ".w");
    auto xb = ReadAll(dir + "/" + name + ".x");
    auto bb = ReadAll(dir + "/" + name + ".bias");
    auto rb = ReadAll(dir + "/" + name + ".resid");
    auto refb = ReadAll(dir + "/" + name + ".ref");
    void* d_w; CK(hipMalloc(&d_w, w.size())); CK(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice));
    float *d_x, *d_bias, *d_y;
    CK(hipMalloc(&d_x, cols * 4)); CK(hipMemcpy(d_x, xb.data(), cols * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&d_bias, rows * 4)); CK(hipMemcpy(d_bias, bb.data(), rows * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&d_y, rows * 4));
    void* d_q; CK(hipMalloc(&d_q, PreqQ81Bytes(1, cols) + 4096));
    if (PrequantizeQ8_1(d_x, d_q, 1, cols, 0) != 0) { std::printf("%s: prequantize failed\n", name.c_str()); bad++; continue; }
    auto run = [&] {
      if (kind == 5) LaunchNarrowQ50Planar(d_w, d_q, d_y, d_bias, cols, rows, 0);
      else LaunchNarrowQ4KRowAdd(d_w, d_q, d_bias, d_y, cols, rows, 0);
    };
    if (kind == 4) CK(hipMemcpy(d_y, rb.data(), rows * 4, hipMemcpyHostToDevice));
    run();
    CK(hipDeviceSynchronize());
    std::vector<float> y(rows);
    CK(hipMemcpy(y.data(), d_y, rows * 4, hipMemcpyDeviceToHost));
    const float* ref = reinterpret_cast<const float*>(refb.data());
    double num = 0, den = 0, maxe = 0;
    for (size_t i = 0; i < rows; ++i) { double e = y[i] - ref[i]; num += e * e; den += double(ref[i]) * ref[i]; if (std::fabs(e) > maxe) maxe = std::fabs(e); }
    const double rel = std::sqrt(num / den);
    const bool ok = rel < 0.01;
    // bandwidth: weights only
    if (kind == 4) CK(hipMemcpy(d_y, rb.data(), rows * 4, hipMemcpyHostToDevice));
    const float us = TimeUs(run, 300);
    std::printf("%-5s rows=%zu k=%zu  rel-L2-err %.4f (max abs %.4f) %s   %.1f us  %.1f GB/s (weights %.2f MB)\n",
                name.c_str(), rows, cols, rel, maxe, ok ? "OK" : "FAIL", us, w.size() / (us * 1e-6) / 1e9, w.size() / 1e6);
    if (!ok) bad++;
    hipFree(d_w); hipFree(d_x); hipFree(d_bias); hipFree(d_y); hipFree(d_q);
  }
  // Cold-cache comparison: rotate over 40 weight copies (one per "layer") so every
  // launch reads DRAM like a real decode token does.
  constexpr int kCopies = 40;
  struct S { const char* n; size_t rows, cols; } shapes[] = {{"q  4096x2880", 4096, 2880}, {"k  512x2880", 512, 2880}, {"wo 2880x4096", 2880, 4096}};
  for (auto s : shapes) {
    float *d_x, *d_y, *d_bias; hipMalloc(&d_x, s.cols * 4); hipMemset(d_x, 0, s.cols * 4); hipMalloc(&d_y, s.rows * 4); hipMalloc(&d_bias, s.rows * 4); hipMemset(d_bias, 0, s.rows * 4);
    void* d_q; hipMalloc(&d_q, PreqQ81Bytes(1, s.cols) + 4096);
    PrequantizeQ8_1(d_x, d_q, 1, s.cols, 0);
    const size_t stride = NarrowQ80PrequantStrideBlocks(s.cols);
    auto bench = [&](const char* label, size_t bytes, auto launch) {
      std::vector<void*> ws(kCopies);
      for (auto& p : ws) { hipMalloc(&p, bytes); hipMemset(p, 0x11, bytes); }
      int i = 0;
      const float us = TimeUs([&] { launch(ws[i++ % kCopies]); }, 800);
      std::printf("  %-10s %-14s %7.1f us  %6.1f GB/s  (weights %.2f MB)\n", label, s.n, us, bytes / (us * 1e-6) / 1e9, bytes / 1e6);
      for (auto p : ws) hipFree(p);
    };
    bench("Q8_0", s.rows * (s.cols / 32) * 34, [&](void* w) { LaunchNarrowQ80Decode(w, d_q, d_y, s.cols, s.rows, 1, stride, 0); });
    bench("Q5_0 planar", NarrowQ50PlanarBytes(s.rows, s.cols), [&](void* w) { LaunchNarrowQ50Planar(w, d_q, d_y, d_bias, s.cols, s.rows, 0); });
    if (s.cols % 256 == 0)
      bench("Q4_K", s.rows * (s.cols / 256) * 144, [&](void* w) { LaunchNarrowQ4KRowAdd(w, d_q, d_bias, d_y, s.cols, s.rows, 0); });
  }
  return bad ? 1 : 0;
}
