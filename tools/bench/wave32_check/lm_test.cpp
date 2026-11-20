// LM-head shaped Q-activation GEMV (GemvBatched types 2 = Q4_0 and 8 = Q8_0): bit-exact against the pre-change implementation and timing in GB/s.
// Built by tools/bench/wave32_check/lm_ab.sh.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace mbufo::core::compute::oldw {
void GemvBatched(const void*, int, std::uint64_t, std::uint64_t, const float*, float*, std::uint32_t, std::uint32_t, float, hipStream_t);
}
namespace mbufo::core::compute::gemm {
void GemvBatched(const void*, int, std::uint64_t, std::uint64_t, const float*, float*, std::uint32_t, std::uint32_t, float, hipStream_t);
}
namespace ow = mbufo::core::compute::oldw;
namespace gg = mbufo::core::compute::gemm;

#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

template <class Fn>
static double Time(Fn fn, int iters = 30) {
  for (int i = 0; i < 5; ++i) fn();
  CK(hipDeviceSynchronize());
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  CK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
  float ms; CK(hipEventElapsedTime(&ms, a, b));
  return ms * 1000.0 / iters;
}

int main() {
  srand(11);
  int bad = 0, checks = 0;
  struct Shape { int type; std::uint32_t m; std::uint32_t k; const char* name; } shapes[] = {
      {2, 262144, 2816, "Q4_0 lm head (gemma 4 26B)"}, {2, 100000, 4096, "Q4_0 m=100000 k=4096"}, {2, 65536, 2816, "Q4_0 m=65536"},
      {8, 131072, 4096, "Q8_0 m=131072 k=4096"},
      {2, 2560, 10240, "Q4_0 E4B ffn_down"}, {2, 2560, 2048, "Q4_0 E4B attn_o"}, {2, 2048, 2560, "Q4_0 E4B attn_q"}, {2, 2560, 256, "Q4_0 E4B ple_proj"}, {2, 256, 2560, "Q4_0 E4B ple_gate"}};
  for (auto s : shapes) {
    const std::size_t blk = s.type == 2 ? 18 : 34;
    const std::size_t rb = s.k / 32 * blk;
    std::vector<std::uint8_t> hw(static_cast<std::size_t>(s.m) * rb);
    for (auto& v : hw) v = static_cast<std::uint8_t>(rand());
    for (std::size_t r = 0; r < s.m; ++r)
      for (std::size_t b = 0; b < s.k / 32; ++b) { std::uint8_t* p = hw.data() + r * rb + b * blk; p[0] = 0x00; p[1] = 0x1c; }
    std::vector<float> hx(8 * s.k);
    for (auto& v : hx) v = (rand() / static_cast<float>(RAND_MAX) - 0.5f) * 2.0f;
    void* dw; float *dx, *ya, *yb;
    CK(hipMalloc(&dw, hw.size())); CK(hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&dx, hx.size() * 4)); CK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&ya, 8ull * s.m * 4)); CK(hipMalloc(&yb, 8ull * s.m * 4));
    for (std::uint32_t nt : {1u, 2u, 3u, 8u})
      for (float cap : {0.0f, 30.0f}) {
        CK(hipMemset(ya, 0, 8ull * s.m * 4)); CK(hipMemset(yb, 0, 8ull * s.m * 4));
        ow::GemvBatched(dw, s.type, rb, s.k, dx, ya, nt, s.m, cap, 0);
        gg::GemvBatched(dw, s.type, rb, s.k, dx, yb, nt, s.m, cap, 0);
        CK(hipDeviceSynchronize());
        std::vector<float> a(static_cast<std::size_t>(nt) * s.m), b(a.size());
        CK(hipMemcpy(a.data(), ya, a.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(b.data(), yb, b.size() * 4, hipMemcpyDeviceToHost));
        const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
        ++checks; bad += !same;
        if (!same) std::printf("  MISMATCH %s tokens %u softcap %.0f\n", s.name, nt, cap);
      }
    for (std::uint32_t nt : {1u, 4u}) {
      const double to = Time([&] { ow::GemvBatched(dw, s.type, rb, s.k, dx, ya, nt, s.m, 0.0f, 0); });
      const double tn = Time([&] { gg::GemvBatched(dw, s.type, rb, s.k, dx, yb, nt, s.m, 0.0f, 0); });
      const double mb = static_cast<double>(hw.size()) / 1e6;
      std::printf("  %-28s n=%u: old %8.1f us (%5.0f GB/s)  new %8.1f us (%5.0f GB/s)  x%.2f\n", s.name, nt, to, mb / to * 1e3, tn, mb / tn * 1e3, to / tn);
    }
    CK(hipFree(dw)); CK(hipFree(dx)); CK(hipFree(ya)); CK(hipFree(yb));
  }
  std::printf("%d comparisons, %d mismatches\n%s\n", checks, bad, bad ? "FAILED" : "all OK");
  return bad ? 1 : 0;
}
