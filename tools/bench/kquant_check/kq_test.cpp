// K-quant decode kernels after their format code moved into RowDot<Q4_K/Q5_K> (LaneDotRows, LoadSuper/DecodeSub16):
// QuantGemvSingle and QuantGemvMulti against their pre-change implementations (git, compiled with the library's exact flags,
// namespace mbufo::core::compute::oldk) -- bit-exact outputs on random packed weights over several shapes and token counts,
// plus timing at Glimmer-like shapes. Built by tools/bench/kquant_check/kq_ab.sh.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"

namespace gg = mbufo::core::compute::gemm;
namespace mbufo::core::compute::oldk {
bool QuantGemvSingle(const void*, int, const float*, float*, std::size_t, std::size_t, hipStream_t);
bool QuantGemvMulti(const void*, int, const float*, float*, std::size_t, std::size_t, std::size_t, hipStream_t);
}  // namespace mbufo::core::compute::oldk
namespace ok = mbufo::core::compute::oldk;

#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

static std::size_t BlockBytes(int t) { return t == 12 ? 144 : 176; }
template <class Fn>
static double Time(Fn fn, int iters = 100) {
  for (int i = 0; i < 10; ++i) fn();
  CK(hipDeviceSynchronize());
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  CK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
  float ms; CK(hipEventElapsedTime(&ms, a, b));
  return ms * 1000.0 / iters;
}

int main() {
  int bad = 0, checks = 0;
  for (int type : {12, 13})
    for (std::size_t k : {2048ul, 6656ul})
      for (std::size_t m : {33ul, 1024ul, 4112ul}) {
        const std::size_t rb = k / 256 * BlockBytes(type);
        std::vector<std::uint8_t> hw(m * rb);
        srand(7 + type + (int)k + (int)m);
        for (auto& v : hw) v = (std::uint8_t)rand();
        const std::uint16_t h = 0x1c00;
        for (std::size_t r = 0; r < m; ++r)
          for (std::size_t b = 0; b < k / 256; ++b) { std::uint8_t* p = hw.data() + r * rb + b * BlockBytes(type); p[0] = h & 0xff; p[1] = h >> 8; p[2] = h & 0xff; p[3] = h >> 8; }
        std::vector<float> hx(32 * k);
        for (auto& v : hx) v = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
        void* dw; float *dx, *dya, *dyb;
        CK(hipMalloc(&dw, hw.size())); CK(hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice));
        CK(hipMalloc(&dx, hx.size() * 4)); CK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
        CK(hipMalloc(&dya, 32 * m * 4)); CK(hipMalloc(&dyb, 32 * m * 4));
        // single token
        CK(hipMemset(dya, 0, m * 4)); CK(hipMemset(dyb, 0, m * 4));
        const bool o1 = ok::QuantGemvSingle(dw, type, dx, dya, m, k, 0), n1 = gg::QuantGemvSingle(dw, type, dx, dyb, m, k, 0);
        CK(hipDeviceSynchronize());
        std::vector<float> a(m), b(m);
        CK(hipMemcpy(a.data(), dya, m * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(b.data(), dyb, m * 4, hipMemcpyDeviceToHost));
        const bool s1 = o1 && n1 && !std::memcmp(a.data(), b.data(), m * 4);
        ++checks; bad += !s1;
        if (!s1) std::printf("  Single type %d m %zu k %zu MISMATCH (old %d new %d)\n", type, m, k, o1, n1);
        // multi token (m must be a multiple of 16)
        if (m % 16 == 0)
          for (std::size_t nt : {2ul, 5ul, 16ul, 17ul, 32ul}) {
            CK(hipMemset(dya, 0, nt * m * 4)); CK(hipMemset(dyb, 0, nt * m * 4));
            const bool o2 = ok::QuantGemvMulti(dw, type, dx, dya, nt, m, k, 0), n2 = gg::QuantGemvMulti(dw, type, dx, dyb, nt, m, k, 0);
            CK(hipDeviceSynchronize());
            std::vector<float> c(nt * m), d(nt * m);
            CK(hipMemcpy(c.data(), dya, nt * m * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(d.data(), dyb, nt * m * 4, hipMemcpyDeviceToHost));
            const bool s2 = o2 && n2 && !std::memcmp(c.data(), d.data(), nt * m * 4);
            ++checks; bad += !s2;
            if (!s2) std::printf("  Multi type %d tokens %zu m %zu k %zu MISMATCH (old %d new %d)\n", type, nt, m, k, o2, n2);
          }
        hipFree(dw); hipFree(dx); hipFree(dya); hipFree(dyb);
      }
  std::printf("%d comparisons, %d mismatches\n", checks, bad);
  // timing at Glimmer-like shapes (rotating nothing: weights far larger than the cache)
  for (int type : {12, 13}) {
    const std::size_t m = 8192, k = 6656, rb = k / 256 * BlockBytes(type);
    void* dw; float *dx, *dy;
    CK(hipMalloc(&dw, m * rb)); CK(hipMemset(dw, 0x11, m * rb));
    CK(hipMalloc(&dx, 32 * k * 4)); CK(hipMemset(dx, 0, 32 * k * 4)); CK(hipMalloc(&dy, 32 * m * 4));
    const double s_o = Time([&] { (void)ok::QuantGemvSingle(dw, type, dx, dy, m, k, 0); });
    const double s_n = Time([&] { (void)gg::QuantGemvSingle(dw, type, dx, dy, m, k, 0); });
    std::printf("timing type %d single token m %zu k %zu: old %.1f us, new %.1f us (x%.3f)\n", type, m, k, s_o, s_n, s_o / s_n);
    for (std::size_t nt : {4ul, 16ul, 32ul}) {
      const double m_o = Time([&] { (void)ok::QuantGemvMulti(dw, type, dx, dy, nt, m, k, 0); });
      const double m_n = Time([&] { (void)gg::QuantGemvMulti(dw, type, dx, dy, nt, m, k, 0); });
      std::printf("timing type %d multi %zu tokens: old %.1f us, new %.1f us (x%.3f)\n", type, nt, m_o, m_n, m_o / m_n);
    }
    hipFree(dw); hipFree(dx); hipFree(dy);
  }
  std::printf(bad ? "FAILED\n" : "all OK\n");
  return bad ? 1 : 0;
}
