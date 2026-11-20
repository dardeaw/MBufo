// Fallback / legacy F32-activation kernels after their Q4_0 and F32 math moved into the row-dot layer: GemvBatched (multi-token),
// GemvGateUpSwiglu, GemvQkv, gemma4's legacy MoeVecGateUp and GemvQSlot, against their pre-change implementations (git, compiled with
// the library's exact flags, namespace mbufo::core::compute::oldw). Environment switches force the paths under test
// (GEMMA4_NO_Q8_BATCH: F32-activation multi-token kernels; GEMMA4_NO_WAVE32_GEMV: legacy block kernels). Built by wave32_check/w32_ab.sh.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"

namespace gg = mbufo::core::compute::gemm;
namespace mbufo::core::compute::oldw {
void GemvBatched(const void*, int, std::uint64_t, std::uint64_t, const float*, float*, std::uint32_t, std::uint32_t, float, hipStream_t);
void GemvGateUpSwiglu(const void*, const void*, int, std::uint64_t, std::uint64_t, const float*, float*, std::uint32_t, std::uint32_t, hipStream_t);
void GemvQkv(const void*, int, std::uint64_t, std::uint32_t, const void*, int, std::uint64_t, const void*, int, std::uint64_t, std::uint32_t,
             std::uint64_t, const float*, float*, float*, float*, std::uint32_t, hipStream_t);
void MoeVecGateUp(const void*, std::uint64_t, std::uint64_t, const float*, const std::int32_t*, float*, float*, std::uint32_t, std::uint32_t,
                  std::uint32_t, std::uint32_t, hipStream_t);
void GemvQSlot(const void*, std::uint64_t, std::uint64_t, const float*, const std::int32_t*, float*, std::uint32_t, std::uint32_t, std::uint32_t,
               hipStream_t);
}  // namespace mbufo::core::compute::oldw
namespace ow = mbufo::core::compute::oldw;

#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

static float Rf() { return (rand() / (float)RAND_MAX - 0.5f) * 2.f; }
static void FillQ40(std::vector<std::uint8_t>& w, std::size_t rows, std::size_t rb) {
  for (auto& v : w) v = (std::uint8_t)rand();
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t b = 0; b < rb / 18; ++b) { std::uint8_t* p = w.data() + r * rb + b * 18; p[0] = 0x00; p[1] = 0x1c; }  // d ~ 2e-4
}
template <class T> static T* Up(const std::vector<T>& h) { T* d; CK(hipMalloc(&d, h.size() * sizeof(T))); CK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return d; }
static bool Same(float* a, float* b, std::size_t n) { std::vector<float> x(n), y(n); CK(hipDeviceSynchronize()); CK(hipMemcpy(x.data(), a, n * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(y.data(), b, n * 4, hipMemcpyDeviceToHost)); return !std::memcmp(x.data(), y.data(), n * 4); }
template <class Fn> static double Time(Fn fn, int iters = 200) { for (int i = 0; i < 20; ++i) fn(); CK(hipDeviceSynchronize()); hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b)); CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) fn(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms * 1000.0 / iters; }

int main() {
  setenv("GEMMA4_NO_Q8_BATCH", "1", 1);
  int bad = 0, checks = 0;
  auto report = [&](const char* what, bool ok) { ++checks; bad += !ok; if (!ok) std::printf("  %s MISMATCH\n", what); };
  srand(3);
  const std::uint32_t cols = 2048, m = 200;
  for (int type : {2, 0}) {
    const std::uint64_t rb = type == 2 ? cols / 32 * 18 : cols * 4;
    std::vector<std::uint8_t> hw(m * rb);
    if (type == 2) FillQ40(hw, m, rb); else { auto* f = reinterpret_cast<float*>(hw.data()); for (std::size_t i = 0; i < m * cols; ++i) f[i] = Rf() * 0.1f; }
    std::vector<float> hx(8 * cols); for (auto& v : hx) v = Rf();
    std::uint8_t* dw = Up(hw); float* dx = Up(hx); float *ya, *yb; CK(hipMalloc(&ya, 8 * m * 4)); CK(hipMalloc(&yb, 8 * m * 4));
    for (std::uint32_t nt : {2u, 3u, 8u}) {
      CK(hipMemset(ya, 0, nt * m * 4)); CK(hipMemset(yb, 0, nt * m * 4));
      ow::GemvBatched(dw, type, rb, cols, dx, ya, nt, m, 0.0f, 0); gg::GemvBatched(dw, type, rb, cols, dx, yb, nt, m, 0.0f, 0);
      char nm[96]; std::snprintf(nm, sizeof nm, "GemvBatched type %d tokens %u", type, nt); report(nm, Same(ya, yb, nt * m));
    }
    if (type == 2) {
      std::vector<std::uint8_t> hu(m * rb); FillQ40(hu, m, rb);
      std::uint8_t* du = Up(hu);
      for (std::uint32_t nt : {2u, 5u, 8u}) {
        CK(hipMemset(ya, 0, nt * m * 4)); CK(hipMemset(yb, 0, nt * m * 4));
        ow::GemvGateUpSwiglu(dw, du, 2, rb, cols, dx, ya, nt, m, 0); gg::GemvGateUpSwiglu(dw, du, 2, rb, cols, dx, yb, nt, m, 0);
        char nm[96]; std::snprintf(nm, sizeof nm, "GemvGateUpSwiglu tokens %u", nt); report(nm, Same(ya, yb, nt * m));
      }
      // QKV: three Q4_0 matrices (qdim 256, kvdim 64)
      const std::uint32_t qdim = 256, kvdim = 64;
      std::vector<std::uint8_t> hq(qdim * rb), hk(kvdim * rb), hv(kvdim * rb); FillQ40(hq, qdim, rb); FillQ40(hk, kvdim, rb); FillQ40(hv, kvdim, rb);
      std::uint8_t *dq = Up(hq), *dk = Up(hk), *dv = Up(hv);
      float *qa, *ka, *va, *qb, *kb, *vb;
      for (float** p : {&qa, &qb}) CK(hipMalloc(p, 8 * qdim * 4)); for (float** p : {&ka, &kb, &va, &vb}) CK(hipMalloc(p, 8 * kvdim * 4));
      for (std::uint32_t nt : {2u, 4u, 8u}) {
        ow::GemvQkv(dq, 2, rb, qdim, dk, 2, rb, dv, 2, rb, kvdim, cols, dx, qa, ka, va, nt, 0);
        gg::GemvQkv(dq, 2, rb, qdim, dk, 2, rb, dv, 2, rb, kvdim, cols, dx, qb, kb, vb, nt, 0);
        char nm[96]; std::snprintf(nm, sizeof nm, "GemvQkv tokens %u", nt);
        report(nm, Same(qa, qb, nt * qdim) && Same(ka, kb, nt * kvdim) && Same(va, vb, nt * kvdim));
      }
    }
  }
  // legacy gemma4 block kernels
  setenv("GEMMA4_NO_WAVE32_GEMV", "1", 1);
  {
    const std::uint32_t hidden = 2816, ff = 704, E = 16, used = 8;
    const std::uint64_t grb = hidden / 32 * 18, drb = ff / 32 * 18, gstride = 2ull * ff * grb, dstride = (std::uint64_t)hidden * drb;
    std::vector<std::uint8_t> hg(E * gstride), hd(E * dstride); FillQ40(hg, E * 2ull * ff, grb); FillQ40(hd, E * (std::size_t)hidden, drb);
    std::vector<float> hx(hidden), hact(used * ff); for (auto& v : hx) v = Rf(); for (auto& v : hact) v = Rf();
    std::vector<std::int32_t> ids(used); for (auto& v : ids) v = rand() % E; ids[3] = -1;
    std::uint8_t *dg = Up(hg), *dd = Up(hd); float *dx = Up(hx), *dact = Up(hact); std::int32_t* dids = Up(ids);
    float *ga, *ua, *gb, *ub, *oa, *ob;
    for (float** p : {&ga, &ua, &gb, &ub}) CK(hipMalloc(p, used * ff * 4)); for (float** p : {&oa, &ob}) CK(hipMalloc(p, used * hidden * 4));
    for (float* p : {ga, ua, gb, ub}) CK(hipMemset(p, 0, used * ff * 4));
    ow::MoeVecGateUp(dg, grb, gstride, dx, dids, ga, ua, used, used, ff, hidden, 0); gg::MoeVecGateUp(dg, grb, gstride, dx, dids, gb, ub, used, used, ff, hidden, 0);
    report("legacy MoeVecGateUp", Same(ga, gb, used * ff) && Same(ua, ub, used * ff));
    CK(hipMemset(oa, 0, used * hidden * 4)); CK(hipMemset(ob, 0, used * hidden * 4));
    ow::GemvQSlot(dd, drb, dstride, dact, dids, oa, used, hidden, ff, 0); gg::GemvQSlot(dd, drb, dstride, dact, dids, ob, used, hidden, ff, 0);
    report("legacy GemvQSlot", Same(oa, ob, used * hidden));
    const double t1o = Time([&] { ow::MoeVecGateUp(dg, grb, gstride, dx, dids, ga, ua, used, used, ff, hidden, 0); });
    const double t1n = Time([&] { gg::MoeVecGateUp(dg, grb, gstride, dx, dids, gb, ub, used, used, ff, hidden, 0); });
    const double t2o = Time([&] { ow::GemvQSlot(dd, drb, dstride, dact, dids, oa, used, hidden, ff, 0); });
    const double t2n = Time([&] { gg::GemvQSlot(dd, drb, dstride, dact, dids, ob, used, hidden, ff, 0); });
    std::printf("timing legacy MoeVecGateUp %.1f -> %.1f us, GemvQSlot %.1f -> %.1f us (old -> new)\n", t1o, t1n, t2o, t2n);
  }
  std::printf("%d comparisons, %d mismatches\n", checks, bad);
  std::printf(bad ? "FAILED\n" : "all OK\n");
  return bad ? 1 : 0;
}
