// Quantized-activation (Q8_1) lane kernels after their weight decode / int dot / float tail moved into RowDot<Q4_0> / RowDot<Q8_0>
// (QLane, LoadQLane, QIntDot, QTail) and the Q4_0 and Q8_0 kernels became one template: GemvBatched (Q4_0 and Q8_0, 1..8 tokens),
// GemvGateUpSwiglu, GemvQkv and gemma4's grouped MoE path (MoeVecGateUpSwiglu + GemvQSlot with duplicate experts) against the
// pre-change implementation (git, compiled with the library's exact flags, namespace mbufo::core::compute::oldq). Built by
// qdot_check/qdot_ab.sh.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"

namespace gg = mbufo::core::compute::gemm;
namespace mbufo::core::compute::oldq {
void GemvBatched(const void*, int, std::uint64_t, std::uint64_t, const float*, float*, std::uint32_t, std::uint32_t, float, hipStream_t);
void GemvGateUpSwiglu(const void*, const void*, int, std::uint64_t, std::uint64_t, const float*, float*, std::uint32_t, std::uint32_t, hipStream_t);
void GemvQkv(const void*, int, std::uint64_t, std::uint32_t, const void*, int, std::uint64_t, const void*, int, std::uint64_t, std::uint32_t,
             std::uint64_t, const float*, float*, float*, float*, std::uint32_t, hipStream_t);
void MoeVecGateUpSwiglu(const void*, std::uint64_t, std::uint64_t, const float*, const std::int32_t*, float*, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t, hipStream_t);
void GemvQSlot(const void*, std::uint64_t, std::uint64_t, const float*, const std::int32_t*, float*, std::uint32_t, std::uint32_t, std::uint32_t,
               hipStream_t);
}  // namespace mbufo::core::compute::oldq
namespace oq = mbufo::core::compute::oldq;

#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

static float Rf() { return (rand() / (float)RAND_MAX - 0.5f) * 2.f; }
static std::size_t BB(int type) { return type == 2 ? 18 : 34; }
static void Fill(std::vector<std::uint8_t>& w, std::size_t rows, std::size_t rb, int type) {
  for (auto& v : w) v = (std::uint8_t)rand();
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t b = 0; b < rb / BB(type); ++b) { std::uint8_t* p = w.data() + r * rb + b * BB(type); p[0] = 0x00; p[1] = 0x1c; }
}
template <class T> static T* Up(const std::vector<T>& h) { T* d; CK(hipMalloc(&d, h.size() * sizeof(T))); CK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return d; }
static bool Same(float* a, float* b, std::size_t n) { std::vector<float> x(n), y(n); CK(hipDeviceSynchronize()); CK(hipMemcpy(x.data(), a, n * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(y.data(), b, n * 4, hipMemcpyDeviceToHost)); return !std::memcmp(x.data(), y.data(), n * 4); }
template <class Fn> static double Time(Fn fn, int iters = 200) { for (int i = 0; i < 20; ++i) fn(); CK(hipDeviceSynchronize()); hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b)); CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) fn(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms * 1000.0 / iters; }

int main() {
  int bad = 0, checks = 0;
  auto report = [&](const char* what, bool ok) { ++checks; bad += !ok; if (!ok) std::printf("  %s MISMATCH\n", what); };
  srand(5);
  const std::uint32_t cols = 2048, m = 200;
  float* ya; float* yb; CK(hipMalloc(&ya, 8 * 512 * 4)); CK(hipMalloc(&yb, 8 * 512 * 4));
  std::vector<float> hx(8 * cols); for (auto& v : hx) v = Rf(); float* dx = Up(hx);
  for (int type : {2, 8}) {
    const std::uint64_t rb = cols / 32 * BB(type);
    std::vector<std::uint8_t> hw(m * rb); Fill(hw, m, rb, type); std::uint8_t* dw = Up(hw);
    for (std::uint32_t nt : {1u, 2u, 3u, 8u}) {
      CK(hipMemset(ya, 0, nt * m * 4)); CK(hipMemset(yb, 0, nt * m * 4));
      oq::GemvBatched(dw, type, rb, cols, dx, ya, nt, m, 0.0f, 0); gg::GemvBatched(dw, type, rb, cols, dx, yb, nt, m, 0.0f, 0);
      char nm[96]; std::snprintf(nm, sizeof nm, "GemvBatched type %d tokens %u", type, nt); report(nm, Same(ya, yb, nt * m));
    }
    if (type == 2) {
      std::vector<std::uint8_t> hu(m * rb); Fill(hu, m, rb, 2); std::uint8_t* du = Up(hu);
      for (std::uint32_t nt : {2u, 5u, 8u}) {
        CK(hipMemset(ya, 0, nt * m * 4)); CK(hipMemset(yb, 0, nt * m * 4));
        oq::GemvGateUpSwiglu(dw, du, 2, rb, cols, dx, ya, nt, m, 0); gg::GemvGateUpSwiglu(dw, du, 2, rb, cols, dx, yb, nt, m, 0);
        char nm[96]; std::snprintf(nm, sizeof nm, "GemvGateUpSwiglu tokens %u", nt); report(nm, Same(ya, yb, nt * m));
      }
      const std::uint32_t qdim = 256, kvdim = 64;
      std::vector<std::uint8_t> hq(qdim * rb), hk(kvdim * rb), hv(kvdim * rb); Fill(hq, qdim, rb, 2); Fill(hk, kvdim, rb, 2); Fill(hv, kvdim, rb, 2);
      std::uint8_t *dq = Up(hq), *dk = Up(hk), *dv = Up(hv);
      float *qa, *ka, *va, *qb, *kb, *vb;
      for (float** p : {&qa, &qb}) CK(hipMalloc(p, 8 * qdim * 4)); for (float** p : {&ka, &kb, &va, &vb}) CK(hipMalloc(p, 8 * kvdim * 4));
      for (std::uint32_t nt : {2u, 4u, 8u}) {
        oq::GemvQkv(dq, 2, rb, qdim, dk, 2, rb, dv, 2, rb, kvdim, cols, dx, qa, ka, va, nt, 0);
        gg::GemvQkv(dq, 2, rb, qdim, dk, 2, rb, dv, 2, rb, kvdim, cols, dx, qb, kb, vb, nt, 0);
        char nm[96]; std::snprintf(nm, sizeof nm, "GemvQkv tokens %u", nt);
        report(nm, Same(qa, qb, nt * qdim) && Same(ka, kb, nt * kvdim) && Same(va, vb, nt * kvdim));
      }
      const double t_o = Time([&] { oq::GemvBatched(dw, 2, rb, cols, dx, ya, 4, m, 0.0f, 0); });
      const double t_n = Time([&] { gg::GemvBatched(dw, 2, rb, cols, dx, yb, 4, m, 0.0f, 0); });
      std::printf("timing GemvBatched Q4_0 x Q8_1, 4 tokens: %.1f -> %.1f us (old -> new)\n", t_o, t_n);
    } else {
      const double t_o = Time([&] { oq::GemvBatched(dw, 8, rb, cols, dx, ya, 4, m, 0.0f, 0); });
      const double t_n = Time([&] { gg::GemvBatched(dw, 8, rb, cols, dx, yb, 4, m, 0.0f, 0); });
      std::printf("timing GemvBatched Q8_0 x Q8_1, 4 tokens: %.1f -> %.1f us (old -> new)\n", t_o, t_n);
    }
  }
  // grouped MoE (gemma4 verification batch): duplicate experts across tokens exercise the unique-expert bucketing
  {
    const std::uint32_t hidden = 2816, ff = 704, E = 16, used = 8;
    const std::uint64_t grb = hidden / 32 * 18, drb = ff / 32 * 18, gstride = 2ull * ff * grb, dstride = (std::uint64_t)hidden * drb;
    std::vector<std::uint8_t> hg(E * gstride), hd(E * dstride); Fill(hg, E * 2ull * ff, grb, 2); Fill(hd, E * (std::size_t)hidden, drb, 2);
    std::uint8_t *dg = Up(hg), *dd = Up(hd);
    for (std::uint32_t nt : {2u, 3u, 5u}) {
      const std::uint32_t slots = nt * used;
      std::vector<float> hxx(nt * (std::size_t)hidden); for (auto& v : hxx) v = Rf(); float* dxx = Up(hxx);
      std::vector<std::int32_t> ids(slots); for (auto& v : ids) v = rand() % E; ids[1] = ids[used + 1];
      std::int32_t* dids = Up(ids);
      float *aa, *ab, *oa, *ob;
      for (float** p : {&aa, &ab}) CK(hipMalloc(p, slots * ff * 4)); for (float** p : {&oa, &ob}) CK(hipMalloc(p, slots * hidden * 4));
      CK(hipMemset(aa, 0, slots * ff * 4)); CK(hipMemset(ab, 0, slots * ff * 4)); CK(hipMemset(oa, 0, slots * hidden * 4)); CK(hipMemset(ob, 0, slots * hidden * 4));
      oq::MoeVecGateUpSwiglu(dg, grb, gstride, dxx, dids, aa, slots, used, ff, hidden, 0); oq::GemvQSlot(dd, drb, dstride, aa, dids, oa, slots, hidden, ff, 0);
      gg::MoeVecGateUpSwiglu(dg, grb, gstride, dxx, dids, ab, slots, used, ff, hidden, 0); gg::GemvQSlot(dd, drb, dstride, ab, dids, ob, slots, hidden, ff, 0);
      char nm[96]; std::snprintf(nm, sizeof nm, "grouped MoE gate/up tokens %u", nt); report(nm, Same(aa, ab, slots * ff));
      std::snprintf(nm, sizeof nm, "grouped MoE down tokens %u", nt); report(nm, Same(oa, ob, slots * hidden));
    }
  }
  std::printf("%d comparisons, %d mismatches\n", checks, bad);
  std::printf(bad ? "FAILED\n" : "all OK\n");
  return bad ? 1 : 0;
}
