// MXFP4 decode MoE kernels (core moe_fused.hip, gpt-oss) after the block math moved into RowDot<MXFP4>: bit-exact against the
// previous implementation (compiled from git into namespace oldmoe), at gpt-oss-20b shapes, masked slots included.
//   built by tools/bench/moe_decode_check/mxfp4_ab.sh (needs the pre-change moe_fused.hip from git)
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"

namespace gg = mbufo::core::compute::gemm;
namespace mbufo::core::compute::oldmoe {
void MoeVecGateUpMxfp4(const void*, std::uint64_t, std::uint64_t, const float*, const std::int32_t*, float*, float*, std::uint32_t,
                       std::uint32_t, std::uint32_t, std::uint32_t, hipStream_t);
void GemvQSlotMxfp4(const void*, std::uint64_t, std::uint64_t, const float*, const std::int32_t*, float*, std::uint32_t, std::uint32_t,
                    std::uint32_t, hipStream_t);
bool MoeDecodeGateUpActMxfp4(const void*, std::uint64_t, std::uint64_t, const float*, const std::int32_t*, const float*, const float*,
                             float*, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, hipStream_t);
bool MoeDecodeDownEpiMxfp4(const void*, std::uint64_t, std::uint64_t, const float*, const std::int32_t*, const float*, const float*,
                           float*, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, hipStream_t);
}  // namespace mbufo::core::compute::oldmoe
namespace om = mbufo::core::compute::oldmoe;

#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

template <class Fn>
static double Time(Fn fn, int iters = 300) {
  for (int i = 0; i < 30; ++i) fn();
  CK(hipDeviceSynchronize());
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  CK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
  float ms; CK(hipEventElapsedTime(&ms, a, b));
  return ms * 1000.0 / iters;
}

static bool Same(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * 4) == 0; }

int main() {
  const std::uint32_t hidden = 2880, ff = 2880, n_exp = 32, used = 4;
  const std::uint64_t grb = hidden / 32 * 17, drb = ff / 32 * 17;
  const std::uint64_t gstride = 2ull * ff * grb, dstride = (std::uint64_t)hidden * drb;
  std::vector<std::uint8_t> hg(n_exp * gstride), hd(n_exp * dstride);
  srand(31);
  for (auto& v : hg) v = (std::uint8_t)rand();
  for (auto& v : hd) v = (std::uint8_t)rand();
  for (std::size_t o = 0; o < hg.size(); o += 17) hg[o] = 118 + rand() % 8;  // e8m0 scale bytes, sane range
  for (std::size_t o = 0; o < hd.size(); o += 17) hd[o] = 118 + rand() % 8;
  int bad = 0;
  void *dg, *dd; float *dx, *dgate_b, *dup_b, *ddown_b, *dwts;
  CK(hipMalloc(&dg, hg.size())); CK(hipMalloc(&dd, hd.size()));
  CK(hipMemcpy(dg, hg.data(), hg.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dd, hd.data(), hd.size(), hipMemcpyHostToDevice));
  std::vector<float> hb(n_exp * (std::size_t)std::max(ff, hidden));
  for (auto& v : hb) v = (rand() / (float)RAND_MAX - 0.5f) * 0.2f;
  CK(hipMalloc(&dgate_b, hb.size() * 4)); CK(hipMalloc(&dup_b, hb.size() * 4)); CK(hipMalloc(&ddown_b, hb.size() * 4));
  CK(hipMemcpy(dgate_b, hb.data(), hb.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dup_b, hb.data(), hb.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(ddown_b, hb.data(), hb.size() * 4, hipMemcpyHostToDevice));
  for (std::uint32_t nt : {1u, 2u, 5u}) {
    const std::uint32_t slots = nt * used;
    std::vector<float> hx((std::size_t)nt * hidden), hw(slots), hact((std::size_t)slots * ff);
    for (auto& v : hx) v = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
    for (auto& v : hw) v = rand() / (float)RAND_MAX;
    for (auto& v : hact) v = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
    std::vector<std::int32_t> ids(slots);
    for (auto& v : ids) v = rand() % n_exp;
    ids[0] = -1;  // masked slot
    std::int32_t* dids; CK(hipMalloc(&dids, slots * 4)); CK(hipMemcpy(dids, ids.data(), slots * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dx, hx.size() * 4)); CK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dwts, hw.size() * 4)); CK(hipMemcpy(dwts, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
    float* dact; CK(hipMalloc(&dact, hact.size() * 4)); CK(hipMemcpy(dact, hact.data(), hact.size() * 4, hipMemcpyHostToDevice));
    const std::size_t nff = (std::size_t)slots * ff, nh = (std::size_t)slots * hidden, nout = (std::size_t)nt * hidden;
    float *ga, *ua, *gb, *ub, *qa, *qb, *aa, *ab, *ea, *eb;
    for (float** p : {&ga, &ua, &gb, &ub, &aa, &ab}) CK(hipMalloc(p, nff * 4));
    for (float** p : {&qa, &qb}) CK(hipMalloc(p, nh * 4));
    for (float** p : {&ea, &eb}) CK(hipMalloc(p, nout * 4));
    auto zero = [&](float* p, std::size_t n) { CK(hipMemset(p, 0, n * 4)); };
    // 1) vec gate/up
    for (float* p : {ga, ua, gb, ub}) zero(p, nff);
    om::MoeVecGateUpMxfp4(dg, grb, gstride, dx, dids, ga, ua, slots, used, ff, hidden, 0);
    gg::MoeVecGateUpMxfp4(dg, grb, gstride, dx, dids, gb, ub, slots, used, ff, hidden, 0);
    // 2) vec down (act from the random activations)
    zero(qa, nh); zero(qb, nh);
    om::GemvQSlotMxfp4(dd, drb, dstride, dact, dids, qa, slots, hidden, ff, 0);
    gg::GemvQSlotMxfp4(dd, drb, dstride, dact, dids, qb, slots, hidden, ff, 0);
    // 3) fused gate/up + SwiGLU-OAI
    zero(aa, nff); zero(ab, nff);
    const bool o3 = om::MoeDecodeGateUpActMxfp4(dg, grb, gstride, dx, dids, dgate_b, dup_b, aa, slots, used, ff, hidden, 0);
    const bool n3 = gg::MoeDecodeGateUpActMxfp4(dg, grb, gstride, dx, dids, dgate_b, dup_b, ab, slots, used, ff, hidden, 0);
    // 4) fused down + weighted epilogue
    zero(ea, nout); zero(eb, nout);
    const bool o4 = om::MoeDecodeDownEpiMxfp4(dd, drb, dstride, dact, dids, dwts, ddown_b, ea, nt, used, hidden, ff, 0);
    const bool n4 = gg::MoeDecodeDownEpiMxfp4(dd, drb, dstride, dact, dids, dwts, ddown_b, eb, nt, used, hidden, ff, 0);
    CK(hipDeviceSynchronize());
    auto get = [&](float* p, std::size_t n) { std::vector<float> v(n); CK(hipMemcpy(v.data(), p, n * 4, hipMemcpyDeviceToHost)); return v; };
    const bool s1 = Same(get(ga, nff).data(), get(gb, nff).data(), nff) && Same(get(ua, nff).data(), get(ub, nff).data(), nff);
    const bool s2 = Same(get(qa, nh).data(), get(qb, nh).data(), nh);
    const bool s3 = o3 && n3 && Same(get(aa, nff).data(), get(ab, nff).data(), nff);
    const bool s4 = o4 && n4 && Same(get(ea, nout).data(), get(eb, nout).data(), nout);
    std::printf("tokens %u: vec gate/up %s, vec down %s, fused gate/up+act %s, fused down+epilogue %s\n", nt, s1 ? "bit-exact" : "MISMATCH",
                s2 ? "bit-exact" : "MISMATCH", s3 ? "bit-exact" : "MISMATCH", s4 ? "bit-exact" : "MISMATCH");
    bad += !(s1 && s2 && s3 && s4);
    if (nt == 1) {  // timing, old vs new at the same compile flags; rotating nothing: same ids (weights ~500 MB, mostly uncached)
      const double t1o = Time([&] { om::MoeVecGateUpMxfp4(dg, grb, gstride, dx, dids, ga, ua, slots, used, ff, hidden, 0); });
      const double t1n = Time([&] { gg::MoeVecGateUpMxfp4(dg, grb, gstride, dx, dids, gb, ub, slots, used, ff, hidden, 0); });
      const double t3o = Time([&] { (void)om::MoeDecodeGateUpActMxfp4(dg, grb, gstride, dx, dids, dgate_b, dup_b, aa, slots, used, ff, hidden, 0); });
      const double t3n = Time([&] { (void)gg::MoeDecodeGateUpActMxfp4(dg, grb, gstride, dx, dids, dgate_b, dup_b, ab, slots, used, ff, hidden, 0); });
      const double t4o = Time([&] { (void)om::MoeDecodeDownEpiMxfp4(dd, drb, dstride, dact, dids, dwts, ddown_b, ea, nt, used, hidden, ff, 0); });
      const double t4n = Time([&] { (void)gg::MoeDecodeDownEpiMxfp4(dd, drb, dstride, dact, dids, dwts, ddown_b, eb, nt, used, hidden, ff, 0); });
      std::printf("timing 1 token: vec gate/up %.1f -> %.1f us, fused gate/up+act %.1f -> %.1f us, fused down+epi %.1f -> %.1f us (old -> new)\n", t1o, t1n, t3o, t3n, t4o, t4n);
    }
    for (float* p : {ga, ua, gb, ub, aa, ab, qa, qb, ea, eb, dact, dx, dwts}) hipFree(p);
    hipFree(dids);
  }
  std::printf(bad ? "FAILED\n" : "all OK\n");
  return bad ? 1 : 0;
}
