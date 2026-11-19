// blocked_splitk: grouped kernel vs legacy kernel vs float64 reference, plus timing.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/core/compute/attention/blocked_splitk.hpp"

using namespace gufo::core::compute::blocked_splitk;

#define CK(x) do { hipError_t err__ = (x); if (err__ != hipSuccess) { std::fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(err__), __LINE__); return 2; } } while (0)

static std::vector<char> ReadAll(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <class F>
static float TimeUs(F f, int iters) {
  hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
  for (int i = 0; i < 10; ++i) f();
  hipDeviceSynchronize();
  hipEventRecord(a);
  for (int i = 0; i < iters; ++i) f();
  hipEventRecord(b); hipEventSynchronize(b);
  float ms; hipEventElapsedTime(&ms, a, b);
  return ms * 1000.0f / iters;
}

static bool Run(unsigned D, const float* q, const std::uint8_t* k, const std::uint8_t* v, float* out, unsigned nt,
                const std::uint32_t* sp, unsigned kv_len, unsigned heads, unsigned kvh, unsigned kbb, unsigned vbb,
                float* partials, size_t pbytes, float scale) {
  if (D == 128)
    return AttentionBlockedSplitK<KvCodecQ8, 128>(q, k, v, out, nt, sp, kv_len, heads, kvh, D, 0, scale, kbb, vbb, partials, pbytes);
  return AttentionBlockedSplitK<KvCodecQ8, 256>(q, k, v, out, nt, sp, kv_len, heads, kvh, D, 0, scale, kbb, vbb, partials, pbytes);
}

int main(int argc, char** argv) {
  const std::string dir = argv[1];
  const bool timing = argc > 2;
  std::ifstream meta(dir + "/meta.txt");
  std::string name; unsigned D, heads, kvh, kv_len, nt, kbb, vbb;
  int bad = 0, total = 0;
  while (meta >> name >> D >> heads >> kvh >> kv_len >> nt >> kbb >> vbb) {
    auto qb = ReadAll(dir + "/" + name + ".q"), kb = ReadAll(dir + "/" + name + ".k"),
         vb = ReadAll(dir + "/" + name + ".v"), rb = ReadAll(dir + "/" + name + ".ref");
    float *d_q, *d_o, *d_p; std::uint8_t *d_k, *d_v; std::uint32_t* d_sp;
    CK(hipMalloc(&d_q, qb.size())); CK(hipMemcpy(d_q, qb.data(), qb.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&d_k, kb.size())); CK(hipMemcpy(d_k, kb.data(), kb.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&d_v, vb.size())); CK(hipMemcpy(d_v, vb.data(), vb.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&d_o, nt * heads * D * 4));
    const size_t pbytes = AttentionBlockedSplitKScratchBytes();
    CK(hipMalloc(&d_p, pbytes));
    const std::uint32_t start = kv_len - nt;
    CK(hipMalloc(&d_sp, 4)); CK(hipMemcpy(d_sp, &start, 4, hipMemcpyHostToDevice));
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const float* ref = reinterpret_cast<const float*>(rb.data());
    std::vector<float> o_new(nt * heads * D), o_old(nt * heads * D);
    unsetenv("GUFO_BLOCKED_SPLITK_LEGACY");
    const bool ok_new = Run(D, d_q, d_k, d_v, d_o, nt, d_sp, kv_len, heads, kvh, kbb, vbb, d_p, pbytes, scale);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(o_new.data(), d_o, o_new.size() * 4, hipMemcpyDeviceToHost));
    setenv("GUFO_BLOCKED_SPLITK_LEGACY", "1", 1);
    const bool ok_old = Run(D, d_q, d_k, d_v, d_o, nt, d_sp, kv_len, heads, kvh, kbb, vbb, d_p, pbytes, scale);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(o_old.data(), d_o, o_old.size() * 4, hipMemcpyDeviceToHost));
    auto rel = [&](const std::vector<float>& a, const float* b) {
      double num = 0, den = 0;
      for (size_t i = 0; i < a.size(); ++i) { double e = a[i] - b[i]; num += e * e; den += double(b[i]) * b[i]; }
      return std::sqrt(num / den);
    };
    const double e_new = rel(o_new, ref), e_old = rel(o_old, ref);
    // the legacy kernel refuses single-split shapes (returns false); only check it when it ran
    const bool ok = ok_new && e_new < 2e-5 && (!ok_old || e_old < 2e-5);
    char tbuf[96] = "";
    if (timing && ok) {
      unsetenv("GUFO_BLOCKED_SPLITK_LEGACY");
      const float us_new = TimeUs([&] { Run(D, d_q, d_k, d_v, d_o, nt, d_sp, kv_len, heads, kvh, kbb, vbb, d_p, pbytes, scale); }, 200);
      setenv("GUFO_BLOCKED_SPLITK_LEGACY", "1", 1);
      const float us_old = !ok_old ? 0.0f : TimeUs([&] { Run(D, d_q, d_k, d_v, d_o, nt, d_sp, kv_len, heads, kvh, kbb, vbb, d_p, pbytes, scale); }, 200);
      std::snprintf(tbuf, sizeof tbuf, "  %7.1f us (legacy %7.1f us, %.1fx)", us_new, us_old, us_old > 0 ? us_old / us_new : 0.0f);
    }
    std::printf("%-4s D=%3u heads=%2u kvh=%2u kv=%5u nt=%u kb=%2u vb=%2u  new %.2e  legacy %.2e  %s%s\n", name.c_str(), D, heads, kvh, kv_len, nt, kbb, vbb, e_new, e_old, ok ? "OK" : "FAIL", tbuf);
    ++total; if (!ok) ++bad;
    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_o); hipFree(d_p); hipFree(d_sp);
  }
  std::printf("%d/%d cases OK\n", total - bad, total);
  return bad ? 1 : 0;
}
