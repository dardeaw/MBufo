// Gemma 4 decode attention (AttentionCross, F16 KV): the current kernels against the pre-change implementation
// (git, compiled with the library's exact flags, namespace mbufo::models::gemma4::oldkernels): bit-exact outputs on random
// data and timing at global-layer shapes (d = 512, 16 heads, 2 kv heads) and sliding-layer shapes (d = 256, 16 heads, 8 kv heads).
// Built by tools/bench/gemma_attn_check/gemma_ab.sh.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "src/models/gemma4/kernels/kernels.hpp"

namespace mbufo::models::gemma4::oldkernels {
void AttentionCross(const float* q, const __half* k_cache, const __half* v_cache, float* out, std::uint32_t n_tokens,
                    std::uint32_t query_pos, std::uint32_t kv_len, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d,
                    std::uint32_t kv_stride, std::uint32_t window, hipStream_t stream);
}
namespace nk = mbufo::models::gemma4::kernels;
namespace ok = mbufo::models::gemma4::oldkernels;

#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

template <class Fn>
static double Time(Fn fn, int iters = 20) {
  for (int i = 0; i < 3; ++i) fn();
  CK(hipDeviceSynchronize());
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  CK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
  float ms; CK(hipEventElapsedTime(&ms, a, b));
  return ms * 1000.0 / iters;
}

int main() {
  std::mt19937 rng(5);
  int bad = 0, checks = 0;
  struct Shape { std::uint32_t d, heads, kv_heads; const char* name; } shapes[] = {{512, 16, 2, "global d512"}, {256, 16, 8, "sliding d256"}};
  const std::uint32_t max_kv = 28800;
  for (auto s : shapes) {
    const std::uint32_t stride = s.kv_heads * s.d;
    std::vector<__half> hk(static_cast<std::size_t>(max_kv) * stride), hv(hk.size());
    for (auto& x : hk) x = __float2half(static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 600.0f);
    for (auto& x : hv) x = __float2half(static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 600.0f);
    std::vector<float> hq(8 * s.heads * s.d);
    for (auto& x : hq) x = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 400.0f;
    __half *dk, *dv; float *dq, *do1, *do2;
    CK(hipMalloc(&dk, hk.size() * 2)); CK(hipMalloc(&dv, hv.size() * 2));
    CK(hipMemcpy(dk, hk.data(), hk.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dv, hv.data(), hv.size() * 2, hipMemcpyHostToDevice));
    CK(hipMalloc(&dq, hq.size() * 4)); CK(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&do1, 8 * s.heads * s.d * 4)); CK(hipMalloc(&do2, 8 * s.heads * s.d * 4));
    for (std::uint32_t n_tok : {1u, 4u})
      for (std::uint32_t kv : {300u, 4096u, 16384u, 28700u}) {
        const std::uint32_t pos = kv - n_tok;  // last row attends all kv keys
        CK(hipMemset(do1, 0, 8 * s.heads * s.d * 4)); CK(hipMemset(do2, 0, 8 * s.heads * s.d * 4));
        nk::AttentionCross(dq, dk, dv, do1, n_tok, pos, kv, s.heads, s.kv_heads, s.d, stride, 0, 0);
        ok::AttentionCross(dq, dk, dv, do2, n_tok, pos, kv, s.heads, s.kv_heads, s.d, stride, 0, 0);
        CK(hipDeviceSynchronize());
        std::vector<float> a(n_tok * s.heads * s.d), b(a.size());
        CK(hipMemcpy(a.data(), do1, a.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(b.data(), do2, b.size() * 4, hipMemcpyDeviceToHost));
        double worst = 0, scale = 0;
        for (std::size_t i = 0; i < a.size(); ++i) { worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i])); scale = std::max(scale, std::fabs(static_cast<double>(b[i]))); }
        const bool bits = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
        const bool same = worst <= 1e-3 * std::max(scale, 1e-2);  // a different summation order is allowed to move the last bits
        ++checks; bad += !same;
        const double tn = Time([&] { nk::AttentionCross(dq, dk, dv, do1, n_tok, pos, kv, s.heads, s.kv_heads, s.d, stride, 0, 0); });
        const double to = Time([&] { ok::AttentionCross(dq, dk, dv, do2, n_tok, pos, kv, s.heads, s.kv_heads, s.d, stride, 0, 0); });
        const double mb = 2.0 * kv * s.kv_heads * s.d * 2 / 1e6;
        std::printf("  %-12s n=%u kv=%5u: new %8.1f us  old %8.1f us  x%.2f  (%.0f MB, new %.0f GB/s)  %s\n", s.name, n_tok, kv, tn, to, to / tn, mb, mb / tn * 1e3,
                    bits ? "bit-identical" : (same ? "within reorder noise" : "DIFFERENT"));
      }
    if (s.d == 512) for (float qs : {1.0f, 0.125f}) {  // accuracy of both kernels against a double-precision host reference (one row, 4096 keys); 0.125 = model-like logit spread
      const std::uint32_t kv = 4096;
      std::vector<float> hq2(hq.size()); for (std::size_t i = 0; i < hq.size(); ++i) hq2[i] = hq[i] * qs;
      float* dq2; CK(hipMalloc(&dq2, hq2.size() * 4)); CK(hipMemcpy(dq2, hq2.data(), hq2.size() * 4, hipMemcpyHostToDevice));
      CK(hipMemset(do1, 0, 8 * s.heads * s.d * 4)); CK(hipMemset(do2, 0, 8 * s.heads * s.d * 4));
      nk::AttentionCross(dq2, dk, dv, do1, 1, kv - 1, kv, s.heads, s.kv_heads, s.d, stride, 0, 0);
      ok::AttentionCross(dq2, dk, dv, do2, 1, kv - 1, kv, s.heads, s.kv_heads, s.d, stride, 0, 0);
      CK(hipDeviceSynchronize());
      std::vector<float> a(s.heads * s.d), b(a.size());
      CK(hipMemcpy(a.data(), do1, a.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(b.data(), do2, b.size() * 4, hipMemcpyDeviceToHost));
      double en = 0, eo = 0, sc = 0;
      for (std::uint32_t h = 0; h < s.heads; ++h) {
        const std::uint32_t kvh = h / (s.heads / s.kv_heads);
        std::vector<double> w(kv); double mx = -1e300;
        for (std::uint32_t j = 0; j < kv; ++j) {
          double d = 0;
          for (std::uint32_t i = 0; i < s.d; ++i) d += static_cast<double>(hq2[h * s.d + i]) * __half2float(hk[static_cast<std::size_t>(j) * stride + kvh * s.d + i]);
          w[j] = d; mx = std::max(mx, d);
        }
        double z = 0; for (auto& x : w) { x = std::exp(x - mx); z += x; }
        for (std::uint32_t i = 0; i < s.d; ++i) {
          double o = 0;
          for (std::uint32_t j = 0; j < kv; ++j) o += w[j] * __half2float(hv[static_cast<std::size_t>(j) * stride + kvh * s.d + i]);
          o /= z;
          en = std::max(en, std::fabs(o - a[h * s.d + i])); eo = std::max(eo, std::fabs(o - b[h * s.d + i])); sc = std::max(sc, std::fabs(o));
        }
      }
      ++checks; const bool acc_ok = qs < 0.5f ? (en <= 4.0 * eo + 1e-6) : (en <= 2e-4);
      bad += !acc_ok;
      std::printf("  global d512 accuracy vs double reference (kv 4096, q x%.3g): new max|err| %.3g, old %.3g (max|ref| %.3g) %s\n", qs, en, eo, sc, acc_ok ? "ok" : "FAIL");
      CK(hipFree(dq2));
    }
    if (s.d == 512) {  // a row of a verify batch equals the same row decoded alone
      for (std::uint32_t kv : {300u, 4096u, 28700u}) {
        const std::uint32_t n_tok = 6, pos0 = kv - n_tok;
        CK(hipMemset(do1, 0, 8 * s.heads * s.d * 4));
        nk::AttentionCross(dq, dk, dv, do1, n_tok, pos0, kv, s.heads, s.kv_heads, s.d, stride, 0, 0);
        CK(hipDeviceSynchronize());
        std::vector<float> batch(n_tok * s.heads * s.d); CK(hipMemcpy(batch.data(), do1, batch.size() * 4, hipMemcpyDeviceToHost));
        int diff = 0;
        for (std::uint32_t t = 0; t < n_tok; ++t) {
          float* qrow = dq + static_cast<std::size_t>(t) * s.heads * s.d;
          // single-row call sees its query at the start of the buffer
          std::vector<float> save(s.heads * s.d); CK(hipMemcpy(save.data(), dq, save.size() * 4, hipMemcpyDeviceToHost));
          CK(hipMemcpy(do2, qrow, s.heads * s.d * 4, hipMemcpyDeviceToDevice));
          float* tmpq; CK(hipMalloc(&tmpq, s.heads * s.d * 4)); CK(hipMemcpy(tmpq, do2, s.heads * s.d * 4, hipMemcpyDeviceToDevice));
          CK(hipMemset(do2, 0, 8 * s.heads * s.d * 4));
          nk::AttentionCross(tmpq, dk, dv, do2, 1, pos0 + t, pos0 + t + 1, s.heads, s.kv_heads, s.d, stride, 0, 0);
          CK(hipDeviceSynchronize());
          std::vector<float> one(s.heads * s.d); CK(hipMemcpy(one.data(), do2, one.size() * 4, hipMemcpyDeviceToHost));
          if (std::memcmp(one.data(), batch.data() + static_cast<std::size_t>(t) * s.heads * s.d, one.size() * 4) != 0) ++diff;
          CK(hipFree(tmpq));
        }
        ++checks; bad += diff != 0;
        std::printf("  global d512 verify batch n=%u kv=%5u: %d of %u rows differ from the single-row decode %s\n", n_tok, kv, diff, n_tok, diff ? "FAIL" : "");
      }
    }
    CK(hipFree(dk)); CK(hipFree(dv)); CK(hipFree(dq)); CK(hipFree(do1)); CK(hipFree(do2));
  }
  std::printf("%d comparisons, %d differ\n%s\n", checks, bad, bad ? "FAILED" : "all OK");
  return bad ? 1 : 0;
}
