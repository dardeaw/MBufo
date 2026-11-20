// gpt-oss decode attention (d = 64, 64 heads / 8 kv heads, per-head sink, window 0 or 128) over the blocked Q8 cache:
// the direct-read row-chunk kernel against (a) a double-precision host reference, (b) the previous route (dequantize the whole
// cache into F16 staging, then flash-sink split-K); batch property (row t of a verify batch == the row alone); timing.
// Built by tools/bench/attn_check/sink_ab.sh.
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

#include "src/core/compute/attention/blocked_kv.hpp"
#include "src/core/compute/attention/blocked_splitk.hpp"
#include "src/core/compute/attention/flash_sink.hpp"

namespace bk = mbufo::core::compute::blocked_kv;
namespace sk = mbufo::core::compute::blocked_splitk;
namespace fs = mbufo::core::compute::flash_sink;

#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

constexpr std::uint32_t kHeads = 64, kKvHeads = 8, kD = 64, kBlk = 34;
constexpr std::size_t kRowBlocks = kKvHeads * kD / 32;  // 16 blocks per key row

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

static void FillKv(std::vector<std::uint8_t>& kv, std::size_t keys, std::mt19937& rng) {
  kv.assign(keys * kRowBlocks * kBlk, 0);
  for (std::size_t j = 0; j < keys * kRowBlocks; ++j) {
    std::uint8_t* b = kv.data() + j * kBlk;
    for (int i = 0; i < 32; ++i) b[i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(rng() % 255) - 127));
    const __half sc = __float2half(0.004f + 0.01f * static_cast<float>(rng() % 100) / 100.0f);
    std::memcpy(b + 32, &sc, 2);
  }
}
static float KvAt(const std::vector<std::uint8_t>& kv, std::size_t key, std::uint32_t kvh, std::uint32_t i) {
  const std::uint8_t* b = kv.data() + (key * kRowBlocks + kvh * (kD / 32) + i / 32) * kBlk;
  __half sc; std::memcpy(&sc, b + 32, 2);
  return static_cast<float>(static_cast<std::int8_t>(b[i % 32])) * __half2float(sc);
}

int main() {
  std::mt19937 rng(7);
  const std::size_t max_keys = 30000 + 16;
  std::vector<std::uint8_t> hk, hv;
  FillKv(hk, max_keys, rng); FillKv(hv, max_keys, rng);
  std::uint8_t *dk, *dv; CK(hipMalloc(&dk, hk.size())); CK(hipMalloc(&dv, hv.size()));
  CK(hipMemcpy(dk, hk.data(), hk.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dv, hv.data(), hv.size(), hipMemcpyHostToDevice));
  __half *fk, *fv; CK(hipMalloc(&fk, max_keys * kKvHeads * kD * 2)); CK(hipMalloc(&fv, max_keys * kKvHeads * kD * 2));
  std::vector<float> hq(8 * kHeads * kD), hs(kHeads);
  for (auto& x : hq) x = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1500.0f;
  for (auto& x : hs) x = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 400.0f;
  float *dq, *dsink, *dout, *dpart;
  CK(hipMalloc(&dq, hq.size() * 4)); CK(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
  CK(hipMalloc(&dsink, hs.size() * 4)); CK(hipMemcpy(dsink, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
  CK(hipMalloc(&dout, 8 * kHeads * kD * 4));
  const std::size_t part_bytes = fs::FlashSinkDecodeScratchBytes(8, kHeads);
  CK(hipMalloc(&dpart, part_bytes));
  int bad = 0;
  const float scale = 0.125f;
  auto stage = [&](std::uint32_t kv) {
    CK(bk::DequantizeKvRows(dk, fk, dv, fv, static_cast<std::size_t>(kv) * kKvHeads, kD, kBlk, kBlk, bk::KvCodec::kQ8, 0) ? hipSuccess : hipErrorUnknown);
  };

  for (std::uint32_t window : {0u, 128u}) {
    // ---- accuracy, one row at n_kv = 6000 ----
    const std::uint32_t n_kv = 6000;
    std::vector<double> ref(kHeads * kD);
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::uint32_t kvh = h / (kHeads / kKvHeads);
      const std::uint32_t lo = (window != 0 && n_kv > window) ? n_kv - window : 0;
      std::vector<double> sc(n_kv - lo);
      double mx = hs[h];
      for (std::uint32_t j = lo; j < n_kv; ++j) {
        double a = 0;
        for (std::uint32_t i = 0; i < kD; ++i) a += static_cast<double>(hq[h * kD + i]) * KvAt(hk, j, kvh, i);
        sc[j - lo] = a * scale; mx = std::max(mx, sc[j - lo]);
      }
      double z = std::exp(hs[h] - mx); std::vector<double> o(kD, 0.0);
      for (std::uint32_t j = lo; j < n_kv; ++j) {
        const double w = std::exp(sc[j - lo] - mx); z += w;
        for (std::uint32_t i = 0; i < kD; ++i) o[i] += w * KvAt(hv, j, kvh, i);
      }
      for (std::uint32_t i = 0; i < kD; ++i) ref[h * kD + i] = o[i] / z;
    }
    auto err = [&](const char* name) {
      std::vector<float> o(kHeads * kD); CK(hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost));
      double worst = 0, sc = 0;
      for (std::size_t i = 0; i < o.size(); ++i) { worst = std::max(worst, std::fabs(o[i] - ref[i])); sc = std::max(sc, std::fabs(ref[i])); }
      std::printf("  window %3u accuracy %-28s max|err| %.3g (max|ref| %.3g)\n", window, name, worst, sc);
      return worst;
    };
    CK(hipMemset(dout, 0, 8 * kHeads * kD * 4));
    stage(n_kv);
    (void)fs::FlashSinkDecode(dq, fk, fv, dsink, dout, 1, n_kv - 1, n_kv, kHeads, kKvHeads, kD, window, scale, 0, dpart, part_bytes);
    CK(hipDeviceSynchronize());
    const double e_old = err("dequantize + flash-sink");
    CK(hipMemset(dout, 0, 8 * kHeads * kD * 4));
    if (!sk::AttentionBlockedSinkRowChunk<sk::KvCodecQ8, 64>(dq, dk, dv, dsink, dout, 1, n_kv - 1, kHeads, kKvHeads, kD, window, 0, scale, dpart, part_bytes, 128)) { std::printf("  launch refused\n"); ++bad; continue; }
    CK(hipDeviceSynchronize());
    const double e_new = err("direct row-chunk");
    if (e_new > 4 * e_old + 1e-6) { std::printf("  FAIL accuracy\n"); ++bad; }

    // ---- batch property ----
    for (std::uint32_t start : {0u, 100u, 5000u, 29900u}) {
      const std::uint32_t n = 6;
      CK(hipMemset(dout, 0, 8 * kHeads * kD * 4));
      if (!sk::AttentionBlockedSinkRowChunk<sk::KvCodecQ8, 64>(dq, dk, dv, dsink, dout, n, start, kHeads, kKvHeads, kD, window, 0, scale, dpart, part_bytes, 128)) { ++bad; continue; }
      CK(hipDeviceSynchronize());
      std::vector<float> batch(n * kHeads * kD); CK(hipMemcpy(batch.data(), dout, batch.size() * 4, hipMemcpyDeviceToHost));
      int diff = 0;
      for (std::uint32_t t = 0; t < n; ++t) {
        float* q1; CK(hipMalloc(&q1, kHeads * kD * 4)); CK(hipMemcpy(q1, hq.data() + static_cast<std::size_t>(t) * kHeads * kD, kHeads * kD * 4, hipMemcpyHostToDevice));
        CK(hipMemset(dout, 0, 8 * kHeads * kD * 4));
        (void)sk::AttentionBlockedSinkRowChunk<sk::KvCodecQ8, 64>(q1, dk, dv, dsink, dout, 1, start + t, kHeads, kKvHeads, kD, window, 0, scale, dpart, part_bytes, 128);
        CK(hipDeviceSynchronize());
        std::vector<float> one(kHeads * kD); CK(hipMemcpy(one.data(), dout, one.size() * 4, hipMemcpyDeviceToHost));
        if (std::memcmp(one.data(), batch.data() + static_cast<std::size_t>(t) * kHeads * kD, one.size() * 4) != 0) ++diff;
        CK(hipFree(q1));
      }
      std::printf("  window %3u batch n=%u start %5u: %d of %u rows differ from the single-row launch %s\n", window, n, start, diff, n, diff ? "FAIL" : "");
      bad += diff != 0;
    }

    // ---- timing: one layer, one token ----
    std::printf("  window %3u timing (us):  kv_len   old(dequant+flash)  new(direct)\n", window);
    for (std::uint32_t kv : {1024u, 6000u, 16000u, 29900u}) {
      stage(kv);
      const double t_old = Time([&] { stage(kv); (void)fs::FlashSinkDecode(dq, fk, fv, dsink, dout, 1, kv - 1, kv, kHeads, kKvHeads, kD, window, scale, 0, dpart, part_bytes); });
      const double t_new = Time([&] { (void)sk::AttentionBlockedSinkRowChunk<sk::KvCodecQ8, 64>(dq, dk, dv, dsink, dout, 1, kv - 1, kHeads, kKvHeads, kD, window, 0, scale, dpart, part_bytes, 128); });
      std::printf("                          %6u   %10.1f      %10.1f   x%.1f\n", kv, t_old, t_new, t_old / t_new);
    }
  }
  std::printf("%s\n", bad ? "FAILED" : "all OK");
  return bad ? 1 : 0;
}
