// Blocked-Q8 decode attention at qwen35moe's geometry (16 q heads, 2 kv heads, d = 256): timing of the scalar single-block read,
// the length-sized Split-K and the row-chunk Split-K (several chunk bases); accuracy against a double-precision host reference;
// and the row-chunk property: row t of an n-row launch equals that row decoded alone at the same position, bit for bit.
// Built by tools/bench/attn_check/attn_ab.sh.
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

namespace bk = mbufo::core::compute::blocked_kv;
namespace sk = mbufo::core::compute::blocked_splitk;

#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

#ifndef KHEADS
#define KHEADS 16
#endif
#ifndef KKVHEADS
#define KKVHEADS 2
#endif
constexpr std::uint32_t kHeads = KHEADS, kKvHeads = KKVHEADS, kD = 256, kBlk = 34;
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
  std::mt19937 rng(99);
  const std::size_t max_keys = 32768 + 16;
  std::vector<std::uint8_t> hk, hv;
  FillKv(hk, max_keys, rng); FillKv(hv, max_keys, rng);
  std::uint8_t *dk, *dv; CK(hipMalloc(&dk, hk.size())); CK(hipMalloc(&dv, hv.size()));
  CK(hipMemcpy(dk, hk.data(), hk.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dv, hv.data(), hv.size(), hipMemcpyHostToDevice));
  std::vector<float> hq(8 * kHeads * kD);
  for (auto& x : hq) x = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 700.0f;
  float *dq, *dout, *dpart; std::uint32_t* dpos;
  CK(hipMalloc(&dq, hq.size() * 4)); CK(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
  CK(hipMalloc(&dout, 8 * kHeads * kD * 4)); CK(hipMalloc(&dpart, 40u << 20)); CK(hipMalloc(&dpos, 4));
  const std::size_t part_bytes = 40u << 20;
  int bad = 0;
  auto set_pos = [&](std::uint32_t p) { CK(hipMemcpy(dpos, &p, 4, hipMemcpyHostToDevice)); };

  // ---- accuracy vs a double-precision reference (one row at pos = 6000 - 1) ----
  {
    const std::uint32_t n_kv = 6000;
    set_pos(n_kv - 1);
    std::vector<double> ref(kHeads * kD);
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::uint32_t kvh = h / (kHeads / kKvHeads);
      std::vector<double> sc(n_kv);
      double mx = -1e300;
      for (std::uint32_t j = 0; j < n_kv; ++j) {
        double a = 0;
        for (std::uint32_t i = 0; i < kD; ++i) a += static_cast<double>(hq[h * kD + i]) * KvAt(hk, j, kvh, i);
        sc[j] = a * 0.0625; mx = std::max(mx, sc[j]);
      }
      double z = 0; std::vector<double> o(kD, 0.0);
      for (std::uint32_t j = 0; j < n_kv; ++j) {
        const double w = std::exp(sc[j] - mx); z += w;
        for (std::uint32_t i = 0; i < kD; ++i) o[i] += w * KvAt(hv, j, kvh, i);
      }
      for (std::uint32_t i = 0; i < kD; ++i) ref[h * kD + i] = o[i] / z;
    }
    auto err = [&](const char* name) {
      std::vector<float> o(kHeads * kD); CK(hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost));
      double worst = 0, scale = 0;
      for (std::size_t i = 0; i < o.size(); ++i) { worst = std::max(worst, std::fabs(o[i] - ref[i])); scale = std::max(scale, std::fabs(ref[i])); }
      const bool ok = worst <= 2e-5 * std::max(scale, 1.0);
      std::printf("  accuracy %-22s max|err| %.3g (max|ref| %.3g) %s\n", name, worst, scale, ok ? "ok" : "FAIL");
      if (!ok) ++bad;
    };
    CK(hipMemset(dout, 0, 8 * kHeads * kD * 4));
    bk::AttentionKv(dq, dk, dv, nullptr, 0, dout, nullptr, 1, 1, dpos, kHeads, kKvHeads, kD, 1, kBlk, kBlk, bk::KvCodec::kQ8, 0);
    CK(hipDeviceSynchronize()); err("scalar single block");
    for (std::uint32_t base : {32u, 128u, 256u}) {
      CK(hipMemset(dout, 0, 8 * kHeads * kD * 4));
      const bool ok = sk::AttentionBlockedSplitKRowChunk<sk::KvCodecQ8, 256>(dq, dk, dv, dout, 1, dpos, n_kv, kHeads, kKvHeads, kD, 0, 0.0625f, kBlk, kBlk, dpart, part_bytes, base);
      CK(hipDeviceSynchronize());
      char name[64]; std::snprintf(name, sizeof name, "row-chunk base %u", base);
      if (!ok) { std::printf("  %s refused\n", name); ++bad; } else err(name);
    }
  }

  // ---- row-chunk property: row t of an n-row launch == the same row alone ----
  for (std::uint32_t base : {32u, 128u})
    for (std::uint32_t start : {0u, 100u, 300u, 4000u, 32700u}) {
      const std::uint32_t n = 8;
      set_pos(start);
      CK(hipMemset(dout, 0, 8 * kHeads * kD * 4));
      if (!sk::AttentionBlockedSplitKRowChunk<sk::KvCodecQ8, 256>(dq, dk, dv, dout, n, dpos, start + n, kHeads, kKvHeads, kD, 0, 0.0625f, kBlk, kBlk, dpart, part_bytes, base)) { ++bad; std::printf("  batch launch refused\n"); continue; }
      CK(hipDeviceSynchronize());
      std::vector<float> batch(n * kHeads * kD); CK(hipMemcpy(batch.data(), dout, batch.size() * 4, hipMemcpyDeviceToHost));
      std::size_t diff = 0;
      for (std::uint32_t t = 0; t < n; ++t) {
        set_pos(start + t);
        CK(hipMemset(dout, 0, 8 * kHeads * kD * 4));
        // the single-row call sees the row at the start of the query buffer: copy row t of q to slot 0
        CK(hipMemcpy(dq, hq.data() + static_cast<std::size_t>(t) * kHeads * kD, kHeads * kD * 4, hipMemcpyHostToDevice));
        (void)sk::AttentionBlockedSplitKRowChunk<sk::KvCodecQ8, 256>(dq, dk, dv, dout, 1, dpos, start + t + 1, kHeads, kKvHeads, kD, 0, 0.0625f, kBlk, kBlk, dpart, part_bytes, base);
        CK(hipDeviceSynchronize());
        std::vector<float> one(kHeads * kD); CK(hipMemcpy(one.data(), dout, one.size() * 4, hipMemcpyDeviceToHost));
        if (std::memcmp(one.data(), batch.data() + static_cast<std::size_t>(t) * kHeads * kD, one.size() * 4) != 0) ++diff;
      }
      CK(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
      std::printf("  row-chunk base %3u start %5u: %u of %u rows differ from the single-row launch %s\n", base, start, static_cast<unsigned>(diff), n, diff ? "FAIL" : "");
      if (diff) ++bad;
    }

  // ---- timing ----
  std::printf("  timing, one layer, one token (us):\n");
  std::printf("    %7s %9s %9s", "kv_len", "scalar", "splitk64");
  const std::uint32_t bases[] = {64, 128, 256};
  for (auto b : bases) std::printf("  rc%-5u", b);
  std::printf("\n");
  for (std::uint32_t kv : {300u, 1024u, 6000u, 16000u, 30000u}) {
    set_pos(kv - 1);
    const double t_s = Time([&] { bk::AttentionKv(dq, dk, dv, nullptr, 0, dout, nullptr, 1, 1, dpos, kHeads, kKvHeads, kD, 1, kBlk, kBlk, bk::KvCodec::kQ8, 0); });
    const double t_k = Time([&] { (void)sk::AttentionBlockedSplitK<sk::KvCodecQ8, 256>(dq, dk, dv, dout, 1, dpos, kv, kHeads, kKvHeads, kD, 0, 0.0625f, kBlk, kBlk, dpart, part_bytes); });
    std::printf("    %7u %9.1f %9.1f", kv, t_s, t_k);
    for (auto b : bases) {
      const double t = Time([&] { (void)sk::AttentionBlockedSplitKRowChunk<sk::KvCodecQ8, 256>(dq, dk, dv, dout, 1, dpos, kv, kHeads, kKvHeads, kD, 0, 0.0625f, kBlk, kBlk, dpart, part_bytes, b); });
      std::printf("  %7.1f", t);
    }
    std::printf("   (KV bytes %.1f MB)\n", 2.0 * kv * kRowBlocks * kBlk / 1e6 / 1.0);
  }
  std::printf("  chunk sweep (us), base = forced chunk when above the rule:\n");
  for (std::uint32_t kv : {3000u, 6000u, 12000u, 16000u, 24000u, 30000u}) {
    set_pos(kv - 1);
    std::printf("    kv %6u:", kv);
    for (std::uint32_t b : {128u, 256u, 384u, 512u, 640u, 768u, 1024u, 1536u}) {
      if (sk::RowChunkKeys(kv, b) != b) { std::printf("  %5u:   -  ", b); continue; }
      const double t = Time([&] { (void)sk::AttentionBlockedSplitKRowChunk<sk::KvCodecQ8, 256>(dq, dk, dv, dout, 1, dpos, kv, kHeads, kKvHeads, kD, 0, 0.0625f, kBlk, kBlk, dpart, part_bytes, b); });
      std::printf("  %5u:%6.1f", b, t);
    }
    std::printf("\n");
  }
  std::printf("  8-row verify launch (us):");
  for (std::uint32_t kv : {6000u, 30000u}) {
    set_pos(kv - 8);
    const double t = Time([&] { (void)sk::AttentionBlockedSplitKRowChunk<sk::KvCodecQ8, 256>(dq, dk, dv, dout, 8, dpos, kv, kHeads, kKvHeads, kD, 0, 0.0625f, kBlk, kBlk, dpart, part_bytes, 128); });
    std::printf("  kv %u: rc128 %.1f", kv, t);
  }
  std::printf("\n%s\n", bad ? "FAILED" : "all OK");
  return bad ? 1 : 0;
}
