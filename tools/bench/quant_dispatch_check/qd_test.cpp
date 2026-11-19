// GemvDecode / QuantGemmPrefill over every listed format vs a float64 reference, plus the refusal paths
// (unlisted type code, missing decode kernel, short scratch, bad shape). Uses the dequant_rows_check data:
//   python3 tools/bench/dequant_rows_check/dq_prep.py <dir>
//   hipcc -O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I<repo> -x hip qd_test.cpp \\
//         -x none <build>/libgufo_core_gemm.a -o qd_test && ./qd_test <dir>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"

using namespace gufo::core::compute::gemm;
namespace q = gufo::core::quant;

#define CK(x) do { hipError_t err__ = (x); if (err__ != hipSuccess) { std::fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(err__), __LINE__); return 2; } } while (0)

static std::vector<char> ReadAll(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static double RelErr(const std::vector<float>& y, const std::vector<double>& ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < y.size(); ++i) { const double e = y[i] - ref[i]; num += e * e; den += ref[i] * ref[i]; }
  return std::sqrt(num / den);
}

static const char* S(q::Status s) {
  switch (s) {
    case q::Status::kOk: return "kOk";
    case q::Status::kUnsupportedType: return "kUnsupportedType";
    case q::Status::kBadShape: return "kBadShape";
    case q::Status::kScratchTooSmall: return "kScratchTooSmall";
  }
  return "?";
}

int main(int argc, char** argv) {
  const std::string dir = argv[1];
  std::ifstream meta(dir + "/meta.txt");
  std::string name; int type; size_t cols, rows;
  int bad = 0, total = 0;
  auto check = [&](bool ok, const char* what, const std::string& label, const std::string& detail) {
    ++total; if (!ok) ++bad;
    std::printf("  %-34s %-10s %s  %s\n", what, label.c_str(), ok ? "OK" : "FAIL", detail.c_str());
  };
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.f, 1.f);
  while (meta >> name >> type >> cols >> rows) {
    auto raw = ReadAll(dir + "/" + name + ".bin");
    auto refb = ReadAll(dir + "/" + name + ".ref.f32");
    const float* wref = reinterpret_cast<const float*>(refb.data());
    void* d_w; CK(hipMalloc(&d_w, raw.size())); CK(hipMemcpy(d_w, raw.data(), raw.size(), hipMemcpyHostToDevice));
    const size_t k = cols, m = rows;
    __half* d_scr; const size_t scr_bytes = m * k * sizeof(__half); CK(hipMalloc(&d_scr, scr_bytes));
    void* d_preq; const size_t preq_bytes = NarrowQ80PrequantScratchBytes(k); CK(hipMalloc(&d_preq, preq_bytes));
    std::printf("%s (type %d, %zu x %zu)\n", name.c_str(), type, m, k);
    for (size_t n : {size_t(1), size_t(3), size_t(16), size_t(21), size_t(32), size_t(300)}) {
      std::vector<float> x(n * k); for (auto& v : x) v = nd(rng);
      std::vector<double> ref(n * m, 0.0);
      for (size_t t = 0; t < n; ++t) for (size_t r = 0; r < m; ++r) {
        double a = 0; for (size_t c = 0; c < k; ++c) a += double(wref[r * k + c]) * x[t * k + c];
        ref[t * m + r] = a;
      }
      float *d_x, *d_y; CK(hipMalloc(&d_x, x.size() * 4)); CK(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice));
      CK(hipMalloc(&d_y, n * m * 4));
      std::vector<float> y(n * m);
      if (n <= kQuantGemvGroupedMaxTokens) {
        const q::Status st = GemvDecode(d_w, type, d_x, d_y, n, m, k, d_preq, preq_bytes, 0);
        CK(hipDeviceSynchronize());
        const bool multi_ok = (type == 12 || type == 13) && m % 16 == 0 && k % 256 == 0;  // multi-token kernel shapes
        if (n > 8 && !multi_ok) {  // only the K-quant multi-token kernel serves 9..32 tokens (and only for m % 16 == 0)
          check(st == q::Status::kUnsupportedType, "GemvDecode n>8 (dense route expected)", name, std::string("n=") + std::to_string(n) + " " + S(st));
        } else if (type == 14) {  // Q6_K has no decode kernel: must be an explicit refusal
          check(st == q::Status::kUnsupportedType, "GemvDecode (no kernel -> refuse)", name, std::string("n=") + std::to_string(n) + " " + S(st));
        } else {
          CK(hipMemcpy(y.data(), d_y, y.size() * 4, hipMemcpyDeviceToHost));
          const double e = RelErr(y, ref);
          char b[96]; std::snprintf(b, sizeof b, "n=%zu rel-L2 %.2e %s", n, e, S(st));
          check(st == q::Status::kOk && e < 3e-2, "GemvDecode", name, b);
        }
      } else {
        std::vector<__half> x16(x.size()); for (size_t i = 0; i < x.size(); ++i) x16[i] = __float2half(x[i]);
        __half* d_x16; CK(hipMalloc(&d_x16, x16.size() * 2)); CK(hipMemcpy(d_x16, x16.data(), x16.size() * 2, hipMemcpyHostToDevice));
        const q::Status st = QuantGemmPrefill(d_w, type, d_x16, d_y, n, m, k, d_scr, scr_bytes, 0);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(y.data(), d_y, y.size() * 4, hipMemcpyDeviceToHost));
        const double e = RelErr(y, ref);
        char b[96]; std::snprintf(b, sizeof b, "n=%zu rel-L2 %.2e %s", n, e, S(st));
        check(st == q::Status::kOk && e < 4e-3, "QuantGemmPrefill", name, b);
        if (type != 8 && type != 1) {  // dequant path needs scratch
          const q::Status s2 = QuantGemmPrefill(d_w, type, d_x16, d_y, n, m, k, d_scr, scr_bytes / 2, 0);
          check(s2 == q::Status::kScratchTooSmall, "QuantGemmPrefill short scratch", name, S(s2));
        }
        hipFree(d_x16);
      }
      hipFree(d_x); hipFree(d_y);
    }
    {  // EmbedGatherStreams must equal EmbedGather replicated per stream, bit for bit
      const std::int32_t ids_h[3] = {0, static_cast<std::int32_t>(m - 1), 1};
      std::int32_t* d_ids; CK(hipMalloc(&d_ids, 12)); CK(hipMemcpy(d_ids, ids_h, 12, hipMemcpyHostToDevice));
      float *d_a, *d_b; CK(hipMalloc(&d_a, 3 * k * 4)); CK(hipMalloc(&d_b, 3 * 3 * k * 4));
      const bool oa = EmbedGather(d_w, type, d_ids, d_a, 3, k, 0);
      const bool ob = EmbedGatherStreams(d_w, type, d_ids, d_b, 3, k, 3, 0);
      CK(hipDeviceSynchronize());
      std::vector<float> a(3 * k), b(9 * k);
      CK(hipMemcpy(a.data(), d_a, a.size() * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(b.data(), d_b, b.size() * 4, hipMemcpyDeviceToHost));
      bool same = oa && ob;
      for (size_t tt = 0; tt < 3 && same; ++tt) for (size_t s2 = 0; s2 < 3 && same; ++s2) for (size_t i = 0; i < k; ++i)
        if (b[(tt * 3 + s2) * k + i] != a[tt * k + i]) { same = false; break; }
      check(same, "EmbedGatherStreams == replicated", name, "");
      hipFree(d_ids); hipFree(d_a); hipFree(d_b);
    }
    hipFree(d_w); hipFree(d_scr); hipFree(d_preq);
  }
  // multi-token K-quant kernel through GemvDecode on a synthetic eligible shape (random packed bytes, exact dequant
  // reference from DequantRowsF16): 9..32 tokens must be served (kOk) and match
  for (int type : {12, 13}) {
    const size_t m = 64, k = 512, bb = type == 12 ? 144 : 176, row_bytes = k / 256 * bb;
    std::vector<uint8_t> raw(m * row_bytes);
    for (auto& b : raw) b = static_cast<uint8_t>(rng());
    for (size_t r = 0; r < m; ++r) for (size_t blk = 0; blk < k / 256; ++blk) { uint8_t* pp = raw.data() + r * row_bytes + blk * bb; pp[0] = 0x1e; pp[1] = 0x21; pp[2] = 0x1e; pp[3] = 0x21; }
    void* d_w; CK(hipMalloc(&d_w, raw.size())); CK(hipMemcpy(d_w, raw.data(), raw.size(), hipMemcpyHostToDevice));
    __half* d_wh; CK(hipMalloc(&d_wh, m * k * 2));
    CK(hipDeviceSynchronize());
    if (!DequantRowsF16(d_w, type, d_wh, m, k, 0)) return 3;
    CK(hipDeviceSynchronize());
    std::vector<__half> wh(m * k); CK(hipMemcpy(wh.data(), d_wh, wh.size() * 2, hipMemcpyDeviceToHost));
    for (size_t n : {size_t(2), size_t(9), size_t(21), size_t(32), size_t(33), size_t(49), size_t(100), size_t(192)}) {
      std::vector<float> x(n * k); for (auto& v : x) v = nd(rng);
      std::vector<double> ref(n * m, 0.0);
      for (size_t t = 0; t < n; ++t) for (size_t r = 0; r < m; ++r) { double a = 0; for (size_t c = 0; c < k; ++c) a += double(__half2float(wh[r * k + c])) * x[t * k + c]; ref[t * m + r] = a; }
      float *d_x, *d_y; CK(hipMalloc(&d_x, x.size() * 4)); CK(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice)); CK(hipMalloc(&d_y, n * m * 4));
      const q::Status st = GemvDecode(d_w, type, d_x, d_y, n, m, k, nullptr, 0, 0);
      CK(hipDeviceSynchronize());
      std::vector<float> y(n * m); CK(hipMemcpy(y.data(), d_y, y.size() * 4, hipMemcpyDeviceToHost));
      const double e = RelErr(y, ref);
      char b[96]; std::snprintf(b, sizeof b, "n=%zu rel-L2 %.2e %s", n, e, S(st));
      check(st == q::Status::kOk && e < 3e-3, type == 12 ? "GemvDecode multi-token Q4_K (synthetic 64x512)" : "GemvDecode multi-token Q5_K (synthetic 64x512)", "-", b);
      hipFree(d_x); hipFree(d_y);
    }
    hipFree(d_w); hipFree(d_wh);
  }
  // refusal paths with no data dependence
  {
    float* d_x; __half* d_h; float* d_y; void* d_w;
    CK(hipMalloc(&d_x, 4096 * 4)); CK(hipMalloc(&d_h, 4096 * 2)); CK(hipMalloc(&d_y, 4096 * 4)); CK(hipMalloc(&d_w, 1 << 20));
    check(GemvDecode(d_w, 99, d_x, d_y, 1, 8, 256, nullptr, 0, 0) == q::Status::kUnsupportedType, "GemvDecode unlisted type 99", "-", "");
    check(QuantGemmPrefill(d_w, 99, d_h, d_y, 4, 8, 256, nullptr, 0, 0) == q::Status::kUnsupportedType, "QuantGemmPrefill unlisted 99", "-", "");
    check(GemvDecode(d_w, 12, d_x, d_y, 1, 8, 2880, nullptr, 0, 0) == q::Status::kBadShape, "GemvDecode Q4_K cols % 256 != 0", "-", "");
    check(QuantGemmPrefill(d_w, 12, d_h, d_y, 4, 8, 2880, nullptr, 0, 0) == q::Status::kBadShape, "QuantGemmPrefill Q4_K cols % 256", "-", "");
    check(GemvDecode(d_w, 8, d_x, d_y, 9, 8, 256, nullptr, 0, 0) == q::Status::kUnsupportedType, "GemvDecode Q8_0 9 tokens (caller goes dense)", "-", "");
    check(GemvDecode(d_w, 12, d_x, d_y, 193, 8, 256, nullptr, 0, 0) == q::Status::kBadShape, "GemvDecode 193 tokens", "-", "");
    check(EmbedGather(d_w, 99, reinterpret_cast<const int32_t*>(d_x), d_y, 1, 256, 0) == false, "EmbedGather unlisted 99", "-", "");
    check(DequantRowBytes(99, 256) == 0, "DequantRowBytes unlisted 99", "-", "");
  }
  std::printf("%d/%d checks OK\n", total - bad, total);
  return bad ? 1 : 0;
}
