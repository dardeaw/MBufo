// Runs FlashSinkDecode on the reference cases and compares with the float64 reference.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/core/compute/attention/flash_sink.hpp"

using namespace mbufo::core::compute::flash_sink;

#define CK(x) do { hipError_t err__ = (x); if (err__ != hipSuccess) { std::fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(err__), __LINE__); return 2; } } while (0)

static std::vector<char> ReadAll(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
  const std::string dir = argv[1];
  std::ifstream meta(dir + "/meta.txt");
  std::string name; unsigned D, heads, kvh, n_kv, window, has_sink;
  int bad = 0, total = 0;
  while (meta >> name >> D >> heads >> kvh >> n_kv >> window >> has_sink) {
    auto qb = ReadAll(dir + "/" + name + ".q"), kb = ReadAll(dir + "/" + name + ".k"),
         vb = ReadAll(dir + "/" + name + ".v"), sb = ReadAll(dir + "/" + name + ".sink"),
         rb = ReadAll(dir + "/" + name + ".ref");
    float *d_q, *d_s, *d_o, *d_p; __half *d_k, *d_v;
    CK(hipMalloc(&d_q, qb.size())); CK(hipMemcpy(d_q, qb.data(), qb.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&d_k, kb.size())); CK(hipMemcpy(d_k, kb.data(), kb.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&d_v, vb.size())); CK(hipMemcpy(d_v, vb.data(), vb.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&d_s, sb.size())); CK(hipMemcpy(d_s, sb.data(), sb.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&d_o, heads * D * 4));
    const size_t pbytes = FlashSinkDecodeScratchBytes(kFlashSinkMaxTokens, heads);
    CK(hipMalloc(&d_p, pbytes));
    const bool ok_launch = FlashSinkDecode(d_q, d_k, d_v, has_sink ? d_s : nullptr, d_o, 1, n_kv - 1, n_kv, heads, kvh, D,
                                           window, 1.0f / std::sqrt(static_cast<float>(D)), 0, d_p, pbytes);
    CK(hipDeviceSynchronize());
    std::vector<float> o(heads * D);
    CK(hipMemcpy(o.data(), d_o, o.size() * 4, hipMemcpyDeviceToHost));
    const float* ref = reinterpret_cast<const float*>(rb.data());
    double num = 0, den = 0, maxe = 0;
    for (size_t i = 0; i < o.size(); ++i) { double e = o[i] - ref[i]; num += e * e; den += double(ref[i]) * ref[i]; if (std::fabs(e) > maxe) maxe = std::fabs(e); }
    const double rel = std::sqrt(num / den);
    const bool ok = ok_launch && rel < 2e-5;
    std::printf("%-4s D=%3u heads=%2u kvh=%u n_kv=%5u window=%3u sink=%u  rel-L2 %.2e max-abs %.2e  %s\n", name.c_str(), D, heads, kvh, n_kv, window, has_sink, rel, maxe, ok ? "OK" : "FAIL");
    ++total; if (!ok) ++bad;
    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_s); hipFree(d_o); hipFree(d_p);
  }
  std::printf("%d/%d cases OK\n", total - bad, total);
  return bad ? 1 : 0;
}
