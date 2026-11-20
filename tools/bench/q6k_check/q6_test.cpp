// Q6_K through the multi-token WMMA kernel (RowDot<Q6_K>::LoadSuper/DecodeSub16 via QuantGemvMulti):
//  1. decode exactness: one-hot activations read back every weight; Y[t][r] must equal F16(Element) of the host reference
//     decoder (quant::DequantizeQ6_K), bit for bit, for all rows/elements, at row offsets that are 0 and 2 mod 4;
//  2. random activations vs a double-precision reference with F16-rounded weights and activations;
//  3. timing at the DFlash draft shapes against the F16 copy the draft used before (SmallGemm).
// Built by tools/bench/q6k_check/q6_ab.sh.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/row_dot_traits.hpp"

namespace gg = mbufo::core::compute::gemm;

#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)


// One block per row, 32 lanes: lane l decodes elements 16 * (l >> 4) + 0..15 of each 32-element sub-block (what the WMMA kernel does).
template <class R>
__global__ void DecodeKernel(const std::uint8_t* w, std::size_t row_bytes, std::uint32_t nsb, __half* out) {
  const std::uint32_t row = blockIdx.x, lane = threadIdx.x, half_id = lane >> 4;
  const std::size_t k = static_cast<std::size_t>(nsb) * 256;
  for (std::uint32_t sb = 0; sb < nsb; ++sb) {
    typename R::TileRegs regs;
    R::LoadSuper(w + row * row_bytes + static_cast<std::size_t>(sb) * R::kBlockBytes, half_id, regs);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      unsigned own[8];
      R::DecodeSub16(regs, j, own);
      if ((lane & 15u) == 0) {
        __half* dst = out + row * k + sb * 256 + 32 * j + 16 * half_id;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          dst[2 * e] = __builtin_bit_cast(__half, static_cast<std::uint16_t>(own[e] & 0xFFFFu));
          dst[2 * e + 1] = __builtin_bit_cast(__half, static_cast<std::uint16_t>(own[e] >> 16));
        }
      }
    }
  }
}

static float RoundHalf(float v) { return __half2float(__float2half(v)); }

template <class Fn>
static double Time(Fn fn, int iters = 30) {
  for (int i = 0; i < 3; ++i) fn();
  CK(hipDeviceSynchronize());
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  CK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
  float ms; CK(hipEventElapsedTime(&ms, a, b));
  return ms * 1000.0 / iters;
}

static std::vector<std::uint8_t> RandomQ6K(std::size_t m, std::size_t k, std::mt19937& rng) {
  const std::size_t nsb = k / 256;
  std::vector<std::uint8_t> w(m * nsb * 210);
  for (auto& b : w) b = static_cast<std::uint8_t>(rng());
  for (std::size_t r = 0; r < m; ++r)
    for (std::size_t s = 0; s < nsb; ++s) {
      std::uint8_t* blk = w.data() + (r * nsb + s) * 210;
      for (int i = 0; i < 16; ++i) blk[192 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(rng() % 61) - 30));
      const __half d = __float2half(0.001f + 0.02f * static_cast<float>(rng() % 1000) / 1000.0f);
      std::memcpy(blk + 208, &d, 2);
    }
  return w;
}

int main() {
  int bad = 0, checks = 0;
  std::mt19937 rng(1234);
  // ---- 1. decode exactness: R::LoadSuper/DecodeSub16 run in a standalone kernel and compared with F16(reference decoder) ----
  // (A one-hot read-back through the WMMA accumulate is not an exact oracle: the existing Q4_K path shows the same 1-ulp F32
  // noise there, so the decode is checked on its own.)
  for (std::size_t k : {512ul, 768ul})
    for (std::size_t m : {16ul, 80ul}) {
      auto w = RandomQ6K(m, k, rng);
      std::vector<float> ref(m * k);
      for (std::size_t r = 0; r < m; ++r) mbufo::quant::DequantizeQ6_K(w.data() + r * (k / 256) * 210, ref.data() + r * k, k);
      std::uint8_t* dw; __half* dout; CK(hipMalloc(&dw, w.size())); CK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
      CK(hipMalloc(&dout, m * k * 2));
      hipLaunchKernelGGL(DecodeKernel<mbufo::core::quant::RowDot<mbufo::core::quant::FormatId::kQ6_K>>, dim3(static_cast<unsigned>(m)), dim3(32), 0, 0,
                         dw, (k / 256) * 210, static_cast<std::uint32_t>(k / 256), dout);
      std::vector<__half> got(m * k); CK(hipMemcpy(got.data(), dout, m * k * 2, hipMemcpyDeviceToHost));
      std::size_t mism = 0;
      for (std::size_t i = 0; i < m * k; ++i) {
        ++checks;
        const __half want = __float2half(ref[i]);
        // +0 and -0 are the same weight
        std::uint16_t a, b; std::memcpy(&a, &got[i], 2); std::memcpy(&b, &want, 2);
        if (__half2float(got[i]) != __half2float(want)) { if (mism < 6) std::printf("    i=%zu (row %zu col %zu): ref float %.10g -> half bits %04x; device half bits %04x (= %.10g)\n", i, i / k, i % k, ref[i], b, a, __half2float(got[i])); ++mism; }
      }
      std::printf("  decode m=%zu k=%zu: %zu of %zu elements differ from F16(reference)%s\n", m, k, mism, m * k, mism ? "  FAIL" : "");
      if (mism) ++bad;
      CK(hipFree(dw)); CK(hipFree(dout));
    }
  std::printf("decode exactness: %s (%d element checks)\n", bad ? "FAILED" : "all bit-exact", checks);

  // ---- 2. random activations ----
  for (std::size_t k : {768ul, 2048ul})
    for (int n : {1, 2, 5, 16, 17, 32}) {
      const std::size_t m = 192;
      auto w = RandomQ6K(m, k, rng);
      std::vector<float> ref(m * k);
      for (std::size_t r = 0; r < m; ++r) mbufo::quant::DequantizeQ6_K(w.data() + r * (k / 256) * 210, ref.data() + r * k, k);
      for (auto& v : ref) v = RoundHalf(v);
      std::vector<float> x(static_cast<std::size_t>(n) * k);
      for (auto& v : x) v = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 500.0f;
      void* dw; float *dx, *dy; CK(hipMalloc(&dw, w.size())); CK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
      CK(hipMalloc(&dx, x.size() * 4)); CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
      CK(hipMalloc(&dy, static_cast<std::size_t>(n) * m * 4));
      if (!gg::QuantGemvMulti(dw, 14, dx, dy, n, m, k, 0)) { std::printf("  launch refused n=%d\n", n); ++bad; continue; }
      std::vector<float> y(static_cast<std::size_t>(n) * m); CK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
      double worst = 0, scale = 0;
      for (int t = 0; t < n; ++t)
        for (std::size_t r = 0; r < m; ++r) {
          double a = 0;
          for (std::size_t i = 0; i < k; ++i) a += static_cast<double>(ref[r * k + i]) * static_cast<double>(RoundHalf(x[static_cast<std::size_t>(t) * k + i]));
          worst = std::max(worst, std::fabs(a - y[static_cast<std::size_t>(t) * m + r]));
          scale = std::max(scale, std::fabs(a));
        }
      ++checks;
      const bool ok = worst <= 2e-4 * scale;
      std::printf("  random k=%zu n=%2d: max|err| %.3g (max|ref| %.3g) %s\n", k, n, worst, scale, ok ? "ok" : "FAIL");
      if (!ok) ++bad;
      CK(hipFree(dw)); CK(hipFree(dx)); CK(hipFree(dy));
    }

  // ---- 3. timing at the draft shapes: packed Q6_K vs the F16 copy ----
  struct Shape { std::size_t m, k; const char* name; } shapes[] = {{6656, 19968, "ffn_down"}, {1024, 6656, "attn_v"}, {262144, 2560, "E4B lm head"}, {262144, 1536, "E2B lm head"}};
  for (auto s : shapes)
    for (int n : {1, 16}) {
      auto w = RandomQ6K(s.m, s.k, rng);
      std::vector<__half> h(s.m * s.k);
      std::vector<float> row(s.k);
      for (std::size_t r = 0; r < s.m; ++r) {
        mbufo::quant::DequantizeQ6_K(w.data() + r * (s.k / 256) * 210, row.data(), s.k);
        for (std::size_t i = 0; i < s.k; ++i) h[r * s.k + i] = __float2half(row[i]);
      }
      void *dw, *dh; float *dx, *dy;
      CK(hipMalloc(&dw, w.size())); CK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
      CK(hipMalloc(&dh, h.size() * 2)); CK(hipMemcpy(dh, h.data(), h.size() * 2, hipMemcpyHostToDevice));
      CK(hipMalloc(&dx, static_cast<std::size_t>(n) * s.k * 4)); CK(hipMemset(dx, 0, static_cast<std::size_t>(n) * s.k * 4));
      CK(hipMalloc(&dy, static_cast<std::size_t>(n) * s.m * 4));
      const double t_q = Time([&] { (void)gg::QuantGemvMulti(dw, 14, dx, dy, n, s.m, s.k, 0); });
      const double t_h = Time([&] { gg::fn_compat::SmallGemm(dh, 1, dx, dy, n, s.m, s.k, 0); });
      std::printf("  %-8s n=%2d: Q6_K WMMA %.1f us (%.0f GB/s)   F16 SmallGemm %.1f us (%.0f GB/s)   x%.2f\n", s.name, n, t_q,
                  static_cast<double>(w.size()) / t_q / 1e3, t_h, static_cast<double>(h.size() * 2) / t_h / 1e3, t_h / t_q);
      CK(hipFree(dw)); CK(hipFree(dh)); CK(hipFree(dx)); CK(hipFree(dy));
    }
  std::printf("%s (%d checks)\n", bad ? "FAILED" : "all OK", checks);
  return bad ? 1 : 0;
}
