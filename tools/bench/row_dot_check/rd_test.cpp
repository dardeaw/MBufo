// Row-dot layer (core/quant/row_dot_traits.hpp) over EVERY format registered in quant::AllFormats: per-element
// decode (Traits::Element) against an independent reference decoder (core CPU dequant where core has one, the ggml
// byte-level spec written out below otherwise), and both reduction geometries (thread DotRow, wave LaneDot) against a
// float64 dot of those reference values. A format added to AllFormats without a matching reference here fails.
//   hipcc -O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I<repo> -x hip rd_test.cpp \
//         -x none <build>/libmbufo_core.a -o rd_test && ./rd_test
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <vector>

#include "src/core/compute/gemm/gemm_device.cuh"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/quant_traits.hpp"
#include "src/core/quant/iq2_tables.hpp"
#include "src/core/quant/row_dot_traits.hpp"
#include "src/models/deepseek_v4_flash/kernels/rocm/iq2_tables.hip.hpp"  // private device copy: compared below

namespace q = mbufo::core::quant;

#define CK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { std::fprintf(stderr, "HIP error %s line %d\n", hipGetErrorString(e__), __LINE__); std::exit(2); } } while (0)

template <class Tr>
__global__ void ElementKernel(const std::uint8_t* w, float* out, std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = Tr::Element(w, i);
}
template <class R>
__global__ void ThreadDotKernel(const std::uint8_t* w, std::size_t rb, const float* x, float* y, int m, int n) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < m) y[r] = R::DotRow(w + static_cast<std::size_t>(r) * rb, x, static_cast<std::size_t>(n));
}
template <class R>
__global__ void LaneDotKernel(const std::uint8_t* w, std::size_t rb, const float* x, float* y, int m, int n) {
  const int r = blockIdx.x;
  const std::uint32_t lane = threadIdx.x & 31u;
  const float p = R::LaneDot(w + static_cast<std::size_t>(r) * rb, x, static_cast<std::size_t>(n), lane);
  const float s = mbufo::core::compute::gemm::device::WaveSum(p);
  if (lane == 0 && r < m) y[r] = s;
}

static float H2F(std::uint16_t h) {
  const std::uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, f = h & 1023;
  float v;
  if (e == 0) v = std::ldexp(static_cast<float>(f), -24);
  else if (e == 31) v = f ? NAN : INFINITY;
  else v = std::ldexp(static_cast<float>(f | 1024), static_cast<int>(e) - 25);
  return s ? -v : v;
}
static std::uint16_t Half(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }

// ---- reference decoders: one row of n elements from packed bytes ----
using Ref = std::function<void(const std::uint8_t*, float*, std::size_t)>;
static Ref MakeRef(int type) {
  switch (type) {
    case 0: return [](const std::uint8_t* s, float* d, std::size_t n) { std::memcpy(d, s, n * 4); };
    case 1: return [](const std::uint8_t* s, float* d, std::size_t n) { for (std::size_t i = 0; i < n; ++i) d[i] = H2F(Half(s + 2 * i)); };
    case 30: return [](const std::uint8_t* s, float* d, std::size_t n) { for (std::size_t i = 0; i < n; ++i) { std::uint32_t u = static_cast<std::uint32_t>(Half(s + 2 * i)) << 16; std::memcpy(&d[i], &u, 4); } };
    case 2: return [](const std::uint8_t* s, float* d, std::size_t n) {  // Q4_0
      for (std::size_t b = 0; b < n / 32; ++b) { const std::uint8_t* k = s + b * 18; const float sc = H2F(Half(k));
        for (int j = 0; j < 16; ++j) { d[b * 32 + j] = ((k[2 + j] & 0xF) - 8) * sc; d[b * 32 + j + 16] = ((k[2 + j] >> 4) - 8) * sc; } } };
    case 3: return [](const std::uint8_t* s, float* d, std::size_t n) {  // Q4_1
      for (std::size_t b = 0; b < n / 32; ++b) { const std::uint8_t* k = s + b * 20; const float sc = H2F(Half(k)), mn = H2F(Half(k + 2));
        for (int j = 0; j < 16; ++j) { d[b * 32 + j] = (k[4 + j] & 0xF) * sc + mn; d[b * 32 + j + 16] = (k[4 + j] >> 4) * sc + mn; } } };
    case 6: return [](const std::uint8_t* s, float* d, std::size_t n) {  // Q5_0
      for (std::size_t b = 0; b < n / 32; ++b) { const std::uint8_t* k = s + b * 22; const float sc = H2F(Half(k));
        std::uint32_t qh; std::memcpy(&qh, k + 2, 4);
        for (int j = 0; j < 16; ++j) { const int x0 = (k[6 + j] & 0xF) | (((qh >> j) & 1) << 4), x1 = (k[6 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
          d[b * 32 + j] = (x0 - 16) * sc; d[b * 32 + j + 16] = (x1 - 16) * sc; } } };
    case 7: return [](const std::uint8_t* s, float* d, std::size_t n) {  // Q5_1
      for (std::size_t b = 0; b < n / 32; ++b) { const std::uint8_t* k = s + b * 24; const float sc = H2F(Half(k)), mn = H2F(Half(k + 2));
        std::uint32_t qh; std::memcpy(&qh, k + 4, 4);
        for (int j = 0; j < 16; ++j) { const int x0 = (k[8 + j] & 0xF) | (((qh >> j) & 1) << 4), x1 = (k[8 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
          d[b * 32 + j] = x0 * sc + mn; d[b * 32 + j + 16] = x1 * sc + mn; } } };
    case 8: return [](const std::uint8_t* s, float* d, std::size_t n) { mbufo::quant::DequantizeQ8_0(s, d, n); };
    case 10: return [](const std::uint8_t* s, float* d, std::size_t n) {  // Q2_K, ggml dequantize_row_q2_K
      for (std::size_t b = 0; b < n / 256; ++b) { const std::uint8_t* k = s + b * 84; const float dd = H2F(Half(k + 80)), mn = H2F(Half(k + 82));
        const std::uint8_t* qs = k + 16; int is = 0; float* y = d + b * 256;
        for (int h = 0; h < 256; h += 128) { int shift = 0;
          for (int j = 0; j < 4; ++j) {
            std::uint8_t sc = k[is++]; float dl = dd * (sc & 0xF), ml = mn * (sc >> 4);
            for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<float>((qs[l] >> shift) & 3) - ml;
            sc = k[is++]; dl = dd * (sc & 0xF); ml = mn * (sc >> 4);
            for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<float>((qs[l + 16] >> shift) & 3) - ml;
            shift += 2; }
          qs += 32; } } };
    case 12: return [](const std::uint8_t* s, float* d, std::size_t n) { mbufo::quant::DequantizeQ4_K(s, d, n); };
    case 13: return [](const std::uint8_t* s, float* d, std::size_t n) { mbufo::quant::DequantizeQ5_K(s, d, n); };
    case 14: return [](const std::uint8_t* s, float* d, std::size_t n) { mbufo::quant::DequantizeQ6_K(s, d, n); };
    case 16: return [](const std::uint8_t* s, float* d, std::size_t n) {  // IQ2_XXS, ggml dequantize_row_iq2_xxs
      static const std::uint8_t kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
      for (std::size_t b = 0; b < n / 256; ++b) { const std::uint8_t* k = s + b * 66; const float dd = H2F(Half(k));
        float* y = d + b * 256;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
          std::uint32_t aux32[2]; std::memcpy(aux32, k + 2 + ib32 * 8, 8);
          const std::uint8_t* aux8 = reinterpret_cast<const std::uint8_t*>(aux32);
          const float db = dd * (0.5f + static_cast<float>(aux32[1] >> 28)) * 0.25f;
          for (int l = 0; l < 4; ++l) {
            const std::uint64_t grid = mbufo::core::quant::detail::kIq2xxsGrid[aux8[l]];
            const std::uint8_t signs = mbufo::core::quant::detail::kIq2Signs[(aux32[1] >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j) { const float g = static_cast<float>((grid >> (8 * j)) & 0xFF); *y++ = db * g * ((signs & kmask[j]) ? -1.f : 1.f); }
          } } } };
    case 39: return [](const std::uint8_t* s, float* d, std::size_t n) { mbufo::quant::DequantizeMxfp4(s, d, n); };
    case 100: return [](const std::uint8_t* s, float* d, std::size_t n) { mbufo::quant::DequantizeQ4_0_ROCMFP4(s, d, n); };
    case 101: return [](const std::uint8_t* s, float* d, std::size_t n) { mbufo::quant::DequantizeQ4_0_ROCMFP4_FAST(s, d, n); };
    case 104: return [](const std::uint8_t* s, float* d, std::size_t n) { mbufo::quant::DequantizeQ3_0_ROCMFPX(s, d, n); };
  }
  return nullptr;
}

// Sane random content: random bytes, then valid scale fields so no NaN/Inf and magnitudes stay O(1).
static void FillRows(int type, std::vector<std::uint8_t>& w, std::size_t rows, std::size_t rb) {
  const std::uint16_t h = 0x1c00;  // ~2e-4
  for (auto& v : w) v = static_cast<std::uint8_t>(rand());
  struct Pos { std::size_t bb; int h0, h1; int byte0, byte1, byte2; };
  std::size_t bb = 0; int h0 = -1, h1 = -1; int e0 = -1, e1 = -1; std::uint8_t ev0 = 0, ev1 = 0;
  switch (type) {
    case 2: bb = 18; h0 = 0; break;
    case 3: bb = 20; h0 = 0; h1 = 2; break;
    case 6: bb = 22; h0 = 0; break;
    case 7: bb = 24; h0 = 0; h1 = 2; break;
    case 8: bb = 34; h0 = 0; break;
    case 10: bb = 84; h0 = 80; h1 = 82; break;
    case 12: bb = 144; h0 = 0; h1 = 2; break;
    case 13: bb = 176; h0 = 0; h1 = 2; break;
    case 14: bb = 210; h0 = 208; break;
    case 16: bb = 66; h0 = 0; break;
    case 39: bb = 17; e0 = 0; ev0 = 127; break;
    case 100: bb = 18; e0 = 16; e1 = 17; ev0 = ev1 = 0x38; break;
    case 101: bb = 17; e0 = 16; ev0 = 0x38; break;
    case 104: bb = 14; e0 = 12; e1 = 13; ev0 = ev1 = 0x38; break;
    default: break;  // F32/F16/BF16 handled below
  }
  if (bb) {
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t b = 0; b < rb / bb; ++b) {
        std::uint8_t* p = w.data() + r * rb + b * bb;
        if (h0 >= 0) { p[h0] = h & 0xff; p[h0 + 1] = h >> 8; }
        if (h1 >= 0) { p[h1] = h & 0xff; p[h1 + 1] = h >> 8; }
        if (e0 >= 0) p[e0] = ev0;
        if (e1 >= 0) p[e1] = ev1;
      }
  } else {  // float formats
    const std::size_t el = type == 0 ? 4 : 2, n = w.size() / el;
    for (std::size_t i = 0; i < n; ++i) {
      const float v = (rand() / static_cast<float>(RAND_MAX) - 0.5f) * 2.f;
      if (type == 0) std::memcpy(&w[i * 4], &v, 4);
      else if (type == 1) { const __half hv = __float2half(v); std::memcpy(&w[i * 2], &hv, 2); }
      else { std::uint32_t u; std::memcpy(&u, &v, 4); const std::uint16_t t = static_cast<std::uint16_t>(u >> 16); std::memcpy(&w[i * 2], &t, 2); }
    }
  }
}

int main() {
  const int m = 48, n = 1024;
  int bad = 0, formats = 0;
  // the core IQ2_XXS tables must equal the DeepSeek kernels' device copy byte for byte
  {
    std::vector<std::uint64_t> g(256); std::vector<std::uint8_t> sg(128);
    CK(hipMemcpyFromSymbol(g.data(), HIP_SYMBOL(hip_iq2xxs_grid), 256 * 8));
    CK(hipMemcpyFromSymbol(sg.data(), HIP_SYMBOL(hip_ksigns_iq2xs), 128));
    const bool eq = !std::memcmp(g.data(), mbufo::core::quant::detail::kIq2xxsGrid, 256 * 8) && !std::memcmp(sg.data(), mbufo::core::quant::detail::kIq2Signs, 128);
    std::printf("IQ2_XXS tables equal the DeepSeek device copy: %s\n", eq ? "yes" : "NO");
    bad += !eq;
  }
  std::vector<float> hx(n);
  srand(17);
  for (auto& v : hx) v = (rand() / static_cast<float>(RAND_MAX) - 0.5f) * 2.f;
  float* dx; CK(hipMalloc(&dx, n * 4)); CK(hipMemcpy(dx, hx.data(), n * 4, hipMemcpyHostToDevice));
  std::vector<int> types;
  (void)0;
  // every registered format: walk the registry through its own dispatch over all plausible codes
  for (int t = 0; t < 128; ++t) {
    bool known = q::RowFormats::Dispatch(t, [](auto) {});
    if (known) types.push_back(t);
  }
  for (int t : types) {
    std::size_t rb = 0, be = 1;
    (void)q::AllFormats::Dispatch(t, [&](auto tr) { using T = decltype(tr); rb = T::RowBytes(n); be = T::kBlockElems; (void)be; });
    const Ref ref = MakeRef(t);
    if (!ref) { std::printf("type %3d: NO REFERENCE DECODER in this test -> FAIL\n", t); ++bad; continue; }
    ++formats;
    std::vector<std::uint8_t> w(static_cast<std::size_t>(m) * rb);
    FillRows(t, w, m, rb);
    std::uint8_t* dw; CK(hipMalloc(&dw, w.size())); CK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    float *dy, *de; CK(hipMalloc(&dy, m * 4)); CK(hipMalloc(&de, static_cast<std::size_t>(n) * 4));
    // reference values + float64 dots
    std::vector<float> rv(static_cast<std::size_t>(m) * n);
    for (int r = 0; r < m; ++r) ref(w.data() + static_cast<std::size_t>(r) * rb, rv.data() + static_cast<std::size_t>(r) * n, n);
    std::vector<double> rd(m, 0.0);
    double ref_mag = 0;
    for (int r = 0; r < m; ++r) for (int i = 0; i < n; ++i) { rd[r] += static_cast<double>(rv[static_cast<std::size_t>(r) * n + i]) * hx[i]; ref_mag = std::fmax(ref_mag, std::fabs(rv[static_cast<std::size_t>(r) * n + i])); }
    // 1) Element vs reference, row 0 and row m-1
    double max_el = 0;
    (void)q::AllFormats::Dispatch(t, [&](auto tr) {
      using T = decltype(tr);
      for (int r : {0, m - 1}) {
        hipLaunchKernelGGL((ElementKernel<T>), dim3((n + 255) / 256), dim3(256), 0, 0, dw + static_cast<std::size_t>(r) * rb, de, static_cast<std::uint32_t>(n));
        CK(hipDeviceSynchronize());
        std::vector<float> got(n);
        CK(hipMemcpy(got.data(), de, n * 4, hipMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) max_el = std::fmax(max_el, std::fabs(got[i] - rv[static_cast<std::size_t>(r) * n + i]) / (std::fabs(rv[static_cast<std::size_t>(r) * n + i]) + 1e-3 * ref_mag));
      }
    });
    // 2) thread and lane geometry dots
    double rel[2] = {0, 0};
    (void)q::RowFormats::Dispatch(t, [&](auto rdot) {
      using R = decltype(rdot);
      std::vector<float> got(m);
      hipLaunchKernelGGL((ThreadDotKernel<R>), dim3((m + 63) / 64), dim3(64), 0, 0, dw, rb, dx, dy, m, n);
      CK(hipDeviceSynchronize()); CK(hipMemcpy(got.data(), dy, m * 4, hipMemcpyDeviceToHost));
      double num = 0, den = 0; for (int r = 0; r < m; ++r) { num += (got[r] - rd[r]) * (got[r] - rd[r]); den += rd[r] * rd[r]; }
      rel[0] = std::sqrt(num / den);
      hipLaunchKernelGGL((LaneDotKernel<R>), dim3(m), dim3(32), 0, 0, dw, rb, dx, dy, m, n);
      CK(hipDeviceSynchronize()); CK(hipMemcpy(got.data(), dy, m * 4, hipMemcpyDeviceToHost));
      num = 0; for (int r = 0; r < m; ++r) num += (got[r] - rd[r]) * (got[r] - rd[r]);
      rel[1] = std::sqrt(num / den);
    });
    const bool ok = max_el < 2e-6 && rel[0] < 1e-5 && rel[1] < 1e-5;
    std::printf("type %3d: element max rel err %.1e  thread-dot rel-L2 %.1e  lane-dot rel-L2 %.1e  %s\n", t, max_el, rel[0], rel[1], ok ? "OK" : "FAIL");
    bad += !ok;
    hipFree(dw); hipFree(dy); hipFree(de);
  }
  // unregistered codes are refused (never decoded as something else)
  bool refuse = true;
  for (int t : {4, 5, 9, 11, 15, 17, 99, 102, 103, 127})
    refuse &= !q::RowFormats::Dispatch(t, [](auto) {});
  std::printf("%d formats checked; unregistered codes refused: %s\n", formats, refuse ? "yes" : "NO");
  bad += !refuse;
  std::printf(bad ? "FAILED\n" : "all OK\n");
  return bad ? 1 : 0;
}
