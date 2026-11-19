// bench: FlashWmmaPrefill timing at engine shapes, random data, no reference.  bench D heads kvh n start window sink
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "src/core/compute/attention/flash_wmma_gqa.hpp"
using namespace gufo::core::compute::flash_wmma_gqa;
int main(int argc, char** argv) {
  struct C { unsigned D, heads, kvh, n, start, window, sink; const char* what; };
  std::vector<C> cases;
  if (argc >= 8) cases.push_back({(unsigned)atoi(argv[1]), (unsigned)atoi(argv[2]), (unsigned)atoi(argv[3]), (unsigned)atoi(argv[4]), (unsigned)atoi(argv[5]), (unsigned)atoi(argv[6]), (unsigned)atoi(argv[7]), "custom"});
  else cases = {{128, 32, 8, 512, 6000, 0, 0, "llama 8B  n=512 @6K"}, {128, 32, 8, 512, 512, 0, 0, "llama 8B  n=512 @1K"},
                {128, 32, 2, 2048, 4000, 0, 0, "glimmer   n=2048 @6K"}, {64, 64, 8, 256, 5000, 0, 1, "gptoss full n=256 @5K"},
                {64, 64, 8, 256, 5000, 128, 1, "gptoss swa  n=256 @5K"}, {64, 64, 8, 2048, 2000, 0, 1, "gptoss full n=2048 @4K"}};
  for (auto c : cases) {
    const size_t kvlen = c.start + c.n;
    float *q, *o, *s = nullptr; __half *k, *v;
    hipMalloc(&q, (size_t)c.n * c.heads * c.D * 4); hipMalloc(&o, (size_t)c.n * c.heads * c.D * 4);
    hipMalloc(&k, kvlen * c.kvh * c.D * 2); hipMalloc(&v, kvlen * c.kvh * c.D * 2);
    std::vector<__half> hk(kvlen * c.kvh * c.D); for (auto& x : hk) x = __float2half((rand() / (float)RAND_MAX - 0.5f) * 2);
    hipMemcpy(k, hk.data(), hk.size() * 2, hipMemcpyHostToDevice); hipMemcpy(v, hk.data(), hk.size() * 2, hipMemcpyHostToDevice);
    std::vector<float> hq((size_t)c.n * c.heads * c.D); for (auto& x : hq) x = (rand() / (float)RAND_MAX - 0.5f) * 2;
    hipMemcpy(q, hq.data(), hq.size() * 4, hipMemcpyHostToDevice);
    if (c.sink) { hipMalloc(&s, c.heads * 4); std::vector<float> hs(c.heads, 1.0f); hipMemcpy(s, hs.data(), c.heads * 4, hipMemcpyHostToDevice); }
    const float scale = 1.0f / std::sqrt((float)c.D);
    for (int i = 0; i < 3; ++i) (void)FlashWmmaPrefill(q, k, v, s, o, c.n, c.start, c.heads, c.kvh, c.D, c.window, scale, 0);
    hipDeviceSynchronize();
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); hipEventRecord(a);
    const int it = 10; for (int i = 0; i < it; ++i) (void)FlashWmmaPrefill(q, k, v, s, o, c.n, c.start, c.heads, c.kvh, c.D, c.window, scale, 0);
    hipEventRecord(b); hipEventSynchronize(b); float ms; hipEventElapsedTime(&ms, a, b); ms /= it;
    double keys = 0; for (unsigned t = 0; t < c.n; ++t) { const unsigned qp = c.start + t; const unsigned lo = c.window ? (qp + 1 > c.window ? qp + 1 - c.window : 0) : 0; keys += qp - lo + 1; }
    printf("%-26s %8.3f ms %6.1f TFLOPS\n", c.what, ms, 4.0 * c.heads * keys * c.D / (ms * 1e-3) / 1e12);
    hipFree(q); hipFree(o); hipFree(k); hipFree(v); if (s) hipFree(s);
  }
  return 0;
}