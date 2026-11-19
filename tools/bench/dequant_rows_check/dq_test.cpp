// Checks core::compute::gemm::EmbedGather / DequantRowsF16 against reference rows.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"

using namespace gufo::core::compute::gemm;

#define CK(x) do { hipError_t err__ = (x); if (err__ != hipSuccess) { std::fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(err__), __LINE__); return 2; } } while (0)

int main(int argc, char** argv) {
  const std::string dir = argv[1];
  std::ifstream meta(dir + "/meta.txt");
  std::string name; int type; size_t cols, nrows;
  int bad = 0;
  while (meta >> name >> type >> cols >> nrows) {
    std::ifstream bf(dir + "/" + name + ".bin", std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(bf)), std::istreambuf_iterator<char>());
    const size_t rb = DequantRowBytes(type, cols);
    if (rb == 0 || raw.size() != rb * nrows) {
      std::printf("%s: row bytes mismatch (file %zu, expected %zu x %zu)\n", name.c_str(), raw.size(), rb, nrows);
      bad++; continue;
    }
    void* d_w; CK(hipMalloc(&d_w, raw.size()));
    CK(hipMemcpy(d_w, raw.data(), raw.size(), hipMemcpyHostToDevice));
    std::vector<int32_t> ids(nrows);
    for (size_t i = 0; i < nrows; ++i) ids[i] = static_cast<int32_t>(i);
    int32_t* d_ids; CK(hipMalloc(&d_ids, ids.size() * 4));
    CK(hipMemcpy(d_ids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
    float* d_e; CK(hipMalloc(&d_e, nrows * cols * 4));
    __half* d_h; CK(hipMalloc(&d_h, nrows * cols * 2));
    if (!EmbedGather(d_w, type, d_ids, d_e, nrows, cols, 0)) { std::printf("%s: EmbedGather refused\n", name.c_str()); bad++; continue; }
    if (!DequantRowsF16(d_w, type, d_h, nrows, cols, 0)) { std::printf("%s: DequantRowsF16 refused\n", name.c_str()); bad++; continue; }
    CK(hipDeviceSynchronize());
    std::vector<float> e(nrows * cols);
    std::vector<__half> h(nrows * cols);
    CK(hipMemcpy(e.data(), d_e, e.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(h.data(), d_h, h.size() * 2, hipMemcpyDeviceToHost));
    std::vector<float> hf(h.size());
    for (size_t i = 0; i < h.size(); ++i) hf[i] = __half2float(h[i]);
    std::ofstream(dir + "/" + name + ".embed.f32", std::ios::binary).write(reinterpret_cast<char*>(e.data()), e.size() * 4);
    std::ofstream(dir + "/" + name + ".deq.f32", std::ios::binary).write(reinterpret_cast<char*>(hf.data()), hf.size() * 4);
    (void)hipFree(d_w); (void)hipFree(d_ids); (void)hipFree(d_e); (void)hipFree(d_h);
  }
  return bad ? 1 : 0;
}
