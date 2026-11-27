// Z-Image block isolation: NPU Block vs pipeline block outputs.
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/dit_npu_common.hpp"
#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/zimage_dit_npu.hpp"

namespace {

std::vector<float> Load(const std::string& path, std::size_t n) {
  std::vector<float> v(n);
  std::ifstream ifs(path, std::ios::binary);
  assert(ifs.good());
  ifs.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
  assert(ifs.good());
  return v;
}

float Cosine(std::span<const float> a, std::span<const float> b) {
  assert(a.size() == b.size());
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += (double)a[i] * b[i];
    na += (double)a[i] * a[i];
    nb += (double)b[i] * b[i];
  }
  return (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}

}  // namespace

int main() {
  using mbufo::core::npu::NpuQuantMode;
  using mbufo::core::npu::dit::ZImageDitNpu;
  constexpr std::size_t D = 3840;
  const std::string dir = "/mnt/nvme-data/mbufo-models/z-image-turbo/transformer";
  const std::string pre = "/home/dardeaw/q_ab/";
  mbufo::core::npu::NpuQuantGemmEngine engine;
  std::string err;
  assert(engine.Init("/mnt/nvme-data/mbufo-models/npu-kernels/quant_gemm", &err));
  ZImageDitNpu dit;
  assert(dit.Open(dir, &engine, NpuQuantMode::kQ8Bfp16, {}, &err));
  double ms = 0.0;
  // Refiner-0 (no mod), cap32 tokens.
  {
    std::vector<float> h = Load(pre + "z_cap32_in.f32", 32 * D);
    std::vector<float> cos = Load(pre + "z_rope_cos.f32", 96 * 128);
    std::vector<float> sin = Load(pre + "z_rope_sin.f32", 96 * 128);
    std::vector<float> ccos(cos.begin() + 64 * 128, cos.end());
    std::vector<float> csin(sin.begin() + 64 * 128, sin.end());
    assert(dit.Block("context_refiner.0", &h, 32, nullptr, ccos, csin, &ms, &err));
    std::vector<float> ref = Load(pre + "z_ref0_out.f32", 32 * D);
    std::printf("[zrefiner0] cos=%.6f\n", Cosine(h, ref));
  }
  // Noise-refiner-0 (mod), img tokens.
  {
    std::vector<float> h = Load(pre + "z_img_in.f32", 64 * D);
    std::vector<float> mod = Load(pre + "z_mod_n0.f32", 4 * D);
    std::vector<float> cos = Load(pre + "z_rope_cos.f32", 96 * 128);
    std::vector<float> sin = Load(pre + "z_rope_sin.f32", 96 * 128);
    std::vector<float> icos(cos.begin(), cos.begin() + 64 * 128);
    std::vector<float> isin(sin.begin(), sin.begin() + 64 * 128);
    assert(dit.Block("noise_refiner.0", &h, 64, mod.data(), icos, isin, &ms, &err));
    std::vector<float> ref = Load(pre + "z_noise0_out.f32", 64 * D);
    std::printf("[znoise0] cos=%.6f\n", Cosine(h, ref));
  }
  std::fflush(stdout);
  return 0;
}
