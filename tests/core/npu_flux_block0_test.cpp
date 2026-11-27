// NPU side of block-0 isolation: load pipeline-built inputs, run one
// DoubleBlock, compare vs pipeline block-0 outputs.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/dit_npu_common.hpp"
#include "src/core/compute/npu/flux_dit_npu.hpp"
#include "src/core/compute/npu/npu_quant_gemm.hpp"

namespace {

std::vector<float> Load(const char* path, std::size_t n) {
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
  using mbufo::core::npu::dit::FluxDitNpu;
  const std::string dir = "/mnt/nvme-data/mbufo-models/flux2-klein-4b/transformer";
  const std::string pre = "/home/dardeaw/q_ab/b0_";
  constexpr std::size_t Si = 64, St = 22, D = 3072;
  mbufo::core::npu::NpuQuantGemmEngine engine;
  std::string err;
  assert(engine.Init("/mnt/nvme-data/mbufo-models/npu-kernels/quant_gemm", &err));
  FluxDitNpu dit;
  assert(dit.Open(dir, &engine, NpuQuantMode::kQ8Bfp16, {}, &err));
  std::vector<float> img = Load((pre + "img_in.f32").c_str(), Si * D);
  std::vector<float> txt = Load((pre + "txt_in.f32").c_str(), St * D);
  std::vector<float> mod_img = Load((pre + "mod_img.f32").c_str(), 6 * D);
  std::vector<float> mod_txt = Load((pre + "mod_txt.f32").c_str(), 6 * D);
  std::vector<float> cos = Load((pre + "cos.f32").c_str(), (Si + St) * 128);
  std::vector<float> sin = Load((pre + "sin.f32").c_str(), (Si + St) * 128);
  double ms = 0.0;
  assert(dit.DebugDoubleBlock(&img, &txt, Si, St, 0, mod_img, mod_txt, cos,
                              sin, &ms, &err));
  std::vector<float> img_ref = Load((pre + "img_ref.f32").c_str(), Si * D);
  std::vector<float> txt_ref = Load((pre + "txt_ref.f32").c_str(), St * D);
  std::printf("[block0] cos_img=%.6f cos_txt=%.6f\n", Cosine(img, img_ref),
              Cosine(txt, txt_ref));
  std::fflush(stdout);
  {
    std::ofstream oi("/home/dardeaw/q_ab/b0_img_npu.f32", std::ios::binary);
    oi.write(reinterpret_cast<const char*>(img.data()),
             static_cast<std::streamsize>(img.size() * sizeof(float)));
    std::ofstream ot("/home/dardeaw/q_ab/b0_txt_npu.f32", std::ios::binary);
    ot.write(reinterpret_cast<const char*>(txt.data()),
             static_cast<std::streamsize>(txt.size() * sizeof(float)));
  }
  // Single-block isolation on pipeline block-0 outputs.
  {
    constexpr std::size_t Sj = 86;
    std::vector<float> joint = Load((pre + "joint_in.f32").c_str(), Sj * D);
    std::vector<float> mod_sgl = Load((pre + "mod_sgl.f32").c_str(), 3 * D);
    std::vector<float> ref = Load((pre + "sgl_ref.f32").c_str(), Sj * D);
    assert(dit.DebugSingleBlock(&joint, Sj, 0, mod_sgl, cos, sin, &ms, &err));
    std::printf("[single0] cos=%.6f\n", Cosine(joint, ref));
    std::fflush(stdout);
  }
  return 0;
}
