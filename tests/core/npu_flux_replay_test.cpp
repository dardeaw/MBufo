// Per-block replay: run my blocks on pipeline stage inputs, compare.
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
  using mbufo::core::npu::dit::FluxDitNpu;
  constexpr std::size_t Si = 64, St = 22, D = 3072, Sj = 86;
  const std::string dir =
      "/mnt/nvme-data/mbufo-models/flux2-klein-4b/transformer";
  const std::string pre = "/home/dardeaw/q_ab/r_";
  mbufo::core::npu::NpuQuantGemmEngine engine;
  std::string err;
  assert(
      engine.Init("/mnt/nvme-data/mbufo-models/npu-kernels/quant_gemm", &err));
  FluxDitNpu dit;
  assert(dit.Open(dir, &engine, NpuQuantMode::kQ8Bfp16, {}, &err));
  const std::vector<float> mod_img = Load(pre + "mod_img.f32", 6 * D);
  const std::vector<float> mod_txt = Load(pre + "mod_txt.f32", 6 * D);
  const std::vector<float> mod_sgl = Load(pre + "mod_sgl.f32", 3 * D);
  const std::vector<float> cos = Load(pre + "cos.f32", Sj * 128);
  const std::vector<float> sin = Load(pre + "sin.f32", Sj * 128);
  double ms = 0.0;
  {
    std::vector<float> my_cos, my_sin;
    dit.BuildRopeTables(St, Si, &my_cos, &my_sin);
    const std::string bp = "/home/dardeaw/q_ab/b0_";
    const std::vector<float> ref_cos = Load(bp + "cos.f32", Sj * 128);
    const std::vector<float> ref_sin = Load(bp + "sin.f32", Sj * 128);
    std::printf("[rope] cos=%.6f sin=%.6f\n", Cosine(my_cos, ref_cos),
                Cosine(my_sin, ref_sin));
    std::fflush(stdout);
  }
  float worst = 1.0F;
  int worst_bi = -99;
  for (int bi = 0; bi < 5; ++bi) {
    std::string b = std::to_string(bi);
    std::string pin = (bi == 0)
                          ? pre + "img0.f32"
                          : pre + "dimg" + std::to_string(bi - 1) + ".f32";
    std::string tin = (bi == 0)
                          ? pre + "txt0.f32"
                          : pre + "dtxt" + std::to_string(bi - 1) + ".f32";
    std::vector<float> img = Load(pin, Si * D);
    std::vector<float> txt = Load(tin, St * D);
    assert(dit.DebugDoubleBlock(&img, &txt, Si, St, bi, mod_img, mod_txt, cos,
                                sin, &ms, &err));
    std::vector<float> ri = Load(pre + "dimg" + b + ".f32", Si * D);
    std::vector<float> rt = Load(pre + "dtxt" + b + ".f32", St * D);
    const float ci = Cosine(img, ri);
    const float ct = Cosine(txt, rt);
    std::printf("[dbl%d] cos_img=%.6f cos_txt=%.6f\n", bi, ci, ct);
    if (ci < worst) {
      worst = ci;
      worst_bi = bi;
    }
    if (ct < worst) {
      worst = ct;
      worst_bi = bi;
    }
  }
  for (int bi = 0; bi < 20; ++bi) {
    std::string b = std::to_string(bi);
    std::string jin =
        (bi == 0) ? pre + "joint0.f32" : pre + "joint" + b + ".f32";
    std::vector<float> joint = Load(jin, Sj * D);
    assert(dit.DebugSingleBlock(&joint, Sj, bi, mod_sgl, cos, sin, &ms, &err));
    std::string nxt = (bi == 19)
                          ? pre + "joint20.f32"
                          : pre + "joint" + std::to_string(bi + 1) + ".f32";
    std::vector<float> ref = Load(nxt, Sj * D);
    const float c = Cosine(joint, ref);
    std::printf("[sgl%d] cos=%.6f\n", bi, c);
    if (c < worst) {
      worst = c;
      worst_bi = 100 + bi;
    }
  }
  std::printf("WORST=%.6f @%d\n", worst, worst_bi);
  std::fflush(stdout);
  return 0;
}
