#pragma once
// FLUX.2-klein DiT NPU path, TRUE PORT of the diffusers pipeline math
// (P8a). v1 followed the parity shell (RMSNorm / separate attention / no
// txt FFN / 1D rope / tanh gates) and diverged from the pipeline (step-0
// cos 0.996, E2E PSNR ~7dB). This version ports the real math:
//
// - LayerNorm (no affine) everywhere the pipeline uses it
// - JOINT attention over concat [txt; img] in double AND single blocks
// - txt FFN (ff_context) with raw c_gate_mlp
// - Flux2Modulation: SiLU(temb) BEFORE the modulation linears; split as
//   [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
// - 4-axis adjacent-pair RoPE (theta=2000, axes 32x4), raw (non-tanh) gates
// - single-block MLP = plain SiLU (no GLU gating)
// - norm_out = AdaLayerNormContinuous (SiLU(temb) -> linear -> chunk)
//
// NPU GEMMs, host norms/activations/softmax. Token layout unchanged:
// img tokens row-major (py*8+px), txt first in joint sequences.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/dit_npu_common.hpp"

namespace mbufo::core::npu::dit {

struct FluxDitConfig {
  std::size_t d = 3072;
  std::size_t heads = 24;
  std::size_t head_dim = 128;
  std::size_t n_double = 5;
  std::size_t n_single = 20;
  float eps = 1.0e-6F;
  float rope_theta = 2000.0F;
  std::size_t rope_axes[4] = {32, 32, 32, 32};
};

struct FluxModulations {
  std::vector<float> img;  // [18432]: msa(shift,scale,gate)+mlp(shift,scale,gate)
  std::vector<float> txt;  // [18432]
  std::vector<float> sgl;  // [9216]: (shift,scale,gate)
};

class FluxDitNpu {
 public:
  FluxDitNpu() = default;

  [[nodiscard]] bool Open(const std::string& transformer_dir,
                          NpuQuantGemmEngine* engine, NpuQuantMode mode,
                          const FluxDitConfig& config = {},
                          std::string* error = nullptr);

  [[nodiscard]] bool TimestepEmbed(float sigma, std::vector<float>* temb,
                                   double* npu_ms = nullptr,
                                   std::string* error = nullptr);
  [[nodiscard]] bool Modulations(std::span<const float> temb,
                                 FluxModulations* mod, double* npu_ms = nullptr,
                                 std::string* error = nullptr);
  [[nodiscard]] bool EncodeContext(std::span<const float> cond7680,
                                   std::size_t tokens,
                                   std::vector<float>* txt,
                                   double* npu_ms = nullptr,
                                   std::string* error = nullptr);
  [[nodiscard]] bool XEmbed(std::span<const float> z, std::size_t patches,
                            std::vector<float>* img, double* npu_ms = nullptr,
                            std::string* error = nullptr);

  /// 4-axis RoPE tables for joint [txt; img]: txt rows (0,0,0,l),
  /// img rows (0,h,w,0) in token order py*8+px.
  void BuildRopeTables(std::size_t St, std::size_t Si,
                       std::vector<float>* cos_t,
                       std::vector<float>* sin_t) const;

  // In-place over img [Si*d] and txt [St*d].
  [[nodiscard]] bool DenoiseBlocks(std::vector<float>* img,
                                   std::vector<float>* txt, std::size_t Si,
                                   std::size_t St,
                                   const FluxModulations& mod,
                                   const std::vector<float>& rope_cos,
                                   const std::vector<float>& rope_sin,
                                   std::size_t n_double, std::size_t n_single,
                                   double* npu_ms = nullptr,
                                   std::string* error = nullptr);
  [[nodiscard]] bool VelocityHead(std::span<const float> img, std::size_t Si,
                                  std::span<const float> temb,
                                  std::vector<float>* v,
                                  double* npu_ms = nullptr,
                                  std::string* error = nullptr);

  /// One Euler step in place over latents z [Si*128].
  [[nodiscard]] bool FullStep(std::vector<float>* z, std::size_t Si,
                              std::span<const float> txt, std::size_t St,
                              float sigma, float dt, std::size_t n_double,
                              std::size_t n_single, double* npu_ms = nullptr,
                              std::string* error = nullptr);

  /// Debug: run one double block in place (input verification harness).
  [[nodiscard]] bool DebugDoubleBlock(std::vector<float>* img,
                                      std::vector<float>* txt, std::size_t Si,
                                      std::size_t St, std::size_t bi,
                                      std::span<const float> mod_img,
                                      std::span<const float> mod_txt,
                                      std::span<const float> rope_cos,
                                      std::span<const float> rope_sin,
                                      double* npu_ms = nullptr,
                                      std::string* error = nullptr) {
    return DoubleBlock(img, txt, Si, St, bi, mod_img, mod_txt, rope_cos,
                       rope_sin, npu_ms, error);
  }

  /// Debug: run one single block on a joint sequence in place.
  [[nodiscard]] bool DebugSingleBlock(std::vector<float>* joint,
                                      std::size_t Sj, std::size_t bi,
                                      std::span<const float> mod_sgl,
                                      std::span<const float> rope_cos,
                                      std::span<const float> rope_sin,
                                      double* npu_ms = nullptr,
                                      std::string* error = nullptr) {
    return SingleBlock(joint, Sj, bi, mod_sgl, rope_cos, rope_sin, npu_ms,
                       error);
  }

  [[nodiscard]] const FluxDitConfig& config() const noexcept { return cfg_; }

 private:
  [[nodiscard]] bool DoubleBlock(std::vector<float>* img,
                                 std::vector<float>* txt, std::size_t Si,
                                 std::size_t St, std::size_t bi,
                                 std::span<const float> mod_img,
                                 std::span<const float> mod_txt,
                                 std::span<const float> rope_cos,
                                 std::span<const float> rope_sin,
                                 double* npu_ms, std::string* error);
  [[nodiscard]] bool SingleBlock(std::vector<float>* joint, std::size_t Sj,
                                 std::size_t bi,
                                 std::span<const float> mod_sgl,
                                 std::span<const float> rope_cos,
                                 std::span<const float> rope_sin,
                                 double* npu_ms, std::string* error);

  FluxDitConfig cfg_;
  NpuShardReader reader_;
  NpuQuantGemmEngine* engine_ = nullptr;
  NpuQuantMode mode_ = NpuQuantMode::kQ8Bfp16;
  DitNpuLinear lin_;
};

}  // namespace mbufo::core::npu::dit
