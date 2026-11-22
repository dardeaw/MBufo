#pragma once
// Z-Image-Turbo DiT NPU path (P5): caption embedder + 2 context-refiner
// blocks + 2 noise-refiner + 30 single-stream main layers, 8-step Euler
// flow. Exact port of RunZImageTurboQuantE2E math with NPU-only GEMMs
// (d=3840, 30 heads x 128, rope_theta=256, eps=1e-5; FFN=10240).
//
// Per-step flow: TEmbed(sigma) -> XEmbed(z) -> NoiseRefiner(img, mod) ->
// concat(img, cap) -> MainLayers(all, per-block mod) -> FinalHead ->
// z -= dt*v. Caption conditioning (CapEmbed over the TE final sequence +
// context refiner) is computed once per prompt. VAE decode stays on the
// existing path.

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/dit_npu_common.hpp"

namespace mbufo::core::npu::dit {

struct ZImageDitConfig {
  std::size_t d = 3840;
  std::size_t heads = 30;
  std::size_t head_dim = 128;
  std::size_t ffn = 10240;
  std::size_t n_refiner = 2;
  std::size_t n_main = 30;
  float eps = 1.0e-5F;
  float rope_theta = 256.0F;
};

class ZImageDitNpu {
 public:
  ZImageDitNpu() = default;

  [[nodiscard]] bool Open(const std::string& transformer_dir,
                          NpuQuantGemmEngine* engine, NpuQuantMode mode,
                          const ZImageDitConfig& config = {},
                          std::string* error = nullptr);

  /// Caption conditioning: TE final raw [S,2560] -> rms -> linear+bias.
  [[nodiscard]] bool CapEmbed(std::span<const float> te_final,
                             std::size_t tokens, std::vector<float>* cap,
                             double* npu_ms = nullptr,
                             std::string* error = nullptr);
  /// Context refiner blocks (no modulation), in place.
  [[nodiscard]] bool RefineContext(std::vector<float>* cap, std::size_t tokens,
                                   std::size_t n_ref,
                                   double* npu_ms = nullptr,
                                   std::string* error = nullptr);
  /// Timestep MLP: sinusoid -> mlp.0+bias -> silu -> mlp.2+bias = temb[256].
  [[nodiscard]] bool TEmbed(float sigma, std::vector<float>* temb,
                            double* npu_ms = nullptr,
                            std::string* error = nullptr);
  /// Image patch embed: z [Si,64] -> linear+bias -> img [Si,3840].
  [[nodiscard]] bool XEmbed(std::span<const float> z, std::size_t patches,
                            std::vector<float>* img, double* npu_ms = nullptr,
                            std::string* error = nullptr);

  // Generic block: norms an1/an2/fn1/fn2 + qk-norm rope attn + SwiGLU FFN.
  // mod is null for context refiner, else [15360] per-block modulation.
  [[nodiscard]] bool Block(const std::string& prefix, std::vector<float>* h,
                           std::size_t seq, const float* mod,
                           double* npu_ms, std::string* error);

  /// One Euler step in place over latents z [Si*64].
  [[nodiscard]] bool FullStep(std::vector<float>* z, std::size_t Si,
                              std::span<const float> cap, std::size_t St,
                              float sigma, float dt, std::size_t n_ref,
                              std::size_t n_main, double* npu_ms = nullptr,
                              std::string* error = nullptr);

  [[nodiscard]] const ZImageDitConfig& config() const noexcept { return cfg_; }

 private:
  ZImageDitConfig cfg_;
  NpuShardReader reader_;
  NpuQuantGemmEngine* engine_ = nullptr;
  NpuQuantMode mode_ = NpuQuantMode::kQ8Bfp16;
  DitNpuLinear lin_;
};

}  // namespace mbufo::core::npu::dit
