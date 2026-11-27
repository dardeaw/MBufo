#pragma once
// FLUX.2-klein DiT NPU path (P4): 5 double-stream + 20 single-stream blocks,
// 4-step distilled Euler flow. Exact port of RunFlux2KleinQuantE2E math with
// NPU-only GEMMs (d=3072, 24 heads x 128, rope_theta=2000, eps=1e-6).
//
// Per-step flow: TimestepEmbed(sigma) -> modulations -> XEmbed(z) ->
// DoubleBlocks(img+txt) -> SingleBlocks(img) -> VelocityHead -> z -= dt*v.
// Text conditioning (context_embedder over the 3 TE taps) is computed once
// per prompt via EncodeContext. VAE decode stays on the existing path.

#include <cstddef>
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
};

struct FluxModulations {
  std::vector<float> img;  // [18432]
  std::vector<float> txt;  // [18432]
  std::vector<float> sgl;  // [9216]
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

  // In-place over img [Si*d] and txt [St*d].
  [[nodiscard]] bool DenoiseBlocks(std::vector<float>* img,
                                   std::vector<float>* txt, std::size_t Si,
                                   std::size_t St,
                                   const FluxModulations& mod,
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

  [[nodiscard]] const FluxDitConfig& config() const noexcept { return cfg_; }

 private:
  [[nodiscard]] bool DoubleBlock(std::vector<float>* img,
                                 std::vector<float>* txt, std::size_t Si,
                                 std::size_t St, std::size_t bi,
                                 std::span<const float> mod_img,
                                 std::span<const float> mod_txt,
                                 double* npu_ms, std::string* error);
  [[nodiscard]] bool SingleBlock(std::vector<float>* img, std::size_t Si,
                                 std::size_t bi,
                                 std::span<const float> mod_sgl,
                                 double* npu_ms, std::string* error);

  FluxDitConfig cfg_;
  NpuShardReader reader_;
  NpuQuantGemmEngine* engine_ = nullptr;
  NpuQuantMode mode_ = NpuQuantMode::kQ8Bfp16;
  DitNpuLinear lin_;
};

}  // namespace mbufo::core::npu::dit
