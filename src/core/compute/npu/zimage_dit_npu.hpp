#pragma once
// Z-Image-Turbo DiT NPU path, TRUE PORT of the diffusers pipeline math
// (P8b). v1 followed the parity shell (1D rope, RMSNorm final head with
// 1+scale, no cap padding) and diverged from the pipeline. This version:
// - complex-cis 3-axis RoPE (axes 32/48/48, theta=256) via adjacent-pair
//   tables (same rotation math as the complex path)
// - cap padded to 32 with learned cap_pad_token; real cap t=1..St,
//   pad rows [0,0,0]; img t=33 (cap_len+1)
// - joint [img(64); cap(32)] = 96 tokens, img rope (t=33,h,w), cap (t,h=0,w=0)
// - FinalLayer: LayerNorm (no affine, eps 1e-6!) + pure scale (no +1)
// - everything else unchanged (RMSNorms, mod chunk(4), tanh gates,
//   SwiGLU FFN, no-mod refiner branch, t_scale=1000)

#include <cstddef>
#include <cstdint>
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
  std::size_t cap_pad = 32;  // SEQ_MULTI_OF padding for cap tokens
  float eps = 1.0e-5F;
  float rope_theta = 256.0F;
  float final_eps = 1.0e-6F;
  // From the model config (NOT the RopeEmbedder defaults): axes (32,48,48).
  std::size_t rope_axes[3] = {32, 48, 48};
};

class ZImageDitNpu {
 public:
  ZImageDitNpu() = default;

  [[nodiscard]] bool Open(const std::string& transformer_dir,
                          NpuQuantGemmEngine* engine, NpuQuantMode mode,
                          const ZImageDitConfig& config = {},
                          std::string* error = nullptr);

  /// Caption conditioning: TE tap34 [S,2560] -> rms -> linear+bias.
  [[nodiscard]] bool CapEmbed(std::span<const float> te_tap34,
                             std::size_t tokens, std::vector<float>* cap,
                             double* npu_ms = nullptr,
                             std::string* error = nullptr);
  /// Pad cap to cap_pad rows with the learned cap_pad_token (pipeline
  /// _prepare_sequence parity: pad positions take pad_token, NOT last-row).
  static void PadCap(std::span<const float> cap_in, std::size_t tokens,
                     std::size_t hidden, std::size_t padded,
                     std::span<const float> pad_token,
                     std::vector<float>* cap_out);
  /// Context refiner blocks (no modulation), in place.
  [[nodiscard]] bool RefineContext(std::vector<float>* cap, std::size_t tokens,
                                   std::size_t n_ref,
                                   std::span<const float> rope_cos,
                                   std::span<const float> rope_sin,
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

  /// 3-axis rope tables for joint [img; cap]: img (t=33,h,w), real cap
  /// rows (t=1..St,h=0,w=0), PAD cap rows [0,0,0] (pipeline _pad_with_ids).
  void BuildRopeTables(std::size_t cap_tokens, std::size_t cap_real,
                       std::size_t Si, std::vector<float>* cos_t,
                       std::vector<float>* sin_t) const;

  // Generic block: norms an1/an2/fn1/fn2 + qk-norm rope attn + SwiGLU FFN.
  // mod is null for context refiner, else [15360] per-block modulation.
  [[nodiscard]] bool Block(const std::string& prefix, std::vector<float>* h,
                           std::size_t seq, const float* mod,
                           std::span<const float> rope_cos,
                           std::span<const float> rope_sin, double* npu_ms,
                           std::string* error);

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
