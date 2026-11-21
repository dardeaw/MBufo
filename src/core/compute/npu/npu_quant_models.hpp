#pragma once
// Unified NPU2 Q8 (bfp16ebs8) & Q4 (q4nx) Runtimes for:
//   1. Qwen3-TTS 12Hz 1.7B (Talker + CodePredictor + SpeechTokenizer 12Hz + Vocoder)
//   2. Qwen3-ASR 1.7B (AudioEncoder + Thinker Text Decoder)
//   3. FLUX.2-klein-4B (Qwen3-4B Text Encoder + Double/Single-Stream DiT + 4-Step Flow + Image Decode)
//   4. Z-Image-Turbo (Qwen3-4B Text Encoder + Context/Noise Refiner + Single-Stream DiT + 8-Step Flow + Image Decode)
//
// All four models share `NpuQuantGemmEngine` (4 XCLBINs total on XDNA2 / AIE2P)
// and stream weights via `SafetensorsIndex::ReadChunk` to keep peak RAM < 2 GiB.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_quant_gemm.hpp"

namespace mbufo::core::npu {

struct NpuTtsVerifyMetrics {
  NpuQuantMode mode{NpuQuantMode::kQ8Bfp16};
  double npu_ms{0.0};
  std::size_t quant_weight_bytes{0};
  float talker_cos_vs_fp32{0.0F};
  float predictor_cos_vs_fp32{0.0F};
  float decoder_cos_vs_fp32{0.0F};
  float audio_rms{0.0F};
  std::size_t audio_samples_24khz{0};
  std::string wav_path;
};

struct NpuAsrVerifyMetrics {
  NpuQuantMode mode{NpuQuantMode::kQ8Bfp16};
  double npu_ms{0.0};
  std::size_t quant_weight_bytes{0};
  float audio_encoder_cos_vs_fp32{0.0F};
  float thinker_cos_vs_fp32{0.0F};
  float logits_cos_vs_fp32{0.0F};
  bool top1_token_match{false};
  std::uint32_t top1_token_id{0};
};

struct NpuT2iVerifyMetrics {
  std::string model_name;
  NpuQuantMode mode{NpuQuantMode::kQ8Bfp16};
  double npu_ms{0.0};
  std::size_t quant_weight_bytes{0};
  int steps{0};
  int width{0};
  int height{0};
  float text_encoder_cos_vs_fp32{0.0F};
  float dit_step1_cos_vs_fp32{0.0F};
  float final_latent_cos_vs_fp32{0.0F};
  std::size_t png_bytes{0};
  std::string png_path;
};

/// Runs Qwen3-TTS 12Hz 1.7B end-to-end on NPU2 in `mode` (Q8 or Q4):
///   - Talker text projection (`linear_fc1`, `linear_fc2`) + Talker layers + `codec_head`
///   - CodePredictor layers (`H=1024, F=3072`)
///   - SpeechTokenizer 12Hz (`input_proj`, `pre_transformer` layers, `output_proj`,
///     ConvNeXt `pwconv1`/`pwconv2`, RVQ codebook & SnakeBeta waveform synthesis)
///   - Saves 24 kHz mono WAV to `out_wav_path` and computes cosine parity vs FP32 reference.
[[nodiscard]] bool RunQwen3TtsQuantE2E(
    NpuQuantGemmEngine& engine, const std::string& model_dir, NpuQuantMode mode,
    std::size_t talker_layers_to_run, std::size_t predictor_layers_to_run,
    std::size_t decoder_layers_to_run, const std::string& out_wav_path,
    NpuTtsVerifyMetrics* out_metrics, std::string* error);

/// Runs Qwen3-ASR 1.7B end-to-end on NPU2 in `mode` (Q8 or Q4):
///   - AudioTower `conv_out [1024, 7680]` + AudioEncoder layers (`d=1024, F=4096`)
///     + `ln_post` + `proj1 [1024, 1024]` + GELU + `proj2 [2048, 1024]`
///   - Thinker Text Decoder layers (`H=2048, F=6144, Q=2048, KV=1024`) + final RMSNorm
///     + LM head logits
///   - Computes cosine parity and top-1 token agreement vs FP32 reference.
[[nodiscard]] bool RunQwen3AsrQuantE2E(
    NpuQuantGemmEngine& engine, const std::string& model_dir, NpuQuantMode mode,
    std::size_t audio_layers_to_run, std::size_t thinker_layers_to_run,
    NpuAsrVerifyMetrics* out_metrics, std::string* error);

/// Runs FLUX.2-klein-4B end-to-end on NPU2 in `mode` (Q8 or Q4):
///   - Qwen3-4B Text Encoder layers (`H=2560, F=9728, Q=4096, KV=1024`) -> 7680-dim conditioning
///   - FLUX.2 DiT: `context_embedder [3072, 7680]`, `time_guidance_embed`,
///     `double_stream_modulation_{img,txt}`, `single_stream_modulation`,
///     Double-Stream DiT blocks (`d=3072, MLP=9216`), Single-Stream DiT blocks
///     (`to_qkv_mlp_proj [27648, 3072]`, `to_out [3072, 12288]`), `norm_out.linear`
///     across 4 distilled Euler flow-matching steps
///   - VAE mid-block self-attention (`to_{q,k,v,out.0} [512, 512]`) + spatial decode to RGB PNG.
[[nodiscard]] bool RunFlux2KleinQuantE2E(
    NpuQuantGemmEngine& engine, const std::string& model_dir, NpuQuantMode mode,
    std::size_t te_layers_to_run, std::size_t double_blocks_to_run,
    std::size_t single_blocks_to_run, const std::string& out_png_path,
    NpuT2iVerifyMetrics* out_metrics, std::string* error);

/// Runs Z-Image-Turbo end-to-end on NPU2 in `mode` (Q8 or Q4):
///   - Qwen3-4B Text Encoder layers (`H=2560, F=9728, Q=4096, KV=1024`)
///   - Z-Image-Turbo DiT: `cap_embedder [3840, 2560]`, `t_embedder`,
///     `context_refiner` blocks (`dim=3840, FFN=10240`), `noise_refiner` blocks,
///     and Single-Stream DiT `layers` (`adaLN_modulation [15360, 256]`,
///     `to_{q,k,v,out.0} [3840, 3840]`, `w1/w3 [10240, 3840]`, `w2 [3840, 10240]`)
///     across 8 Euler flow-matching steps + `all_final_layer` + spatial decode to RGB PNG.
[[nodiscard]] bool RunZImageTurboQuantE2E(
    NpuQuantGemmEngine& engine, const std::string& model_dir, NpuQuantMode mode,
    std::size_t te_layers_to_run, std::size_t refiner_layers_to_run,
    std::size_t main_layers_to_run, const std::string& out_png_path,
    NpuT2iVerifyMetrics* out_metrics, std::string* error);

}  // namespace mbufo::core::npu
