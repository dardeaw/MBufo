#ifndef MBUFO_MODELS_MINIMAX_H3_GENERATION_HPP_
#define MBUFO_MODELS_MINIMAX_H3_GENERATION_HPP_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/minimax_h3/audio_vae.hpp"
#include "src/models/minimax_h3/denoiser.hpp"
#include "src/models/minimax_h3/media.hpp"
#include "src/models/minimax_h3/prompt_encoder.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/video_vae.hpp"

namespace mbufo::minimax_h3 {

inline constexpr std::string_view kH3GenerationSchema =
    "mbufo.minimax-h3-text-generation.v1";

struct GenerationParameters {
  std::string preset;
  int internal_width{0};
  int internal_height{0};
  int output_width{0};
  int output_height{0};
  int frames{0};
  int evaluations{0};
  int active_blocks{0};
  int reuse_interval{0};
  bool row_parallel_attention{true};
  bool decode_audio{true};
  bool mux{true};
  std::vector<int> selected_frames;
};

[[nodiscard]] std::optional<GenerationParameters> ResolveGenerationPreset(
    std::string_view name, std::string* error = nullptr);
/// HTTP duration at 24 fps, snapped upward to the official VAE frame grid.
/// The aligned result must fit in 5..15 seconds. "1" selects the existing
/// 22-frame diagnostic extension.
[[nodiscard]] std::optional<int> ResolveDurationFrames(
    std::string_view seconds, std::string* error = nullptr);
[[nodiscard]] bool ValidateGenerationParameters(
    const GenerationParameters& parameters, std::string* error = nullptr);
[[nodiscard]] std::string GenerationParametersJson(
    const GenerationParameters& parameters, std::uint64_t seed,
    std::string_view prompt_sha256);

// Publishes a complete diagnostic PPM frame set by renaming one sibling
// staging directory. Existing output is preserved and never replaced.
[[nodiscard]] bool WriteDiagnosticFrames(const std::filesystem::path& directory,
                                         const VideoFrames& frames,
                                         const CancellationToken* cancellation,
                                         std::string* error = nullptr);
[[nodiscard]] bool WriteDiagnosticLatents(
    const std::filesystem::path& directory, const GenerationGeometry& geometry,
    std::span<const float> video, std::span<const float> audio,
    const CancellationToken* cancellation, std::string* error = nullptr);

struct GenerationRequest {
  std::filesystem::path model_root;
  std::filesystem::path source_manifest;
  std::filesystem::path output_path;
  std::filesystem::path frames_directory;
  std::filesystem::path latents_directory;
  std::string prompt;
  std::uint64_t seed{0};
  GenerationParameters parameters;
};

struct GenerationTelemetry {
  double inventory_ms{0.0};
  double tokenizer_ms{0.0};
  double prompt_ms{0.0};
  double denoiser_ms{0.0};
  double video_vae_ms{0.0};
  double first_preview_ms{0.0};
  double audio_vae_ms{0.0};
  double media_ms{0.0};
  double total_ms{0.0};
  std::size_t prompt_tokens{0};
  InspectionTelemetry inventory;
  PromptEncoderTelemetry prompt_encoder;
  DenoiserTelemetry denoiser;
  VideoVaeTelemetry video_vae;
  AudioVaeTelemetry audio_vae;
  MediaMuxTelemetry media;
};

using GenerationProgress = void (*)(std::string_view phase, int completed,
                                    int total, void* opaque);

[[nodiscard]] bool GenerateTextVideo(const GenerationRequest& request,
                                     const CancellationToken* cancellation,
                                     GenerationProgress progress,
                                     void* progress_opaque,
                                     GenerationTelemetry* telemetry,
                                     std::string* error = nullptr);

[[nodiscard]] std::string GenerationTelemetryJson(
    const GenerationTelemetry& telemetry);

}  // namespace mbufo::minimax_h3

#endif  // MBUFO_MODELS_MINIMAX_H3_GENERATION_HPP_
