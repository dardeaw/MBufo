#ifndef MBUFO_MODELS_MINIMAX_H3_VIDEO_VAE_HPP_
#define MBUFO_MODELS_MINIMAX_H3_VIDEO_VAE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sampling.hpp"

namespace mbufo::minimax_h3 {

inline constexpr int kH3VideoVaeLatentChannels = 24;
inline constexpr int kH3VideoVaeSpatialRatio = 16;
inline constexpr int kH3VideoVaeChunkLatents = 7;
inline constexpr int kH3VideoVaeChunkFrames = 22;
inline constexpr int kH3VideoVaeChunkStride = 5;
inline constexpr int kH3VideoVaeFrameStride = 17;

struct VideoVaeOptions {
  GenerationGeometry geometry;
};

struct VideoFrames {
  int height{0};
  int width{0};
  std::vector<int> frame_indices;
  // Frame-major, row-major, interleaved RGB F32 in [0, 1].
  std::vector<float> rgb;
};

struct VideoVaeTelemetry {
  double load_ms{0.0};
  std::vector<double> tile_ms;
  double decode_ms{0.0};
  std::uint64_t weight_bytes{0};
  std::uint64_t persistent_bytes{0};
  std::uint64_t scratch_bytes{0};
  std::uint64_t peak_live_bytes{0};
  std::uint64_t cumulative_allocation_bytes{0};
  std::uint64_t minor_page_faults{0};
  std::uint64_t major_page_faults{0};
  std::uint64_t swap_bytes{0};
  std::uint64_t dispatches{0};
  std::uintptr_t scratch_address{0};
  std::string first_non_finite;
};

using VideoVaeProgress = void (*)(int completed_tiles, int total_tiles,
                                  void* opaque);

struct VideoVaeTileAxis {
  int length{0};
  std::vector<int> starts;
  std::vector<int> overlaps;
};

struct VideoVaePlan {
  int output_frames{0};
  int chunks{0};
  int tile_pixels{0};
  VideoVaeTileAxis y_axis;
  VideoVaeTileAxis x_axis;
  std::uint32_t tile_latent_height{0};
  std::uint32_t tile_latent_width{0};
  std::uint32_t patches{0};
  std::uint32_t sequence{0};
};

struct VideoVaeFrameSource {
  int chunk{0};
  int local_frame{0};
  float weight{1.0F};
};

struct VideoVaeFrameRecipe {
  std::array<VideoVaeFrameSource, 2> sources{};
  int source_count{0};
};

[[nodiscard]] std::optional<VideoVaePlan> ResolveVideoVaePlan(
    const GenerationGeometry& geometry, std::string* error = nullptr);

[[nodiscard]] std::optional<VideoVaeFrameRecipe> ResolveVideoVaeFrameRecipe(
    const VideoVaePlan& plan, int frame, std::string* error = nullptr);

[[nodiscard]] bool StitchVideoVaeTiles(
    const VideoVaeTileAxis& y_axis, const VideoVaeTileAxis& x_axis,
    std::size_t frame_count, std::span<const std::vector<float>> tiles,
    std::vector<float>* output, std::string* error = nullptr);

class VideoVaeDecoder {
public:
  ~VideoVaeDecoder();

  VideoVaeDecoder(const VideoVaeDecoder&) = delete;
  VideoVaeDecoder& operator=(const VideoVaeDecoder&) = delete;
  VideoVaeDecoder(VideoVaeDecoder&&) noexcept;
  VideoVaeDecoder& operator=(VideoVaeDecoder&&) noexcept;

  [[nodiscard]] static std::unique_ptr<VideoVaeDecoder> Create(
      const ModelInventory& inventory, const VideoVaeOptions& options,
      const CancellationToken* cancellation, VideoVaeTelemetry* telemetry,
      std::string* error = nullptr);

  [[nodiscard]] bool Decode(std::span<const float> normalized_latent,
                            const CancellationToken* cancellation,
                            VideoVaeProgress progress, void* progress_opaque,
                            VideoFrames* output, VideoVaeTelemetry* telemetry,
                            std::string* error = nullptr);

  // Selected indexes are global output-frame indexes and must be strictly
  // increasing. The model still sees each complete legal temporal chunk;
  // only unused chunks and RGB reconstruction/readback are omitted.
  [[nodiscard]] bool DecodeSelected(std::span<const float> normalized_latent,
                                    std::span<const int> frame_indices,
                                    const CancellationToken* cancellation,
                                    VideoVaeProgress progress,
                                    void* progress_opaque, VideoFrames* output,
                                    VideoVaeTelemetry* telemetry,
                                    std::string* error = nullptr);

  [[nodiscard]] const GenerationGeometry& geometry() const noexcept;
  [[nodiscard]] int output_frames() const noexcept;
  [[nodiscard]] int tile_pixels() const noexcept;

private:
  struct Impl;
  explicit VideoVaeDecoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace mbufo::minimax_h3

#endif  // MBUFO_MODELS_MINIMAX_H3_VIDEO_VAE_HPP_
