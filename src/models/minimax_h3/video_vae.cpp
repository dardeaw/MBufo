#include "src/models/minimax_h3/video_vae.hpp"

#include <limits>
#include <utility>

namespace mbufo::minimax_h3 {
namespace {

constexpr int kMinimumTilePixels = 256;
constexpr int kTileOverlapMinimum = 64;
constexpr int kSuffixRows = 5;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

int TileCountForExtent(int extent, int tile_pixels) {
  if (extent <= tile_pixels) {
    return 1;
  }
  int count = (extent + tile_pixels - 1) / tile_pixels;
  while (tile_pixels * count - kTileOverlapMinimum * (count - 1) < extent) {
    ++count;
  }
  return count;
}

bool BuildTileAxis(int extent, int tile_pixels, VideoVaeTileAxis* axis,
                   std::string* error) {
  *axis = {};
  if (extent < 1 || extent % kH3VideoVaeSpatialRatio != 0) {
    SetError(error, "MiniMax H3 VisualVAE extent must be a multiple of 16");
    return false;
  }
  if (extent <= tile_pixels) {
    axis->length = extent;
    axis->starts = {0};
    return true;
  }
  const int count = TileCountForExtent(extent, tile_pixels);
  axis->length = tile_pixels;
  axis->starts.assign(static_cast<std::size_t>(count), 0);
  axis->overlaps.assign(static_cast<std::size_t>(count - 1),
                        kTileOverlapMinimum);
  const int remaining =
      tile_pixels * count - kTileOverlapMinimum * (count - 1) - extent;
  for (int unit = 0; unit < remaining / kH3VideoVaeSpatialRatio; ++unit) {
    axis->overlaps[static_cast<std::size_t>(unit % (count - 1))] +=
        kH3VideoVaeSpatialRatio;
  }
  for (int index = 1; index < count; ++index) {
    axis->starts[static_cast<std::size_t>(index)] =
        axis->starts[static_cast<std::size_t>(index - 1)] + tile_pixels -
        axis->overlaps[static_cast<std::size_t>(index - 1)];
  }
  return true;
}

}  // namespace

std::optional<VideoVaePlan> ResolveVideoVaePlan(
    const GenerationGeometry& geometry, std::string* error) {
  const int latent_time = geometry.temporal.video_latent_frames;
  if (latent_time < kH3VideoVaeChunkLatents) {
    SetError(
        error,
        "MiniMax H3 VisualVAE decoding requires at least 22 aligned frames");
    return std::nullopt;
  }
  if (geometry.width < 1 || geometry.height < 1 || geometry.latent_width < 1 ||
      geometry.latent_height < 1 ||
      geometry.width != geometry.latent_width * kH3VideoVaeSpatialRatio ||
      geometry.height != geometry.latent_height * kH3VideoVaeSpatialRatio ||
      (latent_time - 2) % kH3VideoVaeChunkStride != 0) {
    SetError(error, "invalid MiniMax H3 VisualVAE geometry");
    return std::nullopt;
  }
  VideoVaePlan plan;
  plan.chunks = (latent_time - 2) / kH3VideoVaeChunkStride;
  plan.output_frames = plan.chunks * kH3VideoVaeFrameStride + 5;
  if (plan.output_frames != geometry.temporal.frames) {
    SetError(error,
             "MiniMax H3 VisualVAE temporal geometry differs from output");
    return std::nullopt;
  }
  // The released Diffusers VisualVAE enables spatial tiling by default with a
  // fixed 256-pixel tile. Tile geometry changes the decoder's attention
  // context and therefore its pixels, so it is part of the quality contract
  // rather than an allocator heuristic.
  plan.tile_pixels = kMinimumTilePixels;
  if (!BuildTileAxis(geometry.height, plan.tile_pixels, &plan.y_axis, error) ||
      !BuildTileAxis(geometry.width, plan.tile_pixels, &plan.x_axis, error)) {
    return std::nullopt;
  }
  plan.tile_latent_height =
      static_cast<std::uint32_t>(plan.y_axis.length / kH3VideoVaeSpatialRatio);
  plan.tile_latent_width =
      static_cast<std::uint32_t>(plan.x_axis.length / kH3VideoVaeSpatialRatio);
  const std::uint64_t patches =
      static_cast<std::uint64_t>(kH3VideoVaeChunkLatents) *
      plan.tile_latent_height * plan.tile_latent_width;
  if (patches > std::numeric_limits<std::uint32_t>::max() - kSuffixRows) {
    SetError(error, "MiniMax H3 VisualVAE tile sequence is too large");
    return std::nullopt;
  }
  plan.patches = static_cast<std::uint32_t>(patches);
  plan.sequence = plan.patches + kSuffixRows;
  return plan;
}

std::optional<VideoVaeFrameRecipe> ResolveVideoVaeFrameRecipe(
    const VideoVaePlan& plan, int frame, std::string* error) {
  if (plan.chunks < 1 ||
      plan.output_frames != plan.chunks * kH3VideoVaeFrameStride + 5 ||
      frame < 0 || frame >= plan.output_frames) {
    SetError(error, "invalid MiniMax H3 VisualVAE output frame");
    return std::nullopt;
  }
  VideoVaeFrameRecipe recipe;
  if (frame < plan.chunks * kH3VideoVaeFrameStride) {
    const int chunk = frame / kH3VideoVaeFrameStride;
    const int local = frame % kH3VideoVaeFrameStride;
    if (chunk > 0 && local < 5) {
      const float alpha = static_cast<float>(local) / 5.0F;
      recipe.sources[0] = {chunk - 1, kH3VideoVaeFrameStride + local,
                           1.0F - alpha};
      recipe.source_count = 1;
      if (alpha > 0.0F) {
        recipe.sources[1] = {chunk, local, alpha};
        recipe.source_count = 2;
      }
    } else {
      recipe.sources[0] = {chunk, local, 1.0F};
      recipe.source_count = 1;
    }
  } else {
    recipe.sources[0] = {
        plan.chunks - 1,
        kH3VideoVaeFrameStride + frame - plan.chunks * kH3VideoVaeFrameStride,
        1.0F};
    recipe.source_count = 1;
  }
  return recipe;
}

bool StitchVideoVaeTiles(const VideoVaeTileAxis& y_axis,
                         const VideoVaeTileAxis& x_axis,
                         std::size_t frame_count,
                         std::span<const std::vector<float>> tiles,
                         std::vector<float>* output, std::string* error) {
  if (output != nullptr) {
    output->clear();
  }
  const auto valid_axis = [](const VideoVaeTileAxis& axis) {
    if (axis.length < 1 || axis.starts.empty() || axis.starts.front() != 0 ||
        axis.overlaps.size() + 1 != axis.starts.size()) {
      return false;
    }
    for (std::size_t index = 0; index < axis.overlaps.size(); ++index) {
      if (axis.overlaps[index] < 1 || axis.overlaps[index] >= axis.length ||
          axis.starts[index + 1] !=
              axis.starts[index] + axis.length - axis.overlaps[index]) {
        return false;
      }
    }
    return true;
  };
  if (frame_count == 0 || output == nullptr || !valid_axis(y_axis) ||
      !valid_axis(x_axis) ||
      tiles.size() != y_axis.starts.size() * x_axis.starts.size()) {
    SetError(error, "invalid MiniMax H3 VisualVAE tile stitch request");
    return false;
  }

  const int full_height = y_axis.starts.back() + y_axis.length;
  const int full_width = x_axis.starts.back() + x_axis.length;
  const std::size_t output_elements =
      frame_count * static_cast<std::size_t>(full_height) * full_width * 3U;
  output->assign(output_elements, 0.0F);
  const std::size_t expected_tile = frame_count *
                                    static_cast<std::size_t>(y_axis.length) *
                                    x_axis.length * 3U;
  const int x_count = static_cast<int>(x_axis.starts.size());
  for (int tile_y = 0; tile_y < static_cast<int>(y_axis.starts.size());
       ++tile_y) {
    for (int tile_x = 0; tile_x < x_count; ++tile_x) {
      const int index = tile_y * x_count + tile_x;
      if (tiles[static_cast<std::size_t>(index)].size() != expected_tile) {
        output->clear();
        SetError(error, "MiniMax H3 VisualVAE tile payload differs");
        return false;
      }
      const float* current = tiles[static_cast<std::size_t>(index)].data();
      const float* above =
          tile_y == 0 ? nullptr
                      : tiles[static_cast<std::size_t>(index - x_count)].data();
      const float* left =
          tile_x == 0 ? nullptr
                      : tiles[static_cast<std::size_t>(index - 1)].data();
      const int overlap_y =
          tile_y == 0 ? 0
                      : y_axis.overlaps[static_cast<std::size_t>(tile_y - 1)];
      const int overlap_x =
          tile_x == 0 ? 0
                      : x_axis.overlaps[static_cast<std::size_t>(tile_x - 1)];
      const int keep_height =
          y_axis.length -
          (tile_y + 1 < static_cast<int>(y_axis.starts.size())
               ? y_axis.overlaps[static_cast<std::size_t>(tile_y)]
               : 0);
      const int keep_width =
          x_axis.length -
          (tile_x + 1 < static_cast<int>(x_axis.starts.size())
               ? x_axis.overlaps[static_cast<std::size_t>(tile_x)]
               : 0);
      for (std::size_t frame = 0; frame < frame_count; ++frame) {
        for (int y = 0; y < keep_height; ++y) {
          for (int x = 0; x < keep_width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
              const std::size_t local =
                  ((frame * static_cast<std::size_t>(y_axis.length) + y) *
                       x_axis.length +
                   x) *
                      3U +
                  channel;
              float result = current[local];
              if (above != nullptr && y < overlap_y) {
                const std::size_t top =
                    ((frame * static_cast<std::size_t>(y_axis.length) +
                      (y_axis.length - overlap_y + y)) *
                         x_axis.length +
                     x) *
                        3U +
                    channel;
                const float alpha =
                    static_cast<float>(y) / static_cast<float>(overlap_y);
                result = above[top] * (1.0F - alpha) + result * alpha;
              }
              if (left != nullptr && x < overlap_x) {
                const std::size_t prior =
                    ((frame * static_cast<std::size_t>(y_axis.length) + y) *
                         x_axis.length +
                     (x_axis.length - overlap_x + x)) *
                        3U +
                    channel;
                const float alpha =
                    static_cast<float>(x) / static_cast<float>(overlap_x);
                result = left[prior] * (1.0F - alpha) + result * alpha;
              }
              const std::size_t destination =
                  ((frame * static_cast<std::size_t>(full_height) +
                    (y_axis.starts[static_cast<std::size_t>(tile_y)] + y)) *
                       full_width +
                   (x_axis.starts[static_cast<std::size_t>(tile_x)] + x)) *
                      3U +
                  channel;
              (*output)[destination] = result;
            }
          }
        }
      }
    }
  }
  return true;
}

#if !defined(ENGINE_ENABLE_HIP)

struct VideoVaeDecoder::Impl {};

VideoVaeDecoder::VideoVaeDecoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
VideoVaeDecoder::~VideoVaeDecoder() = default;
VideoVaeDecoder::VideoVaeDecoder(VideoVaeDecoder&&) noexcept = default;
VideoVaeDecoder& VideoVaeDecoder::operator=(VideoVaeDecoder&&) noexcept =
    default;

std::unique_ptr<VideoVaeDecoder> VideoVaeDecoder::Create(
    const ModelInventory&, const VideoVaeOptions&, const CancellationToken*,
    VideoVaeTelemetry*, std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 VisualVAE requires a HIP-enabled build";
  }
  return nullptr;
}

bool VideoVaeDecoder::Decode(std::span<const float>, const CancellationToken*,
                             VideoVaeProgress, void*, VideoFrames*,
                             VideoVaeTelemetry*, std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 VisualVAE requires a HIP-enabled build";
  }
  return false;
}

bool VideoVaeDecoder::DecodeSelected(std::span<const float>,
                                     std::span<const int>,
                                     const CancellationToken*, VideoVaeProgress,
                                     void*, VideoFrames*, VideoVaeTelemetry*,
                                     std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 VisualVAE requires a HIP-enabled build";
  }
  return false;
}

const GenerationGeometry& VideoVaeDecoder::geometry() const noexcept {
  static const GenerationGeometry empty;
  return empty;
}

int VideoVaeDecoder::output_frames() const noexcept {
  return 0;
}
int VideoVaeDecoder::tile_pixels() const noexcept {
  return 0;
}

#endif

}  // namespace mbufo::minimax_h3
