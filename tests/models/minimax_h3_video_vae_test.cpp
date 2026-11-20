#include <algorithm>
#include <cstddef>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/sampling.hpp"
#include "src/models/minimax_h3/video_vae.hpp"

namespace {

namespace h3 = mbufo::minimax_h3;

int failures = 0;

#define CHECK(condition)                                          \
  do {                                                            \
    if (!(condition)) {                                           \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": " \
                << #condition << '\n';                            \
      ++failures;                                                 \
    }                                                             \
  } while (false)

void CheckAxis(const h3::VideoVaeTileAxis& axis, int extent) {
  CHECK(axis.length > 0);
  CHECK(!axis.starts.empty());
  CHECK(axis.starts.front() == 0);
  CHECK(axis.starts.back() + axis.length == extent);
  CHECK(axis.overlaps.size() + 1 == axis.starts.size());
  for (std::size_t index = 0; index < axis.overlaps.size(); ++index) {
    CHECK(axis.overlaps[index] >= 64);
    CHECK(axis.overlaps[index] % h3::kH3VideoVaeSpatialRatio == 0);
    CHECK(axis.starts[index + 1] ==
          axis.starts[index] + axis.length - axis.overlaps[index]);
  }
}

void CheckPlan(int width, int height, int frames, int tile_pixels,
               std::size_t x_tiles, std::size_t y_tiles, int output_frames) {
  std::string error;
  auto geometry = h3::ResolveGenerationGeometry(width, height, frames, &error);
  CHECK(geometry.has_value());
  if (!geometry) {
    std::cerr << error << '\n';
    return;
  }
  auto plan = h3::ResolveVideoVaePlan(*geometry, &error);
  CHECK(plan.has_value());
  if (!plan) {
    std::cerr << error << '\n';
    return;
  }
  CHECK(plan->tile_pixels == tile_pixels);
  CHECK(plan->x_axis.starts.size() == x_tiles);
  CHECK(plan->y_axis.starts.size() == y_tiles);
  CHECK(plan->output_frames == output_frames);
  CHECK(plan->patches ==
        static_cast<std::uint32_t>(h3::kH3VideoVaeChunkLatents *
                                   plan->tile_latent_height *
                                   plan->tile_latent_width));
  CHECK(plan->sequence == plan->patches + 5);
  CheckAxis(plan->x_axis, width);
  CheckAxis(plan->y_axis, height);
}

void CheckSource(const h3::VideoVaeFrameSource& source, int chunk,
                 int local_frame, float weight) {
  CHECK(source.chunk == chunk);
  CHECK(source.local_frame == local_frame);
  CHECK(source.weight == weight);
}

float Pixel(std::span<const float> rgb, int width, int height, int frame, int y,
            int x) {
  return rgb[((static_cast<std::size_t>(frame) * height + y) * width + x) * 3U];
}

}  // namespace

int main() {
  CheckPlan(256, 256, 22, 256, 1, 1, 22);
  CheckPlan(288, 256, 22, 256, 2, 1, 22);
  CheckPlan(512, 512, 22, 256, 3, 3, 22);
  CheckPlan(1344, 768, 22, 256, 7, 4, 22);
  CheckPlan(256, 256, 39, 256, 1, 1, 39);

  std::string error;
  auto geometry = h3::ResolveGenerationGeometry(256, 256, 22, &error);
  CHECK(geometry.has_value());
  if (geometry) {
    geometry->temporal.video_latent_frames = 8;
    CHECK(!h3::ResolveVideoVaePlan(*geometry, &error).has_value());
  }

  geometry = h3::ResolveGenerationGeometry(256, 256, 39, &error);
  CHECK(geometry.has_value());
  if (geometry) {
    auto plan = h3::ResolveVideoVaePlan(*geometry, &error);
    CHECK(plan.has_value());
    if (plan) {
      auto first = h3::ResolveVideoVaeFrameRecipe(*plan, 0, &error);
      auto before_overlap = h3::ResolveVideoVaeFrameRecipe(*plan, 16, &error);
      auto overlap_start = h3::ResolveVideoVaeFrameRecipe(*plan, 17, &error);
      auto overlap_middle = h3::ResolveVideoVaeFrameRecipe(*plan, 19, &error);
      auto after_overlap = h3::ResolveVideoVaeFrameRecipe(*plan, 22, &error);
      auto final = h3::ResolveVideoVaeFrameRecipe(*plan, 38, &error);
      CHECK(first.has_value());
      CHECK(before_overlap.has_value());
      CHECK(overlap_start.has_value());
      CHECK(overlap_middle.has_value());
      CHECK(after_overlap.has_value());
      CHECK(final.has_value());
      if (first && before_overlap && overlap_start && overlap_middle &&
          after_overlap && final) {
        CHECK(first->source_count == 1);
        CheckSource(first->sources[0], 0, 0, 1.0F);
        CHECK(before_overlap->source_count == 1);
        CheckSource(before_overlap->sources[0], 0, 16, 1.0F);
        CHECK(overlap_start->source_count == 1);
        CheckSource(overlap_start->sources[0], 0, 17, 1.0F);
        CHECK(overlap_middle->source_count == 2);
        CheckSource(overlap_middle->sources[0], 0, 19, 0.6F);
        CheckSource(overlap_middle->sources[1], 1, 2, 0.4F);
        CHECK(after_overlap->source_count == 1);
        CheckSource(after_overlap->sources[0], 1, 5, 1.0F);
        CHECK(final->source_count == 1);
        CheckSource(final->sources[0], 1, 21, 1.0F);
      }
      CHECK(!h3::ResolveVideoVaeFrameRecipe(*plan, -1, &error).has_value());
      CHECK(!h3::ResolveVideoVaeFrameRecipe(*plan, 39, &error).has_value());
    }
  }

  geometry = h3::ResolveGenerationGeometry(512, 512, 22, &error);
  CHECK(geometry.has_value());
  if (geometry) {
    auto plan = h3::ResolveVideoVaePlan(*geometry, &error);
    CHECK(plan.has_value());
    if (plan) {
      const std::size_t tile_elements =
          static_cast<std::size_t>(plan->y_axis.length) * plan->x_axis.length *
          3U;
      std::vector<std::vector<float>> tiles;
      for (int value = 1; value <= 9; ++value) {
        tiles.emplace_back(2U * tile_elements, static_cast<float>(value));
        std::fill(
            tiles.back().begin() + static_cast<std::ptrdiff_t>(tile_elements),
            tiles.back().end(), static_cast<float>(value + 10));
      }
      std::vector<float> rgb;
      CHECK(h3::StitchVideoVaeTiles(plan->y_axis, plan->x_axis, 2, tiles, &rgb,
                                    &error));
      CHECK(rgb.size() == 2U * 512U * 512U * 3U);
      if (!rgb.empty()) {
        CHECK(Pixel(rgb, 512, 512, 0, 100, 100) == 1.0F);
        CHECK(Pixel(rgb, 512, 512, 0, 100, 192) == 1.5F);
        CHECK(Pixel(rgb, 512, 512, 0, 100, 320) == 2.5F);
        CHECK(Pixel(rgb, 512, 512, 0, 192, 100) == 2.5F);
        CHECK(Pixel(rgb, 512, 512, 0, 192, 192) == 3.75F);
        CHECK(Pixel(rgb, 512, 512, 0, 256, 256) == 8.0F);
        CHECK(Pixel(rgb, 512, 512, 0, 320, 100) == 5.5F);
        CHECK(Pixel(rgb, 512, 512, 0, 511, 511) == 9.0F);
        CHECK(Pixel(rgb, 512, 512, 1, 100, 192) == 11.5F);
        CHECK(Pixel(rgb, 512, 512, 1, 192, 192) == 13.75F);
        CHECK(Pixel(rgb, 512, 512, 1, 511, 511) == 19.0F);
      }
      tiles.back().pop_back();
      CHECK(!h3::StitchVideoVaeTiles(plan->y_axis, plan->x_axis, 2, tiles, &rgb,
                                     &error));
      CHECK(rgb.empty());
    }
  }

  if (failures == 0) {
    std::cout << "ok: MiniMax H3 VisualVAE tile/chunk plans\n";
  }
  return failures == 0 ? 0 : 1;
}
