#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sha256.hpp"
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

std::vector<float> ReadF32(const std::filesystem::path& path,
                           std::size_t elements) {
  std::vector<float> values(elements);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(
          reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(float))) ||
      input.peek() != std::ifstream::traits_type::eof()) {
    return {};
  }
  return values;
}

const h3::json::Value& RequireField(const h3::json::Value& object,
                                    std::string_view name) {
  const h3::json::Value* value = object.Find(name);
  if (value == nullptr) {
    throw h3::json::Error("missing fixture field " + std::string(name));
  }
  return *value;
}

void CheckBoundary(std::string_view name, std::span<const float> got,
                   std::span<const float> expected, double relative_l2_bound,
                   double relative_max_bound) {
  CHECK(got.size() == expected.size());
  double error_square = 0.0;
  double value_square = 0.0;
  double maximum_error = 0.0;
  double maximum_value = 0.0;
  for (std::size_t index = 0; index < std::min(got.size(), expected.size());
       ++index) {
    const double delta = static_cast<double>(got[index]) - expected[index];
    error_square += delta * delta;
    value_square += static_cast<double>(expected[index]) * expected[index];
    maximum_error = std::max(maximum_error, std::abs(delta));
    maximum_value =
        std::max(maximum_value, std::abs(static_cast<double>(expected[index])));
  }
  const double relative_l2 =
      std::sqrt(error_square / std::max(value_square, 1.0e-24));
  const double relative_max = maximum_error / std::max(maximum_value, 1.0e-12);
  std::cout << name << " relative_l2=" << relative_l2
            << " relative_max=" << relative_max
            << " maximum_error=" << maximum_error
            << " maximum_value=" << maximum_value << '\n';
  CHECK(relative_l2 < relative_l2_bound);
  CHECK(relative_max < relative_max_bound);
}

struct Progress {
  int calls{0};
  int completed{0};
  int total{0};
};

void RecordProgress(int completed, int total, void* opaque) {
  auto* progress = static_cast<Progress*>(opaque);
  ++progress->calls;
  progress->completed = completed;
  progress->total = total;
}

}  // namespace

int main() {
  constexpr std::array<int, 5> kSelectedIndexes = {0, 5, 11, 16, 21};
  const char* root = std::getenv("MBUFO_H3_MODEL_ROOT");
  const char* golden_value = std::getenv("MBUFO_H3_VIDEO_VAE_GOLDEN");
  if (root == nullptr || golden_value == nullptr) {
    std::cout
        << "SKIP: MBUFO_H3_MODEL_ROOT or MBUFO_H3_VIDEO_VAE_GOLDEN unset\n";
    return 77;
  }
  constexpr std::size_t kLatentElements = 24 * 7 * 16 * 16;
  constexpr std::size_t kFrameElements = 256 * 256 * 3;
  constexpr std::size_t kSelectedElements = 5 * kFrameElements;
  const std::filesystem::path golden(golden_value);
  double relative_l2_bound = 0.0;
  double relative_max_bound = 0.0;
  try {
    const h3::json::Value fixture =
        h3::json::ParseFile(MBUFO_H3_VIDEO_VAE_FIXTURE, 1U << 20U);
    CHECK(RequireField(fixture, "schema").AsString() ==
          "mbufo.minimax-h3-video-vae-golden.v1");
    CHECK(RequireField(fixture, "model_revision").AsString() ==
          "42ed227ee7df40d41602854ae760620d6eb651fe");
    CHECK(RequireField(fixture, "reference_revision").AsString() ==
          "8974cc055ea9c02fcd14cc27dfda3e1027c05153");
    const h3::json::Value& selected =
        RequireField(fixture, "selected_frame_indices");
    CHECK(selected.AsArray().size() == kSelectedIndexes.size());
    for (std::size_t index = 0;
         index < std::min(selected.AsArray().size(), kSelectedIndexes.size());
         ++index) {
      CHECK(selected.AsArray()[index].AsInt64() == kSelectedIndexes[index]);
    }
    const h3::json::Value& bounds = RequireField(fixture, "quality_bounds");
    relative_l2_bound = RequireField(bounds, "relative_l2").AsDouble();
    relative_max_bound = RequireField(bounds, "relative_max").AsDouble();
    CHECK(relative_l2_bound > 0.0 && relative_l2_bound <= 0.05);
    CHECK(relative_max_bound > 0.0 && relative_max_bound <= 0.05);
    const h3::json::Value& files = RequireField(fixture, "files");
    for (std::string_view name : {"latent.f32", "selected_frames.f32"}) {
      const h3::json::Value& entry = RequireField(files, name);
      CHECK(std::filesystem::file_size(golden / name) ==
            RequireField(entry, "bytes").AsUint64());
      CHECK(h3::Sha256File(golden / name) ==
            RequireField(entry, "sha256").AsString());
    }
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
  auto latent = ReadF32(golden / "latent.f32", kLatentElements);
  auto expected_selected =
      ReadF32(golden / "selected_frames.f32", kSelectedElements);
  CHECK(!latent.empty());
  CHECK(!expected_selected.empty());
  if (failures != 0) {
    return 1;
  }

  std::string error;
  auto inventory =
      h3::ModelInventory::Inspect(root, MBUFO_H3_SOURCE_MANIFEST, &error);
  auto geometry = h3::ResolveGenerationGeometry(256, 256, 22, &error);
  CHECK(inventory.has_value());
  CHECK(geometry.has_value());
  if (!inventory || !geometry) {
    std::cerr << error << '\n';
    return 1;
  }
  h3::CancellationToken pre_cancelled;
  pre_cancelled.Cancel();
  h3::VideoVaeTelemetry rejected_telemetry;
  auto rejected =
      h3::VideoVaeDecoder::Create(*inventory, {.geometry = *geometry},
                                  &pre_cancelled, &rejected_telemetry, &error);
  CHECK(rejected == nullptr);
  CHECK(error.find("cancelled") != std::string::npos);
  error.clear();

  h3::VideoVaeTelemetry creation;
  auto decoder = h3::VideoVaeDecoder::Create(
      *inventory, {.geometry = *geometry}, nullptr, &creation, &error);
  CHECK(decoder != nullptr);
  if (decoder == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(decoder->output_frames() == 22);
  CHECK(decoder->tile_pixels() == 256);
  CHECK(creation.weight_bytes > 9ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(creation.peak_live_bytes < 120ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(creation.scratch_address != 0);
  CHECK(creation.swap_bytes == 0);

  h3::CancellationToken cancelled;
  cancelled.Cancel();
  h3::VideoFrames ignored;
  h3::VideoVaeTelemetry telemetry;
  CHECK(!decoder->DecodeSelected(latent, kSelectedIndexes, &cancelled, nullptr,
                                 nullptr, &ignored, &telemetry, &error));
  CHECK(error.find("cancelled") != std::string::npos);
  CHECK(ignored.rgb.empty());
  error.clear();

  Progress selected_progress;
  h3::VideoFrames selected;
  CHECK(decoder->DecodeSelected(latent, kSelectedIndexes, nullptr,
                                RecordProgress, &selected_progress, &selected,
                                &telemetry, &error));
  if (selected.rgb.empty()) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(selected.frame_indices ==
        std::vector<int>(kSelectedIndexes.begin(), kSelectedIndexes.end()));
  CHECK(selected.rgb.size() == kSelectedElements);
  CHECK(selected_progress.calls == 1);
  CHECK(selected_progress.completed == 1);
  CHECK(selected_progress.total == 1);
  CheckBoundary("VisualVAE selected RGB", selected.rgb, expected_selected,
                relative_l2_bound, relative_max_bound);
  CHECK(std::ranges::all_of(selected.rgb, [](float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
  }));
  CHECK(telemetry.tile_ms.size() == 1);
  CHECK(telemetry.dispatches == 36 * 9 + 10);
  CHECK(telemetry.first_non_finite.empty());
  CHECK(telemetry.swap_bytes == 0);
  std::cout << "H3 VisualVAE weight_gib="
            << static_cast<double>(telemetry.weight_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " peak_gib="
            << static_cast<double>(telemetry.peak_live_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " load_ms=" << telemetry.load_ms
            << " selected_tile_ms=" << telemetry.tile_ms.front() << '\n';
  return failures == 0 ? 0 : 1;
}
