#ifndef MBUFO_MODELS_MINIMAX_H3_SAMPLING_HPP_
#define MBUFO_MODELS_MINIMAX_H3_SAMPLING_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mbufo::minimax_h3 {

inline constexpr int kH3CanvasMultiple = 32;
inline constexpr std::int64_t kH3MaximumPixels = 768LL * 1344LL;
inline constexpr int kH3FramesPerSecond = 24;
inline constexpr int kH3AudioLatentFramesPerSecond = 40;
inline constexpr int kH3VaeSpatialRatio = 16;
inline constexpr int kH3MaximumFrames = 345;
inline constexpr int kH3MaximumEvaluations = 1000;

struct TemporalShape {
  int frames{0};
  int video_latent_frames{0};
  int audio_latent_frames{0};
};

struct GenerationGeometry {
  int width{0};
  int height{0};
  int latent_width{0};
  int latent_height{0};
  TemporalShape temporal;
};

[[nodiscard]] int AlignFrameCount(int requested_frames) noexcept;
[[nodiscard]] int VideoLatentFrameCount(int frame_count) noexcept;
[[nodiscard]] int VideoEncoderLatentFrameCount(int frame_count) noexcept;
[[nodiscard]] std::optional<GenerationGeometry> ResolveGenerationGeometry(
    int width, int height, int requested_frames, std::string* error = nullptr);

struct SigmaSchedule {
  std::vector<float> video;
  std::vector<float> audio;

  [[nodiscard]] int evaluations() const noexcept {
    return video.empty() ? 0 : static_cast<int>(video.size()) - 1;
  }
};

[[nodiscard]] std::optional<SigmaSchedule> BuildServingSchedule(
    int evaluations, std::string* error = nullptr);

struct TimeRowSchedule {
  std::vector<std::uint32_t> video_rows;
  std::vector<std::uint32_t> audio_rows;
  std::uint32_t time_rows{0};
};

[[nodiscard]] std::optional<TimeRowSchedule> BuildTimeRowSchedule(
    const SigmaSchedule& schedule, std::string* error = nullptr);

struct Position3D {
  double temporal{0.0};
  double height{0.0};
  double width{0.0};
};

enum class SegmentKind : std::uint8_t {
  kText,
  kAudioTarget,
  kVideoTarget,
};

[[nodiscard]] std::string_view ToString(SegmentKind kind) noexcept;

struct LayoutSegment {
  std::size_t begin{0};
  std::size_t end{0};
  SegmentKind kind{SegmentKind::kText};
};

struct PackedLayout {
  std::vector<Position3D> positions;
  std::vector<LayoutSegment> segments;
  std::size_t text_rows{0};
  std::size_t audio_target_rows{0};
  std::size_t video_target_rows{0};
  int video_latent_frames{0};
  int latent_height{0};
  int latent_width{0};
  int audio_latent_frames{0};

  [[nodiscard]] std::size_t rows() const noexcept { return positions.size(); }
};

[[nodiscard]] std::optional<PackedLayout> BuildTextOnlyLayout(
    std::size_t text_rows, const GenerationGeometry& geometry,
    std::string* error = nullptr);
[[nodiscard]] std::optional<std::vector<std::uint32_t>> BuildModulationRowMap(
    const TimeRowSchedule& time_rows, int step, const PackedLayout& layout,
    std::span<const std::uint8_t> text_tags = {}, std::string* error = nullptr);
[[nodiscard]] std::string PackedLayoutSha256(const PackedLayout& layout);
[[nodiscard]] std::string ModulationRowMapSha256(
    std::span<const std::uint32_t> rows);

[[nodiscard]] bool PatchifyVideo(std::span<const float> latent, int channels,
                                 int time, int height, int width,
                                 std::span<float> rows,
                                 std::string* error = nullptr);
[[nodiscard]] bool UnpatchifyVideo(std::span<const float> rows, int channels,
                                   int time, int height, int width,
                                   std::span<float> latent,
                                   std::string* error = nullptr);
[[nodiscard]] bool PackAudio(std::span<const float> latent, int channels,
                             int time, std::span<float> rows,
                             std::string* error = nullptr);
[[nodiscard]] bool UnpackAudio(std::span<const float> rows, int channels,
                               int time, std::span<float> latent,
                               std::string* error = nullptr);

class NormalRng {
public:
  explicit NormalRng(std::uint64_t seed) noexcept;

  [[nodiscard]] std::uint32_t NextU32() noexcept;
  [[nodiscard]] float NextNormal() noexcept;
  void Fill(std::span<float> output) noexcept;

private:
  std::uint64_t state_{0};
  std::uint64_t increment_{0};
  float spare_{0.0F};
  bool has_spare_{false};
};

struct InitialNoise {
  std::vector<float> video;
  std::vector<float> audio;
};

[[nodiscard]] std::optional<InitialNoise> BuildInitialNoise(
    std::uint64_t seed, std::size_t video_elements, int audio_channels,
    int audio_time, std::string* error = nullptr);

struct EulerStepPlan {
  bool evaluate{false};
  int last_evaluated{-1};
  int previous_evaluated{-1};
  float video_sigma_from_timestep{0.0F};
  float audio_sigma_from_timestep{0.0F};
  float video_ratio{0.0F};
  float audio_ratio{0.0F};
  float video_extrapolation{0.0F};
  float audio_extrapolation{0.0F};
};

[[nodiscard]] float VelocityExtrapolationRatio(float current_sigma,
                                               float last_sigma,
                                               float previous_sigma,
                                               bool have_previous) noexcept;
[[nodiscard]] std::optional<std::vector<std::uint8_t>> BuildReuseSelection(
    int steps, int reuse_interval, std::string* error = nullptr);
[[nodiscard]] std::optional<std::vector<EulerStepPlan>> BuildEulerPlan(
    const SigmaSchedule& schedule, int reuse_interval,
    std::string* error = nullptr);

// sample_f32, last_f32, and previous_f32 must point to device memory.
// stream is a hipStream_t passed opaquely to keep the public host header free
// from ROCm headers. The update remains in device F32 between DiT evaluations.
[[nodiscard]] bool HipEulerUpdate(
    void* sample_f32, std::size_t sample_elements, std::size_t sample_offset,
    const void* last_f32, const void* previous_f32,
    std::size_t velocity_elements, float sigma_from_timestep, float ratio,
    float extrapolation, void* stream, std::string* error = nullptr);

}  // namespace mbufo::minimax_h3

#endif  // MBUFO_MODELS_MINIMAX_H3_SAMPLING_HPP_
