#ifndef MBUFO_MODELS_MINIMAX_H3_AUDIO_VAE_HPP_
#define MBUFO_MODELS_MINIMAX_H3_AUDIO_VAE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"

namespace mbufo::minimax_h3 {

inline constexpr int kH3AudioVaeLatentChannels = 32;
inline constexpr int kH3AudioVaeStereoChannels = 2;
inline constexpr int kH3AudioVaeSampleRate = 32000;
inline constexpr int kH3AudioVaeHopLength = 800;

struct AudioWaveform {
  int channels{0};
  int samples{0};
  int sample_rate{0};
  // Channel-major F32 PCM: [channels, samples].
  std::vector<float> pcm;
};

struct AudioVaeTelemetry {
  double load_ms{0.0};
  std::vector<double> stage_ms;
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

using AudioVaeProgress = void (*)(int completed_stages, int total_stages,
                                  void* opaque);

class AudioVaeDecoder {
public:
  ~AudioVaeDecoder();

  AudioVaeDecoder(const AudioVaeDecoder&) = delete;
  AudioVaeDecoder& operator=(const AudioVaeDecoder&) = delete;
  AudioVaeDecoder(AudioVaeDecoder&&) noexcept;
  AudioVaeDecoder& operator=(AudioVaeDecoder&&) noexcept;

  [[nodiscard]] static std::unique_ptr<AudioVaeDecoder> Create(
      const ModelInventory& inventory, int latent_frames,
      const CancellationToken* cancellation, AudioVaeTelemetry* telemetry,
      std::string* error = nullptr);

  [[nodiscard]] bool Decode(std::span<const float> normalized_latent,
                            const CancellationToken* cancellation,
                            AudioVaeProgress progress, void* progress_opaque,
                            AudioWaveform* output, AudioVaeTelemetry* telemetry,
                            std::string* error = nullptr);

  [[nodiscard]] int latent_frames() const noexcept;
  [[nodiscard]] int samples() const noexcept;

private:
  struct Impl;
  explicit AudioVaeDecoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace mbufo::minimax_h3

#endif  // MBUFO_MODELS_MINIMAX_H3_AUDIO_VAE_HPP_
