#include "src/models/minimax_h3/audio_vae.hpp"

#include <utility>

namespace mbufo::minimax_h3 {

#if !defined(ENGINE_ENABLE_HIP)

struct AudioVaeDecoder::Impl {};

AudioVaeDecoder::AudioVaeDecoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AudioVaeDecoder::~AudioVaeDecoder() = default;
AudioVaeDecoder::AudioVaeDecoder(AudioVaeDecoder&&) noexcept = default;
AudioVaeDecoder& AudioVaeDecoder::operator=(AudioVaeDecoder&&) noexcept =
    default;

std::unique_ptr<AudioVaeDecoder> AudioVaeDecoder::Create(
    const ModelInventory&, int, const CancellationToken*, AudioVaeTelemetry*,
    std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 AudioVAE requires a HIP-enabled build";
  }
  return nullptr;
}

bool AudioVaeDecoder::Decode(std::span<const float>, const CancellationToken*,
                             AudioVaeProgress, void*, AudioWaveform*,
                             AudioVaeTelemetry*, std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 AudioVAE requires a HIP-enabled build";
  }
  return false;
}

int AudioVaeDecoder::latent_frames() const noexcept {
  return 0;
}
int AudioVaeDecoder::samples() const noexcept {
  return 0;
}

#endif

}  // namespace mbufo::minimax_h3
