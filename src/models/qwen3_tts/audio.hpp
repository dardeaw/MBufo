#ifndef MBUFO_MODELS_QWEN3_TTS_AUDIO_HPP_
#define MBUFO_MODELS_QWEN3_TTS_AUDIO_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mbufo::models::qwen3_tts {

struct AudioBuffer {
  std::uint32_t sample_rate{0};
  std::uint32_t channels{0};
  // Interleaved float samples.
  std::vector<float> samples;
};

[[nodiscard]] bool DecodeBase64Wav(std::string_view encoded,
                                   AudioBuffer* output,
                                   std::string* error = nullptr);

[[nodiscard]] bool DecodeWav(std::span<const std::byte> bytes,
                             AudioBuffer* output, std::string* error = nullptr);

[[nodiscard]] std::vector<float> ResampleMono(const AudioBuffer& audio,
                                              std::uint32_t output_sample_rate);

}  // namespace mbufo::models::qwen3_tts

#endif  // MBUFO_MODELS_QWEN3_TTS_AUDIO_HPP_
