#include "src/models/qwen3_tts/audio.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <span>
#include <string>
#include <vector>

namespace {

namespace qwen3_tts = mbufo::models::qwen3_tts;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_audio_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

double RootMeanSquare(std::span<const float> samples) {
  double sum = 0.0;
  for (const float sample : samples) {
    sum += static_cast<double>(sample) * sample;
  }
  return std::sqrt(sum / static_cast<double>(samples.size()));
}

qwen3_tts::AudioBuffer MakeSine(std::uint32_t sample_rate, double frequency,
                                std::size_t frames) {
  qwen3_tts::AudioBuffer audio{
      .sample_rate = sample_rate,
      .channels = 1,
      .samples = {},
  };
  audio.samples.resize(frames);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    audio.samples[frame] =
        static_cast<float>(std::sin(2.0 * std::numbers::pi * frequency *
                                    static_cast<double>(frame) / sample_rate));
  }
  return audio;
}

void TestIdentityAndMonoMix() {
  const qwen3_tts::AudioBuffer audio{
      .sample_rate = 24000,
      .channels = 2,
      .samples = {1.0F, -1.0F, 0.5F, 0.25F},
  };
  const std::vector<float> mono = qwen3_tts::ResampleMono(audio, 24000);
  Check(mono == std::vector<float>({0.0F, 0.375F}),
        "same-rate stereo input is mixed without resampling drift");
}

void TestDownsampleAntiAlias() {
  constexpr std::uint32_t kInputRate = 48000;
  constexpr std::uint32_t kOutputRate = 24000;
  constexpr std::size_t kFrames = kInputRate;
  constexpr std::size_t kTrim = 64;

  const std::vector<float> passband = qwen3_tts::ResampleMono(
      MakeSine(kInputRate, 3000.0, kFrames), kOutputRate);
  const std::vector<float> stopband = qwen3_tts::ResampleMono(
      MakeSine(kInputRate, 18000.0, kFrames), kOutputRate);
  const std::span<const float> passband_middle(passband.data() + kTrim,
                                               passband.size() - 2 * kTrim);
  const std::span<const float> stopband_middle(stopband.data() + kTrim,
                                               stopband.size() - 2 * kTrim);
  const double passband_rms = RootMeanSquare(passband_middle);
  const double stopband_rms = RootMeanSquare(stopband_middle);

  Check(passband.size() == kOutputRate, "downsample frame count is exact");
  Check(passband_rms > 0.68 && passband_rms < 0.73,
        "speech-band energy is preserved");
  Check(stopband_rms < 0.01,
        "above-Nyquist energy is filtered instead of aliased");
}

}  // namespace

int main() {
  TestIdentityAndMonoMix();
  TestDownsampleAntiAlias();
  std::cout << "PASS qwen3_tts_audio_test\n";
  return 0;
}
