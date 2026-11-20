#include <hip/hip_runtime.h>

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
#include <utility>
#include <vector>

#include "src/models/minimax_h3/audio_vae.hpp"
#include "src/models/minimax_h3/audio_vae_ops.hip.hpp"
#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sha256.hpp"

#ifndef MBUFO_H3_SOURCE_MANIFEST
#define MBUFO_H3_SOURCE_MANIFEST ""
#endif

#ifndef MBUFO_H3_AUDIO_VAE_FIXTURE
#define MBUFO_H3_AUDIO_VAE_FIXTURE ""
#endif

namespace {

using mbufo::minimax_h3::AudioVaeDecoder;
using mbufo::minimax_h3::AudioVaeTelemetry;
using mbufo::minimax_h3::AudioWaveform;
using mbufo::minimax_h3::CancellationToken;
using mbufo::minimax_h3::ModelInventory;
using mbufo::minimax_h3::Sha256File;
namespace audio_ops = mbufo::minimax_h3::audio_vae_ops;
namespace json = mbufo::minimax_h3::json;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL minimax_h3_audio_vae_hip_test: " << message << '\n';
  std::exit(1);
}

void Hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    Fail(std::string(operation) + ": " + hipGetErrorString(status));
  }
}

class Buffer {
public:
  explicit Buffer(std::size_t bytes) : bytes_(bytes) {
    Hip(hipMalloc(&data_, bytes), "hipMalloc");
  }
  ~Buffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)) {}
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        (void)hipFree(data_);
      }
      data_ = std::exchange(other.data_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }

  template<typename T>
  T* As() {
    return static_cast<T*>(data_);
  }
  void* data() { return data_; }
  std::size_t bytes() const { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

Buffer Upload(std::span<const float> values) {
  Buffer result(values.size_bytes());
  Hip(hipMemcpy(result.data(), values.data(), values.size_bytes(),
                hipMemcpyHostToDevice),
      "upload");
  return result;
}

std::vector<float> Download(Buffer* buffer, std::size_t count) {
  std::vector<float> result(count);
  Hip(hipMemcpy(result.data(), buffer->data(), result.size() * sizeof(float),
                hipMemcpyDeviceToHost),
      "download");
  return result;
}

void Compare(std::span<const float> got, std::span<const float> expected,
             float tolerance, const char* label) {
  if (got.size() != expected.size()) {
    Fail(std::string(label) + " shape mismatch");
  }
  for (std::size_t index = 0; index < got.size(); ++index) {
    if (!std::isfinite(got[index]) ||
        std::abs(got[index] - expected[index]) > tolerance) {
      Fail(std::string(label) + " differs at " + std::to_string(index) +
           ": got " + std::to_string(got[index]) + ", expected " +
           std::to_string(expected[index]));
    }
  }
}

const json::Value& Required(const json::Value& object, std::string_view key) {
  const json::Value* value = object.Find(key);
  if (value == nullptr) {
    Fail("oracle contract is missing " + std::string(key));
  }
  return *value;
}

std::vector<float> HostConv(std::span<const float> input,
                            std::span<const float> weight,
                            std::span<const float> bias,
                            std::uint32_t input_length,
                            std::uint32_t input_channels,
                            std::uint32_t output_channels, std::uint32_t kernel,
                            std::uint32_t padding, std::uint32_t dilation) {
  std::vector<float> output(static_cast<std::size_t>(input_length) *
                            output_channels);
  for (std::uint32_t time = 0; time < input_length; ++time) {
    for (std::uint32_t out = 0; out < output_channels; ++out) {
      float value = bias[out];
      for (std::uint32_t in = 0; in < input_channels; ++in) {
        for (std::uint32_t tap = 0; tap < kernel; ++tap) {
          const std::int64_t source_time =
              static_cast<std::int64_t>(time) - padding + tap * dilation;
          if (source_time >= 0 &&
              source_time < static_cast<std::int64_t>(input_length)) {
            value = std::fma(
                input[static_cast<std::size_t>(source_time) * input_channels +
                      in],
                weight[(static_cast<std::size_t>(out) * input_channels + in) *
                           kernel +
                       tap],
                value);
          }
        }
      }
      output[static_cast<std::size_t>(time) * output_channels + out] = value;
    }
  }
  return output;
}

std::vector<float> HostTranspose(
    std::span<const float> input, std::span<const float> weight,
    std::span<const float> bias, std::uint32_t input_length,
    std::uint32_t input_channels, std::uint32_t output_channels,
    std::uint32_t kernel, std::uint32_t stride, std::uint32_t padding) {
  const std::uint32_t output_length =
      (input_length - 1U) * stride + kernel - 2U * padding;
  std::vector<float> output(static_cast<std::size_t>(output_length) *
                            output_channels);
  for (std::uint32_t time = 0; time < output_length; ++time) {
    for (std::uint32_t out = 0; out < output_channels; ++out) {
      float value = bias[out];
      for (std::uint32_t in = 0; in < input_channels; ++in) {
        for (std::uint32_t tap = 0; tap < kernel; ++tap) {
          const std::int64_t numerator =
              static_cast<std::int64_t>(time) + padding - tap;
          if (numerator < 0 || numerator % stride != 0) {
            continue;
          }
          const std::int64_t source_time = numerator / stride;
          if (source_time < static_cast<std::int64_t>(input_length)) {
            value = std::fma(
                input[static_cast<std::size_t>(source_time) * input_channels +
                      in],
                weight[(static_cast<std::size_t>(in) * output_channels + out) *
                           kernel +
                       tap],
                value);
          }
        }
      }
      output[static_cast<std::size_t>(time) * output_channels + out] = value;
    }
  }
  return output;
}

std::vector<float> HostSnake(std::span<const float> input,
                             std::span<const float> alpha,
                             std::span<const float> beta,
                             std::span<const float> up,
                             std::span<const float> down, std::uint32_t length,
                             std::uint32_t channels) {
  std::vector<float> output(input.size());
  for (std::uint32_t time = 0; time < length; ++time) {
    for (std::uint32_t channel = 0; channel < channels; ++channel) {
      float result = 0.0F;
      for (int down_tap = 0; down_tap < 12; ++down_tap) {
        int up_time = static_cast<int>(time * 2U) + down_tap - 5;
        up_time = std::clamp(up_time, 0, static_cast<int>(length * 2U) - 1);
        const int raw_time = up_time + 15;
        float upsampled = 0.0F;
        for (int up_tap = 0; up_tap < 12; ++up_tap) {
          const int numerator = raw_time - up_tap;
          if (numerator < 0 || (numerator & 1) != 0) {
            continue;
          }
          const int source_time =
              std::clamp(numerator / 2 - 5, 0, static_cast<int>(length) - 1);
          upsampled = std::fma(
              input[static_cast<std::size_t>(source_time) * channels + channel],
              2.0F * up[up_tap], upsampled);
        }
        const float sine = std::sin(std::exp(alpha[channel]) * upsampled);
        const float activated =
            upsampled + sine * sine / (std::exp(beta[channel]) + 1.0e-9F);
        result = std::fma(activated, down[down_tap], result);
      }
      output[static_cast<std::size_t>(time) * channels + channel] = result;
    }
  }
  return output;
}

int PrimitiveTest() {
  hipStream_t stream = nullptr;
  Hip(hipStreamCreate(&stream), "hipStreamCreate");
  const std::array<float, 8> input = {-0.8F, 0.1F,  -0.2F, 0.4F,
                                      0.5F,  -0.6F, 0.9F,  0.3F};
  const std::array<float, 12> vector = {0.1F,  -0.2F, 0.3F,  0.4F,
                                        -0.5F, 0.6F,  -0.7F, 0.8F,
                                        0.9F,  -1.0F, 1.1F,  -1.2F};
  const std::array<float, 2> magnitude = {0.8F, 1.2F};
  const std::array<float, 2> bias = {0.1F, -0.05F};
  const std::array<float, 2> alpha = {-0.2F, 0.3F};
  const std::array<float, 2> beta = {0.1F, -0.4F};
  const std::array<float, 12> filter = {0.01F, 0.02F, 0.04F, 0.08F,
                                        0.14F, 0.21F, 0.21F, 0.14F,
                                        0.08F, 0.04F, 0.02F, 0.01F};
  std::array<float, 12> normalized{};
  for (std::size_t row = 0; row < 2; ++row) {
    float square = 0.0F;
    for (std::size_t column = 0; column < 6; ++column) {
      square =
          std::fma(vector[row * 6 + column], vector[row * 6 + column], square);
    }
    const float scale = magnitude[row] / std::sqrt(square);
    for (std::size_t column = 0; column < 6; ++column) {
      normalized[row * 6 + column] = vector[row * 6 + column] * scale;
    }
  }
  Buffer input_gpu = Upload(input);
  Buffer vector_gpu = Upload(vector);
  Buffer magnitude_gpu = Upload(magnitude);
  Buffer bias_gpu = Upload(bias);
  Buffer alpha_gpu = Upload(alpha);
  Buffer beta_gpu = Upload(beta);
  Buffer filter_gpu = Upload(filter);
  Buffer weight_gpu(normalized.size() * sizeof(float));
  Buffer conv_gpu(input.size() * sizeof(float));
  Buffer snake_gpu(input.size() * sizeof(float));
  const std::array<float, 8> silence{};
  Buffer silence_gpu = Upload(silence);
  Buffer silence_output_gpu(silence.size() * sizeof(float));
  const std::array<float, 4> unclipped = {-2.0F, -0.5F, 0.5F, 2.0F};
  const std::array<float, 4> clipped = {-1.0F, -0.5F, 0.5F, 1.0F};
  Buffer unclipped_gpu = Upload(unclipped);
  Buffer clipped_gpu(clipped.size() * sizeof(float));
  audio_ops::LaunchWeightNorm(vector_gpu.As<float>(), magnitude_gpu.As<float>(),
                              weight_gpu.As<float>(), 2, 6, stream);
  audio_ops::LaunchConv1d(input_gpu.As<float>(), weight_gpu.As<float>(),
                          bias_gpu.As<float>(), conv_gpu.As<float>(), 1, 4, 2,
                          4, 2, 3, 1, 1, stream);
  audio_ops::LaunchAliasFreeSnake(input_gpu.As<float>(), alpha_gpu.As<float>(),
                                  beta_gpu.As<float>(), filter_gpu.As<float>(),
                                  filter_gpu.As<float>(), snake_gpu.As<float>(),
                                  1, 4, 2, stream);
  audio_ops::LaunchAliasFreeSnake(
      silence_gpu.As<float>(), alpha_gpu.As<float>(), beta_gpu.As<float>(),
      filter_gpu.As<float>(), filter_gpu.As<float>(),
      silence_output_gpu.As<float>(), 1, 4, 2, stream);
  audio_ops::LaunchClipAndReorder(unclipped_gpu.As<float>(),
                                  clipped_gpu.As<float>(), 2, stream);
  Hip(hipStreamSynchronize(stream), "primitive synchronize");
  Compare(Download(&weight_gpu, normalized.size()), normalized, 3.0e-6F,
          "weight norm");
  Compare(Download(&conv_gpu, input.size()),
          HostConv(input, normalized, bias, 4, 2, 2, 3, 1, 1), 3.0e-5F,
          "Conv1d");
  Compare(Download(&snake_gpu, input.size()),
          HostSnake(input, alpha, beta, filter, filter, 4, 2), 3.0e-5F,
          "SnakeBeta");
  Compare(Download(&silence_output_gpu, silence.size()), silence, 0.0F,
          "SnakeBeta silence");
  Compare(Download(&clipped_gpu, clipped.size()), clipped, 0.0F,
          "waveform clipping");

  const std::array<float, 6> transpose_input = {-0.5F, 0.25F, 0.75F,
                                                -0.1F, 0.2F,  0.8F};
  const std::array<float, 8> transpose_weight = {0.2F,  -0.3F, 0.4F, 0.1F,
                                                 -0.2F, 0.5F,  0.3F, -0.4F};
  const std::array<float, 1> transpose_bias = {0.05F};
  Buffer transpose_input_gpu = Upload(transpose_input);
  Buffer transpose_weight_gpu = Upload(transpose_weight);
  Buffer transpose_bias_gpu = Upload(transpose_bias);
  Buffer transpose_gpu(6 * sizeof(float));
  audio_ops::LaunchConvTranspose1d(
      transpose_input_gpu.As<float>(), transpose_weight_gpu.As<float>(),
      transpose_bias_gpu.As<float>(), transpose_gpu.As<float>(), 1, 3, 2, 6, 1,
      4, 2, 1, stream);
  Hip(hipStreamSynchronize(stream), "transpose synchronize");
  Compare(Download(&transpose_gpu, 6),
          HostTranspose(transpose_input, transpose_weight, transpose_bias, 3, 2,
                        1, 4, 2, 1),
          3.0e-5F, "ConvTranspose1d");
  Hip(hipStreamDestroy(stream), "hipStreamDestroy");
  std::cout << "ok: MiniMax H3 AudioVAE HIP primitives match host references\n";
  return 0;
}

std::vector<float> ReadF32(const std::filesystem::path& path,
                           std::size_t elements) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    Fail("cannot open oracle tensor " + path.string());
  }
  std::vector<float> values(elements);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    Fail("oracle tensor has wrong size: " + path.string());
  }
  return values;
}

int ReflectIndex(int index, int length) {
  while (index < 0 || index >= length) {
    index = index < 0 ? -index : 2 * length - 2 - index;
  }
  return index;
}

void CheckSpectrogram(std::span<const float> waveform,
                      std::span<const float> expected, int samples, int frames,
                      double relative_l2_bound, double relative_max_bound) {
  constexpr int kFft = 1024;
  constexpr int kHop = 256;
  constexpr int kBins = kFft / 2 + 1;
  constexpr double kTau = 6.28318530717958647692;
  const std::array<int, 8> frequencies = {0, 1, 7, 23, 64, 128, 256, 512};
  const std::array<int, 3> times = {0, frames / 2, frames - 1};
  if (waveform.size() != static_cast<std::size_t>(2 * samples) ||
      expected.size() != static_cast<std::size_t>(2 * kBins * frames)) {
    Fail("AudioVAE spectrogram geometry differs");
  }
  double error_square = 0.0;
  double value_square = 0.0;
  double maximum_error = 0.0;
  double maximum_value = 0.0;
  for (int channel = 0; channel < 2; ++channel) {
    for (const int frequency : frequencies) {
      for (const int time : times) {
        double real = 0.0;
        double imaginary = 0.0;
        for (int index = 0; index < kFft; ++index) {
          const int source =
              ReflectIndex(time * kHop + index - kFft / 2, samples);
          const double window = 0.5 - 0.5 * std::cos(kTau * index / kFft);
          const double angle =
              -kTau * frequency * static_cast<double>(index) / kFft;
          const double value =
              waveform[static_cast<std::size_t>(channel) * samples + source] *
              window;
          real += value * std::cos(angle);
          imaginary += value * std::sin(angle);
        }
        const double got = std::hypot(real, imaginary);
        const double want =
            expected[(static_cast<std::size_t>(channel) * kBins + frequency) *
                         frames +
                     time];
        const double delta = got - want;
        error_square += delta * delta;
        value_square += want * want;
        maximum_error = std::max(maximum_error, std::abs(delta));
        maximum_value = std::max(maximum_value, std::abs(want));
      }
    }
  }
  const double relative_l2 =
      std::sqrt(error_square / std::max(value_square, 1.0e-24));
  const double relative_max = maximum_error / std::max(maximum_value, 1.0e-12);
  std::cout << "AudioVAE selected spectrogram relative_l2=" << relative_l2
            << " relative_max=" << relative_max << '\n';
  if (relative_l2 >= relative_l2_bound || relative_max >= relative_max_bound) {
    Fail("AudioVAE spectrogram gate failed");
  }
}

int RealTest() {
  const char* model_root = std::getenv("MBUFO_H3_MODEL_ROOT");
  const char* golden_value = std::getenv("MBUFO_H3_AUDIO_VAE_GOLDEN");
  const std::filesystem::path fixture = MBUFO_H3_AUDIO_VAE_FIXTURE;
  if (model_root == nullptr || *model_root == '\0' || golden_value == nullptr ||
      *golden_value == '\0') {
    std::cout << "SKIP: set MBUFO_H3_MODEL_ROOT and "
                 "MBUFO_H3_AUDIO_VAE_GOLDEN\n";
    return 77;
  }
  const json::Value contract = json::ParseFile(fixture, 1U << 20U);
  if (Required(contract, "schema").AsString() !=
          "mbufo.minimax-h3-audio-vae-golden.v1" ||
      Required(contract, "model_revision").AsString() !=
          "42ed227ee7df40d41602854ae760620d6eb651fe" ||
      Required(contract, "reference_revision").AsString() !=
          "8974cc055ea9c02fcd14cc27dfda3e1027c05153") {
    Fail("AudioVAE oracle identity differs");
  }
  const std::filesystem::path golden(golden_value);
  const json::Value& files = Required(contract, "files");
  for (std::string_view name :
       {"audio_latent.f32", "waveform.f32", "spectrogram.f32"}) {
    const json::Value& entry = Required(files, name);
    if (std::filesystem::file_size(golden / name) !=
        Required(entry, "bytes").AsUint64()) {
      Fail("AudioVAE oracle size differs for " + std::string(name));
    }
    if (Sha256File(golden / name) != Required(entry, "sha256").AsString()) {
      Fail("AudioVAE oracle hash differs for " + std::string(name));
    }
  }
  const int latent_frames =
      static_cast<int>(Required(contract, "latent_frames").AsInt64());
  const int samples = static_cast<int>(Required(contract, "samples").AsInt64());
  const int spectrogram_frames =
      static_cast<int>(Required(contract, "spectrogram_frames").AsInt64());
  const std::vector<float> latent =
      ReadF32(golden / "audio_latent.f32",
              static_cast<std::size_t>(32 * 2 * latent_frames));
  const std::vector<float> expected =
      ReadF32(golden / "waveform.f32", static_cast<std::size_t>(2 * samples));
  const std::vector<float> expected_spectrogram =
      ReadF32(golden / "spectrogram.f32",
              static_cast<std::size_t>(2 * 513 * spectrogram_frames));
  std::string error;
  auto inventory =
      ModelInventory::Inspect(model_root, MBUFO_H3_SOURCE_MANIFEST, &error);
  if (!inventory.has_value()) {
    Fail(error);
  }
  CancellationToken pre_cancelled;
  pre_cancelled.Cancel();
  AudioVaeTelemetry cancelled_telemetry;
  if (AudioVaeDecoder::Create(*inventory, latent_frames, &pre_cancelled,
                              &cancelled_telemetry, &error) != nullptr ||
      error.find("cancel") == std::string::npos) {
    Fail("creation cancellation was not honored");
  }
  AudioVaeTelemetry telemetry;
  auto decoder = AudioVaeDecoder::Create(*inventory, latent_frames, nullptr,
                                         &telemetry, &error);
  if (decoder == nullptr) {
    Fail(error);
  }
  CancellationToken resident_cancelled;
  resident_cancelled.Cancel();
  AudioWaveform cancelled_waveform;
  if (decoder->Decode(latent, &resident_cancelled, nullptr, nullptr,
                      &cancelled_waveform, &telemetry, &error) ||
      error.find("cancel") == std::string::npos ||
      !cancelled_waveform.pcm.empty()) {
    Fail("resident decode cancellation was not honored");
  }
  error.clear();
  AudioWaveform waveform;
  if (!decoder->Decode(latent, nullptr, nullptr, nullptr, &waveform, &telemetry,
                       &error)) {
    Fail(error);
  }
  if (waveform.channels != 2 || waveform.samples != samples ||
      waveform.sample_rate != 32000 || waveform.pcm.size() != expected.size()) {
    Fail("AudioVAE returned wrong waveform geometry");
  }
  double maximum = 0.0;
  double square_error = 0.0;
  double square_value = 0.0;
  double left_energy = 0.0;
  double right_energy = 0.0;
  double separation_energy = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (!std::isfinite(waveform.pcm[index]) ||
        waveform.pcm[index] < -1.000001F || waveform.pcm[index] > 1.000001F) {
      Fail("AudioVAE returned invalid PCM");
    }
    const double delta =
        static_cast<double>(waveform.pcm[index]) - expected[index];
    maximum = std::max(maximum, std::abs(delta));
    square_error += delta * delta;
    square_value += static_cast<double>(expected[index]) * expected[index];
    if (index < static_cast<std::size_t>(samples)) {
      left_energy +=
          static_cast<double>(waveform.pcm[index]) * waveform.pcm[index];
    } else {
      right_energy +=
          static_cast<double>(waveform.pcm[index]) * waveform.pcm[index];
      const double separation =
          static_cast<double>(waveform.pcm[index]) -
          waveform.pcm[index - static_cast<std::size_t>(samples)];
      separation_energy += separation * separation;
    }
  }
  const double relative_l2 =
      std::sqrt(square_error / std::max(square_value, 1.0e-24));
  const json::Value& bounds = Required(contract, "quality_bounds");
  const double maximum_gate = Required(bounds, "maximum_error").AsDouble();
  const double l2_gate = Required(bounds, "relative_l2").AsDouble();
  const double spectrogram_l2_gate =
      Required(bounds, "spectrogram_relative_l2").AsDouble();
  const double spectrogram_max_gate =
      Required(bounds, "spectrogram_relative_max").AsDouble();
  if (maximum_gate <= 0.0 || maximum_gate > 1.0e-3 || l2_gate <= 0.0 ||
      l2_gate > 0.05 || spectrogram_l2_gate <= 0.0 ||
      spectrogram_l2_gate > 0.08 || spectrogram_max_gate <= 0.0 ||
      spectrogram_max_gate > 0.1) {
    Fail("AudioVAE oracle bounds are invalid");
  }
  if (maximum >= maximum_gate || relative_l2 >= l2_gate) {
    Fail("AudioVAE oracle gate failed: max=" + std::to_string(maximum) +
         ", relative_l2=" + std::to_string(relative_l2));
  }
  CheckSpectrogram(waveform.pcm, expected_spectrogram, samples,
                   spectrogram_frames, spectrogram_l2_gate,
                   spectrogram_max_gate);
  const AudioVaeTelemetry first_telemetry = telemetry;
  if (left_energy <= 1.0e-8 || right_energy <= 1.0e-8 ||
      separation_energy <= 1.0e-8 || telemetry.swap_bytes != 0 ||
      telemetry.dispatches < 300 || telemetry.scratch_address == 0) {
    Fail("AudioVAE quality or residency telemetry gate failed: left_energy=" +
         std::to_string(left_energy) +
         ", right_energy=" + std::to_string(right_energy) +
         ", separation_energy=" + std::to_string(separation_energy) +
         ", swap_bytes=" + std::to_string(telemetry.swap_bytes) +
         ", dispatches=" + std::to_string(telemetry.dispatches) +
         ", scratch_address=" +
         std::to_string(static_cast<std::uint64_t>(
             telemetry.scratch_address)));
  }
  std::cout << "AudioVAE waveform: max abs " << maximum << ", relative L2 "
            << relative_l2 << ", "
            << first_telemetry.decode_ms / 1000.0 << " s, "
            << static_cast<double>(first_telemetry.peak_live_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " GiB peak, cold major faults "
            << first_telemetry.major_page_faults << '\n';
  std::cout << "ok: MiniMax H3 AudioVAE matches frozen waveform oracle\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    Fail("expected --primitives or --real");
  }
  const std::string mode = argv[1];
  if (mode == "--primitives") {
    return PrimitiveTest();
  }
  if (mode == "--real") {
    return RealTest();
  }
  Fail("unknown mode " + mode);
}
