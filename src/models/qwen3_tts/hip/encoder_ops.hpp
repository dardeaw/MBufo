#ifndef MBUFO_MODELS_QWEN3_TTS_HIP_ENCODER_OPS_HPP_
#define MBUFO_MODELS_QWEN3_TTS_HIP_ENCODER_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace mbufo::models::qwen3_tts::hip {

enum class EncoderPaddingMode : std::uint32_t {
  kZero,
  kReplicate,
  kReflect,
};

void LaunchEncoderConv1dIm2Col(
    const float* input, float* columns, std::size_t input_length,
    std::size_t input_channels, std::size_t output_length, std::size_t kernel,
    std::size_t stride, std::size_t dilation, std::size_t left_padding,
    EncoderPaddingMode padding_mode, hipStream_t stream);

void LaunchElu(float* values, std::size_t elements, hipStream_t stream);
void LaunchRelu(float* values, std::size_t elements, hipStream_t stream);
void LaunchTanh(float* values, std::size_t elements, hipStream_t stream);
void LaunchRoundBfloat16(float* values, std::size_t elements,
                         hipStream_t stream);

void LaunchExtractChannels(const float* input, float* output, std::size_t rows,
                           std::size_t input_channels,
                           std::size_t channel_offset,
                           std::size_t output_channels, hipStream_t stream);

void LaunchInsertChannels(const float* input, float* output, std::size_t rows,
                          std::size_t output_channels,
                          std::size_t channel_offset,
                          std::size_t input_channels, hipStream_t stream);

void LaunchChannelMean(const float* input, float* output, std::size_t rows,
                       std::size_t channels, hipStream_t stream);

void LaunchSigmoidMultiplyChannels(float* values, const float* scale,
                                   std::size_t rows, std::size_t channels,
                                   hipStream_t stream);

void LaunchConcatenateThree(const float* first, const float* second,
                            const float* third, float* output, std::size_t rows,
                            std::size_t channels, hipStream_t stream);

void LaunchAttentiveStatsInput(const float* input, float* output,
                               std::size_t rows, std::size_t channels,
                               float epsilon, hipStream_t stream);

void LaunchSoftmaxOverRows(float* values, std::size_t rows,
                           std::size_t channels, hipStream_t stream);

void LaunchWeightedStats(const float* input, const float* weights,
                         float* output, std::size_t rows, std::size_t channels,
                         float epsilon, hipStream_t stream);

void LaunchQuantizeCodebook(float* residual, const float* embedding,
                            std::uint32_t* codes, std::size_t frames,
                            std::size_t dimension, std::size_t codebook_size,
                            std::size_t code_groups, std::size_t group_index,
                            bool update_residual, hipStream_t stream);

}  // namespace mbufo::models::qwen3_tts::hip
#endif

#endif  // MBUFO_MODELS_QWEN3_TTS_HIP_ENCODER_OPS_HPP_
