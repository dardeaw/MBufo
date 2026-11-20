#ifndef MBUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_HPP_
#define MBUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"

namespace mbufo::minimax_h3 {

inline constexpr std::size_t kPromptEncoderLayers = 50;
inline constexpr std::size_t kPromptEncoderHiddenSize = 5120;
inline constexpr std::size_t kPromptEncoderIntermediateSize = 25600;
inline constexpr std::size_t kPromptEncoderQueryHeads = 64;
inline constexpr std::size_t kPromptEncoderKeyValueHeads = 8;
inline constexpr std::size_t kPromptEncoderHeadDimension = 128;
inline constexpr std::size_t kPromptEncoderVocabularySize = 151936;

struct PromptEncoderOptions {
  std::size_t layer_count{kPromptEncoderLayers};
  bool prefetch_layers{true};
  bool vision_inputs{false};
  bool deepstack_inputs{false};
};

struct PromptEncoderLayerTelemetry {
  std::size_t layer{0};
  std::uint64_t weight_bytes{0};
  double load_ms{0.0};
  double prefetch_wait_ms{0.0};
  double submit_ms{0.0};
  double gpu_ms{0.0};
  std::uint64_t dispatches{0};
};

struct PromptEncoderTelemetry {
  std::vector<PromptEncoderLayerTelemetry> layers;
  std::uint64_t embedding_weight_bytes{0};
  std::uint64_t activation_bytes{0};
  std::uint64_t peak_device_bytes{0};
  std::uint64_t peak_registered_host_bytes{0};
  std::uint64_t live_registered_host_bytes_after{0};
  std::uint64_t total_dispatches{0};
};

struct PromptEmbedding {
  std::size_t tokens{0};
  std::size_t width{0};
  std::vector<std::uint16_t> values;
};

[[nodiscard]] bool EncodePromptLayer50(const ModelInventory& inventory,
                                       std::span<const std::uint32_t> token_ids,
                                       const PromptEncoderOptions& options,
                                       const CancellationToken* cancellation,
                                       PromptEmbedding* output,
                                       PromptEncoderTelemetry* telemetry,
                                       std::string* error = nullptr);

[[nodiscard]] std::uint64_t PromptEncoderLiveRegisteredHostBytes() noexcept;

}  // namespace mbufo::minimax_h3

#endif  // MBUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_HPP_
