#include "src/models/minimax_h3/prompt_encoder.hpp"

namespace mbufo::minimax_h3 {

#if !defined(ENGINE_ENABLE_HIP)
bool EncodePromptLayer50(const ModelInventory&, std::span<const std::uint32_t>,
                         const PromptEncoderOptions&, const CancellationToken*,
                         PromptEmbedding*, PromptEncoderTelemetry*,
                         std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 prompt encoder requires ENGINE_ENABLE_HIP";
  }
  return false;
}

std::uint64_t PromptEncoderLiveRegisteredHostBytes() noexcept {
  return 0;
}
#endif

}  // namespace mbufo::minimax_h3
