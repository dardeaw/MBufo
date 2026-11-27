#pragma once
// TTS NPU router seam (mirrors bge_m3 DecideNpuEmbed idiom).
// Entry point (future): SynthesisHipRuntime::Generate checks DecideNpuTts()
// first; kNpuIron means run talker+predictor+decoder on NPU backend.

#include <cstddef>
#include <cstdint>

#include "src/core/compute/npu/npu_lane.hpp"

namespace mbufo::models::qwen3_tts::npu {

enum class TtsNpuRoute : std::uint8_t { kNone = 0, kNpuIron };

[[nodiscard]] inline TtsNpuRoute DecideNpuTts(
    const mbufo::npu::NpuLaneConfig& config, std::size_t text_tokens,
    bool weights_resident = false, bool device_ready = false) noexcept {
  if (config.tts && weights_resident && device_ready && text_tokens > 0 &&
      text_tokens <= 2048u) {
    return TtsNpuRoute::kNpuIron;
  }
  return TtsNpuRoute::kNone;
}

}  // namespace mbufo::models::qwen3_tts::npu
