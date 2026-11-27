#pragma once
// ASR NPU router seam (mirrors bge_m3 DecideNpuEmbed idiom).
// Entry point (future): TranscriptionHipRuntime::Transcribe checks
// DecideNpuAsr() first; kNpuIron means run audio+thinker on NPU backend.

#include <cstddef>
#include <cstdint>

#include "src/core/compute/npu/npu_lane.hpp"

namespace mbufo::models::qwen3_asr::npu {

enum class AsrNpuRoute : std::uint8_t { kNone = 0, kNpuIron };

[[nodiscard]] inline AsrNpuRoute DecideNpuAsr(
    const mbufo::npu::NpuLaneConfig& config, std::size_t audio_tokens,
    bool weights_resident = false, bool device_ready = false) noexcept {
  if (config.asr && weights_resident && device_ready && audio_tokens > 0 &&
      audio_tokens <= 2048u) {
    return AsrNpuRoute::kNpuIron;
  }
  return AsrNpuRoute::kNone;
}

}  // namespace mbufo::models::qwen3_asr::npu
