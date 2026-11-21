#pragma once
// bge-m3 embedding NPU seam (EngineSession::Sync checks DecideNpuEmbed first).
//
// Until NPU-resident bge-m3 xclbins and weight shards are active on the device,
// DecideNpuEmbed returns kNone, preserving the existing HIP/CPU path untouched.

#include <cstddef>
#include <cstdint>

#include "src/core/compute/npu/npu_lane.hpp"

namespace mbufo::models::bge_m3::npu {

enum class EmbedNpuRoute : std::uint8_t { kNone = 0, kIron };

/// Route decision checked at the entry of EngineSession::Sync.
[[nodiscard]] inline EmbedNpuRoute DecideNpuEmbed(
    const mbufo::npu::NpuLaneConfig& config, std::size_t seq_len,
    bool weights_resident = false, bool device_ready = false) noexcept {
  if (config.embedding && weights_resident && device_ready && seq_len > 0 &&
      (seq_len % 4u == 0u)) {
    return EmbedNpuRoute::kIron;
  }
  return EmbedNpuRoute::kNone;
}

}  // namespace mbufo::models::bge_m3::npu
