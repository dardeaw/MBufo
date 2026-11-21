#pragma once
// bge-reranker NPU seam: routes Score() through DecideNpuScore + NpuScheduler
// when `config.reranker && weights_resident && device_ready`. Default kNone.

#include <cstddef>
#include <cstdint>

#include "src/core/compute/npu/npu_lane.hpp"

namespace mbufo::models::bge_reranker::npu {

enum class ScoreNpuRoute : std::uint8_t { kNone = 0, kIron };

[[nodiscard]] inline ScoreNpuRoute DecideNpuScore(
    const mbufo::npu::NpuLaneConfig& config, std::size_t seq_len,
    bool weights_resident = false, bool device_ready = false) noexcept {
  if (config.reranker && weights_resident && device_ready && seq_len > 0 &&
      seq_len <= 2048u) {
    return ScoreNpuRoute::kIron;
  }
  return ScoreNpuRoute::kNone;
}

}  // namespace mbufo::models::bge_reranker::npu
