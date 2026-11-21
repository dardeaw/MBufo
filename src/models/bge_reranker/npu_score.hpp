#pragma once
// bge-reranker NPU 接縫（代碼框 v0，只宣告、不接線）。
//
// 複用 embedding 的 NPU encoder（同 XLM-R 血統），差別只在最後是 CLS 打分
// 而不是 pooling。encoder 通之前此樁永遠回 kNone。

#include <cstddef>

#include "src/core/compute/npu/npu_lane.hpp"

namespace mbufo::models::bge_reranker::npu {

enum class ScoreNpuRoute : std::uint8_t { kNone = 0, kIron };

[[nodiscard]] inline ScoreNpuRoute DecideNpuScore(
    const mbufo::npu::NpuLaneConfig& config, std::size_t seq_len) noexcept {
  (void)config;
  (void)seq_len;
  return ScoreNpuRoute::kNone;
}

}  // namespace mbufo::models::bge_reranker::npu
