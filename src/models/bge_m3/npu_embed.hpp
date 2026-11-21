#pragma once
// bge-m3 embedding NPU 接縫（代碼框 v0，只宣告、不接線）。
//
// 接線點（未來）：EngineSession::Sync 入口之前，先問
//   DecideNpuEmbed(...) -> kNone 则走現有 HIP/CPU 全序列。
// IRON kernel（LN/gelu/mm/add + Q4NX 橋）落地前，Decide 永遠回 kNone，
// 保證行為零變化。E0/E0.5 spike 證據保留在 ~/coexist_spike 附近筆記。

#include <cstddef>

#include "src/core/compute/npu/npu_lane.hpp"

namespace mbufo::models::bge_m3::npu {

enum class EmbedNpuRoute : std::uint8_t { kNone = 0, kIron };

/// 決策函式：目前永遠 kNone（樁）。未來條件：
//   config.embedding && seq_len % 4 == 0 && NPU lock 可得 && 權重已上 NPU。
[[nodiscard]] inline EmbedNpuRoute DecideNpuEmbed(
    const mbufo::npu::NpuLaneConfig& config, std::size_t seq_len) noexcept {
  (void)config;
  (void)seq_len;
  return EmbedNpuRoute::kNone;
}

}  // namespace mbufo::models::bge_m3::npu
