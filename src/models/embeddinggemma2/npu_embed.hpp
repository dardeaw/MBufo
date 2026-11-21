#pragma once
// EmbeddingGemma 2 NPU route decision seam (for EngineModel::Embed).
// Uses verified 24L FFN (assets/e06_ffn.py) + 24L Attention (assets/e07_attn.py)
// + mean-pool/projection/L2 post-processing (npu_pooling.hpp).

#include <cstddef>
#include <cstdint>

#include "src/core/compute/npu/npu_lane.hpp"

namespace mbufo::models::embeddinggemma2::npu {

enum class EmbedNpuRoute : std::uint8_t { kNone = 0, kIron };

[[nodiscard]] inline constexpr std::size_t PaddedSeqLen(
    std::size_t seq_len, std::size_t align = 32u) noexcept {
  if (seq_len == 0 || align == 0) return seq_len;
  return ((seq_len + align - 1u) / align) * align;
}

[[nodiscard]] inline EmbedNpuRoute DecideNpuEmbed(
    const mbufo::npu::NpuLaneConfig& config, std::size_t seq_len,
    bool weights_resident = false, bool device_ready = false) noexcept {
  if (config.embedding && weights_resident && device_ready && seq_len > 0 &&
      seq_len <= 512u) {
    return EmbedNpuRoute::kIron;
  }
  return EmbedNpuRoute::kNone;
}

}  // namespace mbufo::models::embeddinggemma2::npu
