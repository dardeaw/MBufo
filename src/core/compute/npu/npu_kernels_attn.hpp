#pragma once
// NPU attention specification, hardware BD/L1/GQA guards, and CPU reference
// for EmbeddingGemma 2 (bidirectional unscaled sliding + full attention).
//
// Why `e07_attn.py` instead of SDK `flash_attn_prefill.cc` (`prefill_fv`):
//   1. SDK `mha.cc` and `flash_attn_prefill.cc` hardcode CAUSAL masking and
//      1/sqrt(d) scaling (scale = log2(e) / sqrt(d)).
//   2. EmbeddingGemma 2 (`src/models/embeddinggemma2/hip/session.hip`) uses
//      BIDIRECTIONAL attention with symmetric sliding window (`|q - k| <= 512`)
//      and UNSCALED QK^T logits (`alpha = 1.0F`, already absorbed by q/k norm).
//   3. Therefore, production NPU2 attention uses `assets/e07_attn.py` (20/28-core
//      fused 4-stage pipeline), while `ValidatePrefillFv` records SDK causal-only
//      prefill constraints for standard causal LLMs.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_kernels.hpp"

namespace mbufo::core::npu {

/// Hardware architecture bounds established on RyzenAI-npu5 (XDNA2 / AIE2p).
inline constexpr std::size_t kNpu2CoreTileL1Bytes = 65536;       // 64 KiB
inline constexpr std::size_t kNpu2MaxCoreInDmaChannels = 2;      // CoreTile S2MM
inline constexpr std::size_t kNpu2MaxCoreOutDmaChannels = 2;     // CoreTile MM2S
inline constexpr std::size_t kNpu2MaxCoreBdDimWords = 255;       // 32b words/BD dim
inline constexpr std::size_t kNpu2MaxMemTileBdDimWords = 1023;   // MemTile BD dim
inline constexpr std::size_t kNpu2MaxShimRepeatBatch = 4;        // >4 + repeat>1 -> ERT timeout

struct AttnLayerSpec {
  std::size_t seq_len{128};
  std::size_t hidden_dim{512};
  std::size_t num_q_heads{4};
  std::size_t num_kv_heads{2};
  std::size_t head_dim{256};
  std::size_t sliding_window{512};
  bool is_full{false};
  bool is_causal{false};
  float qk_scale{1.0f};
  // Kernel micro-tiling & DMA batching parameters (validated against NPU2 limits).
  std::size_t m_q_tile{32};
  std::size_t k_tile{64};
  std::size_t d_step{64};
  std::size_t shim_batch_limit{4};
  std::size_t core_in_channels{2};
  std::size_t core_out_channels{1};
  std::size_t l1_budget_bytes{kNpu2CoreTileL1Bytes};
  bool double_buffered{true};
};

struct AttnHardwareBudget {
  std::size_t gqa_ratio{0};
  std::size_t q_tiles{0};
  std::size_t kv_tiles{0};
  std::size_t k_subtiles{0};
  std::size_t b_slots_per_core{0};
  std::size_t shim_batches_per_head{0};
  std::size_t qkt_l1_bytes{0};
  std::size_t pv_l1_bytes{0};
  std::size_t core_bd_inner_words{0};
  std::size_t memtile_bd_words{0};
};

/// True for global-attention layers in EmbeddingGemma 2 (0-based indices 5, 11, 17, 23
/// per model `config.json` `layer_types` and `embeddinggemma2/engine.cpp:69`;
/// note: ember's handoff memo also listed layer 4, kept documented here).
[[nodiscard]] inline constexpr bool EmbGemma2LayerIsFull(
    std::size_t layer_idx, std::size_t full_period = 6) noexcept {
  return full_period != 0 && ((layer_idx + 1u) % full_period) == 0u;
}

[[nodiscard]] inline AttnLayerSpec EmbGemma2LocalSpec(
    std::size_t seq_len = 128) noexcept {
  AttnLayerSpec s;
  s.seq_len = seq_len;
  s.hidden_dim = 512;
  s.num_q_heads = 4;
  s.num_kv_heads = 2;
  s.head_dim = 256;
  s.sliding_window = 512;
  s.is_full = false;
  s.is_causal = false;
  s.qk_scale = 1.0f;
  s.m_q_tile = 32;
  s.k_tile = 64;
  s.d_step = 64;
  return s;
}

[[nodiscard]] inline AttnLayerSpec EmbGemma2FullSpec(
    std::size_t seq_len = 128) noexcept {
  AttnLayerSpec s;
  s.seq_len = seq_len;
  s.hidden_dim = 512;
  s.num_q_heads = 4;
  s.num_kv_heads = 1;
  s.head_dim = 512;
  s.sliding_window = 0;
  s.is_full = true;
  s.is_causal = false;
  s.qk_scale = 1.0f;
  s.m_q_tile = 32;
  s.k_tile = 64;
  s.d_step = 64;
  return s;
}

/// Validates GQA, sliding window, CoreTile L1 SRAM, DMA channels, B-slots,
/// and Shim repeat batch limits to prevent BD overflow and ERT DEADLINE_EXCEEDED.
[[nodiscard]] inline std::string ValidateSlidingAttn(
    const AttnLayerSpec& spec, AttnHardwareBudget* out_budget = nullptr) {
  if (spec.seq_len == 0 || spec.hidden_dim == 0 || spec.num_q_heads == 0 ||
      spec.num_kv_heads == 0 || spec.head_dim == 0) {
    return "attn: dimensions must be nonzero";
  }
  if (spec.num_q_heads % spec.num_kv_heads != 0) {
    return "attn: num_q_heads must be divisible by num_kv_heads (GQA ratio)";
  }
  if (spec.head_dim != 64 && spec.head_dim != 128 && spec.head_dim != 256 &&
      spec.head_dim != 512) {
    return "attn: head_dim must be one of {64, 128, 256, 512}";
  }
  if (const std::string rope_err = ValidateRope(spec.head_dim, true);
      !rope_err.empty()) {
    return "attn: " + rope_err;
  }
  if (spec.m_q_tile == 0 || (spec.m_q_tile % 8 != 0) ||
      (spec.seq_len % spec.m_q_tile != 0)) {
    return "attn: seq_len must be divisible by 8-aligned m_q_tile";
  }
  if (spec.k_tile == 0 || (spec.k_tile % 32 != 0) ||
      (spec.seq_len % spec.k_tile != 0)) {
    return "attn: seq_len must be divisible by 32-aligned k_tile (no BD overflow)";
  }
  if (spec.d_step == 0 || (spec.d_step % 32 != 0) ||
      (spec.head_dim % spec.d_step != 0)) {
    return "attn: head_dim must be divisible by 32-aligned d_step";
  }
  if (!spec.is_full) {
    if (spec.sliding_window == 0 || (spec.sliding_window % 32 != 0)) {
      return "attn: sliding attention requires positive 32-aligned sliding_window";
    }
  }
  if (spec.shim_batch_limit == 0 ||
      spec.shim_batch_limit > kNpu2MaxShimRepeatBatch) {
    return "attn: shim_batch_limit must be in [1, 4] (E0.7: >4 repeat tasks "
           "on one Shim channel hit ERT DEADLINE_EXCEEDED / state 8)";
  }
  if (spec.core_in_channels == 0 ||
      spec.core_in_channels > kNpu2MaxCoreInDmaChannels ||
      spec.core_out_channels == 0 ||
      spec.core_out_channels > kNpu2MaxCoreOutDmaChannels) {
    return "attn: exceeds NPU2 CoreTile DMA channels (max 2 in, 2 out)";
  }

  AttnHardwareBudget budget{};
  budget.gqa_ratio = spec.num_q_heads / spec.num_kv_heads;
  budget.q_tiles = spec.seq_len / spec.m_q_tile;
  budget.kv_tiles = spec.seq_len / spec.k_tile;
  budget.k_subtiles = spec.head_dim / spec.d_step;
  budget.b_slots_per_core = 1;
  budget.shim_batches_per_head =
      (budget.q_tiles + spec.shim_batch_limit - 1u) / spec.shim_batch_limit;

  const std::size_t buf_mult = spec.double_buffered ? 2u : 1u;
  const std::size_t q_bytes = spec.m_q_tile * spec.d_step * sizeof(std::uint16_t);
  const std::size_t k_bytes = spec.k_tile * spec.d_step * sizeof(std::uint16_t);
  const std::size_t s_bytes = spec.m_q_tile * spec.k_tile * sizeof(std::uint16_t);
  budget.qkt_l1_bytes = buf_mult * (q_bytes + k_bytes + s_bytes);

  const std::size_t v_bytes = spec.k_tile * spec.d_step * sizeof(std::uint16_t);
  const std::size_t o_bytes = spec.m_q_tile * spec.d_step * sizeof(std::uint16_t);
  budget.pv_l1_bytes = buf_mult * (s_bytes + v_bytes + o_bytes);

  if (budget.qkt_l1_bytes > spec.l1_budget_bytes ||
      budget.pv_l1_bytes > spec.l1_budget_bytes) {
    return "attn: L1 tile working set exceeds NPU2 CoreTile 64 KiB budget";
  }

  budget.core_bd_inner_words =
      (std::max({spec.d_step, spec.k_tile, spec.m_q_tile}) * sizeof(std::uint16_t)) /
      sizeof(std::uint32_t);
  if (budget.core_bd_inner_words > kNpu2MaxCoreBdDimWords) {
    return "attn: CoreTile BD dim exceeds 255-word hardware limit (BD overflow)";
  }
  budget.memtile_bd_words =
      (spec.head_dim * sizeof(std::uint16_t)) / sizeof(std::uint32_t);
  if (budget.memtile_bd_words > kNpu2MaxMemTileBdDimWords) {
    return "attn: MemTile BD dim exceeds 1023-word hardware limit";
  }

  if (out_budget != nullptr) *out_budget = budget;
  return {};
}

/// Validates SDK `linalg.prefill_fv(head_dim)` eligibility.
[[nodiscard]] inline std::string ValidatePrefillFv(std::size_t head_dim,
                                                   bool is_causal) {
  if (head_dim != 256 && head_dim != 512) {
    return "prefill_fv: head_dim must be 512 (global) or 256 (sliding-window)";
  }
  if (!is_causal) {
    return "prefill_fv only supports causal attention with 1/sqrt(d) scaling; "
           "EmbeddingGemma2 bidirectional unscaled SWA uses assets/e07_attn.py";
  }
  return {};
}

/// CPU fp32 reference for GQA attention (supports bidirectional SWA and full).
inline void RefBidirectionalGqaAttention(
    const AttnLayerSpec& spec, const std::vector<float>& q,
    const std::vector<float>& k, const std::vector<float>& v,
    std::vector<float>& out) {
  const std::size_t S = spec.seq_len;
  const std::size_t Hq = spec.num_q_heads;
  const std::size_t Hkv = spec.num_kv_heads;
  const std::size_t D = spec.head_dim;
  const std::size_t gqa = Hq / Hkv;
  out.assign(Hq * S * D, 0.0f);

  std::vector<double> scores(S, 0.0);
  std::vector<double> probs(S, 0.0);

  for (std::size_t h = 0; h < Hq; ++h) {
    const std::size_t kvh = h / gqa;
    const std::size_t q_head_off = h * S * D;
    const std::size_t kv_head_off = kvh * S * D;
    for (std::size_t qi = 0; qi < S; ++qi) {
      double max_s = -1e30;
      for (std::size_t ki = 0; ki < S; ++ki) {
        bool masked = false;
        if (spec.is_causal && ki > qi) masked = true;
        if (!spec.is_full && spec.sliding_window > 0) {
          const std::size_t diff = (qi >= ki) ? (qi - ki) : (ki - qi);
          if (diff > spec.sliding_window) masked = true;
        }
        if (masked) {
          scores[ki] = -1.0e9;
          continue;
        }
        double dot = 0.0;
        const std::size_t q_row = q_head_off + qi * D;
        const std::size_t k_row = kv_head_off + ki * D;
        for (std::size_t di = 0; di < D; ++di) {
          dot += static_cast<double>(q[q_row + di]) *
                 static_cast<double>(k[k_row + di]);
        }
        dot *= static_cast<double>(spec.qk_scale);
        scores[ki] = dot;
        max_s = std::max(max_s, dot);
      }
      double sum_p = 0.0;
      for (std::size_t ki = 0; ki < S; ++ki) {
        if (scores[ki] <= -1.0e8) {
          probs[ki] = 0.0;
        } else {
          probs[ki] = std::exp(scores[ki] - max_s);
          sum_p += probs[ki];
        }
      }
      const double inv_sum = (sum_p > 0.0) ? (1.0 / sum_p) : 0.0;
      const std::size_t o_row = q_head_off + qi * D;
      for (std::size_t ki = 0; ki < S; ++ki) {
        const double p = probs[ki] * inv_sum;
        if (p == 0.0) continue;
        const std::size_t v_row = kv_head_off + ki * D;
        for (std::size_t di = 0; di < D; ++di) {
          out[o_row + di] += static_cast<float>(p * static_cast<double>(v[v_row + di]));
        }
      }
    }
  }
}

}  // namespace mbufo::core::npu
