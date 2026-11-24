#ifndef MBUFO_MODELS_INKLING_CONFIG_HPP_
#define MBUFO_MODELS_INKLING_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace mbufo::models::inkling {

/// Parsed hyperparameters for `general.architecture == "inkling2"` GGUF artifacts
/// (Inkling-2-Mini: 128 experts top-8, loop-core message passing, NVFP4 experts
/// on blk.0-19 and Q8_0 experts on blk.20-23).
/// Skeleton revision: full load-time geometry is parsed and bound; Forward and
/// the sconv/loop kernels arrive in the next knife.
struct InklingConfig {
  std::string model_name{"Inkling-2-Mini"};
  std::uint32_t vocab_size{201024};
  std::uint32_t vocab_unpadded{200058};
  std::uint32_t hidden_size{2304};
  std::uint32_t intermediate_size{1152};
  std::uint32_t num_layers{24};   // physical blk.0..23 tensor groups
  std::uint32_t block_count{48};  // unrolled schedule (physical + loop repeats)
  std::uint32_t num_q_heads{24};
  std::uint32_t num_kv_heads{8};
  std::uint32_t head_dim{128};
  std::uint32_t value_dim{128};
  std::uint32_t rel_dim{16};
  std::uint32_t rel_extent{1024};

  std::uint32_t context_length{1048576};
  std::uint32_t sliding_window{1024};
  std::vector<bool> is_swa_layer;  // size num_layers; true on blk.5/11/17/23
  float rms_norm_eps{1e-6F};

  std::uint32_t rope_dim{0};  // inkling2.rope.dimension_count == 0 (skeleton)

  std::uint32_t sconv_width{4};  // inkling2.shortconv.l_cache

  // Loop-core message passing (physical blk.6-17).
  std::uint32_t num_loops{3};
  std::uint32_t loop_core_start{6};
  std::uint32_t loop_core_end{18};  // exclusive
  std::uint32_t loop_gate_rank{64};
  std::uint32_t attn_res_block{6};

  // MoE routing.
  std::uint32_t expert_count{128};
  std::uint32_t expert_used_count{8};
  std::uint32_t expert_ffn_size{1152};
  float expert_weights_scale{8.0F};

  // Gating / clipping knobs (consumed by the next-knife Forward).
  float situ_beta{4.0F};
  float situ_linear_beta{25.0F};
  float final_logit_softcap{30.0F};
  float attn_qk_clip{22.627416610717773F};
  float attn_rel_clip{8.0F};
  std::uint32_t log_scale_floor{8192};
  float log_scale_alpha{0.1F};

  [[nodiscard]] static std::optional<InklingConfig> FromGguf(
      const core::GgufReader& gguf, std::string* error_msg = nullptr);
};

}  // namespace mbufo::models::inkling

#endif  // MBUFO_MODELS_INKLING_CONFIG_HPP_