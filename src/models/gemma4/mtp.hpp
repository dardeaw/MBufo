#ifndef GUFO_MODELS_GEMMA4_MTP_HPP_
#define GUFO_MODELS_GEMMA4_MTP_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include "src/core/gguf_reader.hpp"


namespace gufo::models::gemma4 {

/// Gemma 4 Assistant (MTP draft head) — architecture per
/// transformers/modeling_gemma4_assistant.py:
///   4 dense decoder layers, draft hidden 1024, 16 query heads,
///   head_dim 256 for the sliding layers and 512 for the full layer.
///   The draft has NO k/v projections: every draft layer cross-attends the
///   target model's KV cache of the matching layer type, so it never
///   populates or advances a cache of its own.
///   Inputs are pre_projection(concat(target_embedding, target_hidden)),
///   outputs feed post_projection back into the target hidden width.
class MtpDrafter final {
public:
  ~MtpDrafter();
  MtpDrafter(const MtpDrafter&) = delete;
  MtpDrafter& operator=(const MtpDrafter&) = delete;

  /// `backbone_hidden` is the TARGET hidden width (2816 for 26B-A4B).
  [[nodiscard]] static std::unique_ptr<MtpDrafter> Load(
      const std::string& path, std::uint32_t backbone_hidden,
      std::string* error_msg);

  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t HiddenSize() const noexcept;
  [[nodiscard]] std::uint32_t BackboneHidden() const noexcept;
  /// Layer index whose KV the draft layer `draft_layer` cross-attends, or -1
  /// when the caller must pass a null cache (draft layer type unused).
  [[nodiscard]] int SwaLayer() const noexcept;
  [[nodiscard]] int FullLayer() const noexcept;

  /// One draft step. `tgt_emb` / `tgt_hidden` are the target's embedding and
  /// post-norm residual for the last committed token (both backbone_hidden
  /// wide, device F32). Returns the drafted token id (argmax of the draft
  /// logits) and writes the post_projection feedback row, which the next
  /// step consumes as its hidden input.
  [[nodiscard]] bool Step(int32_t token, const float* tgt_emb,
                          const float* tgt_hidden, const void* swa_k,
                          const void* swa_v, std::uint32_t swa_stride,
                          std::uint32_t swa_kv_heads, const void* full_k,
                          const void* full_v, std::uint32_t full_stride,
                          std::uint32_t full_kv_heads, std::uint32_t pos,
                          float rope_theta_swa, float rope_theta_full,
                          std::uint32_t window_swa, float* feedback_hidden,
                          int32_t* out_token, std::string* error_msg);

private:
  MtpDrafter();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_MTP_HPP_
