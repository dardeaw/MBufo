#ifndef GUFO_MODELS_GEMMA4_MTP_HPP_
#define GUFO_MODELS_GEMMA4_MTP_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4 {

/// Gemma 4 Assistant (MTP draft head) — architecture per llama.cpp
/// gemma4-assistant.cpp & common/speculative.cpp (draft-mtp):
///   4 dense decoder layers, draft hidden 1024, 16 query heads,
///   head_dim 256 for the sliding layers and 512 for the full layer.
///   Cross-attends the target model's shared KV caches at position `rope_pos`
///   (`n_past`) over valid cache prefix `[0, kv_len)`.
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
  [[nodiscard]] int SwaLayer() const noexcept;
  [[nodiscard]] int FullLayer() const noexcept;

  [[nodiscard]] bool SetTargetNorm(const float* raw_hidden, const float* norm_w,
                                   float eps);
  [[nodiscard]] const float* TargetNorm() const;

  /// Asynchronous GPU draft step on stream 0. Writes post_projection(`d_norm`)
  /// to `feedback_hidden` and greedy argmax token ID to device `d_out_token`.
  bool Step(const float* tgt_emb, const float* tgt_hidden, const void* swa_k,
            const void* swa_v, std::uint32_t swa_stride,
            std::uint32_t swa_kv_heads, const void* full_k, const void* full_v,
            std::uint32_t full_stride, std::uint32_t full_kv_heads,
            std::uint32_t kv_len, std::uint32_t rope_pos,
            std::uint32_t sliding_window, float rope_theta_swa,
            float rope_theta_full, const float* rope_freqs_full,
            float* feedback_hidden, std::int32_t* d_out_token,
            std::string* error_msg);

private:
  MtpDrafter();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_MTP_HPP_
