#pragma once
// P0 shared GQA KV-cache (header-only, zero wiring).
//
// One layout serves every decoder on the roadmap:
//   ASR thinker (28L, Q=16x128 KV=8x128), TTS talker (28L) + predictor (5L),
//   Qwen3-4B TE is prefill-only (no cache needed, kept for interface unity).
// Host-FP32 resident; NPU GEMM streams weights, K/V never leave host RAM.
// Peak estimate (ASR, seq=512): 28*512*1024*4B*2 ~= 117 MiB. Well under budget.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mbufo::core::kv {

struct GqaKvConfig {
  std::size_t layers{0};
  std::size_t kv_heads{0};
  std::size_t head_dim{128};
  std::size_t max_seq{512};
};

class GqaKvCache {
 public:
  explicit GqaKvCache(const GqaKvConfig& cfg = {}) { Reset(cfg); }
  void Reset(const GqaKvConfig& cfg) {
    cfg_ = cfg;
    const std::size_t kv_dim = cfg.kv_heads * cfg.head_dim;
    k_.assign(cfg.layers * cfg.max_seq * kv_dim, 0.0F);
    v_.assign(cfg.layers * cfg.max_seq * kv_dim, 0.0F);
    filled_ = 0;
  }
  [[nodiscard]] std::size_t filled() const noexcept { return filled_; }
  [[nodiscard]] const GqaKvConfig& config() const noexcept { return cfg_; }
  /// Append one decoded position (K/V rows of kv_dim) for every layer.
  void Append(const std::vector<float>& k_row_per_layer, const std::vector<float>& v_row_per_layer) {
    const std::size_t kv_dim = cfg_.kv_heads * cfg_.head_dim;
    for (std::size_t l = 0; l < cfg_.layers; ++l) {
      float* kd = k_.data() + (l * cfg_.max_seq + filled_) * kv_dim;
      float* vd = v_.data() + (l * cfg_.max_seq + filled_) * kv_dim;
      const float* ks = k_row_per_layer.data() + l * kv_dim;
      const float* vs = v_row_per_layer.data() + l * kv_dim;
      for (std::size_t i = 0; i < kv_dim; ++i) { kd[i] = ks[i]; vd[i] = vs[i]; }
    }
    ++filled_;
  }
  [[nodiscard]] const float* K(std::size_t layer) const noexcept {
    const std::size_t kv_dim = cfg_.kv_heads * cfg_.head_dim;
    return k_.data() + layer * cfg_.max_seq * kv_dim;
  }
  [[nodiscard]] const float* V(std::size_t layer) const noexcept {
    const std::size_t kv_dim = cfg_.kv_heads * cfg_.head_dim;
    return v_.data() + layer * cfg_.max_seq * kv_dim;
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return (k_.size() + v_.size()) * sizeof(float); }

 private:
  GqaKvConfig cfg_{};
  std::vector<float> k_;
  std::vector<float> v_;
  std::size_t filled_{0};
};

}  // namespace mbufo::core::kv
