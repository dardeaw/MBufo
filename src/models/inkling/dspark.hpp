#ifndef MBUFO_MODELS_INKLING_DSPARK_HPP_
#define MBUFO_MODELS_INKLING_DSPARK_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/models/inkling/weights.hpp"

namespace mbufo::models::inkling {

/// DSpark / DFlash 3-layer block-diffusion drafter for Inkling-2-Mini.
class DsparkModel {
 public:
  static constexpr std::uint32_t kNumLayers = 3;
  static constexpr std::uint32_t kBlockSize = 8;
  static constexpr std::uint32_t kMaxDraft = kBlockSize - 1;  // 7
  static constexpr std::uint32_t kNumTargetLayers = 4;

  ~DsparkModel();
  DsparkModel(const DsparkModel&) = delete;
  DsparkModel& operator=(const DsparkModel&) = delete;

  [[nodiscard]] static std::unique_ptr<DsparkModel> Load(
      const std::string& dspark_path, std::uint32_t hidden_size,
      std::uint32_t vocab_size, std::string* error_msg);

  [[nodiscard]] std::size_t ResidentBytes() const noexcept {
    return resident_bytes_;
  }
  [[nodiscard]] std::uint32_t MaxDraft() const noexcept { return kMaxDraft; }
  [[nodiscard]] const std::array<std::uint32_t, kNumTargetLayers>&
  TargetLayers() const noexcept {
    return target_layers_;
  }

  struct Impl;
  [[nodiscard]] Impl& impl() const noexcept { return *impl_; }

 private:
  DsparkModel() = default;
  std::unique_ptr<Impl> impl_;
  std::size_t resident_bytes_{0};
  std::array<std::uint32_t, kNumTargetLayers> target_layers_{5, 23, 41, 48};
};

class DsparkSession {
 public:
  ~DsparkSession();
  DsparkSession(const DsparkSession&) = delete;
  DsparkSession& operator=(const DsparkSession&) = delete;

  [[nodiscard]] static std::unique_ptr<DsparkSession> Create(
      const DsparkModel& model, std::uint32_t max_context,
      std::uint32_t vocab_rows, hipStream_t stream, std::string* error_msg);

  /// Store target hidden states [4][n_tokens, hidden] at [start_pos..start_pos+n_tokens)
  /// into DSpark context KV cache.
  [[nodiscard]] bool CaptureContext(
      const std::array<const float*, DsparkModel::kNumTargetLayers>& target_h,
      std::uint32_t start_pos, std::uint32_t n_tokens, std::string* error_msg);

  /// Propose `n_draft` (1..7) tokens from `anchor_token` at `anchor_pos`.
  [[nodiscard]] bool DraftBlock(
      std::int32_t anchor_token, std::uint32_t anchor_pos,
      std::uint32_t n_draft, const TensorRef& target_token_embd,
      const TensorRef& target_output, std::vector<std::int32_t>* out_drafts,
      std::string* error_msg);

  struct Snapshot {
    std::uint32_t pos{0};
    std::uint32_t kv_dim{0};
    std::uint32_t hidden{0};
    std::array<std::vector<std::uint16_t>, DsparkModel::kNumLayers> k_ctx;
    std::array<std::vector<std::uint16_t>, DsparkModel::kNumLayers> v_ctx;
    std::vector<float> markov_ctx;

    [[nodiscard]] std::size_t PayloadBytes() const noexcept {
      std::size_t total = markov_ctx.size() * sizeof(float);
      for (std::size_t i = 0; i < DsparkModel::kNumLayers; ++i) {
        total += (k_ctx[i].size() + v_ctx[i].size()) * sizeof(std::uint16_t);
      }
      return total;
    }
  };

  [[nodiscard]] std::size_t SnapshotBytes(std::uint32_t pos) const noexcept;
  [[nodiscard]] bool SaveSnapshot(std::uint32_t pos, Snapshot* out,
                                  std::string* error_msg) const;
  [[nodiscard]] bool RestoreSnapshot(const Snapshot& snap,
                                     std::string* error_msg);

 private:
  DsparkSession() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mbufo::models::inkling

#endif  // MBUFO_MODELS_INKLING_DSPARK_HPP_
