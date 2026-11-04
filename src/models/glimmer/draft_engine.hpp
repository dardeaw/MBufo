#ifndef GUFO_MODELS_GLIMMER_DRAFT_ENGINE_HPP_
#define GUFO_MODELS_GLIMMER_DRAFT_ENGINE_HPP_

// DFlash draft runtime (muse-glimmer companion draft, e.g. dflash-kquant):
// a depth-pruned dense decoder (5 layers, same width/dim as the target)
// evaluated autoregressively for speculative proposals. QK-norm retained,
// attention gate and post-norms absent (per the draft tensors). KV lives on
// the shared blocked-Q8 grid (core blocked_kv); no F16 mirror is kept.
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/models/glimmer/weights.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::glimmer::draft {

struct DraftModelOptions {
  std::uint32_t max_context = 32768;
};

class DraftSession;

class DraftModel final : public std::enable_shared_from_this<DraftModel> {
 public:
  ~DraftModel();
  DraftModel(const DraftModel&) = delete;
  DraftModel& operator=(const DraftModel&) = delete;

  [[nodiscard]] static std::shared_ptr<DraftModel> Load(
      const std::string& model_path, const DraftModelOptions& options,
      const std::vector<std::uint32_t>& target_layer_ids,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<DraftSession> CreateSession(
      std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(
      std::string_view text, bool add_bos) const;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] const std::vector<std::uint32_t>& TargetLayerIds() const noexcept {
    return target_layer_ids_;
  }

 private:
  DraftModel() = default;

  DraftModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::vector<void*> device_allocations_;
  std::vector<std::uint32_t> target_layer_ids_;

  friend class DraftSession;
};

/// One draft conversation. Sync feeds a prompt; Evaluate appends one token
/// (greedy proposal). KV is blocked-Q8 shared-grid (core blocked_kv) with
/// greedy argmax sampling.
class DraftSession final {
 public:
  ~DraftSession();
  DraftSession(const DraftSession&) = delete;
  DraftSession& operator=(const DraftSession&) = delete;

  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  [[nodiscard]] bool Forward(std::span<const std::int32_t> tokens,
                             std::uint32_t start_pos,
                             std::string* error_msg = nullptr);
  [[nodiscard]] std::int32_t SampleGreedy();
  [[nodiscard]] std::uint32_t Length() const noexcept;

 private:
  DraftSession(std::shared_ptr<DraftModel> model);

  std::shared_ptr<DraftModel> model_;
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class DraftModel;
};

}  // namespace gufo::models::glimmer::draft

#endif  // GUFO_MODELS_GLIMMER_DRAFT_ENGINE_HPP_
