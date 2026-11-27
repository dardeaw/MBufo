// EmbeddingGemma 2 (text path) encoder: bidirectional Gemma-4-style
// transformer (sliding-window and full-attention layers, per-layer input
// embeddings) with mean pooling, an output projection and L2 normalization.
// The device implementation lives in hip/session.hip; this header is the
// whole surface the serve layer needs.
#ifndef MBUFO_MODELS_EMBEDDINGGEMMA2_ENGINE_HPP_
#define MBUFO_MODELS_EMBEDDINGGEMMA2_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_lane.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/embeddinggemma2/npu_embed.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace mbufo::core::npu {
class NpuScheduler;
}

namespace mbufo::models::embeddinggemma2 {

/// Longest sequence one forward accepts. The model's native window is far
/// larger; this bounds the device scratch.
inline constexpr std::size_t kMaxTokens = 8192;

struct Config {
  std::uint32_t layers{0};
  std::uint32_t hidden{0};
  std::uint32_t ffn{0};
  std::uint32_t heads{0};
  std::uint32_t head_dim_local{0};
  std::uint32_t head_dim_full{0};
  std::uint32_t kv_local{0};
  std::uint32_t kv_full{0};
  std::uint32_t sliding_window{0};
  std::uint32_t ple_dim{0};
  std::uint32_t out_dim{0};
  float rms_eps{1e-6F};
  float rope_local{10000.0F};
  float rope_full{1000000.0F};
  float embed_scale{1.0F};
  std::vector<std::uint8_t> layer_is_full;
};

struct EmbedResult {
  std::vector<float> values;  // L2-normalized, size == Config::out_dim
  double prefill_ms{0.0};
};

class EngineModel {
 public:
  EngineModel(const EngineModel&) = delete;
  EngineModel& operator=(const EngineModel&) = delete;
  ~EngineModel();

  /// Loads an `embedding_gemma2` GGUF (tools/convert/embeddinggemma2_to_gguf.py).
  [[nodiscard]] static std::unique_ptr<EngineModel> Create(
      const std::string& model_path, std::string* error_msg = nullptr);

  [[nodiscard]] const Config& GetConfig() const noexcept { return config_; }
  [[nodiscard]] const tokenization::QwenTokenizer& Tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const core::GgufReader& GetReader() const noexcept {
    return *reader_;
  }

  /// Tokenizes with <bos> ... <eos> as the model expects.
  [[nodiscard]] std::vector<std::uint32_t> Encode(const std::string& text) const;

  /// One bidirectional forward over `ids` (device path). Not thread-safe;
  /// the caller serializes. nullopt + *error_msg on refusal.
  [[nodiscard]] std::optional<EmbedResult> Embed(
      std::span<const std::uint32_t> ids, std::string* error_msg = nullptr);

  /// One device forward over many sequences (concatenated rows, per-sequence
  /// attention and pooling); results are in input order. Groups that exceed the
  /// envelope (kMaxTokens rows, 64 sequences) are split internally. Not
  /// thread-safe; the caller serializes.
  [[nodiscard]] std::optional<std::vector<EmbedResult>> EmbedBatch(
      std::span<const std::vector<std::uint32_t>> batch,
      std::string* error_msg = nullptr);

  /// Loads exported NPU2 IRON kernels and uploads 24-layer weights into resident
  /// HRX buffers (returns false when MBUFO_ENABLE_NPU=OFF).
  [[nodiscard]] bool LoadNpuWeights(const std::string& kernel_dir = "",
                                    std::string* error_msg = nullptr);
  /// Eject: release the NPU weights (waits for an in-flight NPU run).
  void UnloadNpuWeights();
  [[nodiscard]] bool npu_weights_resident() const noexcept {
    return npu_weights_resident_;
  }

  void SetNpuLaneConfig(mbufo::npu::NpuLaneConfig config) noexcept {
    npu_config_ = std::move(config);
  }
  [[nodiscard]] const mbufo::npu::NpuLaneConfig& npu_lane_config()
      const noexcept {
    return npu_config_;
  }
  void SetNpuScheduler(mbufo::core::npu::NpuScheduler* scheduler) noexcept {
    npu_scheduler_ = scheduler;
  }
  [[nodiscard]] mbufo::core::npu::NpuScheduler* npu_scheduler()
      const noexcept {
    return npu_scheduler_;
  }

 private:
  /// Runs the NPU admission + execution gate when Decided == kIron.
  /// Returns true if the NPU lane handled (or explicitly refused) the call.
  [[nodiscard]] bool TryNpuEmbed(std::span<const std::uint32_t> ids,
                                 std::optional<EmbedResult>* out,
                                 std::string* error_msg);

  EngineModel();
  std::unique_ptr<core::GgufReader> reader_;
  mbufo::npu::NpuLaneConfig npu_config_{};
  mbufo::core::npu::NpuScheduler* npu_scheduler_{nullptr};
  mutable std::mutex npu_mu_;
  bool npu_weights_resident_{false};
  struct NpuState;
  std::shared_ptr<NpuState> npu_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  Config config_;
  struct HipState;
  std::unique_ptr<HipState> hip_;
};

}  // namespace mbufo::models::embeddinggemma2

#endif  // MBUFO_MODELS_EMBEDDINGGEMMA2_ENGINE_HPP_
