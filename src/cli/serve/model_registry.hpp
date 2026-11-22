// Multi-model serving registry: one HTTP port, many loaded GGUF models, and
// dynamic load/eject through the admin API. When exactly one model is
// registered the request path is byte-for-byte the single-model behavior; the
// registry only changes *which* backend instance a request is dispatched to.
#ifndef MBUFO_SERVER_MODEL_REGISTRY_HPP_
#define MBUFO_SERVER_MODEL_REGISTRY_HPP_

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"

namespace mbufo::server {

/// Thread-safe map of model name -> loaded text backend. The HTTP layer holds
/// a shared_ptr for the whole request, so an ejected model stays alive until
/// its last in-flight request finishes: Remove() drops the registry's own
/// reference immediately, and the GPU memory follows the last shared_ptr.
struct LoadedModelConfig {
  std::string kv_cache_quant{"q4"};
  std::string speculative{"off"};
  std::string draft_model_path;
  std::string vision_model_path;
  bool enable_yarn{false};
  bool enable_disk_cache{false};
  std::string device{"gpu"};
  std::string npu_quant;
};

class ModelRegistry final {
public:
  ModelRegistry() = default;
  ModelRegistry(const ModelRegistry&) = delete;
  ModelRegistry& operator=(const ModelRegistry&) = delete;

  /// Registers `backend` under `primary` plus any extra names. Fails when the
  /// backend is null, the primary name is empty, or any name is already taken.
  [[nodiscard]] bool Add(const std::string& primary,
                         std::vector<std::string> extra_names,
                         std::shared_ptr<TextGenerationBackend> backend,
                         std::string* error,
                         LoadedModelConfig config = {});

  /// Drops the registry's reference to the model whose *primary* id is `id`.
  /// Returns false when no primary id matches.
  [[nodiscard]] bool Remove(const std::string& id, std::string* error);

  /// Promotes the loaded model whose primary id is `id` to the default
  /// (primary) position, protecting it from LRU eviction and routing requests
  /// that omit `model` to it.
  [[nodiscard]] bool SetDefault(const std::string& id, std::string* error);

  /// Resolves a requested name (primary id or alias). Null = not loaded.
  /// Empty name returns the default model. Touches last_used for accounting.
  [[nodiscard]] std::shared_ptr<TextGenerationBackend> Resolve(
      std::string_view name);

  /// The first registered ready model: used when a request omits `model` and
  /// for the llama-compatible endpoints. Null when the registry is empty.
  [[nodiscard]] std::shared_ptr<TextGenerationBackend> Default();

  /// Snapshot for /admin/models, sorted by primary id.
  struct Info {
    std::string id;
    std::string path;
    std::string backend{"gpu"};
    std::vector<std::string> names;  // primary + aliases
    std::uint32_t max_context{0};
    std::uint64_t last_used_ms{0};
    std::uint64_t idle_ms{0};
    bool ready{false};
    bool is_default{false};
    bool supports_images{false};
    bool speculative{false};
    LoadedModelConfig config;
    std::size_t total_sessions{0};
    std::size_t active_sessions{0};
    std::vector<TextGenerationBackend::SessionState> sessions;
  };
  [[nodiscard]] std::vector<Info> List() const;

  /// True when `name` is some model's primary id (not merely an alias).
  [[nodiscard]] bool HasId(std::string_view name) const;

  /// True when `name` resolves to any loaded model (id or alias).
  [[nodiscard]] bool HasName(std::string_view name) const;

  /// Ejectable candidates for memory-pressure eviction: primary ids ordered
  /// by last_used ascending (least-recently-used first). The default (first
  /// registered ready) model is excluded so eviction never empties the
  /// registry. Empty when there is nothing safe to evict.
  [[nodiscard]] std::vector<std::string> LruCandidates() const;

  [[nodiscard]] bool AnyDeviceLost() const;
  [[nodiscard]] std::optional<DiskCacheSnapshot> InspectDiskCache() const;
  [[nodiscard]] std::optional<DiskCacheDeleteResult> RetireDiskCache(
      const std::vector<std::uint64_t>& ids, bool delete_all,
      bool delete_incompatible) const;
  [[nodiscard]] std::size_t Size() const;

private:
  struct Entry {
    std::string id;
    std::vector<std::string> names;  // primary + aliases
    std::shared_ptr<TextGenerationBackend> backend;
    LoadedModelConfig config;
    std::atomic<std::uint64_t> last_used{0};  // steady_clock ms since epoch
  };

  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<Entry>> order_;  // registration order
  std::unordered_map<std::string, std::weak_ptr<Entry>> by_name_;
};

}  // namespace mbufo::server

#endif  // MBUFO_SERVER_MODEL_REGISTRY_HPP_
