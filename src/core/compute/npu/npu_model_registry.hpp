#pragma once
// NPU model registry (v1, header-only, fully implemented, zero wiring).
//
// Mirrors ModelRegistry (src/cli/serve/model_registry.hpp) method-for-method
// so server code uses one idiom on both sides:
//   Add / Remove / SetDefault / Resolve / Default / List / HasId / HasName /
//   LruCandidates / Size
// Same lifetime contract: the HTTP layer holds a shared_ptr for the whole
// request, so ejecting a model never kills in-flight work; Remove() drops
// only the registry's own reference.
//
// Differences from the GPU side (all NPU-forced):
// - backends are descriptors, not live objects: an NPU entry is either an
//   IRON xclbin set (future in-process) or a sidecar endpoint (flm / gufo
//   serve). `ready` is declared at Add() time and refreshed by probes.
// - six lanes, not one: embedding / reranker / decide / asr / tts / image.
// - LRU eviction for sidecars means "stop the sidecar process", never free().

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mbufo::core::npu {

/// Which of the six NPU lanes an entry serves.
/// kImage covers FLUX.2-klein / Z-Image-Turbo (TE + DiT, in-process NPU).
enum class NpuModelLane : std::uint8_t {
  kEmbedding = 0,
  kReranker,
  kDecide,
  kAsr,
  kTts,
  kImage,
};

/// Where the model actually runs.
enum class NpuArtifactKind : std::uint8_t {
  kIronXclbin = 0,  ///< in-process IRON graph (embedding/reranker future)
  kFlmSidecar,      ///< flm process (D3 decide, Whisper ASR)
  kGufoSidecar,     ///< gufo serve tts/asr process
  kNpuInProcess,    ///< in-process NPU-GEMM servable (this branch: thinker /
                    ///< talker+predictor+frame loop / shared TE / FLUX+ZImage
                    ///< DiTs). On-demand load only, never boot autoload.
};

/// Backend descriptor stored per entry (plain data, no live handles).
struct NpuModelSpec {
  NpuModelLane lane{NpuModelLane::kEmbedding};
  NpuArtifactKind kind{NpuArtifactKind::kFlmSidecar};
  /// xclbin dir (iron) or base URL (sidecar), e.g. http://127.0.0.1:8766.
  std::string artifact;
  /// HTTP path for sidecars, e.g. /v1/systemone. Empty for iron.
  std::string endpoint;
  /// Default port, for the admin UI sidecar cards.
  int default_port{0};
  /// Estimated resident MiB (weights + runtime), for the memory gate.
  std::size_t estimated_resident_mib{0};
  /// Exact launch command for on-demand (non-boot) startup.
  std::string launch_cmd;
};

/// Thread-safe map of model name -> NPU backend descriptor.
class NpuModelRegistry final {
 public:
  NpuModelRegistry() = default;
  NpuModelRegistry(const NpuModelRegistry&) = delete;
  NpuModelRegistry& operator=(const NpuModelRegistry&) = delete;

  /// Registers `spec` under `primary` plus aliases. Fails when primary is
  /// empty or any name is taken.
  [[nodiscard]] bool Add(const std::string& primary,
                         std::vector<std::string> extra_names,
                         NpuModelSpec spec, bool ready, std::string* error) {
    if (primary.empty()) {
      if (error != nullptr) *error = "primary id is empty";
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (by_name_.count(primary) != 0u) {
      if (error != nullptr) *error = "name already taken: " + primary;
      return false;
    }
    for (const auto& n : extra_names) {
      if (by_name_.count(n) != 0u) {
        if (error != nullptr) *error = "name already taken: " + n;
        return false;
      }
    }
    auto entry = std::make_shared<Entry>();
    entry->id = primary;
    entry->names.push_back(primary);
    for (const auto& n : extra_names) entry->names.push_back(n);
    entry->spec = std::move(spec);
    entry->ready = ready;
    entry->is_default = order_.empty();
    entry->last_used = NowMs();
    for (const auto& n : entry->names) by_name_[n] = entry;
    order_.push_back(std::move(entry));
    return true;
  }

  /// Drops the registry's reference to `id`. In-flight holders keep theirs.
  [[nodiscard]] bool Remove(const std::string& id, std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = order_.begin(); it != order_.end(); ++it) {
      if ((*it)->id == id) {
        for (const auto& n : (*it)->names) by_name_.erase(n);
        const bool was_default = (*it)->is_default;
        order_.erase(it);
        if (was_default && !order_.empty()) order_.front()->is_default = true;
        return true;
      }
    }
    if (error != nullptr) *error = "unknown npu model id: " + id;
    return false;
  }

  /// Promote `id` to default (protected from LRU, serves nameless requests).
  [[nodiscard]] bool SetDefault(const std::string& id, std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_ptr<Entry> found;
    for (const auto& e : order_) {
      if (e->id == id) found = e;
    }
    if (found == nullptr) {
      if (error != nullptr) *error = "unknown npu model id: " + id;
      return false;
    }
    for (const auto& e : order_) e->is_default = (e == found);
    return true;
  }

  /// Resolve primary id or alias. Empty name returns Default().
  [[nodiscard]] std::shared_ptr<const NpuModelSpec> Resolve(
      std::string_view name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (name.empty()) {
      for (const auto& e : order_) {
        if (e->is_default && e->ready) {
          e->last_used = NowMs();
          return std::shared_ptr<const NpuModelSpec>(e, &e->spec);
        }
      }
      return nullptr;
    }
    auto it = by_name_.find(std::string(name));
    if (it == by_name_.end()) return nullptr;
    auto e = it->second.lock();
    if (e == nullptr) return nullptr;
    e->last_used = NowMs();
    return std::shared_ptr<const NpuModelSpec>(e, &e->spec);
  }

  /// First registered ready model. Null when empty.
  [[nodiscard]] std::shared_ptr<const NpuModelSpec> Default() {
    return Resolve(std::string_view{});
  }

  /// Flip readiness (sidecar probe results land here).
  bool SetReady(const std::string& id, bool ready) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& e : order_) {
      if (e->id == id) {
        e->ready = ready;
        return true;
      }
    }
    return false;
  }

  /// Snapshot for /admin/models, sorted by primary id.
  struct Info {
    std::string id;
    std::vector<std::string> names;
    NpuModelLane lane{NpuModelLane::kEmbedding};
    NpuArtifactKind kind{NpuArtifactKind::kFlmSidecar};
    std::string artifact;
    std::string endpoint;
    int default_port{0};
    std::size_t estimated_resident_mib{0};
    bool ready{false};
    bool is_default{false};
    std::uint64_t last_used_ms{0};
  };
  [[nodiscard]] std::vector<Info> List() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Info> out;
    for (const auto& e : order_) {
      Info i;
      i.id = e->id;
      i.names = e->names;
      i.lane = e->spec.lane;
      i.kind = e->spec.kind;
      i.artifact = e->spec.artifact;
      i.endpoint = e->spec.endpoint;
      i.default_port = e->spec.default_port;
      i.estimated_resident_mib = e->spec.estimated_resident_mib;
      i.ready = e->ready;
      i.is_default = e->is_default;
      i.last_used_ms = e->last_used.load();
      out.push_back(std::move(i));
    }
    return out;
  }

  [[nodiscard]] bool HasId(std::string_view name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& e : order_) {
      if (e->id == name) return true;
    }
    return false;
  }

  [[nodiscard]] bool HasName(std::string_view name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = by_name_.find(std::string(name));
    return it != by_name_.end() && !it->second.expired();
  }

  /// Evictable sidecars, least-recently-used first, default excluded.
  [[nodiscard]] std::vector<std::string> LruCandidates() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t now = NowMs();
    std::vector<std::pair<std::uint64_t, std::string>> scored;
    for (const auto& e : order_) {
      if (e->is_default) continue;
      scored.emplace_back(now >= e->last_used.load()
                              ? now - e->last_used.load()
                              : 0,
                          e->id);
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::string> out;
    for (const auto& s : scored) out.push_back(s.second);
    return out;
  }

  [[nodiscard]] std::size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return order_.size();
  }

 private:
  struct Entry {
    std::string id;
    std::vector<std::string> names;
    NpuModelSpec spec;
    bool ready{false};
    bool is_default{false};
    std::atomic<std::uint64_t> last_used{0};
  };

  static std::uint64_t NowMs() noexcept {
    using Clock = std::chrono::steady_clock;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now().time_since_epoch())
            .count());
  }

  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<Entry>> order_;
  std::unordered_map<std::string, std::weak_ptr<Entry>> by_name_;
};

}  // namespace mbufo::core::npu
