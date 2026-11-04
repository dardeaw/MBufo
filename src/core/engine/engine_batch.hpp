#ifndef GUFO_CORE_ENGINE_ENGINE_BATCH_HPP_
#define GUFO_CORE_ENGINE_ENGINE_BATCH_HPP_

// L1 engine orchestration: speculative stepping, optional capabilities, and
// the engine factory registry.
//
// The registry mirrors speculative::DraftBackendRegistry (same structure:
// Info + Factory + process-wide Instance + file-scope Registrar), so adding
// a future engine only touches its own adapter files.

#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/engine/engine_backend.hpp"

namespace gufo::core::engine {

/// Result of one SpeculativeSession::DecodeStep.
struct DecodeOutcome {
  std::int32_t token = 0;          ///< Sampled next token (already committed).
  bool stop = false;               ///< The token is a stop/EOS token.
  std::uint32_t tokens_committed{1};  ///< Tokens committed (1 for plain AR).
};

/// An EngineSession with a uniform one-step decode entry point. Engines
/// without speculative machinery report SupportsSpeculative() == false and
/// implement DecodeStep as a plain greedy single-token step (sample the
/// current logits, Evaluate it); callers can drive every engine through
/// DecodeStep without branching on engine kind.
class SpeculativeSession : public EngineSession {
 public:
  [[nodiscard]] virtual bool SupportsSpeculative() const noexcept = 0;
  /// Runs one decode step and commits its token(s). On success *outcome
  /// holds the sampled next token; the session has already advanced past it.
  /// `max_draft_tokens` is a hint for speculative engines (0 = engine
  /// default); plain-fallback engines ignore it.
  [[nodiscard]] virtual bool DecodeStep(std::uint32_t max_draft_tokens,
                                        DecodeOutcome* outcome,
                                        std::string* error_msg = nullptr) = 0;
};

/// Optional capability: the engine can consume encoded images alongside
/// text. Hosts probe with dynamic_cast; engines without vision simply do
/// not derive from this.
class VisionCapable {
 public:
  virtual ~VisionCapable() = default;
  [[nodiscard]] virtual bool SupportsVision() const noexcept = 0;
  /// Attaches one encoded (e.g. PNG/JPEG) image to the live session.
  [[nodiscard]] virtual bool AttachImage(
      std::span<const std::uint8_t> encoded_image,
      std::string* error_msg = nullptr) = 0;
};

/// Optional capability: the engine can snapshot/restore its live KV state
/// into a caller-owned byte buffer. Hosts probe with dynamic_cast.
class SnapshotCapable {
 public:
  virtual ~SnapshotCapable() = default;
  /// Exact payload bytes that Snapshot() writes at the current boundary.
  [[nodiscard]] virtual std::size_t SnapshotPayloadBytes() const = 0;
  /// Serializes the live state into an exactly-sized destination.
  [[nodiscard]] virtual bool Snapshot(std::span<std::uint8_t> destination,
                                      std::string* error_msg = nullptr) const = 0;
  /// Restores a payload previously produced by Snapshot().
  [[nodiscard]] virtual bool RestoreSnapshot(
      std::span<const std::uint8_t> payload,
      std::string* error_msg = nullptr) = 0;
};

/// Creation context for a registered engine factory. Zero/empty fields
/// select the engine default (e.g. max_context 0 keeps the native default).
struct EngineLoadRequest {
  std::string model_path;
  std::uint32_t max_context{0};
  std::string kv_cache_quant;
};

/// Factory producing an engine model from a request. Returns nullptr and
/// sets *error_msg on failure (missing file, bad options, no device).
using EngineModelFactory =
    std::function<std::shared_ptr<EngineModel>(const EngineLoadRequest&,
                                               std::string* error_msg)>;

/// Static metadata published alongside each factory.
struct EngineBackendInfo {
  std::string name;
  std::string description;
};

/// Process-wide registry mapping serve-facing engine names ("llama", ...)
/// to factories. Thread-safe; re-registering a name replaces the entry.
class EngineRegistry {
 public:
  static EngineRegistry& Instance();

  EngineRegistry(const EngineRegistry&) = delete;
  EngineRegistry& operator=(const EngineRegistry&) = delete;

  void Register(EngineBackendInfo info, EngineModelFactory factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    // NOTE: key copy must precede the move below. In
    // `map[key_expr] = Entry{std::move(info), ...}` the right-hand side is
    // sequenced before the left, so the key would copy a moved-from name.
    const std::string key = info.name;
    entries_[key] = Entry{std::move(info), std::move(factory)};
  }

  [[nodiscard]] bool Contains(std::string_view name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.find(std::string(name)) != entries_.end();
  }

  [[nodiscard]] std::vector<EngineBackendInfo> List() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<EngineBackendInfo> out;
    out.reserve(entries_.size());
    for (const auto& [name, entry] : entries_) {
      (void)name;
      out.push_back(entry.info);
    }
    return out;
  }

  [[nodiscard]] std::shared_ptr<EngineModel> Create(
      std::string_view name, const EngineLoadRequest& request,
      std::string* error_msg) const {
    EngineModelFactory factory;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = entries_.find(std::string(name));
      if (it == entries_.end()) {
        if (error_msg != nullptr) {
          *error_msg = "unknown engine backend: " + std::string(name);
        }
        return nullptr;
      }
      factory = it->second.factory;
    }
    EngineLoadRequest named = request;
    return factory(named, error_msg);
  }

 private:
  EngineRegistry() = default;

  struct Entry {
    EngineBackendInfo info;
    EngineModelFactory factory;
  };

  mutable std::mutex mutex_;
  std::unordered_map<std::string, Entry> entries_;
};

/// Convenience wrappers over EngineRegistry::Instance().
inline void RegisterEngineBackend(EngineBackendInfo info,
                                   EngineModelFactory factory) {
  EngineRegistry::Instance().Register(std::move(info), std::move(factory));
}

inline std::shared_ptr<EngineModel> CreateEngineModel(
    std::string_view name, const EngineLoadRequest& request,
    std::string* error_msg = nullptr) {
  return EngineRegistry::Instance().Create(name, request, error_msg);
}

inline std::vector<EngineBackendInfo> ListEngineBackends() {
  return EngineRegistry::Instance().List();
}

/// RAII helper: constructing a Registrar registers a backend; the typical
/// use is a file-scope static in the backend's own translation unit, so a
/// new engine only touches its own files.
class EngineRegistrar {
 public:
  EngineRegistrar(EngineBackendInfo info, EngineModelFactory factory) {
    RegisterEngineBackend(std::move(info), std::move(factory));
  }
};

}  // namespace gufo::core::engine

#endif  // GUFO_CORE_ENGINE_ENGINE_BATCH_HPP_
