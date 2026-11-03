#ifndef GUFO_CORE_SPECULATIVE_DRAFT_REGISTRY_HPP_
#define GUFO_CORE_SPECULATIVE_DRAFT_REGISTRY_HPP_

// Unified speculative-draft backend registry.
//
// Motivation: each engine historically grew its own draft machinery
// (flash-next MTP inside Session/DecodeBatch, 27B DFlash2/MTP behind
// IDraftBackend, gemma4 MTP inside Session::SpeculativeRound, DeepSeek
// DSpark inside its engine). This registry is the single place where a
// draft backend implementation is published under a serve-facing name, so
// adding a future draft only requires implementing IDraftBackend (or a
// session-bound adapter over it) and registering a factory — no engine or
// serve plumbing changes.
//
// What this registry does NOT unify (deliberately model-side):
//   - draft weight loading and head architecture,
//   - the acceptance policy (greedy vs sampled rejection),
//   - the target verify loop (each engine keeps its own verify path; the
//     existing ISpeculativeTargetExecutor/SpeculativeVerifier already plays
//     the TargetVerify role for 27B, flash-next keeps DecodeBatchImpl, and
//     gemma4 keeps its single batched Forward).
//
// Draft KV policy: backends reuse their engine's existing KV store. The
// 27B Qwen path runs on the shared blocked KV layer; flash-next and
// gemma4 adapters operate on their live session KV. No new KV layout is
// introduced here.
//
// Namespace note: the task brief suggested gufo::core::speculative, but the
// pre-existing speculative headers (draft_backend.hpp,
// speculative_verifier.hpp, prompt_lookup.hpp) all live in
// gufo::speculative. This header follows the established convention.

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"

namespace gufo::speculative {

/// Creation context for a registered draft backend factory.
///
/// Backends fall into two binding styles:
///   - target-bound (e.g. 27B DFlash2/MTP): needs the shared target model.
///     The caller passes ownership-preserving `target_model` plus a
///     `target_kind` tag the factory validates (e.g. "qwen27b").
///   - session-bound (e.g. flash-next MTP, gemma4 MTP): the draft head
///     shares the live session KV, so it can only be built from a live
///     session. The caller passes `session_handle` plus `session_kind`
///     (e.g. "flash-next", "gemma4").
///
/// `draft_model_path` is the sidecar GGUF when the backend loads its own
/// weights (empty for embedded/session-bound heads). `max_draft_tokens`
/// follows the serve `--draft-tokens` value; 0 selects the backend default.
struct DraftBackendRequest {
  std::string draft_model_path;
  std::uint32_t max_draft_tokens{0};
  /// Session/KV context bound, 0 selects the backend default (qwen drafts
  /// fall back to the target model's native context length).
  std::uint32_t max_context{0};
  std::shared_ptr<const void> target_model;
  std::string target_kind;
  void* session_handle{nullptr};
  std::string session_kind;
};

/// Factory producing a draft backend from a request. Returns nullptr and
/// sets *error_msg on failure (unknown kind, missing handle, load error).
using DraftBackendFactory =
    std::function<std::unique_ptr<IDraftBackend>(const DraftBackendRequest&,
                                                 std::string* error_msg)>;

/// Static metadata published alongside each factory.
struct DraftBackendInfo {
  std::string name;
  std::string description;
  bool requires_target_model{false};
  bool requires_session{false};
};

/// Process-wide registry mapping serve-facing backend names
/// ("dflash2", "qwen-mtp", "flash-next-mtp", "gemma4-mtp", ...) to
/// factories. Thread-safe; re-registering a name replaces the entry.
class DraftBackendRegistry {
 public:
  static DraftBackendRegistry& Instance() {
    static DraftBackendRegistry instance;
    return instance;
  }

  DraftBackendRegistry(const DraftBackendRegistry&) = delete;
  DraftBackendRegistry& operator=(const DraftBackendRegistry&) = delete;

  void Register(DraftBackendInfo info, DraftBackendFactory factory) {
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

  [[nodiscard]] std::vector<DraftBackendInfo> List() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DraftBackendInfo> out;
    out.reserve(entries_.size());
    for (const auto& [name, entry] : entries_) {
      (void)name;
      out.push_back(entry.info);
    }
    return out;
  }

  [[nodiscard]] std::unique_ptr<IDraftBackend> Create(
      std::string_view name, const DraftBackendRequest& request,
      std::string* error_msg) const {
    DraftBackendFactory factory;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = entries_.find(std::string(name));
      if (it == entries_.end()) {
        if (error_msg != nullptr) {
          *error_msg = "unknown draft backend: " + std::string(name);
        }
        return nullptr;
      }
      factory = it->second.factory;
    }
    DraftBackendRequest named = request;
    return factory(named, error_msg);
  }

 private:
  DraftBackendRegistry() = default;

  struct Entry {
    DraftBackendInfo info;
    DraftBackendFactory factory;
  };

  mutable std::mutex mutex_;
  std::unordered_map<std::string, Entry> entries_;
};

/// Convenience wrappers over DraftBackendRegistry::Instance().
inline void RegisterDraftBackend(DraftBackendInfo info,
                                 DraftBackendFactory factory) {
  DraftBackendRegistry::Instance().Register(std::move(info),
                                            std::move(factory));
}

inline std::unique_ptr<IDraftBackend> CreateDraftBackend(
    std::string_view name, const DraftBackendRequest& request,
    std::string* error_msg = nullptr) {
  return DraftBackendRegistry::Instance().Create(name, request, error_msg);
}

inline std::vector<DraftBackendInfo> ListDraftBackends() {
  return DraftBackendRegistry::Instance().List();
}

/// RAII helper: constructing a Registrar registers a backend; the typical
/// use is a file-scope static in the backend's own translation unit, so a
/// new draft only touches its own files.
class DraftBackendRegistrar {
 public:
  DraftBackendRegistrar(DraftBackendInfo info, DraftBackendFactory factory) {
    RegisterDraftBackend(std::move(info), std::move(factory));
  }
};

}  // namespace gufo::speculative

#endif  // GUFO_CORE_SPECULATIVE_DRAFT_REGISTRY_HPP_
