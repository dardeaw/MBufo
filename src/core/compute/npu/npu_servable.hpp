#pragma once
// NPU servable contract (M1): one lifecycle for every NPU model, whether it is
// an in-process runtime (bge-m3, EmbeddingGemma 2, ASR/TTS/DiT decoders) or an
// HTTP sidecar (flm / gufo serve). Header-only, no hardware dependency, so the
// state machine is unit-testable on any host.
//
//   Unloaded -> Loading -> Warming -> Ready
//                  |          |
//                   +----------+--> Failed (error kept, retry via Load)
//
// `Ready` means "the first real request will not pay a cold-start cost":
// weights are resident AND one dummy inference (Warmup) has completed. The
// admin UI and router must gate dispatch on Ready, not on "process is up".

#include <atomic>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mbufo::core::npu {

enum class NpuServableState : std::uint8_t {
  kUnloaded = 0,
  kLoading,
  kWarming,
  kReady,
  kFailed,
};

[[nodiscard]] inline const char* NpuServableStateName(
    NpuServableState s) noexcept {
  switch (s) {
    case NpuServableState::kUnloaded:
      return "unloaded";
    case NpuServableState::kLoading:
      return "loading";
    case NpuServableState::kWarming:
      return "warming";
    case NpuServableState::kReady:
      return "ready";
    case NpuServableState::kFailed:
      return "failed";
  }
  return "unknown";
}

/// Snapshot of a servable's lifecycle, safe to copy into admin JSON.
struct NpuServableStatus {
  NpuServableState state{NpuServableState::kUnloaded};
  double load_ms{0.0};
  double warmup_ms{0.0};
  bool warmed{false};      ///< Warmup completed successfully at least once.
  std::string last_error;  ///< Non-empty only in kFailed (or a soft warm miss).
};

/// Thread-safe state holder embedded by every servable implementation.
class NpuServableTracker {
public:
  [[nodiscard]] NpuServableStatus Snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    return status_;
  }
  [[nodiscard]] NpuServableState state() const {
    std::lock_guard<std::mutex> lock(mu_);
    return status_.state;
  }
  [[nodiscard]] bool ready() const {
    return state() == NpuServableState::kReady;
  }

  void Set(NpuServableState s) {
    std::lock_guard<std::mutex> lock(mu_);
    status_.state = s;
    if (s == NpuServableState::kLoading) {
      status_.last_error.clear();
      status_.warmed = false;
      status_.load_ms = status_.warmup_ms = 0.0;
    }
  }
  void SetLoadMs(double ms) {
    std::lock_guard<std::mutex> lock(mu_);
    status_.load_ms = ms;
  }
  void SetWarmed(double ms) {
    std::lock_guard<std::mutex> lock(mu_);
    status_.warmed = true;
    status_.warmup_ms = ms;
  }
  void Fail(std::string error) {
    std::lock_guard<std::mutex> lock(mu_);
    status_.state = NpuServableState::kFailed;
    status_.last_error = std::move(error);
  }
  void SoftWarmMiss(std::string error) {
    std::lock_guard<std::mutex> lock(mu_);
    status_.last_error = std::move(error);
  }
  void Reset() {
    std::lock_guard<std::mutex> lock(mu_);
    status_ = NpuServableStatus{};
  }

private:
  mutable std::mutex mu_;
  NpuServableStatus status_;
};

/// What to do when Warmup fails after a successful Load.
enum class NpuWarmupPolicy : std::uint8_t {
  kRequired,   ///< Warmup failure => kFailed (model never reports Ready).
  kBestEffort  ///< Warmup failure => still Ready, `warmed=false`, error kept.
};

struct NpuLoadArgs {
  std::string artifact;  ///< weights dir / model path / sidecar spec id
  std::string quant;     ///< "q8" | "q4" | "" (model default)
  NpuWarmupPolicy warmup_policy{NpuWarmupPolicy::kRequired};
  bool warmup{true};  ///< false = Load only (legacy behaviour)
};

/// The single contract. Implementations only fill in Open/Warm/Close; the
/// lifecycle (timing, state transitions, error capture) lives in the base so
/// every servable behaves identically.
class INpuServable {
public:
  virtual ~INpuServable() = default;

  [[nodiscard]] virtual std::string_view id() const noexcept = 0;

  /// Load + (optional) warmup under one state machine. Idempotent: calling it
  /// on an already-Ready servable is a no-op success.
  [[nodiscard]] bool Load(const NpuLoadArgs& args, std::string* error) {
    if (tracker_.ready())
      return true;
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::time_point a, Clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    tracker_.Set(NpuServableState::kLoading);
    const auto t0 = Clock::now();
    std::string err;
    if (!Open(args, &err)) {
      tracker_.Fail(err.empty() ? "open failed" : err);
      if (error != nullptr)
        *error = tracker_.Snapshot().last_error;
      return false;
    }
    const auto t1 = Clock::now();
    tracker_.SetLoadMs(ms(t0, t1));
    if (args.warmup) {
      tracker_.Set(NpuServableState::kWarming);
      err.clear();
      if (!Warm(&err)) {
        const std::string msg = err.empty() ? "warmup failed" : err;
        if (args.warmup_policy == NpuWarmupPolicy::kRequired) {
          Close();
          tracker_.Fail("warmup: " + msg);
          if (error != nullptr)
            *error = tracker_.Snapshot().last_error;
          return false;
        }
        tracker_.SoftWarmMiss("warmup: " + msg);
      } else {
        tracker_.SetWarmed(ms(t1, Clock::now()));
      }
    }
    tracker_.Set(NpuServableState::kReady);
    return true;
  }

  /// Re-run the warmup of an already-open servable (e.g. after the NPU was
  /// held by another process). Ready is kept on failure only when the
  /// servable was already warmed once.
  [[nodiscard]] bool Rewarm(std::string* error) {
    const auto st = tracker_.state();
    if (st != NpuServableState::kReady && st != NpuServableState::kFailed) {
      if (error != nullptr)
        *error = "servable is not open";
      return false;
    }
    const auto t = std::chrono::steady_clock::now();
    std::string err;
    if (!Warm(&err)) {
      tracker_.SoftWarmMiss("warmup: " + (err.empty() ? "failed" : err));
      if (error != nullptr)
        *error = tracker_.Snapshot().last_error;
      return false;
    }
    tracker_.SetWarmed(std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t)
                           .count());
    tracker_.Set(NpuServableState::kReady);
    return true;
  }

  /// Eject. Every servable must implement Close (pure virtual); Unload is the
  /// only public way to release one, and it always hands the freed heap back
  /// to the OS so an eject reliably returns RAM.
  void Unload() {
    Close();
    tracker_.Reset();
#if defined(__GLIBC__)
    ::malloc_trim(0);
#endif
  }

  [[nodiscard]] NpuServableStatus Status() const { return tracker_.Snapshot(); }
  [[nodiscard]] bool Ready() const { return tracker_.ready(); }

protected:
  /// Acquire resources (weights, xclbins, process). Must be repeat-safe.
  [[nodiscard]] virtual bool Open(const NpuLoadArgs& args,
                                  std::string* error) = 0;
  /// One cheap dummy inference exercising the real kernels/endpoint.
  [[nodiscard]] virtual bool Warm(std::string* error) = 0;
  /// Release resources. Must tolerate being called when not open.
  virtual void Close() = 0;

private:
  NpuServableTracker tracker_;
};

/// Adapter for runtimes whose lifetime is owned elsewhere (bge-m3,
/// bge-reranker, EmbeddingGemma 2 live inside their serve backends). The three
/// hooks are the existing load / dummy-inference / release code, so nothing is
/// re-implemented and the lifecycle + status still come from the shared
/// contract.
class NpuFnServable final : public INpuServable {
public:
  using OpenFn = std::function<bool(const NpuLoadArgs&, std::string*)>;
  using WarmFn = std::function<bool(std::string*)>;
  using CloseFn = std::function<void()>;

  NpuFnServable(std::string id, OpenFn open, WarmFn warm, CloseFn close)
      : id_(std::move(id)),
        open_(std::move(open)),
        warm_(std::move(warm)),
        close_(std::move(close)) {
    // Eject is mandatory: a servable that cannot release its weights is a bug.
    if (!close_) {
      throw std::invalid_argument("NpuFnServable '" + id_ +
                                  "' requires a Close implementation");
    }
  }

  [[nodiscard]] std::string_view id() const noexcept override { return id_; }

  /// Drop the open/warm hooks once the first Load finished. Callers whose
  /// hooks capture locals by reference must call this before returning, so a
  /// later Rewarm fails cleanly instead of touching dead stack objects.
  void DetachLoadHooks() {
    open_ = nullptr;
    warm_ = nullptr;
    hooks_detached_ = true;
  }

protected:
  bool Open(const NpuLoadArgs& a, std::string* e) override {
    return open_ ? open_(a, e) : true;
  }
  bool Warm(std::string* e) override {
    if (hooks_detached_) {
      if (e != nullptr)
        *e = "warmup hook is no longer available";
      return false;
    }
    return warm_ ? warm_(e) : true;
  }
  void Close() override {
    if (close_)
      close_();
  }

private:
  std::string id_;
  OpenFn open_;
  WarmFn warm_;
  CloseFn close_;
  bool hooks_detached_{false};
};

/// Process-wide view of every NPU servable so the admin API can report one
/// uniform status regardless of how the model is hosted.
class NpuServableRegistry {
public:
  /// Admin load/eject tag: HandleLoad sets it (same thread) around the model
  /// loader so servables are registered under the admin model id and can be
  /// dropped again on eject.
  static std::string& CurrentTag() {
    static thread_local std::string tag;
    return tag;
  }
  class TagScope {
  public:
    explicit TagScope(std::string tag) : prev_(CurrentTag()) {
      CurrentTag() = std::move(tag);
    }
    ~TagScope() { CurrentTag() = prev_; }
    TagScope(const TagScope&) = delete;
    TagScope& operator=(const TagScope&) = delete;

  private:
    std::string prev_;
  };

  static NpuServableRegistry& Global() {
    static NpuServableRegistry r;
    return r;
  }
  void Put(const std::string& id, std::shared_ptr<INpuServable> s) {
    std::lock_guard<std::mutex> lock(mu_);
    map_[id] = std::move(s);
  }
  /// Eject by key: runs the servable's Close (+ heap trim) first, then drops
  /// the entry. Must be called while the model object behind Close is alive.
  void Unload(const std::string& id) {
    std::shared_ptr<INpuServable> s;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = map_.find(id);
      if (it == map_.end())
        return;
      s = it->second;
      map_.erase(it);
    }
    s->Unload();
  }
  void Remove(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu_);
    map_.erase(id);
  }
  [[nodiscard]] std::vector<std::pair<std::string, NpuServableStatus>>
  Snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::pair<std::string, NpuServableStatus>> out;
    for (const auto& [k, v] : map_)
      out.emplace_back(k, v->Status());
    return out;
  }

private:
  mutable std::mutex mu_;
  std::map<std::string, std::shared_ptr<INpuServable>> map_;
};

}  // namespace mbufo::core::npu
