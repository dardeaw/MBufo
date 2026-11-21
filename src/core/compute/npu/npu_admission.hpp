#pragma once
// Generic admission-queue primitive (scaffold v0, header-only, no wiring).
//
// Purpose: the half of TextGenerationScheduler that has nothing to do with
// tokens — bounded queue, per-client caps, timeouts, cancel, device-lost —
// extracted so the text path and the future NPU scheduler share one.
// Single-owner pump model (same contract as the text scheduler):
// submitters block in Wait(); exactly one owner thread advances tickets
// with AdmitNext()/Finish(). Nothing in the repo calls this yet.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

namespace mbufo::core {

enum class NpuRequestPhase : std::uint8_t {
  kQueued = 0,
  kAdmitted,
  kRunning,
  kTerminal,
};

/// Mirrors TextSchedulerPolicy knobs, minus everything token-shaped.
struct NpuAdmissionPolicy {
  std::size_t max_pending{16};
  std::size_t max_pending_per_client{4};
  /// Default per-request timeout when Submit() omits one. Zero = wait forever.
  std::chrono::milliseconds request_timeout{0};
};

/// Token/work-shape agnostic admission queue. Thread-safe; owner-pumped.
class NpuAdmissionQueue {
 public:
  struct WaitResult {
    bool admitted{false};
    std::string error;
  };

  explicit NpuAdmissionQueue(NpuAdmissionPolicy policy = {}) noexcept
      : policy_(policy), next_id_(1) {}

  NpuAdmissionQueue(const NpuAdmissionQueue&) = delete;
  NpuAdmissionQueue& operator=(const NpuAdmissionQueue&) = delete;

  /// Enqueue. Returns ticket id (>0), or 0 when over capacity / device lost.
  [[nodiscard]] std::uint64_t Submit(
      std::string client,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) {
    std::unique_lock<std::mutex> lock(mu_);
    if (device_lost_) return 0;
    std::size_t pending = 0, mine = 0;
    for (const auto& e : entries_) {
      if (e.phase == NpuRequestPhase::kTerminal) continue;
      ++pending;
      if (e.client == client) ++mine;
    }
    if (pending >= policy_.max_pending) return 0;
    if (mine >= policy_.max_pending_per_client) return 0;
    Entry e;
    e.id = next_id_++;
    e.client = std::move(client);
    e.phase = NpuRequestPhase::kQueued;
    e.enqueued = Clock::now();
    e.timeout = timeout.count() != 0 ? timeout : policy_.request_timeout;
    entries_.push_back(std::move(e));
    cv_.notify_all();
    return entries_.back().id;
  }

  /// Block until the owner admits this ticket, or timeout/cancel/device-lost.
  [[nodiscard]] WaitResult Wait(std::uint64_t ticket) {
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
      if (device_lost_) return {false, "device-lost"};
      Entry* found = Find(ticket);
      if (found == nullptr) return {false, "unknown-ticket"};
      if (found->cancelled) {
        found->phase = NpuRequestPhase::kTerminal;
        return {false, "cancelled"};
      }
      if (found->phase == NpuRequestPhase::kAdmitted ||
          found->phase == NpuRequestPhase::kRunning) {
        return {true, {}};
      }
      if (found->timeout.count() != 0 &&
          Clock::now() - found->enqueued >= found->timeout) {
        found->phase = NpuRequestPhase::kTerminal;
        return {false, "timeout"};
      }
      if (found->timeout.count() != 0) {
        cv_.wait_until(lock, found->enqueued + found->timeout);
      } else {
        cv_.wait(lock);
      }
    }
  }

  void Cancel(std::uint64_t ticket) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    if (Entry* found = Find(ticket); found != nullptr) {
      found->cancelled = true;
    }
    cv_.notify_all();
  }

  // ---- owner side (exactly one pumping thread) ----

  /// Admit the oldest queued, non-cancelled ticket. 0 when the queue is empty.
  [[nodiscard]] std::uint64_t AdmitNext() {
    std::lock_guard<std::mutex> lock(mu_);
    if (device_lost_) return 0;
    for (auto& e : entries_) {
      if (e.phase != NpuRequestPhase::kQueued || e.cancelled) continue;
      e.phase = NpuRequestPhase::kAdmitted;
      cv_.notify_all();
      return e.id;
    }
    return 0;
  }

  /// Mark a ticket running (owner holds the NPU device lock from here).
  void MarkRunning(std::uint64_t ticket) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    if (Entry* found = Find(ticket); found != nullptr) {
      found->phase = NpuRequestPhase::kRunning;
    }
    cv_.notify_all();
  }

  /// Terminal + reclaim. Late Wait() calls observe unknown-ticket.
  void Finish(std::uint64_t ticket) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->id == ticket) {
        entries_.erase(it);
        break;
      }
    }
    cv_.notify_all();
  }

  void SetDeviceLost() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    device_lost_ = true;
    cv_.notify_all();
  }

  [[nodiscard]] bool device_lost() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return device_lost_;
  }

  [[nodiscard]] std::size_t pending() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::size_t n = 0;
    for (const auto& e : entries_) {
      if (e.phase != NpuRequestPhase::kTerminal) ++n;
    }
    return n;
  }

 private:
  using Clock = std::chrono::steady_clock;
  struct Entry {
    std::uint64_t id{0};
    std::string client;
    NpuRequestPhase phase{NpuRequestPhase::kQueued};
    bool cancelled{false};
    Clock::time_point enqueued{};
    std::chrono::milliseconds timeout{0};
  };

  Entry* Find(std::uint64_t ticket) noexcept {
    for (auto& e : entries_) {
      if (e.id == ticket) return &e;
    }
    return nullptr;
  }

  NpuAdmissionPolicy policy_;
  std::uint64_t next_id_;
  bool device_lost_{false};
  std::deque<Entry> entries_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace mbufo::core
