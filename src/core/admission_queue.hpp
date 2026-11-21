#pragma once
// Shared bounded admission-queue primitive (header-only).
//
// Purpose: the scheduling half common to TextGenerationScheduler and
// NpuScheduler — global/per-client bounded queue, round-robin client fairness,
// per-request timeouts/deadlines, single-ticket and drain-all cancellation,
// wait/admit/running/terminal lifecycle, and sticky device-lost circuit
// breaking.
//
// Phase machines for model execution (TextRequestPhase prefill/decode/MTP vs.
// NPU one-shot run-to-completion / coalesced Sync) stay strictly inside their
// respective schedulers.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace mbufo::core {

enum class AdmissionPhase : std::uint8_t {
  kQueued = 0,
  kAdmitted,
  kRunning,
  kTerminal,
};

enum class AdmissionRejectReason : std::uint8_t {
  kNone = 0,
  kStopping,
  kDeviceLost,
  kQueueFull,
  kClientQueueFull,
  kCancelled,
  kTimeout,
  kUnknownTicket,
};

struct AdmissionPolicy {
  std::size_t max_pending{16};
  std::size_t max_pending_per_client{4};
  /// Default per-request timeout when Submit omits one. Zero = wait forever.
  std::chrono::milliseconds request_timeout{0};
};

struct AdmissionSubmitResult {
  std::uint64_t ticket{0};
  AdmissionRejectReason reject{AdmissionRejectReason::kNone};
  std::size_t queued_depth{0};
  std::size_t client_queued_depth{0};
};

struct AdmissionWaitResult {
  bool admitted{false};
  AdmissionRejectReason reason{AdmissionRejectReason::kNone};
  std::string error;
};

template <typename Payload = std::monostate>
class AdmissionQueue {
 public:
  using Clock = std::chrono::steady_clock;

  struct DroppedQueued {
    std::uint64_t ticket{0};
    Payload payload{};
    AdmissionRejectReason reason{AdmissionRejectReason::kCancelled};
  };

  explicit AdmissionQueue(AdmissionPolicy policy = {}) noexcept
      : policy_(policy), next_id_(1) {}

  AdmissionQueue(const AdmissionQueue&) = delete;
  AdmissionQueue& operator=(const AdmissionQueue&) = delete;

  /// Installs a delta hook invoked under the queue mutex whenever the queued
  /// entry count changes (`delta` in {-k, +1}). Useful for process gauges.
  void SetQueuedDeltaCallback(std::function<void(std::int64_t)> cb) {
    std::lock_guard<std::mutex> lock(mu_);
    on_queued_delta_ = std::move(cb);
  }

  /// Enqueue `payload` for `client`. Empty `client` normalizes to "anonymous".
  /// `on_accepted(ticket, queued_depth, client_queued_depth)` runs under the
  /// mutex before waking waiters/workers so ticket metadata can be published
  /// atomically before the owner thread observes the item.
  [[nodiscard]] AdmissionSubmitResult SubmitPayload(
      std::string client, Payload payload,
      std::optional<Clock::time_point> deadline = std::nullopt,
      const std::function<void(std::uint64_t, std::size_t, std::size_t)>&
          on_accepted = {}) {
    if (client.empty()) client = "anonymous";
    AdmissionSubmitResult res{};
    {
      std::lock_guard<std::mutex> lock(mu_);
      const std::size_t q = queued_count_.load(std::memory_order_relaxed);
      if (stopping_.load(std::memory_order_relaxed)) {
        res.reject = AdmissionRejectReason::kStopping;
        res.queued_depth = q;
        return res;
      }
      if (device_lost_.load(std::memory_order_relaxed)) {
        res.reject = AdmissionRejectReason::kDeviceLost;
        res.queued_depth = q;
        return res;
      }
      if (q >= policy_.max_pending) {
        res.reject = AdmissionRejectReason::kQueueFull;
        res.queued_depth = q;
        return res;
      }
      ClientBucket* bucket = FindBucketLocked(client);
      const std::size_t cq = (bucket != nullptr) ? bucket->tickets.size() : 0u;
      if (cq >= policy_.max_pending_per_client) {
        res.reject = AdmissionRejectReason::kClientQueueFull;
        res.queued_depth = q;
        res.client_queued_depth = cq;
        return res;
      }
      if (bucket == nullptr) {
        client_buckets_.push_back(ClientBucket{client, {}});
        bucket = &client_buckets_.back();
      }
      const std::uint64_t ticket = next_id_++;
      const Clock::time_point now = Clock::now();
      std::optional<Clock::time_point> eff_deadline = deadline;
      if (!eff_deadline.has_value() && policy_.request_timeout.count() > 0) {
        eff_deadline = now + policy_.request_timeout;
      }

      Entry e;
      e.id = ticket;
      e.client = std::move(client);
      e.phase = AdmissionPhase::kQueued;
      e.cancelled = false;
      e.enqueued = now;
      e.deadline = eff_deadline;
      e.payload = std::move(payload);

      res.ticket = ticket;
      res.reject = AdmissionRejectReason::kNone;
      res.queued_depth = q + 1u;
      res.client_queued_depth = cq + 1u;

      if (on_accepted) {
        on_accepted(ticket, res.queued_depth, res.client_queued_depth);
      }

      entries_.emplace(ticket, std::move(e));
      bucket->tickets.push_back(ticket);
      AdjustQueuedLocked(+1);
    }
    cv_.notify_all();
    return res;
  }

  [[nodiscard]] AdmissionSubmitResult SubmitWithTimeout(
      std::string client, Payload payload,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{0},
      const std::function<void(std::uint64_t, std::size_t, std::size_t)>&
          on_accepted = {}) {
    std::optional<Clock::time_point> dl;
    if (timeout.count() > 0) dl = Clock::now() + timeout;
    return SubmitPayload(std::move(client), std::move(payload), dl,
                         on_accepted);
  }

  /// Backward-compatible Submit for default-constructible / monostate queues.
  template <typename P = Payload,
            typename = std::enable_if_t<std::is_default_constructible_v<P>>>
  [[nodiscard]] std::uint64_t Submit(
      std::string client,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) {
    return SubmitWithTimeout(std::move(client), Payload{}, timeout).ticket;
  }

  /// Block until `ticket` is admitted/running, or cancelled/timed-out/stopped/lost.
  [[nodiscard]] AdmissionWaitResult Wait(std::uint64_t ticket) {
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
      if (device_lost_.load(std::memory_order_relaxed)) {
        return {false, AdmissionRejectReason::kDeviceLost, "device-lost"};
      }
      auto it = entries_.find(ticket);
      if (it == entries_.end()) {
        return {false, AdmissionRejectReason::kUnknownTicket, "unknown-ticket"};
      }
      Entry& e = it->second;
      if (e.cancelled) {
        if (e.phase == AdmissionPhase::kQueued) {
          RemoveFromBucketLocked(ticket, e.client);
          AdjustQueuedLocked(-1);
        }
        e.phase = AdmissionPhase::kTerminal;
        return {false, AdmissionRejectReason::kCancelled, "cancelled"};
      }
      if (e.phase == AdmissionPhase::kAdmitted ||
          e.phase == AdmissionPhase::kRunning) {
        return {true, AdmissionRejectReason::kNone, {}};
      }
      if (e.phase == AdmissionPhase::kTerminal) {
        return {false, AdmissionRejectReason::kCancelled, "cancelled"};
      }
      if (stopping_.load(std::memory_order_relaxed)) {
        return {false, AdmissionRejectReason::kStopping, "stopping"};
      }
      if (e.deadline.has_value() && Clock::now() >= *e.deadline) {
        RemoveFromBucketLocked(ticket, e.client);
        AdjustQueuedLocked(-1);
        e.phase = AdmissionPhase::kTerminal;
        return {false, AdmissionRejectReason::kTimeout, "timeout"};
      }
      if (e.deadline.has_value()) {
        cv_.wait_until(lock, *e.deadline);
      } else {
        cv_.wait(lock);
      }
    }
  }

  /// Marks `ticket` cancelled (and optionally removes it immediately if queued).
  bool Cancel(std::uint64_t ticket, bool remove_if_queued = false) noexcept {
    bool found_any = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = entries_.find(ticket);
      if (it != entries_.end() && it->second.phase != AdmissionPhase::kTerminal) {
        it->second.cancelled = true;
        found_any = true;
        if (remove_if_queued && it->second.phase == AdmissionPhase::kQueued) {
          RemoveFromBucketLocked(ticket, it->second.client);
          AdjustQueuedLocked(-1);
          it->second.phase = AdmissionPhase::kTerminal;
        }
      }
    }
    if (found_any) cv_.notify_all();
    return found_any;
  }

  /// Drains all currently queued items in round-robin client order and marks
  /// them terminal+cancelled. Returns drained `(ticket, payload)` pairs.
  [[nodiscard]] std::vector<std::pair<std::uint64_t, Payload>> CancelAll() {
    std::vector<std::pair<std::uint64_t, Payload>> drained;
    {
      std::lock_guard<std::mutex> lock(mu_);
      std::int64_t removed = 0;
      for (auto& bucket : client_buckets_) {
        for (std::uint64_t tid : bucket.tickets) {
          auto it = entries_.find(tid);
          if (it != entries_.end()) {
            it->second.cancelled = true;
            it->second.phase = AdmissionPhase::kTerminal;
            drained.emplace_back(tid, std::move(it->second.payload));
            entries_.erase(it);
            --removed;
          }
        }
      }
      client_buckets_.clear();
      if (removed != 0) AdjustQueuedLocked(removed);
    }
    cv_.notify_all();
    return drained;
  }

  // ---- owner / pump side ----

  /// Pops and admits the oldest eligible queued ticket using round-robin client
  /// fairness. When `skip_cancelled_or_expired` is true, cancelled/expired
  /// queued entries are moved to kTerminal and skipped.
  [[nodiscard]] std::optional<std::pair<std::uint64_t, Payload>>
  AdmitNextPayload(
      const std::function<void(const Payload&)>& before_dequeue = {},
      bool skip_cancelled_or_expired = true,
      std::vector<DroppedQueued>* dropped_out = nullptr) {
    std::optional<std::pair<std::uint64_t, Payload>> out;
    bool state_changed = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (device_lost_.load(std::memory_order_relaxed)) return std::nullopt;
      const Clock::time_point now = Clock::now();
      while (!client_buckets_.empty()) {
        ClientBucket bucket = std::move(client_buckets_.front());
        client_buckets_.pop_front();
        while (!bucket.tickets.empty()) {
          const std::uint64_t tid = bucket.tickets.front();
          bucket.tickets.pop_front();
          auto it = entries_.find(tid);
          if (it == entries_.end() || it->second.phase != AdmissionPhase::kQueued) {
            continue;
          }
          const bool expired = it->second.deadline.has_value() &&
                               now >= *it->second.deadline;
          if (skip_cancelled_or_expired && (it->second.cancelled || expired)) {
            if (dropped_out != nullptr) {
              DroppedQueued dq;
              dq.ticket = tid;
              dq.payload = std::move(it->second.payload);
              dq.reason = it->second.cancelled
                              ? AdmissionRejectReason::kCancelled
                              : AdmissionRejectReason::kTimeout;
              dropped_out->push_back(std::move(dq));
              entries_.erase(it);
            } else {
              it->second.phase = AdmissionPhase::kTerminal;
            }
            AdjustQueuedLocked(-1);
            state_changed = true;
            continue;
          }
          if (!bucket.tickets.empty()) {
            client_buckets_.push_back(std::move(bucket));
          }
          if (before_dequeue) {
            before_dequeue(it->second.payload);
          }
          it->second.phase = AdmissionPhase::kAdmitted;
          AdjustQueuedLocked(-1);
          active_count_.fetch_add(1, std::memory_order_relaxed);
          out.emplace(tid, it->second.payload);
          state_changed = true;
          break;
        }
        if (out.has_value()) break;
      }
    }
    if (state_changed) cv_.notify_all();
    return out;
  }

  /// Convenience ticket-only admit for void/monostate queues.
  [[nodiscard]] std::uint64_t AdmitNext() {
    auto res = AdmitNextPayload({}, true);
    return res.has_value() ? res->first : 0u;
  }

  /// Admits up to `max_batch` queued items whose payloads satisfy
  /// `compatible(leader_payload, candidate_payload)`.
  template <typename Pred>
  [[nodiscard]] std::vector<std::pair<std::uint64_t, Payload>>
  AdmitCompatibleBatch(std::size_t max_batch, Pred&& compatible,
                       std::vector<DroppedQueued>* dropped_out = nullptr) {
    std::vector<std::pair<std::uint64_t, Payload>> batch;
    if (max_batch == 0) max_batch = 1;
    auto first = AdmitNextPayload({}, true, dropped_out);
    if (!first.has_value()) return batch;
    batch.push_back(std::move(*first));
    if (max_batch == 1) return batch;

    bool changed = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!device_lost_.load(std::memory_order_relaxed)) {
        const Clock::time_point now = Clock::now();
        for (auto bit = client_buckets_.begin();
             bit != client_buckets_.end() && batch.size() < max_batch;) {
          for (auto tit = bit->tickets.begin();
               tit != bit->tickets.end() && batch.size() < max_batch;) {
            const std::uint64_t tid = *tit;
            auto it = entries_.find(tid);
            if (it == entries_.end() ||
                it->second.phase != AdmissionPhase::kQueued) {
              tit = bit->tickets.erase(tit);
              continue;
            }
            const bool expired = it->second.deadline.has_value() &&
                                 now >= *it->second.deadline;
            if (it->second.cancelled || expired) {
              if (dropped_out != nullptr) {
                DroppedQueued dq;
                dq.ticket = tid;
                dq.payload = std::move(it->second.payload);
                dq.reason = it->second.cancelled
                                ? AdmissionRejectReason::kCancelled
                                : AdmissionRejectReason::kTimeout;
                dropped_out->push_back(std::move(dq));
                entries_.erase(it);
              } else {
                it->second.phase = AdmissionPhase::kTerminal;
              }
              tit = bit->tickets.erase(tit);
              AdjustQueuedLocked(-1);
              changed = true;
              continue;
            }
            if (compatible(batch.front().second, it->second.payload)) {
              it->second.phase = AdmissionPhase::kAdmitted;
              tit = bit->tickets.erase(tit);
              AdjustQueuedLocked(-1);
              active_count_.fetch_add(1, std::memory_order_relaxed);
              batch.emplace_back(tid, it->second.payload);
              changed = true;
              continue;
            }
            ++tit;
          }
          if (bit->tickets.empty()) {
            bit = client_buckets_.erase(bit);
          } else {
            ++bit;
          }
        }
      }
    }
    if (changed) cv_.notify_all();
    return batch;
  }

  /// Transition an admitted ticket to running.
  void MarkRunning(std::uint64_t ticket) noexcept {
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = entries_.find(ticket);
      if (it != entries_.end() && it->second.phase != AdmissionPhase::kTerminal) {
        it->second.phase = AdmissionPhase::kRunning;
      }
    }
    cv_.notify_all();
  }

  /// Mark ticket terminal and reclaim its storage safely.
  void Finish(std::uint64_t ticket) noexcept {
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = entries_.find(ticket);
      if (it != entries_.end()) {
        if (it->second.phase == AdmissionPhase::kQueued) {
          RemoveFromBucketLocked(ticket, it->second.client);
          AdjustQueuedLocked(-1);
        } else if (it->second.phase == AdmissionPhase::kAdmitted ||
                   it->second.phase == AdmissionPhase::kRunning) {
          const std::size_t cur = active_count_.load(std::memory_order_relaxed);
          if (cur > 0) active_count_.fetch_sub(1, std::memory_order_relaxed);
        }
        entries_.erase(it);
      }
    }
    cv_.notify_all();
  }

  /// Removes a specific queued ticket cleanly, returning its payload if found.
  [[nodiscard]] std::optional<Payload> RemoveQueued(std::uint64_t ticket) {
    std::optional<Payload> removed_payload;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = entries_.find(ticket);
      if (it != entries_.end() && it->second.phase == AdmissionPhase::kQueued) {
        RemoveFromBucketLocked(ticket, it->second.client);
        AdjustQueuedLocked(-1);
        removed_payload.emplace(std::move(it->second.payload));
        entries_.erase(it);
      }
    }
    if (removed_payload.has_value()) cv_.notify_all();
    return removed_payload;
  }

  /// Returns a snapshot of queued `(ticket, payload)` pairs in client-bucket order.
  [[nodiscard]] std::vector<std::pair<std::uint64_t, Payload>> QueuedSnapshot()
      const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::pair<std::uint64_t, Payload>> snap;
    snap.reserve(queued_count_.load(std::memory_order_relaxed));
    for (const auto& bucket : client_buckets_) {
      for (std::uint64_t tid : bucket.tickets) {
        auto it = entries_.find(tid);
        if (it != entries_.end() && it->second.phase == AdmissionPhase::kQueued) {
          snap.emplace_back(tid, it->second.payload);
        }
      }
    }
    return snap;
  }

  /// Removes and returns all queued entries that have been cancelled or whose
  /// deadline has expired.
  [[nodiscard]] std::vector<DroppedQueued> DrainCancelledOrExpired() {
    std::vector<DroppedQueued> dropped;
    {
      std::lock_guard<std::mutex> lock(mu_);
      const Clock::time_point now = Clock::now();
      std::int64_t delta = 0;
      for (auto bit = client_buckets_.begin(); bit != client_buckets_.end();) {
        for (auto tit = bit->tickets.begin(); tit != bit->tickets.end();) {
          auto it = entries_.find(*tit);
          if (it == entries_.end() ||
              it->second.phase != AdmissionPhase::kQueued) {
            tit = bit->tickets.erase(tit);
            continue;
          }
          const bool expired = it->second.deadline.has_value() &&
                               now >= *it->second.deadline;
          if (it->second.cancelled || expired) {
            DroppedQueued dq;
            dq.ticket = it->second.id;
            dq.payload = std::move(it->second.payload);
            dq.reason = it->second.cancelled ? AdmissionRejectReason::kCancelled
                                             : AdmissionRejectReason::kTimeout;
            dropped.push_back(std::move(dq));
            entries_.erase(it);
            tit = bit->tickets.erase(tit);
            --delta;
          } else {
            ++tit;
          }
        }
        if (bit->tickets.empty()) {
          bit = client_buckets_.erase(bit);
        } else {
          ++bit;
        }
      }
      if (delta != 0) AdjustQueuedLocked(delta);
    }
    if (!dropped.empty()) cv_.notify_all();
    return dropped;
  }

  void Stop() noexcept {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
  }

  void SetDeviceLost() noexcept {
    {
      std::lock_guard<std::mutex> lock(mu_);
      device_lost_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
  }

  void Notify() noexcept { cv_.notify_all(); }

  /// Waits under queue mutex for up to `timeout` until `wake_pred()` returns
  /// true. Optional `on_wait_begin` / `on_wait_end` run around the timed sleep
  /// only when `wake_pred()` is initially false.
  template <typename Rep, typename Period, typename WakePred,
            typename OnBegin = std::nullptr_t, typename OnEnd = std::nullptr_t>
  bool WaitIdleFor(std::chrono::duration<Rep, Period> timeout,
                   WakePred&& wake_pred, OnBegin on_begin = nullptr,
                   OnEnd on_end = nullptr) {
    std::unique_lock<std::mutex> lock(mu_);
    if (wake_pred()) return true;
    if constexpr (!std::is_same_v<OnBegin, std::nullptr_t>) {
      on_begin();
    }
    const bool woken = cv_.wait_for(lock, timeout, wake_pred);
    if constexpr (!std::is_same_v<OnEnd, std::nullptr_t>) {
      on_end();
    }
    return woken;
  }

  [[nodiscard]] bool device_lost() const noexcept {
    return device_lost_.load(std::memory_order_acquire);
  }
  [[nodiscard]] bool stopping() const noexcept {
    return stopping_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::size_t queued() const noexcept {
    return queued_count_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::size_t active() const noexcept {
    return active_count_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::size_t pending() const noexcept {
    return queued_count_.load(std::memory_order_relaxed) +
           active_count_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] const AdmissionPolicy& policy() const noexcept {
    return policy_;
  }

 private:
  struct Entry {
    std::uint64_t id{0};
    std::string client;
    AdmissionPhase phase{AdmissionPhase::kQueued};
    bool cancelled{false};
    Clock::time_point enqueued{};
    std::optional<Clock::time_point> deadline;
    Payload payload{};
  };

  struct ClientBucket {
    std::string client_id;
    std::deque<std::uint64_t> tickets;
  };

  ClientBucket* FindBucketLocked(const std::string& client) {
    for (auto& b : client_buckets_) {
      if (b.client_id == client) return &b;
    }
    return nullptr;
  }

  void RemoveFromBucketLocked(std::uint64_t ticket,
                              const std::string& client) {
    for (auto bit = client_buckets_.begin(); bit != client_buckets_.end();
         ++bit) {
      if (bit->client_id != client) continue;
      for (auto tit = bit->tickets.begin(); tit != bit->tickets.end(); ++tit) {
        if (*tit == ticket) {
          bit->tickets.erase(tit);
          if (bit->tickets.empty()) client_buckets_.erase(bit);
          return;
        }
      }
    }
  }

  void AdjustQueuedLocked(std::int64_t delta) noexcept {
    if (delta == 0) return;
    if (delta > 0) {
      queued_count_.fetch_add(static_cast<std::size_t>(delta),
                              std::memory_order_relaxed);
    } else {
      const std::size_t sub = static_cast<std::size_t>(-delta);
      const std::size_t cur = queued_count_.load(std::memory_order_relaxed);
      queued_count_.store(cur >= sub ? (cur - sub) : 0u,
                          std::memory_order_relaxed);
    }
    if (on_queued_delta_) {
      try {
        on_queued_delta_(delta);
      } catch (...) {
      }
    }
  }

  AdmissionPolicy policy_;
  std::uint64_t next_id_;
  std::atomic<std::size_t> queued_count_{0};
  std::atomic<std::size_t> active_count_{0};
  std::atomic<bool> device_lost_{false};
  std::atomic<bool> stopping_{false};
  std::deque<ClientBucket> client_buckets_;
  std::unordered_map<std::uint64_t, Entry> entries_;
  std::function<void(std::int64_t)> on_queued_delta_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace mbufo::core
