#pragma once
// NPU run-to-completion scheduler backed by shared core::AdmissionQueue and
// NpuDeviceLock (header-only).
//
// Work shape: one-shot Sync execution — embedding / reranker / D3 decide.
// No token decode phase machine, no KV, no sampling. Submitters push work via
// Submit() / SubmitAsync(), wait via Wait(), and can Cancel(). A single owner
// pump thread drains cancelled/expired items, acquires NpuDeviceLock first,
// admits 1..max_coalesce_batch compatible items via AdmissionQueue, marks them
// running, runs them (sequentially or via NpuCoalesceExecutor) under the
// device lock, publishes completion, and calls Finish() to reclaim.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/admission_queue.hpp"
#include "src/core/compute/npu/npu_device.hpp"
#include "src/core/compute/npu/npu_lane.hpp"

namespace mbufo::core::npu {

enum class NpuWorkLane : std::uint8_t {
  kEmbedding = 0,
  kReranker,
  kDecide,
};

struct NpuWorkDesc {
  NpuWorkLane lane{NpuWorkLane::kEmbedding};
  std::size_t seq_len{0};
  std::size_t bytes_estimate{0};
  std::string client{"anonymous"};
  /// Non-empty key enables same-lane batch coalescing under a single device
  /// lock acquisition when NpuSchedulerOptions::max_coalesce_batch > 1.
  std::string coalesce_key{};
  /// Optional per-request timeout; zero uses NpuAdmissionPolicy::request_timeout.
  std::chrono::milliseconds timeout{0};
};

struct NpuWorkResult {
  bool ok{false};
  std::string error{"npu-disabled"};
  std::uint64_t ticket_id{0};
  std::size_t coalesced_batch_size{1};
  mbufo::core::AdmissionRejectReason reject_reason{
      mbufo::core::AdmissionRejectReason::kNone};
};

using NpuWork = std::function<NpuWorkResult()>;

struct NpuBatchEntry {
  NpuWorkDesc desc;
  NpuWork work;
  NpuWorkResult result;
};

/// Optional custom batch executor invoked when >1 compatible requests are
/// coalesced into a single NpuDeviceLock acquisition.
using NpuCoalesceExecutor = std::function<void(std::vector<NpuBatchEntry>&)>;

struct NpuSchedulerOptions {
  std::size_t max_coalesce_batch{1};
  std::chrono::milliseconds lock_acquire_timeout{100};
  std::chrono::milliseconds idle_poll_interval{25};
};

namespace detail {

struct NpuWorkItem {
  NpuWorkDesc desc;
  NpuWork work;
  std::mutex mu;
  std::condition_variable cv;
  bool done{false};
  NpuWorkResult result{};

  void Publish(NpuWorkResult res) noexcept {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (done) return;
      result = std::move(res);
      done = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] NpuWorkResult Wait() {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return done; });
    return result;
  }
};

}  // namespace detail

class NpuScheduler;

/// Handle returned by SubmitAsync() supporting Wait() and Cancel().
class NpuTicketHandle {
 public:
  NpuTicketHandle() = default;

  [[nodiscard]] std::uint64_t ticket_id() const noexcept { return ticket_id_; }
  [[nodiscard]] bool valid() const noexcept { return ticket_id_ != 0; }

  [[nodiscard]] NpuWorkResult Wait() {
    if (!item_) return immediate_;
    return item_->Wait();
  }

  bool Cancel() noexcept {
    if (ticket_id_ == 0 || queue_ == nullptr) return false;
    return queue_->Cancel(ticket_id_, false);
  }

 private:
  friend class NpuScheduler;
  NpuWorkResult immediate_{};
  std::uint64_t ticket_id_{0};
  std::shared_ptr<detail::NpuWorkItem> item_;
  mbufo::core::AdmissionQueue<std::shared_ptr<detail::NpuWorkItem>>* queue_{
      nullptr};
};

class NpuScheduler {
 public:
  explicit NpuScheduler(
      mbufo::npu::NpuLaneConfig config = {}, NpuAdmissionPolicy policy = {},
      NpuDeviceLock* external_device_lock = nullptr,
      NpuSchedulerOptions options = {},
      NpuCoalesceExecutor coalesce_executor = {})
      : config_(std::move(config)),
        options_(options),
        coalesce_executor_(std::move(coalesce_executor)),
        device_lock_(external_device_lock != nullptr ? external_device_lock
                                                     : &owned_device_lock_),
        queue_(policy) {}

  ~NpuScheduler() { Stop(); }

  NpuScheduler(const NpuScheduler&) = delete;
  NpuScheduler& operator=(const NpuScheduler&) = delete;

  [[nodiscard]] bool enabled() const noexcept {
    return config_.embedding || config_.reranker || config_.decide;
  }

  /// Explicit pump start (also started lazily on the first enabled Submit).
  void Start() { EnsurePumpStarted(); }

  void Stop() noexcept {
    queue_.Stop();
    {
      std::lock_guard<std::mutex> lock(pump_mu_);
      if (pump_thread_.joinable()) {
        pump_thread_.request_stop();
        queue_.Notify();
        pump_thread_.join();
      }
    }
    for (auto& [ticket, item] : queue_.CancelAll()) {
      ResolveRejectedItem(item, ticket,
                          mbufo::core::AdmissionRejectReason::kStopping,
                          "npu-stopping");
    }
  }

  /// Asynchronous enqueue; returns ticket handle for Wait() / Cancel().
  [[nodiscard]] NpuTicketHandle SubmitAsync(NpuWorkDesc desc, NpuWork work) {
    NpuTicketHandle h;
    if (!LaneOn(desc.lane)) {
      h.immediate_ = NpuWorkResult{
          .ok = false, .error = "npu-disabled", .ticket_id = 0};
      return h;
    }
    EnsurePumpStarted();
    auto item = std::make_shared<detail::NpuWorkItem>();
    item->desc = desc;
    item->work = std::move(work);
    const std::chrono::milliseconds to = item->desc.timeout;
    const auto submit_res =
        queue_.SubmitWithTimeout(item->desc.client, item, to);
    if (submit_res.ticket == 0) {
      h.immediate_.ok = false;
      h.immediate_.ticket_id = 0;
      h.immediate_.reject_reason = submit_res.reject;
      h.immediate_.error = RejectErrorString(submit_res.reject);
      return h;
    }
    h.ticket_id_ = submit_res.ticket;
    h.item_ = std::move(item);
    h.queue_ = &queue_;
    return h;
  }

  /// Synchronous one-shot Sync execution through shared AdmissionQueue + pump.
  [[nodiscard]] NpuWorkResult Submit(NpuWorkDesc desc, NpuWork work) {
    NpuTicketHandle h = SubmitAsync(std::move(desc), std::move(work));
    return h.Wait();
  }

  bool Cancel(std::uint64_t ticket_id) noexcept {
    return queue_.Cancel(ticket_id, false);
  }

  void CancelAll() noexcept {
    for (auto& [ticket, item] : queue_.CancelAll()) {
      ResolveRejectedItem(item, ticket,
                          mbufo::core::AdmissionRejectReason::kCancelled,
                          "cancelled");
    }
  }

  void SetDeviceLost() noexcept {
    device_lock_->SetDeviceLost();
    queue_.SetDeviceLost();
    for (auto& [ticket, item] : queue_.CancelAll()) {
      ResolveRejectedItem(item, ticket,
                          mbufo::core::AdmissionRejectReason::kDeviceLost,
                          "device-lost");
    }
  }

  [[nodiscard]] bool device_lost() const noexcept {
    return queue_.device_lost() || device_lock_->device_lost();
  }

  // ---- owner / diagnostic introspection ----
  [[nodiscard]] std::uint64_t AdmitNext() { return queue_.AdmitNext(); }
  void MarkRunning(std::uint64_t ticket) noexcept {
    queue_.MarkRunning(ticket);
  }
  void Finish(std::uint64_t ticket) noexcept { queue_.Finish(ticket); }
  [[nodiscard]] std::size_t queued() const noexcept { return queue_.queued(); }
  [[nodiscard]] std::size_t active() const noexcept { return queue_.active(); }
  [[nodiscard]] std::size_t pending() const noexcept {
    return queue_.pending();
  }
  [[nodiscard]] NpuDeviceLock* device_lock() noexcept { return device_lock_; }
  [[nodiscard]] const mbufo::npu::NpuLaneConfig& config() const noexcept {
    return config_;
  }

 private:
  using WorkPtr = std::shared_ptr<detail::NpuWorkItem>;

  [[nodiscard]] bool LaneOn(NpuWorkLane lane) const noexcept {
    switch (lane) {
      case NpuWorkLane::kEmbedding:
        return config_.embedding;
      case NpuWorkLane::kReranker:
        return config_.reranker;
      case NpuWorkLane::kDecide:
        return config_.decide;
    }
    return false;
  }

  [[nodiscard]] static const char* RejectErrorString(
      mbufo::core::AdmissionRejectReason r) noexcept {
    switch (r) {
      case mbufo::core::AdmissionRejectReason::kStopping:
        return "npu-stopping";
      case mbufo::core::AdmissionRejectReason::kDeviceLost:
        return "device-lost";
      case mbufo::core::AdmissionRejectReason::kQueueFull:
        return "queue-full";
      case mbufo::core::AdmissionRejectReason::kClientQueueFull:
        return "client-queue-full";
      case mbufo::core::AdmissionRejectReason::kCancelled:
        return "cancelled";
      case mbufo::core::AdmissionRejectReason::kTimeout:
        return "timeout";
      case mbufo::core::AdmissionRejectReason::kUnknownTicket:
        return "unknown-ticket";
      case mbufo::core::AdmissionRejectReason::kNone:
        return "";
    }
    return "rejected";
  }

  static void ResolveRejectedItem(const WorkPtr& item, std::uint64_t ticket,
                                  mbufo::core::AdmissionRejectReason reason,
                                  const char* msg) noexcept {
    if (!item) return;
    NpuWorkResult r;
    r.ok = false;
    r.ticket_id = ticket;
    r.reject_reason = reason;
    r.error = msg;
    item->Publish(std::move(r));
  }

  void EnsurePumpStarted() {
    std::lock_guard<std::mutex> lock(pump_mu_);
    if (!pump_thread_.joinable() && !queue_.stopping()) {
      pump_thread_ = std::jthread(
          [this](std::stop_token st) { PumpLoop(std::move(st)); });
    }
  }

  void PumpLoop(std::stop_token stop_token) {
    while (!stop_token.stop_requested() && !queue_.stopping()) {
      for (auto& dq : queue_.DrainCancelledOrExpired()) {
        ResolveRejectedItem(dq.payload, dq.ticket, dq.reason,
                            RejectErrorString(dq.reason));
      }

      if (device_lock_->device_lost()) {
        queue_.SetDeviceLost();
      }
      if (queue_.device_lost()) {
        for (auto& [ticket, item] : queue_.CancelAll()) {
          ResolveRejectedItem(item, ticket,
                              mbufo::core::AdmissionRejectReason::kDeviceLost,
                              "device-lost");
        }
        break;
      }

      if (queue_.queued() == 0) {
        queue_.WaitIdleFor(options_.idle_poll_interval, [&] {
          return stop_token.stop_requested() || queue_.stopping() ||
                 queue_.device_lost() || queue_.queued() != 0;
        });
        continue;
      }

      // Lock-first contract: hold NpuDeviceLock BEFORE admitting so tickets
      // never strand in Admitted/Running when an external holder owns NPU.
      NpuDeviceLock::Hold dev_hold(device_lock_, options_.lock_acquire_timeout);
      if (!dev_hold.holds()) {
        if (device_lock_->device_lost()) continue;
        queue_.WaitIdleFor(std::chrono::milliseconds{2}, [&] {
          return stop_token.stop_requested() || queue_.stopping() ||
                 queue_.device_lost();
        });
        continue;
      }

      const std::size_t max_b =
          std::max<std::size_t>(1u, options_.max_coalesce_batch);
      std::vector<mbufo::core::AdmissionQueue<WorkPtr>::DroppedQueued> dropped;
      auto batch = queue_.AdmitCompatibleBatch(
          max_b,
          [](const WorkPtr& leader, const WorkPtr& candidate) {
            return leader && candidate &&
                   leader->desc.lane == candidate->desc.lane &&
                   !leader->desc.coalesce_key.empty() &&
                   leader->desc.coalesce_key == candidate->desc.coalesce_key;
          },
          &dropped);
      for (auto& dq : dropped) {
        ResolveRejectedItem(dq.payload, dq.ticket, dq.reason,
                            RejectErrorString(dq.reason));
      }
      if (batch.empty()) {
        continue;
      }

      for (const auto& [ticket, item] : batch) {
        (void)item;
        queue_.MarkRunning(ticket);
      }

      if (batch.size() > 1 && coalesce_executor_) {
        std::vector<NpuBatchEntry> entries;
        entries.reserve(batch.size());
        for (const auto& [ticket, item] : batch) {
          entries.push_back(NpuBatchEntry{
              .desc = item->desc,
              .work = item->work,
              .result = {.ticket_id = ticket,
                         .coalesced_batch_size = batch.size()},
          });
        }
        try {
          coalesce_executor_(entries);
        } catch (const std::exception& e) {
          for (auto& ent : entries) {
            ent.result.ok = false;
            ent.result.error = std::string("npu-coalesce-exception: ") + e.what();
          }
        } catch (...) {
          for (auto& ent : entries) {
            ent.result.ok = false;
            ent.result.error = "npu-coalesce-unknown-exception";
          }
        }
        for (std::size_t i = 0; i < batch.size(); ++i) {
          auto& res = entries[i].result;
          res.ticket_id = batch[i].first;
          res.coalesced_batch_size = batch.size();
          queue_.Finish(batch[i].first);
          batch[i].second->Publish(std::move(res));
        }
      } else {
        for (auto& [ticket, item] : batch) {
          NpuWorkResult res;
          if (!item->work) {
            res.ok = false;
            res.error = "npu-empty-work";
          } else {
            try {
              res = item->work();
            } catch (const std::exception& e) {
              res.ok = false;
              res.error = std::string("npu-exception: ") + e.what();
            } catch (...) {
              res.ok = false;
              res.error = "npu-unknown-exception";
            }
          }
          res.ticket_id = ticket;
          res.coalesced_batch_size = batch.size();
          queue_.Finish(ticket);
          item->Publish(std::move(res));
        }
      }
    }

    for (auto& dq : queue_.DrainCancelledOrExpired()) {
      ResolveRejectedItem(dq.payload, dq.ticket, dq.reason,
                          RejectErrorString(dq.reason));
    }
  }

  mbufo::npu::NpuLaneConfig config_;
  NpuSchedulerOptions options_;
  NpuCoalesceExecutor coalesce_executor_;
  NpuDeviceLock owned_device_lock_;
  NpuDeviceLock* device_lock_;
  mbufo::core::AdmissionQueue<WorkPtr> queue_;
  std::mutex pump_mu_;
  std::jthread pump_thread_;
};

/// Runs one synchronous Sync payload via NpuScheduler (or directly if null).
[[nodiscard]] inline NpuWorkResult RunLaneSync(
    NpuScheduler& scheduler, NpuWorkDesc desc, NpuWork work,
    std::chrono::milliseconds timeout = {}) {
  if (timeout.count() > 0 && desc.timeout.count() == 0) {
    desc.timeout = timeout;
  }
  return scheduler.Submit(std::move(desc), std::move(work));
}

}  // namespace mbufo::core::npu
