#pragma once
// NPU device lock + xclbin table + HRX hardware bind seam.
//
// Why this file exists: XDNA2 runs one graph at a time. All five lanes share
// one device, so exactly one owner thread may submit. This file owns:
//   1. the single device mutex (timed: submissions fail fast, never hang);
//   2. the xclbin registry (register once by name, reuse across lanes);
//   3. the pump bridge: AdmitNext -> hold lock -> run payload -> Finish.
//
// Hardware seam: with MBUFO_ENABLE_NPU, HrxBind() and HrxDeviceName() call into
// npu_device_hrx.cpp (Open -> Sync over libhrx / amdxdna).
// Without MBUFO_ENABLE_NPU, inline stubs report kHrxUnbound so unit tests and
// HIP builds stay dependency-free.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

#include "src/core/compute/npu/npu_admission.hpp"

namespace mbufo::core::npu {

/// Named xclbin set, e.g. {"whisper-enc", "/.../Whisper-V3-Turbo-NPU2/"}.
/// Pure data table: register once, resolve by name on every submit.
class NpuXclbinTable final {
 public:
  /// Returns false when name is empty or already registered.
  bool Register(std::string name, std::string location, std::string* error) {
    if (name.empty()) {
      if (error != nullptr) *error = "xclbin name is empty";
      return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (table_.count(name) != 0u) {
      if (error != nullptr) *error = "xclbin already registered: " + name;
      return false;
    }
    table_[std::move(name)] = std::move(location);
    return true;
  }

  bool Unregister(const std::string& name) {
    std::lock_guard<std::mutex> lock(mu_);
    return table_.erase(name) != 0u;
  }

  [[nodiscard]] bool Has(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mu_);
    return table_.count(name) != 0u;
  }

  [[nodiscard]] std::string Resolve(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(name);
    return it != table_.end() ? it->second : std::string{};
  }

  [[nodiscard]] std::size_t Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return table_.size();
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::string> table_;
};

/// Single NPU device lock. One instance per process (NPU is not partitionable
/// on this platform). Submissions take the lock with a timeout and fail fast
/// with kBusy instead of wedging the HTTP worker.
class NpuDeviceLock final {
 public:
  enum class AcquireResult : std::uint8_t { kOk = 0, kBusy, kDeviceLost };

  NpuDeviceLock() = default;
  NpuDeviceLock(const NpuDeviceLock&) = delete;
  NpuDeviceLock& operator=(const NpuDeviceLock&) = delete;

  /// Try to hold the device for up to `timeout`. Zero = try once, no wait.
  [[nodiscard]] AcquireResult TryAcquire(
      std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) {
    if (device_lost_.load()) return AcquireResult::kDeviceLost;
    if (timeout.count() <= 0) {
      return mutex_.try_lock() ? AcquireResult::kOk : AcquireResult::kBusy;
    }
    return mutex_.try_lock_for(timeout) ? AcquireResult::kOk
                                        : AcquireResult::kBusy;
  }

  void Release() noexcept { mutex_.unlock(); }

  /// RAII hold. Move-only; releases on destruction.
  class Hold {
   public:
    Hold() noexcept = default;
    explicit Hold(NpuDeviceLock* dev,
                  std::chrono::milliseconds timeout) noexcept
        : dev_(nullptr) {
      if (dev != nullptr && dev->TryAcquire(timeout) == AcquireResult::kOk) {
        dev_ = dev;
      }
    }
    Hold(Hold&& o) noexcept : dev_(o.dev_) { o.dev_ = nullptr; }
    Hold& operator=(Hold&& o) noexcept {
      if (this != &o) {
        Reset();
        dev_ = o.dev_;
        o.dev_ = nullptr;
      }
      return *this;
    }
    ~Hold() { Reset(); }
    [[nodiscard]] bool holds() const noexcept { return dev_ != nullptr; }
    void Reset() noexcept {
      if (dev_ != nullptr) {
        dev_->Release();
        dev_ = nullptr;
      }
    }

   private:
    NpuDeviceLock* dev_{nullptr};
  };

  void SetDeviceLost() noexcept { device_lost_.store(true); }
  [[nodiscard]] bool device_lost() const noexcept {
    return device_lost_.load();
  }

  NpuXclbinTable& xclbins() noexcept { return xclbins_; }

  /// Pump one queued ticket: hold device -> admit -> run -> finish.
  /// Lock-first ordering guarantees a ticket is never stranded admitted
  /// when the device is busy (only the single owner thread pumps).
  /// `run` executes while the device lock is held (owner-thread contract).
  /// Returns the finished ticket id, or 0 when the queue is empty or the
  /// device could not be acquired.
  std::uint64_t PumpOnce(NpuAdmissionQueue& queue,
                         std::function<void(std::uint64_t)> run,
                         std::chrono::milliseconds acquire_timeout =
                             std::chrono::milliseconds{100}) {
    Hold hold(this, acquire_timeout);
    if (!hold.holds()) return 0u;
    const std::uint64_t ticket = queue.AdmitNext();
    if (ticket == 0u) return 0u;
    queue.MarkRunning(ticket);
    run(ticket);
    queue.Finish(ticket);
    return ticket;
  }

  /// Acquire the device lock and probe HRX hardware binding.
  [[nodiscard]] inline bool ProbeHardware(
      std::string* error, std::string* device_name = nullptr,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});

 private:
  std::timed_mutex mutex_;
  std::atomic<bool> device_lost_{false};
  NpuXclbinTable xclbins_;
};

/// Hardware bind seam: with MBUFO_ENABLE_NPU this is defined in
/// npu_device_hrx.cpp (real Open -> Sync over libhrx / amdxdna); without it,
/// the inline stub reports kHrxUnbound.
#ifdef MBUFO_ENABLE_NPU
[[nodiscard]] bool HrxBind(std::string* error);
[[nodiscard]] std::string HrxDeviceName(std::string* error);
#else
[[nodiscard]] inline bool HrxBind(std::string* error) {
  if (error != nullptr) {
    *error =
        "kHrxUnbound: built without MBUFO_ENABLE_NPU (configure with FastFlow "
        "SDK to enable real amdxdna Open/Register/Sync)";
  }
  return false;
}

[[nodiscard]] inline std::string HrxDeviceName(std::string* error) {
  (void)HrxBind(error);
  return {};
}
#endif

[[nodiscard]] inline bool NpuDeviceLock::ProbeHardware(
    std::string* error, std::string* device_name,
    std::chrono::milliseconds timeout) {
  Hold hold(this, timeout);
  if (!hold.holds()) {
    if (error != nullptr) {
      *error = device_lost() ? "npu device lost" : "npu device lock busy";
    }
    return false;
  }
  if (!HrxBind(error)) return false;
  if (device_name != nullptr) {
    *device_name = HrxDeviceName(error);
  }
  return true;
}

}  // namespace mbufo::core::npu
