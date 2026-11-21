#pragma once
// HrxSession: real NPU open/register/sync over hrx + npu_xclbin_manager.
// Compiled only with MBUFO_ENABLE_NPU (needs FastFlow SDK headers + libhrx).
// Mirrors spike3's proven sequence line-for-line:
//   hrx::device dev(0) -> npu_xclbin_manager(device_npu2, &dev, false)
//   -> register_xclbin(path) -> synchronize.

#include <climits>  // first: SDK npu_cmd.hpp uses UCHAR_MAX without including it
#include <memory>
#include <string>
#include <unordered_map>

#include "hrx_cpp/hrx_cpp.hpp"
#include "npu_utils/npu_utils_hrx.hpp"

namespace mbufo::core::npu {

/// One NPU device handle + its xclbin registry. Not thread-safe by itself:
/// all calls must happen under NpuDeviceLock (see npu_device.hpp PumpOnce).
class HrxSession {
 public:
  /// Open device `index`, probe with a synchronize. Null + error on failure
  /// (caller maps repeated failure to device_lost).
  static std::unique_ptr<HrxSession> Open(int index, std::string* error);
  ~HrxSession() = default;

  HrxSession(const HrxSession&) = delete;
  HrxSession& operator=(const HrxSession&) = delete;

  /// Register an xclbin file under a logical name (idempotent per path:
  /// the manager dedups by path internally). False + error when the file
  /// cannot be opened or the 16-xclbin budget is exhausted.
  bool RegisterXclbin(const std::string& name, const std::string& path,
                       std::string* error);

  /// Block until the device is idle. Any failure is surfaced (never
  /// swallowed: a failed ERT chain must not become a silent no-op dispatch).
  bool Sync(std::string* error);

  /// App handle for a registered name, or nullptr when unknown. Owned by
  /// the session's manager; valid until session destruction.
  npu_app_manager* App(const std::string& name);
  bool HasApp(const std::string& name) const;

  /// Human-readable device identity (IREE HAL name, e.g. "amdxdna").
  std::string device_info() const;

 private:
  HrxSession(hrx::device dev, int index);
  hrx::device device_;
  npu_xclbin_manager mgr_;
  std::unordered_map<std::string, npu_app_manager*> apps_;
  int index_;
};

}  // namespace mbufo::core::npu
