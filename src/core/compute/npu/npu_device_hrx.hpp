#pragma once
// HrxSession: real NPU open/register/load/run/sync over hrx + npu_xclbin_manager.
// Compiled only with MBUFO_ENABLE_NPU (needs FastFlow SDK headers + libhrx).

#include <climits>  // first: SDK npu_cmd.hpp uses UCHAR_MAX without including it
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hrx_cpp/hrx_cpp.hpp"
#include "npu_utils/npu_utils_hrx.hpp"

namespace mbufo::core::npu {

/// RAII wrapper for a built HRX executable handle.
struct HrxExecutable {
  hrx_executable_t exe{nullptr};
  std::uint32_t ordinal{0};

  HrxExecutable() = default;
  HrxExecutable(hrx_executable_t e, std::uint32_t ord) noexcept
      : exe(e), ordinal(ord) {}
  ~HrxExecutable() { Reset(); }

  HrxExecutable(HrxExecutable&& other) noexcept
      : exe(other.exe), ordinal(other.ordinal) {
    other.exe = nullptr;
    other.ordinal = 0;
  }
  HrxExecutable& operator=(HrxExecutable&& other) noexcept {
    if (this != &other) {
      Reset();
      exe = other.exe;
      ordinal = other.ordinal;
      other.exe = nullptr;
      other.ordinal = 0;
    }
    return *this;
  }
  HrxExecutable(const HrxExecutable&) = delete;
  HrxExecutable& operator=(const HrxExecutable&) = delete;

  void Reset() noexcept {
    if (exe != nullptr) {
      hrx_executable_release(exe);
      exe = nullptr;
    }
    ordinal = 0;
  }
};

/// One NPU device handle + its xclbin/executable registry. Not thread-safe by
/// itself: callers serialize via NpuDeviceLock / NpuScheduler.
class HrxSession {
 public:
  /// Open device `index`, probe with a synchronize. Null + error on failure.
  static std::unique_ptr<HrxSession> Open(int index, std::string* error);
  ~HrxSession();

  HrxSession(const HrxSession&) = delete;
  HrxSession& operator=(const HrxSession&) = delete;

  /// Legacy xclbin registration (npu_xclbin_manager).
  bool RegisterXclbin(const std::string& name, const std::string& path,
                      std::string* error);

  /// Build and cache an IRON executable directly from `.xclbin` + `.insts.bin`.
  bool LoadIronKernel(const std::string& name, const std::string& xclbin_path,
                      const std::string& insts_path, std::string* error);
  void UnloadIronKernel(const std::string& name);
  void ClearIronKernels();
  [[nodiscard]] bool HasIronKernel(const std::string& name) const;

  /// Allocate a persistent-mapped zeroed HRX BO of `bytes` size.
  [[nodiscard]] std::unique_ptr<hrx::ext::bo> AllocBo(std::size_t bytes,
                                                      std::string* error);

  /// Dispatch one loaded IRON kernel synchronously with ordered BO bindings.
  bool RunKernelSync(const std::string& name,
                     const std::vector<const hrx::bo*>& bindings,
                     std::string* error);

  /// Record a sequence of IRON kernel dispatches and wait once for the chain.
  bool RunKernelChain(
      const std::vector<std::pair<std::string, std::vector<const hrx::bo*>>>&
          steps,
      std::string* error);

  /// Block until the device is idle.
  bool Sync(std::string* error);

  npu_app_manager* App(const std::string& name);
  bool HasApp(const std::string& name) const;
  std::string device_info() const;

 private:
  HrxSession(hrx::device dev, int index);
  hrx::device device_;
  npu_xclbin_manager mgr_;
  std::unordered_map<std::string, npu_app_manager*> apps_;
  std::unordered_map<std::string, HrxExecutable> iron_kernels_;
  int index_;
};

}  // namespace mbufo::core::npu
