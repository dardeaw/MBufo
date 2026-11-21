// HrxSession implementation + real HrxBind/HrxDeviceName used when MBUFO_ENABLE_NPU.
// Build: only in libmbufo_npu (CMake option MBUFO_ENABLE_NPU, default OFF).

#include "src/core/compute/npu/npu_device_hrx.hpp"

#include <exception>

namespace mbufo::core::npu {

HrxSession::HrxSession(hrx::device dev, int index)
    : device_(std::move(dev)), mgr_(device_npu2, &device_, false), index_(index) {}

std::unique_ptr<HrxSession> HrxSession::Open(int index, std::string* error) {
  try {
    // hrx::device ctor runs rt().ensure(); probe liveness with a sync,
    // exactly like spike3/coexist_spike did before trusting the handle.
    std::unique_ptr<HrxSession> s(new HrxSession(hrx::device(index), index));
    if (!s->Sync(error)) return nullptr;
    return s;
  } catch (const std::exception& e) {
    if (error != nullptr) *error = std::string("HrxSession::Open: ") + e.what();
    return nullptr;
  }
}

bool HrxSession::RegisterXclbin(const std::string& name, const std::string& path,
                                std::string* error) {
  if (name.empty() || path.empty()) {
    if (error != nullptr) *error = "RegisterXclbin: empty name or path";
    return false;
  }
  try {
    npu_app_manager* app = mgr_.register_xclbin(path);
    if (app == nullptr) {
      if (error != nullptr) {
        *error = "register_xclbin returned null for " + path;
      }
      return false;
    }
    apps_[name] = app;
    return true;
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = std::string("RegisterXclbin(") + path + "): " + e.what();
    }
    return false;
  }
}

bool HrxSession::Sync(std::string* error) {
  // hrx_report prints the decoded status and returns true on error.
  if (hrx::hrx_report(hrx_device_synchronize(hrx::rt().dev),
                      "HrxSession::Sync")) {
    if (error != nullptr) {
      *error = "hrx_device_synchronize reported an error (see stderr)";
    }
    return false;
  }
  return true;
}

npu_app_manager* HrxSession::App(const std::string& name) {
  auto it = apps_.find(name);
  return it != apps_.end() ? it->second : nullptr;
}

bool HrxSession::HasApp(const std::string& name) const {
  return apps_.count(name) != 0u;
}

std::string HrxSession::device_info() const {
  try {
    return device_.get_info<hrx::info::device::name>();
  } catch (...) {
    return std::string{};
  }
}

// Probe used by NpuDeviceLock owners: open + sync + verify device identity.
bool HrxBind(std::string* error) {
  auto s = HrxSession::Open(0, error);
  if (!s) return false;
  const std::string info = s->device_info();
  if (info.empty()) {
    if (error != nullptr) {
      *error = "HrxBind: HRX device opened/synced but device_info() is empty";
    }
    return false;
  }
  return true;
}

std::string HrxDeviceName(std::string* error) {
  auto s = HrxSession::Open(0, error);
  if (!s) return {};
  return s->device_info();
}

}  // namespace mbufo::core::npu
