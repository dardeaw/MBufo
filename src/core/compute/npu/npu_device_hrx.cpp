// HrxSession implementation + real HrxBind/HrxDeviceName used when MBUFO_ENABLE_NPU.
// Build: only in libmbufo_npu (CMake option MBUFO_ENABLE_NPU, default OFF).

#include "src/core/compute/npu/npu_device_hrx.hpp"

#include <cstring>
#include <exception>
#include <fstream>

namespace mbufo::core::npu {
namespace {

bool ReadBinaryBytes(const std::string& path, std::vector<std::uint8_t>* out,
                     std::string* error) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in.is_open()) {
    if (error != nullptr) *error = "cannot open file: " + path;
    return false;
  }
  const auto sz = in.tellg();
  if (sz <= 0) {
    if (error != nullptr) *error = "empty file: " + path;
    return false;
  }
  out->resize(static_cast<std::size_t>(sz));
  in.seekg(0, std::ios::beg);
  if (!in.read(reinterpret_cast<char*>(out->data()), sz)) {
    if (error != nullptr) *error = "failed reading file: " + path;
    return false;
  }
  return true;
}

}  // namespace

HrxSession::HrxSession(hrx::device dev, int index)
    : device_(std::move(dev)), mgr_(device_npu2, &device_, false), index_(index) {}

HrxSession::~HrxSession() { ClearIronKernels(); }

std::unique_ptr<HrxSession> HrxSession::Open(int index, std::string* error) {
  try {
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

bool HrxSession::LoadIronKernel(const std::string& name,
                                const std::string& xclbin_path,
                                const std::string& insts_path,
                                std::string* error) {
  if (name.empty()) {
    if (error != nullptr) *error = "LoadIronKernel: empty name";
    return false;
  }
  std::vector<std::uint8_t> xclbin_bytes;
  if (!ReadBinaryBytes(xclbin_path, &xclbin_bytes, error)) return false;
  std::vector<std::uint8_t> insts_bytes;
  if (!ReadBinaryBytes(insts_path, &insts_bytes, error)) return false;
  if (insts_bytes.size() % sizeof(std::uint32_t) != 0) {
    if (error != nullptr) {
      *error = "LoadIronKernel: insts size not a multiple of 4 in " + insts_path;
    }
    return false;
  }
  std::vector<std::uint32_t> words(insts_bytes.size() / sizeof(std::uint32_t));
  std::memcpy(words.data(), insts_bytes.data(), insts_bytes.size());

  try {
    std::uint32_t ord = 0;
    hrx_executable_t exe = hrx::build_executable(xclbin_bytes, words.data(),
                                                 words.size(), &ord);
    if (exe == nullptr) {
      if (error != nullptr) {
        *error = "LoadIronKernel: hrx::build_executable failed for " + name +
                 " (" + xclbin_path + ")";
      }
      return false;
    }
    iron_kernels_[name] = HrxExecutable(exe, ord);
    return true;
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = std::string("LoadIronKernel(") + name + "): " + e.what();
    }
    return false;
  }
}

void HrxSession::UnloadIronKernel(const std::string& name) {
  iron_kernels_.erase(name);
}

void HrxSession::ClearIronKernels() { iron_kernels_.clear(); }

bool HrxSession::HasIronKernel(const std::string& name) const {
  return iron_kernels_.count(name) != 0u;
}

std::unique_ptr<hrx::ext::bo> HrxSession::AllocBo(std::size_t bytes,
                                                  std::string* error) {
  if (bytes == 0) {
    if (error != nullptr) *error = "AllocBo: zero byte allocation requested";
    return nullptr;
  }
  try {
    auto bo = std::make_unique<hrx::ext::bo>(device_, bytes);
    std::memset(bo->map<void*>(), 0, bytes);
    bo->flush();
    return bo;
  } catch (const std::exception& e) {
    if (error != nullptr) *error = std::string("AllocBo: ") + e.what();
    return nullptr;
  }
}

bool HrxSession::RunKernelSync(const std::string& name,
                               const std::vector<const hrx::bo*>& bindings,
                               std::string* error) {
  auto it = iron_kernels_.find(name);
  if (it == iron_kernels_.end() || it->second.exe == nullptr) {
    if (error != nullptr) *error = "RunKernelSync: unknown kernel " + name;
    return false;
  }
  try {
    hrx_executable_retain(it->second.exe);
    hrx::run r(it->second.exe, it->second.ordinal);
    for (const hrx::bo* b : bindings) {
      if (b == nullptr) {
        if (error != nullptr) *error = "RunKernelSync: null BO binding in " + name;
        return false;
      }
      r.add_binding(*b);
    }
    r.start();
    if (r.wait() != ERT_CMD_STATE_COMPLETED) {
      if (error != nullptr) {
        *error = "RunKernelSync: kernel execution failed for " + name;
      }
      return false;
    }
    return true;
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = std::string("RunKernelSync(") + name + "): " + e.what();
    }
    return false;
  }
}

bool HrxSession::RunKernelChain(
    const std::vector<std::pair<std::string, std::vector<const hrx::bo*>>>& steps,
    std::string* error) {
  if (steps.empty()) return true;
  try {
    hrx::runlist rl;
    for (const auto& [name, bindings] : steps) {
      auto it = iron_kernels_.find(name);
      if (it == iron_kernels_.end() || it->second.exe == nullptr) {
        if (error != nullptr) *error = "RunKernelChain: unknown kernel " + name;
        return false;
      }
      hrx_executable_retain(it->second.exe);
      hrx::run r(it->second.exe, it->second.ordinal);
      for (const hrx::bo* b : bindings) {
        if (b == nullptr) {
          if (error != nullptr) {
            *error = "RunKernelChain: null BO binding in " + name;
          }
          return false;
        }
        r.add_binding(*b);
      }
      rl.add(std::move(r));
    }
    rl.execute();
    if (rl.wait() != ERT_CMD_STATE_COMPLETED) {
      if (error != nullptr) *error = "RunKernelChain: chain execution failed";
      return false;
    }
    return true;
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = std::string("RunKernelChain: ") + e.what();
    }
    return false;
  }
}

bool HrxSession::Sync(std::string* error) {
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
