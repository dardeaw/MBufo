#include "src/core/compute/npu/npu_device.hpp"
#include "src/core/compute/npu/npu_device_hrx.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

int main() {
  using namespace mbufo::core::npu;

  NpuDeviceLock dev;
  std::string err;
  std::string dev_name;
  assert(dev.ProbeHardware(&err, &dev_name));
  assert(dev_name == "amdxdna");

  auto sess = HrxSession::Open(0, &err);
  assert(sess != nullptr);
  assert(sess->device_info() == "amdxdna");

  const std::string xclbin_path =
      "/home/dardeaw/venv-npu/lib/python3.12/site-packages/"
      "fastflow_sdk_fastflowlm/share/flm/xclbins/Embedding-Gemma-300M-NPU2/"
      "mm.xclbin";
  if (std::filesystem::exists(xclbin_path)) {
    assert(sess->RegisterXclbin("eg2_mm", xclbin_path, &err));
    assert(sess->HasApp("eg2_mm"));
    assert(sess->App("eg2_mm") != nullptr);
    assert(sess->Sync(&err));
  }

  std::cout << "HRX_PROBE_PASS dev=" << dev_name << "\n";
  return 0;
}
