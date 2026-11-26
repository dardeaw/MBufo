#pragma once
// NPU warmup helper (M1): run one dummy inference right after weights are
// resident so the first real request does not pay xclbin/HW-context cold
// start. Soft-fail by design: a failed warmup never fails the load, it is
// logged and reported (`warm=false`) so the admin UI can show it.
// MBUFO_NPU_WARMUP=0 disables every NPU warmup.

#include <chrono>
#include <cstdlib>
#include <functional>
#include <string>
#include <string_view>

#include "src/cli/serve/logging.hpp"

namespace mbufo::server {

[[nodiscard]] inline bool NpuWarmupEnabled() {
  const char* v = std::getenv("MBUFO_NPU_WARMUP");
  return v == nullptr || std::string_view(v) != "0";
}

struct NpuWarmupOutcome {
  bool ran{false};
  bool ok{false};
  double ms{0.0};
  std::string error;
};

inline NpuWarmupOutcome RunNpuWarmup(
    const std::string& model, const std::function<bool(std::string*)>& fn) {
  NpuWarmupOutcome out;
  if (!NpuWarmupEnabled()) return out;
  out.ran = true;
  const auto t0 = std::chrono::steady_clock::now();
  try {
    out.ok = fn(&out.error);
  } catch (const std::exception& e) {
    out.ok = false;
    out.error = e.what();
  }
  out.ms = std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
               .count();
  Logger::Info("loader", "event=npu_warmup model=" + model +
                             " ok=" + (out.ok ? "true" : "false") +
                             " ms=" + std::to_string(out.ms) +
                             (out.ok ? "" : " error=" + out.error));
  return out;
}

}  // namespace mbufo::server
