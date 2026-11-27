#pragma once
// NPU warmup helper (M1): run one dummy inference right after weights are
// resident so the first real request does not pay xclbin/HW-context cold
// start. Soft-fail by design: a failed warmup never fails the load, it is
// logged and reported (`warm=false`) so the admin UI can show it.

#include <chrono>
#include <cstdlib>
#include <functional>
#include <string>
#include <string_view>

#include "src/cli/serve/logging.hpp"
#include "src/core/compute/npu/npu_servable.hpp"

namespace mbufo::server {

struct NpuWarmupOutcome {
  bool ran{false};
  bool ok{false};
  double ms{0.0};
  std::string error;
};

inline NpuWarmupOutcome RunNpuWarmup(
    const std::string& model, const std::function<bool(std::string*)>& fn) {
  NpuWarmupOutcome out;
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

/// Load an in-process NPU model through the shared INpuServable lifecycle.
/// `open` runs the model's existing weight load; `warm` is its dummy inference
/// . `close` releases the model's NPU weights and is mandatory: eject calls it
/// (INpuServable::Unload) before the owning backend is destroyed. Warmup is
/// best-effort: a failed warmup leaves the model Ready with warmed=false. Open
/// failure returns false and fills `error`. On success the servable is
/// published to the global registry.
inline bool LoadNpuServable(const std::string& id,
                            std::function<bool(std::string*)> open,
                            std::function<bool(std::string*)> warm,
                            std::function<void()> close, std::string* error) {
  namespace npu = mbufo::core::npu;
  auto s = std::make_shared<npu::NpuFnServable>(
      id, [open](const npu::NpuLoadArgs&, std::string* e) { return open(e); },
      std::move(warm), std::move(close));
  npu::NpuLoadArgs args;
  args.warmup = true;
  args.warmup_policy = npu::NpuWarmupPolicy::kBestEffort;
  if (!s->Load(args, error))
    return false;
  s->DetachLoadHooks();  // hooks capture the caller's locals by reference
  const auto st = s->Status();
  Logger::Info("loader",
               "event=npu_servable model=" + id +
                   " state=" + npu::NpuServableStateName(st.state) +
                   " load_ms=" + std::to_string(st.load_ms) +
                   " warm=" + (st.warmed ? "true" : "false") +
                   " warmup_ms=" + std::to_string(st.warmup_ms) +
                   (st.last_error.empty() ? "" : " error=" + st.last_error));
  {
    const std::string& tag = npu::NpuServableRegistry::CurrentTag();
    npu::NpuServableRegistry::Global().Put(tag.empty() ? id : tag,
                                           std::move(s));
  }
  return true;
}

}  // namespace mbufo::server
