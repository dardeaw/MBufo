// Pure state-machine test for INpuServable (no NPU hardware needed).
#include <cstdlib>
#include <iostream>
#include <string>

#include "src/core/compute/npu/npu_servable.hpp"

using namespace mbufo::core::npu;

namespace {

int g_failures = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " #cond    \
                << "\n";                                                  \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

struct Fake final : INpuServable {
  bool open_ok{true};
  bool warm_ok{true};
  int opens{0}, warms{0}, closes{0};
  std::string_view id() const noexcept override { return "fake"; }

 protected:
  bool Open(const NpuLoadArgs&, std::string* e) override {
    ++opens;
    if (!open_ok) { *e = "no weights"; return false; }
    return true;
  }
  bool Warm(std::string* e) override {
    ++warms;
    if (!warm_ok) { *e = "kernel timeout"; return false; }
    return true;
  }
  void Close() override { ++closes; }
};

}  // namespace

int main() {
  {  // happy path: Ready only after warmup, warmed flag set
    Fake f;
    CHECK(f.Status().state == NpuServableState::kUnloaded);
    std::string err;
    CHECK(f.Load({}, &err));
    CHECK(f.Ready());
    CHECK(f.Status().warmed);
    CHECK(f.opens == 1 && f.warms == 1);
  }
  {  // idempotent: second Load is a no-op
    Fake f;
    std::string err;
    CHECK(f.Load({}, &err));
    CHECK(f.Load({}, &err));
    CHECK(f.opens == 1 && f.warms == 1);
  }
  {  // open failure -> Failed with error, no warmup attempted
    Fake f;
    f.open_ok = false;
    std::string err;
    CHECK(!f.Load({}, &err));
    CHECK(f.Status().state == NpuServableState::kFailed);
    CHECK(err == "no weights");
    CHECK(f.warms == 0 && !f.Ready());
  }
  {  // required warmup failure -> closed + Failed, never Ready
    Fake f;
    f.warm_ok = false;
    std::string err;
    CHECK(!f.Load({}, &err));
    CHECK(f.Status().state == NpuServableState::kFailed);
    CHECK(f.closes == 1);
    CHECK(err.find("kernel timeout") != std::string::npos);
  }
  {  // best-effort warmup failure -> Ready but warmed=false, error kept
    Fake f;
    f.warm_ok = false;
    NpuLoadArgs a;
    a.warmup_policy = NpuWarmupPolicy::kBestEffort;
    std::string err;
    CHECK(f.Load(a, &err));
    CHECK(f.Ready());
    CHECK(!f.Status().warmed);
    CHECK(!f.Status().last_error.empty());
    f.warm_ok = true;  // later rewarm recovers
    CHECK(f.Rewarm(&err));
    CHECK(f.Status().warmed && f.Ready());
  }
  {  // warmup disabled = legacy load-only
    Fake f;
    NpuLoadArgs a;
    a.warmup = false;
    std::string err;
    CHECK(f.Load(a, &err));
    CHECK(f.Ready() && f.warms == 0 && !f.Status().warmed);
  }
  {  // retry after failure works; Unload resets
    Fake f;
    f.open_ok = false;
    std::string err;
    CHECK(!f.Load({}, &err));
    f.open_ok = true;
    CHECK(f.Load({}, &err));
    CHECK(f.Ready());
    f.Unload();
    CHECK(f.Status().state == NpuServableState::kUnloaded);
    CHECK(!f.Rewarm(&err));
  }
  if (g_failures == 0) std::cout << "NPU_SERVABLE_CONTRACT_TESTS_PASS\n";
  return g_failures == 0 ? 0 : 1;
}
