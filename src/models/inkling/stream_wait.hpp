#ifndef MBUFO_MODELS_INKLING_STREAM_WAIT_HPP_
#define MBUFO_MODELS_INKLING_STREAM_WAIT_HPP_

#include <chrono>
#include <cstdlib>
#include <thread>

#include <hip/hip_runtime.h>

namespace mbufo::models::inkling {

/// Waits for all work queued on `stream` without burning a core.
///
/// A device-to-host copy into pageable memory (and hipStreamSynchronize) makes
/// the HIP runtime busy-wait on the HSA completion signal, which pins the
/// calling thread (and HIP's helper thread) at 100% CPU for the whole forward.
/// Recording an event and polling it with a short sleep adds at most one
/// sleep interval of latency. MBUFO_SYNC_SLEEP_US=0 restores the busy-wait.
inline void WaitStream(hipStream_t stream) {
  static const int sleep_us = [] {
    const char* e = std::getenv("MBUFO_SYNC_SLEEP_US");
    return e != nullptr ? std::atoi(e) : 50;
  }();
  if (sleep_us <= 0) return;  // caller's own blocking call does the wait
  thread_local hipEvent_t ev = [] {
    hipEvent_t e = nullptr;
    if (hipEventCreateWithFlags(&e, hipEventDisableTiming) != hipSuccess) {
      e = nullptr;
    }
    return e;
  }();
  if (ev == nullptr || hipEventRecord(ev, stream) != hipSuccess) return;
  while (hipEventQuery(ev) == hipErrorNotReady) {
    std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
  }
}

}  // namespace mbufo::models::inkling

#endif  // MBUFO_MODELS_INKLING_STREAM_WAIT_HPP_
