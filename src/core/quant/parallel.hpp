#ifndef MBUFO_CORE_QUANT_PARALLEL_HPP_
#define MBUFO_CORE_QUANT_PARALLEL_HPP_

// Shared host-side parallel-for helper (OOP contract Task 4).
//
// Why: load-time weight conversion (BF16/K-quant -> F16, NVFP4 -> Q8_0) and
// mapped-region copies each carried their own ad-hoc std::thread/jthread
// pool. A single 9B-model load regressed to ~125s single-threaded; the
// parallel form is ~14s. New engines must reuse this helper instead of
// growing a third pool.
//
// Contract: tasks are indices [0, n); worker(i) runs exactly once per i,
// possibly on another thread. Exceptions propagate: the first exception is
// rethrown on the caller thread after all workers join. max_threads == 0
// means hardware_concurrency (at least 1). Runs inline when n <= 1 or the
// resolved thread count is 1.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace mbufo::quant {

inline void ParallelForTasks(std::size_t n,
                             const std::function<void(std::size_t)>& worker,
                             unsigned max_threads = 0) {
  if (n == 0) return;
  unsigned threads = max_threads;
  if (threads == 0) {
    threads = std::max<unsigned>(1u, std::thread::hardware_concurrency());
  }
  threads = std::min<unsigned>(threads, static_cast<unsigned>(n));
  if (threads <= 1) {
    for (std::size_t i = 0; i < n; ++i) worker(i);
    return;
  }
  std::atomic<std::size_t> next{0};
  std::exception_ptr first_failure;
  std::mutex failure_mutex;
  std::vector<std::thread> pool;
  pool.reserve(threads - 1);
  auto run = [&] {
    for (;;) {
      const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= n) break;
      try {
        worker(i);
      } catch (...) {
        std::lock_guard lock(failure_mutex);
        if (!first_failure) first_failure = std::current_exception();
        // Keep draining so every worker joins promptly; remaining tasks
        // still run but their failures are superseded by the first.
      }
    }
  };
  for (unsigned t = 1; t < threads; ++t) pool.emplace_back(run);
  run();
  for (auto& th : pool) th.join();
  if (first_failure) std::rethrow_exception(first_failure);
}

}  // namespace mbufo::quant

#endif  // MBUFO_CORE_QUANT_PARALLEL_HPP_
