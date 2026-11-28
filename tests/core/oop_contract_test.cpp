// OOP contract tests (Phase 1, Tasks 3-4): CPU-only coverage for the new
// shared helpers. No GPU, no model weights.
#include "src/core/engine/attention_contract.hpp"
#include "src/core/engine/engine_backend.hpp"
#include "src/core/quant/parallel.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++g_failures;
  }
}

void TestParallelForTasks() {
  // Empty range: worker never runs.
  int calls = 0;
  mbufo::quant::ParallelForTasks(0, [&](std::size_t) { ++calls; });
  Check(calls == 0, "ParallelForTasks(0) runs no tasks");

  // Exact coverage: every index runs exactly once across threads.
  constexpr std::size_t kN = 1000;
  std::vector<std::atomic<int>> hits(kN);
  for (auto& h : hits) h.store(0, std::memory_order_relaxed);
  mbufo::quant::ParallelForTasks(
      kN, [&](std::size_t i) { hits[i].fetch_add(1, std::memory_order_relaxed); },
      /*max_threads=*/8);
  for (std::size_t i = 0; i < kN; ++i) {
    if (hits[i].load() != 1) {
      Check(false, "ParallelForTasks covers every index exactly once");
      break;
    }
  }

  // Single-thread path and default thread count.
  std::size_t sum = 0;
  mbufo::quant::ParallelForTasks(
      16, [&](std::size_t i) { sum += i; }, /*max_threads=*/1);
  Check(sum == 120, "ParallelForTasks single-thread sum 0..15 == 120");
  sum = 0;
  mbufo::quant::ParallelForTasks(16, [&](std::size_t i) { sum += i; });
  Check(sum == 120, "ParallelForTasks default-thread sum 0..15 == 120");

  // Exception propagation: first failure rethrown after join.
  bool threw = false;
  try {
    mbufo::quant::ParallelForTasks(8, [&](std::size_t i) {
      if (i == 3) throw std::runtime_error("probe");
    });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "ParallelForTasks propagates worker exceptions");
}

void TestValidateChunkAgainstWindow() {
  using mbufo::core::engine::ValidateChunkAgainstWindow;
  Check(ValidateChunkAgainstWindow(512, 1024).empty(),
        "chunk 512 < window 1024 is valid (Inkling production shape)");
  Check(ValidateChunkAgainstWindow(512, 0).empty(),
        "window 0 (full causal) imposes no constraint");
  Check(!ValidateChunkAgainstWindow(1024, 1024).empty(),
        "chunk == window is rejected");
  Check(!ValidateChunkAgainstWindow(2048, 1024).empty(),
        "chunk > window is rejected");
  Check(!ValidateChunkAgainstWindow(0, 1024).empty(),
        "chunk 0 is rejected");
}

void TestClassifySyncPrefix() {
  using mbufo::core::engine::ClassifySyncPrefix;
  using mbufo::core::engine::SyncPrefix;
  const std::vector<std::int32_t> hist = {1, 2, 3, 4};
  const std::vector<std::int32_t> same = {1, 2, 3, 4};
  const std::vector<std::int32_t> ext = {1, 2, 3, 4, 5, 6};
  const std::vector<std::int32_t> div = {1, 2, 9, 9};
  const std::vector<std::int32_t> empty;
  constexpr std::uint32_t kCtx = 32;
  auto span = [](const std::vector<std::int32_t>& v) {
    return std::span<const std::int32_t>(v.data(), v.size());
  };
  Check(ClassifySyncPrefix(span(hist), span(same), kCtx) == SyncPrefix::kIdentical,
        "identical prompt classifies kIdentical");
  Check(ClassifySyncPrefix(span(hist), span(ext), kCtx) == SyncPrefix::kAppend,
        "strict extension classifies kAppend");
  Check(ClassifySyncPrefix(span(hist), span(div), kCtx) == SyncPrefix::kReset,
        "divergence classifies kReset");
  Check(ClassifySyncPrefix(span(hist), span(empty), kCtx) == SyncPrefix::kInvalid,
        "empty prompt classifies kInvalid");
  const std::vector<std::int32_t> big(33, 1);
  Check(ClassifySyncPrefix(span(hist), span(big), kCtx) == SyncPrefix::kInvalid,
        "over-context prompt classifies kInvalid");
  Check(ClassifySyncPrefix(span(empty), span(ext), kCtx) == SyncPrefix::kAppend,
        "empty history with prompt classifies kAppend");
}

}  // namespace

int main() {
  TestParallelForTasks();
  TestValidateChunkAgainstWindow();
  TestClassifySyncPrefix();
  if (g_failures == 0) {
    std::cout << "oop_contract_test: all checks passed\n";
    return 0;
  }
  std::cerr << "oop_contract_test: " << g_failures << " checks failed\n";
  return 1;
}
