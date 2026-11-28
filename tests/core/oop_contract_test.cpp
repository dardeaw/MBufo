// OOP contract tests (Phase 1, Tasks 3-4; Phase 2 dialect markers):
// CPU-only coverage for the new shared helpers. No GPU, no model weights.
#include "src/cli/serve/chat_dialect.hpp"
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

void TestDialectOpensToolCall() {
  using mbufo::server::AtemDialectInstance;
  using mbufo::server::DeepSeekDialectInstance;
  using mbufo::server::GemmaDialectInstance;
  using mbufo::server::InklingDialectInstance;
  using mbufo::server::JsonDialectInstance;
  using mbufo::server::LlamaDialectInstance;
  using mbufo::server::QwenDialectInstance;
  // Json admits every known opener.
  Check(JsonDialectInstance().OpensToolCall("<tool_call> x", 0),
        "json opens on <tool_call>");
  Check(JsonDialectInstance().OpensToolCall("x <tool_calls> y", 2),
        "json opens on dsml marker at offset");
  Check(JsonDialectInstance().OpensToolCall("<｜DSML｜tool_calls>", 0),
        "json opens on dsml marker");
  Check(!JsonDialectInstance().OpensToolCall("plain prose", 0),
        "json does not open on prose");
  // Single-opener dialects.
  Check(QwenDialectInstance().OpensToolCall("<tool_call> x", 0),
        "qwen opens on its marker");
  Check(!QwenDialectInstance().OpensToolCall("<|tool_call> x", 0),
        "qwen does not open on gemma marker");
  Check(LlamaDialectInstance().OpensToolCall("{\"name\": \"f\"}", 0),
        "llama opens on bare JSON object");
  Check(GemmaDialectInstance().OpensToolCall("<|tool_call>call:f{}<tool_call|>", 0),
        "gemma opens on its marker");
  Check(AtemDialectInstance().OpensToolCall("<atem:function_calls>", 0),
        "atem opens on its marker");
  Check(InklingDialectInstance().OpensToolCall("<|message_model|>", 0),
        "inkling opens on its marker");
  // DeepSeek separator-qualified opener.
  Check(DeepSeekDialectInstance().OpensToolCall("\n\n<｜DSML｜tool_calls>", 0),
        "deepseek opens on separator-qualified marker");
  // Grammar formats match the ToolFormat enum each dialect replaces.
  using mbufo::sampling::JsonConstraint;
  Check(JsonDialectInstance().GrammarFormat() == JsonConstraint::ToolFormat::kJson,
        "json grammar format");
  Check(QwenDialectInstance().GrammarFormat() == JsonConstraint::ToolFormat::kQwen,
        "qwen grammar format");
  Check(DeepSeekDialectInstance().GrammarFormat() == JsonConstraint::ToolFormat::kDeepSeek,
        "deepseek grammar format");
  Check(GemmaDialectInstance().GrammarFormat() == JsonConstraint::ToolFormat::kGemma,
        "gemma grammar format");
  Check(AtemDialectInstance().GrammarFormat() == JsonConstraint::ToolFormat::kAtem,
        "atem grammar format");
  Check(LlamaDialectInstance().GrammarFormat() == JsonConstraint::ToolFormat::kLlama,
        "llama grammar format");
  Check(InklingDialectInstance().GrammarFormat() == JsonConstraint::ToolFormat::kInkling,
        "inkling grammar format");
}

}  // namespace

int main() {
  TestParallelForTasks();
  TestValidateChunkAgainstWindow();
  TestClassifySyncPrefix();
  TestDialectOpensToolCall();
  if (g_failures == 0) {
    std::cout << "oop_contract_test: all checks passed\n";
    return 0;
  }
  std::cerr << "oop_contract_test: " << g_failures << " checks failed\n";
  return 1;
}
