// Mixed-batch vs single-run token agreement for Qwen 27B prefill.
//
// Drives ForwardPromptBatchMulti with {1 decode-equivalent row, 1x64-row
// prefill chunk} (both orders) and requires token-identical next-token
// predictions vs the serial ForwardPromptBatch twins.
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

int Fail(const std::string& m) {
  std::fprintf(stderr, "mixed_batch_test FAIL: %s\n", m.c_str());
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: qwen27b_mixed_batch_test MODEL.gguf\n");
    return 2;
  }
  std::string error;
  auto raw = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!raw) return Fail("open reader: " + error);
  std::shared_ptr<const gufo::core::GgufReader> reader(std::move(raw));
  if (!reader) return Fail("open reader: " + error);
  auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
  if (!model) return Fail("model: " + error);
  auto ex_a = std::make_unique<gufo::hip::QwenGpuExecutor>(model);
  auto ex_b = std::make_unique<gufo::hip::QwenGpuExecutor>(model);

  std::string text;
  for (int i = 0; i < 12; ++i)
    text += "The quick brown fox jumps over the lazy dog. ";
  const auto ids = ex_a->GetTokenizer().Encode(text);
  if (ids.size() < 70) return Fail("prompt too short");
  const std::vector<gufo::tokenization::TokenId> one(ids.begin(),
                                                     ids.begin() + 1);
  const std::vector<gufo::tokenization::TokenId> chunk(ids.begin(),
                                                       ids.begin() + 64);

  const auto t_one = ex_a->ForwardPromptBatch(one, 0, /*compute_logits=*/true);
  const auto t_chunk =
      ex_b->ForwardPromptBatch(chunk, 0, /*compute_logits=*/true);

  int rc = 0;
  // Same-shape control + per-layer bisection via prompt-hidden taps.
  {
    auto e1 = std::make_unique<gufo::hip::QwenGpuExecutor>(model);
    auto e2 = std::make_unique<gufo::hip::QwenGpuExecutor>(model);
    auto e3 = std::make_unique<gufo::hip::QwenGpuExecutor>(model);
    const std::vector<std::uint32_t> taps = {0, 1, 2, 3};
    e1->SetPromptHiddenCapture(true, taps);
    e2->SetPromptHiddenCapture(true, taps);
    e3->SetPromptHiddenCapture(true, taps);
    const auto t_single = e3->ForwardPromptBatch(chunk, 0, true);
    gufo::hip::QwenPromptBatchItem items[2] = {{e1.get(), chunk, 0, true},
                                               {e2.get(), chunk, 0, true}};
    const auto out =
        gufo::hip::QwenGpuExecutor::ForwardPromptBatchMulti(items);
    std::fprintf(stderr, "same-shape single=%d mixed=(%d,%d)\n", (int)t_single,
                 (int)out[0], (int)out[1]);
    const auto v_single = e3->GetPromptHiddenStates();
    const auto v_mixed = e1->GetPromptHiddenStates();
    std::fprintf(stderr, "capture sizes %zu vs %zu\n", v_single.size(),
                 v_mixed.size());
    const std::size_t hidden = 5120;
    for (std::size_t tap = 0; tap < taps.size(); ++tap) {
      double w = 0;
      for (std::size_t r = 0; r < 64; ++r) {
        for (std::size_t k = 0; k < hidden; ++k) {
          const std::size_t idx = (r * taps.size() + tap) * hidden + k;
          if (idx < v_single.size() && idx < v_mixed.size())
            w = std::max(w, std::abs(double(v_single[idx]) -
                                     double(v_mixed[idx])));
        }
      }
      std::fprintf(stderr, "layer %u maxabs=%.3g\n", taps[tap], w);
    }
    if (out.size() != 2 || out[0] != out[1] || out[0] != t_single) rc = 1;
  }
  for (int order = 0; order < 2; ++order) {
    // Mixed orders over fresh sessions per order.
    auto ey = std::make_unique<gufo::hip::QwenGpuExecutor>(model);
    auto ez = std::make_unique<gufo::hip::QwenGpuExecutor>(model);
    gufo::hip::QwenPromptBatchItem items[2];
    if (order == 0) {
      items[0] = {ey.get(), one, 0, true};
      items[1] = {ez.get(), chunk, 0, true};
    } else {
      items[0] = {ez.get(), chunk, 0, true};
      items[1] = {ey.get(), one, 0, true};
    }
    const auto out =
        gufo::hip::QwenGpuExecutor::ForwardPromptBatchMulti(items);
    const auto want0 = (order == 0) ? t_one : t_chunk;
    const auto want1 = (order == 0) ? t_chunk : t_one;
    std::fprintf(stderr, "order=%d single=(%d,%d) mixed=(%d,%d)\n", order,
                 (int)want0, (int)want1, (int)out[0], (int)out[1]);
    if (out.size() != 2 || out[0] != want0 || out[1] != want1) {
      std::fprintf(stderr, "mixed_batch_test MISMATCH order=%d\n", order);
      rc = 1;
    }
  }
  std::fprintf(stderr, rc == 0 ? "mixed_batch_test PASS\n"
                               : "mixed_batch_test FAIL\n");
  return rc;
}
