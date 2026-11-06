// GPT-OSS throughput probe: times prefill (Sync) and greedy decode
// (Evaluate) separately and prints tok/s. Temperature is 0 by construction
// (SampleGreedy only). Usage:
//   gptoss_bench_test MODEL.gguf [PROMPT_TOKENS=256] [GEN_TOKENS=32]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/gptoss/engine.hpp"

namespace g = gufo::models::gptoss;

namespace {

int Fail(const std::string& m) {
  std::fprintf(stderr, "bench FAIL: %s\n", m.c_str());
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::fprintf(stderr, "usage: %s MODEL.gguf [PROMPT_TOKENS] [GEN_TOKENS]\n",
                 argv[0]);
    return 2;
  }
  int prompt_tokens = 256;
  int gen_tokens = 32;
  if (argc >= 3) prompt_tokens = std::atoi(argv[2]);
  if (argc >= 4) gen_tokens = std::atoi(argv[3]);
  if (prompt_tokens <= 0 || prompt_tokens > 32768)
    return Fail("PROMPT_TOKENS 1..32768");
  if (gen_tokens <= 0 || gen_tokens > 256) return Fail("GEN_TOKENS 1..256");

  const std::string model_path = argv[1];
  g::ModelOptions options;
  options.max_context =
      static_cast<std::uint32_t>(prompt_tokens + gen_tokens + 64);
  std::string error;
  auto model = g::Model::Load(model_path, options, &error);
  if (!model) return Fail("load: " + error);
  auto session = model->CreateSession(&error);
  if (!session) return Fail("session: " + error);

  // Build a prompt of exactly prompt_tokens ids by repeating one sentence.
  const std::string sentence =
      "The quick brown fox jumps over the lazy dog. Pack my box with five "
      "dozen liquor jugs. ";
  const auto piece = model->Tokenize(sentence, /*add_bos=*/false);
  if (piece.empty()) return Fail("empty sentence tokenization");
  std::vector<std::int32_t> prompt;
  prompt.reserve(static_cast<std::size_t>(prompt_tokens));
  while (prompt.size() < static_cast<std::size_t>(prompt_tokens)) {
    for (std::int32_t id : piece) {
      if (prompt.size() >= static_cast<std::size_t>(prompt_tokens)) break;
      prompt.push_back(id);
    }
  }

  (void)hipDeviceSynchronize();
  const auto t0 = std::chrono::steady_clock::now();
  if (!session->Sync(prompt, &error)) return Fail("sync: " + error);
  (void)hipDeviceSynchronize();
  const auto t1 = std::chrono::steady_clock::now();
  const double prefill_s =
      std::chrono::duration<double>(t1 - t0).count();
  const double prefill_ts = prompt.size() / (prefill_s > 0 ? prefill_s : 1e-9);

  std::vector<std::int32_t> out;
  out.reserve(static_cast<std::size_t>(gen_tokens));
  const auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < gen_tokens; ++i) {
    const std::int32_t tok = session->SampleGreedy();
    out.push_back(tok);
    if (model->IsStopToken(tok)) break;
    if (!session->Evaluate(tok, &error)) return Fail("eval: " + error);
  }
  (void)hipDeviceSynchronize();
  const auto t3 = std::chrono::steady_clock::now();
  const double decode_s = std::chrono::duration<double>(t3 - t2).count();
  const double decode_ts = out.size() / (decode_s > 0 ? decode_s : 1e-9);

  const std::string head =
      model->Decode(std::span<const std::int32_t>(out.data(), out.size()));
  std::fprintf(stderr,
               "[gptoss-bench] model=%s prompt_tok=%zu gen_tok=%zu "
               "prefill_s=%.3f prefill_tok_s=%.2f "
               "decode_s=%.3f decode_tok_s=%.2f\n",
               model->ModelName().c_str(), prompt.size(), out.size(),
               prefill_s, prefill_ts, decode_s, decode_ts);
  std::fprintf(stderr, "[gptoss-bench] ids:");
  for (std::int32_t id : out) std::fprintf(stderr, " %d", id);
  std::fprintf(stderr, "\n[gptoss-bench] text head: %.120s\n", head.c_str());
  std::fprintf(stderr, "bench PASS\n");
  return 0;
}
