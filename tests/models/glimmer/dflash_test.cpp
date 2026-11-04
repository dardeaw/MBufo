// Glimmer DFlash end-to-end: short-sentence greedy text with the draft on
// must equal the autoregressive baseline, plus acceptance rate and tok/s.
// The draft is EAGLE-style: target tapped-layer features are captured
// during Sync (prompt-first priming) and per round the commit (accepted +
// bonus rows) is appended back to the draft.
// Usage: glimmer_dflash_test TARGET.gguf DRAFT.gguf [N_DRAFT=4]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/speculative/draft_registry.hpp"
#include "src/models/glimmer/draft_backend.hpp"
#include "src/models/glimmer/engine.hpp"

namespace g = gufo::models::glimmer;

namespace {

int Fail(const std::string& m) {
  std::fprintf(stderr, "dflash_test FAIL: %s\n", m.c_str());
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::fprintf(stderr, "usage: glimmer_dflash_test TARGET.gguf DRAFT.gguf [N_DRAFT] [N_STEPS]\n");
    return 2;
  }
  std::uint32_t n_draft = 4;
  if (argc >= 4)
    n_draft = static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10));
  if (n_draft == 0 || n_draft > 8)
    return Fail("N_DRAFT must be 1..8");
  int kSteps = 24;
  if (argc == 5)
    kSteps = std::atoi(argv[4]);
  if (kSteps <= 0 || kSteps > 256)
    return Fail("N_STEPS must be 1..256");
  int rc = 0;

  std::string error;
  g::ModelOptions options;
  auto model = g::Model::Load(argv[1], options, &error);
  if (!model)
    return Fail("target load: " + error);
  if (!gufo::speculative::DraftBackendRegistry::Instance().Contains(
          "glimmer-dflash"))
    return Fail("glimmer-dflash backend is not registered");

  // DFLASH_LONG=1: repeat the sentence to a ~2K-token prompt.
  std::string prompt_text = "The quick brown fox jumps over the lazy dog.";
  if (std::getenv("DFLASH_LONG") != nullptr) {
    std::string rep;
    for (int i = 0; i < 200; ++i)
      rep += prompt_text + " ";
    prompt_text = rep;
  }
  const auto prompt = model->Tokenize(prompt_text, /*add_bos=*/false);
  std::fprintf(stderr, "prompt tokens: %zu\n", prompt.size());

  // (1) Autoregressive baseline (temperature 0).
  std::vector<std::int32_t> base_seq;
  double base_prefill_s = 0.0, base_decode_s = 0.0;
  {
    auto s = model->CreateSession(&error);
    if (!s)
      return Fail("session: " + error);
    auto t0 = std::chrono::steady_clock::now();
    if (!s->Sync(prompt, &error))
      return Fail("sync: " + error);
    base_prefill_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kSteps; ++i) {
      const std::int32_t t = s->SampleGreedy();
      base_seq.push_back(t);
      if (!s->Evaluate(t, &error))
        return Fail("eval: " + error);
    }
    base_decode_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
  }

  // (2) DFlash speculative loop through the registered backend.
  std::fprintf(stderr, "[ckpt] creating backend\n");
  std::vector<std::int32_t> spec_seq;
  double spec_prefill_s = 0.0, spec_decode_s = 0.0;
  std::uint64_t drafted = 0, accepted = 0;
  std::uint64_t rounds = 0;
  {
    gufo::speculative::DraftBackendRequest request;
    request.target_kind = "glimmer";
    request.target_model = model;
    request.draft_model_path = argv[2];
    request.max_draft_tokens = n_draft;
    auto backend = gufo::speculative::CreateDraftBackend(
        "glimmer-dflash", request, &error);
    if (!backend)
      return Fail("backend create: " + error);
    auto* dflash =
        static_cast<g::GlimmerDflashDraftBackend*>(backend.get());
    auto s = model->CreateSession(&error);
    if (!s)
      return Fail("session: " + error);
    const std::vector<std::uint32_t> taps(
        dflash->TargetHiddenLayerIds().begin(),
        dflash->TargetHiddenLayerIds().end());
    if (!s->EnableDraftTaps(taps, &error))
      return Fail("taps: " + error);
    // Prompt Sync with a tap sink assembling per-layer host rows.
    const std::uint32_t hidden = model->weights().config.hidden_size;
    std::vector<std::vector<float>> prompt_rows(taps.size());
    g::Session::TapSink psink;
    psink.layers = taps;
    psink.fn = [&](std::uint32_t il, const float* d_rows, std::uint32_t n,
                   std::uint32_t start) {
      (void)start;
      for (std::size_t t = 0; t < taps.size(); ++t) {
        if (taps[t] == il) {
          const std::size_t was = prompt_rows[t].size();
          prompt_rows[t].resize(was +
                                static_cast<std::size_t>(n) * hidden);
          if (hipMemcpy(prompt_rows[t].data() + was, d_rows,
                        static_cast<std::size_t>(n) * hidden * sizeof(float),
                        hipMemcpyDeviceToHost) != hipSuccess) {
            std::fprintf(stderr, "dflash_test tap copy failed\n");
          }
        }
      }
    };
    auto t0 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[ckpt] sync with taps\n");
    if (!s->Sync(prompt, &error, &psink))
      return Fail("sync: " + error);
    std::fprintf(stderr, "[ckpt] sync done\n");
    spec_prefill_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    // Interleave per-layer rows into [P][n_taps*hidden] feature matrix.
    std::vector<float> prompt_feats;
    if (!prompt_rows.empty() && !prompt_rows[0].empty()) {
      prompt_feats.reserve(prompt.size() * taps.size() * hidden);
      for (std::size_t r = 0; r < prompt.size(); ++r) {
        for (std::size_t t = 0; t < taps.size(); ++t) {
          const float* row = prompt_rows[t].data() +
                             static_cast<std::size_t>(r) * hidden;
          prompt_feats.insert(prompt_feats.end(), row, row + hidden);
        }
      }
    }
    std::vector<gufo::tokenization::TokenId> ptoks;
    ptoks.reserve(prompt.size());
    for (const auto t : prompt)
      ptoks.push_back(static_cast<gufo::tokenization::TokenId>(t));
    gufo::speculative::DraftTargetContext prime;
    prime.prompt_tokens = ptoks;
    prime.prompt_hidden_states = prompt_feats;
    prime.hidden_size = taps.size() * hidden;
    prime.first_token = ptoks.back();
    if (!dflash->PrimeTargetContext(prime))
      return Fail("prime failed");
    std::fprintf(stderr, "[ckpt] primed, entering rounds\n");
    std::vector<std::int32_t> context(prompt.begin(), prompt.end());
    t0 = std::chrono::steady_clock::now();
    while (spec_seq.size() < static_cast<std::size_t>(kSteps)) {
      std::vector<std::int32_t> out;
      if (!s->SpeculativeRound(*backend, context, n_draft, &out, &error))
        return Fail("round: " + error);
      ++rounds;
      if (rounds <= 3 || rounds % 10 == 0)
        std::fprintf(stderr, "[ckpt] round %llu emitted=%zu\n",
                     (unsigned long long)rounds, out.size());
      for (const auto t : out) {
        context.push_back(t);
        spec_seq.push_back(t);
        if (spec_seq.size() >= static_cast<std::size_t>(kSteps))
          break;
      }
    }
    spec_decode_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    drafted = dflash->drafted();
    accepted = dflash->accepted();
    if (context.size() != s->Length())
      return Fail("context/session length desync");
  }

  const bool same = (base_seq == spec_seq);
  std::fprintf(stderr, "baseline ids:");
  for (const auto t : base_seq)
    std::fprintf(stderr, " %d", (int)t);
  std::fprintf(stderr, "\nspec ids:");
  for (const auto t : spec_seq)
    std::fprintf(stderr, " %d", (int)t);
  std::fprintf(stderr, "\nbaseline text: %s\n",
               model->Decode(base_seq).c_str());
  std::fprintf(stderr, "spec text: %s\n", model->Decode(spec_seq).c_str());
  const double accept_rate =
      drafted != 0 ? (double)accepted / (double)drafted : 0.0;
  const double base_tps =
      base_decode_s > 0 ? base_seq.size() / base_decode_s : 0.0;
  const double spec_tps =
      spec_decode_s > 0 ? spec_seq.size() / spec_decode_s : 0.0;
  std::fprintf(stderr,
               "rounds=%llu drafted=%llu accepted=%llu accept_rate=%.3f\n",
               (unsigned long long)rounds, (unsigned long long)drafted,
               (unsigned long long)accepted, accept_rate);
  std::fprintf(stderr, "prefill tok/s: base=%.1f spec=%.1f | decode tok/s: base=%.2f spec=%.2f speedup=%.2fx\n",
               prompt.size() / (base_prefill_s > 0 ? base_prefill_s : 1.0),
               prompt.size() / (spec_prefill_s > 0 ? spec_prefill_s : 1.0),
               base_tps, spec_tps,
               (base_tps > 0 && spec_tps > 0) ? spec_tps / base_tps : 0.0);
  if (!same) {
    std::fprintf(stderr, "dflash_test TEXT MISMATCH\n");
    rc = 1;
  }
  std::fprintf(stderr, rc == 0 ? "dflash_test PASS\n" : "dflash_test FAIL\n");
  return rc;
}
