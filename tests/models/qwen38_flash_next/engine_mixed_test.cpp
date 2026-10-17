// Engine-level mixed-batch check (P2): plain advances plus one prefill chunk
// through Session::EvaluateBatchWithChunk vs legacy Sync/Evaluate twins.
// Asserts exact token/position bookkeeping; logits compare report-only
// (kernel-family gaps). A second chunk exercises the DraftCatchUp protocol.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

static double Worst(const std::vector<float>& a, std::span<const float> b) {
  double w = 0.0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    w = std::max(w, std::abs(double(a[i]) - double(b[i])));
  return w;
}

int main(int argc, char** argv) {
  std::string model_path;
  std::string mtp_path;
  std::uint32_t chunk_rows = 64;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() {
      if (++i >= argc)
        throw std::runtime_error(std::string("missing value for ") + arg);
      return std::string(argv[i]);
    };
    if (arg == "--model")
      model_path = next();
    else if (arg == "--mtp-model")
      mtp_path = next();
    else if (arg == "--chunk")
      chunk_rows = static_cast<std::uint32_t>(std::stoul(next()));
    else {
      std::fprintf(stderr, "unknown arg %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "usage: engine_mixed_test --model M [--mtp-model MTP] [--chunk N]\n");
    return 2;
  }
  qfn::ModelOptions options;
  options.max_context = 2048;
  if (!mtp_path.empty())
    options.mtp_model_path = mtp_path;
  std::string error;
  auto model = qfn::Model::Load(model_path, options, &error);
  if (!model) {
    std::fprintf(stderr, "load failed: %s\n", error.c_str());
    return 1;
  }
  const auto mode = !mtp_path.empty() ? gufo::core::SessionMode::kSpeculative
                                      : gufo::core::SessionMode::kAutoregressive;
  auto make = [&](const char* what) {
    auto s = model->CreateSession(mode, 2048, &error);
    if (!s)
      throw std::runtime_error(std::string(what) + ": " + error);
    return s;
  };
  std::string big;
  for (int i = 0; i < 60; ++i)
    big += "The quick brown fox jumps over the lazy dog. ";
  const auto ids = model->Tokenize(big);
  if (ids.size() < 2 * chunk_rows + 16)
    throw std::runtime_error("prompt too short");
  const std::vector<std::int32_t> seed(ids.begin(), ids.begin() + 8);
  const std::vector<std::int32_t> chunk(ids.begin(), ids.begin() + chunk_rows);
  const std::vector<std::int32_t> chunk2(ids.begin() + chunk_rows,
                                         ids.begin() + 2 * chunk_rows);
  auto d0 = make("d0");
  auto d1 = make("d1");
  auto r0 = make("r0");
  auto r1 = make("r1");
  auto sc = make("sc");
  auto rc = make("rc");
  for (auto* s : {d0.get(), d1.get(), r0.get(), r1.get()}) {
    if (!s->Sync(seed, &error))
      throw std::runtime_error("setup sync failed: " + error);
  }
  const std::int32_t tok = seed.back();
  // Reference: legacy chunk Sync + plain Evaluates.
  const auto rt0 = std::chrono::steady_clock::now();
  {
    std::vector<std::int32_t> full(chunk.begin(), chunk.end());
    if (!rc->Sync(full, &error))
      throw std::runtime_error("ref chunk sync failed: " + error);
  }
  for (auto* s : {r0.get(), r1.get()}) {
    if (!s->Evaluate(tok, &error))
      throw std::runtime_error("ref evaluate failed: " + error);
  }
  const auto rt1 = std::chrono::steady_clock::now();
  // Test: one merged call.
  qfn::Session::BatchOutcome o0, o1;
  std::vector<qfn::Session::AdvanceRequest> adv = {{d0.get(), tok, &o0},
                                                  {d1.get(), tok, &o1}};
  const auto bt0 = std::chrono::steady_clock::now();
  if (!qfn::Session::EvaluateBatchWithChunk(adv, sc.get(), chunk, &error))
    throw std::runtime_error("EvaluateBatchWithChunk failed: " + error);
  const auto bt1 = std::chrono::steady_clock::now();
  if (!o0.completed || !o1.completed)
    throw std::runtime_error("advance outcomes incomplete");
  // Exact bookkeeping checks.
  if (sc->Tokens().size() != rc->Tokens().size() ||
      !std::equal(sc->Tokens().begin(), sc->Tokens().end(), rc->Tokens().begin()))
    throw std::runtime_error("chunk tokens diverge from legacy Sync");
  if (sc->Position() != rc->Position())
    throw std::runtime_error("chunk position diverges from legacy Sync");
  if (d0->Tokens().size() != r0->Tokens().size() ||
      d1->Tokens().size() != r1->Tokens().size())
    throw std::runtime_error("decode token counts diverge");
  const std::vector<float> m0(d0->Logits().begin(), d0->Logits().end());
  const std::vector<float> m1(d1->Logits().begin(), d1->Logits().end());
  const double w0 = Worst(m0, r0->Logits());
  const double w1 = Worst(m1, r1->Logits());
  // Continuation: both sides decode one more token (catches bad chunk state).
  const std::int32_t nt = chunk.back();
  if (!sc->Evaluate(nt, &error) || !rc->Evaluate(nt, &error))
    throw std::runtime_error("continuation failed: " + error);
  const std::vector<float> mc(sc->Logits().begin(), sc->Logits().end());
  const double wc = Worst(mc, rc->Logits());
  const double ref_ms =
      std::chrono::duration<double, std::milli>(rt1 - rt0).count();
  const double bat_ms =
      std::chrono::duration<double, std::milli>(bt1 - bt0).count();
  std::cout << "  engine chunk=" << chunk_rows << " decode_logits worst " << w0
            << "/" << w1 << " continuation worst " << wc << "\n";
  std::cout << "  engine ref_ms " << ref_ms << " batch_ms " << bat_ms << "\n";
  // Second chunk: exercises the DraftCatchUp protocol on MTP sessions.
  {
    std::vector<qfn::Session::AdvanceRequest> adv2 = {{d0.get(), tok, &o0},
                                                     {d1.get(), tok, &o1}};
    o0 = {};
    o1 = {};
    std::vector<std::int32_t> full2(rc->Tokens().begin(), rc->Tokens().end());
    full2.insert(full2.end(), chunk2.begin(), chunk2.end());
    if (!rc->Sync(full2, &error))
      throw std::runtime_error("ref chunk2 sync failed: " + error);
    for (auto* s : {r0.get(), r1.get()}) {
      if (!s->Evaluate(tok, &error))
        throw std::runtime_error("ref evaluate2 failed: " + error);
    }
    if (!qfn::Session::EvaluateBatchWithChunk(adv2, sc.get(), chunk2, &error))
      throw std::runtime_error("chunk2 batch failed: " + error);
    if (sc->Tokens().size() != rc->Tokens().size() ||
        !std::equal(sc->Tokens().begin(), sc->Tokens().end(),
                    rc->Tokens().begin()))
      throw std::runtime_error("chunk2 tokens diverge");
    if (sc->Position() != rc->Position())
      throw std::runtime_error("chunk2 position diverges");
    std::cout << "  engine chunk2 tokens+position exact\n";
  }
  // Teacher-forcing top-1 agreement over 10 steps (identical histories):
  // the product-relevant quality signal for kernel-family gaps.
  {
    int agree0 = 0, agree1 = 0;
    std::int32_t t0 = tok, t1 = tok;
    for (int step = 0; step < 10; ++step) {
      std::vector<qfn::Session::AdvanceRequest> ad = {{d0.get(), t0, &o0},
                                                     {d1.get(), t1, &o1}};
      o0 = {};
      o1 = {};
      std::vector<std::int32_t> one(&t0, &t0 + 1);
      if (!qfn::Session::EvaluateBatchWithChunk(ad, sc.get(), one, &error))
        throw std::runtime_error("teacher batch failed: " + error);
      if (!o0.completed || !o1.completed)
        throw std::runtime_error(
            std::string("teacher advance incomplete: ") + o0.error + " / " +
            o1.error);
      auto argmax = [&](const std::vector<float>& v) {
        std::uint32_t b = 0;
        for (std::uint32_t i = 1; i < v.size(); ++i)
          if (v[i] > v[b])
            b = i;
        return static_cast<std::int32_t>(b);
      };
      const std::vector<float> md0(d0->Logits().begin(), d0->Logits().end());
      const std::vector<float> md1(d1->Logits().begin(), d1->Logits().end());
      if (!r0->Evaluate(t0, &error) || !r1->Evaluate(t1, &error))
        throw std::runtime_error("teacher ref failed: " + error);
      const std::vector<float> rd0(r0->Logits().begin(), r0->Logits().end());
      const std::vector<float> rd1(r1->Logits().begin(), r1->Logits().end());
      const auto n0 = argmax(md0), n1 = argmax(md1);
      agree0 += (n0 == argmax(rd0));
      agree1 += (n1 == argmax(rd1));
      t0 = argmax(rd0);
      t1 = argmax(rd1);
    }
    std::cout << "  engine teacher agree " << agree0 << "/10 " << agree1
              << "/10\n";
  }
  if (!std::isfinite(w0 + w1 + wc))
    throw std::runtime_error("non-finite logits");
  std::cout << "ENGINE_MIXED_PASS\n";
  return 0;
}
