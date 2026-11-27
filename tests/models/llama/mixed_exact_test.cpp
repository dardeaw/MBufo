// Llama batched-lane / merged-chunk bit-exactness (needs weights + GPU; explicit-only).
// Usage: llama_mixed_exact_test MODEL.gguf [kv=f16|q8] [prompt_tokens=1500]
//  T1 Extend over assorted chunk grids == Sync(full prompt) (logits bytes, then 3 decode steps)
//  T2 StepBatch (4 lanes, different lengths) == serial Evaluate, 24 greedy steps
//  T3 DecodeBatchWithChunk == StepBatch + Extend of the same chunk, chunk grid walk
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "src/models/llama/engine.hpp"

namespace g = mbufo::models::llama;
using Lane = g::Session::LaneItem;

namespace {
int failures = 0;
using Logits = std::vector<float>;

void Check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++failures;
}
Logits Snap(g::Session& s) {
  auto l = s.Logits();
  return Logits(l.begin(), l.end());
}
bool Same(const Logits& a, const Logits& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
std::vector<std::int32_t> MakePrompt(g::Model& m, int n, int salt) {
  std::string text;
  for (int i = 0; text.size() < static_cast<std::size_t>(n) * 6; ++i)
    text += "Segment " + std::to_string(i * 7 + salt) + ": the quick brown fox number " + std::to_string(i * salt) +
            " jumps over lazy dogs near river bank " + std::to_string(i % 13) + ". ";
  auto t = m.Tokenize(text, true);
  t.resize(n);
  return t;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  g::ModelOptions opt;
  opt.max_context = 4096;
  if (argc > 2 && std::string(argv[2]) == "q8") opt.kv_cache_quant = "q8";
  const int N = argc > 3 ? std::atoi(argv[3]) : 1500;
  std::string err;
  auto model = g::Model::Load(argv[1], opt, &err);
  if (!model) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 2; }
  std::printf("packed_exact=%d kv=%s N=%d\n", model->PackedLanesExact() ? 1 : 0, opt.kv_cache_quant.c_str(), N);

  // ---- T1 ----
  {
    const auto prompt = MakePrompt(*model, N, 3);
    auto ref = model->CreateSession(&err);
    if (!ref || !ref->Sync(prompt, &err)) { std::fprintf(stderr, "sync %s\n", err.c_str()); return 2; }
    const Logits ref_logits = Snap(*ref);
    std::vector<Logits> ref_steps;
    std::int32_t tok = ref->SampleGreedy();
    for (int i = 0; i < 3; ++i) {
      if (!ref->Evaluate(tok, &err)) return 2;
      ref_steps.push_back(Snap(*ref));
      tok = ref->SampleGreedy();
    }
    ref.reset();
    const std::vector<std::vector<int>> budgets = {{512}, {300}, {128}, {65}, {64}, {63}, {48}, {33}, {16}, {1}, {64, 65, 191, 70, 333}, {1000, 500}, {400, 64, 7}, {100000}};
    for (const auto& budget : budgets) {
      auto s = model->CreateSession(&err);
      std::size_t reported = 0, bi = 0;
      bool ok = true;
      while (reported < prompt.size() && ok) {
        const std::size_t prev = reported;
        reported = std::min<std::size_t>(prompt.size(), reported + budget[bi++ % budget.size()]);
        const std::size_t from = prev == 0 ? 0 : s->Length();
        const std::size_t to = std::max<std::size_t>(from, g::Session::PrefillTarget(reported, prompt.size()));
        if (to > from) ok = s->Extend(std::span<const std::int32_t>(prompt).first(to), from, &err);
      }
      bool same = ok && Same(Snap(*s), ref_logits);
      std::int32_t t = ok ? s->SampleGreedy() : 0;
      for (int i = 0; i < 3 && same; ++i) {
        ok = s->Evaluate(t, &err);
        same = ok && Same(Snap(*s), ref_steps[i]);
        t = s->SampleGreedy();
      }
      Check(same, "T1 budget walk " + std::to_string(budget[0]) + (budget.size() > 1 ? "+..." : "") + " == Sync");
    }
  }  // ---- T2 ----
  {
    constexpr int kLanes = 4;
    const int lens[kLanes] = {300, 517, 700, 1100};
    std::vector<std::vector<Logits>> serial(24);
    {
      std::vector<std::unique_ptr<g::Session>> ss;
      std::vector<std::int32_t> toks(kLanes);
      for (int i = 0; i < kLanes; ++i) {
        ss.push_back(model->CreateSession(&err));
        if (!ss[i]->Sync(MakePrompt(*model, lens[i], 11 + i), &err)) return 2;
        toks[i] = ss[i]->SampleGreedy();
      }
      for (int st = 0; st < 24; ++st)
        for (int i = 0; i < kLanes; ++i) {
          if (!ss[i]->Evaluate(toks[i], &err)) return 2;
          serial[st].push_back(Snap(*ss[i]));
          toks[i] = ss[i]->SampleGreedy();
        }
    }
    for (int width : {2, 3, 4}) {
      std::vector<std::unique_ptr<g::Session>> ss;
      std::vector<std::int32_t> toks(kLanes);
      for (int i = 0; i < width; ++i) {
        ss.push_back(model->CreateSession(&err));
        if (!ss[i]->Sync(MakePrompt(*model, lens[i], 11 + i), &err)) return 2;
        toks[i] = ss[i]->SampleGreedy();
      }
      bool same = true;
      for (int st = 0; st < 24 && same; ++st) {
        std::vector<Lane> lanes;
        for (int i = 0; i < width; ++i) lanes.push_back({ss[i].get(), toks[i]});
        bool refused = false;
        if (!g::Session::StepBatch(lanes, &err, &refused)) { std::printf("step fail %s refused=%d\n", err.c_str(), refused); same = false; break; }
        for (int i = 0; i < width; ++i) {
          same = same && Same(Snap(*ss[i]), serial[st][i]);
          toks[i] = ss[i]->SampleGreedy();
        }
      }
      Check(same, "T2 StepBatch width " + std::to_string(width) + " == serial Evaluate (24 steps)");
    }
  }
  // ---- T3 ----
  {
    const auto cprompt = MakePrompt(*model, N, 29);
    auto cref = model->CreateSession(&err);
    if (!cref->Sync(cprompt, &err)) return 2;
    const Logits cref_logits = Snap(*cref);
    cref.reset();
    constexpr int kLanes = 3;
    const int lens[kLanes] = {200, 410, 900};
    std::vector<std::vector<Logits>> serial(16);
    {
      std::vector<std::unique_ptr<g::Session>> ss;
      std::vector<std::int32_t> toks(kLanes);
      for (int i = 0; i < kLanes; ++i) {
        ss.push_back(model->CreateSession(&err));
        if (!ss[i]->Sync(MakePrompt(*model, lens[i], 41 + i), &err)) return 2;
        toks[i] = ss[i]->SampleGreedy();
      }
      for (int st = 0; st < 16; ++st)
        for (int i = 0; i < kLanes; ++i) {
          if (!ss[i]->Evaluate(toks[i], &err)) return 2;
          serial[st].push_back(Snap(*ss[i]));
          toks[i] = ss[i]->SampleGreedy();
        }
    }
    for (int chunk_rows : {256, 100, 37}) {
      std::vector<std::unique_ptr<g::Session>> ss;
      std::vector<std::int32_t> toks(kLanes);
      for (int i = 0; i < kLanes; ++i) {
        ss.push_back(model->CreateSession(&err));
        if (!ss[i]->Sync(MakePrompt(*model, lens[i], 41 + i), &err)) return 2;
        toks[i] = ss[i]->SampleGreedy();
      }
      auto cs = model->CreateSession(&err);
      std::size_t reported = 0;
      bool same = true;
      int step = 0;
      while (step < 16 && same) {
        std::vector<Lane> lanes;
        for (int i = 0; i < kLanes; ++i) lanes.push_back({ss[i].get(), toks[i]});
        bool refused = false, adv = false;
        if (reported < cprompt.size()) {
          const std::size_t prev = reported;
          reported = std::min<std::size_t>(cprompt.size(), reported + chunk_rows);
          const std::size_t from = prev == 0 ? 0 : cs->Length();
          const std::size_t to = std::max<std::size_t>(from, g::Session::PrefillTarget(reported, cprompt.size()));
          g::Session::ChunkItem item{cs.get(), std::span<const std::int32_t>(cprompt).first(to), from};
          if (!g::Session::DecodeBatchWithChunk(lanes, item, &err, &refused, &adv)) { std::printf("mixed fail %s refused=%d\n", err.c_str(), refused); same = false; break; }
        } else if (!g::Session::StepBatch(lanes, &err)) { same = false; break; }
        for (int i = 0; i < kLanes; ++i) {
          same = same && Same(Snap(*ss[i]), serial[step][i]);
          toks[i] = ss[i]->SampleGreedy();
        }
        ++step;
      }
      while (same && reported < cprompt.size()) {  // finish the walk if decode steps ran out first
        const std::size_t prev = reported;
        reported = std::min<std::size_t>(cprompt.size(), reported + chunk_rows);
        const std::size_t from = prev == 0 ? 0 : cs->Length();
        const std::size_t to = std::max<std::size_t>(from, g::Session::PrefillTarget(reported, cprompt.size()));
        if (to > from) same = cs->Extend(std::span<const std::int32_t>(cprompt).first(to), from, &err);
      }      Check(same && Same(Snap(*cs), cref_logits), "T3 mixed chunk " + std::to_string(chunk_rows) + " rows: lanes == serial and chunk == Sync");
    }
  }
  std::printf("failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}
