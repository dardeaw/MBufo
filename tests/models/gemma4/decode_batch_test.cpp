// Gemma 4 runner checks: (1) incremental Sync prefix (append-only growth
// forwards only the delta) matches a full resync; (2) Session::DecodeBatch
// over two sessions matches serial Evaluate, token-for-token.
#include <cmath>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/gemma4/engine.hpp"

namespace g4 = gufo::models::gemma4;

namespace {

int Fail(const std::string& m) {
  std::fprintf(stderr, "decode_batch_test FAIL: %s\n", m.c_str());
  return 1;
}

double MaxAbs(std::span<const float> a, std::span<const float> b) {
  double w = 0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i)
    w = std::max(w, std::abs(double(a[i]) - double(b[i])));
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: gemma4_decode_batch_test MODEL.gguf [Q8]\n");
    return 2;
  }
  g4::ModelOptions options;
  if (argc == 3) options.kv_cache_quant = argv[2];
  std::string error;
  auto model = g4::Model::Load(argv[1], options, &error);
  if (!model) return Fail("load: " + error);
  int rc = 0;

  std::string text;
  for (int i = 0; i < 10; ++i) text += "The quick brown fox jumps. ";
  const auto ids = model->Tokenize(text);
  if (ids.size() < 40) return Fail("prompt too short");
  const std::vector<std::int32_t> pre(ids.begin(), ids.begin() + 16);
  const std::vector<std::int32_t> full(ids.begin(), ids.begin() + 32);

  // (1) Incremental Sync: Sync(pre)+Sync(full) must equal Sync(full).
  {
    auto s_full = model->CreateSession(&error);
    auto s_inc = model->CreateSession(&error);
    if (!s_full || !s_inc) return Fail("session: " + error);
    if (!s_full->Sync(full, &error)) return Fail("sync full: " + error);
    if (!s_inc->Sync(pre, &error)) return Fail("sync pre: " + error);
    if (!s_inc->Sync(full, &error)) return Fail("sync inc: " + error);
    if (s_full->Length() != s_inc->Length()) {
      std::fprintf(stderr, "length mismatch %u vs %u\n", s_full->Length(),
                   s_inc->Length());
      rc = 1;
    }
    const double w0 = MaxAbs(s_full->Logits(), s_inc->Logits());
    std::fprintf(stderr, "incremental-sync logit maxabs=%.3g\n", w0);
    if (!(w0 == w0) || w0 > 1e3) rc = 1;  // NaN/gross guard; family noise allowed
    for (int step = 0; step < 4; ++step) {
      const auto t_full = s_full->SampleGreedy();
      const auto t_inc = s_inc->SampleGreedy();
      std::fprintf(stderr, "step=%d full=%d inc=%d\n", step, (int)t_full,
                   (int)t_inc);
      if (t_full != t_inc) rc = 1;
      if (!s_full->Evaluate(t_full, &error)) return Fail("eval full");
      if (!s_inc->Evaluate(t_inc, &error)) return Fail("eval inc");
    }
  }

  // (2a) Single-session DecodeBatch vs serial Evaluate (fair: same token).
  {
    auto s_a = model->CreateSession(&error);
    auto s_c = model->CreateSession(&error);
    if (!s_a || !s_c) return Fail("session: " + error);
    if (!s_a->Sync(full, &error) || !s_c->Sync(full, &error))
      return Fail("sync: " + error);
    const std::int32_t ta = s_a->SampleGreedy();
    std::vector<g4::Session*> one = {s_a.get()};
    const std::vector<std::int32_t> t1 = {ta};
    if (!g4::Session::DecodeBatch(one, t1, &error))
      return Fail("decodebatch1: " + error);
    if (!s_c->Evaluate(ta, &error)) return Fail("eval: " + error);
    const double w1 = MaxAbs(s_a->Logits(), s_c->Logits());
    const auto ga = s_a->SampleGreedy();
    const auto gc = s_c->SampleGreedy();
    std::fprintf(stderr, "single decodebatch maxabs=%.3g greedy=%d/%d %s\n",
                 w1, (int)ga, (int)gc,
                 (s_a->Length() == s_c->Length()) ? "len-ok" : "LEN-MISMATCH");
    if (s_a->Length() != s_c->Length() || ga != gc) rc = 1;
    if (!(w1 == w1) || w1 > 1e3) rc = 1;
  }

  // (2c) Legacy internal spread: SplitK vs scalar on identical state.
  {
    auto s_e = model->CreateSession(&error);
    auto s_f = model->CreateSession(&error);
    if (!s_e || !s_f) return Fail("session: " + error);
    if (!s_e->Sync(full, &error) || !s_f->Sync(full, &error))
      return Fail("sync: " + error);
    const std::int32_t te = s_e->SampleGreedy();
    if (!s_e->Evaluate(te, &error)) return Fail("eval: " + error);
    setenv("GEMMA4_NO_SPLITK_ATTN", "1", 1);
    setenv("GEMMA4_ATTN_NO_WMMA", "1", 1);
    if (!s_f->Sync(full, &error)) return Fail("sync: " + error);
    const std::int32_t tf = s_f->SampleGreedy();
    if (tf != te) return Fail("tainted sample");
    if (!s_f->Evaluate(tf, &error)) return Fail("eval: " + error);
    unsetenv("GEMMA4_NO_SPLITK_ATTN");
    unsetenv("GEMMA4_ATTN_NO_WMMA");
    const double w = MaxAbs(s_e->Logits(), s_f->Logits());
    std::fprintf(stderr, "legacy splitk-vs-scalar maxabs=%.3g greedy=%d/%d\n",
                 w, (int)s_e->SampleGreedy(), (int)s_f->SampleGreedy());
    if (!(w == w) || w > 1e3) rc = 1;
  }
  {
    auto s_a = model->CreateSession(&error);
    auto s_b = model->CreateSession(&error);
    auto s_c = model->CreateSession(&error);
    auto s_d = model->CreateSession(&error);
    if (!s_a || !s_b || !s_c || !s_d) return Fail("session: " + error);
    for (auto* s : {s_a.get(), s_b.get(), s_c.get(), s_d.get()})
      if (!s->Sync(full, &error)) return Fail("sync: " + error);
    const std::int32_t ta = s_a->SampleGreedy();
    const std::int32_t tb = s_b->SampleGreedy();
    std::vector<g4::Session*> batch = {s_a.get(), s_b.get()};
    const std::vector<std::int32_t> toks = {ta, tb};
    if (!g4::Session::DecodeBatch(batch, toks, &error))
      return Fail("decodebatch: " + error);
    if (!s_c->Evaluate(ta, &error) || !s_d->Evaluate(tb, &error))
      return Fail("eval: " + error);
    const double wa = MaxAbs(s_a->Logits(), s_c->Logits());
    const double wb = MaxAbs(s_b->Logits(), s_d->Logits());
    const auto ga = s_a->SampleGreedy();
    const auto gc = s_c->SampleGreedy();
    const auto gb = s_b->SampleGreedy();
    const auto gd = s_d->SampleGreedy();
    const bool lenok =
        (s_a->Length() == s_c->Length()) && (s_b->Length() == s_d->Length());
    std::fprintf(stderr,
                 "double decodebatch maxabs=%.3g,%.3g greedy=(%d,%d)/(%d,%d) %s\n",
                 wa, wb, (int)ga, (int)gc, (int)gb, (int)gd,
                 lenok ? "len-ok" : "LEN-MISMATCH");
    if (!lenok || ga != gc || gb != gd) rc = 1;
    if (!(wa == wa) || wa > 1e3 || !(wb == wb) || wb > 1e3) rc = 1;
  }
  std::fprintf(stderr, rc == 0 ? "decode_batch_test PASS\n"
                               : "decode_batch_test FAIL\n");
  return rc;
}
