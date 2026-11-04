// Glimmer mixed-batch checks: (1) incremental Sync prefix (append-only
// growth forwards only the delta) matches a full resync; (2)
// Session::DecodeBatch over decode sessions matches serial Evaluate,
// token-for-token; (3) a mixed {1 decode row, 64 chunk rows} DecodeBatch
// matches the serial per-session runs, token-for-token (both item orders);
// (4) short-sentence greedy text is printed for the f16/q8 cross-check.
// Usage: glimmer_decode_batch_test MODEL.gguf [Q8]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/glimmer/engine.hpp"

namespace g = gufo::models::glimmer;

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

std::vector<std::int32_t> GreedyRun(g::Session& s, int steps,
                                     std::string* error) {
  std::vector<std::int32_t> out;
  for (int i = 0; i < steps; ++i) {
    const std::int32_t t = s.SampleGreedy();
    out.push_back(t);
    if (!s.Evaluate(t, error))
      return {};
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: glimmer_decode_batch_test MODEL.gguf [Q8]\n");
    return 2;
  }
  g::ModelOptions options;
  if (argc == 3)
    options.kv_cache_quant = argv[2];
  std::string error;
  auto model = g::Model::Load(argv[1], options, &error);
  if (!model)
    return Fail("load: " + error);
  int rc = 0;

  std::string text;
  for (int i = 0; i < 10; ++i)
    text += "The quick brown fox jumps over the lazy dog. ";
  const auto ids = model->Tokenize(text, /*add_bos=*/false);
  if (ids.size() < 96)
    return Fail("prompt too short");
  const std::vector<std::int32_t> pre(ids.begin(), ids.begin() + 16);
  const std::vector<std::int32_t> full(ids.begin(), ids.begin() + 32);

  // (1) Incremental Sync: Sync(pre)+Sync(full) must equal Sync(full).
  {
    auto s_full = model->CreateSession(&error);
    auto s_inc = model->CreateSession(&error);
    if (!s_full || !s_inc)
      return Fail("session: " + error);
    if (!s_full->Sync(full, &error))
      return Fail("sync full: " + error);
    if (!s_inc->Sync(pre, &error))
      return Fail("sync pre: " + error);
    if (!s_inc->Sync(full, &error))
      return Fail("sync inc: " + error);
    if (s_full->Length() != s_inc->Length()) {
      std::fprintf(stderr, "length mismatch %u vs %u\n", s_full->Length(),
                   s_inc->Length());
      rc = 1;
    }
    const double w0 = MaxAbs(s_full->Logits(), s_inc->Logits());
    std::fprintf(stderr, "[%s] incremental-sync logit maxabs=%.3g\n",
                 options.kv_cache_quant.c_str(), w0);
    if (!(w0 == w0) || w0 > 1e3)
      rc = 1;  // NaN/gross guard; family noise allowed
    for (int step = 0; step < 4; ++step) {
      const auto t_full = s_full->SampleGreedy();
      const auto t_inc = s_inc->SampleGreedy();
      std::fprintf(stderr, "step=%d full=%d inc=%d\n", step, (int)t_full,
                   (int)t_inc);
      if (t_full != t_inc)
        rc = 1;
      if (!s_full->Evaluate(t_full, &error))
        return Fail("eval full");
      if (!s_inc->Evaluate(t_inc, &error))
        return Fail("eval inc");
    }
  }

  // (2) Single-session DecodeBatch vs serial Evaluate (fair: same token).
  {
    auto s_a = model->CreateSession(&error);
    auto s_c = model->CreateSession(&error);
    if (!s_a || !s_c)
      return Fail("session: " + error);
    if (!s_a->Sync(full, &error) || !s_c->Sync(full, &error))
      return Fail("sync: " + error);
    const std::int32_t ta = s_a->SampleGreedy();
    std::vector<g::Session::BatchItem> one = {{s_a.get(), {&ta, 1}}};
    if (!g::Session::DecodeBatch(one, &error))
      return Fail("decodebatch1: " + error);
    if (!s_c->Evaluate(ta, &error))
      return Fail("eval: " + error);
    const double w1 = MaxAbs(s_a->Logits(), s_c->Logits());
    const auto ga = s_a->SampleGreedy();
    const auto gc = s_c->SampleGreedy();
    std::fprintf(stderr, "[%s] single decodebatch maxabs=%.3g greedy=%d/%d %s\n",
                 options.kv_cache_quant.c_str(), w1, (int)ga, (int)gc,
                 (s_a->Length() == s_c->Length()) ? "len-ok" : "LEN-MISMATCH");
    if (s_a->Length() != s_c->Length() || ga != gc)
      rc = 1;
    if (!(w1 == w1) || w1 > 1e3)
      rc = 1;
  }

  // (3) Double DecodeBatch (two decode sessions) vs serial Evaluates.
  {
    auto s_a = model->CreateSession(&error);
    auto s_b = model->CreateSession(&error);
    auto s_c = model->CreateSession(&error);
    auto s_d = model->CreateSession(&error);
    if (!s_a || !s_b || !s_c || !s_d)
      return Fail("session: " + error);
    for (auto* s : {s_a.get(), s_b.get(), s_c.get(), s_d.get()})
      if (!s->Sync(full, &error))
        return Fail("sync: " + error);
    const std::int32_t ta = s_a->SampleGreedy();
    const std::int32_t tb = s_b->SampleGreedy();
    std::vector<g::Session::BatchItem> batch = {{s_a.get(), {&ta, 1}},
                                                {s_b.get(), {&tb, 1}}};
    if (!g::Session::DecodeBatch(batch, &error))
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
                 "[%s] double decodebatch maxabs=%.3g,%.3g greedy=(%d,%d)/(%d,%d) %s\n",
                 options.kv_cache_quant.c_str(), wa, wb, (int)ga, (int)gc,
                 (int)gb, (int)gd, lenok ? "len-ok" : "LEN-MISMATCH");
    if (!lenok || ga != gc || gb != gd)
      rc = 1;
    if (!(wa == wa) || wa > 1e3 || !(wb == wb) || wb > 1e3)
      rc = 1;
  }

  // (4) Mixed {1 decode row, 64 chunk rows} vs serial runs, both orders.
  // The decode item advances one session by one token; the chunk item runs
  // a 64-token prefill chunk on a second session at its own position.
  {
    const std::vector<std::int32_t> ctx(ids.begin(), ids.begin() + 32);
    const std::vector<std::int32_t> chunk64(ids.begin() + 32,
                                             ids.begin() + 96);
    for (int order = 0; order < 2; ++order) {
      auto s_dec = model->CreateSession(&error);
      auto s_ch = model->CreateSession(&error);
      auto s_dec_ref = model->CreateSession(&error);
      auto s_ch_ref = model->CreateSession(&error);
      if (!s_dec || !s_ch || !s_dec_ref || !s_ch_ref)
        return Fail("session: " + error);
      for (auto* s : {s_dec.get(), s_ch.get(), s_dec_ref.get(), s_ch_ref.get()})
        if (!s->Sync(ctx, &error))
          return Fail("sync: " + error);
      const std::int32_t td = s_dec->SampleGreedy();
      // Serial twins first (fresh sessions, same start state).
      if (!s_dec_ref->Evaluate(td, &error))
        return Fail("eval ref: " + error);
      if (!s_ch_ref->Forward(chunk64, s_ch_ref->Length(), &error))
        return Fail("forward ref: " + error);
      // Mixed batch, both orders.
      std::vector<g::Session::BatchItem> mixed;
      if (order == 0) {
        mixed = {{s_dec.get(), {&td, 1}},
                 {s_ch.get(), {chunk64.data(), chunk64.size()}}};
      } else {
        mixed = {{s_ch.get(), {chunk64.data(), chunk64.size()}},
                 {s_dec.get(), {&td, 1}}};
      }
      if (!g::Session::DecodeBatch(mixed, &error))
        return Fail("mixed decodebatch: " + error);
      const double wd = MaxAbs(s_dec->Logits(), s_dec_ref->Logits());
      const double wc = MaxAbs(s_ch->Logits(), s_ch_ref->Logits());
      const auto gd = s_dec->SampleGreedy();
      const auto gd_ref = s_dec_ref->SampleGreedy();
      const auto gc = s_ch->SampleGreedy();
      const auto gc_ref = s_ch_ref->SampleGreedy();
      const bool lenok = (s_dec->Length() == s_dec_ref->Length()) &&
                         (s_ch->Length() == s_ch_ref->Length());
      std::fprintf(stderr,
                   "[%s] mixed order=%d maxabs dec=%.3g chunk=%.3g "
                   "greedy dec=%d/%d chunk=%d/%d %s\n",
                   options.kv_cache_quant.c_str(), order, wd, wc, (int)gd,
                   (int)gd_ref, (int)gc, (int)gc_ref,
                   lenok ? "len-ok" : "LEN-MISMATCH");
      if (!lenok || gd != gd_ref || gc != gc_ref)
        rc = 1;
      if (!(wd == wd) || wd > 1e3 || !(wc == wc) || wc > 1e3)
        rc = 1;
    }
  }

  // (5) Short-sentence greedy anchor (f16/q8 cross-check reads this).
  auto short_greedy = [&](bool legacy, std::vector<std::int32_t>* seq_out,
                          std::string* error) -> bool {
    if (legacy)
      setenv("GUFO_GLIMMER_UNIFIED_BATCH", "0", 1);
    else
      unsetenv("GUFO_GLIMMER_UNIFIED_BATCH");
    auto s = model->CreateSession(error);
    if (!s)
      return false;
    const auto short_ids =
        model->Tokenize("The quick brown fox jumps over the lazy dog.",
                        /*add_bos=*/false);
    if (!s->Sync(short_ids, error))
      return false;
    seq_out->clear();
    for (int i = 0; i < 12; ++i) {
      const std::int32_t t = s->SampleGreedy();
      seq_out->push_back(t);
      if (model->IsStopToken(t))
        break;
      if (!s->Evaluate(t, error))
        return false;
    }
    return true;
  };
  {
    std::vector<std::int32_t> uni, leg;
    if (!short_greedy(false, &uni, &error))
      return Fail("short uni: " + error);
    if (!short_greedy(true, &leg, &error))
      return Fail("short leg: " + error);
    unsetenv("GUFO_GLIMMER_UNIFIED_BATCH");
    std::fprintf(stderr, "[%s] short greedy ids:", options.kv_cache_quant.c_str());
    for (const auto t : uni)
      std::fprintf(stderr, " %d", (int)t);
    std::fprintf(stderr, "\n[%s] short greedy text: %s\n",
                 options.kv_cache_quant.c_str(),
                 model->Decode(uni).c_str());
    if (uni != leg) {
      std::fprintf(stderr, "[%s] CROSS-MODE MISMATCH short:",
                   options.kv_cache_quant.c_str());
      for (const auto t : leg)
        std::fprintf(stderr, " %d", (int)t);
      std::fprintf(stderr, "\n");
      rc = 1;
    } else {
      std::fprintf(stderr, "[%s] cross-mode short tokens identical (%zu)\n",
                   options.kv_cache_quant.c_str(), uni.size());
    }
  }

  // (6) Cross-mode parity: unified vs true-legacy (valve=0) must agree
  // token-for-token on the mixed batch and its greedy continuation.
  {
    const std::vector<std::int32_t> ctx(ids.begin(), ids.begin() + 32);
    const std::vector<std::int32_t> chunk64(ids.begin() + 32,
                                             ids.begin() + 96);
    std::vector<std::int32_t> uni, leg;
    for (int mode = 0; mode < 2; ++mode) {
      if (mode == 0)
        unsetenv("GUFO_GLIMMER_UNIFIED_BATCH");
      else
        setenv("GUFO_GLIMMER_UNIFIED_BATCH", "0", 1);
      auto s_dec = model->CreateSession(&error);
      auto s_ch = model->CreateSession(&error);
      if (!s_dec || !s_ch)
        return Fail("session: " + error);
      if (!s_dec->Sync(ctx, &error) || !s_ch->Sync(ctx, &error))
        return Fail("sync: " + error);
      const std::int32_t td = s_dec->SampleGreedy();
      std::vector<g::Session::BatchItem> mixed = {
          {s_dec.get(), {&td, 1}},
          {s_ch.get(), {chunk64.data(), chunk64.size()}}};
      if (!g::Session::DecodeBatch(mixed, &error))
        return Fail("mixed decodebatch: " + error);
      std::vector<std::int32_t>& seq = (mode == 0) ? uni : leg;
      seq.push_back(s_dec->SampleGreedy());
      seq.push_back(s_ch->SampleGreedy());
      // Greedy continuation on both sessions amplifies any drift.
      for (int i = 0; i < 4; ++i) {
        const std::int32_t a = s_dec->SampleGreedy();
        const std::int32_t b = s_ch->SampleGreedy();
        seq.push_back(a);
        seq.push_back(b);
        if (!s_dec->Evaluate(a, &error) || !s_ch->Evaluate(b, &error))
          return Fail("eval cont: " + error);
      }
    }
    unsetenv("GUFO_GLIMMER_UNIFIED_BATCH");
    const bool same = (uni == leg);
    std::fprintf(stderr, "[%s] cross-mode mixed+cont %s\n",
                 options.kv_cache_quant.c_str(),
                 same ? "tokens identical" : "MISMATCH");
    if (!same) {
      std::fprintf(stderr, "  uni:");
      for (const auto t : uni)
        std::fprintf(stderr, " %d", (int)t);
      std::fprintf(stderr, "\n  leg:");
      for (const auto t : leg)
        std::fprintf(stderr, " %d", (int)t);
      std::fprintf(stderr, "\n");
      rc = 1;
    }
  }

  std::fprintf(stderr, rc == 0 ? "decode_batch_test PASS\n"
                               : "decode_batch_test FAIL\n");
  return rc;
}
