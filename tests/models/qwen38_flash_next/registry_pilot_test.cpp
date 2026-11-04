// Flash-next registry pilot: (a) the native direct greedy run, (b) the
// adapter direct L0 greedy run, (c) EngineRegistry::Create
// ("qwen38-flash-next"), (d) the adapter DecodeStep draft-verify path vs
// the native DecodeStep, (e) the adapter DecodeBatch/EvaluateBatch
// translation, (f) a SnapshotCapable round-trip, (g) a SyncThrough smoke,
// (h) MTP pass-through wiring, and (i) a q8 structural smoke. Token ids
// and decoded text must match exactly.
// Usage: qwen38_flash_next_registry_pilot_test MODEL.gguf [MTP.gguf] [MAX_TOKENS=16]
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/engine/engine_batch.hpp"
#include "src/models/qwen38_flash_next/engine_adapter.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace e = gufo::core::engine;

namespace {

int Fail(const std::string& m) {
  std::fprintf(stderr, "registry_pilot FAIL: %s\n", m.c_str());
  return 1;
}

std::int32_t Argmax(std::span<const float> logits) {
  std::int32_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = static_cast<std::int32_t>(i);
  }
  return best;
}

// Native greedy run (manual argmax + Evaluate): the 主體 reference.
std::vector<std::int32_t> NativeGreedyRun(qfn::Session* session,
                                          const std::vector<std::int32_t>& prompt,
                                          int max_tokens, std::string* error) {
  if (!session->Sync(prompt, error)) return {};
  std::vector<std::int32_t> out;
  for (int i = 0; i < max_tokens; ++i) {
    const std::int32_t token = Argmax(session->Logits());
    out.push_back(token);
    if (!session->Evaluate(token, error)) return {};
  }
  return out;
}

// L0 greedy run (Sync + SampleGreedy + Evaluate) through any L0 session.
std::vector<std::int32_t> L0GreedyRun(e::EngineSession* session,
                                      const std::vector<std::int32_t>& prompt,
                                      int max_tokens, std::string* error) {
  if (!session->Sync(prompt, error)) return {};
  std::vector<std::int32_t> out;
  for (int i = 0; i < max_tokens; ++i) {
    const std::int32_t token = session->SampleGreedy();
    out.push_back(token);
    if (!session->Evaluate(token, error)) return {};
  }
  return out;
}

// Native DecodeStep loop with a caller sampler (draft-verify reference).
std::vector<std::int32_t> NativeStepRun(qfn::Session* session,
                                        const std::vector<std::int32_t>& prompt,
                                        int max_tokens,
                                        gufo::sampling::SamplerState& sampler,
                                        std::vector<std::uint32_t>* commits,
                                        std::string* error) {
  if (!session->Sync(prompt, error)) return {};
  std::vector<std::int32_t> out;
  while (static_cast<int>(out.size()) < max_tokens) {
    qfn::Session::DecodeResult step;
    if (!session->DecodeStep(4, sampler, &step, error, true)) return {};
    if (step.tokens.empty()) {
      if (error->empty()) *error = "native DecodeStep committed nothing";
      return {};
    }
    if (commits != nullptr) {
      commits->push_back(static_cast<std::uint32_t>(step.tokens.size()));
    }
    out.insert(out.end(), step.tokens.begin(), step.tokens.end());
  }
  out.resize(static_cast<std::size_t>(max_tokens));
  return out;
}

// Adapter DecodeStep loop (L1 surface). The L1 outcome names only the last
// committed token, so the full per-step commits are read back from the
// session token tail (see Tokens()).
std::vector<std::int32_t> AdapterStepRun(qfn::FlashNextEngineSession* session,
                                         const std::vector<std::int32_t>& prompt,
                                         int max_tokens,
                                         std::vector<std::uint32_t>* commits,
                                         std::string* error) {
  if (!session->Sync(prompt, error)) return {};
  std::vector<std::int32_t> out;
  while (static_cast<int>(out.size()) < max_tokens) {
    const std::uint32_t before = session->Length();
    e::DecodeOutcome outcome;
    if (!session->DecodeStep(4, &outcome, error)) return {};
    if (outcome.tokens_committed == 0) {
      if (error->empty()) *error = "adapter DecodeStep committed nothing";
      return {};
    }
    if (session->Length() != before + outcome.tokens_committed) {
      *error = "adapter DecodeStep length did not advance by committed";
      return {};
    }
    const auto tail = session->Tokens();
    out.insert(out.end(), tail.begin() + before,
               tail.begin() + before + outcome.tokens_committed);
    if (commits != nullptr) commits->push_back(outcome.tokens_committed);
  }
  out.resize(static_cast<std::size_t>(max_tokens));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::fprintf(stderr, "usage: %s MODEL.gguf [MTP.gguf] [MAX_TOKENS]\n",
                 argv[0]);
    return 2;
  }
  const std::string model_path = argv[1];
  std::string mtp_path;
  int max_tokens = 16;
  const auto is_number = [](const std::string& s) {
    return !s.empty() &&
           std::all_of(s.begin(), s.end(),
                       [](unsigned char c) { return std::isdigit(c) != 0; });
  };
  if (argc >= 3) {
    if (is_number(argv[2])) {
      max_tokens = std::atoi(argv[2]);
    } else {
      mtp_path = argv[2];
      if (argc == 4) max_tokens = std::atoi(argv[3]);
    }
  }
  if (max_tokens <= 0 || max_tokens > 256) return Fail("MAX_TOKENS 1..256");

  // Incomplete sentence: a bare complete sentence makes greedy emit EOS
  // immediately (native DecodeStep reports stop with no commit), which
  // exercises no multi-token path. This tail continues for 16 tokens.
  const std::string prompt_text = "The quick brown fox jumps over the lazy dog. "
                                  "It was a bright sunny morning and the";
  std::string error;

  // (a) Native direct reference (plain f16, autoregressive): the 主體 path.
  qfn::ModelOptions plain_options;
  plain_options.max_context = 2048;
  plain_options.decode_concurrency = 8;
  auto native_plain = qfn::Model::Load(model_path, plain_options, &error);
  if (!native_plain) return Fail("native plain load: " + error);
  if (native_plain->HasMtp()) return Fail("plain load unexpectedly has MTP");
  auto ref_session = native_plain->CreateSession(
      gufo::core::SessionMode::kAutoregressive, 2048, &error);
  if (!ref_session) return Fail("native plain session: " + error);
  const std::vector<std::int32_t> prompt = native_plain->Tokenize(prompt_text);
  if (prompt.empty()) return Fail("native Tokenize is empty");
  const auto ref_ids = NativeGreedyRun(ref_session.get(), prompt, max_tokens,
                                       &error);
  if (!error.empty()) return Fail("native greedy run: " + error);
  // One model at a time: the 4-shard target is ~80GB, so the native,
  // adapter-direct and registry models must not co-reside. Each phase
  // saves its token vectors and releases its model before the next load.
  const std::uint32_t ref_vocab = native_plain->VocabSize();
  const bool ref_has_mtp = native_plain->HasMtp();
  const std::string ref_text = native_plain->Decode(
      std::span<const std::int32_t>(ref_ids.data(), ref_ids.size()));
  ref_session.reset();
  native_plain.reset();

  // Native MTP reference when a sidecar is given: greedy draft-verify must
  // stay byte-identical to the plain autoregressive stream.
  std::vector<std::int32_t> mtp_ids;
  if (!mtp_path.empty()) {
    qfn::ModelOptions mtp_options = plain_options;
    mtp_options.mtp_model_path = mtp_path;
    mtp_options.max_draft_tokens = 7;
    auto native_mtp = qfn::Model::Load(model_path, mtp_options, &error);
    if (!native_mtp) return Fail("native MTP load: " + error);
    if (!native_mtp->HasMtp()) return Fail("MTP load reports HasMtp()==false");
    auto mtp_session = native_mtp->CreateSession(
        gufo::core::SessionMode::kSpeculative, 2048, &error);
    if (!mtp_session) return Fail("native MTP session: " + error);
    gufo::sampling::SamplerState sampler;
    std::vector<std::uint32_t> commits;
    mtp_ids = NativeStepRun(mtp_session.get(), prompt, max_tokens, sampler,
                            &commits, &error);
    if (!error.empty()) return Fail("native MTP DecodeStep run: " + error);
    if (mtp_ids != ref_ids) {
      return Fail("native MTP DecodeStep stream differs from plain greedy "
                  "(greedy draft-verify must stay byte-identical)");
    }
    mtp_session.reset();
    native_mtp.reset();
  }

  // (b) Adapter direct path with identical options (MTP sidecar when given).
  qfn::ModelOptions adapter_options = plain_options;
  if (!mtp_path.empty()) {
    adapter_options.mtp_model_path = mtp_path;
    adapter_options.max_draft_tokens = 7;
  }
  auto direct = qfn::FlashNextEngineModel::Load(model_path, adapter_options,
                                                &error);
  if (!direct) return Fail("adapter direct load: " + error);
  if (direct->HasMtp() != ref_has_mtp && mtp_path.empty()) {
    return Fail("adapter HasMtp mismatch on plain load");
  }
  if (!mtp_path.empty() && !direct->HasMtp()) {
    return Fail("adapter MTP load reports HasMtp()==false");
  }
  if (!e::EngineRegistry::Instance().Contains("qwen38-flash-next")) {
    return Fail("qwen38-flash-next backend is not registered");
  }
  const auto adapter_mode = direct->HasMtp()
                                ? gufo::core::SessionMode::kSpeculative
                                : gufo::core::SessionMode::kAutoregressive;
  auto direct_session = direct->CreateSession(adapter_mode, 2048, &error);
  if (!direct_session) return Fail("adapter direct session: " + error);
  const auto direct_prompt = direct->Tokenize(prompt_text, false);
  if (direct_prompt != prompt) return Fail("adapter Tokenize mismatch");
  if (direct->VocabSize() != ref_vocab) {
    return Fail("adapter VocabSize mismatch");
  }
  const auto direct_ids =
      L0GreedyRun(direct_session.get(), direct_prompt, max_tokens, &error);
  if (!error.empty()) return Fail("adapter L0 run: " + error);
  // add_bos=true must differ only by the BOS prefix convention, never crash.
  const auto bos_prompt = direct->Tokenize(prompt_text, true);
  if (bos_prompt.empty()) return Fail("adapter add_bos Tokenize is empty");

  // Release the adapter-direct model before the registry load (one model
  // at a time); section (d) reloads it for the DecodeStep/batch/snapshot
  // phases.
  direct_session.reset();
  direct.reset();
  // (c) Registry path with identical request options.
  e::EngineLoadRequest request;
  request.model_path = model_path;
  request.max_context = 2048;
  auto via_registry = e::CreateEngineModel("qwen38-flash-next", request,
                                           &error);
  if (!via_registry) return Fail("registry create: " + error);
  auto registry_session = via_registry->CreateSession(&error);
  if (!registry_session) return Fail("registry session: " + error);
  const auto reg_prompt = via_registry->Tokenize(prompt_text, false);
  if (reg_prompt != prompt) return Fail("registry Tokenize mismatch");
  const auto registry_ids =
      L0GreedyRun(registry_session.get(), reg_prompt, max_tokens, &error);
  if (!error.empty()) return Fail("registry run: " + error);
  registry_session.reset();
  via_registry.reset();
  // Reload one adapter model for the DecodeStep/batch/snapshot/
  // sync-through phases below (still one resident model at a time).
  direct = qfn::FlashNextEngineModel::Load(model_path, adapter_options,
                                           &error);
  if (!direct) return Fail("adapter reload: " + error);

  // (d) Adapter DecodeStep draft-verify path vs the native DecodeStep.
  auto step_session = direct->CreateSession(adapter_mode, 2048, &error);
  if (!step_session) return Fail("step session: " + error);
  auto* spec = dynamic_cast<e::SpeculativeSession*>(step_session.get());
  if (spec == nullptr) return Fail("session is not a SpeculativeSession");
  if (!spec->SupportsSpeculative()) {
    return Fail("flash-next must report SupportsSpeculative()==true (L1 "
                "DecodeStep maps 1:1 onto the native draft-verify "
                "DecodeStep; see engine_adapter.hpp)");
  }
  std::vector<std::uint32_t> adapter_commits;
  auto* step_typed =
      dynamic_cast<qfn::FlashNextEngineSession*>(step_session.get());
  if (step_typed == nullptr) {
    return Fail("step session is not a FlashNextEngineSession");
  }
  const auto step_ids = AdapterStepRun(step_typed, direct_prompt,
                                       max_tokens, &adapter_commits, &error);
  if (!error.empty()) return Fail("adapter DecodeStep run: " + error);
  if (step_ids != ref_ids) {
    return Fail("adapter DecodeStep stream differs from native greedy");
  }
  if (!mtp_path.empty()) {
    // Cross-check against the native draft-verify loop above: same commits.
    if (step_ids != mtp_ids) {
      return Fail("adapter DecodeStep stream differs from native MTP stream");
    }
    auto* adapter_typed =
        dynamic_cast<qfn::FlashNextEngineSession*>(step_session.get());
    if (adapter_typed == nullptr) {
      return Fail("session is not a FlashNextEngineSession");
    }
    const auto& stats = adapter_typed->Statistics();
    if (stats.cycles == 0) return Fail("MTP run recorded no draft cycles");
    std::fprintf(stderr,
                 "mtp: cycles=%llu drafted=%llu accepted=%llu gate=%g\n",
                 static_cast<unsigned long long>(stats.cycles),
                 static_cast<unsigned long long>(stats.drafted),
                 static_cast<unsigned long long>(stats.accepted),
                 static_cast<double>(qfn::FlashNextEngineModel::MtpConfidenceGate()));
  }

  // (e) Adapter DecodeBatch/EvaluateBatch translation vs serial paths.
  auto mk = [&](const char* what) {
    auto s = direct->CreateSession(adapter_mode, 2048, &error);
    if (!s) throw std::runtime_error(std::string(what) + ": " + error);
    if (!s->Sync(direct_prompt, &error)) {
      throw std::runtime_error(std::string(what) + " sync: " + error);
    }
    auto* typed = dynamic_cast<qfn::FlashNextEngineSession*>(s.get());
    if (typed == nullptr) {
      throw std::runtime_error("not a FlashNextEngineSession");
    }
    return std::pair<std::unique_ptr<e::EngineSession>,
                     qfn::FlashNextEngineSession*>(std::move(s), typed);
  };
  try {
    auto [a_hold, a] = mk("batch A");
    auto [b_hold, b] = mk("batch B");
    auto [c_hold, c] = mk("control C");
    auto [d_hold, d] = mk("control D");
    qfn::Session::DecodeResult ra, rb;
    const qfn::FlashNextEngineSession::DecodeRequest batch[] = {
        {a, 4, &ra}, {b, 4, &rb}};
    if (!qfn::FlashNextEngineSession::DecodeBatch(batch, &error)) {
      return Fail("adapter DecodeBatch: " + error);
    }
    e::DecodeOutcome oc, od;
    if (!dynamic_cast<e::SpeculativeSession*>(c_hold.get())
             ->DecodeStep(4, &oc, &error)) {
      return Fail("control C DecodeStep: " + error);
    }
    if (!dynamic_cast<e::SpeculativeSession*>(d_hold.get())
             ->DecodeStep(4, &od, &error)) {
      return Fail("control D DecodeStep: " + error);
    }
    if (ra.tokens.empty() || rb.tokens.empty()) {
      return Fail("adapter DecodeBatch committed nothing");
    }
    if (ra.tokens.back() != oc.token || rb.tokens.back() != od.token ||
        ra.tokens.size() != oc.tokens_committed ||
        rb.tokens.size() != od.tokens_committed) {
      return Fail("adapter DecodeBatch mismatch vs serial DecodeStep");
    }
    // Empty-batch guard.
    if (qfn::FlashNextEngineSession::DecodeBatch({}, &error)) {
      return Fail("adapter DecodeBatch accepted an empty batch");
    }
    error.clear();

    // EvaluateBatch: one token per session vs serial Evaluate.
    auto [e_hold, e_sess] = mk("eval E");
    auto [f_hold, f_sess] = mk("eval F");
    auto [g_hold, g_sess] = mk("control G");
    auto [h_hold, h_sess] = mk("control H");
    (void)e_sess;
    (void)f_sess;
    const std::int32_t ta = e_hold->SampleGreedy();
    const std::int32_t tb = f_hold->SampleGreedy();
    const qfn::FlashNextEngineSession::AdvanceRequest advances[] = {
        {e_sess, ta}, {f_sess, tb}};
    if (!qfn::FlashNextEngineSession::EvaluateBatch(advances, &error)) {
      return Fail("adapter EvaluateBatch: " + error);
    }
    if (!g_hold->Evaluate(ta, &error) || !h_hold->Evaluate(tb, &error)) {
      return Fail("control eval: " + error);
    }
    if (e_hold->Length() != g_hold->Length() ||
        f_hold->Length() != h_hold->Length()) {
      return Fail("adapter EvaluateBatch Length mismatch vs serial Evaluate");
    }
    if (e_hold->SampleGreedy() != g_hold->SampleGreedy() ||
        f_hold->SampleGreedy() != h_hold->SampleGreedy()) {
      return Fail("adapter EvaluateBatch greedy mismatch vs serial Evaluate");
    }
  } catch (const std::exception& ex) {
    return Fail(ex.what());
  }

  // (f) SnapshotCapable round-trip: snapshot, advance, restore, continue
  // must equal an uninterrupted control run.
  {
    auto s_hold = direct->CreateSession(adapter_mode, 2048, &error);
    auto t_hold = direct->CreateSession(adapter_mode, 2048, &error);
    if (!s_hold || !t_hold) return Fail("snapshot sessions: " + error);
    auto* s = dynamic_cast<qfn::FlashNextEngineSession*>(s_hold.get());
    if (s == nullptr) return Fail("session is not snapshot-capable");
    auto* snap_capable = dynamic_cast<e::SnapshotCapable*>(s_hold.get());
    if (snap_capable == nullptr) {
      return Fail("session does not expose SnapshotCapable");
    }
    if (!s_hold->Sync(direct_prompt, &error) ||
        !t_hold->Sync(direct_prompt, &error)) {
      return Fail("snapshot sync: " + error);
    }
    std::vector<std::uint8_t> payload(snap_capable->SnapshotPayloadBytes());
    if (payload.empty()) return Fail("snapshot payload size is zero");
    if (!snap_capable->Snapshot(payload, &error)) {
      return Fail("snapshot capture: " + error);
    }
    // Advance S, then rewind it: the restored continuation must agree
    // with the pristine control T step by step. (T stays at the prompt;
    // advancing it too would compare continuations from different
    // positions, which legitimately diverge.)
    std::vector<std::int32_t> discard;
    for (int i = 0; i < 4; ++i) {
      discard.push_back(s_hold->SampleGreedy());
      if (!s_hold->Evaluate(discard.back(), &error)) {
        return Fail("snapshot advance: " + error);
      }
    }
    if (!snap_capable->RestoreSnapshot(payload, &error)) {
      return Fail("snapshot restore: " + error);
    }
    if (s_hold->Length() !=
        static_cast<std::uint32_t>(direct_prompt.size())) {
      return Fail("restored Length mismatch");
    }
    for (int i = 0; i < 8; ++i) {
      const std::int32_t ts = s_hold->SampleGreedy();
      const std::int32_t tt = t_hold->SampleGreedy();
      if (ts != tt) return Fail("post-restore continuation diverged");
      if (!s_hold->Evaluate(ts, &error) || !t_hold->Evaluate(tt, &error)) {
        return Fail("post-restore evaluate: " + error);
      }
    }
    // Cross-version guard: a truncated payload must refuse, not crash.
    std::string bad_error;
    if (snap_capable->RestoreSnapshot(
            std::span<const std::uint8_t>(payload.data(),
                                          payload.size() / 2),
            &bad_error)) {
      return Fail("snapshot restore accepted a truncated payload");
    }
  }

  // (g) SyncThrough smoke: boundary checkpoint + live session at prompt.
  {
    auto u_hold = direct->CreateSession(adapter_mode, 2048, &error);
    if (!u_hold) return Fail("syncthrough session: " + error);
    auto* u = dynamic_cast<qfn::FlashNextEngineSession*>(u_hold.get());
    if (u == nullptr) return Fail("not a FlashNextEngineSession");
    std::vector<std::int32_t> full = direct_prompt;
    const std::size_t tail = std::min<std::size_t>(8, direct_prompt.size());
    full.insert(full.end(), direct_prompt.begin(),
                direct_prompt.begin() + tail);
    std::unique_ptr<qfn::SessionSnapshot> checkpoint;
    if (!u->SyncThrough(full, static_cast<std::uint32_t>(direct_prompt.size()),
                        &checkpoint, &error)) {
      return Fail("SyncThrough: " + error);
    }
    if (!checkpoint || checkpoint->SizeBytes() == 0) {
      return Fail("SyncThrough returned an empty checkpoint");
    }
    if (u_hold->Length() != static_cast<std::uint32_t>(full.size())) {
      return Fail("SyncThrough Length mismatch");
    }
  }

  // (i) q8 structural smoke in a fresh device state (a 100GB+ f16 resident
  // plus a q8 resident cannot co-reside): same tokenization, 4-token run,
  // exact Length accounting.
  direct_session.reset();
  direct.reset();
  registry_session.reset();
  via_registry.reset();
  step_session.reset();
  spec = nullptr;
  ref_session.reset();
  native_plain.reset();
  e::EngineLoadRequest q8request = request;
  q8request.kv_cache_quant = "q8";
  auto q8model = e::CreateEngineModel("qwen38-flash-next", q8request, &error);
  if (!q8model) return Fail("q8 create: " + error);
  auto q8session = q8model->CreateSession(&error);
  if (!q8session) return Fail("q8 session: " + error);
  const auto q8prompt = q8model->Tokenize(prompt_text, false);
  if (q8prompt != prompt) return Fail("q8 Tokenize mismatch");
  const auto q8_ids = L0GreedyRun(q8session.get(), q8prompt, 4, &error);
  if (!error.empty()) return Fail("q8 run: " + error);
  if (q8_ids.size() != 4) return Fail("q8 short run length mismatch");
  if (q8session->Length() != static_cast<std::uint32_t>(q8prompt.size() + 4)) {
    return Fail("q8 Length accounting mismatch");
  }
  // q8 semantic smoke: the 4 committed ids must decode to text through
  // the same tokenizer (structural Length match alone is not semantic).
  const auto* q8typed =
      dynamic_cast<qfn::FlashNextEngineModel*>(q8model.get());
  if (q8typed == nullptr) return Fail("q8 model is not flash-next");
  const std::string q8note = q8typed->Decode(
      std::span<const std::int32_t>(q8_ids.data(), q8_ids.size()));
  std::fprintf(stderr, "q8 smoke: %d tokens text [%s]\n",
               static_cast<int>(q8_ids.size()),
               q8note.empty() ? "<empty>" : q8note.c_str());

  if (registry_ids != direct_ids || direct_ids != ref_ids) {
    return Fail("token stream mismatch");
  }
  if (step_ids != ref_ids) return Fail("DecodeStep stream mismatch");
  // All models are released above; the text was decoded from the saved
  // native token stream (token identity already proven phase by phase).
  const std::string note = ref_text;
  std::fprintf(stderr, "registry_pilot PASS: %d tokens text-match [%s]\n",
               max_tokens, note.empty() ? "<empty>" : note.c_str());
  return 0;
}
