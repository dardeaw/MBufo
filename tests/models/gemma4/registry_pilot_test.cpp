// Gemma4 registry pilot: the same short greedy sentence through (a) the
// direct GemmaEngineModel adapter (L0 surface) and (b)
// EngineRegistry::Create("gemma4"), plus the DecodeStep plain fallback,
// the adapter DecodeBatch translation onto the native parallel-array
// signature, MTP pass-through wiring, and a q8 structural smoke.
// Token ids and decoded text must match exactly.
// Usage: gemma4_registry_pilot_test MODEL.gguf [MAX_TOKENS=16]
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/engine/engine_batch.hpp"
#include "src/models/gemma4/engine_adapter.hpp"

namespace g = gufo::models::gemma4;
namespace e = gufo::core::engine;

namespace {

int Fail(const std::string& m) {
  std::fprintf(stderr, "registry_pilot FAIL: %s\n", m.c_str());
  return 1;
}

std::vector<std::int32_t> GreedyRun(e::EngineSession* session,
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: %s MODEL.gguf [MAX_TOKENS]\n", argv[0]);
    return 2;
  }
  int max_tokens = 16;
  if (argc == 3) max_tokens = std::atoi(argv[2]);
  if (max_tokens <= 0 || max_tokens > 256) return Fail("MAX_TOKENS 1..256");

  const std::string model_path = argv[1];
  g::ModelOptions options;
  options.max_context = 2048;

  // (a) Direct adapter path.
  std::string error;
  auto direct = g::GemmaEngineModel::Load(model_path, options, &error);
  if (!direct) return Fail("direct load: " + error);
  if (!e::EngineRegistry::Instance().Contains("gemma4")) {
    return Fail("gemma4 backend is not registered");
  }
  auto direct_session = direct->CreateSession(&error);
  if (!direct_session) return Fail("direct session: " + error);
  const std::string prompt_text = "The quick brown fox jumps over the lazy dog.";
  const auto prompt = direct->Tokenize(prompt_text, /*add_bos=*/false);
  const auto direct_ids =
      GreedyRun(direct_session.get(), prompt, max_tokens, &error);
  if (!error.empty()) return Fail("direct run: " + error);
  const std::string direct_text =
      direct->Decode(std::span<const std::int32_t>(direct_ids));

  // (b) Registry path with identical options.
  e::EngineLoadRequest request;
  request.model_path = model_path;
  request.max_context = 2048;
  auto via_registry = e::CreateEngineModel("gemma4", request, &error);
  if (!via_registry) return Fail("registry create: " + error);
  auto registry_session = via_registry->CreateSession(&error);
  if (!registry_session) return Fail("registry session: " + error);
  const auto reg_prompt = via_registry->Tokenize(prompt_text, false);
  if (reg_prompt != prompt) return Fail("registry Tokenize mismatch");
  if (via_registry->VocabSize() != direct->VocabSize()) {
    return Fail("registry VocabSize mismatch");
  }
  const auto registry_ids =
      GreedyRun(registry_session.get(), reg_prompt, max_tokens, &error);
  if (!error.empty()) return Fail("registry run: " + error);

  // (c) DecodeStep plain-fallback path on a fresh registry session.
  auto step_session = via_registry->CreateSession(&error);
  if (!step_session) return Fail("step session: " + error);
  auto* spec = dynamic_cast<e::SpeculativeSession*>(step_session.get());
  if (spec == nullptr) return Fail("session is not a SpeculativeSession");
  if (spec->SupportsSpeculative()) {
    return Fail("gemma4 must report SupportsSpeculative()==false "
                "(the native SpeculativeRound needs EnableMtp(drafts_path) "
                "first, which EngineLoadRequest cannot supply; the "
                "session-bound GemmaMtpDraftBackend likewise builds only "
                "from a live MTP-enabled Session)");
  }
  if (!step_session->Sync(reg_prompt, &error)) {
    return Fail("step sync: " + error);
  }
  std::vector<std::int32_t> step_ids;
  for (int i = 0; i < max_tokens; ++i) {
    e::DecodeOutcome outcome;
    if (!spec->DecodeStep(4, &outcome, &error)) {
      return Fail("DecodeStep: " + error);
    }
    step_ids.push_back(outcome.token);
    if (outcome.stop) break;
  }

  // (d) Adapter DecodeBatch translation: two sessions sharing one model,
  // one token each, must match the serial Evaluate path token-for-token.
  // This exercises the adapter unwrap onto the native parallel-array
  // signature (span<Session*>, span<const int32_t>), which differs from
  // glimmer's BatchItem rows.
  auto batch_session = via_registry->CreateSession(&error);
  if (!batch_session) return Fail("batch session: " + error);
  auto* a_adapter =
      dynamic_cast<g::GemmaEngineSession*>(batch_session.get());
  if (a_adapter == nullptr) {
    return Fail("session is not a GemmaEngineSession");
  }
  auto b_holder = via_registry->CreateSession(&error);
  if (!b_holder) return Fail("batch session B: " + error);
  auto* b_adapter = dynamic_cast<g::GemmaEngineSession*>(b_holder.get());
  if (b_adapter == nullptr) {
    return Fail("session B is not a GemmaEngineSession");
  }
  auto c_holder = via_registry->CreateSession(&error);
  auto d_holder = via_registry->CreateSession(&error);
  if (!c_holder || !d_holder) return Fail("control sessions: " + error);
  for (auto* s :
       {batch_session.get(), b_holder.get(), c_holder.get(), d_holder.get()}) {
    if (!s->Sync(reg_prompt, &error)) return Fail("batch sync: " + error);
  }
  const std::int32_t ta = batch_session->SampleGreedy();
  const std::int32_t tb = b_holder->SampleGreedy();
  std::vector<g::GemmaEngineSession*> batch = {a_adapter, b_adapter};
  const std::vector<std::int32_t> toks = {ta, tb};
  if (!g::GemmaEngineSession::DecodeBatch(batch, toks, &error)) {
    return Fail("adapter DecodeBatch: " + error);
  }
  if (!c_holder->Evaluate(ta, &error) || !d_holder->Evaluate(tb, &error)) {
    return Fail("control eval: " + error);
  }
  if (batch_session->Length() != c_holder->Length() ||
      b_holder->Length() != d_holder->Length()) {
    return Fail("adapter DecodeBatch Length mismatch vs serial Evaluate");
  }
  if (batch_session->SampleGreedy() != c_holder->SampleGreedy() ||
      b_holder->SampleGreedy() != d_holder->SampleGreedy()) {
    return Fail("adapter DecodeBatch greedy mismatch vs serial Evaluate");
  }
  // Size-mismatch translation guard: 2 sessions vs 1 token must refuse.
  const std::vector<std::int32_t> short_toks = {ta};
  if (g::GemmaEngineSession::DecodeBatch(batch, short_toks, &error)) {
    return Fail("adapter DecodeBatch accepted a size-mismatched batch");
  }
  error.clear();  // The expected refusal above leaves error set; a stale
                  // message must not trip the later empty-checks.

  // (e) MTP pass-through wiring on a fresh session (no drafts attached):
  // disabled by default, bogus drafts path fails, and a speculative round
  // without MTP fails cleanly instead of crashing.
  auto mtp_holder = via_registry->CreateSession(&error);
  if (!mtp_holder) return Fail("mtp session: " + error);
  auto* mtp_adapter = dynamic_cast<g::GemmaEngineSession*>(mtp_holder.get());
  if (mtp_adapter == nullptr) return Fail("session is not MTP-capable");
  if (mtp_adapter->MtpEnabled()) return Fail("MTP must default to disabled");
  if (!mtp_holder->Sync(reg_prompt, &error)) return Fail("mtp sync: " + error);
  std::string mtp_error;
  if (mtp_adapter->EnableMtp("/nonexistent/drafts.gguf", &mtp_error)) {
    return Fail("EnableMtp accepted a bogus drafts path");
  }
  std::vector<std::int32_t> mtp_out;
  if (mtp_adapter->SpeculativeRound(ta, 4, &mtp_out, &mtp_error)) {
    return Fail("SpeculativeRound succeeded without MTP attached");
  }
  // NOTE: CommitMtpRound is private to gemma4::Session (reachable only
  // from inside SpeculativeRound), so the adapter deliberately exposes no
  // pass-through for it; touching the engine body is out of scope.

  // (f) q8 structural smoke: same weights, blocked-KV path, same shapes.
  // Runs in a fresh device state: release every f16 object first (a 26B
  // f16 + 26B q8 model cannot co-reside on one gfx1151).
  direct_session.reset();
  direct.reset();
  registry_session.reset();
  via_registry.reset();
  step_session.reset();
  spec = nullptr;
  batch_session.reset();
  a_adapter = nullptr;
  b_holder.reset();
  b_adapter = nullptr;
  c_holder.reset();
  d_holder.reset();
  mtp_holder.reset();
  mtp_adapter = nullptr;
  e::EngineLoadRequest q8request = request;
  q8request.kv_cache_quant = "q8";
  auto q8model = e::CreateEngineModel("gemma4", q8request, &error);
  if (!q8model) return Fail("q8 create: " + error);
  auto q8session = q8model->CreateSession(&error);
  if (!q8session) return Fail("q8 session: " + error);
  const auto q8prompt = q8model->Tokenize(prompt_text, false);
  if (q8prompt != prompt) return Fail("q8 Tokenize mismatch");
  const auto q8_ids = GreedyRun(q8session.get(), q8prompt, 4, &error);
  if (!error.empty()) return Fail("q8 run: " + error);
  if (q8_ids.size() != 4) return Fail("q8 short run length mismatch");
  if (q8session->Length() !=
      static_cast<std::uint32_t>(q8prompt.size() + 4)) {
    return Fail("q8 Length accounting mismatch");
  }

  if (registry_ids != direct_ids) return Fail("token stream mismatch");
  if (step_ids != direct_ids) return Fail("DecodeStep stream mismatch");
  const std::string note =
      (direct_text.empty() ? std::string("<empty>") : direct_text);
  std::fprintf(stderr, "registry_pilot PASS: %d tokens text-match [%s]\n",
               max_tokens, note.c_str());
  return 0;
}
