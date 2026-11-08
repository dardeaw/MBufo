// Glimmer registry pilot: the same short greedy sentence through (a) the
// direct GlimmerEngineModel adapter (L0 surface) and (b)
// EngineRegistry::Create("glimmer"), plus the DecodeStep plain fallback,
// the read-only Verify contract, and a q8 structural smoke.
// Token ids and decoded text must match exactly.
// Usage: glimmer_registry_pilot_test MODEL.gguf [MAX_TOKENS=16]
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/engine/engine_batch.hpp"
#include "src/models/glimmer/engine_adapter.hpp"

namespace g = gufo::models::glimmer;
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
  auto direct = g::GlimmerEngineModel::Load(model_path, options, &error);
  if (!direct) return Fail("direct load: " + error);
  if (!e::EngineRegistry::Instance().Contains("glimmer")) {
    return Fail("glimmer backend is not registered");
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
  auto via_registry = e::CreateEngineModel("glimmer", request, &error);
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
    return Fail("glimmer must report SupportsSpeculative()==false "
                "(native SpeculativeRound needs a caller-owned "
                "glimmer-dflash backend, not expressible via DecodeStep)");
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

  // (d) Verify commit contract on a fresh adapter session: Length()
  // advances by k (Verify shares Forward mechanics; the speculative accept
  // path overwrites with its own Forwards), vbest has width k, and a
  // repeated verify at the same start_pos is deterministic.
  auto verify_session = via_registry->CreateSession(&error);
  if (!verify_session) return Fail("verify session: " + error);
  auto* vadapter =
      dynamic_cast<g::GlimmerEngineSession*>(verify_session.get());
  if (vadapter == nullptr) return Fail("session is not a GlimmerEngineSession");
  if (!verify_session->Sync(reg_prompt, &error)) {
    return Fail("verify sync: " + error);
  }
  const std::uint32_t len_before = verify_session->Length();
  const std::size_t k = prompt.size() < 8 ? prompt.size() : 8;
  std::vector<std::int32_t> probe(prompt.begin(), prompt.begin() + k);
  std::vector<std::int32_t> vbest1, vbest2;
  if (!vadapter->Verify(probe, len_before, vbest1, &error)) {
    return Fail("Verify(1): " + error);
  }
  if (verify_session->Length() != len_before + static_cast<std::uint32_t>(k)) {
    return Fail("Verify did not advance Length by k (Forward mechanics)");
  }
  if (!vadapter->Verify(probe, len_before, vbest2, &error)) {
    return Fail("Verify(2): " + error);
  }
  if (vbest1.size() != k || vbest2.size() != k) {
    return Fail("Verify vbest width mismatch");
  }
  if (vbest1 != vbest2) return Fail("Verify is not deterministic");

  // (e) q8 structural smoke: same weights, blocked-KV path, same shapes.
  // Runs in a fresh device state: release every f16 object first (a 30B
  // f16 + 30B q8 model cannot co-reside on one gfx1151).
  direct_session.reset();
  direct.reset();
  registry_session.reset();
  via_registry.reset();
  step_session.reset();
  spec = nullptr;
  verify_session.reset();
  vadapter = nullptr;
  e::EngineLoadRequest q8request = request;
  q8request.kv_cache_quant = "q8";
  auto q8model = e::CreateEngineModel("glimmer", q8request, &error);
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
