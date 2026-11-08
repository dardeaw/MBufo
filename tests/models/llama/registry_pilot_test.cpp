// Llama registry pilot: the same short greedy sentence through (a) the
// direct LlamaEngineModel adapter (L0 surface) and (b)
// EngineRegistry::Create("llama"), plus the DecodeStep plain fallback.
// Token ids and decoded text must match exactly.
// Usage: llama_registry_pilot_test MODEL.gguf [MAX_TOKENS=16]
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/engine/engine_batch.hpp"
#include "src/models/llama/engine_adapter.hpp"

namespace g = gufo::models::llama;
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

  if (!e::EngineRegistry::Instance().Contains("llama")) {
    // The adapter TU also defines the registrar; referencing
    // LlamaEngineModel::Load below pulls it in from the static lib.
  }

  const std::string model_path = argv[1];
  g::ModelOptions options;
  options.max_context = 2048;

  // (a) Direct adapter path.
  std::string error;
  auto direct = g::LlamaEngineModel::Load(model_path, options, &error);
  if (!direct) return Fail("direct load: " + error);
  if (!e::EngineRegistry::Instance().Contains("llama")) {
    return Fail("llama backend is not registered");
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
  auto via_registry =
      e::CreateEngineModel("llama", request, &error);
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
  auto* spec =
      dynamic_cast<e::SpeculativeSession*>(step_session.get());
  if (spec == nullptr) return Fail("session is not a SpeculativeSession");
  if (spec->SupportsSpeculative()) {
    return Fail("llama must report SupportsSpeculative()==false");
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

  if (registry_ids != direct_ids) return Fail("token stream mismatch");
  if (step_ids != direct_ids) return Fail("DecodeStep stream mismatch");
  const std::string note =
      (direct_text.empty() ? std::string("<empty>") : direct_text);
  std::fprintf(stderr, "registry_pilot PASS: %d tokens text-match [%s]\n",
               max_tokens, note.c_str());
  return 0;
}
