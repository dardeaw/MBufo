// DeepSeek V4 Flash registry pilot: the same greedy continuation through (a) the direct DeepSeekEngineModel adapter and
// (b) the native Model/Session (no adapter) and (c) EngineRegistry::Create("deepseek_v4_flash"); token ids must match
// exactly. Also checks Forward(start_pos) against Sync and the snapshot round trip.
// Usage: deepseek_registry_pilot_test MODEL.gguf [MAX_TOKENS=16] [PROMPT]
// Only one 87 GB model can be resident, so each path is released before the next one loads.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/engine/engine_batch.hpp"
#include "src/models/deepseek_v4_flash/engine_adapter.hpp"

namespace d = gufo::models::deepseek_v4_flash;
namespace e = gufo::core::engine;

namespace {

int Fail(const std::string& m) {
  std::fprintf(stderr, "deepseek registry_pilot FAIL: %s\n", m.c_str());
  return 1;
}

void PrintIds(const char* tag, const std::vector<std::int32_t>& ids) {
  std::fprintf(stderr, "[ds-pilot] %s (%zu):", tag, ids.size());
  for (std::int32_t id : ids) std::fprintf(stderr, " %d", id);
  std::fprintf(stderr, "\n");
}

std::vector<std::int32_t> GreedyRun(e::EngineSession* s, const std::vector<std::int32_t>& prompt, int n,
                                    std::string* error) {
  if (!s->Sync(prompt, error)) return {};
  std::vector<std::int32_t> out;
  for (int i = 0; i < n; ++i) {
    const std::int32_t t = s->SampleGreedy();
    out.push_back(t);
    if (!s->Evaluate(t, error)) return {};
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::fprintf(stderr, "usage: %s MODEL.gguf [MAX_TOKENS] [PROMPT]\n", argv[0]);
    return 2;
  }
  const int max_tokens = argc >= 3 ? std::atoi(argv[2]) : 16;
  if (max_tokens <= 0 || max_tokens > 256) return Fail("MAX_TOKENS 1..256");
  const std::string prompt_text = argc == 4 ? argv[3] : "The quick brown fox jumps over the lazy dog.";
  const std::string model_path = argv[1];
  d::ModelOptions options;
  options.max_context = 2048;
  std::string error;

  // (a) direct adapter
  std::vector<std::int32_t> prompt, direct_ids, tail_ids;
  std::uint32_t vocab = 0;
  std::string name;
  {
    auto model = d::DeepSeekEngineModel::Load(model_path, options, &error);
    if (!model) return Fail("direct load: " + error);
    if (!e::EngineRegistry::Instance().Contains("deepseek_v4_flash")) return Fail("backend is not registered");
    prompt = model->Tokenize(prompt_text, /*add_bos=*/false);
    const auto with_bos = model->Tokenize(prompt_text, /*add_bos=*/true);
    if (with_bos.size() != prompt.size() + 1 || !std::equal(prompt.begin(), prompt.end(), with_bos.begin() + 1))
      return Fail("add_bos did not prepend exactly one token");
    PrintIds("prompt", prompt);
    vocab = model->VocabSize();
    name = model->ModelName();
    if (model->ResidentBytes() == 0) return Fail("ResidentBytes is 0");
    auto session = model->CreateSession(&error);
    if (!session) return Fail("direct session: " + error);
    direct_ids = GreedyRun(session.get(), prompt, max_tokens, &error);
    if (direct_ids.empty()) return Fail("direct run: " + error);
    PrintIds("adapter", direct_ids);
    // Logits() must be consistent with SampleGreedy().
    const auto logits = session->Logits();
    if (logits.size() != vocab) return Fail("Logits size != VocabSize");
    std::size_t best = 0;
    for (std::size_t i = 1; i < logits.size(); ++i)
      if (logits[i] > logits[best]) best = i;
    if (static_cast<std::int32_t>(best) != session->SampleGreedy()) return Fail("Logits argmax != SampleGreedy");
    // Forward(start_pos == Length) over the generated tokens reaches the same state as Sync(prompt + tokens).
    auto s2 = model->CreateSession(&error);
    if (!s2 || !s2->Sync(prompt, &error)) return Fail("s2 sync: " + error);
    const std::span<const std::int32_t> head(direct_ids.data(), direct_ids.size() / 2);
    if (!s2->Forward(head, s2->Length(), &error)) return Fail("Forward: " + error);
    const std::int32_t next_a = s2->SampleGreedy();
    if (next_a != direct_ids[direct_ids.size() / 2]) return Fail("Forward continuation differs from the greedy run");
    // Snapshot round trip.
    auto* snap = dynamic_cast<e::SnapshotCapable*>(s2.get());
    if (snap == nullptr) return Fail("session is not SnapshotCapable");
    std::vector<std::uint8_t> payload(snap->SnapshotPayloadBytes());
    if (!snap->Snapshot(payload, &error)) return Fail("Snapshot: " + error);
    auto s3 = model->CreateSession(&error);
    auto* snap3 = dynamic_cast<e::SnapshotCapable*>(s3.get());
    if (s3 == nullptr || snap3 == nullptr || !snap3->RestoreSnapshot(payload, &error)) return Fail("Restore: " + error);
    if (s3->Length() != s2->Length()) return Fail("restored Length differs");
    if (s3->SampleGreedy() != next_a) return Fail("restored session samples a different token");
    // DecodeStep plain fallback reproduces the next greedy token.
    e::DecodeOutcome outcome;
    if (!s3->Length() || !dynamic_cast<e::SpeculativeSession*>(s3.get())->DecodeStep(0, &outcome, &error))
      return Fail("DecodeStep: " + error);
    if (outcome.token != next_a || outcome.tokens_committed != 1) return Fail("DecodeStep token mismatch");
  }

  // (b) native Model/Session
  std::vector<std::int32_t> native_ids;
  {
    auto model = d::Model::Load(model_path, options, &error);
    if (!model) return Fail("native load: " + error);
    auto session = model->CreateSession(gufo::core::SessionMode::kAutoregressive, options.max_context, &error);
    if (!session) return Fail("native session: " + error);
    if (!session->Sync(prompt, &error)) return Fail("native sync: " + error);
    std::uint64_t rng = 0;
    for (int i = 0; i < max_tokens; ++i) {
      const int t = session->SelectNext(0.0F, &rng);
      native_ids.push_back(t);
      if (!session->Evaluate(t, &error)) return Fail("native eval: " + error);
    }
    PrintIds("native", native_ids);
  }
  if (native_ids != direct_ids) return Fail("adapter ids differ from the native Model/Session");

  // (c) registry
  {
    e::EngineLoadRequest request;
    request.model_path = model_path;
    request.max_context = options.max_context;
    auto via = e::CreateEngineModel("deepseek_v4_flash", request, &error);
    if (!via) return Fail("registry create: " + error);
    if (via->VocabSize() != vocab || via->ModelName() != name) return Fail("registry metadata mismatch");
    auto session = via->CreateSession(&error);
    if (!session) return Fail("registry session: " + error);
    const auto reg_ids = GreedyRun(session.get(), via->Tokenize(prompt_text, false), max_tokens, &error);
    PrintIds("registry", reg_ids);
    if (reg_ids != direct_ids) return Fail("registry ids differ");
  }
  std::fprintf(stderr, "deepseek registry_pilot PASS: %d tokens identical (adapter / native / registry)\n", max_tokens);
  return 0;
}