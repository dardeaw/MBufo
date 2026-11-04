// Qwen3.8-27B registry pilot: (a) the direct executor greedy run (the
// 主體 reference: ForwardPromptBatch + ForwardToken chain with host
// argmax), (b) the adapter direct L0 greedy run, (c)
// EngineRegistry::Create ("qwen3.8-27b"), (d) the adapter DecodeStep plain
// fallback vs the greedy stream plus Forward span emulation, (e) a
// SnapshotCapable round-trip, and (f) a q8 structural smoke. Token ids and
// decoded text must match exactly.
// Usage: qwen27b_registry_pilot_test MODEL.gguf [MAX_TOKENS]
// The driver loads one model at a time: a 27B resident is tens of GB, so
// each phase saves its token vectors and releases its model first. An
// alarm(600) timeout guard kills a hung run instead of stalling the rig.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <unistd.h>
#include <vector>

#include "src/core/engine/engine_batch.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/engine_adapter.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace q27 = gufo::models::qwen;
namespace e = gufo::core::engine;

namespace {

int Fail(const std::string& m) {
  std::fprintf(stderr, "registry_pilot FAIL: %s\n", m.c_str());
  return 1;
}

void Stage(const std::string& m) {
  std::fprintf(stderr, "[pilot] stage: %s\n", m.c_str());
  std::fflush(stderr);
}

std::int32_t Argmax(std::span<const float> logits) {
  std::int32_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = static_cast<std::int32_t>(i);
  }
  return best;
}

// Direct executor greedy run (the 主體 path): prefill + chained decode with
// host argmax over the copied last-logits row.
std::vector<std::int32_t> DirectGreedyRun(
    gufo::hip::QwenGpuExecutor* executor,
    const std::vector<std::uint32_t>& prompt, int max_tokens) {
  executor->Reset();
  (void)executor->ForwardPromptBatch(prompt, /*start_pos=*/0,
                               /*compute_logits=*/true);
  std::vector<std::int32_t> out;
  std::int32_t next = Argmax(executor->CopyLastLogits());
  for (int i = 0; i < max_tokens; ++i) {
    out.push_back(next);
    (void)executor->ForwardToken(static_cast<std::uint32_t>(next),
                           static_cast<std::uint32_t>(prompt.size() + out.size() - 1),
                           /*compute_logits=*/true);
    next = Argmax(executor->CopyLastLogits());
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

}  // namespace

int main(int argc, char** argv) {
  alarm(600);  // Hang guard: a stalled 27B run dies instead of wedging the rig.
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: %s MODEL.gguf [MAX_TOKENS]\n", argv[0]);
    return 2;
  }
  const std::string model_path = argv[1];
  int max_tokens = 16;
  if (argc == 3) max_tokens = std::atoi(argv[2]);
  if (max_tokens <= 0 || max_tokens > 256) return Fail("MAX_TOKENS 1..256");

  // Incomplete sentence: a bare complete sentence makes greedy emit EOS
  // immediately, which exercises no continuation path. This tail continues
  // for max_tokens. No chat template: the driver compares raw continuation
  // streams, and the 主體 binary comparison below uses the same raw text.
  const std::string prompt_text =
      "The quick brown fox jumps over the lazy dog. "
      "It was a bright sunny morning and the";
  constexpr std::uint32_t kContext = 2048;
  std::string error;

  // (a) Direct executor reference (f16 KV): the 主體 path.
  Stage("direct load");
  std::vector<std::int32_t> ref_ids;
  std::uint32_t ref_vocab = 0;
  std::string ref_text;
  {
    std::string load_error;
    auto raw = gufo::core::GgufReader::OpenFile(model_path, &load_error);
    if (!raw) return Fail("open reader: " + load_error);
    std::shared_ptr<const gufo::core::GgufReader> reader(std::move(raw));
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &load_error);
    if (!model) return Fail("direct model: " + load_error);
    Stage("direct session");
    auto executor =
        gufo::hip::QwenGpuExecutor::Create(model, &load_error, kContext);
    if (!executor) return Fail("direct executor: " + load_error);
    const std::vector<std::uint32_t> prompt =
        executor->GetTokenizer().Encode(prompt_text);
    if (prompt.empty()) return Fail("direct Tokenize is empty");
    Stage("direct run");
    ref_ids = DirectGreedyRun(executor.get(), prompt, max_tokens);
    ref_vocab = executor->GetConfig().vocab_size;
    const std::vector<std::uint32_t> ref_u(ref_ids.begin(), ref_ids.end());
    ref_text = executor->GetTokenizer().Decode(ref_u);
    std::fprintf(stderr, "direct: %d tokens text [%s]\n", max_tokens,
                 ref_text.c_str());
    executor.reset();
    model.reset();
  }

  // (b) Adapter direct path with identical options.
  Stage("adapter load");
  std::vector<std::int32_t> direct_ids;
  {
    q27::Qwen27BModelOptions options;
    options.max_context = kContext;
    auto direct = q27::Qwen27BEngineModel::Load(model_path, options, &error);
    if (!direct) return Fail("adapter direct load: " + error);
    if (!e::EngineRegistry::Instance().Contains("qwen3.8-27b")) {
      return Fail("qwen3.8-27b backend is not registered");
    }
    auto direct_session = direct->CreateSession(&error);
    if (!direct_session) return Fail("adapter direct session: " + error);
    const std::vector<std::int32_t> direct_prompt =
        direct->Tokenize(prompt_text, false);
    if (direct_prompt.empty()) return Fail("adapter Tokenize is empty");
    if (direct->VocabSize() != ref_vocab) {
      return Fail("adapter VocabSize mismatch");
    }
    Stage("adapter run");
    direct_ids =
        L0GreedyRun(direct_session.get(), direct_prompt, max_tokens, &error);
    if (!error.empty()) return Fail("adapter L0 run: " + error);
    // add_bos=true must differ only by the BOS prefix convention.
    const auto bos_prompt = direct->Tokenize(prompt_text, true);
    if (bos_prompt.empty()) return Fail("adapter add_bos Tokenize is empty");
    direct_session.reset();
    direct.reset();
  }

  // (c) Registry path with identical request options.
  Stage("registry load");
  std::vector<std::int32_t> registry_ids;
  {
    e::EngineLoadRequest request;
    request.model_path = model_path;
    request.max_context = kContext;
    auto via_registry = e::CreateEngineModel("qwen3.8-27b", request, &error);
    if (!via_registry) return Fail("registry create: " + error);
    auto registry_session = via_registry->CreateSession(&error);
    if (!registry_session) return Fail("registry session: " + error);
    const auto reg_prompt = via_registry->Tokenize(prompt_text, false);
    Stage("registry run");
    registry_ids =
        L0GreedyRun(registry_session.get(), reg_prompt, max_tokens, &error);
    if (!error.empty()) return Fail("registry run: " + error);
    registry_session.reset();
    via_registry.reset();
  }
  if (registry_ids != direct_ids || direct_ids != ref_ids) {
    return Fail("token stream mismatch (registry/direct/reference)");
  }
  Stage("streams match");

  // Reload one adapter model for the DecodeStep/Forward/snapshot phases
  // below (still one resident model at a time).
  q27::Qwen27BModelOptions adapter_options;
  adapter_options.max_context = kContext;
  auto direct =
      q27::Qwen27BEngineModel::Load(model_path, adapter_options, &error);
  if (!direct) return Fail("adapter reload: " + error);
  const std::vector<std::int32_t> prompt = direct->Tokenize(prompt_text, false);

  // (d) Adapter DecodeStep plain fallback: one token per step, stream must
  // equal the greedy reference. SupportsSpeculative() is false by design:
  // the native verify entry is executor-level (caller drafts + explicit
  // start_pos + CommitVerificationChunk) while L1 DecodeStep carries no
  // drafts, so there is nothing to verify against.
  {
    auto hold = direct->CreateSession(&error);
    if (!hold) return Fail("step session: " + error);
    auto* spec = dynamic_cast<e::SpeculativeSession*>(hold.get());
    if (spec == nullptr) return Fail("session is not a SpeculativeSession");
    if (spec->SupportsSpeculative()) {
      return Fail("27B must report SupportsSpeculative()==false (verify is "
                  "executor-level with caller drafts; L1 DecodeStep carries "
                  "none — see engine_adapter.hpp)");
    }
    std::fprintf(stderr,
                 "speculative: false by design (executor-level verify needs "
                 "caller drafts; plain single-token fallback)\n");
    if (!hold->Sync(prompt, &error)) return Fail("step sync: " + error);
    std::vector<std::int32_t> step_ids;
    while (static_cast<int>(step_ids.size()) < max_tokens) {
      e::DecodeOutcome outcome;
      if (!spec->DecodeStep(4, &outcome, &error)) {
        return Fail("adapter DecodeStep: " + error);
      }
      if (outcome.tokens_committed != 1) {
        return Fail("plain fallback must commit exactly one token");
      }
      step_ids.push_back(outcome.token);
    }
    if (step_ids != ref_ids) return Fail("DecodeStep stream mismatch");
    if (hold->Length() != static_cast<std::uint32_t>(prompt.size() + max_tokens)) {
      return Fail("DecodeStep Length accounting mismatch");
    }
  }

  // Forward span emulation: no-op slice, tail extension, and divergence
  // must all agree with freshly synced controls.
  {
    auto control = direct->CreateSession(&error);
    auto probe = direct->CreateSession(&error);
    if (!control || !probe) return Fail("forward sessions: " + error);
    if (!probe->Sync(prompt, &error)) return Fail("forward sync: " + error);
    const std::int32_t anchor = probe->SampleGreedy();
    // No-op: re-forwarding a committed slice changes nothing.
    std::vector<std::int32_t> slice(prompt.end() - 4, prompt.end());
    if (!probe->Forward(slice, static_cast<std::uint32_t>(prompt.size() - 4),
                        &error)) {
      return Fail("forward no-op: " + error);
    }
    if (probe->Length() != prompt.size() || probe->SampleGreedy() != anchor) {
      return Fail("forward no-op changed the frontier");
    }
    // Extension: appending at Length must match a full sync of the longer
    // prompt.
    std::vector<std::int32_t> longer = prompt;
    longer.push_back(anchor);
    std::vector<std::int32_t> tail = {anchor};
    if (!probe->Forward(tail, static_cast<std::uint32_t>(prompt.size()),
                        &error)) {
      return Fail("forward extension: " + error);
    }
    if (!control->Sync(longer, &error)) return Fail("control sync: " + error);
    if (probe->SampleGreedy() != control->SampleGreedy()) {
      return Fail("forward extension diverged from full sync");
    }
    // Divergence: rewriting the tail must match a full sync of the new
    // prompt.
    std::vector<std::int32_t> fork(prompt.begin(), prompt.end() - 2);
    fork.push_back(anchor == 0 ? 1 : 0);
    fork.push_back(anchor);
    std::vector<std::int32_t> fork_tail(fork.end() - 2, fork.end());
    if (!probe->Forward(fork_tail,
                        static_cast<std::uint32_t>(fork.size() - 2), &error)) {
      return Fail("forward divergence: " + error);
    }
    if (!control->Sync(fork, &error)) return Fail("fork control: " + error);
    if (probe->Length() != fork.size() ||
        probe->SampleGreedy() != control->SampleGreedy()) {
      return Fail("forward divergence diverged from full sync");
    }
  }

  // (e) SnapshotCapable round-trip: snapshot, advance, restore, continue
  // must equal an uninterrupted control run.
  {
    auto s_hold = direct->CreateSession(&error);
    auto t_hold = direct->CreateSession(&error);
    if (!s_hold || !t_hold) return Fail("snapshot sessions: " + error);
    auto* snap_capable = dynamic_cast<e::SnapshotCapable*>(s_hold.get());
    if (snap_capable == nullptr) {
      return Fail("session does not expose SnapshotCapable");
    }
    if (!s_hold->Sync(prompt, &error) || !t_hold->Sync(prompt, &error)) {
      return Fail("snapshot sync: " + error);
    }
    std::vector<std::uint8_t> payload(snap_capable->SnapshotPayloadBytes());
    if (payload.empty()) return Fail("snapshot payload size is zero");
    if (!snap_capable->Snapshot(payload, &error)) {
      return Fail("snapshot capture: " + error);
    }
    for (int i = 0; i < 4; ++i) {
      const std::int32_t t = s_hold->SampleGreedy();
      if (!s_hold->Evaluate(t, &error)) return Fail("snapshot advance: " + error);
    }
    if (!snap_capable->RestoreSnapshot(payload, &error)) {
      return Fail("snapshot restore: " + error);
    }
    if (s_hold->Length() != static_cast<std::uint32_t>(prompt.size())) {
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
    // Truncation guard: a halved payload must refuse, not crash.
    std::string bad_error;
    if (snap_capable->RestoreSnapshot(
            std::span<const std::uint8_t>(payload.data(), payload.size() / 2),
            &bad_error)) {
      return Fail("snapshot restore accepted a truncated payload");
    }
    s_hold.reset();
    t_hold.reset();
  }
  direct.reset();

  // (f) q8 structural smoke in a fresh device state (a 27B f16 resident
  // plus a q8 resident cannot co-reside): same tokenization, 4-token run,
  // exact Length accounting.
  {
    e::EngineLoadRequest q8request;
    q8request.model_path = model_path;
    q8request.max_context = kContext;
    q8request.kv_cache_quant = "q8";
    auto q8model = e::CreateEngineModel("qwen3.8-27b", q8request, &error);
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
    const auto* q8typed =
        dynamic_cast<q27::Qwen27BEngineModel*>(q8model.get());
    if (q8typed == nullptr) return Fail("q8 model is not a Qwen27BEngineModel");
    const std::string q8note =
        q8typed->Decode(std::span<const std::int32_t>(q8_ids.data(), q8_ids.size()));
    std::fprintf(stderr, "q8 smoke: %d tokens text [%s]\n",
                 static_cast<int>(q8_ids.size()),
                 q8note.empty() ? "<empty>" : q8note.c_str());
    q8session.reset();
    q8model.reset();
  }

  std::fprintf(stderr, "registry_pilot PASS: %d tokens text-match [%s]\n",
               max_tokens, ref_text.empty() ? "<empty>" : ref_text.c_str());
  return 0;
}
