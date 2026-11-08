// Flash-Next adapter: thin forwarding over qwen38_flash_next::Model/Session
// plus the one-line "qwen38-flash-next" EngineRegistry registration (name
// follows the serve-side model_kind convention).

#include "src/models/qwen38_flash_next/engine_adapter.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace gufo::models::qwen38_flash_next {

FlashNextEngineSession::FlashNextEngineSession(
    std::shared_ptr<Model> model, std::unique_ptr<Session> inner,
    std::size_t default_drafts)
    : model_(std::move(model)),
      inner_(std::move(inner)),
      default_drafts_(default_drafts) {}

void FlashNextEngineSession::ResetSamplerHistory(
    std::span<const std::int32_t> tokens) {
  // SamplerState copies the span (see sampling.cpp ctor), so a local is safe.
  std::vector<sampling::TokenId> history;
  history.reserve(tokens.size());
  for (const std::int32_t token : tokens) {
    history.push_back(token < 0 ? sampling::TokenId{0}
                                : static_cast<sampling::TokenId>(token));
  }
  sampler_.ResetHistory(history);
}

bool FlashNextEngineSession::Sync(std::span<const std::int32_t> prompt,
                                  std::string* error_msg) {
  return Sync(prompt, {}, error_msg);
}

bool FlashNextEngineSession::Sync(std::span<const std::int32_t> prompt,
                                  std::span<const std::int32_t> next,
                                  std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "flash-next session is closed";
    return false;
  }
  // Native incremental-prefix Sync (append-only growth forwards only the
  // delta; anything else resets and refeeds); `next` is the n-gram
  // prefetch hint and never changes the result.
  if (!inner_->Sync(prompt, error_msg, next)) return false;
  ResetSamplerHistory(prompt);
  return true;
}

bool FlashNextEngineSession::Evaluate(std::int32_t token,
                                      std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "flash-next session is closed";
    return false;
  }
  if (!inner_->Evaluate(token, error_msg)) return false;
  if (token >= 0) {
    sampler_.Accept(static_cast<sampling::TokenId>(token));
  }
  return true;
}

bool FlashNextEngineSession::Forward(std::span<const std::int32_t> tokens,
                                     std::uint32_t start_pos,
                                     std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "flash-next session is closed";
    return false;
  }
  // The native Session has no Forward entry: emulate with one Sync over
  // prefix+tokens. Re-forwarding already-committed rows is a no-op (their
  // logits are already current); extending appends exactly the tail;
  // anything else resets and refeeds through the native Sync contract.
  if (tokens.empty()) return true;
  const std::span<const std::int32_t> committed = inner_->Tokens();
  if (start_pos > committed.size()) {
    if (error_msg != nullptr) {
      *error_msg = "flash-next Forward starts past the committed length";
    }
    return false;
  }
  if (static_cast<std::uint64_t>(start_pos) + tokens.size() <=
          committed.size() &&
      std::equal(tokens.begin(), tokens.end(),
                 committed.begin() + start_pos)) {
    return true;
  }
  if (static_cast<std::uint64_t>(start_pos) + tokens.size() >
      inner_->ContextSize()) {
    if (error_msg != nullptr) {
      *error_msg = "flash-next Forward exceeds the session context";
    }
    return false;
  }
  std::vector<std::int32_t> prompt(committed.begin(),
                                   committed.begin() + start_pos);
  prompt.insert(prompt.end(), tokens.begin(), tokens.end());
  return Sync(prompt, error_msg);
}

std::span<const float> FlashNextEngineSession::Logits() const noexcept {
  if (inner_ == nullptr) return {};
  return inner_->Logits();
}

std::uint32_t FlashNextEngineSession::Length() const noexcept {
  if (inner_ == nullptr) return 0;
  return inner_->Position();
}

std::span<const std::int32_t> FlashNextEngineSession::Tokens() const noexcept {
  if (inner_ == nullptr) return {};
  return inner_->Tokens();
}

std::int32_t FlashNextEngineSession::SampleGreedy() {
  const std::span<const float> logits = Logits();
  if (logits.empty()) return model_->EosToken();  // unsynced session: stop
  // Default config is temperature 0 (greedy argmax, no RNG draw), matching
  // the deterministic native draft-verify path.
  return static_cast<std::int32_t>(sampler_.Sample(logits));
}

bool FlashNextEngineSession::DecodeStep(std::uint32_t max_draft_tokens,
                                        core::engine::DecodeOutcome* outcome,
                                        std::string* error_msg) {
  if (outcome == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "flash-next DecodeStep: null outcome";
    }
    return false;
  }
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "flash-next session is closed";
    return false;
  }
  // L1 carries no sampler: the session-owned greedy state goes in, and the
  // native entry accepts committed tokens into it, so serial DecodeStep and
  // the DecodeBatch bridge below share one sampler discipline. Zero means
  // the model-configured draft budget; without an MTP sidecar the native
  // entry runs ordinary single-token decode through this same path.
  const std::size_t budget =
      max_draft_tokens == 0 ? default_drafts_ : max_draft_tokens;
  Session::DecodeResult result;
  if (!inner_->DecodeStep(budget, sampler_, &result, error_msg,
                          /*stop_at_eos=*/true)) {
    return false;
  }
  if (result.tokens.empty()) {
    // An empty commit means the greedy anchor is a stop token (native
    // reports stop with no committed rows, e.g. a bare complete sentence
    // with no continuation). Commit that anchor through Evaluate so the
    // L1 step still advances by exactly one token: the same
    // commit-then-stop the plain single-token fallback of the sibling
    // adapters produces. A non-stop empty result (e.g. full context)
    // stays an error, refused by Evaluate the same way.
    if (Logits().empty()) {
      if (error_msg != nullptr) {
        *error_msg =
            "flash-next DecodeStep produced no tokens (Sync a prompt first)";
      }
      return false;
    }
    const std::int32_t anchor = SampleGreedy();
    if (!model_->IsStopToken(anchor)) {
      if (error_msg != nullptr) {
        *error_msg =
            "flash-next DecodeStep produced no tokens without a stop";
      }
      return false;
    }
    if (!Evaluate(anchor, error_msg)) return false;
    outcome->token = anchor;
    outcome->stop = true;
    outcome->tokens_committed = 1;
    return true;
  }
  outcome->token = result.tokens.back();
  outcome->stop = result.stop;
  outcome->tokens_committed =
      static_cast<std::uint32_t>(result.tokens.size());
  return true;
}

std::size_t FlashNextEngineSession::SnapshotPayloadBytes() const {
  if (inner_ == nullptr) return 0;
  return static_cast<std::size_t>(inner_->SnapshotBytes());
}

bool FlashNextEngineSession::Snapshot(std::span<std::uint8_t> destination,
                                      std::string* error_msg) const {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "flash-next session is closed";
    return false;
  }
  // Materialized: the payload is self-contained (no borrowed live rows),
  // so it restores after the source session is gone.
  auto snapshot = inner_->SaveSnapshot(
      error_msg, Session::SnapshotMode::kMaterialized);
  if (!snapshot) return false;
  if (snapshot->SizeBytes() != destination.size()) {
    if (error_msg != nullptr) {
      *error_msg = "flash-next Snapshot size changed between PayloadBytes (" +
                   std::to_string(destination.size()) + ") and capture (" +
                   std::to_string(snapshot->SizeBytes()) + ")";
    }
    return false;
  }
  return snapshot->CopyTo(destination);
}

bool FlashNextEngineSession::RestoreSnapshot(
    std::span<const std::uint8_t> payload, std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "flash-next session is closed";
    return false;
  }
  // The native payload carries kSnapshotPayloadVersion plus the KV format
  // and refuses mismatches internally; no adapter-side parsing needed.
  if (!inner_->RestoreSnapshot(payload, error_msg)) return false;
  ResetSamplerHistory(inner_->Tokens());
  return true;
}

bool FlashNextEngineSession::DecodeBatch(
    std::span<const DecodeRequest> requests, std::string* error_msg) {
  if (requests.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "flash-next DecodeBatch: empty batch";
    }
    return false;
  }
  std::vector<Session::DecodeRequest> native;
  native.reserve(requests.size());
  for (const auto& request : requests) {
    if (request.session == nullptr ||
        request.session->inner_ == nullptr || request.result == nullptr) {
      if (error_msg != nullptr) {
        *error_msg =
            "flash-next DecodeBatch: null session or result in batch";
      }
      return false;
    }
    if (request.session->model_.get() != requests[0].session->model_.get()) {
      if (error_msg != nullptr) {
        *error_msg = "flash-next DecodeBatch: batch spans more than one model";
      }
      return false;
    }
    native.push_back({request.session->inner_.get(), request.max_tokens,
                      &request.session->sampler_, request.result,
                      request.stop_at_eos, request.outcome});
  }
  return Session::DecodeBatch(native, error_msg);
}

bool FlashNextEngineSession::EvaluateBatch(
    std::span<const AdvanceRequest> requests, std::string* error_msg) {
  if (requests.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "flash-next EvaluateBatch: empty batch";
    }
    return false;
  }
  std::vector<Session::AdvanceRequest> native;
  native.reserve(requests.size());
  // The native batch writes per-peer outcomes; always supply local storage
  // and copy back only where the caller asked, so a null caller outcome is
  // still safe.
  std::vector<Session::BatchOutcome> outcomes(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    if (request.session == nullptr ||
        request.session->inner_ == nullptr) {
      if (error_msg != nullptr) {
        *error_msg = "flash-next EvaluateBatch: null session in batch";
      }
      return false;
    }
    if (request.session->model_.get() != requests[0].session->model_.get()) {
      if (error_msg != nullptr) {
        *error_msg =
            "flash-next EvaluateBatch: batch spans more than one model";
      }
      return false;
    }
    native.push_back(
        {request.session->inner_.get(), request.token, &outcomes[i]});
  }
  const bool ok = Session::EvaluateBatch(native, error_msg);
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (requests[i].outcome != nullptr) {
      *requests[i].outcome = outcomes[i];
    }
    if (requests[i].session != nullptr &&
        requests[i].session->inner_ != nullptr && outcomes[i].completed &&
        requests[i].token >= 0) {
      requests[i].session->sampler_.Accept(
          static_cast<sampling::TokenId>(requests[i].token));
    }
  }
  return ok;
}

std::string FlashNextEngineSession::Decode(
    std::span<const std::int32_t> tokens) const {
  return model_->Decode(tokens);
}

bool FlashNextEngineSession::IsStopToken(std::int32_t token) const noexcept {
  return model_->IsStopToken(token);
}

std::int32_t FlashNextEngineSession::EosToken() const noexcept {
  return model_->EosToken();
}

bool FlashNextEngineSession::HasMtp() const noexcept {
  return model_->HasMtp();
}

bool FlashNextEngineSession::SyncThrough(
    std::span<const std::int32_t> prompt, std::uint32_t boundary,
    std::unique_ptr<SessionSnapshot>* checkpoint, std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "flash-next session is closed";
    return false;
  }
  if (!inner_->SyncThrough(prompt, boundary, checkpoint, error_msg)) {
    return false;
  }
  ResetSamplerHistory(inner_->Tokens());
  return true;
}

void FlashNextEngineSession::ResetDraftPolicy() {
  if (inner_ != nullptr) inner_->ResetDraftPolicy();
}

const Session::SpeculativeStats& FlashNextEngineSession::Statistics()
    const noexcept {
  return inner_->Statistics();
}

FlashNextEngineModel::FlashNextEngineModel(std::shared_ptr<Model> model,
                                           ModelOptions options)
    : model_(std::move(model)), options_(std::move(options)) {}

std::shared_ptr<FlashNextEngineModel> FlashNextEngineModel::Load(
    const std::string& model_path, const ModelOptions& options,
    std::string* error_msg) {
  auto inner = Model::Load(model_path, options, error_msg);
  if (!inner) return nullptr;
  return std::shared_ptr<FlashNextEngineModel>(
      new FlashNextEngineModel(std::move(inner), options));
}

std::shared_ptr<FlashNextEngineModel> FlashNextEngineModel::Load(
    const std::string& model_path, std::string* error_msg) {
  return Load(model_path, ModelOptions{}, error_msg);
}

std::unique_ptr<core::engine::EngineSession>
FlashNextEngineModel::CreateSession(std::string* error_msg) {
  return CreateSession(core::SessionMode::kAutoregressive,
                       model_->MaxContext(), error_msg);
}

std::unique_ptr<core::engine::EngineSession>
FlashNextEngineModel::CreateSession(core::SessionMode mode,
                                    std::uint32_t max_context,
                                    std::string* error_msg) {
  auto inner = model_->CreateSession(mode, max_context, error_msg);
  if (!inner) return nullptr;
  return std::unique_ptr<core::engine::EngineSession>(
      new FlashNextEngineSession(model_, std::move(inner),
                                 options_.max_draft_tokens));
}

std::vector<std::int32_t> FlashNextEngineModel::Tokenize(
    std::string_view text, bool add_bos) const {
  if (!add_bos) return model_->Tokenize(text);
  // Native Model::Tokenize has no add_bos switch; honor the L0 flag through
  // the shared tokenizer options (same convention as the llama/gemma4
  // adapters).
  tokenization::TokenizerOptions options;
  options.add_bos = true;
  options.parse_special_tokens = true;
  const auto ids = model_->tokenizer().Encode(text, options);
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::uint32_t FlashNextEngineModel::VocabSize() const noexcept {
  return model_->VocabSize();
}

std::uint32_t FlashNextEngineModel::MaxContext() const noexcept {
  return model_->MaxContext();
}

std::size_t FlashNextEngineModel::ResidentBytes() const noexcept {
  return model_->ResidentBytes();
}

std::string FlashNextEngineModel::ModelName() const {
  return model_->ModelName();
}

std::string FlashNextEngineModel::Decode(
    std::span<const std::int32_t> tokens) const {
  return model_->Decode(tokens);
}

bool FlashNextEngineModel::IsStopToken(std::int32_t token) const noexcept {
  return model_->IsStopToken(token);
}

std::int32_t FlashNextEngineModel::EosToken() const noexcept {
  return model_->EosToken();
}

bool FlashNextEngineModel::HasMtp() const noexcept {
  return model_->HasMtp();
}

float FlashNextEngineModel::MtpConfidenceGate() noexcept {
  return Model::MtpConfidenceGate();
}

namespace {

// One-line "qwen38-flash-next" registration: maps the generic load request
// onto flash-next ModelOptions (zero/empty fields keep engine defaults:
// f16 KV, 4K context; "q8" selects the blocked INT8 KV path). The request
// carries no MTP sidecar, vision weights, rope scaling, or session mode, so
// registry sessions are autoregressive text sessions; MTP/vision callers
// Load directly with full ModelOptions.
const core::engine::EngineRegistrar kRegisterFlashNextEngine(
    core::engine::EngineBackendInfo{
        "qwen38-flash-next", "Qwen3.8-Flash-Next text (f16/q8 KV)"},
    [](const core::engine::EngineLoadRequest& request,
       std::string* error_msg) -> std::shared_ptr<core::engine::EngineModel> {
      ModelOptions options;
      if (request.max_context != 0) {
        options.max_context = request.max_context;
      }
      if (!request.kv_cache_quant.empty()) {
        options.kv_cache_quant = request.kv_cache_quant;
      }
      return FlashNextEngineModel::Load(request.model_path, options,
                                        error_msg);
    });

}  // namespace

}  // namespace gufo::models::qwen38_flash_next
