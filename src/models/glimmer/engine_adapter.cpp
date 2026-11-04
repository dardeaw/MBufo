// Glimmer adapter: thin forwarding over glimmer::Model/Session plus the
// one-line "glimmer" EngineRegistry registration.

#include "src/models/glimmer/engine_adapter.hpp"

#include <utility>

namespace gufo::models::glimmer {

GlimmerEngineSession::GlimmerEngineSession(std::shared_ptr<Model> model,
                                           std::unique_ptr<Session> inner)
    : model_(std::move(model)), inner_(std::move(inner)) {}

bool GlimmerEngineSession::Sync(std::span<const std::int32_t> prompt,
                                std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "glimmer session is closed";
    return false;
  }
  return inner_->Sync(prompt, error_msg);
}

bool GlimmerEngineSession::Evaluate(std::int32_t token,
                                    std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "glimmer session is closed";
    return false;
  }
  return inner_->Evaluate(token, error_msg);
}

bool GlimmerEngineSession::Forward(std::span<const std::int32_t> tokens,
                                   std::uint32_t start_pos,
                                   std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "glimmer session is closed";
    return false;
  }
  return inner_->Forward(tokens, start_pos, error_msg);
}

std::span<const float> GlimmerEngineSession::Logits() const noexcept {
  return inner_->Logits();
}

std::uint32_t GlimmerEngineSession::Length() const noexcept {
  return inner_->Length();
}

std::int32_t GlimmerEngineSession::SampleGreedy() {
  return inner_->SampleGreedy();
}

bool GlimmerEngineSession::DecodeStep(std::uint32_t max_draft_tokens,
                                      core::engine::DecodeOutcome* outcome,
                                      std::string* error_msg) {
  (void)max_draft_tokens;  // No draft path on the L1 contract; the native
                           // SpeculativeRound needs a caller-owned dflash
                           // backend, so the plain fallback ignores it.
  if (outcome == nullptr) {
    if (error_msg != nullptr) *error_msg = "glimmer DecodeStep: null outcome";
    return false;
  }
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "glimmer session is closed";
    return false;
  }
  const std::int32_t token = inner_->SampleGreedy();
  if (!inner_->Evaluate(token, error_msg)) return false;
  outcome->token = token;
  outcome->stop = model_->IsStopToken(token);
  outcome->tokens_committed = 1;
  return true;
}

std::string GlimmerEngineSession::Decode(
    std::span<const std::int32_t> tokens) const {
  return model_->Decode(tokens);
}

bool GlimmerEngineSession::IsStopToken(std::int32_t token) const noexcept {
  return model_->IsStopToken(token);
}

std::int32_t GlimmerEngineSession::EosToken() const noexcept {
  return model_->EosToken();
}

bool GlimmerEngineSession::Verify(std::span<const std::int32_t> tokens,
                                  std::uint32_t start_pos,
                                  std::vector<std::int32_t>& vbest,
                                  std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "glimmer session is closed";
    return false;
  }
  // Committing by engine contract: shares Forward mechanics (KV rows at
  // [start_pos, start_pos+n), Length advances by n); the speculative
  // accept path overwrites with its own Forwards.
  return inner_->Verify(tokens, start_pos, vbest, error_msg,
                        /*tap=*/nullptr);
}

bool GlimmerEngineSession::EnableDraftTaps(
    std::span<const std::uint32_t> layers, std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "glimmer session is closed";
    return false;
  }
  return inner_->EnableDraftTaps(layers, error_msg);
}

bool GlimmerEngineSession::CopyTappedRow(
    std::span<float> host_out, std::string* error_msg) const {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "glimmer session is closed";
    return false;
  }
  return inner_->CopyTappedRow(host_out, error_msg);
}

bool GlimmerEngineSession::SpeculativeRound(
    speculative::IDraftBackend& draft, std::span<const std::int32_t> prefix,
    std::uint32_t n_draft, std::vector<std::int32_t>* out,
    std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "glimmer session is closed";
    return false;
  }
  return inner_->SpeculativeRound(draft, prefix, n_draft, out, error_msg);
}

GlimmerEngineModel::GlimmerEngineModel(std::shared_ptr<Model> model)
    : model_(std::move(model)) {}

std::shared_ptr<GlimmerEngineModel> GlimmerEngineModel::Load(
    const std::string& model_path, const ModelOptions& options,
    std::string* error_msg) {
  auto inner = Model::Load(model_path, options, error_msg);
  if (!inner) return nullptr;
  return std::shared_ptr<GlimmerEngineModel>(
      new GlimmerEngineModel(std::move(inner)));
}

std::shared_ptr<GlimmerEngineModel> GlimmerEngineModel::Load(
    const std::string& model_path, std::string* error_msg) {
  return Load(model_path, ModelOptions{}, error_msg);
}

std::unique_ptr<core::engine::EngineSession> GlimmerEngineModel::CreateSession(
    std::string* error_msg) {
  auto inner = model_->CreateSession(error_msg);
  if (!inner) return nullptr;
  return std::unique_ptr<core::engine::EngineSession>(
      new GlimmerEngineSession(model_, std::move(inner)));
}

std::vector<std::int32_t> GlimmerEngineModel::Tokenize(
    std::string_view text, bool add_bos) const {
  return model_->Tokenize(text, add_bos);
}

std::uint32_t GlimmerEngineModel::VocabSize() const noexcept {
  return model_->VocabSize();
}

std::uint32_t GlimmerEngineModel::MaxContext() const noexcept {
  return model_->MaxContext();
}

std::size_t GlimmerEngineModel::ResidentBytes() const noexcept {
  return model_->ResidentBytes();
}

std::string GlimmerEngineModel::ModelName() const {
  return model_->ModelName();
}

std::string GlimmerEngineModel::Decode(
    std::span<const std::int32_t> tokens) const {
  return model_->Decode(tokens);
}

bool GlimmerEngineModel::IsStopToken(std::int32_t token) const noexcept {
  return model_->IsStopToken(token);
}

std::int32_t GlimmerEngineModel::EosToken() const noexcept {
  return model_->EosToken();
}

namespace {

// One-line "glimmer" registration: maps the generic load request onto
// glimmer::ModelOptions (zero/empty fields keep engine defaults, same
// convention as the llama pilot: f16 default, "q8" selects the blocked
// grid path with SWA pre-shift).
const core::engine::EngineRegistrar kRegisterGlimmerEngine(
    core::engine::EngineBackendInfo{
        "glimmer", "Muse Glimmer dense GQA+SWA decoder (f16/q8 KV)"},
    [](const core::engine::EngineLoadRequest& request,
       std::string* error_msg) -> std::shared_ptr<core::engine::EngineModel> {
      ModelOptions options;
      if (request.max_context != 0) {
        options.max_context = request.max_context;
      }
      if (!request.kv_cache_quant.empty()) {
        options.kv_cache_quant = request.kv_cache_quant;
      }
      return GlimmerEngineModel::Load(request.model_path, options, error_msg);
    });

}  // namespace

}  // namespace gufo::models::glimmer
