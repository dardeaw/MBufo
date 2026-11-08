// Llama pilot adapter: thin forwarding over llama::Model/Session plus the
// one-line "llama" EngineRegistry registration.

#include "src/models/llama/engine_adapter.hpp"

#include <utility>

namespace gufo::models::llama {

LlamaEngineSession::LlamaEngineSession(std::shared_ptr<Model> model,
                                       std::unique_ptr<Session> inner)
    : model_(std::move(model)), inner_(std::move(inner)) {}

bool LlamaEngineSession::Sync(std::span<const std::int32_t> prompt,
                              std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "llama session is closed";
    return false;
  }
  return inner_->Sync(prompt, error_msg);
}

bool LlamaEngineSession::Evaluate(std::int32_t token,
                                  std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "llama session is closed";
    return false;
  }
  return inner_->Evaluate(token, error_msg);
}

bool LlamaEngineSession::Forward(std::span<const std::int32_t> tokens,
                                 std::uint32_t start_pos,
                                 std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "llama session is closed";
    return false;
  }
  return inner_->Forward(tokens, start_pos, error_msg);
}

std::span<const float> LlamaEngineSession::Logits() const noexcept {
  return inner_->Logits();
}

std::uint32_t LlamaEngineSession::Length() const noexcept {
  return inner_->Length();
}

std::int32_t LlamaEngineSession::SampleGreedy() {
  return inner_->SampleGreedy();
}

bool LlamaEngineSession::DecodeStep(std::uint32_t max_draft_tokens,
                                     core::engine::DecodeOutcome* outcome,
                                     std::string* error_msg) {
  (void)max_draft_tokens;  // No draft path on llama; plain fallback ignores it.
  if (outcome == nullptr) {
    if (error_msg != nullptr) *error_msg = "llama DecodeStep: null outcome";
    return false;
  }
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "llama session is closed";
    return false;
  }
  const std::int32_t token = inner_->SampleGreedy();
  if (!inner_->Evaluate(token, error_msg)) return false;
  outcome->token = token;
  outcome->stop = model_->IsStopToken(token);
  outcome->tokens_committed = 1;
  return true;
}

std::string LlamaEngineSession::Decode(
    std::span<const std::int32_t> tokens) const {
  return model_->Decode(tokens);
}

bool LlamaEngineSession::IsStopToken(std::int32_t token) const noexcept {
  return model_->IsStopToken(token);
}

LlamaEngineModel::LlamaEngineModel(std::shared_ptr<Model> model)
    : model_(std::move(model)) {}

std::shared_ptr<LlamaEngineModel> LlamaEngineModel::Load(
    const std::string& model_path, const ModelOptions& options,
    std::string* error_msg) {
  auto inner = Model::Load(model_path, options, error_msg);
  if (!inner) return nullptr;
  return std::shared_ptr<LlamaEngineModel>(
      new LlamaEngineModel(std::move(inner)));
}

std::shared_ptr<LlamaEngineModel> LlamaEngineModel::Load(
    const std::string& model_path, std::string* error_msg) {
  return Load(model_path, ModelOptions{}, error_msg);
}

std::unique_ptr<core::engine::EngineSession> LlamaEngineModel::CreateSession(
    std::string* error_msg) {
  auto inner = model_->CreateSession(error_msg);
  if (!inner) return nullptr;
  return std::unique_ptr<core::engine::EngineSession>(
      new LlamaEngineSession(model_, std::move(inner)));
}

std::vector<std::int32_t> LlamaEngineModel::Tokenize(
    std::string_view text, bool add_bos) const {
  return model_->Tokenize(text, add_bos);
}

std::uint32_t LlamaEngineModel::VocabSize() const noexcept {
  return model_->VocabSize();
}

std::uint32_t LlamaEngineModel::MaxContext() const noexcept {
  return model_->MaxContext();
}

std::size_t LlamaEngineModel::ResidentBytes() const noexcept {
  return model_->ResidentBytes();
}

std::string LlamaEngineModel::ModelName() const {
  return model_->ModelName();
}

std::string LlamaEngineModel::Decode(
    std::span<const std::int32_t> tokens) const {
  return model_->Decode(tokens);
}

bool LlamaEngineModel::IsStopToken(std::int32_t token) const noexcept {
  return model_->IsStopToken(token);
}

namespace {

// One-line "llama" registration: maps the generic load request onto
// llama::ModelOptions (zero/empty fields keep engine defaults).
const core::engine::EngineRegistrar kRegisterLlamaEngine(
    core::engine::EngineBackendInfo{
        "llama", "Llama-3.1 dense GQA decoder (f16/q8 KV)"},
    [](const core::engine::EngineLoadRequest& request,
       std::string* error_msg) -> std::shared_ptr<core::engine::EngineModel> {
      ModelOptions options;
      if (request.max_context != 0) {
        options.max_context = request.max_context;
      }
      if (!request.kv_cache_quant.empty()) {
        options.kv_cache_quant = request.kv_cache_quant;
      }
      return LlamaEngineModel::Load(request.model_path, options, error_msg);
    });

}  // namespace

}  // namespace gufo::models::llama
