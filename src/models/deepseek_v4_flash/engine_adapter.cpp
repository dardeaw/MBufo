// DeepSeek V4 Flash adapter: thin forwarding over deepseek_v4_flash::Model/Session plus the "deepseek_v4_flash"
// EngineRegistry registration. See engine_adapter.hpp for the mapping of the L0 surface onto the native API.

#include "src/models/deepseek_v4_flash/engine_adapter.hpp"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <utility>

#include "src/core/session_mode.hpp"

namespace gufo::models::deepseek_v4_flash {
namespace {

void SetError(std::string* error_msg, const char* msg) {
  if (error_msg != nullptr) *error_msg = msg;
}

}  // namespace

DeepSeekEngineSession::DeepSeekEngineSession(std::shared_ptr<Model> model, std::unique_ptr<Session> inner)
    : model_(std::move(model)), inner_(std::move(inner)) {}

bool DeepSeekEngineSession::Sync(std::span<const std::int32_t> prompt, std::string* error_msg) {
  if (inner_ == nullptr) {
    SetError(error_msg, "deepseek session is closed");
    return false;
  }
  InvalidateLogits();
  if (prompt.empty()) {
    SetError(error_msg, "DeepSeek V4 Flash prompt is empty");
    return false;
  }
  // The native Sync prefills at most PrefillCapacity() new tokens per call and reuses the already processed prefix,
  // so a long prompt is fed as growing prefixes (exactly what the server does).
  const std::size_t capacity = std::max<std::uint32_t>(1, inner_->PrefillCapacity());
  std::size_t next = 0;
  while (next < prompt.size()) {
    next = std::min(prompt.size(), next + capacity);
    if (!inner_->Sync(prompt.first(next), error_msg)) {
      history_.clear();
      return false;
    }
  }
  history_.assign(prompt.begin(), prompt.end());
  return true;
}

bool DeepSeekEngineSession::Evaluate(std::int32_t token, std::string* error_msg) {
  if (inner_ == nullptr) {
    SetError(error_msg, "deepseek session is closed");
    return false;
  }
  InvalidateLogits();
  if (!inner_->Evaluate(token, error_msg)) return false;
  history_.push_back(token);
  return true;
}

bool DeepSeekEngineSession::Forward(std::span<const std::int32_t> tokens, std::uint32_t start_pos,
                                    std::string* error_msg) {
  if (inner_ == nullptr) {
    SetError(error_msg, "deepseek session is closed");
    return false;
  }
  if (tokens.empty()) {
    SetError(error_msg, "deepseek Forward: no tokens");
    return false;
  }
  if (start_pos > history_.size()) {
    SetError(error_msg, "deepseek Forward: start_pos is past the committed tokens");
    return false;
  }
  std::vector<std::int32_t> prompt(history_.begin(), history_.begin() + start_pos);
  prompt.insert(prompt.end(), tokens.begin(), tokens.end());
  return Sync(prompt, error_msg);
}

std::span<const float> DeepSeekEngineSession::Logits() const noexcept {
  if (inner_ == nullptr) return {};
  if (!logits_valid_) {
    try {
      logits_ = inner_->CopyLogits();
    } catch (const std::exception&) {
      logits_.clear();
    }
    logits_valid_ = !logits_.empty();
  }
  return logits_;
}

std::uint32_t DeepSeekEngineSession::Length() const noexcept {
  const int pos = inner_ != nullptr ? inner_->Position() : 0;
  return pos > 0 ? static_cast<std::uint32_t>(pos) : 0U;
}

std::int32_t DeepSeekEngineSession::SampleGreedy() {
  std::uint64_t rng = 0;  // unused: temperature 0 is the native device argmax
  return inner_->SelectNext(0.0F, &rng);
}

bool DeepSeekEngineSession::DecodeStep(std::uint32_t max_draft_tokens, core::engine::DecodeOutcome* outcome,
                                       std::string* error_msg) {
  (void)max_draft_tokens;  // DSpark cycles stay model-side (DsparkEnabled / inner().DsparkStep)
  if (outcome == nullptr) {
    SetError(error_msg, "deepseek DecodeStep: null outcome");
    return false;
  }
  if (inner_ == nullptr) {
    SetError(error_msg, "deepseek session is closed");
    return false;
  }
  const std::int32_t token = SampleGreedy();
  if (!Evaluate(token, error_msg)) return false;
  outcome->token = token;
  outcome->stop = model_->IsStopToken(token);
  outcome->tokens_committed = 1;
  return true;
}

std::size_t DeepSeekEngineSession::SnapshotPayloadBytes() const {
  return inner_ != nullptr ? static_cast<std::size_t>(inner_->PayloadBytes()) : 0;
}

bool DeepSeekEngineSession::Snapshot(std::span<std::uint8_t> destination, std::string* error_msg) const {
  if (inner_ == nullptr) {
    SetError(error_msg, "deepseek session is closed");
    return false;
  }
  auto snapshot = inner_->SaveSnapshot(error_msg);
  if (snapshot == nullptr) return false;
  if (destination.size() != snapshot->SizeBytes() || !snapshot->CopyTo(destination)) {
    SetError(error_msg, "deepseek Snapshot: destination size does not match the payload");
    return false;
  }
  return true;
}

bool DeepSeekEngineSession::RestoreSnapshot(std::span<const std::uint8_t> payload, std::string* error_msg) {
  if (inner_ == nullptr) {
    SetError(error_msg, "deepseek session is closed");
    return false;
  }
  InvalidateLogits();
  // The restored state's token history is not part of the payload: Forward after a restore must start at Length()
  // with a prefix the caller tracks, so the mirror is dropped and Forward(start_pos > 0) reports it.
  history_.clear();
  return inner_->RestoreSnapshot(payload, error_msg);
}

bool DeepSeekEngineSession::DsparkEnabled() const { return inner_ != nullptr && inner_->DsparkEnabled(); }
bool DeepSeekEngineSession::IsStopToken(std::int32_t token) const { return model_->IsStopToken(token); }
std::string DeepSeekEngineSession::DecodeToken(std::int32_t token) const { return model_->DecodeToken(token); }

DeepSeekEngineModel::DeepSeekEngineModel(std::shared_ptr<Model> model, std::size_t resident_bytes)
    : model_(std::move(model)), resident_bytes_(resident_bytes) {}

std::shared_ptr<DeepSeekEngineModel> DeepSeekEngineModel::Load(const std::string& model_path,
                                                               const ModelOptions& options,
                                                               std::string* error_msg) {
  auto inner = Model::Load(model_path, options, error_msg);
  if (!inner) return nullptr;
  std::error_code ec;
  const auto bytes = std::filesystem::file_size(model_path, ec);  // memory-mapped GGUF: the file is the residency
  return std::shared_ptr<DeepSeekEngineModel>(
      new DeepSeekEngineModel(std::move(inner), ec ? 0 : static_cast<std::size_t>(bytes)));
}

std::unique_ptr<core::engine::EngineSession> DeepSeekEngineModel::CreateSession(std::string* error_msg) {
  auto inner = model_->CreateSession(core::SessionMode::kAutoregressive, model_->MaxContext(), error_msg);
  if (!inner) return nullptr;
  return std::make_unique<DeepSeekEngineSession>(model_, std::move(inner));
}

std::vector<std::int32_t> DeepSeekEngineModel::Tokenize(std::string_view text, bool add_bos) const {
  if (!add_bos) {
    const auto ids = model_->Tokenize(text, /*parse_special_tokens=*/false);
    return std::vector<std::int32_t>(ids.begin(), ids.end());
  }
  // DeepSeek's BOS is a literal special token in the rendered text; parse_special resolves it.
  const std::string rendered = std::string("<\xEF\xBD\x9C" "begin\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>") + std::string(text);
  const auto ids = model_->Tokenize(rendered, /*parse_special_tokens=*/true);
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::uint32_t DeepSeekEngineModel::VocabSize() const noexcept {
  const int v = model_->VocabSize();
  return v > 0 ? static_cast<std::uint32_t>(v) : 0U;
}
std::uint32_t DeepSeekEngineModel::MaxContext() const noexcept { return model_->MaxContext(); }
std::size_t DeepSeekEngineModel::ResidentBytes() const noexcept { return resident_bytes_; }
std::string DeepSeekEngineModel::ModelName() const { return model_->ModelName(); }
std::string DeepSeekEngineModel::DecodeToken(std::int32_t token) const { return model_->DecodeToken(token); }
bool DeepSeekEngineModel::IsStopToken(std::int32_t token) const { return model_->IsStopToken(token); }

namespace {

// "deepseek_v4_flash" registration: max_context 0 keeps the native default; the request's kv_cache_quant does not
// apply (DeepSeek's compressed KV is part of the model) and is ignored; no DSpark support model can come from a
// registry request.
const core::engine::EngineRegistrar kRegisterDeepSeekEngine(
    core::engine::EngineBackendInfo{"deepseek_v4_flash", "DeepSeek V4 Flash MoE text (private kernels)"},
    [](const core::engine::EngineLoadRequest& request,
       std::string* error_msg) -> std::shared_ptr<core::engine::EngineModel> {
      ModelOptions options;
      if (request.max_context != 0) options.max_context = request.max_context;
      return DeepSeekEngineModel::Load(request.model_path, options, error_msg);
    });

}  // namespace

}  // namespace gufo::models::deepseek_v4_flash