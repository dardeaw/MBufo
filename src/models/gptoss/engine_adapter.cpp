// GPT-OSS adapter: thin forwarding over gptoss::Model/Session plus the
// one-line "gptoss" EngineRegistry registration.

#include "src/models/gptoss/engine_adapter.hpp"

#include <utility>

namespace mbufo::models::gptoss {

GptossEngineSession::GptossEngineSession(std::shared_ptr<Model> model,
                                         std::unique_ptr<Session> inner)
    : model_(std::move(model)), inner_(std::move(inner)) {}

bool GptossEngineSession::Sync(std::span<const std::int32_t> prompt,
                               std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "gptoss session is closed";
    return false;
  }
  return inner_->Sync(prompt, error_msg);
}

bool GptossEngineSession::Evaluate(std::int32_t token,
                                   std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "gptoss session is closed";
    return false;
  }
  return inner_->Evaluate(token, error_msg);
}

bool GptossEngineSession::Forward(std::span<const std::int32_t> tokens,
                                  std::uint32_t start_pos,
                                  std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "gptoss session is closed";
    return false;
  }
  return inner_->Forward(tokens, start_pos, error_msg);
}

std::span<const float> GptossEngineSession::Logits() const noexcept {
  return inner_->Logits();
}

std::uint32_t GptossEngineSession::Length() const noexcept {
  return inner_->Length();
}

std::int32_t GptossEngineSession::SampleGreedy() {
  return inner_->SampleGreedy();
}

bool GptossEngineSession::DecodeStep(std::uint32_t max_draft_tokens,
                                     core::engine::DecodeOutcome* outcome,
                                     std::string* error_msg) {
  (void)max_draft_tokens;  // No draft path on gptoss; plain fallback ignores it.
  if (outcome == nullptr) {
    if (error_msg != nullptr) *error_msg = "gptoss DecodeStep: null outcome";
    return false;
  }
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "gptoss session is closed";
    return false;
  }
  const std::int32_t token = inner_->SampleGreedy();
  if (!inner_->Evaluate(token, error_msg)) return false;
  outcome->token = token;
  outcome->stop = model_->IsStopToken(token);
  outcome->tokens_committed = 1;
  return true;
}

std::string GptossEngineSession::Decode(
    std::span<const std::int32_t> tokens) const {
  return model_->Decode(tokens);
}

bool GptossEngineSession::IsStopToken(std::int32_t token) const noexcept {
  return model_->IsStopToken(token);
}

GptossEngineModel::GptossEngineModel(std::shared_ptr<Model> model)
    : model_(std::move(model)) {}

std::shared_ptr<GptossEngineModel> GptossEngineModel::Load(
    const std::string& model_path, const ModelOptions& options,
    std::string* error_msg) {
  auto inner = Model::Load(model_path, options, error_msg);
  if (!inner) return nullptr;
  return std::shared_ptr<GptossEngineModel>(
      new GptossEngineModel(std::move(inner)));
}

std::shared_ptr<GptossEngineModel> GptossEngineModel::Load(
    const std::string& model_path, std::string* error_msg) {
  return Load(model_path, ModelOptions{}, error_msg);
}

std::unique_ptr<core::engine::EngineSession> GptossEngineModel::CreateSession(
    std::string* error_msg) {
  auto inner = model_->CreateSession(error_msg);
  if (!inner) return nullptr;
  return std::unique_ptr<core::engine::EngineSession>(
      new GptossEngineSession(model_, std::move(inner)));
}

std::vector<std::int32_t> GptossEngineModel::Tokenize(
    std::string_view text, bool add_bos) const {
  return model_->Tokenize(text, add_bos);
}

std::uint32_t GptossEngineModel::VocabSize() const noexcept {
  return model_->VocabSize();
}

std::uint32_t GptossEngineModel::MaxContext() const noexcept {
  return model_->MaxContext();
}

core::engine::MoeGeometry DescribeMoeGeometry(const Model& model) {
  const auto& w = model.weights();
  const auto& c = w.config;
  core::engine::MoeGeometry g;
  if (c.num_experts == 0) return g;
  g.experts = c.num_experts;
  g.top_k = c.experts_used;
  g.hidden = c.hidden_size;
  g.expert_ff = c.expert_ff;
  g.shared_ff = 0;
  for (const auto& l : w.layers) {
    if (l.gate_up_exps.size_bytes == 0) continue;
    g.AddLayer(static_cast<int>(l.gate_up_exps.type), static_cast<int>(l.gate_up_exps.type), static_cast<int>(l.down_exps.type));
  }
  return g;
}

core::engine::MoeKernelInfo DescribeMoeKernels() {
  core::engine::MoeKernelInfo k;
  k.origin = core::engine::MoeKernelOrigin::kCoreRowDot;
  k.gate_up_kernel = "core MoeDecodeGateUpActMxfp4 (RowDot<MXFP4> block math, SwiGLU-OAI, biases)";
  k.down_kernel = "core MoeDecodeDownEpiMxfp4 (RowDot<MXFP4> block math, weighted epilogue)";
  k.masks_inactive_slots = true;
  k.graph_capture_safe = true;
  return k;
}

core::engine::MoeGeometry GptossEngineModel::MoeGeometryInfo() const { return DescribeMoeGeometry(*model_); }
core::engine::MoeKernelInfo GptossEngineModel::MoeKernelsInfo() const { return DescribeMoeKernels(); }

std::size_t GptossEngineModel::ResidentBytes() const noexcept {
  return model_->ResidentBytes();
}


core::engine::AttentionGeometry DescribeAttentionGeometry(const Model& model) {
  const auto& c = model.weights().config;
  core::engine::AttentionGeometry g;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    core::engine::AttentionGroup a;
    a.heads = c.num_heads;
    a.kv_heads = c.num_kv_heads;
    a.head_dim = c.head_dim;
    a.window = c.LayerWindow(il);
    a.sink = true;
    g.AddLayer(a);
  }
  return g;
}

core::engine::AttentionKernelInfo DescribeAttentionKernels() {
  core::engine::AttentionKernelInfo k;
  k.decode = {core::engine::AttentionKernelOrigin::kCore, "core AttentionBlockedSinkRowChunk (direct Q8 read, sink + window)", "",
              "F16 KV keeps the earlier sink kernels", true};
  k.prefill = {core::engine::AttentionKernelOrigin::kModelPrivate, "gptoss sink flash prefill",
               "learned-sink softmax prefill is not in the core WMMA flash kernels yet", "", false};
  return k;
}

core::engine::AttentionGeometry GptossEngineModel::AttentionGeometryInfo() const { return DescribeAttentionGeometry(*model_); }
core::engine::AttentionKernelInfo GptossEngineModel::AttentionKernelsInfo() const { return DescribeAttentionKernels(); }

std::string GptossEngineModel::ModelName() const {
  return model_->ModelName();
}

std::string GptossEngineModel::Decode(
    std::span<const std::int32_t> tokens) const {
  return model_->Decode(tokens);
}

bool GptossEngineModel::IsStopToken(std::int32_t token) const noexcept {
  return model_->IsStopToken(token);
}

namespace {

// One-line "gptoss" registration: maps the generic load request onto
// gptoss::ModelOptions (zero/empty fields keep engine defaults).
const core::engine::EngineRegistrar kRegisterGptossEngine(
    core::engine::EngineBackendInfo{
        "gptoss", "GPT-OSS MoE decoder (q8 KV / f16 KV)", true},
    [](const core::engine::EngineLoadRequest& request,
       std::string* error_msg) -> std::shared_ptr<core::engine::EngineModel> {
      ModelOptions options;
      if (request.max_context != 0) {
        options.max_context = request.max_context;
      }
      if (!request.kv_cache_quant.empty()) {
        options.kv_cache_quant = request.kv_cache_quant;
      }
      return GptossEngineModel::Load(request.model_path, options, error_msg);
    });

}  // namespace

}  // namespace mbufo::models::gptoss
