#include "src/models/laguna/engine_adapter.hpp"

#include <utility>

namespace mbufo::models::laguna {

LagunaEngineSession::LagunaEngineSession(std::shared_ptr<Model> model,
                                         std::unique_ptr<Session> inner)
    : model_(std::move(model)), inner_(std::move(inner)) {}

bool LagunaEngineSession::Sync(std::span<const std::int32_t> prompt,
                               std::string* error_msg) {
  return inner_->Sync(prompt, error_msg);
}

bool LagunaEngineSession::Evaluate(std::int32_t token, std::string* error_msg) {
  return inner_->Evaluate(token, error_msg);
}

bool LagunaEngineSession::Forward(std::span<const std::int32_t> tokens,
                                  std::uint32_t start_pos,
                                  std::string* error_msg) {
  return inner_->Forward(tokens, start_pos, error_msg);
}

std::span<const float> LagunaEngineSession::Logits() const noexcept {
  return inner_->Logits();
}

std::uint32_t LagunaEngineSession::Length() const noexcept {
  return inner_->Length();
}

std::int32_t LagunaEngineSession::SampleGreedy() {
  return inner_->SampleGreedy();
}

bool LagunaEngineSession::DecodeStep(std::uint32_t,
                                     core::engine::DecodeOutcome* outcome,
                                     std::string* error_msg) {
  const std::int32_t tok = inner_->SampleGreedy();
  if (!inner_->Evaluate(tok, error_msg)) return false;
  if (outcome != nullptr) {
    outcome->token = tok;
    outcome->stop = model_->IsStopToken(tok);
    outcome->tokens_committed = 1;
  }
  return true;
}

LagunaEngineModel::LagunaEngineModel(std::shared_ptr<Model> model)
    : model_(std::move(model)) {}

std::shared_ptr<LagunaEngineModel> LagunaEngineModel::Load(
    const std::string& model_path, const ModelOptions& options,
    std::string* error_msg) {
  auto m = Model::Load(model_path, options, error_msg);
  if (m == nullptr) return nullptr;
  return std::shared_ptr<LagunaEngineModel>(new LagunaEngineModel(std::move(m)));
}

std::unique_ptr<core::engine::EngineSession> LagunaEngineModel::CreateSession(
    std::string* error_msg) {
  auto s = model_->CreateSession(error_msg);
  if (s == nullptr) return nullptr;
  return std::make_unique<LagunaEngineSession>(model_, std::move(s));
}

std::vector<std::int32_t> LagunaEngineModel::Tokenize(std::string_view text,
                                                      bool add_bos) const {
  return model_->Tokenize(text, add_bos);
}

std::uint32_t LagunaEngineModel::VocabSize() const noexcept {
  return model_->VocabSize();
}

std::uint32_t LagunaEngineModel::MaxContext() const noexcept {
  return model_->MaxContext();
}

std::size_t LagunaEngineModel::ResidentBytes() const noexcept {
  return model_->ResidentBytes();
}

std::string LagunaEngineModel::ModelName() const {
  return model_->ModelName();
}

core::engine::AttentionGeometry LagunaEngineModel::AttentionGeometryInfo() const {
  const auto& cfg = model_->weights().config;
  core::engine::AttentionGeometry g;
  for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
    core::engine::AttentionGroup a;
    a.heads = cfg.q_heads_per_layer[l];
    a.kv_heads = cfg.num_kv_heads;
    a.head_dim = cfg.head_dim;
    a.window = cfg.is_swa_layer[l] ? cfg.sliding_window : 0U;
    a.sink = false;
    g.AddLayer(a);
  }
  return g;
}

core::engine::AttentionKernelInfo LagunaEngineModel::AttentionKernelsInfo() const {
  core::engine::AttentionKernelInfo k;
  k.decode = {core::engine::AttentionKernelOrigin::kModelPrivate,
              "laguna f16 SWA+per-head-gate decode", "", "", false};
  k.prefill = {core::engine::AttentionKernelOrigin::kModelPrivate,
               "laguna f16 SWA+per-head-gate prefill", "", "", false};
  return k;
}

core::engine::MoeGeometry LagunaEngineModel::MoeGeometryInfo() const {
  const auto& w = model_->weights();
  const auto& cfg = w.config;
  core::engine::MoeGeometry g;
  g.experts = cfg.expert_count;
  g.top_k = cfg.expert_used_count;
  g.hidden = cfg.hidden_size;
  g.expert_ff = cfg.expert_ffn_size;
  g.shared_ff = cfg.shared_expert_ffn_size;
  for (const auto& l : w.layers) {
    if (!l.is_moe) continue;
    g.AddLayer(static_cast<int>(l.ffn_gate_exps.type),
               static_cast<int>(l.ffn_up_exps.type),
               static_cast<int>(l.ffn_down_exps.type));
  }
  return g;
}

core::engine::MoeKernelInfo LagunaEngineModel::MoeKernelsInfo() const {
  core::engine::MoeKernelInfo k;
  k.origin = core::engine::MoeKernelOrigin::kCoreRowDot;
  k.gate_up_kernel = "core LaunchDecodeMoeKQuant / RoutedF16Gemm";
  k.down_kernel = "core LaunchDecodeMoeKQuant / RoutedF16Gemm";
  k.masks_inactive_slots = true;
  return k;
}

namespace {
const core::engine::EngineRegistrar kLagunaRegistrar(
    core::engine::EngineBackendInfo{
        .name = "laguna",
        .description = "Laguna 2.1 S/XS hybrid SWA+YaRN MoE (f16 KV)",
        .moe = true,
        .attention = true,
    },
    [](const core::engine::EngineLoadRequest& req, std::string* err)
        -> std::shared_ptr<core::engine::EngineModel> {
      ModelOptions opts;
      if (req.max_context > 0) opts.max_context = req.max_context;
      if (!req.kv_cache_quant.empty()) opts.kv_cache_quant = req.kv_cache_quant;
      return LagunaEngineModel::Load(req.model_path, opts, err);
    });
}  // namespace

}  // namespace mbufo::models::laguna
