#include "src/models/inkling/engine_adapter.hpp"

#include <utility>

namespace mbufo::models::inkling {

InklingEngineSession::InklingEngineSession(std::shared_ptr<Model> model,
                                           std::unique_ptr<Session> inner)
    : model_(std::move(model)), inner_(std::move(inner)) {}

bool InklingEngineSession::Sync(std::span<const std::int32_t> prompt,
                                std::string* error_msg) {
  return inner_->Sync(prompt, error_msg);
}

bool InklingEngineSession::Evaluate(std::int32_t token, std::string* error_msg) {
  return inner_->Evaluate(token, error_msg);
}

bool InklingEngineSession::Forward(std::span<const std::int32_t> tokens,
                                    std::uint32_t start_pos,
                                    std::string* error_msg) {
  return inner_->Forward(tokens, start_pos, error_msg);
}

std::span<const float> InklingEngineSession::Logits() const noexcept {
  return inner_->Logits();
}

std::uint32_t InklingEngineSession::Length() const noexcept {
  return inner_->Length();
}

std::int32_t InklingEngineSession::SampleGreedy() {
  return inner_->SampleGreedy();
}

bool InklingEngineSession::DecodeStep(std::uint32_t,
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

InklingEngineModel::InklingEngineModel(std::shared_ptr<Model> model)
    : model_(std::move(model)) {}

std::shared_ptr<InklingEngineModel> InklingEngineModel::Load(
    const std::string& model_path, const ModelOptions& options,
    std::string* error_msg) {
  auto m = Model::Load(model_path, options, error_msg);
  if (m == nullptr) return nullptr;
  return std::shared_ptr<InklingEngineModel>(new InklingEngineModel(std::move(m)));
}

std::unique_ptr<core::engine::EngineSession> InklingEngineModel::CreateSession(
    std::string* error_msg) {
  auto s = model_->CreateSession(error_msg);
  if (s == nullptr) return nullptr;
  return std::make_unique<InklingEngineSession>(model_, std::move(s));
}

std::vector<std::int32_t> InklingEngineModel::Tokenize(
    std::string_view text, bool add_bos) const {
  return model_->Tokenize(text, add_bos);
}

std::uint32_t InklingEngineModel::VocabSize() const noexcept {
  return model_->VocabSize();
}

std::uint32_t InklingEngineModel::MaxContext() const noexcept {
  return model_->MaxContext();
}

std::size_t InklingEngineModel::ResidentBytes() const noexcept {
  return model_->ResidentBytes();
}

std::string InklingEngineModel::ModelName() const {
  return model_->ModelName();
}

core::engine::AttentionGeometry InklingEngineModel::AttentionGeometryInfo() const {
  const auto& cfg = model_->weights().config;
  core::engine::AttentionGeometry g;
  for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
    core::engine::AttentionGroup a;
    a.heads = cfg.num_q_heads;
    a.kv_heads = cfg.num_kv_heads;
    a.head_dim = cfg.head_dim;
    // is_swa_layer flags the special 5/11/17/23 GLOBAL layers (full
    // attention); the other 20 sliding layers carry the 1024 window.
    a.window = cfg.is_swa_layer[l] ? 0U : cfg.sliding_window;
    a.sink = true;  // per-token per-head attn_sink_proj on every layer
    g.AddLayer(a);
  }
  return g;
}

core::engine::AttentionKernelInfo InklingEngineModel::AttentionKernelsInfo() const {
  core::engine::AttentionKernelInfo k;
  k.decode = {core::engine::AttentionKernelOrigin::kModelPrivate,
              "inkling: static/dynamic sconv + per-token sink + qk/rel clip + "
              "rel-bias gather decode (core split-K refused: qpg 3, per-token "
              "sinks)",
              "", "", false};
  k.prefill = {core::engine::AttentionKernelOrigin::kModelPrivate,
               "inkling: fused causal prefill with SWA window + per-token "
               "sink + rel-bias gather",
               "", "", false};
  return k;
}

core::engine::MoeGeometry InklingEngineModel::MoeGeometryInfo() const {
  const auto& w = model_->weights();
  const auto& cfg = w.config;
  core::engine::MoeGeometry g;
  g.experts = cfg.expert_count;
  g.top_k = cfg.expert_used_count;
  g.hidden = cfg.hidden_size;
  g.expert_ff = cfg.expert_ffn_size;
  g.shared_ff = 0;  // inkling has no shared experts
  for (const auto& l : w.layers) {
    g.AddLayer(static_cast<int>(l.ffn_gate_up_exps.type),
               static_cast<int>(l.ffn_gate_up_exps.type),
               static_cast<int>(l.ffn_down_exps.type));
  }
  return g;
}

core::engine::MoeKernelInfo InklingEngineModel::MoeKernelsInfo() const {
  core::engine::MoeKernelInfo k;
  k.origin = core::engine::MoeKernelOrigin::kCoreRowDot;
  k.gate_up_kernel = "core QuantGemmPrefill/UnquantizedF16Gemm per-expert "
                     "(F16 experts; Q8_0 via core dense path) + fused situ";
  k.down_kernel = "core QuantGemmPrefill/DenseF16Gemm per-expert + core "
                  "MoeEpilogue-style weighted scatter";
  k.masks_inactive_slots = false;
  return k;
}

namespace {
const core::engine::EngineRegistrar kInklingRegistrar(
    core::engine::EngineBackendInfo{
        .name = "inkling",
        .description = "Inkling-2-Mini hybrid sconv+loop MoE (f16/Q8/K8V4 KV)",
        .moe = true,
        .attention = true,
    },
    [](const core::engine::EngineLoadRequest& req, std::string* err)
        -> std::shared_ptr<core::engine::EngineModel> {
      ModelOptions opts;
      if (req.max_context > 0) opts.max_context = req.max_context;
      if (!req.kv_cache_quant.empty()) opts.kv_cache_quant = req.kv_cache_quant;
      opts.kv_cache_quant_supplied = req.kv_cache_quant_supplied;
      return InklingEngineModel::Load(req.model_path, opts, err);
    });
}  // namespace

}  // namespace mbufo::models::inkling