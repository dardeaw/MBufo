// Gemma4 adapter: thin forwarding over gemma4::Model/Session plus the
// one-line "gemma4" EngineRegistry registration.

#include "src/models/gemma4/engine_adapter.hpp"

#include <utility>
#include <vector>

namespace mbufo::models::gemma4 {

GemmaEngineSession::GemmaEngineSession(std::shared_ptr<Model> model,
                                       std::unique_ptr<Session> inner)
    : model_(std::move(model)), inner_(std::move(inner)) {}

bool GemmaEngineSession::Sync(std::span<const std::int32_t> prompt,
                              std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "gemma4 session is closed";
    return false;
  }
  // Native incremental-prefix Sync (append-only growth forwards only the
  // delta; anything else fully resyncs); matches the L0 Sync contract.
  return inner_->Sync(prompt, error_msg);
}

bool GemmaEngineSession::Evaluate(std::int32_t token,
                                  std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "gemma4 session is closed";
    return false;
  }
  return inner_->Evaluate(token, error_msg);
}

bool GemmaEngineSession::Forward(std::span<const std::int32_t> tokens,
                                 std::uint32_t start_pos,
                                 std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "gemma4 session is closed";
    return false;
  }
  // Native all_logits defaults to false; L0 has no per-position logits.
  return inner_->Forward(tokens, start_pos, error_msg);
}

std::span<const float> GemmaEngineSession::Logits() const noexcept {
  return inner_->Logits();
}

std::uint32_t GemmaEngineSession::Length() const noexcept {
  return inner_->Length();
}

std::int32_t GemmaEngineSession::SampleGreedy() {
  // Native SampleGreedy is const; the L0 override is not.
  return inner_->SampleGreedy();
}

bool GemmaEngineSession::DecodeStep(std::uint32_t max_draft_tokens,
                                    core::engine::DecodeOutcome* outcome,
                                    std::string* error_msg) {
  (void)max_draft_tokens;  // No draft path on the L1 contract: the native
                           // SpeculativeRound needs EnableMtp(drafts_path)
                           // first, which a registry request cannot supply,
                           // so the plain fallback ignores it.
  if (outcome == nullptr) {
    if (error_msg != nullptr) *error_msg = "gemma4 DecodeStep: null outcome";
    return false;
  }
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "gemma4 session is closed";
    return false;
  }
  const std::int32_t token = inner_->SampleGreedy();
  if (!inner_->Evaluate(token, error_msg)) return false;
  outcome->token = token;
  outcome->stop = model_->IsStopToken(token);
  outcome->tokens_committed = 1;
  return true;
}

bool GemmaEngineSession::DecodeBatch(
    std::span<GemmaEngineSession*> sessions,
    std::span<const std::int32_t> tokens, std::string* error_msg) {
  // Translate the adapter-side batch (one token per adapter session) onto
  // the native parallel-array signature (span<Session*>,
  // span<const int32_t>). Glimmer instead takes BatchItem{session, token
  // span} rows with variable widths; gemma4 is decode-only, one token
  // per session, so the width check below is exact, not per-row.
  if (sessions.size() != tokens.size()) {
    if (error_msg != nullptr) {
      *error_msg = "gemma4 DecodeBatch: sessions (" +
                   std::to_string(sessions.size()) + ") and tokens (" +
                   std::to_string(tokens.size()) + ") sizes differ";
    }
    return false;
  }
  if (sessions.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "gemma4 DecodeBatch: empty batch";
    }
    return false;
  }
  std::vector<Session*> inners;
  inners.reserve(sessions.size());
  for (const auto* s : sessions) {
    if (s == nullptr || s->inner_ == nullptr) {
      if (error_msg != nullptr) {
        *error_msg = "gemma4 DecodeBatch: null or closed session in batch";
      }
      return false;
    }
    if (s->model_.get() != sessions[0]->model_.get()) {
      if (error_msg != nullptr) {
        *error_msg = "gemma4 DecodeBatch: batch spans more than one model";
      }
      return false;
    }
    inners.push_back(s->inner_.get());
  }
  return Session::DecodeBatch(inners, tokens, error_msg);
}

std::string GemmaEngineSession::Decode(
    std::span<const std::int32_t> tokens) const {
  return model_->Decode(tokens);
}

bool GemmaEngineSession::IsStopToken(std::int32_t token) const noexcept {
  return model_->IsStopToken(token);
}

std::int32_t GemmaEngineSession::EosToken() const noexcept {
  return model_->EosToken();
}

bool GemmaEngineSession::EnableMtp(const std::string& drafts_path,
                                   std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "gemma4 session is closed";
    return false;
  }
  return inner_->EnableMtp(drafts_path, error_msg);
}

bool GemmaEngineSession::MtpEnabled() const noexcept {
  return inner_ != nullptr && inner_->MtpEnabled();
}

bool GemmaEngineSession::SpeculativeRound(std::int32_t last_token,
                                          std::uint32_t n_draft,
                                          std::vector<std::int32_t>* out,
                                          std::string* error_msg) {
  if (inner_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "gemma4 session is closed";
    return false;
  }
  return inner_->SpeculativeRound(last_token, n_draft, out, error_msg);
}

GemmaEngineModel::GemmaEngineModel(std::shared_ptr<Model> model)
    : model_(std::move(model)) {}

std::shared_ptr<GemmaEngineModel> GemmaEngineModel::Load(
    const std::string& model_path, const ModelOptions& options,
    std::string* error_msg) {
  auto inner = Model::Load(model_path, options, error_msg);
  if (!inner) return nullptr;
  return std::shared_ptr<GemmaEngineModel>(
      new GemmaEngineModel(std::move(inner)));
}

std::shared_ptr<GemmaEngineModel> GemmaEngineModel::Load(
    const std::string& model_path, std::string* error_msg) {
  return Load(model_path, ModelOptions{}, error_msg);
}

std::unique_ptr<core::engine::EngineSession> GemmaEngineModel::CreateSession(
    std::string* error_msg) {
  auto inner = model_->CreateSession(error_msg);
  if (!inner) return nullptr;
  return std::unique_ptr<core::engine::EngineSession>(
      new GemmaEngineSession(model_, std::move(inner)));
}

std::vector<std::int32_t> GemmaEngineModel::Tokenize(
    std::string_view text, bool add_bos) const {
  if (!add_bos) return model_->Tokenize(text);
  // Native Model::Tokenize has no add_bos switch; honor the L0 flag through
  // the shared tokenizer options (same convention as the llama adapter).
  tokenization::TokenizerOptions opts;
  opts.add_bos = true;
  opts.parse_special_tokens = true;
  const auto ids = model_->tokenizer().Encode(text, opts);
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::uint32_t GemmaEngineModel::VocabSize() const noexcept {
  return model_->VocabSize();
}

std::uint32_t GemmaEngineModel::MaxContext() const noexcept {
  return model_->MaxContext();
}

core::engine::MoeGeometry DescribeMoeGeometry(const Model& model) {
  const auto& w = model.weights();
  const auto& c = w.config;
  core::engine::MoeGeometry g;
  if (c.num_experts == 0) return g;  // dense Gemma 4: no routed experts
  g.experts = c.num_experts;
  g.top_k = c.experts_used;
  g.hidden = c.hidden_size;
  g.expert_ff = c.expert_ff;
  g.shared_ff = c.shared_ff;
  for (const auto& l : w.layers) {
    if (l.gate_up_exps.size_bytes == 0) continue;  // dense-FFN layer
    // gate and up are one merged tensor
    g.AddLayer(static_cast<int>(l.gate_up_exps.type), static_cast<int>(l.gate_up_exps.type), static_cast<int>(l.down_exps.type));
  }
  return g;
}

core::engine::MoeKernelInfo DescribeMoeKernels() {
  core::engine::MoeKernelInfo k;
  k.origin = core::engine::MoeKernelOrigin::kCoreRowDot;
  k.gate_up_kernel = "core MoeWaveGateUp (lane geometry, RowDot<Q4_0>, GeLU-tanh); grouped Q8_1 kernel for verification batches";
  k.down_kernel = "core MoeWaveDown (lane geometry, RowDot<Q4_0>); grouped Q8_1 kernel for verification batches";
  k.masks_inactive_slots = true;
  k.graph_capture_safe = true;
  return k;
}

core::engine::MoeGeometry GemmaEngineModel::MoeGeometryInfo() const { return DescribeMoeGeometry(*model_); }
core::engine::MoeKernelInfo GemmaEngineModel::MoeKernelsInfo() const { return DescribeMoeKernels(); }

std::size_t GemmaEngineModel::ResidentBytes() const noexcept {
  return model_->ResidentBytes();
}


core::engine::AttentionGeometry DescribeAttentionGeometry(const Model& model) {
  const auto& c = model.weights().config;
  core::engine::AttentionGeometry g;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const std::uint32_t kvh = c.KvHeads(c.KvProducingLayer(il));  // shared-KV layers attend over the producer's cache
    if (kvh == 0) continue;
    core::engine::AttentionGroup a;
    a.heads = c.num_heads;
    a.kv_heads = kvh;
    a.head_dim = c.HeadDim(il);
    a.window = c.LayerIsSwa(il) ? c.sliding_window : 0;
    a.kv_format = a.head_dim == 256 ? core::engine::AttentionKvFormat::kEither : core::engine::AttentionKvFormat::kF16;
    g.AddLayer(a);
  }
  return g;
}

core::engine::AttentionKernelInfo DescribeAttentionKernels() {
  core::engine::AttentionKernelInfo k;
  k.decode = {core::engine::AttentionKernelOrigin::kCore, "core blocked-Q8 row-chunk split-K (multi-wave) on sliding-window d256 layers", "",
              "full d512 layers and F16 KV run gemma4's AttentionRowChunkF16Kernel (private, row invariant)", true};
  k.prefill = {core::engine::AttentionKernelOrigin::kCore, "core flash_wmma_large (d256/d512, F16 + blocked-Q8) and 16-head blocked WMMA", "",
               "n <= 8 and head counts the WMMA kernels do not serve run the scalar read", false};
  return k;
}

core::engine::AttentionGeometry GemmaEngineModel::AttentionGeometryInfo() const { return DescribeAttentionGeometry(*model_); }
core::engine::AttentionKernelInfo GemmaEngineModel::AttentionKernelsInfo() const { return DescribeAttentionKernels(); }

std::string GemmaEngineModel::ModelName() const {
  return model_->ModelName();
}

std::string GemmaEngineModel::Decode(
    std::span<const std::int32_t> tokens) const {
  return model_->Decode(tokens);
}

bool GemmaEngineModel::IsStopToken(std::int32_t token) const noexcept {
  return model_->IsStopToken(token);
}

std::int32_t GemmaEngineModel::EosToken() const noexcept {
  return model_->EosToken();
}

namespace {

// One-line "gemma4" registration: maps the generic load request onto
// gemma4::ModelOptions (zero/empty fields keep engine defaults, same
// convention as the llama/glimmer pilots: f16 default, "q8" selects the
// shared blocked INT8 KV path on SWA d==256 layers with full d==512 layers
// staying F16). decode_concurrency keeps the native default (the request
// carries no such field).
const core::engine::EngineRegistrar kRegisterGemma4Engine(
    core::engine::EngineBackendInfo{
        "gemma4", "Gemma4 26B A4B MoE text (f16/q8 KV)", true},
    [](const core::engine::EngineLoadRequest& request,
       std::string* error_msg) -> std::shared_ptr<core::engine::EngineModel> {
      ModelOptions options;
      if (request.max_context != 0) {
        options.max_context = request.max_context;
      }
      if (!request.kv_cache_quant.empty()) {
        options.kv_cache_quant = request.kv_cache_quant;
      }
      return GemmaEngineModel::Load(request.model_path, options, error_msg);
    });

}  // namespace

}  // namespace mbufo::models::gemma4
