// DeepSeek V4 Flash adapter: thin forwarding over deepseek_v4_flash::Model/Session plus the "deepseek_v4_flash"
// EngineRegistry registration. See engine_adapter.hpp for the mapping of the L0 surface onto the native API.

#include "src/models/deepseek_v4_flash/engine_adapter.hpp"

#include "src/core/gguf_reader.hpp"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <utility>

#include "src/core/session_mode.hpp"

namespace mbufo::models::deepseek_v4_flash {
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
  auto engine = std::shared_ptr<DeepSeekEngineModel>(
      new DeepSeekEngineModel(std::move(inner), ec ? 0 : static_cast<std::size_t>(bytes)));
  // The ds4 runtime hides its config; the contract declaration is read from the same GGUF it loaded.
  if (auto reader = core::GgufReader::OpenFile(model_path)) {
    engine->moe_geometry_ = DescribeMoeGeometry(*reader);
    engine->attention_geometry_ = DescribeAttentionGeometry(*reader);
  }
  return engine;
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
core::engine::MoeGeometry DeepSeekEngineModel::MoeGeometryInfo() const { return moe_geometry_; }
core::engine::MoeKernelInfo DeepSeekEngineModel::MoeKernelsInfo() const { return DescribeMoeKernels(); }

core::engine::MoeGeometry DescribeMoeGeometry(const core::GgufReader& reader) {
  core::engine::MoeGeometry g;
  auto u32 = [&](const char* key) { return reader.GetMetadataUint32(key).value_or(0); };
  const std::uint32_t layers = u32("deepseek4.block_count");
  const std::uint32_t experts = u32("deepseek4.expert_count");
  if (experts == 0) return g;
  for (std::uint32_t i = 0; i < layers; ++i) {
    const std::string base = "blk." + std::to_string(i) + ".";
    const auto* gate = reader.FindTensor(base + "ffn_gate_exps.weight");
    const auto* up = reader.FindTensor(base + "ffn_up_exps.weight");
    const auto* down = reader.FindTensor(base + "ffn_down_exps.weight");
    if (gate == nullptr || up == nullptr || down == nullptr) continue;
    g.AddLayer(static_cast<int>(gate->type), static_cast<int>(up->type), static_cast<int>(down->type));
  }
  g.experts = experts;
  g.top_k = u32("deepseek4.expert_used_count");
  g.hidden = u32("deepseek4.embedding_length");
  g.expert_ff = u32("deepseek4.expert_feed_forward_length");
  g.shared_ff = g.expert_ff * u32("deepseek4.expert_shared_count");
  return g;
}

core::engine::MoeKernelInfo DescribeMoeKernels() {
  core::engine::MoeKernelInfo k;
  k.origin = core::engine::MoeKernelOrigin::kModelPrivate;
  k.gate_up_kernel = "ds4 moe_gate_up_mid_decode_lut_qwarp32 (IQ2_XXS gate/up x Q8_K int dot, LUT)";
  k.down_kernel = "ds4 moe_down_sum6_qwarp32 (Q2_K down x Q8_K int dot, fused 6-expert sum)";
  k.private_reason =
      "models/deepseek_v4_flash/UPSTREAM.md: ds4 numerics stay bit-identical to antirez/ds4 (IQ2_XXS/Q2_K LUT kernels on Q8_K activations, "
      "sorted-tile and hotlist WMMA batch paths), so its kernels are not shared with the Qwen-derived core";
  k.masks_inactive_slots = true;
  k.graph_capture_safe = true;
  return k;
}

std::size_t DeepSeekEngineModel::ResidentBytes() const noexcept { return resident_bytes_; }

core::engine::AttentionGeometry DescribeAttentionGeometry(const core::GgufReader& reader) {
  auto u32 = [&](const char* key) { return reader.GetMetadataUint32(key).value_or(0); };
  core::engine::AttentionGeometry g;
  core::engine::AttentionGroup a;
  a.kind = core::engine::AttentionKind::kMla;
  a.heads = u32("deepseek4.attention.head_count");
  a.kv_heads = u32("deepseek4.attention.head_count_kv");
  a.head_dim = u32("deepseek4.attention.key_length");
  a.kv_format = core::engine::AttentionKvFormat::kModelSpecific;
  for (std::uint32_t i = 0; i < u32("deepseek4.block_count"); ++i) g.AddLayer(a);
  return g;
}

core::engine::AttentionKernelInfo DescribeAttentionKernels() {
  core::engine::AttentionKernelInfo k;
  k.decode = {core::engine::AttentionKernelOrigin::kModelPrivate, "ds4 MLA decode attention (compressed KV + indexer)",
              "multi-head latent attention with a compressed KV cache; the core softmax-GQA kernels cannot serve it", "", false};
  k.prefill = {core::engine::AttentionKernelOrigin::kModelPrivate, "ds4 MLA prefill attention",
               "multi-head latent attention with a compressed KV cache; the core softmax-GQA kernels cannot serve it", "", false};
  return k;
}

core::engine::AttentionGeometry DeepSeekEngineModel::AttentionGeometryInfo() const { return attention_geometry_; }
core::engine::AttentionKernelInfo DeepSeekEngineModel::AttentionKernelsInfo() const { return DescribeAttentionKernels(); }

std::string DeepSeekEngineModel::ModelName() const { return model_->ModelName(); }
std::string DeepSeekEngineModel::DecodeToken(std::int32_t token) const { return model_->DecodeToken(token); }
bool DeepSeekEngineModel::IsStopToken(std::int32_t token) const { return model_->IsStopToken(token); }

namespace {

// "deepseek_v4_flash" registration: max_context 0 keeps the native default; the request's kv_cache_quant does not
// apply (DeepSeek's compressed KV is part of the model) and is ignored; no DSpark support model can come from a
// registry request.
const core::engine::EngineRegistrar kRegisterDeepSeekEngine(
    core::engine::EngineBackendInfo{"deepseek_v4_flash", "DeepSeek V4 Flash MoE text (private kernels)", true},
    [](const core::engine::EngineLoadRequest& request,
       std::string* error_msg) -> std::shared_ptr<core::engine::EngineModel> {
      ModelOptions options;
      if (request.max_context != 0) options.max_context = request.max_context;
      return DeepSeekEngineModel::Load(request.model_path, options, error_msg);
    });

}  // namespace

}  // namespace mbufo::models::deepseek_v4_flash