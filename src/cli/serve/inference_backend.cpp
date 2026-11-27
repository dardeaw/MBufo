#include "src/cli/serve/inference_backend.hpp"

#include <unistd.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#include "src/cli/serve/component_text_cache.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/npu_warmup.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_identity.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/core/pooling_norm.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/vision/image_cache.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/heap_placement.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/host_allocations.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/deepseek_v4_flash/dspark_sampler.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/gemma4/engine.hpp"
#include "src/models/glimmer/draft_backend.hpp"
#include "src/models/glimmer/engine.hpp"
#include "src/models/gptoss/engine.hpp"
#include "src/models/llama/engine.hpp"
#include "src/models/laguna/engine.hpp"
#include "src/models/inkling/engine.hpp"
#include "src/models/qwen/continuation_adapter.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen38_flash_next/continuation_adapter.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen/engine_adapter.hpp"
#include "src/models/qwen38_flash_next/engine_adapter.hpp"
#include "src/models/gptoss/engine_adapter.hpp"
#include "src/models/gemma4/engine_adapter.hpp"
#include "src/models/deepseek_v4_flash/engine_adapter.hpp"
#include "src/core/engine/moe_contract.hpp"
#include "src/models/bge_m3/engine.hpp"
#include "src/models/embeddinggemma2/engine.hpp"
#include "src/models/bge_reranker/engine.hpp"
#include "src/models/qwen_reranker/engine.hpp"
#include "src/cli/serve/encoder_runner.hpp"
#include "src/core/compute/npu/npu_scheduler.hpp"
#include "src/core/compute/npu/npu_lane.hpp"
#ifdef MBUFO_ENABLE_NPU
#include "src/core/compute/npu/npu_graph_models.hpp"
#endif
#endif

// Process-wide NPU scheduler shared by every dual-capable encoder model.
// Lazy singleton: created on first NPU-targeted load, never at boot.
namespace mbufo::server {
namespace {
#ifdef MBUFO_ENABLE_NPU
mbufo::npu::NpuLaneConfig NpuLaneConfigForDevice(const std::string& device) {
  mbufo::npu::NpuLaneConfig cfg;
  if (device == "npu") {
    cfg.embedding = true;
    cfg.reranker = true;
    cfg.decide = true;
    cfg.asr = true;
    cfg.tts = true;
  }
  return cfg;
}
mbufo::core::npu::NpuScheduler& GlobalNpuScheduler() {
  static mbufo::core::npu::NpuScheduler scheduler(
      NpuLaneConfigForDevice("npu"));
  return scheduler;
}
#else
mbufo::npu::NpuLaneConfig NpuLaneConfigForDevice(const std::string&) {
  return mbufo::npu::NpuLaneConfig{};
}
#endif
}  // namespace

// A selected token ends the turn only when the request admits EOS and the token
// is one of the model's stop tokens. The engine stops there itself and the stop
// is reported by the engine, so this stays per-request state.
template<typename State, typename Tokens>
bool EndsTurn(const State& state, const Tokens& tokens, int token) {
  return state.stop_at_eos() && tokens.IsStopToken(token);
}

namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

#if defined(ENGINE_ENABLE_HIP)

struct QwenImageContext final : TextPromptContext {
  std::shared_ptr<const models::qwen::vision::Prompt> prompt;
};

tokenization::ChatTemplateOptions QwenChatOptions(const ChatRequest& request,
                                                  std::uint32_t max_context) {
  auto options = tokenization::ResolveQwenChatOptions(request.reasoning,
                                                      request.add_vision_id);
  // mbufo #285: the rendered prompt may be as long as the context can hold.
  options.max_output_bytes =
      tokenization::RenderedPromptBoundBytes(max_context);
  return options;
}

std::optional<std::size_t> CountQwenPromptTokens(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache) {
  const auto options = QwenChatOptions(request, max_context);
  const auto tools =
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools};
  if (std::ranges::none_of(request.messages,
                           [](const auto& m) { return !m.images.empty(); })) {
    const auto tokens = tokenization::QwenChatTemplate::RenderAndTokenize(
        tokenizer, request.messages, tools, options);
    if (!tokens)
      return std::nullopt;
    return tokens->size();
  }
  // Reuse generation's CPU image preparation, including resize, token
  // expansion and projector validation. Counting does not encode images on
  // the GPU or admit the prompt to an execution context. Keep only the shared
  // layout's signed position limit so oversized prompts can still be sized.
  return models::qwen::vision::Prepare(
             tokenizer, request.messages, tools, options,
             encoder ? encoder->identity() : std::string_view{},
             std::numeric_limits<std::int32_t>::max(),
             request.cache_prompt ? &image_cache : nullptr)
      .tokens.size();
}

TextPreparedPrompt PrepareQwenPrompt(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache) {
  const bool has_images = std::ranges::any_of(
      request.messages, [](const auto& m) { return !m.images.empty(); });
  const auto options = QwenChatOptions(request, max_context);
  auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, request.messages,
          request.tool_choice == ChatRequest::ToolChoice::kNone
              ? std::span<const tokenization::ChatTool>{}
              : std::span<const tokenization::ChatTool>{request.tools},
          options,
          encoder && has_images ? encoder->identity() : std::string_view{},
          max_context, request.cache_prompt ? &image_cache : nullptr));
  if (has_images && Logger::Enabled(LogLevel::kDebug)) {
    const auto stats = image_cache.GetStats();
    Logger::LogFormatted(LogLevel::kDebug, "vision",
                         "event=image_cache enabled=%d entries=%zu "
                         "pixel_bytes=%zu encoded_bytes=%zu "
                         "hits=%zu misses=%zu evictions=%zu",
                         static_cast<int>(request.cache_prompt), stats.entries,
                         stats.pixel_bytes, stats.encoded_bytes, stats.hits,
                         stats.misses, stats.evictions);
  }
  const auto cache_prefix = prompt->stable_prefix_tokens;
  if (prompt->images.empty())
    return {std::move(prompt->tokens), {}, cache_prefix};
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  auto context = std::make_shared<QwenImageContext>();
  context->cache_identity = prompt->cache_identity;
  for (const auto& image : prompt->images) {
    const auto identity = prompt->IdentityForPrefix(image.grid.offset);
    context->cache_prefixes.push_back(
        {image.grid.offset, {identity.begin(), identity.end()}});
  }
  context->prompt = prompt;
  return {prompt->tokens, std::move(context), cache_prefix};
}

std::shared_ptr<const models::qwen::vision::Prompt> QwenPrompt(
    const std::shared_ptr<const TextPromptContext>& context) {
  if (!context)
    return {};
  const auto* image = dynamic_cast<const QwenImageContext*>(context.get());
  if (!image)
    throw std::invalid_argument("invalid Qwen prompt context");
  return image->prompt;
}

// Hash the sidecar and upload its weights while the model loads, before the
// runner claims state capacity and the RAM cache is sized. Otherwise the first
// image request hashes the whole sidecar in its queue time and uploads the
// weights in its prefill, after admission and the cache were sized without
// them.
void PrepareVision(
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder) {
  if (!encoder)
    return;
  const auto started = Clock::now();
  (void)encoder->identity();
  const auto hashed = Clock::now();
  try {
    encoder->Upload();
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("could not upload the Qwen vision sidecar: ") + e.what());
  }
  const auto uploaded = Clock::now();
  std::size_t heap_free = 0, heap_total = 0;
  (void)hipMemGetInfo(&heap_free, &heap_total);
  using Milliseconds = std::chrono::duration<double, std::milli>;
  Logger::Info("loader",
               "event=vision_prepared resident_bytes=" +
                   std::to_string(encoder->ResidentBytes()) + " identity_ms=" +
                   std::to_string(Milliseconds(hashed - started).count()) +
                   " upload_ms=" +
                   std::to_string(Milliseconds(uploaded - hashed).count()) +
                   " device_heap_free_bytes=" + std::to_string(heap_free) +
                   " " + Logger::MemoryStatus());
}

constexpr std::string_view kDeepSeekStateAbi =
    "deepseek-v4-flash-gfx1151-state-v4";
constexpr std::uint32_t kQwenComponentPayloadVersion = 2;
constexpr std::size_t kQwenMaxResumeTokens = 9;

constexpr std::string_view QwenStateAbi(bool speculative,
                                        bool fp16_attention_kv,
                                        bool bf16_recurrent_state) noexcept {
  if (bf16_recurrent_state) {
    if (speculative) {
      return fp16_attention_kv
                 ? "qwen-gfx1151-dflash-state-v4-fp16-kv-bf16-recurrent"
                 : "qwen-gfx1151-dflash-state-v4-fp32-kv-bf16-recurrent";
    }
    return fp16_attention_kv ? "qwen-gfx1151-state-v3-fp16-kv-bf16-recurrent"
                             : "qwen-gfx1151-state-v3-fp32-kv-bf16-recurrent";
  }
  if (speculative) {
    return fp16_attention_kv ? "qwen-gfx1151-dflash-state-v3-fp16-kv"
                             : "qwen-gfx1151-dflash-state-v3-fp32-kv";
  }
  return fp16_attention_kv ? "qwen-gfx1151-state-v2-fp16-kv"
                           : "qwen-gfx1151-state-v2-fp32-kv";
}

bool DiskCacheEnabled(const TextDiskCacheConfig& config) noexcept {
  return !config.directory.empty();
}

bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::vector<std::uint8_t> DeepSeekCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view support_fingerprint,
    std::uint32_t max_context, std::uint32_t max_draft_tokens) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "DeepSeek disk cache requires an artifact fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=mbufo-text-continuation-v2\n"
           << "model_kind=deepseek4\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=joyai-byte-bpe-v1\n"
           << "chat_template=" << models::deepseek_v4_flash::ChatTemplateId()
           << '\n'
           << "chat_template_reference_sha256="
           << models::deepseek_v4_flash::EncoderReferenceSha256() << '\n'
           << "state_abi=" << kDeepSeekStateAbi << '\n'
           << "payload_layout=ds4-rocm-v4-window128-continuation\n"
           // Cached frontiers also depend on the compiled numerical routes.
           << "numerics=ds4-scalar-verifier-v1\n"
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "rope_window_policy=deepseek4-compiled-v1\n"
           << "adapters=none\n";
  identity << "support_id=" << core::kGgufIdentityScheme << ':'
           << support_fingerprint << '\n'
           << "max_draft_tokens=" << max_draft_tokens << '\n'
           << "draft_policy=dspark-cost-v5-window128\n";
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> QwenFlashNextCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view mtp_fingerprint,
    bool has_mtp, std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency, std::string_view kv_cache_quant,
    std::string_view position_policy) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.8-Flash-Next disk cache requires an artifact fingerprint");
  }
  if (has_mtp && !IsSha256Hex(mtp_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.8-Flash-Next MTP disk cache requires a draft artifact "
        "fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=mbufo-text-continuation-v2\n"
           << "model_kind=qwen38-flash-next\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=qwen38-reasoning-compiled-v3\n"
           << "chat_template_reference_sha256="
           << tokenization::QwenChatTemplate::OfficialTemplateSha256() << '\n'
           << "state_abi=qwen38-flash-next-rocm-session-v1\n"
           << "payload_layout=qfn-rocm-session-snapshot-v"
           << models::qwen38_flash_next::Session::kSnapshotPayloadVersion
           << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=" << position_policy << '\n'
           << "kv_quant=" << kv_cache_quant << "\n"
           << "adapters=none\n";
  if (has_mtp) {
    identity << "draft_backend=qfn-mtp-v1\n"
             << "draft_artifact_id=" << core::kGgufIdentityScheme << ':'
             << mtp_fingerprint << '\n'
             << "draft_max_tokens=" << max_draft_tokens << '\n'
             << "draft_cost_concurrency=" << decode_concurrency << '\n';
  }
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> QwenCompatibilityIdentity(
    std::string_view artifact_fingerprint,
    std::string_view draft_artifact_fingerprint, std::uint32_t max_context,
    const hip::QwenExecutionPolicy& execution_policy, bool speculative,
    const speculative::SpeculativeOptions& speculative_options) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen disk cache requires an artifact fingerprint");
  }
  if (speculative && !IsSha256Hex(draft_artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen DFlash disk cache requires a draft artifact fingerprint");
  }
  std::string state_abi =
      std::string(QwenStateAbi(speculative, execution_policy.UsesFp16AttentionKv(),
                               execution_policy.UsesBf16RecurrentState()));
  if (execution_policy.UsesBlockedAttentionKv()) {
    // Blocked-Q8 decode reads quantized sidecars: same weights, different
    // numerics and a different state layout. Never share entries with f16.
    state_abi += "-blocked-q8";
  }
  std::ostringstream identity;
  identity << "schema=mbufo-text-continuation-v2\n"
           << "model_kind=qwen3.8\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=qwen38-reasoning-compiled-v3\n"
           << "chat_template_reference_sha256="
           << tokenization::QwenChatTemplate::OfficialTemplateSha256() << '\n'
           << "state_abi=" << state_abi << "-component-v2\n"
           << "payload_layout=qwen-component-v2\n"
           // Older learned boundaries could keep logits from an earlier
           // prefix. Their payload has no readiness tag to distinguish them.
           << "checkpoint_frontier=complete-prefix-v1\n"
           << "numerics=qwen-bf16-fp32-prefill-v1\n"
           << "rmsnorm=fused-square-sum-v1\n"
           << "prefill_attention=visible-causal-tail-v1\n"
           << "attention_split_min_context="
           << hip::detail::kSplitKDecodeAttentionMinContext << '\n'
           << "attention_split_count="
           << hip::detail::kSplitKDecodeAttentionMaxSplits << '\n'
           << "kv_storage="
           << (execution_policy.UsesFp16AttentionKv() ? "fp16" : "fp32") << '\n'
           << "kv_blocked_q8="
           << (execution_policy.UsesBlockedAttentionKv() ? "true" : "false")
           << '\n'
           << "recurrent_storage="
           << (execution_policy.UsesBf16RecurrentState() ? "bf16" : "fp32")
           << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "rope_window_policy=qwen-gguf-config-v1\n"
           << "adapters=none\n";
  if (speculative) {
    identity
        << "draft_backend=dflash2-gfx1151-v1\n"
        << "draft_artifact_id=" << core::kGgufIdentityScheme << ':'
        << draft_artifact_fingerprint << '\n'
        << "draft_state_layout=dflash-window-kv-and-frontier-v4\n"
        << "draft_policy="
        << speculative::DFlashDraftPolicyName(speculative_options.dflash_policy)
        << "-v3\n"
        << "draft_max_tokens=" << speculative_options.max_draft_tokens << '\n'
        << "draft_min_tokens=" << speculative_options.min_draft_tokens << '\n'
        << "draft_initial_tokens=" << speculative_options.initial_draft_tokens
        << '\n'
        << "draft_rolling_window=" << speculative_options.rolling_window << '\n'
        << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "draft_target_acceptance="
        << speculative_options.target_acceptance_rate << '\n'
        << "draft_adaptive="
        << (speculative_options.enable_adaptive_draft_length ? "true" : "false")
        << '\n'
        << "draft_batched_verification="
        << (speculative_options.use_batched_verification ? "true" : "false")
        << '\n'
        << "draft_target_verification=decode-equivalent-v1\n";
  }
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

class QwenTextRunnerState final : public TextRunnerState {
public:
  QwenTextRunnerState(std::shared_ptr<const hip::QwenGpuModel> model,
                      models::qwen::ContinuationAdapter& adapter,
                      cache::Slot& slot,
                      speculative::SpeculativeOptions options)
      : model_(std::move(model)),
        executor_(&adapter.GetExecutor(slot)),
        draft_backend_(adapter.GetDraftBackend(slot)),
        adapter_(&adapter),
        slot_(&slot),
        frontier_(adapter.GetFrontier(slot).token),
        frontier_logits_(adapter.GetFrontier(slot).logits),
        frontier_published_(adapter.GetFrontier(slot).published),
        resume_tokens_(adapter.GetFrontier(slot).pending) {
    resume_tokens_.reserve(kQwenMaxResumeTokens);
    rollback_tokens_.reserve(kQwenMaxResumeTokens);
    if (draft_backend_) {
      verifier_ = std::make_unique<speculative::SpeculativeVerifier>(
          *executor_, *draft_backend_, options);
      executor_->SetPromptHiddenCapture(true,
                                        draft_backend_->TargetHiddenLayerIds());
    }
  }

  void SetCancellationCheck(const CancellationCheck& check) override {
    executor_->SetCancellationCheck(check);
  }
  ~QwenTextRunnerState() override { executor_->SetCancellationCheck({}); }

  class PrefillScope {
  public:
    explicit PrefillScope(QwenTextRunnerState& state)
        : state_(state), pending_size_(state.resume_tokens_.size()) {
      std::ranges::copy(state.resume_tokens_, pending_.begin());
      state_.adapter_->BeginPrefill(*state_.slot_);
    }
    ~PrefillScope() { state_.adapter_->FinishPrefill(*state_.slot_); }
    void Rollback() {
      state_.adapter_->RollbackPrefill(*state_.slot_);
      state_.resume_tokens_.assign(pending_.begin(),
                                   pending_.begin() + pending_size_);
    }
    PrefillScope(const PrefillScope&) = delete;
    PrefillScope& operator=(const PrefillScope&) = delete;

  private:
    QwenTextRunnerState& state_;
    std::array<TextRunnerToken, kQwenMaxResumeTokens> pending_{};
    std::size_t pending_size_;
  };
  [[nodiscard]] PrefillScope BeginPrefill() { return PrefillScope(*this); }

  void Invalidate() noexcept override {
    (void)adapter_->Invalidate(*slot_);
    sequence_.clear();
    position_ = 0;
    frontier_.reset();
    frontier_logits_.clear();
    frontier_published_ = false;
    resume_tokens_.clear();
    rollback_position_.reset();
    rollback_tokens_.clear();
    rollback_logits_.clear();
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    try {
      auto usage = executor_->GetMemoryUsage();
      if (draft_backend_ != nullptr) {
        const auto draft = draft_backend_->GetMemoryUsage();
        usage.request_state_bytes += draft.request_state_bytes;
        usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      }
      return {
          .per_request_state_bytes = usage.request_state_bytes,
          .temporary_scratch_bytes = usage.temporary_scratch_bytes,
      };
    } catch (...) {
      return {};
    }
  }

  [[nodiscard]] bool HasDecodeFrontier() const {
    return adapter_->HasDecodeFrontier(*slot_);
  }

  [[nodiscard]] bool speculative() const noexcept {
    return verifier_ != nullptr;
  }

  void PrimeSpeculative(std::span<const TextRunnerToken> prompt,
                        bool decode_ready) {
    if (verifier_ == nullptr) {
      throw std::logic_error("Qwen state has no speculative verifier");
    }
    frontier_ = verifier_->Prime(prompt, decode_ready);
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    sequence_.assign(prompt.begin(), prompt.end());
    position_ = prompt.size();
    frontier_published_ = false;
    resume_tokens_.clear();
    rollback_position_.reset();
  }

  void PreparePrefixReuse(std::span<const TextRunnerToken> prefix) {
    if (position_ != prefix.size() || !frontier_.has_value()) {
      throw std::logic_error(
          "Qwen retained prefix does not match its target state");
    }
    if (verifier_ != nullptr) {
      verifier_->BeginRequest();
      sequence_.assign(prefix.begin(), prefix.end());
      // Verification's frontier may be a sampled draw. A new request must
      // select from its distribution again, including when it switches to
      // greedy sampling or omits the previously published pending token.
      if (!resume_tokens_.empty() && !frontier_logits_.empty()) {
        sampling::SamplerState greedy;
        frontier_ = greedy.Sample(frontier_logits_);
      }
    }
    frontier_published_ = false;
    rollback_position_.reset();
  }

  bool ResumeGeneratedToken(TextRunnerToken token, bool decode_ready) {
    if (resume_tokens_.empty())
      return false;
    if (resume_tokens_.front() != token) {
      resume_tokens_.clear();
      return false;
    }
    resume_tokens_.erase(resume_tokens_.begin());
    // This token was already published by the previous request. Consume it
    // with decode arithmetic, exactly as AR did, before chunking the new
    // prompt suffix. Including it in a prefill chunk changes recurrent state.
    const auto position = static_cast<std::uint32_t>(position_);
    frontier_ = verifier_ != nullptr
                    ? verifier_->AdvanceCommittedToken(token, position)
                    : executor_->ForwardToken(token, position);
    if (verifier_ != nullptr)
      sequence_.push_back(token);
    ++position_;
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = executor_->CopyLastLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    return true;
  }

  void ExtendSpeculative(std::span<const TextRunnerToken> suffix,
                         bool decode_ready) {
    if (verifier_ == nullptr || !frontier_.has_value()) {
      throw std::logic_error("Qwen speculative prefix is not initialized");
    }
    if (sequence_.size() != position_) {
      throw std::logic_error(
          "Qwen speculative sequence does not match retained position");
    }
    frontier_ = verifier_->ExtendPrompt(
        suffix, static_cast<std::uint32_t>(position_), decode_ready);
    sequence_.insert(sequence_.end(), suffix.begin(), suffix.end());
    position_ += suffix.size();
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    frontier_published_ = false;
  }

  [[nodiscard]] TextRunnerToken SelectFrontier(
      sampling::SamplerState& sampler) const {
    if (!frontier_.has_value()) {
      throw std::logic_error("Qwen state has no next-token frontier");
    }
    if (sampler.config().can_use_unmodified_argmax() ||
        (verifier_ == nullptr && sampler.CanSelectArgmax(*frontier_))) {
      return *frontier_;
    }
    if (frontier_logits_.empty()) {
      return executor_->SampleLastLogits(sampler);
    }
    return executor_->SampleCachedLogits(frontier_logits_, sampler);
  }

  [[nodiscard]] bool PrepareSpeculativeDecode(std::size_t max_tokens,
                                              sampling::SamplerState& sampler,
                                              TextDecodeStep& result) {
    if (verifier_ == nullptr || !frontier_.has_value()) {
      throw std::logic_error("Qwen speculative state has no frontier");
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen speculative decode budget must be at least one token");
    }
    resume_tokens_.clear();
    if (!frontier_published_) {
      if (!sampler.config().can_use_unmodified_argmax())
        frontier_ = SelectFrontier(sampler);
      if (!AppendSpeculativeSelection(*frontier_, sampler, result))
        return false;
      frontier_published_ = true;
    }
    return result.selections.size() < max_tokens;
  }

  [[nodiscard]] speculative::SpeculativeVerifier::StepRequest
  VerificationRequest(std::size_t remaining, sampling::SamplerState& sampler) {
    rollback_position_ = position_;
    rollback_tokens_.assign(1, *frontier_);
    rollback_logits_ = std::move(frontier_logits_);
    // Verification keeps its own EOS handling: the tokenizer stop set already
    // covers EOS, so a sentinel id here would change nothing. An ignore-EOS
    // request instead publishes the stop token in AppendSpeculativeSelection
    // and continues, at the cost of a window clipped on every EOS crossing.
    return {*verifier_,
            sequence_,
            static_cast<std::uint32_t>(position_),
            *frontier_,
            model_->GetTokenizer().GetEosTokenId(),
            static_cast<std::uint32_t>(std::min<std::size_t>(
                remaining, std::numeric_limits<std::uint32_t>::max())),
            sampler};
  }

  void FinishSpeculativeDecode(
      speculative::SpeculativeVerifier::StepResult verification,
      sampling::SamplerState& sampler, TextDecodeStep& result) {
    result.draft_rounds = verification.draft_count > 0 ? 1 : 0;
    result.draft_tokens = verification.draft_count;
    result.draft_accepted_tokens = verification.accepted_count;
    result.lookup_tokens = verification.lookup_tokens;
    result.lookup_accepted_tokens = verification.lookup_accepted_tokens;
    result.execution_plan = {
        .kind = verification.physical_width > 1
                    ? TextExecutionPlanKind::kBatched
                    : TextExecutionPlanKind::kSerial,
        .physical_width = verification.physical_width,
    };
    for (const TextRunnerToken token : verification.emitted_tokens) {
      // Each prediction consumes one input, including a terminal prediction.
      // EOS itself remains the unconsumed frontier and is not published.
      ++position_;
      if (!AppendSpeculativeSelection(token, sampler, result))
        break;
      rollback_tokens_.push_back(token);
    }
    frontier_ = verification.next_token;
    frontier_logits_ = std::move(verification.next_token_logits);
    frontier_published_ = !result.stop;
    set_pending_token(frontier_published_ ? frontier_ : std::nullopt);
    if (verification.draft_count == 0)
      rollback_position_.reset();
  }

  void PrepareCancellation() {
    if (!rollback_position_)
      return;
    if (rollback_tokens_.empty() ||
        rollback_tokens_.size() > kQwenMaxResumeTokens ||
        rollback_logits_.empty())
      throw std::logic_error(
          "Qwen cancelled verification has no exact frontier");
    // Verification already saved the target state. Its target features are
    // still pending in the draft, so rewinding needs no new per-step copy.
    draft_backend_->DiscardPendingTargetContext(
        static_cast<std::uint32_t>(*rollback_position_));
    executor_->RestoreState();
    executor_->FinishVerification();
    position_ = *rollback_position_;
    sequence_.resize(position_);
    frontier_logits_ = std::move(rollback_logits_);
    sampling::SamplerState greedy;
    frontier_ = greedy.Sample(frontier_logits_);
    frontier_published_ = false;
    resume_tokens_ = std::move(rollback_tokens_);
    rollback_position_.reset();
  }

  [[nodiscard]] TextDecodeStep DecodeSpeculative(
      std::size_t max_tokens, sampling::SamplerState& sampler) {
    TextDecodeStep result;
    sampling::SamplerState working_sampler = sampler;
    if (PrepareSpeculativeDecode(max_tokens, working_sampler, result)) {
      const auto request = VerificationRequest(
          max_tokens - result.selections.size(), working_sampler);
      auto verification = verifier_->VerifyStep(
          sequence_, request.position, request.current_token, request.eos_id,
          request.max_emitted_tokens, working_sampler);
      FinishSpeculativeDecode(std::move(verification), working_sampler, result);
    }
    sampler.SetRngState(working_sampler.rng_state());
    return result;
  }

  [[nodiscard]] hip::QwenGpuExecutor& executor() const { return *executor_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }
  [[nodiscard]] const std::optional<TextRunnerToken>& frontier()
      const noexcept {
    return frontier_;
  }
  void set_frontier(TextRunnerToken frontier) noexcept {
    frontier_ = frontier;
    frontier_logits_.clear();
    frontier_published_ = false;
    resume_tokens_.clear();
  }
  [[nodiscard]] std::vector<float> CopyFrontierLogits() const {
    if (!frontier_logits_.empty()) {
      return frontier_logits_;
    }
    const auto logits = executor_->CopyLastLogits();
    return {logits.begin(), logits.end()};
  }
  void RestoreFrontierLogits(std::vector<float> logits) {
    frontier_logits_ = std::move(logits);
  }
  [[nodiscard]] const std::vector<TextRunnerToken>& pending_tokens()
      const noexcept {
    return resume_tokens_;
  }
  void set_pending_token(std::optional<TextRunnerToken> token) {
    resume_tokens_.clear();
    if (token)
      resume_tokens_.push_back(*token);
  }
  void RestorePendingTokens(std::span<const TextRunnerToken> tokens) {
    resume_tokens_.assign(tokens.begin(), tokens.end());
    rollback_position_.reset();
  }

private:
  bool AppendSpeculativeSelection(TextRunnerToken token,
                                  sampling::SamplerState& sampler,
                                  TextDecodeStep& result) {
    if (stop_at_eos() && model_->GetTokenizer().IsStopToken(token)) {
      result.stop = true;
      return false;
    }
    result.selections.push_back({
        .stop = false,
        .token = token,
        .piece = std::string(model_->GetTokenizer().DecodeToken(token)),
    });
    sequence_.push_back(token);
    set_pending_token(token);
    sampler.Accept(token);
    return true;
  }

  std::shared_ptr<const hip::QwenGpuModel> model_;
  hip::QwenGpuExecutor* executor_{};
  std::unique_ptr<speculative::SpeculativeVerifier> verifier_;
  // Owned by the verifier for standalone execution, or by the component slot.
  hip::QwenDFlashGpuDraftBackend* draft_backend_{nullptr};
  models::qwen::ContinuationAdapter* adapter_{};
  cache::Slot* slot_{};
  std::vector<TextRunnerToken> sequence_;
  std::size_t position_{0};
  std::optional<TextRunnerToken>& frontier_;
  std::vector<float>& frontier_logits_;
  bool& frontier_published_;
  std::vector<TextRunnerToken>& resume_tokens_;
  std::optional<std::size_t> rollback_position_;
  std::vector<TextRunnerToken> rollback_tokens_;
  std::vector<float> rollback_logits_;
};

QwenTextRunnerState& RequireQwenState(TextRunnerState& state) {
  auto* qwen = dynamic_cast<QwenTextRunnerState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("text runner state is not Qwen");
  }
  return *qwen;
}

const QwenTextRunnerState& RequireQwenState(const TextRunnerState& state) {
  const auto* qwen = dynamic_cast<const QwenTextRunnerState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("text runner state is not Qwen");
  }
  return *qwen;
}

// Owns a four-byte device buffer and a private stream from load, so probing
// a lost context needs no allocation. HIP context loss is sticky: after a GPU
// reset even this memset fails with a hard error. Only such an error marks the
// device lost; a memset still pending at the deadline may be queued behind
// long kernels on the shared hardware queues, so it counts as usable.
class HipTextModelRunner : public TextModelRunner {
public:
  HipTextModelRunner() {
    HIP_CHECK(hipStreamCreateWithFlags(&probe_stream_, hipStreamNonBlocking));
    if (const auto error = hipMalloc(&probe_buffer_, sizeof(std::uint32_t));
        error != hipSuccess) {
      hip::LogCleanupError(hipStreamDestroy(probe_stream_));
      throw std::runtime_error(std::string("device probe allocation: ") +
                               hipGetErrorString(error));
    }
  }
  ~HipTextModelRunner() override {
    hip::LogCleanupError(hipFree(probe_buffer_));
    hip::LogCleanupError(hipStreamDestroy(probe_stream_));
  }
  HipTextModelRunner(const HipTextModelRunner&) = delete;
  HipTextModelRunner& operator=(const HipTextModelRunner&) = delete;
  HipTextModelRunner(HipTextModelRunner&&) = delete;
  HipTextModelRunner& operator=(HipTextModelRunner&&) = delete;

  [[nodiscard]] DeviceProbeStatus PollDevice() const override {
    if (!probe_pending_) {
      // Clear the failed work unit's thread-local error before probing.
      (void)hipGetLastError();
      if (hipMemsetAsync(probe_buffer_, 0, sizeof(std::uint32_t),
                         probe_stream_) != hipSuccess ||
          hipGetLastError() != hipSuccess)
        return DeviceProbeStatus::kLost;
      probe_pending_ = true;
    }
    const auto status = hipStreamQuery(probe_stream_);
    if (status == hipErrorNotReady)
      return DeviceProbeStatus::kPending;
    probe_pending_ = false;
    return status == hipSuccess ? DeviceProbeStatus::kUsable
                                : DeviceProbeStatus::kLost;
  }

  [[nodiscard]] bool DeviceUsable() const override {
    constexpr auto kTimeout = std::chrono::seconds(5);
    const auto deadline = Clock::now() + kTimeout;
    for (;;) {
      const auto status = PollDevice();
      if (status == DeviceProbeStatus::kUsable)
        return true;
      if (status == DeviceProbeStatus::kLost)
        return false;
      if (Clock::now() >= deadline) {
        Logger::Warn("scheduler", "event=device_probe_timeout timeout_s=" +
                                      std::to_string(kTimeout.count()));
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

private:
  hipStream_t probe_stream_{};
  void* probe_buffer_{};
  mutable bool probe_pending_{false};
};

template<typename AdapterType>
std::unique_ptr<ComponentCacheResources> CreateHipComponentCacheResources(
    std::unique_ptr<AdapterType> adapter, cache::ResourceLedger& ledger) {
  struct Backing {
    hip::TransferPool streams{64};
    std::unique_ptr<hip::CommittedBackingPool> pool;
    std::map<std::uint32_t, std::size_t> component_classes;
  };
  auto result = std::make_unique<ComponentCacheResources>();
  result->adapter = std::move(adapter);
  auto backing = std::make_shared<Backing>();
  result->transfer_batch_size = 16;
  const auto rounded = [](std::size_t bytes) {
    return (bytes + 4095) / 4096 * 4096;
  };
  for (const auto& d : result->adapter->Components()) {
    const auto bytes = rounded(d.kind == cache::ComponentKind::kPrivateState
                                   ? d.state_bytes
                                   : d.row_bytes * d.rows_per_chunk);
    backing->component_classes.emplace(d.id.value, bytes);
  }
  // The remainder belongs to index, histories, disk catalog and queue
  // metadata. The pools charge their own per-block allocator headroom.
  result->commit_backing = [backing, &ledger, adapter = result->adapter.get()](
                               std::size_t budget) {
    const auto plan = static_cast<AdapterType*>(adapter)->BackingPlan(budget);
    if (plan.size() != 1)
      throw std::invalid_argument("insufficient component backing budget");
    const auto& allocation = plan.front();
    // Pages commit as captures need them. A payload cannot span segments,
    // so segments of about 128 MiB bound registration pauses and stranded
    // edge space, fitting at least two of the largest payloads. Prefill
    // chunks prepare backing before GPU work, after any fallback capture.
    // Growth stops when the host keeps less than its reserve.
    std::size_t largest = 0;
    for (const auto& [id, bytes] : backing->component_classes)
      largest = std::max(largest, bytes);
    const auto block = allocation.block_bytes;
    const auto target =
        std::max<std::size_t>(std::size_t{128} << 20, 2 * largest) / block;
    const auto parts =
        std::max<std::size_t>((allocation.blocks + target - 1) / target, 1);
    const auto segment = (allocation.blocks + parts - 1) / parts * block;
    struct Limited {
      std::mutex mutex;
      std::chrono::steady_clock::time_point refused;
      bool active{};
    };
    backing->pool = std::make_unique<hip::CommittedBackingPool>(
        ledger,
        hip::BackingPoolConfig{
            allocation.blocks * allocation.block_bytes +
                allocation.metadata_bytes,
            allocation.block_bytes, allocation.metadata_bytes, segment,
            [limited = std::make_shared<Limited>()](std::size_t bytes) {
              const std::lock_guard lock(limited->mutex);
              const auto now = std::chrono::steady_clock::now();
              // Pressure retries ask once per capture; recheck at most
              // once a second while the host is short.
              if (limited->active &&
                  now - limited->refused < std::chrono::seconds(1))
                return false;
              const auto ceiling = HostSnapshotCeilingBytes();
              // Pinned host pages come out of the same device heap that
              // session and scratch allocations draw on; keep its dispatch
              // reserve so they do not fall back to managed memory.
              std::size_t heap_free = 0, heap_total = 0;
              const bool heap_fits =
                  hipMemGetInfo(&heap_free, &heap_total) == hipSuccess &&
                  hip::HeapFitsAboveReserve(heap_free, bytes);
              if (ceiling >= bytes && heap_fits) {
                limited->active = false;
                return true;
              }
              if (!limited->active)
                Logger::Warn(
                    "cache",
                    "schema=component-cache-v1 "
                    "event=backing_limited reason=" +
                        std::string(heap_fits ? "host_reserve "
                                              : "device_heap_reserve ") +
                        "segment_bytes=" + std::to_string(bytes) +
                        " host_headroom_bytes=" + std::to_string(ceiling) +
                        " device_heap_free_bytes=" + std::to_string(heap_free));
              limited->active = true;
              limited->refused = now;
              return false;
            }});
    // Prepare a bounded reserve before readiness. The first shared-prefix
    // capture needs more than one segment; preparing 2 GiB avoids moving
    // registration onto that request's capture or decode path.
    backing->pool->GrowAhead(2ULL << 30);
    backing->pool->WaitForGrowth();
    Logger::Info(
        "cache",
        "schema=component-cache-v1 event=backing_prepared committed_bytes=" +
            std::to_string(backing->pool->CommittedBytes()) +
            " long_term_pinned_bytes=" +
            std::to_string(backing->pool->LongTermPinnedBytes()));
  };
  result->prepare_prefill = [backing](std::size_t tokens) {
    // Prepare after any fallback capture, alongside actual prefill work.
    // Short turns need a smaller reserve for private states and row tails.
    backing->pool->GrowAhead((tokens >= 512 ? 2ULL : 1ULL) << 30);
  };
  result->can_capture = [backing](auto payloads) {
    return backing->pool->CanFit(payloads);
  };
  result->owner = backing;
  result->stream = [backing]() -> std::unique_ptr<cache::Stream> {
    return backing->streams.TryAcquire();
  };
  result->allocate = [backing](const cache::PayloadRequest& request) {
    // Bound the logical capture, then assign whole shared backing pages.
    const auto capacity =
        backing->component_classes.at(request.component.value);
    if (request.bytes > capacity)
      throw std::logic_error("component capture exceeds its backing class");
    const auto category =
        request.category == cache::ResourceCategory::kBackingAssigned
            ? cache::ResourceCategory::kBackingAssigned
            : request.category;
    const auto segments = backing->pool->PageCommitAllocations();
    const auto started = std::chrono::steady_clock::now();
    auto block = backing->pool->TryAcquire(category, request.bytes);
    if (backing->pool->PageCommitAllocations() != segments)
      Logger::Info(
          "cache",
          "schema=component-cache-v1 event=backing_grown committed_bytes=" +
              std::to_string(backing->pool->CommittedBytes()) + " segments=" +
              std::to_string(backing->pool->PageCommitAllocations()) +
              " long_term_pinned_bytes=" +
              std::to_string(backing->pool->LongTermPinnedBytes()) +
              " acquire_ms=" +
              std::to_string(std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - started)
                                 .count()));
    if (!block)
      throw cache::ResourceExhausted();
    // Do not start registration between a capture's allocations and copies.
    // Each prefill chunk prepares its own reserve before GPU work.
    auto owner = std::make_shared<hip::BackingBlock>(std::move(*block));
    return ComponentCaptureBuffer{
        owner->Bytes(),
        [owner] {
          owner->Convert();
          return std::move(*owner).TakePayload();
        },
        backing->pool.get(),
        [owner, request](cache::SlotLease& lease, auto source) {
          return cache::Payload::Borrowed(
              std::move(*owner).Borrow(lease, request.component, request.first,
                                       request.end, request.bytes, source));
        }};
  };
  result->staging = [](std::size_t bytes) {
    auto allocation = std::make_shared<hip::HostAllocations>();
    void* data = nullptr;
    const auto status =
        allocation->Allocate(&data, bytes, hipHostMallocCoherent);
    if (status != hipSuccess)
      throw std::runtime_error(hipGetErrorString(status));
    auto owner = std::shared_ptr<void>(std::move(allocation), data);
    std::memset(data, 0, bytes);
    return cache::StagingAllocation{std::move(owner),
                                    {static_cast<std::uint8_t*>(data), bytes}};
  };
  result->copy_observed_rows = [](auto source, auto destination,
                                  cache::Stream& stream) {
    return dynamic_cast<hip::TransferStream&>(stream).Copy(
        destination.data(), source.data(), destination.size(),
        hip::CopyDirection::kDeviceToHost, destination.size());
  };
  return result;
}

class QwenTextRunner final : public HipTextModelRunner {
public:
  // Batch contract (see TextModelRunner): native DecodeBatch/AdvanceBatch; native merged advance + prefill chunk
  // (non-DFlash); DecodeBatchWithPrefillChunk refuses.
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
  /// Shared weights for serve-side single-prefill features (embeddings).
  /// The weights stay owned by the runner; the backend caches one executor
  /// over them (see Impl) instead of rebuilding it per request.
  [[nodiscard]] std::shared_ptr<const hip::QwenGpuModel> QwenModel()
      const noexcept {
    return model_;
  }
  [[nodiscard]] hip::QwenExecutionPolicy ExecutionPolicy() const noexcept {
    return execution_policy_;
  }
  QwenTextRunner(
      std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
      std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model = nullptr,
      speculative::SpeculativeOptions speculative_options = {},
      std::string artifact_fingerprint = {},
      std::string draft_artifact_fingerprint = {},
      hip::QwenExecutionPolicy execution_policy =
          hip::QwenExecutionPolicy::Production())
      : model_(std::move(model)),
        dflash_model_(std::move(dflash_model)),
        max_context_(max_context),
        speculative_options_(speculative_options),
        execution_policy_(execution_policy) {
    if (!artifact_fingerprint.empty()) {
      const bool speculative = dflash_model_ != nullptr;
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = QwenCompatibilityIdentity(
              artifact_fingerprint, draft_artifact_fingerprint, max_context_,
              execution_policy_, speculative, speculative_options_),
          .payload_version = kQwenComponentPayloadVersion,
      };
      component_identity_ = persistence_->compatibility_identity;
    } else {
      std::ostringstream identity;
      identity << "schema=mbufo-process-qwen-component-v2\n"
               << "model_instance=" << model_.get() << '\n'
               << "draft_instance=" << dflash_model_.get() << '\n'
               << "context_tokens=" << max_context_ << '\n';
      const auto canonical = identity.str();
      component_identity_.assign(canonical.begin(), canonical.end());
    }
  }

  [[nodiscard]] bool UsesComponentCache() const override { return true; }
  [[nodiscard]] std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t ram) const override {
    (void)ram;
    auto resources = CreateHipComponentCacheResources(
        std::make_unique<models::qwen::ContinuationAdapter>(
            model_, max_context_, component_identity_, execution_policy_,
            dflash_model_,
            hip::QwenDFlashGpuDraftConfig{
                .max_context = max_context_,
                .max_draft_tokens = speculative_options_.max_draft_tokens,
                .policy = speculative_options_.dflash_policy},
            &ledger),
        ledger);
    // Qwen's 256-row chunks are small. Preserve a bounded group before
    // waiting, instead of synchronizing every half-MiB range individually.
    resources->reassignment_batch_size = 16;
    const auto& adapter =
        static_cast<models::qwen::ContinuationAdapter&>(*resources->adapter);
    resources->pinned_bytes = [qwen = &adapter] {
      return qwen->PinnedCaptureBytes();
    };
    return resources;
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    return std::make_unique<QwenTextRunnerState>(
        model_, dynamic_cast<models::qwen::ContinuationAdapter&>(adapter), slot,
        speculative_options_);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    auto& qwen = RequireQwenState(state);
    qwen.set_position(position);
    if (!qwen.frontier())
      qwen.set_frontier(
          0);  // Partial prefill: the suffix computes its frontier.
  }
  [[nodiscard]] bool HasDecodeFrontier(
      const TextRunnerState& state) const override {
    return RequireQwenState(state).HasDecodeFrontier();
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    const bool speculative_enabled = dflash_model_ != nullptr;
    return {
        .model_id = model_->GetConfig().model_name,
        .state_abi =
            std::string(QwenStateAbi(
                speculative_enabled, execution_policy_.UsesFp16AttentionKv(),
                execution_policy_.UsesBf16RecurrentState())) +
            "-component-v2",
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = !speculative_enabled,
                .incremental_text_is_exact = true,
                .multi_token_decode = speculative_enabled,
                .batched_multi_token_decode =
                    speculative_enabled && MaximumDecodeBatchWidth() > 1,
                .batched_multi_token_decode_max_width =
                    speculative_enabled ? MaximumDecodeBatchWidth() : 0,
                .prefix_reuse = true,
                .partial_prefill_retention = true,
                .prefill_rollback = true,
                .mixed_prefill_decode = dflash_model_ == nullptr,
            },
        .persistence = persistence_,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    auto usage = hip::QwenGpuExecutor::EstimateMemoryUsage(
        model_->GetConfig(), max_context_, execution_policy_);
    usage.request_state_bytes += hip::QwenGpuArena::RecurrentStateBytes(
        model_->GetConfig(), execution_policy_);
    std::size_t resident_weights = model_->GetResidentBytes();
    if (dflash_model_ != nullptr) {
      const auto draft = hip::QwenDFlashGpuExecutor::EstimateMemoryUsage(
          *dflash_model_, max_context_, MaximumDecodeBatchWidth());
      usage.request_state_bytes += draft.request_state_bytes;
      usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      resident_weights += dflash_model_->GetPackedWeightBytes();
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    // Snapshot buffers are device allocations, but on a unified-memory APU
    // they are carved from the same RAM as every host allocation, and HIP's
    // free figure does not see host pressure. Cap them by the host budget too.
    std::size_t snapshot_capacity = HostSnapshotBudgetBytes();
    std::size_t snapshot_ceiling = HostSnapshotCeilingBytes();
    if (capacity.has_value()) {
      snapshot_capacity = std::min(snapshot_capacity, *capacity);
      snapshot_ceiling = std::min(snapshot_ceiling, *capacity);
    }
    return {
        .resident_weights_bytes = resident_weights,
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = usage.request_state_bytes,
        .temporary_scratch_bytes = usage.temporary_scratch_bytes,
        .retained_snapshot_capacity_bytes = snapshot_capacity,
        .retained_snapshot_ceiling_bytes = snapshot_ceiling,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
    };
    // Report actual session widths, including C6. Speculation reserves up to
    // eight token rows per session in the coordinator's existing scratch.
    // (Qwen3.8 Flash-Next MTP is different: 17 rows per session —
    // max_draft_tokens 16 + anchor — see rocm::kMtpMaxChain.)
    for (std::size_t width = 2; width <= MaximumDecodeBatchWidth(); ++width) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = width,
      });
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->GetTokenizer().GetVocabSize(), [this](std::uint32_t id) {
          const auto& tokenizer = model_->GetTokenizer();
          return sampling::ConstraintVocabulary::Piece{
              tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->GetTokenizer().GetVocabSize());
  }

  // llama.cpp's Qwen3-Coder format forces "\n</think>" and also ends
  // reasoning at a tool call.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "\n</think>", {"</think>", "<tool_call>"}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->GetTokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->GetTokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        QwenChatOptions(request, max_context_));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return PrepareQwenPrompt(request, model_->GetTokenizer(),
                             model_->VisionEncoder(), max_context_,
                             image_cache_);
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    return CountQwenPromptTokens(request, model_->GetTokenizer(),
                                 model_->VisionEncoder(), max_context_,
                                 image_cache_);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    RequireQwenState(state).executor().ConfigureVision(QwenPrompt(context),
                                                       model_->VisionEncoder());
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return QwenChatOptions(request, max_context_).enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->GetTokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    throw std::logic_error(
        "Qwen execution slots are owned by the component cache");
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& qwen = RequireQwenState(state);
    qwen.PreparePrefixReuse(prefix);
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qwen = RequireQwenState(state);
    // Keep the entry recurrence pinned so cancellation can stop between
    // layers without discarding already completed prompt work.
    auto activity = qwen.BeginPrefill();
    try {
      if (offset != qwen.position()) {
        throw std::logic_error(
            "Qwen prefill offset does not match retained state");
      }
      if (offset >= prompt.size()) {
        throw std::logic_error("Qwen prefill has no remaining input");
      }
      max_input_tokens = std::min<std::size_t>(
          max_input_tokens, qwen.executor().GetMaxPromptBatch());
      if (max_input_tokens != 0 && offset != 0 &&
          qwen.ResumeGeneratedToken(prompt[offset],
                                    offset + 1 == prompt.size())) {
        return {.consumed_tokens = 1,
                .decode_ready = offset + 1 == prompt.size()};
      }
      if (qwen.speculative()) {
        if (offset == 0) {
          const auto consumed = std::min(max_input_tokens, prompt.size());
          if (consumed == 0) {
            throw std::logic_error(
                "Qwen prefill requires a nonzero token budget");
          }
          qwen.PrimeSpeculative(prompt.first(consumed),
                                consumed == prompt.size());
          return {.consumed_tokens = consumed,
                  .decode_ready = consumed == prompt.size()};
        }
        const std::size_t consumed =
            std::min(max_input_tokens, prompt.size() - offset);
        if (consumed == 0) {
          throw std::logic_error(
              "Qwen DFlash retained prefix has no suffix to prefill");
        }
        qwen.ExtendSpeculative(prompt.subspan(offset, consumed),
                               offset + consumed == prompt.size());
        return {
            .consumed_tokens = consumed,
            .decode_ready = offset + consumed == prompt.size(),
        };
      }

      const std::size_t consumed =
          std::min(max_input_tokens, prompt.size() - offset);
      const bool decode_ready = offset + consumed == prompt.size();
      const auto frontier = qwen.executor().ForwardPromptBatch(
          prompt.subspan(offset, consumed), static_cast<std::uint32_t>(offset),
          decode_ready);
      qwen.set_position(offset + consumed);
      if (decode_ready) {
        qwen.set_frontier(frontier);
      }
      return {
          .consumed_tokens = consumed,
          .decode_ready = decode_ready,
      };
    } catch (const hip::QwenPrefillCancelled&) {
      activity.Rollback();
      return {.interrupted = true};
    }
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& qwen = RequireQwenState(state);
    const TextRunnerToken token = qwen.SelectFrontier(sampler);
    if (EndsTurn(qwen, model_->GetTokenizer(), token)) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    // Keep the selected token separate from the argmax frontier: sampled
    // cancellation and prompt snapshot restore must not replace that argmax.
    qwen.set_pending_token(token);
    return {
        .stop = false,
        .token = token,
        .piece = std::string(model_->GetTokenizer().DecodeToken(token)),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& qwen = RequireQwenState(state);
    if (qwen.speculative()) {
      throw std::logic_error(
          "Qwen DFlash decoding requires a multi-token decode step");
    }
    if (!qwen.frontier().has_value()) {
      throw std::logic_error("Qwen decode state has no retained frontier");
    }
    const auto frontier = qwen.executor().ForwardToken(
        token, static_cast<std::uint32_t>(qwen.position()));
    qwen.set_position(qwen.position() + 1);
    qwen.set_frontier(frontier);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    auto& qwen = RequireQwenState(state);
    if (!qwen.speculative()) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    return qwen.DecodeSpeculative(max_tokens, sampler);
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (dflash_model_ == nullptr || decodes.size() < 2 ||
        decodes.size() > MaximumDecodeBatchWidth()) {
      return SerialDecodeBatch(decodes);
    }
    std::vector<TextDecodeStep> steps(decodes.size());
    std::vector<QwenTextRunnerState*> states;
    std::vector<sampling::SamplerState> samplers;
    std::vector<speculative::SpeculativeVerifier::StepRequest> requests;
    std::vector<std::size_t> indices;
    states.reserve(decodes.size());
    samplers.reserve(decodes.size());
    requests.reserve(decodes.size());
    indices.reserve(decodes.size());
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      const auto& decode = decodes[index];
      auto& state = RequireQwenState(decode.state.get());
      states.push_back(&state);
      auto& sampler = samplers.emplace_back(decode.sampler.get());
      if (state.PrepareSpeculativeDecode(decode.max_tokens, sampler,
                                         steps[index])) {
        requests.push_back(state.VerificationRequest(
            decode.max_tokens - steps[index].selections.size(), sampler));
        indices.push_back(index);
      }
    }
    auto verified = speculative::SpeculativeVerifier::VerifyBatch(requests);
    for (std::size_t item = 0; item < indices.size(); ++item) {
      const std::size_t index = indices[item];
      states[index]->FinishSpeculativeDecode(std::move(verified[item]),
                                             samplers[index], steps[index]);
    }
    for (std::size_t index = 0; index < decodes.size(); ++index)
      decodes[index].sampler.get().SetRngState(samplers[index].rng_state());
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (dflash_model_ != nullptr) {
      throw std::logic_error(
          "Qwen DFlash sessions do not support batch advance");
    }
    if (advances.size() < 2 || advances.size() > 8) {
      throw std::invalid_argument(
          "Qwen batched decode requires two to eight sessions");
    }
    // Blocked-Q8 KV (the production default) has no batched decode kernel:
    // ForwardTokenBatch refuses it, so each session runs its own decode core.
    if (execution_policy_.UsesBlockedAttentionKv())
      return SerialAdvanceBatch(advances);

    std::vector<hip::QwenGpuBatchItem> items;
    std::vector<QwenTextRunnerState*> states;
    items.reserve(advances.size());
    states.reserve(advances.size());
    for (const auto& advance : advances) {
      auto& qwen = RequireQwenState(advance.state.get());
      if (!qwen.frontier().has_value()) {
        throw std::logic_error("Qwen batched state has no retained frontier");
      }
      states.push_back(&qwen);
      items.push_back({
          .executor = &qwen.executor(),
          .token_id = advance.token,
          .position = static_cast<std::uint32_t>(qwen.position()),
      });
    }

    const auto frontiers = hip::QwenGpuExecutor::ForwardTokenBatch(items);
    if (frontiers.size() != states.size()) {
      throw std::runtime_error(
          "Qwen batched decode returned an invalid frontier count");
    }
    for (std::size_t index = 0; index < states.size(); ++index) {
      states[index]->set_position(states[index]->position() + 1);
      states[index]->set_frontier(frontiers[index]);
    }
  }

  // Merged plain-advance + prefill chunk. Every lane runs exactly the code its
  // unmerged advance runs (AdvanceLanes: ForwardTokenBatch on the decode
  // kernels for two or more lanes on F16 KV, the per-session decode core
  // otherwise) and the chunk runs the unchanged Prefill core (ForwardPromptBatch,
  // blocked-Q8 and vision aware), so each half is arithmetically identical to
  // its separate run. Lanes and chunk live in different executors, hence in
  // different arenas and streams, and run back to back (an experiment that
  // submitted the lane pass from a helper thread while the chunk ran changed
  // neither bits nor speed). DFlash runners advertise no merged step.
  [[nodiscard]] bool AdvanceBatchWithPrefillChunk(
      std::span<const TextRunnerAdvance> advances,
      std::span<std::exception_ptr> advance_failures,
      TextRunnerState& prefill_state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens,
      TextPrefillStep& chunk_step) const override {
    static const bool trace = std::getenv("MBUFO_MIXED_TRACE") != nullptr;
    if (dflash_model_ != nullptr)
      throw MixedBatchRefused("Qwen DFlash sessions do not merge");
    auto& chunk = RequireQwenState(prefill_state);
    if (advances.empty() || advances.size() > 8 ||
        advance_failures.size() < advances.size() || max_input_tokens == 0 ||
        offset != chunk.position() || offset >= prompt.size() ||
        chunk.speculative())
      throw MixedBatchRefused("Qwen mixed batch preconditions unmet");
    for (std::size_t i = 0; i < advances.size(); ++i) {
      auto& lane = RequireQwenState(advances[i].state.get());
      if (&lane == &chunk || lane.speculative() ||
          !lane.frontier().has_value() || lane.position() >= max_context_ ||
          advances[i].token > std::numeric_limits<std::uint32_t>::max())
        throw MixedBatchRefused("Qwen mixed batch lane unusable");
      for (std::size_t j = 0; j < i; ++j)
        if (&RequireQwenState(advances[j].state.get()) == &lane)
          throw MixedBatchRefused("Qwen mixed batch lane repeated");
    }
    for (std::size_t i = 0; i < advances.size(); ++i)
      advance_failures[i] = nullptr;

    TextPrefillStep step;
    std::exception_ptr chunk_failure;
    const auto run_chunk = [&] {
      try {
        step = Prefill(prefill_state, prompt, offset, max_input_tokens);
      } catch (...) {
        chunk_failure = std::current_exception();
      }
    };
    AdvanceLanes(advances, advance_failures);
    // A failed shared lane pass leaves every lane failed; the chunk is not run.
    if (advances.size() >= 2 && advance_failures[0]) return false;
    run_chunk();
    if (chunk_failure) {
      if (trace) {
        try {
          std::rethrow_exception(chunk_failure);
        } catch (const std::exception& error) {
          std::cerr << "[qwen-mixed] chunk failed: " << error.what() << '\n';
        } catch (...) {
        }
      }
      // Lanes already advanced; only the chunk request is reported failed.
      return false;
    }
    if (trace)
      std::cerr << "[qwen-mixed] lanes=" << advances.size()
                << " chunk=" << step.consumed_tokens << " offset=" << offset
                << '\n';
    if (step.interrupted || step.consumed_tokens == 0) return false;
    chunk_step = step;
    return true;
  }

  // The merged multi-token step has no Qwen engine pass to ride (DFlash verify
  // is per session), so it refuses explicitly with nothing mutated.
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatchWithPrefillChunk(
      std::span<const TextRunnerDecode>, TextRunnerState&,
      std::span<const TextRunnerToken>, std::size_t, std::size_t,
      TextPrefillStep&) const override {
    RefuseMixedBatch("Qwen has no merged multi-token decode");
  }
  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenState(state).position();
  }
  void PrepareCancellation(TextRunnerState& state) const override {
    RequireQwenState(state).PrepareCancellation();
  }

private:
  // Advances every lane by one token and records per-lane failures (never
  // throws). Two or more lanes on F16 KV share one ForwardTokenBatch weight
  // pass (a failure poisons every lane of that pass); a single lane, or
  // blocked-Q8 KV (no batched decode kernel), runs the per-session decode core.
  void AdvanceLanes(std::span<const TextRunnerAdvance> advances,
                    std::span<std::exception_ptr> failures) const noexcept {
    if (advances.size() >= 2 && !execution_policy_.UsesBlockedAttentionKv()) {
      try {
        std::vector<hip::QwenGpuBatchItem> items;
        std::vector<QwenTextRunnerState*> states;
        items.reserve(advances.size());
        states.reserve(advances.size());
        for (const auto& advance : advances) {
          auto& qwen = RequireQwenState(advance.state.get());
          states.push_back(&qwen);
          items.push_back({.executor = &qwen.executor(),
                           .token_id = advance.token,
                           .position =
                               static_cast<std::uint32_t>(qwen.position())});
        }
        const auto frontiers = hip::QwenGpuExecutor::ForwardTokenBatch(items);
        if (frontiers.size() != states.size())
          throw std::runtime_error(
              "Qwen batched decode returned an invalid frontier count");
        for (std::size_t i = 0; i < states.size(); ++i) {
          states[i]->set_position(states[i]->position() + 1);
          states[i]->set_frontier(frontiers[i]);
        }
      } catch (...) {
        for (std::size_t i = 0; i < advances.size(); ++i)
          failures[i] = std::current_exception();
      }
      return;
    }
    for (std::size_t i = 0; i < advances.size(); ++i) {
      try {
        Advance(advances[i].state.get(), advances[i].token);
      } catch (...) {
        failures[i] = std::current_exception();
      }
    }
  }  [[nodiscard]] std::size_t MaximumDecodeBatchWidth() const noexcept {
    const std::size_t rows_per_session = dflash_model_ != nullptr ? 8 : 1;
    return std::clamp<std::size_t>(max_context_ / rows_per_session, 1, 8);
  }

  mutable models::qwen::vision::ImageCache image_cache_;
  std::shared_ptr<const hip::QwenGpuModel> model_;
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model_;
  std::uint32_t max_context_;
  speculative::SpeculativeOptions speculative_options_;
  hip::QwenExecutionPolicy execution_policy_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
  cache::Identity component_identity_;
};

std::vector<TextRunnerToken> DeepSeekRunnerTokens(std::span<const int> tokens) {
  std::vector<TextRunnerToken> converted;
  converted.reserve(tokens.size());
  for (const int token : tokens) {
    if (token < 0) {
      throw std::invalid_argument("DeepSeek token ID must not be negative");
    }
    converted.push_back(static_cast<TextRunnerToken>(token));
  }
  return converted;
}

std::vector<int> DeepSeekEngineTokens(std::span<const TextRunnerToken> tokens) {
  std::vector<int> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("DeepSeek token ID exceeds engine range");
    }
    converted.push_back(static_cast<int>(token));
  }
  return converted;
}

std::string_view ChatRoleName(tokenization::ChatRole role) {
  switch (role) {
    case tokenization::ChatRole::kSystem:
      return "system";
    case tokenization::ChatRole::kDeveloper:
      return "developer";
    case tokenization::ChatRole::kAssistant:
      return "assistant";
    case tokenization::ChatRole::kTool:
      return "tool";
    case tokenization::ChatRole::kUser:
      return "user";
  }
  return "user";
}

class DeepSeekTextRunnerState final : public TextRunnerState {
public:
  DeepSeekTextRunnerState(
      const std::shared_ptr<models::deepseek_v4_flash::Model>& model,
      std::uint32_t max_context, bool use_dspark) {
    std::string error;
    session_ = model->CreateSession(
        use_dspark ? mbufo::core::SessionMode::kSpeculative
                   : mbufo::core::SessionMode::kAutoregressive,
        max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create DeepSeek session: " + error);
    }
  }

  void SetCancellationCheck(const CancellationCheck& is_cancelled) override {
    session_->SetCancellationCheck(is_cancelled);
  }

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    session_->Invalidate();
    position_ = 0;
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    const std::uint64_t bytes = session_->PayloadBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      return {};
    }
    return {
        .per_request_state_bytes = static_cast<std::size_t>(bytes),
        .temporary_scratch_bytes = std::nullopt,
    };
  }

  [[nodiscard]] models::deepseek_v4_flash::Session& session() const {
    return *session_;
  }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<models::deepseek_v4_flash::Session> session_;
  std::size_t position_{0};
};

class DeepSeekTextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  DeepSeekTextRunnerSnapshot(
      std::shared_ptr<models::deepseek_v4_flash::Model> model,
      std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot,
      std::size_t position)
      : model(std::move(model)),
        snapshot(std::move(snapshot)),
        position(position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    if (snapshot == nullptr ||
        snapshot->SizeBytes() > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::size_t>::max())) {
      return 0;
    }
    return static_cast<std::size_t>(snapshot->SizeBytes());
  }

  std::shared_ptr<models::deepseek_v4_flash::Model> model;
  std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot;
  std::size_t position;
};

DeepSeekTextRunnerState& RequireDeepSeekState(TextRunnerState& state) {
  auto* deepseek = dynamic_cast<DeepSeekTextRunnerState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("text runner state is not DeepSeek");
  }
  return *deepseek;
}

const DeepSeekTextRunnerState& RequireDeepSeekState(
    const TextRunnerState& state) {
  const auto* deepseek = dynamic_cast<const DeepSeekTextRunnerState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("text runner state is not DeepSeek");
  }
  return *deepseek;
}

class DeepSeekTextRunner final : public HipTextModelRunner {
public:
  // Batch contract (see TextModelRunner): native DecodeBatch/AdvanceBatch below; no merged prefill+decode.
  MBUFO_RUNNER_NO_MIXED_BATCH();
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kDeepSeek;
  }
  DeepSeekTextRunner(std::shared_ptr<models::deepseek_v4_flash::Model> model,
                     std::uint32_t max_context, bool use_dspark,
                     std::uint32_t max_draft_tokens,
                     std::string artifact_fingerprint = {},
                     std::string support_fingerprint = {})
      : model_(std::move(model)),
        max_context_(max_context),
        use_dspark_(use_dspark),
        max_draft_tokens_(std::max(max_draft_tokens, 1u)) {
    if (!artifact_fingerprint.empty()) {
      if (use_dspark_ && !IsSha256Hex(support_fingerprint)) {
        throw std::invalid_argument(
            "DSpark disk cache requires a support SHA-256 fingerprint");
      }
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = DeepSeekCompatibilityIdentity(
              artifact_fingerprint,
              use_dspark_ ? support_fingerprint : std::string{}, max_context_,
              max_draft_tokens_),
          .payload_version = DS4_SESSION_PAYLOAD_VERSION,
      };
    }
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    const bool dspark = use_dspark_;
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kDeepSeekStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = dspark,
                .batched_multi_token_decode = dspark,
                .batched_multi_token_decode_max_width = 8u,
                .prefix_reuse = true,
            },
        .persistence = persistence_,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    return {
        .resident_weights_bytes = std::nullopt,
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
    };
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = width,
      });
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              model_->DecodeToken(static_cast<int>(id)),
              model_->IsStopToken(static_cast<int>(id))};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->VocabSize());
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return DeepSeekRunnerTokens(model_->Tokenize(text));
  }
  [[nodiscard]] std::vector<TextRunnerToken> TokenizeControl(
      std::string_view text) const override {
    return DeepSeekRunnerTokens(model_->Tokenize(text, true));
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    std::vector<models::deepseek_v4_flash::ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
      models::deepseek_v4_flash::ChatMessage converted{
          .role = std::string(ChatRoleName(message.role)),
          .content = message.content + message.framing_suffix,
          .reasoning_content = message.thought,
          .tool_calls = {},
          .tool_call_id = message.tool_call_id,
      };
      converted.tool_calls.reserve(message.tool_calls.size());
      for (const auto& call : message.tool_calls) {
        models::deepseek_v4_flash::ChatMessage::ToolCall converted_call{
            .name = call.name,
            .arguments = {},
            .id = call.id,
        };
        converted_call.arguments.reserve(call.arguments.size());
        for (const auto& argument : call.arguments) {
          converted_call.arguments.push_back({
              .name = argument.name,
              .value = argument.value,
              .is_string = argument.is_string,
          });
        }
        converted.tool_calls.push_back(std::move(converted_call));
      }
      messages.push_back(std::move(converted));
    }

    std::vector<models::deepseek_v4_flash::ChatTool> tools;
    if (request.tool_choice != ChatRequest::ToolChoice::kNone) {
      tools.reserve(request.tools.size());
      for (const auto& tool : request.tools) {
        tools.push_back({
            .name = tool.name,
            .description = tool.description,
            .parameters_json = tool.parameters_json,
            .definition_json =
                tool.definition_json.empty()
                    ? std::string{}
                    : json::parse(tool.definition_json)["function"].dump(),
        });
      }
    }
    auto options = models::deepseek_v4_flash::ResolveDeepSeekChatOptions(
        request.reasoning);
    options.tools_present =
        !request.tools.empty() &&
        request.tool_choice != ChatRequest::ToolChoice::kNone;
    auto tokens =
        DeepSeekRunnerTokens(model_->EncodeChat(messages, tools, options));
    if (tokens.empty()) {
      return std::nullopt;
    }
    return tokens;
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return models::deepseek_v4_flash::ResolveDeepSeekChatOptions(
               request.reasoning)
                   .enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    auto prepared = TextModelRunner::PreparePrompt(request);
    if (prepared) {
      // DeepSeek joins adjacent user/tool messages into one user block.
      // A new user turn after a discarded assistant can therefore also change
      // the final BPE token before the generation suffix (e.g. ">\n\n").
      // Find the exact stable frontier using the renderer/tokenizer instead
      // of assuming the suffix alone is the only changed part of the prompt.
      auto continuation = request;
      continuation.messages.emplace_back(tokenization::ChatRole::kUser, "");
      const auto continued = RenderAndTokenize(continuation);
      if (continued) {
        prepared->cache_prefix_tokens = static_cast<std::size_t>(
            std::mismatch(prepared->tokens.begin(), prepared->tokens.end(),
                          continued->begin(), continued->end())
                .first -
            prepared->tokens.begin());
      }
    }
    return prepared;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const TextRunnerToken token : tokens) {
      if (token >
          static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("DeepSeek token ID exceeds engine range");
      }
      text += model_->DecodeToken(static_cast<int>(token));
    }
    return text;
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<DeepSeekTextRunnerState>(model_, max_context_,
                                                     use_dspark_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() != prefix.size()) {
      throw std::logic_error(
          "DeepSeek reused prefix does not match checkpoint");
    }
    deepseek.session().BeginRequest();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (offset != deepseek.position()) {
      throw std::logic_error(
          "DeepSeek prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("DeepSeek prefill has no remaining input");
    }

    const std::size_t consumed =
        std::min<std::size_t>({max_input_tokens, prompt.size() - offset,
                               deepseek.session().PrefillCapacity()});
    const std::size_t next_position = offset + consumed;
    const auto prefix = DeepSeekEngineTokens(prompt.first(next_position));
    std::string error;
    if (!deepseek.session().Sync(prefix, &error)) {
      throw std::runtime_error("DeepSeek prefill failed: " + error);
    }
    deepseek.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return {.stop = true, .piece = {}};
    // A sampled DSpark cycle may have drawn the next token already.
    int token = deepseek.session().TakePendingDsparkToken();
    if (token < 0) {
      std::string error;
      const auto logits = deepseek.session().CopyLogits(&error);
      if (logits.empty()) {
        throw std::runtime_error("DeepSeek token selection failed: " + error);
      }
      token = static_cast<int>(sampler.Sample(logits));
    }
    if (deepseek.stop_at_eos() && model_->IsStopToken(token)) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->DecodeToken(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("DeepSeek token ID exceeds engine range");
    }
    auto& deepseek = RequireDeepSeekState(state);
    std::string error;
    if (!deepseek.session().Evaluate(static_cast<int>(token), &error)) {
      throw std::runtime_error("DeepSeek decode failed: " + error);
    }
    deepseek.set_position(deepseek.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return TextDecodeSelection{.stop = true, .piece = {}};
    std::string error;
    const auto logits = deepseek.session().CopyLogits(&error);
    if (logits.empty())
      throw std::runtime_error("DeepSeek token preview failed: " + error);
    const auto token = sampler.Sample(logits);
    return TextDecodeSelection{
        .stop = deepseek.stop_at_eos() &&
                model_->IsStopToken(static_cast<int>(token)),
        .token = token,
        .piece = model_->DecodeToken(static_cast<int>(token)),
    };
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_dspark_) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "DeepSeek DSpark decode budget must be at least one token");
    }

    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return {.selections = {}, .stop = true};
    const auto stats_before = deepseek.session().DsparkStatistics();
    std::optional<models::deepseek_v4_flash::DsparkSamplerBridge> bridge;
    if (!sampler.config().can_use_unmodified_argmax() ||
        sampler.config().constraint != nullptr ||
        sampler.ForcedToken().has_value()) {
      bridge.emplace(sampler);
    }
    std::vector<int> emitted;
    std::string error;
    if (!deepseek.session().DsparkStep(
            max_tokens, max_draft_tokens_, &emitted, &error,
            bridge ? bridge->hook() : nullptr, deepseek.stop_at_eos())) {
      throw std::runtime_error("DeepSeek DSpark decode failed: " + error);
    }
    if (emitted.empty()) {
      throw std::runtime_error("DeepSeek DSpark decode produced no tokens");
    }
    if (bridge) {
      sampler.SetRngState(bridge->rng_state());
    }

    TextDecodeStep step;
    deepseek.set_position(deepseek.session().Position());
    step.selections.reserve(emitted.size());
    for (const int token : emitted) {
      if (deepseek.stop_at_eos() && model_->IsStopToken(token)) {
        step.stop = true;
        break;
      }
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->DecodeToken(token),
      });
    }
    const auto stats_after = deepseek.session().DsparkStatistics();
    step.draft_rounds = stats_after.steps - stats_before.steps;
    step.draft_tokens =
        stats_after.support_drafted - stats_before.support_drafted;
    step.draft_accepted_tokens =
        stats_after.support_accepted - stats_before.support_accepted;
    return step;
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (!use_dspark_ || decodes.size() < 2 || decodes.size() > 8) {
      return SerialDecodeBatch(decodes);
    }
    if (std::any_of(decodes.begin(), decodes.end(), [&](const auto& item) {
          return RequireDeepSeekState(item.state.get()).position() >=
                 max_context_;
        })) {
      std::vector<TextRunnerDecode> active;
      std::vector<std::size_t> active_indices;
      std::vector<TextDecodeStep> steps(decodes.size());
      for (std::size_t i = 0; i < decodes.size(); ++i) {
        if (RequireDeepSeekState(decodes[i].state.get()).position() >=
            max_context_) {
          steps[i].stop = true;
        } else {
          active.push_back(decodes[i]);
          active_indices.push_back(i);
        }
      }
      if (!active.empty()) {
        auto active_steps = DecodeBatch(active);
        for (std::size_t i = 0; i < active.size(); ++i)
          steps[active_indices[i]] = std::move(active_steps[i]);
      }
      return steps;
    }
    // Greedy and sampled requests share one DSpark cohort; the runtime keeps
    // the greedy verifier for items without a sampler.
    std::array<models::deepseek_v4_flash::SessionDsparkBatchItem, 8> items{};
    std::array<DeepSeekTextRunnerState*, 8> states{};
    std::array<models::deepseek_v4_flash::Session::DsparkStats, 8>
        stats_before{};
    std::array<std::vector<int>, 8> emitted{};
    std::array<std::optional<models::deepseek_v4_flash::DsparkSamplerBridge>, 8>
        bridges{};
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      auto& deepseek = RequireDeepSeekState(decodes[index].state.get());
      states[index] = &deepseek;
      stats_before[index] = deepseek.session().DsparkStatistics();
      auto& sampler = decodes[index].sampler.get();
      if (!sampler.config().can_use_unmodified_argmax() ||
          sampler.config().constraint != nullptr ||
          sampler.ForcedToken().has_value()) {
        bridges[index].emplace(sampler);
      }
      items[index] = {
          .session = &deepseek.session(),
          .max_tokens = decodes[index].max_tokens,
          .max_draft_tokens = max_draft_tokens_,
          .emitted = &emitted[index],
          .sampler = bridges[index] ? bridges[index]->hook() : nullptr,
          .stop_at_eos = states[index]->stop_at_eos(),
      };
    }

    std::string error;
    bool interrupted = false;
    if (!model_->DsparkStepBatch(
            std::span<const models::deepseek_v4_flash::SessionDsparkBatchItem>(
                items.data(), decodes.size()),
            &error, &interrupted)) {
      // Cancellation is rejected before sampling or device-state mutation.
      // Isolate the cancelled row rather than failing every client in its
      // batch. Ordinary cycles retain the shared physical batch.
      if (interrupted)
        return SerialDecodeBatch(decodes);
      throw std::runtime_error("DeepSeek DSpark batch decode failed: " + error);
    }

    std::vector<TextDecodeStep> steps(decodes.size());
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      if (emitted[index].empty()) {
        throw std::runtime_error("DeepSeek DSpark batch produced no tokens");
      }
      if (bridges[index]) {
        decodes[index].sampler.get().SetRngState(bridges[index]->rng_state());
      }
      auto& step = steps[index];
      step.execution_plan = {
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = decodes.size(),
      };
      states[index]->set_position(states[index]->session().Position());
      step.selections.reserve(emitted[index].size());
      for (const int token : emitted[index]) {
        if (states[index]->stop_at_eos() && model_->IsStopToken(token)) {
          step.stop = true;
          break;
        }
        step.selections.push_back({
            .stop = false,
            .token = static_cast<TextRunnerToken>(token),
            .piece = model_->DecodeToken(token),
        });
      }
      const auto stats_after = states[index]->session().DsparkStatistics();
      step.draft_rounds = stats_after.steps - stats_before[index].steps;
      step.draft_tokens =
          stats_after.support_drafted - stats_before[index].support_drafted;
      step.draft_accepted_tokens =
          stats_after.support_accepted - stats_before[index].support_accepted;
    }
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2 || advances.size() > 8) {
      throw std::invalid_argument(
          "DeepSeek batched decode requires two to eight sessions");
    }

    std::array<models::deepseek_v4_flash::SessionBatchItem, 8> items{};
    std::array<DeepSeekTextRunnerState*, 8> states{};
    std::size_t item_count = 0;
    for (const auto& advance : advances) {
      if (advance.token >
          static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("DeepSeek token ID exceeds engine range");
      }
      auto& deepseek = RequireDeepSeekState(advance.state.get());
      states[item_count] = &deepseek;
      items[item_count] = {
          .session = &deepseek.session(),
          .token = static_cast<int>(advance.token),
      };
      ++item_count;
    }

    std::string error;
    bool interrupted = false;
    if (!model_->EvaluateBatch(
            std::span<const models::deepseek_v4_flash::SessionBatchItem>(
                items.data(), item_count),
            &error, &interrupted)) {
      if (interrupted) {
        SerialAdvanceBatch(advances);
        return;
      }
      throw std::runtime_error("DeepSeek batch decode failed: " + error);
    }
    for (std::size_t index = 0; index < item_count; ++index) {
      states[index]->set_position(states[index]->position() + 1);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireDeepSeekState(state).position();
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    const auto& session = RequireDeepSeekState(state).session();
    if (use_dspark_ && session.DsparkStatistics().context_tokens !=
                           static_cast<uint32_t>(session.Position())) {
      throw std::runtime_error("DSpark prefix lacks complete support state");
    }
    const std::uint64_t bytes = session.PayloadBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error("DeepSeek snapshot size is unavailable");
    }
    return static_cast<std::size_t>(bytes);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& deepseek = RequireDeepSeekState(state);
    std::string error;
    auto snapshot = deepseek.session().SaveSnapshot(&error);
    if (snapshot == nullptr) {
      throw std::runtime_error("DeepSeek snapshot failed: " + error);
    }
    return std::make_unique<DeepSeekTextRunnerSnapshot>(
        model_, std::move(snapshot), deepseek.position());
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "DeepSeek snapshot does not belong to this model");
    }
    auto& restored = RequireDeepSeekState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(*deepseek_snapshot->snapshot,
                                            &error)) {
      throw std::runtime_error("DeepSeek snapshot restore failed: " + error);
    }
    restored.set_position(deepseek_snapshot->position);
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "DeepSeek persistent snapshot does not belong to this model");
    }
    return deepseek_snapshot->PayloadBytes();
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr ||
        destination.size() != deepseek_snapshot->PayloadBytes() ||
        !deepseek_snapshot->snapshot->CopyTo(destination)) {
      throw std::invalid_argument(
          "DeepSeek persistent snapshot serialization failed");
    }
    return destination.size();
  }

  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    (void)PersistentSnapshotPayloadBytes(snapshot);
    sink(dynamic_cast<const DeepSeekTextRunnerSnapshot&>(snapshot)
             .snapshot->bytes());
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& restored = RequireDeepSeekState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(payload, &error)) {
      throw std::runtime_error("DeepSeek persistent snapshot restore failed: " +
                               error);
    }
    const int position = restored.session().Position();
    if (position <= 0 || static_cast<std::uint64_t>(position) > max_context_) {
      restored.session().Invalidate();
      throw std::runtime_error(
          "DeepSeek persistent snapshot restored an invalid position");
    }
    restored.set_position(static_cast<std::size_t>(position));
  }

private:
  std::shared_ptr<models::deepseek_v4_flash::Model> model_;
  std::uint32_t max_context_;
  bool use_dspark_;
  std::uint32_t max_draft_tokens_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
};

/// Derives the content identity that keys persistent continuations for one
/// GGUF artifact. Full digests are cached by the open file metadata.
bool FingerprintArtifact(std::string_view label, const core::GgufReader& reader,
                         std::string* fingerprint, std::string* error) {
  const auto start = std::chrono::steady_clock::now();
  Logger::Info("loader", "event=load_phase phase=artifact_identity model=" +
                             std::string(label));
  try {
    *fingerprint = core::GgufIdentityHex(reader);
  } catch (const std::exception& exception) {
    SetError(error, std::string("Failed to fingerprint ") + std::string(label) +
                        " GGUF: " + exception.what());
    return false;
  }
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
  std::ostringstream message;
  message << "Fingerprinted " << label << " artifact ("
          << core::kGgufIdentityScheme << ", " << reader.GetTensorCount()
          << " tensors) in " << std::fixed << std::setprecision(0) << elapsed_ms
          << " ms";
  Logger::Info("engine", message.str());
  return true;
}

bool FingerprintArtifactFile(std::string_view label,
                             const std::filesystem::path& path,
                             std::string* fingerprint, std::string* error) {
  std::string open_error;
  const auto reader = core::GgufReader::OpenFile(path, &open_error);
  if (reader == nullptr) {
    SetError(error, std::string("Failed to open ") + std::string(label) +
                        " GGUF for fingerprinting: " + open_error);
    return false;
  }
  return FingerprintArtifact(label, *reader, fingerprint, error);
}

#endif

#if defined(ENGINE_ENABLE_HIP)
// Each Flash-Next request owns its recurrent/KV state and snapshots. Target
// projections share batches; sampling and rollback remain request-local.
constexpr std::string_view kQwenFlashNextStateAbi =
    "qwen38-flash-next-rocm-session-v1";

using QwenFlashNextModel = models::qwen38_flash_next::Model;
using QwenFlashNextSession = models::qwen38_flash_next::Session;

std::vector<std::int32_t> QwenFlashNextEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

class QwenFlashNextTextRunnerState final : public TextRunnerState {
public:
  QwenFlashNextTextRunnerState(QwenFlashNextSession& session,
                               cache::Adapter& adapter, cache::Slot& slot)
      : session_(&session), adapter_(&adapter), slot_(&slot) {}

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    (void)adapter_->Invalidate(*slot_);
    position_ = 0;
  }
  void SetCancellationCheck(const CancellationCheck& check) override {
    session_->SetCancellationCheck(check);
  }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {.per_request_state_bytes = session_->AllocatedBytes(),
            .temporary_scratch_bytes = 0};
  }

  [[nodiscard]] QwenFlashNextSession& session() const { return *session_; }
  [[nodiscard]] models::qwen38_flash_next::ContinuationAdapter& adapter()
      const {
    return dynamic_cast<models::qwen38_flash_next::ContinuationAdapter&>(
        *adapter_);
  }
  [[nodiscard]] cache::Slot& slot() const { return *slot_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  QwenFlashNextSession* session_{};
  cache::Adapter* adapter_{};
  cache::Slot* slot_{};
  std::size_t position_{0};
};

QwenFlashNextTextRunnerState& RequireQwenFlashNextState(
    TextRunnerState& state) {
  auto* qfn = dynamic_cast<QwenFlashNextTextRunnerState*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.8-Flash-Next");
  }
  return *qfn;
}

const QwenFlashNextTextRunnerState& RequireQwenFlashNextState(
    const TextRunnerState& state) {
  const auto* qfn = dynamic_cast<const QwenFlashNextTextRunnerState*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.8-Flash-Next");
  }
  return *qfn;
}

class QwenFlashNextTextRunner final : public HipTextModelRunner {
public:
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
  QwenFlashNextTextRunner(std::shared_ptr<QwenFlashNextModel> model,
                          std::uint32_t max_context, bool use_mtp,
                          std::uint32_t max_draft_tokens,
                          std::string artifact_fingerprint = {},
                          std::string mtp_fingerprint = {},
                          std::string kv_cache_quant = "f16")
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        max_draft_tokens_(max_draft_tokens),
        kv_cache_quant_(std::move(kv_cache_quant)) {
    // The typed in-process API owns an immutable loaded model but need not
    // supply file fingerprints when disk caching is disabled. Its model
    // lifetime is an exact process-local identity; durable caches still
    // require the validated full artifact fingerprints below.
    if (!artifact_fingerprint.empty()) {
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = QwenFlashNextCompatibilityIdentity(
              artifact_fingerprint, use_mtp_ ? mtp_fingerprint : std::string{},
              use_mtp_, max_context_, max_draft_tokens_,
              model_->DecodeConcurrency(), kv_cache_quant_,
              models::qwen38_flash_next::RopePositionPolicy(
                  model_->config().rope_scaling)),
          .payload_version =
              models::qwen38_flash_next::Session::kSnapshotPayloadVersion,
      };
      component_identity_ = persistence_->compatibility_identity;
    } else {
      std::ostringstream identity;
      identity << "schema=mbufo-process-text-continuation-v1\n"
               << "model_instance=" << model_.get() << '\n'
               << "context_tokens=" << max_context_ << '\n'
               << "mtp=" << use_mtp_ << '\n'
               << "draft_max_tokens=" << max_draft_tokens_ << '\n'
               << "draft_cost_concurrency=" << model_->DecodeConcurrency()
               << '\n';
      const auto canonical = identity.str();
      component_identity_.assign(canonical.begin(), canonical.end());
    }
  }

  [[nodiscard]] bool UsesComponentCache() const override { return true; }
  [[nodiscard]] std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t ram) const override {
    (void)ram;
    return CreateHipComponentCacheResources(
        std::make_unique<models::qwen38_flash_next::ContinuationAdapter>(
            model_,
            use_mtp_ ? core::SessionMode::kSpeculative
                     : core::SessionMode::kAutoregressive,
            max_context_, component_identity_),
        ledger);
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& flash =
        dynamic_cast<models::qwen38_flash_next::ContinuationAdapter&>(adapter);
    return std::make_unique<QwenFlashNextTextRunnerState>(
        flash.GetSession(slot), adapter, slot);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    RequireQwenFlashNextState(state).set_position(position);
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kQwenFlashNextStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_mtp_,
                .batched_multi_token_decode = use_mtp_,
                .batched_multi_token_decode_max_width = use_mtp_ ? 8u : 0u,
                .prefix_reuse = true,
                .in_pass_checkpoint = true,
                .partial_prefill_retention = true,
                .mixed_prefill_decode = true,
            },
        .persistence = persistence_,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      const auto deferred = model_->DeferredScratchBytes();
      capacity = free_bytes > deferred ? free_bytes - deferred : 0;
    }
    // Admission charges complete snapshot payloads. Flash-Next can retain
    // mutable state and protected K/V rows in private device storage.
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = model_->SessionBytes(
            use_mtp_ ? mbufo::core::SessionMode::kSpeculative
                     : mbufo::core::SessionMode::kAutoregressive,
            max_context_),
        // Runtime scratch is shared and already allocated at model load;
        // reserve its remaining lazy buffers once from aggregate capacity.
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->tokenizer().GetVocabSize(), [this](std::uint32_t id) {
          const auto& tokenizer = model_->tokenizer();
          return sampling::ConstraintVocabulary::Piece{
              tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->tokenizer().GetVocabSize());
  }

  // llama.cpp's Qwen3-Coder format forces "\n</think>" and also ends
  // reasoning at a tool call.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "\n</think>", {"</think>", "<tool_call>"}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->tokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        QwenChatOptions(request, max_context_));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return PrepareQwenPrompt(request, model_->tokenizer(),
                             model_->VisionEncoder(), max_context_,
                             image_cache_);
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    return CountQwenPromptTokens(request, model_->tokenizer(),
                                 model_->VisionEncoder(), max_context_,
                                 image_cache_);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    RequireQwenFlashNextState(state).session().ConfigureVision(
        QwenPrompt(context));
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return QwenChatOptions(request, max_context_).enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    throw std::logic_error(
        "Flash-Next execution slots are owned by the component cache");
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() != prefix.size()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next reused prefix does not match checkpoint");
    }
    qfn.session().ResetDraftPolicy();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (offset != qfn.position()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next prefill has no remaining input");
    }
    const std::size_t remaining = prompt.size() - offset;
    const std::size_t consumed =
        std::min<std::size_t>({max_input_tokens, remaining,
                               remaining <= model_->PrefillThroughCapacity()
                                   ? model_->PrefillThroughCapacity()
                                   : model_->PrefillCapacity()});
    const std::size_t next_position = offset + consumed;
    const auto prefix = QwenFlashNextEngineTokens(prompt.first(next_position));
    // The next step most likely takes the same budget; its n-gram rows are
    // read during this one and used for the tokens it shares. A step ending
    // at a cache frontier continues into the pool's lookahead.
    const auto rest = next_position == prompt.size()
                          ? state.prefill_lookahead()
                          : prompt.subspan(next_position);
    const std::size_t after = rest.size();
    const auto next = QwenFlashNextEngineTokens(rest.subspan(
        0, std::min<std::size_t>({max_input_tokens, after,
                                  after <= model_->PrefillThroughCapacity()
                                      ? model_->PrefillThroughCapacity()
                                      : model_->PrefillCapacity()})));
    std::string error;
    if (!qfn.session().Sync(prefix, &error, next)) {
      qfn.set_position(0);
      throw std::runtime_error("Qwen3.8-Flash-Next prefill failed: " + error);
    }
    qfn.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    const auto consumed = std::min<std::size_t>(
        {budget, prompt.size() - offset, model_->PrefillThroughCapacity()});
    if (boundary <= offset + 1 || boundary >= offset + consumed ||
        offset + consumed - boundary > 8)
      return std::nullopt;
    const auto bytes = qfn.session().PrefillCheckpointBytes(
        static_cast<std::uint32_t>(boundary));
    return bytes ? std::optional<std::size_t>(bytes) : std::nullopt;
  }

  [[nodiscard]] TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint,
      const std::function<std::unique_ptr<TextRunnerSnapshot>()>& capture)
      const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (!checkpoint || !capture || offset != qfn.position())
      throw std::logic_error("invalid component in-pass checkpoint request");
    const auto consumed = std::min<std::size_t>(
        {budget, prompt.size() - offset, model_->PrefillThroughCapacity()});
    const auto next_position = offset + consumed;
    const auto prefix = QwenFlashNextEngineTokens(prompt.first(next_position));
    double capture_ms = 0;
    try {
      qfn.adapter().PrefillThrough(
          qfn.slot(), prefix, static_cast<std::uint32_t>(boundary), [&] {
            const auto started = std::chrono::steady_clock::now();
            try {
              *checkpoint = capture();
            } catch (...) {
              // Optional retention may refuse; the complete live prefill is
              // valid.
              checkpoint->reset();
            }
            capture_ms += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
          });
    } catch (...) {
      qfn.set_position(0);
      throw;
    }
    qfn.set_position(next_position);
    return {.consumed_tokens = consumed,
            .decode_ready = next_position == prompt.size(),
            .checkpoint_ms = capture_ms};
  }

  // The engine stops before committing EOS when the request enables it.
  // Every returned token is committed work and is emitted exactly once.
  void ResolveStepTokens(QwenFlashNextTextRunnerState&, bool engine_stopped,
                         std::span<const std::int32_t> tokens,
                         TextDecodeStep* step) const {
    step->stop = engine_stopped;
    step->selections.reserve(tokens.size());
    for (const std::int32_t token : tokens) {
      step->selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->TokenText(token),
      });
    }
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = qfn.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error(
          "Qwen3.8-Flash-Next token selection has no logits");
    }
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    if (EndsTurn(qfn, *model_, token)) {
      return {.stop = true, .token = 0, .piece = {}};
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->TokenText(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next token ID exceeds engine range");
    }
    auto& qfn = RequireQwenFlashNextState(state);
    std::string error;
    if (!qfn.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error("Qwen3.8-Flash-Next decode failed: " + error);
    }
    qfn.set_position(qfn.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp_ || max_tokens == 1) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next MTP decode budget must be at least one token");
    }
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    const auto stats_before = qfn.session().Statistics();
    sampling::SamplerState working_sampler = sampler;
    QwenFlashNextSession::DecodeResult decoded;
    std::string error;
    const auto budget =
        std::min<std::size_t>(max_tokens, std::uint64_t{max_draft_tokens_} + 1);
    // Use the same request policy as SelectNext and the batched MTP path.
    // The engine stops before committing EOS or work beyond it.
    if (!qfn.session().DecodeStep(budget, working_sampler, &decoded, &error,
                                  qfn.stop_at_eos())) {
      throw std::runtime_error("Qwen3.8-Flash-Next MTP decode failed: " +
                               error);
    }
    // The pool accepts the returned tokens once. Publish the RNG and residual
    // draw so the next batch retains the rejection-conditioned distribution.
    sampler.CopyDrawStateFrom(working_sampler);
    TextDecodeStep step;
    ResolveStepTokens(qfn, decoded.stop, decoded.tokens, &step);
    qfn.set_position(qfn.session().Position());
    const auto stats_after = qfn.session().Statistics();
    step.draft_rounds = stats_after.cycles - stats_before.cycles;
    step.draft_tokens = stats_after.drafted - stats_before.drafted;
    step.draft_accepted_tokens = stats_after.accepted - stats_before.accepted;
    step.lookup_tokens = stats_after.lookup - stats_before.lookup;
    step.lookup_accepted_tokens =
        stats_after.lookup_accepted - stats_before.lookup_accepted;
    return step;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) {
      return SerialAdvanceBatch(advances);
    }
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(advances.size());
    std::vector<QwenFlashNextSession::AdvanceRequest> requests;
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Flash-Next token exceeds engine range");
      }
      requests.push_back(
          {&RequireQwenFlashNextState(advance.state.get()).session(),
           static_cast<std::int32_t>(advance.token),
           &outcomes[requests.size()]});
    }
    std::string error;
    (void)QwenFlashNextSession::EvaluateBatch(requests, &error);
    for (std::size_t i = 0; i < advances.size(); ++i) {
      const auto& advance = advances[i];
      if (!outcomes[i].completed) {
        auto failure = std::make_exception_ptr(std::runtime_error(
            "Flash-Next advance failed: " + outcomes[i].error));
        if (!advance.failure)
          std::rethrow_exception(failure);
        *advance.failure = failure;
        continue;
      }
      auto& state = RequireQwenFlashNextState(advance.state.get());
      state.set_position(state.session().Position());
    }
  }

  // Merged plain-advance + prefill-chunk step (P3): one engine
  // EvaluateBatchWithChunk instead of separate Prefill + AdvanceBatch.
  // Plain advances only (MTP speculative decodes use DecodeBatch and never
  // enter here). Any refusal or failure leaves advances untouched when they
  // did not run, so the caller can run the legacy pair; completed advances
  // are committed exactly like AdvanceBatch and never rerun.
  [[nodiscard]] bool AdvanceBatchWithPrefillChunk(
      std::span<const TextRunnerAdvance> advances,
      std::span<std::exception_ptr> advance_failures,
      TextRunnerState& prefill_state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens,
      TextPrefillStep& chunk_step) const {
    auto& qfn = RequireQwenFlashNextState(prefill_state);
    if (advances.empty() || offset != qfn.position() ||
        offset >= prompt.size() || max_input_tokens == 0)
      throw MixedBatchRefused("mixed batch preconditions unmet");
    const std::size_t remaining = prompt.size() - offset;
    const std::size_t consumed =
        std::min<std::size_t>({max_input_tokens, remaining,
                               remaining <= model_->PrefillThroughCapacity()
                                   ? model_->PrefillThroughCapacity()
                                   : model_->PrefillCapacity()});
    if (consumed == 0 || advances.size() + consumed > 1024)
      throw MixedBatchRefused("mixed batch shape exceeds envelope");
    // EvaluateBatchWithChunk keeps no logits for the chunk, and a chunk that
    // reaches the end of the model prompt (the prompt end or a snapshot /
    // checkpoint frontier cut by the pool) must leave a decodable frontier and
    // keep the legacy chunk partition and in-pass capture. Such a chunk runs on
    // the legacy path; only interior chunks merge.
    if (offset + consumed >= prompt.size())
      throw MixedBatchRefused("mixed batch chunk reaches the prompt frontier");
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(advances.size());
    std::vector<QwenFlashNextSession::AdvanceRequest> requests;
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max()))
        throw MixedBatchRefused("mixed batch token exceeds engine range");
      requests.push_back(
          {&RequireQwenFlashNextState(advance.state.get()).session(),
           static_cast<std::int32_t>(advance.token),
           &outcomes[requests.size()]});
    }
    const auto chunk_ids =
        QwenFlashNextEngineTokens(prompt.subspan(offset, consumed));
    std::string error;
    const bool ok = QwenFlashNextSession::EvaluateBatchWithChunk(
        requests, &qfn.session(), std::span<const std::int32_t>(chunk_ids),
        &error);
    bool attempted = false;
    for (const auto& outcome : outcomes) {
      if (outcome.completed || !outcome.error.empty()) {
        attempted = true;
        break;
      }
    }
    if (!ok && !attempted)
      throw MixedBatchRefused("mixed batch refused cleanly: " + error);
    for (std::size_t i = 0; i < advances.size(); ++i) {
      if (!outcomes[i].completed) {
        advance_failures[i] = std::make_exception_ptr(std::runtime_error(
            "Flash-Next mixed advance failed: " + outcomes[i].error));
        continue;
      }
      advance_failures[i] = nullptr;
      auto& state = RequireQwenFlashNextState(advances[i].state.get());
      state.set_position(state.session().Position());
    }
    if (!ok)
      return false;
    qfn.set_position(offset + consumed);
    chunk_step = {.consumed_tokens = consumed,
                  .decode_ready = (offset + consumed) == prompt.size()};
    return true;
  }

  // Merged MTP decode + prefill chunk: one engine DecodeBatchWithChunk runs
  // every decode's MTP step with the chunk riding the verify pass. Refuses
  // (MixedBatchRefused, nothing mutated) when MTP is off, the shape exceeds
  // the 1024-row envelope, or the engine rejects the inputs. A chunk that
  // fails after the batch ran is reported as an interrupted chunk_step with
  // no consumed tokens, so completed decode steps are still delivered.
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatchWithPrefillChunk(
      std::span<const TextRunnerDecode> decodes,
      TextRunnerState& prefill_state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens,
      TextPrefillStep& chunk_step) const override {
    auto& qfn = RequireQwenFlashNextState(prefill_state);
    if (!use_mtp_ || decodes.empty() || decodes.size() > 8 ||
        offset != qfn.position() || offset >= prompt.size() ||
        max_input_tokens == 0)
      throw MixedBatchRefused("merged MTP decode preconditions unmet");
    const std::size_t remaining = prompt.size() - offset;
    const std::size_t verify_rows =
        decodes.size() * (std::size_t{max_draft_tokens_} + 1);
    if (verify_rows >= 1024)
      throw MixedBatchRefused("merged MTP decode exceeds envelope");
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, remaining, 1024 - verify_rows,
         remaining <= model_->PrefillThroughCapacity()
             ? model_->PrefillThroughCapacity()
             : model_->PrefillCapacity()});
    if (consumed == 0)
      throw MixedBatchRefused("merged MTP decode has no chunk room");
    const auto count = decodes.size();
    std::vector<sampling::SamplerState> samplers;
    std::vector<QwenFlashNextSession::DecodeResult> results(count);
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(count);
    std::vector<QwenFlashNextSession::SpeculativeStats> before;
    std::vector<QwenFlashNextSession::DecodeRequest> requests;
    samplers.reserve(count);
    before.reserve(count);
    requests.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& state = RequireQwenFlashNextState(decodes[i].state.get());
      auto& session = state.session();
      auto& sampler = samplers.emplace_back(decodes[i].sampler.get());
      before.push_back(session.Statistics());
      requests.push_back(
          {&session,
           std::min<std::size_t>(decodes[i].max_tokens,
                                 std::uint64_t{max_draft_tokens_} + 1),
           &sampler, &results[i], state.stop_at_eos(), &outcomes[i]});
    }
    const auto chunk_ids =
        QwenFlashNextEngineTokens(prompt.subspan(offset, consumed));
    QwenFlashNextSession::BatchOutcome chunk_outcome;
    std::string error;
    const bool ok = QwenFlashNextSession::DecodeBatchWithChunk(
        requests, &qfn.session(), std::span<const std::int32_t>(chunk_ids),
        &chunk_outcome, &error);
    bool attempted = chunk_outcome.completed || !chunk_outcome.error.empty();
    for (const auto& outcome : outcomes)
      attempted = attempted || outcome.completed || !outcome.error.empty();
    if (!ok && !attempted)
      throw MixedBatchRefused("merged MTP decode refused: " + error);
    const auto active_count = static_cast<std::size_t>(std::count_if(
        results.begin(), results.end(),
        [](const auto& result) { return !result.tokens.empty(); }));
    std::vector<TextDecodeStep> steps(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (!outcomes[i].completed) {
        steps[i].failure = std::make_exception_ptr(std::runtime_error(
            "Flash-Next merged MTP failed: " + outcomes[i].error));
        continue;
      }
      auto& state = RequireQwenFlashNextState(decodes[i].state.get());
      decodes[i].sampler.get().CopyDrawStateFrom(samplers[i]);
      state.set_position(state.session().Position());
      auto& step = steps[i];
      ResolveStepTokens(state, results[i].stop, results[i].tokens, &step);
      step.execution_plan = {.kind = TextExecutionPlanKind::kBatched,
                             .physical_width = active_count + 1};
      const auto stats = state.session().Statistics();
      step.draft_rounds = stats.cycles - before[i].cycles;
      step.draft_tokens = stats.drafted - before[i].drafted;
      step.draft_accepted_tokens = stats.accepted - before[i].accepted;
      step.lookup_tokens = stats.lookup - before[i].lookup;
      step.lookup_accepted_tokens =
          stats.lookup_accepted - before[i].lookup_accepted;
    }
    if (!chunk_outcome.completed) {
      chunk_step = {.consumed_tokens = 0, .decode_ready = false,
                    .interrupted = true};
      return steps;
    }
    qfn.set_position(offset + consumed);
    chunk_step = {.consumed_tokens = consumed,
                  .decode_ready = (offset + consumed) == prompt.size()};
    return steps;
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || !use_mtp_) {
      return SerialDecodeBatch(decodes);
    }
    const auto count = decodes.size();
    std::vector<sampling::SamplerState> samplers;
    std::vector<QwenFlashNextSession::DecodeResult> results(count);
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(count);
    std::vector<QwenFlashNextSession::SpeculativeStats> before;
    std::vector<QwenFlashNextSession::DecodeRequest> requests;
    samplers.reserve(count);
    before.reserve(count);
    requests.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& state = RequireQwenFlashNextState(decodes[i].state.get());
      auto& session = state.session();
      auto& sampler = samplers.emplace_back(decodes[i].sampler.get());
      before.push_back(session.Statistics());
      requests.push_back(
          {&session,
           std::min<std::size_t>(decodes[i].max_tokens,
                                 std::uint64_t{max_draft_tokens_} + 1),
           &sampler, &results[i], state.stop_at_eos(), &outcomes[i]});
    }
    std::string error;
    (void)QwenFlashNextSession::DecodeBatch(requests, &error);
    const auto active_count = static_cast<std::size_t>(std::count_if(
        results.begin(), results.end(),
        [](const auto& result) { return !result.tokens.empty(); }));
    std::vector<TextDecodeStep> steps(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (!outcomes[i].completed) {
        steps[i].failure = std::make_exception_ptr(
            std::runtime_error("Flash-Next MTP failed: " + outcomes[i].error));
        continue;
      }
      auto& state = RequireQwenFlashNextState(decodes[i].state.get());
      decodes[i].sampler.get().CopyDrawStateFrom(samplers[i]);
      state.set_position(state.session().Position());
      auto& step = steps[i];
      ResolveStepTokens(state, results[i].stop, results[i].tokens, &step);
      if (active_count > 1 && !results[i].tokens.empty()) {
        step.execution_plan = {.kind = TextExecutionPlanKind::kBatched,
                               .physical_width = active_count};
      }
      const auto stats = state.session().Statistics();
      step.draft_rounds = stats.cycles - before[i].cycles;
      step.draft_tokens = stats.drafted - before[i].drafted;
      step.draft_accepted_tokens = stats.accepted - before[i].accepted;
      // A request whose peers dropped out runs a single-session step,
      // which can use prompt lookup.
      step.lookup_tokens = stats.lookup - before[i].lookup;
      step.lookup_accepted_tokens =
          stats.lookup_accepted - before[i].lookup_accepted;
    }
    return steps;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenFlashNextState(state).position();
  }

  [[nodiscard]] TextRunnerNgramRows NgramRows(
      const TextRunnerState& state) const override {
    const auto totals =
        RequireQwenFlashNextState(state).session().NgramRowTotals();
    return {totals.rows, totals.dedup_rows, totals.cache_hits,
            totals.disk_reads};
  }

private:
  mutable models::qwen::vision::ImageCache image_cache_;
  std::shared_ptr<QwenFlashNextModel> model_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t max_draft_tokens_;
  std::string kv_cache_quant_ = "f16";
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
  cache::Identity component_identity_;
};

// ---------------------------------------------------------------------------
// Gemma 4 HTTP backend: plain Sync/Evaluate sessions owned per request slot,
// no MTP sidecar and no continuation snapshots in serve. Prompt framing
// mirrors the GGUF canonical template with enable_thinking=false; tool calls
// round-trip through the native <|tool_call>call:NAME{...}<tool_call|>
// envelope (parsed back in openai_chat.cpp ParseGemmaCalls).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GPT-OSS HTTP backend: plain Sync/Evaluate sessions owned per request slot,
// no MTP/draft sidecar and no continuation snapshots in serve. Prompt framing
// mirrors `mbufo prompt` (harmony channels, same system block); tools are
// listed in the developer message and round-trip through the generic JSON
// <tool_call> envelope (instruction + grammar from text_model_runner.cpp,
// parsed back by ParseQwenCalls' compatibility form). Reasoning channels pass
// through as content: no thinking split, so greedy serve output matches
// `mbufo prompt` token for token.
// ---------------------------------------------------------------------------
constexpr std::string_view kGptOssStateAbi = "gptoss-gfx1151-state-v1";

using GptOssModel = models::gptoss::Model;
using GptOssSession = models::gptoss::Session;

std::vector<std::int32_t> GptOssEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("GPT-OSS token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

std::string GptOssReasoningEffort(const ChatRequest& request) {
  if (!request.reasoning.effort) return "medium";
  switch (*request.reasoning.effort) {
    case ReasoningEffort::kMinimal:
    case ReasoningEffort::kLow:
      return "low";
    case ReasoningEffort::kHigh:
    case ReasoningEffort::kXHigh:
    case ReasoningEffort::kMax:
      return "high";
    case ReasoningEffort::kMedium:
      return "medium";
  }
  return "medium";
}

std::string GptOssCurrentDate() {
  const std::time_t now = std::time(nullptr);
  std::tm date{};
#if defined(_WIN32)
  localtime_s(&date, &now);
#else
  localtime_r(&now, &date);
#endif
  std::string out = std::to_string(date.tm_year + 1900) + "-";
  const int mon = date.tm_mon + 1;
  const int mday = date.tm_mday;
  if (mon < 10) out += "0";
  out += std::to_string(mon) + "-";
  if (mday < 10) out += "0";
  out += std::to_string(mday);
  return out;
}

// Compact JSON tool listing for the developer message. The exact call schema
// stays enforced by the kJson grammar; this block tells the model which
// functions exist and what they take.
std::string GptOssToolDefs(std::span<const tokenization::ChatTool> tools) {
  std::string out =
      "# Tools\n\nWhen you need a tool, output ONLY the call on its own "
      "line, with arguments as a JSON object:\n"
      "<tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}"
      "</tool_call>\n"
      "Example: <tool_call>{\"name\": \"get_weather\", \"arguments\": "
      "{\"location\": \"Paris\"}}</tool_call>\n"
      "Available functions:\n";
  for (const auto& tool : tools) {
    out += "- " + tool.name + ": " + tool.description + "\n  Parameters: ";
    const json::Value params =
        tool.parameters_json.empty()
            ? json::Value::object()
            : json::parse(tool.parameters_json);
    out += params.dump() + "\n";
  }
  return out;
}

std::string GptOssArgsJson(
    std::span<const tokenization::ChatMessage::ToolArgument> args) {
  json::Value object = json::Value::object();
  for (const auto& arg : args) {
    if (arg.is_string) {
      object[arg.name] = arg.value;
    } else {
      try {
        object[arg.name] = json::parse(arg.value);
      } catch (const std::exception&) {
        object[arg.name] = arg.value;
      }
    }
  }
  return object.dump();
}

std::string GptOssRenderPrompt(const ChatRequest& request) {
  const auto tools =
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools};
  const auto& messages = request.messages;
  std::string developer_text;
  std::size_t body_begin = 0;
  if (!messages.empty() &&
      (messages.front().role == tokenization::ChatRole::kSystem ||
       messages.front().role == tokenization::ChatRole::kDeveloper)) {
    developer_text =
        messages.front().content + messages.front().framing_suffix;
    body_begin = 1;
  }
  std::string out = "<|start|>system<|message|>";
  out += "You are ChatGPT, a large language model trained by OpenAI.\n";
  out += "Knowledge cutoff: 2024-06\n";
  out += "Current date: ";
  out += GptOssCurrentDate();
  out += "\n\nReasoning: ";
  out += GptOssReasoningEffort(request);
  out += "\n\n# Valid channels: analysis, commentary, final. "
         "Channel must be included for every message.<|end|>";
  if (!developer_text.empty() || !tools.empty()) {
    out += "<|start|>developer<|message|>";
    if (!developer_text.empty()) {
      out += "# Instructions\n\n";
      out += developer_text;
    }
    if (!tools.empty()) {
      if (!developer_text.empty()) out += "\n\n";
      out += GptOssToolDefs(tools);
    }
    out += "<|end|>";
  }
  for (std::size_t i = body_begin; i < messages.size(); ++i) {
    const auto& message = messages[i];
    if (message.role == tokenization::ChatRole::kTool) {
      std::string tool_name = message.name;
      if (tool_name.empty() && !message.tool_call_id.empty()) {
        for (std::size_t j = i; j-- > 0;) {
          if (messages[j].role != tokenization::ChatRole::kAssistant)
            continue;
          for (const auto& call : messages[j].tool_calls) {
            if (call.id == message.tool_call_id)
              tool_name = call.name;
          }
        }
        if (tool_name.empty())
          tool_name = message.tool_call_id;
      }
      if (tool_name.empty())
        tool_name = "unknown";
      out += "<|start|>functions." + tool_name +
             " to=assistant<|channel|>commentary<|message|>" +
             message.content + "<|end|>";
      continue;
    }
    if (message.role == tokenization::ChatRole::kAssistant) {
      out += "<|start|>assistant<|message|>";
      if (!message.thought.empty()) {
        out += "<|channel|>analysis<|message|>" + message.thought;
      }
      for (const auto& call : message.tool_calls) {
        out += "<|channel|>commentary to=functions." + call.name +
               "<|constrain|>json<|message|>" +
               GptOssArgsJson(call.arguments) + "<|call|>";
      }
      if (!message.content.empty()) {
        out += "<|channel|>final<|message|>" + message.content;
      }
      out += "<|end|>";
      continue;
    }
    out += "<|start|>user<|message|>" + message.content + "<|end|>";
  }
  out += "<|start|>assistant";
  return out;
}

class GptOssTextRunnerState final : public TextRunnerState {
 public:
  explicit GptOssTextRunnerState(const std::shared_ptr<GptOssModel>& model) {
    std::string error;
    session_ = model->CreateSession(&error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create GPT-OSS session: " + error);
    }
  }

  void SetCancellationCheck(const CancellationCheck&) override {}
  void Invalidate() noexcept override { position_ = 0; }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {};
  }

  [[nodiscard]] GptOssSession& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

 private:
  std::unique_ptr<GptOssSession> session_;
  std::size_t position_{0};
};

GptOssTextRunnerState& RequireGptOssState(TextRunnerState& state) {
  auto* gptoss = dynamic_cast<GptOssTextRunnerState*>(&state);
  if (gptoss == nullptr) {
    throw std::logic_error("text runner state is not GPT-OSS");
  }
  return *gptoss;
}

const GptOssTextRunnerState& RequireGptOssState(const TextRunnerState& state) {
  const auto* gptoss = dynamic_cast<const GptOssTextRunnerState*>(&state);
  if (gptoss == nullptr) {
    throw std::logic_error("text runner state is not GPT-OSS");
  }
  return *gptoss;
}

class GptOssTextRunner final : public HipTextModelRunner {
 public:
  // Batch contract (see TextModelRunner): serial DecodeBatch (no multi-token
  // decode); native AdvanceBatch and AdvanceBatchWithPrefillChunk (one
  // layer-synchronous engine pass, every lane on its own single-token decode
  // core, bit-identical to serial); DecodeBatchWithPrefillChunk refuses.
  MBUFO_RUNNER_SERIAL_DECODE_BATCH();

  // Plain advance of every lane in one layer-synchronous engine pass. The
  // engine validates before touching state, so a refusal reruns serially.
  void AdvanceBatch(std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) return SerialAdvanceBatch(advances);
    std::vector<GptOssSession*> lanes;
    std::vector<std::int32_t> tokens;
    lanes.reserve(advances.size());
    tokens.reserve(advances.size());
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max()))
        return SerialAdvanceBatch(advances);  // serial path reports the lane error
      lanes.push_back(&RequireGptOssState(advance.state.get()).session());
      tokens.push_back(static_cast<std::int32_t>(advance.token));
    }
    std::string error;
    if (!GptOssSession::EvaluateBatch(lanes, tokens, &error))
      return SerialAdvanceBatch(advances);
    for (const auto& advance : advances) {
      auto& lane = RequireGptOssState(advance.state.get());
      lane.set_position(lane.position() + 1);
    }
  }

  // Merged plain-advance + prefill chunk. Output equals Prefill (Sync) of the
  // chunk followed by the serial Advance of every lane. The chunk stays
  // inside one engine prefill block (Sync cuts blocks at multiples of the
  // chunk size from 0), so it may consume fewer tokens than budgeted. An
  // engine refusal happens before any state is touched.
  [[nodiscard]] bool AdvanceBatchWithPrefillChunk(
      std::span<const TextRunnerAdvance> advances,
      std::span<std::exception_ptr> advance_failures,
      TextRunnerState& prefill_state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens,
      TextPrefillStep& chunk_step) const override {
    auto& chunk = RequireGptOssState(prefill_state);
    if (advances.empty() || offset != chunk.position() ||
        offset >= prompt.size() || max_input_tokens == 0)
      throw MixedBatchRefused("gptoss mixed batch preconditions unmet");
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset,
         GptOssSession::MixedChunkEnd(offset) - offset});
    const std::size_t next_position = offset + consumed;
    if (next_position > max_context_)
      throw MixedBatchRefused("gptoss mixed batch prompt exceeds context");
    std::vector<GptOssSession*> lanes;
    std::vector<std::int32_t> tokens;
    lanes.reserve(advances.size());
    tokens.reserve(advances.size());
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max()))
        throw MixedBatchRefused("gptoss mixed batch token exceeds range");
      auto& lane = RequireGptOssState(advance.state.get());
      if (&lane == &chunk)
        throw MixedBatchRefused("gptoss mixed batch lane is the chunk state");
      lanes.push_back(&lane.session());
      tokens.push_back(static_cast<std::int32_t>(advance.token));
    }
    const auto prefix = GptOssEngineTokens(prompt.first(next_position));
    std::string error;
    if (!GptOssSession::EvaluateBatchWithChunk(lanes, tokens, &chunk.session(),
                                               prefix, offset, &error)) {
      throw MixedBatchRefused("gptoss mixed batch refused cleanly: " + error);
    }
    for (std::size_t i = 0; i < advances.size(); ++i) {
      if (i < advance_failures.size()) advance_failures[i] = nullptr;
      auto& lane = RequireGptOssState(advances[i].state.get());
      lane.set_position(lane.position() + 1);
    }
    chunk.set_position(next_position);
    chunk_step = {.consumed_tokens = consumed,
                  .decode_ready = next_position == prompt.size()};
    static const bool trace = std::getenv("MBUFO_GPTOSS_MIXED_TRACE") != nullptr;
    if (trace)
      std::fprintf(stderr, "[gptoss-mixed] lanes=%zu chunk=%zu offset=%zu\n",
                   advances.size(), consumed, offset);
    return true;
  }

  // GPT-OSS has no multi-token decode, so there is no verify pass to ride.
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatchWithPrefillChunk(
      std::span<const TextRunnerDecode>, TextRunnerState&,
      std::span<const TextRunnerToken>, std::size_t, std::size_t,
      TextPrefillStep&) const override {
    RefuseMixedBatch("gptoss has no merged multi-token decode");
  }
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kJson;
  }

  GptOssTextRunner(std::shared_ptr<GptOssModel> model,
                   std::uint32_t max_context)
      : model_(std::move(model)), max_context_(max_context) {
    // Harmony generation ends at <|return|>; the per-channel <|end|> never
    // stops output on its own.
    for (const char* marker : {"<|return|>"}) {
      if (const auto id = model_->tokenizer().FindSpecialToken(marker)) {
        turn_stops_.push_back(static_cast<std::int32_t>(*id));
      }
    }
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kGptOssStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = false,
                .fork = false,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = false,
                .batched_multi_token_decode = false,
                .batched_multi_token_decode_max_width = 0,
                .prefix_reuse = false,
                .mixed_prefill_decode = true,
            },
        .persistence = std::nullopt,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          const std::int32_t token = static_cast<std::int32_t>(id);
          const std::int32_t unit[1] = {token};
          return sampling::ConstraintVocabulary::Piece{
              model_->Decode(std::span<const std::int32_t>(unit, 1)),
              model_->IsStopToken(token)};
        });
  }

  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return model_->VocabSize();
  }

  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"", "", {}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    std::vector<TextRunnerToken> tokens;
    for (const std::int32_t id : model_->Tokenize(text, /*add_bos=*/true)) {
      tokens.push_back(static_cast<TextRunnerToken>(id));
    }
    return tokens;
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    const std::string prompt = GptOssRenderPrompt(request);
    if (prompt.empty())
      return std::nullopt;
    return Tokenize(prompt);
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest&) const override {
    return TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    const auto ids = GptOssEngineTokens(tokens);
    return model_->Decode(ids);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<GptOssTextRunnerState>(model_);
  }

  [[nodiscard]] bool IsTurnStopToken(std::int32_t token) const noexcept {
    if (model_->IsStopToken(token))
      return true;
    return std::ranges::find(turn_stops_, token) != turn_stops_.end();
  }

  [[nodiscard]] std::string DecodeToken(std::int32_t token) const {
    return model_->Decode(std::span<const std::int32_t>(&token, 1));
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& gptoss = RequireGptOssState(state);
    if (offset != gptoss.position()) {
      throw std::logic_error(
          "GPT-OSS prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("GPT-OSS prefill has no remaining input");
    }
    const std::size_t consumed =
        std::min(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    if (next_position > max_context_) {
      throw std::runtime_error("GPT-OSS prompt exceeds the served context");
    }
    const auto prefix = GptOssEngineTokens(prompt.first(next_position));
    std::string error;
    if (!gptoss.session().Sync(prefix, &error)) {
      gptoss.set_position(0);
      throw std::runtime_error("GPT-OSS prefill failed: " + error);
    }
    gptoss.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& gptoss = RequireGptOssState(state);
    if (gptoss.position() >= max_context_)
      return {.stop = true, .piece = {}};
    const auto logits = gptoss.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error("GPT-OSS token selection has no logits");
    }
    const auto token =
        static_cast<std::int32_t>(sampler.Sample(logits));
    if (gptoss.stop_at_eos() && IsTurnStopToken(token)) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = DecodeToken(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("GPT-OSS token ID exceeds engine range");
    }
    auto& gptoss = RequireGptOssState(state);
    std::string error;
    if (!gptoss.session().Evaluate(static_cast<std::int32_t>(token),
                                    &error)) {
      throw std::runtime_error("GPT-OSS decode failed: " + error);
    }
    gptoss.set_position(gptoss.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireGptOssState(state).position();
  }

 private:
  std::shared_ptr<GptOssModel> model_;
  std::uint32_t max_context_{0};
  std::vector<std::int32_t> turn_stops_;
};

// ---------------------------------------------------------------------------
// Llama HTTP backend: plain Sync/Evaluate sessions owned per request slot,
// no speculative sidecar and no continuation snapshots in serve (dense GQA,
// d=128: attention and GEMM ride the dispatcher's existing paths for q8 KV,
// local dense F16 kernel otherwise). Prompt framing mirrors `mbufo prompt`
// (Llama-3.1 headers); tools are listed in the system message and round-trip
// through the generic JSON <tool_call> envelope, as for GPT-OSS above.
// ---------------------------------------------------------------------------
constexpr std::string_view kLlamaStateAbi = "llama-gfx1151-state-v1";

using LlamaModel = models::llama::Model;
using LlamaSession = models::llama::Session;
using LagunaModel = models::laguna::Model;
using LagunaSession = models::laguna::Session;
constexpr std::string_view kLagunaStateAbi = "mbufo.laguna.state.v1";
using InklingModel = models::inkling::Model;
using InklingSession = models::inkling::Session;
constexpr std::string_view kInklingStateAbi = "mbufo.inkling.state.v1";

std::vector<std::int32_t> LlamaEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Llama token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

std::string LlamaToolDefs(std::span<const tokenization::ChatTool> tools) {
  // Compact by necessity: the Llama prefill kernels fault past ~130 prompt
  // tokens on q8 (~90 on f16; pre-existing engine limit, prompt path
  // included). Name+parameters only; descriptions and wrappers are dropped.
  std::string out;
  for (const auto& tool : tools) {
    out += "{\"name\": " + json::Value(tool.name).dump();
    const json::Value params =
        tool.parameters_json.empty()
            ? json::Value::object()
            : json::parse(tool.parameters_json);
    out += ", \"parameters\": " + params.dump() + "}\n";
  }
  return out;
}

std::string LlamaArgsJson(
    std::span<const tokenization::ChatMessage::ToolArgument> args) {
  json::Value object = json::Value::object();
  for (const auto& arg : args) {
    if (arg.is_string) {
      object[arg.name] = arg.value;
    } else {
      try {
        object[arg.name] = json::parse(arg.value);
      } catch (const std::exception&) {
        object[arg.name] = arg.value;
      }
    }
  }
  return object.dump();
}

std::string LlamaRenderPrompt(const ChatRequest& request) {
  const auto tools =
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools};
  const auto& messages = request.messages;
  std::string system_text;
  std::size_t body_begin = 0;
  if (!messages.empty() &&
      (messages.front().role == tokenization::ChatRole::kSystem ||
       messages.front().role == tokenization::ChatRole::kDeveloper)) {
    system_text = messages.front().content + messages.front().framing_suffix;
    body_begin = 1;
  }
  std::string out = "<|begin_of_text|>";
  // No system block without system text: the tools ride in the first user
  // message, and an Environment: ipython line without builtin tools pulls
  // the model into the ipython turn-taking mode by mistake.
  if (!system_text.empty()) {
    out += "<|start_header_id|>system<|end_header_id|>\n\n";
    out += system_text;
    out += "<|eot_id|>";
  }
  bool tools_prepended = tools.empty();
  for (std::size_t i = body_begin; i < messages.size(); ++i) {
    const auto& message = messages[i];
    if (message.role == tokenization::ChatRole::kTool) {
      out += "<|start_header_id|>ipython<|end_header_id|>\n\n";
      out += message.content;
      out += "<|eot_id|>";
      continue;
    }
    if (message.role == tokenization::ChatRole::kAssistant) {
      out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
      if (!message.tool_calls.empty()) {
        // Canonical custom-tool history: the JSON call only.
        bool first_call = true;
        for (const auto& call : message.tool_calls) {
          if (!first_call) out += "\n";
          first_call = false;
          out += "{\"name\": " + json::Value(call.name).dump();
          out += ", \"parameters\": " + LlamaArgsJson(call.arguments) + "}";
        }
      } else {
        if (!message.thought.empty()) {
          out += message.thought;
        }
        out += message.content;
      }
      out += "<|eot_id|>";
      continue;
    }
    out += "<|start_header_id|>user<|end_header_id|>\n\n";
    if (!tools_prepended) {
      tools_prepended = true;
      out += "Given these functions, answer with a JSON call "
             "{\"name\": NAME, \"parameters\": ARGS}:\n";
      out += LlamaToolDefs(tools);
      out += "\n";
    }
    out += message.content;
    out += "<|eot_id|>";
  }
  if (!tools_prepended) {
    // No user turn carries the tool block; attach it before generation.
    out += "<|start_header_id|>user<|end_header_id|>\n\n";
    out += "Given these functions, answer with a JSON call "
           "{\"name\": NAME, \"parameters\": ARGS}:\n";
    out += LlamaToolDefs(tools);
    out += "<|eot_id|>";
  }
  out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  return out;
}

std::string LagunaRenderPrompt(const ChatRequest& request) {
  std::string out;
  for (const auto& msg : request.messages) {
    if (msg.role == tokenization::ChatRole::kSystem) {
      out += "<system>\n" + msg.content + "\n</system>\n";
    } else if (msg.role == tokenization::ChatRole::kUser) {
      out += "<user>\n" + msg.content + "\n</user>\n";
    } else if (msg.role == tokenization::ChatRole::kAssistant) {
      out += "<assistant>\n" + msg.content + "</assistant>\n";
    } else if (msg.role == tokenization::ChatRole::kTool) {
      out += "<tool_response>\n" + msg.content + "\n</tool_response>\n";
    }
  }
  out += "<assistant>\n";
  return out;
}

class LagunaTextRunnerState final : public TextRunnerState {
 public:
  explicit LagunaTextRunnerState(const std::shared_ptr<LagunaModel>& model) {
    std::string error;
    session_ = model->CreateSession(&error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create Laguna session: " + error);
    }
  }
  void SetCancellationCheck(const CancellationCheck&) override {}
  void Invalidate() noexcept override { position_ = 0; }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources() const noexcept override { return {}; }
  [[nodiscard]] LagunaSession& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }
 private:
  std::unique_ptr<LagunaSession> session_;
  std::size_t position_{0};
};

class LagunaTextRunner final : public HipTextModelRunner {
 public:
  // Batch contract (see TextModelRunner): DecodeBatch is serial (no multi-token decode); AdvanceBatch and the
  // merged plain-advance + prefill chunk are native. Native means one in-order engine pass with a single host
  // sync in which every lane keeps the exact single-session decode core and the chunk keeps the prefill core
  // (bit-identical to Evaluate / Sync; a fused weight-sharing GEMM is not). The multi-token merged entry point
  // is an explicit refusal.
  MBUFO_RUNNER_SERIAL_DECODE_BATCH();
  static constexpr std::size_t kMaxBatchLanes = 16;

  void AdvanceBatch(std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2 || advances.size() > kMaxBatchLanes) return SerialAdvanceBatch(advances);
    std::vector<LagunaSession*> lanes;
    std::vector<std::int32_t> tokens;
    lanes.reserve(advances.size());
    tokens.reserve(advances.size());
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max()))
        return SerialAdvanceBatch(advances);  // the serial path reports the per-lane error
      lanes.push_back(&dynamic_cast<LagunaTextRunnerState&>(advance.state.get()).session());
      tokens.push_back(static_cast<std::int32_t>(advance.token));
    }
    std::string error;
    bool refused = false;
    if (!LagunaSession::AdvanceBatchWithChunk(lanes, tokens, nullptr, {}, 0, &error, &refused)) {
      if (refused) return SerialAdvanceBatch(advances);  // validated before any state moved
      const auto failure = std::make_exception_ptr(std::runtime_error("Laguna batch decode failed: " + error));
      for (const auto& advance : advances) {
        if (!advance.failure) std::rethrow_exception(failure);
        *advance.failure = failure;
      }
      return;
    }
    for (const auto& advance : advances) {
      auto& st = dynamic_cast<LagunaTextRunnerState&>(advance.state.get());
      st.set_position(st.position() + 1);
    }
  }

  [[nodiscard]] bool AdvanceBatchWithPrefillChunk(
      std::span<const TextRunnerAdvance> advances, std::span<std::exception_ptr> advance_failures,
      TextRunnerState& prefill_state, std::span<const TextRunnerToken> prompt, std::size_t offset,
      std::size_t max_input_tokens, TextPrefillStep& chunk_step) const override {
    auto& chunk = dynamic_cast<LagunaTextRunnerState&>(prefill_state);
    if (advances.empty() || advances.size() > kMaxBatchLanes || offset != chunk.position() ||
        offset >= prompt.size() || max_input_tokens == 0)
      throw MixedBatchRefused("laguna mixed batch preconditions unmet");
    const std::size_t consumed =
        std::min<std::size_t>({max_input_tokens, prompt.size() - offset, LagunaSession::MaxChunkTokens()});
    const std::size_t next_position = offset + consumed;
    if (next_position > max_context_) throw MixedBatchRefused("laguna mixed batch prompt exceeds context");
    std::vector<LagunaSession*> lanes;
    std::vector<std::int32_t> tokens;
    lanes.reserve(advances.size());
    tokens.reserve(advances.size());
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max()))
        throw MixedBatchRefused("laguna mixed batch token exceeds range");
      auto& lane = dynamic_cast<LagunaTextRunnerState&>(advance.state.get());
      if (&lane == &chunk) throw MixedBatchRefused("laguna mixed batch lane is the chunk state");
      lanes.push_back(&lane.session());
      tokens.push_back(static_cast<std::int32_t>(advance.token));
    }
    std::vector<std::int32_t> prefix;
    prefix.reserve(next_position);
    for (std::size_t i = 0; i < next_position; ++i) prefix.push_back(static_cast<std::int32_t>(prompt[i]));
    std::string error;
    bool refused = false;
    if (!LagunaSession::AdvanceBatchWithChunk(lanes, tokens, &chunk.session(), prefix, offset, &error, &refused)) {
      if (refused) throw MixedBatchRefused("laguna mixed batch refused cleanly: " + error);
      // Mid-pass: no lane or chunk can be trusted; mirror a failed Prefill / Advance.
      chunk.set_position(0);
      const auto failure = std::make_exception_ptr(std::runtime_error("Laguna mixed advance failed: " + error));
      for (std::size_t i = 0; i < advances.size(); ++i) {
        if (i < advance_failures.size()) advance_failures[i] = failure;
      }
      return false;
    }
    for (std::size_t i = 0; i < advances.size(); ++i) {
      if (i < advance_failures.size()) advance_failures[i] = nullptr;
      auto& lane = dynamic_cast<LagunaTextRunnerState&>(advances[i].state.get());
      lane.set_position(lane.position() + 1);
    }
    chunk.set_position(next_position);
    chunk_step = {.consumed_tokens = consumed, .decode_ready = next_position == prompt.size()};
    static const bool trace = std::getenv("MBUFO_LAGUNA_MIXED_TRACE") != nullptr;
    if (trace) std::fprintf(stderr, "[laguna-mixed] lanes=%zu chunk=%zu offset=%zu\n", advances.size(), consumed, offset);
    return true;
  }

  // Laguna has no multi-token decode, so there is no verify pass a chunk could ride: explicit refusal.
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatchWithPrefillChunk(
      std::span<const TextRunnerDecode>, TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, TextPrefillStep&) const override {
    RefuseMixedBatch("laguna has no merged multi-token decode");
  }
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
  LagunaTextRunner(std::shared_ptr<LagunaModel> model, std::uint32_t max_context)
      : model_(std::move(model)), max_context_(max_context) {
    for (const char* marker : {"</assistant>", "\u3008|EOS|\u3009"}) {
      if (const auto id = model_->tokenizer().FindSpecialToken(marker)) {
        turn_stops_.push_back(static_cast<std::int32_t>(*id));
      }
    }
  }
  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kLagunaStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = false,
                .fork = false,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = false,
                .batched_multi_token_decode = false,
                .batched_multi_token_decode_max_width = 0,
                .prefix_reuse = false,
                .mixed_prefill_decode = true,
            },
        .persistence = std::nullopt,
    };
  }
  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0, total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) capacity = free_bytes;
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }
  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    // Advertise batched widths so the scheduler routes plain decodes through the native AdvanceBatch
    // (DecodeBatch itself stays serial).
    std::vector<TextExecutionPlan> plans{{.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back({.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }
  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary> BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          const std::int32_t token = static_cast<std::int32_t>(id);
          const std::int32_t unit[1] = {token};
          return sampling::ConstraintVocabulary::Piece{
              model_->Decode(std::span<const std::int32_t>(unit, 1)),
              model_->IsStopToken(token)};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override { return model_->VocabSize(); }
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override { return {"<think>", "</think>", {}}; }
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(std::string_view text) const override {
    std::vector<TextRunnerToken> tokens;
    for (const std::int32_t id : model_->Tokenize(text, /*add_bos=*/true)) {
      tokens.push_back(static_cast<TextRunnerToken>(id));
    }
    return tokens;
  }
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(const ChatRequest& request) const override {
    const std::string prompt = LagunaRenderPrompt(request);
    if (prompt.empty()) return std::nullopt;
    return Tokenize(prompt);
  }
  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(const ChatRequest&) const override {
    return TextGenerationBackend::InitialOutputState::kContent;
  }
  [[nodiscard]] std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    std::vector<std::int32_t> ids;
    ids.reserve(tokens.size());
    for (const auto t : tokens) ids.push_back(static_cast<std::int32_t>(t));
    return model_->Decode(ids);
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<LagunaTextRunnerState>(model_);
  }
  [[nodiscard]] bool IsTurnStopToken(std::int32_t token) const noexcept {
    if (model_->IsStopToken(token)) return true;
    return std::ranges::find(turn_stops_, token) != turn_stops_.end();
  }
  [[nodiscard]] TextPrefillStep Prefill(TextRunnerState& state,
                                        std::span<const TextRunnerToken> prompt,
                                        std::size_t offset,
                                        std::size_t max_input_tokens) const override {
    auto& st = dynamic_cast<LagunaTextRunnerState&>(state);
    if (offset != st.position()) {
      throw std::logic_error("Laguna prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Laguna prefill has no remaining input");
    }
    const std::size_t consumed = std::min(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    if (next_position > max_context_) {
      throw std::runtime_error("Laguna prompt exceeds the served context");
    }
    std::vector<std::int32_t> prefix;
    prefix.reserve(next_position);
    for (std::size_t i = 0; i < next_position; ++i) {
      prefix.push_back(static_cast<std::int32_t>(prompt[i]));
    }
    std::string err;
    if (!st.session().Sync(prefix, &err)) {
      st.set_position(0);
      throw std::runtime_error("Laguna prefill failed: " + err);
    }
    st.set_position(next_position);
    return {.consumed_tokens = consumed, .decode_ready = next_position == prompt.size()};
  }
  [[nodiscard]] TextDecodeSelection SelectNext(TextRunnerState& state,
                                               sampling::SamplerState& sampler) const override {
    auto& st = dynamic_cast<LagunaTextRunnerState&>(state);
    if (st.position() >= max_context_) return {.stop = true, .piece = {}};
    const auto logits = st.session().Logits();
    if (logits.empty()) throw std::runtime_error("Laguna token selection has no logits");
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    if (st.stop_at_eos() && IsTurnStopToken(token)) {
      return {.stop = true, .token = 0, .piece = {}};
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->Decode(std::span<const std::int32_t>(&token, 1)),
    };
  }
  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& st = dynamic_cast<LagunaTextRunnerState&>(state);
    std::string err;
    if (!st.session().Evaluate(static_cast<std::int32_t>(token), &err)) {
      throw std::runtime_error("Laguna decode failed: " + err);
    }
    st.set_position(st.position() + 1);
  }
  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }
  [[nodiscard]] std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    return dynamic_cast<const LagunaTextRunnerState&>(state).position();
  }
 private:
  std::shared_ptr<LagunaModel> model_;
  std::uint32_t max_context_{0};
  std::vector<std::int32_t> turn_stops_;
};

// ---------------------------------------------------------------------------
// Inkling-2 chat template (compiled port).
//
// The model ships its reference Jinja template as chat_template.jinja next to
// the weights (also embedded as GGUF `tokenizer.chat_template` metadata). The
// repo has no Jinja runtime - qwen/deepseek ports compile their templates the
// same way - so this is a faithful C++ port of that file:
//   * optional tool block first:
//     `<|message_system|>tool_declare<|content_xml|>{specs}<|end_message|>`
//   * `Thinking effort level: <n>` system block before the first non-system
//     message (level from request.reasoning.effort, default 0.9 = high),
//   * per-message `<|message_*|><|content_*|>...<|end_message|>` blocks with
//     the template's tool/assistant name-suffix and tool-call rules,
//   * `<|content_model_end_sampling|>` after each assistant message,
//   * no trailing generation prompt: the model emits `<|message_model|>`
//     itself as its first generated token.
// Thinking is always on for inkling (the template has no enable_thinking
// switch), so InitialOutputState stays kContent and thinking=off is not
// forced off here.
// ---------------------------------------------------------------------------
std::string InklingEffortLevelText(const ReasoningOptions& reasoning) {
  // Template effort_map: none 0.0, minimal 0.1, low 0.2, medium 0.7,
  // high 0.9, xhigh 0.99, max 0.99; undefined reasoning_effort means 0.9.
  // mbufo has no kNone effort ("none"/"off" only flips enabled, which this
  // template cannot express), so unset renders the 0.9 default.
  if (reasoning.enabled.has_value() && !*reasoning.enabled) return "0";
  if (!reasoning.effort.has_value()) return "0.9";
  switch (*reasoning.effort) {
    case ReasoningEffort::kMinimal:
      return "0.1";
    case ReasoningEffort::kLow:
      return "0.2";
    case ReasoningEffort::kMedium:
      return "0.7";
    case ReasoningEffort::kHigh:
      return "0.9";
    case ReasoningEffort::kXHigh:
    case ReasoningEffort::kMax:
      return "0.99";
  }
  return "0.9";
}

json::Value InklingSortedJson(const json::Value& value) {
  // Jinja tojson(sort_keys=True) spelling at every level; arrays keep order.
  if (value.is_object()) {
    auto members = value.members();
    std::sort(members.begin(), members.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    json::Value out = json::Value::object();
    for (auto& kv : members) {
      out.append_member(kv.first, InklingSortedJson(kv.second));
    }
    return out;
  }
  if (value.is_array()) {
    json::Value out = json::Value::array();
    for (const auto& item : value.items()) {
      out.push_back(InklingSortedJson(item));
    }
    return out;
  }
  return value;
}

json::Value InklingToolSpec(const tokenization::ChatTool& tool) {
  // fn = tool.function ?? tool; type defaults to "function". Keys are emitted
  // in sorted order with compact separators, matching tojson(sort_keys=True).
  json::Value params = json::Value::object();
  if (!tool.parameters_json.empty()) {
    try {
      params = InklingSortedJson(json::parse(tool.parameters_json));
      if (!params.is_object()) params = json::Value::object();
    } catch (const std::exception&) {
      params = json::Value::object();
    }
  }
  json::Value spec = json::Value::object();
  spec.append_member("description", json::Value(tool.description));
  spec.append_member("name", json::Value(tool.name));
  spec.append_member("parameters", std::move(params));
  spec.append_member("type", json::Value("function"));
  return spec;
}

std::string InklingToolCallJson(
    const tokenization::ChatMessage::ToolCall& call) {
  // `{"name":<name>,"args":{<keys sorted>}}`, values as canonical JSON.
  std::vector<tokenization::ChatMessage::ToolArgument> args(
      call.arguments.begin(), call.arguments.end());
  std::sort(args.begin(), args.end(),
            [](const auto& a, const auto& b) { return a.name < b.name; });
  json::Value args_json = json::Value::object();
  for (const auto& arg : args) {
    if (arg.is_string) {
      args_json.append_member(arg.name, json::Value(arg.value));
    } else {
      try {
        args_json.append_member(arg.name, json::parse(arg.value));
      } catch (const std::exception&) {
        args_json.append_member(arg.name, json::Value(arg.value));
      }
    }
  }
  json::Value call_json = json::Value::object();
  call_json.append_member("name", json::Value(call.name));
  call_json.append_member("args", std::move(args_json));
  return call_json.dump();
}

std::optional<std::string> InklingRenderPrompt(const ChatRequest& request) {
  const auto tools =
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools};
  std::string out;
  if (!tools.empty()) {
    json::Value specs = json::Value::array();
    for (const auto& tool : tools) {
      specs.push_back(InklingToolSpec(tool));
    }
    out += "<|message_system|>tool_declare<|content_xml|>";
    out += specs.dump();
    out += "<|end_message|>";
  }
  const auto& messages = request.messages;
  const std::string effort_block =
      std::string("<|message_system|><|content_text|>Thinking effort level: ") +
      InklingEffortLevelText(request.reasoning) + "<|end_message|>";
  bool effort_emitted = false;
  for (const auto& message : messages) {
    const bool is_system = message.role == tokenization::ChatRole::kSystem ||
                           message.role == tokenization::ChatRole::kDeveloper;
    if (!effort_emitted && !is_system) {
      out += effort_block;
      effort_emitted = true;
    }
    std::string header;
    switch (message.role) {
      case tokenization::ChatRole::kSystem:
      case tokenization::ChatRole::kDeveloper:
        header = "<|message_system|>";
        break;
      case tokenization::ChatRole::kUser:
        header = "<|message_user|>";
        break;
      case tokenization::ChatRole::kAssistant:
        header = "<|message_model|>";
        break;
      case tokenization::ChatRole::kTool:
        header = "<|message_tool|>";
        break;
    }
    if (message.role == tokenization::ChatRole::kTool) {
      if (!message.name.empty()) {
        header += message.name;
      } else if (!message.tool_call_id.empty()) {
        for (const auto& prev : messages) {
          if (prev.role != tokenization::ChatRole::kAssistant) continue;
          for (const auto& prev_call : prev.tool_calls) {
            if (prev_call.id == message.tool_call_id &&
                !prev_call.name.empty()) {
              header += prev_call.name;
            }
          }
        }
      }
    } else if (message.role == tokenization::ChatRole::kAssistant &&
               !message.name.empty()) {
      header += message.name;
    }
    const bool is_assistant =
        message.role == tokenization::ChatRole::kAssistant;
    if (is_assistant && !message.thought.empty()) {
      out +=
          header + "<|content_thinking|>" + message.thought + "<|end_message|>";
    }
    for (std::size_t img_i = 0; img_i < message.images.size(); ++img_i) {
      out += header + "<|content_image|>";
      for (std::uint32_t k = 0;
           k < models::inkling::VisionEncoder::kMergedTokens; ++k) {
        out += "<|unused_200054|>";
      }
      out += "<|end_message|>";
    }
    const std::string content =
        is_system ? message.content + message.framing_suffix : message.content;
    if (!content.empty() ||
        (message.images.empty() &&
         !(is_assistant &&
           (!message.tool_calls.empty() || !message.thought.empty())))) {
      out += header + "<|content_text|>" + content + "<|end_message|>";
    }
    if (is_assistant) {
      for (const auto& call : message.tool_calls) {
        if (!message.name.empty() && message.name != call.name) {
          return std::nullopt;
        }
        out += std::string("<|message_model|>") + call.name +
               "<|content_invoke_tool_json|>" + InklingToolCallJson(call) +
               "<|end_message|>";
      }
      out += "<|content_model_end_sampling|>";
    }
  }
  if (!effort_emitted) {
    out += effort_block;
  }
  return out;
}

void LogInklingTemplateSource(const InklingModel& model) {
  // The renderer above is a compiled port of the model's chat_template.jinja
  // (sibling of the weights, also in GGUF `tokenizer.chat_template`); there
  // is no Jinja runtime to interpret the file, so this only records which
  // reference copy was found for operators comparing template revisions.
  std::string source = "compiled-port";
  if (!model.ModelPath().empty()) {
    std::error_code ec;
    const auto sibling = std::filesystem::path(model.ModelPath()).parent_path() /
                         "chat_template.jinja";
    if (std::filesystem::exists(sibling, ec)) {
      source += " file=" + sibling.string();
    } else {
      source += " file-missing";
    }
  }
  if (const auto gguf_template = model.GgufChatTemplate()) {
    source += " gguf_template_bytes=" + std::to_string(gguf_template->size());
  } else {
    source += " gguf-template-missing";
  }
  Logger::Info("loader", "event=inkling_template source=" + source);
}

class InklingTextRunnerState final : public TextRunnerState {
 public:
  explicit InklingTextRunnerState(const std::shared_ptr<InklingModel>& model) {
    std::string error;
    session_ = model->CreateSession(&error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create Inkling session: " + error);
    }
  }
  void SetCancellationCheck(const CancellationCheck&) override {}
  void Invalidate() noexcept override { position_ = 0; }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources() const noexcept override { return {}; }
  [[nodiscard]] InklingSession& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }
 private:
  std::unique_ptr<InklingSession> session_;
  std::size_t position_{0};
};

class InklingTextRunner final : public HipTextModelRunner {
 public:
  // Batch contract (see TextModelRunner): DecodeBatch is serial; AdvanceBatch and the merged plain-advance +
  // prefill chunk are native when the runner has no DSpark draft. Native means one in-order engine pass on one
  // stream with one host sync: every lane relaunches its own captured decode graph (unchanged arithmetic) and the
  // chunk runs its unchanged Sync (which still restarts at position 0 on a non-extension prompt, exactly like the
  // unmixed Prefill). No weight sharing and no concurrency (core scratch is process-wide, single-stream). With a
  // DSpark draft the session captures draft context per forward, so those runners keep the serial fallbacks.
  MBUFO_RUNNER_SERIAL_DECODE_BATCH();
  static constexpr std::size_t kMaxBatchLanes = 16;

  void AdvanceBatch(std::span<const TextRunnerAdvance> advances) const override {
    if (use_dspark() || advances.size() < 2 || advances.size() > kMaxBatchLanes) return SerialAdvanceBatch(advances);
    std::vector<InklingSession*> lanes;
    std::vector<std::int32_t> tokens;
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max()))
        return SerialAdvanceBatch(advances);
      lanes.push_back(&dynamic_cast<InklingTextRunnerState&>(advance.state.get()).session());
      tokens.push_back(static_cast<std::int32_t>(advance.token));
    }
    std::string error;
    bool refused = false;
    if (!InklingSession::AdvanceBatchWithChunk(lanes, tokens, nullptr, {}, &error, &refused)) {
      if (refused) return SerialAdvanceBatch(advances);
      const auto failure = std::make_exception_ptr(std::runtime_error("Inkling batch decode failed: " + error));
      for (const auto& advance : advances) {
        if (!advance.failure) std::rethrow_exception(failure);
        *advance.failure = failure;
      }
      return;
    }
    for (const auto& advance : advances) {
      auto& st = dynamic_cast<InklingTextRunnerState&>(advance.state.get());
      st.set_position(st.position() + 1);
    }
  }

  [[nodiscard]] bool AdvanceBatchWithPrefillChunk(
      std::span<const TextRunnerAdvance> advances, std::span<std::exception_ptr> advance_failures,
      TextRunnerState& prefill_state, std::span<const TextRunnerToken> prompt, std::size_t offset,
      std::size_t max_input_tokens, TextPrefillStep& chunk_step) const override {
    auto& chunk = dynamic_cast<InklingTextRunnerState&>(prefill_state);
    if (use_dspark() || advances.empty() || advances.size() > kMaxBatchLanes || offset != chunk.position() ||
        offset >= prompt.size() || max_input_tokens == 0)
      throw MixedBatchRefused("inkling mixed batch preconditions unmet");
    const std::size_t consumed = std::min(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    if (next_position > max_context_) throw MixedBatchRefused("inkling mixed batch prompt exceeds context");
    std::vector<InklingSession*> lanes;
    std::vector<std::int32_t> tokens;
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max()))
        throw MixedBatchRefused("inkling mixed batch token exceeds range");
      auto& lane = dynamic_cast<InklingTextRunnerState&>(advance.state.get());
      if (&lane == &chunk) throw MixedBatchRefused("inkling mixed batch lane is the chunk state");
      lanes.push_back(&lane.session());
      tokens.push_back(static_cast<std::int32_t>(advance.token));
    }
    std::vector<std::int32_t> prefix;
    prefix.reserve(next_position);
    for (std::size_t i = 0; i < next_position; ++i) prefix.push_back(static_cast<std::int32_t>(prompt[i]));
    std::string error;
    bool refused = false;
    if (!InklingSession::AdvanceBatchWithChunk(lanes, tokens, &chunk.session(), prefix, &error, &refused)) {
      if (refused) throw MixedBatchRefused("inkling mixed batch refused cleanly: " + error);
      chunk.set_position(0);
      const auto failure = std::make_exception_ptr(std::runtime_error("Inkling mixed advance failed: " + error));
      for (std::size_t i = 0; i < advances.size(); ++i) {
        if (i < advance_failures.size()) advance_failures[i] = failure;
      }
      return false;
    }
    for (std::size_t i = 0; i < advances.size(); ++i) {
      if (i < advance_failures.size()) advance_failures[i] = nullptr;
      auto& lane = dynamic_cast<InklingTextRunnerState&>(advances[i].state.get());
      lane.set_position(lane.position() + 1);
    }
    chunk.set_position(next_position);
    chunk_step = {.consumed_tokens = consumed, .decode_ready = next_position == prompt.size()};
    static const bool trace = std::getenv("MBUFO_INKLING_MIXED_TRACE") != nullptr;
    if (trace) std::fprintf(stderr, "[inkling-mixed] lanes=%zu chunk=%zu offset=%zu\n", advances.size(), consumed, offset);
    return true;
  }

  // No multi-token merged verify pass for Inkling: explicit refusal.
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatchWithPrefillChunk(
      std::span<const TextRunnerDecode>, TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, TextPrefillStep&) const override {
    RefuseMixedBatch("inkling has no merged multi-token decode");
  }
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kInkling;
  }
  InklingTextRunner(std::shared_ptr<InklingModel> model, std::uint32_t max_context,
                    std::string dspark_model_path = {},
                    std::uint32_t max_draft_tokens = 0)
      : model_(std::move(model)),
        max_context_(max_context),
        dspark_model_path_(std::move(dspark_model_path)),
        max_draft_tokens_(max_draft_tokens) {
    LogInklingTemplateSource(*model_);
    for (const char* marker : {"</assistant>", "\u3008|EOS|\u3009"}) {
      if (const auto id = model_->tokenizer().FindSpecialToken(marker)) {
        turn_stops_.push_back(static_cast<std::int32_t>(*id));
      }
    }
  }
  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kInklingStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = false,
                .fork = false,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_dspark(),
                .batched_multi_token_decode = false,
                .batched_multi_token_decode_max_width = 0,
                .prefix_reuse = false,
                .mixed_prefill_decode = !use_dspark(),
            },
        .persistence = std::nullopt,
    };
  }
  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0, total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) capacity = free_bytes;
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }
  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{{.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    if (!use_dspark()) {
      for (std::size_t width = 2; width <= 8; ++width) {
        plans.push_back({.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
      }
    }
    return plans;
  }
  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary> BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          const std::int32_t token = static_cast<std::int32_t>(id);
          const std::int32_t unit[1] = {token};
          return sampling::ConstraintVocabulary::Piece{
              model_->Decode(std::span<const std::int32_t>(unit, 1)),
              model_->IsStopToken(token)};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override { return model_->VocabSize(); }
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override { return {"<|content_thinking|>", "<|end_message|>", {}}; }
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(std::string_view text) const override {
    std::vector<TextRunnerToken> tokens;
    for (const std::int32_t id : model_->Tokenize(text, true)) {
      tokens.push_back(static_cast<TextRunnerToken>(id));
    }
    return tokens;
  }
  struct InklingImageContext final : TextPromptContext {
    std::vector<InklingSession::ImageInjection> images;
  };
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(const ChatRequest& request) const override {
    const auto prompt = InklingRenderPrompt(request);
    if (!prompt.has_value() || prompt->empty()) return std::nullopt;
    return Tokenize(*prompt);
  }
  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    const bool has_images = std::ranges::any_of(
        request.messages, [](const auto& m) { return !m.images.empty(); });
    const auto tokens = RenderAndTokenize(request);
    if (!tokens.has_value()) return std::nullopt;
    if (!has_images) return TextPreparedPrompt{*tokens, {}, 0};
    if (!model_->HasVision()) {
      throw std::invalid_argument(
          "image input requires a matching --mmproj BF16 sidecar");
    }
    auto ctx = std::make_shared<InklingImageContext>();
    const auto img_tok = model_->tokenizer().FindSpecialToken("<|unused_200054|>");
    if (!img_tok.has_value()) {
      throw std::runtime_error("missing <|unused_200054|> token in tokenizer");
    }
    const TextRunnerToken placeholder = static_cast<TextRunnerToken>(*img_tok);
    const std::uint32_t kVisTokens =
        models::inkling::VisionEncoder::kMergedTokens;
    std::size_t scan = 0;
    for (const auto& msg : request.messages) {
      for (const auto& img_bytes : msg.images) {
        while (scan < tokens->size() && (*tokens)[scan] != placeholder) ++scan;
        if (scan + kVisTokens > tokens->size()) {
          throw std::runtime_error("inkling vision placeholder mismatch");
        }
        const core::Image decoded = core::DecodeImage(*img_bytes.bytes);
        InklingSession::ImageInjection inj;
        inj.offset = static_cast<std::uint32_t>(scan);
        std::string err;
        if (!model_->EncodeImage(decoded, &inj.embeds, &err)) {
          throw std::runtime_error("inkling vision encode failed: " + err);
        }
        ctx->images.push_back(std::move(inj));
        scan += kVisTokens;
      }
    }
    return TextPreparedPrompt{*tokens, std::move(ctx), 0};
  }
  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    auto& st = dynamic_cast<InklingTextRunnerState&>(state);
    if (!context) {
      st.session().SetImageInjections({});
      return;
    }
    const auto* ic = dynamic_cast<const InklingImageContext*>(context.get());
    if (ic == nullptr) {
      throw std::invalid_argument("invalid Inkling prompt context");
    }
    st.session().SetImageInjections(ic->images);
  }
  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(const ChatRequest&) const override {
    return TextGenerationBackend::InitialOutputState::kContent;
  }
  [[nodiscard]] std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    std::vector<std::int32_t> ids;
    ids.reserve(tokens.size());
    for (const auto t : tokens) ids.push_back(static_cast<std::int32_t>(t));
    return model_->Decode(ids);
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<InklingTextRunnerState>(model_);
  }
  [[nodiscard]] bool IsTurnStopToken(std::int32_t token) const noexcept {
    if (model_->IsStopToken(token)) return true;
    return std::ranges::find(turn_stops_, token) != turn_stops_.end();
  }
  [[nodiscard]] TextPrefillStep Prefill(TextRunnerState& state,
                                        std::span<const TextRunnerToken> prompt,
                                        std::size_t offset,
                                        std::size_t max_input_tokens) const override {
    auto& st = dynamic_cast<InklingTextRunnerState&>(state);
    if (offset != st.position()) {
      throw std::logic_error("Inkling prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Inkling prefill has no remaining input");
    }
    const std::size_t consumed = std::min(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    if (next_position > max_context_) {
      throw std::runtime_error("Inkling prompt exceeds the served context");
    }
    std::vector<std::int32_t> prefix;
    prefix.reserve(next_position);
    for (std::size_t i = 0; i < next_position; ++i) {
      prefix.push_back(static_cast<std::int32_t>(prompt[i]));
    }
    std::string err;
    if (!st.session().Sync(prefix, &err)) {
      st.set_position(0);
      throw std::runtime_error("Inkling prefill failed: " + err);
    }
    st.set_position(next_position);
    return {.consumed_tokens = consumed, .decode_ready = next_position == prompt.size()};
  }
  [[nodiscard]] TextDecodeSelection SelectNext(TextRunnerState& state,
                                               sampling::SamplerState& sampler) const override {
    auto& st = dynamic_cast<InklingTextRunnerState&>(state);
    if (st.position() >= max_context_) return {.stop = true, .piece = {}};
    const auto logits = st.session().Logits();
    if (logits.empty()) throw std::runtime_error("Inkling token selection has no logits");
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    if (st.stop_at_eos() && IsTurnStopToken(token)) {
      return {.stop = true, .token = 0, .piece = {}};
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->Decode(std::span<const std::int32_t>(&token, 1)),
    };
  }
  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& st = dynamic_cast<InklingTextRunnerState&>(state);
    std::string err;
    if (!st.session().Evaluate(static_cast<std::int32_t>(token), &err)) {
      throw std::runtime_error("Inkling decode failed: " + err);
    }
    st.set_position(st.position() + 1);
  }
  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_dspark() || max_tokens <= 1 ||
        sampler.ForcedToken().has_value() ||
        !sampler.config().can_use_unmodified_argmax() ||
        sampler.config().constraint != nullptr ||
        sampler.NeedsConstraintMask()) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    auto& st = dynamic_cast<InklingTextRunnerState&>(state);
    if (st.position() >= max_context_) return {.stop = true};
    const std::uint32_t n_draft = static_cast<std::uint32_t>(
        std::min<std::size_t>(max_draft_tokens_, max_tokens - 1));
    std::vector<std::int32_t> block;
    std::string error;
    if (!st.session().SpeculativeRound(n_draft, &block, &error) ||
        block.empty()) {
      throw std::runtime_error("Inkling speculative decode failed: " + error);
    }
    TextDecodeStep step;
    step.draft_rounds = 1;
    step.draft_tokens = n_draft;
    step.draft_accepted_tokens = block.size() - 1;
    bool stop = false;
    step.selections.reserve(block.size());
    for (const std::int32_t tok : block) {
      if (st.stop_at_eos() && IsTurnStopToken(tok)) {
        stop = true;
        break;
      }
      step.selections.push_back(TextDecodeSelection{
          .stop = false,
          .token = static_cast<TextRunnerToken>(tok),
          .piece = model_->Decode(std::span<const std::int32_t>(&tok, 1)),
      });
    }
    st.set_position(st.position() + block.size());
    step.stop = stop;
    return step;
  }
  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }
  [[nodiscard]] std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    return dynamic_cast<const InklingTextRunnerState&>(state).position();
  }
 private:
  [[nodiscard]] bool use_dspark() const noexcept {
    return !dspark_model_path_.empty() && max_draft_tokens_ > 0 &&
           model_->HasDspark();
  }
  std::shared_ptr<InklingModel> model_;
  std::uint32_t max_context_{0};
  std::string dspark_model_path_;
  std::uint32_t max_draft_tokens_{0};
  std::vector<std::int32_t> turn_stops_;
};

class LlamaTextRunnerState final : public TextRunnerState {
 public:
  explicit LlamaTextRunnerState(const std::shared_ptr<LlamaModel>& model) {
    std::string error;
    session_ = model->CreateSession(&error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create Llama session: " + error);
    }
  }

  void SetCancellationCheck(const CancellationCheck&) override {}
  void Invalidate() noexcept override { position_ = 0; }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {};
  }

  [[nodiscard]] LlamaSession& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

 private:
  std::unique_ptr<LlamaSession> session_;
  std::size_t position_{0};
};

LlamaTextRunnerState& RequireLlamaState(TextRunnerState& state) {
  auto* llama = dynamic_cast<LlamaTextRunnerState*>(&state);
  if (llama == nullptr) {
    throw std::logic_error("text runner state is not Llama");
  }
  return *llama;
}

const LlamaTextRunnerState& RequireLlamaState(const TextRunnerState& state) {
  const auto* llama = dynamic_cast<const LlamaTextRunnerState*>(&state);
  if (llama == nullptr) {
    throw std::logic_error("text runner state is not Llama");
  }
  return *llama;
}

class LlamaTextRunner final : public HipTextModelRunner {
 public:
  // Batch contract (see TextModelRunner): serial DecodeBatch (no speculative decode); native AdvanceBatch and
  // AdvanceBatchWithPrefillChunk for Q8_0 weights (serial / refusing otherwise); the multi-token merged entry point is
  // an explicit refusal.
  MBUFO_RUNNER_SERIAL_DECODE_BATCH();
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kLlama;
  }

  LlamaTextRunner(std::shared_ptr<LlamaModel> model,
                  std::uint32_t max_context)
      : model_(std::move(model)), max_context_(max_context) {
    // A turn ends at <|eot_id|>; the terminal <|end_of_text|> is covered by
    // the engine stop tokens.
    for (const char* marker : {"<|eot_id|>"}) {
      if (const auto id = model_->tokenizer().FindSpecialToken(marker)) {
        turn_stops_.push_back(static_cast<std::int32_t>(*id));
      }
    }
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kLlamaStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = false,
                .fork = false,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = false,
                .batched_multi_token_decode = false,
                .batched_multi_token_decode_max_width = 0,
                .prefix_reuse = false,
                .mixed_prefill_decode = model_->PackedLanesExact(),
            },
        .persistence = std::nullopt,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          const std::int32_t token = static_cast<std::int32_t>(id);
          const std::int32_t unit[1] = {token};
          return sampling::ConstraintVocabulary::Piece{
              model_->Decode(std::span<const std::int32_t>(unit, 1)),
              model_->IsStopToken(token)};
        });
  }

  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return model_->VocabSize();
  }

  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"", "", {}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    std::vector<TextRunnerToken> tokens;
    for (const std::int32_t id : model_->Tokenize(text, /*add_bos=*/true)) {
      tokens.push_back(static_cast<TextRunnerToken>(id));
    }
    return tokens;
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    const std::string prompt = LlamaRenderPrompt(request);
    if (prompt.empty())
      return std::nullopt;
    return Tokenize(prompt);
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest&) const override {
    return TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    const auto ids = LlamaEngineTokens(tokens);
    return model_->Decode(ids);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<LlamaTextRunnerState>(model_);
  }

  [[nodiscard]] bool IsTurnStopToken(std::int32_t token) const noexcept {
    if (model_->IsStopToken(token))
      return true;
    return std::ranges::find(turn_stops_, token) != turn_stops_.end();
  }

  [[nodiscard]] std::string DecodeToken(std::int32_t token) const {
    return model_->Decode(std::span<const std::int32_t>(&token, 1));
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& llama = RequireLlamaState(state);
    if (offset != llama.position()) {
      throw std::logic_error(
          "Llama prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Llama prefill has no remaining input");
    }
    const std::size_t consumed =
        std::min(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    if (next_position > max_context_) {
      throw std::runtime_error("Llama prompt exceeds the served context");
    }
    // Incremental on a canonical grid: the engine forwards only up to the last grid boundary at or before the consumed
    // position (everything once the prompt is complete) and keeps the rest pending, so the forwarded pieces never
    // depend on the scheduler's budget slicing (a piece's arithmetic depends on its row count).
    const std::size_t computed_from = offset == 0 ? 0 : llama.session().Length();
    const std::size_t computed_to =
        std::max(computed_from, std::size_t{LlamaSession::PrefillTarget(next_position, prompt.size())});
    if (computed_to > computed_from) {
      const auto prefix = LlamaEngineTokens(prompt.first(computed_to));
      std::string error;
      if (!llama.session().Extend(prefix, computed_from, &error)) {
        llama.set_position(0);
        throw std::runtime_error("Llama prefill failed: " + error);
      }
    }
    llama.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& llama = RequireLlamaState(state);
    if (llama.position() >= max_context_)
      return {.stop = true, .piece = {}};
    const auto logits = llama.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error("Llama token selection has no logits");
    }
    const auto token =
        static_cast<std::int32_t>(sampler.Sample(logits));
    if (llama.stop_at_eos() && IsTurnStopToken(token)) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = DecodeToken(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Llama token ID exceeds engine range");
    }
    auto& llama = RequireLlamaState(state);
    std::string error;
    if (!llama.session().Evaluate(static_cast<std::int32_t>(token),
                                   &error)) {
      throw std::runtime_error("Llama decode failed: " + error);
    }
    llama.set_position(llama.position() + 1);
  }

  // Plain-token advance for every lane through packed layer-synchronous engine passes of up to 8 lanes (one weight
  // read for the projections of all lanes; per-lane attention; bit-identical to Advance). The engine validates before
  // it mutates, so a refusal reruns that group serially with nothing half-applied.
  void AdvanceBatch(std::span<const TextRunnerAdvance> advances) const override {
    constexpr std::size_t kLanes = LlamaSession::kMaxLanes;
    if (advances.size() < 2 || !model_->PackedLanesExact()) return SerialAdvanceBatch(advances);
    for (std::size_t base = 0; base < advances.size(); base += kLanes) {
      AdvanceGroup(advances.subspan(base, std::min(kLanes, advances.size() - base)));
    }
  }

  // Merged plain-advance + prefill chunk: the lanes step through one packed pass and the chunk runs Extend (the same
  // prefill core as Prefill) in the same call, so the result is bit-identical to Prefill followed by AdvanceBatch.
  // Refuses with nothing mutated for shapes the engine cannot host.
  [[nodiscard]] bool AdvanceBatchWithPrefillChunk(
      std::span<const TextRunnerAdvance> advances, std::span<std::exception_ptr> advance_failures,
      TextRunnerState& prefill_state, std::span<const TextRunnerToken> prompt, std::size_t offset,
      std::size_t max_input_tokens, TextPrefillStep& chunk_step) const override {
    auto& chunk_state = RequireLlamaState(prefill_state);
    if (advances.empty() || advances.size() > LlamaSession::kMaxLanes || advance_failures.size() != advances.size() ||
        offset != chunk_state.position() || offset >= prompt.size() || max_input_tokens == 0 ||
        !model_->PackedLanesExact()) {
      throw MixedBatchRefused("llama mixed batch preconditions unmet");
    }
    const std::size_t consumed = std::min(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    if (next_position > max_context_) throw MixedBatchRefused("llama mixed chunk exceeds the served context");
    std::vector<LlamaSession::LaneItem> items;
    items.reserve(advances.size());
    for (std::size_t i = 0; i < advances.size(); ++i) {
      if (advances[i].token > static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max())) {
        throw MixedBatchRefused("llama mixed token exceeds engine range");
      }
      auto* lane = &RequireLlamaState(advances[i].state.get());
      if (lane == &chunk_state) throw MixedBatchRefused("llama mixed batch lane is the prefill state");
      if (lane->position() + 1 > max_context_) throw MixedBatchRefused("llama mixed lane is out of context");
      items.push_back({&lane->session(), static_cast<std::int32_t>(advances[i].token)});
    }
    const std::size_t computed_from = offset == 0 ? 0 : chunk_state.session().Length();
    const std::size_t computed_to =
        std::max(computed_from, std::size_t{LlamaSession::PrefillTarget(next_position, prompt.size())});
    const auto prefix = LlamaEngineTokens(prompt.first(computed_to));
    const LlamaSession::ChunkItem chunk{&chunk_state.session(), std::span<const std::int32_t>(prefix), computed_from};
    std::string error;
    bool refused = false;
    bool lanes_advanced = false;
    if (!LlamaSession::DecodeBatchWithChunk(items, chunk, &error, &refused, &lanes_advanced)) {
      if (refused) throw MixedBatchRefused("llama mixed batch refused: " + error);
      if (lanes_advanced) {
        // The lanes took their step; only the chunk failed. The pool reports the chunk failure after recording them.
        for (std::size_t i = 0; i < advances.size(); ++i) {
          advance_failures[i] = nullptr;
          auto& lane = RequireLlamaState(advances[i].state.get());
          lane.set_position(lane.position() + 1);
        }
        chunk_state.set_position(0);
        return false;
      }
      chunk_state.set_position(0);
      for (std::size_t i = 0; i < advances.size(); ++i) {
        advance_failures[i] = std::make_exception_ptr(std::runtime_error("Llama mixed advance failed: " + error));
      }
      return false;
    }
    for (std::size_t i = 0; i < advances.size(); ++i) {
      advance_failures[i] = nullptr;
      auto& lane = RequireLlamaState(advances[i].state.get());
      lane.set_position(lane.position() + 1);
    }
    chunk_state.set_position(next_position);
    chunk_step = {.consumed_tokens = consumed, .decode_ready = next_position == prompt.size()};
    static const bool trace = std::getenv("MBUFO_LLAMA_MIXED_TRACE") != nullptr;
    if (trace) {
      std::fprintf(stderr, "[llama-mixed] lanes=%zu consumed=%zu offset=%zu computed=[%zu,%zu)\n", advances.size(),
                   consumed, offset, computed_from, computed_to);
    }
    return true;
  }

  // Llama has no multi-token (speculative) decode, so there is nothing to merge a chunk into.
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatchWithPrefillChunk(
      std::span<const TextRunnerDecode>, TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, TextPrefillStep&) const override {
    RefuseMixedBatch("llama has no merged multi-token decode");
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireLlamaState(state).position();
  }

 private:
  void AdvanceGroup(std::span<const TextRunnerAdvance> group) const {
    if (group.size() < 2) return SerialAdvanceBatch(group);
    std::vector<LlamaSession::LaneItem> items;
    items.reserve(group.size());
    for (std::size_t i = 0; i < group.size(); ++i) {
      if (group[i].token > static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max())) {
        return SerialAdvanceBatch(group);  // the serial path reports the per-lane error
      }
      auto& llama = RequireLlamaState(group[i].state.get());
      if (llama.position() + 1 > max_context_) return SerialAdvanceBatch(group);
      items.push_back({&llama.session(), static_cast<std::int32_t>(group[i].token)});
    }
    std::string error;
    bool refused = false;
    if (!LlamaSession::StepBatch(items, &error, &refused)) {
      if (refused) return SerialAdvanceBatch(group);
      // Mid-pass failure: no lane can be trusted to have advanced.
      for (const auto& advance : group) {
        auto failure = std::make_exception_ptr(std::runtime_error("Llama batched decode failed: " + error));
        if (advance.failure == nullptr) std::rethrow_exception(failure);
        *advance.failure = failure;
      }
      return;
    }
    for (const auto& advance : group) {
      auto& llama = RequireLlamaState(advance.state.get());
      llama.set_position(llama.position() + 1);
    }
  }

  std::shared_ptr<LlamaModel> model_;
  std::uint32_t max_context_{0};
  std::vector<std::int32_t> turn_stops_;
};

// ---------------------------------------------------------------------------
// Muse Glimmer HTTP backend: plain Sync/Evaluate sessions owned per request
// slot, no speculative sidecar and no continuation snapshots in serve. Prompt
// framing mirrors the GGUF canonical template (Onyx ATEM tool format);
// tool calls round-trip through the native ATEM envelope (parsed back in
// openai_chat.cpp ParseAtemCalls).
// ---------------------------------------------------------------------------
constexpr std::string_view kGlimmerStateAbi = "glimmer-gfx1151-state-v1";

using GlimmerModel = models::glimmer::Model;
using GlimmerSession = models::glimmer::Session;

std::vector<std::int32_t> GlimmerEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Glimmer token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}


// Tool namespace of first appearance, mirroring the template's namespace
// grouping for the metadata block and the recipient list.
std::vector<std::string> GlimmerToolNamespaces(
    std::span<const tokenization::ChatTool> tools) {
  std::vector<std::string> namespaces;
  for (const auto& tool : tools) {
    const auto dot = tool.name.find('.');
    const std::string ns =
        dot == std::string::npos ? tool.name : tool.name.substr(0, dot);
    if (std::ranges::find(namespaces, ns) == namespaces.end())
      namespaces.push_back(ns);
  }
  return namespaces;
}

void GlimmerAppendParam(std::string& out, std::string_view key,
                        std::string_view value) {
  out += '<';
  out += "atem:parameter";
  out += ' ';
  out += "name";
  out += '=';
  out += '"';
  out += key;
  out += '"';
  out += '>';
  out += value;
  out += '<';
  out += "/atem:parameter>";
}

std::string GlimmerToolDefs(std::span<const tokenization::ChatTool> tools) {
  std::string out = "Available tools (JSONSchema):\n// Tool metadata\n";
  out += "Sketch:\n<atem:function_calls>\n<atem:invoke name=\"NAME\">\n";
  GlimmerAppendParam(out, "ARG", "VALUE");
  out += "\n</atem:invoke>\n</atem:function_calls>\n";

  for (const auto& ns : GlimmerToolNamespaces(tools)) {
    out += "{\"name\": " + json::Value(ns).dump();
    out += ", \"description\": \"\"}\n";
  }
  out += "// Function schemas";
  for (const auto& tool : tools) {
    const json::Value params = tool.parameters_json.empty()
                                   ? json::Value::object()
                                   : json::parse(tool.parameters_json);
    out += "\n{\"name\": " + json::Value(tool.name).dump();
    out += ", \"description\": " + json::Value(tool.description).dump();
    out += ", \"parameters\": " + params.dump() + "}";
  }
  out += "Example:\nto=example_tool_name.example_function_name\n\n";
  out += "<atem:function_calls>\n<atem:invoke name=\"example_tool_name.example_function_name\">\n";
  GlimmerAppendParam(out, "example_parameter_1", "value_1");
  GlimmerAppendParam(out, "example_parameter_2", "line one\nline two");
  out += "\n</atem:invoke>\n</atem:function_calls>";

  return out;
}
std::string GlimmerSystemMeta(std::span<const tokenization::ChatTool> tools) {
  std::string out = "# Valid recipients: \"self\"";
  for (const auto& ns : GlimmerToolNamespaces(tools)) {
    out += ", \"" + ns + ".*\"";
  }
  out += ", \"user\".";
  return out;
}

std::string GlimmerRenderAtemCall(
    std::string_view name,
    std::span<const tokenization::ChatMessage::ToolArgument> args) {
  std::string out = "<atem:function_calls>\n<atem:invoke name=\"";
  out += name;
  out += "\">\n";
  for (const auto& argument : args) {
    out += std::string("<") + "atem:parameter" + " " + "name" + '=' + '"';
    out += argument.name;
    out += "\">";
    out += argument.value;
    out += "</atem:parameter>\n";
  }
  out += "</atem:invoke>\n</atem:function_calls>";
  return out;
}

std::string GlimmerRenderPrompt(
    std::span<const tokenization::ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools) {
  std::string out = "<|begin_of_text|>";
  std::size_t body_begin = 0;
  std::string system_text;
  if (!messages.empty() &&
      (messages.front().role == tokenization::ChatRole::kSystem ||
       messages.front().role == tokenization::ChatRole::kDeveloper)) {
    system_text = messages.front().content + messages.front().framing_suffix;
    body_begin = 1;
  } else {
    system_text = "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.";
  }
  out += "<|start|>system<|message|>" + system_text;
  out += "\n\nReasoning strength: low.";
  if (!tools.empty()) {
    out += "\n\n" + GlimmerToolDefs(tools);
  }
  out += "\n\n" + GlimmerSystemMeta(tools) + "<|eot|>";
  for (std::size_t i = body_begin; i < messages.size(); ++i) {
    const auto& message = messages[i];
    if (message.role == tokenization::ChatRole::kTool) {
      std::string tool_name = message.name;
      if (tool_name.empty() && !message.tool_call_id.empty()) {
        for (std::size_t j = i; j-- > 0;) {
          if (messages[j].role != tokenization::ChatRole::kAssistant)
            continue;
          for (const auto& call : messages[j].tool_calls) {
            if (call.id == message.tool_call_id)
              tool_name = call.name;
          }
        }
        if (tool_name.empty())
          tool_name = message.tool_call_id;
      }
      if (tool_name.empty())
        tool_name = "unknown";
    out += "<|start|>tool " + tool_name + "<|message|>" + std::string("<") + "tool_output" + " " + "name" + '=' + '"';
      out += tool_name;
    out += "\">\n" + message.content + "\n</tool_output><|eot|>";
      continue;
    }
    if (message.role == tokenization::ChatRole::kAssistant) {
      if (!message.thought.empty()) {
        out += "<|start|>assistant to=self<|message|>";
        out += message.thought;
        out += "<|eom|>";
      }
      if (!message.tool_calls.empty()) {
        for (std::size_t k = 0; k < message.tool_calls.size(); ++k) {
          const auto& call = message.tool_calls[k];
          out += "<|start|>assistant to=" + call.name + "<|message|>";
          out += GlimmerRenderAtemCall(call.name, call.arguments);
          const bool last_turn =
              (k + 1 == message.tool_calls.size()) &&
              (i + 1 == messages.size() ||
               messages[i + 1].role != tokenization::ChatRole::kAssistant);
          out += last_turn ? "<|eot|>" : "<|eom|>";
        }
        continue;
      }
      const bool end_turn =
          (i + 1 == messages.size()) ||
          (messages[i + 1].role != tokenization::ChatRole::kAssistant);
      out += "<|start|>assistant to=user<|message|>" + message.content;
      out += end_turn ? "<|eot|>" : "<|eom|>";
      continue;
    }
    out += "<|start|>user<|message|>" + message.content + "<|eot|>";
  }
  out += "<|start|>assistant";
  return out;
}

class GlimmerTextRunnerState final : public TextRunnerState {
 public:
  using SharedDraft = models::glimmer::GlimmerDflashDraftBackend::SharedDraft;

  /// `shared_draft` (nullable) enables DFlash speculative decoding: the state owns a draft session bound to the shared
  /// draft weights and the target session captures the tapped-layer features the draft consumes.
  explicit GlimmerTextRunnerState(const std::shared_ptr<GlimmerModel>& model,
                                  const std::shared_ptr<SharedDraft>& shared_draft = nullptr,
                                  std::uint32_t max_draft_tokens = 0)
      : hidden_(model->weights().config.hidden_size) {
    std::string error;
    session_ = model->CreateSession(&error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create Glimmer session: " + error);
    }
    if (shared_draft != nullptr) {
      draft_ = models::glimmer::GlimmerDflashDraftBackend::CreateFromShared(shared_draft, max_draft_tokens, &error);
      if (draft_ == nullptr) {
        throw std::runtime_error("Failed to create Glimmer DFlash draft session: " + error);
      }
      if (!session_->EnableDraftTaps(draft_->TargetHiddenLayerIds(), &error)) {
        throw std::runtime_error("Failed to enable Glimmer draft taps: " + error);
      }
      feat_width_ = draft_->TargetHiddenLayerIds().size() * hidden_;
    }
  }

  void SetCancellationCheck(const CancellationCheck&) override {}
  void Invalidate() noexcept override {
    position_ = 0;
    ResetSpeculative();
  }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {};
  }

  [[nodiscard]] GlimmerSession& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

  // ---- DFlash speculative state (empty unless the runner enabled a draft)
  [[nodiscard]] bool speculative_enabled() const noexcept { return draft_ != nullptr; }
  [[nodiscard]] models::glimmer::GlimmerDflashDraftBackend& draft() const { return *draft_; }
  /// True once the draft holds the prompt's features: decode may run speculative rounds.
  [[nodiscard]] bool speculative_primed() const noexcept { return spec_primed_; }
  [[nodiscard]] bool speculative_pending() const noexcept { return spec_pending_; }
  void set_speculative_pending(bool value) noexcept { spec_pending_ = value; }
  /// Committed tokens of the sequence (prompt + emitted); when pending, the last one is committed but not yet forwarded.
  [[nodiscard]] std::vector<std::int32_t>& committed() noexcept { return committed_; }
  [[nodiscard]] std::size_t feat_width() const noexcept { return feat_width_; }
  [[nodiscard]] std::uint32_t hidden() const noexcept { return hidden_; }

  void ResetSpeculative() noexcept {
    spec_primed_ = false;
    spec_pending_ = false;
    committed_.clear();
    feats_.clear();
    if (draft_ != nullptr) draft_->Reset();
  }

  /// Stores the rows of one tapped layer's output (host copy of [n][hidden]) into the prompt feature matrix at their
  /// absolute positions; rows arrive in any layer / chunk order.
  void StoreTapRows(std::size_t slot, std::uint32_t start, std::uint32_t n, const std::vector<float>& rows) {
    const std::size_t need = (static_cast<std::size_t>(start) + n) * feat_width_;
    if (feats_.size() < need) feats_.resize(need, 0.0F);
    for (std::uint32_t i = 0; i < n; ++i) {
      std::copy_n(rows.data() + static_cast<std::size_t>(i) * hidden_, hidden_,
                  feats_.data() + (static_cast<std::size_t>(start) + i) * feat_width_ + slot * hidden_);
    }
  }
  /// Primes the draft with the captured features of the whole prompt; the draft then follows the verified commits.
  [[nodiscard]] bool PrimeDraft(std::span<const TextRunnerToken> prompt) {
    if (draft_ == nullptr || feats_.size() != prompt.size() * feat_width_) return false;
    std::vector<tokenization::TokenId> ids;
    ids.reserve(prompt.size());
    committed_.clear();
    committed_.reserve(prompt.size() + 1024);
    for (const auto token : prompt) {
      ids.push_back(static_cast<tokenization::TokenId>(token));
      committed_.push_back(static_cast<std::int32_t>(token));
    }
    speculative::DraftTargetContext ctx;
    ctx.prompt_tokens = ids;
    ctx.prompt_hidden_states = feats_;
    ctx.hidden_size = static_cast<std::uint32_t>(feat_width_);
    ctx.first_token = ids.back();
    spec_pending_ = false;
    spec_primed_ = draft_->PrimeTargetContext(ctx);
    feats_.clear();
    feats_.shrink_to_fit();
    return spec_primed_;
  }

 private:
  std::unique_ptr<GlimmerSession> session_;
  std::size_t position_{0};
  std::uint32_t hidden_{0};
  std::size_t feat_width_{0};
  std::unique_ptr<models::glimmer::GlimmerDflashDraftBackend> draft_;
  std::vector<std::int32_t> committed_;
  std::vector<float> feats_;
  bool spec_primed_{false};
  bool spec_pending_{false};
};

GlimmerTextRunnerState& RequireGlimmerState(TextRunnerState& state) {
  auto* glimmer = dynamic_cast<GlimmerTextRunnerState*>(&state);
  if (glimmer == nullptr) {
    throw std::logic_error("text runner state is not Glimmer");
  }
  return *glimmer;
}

const GlimmerTextRunnerState& RequireGlimmerState(
    const TextRunnerState& state) {
  const auto* glimmer = dynamic_cast<const GlimmerTextRunnerState*>(&state);
  if (glimmer == nullptr) {
    throw std::logic_error("text runner state is not Glimmer");
  }
  return *glimmer;
}

class GlimmerTextRunner final : public HipTextModelRunner {
 public:
  // Batch contract (see TextModelRunner): native DecodeBatch / AdvanceBatch / AdvanceBatchWithPrefillChunk (all
  // without a shared draft; with one they fall back to serial or refuse); the multi-token merged entry point is an
  // explicit refusal because the DFlash verify path is per-session.
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kAtem;
  }

  GlimmerTextRunner(std::shared_ptr<GlimmerModel> model,
                    std::uint32_t max_context,
                    std::shared_ptr<GlimmerTextRunnerState::SharedDraft> shared_draft = nullptr,
                    std::uint32_t max_draft_tokens = 0)
      : model_(std::move(model)), max_context_(max_context), shared_draft_(std::move(shared_draft)),
        max_draft_tokens_(max_draft_tokens) {
    // Turn-boundary stops from the template: the end-of-turn markers end
    // generation, while the continuation marker never does on its own.
    for (const char* marker : {"<|eot|>", "<|eot_id|>"}) {
      if (const auto id = model_->tokenizer().FindSpecialToken(marker)) {
        turn_stops_.push_back(static_cast<std::int32_t>(*id));
      }
    }
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kGlimmerStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = false,
                .fork = false,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = shared_draft_ != nullptr,
                .batched_multi_token_decode = false,
                .batched_multi_token_decode_max_width = 0,
                .prefix_reuse = false,
                .mixed_prefill_decode = shared_draft_ == nullptr,
            },
        .persistence = std::nullopt,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          const std::int32_t token = static_cast<std::int32_t>(id);
          const std::int32_t unit[1] = {token};
          return sampling::ConstraintVocabulary::Piece{
              model_->Decode(std::span<const std::int32_t>(unit, 1)),
              model_->IsStopToken(token)};
        });
  }

  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return model_->VocabSize();
  }

  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"", "", {}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    std::vector<TextRunnerToken> tokens;
    for (const std::int32_t id : model_->Tokenize(text, /*add_bos=*/true)) {
      tokens.push_back(static_cast<TextRunnerToken>(id));
    }
    return tokens;
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    const auto tools =
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools};
    const std::string prompt = GlimmerRenderPrompt(request.messages, tools);
    if (prompt.empty())
      return std::nullopt;
    return Tokenize(prompt);
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest&) const override {
    return TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    const auto ids = GlimmerEngineTokens(tokens);
    return model_->Decode(ids);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<GlimmerTextRunnerState>(model_, shared_draft_, max_draft_tokens_);
  }

  [[nodiscard]] bool IsTurnStopToken(std::int32_t token) const noexcept {
    if (model_->IsStopToken(token))
      return true;
    return std::ranges::find(turn_stops_, token) != turn_stops_.end();
  }

  [[nodiscard]] std::string DecodeToken(std::int32_t token) const {
    return model_->Decode(std::span<const std::int32_t>(&token, 1));
  }

  // Glimmer Sync replays only the delta when the prompt extends the retained
  // prefix and resynchronizes from zero otherwise, so every prefill step
  // replays the whole retained prefix through one Sync call.
  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& glimmer = RequireGlimmerState(state);
    if (offset != glimmer.position()) {
      throw std::logic_error(
          "Glimmer prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Glimmer prefill has no remaining input");
    }
    const std::size_t consumed =
        std::min(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    if (next_position > max_context_) {
      throw std::runtime_error("Glimmer prompt exceeds the served context");
    }
    const auto prefix = GlimmerEngineTokens(prompt.first(next_position));
    std::string error;
    if (glimmer.speculative_enabled()) {
      // DFlash: capture the tapped-layer features of every prompt row during the prefill; the draft is primed with
      // them once the prompt is complete.
      if (offset == 0) glimmer.ResetSpeculative();
      const auto tap_ids = glimmer.draft().TargetHiddenLayerIds();
      GlimmerSession::TapSink sink;
      sink.layers.assign(tap_ids.begin(), tap_ids.end());
      std::vector<float> staging;
      sink.fn = [&](std::uint32_t layer, const float* d_rows, std::uint32_t n, std::uint32_t start) {
        for (std::size_t slot = 0; slot < sink.layers.size(); ++slot) {
          if (sink.layers[slot] != layer) continue;
          staging.resize(static_cast<std::size_t>(n) * glimmer.hidden());
          if (hipMemcpy(staging.data(), d_rows, staging.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
            throw std::runtime_error("Glimmer DFlash feature capture failed");
          glimmer.StoreTapRows(slot, start, n, staging);
        }
      };
      if (!glimmer.session().Sync(prefix, &error, &sink)) {
        glimmer.set_position(0);
        glimmer.ResetSpeculative();
        throw std::runtime_error("Glimmer prefill failed: " + error);
      }
      if (next_position == prompt.size() && !glimmer.PrimeDraft(prompt.first(next_position))) {
        // Priming refused (e.g. context too long for the draft): decode stays plain autoregressive.
        glimmer.ResetSpeculative();
      }
    } else if (!glimmer.session().Sync(prefix, &error)) {
      glimmer.set_position(0);
      throw std::runtime_error("Glimmer prefill failed: " + error);
    }
    glimmer.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& glimmer = RequireGlimmerState(state);
    if (glimmer.position() >= max_context_)
      return {.stop = true, .piece = {}};
    const auto logits = glimmer.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error("Glimmer token selection has no logits");
    }
    const auto token =
        static_cast<std::int32_t>(sampler.Sample(logits));
    if (glimmer.stop_at_eos() && IsTurnStopToken(token)) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = DecodeToken(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Glimmer token ID exceeds engine range");
    }
    auto& glimmer = RequireGlimmerState(state);
    std::string error;
    if (!glimmer.session().Evaluate(static_cast<std::int32_t>(token),
                                     &error)) {
      throw std::runtime_error("Glimmer decode failed: " + error);
    }
    glimmer.set_position(glimmer.position() + 1);
  }

  // Batched single-token decode across sessions through the engine's unified
  // mixed-batch launch. Each lane samples its current frontier on the host,
  // then every pending token advances in one DecodeBatch; a launch refusal
  // falls back to the serial base path. (The scheduler only routes here when
  // the batched multi-token capability is advertised.)
  // DFlash speculative decode (greedy requests only): each step commits the accepted drafts plus the target's own next
  // token. The last committed token stays "pending" (committed but not yet forwarded) and anchors the next round, so the
  // verify batch [pending, drafts...] is the only target forward per round. Any other request (sampling, penalties,
  // grammar) decodes plainly through the base step.
  [[nodiscard]] TextDecodeStep DecodeStep(TextRunnerState& state, std::size_t max_tokens,
                                          sampling::SamplerState& sampler) const override {
    if (max_tokens == 0) {
      throw std::invalid_argument("text runner decode step budget must be at least one token");
    }
    auto& glimmer = RequireGlimmerState(state);
    const bool speculate = shared_draft_ != nullptr && glimmer.speculative_enabled() && glimmer.speculative_primed() &&
                           sampler.config().can_use_unmodified_argmax() && !sampler.ForcedToken().has_value() &&
                           sampler.config().constraint == nullptr && !sampler.NeedsConstraintMask();
    if (!speculate) return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    TextDecodeStep result;
    if (glimmer.position() >= max_context_) {
      result.stop = true;
      return result;
    }
    std::vector<std::int32_t> out;
    if (!glimmer.speculative_pending()) {
      // First token: the target's own prediction after the prompt, committed and left pending.
      out.assign(1, glimmer.session().SampleGreedy());
    } else {
      std::string error;
      const std::uint64_t drafted_before = glimmer.draft().drafted();
      const std::uint64_t accepted_before = glimmer.draft().accepted();
      if (!glimmer.session().SpeculativeStep(glimmer.draft(), glimmer.committed(), max_draft_tokens_, &out, &error,
                                             static_cast<std::uint32_t>(std::min<std::size_t>(max_tokens, 32)))) {
        throw std::runtime_error("Glimmer speculative decode failed: " + error);
      }
      result.draft_rounds = 1;
      result.draft_tokens = static_cast<std::size_t>(glimmer.draft().drafted() - drafted_before);
      result.draft_accepted_tokens = static_cast<std::size_t>(glimmer.draft().accepted() - accepted_before);
    }
    for (const std::int32_t token : out) {
      glimmer.committed().push_back(token);
      if (result.stop) continue;
      if (glimmer.stop_at_eos() && IsTurnStopToken(token)) {
        result.stop = true;
        continue;
      }
      result.selections.push_back(TextDecodeSelection{
          .stop = false, .token = static_cast<TextRunnerToken>(token), .piece = DecodeToken(token)});
      sampler.Accept(static_cast<sampling::TokenId>(token));
    }
    glimmer.set_speculative_pending(true);
    glimmer.set_position(glimmer.committed().size());
    return result;
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || shared_draft_ != nullptr) {
      return SerialDecodeBatch(decodes);
    }
    std::vector<std::int32_t> pending;
    pending.reserve(decodes.size());
    std::vector<std::size_t> active;
    active.reserve(decodes.size());
    std::vector<TextDecodeStep> steps(decodes.size());
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      auto& glimmer = RequireGlimmerState(decodes[index].state.get());
      if (glimmer.position() >= max_context_) {
        steps[index] = {.stop = true};
        continue;
      }
      const auto logits = glimmer.session().Logits();
      if (logits.empty()) {
        TextDecodeStep failed;
        failed.failure = std::make_exception_ptr(
            std::runtime_error("Glimmer batched decode has no logits"));
        steps[index] = std::move(failed);
        continue;
      }
      const auto token = static_cast<std::int32_t>(
          decodes[index].sampler.get().Sample(logits));
      if (glimmer.stop_at_eos() && IsTurnStopToken(token)) {
        steps[index] = {.stop = true};
        continue;
      }
      pending.push_back(token);
      active.push_back(index);
    }
    if (!active.empty()) {
      std::vector<GlimmerSession::BatchItem> items;
      items.reserve(active.size());
      for (std::size_t k = 0; k < active.size(); ++k) {
        auto& glimmer = RequireGlimmerState(decodes[active[k]].state.get());
        items.push_back({&glimmer.session(),
                         std::span<const std::int32_t>(&pending[k], 1)});
      }
      std::string error;
      if (!GlimmerSession::DecodeBatch(items, &error)) {
        return SerialDecodeBatch(decodes);
      }
      for (std::size_t k = 0; k < active.size(); ++k) {
        const std::size_t lane = active[k];
        auto& glimmer = RequireGlimmerState(decodes[lane].state.get());
        glimmer.set_position(glimmer.position() + 1);
        steps[lane].selections.push_back(TextDecodeSelection{
            .stop = false,
            .token = static_cast<TextRunnerToken>(pending[k]),
            .piece = DecodeToken(pending[k]),
        });
      }
    }
    return steps;
  }

  // Plain-token advance for every lane through one layer-synchronous engine pass (StepBatchExact: per-item
  // arithmetic bit-identical to Advance). The engine validates before mutating, so a refusal reruns serially with
  // nothing half-applied. A shared draft keeps the serial path (Advance there also feeds the speculative
  // bookkeeping).
  void AdvanceBatch(std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2 || shared_draft_ != nullptr || advances.size() > kGlimmerMaxBatchLanes) {
      return SerialAdvanceBatch(advances);
    }
    std::vector<std::int32_t> tokens(advances.size());
    std::vector<GlimmerSession::BatchItem> items;
    items.reserve(advances.size());
    for (std::size_t i = 0; i < advances.size(); ++i) {
      if (advances[i].token > static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max())) {
        return SerialAdvanceBatch(advances);  // serial path reports the per-lane error
      }
      tokens[i] = static_cast<std::int32_t>(advances[i].token);
      auto& glimmer = RequireGlimmerState(advances[i].state.get());
      for (std::size_t j = 0; j < i; ++j) {
        if (&advances[j].state.get() == &advances[i].state.get()) return SerialAdvanceBatch(advances);
      }
      items.push_back({&glimmer.session(), std::span<const std::int32_t>(&tokens[i], 1)});
    }
    std::string error;
    if (!GlimmerSession::StepBatchExact(items, &error)) {
      return SerialAdvanceBatch(advances);
    }
    for (const auto& advance : advances) {
      auto& glimmer = RequireGlimmerState(advance.state.get());
      glimmer.set_position(glimmer.position() + 1);
    }
  }

  // Merged plain-advance + prefill chunk: the chunk rows ride the same layer-synchronous engine pass as the
  // single-token advances. Bit-identical to Prefill (Sync of the same delta) followed by AdvanceBatch; the chunk is
  // capped at the engine's 512-row step, which only matters when the scheduler budget exceeds it (the remainder
  // goes through the next call). Refuses with nothing mutated for shapes the engine step cannot host.
  [[nodiscard]] bool AdvanceBatchWithPrefillChunk(
      std::span<const TextRunnerAdvance> advances, std::span<std::exception_ptr> advance_failures,
      TextRunnerState& prefill_state, std::span<const TextRunnerToken> prompt, std::size_t offset,
      std::size_t max_input_tokens, TextPrefillStep& chunk_step) const override {
    static const bool trace = std::getenv("MBUFO_GLIMMER_MIXED_TRACE") != nullptr;
    const auto refuse = [&](const std::string& why) {
      if (trace) std::fprintf(stderr, "[glimmer-mixed] refused: %s\n", why.c_str());
      throw MixedBatchRefused(why);
    };
    auto& chunk_state = RequireGlimmerState(prefill_state);
    if (trace) std::fprintf(stderr, "[glimmer-mixed] enter lanes=%zu offset=%zu pos=%zu\n", advances.size(), offset, chunk_state.position());
    if (shared_draft_ != nullptr || advances.empty() || advances.size() + 1 > kGlimmerMaxBatchLanes ||
        offset != chunk_state.position() || offset >= prompt.size() || max_input_tokens == 0) {
      refuse("glimmer mixed batch preconditions unmet");
    }
    const std::size_t consumed = std::min({max_input_tokens, prompt.size() - offset, kGlimmerMaxChunkRows});
    const std::size_t next_position = offset + consumed;
    if (next_position > max_context_) refuse("glimmer mixed chunk exceeds the served context");
    std::vector<std::int32_t> tokens(advances.size());
    std::vector<GlimmerSession::BatchItem> items;
    items.reserve(advances.size());
    for (std::size_t i = 0; i < advances.size(); ++i) {
      if (advances[i].token > static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max())) {
        refuse("glimmer mixed token exceeds engine range");
      }
      tokens[i] = static_cast<std::int32_t>(advances[i].token);
      auto* lane = &RequireGlimmerState(advances[i].state.get());
      if (lane == &chunk_state) refuse("glimmer mixed batch lane is the prefill state");
      for (std::size_t j = 0; j < i; ++j) {
        if (&advances[j].state.get() == &advances[i].state.get())
          refuse("glimmer mixed batch has a duplicate lane");
      }
      items.push_back({&lane->session(), std::span<const std::int32_t>(&tokens[i], 1)});
    }
    const auto prefix = GlimmerEngineTokens(prompt.first(next_position));
    GlimmerSession::ChunkItem chunk{&chunk_state.session(), std::span<const std::int32_t>(prefix), offset};
    std::string error;
    bool refused = false;
    if (!GlimmerSession::DecodeBatchWithChunk(items, chunk, &error, &refused)) {
      if (refused) refuse("glimmer mixed batch refused: " + error);
      // The engine validated up front, so a failure here is mid-pass: no lane can be trusted to have advanced.
      chunk_state.set_position(0);
      for (std::size_t i = 0; i < advances.size(); ++i) {
        advance_failures[i] =
            std::make_exception_ptr(std::runtime_error("Glimmer mixed advance failed: " + error));
      }
      return false;
    }
    for (std::size_t i = 0; i < advances.size(); ++i) {
      advance_failures[i] = nullptr;
      auto& lane = RequireGlimmerState(advances[i].state.get());
      lane.set_position(lane.position() + 1);
    }
    chunk_state.set_position(next_position);
    chunk_step = {.consumed_tokens = consumed, .decode_ready = next_position == prompt.size()};
    if (trace) {
      std::fprintf(stderr, "[glimmer-mixed] lanes=%zu chunk=%zu offset=%zu\n", advances.size(), consumed, offset);
    }
    return true;
  }

  // The DFlash verify path is per-session (draft rounds are sequential per lane), so it cannot host a chunk.
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatchWithPrefillChunk(
      std::span<const TextRunnerDecode>, TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, TextPrefillStep&) const override {
    RefuseMixedBatch("glimmer has no merged multi-token decode");
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireGlimmerState(state).position();
  }

 private:
  static constexpr std::size_t kGlimmerMaxBatchLanes = 32;   ///< engine mixed-batch block limit
  static constexpr std::size_t kGlimmerMaxChunkRows = 512;   ///< engine per-step row limit
  std::shared_ptr<GlimmerModel> model_;
  std::uint32_t max_context_;
  std::shared_ptr<GlimmerTextRunnerState::SharedDraft> shared_draft_;  ///< non-null: DFlash speculative decode
  std::uint32_t max_draft_tokens_{0};
  // Turn-boundary stops from the template's turn closers.
  std::vector<std::int32_t> turn_stops_;
};

constexpr std::string_view kGemma4StateAbi = "gemma4-gfx1151-state-v1";

using Gemma4Model = models::gemma4::Model;
using Gemma4Session = models::gemma4::Session;

std::vector<std::int32_t> Gemma4EngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Gemma4 token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

std::string Gemma4Trim(std::string_view text) {
  const auto begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string_view::npos)
    return {};
  const auto end = text.find_last_not_of(" \t\r\n");
  return std::string(text.substr(begin, end - begin + 1));
}

std::string Gemma4Quote(std::string_view value) {
  std::string quoted("<|\"|>");
  quoted += value;
  quoted += "<|\"|>";
  return quoted;
}

std::string Gemma4Upper(std::string value) {
  for (char& c : value) {
    if (c >= 'a' && c <= 'z')
      c = static_cast<char>(c - 'a' + 'A');
  }
  return value;
}

// Mirror of the template's format_argument macro: strings use the <|"|>
// quote token, mappings sort keys and leave them unquoted, sequences keep
// order. Used for tool declarations and for re-rendering stored tool calls.
std::string Gemma4FormatJson(const json::Value& value) {
  if (value.is_string())
    return Gemma4Quote(value.get_str());
  if (value.is_bool())
    return value.as_bool() ? "true" : "false";
  if (value.is_null())
    return "null";
  if (value.is_number())
    return value.dump();
  if (value.is_array()) {
    std::string out("[");
    bool first = true;
    for (const auto& item : value.items()) {
      if (!first)
        out += ",";
      first = false;
      out += Gemma4FormatJson(item);
    }
    out += "]";
    return out;
  }
  if (value.is_object()) {
    std::vector<const json::Value::Member*> members;
    for (const auto& member : value.members())
      members.push_back(&member);
    std::sort(members.begin(), members.end(),
              [](const json::Value::Member* a, const json::Value::Member* b) {
                return a->first < b->first;
              });
    std::string out("{");
    bool first = true;
    for (const auto* member : members) {
      if (!first)
        out += ",";
      first = false;
      out += member->first;
      out += ":";
      out += Gemma4FormatJson(member->second);
    }
    out += "}";
    return out;
  }
  return value.dump();
}

std::optional<json::Value> Gemma4TryParseJson(std::string_view text) {
  try {
    return json::parse(text);
  } catch (...) {
    return std::nullopt;
  }
}

// Mirror of the template's format_parameters entry: description, enum, type
// and nested object shapes. Array item shapes are summarized by their type
// (the tool ladder only uses flat string/integer/enum properties).
void Gemma4RenderProperty(const std::string& name, const json::Value& schema,
                          std::string& out) {
  out += name;
  out += ":{";
  bool first = true;
  const auto comma = [&] {
    if (!first)
      out += ",";
    first = false;
  };
  if (const auto* description = schema.find("description");
      description != nullptr && description->is_string() &&
      !description->get_str().empty()) {
    comma();
    out += "description:";
    out += Gemma4Quote(description->get_str());
  }
  if (const auto* enumeration = schema.find("enum");
      enumeration != nullptr && enumeration->is_array() &&
      !enumeration->empty()) {
    comma();
    out += "enum:";
    out += Gemma4FormatJson(*enumeration);
  }
  if (const auto* properties = schema.find("properties");
      properties != nullptr && properties->is_object() &&
      !properties->empty()) {
    comma();
    out += "properties:{";
    std::vector<const json::Value::Member*> members;
    for (const auto& member : properties->members())
      members.push_back(&member);
    std::sort(members.begin(), members.end(),
              [](const json::Value::Member* a, const json::Value::Member* b) {
                return a->first < b->first;
              });
    bool inner_first = true;
    for (const auto* member : members) {
      if (!inner_first)
        out += ",";
      inner_first = false;
      Gemma4RenderProperty(member->first, member->second, out);
    }
    out += "}";
  }
  if (const auto* required = schema.find("required");
      required != nullptr && required->is_array() && !required->empty()) {
    comma();
    out += "required:[";
    bool required_first = true;
    for (const auto& item : required->items()) {
      if (!required_first)
        out += ",";
      required_first = false;
      out += Gemma4Quote(item.is_string() ? item.get_str() : item.dump());
    }
    out += "]";
  }
  if (const auto* nullable = schema.find("nullable");
      nullable != nullptr && nullable->is_bool() && nullable->as_bool()) {
    comma();
    out += "nullable:true";
  }
  if (const auto* type = schema.find("type");
      type != nullptr && type->is_string() && !type->get_str().empty()) {
    comma();
    out += "type:";
    out += Gemma4Quote(Gemma4Upper(type->get_str()));
  }
  out += "}";
}

// Mirror of the template's format_function_declaration macro (the response
// declaration has no OpenAI counterpart and is omitted).
std::string Gemma4RenderTool(const tokenization::ChatTool& tool) {
  std::string out("<|tool>declaration:");
  out += tool.name;
  out += "{description:";
  out += Gemma4Quote(tool.description);
  const auto schema = Gemma4TryParseJson(tool.parameters_json);
  if (schema && schema->is_object() && !schema->empty()) {
    out += ",parameters:{";
    bool first = true;
    if (const auto* properties = schema->find("properties");
        properties != nullptr && properties->is_object() &&
        !properties->empty()) {
      first = false;
      out += "properties:{";
      std::vector<const json::Value::Member*> members;
      for (const auto& member : properties->members())
        members.push_back(&member);
      std::sort(members.begin(), members.end(),
                [](const json::Value::Member* a, const json::Value::Member* b) {
                  return a->first < b->first;
                });
      bool property_first = true;
      for (const auto* member : members) {
        if (!property_first)
          out += ",";
        property_first = false;
        Gemma4RenderProperty(member->first, member->second, out);
      }
      out += "}";
    }
    if (const auto* required = schema->find("required");
        required != nullptr && required->is_array() && !required->empty()) {
      if (!first)
        out += ",";
      first = false;
      out += "required:[";
      bool required_first = true;
      for (const auto& item : required->items()) {
        if (!required_first)
          out += ",";
        required_first = false;
        out += Gemma4Quote(item.is_string() ? item.get_str() : item.dump());
      }
      out += "]";
    }
    if (const auto* type = schema->find("type");
        type != nullptr && type->is_string() && !type->get_str().empty()) {
      if (!first)
        out += ",";
      out += "type:";
      out += Gemma4Quote(Gemma4Upper(type->get_str()));
    }
    out += "}";
  }
  out += "}<tool|>";
  return out;
}

// Mirror of the template's strip_thinking macro for replayed assistant text.
std::string Gemma4StripThinking(std::string_view text) {
  std::string out;
  std::size_t cursor = 0;
  while (cursor <= text.size()) {
    const auto end = text.find("<channel|>", cursor);
    const auto part = text.substr(
        cursor, end == std::string_view::npos ? end : end - cursor);
    const auto channel = part.find("<|channel>");
    out += channel == std::string_view::npos ? part : part.substr(0, channel);
    if (end == std::string_view::npos)
      break;
    cursor = end + sizeof("<channel|>") - 1;
  }
  return Gemma4Trim(out);
}

// Canonical Gemma 4 chat framing with enable_thinking=false: a system turn
// carrying tool declarations, per-role turns, tool calls inline on the
// assistant turn with forward-scanned tool responses (OpenAI Chat
// Completions shape, as the template's tool_calls branch does), and the
// empty-thought generation prompt.
std::string Gemma4RenderPrompt(
    std::span<const tokenization::ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools,
    bool empty_thought_prompt) {
  std::string out;
  std::size_t body_begin = 0;
  std::string system_text;
  if (!messages.empty() &&
      (messages.front().role == tokenization::ChatRole::kSystem ||
       messages.front().role == tokenization::ChatRole::kDeveloper)) {
    system_text = Gemma4Trim(messages.front().content +
                             messages.front().framing_suffix);
    body_begin = 1;
  }
  if (!system_text.empty() || !tools.empty()) {
    out += "<|turn>system\n";
    out += system_text;
    for (const auto& tool : tools)
      out += Gemma4RenderTool(tool);
    out += "<turn|>\n";
  }
  std::string previous_non_tool_role;
  for (std::size_t i = body_begin; i < messages.size(); ++i) {
    const auto& message = messages[i];
    if (message.role == tokenization::ChatRole::kTool) {
      // Standalone tool messages (no preceding assistant call) render
      // nothing, matching the template's tool-only forward scan.
      continue;
    }
    const std::string role =
        std::string(tokenization::ToString(message.role));
    const std::string turn_role = role == "assistant" ? "model" : role;
    const bool continue_model_turn =
        turn_role == "model" && previous_non_tool_role == "assistant";
    if (!continue_model_turn) {
      out += "<|turn>";
      out += turn_role;
      out += "\n";
    }
    if (message.role == tokenization::ChatRole::kAssistant) {
      for (const auto& call : message.tool_calls) {
        out += "<|tool_call>call:";
        out += call.name;
        out += "{";
        bool first = true;
        for (const auto& argument : call.arguments) {
          if (!first)
            out += ",";
          first = false;
          out += argument.name;
          out += ":";
          out += argument.is_string ? Gemma4Quote(argument.value)
                                    : argument.value;
        }
        out += "}<tool_call|>";
      }
      bool saw_tool_response = false;
      std::size_t next = i + 1;
      for (; next < messages.size() &&
             messages[next].role == tokenization::ChatRole::kTool;
           ++next) {
        const auto& follow = messages[next];
        std::string tool_name = follow.name;
        if (!follow.tool_call_id.empty()) {
          for (const auto& call : message.tool_calls) {
            if (call.id == follow.tool_call_id) {
              tool_name = call.name;
              break;
            }
          }
        }
        if (tool_name.empty())
          tool_name = "unknown";
        out += "<|tool_response>response:";
        out += tool_name;
        out += "{value:";
        out += Gemma4Quote(Gemma4Trim(follow.content));
        out += "}<tool_response|>";
        saw_tool_response = true;
      }
      out += Gemma4StripThinking(message.content);
      if (!message.tool_calls.empty() && !saw_tool_response) {
        // History ends mid-call: the template opens the tool response the
        // model would write next instead of closing the turn.
        out += "<|tool_response>";
      } else {
        out += "<turn|>\n";
      }
      i = next - 1;
    } else {
      out += Gemma4Trim(message.content + message.framing_suffix);
      out += "<turn|>\n";
    }
    previous_non_tool_role = role;
  }
  // The empty thought channel belongs to the 12B / 26B-A4B templates only
  // (see Model::EmptyThoughtGenerationPrompt); E2B / E4B end at the turn open.
  out += "<|turn>model\n";
  if (empty_thought_prompt)
    out += "<|channel>thought\n<channel|>";
  return out;
}

class Gemma4TextRunnerState final : public TextRunnerState {
 public:
  explicit Gemma4TextRunnerState(const std::shared_ptr<Gemma4Model>& model,
                                const std::string& mtp_model_path = {}) {
    std::string error;
    session_ = model->CreateSession(&error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create Gemma4 session: " + error);
    }
    if (!mtp_model_path.empty() &&
        !session_->EnableMtp(mtp_model_path, &error)) {
      throw std::runtime_error("Failed to enable Gemma4 MTP: " + error);
    }
  }

  void SetCancellationCheck(const CancellationCheck&) override {}
  void Invalidate() noexcept override { position_ = 0; }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {};
  }

  [[nodiscard]] Gemma4Session& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

 private:
  std::unique_ptr<Gemma4Session> session_;
  std::size_t position_{0};
};

Gemma4TextRunnerState& RequireGemma4State(TextRunnerState& state) {
  auto* gemma = dynamic_cast<Gemma4TextRunnerState*>(&state);
  if (gemma == nullptr) {
    throw std::logic_error("text runner state is not Gemma4");
  }
  return *gemma;
}

const Gemma4TextRunnerState& RequireGemma4State(const TextRunnerState& state) {
  const auto* gemma = dynamic_cast<const Gemma4TextRunnerState*>(&state);
  if (gemma == nullptr) {
    throw std::logic_error("text runner state is not Gemma4");
  }
  return *gemma;
}

class Gemma4TextRunner final : public HipTextModelRunner {
 public:
  // Batch contract (see TextModelRunner): serial DecodeBatch/AdvanceBatch;
  // native merged plain-advance + prefill chunk (one layer pass carries the
  // decode lanes and the chunk); MTP multi-token decode refuses the merge.
  MBUFO_RUNNER_SERIAL_DECODE_BATCH();
  MBUFO_RUNNER_SERIAL_ADVANCE_BATCH();

  // Merged plain-advance + prefill chunk. Output equals Prefill (Sync) of the
  // chunk followed by the serial Advance of every lane. Any engine refusal
  // happens before state is touched, so it surfaces as MixedBatchRefused and
  // the scheduler runs the legacy pair. MTP sessions are refused by the engine.
  [[nodiscard]] bool AdvanceBatchWithPrefillChunk(
      std::span<const TextRunnerAdvance> advances,
      std::span<std::exception_ptr> advance_failures,
      TextRunnerState& prefill_state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens,
      TextPrefillStep& chunk_step) const override {
    auto& chunk = RequireGemma4State(prefill_state);
    if (use_mtp() || advances.empty() || offset != chunk.position() ||
        offset >= prompt.size() || max_input_tokens == 0)
      throw MixedBatchRefused("gemma4 mixed batch preconditions unmet");
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset,
         Gemma4Session::MaxChunkTokens()});
    const std::size_t next_position = offset + consumed;
    if (next_position > max_context_)
      throw MixedBatchRefused("gemma4 mixed batch prompt exceeds context");
    std::vector<Gemma4Session*> lanes;
    std::vector<std::int32_t> tokens;
    lanes.reserve(advances.size());
    tokens.reserve(advances.size());
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max()))
        throw MixedBatchRefused("gemma4 mixed batch token exceeds range");
      auto& lane = RequireGemma4State(advance.state.get());
      if (&lane == &chunk)
        throw MixedBatchRefused("gemma4 mixed batch lane is the chunk state");
      lanes.push_back(&lane.session());
      tokens.push_back(static_cast<std::int32_t>(advance.token));
    }
    const auto prefix = Gemma4EngineTokens(prompt.first(next_position));
    std::string error;
    if (!Gemma4Session::DecodeBatchWithChunk(lanes, tokens, &chunk.session(),
                                             prefix, offset, &error)) {
      throw MixedBatchRefused("gemma4 mixed batch refused cleanly: " + error);
    }
    for (std::size_t i = 0; i < advances.size(); ++i) {
      if (i < advance_failures.size()) advance_failures[i] = nullptr;
      auto& lane = RequireGemma4State(advances[i].state.get());
      lane.set_position(lane.position() + 1);
    }
    chunk.set_position(next_position);
    chunk_step = {.consumed_tokens = consumed,
                  .decode_ready = next_position == prompt.size()};
    return true;
  }

  // The engine has no MTP verify pass the chunk can ride: explicit refusal.
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatchWithPrefillChunk(
      std::span<const TextRunnerDecode>, TextRunnerState&,
      std::span<const TextRunnerToken>, std::size_t, std::size_t,
      TextPrefillStep&) const override {
    RefuseMixedBatch("gemma4 has no merged multi-token decode");
  }
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kGemma;
  }
  Gemma4TextRunner(std::shared_ptr<Gemma4Model> model,
                   std::uint32_t max_context,
                   std::string mtp_model_path = {},
                   std::uint32_t max_draft_tokens = 0)
      : model_(std::move(model)),
        max_context_(max_context),
        mtp_model_path_(std::move(mtp_model_path)),
        max_draft_tokens_(max_draft_tokens) {}

  // MTP sidecar decoding is on when a draft model path and a positive draft
  // limit were both supplied (mirrors --speculative mtp --mtp-model).
  [[nodiscard]] bool use_mtp() const noexcept {
    return !mtp_model_path_.empty() && max_draft_tokens_ > 0;
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kGemma4StateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = false,
                .fork = false,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_mtp(),
                .batched_multi_token_decode = false,
                .batched_multi_token_decode_max_width = 0,
                .prefix_reuse = false,
                .mixed_prefill_decode = !use_mtp(),
            },
        .persistence = std::nullopt,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              model_->TokenText(static_cast<std::int32_t>(id)),
              model_->IsStopToken(static_cast<std::int32_t>(id))};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return model_->VocabSize();
  }

  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"", "", {}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    std::vector<TextRunnerToken> tokens;
    for (const std::int32_t id :
           model_->tokenizer().Encode(text, [] {
             tokenization::TokenizerOptions o;
             o.add_bos = true;
             return o;
           }())) {
      if (id < 0) {
        throw std::invalid_argument("Gemma4 tokenizer returned a token ID "
                                    "outside the engine range");
      }
      tokens.push_back(static_cast<TextRunnerToken>(id));
    }
    return tokens;
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    const auto tools =
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools};
    const std::string prompt = Gemma4RenderPrompt(
        request.messages, tools, model_->EmptyThoughtGenerationPrompt());
    if (prompt.empty())
      return std::nullopt;
    return Tokenize(prompt);
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest&) const override {
    return TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    const auto ids = Gemma4EngineTokens(tokens);
    return model_->Decode(ids);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<Gemma4TextRunnerState>(model_, mtp_model_path_);
  }

  // Gemma4 Sync reprocesses the prompt from position 0 (no prefix reuse in
  // the engine), so every prefill step replays the whole retained prefix.
  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& gemma = RequireGemma4State(state);
    if (offset != gemma.position()) {
      throw std::logic_error(
          "Gemma4 prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Gemma4 prefill has no remaining input");
    }
    const std::size_t consumed =
        std::min(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    if (next_position > max_context_) {
      throw std::runtime_error("Gemma4 prompt exceeds the served context");
    }
    const auto prefix = Gemma4EngineTokens(prompt.first(next_position));
    std::string error;
    if (!gemma.session().Sync(prefix, &error)) {
      gemma.set_position(0);
      throw std::runtime_error("Gemma4 prefill failed: " + error);
    }
    gemma.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& gemma = RequireGemma4State(state);
    if (gemma.position() >= max_context_)
      return {.stop = true, .piece = {}};
    const auto logits = gemma.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error("Gemma4 token selection has no logits");
    }
    // Logits live in device memory (unified APU address space); the host
    // sampler must read a host copy, exactly like the prompt CLI does.
    // Blocking D2H on the null stream serializes after the engine's work.
    thread_local std::vector<float> host_logits;
    if (host_logits.size() != logits.size())
      host_logits.resize(logits.size());
    if (hipMemcpy(host_logits.data(), logits.data(),
                  logits.size() * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      throw std::runtime_error("Gemma4 logit download failed");
    }
    const auto token =
        static_cast<std::int32_t>(sampler.Sample(host_logits));
    if (gemma.stop_at_eos() && model_->IsStopToken(token)) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->TokenText(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Gemma4 token ID exceeds engine range");
    }
    auto& gemma = RequireGemma4State(state);
    std::string error;
    if (!gemma.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error("Gemma4 decode failed: " + error);
    }
    gemma.set_position(gemma.position() + 1);
  }

  // Serial speculative path: one engine SpeculativeRound per scheduler step.
  // Served only for pure-greedy requests (temperature 0, no penalties,
  // constraints, forced tokens or reasoning budgets — the sampler would
  // otherwise disagree with the engine-side argmax verify). Anything else
  // falls back to the single-token base path. The pool Accepts published
  // selections into the sampler itself, so this override advances only the
  // engine and the state position.
  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp() || sampler.ForcedToken().has_value() ||
        !sampler.config().can_use_unmodified_argmax() ||
        sampler.config().constraint != nullptr ||
        sampler.NeedsConstraintMask()) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "text runner decode step budget must be at least one token");
    }
    auto& gemma = RequireGemma4State(state);
    if (gemma.position() >= max_context_) {
      return {.stop = true};
    }
    // Block size never exceeds the scheduler budget (the first selection
    // replays the already-published preview token).
    const std::uint32_t n_draft = static_cast<std::uint32_t>(
        std::min<std::size_t>(max_draft_tokens_, max_tokens - 1));
    std::vector<std::int32_t> block;
    std::string error;
    if (!gemma.session().SpeculativeRound(0, n_draft, &block, &error) ||
        block.empty()) {
      throw std::runtime_error("Gemma4 speculative decode failed: " + error);
    }
    TextDecodeStep step;
    step.draft_rounds = 1;
    step.draft_tokens = n_draft;
    step.draft_accepted_tokens = block.size() - 1;
    bool stop = false;
    step.selections.reserve(block.size());
    for (const std::int32_t tok : block) {
      // A stop token ends the sequence without being published, matching the
      // single-token path (the engine has still consumed it, so the state
      // position advances past it below).
      if (gemma.stop_at_eos() && model_->IsStopToken(tok)) {
        stop = true;
        break;
      }
      step.selections.push_back(TextDecodeSelection{
          .stop = false,
          .token = static_cast<TextRunnerToken>(tok),
          .piece = model_->TokenText(tok),
      });
    }
    gemma.set_position(gemma.position() + block.size());
    step.stop = stop;
    return step;
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireGemma4State(state).position();
  }

 private:
  std::shared_ptr<Gemma4Model> model_;
  std::uint32_t max_context_;
  std::string mtp_model_path_;
  std::uint32_t max_draft_tokens_;
};
#endif

}  // namespace

struct InferenceBackend::Impl {
#if defined(ENGINE_ENABLE_HIP)
  struct State {
    std::shared_ptr<TextGenerationScheduler> scheduler;
    std::string model_id;
    std::vector<std::string> extra_model_ids;
    SamplingDefaults sampling_defaults;
    std::uint32_t max_context{0};
    bool supports_images{false};
    ReasoningOptions reasoning_defaults;
    std::string model_path;
    std::string build_info;
  };

  class ScheduledGenerationRequest final : public GenerationRequest {
  public:
    ScheduledGenerationRequest(
        std::shared_ptr<const State> model_state,
        TextGenerationScheduler::Request scheduled_request,
        InitialOutputState initial = InitialOutputState::kContent,
        std::optional<sampling::JsonConstraint::ToolFormat> tool_format = {})
        : state_(std::move(model_state)),
          request_(std::move(scheduled_request)),
          tool_format_(tool_format) {
      if (initial == InitialOutputState::kReasoning)
        reasoning_end_ =
            state_->scheduler->runner().TokenizeControl("</think>");
    }

    Result Wait(const TokenCallback& on_token,
                const ProgressCallback& on_progress,
                const StartCallback& on_start) override {
      auto result = request_.Wait(on_token, on_progress, on_start);
      if (!reasoning_end_.empty()) {
        const auto end =
            std::search(result.tokens.begin(), result.tokens.end(),
                        reasoning_end_.begin(), reasoning_end_.end());
        result.reasoning_tokens =
            static_cast<std::size_t>(end - result.tokens.begin());
      }
      return result;
    }

    void Cancel() noexcept override { request_.Cancel(); }

    std::optional<sampling::JsonConstraint::ToolFormat> ToolFormat()
        const override {
      return tool_format_;
    }

  private:
    std::shared_ptr<const State> state_;
    TextGenerationScheduler::Request request_;
    std::vector<tokenization::TokenId> reasoning_end_;
    const std::optional<sampling::JsonConstraint::ToolFormat> tool_format_;
  };

  [[nodiscard]] std::shared_ptr<const State> Snapshot() const {
    const std::lock_guard<std::mutex> lock(state_mutex);
    return state;
  }

  Result GenerateScheduled(
      std::shared_ptr<const State> current,
      std::vector<TextRunnerToken> prompt_tokens,
      Clock::time_point request_start, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled, const TokenCallback& on_token,
      std::string client_id,
      std::shared_ptr<const TextPromptContext> context = {},
      bool cache_prompt = true, std::size_t cache_prefix_tokens = 0,
      const std::vector<std::string>& stop_sequences = {},
      InitialOutputState initial = InitialOutputState::kContent) const {
    Result result;
    result.prompt_tokens = prompt_tokens.size();
    result.client_id = client_id.empty() ? "anonymous" : client_id;
    if (current == nullptr || prompt_tokens.empty()) {
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      return result;
    }

    auto request = current->scheduler->Submit(
        std::move(prompt_tokens), max_tokens, sampling, is_cancelled,
        static_cast<bool>(on_token),
        TextRequestMetadata{
            .client_id = std::move(client_id),
            .deadline = std::nullopt,
            .request_start = request_start,
            .prompt_context = std::move(context),
            .cache_prompt = cache_prompt,
            .cache_prefix_tokens = cache_prefix_tokens,
            .stop_sequences = stop_sequences,
        });
    ScheduledGenerationRequest generation(std::move(current),
                                          std::move(request), initial);
    return generation.Wait(on_token, {}, {});
  }

  mutable std::mutex state_mutex;
  std::shared_ptr<const State> state;
  // Encoder runners (encoder_runner.hpp). A dedicated encoder artifact
  // (EmbeddingGemma 2, bge-m3, bge-reranker) installs its runner at load with
  // a null key; Qwen embedding / reranking runners are built lazily on the
  // shared Qwen weights and keyed by the model identity, so a reload retires
  // the stale one. encoder_mutex guards the pointers; each runner serializes
  // its own device use.
  mutable std::mutex encoder_mutex;
  std::unique_ptr<EmbeddingRunner> embedder;
  const void* embedder_key{nullptr};
  std::unique_ptr<RerankRunner> reranker;
  const void* reranker_key{nullptr};
  std::string device_target{"gpu"};
#endif
};

InferenceBackend::InferenceBackend() : impl_(std::make_unique<Impl>()) {}

InferenceBackend::~InferenceBackend() {
#if defined(ENGINE_ENABLE_HIP)
  const bool lost = device_lost();
  impl_.reset();
  if (!lost) {
    (void)hipDeviceSynchronize();
    hipMemPool_t pool = nullptr;
    if (hipDeviceGetDefaultMemPool(&pool, 0) == hipSuccess && pool != nullptr) {
      (void)hipMemPoolTrimTo(pool, 0);
    }
  }
#else
  impl_.reset();
#endif
#if defined(__GLIBC__)
  ::malloc_trim(0);
#endif
}

// MoE engines must declare their geometry, expert formats and decode-kernel origin (core/engine/moe_contract.hpp). The registry enforces it
// for models it creates; serve loads engines directly, so the same check runs here on the loaded model, and the declaration is logged.
bool EnforceMoeContract(const char* engine, const core::engine::MoeGeometry& geometry,
                        const core::engine::MoeKernelInfo& kernels, std::string* error) {
  const auto report = core::engine::ValidateMoeContract(geometry, kernels);
  if (!report.ok()) {
    SetError(error, std::string(engine) + " violates the MoE contract: " + report.problems.front());
    return false;
  }
  if (geometry.moe_layers != 0) {
    std::cerr << "[moe-contract] " << engine << ": " << geometry.moe_layers << " MoE layers, " << geometry.experts << " experts top-"
              << geometry.top_k << ", kernels " << core::engine::MoeKernelOriginName(kernels.origin) << ", routed bytes/token "
              << static_cast<double>(core::engine::ExpectedRoutedBytesPerToken(geometry)) / 1e9 << " GB\n";
  }
  return true;
}

bool InferenceBackend::load(const std::string& model_path, std::string* error,
                            std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            const TextSpeculativeConfig& speculative_config,
                            const TextDiskCacheConfig& disk_cache_config,
                            const std::string& vision_model_path,
                            TextRunnerRamCacheOptions ram_cache_config,
                            const std::string& kv_cache_quant,
                            bool enable_yarn,
                            bool kv_cache_quant_supplied,
                            const std::string& device_target,
                            const std::string& npu_quant) {
#ifndef MBUFO_ENABLE_NPU
  (void)npu_quant;
#endif
#if defined(ENGINE_ENABLE_HIP)
  TextDiskCacheConfig resolved_disk_cache_config = disk_cache_config;
  const std::string target_dev =
      device_target.empty() ? std::string("gpu") : device_target;
  {
    const std::lock_guard<std::mutex> lock(impl_->state_mutex);
    impl_->device_target = target_dev;
  }
  std::string load_error;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
  if (reader_owner == nullptr) {
    SetError(error, "Failed to open GGUF: " + load_error);
    return false;
  }
  const std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  const auto architecture =
      reader->GetMetadataString("general.architecture").value_or("");
  if (target_dev == "npu" && architecture != "embedding_gemma2" &&
      architecture != "bert" && architecture != "qwen3") {
    SetError(error,
             "NPU device is only supported for bge-m3, bge-reranker-v2-m3, "
             "embeddinggemma-2, Qwen3-Embedding and Qwen3-Reranker GGUFs "
             "(got architecture '" +
                 std::string(architecture) + "')");
    return false;
  }
  if (enable_yarn && architecture != "qwen4exp") {
    SetError(error, "--enable-yarn applies only to Qwen3.8-Flash-Next");
    return false;
  }
  // Read once, reused below to derive YaRN from --context for qwen4exp.
  const std::uint32_t qwen4exp_native_context =
      architecture == "qwen4exp"
          ? reader->GetMetadataUint32("qwen4exp.context_length").value_or(0)
          : 0;
  if (max_context == 0) {
    const std::uint64_t native =
        architecture == "qwen4exp"
            ? qwen4exp_native_context
            : reader
                  ->GetMetadataUint64(std::string(architecture) +
                                      ".context_length")
                  .value_or(0);
    if (native < 2 || native > std::numeric_limits<std::uint32_t>::max()) {
      SetError(error,
               "GGUF has no valid native context length; specify --context");
      return false;
    }
    max_context = static_cast<std::uint32_t>(native);
  }
  // Static YaRN changes every request, so a context above Flash-Next's
  // native one needs --enable-yarn. Checked before any weights load.
  std::optional<models::qwen38_flash_next::RopeScaling> rope_scaling;
  if (architecture == "qwen4exp" &&
      !models::qwen38_flash_next::ResolveRopeScaling(
          max_context, qwen4exp_native_context, enable_yarn, &rope_scaling,
          error)) {
    return false;
  }

  if (reader->GetMetadataString("general.architecture") == "deepseek4") {
    if (!vision_model_path.empty()) {
      SetError(error, "DeepSeek does not support --mmproj");
      return false;
    }
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kDSpark) {
      SetError(error,
               "DeepSeek HTTP models support only DSpark speculative decoding");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kDSpark &&
        speculative_config.draft_model_path.empty()) {
      SetError(error, "DeepSeek DSpark HTTP decoding requires --dspark-model");
      return false;
    }
    if (!models::deepseek_v4_flash::ValidateGgufTemplate(*reader,
                                                         &load_error)) {
      SetError(error, "Unsupported DeepSeek chat template: " + load_error);
      return false;
    }
    auto model = models::deepseek_v4_flash::Model::Load(
        model_path,
        models::deepseek_v4_flash::ModelOptions{
            .max_context = max_context,
            .dspark_model_path =
                speculative_config.backend == TextSpeculativeBackend::kDSpark
                    ? speculative_config.draft_model_path
                    : std::string{},
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create DeepSeek model: " + load_error);
      return false;
    }
    if (!EnforceMoeContract("deepseek_v4_flash", models::deepseek_v4_flash::DescribeMoeGeometry(*reader), models::deepseek_v4_flash::DescribeMoeKernels(), error)) return false;
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
        !FingerprintArtifact(
            "DeepSeek", *reader,
            &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        speculative_config.backend == TextSpeculativeBackend::kDSpark &&
        resolved_disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !FingerprintArtifactFile(
            "DSpark", speculative_config.draft_model_path,
            &resolved_disk_cache_config.draft_model_artifact_fingerprint,
            error)) {
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config), ram_cache_config);
  }
  if (reader->GetMetadataString("general.architecture") == "qwen4exp") {
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kMtp) {
      SetError(error,
               "Qwen3.8-Flash-Next HTTP models support only MTP speculative "
               "decoding (--speculative mtp --mtp-model)");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        speculative_config.draft_model_path.empty()) {
      SetError(error,
               "Qwen3.8-Flash-Next MTP HTTP decoding requires --mtp-model");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        (speculative_config.max_draft_tokens == 0 ||
         speculative_config.min_draft_tokens != 1)) {
      SetError(error,
               "Flash-Next MTP requires a positive draft limit and "
               "--min-draft-tokens 1");
      return false;
    }
    // The compiled Qwen3.8 chat template renders through the artifact's
    // own tokenizer (the same vocabulary and pre-tokenizer as Qwen3.8).
    if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                              &load_error)) {
      SetError(error,
               "Unsupported Qwen3.8-Flash-Next chat template: " + load_error);
      return false;
    }
    // The model owns prefill geometry for both bulk and scheduled requests.
    auto model = models::qwen38_flash_next::Model::Load(
        model_path,
        models::qwen38_flash_next::ModelOptions{
            .max_context = max_context,
            .mtp_model_path =
                speculative_config.backend == TextSpeculativeBackend::kMtp
                    ? speculative_config.draft_model_path
                    : std::string{},
            .max_draft_tokens = speculative_config.max_draft_tokens,
            .vision_model_path = vision_model_path,
            .decode_concurrency = static_cast<std::uint32_t>(
                std::clamp<std::size_t>(session_count, 1, 8)),
            .kv_cache_quant = kv_cache_quant,
            .rope_scaling = rope_scaling,
        },
        &load_error);
    if (model == nullptr) {
      SetError(error,
               "Failed to create Qwen3.8-Flash-Next model: " + load_error);
      return false;
    }
    if (!EnforceMoeContract("qwen38_flash_next", models::qwen38_flash_next::DescribeMoeGeometry(*model), models::qwen38_flash_next::DescribeMoeKernels(), error)) return false;
    if (rope_scaling) {
      Logger::Info(
          "loader",
          "event=yarn factor=" + std::to_string(rope_scaling->factor) +
              " original_context=" + std::to_string(qwen4exp_native_context) +
              " context=" + std::to_string(max_context));
    }
    // RAM checkpoints cannot outlive this loaded model. Their process-local
    // identity is sufficient; hashing every shard here adds a complete model
    // read on each fresh container start even when disk caching is disabled.
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
        !FingerprintArtifact(
            "Qwen3.8-Flash-Next", *reader,
            &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        speculative_config.backend == TextSpeculativeBackend::kMtp &&
        resolved_disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !FingerprintArtifactFile(
            "Qwen3.8-Flash-Next MTP", speculative_config.draft_model_path,
            &resolved_disk_cache_config.draft_model_artifact_fingerprint,
            error)) {
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config), ram_cache_config);
  }
  if (reader->GetMetadataString("general.architecture") == "gemma4") {
    if (!vision_model_path.empty()) {
      SetError(error, "Gemma4 HTTP models do not support --mmproj");
      return false;
    }
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kMtp) {
      SetError(error,
               "Gemma4 HTTP models support only MTP speculative decoding "
               "(--speculative mtp --mtp-model)");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        speculative_config.draft_model_path.empty()) {
      SetError(error, "Gemma4 MTP HTTP decoding requires --mtp-model");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        (speculative_config.max_draft_tokens == 0 ||
         speculative_config.max_draft_tokens > 16)) {
      SetError(error,
               "Gemma4 MTP requires a draft limit in [1, 16] "
               "(--draft-tokens)");
      return false;
    }
    auto model = models::gemma4::Model::Load(
        model_path,
        models::gemma4::ModelOptions{
            .max_context = max_context,
            .decode_concurrency = static_cast<std::uint32_t>(
                std::clamp<std::size_t>(session_count, 1, 8)),
            .kv_cache_quant = kv_cache_quant,
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create Gemma4 model: " + load_error);
      return false;
    }
    if (!EnforceMoeContract("gemma4", models::gemma4::DescribeMoeGeometry(*model), models::gemma4::DescribeMoeKernels(), error)) return false;
    // Gemma4 sessions reprocess the prompt from position 0 and expose no
    // continuation snapshots, so there is no persistent disk-cache identity.
    if (DiskCacheEnabled(resolved_disk_cache_config)) {
      SetError(error,
               "Gemma4 HTTP models do not support the persistent disk cache");
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config), ram_cache_config);
  }
#ifdef MBUFO_ENABLE_NPU
  if (target_dev == "npu" && architecture == "qwen3") {
    if (!vision_model_path.empty()) {
      SetError(error, "Qwen3 NPU encoders do not support --mmproj");
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) ||
        speculative_config.backend != TextSpeculativeBackend::kDisabled) {
      SetError(error,
               "Qwen3 NPU encoders: no disk cache or speculative decoding");
      return false;
    }
    if (!npu_quant.empty() && npu_quant != "q8") {
      SetError(error,
               "Qwen3 NPU encoders run Q8 (bfp16ebs8) only; got "
               "npu_quant='" +
                   npu_quant + "'");
      return false;
    }
    std::string ident(reader->GetMetadataString("general.name").value_or(""));
    ident += ' ';
    ident += reader->GetMetadataString("general.basename").value_or("");
    for (auto& ch : ident) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    const bool qwen_is_embed =
        reader->GetMetadataUint32("qwen3.pooling_type").has_value();
    const bool qwen_is_rerank =
        !qwen_is_embed && ident.find("rerank") != std::string::npos;
    if (!qwen_is_embed && !qwen_is_rerank) {
      SetError(error,
               "NPU qwen3 supports only Qwen3-Embedding (pooling_type "
               "declared) and Qwen3-Reranker GGUFs");
      return false;
    }
    using NpuEnc = mbufo::core::npu::NpuQwen3Encoder;
    const auto qwen_role =
        qwen_is_embed ? NpuEnc::Role::kEmbedding : NpuEnc::Role::kReranker;
    const std::string qwen_id =
        qwen_is_embed ? "qwen3-embedding" : "qwen3-reranker";
    std::unique_ptr<NpuEnc> enc;
    auto qwen_slot = std::make_shared<NpuEnc*>(nullptr);
    std::string npu_err;
    const bool q_ok = LoadNpuServable(
        qwen_id,
        [&](std::string* e) {
          enc = NpuEnc::Load(model_path, qwen_role,
                             mbufo::core::npu::graph::LinearBackend::kNpuQ8, e);
          *qwen_slot = enc.get();
          return enc != nullptr;
        },
        [&](std::string* e) {
          if (qwen_is_embed) {
            std::vector<float> v;
            double ms = 0.0;
            std::size_t toks = 0;
            return enc->Embed("npu warmup", &v, &toks, &ms, e);
          }
          float score = 0.0F;
          double ms = 0.0;
          std::size_t toks = 0;
          return enc->ScorePair("warmup query", "warmup document", "", &score,
                                &toks, &ms, e);
        },
        [qwen_slot]() {
          if (*qwen_slot != nullptr)
            (*qwen_slot)->Release();
        },
        &npu_err);
    if (!q_ok) {
      SetError(error, "Qwen3 NPU load failed: " + npu_err);
      return false;
    }
    Logger::Info("loader", "event=npu_loaded model=" + qwen_id +
                               " runtime=NpuOpGraph quant=q8");
    const std::uint32_t qwen_cap =
        static_cast<std::uint32_t>(enc->max_tokens());
    if (max_context == 0 || max_context > qwen_cap)
      max_context = qwen_cap;
    {
      auto new_state = std::make_shared<Impl::State>();
      new_state->model_id = qwen_id;
      new_state->max_context = max_context;
      new_state->model_path = model_path;
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    {
      const std::lock_guard<std::mutex> lock(impl_->encoder_mutex);
      impl_->embedder.reset();
      impl_->reranker.reset();
      impl_->embedder_key = nullptr;
      impl_->reranker_key = nullptr;
      if (qwen_is_embed)
        impl_->embedder = MakeNpuQwenEmbeddingRunner(std::move(enc));
      else
        impl_->reranker = MakeNpuQwenRerankRunner(std::move(enc));
    }
    return true;
  }
#endif
  if (architecture == "embedding_gemma2") {
    if (!vision_model_path.empty()) {
      SetError(error, "EmbeddingGemma 2 does not support --mmproj");
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) ||
        speculative_config.backend != TextSpeculativeBackend::kDisabled) {
      SetError(error,
               "EmbeddingGemma 2 is an embedding encoder: no disk cache or "
               "speculative decoding");
      return false;
    }
    std::string eg_error;
    auto eg_model =
        models::embeddinggemma2::EngineModel::Create(model_path, &eg_error);
    if (eg_model == nullptr) {
      SetError(error, "Failed to create EmbeddingGemma 2 model: " + eg_error);
      return false;
    }
    // NPU path: upload weights & kernels when target_dev == "npu".
    if (target_dev == "npu") {
#ifdef MBUFO_ENABLE_NPU
      if (!npu_quant.empty()) {
        if (npu_quant == "bf16") {
          ::unsetenv("MBUFO_NPU_QUANT");
        } else {
          ::setenv("MBUFO_NPU_QUANT", npu_quant.c_str(), 1);
        }
      }
      std::string npu_err;
      const bool eg_ok = LoadNpuServable(
          "embeddinggemma-2",
          [&](std::string* e) {
            if (!eg_model->LoadNpuWeights("", e))
              return false;
            eg_model->SetNpuScheduler(&GlobalNpuScheduler());
            eg_model->SetNpuLaneConfig(NpuLaneConfigForDevice("npu"));
            return true;
          },
          [&](std::string* e) {
            const std::vector<std::uint32_t> ids{2, 4, 5, 6, 7, 8, 9, 10};
            return eg_model->Embed(ids, e).has_value();
          },
          [m = eg_model.get()]() { m->UnloadNpuWeights(); }, &npu_err);
      if (!eg_ok) {
        SetError(error, "EmbeddingGemma 2 NPU weight load failed: " + npu_err);
        return false;
      }
      const bool eg_graph = mbufo::core::npu::EmbGemma2UseGraph();
      Logger::Info("loader",
                   "event=npu_loaded model=embeddinggemma-2 runtime=" +
                       std::string(eg_graph ? "NpuOpGraph quant=q8"
                                            : "NpuEmbGemma2Runtime quant=" +
                                                  (npu_quant.empty()
                                                       ? std::string("default")
                                                       : npu_quant)));
#else
      SetError(error,
               "NPU backend is disabled in this build (rebuild with "
               "-DMBUFO_ENABLE_NPU=ON)");
      return false;
#endif
    } else {
      eg_model->SetNpuLaneConfig(NpuLaneConfigForDevice("gpu"));
    }
    if (max_context == 0 || max_context > models::embeddinggemma2::kMaxTokens) {
      max_context = models::embeddinggemma2::kMaxTokens;
    }
    {
      auto new_state = std::make_shared<Impl::State>();
      new_state->model_id = "embeddinggemma-2";
      new_state->max_context = max_context;
      new_state->model_path = model_path;
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    {
      const std::lock_guard<std::mutex> lock(impl_->encoder_mutex);
      impl_->embedder = MakeEmbeddingGemma2Runner(std::move(eg_model));
      impl_->embedder_key = nullptr;
      impl_->reranker.reset();
      impl_->reranker_key = nullptr;
    }
    return true;
  }
  if (architecture == "bert") {
    if (!vision_model_path.empty()) {
      SetError(error, "BERT reranker models do not support --mmproj");
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config)) {
      SetError(error,
               "BERT reranker models do not support the persistent disk "
               "cache");
      return false;
    }
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled) {
      SetError(error,
               "BERT reranker models do not support speculative decoding");
      return false;
    }
    if (reader->GetMetadataUint32("bert.pooling_type").has_value()) {
      // BERT embedding encoder (e.g. bge-m3): serves /v1/embeddings.
      std::string bge_error;
      auto bge_model =
          models::bge_m3::EngineModel::Create(model_path, &bge_error);
      if (bge_model == nullptr) {
        SetError(error, "Failed to create BERT embedding model: " + bge_error);
        return false;
      }
      if (target_dev == "npu") {
#ifdef MBUFO_ENABLE_NPU
        std::string npu_err;
        const bool bge_ok = LoadNpuServable(
            "bge-m3",
            [&](std::string* e) { return bge_model->LoadNpuWeights("", e); },
            [&](std::string* e) {
              std::vector<std::uint32_t> ids{0, 6, 4, 5, 2};
              std::vector<float> cls;
              double ms = 0.0;
              if (!bge_model->RunNpuCls(ids, &cls, &ms, e))
                return false;
              ids.assign(70, 6);  // spans two 64-row chunks
              ids.front() = 0;
              ids.back() = 2;
              return bge_model->RunNpuCls(ids, &cls, &ms, e);
            },
            [m = bge_model.get()]() { m->UnloadNpuWeights(); }, &npu_err);
        if (!bge_ok) {
          SetError(error, "bge-m3 NPU weight load failed: " + npu_err);
          return false;
        }
        Logger::Info(
            "loader",
            "event=npu_loaded model=bge-m3 runtime=NpuOpGraph quant=q8");
#else
        SetError(error,
                 "NPU backend is disabled in this build (rebuild with "
                 "-DMBUFO_ENABLE_NPU=ON)");
        return false;
#endif
      }
      // NPU scheduler/lane attach to the EngineSession (see Embed() below);
      // SetNpuLaneConfig lives on EngineSession, not EngineModel.
      const std::uint32_t native_context =
          bge_model->Config().max_position_embeddings;
      if (max_context == 0 || max_context > native_context) {
        max_context = native_context;
      }
      if (max_context < 2) {
        SetError(error, "HTTP context exceeds the BERT embedding context");
        return false;
      }
      {
        auto new_state = std::make_shared<Impl::State>();
        new_state->model_id = bge_model->Config().model_name;
        new_state->max_context = max_context;
        new_state->model_path = model_path;
        const std::lock_guard<std::mutex> lock(impl_->state_mutex);
        impl_->state = std::move(new_state);
      }
      {
        std::function<void(models::bge_m3::EngineSession&)> configure;
#ifdef MBUFO_ENABLE_NPU
        if (target_dev == "npu") {
          configure = [](models::bge_m3::EngineSession& session) {
            session.SetNpuScheduler(&GlobalNpuScheduler());
            session.SetNpuLaneConfig(NpuLaneConfigForDevice("npu"));
          };
        }
#endif
        const std::lock_guard<std::mutex> lock(impl_->encoder_mutex);
        impl_->embedder = MakeBgeM3EmbeddingRunner(
            std::move(bge_model), /*serial_only=*/target_dev == "npu",
            std::move(configure));
        impl_->embedder_key = nullptr;
        impl_->reranker.reset();
        impl_->reranker_key = nullptr;
      }
      return true;
    }
    // Rerank-only encoder model: no text runner, no scheduler, no disk
    // cache. chat/complete refuse explicitly; Rerank + /v1/reranking serve
    // it. served_model_names/build_info are set by the caller as usual.
    std::string rerank_error;
    auto rerank_model =
        models::bge_reranker::EngineModel::Create(model_path, &rerank_error);
    if (rerank_model == nullptr) {
      SetError(error,
               "Failed to create BERT reranker model: " + rerank_error);
      return false;
    }
    if (target_dev == "npu") {
#ifdef MBUFO_ENABLE_NPU
      std::string npu_err;
      const bool rr_ok = LoadNpuServable(
          "bge-reranker",
          [&](std::string* e) { return rerank_model->LoadNpuWeights("", e); },
          [&](std::string* e) {
            std::vector<std::uint32_t> ids{0, 6, 4, 2, 2, 5, 7, 2};
            std::vector<float> cls;
            double ms = 0.0;
            return rerank_model->Encoder().RunNpuCls(ids, &cls, &ms, e);
          },
          [m = rerank_model.get()]() { m->UnloadNpuWeights(); }, &npu_err);
      if (!rr_ok) {
        SetError(error, "bge-reranker NPU weight load failed: " + npu_err);
        return false;
      }
      Logger::Info(
          "loader",
          "event=npu_loaded model=bge-reranker runtime=NpuOpGraph quant=q8");
#else
      SetError(error,
               "NPU backend is disabled in this build (rebuild with "
               "-DMBUFO_ENABLE_NPU=ON)");
      return false;
#endif
    }
    // NPU scheduler attaches when the rerank session is created (see below);
    // SetNpuLaneConfig lives on EngineSession, not EngineModel.
    const std::uint32_t native_context =
        rerank_model->Config().max_position_embeddings;
    if (max_context == 0) {
      max_context = native_context;
    }
    if (max_context < 2 || max_context > native_context) {
      SetError(error, "HTTP context exceeds the BERT reranker context");
      return false;
    }
    {
      auto new_state = std::make_shared<Impl::State>();
      new_state->model_id = rerank_model->Config().model_name;
      new_state->max_context = max_context;
      new_state->model_path = model_path;
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    {
      std::function<void(models::bge_reranker::EngineSession&)> configure;
#ifdef MBUFO_ENABLE_NPU
      if (target_dev == "npu") {
        configure = [](models::bge_reranker::EngineSession& session) {
          session.SetNpuScheduler(&GlobalNpuScheduler());
          session.SetNpuLaneConfig(NpuLaneConfigForDevice("npu"));
        };
      }
#endif
      const std::lock_guard<std::mutex> lock(impl_->encoder_mutex);
      impl_->reranker = MakeBgeRerankRunner(std::move(rerank_model),
                                            std::move(configure));
      impl_->reranker_key = nullptr;
      impl_->embedder.reset();
      impl_->embedder_key = nullptr;
    }
    return true;
  }

  if (reader->GetMetadataString("general.architecture") == "muse-glimmer") {
    if (!vision_model_path.empty()) {
      SetError(error, "Glimmer HTTP models do not support --mmproj");
      return false;
    }
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kDFlash) {
      SetError(error,
               "Glimmer HTTP models support only DFlash speculative decoding "
               "(--speculative dflash2 --dflash-model PATH) or --speculative off");
      return false;
    }
    auto model = models::glimmer::Model::Load(
        model_path,
        models::glimmer::ModelOptions{
            .max_context = max_context,
            .kv_cache_quant = kv_cache_quant,
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create Glimmer model: " + load_error);
      return false;
    }
    // Glimmer sessions expose no continuation snapshots, so there is no
    // persistent disk-cache identity; the RAM pool owns every session.
    if (DiskCacheEnabled(resolved_disk_cache_config)) {
      SetError(error,
               "Glimmer HTTP models do not support the persistent disk cache");
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config), ram_cache_config);
  }

  if (reader->GetMetadataString("general.architecture") == "laguna") {
    if (!vision_model_path.empty()) {
      SetError(error, "Laguna HTTP models do not support --mmproj");
      return false;
    }
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled) {
      SetError(error, "Laguna HTTP models do not support speculative decoding yet (pass --speculative off)");
      return false;
    }
    auto model = models::laguna::Model::Load(
        model_path,
        models::laguna::ModelOptions{
            .max_context = max_context,
            .kv_cache_quant = kv_cache_quant,
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create Laguna model: " + load_error);
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config)) {
      SetError(error, "Laguna HTTP models do not support the persistent disk cache");
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config), ram_cache_config);
  }


  if (reader->GetMetadataString("general.architecture") == "inkling2") {
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kDSpark &&
        speculative_config.backend != TextSpeculativeBackend::kMtp) {
      SetError(error,
               "Inkling HTTP models support only DSpark speculative decoding "
               "(--speculative dspark --dspark-model PATH) or --speculative off");
      return false;
    }
    const bool want_dspark =
        speculative_config.backend == TextSpeculativeBackend::kDSpark ||
        speculative_config.backend == TextSpeculativeBackend::kMtp;
    if (want_dspark && speculative_config.draft_model_path.empty()) {
      SetError(error, "Inkling DSpark HTTP decoding requires --dspark-model (or --mtp-model)");
      return false;
    }
    auto model = models::inkling::Model::Load(
        model_path,
        models::inkling::ModelOptions{
            .max_context = max_context,
            .kv_cache_quant = kv_cache_quant,
            .kv_cache_quant_supplied = kv_cache_quant_supplied,
            .vision_model_path = vision_model_path,
            .dspark_model_path =
                want_dspark ? speculative_config.draft_model_path
                            : std::string{},
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create Inkling model: " + load_error);
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config)) {
      SetError(error, "Inkling HTTP models do not support the persistent disk cache");
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config), ram_cache_config);
  }

  if (reader->GetMetadataString("general.architecture") == "llama") {
    if (!vision_model_path.empty()) {
      SetError(error, "Llama HTTP models do not support --mmproj");
      return false;
    }
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled) {
      SetError(error,
               "Llama HTTP models do not support speculative decoding "
               "(pass --speculative off)");
      return false;
    }
    auto model = models::llama::Model::Load(
        model_path,
        models::llama::ModelOptions{
            .max_context = max_context,
            .kv_cache_quant = kv_cache_quant,
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create Llama model: " + load_error);
      return false;
    }
    std::cerr << "[llama] kv_cache_quant=" << kv_cache_quant << '\n';
    // Llama sessions expose no continuation snapshots, so there is no
    // persistent disk-cache identity; the RAM pool owns every session.
    if (DiskCacheEnabled(resolved_disk_cache_config)) {
      SetError(error,
               "Llama HTTP models do not support the persistent disk cache");
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config), ram_cache_config);
  }

  if (reader->GetMetadataString("general.architecture") == "gpt-oss") {
    if (!vision_model_path.empty()) {
      SetError(error, "GPT-OSS HTTP models do not support --mmproj");
      return false;
    }
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled) {
      SetError(error,
               "GPT-OSS HTTP models do not support speculative decoding "
               "(MTP and draft backends have no serve path; pass "
               "--speculative off)");
      return false;
    }
    auto model = models::gptoss::Model::Load(
        model_path,
        models::gptoss::ModelOptions{
            .max_context = max_context,
            .kv_cache_quant = kv_cache_quant,
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create GPT-OSS model: " + load_error);
      return false;
    }
    if (!EnforceMoeContract("gptoss", models::gptoss::DescribeMoeGeometry(*model), models::gptoss::DescribeMoeKernels(), error)) return false;
    std::cerr << "[gpt-oss] kv_cache_quant=" << kv_cache_quant << '\n';
    // GPT-OSS sessions expose no continuation snapshots, so there is no
    // persistent disk-cache identity; the RAM pool owns every session.
    if (DiskCacheEnabled(resolved_disk_cache_config)) {
      SetError(error,
               "GPT-OSS HTTP models do not support the persistent disk cache");
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config), ram_cache_config);
  }
  std::shared_ptr<models::qwen::vision::Encoder> vision;
  try {
    const auto qwen_embed_len =
        reader->GetMetadataUint64("qwen35.embedding_length")
            .value_or(reader->GetMetadataUint64("qwen35moe.embedding_length")
                          .value_or(0));
    if (qwen_embed_len == 5120 || qwen_embed_len == 2048) {
      vision = models::qwen::vision::Encoder::Open(
          model_path, vision_model_path,
          static_cast<std::uint32_t>(qwen_embed_len));
    } else if (!vision_model_path.empty()) {
      throw std::invalid_argument(
          "image input supports Qwen3.8-27B, Flash-Next, and Qwen3.5/Ornith 35B-A3B");
    }
  } catch (const std::exception& e) {
    SetError(error, e.what());
    return false;
  }
  Logger::Info("loader", "event=load_phase phase=target_weights " +
                             Logger::MemoryStatus());
  auto model =
      hip::QwenGpuModel::CreateFromGguf(reader, &load_error, std::move(vision));
  if (model == nullptr) {
    SetError(error, "Failed to create GPU model: " + load_error);
    return false;
  }
  if (DiskCacheEnabled(resolved_disk_cache_config) &&
      resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
      !FingerprintArtifact(
          "Qwen", *reader,
          &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
    return false;
  }
  return load(std::move(model), error, max_context, session_count,
              prefill_policy, scheduler_policy, speculative_config,
              std::move(resolved_disk_cache_config), ram_cache_config,
              kv_cache_quant, kv_cache_quant_supplied);
#else
  (void)model_path;
  (void)max_context;
  (void)session_count;
  (void)prefill_policy;
  (void)scheduler_policy;
  (void)speculative_config;
  (void)disk_cache_config;
  (void)vision_model_path;
  (void)ram_cache_config;
  (void)enable_yarn;
  (void)kv_cache_quant_supplied;
  (void)device_target;
  (void)npu_quant;
  SetError(error, "HTTP inference requires the HIP backend");
  return false;
#endif
}

#if defined(ENGINE_ENABLE_HIP)
bool InferenceBackend::load(std::shared_ptr<const hip::QwenGpuModel> model,
                            std::string* error, std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            TextSpeculativeConfig speculative_config,
                            TextDiskCacheConfig disk_cache_config,
                            TextRunnerRamCacheOptions ram_cache_config,
                            const std::string& kv_cache_quant,
                            bool kv_cache_quant_supplied) {
  if (model == nullptr) {
    SetError(error, "Qwen GPU model must not be null");
    return false;
  }
  if (!EnforceMoeContract("qwen", models::qwen::DescribeQwenMoeGeometry(*model), models::qwen::DescribeQwenMoeKernels(), error)) return false;
  if (max_context == 0)
    max_context = model->GetConfig().context_length;
  if (max_context < 2 || max_context > model->GetConfig().context_length) {
    SetError(error, "HTTP context exceeds the loaded Qwen model context");
    return false;
  }

  if (speculative_config.backend == TextSpeculativeBackend::kDSpark) {
    SetError(error, "DSpark HTTP decoding requires a DeepSeek model");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kMtp) {
    SetError(error, "MTP HTTP decoding requires a Qwen Flash-Next model");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (!disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error, "Qwen persistent disk cache configuration is invalid");
    return false;
  }
  if (speculative_config.max_draft_tokens == 0 ||
      speculative_config.min_draft_tokens == 0 ||
      speculative_config.min_draft_tokens >
          speculative_config.max_draft_tokens) {
    SetError(error, "HTTP speculative draft limits are invalid");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kDFlash &&
      speculative_config.min_draft_tokens != 1) {
    SetError(error,
             "DFlash2 requires --min-draft-tokens 1; bound blocks with "
             "--draft-tokens");
    return false;
  }
  // Blocked-Q8 attention KV for Qwen3.8-27B-class models. An explicit
  // --kv-quant flag (or admin kv_cache_quant field) wins over the `mbufo
  // prompt` env switch (QWEN_KV_QUANT); with neither, f16 stays the default.
  // QwenGpuModel::CreateFromGguf needs no policy parameter: the policy
  // applies at executor/adapter level, not weight loading.
  auto qwen_policy = hip::QwenExecutionPolicy::Production();
  std::string qwen_kv_quant = "f16";
  const char* qwen_kv_source = "default";
  if (kv_cache_quant_supplied) {
    qwen_kv_quant = kv_cache_quant;
    qwen_kv_source = "flag";
  } else if (const char* kv_quant = std::getenv("QWEN_KV_QUANT")) {
    qwen_kv_quant = kv_quant;
    qwen_kv_source = "env";
  }
  if (qwen_kv_quant != "f16" && qwen_kv_quant != "q8") {
    SetError(error, "kv_cache_quant must be f16 or q8 (flag or QWEN_KV_QUANT)");
    return false;
  }
  if (qwen_kv_quant == "q8" && model->GetConfig().head_dim != 256) {
    qwen_kv_quant = "f16";
    qwen_kv_source = "auto-f16(head_dim!=256)";
  }
  qwen_policy.blocked_q8_attention = (qwen_kv_quant == "q8");
  std::cerr << "[qwen] kv_cache_quant=" << qwen_kv_quant
            << " source=" << qwen_kv_source << '\n';
  if (qwen_policy.blocked_q8_attention &&
      speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      speculative_config.backend != TextSpeculativeBackend::kDFlash) {
    SetError(error,
             "q8 KV cache supports only --speculative dflash2 "
             "(MTP has no blocked-KV path); pass --speculative dflash2 or off");
    return false;
  }

  try {
    std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model;
    speculative::SpeculativeOptions speculative_options;
    if (speculative_config.backend == TextSpeculativeBackend::kDFlash) {
      if (speculative_config.draft_model_path.empty()) {
        SetError(error, "DFlash HTTP decoding requires --dflash-model");
        return false;
      }
      std::string dflash_error;
      auto dflash_reader_owner = core::GgufReader::OpenFile(
          speculative_config.draft_model_path, &dflash_error);
      if (dflash_reader_owner == nullptr) {
        SetError(error, "Failed to open DFlash GGUF: " + dflash_error);
        return false;
      }
      std::shared_ptr<const core::GgufReader> dflash_reader(
          std::move(dflash_reader_owner));
      if (DiskCacheEnabled(disk_cache_config) &&
          disk_cache_config.draft_model_artifact_fingerprint.empty() &&
          !FingerprintArtifact(
              "DFlash", *dflash_reader,
              &disk_cache_config.draft_model_artifact_fingerprint, error)) {
        return false;
      }
      if (DiskCacheEnabled(disk_cache_config) &&
          !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) {
        SetError(error,
                 "Qwen DFlash persistent disk cache configuration is "
                 "invalid");
        return false;
      }
      Logger::Info("loader", "event=load_phase phase=draft_weights " +
                                 Logger::MemoryStatus());
      dflash_model = hip::QwenDFlashGpuModel::Create(std::move(dflash_reader),
                                                     model, &dflash_error);
      if (dflash_model == nullptr) {
        SetError(error, "Failed to create DFlash model: " + dflash_error);
        return false;
      }

      speculative_options.dflash_policy = speculative_config.dflash_policy;
      speculative_options.max_draft_tokens =
          speculative_config.max_draft_tokens;
      speculative_options.min_draft_tokens =
          speculative_config.min_draft_tokens;
      speculative_options.initial_draft_tokens =
          speculative_config.max_draft_tokens;
      speculative_options.use_batched_verification =
          std::getenv("MBUFO_NO_BATCHED_VERIFY") == nullptr;
      speculative_options.retain_frontier_logits = true;
      speculative_options.enable_adaptive_draft_length = false;
    }

    PrepareVision(model->VisionEncoder());
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model =
        sampling::TextPreset(model->GetConfig());
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = model->VisionEncoder() != nullptr;
    auto runner = std::make_shared<QwenTextRunner>(
        std::move(model), max_context, std::move(dflash_model),
        speculative_options, disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint, qwen_policy);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    Logger::Info("loader",
                 "event=load_phase phase=sessions " + Logger::MemoryStatus());
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache),
        ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::deepseek_v4_flash::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    TextRunnerRamCacheOptions ram_cache_config) {
  if (model == nullptr) {
    SetError(error, "DeepSeek model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->MaxContext();

  const bool use_dspark =
      speculative_config.backend == TextSpeculativeBackend::kDSpark;
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      (!use_dspark || !model->HasDspark())) {
    SetError(error, "DeepSeek DSpark requires a loaded support model");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded DeepSeek model context");
    return false;
  }
  if (use_dspark && (speculative_config.max_draft_tokens == 0 ||
                     speculative_config.min_draft_tokens == 0 ||
                     speculative_config.min_draft_tokens >
                         speculative_config.max_draft_tokens)) {
    SetError(error, "DeepSeek DSpark draft limits are invalid");
    return false;
  }
  if (use_dspark && speculative_config.min_draft_tokens != 1) {
    SetError(error,
             "DSpark uses model-owned adaptive drafting; custom draft "
             "floors are unsupported");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (use_dspark &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error, "DeepSeek persistent disk cache configuration is invalid");
    return false;
  }

  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model =
        sampling::TextModelPreset::kDeepSeekV4Flash;
    new_state->sampling_defaults.supplied = {};
    auto runner = std::make_shared<DeepSeekTextRunner>(
        std::move(model), max_context, use_dspark,
        speculative_config.max_draft_tokens,
        disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache),
        ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::qwen38_flash_next::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    TextRunnerRamCacheOptions ram_cache_config) {
  if (model == nullptr) {
    SetError(error, "Qwen3.8-Flash-Next model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->MaxContext();

  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(
        error,
        "HTTP context exceeds the loaded Qwen3.8-Flash-Next model context");
    return false;
  }
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      (speculative_config.backend != TextSpeculativeBackend::kMtp ||
       !model->HasMtp())) {
    SetError(error,
             "Flash-Next MTP requires a model loaded with its draft sidecar");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
      (speculative_config.max_draft_tokens == 0 ||
       speculative_config.min_draft_tokens != 1)) {
    SetError(error,
             "Qwen3.8-Flash-Next MTP drafts a fixed chain; custom draft "
             "floors are unsupported");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error,
             "Qwen3.8-Flash-Next persistent disk cache configuration is "
             "invalid");
    return false;
  }
  try {
    PrepareVision(model->VisionEncoder());
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model = sampling::TextModelPreset::kQwen38;
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = model->VisionEncoder() != nullptr;
    const std::string kv_cache_quant = model->KvCacheQuant();
    auto runner = std::make_shared<QwenFlashNextTextRunner>(
        std::move(model), max_context,
        speculative_config.backend == TextSpeculativeBackend::kMtp,
        speculative_config.max_draft_tokens,
        disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint, kv_cache_quant);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache),
        ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::glimmer::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    TextRunnerRamCacheOptions ram_cache_config) {
  if (model == nullptr) {
    SetError(error, "Glimmer model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->MaxContext();

  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded Glimmer model context");
    return false;
  }
  std::shared_ptr<GlimmerTextRunnerState::SharedDraft> shared_draft;
  if (speculative_config.backend == TextSpeculativeBackend::kDFlash) {
    if (speculative_config.draft_model_path.empty()) {
      SetError(error, "Glimmer DFlash speculative decoding requires --dflash-model");
      return false;
    }
    std::string draft_error;
    shared_draft = models::glimmer::GlimmerDflashDraftBackend::LoadShared(model, speculative_config.draft_model_path,
                                                                          max_context, &draft_error);
    if (shared_draft == nullptr) {
      SetError(error, "Failed to load the Glimmer DFlash draft: " + draft_error);
      return false;
    }
  } else if (speculative_config.backend != TextSpeculativeBackend::kDisabled) {
    SetError(error,
             "Glimmer HTTP models support only DFlash speculative decoding");
    return false;
  }
  // No continuation snapshots exist for Glimmer sessions, so the disk cache
  // has no identity to key on; the RAM pool owns every execution slot.
  if (DiskCacheEnabled(disk_cache_config)) {
    SetError(error,
             "Glimmer HTTP models do not support the persistent disk cache");
    return false;
  }
  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model =
        sampling::TextModelPreset::kUnspecified;
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = false;
    auto runner = std::make_shared<GlimmerTextRunner>(
        std::move(model), max_context, std::move(shared_draft),
        // 15 draft tokens (16 verify rows = one WMMA token tile) measured best on Glimmer Q4 (mean 37.9 tok/s over six
        // prompts vs 33.4 at 7, 26.6 at 20); the CLI default of 5 is treated as "unspecified" (pass any other value to override).
        std::min<std::uint32_t>(speculative_config.max_draft_tokens == 5 ? 15u : speculative_config.max_draft_tokens,
                                models::glimmer::kMaxDraftTokens));
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::nullopt, ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::gptoss::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    TextRunnerRamCacheOptions ram_cache_config) {
  if (model == nullptr) {
    SetError(error, "GPT-OSS model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->MaxContext();

  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded GPT-OSS model context");
    return false;
  }
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled) {
    SetError(error,
             "GPT-OSS HTTP models do not support speculative decoding");
    return false;
  }
  // No continuation snapshots exist for GPT-OSS sessions, so the disk cache
  // has no identity to key on; the RAM pool owns every execution slot.
  if (DiskCacheEnabled(disk_cache_config)) {
    SetError(error,
             "GPT-OSS HTTP models do not support the persistent disk cache");
    return false;
  }
  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model =
        sampling::TextModelPreset::kUnspecified;
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = false;
    auto runner =
        std::make_shared<GptOssTextRunner>(std::move(model), max_context);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::nullopt, ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::inkling::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    TextRunnerRamCacheOptions ram_cache_config) {
  if (model == nullptr) {
    SetError(error, "Inkling model must not be null");
    return false;
  }
  if (max_context == 0) max_context = model->MaxContext();
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded Inkling model context");
    return false;
  }
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      speculative_config.backend != TextSpeculativeBackend::kDSpark &&
      speculative_config.backend != TextSpeculativeBackend::kMtp) {
    SetError(error, "Inkling HTTP models support only DSpark speculative decoding");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config)) {
    SetError(error, "Inkling HTTP models do not support the persistent disk cache");
    return false;
  }
  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model = sampling::TextModelPreset::kUnspecified;
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = model->HasVision();
    const bool want_dspark =
        (speculative_config.backend == TextSpeculativeBackend::kDSpark ||
         speculative_config.backend == TextSpeculativeBackend::kMtp) &&
        model->HasDspark();
    const std::uint32_t max_draft = want_dspark
        ? (speculative_config.max_draft_tokens > 0
               ? std::min(speculative_config.max_draft_tokens,
                          model->MaxDraftTokens())
               : model->MaxDraftTokens())
        : 0;
    auto runner = std::make_shared<InklingTextRunner>(
        std::move(model), max_context,
        want_dspark ? speculative_config.draft_model_path : std::string{},
        max_draft);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::nullopt, ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::laguna::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    TextRunnerRamCacheOptions ram_cache_config) {
  if (model == nullptr) {
    SetError(error, "Laguna model must not be null");
    return false;
  }
  if (max_context == 0) max_context = model->MaxContext();
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded Laguna model context");
    return false;
  }
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled) {
    SetError(error, "Laguna HTTP models do not support speculative decoding");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config)) {
    SetError(error, "Laguna HTTP models do not support the persistent disk cache");
    return false;
  }
  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model = sampling::TextModelPreset::kUnspecified;
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = false;
    auto runner = std::make_shared<LagunaTextRunner>(std::move(model), max_context);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::nullopt, ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::llama::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    TextRunnerRamCacheOptions ram_cache_config) {
  if (model == nullptr) {
    SetError(error, "Llama model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->MaxContext();

  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded Llama model context");
    return false;
  }
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled) {
    SetError(error,
             "Llama HTTP models do not support speculative decoding");
    return false;
  }
  // No continuation snapshots exist for Llama sessions, so the disk cache
  // has no identity to key on; the RAM pool owns every execution slot.
  if (DiskCacheEnabled(disk_cache_config)) {
    SetError(error,
             "Llama HTTP models do not support the persistent disk cache");
    return false;
  }
  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model =
        sampling::TextModelPreset::kUnspecified;
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = false;
    auto runner =
        std::make_shared<LlamaTextRunner>(std::move(model), max_context);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::nullopt, ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::gemma4::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    TextRunnerRamCacheOptions ram_cache_config) {
  if (model == nullptr) {
    SetError(error, "Gemma4 model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->MaxContext();

  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded Gemma4 model context");
    return false;
  }
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      speculative_config.backend != TextSpeculativeBackend::kMtp) {
    SetError(error,
             "Gemma4 HTTP models support only MTP speculative decoding "
             "(--speculative mtp --mtp-model)");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kMtp) {
    if (speculative_config.draft_model_path.empty()) {
      SetError(error, "Gemma4 MTP HTTP decoding requires --mtp-model");
      return false;
    }
    if (speculative_config.max_draft_tokens == 0 ||
        speculative_config.max_draft_tokens > 16) {
      SetError(error,
               "Gemma4 MTP requires a draft limit in [1, 16] "
               "(--draft-tokens)");
      return false;
    }
  }
  // No continuation snapshots exist for Gemma4 sessions, so the disk cache
  // has no identity to key on; the RAM pool owns every execution slot.
  if (DiskCacheEnabled(disk_cache_config)) {
    SetError(error,
             "Gemma4 HTTP models do not support the persistent disk cache");
    return false;
  }
  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model =
        sampling::TextModelPreset::kUnspecified;
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = false;
    const bool use_mtp =
        speculative_config.backend == TextSpeculativeBackend::kMtp;
    auto runner = std::make_shared<Gemma4TextRunner>(
        std::move(model), max_context,
        use_mtp ? speculative_config.draft_model_path : std::string{},
        use_mtp ? speculative_config.max_draft_tokens : 0u);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::nullopt, ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}
#endif

std::string InferenceBackend::model_id() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_id : "unknown";
#else
  return "unknown";
#endif
}

std::vector<std::string> InferenceBackend::model_ids() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr)
    return {"unknown"};
  std::vector<std::string> ids{state->model_id};
  ids.insert(ids.end(), state->extra_model_ids.begin(),
             state->extra_model_ids.end());
  return ids;
#else
  return {"unknown"};
#endif
}

bool InferenceBackend::ready() const {
#if defined(ENGINE_ENABLE_HIP)
  return impl_->Snapshot() != nullptr;
#else
  return false;
#endif
}

bool InferenceBackend::device_lost() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->scheduler != nullptr &&
         state->scheduler->device_lost();
#else
  return false;
#endif
}

bool InferenceBackend::supports_images() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->supports_images;
#else
  return false;
#endif
}

std::uint32_t InferenceBackend::max_context() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->max_context : 0;
#else
  return 0;
#endif
}

std::vector<InferenceBackend::SessionState> InferenceBackend::session_states()
    const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->scheduler != nullptr
             ? state->scheduler->SessionStates()
             : std::vector<SessionState>{};
#else
  return {};
#endif
}

InferenceBackend::SamplingDefaults InferenceBackend::sampling_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->sampling_defaults : SamplingDefaults{};
#else
  return {};
#endif
}

ReasoningOptions InferenceBackend::reasoning_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->reasoning_defaults : ReasoningOptions{};
#else
  return {};
#endif
}

InferenceBackend::InitialOutputState InferenceBackend::initial_output_state(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr || state->scheduler == nullptr) {
    return InitialOutputState::kContent;
  }
  return state->scheduler->runner().InitialOutputState(request);
#else
  (void)request;
  return InitialOutputState::kAuto;
#endif
}

std::string InferenceBackend::model_path() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_path : std::string{};
#else
  return {};
#endif
}

std::string InferenceBackend::device_target() const {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  return impl_->device_target.empty() ? std::string("gpu")
                                      : impl_->device_target;
#else
  return "gpu";
#endif
}

std::optional<DiskCacheSnapshot> InferenceBackend::disk_cache_snapshot() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state != nullptr && state->scheduler != nullptr)
    return state->scheduler->ListDiskCache();
#endif
  return std::nullopt;
}

DiskCacheDeleteResult InferenceBackend::retire_disk_cache(
    const std::vector<std::uint64_t>& ids, bool delete_all,
    bool delete_incompatible) {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state != nullptr && state->scheduler != nullptr)
    return state->scheduler->RetireDiskCache(ids, delete_all,
                                             delete_incompatible);
#else
  (void)ids;
  (void)delete_all;
  (void)delete_incompatible;
#endif
  return {};
}

std::string InferenceBackend::build_info() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->build_info : std::string{};
#else
  return {};
#endif
}

void InferenceBackend::set_served_artifact(const std::string& model_path,
                                           const std::string& build_info) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_path = model_path;
  updated->build_info = build_info;
  impl_->state = std::move(updated);
#else
  (void)model_path;
  (void)build_info;
#endif
}

void InferenceBackend::set_model_ids(std::vector<std::string> model_ids) {
#if defined(ENGINE_ENABLE_HIP)
  if (model_ids.empty()) {
    return;
  }
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_id = std::move(model_ids.front());
  updated->extra_model_ids.assign(
      std::make_move_iterator(model_ids.begin() + 1),
      std::make_move_iterator(model_ids.end()));
  impl_->state = std::move(updated);
#else
  (void)model_ids;
#endif
}

void InferenceBackend::set_sampling_defaults(
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config,
    sampling::SamplingOverrides supplied) {
#if defined(ENGINE_ENABLE_HIP)
  sampling_config.Validate();
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->sampling_defaults.max_tokens = max_tokens;
  updated->sampling_defaults.sampling = sampling_config;
  updated->sampling_defaults.supplied = supplied;
  impl_->state = std::move(updated);
#else
  (void)max_tokens;
  (void)sampling_config;
  (void)supplied;
#endif
}

void InferenceBackend::set_reasoning_defaults(
    const ReasoningOptions& reasoning) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->reasoning_defaults = reasoning;
  impl_->state = std::move(updated);
#else
  (void)reasoning;
#endif
}

InferenceBackend::Result InferenceBackend::complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool cache_prompt) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  if (state->scheduler == nullptr) {
    throw std::invalid_argument(
        "this model serves rerank requests only (no text generation)");
  }
  auto prompt_tokens = state->scheduler->runner().Tokenize(prompt);
  return impl_->GenerateScheduled(
      state, std::move(prompt_tokens), request_start, max_tokens,
      sampling_config, is_cancelled, on_token, std::string(client_id), {},
      cache_prompt, 0, stop_sequences);
#else
  (void)cache_prompt;
  (void)client_id;
  (void)stop_sequences;
  (void)prompt;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const ChatRequest& request, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  if (state->scheduler == nullptr) {
    throw std::invalid_argument(
        "this model serves rerank requests only (no text generation)");
  }
  auto effective_sampling = sampling_config;
  auto constrained = ConstrainChatRequest(request, state->scheduler->runner(),
                                          &effective_sampling);
  const auto& effective_request = constrained ? *constrained : request;
  auto prompt = state->scheduler->runner().PreparePrompt(effective_request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return {};
  }
  const auto initial =
      state->scheduler->runner().InitialOutputState(effective_request);
  if (request.reasoning.budget_tokens &&
      initial == TextGenerationBackend::InitialOutputState::kReasoning)
    effective_sampling.reasoning_budget =
        state->scheduler->runner().BuildReasoningBudget(
            *request.reasoning.budget_tokens, prompt->tokens);
  return impl_->GenerateScheduled(
      state, std::move(prompt->tokens), request_start, max_tokens,
      effective_sampling, is_cancelled, on_token, request.client_id,
      std::move(prompt->context), request.cache_prompt,
      prompt->cache_prefix_tokens, request.stop_sequences, initial);
#else
  (void)request;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, bool stream_output, bool ignore_eos,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool return_progress, bool cache_prompt) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr || state->scheduler == nullptr) {
    return TextGenerationBackend::start_complete(
        prompt, max_tokens, sampling_config, is_cancelled, stream_output,
        ignore_eos, client_id, stop_sequences, return_progress, cache_prompt);
  }
  auto prompt_tokens = state->scheduler->runner().Tokenize(prompt);
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt_tokens), max_tokens, sampling_config, is_cancelled,
      stream_output,
      TextRequestMetadata{
          .client_id = client_id.empty() ? "anonymous" : std::string(client_id),
          .deadline = std::nullopt,
          .request_start = request_start,
          .prompt_context = {},
          .cache_prompt = cache_prompt,
          .cache_prefix_tokens = 0,
          .stop_at_eos = !ignore_eos,
          .stop_sequences = stop_sequences,
          .return_progress = return_progress,
      });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request));
#else
  return TextGenerationBackend::start_complete(
      prompt, max_tokens, sampling_config, is_cancelled, stream_output,
      ignore_eos, client_id, stop_sequences, return_progress, cache_prompt);
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_chat(const ChatRequest& request, std::size_t max_tokens,
                             const sampling::SamplingConfig& sampling_config,
                             const CancellationCheck& is_cancelled,
                             bool stream_output) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr || state->scheduler == nullptr) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  auto effective_sampling = sampling_config;
  std::optional<sampling::JsonConstraint::ToolFormat> tool_format;
  auto constrained = ConstrainChatRequest(request, state->scheduler->runner(),
                                          &effective_sampling, &tool_format);
  const auto& effective_request = constrained ? *constrained : request;
  auto prompt = state->scheduler->runner().PreparePrompt(effective_request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  const auto initial =
      state->scheduler->runner().InitialOutputState(effective_request);
  if (request.reasoning.budget_tokens &&
      initial == TextGenerationBackend::InitialOutputState::kReasoning)
    effective_sampling.reasoning_budget =
        state->scheduler->runner().BuildReasoningBudget(
            *request.reasoning.budget_tokens, prompt->tokens);
  const std::string client_id =
      request.client_id.empty() ? "anonymous" : request.client_id;
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt->tokens), max_tokens, effective_sampling, is_cancelled,
      stream_output,
      TextRequestMetadata{
          .client_id = client_id,
          .deadline = std::nullopt,
          .request_start = request_start,
          .prompt_context = std::move(prompt->context),
          .cache_prompt = request.cache_prompt,
          .cache_prefix_tokens = prompt->cache_prefix_tokens,
          .stop_sequences = request.stop_sequences,
          .return_progress = request.return_progress,
      });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request), initial, tool_format);
#else
  return TextGenerationBackend::start_chat(request, max_tokens, sampling_config,
                                           is_cancelled, stream_output);
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const std::vector<tokenization::ChatMessage>& messages,
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled) {
  return chat(ChatRequest{messages}, max_tokens, sampling_config, is_cancelled);
}

std::size_t InferenceBackend::count_tokens(std::string_view text) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr || state->scheduler == nullptr) {
    return 0;
  }
  return state->scheduler->runner().Tokenize(text).size();
#else
  (void)text;
  return 0;
#endif
}

std::optional<std::size_t> InferenceBackend::count_chat_tokens(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr || state->scheduler == nullptr) {
    return std::nullopt;
  }
  // The same preparation as chat(): constrained tool or format requests add
  // prompt text before the template is rendered.
  auto sampling = sampling::SamplingConfig{};
  auto constrained =
      ConstrainChatRequest(request, state->scheduler->runner(), &sampling);
  const auto& effective_request = constrained ? *constrained : request;
  return state->scheduler->runner().CountPromptTokens(effective_request);
#else
  (void)request;
  return std::nullopt;
#endif
}


// Rerank dispatch: BERT cross-encoder when a bert artifact is loaded,
// Qwen yes/no head for Qwen dense artifacts.
std::optional<RerankResult> InferenceBackend::Rerank(
    const std::string& query, const std::vector<std::string>& documents,
    std::string* error) {
#if defined(ENGINE_ENABLE_HIP)
  auto refuse = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return std::optional<RerankResult>{};
  };
  const auto current = impl_->Snapshot();
  if (current == nullptr) return refuse("no model is loaded");
  if (query.empty()) return refuse("rerank query must not be empty");
  if (documents.empty()) return refuse("rerank documents must not be empty");
  const std::lock_guard<std::mutex> lock(impl_->encoder_mutex);
  // A dedicated reranker (bge-reranker) was installed at load with a null key.
  // Otherwise the Qwen yes/no head shares the loaded Qwen weights and is
  // (re)built whenever the model identity changes.
  if (impl_->reranker == nullptr || impl_->reranker_key != nullptr) {
    if (current->scheduler == nullptr) {
      impl_->reranker.reset();
      impl_->reranker_key = nullptr;
      return refuse("rerank is not supported by this model");
    }
    const auto* qwen =
        dynamic_cast<const QwenTextRunner*>(&current->scheduler->runner());
    if (qwen == nullptr) {
      impl_->reranker.reset();
      impl_->reranker_key = nullptr;
      return refuse(
          "rerank requires a Qwen dense or BERT reranker model; this backend "
          "serves a different model kind");
    }
    const void* key = qwen->QwenModel().get();
    if (impl_->reranker == nullptr || impl_->reranker_key != key) {
      std::string wrap_error;
      auto runner = MakeQwenRerankRunner(qwen->QwenModel(),
                                         qwen->ExecutionPolicy(), &wrap_error);
      if (runner == nullptr) return refuse(wrap_error);
      impl_->reranker = std::move(runner);
      impl_->reranker_key = key;
    }
  }
  return impl_->reranker->RerankBatch(query, documents, error);
#else
  (void)query;
  (void)documents;
  if (error != nullptr) *error = "reranking requires the HIP backend";
  return std::nullopt;
#endif
}

// Embeddings for POST /v1/embeddings: the loaded model's EmbeddingRunner runs
// one prefill per text (no decode, no KV retention), fused across the texts
// when the encoder supports it. Qwen3-Embedding is built lazily on the shared
// Qwen weights.
std::optional<std::vector<EmbeddingResult>> InferenceBackend::EmbedBatch(
    const std::vector<std::string>& texts, std::string* error) {
#if defined(ENGINE_ENABLE_HIP)
  auto refuse = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return std::optional<std::vector<EmbeddingResult>>{};
  };
  const auto current = impl_->Snapshot();
  if (current == nullptr) return refuse("no model is loaded");
  if (texts.empty()) return refuse("embedding input must not be empty");
  const std::lock_guard<std::mutex> lock(impl_->encoder_mutex);
  if (impl_->embedder == nullptr || impl_->embedder_key != nullptr) {
    if (current->scheduler == nullptr) {
      impl_->embedder.reset();
      impl_->embedder_key = nullptr;
      return refuse(
          "embeddings require a Qwen dense or BGE-M3 model; this backend serves "
          "a reranker-only model kind");
    }
    const auto* qwen =
        dynamic_cast<const QwenTextRunner*>(&current->scheduler->runner());
    if (qwen == nullptr) {
      impl_->embedder.reset();
      impl_->embedder_key = nullptr;
      return refuse(
          "embeddings require a Qwen dense model; this backend serves a "
          "different model kind");
    }
    const auto model = qwen->QwenModel();
    if (model == nullptr) return refuse("Qwen model is unavailable");
    const void* key = model.get();
    if (impl_->embedder == nullptr || impl_->embedder_key != key) {
      // Leg-1 window: the cached executor sizes its KV arena to this bound
      // once, so longer inputs are refused explicitly instead of
      // oversubscribing.
      constexpr std::uint32_t kEmbeddingMaxTokens = 4096;
      const std::uint32_t embed_context =
          std::min(current->max_context, kEmbeddingMaxTokens);
      // Weak: a stale runner must not keep a replaced model state alive.
      std::weak_ptr<const Impl::State> weak = current;
      TokenizeFn tokenize = [weak](const std::string& text) {
        const auto state = weak.lock();
        if (state == nullptr || state->scheduler == nullptr)
          throw std::runtime_error("the embedding model was replaced");
        const auto tokens = state->scheduler->runner().Tokenize(text);
        return std::vector<std::uint32_t>(tokens.begin(), tokens.end());
      };
      impl_->embedder = MakeQwenEmbeddingRunner(
          model, qwen->ExecutionPolicy(), embed_context, current->model_id,
          std::move(tokenize));
      impl_->embedder_key = key;
    }
  }
  return impl_->embedder->EmbedBatch(texts, error);
#else
  (void)texts;
  if (error != nullptr) *error = "embeddings require the HIP backend";
  return std::nullopt;
#endif
}

}  // namespace mbufo::server
