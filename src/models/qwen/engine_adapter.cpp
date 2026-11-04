// Qwen3.8-27B adapter: thin forwarding over the session-less
// QwenGpuExecutor plus the one-line "qwen3.8-27b" EngineRegistry
// registration (name follows the serve model_kind convention with the
// dotted series prefix, matching the artifact family name).

#include "src/models/qwen/engine_adapter.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/blocked_q8.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace gufo::models::qwen {
namespace {

// Adapter snapshot frame: magic + version + valid_context + token ledger
// followed by the native compact bytes. The ledger makes the payload
// self-describing: restore rewinds both the device frontier and the token
// history to the same boundary, so a foreign-but-shape-compatible payload
// cannot desynchronize them.
constexpr char kSnapshotMagic[8] = {'Q', '2', '7', 'B', 'S', 'N', 'A', 'P'};
constexpr std::uint32_t kSnapshotVersion = 1;
constexpr std::size_t kSnapshotHeaderBytes = 8 + 4 + 4 + 4;

void PutLittleEndian32(std::span<std::uint8_t> dst, std::size_t offset,
                       std::uint32_t value) {
  dst[offset] = static_cast<std::uint8_t>(value);
  dst[offset + 1] = static_cast<std::uint8_t>(value >> 8);
  dst[offset + 2] = static_cast<std::uint8_t>(value >> 16);
  dst[offset + 3] = static_cast<std::uint8_t>(value >> 24);
}

[[nodiscard]] std::uint32_t GetLittleEndian32(std::span<const std::uint8_t> src,
                                              std::size_t offset) {
  return static_cast<std::uint32_t>(src[offset]) |
         (static_cast<std::uint32_t>(src[offset + 1]) << 8) |
         (static_cast<std::uint32_t>(src[offset + 2]) << 16) |
         (static_cast<std::uint32_t>(src[offset + 3]) << 24);
}

[[nodiscard]] std::int32_t Argmax(std::span<const float> logits) {
  std::int32_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = static_cast<std::int32_t>(i);
  }
  return best;
}

}  // namespace

Qwen27BEngineSession::Qwen27BEngineSession(
    std::shared_ptr<const hip::QwenGpuModel> model,
    std::unique_ptr<hip::QwenGpuExecutor> executor, std::uint32_t max_context)
    : model_(std::move(model)),
      executor_(std::move(executor)),
      max_context_(max_context) {}

bool Qwen27BEngineSession::EnsureOpen(std::string* error_msg) const {
  if (executor_ == nullptr || model_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "qwen3.8-27b session is closed";
    return false;
  }
  return true;
}

bool Qwen27BEngineSession::Sync(std::span<const std::int32_t> prompt,
                                std::string* error_msg) {
  if (!EnsureOpen(error_msg)) return false;
  try {
    for (const std::int32_t token : prompt) {
      if (token < 0) {
        if (error_msg != nullptr) {
          *error_msg = "qwen3.8-27b Sync token is negative";
        }
        return false;
      }
    }
    if (prompt.size() > max_context_) {
      if (error_msg != nullptr) {
        *error_msg = "qwen3.8-27b Sync exceeds the session context";
      }
      return false;
    }
    // Identical prompt: the frontier already represents it.
    if (prompt.size() == tokens_.size() &&
        std::equal(prompt.begin(), prompt.end(), tokens_.begin(),
                   [](std::int32_t a, std::uint32_t b) {
                     return static_cast<std::uint32_t>(a) == b;
                   })) {
      return true;
    }
    if (prompt.empty()) {
      executor_->Reset();
      tokens_.clear();
      logits_.clear();
      return true;
    }
    std::vector<tokenization::TokenId> ids;
    ids.reserve(prompt.size());
    for (const std::int32_t token : prompt) {
      ids.push_back(static_cast<tokenization::TokenId>(token));
    }
    // Strict extension of the committed ledger: append only the tail at
    // the live frontier (the GenerateFromPrefix incremental-prefix shape).
    // Anything else resets and refeeds the full prompt.
    if (ids.size() > tokens_.size() &&
        std::equal(ids.begin(), ids.begin() + tokens_.size(), tokens_.begin())) {
      const std::span<const tokenization::TokenId> tail(
          ids.data() + tokens_.size(), ids.size() - tokens_.size());
      (void)executor_->ForwardPromptBatch(
          tail, static_cast<std::uint32_t>(tokens_.size()),
          /*compute_logits=*/true);
    } else {
      executor_->Reset();
      (void)executor_->ForwardPromptBatch(ids, /*start_pos=*/0,
                                          /*compute_logits=*/true);
    }
    tokens_ = std::move(ids);
    const std::span<const float> logits = executor_->CopyLastLogits();
    logits_.assign(logits.begin(), logits.end());
    return true;
  } catch (const std::exception& ex) {
    if (error_msg != nullptr) *error_msg = ex.what();
    return false;
  } catch (...) {
    if (error_msg != nullptr) *error_msg = "qwen3.8-27b Sync failed";
    return false;
  }
}

bool Qwen27BEngineSession::Evaluate(std::int32_t token,
                                    std::string* error_msg) {
  if (!EnsureOpen(error_msg)) return false;
  try {
    if (token < 0) {
      if (error_msg != nullptr) {
        *error_msg = "qwen3.8-27b Evaluate token is negative";
      }
      return false;
    }
    if (tokens_.size() >= max_context_) {
      if (error_msg != nullptr) {
        *error_msg = "qwen3.8-27b Evaluate exceeds the session context";
      }
      return false;
    }
    // ForwardToken computes the row logits and returns the device argmax;
    // the host mirror is refreshed from the same row so SampleGreedy
    // agrees with the prediction the Generate loop would consume.
    (void)executor_->ForwardToken(static_cast<tokenization::TokenId>(token),
                                      static_cast<std::uint32_t>(tokens_.size()),
                                      /*compute_logits=*/true);
    tokens_.push_back(static_cast<tokenization::TokenId>(token));
    const std::span<const float> logits = executor_->CopyLastLogits();
    logits_.assign(logits.begin(), logits.end());
    return true;
  } catch (const std::exception& ex) {
    if (error_msg != nullptr) *error_msg = ex.what();
    return false;
  } catch (...) {
    if (error_msg != nullptr) *error_msg = "qwen3.8-27b Evaluate failed";
    return false;
  }
}

bool Qwen27BEngineSession::Forward(std::span<const std::int32_t> tokens,
                                   std::uint32_t start_pos,
                                   std::string* error_msg) {
  if (!EnsureOpen(error_msg)) return false;
  if (tokens.empty()) return true;
  if (start_pos > tokens_.size()) {
    if (error_msg != nullptr) {
      *error_msg = "qwen3.8-27b Forward starts past the committed length";
    }
    return false;
  }
  if (static_cast<std::uint64_t>(start_pos) + tokens.size() > max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "qwen3.8-27b Forward exceeds the session context";
    }
    return false;
  }
  // Re-forwarding already-committed rows is a no-op.
  if (static_cast<std::uint64_t>(start_pos) + tokens.size() <=
          tokens_.size() &&
      std::equal(tokens.begin(), tokens.end(), tokens_.begin() + start_pos,
                 [](std::int32_t a, std::uint32_t b) {
                   return a >= 0 && static_cast<std::uint32_t>(a) == b;
                 })) {
    return true;
  }
  std::vector<std::int32_t> prompt;
  prompt.reserve(static_cast<std::size_t>(start_pos) + tokens.size());
  for (std::uint32_t i = 0; i < start_pos; ++i) {
    prompt.push_back(static_cast<std::int32_t>(tokens_[i]));
  }
  prompt.insert(prompt.end(), tokens.begin(), tokens.end());
  return Sync(prompt, error_msg);
}

std::span<const float> Qwen27BEngineSession::Logits() const noexcept {
  return {logits_.data(), logits_.size()};
}

std::uint32_t Qwen27BEngineSession::Length() const noexcept {
  return static_cast<std::uint32_t>(tokens_.size());
}

std::span<const std::uint32_t> Qwen27BEngineSession::Tokens() const noexcept {
  return {tokens_.data(), tokens_.size()};
}

std::int32_t Qwen27BEngineSession::SampleGreedy() {
  if (logits_.empty()) return EosToken();  // Unsynced session: stop.
  return Argmax(logits_);
}

bool Qwen27BEngineSession::DecodeStep(std::uint32_t max_draft_tokens,
                                      core::engine::DecodeOutcome* outcome,
                                      std::string* error_msg) {
  (void)max_draft_tokens;  // No draft path on 27B; plain fallback ignores it.
  if (outcome == nullptr) {
    if (error_msg != nullptr) *error_msg = "qwen3.8-27b DecodeStep: null outcome";
    return false;
  }
  if (!EnsureOpen(error_msg)) return false;
  if (tokens_.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "qwen3.8-27b DecodeStep produced no tokens (Sync a prompt first)";
    }
    return false;
  }
  const std::int32_t token = SampleGreedy();
  if (!Evaluate(token, error_msg)) return false;
  outcome->token = token;
  outcome->stop = IsStopToken(token);
  outcome->tokens_committed = 1;
  return true;
}

std::size_t Qwen27BEngineSession::SnapshotPayloadBytes() const {
  if (executor_ == nullptr) return 0;
  // Pure ledger math: no device traffic (see file doc).
  return kSnapshotHeaderBytes + tokens_.size() * sizeof(std::uint32_t);
}

bool Qwen27BEngineSession::Snapshot(std::span<std::uint8_t> destination,
                                    std::string* error_msg) const {
  if (!EnsureOpen(error_msg)) return false;
  const auto count = static_cast<std::uint32_t>(tokens_.size());
  if (destination.size() != SnapshotPayloadBytes()) {
    if (error_msg != nullptr) {
      *error_msg = "qwen3.8-27b Snapshot destination size mismatch";
    }
    return false;
  }
  std::memcpy(destination.data(), kSnapshotMagic, 8);
  PutLittleEndian32(destination, 8, kSnapshotVersion);
  PutLittleEndian32(destination, 12, count);
  PutLittleEndian32(destination, 16, count);
  for (std::size_t i = 0; i < tokens_.size(); ++i) {
    PutLittleEndian32(destination, kSnapshotHeaderBytes + i * 4, tokens_[i]);
  }
  return true;
}

bool Qwen27BEngineSession::RestoreSnapshot(
    std::span<const std::uint8_t> payload, std::string* error_msg) {
  if (!EnsureOpen(error_msg)) return false;
  try {
    if (payload.size() < kSnapshotHeaderBytes ||
        std::memcmp(payload.data(), kSnapshotMagic, 8) != 0) {
      if (error_msg != nullptr) {
        *error_msg = "qwen3.8-27b snapshot header is invalid";
      }
      return false;
    }
    const std::span<const std::uint8_t> view(payload.data(), payload.size());
    if (GetLittleEndian32(view, 8) != kSnapshotVersion) {
      if (error_msg != nullptr) {
        *error_msg = "qwen3.8-27b snapshot version is invalid";
      }
      return false;
    }
    const std::uint32_t valid_context = GetLittleEndian32(view, 12);
    const std::uint32_t token_count = GetLittleEndian32(view, 16);
    if (token_count != valid_context) {
      if (error_msg != nullptr) {
        *error_msg = "qwen3.8-27b snapshot ledger does not match its frontier";
      }
      return false;
    }
    // Exact frame match: truncation and trailing-garbage payloads are
    // refused before any device work.
    const std::size_t ledger_bytes =
        static_cast<std::size_t>(token_count) * 4;
    if (payload.size() != kSnapshotHeaderBytes + ledger_bytes) {
      if (error_msg != nullptr) {
        *error_msg = "qwen3.8-27b snapshot size is invalid";
      }
      return false;
    }
    std::vector<std::uint32_t> ledger(token_count);
    for (std::uint32_t i = 0; i < token_count; ++i) {
      ledger[i] =
          GetLittleEndian32(view, kSnapshotHeaderBytes + i * 4);
    }
    // Rewind by Reset + full re-prefill of the ledger (see file doc): the
    // SSM rolling state is recomputed from scratch and the frontier
    // truncates to the ledger boundary. On failure the device is left
    // reset, so the caller must re-Sync before further L0 calls.
    executor_->Reset();
    std::vector<float> fresh_logits;
    if (!ledger.empty()) {
      (void)executor_->ForwardPromptBatch(
          std::span<const tokenization::TokenId>(ledger.data(), ledger.size()),
          /*start_pos=*/0, /*compute_logits=*/true);
      const std::span<const float> logits = executor_->CopyLastLogits();
      fresh_logits.assign(logits.begin(), logits.end());
    }
    tokens_ = std::move(ledger);
    logits_ = std::move(fresh_logits);
    return true;
  } catch (const std::exception& ex) {
    if (error_msg != nullptr) *error_msg = ex.what();
    return false;
  } catch (...) {
    if (error_msg != nullptr) {
      *error_msg = "qwen3.8-27b snapshot restore failed";
    }
    return false;
  }
}

std::vector<std::uint32_t> Qwen27BEngineSession::ForwardVerificationChunk(
    std::span<const std::uint32_t> candidate_tokens, std::uint32_t start_pos,
    bool capture_logits, std::string* error_msg) {
  if (!EnsureOpen(error_msg)) return {};
  try {
    if (static_cast<std::uint64_t>(start_pos) + candidate_tokens.size() >
        max_context_) {
      if (error_msg != nullptr) {
        *error_msg = "qwen3.8-27b verification exceeds the session context";
      }
      return {};
    }
    return executor_->ForwardVerificationChunk(
        std::span<const tokenization::TokenId>(candidate_tokens.data(),
                                               candidate_tokens.size()),
        start_pos, capture_logits);
  } catch (const std::exception& ex) {
    if (error_msg != nullptr) *error_msg = ex.what();
    return {};
  } catch (...) {
    if (error_msg != nullptr) {
      *error_msg = "qwen3.8-27b ForwardVerificationChunk failed";
    }
    return {};
  }
}

bool Qwen27BEngineSession::CommitVerificationChunk(
    std::span<const std::uint32_t> committed_tokens, std::uint32_t start_pos,
    std::string* error_msg) {
  if (!EnsureOpen(error_msg)) return false;
  try {
    executor_->CommitVerificationChunk(
        std::span<const tokenization::TokenId>(committed_tokens.data(),
                                               committed_tokens.size()),
        start_pos);
    return true;
  } catch (const std::exception& ex) {
    if (error_msg != nullptr) *error_msg = ex.what();
    return false;
  } catch (...) {
    if (error_msg != nullptr) {
      *error_msg = "qwen3.8-27b CommitVerificationChunk failed";
    }
    return false;
  }
}

bool Qwen27BEngineSession::FinishVerification(std::string* error_msg) {
  if (!EnsureOpen(error_msg)) return false;
  try {
    executor_->FinishVerification();
    return true;
  } catch (const std::exception& ex) {
    if (error_msg != nullptr) *error_msg = ex.what();
    return false;
  } catch (...) {
    if (error_msg != nullptr) {
      *error_msg = "qwen3.8-27b FinishVerification failed";
    }
    return false;
  }
}

std::string Qwen27BEngineSession::Decode(
    std::span<const std::int32_t> tokens) const {
  if (model_ == nullptr) return {};
  std::vector<tokenization::TokenId> ids;
  ids.reserve(tokens.size());
  for (const std::int32_t token : tokens) {
    ids.push_back(token < 0 ? tokenization::kInvalidTokenId
                            : static_cast<tokenization::TokenId>(token));
  }
  return model_->GetTokenizer().Decode(ids);
}

bool Qwen27BEngineSession::IsStopToken(std::int32_t token) const noexcept {
  if (model_ == nullptr || token < 0) return false;
  return model_->GetTokenizer().IsStopToken(
             static_cast<tokenization::TokenId>(token)) ||
         token == EosToken();
}

std::int32_t Qwen27BEngineSession::EosToken() const noexcept {
  if (model_ == nullptr) return 0;
  return static_cast<std::int32_t>(model_->GetTokenizer().GetEosTokenId());
}

bool Qwen27BEngineSession::HasVision() const noexcept {
  return model_ != nullptr && model_->VisionEncoder() != nullptr;
}

bool Qwen27BEngineSession::ConfigureVision(
    std::shared_ptr<const vision::Prompt> prompt,
    std::shared_ptr<vision::Encoder> encoder, std::string* error_msg) {
  if (!EnsureOpen(error_msg)) return false;
  try {
    executor_->ConfigureVision(std::move(prompt), std::move(encoder));
    return true;
  } catch (const std::exception& ex) {
    if (error_msg != nullptr) *error_msg = ex.what();
    return false;
  } catch (...) {
    if (error_msg != nullptr) {
      *error_msg = "qwen3.8-27b ConfigureVision failed";
    }
    return false;
  }
}

Qwen27BEngineModel::Qwen27BEngineModel(
    std::shared_ptr<const hip::QwenGpuModel> model, std::string model_path,
    std::uint32_t max_context, hip::QwenExecutionPolicy policy)
    : model_(std::move(model)),
      model_path_(std::move(model_path)),
      max_context_(max_context),
      policy_(policy) {}

std::shared_ptr<Qwen27BEngineModel> Qwen27BEngineModel::Load(
    const std::string& model_path, const Qwen27BModelOptions& options,
    std::string* error_msg) {
  try {
    std::string load_error;
    auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
    if (reader_owner == nullptr) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to open GGUF: " + load_error;
      }
      return nullptr;
    }
    std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
    auto model =
        hip::QwenGpuModel::CreateFromGguf(reader, &load_error);
    if (!model) {
      if (error_msg != nullptr) *error_msg = load_error;
      return nullptr;
    }
    hip::QwenExecutionPolicy policy = hip::QwenExecutionPolicy::Production();
    if (options.kv_cache_quant.empty() || options.kv_cache_quant == "f16") {
      // Default: native f16 KV path.
    } else if (options.kv_cache_quant == "q8") {
      policy.blocked_q8_attention = true;
      std::string blocked_error;
      if (!hip::ValidateQwenBlockedQ8Config(model->GetConfig(),
                                            &blocked_error)) {
        if (error_msg != nullptr) {
          *error_msg = "blocked-Q8 attention rejected: " + blocked_error;
        }
        return nullptr;
      }
    } else {
      if (error_msg != nullptr) {
        *error_msg = "kv cache quant must be f16 or q8";
      }
      return nullptr;
    }
    const std::uint32_t native_context = model->GetConfig().context_length;
    const std::uint32_t resolved =
        options.max_context != 0
            ? options.max_context
            : (native_context != 0 ? native_context : 4096);
    if (resolved == 0 ||
        (native_context != 0 && resolved > native_context)) {
      if (error_msg != nullptr) {
        *error_msg = "Requested GPU context exceeds the model context length";
      }
      return nullptr;
    }
    return std::shared_ptr<Qwen27BEngineModel>(new Qwen27BEngineModel(
        std::move(model), model_path, resolved, policy));
  } catch (const std::exception& ex) {
    if (error_msg != nullptr) *error_msg = ex.what();
    return nullptr;
  } catch (...) {
    if (error_msg != nullptr) *error_msg = "qwen3.8-27b Load failed";
    return nullptr;
  }
}

std::shared_ptr<Qwen27BEngineModel> Qwen27BEngineModel::Load(
    const std::string& model_path, std::string* error_msg) {
  return Load(model_path, Qwen27BModelOptions{}, error_msg);
}

std::unique_ptr<core::engine::EngineSession>
Qwen27BEngineModel::CreateSession(std::string* error_msg) {
  if (model_ == nullptr) {
    if (error_msg != nullptr) *error_msg = "qwen3.8-27b model is closed";
    return nullptr;
  }
  try {
    // One exclusive executor per session: the executor IS the session, so
    // sessions are never shared or pooled (see file doc).
    auto executor =
        hip::QwenGpuExecutor::Create(model_, error_msg, max_context_, policy_);
    if (!executor) return nullptr;
    return std::unique_ptr<core::engine::EngineSession>(new Qwen27BEngineSession(
        model_, std::move(executor), max_context_));
  } catch (const std::exception& ex) {
    if (error_msg != nullptr) *error_msg = ex.what();
    return nullptr;
  } catch (...) {
    if (error_msg != nullptr) {
      *error_msg = "qwen3.8-27b CreateSession failed";
    }
    return nullptr;
  }
}

std::vector<std::int32_t> Qwen27BEngineModel::Tokenize(
    std::string_view text, bool add_bos) const {
  tokenization::TokenizerOptions options;
  options.add_bos = add_bos;
  options.parse_special_tokens = true;
  const auto ids = model_->GetTokenizer().Encode(text, options);
  std::vector<std::int32_t> out;
  out.reserve(ids.size());
  for (const tokenization::TokenId id : ids) {
    out.push_back(static_cast<std::int32_t>(id));
  }
  return out;
}

std::uint32_t Qwen27BEngineModel::VocabSize() const noexcept {
  return model_->GetConfig().vocab_size;
}

std::uint32_t Qwen27BEngineModel::MaxContext() const noexcept {
  return max_context_;
}

std::size_t Qwen27BEngineModel::ResidentBytes() const noexcept {
  if (model_ == nullptr) return 0;
  return model_->GetResidentBytes();
}

std::string Qwen27BEngineModel::ModelName() const { return model_path_; }

std::string Qwen27BEngineModel::Decode(
    std::span<const std::int32_t> tokens) const {
  if (model_ == nullptr) return {};
  std::vector<tokenization::TokenId> ids;
  ids.reserve(tokens.size());
  for (const std::int32_t token : tokens) {
    ids.push_back(token < 0 ? tokenization::kInvalidTokenId
                            : static_cast<tokenization::TokenId>(token));
  }
  return model_->GetTokenizer().Decode(ids);
}

bool Qwen27BEngineModel::IsStopToken(std::int32_t token) const noexcept {
  if (model_ == nullptr || token < 0) return false;
  return model_->GetTokenizer().IsStopToken(
             static_cast<tokenization::TokenId>(token)) ||
         token == EosToken();
}

std::int32_t Qwen27BEngineModel::EosToken() const noexcept {
  if (model_ == nullptr) return 0;
  return static_cast<std::int32_t>(model_->GetTokenizer().GetEosTokenId());
}

bool Qwen27BEngineModel::HasVision() const noexcept {
  return model_ != nullptr && model_->VisionEncoder() != nullptr;
}

namespace {

// One-line "qwen3.8-27b" registration: maps the generic load request onto
// the 27B load (zero/empty fields keep engine defaults: full native
// context, f16 KV; "q8" selects the blocked INT8 KV path). The request
// carries no drafts artifact or vision weights, so registry sessions are
// plain single-token text sessions; speculative/vision callers use the
// pass-throughs above with their own artifacts.
const core::engine::EngineRegistrar kRegisterQwen27BEngine(
    core::engine::EngineBackendInfo{
        "qwen3.8-27b", "Qwen3.8-27B hybrid SSM/attention (f16/q8 KV)"},
    [](const core::engine::EngineLoadRequest& request,
       std::string* error_msg) -> std::shared_ptr<core::engine::EngineModel> {
      Qwen27BModelOptions options;
      if (request.max_context != 0) {
        options.max_context = request.max_context;
      }
      if (!request.kv_cache_quant.empty()) {
        options.kv_cache_quant = request.kv_cache_quant;
      }
      return Qwen27BEngineModel::Load(request.model_path, options, error_msg);
    });

}  // namespace

}  // namespace gufo::models::qwen

#endif  // defined(ENGINE_ENABLE_HIP)
