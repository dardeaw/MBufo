// FlashNextMtpDraftBackend implementation + "flash-next-mtp" registration.
//
// Deferred work (documented, not attempted here): the engine's own hot path
// keeps its interleaved propose loop (prompt-lookup copies, small-head
// subset, confidence gate, sampled compact proposals with GPU rejection,
// multi-session DraftCatchUpBatch) and its batched verify loop
// (DecodeBatchImpl + FinishDecode accept logic). Moving those behind
// IDraftBackend/ISpeculativeTargetExecutor would rewrite the scheduler's
// core and risk behavior drift; the propose (ProposeChain) and rollback
// (Rollback) phases are the seam that can be unified verbatim today.

#include "src/models/qwen38_flash_next/draft_backend.hpp"

#include <algorithm>
#include <cstdint>
#include <string>

#include "src/core/speculative/draft_registry.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::qwen38_flash_next {

std::unique_ptr<FlashNextMtpDraftBackend> FlashNextMtpDraftBackend::Create(
    Session& session, std::string* error_msg) {
  if (!session.MtpEnabled()) {
    if (error_msg != nullptr) {
      *error_msg = "flash-next-mtp backend requires a session with MTP enabled";
    }
    return nullptr;
  }
  return std::unique_ptr<FlashNextMtpDraftBackend>(
      new FlashNextMtpDraftBackend(session));
}

speculative::DraftProposal FlashNextMtpDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  speculative::DraftProposal out;
  out.start_pos = current_pos;
  out.from_lookup = false;
  if (!session_->MtpEnabled() || max_tokens == 0 || prompt_tokens.empty()) {
    return out;
  }
  // Session-consistent prefix: committed history + anchor last.
  const auto& committed = session_->tokens_;
  if (current_pos != committed.size() ||
      prompt_tokens.size() != committed.size() + 1) {
    return out;
  }
  for (std::size_t i = 0; i < committed.size(); ++i) {
    if (static_cast<std::int32_t>(prompt_tokens[i]) != committed[i]) {
      return out;
    }
  }
  rocm::Executor& exec = *session_->model_->executor_;
  const std::size_t room = session_->ContextSize() - committed.size();
  const auto cap = static_cast<std::uint32_t>(
      std::min<std::size_t>({max_tokens, room, exec.max_speculative()}));
  if (cap == 0) {
    return out;
  }
  const std::int32_t anchor =
      static_cast<std::int32_t>(prompt_tokens.back());
  std::string error;
  if (!session_->DraftCatchUp(anchor, true, &error, nullptr, false)) {
    return out;
  }
  out.tokens.reserve(cap);
  out.tokens.push_back(
      static_cast<tokenization::TokenId>(session_->draft_token_));
  std::int32_t tok = session_->draft_token_;
  while (out.tokens.size() < cap) {
    if (session_->model_->IsStopToken(tok)) {
      break;
    }
    if (!exec.MtpForward(*session_->session_,
                         std::span<const std::int32_t>(&tok, 1), -1,
                         {.token = &tok}, &error)) {
      out.tokens.clear();
      return out;
    }
    out.tokens.push_back(static_cast<tokenization::TokenId>(tok));
  }
  session_->draft_token_ = tok;
  return out;
}

void FlashNextMtpDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)correction_token;
  accepted_ += accepted.size();
  drafted_ += accepted.size() + 1;
}

void FlashNextMtpDraftBackend::Reset() noexcept {
  drafted_ = 0;
  accepted_ = 0;
}

bool FlashNextMtpDraftBackend::ProposeChain(
    std::int32_t anchor, MtpCandidateLogits* candidates, bool draft_subset,
    std::string* error_msg) {
  return session_->DraftCatchUp(anchor, true, error_msg, candidates,
                                draft_subset);
}

bool FlashNextMtpDraftBackend::Rollback(std::uint32_t keep,
                                        bool keep_logits_row,
                                        std::string* error_msg) {
  rocm::Executor& exec = *session_->model_->executor_;
  return exec.Rollback(*session_->session_, keep, error_msg,
                       keep_logits_row ? session_->logits_.data() : nullptr);
}

namespace {

bool RegisterFlashNextMtpDraftBackend() {
  speculative::RegisterDraftBackend(
      speculative::DraftBackendInfo{
          .name = "flash-next-mtp",
          .description =
              "Flash-Next MTP head (session-bound, greedy pure-MTP chain; "
              "propose/rollback phases of the engine driver)",
          .requires_target_model = false,
          .requires_session = true,
      },
      [](const speculative::DraftBackendRequest& request,
         std::string* error_msg) -> std::unique_ptr<speculative::IDraftBackend> {
        if (request.session_kind != "flash-next" ||
            request.session_handle == nullptr) {
          if (error_msg != nullptr) {
            *error_msg =
                "flash-next-mtp backend requires a live flash-next Session "
                "(session_kind=\"flash-next\")";
          }
          return nullptr;
        }
        auto* session = static_cast<Session*>(request.session_handle);
        return FlashNextMtpDraftBackend::Create(*session, error_msg);
      });
  return true;
}

const bool kFlashNextMtpDraftRegistered = RegisterFlashNextMtpDraftBackend();

}  // namespace

}  // namespace gufo::models::qwen38_flash_next
