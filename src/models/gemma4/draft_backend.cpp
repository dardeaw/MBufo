// GemmaMtpDraftBackend IDraftBackend surface + "gemma4-mtp" registration.
//
// The round-critical methods (ProposeDrafts, GemmaTargetVerify::Verify,
// Session::CommitMtpRound, Session::ResolveDraftAnchor) live in engine.cpp
// next to Session::Impl. This TU only contains the interface surface that
// needs nothing beyond the public Session API, plus the file-scope
// registrar so a future draft only touches its own files.

#include "src/models/gemma4/draft_backend.hpp"

#include <algorithm>
#include <cstdint>

#include "src/core/speculative/draft_registry.hpp"
#include "src/models/gemma4/engine.hpp"

namespace gufo::models::gemma4 {

speculative::DraftProposal GemmaMtpDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  (void)prompt_tokens;
  speculative::DraftProposal out;
  out.start_pos = current_pos;
  out.from_lookup = false;
  if (!session_->MtpEnabled() || max_tokens == 0) {
    return out;
  }
  std::uint32_t n_draft = std::min(max_tokens, max_draft_tokens_);
  if (n_draft > 7u) {
    n_draft = 7u;
  }
  const std::int32_t cur = session_->ResolveDraftAnchor();
  const std::uint32_t old_length = session_->Length();
  std::vector<std::int32_t> draft;
  if (!ProposeDrafts(cur, n_draft, old_length, &draft, nullptr)) {
    return speculative::DraftProposal{};
  }
  out.tokens.reserve(draft.size());
  for (const std::int32_t tok : draft) {
    out.tokens.push_back(static_cast<tokenization::TokenId>(tok));
  }
  return out;
}

void GemmaMtpDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)correction_token;
  accepted_ += accepted.size();
  // Each round proposes accepted.size() drafts plus one target correction.
  drafted_ += accepted.size() + 1;
}

void GemmaMtpDraftBackend::Reset() noexcept {
  drafted_ = 0;
  accepted_ = 0;
}

namespace {

bool RegisterGemmaMtpDraftBackend() {
  speculative::RegisterDraftBackend(
      speculative::DraftBackendInfo{
          .name = "gemma4-mtp",
          .description =
              "Gemma 4 MTP draft head (session-bound, greedy only; "
              "cross-attends the live session KV, no target rollback needed)",
          .requires_target_model = false,
          .requires_session = true,
      },
      [](const speculative::DraftBackendRequest& request,
         std::string* error_msg) -> std::unique_ptr<speculative::IDraftBackend> {
        if (request.session_kind != "gemma4" ||
            request.session_handle == nullptr) {
          if (error_msg != nullptr) {
            *error_msg =
                "gemma4-mtp backend requires a live gemma4 Session "
                "(session_kind=\"gemma4\")";
          }
          return nullptr;
        }
        auto* session = static_cast<Session*>(request.session_handle);
        if (!session->MtpEnabled()) {
          if (error_msg != nullptr) {
            *error_msg =
                "gemma4-mtp backend requires EnableMtp() on the session first";
          }
          return nullptr;
        }
        std::uint32_t max_tokens = request.max_draft_tokens == 0
                                       ? 4u
                                       : request.max_draft_tokens;
        if (max_tokens > 7u) {
          max_tokens = 7u;
        }
        return std::make_unique<GemmaMtpDraftBackend>(*session, max_tokens);
      });
  return true;
}

const bool kGemmaMtpDraftRegistered = RegisterGemmaMtpDraftBackend();

}  // namespace

}  // namespace gufo::models::gemma4
