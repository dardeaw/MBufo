// GlimmerDflashDraftBackend IDraftBackend surface + "glimmer-dflash"
// registration (27B DFlash2 pattern: target-bound, requires_target_model).
//
// The round-critical paths (prime/append/propose over fused target
// features) live here; the greedy accept comparison and the target commit
// stay model-side in Session::SpeculativeRound. The draft is block
// diffusion (reference: llama.cpp draft-dflash): fused features seed the
// draft KV, proposals decode [id_last, mask * k] non-causally.

#include "src/models/glimmer/draft_backend.hpp"

#include <algorithm>
#include <cstdint>

#include "src/core/speculative/draft_registry.hpp"

#include <cstdio>
#include <cstdlib>
#include "src/models/glimmer/engine.hpp"


namespace gufo::models::glimmer {

std::unique_ptr<GlimmerDflashDraftBackend> GlimmerDflashDraftBackend::Create(
    std::shared_ptr<const Model> target, const std::string& draft_model_path,
    std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::string* error_msg) {
  if (target == nullptr) {
    if (error_msg != nullptr && error_msg->empty())
      *error_msg = "glimmer-dflash backend needs a target model";
    return nullptr;
  }
  if (draft_model_path.empty()) {
    if (error_msg != nullptr && error_msg->empty())
      *error_msg = "glimmer-dflash backend needs draft_model_path";
    return nullptr;
  }
  if (max_draft_tokens == 0)
    max_draft_tokens = 4;
  if (max_draft_tokens > 8)
    max_draft_tokens = 8;  // target verify width (kMaxVerify)
  std::string error;
  draft::DraftModelOptions draft_options;
  draft_options.max_context =
      max_context != 0 ? max_context : target->MaxContext();
  auto draft_model = draft::DraftModel::Load(draft_model_path, draft_options,
                                             {}, &error);
  if (draft_model == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "glimmer-dflash draft load failed: " + error;
    return nullptr;
  }
  // Normalize the artifact's 1-based layer-input taps to zero-based layer
  // outputs (27B DFlash2 convention: the converter numbers layer inputs,
  // Gufo taps layer outputs).
  std::vector<std::uint32_t> taps;
  taps.reserve(draft_model->TargetLayerIds().size());
  const std::uint32_t num_target_layers = target->weights().config.num_layers;
  for (const std::uint32_t encoded : draft_model->TargetLayerIds()) {
    if (encoded == 0 || encoded > num_target_layers) {
      if (error_msg != nullptr)
        *error_msg = "glimmer-dflash tap layer out of range";
      return nullptr;
    }
    taps.push_back(encoded - 1);
  }
  if (taps.empty()) {
    if (error_msg != nullptr && error_msg->empty())
      *error_msg = "glimmer-dflash draft has no tap layers";
    return nullptr;
  }
  const auto& tc = target->weights().config;
  draft::TargetHead head;
  head.output = target->weights().output;
  head.logit_scale = tc.logit_scale;
  head.final_logit_softcapping = tc.final_logit_softcapping;
  head.vocab_size = tc.vocab_size;
  if (head.output.empty() || head.vocab_size == 0) {
    if (error_msg != nullptr)
      *error_msg = "glimmer-dflash target head is missing";
    return nullptr;
  }
  draft_model->SetTargetHead(head);
  draft::TargetEmbed embed;
  embed.device = target->weights().token_embd.device;
  embed.type = target->weights().token_embd.type;
  if (embed.device == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "glimmer-dflash target embedding is missing";
    return nullptr;
  }
  if (tc.hidden_size != draft_model->HiddenSize()) {
    if (error_msg != nullptr)
      *error_msg = "glimmer-dflash target/draft hidden size mismatch";
    return nullptr;
  }
  if (draft_model->MaskTokenId() == 0 ||
      draft_model->MaskTokenId() >= tc.vocab_size) {
    if (error_msg != nullptr)
      *error_msg = "glimmer-dflash mask token id is invalid";
    return nullptr;
  }
  draft_model->SetTargetEmbed(embed);
  auto session = draft_model->CreateSession(&error);
  if (session == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "glimmer-dflash session failed: " + error;
    return nullptr;
  }
  const std::uint32_t feat_width =
      static_cast<std::uint32_t>(taps.size()) * tc.hidden_size;
  return std::unique_ptr<GlimmerDflashDraftBackend>(
      new GlimmerDflashDraftBackend(std::move(draft_model),
                                    std::move(session), std::move(taps),
                                    max_draft_tokens, feat_width));
}

bool GlimmerDflashDraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  Reset();
  if (context.prompt_tokens.empty() || context.hidden_size != feat_width_ ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * feat_width_) {
    return false;
  }
  const auto rows = static_cast<std::uint32_t>(context.prompt_tokens.size());
  if (rows > draft_model_->MaxContext()) {
    return false;
  }
  std::string error;
  if (!draft_session_->PrimeRows(context.prompt_hidden_states, rows, 0,
                                 &error)) {
    return false;
  }
  // Carry the last primed row: the next block overwrites position rows-1
  // with the anchor, and the commit restores it from this saved copy
  // (reference: the verify batch re-injects the anchor position with true
  // target features).
  tail_feats_.assign(context.prompt_hidden_states.end() - feat_width_,
                     context.prompt_hidden_states.end());
  has_tail_ = true;
  primed_ = true;
  return true;
}

bool GlimmerDflashDraftBackend::AppendTargetContext(
    const speculative::DraftTargetContext& context, std::uint32_t position) {
  if (!primed_ || context.hidden_size != feat_width_ ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * feat_width_ ||
      context.prompt_tokens.empty()) {
    return false;
  }
  // Drop the last speculative chain, then encode the committed rows at
  // their exact positions (re-encoding from true features keeps the draft
  // KV position-exact; the bonus row is included). The proposal stays
  // active across the rewind: AcceptFeedback (called after the commit)
  // consumes the drafted/accepted counters and clears it.
  // The block overlaps the committed prefix by one row (the anchor
  // overwrote position propose_pos_): rewind there, then the commit rows
  // land exactly at [position, ...). The overwritten anchor row is restored
  // from the carried tail (true target features saved at the previous
  // prime/commit); without this the most recent prefix row stays
  // anchor-derived and later masks lose coordination.
  std::vector<float> commit_feats;
  std::uint32_t commit_pos = position;
  std::uint32_t commit_rows =
      static_cast<std::uint32_t>(context.prompt_tokens.size());
  if (proposal_active_) {
    std::string error;
    if (!draft_session_->Rewind(propose_pos_, &error)) {
      return false;
    }
    if (draft_session_->Length() != propose_pos_) {
      return false;
    }
    if (has_tail_ && tail_feats_.size() == feat_width_ &&
        propose_pos_ + 1 == position) {
      commit_feats.reserve(
          (static_cast<std::size_t>(commit_rows) + 1) * feat_width_);
      commit_feats.insert(commit_feats.end(), tail_feats_.begin(),
                          tail_feats_.end());
      commit_feats.insert(commit_feats.end(),
                          context.prompt_hidden_states.begin(),
                          context.prompt_hidden_states.end());
      commit_pos = position - 1;
      commit_rows += 1;
    }
  } else if (draft_session_->Length() != position) {
    return false;
  }
  std::string error;
  const float* feat_ptr = commit_feats.empty()
                              ? context.prompt_hidden_states.data()
                              : commit_feats.data();
  if (!draft_session_->PrimeRows(
          std::span<const float>(feat_ptr,
                                 static_cast<std::size_t>(commit_rows) *
                                     feat_width_),
          commit_rows, commit_pos, &error)) {
    return false;
  }
  // Carry the last committed row for the next anchor restore.
  tail_feats_.assign(context.prompt_hidden_states.end() - feat_width_,
                     context.prompt_hidden_states.end());
  has_tail_ = true;
  return true;
}

speculative::DraftProposal GlimmerDflashDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  speculative::DraftProposal out;
  out.start_pos = current_pos;
  out.from_lookup = false;
  if (!primed_ || max_tokens == 0)
    return out;
  auto& session = *draft_session_;
  if (session.Length() != current_pos)
    return out;
  // The anchor is the last committed token (reference block position 0).
  if (prompt_tokens.size() != current_pos || prompt_tokens.empty())
    return out;
  if (current_pos == 0)
    return out;
  // Reference block layout (llama.cpp draft-dflash): the anchor overwrites
  // the last committed row, masks append after it. Block rows cover
  // [P-1, P+k); the draft length advances by k (one overlapping row).
  std::uint32_t k = std::min(max_tokens, max_draft_tokens_);
  if (current_pos + k > draft_model_->MaxContext()) {
    k = current_pos < draft_model_->MaxContext()
            ? draft_model_->MaxContext() - current_pos
            : 0;
  }
  if (k == 0)
    return out;
  const std::uint32_t base = session.Length();
  const std::uint32_t block_pos = base - 1;
  std::int32_t anchor =
      static_cast<std::int32_t>(prompt_tokens.back());
  if (std::getenv("GLIMMER_ZERO_ANCHOR") != nullptr)
    anchor = static_cast<std::int32_t>(draft_model_->MaskTokenId());
  const std::int32_t mask =
      static_cast<std::int32_t>(draft_model_->MaskTokenId());
  std::vector<std::int32_t> block;
  block.reserve(k + 1);
  block.push_back(anchor);
  for (std::uint32_t i = 0; i < k; ++i)
    block.push_back(mask);
  std::string error;
  if (!session.EmbedBlock(block, &error) ||
      !session.ForwardBlock(k + 1, block_pos, &error)) {
    (void)session.Rewind(base, nullptr);
    return speculative::DraftProposal{};
  }
  // Greedy argmax per mask position (reference top_k=10/p_min=0 at temp 0).
  for (std::uint32_t i = 1; i <= k; ++i) {
    if (!session.ScoreRow(i, &error)) {
      (void)session.Rewind(base, nullptr);
      return speculative::DraftProposal{};
    }
    const std::int32_t tok = session.SampleGreedy();
    out.tokens.push_back(static_cast<tokenization::TokenId>(tok));
  }
  propose_pos_ = block_pos;
  proposed_ = k;
  proposal_active_ = true;
  if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr) {
    std::fprintf(stderr, "[draft] round pos=%u anchor=%d drafts=",
                 current_pos, anchor);
    for (const auto t : out.tokens)
      std::fprintf(stderr, " %d", static_cast<int>(t));
    std::fprintf(stderr, "\n");
  }
  return out;
}

void GlimmerDflashDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)correction_token;
  if (!proposal_active_)
    return;
  drafted_ += proposed_;
  accepted_ += std::min(accepted.size(), static_cast<std::size_t>(proposed_));
  proposal_active_ = false;
}

void GlimmerDflashDraftBackend::Reset() noexcept {
  (void)draft_session_->Rewind(0, nullptr);
  drafted_ = 0;
  accepted_ = 0;
  proposal_active_ = false;
  primed_ = false;
  tail_feats_.clear();
  has_tail_ = false;
}

namespace {

bool RegisterGlimmerDflashDraftBackend() {
  speculative::RegisterDraftBackend(
      speculative::DraftBackendInfo{
          .name = "glimmer-dflash",
          .description =
              "Muse-Glimmer DFlash EAGLE-style feature draft "
              "(target-bound; prompt-first priming on fused tapped-layer "
              "features, greedy only, draft KV on the shared blocked-Q8 "
              "grid, predicts with the target head)",
          .requires_target_model = true,
          .requires_session = false,
      },
      [](const speculative::DraftBackendRequest& request,
         std::string* error_msg) -> std::unique_ptr<speculative::IDraftBackend> {
        if (request.target_kind != "glimmer" ||
            request.target_model == nullptr) {
          if (error_msg != nullptr) {
            *error_msg =
                "glimmer-dflash backend requires a glimmer target model "
                "(target_kind=\"glimmer\")";
          }
          return nullptr;
        }
        if (request.draft_model_path.empty()) {
          if (error_msg != nullptr) {
            *error_msg = "glimmer-dflash backend requires draft_model_path "
                         "(muse-glimmer DFlash GGUF)";
          }
          return nullptr;
        }
        auto target =
            std::static_pointer_cast<const Model>(request.target_model);
        std::uint32_t max_tokens = request.max_draft_tokens == 0
                                       ? 4u
                                       : request.max_draft_tokens;
        if (max_tokens > 8u) {
          max_tokens = 8u;
        }
        return GlimmerDflashDraftBackend::Create(
            std::move(target), request.draft_model_path, request.max_context,
            max_tokens, error_msg);
      });
  return true;
}

const bool kGlimmerDflashDraftRegistered = RegisterGlimmerDflashDraftBackend();

}  // namespace

}  // namespace gufo::models::glimmer
