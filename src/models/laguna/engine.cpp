#include "src/models/laguna/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/laguna/kernels/kernels.hpp"

namespace mbufo::models::laguna {
namespace {

constexpr std::uint32_t kMaxChunk = 512;

bool Check(hipError_t err, std::string* error_msg, const char* what) {
  if (err == hipSuccess) return true;
  if (error_msg != nullptr && error_msg->empty()) {
    *error_msg = std::string("laguna HIP failure in ") + what + ": " +
                 hipGetErrorString(err);
  }
  return false;
}

std::uint16_t F32ToF16Bits(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::int32_t exp =
      static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 112;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;
  if ((bits & 0x7FFFFFFFU) == 0) return static_cast<std::uint16_t>(sign);
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7BFFU);
  if (exp <= 0) return static_cast<std::uint16_t>(sign);
  const std::uint32_t half =
      sign | (static_cast<std::uint32_t>(exp) << 10U) | (mantissa >> 13U);
  const std::uint32_t round_bit = (mantissa >> 12U) & 1U;
  const std::uint32_t sticky = mantissa & 0xFFFU;
  if (round_bit != 0 && (sticky != 0 || (half & 1U) != 0)) {
    return static_cast<std::uint16_t>(half + 1U);
  }
  return static_cast<std::uint16_t>(half);
}

bool MirrorTensor(TensorRef& t, std::vector<void*>& owned,
                  std::string* error_msg) {
  if (t.empty()) return true;
  // Convert Q6_K routed expert-down weights to Q8_0 so LaunchDecodeMoeKQuant
  // handles every MoE layer uniformly.
  if (t.depth > 1 && t.type == core::GgmlType::kQ6_K) {
    const std::size_t elems = t.cols * t.rows * t.depth;
    if (elems > 0 && elems % 32 == 0) {
      std::vector<float> host_f32(elems);
      ::mbufo::quant::DequantizeQ6_K(t.data, host_f32.data(), elems);
      const std::size_t n_blocks = elems / 32;
      std::vector<std::uint8_t> q80(n_blocks * 34);
      for (std::size_t b = 0; b < n_blocks; ++b) {
        float amax = 0.0F;
        for (std::size_t i = 0; i < 32; ++i) {
          amax = std::max(amax, std::abs(host_f32[b * 32 + i]));
        }
        const float d = amax / 127.0F;
        const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
        auto* blk = q80.data() + b * 34;
        const std::uint16_t sb = F32ToF16Bits(d);
        std::memcpy(blk, &sb, sizeof(sb));
        for (std::size_t i = 0; i < 32; ++i) {
          long q = std::lround(host_f32[b * 32 + i] * id);
          q = std::clamp(q, -127L, 127L);
          blk[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
        }
      }
      void* dev = nullptr;
      if (!Check(hipMalloc(&dev, q80.size()), error_msg, "q80 expert alloc")) {
        return false;
      }
      if (!Check(hipMemcpy(dev, q80.data(), q80.size(), hipMemcpyHostToDevice),
                 error_msg, "q80 expert copy")) {
        (void)hipFree(dev);
        return false;
      }
      t.device = dev;
      t.type = core::GgmlType::kQ8_0;
      t.size_bytes = q80.size();
      owned.push_back(dev);
      return true;
    }
  }

  void* dev = nullptr;
  if (!Check(hipMalloc(&dev, t.size_bytes), error_msg, "weight mirror alloc")) {
    return false;
  }
  if (!Check(hipMemcpy(dev, t.data, t.size_bytes, hipMemcpyHostToDevice),
             error_msg, "weight mirror copy")) {
    (void)hipFree(dev);
    return false;
  }
  t.device = dev;
  owned.push_back(dev);
  if (t.type == core::GgmlType::kBF16) {
    kernels::ConvertBf16ToF16(static_cast<const std::uint16_t*>(dev),
                              static_cast<__half*>(dev), t.size_bytes / 2,
                              nullptr);
    if (!Check(hipStreamSynchronize(nullptr), error_msg, "bf16 convert")) {
      return false;
    }
    t.type = core::GgmlType::kF16;
  }
  return true;
}

bool UploadWeights(ModelWeights& w, std::vector<void*>& owned,
                   std::string* error_msg) {
  if (!MirrorTensor(w.token_embd, owned, error_msg) ||
      !MirrorTensor(w.output_norm, owned, error_msg) ||
      !MirrorTensor(w.output, owned, error_msg)) {
    return false;
  }
  for (auto& l : w.layers) {
    if (!MirrorTensor(l.attn_norm, owned, error_msg) ||
        !MirrorTensor(l.wq, owned, error_msg) ||
        !MirrorTensor(l.wk, owned, error_msg) ||
        !MirrorTensor(l.wv, owned, error_msg) ||
        !MirrorTensor(l.wo, owned, error_msg) ||
        !MirrorTensor(l.attn_q_norm, owned, error_msg) ||
        !MirrorTensor(l.attn_k_norm, owned, error_msg) ||
        !MirrorTensor(l.attn_gate, owned, error_msg) ||
        !MirrorTensor(l.ffn_norm, owned, error_msg)) {
      return false;
    }
    if (!l.is_moe) {
      if (!MirrorTensor(l.ffn_gate, owned, error_msg) ||
          !MirrorTensor(l.ffn_up, owned, error_msg) ||
          !MirrorTensor(l.ffn_down, owned, error_msg)) {
        return false;
      }
    } else {
      if (!MirrorTensor(l.ffn_gate_inp, owned, error_msg) ||
          !MirrorTensor(l.exp_probs_b, owned, error_msg) ||
          !MirrorTensor(l.ffn_gate_exps, owned, error_msg) ||
          !MirrorTensor(l.ffn_up_exps, owned, error_msg) ||
          !MirrorTensor(l.ffn_down_exps, owned, error_msg) ||
          !MirrorTensor(l.ffn_gate_shexp, owned, error_msg) ||
          !MirrorTensor(l.ffn_up_shexp, owned, error_msg) ||
          !MirrorTensor(l.ffn_down_shexp, owned, error_msg)) {
        return false;
      }
    }
  }
  return true;
}

core::compute::gemm::WeightType ToWeightType(core::GgmlType t) {
  return static_cast<core::compute::gemm::WeightType>(static_cast<int>(t));
}

void LagunaGemm(const TensorRef& w, const float* x, __half* x16, float* y,
                std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                hipStream_t stream) {
  namespace gemm = core::compute::gemm;
  const void* dev = w.device != nullptr ? w.device : w.data;
  if (n_tokens > 8 && w.type == core::GgmlType::kF16) {
    gemm::ConvertF32ToF16(x, x16, static_cast<std::size_t>(n_tokens) * k, stream);
    gemm::UnquantizedF16Gemm(dev, x16, y, n_tokens, m, k, stream);
    return;
  }
  if (n_tokens > 8 && w.type == core::GgmlType::kQ8_0) {
    gemm::ConvertF32ToF16(x, x16, static_cast<std::size_t>(n_tokens) * k, stream);
    gemm::DenseF16Gemm(dev, x16, y, n_tokens, m, k, stream);
    return;
  }
  for (std::uint32_t off = 0; off < n_tokens; off += 8) {
    const std::uint32_t chunk = std::min(8U, n_tokens - off);
    gemm::GemvBatched(dev, static_cast<int>(w.type), w.RowBytes(), k,
                      x + static_cast<std::size_t>(off) * k,
                      y + static_cast<std::size_t>(off) * m,
                      chunk, m, 0.0F, stream);
  }
}

}  // namespace

Model::~Model() {
  for (void* p : device_allocations_) {
    if (p != nullptr) (void)hipFree(p);
  }
}

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  auto reader = core::GgufReader::OpenFile(model_path, error_msg);
  if (reader == nullptr) return nullptr;

  auto weights = ModelWeights::LoadFromGguf(*reader, error_msg);
  if (!weights) return nullptr;

  auto tokenizer = tokenization::QwenTokenizer::CreateFromGguf(*reader, error_msg);
  if (tokenizer == nullptr) return nullptr;

  auto m = std::shared_ptr<Model>(new Model());
  m->options_ = options;
  if (m->options_.max_context == 0 ||
      m->options_.max_context > weights->config.context_length) {
    m->options_.max_context = weights->config.context_length;
  }
  m->reader_ = std::move(reader);
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  m->tokenizer_ = std::move(tokenizer);

  if (!UploadWeights(*m->weights_, m->device_allocations_, error_msg)) {
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(std::string* error_msg) {
  auto s = std::unique_ptr<Session>(new Session(shared_from_this()));
  if (!s->Init(error_msg)) return nullptr;
  return s;
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text,
                                          bool add_bos) const {
  const auto raw = tokenizer_->Encode(text);
  std::vector<std::int32_t> out;
  out.reserve(raw.size() + (add_bos ? 1 : 0));
  if (add_bos && tokenizer_->GetBosTokenId() != tokenization::kInvalidTokenId) {
    out.push_back(static_cast<std::int32_t>(tokenizer_->GetBosTokenId()));
  }
  for (const auto id : raw) out.push_back(static_cast<std::int32_t>(id));
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids;
  ids.reserve(tokens.size());
  for (const auto t : tokens) ids.push_back(static_cast<tokenization::TokenId>(t));
  return tokenizer_->Decode(ids);
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  const auto id = static_cast<tokenization::TokenId>(token);
  return id == tokenizer_->GetEosTokenId() ||
         id == tokenizer_->GetEosTokenId();
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::string Model::ModelName() const {
  return weights_->config.model_name;
}

std::size_t Model::ResidentBytes() const noexcept {
  std::size_t total = 0;
  for (const auto& l : weights_->layers) {
    total += l.attn_norm.size_bytes + l.wq.size_bytes + l.wk.size_bytes +
             l.wv.size_bytes + l.wo.size_bytes + l.attn_q_norm.size_bytes +
             l.attn_k_norm.size_bytes + l.attn_gate.size_bytes +
             l.ffn_norm.size_bytes + l.ffn_gate.size_bytes +
             l.ffn_up.size_bytes + l.ffn_down.size_bytes +
             l.ffn_gate_inp.size_bytes + l.exp_probs_b.size_bytes +
             l.ffn_gate_exps.size_bytes + l.ffn_up_exps.size_bytes +
             l.ffn_down_exps.size_bytes + l.ffn_gate_shexp.size_bytes +
             l.ffn_up_shexp.size_bytes + l.ffn_down_shexp.size_bytes;
  }
  total += weights_->token_embd.size_bytes + weights_->output_norm.size_bytes +
           weights_->output.size_bytes;
  return total;
}

Session::Session(std::shared_ptr<Model> model) : model_(std::move(model)) {}

Session::~Session() {
  if (stream_ != nullptr) (void)hipStreamSynchronize(stream_);
  for (void* p : k_cache_) if (p != nullptr) (void)hipFree(p);
  for (void* p : v_cache_) if (p != nullptr) (void)hipFree(p);
  auto free_p = [](auto* p) { if (p != nullptr) (void)hipFree(p); };
  free_p(d_tokens_);
  free_p(d_argmax_);
  free_p(d_hidden_);
  free_p(d_normed_);
  free_p(d_q_);
  free_p(d_k_);
  free_p(d_v_);
  free_p(d_gate_logits_);
  free_p(d_attn_out_);
  free_p(d_proj_out_);
  free_p(d_mlp_gate_);
  free_p(d_mlp_up_);
  free_p(d_mlp_out_);
  free_p(d_shared_out_);
  free_p(d_router_logits_);
  free_p(d_exp_ids_);
  free_p(d_exp_weights_);
  free_p(d_act_scratch_);
  free_p(d_down_e_);
  free_p(d_x16_);
  free_p(d_routed_act16_);
  free_p(d_exp_counts_);
  free_p(d_pad_bounds_);
  free_p(d_cursors_);
  free_p(d_rows_token_);
  free_p(d_rows_slot_);
  free_p(d_logits_);
  if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
}

bool Session::Init(std::string* error_msg) {
  if (!Check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
             error_msg, "stream create")) {
    return false;
  }
  const auto& cfg = model_->weights_->config;
  const std::size_t ctx = model_->MaxContext();
  const std::size_t kv_bytes =
      ctx * cfg.num_kv_heads * cfg.head_dim * sizeof(__half);

  k_cache_.resize(cfg.num_layers, nullptr);
  v_cache_.resize(cfg.num_layers, nullptr);
  for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
    if (!Check(hipMalloc(&k_cache_[l], kv_bytes), error_msg, "k_cache alloc") ||
        !Check(hipMalloc(&v_cache_[l], kv_bytes), error_msg, "v_cache alloc")) {
      return false;
    }
  }

  const std::size_t h = cfg.hidden_size;
  const std::size_t q_dim = static_cast<std::size_t>(cfg.max_q_heads) * cfg.head_dim;
  const std::size_t kv_dim = static_cast<std::size_t>(cfg.num_kv_heads) * cfg.head_dim;
  const std::size_t max_ff = std::max(cfg.intermediate_size, cfg.shared_expert_ffn_size);
  const std::size_t slots = static_cast<std::size_t>(kMaxChunk) * cfg.expert_used_count;
  routed_compact_rows_ =
      core::compute::gemm::RoutedCompactRows(slots, cfg.expert_count);

  auto alloc = [&](auto** ptr, std::size_t count, const char* what) -> bool {
    return Check(hipMalloc(reinterpret_cast<void**>(ptr),
                           count * sizeof(**ptr)),
                 error_msg, what);
  };

  if (!alloc(&d_tokens_, kMaxChunk, "d_tokens") ||
      !alloc(&d_argmax_, 1, "d_argmax") ||
      !alloc(&d_hidden_, kMaxChunk * h, "d_hidden") ||
      !alloc(&d_normed_, kMaxChunk * h, "d_normed") ||
      !alloc(&d_q_, kMaxChunk * q_dim, "d_q") ||
      !alloc(&d_k_, kMaxChunk * kv_dim, "d_k") ||
      !alloc(&d_v_, kMaxChunk * kv_dim, "d_v") ||
      !alloc(&d_gate_logits_, kMaxChunk * cfg.max_q_heads, "d_gate_logits") ||
      !alloc(&d_attn_out_, kMaxChunk * q_dim, "d_attn_out") ||
      !alloc(&d_proj_out_, kMaxChunk * h, "d_proj_out") ||
      !alloc(&d_mlp_gate_, kMaxChunk * max_ff, "d_mlp_gate") ||
      !alloc(&d_mlp_up_, kMaxChunk * max_ff, "d_mlp_up") ||
      !alloc(&d_mlp_out_, kMaxChunk * h, "d_mlp_out") ||
      !alloc(&d_shared_out_, kMaxChunk * h, "d_shared_out") ||
      !alloc(&d_router_logits_, kMaxChunk * cfg.expert_count, "d_router_logits") ||
      !alloc(&d_exp_ids_, slots, "d_exp_ids") ||
      !alloc(&d_exp_weights_, slots, "d_exp_weights") ||
      !alloc(&d_act_scratch_, routed_compact_rows_ * cfg.expert_ffn_size, "d_act_scratch") ||
      !alloc(&d_down_e_, slots * h, "d_down_e") ||
      !alloc(&d_x16_, kMaxChunk * std::max(h, q_dim), "d_x16") ||
      !alloc(&d_routed_act16_, routed_compact_rows_ * cfg.expert_ffn_size, "d_routed_act16") ||
      !alloc(&d_exp_counts_, cfg.expert_count, "d_exp_counts") ||
      !alloc(&d_pad_bounds_, cfg.expert_count + 1, "d_pad_bounds") ||
      !alloc(&d_cursors_, cfg.expert_count, "d_cursors") ||
      !alloc(&d_rows_token_, routed_compact_rows_, "d_rows_token") ||
      !alloc(&d_rows_slot_, routed_compact_rows_, "d_rows_slot") ||
      !alloc(&d_logits_, cfg.vocab_size, "d_logits")) {
    return false;
  }

  host_logits_.resize(cfg.vocab_size, 0.0F);
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.empty()) {
    if (error_msg != nullptr) *error_msg = "laguna Sync: empty prompt";
    return false;
  }
  if (prompt.size() > model_->MaxContext()) {
    if (error_msg != nullptr) *error_msg = "laguna Sync: prompt exceeds context";
    return false;
  }
  std::size_t common = 0;
  while (common < history_.size() && common < prompt.size() &&
         history_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && history_.size() == prompt.size()) {
    return true;
  }
  if (common == prompt.size() && history_.size() > prompt.size()) {
    // Re-evaluate final token so logits_ match prompt.back()
    common = prompt.size() - 1;
  }
  history_.assign(prompt.begin(), prompt.begin() + common);
  return Forward(prompt.subspan(common), static_cast<std::uint32_t>(common),
                 error_msg);
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  const std::int32_t arr[1] = {token};
  return Forward(arr, static_cast<std::uint32_t>(history_.size()), error_msg);
}

bool Session::Forward(std::span<const std::int32_t> tokens,
                      std::uint32_t start_pos, std::string* error_msg) {
  if (tokens.empty()) return true;
  if (start_pos > history_.size()) {
    if (error_msg != nullptr) *error_msg = "laguna Forward: start_pos out of range";
    return false;
  }
  if (start_pos + tokens.size() > model_->MaxContext()) {
    if (error_msg != nullptr) *error_msg = "laguna Forward: context overflow";
    return false;
  }
  history_.resize(start_pos);
  std::size_t off = 0;
  while (off < tokens.size()) {
    const std::uint32_t chunk = static_cast<std::uint32_t>(
        std::min<std::size_t>(kMaxChunk, tokens.size() - off));
    if (!ForwardTokens(tokens.subspan(off, chunk),
                       start_pos + static_cast<std::uint32_t>(off),
                       error_msg)) {
      return false;
    }
    history_.insert(history_.end(), tokens.begin() + off,
                    tokens.begin() + off + chunk);
    off += chunk;
  }
  return true;
}

bool Session::ForwardTokens(std::span<const std::int32_t> tokens,
                            std::uint32_t start_pos, std::string* error_msg) {
  namespace gemm = core::compute::gemm;
  const auto& w = *model_->weights_;
  const auto& cfg = w.config;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  const std::uint32_t h = cfg.hidden_size;

  if (!Check(hipMemcpyAsync(d_tokens_, tokens.data(), n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             error_msg, "copy tokens")) {
    return false;
  }

  gemm::EmbedGather(w.token_embd.device, static_cast<int>(w.token_embd.type),
                    d_tokens_, d_hidden_, n, h, stream_);

  const float attn_scale = 1.0F / std::sqrt(static_cast<float>(cfg.head_dim));

  for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
    const auto& lw = w.layers[l];
    const std::uint32_t q_heads = lw.q_heads;
    const std::uint32_t q_dim = q_heads * cfg.head_dim;
    const std::uint32_t kv_dim = cfg.num_kv_heads * cfg.head_dim;

    kernels::RmsNorm(d_hidden_, static_cast<const float*>(lw.attn_norm.device),
                     d_normed_, n, h, cfg.rms_norm_eps, stream_);

    LagunaGemm(lw.wq, d_normed_, d_x16_, d_q_, n, q_dim, h, stream_);
    LagunaGemm(lw.wk, d_normed_, d_x16_, d_k_, n, kv_dim, h, stream_);
    LagunaGemm(lw.wv, d_normed_, d_x16_, d_v_, n, kv_dim, h, stream_);
    LagunaGemm(lw.attn_gate, d_normed_, d_x16_, d_gate_logits_, n, q_heads, h,
               stream_);

    kernels::HeadRmsNorm(d_q_, static_cast<const float*>(lw.attn_q_norm.device),
                         n, q_heads, cfg.head_dim, cfg.rms_norm_eps, stream_);
    kernels::HeadRmsNorm(d_k_, static_cast<const float*>(lw.attn_k_norm.device),
                         n, cfg.num_kv_heads, cfg.head_dim, cfg.rms_norm_eps,
                         stream_);

    const std::uint32_t rot_dim = lw.is_swa ? cfg.rotary_dim_swa : cfg.rotary_dim;
    const float theta = lw.is_swa ? cfg.rope_theta_swa : cfg.rope_theta;
    kernels::ApplyDualRope(d_q_, d_k_, n, start_pos, q_heads, cfg.num_kv_heads,
                           cfg.head_dim, rot_dim, theta, lw.is_swa,
                           cfg.yarn_factor, cfg.yarn_orig_ctx,
                           cfg.yarn_beta_fast, cfg.yarn_beta_slow,
                           cfg.yarn_attn_factor, stream_);

    auto* kc = static_cast<__half*>(k_cache_[l]);
    auto* vc = static_cast<__half*>(v_cache_[l]);
    kernels::StoreKvF16(d_k_, d_v_, kc, vc, n, start_pos, cfg.num_kv_heads,
                        cfg.head_dim, stream_);

    const std::uint32_t swa = lw.is_swa ? cfg.sliding_window : 0U;
    if (n == 1) {
      kernels::F16AttentionDecode(d_q_, kc, vc, d_attn_out_, start_pos + 1, swa,
                                  q_heads, cfg.num_kv_heads, cfg.head_dim,
                                  attn_scale, stream_);
    } else {
      kernels::F16AttentionPrefill(d_q_, kc, vc, d_attn_out_, n, start_pos, swa,
                                   q_heads, cfg.num_kv_heads, cfg.head_dim,
                                   attn_scale, stream_);
    }

    kernels::SoftplusPerHeadGateMul(d_attn_out_, d_gate_logits_, n, q_heads,
                                    cfg.head_dim, stream_);
    LagunaGemm(lw.wo, d_attn_out_, d_x16_, d_proj_out_, n, h, q_dim, stream_);
    kernels::AddResidual(d_hidden_, d_proj_out_,
                         static_cast<std::size_t>(n) * h, stream_);

    kernels::RmsNorm(d_hidden_, static_cast<const float*>(lw.ffn_norm.device),
                     d_normed_, n, h, cfg.rms_norm_eps, stream_);

    if (!lw.is_moe) {
      const std::uint32_t ff = cfg.intermediate_size;
      LagunaGemm(lw.ffn_gate, d_normed_, d_x16_, d_mlp_gate_, n, ff, h, stream_);
      LagunaGemm(lw.ffn_up, d_normed_, d_x16_, d_mlp_up_, n, ff, h, stream_);
      kernels::SiluMul(d_mlp_gate_, d_mlp_up_, d_mlp_gate_,
                       static_cast<std::size_t>(n) * ff, stream_);
      LagunaGemm(lw.ffn_down, d_mlp_gate_, d_x16_, d_mlp_out_, n, h, ff, stream_);
      kernels::AddResidual(d_hidden_, d_mlp_out_,
                           static_cast<std::size_t>(n) * h, stream_);
    } else {
      const std::uint32_t sff = cfg.shared_expert_ffn_size;
      LagunaGemm(lw.ffn_gate_shexp, d_normed_, d_x16_, d_mlp_gate_, n, sff, h,
                 stream_);
      LagunaGemm(lw.ffn_up_shexp, d_normed_, d_x16_, d_mlp_up_, n, sff, h,
                 stream_);
      kernels::SiluMul(d_mlp_gate_, d_mlp_up_, d_mlp_gate_,
                       static_cast<std::size_t>(n) * sff, stream_);
      LagunaGemm(lw.ffn_down_shexp, d_mlp_gate_, d_x16_, d_shared_out_, n, h,
                 sff, stream_);

      LagunaGemm(lw.ffn_gate_inp, d_normed_, d_x16_, d_router_logits_, n,
                 cfg.expert_count, h, stream_);
      kernels::SigmoidBiasTopKRouter(
          d_router_logits_, static_cast<const float*>(lw.exp_probs_b.device),
          d_exp_ids_, d_exp_weights_, n, cfg.expert_count,
          cfg.expert_used_count, cfg.expert_weights_norm,
          cfg.expert_weights_scale, stream_);

      const bool decode_moe_ok =
          (n <= 8) &&
          gemm::LaunchDecodeMoeKQuant(
              lw.ffn_gate_exps.device, ToWeightType(lw.ffn_gate_exps.type),
              lw.ffn_up_exps.device, lw.ffn_down_exps.device,
              ToWeightType(lw.ffn_down_exps.type), d_normed_, d_exp_ids_,
              d_act_scratch_, d_down_e_, n, cfg.expert_used_count, h,
              cfg.expert_ffn_size, stream_);

      if (decode_moe_ok) {
        gemm::MoeEpilogue(d_down_e_, d_exp_weights_, d_mlp_out_, n,
                          cfg.expert_used_count, h, stream_);
      } else {
        gemm::ExpertCounts(d_exp_ids_, d_exp_counts_, n, cfg.expert_count,
                           cfg.expert_used_count, stream_);
        gemm::RoutedCompact(d_exp_ids_, d_exp_counts_, d_pad_bounds_,
                            d_cursors_, d_rows_token_, d_rows_slot_, n,
                            cfg.expert_used_count, cfg.expert_count, stream_);
        // Prefill routed path via RoutedF16Gemm
        const std::uint32_t eff = cfg.expert_ffn_size;
        gemm::ConvertF32ToF16(d_normed_, d_x16_,
                              static_cast<std::size_t>(n) * h, stream_);
        gemm::RoutedF16Gemm(
            lw.ffn_gate_exps.device, ToWeightType(lw.ffn_gate_exps.type),
            d_x16_, nullptr, 0, 16, d_pad_bounds_, d_rows_token_, nullptr,
            nullptr, d_act_scratch_, nullptr, eff, h, stream_);
        gemm::RoutedF16Gemm(
            lw.ffn_up_exps.device, ToWeightType(lw.ffn_up_exps.type),
            d_x16_, nullptr, 0, 16, d_pad_bounds_, d_rows_token_, nullptr,
            d_act_scratch_, nullptr, d_routed_act16_, eff, h, stream_);
        gemm::RoutedF16Gemm(
            lw.ffn_down_exps.device, ToWeightType(lw.ffn_down_exps.type),
            d_routed_act16_, nullptr, 0, 16, d_pad_bounds_, nullptr,
            d_rows_slot_, nullptr, d_down_e_, nullptr, h, eff, stream_);
        gemm::MoeEpilogue(d_down_e_, d_exp_weights_, d_mlp_out_, n,
                          cfg.expert_used_count, h, stream_);
      }

      kernels::AddResidual(d_mlp_out_, d_shared_out_,
                           static_cast<std::size_t>(n) * h, stream_);
      kernels::AddResidual(d_hidden_, d_mlp_out_,
                           static_cast<std::size_t>(n) * h, stream_);
    }
  }

  const float* last_hidden = d_hidden_ + static_cast<std::size_t>(n - 1) * h;
  kernels::RmsNorm(last_hidden,
                   static_cast<const float*>(w.output_norm.device), d_normed_,
                   1, h, cfg.rms_norm_eps, stream_);
  LagunaGemm(w.output, d_normed_, d_x16_, d_logits_, 1, cfg.vocab_size, h,
             stream_);

  if (!Check(hipMemcpyAsync(host_logits_.data(), d_logits_,
                            cfg.vocab_size * sizeof(float),
                            hipMemcpyDeviceToHost, stream_),
             error_msg, "copy logits") ||
      !Check(hipStreamSynchronize(stream_), error_msg, "sync forward")) {
    return false;
  }
  return true;
}

std::span<const float> Session::Logits() const noexcept {
  return host_logits_;
}

std::uint32_t Session::Length() const noexcept {
  return static_cast<std::uint32_t>(history_.size());
}

std::int32_t Session::SampleGreedy() {
  return kernels::Argmax(d_logits_, model_->VocabSize(), d_argmax_, stream_);
}

}  // namespace mbufo::models::laguna
