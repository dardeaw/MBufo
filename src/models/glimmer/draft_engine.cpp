// DFlash draft runtime (muse-glimmer companion draft, e.g. dflash-kquant):
// an EAGLE-style feature draft. A depth-pruned dense decoder (5 layers,
// same width/dim as the target) consumes FUSED TARGET hidden states and
// predicts with the TARGET's output head. Weights are F32 device mirrors
// (dequantized once at load); GEMM rides the shared SmallGemm core. KV
// lives on the shared blocked-Q8 grid (core blocked_kv); no F16 mirror.
#include "src/models/glimmer/draft_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "src/core/compute/attention/attention_dispatcher.hpp"
#include "src/core/compute/attention/blocked_kv.hpp"
#include "src/core/compute/attention/blocked_splitk.hpp"
#include "src/core/compute/gemm/gemm.hpp"
#include "src/models/glimmer/draft_model.hpp"
#include "src/models/glimmer/kernels/kernels.hpp"

namespace gufo::models::glimmer::draft {
namespace {

/// Draft prefill chunk width (priming loops in these blocks).
constexpr std::uint32_t kDraftChunk = 128;

// One F32 projection over the shared SmallGemm core (no activation
// staging: the core reads F32 rows directly).
void DraftGemm(const DraftTensorRef& w, const float* x, float* y,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream) {
  core::compute::gemm::fn_compat::SmallGemm(w.device, /*type=*/0, x, y,
                                            n_tokens, m, k, stream);
}

struct DraftSessionBuffers {
  std::vector<void*> allocations;
  // Blocked-Q8 KV caches, indexed by draft layer.
  std::vector<std::uint8_t*> kv_bk;
  std::vector<std::uint8_t*> kv_bv;
  std::uint32_t kv_row_bytes{0};

  template <typename T>
  T* Alloc(std::size_t count, std::string* error) {
    void* p = nullptr;
    if (hipMalloc(&p, count * sizeof(T)) != hipSuccess || p == nullptr) {
      if (error != nullptr && error->empty())
        *error = "glimmer draft session allocation failed";
      return nullptr;
    }
    allocations.push_back(p);
    return static_cast<T*>(p);
  }

  ~DraftSessionBuffers() {
    for (void* p : allocations) {
      (void)hipFree(p);
    }
  }
};

}  // namespace

struct DraftSession::Impl {
  DraftSessionBuffers buffers;
  float* feat{nullptr};  ///< [chunk][5h] uploaded target features (device)
  float* enc_normed{nullptr};  ///< [chunk][5h] encoder-norm staging
  float* fused{nullptr};  ///< [chunk][h] fused rows (device staging)
  float* hidden{nullptr};
  float* normed{nullptr};
  float* qbuf{nullptr};
  float* kbuf{nullptr};
  float* vbuf{nullptr};
  float* attn_out{nullptr};
  float* gate_out{nullptr};
  float* up_out{nullptr};
  float* ffn_out{nullptr};
  float* logits{nullptr};
  std::uint32_t* d_pos{nullptr};
  std::uint32_t* d_pos_win{nullptr};
  float* splitk_partials{nullptr};
  core::compute::gemm::ArgmaxCandidate* argmax_scratch{nullptr};
  std::int32_t* argmax_out{nullptr};
  std::uint32_t capacity{0};
  std::uint32_t length{0};
  std::uint32_t last_n{0};
  std::uint32_t feat_width{0};  ///< n_taps * hidden
  std::vector<float> host_logits;

  [[nodiscard]] bool Ensure(const DraftConfig& c, std::uint32_t capacity,
                            std::uint32_t vocab, std::string* error) {
    this->capacity = capacity;
    this->feat_width =
        static_cast<std::uint32_t>(c.target_layer_ids.size()) * c.hidden_size;
    const std::uint32_t hidden_dim = c.hidden_size;
    const std::uint32_t qdim = c.num_heads * c.head_dim;
    const std::uint32_t kvdim = c.num_kv_heads * c.head_dim;
    const std::size_t chunk = kDraftChunk;
    feat = buffers.Alloc<float>(chunk * feat_width, error);
    enc_normed = buffers.Alloc<float>(chunk * feat_width, error);
    fused = buffers.Alloc<float>(chunk * hidden_dim, error);
    hidden = buffers.Alloc<float>(chunk * hidden_dim, error);
    normed = buffers.Alloc<float>(chunk * hidden_dim, error);
    qbuf = buffers.Alloc<float>(chunk * qdim, error);
    kbuf = buffers.Alloc<float>(chunk * kvdim, error);
    vbuf = buffers.Alloc<float>(chunk * kvdim, error);
    attn_out = buffers.Alloc<float>(chunk * qdim, error);
    gate_out = buffers.Alloc<float>(chunk * c.intermediate_size, error);
    up_out = buffers.Alloc<float>(chunk * c.intermediate_size, error);
    ffn_out = buffers.Alloc<float>(chunk * hidden_dim, error);
    logits = buffers.Alloc<float>(vocab, error);
    d_pos = buffers.Alloc<std::uint32_t>(1, error);
    d_pos_win = buffers.Alloc<std::uint32_t>(1, error);
    splitk_partials = buffers.Alloc<float>(
        core::compute::blocked_splitk::
            AttentionBlockedSplitKScratchBytes() /
            sizeof(float),
        error);
    argmax_scratch = buffers.Alloc<core::compute::gemm::ArgmaxCandidate>(
        core::compute::gemm::kArgmaxParts, error);
    argmax_out = buffers.Alloc<std::int32_t>(1, error);
    host_logits.assign(vocab, 0.0f);
    return feat != nullptr && enc_normed != nullptr &&
           fused != nullptr && hidden != nullptr &&
           normed != nullptr && qbuf != nullptr && kbuf != nullptr &&
           vbuf != nullptr && attn_out != nullptr && gate_out != nullptr &&
           up_out != nullptr && ffn_out != nullptr && logits != nullptr &&
           d_pos != nullptr && d_pos_win != nullptr &&
           splitk_partials != nullptr && argmax_scratch != nullptr &&
           argmax_out != nullptr;
  }
};

DraftModel::~DraftModel() {
  for (void* p : device_allocations_) {
    (void)hipFree(p);
  }
}

std::shared_ptr<DraftModel> DraftModel::Load(
    const std::string& model_path, const DraftModelOptions& options,
    const std::vector<std::uint32_t>& target_layer_ids,
    std::string* error_msg) {
  std::string load_error;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
  if (reader_owner == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "Failed to open dflash GGUF: " + load_error;
    return nullptr;
  }
  std::shared_ptr<core::GgufReader> reader(std::move(reader_owner));
  std::vector<void*> owned;
  auto weights = DraftModelWeights::Bind(*reader, owned, &load_error);
  if (!weights.has_value()) {
    if (error_msg != nullptr)
      *error_msg = "Failed to bind dflash weights: " + load_error;
    return nullptr;
  }
  if (weights->config.head_dim != 128) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft runtime supports head_dim 128 only";
    return nullptr;
  }
  if (!target_layer_ids.empty() &&
      target_layer_ids != weights->config.target_layer_ids) {
    if (error_msg != nullptr)
      *error_msg = "dflash target_layer_ids mismatch";
    return nullptr;
  }
  auto model_owner = std::shared_ptr<DraftModel>(new DraftModel());
  model_owner->options_ = options;
  model_owner->reader_ = std::move(reader);
  model_owner->weights_ =
      std::make_unique<DraftModelWeights>(std::move(*weights));
  model_owner->device_allocations_ = std::move(owned);
  model_owner->target_layer_ids_ = model_owner->weights_->config.target_layer_ids;
  return model_owner;
}

DraftSession::DraftSession(std::shared_ptr<DraftModel> model)
    : model_(std::move(model)), impl_(std::make_unique<Impl>()) {}

DraftSession::~DraftSession() = default;

std::unique_ptr<DraftSession> DraftModel::CreateSession(
    std::string* error_msg) {
  if (target_head_.output.empty() || target_head_.vocab_size == 0) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft target head is not bound";
    return nullptr;
  }
  auto session = std::unique_ptr<DraftSession>(new DraftSession(
      shared_from_this()));
  std::string error;
  if (!session->impl_->Ensure(session->model_->weights_->config,
                              options_.max_context,
                              target_head_.vocab_size, &error)) {
    if (error_msg != nullptr)
      *error_msg = error;
    return nullptr;
  }
  const auto& c = session->model_->weights_->config;
  auto& impl = *session->impl_;
  impl.buffers.kv_bk.assign(c.num_layers, nullptr);
  impl.buffers.kv_bv.assign(c.num_layers, nullptr);
  // Shared blocked grid: (d/32) blocks of 34 bytes per KV head, packed per
  // row (same convention as the target q8 path).
  impl.buffers.kv_row_bytes =
      (c.head_dim / 32) * 34 * c.num_kv_heads;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const std::size_t bytes = static_cast<std::size_t>(
                                  options_.max_context) *
                              impl.buffers.kv_row_bytes;
    auto* kb = impl.buffers.Alloc<std::uint8_t>(bytes, &error);
    auto* vb = impl.buffers.Alloc<std::uint8_t>(bytes, &error);
    if (kb == nullptr || vb == nullptr) {
      if (error_msg != nullptr)
        *error_msg = "draft blocked KV cache allocation failed (layer " +
                     std::to_string(il) + ")";
      return nullptr;
    }
    (void)hipMemset(kb, 0, bytes);
    (void)hipMemset(vb, 0, bytes);
    impl.buffers.kv_bk[il] = kb;
    impl.buffers.kv_bv[il] = vb;
  }
  return session;
}

void DraftSession::ForwardLayer(DraftSession& s, std::uint32_t il,
                               std::uint32_t n, std::uint32_t start_pos,
                               hipStream_t stream) {
  auto& w = *s.model_->weights_;
  const auto& c = w.config;
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t d = c.head_dim;
  const std::uint32_t qdim = c.num_heads * d;
  const std::uint32_t kvdim = c.num_kv_heads * d;
  const auto& l = w.layers[il];
  auto& impl = *s.impl_;
  const float score_scale = 1.0F / std::sqrt(static_cast<float>(d));
  const std::uint32_t window = c.sliding_window;

  float* hidden_buf = impl.hidden;
  // Pre-attention norm + QKV (F32 SmallGemm, no staging).
  kernels::RmsNorm(hidden_buf,
                   static_cast<const float*>(l.attn_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  DraftGemm(l.wq, impl.normed, impl.qbuf, n, qdim, hidden, stream);
  DraftGemm(l.wk, impl.normed, impl.kbuf, n, kvdim, hidden, stream);
  DraftGemm(l.wv, impl.normed, impl.vbuf, n, kvdim, hidden, stream);
  // QK-norm (per-head RMS over d).
  kernels::RmsNorm(impl.qbuf,
                   static_cast<const float*>(l.attn_q_norm.device),
                   impl.qbuf, n * c.num_heads, d, c.rms_eps, stream);
  kernels::RmsNorm(impl.kbuf,
                   static_cast<const float*>(l.attn_k_norm.device),
                   impl.kbuf, n * c.num_kv_heads, d, c.rms_eps, stream);
  // RoPE on every layer (the draft is all-SWA).
  kernels::Rope(impl.qbuf, n, c.num_heads, d, start_pos, c.rope_theta,
                stream);
  kernels::Rope(impl.kbuf, n, c.num_kv_heads, d, start_pos, c.rope_theta,
                stream);
  // Post-RoPE rows quantize into the shared blocked-Q8 grid.
  namespace blocked = core::compute::blocked_kv;
  (void)blocked::QuantizeKvRows(
      impl.kbuf, kvdim, d, impl.buffers.kv_bk[il], kvdim, c.num_kv_heads,
      d, n, blocked::KvCodec::kQ8, 127.0F, 34, impl.d_pos, stream);
  (void)blocked::QuantizeKvRows(
      impl.vbuf, kvdim, d, impl.buffers.kv_bv[il], kvdim, c.num_kv_heads,
      d, n, blocked::KvCodec::kQ8, 127.0F, 34, impl.d_pos, stream);
  // Blocked-Q8 read: SWA layers pre-shift past the window (dense-causal
  // from here), then the shared dispatcher picks Split-K decode for long
  // contexts and the scalar direct read otherwise.
  const std::uint8_t* k2 = impl.buffers.kv_bk[il];
  const std::uint8_t* v2 = impl.buffers.kv_bv[il];
  const std::uint32_t* sp = impl.d_pos;
  std::uint32_t base = start_pos;
  if (window != 0 && start_pos + 1 > window) {
    const std::uint32_t lo = (start_pos + 1 - window) & ~3u;
    if (lo != 0 && lo <= start_pos) {
      base = start_pos - lo;
      if (hipMemcpy(impl.d_pos_win, &base, sizeof(base),
                    hipMemcpyHostToDevice) == hipSuccess) {
        k2 += static_cast<std::size_t>(lo) * impl.buffers.kv_row_bytes;
        v2 += static_cast<std::size_t>(lo) * impl.buffers.kv_row_bytes;
        sp = impl.d_pos_win;
      }
    }
  }
  namespace attn_disp = core::compute::attention_dispatcher;
  attn_disp::AttentionDispatchInput din;
  din.q_len = n;
  din.kv_len = base + n;
  din.head_dim = d;
  din.heads = c.num_heads;
  din.kv_heads = c.num_kv_heads;
  din.k_codec = attn_disp::DispatchCodec::kQ8;
  din.v_codec = attn_disp::DispatchCodec::kQ8;
  din.window = 0;
  din.is_verify = false;
  const auto route = attn_disp::RouteAttention(
      din, attn_disp::AttentionDispatchPolicy::WithEnvDefaults());
  bool ran = false;
  if (route == attn_disp::AttentionRoute::kBlockedSplitK) {
    ran = core::compute::blocked_splitk::AttentionBlockedSplitK<
        core::compute::blocked_splitk::KvCodecQ8, 128>(
        impl.qbuf, k2, v2, impl.attn_out, n, sp, base + n, c.num_heads,
        c.num_kv_heads, d, stream, score_scale, 34, 34,
        impl.splitk_partials,
        core::compute::blocked_splitk::
            AttentionBlockedSplitKScratchBytes());
  }
  if (!ran) {
    blocked::AttentionKv(
        impl.qbuf, k2, v2, /*mask=*/nullptr, /*mask_words=*/0,
        impl.attn_out, /*partials=*/nullptr, /*splits=*/1, n, sp,
        c.num_heads, c.num_kv_heads, d, /*ratio=*/1, 34, 34,
        blocked::KvCodec::kQ8, stream, score_scale);
  }
  // No attention gate, no post-attention norm: plain residual add.
  DraftGemm(l.wo, impl.attn_out, impl.ffn_out, n, hidden, qdim, stream);
  kernels::ResidualAdd(impl.ffn_out, hidden_buf,
                       static_cast<std::size_t>(n) * hidden, stream);

  // FFN block (SwiGLU, no post-FFN norm).
  kernels::RmsNorm(hidden_buf,
                   static_cast<const float*>(l.ffn_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  DraftGemm(l.ffn_gate, impl.normed, impl.gate_out, n,
            c.intermediate_size, hidden, stream);
  DraftGemm(l.ffn_up, impl.normed, impl.up_out, n, c.intermediate_size,
            hidden, stream);
  kernels::SiluMul(impl.gate_out, impl.up_out, impl.gate_out,
                   static_cast<std::size_t>(n) * c.intermediate_size,
                   stream);
  DraftGemm(l.ffn_down, impl.gate_out, impl.ffn_out, n, hidden,
            c.intermediate_size, stream);
  kernels::ResidualAdd(impl.ffn_out, hidden_buf,
                       static_cast<std::size_t>(n) * hidden, stream);
}

void DraftSession::ForwardFused(DraftSession& s, std::uint32_t n,
                               std::uint32_t start_pos, hipStream_t stream) {
  auto& w = *s.model_->weights_;
  auto& impl = *s.impl_;
  for (std::uint32_t il = 0; il < w.config.num_layers; ++il) {
    ForwardLayer(s, il, n, start_pos, stream);
    (void)hipStreamSynchronize(stream);
  }
  impl.length = start_pos + n;
  impl.last_n = n;
}

bool DraftSession::FuseRows(std::span<const float> feats, std::uint32_t n,
                            std::string* error_msg) {
  const std::uint32_t hidden = model_->weights_->config.hidden_size;
  auto& impl = *impl_;
  if (feats.size() < static_cast<std::size_t>(n) * impl.feat_width) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft FuseRows needs n fused feature rows";
    return false;
  }
  if (n == 0 || n > kDraftChunk) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft FuseRows needs 1..128 rows";
    return false;
  }
  hipStream_t stream = 0;
  (void)hipMemcpyAsync(impl.feat, feats.data(),
                       static_cast<std::size_t>(n) * impl.feat_width *
                           sizeof(float),
                       hipMemcpyHostToDevice, stream);
  (void)hipStreamSynchronize(stream);
  const auto& dw = *model_->weights_;
  // 27B DFlash2 convention: project the concatenated tapped features
  // first, then normalize the fused row (enc norm is [hidden]-wide).
  DraftGemm(dw.fc, impl.feat, impl.fused, n, hidden, impl.feat_width,
            stream);
  kernels::RmsNorm(impl.fused,
                   static_cast<const float*>(dw.enc_norm.device),
                   impl.fused, n, hidden, dw.config.rms_eps, stream);
  (void)hipStreamSynchronize(stream);
  return true;
}

bool DraftSession::PrimeRows(std::span<const float> feats, std::uint32_t n,
                             std::uint32_t start_pos, std::string* error_msg) {
  if (n == 0) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft PrimeRows needs rows";
    return false;
  }
  if (start_pos + n > model_->options_.max_context) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft context exhausted";
    return false;
  }
  auto& impl = *impl_;
  hipStream_t stream = 0;
  for (std::uint32_t off = 0; off < n;) {
    const std::uint32_t len =
        std::min(kDraftChunk, n - off);
    if (!FuseRows(feats.subspan(static_cast<std::size_t>(off) *
                                    impl.feat_width,
                                static_cast<std::size_t>(len) *
                                    impl.feat_width),
                  len, error_msg))
      return false;
    const std::uint32_t pos = start_pos + off;
    (void)hipMemcpyAsync(impl.d_pos, &pos, sizeof(pos),
                         hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(impl.hidden, impl.fused,
                         static_cast<std::size_t>(len) *
                             model_->weights_->config.hidden_size *
                             sizeof(float),
                         hipMemcpyDeviceToDevice, stream);
    ForwardFused(*this, len, pos, stream);
    off += len;
  }
  (void)hipStreamSynchronize(0);
  return true;
}

bool DraftSession::ExtendRow(const float* d_row, std::string* error_msg) {
  auto& impl = *impl_;
  if (impl.length + 1 > model_->options_.max_context) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft context exhausted";
    return false;
  }
  hipStream_t stream = 0;
  const std::uint32_t pos = impl.length;
  (void)hipMemcpyAsync(impl.d_pos, &pos, sizeof(pos), hipMemcpyHostToDevice,
                       stream);
  if (d_row != impl.hidden) {
    (void)hipMemcpyAsync(impl.hidden, d_row,
                         model_->weights_->config.hidden_size * sizeof(float),
                         hipMemcpyDeviceToDevice, stream);
  }
  ForwardFused(*this, 1, pos, stream);
  return true;
}

bool DraftSession::ScoreLast(std::string* error_msg) {
  auto& w = *model_->weights_;
  const auto& head = model_->target_head_;
  if (head.output.empty() || head.vocab_size == 0) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft target head is not bound";
    return false;
  }
  const std::uint32_t hidden = model_->weights_->config.hidden_size;
  auto& impl = *impl_;
  hipStream_t stream = 0;
  const std::uint32_t n = impl.last_n == 0 ? 1 : impl.last_n;
  const float* last_row =
      impl.hidden + static_cast<std::size_t>(n - 1) * hidden;
  kernels::RmsNorm(last_row,
                   static_cast<const float*>(w.output_norm.device),
                   impl.normed, 1, hidden, w.config.rms_eps, stream);
  const void* dev =
      head.output.device != nullptr ? head.output.device : head.output.data;
  core::compute::gemm::GemvBatched(dev, static_cast<int>(head.output.type),
                                   head.output.RowBytes(), hidden,
                                   impl.normed, impl.logits, 1,
                                   head.vocab_size, 0.0F, stream);
  if (head.logit_scale != 1.0F) {
    kernels::ScaleInplace(impl.logits, head.logit_scale, head.vocab_size,
                          stream);
  }
  kernels::SoftCap(impl.logits, head.final_logit_softcapping, head.vocab_size,
                   stream);
  (void)hipMemcpyAsync(impl.host_logits.data(), impl.logits,
                       impl.host_logits.size() * sizeof(float),
                       hipMemcpyDeviceToHost, stream);
  return true;
}

const float* DraftSession::FusedStaging() const noexcept {
  return impl_->fused;
}

const float* DraftSession::LastHiddenRow() const noexcept {
  const std::uint32_t n = impl_->last_n == 0 ? 1 : impl_->last_n;
  return impl_->hidden +
         static_cast<std::size_t>(n - 1) * model_->weights_->config.hidden_size;
}

bool DraftSession::Rewind(std::uint32_t new_length, std::string* error_msg) {
  if (new_length > impl_->length) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft rewind past the current length";
    return false;
  }
  // KV rows beyond new_length stay valid and are overwritten by the next
  // forwards; only the length marker moves back.
  impl_->length = new_length;
  return true;
}

std::uint32_t DraftSession::Length() const noexcept { return impl_->length; }

std::int32_t DraftSession::SampleGreedy() {
  auto& impl = *impl_;
  core::compute::gemm::Argmax(impl.logits, impl.argmax_scratch,
                              impl.argmax_out, 1,
                              model_->target_head_.vocab_size, 0);
  std::int32_t tok = -1;
  (void)hipMemcpy(&tok, impl.argmax_out, sizeof(tok), hipMemcpyDeviceToHost);
  return tok;
}

}  // namespace gufo::models::glimmer::draft
