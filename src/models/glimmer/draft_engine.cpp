// DFlash draft runtime (muse-glimmer companion draft, e.g. dflash-kquant):
// a block-diffusion feature draft. A depth-pruned dense decoder (5 layers,
// same width/dim as the target) consumes FUSED TARGET hidden states and
// predicts with the TARGET's output head. Weights are F32 device mirrors
// (dequantized once at load); KV lives on the shared blocked-Q8 grid
// (core blocked_kv); no F16 mirror.
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
  std::uint32_t* d_pos_full{nullptr};  ///< shared end position (device)
  std::int32_t* d_tok{nullptr};  ///< [chunk] staged block token ids (device)
  float* splitk_partials{nullptr};
  core::compute::gemm::ArgmaxCandidate* argmax_scratch{nullptr};
  std::int32_t* argmax_out{nullptr};
  std::uint32_t capacity{0};
  std::uint32_t length{0};
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
    d_pos_full = buffers.Alloc<std::uint32_t>(1, error);
    d_tok = buffers.Alloc<std::int32_t>(chunk, error);
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
           d_pos_full != nullptr && d_tok != nullptr &&
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

std::uint32_t DraftModel::HiddenSize() const noexcept {
  return weights_ != nullptr ? weights_->config.hidden_size : 0;
}

std::uint32_t DraftModel::MaskTokenId() const noexcept {
  return weights_ != nullptr ? weights_->config.mask_token_id : 0;
}

DraftSession::DraftSession(std::shared_ptr<DraftModel> model)
    : model_(std::move(model)), impl_(std::make_unique<Impl>()) {}

DraftSession::~DraftSession() = default;

std::unique_ptr<DraftSession> DraftModel::CreateSession(
    std::string* error_msg) {
  if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr) {
    const auto& ww = weights_->config;
    (void)ww;
    const auto& layers = weights_->layers;
    std::fprintf(stderr, "[draft] wq head vals:");
    for (std::uint32_t il = 0; il < layers.size(); ++il) {
      float h[4] = {0, 0, 0, 0};
      (void)hipMemcpy(h, layers[il].wq.device, sizeof(h),
                      hipMemcpyDeviceToHost);
      std::fprintf(stderr, " L%u=%.5f", il, h[0]);
    }
    std::fprintf(stderr, "\n");
  }
  if (target_head_.output.empty() || target_head_.vocab_size == 0) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft target head is not bound";
    return nullptr;
  }
  if (target_embed_.device == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft target embedding is not bound";
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

// One diffusion-decoder layer over staged block rows with FULL visibility:
// like the reference non-causal block decode, every query attends over the
// whole prefix plus the block (the block K/V is appended first, then each
// query re-reads with the shared end position). QKV/FFN math mirrors the
// reference token branch (pre-attn norm, QK-norm, RoPE, plain residuals,
// SwiGLU FFN without post norms).
void DraftSession::ForwardBlockLayer(DraftSession& s, std::uint32_t il,
                                     std::uint32_t n,
                                     std::uint32_t start_pos,
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
  // NeoX RoPE on every layer (the draft is all-SWA): the dflash artifact
  // trains with split-half pairing (LLAMA_ROPE_TYPE_NEOX), not the
  // target's consecutive-pair rotation.
  if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr && il == 0) {
    (void)hipStreamSynchronize(stream);
    std::vector<float> qpre(static_cast<std::size_t>(n) * qdim);
    (void)hipMemcpy(qpre.data(), impl.qbuf, qpre.size() * sizeof(float),
                    hipMemcpyDeviceToHost);
    std::fprintf(stderr, "[draft] rope-pre rows:");
    for (std::uint32_t r = 0; r < n; ++r) {
      const float* row = qpre.data() + static_cast<std::size_t>(r) * qdim;
      std::fprintf(stderr, " %.4f/%.4f", row[0], row[1]);
    }
    std::fprintf(stderr, "\n");
  }
  kernels::RopeNeox(impl.qbuf, n, c.num_heads, d, start_pos, c.rope_theta,
                stream);
  if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr && il == 0) {
    (void)hipStreamSynchronize(stream);
    std::vector<float> qpost(static_cast<std::size_t>(n) * qdim);
    (void)hipMemcpy(qpost.data(), impl.qbuf, qpost.size() * sizeof(float),
                    hipMemcpyDeviceToHost);
    std::fprintf(stderr, "[draft] rope-post rows:");
    for (std::uint32_t r = 0; r < n; ++r) {
      const float* row = qpost.data() + static_cast<std::size_t>(r) * qdim;
      std::fprintf(stderr, " %.4f/%.4f", row[0], row[1]);
    }
    std::fprintf(stderr, "\n");
  }
  kernels::RopeNeox(impl.kbuf, n, c.num_kv_heads, d, start_pos, c.rope_theta,
                stream);
  // Append the block K/V rows, then re-read every query with the shared
  // end position: non-causal (full-prefix-plus-block) visibility.
  namespace blocked = core::compute::blocked_kv;
  (void)hipMemcpyAsync(impl.d_pos, &start_pos, sizeof(start_pos),
                       hipMemcpyHostToDevice, stream);
  (void)blocked::QuantizeKvRows(
      impl.kbuf, kvdim, d, impl.buffers.kv_bk[il], kvdim, c.num_kv_heads,
      d, n, blocked::KvCodec::kQ8, 127.0F, 34, impl.d_pos, stream);
  (void)blocked::QuantizeKvRows(
      impl.vbuf, kvdim, d, impl.buffers.kv_bv[il], kvdim, c.num_kv_heads,
      d, n, blocked::KvCodec::kQ8, 127.0F, 34, impl.d_pos, stream);
  std::uint32_t pos_full = start_pos + n - 1;
  if (std::getenv("GLIMMER_PREFIX_ONLY") != nullptr && start_pos > 0) {
    pos_full = start_pos - 1;
  }
  if (std::getenv("GLIMMER_ZERO_FUTUREKV") != nullptr && n > 2) {
    const std::size_t rb = impl.buffers.kv_row_bytes;
    (void)hipMemsetAsync(
        const_cast<std::uint8_t*>(impl.buffers.kv_bk[il]) +
            static_cast<std::size_t>(start_pos + 2) * rb,
        0, static_cast<std::size_t>(n - 2) * rb, stream);
    (void)hipMemsetAsync(
        impl.buffers.kv_bv[il] + static_cast<std::size_t>(start_pos + 2) * rb,
        0, static_cast<std::size_t>(n - 2) * rb, stream);
    (void)hipStreamSynchronize(stream);
  }
  const std::uint8_t* k2 = impl.buffers.kv_bk[il];
  const std::uint8_t* v2 = impl.buffers.kv_bv[il];
  const std::uint32_t* sp = impl.d_pos_full;
  std::uint32_t base = pos_full;
  (void)hipMemcpyAsync(impl.d_pos_full, &pos_full, sizeof(pos_full),
                       hipMemcpyHostToDevice, stream);
  if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr) {
    (void)hipStreamSynchronize(stream);
    std::uint32_t pf_back = 0xFFFFFFFFu;
    std::uint32_t w_back = 0xFFFFFFFFu;
    (void)hipMemcpy(&pf_back, impl.d_pos_full, sizeof(pf_back),
                    hipMemcpyDeviceToHost);
    (void)hipMemcpy(&w_back, impl.d_pos_win, sizeof(w_back),
                    hipMemcpyDeviceToHost);
    std::fprintf(stderr, "[draft] fullvis pos_full=%u d_pos_full=%u win=%u\n",
                 pos_full, pf_back, w_back);
  }
  if (window != 0 && pos_full + 1 > window) {
    const std::uint32_t lo = (pos_full + 1 - window) & ~3u;
    if (lo != 0 && lo <= pos_full) {
      base = pos_full - lo;
      (void)hipMemcpyAsync(impl.d_pos_win, &base, sizeof(base),
                           hipMemcpyHostToDevice, stream);
      k2 += static_cast<std::size_t>(lo) * impl.buffers.kv_row_bytes;
      v2 += static_cast<std::size_t>(lo) * impl.buffers.kv_row_bytes;
      sp = impl.d_pos_win;
    }
  }
  const std::uint32_t kv_len = base + 1;
  namespace attn_disp = core::compute::attention_dispatcher;
  attn_disp::AttentionDispatchInput din;
  din.q_len = 1;
  din.kv_len = kv_len;
  din.head_dim = d;
  din.heads = c.num_heads;
  din.kv_heads = c.num_kv_heads;
  din.k_codec = attn_disp::DispatchCodec::kQ8;
  din.v_codec = attn_disp::DispatchCodec::kQ8;
  din.window = 0;
  din.is_verify = false;
  const auto route = attn_disp::RouteAttention(
      din, attn_disp::AttentionDispatchPolicy::WithEnvDefaults());
  if (std::getenv("GLIMMER_FORCE_CAUSAL") != nullptr) {
    (void)hipMemcpyAsync(impl.d_pos, &start_pos, sizeof(start_pos),
                         hipMemcpyHostToDevice, stream);
    const std::uint32_t* spc = impl.d_pos;
    const std::uint8_t* k2c = impl.buffers.kv_bk[il];
    const std::uint8_t* v2c = impl.buffers.kv_bv[il];
    if (route == attn_disp::AttentionRoute::kBlockedSplitK) {
      (void)core::compute::blocked_splitk::AttentionBlockedSplitK<
          core::compute::blocked_splitk::KvCodecQ8, 128>(
          impl.qbuf, k2c, v2c, impl.attn_out, n, spc, start_pos + n,
          c.num_heads, c.num_kv_heads, d, stream, score_scale, 34, 34,
          impl.splitk_partials,
          core::compute::blocked_splitk::
              AttentionBlockedSplitKScratchBytes());
    } else {
      blocked::AttentionKv(
          impl.qbuf, k2c, v2c, /*mask=*/nullptr, /*mask_words=*/0,
          impl.attn_out, /*partials=*/nullptr, /*splits=*/1, n, spc,
          c.num_heads, c.num_kv_heads, d, /*ratio=*/1, 34, 34,
          blocked::KvCodec::kQ8, stream, score_scale);
    }
  } else for (std::uint32_t t = 0; t < n; ++t) {
    const float* q_row =
        impl.qbuf + static_cast<std::size_t>(t) * qdim;
    float* out_row =
        impl.attn_out + static_cast<std::size_t>(t) * qdim;
    bool ran = false;
    if (route == attn_disp::AttentionRoute::kBlockedSplitK) {
      ran = core::compute::blocked_splitk::AttentionBlockedSplitK<
          core::compute::blocked_splitk::KvCodecQ8, 128>(
          q_row, k2, v2, out_row, 1, sp, kv_len, c.num_heads,
          c.num_kv_heads, d, stream, score_scale, 34, 34,
          impl.splitk_partials,
          core::compute::blocked_splitk::
              AttentionBlockedSplitKScratchBytes());
    }
    if (!ran) {
      blocked::AttentionKv(
          q_row, k2, v2, /*mask=*/nullptr, /*mask_words=*/0,
          out_row, /*partials=*/nullptr, /*splits=*/1, 1, sp,
          c.num_heads, c.num_kv_heads, d, /*ratio=*/1, 34, 34,
          blocked::KvCodec::kQ8, stream, score_scale);
    }
  }
  if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr && il == 0 &&
      n >= 3) {
    // F32-vs-Q8 fidelity: compare kbuf (pre-quant F32) against the cached
    {
      std::vector<float> kf(static_cast<std::size_t>(n) * kvdim);
      (void)hipMemcpy(kf.data(), impl.kbuf, kf.size() * sizeof(float),
                      hipMemcpyDeviceToHost);
      const std::uint32_t rb = impl.buffers.kv_row_bytes;
      std::vector<std::uint8_t> kq(rb);
      (void)hipMemcpy(kq.data(),
                      impl.buffers.kv_bk[0] +
                          static_cast<std::size_t>(start_pos) * rb,
                      rb, hipMemcpyDeviceToHost);
      auto f16toF = [](std::uint16_t v) {
        std::uint32_t sign = (v >> 15) & 1u;
        std::uint32_t exp = (v >> 10) & 0x1Fu;
        std::uint32_t mant = v & 0x3FFu;
        std::uint32_t f;
        if (exp == 0)
          f = sign << 31;
        else if (exp == 31)
          f = (sign << 31) | 0x7F800000u;
        else
          f = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
        float out = 0.0F;
        std::memcpy(&out, &f, sizeof(float));
        return out;
      };
      std::fprintf(stderr, "[draft] q8err rows:");
      for (std::uint32_t r = 0; r < n; ++r) {
        (void)hipMemcpy(kq.data(),
                        impl.buffers.kv_bk[0] +
                            static_cast<std::size_t>(start_pos + r) * rb,
                        rb, hipMemcpyDeviceToHost);
        double se = 0, sv = 0;
        for (std::uint32_t hh = 0; hh < 4; ++hh) {
          for (std::uint32_t e = 0; e < 32; ++e) {
            const std::uint32_t i = hh * 32u + e;
            const std::uint8_t* blk = kq.data() + (i / 32u) * 34u;
            std::int8_t code = static_cast<std::int8_t>(blk[i % 32u]);
            std::uint16_t scb =
                static_cast<std::uint16_t>(blk[32] | (blk[33] << 8));
            const float dq = static_cast<float>(code) * f16toF(scb);
            const float fv =
                kf[static_cast<std::size_t>(r) * kvdim + i];
            se += (dq - fv) * (dq - fv);
            sv += fv * fv;
          }
        }
        std::fprintf(stderr, " r%u=%.4f", r, se / (sv + 1e-9));
      }
      std::fprintf(stderr, "\n");
    }
    // Host-side attention audit for head 0: dequant K rows, score Q1/Q2.
    const std::uint32_t rb = impl.buffers.kv_row_bytes;
    const std::uint32_t nk = start_pos + n;
    std::vector<std::uint8_t> kraw(static_cast<std::size_t>(nk) * rb);
    (void)hipMemcpy(kraw.data(), impl.buffers.kv_bk[0],
                    kraw.size() * sizeof(std::uint8_t),
                    hipMemcpyDeviceToHost);
    std::vector<float> qq(static_cast<std::size_t>(n) * qdim);
    (void)hipMemcpy(qq.data(), impl.qbuf, qq.size() * sizeof(float),
                    hipMemcpyDeviceToHost);
    auto f16toF = [](std::uint16_t v) {
      std::uint32_t sign = (v >> 15) & 1u;
      std::uint32_t exp = (v >> 10) & 0x1Fu;
      std::uint32_t mant = v & 0x3FFu;
      std::uint32_t f;
      if (exp == 0)
        f = sign << 31;
      else if (exp == 31)
        f = (sign << 31) | 0x7F800000u;
      else
        f = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
      float out = 0.0F;
      std::memcpy(&out, &f, sizeof(float));
      return out;
    };
    auto krow = [&](std::uint32_t kr, std::uint32_t i) {
      // head 0, dim i: block ib=i/32, lane=i%32.
      const std::uint8_t* base = kraw.data() + static_cast<std::size_t>(kr) *
                                                     rb;
      const std::uint32_t ib = i / 32u;
      const std::uint8_t* blk = base + ib * 34u;
      std::int8_t code = static_cast<std::int8_t>(blk[i % 32u]);
      std::uint16_t scb = static_cast<std::uint16_t>(blk[32] | (blk[33] << 8));
      return static_cast<float>(code) * f16toF(scb);
    };
    for (std::uint32_t qr = 0; qr <= 2; ++qr) {
      const float* qv = qq.data() + static_cast<std::size_t>(qr) * qdim;
      float mx = -1e30F;
      std::vector<float> sc(nk);
      for (std::uint32_t kr = 0; kr < nk; ++kr) {
        float dot = 0.0F;
        for (std::uint32_t i = 0; i < d; ++i)
          dot += qv[i] * krow(kr, i);
        dot /= std::sqrt(static_cast<float>(d));
        sc[kr] = dot;
        mx = std::max(mx, dot);
      }
      float sum = 0.0F;
      for (auto& v : sc) {
        v = std::exp(v - mx);
        sum += v;
      }
      std::uint32_t b1 = 0, b2 = 0, b3 = 0;
      for (std::uint32_t kr = 0; kr < nk; ++kr) {
        const float w = sc[kr] / sum;
        if (w > sc[b1] / sum) {
          b3 = b2;
          b2 = b1;
          b1 = kr;
        } else if (w > sc[b2] / sum) {
          b3 = b2;
          b2 = kr;
        } else if (w > sc[b3] / sum) {
          b3 = kr;
        }
      }
      std::fprintf(stderr,
                   "[draft] attn-audit Q%u topkeys: %u:%.3f %u:%.3f %u:%.3f "
                   "maxlogit=%.2f\n",
                   qr, b1, sc[b1] / sum, b2, sc[b2] / sum, b3,
                   sc[b3] / sum, mx);
      if (qr == 1 && nk >= 11) {
        double n9 = 0, n10 = 0, dot910 = 0;
        for (std::uint32_t i = 0; i < d; ++i) {
          const float a = krow(9, i);
          const float b = krow(10, i);
          n9 += a * a;
          n10 += b * b;
          dot910 += a * b;
        }
        std::fprintf(stderr,
                     "[draft] key9/10 norms=%.3f/%.3f cos=%.3f\n",
                     std::sqrt(n9), std::sqrt(n10),
                     dot910 / (std::sqrt(n9 * n10) + 1e-9));
      }
      float best = -1e30F;
      std::uint32_t bestk = 0, besth = 0;
      for (std::uint32_t h = 0; h < 32; ++h) {
        const float* qh = qq.data() + static_cast<std::size_t>(qr) * qdim +
                          static_cast<std::size_t>(h) * d;
        for (std::uint32_t kr = 0; kr < nk; ++kr) {
          float dot = 0.0F;
          const std::uint32_t kvh = h / 4u;
          for (std::uint32_t i = 0; i < d; ++i) {
            const std::uint8_t* base =
                kraw.data() + static_cast<std::size_t>(kr) * rb +
                static_cast<std::size_t>(kvh) * 4u * 34u;
            const std::uint8_t* blk = base + (i / 32u) * 34u;
            std::int8_t code = static_cast<std::int8_t>(blk[i % 32u]);
            std::uint16_t scb =
                static_cast<std::uint16_t>(blk[32] | (blk[33] << 8));
            float kv = static_cast<float>(code) * f16toF(scb);
            dot += qh[i] * kv;
          }
          dot /= std::sqrt(static_cast<float>(d));
          if (dot > best) {
            best = dot;
            bestk = kr;
            besth = h;
          }
        }
      }
      std::fprintf(stderr, "[draft] Q%u bestmatch key=%u head=%u logit=%.2f\n",
                   qr, bestk, besth, best);
      if (qr == 1) {
        std::vector<float> ao(static_cast<std::size_t>(n) * qdim);
        (void)hipMemcpy(ao.data(), impl.attn_out,
                        ao.size() * sizeof(float), hipMemcpyDeviceToHost);
        std::vector<float> vv(static_cast<std::size_t>(nk) * 128, 0.0F);
        {
          std::vector<std::uint8_t> vraw(static_cast<std::size_t>(nk) *
                                         impl.buffers.kv_row_bytes);
          (void)hipMemcpy(vraw.data(), impl.buffers.kv_bv[0],
                          vraw.size() * sizeof(std::uint8_t),
                          hipMemcpyDeviceToHost);
          auto f16toF2 = [](std::uint16_t v) {
            std::uint32_t sign = (v >> 15) & 1u;
            std::uint32_t e2 = (v >> 10) & 0x1Fu;
            std::uint32_t mant = v & 0x3FFu;
            std::uint32_t f;
            if (e2 == 0)
              f = sign << 31;
            else if (e2 == 31)
              f = (sign << 31) | 0x7F800000u;
            else
              f = (sign << 31) | ((e2 + 112u) << 23) | (mant << 13);
            float out = 0.0F;
            std::memcpy(&out, &f, sizeof(float));
            return out;
          };
          for (std::uint32_t _kr = 0; _kr < nk; ++_kr) {
            for (std::uint32_t _i = 0; _i < 128; ++_i) {
              const std::uint8_t* blk =
                  vraw.data() + (size_t)_kr * impl.buffers.kv_row_bytes +
                  (_i / 32u) * 34u;
              std::int8_t code = (std::int8_t)blk[_i % 32u];
              std::uint16_t scb = (std::uint16_t)(blk[32] | (blk[33] << 8));
              vv[(size_t)_kr * 128 + _i] = (float)code * f16toF2(scb);
            }
          }
        }
        double _se = 0, _sv = 0;
        {
          const float* _q = qq.data() + (size_t)qr * qdim;
          std::vector<float> _sc(nk), _pw(nk);
          float _mx = -1e30f, _sum = 0;
          for (std::uint32_t _kr = 0; _kr < nk; ++_kr) {
            float _dot = 0;
            for (std::uint32_t _i = 0; _i < 128u; ++_i)
              _dot += _q[_i] * krow(_kr, _i);
            _dot /= std::sqrt(128.0f);
            _sc[_kr] = _dot;
            if (_dot > _mx)
              _mx = _dot;
          }
          for (std::uint32_t _kr = 0; _kr < nk; ++_kr) {
            _pw[_kr] = std::exp(_sc[_kr] - _mx);
            _sum += _pw[_kr];
          }
          for (std::uint32_t _i = 0; _i < 128u; ++_i) {
            float _o = 0;
            for (std::uint32_t _kr = 0; _kr < nk; ++_kr)
              _o += (_pw[_kr] / _sum) * vv[(size_t)_kr * 128 + _i];
            const float _dv = ao[(size_t)qr * qdim + _i];
            _se += (_o - _dv) * (_o - _dv);
            _sv += _dv * _dv;
          }
        }
        std::fprintf(stderr, "[draft] host-vs-dev attn_out relerr=%.6f\n",
                     _se / (_sv + 1e-9));
      }
      {
        const float* qrow =
            qq.data() + static_cast<std::size_t>(qr) * qdim;
        double mn = 1e30, mx = -1e30, s = 0;
        std::uint32_t nnan = 0;
        for (std::uint32_t i = 0; i < qdim; ++i) {
          const float v = qrow[i];
          if (!(v <= 1e30F && v >= -1e30F)) {
            ++nnan;
            continue;
          }
          if (v < mn)
            mn = v;
          if (v > mx)
            mx = v;
          s += v < 0 ? -v : v;
        }
        std::fprintf(stderr, "[draft] Q%u stats min=%.3f max=%.3f meanabs=%.4f nan=%u\n",
                     qr, mn, mx, s / qdim, nnan);
      }
      std::fprintf(stderr, "[draft] Q%u blkkeys:", qr);
      for (std::uint32_t kr = start_pos; kr < nk; ++kr) {
        float dot = 0.0F;
        for (std::uint32_t i = 0; i < d; ++i)
          dot += qv[i] * krow(kr, i);
        dot /= std::sqrt(static_cast<float>(d));
        std::fprintf(stderr, " k%u=%.2f", kr, dot);
      }
      std::fprintf(stderr, "\n");
    }
  }
  if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr &&
      (il == 0 || il + 1 == c.num_layers)) {
    (void)hipStreamSynchronize(stream);
    auto rowabs = [&](const float* d, std::uint32_t r,
                      std::uint32_t width) {
      std::vector<float> h(width);
      (void)hipMemcpy(h.data(), d + static_cast<std::size_t>(r) * width,
                      width * sizeof(float), hipMemcpyDeviceToHost);
      double s = 0;
      for (float x : h)
        s += x < 0 ? -x : x;
      return s / width;
    };
    std::fprintf(stderr,
                 "[draft] block L%u hidden/attn rows:", il);
    for (std::uint32_t r = 0; r < n; ++r)
      std::fprintf(stderr, " h%u=%.3f", r,
                   rowabs(hidden_buf, r, hidden));
    for (std::uint32_t r = 0; r < n; ++r)
      std::fprintf(stderr, " a%u=%.3f", r,
                   rowabs(impl.attn_out, r, qdim));
    std::fprintf(stderr, "\n");
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
  if (std::getenv("GLIMMER_ZERO_PREFIX") != nullptr) {
    (void)hipMemset(impl.fused, 0,
                    static_cast<std::size_t>(n) * hidden * sizeof(float));
  }
  kernels::RmsNorm(impl.fused,
                   static_cast<const float*>(dw.enc_norm.device),
                   impl.fused, n, hidden, dw.config.rms_eps, stream);
  (void)hipStreamSynchronize(stream);
  if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr) {
    std::vector<float> hf(static_cast<std::size_t>(n) * hidden);
    (void)hipMemcpy(hf.data(), impl.fused, hf.size() * sizeof(float),
                    hipMemcpyDeviceToHost);
    double s = 0;
    for (float v : hf)
      s += v < 0 ? -v : v;
    std::fprintf(stderr, "[draft] enc n=%u meanabs=%.5f head=", n,
                 hf.empty() ? 0.0 : s / hf.size());
    for (std::uint32_t i = 0; i < 5 && i < hf.size(); ++i)
      std::fprintf(stderr, " %.5f", hf[i]);
    std::fprintf(stderr, "\n");
  }
  return true;
}

// Reference embd-batch branch: the fused encoder output seeds the draft
// KV directly (K = qk-norm(wk(enc)) + RoPE @ target pos, V = wv(enc)
// with no norm and no RoPE). No queries, no attention, no FFN.
bool DraftSession::InjectFused(std::uint32_t n, std::uint32_t start_pos,
                               std::string* error_msg) {
  if (n == 0 || n > kDraftChunk) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft InjectFused needs 1..128 rows";
    return false;
  }
  auto& w = *model_->weights_;
  const auto& c = w.config;
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t d = c.head_dim;
  const std::uint32_t kvdim = c.num_kv_heads * d;
  auto& impl = *impl_;
  hipStream_t stream = 0;
  (void)hipMemcpyAsync(impl.d_pos, &start_pos, sizeof(start_pos),
                       hipMemcpyHostToDevice, stream);
  if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr) {
    (void)hipStreamSynchronize(stream);
    std::uint32_t pos_back = 0xFFFFFFFFu;
    (void)hipMemcpy(&pos_back, impl.d_pos, sizeof(pos_back),
                    hipMemcpyDeviceToHost);
    std::fprintf(stderr, "[draft] inject start_pos=%u d_pos=%u\n", start_pos,
                 pos_back);
  }
  namespace blocked = core::compute::blocked_kv;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const auto& l = w.layers[il];
    DraftGemm(l.wk, impl.fused, impl.kbuf, n, kvdim, hidden, stream);
    if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr && il == 0) {
      (void)hipStreamSynchronize(stream);
      std::vector<float> hk(static_cast<std::size_t>(n) * kvdim);
      std::vector<float> hv(static_cast<std::size_t>(n) * kvdim);
      DraftGemm(l.wv, impl.fused, impl.vbuf, n, kvdim, hidden, stream);
      (void)hipStreamSynchronize(stream);
      (void)hipMemcpy(hk.data(), impl.kbuf, hk.size() * sizeof(float),
                      hipMemcpyDeviceToHost);
      (void)hipMemcpy(hv.data(), impl.vbuf, hv.size() * sizeof(float),
                      hipMemcpyDeviceToHost);
      auto meanabs = [](const std::vector<float>& v) {
        double s = 0;
        for (float x : v)
          s += x < 0 ? -x : x;
        return v.empty() ? 0.0 : s / v.size();
      };
      std::fprintf(stderr,
                   "[draft] inject pre-norm K meanabs=%.4f V meanabs=%.4f\n",
                   meanabs(hk), meanabs(hv));
    }
    DraftGemm(l.wv, impl.fused, impl.vbuf, n, kvdim, hidden, stream);
    kernels::RmsNorm(impl.kbuf,
                     static_cast<const float*>(l.attn_k_norm.device),
                     impl.kbuf, n * c.num_kv_heads, d, c.rms_eps, stream);
    kernels::RopeNeox(impl.kbuf, n, c.num_kv_heads, d, start_pos, c.rope_theta,
                  stream);
    (void)blocked::QuantizeKvRows(
        impl.kbuf, kvdim, d, impl.buffers.kv_bk[il], kvdim,
        c.num_kv_heads, d, n, blocked::KvCodec::kQ8, 127.0F, 34,
        impl.d_pos, stream);
    (void)blocked::QuantizeKvRows(
        impl.vbuf, kvdim, d, impl.buffers.kv_bv[il], kvdim,
        c.num_kv_heads, d, n, blocked::KvCodec::kQ8, 127.0F, 34,
        impl.d_pos, stream);
    if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr && il == 0) {
      (void)hipStreamSynchronize(stream);
      std::vector<float> kf(static_cast<std::size_t>(n) * kvdim);
      {
        std::vector<float> _kn(static_cast<std::size_t>(n) * kvdim);
        (void)hipMemcpy(_kn.data(), impl.kbuf,
                        _kn.size() * sizeof(float), hipMemcpyDeviceToHost);
        std::fprintf(stderr, "[draft] k rownorms:");
        for (std::uint32_t _r = 0; _r < n; ++_r) {
          double _s = 0;
          for (std::uint32_t _i = 0; _i < kvdim; ++_i) {
            const float _v =
                _kn[static_cast<std::size_t>(_r) * kvdim + _i];
            _s += _v * _v;
          }
          std::fprintf(stderr, " %.3f", std::sqrt(_s / kvdim));
        }
        std::fprintf(stderr, "\n");
      }
      std::vector<std::uint8_t> kq(static_cast<std::size_t>(n) *
                                   impl.buffers.kv_row_bytes);
      auto f16toF = [](std::uint16_t v) {
        std::uint32_t sign = (v >> 15) & 1u;
        std::uint32_t e2 = (v >> 10) & 0x1Fu;
        std::uint32_t mant = v & 0x3FFu;
        std::uint32_t f;
        if (e2 == 0)
          f = sign << 31;
        else if (e2 == 31)
          f = (sign << 31) | 0x7F800000u;
        else
          f = (sign << 31) | ((e2 + 112u) << 23) | (mant << 13);
        float out = 0.0F;
        std::memcpy(&out, &f, sizeof(float));
        return out;
      };
      std::fprintf(stderr, "[draft] prefix q8err maxblk:");
      for (std::uint32_t r = 0; r < n; ++r) {
        (void)hipMemcpy(
            const_cast<float*>(kf.data()) + static_cast<std::size_t>(r) *
                                                kvdim,
            impl.kbuf + static_cast<std::size_t>(r) * kvdim,
            kvdim * sizeof(float), hipMemcpyDeviceToHost);
        (void)hipMemcpy(kq.data() + static_cast<std::size_t>(r) *
                                        impl.buffers.kv_row_bytes,
                        impl.buffers.kv_bk[0] +
                            static_cast<std::size_t>(start_pos + r) *
                                impl.buffers.kv_row_bytes,
                        impl.buffers.kv_row_bytes, hipMemcpyDeviceToHost);
        double worst = 0;
        std::uint32_t worstb = 0;
        for (std::uint32_t bb = 0; bb < 32; ++bb) {
          double se = 0, sv = 0;
          for (std::uint32_t e = 0; e < 32; ++e) {
            const std::uint32_t i = bb * 32u + e;
            const std::uint8_t* blk =
                kq.data() + static_cast<std::size_t>(r) *
                                impl.buffers.kv_row_bytes +
                (i / 32u) * 34u;
            std::int8_t code = static_cast<std::int8_t>(blk[i % 32u]);
            std::uint16_t scb =
                static_cast<std::uint16_t>(blk[32] | (blk[33] << 8));
            const float dq = static_cast<float>(code) * f16toF(scb);
            const float fv = kf[static_cast<std::size_t>(r) * kvdim + i];
            se += (dq - fv) * (dq - fv);
            sv += fv * fv;
          }
          const double rel = se / (sv + 1e-9);
          if (rel > worst) {
            worst = rel;
            worstb = bb;
          }
        }
        std::fprintf(stderr, " r%u=%.4f@%u", start_pos + r, worst, worstb);
      }
      std::fprintf(stderr, "\n");
    }
    (void)hipStreamSynchronize(stream);
  }
  impl.length = start_pos + n;
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
  for (std::uint32_t off = 0; off < n;) {
    const std::uint32_t len =
        std::min(kDraftChunk, n - off);
    if (!FuseRows(feats.subspan(static_cast<std::size_t>(off) *
                                    impl.feat_width,
                                static_cast<std::size_t>(len) *
                                    impl.feat_width),
                  len, error_msg))
      return false;
    if (!InjectFused(len, start_pos + off, error_msg))
      return false;
    off += len;
  }
  (void)hipStreamSynchronize(0);
  return true;
}

bool DraftSession::EmbedBlock(std::span<const std::int32_t> toks,
                              std::string* error_msg) {
  const std::uint32_t n = static_cast<std::uint32_t>(toks.size());
  if (n == 0 || n > kDraftChunk) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft EmbedBlock needs 1..128 tokens";
    return false;
  }
  auto& impl = *impl_;
  const auto& embed = model_->target_embed_;
  if (embed.device == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft target embedding is not bound";
    return false;
  }
  hipStream_t stream = 0;
  (void)hipMemcpyAsync(impl.d_tok, toks.data(),
                       static_cast<std::size_t>(n) * sizeof(std::int32_t),
                       hipMemcpyHostToDevice, stream);
  // Raw gather without the target's embd_norm: the reference decoder
  // embeds [id_last, mask...] straight from the shared token table.
  kernels::Embed(embed.device, embed.type, impl.d_tok, impl.hidden, n,
                 model_->weights_->config.hidden_size, stream);
  (void)hipStreamSynchronize(stream);
  return true;
}

bool DraftSession::ForwardBlock(std::uint32_t n, std::uint32_t start_pos,
                                std::string* error_msg) {
  if (n == 0 || n > kDraftChunk) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft ForwardBlock needs 1..128 rows";
    return false;
  }
  // Reference block layout (llama.cpp draft-dflash): the anchor overwrites
  // the last committed row, masks append after it. The block extends from
  // Length with one overlapping row: Length == start_pos + 1.
  if (start_pos + 1 != impl_->length) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft block must overlap the current length";
    return false;
  }
  if (start_pos + n > model_->options_.max_context) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft context exhausted";
    return false;
  }
  hipStream_t stream = 0;
  auto& w = *model_->weights_;
  for (std::uint32_t il = 0; il < w.config.num_layers; ++il) {
    ForwardBlockLayer(*this, il, n, start_pos, stream);
    (void)hipStreamSynchronize(stream);
    if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr && n >= 3) {
      for (std::uint32_t r = 1; r <= 2; ++r) {
        std::string err;
        if (!ScoreRow(r, &err))
          break;
        (void)hipStreamSynchronize(stream);
        const auto& hl = impl_->host_logits;
        std::uint32_t b1 = 0, b2 = 0;
        for (std::uint32_t t = 0; t < hl.size(); ++t) {
          if (hl[t] > hl[b1]) {
            b2 = b1;
            b1 = t;
          } else if (hl[t] > hl[b2]) {
            b2 = t;
          }
        }
        std::fprintf(stderr, "[draft] L%u row%u top2: %u:%.2f %u:%.2f\n",
                     il, r, b1, hl[b1], b2, hl[b2]);
      }
    }
  }
  impl_->length = start_pos + n;
  if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr) {
    (void)hipStreamSynchronize(0);
    const std::uint32_t rb = impl_->buffers.kv_row_bytes;
    std::vector<std::uint8_t> raw(rb);
    std::fprintf(stderr, "[draft] L0 K nonzero bytes:");
    const std::uint32_t rows[4] = {0, start_pos > 0 ? start_pos - 1 : 0,
                                   start_pos, start_pos + n - 1};
    for (int i = 0; i < 4; ++i) {
      const std::uint32_t r = rows[i];
      (void)hipMemcpy(raw.data(),
                      impl_->buffers.kv_bk[0] +
                          static_cast<std::size_t>(r) * rb,
                      rb, hipMemcpyDeviceToHost);
      std::uint32_t nz = 0;
      for (std::uint32_t b = 0; b < rb; ++b)
        nz += raw[b] != 0 ? 1 : 0;
      std::fprintf(stderr, " r%u=%u", r, nz);
    }
    std::fprintf(stderr, "\n");
  }
  (void)hipStreamSynchronize(0);
  return true;
}

bool DraftSession::ScoreRow(std::uint32_t i, std::string* error_msg) {
  auto& w = *model_->weights_;
  const auto& head = model_->target_head_;
  if (head.output.empty() || head.vocab_size == 0) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft target head is not bound";
    return false;
  }
  if (i >= kDraftChunk) {
    if (error_msg != nullptr)
      *error_msg = "glimmer draft ScoreRow is out of range";
    return false;
  }
  const std::uint32_t hidden = model_->weights_->config.hidden_size;
  auto& impl = *impl_;
  hipStream_t stream = 0;
  const float* row = impl.hidden + static_cast<std::size_t>(i) * hidden;
  kernels::RmsNorm(row,
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
  if (std::getenv("GLIMMER_DRAFT_DUMP") != nullptr) {
    (void)hipStreamSynchronize(stream);
    const auto& hl = impl.host_logits;
    std::uint32_t best[5] = {0, 0, 0, 0, 0};
    for (std::uint32_t t = 0; t < hl.size(); ++t) {
      for (int b = 0; b < 5; ++b) {
        if (hl[t] > hl[best[b]]) {
          for (int c = 4; c > b; --c)
            best[c] = best[c - 1];
          best[b] = t;
          break;
        }
      }
    }
    std::fprintf(stderr, "[draft] row %u top5:", i);
    for (int b = 0; b < 5; ++b)
      std::fprintf(stderr, " %u:%.3f", best[b], hl[best[b]]);
    std::fprintf(stderr, "\n");
  }
  return true;
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
