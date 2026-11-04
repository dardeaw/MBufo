// glimmer dense-decoder runtime: GGUF load, chunked prefill, single-token
// decode, greedy sampling. GEMM rides the shared core (GemvBatched decode,
// DenseF16Gemm prefill); attention is a local dense/SWA F16 kernel (F16 KV
// planes, or the shared blocked-Q8 grid with --kv-quant q8). The attention
// phase of every forward runs through the shared unified mixed-batch
// launcher first (single block for Forward, one block per session for
// DecodeBatch) and falls back to the legacy cores on refusal, so F16
// semantics never change. Sync is incremental on append-only growth.
#include "src/models/glimmer/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "src/core/compute/attention/attention_dispatcher.hpp"
#include "src/core/compute/attention/blocked_kv.hpp"
#include "src/core/compute/attention/blocked_splitk.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/speculative/draft_backend.hpp"
#include "src/models/glimmer/draft_backend.hpp"
#include "src/models/glimmer/kernels/kernels.hpp"

namespace gufo::models::glimmer {
namespace {

constexpr std::uint32_t kMaxChunk = 512;
/// Batched-verify row cap: speculative verification batches are short
/// (1 anchor + a handful of drafts); wider batches fall back to serial.
constexpr std::uint32_t kMaxVerify = 8;

/// Unified-batch opt-out valve (default on): GUFO_GLIMMER_UNIFIED_BATCH=0
/// keeps the legacy per-session attention cores everywhere (Forward and
/// DecodeBatch). Read per call so tests can switch modes mid-process.
[[nodiscard]] bool UnifiedBatchEnabled() {
  const char* e = std::getenv("GUFO_GLIMMER_UNIFIED_BATCH");
  return e == nullptr || e[0] != '0';
}

bool Check(hipError_t err, std::string* error_msg, const char* what) {
  if (err == hipSuccess)
    return true;
  if (error_msg != nullptr && error_msg->empty())
    *error_msg = std::string("glimmer HIP failure in ") + what;
  return false;
}

// Mirrors one weight tensor to device; norms stay F32, projections stay
// packed (the shared GEMM launchers dequantize on the fly).
bool MirrorTensor(TensorRef& t, std::vector<void*>& owned,
                  std::string* error_msg) {
  if (t.empty())
    return true;
  void* dev = nullptr;
  if (!Check(hipMalloc(&dev, t.size_bytes), error_msg, "weight mirror alloc"))
    return false;
  if (!Check(hipMemcpy(dev, t.data, t.size_bytes, hipMemcpyHostToDevice),
             error_msg, "weight mirror copy")) {
    (void)hipFree(dev);
    return false;
  }
  t.device = dev;
  owned.push_back(dev);
  return true;
}

bool UploadWeights(ModelWeights& w, std::vector<void*>& owned,
                   std::string* error_msg) {
  if (!MirrorTensor(w.token_embd, owned, error_msg))
    return false;
  if (!MirrorTensor(w.output_norm, owned, error_msg))
    return false;
  if (!MirrorTensor(w.output, owned, error_msg))
    return false;
  for (auto& l : w.layers) {
    if (!MirrorTensor(l.attn_norm, owned, error_msg))
      return false;
    if (!MirrorTensor(l.wq, owned, error_msg))
      return false;
    if (!MirrorTensor(l.wk, owned, error_msg))
      return false;
    if (!MirrorTensor(l.wv, owned, error_msg))
      return false;
    if (!MirrorTensor(l.attn_q_norm, owned, error_msg))
      return false;
    if (!MirrorTensor(l.attn_k_norm, owned, error_msg))
      return false;
    if (!MirrorTensor(l.gate, owned, error_msg))
      return false;
    if (!MirrorTensor(l.wo, owned, error_msg))
      return false;
    if (!MirrorTensor(l.attn_post_norm, owned, error_msg))
      return false;
    if (!MirrorTensor(l.ffn_norm, owned, error_msg))
      return false;
    if (!MirrorTensor(l.ffn_gate, owned, error_msg))
      return false;
    if (!MirrorTensor(l.ffn_up, owned, error_msg))
      return false;
    if (!MirrorTensor(l.ffn_down, owned, error_msg))
      return false;
    if (!MirrorTensor(l.ffn_post_norm, owned, error_msg))
      return false;
  }
  return true;
}

// One projection: narrow decode rides the shared prequant Q8_0 narrow GEMV
// (preq-q81, same core as the Qwen/gemma4 decode path); large-batch
// prefill quantizes activations once and rides the shared W8A8 core (WMMA
// int8, same kernels as the Qwen prefill path). Batches below 96 stay on the
// Dense F16 path so short prompts are bit-identical to the f16 reference;
// shapes the narrow/W8A8 launchers reject fall through to the legacy
// GemvBatched route (narrow misses keep yesterday's numerics bit-for-bit).
void GlimmerGemm(const TensorRef& w, const float* x, __half* x16,
                 void* q8tiled, void* preq_scratch, std::size_t preq_bytes,
                 float* y, std::uint32_t n_tokens,
                 std::uint32_t m, std::uint32_t k, hipStream_t stream) {
  const void* dev = w.device != nullptr ? w.device : w.data;
  if (n_tokens <= 8) {
    // preq-q81: Q8_0 narrow decode shapes run the shared quantize ->
    // narrow-GEMV on the session-arena scratch (§4.2); misses (non-Q8_0,
    // out-of-contract shapes, short scratch) keep GemvBatched bit-identical.
    if (w.type == core::GgmlType::kQ8_0 &&
        core::compute::gemm::DecideNarrowDecodePrequantRoute(
            /*weight_q8_0=*/true, /*gated=*/false, n_tokens, k,
            m) == core::compute::gemm::NarrowDecodeRoute::kNarrowQ80Prequant &&
        core::compute::gemm::LaunchNarrowQ80Prequant(
            dev, x, y, k, m, n_tokens, preq_scratch, preq_bytes,
            stream)) {
      return;
    }
    core::compute::gemm::GemvBatched(dev, static_cast<int>(w.type),
                                     w.RowBytes(), k, x, y, n_tokens, m, 0.0F,
                                     stream);
    return;
  }
  if (n_tokens >= 96 && q8tiled != nullptr &&
      w.type == core::GgmlType::kQ8_0) {
    // Host-side 16-split (dispatcher contract): mirror W8A8TailSplit so
    // only rows the scalar tail kernel would otherwise burn (~40x slower
    // per row, k <= 10240 only) divert to the Dense F16 path. Every other
    // batch rides one W8A8 launch whose edge clamping is exact, so common
    // shapes keep yesterday's numerics bit-for-bit.
    std::size_t main_n = n_tokens, tail_n = 0;
    const std::size_t tail128 = n_tokens % 128;
    if (n_tokens > 128 && tail128 != 0 && tail128 <= 16 &&
        k <= core::compute::gemm::kW8A8VarlenMaxK) {
      main_n = n_tokens - tail128;
      tail_n = tail128;
    }
    bool w8a8_ok = (main_n == 0);
    if (main_n != 0) {
      core::compute::gemm::QuantizeQ8Tiled(x, q8tiled, main_n, k, stream);
      w8a8_ok = core::compute::gemm::W8A8Gemm(dev, q8tiled, y, main_n, m, k,
                                              stream);
    }
    if (w8a8_ok && tail_n != 0) {
      const float* xt = x + static_cast<std::size_t>(main_n) * k;
      float* yt = y + static_cast<std::size_t>(main_n) * m;
      kernels::F32ToF16(xt, x16, tail_n * k, stream);
      core::compute::gemm::DenseF16Gemm(dev, x16, yt, tail_n, m, k, stream);
      return;
    }
    if (w8a8_ok) {
      return;
    }
  }
  kernels::F32ToF16(x, x16, static_cast<std::size_t>(n_tokens) * k, stream);
  if (w.type == core::GgmlType::kF16) {
    core::compute::gemm::UnquantizedF16Gemm(dev, x16, y, n_tokens, m, k,
                                           stream);
  } else {
    core::compute::gemm::DenseF16Gemm(dev, x16, y, n_tokens, m, k, stream);
  }
}

struct SessionBuffers {
  std::vector<void*> allocations;
  // KV caches, indexed by layer. F16 planes for the f16 path; shared
  // blocked-Q8 grids (with per-layer row bytes) for the q8 path.
  std::vector<__half*> kv_k;
  std::vector<__half*> kv_v;
  std::vector<std::uint8_t*> kv_bk;
  std::vector<std::uint8_t*> kv_bv;
  std::vector<std::uint32_t> kv_row_bytes_k;
  std::vector<std::uint32_t> kv_row_bytes_v;

  template <typename T>
  T* Alloc(std::size_t count, std::string* error) {
    void* p = nullptr;
    if (hipMalloc(&p, count * sizeof(T)) != hipSuccess || p == nullptr) {
      if (error != nullptr && error->empty())
        *error = "glimmer session allocation failed";
      return nullptr;
    }
    allocations.push_back(p);
    return static_cast<T*>(p);
  }

  ~SessionBuffers() {
    for (void* p : allocations) {
      (void)hipFree(p);
    }
  }
};

}  // namespace

struct Session::Impl {
  SessionBuffers buffers;
  float* hidden{nullptr};
  float* normed{nullptr};
  float* qbuf{nullptr};
  float* kbuf{nullptr};
  float* vbuf{nullptr};
  float* attn_out{nullptr};
  float* gate_out{nullptr};
  float* up_out{nullptr};
  float* gate_attn{nullptr};  ///< attention output gate activations [n, qdim]
  float* ffn_out{nullptr};
  float* logits{nullptr};
  float* verify_logits{nullptr};  ///< [kMaxVerify, vocab] batched verify rows
  std::int32_t* verify_out{nullptr};  ///< [kMaxVerify] device verify argmax
  __half* x16{nullptr};
  void* q8tiled{nullptr};
  void* preq_scratch{nullptr};  ///< session arena for preq-q81 narrow decode
  std::size_t preq_bytes{0};
  std::int32_t* d_tokens{nullptr};
  std::uint32_t* d_pos{nullptr};  ///< device position scalar (blocked path)
  std::uint32_t* d_pos_win{nullptr};  ///< rebased position (SWA shift)
  float* splitk_partials{nullptr};  ///< blocked Split-K partials (q8 path)
  core::compute::gemm::ArgmaxCandidate* argmax_scratch{nullptr};
  std::int32_t* argmax_out{nullptr};
  std::uint32_t capacity{0};
  std::uint32_t length{0};
  std::vector<float> host_logits;
  /// Previously synced prompt tokens: append-only growth forwards only the
  /// delta (O(n) prefix + delta instead of O(n^2) replays).
  std::vector<std::int32_t> synced_;
  /// Zero-based tapped layer outputs for draft feature capture (empty =
  /// taps disabled). Staging holds the last forwarded row per tap.
  std::vector<std::uint32_t> tap_layers_;
  float* tap_staging_{nullptr};

  [[nodiscard]] bool Ensure(const ModelWeights& w, std::uint32_t capacity,
                            bool use_blocked, std::string* error) {
    this->capacity = capacity;
    const std::uint32_t hidden_dim = w.config.hidden_size;
    const std::uint32_t qdim = w.config.num_heads * w.config.head_dim;
    const std::uint32_t kvdim = w.config.num_kv_heads * w.config.head_dim;
    const std::uint32_t ff = w.config.intermediate_size;
    // x16 stages F16 activations for every prefill GEMM, including the FFN
    // down-projection whose row width (intermediate) exceeds hidden. Size it
    // for the widest row: with hidden only, prompts past
    // chunk*hidden/intermediate tokens write past the buffer and corrupt the
    // session heap (GPU page fault on long prompts).
    const std::uint32_t wide_dim = std::max({hidden_dim, ff, qdim});
    const std::size_t chunk = kMaxChunk;
    hidden = buffers.Alloc<float>(chunk * hidden_dim, error);
    normed = buffers.Alloc<float>(chunk * hidden_dim, error);
    qbuf = buffers.Alloc<float>(chunk * qdim, error);
    kbuf = buffers.Alloc<float>(chunk * kvdim, error);
    vbuf = buffers.Alloc<float>(chunk * kvdim, error);
    attn_out = buffers.Alloc<float>(chunk * qdim, error);
    gate_out = buffers.Alloc<float>(chunk * ff, error);
    up_out = buffers.Alloc<float>(chunk * ff, error);
    gate_attn = buffers.Alloc<float>(chunk * qdim, error);
    ffn_out = buffers.Alloc<float>(chunk * hidden_dim, error);
    logits = buffers.Alloc<float>(w.config.vocab_size, error);
    verify_logits = buffers.Alloc<float>(
        static_cast<std::size_t>(kMaxVerify) * w.config.vocab_size, error);
    x16 = buffers.Alloc<__half>(chunk * wide_dim, error);
    q8tiled = buffers.Alloc<std::uint8_t>(
        core::compute::gemm::Q8TiledBytes(chunk, ff), error);
    // §4.2 prequant arena (no statics): 8-token ceiling over the widest
    // decode projection input row (ffn_down K == intermediate_size);
    // LaunchNarrowQ80Prequant only uses it, never grows it.
    preq_bytes = core::compute::gemm::NarrowQ80PrequantScratchBytes(wide_dim);
    preq_scratch = buffers.Alloc<std::uint8_t>(preq_bytes, error);
    d_tokens = buffers.Alloc<std::int32_t>(chunk, error);
    d_pos = buffers.Alloc<std::uint32_t>(1, error);
    d_pos_win = buffers.Alloc<std::uint32_t>(1, error);
    if (use_blocked) {
      splitk_partials = buffers.Alloc<float>(
          core::compute::blocked_splitk::
              AttentionBlockedSplitKScratchBytes() /
              sizeof(float),
          error);
    }
    argmax_scratch = buffers.Alloc<core::compute::gemm::ArgmaxCandidate>(
        core::compute::gemm::kArgmaxParts, error);
    argmax_out = buffers.Alloc<std::int32_t>(1, error);
    verify_out = buffers.Alloc<std::int32_t>(kMaxVerify, error);
    host_logits.assign(w.config.vocab_size, 0.0f);
    const bool ok = hidden != nullptr && normed != nullptr &&
           qbuf != nullptr && kbuf != nullptr && vbuf != nullptr &&
           attn_out != nullptr && gate_out != nullptr && up_out != nullptr &&
           gate_attn != nullptr && ffn_out != nullptr && logits != nullptr &&
           verify_logits != nullptr && x16 != nullptr && q8tiled != nullptr &&
           preq_scratch != nullptr && d_tokens != nullptr &&
           d_pos != nullptr && d_pos_win != nullptr &&
           argmax_scratch != nullptr && argmax_out != nullptr &&
           verify_out != nullptr;
    return ok && (!use_blocked || splitk_partials != nullptr);
  }
};

Model::~Model() {
  for (void* p : device_allocations_) {
    (void)hipFree(p);
  }
  if (mixed_scratch_ != nullptr) {
    (void)hipFree(mixed_scratch_);
    mixed_scratch_ = nullptr;
    mixed_scratch_bytes_ = 0;
  }
}

bool Model::EnsureMixedScratch(std::size_t need, std::string* error_msg) {
  if (mixed_scratch_ != nullptr && mixed_scratch_bytes_ >= need)
    return true;
  if (mixed_scratch_ != nullptr) {
    (void)hipFree(mixed_scratch_);
    mixed_scratch_ = nullptr;
    mixed_scratch_bytes_ = 0;
  }
  void* p = nullptr;
  if (hipMalloc(&p, need) != hipSuccess || p == nullptr) {
    if (error_msg != nullptr && error_msg->empty())
      *error_msg = "glimmer mixed-batch scratch allocation failed";
    return false;
  }
  mixed_scratch_ = p;
  mixed_scratch_bytes_ = need;
  return true;
}

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::string load_error;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
  if (reader_owner == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "Failed to open GGUF: " + load_error;
    return nullptr;
  }
  std::shared_ptr<core::GgufReader> reader(std::move(reader_owner));
  auto weights = ModelWeights::Bind(*reader, &load_error);
  if (!weights.has_value()) {
    if (error_msg != nullptr)
      *error_msg = "Failed to bind glimmer weights: " + load_error;
    return nullptr;
  }
  if (options.kv_cache_quant != "f16" && options.kv_cache_quant != "q8") {
    if (error_msg != nullptr)
      *error_msg = "glimmer kv_cache_quant must be f16 or q8";
    return nullptr;
  }
  if (weights->config.head_dim != 128) {
    if (error_msg != nullptr)
      *error_msg = "glimmer runtime supports head_dim 128 only";
    return nullptr;
  }
  auto model_owner = std::shared_ptr<Model>(new Model());
  model_owner->options_ = options;
  model_owner->reader_ = std::move(reader);
  model_owner->weights_ =
      std::make_unique<ModelWeights>(std::move(*weights));
  if (!UploadWeights(*model_owner->weights_, model_owner->device_allocations_,
                     &load_error)) {
    if (error_msg != nullptr)
      *error_msg = "Failed to mirror glimmer weights to device: " + load_error;
    return nullptr;
  }
  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*model_owner->reader_,
                                                  &load_error);
  if (tokenizer == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "Failed to create glimmer tokenizer: " + load_error;
    return nullptr;
  }
  model_owner->tokenizer_ = std::move(tokenizer);
  return model_owner;
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text,
                                           bool add_bos) const {
  tokenization::TokenizerOptions opts;
  opts.add_bos = add_bos;
  opts.parse_special_tokens = true;
  const auto ids = tokenizer_->Encode(text, opts);
  std::vector<std::int32_t> out(ids.begin(), ids.end());
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStopToken(static_cast<tokenization::TokenId>(token)) ||
         token == EosToken();
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::string Model::ModelName() const { return "glimmer"; }

std::size_t Model::ResidentBytes() const noexcept {
  std::size_t total = 0;
  auto add = [&](const TensorRef& t) { total += t.size_bytes; };
  add(weights_->token_embd);
  add(weights_->output_norm);
  add(weights_->output);
  for (const auto& l : weights_->layers) {
    add(l.attn_norm);
    add(l.wq);
    add(l.wk);
    add(l.wv);
    add(l.attn_q_norm);
    add(l.attn_k_norm);
    add(l.gate);
    add(l.wo);
    add(l.attn_post_norm);
    add(l.ffn_norm);
    add(l.ffn_gate);
    add(l.ffn_up);
    add(l.ffn_down);
    add(l.ffn_post_norm);
  }
  return total;
}

Session::Session(std::shared_ptr<Model> model)
    : model_(std::move(model)), impl_(std::make_unique<Impl>()) {}

Session::~Session() = default;

std::unique_ptr<Session> Model::CreateSession(std::string* error_msg) {
  const bool use_blocked = options_.kv_cache_quant == "q8";
  auto session = std::unique_ptr<Session>(new Session(shared_from_this()));
  std::string error;
  if (!session->impl_->Ensure(*weights_, options_.max_context, use_blocked,
                              &error)) {
    if (error_msg != nullptr)
      *error_msg = error;
    return nullptr;
  }
  const auto& w = *weights_;
  auto& impl = *session->impl_;
  impl.buffers.kv_k.assign(w.config.num_layers, nullptr);
  impl.buffers.kv_v.assign(w.config.num_layers, nullptr);
  impl.buffers.kv_bk.assign(w.config.num_layers, nullptr);
  impl.buffers.kv_bv.assign(w.config.num_layers, nullptr);
  impl.buffers.kv_row_bytes_k.assign(w.config.num_layers, 0);
  impl.buffers.kv_row_bytes_v.assign(w.config.num_layers, 0);
  const std::uint32_t kvdim =
      w.config.num_kv_heads * w.config.head_dim;
  // Shared blocked grid: (d/32) blocks of 34 bytes per KV head, packed per
  // row (llama q8 convention: one row_bytes stride for every layer).
  const std::uint32_t row_bytes =
      (w.config.head_dim / 32) * 34 * w.config.num_kv_heads;
  for (std::uint32_t il = 0; il < w.config.num_layers; ++il) {
    if (use_blocked) {
      // Post-RoPE rows quantize straight into the shared blocked layout
      // (per-32-block f16 scale, 34 bytes per head-block for Q8).
      const std::size_t bytes =
          static_cast<std::size_t>(options_.max_context) * row_bytes;
      auto* kb = impl.buffers.Alloc<std::uint8_t>(bytes, &error);
      auto* vb = impl.buffers.Alloc<std::uint8_t>(bytes, &error);
      if (kb == nullptr || vb == nullptr) {
        if (error_msg != nullptr)
          *error_msg = "blocked KV cache allocation failed (layer " +
                       std::to_string(il) + ")";
        return nullptr;
      }
      (void)hipMemset(kb, 0, bytes);
      (void)hipMemset(vb, 0, bytes);
      impl.buffers.kv_bk[il] = kb;
      impl.buffers.kv_bv[il] = vb;
      impl.buffers.kv_row_bytes_k[il] = row_bytes;
      impl.buffers.kv_row_bytes_v[il] = row_bytes;
      continue;
    }
    const std::size_t elems =
        static_cast<std::size_t>(options_.max_context) * kvdim;
    auto* kk = impl.buffers.Alloc<__half>(elems, &error);
    auto* vv = impl.buffers.Alloc<__half>(elems, &error);
    if (kk == nullptr || vv == nullptr) {
      if (error_msg != nullptr)
        *error_msg = "KV cache allocation failed (layer " +
                     std::to_string(il) + ")";
      return nullptr;
    }
    (void)hipMemset(kk, 0, elems * sizeof(__half));
    (void)hipMemset(vv, 0, elems * sizeof(__half));
    impl.buffers.kv_k[il] = kk;
    impl.buffers.kv_v[il] = vv;
  }
  return session;
}

void Session::ForwardLayerPre(Session& s, std::uint32_t il, std::uint32_t n,
                              std::uint32_t start_pos, hipStream_t stream) {
  auto& w = *s.model_->weights_;
  const auto& c = w.config;
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t d = c.head_dim;
  const std::uint32_t qdim = c.num_heads * d;
  const std::uint32_t kvdim = c.num_kv_heads * d;
  const auto& l = w.layers[il];
  auto& impl = *s.impl_;
  const bool swa = c.LayerIsSwa(il);
  const bool use_blocked = s.model_->options_.kv_cache_quant == "q8";

  float* hidden_buf = impl.hidden;
  // Pre-attention norm.
  kernels::RmsNorm(hidden_buf,
                   static_cast<const float*>(l.attn_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  GlimmerGemm(l.wq, impl.normed, impl.x16, impl.q8tiled, impl.preq_scratch, impl.preq_bytes, impl.qbuf, n, qdim,
              hidden, stream);
  GlimmerGemm(l.wk, impl.normed, impl.x16, impl.q8tiled, impl.preq_scratch, impl.preq_bytes, impl.kbuf, n, kvdim,
              hidden, stream);
  GlimmerGemm(l.wv, impl.normed, impl.x16, impl.q8tiled, impl.preq_scratch, impl.preq_bytes, impl.vbuf, n, kvdim,
              hidden, stream);
  // QK-norm (per-head RMS over d; absorbs the qk_scale_factor).
  kernels::RmsNorm(impl.qbuf,
                   static_cast<const float*>(l.attn_q_norm.device),
                   impl.qbuf, n * c.num_heads, d, c.rms_eps, stream);
  kernels::RmsNorm(impl.kbuf,
                   static_cast<const float*>(l.attn_k_norm.device),
                   impl.kbuf, n * c.num_kv_heads, d, c.rms_eps, stream);
  // RoPE on SWA layers only (dense layers skip it, per the reference).
  if (swa) {
    kernels::Rope(impl.qbuf, n, c.num_heads, d, start_pos, c.rope_theta_swa,
                  stream);
    kernels::Rope(impl.kbuf, n, c.num_kv_heads, d, start_pos,
                  c.rope_theta_swa, stream);
  }
  if (use_blocked) {
    // Post-RoPE rows quantize straight into the shared blocked layout
    // (per-32-block f16 scale, 34 bytes per head-block for Q8), exactly
    // like the llama q8 path; d_pos (uploaded by the caller) selects the
    // destination rows [pos, pos+n).
    namespace blocked = core::compute::blocked_kv;
    (void)blocked::QuantizeKvRows(
        impl.kbuf, kvdim, d, impl.buffers.kv_bk[il], kvdim, c.num_kv_heads,
        d, n, blocked::KvCodec::kQ8, 127.0F, 34, impl.d_pos, stream);
    (void)blocked::QuantizeKvRows(
        impl.vbuf, kvdim, d, impl.buffers.kv_bv[il], kvdim, c.num_kv_heads,
        d, n, blocked::KvCodec::kQ8, 127.0F, 34, impl.d_pos, stream);
    return;
  }
  // F16 KV store (v1 dense planes).
  kernels::StoreKvF16(impl.kbuf, impl.buffers.kv_k[il], n, kvdim,
                      start_pos, stream);
  kernels::StoreKvF16(impl.vbuf, impl.buffers.kv_v[il], n, kvdim,
                      start_pos, stream);
}

void Session::SwaShiftedKV(Session& s, std::uint32_t il, std::uint32_t window,
                           std::uint32_t start_pos, hipStream_t stream,
                           const std::uint8_t*& k2, const std::uint8_t*& v2,
                           const std::uint32_t*& sp, std::uint32_t& rebased) {
  (void)stream;
  k2 = s.impl_->buffers.kv_bk[il];
  v2 = s.impl_->buffers.kv_bv[il];
  sp = s.impl_->d_pos;
  rebased = start_pos;
  if (window != 0 && start_pos + 1 > window) {
    const std::uint32_t lo = (start_pos + 1 - window) & ~3u;
    if (lo != 0 && lo <= start_pos) {
      rebased = start_pos - lo;
      if (hipMemcpy(s.impl_->d_pos_win, &rebased, sizeof(rebased),
                    hipMemcpyHostToDevice) == hipSuccess) {
        k2 += static_cast<std::size_t>(lo) *
              s.impl_->buffers.kv_row_bytes_k[il];
        v2 += static_cast<std::size_t>(lo) *
              s.impl_->buffers.kv_row_bytes_v[il];
        sp = s.impl_->d_pos_win;
      }
    }
  }
}

void Session::ForwardLayerAttn(Session& s, std::uint32_t il, std::uint32_t n,
                               std::uint32_t start_pos, hipStream_t stream,
                               bool try_unified) {
  auto& w = *s.model_->weights_;
  const auto& c = w.config;
  const std::uint32_t d = c.head_dim;
  auto& impl = *s.impl_;
  const bool swa = c.LayerIsSwa(il);
  const float score_scale = 1.0F / std::sqrt(static_cast<float>(d));
  const std::uint32_t window = swa ? c.sliding_window : 0;
  const bool use_blocked = s.model_->options_.kv_cache_quant == "q8";

  if (use_blocked) {
    // Blocked-Q8 read (llama convention): SWA layers pre-shift past the
    // window and rebase to dense-causal; the shared dispatcher picks
    // Split-K decode for long contexts and the scalar direct read
    // otherwise (same math, fused dequant).
    const std::uint8_t* k2 = impl.buffers.kv_bk[il];
    const std::uint8_t* v2 = impl.buffers.kv_bv[il];
    const std::uint32_t* sp = impl.d_pos;
    std::uint32_t base = start_pos;
    if (window != 0) {
      SwaShiftedKV(s, il, window, start_pos, stream, k2, v2, sp, base);
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
    if (route == attn_disp::AttentionRoute::kBlockedSplitK &&
        impl.splitk_partials != nullptr) {
      ran = core::compute::blocked_splitk::AttentionBlockedSplitK<
          core::compute::blocked_splitk::KvCodecQ8, 128>(
          impl.qbuf, k2, v2, impl.attn_out, n, sp, base + n, c.num_heads,
          c.num_kv_heads, d, stream, score_scale, 34, 34,
          impl.splitk_partials,
          core::compute::blocked_splitk::
              AttentionBlockedSplitKScratchBytes());
    }
    if (!ran) {
      core::compute::blocked_kv::AttentionKv(
          impl.qbuf, k2, v2, /*mask=*/nullptr, /*mask_words=*/0,
          impl.attn_out, /*partials=*/nullptr, /*splits=*/1, n, sp,
          c.num_heads, c.num_kv_heads, d, /*ratio=*/1, 34, 34,
          core::compute::blocked_kv::KvCodec::kQ8, stream, score_scale);
    }
    return;
  }

  // F16 path: one unified mixed-batch launch (single block) first; refusal
  // falls back to the legacy cores below with F16 semantics unchanged.
  bool attn_unified_ran = false;
  if (try_unified) {
    namespace mb = core::compute::mixed_batch;
    mb::MixedBlk blk{};
    blk.q = impl.qbuf;
    blk.k = impl.buffers.kv_k[il];
    blk.v = impl.buffers.kv_v[il];
    blk.gate = nullptr;  // the output gate stays a post-hoc SigmoidMul
    blk.out = impl.attn_out;
    blk.kv_len = start_pos + n;
    blk.q_pos_base = start_pos;
    blk.n_rows = n;
    blk.k_codec = mb::MixedKvCodec::kF16;
    blk.v_codec = mb::MixedKvCodec::kF16;
    blk.head_dim = d;
    blk.heads = c.num_heads;
    blk.kv_heads = c.num_kv_heads;
    blk.score_scale = score_scale;
    blk.q_stride = 0;
    blk.k_stride = 0;
    blk.v_stride = 0;
    blk.k_head_stride = 0;
    blk.v_head_stride = 0;
    blk.gate_stride = 0;
    blk.out_stride = 0;
    blk.window = window;
    blk.key_block_bytes = 34;
    blk.value_block_bytes = 34;
    std::string scratch_error;
    if (s.model_->EnsureMixedScratch(mb::MixedBatchF16ScratchBytes(1),
                                     &scratch_error)) {
      attn_unified_ran = mb::LaunchMixedBatchAttention(
          &blk, 1, s.model_->mixed_scratch_, s.model_->mixed_scratch_bytes_,
          stream);
    }
  }
  if (!attn_unified_ran) {
    // Prefill attention consults the shared dispatcher and stays on the F16
    // legacy route; the multi-query blocking kernel below is that route's
    // accelerator (same online-softmax math, F16 semantics unchanged).
    // `false` = unsupported shape (tiny/decode/non-d128) -> legacy runs.
    bool attn_mqb_ran = false;
    {
      namespace attn_disp = core::compute::attention_dispatcher;
      attn_disp::AttentionDispatchInput din;
      din.q_len = n;
      din.kv_len = start_pos + n;
      din.head_dim = d;
      din.heads = c.num_heads;
      din.kv_heads = c.num_kv_heads;
      din.k_codec = attn_disp::DispatchCodec::kF16;
      din.v_codec = attn_disp::DispatchCodec::kF16;
      din.window = window;
      din.is_verify = false;
      const auto attn_policy =
          attn_disp::AttentionDispatchPolicy::WithEnvDefaults();
      if (attn_policy.dispatch_enabled &&
          attn_disp::RouteAttention(din, attn_policy) ==
              attn_disp::AttentionRoute::kLegacy) {
        attn_mqb_ran = kernels::AttentionF16MQB(
            impl.qbuf, impl.buffers.kv_k[il], impl.buffers.kv_v[il],
            impl.attn_out, n, start_pos, c.num_heads, c.num_kv_heads, d,
            stream, score_scale, window);
      }
    }
    if (!attn_mqb_ran) {
      kernels::AttentionF16(impl.qbuf, impl.buffers.kv_k[il],
                            impl.buffers.kv_v[il], impl.attn_out, n, start_pos,
                            c.num_heads, c.num_kv_heads, d, stream,
                            score_scale, window);
    }
  }
}

void Session::ForwardLayerPost(Session& s, std::uint32_t il, std::uint32_t n,
                               std::uint32_t start_pos, hipStream_t stream,
                               TapSink* tap) {
  auto& w = *s.model_->weights_;
  const auto& c = w.config;
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t d = c.head_dim;
  const std::uint32_t qdim = c.num_heads * d;
  const auto& l = w.layers[il];
  auto& impl = *s.impl_;

  float* hidden_buf = impl.hidden;
  // Attention output gate: attn_out *= sigmoid(gate_proj(pre-attn hidden)).
  GlimmerGemm(l.gate, impl.normed, impl.x16, impl.q8tiled, impl.preq_scratch, impl.preq_bytes, impl.gate_attn, n, qdim,
              hidden, stream);
  kernels::SigmoidMul(impl.gate_attn, impl.attn_out, impl.attn_out,
                      static_cast<std::size_t>(n) * qdim, stream);
  // o_proj, post-attention norm (eps 1e-8), residual add.
  GlimmerGemm(l.wo, impl.attn_out, impl.x16, impl.q8tiled, impl.preq_scratch, impl.preq_bytes, impl.ffn_out, n, hidden,
              qdim, stream);
  kernels::RmsNormAdd(impl.ffn_out,
                      static_cast<const float*>(l.attn_post_norm.device),
                      hidden_buf, n, hidden, c.post_norm_eps, stream);

  // FFN block (SwiGLU).
  kernels::RmsNorm(hidden_buf,
                   static_cast<const float*>(l.ffn_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  GlimmerGemm(l.ffn_gate, impl.normed, impl.x16, impl.q8tiled, impl.preq_scratch, impl.preq_bytes, impl.gate_out, n,
            c.intermediate_size, hidden, stream);
  GlimmerGemm(l.ffn_up, impl.normed, impl.x16, impl.q8tiled, impl.preq_scratch, impl.preq_bytes, impl.up_out, n,
            c.intermediate_size, hidden, stream);
  kernels::SiluMul(impl.gate_out, impl.up_out, impl.gate_out,
                   static_cast<std::size_t>(n) * c.intermediate_size,
                   stream);
  GlimmerGemm(l.ffn_down, impl.gate_out, impl.x16, impl.q8tiled, impl.preq_scratch, impl.preq_bytes, impl.ffn_out, n,
            hidden, c.intermediate_size, stream);
  if (getenv("GLIMMER_DUMP_QKV") != nullptr && n <= 32 && il < 2) {
    (void)hipStreamSynchronize(stream);
    auto dump3 = [&](const float* dev, std::size_t cnt, const char* name) {
      std::vector<float> h(cnt);
      (void)hipMemcpy(h.data(), dev, cnt * sizeof(float),
                      hipMemcpyDeviceToHost);
      std::string path = std::string("/tmp/GLIMMER_qkv_L") +
                         std::to_string(il) + "_" + name + "_" +
                         std::to_string(start_pos) + ".bin";
      FILE* f = std::fopen(path.c_str(), "wb");
      if (f != nullptr) {
        (void)std::fwrite(h.data(), sizeof(float), cnt, f);
        (void)std::fclose(f);
      }
    };
    dump3(impl.gate_out, static_cast<std::size_t>(n) * c.intermediate_size,
          "act");
    dump3(impl.ffn_out, static_cast<std::size_t>(n) * hidden, "down_out");
    dump3(impl.normed, static_cast<std::size_t>(n) * hidden, "ffn_in");
  }
  // Post-FFN norm (eps 1e-8) + residual add.
  kernels::RmsNormAdd(impl.ffn_out,
                      static_cast<const float*>(l.ffn_post_norm.device),
                      hidden_buf, n, hidden, c.post_norm_eps, stream);
  // Draft feature taps: refresh the staged last row for tapped layers and
  // fire the one-shot sink (prompt priming / verify capture). Both see the
  // finished residual stream of this layer.
  if (!impl.tap_layers_.empty() || tap != nullptr) {
    const float* last_row =
        hidden_buf + static_cast<std::size_t>(n - 1) * hidden;
    for (std::size_t t = 0; t < impl.tap_layers_.size(); ++t) {
      if (impl.tap_layers_[t] == il && impl.tap_staging_ != nullptr) {
        (void)hipMemcpyAsync(
            impl.tap_staging_ + static_cast<std::size_t>(t) * hidden,
            last_row, hidden * sizeof(float), hipMemcpyDeviceToDevice,
            stream);
      }
    }
    if (tap != nullptr) {
      for (const std::uint32_t tapped : tap->layers) {
        if (tapped == il && tap->fn) {
          tap->fn(il, hidden_buf, n, start_pos);
        }
      }
    }
  }
  if (getenv("GLIMMER_DUMP_QKV") != nullptr && n <= 32) {
    (void)hipStreamSynchronize(stream);
    std::vector<float> h(static_cast<std::size_t>(n) * hidden);
    (void)hipMemcpy(h.data(), hidden_buf, h.size() * sizeof(float),
                    hipMemcpyDeviceToHost);
    std::string path = std::string("/tmp/GLIMMER_qkv_L") +
                       std::to_string(il) + "_full_" +
                       std::to_string(start_pos) + ".bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f != nullptr) {
      (void)std::fwrite(h.data(), sizeof(float), h.size(), f);
      (void)std::fclose(f);
    }
  }
}

void Session::ForwardFinish(Session& s, std::uint32_t n,
                            std::uint32_t start_pos, hipStream_t stream,
                            bool all_logits) {
  auto& w = *s.model_->weights_;
  const auto& c = w.config;
  const std::uint32_t hidden = c.hidden_size;
  auto& impl = *s.impl_;
  kernels::RmsNorm(impl.hidden,
                   static_cast<const float*>(w.output_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  if (all_logits && n <= kMaxVerify) {
    // Batched verify: score every row (n <= kMaxVerify, dense staging).
    for (std::uint32_t r = 0; r < n; ++r) {
      const float* row_in = impl.normed + static_cast<std::size_t>(r) * hidden;
      float* row_out =
          impl.verify_logits + static_cast<std::size_t>(r) * c.vocab_size;
      GlimmerGemm(w.output, row_in, impl.x16, impl.q8tiled, impl.preq_scratch, impl.preq_bytes, row_out, 1, c.vocab_size,
                  hidden, stream);
      if (c.logit_scale != 1.0F) {
        kernels::ScaleInplace(row_out, c.logit_scale,
                              static_cast<std::size_t>(1) * c.vocab_size,
                              stream);
      }
      kernels::SoftCap(row_out, c.final_logit_softcapping,
                       static_cast<std::size_t>(1) * c.vocab_size, stream);
    }
  } else {
    const float* last_row =
        impl.normed + static_cast<std::size_t>(n - 1) * hidden;
    GlimmerGemm(w.output, last_row, impl.x16, impl.q8tiled, impl.preq_scratch, impl.preq_bytes, impl.logits, 1,
                c.vocab_size, hidden, stream);
    // Output gain then tanh softcap (matches the reference order).
    if (c.logit_scale != 1.0F) {
      kernels::ScaleInplace(impl.logits, c.logit_scale,
                            static_cast<std::size_t>(1) * c.vocab_size,
                            stream);
    }
    kernels::SoftCap(impl.logits, c.final_logit_softcapping,
                     static_cast<std::size_t>(1) * c.vocab_size, stream);
  }
  if (getenv("GLIMMER_DUMP_QKV") != nullptr && n <= 32) {
    (void)hipStreamSynchronize(stream);
    auto dump = [&](const float* dev, std::size_t cnt, const char* name) {
      std::vector<float> h(cnt);
      (void)hipMemcpy(h.data(), dev, cnt * sizeof(float),
                      hipMemcpyDeviceToHost);
      std::string path = std::string("/tmp/GLIMMER_") + name + "_" +
                         std::to_string(start_pos) + ".bin";
      FILE* f = std::fopen(path.c_str(), "wb");
      if (f != nullptr) {
        (void)std::fwrite(h.data(), sizeof(float), cnt, f);
        (void)std::fclose(f);
      }
    };
    dump(impl.normed + static_cast<std::size_t>(n - 1) * hidden, hidden,
         "finalhidden");
    dump(impl.logits, c.vocab_size, "logits");
  }
  (void)hipMemcpyAsync(impl.host_logits.data(), impl.logits,
                       impl.host_logits.size() * sizeof(float),
                       hipMemcpyDeviceToHost, stream);
  impl.length = start_pos + n;
}

bool Session::Forward(std::span<const std::int32_t> tokens,
                      std::uint32_t start_pos, std::string* error_msg,
                      bool all_logits, TapSink* tap) {
  const auto& w = *model_->weights_;
  const std::uint32_t hidden = w.config.hidden_size;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > kMaxChunk) {
    if (error_msg != nullptr)
      *error_msg = "glimmer Forward needs 1..512 tokens";
    return false;
  }
  if (start_pos + n > model_->options_.max_context) {
    if (error_msg != nullptr)
      *error_msg = "glimmer context exhausted";
    return false;
  }
  hipStream_t stream = 0;
  auto& impl = *impl_;
  if (model_->options_.kv_cache_quant == "q8") {
    // Blocked-path destination rows [pos, pos+n) for every layer below.
    (void)hipMemcpyAsync(impl.d_pos, &start_pos, sizeof(start_pos),
                         hipMemcpyHostToDevice, stream);
  }
  std::vector<std::int32_t> host_tokens(tokens.begin(), tokens.end());
  if (host_tokens.size() > kMaxChunk) {
    if (error_msg != nullptr)
      *error_msg = "glimmer chunk exceeds device token staging";
    return false;
  }
  (void)hipMemcpyAsync(impl.d_tokens, host_tokens.data(),
                       host_tokens.size() * sizeof(std::int32_t),
                       hipMemcpyHostToDevice, stream);
  if (getenv("GLIMMER_DEBUG_SUMS") != nullptr && n <= 32) {
    (void)hipMemset(impl.hidden, 0,
                    static_cast<std::size_t>(n) * hidden * sizeof(float));
    std::vector<std::int32_t> ids_back(n, -999);
    (void)hipMemcpy(ids_back.data(), impl.d_tokens, n * sizeof(std::int32_t),
                    hipMemcpyDeviceToHost);
    std::fprintf(stderr, "[glimmer] d_tokens_back:");
    for (std::uint32_t ti = 0; ti < n && ti < 20; ++ti)
      std::fprintf(stderr, " %d", ids_back[ti]);
    std::fprintf(stderr, "\n");
  }
  kernels::Embed(w.token_embd.device, w.token_embd.type,
                 impl.d_tokens, impl.hidden, n, hidden,
                 stream);
  // Embedding RMSNorm (no gamma), mirroring the reference embd_norm.
  kernels::RmsNormPlain(impl.hidden, impl.hidden, n, hidden,
                        w.config.rms_eps,
                        stream);
  if (getenv("GLIMMER_DEBUG_SUMS") != nullptr && n <= 32) {
    (void)hipStreamSynchronize(stream);
    std::fprintf(stderr, "[glimmer] embd type=%d bytes=%zu ids:",
                 static_cast<int>(w.token_embd.type),
                 w.token_embd.size_bytes);
    for (std::uint32_t ti = 0; ti < n && ti < 20; ++ti)
      std::fprintf(stderr, " %d", host_tokens[ti]);
    std::fprintf(stderr, "\n");
    auto sum_of = [&](const float* d, std::size_t cnt, const char* tag) {
      std::vector<float> h(cnt);
      (void)hipMemcpy(h.data(), d, cnt * sizeof(float),
                      hipMemcpyDeviceToHost);
      double s = 0.0;
      float mx = 0.0f;
      for (float v : h) {
        s += (double)v;
        float a = v < 0 ? -v : v;
        if (!(a <= mx))
          mx = a;  // NaN-safe max
      }
      std::fprintf(stderr, "[glimmer] %s sum=%f maxabs=%f\n", tag, s, mx);
    };
    sum_of(impl.hidden, static_cast<std::size_t>(n) * hidden, "embed");
  }
  for (std::uint32_t il = 0; il < w.config.num_layers; ++il) {
    ForwardLayerPre(*this, il, n, start_pos, stream);
    ForwardLayerAttn(*this, il, n, start_pos, stream,
                     /*try_unified=*/UnifiedBatchEnabled());
    ForwardLayerPost(*this, il, n, start_pos, stream, tap);
    if (getenv("GLIMMER_DEBUG_SUMS") != nullptr && n <= 32 &&
        (il < 2 || il % 4 == 0 || il + 1 == w.config.num_layers)) {
      (void)hipStreamSynchronize(stream);
      auto sum_of = [&](const float* d, std::size_t cnt, const char* tag,
                        std::uint32_t layer) {
        std::vector<float> h(cnt);
        (void)hipMemcpy(h.data(), d, cnt * sizeof(float),
                        hipMemcpyDeviceToHost);
        double s = 0.0;
        float mx = 0.0f;
        for (float v : h) {
          s += (double)v;
          float a = v < 0 ? -v : v;
          if (!(a <= mx))
            mx = a;
        }
        std::fprintf(stderr, "[glimmer] L%u %s sum=%f maxabs=%f\n", layer, tag,
                     s, mx);
      };
      const std::uint32_t qdim_l =
          w.config.num_heads * w.config.head_dim;
      sum_of(impl.qbuf, static_cast<std::size_t>(n) * qdim_l, "qbuf", il);
      sum_of(impl.attn_out, static_cast<std::size_t>(n) * qdim_l, "attn", il);
      sum_of(impl.hidden, static_cast<std::size_t>(n) * hidden, "post", il);
    }
  }
  ForwardFinish(*this, n, start_pos, stream, all_logits);
  return true;
}

bool Session::DecodeBatch(std::span<BatchItem> items,
                          std::string* error_msg) {
  namespace mb = core::compute::mixed_batch;
  const auto fail = [&](const std::string& m) {
    if (error_msg != nullptr && error_msg->empty())
      *error_msg = m;
    return false;
  };
  if (items.empty() || items.size() > mb::kMaxMixedBlks)
    return fail("glimmer DecodeBatch needs 1..32 batch items");
  Model* model =
      (items[0].session != nullptr) ? items[0].session->model_.get() : nullptr;
  if (model == nullptr)
    return fail("glimmer DecodeBatch has a null session");
  const auto& w = *model->weights_;
  const auto& c = w.config;
  const bool use_blocked = model->options_.kv_cache_quant == "q8";
  hipStream_t stream = 0;
  std::vector<std::uint32_t> rows;
  rows.reserve(items.size());
  for (const auto& item : items) {
    if (item.session == nullptr || item.session->model_.get() != model)
      return fail("glimmer DecodeBatch sessions must share one model");
    const std::uint32_t n = static_cast<std::uint32_t>(item.tokens.size());
    if (n == 0 || n > kMaxChunk)
      return fail("glimmer DecodeBatch item needs 1..512 tokens");
    if (item.session->impl_->length + n > item.session->impl_->capacity)
      return fail("glimmer DecodeBatch token exceeds KV capacity");
    rows.push_back(n);
  }
  // Stage one token span per session; the blocked path additionally stages
  // the per-session decode position each layer's QuantizeKvRows writes to.
  for (std::size_t i = 0; i < items.size(); ++i) {
    auto& impl = *items[i].session->impl_;
    std::vector<std::int32_t> host_tokens(items[i].tokens.begin(),
                                          items[i].tokens.end());
    (void)hipMemcpyAsync(impl.d_tokens, host_tokens.data(),
                         host_tokens.size() * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, stream);
    if (use_blocked) {
      const std::uint32_t pos = impl.length;
      (void)hipMemcpyAsync(impl.d_pos, &pos, sizeof(pos),
                           hipMemcpyHostToDevice, stream);
    }
  }
  // Embed + embd_norm prologue per session (mirrors Forward): each item's
  // rows enter its own hidden staging before the joint layer loop.
  for (std::size_t i = 0; i < items.size(); ++i) {
    auto& impl = *items[i].session->impl_;
    kernels::Embed(w.token_embd.device, w.token_embd.type, impl.d_tokens,
                   impl.hidden, rows[i], c.hidden_size, stream);
    kernels::RmsNormPlain(impl.hidden, impl.hidden, rows[i], c.hidden_size,
                          c.rms_eps, stream);
  }
  // Caller-owned arena (§4.2): sized for this width, regrown on demand.
  // Opt-out valve (default on): GUFO_GLIMMER_UNIFIED_BATCH=0 keeps the
  // legacy per-session attention cores everywhere (Forward and DecodeBatch).
  const bool use_unified = UnifiedBatchEnabled();
  if (use_unified) {
    std::string scratch_error;
    if (!model->EnsureMixedScratch(mb::MixedBatchScratchBytes(items.size()),
                                   &scratch_error)) {
      return fail("glimmer DecodeBatch scratch allocation failed: " +
                  scratch_error);
    }
  }
  mb::MixedBlk blks[mb::kMaxMixedBlks];
  const float score_scale =
      1.0F / std::sqrt(static_cast<float>(c.head_dim));
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    for (std::size_t i = 0; i < items.size(); ++i) {
      ForwardLayerPre(*items[i].session, il, rows[i],
                      items[i].session->impl_->length, stream);
    }
    bool attn_ran = false;
    if (use_unified) {
      const bool swa = c.LayerIsSwa(il);
      const std::uint32_t window = swa ? c.sliding_window : 0;
      for (std::size_t i = 0; i < items.size(); ++i) {
        Session* s = items[i].session;
        auto& impl = *s->impl_;
        mb::MixedBlk& b = blks[i];
        b.q = impl.qbuf;
        b.gate = nullptr;  // the output gate stays a post-hoc SigmoidMul
        b.out = impl.attn_out;
        b.head_dim = c.head_dim;
        b.heads = c.num_heads;
        b.kv_heads = c.num_kv_heads;
        b.score_scale = score_scale;
        b.q_stride = 0;
        b.k_stride = 0;
        b.v_stride = 0;
        b.k_head_stride = 0;
        b.v_head_stride = 0;
        b.gate_stride = 0;
        b.out_stride = 0;
        b.key_block_bytes = 34;
        b.value_block_bytes = 34;
        if (!use_blocked) {
          b.k = impl.buffers.kv_k[il];
          b.v = impl.buffers.kv_v[il];
          b.k_codec = mb::MixedKvCodec::kF16;
          b.v_codec = mb::MixedKvCodec::kF16;
          b.window = window;
          b.kv_len = impl.length + rows[i];
          b.q_pos_base = impl.length;
          b.n_rows = rows[i];
        } else {
          // Blocked SWA layers are pre-shifted past the window (dense-causal
          // from here, exactly like the single-session path above).
          const std::uint8_t* k2 = nullptr;
          const std::uint8_t* v2 = nullptr;
          const std::uint32_t* sp = nullptr;
          std::uint32_t rebased = 0;
          SwaShiftedKV(*s, il, window, impl.length, stream, k2, v2, sp,
                       rebased);
          (void)sp;
          b.k = k2;
          b.v = v2;
          b.k_codec = mb::MixedKvCodec::kQ8;
          b.v_codec = mb::MixedKvCodec::kQ8;
          b.window = 0;
          b.kv_len = rebased + rows[i];
          b.q_pos_base = rebased;
          b.n_rows = rows[i];
        }
      }
      attn_ran = mb::LaunchMixedBatchAttention(
          blks, items.size(), model->mixed_scratch_,
          model->mixed_scratch_bytes_, stream);
    }
    if (!attn_ran) {
      // Unified refusal (or opt-out): legacy single-session cores,
      // numerically identical to the serial per-session forwards.
      for (std::size_t i = 0; i < items.size(); ++i) {
        ForwardLayerAttn(*items[i].session, il, rows[i],
                         items[i].session->impl_->length, stream,
                         /*try_unified=*/false);
      }
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      ForwardLayerPost(*items[i].session, il, rows[i],
                       items[i].session->impl_->length, stream);
    }
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    ForwardFinish(*items[i].session, rows[i],
                  items[i].session->impl_->length, stream,
                  /*all_logits=*/false);
  }
  (void)hipStreamSynchronize(0);
  return true;
}

bool Session::Verify(std::span<const std::int32_t> tokens,
                     std::uint32_t start_pos, std::vector<std::int32_t>& vbest,
                     std::string* error_msg, TapSink* tap) {
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > kMaxVerify) {
    if (error_msg != nullptr)
      *error_msg = "glimmer Verify needs 1..8 tokens";
    return false;
  }
  if (!Forward(tokens, start_pos, error_msg, /*all_logits=*/true, tap))
    return false;
  auto& impl = *impl_;
  // Device-side argmax over every verify row (the host must never
  // dereference the device verify_logits staging directly).
  core::compute::gemm::Argmax(impl.verify_logits, impl.argmax_scratch,
                              impl.verify_out, n,
                              model_->weights_->config.vocab_size, 0);
  vbest.assign(n, -1);
  (void)hipMemcpy(vbest.data(), impl.verify_out, n * sizeof(std::int32_t),
                  hipMemcpyDeviceToHost);
  // Keep both logits views on the last row so SampleGreedy/Logits work
  // right after a verify (the round commit moves them to the accepted row).
  const std::size_t vocab = impl.host_logits.size();
  hipStream_t stream = 0;
  (void)hipMemcpyAsync(impl.logits,
                       impl.verify_logits +
                           static_cast<std::size_t>(n - 1) * vocab,
                       vocab * sizeof(float), hipMemcpyDeviceToDevice, stream);
  (void)hipMemcpyAsync(impl.host_logits.data(), impl.logits,
                       vocab * sizeof(float), hipMemcpyDeviceToHost, stream);
  (void)hipStreamSynchronize(0);
  return true;
}

bool Session::EnableDraftTaps(std::span<const std::uint32_t> layers,
                               std::string* error_msg) {
  auto& impl = *impl_;
  impl.tap_layers_.assign(layers.begin(), layers.end());
  const auto& w = *model_->weights_;
  for (const std::uint32_t il : impl.tap_layers_) {
    if (il >= w.config.num_layers) {
      if (error_msg != nullptr)
        *error_msg = "glimmer tap layer out of range";
      impl.tap_layers_.clear();
      return false;
    }
  }
  if (!impl.tap_layers_.empty() && impl.tap_staging_ == nullptr) {
    std::string error;
    impl.tap_staging_ = impl.buffers.Alloc<float>(
        impl.tap_layers_.size() * w.config.hidden_size, &error);
    if (impl.tap_staging_ == nullptr) {
      if (error_msg != nullptr)
        *error_msg = error;
      impl.tap_layers_.clear();
      return false;
    }
  }
  return true;
}

bool Session::CopyTappedRow(std::span<float> host_out,
                            std::string* error_msg) const {
  const auto& impl = *impl_;
  const std::size_t need =
      impl.tap_layers_.size() * model_->weights_->config.hidden_size;
  if (impl.tap_staging_ == nullptr || host_out.size() < need) {
    if (error_msg != nullptr)
      *error_msg = "glimmer tapped rows unavailable (enable taps first)";
    return false;
  }
  if (hipMemcpy(host_out.data(), impl.tap_staging_,
                need * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
    if (error_msg != nullptr && error_msg->empty())
      *error_msg = "glimmer tapped row copy failed";
    return false;
  }
  return true;
}

bool Session::SpeculativeRound(speculative::IDraftBackend& draft,
                               std::span<const std::int32_t> prefix,
                               std::uint32_t n_draft,
                               std::vector<std::int32_t>* out,
                               std::string* error_msg) {
  const auto fail = [&](const std::string& m) {
    if (error_msg != nullptr && error_msg->empty())
      *error_msg = m;
    return false;
  };
  if (out == nullptr)
    return fail("glimmer SpeculativeRound needs an output");
  if (impl_->length == 0)
    return fail("speculative round needs a prompt");
  if (prefix.size() != impl_->length)
    return fail("speculative prefix desync (needs Length())");
  auto* dflash = dynamic_cast<GlimmerDflashDraftBackend*>(&draft);
  if (dflash == nullptr)
    return fail("glimmer SpeculativeRound needs the glimmer-dflash backend");
  const auto tap_span = dflash->TargetHiddenLayerIds();
  const std::vector<std::uint32_t> taps(tap_span.begin(), tap_span.end());
  if (taps.empty() || impl_->tap_layers_ != taps)
    return fail("draft tap mismatch (call EnableDraftTaps first)");
  const std::uint32_t hidden = model_->weights_->config.hidden_size;
  const std::uint32_t feat_width =
      static_cast<std::uint32_t>(taps.size()) * hidden;
  const std::uint32_t P = impl_->length;
  // Anchor prediction for position P (from the last committed row).
  const std::int32_t T0 = SampleGreedy();
  // Plain step shared by the spec-off / no-room / no-proposal paths.
  // It advances the target AND the draft (bonus-equivalent single row),
  // keeping the two in lockstep.
  auto plain_step = [&](std::string* err) -> bool {
    out->assign(1, T0);
    if (!Forward({&T0, 1}, P, err))
      return false;
    if (dflash->primed()) {
      std::vector<float> anchor_feats(feat_width);
      if (!CopyTappedRow(anchor_feats, err))
        return false;
      std::vector<tokenization::TokenId> toks = {
          static_cast<tokenization::TokenId>(T0)};
      speculative::DraftTargetContext ctx;
      ctx.prompt_tokens = toks;
      ctx.prompt_hidden_states = anchor_feats;
      ctx.hidden_size = feat_width;
      ctx.first_token = toks[0];
      if (!dflash->AppendTargetContext(ctx, P)) {
        if (err != nullptr && err->empty())
          *err = "glimmer draft append failed";
        return false;
      }
      std::vector<tokenization::TokenId> empty;
      dflash->AcceptFeedback(empty,
                             static_cast<tokenization::TokenId>(T0));
    }
    return true;
  };
  // Cap the round: verify width and context (commit + bonus must fit).
  if (P + 1 > impl_->capacity)
    return fail("glimmer context exhausted");
  std::uint32_t k = std::min({n_draft, kMaxVerify, impl_->capacity - P - 1});
  if (k == 0)
    return plain_step(error_msg);
  // Advisory prefix for the draft (the draft tracks positions itself).
  std::vector<tokenization::TokenId> advisory;
  advisory.reserve(prefix.size());
  for (const std::int32_t t : prefix)
    advisory.push_back(static_cast<tokenization::TokenId>(t));
  // The verify below performs KV surgery past P; drop the
  // incremental-Sync marker conservatively (next Sync fully resyncs).
  impl_->synced_.clear();
  auto proposal = draft.Propose(advisory, P, k);
  if (proposal.tokens.empty())
    return plain_step(error_msg);
  k = std::min(static_cast<std::uint32_t>(proposal.tokens.size()),
               kMaxVerify);
  if (k == 0)
    return plain_step(error_msg);
  // Verify [d_0..] with a one-shot tap capturing every verify row per
  // tapped layer (host, [taps][k][hidden]).
  std::vector<std::vector<float>> tap_rows(taps.size());
  TapSink sink;
  sink.layers = taps;
  sink.fn = [&](std::uint32_t il, const float* d_rows, std::uint32_t n,
                std::uint32_t start) {
    (void)start;
    for (std::size_t s = 0; s < taps.size(); ++s) {
      if (taps[s] == il) {
        tap_rows[s].assign(static_cast<std::size_t>(n) * hidden, 0.0F);
        (void)hipMemcpy(tap_rows[s].data(), d_rows,
                        static_cast<std::size_t>(n) * hidden * sizeof(float),
                        hipMemcpyDeviceToHost);
      }
    }
  };
  std::vector<std::int32_t> vtoks;
  vtoks.reserve(k);
  for (std::uint32_t i = 0; i < k; ++i)
    vtoks.push_back(static_cast<std::int32_t>(proposal.tokens[i]));
  std::vector<std::int32_t> vbest;
  if (!Verify(vtoks, P, vbest, error_msg, &sink))
    return false;
  // Greedy prefix-accept: T[0] is the carried anchor prediction,
  // T[j>=1] is verify row j-1 (row j-1 predicts position P+j).
  std::uint32_t accepted = 0;
  while (accepted < k &&
         static_cast<std::int32_t>(proposal.tokens[accepted]) ==
             (accepted == 0 ? T0 : vbest[accepted - 1])) {
    ++accepted;
  }
  const std::int32_t bonus = (accepted == 0) ? T0 : vbest[accepted - 1];
  out->clear();
  for (std::uint32_t i = 0; i < accepted; ++i)
    out->push_back(static_cast<std::int32_t>(proposal.tokens[i]));
  out->push_back(bonus);
  // Commit the target: rewind to P+accepted, then forward the bonus
  // (extends the KV past the accepted rows and refreshes the tap
  // staging with the bonus features for the next round).
  impl_->length = P + accepted;
  if (!Evaluate(bonus, error_msg))
    return false;
  // Commit the draft: accepted rows (true verify features) + bonus row.
  std::vector<tokenization::TokenId> commit_toks;
  commit_toks.reserve(accepted + 1);
  for (std::uint32_t i = 0; i < accepted; ++i)
    commit_toks.push_back(proposal.tokens[i]);
  commit_toks.push_back(static_cast<tokenization::TokenId>(bonus));
  std::vector<float> commit_feats;
  commit_feats.reserve(static_cast<std::size_t>(accepted + 1) * feat_width);
  for (std::uint32_t j = 0; j < accepted; ++j) {
    for (std::size_t s = 0; s < taps.size(); ++s) {
      const float* row =
          tap_rows[s].data() + static_cast<std::size_t>(j) * hidden;
      commit_feats.insert(commit_feats.end(), row, row + hidden);
    }
  }
  std::vector<float> bonus_feats(feat_width);
  if (!CopyTappedRow(bonus_feats, error_msg))
    return false;
  commit_feats.insert(commit_feats.end(), bonus_feats.begin(),
                      bonus_feats.end());
  speculative::DraftTargetContext commit_ctx;
  commit_ctx.prompt_tokens = commit_toks;
  commit_ctx.prompt_hidden_states = commit_feats;
  commit_ctx.hidden_size = feat_width;
  commit_ctx.first_token = commit_toks[0];
  if (!dflash->AppendTargetContext(commit_ctx, P)) {
    if (error_msg != nullptr && error_msg->empty())
      *error_msg = "glimmer draft append failed";
    return false;
  }
  std::vector<tokenization::TokenId> matched(
      proposal.tokens.begin(), proposal.tokens.begin() + accepted);
  dflash->AcceptFeedback(matched, static_cast<tokenization::TokenId>(bonus));
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg, TapSink* tap) {
  if (prompt.size() > impl_->capacity) {
    if (error_msg != nullptr)
      *error_msg = "prompt exceeds glimmer KV capacity";
    return false;
  }
  // Incremental prefix: when the new prompt extends the previously synced
  // prefix, KV rows [0, synced) are intact (Evaluate only appends past
  // them) and only the delta is forwarded. Anything else falls back to a
  // full resync from 0. This turns append-only multi-turn growth from O(n^2)
  // replays into O(n) prefix + delta.
  std::size_t start = 0;
  const auto& prev = impl_->synced_;
  if (!prev.empty() && prompt.size() >= prev.size() &&
      impl_->length >= prev.size() &&
      std::equal(prev.begin(), prev.end(), prompt.begin())) {
    start = prev.size();
  } else {
    impl_->length = 0;
  }
  for (std::size_t off = start; off < prompt.size(); off += kMaxChunk) {
    const std::size_t len = std::min<std::size_t>(kMaxChunk, prompt.size() - off);
    if (!Forward(prompt.subspan(off, len), static_cast<std::uint32_t>(off),
                 error_msg, /*all_logits=*/false, tap))
      return false;
  }
  impl_->synced_.assign(prompt.begin(), prompt.end());
  (void)hipStreamSynchronize(0);
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  return Forward({&token, 1}, impl_->length, error_msg);
}

std::span<const float> Session::Logits() const noexcept {
  return impl_->host_logits;
}

std::uint32_t Session::Length() const noexcept { return impl_->length; }

std::int32_t Session::SampleGreedy() {
  auto& impl = *impl_;
  core::compute::gemm::Argmax(impl.logits, impl.argmax_scratch,
                              impl.argmax_out, 1, model_->weights_->config.vocab_size,
                              0);
  std::int32_t tok = -1;
  (void)hipMemcpy(&tok, impl.argmax_out, sizeof(tok), hipMemcpyDeviceToHost);
  if (getenv("GLIMMER_DUMP_TOPK") != nullptr) {
    const auto& hl = impl.host_logits;
    std::vector<std::uint32_t> idx(hl.size());
    for (std::uint32_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        return hl[a] > hl[b];
                      });
    std::fprintf(stderr, "[glimmer] topk:");
    for (int k = 0; k < 20; ++k)
      std::fprintf(stderr, " %u:%.3f", idx[k], hl[idx[k]]);
    std::fprintf(stderr, " greedy=%d\n", tok);
  }
  return tok;
}

}  // namespace gufo::models::glimmer
