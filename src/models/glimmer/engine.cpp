// glimmer dense-decoder runtime: GGUF load, chunked prefill, single-token
// decode, greedy sampling. GEMM rides the shared core (GemvBatched decode,
// DenseF16Gemm prefill); attention is a local dense/SWA F16 kernel
// (v1 F16 KV only; blocked-Q8 arrives with DFlash serve).
#include "src/models/glimmer/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/models/glimmer/kernels/kernels.hpp"

namespace gufo::models::glimmer {
namespace {

constexpr std::uint32_t kMaxChunk = 512;
/// Batched-verify row cap: speculative verification batches are short
/// (1 anchor + a handful of drafts); wider batches fall back to serial.
constexpr std::uint32_t kMaxVerify = 8;

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

// One projection: narrow decode rides the fused quant GEMV, prefill converts
// activations to F16 once and rides the shared dense kernel (Q8_0 weights
// dequantize in LDS).
void GlimmerGemm(const TensorRef& w, const float* x, __half* x16, float* y,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream) {
  const void* dev = w.device != nullptr ? w.device : w.data;
  if (n_tokens <= 8) {
    core::compute::gemm::GemvBatched(dev, static_cast<int>(w.type),
                                     w.RowBytes(), k, x, y, n_tokens, m, 0.0F,
                                     stream);
    return;
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
  // KV caches, indexed by layer (F16 planes, v1).
  std::vector<__half*> kv_k;
  std::vector<__half*> kv_v;

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
  __half* x16{nullptr};
  std::int32_t* d_tokens{nullptr};
  core::compute::gemm::ArgmaxCandidate* argmax_scratch{nullptr};
  std::int32_t* argmax_out{nullptr};
  std::uint32_t capacity{0};
  std::uint32_t length{0};
  std::vector<float> host_logits;

  [[nodiscard]] bool Ensure(const ModelWeights& w, std::uint32_t capacity,
                            std::string* error) {
    this->capacity = capacity;
    const std::uint32_t hidden_dim = w.config.hidden_size;
    const std::uint32_t qdim = w.config.num_heads * w.config.head_dim;
    const std::uint32_t kvdim = w.config.num_kv_heads * w.config.head_dim;
    const std::uint32_t ff = w.config.intermediate_size;
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
    x16 = buffers.Alloc<__half>(chunk * hidden_dim, error);
    d_tokens = buffers.Alloc<std::int32_t>(chunk, error);
    argmax_scratch = buffers.Alloc<core::compute::gemm::ArgmaxCandidate>(
        core::compute::gemm::kArgmaxParts, error);
    argmax_out = buffers.Alloc<std::int32_t>(1, error);
    host_logits.assign(w.config.vocab_size, 0.0f);
    return hidden != nullptr && normed != nullptr && qbuf != nullptr &&
           kbuf != nullptr && vbuf != nullptr && attn_out != nullptr &&
           gate_out != nullptr && up_out != nullptr && gate_attn != nullptr &&
           ffn_out != nullptr && logits != nullptr && verify_logits != nullptr &&
           x16 != nullptr && d_tokens != nullptr &&
           argmax_scratch != nullptr && argmax_out != nullptr;
  }
};

Model::~Model() {
  for (void* p : device_allocations_) {
    (void)hipFree(p);
  }
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
  if (options.kv_cache_quant != "f16") {
    if (error_msg != nullptr)
      *error_msg = "glimmer v1 serves f16 KV only (q8 arrives with DFlash serve)";
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
  auto session = std::unique_ptr<Session>(new Session(shared_from_this()));
  std::string error;
  if (!session->impl_->Ensure(*weights_, options_.max_context, &error)) {
    if (error_msg != nullptr)
      *error_msg = error;
    return nullptr;
  }
  const auto& w = *weights_;
  auto& impl = *session->impl_;
  impl.buffers.kv_k.assign(w.config.num_layers, nullptr);
  impl.buffers.kv_v.assign(w.config.num_layers, nullptr);
  const std::uint32_t kvdim =
      w.config.num_kv_heads * w.config.head_dim;
  for (std::uint32_t il = 0; il < w.config.num_layers; ++il) {
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

void Session::ForwardLayer(Session& s, std::uint32_t il, std::uint32_t n,
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
  const float score_scale = 1.0F / std::sqrt(static_cast<float>(d));
  const std::uint32_t window = swa ? c.sliding_window : 0;

  float* hidden_buf = impl.hidden;
  // Pre-attention norm.
  kernels::RmsNorm(hidden_buf,
                   static_cast<const float*>(l.attn_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  GlimmerGemm(l.wq, impl.normed, impl.x16, impl.qbuf, n, qdim,
              hidden, stream);
  GlimmerGemm(l.wk, impl.normed, impl.x16, impl.kbuf, n, kvdim,
              hidden, stream);
  GlimmerGemm(l.wv, impl.normed, impl.x16, impl.vbuf, n, kvdim,
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
  // F16 KV store + dense/SWA causal attention (v1; no blocked-Q8 path).
  kernels::StoreKvF16(impl.kbuf, impl.buffers.kv_k[il], n, kvdim,
                      start_pos, stream);
  kernels::StoreKvF16(impl.vbuf, impl.buffers.kv_v[il], n, kvdim,
                      start_pos, stream);
  kernels::AttentionF16(impl.qbuf, impl.buffers.kv_k[il],
                        impl.buffers.kv_v[il], impl.attn_out, n, start_pos,
                        c.num_heads, c.num_kv_heads, d, stream, score_scale,
                        window);
  // Attention output gate: attn_out *= sigmoid(gate_proj(pre-attn hidden)).
  GlimmerGemm(l.gate, impl.normed, impl.x16, impl.gate_attn, n, qdim,
              hidden, stream);
  kernels::SigmoidMul(impl.gate_attn, impl.attn_out, impl.attn_out,
                      static_cast<std::size_t>(n) * qdim, stream);
  // o_proj, post-attention norm (eps 1e-8), residual add.
  GlimmerGemm(l.wo, impl.attn_out, impl.x16, impl.ffn_out, n, hidden,
              qdim, stream);
  kernels::RmsNormAdd(impl.ffn_out,
                      static_cast<const float*>(l.attn_post_norm.device),
                      hidden_buf, n, hidden, c.post_norm_eps, stream);

  // FFN block (SwiGLU).
  kernels::RmsNorm(hidden_buf,
                   static_cast<const float*>(l.ffn_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  GlimmerGemm(l.ffn_gate, impl.normed, impl.x16, impl.gate_out, n,
            c.intermediate_size, hidden, stream);
  GlimmerGemm(l.ffn_up, impl.normed, impl.x16, impl.up_out, n,
            c.intermediate_size, hidden, stream);
  kernels::SiluMul(impl.gate_out, impl.up_out, impl.gate_out,
                   static_cast<std::size_t>(n) * c.intermediate_size,
                   stream);
  GlimmerGemm(l.ffn_down, impl.gate_out, impl.x16, impl.ffn_out, n,
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
      GlimmerGemm(w.output, row_in, impl.x16, row_out, 1, c.vocab_size,
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
    GlimmerGemm(w.output, last_row, impl.x16, impl.logits, 1,
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
                      bool all_logits) {
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
    ForwardLayer(*this, il, n, start_pos, stream);
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

bool Session::Verify(std::span<const std::int32_t> tokens,
                     std::uint32_t start_pos, std::vector<std::int32_t>& vbest,
                     std::string* error_msg) {
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > kMaxVerify) {
    if (error_msg != nullptr)
      *error_msg = "glimmer Verify needs 1..8 tokens";
    return false;
  }
  if (!Forward(tokens, start_pos, error_msg, /*all_logits=*/true))
    return false;
  vbest.assign(n, -1);
  auto& impl = *impl_;
  (void)hipStreamSynchronize(0);
  const std::size_t vocab = impl.host_logits.size();
  for (std::uint32_t t = 0; t < n; ++t) {
    const float* row = impl.verify_logits + static_cast<std::size_t>(t) * vocab;
    std::int32_t best = 0;
    float best_v = row[0];
    for (std::size_t i = 1; i < vocab; ++i) {
      if (row[i] > best_v) {
        best_v = row[i];
        best = static_cast<std::int32_t>(i);
      }
    }
    vbest[t] = best;
  }
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.size() > model_->options_.max_context) {
    if (error_msg != nullptr)
      *error_msg = "prompt exceeds glimmer KV capacity";
    return false;
  }
  impl_->length = 0;
  for (std::size_t off = 0; off < prompt.size(); off += kMaxChunk) {
    const std::size_t len = std::min<std::size_t>(kMaxChunk, prompt.size() - off);
    if (!Forward(prompt.subspan(off, len), static_cast<std::uint32_t>(off),
                 error_msg))
      return false;
  }
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
