// Llama dense-decoder runtime: GGUF load, chunked prefill, single-token
// decode, greedy sampling. GEMM rides the shared core (GemvBatched for
// n <= 8, DenseF16Gemm prefill); q8 KV rides the shared dispatcher
// (Split-K-128 decode, scalar-blocked prefill); f16 KV uses the local
// dense kernel. No SWA, no MoE, no speculative drafts in v1.
#include "src/models/llama/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "src/core/compute/attention/attention_dispatcher.hpp"
#include "src/core/compute/attention/blocked_kv.hpp"
#include "src/core/compute/attention/blocked_splitk.hpp"
#include "src/core/compute/gemm/gemm.hpp"
#include "src/models/llama/kernels/kernels.hpp"

namespace gufo::models::llama {
namespace {

constexpr std::uint32_t kMaxChunk = 512;

bool Check(hipError_t err, std::string* error_msg, const char* what) {
  if (err == hipSuccess)
    return true;
  if (error_msg != nullptr && error_msg->empty())
    *error_msg = std::string("llama HIP failure in ") + what;
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
    if (!MirrorTensor(l.wo, owned, error_msg))
      return false;
    if (!MirrorTensor(l.ffn_norm, owned, error_msg))
      return false;
    if (!MirrorTensor(l.ffn_gate, owned, error_msg))
      return false;
    if (!MirrorTensor(l.ffn_up, owned, error_msg))
      return false;
    if (!MirrorTensor(l.ffn_down, owned, error_msg))
      return false;
  }
  return true;
}

// One projection: narrow decode rides the fused quant GEMV, prefill converts
// activations to F16 once and rides the shared dense kernel (Q8_0 weights
// dequantize in LDS).
void LlamaGemm(const TensorRef& w, const float* x, __half* x16, float* y,
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
  if (getenv("LLAMA_DUMP_QKV") != nullptr && m == 4096 && k == 4096 &&
      n_tokens == 17) {
    (void)hipStreamSynchronize(stream);
    std::string xpath = "/tmp/llama_x16_first.bin";
    FILE* probe = std::fopen(xpath.c_str(), "rb");
    if (probe == nullptr) {
      std::vector<__half> xh(static_cast<std::size_t>(n_tokens) * k);
      (void)hipMemcpy(xh.data(), x16, xh.size() * sizeof(__half),
                      hipMemcpyDeviceToHost);
      FILE* xf = std::fopen(xpath.c_str(), "wb");
      if (xf != nullptr) {
        (void)std::fwrite(xh.data(), sizeof(__half), xh.size(), xf);
        (void)std::fclose(xf);
      }
      std::vector<float> yh(static_cast<std::size_t>(n_tokens) * m);
      (void)hipMemcpy(yh.data(), y, yh.size() * sizeof(float),
                      hipMemcpyDeviceToHost);
      std::string ypath = "/tmp/llama_y_first.bin";
      FILE* yf = std::fopen(ypath.c_str(), "wb");
      if (yf != nullptr) {
        (void)std::fwrite(yh.data(), sizeof(float), yh.size(), yf);
        (void)std::fclose(yf);
      }
    } else {
      (void)std::fclose(probe);
    }
    return;
  }
}

struct SessionBuffers {
  std::vector<void*> allocations;
  // KV caches, indexed by layer: F16 planes or blocked-Q8 rows.
  std::vector<__half*> kv_k;
  std::vector<__half*> kv_v;
  std::vector<std::uint8_t*> kv_bk;
  std::vector<std::uint8_t*> kv_bv;

  template <typename T>
  T* Alloc(std::size_t count, std::string* error) {
    void* p = nullptr;
    if (hipMalloc(&p, count * sizeof(T)) != hipSuccess || p == nullptr) {
      if (error != nullptr && error->empty())
        *error = "llama session allocation failed";
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
  float* ffn_out{nullptr};
  float* logits{nullptr};
  __half* x16{nullptr};
  std::int32_t* d_tokens{nullptr};
  std::uint32_t* d_pos{nullptr};
  float* splitk_partials{nullptr};
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
    ffn_out = buffers.Alloc<float>(chunk * hidden_dim, error);
    logits = buffers.Alloc<float>(w.config.vocab_size, error);
    // ffn_down stages n * intermediate rows through x16 (not n *
    // hidden); size it for the widest GEMM K or long prefills
    // overflow the heap (faulted at n >= 147 for hidden=4096).
    const std::uint32_t x16_width =
        ff > hidden_dim ? ff : hidden_dim;
    x16 = buffers.Alloc<__half>(chunk * x16_width, error);
    d_tokens = buffers.Alloc<std::int32_t>(chunk, error);
    d_pos = buffers.Alloc<std::uint32_t>(1, error);
    splitk_partials = buffers.Alloc<float>(
        core::compute::blocked_splitk::AttentionBlockedSplitKScratchBytes() /
            sizeof(float),
        error);
    argmax_scratch = buffers.Alloc<core::compute::gemm::ArgmaxCandidate>(
        core::compute::gemm::kArgmaxParts, error);
    argmax_out = buffers.Alloc<std::int32_t>(1, error);
    host_logits.assign(w.config.vocab_size, 0.0f);
    return hidden != nullptr && normed != nullptr && qbuf != nullptr &&
           kbuf != nullptr && vbuf != nullptr && attn_out != nullptr &&
           gate_out != nullptr && up_out != nullptr && ffn_out != nullptr &&
           logits != nullptr && x16 != nullptr && d_tokens != nullptr &&
           d_pos != nullptr && splitk_partials != nullptr &&
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
      *error_msg = "Failed to bind llama weights: " + load_error;
    return nullptr;
  }
  if (options.kv_cache_quant != "f16" && options.kv_cache_quant != "q8") {
    if (error_msg != nullptr)
      *error_msg = "kv cache quant must be f16 or q8";
    return nullptr;
  }
  if (weights->config.head_dim != 128) {
    if (error_msg != nullptr)
      *error_msg = "llama runtime supports head_dim 128 only";
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
      *error_msg = "Failed to mirror llama weights to device: " + load_error;
    return nullptr;
  }
  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*model_owner->reader_,
                                                  &load_error);
  if (tokenizer == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "Failed to create llama tokenizer: " + load_error;
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

std::string Model::ModelName() const { return "llama"; }

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
    add(l.wo);
    add(l.ffn_norm);
    add(l.ffn_gate);
    add(l.ffn_up);
    add(l.ffn_down);
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
  const bool use_blocked = options_.kv_cache_quant == "q8";
  impl.buffers.kv_k.assign(w.config.num_layers, nullptr);
  impl.buffers.kv_v.assign(w.config.num_layers, nullptr);
  impl.buffers.kv_bk.assign(w.config.num_layers, nullptr);
  impl.buffers.kv_bv.assign(w.config.num_layers, nullptr);
  const std::uint32_t kvdim =
      w.config.num_kv_heads * w.config.head_dim;
  for (std::uint32_t il = 0; il < w.config.num_layers; ++il) {
    if (use_blocked) {
      // Post-RoPE rows quantize straight into the shared blocked layout
      // (per-32-block f16 scale, 34 bytes per head-block for Q8).
      const std::uint32_t row_bytes =
          (w.config.head_dim / 32) * 34 * w.config.num_kv_heads;
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
  const bool use_blocked = s.model_->options_.kv_cache_quant == "q8";
  const float score_scale = 1.0F / std::sqrt(static_cast<float>(d));

  float* hidden_buf = impl.hidden;
  // Attention block.
  kernels::RmsNorm(hidden_buf,
                   static_cast<const float*>(l.attn_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  LlamaGemm(l.wq, impl.normed, impl.x16, impl.qbuf, n, qdim,
            hidden, stream);
  LlamaGemm(l.wk, impl.normed, impl.x16, impl.kbuf, n, kvdim,
            hidden, stream);
  LlamaGemm(l.wv, impl.normed, impl.x16, impl.vbuf, n, kvdim,
            hidden, stream);
  if (getenv("LLAMA_DUMP_QKV") != nullptr && n <= 32 && il < 2) {
    (void)hipStreamSynchronize(stream);
    auto dump0 = [&](const float* dev, std::size_t cnt, const char* name) {
      std::vector<float> h(cnt);
      (void)hipMemcpy(h.data(), dev, cnt * sizeof(float),
                      hipMemcpyDeviceToHost);
      std::string path = std::string("/tmp/llama_qkv_L") +
                         std::to_string(il) + "_" + name + "_" +
                         std::to_string(start_pos) + ".bin";
      FILE* f = std::fopen(path.c_str(), "wb");
      if (f != nullptr) {
        (void)std::fwrite(h.data(), sizeof(float), cnt, f);
        (void)std::fclose(f);
      }
    };
    dump0(impl.qbuf, static_cast<std::size_t>(n) * qdim, "qprerope");
  }
  kernels::Rope(impl.qbuf, n, c.num_heads, d, start_pos, c.rope_theta,
                stream);
  kernels::Rope(impl.kbuf, n, c.num_kv_heads, d, start_pos, c.rope_theta,
                stream);
  if (use_blocked) {
    namespace blocked = core::compute::blocked_kv;
    (void)blocked::QuantizeKvRows(
        impl.kbuf, kvdim, d, impl.buffers.kv_bk[il], kvdim,
        c.num_kv_heads, d, n, blocked::KvCodec::kQ8, 127.0F, 34,
        impl.d_pos, stream);
    (void)blocked::QuantizeKvRows(
        impl.vbuf, kvdim, d, impl.buffers.kv_bv[il], kvdim,
        c.num_kv_heads, d, n, blocked::KvCodec::kQ8, 127.0F, 34,
        impl.d_pos, stream);
    // Shared dispatcher: long-context decode rides Split-K-128, everything
    // else runs the scalar direct read (same math, fused dequant).
    namespace attn_disp = core::compute::attention_dispatcher;
    attn_disp::AttentionDispatchInput in;
    in.q_len = n;
    in.kv_len = start_pos + n;
    in.head_dim = d;
    in.heads = c.num_heads;
    in.kv_heads = c.num_kv_heads;
    in.k_codec = attn_disp::DispatchCodec::kQ8;
    in.v_codec = attn_disp::DispatchCodec::kQ8;
    in.window = 0;
    const auto route = attn_disp::RouteAttention(
        in, attn_disp::AttentionDispatchPolicy::WithEnvDefaults());
    bool ran = false;
    if (route == attn_disp::AttentionRoute::kBlockedSplitK) {
      ran = core::compute::blocked_splitk::AttentionBlockedSplitK<
          core::compute::blocked_splitk::KvCodecQ8, 128>(
          impl.qbuf, impl.buffers.kv_bk[il], impl.buffers.kv_bv[il],
          impl.attn_out, n, impl.d_pos, start_pos + n, c.num_heads,
          c.num_kv_heads, d, stream, score_scale, 34, 34,
          impl.splitk_partials,
          core::compute::blocked_splitk::
              AttentionBlockedSplitKScratchBytes());
    }
    if (!ran) {
      blocked::AttentionKv(
          impl.qbuf, impl.buffers.kv_bk[il], impl.buffers.kv_bv[il],
          /*mask=*/nullptr, /*mask_words=*/0, impl.attn_out,
          /*partials=*/nullptr, /*splits=*/1, n, impl.d_pos,
          c.num_heads, c.num_kv_heads, d, /*ratio=*/1, 34, 34,
          blocked::KvCodec::kQ8, stream, score_scale);
    }
  } else {
    kernels::StoreKvF16(impl.kbuf, impl.buffers.kv_k[il], n, kvdim,
                        start_pos, stream);
    kernels::StoreKvF16(impl.vbuf, impl.buffers.kv_v[il], n, kvdim,
                        start_pos, stream);
    kernels::AttentionF16(impl.qbuf, impl.buffers.kv_k[il],
                          impl.buffers.kv_v[il], impl.attn_out, n, start_pos,
                          c.num_heads, c.num_kv_heads, d, stream,
                          score_scale);
    if (getenv("LLAMA_DUMP_QKV") != nullptr && n <= 32) {
      (void)hipStreamSynchronize(stream);
      auto dump = [&](const float* dev, std::size_t cnt, const char* name) {
        std::vector<float> h(cnt);
        (void)hipMemcpy(h.data(), dev, cnt * sizeof(float),
                        hipMemcpyDeviceToHost);
        std::string path = std::string("/tmp/llama_qkv_L") +
                           std::to_string(il) + "_" + name + "_" +
                           std::to_string(start_pos) + ".bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (f != nullptr) {
          (void)std::fwrite(h.data(), sizeof(float), cnt, f);
          (void)std::fclose(f);
        }
      };
    dump(impl.qbuf, static_cast<std::size_t>(n) * qdim, "q");
    dump(impl.kbuf, static_cast<std::size_t>(n) * kvdim, "k");
    dump(impl.vbuf, static_cast<std::size_t>(n) * kvdim, "v");
    dump(impl.attn_out, static_cast<std::size_t>(n) * qdim, "attn");
    dump(impl.normed, static_cast<std::size_t>(n) * hidden, "normed");
    if (il == 0) {
      // Raw Q8_0 mirror bytes of L0 wq for mirror-vs-file audit.
      const std::size_t wq_bytes = w.layers[0].wq.RowBytes() *
                                   w.layers[0].wq.rows;
      std::vector<std::uint8_t> wh(wq_bytes);
      (void)hipMemcpy(wh.data(), w.layers[0].wq.device, wq_bytes,
                      hipMemcpyDeviceToHost);
      std::string wpath = std::string("/tmp/llama_qkv_L0_wqmirror_") +
                          std::to_string(start_pos) + ".bin";
      FILE* wf = std::fopen(wpath.c_str(), "wb");
      if (wf != nullptr) {
        (void)std::fwrite(wh.data(), 1, wq_bytes, wf);
        (void)std::fclose(wf);
      }
    }
  }
  }
  LlamaGemm(l.wo, impl.attn_out, impl.x16, impl.ffn_out, n, hidden,
            qdim, stream);
  // Llama has no post-attention norm: plain residual add.
  kernels::ResidualAdd(impl.ffn_out, hidden_buf,
                       static_cast<std::size_t>(n) * hidden, stream);
  if (getenv("LLAMA_DUMP_QKV") != nullptr && n <= 32 && il < 2) {
    (void)hipStreamSynchronize(stream);
    auto dump2 = [&](const float* dev, std::size_t cnt, const char* name) {
      std::vector<float> h(cnt);
      (void)hipMemcpy(h.data(), dev, cnt * sizeof(float),
                      hipMemcpyDeviceToHost);
      std::string path = std::string("/tmp/llama_qkv_L") +
                         std::to_string(il) + "_" + name + "_" +
                         std::to_string(start_pos) + ".bin";
      FILE* f = std::fopen(path.c_str(), "wb");
      if (f != nullptr) {
        (void)std::fwrite(h.data(), sizeof(float), cnt, f);
        (void)std::fclose(f);
      }
    };
    dump2(impl.ffn_out, static_cast<std::size_t>(n) * hidden, "wo_out");
  }

  // FFN block (SwiGLU).
  kernels::RmsNorm(hidden_buf,
                   static_cast<const float*>(l.ffn_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  LlamaGemm(l.ffn_gate, impl.normed, impl.x16, impl.gate_out, n,
            c.intermediate_size, hidden, stream);
  LlamaGemm(l.ffn_up, impl.normed, impl.x16, impl.up_out, n,
            c.intermediate_size, hidden, stream);
  kernels::SiluMul(impl.gate_out, impl.up_out, impl.gate_out,
                   static_cast<std::size_t>(n) * c.intermediate_size,
                   stream);
  LlamaGemm(l.ffn_down, impl.gate_out, impl.x16, impl.ffn_out, n,
            hidden, c.intermediate_size, stream);
  if (getenv("LLAMA_DUMP_QKV") != nullptr && n <= 32 && il < 2) {
    (void)hipStreamSynchronize(stream);
    auto dump3 = [&](const float* dev, std::size_t cnt, const char* name) {
      std::vector<float> h(cnt);
      (void)hipMemcpy(h.data(), dev, cnt * sizeof(float),
                      hipMemcpyDeviceToHost);
      std::string path = std::string("/tmp/llama_qkv_L") +
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
  // Llama has no post-FFN norm: plain residual add.
  kernels::ResidualAdd(impl.ffn_out, hidden_buf,
                       static_cast<std::size_t>(n) * hidden, stream);
  if (getenv("LLAMA_DUMP_QKV") != nullptr && n <= 32) {
    (void)hipStreamSynchronize(stream);
    std::vector<float> h(static_cast<std::size_t>(n) * hidden);
    (void)hipMemcpy(h.data(), hidden_buf, h.size() * sizeof(float),
                    hipMemcpyDeviceToHost);
    std::string path = std::string("/tmp/llama_qkv_L") +
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
                            std::uint32_t start_pos, hipStream_t stream) {
  auto& w = *s.model_->weights_;
  const auto& c = w.config;
  const std::uint32_t hidden = c.hidden_size;
  auto& impl = *s.impl_;
  kernels::RmsNorm(impl.hidden,
                   static_cast<const float*>(w.output_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  const float* last_row =
      impl.normed + static_cast<std::size_t>(n - 1) * hidden;
  LlamaGemm(w.output, last_row, impl.x16, impl.logits, 1,
            c.vocab_size, hidden, stream);
  if (getenv("LLAMA_DUMP_QKV") != nullptr && n <= 32) {
    (void)hipStreamSynchronize(stream);
    auto dump = [&](const float* dev, std::size_t cnt, const char* name) {
      std::vector<float> h(cnt);
      (void)hipMemcpy(h.data(), dev, cnt * sizeof(float),
                      hipMemcpyDeviceToHost);
      std::string path = std::string("/tmp/llama_") + name + "_" +
                         std::to_string(start_pos) + ".bin";
      FILE* f = std::fopen(path.c_str(), "wb");
      if (f != nullptr) {
        (void)std::fwrite(h.data(), sizeof(float), cnt, f);
        (void)std::fclose(f);
      }
    };
    dump(last_row, hidden, "finalhidden");
    dump(impl.logits, c.vocab_size, "logits");
  }
  (void)hipMemcpyAsync(impl.host_logits.data(), impl.logits,
                       impl.host_logits.size() * sizeof(float),
                       hipMemcpyDeviceToHost, stream);
  impl.length = start_pos + n;
}

bool Session::Forward(std::span<const std::int32_t> tokens,
                      std::uint32_t start_pos, std::string* error_msg) {
  const auto& w = *model_->weights_;
  const std::uint32_t hidden = w.config.hidden_size;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > kMaxChunk) {
    if (error_msg != nullptr)
      *error_msg = "llama Forward needs 1..512 tokens";
    return false;
  }
  if (start_pos + n > model_->options_.max_context) {
    if (error_msg != nullptr)
      *error_msg = "llama context exhausted";
    return false;
  }
  hipStream_t stream = 0;
  auto& impl = *impl_;
  if (model_->options_.kv_cache_quant == "q8") {
    (void)hipMemcpyAsync(impl.d_pos, &start_pos, sizeof(start_pos),
                         hipMemcpyHostToDevice, stream);
  }
  std::vector<std::int32_t> host_tokens(tokens.begin(), tokens.end());
  if (host_tokens.size() > kMaxChunk) {
    if (error_msg != nullptr)
      *error_msg = "llama chunk exceeds device token staging";
    return false;
  }
  (void)hipMemcpyAsync(impl.d_tokens, host_tokens.data(),
                       host_tokens.size() * sizeof(std::int32_t),
                       hipMemcpyHostToDevice, stream);
  if (getenv("LLAMA_DEBUG_SUMS") != nullptr && n <= 32) {
    (void)hipMemset(impl.hidden, 0,
                    static_cast<std::size_t>(n) * hidden * sizeof(float));
    std::vector<std::int32_t> ids_back(n, -999);
    (void)hipMemcpy(ids_back.data(), impl.d_tokens, n * sizeof(std::int32_t),
                    hipMemcpyDeviceToHost);
    std::fprintf(stderr, "[llama] d_tokens_back:");
    for (std::uint32_t ti = 0; ti < n && ti < 20; ++ti)
      std::fprintf(stderr, " %d", ids_back[ti]);
    std::fprintf(stderr, "\n");
  }
  kernels::Embed(w.token_embd.device, w.token_embd.type,
                 impl.d_tokens, impl.hidden, n, hidden,
                 stream);
  if (getenv("LLAMA_DEBUG_SUMS") != nullptr && n <= 32) {
    (void)hipStreamSynchronize(stream);
    std::fprintf(stderr, "[llama] embd type=%d bytes=%zu ids:",
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
      std::fprintf(stderr, "[llama] %s sum=%f maxabs=%f\n", tag, s, mx);
    };
    sum_of(impl.hidden, static_cast<std::size_t>(n) * hidden, "embed");
  }
  for (std::uint32_t il = 0; il < w.config.num_layers; ++il) {
    ForwardLayer(*this, il, n, start_pos, stream);
    if (getenv("LLAMA_DEBUG_SUMS") != nullptr && n <= 32 &&
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
        std::fprintf(stderr, "[llama] L%u %s sum=%f maxabs=%f\n", layer, tag,
                     s, mx);
      };
      const std::uint32_t qdim_l =
          w.config.num_heads * w.config.head_dim;
      sum_of(impl.qbuf, static_cast<std::size_t>(n) * qdim_l, "qbuf", il);
      sum_of(impl.attn_out, static_cast<std::size_t>(n) * qdim_l, "attn", il);
      sum_of(impl.hidden, static_cast<std::size_t>(n) * hidden, "post", il);
    }
  }
  ForwardFinish(*this, n, start_pos, stream);
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.size() > model_->options_.max_context) {
    if (error_msg != nullptr)
      *error_msg = "prompt exceeds llama KV capacity";
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
  if (getenv("LLAMA_DUMP_TOPK") != nullptr) {
    const auto& hl = impl.host_logits;
    std::vector<std::uint32_t> idx(hl.size());
    for (std::uint32_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        return hl[a] > hl[b];
                      });
    std::fprintf(stderr, "[llama] topk:");
    for (int k = 0; k < 20; ++k)
      std::fprintf(stderr, " %u:%.3f", idx[k], hl[idx[k]]);
    std::fprintf(stderr, " greedy=%d\n", tok);
  }
  return tok;
}

}  // namespace gufo::models::llama
