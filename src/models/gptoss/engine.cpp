// GPT-OSS runtime: GGUF load, chunked prefill, single-token decode, greedy
// sampling. Dense GEMM rides the shared core (narrow-prequant decode,
// DenseF16Gemm prefill); the MoE rides the shared vec path (MoeVecGateUp /
// GemvQSlot / MoeSlotEpilogueOverwrite over merged Q4_0 expert slabs) with a
// model-local SwiGLU-OAI activation; attention is a model-local dense F16
// kernel (per-head sinks + sliding window) over f16 KV or over shared
// blocked-Q8 KV dequantized with blocked_kv::DequantizeKvRows.
#include "src/models/gptoss/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "src/core/compute/attention/attention_dispatcher.hpp"
#include "src/core/compute/attention/blocked_kv.hpp"
#include "src/core/compute/gemm/gemm.hpp"
#include "src/models/gptoss/kernels/kernels.hpp"

namespace gufo::models::gptoss {
namespace {

constexpr std::uint32_t kMaxChunk = 128;

bool Check(hipError_t err, std::string* error_msg, const char* what) {
  if (err == hipSuccess)
    return true;
  if (error_msg != nullptr && error_msg->empty())
    *error_msg = std::string("gptoss HIP failure in ") + what;
  return false;
}

// Mirrors one weight tensor to device; converted tensors stay packed (the
// shared launchers dequantize on the fly).
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
    if (!MirrorTensor(l.wo_b, owned, error_msg))
      return false;
    if (!MirrorTensor(l.attn_sinks, owned, error_msg))
      return false;
    if (!MirrorTensor(l.post_norm, owned, error_msg))
      return false;
    if (!MirrorTensor(l.router, owned, error_msg))
      return false;
    if (!MirrorTensor(l.router_b, owned, error_msg))
      return false;
    if (!MirrorTensor(l.gate_b, owned, error_msg))
      return false;
    if (!MirrorTensor(l.up_b, owned, error_msg))
      return false;
    if (!MirrorTensor(l.down_b, owned, error_msg))
      return false;
    if (!MirrorTensor(l.gate_up_exps, owned, error_msg))
      return false;
    if (!MirrorTensor(l.down_exps, owned, error_msg))
      return false;
  }
  return true;
}

// One projection: narrow decode rides the shared prequant -> narrow-Q8 GEMV
// when the contract hits (Q8_0 weights, ungated, K % 32 == 0), otherwise the
// legacy wave32 GEMV; prefill converts activations to F16 once and rides the
// shared dense kernel (Q8_0 weights dequantize in LDS). F32 weights (router)
// ride the legacy wave32 GEMV, chunked to 8 rows by the caller.
void GptossGemm(const TensorRef& w, const float* x, __half* x16, float* y,
                std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                hipStream_t stream, void* preq_scratch = nullptr,
                std::size_t preq_bytes = 0) {
  namespace gemm = core::compute::gemm;
  const void* dev = w.device != nullptr ? w.device : w.data;
  if (n_tokens <= 8) {
    if (w.type == core::GgmlType::kQ8_0 && preq_scratch != nullptr) {
      if (gemm::DecideNarrowDecodePrequantRoute(
              /*weight_q8_0=*/true, /*gated=*/false, n_tokens, k, m) ==
          gemm::NarrowDecodeRoute::kNarrowQ80Prequant) {
        if (gemm::LaunchNarrowQ80Prequant(dev, x, y, k, m, n_tokens,
                                          preq_scratch, preq_bytes, stream)) {
          return;
        }
      }
    }
    gemm::GemvBatched(dev, static_cast<int>(w.type), w.RowBytes(), k, x, y,
                      n_tokens, m, 0.0F, stream);
    return;
  }
  if (w.type == core::GgmlType::kF32) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      gemm::GemvBatched(dev, static_cast<int>(w.type), w.RowBytes(), k,
                        x + static_cast<std::size_t>(t) * k,
                        y + static_cast<std::size_t>(t) * m, 1, m, 0.0F,
                        stream);
    }
    return;
  }
  kernels::F32ToF16(x, x16, static_cast<std::size_t>(n_tokens) * k, stream);
  if (w.type == core::GgmlType::kF16) {
    gemm::UnquantizedF16Gemm(dev, x16, y, n_tokens, m, k, stream);
  } else {
    gemm::DenseF16Gemm(dev, x16, y, n_tokens, m, k, stream);
  }
}

struct SessionBuffers {
  std::vector<void*> allocations;
  // KV caches, indexed by layer: F16 planes or blocked-Q8 rows (+F16 staging
  // for the q8 dequant-then-attend path).
  std::vector<__half*> kv_k;
  std::vector<__half*> kv_v;
  std::vector<std::uint8_t*> kv_bk;
  std::vector<std::uint8_t*> kv_bv;

  template <typename T>
  T* Alloc(std::size_t count, std::string* error) {
    void* p = nullptr;
    if (hipMalloc(&p, count * sizeof(T)) != hipSuccess || p == nullptr) {
      if (error != nullptr && error->empty())
        *error = "gptoss session allocation failed";
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
  float* router_logits{nullptr};
  std::int32_t* expert_ids{nullptr};
  float* expert_weights{nullptr};
  float* moe_gate{nullptr};
  float* moe_up{nullptr};
  float* moe_act{nullptr};
  float* moe_down{nullptr};
  float* ffn_out{nullptr};
  float* logits{nullptr};
  __half* x16{nullptr};
  std::int32_t* d_tokens{nullptr};
  std::uint32_t* d_pos{nullptr};
  void* preq_scratch{nullptr};
  std::size_t preq_bytes{0};
  core::compute::gemm::ArgmaxCandidate* argmax_scratch{nullptr};
  std::int32_t* argmax_out{nullptr};
  std::uint32_t capacity{0};
  std::uint32_t length{0};
  std::vector<float> host_logits;

  [[nodiscard]] bool Ensure(const ModelWeights& w, std::uint32_t capacity,
                            std::string* error) {
    namespace gemm = core::compute::gemm;
    this->capacity = capacity;
    const std::uint32_t hidden_dim = w.config.hidden_size;
    const std::uint32_t qdim = w.config.num_heads * w.config.head_dim;
    const std::uint32_t kvdim = w.config.num_kv_heads * w.config.head_dim;
    const std::uint32_t ff = w.config.expert_ff;
    const std::uint32_t used = w.config.experts_used;
    const std::size_t chunk = kMaxChunk;
    const std::size_t slots = chunk * used;
    hidden = buffers.Alloc<float>(chunk * hidden_dim, error);
    normed = buffers.Alloc<float>(chunk * hidden_dim, error);
    qbuf = buffers.Alloc<float>(chunk * qdim, error);
    kbuf = buffers.Alloc<float>(chunk * kvdim, error);
    vbuf = buffers.Alloc<float>(chunk * kvdim, error);
    attn_out = buffers.Alloc<float>(chunk * qdim, error);
    router_logits =
        buffers.Alloc<float>(chunk * w.config.num_experts, error);
    expert_ids = buffers.Alloc<std::int32_t>(chunk * used, error);
    expert_weights = buffers.Alloc<float>(chunk * used, error);
    moe_gate = buffers.Alloc<float>(slots * ff, error);
    moe_up = buffers.Alloc<float>(slots * ff, error);
    moe_act = buffers.Alloc<float>(slots * ff, error);
    moe_down = buffers.Alloc<float>(slots * hidden_dim, error);
    ffn_out = buffers.Alloc<float>(chunk * hidden_dim, error);
    logits = buffers.Alloc<float>(w.config.vocab_size, error);
    const std::uint32_t max_k =
        hidden_dim > qdim ? hidden_dim : qdim;
    x16 = buffers.Alloc<__half>(chunk * max_k, error);
    d_tokens = buffers.Alloc<std::int32_t>(chunk, error);
    d_pos = buffers.Alloc<std::uint32_t>(1, error);
    preq_bytes = gemm::NarrowQ80PrequantScratchBytes(w.config.vocab_size);
    preq_scratch = buffers.Alloc<std::uint8_t>(preq_bytes, error);
    argmax_scratch = buffers.Alloc<core::compute::gemm::ArgmaxCandidate>(
        core::compute::gemm::kArgmaxParts, error);
    argmax_out = buffers.Alloc<std::int32_t>(1, error);
    host_logits.assign(w.config.vocab_size, 0.0f);
    return hidden != nullptr && normed != nullptr && qbuf != nullptr &&
           kbuf != nullptr && vbuf != nullptr && attn_out != nullptr &&
           router_logits != nullptr && expert_ids != nullptr &&
           expert_weights != nullptr && moe_gate != nullptr &&
           moe_up != nullptr && moe_act != nullptr && moe_down != nullptr &&
           ffn_out != nullptr && logits != nullptr && x16 != nullptr &&
           d_tokens != nullptr && d_pos != nullptr &&
           preq_scratch != nullptr && argmax_scratch != nullptr &&
           argmax_out != nullptr;
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
      *error_msg = "Failed to bind gptoss weights: " + load_error;
    return nullptr;
  }
  if (options.kv_cache_quant != "f16" && options.kv_cache_quant != "q8") {
    if (error_msg != nullptr)
      *error_msg = "kv cache quant must be f16 or q8";
    return nullptr;
  }
  if (weights->config.head_dim != 64) {
    if (error_msg != nullptr)
      *error_msg = "gptoss runtime supports head_dim 64 only";
    return nullptr;
  }
  auto model_owner = std::shared_ptr<Model>(new Model());
  model_owner->options_ = options;
  model_owner->reader_ = std::move(reader);
  model_owner->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  if (!UploadWeights(*model_owner->weights_, model_owner->device_allocations_,
                     &load_error)) {
    if (error_msg != nullptr)
      *error_msg = "Failed to mirror gptoss weights to device: " + load_error;
    return nullptr;
  }
  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*model_owner->reader_,
                                                  &load_error);
  if (tokenizer == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "Failed to create gptoss tokenizer: " + load_error;
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

std::string Model::ModelName() const { return "gptoss"; }

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
    add(l.wo_b);
    add(l.attn_sinks);
    add(l.post_norm);
    add(l.router);
    add(l.router_b);
    add(l.gate_b);
    add(l.up_b);
    add(l.down_b);
    add(l.gate_up_exps);
    add(l.down_exps);
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
      // F16 staging for the dequant-then-attend read path.
      const std::size_t felems =
          static_cast<std::size_t>(options_.max_context) * kvdim;
      auto* kk = impl.buffers.Alloc<__half>(felems, &error);
      auto* vv = impl.buffers.Alloc<__half>(felems, &error);
      if (kb == nullptr || vb == nullptr || kk == nullptr || vv == nullptr) {
        if (error_msg != nullptr)
          *error_msg = "blocked KV cache allocation failed (layer " +
                       std::to_string(il) + ")";
        return nullptr;
      }
      (void)hipMemset(kb, 0, bytes);
      (void)hipMemset(vb, 0, bytes);
      impl.buffers.kv_bk[il] = kb;
      impl.buffers.kv_bv[il] = vb;
      impl.buffers.kv_k[il] = kk;
      impl.buffers.kv_v[il] = vv;
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
  namespace gemm = core::compute::gemm;
  auto& w = *s.model_->weights_;
  const auto& c = w.config;
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t d = c.head_dim;
  const std::uint32_t qdim = c.num_heads * d;
  const std::uint32_t kvdim = c.num_kv_heads * d;
  const std::uint32_t nex = c.num_experts;
  const std::uint32_t used = c.experts_used;
  const std::uint32_t ff = c.expert_ff;
  const auto& l = w.layers[il];
  auto& impl = *s.impl_;
  const bool use_blocked = s.model_->options_.kv_cache_quant == "q8";
  const float score_scale = 1.0F / std::sqrt(static_cast<float>(d));
  const std::uint32_t window = c.LayerWindow(il);

  float* hidden_buf = impl.hidden;
  // Attention block.
  kernels::RmsNorm(hidden_buf,
                   static_cast<const float*>(l.attn_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  GptossGemm(l.wq, impl.normed, impl.x16, impl.qbuf, n, qdim, hidden, stream,
             impl.preq_scratch, impl.preq_bytes);
  GptossGemm(l.wk, impl.normed, impl.x16, impl.kbuf, n, kvdim, hidden, stream,
             impl.preq_scratch, impl.preq_bytes);
  GptossGemm(l.wv, impl.normed, impl.x16, impl.vbuf, n, kvdim, hidden, stream,
             impl.preq_scratch, impl.preq_bytes);
  kernels::RopeYarn(impl.qbuf, n, c.num_heads, d, start_pos, c.rope_theta,
                    c.yarn.low, c.yarn.high, c.yarn.inv_factor, c.yarn.mscale,
                    stream);
  kernels::RopeYarn(impl.kbuf, n, c.num_kv_heads, d, start_pos, c.rope_theta,
                    c.yarn.low, c.yarn.high, c.yarn.inv_factor, c.yarn.mscale,
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
    // Dispatcher consult: d == 64 has no blocked-read kernel (the scalar
    // direct read serves 128/256 only) and the sink/window math is
    // model-local, so every call routes kLegacy here. The consult runs
    // anyway (env-gated diag) so the contract evaluation is explicit, and
    // the shared blocked store is dequantized into F16 staging for the
    // exact local kernel.
    if (getenv("GPTOSS_ATTN_DIAG") != nullptr && il == 0) {
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
      std::fprintf(stderr, "[gptoss] q8 attn route: %s\n",
                   attn_disp::AttentionRouteName(route));
    }
    const std::uint32_t kv_len = start_pos + n;
    (void)blocked::DequantizeKvRows(
        impl.buffers.kv_bk[il], impl.buffers.kv_k[il],
        impl.buffers.kv_bv[il], impl.buffers.kv_v[il],
        kv_len * c.num_kv_heads, d, 34, 34, blocked::KvCodec::kQ8, stream);
    kernels::AttentionF16Sink(
        impl.qbuf, impl.buffers.kv_k[il], impl.buffers.kv_v[il],
        static_cast<const float*>(l.attn_sinks.device), impl.attn_out, n,
        start_pos, c.num_heads, c.num_kv_heads, d, window, stream,
        score_scale);
  } else {
    kernels::StoreKvF16(impl.kbuf, impl.buffers.kv_k[il], n, kvdim,
                        start_pos, stream);
    kernels::StoreKvF16(impl.vbuf, impl.buffers.kv_v[il], n, kvdim,
                        start_pos, stream);
    kernels::AttentionF16Sink(
        impl.qbuf, impl.buffers.kv_k[il], impl.buffers.kv_v[il],
        static_cast<const float*>(l.attn_sinks.device), impl.attn_out, n,
        start_pos, c.num_heads, c.num_kv_heads, d, window, stream,
        score_scale);
  }
  GptossGemm(l.wo, impl.attn_out, impl.x16, impl.ffn_out, n, hidden, qdim,
             stream, impl.preq_scratch, impl.preq_bytes);
  // Attention output bias (gpt-oss wo_b) then residual add.
  kernels::AddRowBias(impl.ffn_out, static_cast<const float*>(l.wo_b.device),
                      n, hidden, stream);
  kernels::ResidualAdd(impl.ffn_out, hidden_buf,
                       static_cast<std::size_t>(n) * hidden, stream);

  // MoE block: post-attention norm, router, shared vec MoE.
  kernels::RmsNorm(hidden_buf,
                   static_cast<const float*>(l.post_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  // Router GEMM (F32) in <= 8-row pieces; bias folded in after.
  for (std::uint32_t off = 0; off < n; off += 8) {
    const std::uint32_t nr = std::min<std::uint32_t>(8, n - off);
    gemm::GemvBatched(l.router.device, static_cast<int>(l.router.type),
                      l.router.RowBytes(), hidden,
                      impl.normed + static_cast<std::size_t>(off) * hidden,
                      impl.router_logits + static_cast<std::size_t>(off) * nex,
                      nr, nex, 0.0F, stream);
  }
  kernels::AddRowBias(impl.router_logits,
                      static_cast<const float*>(l.router_b.device), n, nex,
                      stream);
  gemm::RouterTopK(impl.router_logits, nex, impl.expert_ids,
                   impl.expert_weights, n, nex, used, stream);
  const std::uint32_t slots = n * used;
  const void* gu_dev = l.gate_up_exps.device != nullptr
                           ? l.gate_up_exps.device
                           : l.gate_up_exps.data;
  const void* dn_dev = l.down_exps.device != nullptr
                           ? l.down_exps.device
                           : l.down_exps.data;
  const std::uint64_t gu_stride =
      static_cast<std::uint64_t>(2) * ff * l.gate_up_exps.RowBytes();
  const std::uint64_t dn_stride =
      static_cast<std::uint64_t>(hidden) * l.down_exps.RowBytes();
  gemm::MoeVecGateUpMxfp4(gu_dev, l.gate_up_exps.RowBytes(), gu_stride,
                     impl.normed, impl.expert_ids, impl.moe_gate, impl.moe_up,
                     slots, used, ff, hidden, stream);
  kernels::SwigluOaiBias(impl.moe_gate, impl.moe_up, impl.expert_ids,
                         static_cast<const float*>(l.gate_b.device),
                         static_cast<const float*>(l.up_b.device),
                         impl.moe_act, slots, ff, stream);
  gemm::GemvQSlotMxfp4(dn_dev, l.down_exps.RowBytes(), dn_stride, impl.moe_act,
                  impl.expert_ids, impl.moe_down, slots, hidden, ff, stream);
  kernels::DownBiasAdd(impl.moe_down, impl.expert_ids,
                       static_cast<const float*>(l.down_b.device), slots,
                       hidden, stream);
  gemm::MoeSlotEpilogueOverwrite(
      impl.moe_down, impl.expert_ids, impl.expert_weights,
      /*dscale=*/nullptr, impl.ffn_out, n, used, hidden, stream);
  kernels::ResidualAdd(impl.ffn_out, hidden_buf,
                       static_cast<std::size_t>(n) * hidden, stream);
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
  GptossGemm(w.output, last_row, impl.x16, impl.logits, 1, c.vocab_size,
             hidden, stream, impl.preq_scratch, impl.preq_bytes);
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
      *error_msg = "gptoss Forward needs 1..128 tokens";
    return false;
  }
  if (start_pos + n > model_->options_.max_context) {
    if (error_msg != nullptr)
      *error_msg = "gptoss context exhausted";
    return false;
  }
  hipStream_t stream = 0;
  auto& impl = *impl_;
  if (model_->options_.kv_cache_quant == "q8") {
    (void)hipMemcpyAsync(impl.d_pos, &start_pos, sizeof(start_pos),
                         hipMemcpyHostToDevice, stream);
  }
  std::vector<std::int32_t> host_tokens(tokens.begin(), tokens.end());
  (void)hipMemcpyAsync(impl.d_tokens, host_tokens.data(),
                       host_tokens.size() * sizeof(std::int32_t),
                       hipMemcpyHostToDevice, stream);
  kernels::Embed(w.token_embd.device, w.token_embd.type, impl.d_tokens,
                 impl.hidden, n, hidden, stream);
  for (std::uint32_t il = 0; il < w.config.num_layers; ++il) {
    ForwardLayer(*this, il, n, start_pos, stream);
  }
  ForwardFinish(*this, n, start_pos, stream);
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.size() > model_->options_.max_context) {
    if (error_msg != nullptr)
      *error_msg = "prompt exceeds gptoss KV capacity";
    return false;
  }
  impl_->length = 0;
  for (std::size_t off = 0; off < prompt.size(); off += kMaxChunk) {
    const std::size_t len =
        std::min<std::size_t>(kMaxChunk, prompt.size() - off);
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
                              impl.argmax_out, 1,
                              model_->weights_->config.vocab_size, 0);
  std::int32_t tok = -1;
  (void)hipMemcpy(&tok, impl.argmax_out, sizeof(tok), hipMemcpyDeviceToHost);
  if (getenv("GPTOSS_DUMP_TOPK") != nullptr) {
    const auto& hl = impl.host_logits;
    std::vector<std::uint32_t> idx(hl.size());
    for (std::uint32_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        return hl[a] > hl[b];
                      });
    std::fprintf(stderr, "[gptoss] topk:");
    for (int k = 0; k < 10; ++k)
      std::fprintf(stderr, " %u:%.3f", idx[k], hl[idx[k]]);
    std::fprintf(stderr, " greedy=%d\n", tok);
  }
  return tok;
}

}  // namespace gufo::models::gptoss
