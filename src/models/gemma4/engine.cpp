#include "src/models/gemma4/engine.hpp"

#include <cmath>
#include <cstring>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <limits>
#include <unordered_set>

#include "src/models/gemma4/kernels/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

#include "src/models/gemma4/weights.hpp"

namespace gufo::models::gemma4 {

namespace {

namespace fn = ::gufo::models::qwen38_flash_next::rocm;

/// Maps a GGML type to the FN kernel WeightType enum.
[[nodiscard]] fn::WeightType SmallType(core::GgmlType t) {
  switch (t) {
    case core::GgmlType::kF32:
      return fn::WeightType::kF32;
    case core::GgmlType::kF16:
      return fn::WeightType::kF16;
    case core::GgmlType::kQ8_0:
      return fn::WeightType::kQ8_0;
    case core::GgmlType::kQ5_1:
      return fn::WeightType::kQ5_1;
    case core::GgmlType::kQ4_K:
      return fn::WeightType::kQ4_K;
    case core::GgmlType::kQ5_K:
      return fn::WeightType::kQ5_K;
    case core::GgmlType::kBF16:
      return fn::WeightType::kBF16;
    default:
      return fn::WeightType::kF32;
  }
}

/// y[n][m] = x[n][k] @ W[m][k]^T over GGML quant weights: dequantize the
/// matrix into the F16 staging slab, then run one hipBLAS GEMM.
void F32Gemm(const TensorRef& w, const float* x, float* y,
             std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
             hipStream_t stream, __half* slab, __half* x_slab) {
  TensorRef dev = w;
  dev.data = w.device != nullptr ? w.device : w.data;
  kernels::DequantSlab(dev, m, slab, stream);
  std::fflush(stderr);
  const hipError_t deq_err = hipStreamSynchronize(stream);
  kernels::F32ToF16(x, x_slab, static_cast<std::size_t>(n_tokens) * k, stream);
  if (m == 4096 || m == 2048) {
    std::vector<float> xa(static_cast<std::size_t>(n_tokens) * k);
    hipMemcpy(xa.data(), x, xa.size() * sizeof(float), hipMemcpyDeviceToHost);
    double s1 = 0; for (std::size_t i = 0; i < xa.size(); ++i) s1 += xa[i];
    // probe x directly (f32) to compare with x_slab
    std::vector<float> xd(static_cast<std::size_t>(n_tokens) * k);
    hipMemcpy(xd.data(), x, xd.size() * sizeof(float), hipMemcpyDeviceToHost);
    double s = 0; float mx = 0;
    for (std::size_t i = 0; i < xd.size(); ++i) { s += xd[i]; float a = xd[i] < 0 ? -xd[i] : xd[i]; if (a > mx) mx = a; }
  }
  if (m == 4096 || m == 2048) {
    const hipError_t xe = hipStreamSynchronize(stream);
    std::vector<__half> xh(static_cast<std::size_t>(n_tokens) * k);
    const hipError_t xme = hipMemcpy(xh.data(), x_slab, xh.size() * sizeof(__half), hipMemcpyDeviceToHost);
    double s = 0; float mx = 0;
    for (std::size_t i = 0; i < xh.size(); ++i) { float v = __half2float(xh[i]); s += v; float a = v < 0 ? -v : v; if (a > mx) mx = a; }
  }
  if (m == 4096) {
    const hipError_t xe = hipStreamSynchronize(stream);
    std::vector<float> xf(static_cast<std::size_t>(n_tokens) * k);
    hipMemcpy(xf.data(), x, xf.size() * sizeof(float), hipMemcpyDeviceToHost);
    double s = 0; float mx = 0;
    for (std::size_t i = 0; i < xf.size(); ++i) { s += xf[i]; float a = xf[i] < 0 ? -xf[i] : xf[i]; if (a > mx) mx = a; }
  }
  if (m == 262144) {
    const hipError_t f2e = hipStreamSynchronize(stream);
  }
  if (m == 4096) {
    std::vector<float> xf(static_cast<std::size_t>(n_tokens) * k);
    hipMemcpy(xf.data(), x, xf.size() * sizeof(float), hipMemcpyDeviceToHost);
    double s = 0; float mx = 0;
    for (std::size_t i = 0; i < xf.size(); ++i) { s += xf[i]; float a = xf[i] < 0 ? -xf[i] : xf[i]; if (a > mx) mx = a; }
  }

  const hipblasStatus_t gst = kernels::F16GemmStatus(slab, x_slab, y, n_tokens, m, k, stream);
  if (m == 262144) {
    const hipError_t g2 = hipStreamSynchronize(stream);
  }
}

[[nodiscard]] bool Check(hipError_t err, std::string* error, const char* what) {
  if (err == hipSuccess)
    return true;
  if (error != nullptr)
    *error = std::string(what) + ": " + hipGetErrorString(err);
  return false;
}

struct DeviceBuffers {
  std::vector<void*> allocations;
  ~DeviceBuffers() {
    for (auto* p : allocations) hipFree(p);
  }
  template <typename T>
  T* Alloc(std::size_t elements, std::string* error) {
    void* p = nullptr;
    if (!Check(hipMalloc(&p, elements * sizeof(T)), error, "hipMalloc"))
      return nullptr;
    allocations.push_back(p);
    return static_cast<T*>(p);
  }
};

/// Upload every bound tensor to device memory (the GGUF mmap is host-only).
[[nodiscard]] bool UploadWeights(ModelWeights& w, DeviceBuffers& buffers,
                                 std::string* error) {
  std::vector<TensorRef*> refs;
  std::size_t total0 = 0;
  auto collect = [&](TensorRef& t) {
    if (!t.empty()) {
      refs.push_back(&t);
      if (t.size_bytes == 0)
        t.size_bytes = static_cast<std::size_t>(t.RowBytes() * t.rows *
                                                t.experts);
      total0 += (t.size_bytes + 255) / 256 * 256;
    }
  };
  collect(w.token_embd);
  collect(w.output_norm);
  const bool tied_output =
      !w.output.empty() && w.output.data == w.token_embd.data;
  if (!w.output.empty() && !tied_output)
    collect(w.output);
  for (auto& l : w.layers) {
    for (TensorRef* t : {&l.attn_norm, &l.wq, &l.wk, &l.wv, &l.wo,
                         &l.attn_q_norm, &l.attn_k_norm, &l.attn_v_norm,
                         &l.attn_post_norm,
                         &l.out_scale, &l.rope_freqs, &l.ffn_norm,
                         &l.ffn_gate, &l.ffn_up, &l.ffn_down,
                         &l.ffn_post_norm, &l.router, &l.router_scale,
                         &l.ffn_pre_norm_2, &l.ffn_post_norm_1,
                         &l.ffn_post_norm_2, &l.gate_exps, &l.up_exps,
                         &l.gate_up_exps, &l.down_exps, &l.down_exps_scale})
      collect(*t);
  }
  // One device arena; each tensor gets a 256-byte aligned slice.
  const std::size_t total = total0;
  void* arena = nullptr;
  if (!Check(hipMalloc(&arena, total), error, "hipMalloc weight arena"))
    return false;
  std::size_t cursor = 0;
  for (auto* t : refs) {
    void* d = static_cast<std::uint8_t*>(arena) + cursor;
    cursor += (t->size_bytes + 255) / 256 * 256;
    if (!Check(hipMemcpy(d, t->data, t->size_bytes, hipMemcpyHostToDevice),
               error, "weight mirror copy"))
      return false;
    // Verify the mirror: sample 64KB at 3 offsets and byte-compare.
    const auto* hb = static_cast<const std::uint8_t*>(t->data);
    const auto* db = static_cast<const std::uint8_t*>(d);
    for (std::size_t off : {std::size_t(0), t->size_bytes / 2,
                            t->size_bytes > 65536 ? t->size_bytes - 65536
                                                  : std::size_t(0)}) {
      const std::size_t n = std::min<std::size_t>(65536, t->size_bytes - off);
      std::vector<std::uint8_t> back(n);
      hipMemcpy(back.data(), db + off, n, hipMemcpyDeviceToHost);
      std::size_t diff = 0;
      for (std::size_t i = 0; i < n; ++i)
        diff += (back[i] != hb[off + i]) ? 1 : 0;
      (void)diff;
    }
    t->device = d;
    t->host_mapped = true;
  }
  if (tied_output)
    w.output.device = w.token_embd.device;
  return true;
}


}  // namespace

struct Model::Impl {};

Model::~Model() = default;

namespace {
// llama.cpp Gemma4 conversion synthesises rope_freqs for the proportional
// RoPE of full-attention layers: 1.0 for the rotated half-entries, 1e30 for
// the unrotated ones (effectively freezing them).
std::vector<float> SyntheticRopeFreqs(std::uint32_t head_dim,
                                      std::uint32_t rotary_dim) {
  // llama.cpp conversion: partial_rotary_factor=0.25 for full_attention.
  const std::uint32_t half = head_dim / 2;
  const std::uint32_t rotated = head_dim / 8;
  std::vector<float> f(half, 1e30f);
  for (std::uint32_t i = 0; i < rotated && i < half; ++i)
    f[i] = 1.0f;
  return f;
}
}  // namespace

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
      *error_msg = "Failed to bind gemma4 weights: " + load_error;
    return nullptr;
  }
  auto model_owner = std::shared_ptr<Model>(new Model());
  model_owner->options_ = options;
  model_owner->reader_ = std::shared_ptr<core::GgufReader>(
      std::move(reader));
  model_owner->weights_ = std::make_unique<ModelWeights>(
      std::move(*weights));
  DeviceBuffers upload;
  if (!UploadWeights(*model_owner->weights_, upload, &load_error)) {
    if (error_msg != nullptr)
      *error_msg = "Failed to mirror gemma4 weights to device: " + load_error;
    return nullptr;
  }
  // Full-attention layers need rope_freqs (proportional rope). The GGUF may
  // omit it; synthesise the llama.cpp conversion pattern and mirror it.
  {
    auto& w = *model_owner->weights_;
    std::vector<float> synth;
    void* synth_dev = nullptr;
    std::uint32_t synth_dim = 0;
    for (std::size_t li = 0; li < w.layers.size(); ++li) {
      auto& l = w.layers[li];
      if (l.swa || !l.rope_freqs.empty())
        continue;
      const std::uint32_t hd = w.config.HeadDim(static_cast<std::uint32_t>(li));
      if (synth.empty() || synth_dim != hd) {
        synth = SyntheticRopeFreqs(hd, hd);
        synth_dim = hd;
        if (!Check(hipMalloc(&synth_dev, synth.size() * sizeof(float)),
                   &load_error, "rope_freqs alloc"))
          return nullptr;
        if (!Check(hipMemcpy(synth_dev, synth.data(),
                             synth.size() * sizeof(float),
                             hipMemcpyHostToDevice),
                   &load_error, "rope_freqs copy"))
          return nullptr;
      }
      l.rope_freqs.data = synth.data();
      l.rope_freqs.device = synth_dev;
      l.rope_freqs.rows = static_cast<std::uint32_t>(synth.size());
      l.rope_freqs.cols = 1;
      l.rope_freqs.type = core::GgmlType::kF32;
    }
  }
  upload.allocations.clear();  // arena lives for the process
  auto tokenizer = tokenization::QwenTokenizer::CreateFromGguf(
      *model_owner->reader_, &load_error);
  if (tokenizer == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "Failed to create gemma4 tokenizer: " + load_error;
    return nullptr;
  }
  model_owner->tokenizer_ = std::move(tokenizer);
  return model_owner;
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  const auto ids = tokenizer_->Encode(text);
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  const std::span<const std::int32_t> one(&token, 1);
  return Decode(one);
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStopToken(static_cast<tokenization::TokenId>(token));
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name").value_or(
      "gemma4"));
}

std::size_t Model::ResidentBytes() const noexcept {
  return 0;  // weights stream from the mapped file; no resident copy.
}

struct Session::Impl {
  DeviceBuffers buffers;
  // KV caches: one F16 cache per KV-producing layer.
  std::vector<__half*> kv_k;
  std::vector<__half*> kv_v;
  std::vector<std::uint32_t> kv_stride;  // elements per token row
  std::uint32_t capacity{0};
  std::uint32_t length{0};

  // Activations (max one prefill chunk of 512 tokens).
  static constexpr std::uint32_t kMaxChunk = 512;
  float* hidden{nullptr};     // [chunk][hidden]
  float* normed{nullptr};     // [chunk][hidden]
  float* qbuf{nullptr};       // [chunk][heads*head_dim]
  float* kbuf{nullptr};       // [chunk][kv_heads*head_dim]
  float* vbuf{nullptr};       // [chunk][kv_heads*head_dim]
  float* attn_out{nullptr};   // [chunk][hidden]
  float* router_tmp{nullptr}; // [chunk][hidden]
  float* router_logits{nullptr};  // [chunk][experts]
  std::int32_t* expert_ids{nullptr};   // [chunk][experts_used]
  float* expert_weights{nullptr};      // [chunk][experts_used]
  std::uint32_t* expert_counts{nullptr};  // [experts]
  float* expert_out{nullptr};  // [chunk*experts_used][expert_ff]
  float* mlp_out{nullptr};     // [chunk][hidden]
  float* moe_out{nullptr};     // [chunk][hidden]
  float* expert_down{nullptr}; // [chunk][hidden] routed-expert down output
  float* logits{nullptr};      // [kMaxChunk][vocab]
  std::uint32_t last_chunk_n{0};  // tokens in the last chunk
  __half* w_slab{nullptr};     // dequant staging (largest matrix)
  __half* w_slab2{nullptr};    // second staging (up projection)
  __half* x_slab{nullptr};     // F16 activation staging
  std::size_t w_slab_elements{0};

  [[nodiscard]] bool Ensure(const ModelWeights& w, std::uint32_t capacity,
                            std::string* error) {
    this->capacity = capacity;
    const std::uint32_t hidden = w.config.hidden_size;
    const std::uint32_t chunk = kMaxChunk;
    this->hidden = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * hidden,
                                        error);
    normed = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * hidden,
                                  error);
    qbuf = buffers.Alloc<float>(
        static_cast<std::size_t>(chunk) * w.config.num_heads *
            w.config.head_dim_full,
        error);
    kbuf = buffers.Alloc<float>(
        static_cast<std::size_t>(chunk) * 8 * w.config.head_dim_full, error);
    vbuf = buffers.Alloc<float>(
        static_cast<std::size_t>(chunk) * 8 * w.config.head_dim_full, error);
    attn_out = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * hidden,
                                    error);
    router_tmp = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * hidden,
                                      error);
    router_logits =
        buffers.Alloc<float>(static_cast<std::size_t>(chunk) *
                                 w.config.num_experts,
                             error);
    expert_ids = buffers.Alloc<std::int32_t>(
        static_cast<std::size_t>(chunk) * w.config.experts_used, error);
    expert_weights = buffers.Alloc<float>(
        static_cast<std::size_t>(chunk) * w.config.experts_used, error);
    expert_counts = buffers.Alloc<std::uint32_t>(w.config.num_experts, error);
    expert_out = buffers.Alloc<float>(static_cast<std::size_t>(chunk) *
                                          w.config.experts_used *
                                          w.config.expert_ff,
                                      error);
    mlp_out = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * hidden,
                                   error);
    moe_out = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * hidden,
                                   error);
    expert_down = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * hidden,
                                       error);
    logits = buffers.Alloc<float>(
        static_cast<std::size_t>(kMaxChunk) * w.config.vocab_size, error);
    // Weight slab: the LM head (vocab x hidden) is the largest projection.
    const std::size_t slab = static_cast<std::size_t>(
        static_cast<std::uint64_t>(w.config.vocab_size) * hidden);
    w_slab_elements = slab;
    w_slab = buffers.Alloc<__half>(slab, error);
    w_slab2 = buffers.Alloc<__half>(
        static_cast<std::size_t>(w.config.shared_ff) * hidden, error);
    const std::uint32_t qdim_max =
        w.config.num_heads * w.config.head_dim_full;
    const std::uint32_t xwidth =
        qdim_max > hidden ? qdim_max : hidden;
    x_slab = buffers.Alloc<__half>(
        static_cast<std::size_t>(chunk) * xwidth, error);
    if (this->hidden == nullptr || normed == nullptr || qbuf == nullptr ||
        kbuf == nullptr || vbuf == nullptr || attn_out == nullptr ||
        router_tmp == nullptr || router_logits == nullptr ||
        expert_ids == nullptr || expert_weights == nullptr ||
        expert_counts == nullptr || expert_out == nullptr ||
        mlp_out == nullptr || moe_out == nullptr || logits == nullptr ||
        w_slab == nullptr || w_slab2 == nullptr || x_slab == nullptr) {
      if (error != nullptr && error->empty())
        *error = "gemma4 session allocation failed";
      return false;
    }
    return true;
  }
};

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
  // Allocate one KV cache per producing layer, indexed by layer id.
  impl.kv_k.assign(w.config.num_layers, nullptr);
  impl.kv_v.assign(w.config.num_layers, nullptr);
  impl.kv_stride.assign(w.config.num_layers, 0);
  for (std::uint32_t il = 0; il < w.config.num_layers; ++il) {
    if (!w.config.LayerHasKv(il))
      continue;
    // Every non-SWA layer owns its own KV cache (llama.cpp: only SWA layers
    // share the preceding producer's cache).
    if (w.config.KvProducingLayer(il) != il) {
    }
    const std::uint32_t dim = w.config.KvHeads(il) * w.config.HeadDim(il);
    impl.kv_stride[il] = dim;
    impl.kv_k[il] = impl.buffers.Alloc<__half>(
        static_cast<std::size_t>(options_.max_context) * dim, error_msg);
    impl.kv_v[il] = impl.buffers.Alloc<__half>(
        static_cast<std::size_t>(options_.max_context) * dim, error_msg);
    if (impl.kv_k[il] == nullptr || impl.kv_v[il] == nullptr) {
      if (error_msg != nullptr && error_msg->empty())
        *error_msg = "KV cache allocation failed (layer " +
                     std::to_string(il) + ", dim " + std::to_string(dim) + ")";
      return nullptr;
    }
  }
  return session;
}

std::span<const float> Session::Logits() const noexcept {
  // Last token's logits row.
  // Last chunk's last-token logits row.
  const std::uint32_t chunk_n = impl_->last_chunk_n > 0
                                    ? impl_->last_chunk_n - 1
                                    : 0;
  return {impl_->logits + static_cast<std::size_t>(chunk_n) *
                              model_->weights_->config.vocab_size,
          model_->weights_->config.vocab_size};
}

std::uint32_t Session::Length() const noexcept { return impl_->length; }

std::int32_t Session::SampleGreedy() const {
  const auto logits = Logits();
  std::uint32_t best = 0;
  float best_v = -std::numeric_limits<float>::infinity();
  std::vector<float> host(logits.size());
  hipMemcpy(host.data(), logits.data(), logits.size() * sizeof(float),
             hipMemcpyDeviceToHost);
  for (std::size_t i = 0; i < host.size(); ++i) {
    if (host[i] > best_v) {
      best_v = host[i];
      best = static_cast<std::uint32_t>(i);
    }
  }
  return static_cast<std::int32_t>(best);
}

namespace {

/// y = x @ W^T over GGML quant weights (correctness-first path).
void StagedGemm(const TensorRef& w, std::uint64_t m_rows, const float* x,
                float* y, std::uint32_t n_tokens, __half* slab,
                __half* x_slab, hipStream_t stream) {
  F32Gemm(w, x, y, n_tokens, static_cast<std::uint32_t>(m_rows),
          static_cast<std::uint32_t>(w.cols), stream, slab, x_slab);
}

}  // namespace


static void SyncSum(const float* p, const char* tag, std::uint32_t n,
                    std::uint32_t hidden, std::uint32_t il) {
  if (il != 0 || !getenv("GEMMA4_DEBUG_EMBED") || !p) return;
  hipDeviceSynchronize();
  std::vector<float> h(static_cast<std::size_t>(n) * hidden);
  hipMemcpy(h.data(), p, h.size() * sizeof(float), hipMemcpyDeviceToHost);
  double s = 0; float mx = 0;
  for (auto v : h) { s += v; float a = v < 0 ? -v : v; if (a > mx) mx = a; }
  std::fprintf(stderr, "[gemma4] SYNC L%u %s sum=%.4f maxabs=%.4f\n", il, tag, s, mx);
}

static float DownRow(const float* p, std::uint32_t t, std::uint32_t hidden) {
  std::vector<float> h(hidden);
  hipMemcpy(h.data(), p + static_cast<std::size_t>(t) * hidden,
            h.size() * sizeof(float), hipMemcpyDeviceToHost);
  return h[0];
}


namespace {
}  // namespace

bool Session::Forward(std::span<const std::int32_t> tokens,
                      std::uint32_t start_pos, std::string* error_msg) {
  const auto& w = *model_->weights_;
  const auto& c = w.config;
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  hipStream_t stream = 0;
  std::vector<std::int32_t> host_tokens(tokens.begin(), tokens.end());
  if (getenv("GEMMA4_DEBUG_EMBED"))
    std::fprintf(stderr, "[gemma4] fwd n=%u pos=%u tok0=%d\n", n, start_pos,
                 host_tokens[0]);
  std::uint32_t* d_pos = nullptr;
  if (!Check(hipMalloc(&d_pos, sizeof(std::uint32_t)), error_msg,
             "d_pos alloc"))
    return false;
  if (!Check(hipMemcpyAsync(d_pos, &start_pos, sizeof(std::uint32_t),
                            hipMemcpyHostToDevice, stream),
             error_msg, "d_pos copy")) {
    hipFree(d_pos);
    return false;
  }

  // Embedding lookup from the device mirror (Q4_0 table).
  kernels::Embed(w.token_embd.device, w.token_embd.type, host_tokens.data(),
                 impl_->hidden, n, hidden, stream);

  // llama.cpp parity: scale the token embedding by sqrt(hidden).
  kernels::ScaleInplace(impl_->hidden, std::sqrt(static_cast<float>(hidden)),
                        static_cast<std::size_t>(n) * hidden, stream);
  if (!Check(hipStreamSynchronize(stream), error_msg, "embed sync"))
    return false;

  if (!Check(hipStreamSynchronize(stream), error_msg, "embed sync"))
    return false;
  // Gemma 4 scales token embeddings by sqrt(hidden) (llama.cpp parity).
  // NOTE: llama.cpp scales token embeddings by sqrt(n_embd); Gemma 4's
  // trained embedding table already carries that scale (llama.cpp's own
  // table is F32 ~O(1)). Q4_0 quantisation of the pre-scaled table keeps
  // it pre-scaled, so NO extra sqrt(hidden) here.

  std::uint32_t pos = start_pos;

  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const auto& l = w.layers[il];
    const std::uint32_t kv_layer = c.KvProducingLayer(il);
    const std::uint32_t head_dim = c.HeadDim(il);
    const std::uint32_t kv_heads = c.KvHeads(kv_layer);
    const std::uint32_t qdim = c.num_heads * head_dim;
    const std::uint32_t kvdim = kv_heads * head_dim;

    // Attention block.
    kernels::RmsNorm(impl_->hidden, nullptr, impl_->normed, n, hidden,
                     c.rms_eps, stream);
    kernels::MulInplace(impl_->normed,
                        static_cast<const float*>(l.attn_norm.device),
                        static_cast<std::size_t>(n) * hidden, hidden, stream);
    if (!Check(hipStreamSynchronize(stream), error_msg, "rmsnorm sync"))
      return false;
    if (!Check(hipGetLastError(), error_msg, "after rmsnorm"))
      return false;
    StagedGemm(l.wq, qdim, impl_->normed, impl_->qbuf, n, impl_->w_slab,
               impl_->x_slab, stream);
    if (!Check(hipStreamSynchronize(stream), error_msg, "q gemm sync"))
      return false;
    // QK norm (per-head gamma over head_dim).
    kernels::RmsNorm(impl_->qbuf, static_cast<const float*>(l.attn_q_norm.device),
                     impl_->qbuf, n * c.num_heads, head_dim, c.rms_eps,
                     stream);
    if (!c.LayerIsSwa(il) && !l.rope_freqs.empty()) {
      if (getenv("GEMMA4_DEBUG_EMBED")) {
        std::vector<float> ff(256);
        hipMemcpy(ff.data(), l.rope_freqs.device, 256 * sizeof(float),
                  hipMemcpyDeviceToHost);
        std::fprintf(stderr,
                     "[gemma4] L%u ff[0]=%f ff[63]=%f ff[64]=%f ff[127]=%f "
                     "ff[128]=%f ff[255]=%f theta=%f pos=%u\n",
                     il, ff[0], ff[63], ff[64], ff[127], ff[128], ff[255],
                     c.RopeTheta(il), start_pos);
      }
      kernels::RopeFrequenciesMulti(impl_->qbuf, n, c.num_heads, head_dim,
                                    c.RotaryDim(il), start_pos,
                                    c.RopeTheta(il),
                                    static_cast<const float*>(l.rope_freqs.device),
                                    stream);
    } else {
      kernels::Rope(impl_->qbuf, n, c.num_heads, head_dim, c.RotaryDim(il),
                    start_pos, c.RopeTheta(il), stream);
    }
    if (!Check(hipStreamSynchronize(stream), error_msg, "q rope sync"))
      return false;

    if (l.has_kv && kv_heads > 0) {
      StagedGemm(l.wk, kvdim, impl_->normed, impl_->kbuf, n, impl_->w_slab,
                 impl_->x_slab, stream);
      kernels::RmsNorm(impl_->kbuf,
                       static_cast<const float*>(l.attn_k_norm.device),
                       impl_->kbuf, n * kv_heads, head_dim, c.rms_eps, stream);
      if (!Check(hipStreamSynchronize(stream), error_msg, "k norm sync"))
        return false;
      // No v_proj on this layer (llama: Vcur=Kcur pre-rope). Capture BEFORE k rope.
      if (!l.has_v_proj) {
        hipMemcpyAsync(impl_->vbuf, impl_->kbuf,
                       static_cast<std::size_t>(n) * kvdim * sizeof(float),
                       hipMemcpyDeviceToDevice, stream);
      }
      if (!c.LayerIsSwa(il) && !l.rope_freqs.empty()) {
        kernels::RopeFrequenciesMulti(impl_->kbuf, n, kv_heads, head_dim,
                                      c.RotaryDim(il), start_pos,
                                      c.RopeTheta(il),
                                      static_cast<const float*>(l.rope_freqs.device),
                                      stream);
      } else {
        kernels::Rope(impl_->kbuf, n, kv_heads, head_dim, c.RotaryDim(il),
                      start_pos, c.RopeTheta(il), stream);
      }
      if (!Check(hipStreamSynchronize(stream), error_msg, "k rope sync"))
        return false;
      if (l.has_v_proj) {
        StagedGemm(l.wv, kvdim, impl_->normed, impl_->vbuf, n, impl_->w_slab,
                   impl_->x_slab, stream);
      }
      // (no-v-proj reuse already captured pre-rope above)
      // V RMSNorm (Gemma 4 normalizes values too).
      kernels::RmsNorm(impl_->vbuf,
                       static_cast<const float*>(l.attn_v_norm.device),
                       impl_->vbuf, n * kv_heads, head_dim, c.rms_eps,
                       stream);
      // Store KV at positions [pos, pos+n).
      kernels::StoreKv(impl_->kbuf, impl_->kv_k[kv_layer], n, kvdim, start_pos,
                       stream);
      if (!Check(hipStreamSynchronize(stream), error_msg, "storekv k sync"))
        return false;
      kernels::StoreKv(impl_->vbuf, impl_->kv_v[kv_layer], n, kvdim, start_pos,
                       stream);
      if (!Check(hipStreamSynchronize(stream), error_msg, "storekv v sync"))
        return false;
      kernels::Attention(impl_->qbuf, impl_->kv_k[kv_layer],
                         impl_->kv_v[kv_layer], impl_->attn_out, n, start_pos,
                         c.num_heads, kv_heads, head_dim,
                         impl_->kv_stride[kv_layer],
                         c.LayerIsSwa(il) ? c.sliding_window : 0, stream);
      if (!Check(hipStreamSynchronize(stream), error_msg, "attn sync"))
        return false;
    }

    // wo projection.
    StagedGemm(l.wo, hidden, impl_->attn_out, impl_->attn_out, n,
               impl_->w_slab, impl_->x_slab, stream);
    if (!Check(hipStreamSynchronize(stream), error_msg, "wo gemm sync"))
      return false;
    if (getenv("GEMMA4_DEBUG_EMBED") && il < 3) {
      std::vector<float> probe(static_cast<std::size_t>(n) * hidden);
      hipMemcpy(probe.data(), impl_->attn_out, probe.size() * sizeof(float),
                hipMemcpyDeviceToHost);
      double sum = 0.0, mx = 0.0;
      for (std::size_t i = 0; i < probe.size(); ++i) {
        sum += probe[i];
        double a = probe[i] < 0 ? -probe[i] : probe[i];
        if (a > mx)
          mx = a;
      }
      std::fprintf(stderr, "[gemma4] L%u wo_out sync sum=%f maxabs=%f\n", il,
                   sum, mx);
    }
    SyncSum(impl_->attn_out, "wo_out", n, hidden, il);
    kernels::RmsNorm(impl_->attn_out, nullptr, impl_->normed, n, hidden,
                     c.rms_eps, stream);
    kernels::MulInplace(impl_->normed,
                        static_cast<const float*>(l.attn_post_norm.device),
                        static_cast<std::size_t>(n) * hidden, hidden, stream);
    SyncSum(impl_->normed, "attn_postnormed", n, hidden, il);
    // Residual (llama.cpp: attn_out = attn_post_norm(cur) + inpL).
    kernels::AddInplace(impl_->hidden, impl_->normed,
                        static_cast<std::size_t>(n) * hidden, stream);
    if (!Check(hipStreamSynchronize(stream), error_msg, "attn resid sync"))
      return false;
    SyncSum(impl_->hidden, "attn_resid", n, hidden, il);

    // FFN block (llama.cpp gemma4.cpp parity).
    if (!l.IsMoe()) {
      kernels::RmsNorm(impl_->hidden,
                       static_cast<const float*>(l.ffn_norm.device),
                       impl_->normed, n, hidden, c.rms_eps, stream);
      F32Gemm(l.ffn_gate, impl_->normed, impl_->mlp_out, n, c.shared_ff,
              hidden, stream, impl_->w_slab, impl_->x_slab);
      F32Gemm(l.ffn_up, impl_->normed, impl_->expert_out, n, c.shared_ff,
              hidden, stream, impl_->w_slab2, impl_->x_slab);
      kernels::GeluTanhSwiglu(impl_->mlp_out, impl_->expert_out,
                              static_cast<std::size_t>(n) * c.shared_ff,
                              stream);
      F32Gemm(l.ffn_down, impl_->mlp_out, impl_->moe_out, n, hidden,
              c.shared_ff, stream, impl_->w_slab, impl_->x_slab);
      if (!Check(hipStreamSynchronize(stream), error_msg, "ffn down sync"))
        return false;
      kernels::RmsNorm(impl_->moe_out, nullptr, impl_->expert_out, n, hidden,
                       c.rms_eps, stream);
      kernels::MulInplace(impl_->expert_out,
                          static_cast<const float*>(l.ffn_post_norm.device),
                          static_cast<std::size_t>(n) * hidden, hidden,
                          stream);
      kernels::AddInplace(impl_->hidden, impl_->expert_out,
                          static_cast<std::size_t>(n) * hidden, stream);
      if (!Check(hipStreamSynchronize(stream), error_msg, "ffn resid sync"))
        return false;
    SyncSum(impl_->hidden, "ffn_resid", n, hidden, il);
    } else {
      // Shared expert MLP on attn_out.
      kernels::RmsNorm(impl_->hidden,
                       static_cast<const float*>(l.ffn_norm.device),
                       impl_->normed, n, hidden, c.rms_eps, stream);
      SyncSum(impl_->normed, "ffn_normed", n, hidden, il);
      F32Gemm(l.ffn_gate, impl_->normed, impl_->mlp_out, n, c.shared_ff,
              hidden, stream, impl_->w_slab, impl_->x_slab);
      F32Gemm(l.ffn_up, impl_->normed, impl_->expert_out, n, c.shared_ff,
              hidden, stream, impl_->w_slab2, impl_->x_slab);
      kernels::GeluTanhSwiglu(impl_->mlp_out, impl_->expert_out,
                              static_cast<std::size_t>(n) * c.shared_ff,
                              stream);
      F32Gemm(l.ffn_down, impl_->mlp_out, impl_->moe_out, n, hidden,
              c.shared_ff, stream, impl_->w_slab, impl_->x_slab);
      SyncSum(impl_->moe_out, "shared_down", n, hidden, il);
      kernels::RmsNorm(impl_->moe_out, nullptr, impl_->expert_out, n, hidden,
                       c.rms_eps, stream);
      kernels::MulInplace(impl_->expert_out,
                          static_cast<const float*>(l.ffn_post_norm_1.device),
                          static_cast<std::size_t>(n) * hidden, hidden,
                          stream);
      hipMemcpyAsync(impl_->moe_out, impl_->expert_out,
                     static_cast<std::size_t>(n) * hidden * sizeof(float),
                     hipMemcpyDeviceToDevice, stream);
      // moe_out holds the shared-expert contribution; mlp_out is reused.
      kernels::RmsNorm(impl_->hidden,
                       static_cast<const float*>(l.ffn_pre_norm_2.device),
                       impl_->normed, n, hidden, c.rms_eps, stream);
      // Router: tmp = rmsnorm(hidden) * (1/sqrt(hidden)) * router_scale.
      kernels::RouterInput(impl_->hidden,
                           static_cast<const float*>(l.router_scale.device),
                           impl_->router_tmp, n, hidden, c.rms_eps, stream);
      StagedGemm(l.router, c.num_experts, impl_->router_tmp,
                 impl_->router_logits, n, impl_->w_slab, impl_->x_slab,
                 stream);
      fn::RouterTopK(impl_->router_logits, c.num_experts, impl_->expert_ids,
                     impl_->expert_weights, n, c.num_experts, c.experts_used,
                     stream);
      std::vector<std::int32_t> ids(static_cast<std::size_t>(n) *
                                    c.experts_used);
      hipMemcpy(ids.data(), impl_->expert_ids,
                ids.size() * sizeof(std::int32_t), hipMemcpyDeviceToHost);
      std::vector<float> wts(static_cast<std::size_t>(n) * c.experts_used);
      hipMemcpy(wts.data(), impl_->expert_weights,
                wts.size() * sizeof(float), hipMemcpyDeviceToHost);
      hipMemsetAsync(impl_->attn_out, 0,
                     static_cast<std::size_t>(n) * hidden * sizeof(float),
                     stream);
      std::vector<bool> active(c.num_experts, false);
      for (std::size_t idx = 0; idx < ids.size(); ++idx) {
        if (ids[idx] >= 0 && ids[idx] < static_cast<std::int32_t>(c.num_experts))
          active[static_cast<std::size_t>(ids[idx])] = true;
      }
      for (std::uint32_t e = 0; e < c.num_experts; ++e) {
        if (!active[e])
          continue;
        float down_scale = 1.0F;
        if (!l.down_exps_scale.empty()) {
          kernels::DownloadMirror(
              &down_scale,
              static_cast<const std::uint8_t*>(l.down_exps_scale.device) +
                  sizeof(float) * e,
              sizeof(float));
        }
        TensorRef down_ref = l.down_exps;
        down_ref.data = l.down_exps.DeviceExpert(e);
        down_ref.device = down_ref.data;
        down_ref.experts = 1;
        if (!l.gate_up_exps.empty()) {
          TensorRef gate_ref = l.gate_up_exps;
          gate_ref.data = l.gate_up_exps.DeviceExpert(e);
          gate_ref.device = gate_ref.data;
          gate_ref.experts = 1;
          gate_ref.rows = c.expert_ff;
          TensorRef up_ref = gate_ref;
          up_ref.data = static_cast<const std::uint8_t*>(gate_ref.data) +
                        static_cast<std::size_t>(c.expert_ff) *
                            gate_ref.RowBytes();
          up_ref.device = up_ref.data;
          F32Gemm(gate_ref, impl_->normed, impl_->mlp_out, n, c.expert_ff,
                  hidden, stream, impl_->w_slab, impl_->x_slab);
          F32Gemm(up_ref, impl_->normed, impl_->expert_out, n, c.expert_ff,
                  hidden, stream, impl_->w_slab2, impl_->x_slab);
        } else {
          TensorRef gate_ref = l.gate_exps;
          gate_ref.data = l.gate_exps.DeviceExpert(e);
          gate_ref.device = gate_ref.data;
          gate_ref.experts = 1;
          TensorRef up_ref = l.up_exps;
          up_ref.data = l.up_exps.DeviceExpert(e);
          up_ref.device = up_ref.data;
          up_ref.experts = 1;
          F32Gemm(gate_ref, impl_->normed, impl_->mlp_out, n, c.expert_ff,
                  hidden, stream, impl_->w_slab, impl_->x_slab);
          F32Gemm(up_ref, impl_->normed, impl_->expert_out, n, c.expert_ff,
                  hidden, stream, impl_->w_slab2, impl_->x_slab);
        }
        kernels::GeluTanhSwiglu(impl_->mlp_out, impl_->expert_out,
                                static_cast<std::size_t>(n) * c.expert_ff,
                                stream);
        F32Gemm(down_ref, impl_->mlp_out, impl_->expert_down, n, hidden,
                c.expert_ff, stream, impl_->w_slab, impl_->x_slab);
        if (down_scale != 1.0F) {
          kernels::ScaleInplace(impl_->expert_down, down_scale,
                                static_cast<std::size_t>(n) * hidden, stream);
        }
        for (std::uint32_t t = 0; t < n; ++t) {
          float weight = 0.0F;
          for (std::uint32_t s = 0; s < c.experts_used; ++s) {
            const auto idx = static_cast<std::size_t>(t) * c.experts_used + s;
            if (ids[idx] == static_cast<std::int32_t>(e))
              weight += wts[idx];
          }
          if (weight == 0.0F)
            continue;
          kernels::ScaleInplace(impl_->expert_down + static_cast<std::size_t>(t) *
                                                        hidden,
                                weight, hidden, stream);
          kernels::AddInplace(impl_->attn_out + static_cast<std::size_t>(t) *
                                                    hidden,
                              impl_->expert_down + static_cast<std::size_t>(t) *
                                                       hidden,
                              hidden, stream);
        }
      }

      SyncSum(impl_->attn_out, "moe_accum", n, hidden, il);

      kernels::RmsNorm(impl_->attn_out, nullptr, impl_->normed, n, hidden,
                       c.rms_eps, stream);
      kernels::MulInplace(impl_->normed,
                          static_cast<const float*>(l.ffn_post_norm_2.device),
                          static_cast<std::size_t>(n) * hidden, hidden,
                          stream);
      hipMemcpyAsync(impl_->attn_out, impl_->normed,
                     static_cast<std::size_t>(n) * hidden * sizeof(float),
                     hipMemcpyDeviceToDevice, stream);
      // (routed post_norm_2 already applied above: norm + MulInplace + copy)
      // Combine shared + routed experts, then the final ffn post norm.
      if (il == 0 && n > 1 && getenv("GEMMA4_DEBUG_EMBED")) {
        SyncSum(impl_->attn_out, "accum_pre_add", n, hidden, il);
        SyncSum(impl_->moe_out, "moeout_pre_add", n, hidden, il);
      }
      kernels::AddInplace(impl_->attn_out, impl_->moe_out,
                          static_cast<std::size_t>(n) * hidden, stream);
      SyncSum(impl_->attn_out, "combined", n, hidden, il);
      hipStreamSynchronize(stream);
      kernels::RmsNorm(impl_->attn_out, nullptr, impl_->normed, n, hidden,
                       c.rms_eps, stream);
      kernels::MulInplace(impl_->normed,
                          static_cast<const float*>(l.ffn_post_norm.device),
                          static_cast<std::size_t>(n) * hidden, hidden,
                          stream);
      SyncSum(impl_->normed, "final_scaled", n, hidden, il);
    SyncSum(impl_->normed, "ffn_contrib", n, hidden, il);
      kernels::AddInplace(impl_->hidden, impl_->normed,
                          static_cast<std::size_t>(n) * hidden, stream);
      if (!Check(hipStreamSynchronize(stream), error_msg, "moe resid sync"))
        return false;
    SyncSum(impl_->hidden, "moe_resid", n, hidden, il);
    }

    if (getenv("GEMMA4_DEBUG_EMBED") && il < 8) {
      std::vector<float> probe(static_cast<std::size_t>(n) * hidden);
      hipMemcpy(probe.data(), impl_->hidden, probe.size() * sizeof(float),
                hipMemcpyDeviceToHost);
      double sum = 0.0, mx = 0.0;
      int nan = 0;
      for (std::size_t i = 0; i < probe.size(); ++i) {
        if (std::isnan(probe[i])) {
          ++nan;
          continue;
        }
        sum += probe[i];
        double a = probe[i] < 0 ? -probe[i] : probe[i];
        if (a > mx)
          mx = a;
      }
      std::vector<float> gamma(static_cast<std::size_t>(hidden));
      hipMemcpy(gamma.data(), l.ffn_norm.device, hidden * sizeof(float),
                hipMemcpyDeviceToHost);
      double gsum = 0.0, gmx = 0.0;
      for (std::uint32_t i = 0; i < hidden; ++i) {
        gsum += gamma[i];
        double a = gamma[i] < 0 ? -gamma[i] : gamma[i];
        if (a > gmx)
          gmx = a;
      }
    }
    if (l.has_out_scale) {
      // llama.cpp: layer_scalar applies AFTER the FFN residual, to the
      // layer output (not to the running residual stream).
      float scale_value = 1.0F;
      kernels::DownloadMirror(&scale_value, l.out_scale.device,
                              sizeof(float));
      kernels::LayerScale(impl_->hidden, scale_value,
                          static_cast<std::size_t>(n) * hidden, stream);
    }
    pos += n;
  }

  // Output norm + LM head.
  kernels::RmsNorm(impl_->hidden,
                   static_cast<const float*>(w.output_norm.device),
                   impl_->normed, n, hidden, c.rms_eps, stream);
  if (!Check(hipDeviceSynchronize(), nullptr, "pre-lmhead sync")) {
  }
  StagedGemm(w.output, c.vocab_size, impl_->normed, impl_->logits, n,
             impl_->w_slab, impl_->x_slab, stream);
  if (!Check(hipStreamSynchronize(stream), nullptr, "lmhead sync")) {
  }
  if (c.final_logit_softcapping > 0.0F) {
    kernels::SoftCap(impl_->logits, c.final_logit_softcapping,
                     static_cast<std::size_t>(n) * c.vocab_size, stream);
  }
  hipFree(d_pos);
  impl_->last_chunk_n = n;
  impl_->length = start_pos + n;
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  impl_->length = 0;
  for (std::size_t off = 0; off < prompt.size();
       off += Session::Impl::kMaxChunk) {
    const std::size_t len =
        std::min<std::size_t>(Session::Impl::kMaxChunk, prompt.size() - off);
    if (!Forward(prompt.subspan(off, len), static_cast<std::uint32_t>(off),
                 error_msg))
      return false;
  }
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  return Forward({&token, 1}, impl_->length, error_msg);
}

}  // namespace gufo::models::gemma4
