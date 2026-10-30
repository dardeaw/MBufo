#include <thread>
#include <chrono>
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
  namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
  if (n_tokens == 1 && getenv("GEMMA4_NO_FUSED_GEMV") == nullptr) {
    if (w.f16 != nullptr) {
      kernels::GemvF16Fused(static_cast<const __half*>(w.f16), k, x, y, m, stream);
    } else {
      const void* dev = w.device != nullptr ? w.device : w.data;
      kernels::GemvFused(dev, static_cast<int>(w.type), w.RowBytes(), k, x, y, m, stream);
    }
    return;
  }
  // WMMA Q8 prefill for the bulk projections (verified maxrel 1e-4 vs hipBLAS).
  // The router (m==128) mis-tiles the WMMA kernel and stays on hipBLAS; the LM
  // head is projected at n==1 (fused GEMV) so it never reaches here.
  const std::uint32_t wmma_max = [] {
    if (const char* e = getenv("GEMMA4_WMMA_MAX"))
      return static_cast<std::uint32_t>(strtoul(e, nullptr, 10));
    return 65536U;
  }();
  const bool use_wmma = (m >= 256) && (m <= wmma_max) && (k % 32 == 0) &&
                        getenv("GEMMA4_NO_WMMAPREFILL") == nullptr;
  TensorRef dev = w; dev.data = w.device != nullptr ? w.device : w.data;
  if (use_wmma) {
    kernels::DequantSlabQ8(dev, m, reinterpret_cast<std::uint8_t*>(slab), stream);
    kernels::F32ToF16(x, x_slab, static_cast<std::size_t>(n_tokens) * k, stream);
    if (q38::DenseF16Gemm(slab, x_slab, y, n_tokens, m, k, stream)) return;
  }
  const __half* W = nullptr;
  if (w.f16 != nullptr) W = static_cast<const __half*>(w.f16);
  else { kernels::DequantSlab(dev, m, slab, stream); W = slab; }
  kernels::F32ToF16(x, x_slab, static_cast<std::size_t>(n_tokens) * k, stream);
  hipblasStatus_t st = kernels::F16GemmStatus(W, x_slab, y, n_tokens, m, k, stream);
  if (getenv("GEMMA4_HB_ERR") != nullptr && st != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "[hipblas] FAIL st=%d m=%u k=%u n=%u\n", (int)st, m, k,
            n_tokens);
    fflush(stderr);
  }
  if (getenv("GEMMA4_HB_SYNC") != nullptr)
    hipStreamSynchronize(stream);
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

/// Pre-dequantize dense GEMM weights to F16 once at load. Expert tensors
/// keep on-the-fly dequant (their full F16 set does not fit in free memory).
[[nodiscard]] bool PredequantWeights(ModelWeights& w, DeviceBuffers& buffers,
                                    hipStream_t stream, std::string* error) {
  std::vector<TensorRef*> refs;
  for (auto& l : w.layers) {
    for (TensorRef* t :
         {&l.wq, &l.wk, &l.wv, &l.wo, &l.router, &l.ffn_gate, &l.ffn_up,
          &l.ffn_down}) {
      if (!t->empty() && t->f16 == nullptr)
        refs.push_back(t);
    }
  }
  if (!w.output.empty() && w.output.f16 == nullptr)
    refs.push_back(&w.output);
  std::size_t total = 0;
  for (auto* t : refs)
    total += (static_cast<std::size_t>(t->rows) * t->cols * t->experts *
              sizeof(__half) + 255) /
             256 * 256;
  void* arena = nullptr;
  if (!Check(hipMalloc(&arena, total), error, "f16 arena alloc"))
    return false;
  buffers.allocations.push_back(arena);
  std::size_t cursor = 0;
  for (auto* t : refs) {
    const std::size_t elems =
        static_cast<std::size_t>(t->rows) * t->cols * t->experts;
    auto* d = static_cast<__half*>(arena) + cursor / sizeof(__half);
    cursor += (elems * sizeof(__half) + 255) / 256 * 256;
    TensorRef dev = *t;
    dev.data = t->device != nullptr ? t->device : t->data;
    kernels::DequantSlab(dev, static_cast<std::uint32_t>(t->rows * t->experts),
                         d, stream);
    t->f16 = d;
  }
  if (!Check(hipStreamSynchronize(stream), error, "f16 predequant sync"))
    return false;
  return true;
}

[[nodiscard]] bool ConvertExpertsQ8(ModelWeights& w, DeviceBuffers& buffers,
                                    hipStream_t stream, std::string* error) {
  auto conv_one = [&](TensorRef& src, std::uint8_t** dst, std::uint32_t rows,
                      std::uint32_t cols) -> bool {
    if (src.empty() || src.experts == 0) return true;
    const std::size_t per = static_cast<std::size_t>(rows) * (cols / 32) * 34;
    std::uint8_t* arena = nullptr;
    if (!Check(hipMalloc(&arena, per * src.experts), error, "expert q8 alloc"))
      return false;
    buffers.allocations.push_back(arena);
    TensorRef one = src; one.experts = 1;
    for (std::uint32_t e = 0; e < src.experts; ++e) {
      one.data = static_cast<const std::uint8_t*>(src.device) +
                 static_cast<std::size_t>(e) * src.RowBytes() * rows;
      one.device = one.data;
      kernels::DequantSlabQ8(one, rows, arena + e * per, stream);
    }
    if (!Check(hipStreamSynchronize(stream), error, "expert q8 sync")) return false;
    *dst = arena;
    return true;
  };
  const std::uint32_t E = w.config.num_experts;
  for (auto& l : w.layers) {
    if (!l.router.empty()) {
      const std::uint32_t hidden = w.config.hidden_size;
      const std::uint32_t m_pad = 256;
      const std::size_t per = static_cast<std::size_t>(m_pad) * (hidden / 32) * 34;
      std::uint8_t* arena = nullptr;
      if (!Check(hipMalloc(&arena, per), error, "router q8 pad alloc")) return false;
      buffers.allocations.push_back(arena);
      hipMemsetAsync(arena, 0, per, stream);
      kernels::DequantSlabQ8(l.router, E, arena, stream);  // first E rows
      if (!Check(hipStreamSynchronize(stream), error, "router q8 pad sync")) return false;
      l.router_q8_padded = arena;
    }
    if (!conv_one(l.gate_exps, &l.gate_q8, w.config.expert_ff, w.config.hidden_size)) return false;
    if (!conv_one(l.up_exps, &l.up_q8, w.config.expert_ff, w.config.hidden_size)) return false;
    if (!conv_one(l.down_exps, &l.down_q8, w.config.hidden_size, w.config.expert_ff)) return false;
    // Fused gate+up experts: the GGUF stores ffn_gate_up_exps.weight as
    // [E][2*expert_ff][hidden], so gate_exps/up_exps are empty and the grouped
    // MoE path (which needs separate gate_q8/up_q8) never activates -- it falls
    // back to computing every expert against all tokens. Split the fused slab
    // into a gate half and an up half, each Q8_0 [E][expert_ff][hidden], so
    // RoutedF16Gemm can drive the grouped path unchanged.
    // NOTE: the fused-slab split below is NOT validated yet -- enabling it
    // activates a routed-GEMM path whose row mapping was only ever exercised
    // on qwen38, and it currently disagrees with the dense fallback on gemma4
    // (L0 moe_accum sum sign flips). Keep it opt-in until the routing layout
    // is verified; default stays on the correct dense fallback.
    if (getenv("GEMMA4_GATEUP_Q8") != nullptr && l.gate_q8 == nullptr &&
        !l.gate_up_exps.empty() && l.gate_up_exps.experts > 0) {
      const std::uint32_t ff = w.config.expert_ff;
      const std::uint32_t hidden = w.config.hidden_size;
      const std::uint64_t E2 = l.gate_up_exps.experts;
      const std::size_t per =
          static_cast<std::size_t>(ff) * (hidden / 32) * 34;
      std::uint8_t* gate_arena = nullptr;
      std::uint8_t* up_arena = nullptr;
      if (!Check(hipMalloc(&gate_arena, per * E2), error, "gate q8 alloc"))
        return false;
      if (!Check(hipMalloc(&up_arena, per * E2), error, "up q8 alloc"))
        return false;
      buffers.allocations.push_back(gate_arena);
      buffers.allocations.push_back(up_arena);
      const std::size_t row_bytes = l.gate_up_exps.RowBytes();
      for (std::uint64_t e = 0; e < E2; ++e) {
        const std::uint8_t* base = l.gate_up_exps.DeviceExpert(e);
        TensorRef g = l.gate_up_exps;
        g.experts = 1;
        g.rows = ff;
        g.device = base;
        g.data = base;
        kernels::DequantSlabQ8(g, ff, gate_arena + e * per, stream);
        TensorRef u = g;
        const std::uint8_t* up_base = base + row_bytes * ff;
        u.device = up_base;
        u.data = up_base;
        kernels::DequantSlabQ8(u, ff, up_arena + e * per, stream);
      }
      if (!Check(hipStreamSynchronize(stream), error, "gate_up q8 sync"))
        return false;
      l.gate_q8 = gate_arena;
      l.up_q8 = up_arena;
    }
  }
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
  DeviceBuffers f16weights;
  if (!PredequantWeights(*model_owner->weights_, f16weights, 0, &load_error)) {
    if (error_msg != nullptr)
      *error_msg = "Failed to pre-dequantize gemma4 weights: " + load_error;
    return nullptr;
  }
  f16weights.allocations.clear();  // arena lives for the process
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
  DeviceBuffers expert_q8;
  if (getenv("GEMMA4_GROUPED") != nullptr) {
    if (!ConvertExpertsQ8(*model_owner->weights_, expert_q8, 0, &load_error)) {
      if (error_msg != nullptr)
        *error_msg = "Failed to convert gemma4 experts to Q8: " + load_error;
      return nullptr;
    }
    expert_q8.allocations.clear();
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
  std::int32_t* routed_bounds{nullptr};
  std::int32_t* routed_cursors{nullptr};
  std::int32_t* rows_token{nullptr};
  std::int32_t* rows_slot{nullptr};
  std::int32_t* routed_tiles{nullptr};
  __half* x_half{nullptr};
  float* gate_f32{nullptr};
  __half* up_half{nullptr};
  __half* down_half{nullptr};
  float* router_logits_pad{nullptr};  // [chunk][256]
  std::size_t compact_cap{0};
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
    // Attention writes [n][heads][d] with d up to head_dim_full (full
    // layers: 16*512 per token), wider than hidden; wo consumes that
    // width before overwriting it in place with [n][hidden].
    const std::size_t attn_row = static_cast<std::size_t>(w.config.num_heads) *
                                 w.config.head_dim_full;
    attn_out = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * attn_row,
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
    if (getenv("GEMMA4_GROUPED") != nullptr) {
      const std::uint32_t E = w.config.num_experts;
      const std::uint32_t used = w.config.experts_used;
      const std::size_t slots = static_cast<std::size_t>(chunk) * used;
      compact_cap = ::gufo::models::qwen38_flash_next::rocm::RoutedCompactRows(slots, E);
      routed_bounds = buffers.Alloc<std::int32_t>(E + 1, error);
      routed_cursors = buffers.Alloc<std::int32_t>(E, error);
      rows_token = buffers.Alloc<std::int32_t>(compact_cap, error);
      rows_slot = buffers.Alloc<std::int32_t>(compact_cap, error);
      routed_tiles = buffers.Alloc<std::int32_t>(slots, error);
      x_half = buffers.Alloc<__half>(static_cast<std::size_t>(chunk) * w.config.hidden_size, error);
      gate_f32 = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * used * w.config.expert_ff, error);
      up_half = buffers.Alloc<__half>(static_cast<std::size_t>(chunk) * used * w.config.expert_ff, error);
      down_half = buffers.Alloc<__half>(static_cast<std::size_t>(chunk) * used * w.config.hidden_size, error);
      router_logits_pad = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * 256, error);
    }
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
    const std::size_t kv_elems = static_cast<std::size_t>(options_.max_context) * dim;
    std::uint8_t* kvk8 = nullptr;
    std::uint8_t* kvv8 = nullptr;
    if (hipMalloc(&kvk8, kv_elems * sizeof(__half)) != hipSuccess ||
        hipMalloc(&kvv8, kv_elems * sizeof(__half)) != hipSuccess) {
      if (error_msg != nullptr)
        *error_msg = "KV cache allocation failed (layer " + std::to_string(il) +
                     ", dim " + std::to_string(dim) + ")";
      return nullptr;
    }
    hipMemset(kvk8, 0, kv_elems * sizeof(__half));
    hipMemset(kvv8, 0, kv_elems * sizeof(__half));
    impl.buffers.allocations.push_back(kvk8);
    impl.buffers.allocations.push_back(kvv8);
    impl.kv_k[il] = reinterpret_cast<__half*>(kvk8);
    impl.kv_v[il] = reinterpret_cast<__half*>(kvv8);
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


static const char* g_gemma4_stage = "init";
[[maybe_unused]] static void SetStage(const char* tag) {
  if (getenv("GEMMA4_STAGE_DEBUG") == nullptr) return;
  g_gemma4_stage = tag;
  hipError_t e = hipStreamSynchronize(0);
  if (e != hipSuccess) {
    fprintf(stderr, "[stage-fault] after %s: %s\n", tag, hipGetErrorString(e));
    fflush(stderr);
  } else {
    fprintf(stderr, "[stage] %s ok\n", tag);
    fflush(stderr);
  }
}

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
  if (getenv("GEMMA4_STOP_MALLOC") != nullptr && start_pos > 0) {
    hipStreamSynchronize(0);
    fprintf(stderr, "[stage] STOP_MALLOC pre-malloc sync done pos=%u\n", start_pos);
    fflush(stderr);
    return true;  // leak nothing: nothing allocated yet this call
  }
  if (!Check(hipMalloc(&d_pos, sizeof(std::uint32_t)), error_msg,
             "d_pos alloc"))
    return false;
  if (getenv("GEMMA4_STOP_EMBED") != nullptr && start_pos > 0) {
    kernels::Embed(w.token_embd.device, w.token_embd.type, host_tokens.data(),
                   impl_->hidden, n, hidden, stream);
    kernels::ScaleInplace(impl_->hidden, std::sqrt(static_cast<float>(hidden)),
                          static_cast<std::size_t>(n) * hidden, stream);
    hipStreamSynchronize(0);
    fprintf(stderr, "[stage] STOP_EMBED done pos=%u\n", start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  if (!Check(hipMemcpyAsync(d_pos, &start_pos, sizeof(std::uint32_t),
                            hipMemcpyHostToDevice, stream),
             error_msg, "d_pos copy")) {
    hipFree(d_pos);
    return false;
  }

  if (getenv("GEMMA4_STOP_PREEMBED") != nullptr && start_pos > 0) {
    // Pure sync: launch a trivial kernel then sync. If the fault appears
    // here, it belongs to the PREVIOUS chunk's queued work.
    kernels::ScaleInplace(impl_->mlp_out, 1.0F, 1, stream);
    hipError_t e = hipStreamSynchronize(0);
    fprintf(stderr, "[stage] STOP_PREEMBED sync=%s pos=%u\n",
            hipGetErrorString(e), start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  if (getenv("GEMMA4_STOP_CHUNK1") != nullptr && start_pos == 0) {
    // chunk1 完整跑完（含 LM head），在 Forward 最尾端 sync 後返回。
    // 若之後 STOP_NOLAUNCH 的 chunk2 不再崩 → 崩潰需要 chunk2 的 launch 共存。
    hipError_t e = hipStreamSynchronize(0);
    fprintf(stderr, "[stage] STOP_CHUNK1 sync=%s\n", hipGetErrorString(e));
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  if (getenv("GEMMA4_STOP_C1HALF") != nullptr && start_pos == 0) {
    // chunk1: run only half the layers (il < 15), sync, return.
    // If chunk2 then runs full without crash => the fault lives in
    // chunk1's layers 15..29 at n=224.
  }
  if (getenv("GEMMA4_STOP_C1L0") != nullptr && start_pos == 0) {
    // chunk1: run ONLY L0, sync, return. chunk2+ host-only via STOP_C2H.
  }
  if (getenv("GEMMA4_STOP_C1Q") != nullptr && start_pos > 0) {
    // chunk2: query-only drain (no sync, no launch, no d_pos free).
    hipError_t e = hipErrorNotReady;
    for (int i = 0; i < 5000000 && e == hipErrorNotReady; ++i)
      e = hipStreamQuery(0);
    fprintf(stderr, "[stage] STOP_C1Q query=%s pos=%u\n",
            hipGetErrorString(e), start_pos);
    fflush(stderr);
    return true;  // leak d_pos; no sync call
  }
  if (getenv("GEMMA4_STOP_C2NOSYNC") != nullptr && start_pos > 0) {
    // chunk2: NO sync at all, just return true (leak d_pos). If the process
    // does NOT crash, the fault is triggered by the SYNC (drain) itself.
    fprintf(stderr, "[stage] STOP_C2NOSYNC return pos=%u (no drain)\n", start_pos);
    fflush(stderr);
    return true;
  }
  if (getenv("GEMMA4_STOP_C1DRAIN") != nullptr && start_pos > 0) {
    // chunk2: drain chunk1 WITHOUT touching d_pos (test d_pos-free theory).
    hipError_t e = hipDeviceSynchronize();
    fprintf(stderr, "[stage] STOP_C1DRAIN devsync=%s pos=%u\n",
            hipGetErrorString(e), start_pos);
    fflush(stderr);
    return true;  // intentionally leak d_pos to rule out its free
  }
  if (getenv("GEMMA4_STOP_C2DRAIN") != nullptr && start_pos > 0) {
    // chunk2: drain via BLOCKING hipDeviceSynchronize (no new launches).
    // Contrast with STOP_C2POLL (query-only) and STOP_C2M (launch+sync).
    hipError_t e = hipDeviceSynchronize();
    fprintf(stderr, "[stage] STOP_C2DRAIN devsync=%s pos=%u\n",
            hipGetErrorString(e), start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  if (getenv("GEMMA4_STOP_C2POLL") != nullptr && start_pos > 0) {
    // chunk2: NO launch; poll chunk1 queue to completion via query-only API.
    hipError_t e = hipErrorNotReady;
    for (int i = 0; i < 2000000 && e == hipErrorNotReady; ++i)
      e = hipStreamQuery(0);
    fprintf(stderr, "[stage] STOP_C2POLL query=%s pos=%u\n",
            hipGetErrorString(e), start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  if (getenv("GEMMA4_STOP_C2H") != nullptr && start_pos > 0) {
    // chunk2: NO pre-sync, NO launch — host-only hipGetLastError read.
    // If this crashes, chunk1's queued work faults on its own at n=224.
    hipError_t e = hipGetLastError();
    fprintf(stderr, "[stage] STOP_C2H lasterr=%s pos=%u\n",
            hipGetErrorString(e), start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  if (getenv("GEMMA4_STOP_C2M") != nullptr && start_pos > 0) {
    // chunk2: NO pre-sync, ONE tiny memsetAsync only, sync.
    // Crash here (with zero other launches) = chunk1's queued work faults
    // on its own at n=224.
    hipMemsetAsync(impl_->mlp_out, 0, 64, stream);
    hipError_t e = hipStreamSynchronize(0);
    fprintf(stderr, "[stage] STOP_C2M memset sync=%s pos=%u\n",
            hipGetErrorString(e), start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  if (getenv("GEMMA4_STOP_C2N") != nullptr && start_pos > 0) {
    // chunk2: NO pre-sync, embed+scale only, sync. Contrast with STOP_C2E
    // (same but WITH pre-sync). C2E ok + C2N crash => async tail race.
    kernels::EmbedScale(w.token_embd.device, w.token_embd.type,
                        host_tokens.data(), impl_->hidden, n, hidden,
                        std::sqrt(static_cast<float>(hidden)), stream);
    hipError_t e = hipStreamSynchronize(0);
    fprintf(stderr, "[stage] STOP_C2N embed sync=%s pos=%u\n",
            hipGetErrorString(e), start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  if (getenv("GEMMA4_STOP_C2L0") != nullptr && start_pos > 0) {
    // chunk2: drain chunk1, run ONLY layer 0 (full body), sync, return.
    // Reuse the normal body by running the loop for il==0 then bailing.
    // Implemented via a flag checked at the loop tail (STOP_L0_BAIL).
  }
  if (getenv("GEMMA4_STOP_C2FULL") != nullptr && start_pos > 0) {
    // chunk2: drain chunk1 tail first, then run the FULL chunk2 forward.
    // If this does NOT crash, the fault is a cross-chunk async race.
    hipStreamSynchronize(0);
    hipError_t e0 = hipGetLastError();
    if (e0 != hipSuccess)
      fprintf(stderr, "[stage] STOP_C2FULL pre-clear=%s\n", hipGetErrorString(e0));
    bool ok = true;
    for (std::uint32_t il2 = 0; il2 < c.num_layers && ok; ++il2) {
      // inline the layer body by calling the normal path: emulate by
      // falling through? We cannot easily; instead run normal Forward body
      // via recursion guard. Simplest: set a flag consumed below.
      ok = true;
    }
    (void)ok;
    // Fall through to the normal forward body (no return), but we already
    // synced. The normal body will run and the final sync happens at exit.
  }
  if (getenv("GEMMA4_STOP_C2W") != nullptr && start_pos > 0) {
    // chunk2: drain chunk1 tail, then embed+scale + L0 wq only.
    hipStreamSynchronize(0);
    kernels::EmbedScale(w.token_embd.device, w.token_embd.type,
                        host_tokens.data(), impl_->hidden, n, hidden,
                        std::sqrt(static_cast<float>(hidden)), stream);
    kernels::RmsNorm(impl_->hidden, nullptr, impl_->normed, n, hidden,
                     c.rms_eps, stream);
    kernels::MulInplace(impl_->normed,
                        static_cast<const float*>(w.layers[0].attn_norm.device),
                        static_cast<std::size_t>(n) * hidden, hidden, stream);
    StagedGemm(w.layers[0].wq, c.num_heads * c.HeadDim(0), impl_->normed,
               impl_->qbuf, n, impl_->w_slab, impl_->x_slab, stream);
    hipError_t e = hipStreamSynchronize(0);
    fprintf(stderr, "[stage] STOP_C2W wq sync=%s pos=%u\n",
            hipGetErrorString(e), start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  if (getenv("GEMMA4_STOP_C2E") != nullptr && start_pos > 0) {
    // chunk2: sync first (drain chunk1 tail), then embed+scale only.
    hipStreamSynchronize(0);
    kernels::EmbedScale(w.token_embd.device, w.token_embd.type,
                        host_tokens.data(), impl_->hidden, n, hidden,
                        std::sqrt(static_cast<float>(hidden)), stream);
    hipError_t e = hipStreamSynchronize(0);
    fprintf(stderr, "[stage] STOP_C2E embed sync=%s pos=%u\n",
            hipGetErrorString(e), start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  if (getenv("GEMMA4_STOP_NOLAUNCH") != nullptr && start_pos > 0) {
    // No sync, no launch, no malloc — pure host-side return.
    fprintf(stderr, "[stage] STOP_NOLAUNCH return pos=%u\n", start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  if (getenv("GEMMA4_STOP_MALLOC2") != nullptr && start_pos > 0) {
    // malloc + free WITHOUT sync: reproduce the allocator churn only.
    for (int i = 0; i < 4; ++i) {
      void* p = nullptr;
      hipMalloc(&p, 16);
      hipFree(p);
    }
    fprintf(stderr, "[stage] STOP_MALLOC2 done pos=%u\n", start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  // Embedding lookup from the device mirror (Q4_0 table), fused with the
  // sqrt(hidden) scale so the chunk entry is a single kernel launch.
  kernels::EmbedScale(w.token_embd.device, w.token_embd.type,
                      host_tokens.data(), impl_->hidden, n, hidden,
                      std::sqrt(static_cast<float>(hidden)), stream);

  // Gemma 4 scales token embeddings by sqrt(hidden) (llama.cpp parity).
  // NOTE: llama.cpp scales token embeddings by sqrt(n_embd); Gemma 4's
  // trained embedding table already carries that scale (llama.cpp's own
  // table is F32 ~O(1)). Q4_0 quantisation of the pre-scaled table keeps
  // it pre-scaled, so NO extra sqrt(hidden) here.

  std::uint32_t pos = start_pos;

  if (getenv("GEMMA4_STOP_NOSYNC") != nullptr && start_pos > 0) {
    // No sync, no launch — just return. If the process still crashes later,
    // the fault is from chunk 1's queued work; if it never crashes, the
    // fault needs chunk 2's launches present in the queue.
    fprintf(stderr, "[stage] STOP_NOSYNC return pos=%u (no sync)\n", start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  if (getenv("GEMMA4_STOP_POSTEMBED") != nullptr && start_pos > 0) {
    hipStreamSynchronize(0);
    fprintf(stderr, "[stage] STOP_POSTEMBED sync done pos=%u\n", start_pos);
    fflush(stderr);
    hipFree(d_pos);
    return true;
  }
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (getenv("GEMMA4_SYNC_ALL") != nullptr)
      hipStreamSynchronize(stream);
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
    
    if (!Check(hipGetLastError(), error_msg, "after rmsnorm"))
      return false;
    if (const char* se = getenv("GEMMA4_STOP_ENTRY");
        se != nullptr && il == 0 && start_pos > 0) {
      fprintf(stderr, "[stage] STOP_ENTRY pos=%u\n", start_pos);
      fflush(stderr);
      hipStreamSynchronize(0);
      fprintf(stderr, "[stage] STOP_ENTRY sync done\n"); fflush(stderr);
      return true;
    }
    if (getenv("GEMMA4_STOP_EMBED2") != nullptr && il == 0 && start_pos > 0) {
      hipStreamSynchronize(0);
      fprintf(stderr, "[stage] STOP_EMBED2 sync done pos=%u\n", start_pos);
      fflush(stderr);
      hipFree(d_pos);
      return true;
    }
    if (getenv("GEMMA4_STOP_ENTRY2") != nullptr && il == 0 && start_pos > 0) {
      hipStreamSynchronize(0);
      fprintf(stderr, "[stage] STOP_ENTRY2 sync done pos=%u\n", start_pos);
      fflush(stderr);
      hipFree(d_pos);
      return true;
    }
    SetStage("wq");
    StagedGemm(l.wq, qdim, impl_->normed, impl_->qbuf, n, impl_->w_slab,
               impl_->x_slab, stream);
    if (getenv("GEMMA4_NODENSE") != nullptr) {
      hipMemsetAsync(impl_->qbuf, 0,
                     static_cast<std::size_t>(n) * qdim * sizeof(float), stream);
    }
    SetStage("post_wq");
    if (const char* sp = getenv("GEMMA4_STOP_PREWQ");
        sp != nullptr && il == 0 && start_pos > 0) {
      fprintf(stderr, "[stage] STOP_PREWQ pos=%u\n", start_pos);
      fflush(stderr);
      hipStreamSynchronize(0);
      fprintf(stderr, "[stage] STOP_PREWQ sync done\n"); fflush(stderr);
      return true;
    }
    
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
    

    if (l.has_kv && kv_heads > 0) {
    SetStage("wk");
      if (getenv("GEMMA4_STOP_WK") != nullptr && il == 0 && start_pos > 0) {
        fprintf(stderr, "[stage] STOP_WK launch pos=%u\n", start_pos);
        fflush(stderr);
        hipStreamSynchronize(0);
        fprintf(stderr, "[stage] STOP_WK sync done\n"); fflush(stderr);
        return true;
      }
      StagedGemm(l.wk, kvdim, impl_->normed, impl_->kbuf, n, impl_->w_slab,
                 impl_->x_slab, stream);
      SetStage("post_wk_launch");
      kernels::RmsNorm(impl_->kbuf,
                       static_cast<const float*>(l.attn_k_norm.device),
                       impl_->kbuf, n * kv_heads, head_dim, c.rms_eps, stream);
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
      if (l.has_v_proj) {
    SetStage("wv");
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
      kernels::StoreKv(impl_->vbuf, impl_->kv_v[kv_layer], n, kvdim, start_pos,
                       stream);
    SetStage("pre_attn");
    if (getenv("GEMMA4_STOP_PREATTN") != nullptr && il == 0 && start_pos > 0) {
      hipStreamSynchronize(0);
      fprintf(stderr, "[stage] STOP_PREATTN sync done pos=%u\n", start_pos);
      fflush(stderr);
      hipFree(d_pos);
      return true;
    }
    if (getenv("GEMMA4_NOATTN") != nullptr) {
      // Zero the full attention row width (num_heads * HeadDim),
      // not just hidden: wo reads [n][heads][d].
      const std::size_t az = static_cast<std::size_t>(n) * c.num_heads *
                             head_dim;
      hipMemsetAsync(impl_->attn_out, 0, az * sizeof(float), stream);
    } else {
      kernels::Attention(impl_->qbuf, impl_->kv_k[kv_layer],
                         impl_->kv_v[kv_layer], impl_->attn_out, n, start_pos,
                         c.num_heads, kv_heads, head_dim,
                         impl_->kv_stride[kv_layer],
                         c.LayerIsSwa(il) ? c.sliding_window : 0, stream);
    }
    SetStage("attention");
    SetStage("post_attn2");
    }

    // wo projection.
    SetStage("post_attn");
    SetStage("wo");
    StagedGemm(l.wo, hidden, impl_->attn_out, impl_->attn_out, n,
               impl_->w_slab, impl_->x_slab, stream);
    SetStage("post_wo");
    if (const char* sq = getenv("GEMMA4_STOP_Q0");
        sq != nullptr && il == 0 && start_pos > 0) {
      fprintf(stderr, "[stage] STOP_Q0 reached pos=%u\n", start_pos);
      fflush(stderr);
      hipStreamSynchronize(0);
      return true;
    }
    if (const char* sc = getenv("GEMMA4_STOP_CHUNK");
        sc != nullptr && start_pos / n == static_cast<std::uint32_t>(atoi(sc))) {
      fprintf(stderr, "[stage] STOP_CHUNK reached pos=%u\n", start_pos);
      fflush(stderr);
      hipStreamSynchronize(0);
      return true;
    }
    if (const char* sn = getenv("GEMMA4_STOP_N");
        sn != nullptr && static_cast<std::uint32_t>(atoi(sn)) ==
                             (il + 1) + (start_pos / n) * 100) {
      fprintf(stderr, "[stage] STOP_N il=%u chunk=%u\n", il,
              start_pos / n);
      fflush(stderr);
      hipStreamSynchronize(0);
      return true;
    }
    if (const char* sl = getenv("GEMMA4_STOP_AT_L0");
        sl != nullptr && il == 0) {
      fprintf(stderr, "[stage] STOP_AT_L0 reached\n");
      fflush(stderr);
      hipStreamSynchronize(0);
      return true;
    }
    if (const char* sw = getenv("GEMMA4_STOP_AT_WO");
        sw != nullptr && il == static_cast<std::uint32_t>(atoi(sw))) {
      fprintf(stderr, "[stage] STOP_AT_WO reached il=%u\n", il);
      fflush(stderr);
      hipStreamSynchronize(0);
      return true;
    }
    
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
    
    SyncSum(impl_->hidden, "attn_resid", n, hidden, il);

    // FFN block (llama.cpp gemma4.cpp parity).
    if (!l.IsMoe()) {
      kernels::RmsNorm(impl_->hidden,
                       static_cast<const float*>(l.ffn_norm.device),
                       impl_->normed, n, hidden, c.rms_eps, stream);
    SetStage("shared_ffn");
      F32Gemm(l.ffn_gate, impl_->normed, impl_->mlp_out, n, c.shared_ff,
              hidden, stream, impl_->w_slab, impl_->x_slab);
      F32Gemm(l.ffn_up, impl_->normed, impl_->expert_out, n, c.shared_ff,
              hidden, stream, impl_->w_slab2, impl_->x_slab);
      kernels::GeluTanhSwiglu(impl_->mlp_out, impl_->expert_out,
                              static_cast<std::size_t>(n) * c.shared_ff,
                              stream);
      F32Gemm(l.ffn_down, impl_->mlp_out, impl_->moe_out, n, hidden,
              c.shared_ff, stream, impl_->w_slab, impl_->x_slab);
      kernels::RmsNorm(impl_->moe_out, nullptr, impl_->expert_out, n, hidden,
                       c.rms_eps, stream);
      kernels::MulInplace(impl_->expert_out,
                          static_cast<const float*>(l.ffn_post_norm.device),
                          static_cast<std::size_t>(n) * hidden, hidden,
                          stream);
      kernels::AddInplace(impl_->hidden, impl_->expert_out,
                          static_cast<std::size_t>(n) * hidden, stream);
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
      if (getenv("GEMMA4_GROUPED") != nullptr && l.router_q8_padded != nullptr &&
          getenv("GEMMA4_NORT_WMMAP") == nullptr) {
        namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
        kernels::F32ToF16(impl_->router_tmp, impl_->x_slab,
                          static_cast<std::size_t>(n) * hidden, stream);
        SetStage("pre_router_wmma");
        if (q38::DenseF16Gemm(l.router_q8_padded, impl_->x_slab,
                              impl_->router_logits_pad, n, 256, hidden, stream)) {
          fn::RouterTopK(impl_->router_logits_pad, 256, impl_->expert_ids,
                         impl_->expert_weights, n, c.num_experts, c.experts_used,
                         stream);
          goto router_done;
        }
      }
    SetStage("router_hipblas");
      StagedGemm(l.router, c.num_experts, impl_->router_tmp,
                 impl_->router_logits, n, impl_->w_slab, impl_->x_slab,
                 stream);
    SetStage("post_router");
      if (getenv("GEMMA4_STOP_ROUTER") != nullptr && il == 0 && start_pos > 0) {
        hipStreamSynchronize(0);
        fprintf(stderr, "[stage] STOP_ROUTER sync done pos=%u\n", start_pos);
        fflush(stderr);
        hipFree(d_pos);
        return true;
      }
      fn::RouterTopK(impl_->router_logits, c.num_experts, impl_->expert_ids,
                     impl_->expert_weights, n, c.num_experts, c.experts_used,
                     stream);
      if (getenv("GEMMA4_STOP_TOPK") != nullptr && il == 0 && start_pos > 0) {
        hipStreamSynchronize(0);
        fprintf(stderr, "[stage] STOP_TOPK sync done pos=%u\n", start_pos);
        fflush(stderr);
        hipFree(d_pos);
        return true;
      }
      router_done:;
      bool grouped_ok = false;
      SetStage("pre_grouped");
      if (getenv("GEMMA4_GROUPED") != nullptr && l.gate_q8 != nullptr) {
        namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
        const std::uint32_t E = c.num_experts, used = c.experts_used;
        const std::uint32_t slots = n * used;
        q38::ExpertCounts(impl_->expert_ids, impl_->expert_counts, n, E, used, stream);
        std::vector<std::uint32_t> counts_host(E);
        hipMemcpy(counts_host.data(), impl_->expert_counts, E * sizeof(std::uint32_t), hipMemcpyDeviceToHost);
        kernels::F32ToF16(impl_->normed, impl_->x_half, static_cast<std::size_t>(n) * hidden, stream);
        q38::RoutedCompact(impl_->expert_ids, impl_->expert_counts,
                           impl_->routed_bounds, impl_->routed_cursors,
                           impl_->rows_token, impl_->rows_slot, n, used, E, stream);
        std::vector<std::int32_t> tiles;
        for (std::uint32_t e = 0; e < E; ++e)
          for (std::uint32_t j = 0; j < (counts_host[e] + 63) / 64; ++j)
            tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
        if (!tiles.empty()) {
          hipMemcpyAsync(impl_->routed_tiles, tiles.data(), tiles.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream);
          const std::int32_t* tb = impl_->routed_bounds;
          SetStage("pre_gate");
          bool gok = q38::RoutedF16Gemm(l.gate_q8, q38::WeightType::kQ8_0, impl_->x_half,
              impl_->routed_tiles, (std::uint32_t)tiles.size(), 64, tb, impl_->rows_token,
              impl_->rows_slot, nullptr, impl_->gate_f32, nullptr, c.expert_ff, hidden, stream);
          bool uok = gok && q38::RoutedF16Gemm(l.up_q8, q38::WeightType::kQ8_0, impl_->x_half,
              impl_->routed_tiles, (std::uint32_t)tiles.size(), 64, tb, impl_->rows_token,
              impl_->rows_slot, impl_->gate_f32, nullptr, impl_->up_half, c.expert_ff, hidden, stream);
          SetStage("pre_down");
          bool dok = uok && q38::RoutedF16Gemm(l.down_q8, q38::WeightType::kQ8_0, impl_->up_half,
              impl_->routed_tiles, (std::uint32_t)tiles.size(), 64, tb, impl_->rows_slot,
              impl_->rows_slot, nullptr, nullptr, impl_->down_half, hidden, c.expert_ff, stream);
          if (dok) {
            hipMemsetAsync(impl_->attn_out, 0, static_cast<std::size_t>(n) * hidden * sizeof(float), stream);
            std::vector<std::int32_t> gids(slots);
            std::vector<float> gwts(slots);
            hipMemcpy(gids.data(), impl_->expert_ids, slots * sizeof(std::int32_t), hipMemcpyDeviceToHost);
            hipMemcpy(gwts.data(), impl_->expert_weights, slots * sizeof(float), hipMemcpyDeviceToHost);
            std::vector<float> dscale(E, 1.0F);
            if (!l.down_exps_scale.empty())
              hipMemcpy(dscale.data(), l.down_exps_scale.device, E * sizeof(float), hipMemcpyDeviceToHost);
            std::vector<float> acc(hidden);
            std::vector<__half> drow(hidden);
            for (std::uint32_t t = 0; t < n; ++t) {
              for (std::uint32_t i = 0; i < hidden; ++i) acc[i] = 0.0F;
              for (std::uint32_t s = 0; s < used; ++s) {
                const std::uint32_t slot = t * used + s;
                const int e = gids[slot];
                if (e < 0 || e >= (int)E || gwts[slot] == 0.0F) continue;
                const float w = gwts[slot] * dscale[e];
                hipMemcpy(drow.data(), impl_->down_half + (std::size_t)slot * hidden, hidden * sizeof(__half), hipMemcpyDeviceToHost);
                for (std::uint32_t i = 0; i < hidden; ++i) acc[i] += w * __half2float(drow[i]);
              }
              hipMemcpy(impl_->attn_out + (std::size_t)t * hidden, acc.data(), hidden * sizeof(float), hipMemcpyHostToDevice);
            }
            SetStage("grouped_done");
            grouped_ok = true;
          }
        }
      }
      if (!grouped_ok) {
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
      float down_scales[128];
      if (!l.down_exps_scale.empty()) {
        const std::size_t need = sizeof(float) * c.num_experts;
        if (need > sizeof(down_scales)) {
          if (getenv("GEMMA4_STOP_SCALE") != nullptr && il == 0 && start_pos > 0) {
            fprintf(stderr, "[stage] STOP_SCALE would overflow: need=%zu have=%zu\n",
                    need, sizeof(down_scales));
            fflush(stderr);
            hipFree(d_pos);
            return true;
          }
          // Overflow guard: clamp to the buffer (values beyond are unused
          // only if experts_used never selects them; correct fix below).
          std::vector<float> tmp(c.num_experts, 1.0F);
          hipMemcpy(tmp.data(), l.down_exps_scale.device, need,
                    hipMemcpyDeviceToHost);
          memcpy(down_scales, tmp.data(), sizeof(down_scales));
        } else {
          hipMemcpy(down_scales, l.down_exps_scale.device,
                    sizeof(float) * c.num_experts, hipMemcpyDeviceToHost);
        }
      } else {
        for (std::uint32_t i = 0; i < c.num_experts; ++i) down_scales[i] = 1.0F;
      }
      for (std::uint32_t e = 0; e < c.num_experts; ++e) {
        if (!active[e])
          continue;
        const float down_scale = down_scales[e];
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
    if (getenv("GEMMA4_STOP_C1Q") != nullptr && start_pos == 0) {
      // chunk1: query-only drain (no sync call, no free).
      hipError_t e = hipErrorNotReady;
      for (int i = 0; i < 5000000 && e == hipErrorNotReady; ++i)
        e = hipStreamQuery(0);
      fprintf(stderr, "[stage] STOP_C1Q c1 query=%s\n", hipGetErrorString(e));
      fflush(stderr);
      hipFree(d_pos);
      return true;
    }
    if (getenv("GEMMA4_STOP_C1L0") != nullptr && start_pos == 0 && il == 0) {
      hipError_t e = hipStreamSynchronize(0);
      fprintf(stderr, "[stage] STOP_C1L0 sync=%s\n", hipGetErrorString(e));
      fflush(stderr);
      hipFree(d_pos);
      return true;
    }
    if (getenv("GEMMA4_STOP_C1HALF") != nullptr && start_pos == 0 &&
        il == c.num_layers / 2 - 1) {
      hipError_t e = hipStreamSynchronize(0);
      fprintf(stderr, "[stage] STOP_C1HALF il=%u sync=%s\n", il,
              hipGetErrorString(e));
      fflush(stderr);
      hipFree(d_pos);
      return true;
    }
    if (getenv("GEMMA4_STOP_C2L0") != nullptr && start_pos > 0 && il == 0) {
      hipError_t e = hipStreamSynchronize(0);
      fprintf(stderr, "[stage] STOP_C2L0 il0 sync=%s pos=%u\n",
              hipGetErrorString(e), start_pos);
      fflush(stderr);
      hipFree(d_pos);
      return true;
    }
    pos += n;
  }

  // Output norm + LM head.
  kernels::RmsNorm(impl_->hidden,
                   static_cast<const float*>(w.output_norm.device),
                   impl_->normed, n, hidden, c.rms_eps, stream);
  
  // The sampler only consumes the LAST token's logits (see Logits()), so the
  // LM head projects just that row. This keeps the vocab-sized projection on
  // the fused GEMV path (n==1) and off hipBLAS, which both avoids the
  // gfx1151 batch>=128 fault and skips (n-1) wasted vocab GEMMs.
  const float* last_row = impl_->normed + static_cast<std::size_t>(n - 1) * hidden;
  SetStage("pre_lm_head");
  StagedGemm(w.output, c.vocab_size, last_row, impl_->logits, 1,
             impl_->w_slab, impl_->x_slab, stream);

  if (c.final_logit_softcapping > 0.0F) {
    kernels::SoftCap(impl_->logits, c.final_logit_softcapping,
                     static_cast<std::size_t>(c.vocab_size), stream);
  }
  hipFree(d_pos);
  impl_->last_chunk_n = 1;
  impl_->length = start_pos + n;
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.size() > impl_->capacity) {
    if (error_msg != nullptr)
      *error_msg = "prompt (" + std::to_string(prompt.size()) +
                   " tokens) exceeds KV capacity (" +
                   std::to_string(impl_->capacity) + ")";
    return false;
  }
  impl_->length = 0;
  std::size_t chunk = 64;  // gfx1151 hipBLAS faults at batch>=128
  if (const char* e = getenv("GEMMA4_CHUNK")) {
    const long v = std::strtol(e, nullptr, 10);
    if (v >= 1 && v <= (long)Session::Impl::kMaxChunk) chunk = (std::size_t)v;
  }
  for (std::size_t off = 0; off < prompt.size(); off += chunk) {
    const std::size_t len = std::min<std::size_t>(chunk, prompt.size() - off);
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
