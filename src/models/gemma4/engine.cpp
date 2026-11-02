#include <thread>
#include <chrono>
#include "src/models/gemma4/engine.hpp"
#include "src/models/gemma4/mtp.hpp"

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

namespace {
// Grow-only F32 staging for in-place fused GEMV (x == y, as used by wo):
// blocks read the whole x row while other blocks write y, so reading the
// live buffer races. Staging one exact F32 copy removes the race.
float* g_gemv_stage = nullptr;
std::size_t g_gemv_stage_cap = 0;
const float* StageGemvInput(const float* x, std::size_t count,
                            hipStream_t stream) {
  if (count > g_gemv_stage_cap) {
    void* grown = nullptr;
    if (hipMalloc(&grown, count * sizeof(float)) != hipSuccess) {
      return x;
    }
    g_gemv_stage = static_cast<float*>(grown);
    g_gemv_stage_cap = count;
  }
  (void)hipMemcpyAsync(g_gemv_stage, x, count * sizeof(float),
                       hipMemcpyDeviceToDevice, stream);
  return g_gemv_stage;
}
}  // namespace

/// y[n][m] = x[n][k] @ W[m][k]^T over GGML quant weights.
void F32Gemm(const TensorRef& w, const float* x, float* y,
             std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
             hipStream_t stream, __half* slab, __half* x_slab,
             float softcap = 0.0F, std::uint8_t* x_q8_slab = nullptr,
             bool reuse_x = false) {
  namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
  if (n_tokens <= 8 && getenv("GEMMA4_NO_FUSED_GEMV") == nullptr) {
    const float* xr = x;
    if (x == y) {
      xr = StageGemvInput(x, static_cast<std::size_t>(n_tokens) * k, stream);
    }
    if (w.f16 != nullptr) {
      kernels::GemvBatched(w.f16, 1,
                           static_cast<std::uint64_t>(k) * sizeof(__half), k,
                           xr, y, n_tokens, m, softcap, stream);
    } else {
      const void* dev = w.device != nullptr ? w.device : w.data;
      kernels::GemvBatched(dev, static_cast<int>(w.type), w.RowBytes(), k, xr,
                           y, n_tokens, m, softcap, stream);
    }
    return;
  }
  // WMMA Q8 prefill for the bulk projections (verified maxrel 1e-4 vs hipBLAS).
  const std::uint32_t wmma_max = [] {
    if (const char* e = getenv("GEMMA4_WMMA_MAX"))
      return static_cast<std::uint32_t>(strtoul(e, nullptr, 10));
    return 65536U;
  }();
  const bool use_wmma = (m >= 256) && (m <= wmma_max) && (k % 32 == 0) &&
                        getenv("GEMMA4_NO_WMMAPREFILL") == nullptr;
  TensorRef dev = w; dev.data = w.device != nullptr ? w.device : w.data;
  if (use_wmma &&
      (w.q8 != nullptr || w.type == core::GgmlType::kQ4_0 ||
       w.type == core::GgmlType::kQ8_0)) {
    const void* q8_src = w.q8;
    if (q8_src == nullptr) {
      if (w.type == core::GgmlType::kQ8_0) {
        q8_src = dev.data;
      } else {
        kernels::DequantSlabQ8(dev, m, reinterpret_cast<std::uint8_t*>(slab),
                               stream);
        q8_src = slab;
      }
    }
    if (x_q8_slab != nullptr && getenv("GEMMA4_PREFILL_W8A8") != nullptr) {
      if (!reuse_x) {
        q38::QuantizeQ8Tiled(x, x_q8_slab, n_tokens, k, stream);
      }
      if (q38::W8A8Gemm(q8_src, x_q8_slab, y, n_tokens, m, k, stream)) {
        if (softcap > 0.0F) {
          kernels::SoftCap(y, softcap, static_cast<std::size_t>(n_tokens) * m,
                           stream);
        }
        return;
      }
    }
    if (!reuse_x) {
      kernels::F32ToF16(x, x_slab, static_cast<std::size_t>(n_tokens) * k,
                        stream);
    }
    if (q38::DenseF16Gemm(q8_src, x_slab, y, n_tokens, m, k, stream)) {
      if (softcap > 0.0F) {
        kernels::SoftCap(y, softcap, static_cast<std::size_t>(n_tokens) * m,
                         stream);
      }
      return;
    }
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
  if (softcap > 0.0F) {
    kernels::SoftCap(y, softcap, static_cast<std::size_t>(n_tokens) * m, stream);
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

/// Lossless Q4_0 -> Q8_0 pre-conversion of dense per-layer projections once at
/// load (~1.67 GiB for Gemma 4 26B-A4B), eliminating 210 per-chunk
/// DequantSlabQ8 launches and 3.5 GB of slab traffic during prefill.
[[nodiscard]] bool PreconvertDenseQ8(ModelWeights& w, DeviceBuffers& buffers,
                                     hipStream_t stream, std::string* error) {
  std::vector<TensorRef*> refs;
  std::size_t total = 0;
  for (auto& l : w.layers) {
    for (TensorRef* t : {&l.wq, &l.wk, &l.wv, &l.wo, &l.ffn_gate, &l.ffn_up,
                         &l.ffn_down}) {
      if (t->empty() || t->q8 != nullptr || t->experts != 1 || t->rows < 256 ||
          (t->cols % 32) != 0) {
        continue;
      }
      if (t->type == core::GgmlType::kQ8_0) {
        t->q8 = t->device != nullptr ? t->device : t->data;
        continue;
      }
      if (t->type == core::GgmlType::kQ4_0) {
        refs.push_back(t);
        const std::size_t bytes =
            static_cast<std::size_t>(t->rows) * (t->cols / 32) * 34;
        total += (bytes + 255) / 256 * 256;
      }
    }
  }
  if (refs.empty()) return true;
  void* arena = nullptr;
  if (!Check(hipMalloc(&arena, total), error, "q8 dense arena alloc"))
    return false;
  buffers.allocations.push_back(arena);
  std::size_t cursor = 0;
  for (auto* t : refs) {
    const std::size_t bytes =
        static_cast<std::size_t>(t->rows) * (t->cols / 32) * 34;
    auto* d = static_cast<std::uint8_t*>(arena) + cursor;
    cursor += (bytes + 255) / 256 * 256;
    TensorRef dev = *t;
    dev.data = t->device != nullptr ? t->device : t->data;
    kernels::DequantSlabQ8(dev, t->rows, d, stream);
    t->q8 = d;
  }
  if (!Check(hipStreamSynchronize(stream), error, "q8 preconvert sync"))
    return false;
  return true;
}

/// GEMMA4_F16WQ: dequantize the fused gate_up and down expert slabs to raw
/// F16 once at load (~12.7 GB VRAM for this model). The compact MoE path then
/// runs UnquantizedF16Gemm straight off these arenas with zero per-expert
/// dequant launches.
[[nodiscard]] bool ConvertExpertsF16(ModelWeights& w, DeviceBuffers& buffers,
                                     hipStream_t stream, std::string* error) {
  const std::uint32_t ff = w.config.expert_ff;
  const std::uint32_t hidden = w.config.hidden_size;
  for (auto& l : w.layers) {
    if (!l.gate_up_exps.empty() && l.gate_up_exps.experts > 0) {
      const std::size_t per =
          static_cast<std::size_t>(2) * ff * hidden * sizeof(__half);
      __half* arena = nullptr;
      if (!Check(hipMalloc(&arena, per * l.gate_up_exps.experts),
                 error, "gate_up f16 alloc"))
        return false;
      buffers.allocations.push_back(arena);
      for (std::uint32_t e = 0; e < l.gate_up_exps.experts; ++e) {
        TensorRef one = l.gate_up_exps;
        one.experts = 1;
        one.data = l.gate_up_exps.DeviceExpert(e);
        one.device = one.data;
        kernels::DequantSlab(one, static_cast<std::uint32_t>(2) * ff,
                             arena + static_cast<std::size_t>(e) * (per / sizeof(__half)),
                             stream);
      }
      l.gate_up_f16 = arena;
    }
    if (!l.down_exps.empty() && l.down_exps.experts > 0) {
      const std::size_t per =
          static_cast<std::size_t>(hidden) * ff * sizeof(__half);
      __half* arena = nullptr;
      if (!Check(hipMalloc(&arena, per * l.down_exps.experts),
                 error, "down f16 alloc"))
        return false;
      buffers.allocations.push_back(arena);
      for (std::uint32_t e = 0; e < l.down_exps.experts; ++e) {
        TensorRef one = l.down_exps;
        one.experts = 1;
        one.data = l.down_exps.DeviceExpert(e);
        one.device = one.data;
        kernels::DequantSlab(one, hidden,
                             arena + static_cast<std::size_t>(e) * (per / sizeof(__half)),
                             stream);
      }
      l.down_f16 = arena;
    }
  }
  if (!Check(hipStreamSynchronize(stream), error, "expert f16 sync"))
    return false;
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
  if (options.kv_cache_quant != "f16" && options.kv_cache_quant != "q8") {
    if (error_msg != nullptr)
      *error_msg = "kv cache quant must be f16 or q8";
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
  if (getenv("GEMMA4_F16_DENSE") != nullptr) {
    DeviceBuffers f16weights;
    if (!PredequantWeights(*model_owner->weights_, f16weights, 0,
                           &load_error)) {
      if (error_msg != nullptr)
        *error_msg = "Failed to pre-dequantize gemma4 weights: " + load_error;
      return nullptr;
    }
    f16weights.allocations.clear();
  }
  if (getenv("GEMMA4_NO_PREQ8") == nullptr) {
    DeviceBuffers q8weights;
    if (!PreconvertDenseQ8(*model_owner->weights_, q8weights, 0,
                           &load_error)) {
      if (error_msg != nullptr)
        *error_msg = "Failed to pre-convert gemma4 dense Q8 weights: " +
                     load_error;
      return nullptr;
    }
    q8weights.allocations.clear();
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
  DeviceBuffers expert_f16;
  if (getenv("GEMMA4_F16WQ") != nullptr) {
    if (!ConvertExpertsF16(*model_owner->weights_, expert_f16, 0, &load_error)) {
      return nullptr;
    }
    expert_f16.allocations.clear();  // owned by the engine lifetime
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
  // KV caches: one cache per KV-producing layer. F16 layers use kv_k/kv_v
  // (half rows, kv_stride elements per token row); blocked-Q8 SWA layers use
  // kv_bk/kv_bv (34-byte blocks, kv_row_bytes bytes per token row).
  std::vector<__half*> kv_k;
  std::vector<__half*> kv_v;
  std::vector<std::uint32_t> kv_stride;  // elements per token row
  std::vector<std::uint8_t*> kv_bk;      // blocked Q8 K (null on F16 layers)
  std::vector<std::uint8_t*> kv_bv;      // blocked Q8 V (null on F16 layers)
  std::vector<std::uint32_t> kv_row_bytes;  // blocked bytes per token row
  std::uint32_t* d_pos{nullptr};         // device-side start_pos scalar
  std::uint32_t* d_pos_win{nullptr};     // device-side window-shifted pos
  std::uint32_t capacity{0};
  std::uint32_t length{0};

  // Activations (max one prefill chunk of 2048 tokens).
  static constexpr std::uint32_t kMaxChunk = 2048;
  float* hidden{nullptr};     // [chunk][hidden]
  float* normed{nullptr};     // [chunk][hidden]
  float* normed_compact{nullptr}; // compact-routed gather target
  float* compact_gate{nullptr};   // [chunk][expert_ff] compact gate output
  float* compact_up{nullptr};     // [chunk][expert_ff] compact up output
  float* compact_down{nullptr};   // [chunk][hidden] compact down output
  float* compact_wts{nullptr}; // [chunk][used] routing weights (compact view)
  __half* compact_xhalf{nullptr}; // [compact_cap][hidden] gathered activations
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
  std::int32_t* q4_tiles{nullptr};    // grouped-MoE tile map (expert|tile<<16)
  std::int32_t* q4_identity{nullptr}; // identity rows_out map [compact_cap]
  std::uint32_t* d_moe_meta{nullptr}; // [n_tiles, q4_rows]
  float* q4_gateup{nullptr};          // [compact_cap][2*expert_ff] grouped out
  __half* q4_act{nullptr};            // [compact_cap][expert_ff] silu act
  bool q4_identity_ready{false};
  __half* x_half{nullptr};
  std::size_t compact_cap{0};
  bool compact_xhalf_zeroed{false};
  float* expert_out{nullptr};  // [chunk*experts_used][expert_ff]
  float* mlp_out{nullptr};     // [chunk][max(hidden, shared_ff)]
  float* moe_out{nullptr};     // [chunk][hidden]
  float* expert_down{nullptr}; // [chunk][hidden] routed-expert down output
  // Decode fused-MoE staging (slot-major): [chunk*used][expert_ff] and
  // [chunk*used][hidden].
  float* moe_vec_gate{nullptr};
  float* moe_vec_up{nullptr};
  float* moe_vec_down{nullptr};
  std::uint32_t logit_cap{32};
  float* logits{nullptr};      // [logit_cap][vocab]
  std::uint32_t last_chunk_n{0};  // logit rows in the last chunk
  std::uint32_t last_fwd_n{1};    // tokens in the last forward
  // MTP (Gemma 4 Assistant) speculative draft head, optional.
  std::unique_ptr<MtpDrafter> mtp;
  float* d_tgt_emb{nullptr};    // [hidden] target embedding of the last token
  float* d_draft_fb{nullptr};   // [hidden] draft feedback hidden
  int32_t* d_draft_tok{nullptr};
  float* logits_prev{nullptr};   // [vocab] pre-verification row
  fn::ArgmaxCandidate* d_argmax_scratch{nullptr};
  std::int32_t* d_argmax_out{nullptr};
  float* d_pending_h{nullptr};
  bool has_pending_cur{false};
  std::int32_t pending_cur{0};
  __half* w_slab{nullptr};     // dequant staging (largest matrix)
  __half* w_slab2{nullptr};    // second staging (up projection)
  __half* x_slab{nullptr};     // F16 activation staging
  std::uint8_t* x_q8_slab{nullptr}; // tiled Q8 activation staging
  std::size_t w_slab_elements{0};

  [[nodiscard]] bool Ensure(const ModelWeights& w, std::uint32_t capacity,
                            std::string* error) {
    this->capacity = capacity;
    d_pos = buffers.Alloc<std::uint32_t>(1, error);
    d_pos_win = buffers.Alloc<std::uint32_t>(1, error);
    if (d_pos == nullptr || d_pos_win == nullptr) {
      if (error != nullptr && error->empty())
        *error = "blocked KV position scalar allocation failed";
      return false;
    }
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
    const std::uint32_t mlp_width =
        hidden > w.config.shared_ff ? hidden : w.config.shared_ff;
    mlp_out = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * mlp_width,
                                   error);
    moe_out = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * hidden,
                                   error);
    expert_down = buffers.Alloc<float>(static_cast<std::size_t>(chunk) * hidden,
                                       error);
    moe_vec_gate = buffers.Alloc<float>(
        static_cast<std::size_t>(chunk) * w.config.experts_used *
            w.config.expert_ff,
        error);
    moe_vec_up = buffers.Alloc<float>(
        static_cast<std::size_t>(chunk) * w.config.experts_used *
            w.config.expert_ff,
        error);
    moe_vec_down = buffers.Alloc<float>(
        static_cast<std::size_t>(chunk) * w.config.experts_used * hidden,
        error);
    logit_cap =
        getenv("GEMMA4_FORCE_ALLLOGITS") != nullptr ? kMaxChunk : 32u;
    logits = buffers.Alloc<float>(
        static_cast<std::size_t>(logit_cap) * w.config.vocab_size, error);
    // Compact-routing buffers (used by the compact MoE prefill path).
    compact_cap = ::gufo::models::qwen38_flash_next::rocm::RoutedCompactRows(
        static_cast<std::size_t>(chunk) * w.config.experts_used,
        w.config.num_experts);
    routed_bounds = buffers.Alloc<std::int32_t>(w.config.num_experts + 1, error);
    routed_cursors = buffers.Alloc<std::int32_t>(w.config.num_experts, error);
    rows_token = buffers.Alloc<std::int32_t>(compact_cap, error);
    rows_slot = buffers.Alloc<std::int32_t>(compact_cap, error);
    q4_tiles = buffers.Alloc<std::int32_t>(compact_cap, error);
    q4_identity = buffers.Alloc<std::int32_t>(compact_cap, error);
    d_moe_meta = buffers.Alloc<std::uint32_t>(2, error);
    if (q4_identity != nullptr) {
      std::vector<std::int32_t> ident(compact_cap);
      for (std::size_t i = 0; i < ident.size(); ++i)
        ident[i] = static_cast<std::int32_t>(i);
      (void)hipMemcpy(q4_identity, ident.data(),
                      ident.size() * sizeof(std::int32_t),
                      hipMemcpyHostToDevice);
      q4_identity_ready = true;
    }
    q4_gateup = buffers.Alloc<float>(
        compact_cap * 2 * w.config.expert_ff, error);
    q4_act = buffers.Alloc<__half>(compact_cap * w.config.expert_ff, error);
    normed_compact = buffers.Alloc<float>(compact_cap * w.config.hidden_size,
                                          error);
    x_half = buffers.Alloc<__half>(
        static_cast<std::size_t>(chunk) * w.config.hidden_size, error);
    compact_gate = buffers.Alloc<float>(compact_cap * w.config.expert_ff,
                                        error);
    compact_up = buffers.Alloc<float>(compact_cap * w.config.expert_ff,
                                       error);
    compact_down = buffers.Alloc<float>(compact_cap * hidden, error);
    compact_wts = buffers.Alloc<float>(
        static_cast<std::size_t>(chunk) * w.config.experts_used, error);
    compact_xhalf = buffers.Alloc<__half>(
        compact_cap * w.config.hidden_size, error);
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
    x_q8_slab = buffers.Alloc<std::uint8_t>(
        fn::Q8TiledBytes(chunk, xwidth), error);
    d_tgt_emb = buffers.Alloc<float>(hidden, error);
    d_draft_fb = buffers.Alloc<float>(hidden, error);
    d_draft_tok = buffers.Alloc<std::int32_t>(16, error);
    logits_prev = buffers.Alloc<float>(w.config.vocab_size, error);
    d_argmax_scratch =
        buffers.Alloc<fn::ArgmaxCandidate>(16 * fn::kArgmaxParts, error);
    d_argmax_out = buffers.Alloc<std::int32_t>(16, error);
    d_pending_h = buffers.Alloc<float>(hidden, error);
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
  impl.kv_bk.assign(w.config.num_layers, nullptr);
  impl.kv_bv.assign(w.config.num_layers, nullptr);
  impl.kv_row_bytes.assign(w.config.num_layers, 0);
  for (std::uint32_t il = 0; il < w.config.num_layers; ++il) {
    if (!w.config.LayerHasKv(il))
      continue;
    // Every non-SWA layer owns its own KV cache (llama.cpp: only SWA layers
    // share the preceding producer's cache).
    if (w.config.KvProducingLayer(il) != il) {
    }
    const std::uint32_t head_dim_il = w.config.HeadDim(il);
    const std::uint32_t kv_heads_il = w.config.KvHeads(il);
    const std::uint32_t dim = kv_heads_il * head_dim_il;
    impl.kv_stride[il] = dim;
    // Shared blocked Q8 covers SWA layers only (d == 256, plain theta RoPE,
    // self-produced KV). Full layers (d == 512, proportional rope_freqs)
    // stay F16: the shared write kernel rejects d > 256 and the shared read
    // kernel is d == 256-only. Anything else under q8 falls back loudly,
    // never silently.
    const bool layer_blocked = options_.kv_cache_quant == "q8" &&
                               kv_heads_il > 0 && head_dim_il == 256 &&
                               w.config.KvProducingLayer(il) == il;
    if (options_.kv_cache_quant == "q8" && !layer_blocked &&
        w.config.LayerHasKv(il)) {
      std::fprintf(stderr,
                   "[gemma4] kv_cache_quant=q8: layer %u not blocked "
                   "(head_dim=%u kv_heads=%u producer=%u), staying F16\n",
                   il, head_dim_il, kv_heads_il,
                   w.config.KvProducingLayer(il));
    }
    if (layer_blocked) {
      const std::size_t row_bytes =
          (static_cast<std::size_t>(head_dim_il) + 31) / 32 * 34 * kv_heads_il;
      impl.kv_row_bytes[il] = static_cast<std::uint32_t>(row_bytes);
      const std::size_t kv_bytes =
          static_cast<std::size_t>(options_.max_context) * row_bytes;
      std::uint8_t* kvk8 = nullptr;
      std::uint8_t* kvv8 = nullptr;
      if (hipMalloc(&kvk8, kv_bytes) != hipSuccess ||
          hipMalloc(&kvv8, kv_bytes) != hipSuccess) {
        if (error_msg != nullptr)
          *error_msg = "blocked KV cache allocation failed (layer " +
                       std::to_string(il) + ")";
        return nullptr;
      }
      hipMemset(kvk8, 0, kv_bytes);
      hipMemset(kvv8, 0, kv_bytes);
      impl.buffers.allocations.push_back(kvk8);
      impl.buffers.allocations.push_back(kvv8);
      impl.kv_bk[il] = kvk8;
      impl.kv_bv[il] = kvv8;
      if (impl.kv_bk[il] == nullptr || impl.kv_bv[il] == nullptr) {
        if (error_msg != nullptr && error_msg->empty())
          *error_msg = "blocked KV cache allocation failed (layer " +
                       std::to_string(il) + ")";
        return nullptr;
      }
      continue;
    }
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
  if (impl_->d_argmax_scratch != nullptr && impl_->d_argmax_out != nullptr &&
      getenv("GEMMA4_HOST_ARGMAX") == nullptr) {
    fn::Argmax(logits.data(), impl_->d_argmax_scratch, impl_->d_argmax_out, 1u,
               static_cast<std::uint32_t>(logits.size()), 0);
    std::int32_t best = 0;
    (void)hipMemcpy(&best, impl_->d_argmax_out, sizeof(std::int32_t),
                    hipMemcpyDeviceToHost);
    return best;
  }
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
                __half* x_slab, hipStream_t stream, float softcap = 0.0F,
                std::uint8_t* x_q8_slab = nullptr, bool reuse_x = false) {
  F32Gemm(w, x, y, n_tokens, static_cast<std::uint32_t>(m_rows),
          static_cast<std::uint32_t>(w.cols), stream, slab, x_slab, softcap,
          x_q8_slab, reuse_x);
}

}  // namespace

// L0 correctness probe: prints a buffer's sum/maxabs under DEBUG_EMBED so
// A/B runs can be compared bit-for-bit (used by the MoE correctness protocol).
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

namespace {
}  // namespace

bool Session::Forward(std::span<const std::int32_t> tokens,
                      std::uint32_t start_pos, std::string* error_msg,
                      bool all_logits) {
  const auto& w = *model_->weights_;
  const auto& c = w.config;
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  hipStream_t stream = 0;
  impl_->has_pending_cur = false;
  // Device-side position scalar for the shared blocked kernels (blocked
  // writes and direct reads index the cache with *start_pos on-device).
  if (model_->options_.kv_cache_quant == "q8") {
    (void)hipMemcpyAsync(impl_->d_pos, &start_pos, sizeof(start_pos),
                         hipMemcpyHostToDevice, stream);
  }
  std::vector<std::int32_t> host_tokens(tokens.begin(), tokens.end());
  if (getenv("GEMMA4_DEBUG_EMBED"))
    std::fprintf(stderr, "[gemma4] fwd n=%u pos=%u tok0=%d\n", n, start_pos,
                 host_tokens[0]);

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

  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const auto& l = w.layers[il];
    const std::uint32_t kv_layer = c.KvProducingLayer(il);
    const std::uint32_t head_dim = c.HeadDim(il);
    const std::uint32_t kv_heads = c.KvHeads(kv_layer);
    const std::uint32_t qdim = c.num_heads * head_dim;
    const std::uint32_t kvdim = kv_heads * head_dim;
    // Shared blocked Q8 covers SWA layers only (d == 256, plain theta RoPE,
    // self-produced KV). Full layers (d == 512, proportional rope) stay F16;
    // see the allocation fallback above.
    const bool use_blocked =
        model_->options_.kv_cache_quant == "q8" && l.has_kv && kv_heads > 0 &&
        head_dim == 256 && kv_layer == il;

    // Attention block.
    const bool debug_embed = getenv("GEMMA4_DEBUG_EMBED") != nullptr;
    const bool allow_fused_decode =
        !debug_embed && getenv("GEMMA4_NO_FUSED_DECODE") == nullptr;

    kernels::RmsNorm(impl_->hidden,
                     static_cast<const float*>(l.attn_norm.device),
                     impl_->normed, n, hidden, c.rms_eps, stream);

    const bool use_fused_qkv_gemv =
        allow_fused_decode && n <= 8 &&
        getenv("GEMMA4_NO_FUSED_GEMV") == nullptr && l.has_kv && kv_heads > 0 &&
        l.wq.f16 == nullptr && l.wk.f16 == nullptr &&
        (!l.has_v_proj || l.wv.f16 == nullptr);
    if (use_fused_qkv_gemv) {
      const void* q_dev = l.wq.device != nullptr ? l.wq.device : l.wq.data;
      const void* k_dev = l.wk.device != nullptr ? l.wk.device : l.wk.data;
      const void* v_dev =
          l.has_v_proj ? (l.wv.device != nullptr ? l.wv.device : l.wv.data)
                       : nullptr;
      kernels::GemvQkv(
          q_dev, static_cast<int>(l.wq.type), l.wq.RowBytes(), qdim, k_dev,
          static_cast<int>(l.wk.type), l.wk.RowBytes(), v_dev,
          l.has_v_proj ? static_cast<int>(l.wv.type) : 0,
          l.has_v_proj ? l.wv.RowBytes() : 0, kvdim, hidden, impl_->normed,
          impl_->qbuf, impl_->kbuf, impl_->vbuf, n, stream);
    } else {
      const bool can_reuse_qkv_x = (n > 8u && qdim >= 256u && kvdim >= 256u);
      StagedGemm(l.wq, qdim, impl_->normed, impl_->qbuf, n, impl_->w_slab,
                 impl_->x_slab, stream, 0.0F, impl_->x_q8_slab, false);
      if (l.has_kv && kv_heads > 0) {
        StagedGemm(l.wk, kvdim, impl_->normed, impl_->kbuf, n, impl_->w_slab,
                   impl_->x_slab, stream, 0.0F, impl_->x_q8_slab,
                   can_reuse_qkv_x);
        if (l.has_v_proj) {
          StagedGemm(l.wv, kvdim, impl_->normed, impl_->vbuf, n, impl_->w_slab,
                     impl_->x_slab, stream, 0.0F, impl_->x_q8_slab,
                     can_reuse_qkv_x);
        }
      }
    }

    if (debug_embed && il == 0 && n <= 4) {
      hipDeviceSynchronize();
      auto hsum = [&](const float* d, std::size_t cnt) {
        std::vector<float> h(cnt);
        (void)hipMemcpy(h.data(), d, cnt * sizeof(float),
                        hipMemcpyDeviceToHost);
        double ss = 0;
        for (float v : h) ss += (double)v;
        return ss;
      };
      std::fprintf(stderr, "[gemma4] L0 n=%u norm0=%.6f qbuf0=%.6f\n", n,
                   hsum(impl_->normed, hidden),
                   hsum(impl_->qbuf, static_cast<std::size_t>(qdim)));
    }

    const std::uint32_t rot_dim = c.RotaryDim(il);
    const float* rope_ff =
        (!c.LayerIsSwa(il) && !l.rope_freqs.empty())
            ? static_cast<const float*>(l.rope_freqs.device)
            : nullptr;
    if (!use_blocked && allow_fused_decode && head_dim <= 512 &&
        rot_dim <= head_dim) {
      const bool write_kv = (l.has_kv && kv_heads > 0);
      kernels::PrepAttnQkv(
          impl_->qbuf, impl_->kbuf, impl_->vbuf,
          static_cast<const float*>(l.attn_q_norm.device),
          write_kv ? static_cast<const float*>(l.attn_k_norm.device) : nullptr,
          write_kv ? static_cast<const float*>(l.attn_v_norm.device) : nullptr,
          write_kv ? impl_->kv_k[kv_layer] : nullptr,
          write_kv ? impl_->kv_v[kv_layer] : nullptr, n, c.num_heads, kv_heads,
          write_kv, l.has_v_proj, head_dim, rot_dim, start_pos, c.RopeTheta(il),
          rope_ff, c.rms_eps, stream);
    } else {
      // QK norm (per-head gamma over head_dim).
      kernels::RmsNorm(impl_->qbuf,
                       static_cast<const float*>(l.attn_q_norm.device),
                       impl_->qbuf, n * c.num_heads, head_dim, c.rms_eps,
                       stream);
      if (rope_ff != nullptr) {
        if (debug_embed) {
          std::vector<float> ff(256);
          (void)hipMemcpy(ff.data(), l.rope_freqs.device, 256 * sizeof(float),
                          hipMemcpyDeviceToHost);
          std::fprintf(stderr,
                       "[gemma4] L%u ff[0]=%f ff[63]=%f ff[64]=%f ff[127]=%f "
                       "ff[128]=%f ff[255]=%f theta=%f pos=%u\n",
                       il, ff[0], ff[63], ff[64], ff[127], ff[128], ff[255],
                       c.RopeTheta(il), start_pos);
        }
        kernels::RopeFrequenciesMulti(impl_->qbuf, n, c.num_heads, head_dim,
                                      rot_dim, start_pos, c.RopeTheta(il),
                                      rope_ff, stream);
      } else {
        kernels::Rope(impl_->qbuf, n, c.num_heads, head_dim, rot_dim, start_pos,
                      c.RopeTheta(il), stream);
      }

      if (l.has_kv && kv_heads > 0) {
        kernels::RmsNorm(impl_->kbuf,
                         static_cast<const float*>(l.attn_k_norm.device),
                         impl_->kbuf, n * kv_heads, head_dim, c.rms_eps, stream);
        // No v_proj on this layer (llama: Vcur=Kcur pre-rope). Capture BEFORE k rope.
        if (!l.has_v_proj) {
          (void)hipMemcpyAsync(impl_->vbuf, impl_->kbuf,
                               static_cast<std::size_t>(n) * kvdim *
                                   sizeof(float),
                               hipMemcpyDeviceToDevice, stream);
        }
        if (rope_ff != nullptr) {
          kernels::RopeFrequenciesMulti(impl_->kbuf, n, kv_heads, head_dim,
                                        rot_dim, start_pos, c.RopeTheta(il),
                                        rope_ff, stream);
        } else {
          kernels::Rope(impl_->kbuf, n, kv_heads, head_dim, rot_dim, start_pos,
                        c.RopeTheta(il), stream);
        }
        // V RMSNorm (Gemma 4 normalizes values too).
        kernels::RmsNorm(impl_->vbuf,
                         static_cast<const float*>(l.attn_v_norm.device),
                         impl_->vbuf, n * kv_heads, head_dim, c.rms_eps,
                         stream);
        // Store KV at positions [pos, pos+n).
        if (use_blocked) {
          // The fused writer is disabled for blocked layers, so this is the
          // only writer: the final (post-norm, post-RoPE) rows quantize
          // straight into the shared blocked layout (per-32-block scale).
          kernels::StoreBlockedKv(impl_->kbuf, impl_->vbuf,
                                  impl_->kv_bk[kv_layer],
                                  impl_->kv_bv[kv_layer], n, kv_heads, head_dim,
                                  impl_->d_pos, stream);
        } else {
          kernels::StoreKv(impl_->kbuf, impl_->kv_k[kv_layer], n, kvdim,
                           start_pos, stream);
          kernels::StoreKv(impl_->vbuf, impl_->kv_v[kv_layer], n, kvdim,
                           start_pos, stream);
        }
      }
    }
    if (use_blocked) {
      // Blocked direct read (fused dequant, score scale 1.0, no F16
      // staging). Same sliding-window pointer shift as Attention: keys at or
      // below start_pos - window fall out of the window, so advance the
      // cache base and rebase the position, keeping dense-causal semantics.
      const std::uint8_t* k2 = impl_->kv_bk[kv_layer];
      const std::uint8_t* v2 = impl_->kv_bv[kv_layer];
      const std::uint32_t* sp = impl_->d_pos;
      const std::uint32_t window = c.LayerIsSwa(il) ? c.sliding_window : 0;
      std::uint32_t shifted = 0;
      if (window != 0 && start_pos + 1 > window) {
        const std::uint32_t lo = (start_pos + 1 - window) & ~3u;
        if (lo != 0 && lo <= start_pos) {
          shifted = start_pos - lo;
          if (hipMemcpy(impl_->d_pos_win, &shifted, sizeof(shifted),
                        hipMemcpyHostToDevice) == hipSuccess) {
            const std::size_t off =
                static_cast<std::size_t>(lo) * impl_->kv_row_bytes[kv_layer];
            k2 += off;
            v2 += off;
            sp = impl_->d_pos_win;
          }
        }
      }
      kernels::AttentionBlocked(impl_->qbuf, k2, v2, impl_->attn_out, n, sp,
                                c.num_heads, kv_heads, head_dim, stream);
    } else {
      kernels::Attention(impl_->qbuf, impl_->kv_k[kv_layer],
                         impl_->kv_v[kv_layer], impl_->attn_out, n, start_pos,
                         c.num_heads, kv_heads, head_dim,
                         impl_->kv_stride[kv_layer],
                         c.LayerIsSwa(il) ? c.sliding_window : 0, stream);
    }

    // wo projection (write into idle mlp_out to avoid x == y staging copy).
    if (debug_embed && il == 0 && n <= 4) {
      hipDeviceSynchronize();
      std::vector<float> ah(hidden);
      (void)hipMemcpy(ah.data(), impl_->attn_out, hidden * sizeof(float),
                      hipMemcpyDeviceToHost);
      double asum = 0;
      for (float v : ah) asum += (double)v;
      std::fprintf(stderr, "[gemma4] L0 n=%u attn_raw0=%.6f\n", n, asum);
    }
    StagedGemm(l.wo, hidden, impl_->attn_out, impl_->mlp_out, n,
               impl_->w_slab, impl_->x_slab, stream, 0.0F, impl_->x_q8_slab,
               false);
    if (debug_embed && il == 0 && n == 1) {
      hipDeviceSynchronize();
      std::vector<float> ah2(static_cast<std::size_t>(n) * hidden);
      (void)hipMemcpy(ah2.data(), impl_->mlp_out, ah2.size() * sizeof(float),
                      hipMemcpyDeviceToHost);
      double asum2 = 0;
      for (float v : ah2) asum2 += (double)v;
      std::fprintf(stderr, "[gemma4] L0decode attn_post_wo=%.6f\n", asum2);
    }

    if (debug_embed && il < 3) {
      std::vector<float> probe(static_cast<std::size_t>(n) * hidden);
      (void)hipMemcpy(probe.data(), impl_->mlp_out, probe.size() * sizeof(float),
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
    kernels::RmsNormAdd(impl_->mlp_out,
                        static_cast<const float*>(l.attn_post_norm.device),
                        impl_->hidden, n, hidden, c.rms_eps, 1.0F, stream);

    // FFN block (llama.cpp gemma4.cpp parity).
    if (!l.IsMoe()) {
      kernels::RmsNorm(impl_->hidden,
                       static_cast<const float*>(l.ffn_norm.device),
                       impl_->normed, n, hidden, c.rms_eps, stream);
      if (n <= 8 && l.ffn_gate.type == l.ffn_up.type &&
          l.ffn_gate.f16 == nullptr && l.ffn_up.f16 == nullptr &&
          getenv("GEMMA4_NO_FUSED_GEMV") == nullptr &&
          getenv("GEMMA4_NO_WAVE32_GEMV") == nullptr) {
        const void* g_dev =
            l.ffn_gate.device != nullptr ? l.ffn_gate.device : l.ffn_gate.data;
        const void* u_dev =
            l.ffn_up.device != nullptr ? l.ffn_up.device : l.ffn_up.data;
        kernels::GemvGateUpSwiglu(g_dev, u_dev,
                                  static_cast<int>(l.ffn_gate.type),
                                  l.ffn_gate.RowBytes(), hidden, impl_->normed,
                                  impl_->mlp_out, n, c.shared_ff, stream);
      } else {
        F32Gemm(l.ffn_gate, impl_->normed, impl_->mlp_out, n, c.shared_ff,
                hidden, stream, impl_->w_slab, impl_->x_slab, 0.0F,
                impl_->x_q8_slab, false);
        F32Gemm(l.ffn_up, impl_->normed, impl_->expert_out, n, c.shared_ff,
                hidden, stream, impl_->w_slab2, impl_->x_slab, 0.0F,
                impl_->x_q8_slab, (n > 8u && c.shared_ff >= 256u));
        kernels::GeluTanhSwiglu(impl_->mlp_out, impl_->expert_out,
                                static_cast<std::size_t>(n) * c.shared_ff,
                                stream);
      }
      F32Gemm(l.ffn_down, impl_->mlp_out, impl_->moe_out, n, hidden,
              c.shared_ff, stream, impl_->w_slab, impl_->x_slab, 0.0F,
              impl_->x_q8_slab, false);
      kernels::RmsNormAdd(
          impl_->moe_out, static_cast<const float*>(l.ffn_post_norm.device),
          impl_->hidden, n, hidden, c.rms_eps,
          l.has_out_scale ? l.out_scale_val : 1.0F, stream);
    } else {
      const bool decode_vec_moe =
          (n <= 8 && !l.gate_up_exps.empty() && !l.down_exps.empty() &&
           l.down_exps.type == core::GgmlType::kQ4_0 &&
           getenv("GEMMA4_NO_DECODE_VEC") == nullptr);
      const bool fast_q4d_prefill =
          (!decode_vec_moe && n > 1 && allow_fused_decode &&
           impl_->rows_token != nullptr && impl_->d_moe_meta != nullptr &&
           getenv("GEMMA4_NO_COMPACT") == nullptr &&
           getenv("GEMMA4_NO_Q4DIRECT") == nullptr &&
           getenv("GEMMA4_NO_FAST_Q4D") == nullptr &&
           !l.gate_up_exps.empty() && l.gate_up_exps.device != nullptr &&
           l.gate_up_exps.type == core::GgmlType::kQ4_0 &&
           !l.down_exps.empty() && l.down_exps.device != nullptr &&
           l.down_exps.type == core::GgmlType::kQ4_0);
      const bool fused_moe_norms =
          (decode_vec_moe || fast_q4d_prefill) && allow_fused_decode;

      if (fused_moe_norms) {
        kernels::FfnTripleNorm(
            impl_->hidden, static_cast<const float*>(l.ffn_norm.device),
            static_cast<const float*>(l.ffn_pre_norm_2.device),
            static_cast<const float*>(l.router_scale.device), impl_->normed,
            impl_->expert_down, impl_->router_tmp, n, hidden, c.rms_eps,
            stream);
      } else {
        kernels::RmsNorm(impl_->hidden,
                         static_cast<const float*>(l.ffn_norm.device),
                         impl_->normed, n, hidden, c.rms_eps, stream);
      }

      if (n <= 8 && l.ffn_gate.type == l.ffn_up.type &&
          l.ffn_gate.f16 == nullptr && l.ffn_up.f16 == nullptr &&
          getenv("GEMMA4_NO_FUSED_GEMV") == nullptr &&
          getenv("GEMMA4_NO_WAVE32_GEMV") == nullptr) {
        const void* g_dev =
            l.ffn_gate.device != nullptr ? l.ffn_gate.device : l.ffn_gate.data;
        const void* u_dev =
            l.ffn_up.device != nullptr ? l.ffn_up.device : l.ffn_up.data;
        kernels::GemvGateUpSwiglu(g_dev, u_dev,
                                  static_cast<int>(l.ffn_gate.type),
                                  l.ffn_gate.RowBytes(), hidden, impl_->normed,
                                  impl_->mlp_out, n, c.shared_ff, stream);
      } else {
        F32Gemm(l.ffn_gate, impl_->normed, impl_->mlp_out, n, c.shared_ff,
                hidden, stream, impl_->w_slab, impl_->x_slab, 0.0F,
                impl_->x_q8_slab, false);
        F32Gemm(l.ffn_up, impl_->normed, impl_->expert_out, n, c.shared_ff,
                hidden, stream, impl_->w_slab2, impl_->x_slab, 0.0F,
                impl_->x_q8_slab, (n > 8u && c.shared_ff >= 256u));
        kernels::GeluTanhSwiglu(impl_->mlp_out, impl_->expert_out,
                                static_cast<std::size_t>(n) * c.shared_ff,
                                stream);
      }
      F32Gemm(l.ffn_down, impl_->mlp_out, impl_->expert_out, n, hidden,
              c.shared_ff, stream, impl_->w_slab, impl_->x_slab, 0.0F,
              impl_->x_q8_slab, false);

      if (!fused_moe_norms) {
        kernels::RmsNorm(impl_->expert_out,
                         static_cast<const float*>(l.ffn_post_norm_1.device),
                         impl_->moe_out, n, hidden, c.rms_eps, stream);
        // moe_out holds the shared-expert contribution; mlp_out is reused.
        kernels::RmsNorm(impl_->hidden,
                         static_cast<const float*>(l.ffn_pre_norm_2.device),
                         impl_->normed, n, hidden, c.rms_eps, stream);
        // Router: tmp = rmsnorm(hidden) * (1/sqrt(hidden)) * router_scale.
        kernels::RouterInput(impl_->hidden,
                             static_cast<const float*>(l.router_scale.device),
                             impl_->router_tmp, n, hidden, c.rms_eps, stream);
      }
      StagedGemm(l.router, c.num_experts, impl_->router_tmp,
                 impl_->router_logits, n, impl_->w_slab, impl_->x_slab,
                 stream);
      fn::RouterTopK(impl_->router_logits, c.num_experts, impl_->expert_ids,
                     impl_->expert_weights, n, c.num_experts, c.experts_used,
                     stream);
      if (debug_embed && il == 0 && n <= 4) {
        hipDeviceSynchronize();
        std::vector<std::int32_t> eids(static_cast<std::size_t>(n) * c.experts_used);
        std::vector<float> ewts(static_cast<std::size_t>(n) * c.experts_used);
        (void)hipMemcpy(eids.data(), impl_->expert_ids,
                        eids.size() * sizeof(std::int32_t), hipMemcpyDeviceToHost);
        (void)hipMemcpy(ewts.data(), impl_->expert_weights,
                        ewts.size() * sizeof(float), hipMemcpyDeviceToHost);
        std::fprintf(stderr, "[gemma4] L0 n=%u experts0=", n);
        for (std::uint32_t k = 0; k < c.experts_used; ++k)
          std::fprintf(stderr, "%d:%.3f ", eids[k], ewts[k]);
        std::fprintf(stderr, "\n");
      }
      if (debug_embed && il == 0) {
        hipDeviceSynchronize();
        std::vector<float> rl(static_cast<std::size_t>(n) * c.num_experts);
        (void)hipMemcpy(rl.data(), impl_->router_logits,
                        rl.size() * sizeof(float), hipMemcpyDeviceToHost);
        double rs = 0; float rm = 0;
        for (float v : rl) {
          rs += (double)v;
          float a = v < 0 ? -v : v;
          if (a > rm) rm = a;
        }
        std::fprintf(stderr, "[gemma4] L0 router sum=%.4f maxabs=%.4f\n", rs,
                     rm);
      }
      // Decode & small-batch (MTP verify) fast path (n <= 8): device-side
      // Wave32 MoE over all n * experts_used slots with zero host syncs.
      if (decode_vec_moe) {
        const std::uint32_t used = c.experts_used;
        const std::uint32_t slots = n * used;
        const std::uint32_t ff = c.expert_ff;
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
        const float* moe_x =
            fused_moe_norms ? impl_->expert_down : impl_->normed;
        kernels::MoeVecGateUpSwiglu(gu_dev, l.gate_up_exps.RowBytes(),
                                    gu_stride, moe_x, impl_->expert_ids,
                                    impl_->moe_vec_gate, slots, used, ff,
                                    hidden, stream);
        kernels::GemvQSlot(dn_dev, l.down_exps.RowBytes(), dn_stride,
                           impl_->moe_vec_gate, impl_->expert_ids,
                           impl_->moe_vec_down, slots, hidden, ff, stream);
        if (fused_moe_norms) {
          kernels::MoeCombinePostNormResidual(
              impl_->moe_vec_down, impl_->expert_ids, impl_->expert_weights,
              l.down_exps_scale.empty()
                  ? nullptr
                  : static_cast<const float*>(l.down_exps_scale.device),
              impl_->expert_out,
              static_cast<const float*>(l.ffn_post_norm_1.device),
              static_cast<const float*>(l.ffn_post_norm_2.device),
              static_cast<const float*>(l.ffn_post_norm.device), impl_->hidden,
              n, used, hidden, c.rms_eps,
              l.has_out_scale ? l.out_scale_val : 1.0F, stream);
        } else {
          kernels::MoeSlotEpilogueOverwrite(
              impl_->moe_vec_down, impl_->expert_ids, impl_->expert_weights,
              l.down_exps_scale.empty()
                  ? nullptr
                  : static_cast<const float*>(l.down_exps_scale.device),
              impl_->attn_out, n, used, hidden, stream);
        }
      } else if (fast_q4d_prefill) {
        namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
        const std::uint32_t used = c.experts_used;
        const std::uint32_t ff = c.expert_ff;
        std::uint32_t tile_rows = 64u;
        if (const char* tr_env = getenv("GEMMA4_MOE_TILE")) {
          const unsigned long v = std::strtoul(tr_env, nullptr, 10);
          if (v == 16ul || v == 48ul || v == 64ul)
            tile_rows = static_cast<std::uint32_t>(v);
        }
        q38::ExpertCounts(impl_->expert_ids, impl_->expert_counts, n,
                          c.num_experts, used, stream);
        q38::RoutedCompact(impl_->expert_ids, impl_->expert_counts,
                           impl_->routed_bounds, impl_->routed_cursors,
                           impl_->rows_token, impl_->rows_slot, n, used,
                           c.num_experts, stream);
        kernels::BuildRoutedTiles(impl_->routed_bounds, impl_->q4_tiles,
                                  impl_->d_moe_meta, c.num_experts, tile_rows,
                                  stream);
        std::uint32_t h_meta[2] = {0u, 0u};
        (void)hipMemcpy(h_meta, impl_->d_moe_meta, sizeof(h_meta),
                        hipMemcpyDeviceToHost);
        const std::uint32_t n_tiles = h_meta[0];
        const std::uint32_t q4_rows = h_meta[1];
        if (n_tiles > 0u && q4_rows > 0u) {
          kernels::F32ToF16(impl_->expert_down, impl_->x_half,
                            static_cast<std::size_t>(n) * hidden, stream);
          q38::RoutedF16Gemm(l.gate_up_exps.device, q38::WeightType::kQ4_0,
                             impl_->x_half, impl_->q4_tiles, n_tiles, tile_rows,
                             impl_->routed_bounds, impl_->rows_token,
                             impl_->q4_identity, nullptr, impl_->q4_gateup,
                             nullptr, 2 * ff, hidden, stream);
          kernels::SwigluSplitF16(impl_->q4_gateup, impl_->q4_act, q4_rows, ff,
                                  stream);
          q38::RoutedF16Gemm(l.down_exps.device, q38::WeightType::kQ4_0,
                             impl_->q4_act, impl_->q4_tiles, n_tiles, tile_rows,
                             impl_->routed_bounds, impl_->q4_identity,
                             impl_->rows_slot, nullptr, impl_->moe_vec_down,
                             nullptr, hidden, ff, stream);
        } else {
          (void)hipMemsetAsync(impl_->moe_vec_down, 0,
                               static_cast<std::size_t>(n) * used * hidden *
                                   sizeof(float),
                               stream);
        }
        kernels::MoeCombinePostNormResidual(
            impl_->moe_vec_down, impl_->expert_ids, impl_->expert_weights,
            l.down_exps_scale.empty()
                ? nullptr
                : static_cast<const float*>(l.down_exps_scale.device),
            impl_->expert_out,
            static_cast<const float*>(l.ffn_post_norm_1.device),
            static_cast<const float*>(l.ffn_post_norm_2.device),
            static_cast<const float*>(l.ffn_post_norm.device), impl_->hidden, n,
            used, hidden, c.rms_eps, l.has_out_scale ? l.out_scale_val : 1.0F,
            stream);
      } else {
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
      bool compact_ok = false;
      std::vector<std::int32_t> bounds_host;
      if (getenv("GEMMA4_NO_COMPACT") == nullptr && n > 1 &&
          impl_->rows_token != nullptr) {
        namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
        q38::ExpertCounts(impl_->expert_ids, impl_->expert_counts, n,
                          c.num_experts, c.experts_used, stream);
        q38::RoutedCompact(impl_->expert_ids, impl_->expert_counts,
                           impl_->routed_bounds, impl_->routed_cursors,
                           impl_->rows_token, impl_->rows_slot, n,
                           c.experts_used, c.num_experts, stream);
        // RoutedPadBoundsKernel only clears padding rows; rows beyond the
        // last expert's bucket keep stale assignments from earlier (larger)
        // chunks. Clear the tail so scatter never reads a stale token index.
        {
          std::int32_t last_bound = 0;
          hipMemcpy(&last_bound, impl_->routed_bounds + c.num_experts,
                    sizeof(std::int32_t), hipMemcpyDeviceToHost);
          const std::size_t cap =
              ::gufo::models::qwen38_flash_next::rocm::RoutedCompactRows(
                  static_cast<std::size_t>(n) * c.experts_used, c.num_experts);
          if (static_cast<std::size_t>(last_bound) < cap) {
            hipMemsetAsync(impl_->rows_token + last_bound, 0xFF,
                           (cap - last_bound) * sizeof(std::int32_t), stream);
            hipMemsetAsync(impl_->rows_slot + last_bound, 0xFF,
                           (cap - last_bound) * sizeof(std::int32_t), stream);
          }
        }
        bounds_host.assign(c.num_experts + 1, 0);
        hipMemcpy(bounds_host.data(), impl_->routed_bounds,
                  (c.num_experts + 1) * sizeof(std::int32_t),
                  hipMemcpyDeviceToHost);
        // The compact scatter indexes routing weights by rows_slot, a
        // chunk-local slot index. expert_weights is a per-layer buffer that
        // later layers overwrite, so the compact view keeps its own copy for
        // the lifetime of this chunk's expert loop.
        hipMemcpyAsync(impl_->compact_wts, impl_->expert_weights,
                       static_cast<std::size_t>(n) * c.experts_used *
                           sizeof(float),
                       hipMemcpyDeviceToDevice, stream);
        compact_ok = true;
      }
      // Q4-direct grouped MoE (GEMMA4_Q4DIRECT): the fused gate/up slab goes
      // through ONE RoutedF16Gemm (m = 2*ff, so the kernel's expert stride
      // matches the fused layout) reading Q4_0 straight off the GGUF slabs
      // with in-LDS dequant; a fused split+silu writes the down input; ONE
      // more RoutedF16Gemm does down for every expert. No compact gather,
      // no per-expert dequant/GEMM launches, half the weight traffic of the
      // F16-arena path.
      // Q4-direct grouped MoE is the default: it reads Q4_0 straight off the
      // GGUF slabs (in-LDS dequant) instead of materialising a 12.7 GB F16
      // expert arena, halving weight traffic. GEMMA4_NO_Q4DIRECT opts out.
      const bool q4d =
          compact_ok && !l.gate_up_exps.empty() &&
          l.gate_up_exps.device != nullptr && !l.down_exps.empty() &&
          l.down_exps.device != nullptr &&
          getenv("GEMMA4_NO_Q4DIRECT") == nullptr;
      if (q4d) {
        namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
        // Full-token F16 activations; the grouped kernel gathers rows itself
        // through rows_token, so CompactGatherF16 is skipped below.
        kernels::F32ToF16(impl_->normed, impl_->x_half,
                          static_cast<std::size_t>(n) * hidden, stream);
        if (!impl_->q4_identity_ready) {
          std::vector<std::int32_t> ident(impl_->compact_cap);
          for (std::size_t i = 0; i < ident.size(); ++i)
            ident[i] = static_cast<std::int32_t>(i);
          hipMemcpyAsync(impl_->q4_identity, ident.data(),
                         ident.size() * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, stream);
          impl_->q4_identity_ready = true;
        }
        // One (expert | tile<<16) entry per 16-row bucket tile.
        std::vector<std::int32_t> tiles;
        tiles.reserve(impl_->compact_cap);
        for (std::uint32_t e = 0; e < c.num_experts; ++e) {
          const std::uint32_t rows_e = static_cast<std::uint32_t>(
              bounds_host[e + 1] - bounds_host[e]);
          for (std::uint32_t j = 0; j < (rows_e + 15) / 16; ++j)
            tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
        }
        const std::uint32_t q4_rows =
            static_cast<std::uint32_t>(bounds_host[c.num_experts]);
        if (!tiles.empty()) {
          hipMemcpyAsync(impl_->q4_tiles, tiles.data(),
                         tiles.size() * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, stream);
          const std::uint32_t ff = c.expert_ff;
          q38::RoutedF16Gemm(l.gate_up_exps.device, q38::WeightType::kQ4_0,
                             impl_->x_half, impl_->q4_tiles,
                             static_cast<std::uint32_t>(tiles.size()), 16,
                             impl_->routed_bounds, impl_->rows_token,
                             impl_->q4_identity, nullptr, impl_->q4_gateup,
                             nullptr, 2 * ff, hidden, stream);
          kernels::SwigluSplitF16(impl_->q4_gateup, impl_->q4_act, q4_rows,
                                  ff, stream);
          q38::RoutedF16Gemm(l.down_exps.device, q38::WeightType::kQ4_0,
                             impl_->q4_act, impl_->q4_tiles,
                             static_cast<std::uint32_t>(tiles.size()), 16,
                             impl_->routed_bounds, impl_->q4_identity,
                             impl_->q4_identity, nullptr, impl_->compact_down,
                             nullptr, hidden, ff, stream);
        }
      }
      std::vector<bool> active(c.num_experts, false);
      for (std::size_t idx = 0; idx < ids.size(); ++idx) {
        if (ids[idx] >= 0 && ids[idx] < static_cast<std::int32_t>(c.num_experts))
          active[static_cast<std::size_t>(ids[idx])] = true;
      }
      float down_scales[128];
      if (!l.down_exps_scale.empty()) {
        const std::size_t need = sizeof(float) * c.num_experts;
        if (need > sizeof(down_scales)) {
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
        // Compact routing: compute this expert only for the tokens routed
        // to it (rows_token/pad_bounds from RoutedCompact) instead of the
        // whole chunk. Same kernels and same math per row.
        std::uint32_t rows_begin = 0, rows_end = n;
        if (compact_ok) {
          rows_begin = static_cast<std::uint32_t>(bounds_host[e]);
          rows_end = static_cast<std::uint32_t>(bounds_host[e + 1]);
          if (rows_end <= rows_begin) continue;
          if (!q4d)
            kernels::CompactGatherF16(impl_->normed, impl_->compact_xhalf,
                                      impl_->rows_token, rows_begin, rows_end,
                                      hidden, stream);
        }
        const std::uint32_t batch = compact_ok ? (rows_end - rows_begin) : n;
        const bool f16w = compact_ok && getenv("GEMMA4_F16W") != nullptr;
        const bool f16wd =
            compact_ok && getenv("GEMMA4_F16WD") != nullptr;
        const __half* xin_f16 = compact_ok
                                    ? static_cast<const __half*>(impl_->compact_xhalf)
                                    : nullptr;
        float* xin = compact_ok ? impl_->normed_compact : impl_->normed;
        float* gout = impl_->mlp_out;
        float* uout = impl_->expert_out;
        float* dout = impl_->expert_down;
        if (compact_ok) {
          gout = impl_->compact_gate;
          uout = impl_->compact_up;
          dout = impl_->compact_down;
        }
        // Expert-local views: the compact buffers are indexed by the GLOBAL
        // compact row, so every per-expert kernel works on the [begin,end)
        // slice. The WMMA GEMM reads x rows [0,batch) from xg and writes the
        // expert's own rows via gg/gu/gd.
        const __half* xg = nullptr;
        float *gg = nullptr, *gu = nullptr, *gd = nullptr;
        if (compact_ok) {
          xg = static_cast<const __half*>(impl_->compact_xhalf) +
               static_cast<std::size_t>(rows_begin) * hidden;
          gg = gout + static_cast<std::size_t>(rows_begin) * c.expert_ff;
          gu = uout + static_cast<std::size_t>(rows_begin) * c.expert_ff;
          gd = dout + static_cast<std::size_t>(rows_begin) * hidden;
        }
        // Q4-direct: grouped gate/up + down already ran above; only the
        // per-expert scale + scatter remain for this expert.
        if (q4d) {
          if (down_scale != 1.0F) {
            kernels::ScaleInplace(gd, down_scale,
                                  static_cast<std::size_t>(batch) * hidden,
                                  stream);
          }
          kernels::MoeScatterAdd(gd, impl_->rows_token, impl_->rows_slot,
                                 impl_->compact_wts, rows_begin, rows_end,
                                 impl_->attn_out, hidden, stream);
          continue;
        }
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
          if (f16w && l.gate_up_f16 != nullptr) {
              // Pre-dequantized F16 arena: zero per-expert dequant launches.
              namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
              const __half* wbase = static_cast<const __half*>(l.gate_up_f16) +
                  static_cast<std::size_t>(e) * 2 * c.expert_ff * hidden;
              q38::UnquantizedF16Gemm(wbase, xg, gg, batch,
                                      c.expert_ff, hidden, stream);
              q38::UnquantizedF16Gemm(
                  wbase + static_cast<std::size_t>(c.expert_ff) * hidden, xg,
                  gu, batch, c.expert_ff, hidden, stream);
            } else if (f16w) {
              // F16 dequant path: write the dequantized weights once as raw
              // F16 rows (half the Q8_0 slab traffic) and feed the raw-F16
              // GEMM variant.
              namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
              kernels::DequantSlabF16(gate_ref, c.expert_ff, impl_->w_slab,
                                      stream);
              q38::UnquantizedF16Gemm(impl_->w_slab, xg, gg, batch,
                                      c.expert_ff, hidden, stream);
              kernels::DequantSlabF16(up_ref, c.expert_ff, impl_->w_slab,
                                      stream);
              q38::UnquantizedF16Gemm(impl_->w_slab, xg, gu, batch,
                                      c.expert_ff, hidden, stream);
            } else if (compact_ok) {
              // WMMA path with padded batch (multiple of 16): the small-batch
              // hipBLAS fallback faults on odd batches below 96 rows.
              namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
              kernels::DequantSlabQ8(gate_ref, c.expert_ff,
                                     reinterpret_cast<std::uint8_t*>(impl_->w_slab),
                                     stream);
              q38::DenseF16Gemm(impl_->w_slab, xg, gg, batch,
                                c.expert_ff, hidden, stream);
              kernels::DequantSlabQ8(up_ref, c.expert_ff,
                                     reinterpret_cast<std::uint8_t*>(impl_->w_slab),
                                     stream);
              q38::DenseF16Gemm(impl_->w_slab, xg, gu, batch,
                                c.expert_ff, hidden, stream);
            } else {
              F32Gemm(gate_ref, xin, gout, batch, c.expert_ff,
                      hidden, stream, impl_->w_slab, impl_->x_slab);
              F32Gemm(up_ref, xin, uout, batch, c.expert_ff,
                      hidden, stream, impl_->w_slab2, impl_->x_slab);
            }
        } else {
          TensorRef gate_ref = l.gate_exps;
          gate_ref.data = l.gate_exps.DeviceExpert(e);
          gate_ref.device = gate_ref.data;
          gate_ref.experts = 1;
          TensorRef up_ref = l.up_exps;
          up_ref.data = l.up_exps.DeviceExpert(e);
          up_ref.device = up_ref.data;
          up_ref.experts = 1;
          F32Gemm(gate_ref, xin, gout, batch, c.expert_ff,
                  hidden, stream, impl_->w_slab, impl_->x_slab);
          F32Gemm(up_ref, xin, uout, batch, c.expert_ff,
                  hidden, stream, impl_->w_slab2, impl_->x_slab);
        }
        if (compact_ok) {
          {
            if (f16wd && l.down_f16 != nullptr) {
              namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
              // Fused epilogue: act_half = f16(silu(gate)*up) in one pass
              // (replaces the GeluTanhSwiglu + F32ToF16 pair, bit-identical).
              kernels::SwigluF16(gg, gu, impl_->x_half,
                                 static_cast<std::size_t>(batch) * c.expert_ff,
                                 stream);
              q38::UnquantizedF16Gemm(
                  static_cast<const __half*>(l.down_f16) +
                      static_cast<std::size_t>(e) * hidden * c.expert_ff,
                  impl_->x_half, gd, batch, hidden, c.expert_ff, stream);
            } else if (f16wd) {
              namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
              kernels::DequantSlabF16(down_ref, hidden, impl_->w_slab,
                                      stream);
              kernels::SwigluF16(gg, gu, impl_->x_half,
                                 static_cast<std::size_t>(batch) * c.expert_ff,
                                 stream);
              q38::UnquantizedF16Gemm(impl_->w_slab, impl_->x_half, gd,
                                      batch, hidden, c.expert_ff, stream);
            } else if (getenv("GEMMA4_COMPACT_DOWN_WMMA") != nullptr) {
              // WMMA path for the down GEMM too (m=2816>=256, batch is a
              // multiple of 16); the hipBLAS fallback mis-handles the compact
              // batch here.
              namespace q38 = ::gufo::models::qwen38_flash_next::rocm;
              kernels::DequantSlabQ8(down_ref, hidden,
                                     reinterpret_cast<std::uint8_t*>(impl_->w_slab),
                                     stream);
              kernels::SwigluF16(gg, gu, impl_->x_half,
                                 static_cast<std::size_t>(batch) * c.expert_ff,
                                 stream);
              q38::DenseF16Gemm(impl_->w_slab, impl_->x_half, dout, batch,
                                hidden, c.expert_ff, stream);
            } else {
              // Compact fallback was missing the SwiGLU activation: gg held the
              // raw gate projection and gu was discarded, so down() saw
              // down(gate) instead of down(silu(gate)*up). Match the non-compact
              // path (gate activated in place, up consumed).
              kernels::GeluTanhSwiglu(
                  gg, gu, static_cast<std::size_t>(batch) * c.expert_ff,
                  stream);
              F32Gemm(down_ref, gg, gd, batch, hidden,
                      c.expert_ff, stream, impl_->w_slab, impl_->x_slab);
            }
            if (down_scale != 1.0F) {
              kernels::ScaleInplace(gd, down_scale,
                                    static_cast<std::size_t>(batch) * hidden,
                                    stream);
            }
          }
          kernels::MoeScatterAdd(gd, impl_->rows_token, impl_->rows_slot,
                                   impl_->compact_wts, rows_begin, rows_end,
                                   impl_->attn_out, hidden, stream);
        } else {
          kernels::GeluTanhSwiglu(impl_->mlp_out, impl_->expert_out,
                                  static_cast<std::size_t>(n) * c.expert_ff,
                                  stream);
          F32Gemm(down_ref, impl_->mlp_out, impl_->expert_down, n, hidden,
                  c.expert_ff, stream, impl_->w_slab, impl_->x_slab);
          if (down_scale != 1.0F) {
            kernels::ScaleInplace(impl_->expert_down, down_scale,
                                  static_cast<std::size_t>(n) * hidden, stream);
          }
          kernels::MoeRoutedAdd(impl_->expert_down, impl_->expert_ids,
                                impl_->expert_weights, e, impl_->attn_out, n,
                                c.experts_used, hidden, stream);
        }
      }
      }

      if (!fused_moe_norms) {
        SyncSum(impl_->attn_out, "moe_accum", n, hidden, il);
        kernels::RmsNormAdd(impl_->attn_out,
                            static_cast<const float*>(l.ffn_post_norm_2.device),
                            impl_->moe_out, n, hidden, c.rms_eps, 1.0F, stream);
        SyncSum(impl_->moe_out, "combined", n, hidden, il);
        if (debug_embed && il == 0) {
          hipDeviceSynchronize();
          std::vector<float> mr(hidden);
          (void)hipMemcpy(mr.data(), impl_->moe_out, hidden * sizeof(float),
                          hipMemcpyDeviceToHost);
          double ms = 0;
          for (float v : mr) ms += (double)v;
          std::fprintf(stderr, "[gemma4] L%d n=%u moe_row0=%.6f\n", il, n, ms);
        }
        if (debug_embed) {
          hipDeviceSynchronize();
          std::vector<float> hb(static_cast<std::size_t>(n) * hidden);
          (void)hipMemcpy(hb.data(), impl_->hidden, hb.size() * sizeof(float),
                          hipMemcpyDeviceToHost);
          double hs = 0;
          float hm = 0;
          for (float v : hb) {
            hs += (double)v;
            float a = v < 0 ? -v : v;
            if (a > hm) hm = a;
          }
          std::fprintf(stderr, "[gemma4] L%u hidden sum=%.4f maxabs=%.4f\n", il,
                       hs, hm);
        }
        kernels::RmsNormAdd(
            impl_->moe_out, static_cast<const float*>(l.ffn_post_norm.device),
            impl_->hidden, n, hidden, c.rms_eps,
            l.has_out_scale ? l.out_scale_val : 1.0F, stream);
      }
    }

    pos += n;
  }

  // Output norm + LM head.
  kernels::RmsNorm(impl_->hidden,
                   static_cast<const float*>(w.output_norm.device),
                   impl_->normed, n, hidden, c.rms_eps, stream);
  if (getenv("GEMMA4_DEBUG_EMBED") != nullptr && n <= 4) {
    hipDeviceSynchronize();
    std::vector<float> fh(hidden);
    (void)hipMemcpy(fh.data(), impl_->normed, hidden * sizeof(float),
                    hipMemcpyDeviceToHost);
    double fs = 0;
    for (float v : fh) fs += (double)v;
    std::fprintf(stderr, "[gemma4] finalnorm_row0_n%u=%.6f\n", n, fs);
  }

  const bool force_all = getenv("GEMMA4_FORCE_ALLLOGITS") != nullptr;
  const std::uint32_t logit_rows =
      (all_logits || force_all) ? std::min(n, impl_->logit_cap) : 1u;
  if (getenv("GEMMA4_DEBUG_EMBED") != nullptr && n <= 16) {
    std::fprintf(stderr, "[gemma4] head n=%u all_logits=%d logit_rows=%u\n", n,
                 (int)all_logits, logit_rows);
  }
  const float* last_row =
      (all_logits || force_all)
          ? (impl_->normed + static_cast<std::size_t>(n - logit_rows) * hidden)
          : (impl_->normed + static_cast<std::size_t>(n - 1) * hidden);
  StagedGemm(w.output, c.vocab_size, last_row, impl_->logits, logit_rows,
             impl_->w_slab, impl_->x_slab, stream, c.final_logit_softcapping);

  impl_->last_fwd_n = n;
  impl_->last_chunk_n = logit_rows;
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
  impl_->has_pending_cur = false;
  std::size_t chunk = Session::Impl::kMaxChunk;
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
  (void)hipStreamSynchronize(0);
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  return Forward({&token, 1}, impl_->length, error_msg);
}

bool Session::EnableMtp(const std::string& drafts_path, std::string* error_msg) {
  if (model_->options_.kv_cache_quant != "f16") {
    if (error_msg != nullptr)
      *error_msg =
          "MTP requires f16 KV caches (blocked q8 has no draft-cache path)";
    return false;
  }
  const auto& w = *model_->weights_;
  auto drafter = MtpDrafter::Load(drafts_path, w.config.hidden_size, error_msg);
  if (drafter == nullptr)
    return false;
  if (drafter->VocabSize() != w.config.vocab_size) {
    if (error_msg != nullptr)
      *error_msg = "MTP vocab (" + std::to_string(drafter->VocabSize()) +
                   ") does not match the target vocab (" +
                   std::to_string(w.config.vocab_size) + ")";
    return false;
  }
  impl_->mtp = std::move(drafter);
  return true;
}

bool Session::MtpEnabled() const noexcept { return impl_->mtp != nullptr; }

bool Session::SpeculativeRound(std::int32_t last_token, std::uint32_t n_draft,
                               std::vector<std::int32_t>* out,
                               std::string* error_msg) {
  (void)last_token;
  if (!impl_->mtp || out == nullptr) {
    return false;
  }
  const auto& w = *model_->weights_;
  const auto& c = w.config;
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t vocab = c.vocab_size;
  hipStream_t stream = 0;
  if (impl_->length == 0) {
    if (error_msg != nullptr) *error_msg = "speculative round needs a prompt";
    return false;
  }
  if (n_draft > 7u) n_draft = 7u;

  // Cross-attention sources: full-length KV of the last sliding layer and of
  // the last full-attention layer (llama.cpp gemma4-assistant.cpp parity).
  int swa_l = -1;
  int full_l = -1;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (!c.LayerHasKv(il) || impl_->kv_k[il] == nullptr) continue;
    if (c.LayerIsSwa(il)) {
      swa_l = static_cast<int>(il);
    } else {
      full_l = static_cast<int>(il);
    }
  }
  if (swa_l < 0 || full_l < 0) {
    if (error_msg != nullptr) *error_msg = "speculative KV layers missing";
    return false;
  }

  const std::uint32_t old_length = impl_->length;
  std::int32_t cur = 0;
  if (impl_->has_pending_cur) {
    cur = impl_->pending_cur;
  } else {
    cur = SampleGreedy();
    (void)hipMemcpyAsync(
        impl_->d_pending_h,
        impl_->normed +
            static_cast<std::size_t>(impl_->last_fwd_n - 1u) * hidden,
        static_cast<std::size_t>(hidden) * sizeof(float),
        hipMemcpyDeviceToDevice, stream);
  }

  if (n_draft == 0 || getenv("GEMMA4_MTP_NOVERIFY") != nullptr) {
    out->assign(1, cur);
    if (!Forward({&cur, 1}, old_length, error_msg)) return false;
    return true;
  }

  const float emb_scale = std::sqrt(static_cast<float>(hidden));
  const float* full_rope_freqs =
      !w.layers[static_cast<std::uint32_t>(full_l)].rope_freqs.empty()
          ? static_cast<const float*>(
                w.layers[static_cast<std::uint32_t>(full_l)].rope_freqs.device)
          : nullptr;
  const std::uint32_t swa_kvh = c.KvHeads(static_cast<std::uint32_t>(swa_l));
  const std::uint32_t full_kvh = c.KvHeads(static_cast<std::uint32_t>(full_l));

  const bool mtp_timing = getenv("GEMMA4_MTP_TIMING") != nullptr;
  const auto t_round_start = std::chrono::steady_clock::now();

  // Draft n_draft tokens asynchronously at kv_len = old_length, rope_pos = old_length.
  for (std::uint32_t i = 0; i < n_draft; ++i) {
    if (i == 0u) {
      kernels::EmbedScale(w.token_embd.device, w.token_embd.type, &cur,
                          impl_->d_tgt_emb, 1u, hidden, emb_scale, stream);
    } else {
      kernels::EmbedScaleDevice(w.token_embd.device, w.token_embd.type,
                                impl_->d_draft_tok + (i - 1u), impl_->d_tgt_emb,
                                1u, hidden, emb_scale, stream);
    }
    const float* feed_hidden =
        (i == 0u) ? impl_->d_pending_h : impl_->d_draft_fb;
    float* next_fb = (i + 1u < n_draft) ? impl_->d_draft_fb : nullptr;
    if (!impl_->mtp->Step(
            impl_->d_tgt_emb, feed_hidden, impl_->kv_k[swa_l],
            impl_->kv_v[swa_l], impl_->kv_stride[swa_l], swa_kvh,
            impl_->kv_k[full_l], impl_->kv_v[full_l], impl_->kv_stride[full_l],
            full_kvh, old_length, old_length, c.sliding_window,
            c.rope_theta_swa, c.rope_theta_full, full_rope_freqs,
            next_fb, impl_->d_draft_tok + i, error_msg)) {
      return false;
    }
  }

  std::vector<std::int32_t> draft(n_draft, 0);
  (void)hipMemcpy(draft.data(), impl_->d_draft_tok,
                  n_draft * sizeof(std::int32_t), hipMemcpyDeviceToHost);
  const auto t_after_draft = std::chrono::steady_clock::now();

  // Verify [cur, draft[0..n_draft-1]] in ONE single target Forward at start_pos = old_length.
  const std::uint32_t verify_n = 1u + n_draft;
  std::vector<std::int32_t> vtoks(verify_n);
  vtoks[0] = cur;
  for (std::uint32_t i = 0; i < n_draft; ++i) vtoks[1u + i] = draft[i];

  if (!Forward(vtoks, old_length, error_msg, /*all_logits=*/true)) {
    return false;
  }

  fn::Argmax(impl_->logits, impl_->d_argmax_scratch, impl_->d_argmax_out,
             verify_n, vocab, stream);
  std::int32_t vbest[8] = {0};
  (void)hipMemcpy(vbest, impl_->d_argmax_out,
                  verify_n * sizeof(std::int32_t), hipMemcpyDeviceToHost);
  if (mtp_timing) {
    static int t_cnt = 0;
    if (t_cnt < 10) {
      const double d_ms =
          std::chrono::duration<double, std::milli>(t_after_draft -
                                                    t_round_start)
              .count();
      const double v_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t_after_draft)
              .count();
      std::fprintf(stderr, "[mtp-timing] round=%d draft=%.2fms verify=%.2fms\n",
                   t_cnt, d_ms, v_ms);
      ++t_cnt;
    }
  }

  std::uint32_t accepted = 0;
  while (accepted < n_draft && draft[accepted] == vbest[accepted]) {
    ++accepted;
  }

  if (getenv("GEMMA4_MTP_TRACE") != nullptr) {
    static int traced = 0;
    if (traced < 32) {
      std::fprintf(stderr,
                   "[mtp] pos=%u cur=%d draft=[%d,%d,%d,%d] vbest=[%d,%d,%d,%d] acc=%u\n",
                   old_length, cur,
                   n_draft > 0 ? draft[0] : -1, n_draft > 1 ? draft[1] : -1,
                   n_draft > 2 ? draft[2] : -1, n_draft > 3 ? draft[3] : -1,
                   vbest[0], n_draft > 1 ? vbest[1] : -1,
                   n_draft > 2 ? vbest[2] : -1, n_draft > 3 ? vbest[3] : -1,
                   accepted);
      ++traced;
    }
  }

  out->assign(vtoks.begin(), vtoks.begin() + 1u + accepted);

  // KV slots [old_length .. old_length + accepted] already hold valid K/V for
  // [cur, draft[0..accepted-1]] — even when accepted == 0, slot old_length
  // holds cur, so no fallback Forward is ever needed.
  impl_->length = old_length + 1u + accepted;
  impl_->pending_cur = vbest[accepted];
  (void)hipMemcpyAsync(
      impl_->d_pending_h,
      impl_->normed + static_cast<std::size_t>(accepted) * hidden,
      static_cast<std::size_t>(hidden) * sizeof(float), hipMemcpyDeviceToDevice,
      stream);
  impl_->has_pending_cur = true;
  impl_->last_chunk_n = accepted + 1u;
  return true;
}


}  // namespace gufo::models::gemma4
