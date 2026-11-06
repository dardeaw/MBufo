// GPT-OSS runtime: GGUF load, chunked prefill, single-token decode, greedy
// sampling. Dense GEMM rides the shared core (narrow-prequant decode,
// DenseF16Gemm prefill, fused QKV/wo narrow variants for decode); prefill
// MoE rides the shared compact routed GEMM (RoutedCompact/RoutedF16Gemm over
// native MXFP4 slabs, model-local SwiGLU-OAI split) while decode takes the
// fused core MoE pair (MoeDecodeGateUpActMxfp4 + MoeDecodeDownEpiMxfp4, one
// wave per output row) with the routed/vec sequence as contract fallback;
// attention is a model-local dense F16 kernel (per-head sinks + sliding
// window) over f16 KV or over shared blocked-Q8 KV dequantized with
// blocked_kv::DequantizeKvRows.
#include "src/models/gptoss/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/compute/attention/attention_dispatcher.hpp"
#include "src/core/compute/attention/blocked_kv.hpp"
#include "src/core/compute/attention/flash_blas.hpp"
#include "src/core/compute/attention/flash_sink.hpp"
#include "src/core/compute/gemm/gemm.hpp"
#include "src/models/gptoss/kernels/kernels.hpp"

namespace gufo::models::gptoss {
namespace {

constexpr std::uint32_t kMaxChunk = 256;

// Env-gated per-section timing (GPTOSS_LAYER_TIME=1). Zero cost when unset
// (one cached getenv). Accumulates hipEvent deltas per ForwardLayer section,
// split by decode (n == 1) vs prefill (n > 1), plus ForwardFinish, and dumps
// averages at exit. Diag only; never affects numerics or launch order.
enum class LayerSection : std::uint32_t {
  kQkv = 0,
  kRopeKv,
  kAttn,
  kWo,
  kRouter,
  kMoe,
  kTail,
  kFinish,
  kCount
};

struct LayerTiming {
  bool ready{false};
  bool enabled{false};
  hipEvent_t ev[static_cast<std::size_t>(LayerSection::kCount) + 1]{};
  double decode_ms[static_cast<std::size_t>(LayerSection::kCount)]{};
  double prefill_ms[static_cast<std::size_t>(LayerSection::kCount)]{};
  std::uint64_t decode_calls{0};
  std::uint64_t prefill_calls{0};
};

inline LayerTiming& TimingState() {
  static LayerTiming state;
  return state;
}

inline void TimingDump() {
  LayerTiming& t = TimingState();
  if (!t.enabled) return;
  static const char* kNames[] = {"qkv",   "ropekv", "attn", "wo",
                                 "router", "moe",   "tail", "finish"};
  std::fprintf(stderr, "[gptoss-time] layer calls: decode=%llu prefill=%llu\n",
               static_cast<unsigned long long>(t.decode_calls),
               static_cast<unsigned long long>(t.prefill_calls));
  for (std::size_t i = 0;
       i < static_cast<std::size_t>(LayerSection::kCount); ++i) {
    const double d =
        t.decode_calls != 0 ? t.decode_ms[i] / t.decode_calls : 0.0;
    const double p =
        t.prefill_calls != 0 ? t.prefill_ms[i] / t.prefill_calls : 0.0;
    std::fprintf(stderr,
                 "[gptoss-time] %-6s decode_avg=%7.3fms prefill_avg=%7.3fms\n",
                 kNames[i], d, p);
  }
}

inline void TimingEnsure() {
  LayerTiming& t = TimingState();
  if (t.ready) return;
  t.ready = true;
  t.enabled = std::getenv("GPTOSS_LAYER_TIME") != nullptr;
  if (!t.enabled) return;
  for (auto& e : t.ev) {
    if (hipEventCreate(&e) != hipSuccess) {
      t.enabled = false;
      return;
    }
  }
  (void)std::atexit(TimingDump);
}

inline void TimingMark(LayerSection s) {
  LayerTiming& t = TimingState();
  if (t.enabled) {
    (void)hipEventRecord(t.ev[static_cast<std::size_t>(s)], 0);
  }
}

// Ends a ForwardLayer call: syncs (diag only), accumulates the 7 section
// deltas ev[0..7] into the decode/prefill bucket selected by n.
inline void TimingEndLayer(std::uint32_t n) {
  LayerTiming& t = TimingState();
  if (!t.enabled) return;
  (void)hipStreamSynchronize(0);
  double* bucket = (n == 1) ? t.decode_ms : t.prefill_ms;
  for (std::size_t i = 0; i < 7; ++i) {
    float ms = 0.0F;
    (void)hipEventElapsedTime(&ms, t.ev[i], t.ev[i + 1]);
    bucket[i] += ms;
  }
  if (n == 1) {
    ++t.decode_calls;
  } else {
    ++t.prefill_calls;
  }
}

// Ends a ForwardFinish call: accumulates ev[0]->ev[1] into kFinish.
inline void TimingEndFinish(std::uint32_t n) {
  LayerTiming& t = TimingState();
  if (!t.enabled) return;
  (void)hipStreamSynchronize(0);
  float ms = 0.0F;
  (void)hipEventElapsedTime(&ms, t.ev[0], t.ev[1]);
  if (n == 1) {
    t.decode_ms[static_cast<std::size_t>(LayerSection::kFinish)] += ms;
  } else {
    t.prefill_ms[static_cast<std::size_t>(LayerSection::kFinish)] += ms;
  }
}

bool Check(hipError_t err, std::string* error_msg, const char* what) {
  if (err == hipSuccess)
    return true;
  if (error_msg != nullptr && error_msg->empty())
    *error_msg = std::string("gptoss HIP failure in ") + what;
  return false;
}

// Mirrors one weight tensor to device; converted tensors stay packed (the
// shared launchers dequantize on the fly). The H2D copy is chunked to
// <=256MB per hipMemcpy so one multi-GB tensor (e.g. merged 128-expert MoE
// slabs in 120B) never pins a giant range inside
// svm_range_cpu_invalidate_pagetables (a single 59GB H2D wedged the box in
// D-state). Sync-copy semantics unchanged: same bytes, same order.
bool MirrorTensor(TensorRef& t, std::vector<void*>& owned,
                  std::string* error_msg) {
  if (t.empty())
    return true;
  void* dev = nullptr;
  if (!Check(hipMalloc(&dev, t.size_bytes), error_msg, "weight mirror alloc"))
    return false;
  constexpr std::size_t kMirrorChunkBytes = 256u * 1024u * 1024u;
  const auto* src = static_cast<const char*>(t.data);
  auto* dst = static_cast<char*>(dev);
  for (std::size_t off = 0; off < t.size_bytes; off += kMirrorChunkBytes) {
    const std::size_t n = std::min(kMirrorChunkBytes, t.size_bytes - off);
    if (!Check(hipMemcpy(dst + off, src + off, n, hipMemcpyHostToDevice),
               error_msg, "weight mirror copy")) {
      (void)hipFree(dev);
      return false;
    }
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
  // Compact routed-MoE prefill scratch (shared RoutedCompact/F16Gemm over
  // native MXFP4 slabs). Sized by RoutedCompactRows(chunk*used, experts).
  std::uint32_t* routed_counts{nullptr};
  std::int32_t* routed_bounds{nullptr};
  std::int32_t* routed_cursors{nullptr};
  std::int32_t* routed_rows_token{nullptr};
  std::int32_t* routed_rows_slot{nullptr};
  std::int32_t* routed_rows_expert{nullptr};
  std::int32_t* routed_tiles{nullptr};
  std::int32_t* routed_identity{nullptr};
  float* routed_gateup{nullptr};
  __half* routed_act{nullptr};
  std::vector<std::int32_t> routed_bounds_host;
  std::vector<std::int32_t> routed_tiles_host;
  std::vector<std::int32_t> routed_expert_host;
  float* ffn_out{nullptr};
  float* logits{nullptr};
  float* flash_partials{nullptr};
  std::size_t flash_partials_bytes{0};
  void* flash_blas_scratch{nullptr};
  std::size_t flash_blas_bytes{0};
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
    const std::size_t compact_cap =
        gemm::RoutedCompactRows(slots, w.config.num_experts);
    routed_counts = buffers.Alloc<std::uint32_t>(w.config.num_experts, error);
    routed_bounds =
        buffers.Alloc<std::int32_t>(w.config.num_experts + 1, error);
    routed_cursors =
        buffers.Alloc<std::int32_t>(w.config.num_experts, error);
    routed_rows_token = buffers.Alloc<std::int32_t>(compact_cap, error);
    routed_rows_slot = buffers.Alloc<std::int32_t>(compact_cap, error);
    routed_rows_expert = buffers.Alloc<std::int32_t>(compact_cap, error);
    routed_tiles = buffers.Alloc<std::int32_t>(compact_cap, error);
    routed_identity = buffers.Alloc<std::int32_t>(compact_cap, error);
    routed_gateup = buffers.Alloc<float>(compact_cap * 2 * ff, error);
    routed_act = buffers.Alloc<__half>(compact_cap * ff, error);
    routed_bounds_host.assign(w.config.num_experts + 1, 0);
    routed_tiles_host.reserve(compact_cap);
    routed_expert_host.assign(compact_cap, -1);
    {
      // Identity rows for the compact down-GEMM gather (one per session).
      std::vector<std::int32_t> ident(compact_cap);
      for (std::size_t i = 0; i < compact_cap; ++i)
        ident[i] = static_cast<std::int32_t>(i);
      if (hipMemcpy(routed_identity, ident.data(),
                    compact_cap * sizeof(std::int32_t),
                    hipMemcpyHostToDevice) != hipSuccess) {
        if (error != nullptr && error->empty())
          *error = "gptoss routed identity upload failed";
        return false;
      }
    }
    ffn_out = buffers.Alloc<float>(chunk * hidden_dim, error);
    logits = buffers.Alloc<float>(w.config.vocab_size, error);
    flash_partials_bytes =
        core::compute::flash_sink::FlashSinkDecodeScratchBytes(
            core::compute::flash_sink::kFlashSinkMaxTokens,
            w.config.num_heads);
    flash_partials =
        buffers.Alloc<float>(flash_partials_bytes / sizeof(float), error);
    flash_blas_bytes =
        core::compute::flash_blas::FlashBlasScratchBytes(chunk);
    flash_blas_scratch = buffers.Alloc<std::uint8_t>(flash_blas_bytes, error);
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
           routed_counts != nullptr && routed_bounds != nullptr &&
           routed_cursors != nullptr && routed_rows_token != nullptr &&
           routed_rows_slot != nullptr && routed_rows_expert != nullptr &&
           routed_tiles != nullptr && routed_identity != nullptr &&
           routed_gateup != nullptr && routed_act != nullptr &&
           ffn_out != nullptr && flash_partials != nullptr &&
           flash_blas_scratch != nullptr && logits != nullptr &&
           x16 != nullptr && d_tokens != nullptr && d_pos != nullptr && preq_scratch != nullptr &&
           argmax_scratch != nullptr && argmax_out != nullptr;
  }
};

Model::~Model() {
  for (void* p : device_allocations_) {
    (void)hipFree(p);
  }
  if (blas_handle_ != nullptr) {
    (void)hipblasDestroy(blas_handle_);
    blas_handle_ = nullptr;
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
  // rocBLAS handle for batched-GEMM attention (best effort; null disables).
  hipblasHandle_t blas_handle = nullptr;
  if (hipblasCreate(&blas_handle) == HIPBLAS_STATUS_SUCCESS) {
    model_owner->blas_handle_ = blas_handle;
  }
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
  TimingEnsure();
  TimingMark(LayerSection::kQkv);
  kernels::RmsNorm(hidden_buf,
                   static_cast<const float*>(l.attn_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  // Fused QKV narrow decode: one prequant + one launch for wq/wk/wv when the
  // narrow-QKV contract holds (decode, all-Q8_0, shared activation row and
  // K). GPTOSS_NO_FUSED forces the three legacy GptossGemm calls for A/B.
  const bool no_fused = std::getenv("GPTOSS_NO_FUSED") != nullptr;
  const bool no_fused_qkv =
      no_fused || std::getenv("GPTOSS_NO_FUSED_QKV") != nullptr;
  const bool no_fused_moe =
      no_fused || std::getenv("GPTOSS_NO_FUSED_MOE") != nullptr;
  const bool no_fused_router =
      no_fused || std::getenv("GPTOSS_NO_FUSED_ROUTER") != nullptr;
  const bool no_fused_wo =
      no_fused || std::getenv("GPTOSS_NO_FUSED_WO") != nullptr;
  bool qkv_done = false;
  if (!no_fused_qkv &&
      gemm::DecideNarrowQkvPrequantRoute(
          l.wq.type == core::GgmlType::kQ8_0 &&
              l.wk.type == core::GgmlType::kQ8_0 &&
              l.wv.type == core::GgmlType::kQ8_0,
          n, hidden, qdim, kvdim) ==
          gemm::NarrowDecodeRoute::kNarrowQ80Prequant) {
    const void* wqd = l.wq.device != nullptr ? l.wq.device : l.wq.data;
    const void* wkd = l.wk.device != nullptr ? l.wk.device : l.wk.data;
    const void* wvd = l.wv.device != nullptr ? l.wv.device : l.wv.data;
    if (gemm::PrequantizeQ8_1(impl.normed, impl.preq_scratch, n, hidden,
                              stream) == 0) {
      qkv_done = gemm::LaunchNarrowQ80QkvDecode(
          wqd, wkd, wvd, impl.preq_scratch, impl.qbuf, impl.kbuf, impl.vbuf,
          hidden, qdim, kvdim, n,
          gemm::NarrowQ80PrequantStrideBlocks(hidden), stream);
    }
  }
  if (!qkv_done) {
    GptossGemm(l.wq, impl.normed, impl.x16, impl.qbuf, n, qdim, hidden, stream,
               impl.preq_scratch, impl.preq_bytes);
    GptossGemm(l.wk, impl.normed, impl.x16, impl.kbuf, n, kvdim, hidden, stream,
               impl.preq_scratch, impl.preq_bytes);
    GptossGemm(l.wv, impl.normed, impl.x16, impl.vbuf, n, kvdim, hidden, stream,
               impl.preq_scratch, impl.preq_bytes);
  }
  TimingMark(LayerSection::kRopeKv);
  kernels::RopeYarn(impl.qbuf, n, c.num_heads, d, start_pos, c.rope_theta,
                    c.yarn.low, c.yarn.high, c.yarn.inv_factor, c.yarn.mscale,
                    stream);
  kernels::RopeYarn(impl.kbuf, n, c.num_kv_heads, d, start_pos, c.rope_theta,
                    c.yarn.low, c.yarn.high, c.yarn.inv_factor, c.yarn.mscale,
                    stream);
  TimingMark(LayerSection::kAttn);
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
    // F16 KV: shared flash-sink (tiled prefill / split-K decode) when the
    // d == 64 contract holds; GPTOSS_NO_FLASH forces the local kernel.
    // kv_len is start_pos + n (caches hold exactly the valid prefix).
    namespace flash = core::compute::flash_sink;
    namespace blas = core::compute::flash_blas;
    bool flashed = false;
    if (d == 64 && getenv("GPTOSS_NO_FLASH") == nullptr) {
      if (n == 1) {
        flashed = flash::FlashSinkDecode(
            impl.qbuf, impl.buffers.kv_k[il], impl.buffers.kv_v[il],
            static_cast<const float*>(l.attn_sinks.device), impl.attn_out, n,
            start_pos, start_pos + n, c.num_heads, c.num_kv_heads, d, window,
            score_scale, stream, impl.flash_partials,
            impl.flash_partials_bytes);
      } else if (s.model_->blas_handle_ != nullptr &&
                 getenv("GPTOSS_NO_BLAS") == nullptr) {
        flashed = blas::FlashBlasPrefill(
            impl.qbuf, impl.buffers.kv_k[il], impl.buffers.kv_v[il],
            static_cast<const float*>(l.attn_sinks.device), impl.attn_out, n,
            start_pos, start_pos + n, c.num_heads, c.num_kv_heads, d, window,
            score_scale, stream, s.model_->blas_handle_,
            impl.flash_blas_scratch, impl.flash_blas_bytes);
      }
      if (!flashed && n > 1) {
        // Fallback: naive single-query kernel (proven bitwise vs legacy).
        flashed = flash::FlashSinkPrefill(
            impl.qbuf, impl.buffers.kv_k[il], impl.buffers.kv_v[il],
            static_cast<const float*>(l.attn_sinks.device), impl.attn_out, n,
            start_pos, start_pos + n, c.num_heads, c.num_kv_heads, d, window,
            score_scale, stream);
      }
    }
    if (!flashed) {
      kernels::AttentionF16Sink(
          impl.qbuf, impl.buffers.kv_k[il], impl.buffers.kv_v[il],
          static_cast<const float*>(l.attn_sinks.device), impl.attn_out, n,
          start_pos, c.num_heads, c.num_kv_heads, d, window, stream,
          score_scale);
    }
  }
  TimingMark(LayerSection::kWo);
  // Fused wo narrow decode: prequant + GEMV + wo bias + residual add in one
  // core launch (bitwise-identical store order). Falls back to the
  // three-launch sequence out of contract or under GPTOSS_NO_FUSED.
  bool wo_done = false;
  if (!no_fused_wo &&
      gemm::DecideNarrowDecodePrequantRoute(
          l.wo.type == core::GgmlType::kQ8_0, /*gated=*/false, n, qdim,
          hidden) == gemm::NarrowDecodeRoute::kNarrowQ80Prequant) {
    const void* wod = l.wo.device != nullptr ? l.wo.device : l.wo.data;
    wo_done = gemm::LaunchNarrowQ80PrequantRowAdd(
        wod, impl.attn_out,
        static_cast<const float*>(l.wo_b.device), hidden_buf, qdim, hidden, n,
        impl.preq_scratch, impl.preq_bytes, stream);
  }
  if (!wo_done) {
    GptossGemm(l.wo, impl.attn_out, impl.x16, impl.ffn_out, n, hidden, qdim,
               stream, impl.preq_scratch, impl.preq_bytes);
    // Attention output bias (gpt-oss wo_b) then residual add.
    kernels::AddRowBias(impl.ffn_out, static_cast<const float*>(l.wo_b.device),
                        n, hidden, stream);
    kernels::ResidualAdd(impl.ffn_out, hidden_buf,
                         static_cast<std::size_t>(n) * hidden, stream);
  }

  // MoE block: post-attention norm, router, shared vec MoE.
  TimingMark(LayerSection::kRouter);
  kernels::RmsNorm(hidden_buf,
                   static_cast<const float*>(l.post_norm.device),
                   impl.normed, n, hidden, c.rms_eps, stream);
  // Router GEMM (F32): shared SmallGemm narrow projection covers any n in
  // one launch; bias folded in after. GPTOSS_NO_FUSED(_ROUTER) keeps the
  // legacy wave32 GemvBatched path for A/B.
  if (!no_fused_router) {
    gemm::fn_compat::SmallGemm(l.router.device, 0, impl.normed,
                               impl.router_logits, n, nex, hidden, stream);
  } else {
    for (std::uint32_t off = 0; off < n; off += 8) {
      const std::uint32_t nr = std::min<std::uint32_t>(8, n - off);
      gemm::GemvBatched(l.router.device, static_cast<int>(l.router.type),
                        l.router.RowBytes(), hidden,
                        impl.normed + static_cast<std::size_t>(off) * hidden,
                        impl.router_logits + static_cast<std::size_t>(off) * nex,
                        nr, nex, 0.0F, stream);
    }
  }
  kernels::AddRowBias(impl.router_logits,
                      static_cast<const float*>(l.router_b.device), n, nex,
                      stream);
  gemm::RouterTopK(impl.router_logits, nex, impl.expert_ids,
                   impl.expert_weights, n, nex, used, stream);
  TimingMark(LayerSection::kMoe);
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
  // Fused MoE decode (core, two launches): gate+up GEMV + SwiGLU-OAI in one
  // wave-per-row kernel, down GEMV + bias + weighted epilogue in a second.
  // Takes precedence over the routed/vec sequence when the fused-MXFP4
  // contract holds; GPTOSS_NO_FUSED forces the legacy sequence for A/B.
  bool moe_done = false;
  if (!no_fused_moe &&
      gemm::DecideMoeDecodeFusedRoute(
          gu_dev != nullptr && dn_dev != nullptr &&
              l.gate_up_exps.type == core::GgmlType::kMXFP4 &&
              l.down_exps.type == core::GgmlType::kMXFP4,
          n, used, ff, hidden) == gemm::MoeDecodeFusedRoute::kFusedMxfp4) {
    moe_done =
        gemm::MoeDecodeGateUpActMxfp4(
            gu_dev, l.gate_up_exps.RowBytes(), gu_stride, impl.normed,
            impl.expert_ids,
            static_cast<const float*>(l.gate_b.device),
            static_cast<const float*>(l.up_b.device), impl.moe_act, slots,
            used, ff, hidden, stream) &&
        gemm::MoeDecodeDownEpiMxfp4(
            dn_dev, l.down_exps.RowBytes(), dn_stride, impl.moe_act,
            impl.expert_ids, impl.expert_weights,
            static_cast<const float*>(l.down_b.device), impl.ffn_out, n,
            used, hidden, ff, stream);
  }
  // Prefill AND decode MoE ride the shared compact routed GEMM over the
  // native MXFP4 slabs (one grouped GEMM per projection instead of one
  // GEMV per slot); any contract rejection falls back to the vec path.
  // GPTOSS_NO_ROUTED forces the vec path for A/B comparisons.
  std::uint32_t tile_rows = 64;
  if (const char* tr_env = getenv("GPTOSS_MOE_TILE")) {
    const unsigned long v = std::strtoul(tr_env, nullptr, 10);
    if (v == 16ul || v == 48ul || v == 64ul)
      tile_rows = static_cast<std::uint32_t>(v);
  }
  bool routed = getenv("GPTOSS_NO_ROUTED") == nullptr && !moe_done &&
                impl.routed_gateup != nullptr && gu_dev != nullptr &&
                dn_dev != nullptr;
  if (routed) {
    // F16 staging reuses x16 (chunk x max_k, max_k >= hidden); the dense
    // projections above are done with it.
    kernels::F32ToF16(impl.normed, impl.x16,
                      static_cast<std::size_t>(n) * hidden, stream);
    gemm::ExpertCounts(impl.expert_ids, impl.routed_counts, n, nex, used,
                       stream);
    gemm::RoutedCompact(impl.expert_ids, impl.routed_counts,
                        impl.routed_bounds, impl.routed_cursors,
                        impl.routed_rows_token, impl.routed_rows_slot, n, used,
                        nex, stream);
    // One (expert | tile<<16) entry per bucket tile (host loop over the
      // compact bounds, same contract as the gemma4 Q4-direct path).
      if (hipMemcpy(impl.routed_bounds_host.data(), impl.routed_bounds,
                  static_cast<std::size_t>(nex + 1) * sizeof(std::int32_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        routed = false;
      } else {
        auto& tiles_host = impl.routed_tiles_host;
        tiles_host.clear();
        auto& expert_host = impl.routed_expert_host;
        for (std::uint32_t e = 0; e < nex; ++e) {
          const std::uint32_t row_begin = static_cast<std::uint32_t>(
              impl.routed_bounds_host[e]);
          const std::uint32_t rows_e = static_cast<std::uint32_t>(
              impl.routed_bounds_host[e + 1]) - row_begin;
          const std::uint32_t n_e_tiles =
              (rows_e + tile_rows - 1) / tile_rows;
          for (std::uint32_t j = 0; j < n_e_tiles; ++j)
            tiles_host.push_back(static_cast<std::int32_t>(e | (j << 16)));
          for (std::uint32_t r = 0; r < rows_e; ++r)
            expert_host[row_begin + r] = static_cast<std::int32_t>(e);
        }
      const std::uint32_t n_tiles =
          static_cast<std::uint32_t>(tiles_host.size());
      const std::uint32_t q4_rows =
          static_cast<std::uint32_t>(impl.routed_bounds_host[nex]);
      if (n_tiles == 0 || q4_rows == 0) {
        routed = false;
      } else if (hipMemcpyAsync(impl.routed_tiles, tiles_host.data(),
                                static_cast<std::size_t>(n_tiles) *
                                    sizeof(std::int32_t),
                                hipMemcpyHostToDevice,
                                stream) != hipSuccess ||
                 hipMemcpyAsync(impl.routed_rows_expert,
                                expert_host.data(),
                                static_cast<std::size_t>(q4_rows) *
                                    sizeof(std::int32_t),
                                hipMemcpyHostToDevice,
                                stream) != hipSuccess) {
        routed = false;
      } else {
        routed = gemm::RoutedF16Gemm(
            gu_dev, gemm::WeightType::kMXFP4, impl.x16, impl.routed_tiles,
            n_tiles, tile_rows, impl.routed_bounds, impl.routed_rows_token,
            impl.routed_identity, nullptr, impl.routed_gateup, nullptr,
            static_cast<std::size_t>(2) * ff, hidden, stream);
          if (routed) {
            kernels::MoeCompactSplitBiasAct(
                impl.routed_gateup, impl.routed_rows_token,
                impl.routed_rows_expert,
                static_cast<const float*>(l.gate_b.device),
                static_cast<const float*>(l.up_b.device), impl.routed_act,
                q4_rows, ff, stream);
            routed = gemm::RoutedF16Gemm(
                dn_dev, gemm::WeightType::kMXFP4, impl.routed_act,
                impl.routed_tiles, n_tiles, tile_rows, impl.routed_bounds,
                impl.routed_identity, impl.routed_rows_slot, nullptr,
                impl.moe_down, nullptr, hidden, ff, stream);
          }
      }
    }
  }
  if (!moe_done && !routed) {
    gemm::MoeVecGateUpMxfp4(gu_dev, l.gate_up_exps.RowBytes(), gu_stride,
                       impl.normed, impl.expert_ids, impl.moe_gate,
                       impl.moe_up, slots, used, ff, hidden, stream);
    kernels::SwigluOaiBias(impl.moe_gate, impl.moe_up, impl.expert_ids,
                           static_cast<const float*>(l.gate_b.device),
                           static_cast<const float*>(l.up_b.device),
                           impl.moe_act, slots, ff, stream);
    gemm::GemvQSlotMxfp4(dn_dev, l.down_exps.RowBytes(), dn_stride,
                    impl.moe_act, impl.expert_ids, impl.moe_down, slots,
                    hidden, ff, stream);
  }
  if (!moe_done) {
    kernels::DownBiasAdd(impl.moe_down, impl.expert_ids,
                         static_cast<const float*>(l.down_b.device), slots,
                         hidden, stream);
    gemm::MoeSlotEpilogueOverwrite(
        impl.moe_down, impl.expert_ids, impl.expert_weights,
        /*dscale=*/nullptr, impl.ffn_out, n, used, hidden, stream);
  }
  TimingMark(LayerSection::kTail);
  kernels::ResidualAdd(impl.ffn_out, hidden_buf,
                       static_cast<std::size_t>(n) * hidden, stream);
  TimingMark(LayerSection::kFinish);  // layer-end edge ev[6] -> ev[7]
  TimingEndLayer(n);
}
void Session::ForwardFinish(Session& s, std::uint32_t n,
                            std::uint32_t start_pos, hipStream_t stream) {
  auto& w = *s.model_->weights_;
  const auto& c = w.config;
  const std::uint32_t hidden = c.hidden_size;
  auto& impl = *s.impl_;
  TimingEnsure();
  TimingMark(LayerSection::kQkv);
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
  TimingMark(LayerSection::kRopeKv);  // finish delta = ev[0] -> ev[1]
  TimingEndFinish(n);
}

bool Session::Forward(std::span<const std::int32_t> tokens,
                      std::uint32_t start_pos, std::string* error_msg) {
  const auto& w = *model_->weights_;
  const std::uint32_t hidden = w.config.hidden_size;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > kMaxChunk) {
    if (error_msg != nullptr)
      *error_msg = "gptoss Forward needs 1..256 tokens";
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
