#include "src/models/laguna/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <atomic>
#include <cerrno>
#include <thread>
#include <vector>

#include "src/core/compute/attention/flash_sink.hpp"
#include "src/core/compute/attention/flash_wmma_gqa.hpp"
#include "src/core/compute/gemm/gemm.hpp" 
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/parallel.hpp"
#include "src/models/laguna/kernels/kernels.hpp"

namespace mbufo::models::laguna {
namespace {

constexpr std::uint32_t kMaxChunk = 512;

bool Check(hipError_t err, std::string* error_msg, const char* what) {
  if (err == hipSuccess) return true;
  if (error_msg != nullptr && error_msg->empty()) {
    *error_msg = std::string("laguna HIP failure in ") + what + ": " +
                 hipGetErrorString(err);
  }
  return false;
}

std::uint16_t F32ToF16Bits(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::int32_t exp =
      static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 112;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;
  if ((bits & 0x7FFFFFFFU) == 0) return static_cast<std::uint16_t>(sign);
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7BFFU);
  if (exp <= 0) return static_cast<std::uint16_t>(sign);
  const std::uint32_t half =
      sign | (static_cast<std::uint32_t>(exp) << 10U) | (mantissa >> 13U);
  const std::uint32_t round_bit = (mantissa >> 12U) & 1U;
  const std::uint32_t sticky = mantissa & 0xFFFU;
  if (round_bit != 0 && (sticky != 0 || (half & 1U) != 0)) {
    return static_cast<std::uint16_t>(half + 1U);
  }
  return static_cast<std::uint16_t>(half);
}


static void DropMappedPages(const void* data, std::size_t bytes) noexcept {
  if (data == nullptr || bytes < 4096) return;
  const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(data);
  const std::uintptr_t start = addr & ~std::uintptr_t(4095);
  const std::size_t len =
      (addr + bytes - start + 4095) & ~std::uintptr_t(4095);
  (void)madvise(reinterpret_cast<void*>(start), len, MADV_DONTNEED);
}


static bool CopyMappedRegion(const core::GgufMappedRegion& source, void* copy,
                             std::string* error_msg) {
  constexpr std::size_t kChunkBytes = 64ULL << 20;
  const std::size_t chunks = (source.size + kChunkBytes - 1) / kChunkBytes;
  try {
    ::mbufo::quant::ParallelForTasks(
        chunks,
        [&](std::size_t idx) {
          const std::size_t offset = idx * kChunkBytes;
          const std::size_t bytes = std::min(kChunkBytes, source.size - offset);
          auto* input = const_cast<std::uint8_t*>(
                            static_cast<const std::uint8_t*>(source.data)) +
                        offset;
#ifdef MADV_POPULATE_READ
          if (madvise(input, bytes, MADV_POPULATE_READ) != 0) {
            throw std::runtime_error("laguna: failed to read mapped GGUF weights");
          }
#endif
          std::memcpy(static_cast<std::uint8_t*>(copy) + offset, input, bytes);
          (void)madvise(input, bytes, MADV_DONTNEED);
        },
        16);
  } catch (const std::exception&) {
    if (error_msg != nullptr) {
      *error_msg = "laguna: failed to read mapped GGUF weights";
    }
    return false;
  }
  return true;
}

static void* ResolveRegisteredDevice(
    const void* host_ptr,
    const std::vector<Model::RegisteredRegion>& regions) noexcept {
  const auto p = reinterpret_cast<std::uintptr_t>(host_ptr);
  for (const auto& r : regions) {
    const auto base = reinterpret_cast<std::uintptr_t>(r.host_data);
    if (p >= base && p < base + r.size) {
      return static_cast<std::uint8_t*>(r.device_data) + (p - base);
    }
  }
  return nullptr;
}

bool MirrorTensor(TensorRef& t,
                  const std::vector<Model::RegisteredRegion>& regions,
                  std::vector<void*>& owned, std::string* error_msg) {
  if (t.empty()) return true;
  // Convert Q6_K routed expert-down weights to Q8_0 on GPU so
  // LaunchDecodeMoeKQuant & RoutedF16Gemm handle every MoE layer uniformly.
  void* mapped_dev = ResolveRegisteredDevice(t.data, regions);
  if (t.type == core::GgmlType::kQ6_K) {
    const std::size_t elems = t.cols * t.rows * t.depth;
    if (elems > 0 && elems % 256 == 0) {
      void* q6k_src = mapped_dev;
      void* q6k_tmp = nullptr;
      if (q6k_src == nullptr) {
        if (!Check(hipMalloc(&q6k_tmp, t.size_bytes), error_msg, "q6k tmp alloc") ||
            !Check(hipMemcpy(q6k_tmp, t.data, t.size_bytes, hipMemcpyHostToDevice),
                   error_msg, "q6k tmp copy")) {
          if (q6k_tmp != nullptr) (void)hipFree(q6k_tmp);
          return false;
        }
        q6k_src = q6k_tmp;
      }
      const std::size_t q80_bytes = (elems / 32) * 34;
      void* dev = nullptr;
      if (!Check(hipMalloc(&dev, q80_bytes), error_msg, "q80 expert alloc")) {
        if (q6k_tmp != nullptr) (void)hipFree(q6k_tmp);
        return false;
      }
      kernels::ConvertQ6KToQ80(q6k_src, dev, elems, nullptr);
      const bool sync_ok =
          Check(hipStreamSynchronize(nullptr), error_msg, "q6k->q80 convert");
      if (q6k_tmp != nullptr) (void)hipFree(q6k_tmp);
      if (!sync_ok) {
        (void)hipFree(dev);
        return false;
      }
      t.device = dev;
      t.type = core::GgmlType::kQ8_0;
      t.size_bytes = q80_bytes;
      owned.push_back(dev);
      return true;
    }
  }
  if (mapped_dev != nullptr && t.type != core::GgmlType::kBF16) {
    t.device = mapped_dev;
    return true;
  }

  void* dev = nullptr;
  if (!Check(hipMalloc(&dev, t.size_bytes), error_msg, "weight mirror alloc")) {
    return false;
  }
  if (!Check(hipMemcpy(dev, t.data, t.size_bytes, hipMemcpyHostToDevice),
             error_msg, "weight mirror copy")) {
    (void)hipFree(dev);
    return false;
  }
  DropMappedPages(t.data, t.size_bytes);
  t.device = dev;
  owned.push_back(dev);
  if (t.type == core::GgmlType::kBF16) {
    kernels::ConvertBf16ToF16(static_cast<const std::uint16_t*>(dev),
                              static_cast<__half*>(dev), t.size_bytes / 2,
                              nullptr);
    if (!Check(hipStreamSynchronize(nullptr), error_msg, "bf16 convert")) {
      return false;
    }
    t.type = core::GgmlType::kF16;
  }
  return true;
}

bool UploadWeights(ModelWeights& w,
                   const std::vector<Model::RegisteredRegion>& regions,
                   std::vector<void*>& owned, std::string* error_msg) {
  if (!MirrorTensor(w.token_embd, regions, owned, error_msg) ||
      !MirrorTensor(w.output_norm, regions, owned, error_msg) ||
      !MirrorTensor(w.output, regions, owned, error_msg)) {
    return false;
  }
  for (auto& l : w.layers) {
    if (!MirrorTensor(l.attn_norm, regions, owned, error_msg) ||
        !MirrorTensor(l.wq, regions, owned, error_msg) ||
        !MirrorTensor(l.wk, regions, owned, error_msg) ||
        !MirrorTensor(l.wv, regions, owned, error_msg) ||
        !MirrorTensor(l.wo, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_q_norm, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_k_norm, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_gate, regions, owned, error_msg) ||
        !MirrorTensor(l.ffn_norm, regions, owned, error_msg)) {
      return false;
    }
    if (!l.is_moe) {
      if (!MirrorTensor(l.ffn_gate, regions, owned, error_msg) ||
          !MirrorTensor(l.ffn_up, regions, owned, error_msg) ||
          !MirrorTensor(l.ffn_down, regions, owned, error_msg)) {
        return false;
      }
    } else {
      if (!MirrorTensor(l.ffn_gate_inp, regions, owned, error_msg) ||
          !MirrorTensor(l.exp_probs_b, regions, owned, error_msg) ||
          !MirrorTensor(l.ffn_gate_exps, regions, owned, error_msg) ||
          !MirrorTensor(l.ffn_up_exps, regions, owned, error_msg) ||
          !MirrorTensor(l.ffn_down_exps, regions, owned, error_msg) ||
          !MirrorTensor(l.ffn_gate_shexp, regions, owned, error_msg) ||
          !MirrorTensor(l.ffn_up_shexp, regions, owned, error_msg) ||
          !MirrorTensor(l.ffn_down_shexp, regions, owned, error_msg)) {
        return false;
      }
    }
  }
  return true;
}

core::compute::gemm::WeightType ToWeightType(core::GgmlType t) {
  return static_cast<core::compute::gemm::WeightType>(static_cast<int>(t));
}



void LagunaGemm(const TensorRef& w, const float* x, __half* x16,
                __half* wdq_scratch, std::size_t wdq_bytes,
                void* preq_scratch, std::size_t preq_bytes, float* y,
                std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                hipStream_t stream) {
  namespace gemm = core::compute::gemm;
  const void* dev = w.device != nullptr ? w.device : w.data;
  if (n_tokens <= 8 &&
      gemm::GemvDecode(dev, static_cast<int>(w.type), x, y, n_tokens, m, k,
                       preq_scratch, preq_bytes, stream) == ::mbufo::core::quant::Status::kOk) {
    return;
  }
  gemm::ConvertF32ToF16(x, x16, static_cast<std::size_t>(n_tokens) * k, stream);
  if (gemm::QuantGemmPrefill(dev, static_cast<int>(w.type), x16, y, n_tokens,
                             m, k, wdq_scratch, wdq_bytes,
                             stream) == ::mbufo::core::quant::Status::kOk) {
    return;
  }
  for (std::uint32_t off = 0; off < n_tokens; off += 8) {
    const std::uint32_t chunk = std::min(8U, n_tokens - off);
    gemm::GemvBatched(dev, static_cast<int>(w.type), w.RowBytes(), k,
                      x + static_cast<std::size_t>(off) * k,
                      y + static_cast<std::size_t>(off) * m,
                      chunk, m, 0.0F, stream);
  }
}

}  // namespace

Model::~Model() {
  for (void* p : device_allocations_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (const auto& r : registered_regions_) {
    if (r.host_copy != nullptr) {
      (void)hipHostUnregister(r.host_copy);
      (void)munmap(r.host_copy, r.size);
    }
  }
}

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  auto reader = core::GgufReader::OpenFile(model_path, error_msg);
  if (reader == nullptr) return nullptr;

  auto weights = ModelWeights::LoadFromGguf(*reader, error_msg);
  if (!weights) return nullptr;

  auto tokenizer = tokenization::QwenTokenizer::CreateFromGguf(*reader, error_msg);
  if (tokenizer == nullptr) return nullptr;

  auto m = std::shared_ptr<Model>(new Model());
  m->options_ = options;
  if (m->options_.max_context == 0 ||
      m->options_.max_context > weights->config.context_length) {
    m->options_.max_context = weights->config.context_length;
  }
  m->reader_ = std::move(reader);
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  m->tokenizer_ = std::move(tokenizer);

  std::size_t total_reg_bytes = 0;
  for (const auto& reg : m->reader_->GetMappedRegions()) total_reg_bytes += reg.size;
  if (total_reg_bytes <= (32ULL << 30))
  for (const auto& reg : m->reader_->GetMappedRegions()) {
    void* host_copy = mmap(nullptr, reg.size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (host_copy == MAP_FAILED) {
      if (error_msg != nullptr) *error_msg = "laguna: mmap host_copy failed";
      return nullptr;
    }
    (void)madvise(host_copy, reg.size, MADV_HUGEPAGE);
    if (!CopyMappedRegion(reg, host_copy, error_msg)) {
      (void)munmap(host_copy, reg.size);
      return nullptr;
    }
    if (!Check(hipHostRegister(host_copy, reg.size,
                               hipHostRegisterMapped | hipHostRegisterReadOnly),
               error_msg, "hipHostRegister")) {
      (void)munmap(host_copy, reg.size);
      return nullptr;
    }
    void* device_data = nullptr;
    if (!Check(hipHostGetDevicePointer(&device_data, host_copy, 0), error_msg,
               "hipHostGetDevicePointer")) {
      (void)hipHostUnregister(host_copy);
      (void)munmap(host_copy, reg.size);
      return nullptr;
    }
    m->registered_regions_.push_back(
        {reg.data, device_data, host_copy, reg.size});
  }
  if (!UploadWeights(*m->weights_, m->registered_regions_,
                     m->device_allocations_, error_msg)) {
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(std::string* error_msg) {
  auto s = std::unique_ptr<Session>(new Session(shared_from_this()));
  if (!s->Init(error_msg)) return nullptr;
  return s;
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text,
                                          bool add_bos) const {
  const auto raw = tokenizer_->Encode(text);
  std::vector<std::int32_t> out;
  out.reserve(raw.size() + (add_bos ? 1 : 0));
  if (add_bos && tokenizer_->GetBosTokenId() != tokenization::kInvalidTokenId) {
    out.push_back(static_cast<std::int32_t>(tokenizer_->GetBosTokenId()));
  }
  for (const auto id : raw) out.push_back(static_cast<std::int32_t>(id));
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids;
  ids.reserve(tokens.size());
  for (const auto t : tokens) ids.push_back(static_cast<tokenization::TokenId>(t));
  return tokenizer_->Decode(ids);
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  const auto id = static_cast<tokenization::TokenId>(token);
  return id == tokenizer_->GetEosTokenId() ||
         id == tokenizer_->GetEosTokenId();
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::string Model::ModelName() const {
  return weights_->config.model_name;
}

std::size_t Model::ResidentBytes() const noexcept {
  std::size_t total = 0;
  for (const auto& l : weights_->layers) {
    total += l.attn_norm.size_bytes + l.wq.size_bytes + l.wk.size_bytes +
             l.wv.size_bytes + l.wo.size_bytes + l.attn_q_norm.size_bytes +
             l.attn_k_norm.size_bytes + l.attn_gate.size_bytes +
             l.ffn_norm.size_bytes + l.ffn_gate.size_bytes +
             l.ffn_up.size_bytes + l.ffn_down.size_bytes +
             l.ffn_gate_inp.size_bytes + l.exp_probs_b.size_bytes +
             l.ffn_gate_exps.size_bytes + l.ffn_up_exps.size_bytes +
             l.ffn_down_exps.size_bytes + l.ffn_gate_shexp.size_bytes +
             l.ffn_up_shexp.size_bytes + l.ffn_down_shexp.size_bytes;
  }
  total += weights_->token_embd.size_bytes + weights_->output_norm.size_bytes +
           weights_->output.size_bytes;
  return total;
}

Session::Session(std::shared_ptr<Model> model) : model_(std::move(model)) {}

Session::~Session() {
  if (stream_ != nullptr) (void)hipStreamSynchronize(stream_);
  for (void* p : k_cache_) if (p != nullptr) (void)hipFree(p);
  for (void* p : v_cache_) if (p != nullptr) (void)hipFree(p);
  auto free_p = [](auto* p) { if (p != nullptr) (void)hipFree(p); };
  free_p(d_tokens_);
  free_p(d_argmax_);
  free_p(d_hidden_);
  free_p(d_normed_);
  free_p(d_q_);
  free_p(d_k_);
  free_p(d_v_);
  free_p(d_gate_logits_);
  free_p(d_attn_out_);
  free_p(d_proj_out_);
  free_p(d_mlp_gate_);
  free_p(d_mlp_up_);
  free_p(d_mlp_out_);
  free_p(d_shared_out_);
  free_p(d_router_logits_);
  free_p(d_exp_ids_);
  free_p(d_exp_weights_);
  free_p(d_act_scratch_);
  free_p(d_down_e_);
  free_p(d_x16_);
  free_p(d_routed_act16_);
  free_p(d_exp_counts_);
  free_p(d_pad_bounds_);
  free_p(d_cursors_);
  free_p(d_rows_token_);
  free_p(d_rows_slot_);
  free_p(d_tiles_);
  free_p(d_n_tiles_);
  free_p(d_tok_flags_);
  free_p(d_wdq_scratch_);
  free_p(d_attn_partials_);
  free_p(d_preq_scratch_);
  free_p(d_logits_);
  if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
}

bool Session::Init(std::string* error_msg) {
  if (!Check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
             error_msg, "stream create")) {
    return false;
  }
  const auto& cfg = model_->weights_->config;
  const std::size_t ctx = model_->MaxContext();
  const std::size_t kv_bytes =
      ctx * cfg.num_kv_heads * cfg.head_dim * sizeof(__half);

  k_cache_.resize(cfg.num_layers, nullptr);
  v_cache_.resize(cfg.num_layers, nullptr);
  for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
    if (!Check(hipMalloc(&k_cache_[l], kv_bytes), error_msg, "k_cache alloc") ||
        !Check(hipMalloc(&v_cache_[l], kv_bytes), error_msg, "v_cache alloc")) {
      return false;
    }
  }

  const std::size_t h = cfg.hidden_size;
  const std::size_t q_dim = static_cast<std::size_t>(cfg.max_q_heads) * cfg.head_dim;
  const std::size_t kv_dim = static_cast<std::size_t>(cfg.num_kv_heads) * cfg.head_dim;
  const std::size_t max_ff = std::max(cfg.intermediate_size, cfg.shared_expert_ffn_size);
  const std::size_t slots = static_cast<std::size_t>(kMaxChunk) * cfg.expert_used_count;
  routed_compact_rows_ =
      core::compute::gemm::RoutedCompactRows(slots, cfg.expert_count);
  tile_cap_ = std::max<std::size_t>(
      2 * slots,
      (slots + static_cast<std::size_t>(cfg.expert_count) * 15) / 16 +
          cfg.expert_count + 1);
  const std::size_t max_dense_elems = std::max<std::size_t>({
      q_dim * h,
      static_cast<std::size_t>(cfg.intermediate_size) * h,
      static_cast<std::size_t>(cfg.shared_expert_ffn_size) * h,
  });
  wdq_scratch_bytes_ = max_dense_elems * sizeof(__half);
  attn_partials_bytes_ = core::compute::flash_sink::FlashSinkDecodeScratchBytes(
      1, cfg.max_q_heads);
  preq_scratch_bytes_ = 256ULL * 1024ULL;

  auto alloc = [&](auto** ptr, std::size_t count, const char* what) -> bool {
    return Check(hipMalloc(reinterpret_cast<void**>(ptr),
                           count * sizeof(**ptr)),
                 error_msg, what);
  };

  if (!alloc(&d_tokens_, kMaxChunk, "d_tokens") ||
      !alloc(&d_argmax_, 1, "d_argmax") ||
      !alloc(&d_hidden_, kMaxChunk * h, "d_hidden") ||
      !alloc(&d_normed_, kMaxChunk * h, "d_normed") ||
      !alloc(&d_q_, kMaxChunk * q_dim, "d_q") ||
      !alloc(&d_k_, kMaxChunk * kv_dim, "d_k") ||
      !alloc(&d_v_, kMaxChunk * kv_dim, "d_v") ||
      !alloc(&d_gate_logits_, kMaxChunk * cfg.max_q_heads, "d_gate_logits") ||
      !alloc(&d_attn_out_, kMaxChunk * q_dim, "d_attn_out") ||
      !alloc(&d_proj_out_, kMaxChunk * h, "d_proj_out") ||
      !alloc(&d_mlp_gate_, kMaxChunk * max_ff, "d_mlp_gate") ||
      !alloc(&d_mlp_up_, kMaxChunk * max_ff, "d_mlp_up") ||
      !alloc(&d_mlp_out_, kMaxChunk * h, "d_mlp_out") ||
      !alloc(&d_shared_out_, kMaxChunk * h, "d_shared_out") ||
      !alloc(&d_router_logits_, kMaxChunk * cfg.expert_count, "d_router_logits") ||
      !alloc(&d_exp_ids_, slots, "d_exp_ids") ||
      !alloc(&d_exp_weights_, slots, "d_exp_weights") ||
      !alloc(&d_act_scratch_, routed_compact_rows_ * cfg.expert_ffn_size, "d_act_scratch") ||
      !alloc(&d_down_e_, slots * h, "d_down_e") ||
      !alloc(&d_x16_, kMaxChunk * std::max(h, q_dim), "d_x16") ||
      !alloc(&d_routed_act16_, routed_compact_rows_ * cfg.expert_ffn_size, "d_routed_act16") ||
      !alloc(&d_exp_counts_, cfg.expert_count, "d_exp_counts") ||
      !alloc(&d_pad_bounds_, cfg.expert_count + 1, "d_pad_bounds") ||
      !alloc(&d_cursors_, cfg.expert_count, "d_cursors") ||
      !alloc(&d_rows_token_, routed_compact_rows_, "d_rows_token") ||
      !alloc(&d_rows_slot_, routed_compact_rows_, "d_rows_slot") ||
      !alloc(&d_tiles_, tile_cap_, "d_tiles") ||
      !alloc(&d_n_tiles_, 1, "d_n_tiles") ||
      !alloc(&d_tok_flags_, kMaxChunk, "d_tok_flags") ||
      !alloc(&d_wdq_scratch_, max_dense_elems, "d_wdq_scratch") ||
      !Check(hipMalloc(reinterpret_cast<void**>(&d_attn_partials_),
                       attn_partials_bytes_),
             error_msg, "d_attn_partials") ||
      !Check(hipMalloc(&d_preq_scratch_, preq_scratch_bytes_), error_msg, "d_preq_scratch") ||
      !alloc(&d_logits_, cfg.vocab_size, "d_logits")) {
    return false;
  }

  host_logits_.resize(cfg.vocab_size, 0.0F);
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.empty()) {
    if (error_msg != nullptr) *error_msg = "laguna Sync: empty prompt";
    return false;
  }
  if (prompt.size() > model_->MaxContext()) {
    if (error_msg != nullptr) *error_msg = "laguna Sync: prompt exceeds context";
    return false;
  }
  std::size_t common = 0;
  while (common < history_.size() && common < prompt.size() &&
         history_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && history_.size() == prompt.size()) {
    return true;
  }
  if (common == prompt.size() && history_.size() > prompt.size()) {
    // Re-evaluate final token so logits_ match prompt.back()
    common = prompt.size() - 1;
  }
  history_.assign(prompt.begin(), prompt.begin() + common);
  return Forward(prompt.subspan(common), static_cast<std::uint32_t>(common),
                 error_msg);
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  const std::int32_t arr[1] = {token};
  return Forward(arr, static_cast<std::uint32_t>(history_.size()), error_msg);
}

bool Session::Forward(std::span<const std::int32_t> tokens,
                      std::uint32_t start_pos, std::string* error_msg) {
  if (tokens.empty()) return true;
  if (start_pos > history_.size()) {
    if (error_msg != nullptr) *error_msg = "laguna Forward: start_pos out of range";
    return false;
  }
  if (start_pos + tokens.size() > model_->MaxContext()) {
    if (error_msg != nullptr) *error_msg = "laguna Forward: context overflow";
    return false;
  }
  history_.resize(start_pos);
  std::size_t off = 0;
  while (off < tokens.size()) {
    const std::uint32_t chunk = static_cast<std::uint32_t>(
        std::min<std::size_t>(kMaxChunk, tokens.size() - off));
    if (!ForwardTokens(tokens.subspan(off, chunk),
                       start_pos + static_cast<std::uint32_t>(off),
                       error_msg)) {
      return false;
    }
    history_.insert(history_.end(), tokens.begin() + off,
                    tokens.begin() + off + chunk);
    off += chunk;
  }
  return true;
}

bool Session::EnqueueForward(std::span<const std::int32_t> tokens,
                             std::uint32_t start_pos, hipStream_t stream,
                             std::string* error_msg) {
  namespace gemm = core::compute::gemm;
  const auto& w = *model_->weights_;
  const auto& cfg = w.config;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  const std::uint32_t h = cfg.hidden_size;

  if (!Check(hipMemcpyAsync(d_tokens_, tokens.data(), n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream),
             error_msg, "copy tokens")) {
    return false;
  }

  gemm::EmbedGather(w.token_embd.device, static_cast<int>(w.token_embd.type),
                    d_tokens_, d_hidden_, n, h, stream);

  const float attn_scale = 1.0F / std::sqrt(static_cast<float>(cfg.head_dim));

  for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
    const auto& lw = w.layers[l];
    const std::uint32_t q_heads = lw.q_heads;
    const std::uint32_t q_dim = q_heads * cfg.head_dim;
    const std::uint32_t kv_dim = cfg.num_kv_heads * cfg.head_dim;

    kernels::RmsNorm(d_hidden_, static_cast<const float*>(lw.attn_norm.device),
                     d_normed_, n, h, cfg.rms_norm_eps, stream);

    LagunaGemm(lw.wq, d_normed_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_q_, n, q_dim, h, stream);
    LagunaGemm(lw.wk, d_normed_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_k_, n, kv_dim, h, stream);
    LagunaGemm(lw.wv, d_normed_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_v_, n, kv_dim, h, stream);
    LagunaGemm(lw.attn_gate, d_normed_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_gate_logits_, n, q_heads, h, stream);

    kernels::HeadRmsNorm(d_q_, static_cast<const float*>(lw.attn_q_norm.device),
                         n, q_heads, cfg.head_dim, cfg.rms_norm_eps, stream);
    kernels::HeadRmsNorm(d_k_, static_cast<const float*>(lw.attn_k_norm.device),
                         n, cfg.num_kv_heads, cfg.head_dim, cfg.rms_norm_eps,
                         stream);

    const std::uint32_t rot_dim = lw.is_swa ? cfg.rotary_dim_swa : cfg.rotary_dim;
    const float theta = lw.is_swa ? cfg.rope_theta_swa : cfg.rope_theta;
    kernels::ApplyDualRope(d_q_, d_k_, n, start_pos, q_heads, cfg.num_kv_heads,
                           cfg.head_dim, rot_dim, theta, lw.is_swa,
                           cfg.yarn_factor, cfg.yarn_orig_ctx,
                           cfg.yarn_beta_fast, cfg.yarn_beta_slow,
                           cfg.yarn_attn_factor, stream);

    auto* kc = static_cast<__half*>(k_cache_[l]);
    auto* vc = static_cast<__half*>(v_cache_[l]);
    kernels::StoreKvF16(d_k_, d_v_, kc, vc, n, start_pos, cfg.num_kv_heads,
                        cfg.head_dim, stream);

    const std::uint32_t swa = lw.is_swa ? cfg.sliding_window : 0U;
    if (n == 1) {
      if (!core::compute::flash_sink::FlashSinkDecode(
              d_q_, kc, vc, /*sinks=*/nullptr, d_attn_out_, 1, start_pos,
              start_pos + 1, q_heads, cfg.num_kv_heads, cfg.head_dim, swa,
              attn_scale, stream, d_attn_partials_, attn_partials_bytes_)) {
        kernels::F16AttentionDecode(d_q_, kc, vc, d_attn_out_, start_pos + 1,
                                    swa, q_heads, cfg.num_kv_heads,
                                    cfg.head_dim, attn_scale, stream);
      }
    } else {
      const bool wmma_ok =
          (n >= 16) &&
          core::compute::flash_wmma_gqa::IsFlashWmmaSupported(
              q_heads, cfg.num_kv_heads, cfg.head_dim) &&
          core::compute::flash_wmma_gqa::FlashWmmaPrefill(
              d_q_, kc, vc, /*sinks=*/nullptr, d_attn_out_, n, start_pos,
              q_heads, cfg.num_kv_heads, cfg.head_dim, swa, attn_scale,
              stream);
      if (!wmma_ok) {
        kernels::F16AttentionPrefill(d_q_, kc, vc, d_attn_out_, n, start_pos,
                                     swa, q_heads, cfg.num_kv_heads,
                                     cfg.head_dim, attn_scale, stream);
      }
    }

    kernels::SoftplusPerHeadGateMul(d_attn_out_, d_gate_logits_, n, q_heads,
                                    cfg.head_dim, stream);
    LagunaGemm(lw.wo, d_attn_out_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_proj_out_, n, h, q_dim, stream);
      kernels::AddResidual(d_hidden_, d_proj_out_,
                         static_cast<std::size_t>(n) * h, stream);

    kernels::RmsNorm(d_hidden_, static_cast<const float*>(lw.ffn_norm.device),
                     d_normed_, n, h, cfg.rms_norm_eps, stream);

    if (!lw.is_moe) {
      const std::uint32_t ff = cfg.intermediate_size;
      LagunaGemm(lw.ffn_gate, d_normed_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_mlp_gate_, n, ff, h, stream);
      LagunaGemm(lw.ffn_up, d_normed_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_mlp_up_, n, ff, h, stream);
      kernels::SiluMul(d_mlp_gate_, d_mlp_up_, d_mlp_gate_,
                       static_cast<std::size_t>(n) * ff, stream);
      LagunaGemm(lw.ffn_down, d_mlp_gate_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_mlp_out_, n, h, ff, stream);
          kernels::AddResidual(d_hidden_, d_mlp_out_,
                           static_cast<std::size_t>(n) * h, stream);
    } else {
      const std::uint32_t sff = cfg.shared_expert_ffn_size;
      LagunaGemm(lw.ffn_gate_shexp, d_normed_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_mlp_gate_, n, sff, h, stream);
      LagunaGemm(lw.ffn_up_shexp, d_normed_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_mlp_up_, n, sff, h, stream);
      kernels::SiluMul(d_mlp_gate_, d_mlp_up_, d_mlp_gate_,
                       static_cast<std::size_t>(n) * sff, stream);
      LagunaGemm(lw.ffn_down_shexp, d_mlp_gate_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_shared_out_, n, h, sff, stream);

      LagunaGemm(lw.ffn_gate_inp, d_normed_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_router_logits_, n, cfg.expert_count, h, stream);
      kernels::SigmoidBiasTopKRouter(
          d_router_logits_, static_cast<const float*>(lw.exp_probs_b.device),
          d_exp_ids_, d_exp_weights_, n, cfg.expert_count,
          cfg.expert_used_count, cfg.expert_weights_norm,
          cfg.expert_weights_scale, stream);

      const bool decode_moe_ok =
          (n <= 8) &&
          gemm::LaunchDecodeMoeKQuant(
              lw.ffn_gate_exps.device, ToWeightType(lw.ffn_gate_exps.type),
              lw.ffn_up_exps.device, lw.ffn_down_exps.device,
              ToWeightType(lw.ffn_down_exps.type), d_normed_, d_exp_ids_,
              d_act_scratch_, d_down_e_, n, cfg.expert_used_count, h,
              cfg.expert_ffn_size, stream);

      if (decode_moe_ok) {
        gemm::MoeEpilogue(d_down_e_, d_exp_weights_, d_mlp_out_, n,
                          cfg.expert_used_count, h, stream);
      } else {
        const std::size_t slots =
            static_cast<std::size_t>(n) * cfg.expert_used_count;
        gemm::ExpertCounts(d_exp_ids_, d_exp_counts_, n, cfg.expert_count,
                           cfg.expert_used_count, stream);
        gemm::RoutedCompact(d_exp_ids_, d_exp_counts_, d_pad_bounds_,
                            d_cursors_, d_rows_token_, d_rows_slot_, n,
                            cfg.expert_used_count, cfg.expert_count, stream);
        const bool small_batch = (n <= 8);
        const std::uint32_t tile_rows =
            small_batch ? 48U
                        : (slots >= static_cast<std::size_t>(16) * cfg.expert_count
                               ? 48U
                               : 16U);
        std::uint32_t n_tiles_host = 0;
        if (small_batch) {
          kernels::FillSlotTiles(d_exp_ids_, d_tiles_, slots, stream);
          n_tiles_host = static_cast<std::uint32_t>(2 * slots);
        } else {
          if (!Check(hipMemsetAsync(d_n_tiles_, 0, sizeof(std::uint32_t),
                                    stream),
                     error_msg, "zero n_tiles")) {
            return false;
          }
          kernels::BuildRoutedTiles(d_exp_counts_, cfg.expert_count, tile_rows,
                                    d_tiles_, d_n_tiles_, stream);
          if (!Check(hipMemcpyAsync(&n_tiles_host, d_n_tiles_,
                                    sizeof(std::uint32_t),
                                    hipMemcpyDeviceToHost, stream),
                     error_msg, "copy n_tiles") ||
              !Check(hipStreamSynchronize(stream), error_msg, "sync n_tiles")) {
            return false;
          }
        }
        const std::uint32_t eff = cfg.expert_ffn_size;
        gemm::ConvertF32ToF16(d_normed_, d_x16_,
                              static_cast<std::size_t>(n) * h, stream);
        gemm::RoutedF16Gemm(
            lw.ffn_gate_exps.device, ToWeightType(lw.ffn_gate_exps.type),
            d_x16_, d_tiles_, n_tiles_host, tile_rows, d_pad_bounds_,
            d_rows_token_, d_rows_slot_, nullptr, d_act_scratch_, nullptr,
            eff, h, stream);
        gemm::RoutedF16Gemm(
            lw.ffn_up_exps.device, ToWeightType(lw.ffn_up_exps.type),
            d_x16_, d_tiles_, n_tiles_host, tile_rows, d_pad_bounds_,
            d_rows_token_, d_rows_slot_, d_act_scratch_, nullptr,
            d_routed_act16_, eff, h, stream);
        gemm::RoutedF16Gemm(
            lw.ffn_down_exps.device, ToWeightType(lw.ffn_down_exps.type),
            d_routed_act16_, d_tiles_, n_tiles_host, tile_rows, d_pad_bounds_,
            d_rows_slot_, d_rows_slot_, nullptr, d_down_e_, nullptr, h, eff,
            stream);
        // The routed GEMMs hand the down projection its input as F16. A
        // token with a massive expert activation (the first token in the
        // last layers: gate ~570 times a large up) overflows F16 to Inf and
        // NaN-poisons that token's hidden state, and through attention every
        // prompt shorter than the sliding window. Recompute just those
        // tokens on the F32 decode MoE path, which keeps the activation in
        // F32; every other token keeps the batched result untouched.
        kernels::FlagNonFiniteTokens(
            d_down_e_, n, static_cast<std::size_t>(cfg.expert_used_count) * h,
            d_tok_flags_, stream);
        std::uint8_t tok_flags[kMaxChunk];
        if (!Check(hipMemcpyAsync(tok_flags, d_tok_flags_, n,
                                  hipMemcpyDeviceToHost, stream),
                   error_msg, "copy token flags") ||
            !Check(hipStreamSynchronize(stream), error_msg,
                   "sync token flags")) {
          return false;
        }
        for (std::uint32_t t = 0; t < n; ++t) {
          if (tok_flags[t] == 0) continue;
          const std::size_t used = cfg.expert_used_count;
          if (!gemm::LaunchDecodeMoeKQuant(
                  lw.ffn_gate_exps.device, ToWeightType(lw.ffn_gate_exps.type),
                  lw.ffn_up_exps.device, lw.ffn_down_exps.device,
                  ToWeightType(lw.ffn_down_exps.type),
                  d_normed_ + static_cast<std::size_t>(t) * h,
                  d_exp_ids_ + t * used, d_act_scratch_ + t * used * eff,
                  d_down_e_ + t * used * h, 1, cfg.expert_used_count, h, eff,
                  stream)) {
            if (error_msg != nullptr) {
              *error_msg = "laguna MoE: F16 routed overflow and no F32 fallback";
            }
            return false;
          }
        }
        gemm::MoeEpilogue(d_down_e_, d_exp_weights_, d_mlp_out_, n,
                          cfg.expert_used_count, h, stream);
      }

          kernels::AddResidual(d_mlp_out_, d_shared_out_,
                           static_cast<std::size_t>(n) * h, stream);
      kernels::AddResidual(d_hidden_, d_mlp_out_,
                           static_cast<std::size_t>(n) * h, stream);
    }
  }

  const float* last_hidden = d_hidden_ + static_cast<std::size_t>(n - 1) * h;
  kernels::RmsNorm(last_hidden,
                   static_cast<const float*>(w.output_norm.device), d_normed_,
                   1, h, cfg.rms_norm_eps, stream);
  LagunaGemm(w.output, d_normed_, d_x16_, d_wdq_scratch_, wdq_scratch_bytes_, d_preq_scratch_, preq_scratch_bytes_, d_logits_, 1, cfg.vocab_size, h, stream);

  return true;
}

bool Session::EnqueueLogitsCopy(hipStream_t stream, std::string* error_msg) {
  const auto& cfg = model_->weights_->config;
  return Check(hipMemcpyAsync(host_logits_.data(), d_logits_,
                              cfg.vocab_size * sizeof(float),
                              hipMemcpyDeviceToHost, stream),
               error_msg, "copy logits");
}

bool Session::ForwardTokens(std::span<const std::int32_t> tokens,
                            std::uint32_t start_pos, std::string* error_msg) {
  if (!EnqueueForward(tokens, start_pos, stream_, error_msg) ||
      !EnqueueLogitsCopy(stream_, error_msg) ||
      !Check(hipStreamSynchronize(stream_), error_msg, "sync forward")) {
    return false;
  }
  return true;
}

static_assert(kMaxChunk == 512, "Session::MaxChunkTokens() must match kMaxChunk");

bool Session::AdvanceBatchWithChunk(
    std::span<Session* const> lanes, std::span<const std::int32_t> tokens,
    Session* chunk, std::span<const std::int32_t> prefix, std::size_t offset,
    std::string* error_msg, bool* refused) {
  const auto refuse = [&](const char* why) {
    if (refused != nullptr) *refused = true;
    if (error_msg != nullptr) *error_msg = std::string("laguna batch refused: ") + why;
    return false;
  };
  if (lanes.size() != tokens.size()) return refuse("lane/token count mismatch");
  if (lanes.empty() && chunk == nullptr) return refuse("nothing to run");
  const Model* model = chunk != nullptr ? chunk->model_.get() : lanes.front()->model_.get();
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    Session* lane = lanes[i];
    if (lane == nullptr || lane == chunk || lane->model_.get() != model)
      return refuse("invalid lane");
    for (std::size_t j = 0; j < i; ++j) {
      if (lanes[j] == lane) return refuse("duplicate lane");
    }
    if (lane->history_.empty() || lane->history_.size() + 1 > model->MaxContext())
      return refuse("lane has no retained prefix or would overflow the context");
  }
  if (chunk != nullptr) {
    if (prefix.size() <= offset || prefix.size() - offset > kMaxChunk ||
        prefix.size() > model->MaxContext() || chunk->history_.size() != offset ||
        !std::equal(chunk->history_.begin(), chunk->history_.end(), prefix.begin())) {
      return refuse("chunk does not continue the retained prefix");
    }
  }

  hipStream_t stream = chunk != nullptr ? chunk->stream_ : lanes.front()->stream_;
  const auto fail = [&]() {
    (void)hipStreamSynchronize(stream);
    return false;
  };
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    if (!lanes[i]->EnqueueForward(tokens.subspan(i, 1),
                                  static_cast<std::uint32_t>(lanes[i]->history_.size()),
                                  stream, error_msg)) {
      return fail();
    }
  }
  if (chunk != nullptr &&
      !chunk->EnqueueForward(prefix.subspan(offset), static_cast<std::uint32_t>(offset),
                             stream, error_msg)) {
    return fail();
  }
  for (Session* lane : lanes) {
    if (!lane->EnqueueLogitsCopy(stream, error_msg)) return fail();
  }
  if (chunk != nullptr && !chunk->EnqueueLogitsCopy(stream, error_msg)) return fail();
  if (!Check(hipStreamSynchronize(stream), error_msg, "sync batch forward")) return false;

  for (std::size_t i = 0; i < lanes.size(); ++i) lanes[i]->history_.push_back(tokens[i]);
  if (chunk != nullptr) chunk->history_.assign(prefix.begin(), prefix.end());
  return true;
}

std::span<const float> Session::Logits() const noexcept {
  return host_logits_;
}

std::uint32_t Session::Length() const noexcept {
  return static_cast<std::uint32_t>(history_.size());
}

std::int32_t Session::SampleGreedy() {
  return kernels::Argmax(d_logits_, model_->VocabSize(), d_argmax_, stream_);
}

}  // namespace mbufo::models::laguna
