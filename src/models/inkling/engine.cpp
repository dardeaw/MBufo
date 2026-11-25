#include "src/models/inkling/engine.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <thread>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/inkling/kernels/kernels.hpp"

namespace mbufo::models::inkling {
namespace {

constexpr std::uint32_t kChunk = 2048;  // mirrors Session::kChunk
constexpr float kAttnScale = 0.08838834764831845F;  // 1/sqrt(128)
constexpr float kResAttnScale = 0.02083333333333333F;  // 1/sqrt(2304), documented choice
constexpr float kQkClip = 22.627416610717773F;
constexpr float kRelClip = 8.0F;
constexpr float kInvSqrt5 = 0.4472135954999579F;
constexpr float kInvSqrt9 = 0.3333333333333333F;

bool Check(hipError_t err, std::string* error_msg, const char* what) {
  if (err == hipSuccess) return true;
  if (error_msg != nullptr && error_msg->empty()) {
    *error_msg = std::string("inkling HIP failure in ") + what + ": " +
                 hipGetErrorString(err);
  }
  return false;
}


inline bool InklingDebugEnabled() {
  static int cached = -1;
  if (cached < 0) {
    const char* e = std::getenv("MBUFO_INKLING_DEBUG");
    cached = (e != nullptr && e[0] == '1') ? 1 : 0;
  }
  return cached == 1;
}

inline void CheckFinite(const char* name, const float* dev_ptr,
                        std::size_t n_elems, hipStream_t stream) {
  if (!InklingDebugEnabled()) return;
  if (dev_ptr == nullptr || n_elems == 0) {
    fprintf(stderr, "[inkling-dbg] %s: null/empty\n", name);
    fflush(stderr);
    return;
  }
  (void)hipStreamSynchronize(stream);
  const std::size_t m = n_elems < 512 ? n_elems : 512;
  std::vector<float> h(m);
  if (hipMemcpy(h.data(), dev_ptr, m * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    fprintf(stderr, "[inkling-dbg] %s: copy fail\n", name);
    fflush(stderr);
    return;
  }
  std::size_t nan = 0, inf = 0;
  float mn = h[0], mx = h[0];
  for (std::size_t i = 0; i < m; ++i) {
    const float v = h[i];
    if (std::isnan(v))
      ++nan;
    else if (std::isinf(v))
      ++inf;
    if (!(v >= mn)) {
      if (v == v) mn = v;
    }
    if (v > mx || std::isnan(mx)) mx = v;
    if (std::isnan(v)) { mn = v; mx = v; }
  }
  // recompute min/max over finite only for readability
  float fmn = 0, fmx = 0; bool have = false;
  for (std::size_t i = 0; i < m; ++i) {
    const float v = h[i];
    if (std::isnan(v) || std::isinf(v)) continue;
    if (!have) { fmn = v; fmx = v; have = true; }
    else { if (v < fmn) fmn = v; if (v > fmx) fmx = v; }
  }
  fprintf(stderr,
          "[inkling-dbg] %s: n=%zu min=%g max=%g fmin=%g fmax=%g nan=%zu inf=%zu "
          "h0=%g h1=%g h2=%g h3=%g\n",
          name, n_elems, (double)mn, (double)mx,
          have ? (double)fmn : 0.0, have ? (double)fmx : 0.0, nan, inf,
          (double)h[0], m > 1 ? (double)h[1] : 0.0,
          m > 2 ? (double)h[2] : 0.0, m > 3 ? (double)h[3] : 0.0);
  fflush(stderr);
}

inline void CheckFiniteHost(const char* name, const float* h, std::size_t n) {
  if (!InklingDebugEnabled()) return;
  if (h == nullptr || n == 0) {
    fprintf(stderr, "[inkling-dbg] %s(HOST): null/empty\n", name);
    fflush(stderr);
    return;
  }
  const std::size_t m = n < 512 ? n : 512;
  std::size_t nan = 0, inf = 0;
  float mn = h[0], mx = h[0];
  for (std::size_t i = 0; i < m; ++i) {
    const float v = h[i];
    if (std::isnan(v)) ++nan;
    else if (std::isinf(v)) ++inf;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    if (std::isnan(v)) { mn = v; mx = v; }
  }
  float fmn = 0, fmx = 0; bool have = false;
  for (std::size_t i = 0; i < m; ++i) {
    const float v = h[i];
    if (std::isnan(v) || std::isinf(v)) continue;
    if (!have) { fmn = v; fmx = v; have = true; }
    else { if (v < fmn) fmn = v; if (v > fmx) fmx = v; }
  }
  fprintf(stderr,
          "[inkling-dbg] %s(HOST): n=%zu min=%g max=%g fmin=%g fmax=%g nan=%zu "
          "inf=%zu h0=%g h1=%g\n",
          name, n, (double)mn, (double)mx,
          have ? (double)fmn : 0.0, have ? (double)fmx : 0.0, nan, inf,
          (double)h[0], m > 1 ? (double)h[1] : 0.0);
  fflush(stderr);
}

inline bool ParityEnabled() {
  static int cached = -1;
  if (cached < 0) {
    const char* e = std::getenv("MBUFO_INKLING_PARITY");
    cached = (e != nullptr && e[0] == '1') ? 1 : 0;
  }
  return cached == 1;
}

inline bool ProfileEnabled() {
  static int cached = -1;
  if (cached < 0) {
    const char* e = std::getenv("MBUFO_INKLING_PROFILE");
    cached = (e != nullptr && e[0] == '1') ? 1 : 0;
  }
  return cached == 1;
}

struct ProfStats {
  double res_qkv_ms = 0;
  double attn_ms = 0;
  double post_attn_ms = 0;
  double moe_router_ms = 0;
  double moe_exps_ms = 0;
  double post_moe_ms = 0;
  double lm_head_ms = 0;
  double fuse_hidden_ms = 0;
  double total_ms = 0;
  std::uint32_t last_n = 0;
  void Reset(std::uint32_t n) {
    res_qkv_ms = attn_ms = post_attn_ms = moe_router_ms = moe_exps_ms =
        post_moe_ms = lm_head_ms = fuse_hidden_ms = total_ms = 0;
    last_n = n;
  }
  void Dump() const {
    fprintf(stderr,
            "[inkling-prof] n=%u total=%.2fms | res_qkv=%.2f attn=%.2f "
            "post_attn=%.2f moe_router=%.2f moe_exps=%.2f post_moe=%.2f "
            "lm_head=%.2f (fuse_hidden=%.2f)\n",
            last_n, total_ms, res_qkv_ms, attn_ms, post_attn_ms, moe_router_ms,
            moe_exps_ms, post_moe_ms, lm_head_ms, fuse_hidden_ms);
    fflush(stderr);
  }
};
inline thread_local ProfStats g_prof;

inline double ProfTick(hipStream_t stream,
                       std::chrono::steady_clock::time_point* prev) {
  if (!ProfileEnabled()) return 0.0;
  (void)hipStreamSynchronize(stream);
  auto now = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(now - *prev).count();
  *prev = now;
  return ms;
}

inline void ParityDump(const char* name, const float* dev_ptr,
                       std::size_t n_tokens, std::size_t h,
                       hipStream_t stream) {
  if (!ParityEnabled()) return;
  if (dev_ptr == nullptr || n_tokens == 0 || h == 0) return;
  (void)hipStreamSynchronize(stream);
  const float* last = dev_ptr + (n_tokens - 1) * h;
  std::vector<float> row(h);
  if (hipMemcpy(row.data(), last, h * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    fprintf(stderr, "[inkling-parity] %s: copy fail\n", name);
    fflush(stderr);
    return;
  }
  double sum = 0.0;
  float absmax = 0.0F;
  for (std::size_t i = 0; i < h; ++i) {
    sum += static_cast<double>(row[i]);
    const float a = std::fabs(row[i]);
    if (a > absmax) absmax = a;
  }
  const double mean = sum / static_cast<double>(h);
  fprintf(stderr, "[inkling-parity] %s: n=%zu h=%zu mean=%.6g absmax=%.6g first8=%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n",
          name, n_tokens, h, mean, static_cast<double>(absmax),
           static_cast<double>(row[0]), static_cast<double>(row[1]),
           static_cast<double>(row[2]), static_cast<double>(row[3]),
           static_cast<double>(row[4]), static_cast<double>(row[5]),
           static_cast<double>(row[6]), static_cast<double>(row[7]));
  fflush(stderr);
}

// Full last-token row (space-separated) for pattern analysis across channels.
inline void ParityDumpFull(const char* name, const float* dev_ptr,
                           std::size_t n_tokens, std::uint32_t h,
                           hipStream_t stream) {
  if (!ParityEnabled()) return;
  if (dev_ptr == nullptr || n_tokens == 0 || h == 0) return;
  (void)hipStreamSynchronize(stream);
  const float* last = dev_ptr + (n_tokens - 1) * h;
  std::vector<float> row(h);
  if (hipMemcpy(row.data(), last, h * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return;
  }
  fprintf(stderr, "[inkling-parity-full] %s: n=%zu h=%u", name, n_tokens, h);
  for (std::uint32_t i = 0; i < h; ++i) {
    fprintf(stderr, " %.6g", static_cast<double>(row[i]));
  }
  fprintf(stderr, "\n");
  fflush(stderr);
}

// Component parity for layers 4/5 (L0 keeps its legacy dbg-L0-* names).
inline void DbgDump(std::uint32_t layer, const char* tag, const float* dev_ptr,
                    std::size_t n_tokens, std::uint32_t h,
                    hipStream_t stream) {
  if (layer == 0 || layer > 23) return;
  char nm[64];
  snprintf(nm, sizeof(nm), "dbg-L%u-%s", layer, tag);
  ParityDump(nm, dev_ptr, n_tokens, h, stream);
}

// Full [n, h] matrix dump (token-major) for forensic least-squares.
inline void ParityDumpFullM(const char* name, const float* dev_ptr,
                            std::size_t n_tokens, std::uint32_t h,
                            hipStream_t stream) {
  if (!ParityEnabled()) return;
  if (dev_ptr == nullptr || n_tokens == 0 || h == 0) return;
  (void)hipStreamSynchronize(stream);
  const std::size_t count = n_tokens * h;
  std::vector<float> buf(count);
  if (hipMemcpy(buf.data(), dev_ptr, count * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return;
  }
  fprintf(stderr, "[inkling-parity-fullm] %s: n=%zu h=%u", name, n_tokens, h);
  for (std::size_t i = 0; i < count; ++i) {
    fprintf(stderr, " %.6g", static_cast<double>(buf[i]));
  }
  fprintf(stderr, "\n");
  fflush(stderr);
}
inline void DbgDumpFullM(std::uint32_t layer, const char* tag,
                         const float* dev_ptr, std::size_t n_tokens,
                         std::uint32_t h, hipStream_t stream) {
  if (layer == 0 || layer > 23) return;
  char nm[64];
  snprintf(nm, sizeof(nm), "dbg-L%u-%s", layer, tag);
  ParityDumpFullM(nm, dev_ptr, n_tokens, h, stream);
}
// Full last-row variant of DbgDump for pattern analysis (L4/L5 only).
inline void DbgDumpFull(std::uint32_t layer, const char* tag,
                        const float* dev_ptr, std::size_t n_tokens,
                        std::uint32_t h, hipStream_t stream) {
  if (layer == 0 || layer > 23) return;
  char nm[64];
  snprintf(nm, sizeof(nm), "dbg-L%u-%s", layer, tag);
  ParityDumpFull(nm, dev_ptr, n_tokens, h, stream);
}

// Router decision for the last token (L4/L5): top-8 ids + weights + logits.
inline void DbgRouterDump(std::uint32_t layer, const std::int32_t* dev_ids,
                          const float* dev_w, std::uint32_t n_tokens,
                          std::uint32_t top_k, const float* dev_logits,
                          std::uint32_t n_experts, const float* dev_bias,
                          hipStream_t stream) {
  if (!ParityEnabled()) return;
  if (layer == 0 || layer > 23) return;
  if (n_tokens == 0) return;
  (void)hipStreamSynchronize(stream);
  const std::size_t last = static_cast<std::size_t>(n_tokens) - 1;
  std::vector<std::int32_t> ids(top_k, -1);
  std::vector<float> w(top_k, 0.0F);
  std::vector<float> logits(n_experts, 0.0F);
  if (hipMemcpy(ids.data(), dev_ids + last * top_k, top_k * sizeof(std::int32_t),
                hipMemcpyDeviceToHost) != hipSuccess) return;
  if (hipMemcpy(w.data(), dev_w + last * top_k, top_k * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) return;
  if (hipMemcpy(logits.data(), dev_logits + last * n_experts,
                n_experts * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return;
  DbgDumpFull(layer, "rlogits", dev_logits, n_tokens, n_experts, stream);
  if (dev_bias != nullptr) {
    DbgDumpFullM(layer, "m-expb", dev_bias, 1, n_experts, stream);
  }
  double lsum = 0.0;
  float labsmax = 0.0F;
  for (std::uint32_t e = 0; e < n_experts; ++e) {
    lsum += static_cast<double>(logits[e]);
    const float a = std::fabs(logits[e]);
    if (a > labsmax) labsmax = a;
  }
  fprintf(stderr,
          "[inkling-parity] dbg-L%u-router: n=%u k=%u logitmean=%.6g logitabsmax=%.6g ids=%d %d %d %d %d %d %d %d w=%.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g\n",
          layer, n_tokens, top_k, lsum / static_cast<double>(n_experts),
          static_cast<double>(labsmax), ids[0], ids[1], ids[2], ids[3],
          ids[4], ids[5], ids[6], ids[7], static_cast<double>(w[0]),
          static_cast<double>(w[1]), static_cast<double>(w[2]),
          static_cast<double>(w[3]), static_cast<double>(w[4]),
          static_cast<double>(w[5]), static_cast<double>(w[6]),
          static_cast<double>(w[7]));
  fflush(stderr);
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

std::uint16_t Bf16ToF16Bits(std::uint16_t b) noexcept {
  std::uint32_t f32 = static_cast<std::uint32_t>(b) << 16U;
  float f = 0.0F;
  std::memcpy(&f, &f32, sizeof(f));
  return F32ToF16Bits(f);
}

void DropMappedPages(const void* data, std::size_t bytes) noexcept {
  if (data == nullptr || bytes < 4096) return;
  const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(data);
  const std::uintptr_t start = addr & ~std::uintptr_t(4095);
  const std::size_t len =
      (addr + bytes - start + 4095) & ~std::uintptr_t(4095);
  (void)madvise(reinterpret_cast<void*>(start), len, MADV_DONTNEED);
}

bool CopyMappedRegion(const core::GgufMappedRegion& source, void* copy,
                      std::string* error_msg) {
  constexpr std::size_t kChunkBytes = 64ULL << 20;
  const std::size_t chunks = (source.size + kChunkBytes - 1) / kChunkBytes;
  std::atomic<std::size_t> next{0};
  std::atomic<int> failure{0};
  auto worker = [&]() {
    while (failure.load(std::memory_order_relaxed) == 0) {
      const std::size_t idx = next.fetch_add(1, std::memory_order_relaxed);
      if (idx >= chunks) break;
      const std::size_t offset = idx * kChunkBytes;
      const std::size_t bytes = std::min(kChunkBytes, source.size - offset);
      auto* input = const_cast<std::uint8_t*>(
                        static_cast<const std::uint8_t*>(source.data)) +
                    offset;
#ifdef MADV_POPULATE_READ
      if (madvise(input, bytes, MADV_POPULATE_READ) != 0) {
        failure.store(errno != 0 ? errno : EIO, std::memory_order_relaxed);
        break;
      }
#endif
      std::memcpy(static_cast<std::uint8_t*>(copy) + offset, input, bytes);
      (void)madvise(input, bytes, MADV_DONTNEED);
    }
  };
  const std::size_t n_threads =
      std::min<std::size_t>(16, std::max<std::size_t>(1, chunks));
  std::vector<std::jthread> workers;
  workers.reserve(n_threads > 0 ? n_threads - 1 : 0);
  for (std::size_t i = 1; i < n_threads; ++i) {
    workers.emplace_back(worker);
  }
  worker();
  workers.clear();
  if (failure.load() != 0) {
    if (error_msg != nullptr) {
      *error_msg = "inkling: failed to read mapped GGUF weights";
    }
    return false;
  }
  return true;
}

void* ResolveRegisteredDevice(
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

// Load-time CPU casts for formats core GEMM does not consume directly:
// NVFP4 experts (blk.0-19) -> Q8_0 (see ConvertNvfp4ToQ80), BF16
// projections and Q8_0 dense tensors -> F16. Converted buffers are appended
// to `owned` (caller keeps them alive in the Model) and the TensorRef is
// repointed at the converted copy.
bool ConvertTensorToF16(TensorRef& t, const TensorRef* scale_or_null,
                        std::vector<std::vector<std::uint16_t>>* owned,
                        std::string* error_msg) {
  if (t.empty()) return true;
  const std::size_t elems =
      static_cast<std::size_t>(t.cols) * t.rows * t.depth;
  if (elems == 0) {
    if (error_msg != nullptr) *error_msg = "inkling: empty tensor convert";
    return false;
  }
  std::vector<std::uint16_t> buf(elems);
  if (IsNvfp4(t.type)) {
    const float* scales = nullptr;
    if (scale_or_null != nullptr && !scale_or_null->empty()) {
      if (scale_or_null->type != core::GgmlType::kF32 ||
          scale_or_null->rows * scale_or_null->cols != t.depth) {
        if (error_msg != nullptr) {
          *error_msg = "inkling: bad NVFP4 per-expert scale";
        }
        return false;
      }
      scales = static_cast<const float*>(scale_or_null->data);
    }
    if (!::mbufo::quant::DequantizeNvfp4ToF16(
            t.data, scales, buf.data(), elems,
            static_cast<std::uint32_t>(t.depth))) {
      if (error_msg != nullptr) *error_msg = "inkling: NVFP4 convert failed";
      return false;
    }
  } else if (t.type == core::GgmlType::kF32) {
    const auto* src = static_cast<const float*>(t.data);
    for (std::size_t i = 0; i < elems; ++i) buf[i] = F32ToF16Bits(src[i]);
  } else if (t.type == core::GgmlType::kBF16) {
    const auto* src = static_cast<const std::uint16_t*>(t.data);
    for (std::size_t i = 0; i < elems; ++i) buf[i] = Bf16ToF16Bits(src[i]);
  } else if (t.type == core::GgmlType::kF16) {
    return true;  // already converted
  } else if (t.type == core::GgmlType::kQ8_0) {
    // Inkling Q8_0 (token_embd, output, Q/K/V/G/O, rel) decodes on the host
    // to F16 once at load: host DequantizeQ8_0 (FP16 scale, verified against
    // python GGUF Q8_0 hand-dequant: token100 first8 small, max scale 0.09)
    // is the reference; the GPU Q8_0 narrow/prefill path misreads these
    // tensors on this model (embed-norm -4854/16425, L0-attn-out-raw 0,
    // gate/wo NaN), so bypass it entirely and run dense F16.
    // Streaming decode in 1 Mi-elem blocks with a reused scratch buffer: a
    // full-elems float tmp costs 4x transient host memory (a 648 MiB F16
    // tensor needs a 1.3 GiB tmp), a major peak contributor in the 2026-11-23
    // load wedge. DequantizeQ8_0 is block-local (32 elems/block), so chunking
    // on 32-elem boundaries is bit-identical. Elems past the last full block
    // keep the old value-init (zero) behavior.
    constexpr std::size_t kDeqChunkElems = 1ULL << 20;
    static_assert((kDeqChunkElems % 32) == 0, "Q8_0 block is 32 elems");
    const std::size_t kQ8BlockBytes = sizeof(::mbufo::quant::block_q8_0);
    std::vector<float> tmp(std::min(kDeqChunkElems, elems));
    std::size_t done = 0;
    const std::size_t full = (elems / 32) * 32;
    while (done < full) {
      const std::size_t n = std::min(kDeqChunkElems, full - done);
      const auto* blk = static_cast<const std::uint8_t*>(t.data) +
                        (done / 32) * kQ8BlockBytes;
      ::mbufo::quant::DequantizeQ8_0(blk, tmp.data(), n);
      for (std::size_t i = 0; i < n; ++i) buf[done + i] = F32ToF16Bits(tmp[i]);
      done += n;
    }
    for (; done < elems; ++done) buf[done] = F32ToF16Bits(0.0F);
  } else if (t.type == core::GgmlType::kQ4_K ||
             t.type == core::GgmlType::kQ5_K ||
             t.type == core::GgmlType::kQ6_K) {
    const auto kind =
        (t.type == core::GgmlType::kQ4_K)
            ? ::mbufo::quant::KQuantKind::kQ4_K
            : ((t.type == core::GgmlType::kQ5_K)
                   ? ::mbufo::quant::KQuantKind::kQ5_K
                   : ::mbufo::quant::KQuantKind::kQ6_K);
    if (!::mbufo::quant::ConvertKQuantToF16(kind, t.data, buf.data(), elems)) {
      if (error_msg != nullptr) *error_msg = "inkling: K-quant F16 convert failed";
      return false;
    }
  } else {
    if (error_msg != nullptr) {
      *error_msg = "inkling: unexpected type in load conversion";
    }
    return false;
  }
  owned->push_back(std::move(buf));
  t.data = owned->back().data();
  t.type = core::GgmlType::kF16;
  t.size_bytes = elems * 2;
  return true;
}

// NVFP4 experts (blk.0-19) -> Q8_0. Host dequant NVFP4->F32 applies the
// per-expert F32 scale (outer scale per expert, inner UE4M3 per 16-elem
// sub-block), then core stock QuantizeQ8_0 (symmetric per-32 amax/127; the
// Q8 step is ~64x finer than the 4-bit source step, so the round-trip error
// is negligible). Decoding runs expert-by-expert into a transient F32
// scratch (one expert slab at a time, freed per tensor): the NVFP4 layout
// is expert-major, so slicing on expert boundaries is exact, and expert
// slabs (hidden*ffn_size elems) keep the transient peak in the tens of MiB
// instead of a full-tensor 4x float copy. Converted bytes are appended to
// `owned_q80`; the TensorRef is repointed at the Q8_0 copy with matching
// size_bytes so RowBytes()/ExpertStrideBytes() stay exact and the MoE
// consume side rides its existing type-driven Q8 path untouched.

bool ConvertNvfp4ToQ40(TensorRef& t, const TensorRef* scale_or_null,
                       std::vector<std::vector<std::uint8_t>>* owned_q40,
                       std::string* error_msg) {
  if (t.empty()) return true;
  if (!IsNvfp4(t.type)) {
    if (error_msg != nullptr) *error_msg = "inkling: ConvertNvfp4ToQ40 got non-NVFP4 tensor";
    return false;
  }
  const std::size_t elems = static_cast<std::size_t>(t.cols) * t.rows * t.depth;
  const float* scales = (scale_or_null != nullptr && !scale_or_null->empty()) ? static_cast<const float*>(scale_or_null->data) : nullptr;
  std::vector<std::uint8_t> out((elems / 32) * 18ULL);
  if (!::mbufo::quant::ConvertNvfp4ToQ4_0(t.data, scales, out.data(), elems, static_cast<std::uint32_t>(t.depth))) {
    if (error_msg != nullptr) *error_msg = "inkling: NVFP4->Q4_0 decode failed";
    return false;
  }
  owned_q40->push_back(std::move(out));
  t.data = owned_q40->back().data();
  t.type = core::GgmlType::kQ4_0;
  t.size_bytes = owned_q40->back().size();
  return true;
}

bool ConvertNvfp4ToQ80(TensorRef& t, const TensorRef* scale_or_null,
                       std::vector<std::vector<std::uint8_t>>* owned_q80,
                       std::string* error_msg) {
  if (t.empty()) return true;
  if (!IsNvfp4(t.type)) {
    if (error_msg != nullptr) *error_msg = "inkling: ConvertNvfp4ToQ80 got non-NVFP4 tensor";
    return false;
  }
  const std::size_t elems = static_cast<std::size_t>(t.cols) * t.rows * t.depth;
  const float* scales = (scale_or_null != nullptr && !scale_or_null->empty()) ? static_cast<const float*>(scale_or_null->data) : nullptr;
  std::vector<std::uint8_t> buf((elems / 32) * sizeof(::mbufo::quant::block_q8_0));
  if (!::mbufo::quant::ConvertNvfp4ToQ8_0(t.data, scales, buf.data(), elems, static_cast<std::uint32_t>(t.depth))) {
    if (error_msg != nullptr) *error_msg = "inkling: NVFP4->Q8_0 decode failed";
    return false;
  }
  owned_q80->push_back(std::move(buf));
  t.data = owned_q80->back().data();
  t.type = core::GgmlType::kQ8_0;
  t.size_bytes = owned_q80->back().size();
  return true;
}

// GGUF attn_rel_bias is stored [1024,16]-major (bit-exact vs the HF
// [16,1024] transpose, verified against safetensors); the attention kernel
// reads reference [16,1024] layout, so transpose once at load into an owned
// heap buffer and repoint the TensorRef (mirrored to device by the standard
// heap-region path, like converted_f16_).
bool ConvertKQuantToQ80(TensorRef& t,
                        std::vector<std::vector<std::uint8_t>>* owned_q80,
                        std::string* error_msg) {
  if (t.empty()) return true;
  ::mbufo::quant::KQuantKind kind = (t.type == core::GgmlType::kQ4_K) ? ::mbufo::quant::KQuantKind::kQ4_K
      : ((t.type == core::GgmlType::kQ5_K) ? ::mbufo::quant::KQuantKind::kQ5_K : ::mbufo::quant::KQuantKind::kQ6_K);
  const std::size_t total_rows = static_cast<std::size_t>(t.rows) * t.depth;
  std::vector<std::uint8_t> buf(total_rows * (t.cols / 32) * sizeof(::mbufo::quant::block_q8_0));
  if (!::mbufo::quant::ConvertKQuantToQ8_0(kind, t.data, t.RowBytes(), buf.data(), total_rows, t.cols)) {
    if (error_msg != nullptr) *error_msg = "inkling: K-quant->Q8_0 failed";
    return false;
  }
  owned_q80->push_back(std::move(buf));
  t.data = owned_q80->back().data();
  t.type = core::GgmlType::kQ8_0;
  t.size_bytes = owned_q80->back().size();
  return true;
}

bool TransposeRelBias(ModelWeights& w,
                      std::vector<std::vector<float>>* owned,
                      std::string* error_msg) {
  for (auto& l : w.layers) {
    auto& t = l.attn_rel_bias;
    if (t.empty()) continue;
    if (t.type != core::GgmlType::kF32 || t.cols != 16 || t.rows != 1024 ||
        t.size_bytes != 16 * 1024 * sizeof(float)) {
      if (error_msg != nullptr) {
        *error_msg = "inkling: unexpected attn_rel_bias layout";
      }
      return false;
    }
    // Kernel now reads native GGUF [1024, 16] layout directly (dist * 16 + e).
    (void)owned;
  }
  return true;
}

bool ConvertLoadTimeFormats(ModelWeights& w,
                            std::vector<std::vector<std::uint16_t>>* owned,
                            std::vector<std::vector<std::uint8_t>>* owned_q80,
                            std::string* error_msg) {
  // Q8_0 dense path (token_embd, output, Q/K/V/G/O, rel) is bypassed on
  // inkling: host DequantizeQ8_0 (verified small: token100 first8, max scale
  // 0.09) -> F16, then dense F16 GEMM/EmbedGather. Fixes embed-norm
  // -4854/16425, L0-attn-out-raw 0, gate/wo NaN without touching core calls.
  // NVFP4 experts (blk.0-19) go NVFP4->Q8_0 instead (ConvertNvfp4ToQ80);
  // Q8_0 experts (blk.20-23) keep the existing Q8_0->F16 path untouched.
  auto convert_experts = [&](TensorRef& t, const TensorRef* scale) -> bool {
    if (IsNvfp4(t.type)) {
      return ConvertNvfp4ToQ40(t, scale, owned_q80, error_msg);
    }
    if (t.type == core::GgmlType::kQ8_0 || t.type == core::GgmlType::kQ4_0) {
      return true;
    }
    if (t.type == core::GgmlType::kQ4_K || t.type == core::GgmlType::kQ5_K ||
        t.type == core::GgmlType::kQ6_K) {
      return ConvertKQuantToQ80(t, owned_q80, error_msg);
    }
    return ConvertTensorToF16(t, scale, owned, error_msg);
  };
  auto keep_q8_or_f16 = [&](TensorRef& t) -> bool {
    if (t.type == core::GgmlType::kQ8_0 || t.type == core::GgmlType::kQ6_K ||
        t.type == core::GgmlType::kQ5_K || t.type == core::GgmlType::kQ4_K) {
      return true;
    }
    return ConvertTensorToF16(t, nullptr, owned, error_msg);
  };
  if (!ConvertTensorToF16(w.token_embd, nullptr, owned, error_msg) ||
      !ConvertTensorToF16(w.output, nullptr, owned, error_msg)) {
    return false;
  }
  for (auto& l : w.layers) {
    if (!keep_q8_or_f16(l.wq) ||
        !keep_q8_or_f16(l.wk) ||
        !keep_q8_or_f16(l.wv) ||
        !keep_q8_or_f16(l.wo) ||
        !keep_q8_or_f16(l.attn_gate) ||
        !keep_q8_or_f16(l.attn_rel) ||
        !convert_experts(l.ffn_gate_up_exps, &l.ffn_gate_up_scale) ||
        !convert_experts(l.ffn_down_exps, &l.ffn_down_scale) ||
        !keep_q8_or_f16(l.attn_sink_proj) ||
        !keep_q8_or_f16(l.attn_k_conv_proj) ||
        !keep_q8_or_f16(l.attn_v_conv_proj)) {
      return false;
    }
    for (auto* lt : {&l.loop_attn, &l.loop_k, &l.loop_mlp, &l.loop_v}) {
      if (!ConvertTensorToF16(lt->down, nullptr, owned, error_msg) ||
          !ConvertTensorToF16(lt->forget, nullptr, owned, error_msg) ||
          !ConvertTensorToF16(lt->gain, nullptr, owned, error_msg)) {
        return false;
      }
    }
  }
  return true;
}

// Converted F16/Q8_0 buffers live on the heap, so the reader-region zero-copy
// lookup cannot see them. Register each buffer host-pinned + mapped (the
// qwen model_loader recipe) and publish a heap-backed region, keeping the
// large expert slabs off device-local hipMalloc: this box carries only a
// 512 MiB VRAM carveout and the ROCm runtime segfaults inside
// deviceLocalAlloc on GB-scale hipMalloc.
template <typename Elem>
bool RegisterConvertedBuffers(
    std::vector<std::vector<Elem>>* owned,
    std::vector<Model::RegisteredRegion>* regions, std::string* error_msg) {
  for (auto& buf : *owned) {
    if (buf.empty()) continue;
    void* host = static_cast<void*>(buf.data());
    const std::size_t bytes = buf.size() * sizeof(Elem);
    if (!Check(hipHostRegister(host, bytes,
                               hipHostRegisterMapped | hipHostRegisterReadOnly),
               error_msg, "converted weight register")) {
      return false;
    }
    void* dev = nullptr;
    if (!Check(hipHostGetDevicePointer(&dev, host, 0), error_msg,
               "converted weight device pointer")) {
      (void)hipHostUnregister(host);
      return false;
    }
    Model::RegisteredRegion reg;
    reg.host_data = host;
    reg.device_data = dev;
    reg.host_copy = host;
    reg.size = bytes;
    reg.heap_backed = true;
    regions->push_back(reg);
  }
  return true;
}

bool MirrorDeviceCopy(TensorRef& t, std::vector<void*>& owned,
                        bool keep_host_pages, std::string* error_msg) {
  // Serial index across all mirror calls in this load. The 2026-11-23 wedge
  // hung inside a single 648 MiB hipMemcpy H2D with no progress marker, so
  // every tensor and every 64 MiB slice is logged to stderr (unbuffered).
  // Copy semantics are unchanged: one hipMalloc + sequential H2D slices.
  static std::atomic<std::size_t> g_mirror_idx{0};
  const std::size_t idx = g_mirror_idx.fetch_add(1, std::memory_order_relaxed);
  constexpr std::size_t kCopyChunk = 64ULL << 20;
  const double mib = static_cast<double>(t.size_bytes) / (1 << 20);
  fprintf(stderr, "[inkling-load] mirror #%zu size=%.1fMiB keep_host=%d\n",
          idx, mib, keep_host_pages ? 1 : 0);
  fflush(stderr);
  void* dev = nullptr;
  if (!Check(hipMalloc(&dev, t.size_bytes), error_msg, "weight mirror alloc")) {
    return false;
  }
  const auto* src = static_cast<const std::uint8_t*>(t.data);
  auto* dst = static_cast<std::uint8_t*>(dev);
  const std::size_t n_chunks = (t.size_bytes + kCopyChunk - 1) / kCopyChunk;
  std::size_t off = 0;
  std::size_t chunk_no = 0;
  while (off < t.size_bytes) {
    const std::size_t bytes = std::min(kCopyChunk, t.size_bytes - off);
    if (!Check(hipMemcpy(dst + off, src + off, bytes, hipMemcpyHostToDevice),
               error_msg, "weight mirror copy")) {
      fprintf(stderr, "[inkling-load] mirror #%zu FAILED chunk %zu/%zu off=%zu\n",
              idx, chunk_no, n_chunks, off);
      fflush(stderr);
      (void)hipFree(dev);
      return false;
    }
    off += bytes;
    ++chunk_no;
    if (n_chunks > 1) {
      fprintf(stderr, "[inkling-load] mirror #%zu chunk %zu/%zu done\n", idx,
              chunk_no, n_chunks);
      fflush(stderr);
    }
  }
  if (!keep_host_pages) DropMappedPages(t.data, t.size_bytes);
  t.device = dev;
  owned.push_back(dev);
  fprintf(stderr, "[inkling-load] mirror #%zu complete\n", idx);
  fflush(stderr);
  return true;
}

bool MirrorTensor(TensorRef& t,
                  const std::vector<Model::RegisteredRegion>& regions,
                  std::vector<void*>& owned, std::string* error_msg) {
  if (t.empty()) return true;
  // Load-converted F16 tensors live in heap buffers whose device mapping is
  // a host identity pointer. Core GEMM (hipBLASLt F16 dense) fails
  // shape-dependently on such pointers (e.g. wo/wv NaN while wq is fine), so
  // converted tensors are mirrored to device-local memory. They match their
  // heap region by exact base address only: heap ranges can numerically
  // shadow reader ranges in a range scan, and converted tensors always point
  // at their buffer start.
  for (const auto& r : regions) {
    if (r.heap_backed && r.host_data == t.data) {
      return MirrorDeviceCopy(t, owned, /*keep_host_pages=*/true, error_msg);
    }
  }
  // Reader-backed tensors: range-scan reader regions only (heap regions never
  // participate, so no shadowing either way).
  for (const auto& r : regions) {
    if (r.heap_backed) continue;
    const auto p = reinterpret_cast<std::uintptr_t>(t.data);
    const auto base = reinterpret_cast<std::uintptr_t>(r.host_data);
    if (p >= base && p < base + r.size) {
      if (t.type == core::GgmlType::kBF16 || IsNvfp4(t.type)) {
        if (error_msg != nullptr) {
          *error_msg = "inkling: unconverted tensor reached GPU mirror";
        }
        return false;
      }
      t.device = static_cast<std::uint8_t*>(r.device_data) + (p - base);
      return true;
    }
  }

  return MirrorDeviceCopy(t, owned, /*keep_host_pages=*/false, error_msg);
}

bool UploadWeights(ModelWeights& w,
                   const std::vector<Model::RegisteredRegion>& regions,
                   std::vector<void*>& owned, std::string* error_msg) {
  auto mirror_loops = [&](LoopTensors& lt) -> bool {
    return MirrorTensor(lt.down, regions, owned, error_msg) &&
           MirrorTensor(lt.forget, regions, owned, error_msg) &&
           MirrorTensor(lt.gain, regions, owned, error_msg);
  };
  if (!MirrorTensor(w.token_embd, regions, owned, error_msg) ||
      !MirrorTensor(w.token_embd_norm, regions, owned, error_msg) ||
      !MirrorTensor(w.output, regions, owned, error_msg) ||
      !MirrorTensor(w.output_norm, regions, owned, error_msg) ||
      !MirrorTensor(w.loop_kv_res_query, regions, owned, error_msg) ||
      !MirrorTensor(w.loop_res_query, regions, owned, error_msg) ||
      !MirrorTensor(w.output_res_query, regions, owned, error_msg)) {
    return false;
  }
  for (auto& l : w.layers) {
    if (!MirrorTensor(l.attn_norm, regions, owned, error_msg) ||
        !MirrorTensor(l.wq, regions, owned, error_msg) ||
        !MirrorTensor(l.wk, regions, owned, error_msg) ||
        !MirrorTensor(l.wv, regions, owned, error_msg) ||
        !MirrorTensor(l.wo, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_gate, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_rel, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_rel_bias, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_rel_clip, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_q_clip, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_sink_proj, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_k_conv, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_v_conv, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_post_conv, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_q_inp_norm, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_k_inp_norm, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_v_inp_norm, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_k_conv_proj, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_v_conv_proj, regions, owned, error_msg) ||
        !MirrorTensor(l.attn_res_query, regions, owned, error_msg) ||
        !MirrorTensor(l.ffn_norm, regions, owned, error_msg) ||
        !MirrorTensor(l.ffn_post_conv, regions, owned, error_msg) ||
        !MirrorTensor(l.ffn_res_query, regions, owned, error_msg) ||
        !MirrorTensor(l.ffn_gate_inp, regions, owned, error_msg) ||
        !MirrorTensor(l.exp_probs_b, regions, owned, error_msg) ||
        !MirrorTensor(l.ffn_gate_up_exps, regions, owned, error_msg) ||
        !MirrorTensor(l.ffn_down_exps, regions, owned, error_msg) ||
        !mirror_loops(l.loop_attn) || !mirror_loops(l.loop_k) ||
        !mirror_loops(l.loop_mlp) || !mirror_loops(l.loop_v)) {
      return false;
    }
  }
  return true;
}

// Dense GEMM through shared core launchers (laguna recipe verbatim):
// narrow decode -> prefill -> batched fallback. All inkling projections
// (Q/K/V/O, gate, rel, sink, router, loop gates, expert slabs, lm head)
// run through this helper; no model-local GEMM exists.
void InklingGemm(const TensorRef& w, const float* x, __half* x16,
                 __half* wdq_scratch, std::size_t wdq_bytes,
                 void* preq_scratch, std::size_t preq_bytes, float* y,
                 std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                 hipStream_t stream) {
  namespace gemm = core::compute::gemm;
  const void* dev = w.device != nullptr ? w.device : w.data;
  if (n_tokens >= 1U && n_tokens <= 8U && w.type == core::GgmlType::kQ8_0 &&
      (k % 32U) == 0U && preq_bytes >= gemm::NarrowQ80PrequantBytes(n_tokens, k) &&
      gemm::PrequantizeQ8_1(x, preq_scratch, n_tokens, k, stream) == 0 &&
      gemm::LaunchNarrowQ80Decode(dev, preq_scratch, y, k, m, n_tokens,
                                  gemm::NarrowQ80PrequantStrideBlocks(k),
                                  stream)) {
    return;
  }
  if (n_tokens == 1 &&
      gemm::GemvDecode(dev, static_cast<int>(w.type), x, y, n_tokens, m, k,
                       preq_scratch, preq_bytes, stream) ==
          ::mbufo::core::quant::Status::kOk) {
    return;
  }
  if (n_tokens > 1 && n_tokens <= gemm::kQuantGemvMultiMaxTokens &&
      gemm::QuantGemvMulti(dev, static_cast<int>(w.type), x, y, n_tokens, m, k,
                           stream)) {
    return;
  }
  gemm::ConvertF32ToF16(x, x16, static_cast<std::size_t>(n_tokens) * k, stream);
  const auto pst = gemm::QuantGemmPrefill(dev, static_cast<int>(w.type), x16, y, n_tokens,
                             m, k, wdq_scratch, wdq_bytes,
                             stream);
  if (pst == ::mbufo::core::quant::Status::kOk) {
    return;
  }
  if (w.type == core::GgmlType::kQ4_K || w.type == core::GgmlType::kQ5_K ||
      w.type == core::GgmlType::kQ6_K) {
    for (std::uint32_t off = 0; off < n_tokens; ++off) {
      (void)gemm::GemvDecode(dev, static_cast<int>(w.type),
                       x + static_cast<std::size_t>(off) * k,
                       y + static_cast<std::size_t>(off) * m, 1, m, k,
                       preq_scratch, preq_bytes, stream);
    }
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

// Scalar tensors (norms, clips, biases, res queries) must be F32; anything
// else is a loud load/forward error, never a silent reinterpretation.
bool F32Check(const TensorRef& t, const char* what, std::string* error_msg) {
  if (t.empty()) return true;
  if (t.type != core::GgmlType::kF32) {
    if (error_msg != nullptr) {
      *error_msg = std::string("inkling: expected F32 for ") + what;
    }
    return false;
  }
  return true;
}

}  // namespace

Model::~Model() {
  for (void* dev : device_allocations_) {
    if (dev != nullptr) (void)hipFree(dev);
  }
  for (const auto& r : registered_regions_) {
    if (r.host_copy == nullptr) continue;
    (void)hipHostUnregister(r.host_copy);
    // Converted F16/Q8_0 buffers are heap vectors owned by converted_f16_ /
    // converted_q80_, not anonymous mmaps: unregister only, never munmap.
    if (r.heap_backed) continue;
    (void)munmap(r.host_copy, r.size);
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
  m->model_path_ = model_path;
  if (m->options_.max_context == 0 ||
      m->options_.max_context > weights->config.context_length) {
    m->options_.max_context = weights->config.context_length;
  }
  m->reader_ = std::move(reader);
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  m->tokenizer_ = std::move(tokenizer);

  // NVFP4 experts (blk.0-19) become owned Q8_0 buffers, BF16 projections
  // and Q8_0 dense tensors become owned F16 buffers, before any GPU upload,
  // so core GEMM only sees F16/Q8_0/F32.
  if (!ConvertLoadTimeFormats(*m->weights_, &m->converted_f16_,
                              &m->converted_q80_, error_msg)) {
    return nullptr;
  }
  if (!TransposeRelBias(*m->weights_, &m->converted_f32_, error_msg)) {
    return nullptr;
  }
  // Heap F16/Q8_0 buffers must be visible to MirrorTensor without hipMalloc.
  if (!RegisterConvertedBuffers(&m->converted_f16_, &m->registered_regions_,
                                error_msg)) {
    return nullptr;
  }
  if (!RegisterConvertedBuffers(&m->converted_q80_, &m->registered_regions_,
                                error_msg)) {
    return nullptr;
  }
  if (!RegisterConvertedBuffers(&m->converted_f32_, &m->registered_regions_,
                                error_msg)) {
    return nullptr;
  }

  std::size_t total_reg_bytes = 0;
  for (const auto& reg : m->reader_->GetMappedRegions()) total_reg_bytes += reg.size;
  if (total_reg_bytes <= (32ULL << 30) && std::getenv("MBUFO_INKLING_NO_HOST_MAP") == nullptr)
  for (const auto& reg : m->reader_->GetMappedRegions()) {
    void* host_copy = mmap(nullptr, reg.size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (host_copy == MAP_FAILED) {
      if (error_msg != nullptr) *error_msg = "inkling: mmap host_copy failed";
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
  if (!options.vision_model_path.empty()) {
    m->vision_ = VisionEncoder::Load(
        options.vision_model_path,
        static_cast<std::uint32_t>(m->weights_->config.hidden_size), nullptr,
        error_msg);
    if (!m->vision_) return nullptr;
  }
  if (!options.dspark_model_path.empty()) {
    m->dspark_ = DsparkModel::Load(
        options.dspark_model_path,
        static_cast<std::uint32_t>(m->weights_->config.hidden_size),
        m->weights_->config.vocab_size, error_msg);
    if (!m->dspark_) return nullptr;
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
  auto acc = [&](const TensorRef& t) { total += t.size_bytes; };
  for (const auto& l : weights_->layers) {
    acc(l.attn_norm);
    acc(l.wq);
    acc(l.wk);
    acc(l.wv);
    acc(l.wo);
    acc(l.attn_gate);
    acc(l.attn_rel);
    acc(l.attn_rel_bias);
    acc(l.attn_rel_clip);
    acc(l.attn_q_clip);
    acc(l.attn_sink_proj);
    acc(l.attn_k_conv);
    acc(l.attn_v_conv);
    acc(l.attn_post_conv);
    acc(l.attn_q_inp_norm);
    acc(l.attn_k_inp_norm);
    acc(l.attn_v_inp_norm);
    acc(l.attn_k_conv_proj);
    acc(l.attn_v_conv_proj);
    acc(l.attn_res_query);
    acc(l.ffn_norm);
    acc(l.ffn_post_conv);
    acc(l.ffn_res_query);
    acc(l.ffn_gate_inp);
    acc(l.exp_probs_b);
    acc(l.ffn_gate_up_exps);
    acc(l.ffn_down_exps);
    for (const auto* lt :
         {&l.loop_attn, &l.loop_k, &l.loop_mlp, &l.loop_v}) {
      acc(lt->down);
      acc(lt->forget);
      acc(lt->gain);
    }
  }
  acc(weights_->token_embd);
  acc(weights_->token_embd_norm);
  acc(weights_->output);
  acc(weights_->output_norm);
  acc(weights_->loop_kv_res_query);
  acc(weights_->loop_res_query);
  acc(weights_->output_res_query);
  if (vision_ != nullptr) total += vision_->ResidentBytes();
  if (dspark_ != nullptr) total += dspark_->ResidentBytes();
  return total;
}

Session::Session(std::shared_ptr<Model> model) : model_(std::move(model)) {}

Session::~Session() {
  if (stream_ != nullptr) (void)hipStreamSynchronize(stream_);
  if (decode_graph_exec_ != nullptr) (void)hipGraphExecDestroy(decode_graph_exec_);
  if (decode_graph_ != nullptr) (void)hipGraphDestroy(decode_graph_);
  for (void* p : k_cache_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (void* p : v_cache_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (void* p : arena_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (void* p : sconv_k_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (void* p : sconv_v_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (void* p : sconv_pa_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (void* p : sconv_pm_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (void* p : kv_backup_k_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (void* p : kv_backup_v_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (void* p : sconv_bk_k_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (void* p : sconv_bk_v_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (void* p : sconv_bk_pa_) {
    if (p != nullptr) (void)hipFree(p);
  }
  for (void* p : sconv_bk_pm_) {
    if (p != nullptr) (void)hipFree(p);
  }
  if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
}

bool Session::Init(std::string* error_msg) {
  if (!Check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
             error_msg, "stream create")) {
    return false;
  }
  const auto& cfg = model_->weights_->config;
  const std::size_t ctx = model_->MaxContext();
  const std::size_t h = cfg.hidden_size;
  const std::size_t q_dim =
      static_cast<std::size_t>(cfg.num_q_heads) * cfg.head_dim;
  const std::size_t kv_dim =
      static_cast<std::size_t>(cfg.num_kv_heads) * cfg.head_dim;
  const std::size_t rel_dim =
      static_cast<std::size_t>(cfg.num_q_heads) * cfg.rel_dim;
  const std::size_t fused = static_cast<std::size_t>(cfg.expert_ffn_size) * 2;
  const std::size_t ff = cfg.expert_ffn_size;
  const std::size_t slots = static_cast<std::size_t>(kChunk) * kTopK;
  const std::size_t x16_cols =
      std::max(std::max(h, q_dim), static_cast<std::size_t>(2) * h);
  moe_tile_cap_ = (slots + cfg.expert_count * 15) / 16 + cfg.expert_count + 1;
  vocab_rows_ = static_cast<std::uint32_t>(model_->weights_->output.rows);

  auto alloc = [&](auto** ptr, std::size_t count,
                   const char* what) -> bool {
    void* dev = nullptr;
    if (!Check(hipMalloc(&dev, count * sizeof(**ptr)), error_msg, what)) {
      return false;
    }
    *ptr = static_cast<std::decay_t<decltype(**ptr)>*>(dev);
    arena_.push_back(dev);
    return true;
  };

  if (!alloc(&d_tokens_, kChunk, "d_tokens") ||
      !alloc(&d_pos_, 1, "d_pos") ||
      !alloc(&d_hidden_, kChunk * h, "d_hidden") ||
      !alloc(&d_normed_, kChunk * h, "d_normed") ||
      !alloc(&d_fuse_a_, kChunk * h, "d_fuse_a") ||
      !alloc(&d_fuse_b_, kChunk * h, "d_fuse_b") ||
      !alloc(&d_fuse_c_, kChunk * h, "d_fuse_c") ||
      !alloc(&d_q_, kChunk * q_dim, "d_q") ||
      !alloc(&d_k_, kChunk * kv_dim, "d_k") ||
      !alloc(&d_v_, kChunk * kv_dim, "d_v") ||
      !alloc(&d_rel_, kChunk * rel_dim, "d_rel") ||
      !alloc(&d_sink_, kChunk * cfg.num_q_heads, "d_sink") ||
      !alloc(&d_gate_, kChunk * q_dim, "d_gate") ||
      !alloc(&d_attn_out_, kChunk * q_dim, "d_attn_out") ||
      !alloc(&d_attn_splitk_, 16ULL * cfg.num_q_heads * 32ULL * 130ULL, "d_attn_splitk") ||
      !alloc(&d_proj_out_, kChunk * h, "d_proj_out") ||
      !alloc(&d_conv_delta_, kChunk * 32, "d_conv_delta") ||
      !alloc(&d_tau_, kChunk, "d_tau") ||
      !alloc(&d_router_logits_, kChunk * cfg.expert_count, "d_router") ||
      !alloc(&d_exp_ids_, slots, "d_exp_ids") ||
      !alloc(&d_exp_weights_, slots, "d_exp_weights") ||
      !alloc(&d_x16_, kChunk * x16_cols, "d_x16") ||
      !alloc(&d_expert_in16_, kChunk * h, "d_expert_in16") ||
      !alloc(&d_expert_gu_, slots * fused, "d_expert_gu") ||
      !alloc(&d_expert_act_, slots * ff, "d_expert_act") ||
      !alloc(&d_expert_act16_, slots * ff, "d_expert_act16") ||
      !alloc(&d_expert_down_, slots * h, "d_expert_down") ||
      !alloc(&d_moe_out_, kChunk * h, "d_moe_out") ||
      !alloc(&d_moe_counts_, cfg.expert_count, "d_moe_counts") ||
      !alloc(&d_moe_pad_bounds_, cfg.expert_count + 1, "d_moe_pad_bounds") ||
      !alloc(&d_moe_cursors_, cfg.expert_count, "d_moe_cursors") ||
      !alloc(&d_moe_rows_token_,
             core::compute::gemm::RoutedCompactRows(slots, cfg.expert_count),
             "d_moe_rows_token") ||
      !alloc(&d_moe_rows_slot_,
             core::compute::gemm::RoutedCompactRows(slots, cfg.expert_count),
             "d_moe_rows_slot") ||
      !alloc(&d_moe_tiles_,
             (slots + cfg.expert_count * 15) / 16 + cfg.expert_count + 1,
             "d_moe_tiles") ||
      !alloc(&d_moe_n_tiles_, 1, "d_moe_n_tiles") ||
      !alloc(&d_pair_, kChunk * h * 2, "d_pair") ||
      !alloc(&d_g64_, kChunk * cfg.loop_gate_rank, "d_g64") ||
      !alloc(&d_fg_, kChunk * h, "d_fg") ||
      !alloc(&d_mg_, kChunk * h, "d_mg") ||
      !alloc(&d_msg_, kChunk * h, "d_msg") ||
      !alloc(&d_kvmsg_, kChunk * h, "d_kvmsg") ||
      !alloc(&d_msgv_, kChunk * h, "d_msgv") ||
      !alloc(&d_msgm_, kChunk * h, "d_msgm") ||
      !alloc(&d_prevbank_,
             kChunk * static_cast<std::size_t>(kMaxSnaps) * h, "d_prevbank") ||
      !alloc(&d_snaps_,
             kChunk * static_cast<std::size_t>(kMaxSnaps) * h, "d_snaps") ||
      !alloc(&d_h5_, kChunk * h, "d_h5") ||
      !alloc(&d_hsave_, kChunk * h, "d_hsave") ||
      !alloc(&d_logits_, vocab_rows_, "d_logits") ||
      !alloc(&d_argmax_, 1, "d_argmax")) {
    return false;
  }
  {
    void* dev = nullptr;
    if (!Check(hipMalloc(&dev, 8 * 64 *
                                   sizeof(core::compute::gemm::ArgmaxCandidate)),
               error_msg, "d_argmax_scratch")) {
      return false;
    }
    d_argmax_scratch_ = dev;
    arena_.push_back(dev);
  }

  wdq_elems_ = std::max(q_dim * h, static_cast<std::size_t>(fused) * h);
  {
    void* dev = nullptr;
    if (!Check(hipMalloc(&dev, wdq_elems_ * sizeof(__half)), error_msg,
               "d_wdq")) {
      return false;
    }
    d_wdq_ = static_cast<__half*>(dev);
    arena_.push_back(dev);
  }
  preq_bytes_ = 256ULL * 1024ULL;
  {
    void* dev = nullptr;
    if (!Check(hipMalloc(&dev, preq_bytes_), error_msg, "d_preq")) {
      return false;
    }
    d_preq_ = dev;
    arena_.push_back(dev);
  }

  const std::size_t kv_bytes = ctx * kv_dim * sizeof(__half);
  k_cache_.assign(cfg.num_layers, nullptr);
  v_cache_.assign(cfg.num_layers, nullptr);
  sconv_k_.assign(cfg.num_layers, nullptr);
  sconv_v_.assign(cfg.num_layers, nullptr);
  sconv_pa_.assign(cfg.num_layers, nullptr);
  sconv_pm_.assign(cfg.num_layers, nullptr);
  kv_backup_k_.assign(cfg.num_layers, nullptr);
  kv_backup_v_.assign(cfg.num_layers, nullptr);
  for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
    const bool is_loop =
        (l >= cfg.loop_core_start && l < cfg.loop_core_end);
    const std::uint32_t pass_slots = is_loop ? 3 : 1;
    void *k = nullptr, *v = nullptr;
    if (!Check(hipMalloc(&k, kv_bytes), error_msg, "k_cache alloc") ||
        !Check(hipMalloc(&v, kv_bytes), error_msg, "v_cache alloc")) {
      if (k != nullptr) (void)hipFree(k);
      return false;
    }
    k_cache_[l] = k;
    v_cache_[l] = v;
    auto alloc_state = [&](std::vector<void*>& vec, std::size_t dim) -> bool {
      void* s = nullptr;
      if (!Check(hipMalloc(&s, pass_slots * dim * 3 * sizeof(float)), error_msg,
                 "sconv state alloc")) {
        return false;
      }
      if (!Check(hipMemset(s, 0, pass_slots * dim * 3 * sizeof(float)),
                 error_msg, "sconv state zero")) {
        (void)hipFree(s);
        return false;
      }
      vec[l] = s;
      return true;
    };
    if (!alloc_state(sconv_k_, kv_dim) || !alloc_state(sconv_v_, kv_dim) ||
        !alloc_state(sconv_pa_, h) || !alloc_state(sconv_pm_, h)) {
      return false;
    }
    if (is_loop) {
      const std::size_t bk_bytes = kChunk * kv_dim * sizeof(__half);
      void *bk = nullptr, *bv = nullptr;
      if (!Check(hipMalloc(&bk, bk_bytes), error_msg, "kv backup k") ||
          !Check(hipMalloc(&bv, bk_bytes), error_msg, "kv backup v")) {
        if (bk != nullptr) (void)hipFree(bk);
        return false;
      }
      kv_backup_k_[l] = bk;
      kv_backup_v_[l] = bv;
    }
  }

  // Scalar-tensor type contracts (loud failures, never silent casts).
  const auto& w = *model_->weights_;
  for (const auto& l : w.layers) {
    if (!F32Check(l.attn_norm, "attn_norm", error_msg) ||
        !F32Check(l.attn_q_inp_norm, "attn_q_inp_norm", error_msg) ||
        !F32Check(l.attn_k_inp_norm, "attn_k_inp_norm", error_msg) ||
        !F32Check(l.attn_v_inp_norm, "attn_v_inp_norm", error_msg) ||
        !F32Check(l.attn_rel_clip, "attn_rel_clip", error_msg) ||
        !F32Check(l.attn_q_clip, "attn_q_clip", error_msg) ||
        !F32Check(l.attn_rel_bias, "attn_rel_bias", error_msg) ||
        !F32Check(l.ffn_norm, "ffn_norm", error_msg) ||
        !F32Check(l.exp_probs_b, "exp_probs_b", error_msg) ||
        !F32Check(l.attn_res_query, "attn_res_query", error_msg) ||
        !F32Check(l.ffn_res_query, "ffn_res_query", error_msg)) {
      return false;
    }
  }
  if (!F32Check(w.token_embd_norm, "token_embd_norm", error_msg) ||
      !F32Check(w.output_norm, "output_norm", error_msg) ||
      !F32Check(w.loop_kv_res_query, "loop_kv_res_query", error_msg) ||
      !F32Check(w.loop_res_query, "loop_res_query", error_msg) ||
      !F32Check(w.output_res_query, "output_res_query", error_msg)) {
    return false;
  }

  if (model_->dspark_ != nullptr) {
    dspark_ = DsparkSession::Create(*model_->dspark_,
                                    static_cast<std::uint32_t>(ctx),
                                    vocab_rows_, stream_, error_msg);
    if (!dspark_) return false;
    auto alloc_arena = [&](auto*& ptr, std::size_t bytes, const char* what) {
      void* dev = nullptr;
      if (!Check(hipMalloc(&dev, bytes), error_msg, what)) return false;
      ptr = reinterpret_cast<std::remove_reference_t<decltype(ptr)>>(dev);
      arena_.push_back(dev);
      return true;
    };
    const std::size_t h_bytes = kChunk * h * sizeof(float);
    if (!alloc_arena(d_dspark_h5_, h_bytes, "d_dspark_h5") ||
        !alloc_arena(d_dspark_h23_, h_bytes, "d_dspark_h23") ||
        !alloc_arena(d_dspark_h41_, h_bytes, "d_dspark_h41") ||
        !alloc_arena(d_dspark_h48_, h_bytes, "d_dspark_h48") ||
        !alloc_arena(d_all_logits_,
                     static_cast<std::size_t>(DsparkModel::kBlockSize) *
                         vocab_rows_ * sizeof(float),
                     "d_all_logits") ||
        !alloc_arena(d_all_argmax_,
                     DsparkModel::kBlockSize * sizeof(std::int32_t),
                     "d_all_argmax")) {
      return false;
    }
    sconv_bk_k_.assign(cfg.num_layers, nullptr);
    sconv_bk_v_.assign(cfg.num_layers, nullptr);
    sconv_bk_pa_.assign(cfg.num_layers, nullptr);
    sconv_bk_pm_.assign(cfg.num_layers, nullptr);
    for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
      const bool is_loop =
          (l >= cfg.loop_core_start && l < cfg.loop_core_end);
      const std::uint32_t pass_slots = is_loop ? 3 : 1;
      auto alloc_bk = [&](std::vector<void*>& vec, std::size_t dim) -> bool {
        void* s = nullptr;
        if (!Check(hipMalloc(&s, static_cast<std::size_t>(DsparkModel::kBlockSize) *
                                     pass_slots * dim * 3 * sizeof(float)),
                   error_msg, "sconv backup alloc")) {
          return false;
        }
        vec[l] = s;
        return true;
      };
      if (!alloc_bk(sconv_bk_k_, kv_dim) || !alloc_bk(sconv_bk_v_, kv_dim) ||
          !alloc_bk(sconv_bk_pa_, h) || !alloc_bk(sconv_bk_pm_, h)) {
        return false;
      }
    }
  }

  host_logits_.assign(vocab_rows_, 0.0F);
  return true;
}

void Session::ResetStreamState() {
  const auto& cfg = model_->weights_->config;
  const std::size_t h = cfg.hidden_size;
  const std::size_t kv_dim =
      static_cast<std::size_t>(cfg.num_kv_heads) * cfg.head_dim;
  for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
    const bool is_loop =
        (l >= cfg.loop_core_start && l < cfg.loop_core_end);
    const std::size_t rep = is_loop ? 3 : 1;
    (void)hipMemsetAsync(sconv_k_[l], 0, rep * kv_dim * 3 * sizeof(float),
                         stream_);
    (void)hipMemsetAsync(sconv_v_[l], 0, rep * kv_dim * 3 * sizeof(float),
                         stream_);
    (void)hipMemsetAsync(sconv_pa_[l], 0, rep * h * 3 * sizeof(float),
                         stream_);
    (void)hipMemsetAsync(sconv_pm_[l], 0, rep * h * 3 * sizeof(float),
                         stream_);
  }
}
bool LoopTensorsEmpty(const LoopTensors& lt) {
  return lt.down.empty() && lt.forget.empty() && lt.gain.empty();
}

bool Session::FuseHidden(const float* h, const float* msg,
                         const LoopTensors& lt, float* out, std::uint32_t n,
                         std::string* error_msg) {
  auto prof_fh = std::chrono::steady_clock::now();
  (void)ProfTick(stream_, &prof_fh);
  const auto& cfg = model_->weights_->config;
  const std::size_t hs = cfg.hidden_size;
  const std::uint32_t rank = cfg.loop_gate_rank;
  if (n <= 8U && lt.down.type == core::GgmlType::kF16 &&
      lt.forget.type == core::GgmlType::kF16 &&
      lt.gain.type == core::GgmlType::kF16 && rank <= 64U && (hs % 8U) == 0U) {
    kernels::FusedPairNormDown(
        h, msg, static_cast<const __half*>(lt.down.device), d_g64_, n,
        static_cast<std::uint32_t>(hs), rank, cfg.rms_norm_eps, stream_);
    kernels::FuseGateProjApply(
        h, msg, d_g64_, static_cast<const __half*>(lt.forget.device),
        static_cast<const __half*>(lt.gain.device), out, n,
        static_cast<std::uint32_t>(hs), rank, stream_);
    g_prof.fuse_hidden_ms += ProfTick(stream_, &prof_fh);
    (void)error_msg;
    return true;
  }
  kernels::PairNormConcat(h, msg, d_pair_, n, static_cast<std::uint32_t>(hs),
                          cfg.rms_norm_eps, stream_);
  InklingGemm(lt.down, d_pair_, d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_,
              preq_bytes_, d_g64_, n, rank,
              static_cast<std::uint32_t>(hs) * 2, stream_);
  if (lt.forget.type == core::GgmlType::kF16 &&
      lt.gain.type == core::GgmlType::kF16 &&
      ((n <= 8U && rank <= 64U) || (n > 8U && rank == 64U))) {
    kernels::FuseGateProjApply(
        h, msg, d_g64_, static_cast<const __half*>(lt.forget.device),
        static_cast<const __half*>(lt.gain.device), out, n,
        static_cast<std::uint32_t>(hs), rank, stream_);
  } else {
    InklingGemm(lt.forget, d_g64_, d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_,
                preq_bytes_, d_fg_, n, static_cast<std::uint32_t>(hs), rank,
                stream_);
    InklingGemm(lt.gain, d_g64_, d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_,
                preq_bytes_, d_mg_, n, static_cast<std::uint32_t>(hs), rank,
                stream_);
    kernels::GateFuseApply(h, msg, d_fg_, d_mg_, out, n,
                           static_cast<std::uint32_t>(hs), stream_);
  }
  g_prof.fuse_hidden_ms += ProfTick(stream_, &prof_fh);
  (void)error_msg;
  return true;
}

bool Session::ForwardMoe(const LayerWeights& lw, const float* ffn_in,
                             float* moe_out, std::uint32_t n,
                             std::uint32_t layer, std::string* error_msg) {
  namespace gemm = core::compute::gemm;
  const auto& cfg = model_->weights_->config;
  const std::size_t h = cfg.hidden_size;
  const std::size_t fused = static_cast<std::size_t>(cfg.expert_ffn_size) * 2;
  const std::size_t ff = cfg.expert_ffn_size;
  const std::size_t experts = cfg.expert_count;
  const std::size_t slots = static_cast<std::size_t>(n) * kTopK;

  auto prof_t = std::chrono::steady_clock::now();
  InklingGemm(lw.ffn_gate_inp, ffn_in, d_x16_, d_wdq_, wdq_elems_ * 2,
              d_preq_, preq_bytes_, d_router_logits_, n,
              static_cast<std::uint32_t>(experts),
              static_cast<std::uint32_t>(h), stream_);
    kernels::SigmoidRouterTop8(
        d_router_logits_, static_cast<const float*>(lw.exp_probs_b.device),
        d_exp_ids_, d_exp_weights_, n, static_cast<std::uint32_t>(experts),
        cfg.expert_weights_scale, stream_);
    DbgRouterDump(layer, d_exp_ids_, d_exp_weights_, n, kTopK,
                  d_router_logits_, static_cast<std::uint32_t>(experts),
                  static_cast<const float*>(lw.exp_probs_b.device), stream_);

  auto map_moe_wt = [](core::GgmlType tp, gemm::WeightType* out) -> bool {
    if (tp == core::GgmlType::kQ8_0) { *out = gemm::WeightType::kQ8_0; return true; }
    if (tp == core::GgmlType::kQ4_0) { *out = gemm::WeightType::kQ4_0; return true; }
    return false;
  };
  gemm::WeightType gu_wt = gemm::WeightType::kQ8_0;
  gemm::WeightType down_wt = gemm::WeightType::kQ8_0;
  const bool moe_wt_ok =
      map_moe_wt(lw.ffn_gate_up_exps.type, &gu_wt) &&
      map_moe_wt(lw.ffn_down_exps.type, &down_wt);

  if (n <= 8 && !InklingDebugEnabled() && moe_wt_ok) {
    const bool q8_gu_ok =
        preq_bytes_ >= gemm::NarrowQ80PrequantBytes(n, h) &&
        gemm::PrequantizeQ8_1(ffn_in, d_preq_, n, h, stream_) == 0 &&
        gemm::LaunchMoeWaveGateUpQ8(
            lw.ffn_gate_up_exps.device, gu_wt,
            lw.ffn_gate_up_exps.RowBytes(),
            lw.ffn_gate_up_exps.ExpertStrideBytes(), d_preq_,
            gemm::NarrowQ80PrequantStrideBlocks(h), d_exp_ids_, d_expert_act_,
            static_cast<std::uint32_t>(slots), kTopK,
            static_cast<std::uint32_t>(ff), static_cast<std::uint32_t>(h),
            gemm::MoeGateAct::kSituSoftcap, stream_);
    const bool gu_ok =
        q8_gu_ok ||
        gemm::LaunchMoeWaveGateUp(
            lw.ffn_gate_up_exps.device, gu_wt,
            lw.ffn_gate_up_exps.RowBytes(),
            lw.ffn_gate_up_exps.ExpertStrideBytes(), ffn_in, d_exp_ids_,
            d_expert_act_, static_cast<std::uint32_t>(slots), kTopK,
            static_cast<std::uint32_t>(ff), static_cast<std::uint32_t>(h),
            gemm::MoeGateAct::kSituSoftcap, stream_);
    const bool q8_dn_ok =
        gu_ok &&
        preq_bytes_ >= gemm::NarrowQ80PrequantBytes(slots, ff) &&
        gemm::PrequantizeQ8_1(d_expert_act_, d_preq_, slots, ff, stream_) == 0 &&
        gemm::LaunchMoeWaveDownQ8(
            lw.ffn_down_exps.device, down_wt,
            lw.ffn_down_exps.RowBytes(), lw.ffn_down_exps.ExpertStrideBytes(),
            d_preq_, gemm::NarrowQ80PrequantStrideBlocks(ff), d_exp_ids_,
            d_expert_down_, static_cast<std::uint32_t>(slots),
            static_cast<std::uint32_t>(h), static_cast<std::uint32_t>(ff),
            stream_);
    if (gu_ok &&
        (q8_dn_ok ||
         gemm::LaunchMoeWaveDown(
             lw.ffn_down_exps.device, down_wt,
             lw.ffn_down_exps.RowBytes(), lw.ffn_down_exps.ExpertStrideBytes(),
             d_expert_act_, d_exp_ids_, d_expert_down_,
             static_cast<std::uint32_t>(slots), static_cast<std::uint32_t>(h),
             static_cast<std::uint32_t>(ff), stream_))) {
      gemm::MoeEpilogue(d_expert_down_, d_exp_weights_, moe_out, n, kTopK,
                        static_cast<std::uint32_t>(h), stream_);
      g_prof.moe_exps_ms += ProfTick(stream_, &prof_t);
      return true;
    }
  }

  if (n > 8 && !InklingDebugEnabled() && moe_wt_ok) {
    gemm::ExpertCounts(d_exp_ids_, d_moe_counts_, n,
                       static_cast<std::uint32_t>(experts), kTopK, stream_);
    gemm::RoutedCompact(d_exp_ids_, d_moe_counts_, d_moe_pad_bounds_,
                        d_moe_cursors_, d_moe_rows_token_, d_moe_rows_slot_,
                        n, kTopK, static_cast<std::uint32_t>(experts), stream_);
    const std::uint32_t tile_rows =
        (slots >= static_cast<std::size_t>(32) * experts)
            ? 64U
            : ((slots >= static_cast<std::size_t>(16) * experts) ? 48U : 16U);
    kernels::BuildRoutedTiles(d_moe_counts_,
                              static_cast<std::uint32_t>(experts), tile_rows,
                              d_moe_tiles_, d_moe_n_tiles_, stream_);
    const std::uint32_t max_tiles = static_cast<std::uint32_t>(
        std::min<std::size_t>(moe_tile_cap_, (slots + tile_rows - 1) / tile_rows + experts));
    if (max_tiles > 0 && max_tiles <= moe_tile_cap_ &&
        gemm::ConvertF32ToF16(ffn_in, d_expert_in16_,
                               static_cast<std::size_t>(n) * h, stream_) &&
        gemm::RoutedF16Gemm(
            lw.ffn_gate_up_exps.device, gu_wt,
            d_expert_in16_, d_moe_tiles_, max_tiles, tile_rows,
            d_moe_pad_bounds_, d_moe_rows_token_, d_moe_rows_slot_, nullptr,
            nullptr, reinterpret_cast<__half*>(d_expert_gu_), fused, h,
            stream_, d_moe_n_tiles_)) {
      kernels::SituFuseHalfToF16(reinterpret_cast<const __half*>(d_expert_gu_),
                                 d_expert_act16_,
                                 static_cast<std::uint32_t>(slots),
                                 static_cast<std::uint32_t>(ff), stream_);
      if (gemm::RoutedF16Gemm(
              lw.ffn_down_exps.device, down_wt,
              d_expert_act16_, d_moe_tiles_, max_tiles, tile_rows,
              d_moe_pad_bounds_, d_moe_rows_slot_, d_moe_rows_slot_, nullptr,
              nullptr, reinterpret_cast<__half*>(d_expert_down_), h, ff,
              stream_, d_moe_n_tiles_)) {
        kernels::MoeEpilogueHalf(
            reinterpret_cast<const __half*>(d_expert_down_), d_exp_weights_,
            moe_out, n, kTopK, static_cast<std::uint32_t>(h), stream_);
        g_prof.moe_exps_ms += ProfTick(stream_, &prof_t);
        return true;
      }
    }
  }

  std::vector<std::int32_t> host_ids(slots);
  std::vector<float> host_w(slots);
  if (!Check(hipMemcpyAsync(host_ids.data(), d_exp_ids_,
                            slots * sizeof(std::int32_t),
                            hipMemcpyDeviceToHost, stream_),
             error_msg, "copy router ids") ||
      !Check(hipMemcpyAsync(host_w.data(), d_exp_weights_,
                            slots * sizeof(float), hipMemcpyDeviceToHost,
                            stream_),
             error_msg, "copy router weights") ||
      !Check(hipStreamSynchronize(stream_), error_msg, "sync router")) {
    return false;
  }

    if (InklingDebugEnabled()) {
    CheckFiniteHost("moe-router-w", host_w.data(), host_w.size());
    CheckFinite("moe-router-logits", d_router_logits_,
                static_cast<std::size_t>(n) * experts, stream_);
  }
  if (!Check(hipMemsetAsync(moe_out, 0, static_cast<std::size_t>(n) * h *
                                            sizeof(float),
                            stream_),
             error_msg, "zero moe_out")) {
    return false;
  }

  g_prof.moe_router_ms += ProfTick(stream_, &prof_t);
  const void* gu_dev = lw.ffn_gate_up_exps.device;
  const void* down_dev = lw.ffn_down_exps.device;
  const std::uint64_t gu_stride = lw.ffn_gate_up_exps.ExpertStrideBytes();
  const std::uint64_t down_stride = lw.ffn_down_exps.ExpertStrideBytes();
  std::vector<std::int32_t> row_list;
  row_list.reserve(n);
  std::vector<float> w_list;
  w_list.reserve(n);

  for (std::uint32_t e = 0; e < experts; ++e) {
    row_list.clear();
    w_list.clear();
    for (std::size_t s = 0; s < slots; ++s) {
      if (host_ids[s] == static_cast<std::int32_t>(e)) {
        row_list.push_back(static_cast<std::int32_t>(s / kTopK));
        w_list.push_back(host_w[s]);
      }
    }
    if (row_list.empty()) continue;
    const std::uint32_t b = static_cast<std::uint32_t>(row_list.size());
    if (!Check(hipMemcpyAsync(d_exp_ids_, row_list.data(),
                              static_cast<std::size_t>(b) * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               error_msg, "upload expert rows") ||
        !Check(hipMemcpyAsync(d_exp_weights_, w_list.data(),
                              static_cast<std::size_t>(b) * sizeof(float),
                              hipMemcpyHostToDevice, stream_),
               error_msg, "upload expert weights")) {
      return false;
    }
    kernels::GatherRows(ffn_in, d_exp_ids_, d_expert_down_, b,
                        static_cast<std::uint32_t>(h), stream_);
    if (!gemm::ConvertF32ToF16(d_expert_down_, d_expert_in16_,
                               static_cast<std::size_t>(b) * h, stream_)) {
      if (error_msg != nullptr) *error_msg = "inkling: expert gather convert";
      return false;
    }
    const auto* gu_ptr = static_cast<const std::uint8_t*>(gu_dev) +
                         static_cast<std::size_t>(e) * gu_stride;
    const auto* down_ptr = static_cast<const std::uint8_t*>(down_dev) +
                           static_cast<std::size_t>(e) * down_stride;
    const std::uint64_t gu_row = lw.ffn_gate_up_exps.RowBytes();
    const std::uint64_t down_row = lw.ffn_down_exps.RowBytes();
    bool ok =
        gemm::QuantGemmPrefill(gu_ptr, static_cast<int>(lw.ffn_gate_up_exps.type),
                               d_expert_in16_, d_expert_gu_, b,
                               static_cast<std::size_t>(fused),
                               h, d_wdq_, wdq_elems_ * 2, stream_) ==
        ::mbufo::core::quant::Status::kOk;
    if (!ok) {
      for (std::uint32_t r = 0; r < b; ++r) {
        gemm::GemvBatched(gu_ptr, static_cast<int>(lw.ffn_gate_up_exps.type),
                          gu_row, h,
                          d_expert_down_ + static_cast<std::size_t>(r) * h,
                          d_expert_gu_ + static_cast<std::size_t>(r) * fused, 1,
                          static_cast<std::uint32_t>(fused), 0.0F, stream_);
      }
    }
    kernels::SituFuse(d_expert_gu_, d_expert_act_, b,
                      static_cast<std::uint32_t>(ff), stream_);
    if (!gemm::ConvertF32ToF16(d_expert_act_, d_expert_act16_,
                               static_cast<std::size_t>(b) * ff, stream_)) {
      if (error_msg != nullptr) *error_msg = "inkling: expert act convert";
      return false;
    }
    ok = gemm::QuantGemmPrefill(down_ptr,
                                static_cast<int>(lw.ffn_down_exps.type),
                                d_expert_act16_, d_expert_down_, b, h, ff,
                                d_wdq_, wdq_elems_ * 2, stream_) ==
         ::mbufo::core::quant::Status::kOk;
    if (!ok) {
      for (std::uint32_t r = 0; r < b; ++r) {
        gemm::GemvBatched(down_ptr, static_cast<int>(lw.ffn_down_exps.type),
                          down_row, ff,
                          d_expert_act_ + static_cast<std::size_t>(r) * ff,
                          d_expert_down_ + static_cast<std::size_t>(r) * h, 1,
                          static_cast<std::uint32_t>(h), 0.0F, stream_);
      }
    }
    kernels::ScatterAddWeighted(d_expert_down_, d_exp_weights_, d_exp_ids_,
                                moe_out, b, static_cast<std::uint32_t>(h),
                                stream_);
  }
  g_prof.moe_exps_ms += ProfTick(stream_, &prof_t);
  return true;
}

bool Session::ForwardLayer(std::uint32_t layer, float* hidden, std::uint32_t n,
                           std::uint32_t start_pos, const float* msg_q,
                           const float* msg_k, const float* msg_v,
                           const float* msg_m, float* snap_bank,
                           std::uint32_t snap_stride,
                           std::uint32_t* snap_count, std::string* error_msg) {
  const auto& cfg = model_->weights_->config;
  const auto& lw = model_->weights_->layers[layer];
  const std::size_t h = cfg.hidden_size;
  const std::uint32_t q_dim =
      lw.q_heads * cfg.head_dim;
  const std::uint32_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
  auto prof_lt = std::chrono::steady_clock::now();
  const bool is_global = lw.is_swa;  // metadata flag names the 5/11/17/23
                                     // global layers; see resolution note.
  const bool is_loop =
      (layer >= cfg.loop_core_start && layer < cfg.loop_core_end);
  const std::uint32_t window = is_global ? 0 : cfg.sliding_window;

  // Reference boundary/bank rule (SlidingDecoderLayer / FullDecoderLayer /
  // loop variants, TextModel bank pushes): block_residual = hidden (layer
  // input). On block boundaries (layer % attn_res_block == 0) the input is
  // pushed to the snapshot bank and the attention residual carries the attn
  // branch only; otherwise residual = input + attn branch. The attention
  // input and the MLP input are REPLACED by BlockAttentionResidual mixes
  // over bank + current partial (never added). attn_res_query is absent on
  // blk.0, whose attention input is therefore used raw (reference: layer 0
  // has no attention_input); its MLP mix still runs over [embed].
  const bool is_boundary =
      (cfg.attn_res_block != 0 && (layer % cfg.attn_res_block) == 0);
  const std::uint32_t n_bank = (snap_count != nullptr) ? *snap_count : 0;
  bool split_qkv_mix = false;
  // attn_base: bank-mixed stream feeding attention (== raw hidden when no
  // mix ran). `hidden` itself keeps the raw layer input for the residual.
  const float* attn_base = hidden;
  if (snap_bank != nullptr && snap_count != nullptr &&
      (!lw.attn_res_query.empty() || is_boundary) && n_bank < kMaxSnaps) {
    if (!lw.attn_res_query.empty()) {
      const std::uint32_t nq =
          static_cast<std::uint32_t>(lw.attn_res_query.rows);
      const float* qptr =
          static_cast<const float*>(lw.attn_res_query.device);
      if (!is_global || nq <= 1) {
        kernels::MultiQueryResAttnDirect(
            snap_bank, n_bank, snap_stride, hidden, is_boundary, qptr, nq,
            d_attn_out_, nullptr, nullptr, n, static_cast<std::uint32_t>(h),
            kResAttnScale, stream_);
        attn_base = d_attn_out_;
      } else {
        kernels::MultiQueryResAttnDirect(
            snap_bank, n_bank, snap_stride, hidden, is_boundary, qptr, 3U,
            d_attn_out_, d_proj_out_, d_moe_out_, n,
            static_cast<std::uint32_t>(h), kResAttnScale, stream_);
        attn_base = d_attn_out_;
        split_qkv_mix = true;
      }
    } else if (is_boundary) {
      kernels::SnapPush(hidden, snap_bank, n, static_cast<std::uint32_t>(h),
                        snap_stride, n_bank, stream_);
    }
    if (is_boundary) {
      *snap_count = n_bank + 1;
    }
  }
  if (split_qkv_mix) {
    DbgDump(layer, "mixq", attn_base, n, h, stream_);
    DbgDump(layer, "mixk", d_proj_out_, n, h, stream_);
    DbgDump(layer, "mixv", d_moe_out_, n, h, stream_);
  } else {
    DbgDump(layer, "mix", attn_base, n, h, stream_);
  }
  float* sconv_k =
      static_cast<float*>(sconv_k_[layer]) + sconv_slot_ * kv_dim * 3;
  float* sconv_v =
      static_cast<float*>(sconv_v_[layer]) + sconv_slot_ * kv_dim * 3;
  float* sconv_pa =
      static_cast<float*>(sconv_pa_[layer]) + sconv_slot_ * h * 3;
  float* sconv_pm =
      static_cast<float*>(sconv_pm_[layer]) + sconv_slot_ * h * 3;
  const bool is_loop_l = (layer >= cfg.loop_core_start && layer < cfg.loop_core_end);
  const std::size_t pfx_slots = is_loop_l ? 3 : 1;
  const std::size_t pfx_stride_kv = pfx_slots * kv_dim * 3;
  const std::size_t pfx_stride_h = pfx_slots * h * 3;
  float* pfx_k = save_sconv_prefixes_ ? (static_cast<float*>(sconv_bk_k_[layer]) + sconv_slot_ * kv_dim * 3) : nullptr;
  float* pfx_v = save_sconv_prefixes_ ? (static_cast<float*>(sconv_bk_v_[layer]) + sconv_slot_ * kv_dim * 3) : nullptr;
  float* pfx_pa = save_sconv_prefixes_ ? (static_cast<float*>(sconv_bk_pa_[layer]) + sconv_slot_ * h * 3) : nullptr;
  float* pfx_pm = save_sconv_prefixes_ ? (static_cast<float*>(sconv_bk_pm_[layer]) + sconv_slot_ * h * 3) : nullptr;

  float* hn = d_normed_;
  float* q_in = nullptr;
  float* k = nullptr;
  float* v = nullptr;

  if (!is_global) {
    if (msg_q != nullptr && is_loop) {
      // Message gates before the input norm: fuse the bank-mixed stream
      // with the per-stream loop messages, then norm each stream into its
      // fixed buffer. (Reference: q/k/v_message_gate on the mix, then
      // input_layernorm per stream.)
      const float* attn_in = attn_base;
      const float* k_in = attn_base;
      const float* v_in = attn_base;
      if (!LoopTensorsEmpty(lw.loop_attn)) {
        if (!FuseHidden(attn_base, msg_q, lw.loop_attn, d_fuse_a_, n,
                        error_msg)) {
          return false;
        }
        attn_in = d_fuse_a_;
      }
      if (!LoopTensorsEmpty(lw.loop_k)) {
        if (!FuseHidden(attn_base, msg_k, lw.loop_k, d_fuse_b_, n, error_msg)) {
          return false;
        }
        k_in = d_fuse_b_;
      }
      if (!LoopTensorsEmpty(lw.loop_v)) {
        if (!FuseHidden(attn_base, msg_v, lw.loop_v, d_fuse_c_, n, error_msg)) {
          return false;
        }
        v_in = d_fuse_c_;
      }
      kernels::RmsNormGamma(attn_in,
                            static_cast<const float*>(lw.attn_norm.device),
                            d_fuse_a_, n, static_cast<std::uint32_t>(h),
                            cfg.rms_norm_eps, stream_);
      kernels::RmsNormGamma(k_in,
                            static_cast<const float*>(lw.attn_norm.device),
                            d_fuse_b_, n, static_cast<std::uint32_t>(h),
                            cfg.rms_norm_eps, stream_);
      kernels::RmsNormGamma(v_in,
                            static_cast<const float*>(lw.attn_norm.device),
                            d_fuse_c_, n, static_cast<std::uint32_t>(h),
                            cfg.rms_norm_eps, stream_);
      q_in = d_fuse_a_;
      k = d_fuse_b_;
      v = d_fuse_c_;
    } else {
      kernels::RmsNormGamma(attn_base,
                            static_cast<const float*>(lw.attn_norm.device),
                            hn, n, static_cast<std::uint32_t>(h),
                            cfg.rms_norm_eps, stream_);
      q_in = hn;
      k = hn;
      v = hn;
    }
  } else {
    kernels::RmsNormGamma(attn_base,
                          static_cast<const float*>(lw.attn_q_inp_norm.device),
                          d_fuse_a_, n, static_cast<std::uint32_t>(h),
                          cfg.rms_norm_eps, stream_);
    kernels::RmsNormGamma(split_qkv_mix ? d_proj_out_ : attn_base,
                          static_cast<const float*>(lw.attn_k_inp_norm.device),
                          d_fuse_b_, n, static_cast<std::uint32_t>(h),
                          cfg.rms_norm_eps, stream_);
    kernels::RmsNormGamma(split_qkv_mix ? d_moe_out_ : attn_base,
                          static_cast<const float*>(lw.attn_v_inp_norm.device),
                          d_fuse_c_, n, static_cast<std::uint32_t>(h),
                          cfg.rms_norm_eps, stream_);
    q_in = d_fuse_a_;
    k = d_fuse_b_;
    v = d_fuse_c_;
    if (msg_q != nullptr && is_loop) {
      // Reference FullLoopDecoderLayer: message gates fuse the q/k/v mixes
      // first, then q/k/v_input_layernorm. Fuse the RAW mixes (d_attn_out_ /
      // d_proj_out_ / d_moe_out_, still intact here) into the norm buffers,
      // then norm (same pattern as the sliding path above).
      if (!LoopTensorsEmpty(lw.loop_attn)) {
        if (!FuseHidden(d_attn_out_, msg_q, lw.loop_attn, d_fuse_a_, n,
                        error_msg)) {
          return false;
        }
      }
      if (!LoopTensorsEmpty(lw.loop_k)) {
        if (!FuseHidden(d_proj_out_, msg_k, lw.loop_k, d_fuse_b_, n,
                        error_msg)) {
          return false;
        }
      }
      if (!LoopTensorsEmpty(lw.loop_v)) {
        if (!FuseHidden(d_moe_out_, msg_v, lw.loop_v, d_fuse_c_, n,
                        error_msg)) {
          return false;
        }
      }
      kernels::RmsNormGamma(d_fuse_a_,
                            static_cast<const float*>(lw.attn_q_inp_norm.device),
                            d_fuse_a_, n, static_cast<std::uint32_t>(h),
                            cfg.rms_norm_eps, stream_);
      kernels::RmsNormGamma(d_fuse_b_,
                            static_cast<const float*>(lw.attn_k_inp_norm.device),
                            d_fuse_b_, n, static_cast<std::uint32_t>(h),
                            cfg.rms_norm_eps, stream_);
      kernels::RmsNormGamma(d_fuse_c_,
                            static_cast<const float*>(lw.attn_v_inp_norm.device),
                            d_fuse_c_, n, static_cast<std::uint32_t>(h),
                            cfg.rms_norm_eps, stream_);
    }
    // Reference (Inkling2KeyValueProjection): dynamic taps are projected
    // from the post-norm K/V streams (same tensors feeding wk/wv below).
    {
      const std::uint64_t proj_dim =
          static_cast<std::uint64_t>(cfg.num_kv_heads) * cfg.sconv_width;
      const bool proj_ok =
          (lw.attn_k_conv_proj.cols == cfg.hidden_size &&
           lw.attn_k_conv_proj.rows == proj_dim &&
           lw.attn_v_conv_proj.cols == cfg.hidden_size &&
           lw.attn_v_conv_proj.rows == proj_dim);
      if (!proj_ok) {
        if (error_msg != nullptr) {
          *error_msg =
              "inkling: k/v_conv_proj is not [hidden,kv_heads*sconv_width]; "
              "dynamic-tap layout unconfirmed, refusing to guess";
        }
        return false;
      }
      InklingGemm(lw.attn_k_conv_proj, k, d_x16_, d_wdq_, wdq_elems_ * 2,
                  d_preq_, preq_bytes_, d_conv_delta_, n, 32,
                  static_cast<std::uint32_t>(h), stream_);
      DbgDump(layer, "delta-k", d_conv_delta_, n, 32, stream_);
    }
  }
  DbgDump(layer, "qnorm", q_in, n, h, stream_);
  DbgDump(layer, "knorm", k, n, h, stream_);
  DbgDump(layer, "vnorm", v, n, h, stream_);

  auto dev_ptr = [](const TensorRef& t) -> const void* {
    return t.device != nullptr ? t.device : t.data;
  };
  bool qin_prequantized = false;
  bool qin_f16_ready = false; qin_prequantized = false;
  auto inkling_gemm_qin = [&](const TensorRef& w, float* y, std::uint32_t m) {
    if (n >= 1 && n <= 8 &&
        preq_bytes_ >= core::compute::gemm::NarrowQ80PrequantBytes(n, h)) {
      if (!qin_prequantized && w.type == core::GgmlType::kQ8_0 &&
          core::compute::gemm::PrequantizeQ8_1(q_in, d_preq_, n, h, stream_) ==
              0) {
        qin_prequantized = true;
      }
      if (qin_prequantized && w.type == core::GgmlType::kQ8_0 &&
          core::compute::gemm::LaunchNarrowQ80Decode(
              dev_ptr(w), d_preq_, y, h, m, n,
              core::compute::gemm::NarrowQ80PrequantStrideBlocks(h), stream_)) {
        return;
      }
      InklingGemm(w, q_in, d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_, preq_bytes_,
                  y, n, m, static_cast<std::uint32_t>(h), stream_);
      return;
    }
    if (n <= core::compute::gemm::kQuantGemvMultiMaxTokens &&
        core::compute::gemm::QuantGemvMulti(
            dev_ptr(w), static_cast<int>(w.type), q_in, y, n, m, h, stream_)) {
      return;
    }
    if (!qin_f16_ready) {
      core::compute::gemm::ConvertF32ToF16(
          q_in, d_x16_, static_cast<std::size_t>(n) * h, stream_);
      qin_f16_ready = true;
    }
    if (core::compute::gemm::QuantGemmPrefill(
            dev_ptr(w), static_cast<int>(w.type), d_x16_, y, n, m, h, d_wdq_,
            wdq_elems_ * 2, stream_) != ::mbufo::core::quant::Status::kOk) {
      InklingGemm(w, q_in, d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_, preq_bytes_,
                  y, n, m, static_cast<std::uint32_t>(h), stream_);
    }
  };
  bool multi_gate_rel_done = false;
  if (!is_global) {
    if (n >= 1 && n <= 8 && q_in == k && q_in == v &&
        preq_bytes_ >= core::compute::gemm::NarrowQ80PrequantBytes(n, h) &&
        lw.wq.type == core::GgmlType::kQ8_0 &&
        lw.wk.type == core::GgmlType::kQ8_0 &&
        lw.wv.type == core::GgmlType::kQ8_0 &&
        lw.attn_gate.type == core::GgmlType::kQ8_0 &&
        lw.attn_rel.type == core::GgmlType::kQ8_0 &&
        core::compute::gemm::PrequantizeQ8_1(q_in, d_preq_, n, h, stream_) == 0) {
      const core::compute::gemm::NarrowQ80Target targets[5] = {
          {dev_ptr(lw.wq), d_q_, q_dim},
          {dev_ptr(lw.wk), d_k_, kv_dim},
          {dev_ptr(lw.wv), d_v_, kv_dim},
          {dev_ptr(lw.attn_gate), d_gate_, q_dim},
          {dev_ptr(lw.attn_rel), d_rel_,
           static_cast<std::uint32_t>(cfg.num_q_heads * cfg.rel_dim)},
      };
      if (core::compute::gemm::LaunchNarrowQ80MultiDecode(
              targets, 5, d_preq_, h, n,
              core::compute::gemm::NarrowQ80PrequantStrideBlocks(h), stream_)) {
        qin_prequantized = true;
        multi_gate_rel_done = true;
      }
    }
    if (!multi_gate_rel_done && n >= 1 && n <= 8 && q_in == k && q_in == v &&
        preq_bytes_ >= core::compute::gemm::NarrowQ80PrequantBytes(n, h) &&
        lw.wq.type == core::GgmlType::kQ8_0 &&
        lw.wk.type == core::GgmlType::kQ8_0 &&
        lw.wv.type == core::GgmlType::kQ8_0 &&
        core::compute::gemm::PrequantizeQ8_1(q_in, d_preq_, n, h, stream_) == 0 &&
        core::compute::gemm::LaunchNarrowQ80QkvDecode(
            dev_ptr(lw.wq), dev_ptr(lw.wk), dev_ptr(lw.wv), d_preq_, d_q_,
            d_k_, d_v_, h, q_dim, kv_dim, n,
            core::compute::gemm::NarrowQ80PrequantStrideBlocks(h), stream_)) {
      qin_prequantized = true;
    } else if (!multi_gate_rel_done) {
      inkling_gemm_qin(lw.wq, d_q_, q_dim);
      if (q_in == k && q_in == v) {
        inkling_gemm_qin(lw.wk, d_k_, kv_dim);
        inkling_gemm_qin(lw.wv, d_v_, kv_dim);
      } else {
        InklingGemm(lw.wk, k, d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_, preq_bytes_,
                    d_k_, n, kv_dim, static_cast<std::uint32_t>(h), stream_);
        InklingGemm(lw.wv, v, d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_, preq_bytes_,
                    d_v_, n, kv_dim, static_cast<std::uint32_t>(h), stream_);
        qin_f16_ready = false; qin_prequantized = false;
      }
    }
    kernels::StaticSconvForward(d_k_, sconv_k,
                                static_cast<const float*>(lw.attn_k_conv.device),
                                d_k_, n, kv_dim, kInvSqrt5, stream_, reinterpret_cast<float*>(d_x16_),
                                pfx_k, pfx_stride_kv);
    kernels::StaticSconvForward(d_v_, sconv_v,
                                static_cast<const float*>(lw.attn_v_conv.device),
                                d_v_, n, kv_dim, kInvSqrt5, stream_, reinterpret_cast<float*>(d_x16_),
                                pfx_v, pfx_stride_kv);
  } else {
    // SWA/global layers only (blk.5/11/17/23): dynamic taps use the
    // precomputed d_conv_delta_ (k) above; v taps are projected here from
    // the post-norm V stream (reference KeyValueProjection input).
    inkling_gemm_qin(lw.wq, d_q_, q_dim);
    InklingGemm(lw.wk, k, d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_, preq_bytes_,
                d_k_, n, kv_dim, static_cast<std::uint32_t>(h), stream_);
    InklingGemm(lw.wv, v, d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_, preq_bytes_,
                d_v_, n, kv_dim, static_cast<std::uint32_t>(h), stream_);
    qin_f16_ready = false; qin_prequantized = false;
  DbgDump(layer, "klin", d_k_, n, kv_dim, stream_);
  DbgDump(layer, "vlin", d_v_, n, kv_dim, stream_);
  DbgDumpFull(layer, "klin", d_k_, n, kv_dim, stream_);
  DbgDumpFull(layer, "vlin", d_v_, n, kv_dim, stream_);
  DbgDumpFullM(layer, "m-klin", d_k_, n, kv_dim, stream_);
  DbgDumpFullM(layer, "m-vlin", d_v_, n, kv_dim, stream_);
    kernels::DynamicSconvForward(
        d_k_, sconv_k,
        static_cast<const float*>(lw.attn_k_conv.device), d_conv_delta_, d_k_,
        n, kv_dim, cfg.num_kv_heads, kInvSqrt9, stream_, reinterpret_cast<float*>(d_x16_),
        pfx_k, pfx_stride_kv);
    InklingGemm(lw.attn_v_conv_proj,
                v,
                d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_, preq_bytes_,
                d_conv_delta_, n, 32, static_cast<std::uint32_t>(h), stream_);
    qin_f16_ready = false; qin_prequantized = false;
    DbgDump(layer, "delta-v", d_conv_delta_, n, 32, stream_);
    kernels::DynamicSconvForward(
        d_v_, sconv_v,
        static_cast<const float*>(lw.attn_v_conv.device), d_conv_delta_, d_v_,
        n, kv_dim, cfg.num_kv_heads, kInvSqrt9, stream_, reinterpret_cast<float*>(d_x16_),
        pfx_v, pfx_stride_kv);
  }

  if (layer == 0) {
    ParityDump("dbg-L0-kpre", d_k_, n, kv_dim, stream_);
    ParityDump("dbg-L0-vpre", d_v_, n, kv_dim, stream_);
  }
  DbgDump(layer, "kpre", d_k_, n, kv_dim, stream_);
  DbgDump(layer, "vpre", d_v_, n, kv_dim, stream_);
  DbgDumpFull(layer, "kpre", d_k_, n, kv_dim, stream_);
  DbgDumpFull(layer, "vpre", d_v_, n, kv_dim, stream_);
  DbgDumpFullM(layer, "m-kpre", d_k_, n, kv_dim, stream_);
  DbgDumpFullM(layer, "m-vpre", d_v_, n, kv_dim, stream_);
  kernels::QScaleClip(d_q_, static_cast<const float*>(lw.attn_q_clip.device),
                       n, lw.q_heads, cfg.head_dim, kQkClip, stream_);
  kernels::L2Clip(d_k_, n, cfg.num_kv_heads, cfg.head_dim, kQkClip, stream_);
  kernels::RmsNormNoGamma(d_v_, d_v_, n * cfg.num_kv_heads, cfg.head_dim, cfg.rms_norm_eps, stream_);
  if (layer == 0) {
    ParityDump("dbg-L0-q", d_q_, n, q_dim, stream_);
    ParityDump("dbg-L0-k", d_k_, n, kv_dim, stream_);
    ParityDump("dbg-L0-v", d_v_, n, kv_dim, stream_);
  }
  DbgDump(layer, "q", d_q_, n, q_dim, stream_);
  DbgDump(layer, "k", d_k_, n, kv_dim, stream_);
  DbgDump(layer, "v", d_v_, n, kv_dim, stream_);

  // StaticSconvForward/DynamicSconvForward above used d_x16_ as scratch.
  qin_f16_ready = false; qin_prequantized = false;
  bool gate_done = multi_gate_rel_done;
  bool sink_done = false;
  if (!gate_done && n == 1 && lw.attn_rel.type == core::GgmlType::kQ8_0 &&
      lw.attn_sink_proj.type == core::GgmlType::kQ8_0 &&
      lw.attn_gate.type == core::GgmlType::kQ8_0) {
    if (!qin_prequantized &&
        core::compute::gemm::PrequantizeQ8_1(q_in, d_preq_, n, h, stream_) ==
            0) {
      qin_prequantized = true;
    }
    if (qin_prequantized &&
        core::compute::gemm::LaunchNarrowQ80TripleDecode(
            dev_ptr(lw.attn_rel), dev_ptr(lw.attn_sink_proj),
            dev_ptr(lw.attn_gate), d_preq_, d_rel_, d_sink_, d_gate_, h,
            cfg.num_q_heads * cfg.rel_dim, lw.q_heads, q_dim, n, h / 32,
            stream_)) {
      gate_done = true;
      sink_done = true;
    }
  }
  if (!gate_done && n >= 1 && n <= 8 &&
      preq_bytes_ >= core::compute::gemm::NarrowQ80PrequantBytes(n, h) &&
      lw.attn_rel.type == core::GgmlType::kQ8_0 &&
      lw.attn_gate.type == core::GgmlType::kQ8_0) {
    if (!qin_prequantized &&
        core::compute::gemm::PrequantizeQ8_1(q_in, d_preq_, n, h, stream_) == 0) {
      qin_prequantized = true;
    }
    const core::compute::gemm::NarrowQ80Target targets[2] = {
        {dev_ptr(lw.attn_rel), d_rel_,
         static_cast<std::uint32_t>(cfg.num_q_heads * cfg.rel_dim)},
        {dev_ptr(lw.attn_gate), d_gate_, q_dim},
    };
    if (qin_prequantized &&
        core::compute::gemm::LaunchNarrowQ80MultiDecode(
            targets, 2, d_preq_, h, n,
            core::compute::gemm::NarrowQ80PrequantStrideBlocks(h), stream_)) {
      gate_done = true;
    }
  }
  if (!gate_done) {
    inkling_gemm_qin(lw.attn_rel, d_rel_, cfg.num_q_heads * cfg.rel_dim);
  }
  if (capture_by_pos_) {
    kernels::FillLogScaleTauByPos(d_tau_, d_pos_, cfg.log_scale_floor,
                                  cfg.log_scale_alpha, is_global, stream_);
  } else {
    kernels::FillLogScaleTau(d_tau_, n, start_pos, cfg.log_scale_floor,
                             cfg.log_scale_alpha, is_global, stream_);
  }
  kernels::RelClipTau(d_rel_, static_cast<const float*>(lw.attn_rel_clip.device),
                      d_tau_, n, lw.q_heads, kRelClip, stream_);
  if (layer == 0) ParityDump("dbg-L0-rel", d_rel_, n, lw.q_heads * cfg.rel_dim, stream_);
  DbgDump(layer, "rel", d_rel_, n, lw.q_heads * cfg.rel_dim, stream_);
  if (is_global) {
    // LogScale on Q (transformers v1 applies tau to query and position bias;
    // v2 keeps the query term; the rel term rides RelClipTau above).
    kernels::ScaleRows(d_q_, d_tau_, n, q_dim, stream_);
  }

  if (!sink_done) inkling_gemm_qin(lw.attn_sink_proj, d_sink_, lw.q_heads);
  if (layer == 0) ParityDump("dbg-L0-sink", d_sink_, n, lw.q_heads, stream_);
  DbgDump(layer, "sink", d_sink_, n, lw.q_heads, stream_);

  if (capture_by_pos_) {
    kernels::StoreKvF16ByPos(d_k_, d_v_, static_cast<__half*>(k_cache_[layer]),
                             static_cast<__half*>(v_cache_[layer]), d_pos_,
                             cfg.num_kv_heads, cfg.head_dim, stream_);
    kernels::AttentionDecodeByPos(
        d_q_, static_cast<const __half*>(k_cache_[layer]),
        static_cast<const __half*>(v_cache_[layer]), d_sink_, d_rel_,
        static_cast<const float*>(lw.attn_rel_bias.device), d_attn_out_,
        d_pos_, window, lw.q_heads, cfg.num_kv_heads, cfg.head_dim,
        cfg.rel_extent, kAttnScale, stream_, d_attn_splitk_);
  } else {
    kernels::StoreKvF16(d_k_, d_v_, static_cast<__half*>(k_cache_[layer]),
                        static_cast<__half*>(v_cache_[layer]), n, start_pos,
                        cfg.num_kv_heads, cfg.head_dim, stream_);
  g_prof.res_qkv_ms += ProfTick(stream_, &prof_lt);
  if (n == 1) {
    kernels::AttentionDecode(
        d_q_, static_cast<const __half*>(k_cache_[layer]),
        static_cast<const __half*>(v_cache_[layer]), d_sink_, d_rel_,
        static_cast<const float*>(lw.attn_rel_bias.device), d_attn_out_,
        start_pos, start_pos + n, window, lw.q_heads, cfg.num_kv_heads,
        cfg.head_dim, cfg.rel_extent, kAttnScale, stream_, d_attn_splitk_);
  } else {
    kernels::AttentionPrefill(
        d_q_, static_cast<const __half*>(k_cache_[layer]),
        static_cast<const __half*>(v_cache_[layer]), d_sink_, d_rel_,
        static_cast<const float*>(lw.attn_rel_bias.device), d_attn_out_, n,
        start_pos, start_pos + n, window, lw.q_heads, cfg.num_kv_heads,
        cfg.head_dim, cfg.rel_extent, kAttnScale, stream_, d_attn_splitk_);
  }
  }

  if (layer == 0) ParityDump("dbg-L0-attn", d_attn_out_, n, q_dim, stream_);
  DbgDump(layer, "attn", d_attn_out_, n, q_dim, stream_);
  g_prof.attn_ms += ProfTick(stream_, &prof_lt);
  if (!gate_done) inkling_gemm_qin(lw.attn_gate, d_gate_, q_dim);
  { char nm[64]; snprintf(nm, sizeof(nm), "L%u-attn-out-raw", layer);
    CheckFinite(nm, d_attn_out_, static_cast<std::size_t>(n) * q_dim, stream_); }
  if (layer == 0) ParityDump("dbg-L0-gate", d_gate_, n, q_dim, stream_);
  DbgDump(layer, "gate", d_gate_, n, q_dim, stream_);
  kernels::SigmoidGateMul(d_attn_out_, d_gate_, n, lw.q_heads, cfg.head_dim,
                          stream_);
  InklingGemm(lw.wo, d_attn_out_, d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_,
              preq_bytes_, d_proj_out_, n, static_cast<std::uint32_t>(h),
              q_dim, stream_);
  if (layer == 0) ParityDump("dbg-L0-wo", d_proj_out_, n, static_cast<std::uint32_t>(h), stream_);
  DbgDump(layer, "wo", d_proj_out_, n, static_cast<std::uint32_t>(h), stream_);
  kernels::StaticSconvForward(d_proj_out_, sconv_pa,
                              static_cast<const float*>(lw.attn_post_conv.device),
                              d_proj_out_, n, static_cast<std::uint32_t>(h),
                              kInvSqrt5, stream_, reinterpret_cast<float*>(d_x16_),
                              pfx_pa, pfx_stride_h);
  DbgDumpFullM(layer, "m-postattn", d_proj_out_, n, static_cast<std::uint32_t>(h), stream_);
  if (is_boundary) {
    // Block-boundary residual: attention branch only, no skip of the layer
    // input (the input already went to the snapshot bank above).
    if (!Check(hipMemcpyAsync(hidden, d_proj_out_,
                              static_cast<std::size_t>(n) * h * sizeof(float),
                              hipMemcpyDeviceToDevice, stream_),
               error_msg, "boundary residual copy")) {
      return false;
    }
  } else {
    kernels::AddResidual(hidden, d_proj_out_,
                         static_cast<std::size_t>(n) * h, stream_);
  }
  { char nm1[64], nm2[64];
    snprintf(nm1, sizeof(nm1), "L%u-attn-proj", layer);
    snprintf(nm2, sizeof(nm2), "L%u-hidden-after-attn", layer);
    CheckFinite(nm1, d_proj_out_, static_cast<std::size_t>(n) * h, stream_);
    CheckFinite(nm2, hidden, static_cast<std::size_t>(n) * h, stream_); }
  if (layer == 0) ParityDump("parity-L0-after-attn", hidden, n, h, stream_);
  DbgDump(layer, "after-attn", hidden, n, h, stream_);
  DbgDumpFullM(layer, "m-afattn", hidden, n, h, stream_);

  // MLP-side bank mix writes to d_attn_out_ (free since the o_proj
  // consumed the attention output) so the residual base in `hidden` survives
  // for the final add below (reference adds residual R, never the mixed T).
  float* mix_out = hidden;
  if (snap_bank != nullptr && snap_count != nullptr &&
      !lw.ffn_res_query.empty() && *snap_count < kMaxSnaps) {
    kernels::MultiQueryResAttnDirect(
        snap_bank, *snap_count, snap_stride, hidden, false,
        static_cast<const float*>(lw.ffn_res_query.device), 1, d_attn_out_,
        nullptr, nullptr, n, static_cast<std::uint32_t>(h), kResAttnScale,
        stream_);
    mix_out = d_attn_out_;
  }

  if (layer == 0) ParityDump("dbg-L0-mlpmix", mix_out, n, h, stream_);
  DbgDump(layer, "mlpmix", mix_out, n, h, stream_);

  // Loop layers: message fusion on the mlp-mixed stream with the per-layer
  // mlp message (reference: mlp_message_gate before post_attention_layernorm).
  float* ffn_in = mix_out;
  if (msg_m != nullptr && is_loop && !LoopTensorsEmpty(lw.loop_mlp)) {
    if (!FuseHidden(mix_out, msg_m, lw.loop_mlp, d_fuse_b_, n, error_msg)) {
      return false;
    }
    ffn_in = d_fuse_b_;
  }

  kernels::RmsNormGamma(ffn_in,
                        static_cast<const float*>(lw.ffn_norm.device), d_normed_,
                        n, static_cast<std::uint32_t>(h), cfg.rms_norm_eps,
                        stream_);
  if (layer == 0) ParityDump("dbg-L0-moeln", d_normed_, n, h, stream_);
  DbgDump(layer, "moeln", d_normed_, n, h, stream_);
  g_prof.post_attn_ms += ProfTick(stream_, &prof_lt);
  if (!ForwardMoe(lw, d_normed_, d_moe_out_, n, layer, error_msg)) return false;
  (void)ProfTick(stream_, &prof_lt);
  { char nm[64]; snprintf(nm, sizeof(nm), "L%u-moe-out", layer);
    CheckFinite(nm, d_moe_out_, static_cast<std::size_t>(n) * h, stream_); }
  if (layer == 0) ParityDump("parity-L0-moe-raw", d_moe_out_, n, h, stream_);
  DbgDump(layer, "moe-raw", d_moe_out_, n, h, stream_);
  DbgDumpFullM(layer, "m-moeraw", d_moe_out_, n, h, stream_);
  kernels::StaticSconvForward(d_moe_out_, sconv_pm,
                              static_cast<const float*>(lw.ffn_post_conv.device),
                              d_moe_out_, n, static_cast<std::uint32_t>(h),
                              kInvSqrt5, stream_, reinterpret_cast<float*>(d_x16_),
                              pfx_pm, pfx_stride_h);
  DbgDump(layer, "postmoe", d_moe_out_, n, h, stream_);
  DbgDumpFullM(layer, "m-postmoe", d_moe_out_, n, h, stream_);
  kernels::AddResidual(hidden, d_moe_out_,
                       static_cast<std::size_t>(n) * h, stream_);
  if (layer == 0) ParityDump("parity-L0-after-moe", hidden, n, h, stream_);
  DbgDump(layer, "after-moe", hidden, n, h, stream_);
  DbgDumpFullM(layer, "m-afmoe", hidden, n, h, stream_);
  g_prof.post_moe_ms += ProfTick(stream_, &prof_lt);
  return true;
}
bool Session::ForwardChunk(std::span<const std::int32_t> tokens,
                              std::uint32_t start_pos,
                              std::string* error_msg, bool all_logits) {
  auto prof_ct = std::chrono::steady_clock::now();
  if (ProfileEnabled()) g_prof.Reset(static_cast<std::uint32_t>(tokens.size()));
  const auto& cfg = model_->weights_->config;
  const auto& w = *model_->weights_;
  const std::size_t h = cfg.hidden_size;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0) return true;
  if (InklingDebugEnabled()) {
    fprintf(stderr, "[inkling-dbg] tokens: n=%u vocab=%u h=%zu type=%d rows=%llu cols=%llu",
            n, cfg.vocab_size, h, static_cast<int>(w.token_embd.type),
            (unsigned long long)w.token_embd.rows,
            (unsigned long long)w.token_embd.cols);
    for (std::uint32_t i = 0; i < n; ++i) fprintf(stderr, " %d", tokens[i]);
    fprintf(stderr, "\n");
    fflush(stderr);
  }

  const std::uint32_t snap_stride =
      static_cast<std::uint32_t>(kMaxSnaps * h);
  sconv_slot_ = 0;

  if (!Check(hipMemcpyAsync(d_tokens_, tokens.data(),
                            static_cast<std::size_t>(n) * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             error_msg, "copy tokens")) {
    return false;
  }
  const bool can_graph =
      (n == 1 && !all_logits && !InklingDebugEnabled() && !ParityEnabled() &&
       !ProfileEnabled());
  auto prof_lmt = std::chrono::steady_clock::now();
  auto run_forward_body = [&]() -> bool {
  if (!core::compute::gemm::EmbedGather(
          w.token_embd.device, static_cast<int>(w.token_embd.type), d_tokens_,
          d_hidden_, n, static_cast<std::uint32_t>(h), stream_)) {
    if (error_msg != nullptr) *error_msg = "inkling: EmbedGather failed";
    return false;
  }
  kernels::RmsNormGamma(d_hidden_,
                        static_cast<const float*>(w.token_embd_norm.device),
                        d_hidden_, n, static_cast<std::uint32_t>(h),
                        cfg.rms_norm_eps, stream_);
  if (!image_injections_.empty()) {
    const std::uint32_t chunk_end = start_pos + n;
    for (const auto& img : image_injections_) {
      const std::uint32_t img_len =
          static_cast<std::uint32_t>(img.embeds.size() / h);
      const std::uint32_t img_end = img.offset + img_len;
      const std::uint32_t ov_start = std::max(start_pos, img.offset);
      const std::uint32_t ov_end = std::min(chunk_end, img_end);
      if (ov_start < ov_end) {
        const std::uint32_t cnt = ov_end - ov_start;
        if (!Check(hipMemcpyAsync(
                       d_hidden_ + static_cast<std::size_t>(ov_start - start_pos) * h,
                       img.embeds.data() +
                           static_cast<std::size_t>(ov_start - img.offset) * h,
                       static_cast<std::size_t>(cnt) * h * sizeof(float),
                       hipMemcpyHostToDevice, stream_),
                   error_msg, "inject vision embeds")) {
          return false;
        }
      }
    }
  }
  CheckFinite("embed-norm", d_hidden_, static_cast<std::size_t>(n) * h, stream_);
  ParityDump("parity-embed", d_hidden_, n, h, stream_);

  std::uint32_t snap_count = 0;
  // Snapshot pushes happen inside ForwardLayer on block boundaries
  // (layers 0, 6, 12, 18 push their inputs); see the bank rule there.

  // Layers 0-5 (once per chunk; non-loop, no fusion).
  for (std::uint32_t l = 0; l <= 5; ++l) {
    if (!ForwardLayer(l, d_hidden_, n, start_pos, nullptr, nullptr, nullptr,
                        nullptr, d_snaps_,
                      snap_stride, &snap_count, error_msg)) {
      return false;
    }
    if (d_dspark_h5_ != nullptr && l == 4) {
      if (!Check(hipMemcpyAsync(d_dspark_h5_, d_hidden_,
                                static_cast<std::size_t>(n) * h * sizeof(float),
                                hipMemcpyDeviceToDevice, stream_),
                 error_msg, "save dspark h5")) {
        return false;
      }
    }
    if (l == 5) {
      if (!Check(hipMemcpyAsync(d_h5_, d_hidden_,
                                static_cast<std::size_t>(n) * h * sizeof(float),
                                hipMemcpyDeviceToDevice, stream_),
                 error_msg, "save h5")) {
        return false;
      }
    }
    { char nm[64]; snprintf(nm, sizeof(nm), "chunk-layer%u-exit-hidden", l);
      if (l == 5) CheckFinite(nm, d_hidden_, static_cast<std::size_t>(n) * h, stream_); }
    if (l == 5) ParityDump("parity-layer05", d_hidden_, n, h, stream_);
    if (l <= 4) { char pnm[64]; snprintf(pnm, sizeof(pnm), "parity-layer%02u", l); ParityDump(pnm, d_hidden_, n, h, stream_); }
  }

  const std::uint32_t loop_start = cfg.loop_core_start;  // 6
  const std::uint32_t loop_end = cfg.loop_core_end;      // 18
  const std::size_t kv_row = static_cast<std::size_t>(cfg.num_kv_heads) *
                             cfg.head_dim;
  const std::size_t kv_bytes = static_cast<std::size_t>(n) * kv_row * 2;

  // Pass 1: loop core without fusion; snapshots to slots 1..3.
  for (std::uint32_t l = loop_start; l < loop_end; ++l) {
    if (!ForwardLayer(l, d_hidden_, n, start_pos, nullptr, nullptr, nullptr,
                        nullptr, d_snaps_,
                      snap_stride, &snap_count, error_msg)) {
      return false;
    }
    { if (l == 11 || l == 17) { char nm[64]; snprintf(nm, sizeof(nm), "chunk-layer%u-exit-hidden", l);
    if (l == 11) ParityDump("parity-p1-layer11", d_hidden_, n, h, stream_);
    if (l == 17) ParityDump("parity-p1-layer17", d_hidden_, n, h, stream_);
      CheckFinite(nm, d_hidden_, static_cast<std::size_t>(n) * h, stream_); } }
  }
  CheckFinite("pass1-out-hidden", d_hidden_, static_cast<std::size_t>(n) * h, stream_);
  ParityDump("parity-pass1-out", d_hidden_, n, h, stream_);
  if (!Check(hipMemcpyAsync(d_hsave_, d_hidden_,
                            static_cast<std::size_t>(n) * h * sizeof(float),
                            hipMemcpyDeviceToDevice, stream_),
             error_msg, "save pass1 hidden")) {
    return false;
  }

  // Deep-copy loop KV + sconv state (restored after passes 2/3 so the
  // persistent cache always holds pass-1 products).
  for (std::uint32_t l = loop_start; l < loop_end; ++l) {
    if (capture_by_pos_) {
      kernels::CopyKvSlotByPos(static_cast<__half*>(k_cache_[l]),
                               static_cast<__half*>(kv_backup_k_[l]), d_pos_,
                               static_cast<std::uint32_t>(kv_row), true,
                               stream_);
      kernels::CopyKvSlotByPos(static_cast<__half*>(v_cache_[l]),
                               static_cast<__half*>(kv_backup_v_[l]), d_pos_,
                               static_cast<std::uint32_t>(kv_row), true,
                               stream_);
    } else if (!Check(hipMemcpyAsync(kv_backup_k_[l],
                              static_cast<std::uint8_t*>(k_cache_[l]) +
                                  static_cast<std::size_t>(start_pos) * kv_row * 2,
                              kv_bytes, hipMemcpyDeviceToDevice, stream_),
               error_msg, "backup loop k") ||
        !Check(hipMemcpyAsync(kv_backup_v_[l],
                              static_cast<std::uint8_t*>(v_cache_[l]) +
                                  static_cast<std::size_t>(start_pos) * kv_row * 2,
                              kv_bytes, hipMemcpyDeviceToDevice, stream_),
               error_msg, "backup loop v")) {
      return false;
    }
  }
  sconv_slot_ = 0;

  // Passes 2/3: reset to layer-5 exit, fuse with previous-pass products.
  // Sconv states are per-pass slots, so no backup is needed there; the KV
  // planes are positional (restored below).
  for (std::uint32_t pass = 2; pass <= 3; ++pass) {
    sconv_slot_ = pass - 1;
    if (!Check(hipMemcpyAsync(d_hidden_, d_h5_,
                              static_cast<std::size_t>(n) * h * sizeof(float),
                              hipMemcpyDeviceToDevice, stream_),
               error_msg, "reset h5")) {
      return false;
    }
    CheckFinite("pass-in-hidden-reset", d_hidden_, static_cast<std::size_t>(n) * h, stream_);
    { char pnm[64]; snprintf(pnm, sizeof(pnm), "parity-pass%d-in-reset", pass); ParityDump(pnm, d_hidden_, n, h, stream_); }
    // Per-layer loop messages (reference Inkling2LoopMessageAttentionResidual:
    // messages = attention over previous-pass bank + hidden with the layer's
    // own 4 query vectors, one stream per message).
    // Query mapping A: loop_res_query holds interleaved (q, mlp) pairs per
    // loop layer, loop_kv_res_query holds (k, v) pairs: loop layer l
    // (li = l - loop_start) uses res vectors 2*li (q), 2*li+1 (mlp) and
    // kv vectors 2*li (k), 2*li+1 (v).
    const float* res_qbase =
        static_cast<const float*>(w.loop_res_query.device);
    const float* kv_qbase =
        static_cast<const float*>(w.loop_kv_res_query.device);
    // Stage the previous-pass hidden into slot 3; messages attend over the
    // frozen slots {2, 3} (slot 2 = prev-pass layer-12 input).
    kernels::SnapPush(d_hsave_, d_snaps_, n, static_cast<std::uint32_t>(h),
                      snap_stride, 3, stream_);
    // Freeze the message bank: the layer loop below rebuilds slots 1..2
    // progressively, so per-layer messages must read from this copy.
    if (!Check(hipMemcpyAsync(d_prevbank_, d_snaps_,
                              static_cast<std::size_t>(n) * snap_stride *
                                  sizeof(float),
                              hipMemcpyDeviceToDevice, stream_),
               error_msg, "snapshot prev-pass bank")) {
      return false;
    }
    // Rewind the bank to the pre-loop prefix ([embed]); layer 6/12
    // pushes rebuild slots 1..2 progressively, overwriting the prev pass.
    snap_count = 1;
    for (std::uint32_t l = loop_start; l < loop_end; ++l) {
      const std::uint32_t li = l - loop_start;
      const float* qptr_q = res_qbase + static_cast<std::size_t>(2) * li * h;
      const float* qptr_m = qptr_q + h;
      const float* qptr_k = kv_qbase + static_cast<std::size_t>(2) * li * h;
      const float* qptr_v = qptr_k + h;
      const float* msg_states = d_prevbank_ + static_cast<std::size_t>(2) * h;
      kernels::MultiQueryResAttn4(
          msg_states, 2, snap_stride, qptr_q, qptr_k, qptr_v, qptr_m, d_msg_,
          d_kvmsg_, d_msgv_, d_msgm_, n, static_cast<std::uint32_t>(h),
          kResAttnScale, stream_);
      { char qnm[64];
        snprintf(qnm, sizeof(qnm), "parity-pass%d-L%02u-msgq", pass, l);
        ParityDump(qnm, d_msg_, n, h, stream_);
        snprintf(qnm, sizeof(qnm), "parity-pass%d-L%02u-msgk", pass, l);
        ParityDump(qnm, d_kvmsg_, n, h, stream_);
        snprintf(qnm, sizeof(qnm), "parity-pass%d-L%02u-msgv", pass, l);
        ParityDump(qnm, d_msgv_, n, h, stream_);
        snprintf(qnm, sizeof(qnm), "parity-pass%d-L%02u-msgm", pass, l);
        ParityDump(qnm, d_msgm_, n, h, stream_); }
      if (!ForwardLayer(l, d_hidden_, n, start_pos, d_msg_, d_kvmsg_, d_msgv_,
                        d_msgm_, d_snaps_, snap_stride, &snap_count,
                        error_msg)) {
        return false;
      }
      if (d_dspark_h23_ != nullptr && pass == 2 && l == 10) {
        if (!Check(hipMemcpyAsync(d_dspark_h23_, d_hidden_,
                                  static_cast<std::size_t>(n) * h * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream_),
                   error_msg, "save dspark h23")) {
          return false;
        }
      }
      if (d_dspark_h41_ != nullptr && pass == 3 && l == 16) {
        if (!Check(hipMemcpyAsync(d_dspark_h41_, d_hidden_,
                                  static_cast<std::size_t>(n) * h * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream_),
                   error_msg, "save dspark h41")) {
          return false;
        }
      }
      if (l == 11 || l == 17) { char pnm[64]; snprintf(pnm, sizeof(pnm), "parity-pass%d-layer%u", pass, l); ParityDump(pnm, d_hidden_, n, h, stream_); }
      { char pnm2[64]; snprintf(pnm2, sizeof(pnm2), "parity-pass%d-L%02u-out", pass, l); ParityDump(pnm2, d_hidden_, n, h, stream_); }
    }
    CheckFinite(pass == 2 ? "pass2-out-hidden" : "pass3-out-hidden", d_hidden_,
                static_cast<std::size_t>(n) * h, stream_);
    { char pnm[64]; snprintf(pnm, sizeof(pnm), "parity-pass%d-out", pass); ParityDump(pnm, d_hidden_, n, h, stream_); }
    if (pass == 2) {
      if (!Check(hipMemcpyAsync(d_hsave_, d_hidden_,
                                static_cast<std::size_t>(n) * h * sizeof(float),
                                hipMemcpyDeviceToDevice, stream_),
                 error_msg, "save pass2 hidden")) {
        return false;
      }
    }
  }

  // Restore pass-1 loop KV (sconv slots are per-pass; nothing to restore).
  for (std::uint32_t l = loop_start; l < loop_end; ++l) {
    if (capture_by_pos_) {
      kernels::CopyKvSlotByPos(static_cast<__half*>(k_cache_[l]),
                               static_cast<__half*>(kv_backup_k_[l]), d_pos_,
                               static_cast<std::uint32_t>(kv_row), false,
                               stream_);
      kernels::CopyKvSlotByPos(static_cast<__half*>(v_cache_[l]),
                               static_cast<__half*>(kv_backup_v_[l]), d_pos_,
                               static_cast<std::uint32_t>(kv_row), false,
                               stream_);
    } else if (!Check(hipMemcpyAsync(static_cast<std::uint8_t*>(k_cache_[l]) +
                                  static_cast<std::size_t>(start_pos) * kv_row * 2,
                              kv_backup_k_[l], kv_bytes, hipMemcpyDeviceToDevice,
                              stream_),
               error_msg, "restore loop k") ||
        !Check(hipMemcpyAsync(static_cast<std::uint8_t*>(v_cache_[l]) +
                                  static_cast<std::size_t>(start_pos) * kv_row * 2,
                              kv_backup_v_[l], kv_bytes, hipMemcpyDeviceToDevice,
                              stream_),
               error_msg, "restore loop v")) {
      return false;
    }
  }
  // Layers 18-23 (once; pass-3 hidden). Non-loop layers use slot 0.
  sconv_slot_ = 0;
  for (std::uint32_t l = loop_end; l < cfg.num_layers; ++l) {
    if (!ForwardLayer(l, d_hidden_, n, start_pos, nullptr, nullptr, nullptr,
                        nullptr, d_snaps_,
                      snap_stride, &snap_count, error_msg)) {
      return false;
    }
    if (d_dspark_h48_ != nullptr && l == 23) {
      if (!Check(hipMemcpyAsync(d_dspark_h48_, d_hidden_,
                                static_cast<std::size_t>(n) * h * sizeof(float),
                                hipMemcpyDeviceToDevice, stream_),
                 error_msg, "save dspark h48")) {
        return false;
      }
    }
    { if (l == 23) { char nm[64]; snprintf(nm, sizeof(nm), "chunk-layer%u-exit-hidden", l);
    if (l == 23) ParityDump("parity-layer23", d_hidden_, n, h, stream_);
      CheckFinite(nm, d_hidden_, static_cast<std::size_t>(n) * h, stream_); }
      { char lnm[64]; snprintf(lnm, sizeof(lnm), "parity-layer%02u", l); ParityDump(lnm, d_hidden_, n, h, stream_); } }
  }

  prof_lmt = std::chrono::steady_clock::now();
  if (ProfileEnabled()) (void)hipStreamSynchronize(stream_);
  // Output: res-attention mix over the final bank + last hidden, then
  // norm (reference: final_input mix first, norm after), lm head, softcap.
  if (!w.output_res_query.empty() && snap_count > 0 &&
      snap_count < kMaxSnaps) {
    kernels::MultiQueryResAttnDirect(
        d_snaps_, snap_count, snap_stride, d_hidden_, false,
        static_cast<const float*>(w.output_res_query.device), 1, d_hidden_,
        nullptr, nullptr, n, static_cast<std::uint32_t>(h), kResAttnScale,
        stream_);
  }
  kernels::RmsNormGamma(d_hidden_,
                        static_cast<const float*>(w.output_norm.device),
                        d_hidden_, n, static_cast<std::uint32_t>(h),
                        cfg.rms_norm_eps, stream_);
  if (all_logits && d_all_logits_ != nullptr &&
      n <= DsparkModel::kBlockSize) {
    InklingGemm(w.output, d_hidden_, d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_,
                preq_bytes_, d_all_logits_, n, vocab_rows_,
                static_cast<std::uint32_t>(h), stream_);
    core::compute::gemm::SoftCap(
        d_all_logits_, cfg.final_logit_softcap,
        static_cast<std::size_t>(n) * vocab_rows_, stream_);
    if (!Check(hipMemcpyAsync(
                   d_logits_,
                   d_all_logits_ +
                       static_cast<std::size_t>(n - 1) * vocab_rows_,
                   static_cast<std::size_t>(vocab_rows_) * sizeof(float),
                   hipMemcpyDeviceToDevice, stream_),
               error_msg, "copy last all_logits")) {
      return false;
    }
  } else {
    const float* last = d_hidden_ + static_cast<std::size_t>(n - 1) * h;
  ParityDump("parity-final-hidden", d_hidden_, n, h, stream_);
    InklingGemm(w.output, last, d_x16_, d_wdq_, wdq_elems_ * 2, d_preq_,
                preq_bytes_, d_logits_, 1, vocab_rows_,
                static_cast<std::uint32_t>(h), stream_);
    core::compute::gemm::SoftCap(d_logits_, cfg.final_logit_softcap,
                                 vocab_rows_, stream_);
  }
  CheckFinite("final-logits-precopy", d_logits_, vocab_rows_, stream_);
  return true;
  };
  if (can_graph) {
    if (!Check(hipMemcpyAsync(d_pos_, &start_pos, sizeof(std::uint32_t),
                              hipMemcpyHostToDevice, stream_),
               error_msg, "copy pos")) {
      return false;
    }
    if (decode_graph_exec_ != nullptr) {
      if (!Check(hipGraphLaunch(decode_graph_exec_, stream_), error_msg,
                 "launch decode graph")) {
        return false;
      }
    } else {
      if (!Check(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal),
                 error_msg, "begin decode graph")) {
        return false;
      }
      capture_by_pos_ = true;
      const bool ok = run_forward_body();
      capture_by_pos_ = false;
      if (!ok) {
        hipGraph_t tmp = nullptr;
        (void)hipStreamEndCapture(stream_, &tmp);
        if (tmp != nullptr) (void)hipGraphDestroy(tmp);
        return false;
      }
      if (!Check(hipStreamEndCapture(stream_, &decode_graph_), error_msg,
                 "end decode graph") ||
          !Check(hipGraphInstantiate(&decode_graph_exec_, decode_graph_,
                                     nullptr, nullptr, 0),
                 error_msg, "instantiate decode graph") ||
          !Check(hipGraphLaunch(decode_graph_exec_, stream_), error_msg,
                 "launch captured decode graph")) {
        return false;
      }
    }
  } else {
    if (!run_forward_body()) return false;
  }

  if (dspark_ != nullptr) {
    const std::array<const float*, DsparkModel::kNumTargetLayers> th = {
        d_dspark_h5_, d_dspark_h23_, d_dspark_h41_, d_hidden_};
    if (!dspark_->CaptureContext(th, start_pos, n, error_msg)) {
      return false;
    }
  }

  if (!Check(hipMemcpyAsync(host_logits_.data(), d_logits_,
                            static_cast<std::size_t>(vocab_rows_) *
                                sizeof(float),
                            hipMemcpyDeviceToHost, stream_),
             error_msg, "copy logits") ||
      !Check(hipStreamSynchronize(stream_), error_msg, "sync forward")) {
    return false;
  }
  if (ProfileEnabled()) {
    g_prof.lm_head_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prof_lmt).count();
    g_prof.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prof_ct).count();
    g_prof.Dump();
  }
    CheckFinite("final-logits", d_logits_, vocab_rows_, stream_);
  CheckFiniteHost("final-logits-host", host_logits_.data(), host_logits_.size());
  if (ParityEnabled() && !host_logits_.empty()) {
    std::size_t amx = 0;
    for (std::size_t i = 1; i < host_logits_.size(); ++i) if (host_logits_[i] > host_logits_[amx]) amx = i;
    fprintf(stderr, "[inkling-parity] final-argmax: id=%zu logit=%.6g\n", amx, (double)host_logits_[amx]);
    fflush(stderr);
  }
  return true;
}

bool Session::Forward(std::span<const std::int32_t> tokens,
                      std::uint32_t start_pos, std::string* error_msg) {
  if (tokens.empty()) return true;
  if (start_pos > history_.size()) {
    if (error_msg != nullptr) {
      *error_msg = "inkling Forward: start_pos out of range";
    }
    return false;
  }
  if (start_pos + tokens.size() > model_->MaxContext()) {
    if (error_msg != nullptr) *error_msg = "inkling Forward: context overflow";
    return false;
  }
  if (start_pos == 0) ResetStreamState();
  history_.resize(start_pos);
  std::size_t off = 0;
  while (off < tokens.size()) {
    const std::uint32_t chunk = static_cast<std::uint32_t>(
        std::min<std::size_t>(kChunk, tokens.size() - off));
    if (!ForwardChunk(tokens.subspan(off, chunk),
                      start_pos + static_cast<std::uint32_t>(off), error_msg)) {
      return false;
    }
    history_.insert(history_.end(), tokens.begin() + off,
                    tokens.begin() + off + chunk);
    off += chunk;
  }
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.empty()) {
    if (error_msg != nullptr) *error_msg = "inkling Sync: empty prompt";
    return false;
  }
  if (prompt.size() > model_->MaxContext()) {
    if (error_msg != nullptr) *error_msg = "inkling Sync: prompt exceeds context";
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
  // Rolling sconv state cannot rewind to an arbitrary prefix: restart the
  // stream from position 0 whenever the prompt diverges.
  history_.clear();
  return Forward(prompt, 0, error_msg);
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  history_.push_back(token);
  return Forward(std::span<const std::int32_t>(&token, 1),
                 static_cast<std::uint32_t>(history_.size() - 1), error_msg);
}

std::span<const float> Session::Logits() const noexcept {
  return std::span<const float>(host_logits_.data(), host_logits_.size());
}

std::uint32_t Session::Length() const noexcept {
  return static_cast<std::uint32_t>(history_.size());
}

std::int32_t Session::SampleGreedy() {
  namespace gemm = core::compute::gemm;
  auto* scratch =
      static_cast<gemm::ArgmaxCandidate*>(d_argmax_scratch_);
  gemm::Argmax(d_logits_, scratch, d_argmax_, 1, vocab_rows_, stream_);
  std::int32_t host_out = 0;
  (void)hipMemcpyAsync(&host_out, d_argmax_, sizeof(host_out),
                       hipMemcpyDeviceToHost, stream_);
  (void)hipStreamSynchronize(stream_);
  return host_out;
}

bool Session::RestoreSconvPrefix(std::uint32_t prefix_idx,
                                 std::string* error_msg) {
  const auto& cfg = model_->weights_->config;
  const std::size_t h = cfg.hidden_size;
  const std::size_t kv_dim =
      static_cast<std::size_t>(cfg.num_kv_heads) * cfg.head_dim;
  for (std::uint32_t l = 0; l < cfg.num_layers; ++l) {
    const bool is_loop =
        (l >= cfg.loop_core_start && l < cfg.loop_core_end);
    const std::size_t rep = is_loop ? 3 : 1;
    const std::size_t kv_elems = rep * kv_dim * 3;
    const std::size_t h_elems = rep * h * 3;
    const float* bk_k = static_cast<const float*>(sconv_bk_k_[l]) + prefix_idx * kv_elems;
    const float* bk_v = static_cast<const float*>(sconv_bk_v_[l]) + prefix_idx * kv_elems;
    const float* bk_pa = static_cast<const float*>(sconv_bk_pa_[l]) + prefix_idx * h_elems;
    const float* bk_pm = static_cast<const float*>(sconv_bk_pm_[l]) + prefix_idx * h_elems;
    if (!Check(hipMemcpyAsync(sconv_k_[l], bk_k, kv_elems * sizeof(float),
                              hipMemcpyDeviceToDevice, stream_),
               error_msg, "restore sconv k") ||
        !Check(hipMemcpyAsync(sconv_v_[l], bk_v, kv_elems * sizeof(float),
                              hipMemcpyDeviceToDevice, stream_),
               error_msg, "restore sconv v") ||
        !Check(hipMemcpyAsync(sconv_pa_[l], bk_pa, h_elems * sizeof(float),
                              hipMemcpyDeviceToDevice, stream_),
               error_msg, "restore sconv pa") ||
        !Check(hipMemcpyAsync(sconv_pm_[l], bk_pm, h_elems * sizeof(float),
                              hipMemcpyDeviceToDevice, stream_),
               error_msg, "restore sconv pm")) {
      return false;
    }
  }
  return true;
}

bool Session::ForwardWithAllLogits(std::span<const std::int32_t> tokens,
                                   std::uint32_t start_pos,
                                   std::vector<std::int32_t>* out_argmax,
                                   std::string* error_msg) {
  save_sconv_prefixes_ = true;
  const bool ok = ForwardChunk(tokens, start_pos, error_msg, /*all_logits=*/true);
  save_sconv_prefixes_ = false;
  if (!ok) return false;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  namespace gemm = core::compute::gemm;
  auto* scratch =
      static_cast<gemm::ArgmaxCandidate*>(d_argmax_scratch_);
  gemm::Argmax(d_all_logits_, scratch, d_all_argmax_, n, vocab_rows_, stream_);
  out_argmax->resize(n);
  if (!Check(hipMemcpyAsync(out_argmax->data(), d_all_argmax_,
                            static_cast<std::size_t>(n) * sizeof(std::int32_t),
                            hipMemcpyDeviceToHost, stream_),
             error_msg, "copy all argmax") ||
      !Check(hipStreamSynchronize(stream_), error_msg, "sync all argmax")) {
    return false;
  }
  return true;
}

bool Session::SpeculativeRound(std::uint32_t n_draft,
                               std::vector<std::int32_t>* out_tokens,
                               std::string* error_msg) {
  if (!dspark_) {
    if (error_msg != nullptr) {
      *error_msg = "inkling SpeculativeRound called without DSpark model";
    }
    return false;
  }
  const std::uint32_t anchor_pos = static_cast<std::uint32_t>(history_.size());
  if (anchor_pos >= model_->MaxContext()) {
    if (error_msg != nullptr) *error_msg = "inkling SpeculativeRound: context full";
    return false;
  }
  const std::uint32_t max_avail =
      model_->MaxContext() - anchor_pos - 1;
  n_draft = std::min({n_draft, DsparkModel::kMaxDraft, max_avail});

  const std::int32_t t0 = SampleGreedy();

  if (n_draft == 0 || model_->IsStopToken(t0)) {
    if (!Evaluate(t0, error_msg)) return false;
    out_tokens->assign(1, t0);
    return true;
  }

  std::vector<std::int32_t> drafts;
  const auto& w = *model_->weights_;
  if (!dspark_->DraftBlock(t0, anchor_pos, n_draft, w.token_embd, w.output,
                           &drafts, error_msg)) {
    return false;
  }

  std::vector<std::int32_t> verify_tokens;
  verify_tokens.reserve(1 + n_draft);
  verify_tokens.push_back(t0);
  verify_tokens.insert(verify_tokens.end(), drafts.begin(), drafts.end());

  std::vector<std::int32_t> target_next;
  if (!ForwardWithAllLogits(verify_tokens, anchor_pos, &target_next,
                            error_msg)) {
    return false;
  }

  std::uint32_t accepted = 0;
  while (accepted < n_draft && target_next[accepted] == drafts[accepted]) {
    ++accepted;
  }
  dspark_adapt_ema_ =
      0.85F * dspark_adapt_ema_ +
      0.15F * (static_cast<float>(accepted) / static_cast<float>(n_draft));

  if (const char* env = std::getenv("MBUFO_DSPARK_STATS"); env && env[0] == '1') {
    static uint64_t dbg_rounds = 0, dbg_drafted = 0, dbg_accepted = 0;
    ++dbg_rounds;
    dbg_drafted += n_draft;
    dbg_accepted += accepted;
    if (dbg_rounds % 25 == 0 || model_->IsStopToken(t0)) {
      std::fprintf(stderr,
          "[dspark-stats] r=%llu acc=%llu/%llu (%.1f%%, %.2f tok/step)\n",
          (unsigned long long)dbg_rounds, (unsigned long long)dbg_accepted,
          (unsigned long long)dbg_drafted,
          100.0 * dbg_accepted / (dbg_drafted ? dbg_drafted : 1),
          1.0 + static_cast<double>(dbg_accepted) / dbg_rounds);
    }
  }
  out_tokens->clear();
  out_tokens->reserve(1 + accepted);
  out_tokens->push_back(t0);
  for (std::uint32_t i = 0; i < accepted; ++i) {
    out_tokens->push_back(drafts[i]);
  }
  history_.insert(history_.end(), out_tokens->begin(), out_tokens->end());

  if (accepted == n_draft) {
    return true;
  }
  if (!RestoreSconvPrefix(accepted, error_msg)) return false;
  const float* src_logits =
      d_all_logits_ + static_cast<std::size_t>(accepted) * vocab_rows_;
  if (!Check(hipMemcpyAsync(d_logits_, src_logits,
                            static_cast<std::size_t>(vocab_rows_) * sizeof(float),
                            hipMemcpyDeviceToDevice, stream_),
             error_msg, "restore d_logits") ||
      !Check(hipMemcpyAsync(host_logits_.data(), d_logits_,
                            static_cast<std::size_t>(vocab_rows_) * sizeof(float),
                            hipMemcpyDeviceToHost, stream_),
             error_msg, "restore host_logits") ||
      !Check(hipStreamSynchronize(stream_), error_msg, "sync restore prefix")) {
    return false;
  }
  return true;
}

}  // namespace mbufo::models::inkling
