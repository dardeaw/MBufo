#ifndef GUFO_MODELS_GEMMA4_KERNELS_KERNELS_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_KERNELS_HPP_

#include <cstdint>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "src/models/gemma4/weights.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"

namespace gufo::models::gemma4 {
namespace kernels {

/// Gemma 4 uses GELU-tanh (exact tanh approximation), not the quick-sigmoid
/// SwiGLU used by Qwen. out[i] = gelu(gate[i]) * up[i] over `count` pairs.
void GeluTanhSwiglu(float* gate, const float* up, std::size_t count,
                    hipStream_t stream);

/// out[i] = gelu_tanh(x[i]) over `count` elements.
void GeluTanh(float* x, std::size_t count, hipStream_t stream);

/// Elementwise: out[i] = tanh(x[i] / cap) * cap over `count` elements.
void SoftCap(float* x, float cap, std::size_t count, hipStream_t stream);

/// Elementwise add: out[i] += add[i] over `count` elements.
void AddInplace(float* out, const float* add, std::size_t count,
                hipStream_t stream);

/// Elementwise scale: out[i] *= scale over `count` elements.
void ScaleInplace(float* out, float scale, std::size_t count,
                  hipStream_t stream);

/// RMSNorm over rows: out[t] = rms(x[t]) * gamma. `groups` independent
/// norms share the gamma row (gamma may be null).
void RmsNorm(const float* x, const float* gamma, float* out,
             std::uint32_t n_rows, std::uint32_t dim, float eps,
             hipStream_t stream);

/// Copy a host buffer into device memory (plain hipMalloc + full copy).
/// Returns nullptr on failure.
void* UploadMirror(const void* host, std::size_t bytes);

/// Embedding lookup from a device-mirrored table (Q4_0 / Q8_0 / F32 / F16).
void DebugRow(const void* table, std::uint32_t row_bytes,
              std::uint32_t token, float* d_out, std::uint32_t n,
              hipStream_t stream);

void Embed(const void* table, core::GgmlType type,
           const std::int32_t* tokens, float* res, std::uint32_t n_tokens,
           std::uint32_t hidden, hipStream_t stream);

/// Upload token ids to the persistent device staging buffer used by Embed.
/// Call before any kernel that needs the ids on-device (e.g. the fused
/// embed+scale path).
void EmbedScale(const void* table, core::GgmlType type,
                const std::int32_t* tokens, float* res, std::uint32_t n_tokens,
                std::uint32_t hidden, float scale, hipStream_t stream);

void UploadTokens(const std::int32_t* tokens, std::uint32_t n_tokens,
                  hipStream_t stream);
std::int32_t* DeviceTokens();

/// Copy a host buffer into an existing device buffer via the copy engine.
bool CopyHostToDevice(void* dst, const void* host, std::size_t bytes);

/// Copy a device buffer back to its host mirror (for host-mapped tensors).
void DownloadMirror(void* host, const void* device, std::size_t bytes);

/// Convert an F32 row-major matrix to F16.
void F32ToF16(const float* in, __half* out, std::size_t count,
              hipStream_t stream);

/// y = x @ W^T over an F16 weight matrix (m rows of k) and F16 activations.
/// y is F32 [n_tokens][m].
hipblasStatus_t F16GemmStatus(const __half* w, const __half* x, float* y,
             std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
             hipStream_t stream);

/// RMSNorm with an extra elementwise gamma: out = rms(x) * gamma * mul.
void RmsNormMul(const float* x, const float* gamma, const float* mul,
                float* out, std::uint32_t n_rows, std::uint32_t dim, float eps,
                hipStream_t stream);

/// RoPE over [n_tokens][heads][d] with per-layer theta and rotary width.
void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          std::uint32_t start_pos, float theta, hipStream_t stream);

/// Causal attention over an F16 KV cache with an optional sliding window.
/// q: [n_tokens][heads][d]; k/v caches: [capacity][kv_heads][d] at
/// positions start_pos + t. When window > 0, keys older than
/// (start_pos + t - window + 1) are masked out for query t.
namespace wmma {
// WMMA flash prefill attention (sliding-window geometry only). Returns false
// if the shape is unsupported; the caller then uses the scalar kernel.
bool AttentionWmma(const float* q, const __half* k_cache, const __half* v_cache,
                   float* out, std::uint32_t n_tokens, std::uint32_t start_pos,
                   std::uint32_t d, std::uint32_t kv_stride,
                   std::uint32_t window, hipStream_t stream);
}  // namespace wmma

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               float* out, std::uint32_t n_tokens,
               std::uint32_t start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t d,
               std::uint32_t kv_stride, std::uint32_t window,
               hipStream_t stream);

/// Stores f32 rows into the f16 cache at positions start_pos + t.
void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, std::uint32_t start_pos,
             hipStream_t stream);

/// MoE router for Gemma 4: tmp = rmsnorm(attn_out, eps) * (1/sqrt(hidden))
/// * router_scale, then logits = tmp @ router^T via the caller-supplied
/// GEMM. This kernel produces `tmp` (the normalized, scaled router input).
void RouterInput(const float* attn_out, const float* router_scale,
                 float* tmp, std::uint32_t n_tokens, std::uint32_t hidden,
                 float eps, hipStream_t stream);

/// out[t][i] = sum_s weights[t][s] * expert_out[t*k + s][i] (no shared
/// expert gate — Gemma 4 combines the shared MLP separately).
void MoeEpilogue(const float* expert_out, const float* weights, float* out,
                 std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                 hipStream_t stream);

/// Per-layer output scale: out[t][i] *= scale (single scalar).
void LayerScale(float* out, float scale, std::size_t count,
                hipStream_t stream);

/// Elementwise: out[i] *= mul[i % mul_count].
void MulInplace(float* out, const float* mul, std::size_t count,
                std::size_t mul_count, hipStream_t stream);

/// RoPE with per-pair frequency factors (full-attention layers): the angle
/// for pair i is position * theta^(-2i/d) / freq_factors[i] (ggml yarn:
/// theta/ff, so a 1e30 factor freezes the pair).
void RopeFrequencies(float* x, std::uint32_t n_tokens, std::uint32_t heads,
                     std::uint32_t d, std::uint32_t rotary_dim,
                     const std::uint32_t* start_pos, float theta,
                     const float* freq_factors, hipStream_t stream);

/// Per-token-position rope: each token t uses position start_pos + t.
void RopeFrequenciesMulti(float* x, std::uint32_t n_tokens,
                          std::uint32_t heads, std::uint32_t d,
                          std::uint32_t rotary_dim, std::uint32_t start_pos,
                          float theta, const float* freq_factors,
                          hipStream_t stream);

/// Lossless Q4_0 -> Q8_0 requant of `rows` rows into a Q8_0 buffer
/// (34 bytes per 32 elements). For the WMMA DenseF16Gemm prefill path.
void DequantSlabQ8(const TensorRef& t, std::uint64_t rows, std::uint8_t* dst,
                   hipStream_t stream);

/// Dequantize `rows` rows of `cols` elements from a GGML quant tensor into
/// the F16 staging slab.
void DequantSlab(const TensorRef& t, std::uint64_t rows, __half* dst,
                 hipStream_t stream);

/// Fused n == 1 decode GEMV over quant weights (no staging slabs).
void GemvFused(const void* wdev, int type, std::uint64_t row_bytes,
               std::uint64_t cols, const float* x, float* y, std::uint32_t m,
               hipStream_t stream);

/// Fused n == 1 decode GEMV over dense F16 weights.
void GemvF16Fused(const __half* wdev, std::uint64_t cols, const float* x,
                  float* y, std::uint32_t m, hipStream_t stream);

}  // namespace kernels
}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_KERNELS_KERNELS_HPP_
