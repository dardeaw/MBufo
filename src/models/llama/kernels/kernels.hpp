#ifndef GUFO_MODELS_LLAMA_KERNELS_KERNELS_HPP_
#define GUFO_MODELS_LLAMA_KERNELS_KERNELS_HPP_

// Llama dense-decoder kernels (gfx1151): embedding gather, RMSNorm,
// standard NeoX RoPE, SiLU-gated MLP activation, dense F16 KV store,
// dense F16 causal attention, and F32->F16 conversion for the shared
// DenseF16Gemm prefill path. The blocked-Q8 KV write/read ride the shared
// core (blocked_kv::QuantizeKvRows / dispatcher), not these kernels.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::llama::kernels {

/// Embedding lookup from a device-mirrored table (Q8_0 / F32 / F16).
/// `ids` are device token ids; one gathered row per id.
void Embed(const void* table, core::GgmlType type, const std::int32_t* ids,
           float* out, std::uint32_t n_tokens, std::uint32_t hidden,
           hipStream_t stream);

/// RMSNorm over rows: out[t] = rms(x[t]) * gamma.
void RmsNorm(const float* x, const float* gamma, float* out,
             std::uint32_t n_tokens, std::uint32_t dim, float eps,
             hipStream_t stream);

/// Fused post norm + residual: residual[t] += rms(x[t]) * gamma.
void RmsNormAdd(const float* x, const float* gamma, float* residual,
                std::uint32_t n_tokens, std::uint32_t dim, float eps,
                hipStream_t stream);

/// Plain residual add (llama has no post-attention/post-FFN norms):
/// residual[i] += x[i] over `count` elements.
void ResidualAdd(const float* x, float* residual, std::size_t count,
                 hipStream_t stream);

/// Standard NeoX RoPE over the full head dim (llama arch): pair
/// (x[i], x[i+d/2]) rotates by position * theta^(-2i/d), i in [0, d/2).
void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t head_dim, std::uint32_t start_pos, float theta,
          hipStream_t stream);

/// out[i] = silu(gate[i]) * up[i] over `count` elements.
void SiluMul(const float* gate, const float* up, float* out,
             std::size_t count, hipStream_t stream);

/// F32 -> F16 elementwise (activation staging for DenseF16Gemm).
void F32ToF16(const float* in, __half* out, std::size_t count,
              hipStream_t stream);

/// F32 rows -> F16 KV planes at [pos, pos+n).
void StoreKvF16(const float* src, __half* cache, std::uint32_t n_tokens,
                std::uint32_t dim, std::uint32_t start_pos,
                hipStream_t stream);

/// Dense causal attention over F16 KV (the f16 default path): q is F32,
/// caches are [kv_len][kv_heads][d] F16 planes. `score_scale` multiplies
/// the QK scores (llama passes 1/sqrt(d)).
void AttentionF16(const float* q, const __half* k_cache,
                  const __half* v_cache, float* out, std::uint32_t n_tokens,
                  std::uint32_t start_pos, std::uint32_t heads,
                  std::uint32_t kv_heads, std::uint32_t d, hipStream_t stream,
                  float score_scale);

}  // namespace gufo::models::llama::kernels

#endif  // GUFO_MODELS_LLAMA_KERNELS_KERNELS_HPP_
