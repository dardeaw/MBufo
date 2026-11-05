#ifndef GUFO_MODELS_GPTOSS_KERNELS_KERNELS_HPP_
#define GUFO_MODELS_GPTOSS_KERNELS_KERNELS_HPP_

// GPT-OSS dense-decoder kernels (gfx1151): embedding gather, RMSNorm, NeoX
// YaRN RoPE, dense F16 KV store, causal attention with per-head sinks and an
// optional sliding window, SwiGLU-OAI expert activation with fused expert
// bias adds, and F32->F16 staging for the shared DenseF16Gemm prefill path.
// Quantized GEMM, MoE routing/GEMV and the blocked-Q8 KV store/dequant ride
// the shared core; these kernels handle the elementwise and narrow-band
// work that is specific to the gpt-oss math.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gptoss::kernels {

/// Embedding lookup from a device-mirrored table (Q8_0 / F32 / F16).
void Embed(const void* table, core::GgmlType type, const std::int32_t* ids,
           float* out, std::uint32_t n_tokens, std::uint32_t hidden,
           hipStream_t stream);

/// RMSNorm over rows: out[t] = rms(x[t]) * gamma.
void RmsNorm(const float* x, const float* gamma, float* out,
             std::uint32_t n_tokens, std::uint32_t dim, float eps,
             hipStream_t stream);

/// Plain residual add: residual[i] += x[i] over `count` elements.
void ResidualAdd(const float* x, float* residual, std::size_t count,
                 hipStream_t stream);

/// Row-vector bias add: out[t][j] += bias[j].
void AddRowBias(float* out, const float* bias, std::uint32_t n_rows,
                std::uint32_t cols, hipStream_t stream);

/// NeoX RoPE with a YaRN frequency ramp: pair (x[i], x[i+d/2]) rotates by
/// position * theta^(-2i/d) * multiplier(i), cos/sin scaled by mscale.
/// multiplier(i) ramps 1 -> inv_factor between low and high (HF linear
/// ramp); inv_factor == 1 disables the ramp (plain NeoX).
void RopeYarn(float* x, std::uint32_t n_tokens, std::uint32_t heads,
              std::uint32_t head_dim, std::uint32_t start_pos, float theta,
              float yarn_low, float yarn_high, float yarn_inv_factor,
              float yarn_mscale, hipStream_t stream);

/// F32 -> F16 elementwise (activation staging for DenseF16Gemm).
void F32ToF16(const float* in, __half* out, std::size_t count,
              hipStream_t stream);

/// F32 rows -> F16 KV planes at [pos, pos+n).
void StoreKvF16(const float* src, __half* cache, std::uint32_t n_tokens,
                std::uint32_t dim, std::uint32_t start_pos,
                hipStream_t stream);

/// Dense causal attention over F16 KV with per-head attention sinks and an
/// optional sliding window. q is F32, caches are [kv_len][kv_heads][d] F16
/// planes. Softmax runs over [sink[h], scores...]; when window > 0, keys
/// older than (start_pos + t - window + 1) are masked out for query t.
/// `score_scale` multiplies the QK scores (gpt-oss passes 1/sqrt(d)).
void AttentionF16Sink(const float* q, const __half* k_cache,
                      const __half* v_cache, const float* sinks, float* out,
                      std::uint32_t n_tokens, std::uint32_t start_pos,
                      std::uint32_t heads, std::uint32_t kv_heads,
                      std::uint32_t d, std::uint32_t window,
                      hipStream_t stream, float score_scale);

/// Fused expert activation with bias adds (gpt-oss SwiGLU-OAI):
/// out[s][j] = oai(gate[s][j] + gate_b[e][j], up[s][j] + up_b[e][j]),
/// oai(x, y) = (min(x,7) / (1 + exp(-1.702*min(x,7)))) * (clamp(y,-7,7) + 1),
/// e = ids[s] (negative ids write zeros). gate/up/out are [slots][ff];
/// gate_b/up_b are [ff][experts] row-major per expert.
void SwigluOaiBias(const float* gate, const float* up,
                   const std::int32_t* ids, const float* gate_b,
                   const float* up_b, float* out, std::uint32_t slots,
                   std::uint32_t ff, hipStream_t stream);

/// Expert down-projection bias add: out[s][i] += down_b[e][i], e = ids[s].
/// down_b is [hidden][experts] row-major per expert.
void DownBiasAdd(float* out, const std::int32_t* ids, const float* down_b,
                 std::uint32_t slots, std::uint32_t hidden,
                 hipStream_t stream);

}  // namespace gufo::models::gptoss::kernels

#endif  // GUFO_MODELS_GPTOSS_KERNELS_KERNELS_HPP_
