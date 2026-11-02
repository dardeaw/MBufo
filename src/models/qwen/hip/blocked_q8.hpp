#ifndef GUFO_MODELS_QWEN_HIP_BLOCKED_Q8_HPP_
#define GUFO_MODELS_QWEN_HIP_BLOCKED_Q8_HPP_

#include <cstddef>
#include <cstdint>
#include <string>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#include "src/core/model_config.hpp"

namespace gufo::hip {

// Shared blocked-Q8 attention KV for Qwen attention layers (opt-in q8 path).
// Element codec: symmetric INT8 (amax/127) with a per-32-block f16 scale,
// 34 bytes per block -- the same layout as core/compute/attention. Only
// head_dim == 256 layers are covered (the shared direct-read kernel is
// d == 256-only); every other shape must stay on the F16 path.
inline constexpr std::uint32_t kQwenBlockedQ8BlockBytes{34};
inline constexpr float kQwenBlockedQ8Range{127.0F};

// Loud prerequisite check for the q8 path. Returns false with a reason when
// the model cannot run blocked-Q8 attention (wrong head_dim, no attention
// layers); never silently degrades.
[[nodiscard]] bool ValidateQwenBlockedQ8Config(const core::ModelConfig& config,
                                              std::string* error);

#if defined(ENGINE_ENABLE_HIP)
// Appends this chunk's final (post-norm, post-RoPE) K/V rows into the
// layer's blocked caches. `k_rows`/`v_rows` hold `n_tokens` packed rows of
// `kv_heads * head_dim` floats; `d_pos` is a device scalar with the chunk's
// base position. `blocked_k`/`blocked_v` point at the layer's plane bases.
void WriteQwenBlockedKv(const float* k_rows, const float* v_rows,
                        std::uint8_t* blocked_k, std::uint8_t* blocked_v,
                        std::uint32_t n_tokens, std::uint32_t kv_heads,
                        std::uint32_t head_dim, const std::uint32_t* d_pos,
                        hipStream_t stream);

// Decode read: shared blocked direct-read (fused dequant, 1/sqrt(d) scale,
// no F16 staging) plus the SiLU gate the F16 decode kernels fuse into the
// output. `out` holds `n_tokens * heads * head_dim` floats; `gate` may be
// null (no gating) but is always present for full-attention layers.
void ReadQwenBlockedAttention(const float* q, const std::uint8_t* blocked_k,
                              const std::uint8_t* blocked_v, const float* gate,
                              float* out, std::uint32_t n_tokens,
                              const std::uint32_t* d_pos, std::uint32_t heads,
                              std::uint32_t kv_heads, std::uint32_t head_dim,
                              hipStream_t stream);

// Elementwise out[i] *= sigmoid(gate[i]) over `n` floats.
void LaunchSigmoidGateMultiply(float* out, const float* gate, std::size_t n,
                               hipStream_t stream);
#endif

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_BLOCKED_Q8_HPP_
