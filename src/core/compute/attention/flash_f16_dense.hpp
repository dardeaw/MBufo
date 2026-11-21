#ifndef MBUFO_CORE_COMPUTE_ATTENTION_FLASH_F16_DENSE_HPP_
#define MBUFO_CORE_COMPUTE_ATTENTION_FLASH_F16_DENSE_HPP_

// Dense bidirectional F16-WMMA flash attention (online softmax, no scores
// materialization) for small-head MHA encoders.
//
// Geometry served today: head_dim == 64, any heads >= 1, any n >= 1
// (bidirectional: every query attends every key; tails masked). Q/K/V/O are
// F32 row-major [(heads*n), head_dim] (per-head contiguous [n][d]); the WMMA
// math runs F16 in / F32 accumulate with an F32 online softmax and F16 P,
// mirroring the data flow of the proven qwen dense WMMA kernel (same
// LoadFrag/LDS-exchange/softmax layout, retiled from d=256 GQA-causal to
// d=64 MHA-bidirectional). gfx1151 tuning notes inline in the .hip.cpp.
//
// Dispatcher note (deliberately NOT routed): attention_dispatcher only
// serves quantized blocked-KV geometries (Q8/Q4/Fp8, d=128/256) and its
// iron rule keeps every F16-input call on kLegacy; a dense-F16 d=64 route
// would contradict that rule, so callers branch explicitly (see bge-m3
// session.hip: flash above kFlashMinTokens, legacy SGEMM below).
//
// This header is host-safe: launcher declaration only.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace mbufo::core::compute::flash_f16 {

/// Forward pass. `scale` multiplies QK (typically 1/sqrt(head_dim)).
/// Returns false (no launch) when head_dim != 64, n == 0, heads == 0, or any
/// pointer is null; the caller keeps its fallback path. Stream-ordered.
[[nodiscard]] bool FlashF16DenseMha(const float* q, const float* k,
                                    const float* v, float* out,
                                    std::uint32_t n, std::uint32_t heads,
                                    std::uint32_t head_dim, float scale,
                                    hipStream_t stream);

/// Same kernel over a length-padded tile: every [n_rows][head_dim] tile has `n_rows` as its row stride (the padded
/// length), but only the first `n_valid` rows are real. Pad keys are not attended, pad query rows write zeros, and the
/// key sweep stops at `n_valid`, so cost follows the real length while shapes can be bucketed.
[[nodiscard]] bool FlashF16DenseMhaPadded(const float* q, const float* k, const float* v, float* out,
                                          std::uint32_t n_rows, std::uint32_t n_valid, std::uint32_t heads,
                                          std::uint32_t head_dim, float scale, hipStream_t stream);

/// Batched variant for padded multi-sequence encoders: tiles are [(batch * heads)][n_rows][head_dim] (tile index =
/// b * heads + h) and sequence b has `lens[b]` real rows (device pointer, lens[b] <= n_rows).
[[nodiscard]] bool FlashF16DenseMhaBatched(const float* q, const float* k, const float* v, float* out,
                                           std::uint32_t batch, std::uint32_t n_rows, const std::uint32_t* lens,
                                           std::uint32_t heads, std::uint32_t head_dim, float scale,
                                           hipStream_t stream);

}  // namespace mbufo::core::compute::flash_f16

#endif  // MBUFO_CORE_COMPUTE_ATTENTION_FLASH_F16_DENSE_HPP_
