#ifndef GUFO_CORE_COMPUTE_ATTENTION_FLASH_BLAS_HPP_
#define GUFO_CORE_COMPUTE_ATTENTION_FLASH_BLAS_HPP_

// rocBLAS batched-GEMM flash attention with per-head sinks (head_dim == 64).
// QK^T and P*V ride tensor-core batched GEMMs; a fused online-softmax kernel
// handles causal masking, sliding windows, sinks and the cross-tile reduction.
// K/V stay in F16 device caches; Q is staged to F16 once per forward.
//
// Contract: d == 64, F16 KV [kv_len][kv_heads][d], F32 Q [n][heads][d],
// F32 per-head sinks, GQA. Out-of-contract shapes, null pointers or short
// scratch refuse cleanly (caller keeps its kernel).
//
// Scratch (per call, n <= kFlashBlasMaxTok, any kv_len via K tiling):
//   q_half [n][heads][d] F16, scores [heads][n][Kt] F32,
//   acc [heads][n][d] F32, m/l [heads][n] F32.
// FlashBlasScratchBytes(max_tokens) covers it (Kt = 256 fixed).
//
// This header is host-safe (includable from plain C++ TUs). Pass a
// caller-owned hipblas handle (stays valid across calls; stream is set
// per call from the provided hipStream_t).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <cstdint>

namespace gufo::core::compute::flash_blas {

/// Maximum prefill tokens per call (session chunk ceiling).
inline constexpr std::uint32_t kFlashBlasMaxTok = 256;
/// K rows per tile (scores tile is [heads][n][Kt] F32).
inline constexpr std::uint32_t kFlashBlasKt = 256;

/// Scratch bytes for max_tokens (any kv_len: K tiles stream).
[[nodiscard]] inline constexpr std::size_t FlashBlasScratchBytes(
    std::uint32_t max_tokens) noexcept {
  const std::size_t n = max_tokens;
  return 64u * n * 64u * sizeof(float) +       // qh [heads][n][d] F32
         64u * n * kFlashBlasKt * sizeof(float) +  // scores tile
         64u * n * 64u * sizeof(float) +          // acc
         2u * 64u * n * sizeof(float) +           // m + l
         2u * 64u * kFlashBlasKt * 64u * sizeof(float);  // k_tile + v_tile
}

/// Batched-GEMM flash prefill. kv_len is the host-visible cache length
/// (>= start_pos + n_tokens). window = 0 for dense causal.
bool FlashBlasPrefill(const float* q, const __half* k_cache,
                      const __half* v_cache, const float* sinks, float* out,
                      std::uint32_t n_tokens, std::uint32_t start_pos,
                      std::uint32_t kv_len, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t d,
                      std::uint32_t window, float score_scale,
                      hipStream_t stream, hipblasHandle_t handle,
                      void* scratch, std::size_t scratch_bytes);

}  // namespace gufo::core::compute::flash_blas

#endif  // GUFO_CORE_COMPUTE_ATTENTION_FLASH_BLAS_HPP_
