#ifndef GUFO_CORE_COMPUTE_GEMM_DISPATCHER_HPP_
#define GUFO_CORE_COMPUTE_GEMM_DISPATCHER_HPP_

// Phase B-3 GEMM dispatcher (GGUF通用化.md v2 §4): a pure routing decision,
// no kernels. Callers keep their legacy path for kLegacy, so wiring the
// dispatcher in is bit-identical until a route engages.
//
//   * small m / decode   -> kLegacy      (existing sdot4 GEMV, untouched)
//   * prefill Q8 Wave64   -> kW8A8Varlen  (host-side 16-split, §3.2)
//   * prefill F16/BF16    -> kHipBlasLt   (plan-cache runner, staged)
//   * MoE                 -> kLegacy      (existing fused MoE, untouched)
//
// quant_code uses the GUFO_GEMM_* codes from gemm.hpp; this header is
// included after those defines (see gemm.hpp).

#include <cstddef>

namespace gufo::core::compute::gemm {

// W8A8 varlen fixed shape: the Wave64 kernel family instantiation moved here
// verbatim (see w8a8_wmma.hip). main = batch & ~15 runs the original kernel
// binary; tail (<= 15 by construction) runs the scalar tail kernel, which
// covers k <= 10240 (proven limits of the tail it was copied from).
constexpr std::size_t kW8A8VarlenM = 2560;
constexpr std::size_t kW8A8VarlenK = 6144;
constexpr std::size_t kW8A8VarlenAlign = 16;
constexpr std::size_t kW8A8VarlenMaxK = 10240;

enum class PrefillRoute { kLegacy, kW8A8Varlen, kHipBlasLt };

[[nodiscard]] inline bool W8A8VarlenCovers(std::size_t m, std::size_t k) {
  return m == kW8A8VarlenM && k == kW8A8VarlenK && (k % 32 == 0);
}

[[nodiscard]] inline PrefillRoute DecidePrefillRoute(std::size_t m,
                                                     std::size_t /*batch*/,
                                                     std::size_t k,
                                                     int quant_code) {
  if (quant_code == GUFO_GEMM_Q8_0 && W8A8VarlenCovers(m, k)) {
    return PrefillRoute::kW8A8Varlen;
  }
  if (quant_code == GUFO_GEMM_F16 || quant_code == GUFO_GEMM_BF16) {
    return PrefillRoute::kHipBlasLt;
  }
  return PrefillRoute::kLegacy;
}

// Host-side 16-split (§3.2): the aligned prefix keeps the original kernel
// and its launch geometry; only the tail (if any) takes another path.
[[nodiscard]] inline void W8A8SplitBatch(std::size_t batch,
                                         std::size_t* main_out,
                                         std::size_t* tail_out) {
  const std::size_t tail = batch & (kW8A8VarlenAlign - 1);
  *main_out = batch - tail;
  *tail_out = tail;
}

// Narrow Q8_0 decode route (narrow-gemm): the shared narrow-Q8 core owns
// the ungated 1..8-token Q8_0 x prequantized-Q8_1 GEMV with K % 32 == 0 and an
// explicit activation row stride (>= K / 32 blocks). The single-row latency
// specialization (rows <= 64, K % 5120 == 0) is excluded: its accumulation
// order differs, so it is not covered even though the shape test would pass.
// Gated (fused SwiGLU), HC/matrix geometries and N > 8 tiles stay legacy;
// callers enforce those model-local exclusions before consulting this route.
enum class NarrowDecodeRoute { kLegacy, kNarrowQ80, kNarrowQ80Prequant };

[[nodiscard]] inline NarrowDecodeRoute DecideNarrowDecodeRoute(
    bool weight_q8_0, bool act_preq_q8_1, bool gated, std::size_t n_tokens,
    std::size_t k, std::size_t rows, std::size_t stride_blocks) {
  if (!weight_q8_0 || !act_preq_q8_1 || gated) {
    return NarrowDecodeRoute::kLegacy;
  }
  if (n_tokens < 1 || n_tokens > 8) {
    return NarrowDecodeRoute::kLegacy;
  }
  if (k == 0 || k % 32 != 0) {
    return NarrowDecodeRoute::kLegacy;
  }
  if (rows == 0 || stride_blocks < k / 32) {
    return NarrowDecodeRoute::kLegacy;
  }
  if (n_tokens == 1 && rows <= 64 && k % 5120 == 0) {
    return NarrowDecodeRoute::kLegacy;
  }
  return NarrowDecodeRoute::kNarrowQ80;
}


// Prequantized-activation branch (preq-q81): same geometry as
// DecideNarrowDecodeRoute but for F32 activations that need the shared
// quantize pass first. Q8_0 weights only; ungated; 1..8 tokens; K % 32 == 0;
// rows > 0. The single-row latency specialization (rows <= 64, K % 5120 == 0)
// and the HC 320x10240 geometry stay legacy (different accumulation order /
// prefetch). Returns kNarrowQ80Prequant when the caller should run
// PrequantizeQ8_1 -> LaunchNarrowQ80Decode; otherwise kLegacy and the caller
// keeps its F32-activation path bit-identical.
[[nodiscard]] inline NarrowDecodeRoute DecideNarrowDecodePrequantRoute(
    bool weight_q8_0, bool gated, std::size_t n_tokens, std::size_t k,
    std::size_t rows) {
  if (!weight_q8_0 || gated) {
    return NarrowDecodeRoute::kLegacy;
  }
  if (n_tokens < 1 || n_tokens > 8) {
    return NarrowDecodeRoute::kLegacy;
  }
  if (k == 0 || k % 32 != 0) {
    return NarrowDecodeRoute::kLegacy;
  }
  if (rows == 0) {
    return NarrowDecodeRoute::kLegacy;
  }
  if (n_tokens == 1 && rows <= 64 && k % 5120 == 0) {
    return NarrowDecodeRoute::kLegacy;
  }
  if (rows == 320 && k == 10240) {
    return NarrowDecodeRoute::kLegacy;
  }
  return NarrowDecodeRoute::kNarrowQ80Prequant;
}

}  // namespace gufo::core::compute::gemm

#endif  // GUFO_CORE_COMPUTE_GEMM_DISPATCHER_HPP_
