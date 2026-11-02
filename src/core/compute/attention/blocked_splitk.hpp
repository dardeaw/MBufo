#ifndef GUFO_CORE_COMPUTE_ATTENTION_BLOCKED_SPLITK_HPP_
#define GUFO_CORE_COMPUTE_ATTENTION_BLOCKED_SPLITK_HPP_

// Blocked-Q8 Split-K decode attention for SWA (d == 256) layers: the same
// shared 34-byte blocked KV caches as the scalar direct read (32x int8 codes
// + f16 per-32-block scale, dequantized in registers), but the visible KV
// range is cut into K chunks scanned by parallel blocks with per-chunk
// online-softmax partials and a final deterministic log-sum-exp merge — the
// split philosophy of gemma4's F16 AttentionSplitK(Multi)Kernel, retargeted
// at the narrow-batch decode case where WMMA tiles stay empty and the scalar
// single-block sweep leaves most CUs idle.
//
// SWA window semantics are inherited from the caller: the engine applies the
// same pointer-shift/rebase as the scalar AttentionBlocked path before
// calling, so this kernel always sees dense-causal keys [0, *start_pos].
//
// This header is host-safe (includable from plain C++ TUs): it carries only
// layout tags, tuning constants and the launcher declaration. The device
// codec traits live in blocked_splitk.hip.cpp.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::core::compute::blocked_splitk {

// Element codec tag of the shared blocked Q8 KV layout: symmetric INT8 over
// amax/127 with a per-32-block f16 scale (34 bytes per block).
// Bit-compatible with blocked_kv::KvCodecQ8 and blocked_wmma::KvCodecQ8;
// redeclared as a dependency-free tag so this header stays includable from
// model engines without pulling in the flash-next headers.
struct KvCodecQ8 {
  using Elem = std::uint8_t;
  static constexpr float kRange = 127.0f;
  static constexpr std::uint32_t kBlockBytes = 34;
};

/// Upper token bound of the Split-K decode path (mirrors the F16 Split-K
/// narrow-batch shape: decode and other single-digit forwards). Longer
/// forwards stay on WMMA (prefill) or the scalar direct read.
inline constexpr std::uint32_t kBlockedSplitKMaxTokens = 8;

/// Minimum visible KV length that pays off: shorter contexts stay on the
/// scalar single-block scan, whose launch + merge overhead wins below this.
inline constexpr std::uint32_t kBlockedSplitKMinKv = 1024;

/// Maximum KV splits (matches the F16 Split-K cap).
inline constexpr std::uint32_t kBlockedSplitKMaxSplits = 64;

/// Split-K decode over blocked Q8 caches (SWA geometry only). `k_cache` /
/// `v_cache` and `start_pos` must already carry the SWA window
/// pointer-shift/rebase (identical to the AttentionBlocked call: keys are
/// dense-causal [0, *start_pos]). `kv_len` is the host-visible key count
/// (rebase position + n_tokens) used to size the split; the kernel clamps
/// each chunk to its own device-side `*start_pos + t + 1`. `score_scale`
/// multiplies the QK scores (gemma4 passes 1.0, its f_attention_scale).
/// First version instantiates <KvCodecQ8, 256>; other shapes return false so
/// the caller falls back to the scalar direct read. Never launches an empty
/// grid. The merge folds chunks in fixed order, so reruns are deterministic.
template <typename Codec, std::uint32_t kHeadDim>
bool AttentionBlockedSplitK(const float* q, const std::uint8_t* k_cache,
                            const std::uint8_t* v_cache, float* out,
                            std::uint32_t n_tokens,
                            const std::uint32_t* start_pos,
                            std::uint32_t kv_len, std::uint32_t heads,
                            std::uint32_t kv_heads, std::uint32_t d,
                            hipStream_t stream, float score_scale = 1.0F);

}  // namespace gufo::core::compute::blocked_splitk

#endif  // GUFO_CORE_COMPUTE_ATTENTION_BLOCKED_SPLITK_HPP_
