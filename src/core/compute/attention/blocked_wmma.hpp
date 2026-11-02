#ifndef GUFO_CORE_COMPUTE_ATTENTION_BLOCKED_WMMA_HPP_
#define GUFO_CORE_COMPUTE_ATTENTION_BLOCKED_WMMA_HPP_

// Blocked-Q8 WMMA prefill attention for SWA (d == 256) layers: the shared
// 34-byte blocked KV caches (32x int8 codes + f16 per-32-block scale) are
// read in place and dequantized in registers, feeding the same F16
// 16x16x16 WMMA S = QK^T / PV path as the F16 prefill kernel. Q stays
// F32->F16; the score multiplier is a launcher parameter (gemma4
// f_attention_scale = 1.0: no 1/sqrt(d)).
//
// This header is host-safe (included from plain C++ TUs): it carries only
// layout tags and the launcher declaration. The device-side codec traits
// live in blocked_wmma.hip.cpp.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::core::compute::blocked_wmma {

// Element codec tag of the shared blocked Q8 KV layout: symmetric INT8 over
// amax/127 with a per-32-block f16 scale (34 bytes per block).
// Bit-compatible with blocked_kv::KvCodecQ8; redeclared as a dependency-free
// tag so this header stays includable from model engines without pulling in
// the flash-next headers.
struct KvCodecQ8 {
  using Elem = std::uint8_t;
  static constexpr float kRange = 127.0f;
  static constexpr std::uint32_t kBlockBytes = 34;
};

/// Blocked-Q8 WMMA prefill attention (SWA geometry only: heads == 16,
/// kv_heads == 8, d == `kHeadDim`). First version instantiates
/// <KvCodecQ8, 256>; full (d == 512) layers stay on the F16 path.
/// `window` is the SWA window (0 = dense causal); the pointer shift is
/// applied inside. `score_scale` multiplies the QK scores (default 1.0).
/// `key_block_bytes` / `value_block_bytes` select each side's block layout
/// (34 = 8-bit codes, 18 = packed 4-bit codes for the GUFO_KV_Q4/V4 grids).
/// Returns false when the shape is unsupported (caller falls back to the
/// scalar direct read); never launches with a zero grid.
template <typename Codec, std::uint32_t kHeadDim>
bool AttentionBlockedWmma(const float* q, const std::uint8_t* k_cache,
                          const std::uint8_t* v_cache, float* out,
                          std::uint32_t n_tokens, std::uint32_t start_pos,
                          std::uint32_t heads, std::uint32_t kv_heads,
                          std::uint32_t window, hipStream_t stream,
                          float score_scale = 1.0F,
                          std::uint32_t key_block_bytes = 34,
                          std::uint32_t value_block_bytes = 34);

}  // namespace gufo::core::compute::blocked_wmma

#endif  // GUFO_CORE_COMPUTE_ATTENTION_BLOCKED_WMMA_HPP_
