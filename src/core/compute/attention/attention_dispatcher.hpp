// Attention dispatcher (GGUF plan v2 §4 Phase A).
//
// `LaunchMixedBatchAttention` already dispatches per call; this header makes
// the decision tree explicit and shared: input
// (q_len, kv_len, head_dim, per-side codec, batch composition, window),
// output which kernel runs. The tree carries the measured thresholds:
//
//   prefill q_len >= wmma_min_tokens (32)  -> blocked WMMA
//   decode  q_len <= splitk_max_tokens (8) -> blocked Split-K
//          AND kv_len >= splitk_min_kv (1024)
//   else                                   -> scalar (blocked_kv::AttentionKv)
//   anything unsupported / mixed codecs    -> kLegacy (caller runs legacy)
//
// Threshold provenance (measured, do not retune without re-measuring):
//   32   == gemma4 kBlockedWmmaMinTokens (src/models/gemma4/engine.cpp):
//          prefill at or above this fills a 16-query WMMA tile grid and
//          beats the scalar direct read; shorter prompts stay scalar.
//    8   == blocked_splitk::kBlockedSplitKMaxTokens
//          (src/core/compute/attention/blocked_splitk.hpp): the Split-K
//          decode shape (decode + other single-digit forwards).
// 1024   == blocked_splitk::kBlockedSplitKMinKv: shorter contexts stay on
//          the scalar single-block scan.
//
// Kernel-geometry facts (hardcoded below, with file citations — these are
// kernel properties, not tunable policy):
//   WMMA   supports <Q8,256> + Q4 nibble loads, SWA geometry only
//          (heads == 16, kv_heads == 8, d == 256; static_assert in
//          blocked_wmma.hip.cpp). d == 128 prefill is tile-locked out
//          (kOTilesPerWave == 2*kRowBlocks needs KSteps == 16, i.e.
//          d == 256); those calls run the scalar direct read instead.
//          No Fp8 trait exists.
//   SplitK supports <Q8,256> (heads <= 16) and <Q8,128> (heads <= 32)
//          + Q4 nibble loads (static_assert in blocked_splitk.hip.cpp).
//          No Fp8 trait exists.
//   Scalar (blocked_kv::AttentionKv) serves d == 256 and d == 128
//          (d%32 == 0 required for the 32-elem block addressing), Q8/Fp8
//          codecs plus Q4 nibble rows; anything else is refused by the
//          launcher and the caller falls back.
//
// Iron rules (plan §3.3, enforced structurally):
//   * F16 inputs always return kLegacy: the F16 default path is untouched.
//   * MTP/speculative verify calls pass is_verify = true and always return
//     kLegacy. Mixed-batch callers additionally exclude speculative items
//     before launching (docs/MIXED_BATCH.md P1).
//
// This header is host-safe (no HIP include): the decision is pure arithmetic
// over the input + policy, noexcept, no statics, no device state. Callers
// map their own codec enums onto DispatchCodec.
//
// Env降级阀 (same style as the existing GUFO_*_UNIFIED_BATCH=0 valves):
//   GUFO_ATTN_DISPATCH=0        master kill-switch -> always kLegacy.
//   GUFO_ATTN_WMMA_MIN_TOKENS   override wmma_min_tokens.
//   GUFO_ATTN_SPLITK_MAX_TOKENS override splitk_max_tokens.
//   GUFO_ATTN_SPLITK_MIN_KV     override splitk_min_kv.
// Malformed values are ignored (measured defaults stand).
#ifndef GUFO_CORE_COMPUTE_ATTENTION_ATTENTION_DISPATCHER_HPP_
#define GUFO_CORE_COMPUTE_ATTENTION_ATTENTION_DISPATCHER_HPP_

#include <cstdint>
#include <cstdlib>

namespace gufo::core::compute::attention_dispatcher {

// Which kernel the caller should run for this attention call.
// kLegacy means "run your existing code path unchanged": F16 inputs outside
// the (16, 8, d=128) fused route, MTP verify, sparse/checkpoint callers,
// Fp8 blocked (scalar-only codec, served by the caller's existing scalar
// path), and every unsupported shape. Legacy is always safe; the blocked
// routes are only returned when a kernel with the exact geometry exists.
enum class AttentionRoute : std::uint32_t {
  kLegacy = 0,
  kScalarBlocked = 1,  // blocked_kv::AttentionKv direct read
  kBlockedWmma = 2,    // blocked_wmma::AttentionBlockedWmma prefill
  kBlockedSplitK = 3,  // blocked_splitk::AttentionBlockedSplitK decode
  kFusedFaGqa128 = 4,  // fused_fa_gqa128::LaunchFusedFaGqa128 prefill
};

// Per-side KV codec of the call being routed. Values mirror
// mixed_batch::MixedKvCodec (kF16=0, kQ8=1, kQ4=2); kFp8 has no mixed-batch
// counterpart because the launcher refuses blocked Fp8 outright.
enum class DispatchCodec : std::uint32_t {
  kF16 = 0,
  kQ8 = 1,
  kQ4 = 2,
  kFp8 = 3,
};

// Measured thresholds. Every model fills the same struct (today all three
// use the defaults); per-model overrides stay possible without touching
// the decision function.
struct AttentionDispatchPolicy {
  std::uint32_t wmma_min_tokens = 32;   // prefill rows to take WMMA
  std::uint32_t splitk_max_tokens = 8;  // decode rows to consider Split-K
  std::uint32_t splitk_min_kv = 1024;   // visible keys to pay off Split-K
  bool dispatch_enabled = true;         // false -> always kLegacy

  // Measured defaults with the GUFO_ATTN_* env overrides applied.
  // No statics: the env is re-read per call so test harnesses can force
  // routes (e.g. scalar-vs-WMMA parity) within one process.
  static AttentionDispatchPolicy WithEnvDefaults() {
    AttentionDispatchPolicy p;
    p.ApplyEnvOverrides();
    return p;
  }

  void ApplyEnvOverrides() {
    if (const char* e = std::getenv("GUFO_ATTN_DISPATCH")) {
      if (e[0] == '0' && e[1] == '\0')
        dispatch_enabled = false;
    }
    if (const char* e = std::getenv("GUFO_ATTN_WMMA_MIN_TOKENS")) {
      const unsigned long v = std::strtoul(e, nullptr, 10);
      if (v > 0 && v <= 1048576)
        wmma_min_tokens = static_cast<std::uint32_t>(v);
    }
    if (const char* e = std::getenv("GUFO_ATTN_SPLITK_MAX_TOKENS")) {
      const unsigned long v = std::strtoul(e, nullptr, 10);
      if (v > 0 && v <= 1024)
        splitk_max_tokens = static_cast<std::uint32_t>(v);
    }
    if (const char* e = std::getenv("GUFO_ATTN_SPLITK_MIN_KV")) {
      const unsigned long v = std::strtoul(e, nullptr, 10);
      if (v > 0 && v <= 1048576)
        splitk_min_kv = static_cast<std::uint32_t>(v);
    }
  }
};

// One attention call to route. kv_len is the host-visible key count for
// this call (post window-rebase when the caller pre-shifts; see window).
// window == 0 means dense-causal from the caller's perspective; a nonzero
// SWA window is only served by the WMMA route (it absorbs the shift
// internally) — every other blocked route needs the caller pre-shifted,
// then re-consulted with window == 0 (gemma4 SwaShiftedKV convention).
struct AttentionDispatchInput {
  std::uint32_t q_len = 0;     // query rows of this call
  std::uint32_t kv_len = 0;    // visible keys (rebase position + q_len)
  std::uint32_t head_dim = 0;  // per-head dim
  std::uint32_t heads = 0;     // query heads
  std::uint32_t kv_heads = 0;  // KV heads
  DispatchCodec k_codec = DispatchCodec::kF16;
  DispatchCodec v_codec = DispatchCodec::kF16;
  std::uint32_t window = 0;  // 0 = dense-causal, else SWA window
  bool is_verify = false;    // MTP/speculative verify -> kLegacy, always
};

// Pure decision function: no HIP, no statics, noexcept. Never returns a
// blocked route whose kernel geometry is unsupported; such calls get
// kLegacy and the caller falls back.
inline AttentionRoute RouteAttention(
    const AttentionDispatchInput& in,
    const AttentionDispatchPolicy& policy) noexcept {
  if (!policy.dispatch_enabled)
    return AttentionRoute::kLegacy;
  // Iron rule: MTP/speculative verify never enters a routed blocked kernel.
  if (in.is_verify)
    return AttentionRoute::kLegacy;
  if (in.q_len == 0 || in.kv_len == 0 || in.head_dim == 0 || in.heads == 0 ||
      in.kv_heads == 0 || (in.heads % in.kv_heads) != 0) {
    return AttentionRoute::kLegacy;
  }
  const bool k_f16 = in.k_codec == DispatchCodec::kF16;
  const bool v_f16 = in.v_codec == DispatchCodec::kF16;
  // Dense-F16 fused flash prefill (core fused_fa_gqa128): the one narrow
  // exception to the F16-stays-legacy rule -- exact GQA geometry
  // (heads == 16, kv_heads == 8, head_dim == 128), dense-causal only.
  // The launcher re-checks the geometry and returns false otherwise, so an
  // unsupported call still lands on the caller's legacy chain.
  if (k_f16 && v_f16 && in.window == 0 && in.heads == 16 &&
      in.kv_heads == 8 && in.head_dim == 128) {
    return AttentionRoute::kFusedFaGqa128;
  }
  // F16 default path stays untouched (both sides F16), and mixed
  // F16/blocked batches are refused outright (caller falls back).
  if (k_f16 && v_f16)
    return AttentionRoute::kLegacy;
  if (k_f16 != v_f16)
    return AttentionRoute::kLegacy;
  if (in.k_codec != in.v_codec)
    return AttentionRoute::kLegacy;
  // Blocked Fp8 (E4M3) has no WMMA/Split-K kernel (no CodecTraits<Fp8> in
  // either kernel TU); the scalar path serves it inside the caller's
  // existing legacy code, so route it there.
  if (in.k_codec == DispatchCodec::kFp8)
    return AttentionRoute::kLegacy;
  // Q8/Q4 from here. Geometry support per kernel TU (see header comment):
  // WMMA is d == 256 SWA-geometry only; Split-K covers d == 256
  // (heads <= 16) and d == 128 (heads <= 32); the scalar direct read
  // serves d == 256 and d == 128 (32-elem block addressing).
  const bool wmma_geom =
      in.head_dim == 256 && in.heads == 16 && in.kv_heads == 8;
  const bool splitk_geom = (in.head_dim == 256 && in.heads <= 16) ||
                           (in.head_dim == 128 && in.heads <= 32);
  const bool scalar_geom = in.head_dim == 256 || in.head_dim == 128;
  if (in.window != 0) {
    // Only WMMA absorbs the SWA shift internally. Anything else needs the
    // caller pre-shifted (then re-consulted with window == 0).
    if (in.q_len >= policy.wmma_min_tokens && wmma_geom) {
      return AttentionRoute::kBlockedWmma;
    }
    return AttentionRoute::kLegacy;
  }
  if (in.q_len >= policy.wmma_min_tokens) {
    // Prefill-shaped: WMMA when the geometry exists (SWA layers). Other
    // servable blocked geometries (e.g. d == 128 dense) run the scalar
    // direct read (fused dequant, same math); only unservable shapes stay
    // on kLegacy.
    if (wmma_geom)
      return AttentionRoute::kBlockedWmma;
    return scalar_geom ? AttentionRoute::kScalarBlocked
                       : AttentionRoute::kLegacy;
  }
  if (in.q_len <= policy.splitk_max_tokens &&
      in.kv_len >= policy.splitk_min_kv) {
    // Decode-shaped with a long context: Split-K when the geometry fits,
    // else the scalar direct read (fused dequant, same math).
    if (splitk_geom)
      return AttentionRoute::kBlockedSplitK;
    return scalar_geom ? AttentionRoute::kScalarBlocked
                       : AttentionRoute::kLegacy;
  }
  // Short decode / small prefill: scalar direct read when servable.
  return scalar_geom ? AttentionRoute::kScalarBlocked : AttentionRoute::kLegacy;
}

// Log-friendly route name (for the [mixed]/[attn-dispatch] diags).
inline const char* AttentionRouteName(AttentionRoute route) noexcept {
  switch (route) {
    case AttentionRoute::kLegacy:
      return "legacy";
    case AttentionRoute::kScalarBlocked:
      return "scalar-blocked";
    case AttentionRoute::kBlockedWmma:
      return "blocked-wmma";
    case AttentionRoute::kBlockedSplitK:
      return "blocked-splitk";
    case AttentionRoute::kFusedFaGqa128:
      return "fused-fa-gqa128";
  }
  return "unknown";
}

}  // namespace gufo::core::compute::attention_dispatcher

#endif  // GUFO_CORE_COMPUTE_ATTENTION_ATTENTION_DISPATCHER_HPP_
