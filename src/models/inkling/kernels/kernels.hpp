#ifndef MBUFO_MODELS_INKLING_KERNELS_KERNELS_HPP_
#define MBUFO_MODELS_INKLING_KERNELS_KERNELS_HPP_

#include <cstddef>
#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/compute/attention/blocked_kv_codec.hpp"

namespace mbufo::models::inkling::kernels {

// Cache view for the shared blocked-Q8 KV rows (see blocked_kv_codec.hpp:
// the view pins the byte geometry; the Q8 entry points below only accept
// views, never raw pointers + ad-hoc strides).
using BlockedKvQ8View = ::mbufo::core::compute::blocked_kv::BlockedKvView<
    ::mbufo::core::compute::blocked_kv::KvCodecQ8>;
// K8V4 value rows (18B packed-nibble blocks); keys stay BlockedKvQ8View.
using BlockedKvQ4View = ::mbufo::core::compute::blocked_kv::BlockedKvView<
    ::mbufo::core::compute::blocked_kv::KvCodecQ4>;

[[nodiscard]] const char* KernelsVersion();

// --- Elementwise norms (model-private; core carries no callable HIP RMSNorm
// for raw buffers, and the no-gamma form is inkling-unique) ---
void RmsNormGamma(const float* x, const float* weight, float* out,
                  std::uint32_t n_tokens, std::uint32_t hidden, float eps,
                  hipStream_t stream);
// RMSNorm without gamma (value stream, gate inputs, message keys).
void RmsNormNoGamma(const float* x, float* out, std::uint32_t n_tokens,
                    std::uint32_t hidden, float eps, hipStream_t stream);

void AddResidual(float* a, const float* b, std::size_t count,
                 hipStream_t stream);

// --- Causal depthwise short-convolutions (inkling-unique) ---
// Static taps: out = (conv(x) + x) * norm (sliding K/V and both post-convs
// use norm = 1/sqrt(5)). `state` is [dim x 3] rolling previous inputs,
// updated in place; taps are [4 x dim] row-major (tap, dim).
void StaticSconvForward(const float* x, float* state, const float* taps,
                        float* out, std::uint32_t n_tokens, std::uint32_t dim,
                        float norm, hipStream_t stream, float* scratch = nullptr,
                        float* pfx_out = nullptr, std::size_t pfx_stride = 0);
// Dynamic taps (global K/V): tap_eff[head, t] = taps[t][dim] + delta[head, t]
// with delta [n_tokens x n_heads x 4]; out = (dynconv(x) + x) * norm with
// norm = 1/3 (1/sqrt(9)). `state` layout matches the static form.
void DynamicSconvForward(const float* x, float* state, const float* taps,
                         const float* delta, float* out,
                         std::uint32_t n_tokens, std::uint32_t dim,
                         std::uint32_t n_heads, float norm,
                         hipStream_t stream, float* scratch = nullptr,
                         float* pfx_out = nullptr, std::size_t pfx_stride = 0);

// --- Clips / rescale (inkling-unique; scale tensors are model weights) ---
// Q path: out = x * scale[d, h] elementwise, then per-head L2 clip to `clip`.
void QScaleClip(float* x, const float* scale, std::uint32_t n_tokens,
                std::uint32_t n_heads, std::uint32_t head_dim, float clip,
                hipStream_t stream);
// K path: per-head L2 clip to `clip` (no learned scale).
void L2Clip(float* x, std::uint32_t n_tokens, std::uint32_t n_heads,
            std::uint32_t head_dim, float clip, hipStream_t stream);
// Rel path: r[n, heads, 16] *= clip[e, h], per-(n,h) L2 clip to `clip_val`
// (8.0), then *= tau[t] (LogScale; 1.0 on sliding layers).
void FillLogScaleTau(float* tau, std::uint32_t n_tokens,
                     std::uint32_t start_pos, std::uint32_t floor_val,
                     float alpha, bool is_global, hipStream_t stream);

void RelClipTau(float* r, const float* clip, const float* tau,
                std::uint32_t n_tokens, std::uint32_t n_heads, float clip_val,
                hipStream_t stream);

// --- Attention (model-private decode/prefill over F16 KV planes) ---
// The core split-K path cannot serve inkling: it has no qpg == 3 entry and
// its sinks are per-head statics, while inkling sinks are per-token logits.
// Fused causal attention with GQA (qpg == 3), optional sliding window
// (0 = dense), per-token per-head sinks (value 0), and relative-bias gather:
//   bias[h][d] = (0 <= dist < extent) ? <rel[t,h,:], bank[:,dist]> : 0
// with dist = query_pos - key_pos, rel [n_tokens, n_heads, 16] (post
// RelClipTau), bank [16, extent] F32 (attn_rel_bias). `kv_len` is the
// host-visible cache length (>= start_pos + n_tokens).
void AttentionDecode(const float* q, const __half* k_cache,
                     const __half* v_cache, const float* sinks,
                     const float* rel, const float* rel_bank, float* out,
                     std::uint32_t start_pos, std::uint32_t kv_len,
                     std::uint32_t window, std::uint32_t n_heads,
                     std::uint32_t n_kv_heads, std::uint32_t head_dim,
                     std::uint32_t rel_extent, float scale,
                     hipStream_t stream, float* splitk_scratch = nullptr);
void AttentionPrefill(const float* q, const __half* k_cache,
                      const __half* v_cache, const float* sinks,
                      const float* rel, const float* rel_bank, float* out,
                      std::uint32_t n_tokens, std::uint32_t start_pos,
                      std::uint32_t kv_len, std::uint32_t window,
                      std::uint32_t n_heads, std::uint32_t n_kv_heads,
                      std::uint32_t head_dim, std::uint32_t rel_extent,
                      float scale, hipStream_t stream,
                      float* splitk_scratch = nullptr);

// out *= sigmoid(gate[t, h]) broadcast over the head dim.
void SigmoidGateMul(float* attn_out, const float* gate, std::uint32_t n_tokens,
                    std::uint32_t n_heads, std::uint32_t head_dim,
                    hipStream_t stream);

// Row-wise scaling: x[t, :] *= scale[t].
void ScaleRows(float* x, const float* scale, std::uint32_t n_tokens,
               std::uint32_t row_elems, hipStream_t stream);

// Snapshot push: bank[t * stride + slot * hidden + i] = h[t * hidden + i].
void SnapPush(const float* h, float* bank, std::uint32_t n_tokens,
              std::uint32_t hidden, std::uint32_t stride, std::uint32_t slot,
              hipStream_t stream);

void StoreKvF16(const float* k_in, const float* v_in, __half* k_cache,
                __half* v_cache, std::uint32_t n_tokens,
                std::uint32_t start_pos, std::uint32_t n_kv_heads,
                std::uint32_t head_dim, hipStream_t stream,
                std::uint32_t wrap = 0);

// --- Sigmoid top-8 router (inkling-unique selection/renorm) ---
// p = sigmoid(logit); select top-8 by (p + bias); weights = p / (sum + 1e-20)
// * route_scale (bias is selection-only, matching transformers InklingTopkRouter
// with no shared experts: softmax over top-k of logsigmoid == renorm sigmoids).
void SigmoidRouterTop8(const float* logits, const float* bias,
                       std::int32_t* ids, float* weights,
                       std::uint32_t n_tokens, std::uint32_t n_experts,
                       float route_scale, hipStream_t stream);

// --- Situ activation (inkling-unique fused form) ---
// act[i] = softcap4(gate[i]) * sigmoid(gate[i]) * softcap25(up[i]),
// gateup rows are [gate(ff), up(ff)]; softcap(c, x) = c * tanh(x / c).
// (Fused equivalent of two core SoftCap calls plus a sigmoid multiply.)
void MoeEpilogueHalf(const __half* expert_out, const float* weights,
                     float* out, std::uint32_t n_tokens, std::uint32_t k,
                     std::uint32_t dim, hipStream_t stream);
void SituFuseHalfToF16(const __half* gateup, __half* act16, std::uint32_t rows,
                       std::uint32_t ff, hipStream_t stream);
void SituFuseToF16(const float* gateup, __half* act16, std::uint32_t rows,
                   std::uint32_t ff, hipStream_t stream);

void SituFuse(const float* gateup, float* act, std::uint32_t rows,
              std::uint32_t ff, hipStream_t stream);

// --- MoE row movement (layout helpers; math stays in core GEMMs) ---
void GatherRows(const float* x, const std::int32_t* list, float* out,
                std::uint32_t batch, std::uint32_t hidden, hipStream_t stream);
void ScatterAddWeighted(const float* x, const float* weights,
                        const std::int32_t* list, float* out,
                        std::uint32_t batch, std::uint32_t hidden,
                        hipStream_t stream);

// --- Loop message / residual attention (inkling-unique) ---
// Multi-query attention over `n_states` state rows (non-causal, no mask):
//   k = v = RmsNormNoGamma(state) (keys and values both read normalized
//   states, matching the reference which attends over normalized states),
//   score = <query, k> * scale, softmax, out = mean over queries.
// states is [n_tokens, state_stride, hidden]; queries [n_queries, hidden].
void MultiQueryResAttn4(const float* states, std::uint32_t n_states,
                        std::uint32_t state_stride, const float* q0,
                        const float* q1, const float* q2, const float* q3,
                        float* out0, float* out1, float* out2, float* out3,
                        std::uint32_t n_tokens, std::uint32_t hidden,
                        float scale, hipStream_t stream);

void MultiQueryResAttn(const float* states, std::uint32_t n_states,
                       std::uint32_t state_stride, const float* queries,
                       std::uint32_t n_queries, float* out,
                       std::uint32_t n_tokens, std::uint32_t hidden,
                       float scale, hipStream_t stream);

// Same attention residual mix, reading `n_bank` states from `bank` plus an
// optional `extra_state` (e.g. current `hidden`) without a separate SnapPush.
// When `push_extra_to_bank` is true, also stores `extra_state` into
// `bank[n_bank * hidden]`. When `out1` and `out2` are non-null (and
// `n_queries == 3`), writes each query's output separately.
void MultiQueryResAttnDirect(const float* bank, std::uint32_t n_bank,
                             std::uint32_t bank_stride,
                             const float* extra_state,
                             bool push_extra_to_bank, const float* queries,
                             std::uint32_t n_queries, float* out0,
                             float* out1, float* out2, std::uint32_t n_tokens,
                             std::uint32_t hidden, float scale,
                             hipStream_t stream);

// Gate fusion: fg = tanh(fg_raw), mg = sigmoid(mg_raw),
// out = (1 - fg) * h + mg * msg. (mg = sigmoid is a documented assumption;
// the spec pins tanh only for the forget path.)
void FuseGateProjApply(const float* h, const float* msg, const float* g64,
                       const __half* forget_w, const __half* gain_w, float* out,
                       std::uint32_t n_tokens, std::uint32_t hidden,
                       std::uint32_t rank, hipStream_t stream);

void GateFuseApply(const float* h, const float* msg, const float* fg_raw,
                   const float* mg_raw, float* out, std::uint32_t n_tokens,
                   std::uint32_t hidden, hipStream_t stream);

// concat [RmsNormNoGamma(h), RmsNormNoGamma(msg)] into pair rows.
void FusedPairNormDown(const float* h, const float* msg, const __half* down_w,
                       float* out, std::uint32_t n_tokens, std::uint32_t hs,
                       std::uint32_t rank, float eps, hipStream_t stream);

void PairNormConcat(const float* h, const float* msg, float* pair,
                    std::uint32_t n_tokens, std::uint32_t hidden, float eps,
                    hipStream_t stream);

void BuildRoutedTiles(const std::uint32_t* counts, std::uint32_t n_experts,
                      std::uint32_t tile_rows, std::int32_t* tiles,
                      std::uint32_t* n_tiles, hipStream_t stream);

void FillLogScaleTauByPos(float* tau, const std::uint32_t* d_pos,
                          std::uint32_t floor_val, float alpha, bool is_global,
                          hipStream_t stream);

void StoreKvF16ByPos(const float* k_in, const float* v_in, __half* k_cache,
                     __half* v_cache, const std::uint32_t* d_pos,
                     std::uint32_t n_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream, std::uint32_t wrap = 0);

void AttentionDecodeByPos(const float* q, const __half* k_cache,
                          const __half* v_cache, const float* sinks,
                          const float* rel, const float* rel_bank, float* out,
                          const std::uint32_t* d_pos, std::uint32_t window,
                          std::uint32_t n_heads, std::uint32_t n_kv_heads,
                          std::uint32_t head_dim, std::uint32_t rel_extent,
                          float scale, hipStream_t stream,
                      float* splitk_scratch = nullptr);

void CopyKvSlotByPos(__half* cache, __half* backup, const std::uint32_t* d_pos,
                     std::uint32_t kv_dim, bool to_backup, hipStream_t stream,
                     std::uint32_t wrap = 0);

// --- Blocked-Q8 KV (shared 34B layout; codec-identical to
// core::compute::blocked_kv::KvCodecQ8) ---
// Store float K/V rows as blocked-Q8 (32x int8 + f16 scale per 32 elems).
// Strides come from the views (MakeBlockedKvView); `wrap` is the ring
// window (0 = dense, same KvSlot rule as F16).
void StoreKvQ8ByPos(const float* k_in, const float* v_in,
                     const BlockedKvQ8View& k_cache,
                     const BlockedKvQ8View& v_cache,
                     const std::uint32_t* d_pos, std::uint32_t n_kv_heads,
                     std::uint32_t head_dim, hipStream_t stream,
                     std::uint32_t wrap = 0);
// Fused-dequant Q8 twins of AttentionDecode/Prefill(/ByPos): identical
// sinks/rel-bias/window/qpg==3 semantics, K/V read from blocked-Q8 rows
// (`row_stride` bytes each) and dequantized in registers.
void AttentionDecodeQ8(const float* q, const BlockedKvQ8View& k_cache,
                       const BlockedKvQ8View& v_cache, const float* sinks,
                       const float* rel, const float* rel_bank, float* out,
                       std::uint32_t start_pos, std::uint32_t window,
                       std::uint32_t n_heads, std::uint32_t n_kv_heads,
                       std::uint32_t head_dim, std::uint32_t rel_extent,
                       float scale, hipStream_t stream,
                       float* splitk_scratch);
void AttentionPrefillQ8(const float* q, const BlockedKvQ8View& k_cache,
                        const BlockedKvQ8View& v_cache, const float* sinks,
                        const float* rel, const float* rel_bank, float* out,
                        std::uint32_t n_tokens, std::uint32_t start_pos,
                        std::uint32_t window, std::uint32_t n_heads,
                        std::uint32_t n_kv_heads, std::uint32_t head_dim,
                        std::uint32_t rel_extent, float scale,
                        hipStream_t stream, float* splitk_scratch);
void AttentionDecodeQ8ByPos(const float* q, const BlockedKvQ8View& k_cache,
                            const BlockedKvQ8View& v_cache, const float* sinks,
                            const float* rel, const float* rel_bank,
                            float* out, const std::uint32_t* d_pos,
                            std::uint32_t window, std::uint32_t n_heads,
                            std::uint32_t n_kv_heads, std::uint32_t head_dim,
                            std::uint32_t rel_extent, float scale,
                            hipStream_t stream, float* splitk_scratch);
void AttentionWave32Q8(const float* q, const BlockedKvQ8View& k_cache,
                       const BlockedKvQ8View& v_cache, const float* sinks,
                       const float* rel, const float* rel_bank, float* out,
                       std::uint32_t start_pos, std::uint32_t window,
                       std::uint32_t n_tokens, std::uint32_t n_q_heads,
                       std::uint32_t n_kv_heads, std::uint32_t rel_extent,
                       float scale, hipStream_t stream);
void CopyKvSlotQ8ByPos(const BlockedKvQ8View& cache,
                       const BlockedKvQ8View& backup,
                       const std::uint32_t* d_pos,
                       bool to_backup, hipStream_t stream,
                       std::uint32_t wrap = 0);

// --- K8V4 overloads (keys stay Q8; values use the 18B nibble codec).
// Same signatures with a Q4 value view; overload resolution picks the
// kernel instantiation, so call sites stay identical apart from the view.
void StoreKvQ8ByPos(const float* k_in, const float* v_in,
                    const BlockedKvQ8View& k_cache,
                    const BlockedKvQ4View& v_cache,
                    const std::uint32_t* d_pos, std::uint32_t n_kv_heads,
                    std::uint32_t head_dim, hipStream_t stream,
                    std::uint32_t wrap = 0);
void AttentionDecodeQ8(const float* q, const BlockedKvQ8View& k_cache,
                       const BlockedKvQ4View& v_cache, const float* sinks,
                       const float* rel, const float* rel_bank, float* out,
                       std::uint32_t start_pos, std::uint32_t window,
                       std::uint32_t n_heads, std::uint32_t n_kv_heads,
                       std::uint32_t head_dim, std::uint32_t rel_extent,
                       float scale, hipStream_t stream,
                       float* splitk_scratch);
void AttentionPrefillQ8(const float* q, const BlockedKvQ8View& k_cache,
                        const BlockedKvQ4View& v_cache, const float* sinks,
                        const float* rel, const float* rel_bank, float* out,
                        std::uint32_t n_tokens, std::uint32_t start_pos,
                        std::uint32_t window, std::uint32_t n_heads,
                        std::uint32_t n_kv_heads, std::uint32_t head_dim,
                        std::uint32_t rel_extent, float scale,
                        hipStream_t stream, float* splitk_scratch);
void AttentionDecodeQ8ByPos(const float* q, const BlockedKvQ8View& k_cache,
                            const BlockedKvQ4View& v_cache, const float* sinks,
                            const float* rel, const float* rel_bank,
                            float* out, const std::uint32_t* d_pos,
                            std::uint32_t window, std::uint32_t n_heads,
                            std::uint32_t n_kv_heads, std::uint32_t head_dim,
                            std::uint32_t rel_extent, float scale,
                            hipStream_t stream, float* splitk_scratch);
void AttentionWave32Q8(const float* q, const BlockedKvQ8View& k_cache,
                       const BlockedKvQ4View& v_cache, const float* sinks,
                       const float* rel, const float* rel_bank, float* out,
                       std::uint32_t start_pos, std::uint32_t window,
                       std::uint32_t n_tokens, std::uint32_t n_q_heads,
                       std::uint32_t n_kv_heads, std::uint32_t rel_extent,
                       float scale, hipStream_t stream);
void CopyKvSlotQ8ByPos(const BlockedKvQ4View& cache,
                       const BlockedKvQ4View& backup,
                       const std::uint32_t* d_pos,
                       bool to_backup, hipStream_t stream,
                       std::uint32_t wrap = 0);

}  // namespace mbufo::models::inkling::kernels

#endif  // MBUFO_MODELS_INKLING_KERNELS_KERNELS_HPP_
