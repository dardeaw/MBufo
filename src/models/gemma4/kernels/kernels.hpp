#ifndef GUFO_MODELS_GEMMA4_KERNELS_KERNELS_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_KERNELS_HPP_

#include <cstdint>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "src/models/gemma4/weights.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"

namespace gufo::models::gemma4 {
namespace kernels {

/// Gemma 4 uses GELU-tanh (exact tanh approximation), not the quick-sigmoid
/// SwiGLU used by Qwen. out[i] = gelu(gate[i]) * up[i] over `count` pairs.
void GeluTanhSwiglu(float* gate, const float* up, std::size_t count,
                    hipStream_t stream);

/// act[i] = f16(gelu_tanh(gate[i]) * up[i]) in one pass. Bit-identical to
/// GeluTanhSwiglu followed by F32ToF16; used by the compact MoE prefill to
/// skip the F32 intermediate and one launch.
void SwigluF16(const float* gate, const float* up, __half* act,
               std::size_t count, hipStream_t stream);
/// act[c*ff+j] = f16(silu(comb[c*2ff+j]) * comb[c*2ff+ff+j]) over `rows`
/// compact rows. Fused split+silu for the Q4-direct grouped MoE (the grouped
/// gate/up GEMM writes gate and up stacked); same rounding chain as
/// SwigluF16.
void SwigluSplitF16(const float* combined, __half* act, std::size_t rows,
                    std::uint32_t ff, hipStream_t stream);

/// out[i] = gelu_tanh(x[i]) over `count` elements.
void GeluTanh(float* x, std::size_t count, hipStream_t stream);

/// Elementwise: out[i] = tanh(x[i] / cap) * cap over `count` elements.
void SoftCap(float* x, float cap, std::size_t count, hipStream_t stream);

/// Elementwise add: out[i] += add[i] over `count` elements.
void AddInplace(float* out, const float* add, std::size_t count,
                hipStream_t stream);

/// Elementwise scale: out[i] *= scale over `count` elements.
void ScaleInplace(float* out, float scale, std::size_t count,
                  hipStream_t stream);

/// RMSNorm over rows: out[t] = rms(x[t]) * gamma. `groups` independent
/// norms share the gamma row (gamma may be null).
void RmsNorm(const float* x, const float* gamma, float* out,
             std::uint32_t n_rows, std::uint32_t dim, float eps,
             hipStream_t stream);

/// Fused post-norm + residual (+ optional post_scale):
/// residual_inout[t] = (residual_inout[t] + rms(x[t]) * gamma) * post_scale.
void RmsNormAdd(const float* x, const float* gamma, float* residual_inout,
                std::uint32_t n_rows, std::uint32_t dim, float eps,
                float post_scale, hipStream_t stream);

/// Copy a host buffer into device memory (plain hipMalloc + full copy).
/// Returns nullptr on failure.
void* UploadMirror(const void* host, std::size_t bytes);

/// Embedding lookup from a device-mirrored table (Q4_0 / Q8_0 / F32 / F16).
void DebugRow(const void* table, std::uint32_t row_bytes,
              std::uint32_t token, float* d_out, std::uint32_t n,
              hipStream_t stream);

void Embed(const void* table, core::GgmlType type,
           const std::int32_t* tokens, float* res, std::uint32_t n_tokens,
           std::uint32_t hidden, hipStream_t stream);

/// Upload token ids to the persistent device staging buffer used by Embed.
/// Call before any kernel that needs the ids on-device (e.g. the fused
/// embed+scale path).
void EmbedScale(const void* table, core::GgmlType type,
                const std::int32_t* tokens, float* res, std::uint32_t n_tokens,
                std::uint32_t hidden, float scale, hipStream_t stream);

/// Device-token variant of EmbedScale (no H2D copy; reads `d_tokens` directly).
void EmbedScaleDevice(const void* table, core::GgmlType type,
                      const std::int32_t* d_tokens, float* res,
                      std::uint32_t n_tokens, std::uint32_t hidden, float scale,
                      hipStream_t stream);

void UploadTokens(const std::int32_t* tokens, std::uint32_t n_tokens,
                  hipStream_t stream);
std::int32_t* DeviceTokens();

/// Copy a host buffer into an existing device buffer via the copy engine.
bool CopyHostToDevice(void* dst, const void* host, std::size_t bytes);

/// Copy a device buffer back to its host mirror (for host-mapped tensors).
void DownloadMirror(void* host, const void* device, std::size_t bytes);

/// Convert an F32 row-major matrix to F16.
void F32ToF16(const float* in, __half* out, std::size_t count,
              hipStream_t stream);

/// y = x @ W^T over an F16 weight matrix (m rows of k) and F16 activations.
/// y is F32 [n_tokens][m].
hipblasStatus_t F16GemmStatus(const __half* w, const __half* x, float* y,
             std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
             hipStream_t stream);

/// RMSNorm with an extra elementwise gamma: out = rms(x) * gamma * mul.
void RmsNormMul(const float* x, const float* gamma, const float* mul,
                float* out, std::uint32_t n_rows, std::uint32_t dim, float eps,
                hipStream_t stream);

/// RoPE over [n_tokens][heads][d] with per-layer theta and rotary width.
void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          std::uint32_t start_pos, float theta, hipStream_t stream);

/// Causal attention over an F16 KV cache with an optional sliding window.
/// q: [n_tokens][heads][d]; k/v caches: [capacity][kv_heads][d] at
/// positions start_pos + t. When window > 0, keys older than
/// (start_pos + t - window + 1) are masked out for query t.
namespace wmma {
// WMMA flash prefill attention (sliding-window geometry only). Returns false
// if the shape is unsupported; the caller then uses the scalar kernel.
bool AttentionWmma(const float* q, const __half* k_cache, const __half* v_cache,
                   float* out, std::uint32_t n_tokens, std::uint32_t start_pos,
                   std::uint32_t d, std::uint32_t kv_stride,
                   std::uint32_t window, hipStream_t stream);
}  // namespace wmma

namespace wmma512 {
// WMMA flash prefill attention for the full (d=512) layers. Same kernel with
// head_dim 512 geometry (16 query heads / 2 KV heads). window is 0 (dense
// causal) for these layers.
bool AttentionWmma512(const float* q, const __half* k_cache,
                      const __half* v_cache, float* out,
                      std::uint32_t n_tokens, std::uint32_t start_pos,
                      std::uint32_t d, std::uint32_t kv_stride,
                      std::uint32_t window, hipStream_t stream);
}  // namespace wmma512

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               float* out, std::uint32_t n_tokens,
               std::uint32_t start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t d,
               std::uint32_t kv_stride, std::uint32_t window,
               hipStream_t stream);

/// Blocked-Q8 direct-read attention for SWA (d == 256) layers: the shared
/// blocked KV caches are read in place (fused dequant, score scale 1.0 to
/// match the gemma4 F16 kernels), no F16 staging. The engine applies the
/// sliding-window pointer shift before calling. Full (d == 512) layers stay
/// on the F16 Attention path. `start_pos` is a device scalar.
void AttentionBlocked(const float* q, const std::uint8_t* k_cache,
                      const std::uint8_t* v_cache, float* out,
                      std::uint32_t n_tokens, const std::uint32_t* start_pos,
                      std::uint32_t heads, std::uint32_t kv_heads,
                      std::uint32_t d, hipStream_t stream,
                      std::uint32_t key_block_bytes = 34,
                      std::uint32_t value_block_bytes = 34);

/// Blocked KV store: symmetric per-32-block quantize of the final
/// (post-norm, post-RoPE) K/V rows into the shared blocked layout.
/// `key_block_bytes` / `value_block_bytes` select each side's grid (34 =
/// 8-bit codes with range 127, 18 = packed 4-bit nibbles with range 7 for
/// the GUFO_KV_Q4/V4 grids). d must be 256 (SWA layers); other layers stay
/// on StoreKv. `start_pos` is a device scalar.
void StoreBlockedKv(const float* k_rows, const float* v_rows,
                    std::uint8_t* k_cache, std::uint8_t* v_cache,
                    std::uint32_t n_tokens, std::uint32_t kv_heads,
                    std::uint32_t d, const std::uint32_t* start_pos,
                    hipStream_t stream, std::uint32_t key_block_bytes = 34,
                    std::uint32_t value_block_bytes = 34);

/// Cross/decode attention with explicit query position base (`query_pos + t`)
/// and explicit valid KV cache length (`kv_len`). Uses Split-K Flash-Decoding
/// GQA across all CUs for narrow batches (decode, MTP verify, MTP draft).
void AttentionCross(const float* q, const __half* k_cache,
                    const __half* v_cache, float* out, std::uint32_t n_tokens,
                    std::uint32_t query_pos, std::uint32_t kv_len,
                    std::uint32_t heads, std::uint32_t kv_heads,
                    std::uint32_t d, std::uint32_t kv_stride,
                    std::uint32_t window, hipStream_t stream);

/// Stores f32 rows into the f16 cache at positions start_pos + t.
void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, std::uint32_t start_pos,
             hipStream_t stream);

/// MoE router for Gemma 4: tmp = rmsnorm(attn_out, eps) * (1/sqrt(hidden))
/// * router_scale, then logits = tmp @ router^T via the caller-supplied
/// GEMM. This kernel produces `tmp` (the normalized, scaled router input).
void RouterInput(const float* attn_out, const float* router_scale,
                 float* tmp, std::uint32_t n_tokens, std::uint32_t hidden,
                 float eps, hipStream_t stream);

/// out[t][i] = sum_s weights[t][s] * expert_out[t*k + s][i] (no shared
/// expert gate — Gemma 4 combines the shared MLP separately).
/// Gather + convert to F16 in one pass for the compact WMMA GEMM.
void CompactGatherF16(const float* src, __half* dst, const std::int32_t* rows,
                      std::uint32_t begin, std::uint32_t end,
                      std::uint32_t dim, hipStream_t stream);
/// Gather rows routed to one expert into a compact batch (dst==src in-place,
/// padding rows -1 zero-filled).
void CompactGather(const float* src, float* dst, const std::int32_t* rows,
                   std::uint32_t begin, std::uint32_t end, std::uint32_t dim,
                   hipStream_t stream);
/// Scatter variant: out[rows_token[i]] += w(slot)*down[i] over compact rows.
void MoeScatterAdd(const float* down, const std::int32_t* rows_token,
                   const std::int32_t* rows_slot, const float* wts,
                   std::uint32_t begin, std::uint32_t end, float* out,
                   std::uint32_t dim, hipStream_t stream);
/// Weighted routed accumulate for one expert: out[t] += w(t,e)*expert_down[t]
/// with w(t,e) summed from the routing slots. One launch per expert.
/// Decode-time fused MoE gate/up over Q4_0 experts with DEVICE-side
/// routing: one launch computes gate/up for all `slots` (= n*used)
/// (token,slot) rows; ids select the expert (negative = skip). gate_up is the
/// fused [E][2*ff][hidden] slab; x is [n][hidden]; gout/uout are [slots][ff].
void MoeVecGateUp(const void* gate_up_dev, std::uint64_t row_bytes,
                  std::uint64_t expert_stride_bytes, const float* x,
                  const std::int32_t* ids, float* gout, float* uout,
                  std::uint32_t slots, std::uint32_t used, std::uint32_t ff,
                  std::uint32_t hidden, hipStream_t stream);

/// Wave32 fused MoE gate+up+GeluTanhSwiglu: writes act_out[slots][ff] directly.
void MoeVecGateUpSwiglu(const void* gate_up_dev, std::uint64_t row_bytes,
                        std::uint64_t expert_stride_bytes, const float* x,
                        const std::int32_t* ids, float* act_out,
                        std::uint32_t slots, std::uint32_t used,
                        std::uint32_t ff, std::uint32_t hidden,
                        hipStream_t stream);

/// Fused Q4_0 down GEMV for the decode MoE: grid (slots, hidden); block
/// (slot, h) = dot(dequant(down[e][h][:]), act[slot][:]) with e = ids[slot]
/// read on the device (negative -> writes 0). down slab [E][hidden][ff].
void GemvQSlot(const void* down_dev, std::uint64_t row_bytes,
               std::uint64_t expert_stride_bytes, const float* act,
               const std::int32_t* ids, float* out, std::uint32_t slots,
               std::uint32_t hidden, std::uint32_t ff, hipStream_t stream);

/// Decode routed-MoE epilogue: out[t] += sum_s wts[t*used+s] * dscale[ids[..]]
/// * slot_down[t*used+s]. Device-side routing; replaces the per-expert
/// ScaleInplace + MoeRoutedAdd pair (and the host ids/wts readback).
void MoeSlotEpilogue(const float* slot_down, const std::int32_t* ids,
                     const float* wts, const float* dscale, float* out,
                     std::uint32_t n_tokens, std::uint32_t used,
                     std::uint32_t dim, hipStream_t stream);

/// Overwriting twin of MoeSlotEpilogue (`out[t] = sum_s ...`), removing the
/// separate hipMemsetAsync(out, 0) while guaranteeing zero stale residue.
void MoeSlotEpilogueOverwrite(const float* slot_down, const std::int32_t* ids,
                              const float* wts, const float* dscale, float* out,
                              std::uint32_t n_tokens, std::uint32_t used,
                              std::uint32_t dim, hipStream_t stream);

void MoeRoutedAdd(const float* expert_down, const std::int32_t* ids,
                  const float* wts, std::uint32_t expert, float* out,
                  std::uint32_t n_tokens, std::uint32_t used,
                  std::uint32_t dim, hipStream_t stream);

void MoeEpilogue(const float* expert_out, const float* weights, float* out,
                 std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                 hipStream_t stream);

/// Per-layer output scale: out[t][i] *= scale (single scalar).
void LayerScale(float* out, float scale, std::size_t count,
                hipStream_t stream);

/// Elementwise: out[i] *= mul[i % mul_count].
void MulInplace(float* out, const float* mul, std::size_t count,
                std::size_t mul_count, hipStream_t stream);

/// RoPE with per-pair frequency factors (full-attention layers): the angle
/// for pair i is position * theta^(-2i/d) / freq_factors[i] (ggml yarn:
/// theta/ff, so a 1e30 factor freezes the pair).
void RopeFrequencies(float* x, std::uint32_t n_tokens, std::uint32_t heads,
                     std::uint32_t d, std::uint32_t rotary_dim,
                     const std::uint32_t* start_pos, float theta,
                     const float* freq_factors, hipStream_t stream);

/// Per-token-position rope: each token t uses position start_pos + t.
void RopeFrequenciesMulti(float* x, std::uint32_t n_tokens,
                          std::uint32_t heads, std::uint32_t d,
                          std::uint32_t rotary_dim, std::uint32_t start_pos,
                          float theta, const float* freq_factors,
                          hipStream_t stream);

/// Lossless Q4_0 -> Q8_0 requant of `rows` rows into a Q8_0 buffer
/// (34 bytes per 32 elements). For the WMMA DenseF16Gemm prefill path.
void DequantSlabQ8(const TensorRef& t, std::uint64_t rows, std::uint8_t* dst,
                   hipStream_t stream);

/// Q4_0 -> F16 dequant of `rows` rows into row-major [cols] halves, matching
/// the UnquantizedF16Gemm weight contract (row stride = cols halves).
void DequantSlabF16(const TensorRef& t, std::uint64_t rows, __half* dst,
                    hipStream_t stream);

/// Dequantize `rows` rows of `cols` elements from a GGML quant tensor into
/// the F16 staging slab.
void DequantSlab(const TensorRef& t, std::uint64_t rows, __half* dst,
                 hipStream_t stream);

/// Fused n == 1 decode GEMV over quant weights (no staging slabs).
void GemvFused(const void* wdev, int type, std::uint64_t row_bytes,
               std::uint64_t cols, const float* x, float* y, std::uint32_t m,
               hipStream_t stream);

/// Fused n == 1 decode GEMV over dense F16 weights.
void GemvF16Fused(const __half* wdev, std::uint64_t cols, const float* x,
                  float* y, std::uint32_t m, hipStream_t stream);

/// Wave32 multi-row GEMV/narrow-GEMM for n_tokens in [1, 8], reading the
/// quantized/F16 weight matrix once across all tokens. Optional softcap > 0
/// applies tanh(v / softcap) * softcap in registers before writing y.
void GemvBatched(const void* wdev, int type, std::uint64_t row_bytes,
                 std::uint64_t cols, const float* x, float* y,
                 std::uint32_t n_tokens, std::uint32_t m, float softcap,
                 hipStream_t stream);

/// Wave32 fused gate + up + GeluTanhSwiglu for n_tokens in [1, 8].
void GemvGateUpSwiglu(const void* w_gate, const void* w_up, int type,
                      std::uint64_t row_bytes, std::uint64_t cols,
                      const float* x, float* out_act, std::uint32_t n_tokens,
                      std::uint32_t m, hipStream_t stream);

/// Wave32 fused Q + K (+ optional V) narrow-batch GEMV for n_tokens in [1, 8].
void GemvQkv(const void* w_q, int type_q, std::uint64_t rb_q,
             std::uint32_t qdim, const void* w_k, int type_k,
             std::uint64_t rb_k, const void* w_v, int type_v,
             std::uint64_t rb_v, std::uint32_t kvdim, std::uint64_t cols,
             const float* x, float* q_out, float* k_out, float* v_out,
             std::uint32_t n_tokens, hipStream_t stream);

/// Fused Q/K/V per-head RMSNorm + RoPE/RopeFrequencies + F16 KV cache store.
void PrepAttnQkv(float* qbuf, const float* kbuf, const float* vbuf,
                 const float* q_norm, const float* k_norm, const float* v_norm,
                 __half* kv_k, __half* kv_v, std::uint32_t n_tokens,
                 std::uint32_t num_heads, std::uint32_t kv_heads,
                 bool has_kv, bool has_v_proj, std::uint32_t head_dim,
                 std::uint32_t rotary_dim, std::uint32_t start_pos,
                 float theta, const float* freq_factors, float eps,
                 hipStream_t stream);

/// Fused MoE pre-FFN triple norm: computes rms(x) once and writes
/// out_ffn = rms(x) * gamma_ffn, out_pre2 = rms(x) * gamma_pre2, and
/// out_router = rms(x) * (1/sqrt(dim)) * router_scale.
void FfnTripleNorm(const float* x, const float* gamma_ffn,
                   const float* gamma_pre2, const float* router_scale,
                   float* out_ffn, float* out_pre2, float* out_router,
                   std::uint32_t n_rows, std::uint32_t dim, float eps,
                   hipStream_t stream);

/// Fused MoE combine + post norms + residual + layer output scale:
/// routed = sum_s wts * dscale * slot_down
/// combined = rms(shared_down) * gamma_post1 + rms(routed) * gamma_post2
/// residual_inout = (residual_inout + rms(combined) * gamma_post) * post_scale
void MoeCombinePostNormResidual(
    const float* slot_down, const std::int32_t* ids, const float* wts,
    const float* dscale, const float* shared_down, const float* gamma_post1,
    const float* gamma_post2, const float* gamma_post, float* residual_inout,
    std::uint32_t n_tokens, std::uint32_t used, std::uint32_t dim, float eps,
    float post_scale, hipStream_t stream);

/// Device-side builder for RoutedF16Gemm's `(expert | (tile << 16))` list from
/// `routed_bounds[0..num_experts]`. Writes `[n_tiles, q4_rows]` to `d_meta`.
void BuildRoutedTiles(const std::int32_t* routed_bounds,
                      std::int32_t* q4_tiles, std::uint32_t* d_meta,
                      std::uint32_t num_experts, std::uint32_t tile_rows,
                      hipStream_t stream);

}  // namespace kernels
}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_KERNELS_KERNELS_HPP_
