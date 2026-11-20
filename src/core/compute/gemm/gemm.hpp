#ifndef MBUFO_CORE_COMPUTE_GEMM_GEMM_HPP_
#define MBUFO_CORE_COMPUTE_GEMM_GEMM_HPP_

// Unified decode GEMV / fused-MoE launchers (Phase B, mechanical merge).
//
// Contents (verbatim from the model backends, namespace relocated):
//  * gemv_wave32_sdot4.hip: gemma4 Wave32 sdot4 GEMV (Q8_0/Q4_0/F32/F16/BF16,
//    fused gate/up, QKV) as the base; 27B gemv_quant.hip shapes
//    (Q8_0/Q8_K/Q4_K/Q5_K/IQ4_XS block GEMV); flash-next decode SmallGemm
//    (F32/F16/BF16 narrow projections, in fn_compat).
//  * moe_fused.hip: gemma4 fused MoE (grouped/vec gate+up+GeluTanh, Q4,
//    slot epilogues); 27B fused SwiGLU GEMV; flash-next MoE epilogue
//    (fn_compat, HC geometry).
// Only equivalent merging here: no algorithm changes.
//
// GGML type codes used by the int-typed launchers below.
#define MBUFO_GEMM_F32 0
#define MBUFO_GEMM_F16 1
#define MBUFO_GEMM_Q4_0 2
#define MBUFO_GEMM_Q8_0 8
#define MBUFO_GEMM_Q4_K 12
#define MBUFO_GEMM_Q5_K 13
#define MBUFO_GEMM_Q6_K 14
#define MBUFO_GEMM_BF16 30

#include <cstddef>
#include <cstdint>
#include <limits>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/quant_status.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/compute/gemm/gemm_dispatcher.hpp"

namespace mbufo::core::compute::gemm {

constexpr std::uint32_t kThreads = 256;

[[nodiscard]] inline std::uint32_t Blocks(std::size_t n) {
  const std::size_t b = (n + kThreads - 1) / kThreads;
  return static_cast<std::uint32_t>(b);
}

// Shared Q8_1 activation block (gemma4 BlockQ8_1 layout): 8 int32 lanes plus
// the block scale and pre-scaled row sum used by the sdot4 kernels.
struct alignas(8) BlockQ8_1 {
  float d;
  float s;
  std::int32_t qs[8];
};

namespace detail {
// Device-side Q8_1 scratch for the Wave32 sdot4 launchers (single instance,
// defined in gemv_wave32_sdot4.hip). Prompt-time use is single-stream.
BlockQ8_1* GemvQ8Scratch();
std::size_t GemvQ8ScratchCap();
BlockQ8_1* QuantizeToQ8Scratch(const float* x, std::size_t n_blocks,
                               hipStream_t stream);
}  // namespace detail

// ---- Wave32 sdot4 GEMV (gemma4 shapes) ----
void GemvBatched(const void* wdev, int type, std::uint64_t row_bytes,
                 std::uint64_t cols, const float* x, float* y,
                 std::uint32_t n_tokens, std::uint32_t m, float softcap,
                 hipStream_t stream);
void GemvGateUpSwiglu(const void* w_gate, const void* w_up, int type,
                      std::uint64_t row_bytes, std::uint64_t cols,
                      const float* x, float* out_act, std::uint32_t n_tokens,
                      std::uint32_t m, hipStream_t stream);
void GemvFused(const void* wdev, int type, std::uint64_t row_bytes,
               std::uint64_t cols, const float* x, float* y, std::uint32_t m,
               hipStream_t stream);
void GemvF16Fused(const __half* wdev, std::uint64_t cols, const float* x,
                  float* y, std::uint32_t m, hipStream_t stream);
void GemvQkv(const void* w_q, int type_q, std::uint64_t rb_q,
             std::uint32_t qdim, const void* w_k, int type_k,
             std::uint64_t rb_k, const void* w_v, int type_v,
             std::uint64_t rb_v, std::uint32_t kvdim, std::uint64_t cols,
             const float* x, float* q_out, float* k_out, float* v_out,
             std::uint32_t n_tokens, hipStream_t stream);
void SoftCap(float* x, float cap, std::size_t count, hipStream_t stream);

/// Single-token decode GEMV for K-quant weights (Q4_K, Q5_K), F32 activations: Y[r] = sum_k W[r][k] * X[k]. Rows share one
/// load of each activation slice (the generic block GEMV reloads it per row). Returns false for other formats / shapes
/// (k % 256 == 0, 16-byte aligned pointers).
bool QuantGemvSingle(const void* w, int type, const float* x, float* y, std::size_t m, std::size_t k, hipStream_t stream);
/// Multi-token decode / speculative-verify GEMM for K-quant weights (Q4_K, Q5_K): Y[t][r] = sum_k W[r][k] * X[t][k] with
/// X F32 [n_tokens][k] (16-byte aligned), Y F32 [n_tokens][m]. Every weight byte is read once for all tokens (WMMA with
/// in-register dequantization). Returns false (nothing launched) for other formats or shapes: n_tokens 1..32,
/// m % 16 == 0, k % 256 == 0, 16-byte aligned pointers.
inline constexpr std::size_t kQuantGemvMultiMaxTokens = 32;
/// GemvDecode serves up to this many tokens for K-quant weights by splitting them evenly into groups of at most
/// kQuantGemvMultiMaxTokens (each group re-reads the weights once, ~100 ms per group on a 30B model): far cheaper than
/// the dequantize-everything route (>1 s) for the short prompts / follow-up turns that dominate chat latency.
inline constexpr std::size_t kQuantGemvGroupedMaxTokens = 192;
bool QuantGemvMulti(const void* w, int type, const float* x, float* y, std::size_t n_tokens, std::size_t m,
                    std::size_t k, hipStream_t stream);

// ---- 27B block GEMV shapes (from gemv_quant.hip) ----
void LaunchQ8KBlockGEMV(const void* A, core::GgmlType type, const float* x,
                        float* y, std::size_t M, std::size_t K,
                        hipStream_t stream);

// ---- narrow Q8_0 decode GEMV (from flash-next MMQ, model-agnostic) ----
// Contract: Q8_0 weights (34B blocks) x prequantized Q8_1 activations (36B
// blocks), ungated, 1..8 tokens, K % 32 == 0, stride_blocks >= K / 32,
// excluding N == 1 && M <= 64 && K % 5120 == 0 (model-local accumulation).
// Returns false for out-of-contract shapes; the caller keeps its legacy path.
bool LaunchNarrowQ80Decode(const void* weights, const void* input,
                           float* output, std::size_t k, std::size_t rows,
                           std::size_t tokens, std::size_t input_stride_blocks,
                           hipStream_t stream);

// ---- shared Q8_1 prequantize (from flash-next MMQ, model-agnostic) ----
// Row-major F32 [n, k] -> canonical block_q8_1 rows with pad(k,512) width.
// Byte size matches qfn_mmq_q8_1_bytes(n, k). Returns 0 on success.
std::size_t PreqQ81RowPad(std::size_t k);
std::size_t PreqQ81Bytes(std::size_t n, std::size_t k);
int PrequantizeQ8_1(const float* x_f32, void* x_q8, std::size_t n,
                    std::size_t k, hipStream_t stream);
// Scratch + geometry helpers for the prequant -> narrow-GEMV path.
// §4.2 Slot-concurrency iron rule: no `static` buffers inside operators.
// The prequant scratch is caller-owned (executor/session arena, sized by
// NarrowQ80PrequantScratchBytes for the 8-token decode ceiling); the arena
// is allocated outside graph capture, so capture only uses it, never grows
// it. Null or short scratch refuses cleanly and the caller keeps its legacy
// path.
std::size_t NarrowQ80PrequantScratchBytes(std::size_t max_k);
std::size_t NarrowQ80PrequantBytes(std::size_t tokens, std::size_t k);
std::size_t NarrowQ80PrequantStrideBlocks(std::size_t k);
// Shared quantize -> narrow-GEMV for F32 activations (27B / gemma4 decode).
// `scratch` is caller-owned device memory of at least
// NarrowQ80PrequantBytes(tokens, k) bytes (§4.2). Returns false for
// out-of-contract shapes or short scratch; the caller keeps its legacy path.
bool LaunchNarrowQ80Prequant(const void* weights, const float* x_f32,
                             float* output, std::size_t k, std::size_t rows,
                             std::size_t tokens, void* scratch,
                             std::size_t scratch_bytes, hipStream_t stream);
// ---- Fused gated narrow decode (shared-expert gate/up + SwiGLU) ----
// Single-token decode only: silu(gate*x)*(up*x) with one prequant pass.
// Replaces the gate/up Prequant pair plus the SwiGLU kernel; the has_gate
// core evaluates the same dots in the same order. Returns false out of
// contract (caller keeps the trio).
bool LaunchNarrowQ80GatedPrequant(const void* up_w, const void* gate_w,
                                 const float* x_f32, float* output,
                                 std::size_t k, std::size_t rows, void* scratch,
                                 std::size_t scratch_bytes,
                                 hipStream_t stream);
// ---- Fused QKV narrow decode (Q8_0 x prequantized-Q8_1, ungated) ----
// Same per-row arithmetic as the narrow decode kernel; the three projections
// share one activation row, so one launch covers qrows + kvrows + kvrows
// rows. Returns false out of contract (caller keeps three narrow launches).
bool LaunchNarrowQ80QkvDecode(const void* wq, const void* wk, const void* wv,
                             const void* input, float* out_q, float* out_k,
                             float* out_v, std::size_t k, std::size_t qrows,
                             std::size_t kvrows, std::size_t tokens,
                             std::size_t input_stride_blocks,
                             hipStream_t stream);
// ---- Narrow decode with fused row-bias + residual add (Q8_0, ungated) ----
// Bitwise-identical epilogue to AddRowBias + ResidualAdd after the plain
// narrow kernel (inout[t*M+row] += dot + bias[row]). One launch replaces
// three. Returns false out of contract (caller keeps its sequence).
bool LaunchNarrowQ80PrequantRowAdd(const void* weights, const float* x_f32,
                                   const float* bias, float* inout,
                                   std::size_t k, std::size_t rows,
                                   std::size_t tokens, void* scratch,
                                   std::size_t scratch_bytes,
                                   hipStream_t stream);

// ---- Narrow decode for native Q5_0 / Q4_K weights (prequantized Q8_1 input) ----
// Single token. `x_q8_1` is PrequantizeQ8_1 output for one row (k a multiple of
// 32 for Q5_0, 256 for Q4_K). Q5_0 weights are a planar repack: for each row r and
// block b (index r * (k/32) + b) a qs plane (16 B/block), then a qh plane (4 B),
// then a d plane (2 B); NarrowQ50PlanarBytes is the total. y[row] = dot + bias[row]
// (bias may be null). The Q4_K variant reads native 144 B blocks in place and
// applies the attention-output epilogue inout[row] += dot + bias[row].
std::size_t NarrowQ50PlanarBytes(std::size_t rows, std::size_t cols);
bool LaunchNarrowQ50Planar(const void* planar_w, const void* x_q8_1, float* y,
                           const float* bias, std::size_t k, std::size_t rows,
                           hipStream_t stream);
bool LaunchNarrowQ4KRowAdd(const void* w, const void* x_q8_1, const float* bias,
                           float* inout, std::size_t k, std::size_t rows,
                           hipStream_t stream);

// ---- flash-next decode SmallGemm (F32/F16/BF16 narrow projections) ----
namespace fn_compat {
// type: 0 = F32, 1 = F16, 30 = BF16.
void SmallGemm(const void* w, int type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream);
// MoE epilogue with the shared-expert/gate terms (HC geometry).
void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream);
}  // namespace fn_compat

// ---- fused MoE (gemma4 shapes + 27B dense fusion) ----
void MoeVecGateUp(const void* gate_up_dev, std::uint64_t row_bytes,
                  std::uint64_t expert_stride_bytes, const float* x,
                  const std::int32_t* ids, float* gout, float* uout,
                  std::uint32_t slots, std::uint32_t used, std::uint32_t ff,
                  std::uint32_t hidden, hipStream_t stream);
void MoeVecGateUpSwiglu(const void* gate_up_dev, std::uint64_t row_bytes,
                        std::uint64_t expert_stride_bytes, const float* x,
                        const std::int32_t* ids, float* act_out,
                        std::uint32_t slots, std::uint32_t used,
                        std::uint32_t ff, std::uint32_t hidden,
                        hipStream_t stream);
// Native MXFP4 vec MoE (OCP microscaling, GGML type 39, 17 bytes/block).
// Same geometry as the Q4_0 vec path; weights stay MXFP4 (no conversion).
void MoeVecGateUpMxfp4(const void* gate_up_dev, std::uint64_t row_bytes,
                       std::uint64_t expert_stride_bytes, const float* x,
                       const std::int32_t* ids, float* gout, float* uout,
                       std::uint32_t slots, std::uint32_t used,
                       std::uint32_t ff, std::uint32_t hidden,
                       hipStream_t stream);
void GemvQSlotMxfp4(const void* down_dev, std::uint64_t row_bytes,
                    std::uint64_t expert_stride_bytes, const float* act,
                    const std::int32_t* ids, float* out, std::uint32_t slots,
                    std::uint32_t hidden, std::uint32_t ff,
                    hipStream_t stream);
// ---- Fused MoE decode (native MXFP4, small-slot specialization) ----
// Gate+up GEMV + SwiGLU-OAI biases/activation in one launch; down GEMV +
// down bias + router-weighted overwrite epilogue in a second launch.
// Model-agnostic (MXFP4 slabs, SwiGLU-OAI scalar math). Return false out of
// contract; the caller keeps its routed/vec path.
bool MoeDecodeGateUpActMxfp4(const void* gate_up_dev, std::uint64_t row_bytes,
                             std::uint64_t expert_stride_bytes, const float* x,
                             const std::int32_t* ids, const float* gate_b,
                             const float* up_b, float* act, std::uint32_t slots,
                             std::uint32_t used, std::uint32_t ff,
                             std::uint32_t hidden, hipStream_t stream);
bool MoeDecodeDownEpiMxfp4(const void* down_dev, std::uint64_t row_bytes,
                           std::uint64_t expert_stride_bytes, const float* act,
                           const std::int32_t* ids, const float* wts,
                           const float* down_b, float* out,
                           std::uint32_t n_tokens, std::uint32_t used,
                           std::uint32_t hidden, std::uint32_t ff,
                           hipStream_t stream);
void GemvQSlot(const void* down_dev, std::uint64_t row_bytes,
               std::uint64_t expert_stride_bytes, const float* act,
               const std::int32_t* ids, float* out, std::uint32_t slots,
               std::uint32_t hidden, std::uint32_t ff, hipStream_t stream);
void MoeSlotEpilogue(const float* slot_down, const std::int32_t* ids,
                     const float* wts, const float* dscale, float* out,
                     std::uint32_t n_tokens, std::uint32_t used,
                     std::uint32_t dim, hipStream_t stream);
void MoeSlotEpilogueOverwrite(const float* slot_down, const std::int32_t* ids,
                              const float* wts, const float* dscale, float* out,
                              std::uint32_t n_tokens, std::uint32_t used,
                              std::uint32_t dim, hipStream_t stream);
void MoeRoutedAdd(const float* expert_down, const std::int32_t* ids,
                  const float* wts, std::uint32_t expert, float* out,
                  std::uint32_t n_tokens, std::uint32_t used, std::uint32_t dim,
                  hipStream_t stream);
void MoeEpilogue(const float* expert_out, const float* weights, float* out,
                 std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                 hipStream_t stream);
void LaunchFusedSwiGLUGEMV(const void* gate_w, core::GgmlType gate_type,
                           const void* up_w, core::GgmlType up_type,
                           const float* x, float* out,
                           std::size_t intermediate_size,
                           std::size_t hidden_size, hipStream_t stream);

// ---- W8A8 varlen (Phase B-3, §3.2) ----
// Fixed-shape Wave64 launch (verbatim geometry of flash-next W8A8GemmWave64),
// its warmup probe, and the host-side 16-split dispatcher: main = batch &
// ~15 via the original kernel binary, tail (if any) via a scalar kernel.
// Returns false for uncovered shapes; callers keep the legacy route.
void W8A8GemmWave64Core(const void* w, const void* x_tiled, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream);
hipError_t PrepareW8A8GemmWave64Core();
bool W8A8GemmVarlen(const void* w, const void* x_tiled, float* out,
                    std::size_t batch, std::size_t m, std::size_t k,
                    hipStream_t stream);

// Phase C-1 (GGUF v2 section 4): shared quantized-GEMM launchers relocated
// verbatim from models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp.
// gemma4 calls these; flash-next keeps its own copies until Phase C-2.

/// GGML type ids accepted by the relocated launchers below.
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ4_0 = 2,
  kQ4_K = 12,
  kQ5_K = 13,
  kBF16 = 30,
  kQ4_0_ROCMFP4_FAST = 101,
  kMXFP4 = 102,
};

inline constexpr std::uint32_t kArgmaxParts = 64;
struct ArgmaxCandidate {
  float value;
  std::int32_t index;
};

/// W8A8 tiled-activation bytes for `batch` rows over `k` columns.
std::size_t Q8TiledBytes(std::size_t batch, std::size_t k);
/// Quantize F32 rows into the W8A8 tiled fragment layout.
void QuantizeQ8Tiled(const float* x, void* out, std::size_t batch,
                     std::size_t k, hipStream_t stream);
/// Legacy W8A8 dispatcher (general shapes; the varlen fast path above
/// handles the aligned (2560, 6144) case via W8A8GemmVarlen).
bool W8A8Gemm(const void* w, const void* x_tiled, float* out,
              std::size_t batch, std::size_t m, std::size_t k,
              hipStream_t stream);
/// Bulk F32 -> F16 narrowing (round-to-nearest-even) for routed-expert and
/// shared-expert F16 GEMM inputs. Model-agnostic; returns false on nulls.
bool ConvertF32ToF16(const float* in, __half* out, std::size_t n,
                     hipStream_t stream);
/// F16 activation rows, F16/unquantized weights, F32 accumulation.
bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream);
/// F16 activations, Q8_0/Q4 weights dequantized in LDS, F32 accumulation.
bool DenseF16Gemm(const void* w, const __half* x, float* out,
                  std::size_t batch, std::size_t m, std::size_t k,
                  hipStream_t stream, std::size_t input_stride = 0);

/// Type-dispatched single-token decode GEMV over core::quant::AllFormats (see quant_traits.hpp):
/// y[t][r] = dot(row r of W, x[t]) for 1..8 tokens. `x` is F32 [n][k], y is [n][m]. Each format
/// declares how it is served (narrow Q8 prequant, K-quant block GEMV, wave32 GEMV). Returns
/// kUnsupportedType for a code that is not in the list and for formats without a decode kernel
/// (Q6_K): then dequantize with QuantGemmPrefill, which works for any batch. `preq_scratch` is the
/// narrow-Q8 prequant arena (may be null: Q8_0 then uses the plain GEMV).
[[nodiscard]] quant::Status GemvDecode(const void* w, int type, const float* x, float* y, std::size_t n_tokens,
                                       std::size_t m, std::size_t k, void* preq_scratch, std::size_t preq_bytes,
                                       hipStream_t stream);
/// Type-dispatched batched GEMM with F16 activations: y[b][r] = dot(row r of W, x16[b]). Q8_0 rides the
/// dense F16 GEMM, F16 the unquantized GEMM, everything else is dequantized into `wdq_scratch`
/// (m * k * 2 bytes) first. kUnsupportedType for a code that is not in the list, kScratchTooSmall when
/// the scratch is short.
[[nodiscard]] quant::Status QuantGemmPrefill(const void* w, int type, const __half* x16, float* y,
                                             std::size_t batch, std::size_t m, std::size_t k,
                                             __half* wdq_scratch, std::size_t wdq_bytes, hipStream_t stream);

/// Bytes of one packed GGML row of `cols` elements (type = MBUFO_GEMM_* code:
/// F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K); 0 when the type is unsupported or
/// `cols` is not a whole number of blocks.
std::size_t DequantRowBytes(int type, std::size_t cols);
/// Embedding gather: out[t, :] = dequantized row ids[t] of `table` (F32).
/// Shared by every model so an embedding table in any supported format works.
/// Returns false (nothing launched) for unsupported type/shape.
bool EmbedGather(const void* table, int type, const std::int32_t* ids,
                 float* out, std::size_t n_tokens, std::size_t hidden,
                 hipStream_t stream);
/// EmbedGather with every element multiplied by `scale` (embedding scaling).
bool EmbedGatherScaled(const void* table, int type, const std::int32_t* ids, float* out,
                       std::size_t n_tokens, std::size_t hidden, float scale, hipStream_t stream);
/// EmbedGather writing every row `streams` times: out[(t * streams + s) * hidden + i] (the multi-stream
/// residual layout of flash-next).
bool EmbedGatherStreams(const void* table, int type, const std::int32_t* ids, float* out, std::size_t n_tokens,
                        std::size_t hidden, std::uint32_t streams, hipStream_t stream);
/// Single token passed by value (no device id buffer): out[0..hidden) = row `token`.
bool EmbedGatherOne(const void* table, int type, std::uint32_t token, float* out, std::size_t hidden,
                    hipStream_t stream);
/// Dequantize `rows` packed rows of `cols` elements into dense row-major F16
/// (prefill fallback for packed weights without a WMMA/MMQ kernel).
bool DequantRowsF16(const void* w, int type, __half* out, std::size_t rows,
                    std::size_t cols, hipStream_t stream);
/// Tiled F16 weight layout used by TiledF16Gemm: rows are grouped in tiles of kTiledRows and, inside a tile, stored
/// K-block major: element (r, c) lives at ((r / 128 * (cols / 32) + c / 32) * 128 + r % 128) * 32 + c % 32.
/// Requires rows % 128 == 0 and cols % 32 == 0.
inline constexpr std::size_t kTiledRows = 128;
/// DequantRowsF16 writing the tiled layout (the prefill GEMM's weight scratch; nothing else reads it).
bool DequantRowsF16Tiled(const void* w, int type, __half* out, std::size_t rows, std::size_t cols,
                         hipStream_t stream);
/// Y[t][r] = sum_k X[t][k] * W[r][k], W in the tiled layout above, X F16 row-major [batch][k], Y F32 [batch][m].
/// WMMA 16x16x16, single K-ordered F32 accumulation chain. Returns false (nothing launched) on unsupported shapes.
bool TiledF16Gemm(const __half* w_tiled, const __half* x, float* y, std::size_t batch, std::size_t m, std::size_t k,
                  hipStream_t stream);
/// Routed-expert compaction sizing and launchers (16-row padded buckets).
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream);
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream);
/// Softmax + top-k routing.
void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream);
void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream);
// Decode MoE (qwen35moe, moe_decode_kq.hip): packed-weight x SwiGLU for 1..8 tokens, top-k sparse, as two
// launches over many blocks (gate/up -> act_scratch, then down -> down_e). Weight formats come from the
// RowDot trait list (src/core/quant/row_dot_traits.hpp: Q4_K, Q5_K, Q8_0 today; a type code that is not
// listed, or a width that is not a multiple of the format's block, returns false). x is F32
// [n_tokens x hidden]; ids/weights come from RouterTopK; act_scratch is F32 [slots x expert_ff] and down_e
// F32 [slots x hidden] for the shared fn_compat::MoeEpilogue. Slots at or above n_tokens*top_k and
// negative expert ids are masked (down_e left untouched), so a fixed grid is capture-compatible. hidden and
// expert_ff must be nonzero multiples of 256 with hidden + expert_ff <= 16384. Returns false for
// out-of-contract shapes or types; the caller keeps the routed WMMA legacy path.
bool LaunchDecodeMoeKQuant(const void* gate_w, WeightType gate_type,
                           const void* up_w, const void* down_w,
                           WeightType down_type, const float* x,
                           const std::int32_t* ids, float* act_scratch,
                           float* down_e, std::uint32_t n_tokens,
                           std::uint32_t top_k, std::size_t hidden,
                           std::size_t expert_ff, hipStream_t stream,
                           std::uint32_t rows_hint = 0);
// Decode MoE, lane geometry (moe_decode_wave.hip; gemma4): one wave per row, merged gate/up layout, masked
// slots (negative ids) write zeros. Weight formats come from the LaneRowFormat trait list
// (src/core/quant/row_dot_traits.hpp: Q4_0, Q8_0 today); a type that is not listed, or a row_bytes that
// does not match the format, returns false and launches nothing so the caller keeps its existing path.
// Gate/up rows of expert e are [e * expert_stride_bytes, +ff * row_bytes) gate, then ff up rows; act_out is
// F32 [slots x ff]; down output is F32 [slots x hidden]. Bit-identical to MoeVecGateUpSwiglu / GemvQSlot.
enum class MoeGateAct : std::uint32_t { kSiluMul = 0, kGeluTanhMul = 1, kSwigluOaiBias = 2 };
// kSwigluOaiBias (gpt-oss) adds gate_b / up_b ([experts x ff] F32, both required) to the dots before the activation.
bool LaunchMoeWaveGateUp(const void* gate_up, WeightType type, std::uint64_t row_bytes,
                         std::uint64_t expert_stride_bytes, const float* x, const std::int32_t* ids, float* act_out,
                         std::uint32_t slots, std::uint32_t used, std::uint32_t ff, std::uint32_t hidden,
                         MoeGateAct act, hipStream_t stream, const float* gate_b = nullptr,
                         const float* up_b = nullptr);
bool LaunchMoeWaveDown(const void* down, WeightType type, std::uint64_t row_bytes, std::uint64_t expert_stride_bytes,
                       const float* act, const std::int32_t* ids, float* out, std::uint32_t slots,
                       std::uint32_t hidden, std::uint32_t ff, hipStream_t stream);
// Fused shared-gate + combine epilogue for the 35B decode MoE path
// (moe_decode_kq.hip; the shared fn_compat::MoeEpilogue is untouched).
// Replaces the per-layer hipblas 1xH glogit SGEMM plus the shared epilogue:
// glogit[t] = shexp_vec . x[t] (F32 sequential dot), then out =
// sum_s w_s * down_e + sigmoid(glogit) * shared, mirroring the shared
// epilogue's k<=16 branch exactly. n_tokens 1..8, top_k 1..16, hidden
// nonzero multiple of 256. Returns false for out-of-contract shapes; the
// caller keeps the SGEMM + shared-epilogue legacy pair.
bool LaunchDecodeMoeEpilogue(const float* expert_out, const float* weights,
                             const float* shared_out,
                             const float* shexp_vec, const float* x,
                             float* out, std::uint32_t n_tokens,
                             std::uint32_t top_k, std::size_t hidden,
                             hipStream_t stream);
// Order-preserving skinny router GEMV (q36, router_gemv.hip): F32
// [n_tokens x hidden] x [hidden x n_experts] for the 256-expert /
// 2048-hidden router shape, replaying the hipblas SGEMM StaggerU order
// (32-row M-tiles start K at tile*64, single FMA chain per output) so
// logits are bitwise identical to the SGEMM path. n_tokens 1..8 only;
// returns false for any other shape and the caller keeps hipblas.
bool LaunchRouterStaggerGemv(const float* x, const float* w, float* out,
                             std::uint32_t n_tokens, std::uint32_t n_experts,
                             std::size_t hidden, hipStream_t stream);
/// Full-vocabulary greedy argmax; scratch holds n_tokens * kArgmaxParts.
void Argmax(const float* logits, ArgmaxCandidate* scratch, std::int32_t* out,
            std::uint32_t n_tokens, std::uint32_t vocab, hipStream_t stream);


// Phase C-3 (27B decode triage, mirrors QwenGemmMode::kHipDecode): narrow
// decode GEMV over the shared cores. kQ8KBlock runs the existing core Q8K
// block kernel (identical to today's kHipQuantDirect route); kDenseLocal
// keeps the caller's local F32/BF16 kernels; kReject preserves the old
// abort behavior (zero/overflow shapes, F16 and other formats with no HIP
// decode kernel, misaligned quant K). Coupling note: the covered quant set
// must stay identical to DescribeQwenGemmFormat's hip_decode_direct set
// (src/models/qwen/gemm_route.hpp); drift would silently reroute shapes.
enum class QwenDecodeRoute { kReject, kQ8KBlock, kDenseLocal };

[[nodiscard]] inline QwenDecodeRoute DecideQwenDecodeRoute(
    core::GgmlType type, std::size_t m, std::size_t k) noexcept {
  if (m == 0 || k == 0) {
    return QwenDecodeRoute::kReject;
  }
  const std::size_t max_size = std::numeric_limits<std::size_t>::max();
  if (m > max_size / k ||
      m > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return QwenDecodeRoute::kReject;
  }
  switch (type) {
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kQ3_K:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kIQ4_NL:
    case core::GgmlType::kIQ4_XS:
    case core::GgmlType::kIQ3_S:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kQ4_0_ROCMFP4:
    case core::GgmlType::kQ4_0_ROCMFP4_FAST:
    case core::GgmlType::kQ8_K:
      break;
    case core::GgmlType::kF32:
    case core::GgmlType::kBF16:
      return QwenDecodeRoute::kDenseLocal;
    default:
      return QwenDecodeRoute::kReject;
  }
  const std::size_t block_elements =
      ::mbufo::quant::QuantizedBlockElements(type);
  if (block_elements == 0 || k % block_elements != 0) {
    return QwenDecodeRoute::kReject;
  }
  return QwenDecodeRoute::kQ8KBlock;
}

}  // namespace mbufo::core::compute::gemm

#endif  // MBUFO_CORE_COMPUTE_GEMM_GEMM_HPP_


