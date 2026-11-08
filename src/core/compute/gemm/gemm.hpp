#ifndef GUFO_CORE_COMPUTE_GEMM_GEMM_HPP_
#define GUFO_CORE_COMPUTE_GEMM_GEMM_HPP_

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
#define GUFO_GEMM_F32 0
#define GUFO_GEMM_F16 1
#define GUFO_GEMM_Q4_0 2
#define GUFO_GEMM_Q8_0 8
#define GUFO_GEMM_BF16 30

#include <cstddef>
#include <cstdint>
#include <limits>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/compute/gemm/gemm_dispatcher.hpp"

namespace gufo::core::compute::gemm {

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
// Lightweight decode MoE (qwen35moe, moe_decode_kq.hip): fused per-slot
// K-quant x SwiGLU for 1..8 tokens, top-k sparse. gate/up are Q4_K or Q5_K
// (hidden K, 256-wide blocks); down is Q5_K (256-wide) or load-converted
// Q8_0 (32-wide). x is F32 [n_tokens x hidden]; ids/weights come from
// RouterTopK; down_e receives F32 [slots x hidden] for the shared
// fn_compat::MoeEpilogue. Fixed 64-slot grid with a slot>=live mask
// (capture-compatible); masked slots leave down_e untouched. hidden and
// expert_ff must be nonzero multiples of 256 with hidden + expert_ff <=
// 16384 (LDS budget). Returns false for out-of-contract shapes or types;
// the caller keeps the routed WMMA legacy path.
bool LaunchDecodeMoeKQuant(const void* gate_w, WeightType gate_type,
                           const void* up_w, const void* down_w,
                           WeightType down_type, const float* x,
                           const std::int32_t* ids, float* down_e,
                           std::uint32_t n_tokens, std::uint32_t top_k,
                           std::size_t hidden, std::size_t expert_ff,
                           hipStream_t stream);
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
      ::gufo::quant::QuantizedBlockElements(type);
  if (block_elements == 0 || k % block_elements != 0) {
    return QwenDecodeRoute::kReject;
  }
  return QwenDecodeRoute::kQ8KBlock;
}

}  // namespace gufo::core::compute::gemm

#endif  // GUFO_CORE_COMPUTE_GEMM_GEMM_HPP_


