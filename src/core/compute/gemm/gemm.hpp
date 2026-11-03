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

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"

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

}  // namespace gufo::core::compute::gemm

#endif  // GUFO_CORE_COMPUTE_GEMM_GEMM_HPP_
