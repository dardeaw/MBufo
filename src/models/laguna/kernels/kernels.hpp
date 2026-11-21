#ifndef MBUFO_MODELS_LAGUNA_KERNELS_KERNELS_HPP_
#define MBUFO_MODELS_LAGUNA_KERNELS_KERNELS_HPP_

#include <cstddef>
#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace mbufo::models::laguna::kernels {

void ConvertBf16ToF16(const std::uint16_t* in, __half* out, std::size_t count,
                      hipStream_t stream);

void ConvertQ6KToQ80(const void* q6k_dev, void* q80_dev, std::size_t n_elems,
                    hipStream_t stream);

void FillSlotTiles(const std::int32_t* ids, std::int32_t* tiles,
                   std::size_t slots, hipStream_t stream);

void BuildRoutedTiles(const std::uint32_t* counts, std::uint32_t n_experts,
                      std::uint32_t tile_rows, std::int32_t* tiles,
                      std::uint32_t* n_tiles, hipStream_t stream);

void RmsNorm(const float* x, const float* weight, float* out,
             std::uint32_t n_tokens, std::uint32_t hidden, float eps,
             hipStream_t stream);

void HeadRmsNorm(float* x, const float* weight, std::uint32_t n_tokens,
                 std::uint32_t n_heads, std::uint32_t head_dim, float eps,
                 hipStream_t stream);

void ApplyDualRope(float* q, float* k, std::uint32_t n_tokens,
                   std::uint32_t start_pos, std::uint32_t n_q_heads,
                   std::uint32_t n_kv_heads, std::uint32_t head_dim,
                   std::uint32_t rotary_dim, float rope_theta,
                   bool is_swa, float yarn_factor, std::uint32_t yarn_orig_ctx,
                   float yarn_beta_fast, float yarn_beta_slow,
                   float yarn_attn_factor, hipStream_t stream);

void SoftplusPerHeadGateMul(float* attn_out, const float* gate_logits,
                            std::uint32_t n_tokens, std::uint32_t n_heads,
                            std::uint32_t head_dim, hipStream_t stream);

void SigmoidBiasTopKRouter(const float* logits, const float* bias,
                           std::int32_t* ids, float* weights,
                           std::uint32_t n_tokens, std::uint32_t n_experts,
                           std::uint32_t top_k, bool norm_topk,
                           float scale, hipStream_t stream);

void SiluMul(const float* gate, const float* up, float* out,
             std::size_t count, hipStream_t stream);

void AddResidual(float* a, const float* b, std::size_t count,
                 hipStream_t stream);

void StoreKvF16(const float* k_in, const float* v_in, __half* k_cache,
                __half* v_cache, std::uint32_t n_tokens,
                std::uint32_t start_pos, std::uint32_t n_kv_heads,
                std::uint32_t head_dim, hipStream_t stream);

void F16AttentionDecode(const float* q, const __half* k_cache,
                        const __half* v_cache, float* out,
                        std::uint32_t seq_len, std::uint32_t sliding_window,
                        std::uint32_t n_q_heads, std::uint32_t n_kv_heads,
                        std::uint32_t head_dim, float scale,
                        hipStream_t stream);

void F16AttentionPrefill(const float* q, const __half* k_cache,
                         const __half* v_cache, float* out,
                         std::uint32_t n_tokens, std::uint32_t start_pos,
                         std::uint32_t sliding_window, std::uint32_t n_q_heads,
                         std::uint32_t n_kv_heads, std::uint32_t head_dim,
                         float scale, hipStream_t stream);

std::int32_t Argmax(const float* logits, std::uint32_t vocab_size,
                    std::int32_t* d_out, hipStream_t stream);

}  // namespace mbufo::models::laguna::kernels

#endif  // MBUFO_MODELS_LAGUNA_KERNELS_KERNELS_HPP_
