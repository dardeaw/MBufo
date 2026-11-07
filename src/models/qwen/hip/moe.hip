#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen/hip/moe.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"

namespace gufo::hip {
namespace {

namespace gemm = ::gufo::core::compute::gemm;

// Device-only routed-tile builder: one entry per (expert, bucket chunk) with
// the chunk index in the high bits, matching RoutedF16Gemm's tile decoding.
// No host sync; capacity follows flash-next RoutedTileCapacity.
// Capture-safe decode tile map: two chunk copies per slot at BN = 48
// (K-quants refuse BN > 48; two 48-row chunks cover every decode bucket of
// <= 64 rows). Duplicate experts recompute the same bucket rows
// idempotently (overwrites, single reader downstream).
__global__ void FillSlotTilesKernel(const std::int32_t* __restrict__ ids,
                                    std::int32_t* __restrict__ tiles,
                                    std::size_t slots) {
  const std::size_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < slots) {
    tiles[2 * s] = ids[s];
    tiles[2 * s + 1] = ids[s] | (1 << 16);
  }
}

__global__ void BuildRoutedTilesKernel(const std::uint32_t* __restrict__ counts,
                                       std::uint32_t n_experts,
                                       std::uint32_t tile_rows,
                                       std::int32_t* __restrict__ tiles,
                                       std::uint32_t* __restrict__ n_tiles) {
  const std::uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_experts) {
    return;
  }
  const std::uint32_t c = counts[e];
  const std::uint32_t chunks = (c + tile_rows - 1) / tile_rows;
  for (std::uint32_t j = 0; j < chunks; ++j) {
    const std::uint32_t idx = atomicAdd(n_tiles, 1u);
    tiles[idx] = static_cast<std::int32_t>(e | (j << 16));
  }
}

[[nodiscard]] bool MapGateWeightType(core::GgmlType type,
                                     gemm::WeightType* out) noexcept {
  if (type == core::GgmlType::kQ4_K) {
    *out = gemm::WeightType::kQ4_K;
    return true;
  }
  if (type == core::GgmlType::kQ5_K) {
    *out = gemm::WeightType::kQ5_K;
    return true;
  }
  return false;
}

[[nodiscard]] bool MapDownWeightType(core::GgmlType type,
                                     gemm::WeightType* out) noexcept {
  if (type == core::GgmlType::kQ5_K) {
    *out = gemm::WeightType::kQ5_K;
    return true;
  }
  // Load-time converted Q6_K down weights ride the Q8_0 contract.
  if (type == core::GgmlType::kQ8_0) {
    *out = gemm::WeightType::kQ8_0;
    return true;
  }
  return false;
}

void AssignError(std::string* error_msg, const char* text) {
  if (error_msg != nullptr) {
    *error_msg = text;
  }
}

}  // namespace

bool MoeForwardHip(const core::ModelConfig& config,
                   const models::QwenLayerWeights& layer, const float* x,
                   const void* x_bf16, float* out, std::uint32_t n_tokens,
                   const QwenMoeScratch& scratch, hipStream_t stream,
                   hipblasHandle_t blas, void* preq_scratch,
                   std::size_t preq_bytes, std::string* error_msg) {
  const std::size_t hidden = config.hidden_size;
  const std::size_t num_experts = config.expert_count;
  const std::size_t top_k = config.expert_used_count;
  const std::size_t expert_ff = config.expert_ffn_size;
  const std::size_t shexp_ff = config.shared_expert_ffn_size;
  if (!layer.is_moe || hidden == 0 || num_experts == 0 || top_k == 0 ||
      expert_ff == 0 || shexp_ff == 0 || x == nullptr || out == nullptr ||
      blas == nullptr) {
    AssignError(error_msg, "MoE forward received an invalid invocation");
    return false;
  }
  if (n_tokens == 0) {
    return true;
  }
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * top_k;
  if (slots > scratch.ids.size() || slots > scratch.weights.size() ||
      static_cast<std::size_t>(n_tokens) * hidden > scratch.x_half.size() ||
      slots * expert_ff > scratch.gate_e.size() ||
      slots * expert_ff > scratch.up_half.size() ||
      slots * hidden > scratch.down_e.size() ||
      static_cast<std::size_t>(n_tokens) * shexp_ff >
          scratch.shexp_gate.size() ||
      static_cast<std::size_t>(n_tokens) * hidden > scratch.shexp_out.size()) {
    AssignError(error_msg, "MoE scratch is smaller than the token batch");
    return false;
  }
  if (scratch.router.size() <
      static_cast<std::size_t>(n_tokens) * num_experts) {
    AssignError(error_msg, "MoE router scratch is smaller than the batch");
    return false;
  }

  gemm::WeightType gate_type = gemm::WeightType::kQ4_K;
  if (!MapGateWeightType(layer.ffn_gate_exps.type, &gate_type)) {
    AssignError(error_msg, "MoE gate/up type misses the routed contract");
    return false;
  }
  if (layer.ffn_up_exps.type != layer.ffn_gate_exps.type) {
    AssignError(error_msg, "MoE gate/up types differ within a layer");
    return false;
  }
  gemm::WeightType down_type = gemm::WeightType::kQ5_K;
  if (!MapDownWeightType(layer.ffn_down_exps.type, &down_type)) {
    AssignError(error_msg, "MoE down type misses the routed contract");
    return false;
  }

  HIPBLAS_CHECK(hipblasSetStream(blas, stream));

  // 1. Router logits: [T x N] = X[T x H] * W^T, W = gate_inp [N x H].
  {
    const float alpha = 1.0F, beta = 0.0F;
    HIPBLAS_CHECK(hipblasSgemm(
        blas, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(num_experts),
        static_cast<int>(n_tokens), static_cast<int>(hidden), &alpha,
        static_cast<const float*>(layer.ffn_gate_inp.data),
        static_cast<int>(hidden), x, static_cast<int>(hidden), &beta,
        scratch.router.data(), static_cast<int>(num_experts)));
  }

  // 2. Softmax top-k router (renormalized, matching llama build_moe_ffn).
  gemm::RouterTopK(scratch.router.data(),
                   static_cast<std::uint32_t>(num_experts), scratch.ids.data(),
                   scratch.weights.data(), n_tokens,
                   static_cast<std::uint32_t>(num_experts),
                   static_cast<std::uint32_t>(top_k), stream);

  // 3. Compact slots by expert into 16-row padded buckets.
  gemm::ExpertCounts(scratch.ids.data(), scratch.counts.data(), n_tokens,
                     static_cast<std::uint32_t>(num_experts),
                     static_cast<std::uint32_t>(top_k), stream);
  gemm::RoutedCompact(scratch.ids.data(), scratch.counts.data(),
                      scratch.pad_bounds.data(), scratch.cursors.data(),
                      scratch.rows_token.data(), scratch.rows_slot.data(),
                      n_tokens, static_cast<std::uint32_t>(top_k),
                      static_cast<std::uint32_t>(num_experts), stream);

  // 4. Tile map. Decode batches (<= 8 tokens, captured graphs) use a
  // sync-free per-slot map: buckets never exceed 64 rows, so one 64-row
  // chunk covers every bucket. Larger batches build (expert, chunk) tiles
  // on device and read back the exact count (prefill is never captured).
  const bool small_batch = n_tokens <= 8;
  const std::uint32_t tile_rows =
      small_batch
          ? 48
          : (slots >= static_cast<std::size_t>(16) * num_experts ? 48 : 16);
  if (tile_rows != 16 && tile_rows != 48 && tile_rows != 64) {
    AssignError(error_msg, "MoE tile rows miss the routed contract");
    return false;
  }
  if (small_batch) {
    const unsigned blocks = static_cast<unsigned>((2 * slots + 255) / 256);
    hipLaunchKernelGGL(FillSlotTilesKernel, dim3(blocks), dim3(256), 0, stream,
                       scratch.ids.data(), scratch.tiles.data(), slots);
  } else {
    HIP_CHECK(hipMemsetAsync(scratch.n_tiles.data(), 0, sizeof(std::uint32_t),
                             stream));
    const unsigned blocks = static_cast<unsigned>((num_experts + 255) / 256);
    hipLaunchKernelGGL(BuildRoutedTilesKernel, dim3(blocks), dim3(256), 0,
                       stream, scratch.counts.data(),
                       static_cast<std::uint32_t>(num_experts), tile_rows,
                       scratch.tiles.data(), scratch.n_tiles.data());
  }

  // 5. F16 activations for the routed kernels.
  if (!gemm::ConvertF32ToF16(x, scratch.x_half.data(),
                             static_cast<std::size_t>(n_tokens) * hidden,
                             stream)) {
    AssignError(error_msg, "MoE activation narrowing failed");
    return false;
  }

  // Tile count on host: decode batches know it exactly (one tile per
  // slot); larger batches download the device-built count (prefill is
  // never captured, so one sync per layer is legal there).
  const std::size_t tile_cap =
      (slots + num_experts * 15) / 16 + num_experts + 1;
  if (tile_cap > scratch.tiles.size()) {
    AssignError(error_msg, "MoE tile scratch is smaller than the capacity");
    return false;
  }
  std::uint32_t n_tiles_host = 0;
  if (small_batch) {
    n_tiles_host = static_cast<std::uint32_t>(2 * slots);
  } else {
    HIP_CHECK(hipMemcpyAsync(&n_tiles_host, scratch.n_tiles.data(),
                             sizeof(std::uint32_t), hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  if (n_tiles_host == 0 || n_tiles_host > tile_cap) {
    AssignError(error_msg, "MoE tile builder produced no work");
    return false;
  }

  // 6. Routed gate/up (SwiGLU fused on the up call).
  const std::size_t gate_m = expert_ff;
  if (!gemm::RoutedF16Gemm(layer.ffn_gate_exps.data, gate_type,
                           scratch.x_half.data(), scratch.tiles.data(),
                           n_tiles_host, tile_rows, scratch.pad_bounds.data(),
                           scratch.rows_token.data(), scratch.rows_slot.data(),
                           nullptr, scratch.gate_e.data(), nullptr, gate_m,
                           hidden, stream)) {
    AssignError(error_msg, "MoE routed gate GEMM failed");
    return false;
  }
  if (!gemm::RoutedF16Gemm(layer.ffn_up_exps.data, gate_type,
                           scratch.x_half.data(), scratch.tiles.data(),
                           n_tiles_host, tile_rows, scratch.pad_bounds.data(),
                           scratch.rows_token.data(), scratch.rows_slot.data(),
                           scratch.gate_e.data(), nullptr,
                           scratch.up_half.data(), gate_m, hidden, stream)) {
    AssignError(error_msg, "MoE routed up GEMM failed");
    return false;
  }

  // 7. Routed down into slot-ordered F32.
  if (!gemm::RoutedF16Gemm(layer.ffn_down_exps.data, down_type,
                           scratch.up_half.data(), scratch.tiles.data(),
                           n_tiles_host, tile_rows, scratch.pad_bounds.data(),
                           scratch.rows_slot.data(), scratch.rows_slot.data(),
                           nullptr, scratch.down_e.data(), nullptr, hidden,
                           expert_ff, stream)) {
    AssignError(error_msg, "MoE routed down GEMM failed");
    return false;
  }

  // 8. Shared expert: Q8_0 SwiGLU dense + sigmoid gate scale.
  // Decode (<= 8 tokens) rides the shared narrow-Q8 core; larger batches
  // quantize into the max-sized x_q8 scratch (aliased over the x_bf16
  // storage; sequential use only) and run the prequantized Q8_0 kernels
  // directly. The batch-sized thread-local wrapper over-reads below 64
  // rows, so MoeForwardHip never calls it.
  const bool use_narrow =
      n_tokens >= 1 && n_tokens <= 8 && preq_scratch != nullptr &&
      gemm::DecideNarrowDecodePrequantRoute(
          /*weight_q8_0=*/true, /*gated=*/false, n_tokens, hidden, shexp_ff) ==
          gemm::NarrowDecodeRoute::kNarrowQ80Prequant &&
      gemm::DecideNarrowDecodePrequantRoute(
          /*weight_q8_0=*/true, /*gated=*/false, n_tokens, shexp_ff, hidden) ==
          gemm::NarrowDecodeRoute::kNarrowQ80Prequant &&
      preq_bytes >= gemm::NarrowQ80PrequantBytes(n_tokens, hidden);
  if (use_narrow) {
    if (!gemm::LaunchNarrowQ80Prequant(
            layer.ffn_gate_shexp.data, x, scratch.shexp_gate.data(), hidden,
            shexp_ff, n_tokens, preq_scratch, preq_bytes, stream) ||
        !gemm::LaunchNarrowQ80Prequant(
            layer.ffn_up_shexp.data, x, scratch.shexp_up.data(), hidden,
            shexp_ff, n_tokens, preq_scratch, preq_bytes, stream)) {
      AssignError(error_msg, "MoE shared-expert narrow GEMM failed");
      return false;
    }
    LaunchBatchedSwiGLUActivation(
        scratch.shexp_gate.data(), scratch.shexp_up.data(),
        scratch.shexp_act.data(), nullptr,
        static_cast<std::size_t>(n_tokens) * shexp_ff, stream);
    if (!gemm::LaunchNarrowQ80Prequant(
            layer.ffn_down_shexp.data, scratch.shexp_act.data(),
            scratch.shexp_out.data(), shexp_ff, hidden, n_tokens, preq_scratch,
            preq_bytes, stream)) {
      AssignError(error_msg, "MoE shared-expert narrow down GEMM failed");
      return false;
    }
  } else {
    const void* bf16_in = x_bf16;
    if (bf16_in == nullptr) {
      LaunchFloatToBfloat16(x, scratch.x_bf16.data(),
                            static_cast<std::size_t>(n_tokens) * hidden,
                            stream);
      bf16_in = static_cast<const void*>(scratch.x_bf16.data());
    }
    if (scratch.x_q8.size() <
        Q8ActTiledBytes(n_tokens, hidden) + Q8ActSumBytes(n_tokens, hidden)) {
      AssignError(error_msg, "MoE x_q8 scratch is smaller than the batch");
      return false;
    }
    LaunchQuantizeActivationQ8_1(bf16_in, scratch.x_q8.data(), n_tokens, hidden,
                                 stream);
    LaunchBatchedQuantGEMMPreQuantized(
        layer.ffn_gate_shexp.type, layer.ffn_gate_shexp.data,
        scratch.x_q8.data(), scratch.shexp_gate.data(), n_tokens, shexp_ff,
        hidden, stream);
    LaunchQuantizeActivationQ8_1(bf16_in, scratch.x_q8.data(), n_tokens, hidden,
                                 stream);
    LaunchBatchedQuantGEMMPreQuantized(
        layer.ffn_up_shexp.type, layer.ffn_up_shexp.data, scratch.x_q8.data(),
        scratch.shexp_up.data(), n_tokens, shexp_ff, hidden, stream);
    LaunchBatchedSwiGLUActivation(
        scratch.shexp_gate.data(), scratch.shexp_up.data(),
        scratch.shexp_act.data(), nullptr,
        static_cast<std::size_t>(n_tokens) * shexp_ff, stream);
    LaunchQuantizeActivationQ8_1FromFp32(scratch.shexp_act.data(),
                                         scratch.x_q8.data(), n_tokens,
                                         shexp_ff, stream);
    LaunchBatchedQuantGEMMPreQuantized(
        layer.ffn_down_shexp.type, layer.ffn_down_shexp.data,
        scratch.x_q8.data(), scratch.shexp_out.data(), n_tokens, hidden,
        shexp_ff, stream);
  }

  // 9. Shared gate logit: glogit[t] = shexp_glogit_vec . x[t].
  {
    const float alpha = 1.0F, beta = 0.0F;
    HIPBLAS_CHECK(hipblasSgemm(
        blas, HIPBLAS_OP_N, HIPBLAS_OP_N, 1, static_cast<int>(n_tokens),
        static_cast<int>(hidden), &alpha,
        static_cast<const float*>(layer.ffn_gate_inp_shexp.data), 1, x,
        static_cast<int>(hidden), &beta, scratch.shexp_glogit.data(), 1));
  }

  // 10. Combine: out = sum_s w*down + sigmoid(glogit)*shared.
  gemm::fn_compat::MoeEpilogue(
      scratch.down_e.data(), scratch.weights.data(), scratch.shexp_out.data(),
      scratch.shexp_glogit.data(),
      /*gate_stride=*/1, out, n_tokens, static_cast<std::uint32_t>(top_k),
      static_cast<std::uint32_t>(hidden), stream);
  return true;
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)
