#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/gemm_route.hpp"
#include "src/models/qwen/hip/blocked_q8.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"

namespace gufo::hip {
namespace {
bool UseQwen27bFp16Prefill(const models::QwenModelWeights& weights,
                           std::size_t batch) {
  const auto& config = weights.config;
  // Large Qwen27B low-bit prefill is qualified independently from the Q8
  // target and from the small-batch decode/verification kernels.
  // ROCmFP4's sub-threshold fallback (dequantize whole weight -> BF16 scratch
  // -> hipBLASLt) re-dequantizes ALL weights on every forward: a ~2.4s fixed
  // cost that dominates short prompts (measured on gfx1151: 13-token TTFT 2.5s
  // vs 0.7s on the in-kernel-decode FP16 WMMA path, which is bit-identical to
  // the fallback from batch 13 up). So FP4 uses WMMA for essentially all
  // prefill; only sub-8-token prompts (rare) fall back. Other low-bit types
  // keep 1024 because their fallback is already the fast int8 WMMA.
  // GUFO_HALF_PREFILL_MIN_BATCH overrides for tuning.
  static const std::size_t kEnvMin = [] {
    const char* e = std::getenv("GUFO_HALF_PREFILL_MIN_BATCH");
    const unsigned long v = e ? std::strtoul(e, nullptr, 10) : 0UL;
    return v >= 2UL ? static_cast<std::size_t>(v) : 0UL;
  }();
  const bool fp4 =
      !weights.layers.empty() &&
      (weights.layers[0].ffn_gate.type == core::GgmlType::kQ4_0_ROCMFP4_FAST ||
       weights.layers[0].ffn_gate.type == core::GgmlType::kQ4_0_ROCMFP4);
  const std::size_t kMinBatch = kEnvMin ? kEnvMin : (fp4 ? 8UL : 1024UL);
  if (batch < kMinBatch || config.hidden_size != 5120 ||
      config.intermediate_size != 17408 || config.num_layers != 64 ||
      config.AttentionSize() != 6144 || config.ssm_inner_size != 6144)
    return false;
  const auto supported = [](const models::QwenTensorRef& tensor) {
    return tensor.type == core::GgmlType::kQ8_0 ||
           detail::IsNativeWmmaQuant(tensor.type) ||
           tensor.type == core::GgmlType::kQ4_0_ROCMFP4_FAST ||
           tensor.type == core::GgmlType::kQ4_0_ROCMFP4;
  };
  bool low_bit_ffn = false;
  for (const auto& layer : weights.layers) {
    if (!supported(layer.ffn_gate) || !supported(layer.ffn_up) ||
        !supported(layer.ffn_down))
      return false;
    low_bit_ffn |= detail::IsNativeWmmaQuant(layer.ffn_gate.type) ||
                   detail::IsNativeWmmaQuant(layer.ffn_up.type) ||
                   detail::IsNativeWmmaQuant(layer.ffn_down.type) ||
                   layer.ffn_gate.type == core::GgmlType::kQ4_0_ROCMFP4_FAST ||
                   layer.ffn_gate.type == core::GgmlType::kQ4_0_ROCMFP4;
    if (layer.is_full_attention) {
      if (!supported(layer.attn_q) || !supported(layer.attn_k) ||
          !supported(layer.attn_v) || !supported(layer.attn_output))
        return false;
    } else if (!supported(layer.attn_qkv) || !supported(layer.attn_gate) ||
               !supported(layer.ssm_alpha) || !supported(layer.ssm_beta) ||
               !supported(layer.ssm_out))
      return false;
  }
  return low_bit_ffn;
}
}  // namespace

tokenization::TokenId QwenGpuExecutor::ForwardPromptChunk(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t batch_size = prompt_tokens.size();
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  const float eps = 1e-6F;

  if (batch_size == 0) {
    return 0;
  }
  if (batch_size > arena_.GetMaxBatch()) {
    throw std::length_error("prompt chunk exceeds the GPU batch length");
  }
  models::qwen::ContinuationWrite write(
      continuation_hooks_, start_pos,
      start_pos + static_cast<std::uint32_t>(batch_size));
  auto scratch = arena_.GetScratchView(batch_size);
  // Blocked-Q8 sidecar (q8 policy only): device copy of the chunk base
  // position for the blocked write kernel. Prefill is eager, so one H2D
  // per chunk on the arena stream is safe (no graph capture here).
  if (policy_.UsesBlockedAttentionKv()) {
    HIP_CHECK(hipMemcpyAsync(arena_.d_blocked_pos, &start_pos,
                             sizeof(start_pos), hipMemcpyHostToDevice,
                             arena_.stream));
  }
  const auto attention_workspace =
      arena_.GetScratchView(arena_.GetMaxBatch()).ffn;
  const std::size_t target_layer_count = arena_.GetTargetLayerCapture().size();
  const std::size_t prompt_capture_offset = h_prompt_hidden_.Size();
  if (capture_prompt_hidden_ && target_layer_count > 0) {
    h_prompt_hidden_.Resize(prompt_capture_offset +
                            batch_size * target_layer_count * hidden_size);
  }

  // 1. Copy prompt token IDs to GPU
  std::vector<std::uint32_t> host_tokens(prompt_tokens.begin(),
                                         prompt_tokens.end());
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_tokens.data(),
                           batch_size * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));

  // 2. Batched Embedding lookup: d_hidden [B, hidden_size]
  LaunchBatchedEmbeddingLookup(
      weights_.token_embd.data, weights_.token_embd.type,
      scratch.decode.prompt_tokens.data(), scratch.decode.hidden.data(),
      batch_size, hidden_size, arena_.stream);

  vision_input_.Inject(scratch.decode.hidden.data(), start_pos, batch_size,
                       hidden_size, 1, arena_.stream);

  const bool half_prefill = UseQwen27bFp16Prefill(weights_, batch_size);
  const auto gemm_weight = [&](const models::QwenTensorRef& w,
                               const void* bf16_input, const float* fp32_input,
                               float* output, std::size_t m, std::size_t k,
                               const void* q8_act = nullptr) {
    if (half_prefill) {
      LaunchBatchedQuantGEMMFp16(w.type, w.data, bf16_input, output, batch_size,
                                 m, k, arena_.stream);
      return;
    }
    // Phase C-3 note: prefill shapes stay on this legacy fine triage
    // (quant-direct WMMA / hipblas / exact-fp32 / dequant+hipblaslt). The
    // shared dispatcher only takes W8A8-varlen-covered shapes, which need
    // tiled-Q8 activations this path does not stage; the hipblaslt runner
    // stays declined (staged but unwired).
    const auto resolution = models::qwen::ResolveQwenGemmRoute(
        {.type = w.type,
         .batch_size = batch_size,
         .m = m,
         .k = k,
         .mode = models::qwen::QwenGemmMode::kHipPrefill});
    if (!resolution.accepted()) {
      std::abort();
    }
    switch (resolution.route) {
      case models::qwen::QwenGemmRoute::kHipPrefillBf16Fp32:
        // Preserve FP32 activation precision and a fixed reduction order.
        LaunchExactBf16GEMMFp32SmallBatch(w.data, fp32_input, output,
                                          batch_size, m, k, arena_.stream);
        return;
      case models::qwen::QwenGemmRoute::kHipPrefillF32Blas:
        LaunchHipblasGEMM(arena_.hipblas_handle, w.data, false, fp32_input,
                          output, batch_size, m, k, arena_.d_scratch_bf16,
                          arena_.stream);
        return;
      case models::qwen::QwenGemmRoute::kHipPrefillQuantDirect: {
        if (w.type == core::GgmlType::kQ8_0 ||
            detail::IsNativeWmmaQuant(w.type)) {
          // Route Q8_0 and every K-quant/IQ format the UD-Q4_K_XL shard uses
          // through the native W-quant x A8 WMMA matrix-core kernel with zero
          // scratch dequantization. Without the second half of this condition
          // the K-quants fell into the branch below, which dequantizes the
          // whole weight matrix to a BF16 scratch buffer and then runs a BF16
          // hipBLASLt GEMM -- paying the dequant AND giving up the 2x matrix
          // -core throughput of the int8 path.
          if (q8_act != nullptr) {
            LaunchBatchedQuantGEMMPreQuantized(w.type, w.data, q8_act, output,
                                               batch_size, m, k, arena_.stream);
          } else {
            LaunchBatchedQuantGEMM(w.type, w.data, bf16_input, output,
                                   batch_size, m, k, arena_.stream);
          }
        } else if (batch_size > 1) {
          LaunchDequantizeToBf16(w.type, w.data, arena_.d_weights_bf16, m * k,
                                 arena_.stream);
          if (arena_.hipblaslt_gemm != nullptr && m >= 1024 && k >= 1024 &&
              arena_.hipblaslt_gemm->RunBf16(arena_.d_weights_bf16, bf16_input,
                                             output, batch_size, m, k,
                                             arena_.stream)) {
            // hipBLASLt executed successfully
          } else {
            LaunchHipblasGEMMBF16(arena_.hipblas_handle, arena_.d_weights_bf16,
                                  bf16_input, output, batch_size, m, k,
                                  arena_.stream);
          }
        } else {
          LaunchBatchedQuantGEMM(w.type, w.data, bf16_input, output, batch_size,
                                 m, k, arena_.stream);
        }
        return;
      }
      default:
        std::abort();
    }
  };

  // 3. Layer stack. One recurrent pass per chunk.
  const auto recurrent_state = arena_.BeginRecurrentWrite();
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    PrefillLayerCheckpoint(l);
    const auto& layer = weights_.layers[l];
    // Cache updates retain every row. The final layer's query/FFN outputs
    // only feed the last token unless this layer is explicitly captured.
    // Keep its original 256-token GEMM tile, including the original tail mask
    // and FP16 rounding boundaries.
    const bool final_output_tile =
        half_prefill && l + 1 == config.num_layers &&
        (!capture_prompt_hidden_ ||
         (target_layer_count != 0 &&
          !arena_.GetTargetLayerCaptureIndex(l).has_value()));
    const std::size_t output_first_row =
        final_output_tile ? ((batch_size - 1) / 256) * 256 : 0;
    const std::size_t output_rows = batch_size - output_first_row;
    float* const output_hidden =
        arena_.d_hidden + output_first_row * hidden_size;
    const auto route_resolution = ResolveQwenLayerRouteWithReasons(
        policy_, QwenExecutionMode::kPrefill, layer.is_full_attention);
    const auto& route_plan = route_resolution.plan;
    detail::EmitQwenRouteResolution(
        "prefill", l, layer.is_full_attention ? "attention" : "ssm",
        route_plan.Fingerprint(),
        static_cast<std::uint32_t>(route_resolution.rejected));

    // opt-q4kxl: "reads the tiled Q8_1 activation" is the property every gate
    // below actually cares about, and it is no longer synonymous with Q8_0.
    // The K-quant/IQ formats in the UD-Q4_K_XL shard take the same
    // pre-quantized WMMA route, so they must drive the same fusion decisions --
    // a gate left at `== kQ8_0` would leave the activation buffer unwritten
    // under a route that reads it.
    const auto reads_q8_act = [&](const models::QwenTensorRef& w) {
      return !half_prefill && (w.type == core::GgmlType::kQ8_0 ||
                               detail::IsNativeWmmaQuant(w.type));
    };
    const auto is_q8 = reads_q8_act;
    // opt-c173-norm-quant: when every projection reading a norm is Q8_0 the
    // FP32 normed row and the BF16 staging copy are both dead, so the norm can
    // write the tiled Q8_1 activation directly and skip two round trips.
    const bool norm_feeds_q8_only =
        IsFusedRMSNormQuantizeQ8_1Supported(hidden_size) &&
        (layer.is_full_attention
             ? (is_q8(layer.attn_q) && is_q8(layer.attn_k) &&
                is_q8(layer.attn_v))
             : (is_q8(layer.attn_qkv) && is_q8(layer.attn_gate) &&
                is_q8(layer.ssm_alpha) && is_q8(layer.ssm_beta)));
    // Where these hold, the BF16 and FP32 buffers passed to gemm_weight below
    // are stale: the route for a Q8_0 tensor always reads the quantized
    // activation, and the flags are exactly the condition that every consumer
    // is Q8_0, so no route can reach them.
    const bool ffn_feeds_q8_only =
        IsFusedRMSNormQuantizeQ8_1Supported(hidden_size) &&
        is_q8(layer.ffn_gate) && is_q8(layer.ffn_up);

    // Pre-layer RMSNorm (generates BF16 into d_scratch_bf16 directly)
    if (half_prefill) {
      LaunchBatchedRMSNormFp16(arena_.d_hidden, nullptr,
                               static_cast<const float*>(layer.attn_norm.data),
                               nullptr, arena_.d_scratch_bf16, batch_size,
                               hidden_size, eps, arena_.stream);
    } else if (norm_feeds_q8_only) {
      LaunchBatchedFusedRMSNormQuantizeQ8_1(
          arena_.d_hidden, /*residual=*/nullptr,
          static_cast<const float*>(layer.attn_norm.data), /*sum_out=*/nullptr,
          arena_.d_scratch_q8_act, batch_size, hidden_size, eps, arena_.stream);
    } else {
      LaunchBatchedRMSNorm(arena_.d_hidden,
                           static_cast<const float*>(layer.attn_norm.data),
                           arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                           hidden_size, eps, arena_.stream);
    }

    if (layer.is_full_attention) {
      // The quantized activation is read only by Q8_0 projections; when q/k/v
      // are all BF16 (as in the Q8_K_XL artifact) nothing consumes it, and when
      // they are all Q8_0 the fused norm already wrote it.
      if (!norm_feeds_q8_only &&
          (reads_q8_act(layer.attn_q) || reads_q8_act(layer.attn_k) ||
           reads_q8_act(layer.attn_v))) {
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     hidden_size, arena_.stream);
      }
      gemm_weight(layer.attn_q, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_qkv, q_projection_size, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.attn_k, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_k, kv_size, hidden_size, arena_.d_scratch_q8_act);
      gemm_weight(layer.attn_v, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_v, kv_size, hidden_size, arena_.d_scratch_q8_act);

      LaunchBatchedUnpackQG(arena_.d_ssm_qkv, arena_.d_q, arena_.d_ssm_gate,
                            batch_size, config.num_attention_heads,
                            config.head_dim, arena_.stream);

      const std::size_t total_k = arena_.GetAttentionKvPlaneElements();
      const std::uint32_t attn_layer_idx = l / config.full_attention_interval;

      // QK-Norm + RoPE + KV-cache write fused into one kernel
      // (opt-c010-qk-rope-kv). The unfused chain stays wired behind the policy
      // toggle as the independent reference.
      const bool fused_qknorm_rope_kv =
          route_plan.fuse_qk_norm_rope_kv &&
          detail::IsFusedQkNormSupported(config.head_dim);
      if (fused_qknorm_rope_kv) {
        LaunchBatchedFusedQKNormRoPEKvWrite(
            arena_.d_q, arena_.d_k, arena_.d_v,
            static_cast<const float*>(layer.attn_q_norm.data),
            static_cast<const float*>(layer.attn_k_norm.data), arena_.d_q,
            arena_.d_k, arena_.d_kv_cache,
            OffsetIfPresent(arena_.d_kv_cache, total_k),
            arena_.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                total_k),
            attn_layer_idx, start_pos, batch_size, arena_.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.rotary_dim, config.rope_theta, eps,
            arena_.stream, vision_input_.rope());
      } else {
        if (!layer.attn_q_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              arena_.d_q, static_cast<const float*>(layer.attn_q_norm.data),
              arena_.d_q, batch_size, config.num_attention_heads,
              config.head_dim, eps, arena_.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              arena_.d_k, static_cast<const float*>(layer.attn_k_norm.data),
              arena_.d_k, batch_size, config.num_key_value_heads,
              config.head_dim, eps, arena_.stream);
        }

        LaunchBatchedRoPE(
            arena_.d_q, arena_.d_k, batch_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, config.rotary_dim,
            start_pos, config.rope_theta, arena_.stream, vision_input_.rope());
      }

      // Blocked-Q8 sidecar (q8 policy only): the fused/unfused write above
      // keeps the F16 plane canonical for the legacy prefill read below, and
      // the same final rows are appended to the layer's blocked cache here
      // for later blocked decode reads. Both branches leave the final rows
      // in arena_.d_k/d_v.
      if (policy_.UsesBlockedAttentionKv()) {
        const std::size_t blocked_layer_bytes = arena_.BlockedKvLayerBytes();
        WriteQwenBlockedKv(
            arena_.d_k, arena_.d_v,
            arena_.d_attention_kv_blocked_k +
                static_cast<std::size_t>(attn_layer_idx) * blocked_layer_bytes,
            arena_.d_attention_kv_blocked_v +
                static_cast<std::size_t>(attn_layer_idx) * blocked_layer_bytes,
            static_cast<std::uint32_t>(batch_size),
            config.num_key_value_heads, config.head_dim, arena_.d_blocked_pos,
            arena_.stream);
      }

      const std::size_t visible_context =
          static_cast<std::size_t>(start_pos) + batch_size;
      // One masked WMMA pass covers the visible prefix and causal diagonal.
      // Unsupported shapes use the tiled or scalar fallback below.
      // opt-c180-kv-resync: the fused QK-norm/RoPE kernel above already wrote
      // this chunk's K and V into the selected canonical cache plane. Earlier
      // chunks and all three decode paths maintain that same plane. The
      // attention launcher's own pack pass would therefore rewrite identical
      // bytes, and an FP16 prefix sync would reconvert a prefix that already
      // matches -- work that grows linearly with depth. The unfused fallback
      // below only applies RoPE in place and never touches the cache, so it
      // still needs the pack.
      const bool kv_already_written = fused_qknorm_rope_kv;
      const bool wmma_attention = LaunchQwenWmmaAttention(
          arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
          arena_.d_kv_cache, OffsetIfPresent(arena_.d_kv_cache, total_k),
          arena_.d_attention_kv_f16,
          OffsetIfPresent(
              static_cast<std::uint16_t*>(arena_.d_attention_kv_f16), total_k),
          arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
          arena_.GetMaxContext(), config.num_attention_heads,
          config.num_key_value_heads, config.head_dim, arena_.stream,
          /*lse_out=*/nullptr, /*key_begin=*/0,
          /*skip_kv_write=*/kv_already_written, attention_workspace.gate,
          attention_workspace.up, output_first_row);
      if (wmma_attention) {
        detail::EmitAttentionDispatch("prefill_wmma", "");
      }

      if (!wmma_attention) {
        bool tiled_attention = false;
        detail::DispatchPrefillAttention(
            visible_context,
            [&] {
              tiled_attention = LaunchBatchedAttentionTile(
                  arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                  arena_.d_kv_cache,
                  OffsetIfPresent(arena_.d_kv_cache, total_k),
                  arena_.d_attention_kv_f16,
                  OffsetIfPresent(
                      static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                      total_k),
                  arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
                  arena_.GetMaxContext(), config.num_attention_heads,
                  config.num_key_value_heads, config.head_dim, arena_.stream);
              return tiled_attention;
            },
            [&] {
              LaunchBatchedAttention(
                  arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                  arena_.d_kv_cache,
                  OffsetIfPresent(arena_.d_kv_cache, total_k),
                  arena_.d_attention_kv_f16,
                  OffsetIfPresent(
                      static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                      total_k),
                  arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
                  arena_.GetMaxContext(), config.num_attention_heads,
                  config.num_key_value_heads, config.head_dim, arena_.stream,
                  fused_qknorm_rope_kv);
            });
        if (tiled_attention) {
          detail::EmitAttentionDispatch("prefill_tiled", "");
        } else if (!detail::BatchedAttentionScoresFitLds(visible_context)) {
          detail::EmitAttentionDispatch(
              arena_.d_kv_cache != nullptr ? "prefill_online_fp32"
                                           : "prefill_online_fp16",
              "prefill_baseline: score_rows_exceed_lds");
        } else {
          detail::EmitAttentionDispatch(
              "prefill_baseline",
              detail::ShouldAttemptOptimizedAttention(visible_context)
                  ? "prefill_tiled: rejected"
                  : "prefill_tiled: below_threshold");
        }
      }

      // opt-c172-fp32-quant: the Q8_0 route reads only the quantized
      // activation, so the BF16 staging buffer is dead. Quantizing straight
      // from FP32 drops one launch and one round trip (about 75 MB per layer
      // at batch 2048) and keeps the activation's full precision going into the
      // Q8_1 codes instead of rounding to BF16 first.
      if (half_prefill) {
        LaunchFloatToFp16(arena_.d_ssm_out + output_first_row * attention_size,
                          arena_.d_scratch_bf16, output_rows * attention_size,
                          arena_.stream);
      } else if (reads_q8_act(layer.attn_output)) {
        LaunchQuantizeActivationQ8_1FromFp32(
            arena_.d_ssm_out, arena_.d_scratch_q8_act, batch_size,
            attention_size, arena_.stream);
      } else {
        LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                              batch_size * attention_size, arena_.stream);
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     attention_size, arena_.stream);
      }
      if (half_prefill) {
        LaunchBatchedQuantGEMMResidualFp16(
            layer.attn_output.type, layer.attn_output.data,
            arena_.d_scratch_bf16, output_hidden, output_rows, hidden_size,
            attention_size, arena_.stream);
      } else {
        gemm_weight(layer.attn_output, arena_.d_scratch_bf16, arena_.d_ssm_out,
                    arena_.d_attn_out, hidden_size, attention_size,
                    arena_.d_scratch_q8_act);
      }
    } else {
      if (!half_prefill && !norm_feeds_q8_only) {
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     hidden_size, arena_.stream);
      }
      gemm_weight(layer.attn_qkv, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_qkv, ssm_qkv_size, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.attn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_gate, ssm_inner_size, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.ssm_alpha, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_alpha_buf, time_step_rank, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.ssm_beta, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_beta_buf, time_step_rank, hidden_size,
                  arena_.d_scratch_q8_act);
      if (arena_.IsSsmReplayCaptureActive()) {
        LaunchCaptureBatchedSsmReplay(
            arena_.d_ssm_qkv, arena_.d_alpha_buf, arena_.d_beta_buf,
            arena_.GetSsmReplayCapture(), config.SsmLayerIndex(l), start_pos,
            batch_size, ssm_qkv_size, time_step_rank, arena_.stream);
      }

      // The row-split recurrence parallelizes independent state rows. Its
      // epilogue emits the activation consumed by ssm_out directly.
      const bool ssm_epilogue_q8 = reads_q8_act(layer.ssm_out) &&
                                   IsFusedSSMEpilogueQuantizeQ8_1Supported(
                                       config.SsmValueSize(), ssm_inner_size);
      const bool ssm_row_split = detail::ShouldUseSsmRowSplitRecurrence(
          IsDeltaNetRowSplitSupported(config.ssm_state_size,
                                      config.SsmValueSize()),
          arena_.d_ssm_kq_scales != nullptr &&
              arena_.d_ssm_alpha_beta != nullptr);
      if (ssm_row_split) {
        LaunchBatchedSSMConvRecurrenceRowSplit(
            arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
            recurrent_state, arena_.d_conv_out, arena_.d_alpha_buf,
            arena_.d_beta_buf, static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
            arena_.d_ssm_out,
            ssm_epilogue_q8 ? arena_.d_scratch_q8_act : nullptr,
            arena_.d_ssm_kq_scales, arena_.d_ssm_alpha_beta,
            config.SsmLayerIndex(l), batch_size, ssm_qkv_size,
            config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(), arena_.stream,
            arena_.GetRecurrentStateStorage(),
            half_prefill ? arena_.d_scratch_bf16 : nullptr);
      } else {
        LaunchBatchedSSMConvRecurrence(
            arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
            recurrent_state, arena_.d_conv_out, arena_.d_alpha_buf,
            arena_.d_beta_buf, static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
            arena_.d_ssm_out, config.SsmLayerIndex(l), batch_size, ssm_qkv_size,
            config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(), arena_.stream,
            arena_.GetRecurrentStateStorage());
      }

      if (half_prefill) {
        if (!ssm_row_split) {
          LaunchFloatToFp16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                            batch_size * ssm_inner_size, arena_.stream);
        }
      } else if (ssm_row_split && ssm_epilogue_q8) {
        // The recurrence epilogue already wrote the quantized activation.
      } else if (reads_q8_act(layer.ssm_out)) {
        LaunchQuantizeActivationQ8_1FromFp32(
            arena_.d_ssm_out, arena_.d_scratch_q8_act, batch_size,
            ssm_inner_size, arena_.stream);
      } else {
        LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                              batch_size * ssm_inner_size, arena_.stream);
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     ssm_inner_size, arena_.stream);
      }
      if (half_prefill) {
        LaunchBatchedQuantGEMMResidualFp16(
            layer.ssm_out.type, layer.ssm_out.data, arena_.d_scratch_bf16,
            arena_.d_hidden, batch_size, hidden_size, ssm_inner_size,
            arena_.stream);
      } else {
        gemm_weight(layer.ssm_out, arena_.d_scratch_bf16, arena_.d_ssm_out,
                    arena_.d_attn_out, hidden_size, ssm_inner_size,
                    arena_.d_scratch_q8_act);
      }
    }

    if (half_prefill) {
      LaunchBatchedRMSNormFp16(output_hidden, nullptr,
                               static_cast<const float*>(layer.ffn_norm.data),
                               nullptr, arena_.d_scratch_bf16, output_rows,
                               hidden_size, eps, arena_.stream);
    } else if (ffn_feeds_q8_only) {
      // The post-attention residual add folds into the norm: one pass reads the
      // hidden state and the attention output, writes the updated hidden state
      // for the next residual link, and emits the Q8_1 activation.
      LaunchBatchedFusedRMSNormQuantizeQ8_1(
          arena_.d_hidden, arena_.d_attn_out,
          static_cast<const float*>(layer.ffn_norm.data), arena_.d_hidden,
          arena_.d_scratch_q8_act, batch_size, hidden_size, eps, arena_.stream);
    } else {
      LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                               arena_.d_hidden, batch_size, hidden_size,
                               arena_.stream);

      // FFN RMSNorm (generates BF16 into d_scratch_bf16 directly)
      LaunchBatchedRMSNorm(arena_.d_hidden,
                           static_cast<const float*>(layer.ffn_norm.data),
                           arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                           hidden_size, eps, arena_.stream);
    }

    // Which buffer holds the Q8_1 form of the SwiGLU output ffn_down consumes.
    // The fused epilogue below moves it out of the shared activation scratch.
    const void* ffn_down_q8_act = arena_.d_scratch_q8_act;
    const void* ffn_down_input = arena_.d_scratch_bf16;

    {
      if (!half_prefill && !ffn_feeds_q8_only) {
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     hidden_size, arena_.stream);
      }
      // opt-c192-swiglu-epilogue: when ffn_down also reads the Q8_1 activation,
      // the whole gate -> up -> SwiGLU -> quantize chain collapses into the two
      // projections, because the up projection can apply SwiGLU against the
      // already-written gate and emit the quantized activation from its own
      // accumulator. The FP32 up intermediate is then dead, so its allocation
      // is where the Q8_1 result goes -- reusing it keeps resident memory flat
      // and avoids writing the [batch, k] activation the same kernel is
      // reading.
      const bool fused_dual_swiglu =
          reads_q8_act(layer.ffn_gate) && reads_q8_act(layer.ffn_up) &&
          layer.ffn_gate.type == layer.ffn_up.type &&
          reads_q8_act(layer.ffn_down) &&
          IsFusedSwiGluGemmEpilogueSupported(
              layer.ffn_gate.type, batch_size, intermediate_size,
              batch_size * intermediate_size * sizeof(float));
      if (half_prefill) {
        const bool paired = TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
            layer.ffn_gate.type, layer.ffn_up.type, layer.ffn_gate.data,
            layer.ffn_up.data, arena_.d_scratch_bf16, arena_.d_ffn_up,
            output_rows, intermediate_size, hidden_size, arena_.stream);
        if (!paired) {
          LaunchBatchedQuantGEMMFp16(layer.ffn_gate.type, layer.ffn_gate.data,
                                     arena_.d_scratch_bf16, arena_.d_ffn_gate,
                                     output_rows, intermediate_size,
                                     hidden_size, arena_.stream);
          LaunchBatchedQuantGEMMSwiGLUFp16(
              layer.ffn_up.type, layer.ffn_up.data, arena_.d_scratch_bf16,
              arena_.d_ffn_gate, arena_.d_ffn_up, output_rows,
              intermediate_size, hidden_size, arena_.stream);
        }
        ffn_down_input = arena_.d_ffn_up;
      } else if (fused_dual_swiglu) {
        LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1(
            layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data,
            arena_.d_scratch_q8_act, arena_.d_ffn_gate, arena_.d_ffn_up,
            batch_size, intermediate_size, hidden_size, arena_.stream);
        ffn_down_q8_act = arena_.d_ffn_up;
      } else if (reads_q8_act(layer.ffn_gate) && reads_q8_act(layer.ffn_up) &&
                 layer.ffn_gate.type == layer.ffn_up.type) {
        LaunchBatchedDualQuantGEMMPreQuantized(
            layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data,
            arena_.d_scratch_q8_act, arena_.d_ffn_gate, arena_.d_ffn_up,
            batch_size, intermediate_size, hidden_size, arena_.stream);
      } else {
        gemm_weight(layer.ffn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_ffn_gate, intermediate_size, hidden_size,
                    arena_.d_scratch_q8_act);

        gemm_weight(layer.ffn_up, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_ffn_up, intermediate_size, hidden_size,
                    arena_.d_scratch_q8_act);
      }

      // opt-c164-swiglu-quant: when ffn_down reads Q8_0 the only consumer of
      // the activation is the quantized buffer, so SwiGLU can write it
      // directly. That drops the FP32 activation and BF16 scratch round trips
      // (about 500 MB per layer at batch 2048) and one kernel launch. Other
      // ffn_down formats still need the FP32/BF16 forms, so they keep the
      // unfused chain.
      if (half_prefill || fused_dual_swiglu) {
        // The up projection's epilogue already wrote the quantized activation.
      } else if (reads_q8_act(layer.ffn_down)) {
        LaunchBatchedFusedSwiGLUQuantizeQ8_1(
            arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_scratch_q8_act,
            batch_size, intermediate_size, arena_.stream);
      } else {
        LaunchBatchedSwiGLUActivation(arena_.d_ffn_gate, arena_.d_ffn_up,
                                      arena_.d_ffn_act, arena_.d_scratch_bf16,
                                      batch_size * intermediate_size,
                                      arena_.stream);
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     intermediate_size, arena_.stream);
      }
    }

    if (half_prefill) {
      LaunchBatchedQuantGEMMResidualFp16(
          layer.ffn_down.type, layer.ffn_down.data, ffn_down_input,
          output_hidden, output_rows, hidden_size, intermediate_size,
          arena_.stream);
    } else {
      gemm_weight(layer.ffn_down, ffn_down_input, arena_.d_ffn_act,
                  arena_.d_ffn_out, hidden_size, intermediate_size,
                  ffn_down_q8_act);
      LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out,
                               arena_.d_hidden, batch_size, hidden_size,
                               arena_.stream);
    }


    if (capture_prompt_hidden_) {
      if (const auto tap_index = arena_.GetTargetLayerCaptureIndex(l);
          tap_index.has_value()) {
        float* destination = h_prompt_hidden_.Data() + prompt_capture_offset +
                             (*tap_index * hidden_size);
        HIP_CHECK(hipMemcpy2DAsync(
            destination, target_layer_count * hidden_size * sizeof(float),
            scratch.decode.hidden.data(), hidden_size * sizeof(float),
            hidden_size * sizeof(float), batch_size, hipMemcpyDeviceToHost,
            arena_.stream));
        // Bound the work queued past a tap, as the blocking pageable copy
        // did, so cancelling a prefill stays prompt.
        if (cancellation_check_) {
          HIP_CHECK(hipStreamSynchronize(arena_.stream));
        }
      }
    }
  }

  if (capture_prompt_hidden_) {
    if (target_layer_count == 0) {
      const std::size_t old_size = h_prompt_hidden_.Size();
      const std::size_t chunk_elements = batch_size * hidden_size;
      h_prompt_hidden_.Resize(old_size + chunk_elements);
      HIP_CHECK(hipMemcpyAsync(h_prompt_hidden_.Data() + old_size,
                               scratch.decode.hidden.data(),
                               chunk_elements * sizeof(float),
                               hipMemcpyDeviceToHost, arena_.stream));
    }
    HIP_CHECK(hipStreamSynchronize(arena_.stream));
  }
  last_hidden_offset_ = (batch_size - 1) * hidden_size;

  if (!compute_logits) {
    write.Commit(start_pos + static_cast<std::uint32_t>(batch_size), false);
    return 0;
  }

  // 4. Output Norm for final token
  const float* final_hidden =
      scratch.decode.hidden.data() + ((batch_size - 1) * hidden_size);
  LaunchRMSNorm(final_hidden,
                static_cast<const float*>(weights_.output_norm.data),
                scratch.decode.normed.data(), hidden_size, eps, arena_.stream);

  // 5. LM Head Logits GEMV on final token
  LaunchGEMV(weights_.output.data, weights_.output.type,
             scratch.decode.normed.data(), scratch.decode.logits.data(),
             vocab_size, hidden_size, arena_.stream, arena_.d_preq_q81,
             arena_.preq_q81_bytes);

  // 6. GPU Argmax. The sampled-token view enters its documented alias epoch
  // only after all SSM layer uses of alpha have completed.
  auto* d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUArgmax(scratch.decode.logits.data(), d_out_token, vocab_size,
                  arena_.stream);

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  const auto result = CheckedSampleToken(next_token_id, config.vocab_size);
  write.Commit(start_pos + static_cast<std::uint32_t>(batch_size), true);
  return result;
}

std::vector<tokenization::TokenId> QwenGpuExecutor::ForwardVerificationChunk(
    std::span<const tokenization::TokenId> candidate_tokens,
    std::uint32_t start_pos, bool capture_logits) {
  // Blocked-Q8 verification is maintained by the batched decode-equivalent
  // path (F16 plane plus blocked sidecars), matching q8 AR decode exactly.
  return ForwardDecodeEquivalentVerificationChunk(candidate_tokens, start_pos,
                                                  capture_logits);
}

void QwenGpuExecutor::CommitVerificationChunk(
    std::span<const tokenization::TokenId> committed_tokens,
    std::uint32_t start_pos) {
  if (committed_tokens.empty()) {
    return;
  }
  CheckReset();
  if (start_pos > arena_.GetMaxContext() ||
      committed_tokens.size() > arena_.GetMaxContext() - start_pos)
    throw std::length_error("committed Qwen chunk exceeds context");
  models::qwen::ContinuationWrite write(
      continuation_hooks_, start_pos,
      start_pos + static_cast<std::uint32_t>(committed_tokens.size()));
  std::size_t replayed = 0;
  if (replaying_ssm_state_) {
    while (replayed < committed_tokens.size() &&
           arena_.CanReplaySsmPosition(start_pos +
                                       static_cast<std::uint32_t>(replayed)))
      ++replayed;
    if (replayed != 0)
      ReplaySsmState(start_pos, static_cast<std::uint32_t>(replayed));
  }
  replaying_ssm_state_ = false;
  if (replayed == committed_tokens.size()) {
    write.Commit(
        start_pos + static_cast<std::uint32_t>(committed_tokens.size()), false);
    return;
  }
  arena_.DisableSsmReplayCapture();

  const bool capture_hidden = capture_prompt_hidden_;
  capture_prompt_hidden_ = false;
  try {
    // Unrecorded rows must use the original decode arithmetic. Prefill
    // projections can round differently and leave a different recurrent state.
    for (std::size_t row = replayed; row < committed_tokens.size(); ++row)
      (void)ForwardToken(committed_tokens[row],
                         start_pos + static_cast<std::uint32_t>(row), false);
  } catch (...) {
    capture_prompt_hidden_ = capture_hidden;
    throw;
  }
  capture_prompt_hidden_ = capture_hidden;
  write.Commit(start_pos + static_cast<std::uint32_t>(committed_tokens.size()),
               false);
}

// ---- Cross-session prefill batch (unified mixed-batch attention) ----
// New in mixed-batch-unified: layer-synchronous prefill over independent
// sessions. Single-session ForwardPromptChunk above is untouched; the
// per-session phases below duplicate its call sequence against the same
// Launch* primitives, with one unified attention launch per attention
// layer covering every session's rows.

namespace {

bool ReadsQ8ActivationForPrefill(core::GgmlType type, bool half_prefill) {
  return !half_prefill && (type == core::GgmlType::kQ8_0 ||
                           detail::IsNativeWmmaQuant(type));
}

}  // namespace

void QwenGpuExecutor::PrefillGemmWeight(
    const models::QwenTensorRef& w, const void* bf16_input,
    const float* fp32_input, float* output, std::size_t batch, std::size_t m,
    std::size_t k, const void* q8_act, bool half_prefill) {
  if (half_prefill) {
    LaunchBatchedQuantGEMMFp16(w.type, w.data, bf16_input, output, batch, m,
                               k, arena_.stream);
    return;
  }
  // Phase C-3 note: prefill shapes stay on this legacy fine triage
  // (quant-direct WMMA / hipblas / exact-fp32 / dequant+hipblaslt). The
  // shared dispatcher only takes W8A8-varlen-covered shapes, which need
  // tiled-Q8 activations this path does not stage; the hipblaslt runner
  // stays declined (staged but unwired).
  const auto resolution = models::qwen::ResolveQwenGemmRoute(
      {.type = w.type,
       .batch_size = batch,
       .m = m,
       .k = k,
       .mode = models::qwen::QwenGemmMode::kHipPrefill});
  if (!resolution.accepted()) {
    std::abort();
  }
  switch (resolution.route) {
    case models::qwen::QwenGemmRoute::kHipPrefillBf16Fp32:
      LaunchExactBf16GEMMFp32SmallBatch(w.data, fp32_input, output, batch, m,
                                        k, arena_.stream);
      return;
    case models::qwen::QwenGemmRoute::kHipPrefillF32Blas:
      LaunchHipblasGEMM(arena_.hipblas_handle, w.data, false, fp32_input,
                        output, batch, m, k, arena_.d_scratch_bf16,
                        arena_.stream);
      return;
    case models::qwen::QwenGemmRoute::kHipPrefillQuantDirect: {
      if (w.type == core::GgmlType::kQ8_0 ||
          detail::IsNativeWmmaQuant(w.type)) {
        if (q8_act != nullptr) {
          LaunchBatchedQuantGEMMPreQuantized(w.type, w.data, q8_act, output,
                                             batch, m, k, arena_.stream);
        } else {
          LaunchBatchedQuantGEMM(w.type, w.data, bf16_input, output, batch, m,
                                 k, arena_.stream);
        }
      } else if (batch > 1) {
        LaunchDequantizeToBf16(w.type, w.data, arena_.d_weights_bf16, m * k,
                               arena_.stream);
        if (arena_.hipblaslt_gemm != nullptr && m >= 1024 && k >= 1024 &&
            arena_.hipblaslt_gemm->RunBf16(arena_.d_weights_bf16, bf16_input,
                                           output, batch, m, k, arena_.stream)) {
        } else {
          LaunchHipblasGEMMBF16(arena_.hipblas_handle, arena_.d_weights_bf16,
                                bf16_input, output, batch, m, k,
                                arena_.stream);
        }
      } else {
        LaunchBatchedQuantGEMM(w.type, w.data, bf16_input, output, batch, m,
                               k, arena_.stream);
      }
      return;
    }
    default:
      std::abort();
  }
}

void QwenGpuExecutor::PrefillLayerNorm(const PrefillChunkCtx& ctx,
                                         std::uint32_t l) {
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t batch_size = ctx.batch_size;
  const float eps = 1e-6F;
  const auto& layer = weights_.layers[l];
  const bool half_prefill = ctx.half_prefill;

  // Pre-layer RMSNorm (generates BF16 into d_scratch_bf16 directly).
  // Shared by the attention and SSM paths: single runs it for every layer
  // before branching.
  if (half_prefill) {
    LaunchBatchedRMSNormFp16(arena_.d_hidden, nullptr,
                             static_cast<const float*>(layer.attn_norm.data),
                             nullptr, arena_.d_scratch_bf16, batch_size,
                             hidden_size, eps, arena_.stream);
  } else if (ctx.norm_feeds_q8_only) {
    LaunchBatchedFusedRMSNormQuantizeQ8_1(
        arena_.d_hidden, /*residual=*/nullptr,
        static_cast<const float*>(layer.attn_norm.data), /*sum_out=*/nullptr,
        arena_.d_scratch_q8_act, batch_size, hidden_size, eps, arena_.stream);
  } else {
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.attn_norm.data),
                         arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                         hidden_size, eps, arena_.stream);
  }
}

void QwenGpuExecutor::PrefillAttentionPre(const PrefillChunkCtx& ctx,
                                           std::uint32_t l) {
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t batch_size = ctx.batch_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const float eps = 1e-6F;
  const auto& layer = weights_.layers[l];
  const bool half_prefill = ctx.half_prefill;

  PrefillLayerNorm(ctx, l);

  if (!ctx.norm_feeds_q8_only &&
      (ReadsQ8ActivationForPrefill(layer.attn_q.type, half_prefill) ||
       ReadsQ8ActivationForPrefill(layer.attn_k.type, half_prefill) ||
       ReadsQ8ActivationForPrefill(layer.attn_v.type, half_prefill))) {
    LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                 arena_.d_scratch_q8_act, batch_size,
                                 hidden_size, arena_.stream);
  }
  PrefillGemmWeight(layer.attn_q, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_ssm_qkv, batch_size, q_projection_size, hidden_size,
                    arena_.d_scratch_q8_act, half_prefill);
  PrefillGemmWeight(layer.attn_k, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_k, batch_size, kv_size, hidden_size, arena_.d_scratch_q8_act,
                    half_prefill);
  PrefillGemmWeight(layer.attn_v, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_v, batch_size, kv_size, hidden_size, arena_.d_scratch_q8_act,
                    half_prefill);

  LaunchBatchedUnpackQG(arena_.d_ssm_qkv, arena_.d_q, arena_.d_ssm_gate,
                        batch_size, config.num_attention_heads,
                        config.head_dim, arena_.stream);

  const std::size_t total_k = arena_.GetAttentionKvPlaneElements();
  const std::uint32_t attn_layer_idx = l / config.full_attention_interval;
  if (ctx.fuse_qknorm_rope_kv &&
      detail::IsFusedQkNormSupported(config.head_dim)) {
    LaunchBatchedFusedQKNormRoPEKvWrite(
        arena_.d_q, arena_.d_k, arena_.d_v,
        static_cast<const float*>(layer.attn_q_norm.data),
        static_cast<const float*>(layer.attn_k_norm.data), arena_.d_q,
        arena_.d_k, arena_.d_kv_cache,
        OffsetIfPresent(arena_.d_kv_cache, total_k),
        arena_.d_attention_kv_f16,
        OffsetIfPresent(
            static_cast<std::uint16_t*>(arena_.d_attention_kv_f16), total_k),
        attn_layer_idx, ctx.start_pos, batch_size, arena_.GetMaxContext(),
        config.num_attention_heads, config.num_key_value_heads,
        config.head_dim, config.rotary_dim, config.rope_theta, eps,
        arena_.stream, vision_input_.rope());
  } else {
    if (!layer.attn_q_norm.empty()) {
      LaunchBatchedPerHeadRMSNorm(
          arena_.d_q, static_cast<const float*>(layer.attn_q_norm.data),
          arena_.d_q, batch_size, config.num_attention_heads,
          config.head_dim, eps, arena_.stream);
    }
    if (!layer.attn_k_norm.empty()) {
      LaunchBatchedPerHeadRMSNorm(
          arena_.d_k, static_cast<const float*>(layer.attn_k_norm.data),
          arena_.d_k, batch_size, config.num_key_value_heads,
          config.head_dim, eps, arena_.stream);
    }
    LaunchBatchedRoPE(
        arena_.d_q, arena_.d_k, batch_size, config.num_attention_heads,
        config.num_key_value_heads, config.head_dim, config.rotary_dim,
        ctx.start_pos, config.rope_theta, arena_.stream, vision_input_.rope());
  }
}

void QwenGpuExecutor::PrefillAttentionPost(const PrefillChunkCtx& ctx,
                                            std::uint32_t l, bool run_legacy) {
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t batch_size = ctx.batch_size;
  const std::size_t attention_size = config.AttentionSize();
  const bool half_prefill = ctx.half_prefill;
  const auto& layer = weights_.layers[l];

  if (run_legacy) {
    const std::size_t total_k = arena_.GetAttentionKvPlaneElements();
    const std::uint32_t attn_layer_idx = l / config.full_attention_interval;
    const std::size_t visible_context =
        static_cast<std::size_t>(ctx.start_pos) + batch_size;
    const bool kv_already_written = ctx.fuse_qknorm_rope_kv;
    const bool wmma_attention = LaunchQwenWmmaAttention(
        arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
        arena_.d_kv_cache, OffsetIfPresent(arena_.d_kv_cache, total_k),
        arena_.d_attention_kv_f16,
        OffsetIfPresent(
            static_cast<std::uint16_t*>(arena_.d_attention_kv_f16), total_k),
        arena_.d_ssm_out, attn_layer_idx, ctx.start_pos, batch_size,
        arena_.GetMaxContext(), config.num_attention_heads,
        config.num_key_value_heads, config.head_dim, arena_.stream,
        /*lse_out=*/nullptr, /*key_begin=*/0,
        /*skip_kv_write=*/kv_already_written, ctx.ws_gate, ctx.ws_up,
        ctx.output_first_row);
    if (wmma_attention) {
      detail::EmitAttentionDispatch("prefill_wmma", "");
    }
    if (!wmma_attention) {
      bool tiled_attention = false;
      detail::DispatchPrefillAttention(
          visible_context,
          [&] {
            tiled_attention = LaunchBatchedAttentionTile(
                arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                arena_.d_kv_cache,
                OffsetIfPresent(arena_.d_kv_cache, total_k),
                arena_.d_attention_kv_f16,
                OffsetIfPresent(
                    static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                    total_k),
                arena_.d_ssm_out, attn_layer_idx, ctx.start_pos, batch_size,
                arena_.GetMaxContext(), config.num_attention_heads,
                config.num_key_value_heads, config.head_dim, arena_.stream);
            return tiled_attention;
          },
          [&] {
            LaunchBatchedAttention(
                arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                arena_.d_kv_cache,
                OffsetIfPresent(arena_.d_kv_cache, total_k),
                arena_.d_attention_kv_f16,
                OffsetIfPresent(
                    static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                    total_k),
                arena_.d_ssm_out, attn_layer_idx, ctx.start_pos, batch_size,
                arena_.GetMaxContext(), config.num_attention_heads,
                config.num_key_value_heads, config.head_dim, arena_.stream,
                ctx.fuse_qknorm_rope_kv);
          });
      if (tiled_attention) {
        detail::EmitAttentionDispatch("prefill_tiled", "");
      } else if (!detail::BatchedAttentionScoresFitLds(visible_context)) {
        detail::EmitAttentionDispatch(
            arena_.d_kv_cache != nullptr ? "prefill_online_fp32"
                                         : "prefill_online_fp16",
            "prefill_baseline: score_rows_exceed_lds");
      } else {
        detail::EmitAttentionDispatch(
            "prefill_baseline",
            detail::ShouldAttemptOptimizedAttention(visible_context)
                ? "prefill_tiled: rejected"
                : "prefill_tiled: below_threshold");
      }
    }
  }

  if (half_prefill) {
    LaunchFloatToFp16(arena_.d_ssm_out + ctx.output_first_row * attention_size,
                      arena_.d_scratch_bf16, ctx.output_rows * attention_size,
                      arena_.stream);
  } else if (ReadsQ8ActivationForPrefill(layer.attn_output.type,
                                         half_prefill)) {
    LaunchQuantizeActivationQ8_1FromFp32(
        arena_.d_ssm_out, arena_.d_scratch_q8_act, batch_size,
        attention_size, arena_.stream);
  } else {
    LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                          batch_size * attention_size, arena_.stream);
    LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                 arena_.d_scratch_q8_act, batch_size,
                                 attention_size, arena_.stream);
  }
  if (half_prefill) {
    LaunchBatchedQuantGEMMResidualFp16(
        layer.attn_output.type, layer.attn_output.data,
        arena_.d_scratch_bf16, arena_.d_hidden + ctx.output_first_row * hidden_size,
        ctx.output_rows, hidden_size, attention_size, arena_.stream);
  } else {
    PrefillGemmWeight(layer.attn_output, arena_.d_scratch_bf16, arena_.d_ssm_out,
                      arena_.d_attn_out, batch_size, hidden_size, attention_size,
                      arena_.d_scratch_q8_act, half_prefill);
  }
}

void QwenGpuExecutor::PrefillSsmLayer(const PrefillChunkCtx& ctx,
                                       std::uint32_t l,
                                       QwenRecurrentStateIo recurrent_state) {
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t batch_size = ctx.batch_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  const auto& layer = weights_.layers[l];
  const bool half_prefill = ctx.half_prefill;

  PrefillLayerNorm(ctx, l);

  if (!half_prefill && !ctx.norm_feeds_q8_only) {
    LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                 arena_.d_scratch_q8_act, batch_size,
                                 hidden_size, arena_.stream);
  }
  PrefillGemmWeight(layer.attn_qkv, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_ssm_qkv, batch_size, ssm_qkv_size, hidden_size,
                    arena_.d_scratch_q8_act, half_prefill);
  PrefillGemmWeight(layer.attn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_ssm_gate, batch_size, ssm_inner_size, hidden_size,
                    arena_.d_scratch_q8_act, half_prefill);
  PrefillGemmWeight(layer.ssm_alpha, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_alpha_buf, batch_size, time_step_rank, hidden_size,
                    arena_.d_scratch_q8_act, half_prefill);
  PrefillGemmWeight(layer.ssm_beta, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_beta_buf, batch_size, time_step_rank, hidden_size,
                    arena_.d_scratch_q8_act, half_prefill);
  if (arena_.IsSsmReplayCaptureActive()) {
    LaunchCaptureBatchedSsmReplay(
        arena_.d_ssm_qkv, arena_.d_alpha_buf, arena_.d_beta_buf,
        arena_.GetSsmReplayCapture(), config.SsmLayerIndex(l), ctx.start_pos,
        batch_size, ssm_qkv_size, time_step_rank, arena_.stream);
  }

  const bool ssm_epilogue_q8 =
      ReadsQ8ActivationForPrefill(layer.ssm_out.type, half_prefill) &&
      IsFusedSSMEpilogueQuantizeQ8_1Supported(config.SsmValueSize(),
                                              ssm_inner_size);
  const bool ssm_row_split = detail::ShouldUseSsmRowSplitRecurrence(
      IsDeltaNetRowSplitSupported(config.ssm_state_size,
                                  config.SsmValueSize()),
      arena_.d_ssm_kq_scales != nullptr && arena_.d_ssm_alpha_beta != nullptr);
  if (ssm_row_split) {
    LaunchBatchedSSMConvRecurrenceRowSplit(
        arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
        recurrent_state, arena_.d_conv_out, arena_.d_alpha_buf,
        arena_.d_beta_buf, static_cast<const float*>(layer.ssm_a.data),
        static_cast<const float*>(layer.ssm_dt.data),
        static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
        arena_.d_ssm_out,
        ssm_epilogue_q8 ? arena_.d_scratch_q8_act : nullptr,
        arena_.d_ssm_kq_scales, arena_.d_ssm_alpha_beta,
        config.SsmLayerIndex(l), batch_size, ssm_qkv_size,
        config.ssm_group_count, config.ssm_time_step_rank,
        config.ssm_state_size, config.SsmValueSize(), arena_.stream,
        arena_.GetRecurrentStateStorage(),
        half_prefill ? arena_.d_scratch_bf16 : nullptr);
  } else {
    LaunchBatchedSSMConvRecurrence(
        arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
        recurrent_state, arena_.d_conv_out, arena_.d_alpha_buf,
        arena_.d_beta_buf, static_cast<const float*>(layer.ssm_a.data),
        static_cast<const float*>(layer.ssm_dt.data),
        static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
        arena_.d_ssm_out, config.SsmLayerIndex(l), batch_size, ssm_qkv_size,
        config.ssm_group_count, config.ssm_time_step_rank,
        config.ssm_state_size, config.SsmValueSize(), arena_.stream,
        arena_.GetRecurrentStateStorage());
  }

  if (half_prefill) {
    if (!ssm_row_split) {
      LaunchFloatToFp16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                        batch_size * ssm_inner_size, arena_.stream);
    }
  } else if (ssm_row_split && ssm_epilogue_q8) {
  } else if (ReadsQ8ActivationForPrefill(layer.ssm_out.type, half_prefill)) {
    LaunchQuantizeActivationQ8_1FromFp32(
        arena_.d_ssm_out, arena_.d_scratch_q8_act, batch_size,
        ssm_inner_size, arena_.stream);
  } else {
    LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                          batch_size * ssm_inner_size, arena_.stream);
    LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                 arena_.d_scratch_q8_act, batch_size,
                                 ssm_inner_size, arena_.stream);
  }
  if (half_prefill) {
    LaunchBatchedQuantGEMMResidualFp16(
        layer.ssm_out.type, layer.ssm_out.data, arena_.d_scratch_bf16,
        arena_.d_hidden, batch_size, hidden_size, ssm_inner_size,
        arena_.stream);
  } else {
    PrefillGemmWeight(layer.ssm_out, arena_.d_scratch_bf16, arena_.d_ssm_out,
                      arena_.d_attn_out, batch_size, hidden_size, ssm_inner_size,
                      arena_.d_scratch_q8_act, half_prefill);
  }
}

void QwenGpuExecutor::PrefillFfnTail(const PrefillChunkCtx& ctx,
                                      std::uint32_t l) {
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t batch_size = ctx.batch_size;
  const float eps = 1e-6F;
  const auto& layer = weights_.layers[l];
  const bool half_prefill = ctx.half_prefill;

  float* output_hidden =
      arena_.d_hidden + ctx.output_first_row * hidden_size;

  if (half_prefill) {
    LaunchBatchedRMSNormFp16(output_hidden, nullptr,
                             static_cast<const float*>(layer.ffn_norm.data),
                             nullptr, arena_.d_scratch_bf16, ctx.output_rows,
                             hidden_size, eps, arena_.stream);
  } else if (ctx.ffn_feeds_q8_only) {
    LaunchBatchedFusedRMSNormQuantizeQ8_1(
        arena_.d_hidden, arena_.d_attn_out,
        static_cast<const float*>(layer.ffn_norm.data), arena_.d_hidden,
        arena_.d_scratch_q8_act, batch_size, hidden_size, eps, arena_.stream);
  } else {
    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                             arena_.d_hidden, batch_size, hidden_size,
                             arena_.stream);
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.ffn_norm.data),
                         arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                         hidden_size, eps, arena_.stream);
  }

  const void* ffn_down_q8_act = arena_.d_scratch_q8_act;
  const void* ffn_down_input = arena_.d_scratch_bf16;

  {
    if (!half_prefill && !ctx.ffn_feeds_q8_only) {
      LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                   arena_.d_scratch_q8_act, batch_size,
                                   hidden_size, arena_.stream);
    }
    const bool fused_dual_swiglu =
        ReadsQ8ActivationForPrefill(layer.ffn_gate.type, half_prefill) &&
        ReadsQ8ActivationForPrefill(layer.ffn_up.type, half_prefill) &&
        layer.ffn_gate.type == layer.ffn_up.type &&
        ReadsQ8ActivationForPrefill(layer.ffn_down.type, half_prefill) &&
        IsFusedSwiGluGemmEpilogueSupported(
            layer.ffn_gate.type, batch_size, intermediate_size,
            batch_size * intermediate_size * sizeof(float));
    if (half_prefill) {
      const bool paired = TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
          layer.ffn_gate.type, layer.ffn_up.type, layer.ffn_gate.data,
          layer.ffn_up.data, arena_.d_scratch_bf16, arena_.d_ffn_up,
          ctx.output_rows, intermediate_size, hidden_size, arena_.stream);
      if (!paired) {
        LaunchBatchedQuantGEMMFp16(layer.ffn_gate.type, layer.ffn_gate.data,
                                   arena_.d_scratch_bf16, arena_.d_ffn_gate,
                                   ctx.output_rows, intermediate_size,
                                   hidden_size, arena_.stream);
        LaunchBatchedQuantGEMMSwiGLUFp16(
            layer.ffn_up.type, layer.ffn_up.data, arena_.d_scratch_bf16,
            arena_.d_ffn_gate, arena_.d_ffn_up, ctx.output_rows,
            intermediate_size, hidden_size, arena_.stream);
      }
      ffn_down_input = arena_.d_ffn_up;
    } else if (fused_dual_swiglu) {
      LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1(
          layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data,
          arena_.d_scratch_q8_act, arena_.d_ffn_gate, arena_.d_ffn_up,
          batch_size, intermediate_size, hidden_size, arena_.stream);
      ffn_down_q8_act = arena_.d_ffn_up;
    } else if (ReadsQ8ActivationForPrefill(layer.ffn_gate.type, half_prefill) &&
               ReadsQ8ActivationForPrefill(layer.ffn_up.type, half_prefill) &&
               layer.ffn_gate.type == layer.ffn_up.type) {
      LaunchBatchedDualQuantGEMMPreQuantized(
          layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data,
          arena_.d_scratch_q8_act, arena_.d_ffn_gate, arena_.d_ffn_up,
          batch_size, intermediate_size, hidden_size, arena_.stream);
    } else {
      PrefillGemmWeight(layer.ffn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                        arena_.d_ffn_gate, batch_size, intermediate_size, hidden_size,
                        arena_.d_scratch_q8_act, half_prefill);
      PrefillGemmWeight(layer.ffn_up, arena_.d_scratch_bf16, arena_.d_normed,
                        arena_.d_ffn_up, batch_size, intermediate_size, hidden_size,
                        arena_.d_scratch_q8_act, half_prefill);
    }

    if (half_prefill || fused_dual_swiglu) {
    } else if (ReadsQ8ActivationForPrefill(layer.ffn_down.type, half_prefill)) {
      LaunchBatchedFusedSwiGLUQuantizeQ8_1(
          arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_scratch_q8_act,
          batch_size, intermediate_size, arena_.stream);
    } else {
      LaunchBatchedSwiGLUActivation(arena_.d_ffn_gate, arena_.d_ffn_up,
                                    arena_.d_ffn_act, arena_.d_scratch_bf16,
                                    batch_size * intermediate_size,
                                    arena_.stream);
      LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                   arena_.d_scratch_q8_act, batch_size,
                                   intermediate_size, arena_.stream);
    }
  }

  if (half_prefill) {
    LaunchBatchedQuantGEMMResidualFp16(
        layer.ffn_down.type, layer.ffn_down.data, ffn_down_input,
        output_hidden, ctx.output_rows, hidden_size, intermediate_size,
        arena_.stream);
  } else {
    PrefillGemmWeight(layer.ffn_down, ffn_down_input, arena_.d_ffn_act,
                      arena_.d_ffn_out, batch_size, hidden_size, intermediate_size,
                      ffn_down_q8_act, half_prefill);
    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out,
                             arena_.d_hidden, batch_size, hidden_size,
                             arena_.stream);
  }

  if (capture_prompt_hidden_) {
    if (const auto tap_index = arena_.GetTargetLayerCaptureIndex(l);
        tap_index.has_value()) {
      float* destination = h_prompt_hidden_.Data() + ctx.prompt_capture_offset +
                           (*tap_index * hidden_size);
      HIP_CHECK(hipMemcpy2DAsync(
          destination, ctx.target_layer_count * hidden_size * sizeof(float),
          ctx.decode_hidden, hidden_size * sizeof(float),
          hidden_size * sizeof(float), batch_size, hipMemcpyDeviceToHost,
          arena_.stream));
      if (cancellation_check_) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
      }
    }
  }
}

std::vector<tokenization::TokenId> QwenGpuExecutor::ForwardPromptBatchMulti(
    std::span<const QwenPromptBatchItem> items) {
  namespace mb = ::gufo::core::compute::mixed_batch;
  constexpr std::size_t kMaxPromptBatchMulti = 8;
  // Opt-out valve (default on): GUFO_27B_UNIFIED_BATCH=0 keeps the legacy
  // per-session attention dispatch inside the batch envelope.
  const bool use_unified = [] {
    const char* e = std::getenv("GUFO_27B_UNIFIED_BATCH");
    return e == nullptr || e[0] != '0';
  }();
  if (items.size() < 2 || items.size() > kMaxPromptBatchMulti) {
    throw std::invalid_argument(
        "Qwen prompt batch needs two to eight sessions");
  }
  QwenGpuExecutor* coordinator = items.front().executor;
  if (coordinator == nullptr) {
    throw std::invalid_argument("Qwen prompt batch contains a null session");
  }
  // Blocked-Q8 prefill keeps the F16 plane canonical for the legacy single
  // path; the unified batch reads the half plane, so refuse it loudly like
  // the decode batch does (batched decode under q8 is unsupported there).
  if (coordinator->policy_.UsesBlockedAttentionKv()) {
    throw std::logic_error(
        "blocked-Q8 attention does not support batched prefill; use "
        "QWEN_KV_QUANT=f16");
  }
  if (!coordinator->policy_.UsesFp16AttentionKv()) {
    throw std::logic_error(
        "batched prefill needs the F16 attention KV plane; use "
        "QWEN_KV_QUANT=f16");
  }
  const auto* shared_model = coordinator->model_.get();
  const std::uint64_t shared_policy = coordinator->policy_.Fingerprint();
  std::size_t total_rows = 0;
  for (std::size_t index = 0; index < items.size(); ++index) {
    QwenGpuExecutor* executor = items[index].executor;
    if (executor == nullptr || executor->model_.get() != shared_model ||
        executor->policy_.Fingerprint() != shared_policy) {
      throw std::invalid_argument(
          "Qwen prompt batch sessions are not execution-compatible");
    }
    if (items[index].tokens.empty() ||
        items[index].tokens.size() > executor->arena_.GetMaxBatch()) {
      throw std::length_error("Qwen prompt batch chunk exceeds GPU batch");
    }
    if (static_cast<std::size_t>(items[index].start_pos) +
            items[index].tokens.size() >
        executor->arena_.GetMaxContext()) {
      throw std::length_error("prompt exceeds the GPU context length");
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (items[previous].executor == executor) {
        throw std::invalid_argument(
            "Qwen prompt batch contains a duplicate session");
      }
    }
    total_rows += items[index].tokens.size();
  }
  if (total_rows > coordinator->arena_.GetMaxBatch()) {
    throw std::length_error(
        "Qwen prompt batch width exceeds the coordinator arena capacity");
  }
  const auto& config = coordinator->weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const float eps = 1e-6F;

  struct SessionBatchState {
    QwenGpuExecutor* executor;
    std::span<const tokenization::TokenId> tokens;
    std::uint32_t start_pos;
    bool compute_logits;
    QwenGpuExecutor::PrefillChunkCtx ctx;
    QwenRecurrentStateIo recurrent;
    // Owns the host token staging until the batch drains: the prompt H2D is
    // queued async on the session stream (single keeps its host_tokens alive
    // to end-of-function the same way).
    std::vector<std::uint32_t> host_tokens;
  };
  std::vector<SessionBatchState> states;
  states.reserve(items.size());
  std::array<models::qwen::ContinuationWrite, kMaxPromptBatchMulti> writes;
  for (const auto& item : items) {
    QwenGpuExecutor* executor = item.executor;
    executor->CheckReset();
    executor->CheckPrefillCancellation();
    executor->replaying_ssm_state_ = false;
    executor->arena_.DisableSsmReplayCapture();
    const std::size_t batch_size = item.tokens.size();
    if (executor->capture_prompt_hidden_) {
      const std::size_t captured_layers = std::max<std::size_t>(
          executor->arena_.GetTargetLayerCapture().size(), 1U);
      const std::size_t row = captured_layers * hidden_size;
      executor->h_prompt_hidden_.Reserve(
          executor->h_prompt_hidden_.Size() + batch_size * row,
          std::size_t{executor->GetMaxPromptBatch()} * row);
    }
    writes[states.size()].Begin(executor->continuation_hooks_, item.start_pos,
                                item.start_pos +
                                    static_cast<std::uint32_t>(batch_size));
    auto scratch = executor->arena_.GetScratchView(batch_size);
    SessionBatchState st;
    st.executor = executor;
    st.tokens = item.tokens;
    st.start_pos = item.start_pos;
    st.compute_logits = item.compute_logits;
    st.host_tokens.assign(item.tokens.begin(), item.tokens.end());
    HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                             st.host_tokens.data(),
                             batch_size * sizeof(std::uint32_t),
                             hipMemcpyHostToDevice, executor->arena_.stream));
    LaunchBatchedEmbeddingLookup(
        executor->weights_.token_embd.data, executor->weights_.token_embd.type,
        scratch.decode.prompt_tokens.data(), scratch.decode.hidden.data(),
        batch_size, hidden_size, executor->arena_.stream);
    executor->vision_input_.Inject(scratch.decode.hidden.data(),
                                   item.start_pos, batch_size, hidden_size, 1,
                                   executor->arena_.stream);

    QwenGpuExecutor::PrefillChunkCtx ctx;
    ctx.batch_size = batch_size;
    ctx.start_pos = item.start_pos;
    ctx.half_prefill =
        UseQwen27bFp16Prefill(executor->weights_, batch_size);
    const auto attention_workspace =
        executor->arena_.GetScratchView(executor->arena_.GetMaxBatch()).ffn;
    ctx.ws_gate = attention_workspace.gate;
    ctx.ws_up = attention_workspace.up;
    ctx.prompt_capture_offset = executor->h_prompt_hidden_.Size();
    ctx.target_layer_count =
        executor->arena_.GetTargetLayerCapture().size();
    ctx.decode_hidden = scratch.decode.hidden.data();
    st.ctx = ctx;
    st.recurrent = executor->arena_.BeginRecurrentWrite();
    states.push_back(std::move(st));
    if (executor->capture_prompt_hidden_ &&
        ctx.target_layer_count > 0) {
      executor->h_prompt_hidden_.Resize(
          ctx.prompt_capture_offset + batch_size * ctx.target_layer_count *
                                         hidden_size);
    }
  }

  // Coordinator takes stream ownership after the other sessions' queued work
  // finishes (mirrors ForwardTokenBatch).
  for (const auto& st : states) {
    if (st.executor != coordinator) {
      HIP_CHECK(hipStreamSynchronize(st.executor->arena_.stream));
    }
  }
  // Unified F16 scratch: device descriptor table only (no split-K partials
  // on the prefill path). Freed below after the final synchronize.
  void* mixed_scratch = nullptr;
  {
    const std::size_t need =
        mb::MixedBatchF16ScratchBytes(states.size());
    if (hipMalloc(&mixed_scratch, need) != hipSuccess) {
      throw std::runtime_error("Qwen prompt batch scratch allocation failed");
    }
  }
  mb::MixedBlk ublks[kMaxPromptBatchMulti];

  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    for (auto& st : states) {
      st.executor->PrefillLayerCheckpoint(l);
    }
    const auto& layer = coordinator->weights_.layers[l];
    const auto route_resolution = ResolveQwenLayerRouteWithReasons(
        coordinator->policy_, QwenExecutionMode::kPrefill,
        layer.is_full_attention);
    for (auto& st : states) {
      QwenGpuExecutor* executor = st.executor;
      // Per-session GEMM/quantization route flags (batch-size dependent, so
      // each session keeps the route its single run would take).
      const auto is_q8 = [&](const models::QwenTensorRef& w) {
        return ReadsQ8ActivationForPrefill(w.type, st.ctx.half_prefill);
      };
      st.ctx.norm_feeds_q8_only =
          IsFusedRMSNormQuantizeQ8_1Supported(hidden_size) &&
          (layer.is_full_attention
               ? (is_q8(layer.attn_q) && is_q8(layer.attn_k) &&
                  is_q8(layer.attn_v))
               : (is_q8(layer.attn_qkv) && is_q8(layer.attn_gate) &&
                  is_q8(layer.ssm_alpha) && is_q8(layer.ssm_beta)));
      st.ctx.ffn_feeds_q8_only =
          IsFusedRMSNormQuantizeQ8_1Supported(hidden_size) &&
          is_q8(layer.ffn_gate) && is_q8(layer.ffn_up);
      st.ctx.output_first_row =
          (st.ctx.half_prefill && l + 1 == config.num_layers) ? 0 : 0;
      st.ctx.output_rows = st.ctx.batch_size;
      st.ctx.fuse_qknorm_rope_kv =
          route_resolution.plan.fuse_qk_norm_rope_kv &&
          detail::IsFusedQkNormSupported(config.head_dim);
      if (layer.is_full_attention) {
        executor->PrefillAttentionPre(st.ctx, l);
      }
    }
    // Unified attention needs the KV cache already written by Pre: only
    // the fused QK-norm/RoPE/KV-write path maintains the plane before the
    // attention step. Unfused sessions keep the legacy dispatch (which packs
    // KV itself), exactly like their single runs.
    bool all_fused = true;
    for (const auto& st : states) {
      if (!st.ctx.fuse_qknorm_rope_kv) all_fused = false;
    }
    if (use_unified && layer.is_full_attention && all_fused) {
      // One unified launch over every session's rows (varlen: decode rows
      // and prefill chunks side by side).
      for (std::size_t i = 0; i < states.size(); ++i) {
        const auto& st = states[i];
        QwenGpuExecutor* executor = st.executor;
        const std::size_t st_total_k =
            executor->arena_.GetAttentionKvPlaneElements();
        const std::uint32_t st_maxctx = executor->arena_.GetMaxContext();
        const std::uint32_t attn_layer_idx =
            l / config.full_attention_interval;
        const std::size_t layer_el =
            static_cast<std::size_t>(config.num_key_value_heads) * st_maxctx *
            config.head_dim;
        const auto* f16_base =
            static_cast<const __half*>(executor->arena_.d_attention_kv_f16);
        mb::MixedBlk& b = ublks[i];
        b.q = executor->arena_.d_q;
        b.k = f16_base +
              static_cast<std::size_t>(attn_layer_idx) * layer_el;
        b.v = f16_base + st_total_k +
              static_cast<std::size_t>(attn_layer_idx) * layer_el;
        b.gate = executor->arena_.d_ssm_gate;
        b.out = executor->arena_.d_ssm_out;
        b.kv_len = st.start_pos + static_cast<std::uint32_t>(st.ctx.batch_size);
        b.q_pos_base = st.start_pos;
        b.n_rows = static_cast<std::uint32_t>(st.ctx.batch_size);
        b.k_codec = mb::MixedKvCodec::kF16;
        b.v_codec = mb::MixedKvCodec::kF16;
        b.head_dim = config.head_dim;
        b.heads = config.num_attention_heads;
        b.kv_heads = config.num_key_value_heads;
        b.score_scale = 0.0F;
        b.q_stride = 0;
        b.k_stride = config.head_dim;
        b.v_stride = config.head_dim;
        b.k_head_stride = st_maxctx * config.head_dim;
        b.v_head_stride = st_maxctx * config.head_dim;
        b.gate_stride = 0;
        b.out_stride = 0;
        b.window = 0;
        b.key_block_bytes = 34;
        b.value_block_bytes = 34;
      }
      // Pre-phase kernels ran on each session's stream; drain them before
      // the coordinator-stream unified launch (two host syncs per layer,
      // mirroring the decode batch's ownership transfer).
      for (const auto& st : states) {
        if (st.executor != coordinator) {
          HIP_CHECK(hipStreamSynchronize(st.executor->arena_.stream));
        }
      }
      const bool unified_ok = mb::LaunchMixedBatchAttention(
          ublks, states.size(), mixed_scratch,
          mb::MixedBatchF16ScratchBytes(states.size()),
          coordinator->arena_.stream);
      HIP_CHECK(hipStreamSynchronize(coordinator->arena_.stream));
      for (auto& st : states) {
        st.executor->PrefillAttentionPost(st.ctx, l,
                                          /*run_legacy=*/!unified_ok);
      }
    } else if (layer.is_full_attention) {
      for (auto& st : states) {
        st.executor->PrefillAttentionPost(st.ctx, l,
                                          /*run_legacy=*/true);
      }
    } else {
      for (auto& st : states) {
        st.executor->PrefillSsmLayer(st.ctx, l, st.recurrent);
      }
    }
    for (auto& st : states) {
      st.executor->PrefillFfnTail(st.ctx, l);
    }
  }

  std::vector<tokenization::TokenId> next_tokens;
  next_tokens.reserve(states.size());
  for (std::size_t si = 0; si < states.size(); ++si) {
    auto& st = states[si];
    QwenGpuExecutor* executor = st.executor;
    const std::size_t batch_size = st.ctx.batch_size;
    executor->last_hidden_offset_ = (batch_size - 1) * hidden_size;
    if (executor->capture_prompt_hidden_) {
      if (st.ctx.target_layer_count == 0) {
        const std::size_t old_size = executor->h_prompt_hidden_.Size();
        const std::size_t chunk_elements = batch_size * hidden_size;
        executor->h_prompt_hidden_.Resize(old_size + chunk_elements);
        HIP_CHECK(hipMemcpyAsync(executor->h_prompt_hidden_.Data() + old_size,
                                st.ctx.decode_hidden,
                                chunk_elements * sizeof(float),
                                hipMemcpyDeviceToHost, executor->arena_.stream));
      }
      HIP_CHECK(hipStreamSynchronize(executor->arena_.stream));
    }
    if (!st.compute_logits) {
      writes[si].Commit(
          st.start_pos + static_cast<std::uint32_t>(batch_size), false);
      next_tokens.push_back(0);
      continue;
    }
    auto scratch = executor->arena_.GetScratchView(batch_size);
    const float* final_hidden =
        scratch.decode.hidden.data() + ((batch_size - 1) * hidden_size);
    LaunchRMSNorm(final_hidden,
                  static_cast<const float*>(
                      executor->weights_.output_norm.data),
                  scratch.decode.normed.data(), hidden_size, eps,
                  executor->arena_.stream);
    LaunchGEMV(executor->weights_.output.data,
               executor->weights_.output.type, scratch.decode.normed.data(),
               scratch.decode.logits.data(), config.vocab_size, hidden_size,
               executor->arena_.stream, executor->arena_.d_preq_q81,
               executor->arena_.preq_q81_bytes);
    auto* d_out_token = scratch.decode.sampled_token.data();
    LaunchGPUArgmax(scratch.decode.logits.data(), d_out_token,
                    config.vocab_size, executor->arena_.stream);
    std::uint32_t next_token_id = 0;
    HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                             hipMemcpyDeviceToHost, executor->arena_.stream));
    HIP_CHECK(hipStreamSynchronize(executor->arena_.stream));
    const auto result =
        CheckedSampleToken(next_token_id, config.vocab_size);
    writes[si].Commit(st.start_pos + static_cast<std::uint32_t>(batch_size),
                      true);
    next_tokens.push_back(result);
  }
  HIP_CHECK(hipStreamSynchronize(coordinator->arena_.stream));
  (void)hipFree(mixed_scratch);
  return next_tokens;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
