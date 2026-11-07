# GPT-OSS (gpt-oss MoE)

GPT-OSS-20B GGUF runtime (`gpt-oss-20b-Q4_K_M.gguf`, ~12 GB, single
file). Same architecture and code path as the 120B target; only the weight
file changes when 120B weights land.

## Measured architecture (from GGUF metadata, not guessed)

| Key | Value |
|---|---|
| `general.architecture` | `gpt-oss` |
| layers / hidden | 24 / 2880 |
| heads / kv-heads / head_dim | 64 / 8 / 64 |
| experts / top-k / expert_ff | 32 / 4 / 2880 |
| vocab | 201088 |
| sliding window | 128, even layers only (llama.cpp SWA period 2, dense-first=false) |
| rope | NeoX, theta 150000, YaRN factor 32, orig ctx 4096, ctx 131072 |
| norms / eps | RMSNorm everywhere, 1e-5 |
| extras | per-head attention sinks, Q/K/V + wo + router + expert biases |

Tensor types in the Q4_K_M file: dense Q5_0/Q4_K/Q8_0/F32, experts MXFP4,
all biases/norms/sinks F32. Q/K/V biases are applied right after projection,
before RoPE (matching llama.cpp `build_qkv`, which adds `wq_b`/`wk_b`/`wv_b`
when the artifact carries them). `output.weight` is standalone Q8_0
(not tied).

## Runtime mapping

| Item | Shared layer | Contract |
|---|---|---|
| dense GEMM decode | narrow prequant (`DecideNarrowDecodePrequantRoute` + `LaunchNarrowQ80Prequant`); fused QKV (`LaunchNarrowQ80QkvDecode`, one prequant + one launch) and fused wo row-add (`LaunchNarrowQ80PrequantRowAdd`, GEMV+bias+residual in one launch) | Q8_0 (converted), K % 32 == 0, 1..8 tokens; falls back to `GemvBatched` / the three-launch sequence |
| dense GEMM prefill | `DenseF16Gemm` | Q8_0 (converted), F16 activations |
| MoE routing | `RouterTopK` (shared) + `SmallGemm` F32 projection | softmax over SELECTED top-k (= llama SOFTMAX_WEIGHT); verified against `llama-graph.cpp` |
| MoE GEMM decode | `MoeDecodeGateUpActMxfp4` + `MoeDecodeDownEpiMxfp4` (fused core decode pair: gate+up+act, then down+bias+epilogue; two launches replace five) | native MXFP4 experts, merged gate_up slab, slots <= 64; routed/vec sequence is the contract fallback (`GPTOSS_NO_FUSED*` forces legacy for A/B) |
| MoE GEMM prefill | compact routed GEMM (`RoutedCompact` + `RoutedF16Gemm` over native MXFP4 slabs, model-local `MoeCompactSplitBiasAct`) | grouped buckets (16-row padded), `GPTOSS_MOE_TILE` tunable; vec path is the fallback |
| KV store | `blocked_kv::QuantizeKvRows` | post-RoPE rows, per-32-block f16 scale, generic over d |
| KV read (q8) | `blocked_kv::DequantizeKvRows` into F16 staging | generic over d |
| attention | model-local dense F16 kernel | sinks + sliding window have no shared kernel |
| dispatcher | consulted (env-gated diag) | d == 64 has no blocked-read kernel, so every call is `kLegacy` by contract; the dequant-then-exact-kernel fallback above is exact |

Host-side load conversions (all standard GGML layouts): Q5_0/Q4_K dense to
Q8_0 (near-lossless, 1-ulp scale noise). Experts stay native MXFP4
(zero conversion loss — a Q4_0 round-trip crushes small values in
outlier blocks and flips routing; measured, not assumed). Biases are
applied at runtime, never folded into quants.

## Legacy fallbacks (shapes that miss the fast path, and why)

- Router GEMM (F32, 32 x 2880): `GemvBatched` legacy wave32 — F32 has no
  narrow contract (narrow is Q8_0-only by design). Correct, just not fused.
- Any Q8_0 GEMM with K % 32 != 0, 0 rows, or prequant-scratch short:
  `GemvBatched` legacy. Does not occur on gpt-oss shapes (all K % 32 == 0).
- Narrow exclusions (`n==1 && rows<=64 && k%5120==0`, `rows==320 &&
  k==10240`): do not occur on gpt-oss shapes (k = 2880/4096/512).
- Attention: always `kLegacy` (d == 64 has no blocked-read kernel);
  the q8 path dequantizes the shared blocked store into F16 staging and
  runs the exact model-local sink/window kernel. No precision is lost
  beyond the q8 store itself.
- MoE decode/prefill: vec path (`MoeVecGateUpMxfp4`/`GemvQSlotMxfp4`),
  not the grouped/compacted path (no bucket setup needed at top-4).

## Anchor grades vs llama.cpp (same Q4_K_M file, raw prompt, greedy)

| Prompt | gufo f16 | llama.cpp | Grade |
|---|---|---|---|
| "Hello world, this is a test." (8 tok) | 1328:13.67 | 1328:13.05 | A (top-1+top-2 match, <1.0) |
| "The quick brown fox..." (10 tok) | 623:13.73 | 623:13.81 | A (top-1 match, 0.07) |
| + two more sentences (20 tok) | 623:14.81 | 623:13.12 | B+ (top-1 match, 1.7) |
| 26-30 tok / single tokens / trailing-0 prompts | top-1 flips, top-5 partly overlaps | — | C (knife-edge routing flips from fp/quant noise amplified by outlier experts) |

Grade B/C flips are inherent sensitivity (verified: kernels match a
numpy reference layer-by-layer on identical weights; single e=134
outlier blocks and knife-edge top-k gaps amplify tiny differences).
The implementation is deterministic (bit-identical across runs).

## Core changes (this branch)

- `src/core/gguf_reader.hpp`: `GgmlType::kMXFP4 = 39` + name.
- `src/core/quant/ggml_dequant.{hpp,cpp}`: `block_mxfp4` (17 B), row
  support. Required just to *open* the file (`GgufReader` rejects
  unknown types).
- `src/core/compute/gemm/moe_fused.hip` + `gemm.hpp`:
  `MoeVecGateUpMxfp4` / `GemvQSlotMxfp4` (native MXFP4 vec MoE,
  E8M0 scale + doubled-E2M1 codebook matching llama.cpp bit-for-bit).

## References

- llama.cpp `src/models/openai-moe.cpp` (graph math), `ggml-quants.c`
  (Q4_0/Q8_0/Q5_0/MXFP4 layouts), `llama-graph.cpp` `build_moe_ffn`
  (SOFTMAX_WEIGHT gating = top-k on raw logits, softmax over top-k;
  `swiglu_oai` alpha 1.702 / limit 7 with the +1 offset on up).
- Q5_0 high-half bit placement matches llama.cpp inference (`qh` bit
  `j + 12`, CPU + CUDA + vecdot agree); the converter copies it verbatim.
- MXFP4 verified against llama.cpp (`kvalues_mxfp4`, nibble order,
  `GGML_E8M0_TO_FP32_HALF`) + E8M0 scale census (no NaN/Inf encodings).
- Anchor: `tests/models/gptoss/registry_pilot_test` (direct / registry /
  DecodeStep token-exact + graded prefill-chunk parity + q8 smoke).
