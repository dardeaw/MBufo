# Inkling-2-Mini benchmarks

Scope: single instance (`-c 4096 --sessions 1`), Strix Halo gfx1151,
build `inkling-engine` (Q8 dense path). Dates are 2026-11-24 unless noted.
`MBUFO_INKLING_PARITY` off unless stated.

## Throughput

| Test | Prompt | Decode | Result |
| --- | --- | --- | --- |
| Prefill (PP) | ~1176 tokens | 1 token | 166 tok/s (TTFT 7.1s) |
| Prefill (PP) | ~2338 tokens | 1 token | 134 tok/s (TTFT 17.5s) |
| Decode (TG) | 18 tokens | 89 tokens | 13.3 tok/s |

Notes:

- Chunked prefill (`kChunk=128`) verified up to 2338 tokens after fixing an
  F16-staging overrun (`x16` now covers the loop-gate `2*hidden` GEMM input;
  previously any prompt over ~128 tokens crashed the server with a GPU
  page fault).
- llama.cpp (local `llama-inkling` build) cannot load this file
  (`check_tensor_dims: tensor 'blk.24.attn_q.weight' not found`; the file has
  blk.0-23, the loader expects 24+). No head-to-head PP/TG race was possible.
  Relative expectation stands: PP wins big (fused Q8 GEMMs vs F32), TG not
  worse. Revisit when a mutually loadable file exists.

## Parity (vs transformers CPU reference, `Say OK.` prompt, 18 tokens)

`MBUFO_INKLING_PARITY=1`, bar rel-mean < 5% on layer outputs.

| Stage | rel-mean | Verdict |
| --- | --- | --- |
| L0-L3 | 0.003-0.017 | pass |
| L4 | 0.054 | documented noise flip (single router flip, §Quality) |
| L5 | 0.020 | pass (after delta-input + tap-order fixes) |
| loop pass 1 (L6-17) | <= 0.055 | pass (L11 borderline, noise family) |
| loop pass 2 | L06-11, L14-17 pass; L12 0.24, L13 0.07 | L12 boundary amplification, under review |
| loop pass 3 | L06-10, L13-17 pass; L11 0.057, L12 0.21 | same family as pass 2 |
| layers 18-22 | pass (L22 0.068 borderline) | - |
| layer 23 + final | see Quality (reference `hs48` is stale in transformers) | engine matches true forward output to ~5% |
| final argmax | id 200001, logit 27.28 vs 27.3 ref | pass |

## TODO

- [ ] L4 / boundary-layer router flips: reduce sensitivity or document as noise floor.
- [ ] Re-run PP/TG against llama once a mutually loadable file exists.
- [ ] Long-context (8K+) prefill/decode throughput curve.
