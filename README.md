# MBufo

A [gufo](https://github.com/gufo-org/gufo) fork for AMD Strix Halo, built as a
**fair, low-cost inference engine** — fairness from scheduling that never lets
queues cut in line, low cost from low-precision compute.

> [繁體中文版本](README.zh-Hant.md)

Upstream gufo owns single-stream peak performance. MBufo asks the serving
question instead: what does one Strix Halo node do with sixteen requests at
once, and how much context can each of them hold? Three pillars answer it —
fair scheduling under load (§1), low-precision compute that spends less (§2),
and GGUF coverage that serves many models on one port (§3). The engine, kernels
and the bulk of the code are upstream's; thanks to gufo-org and all
contributors.

**Contents:** [Three numbers](#mbufo-in-three-numbers) ·
[§1 High concurrency](#1-high-concurrency-architecture) ·
[§2 Low-precision compute](#2-low-precision-compute-cost) ·
[§3 GGUF coverage](#3-gguf-coverage-compatibility) ·
[Quickstart](#quickstart) ·
[Benchmark](#benchmark-mbufo-v0255-gfx1151-strix-halo) ·
[Upstream tracking](#upstream-tracking) ·
[Build](#build-from-source) · [Docs](#documentation-index) ·
[Limits](#compatibility-and-limits)

## MBufo in three numbers

| # | Number | What it means |
| --- | --- | --- |
| 1 | **short-request TTFT 0.7 s** | A short request stuck behind a 250K-token prefill used to wait 3.4 s; with the fairness scheduler it waits 0.7 s (fairness-A 160 → 7.32, fairness-B 856 → 191, ITL unchanged, greedy byte-identical; v0.25.3). |
| 2 | **KV memory cut in half** | The blocked-Q8 cache holds a 262144-token session in 3.42 GB instead of 6.44 GB (K4V4: 1.81 GB), with the quality cost measured, not assumed (v0.25.2). |
| 3 | **11 GGUF families** | One HTTP port serves all of them — 8 newly usable in v0.28.0 on top of Flash-Next, 27B and Gemma 4 (see [§3 model table](#supported-models-v0280)). |

## 1. High concurrency (architecture)

Concurrency is not one number. Two properties decide whether a loaded server is
actually usable, and MBufo is built around both:

- **ITL (inter-token latency)** — once a request is streaming, the gap between
  its tokens. A server that batches efficiently but freezes a stream for a whole
  prefill step is not serving, it is queueing with extra steps.
- **Fairness** — no session, client or prompt may starve another. A 250K-token
  prompt arriving mid-conversation must not buy itself a bigger share of the
  device than sixteen short chats.

Upstream's scheduler already has the right shape: a prefill chunk is bounded
(`--prefill-chunk`), a due decoder is served before more prefill, and admission
is capped per client. What it lacks is the device time to honour all three
cheaply. A bounded chunk still costs a full tick in which **no** decoder
produces a token, because the chunk and the decodes are two separate passes over
the weights. That tick is exactly where ITL and fairness leak: every cycle a
streaming request pays a prefill step it does not benefit from.

MBufo merges them — the chunk rides the same `ForwardBatch` and the same
attention kernel as the decode rows — so a prefill tick also emits every token
that is due. The fairness policy is not bypassed by the merge; it is what makes
the merge safe.

### The unified mixed-batch path

One `ForwardBatch` carries decode rows and prefill chunks together through one
attention kernel instead of two separate weight passes. Since v0.27.0 the
decode/prefill routing runs through a single shared dispatcher — a unified
varlen mixed-batch attention interface over shared blocked-KV codecs, with a
WMMA prefill reader and a Split-K decode reader. The 27B attention layers,
Flash-Next decode attention and Gemma 4 SWA layers all resolve through it;
F16 paths are untouched, and rewiring kept F16 output byte-identical.

| Measurement | Result |
| --- | --- |
| Token gap for a streaming request, 5-row batch | 121 ms as two passes → **57 ms** merged (**2.1x**): the gap is now one weight pass, not two |
| 16-way sustained serving, 3 rounds | **15.3 s mean vs 17.0 s** with the merge off (rounds 14/17/15 vs 15/16/20), tail latency observed to improve |
| 4-way mixed-prompt A/B | **5.3 s vs 6.3 s**, the merge won every round, outputs exact |
| 16 concurrent requests | 99% GPU utilization, all outputs correct, zero errors |
| Batch envelope | 32 sessions / 1024 rows (upstream: 8 / 8 — three hard caps fixed: GDN, MoE, logits) |

One caveat, stated up front: at 66 rows a merged step costs 316 ms against
247 ms for the two separate passes, because batch-MoE falls to the memory-bound
vec path there. The end-to-end numbers above still win at serving concurrency —
one weight pass per cycle instead of two — but the large-batch FLOP side is
tracked separately.

Mixed batching changes when work runs, never what the model computes. It is on
by default; `GUFO_MIXED_BATCH=0` restores upstream scheduling exactly, tick-tock
included. Design invariants, the validation matrix and per-step measurements
live in [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md).

### DecodeBatch + Split-K decode

The serve layer steps decodes as prepared batches
(`StepDecodeBatch` / `ExecutePreparedDecodeBatch` /
`TextModelRunner::DecodeBatch` in `src/cli/serve/`), and the attention work
inside them reads through the shared Split-K decode reader (`blocked_splitk`,
v0.26.1) instead of the scalar path. Concretely: Gemma 4 SWA layers on shared
Q8 decode 6.4% under the pre-move baseline (was 8.0% on the scalar reader),
27B attention layers decode at −1.0% with direct read, and F16 output stays
byte-identical to the pre-move binary.

### Fairness: no queue-jumping (SRTF + starvation preemption + near-done gate)

Long prefills used to squeeze short-request TTFT (the 5-into-4
anti-starvation scenario). Three mechanisms fix it, keeping greedy output
byte-identical throughout (sha comparison passes: 24K prefill alone vs
concurrent produce the same sha):

- **SRTF prefill selection** — a near-done prefill with ≤1 chunk remaining is
  scheduled ahead of long-prefill continuation chunks (FIFO on ties).
- **Starvation decode-priority preemption** — while requests queue for a
  session slot, long prefill chunks are deferred and resident decode is drained
  instead, so waiters are admitted as soon as any resident finishes; bounded to
  ≤32 rounds so prefill cannot starve.
- **Near-done gate** — preemption triggers only when some resident decoder is
  within ≤64 tokens of `token_limit` (a slot is about to free), avoiding
  pointless prefill delay when all residents are long-output decoders. This gate
  alone restored fairness-A from 76.5 back to 7.32.

| Measurement (gfx1151, FP4 + f16 + MTP + sessions 4, same harness) | Result |
| --- | --- |
| Fairness-A degradation ratio | **160 → 7.32** |
| Fairness-B degradation ratio | **856 → 191** |
| Short-request TTFT | **3.4 s → 0.7 s** |
| ITL | unchanged |
| Greedy output | byte-identical (sha passes) |

### Three engines under load (MTP on)

Kept from the earlier concurrent-serving comparison (same harness, warmed up;
total / per-session decode TPS at concurrency 1 / 2 / 3 / 4 — A Halogen ·
B gufo-upstream · C MBufo):

| Engine | total TPS | per-session TPS | fairness A / B |
| --- | --- | --- | --- |
| A Halogen | 85 / 82 / 75 / 88 | 85 / 41 / 25 / 22 | 322 / 1140 |
| B upstream | 57 / 64 / 108 / 127 | 57 / 32 / 37 / 33 | 15 / 399 |
| C MBufo | 66 / 83 / 110 / 129 | 66 / 42 / 38 / 33 | **19 / 187** |

A has the best single-stream peak but only speculates when running alone — under
concurrency its draft head switches off and per-session TPS falls to ~22. Under load
**C scales best** (linear to ~129 TPS, flattest per-session) and, from v0.25.3, has the
best fairness too (SRTF prefill + starvation decode-priority cut the long-prefill
starvation of short requests ~2.3x). Full method and the other axes (prefill, fairness,
ITL): `高併發測速方法.md`.

### Speculative decoding (MTP / DFlash)

Speculation is per-model draft backends behind one draft registry (v0.27.0):

- **Flash-Next MTP** — batch_width 2/3/4 acceptance **48.8/38.9/41.7% →
  88.7/91.6/89.5%** (v0.25.1 fix: speculative batches bypass the packed mixed
  WMMA kernel that disagreed with the draft path). Acceptance no longer decays
  with batch size — upstream decays 95 → 82 → 68, this build stays steady
  ~90% — and aggregate throughput overtakes upstream at every width
  (97/111/133 vs 70/105/112). (Note: "flash 92%" in earlier notes is the
  width-3 peak 91.6% rounded; the retained figures are the three above.)
- **27B DFlash2** — works with blocked-Q8 KV (verify appends to the blocked
  sidecars, continuation cache and serve disk identity carry them
  bit-exactly); the v0.24.1 verify-routing fix recovered **+248–340%** decode
  (short/prose/code/json 5.8/5.6/13.2/11.9 → 22.9/19.5/56.2/52.3 tok/s).
- **Glimmer DFlash** — end-to-end speculative draft at **~89%** (E4 falsified
  as non-bug; v0.28.0 known limit, stated as-is).
- **Gemma 4 MTP** — after the async-GPU draft rewrite and single-forward
  verify path, decode reaches **74–86 tok/s** (**+50% at long KV**: p2k
  56.9 → 86.2, p8k 50.2 → 75.7 with `d=3`). Best draft depth is `d=3`
  (highest net throughput, **1.74–1.78 accepted/round**); `d=4` reaches the
  highest raw acceptance (1.98) at slightly higher verify cost. (Note:
  "~1.6/round" in earlier notes is corrected to 1.74–1.78 per CHANGELOG
  v0.26.0.)

### Prefix RAM cache + production sessions

Repeated prefixes are served from the upstream RAM cache (shared-prefix
boundaries, explicit RAM limit, coherent prefix index) instead of recomputed,
and continuity snapshots keep long conversations resumable. The production
shape is sessions × context, with honest session counts since the v0.25.4
`-j` fix (`-j 8` now really runs 8 ways: the TTFT staircase is gone, 8-stream
decode totals 124.5 → 148.2 / 121.1 → 144.4) and recommended defaults baked
into the CLI (`--sessions` 1 → 4, `--kv-quant` f16 → q8 on the llm path;
v0.25.2).

| Goal | Setting |
| --- | --- |
| 16 streams, high concurrency | `--sessions 16 -c 65536 --kv-quant q8` (about 103 GB; 20 GB left) |
| 8 streams, long context | `--sessions 8 -c 131072 --kv-quant q8` (about 101 GB; 20 GB left) |
| Single stream, max context | `--sessions 1 -c 262144 --kv-quant q8` |
| 4 streams, max context | `--sessions 4 -c 262144 --kv-quant q8` (about 101 GB; 20 GB left — same total positions as 8×131072) |
| Single stream, fastest & most precise prefill | `--sessions 1 -c 262144 --kv-quant f16` (full-precision KV; heaviest per position, but one stream fits easily) |
| Restore upstream scheduling | env `GUFO_MIXED_BATCH=0` |

Sessions times context is what the loader bounds, not context alone. On a
128 GB Strix Halo with the blocked Q8 cache and 4-bit value rows, eight streams
can even hold 192K each (`-c 196608`, 107 GB) if you want the reach and can live
with 16 GB of headroom, while `-c 98384` at sixteen streams is refused with
`text runner request-state claim exceeds state capacity`. Measurements use
UD-Q4_K_XL plus the MTP Q8_0 sidecar and BF16 mmproj, so the old 32K cap no
longer applies.

### What the scheduler guarantees

| Guarantee | Mechanism |
| --- | --- |
| A due decoder is never queued behind prefill | The merged step is entered only when no decoder is due (or while assembling a first batch), so prefill runs in the gaps and never ahead of a streaming request. |
| A chunk cannot push a decoder out of the rotation | `MarkAllDecodersDue` re-arms every resident decoder after each merged or prefill step; the following step serves all of them. |
| No prompt can monopolize the device | The chunk is a bounded slice (`--prefill-chunk`, 512 tokens by default), so a 250K-token prompt is fed one slice per cycle. |
| Equal sessions take turns | The decode queue rotates (`PopDecoder` moves a served request to the back) instead of fixing a head-of-line order, so no session keeps winning the front slot. |
| One client cannot fill the queue | `--max-pending-per-client` caps a client's pending requests (`client_quota` refusal), and dequeue round-robins across clients, so a burst from one client cannot starve another. |
| A layer that refuses loses nothing | Every layer degrades independently; a refused speculative row falls back to the original path — correct, just not accelerated. |

### Multi-model router

One port, many models. `serve llm --multi-model` turns the server into a
registry: the startup model becomes the **default**, and every request routes
to the model named in its `model` field. Models are added and dropped at
runtime through an admin API — no restart, no dropped connections. (The
[§3 model table](#supported-models-v0280) lists everything one port can serve.)

Start with a default model and enable the router:

```sh
./build/release/gufo serve llm --multi-model \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --served-model-name flash-next -c 262144 --kv-quant q8 --sessions 4
```

Route by name (omit `model` to hit the default):

```sh
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "flash-next", "messages": [{"role": "user", "content": "hi"}]}'
```

Load another model at runtime (e.g. Gemma 4), then route to it:

```sh
curl http://localhost:8080/admin/load -H "Content-Type: application/json" \
  -d '{"id": "gemma4",
       "path": "models/gemma-4-26b/gemma-4-26B-A4B-it-Q4_0.gguf",
       "max_context": 262144, "session_count": 2, "kv_cache_quant": "q8"}'

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "gemma4", "messages": [{"role": "user", "content": "hi"}]}'
```

List what is loaded, and drop a model when done:

```sh
curl http://localhost:8080/admin/models
curl http://localhost:8080/admin/eject -H "Content-Type: application/json" \
  -d '{"id": "gemma4"}'
```

**Memory safety.** Each `/admin/load` estimates the model's resident bytes and
checks free device memory first. If it does not fit, the router **LRU-evicts**
the least-recently-used non-default models to make room (reporting which in
`evicted`); pass `"allow_evict": false` to fail with `409 insufficient_memory`
instead. The default model is never evicted, so the registry can never empty.
Ejecting a model releases its GPU memory only after its in-flight requests
finish, so nothing is cut off mid-generation.

| Admin endpoint | Method | Body | Effect |
| --- | --- | --- | --- |
| `/admin/models` | GET | — | list loaded models, context, last-used |
| `/admin/load` | POST | `{id, path, max_context?, session_count?, kv_cache_quant?, vision_model_path?, enable_yarn?, allow_evict?}` | load + register a model (LRU-evicts on pressure) |
| `/admin/eject` | POST | `{id}` | drop a model; memory frees after in-flight requests drain |

The router is opt-in: without `--multi-model` the server is exactly the
single-model build, byte-for-byte.

## 2. Low-precision compute (cost)

Cost is argued with numbers. KV precision, weight precision and attention
variants are each measured against a baseline; nothing here is "narrower is
free". The one-line recommendation is `--kv-quant q8` with ROCmFP4 or Q4_K
weights — everything narrower is an explicit trade, documented below.

### KV cache: blocked Q8 / Q4

The upstream blocked Q8 KV path is generalized into a format + codec
abstraction, and both the key and value axes can be compressed further to 4-bit
18-byte packed blocks (18 bytes per 32 elements). The two axes are selected
independently, so K8V8, K4V8, K8V4 and K4V4 are all reachable. Since v0.26.1 the
codecs live in one shared core (`src/core/compute/attention/`:
`gufo_core_blocked_kv`) — blocked writer, scalar direct reader (`blocked_kv`),
WMMA prefill reader (`blocked_wmma`), Split-K decode reader (`blocked_splitk`).
Quantized KV is a shared capability; F16 default paths are untouched everywhere.

| Mode | Status | Notes |
| --- | --- | --- |
| `f16` | default | Baseline numerics. No KV quantization. |
| `q8` | supported | Blocked 8-bit, one f16 scale per 32 elements (34 bytes). |
| `q8` + `GUFO_KV_V4=1` | manual, K8V4 | 4-bit value packed to 18 bytes, key stays 8-bit. |
| `q8` + `GUFO_KV_Q4=1` | manual, K4V8 | 4-bit key packed to 18 bytes, value stays 8-bit. |
| `q8` + both | manual, K4V4 | key and value both 4-bit. |

**Use `q8`.** The blocked mode is bit-exact at the codec level; `f16` remains
the default and baseline. The 4-bit paths are environment-variable hooks for
machines where KV memory is the bottleneck, not `--kv-quant` format options, and
they change no default — with them unset the engine is bit-identical to a build
without these paths (verified by logit-dump hash comparison). Full details in
[docs/KV-V4.md](docs/KV-V4.md).

#### Footprint

Bytes per KV position, and the total for one session at `-c 262144` across
Qwen3.8 Flash-Next's 12 attention layers:

| Mode | Bytes/position | 262144 single session |
| --- | --- | --- |
| `f16` | 2048 | 6.44 GB |
| `q8` | 1088 | 3.42 GB |
| K8V4 / K4V8 | 832 | 2.62 GB |
| K4V4 | 576 | 1.81 GB |

#### What it costs

Correctness vs the `q8` baseline: three independent 400-token contexts, 98
matched-token decode steps each, averaged over the three contexts:

| Mode | decode KL | max decode KL | greedy argmax agreement |
| --- | --- | --- | --- |
| K4V8 | 0.0214 | 0.258 | 90.5% |
| K8V4 | 0.0209 | 0.120 | 93.2% |
| K4V4 | 0.0335 | 0.411 | 89.8% |

Read this table before enabling a mode narrower than `q8`:

- **K4V8 and K8V4 use exactly the same memory.** Keys are 512 elements; a 4-bit
  key with an 8-bit value and an 8-bit key with a 4-bit value both land on 832
  bytes. Moving precision from one axis to the other buys no memory but costs
  ~2.7 points of greedy agreement, so K4V8 has no use.
- **Only K4V4 actually shrinks the footprint**, at the cost of +60% decode KL
  and a 3.4x larger tail — roughly one greedy decision in ten changes. This is
  an explicit correctness-for-memory trade, not a free lunch.
- **Do not pick KV precision from a short prefill probe.** On a 33-token prompt
  K4V4 measures the same mean KL as K8V4 (0.0274): the per-position errors of
  the two axes are nearly uncorrelated and partially cancel when both are
  quantized, so small samples hide the damage. Only multi-context, decode-phase
  numbers order these modes correctly.
- The 4-bit block is the GGML `q4_0` layout (16 packed nibbles + one f16
  scale); per-axis K/V precision is already an upstream feature
  (`--cache-type-k` / `--cache-type-v`). What is specific to this engine is
  that both axes run through the blocked kernels, the mixed-batch path and
  snapshot identity, and the cost is measured, not assumed.

Measured `q8` vs a CPU f16 baseline: end-to-end deviation stays within the
blocked-KV design envelope, MTP draft hit rate stays in the 57–81% band,
and arithmetic and long-form answers stay coherent.

One thing to state plainly: under greedy decoding these blocked modes diverge
from `f16` more than raw numerical error alone would predict. That is a property
of the pre-existing blocked-KV design, not of the 4-bit blocks, and is
tracked separately; layering 4-bit blocks on top only widens the gap, it does
not stack gracefully.

### Weight precision: W8A8 / FP4 / MXFP4 / Q4_K

- **W8A8** — host-side 16-split varlen GEMM in the shared GEMM core
  (`src/core/compute/gemm`, v0.27.0); Flash-Next prefill W8A8 was rewired to
  the shared cores with byte-identical output. Glimmer runs W8A8/Q8_0 paths.
- **ROCmFP4 / ROCmFP4-FAST** — 4.25 bpw single-scale FAST down experts decode
  through the fp16 WMMA routed-expert GEMM (perm codebook LUT feeds the
  existing Q8_0 dequant), so MoE prefill rides the matrix cores instead of
  generic `mul_mat_q`. On gfx1151 a FAST model matches or beats the UD-Q4_K_XL
  baseline (1564 / 1368 vs 1556 / 1324 tok/s at 1.9K / 5K tokens) with decode
  speed and 64K tool-call accuracy unchanged. Large dense projections keep
  Q4_1 copies during decode (20 B/block, 41% fewer bytes than Q8_0): **+17.4%**
  decode (40.63 → 47.68 tok/s, v0.20.5).
- **MXFP4** — GPT-OSS 20B/120B run a native MXFP4 routed GEMM with fused
  decode MoE/QKV/`wo`/router paths; 120B loads via zero-copy streaming upload
  with chunked H2D mirrors.
- **Q4_K** — UD-Q4_K_XL is the recommended 27B build (beats single-scale FP4 on
  decode and draft acceptance) and the Flash-Next reference; bge-m3 ships a
  Q4_K_M loader; Gemma 4 runs Q4_0 weights.

### Attention variants: WMMA / flash / Split-K

- **WMMA prefill reader** (`blocked_wmma`) — single-session causal prefill
  reads the blocked int8 KV directly, skipping F16 staging dequant, for every
  layout the cache can hold (K34/K18 × V34/V18, dense and sparse). **+30%** on
  a 14K single-stream prefill (1085 → 1408 tok/s) over the scalar path.
- **Split-K decode reader** (`blocked_splitk`) — shared since v0.26.1; see
  [DecodeBatch + Split-K](#decodebatch--split-k-decode) for the measured
  deltas.
- **Flash-decoding** — Gemma 4 pairs Split-K flash-decoding attention with
  fused decode GEMV, fused norm/residual/epilogue kernels and device-side
  greedy argmax (p8k prefill 1321 → 1446 tok/s, byte-identical output), plus a
  SWA sliding-window KV skip in the WMMA attention.
- **Scalar direct reader** (`blocked_kv`) — the fallback every layout keeps.
- **F16 paths untouched** — every variant above is a quantized path; default
  F16 decode/prefill is never rerouted.

### Per-model precision

| Model | Weight precision | KV | F16 stays where |
| --- | --- | --- | --- |
| Qwen3.8 Flash-Next (primary) | ROCmFP4-FAST (4.25 bpw) or UD-Q4_K_XL; decode Q4_1 copies (+17.4%) | blocked-Q8 (`--kv-quant q8`); MTP Q8_0 sidecar | KV default `f16`; all F16 paths untouched |
| Qwen3.8 27B | UD-Q4_K_XL (recommended) or ROCmFP4-FAST | blocked-Q8 incl. DFlash2 (`QWEN_KV_QUANT`, `--kv-quant` override) | same as above |
| Gemma 4 26B-A4B | Q4_0; Q4→Q8 dense preconvert (one-time, −210 dequant launches/chunk) | SWA layers shared Q8 (prefill −2.8%, decode −6.4%) | **full-512 layers stay F16**; MTP drafts from a per-session F16 mirror |
| Qwen3.6-35B-A3B | Q6_K→Q8_0 load; split-K router GEMV | engine/bench path only | no serve backend yet |
| GPT-OSS 20B/120B | native MXFP4 routed GEMM | text-backend serving path | no speculative draft |
| Muse Glimmer 30B | W8A8 / Q8_0; MQB-first attention | `--kv-quant` flag on serve backend | DFlash draft, no F16 exception |
| Llama-3.1-8B | dense GQA (see model card) | text-backend serving path | no draft |
| Qwen3-Embedding-0.6B | dense Qwen attention | prefill-only single-prefill Embed | no ladder / draft (scoring N/A) |
| bge-m3 | Q4_K_M loader; per-shape autotune | prefill-only `/v1/embeddings` | **F16 hipBLASLt fused-QKV path** |
| bge-reranker-v2-m3 | shared rerank scoring heads | padded ScoreBatch (`+15%` over target) | scoring-only, no draft |
| Qwen3-Reranker | yes/no head, Qwen dispatch | scoring-only | scoring-only, no draft |

### Kernel-level speedups

These lift single-stream and batched throughput alike; none of them change
model semantics. (Kept here as the implementation detail behind §2's numbers.)

- **ROCmFP4-FAST experts on the fp16 WMMA core.** The 4.25 bpw
  `Q4_0_ROCMFP4_FAST` down experts now decode through the same fp16 WMMA
  routed-expert GEMM as the Q4_K/Q8_0 experts (a perm codebook LUT feeds the
  existing Q8_0 dequant), so a FAST model's MoE prefill rides the matrix cores
  instead of the generic `mul_mat_q` path. On gfx1151 a FAST model's prefill
  matches or beats the UD-Q4_K_XL baseline (1564 / 1368 vs 1556 / 1324 tok/s at
  1.9K / 5K tokens) with decode speed and 64K tool-call accuracy unchanged.
- **int8 WMMA direct prefill.** The single-session causal prefill
  (`WmmaCausalAttention`) now reads the blocked int8 KV cache directly, skipping
  the F16 staging dequant. It covers every KV block layout the cache can hold
  (K34/K18 x V34/V18, dense and sparse) and is selected by the same format/env
  the cache writer uses, so a layout can never be misread. On a 14K
  single-stream prefill this is **+30%** (1085 → 1408 tok/s) over the scalar
  path.
- **Wide prefill for every pass.** The wide prefill routes (fused projections,
  paired routed expert tiles, WMMA attention) apply to every prefill pass wider
  than a vector batch, not just 1024+ tokens.

### vs llama.cpp (v0.28.0, gfx1151)

Same weights, release build, greedy; prefill = cold-KV single-stream tok/s,
decode = speculative-on tok/s unless noted. Positive = MBufo faster. All six
figures are the measured v0.28.0 development numbers restated from CHANGELOG.

| Workload | Prefill delta | Decode delta | Conditions / notes |
| --- | --- | --- | --- |
| Flash-Next | **+52%** | on par or better | cold KV, greedy, MTP on |
| Qwen3.6-35B-A3B | **+52.5%** | parity-or-better | split-K router GEMV decode; greedy |
| bge-m3 (long docs) | **+35~50%** | n/a (prefill-only) | long-doc shapes, hipBLASLt autotune, cached executor |
| Qwen3-Embedding-0.6B (long docs) | **+41~50%** | n/a (prefill-only) | long-doc shapes, single-prefill Embed |
| Reranker (batched scoring) | n/a | **+15% over target** | padded multi-sequence ScoreBatch |
| GPT-OSS 120B | **+62%** | **-17%** | structural gap, stated as-is: decode stays memory-bound on this path |

No model-serving perf tests were run for this release bump; figures above are
the measured v0.28.0 development numbers restated, not re-benchmarked here.

## 3. GGUF coverage (compatibility)

### Supported models (v0.28.0)

One port serves all of these through the [multi-model router](#multi-model-router).
`Serve` = HTTP backend present; `Ladder` = registry/pilot or scoring test present;
`MTP/draft` = speculative draft backend present. 11 rows: 8 newly usable in
v0.28.0 plus the three incumbents (Flash-Next, 27B, Gemma 4).

| Model | Params | Serve | Ladder | MTP/draft | Notes |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8 Flash-Next (primary) | 30B-A3B MoE | yes | pilot | MTP | ROCmFP4-FAST preset, blocked-Q8 KV |
| Qwen3.8 27B | 27B | yes (`QWEN_KV_QUANT`, `--kv-quant` override) | pilot | DFlash2 (incl. q8) | dense + MoE paths |
| Gemma 4 | 26B-A4B | yes (HTTP backend, native tools) | ladder 15/15 @24k | MTP sidecar (q8 via F16 mirror) | full-512 layers stay F16 |
| Qwen3.6-35B-A3B | 35B-A3B MoE | engine/bench only | — | none | Q6_K→Q8_0 load, split-K decode MoE; keep ≤60K tokens above 64K ctx (MODE2 reset risk) |
| GPT-OSS 20B / 120B | 20B / 117B MoE (MXFP4) | yes (text backend) | pilot | none | zero-copy streaming upload + chunked H2D mirrors for 120B |
| Muse Glimmer 30B | 30B | yes (ATEM tools, kv-quant) | prompt path | DFlash end-to-end (~89%) | DFlash 89% rate is measured, E4 falsified as non-bug |
| Llama-3.1-8B | 8B dense GQA | yes (text backend) | pilot adapter | none | — |
| Qwen3-Embedding-0.6B | 0.6B | yes (`/v1/embeddings`) | n/a (prefill-only) | none | pooled single-prefill Embed |
| bge-m3 | 0.6B XLM-R | yes (`/v1/embeddings`) | encoder tests | none | CLS pooling, Q4_K_M loader, hipBLASLt autotune |
| bge-reranker-v2-m3 | cross-encoder | yes (`/v1/reranking`, BERT dispatch) | ScoreBatch tests | none | padded multi-sequence batching |
| Qwen3-Reranker | yes/no head | yes (`/v1/reranking`, Qwen dispatch) | scoring tests | none | shared rerank scoring heads |

DSpark remains legacy. Model weights are not bundled; each guide under
[docs/models](docs/models/README.md) lists the required files.

### Endpoints + tool mapping

| Endpoint | Method | Serves | Notes |
| --- | --- | --- | --- |
| `/v1/chat/completions` | POST | text chat (+images, tools, structured output, streaming) | empty `model` routes to the default instead of `400 missing_model` |
| `/v1/embeddings` | POST | bge-m3, Qwen3-Embedding | cached single-prefill Embed executor |
| `/v1/reranking` | POST | bge-reranker-v2-m3 (BERT dispatch), Qwen3-Reranker (Qwen dispatch) | padded multi-sequence ScoreBatch |
| `/admin/models`, `/admin/load`, `/admin/eject` | GET / POST | router registry | see [memory safety](#multi-model-router) |

Tool calls map per backend family — ATEM (Glimmer), Gemma native (`<\|tool|>`
template, ladder 15/15 @24k), Qwen — and the Qwen `--kv-quant` explicit flag
now overrides `QWEN_KV_QUANT` on the Qwen path. For Open WebUI, VS Code,
OpenAI SDK and Responses-API clients, point the base URL at
`http://localhost:8080/v1`; see the [API contract](docs/SERVER.md).

### Fallback semantics

- Unknown model id is rejected without affecting loaded models.
- Load-time OOM LRU-evicts cold non-default models (`allow_evict:false` →
  `409 insufficient_memory`); the default is never evicted.
- Eject frees GPU memory after in-flight requests drain.
- Without `--multi-model` the server is byte-for-byte the single-model build.
- Any attention/GEMM layer that cannot serve a batch falls back to the legacy
  path — correct, just not accelerated.
- Snapshots are versioned by KV format; a snapshot written in one format is not
  reused by another.

### Wiring a new model (five steps)

There is not yet a standalone onboarding manual (no `GGUF通用化.md` in this
tree — verified 2026-11-09); until one lands, this checklist plus the
per-model guides under [docs/models](docs/models/README.md) are the procedure.
Each step names the existing models that already went through it:

1. **EngineRegistry adapter** — register the family on the L0/L1 factory with
   per-request `model` dispatch (precedents: 27B, flash-next, gemma4, glimmer,
   llama).
2. **Serve backend** — text backend, or the embeddings single-prefill Embed, or
   the reranking BERT/Qwen dispatch; carry the `--kv-quant` flag through.
3. **Ladder pilot or scoring test** — registry pilot (27B, flash-next, gpt-oss,
   llama), prompt path (glimmer), encoder/scoring tests (bge family);
   prefill-only models mark ladder N/A.
4. **Draft backend (or explicit none)** — MTP / DFlash2 behind the draft
   registry (gemma4 drafts from a per-session F16 mirror to cover 4-bit
   grids); record `none` with the reason (prefill-only, no draft yet).
5. **Serve registration + docs** — `--served-model-name`, free-memory gate
   estimate, a `docs/models/<family>/` guide (README / BENCHMARKS / QUALITY /
   EXPERIMENTS per [docs/models](docs/models/README.md)), and a CHANGELOG
   Known-limits line if it has one.

## Quickstart

Start Qwen3.8 Flash-Next with the recommended cache settings:

```sh
./build/release/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mmproj models/qwen3.8-flash-next/mmproj-BF16.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/agentionai-MTP-Q8_0.gguf \
  --speculative mtp --think off \
  --served-model-name gufo-qwen3.8-flash-next \
  --sessions 2 -c 262144 --kv-quant q8
```

Ask it something through the OpenAI-compatible API:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gufo-qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "Say something"}]
  }'
```

For Open WebUI, VS Code, OpenAI SDK and Responses-API clients, point the base
URL at `http://localhost:8080/v1`. Chat Completions and Responses support text,
images, function tools, structured output and streaming; see the
[API contract](docs/SERVER.md).

### What v0.28.0 adds

| Router function | Behavior |
| --- | --- |
| Multi-model hosting | L0/L1 EngineRegistry factory with adapters for 27B, flash-next, gemma4, glimmer and llama; `/admin/load` + `/admin/eject` at runtime, startup model is the never-evicted default |
| Automatic dispatch | each request routes by its `model` field; empty `model` on `openai_chat` routes to the default instead of `400 missing_model` |
| Fallback semantics | unknown model id is rejected without loaded models affected; load-time OOM LRU-evicts cold non-default models (`allow_evict:false` → `409 insufficient_memory`); eject frees GPU memory after in-flight requests drain |
| Embeddings | `POST /v1/embeddings` via cached single-prefill Embed (bge-m3, Qwen3-Embedding) |
| Reranking | `POST /v1/reranking` with BERT (bge-reranker-v2-m3) + Qwen (Qwen3-Reranker) dispatch and padded batching |
| Tool calls | ATEM / Gemma / Qwen tool-call mapping across backends |
| Qwen KV flag fix | explicit `--kv-quant` now overrides `QWEN_KV_QUANT` on the Qwen path |

## Benchmark (MBufo v0.25.5, gfx1151 Strix Halo)

Cold KV (unique prompt per request, `cached_tokens≈0`), think off, **F16 KV**,
`-j 1`, ctx 262144. Four configs: 27B (UD-Q4_K_XL vs ROCmFP4-FAST) + Flash-Next
(ROCmFP4-FAST vs UD-Q4_K_XL). Speculative: 27B = DFlash2, Flash-Next = MTP.

These are single-stream numbers: they isolate the model and build choice. What
changes under load is how many of them run at once, which is the
[mixed-batch](#1-high-concurrency-architecture) story.

### Prefill (tok/s, higher is better)

| tokens | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---:|---:|---:|---:|---:|
| 2K | 605 | 455 | **1739** | 1301 |
| 8K | 615 | 564 | **1701** | 1549 |
| 32K | 567 | 551 | 1598 | **1618** |
| 64K | 508 | 496 | **1695** | 1678 |
| 128K | 409 | 407 | **1504** | 1507 |
| 250K | — | — | **1438** | 1415 |

> 27B ROCmFP4-**FAST** prefill ≈ the non-FAST STRIX build — single-scale only
> changes the codebook, not the fp16-WMMA prefill engine (455/564/551/407 vs
> 483/523/520/383). FAST's real speed comes from the int8-MMQ consumer path,
> which is wired for Flash-Next's MoE experts, not the dense 27B.

### Decode (tok/s, speculative on)

Standard: 5 cold-KV scenarios (temp labelled) + an 8-task ReAct agent suite.

| scenario | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---|---:|---:|---:|---:|
| prose (temp 0.4) | 23.5 | 19.7 | 43.3 | **43.6** |
| fibonacci (temp 0.4) | 67.9 | 56.2 | **86.3** | 84.7 |
| json (temp 0) | **55.1** | 46.5 | 51.5 | 51.7 |
| greedy code /128 (temp 0) | 61.9 | 55.3 | 70.5 | **73.2** |
| repeat text /256 (temp 0) | 70.6 | 60.8 | 74.8 | **75.3** |

### Agentic (agent_bench_v2, t1–t8, sandboxed tools)

| | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---|---:|---:|---:|---:|
| passed | 8/8 | 7/8 | 7/8 | 7/8 |
| nuclear (must be 0) | 0 | 0 | 0 | 0 |
| gen tps range | 9.1–31.1 | 9.3–21.5 | 16.7–56.6 | 16.8–55.1 |

### Recommended build per model

- **Qwen 3.8 27B → UD-Q4_K_XL.** FP4-FAST wins the FP4 family (all 5 decode
  scenarios + 8/8 agentic vs STRIX's 7/8), but single-scale FP4 still loses
  decode and draft-acceptance to Q4_K_XL, so Q4_K_XL is the best 27B build.
- **Qwen 3.8 Flash-Next → ROCmFP4.** Fastest decode (prose 45.0, fibonacci 87.0),
  8/8 agentic, and ~94 GB vs Q4_K_XL's ~111 GB — ROCmFP4 is the best Flash-Next
  build.

> 27B ROCmFP4 decode is the **post-v0.24.1-fix** figure: before the fix, DFlash2
> verify fell to a per-row GEMV loop and collapsed; after, FAST matches non-FAST
> and edges it.

## Upstream tracking

MBufo follows upstream `main` by merging, not by forking off. Upstream release
tags are immutable; MBufo release tags always carry the `mbufo-vX.Y.Z` prefix,
so they can never collide with an upstream `vX.Y.Z`.

| Item | Value |
| --- | --- |
| Upstream repo | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| Upstream base release | `v0.26.0` (cherry-picked safe fixes; full merge reverted — see below) |
| Merged upstream `main` | cherry-picked to `v0.26.0` (2026-10-28): #732 neutral sampler fields, #736 repeated `--served-model-name`. Full 0.21→0.26 merge **reverted**: #758 chunked-DeltaNet prefill regressed gfx1151 FN prefill 30-46% (upstream #850 later removed it); audio/Windows not applicable to this gfx1151 Linux fork. |
| Merge policy | upstream `main` merged with `--no-ff`; MBufo-specific commits stay identifiable, upstream history is never rewritten |

When upstream ships a new release, the "upstream base release" row is updated in
the same commit that merges it, so this README always states exactly which
upstream state this tree contains.

## Build from source

The supported target is Linux x86-64 on AMD Strix Halo (`gfx1151`). The build
is the upstream build; MBufo adds no build-system requirements.

```sh
cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
```

A C++20 compiler, CMake 3.21+, Ninja, pkg-config and the ROCm development
libraries (HIP, hipBLAS, hipBLASLt, rocBLAS, hipCUB, rocPRIM, rocWMMA) plus the
ordinary system libraries (ICU, libcurl, OpenSSL, libpng, libjpeg, libwebp,
FFmpeg) are required; the qualified toolchain is GCC 15.3 with ROCm 7.2.3. A
Nix build is also provided (`nix build`). The GPU driver must let your user
access `/dev/kfd` and `/dev/dri`; model weights are acquired separately. Each
model guide under [docs/models](docs/models/README.md) lists the required files
and benchmark settings. Full dependency details, the ROCr queue-memory patch and
container notes live in the
[upstream build docs](https://github.com/gufo-org/gufo#build-from-source).

## Documentation index

| Doc | Contents |
| --- | --- |
| [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md) | Mixed-batch design: invariants, validation matrix, measurements |
| [docs/KV-CACHE.md](docs/KV-CACHE.md) | Conversation and component cache (upstream) |
| [docs/KV-V4.md](docs/KV-V4.md) | 4-bit K/V blocks: layout, correctness tables |
| [docs/SERVER.md](docs/SERVER.md) | HTTP API contract |
| [docs/CLI.md](docs/CLI.md) | Command-line reference |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Build, testing, presets |
| [docs/TESTING.md](docs/TESTING.md) | Hosted CI suites and local quality checks |
| [docs/models/README.md](docs/models/README.md) | Per-model guides: README / BENCHMARKS / QUALITY / EXPERIMENTS |
| [CHANGELOG.md](CHANGELOG.md) | Upstream release history and MBufo releases |

## Compatibility and limits

- Mixed batching changes scheduling, not model semantics. Any layer that cannot
  serve the whole batch falls back to the old path.
- `--kv-quant q8` trades "no longer bit-identical to the
  baseline" for memory. When exact agreement with the baseline matters more than
  capacity, use `f16`.
- Conversation state is versioned by KV format; a snapshot written in one format
  is not reused by another.
- Upstream release tags are immutable and are never reused as MBufo tags.
- Known limits (v0.28.0, short): qwen35 above 64K context should stay under
  60K tokens (driver MODE2 reset risk); DSpark stays legacy; gemma4 full-512
  layers stay F16; Glimmer DFlash sits at ~89% (E4 falsified as non-bug).

## License and credits

Upstream gufo is MIT licensed; adapted code and dependencies keep their own
notices in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and `licenses/`. MBufo preserves those notices. Model weights are not bundled
and remain under their publishers' terms.

MBufo is a fork of [gufo](https://github.com/gufo-org/gufo) (by gufo-org and
contributors). The low-precision KV and performance work builds on upstream's
Q8 KV cache, int8 WMMA, persistence and history-drafting changes.
