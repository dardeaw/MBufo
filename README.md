# MBufo

A [gufo](https://github.com/gufo-org/gufo) fork for AMD Strix Halo, built around
two things: **high-concurrency mixed-batch scheduling** and a **low-bit KV
cache**.

> [繁體中文版本](README.zh-Hant.md)

Upstream gufo owns single-stream peak performance. MBufo asks the serving
question instead: what does one Strix Halo node do with sixteen requests at
once, and how much context can each of them hold? Those are the only two
subsystems this fork changes. The engine, kernels and the bulk of the code are
upstream's; thanks to gufo-org and all contributors.

| Pillar | What changes | What you get |
| --- | --- | --- |
| **Mixed-batch scheduling** | Decode rows and prefill chunks share one `ForwardBatch` and one attention kernel; the batch envelope grows to 32 sessions / 1024 rows. | Steady ITL and fair service under load: a streaming token waits one step instead of two (121 ms → 57 ms per token on a 5-row batch), 16 concurrent requests at 99% GPU, 15.3 s vs 17.0 s at 16-way. |
| **Low-bit KV cache** | The blocked Q8 path becomes a format + codec abstraction, with 4-bit 18-byte blocks on the K and V axes independently. | 3.42 GB instead of 6.44 GB for a 262144-token session, 1.81 GB at K4V4 — quality cost measured, not assumed. |

Mixed batching is on by default. The KV cache defaults to `f16`; `q8` is a
one-flag opt-in. Neither changes model semantics.

Target hardware: Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`), up to 128 GiB
unified memory. Model focus: Qwen3.8 Flash-Next and Qwen3.8 27B.

**Contents:** [Quickstart](#quickstart) ·
[Mixed-batch scheduling](#mixed-batch-scheduling-high-concurrency) ·
[Low-bit KV cache](#low-bit-kv-cache) ·
[Kernel-level speedups](#kernel-level-speedups) ·
[Serving presets](#recommended-serving-settings) ·
[Benchmark](#benchmark-mbufo-v0255-gfx1151-strix-halo) ·
[Upstream tracking](#upstream-tracking) ·
[Build](#build-from-source) · [Docs](#documentation-index) ·
[Limits](#compatibility-and-limits)

## Benchmarks at a glance

Two result sets, both on one Strix Halo (gfx1151), cold KV, greedy.

**Single-stream (cold KV)** — Qwen3.8 27B Q4_K_XL and Flash-Next ROCmFP4, one
request at a time: prefill and speculative-decode throughput. Full tables:
[Benchmark](#benchmark-mbufo-v0255-gfx1151-strix-halo).

**High-concurrency, three engines (MTP on)** — A Halogen · B gufo-upstream · C MBufo,
1–4 concurrent streams, same harness, warmed up. Total / per-session decode TPS at
concurrency 1 / 2 / 3 / 4:

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

### Recommended serving settings

| Goal | Setting |
| --- | --- |
| 16 streams, high concurrency | `--sessions 16 -c 65536 --kv-quant q8` (about 103 GB; 20 GB left) |
| 8 streams, long context | `--sessions 8 -c 131072 --kv-quant q8` (about 101 GB; 21 GB left) |
| Single stream, max context | `--sessions 1 -c 262144 --kv-quant q8` |
| 4 streams, max context | `--sessions 4 -c 262144 --kv-quant q8` (about 101 GB; 21 GB left — same total positions as 8×131072) |
| Single stream, fastest & most precise prefill | `--sessions 1 -c 262144 --kv-quant f16` (full-precision KV; heaviest per position, but one stream fits easily) |
| Restore upstream scheduling | env `GUFO_MIXED_BATCH=0` |

Sessions times context is what the loader bounds, not context alone. On a
128 GB Strix Halo with the blocked Q8 cache and 4-bit value rows, eight streams
can even hold 192K each (`-c 196608`, 107 GB) if you want the reach and can live
with 16 GB of headroom, while `-c 98304` at sixteen streams is refused with
`text runner request-state claim exceeds state capacity`. Measurements use
UD-Q4_K_XL plus the MTP Q8_0 sidecar and BF16 mmproj, so the old 32K cap no
longer applies.

## Mixed-batch scheduling (high concurrency)

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

### What the scheduler guarantees

| Guarantee | Mechanism |
| --- | --- |
| A due decoder is never queued behind prefill | The merged step is entered only when no decoder is due (or while assembling a first batch), so prefill runs in the gaps and never ahead of a streaming request. |
| A chunk cannot push a decoder out of the rotation | `MarkAllDecodersDue` re-arms every resident decoder after each merged or prefill step; the following step serves all of them. |
| No prompt can monopolize the device | The chunk is a bounded slice (`--prefill-chunk`, 512 tokens by default), so a 250K-token prompt is fed one slice per cycle. |
| Equal sessions take turns | The decode queue rotates (`PopDecoder` moves a served request to the back) instead of fixing a head-of-line order, so no session keeps winning the front slot. |
| One client cannot fill the queue | `--max-pending-per-client` caps a client's pending requests (`client_quota` refusal), and dequeue round-robins across clients, so a burst from one client cannot starve another. |
| A layer that refuses loses nothing | Every layer degrades independently; a refused speculative row falls back to the original path — correct, just not accelerated. |

### What it buys

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

## Low-bit KV cache

The upstream blocked Q8 KV path is generalized into a format + codec
abstraction, and both the key and value axes can be compressed further to 4-bit
18-byte packed blocks (18 bytes per 32 elements). The two axes are selected
independently, so K8V8, K4V8, K8V4 and K4V4 are all reachable.

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

### Footprint

Bytes per KV position, and the total for one session at `-c 262144` across
Qwen3.8 Flash-Next's 12 attention layers:

| Mode | Bytes/position | 262144 single session |
| --- | --- | --- |
| `f16` | 2048 | 6.44 GB |
| `q8` | 1088 | 3.42 GB |
| K8V4 / K4V8 | 832 | 2.62 GB |
| K4V4 | 576 | 1.81 GB |

### What it costs

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

## Kernel-level speedups

These lift single-stream and batched throughput alike; none of them change
model semantics.

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

## Benchmark (MBufo v0.25.5, gfx1151 Strix Halo)

Cold KV (unique prompt per request, `cached_tokens≈0`), think off, **F16 KV**,
`-j 1`, ctx 262144. Four configs: 27B (UD-Q4_K_XL vs ROCmFP4-FAST) + Flash-Next
(ROCmFP4-FAST vs UD-Q4_K_XL). Speculative: 27B = DFlash2, Flash-Next = MTP.

These are single-stream numbers: they isolate the model and build choice. What
changes under load is how many of them run at once, which is the
[mixed-batch](#mixed-batch-scheduling-high-concurrency) story.

### Prefill (tok/s, higher is better)

| tokens | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---:|---:|---:|---:|---:|
| 2K | 605 | 455 | **1739** | 1301 |
| 8K | 615 | 564 | **1701** | 1549 |
| 32K | 567 | 551 | 1598 | **1618** |
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
| Upstream base release | `v0.20.0` |
| Merged upstream `main` | `0c03748a` — *chore(main): release 0.20.0* (2026-10-21) |
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

## License and credits

Upstream gufo is MIT licensed; adapted code and dependencies keep their own
notices in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and `licenses/`. MBufo preserves those notices. Model weights are not bundled
and remain under their publishers' terms.

MBufo is a fork of [gufo](https://github.com/gufo-org/gufo) (by gufo-org and
contributors). The low-precision KV and performance work builds on upstream's
Q8 KV cache, int8 WMMA, persistence and history-drafting changes.
