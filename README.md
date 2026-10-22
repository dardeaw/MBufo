# MBufo

A [gufo](https://github.com/gufo-org/gufo) fork for AMD Strix Halo, focused on
**low-precision KV caching** and **performance tuning**.

> [繁體中文版本](README.zh-Hant.md)

MBufo keeps the upstream engine intact and changes two things: how much
precision the KV cache uses, and how hard the chip works under concurrent load.
Upstream gufo owns single-stream peak performance; MBufo adds low-precision KV
formats — with a measured, not assumed, quality cost — and the tuning that keeps
the GPU saturated across many concurrent requests without changing model
semantics. The base engine, kernels and the bulk of the code come from the
upstream open source project; thanks to gufo-org and all contributors.

Target hardware: Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`), up to 128 GiB
unified memory. Model focus: Qwen3.8 Flash-Next and Qwen3.8 27B.

## What MBufo adds

### Low-precision KV cache

The upstream blocked Q8 KV path is generalized into a format + codec
abstraction, and both the key and value axes can
be compressed further to 4-bit 18-byte packed blocks (18 bytes per 32
elements). The two axes are selected independently, so K8V8, K4V8, K8V4 and
K4V4 are all reachable.

| Mode | Status | Notes |
| --- | --- | --- |
| `f16` | default | Baseline numerics. No KV quantization. |
| `q8` | supported | Blocked 8-bit, one f16 scale per 32 elements (34 bytes). |
| `q8` + `GUFO_KV_V4=1` | manual, K8V4 | 4-bit value packed to 18 bytes, key stays 8-bit. |
| `q8` + `GUFO_KV_Q4=1` | manual, K4V8 | 4-bit key packed to 18 bytes, value stays 8-bit. |
| `q8` + both | manual, K4V4 | key and value both 4-bit. |

**Use `q8`.** The blocked mode is bit-exact at the codec level; `f16` remains the default and baseline. The
4-bit paths are environment-variable hooks for machines where KV memory is the
bottleneck, not `--kv-quant` format options, and they change no default — with
them unset the engine is bit-identical to a build without these paths (verified
by logit-dump hash comparison). Full details in [docs/KV-V4.md](docs/KV-V4.md).

#### Correctness first

Bytes per KV position, and the total for one session at `-c 262144` across
Qwen3.8 Flash-Next's 12 attention layers:

| Mode | Bytes/position | 262144 single session |
| --- | --- | --- |
| `f16` | 2048 | 6.44 GB |
| `q8` | 1088 | 3.42 GB |
| K8V4 / K4V8 | 832 | 2.62 GB |
| K4V4 | 576 | 1.81 GB |

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

### Performance tuning

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
- **Mixed-batch attention.** One decode row plus N prefill-chunk rows share a
  single `ForwardBatch` and one mixed attention kernel, replacing per-item
  attention. This removes the decode/prefill idle alternation at low
  concurrency: 2.1x on small batches, 5.3 s vs 6.3 s measured at the serving
  layer.
- **32-session envelope.** `ForwardBatch` grows from 8 sessions / 8 rows to
  32 sessions / 1024 rows, fixing three hard caps along the way (GDN, MoE,
  logits buffer). 16 concurrent requests hold 99% GPU utilization with all
  outputs correct and zero errors.
- **MTP parity for batched work.** Chunks entering the mixed batch get their
  predictor-KV (frontier check plus tail) so they behave like a single forward.
  A layer that cannot serve a speculative row degrades that row to the original
  path: correct, just not accelerated. The default MTP draft depth is 5 (the
  loader caps Flash-Next at 7); depth 5 matches 7 on decode and leads it by
  5–6 tok/s on greedy workloads.
- **Serve batching.** The scheduler merges prefill chunks with ordinary decode
  into one batch (`TryStepMixedBatch`). Every layer can degrade independently;
  any step that refuses falls back to the old dual path.

Mixed batching is on by default. Set `GUFO_MIXED_BATCH=0` to restore upstream
behavior exactly.

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
| Single stream, max context | `--sessions 1 -c 262144 --kv-quant q8` |
| High concurrency | `--sessions 16 -c 32768` (about 99 GB; 32K caps the KV size) |
| Restore upstream scheduling | env `GUFO_MIXED_BATCH=0` |

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
