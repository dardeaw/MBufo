# MBufo

A high-concurrency fork of [gufo](https://github.com/gufo-org/gufo) for AMD
Strix Halo.

> [繁體中文版](README.zh-Hant.md)

MBufo keeps the upstream engine intact and changes the parts that decide how
much of the chip is busy when more than one request is in flight. Upstream gufo
optimizes the single-stream peak; MBufo saturates the hardware under concurrent
load without changing model output semantics. Models, kernels and the large
majority of the code come from the upstream open-source project; credit to
gufo-org and all contributors.

Target hardware: Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`), up to 128 GiB
unified memory. Model focus: Qwen3.8 Flash-Next and Qwen3.8 27B.

## Upstream tracking

MBufo follows upstream `main` and merges it rather than diverging. Upstream
releases stay immutable; MBufo release tags are namespaced `mbufo-vX.Y.Z` so
they can never collide with upstream `vX.Y.Z` tags.

| Item | Value |
| --- | --- |
| Upstream repository | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| Upstream base release | `v0.17.1` |
| Upstream `main` merged | `e7730942` — *feat(serve): prepare the structured-output vocabulary before serving (#638)* (2026-10-18) |
| Commits past the upstream release | 51 |
| Merge policy | upstream `main` is merged with `--no-ff`; MBufo-only commits stay identifiable and upstream history is never rewritten |

When upstream releases a new version, the base-release row above is updated in
the same pull request that merges it, so the README always states exactly which
upstream state this tree contains.

## What MBufo adds

- **Mixed-batch attention.** One decode row and N prefill-chunk rows are packed
  into a single `ForwardBatch`, replacing per-item attention with one mixed
  attention kernel. This removes the decode/prefill tick-tock that idles the GPU
  at low concurrency: 2.1x on small batches, and 5.3 s versus 6.3 s measured at
  the serving level.
- **A 32-session envelope.** `ForwardBatch` was raised from 8 sessions / 8 rows
  to 32 sessions / 1024 rows, which also required fixing three hard caps (GDN,
  MoE, logits buffer). 16-way concurrency pins GPU utilization at 99% with
  all-correct outputs and zero errors.
- **MTP parity for batched work.** Chunks entering a mixed batch get
  predictor-KV feeding (frontier check plus tail), matching single-forward
  behavior for MTP sessions. Speculative rows fall back to the legacy path when
  a layer cannot serve them: correct, just not accelerated.
- **Serve merging.** The scheduler fuses a prefill chunk with plain decodes
  (`TryStepMixedBatch`). Every layer can degrade independently, and any refusal
  falls back to the legacy pair.
- **KV cache quantization.** Upstream's blocked Q8 KV path is generalized behind
  a format/codec abstraction, and an opt-in FP8 cache is added alongside it.

Mixed batches are on by default. Set `GUFO_MIXED_BATCH=0` to restore upstream
behavior exactly.

## KV cache selection (`--kv-quant`)

| Mode | Status | Notes |
| --- | --- | --- |
| `f16` | default | Reference numerics. No quantization of the KV cache. |
| `q8` | supported | Blocked Q8 with a per-32 f16 scale (34 bytes per 32 elements). |
| `fp8` | supported, recommended | Software E4M3 with a per-32 f16 scale, same 34-byte block layout as Q8. |

**Recommendation: use `--kv-quant fp8`.** It is bit-exact at the codec level and
matches or slightly beats `q8` end to end, at the same memory footprint as
`q8` and half the footprint of `f16`.

What was measured, against the CPU f16 reference:

- The E4M3 encoder reproduces all 21 OCP format-table entries (including
  boundaries and subnormals) and every representable code round-trips exactly.
- End-to-end deviation is on par with `q8`: at the first position the KL is
  0.11752 for `fp8` versus 0.11700 for `q8`; at later positions `fp8` is
  slightly closer to the reference. Over a 50-position probe, the greedy
  argmax matches f16 38/50 times for `fp8` versus 33/50 for `q8`.
- Serving is unaffected: MTP draft acceptance stays in the 57% to 81% range and
  responses remain coherent for both arithmetic and long-form generation.
- FP8 quantization error is larger than INT8 on smooth data but no larger on
  outlier-heavy data, which is the distribution that dominates post-RoPE keys.
  That is why the end-to-end result tracks `q8` rather than trailing it.

Caveat, stated plainly: both blocked modes (`q8` and `fp8`) diverge from `f16`
under greedy decoding more than a pure numerical-error argument would suggest.
This is a property of the existing blocked-KV design, not something FP8
introduced, and it is tracked separately.

## Quickstart

Serve Qwen3.8 Flash-Next with the recommended cache setting:

```sh
./build/release/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mmproj models/qwen3.8-flash-next/mmproj-BF16.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/agentionai-MTP-Q8_0.gguf \
  --speculative mtp --think off \
  --served-model-name gufo-qwen3.8-flash-next \
  --sessions 2 -c 262144 --kv-quant fp8
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

For Open WebUI, VS Code, OpenAI SDK and Responses-API clients, point the base URL
at `http://localhost:8080/v1`. Chat Completions and Responses support text,
images, function tools, structured output and streaming; see the
[API contract](docs/SERVER.md).

### Recommended serving settings

| Goal | Setting |
| --- | --- |
| Single stream, maximum context | `--sessions 1 -c 262144 --kv-quant fp8` |
| High concurrency | `--sessions 16 -c 32768` (uses roughly 99 GB; 32K bounds the KV cache) |
| Restore upstream scheduling | `GUFO_MIXED_BATCH=0` in the environment |

## Build from source

Linux x86-64 on AMD Strix Halo (`gfx1151`) is the supported target. The build is
the upstream build; MBufo adds no build-system requirements.

### With Nix

```sh
nix build
./result/bin/gufo diagnose
./result/bin/gufo serve llm --model /path/to/model.gguf
```

### Without Nix

Install a C++20 compiler, CMake 3.21+, Ninja, pkg-config, the ROCm development
libraries (HIP, hipBLAS, hipBLASLt, rocBLAS, hipCUB, rocPRIM, rocWMMA) and the
ordinary system libraries (ICU, libcurl, OpenSSL, libpng, libjpeg, libwebp,
FFmpeg). The currently qualified toolchain is GCC 15.3 with ROCm 7.2.3.

```sh
cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
```

The GPU driver must allow your user to access `/dev/kfd` and `/dev/dri`. Model
weights are acquired separately; each model guide under [docs/models](docs/models/README.md)
lists the required files and benchmark settings. For the full dependency table,
the ROCr queue-memory patch, and container guidance, see the
[upstream build documentation](https://github.com/gufo-org/gufo#build-from-source).

## Documentation

| Document | Contents |
| --- | --- |
| [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md) | Mixed-batch design: invariants, validation matrix, measurements |
| [docs/KV-CACHE.md](docs/KV-CACHE.md) | Conversation and component cache (upstream) |
| [docs/SERVER.md](docs/SERVER.md) | HTTP API contract |
| [docs/CLI.md](docs/CLI.md) | Command-line reference |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Building, tests, presets |
| [docs/TESTING.md](docs/TESTING.md) | Hosted CI suite and local quality checks |
| [CHANGELOG.md](CHANGELOG.md) | Upstream release history plus MBufo releases |

## Compatibility and limits

- Mixed batching changes scheduling, not model semantics. Any layer that cannot
  serve a packed batch falls back to the legacy path.
- `--kv-quant q8` and `--kv-quant fp8` trade reference-identical numerics for
  memory. Use `f16` when exact reference agreement matters more than capacity.
- Conversation state is versioned by KV format; a snapshot written with one
  format is not reused by another.
- Upstream release tags are immutable and are not reused as MBufo tags.

## License and credits

Upstream gufo is MIT licensed; adapted code and dependencies retain their own
notices in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and `licenses/`. MBufo keeps those notices. Model weights are not bundled and
retain their publishers' terms.

MBufo is a fork of [gufo](https://github.com/gufo-org/gufo) by gufo-org and
contributors. The mixed-batch work builds on upstream Q8 KV cache, int8 WMMA,
persistence and history-drafting changes.
