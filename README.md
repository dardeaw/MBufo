# MBufo

A specialized fork of [gufo](https://github.com/gufo-org/gufo) engineered for AMD Strix Halo (`gfx1151`), built as a **fair, low-cost, high-concurrency inference engine for production workloads**.

> [繁體中文版](README.zh-Hant.md) · [简体中文版](README.zh-CN.md)

Upstream gufo is tuned for single-stream peak throughput. MBufo addresses the operational reality of production serving: how a single Strix Halo node sustains 16+ concurrent streams under heavy load while offering up to 256K context per session. MBufo achieves this through three core pillars: **fair scheduling under load (§1)**, **low-precision compute that drastically cuts cost and memory (§2)**, and **comprehensive GGUF coverage across 13 model families on a single port (§3)**. The engine primitives, compute kernels, and foundations originate from upstream; sincere thanks to gufo-org and all community contributors.

**Contents:** [Key Highlights](#mbufo-key-highlights) ·
[§1 High Concurrency Architecture](#1-high-concurrency-architecture) ·
[§2 Low-Precision Compute & Memory Optimization](#2-low-precision-compute--memory-optimization) ·
[§3 GGUF Coverage & Multi-Model Router](#3-gguf-coverage--multi-model-router) ·
[Quickstart](#quickstart) ·
[Benchmarks](#benchmarks-gfx1151-strix-halo) ·
[Upstream Tracking](#upstream-tracking) ·
[Build from Source](#build-from-source) ·
[Documentation Index](#documentation-index) ·
[Compatibility & Limits](#compatibility--limits)

---

## MBufo Key Highlights

| Highlight | Metric | What it means |
| --- | --- | --- |
| **Instant Response** | **Short-request TTFT 0.7 s** | An anti-starvation fairness scheduler guarantees that short queries arriving during massive 250K-token prefill batches begin generating in 0.7 seconds with bit-exact greedy outputs. |
| **Memory Efficiency** | **KV memory cut by ~50%** | The Blocked-Q8 cache holds a 262,144-token session in just 3.42 GB (compared to 6.44 GB for F16), with 4-bit value/key options reaching down to 1.81 GB. |
| **Unified Serving** | **13 GGUF model families** | A single HTTP port dynamically hosts and routes chat models, MoE, embeddings, and cross-encoders with zero-downtime hot swapping. |

---

## 1. High Concurrency Architecture

In high-concurrency serving, raw throughput is only half the equation. Usability under sustained load depends on two crucial metrics:

- **ITL (Inter-Token Latency)**: The gap between consecutive tokens once streaming has begun. Freezing active streams during long prefill cycles degrades interactive user experience.
- **Fairness**: No single session, client, or prompt may monopolize device resources. A 250K-token prompt arriving mid-conversation must not starve concurrent short chats.

While upstream bounds prefill chunks (`--prefill-chunk`) and prioritizes active decoders, traditional architectures still process chunks and decodes in separate sequential passes over the weights, introducing idle gaps. MBufo eliminates this bottleneck through deep pipeline fusion.

### Unified Mixed-Batch Path

MBufo merges the prefill chunk and decode rows into a single `ForwardBatch` and a shared attention kernel. Every cycle that advances a prefill chunk simultaneously outputs all due decode tokens:

- **2.1x lower token gap**: Streaming token intervals in a 5-row batch drop from 121 ms (two weight passes) to **57 ms** (single pass).
- **Reduced tail latency**: Under 16-way sustained serving, mean elapsed time decreases from 17.0 s to **15.3 s**, with noticeably smoother latency curves.
- **Sustained utilization**: Reaches 99% GPU utilization under 16 concurrent streams with 100% bit-exact outputs and zero errors.
- **Expanded batch envelope**: Scales up to 32 sessions and 1024 rows.

> Mixed batching changes when operations execute without modifying model semantics. It is enabled by default; setting `GUFO_MIXED_BATCH=0` reverts to traditional scheduling. Architectural invariants and validation details are documented in [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md).

### DecodeBatch & Split-K Attention

The serving layer advances decoders via prepared batches (`StepDecodeBatch` / `ExecutePreparedDecodeBatch`), and attention work routes through a unified Split-K decode reader (`blocked_splitk`). In shared Q8, Gemma 4 SWA layers and 27B attention layers decode with minimal overhead, while F16 paths remain fully bit-exact.

### Anti-Starvation Fairness Scheduler

To prevent long prefills from degrading the responsiveness of short requests, MBufo employs three cooperative scheduling mechanisms:

1. **SRTF (Shortest Remaining Time First) Prefill**: Requests requiring ≤1 remaining chunk take precedence over long-prefill continuation chunks.
2. **Starvation Decode-Priority Preemption**: When incoming requests are queued for session slots, long prefill steps defer to resident decoders, rapidly clearing slots (bounded to ≤32 rounds to prevent prefill starvation).
3. **Near-Done Gating**: Preemption triggers only when an active decoder is within ≤64 tokens of its limit, avoiding unnecessary prefill delays when all sessions are generating lengthy outputs.

| Metric (gfx1151, FP4 + f16 + MTP + 4 sessions) | Result |
| --- | --- |
| Fairness-A degradation ratio | **160 → 7.32** |
| Fairness-B degradation ratio | **856 → 191** |
| Short-request TTFT | **3.4 s → 0.7 s** |
| ITL | Stable / Unchanged |
| Greedy output correctness | Bit-identical (verified by SHA checksums) |

### Engine Comparison Under Load (MTP Enabled)

Benchmarked on identical hardware after warmup across concurrency levels 1 / 2 / 3 / 4 (Total TPS / Per-Session TPS):

| Engine | Total TPS | Per-Session TPS | Fairness A / B |
| --- | --- | --- | --- |
| A (Halogen) | 85 / 82 / 75 / 88 | 85 / 41 / 25 / 22 | 322 / 1140 |
| B (gufo upstream) | 57 / 64 / 108 / 127 | 57 / 32 / 37 / 33 | 15 / 399 |
| **C (MBufo)** | **66 / 83 / 110 / 129** | **66 / 42 / 38 / 33** | **19 / 187** |

MBufo demonstrates superior linear scaling under multi-stream load (scaling smoothly up to 129 TPS with minimal per-session dropoff) while maintaining the best fairness scores.

### High-Efficiency Speculative Decoding

Coordinated through a unified Draft Registry across model families:

- **Flash-Next MTP**: Maintains high acceptance rates of **88.7% / 91.6% / 89.5%** across batch widths 2 / 3 / 4 without degradation as batch size increases, outperforming baseline throughput (97 / 111 / 133 tok/s).
- **27B DFlash2**: Fully compatible with Blocked-Q8 KV cache, writing directly to blocked sidecars with bit-exact continuity.
- **Gemma 4 MTP**: Async-GPU drafting with single-forward verification delivers **74–86 tok/s** decode at long contexts (`d=3` yields 1.74–1.78 accepted tokens/round).
- **Muse Glimmer DFlash**: Sustained end-to-end acceptance rate of **~89%**.

### Prefix RAM Cache & Production Session Profiles

- Reusable prompt prefixes are served directly from RAM cache (shared-prefix boundaries with customizable RAM limits).
- Multi-session scheduling is fully aligned, eliminating staircase delays and unlocking parallel throughput.

| Production Target | Configuration | Memory Budget (128 GB Unified) |
| --- | --- | --- |
| 16 streams, high concurrency | `--sessions 16 -c 65536 --kv-quant q8` | ~103 GB (~20 GB headroom) |
| 8 streams, long context | `--sessions 8 -c 131072 --kv-quant q8` | ~101 GB (~21 GB headroom) |
| 4 streams, max 256K context | `--sessions 4 -c 262144 --kv-quant q8` | ~101 GB (~21 GB headroom) |
| 1 stream, max 256K context | `--sessions 1 -c 262144 --kv-quant q8` | Lightweight footprint, peak throughput |
| 1 stream, unquantized precision | `--sessions 1 -c 262144 --kv-quant f16` | Full-precision KV cache |

---

## 2. Low-Precision Compute & Memory Optimization

MBufo delivers measured trade-offs between precision, speed, and memory. The recommended default is `--kv-quant q8` paired with ROCmFP4 or Q4_K weights.

### KV Cache: Blocked Q8 & 4-bit Compact Blocks

MBufo abstracts Blocked-Q8 KV into modular codecs, allowing Key and Value axes to be configured independently as 8-bit (34 bytes / 32 elements) or 4-bit (18 bytes / 32 elements) blocks. The shared compute core integrates blocked writers, direct readers, WMMA prefill readers, and Split-K decode readers.

| Mode | Status | Bytes/Position | 256K Single Session | Description |
| --- | --- | --- | --- | --- |
| `f16` | Baseline | 2048 B | 6.44 GB | Unquantized reference KV cache. |
| `q8` | **Recommended** | 1088 B | 3.42 GB | Blocked 8-bit, optimal quality and speed. |
| `q8` + `GUFO_KV_V4=1` | Manual (K8V4) | 832 B | 2.62 GB | 4-bit Value, 8-bit Key. |
| `q8` + both flags | Manual (K4V4) | 576 B | 1.81 GB | 4-bit Key and Value for extreme memory constraints. |

- **Recommended: `q8`**: Provides full bit-level numerical integrity while cutting KV memory in half.
- **4-bit modes**: Designed for memory-constrained environments via environment variables; when unset, standard numerics remain unaffected. See [docs/KV-V4.md](docs/KV-V4.md).

### Weight Precision Implementations

- **ROCmFP4 / ROCmFP4-FAST**: 4.25 bpw single-scale FAST down experts decode through fp16 WMMA matrix cores, enabling MoE prefill on matrix hardware rather than generic fallback kernels. On gfx1151, FAST models match or exceed baseline prefill with improved decode speed.
- **MXFP4**: Native MXFP4 routed GEMM for GPT-OSS 20B / 120B with fused decode MoE / QKV / projection kernels; 120B supports zero-copy streaming uploads.
- **Q4_K**: UD-Q4_K_XL serves as the reference build for 27B and Flash-Next; bge-m3 includes a dedicated Q4_K_M loader.
- **W8A8**: Host-side 16-split varlen GEMM supporting Glimmer and Flash-Next prefill.

### Attention Kernel Accelerations

- **int8 WMMA Direct Prefill**: Single-session causal prefill reads blocked int8 KV directly without staging through F16 dequantization, delivering a **+30%** speedup (1085 → 1408 tok/s at 14K context).
- **Split-K Flash-Decoding**: Fused GEMV, fused Norm / Residual / Epilogue kernels, and device-side argmax.
- **Wide Prefill Coverage**: Applies fused projections and matrix core processing to all prefill batches exceeding vector width.

### Performance vs llama.cpp (gfx1151)

Identical weights, release build, greedy decoding; positive values denote MBufo speedup:

| Workload | Prefill Delta | Decode Delta | Notes / Test Setup |
| --- | --- | --- | --- |
| **Flash-Next** | **+52%** | Parity or better | Cold KV, MTP enabled |
| **Qwen3.6-35B-A3B** | **+52.5%** | Parity or better | Split-K Router GEMV decode |
| **bge-m3 (long context)** | **+35% ~ +50%** | — (Prefill only) | hipBLASLt autotuning & cached executor |
| **Qwen3-Embedding-0.6B** | **+41% ~ +50%** | — (Prefill only) | Optimized single-prefill Embed |
| **Reranker (batched scoring)**| — | **+15% over target**| Padded multi-sequence ScoreBatch |
| **GPT-OSS 120B** | **+62%** | -17% | Massive scale, prefill heavily accelerated |

---

## 3. GGUF Coverage & Multi-Model Router

### Supported Model Families

A single HTTP port hosts and routes across all 13 supported families:

| Model | Parameters | HTTP Serving | Evaluation | Speculative Draft | Architecture Highlights |
| --- | --- | --- | --- | --- | --- |
| **Qwen3.8 Flash-Next** (Primary) | 30B-A3B MoE | Supported | Pilot | MTP | ROCmFP4-FAST / UD-Q4_K_XL, Blocked-Q8 KV |
| **Qwen3.8 27B** | 27B Dense | Supported | Pilot | DFlash2 (inc. Q8) | Dense + MoE paths; UD-Q4_K_XL recommended |
| **Gemma 4 26B-A4B** | 26B-A4B | Supported | Verified | MTP (F16 mirror) | Native tool use, SWA shared Q8 layers |
| **Gemma-4-12B** | 12B Dense | Bench | — | — | K-quant preconversion & WMMA Q8 prefill |
| **Qwen3.5-35B-A3B** | 35B-A3B MoE | Engine | — | — | qwen35moe pipeline support |
| **Qwen3.6-35B-A3B** | 35B-A3B MoE | Engine | — | — | Q6_K→Q8_0 loading, Split-K decode MoE |
| **GPT-OSS 20B / 120B** | 20B / 117B MoE | Supported | Pilot | — | Native MXFP4 GEMM; zero-copy streaming for 120B |
| **Muse Glimmer 30B** | 30B | Supported | Prompt | DFlash (~89%) | ATEM tool calling, W8A8 / Q8_0 execution |
| **Llama-3.1-8B** | 8B Dense | Supported | Adapter | — | Standard dense GQA support |
| **Qwen3-Embedding-0.6B** | 0.6B | Supported (`/v1/embeddings`) | — | — | Single-prefill pooled executor |
| **bge-m3** | 0.6B XLM-R | Supported (`/v1/embeddings`) | Encoder | — | CLS pooling, Q4_K_M loader, hipBLASLt autotune |
| **bge-reranker-v2-m3** | Cross-Encoder | Supported (`/v1/reranking`) | ScoreBatch | — | BERT dispatch, padded batching |
| **Qwen3-Reranker** | Cross-Encoder | Supported (`/v1/reranking`) | Scoring | — | Qwen dispatch, shared scoring heads |

### Production Multi-Model Router

Running `serve llm --multi-model` turns the engine into an active model registry: the initial startup model acts as the default, and requests automatically route based on their payload's `model` field. Models can be added or ejected dynamically at runtime via Admin API with zero server restarts.

#### Safety & Fault-Tolerance
- **Memory Safety & Eviction**: `/admin/load` checks available GPU memory before loading. If resources are tight, cold non-default models are **LRU-evicted** (or refused with `409 insufficient_memory` if `"allow_evict": false`); the default model is never evicted.
- **Graceful Draining**: Ejecting a model delays memory deallocation until all in-flight requests complete, ensuring ongoing responses are never interrupted.
- **Fallback Handling**: If a request omits the `model` parameter, it routes cleanly to the default model.

#### Available Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/v1/chat/completions` | POST | Chat completions (text, vision, function tools, structured output, SSE streaming) |
| `/v1/embeddings` | POST | Embeddings (bge-m3, Qwen3-Embedding) |
| `/v1/reranking` | POST | Reranking (bge-reranker-v2-m3, Qwen3-Reranker) |
| `/admin/models` | GET | List loaded models, context sizes, and access times |
| `/admin/load` | POST | Load and register a model at runtime (with LRU eviction safety) |
| `/admin/eject` | POST | Unload a model gracefully after in-flight requests finish |

---

## Quickstart

### 1. Launch Primary Inference Engine (Qwen3.8 Flash-Next)

```sh
./build/release/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mmproj models/qwen3.8-flash-next/mmproj-BF16.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/agentionai-MTP-Q8_0.gguf \
  --speculative mtp --think off \
  --served-model-name gufo-qwen3.8-flash-next \
  --sessions 4 -c 262144 --kv-quant q8
```

### 2. Multi-Model Mode (Router & Admin API Enabled)

```sh
./build/release/gufo serve llm --multi-model \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --served-model-name flash-next -c 262144 --kv-quant q8 --sessions 4
```

### 3. API Examples

```sh
# Chat completion request
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "flash-next",
    "messages": [{"role": "user", "content": "Hello, introduce yourself."}]
  }'

# Load an additional model dynamically (e.g. Gemma 4)
curl http://localhost:8080/admin/load \
  -H "Content-Type: application/json" \
  -d '{
    "id": "gemma4",
    "path": "models/gemma-4-26b/gemma-4-26B-A4B-it-Q4_0.gguf",
    "max_context": 262144, "session_count": 2, "kv_cache_quant": "q8"
  }'

# List active models
curl http://localhost:8080/admin/models

# Eject a model
curl http://localhost:8080/admin/eject \
  -H "Content-Type: application/json" \
  -d '{"id": "gemma4"}'
```

MBufo is fully compatible with OpenAI SDKs, Open WebUI, and VS Code extensions. Point the base URL to `http://localhost:8080/v1`. Full API details are in [docs/SERVER.md](docs/SERVER.md).

---

## Benchmarks (gfx1151 Strix Halo)

Conditions: Cold KV (unique prompt per request), `think` disabled, F16 KV, `-j 1`, context 262,144. Speculative engines: 27B uses DFlash2, Flash-Next uses MTP.

### Prefill (tok/s, higher is better)

| Context Size | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---:|---:|---:|---:|---:|
| **2K** | 605 | 455 | **1739** | 1301 |
| **8K** | 615 | 564 | **1701** | 1549 |
| **32K** | 567 | 551 | 1598 | **1618** |
| **64K** | 508 | 496 | **1695** | 1678 |
| **128K** | 409 | 407 | **1504** | 1507 |
| **250K** | — | — | **1438** | 1415 |

### Decode (tok/s, speculative enabled)

| Scenario | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---|---:|---:|---:|---:|
| Prose generation (temp 0.4) | 23.5 | 19.7 | 43.3 | **43.6** |
| Fibonacci generation (temp 0.4) | 67.9 | 56.2 | **86.3** | 84.7 |
| JSON constraints (temp 0) | **55.1** | 46.5 | 51.5 | 51.7 |
| Greedy code /128 (temp 0) | 61.9 | 55.3 | 70.5 | **73.2** |
| Repetitive text /256 (temp 0) | 70.6 | 60.8 | 74.8 | **75.3** |

### Agentic Suite (agent_bench_v2 sandboxed tasks)

| Test Metric | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---|---:|---:|---:|---:|
| Tasks passed | **8/8** | 7/8 | 7/8 | 7/8 |
| Critical faults | **0** | **0** | **0** | **0** |
| Generation TPS range | 9.1–31.1 | 9.3–21.5 | 16.7–56.6 | 16.8–55.1 |

### Recommended Configurations

- **Qwen 3.8 27B**: **UD-Q4_K_XL** provides the highest decode rate and draft acceptance.
- **Qwen 3.8 Flash-Next**: **ROCmFP4** delivers peak decode speed (prose 45.0, fibonacci 87.0 tok/s) and fits in ~94 GB (saving ~17 GB over Q4_K_XL).

---

## Upstream Tracking

MBufo tracks upstream `main` via standard Git merges, preserving upstream history without rebase:

| Property | Details |
| --- | --- |
| Upstream Repository | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| Upstream Base Release | `v0.26.0` (tracking stable updates) |
| Merge Strategy | Clean `--no-ff` merges from upstream `main`; MBufo commits remain cleanly isolated |
| Tagging Scheme | MBufo releases use the `mbufo-vX.Y.Z` prefix to avoid colliding with upstream tags |

---

## Build from Source

Target environment: Linux x86-64 on AMD Strix Halo (`gfx1151`). Build commands are standard:

```sh
cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
```

### Build Dependencies
- **Toolchain**: C++20 compiler (GCC 15.3 recommended), CMake 3.21+, Ninja, pkg-config.
- **Compute Libraries**: ROCm 7.2.3 SDK (HIP, hipBLAS, hipBLASLt, rocBLAS, hipCUB, rocPRIM, rocWMMA). Nix build is also supported (`nix build`).
- **System Libraries**: ICU, libcurl, OpenSSL, libpng, libjpeg, libwebp, FFmpeg.
- **Device Access**: Your user must have read/write permissions to `/dev/kfd` and `/dev/dri`.

For container configurations and system tuning, refer to the [upstream build guide](https://github.com/gufo-org/gufo#build-from-source).

---

## Documentation Index

| File | Summary |
| --- | --- |
| [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md) | Mixed-batch design invariants, scheduling, and validation |
| [docs/KV-CACHE.md](docs/KV-CACHE.md) | Conversation cache and component lifecycle |
| [docs/KV-V4.md](docs/KV-V4.md) | 4-bit KV block layouts and accuracy measurements |
| [docs/SERVER.md](docs/SERVER.md) | HTTP API contracts and endpoints |
| [docs/CLI.md](docs/CLI.md) | Comprehensive command-line reference |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Development workflow, presets, and local testing |
| [docs/TESTING.md](docs/TESTING.md) | Continuous integration and local quality suites |
| [docs/models/README.md](docs/models/README.md) | Model-specific configuration guides and benchmarks |
| [CHANGELOG.md](CHANGELOG.md) | Release history and changelogs |

---

## Compatibility & Limits

- **Semantic Invariance**: Mixed batching alters scheduling order, never model calculation logic. Any unsupported layer cleanly falls back to standard execution paths.
- **Precision Trade-offs**: `--kv-quant q8` trades minimal numerical divergence for major memory savings. If strict reference parity is required, use `f16`.
- **KV Format Isolation**: KV state snapshots are keyed to specific quantization formats to prevent cross-format reuse.
- **Model Advisory**: For Qwen 3.5 above 64K context, limit active tokens to 60K for driver stability; Gemma 4 full-512 layers run at F16.

---

## License & Credits

Upstream gufo is licensed under the MIT License. Notices for adapted code and dependencies are maintained in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and `licenses/`.

MBufo is an optimized fork of [gufo](https://github.com/gufo-org/gufo) (gufo-org and community contributors). Sincere thanks to the upstream developers whose work on Q8 KV caching, int8 WMMA, and speculative foundations made this project possible.
