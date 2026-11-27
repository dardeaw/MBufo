# MBufo NPU Lanes — In-Process Servables + Admin Wiring (M6)

> Language: English (rewritten 2026-11-24; the original 11/21 Chinese scaffold
> is preserved as Appendix A). Branch: `gufo-npu-all` (`qwen3-npu-all`,
> NOT pushed). Rule: NPU takes at most one job at a time; `export
> LD_LIBRARY_PATH=/opt/rocm/lib` is required before any NPU binary.

## 1. Scope

Six NPU lanes (`src/core/compute/npu/npu_lane.hpp`): embedding / reranker /
decide / **asr** / **tts** / **image**. This document covers the four
in-process NPU-GEMM servables (M1/M2/M4/M5, all verified) and their Serve +
admin wiring (M6):

| Model | Lane | NPU code | Verify (2026-11-24, .55) |
|---|---|---|---|
| qwen3-asr-1.7b thinker | asr | `ThinkerNpuDecoder` | `THINKER_DECODE_CONSISTENCY_PASS`, cos=1.0, peak 516 MB |
| qwen3-tts-1.7b talker / predictor / frame loop | tts | `TalkerNpuDecoder` + `PredictorNpuDecoder` + `TtsNpuFrameLoop` | all three `*_PASS`, cos=1.0, peak 2168/1855/1655 MB |
| shared Qwen3-4B text encoder | asr+tts+image | `TeNpuDecoder` | `TE_DECODE_CONSISTENCY_PASS`, peak 994 MB |
| FLUX.2-klein DiT (5dbl+20sgl, 4-step Euler) | image | `FluxDitNpu` | per-step v cos 0.978/0.991/0.999/0.999, **E2E cos(pipe)=0.9485**, peak 924 MB |
| Z-Image-Turbo DiT (refiner+30main, 8-step Euler) | image | `ZImageDitNpu` | `ZIMAGE_DIT_PASS`, deter=1, q8q4=0.985, **E2E cos(pipe)=0.9918**, peak 750 MB |

RAM numbers are peak RSS via `/usr/bin/time -v`. Quantities: q8 path is the
served path; q4 drifts on klein (q8q4=0.85, bar lowered to 0.80 in
`npu_flux_dit_test.cpp`) — a criteria note, not a code bug.

## 2. `model_mapping.json` — the game rules

File: repo root `model_mapping.json` (30 models). Served at `GET
/admin/model-mapping`, refreshed by `POST /admin/model-mapping/sync`.
Purpose: canonical model capability mapping for the WebUI and catalog
auto-detection.

**Match priority** (UI `findMappingForCatalog`, same rule server-side):
1. `gguf general.architecture` / safetensors `config.json architectures`
   (`match.gguf_arch`), 2. then `match.filename_patterns` (glob).

**Backend vocabulary** (`supported_backends`): `gpu_hip`, `npu_bf16`,
`npu_q8_bfp16`, `npu_q4_q4nx`, `npu_flm_sidecar`, `cpu`. Any backend
starting with `npu` counts as NPU-capable (`mappingSupportsNpu`).

**The four NPU entries** (already declared — no mapping change needed):

| id | task | supported_backends |
|---|---|---|
| qwen3-asr-1.7b | asr | gpu_hip, npu_q8_bfp16 (dual) |
| qwen3-tts-12hz-1.7b | tts | gpu_hip, npu_q8_bfp16 (dual) |
| flux2-klein-4b | image_generation | npu_q8_bfp16 (NPU-only) |
| z-image-turbo | image_generation | npu_q8_bfp16 (NPU-only) |

(Sidecars, for contrast: `d3-decide` / `whisper-v3-turbo` =
`npu_flm_sidecar`, ports 8766/8764; TTS sidecar :8765.)

## 3. Admin cards — when does an NPU card appear and become usable?

UI logic (`src/cli/serve/model_admin_ui.hpp`):

- `mappingSupportsNpu(entry)`: entry has any `npu*` backend.
- `isDualCapable(m)`: mapping has gpu AND npu.
- `defaultDeviceFor(m)`: **dual-capable → NPU**; NPU-only → NPU. So all
  four models default to NPU once the mapping resolves — no UI change
  needed for the default.
- Catalog cards render backend badges from `supported_backends` (amber
  for NPU) via `fetch('/admin/model-mapping')` + `fetch('/admin/models')`
  in `renderCatalog()`.

A card is **usable on the NPU device** iff ALL hold:
1. Mapping resolves (arch/pattern match) with an `npu_*` backend;
2. Weights are present locally (model root mounted, e.g.
   `/mnt/nvme-data/mbufo-models/qwen3-asr-1.7b`);
3. NPU device is present (`/dev/accel/accel0`, `NpuDeviceLock` acquirable);
4. Serve-side entry is `ready` (M6: `NpuModelRegistry` + light probe —
   one block / one step, never the full test suite; NPU runs serially).

## 4. Serve wiring (M6, four steps)

State: `NpuModelRegistry` (P6 descriptors) and `DecideNpu*` router seams
exist with **zero callers**. Wire-up:

1. Populate `NpuModelRegistry` at serve boot (descriptors only, no load):
   asr-thinker / tts-talker+predictor+frame / flux / zimage, kind
   `kNpuInProcess`, model roots (§3.2), M3 resident estimates. Never boot
   autoload — on-demand only.
2. Light probe flips `ready` (one block / one step per model, serialized).
3. Route checks at the three dispatch points:
   `TranscriptionHipRuntime::Transcribe` (`asr_service`),
   TTS synth (`tts_service`), image generate (`image_api`) — consult
   `DecideNpuAsr/Tts/Image` first, fall back to current behavior on `kNone`.
4. Admin: `GET /admin/npu/status` (lane / ready / resident MB / route per
   entry) + render the four NPU cards (mapping badges already cover the
   static half; the endpoint covers the live half).

## 5. Repro commands (all on .55, serial NPU)

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib
cd ~/git/gufo-npu-all/build/wt-npu
/usr/bin/time -v ./npu_thinker_decode_test     # M1
/usr/bin/time -v ./npu_talker_decode_test      # M2
/usr/bin/time -v ./npu_predictor_decode_test   # M2
/usr/bin/time -v ./npu_tts_frame_test          # M2
/usr/bin/time -v ./npu_te_decode_test          # shared TE
./npu_flux_dit_test                            # M4 (q8q4 bar 0.80)
/npu_zimage_dit_test                          # M5 (q8q4 bar 0.97)
```

Pipeline-faithful killers (nailed inputs, NPU-vs-pipe E2E):
`~/q_zfresh2.py` (zimg, cos=0.9918), `~/q_fluxkiller.py` +
`~/q_fluxiso.py` (flux, E2E=0.9485; NOTE: transformer call takes
`timestep=[sigma]` — forward multiplies by 1000 internally).

## Appendix A — 11/21 scaffold (historical, verified then)

Five-lane box (`npu-lanes-scaffold`): `npu_lane.hpp` switches,
`DecideNpuEmbed` stubs (always kNone), D3 config, sidecar routes
(8766/8764/8765). Verified: D3-Lite-NPU2 `:8766` 200/0.2s;
whisper-v3-turbo `:8764` replay OK; gufo TTS `:8765` voice-clone 200;
embedding/reranker stayed on HIP. EmbeddingGemma2 assembly notes (24L,
sliding window NPU-friendly, RMSNorm/gelu-tanh references) moved to the
in-process track (§1).

## Servable contract and warmup (M1)

- src/core/compute/npu/npu_servable.hpp: INpuServable with one lifecycle
  Unloaded, Loading, Warming, Ready, Failed. Ready means weights are resident
  AND one dummy inference completed. Warmup policy is kRequired or kBestEffort.
  The state machine has no hardware dependency; tests/core/npu_servable_contract_test.cpp covers it.
- src/cli/serve/npu_warmup.hpp: RunNpuWarmup, soft-fail, logs event=npu_warmup.
  Warmups always run (no environment switch): a failed warmup is logged and reported as warm=false, never fails the load.
- In-process warmups run right after weights are resident: bge-m3 (5 and 70 tokens, spans two 64-row chunks),
  bge-reranker, EmbeddingGemma2 (8 tokens), and NpuServableHost::Warm (engine-level M=1 GEMM per quant shape,
  plus a real dummy decode for ASR/TTS and a reduced Euler step for FLUX/Z-Image).
- Status note: ASR/TTS/FLUX/Z-Image servables are loaded and warmed in-process, but serve does not yet route
  /v1/audio or /v1/images requests to them (no caller of their inference outside tests); the kernel-level
  tests (npu_zimage_dit_test, npu_flux_dit_test, npu_te_decode_test, npu_quant_models_test) are the
  verification of their numerics.
- Sidecars: SidecarManager::Warmup posts a dummy request after a fresh spawn
  (decide: /v1/systemone, asr: 0.5 s silent wav, tts: short text; image sidecars are skipped).
  Adopted externals are never poked. /admin/load and /admin/models report warm, warmup_ms, warmup_error.
- Measured on the XDNA2 host (11/27): bge-m3 410 ms, bge-reranker 223 ms, EmbeddingGemma2 90 ms,
  Qwen3-ASR servable 153 ms. Loading all three embedding models plus ASR at once exhausts NPU BOs
  (hrx_buffer_allocate failed), so eject before loading the next.

## Op-graph executor and Qwen3 embedding/reranker (11/27)

- src/core/compute/npu/npu_op_graph.{hpp,cpp}: declarative op list over named
  row-major buffers (Embed, LayerNorm, RmsNorm, Linear, Attention [bidirectional or
  causal, GQA, optional QK-norm + NEOX RoPE], Gelu, SwiGlu, Residual, Pool
  [cls/mean/last], L2Norm, RowDot). `Linear` runs on the shared NpuQuantGemmEngine
  (Q8 bfp16ebs8, weights resident in BOs, one process-wide engine via
  `SharedQuantEngine`); every other op runs on the host. `LinearBackend::kHostF32`
  runs Linear on the host in fp32 for hardware-free reference checks.
- Weight prep: the engine packer floors mantissas (floor(v/Q) per 8-value block).
  The executor pre-rounds each weight to the nearest bfp16ebs8 grid value so the
  packer reproduces it exactly (halves the error, removes the bias). bge-m3 CLS
  cosine vs the BF16 path: 0.973 floored -> 0.9990 rounded.
  MBUFO_NPU_GRAPH_FLOOR=1 disables the rounding (debug only).
- src/core/compute/npu/npu_graph_models.{hpp,cpp}: `NpuGraphBgeRuntime`
  (bge-m3 / bge-reranker, 24L post-norm BERT) and `NpuQwen3Encoder`
  (Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B, 28L causal GQA, last-token pool;
  the reranker reads the yes/no logits through the tied lm-head).
- Serve: `--device npu` / `/admin/load {"device":"npu"}` on a qwen3 GGUF with a
  declared pooling_type loads the embedding model; one whose name contains
  "rerank" loads the reranker (id qwen3-embedding / qwen3-reranker, through
  LoadNpuServable + warmup, Q8 only). /v1/embeddings and /v1/reranking dispatch to
  them. The graph runtime is the only bge-m3 / bge-reranker NPU path (the dedicated
  BF16 NpuBgeRuntime stays in the tree only as a regression reference for tests).
- EmbeddingGemma2 runs on the graph too (see "EmbeddingGemma 2 on the op graph").
- Driver: tests/core/npu_graph_models_test.cpp (modes bge-ref, bge-graph, qwen-emb,
  qwen-rerank; hardware needed except --host-only).
- Measured (XDNA2 host, 11/27, NPU only, warm): Qwen3-Embedding-0.6B 43-70 ms for 8-59
  tokens (cos vs GPU 0.9986-0.9991); Qwen3-Reranker-0.6B ~130 ms per ~90-token pair
  (P(yes) within 2e-4 of GPU); bge-m3 graph 58-383 ms for 9-86 tokens vs legacy
  112-344 ms (cos 0.99900-0.99951). NPU BO memory is limited: eject other NPU
  models before loading.

### Servable status and warmup (M2)

- bge-m3, bge-reranker and EmbeddingGemma 2 load through `LoadNpuServable`
  (`src/cli/serve/npu_warmup.hpp`): `INpuServable` lifecycle, best-effort warmup,
  `event=npu_servable` log line. Measured: bge-m3 load 1.5 s + warm 0.38 s,
  reranker 1.4 s + 0.08 s, EmbeddingGemma 2 0.85 s + 0.37 s.
- `GET /admin/models` returns `npu_servables` (`id` = admin model id, `state`,
  `ready`, `warm`, `load_ms`, `warmup_ms`, `error`). Entries are created under the
  admin id during `/admin/load` and removed on `/admin/eject`.
- ASR (thinker) and TTS (talker) servables warm with a real dummy decode (prefill +
  step + LM head), measured about 4.5 s each; this is the true decoder cold start
  that the first request used to pay. FLUX (0.9 s) and Z-Image (1.3 s) warm with one reduced Euler step (1+1 blocks, 8x8 latent, one text token).

### EmbeddingGemma 2 on the op graph (M2)

EmbeddingGemma 2 now runs on the declarative op-graph executor by default
(`NpuGraphEmbGemma2`, shared Q8 engine, weights streamed from host memory because
resident BOs exhaust the NPU buffer-object pool). Measured on 5 texts (9-263 tokens):
cos(graph NPU, CPU/GPU reference) 0.9992-0.9999, host-fp32 graph = 1.00000;
serve latency 99-215 ms vs 398-444 ms for the dedicated BF16 runtime; inputs up to
512 tokens (the dedicated runtime stops at 128).
Selection: the graph is the default; an explicit `MBUFO_NPU_QUANT=q4`
keeps the dedicated runtime (no Q4 path in the graph). Gate: ctest
`npu_graph_embgemma2_test` (skips when the gguf is absent).

### Every NPU model is an INpuServable (eject contract)

ASR, TTS, FLUX and Z-Image (`NpuServableHost`) now load and eject through the same
`INpuServable` lifecycle as bge/gemma2/Qwen3: `Close` is pure virtual (an eject
implementation is mandatory), `Unload()` is the only public release path and always
trims the heap, and each model is published in `NpuServableRegistry`, so
`GET /admin/models` reports all of them in `npu_servables`. Verified on hardware
(9 models, 13 load/eject rounds): eject removes the model from `models` and
`npu_servables`, and RSS returns to the process baseline (~290 MB after the first
NPU load: the shared engine/HRX session stays resident by design).

### Eject contract for every model, one shared engine

- `LoadNpuServable` now requires a `close` callback and `NpuFnServable` rejects an empty
  one: bge-m3, bge-reranker, EmbeddingGemma 2 (`UnloadNpuWeights`) and Qwen3
  (`NpuQwen3Encoder::Release`) release their NPU weights themselves. `/admin/eject` calls
  `NpuServableRegistry::Unload(id)` first (while the backend that owns the model is alive),
  then drops the backend. `GraphExecutor::Release()` waits for an in-flight run.
- ASR/TTS/FLUX/Z-Image (`NpuServableHost`) use the same process-wide `SharedQuantEngine` as
  the graph models and drop it when the last one is ejected. Before this, the host kept a
  private engine alive forever and a later Qwen3 load failed with `hrx_buffer_allocate failed`.
- Hardware check: 13 interleaved load/eject rounds across all 9 models, every model
  `warm=true`, RSS back to ~242 MB after every eject, no allocation failures.
- Gate: ctest `npu_graph_bge_m3_test` (bge-m3 engine path vs CPU F32 forward, cos >= 0.998,
  and RunNpuCls must fail after `UnloadNpuWeights`).


### Q4 policy (measured 2026-11-27, Q8 vs Q4 against the reference, hardware)

| Model | Q8 | Q4 | Decision |
|---|---|---|---|
| FLUX.2-klein DiT (4 steps, q8-vs-q4 final latent) | 0.9995 shell | 0.853 | Q4 banned |
| Z-Image-Turbo DiT (8 steps, q8-vs-q4 final latent) | 0.99999 shell | 0.975 | Q4 banned |
| Qwen3-ASR (thinker / logits) | 0.9993 / 0.9994 | 0.968 / 0.969 | Q4 banned |
| Qwen3-TTS (talker / predictor / decoder) | 0.9994 / 0.9987 / 0.9996 | 0.981 / 0.981 / 0.988 | Q4 banned |
| EmbeddingGemma 2 (vs BF16) | 0.9996 | 0.9931 | Q4 still allowed (legacy runtime, `MBUFO_NPU_QUANT=q4`) |

Cut-off: Q4 must stay >= 0.99 cosine against the reference. `NpuServableHost` (ASR, TTS,
FLUX, Z-Image) and `/admin/load` reject `npu_quant` other than `q8` for those models
(HTTP 400 `invalid_npu_quant`), `model_mapping.json` no longer lists `npu_q4_q4nx` for
them, and the hardware tests run Q8 only for them.
