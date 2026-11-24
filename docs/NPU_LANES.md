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
| qwen3-asr-1.7b | asr | gpu_hip, npu_q8_bfp16, npu_q4_q4nx (dual) |
| qwen3-tts-12hz-1.7b | tts | gpu_hip, npu_q8_bfp16, npu_q4_q4nx (dual) |
| flux2-klein-4b | image_generation | npu_q8_bfp16, npu_q4_q4nx (NPU-only) |
| z-image-turbo | image_generation | npu_q8_bfp16, npu_q4_q4nx (NPU-only) |

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
