# Inkling-2-Mini

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md)

MBufo runs Inkling-2-Mini (thinkingmachines, hybrid sliding + loop architecture)
on Linux gfx1151 (Strix Halo): prefill, 3-pass loop execution with fused
message gates, per-head dynamic short-convolution, Top-8 MoE, and single-stream
decoding. Model code lives in
[`src/models/inkling`](../../../src/models/inkling).
Status is **experimental**: text chat works end to end; long prompts are
supported via chunked prefill.

## Model acquisition

| Artifact | Pinned identity |
| --- | --- |
| thinkingmachines Inkling-2-Mini GGUF (h6-lbf16-UD-Q4_K_M) | operator-supplied file |
| Reference HF checkpoint (ground truth only) | local `inkling-2-mini-hf` safetensors |

The operator obtains the checkpoint and accepts its terms independently.
MBufo distributes engine code and binaries, without weights, tokenizer
payloads, converted weights, or private authorization documents. Build,
installation, and server startup do not download this model.

The h6-lbf16-UD-Q4_K_M file contains zero Q4_K blocks: 40x NVFP4 experts
(blk.0-19, ~10.68GB) plus 154x Q8_0 / F32 / BF32 tensors. The engine converts
NVFP4 experts once at load time and runs the dense path in Q8 (load ~105s,
~30GB VRAM). The F16 variant needs ~38GB and does not fit alongside the
desktop session, so Q8 is the only supported load path.

## Serve

Build with `nix build` (or the repo CMake `build/inkling` target) and use
`result/bin/mbufo`. The model registers under id `inkling-2-mini`
(`model_mapping.json`, arch `inkling2`):

```sh
mbufo serve llm \
  --model <models-dir>/inkling-2-mini/Inkling-2-Mini-h6-lbf16-UD-Q4_K_M.gguf \
  -c 4096 --sessions 1 -p 8742 -i 127.0.0.1
```

Required environment for single-instance tests (production service must be
stopped first; `MBUFO_CONFIG_FILE` overrides `--model` and must be unset):

```sh
unset MBUFO_CONFIG_FILE
export LD_LIBRARY_PATH=/opt/rocm/lib MBUFO_KV_WMMA=1 MBUFO_KV_I8=2
```

Single-token smoke test (`Say OK.`, greedy):

```sh
curl -s http://127.0.0.1:8742/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Inkling-2-Mini","messages":[{"role":"user","content":"Say OK."}],"max_tokens":1,"temperature":0}'
```

Expected first token is `200001` (`<|message_model|>`).

## Architecture notes (verified against transformers reference)

- Sliding layers: single-query bank mix, static short-conv, sliding-window
  attention with log-scaled QK clips and sinks, gated output proj.
- Global layers (blk 5/11/17/23): 3-query bank mix, per-stream input norms,
  dynamic short-conv whose taps are projected from the **post-norm** K/V
  streams, and K-major tap layout read dim-major (GGUF file order).
- Loop core (blk 6-17, 3 passes): per-layer message attention over the
  previous pass bank (4 streams: q/k/v/mlp), message gates fused **before**
  the input norms (`tanh` on both forget and message gates), no post-norm
  fusion, delta projections after fusion.
- MoE: sigmoid Top-8 router with `e_score_correction_bias`, per-expert dense
  GEMM, situ-fused SwiGLU, weighted scatter-add.

See [Quality](QUALITY.md) for the parity evidence behind each of these.
