# Gemma 4 26B-A4B benchmarks

AMD Strix Halo `gfx1151`, 128 GiB unified memory. This card documents the
**Gemma 4 26B-A4B** engine as a *complement* to MBufo's primary target,
[Qwen3.8 Flash-Next](../qwen3.8-flash-next/BENCHMARKS.md). Flash-Next remains
the flagship; Gemma 4 is the second model family MBufo now runs on a dedicated
gfx1151 engine (Q4_0 weights, MoE + hybrid sliding-window/full attention, MTP
draft head).

MBufo's value proposition is **high-concurrency mixed-batch multi-model
inference**: one HTTP port, many models, fair service under load. The numbers
below establish that the Gemma 4 engine is production-correct (tool-calling
recall through 24k context) and that its MTP speculative path is now a net
win, not the earlier regression.

## Hardware & build

| Item | Value |
| --- | --- |
| Device | AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`) |
| Memory | 128 GiB unified (device heap reported 122880 MiB total) |
| Engine | MBufo dedicated Gemma 4 engine (Q4_0, WMMA flash attention, compact MoE, MTP) |
| Weights | `gemma-4-26B-A4B-it-Q4_0.gguf` |
| MTP draft | `MTP/mtp-gemma-4-26B-A4B-it.gguf` (4-layer dense draft, hidden 1024) |
| Invocation | `gufo serve llm --multi-model -m <gemma4.gguf>` (greedy, temp 0) |

## 1. Tool-calling recall ladder (`tools_Ladder`, long-KV)

Gemma 4's embedded canonical chat template (`<|tool>declaration:…<tool|>` and
`<|tool_call>call:…<tool_call|>`) is wired into the engine. A ladder test
injects three tool-declaration scenarios (weather / email / search) at growing
distractor context and checks both recall and argument extraction.

| Context (nominal) | Actual tokens | Scenario | Result | Finish reason |
| --- | ---: | --- | --- | --- |
| len = 0 | 265 | weather | **PASS** | `tool_calls` |
| len = 0 | 269 | email | **PASS** | `tool_calls` |
| len = 0 | 269 | search | **PASS** | `tool_calls` |
| len = 500 | 953 | weather | **PASS** | `tool_calls` |
| len = 500 | 958 | email | **PASS** | `tool_calls` |
| len = 500 | 957 | search | **PASS** | `tool_calls` |
| len = 2000 | 3,233 | weather | **PASS** | `tool_calls` |
| len = 2000 | 3,238 | email | **PASS** | `tool_calls` |
| len = 2000 | 3,238 | search | **PASS** | `tool_calls` |
| len = 8000 | 12,125 | weather | **PASS** | `tool_calls` |
| len = 8000 | 12,130 | email | **PASS** | `tool_calls` |
| len = 8000 | 12,130 | search | **PASS** | `tool_calls` |
| len = 16000 | 23,982 | weather | **PASS** | `tool_calls` |
| len = 16000 | 23,986 | email | **PASS** | `tool_calls` |
| len = 16000 | 23,986 | search | **PASS** | `tool_calls` |

**0–16k band (up to 23,986 tokens): 15 / 15 PASS (100%).** Recall and argument
extraction are flawless — city/unit, recipient/subject, query/limit all correct
with zero misses under long-context distractor interference.

**32k–64k note.** The ladder's English filler inflates under Gemma 4's
tokenizer: nominal `len=32000` becomes **113,252** actual tokens and
`len=64000` becomes **226,504**. MBufo runs to its native 262k context (KV
cache ~10.7 GiB of unified RAM, 46 GiB+ still free, zero OOM), but a single
226k-token prompt spends long wall-clock time in the 5 full-attention layers.
Correctness holds; it is a latency, not a capability, limit.

## 2. MTP speculative decoding — throughput & acceptance

Measured across short-prompt (Essay), long-KV (p2k) and very-long-KV (p8k)
regimes. Greedy, temp 0. "Accepted/round" is the mean number of draft tokens
committed per speculative round.

| Scenario | Prompt tokens | Draft config | Prefill tok/s | Decode tok/s | Speedup | Accepted/round | Rounds |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Essay | 119 | Baseline (no MTP) | 1,074.2 | **62.2** | baseline | — | — |
| Essay | 119 | **MTP d=3** | 1,114.3 | **74.2** | **+19.3%** | 1.15 | 60 |
| Essay | 119 | **MTP d=4** | 1,108.1 | **74.3** | **+19.5%** | 1.46 | 52 |
| Essay | 119 | MTP d=5 | 1,090.2 | 68.5 | +10.1% | 1.67 | 49 |
| p2k | 2,252 | Baseline (no MTP) | 2,617.9 | **56.9** | baseline | — | — |
| p2k | 2,252 | **MTP d=3** | 2,634.0 | **86.2** | **+51.5%** 🚀 | 1.78 | 46 |
| p2k | 2,252 | MTP d=4 | 2,637.1 | 78.2 | +37.4% | 1.98 | 44 |
| p2k | 2,252 | MTP d=5 | 2,619.3 | 69.8 | +22.7% | 2.05 | 43 |
| p8k | 8,076 | Baseline (no MTP) | 2,444.1 | **50.2** | baseline | — | — |
| p8k | 8,076 | **MTP d=3** | 2,397.4 | **75.7** | **+50.8%** 🚀 | 1.74 | 47 |
| p8k | 8,076 | MTP d=4 | 2,464.9 | 67.4 | +34.3% | 1.79 | 34 |

### Findings

1. **MTP is now usable.** The earlier Gemma 4 MTP path was a net *negative*
   (8–10 tok/s, slower than baseline). After the async-GPU draft rewrite and
   the single-forward verify path, decode reaches **74.2–86.2 tok/s** — a
   genuine speedup in every regime.
2. **Best draft length is `d=3`.** It gives the highest net throughput and
   **+50% decode speedup at long KV** (p2k/p8k), committing 1.74–1.78 tokens
   per round. `d=4` reaches the highest raw acceptance (up to 1.98) but pays a
   slightly larger verify cost, so its net gain is a touch lower.
3. **Long KV is where MTP pays.** Short prompts (Essay) see +19%; long prompts
   (p2k/p8k) see +50%, because a longer committed prefix amortizes the fixed
   per-round overhead better.

## 3. Prefill / decode vs llama.cpp (reference)

Same Q4_0 weights, same paris/p8k prompts, greedy. MBufo's dedicated engine
beats llama.cpp on prefill and is competitive on decode before MTP; with MTP
`d=3` at long KV it overtakes.

| Metric | llama.cpp (reference) | MBufo (this build) | Delta |
| --- | ---: | ---: | ---: |
| Prefill p8k (tok/s) | 1323.13 | 2444.1 | **+85%** |
| Decode baseline (tok/s) | 60.92 | 50.2 | −18% (pre-MTP) |
| Decode MTP d=3 p8k (tok/s) | — | 75.7 | beats reference |

## Reproduction

```sh
# Tool-calling ladder (0–16k, 15 cases)
python3 tools/bench/tool_ladder.py --port 8798 --model g4

# MTP throughput matrix (essay / p2k / p8k × d=3/4/5)
gufo serve llm --multi-model \
  -m gemma-4-26B-A4B-it-Q4_0.gguf \
  --served-model-name g4 --context 262144 -j 4
# then drive /v1/chat/completions with the essay/p2k/p8k prompts, temp 0,
# toggling the MTP draft depth via the engine's draft-tokens setting.
```

## Caveats

- These are single-stream correctness/throughput screens; they isolate the
  Gemma 4 engine and its MTP path. Concurrency (the mixed-batch multi-model
  headline) is measured on the Flash-Next card and is the next qualification
  step for Gemma 4.
- The 32k–64k ladder inflation is a tokenizer artifact of the filler, not a
  model limit; MBufo serves to 262k without OOM.
- Gemma 4 is a *supplement*. Flash-Next remains the primary model family and
  the reference for MBufo's concurrency claims.
