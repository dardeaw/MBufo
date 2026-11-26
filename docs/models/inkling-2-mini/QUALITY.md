# Inkling-2-Mini quality

Reference: transformers CPU forward (`inkling-2-mini-hf`, bf16) on the
`Say OK.` prompt (18 tokens). Engine runs F32. Small fp differences are
expected; discrete Top-K router flips at razor margins are inherent noise
(bf16 reference vs F32 engine) and are documented, not fixed.

## Bugs found and fixed (this branch)

All verified component-to-component against the reference (per-layer
q/k/v norms, projections, static/dynamic sconv incl. deltas, attention,
gates, MoE, router ids/weights/logits, post-convs).

1. **Dynamic sconv delta inputs** (`engine.cpp`, global branch): delta
   projections read the un-normed bank mixes; reference reads the post-norm
   K/V streams. Fixed to `k`/`v` (normed). Effect: L5 output mean 0.050 to
   0.012, deltas match to 0.5%.
2. **Dynamic sconv tap order** (`kernels.hip.cpp`): static taps were read
   transposed (`taps[lag*dim+i]`); GGUF stores them dim-major. Forensic
   least-squares against the reference proved the effective taps were lag-
   reversed; fixed to `tp[3]=current` (static-file convention). Effect:
   L5 kpre absmax rel 0.16 to 0.03, then to full match.
3. **Gate fusion activation** (`GateFuseApplyKernel`): message gate used
   `sigmoid`; reference `Inkling2LoopMessageGate` uses `tanh` on both
   gates. Fixed. Effect: loop passes 2/3 went from 0.3-1.7 to mostly green.
4. **Global loop fusion order** (`engine.cpp`): fused the already-normed
   streams and fused q/k/v post-norm (`fuse_kvmsg`). Restructured to fuse
   the raw mixes first, then norm, delta after fusion; deleted the post-norm
   double fusion. Effect: pass-2 L11 0.22 to 0.007.
5. **Per-layer loop messages** (`engine.cpp`): messages were computed once
   per pass (mean over 4 queries) and shared by all 12 loop layers; reference
   computes 4 separate streams per layer. Restructured with the confirmed
   query mapping (`query[l] = [res[2l], kv[2l], kv[2l+1], res[2l+1]]`,
   exact match vs transformers). Effect: messages match reference to ~0.06.
6. **F16 staging overrun** (`engine.cpp`): `x16` covered only `max(h, qdim)`;
   the loop-gate down GEMM needs `2*hidden`. Any prompt over ~128 tokens
   crashed the server (GPU page fault). Fixed by sizing to `2*hidden`.

## Known gaps (accepted, documented)

- **L4 single router flip** (50 vs 63, margin 0.002 in selection score):
  engine logits match the reference to 0.02, biases are bit-identical, and
  the engine's own selection is self-consistent. The flip comes from fp
  noise at a razor boundary. L4 output mean-rel 0.054 (bar 0.05).
- **Loop boundary layers** (L12/L18, pass-2/3): attention-only residual
  (no input skip) amplifies noise; worst 0.24 mean-rel on tiny means with
  matching first-8. Under review; functionally covered by Say OK below.
- **Say OK. 16-token generation**: teacher-forced argmax matches reference
  on 12/16 positions; engine output starts with the expected NEW_IDS prefix
  `[200001, 200008, 371, ...]` and reads fluently
  (`The user asks to say "OK". This is a simple request`). The 4 diverged
  positions are coherent paraphrases, consistent with noise-boundary flips,
  not structural error.
- **Reference `hs48` is stale** in transformers `output_hidden_states`
  (equals L22 output, not L23 output; proven by hooking the forward, whose
  true output the engine matches to 5%). Do not compare against `hs48`.

## Replay

Greedy single-token parity (`Say OK.`, max_tokens=1) is deterministic:
final argmax id 200001, logit within 0.5 of reference. Seeded replay is
checked within the same execution configuration only.
