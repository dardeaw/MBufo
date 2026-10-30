# GDN chunked kernel design (draft)

Status: draft, not implemented, not validated. The quality gate needs a GPU
session; everything below that can be settled without one has been settled in
[the feasibility study](EXPERIMENTS.md#gdn-chunked-feasibility-october-31-2026)
and [the precision yardstick](EXPERIMENTS.md#float16-gdn-against-an-accepted-yardstick-october-31-2026).

## Target

Replace `GdnRowSplitKernel` with a chunked form for prefill batches only:

- 36 linear layers, about 1.78 ms each, 8.1% of prefill kernel time, currently
  FP32 and serial over tokens with no matrix-core work.
- Decode, verification and speculative batches keep the serial kernel: a
  triangular solve cannot be amortized at one to eight rows.
- Gate: `WidePrefill(n_tokens)` and `GUFO_GDN_CHUNK` above 0.

## Recurrence (the contract)

State `S[j][i]`, `j` over the value dimension and `i` over keys, one `128x128`
matrix per value head. Per token:

```
S *= decay_t
u[j]     = sum_i S[j][i] k_t[i]          (the kernel also folds p = S q here)
delta[j] = b_t (v_t[j] - u[j])
S[j][i] += delta[j] k_t[i]
o[j]     = sum_i S[j][i] q_t[i] * q_scale
```

## Chunked form (verified)

Per chunk of `C` tokens with `S_0` entering it:

```
P_t     = prod_{u<=t} decay_u
M[t][s] = b_t (P_t / P_s) (k_t . k_s)          for s < t, strictly lower
delta   = solve(I + M, b * (v - P * (S_0 K^T)))
O_t     = P_t (S_0 q_t) + sum_{s<=t} (P_t / P_s) (k_s . q_t) delta_s
S_C     = P_C S_0 + sum_s (P_C / P_s) delta_s k_s^T
```

Checked in numpy against a float64 serial truth, `d = 128`, 48 value heads,
1,734 tokens: float64 chunked 5e-16, float32 chunked 1.9e-7 (serial float32
1.7e-7), float16 operands and state 4.4e-4.

## Kernel plan

One launch per layer, block = (value head, 32 value rows), which gives 48 * 4 =
192 blocks per layer and keeps each block's state slice in LDS:

| Piece | Shape | Storage |
| --- | --- | --- |
| state slice, float16 operands | 32 x 128 | 8 KB LDS |
| `k` chunk, `q` chunk | 32 x 128 each | 8 KB + 8 KB LDS |
| `M` / triangular solve | 32 x 32 float32 | 4 KB LDS |
| `delta`, `rhs` | 32 x 32 float32 | 4 KB registers/LDS |

About 32 KB LDS per block, 2 blocks per CU.

Inside the block, per chunk:

1. `P` from the chunk decays (a 32-element scan, one wave).
2. `K K^T` and `Q K^T` (32x128 by 128x32) on WMMA float16 with float32
   accumulate, then scale by `P_t / P_s` into `M` and the output coupling.
3. State terms: `S_0 K^T`, `S_0 Q^T` (32x128 by 128x32) on the same fragments.
4. Unit-lower solve of `(I + M) x = rhs`, 32 forward steps of a 32-wide FMA per
   value row, kept in registers.
5. Output rows through the existing epilogue path (per-head RMSNorm, sigmoid
   gate) before the block writes them out.
6. `S_C = P_C S_0 + Delta^T K` (32x32 by 32x128) in place.

The state lives in LDS across chunks, so the loop never round-trips it to
global memory. No cross-block synchronisation is needed: value rows are
independent, as they already are in `GdnRowSplitKernel`.

## Checkpoints

`GdnCheckpoint` writes the state at an arbitrary token, and MTP rollback
restores at arbitrary positions. A chunked kernel only commits at chunk
boundaries.

- Rule: the executor takes the chunked path only when
  `checkpoint.tokens % C == 0`; otherwise the pass runs the serial kernel.
- MTP verification batches are already below the wide-prefill threshold, so they
  keep the serial kernel and their per-token rollback unchanged.

## Switches

| Variable | Meaning |
| --- | --- |
| `GUFO_GDN_CHUNK` | 0 (default, off), 8, 16 or 32 tokens per chunk |
| `GUFO_GDN_F16` | 0 (default) keeps float32 operands, 1 selects the float16 matrix-core operands that cost 4.4e-4 |

`GUFO_GDN_F16=1` is the only mode that can reach matrix cores; the float32 mode
has the same FLOPs as the serial rule (3 d^2 per token) and would only add the
solve, so it exists to isolate the reformulation from the precision change
during validation.

## Validation protocol (needs the GPU, about two hours)

1. Layer unit check against `ReferenceModel` GDN with random weights: relative
   L2 at or below 1e-6 for float32 operands, 5e-4 for float16.
2. `--validate-prefill 1024`: exact for float32 operands; for float16 report
   mean KL and top-1 agreement against the sparse-attention bar (5.8e-4,
   256/256).
3. Deep context: KL against the float32 recurrence at 32K and 128K.
4. Prefill throughput: eight alternating pp2048 pairs at `-r 3`; fresh-process
   variance on this host is 1.5 to 2.6%, so a smaller delta is not a result.
5. Decode unchanged: MTP draft counts and TokenTrace hashes for tg128.

## Expected gain

The row-split kernel is 64 ms of 1.109 s prefill kernel time. Two and a half to
three times on that block is about 40 ms, roughly 3.6% of prefill. Decode is
untouched, so the service-level gain shows up only in time to first token on
prompts above the wide-prefill threshold.
