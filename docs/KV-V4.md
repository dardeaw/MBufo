# 4-bit value cache (`GUFO_KV_V4`)

Status: **usable, opt-in.** Not the default. Values are stored on a signed
4-bit grid while keys stay 8-bit.

## Why split the two

Keys and values fail differently. A key error moves every attention score, so
it changes *which* positions the model looks at; a value error only perturbs
*what* it reads once the choice is made, and it averages out across the
residual stream. On this model the attention is close to one-hot, which makes
the value route the one place where precision can be spent.

## What changes

| | key block | value block | K + V per position |
| --- | --- | --- | --- |
| `q8` | 34 B | 34 B | 1088 B |
| `q8` + `GUFO_KV_V4=1` | 34 B | **18 B** | **832 B** (−23.5%) |

The value block is 16 bytes of packed nibbles plus the same `f16` block scale,
which moves from offset 32 to offset 16. Element `2b` is the low nibble and
`2b + 1` the high nibble of byte `b`; codes are signed two's complement on a
`max|block| / 7` grid. Keys, the indexer and the pooled block keys are
unchanged.

At 262144 context over the 12 full-attention layers that is 3.42 GB → 2.62 GB
of KV per session, about 800 MB saved per session.

## Enabling it

The engine reads the variable at session construction, so it must be in the
service environment, not the shell that talks to the API.

```bash
sudo install -D -m644 /dev/stdin /etc/systemd/system/gufo.service.d/kv-v4.conf <<'EOF'
[Service]
Environment=GUFO_KV_V4=1
EOF
sudo systemctl daemon-reload
sudo systemctl restart gufo
```

Verify it reached the process, not just the unit:

```bash
PID=$(systemctl show -p MainPID --value gufo)
tr '\0' '\n' < /proc/$PID/environ | grep GUFO_KV_V4
```

It only takes effect with `--kv-quant q8`. `f16` has no blocked layout to pack
and `fp8` already spends its bits on exponent range, so both ignore the flag.

## What was measured

Against plain `q8`, same binary, same 33-token prompt:

| implementation | mean KL | max KL | argmax agreement |
| --- | --- | --- | --- |
| 4-bit values, unpacked (study hook) | 0.02742 | 0.0755 | 32/33 |
| 4-bit values, 18-byte packing | 0.02742 | 0.0755 | 32/33 |

The two rows are the check that matters: packing must be *exact*, not close. A
nibble-order, sign-extension or scale-offset mistake would separate them.

Against the `f16` reference, `q8` alone sits at mean KL 0.0212 with 32/33
argmax agreement, so this mode buys 23.5% of the KV cache for roughly a third
more distribution drift. Greeting-style behaviour batteries (8 seeds) did not
separate the two configurations; that is a weak instrument, so treat real
long-running sessions as the remaining test.

## Turning it off

```bash
sudo rm /etc/systemd/system/gufo.service.d/kv-v4.conf
sudo systemctl daemon-reload
sudo systemctl restart gufo
```

## Reproducing the numbers

```bash
PROBE=build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe
MODEL=/path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
PROBE_ARGS="--model $MODEL --kv-q8 --prompt-file prompt.txt"

$PROBE       $PROBE_ARGS --dump q8.bin
GUFO_KV_V4=1 $PROBE $PROBE_ARGS --dump q8v4.bin
```

Both dumps are `rows * 248320` float32 logits. Compare with mean KL over the
softmax rows plus top-1 agreement. The probe needs the whole device, so stop
any served instance first.

## Scope

`GUFO_KV_V4` is a precision knob, not a correctness switch: there is no repair
path behind it, and no fallback if the value grid is too coarse for a given
prompt. It is documented so it can be measured, not so it can be switched on
blindly.

## 4-bit keys (`GUFO_KV_Q4`)

Keys use the same 18-byte block as values once `GUFO_KV_Q4` is set: 16 packed
nibble bytes (element `2b` in the low nibble, `2b+1` in the high) followed by
an f16 scale at offset 16. One lane owns a whole 32-element block, so both
nibbles of a payload byte are written by the same lane and no read-modify-write
window exists. Keys are range 7 (4-bit signed) in this mode instead of range
127.

The switch only applies to `--kv-quant q8`. With it unset the engine is
byte-identical to a build without the key path (verified by comparing logit
dumps: identical md5). Because keys and values now select their precision
independently, four configurations are reachable: K8V8, K4V8, K8V4, K4V4.

### Measured cost

`qwen3.8-flash-next`, three 400-token contexts, 64 sampled prefill rows and 98
matched-token decode steps per context, mean over the three contexts:

| config | prefill KL | decode KL | decode max KL | decode argmax | KV bytes/pos |
|---|---|---|---|---|---|
| K8V8 | - | - | - | - | 1088 |
| K4V8 | 0.01770 | 0.02139 | 0.258 | 90.5 % | 832 |
| K8V4 | 0.01970 | 0.02093 | 0.120 | 93.2 % | 832 |
| K4V4 | 0.02701 | 0.03351 | 0.411 | 89.8 % | 576 |

Two observations worth keeping:

* K4V8 and K8V4 occupy the same memory (832 bytes per position). Moving
  precision from one axis to the other buys nothing; only K4V4 reduces the
  footprint.
* On a 33-token prompt K4V4 measured the same mean KL as K8V4 (0.0274). That
  coincidence is a cancellation artifact of a tiny sample: the per-position
  errors of the two axes are nearly uncorrelated, some positions improve when
  both are quantised, and the average hides it. At 400 tokens, and in the
  decode phase, K4V4 is clearly worse (+60 % KL, 3.4x tail). Do not gate KV
  precision on short prefill probes.

Keys and values do not add in quadrature at these bit widths; the decode table
is the one to trust.
