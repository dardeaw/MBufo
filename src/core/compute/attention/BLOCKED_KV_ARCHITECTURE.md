# Shared Blocked-KV Architecture

Status: current implementation, `src/core/compute/attention/`.
Audience: anyone touching KV cache layout, quantization, or attention kernels.

The goal is one definition of *what a blocked KV row is* and one definition of
*how a block is encoded/decoded*, shared by every model that keeps a quantized
KV cache. Models keep their own attention math where they have model-specific
requirements (ring windows, relative bias, sinks, GQA ratios); they must not
keep their own copy of the codec.

---

## 1. Layering

```
L0  blocked_kv_codec.hpp        codecs, byte geometry, view, interface guard
      ├── KvCodecQ8 / KvCodecFp8 / KvCodecQ4
      ├── FloatToHalfBits / HalfBitsToFloat
      ├── BlockedRowBytes / BlockedHeadStrideBytes / BlockOfLane / OffsetOfLane
      └── BlockedKvView<Codec> + MakeBlockedKvView + IsBlockedKvView/RequireKvView

L1  blocked_kv.hip.cpp          generic row operations, per-side block sizes
      ├── QuantizeKvRows        float  -> blocked (34B or 18B blocks)
      ├── DequantizeKvRows      blocked -> f16 (dual-plane, one launch)
      ├── PrepareAttentionKv    fused q/k/v prep + store (roPE path)
      └── AttentionKv           mask-based blocked attention

L2  blocked_splitk.hip.cpp      split-K decode/prefill, templated on codec
    blocked_wmma.hip.cpp        WMMA prefill, templated on codec

L3  model adapters              consume L0 views + L0/L1 decode helpers
      └── e.g. models/inkling/kernels: ring-wrapped, rel-biased,
          sink-augmented, qpg==3 fused GQA kernels
```

Dependency rule: an upper layer may call the layer below it. No layer may
redefine a codec, a scale conversion, or a row-stride formula — those exist
exactly once, in L0.

---

## 2. L0 — block layout

A blocked row is `ceil(kv_dim / 32)` blocks. Each block covers 32 consecutive
elements of one head and carries its own f16 scale.

| Codec         | Block bytes | Payload                    | Scale at | Range | Notes |
|---------------|-------------|----------------------------|----------|-------|-------|
| `KvCodecQ8`   | 34          | 32 x int8 codes            | `+32`    | 127   | symmetric, `rintf` |
| `KvCodecFp8`  | 34          | 32 x E4M3 bytes            | `+32`    | 448   | software emulated; `0x7F/0xFF` are NaN |
| `KvCodecQ4`   | 18          | 16 bytes = 32 signed nibbles | `+16`  | 7     | K8V4 value side only |

Derived geometry (identical formula everywhere):

```
row_bytes  = ceil(kv_dim  / 32) * Codec::kBlockBytes
head_bytes = ceil(head_dim / 32) * Codec::kBlockBytes
```

For the Inkling-2-Mini geometry (`kv_dim = 1024`, `head_dim = 128`):

| Layout | block | row_bytes | head_bytes | bytes / token (K + V) |
|--------|-------|-----------|------------|-----------------------|
| F16    | —     | 2048      | 256        | 4096                  |
| Q8     | 34    | 1088      | 136        | 2176                  |
| K8V4   | K 34 / V 18 | K 1088 / V 576 | K 136 / V 72 | 1664        |

Scale rule: `scale = amax / range`, and an all-zero block uses `scale = 1.0`
so its codes stay exact zeros. `Encode(v)` expects `v` **already divided by
the scale**; it only clamps and rounds.

Nibble packing (`KvCodecQ4`): payload byte `j` holds element `2j` in the low
nibble and element `2j+1` in the high nibble; `-1` encodes as `0xF`
(two's-complement nibble). `BlockedDequant4` and the quantizer agree on this
bit layout, so packing on the store side and unpacking on the read side are
the same diagram.

Scale conversion is `FloatToHalfBits` / `HalfBitsToFloat` with round-to-nearest-
even and a `memcpy` bit-cast. Do not use `__float_as_uint` / `__half_as_ushort`
here: they are not bit-identical across HIP targets and the historical inline
copies disagreed on subnormals.

## 3. L0 — the view and the interface guard

```cpp
template <typename Codec>
struct BlockedKvView {
  using codec = Codec;
  std::uint8_t* data = nullptr;
  std::uint32_t row_stride = 0;   // bytes per token row
  std::uint32_t head_stride = 0;  // bytes per head inside a row
};
```

`MakeBlockedKvView<Codec>(data, kv_dim, head_dim)` is the only sanctioned way
to build one: it derives both strides from the codec, so a wrong stride is a
single-point failure instead of silent misaddressing.

Interface enforcement: `IsBlockedKvView<V>` is a type trait and
`RequireKvView(v)` static_asserts it with a message telling the caller to
build a view. This is the C++17 spelling of a concept — the HIP toolchain is
still compiled as C++17, so the `concept` keyword is unavailable. When
`CMAKE_HIP_STANDARD` moves to 20, `KvViewLike` becomes a real concept with no
call-site changes.

Consequences that are intentional, not accidental:

* Concrete entry points take `const BlockedKvView<Codec>&`, so raw pointers
  and ad-hoc `(ptr, row_stride)` tuples no longer compile.
* The codec is part of the parameter type. Q8 and K8V4 call the *same* source
  line and the compiler picks the instantiation. There is no runtime branch
  on the codec inside a kernel.

## 4. L1 — per-side block sizes

Keys and values are parameterized independently:

```cpp
void QuantizeKvRows(const float* src, uint32_t src_tok_stride,
                    uint32_t src_head_stride, uint8_t* dst, uint32_t kv_width,
                    uint32_t kv_heads, uint32_t d, uint32_t n_tokens,
                    KvCodec codec, float range, uint32_t block_bytes,
                    const uint32_t* start_pos, hipStream_t stream);

bool DequantizeKvRows(const uint8_t* src_k, __half* dst_k,
                      const uint8_t* src_v, __half* dst_v, uint32_t n_rows,
                      uint32_t row_elems, uint32_t key_block_bytes,
                      uint32_t value_block_bytes, KvCodec codec,
                      hipStream_t stream);

void AttentionKv(..., uint32_t key_block_bytes, uint32_t value_block_bytes,
                 KvCodec codec, hipStream_t stream, float score_scale);
```

`block_bytes == 34` selects 8-bit codes, `block_bytes == 18` selects packed
nibbles. Because one thread owns a whole 32-element block, nibble packing
never has to share a byte across lanes.

`QuantizeKvRows(src, ..., start_pos)` treats `*start_pos` as the **destination
row index**, not as an absolute token position: it does not apply a modulo
and it does not know about ring buffers. The caller pre-splits a ring-
straddling chunk into at most two launches and hands over the slot index for
each. This keeps the shared kernel free of per-model ring semantics.

## 5. L2 — shared attention compute

`blocked_splitk.hip.cpp` (split-K) and `blocked_wmma.hip.cpp` (WMMA prefill)
are templated on the codec and read scales through `blocked_kv::LoadScaleF16`.
They take `key_block_bytes` / `value_block_bytes` so one instantiation serves
Q8 (34/34) and K8V4 (34/18).

These files previously carried their own `CodecTraits` and their own scale
loaders; both were deleted. If you find a `KvCodecQ8` struct or a hand-rolled
half conversion inside a kernel file, it is dead code or a regression.

## 6. L3 — model adapters

Inkling-2-Mini does not call `AttentionKv`: its attention needs a sliding
window ring, a relative-position bias bank, per-head sinks, and a fixed
`qpg == 3` GQA mapping. Those requirements live in
`models/inkling/kernels/kernels.hip.cpp`, which still shares:

* L0 codecs, scale conversion, lane addressing;
* L0 views (`MakeBlockedKvView`) for every cache plane;
* `BlockedDequant4<Codec>` — a 4-elements-per-lane fused dequant built on the
  L0 codec, so Q8 reads 4 int8 codes and Q4 reads 2 payload bytes;
* the L1 quantizer for dense multi-token stores.

Kernel bodies are templated on a value codec only:

```cpp
template <uint32_t kQpg, typename VCodec> InklingAttnSplitKGqaQ8Kernel;
template <typename VCodec>                InklingAttnWave32Q8Kernel;
template <typename VCodec>                InklingAttnWave32Q8ByPosKernel;
template <typename VCodec>                StoreKvQ8ByPosKernel;
```

with two public overloads each, chosen by the value view type:

```cpp
StoreKvQ8ByPos(k, v, BlockedKvQ8View<K>, BlockedKvQ8View<Q8>, ...);  // Q8
StoreKvQ8ByPos(k, v, BlockedKvQ8View<K>, BlockedKvQ4View<Q4>, ...);  // K8V4
```

Keys are **always** Q8. Post-roPE key outliers need 8-bit range; only values
drop to 4 bits.

### Ring / slot rule

Every blocked kernel takes `wrap` (`0` = dense). The slot for position `p` is
`KvSlot(wrap, p) = wrap ? p % wrap : p`. Storing, reading, backup and restore
all use the same rule, so SWA layers keep only `sliding_window` rows while
global layers keep the full context — without any kernel knowing which kind
of layer it is serving.

## 7. Mode selection (Inkling)

| `kv_cache_quant` | env | mode |
|---|---|---|
| `q8` | — | `kQ8` |
| `q4`, `k8v4` | — | `kK8V4` |
| `k4v8`, `fp8` | — | `kF16` |
| — | `MBUFO_INKLING_KV_Q8=1` | `kQ8` |
| — | `MBUFO_KV_V4=1` | `kK8V4` (with blocked on) |
| — | `MBUFO_INKLING_KV_Q4=1` | `kK8V4` (forces blocked on) |
| `k4v8`, `fp8` | any | `kF16` — the catalog string wins last |

Precedence, narrowest first: `MBUFO_INKLING_KV_Q4` / `MBUFO_INKLING_KV_V4` /
`MBUFO_INKLING_KV_Q8` (inkling-only, never touched by the CLI) beat the
model's own `kv_cache_quant` string, which beats the process-wide
`MBUFO_KV_V4` that `serve.cpp` derives from `--kv-quant`. That last one is a
default shared with gemma4 and flash-next, so it must not be able to turn a
per-model `q4` back into Q8. An explicit `k4v8`/`fp8` request switches
blocked off.

Blocked modes additionally require `num_q_heads == 3 * num_kv_heads` and
`head_dim == 128` (the fused kernels assume that geometry). Init fails loudly
rather than falling back.

Engine bookkeeping:

* `kv_row_stride_k_`, `kv_row_stride_v_`: per-side row bytes; allocation, ring
  straddle copies and backup planes all use the side's own stride.
* `KvMode` (`kF16`, `kQ8`, `kK8V4`) replaces the old `kv_q8_` boolean.
* Graph-capture path (`capture_by_pos_`): single-token store + attention read
  the position from a device scalar.
* Non-captured path: dense multi-token store through `QuantizeKvRows`,
  pre-split into at most two launches on a ring straddle.

## 8. Rules for new code

Do:

* Take `BlockedKvView<Codec>` by reference; build it with `MakeBlockedKvView`.
* Get scale math from `ScaleFromAmax` / `StoreScaleF16` / `LoadScaleF16`.
* Get block/row/head arithmetic from `BlockedRowBytes`,
  `BlockedHeadStrideBytes`, `BlockOfLane`, `OffsetOfLane`.
* Parameterize on the codec with templates; do not pass a codec enum into a
  kernel that then branches per element.

Do not:

* Re-declare `KvCodecQ8` (or any codec) inside a kernel file.
* Hand-roll a float/half conversion.
* Add a `(ptr, row_stride)` entry point: that is what the view exists for.
* Assume both sides share a block size — K and V are independent.

`tools/ci/check-kv-codec.py` enforces the "no new inline codec" rule. It
reports 0 violations for new code; 63 pre-existing sites are grandfathered.

## 9. Tests

| Test | What it pins down |
|---|---|
| `tools/bench/kv_codec_check/kv_test_host.cpp` (g++, C++17) | half round-trip vs FP64 oracle over 65536 values, Q8/nibble/Fp8 known-answer vectors, view stride checks |
| `tools/bench/kv_codec_check/kv_test_dev.cpp` (hipcc) | 34/34 and 34/18 row round-trips on device, `AttentionKv` / `AttentionBlockedSplitK` vs FP32 reference |
| parity harness on .55 (`*_parity_run.sh` + `cmp_*.py`) | F16 baseline byte-identical, Q8 and K8V4 vs F16 per-layer relative error, final-hidden, argmax |
| long-context smoke | ring wrap past `sliding_window` must not crash or drift |

## 10. Known rough edges

1. `KvCodec` (the enum passed to L1/L2) has only `kQ8` and `kFp8`. The 4-bit
   layout is selected implicitly by `block_bytes == 18`. It works, but
   `codec = kQ8` + `block_bytes = 18` reads as a contradiction. Adding an
   explicit `kQ4` case is a cleanup, not a behavior change.
2. Attention kernel *bodies* are not unified: Inkling keeps its own
   ring/rel-bias/sink/qpg-3 kernels. Unifying them means turning ring, bias and
   sinks into optional parameters of the shared kernel — a separate piece of
   work, and one that should be done only with the parity harness in place.
3. `prepare/flash-next/gemma` paths were not re-run after the trait
   unification in this branch; they share L0/L1/L2, so a regression there
   would surface as a compile error or a codec KAT failure, not silently.
