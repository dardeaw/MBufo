"""blocked_splitk check: GQA-grouped kernel vs legacy kernel vs float64 reference.

  python3 bs_prep.py <outdir>
  hipcc -O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -ffast-math -fno-finite-math-only \\
      -I<repo> -x hip bs_test.cpp <repo>/src/core/compute/attention/blocked_splitk.hip.cpp -o bs_test
  ./bs_test <outdir> [timing]    # 26 cases; add any 2nd argument for per-case timings

The legacy kernel is selected with GUFO_BLOCKED_SPLITK_LEGACY=1 (the test toggles it).
"""
import os, sys
import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.RandomState(33)

def make_q8(x):  # x [rows, kvh*D] float32 -> bytes [rows, nblk*34], decoded float64
    rows, width = x.shape
    nblk = width // 32
    B = x.reshape(rows, nblk, 32).astype(np.float64)
    scale = (np.abs(B).max(axis=2) / 127.0).astype(np.float16)
    sc = scale.astype(np.float64)
    sc_safe = np.where(sc == 0, 1.0, sc)[:, :, None]
    codes = np.clip(np.round(B / sc_safe), -127, 127).astype(np.int8)
    raw = np.zeros((rows, nblk, 34), dtype=np.uint8)
    raw[:, :, :32] = codes.view(np.uint8)
    raw[:, :, 32:34] = scale.view(np.uint8).reshape(rows, nblk, 2)
    dec = (codes.astype(np.float64) * sc[:, :, None]).reshape(rows, width)
    return raw.reshape(rows, nblk * 34), dec

def make_q4(rows, width):
    nblk = width // 32
    nib = rng.randint(0, 256, size=(rows, nblk, 16)).astype(np.uint8)
    scale = rng.uniform(0.02, 0.3, size=(rows, nblk)).astype(np.float16)
    raw = np.zeros((rows, nblk, 18), dtype=np.uint8)
    raw[:, :, :16] = nib
    raw[:, :, 16:18] = scale.view(np.uint8).reshape(rows, nblk, 2)
    lo = (nib & 0xF).astype(np.int64); hi = (nib >> 4).astype(np.int64)
    lo = np.where(lo & 8, lo - 16, lo); hi = np.where(hi & 8, hi - 16, hi)
    vals = np.zeros((rows, nblk, 32), dtype=np.float64)
    vals[:, :, 0::2] = lo; vals[:, :, 1::2] = hi
    dec = (vals * scale.astype(np.float64)[:, :, None]).reshape(rows, width)
    return raw.reshape(rows, nblk * 18), dec

# (D, heads, kv_heads, kv_len, n_tokens, key_block_bytes, value_block_bytes)
cases = []
for kv_len in (128, 129, 1000, 4097):
    cases.append((128, 32, 2, kv_len, 1, 34, 34))     # glimmer: qpg 16
    cases.append((128, 32, 8, kv_len, 1, 34, 34))     # llama: qpg 4
    cases.append((128, 16, 2, kv_len, 3, 34, 34))     # qpg 8, 3 tokens
    cases.append((256, 16, 8, kv_len, 1, 34, 34))     # gemma4 SWA: qpg 2
    cases.append((256, 16, 2, kv_len, 2, 34, 34))     # qpg 8, 2 tokens
cases.append((128, 32, 2, 2000, 1, 18, 34))
cases.append((128, 32, 2, 2000, 1, 34, 18))
cases.append((128, 32, 8, 2000, 1, 18, 18))
cases.append((256, 16, 8, 1500, 1, 18, 18))
cases.append((128, 16, 16, 700, 1, 34, 34))           # qpg 1
cases.append((128, 24, 8, 500, 1, 34, 34))            # qpg 3 -> legacy kernel
meta = []
for ci, (D, heads, kvh, kv_len, nt, kbb, vbb) in enumerate(cases):
    width = kvh * D
    kx = (rng.randn(kv_len, width)).astype(np.float32)
    vx = (rng.randn(kv_len, width)).astype(np.float32)
    kraw, kdec = make_q8(kx) if kbb == 34 else make_q4(kv_len, width)
    vraw, vdec = make_q8(vx) if vbb == 34 else make_q4(kv_len, width)
    q = rng.randn(nt, heads, D).astype(np.float32)
    start = kv_len - nt
    scale = 1.0 / np.sqrt(D)
    ref = np.zeros((nt, heads, D), dtype=np.float64)
    qpg = heads // kvh
    K = kdec.reshape(kv_len, kvh, D); V = vdec.reshape(kv_len, kvh, D)
    for t in range(nt):
        n_kv = start + t + 1
        for h in range(heads):
            kh = h // qpg
            s = (K[:n_kv, kh, :] @ q[t, h].astype(np.float64)) * scale
            w = np.exp(s - s.max())
            ref[t, h] = (w[:, None] * V[:n_kv, kh, :]).sum(axis=0) / w.sum()
    name = f'c{ci}'
    q.tofile(f'{out}/{name}.q'); kraw.tofile(f'{out}/{name}.k'); vraw.tofile(f'{out}/{name}.v')
    ref.astype(np.float32).tofile(f'{out}/{name}.ref')
    meta.append((name, D, heads, kvh, kv_len, nt, kbb, vbb))
with open(f'{out}/meta.txt', 'w') as f:
    for m in meta:
        f.write('%s %d %d %d %d %d %d %d\n' % m)
print(len(meta), 'cases')
