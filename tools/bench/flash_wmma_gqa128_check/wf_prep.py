"""Reference data for FlashWmmaGqa128Prefill: float64 causal (optionally windowed) GQA attention, head_dim 128."""
import os, sys
import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.RandomState(41)
D = 128
# (heads, kv_heads, n_tokens, start_pos, window)
cases = []
for heads, kvh in ((32, 2), (32, 8), (16, 8), (16, 2), (8, 4)):
    for n, start in ((1, 0), (16, 0), (17, 37), (100, 1000), (512, 0), (512, 1500)):
        cases.append((heads, kvh, n, start, 0))
    cases.append((heads, kvh, 100, 700, 128))
    cases.append((heads, kvh, 512, 900, 300))
meta = []
for ci, (heads, kvh, n, start, window) in enumerate(cases):
    kv_len = start + n
    q = rng.randn(n, heads, D).astype(np.float32)
    k = rng.randn(kv_len, kvh, D).astype(np.float16)
    v = rng.randn(kv_len, kvh, D).astype(np.float16)
    scale = 1.0 / np.sqrt(D)
    qpg = heads // kvh
    ref = np.zeros((n, heads, D), dtype=np.float64)
    K = k.astype(np.float64); V = v.astype(np.float64)
    for t in range(n):
        qp = start + t
        lo = max(0, qp - window + 1) if window else 0
        for h in range(heads):
            kh = h // qpg
            s = (K[lo:qp + 1, kh, :] @ q[t, h].astype(np.float64)) * scale
            w = np.exp(s - s.max())
            ref[t, h] = (w[:, None] * V[lo:qp + 1, kh, :]).sum(axis=0) / w.sum()
    name = f'c{ci}'
    q.tofile(f'{out}/{name}.q'); k.tofile(f'{out}/{name}.k'); v.tofile(f'{out}/{name}.v')
    ref.astype(np.float32).tofile(f'{out}/{name}.ref')
    meta.append((name, heads, kvh, n, start, window))
with open(f'{out}/meta.txt', 'w') as f:
    for m in meta:
        f.write('%s %d %d %d %d %d\n' % m)
print(len(meta), 'cases')
