"""Reference data for FlashWmmaPrefill: float64 causal (optionally windowed, optionally sinked) GQA attention,
head_dim 64 or 128. Writes <name>.q/.k/.v/.ref (+ .s for sinks) and meta.txt:  name heads kvh n start window D has_sink"""
import os, sys
import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.RandomState(41)
cases = []  # (heads, kv_heads, n_tokens, start_pos, window, head_dim, has_sink)
for D, sink in ((128, 0), (64, 1), (64, 0)):
    for heads, kvh in ((32, 2), (32, 8), (16, 8), (8, 4), (64, 8)):
        for n, start in ((1, 0), (16, 0), (17, 37), (100, 1000), (512, 0), (512, 1500)):
            cases.append((heads, kvh, n, start, 0, D, sink))
        cases.append((heads, kvh, 100, 700, 128, D, sink))
        cases.append((heads, kvh, 512, 900, 300, D, sink))
meta = []
for ci, (heads, kvh, n, start, window, D, sink) in enumerate(cases):
    kv_len = start + n
    q = rng.randn(n, heads, D).astype(np.float32)
    k = rng.randn(kv_len, kvh, D).astype(np.float16)
    v = rng.randn(kv_len, kvh, D).astype(np.float16)
    sinks = (rng.randn(heads) * 2.0).astype(np.float32) if sink else None
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
            m = s.max()
            if sink: m = max(m, float(sinks[h]))
            w = np.exp(s - m)
            den = w.sum() + (np.exp(float(sinks[h]) - m) if sink else 0.0)
            ref[t, h] = (w[:, None] * V[lo:qp + 1, kh, :]).sum(axis=0) / den
    name = f'c{ci}'
    q.tofile(f'{out}/{name}.q'); k.tofile(f'{out}/{name}.k'); v.tofile(f'{out}/{name}.v')
    if sink: sinks.tofile(f'{out}/{name}.s')
    ref.astype(np.float32).tofile(f'{out}/{name}.ref')
    meta.append((name, heads, kvh, n, start, window, D, sink))
with open(f'{out}/meta.txt', 'w') as f:
    for m in meta:
        f.write('%s %d %d %d %d %d %d %d\n' % m)
print(len(meta), 'cases')