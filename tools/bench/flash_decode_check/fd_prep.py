"""Reference data for FlashSinkDecode (shared lane-per-key split-K F16 decode).

float64 softmax attention with an optional sink term and sliding window over
random q/k/v; 31 cases (head_dim 64 with sinks, head_dim 128 without, windows 0 /
128 / 512, key counts 1..4000).

  python3 fd_prep.py <outdir>
  hipcc -O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -ffast-math -fno-finite-math-only \\
      -I<repo> -x hip fd_test.cpp <repo>/src/core/compute/attention/flash_sink.hip.cpp -o fd_test
  ./fd_test <outdir>     # every case must be OK (rel-L2 < 2e-5; measured ~1e-7)
"""
"""Reference data for FlashSinkDecode (shared split-K F16 decode attention).
float64 softmax attention with an optional sink term and sliding window."""
import os, sys
import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.RandomState(21)
# (D, heads, kv_heads, n_kv, window, has_sink)
cases = []
for n_kv in (1, 2, 31, 32, 33, 100, 129, 300, 1025, 4000):
    cases.append((64, 64, 8, n_kv, 0, 1))      # gpt-oss: sinks, GQA 8
    cases.append((64, 64, 8, n_kv, 128, 1))    # gpt-oss sliding-window layer
    cases.append((128, 32, 2, n_kv, 0, 0))     # glimmer / llama style: no sink, GQA 16
cases.append((128, 32, 8, 2000, 512, 0))       # windowed, no sink
meta = []
for ci, (D, heads, kvh, n_kv, window, has_sink) in enumerate(cases):
    q = (rng.randn(heads, D) * 1.0).astype(np.float32)
    k = (rng.randn(n_kv, kvh, D) * 1.0).astype(np.float16)
    v = (rng.randn(n_kv, kvh, D) * 1.0).astype(np.float16)
    sink = (rng.randn(heads) * 2.0).astype(np.float32)
    scale = 1.0 / np.sqrt(D)
    lo = max(0, n_kv - window) if window else 0
    ref = np.zeros((heads, D), dtype=np.float64)
    for h in range(heads):
        kh = h // (heads // kvh)
        s = (k[lo:n_kv, kh, :].astype(np.float64) @ q[h].astype(np.float64)) * scale
        logits = np.concatenate([s, [sink[h]]]) if has_sink else s
        m = logits.max()
        w = np.exp(s - m)
        denom = w.sum() + (np.exp(sink[h] - m) if has_sink else 0.0)
        ref[h] = (w[:, None] * v[lo:n_kv, kh, :].astype(np.float64)).sum(axis=0) / denom
    name = f'c{ci}'
    q.tofile(f'{out}/{name}.q'); k.tofile(f'{out}/{name}.k'); v.tofile(f'{out}/{name}.v')
    sink.tofile(f'{out}/{name}.sink'); ref.astype(np.float32).tofile(f'{out}/{name}.ref')
    meta.append((name, D, heads, kvh, n_kv, window, has_sink))
with open(f'{out}/meta.txt', 'w') as f:
    for m in meta:
        f.write('%s %d %d %d %d %d %d\n' % m)
print(len(meta), 'cases')
