"""Reference data for the native Q5_0 / Q4_K narrow decode GEMV check.

Reads real gpt-oss-120b layer-0 attention tensors (edit `fn` for your machine),
builds the planar Q5_0 repack the loader produces, and computes
y_ref = dequant(W) @ x (+ bias, + residual for Q4_K) in float64.

  python3 nq_prep.py <outdir>
  hipcc -O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I<repo> -x hip nq_test.cpp \\
      <repo>/src/core/compute/gemm/gemv_narrow_quant.hip -x none <build>/libmbufo_core_gemm.a -o nq_test
  ./nq_test <outdir>     # relative L2 error (< 1%) and cold-cache GB/s vs the Q8_0 narrow kernel
"""
"""Reference data for the narrow Q5_0 / Q4_K GEMV check.
Uses real gpt-oss-120b layer-0 attention tensors; y_ref = dequant(W) @ x in float64."""
import os
MODELS = os.environ.get('MBUFO_MODELS_DIR', 'models')
import gguf, os, sys
import numpy as np
from gguf import quants

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
fn = MODELS + '/gpt-oss-120b/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf'
r = gguf.GGUFReader(fn)
rng = np.random.RandomState(11)
meta = []
for case, tname in (('q50', 'blk.0.attn_q.weight'), ('q50k', 'blk.0.attn_k.weight'), ('q4k', 'blk.0.attn_output.weight')):
    t = next(x for x in r.tensors if x.name == tname)
    cols, rows = int(t.shape[0]), int(t.shape[1])
    data = np.ascontiguousarray(np.asarray(t.data).reshape(rows, -1))
    W = quants.dequantize(data, t.tensor_type).astype(np.float64).reshape(rows, cols)
    x = (rng.randn(cols) * 0.7).astype(np.float32)
    bias = (rng.randn(rows) * 0.1).astype(np.float32)
    resid = (rng.randn(rows)).astype(np.float32)
    y = W @ x.astype(np.float64)
    if case.startswith('q50'):
        nb = cols // 32
        blocks = data.reshape(rows, nb, 22)
        d = np.ascontiguousarray(blocks[:, :, 0:2]).reshape(-1)
        qh = np.ascontiguousarray(blocks[:, :, 2:6]).reshape(-1)
        qs = np.ascontiguousarray(blocks[:, :, 6:22]).reshape(-1)
        planar = np.concatenate([qs, qh, d])
        assert planar.size == rows * nb * 22
        planar.tofile(f'{out}/{case}.w')
        ref = (y + bias).astype(np.float32)
    else:
        data.tofile(f'{out}/{case}.w')
        ref = (resid.astype(np.float64) + y + bias).astype(np.float32)
    x.tofile(f'{out}/{case}.x'); bias.tofile(f'{out}/{case}.bias'); resid.tofile(f'{out}/{case}.resid'); ref.tofile(f'{out}/{case}.ref')
    meta.append((case, 5 if case.startswith('q50') else 4, cols, rows))
    print(case, tname, 'cols', cols, 'rows', rows, 'ref-norm %.3f' % np.linalg.norm(ref))
with open(f'{out}/meta.txt', 'w') as f:
    for m in meta:
        f.write('%s %d %d %d\n' % m)
