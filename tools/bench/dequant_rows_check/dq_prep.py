"""Reference data for the dequant_rows check (EmbedGather / DequantRowsF16).

Pulls rows of real Q4_0/Q8_0/Q4_K/Q5_K/Q6_K tensors from local GGUF files (paths
in SRC; edit them for your machine), dequantizes them with the `gguf` Python
package as the reference and adds synthetic F16/BF16 rows.

  python3 dq_prep.py <outdir>
  hipcc -O2 -std=c++20 -x hip --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I<repo> \\
      dq_test.cpp <repo>/src/core/compute/gemm/dequant_rows.hip -o dq_test
  ./dq_test <outdir> && python3 dq_cmp.py <outdir>
"""
import os
MODELS = os.environ.get('MBUFO_MODELS_DIR', 'models')
import gguf, json, os, random, sys
import numpy as np
from gguf import quants

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
CODE = {'Q4_0': 2, 'Q8_0': 8, 'Q4_K': 12, 'Q5_K': 13, 'Q6_K': 14}
SRC = {
    'Q4_0': (MODELS + '/gemma-4-26b/gemma-4-26B-A4B-it-Q4_0.gguf', 'blk.0.attn_k.weight'),
    'Q8_0': (MODELS + '/muse-glimmer-30b/Muse-Glimmer-30B-Q8_0.gguf', 'output.weight'),
    'Q4_K': (MODELS + '/muse-glimmer-30b/Muse-Glimmer-30B-UD-Q4_K_XL.gguf', 'token_embd.weight'),
    'Q5_K': (MODELS + '/muse-glimmer-30b/Muse-Glimmer-30B-UD-Q4_K_XL.gguf', 'output.weight'),
    'Q6_K': (MODELS + '/gemma-4-12b/gemma-4-12b-it-Q4_K_M.gguf', 'blk.0.attn_v.weight'),
}
rng = random.Random(7)
meta = []
for name, (fn, tname) in SRC.items():
    r = gguf.GGUFReader(fn)
    t = next(x for x in r.tensors if x.name == tname)
    cols, rows = int(t.shape[0]), int(t.shape[1])
    data = np.asarray(t.data).reshape(rows, -1)
    idx = sorted(set(list(range(8)) + [rng.randrange(rows) for _ in range(40)] + [rows - 1]))
    sel = np.ascontiguousarray(data[idx])
    ref = quants.dequantize(sel, t.tensor_type).astype(np.float32).reshape(len(idx), cols)
    sel.tofile(f'{out}/{name}.bin')
    ref.tofile(f'{out}/{name}.ref.f32')
    meta.append((name, CODE[name], cols, len(idx)))
    print(name, tname, 'cols', cols, 'rows', len(idx), 'row_bytes', sel.shape[1])
# synthetic F16 / BF16
for name, code in (('F16', 1), ('BF16', 30)):
    cols, nrows = 768, 16
    v = (np.random.RandomState(3).randn(nrows, cols) * 3).astype(np.float32)
    if name == 'F16':
        raw = v.astype(np.float16); ref = raw.astype(np.float32); raw.tofile(f'{out}/{name}.bin')
    else:
        u = v.view(np.uint32) >> 16
        raw = u.astype(np.uint16); raw.tofile(f'{out}/{name}.bin')
        ref = (raw.astype(np.uint32) << 16).view(np.float32)
    ref.astype(np.float32).tofile(f'{out}/{name}.ref.f32')
    meta.append((name, code, cols, nrows))
with open(f'{out}/meta.txt', 'w') as f:
    for m in meta:
        f.write('%s %d %d %d\n' % m)
