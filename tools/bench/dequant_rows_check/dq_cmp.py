import numpy as np, sys
d = sys.argv[1]
ok = True
for line in open(f'{d}/meta.txt'):
    name, code, cols, nrows = line.split(); cols = int(cols); nrows = int(nrows)
    ref = np.fromfile(f'{d}/{name}.ref.f32', dtype=np.float32).reshape(nrows, cols)
    try:
        e = np.fromfile(f'{d}/{name}.embed.f32', dtype=np.float32).reshape(nrows, cols)
        h = np.fromfile(f'{d}/{name}.deq.f32', dtype=np.float32).reshape(nrows, cols)
    except Exception as ex:
        print(f'{name:5s} MISSING output ({ex})'); ok = False; continue
    scale = float(np.abs(ref).max()) or 1.0
    ee = float(np.abs(e - ref).max()); eh = float(np.abs(h - ref).max())
    # embed is exact F32 math; deq goes through one F16 rounding (<= 2^-11 relative + subnormal slack)
    h_tol = np.abs(ref) * 2.0 ** -10 + 1e-6
    ok_e = ee <= 1e-5 * scale
    ok_h = bool((np.abs(h - ref) <= h_tol).all())
    print(f'{name:5s} type={code:>2s} rows={nrows:3d} cols={cols:5d} scale={scale:9.4f} embed_maxerr={ee:.3e} {"OK" if ok_e else "FAIL"}  deq_f16_maxerr={eh:.3e} {"OK" if ok_h else "FAIL"}')
    ok = ok and ok_e and ok_h
print('ALL OK' if ok else 'FAILURES')
sys.exit(0 if ok else 1)
