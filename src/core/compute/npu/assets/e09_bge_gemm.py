#!/usr/bin/env python3
"""BGE-M3 / BGE-Reranker NPU2 BF16 GEMM Kernel (64x1024x1024) + Bundle Exporter.

Executes row-major C[64, 1024] = A[64, 1024] @ B[1024, 1024] across a 4x4 AIE2p
core array (MT=16, KT=64, NT=64, 4x4=16 cores, K.mm + conv_even).
Handles arbitrary BGE-M3 / bge-reranker-v2-m3 (hidden=1024, ffn=4096) linear
projections in 64-row chunks and 1024-wide slices via a single compact XCLBIN.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

_ASSET_DIR = Path(__file__).resolve().parent
if str(_ASSET_DIR) not in sys.path:
    sys.path.insert(0, str(_ASSET_DIR))

import aie.iron as iron
from aie import ir
from aie.dialects import memref
from aie.extras.dialects.arith import constant
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import In, Out, kernels as K
from aie.iron.dataflow import ObjectFifo
from aie.iron.device import Tile
from aie.iron.program import Program
from aie.iron.runtime import Runtime, TaskGroup
from aie.iron.worker import Worker
from aie.utils import get_current_device

from aie.iron.controlflow import range_

DEV = get_current_device()
iron.set_current_device(DEV)
bf16 = bfloat16


def build_gemm_bf16_64x1024x1024():
    """16-core 4x4 row-major BF16 GEMM: C[64, 1024] = A[64, 1024] @ B[1024, 1024]."""
    M, K_dim, N_dim = 64, 1024, 1024
    R, C, MT, KT, NT = 4, 4, 16, 64, 64
    M_pass = R * MT  # 64 -> single pass
    K_tiles = K_dim // KT  # 16
    N_col = N_dim // C  # 256
    N_tiles_col = N_col // NT  # 4
    bands = MT // 4  # 4

    mm_k = K.mm(MT, KT, NT, bf16, bf16, b_col_maj=False)
    _, lay_b, _ = mm_k.contract.layouts
    z_k, ce_k = K.zero((MT * NT,), bf16), K.conv_even()

    a_ty = np.ndarray[(MT * KT,), np.dtype[bf16]]
    b_ty = np.ndarray[(KT * NT,), np.dtype[bf16]]
    c_ty = np.ndarray[(MT * NT,), np.dtype[bf16]]
    c2_ty = np.ndarray[(2 * MT * NT,), np.dtype[bf16]]

    # Stream single [MT, KT] tiles with depth=2 to stay well below 16 DMA BDs/core
    tap_tile_a2 = TensorAccessPattern([MT * KT], 0, [bands, KT // 8, 4, 8], [4 * KT, 8, KT, 1])
    tap_c1 = TensorAccessPattern([MT * NT], 0, [bands, 4, 8, 8], [256, 8, 32, 1])
    a_cols = [C + r for r in range(R)]  # cols 4, 5, 6, 7 (disjoint from B/C cols 0..3)

    A_ty = np.ndarray[(M * K_dim,), np.dtype[bf16]]
    B_ty = np.ndarray[(K_dim * N_dim,), np.dtype[bf16]]
    C_ty = np.ndarray[(M * N_dim,), np.dtype[bf16]]

    @iron.jit
    def gemm_bf16_64x1024x1024(A_in: In, B_in: In, C_out: Out):
        of_a_in = [ObjectFifo(a_ty, depth=4, name=f"bg_ai_{r}") for r in range(R)]
        of_a_bc = [
            of_a_in[r].cons(depth=2).forward(tile=Tile(a_cols[r], 1), depth=2, to_stream=tap_tile_a2, name=f"bg_ab_{r}")
            for r in range(R)
        ]

        of_b_in = [ObjectFifo(b_ty, depth=4, name=f"bg_bi_{c}") for c in range(C)]
        of_b_mem = [of_b_in[c].cons(depth=2).forward(tile=Tile(c, 1), depth=2, to_stream=lay_b.stream, name=f"bg_bm_{c}") for c in range(C)]

        of_c_out = [[ObjectFifo(c2_ty, depth=2, to_stream=tap_c1, name=f"bg_co_{c}_{pr}") for pr in range(2)] for c in range(C)]
        of_c_in = [[None] * R for _ in range(C)]
        for c in range(C):
            for pr in range(2):
                subs = of_c_out[c][pr].prod().join(
                    [0, MT * NT], tile=Tile(c, 1), depths=[2, 2], obj_types=[c_ty, c_ty],
                    names=[f"bg_ci_{c}_{pr*2}", f"bg_ci_{c}_{pr*2+1}"],
                )
                of_c_in[c][pr * 2], of_c_in[c][pr * 2 + 1] = subs[0], subs[1]

        def gemm_body(oa, ob, oc, mm_fn, z_fn, ce_fn):
            ce_fn()
            for _ in range_(N_tiles_col):
                ct = oc.acquire(1)
                z_fn(ct)
                for _ in range_(K_tiles):
                    a = oa.acquire(1)
                    w = ob.acquire(1)
                    mm_fn(a, w, ct)
                    oa.release(1)
                    ob.release(1)
                oc.release(1)

        workers = [
            Worker(gemm_body, fn_args=[of_a_bc[r].cons(), of_b_mem[c].cons(), of_c_in[c][r].prod(), mm_k, z_k, ce_k], tile=Tile(c, 2 + r))
            for c in range(C)
            for r in range(R)
        ]

        a_hdls = [of_a_in[r].prod(tile=Tile(a_cols[r], 0)) for r in range(R)]
        b_hdls = [of_b_in[c].prod(tile=Tile(c, 0)) for c in range(C)]
        c_hdls = [of_c_out[c][pr].cons(tile=Tile(c, 0)) for c in range(C) for pr in range(2)]

        def seq(A_t, B_t, C_t, *hdls):
            ahs, bhs, chs = hdls[:R], hdls[R : R + C], hdls[R + C :]
            tg = TaskGroup()
            for r in range(R):
                tap_a = TensorAccessPattern([M * K_dim], (r * MT) * K_dim, [K_tiles, MT, KT], [KT, K_dim, 1])
                for _ in range(N_tiles_col):
                    ahs[r].fill(A_t, tap=tap_a, group=tg)
            for c in range(C):
                tap_w = TensorAccessPattern([K_dim * N_dim], c * N_col, [N_tiles_col, K_tiles, KT, NT], [NT, KT * N_dim, N_dim, 1])
                bhs[c].fill(B_t, tap=tap_w, group=tg)
                for pr in range(2):
                    ch = chs[c * 2 + pr]
                    row_off = pr * (2 * MT)
                    tap_c = TensorAccessPattern([M * N_dim], row_off * N_dim + c * N_col, [N_tiles_col, 2 * MT, NT], [NT, N_dim, 1])
                    ch.drain(C_t, tap=tap_c, wait=True, group=tg)
            tg.finish()

        rt = Runtime(seq, [A_ty, B_ty, C_ty, *a_hdls, *b_hdls, *c_hdls])
        return Program(DEV, rt, workers=workers).resolve_program()

    return gemm_bf16_64x1024x1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-bundle", type=Path, default=None)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    fn = build_gemm_bf16_64x1024x1024()
    rng = np.random.default_rng(7)
    A = rng.normal(0.0, 0.5, (64, 1024)).astype(bf16)
    B = rng.normal(0.0, 0.04, (1024, 1024)).astype(bf16)
    A_dev = iron.tensor(A.reshape(-1), dtype=bf16)
    B_dev = iron.tensor(B.reshape(-1), dtype=bf16)
    C_dev = iron.zeros(64 * 1024, dtype=bf16)

    # Warmup + resolve handle
    hdl, kres = fn(A_dev, B_dev, C_dev)
    for _ in range(3):
        _, kres = fn(A_dev, B_dev, C_dev)

    times_us = []
    t0 = time.perf_counter()
    for _ in range(args.iters):
        _, kres = fn(A_dev, B_dev, C_dev)
        times_us.append(kres.npu_time / 1e3)
    wall_us = (time.perf_counter() - t0) * 1e6 / args.iters

    got = C_dev.numpy().reshape(64, 1024)
    ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(bf16)
    got32, ref32 = got.astype(np.float32), ref.astype(np.float32)
    max_abs = float(np.abs(got32 - ref32).max())
    cos = float(np.dot(got32.ravel(), ref32.ravel()) / (np.linalg.norm(got32) * np.linalg.norm(ref32)))

    print(
        f"[BGE GEMM 64x1024x1024] npu_median={np.median(times_us):.1f}us wall={wall_us:.1f}us | "
        f"max_abs={max_abs:.4f} cos={cos:.7f}",
        flush=True,
    )
    if cos <= 0.9999:
        print("BGE_GEMM_RESULT=FAIL", flush=True)
        sys.exit(1)
    print("BGE_GEMM_RESULT=PASS", flush=True)

    if args.export_bundle is not None:
        out_dir = args.export_bundle
        out_dir.mkdir(parents=True, exist_ok=True)
        xcl_dst = out_dir / "gemm_1024x1024.xclbin"
        inst_dst = out_dir / "gemm_1024x1024.insts.bin"
        shutil.copyfile(hdl.xclbin_path, xcl_dst)
        shutil.copyfile(hdl.insts_path, inst_dst)
        manifest = {
            "M": 64, "K": 1024, "N": 1024,
            "kernels": {
                "gemm_1024x1024": {
                    "xclbin": xcl_dst.name,
                    "insts": inst_dst.name,
                    "kernel_name": hdl.kernel_name or "MLIR_AIE",
                }
            },
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[Export] Saved gemm_1024x1024 bundle to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
