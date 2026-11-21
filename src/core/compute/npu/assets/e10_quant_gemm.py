#!/usr/bin/env python3
"""Universal NPU2 Shared Quantized GEMM Kernels (Q8 bfp16ebs8 & Q4 q4nx) + Bundle Exporter.

Builds and verifies 4 shared 16-core (4x4, MT=16, KT=64, NT=64) AIE2p GEMM kernels
using native 8x8x8 BFP16 MACs (`K.mm_bfp(16, 64, 64, mixed=True)`):
  1. q8_gemm_64x256x256   (W8A16 bfp16ebs8, K_tiles=4,  N_tiles_col=1)
  2. q8_gemm_64x1024x1024 (W8A16 bfp16ebs8, K_tiles=16, N_tiles_col=4)
  3. q4_gemm_64x256x256   (W4A16 q4nx -> on-core L1 q4nx_dequant -> mm_bfp)
  4. q4_gemm_64x1024x1024 (W4A16 q4nx -> on-core L1 q4nx_dequant -> mm_bfp)

These 4 kernels serve all 5 models (EmbeddingGemma 2, Qwen3-TTS 12Hz 1.7B,
Qwen3-ASR 1.7B, FLUX.2-klein-4B, and Z-Image-Turbo) within 4 hwctx slots.
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
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, In, Out, kernels as K
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifo
from aie.iron.device import Tile
from aie.iron.program import Program
from aie.iron.runtime import Runtime, TaskGroup
from aie.iron.worker import Worker
from aie.utils import bfp, get_current_device

DEV = get_current_device()
iron.set_current_device(DEV)
bf16 = bfloat16

Q8_TILE_BYTES = 64 * 64 * 9 // 8  # 4608 bytes per 64x64 tile
Q4_TILE_BYTES = 64 * 64 // 2 + 4 * (64 * 64 // 32)  # 2560 bytes per 64x64 tile


def pack_tile_q8_bfp16(tile_kn: np.ndarray) -> np.ndarray:
    """Pack a (64, 64) float32/bf16 weight tile into mm_bfp(mixed=True) B order (4608 uint8)."""
    bt = np.ascontiguousarray(tile_kn.astype(np.float32).T)  # (N=64, K=64)
    # (n8=8, t=8, k_step=8, s=8) -> (n8=8, k_step=8, t=8, s=8)
    blocks = bt.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8)
    return bfp.encode(blocks).reshape(-1)


def pack_matrix_q8_bfp16(W_kn: np.ndarray, K_dim: int, N_dim: int) -> np.ndarray:
    """Pack W[K_dim, N_dim] into linear per-column [C=4, N_tiles_col, K_tiles, 4608] uint8."""
    assert W_kn.shape == (K_dim, N_dim)
    C, KT, NT = 4, 64, 64
    K_tiles = K_dim // KT
    N_col = N_dim // C
    N_tiles_col = N_col // NT
    out = np.empty((C, N_tiles_col, K_tiles, Q8_TILE_BYTES), dtype=np.uint8)
    for c in range(C):
        for nt in range(N_tiles_col):
            n0 = c * N_col + nt * NT
            for kt in range(K_tiles):
                k0 = kt * KT
                out[c, nt, kt] = pack_tile_q8_bfp16(W_kn[k0 : k0 + KT, n0 : n0 + NT])
    return out.reshape(-1)


def pack_tile_q4_q4nx(tile_kn: np.ndarray) -> np.ndarray:
    """Quantize & pack a (64, 64) weight tile into q4nx (m_tile=64, k_tile=64, group=32, ct_k=64) (2560 uint8)."""
    w = np.ascontiguousarray(tile_kn, dtype=np.float32)  # (K=64, N=64)
    wg = w.reshape(2, 32, 64)  # (k_group=2, group=32, N=64)
    w_min = wg.min(axis=1)
    w_max = wg.max(axis=1)
    mins_bf16 = w_min.astype(bf16)
    mins_f32 = mins_bf16.astype(np.float32)
    scales_bf16 = ((w_max - mins_f32) / 15.0).astype(bf16)
    scales_f32 = scales_bf16.astype(np.float32)
    safe_scales = np.where(scales_f32 > 0.0, scales_f32, 1.0)
    q = np.clip(np.rint((wg - mins_f32[:, None, :]) / safe_scales[:, None, :]), 0, 15).astype(np.uint8)
    q = np.where(scales_f32[:, None, :] > 0.0, q, 0).reshape(64, 64)
    # Order nibbles as [n//16, k, n%16], low nibble first
    q_ord = q.reshape(64, 4, 16).transpose(1, 0, 2).reshape(-1)
    packed = ((q_ord[0::2] & 0x0F) | ((q_ord[1::2] & 0x0F) << 4)).astype(np.uint8)
    params = np.concatenate([scales_bf16.reshape(-1), mins_bf16.reshape(-1)]).view(np.uint16).astype("<u2").view(np.uint8)
    return np.concatenate([params, packed])


def pack_matrix_q4_q4nx(W_kn: np.ndarray, K_dim: int, N_dim: int) -> np.ndarray:
    """Pack W[K_dim, N_dim] into linear per-column [C=4, N_tiles_col, K_tiles, 2560] uint8."""
    assert W_kn.shape == (K_dim, N_dim)
    C, KT, NT = 4, 64, 64
    K_tiles = K_dim // KT
    N_col = N_dim // C
    N_tiles_col = N_col // NT
    out = np.empty((C, N_tiles_col, K_tiles, Q4_TILE_BYTES), dtype=np.uint8)
    for c in range(C):
        for nt in range(N_tiles_col):
            n0 = c * N_col + nt * NT
            for kt in range(K_tiles):
                k0 = kt * KT
                out[c, nt, kt] = pack_tile_q4_q4nx(W_kn[k0 : k0 + KT, n0 : n0 + NT])
    return out.reshape(-1)


def build_gemm_q8(K_dim: int, N_dim: int):
    """16-core 4x4 W8A16 (bfp16ebs8) GEMM: C[64, N_dim] = A[64, K_dim] @ B_q8[K_dim, N_dim]."""
    M, R, C, MT, KT, NT = 64, 4, 4, 16, 64, 64
    K_tiles = K_dim // KT
    N_col = N_dim // C
    N_tiles_col = N_col // NT

    mm_base = K.mm_bfp(MT, KT, NT, mixed=True)
    a_ty = np.ndarray[(MT * KT,), np.dtype[bf16]]
    b_ty = np.ndarray[(Q8_TILE_BYTES,), np.dtype[np.uint8]]
    c_ty = np.ndarray[(MT * NT,), np.dtype[bf16]]
    c2_ty = np.ndarray[(2 * MT * NT,), np.dtype[bf16]]

    mm_k = mm_base.object_file.bind(
        "matmul_vectorized_different_datatypes", [a_ty, b_ty, c_ty]
    )
    z_k, ce_k = K.zero((MT * NT,), bf16), K.conv_even()

    tap_tile_a = TensorAccessPattern([MT * KT], 0, [2, 8, 8, 8], [512, 8, 64, 1])
    tap_c1 = TensorAccessPattern([MT * NT], 0, [2, 8, 8, 8], [512, 8, 64, 1])
    a_cols = [C + r for r in range(R)]

    col_b_bytes = N_tiles_col * K_tiles * Q8_TILE_BYTES
    total_b_bytes = C * col_b_bytes

    A_ty = np.ndarray[(M * K_dim,), np.dtype[bf16]]
    B_ty = np.ndarray[(total_b_bytes,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N_dim,), np.dtype[bf16]]

    @iron.jit
    def gemm_q8_kernel(A_in: In, B_in: In, C_out: Out):
        of_a_in = [ObjectFifo(a_ty, depth=4, name=f"q8_ai_{r}") for r in range(R)]
        of_a_bc = [
            of_a_in[r].cons(depth=2).forward(tile=Tile(a_cols[r], 1), depth=2, to_stream=tap_tile_a, name=f"q8_ab_{r}")
            for r in range(R)
        ]

        of_b_in = [ObjectFifo(b_ty, depth=4, name=f"q8_bi_{c}") for c in range(C)]
        of_b_mem = [
            of_b_in[c].cons(depth=2).forward(tile=Tile(c, 1), depth=2, name=f"q8_bm_{c}")
            for c in range(C)
        ]

        of_c_out = [[ObjectFifo(c2_ty, depth=2, to_stream=tap_c1, name=f"q8_co_{c}_{pr}") for pr in range(2)] for c in range(C)]
        of_c_in = [[None] * R for _ in range(C)]
        for c in range(C):
            for pr in range(2):
                subs = of_c_out[c][pr].prod().join(
                    [0, MT * NT], tile=Tile(c, 1), depths=[2, 2], obj_types=[c_ty, c_ty],
                    names=[f"q8_ci_{c}_{pr*2}", f"q8_ci_{c}_{pr*2+1}"],
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
            Worker(
                gemm_body,
                fn_args=[of_a_bc[r].cons(), of_b_mem[c].cons(), of_c_in[c][r].prod(), mm_k, z_k, ce_k],
                tile=Tile(c, 2 + r),
            )
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
                tap_w = TensorAccessPattern(
                    [total_b_bytes], c * col_b_bytes,
                    [1, 1, N_tiles_col * K_tiles, Q8_TILE_BYTES],
                    [1, 1, Q8_TILE_BYTES, 1],
                )
                bhs[c].fill(B_t, tap=tap_w, group=tg)
                for pr in range(2):
                    ch = chs[c * 2 + pr]
                    row_off = pr * (2 * MT)
                    tap_c = TensorAccessPattern([M * N_dim], row_off * N_dim + c * N_col, [N_tiles_col, 2 * MT, NT], [NT, N_dim, 1])
                    ch.drain(C_t, tap=tap_c, wait=True, group=tg)
            tg.finish()

        rt = Runtime(seq, [A_ty, B_ty, C_ty, *a_hdls, *b_hdls, *c_hdls])
        return Program(DEV, rt, workers=workers).resolve_program()

    return gemm_q8_kernel


def build_gemm_q4(K_dim: int, N_dim: int):
    """16-core 4x4 W4A16 (q4nx -> on-core L1 q4nx_dequant -> mm_bfp) GEMM: C[64, N_dim] = A[64, K_dim] @ B_q4."""
    M, R, C, MT, KT, NT = 64, 4, 4, 16, 64, 64
    K_tiles = K_dim // KT
    N_col = N_dim // C
    N_tiles_col = N_col // NT

    mm_base = K.mm_bfp(MT, KT, NT, mixed=True)
    q4_k = K.quant.q4nx_dequant(m_tile=NT, k_tile=KT, group=32, ct_k=KT, s=8, t=8)

    a_ty = np.ndarray[(MT * KT,), np.dtype[bf16]]
    bq_ty = np.ndarray[(Q4_TILE_BYTES,), np.dtype[np.uint8]]
    bw_ty = np.ndarray[(Q8_TILE_BYTES,), np.dtype[np.uint8]]
    c_ty = np.ndarray[(MT * NT,), np.dtype[bf16]]
    c2_ty = np.ndarray[(2 * MT * NT,), np.dtype[bf16]]

    mm_k = mm_base.object_file.bind(
        "matmul_vectorized_different_datatypes", [a_ty, bw_ty, c_ty]
    )
    z_k, ce_k = K.zero((MT * NT,), bf16), K.conv_even()

    tap_tile_a = TensorAccessPattern([MT * KT], 0, [2, 8, 8, 8], [512, 8, 64, 1])
    tap_c1 = TensorAccessPattern([MT * NT], 0, [2, 8, 8, 8], [512, 8, 64, 1])
    a_cols = [C + r for r in range(R)]

    col_b_bytes = N_tiles_col * K_tiles * Q4_TILE_BYTES
    total_b_bytes = C * col_b_bytes

    A_ty = np.ndarray[(M * K_dim,), np.dtype[bf16]]
    B_ty = np.ndarray[(total_b_bytes,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N_dim,), np.dtype[bf16]]

    @iron.jit
    def gemm_q4_kernel(A_in: In, B_in: In, C_out: Out):
        of_a_in = [ObjectFifo(a_ty, depth=4, name=f"q4_ai_{r}") for r in range(R)]
        of_a_bc = [
            of_a_in[r].cons(depth=2).forward(tile=Tile(a_cols[r], 1), depth=2, to_stream=tap_tile_a, name=f"q4_ab_{r}")
            for r in range(R)
        ]

        of_b_in = [ObjectFifo(bq_ty, depth=4, name=f"q4_bi_{c}") for c in range(C)]
        of_b_mem = [
            of_b_in[c].cons(depth=2).forward(tile=Tile(c, 1), depth=2, name=f"q4_bm_{c}")
            for c in range(C)
        ]

        of_c_out = [[ObjectFifo(c2_ty, depth=2, to_stream=tap_c1, name=f"q4_co_{c}_{pr}") for pr in range(2)] for c in range(C)]
        of_c_in = [[None] * R for _ in range(C)]
        for c in range(C):
            for pr in range(2):
                subs = of_c_out[c][pr].prod().join(
                    [0, MT * NT], tile=Tile(c, 1), depths=[2, 2], obj_types=[c_ty, c_ty],
                    names=[f"q4_ci_{c}_{pr*2}", f"q4_ci_{c}_{pr*2+1}"],
                )
                of_c_in[c][pr * 2], of_c_in[c][pr * 2 + 1] = subs[0], subs[1]

        w_l1 = [
            [Buffer(bw_ty, name=f"q4_wl1_{c}_{r}", tile=Tile(c, 2 + r)) for r in range(R)]
            for c in range(C)
        ]

        def gemm_q4_body(oa, ob, oc, w_buf, q4_fn, mm_fn, z_fn, ce_fn):
            ce_fn()
            for _ in range_(N_tiles_col):
                ct = oc.acquire(1)
                z_fn(ct)
                for _ in range_(K_tiles):
                    a = oa.acquire(1)
                    wq = ob.acquire(1)
                    q4_fn(wq, w_buf)
                    mm_fn(a, w_buf, ct)
                    oa.release(1)
                    ob.release(1)
                oc.release(1)

        workers = [
            Worker(
                gemm_q4_body,
                fn_args=[
                    of_a_bc[r].cons(),
                    of_b_mem[c].cons(),
                    of_c_in[c][r].prod(),
                    w_l1[c][r],
                    q4_k,
                    mm_k,
                    z_k,
                    ce_k,
                ],
                tile=Tile(c, 2 + r),
                stack_size=2048,
            )
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
                tap_w = TensorAccessPattern(
                    [total_b_bytes], c * col_b_bytes,
                    [1, 1, N_tiles_col * K_tiles, Q4_TILE_BYTES],
                    [1, 1, Q4_TILE_BYTES, 1],
                )
                bhs[c].fill(B_t, tap=tap_w, group=tg)
                for pr in range(2):
                    ch = chs[c * 2 + pr]
                    row_off = pr * (2 * MT)
                    tap_c = TensorAccessPattern([M * N_dim], row_off * N_dim + c * N_col, [N_tiles_col, 2 * MT, NT], [NT, N_dim, 1])
                    ch.drain(C_t, tap=tap_c, wait=True, group=tg)
            tg.finish()

        rt = Runtime(seq, [A_ty, B_ty, C_ty, *a_hdls, *b_hdls, *c_hdls])
        return Program(DEV, rt, workers=workers).resolve_program()

    return gemm_q4_kernel


def verify_and_bench(name: str, fn, K_dim: int, N_dim: int, mode: str, iters: int = 10):
    rng = np.random.default_rng(42 + K_dim + (1 if mode == "q4" else 0))
    A = rng.normal(0.0, 0.5, (64, K_dim)).astype(bf16)
    W = rng.normal(0.0, 0.05, (K_dim, N_dim)).astype(np.float32)

    if mode == "q8":
        B_packed = pack_matrix_q8_bfp16(W, K_dim, N_dim)
    else:
        B_packed = pack_matrix_q4_q4nx(W, K_dim, N_dim)

    A_dev = iron.tensor(A.reshape(-1), dtype=bf16)
    B_dev = iron.tensor(B_packed, dtype=np.uint8)
    C_dev = iron.zeros(64 * N_dim, dtype=bf16)

    hdl, kres = fn(A_dev, B_dev, C_dev)
    for _ in range(2):
        _, kres = fn(A_dev, B_dev, C_dev)

    times_us = []
    t0 = time.perf_counter()
    for _ in range(iters):
        _, kres = fn(A_dev, B_dev, C_dev)
        times_us.append(kres.npu_time / 1e3)
    wall_us = (time.perf_counter() - t0) * 1e6 / iters

    got32 = C_dev.numpy().reshape(64, N_dim).astype(np.float32)
    ref32 = (A.astype(np.float32) @ W).astype(bf16).astype(np.float32)
    max_abs = float(np.abs(got32 - ref32).max())
    cos = float(np.dot(got32.ravel(), ref32.ravel()) / (np.linalg.norm(got32) * np.linalg.norm(ref32)))

    print(
        f"[{name}] mode={mode.upper()} shape=64x{K_dim}x{N_dim} B_bytes={B_packed.size} | "
        f"npu_median={np.median(times_us):.1f}us wall={wall_us:.1f}us | max_abs={max_abs:.4f} cos={cos:.7f}",
        flush=True,
    )
    min_cos = 0.999 if mode == "q8" else 0.995
    if cos < min_cos:
        raise RuntimeError(f"{name} cosine {cos:.7f} < {min_cos}")
    return hdl, cos, float(np.median(times_us))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--export-bundle",
        type=Path,
        default=Path("/mnt/nvme-data/mbufo-models/npu-kernels/quant_gemm"),
    )
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()

    # Self-test packing parity against IRON contract layout & q4nx_dequant_ref
    rng = np.random.default_rng(123)
    w_test = rng.normal(0.0, 0.25, (64, 64)).astype(np.float32)
    mm_test = K.mm_bfp(16, 64, 64, mixed=True)
    ref_pack = mm_test.contract.layouts[1].pack(w_test[None, ...])[0]
    our_pack = pack_tile_q8_bfp16(w_test)
    assert np.array_equal(ref_pack, our_pack), "Q8 pack mismatch vs mm_bfp contract layout[1]!"
    q4_bytes = pack_tile_q4_q4nx(w_test)
    q4_deq = K.quant.q4nx_dequant_ref(
        q4_bytes[None, :], m_tile=64, k_tile=64, group=32, ct_k=64, s=8, t=8
    )[0]
    w_deq = mm_test.contract.layouts[1].unpack(q4_deq[None, ...])[0]
    q4_tile_cos = float(np.dot(w_test.ravel(), w_deq.ravel()) / (np.linalg.norm(w_test) * np.linalg.norm(w_deq)))
    print(f"[PackSelfTest] Q8 byte-exact=PASS | Q4 64x64 dequant tile cos={q4_tile_cos:.7f}", flush=True)
    assert q4_tile_cos > 0.996

    specs = [
        ("q8_256x256", build_gemm_q8(256, 256), 256, 256, "q8"),
        ("q8_1024x1024", build_gemm_q8(1024, 1024), 1024, 1024, "q8"),
        ("q4_256x256", build_gemm_q4(256, 256), 256, 256, "q4"),
        ("q4_1024x1024", build_gemm_q4(1024, 1024), 1024, 1024, "q4"),
    ]

    out_dir = args.export_bundle
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"M": 64, "kernels": {}}

    for name, fn, kd, nd, mode in specs:
        hdl, cos, med_us = verify_and_bench(name, fn, kd, nd, mode, iters=args.iters)
        xcl_dst = out_dir / f"{name}.xclbin"
        inst_dst = out_dir / f"{name}.insts.bin"
        shutil.copyfile(hdl.xclbin_path, xcl_dst)
        shutil.copyfile(hdl.insts_path, inst_dst)
        manifest["kernels"][name] = {
            "mode": mode,
            "M": 64,
            "K": kd,
            "N": nd,
            "xclbin": xcl_dst.name,
            "insts": inst_dst.name,
            "kernel_name": hdl.kernel_name or "MLIR_AIE",
            "cos_vs_fp32": cos,
            "npu_median_us": med_us,
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Export] Saved 4 quant GEMM kernels to {out_dir}", flush=True)
    print("QUANT_GEMM_SUITE_RESULT=PASS", flush=True)


if __name__ == "__main__":
    main()
