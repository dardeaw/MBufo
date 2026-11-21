#!/usr/bin/env python3
"""Optimized, decoupled EmbGemma2 FFN on Ryzen AI NPU2 (AIE2p).

Decoupling & shared-compute optimizations over e06_v6.py:
  1. Host-API decoupling: single 4-tensor ffn(X, Wup, Wdn, Out) interface;
     weights are sliced per column inside runtime TensorAccessPatterns.
  2. Single-pass RMS + 3D->4D MemTile A-transform + hardware multicast to all lanes.
  3. Zero intermediate MemTile round-trips: up.C == dn.A (4, 8) micro-tile order
     whenever NT_up == KT_dn; GELU runs in-place via gelu_tile_bf16 on the
     producing core; partial sums reduce West->East in shared L1 SRAM.
  4. Two execution modes:
       - spatial: up[p] at (p, 2) -> shared L1 -> dn[p] at (p, 3), 2 Shim MM2S/col.
         Supports asymmetric (KT_up, NT_up)=(64, 128) + (KT_dn, NT_dn)=(128, 64)
         for 16 KB B-tiles (2x fewer BDs & mm calls, C fits 8 KB).
       - unified: single shared core at (p, 2) runs up + in-place gelu + dn + reduce,
         sharing one mm kernel, one MemTile transform, and one Shim MM2S channel.
  5. Explicit conv_even() setup on all mm cores (fixes floor-rounding bias on aie2p).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie import ir
from aie.dialects import memref
from aie.extras.dialects.arith import constant
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import In, Out, kernels
from aie.iron.buffer import Buffer
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifo
from aie.iron.device import Tile
from aie.iron.program import Program
from aie.iron.runtime import Runtime, TaskGroup
from aie.iron.worker import Worker
from aie.utils import DefaultNPURuntime, get_current_device

DEV = get_current_device()
iron.set_current_device(DEV)
bf16 = bfloat16


@dataclass(frozen=True)
class FfnConfig:
    M: int = 8
    D: int = 512
    F: int = 2048
    KT: int = 64
    NT: int = 64
    KT_dn: int | None = None
    NT_dn: int | None = None
    lanes: int = 8
    mode: str = "unified"  # "unified" | "spatial"

    @property
    def kt_up(self) -> int:
        return self.KT

    @property
    def nt_up(self) -> int:
        return self.NT

    @property
    def kt_dn(self) -> int:
        return self.KT if self.KT_dn is None else self.KT_dn

    @property
    def nt_dn(self) -> int:
        return self.NT if self.NT_dn is None else self.NT_dn

    def __post_init__(self) -> None:
        if self.mode not in ("unified", "spatial"):
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.nt_up != self.kt_dn:
            raise ValueError(
                f"Direct up->gelu->dn handoff requires NT_up == KT_dn, "
                f"got {self.nt_up} != {self.kt_dn}"
            )
        if self.mode == "unified" and (self.kt_up, self.nt_up) != (self.kt_dn, self.nt_dn):
            raise ValueError("unified mode shares one B FIFO and requires equal up/dn tile dims")
        if self.M % 8 != 0 or self.D % self.kt_up != 0 or self.D % self.nt_dn != 0:
            raise ValueError("M must divide by 8; D must divide by kt_up and nt_dn")
        if self.F % self.lanes != 0:
            raise ValueError(f"F={self.F} not divisible by lanes={self.lanes}")
        fp = self.F // self.lanes
        if fp % self.nt_up != 0 or fp % self.kt_dn != 0:
            raise ValueError(f"F/lanes={fp} must divide by nt_up={self.nt_up} and kt_dn={self.kt_dn}")
        max_cols = 8 if self.mode == "unified" else 7
        if self.lanes < 1 or self.lanes > max_cols:
            raise ValueError(f"lanes={self.lanes} out of range [1, {max_cols}] for {self.mode}")

    @property
    def fp(self) -> int:
        return self.F // self.lanes

    @property
    def k_up(self) -> int:
        return self.D // self.kt_up

    @property
    def n_up(self) -> int:
        return self.fp // self.nt_up

    @property
    def k_dn(self) -> int:
        return self.fp // self.kt_dn

    @property
    def n_dn(self) -> int:
        return self.D // self.nt_dn


class KernelBank:
    """Shared AIE2p kernel instances built once per tile geometry."""

    def __init__(self, cfg: FfnConfig):
        up_c_elems = cfg.M * cfg.nt_up
        dn_c_elems = cfg.M * cfg.nt_dn
        self.rms = kernels.rms_norm_eps(cols=cfg.D)
        self.up_mm = kernels.mm(
            dim_m=cfg.M, dim_k=cfg.kt_up, dim_n=cfg.nt_up,
            input_dtype=bf16, output_dtype=bf16,
        )
        if (cfg.kt_dn, cfg.nt_dn) == (cfg.kt_up, cfg.nt_up):
            self.dn_mm = self.up_mm
        else:
            self.dn_mm = kernels.mm(
                dim_m=cfg.M, dim_k=cfg.kt_dn, dim_n=cfg.nt_dn,
                input_dtype=bf16, output_dtype=bf16,
            )
        self.zero_up = kernels.zero(up_c_elems, bf16)
        self.zero_dn = (
            self.zero_up if dn_c_elems == up_c_elems
            else kernels.zero(dn_c_elems, bf16)
        )
        gelu_base = kernels.gelu_sized(up_c_elems)
        self.gelu_ip = gelu_base.object_file.bind(
            "gelu_tile_bf16",
            [np.int32, np.ndarray[(up_c_elems,), np.dtype[bf16]]],
        )
        self.add = kernels.add_sized(dn_c_elems)
        self.conv_even = kernels.conv_even()
        assert self.up_mm.stream_dims.C.inverse().sizes == self.dn_mm.stream_dims.A.sizes
        assert self.up_mm.stream_dims.C.inverse().strides == self.dn_mm.stream_dims.A.strides


def _make_views(byte_buf, count: int, elems: int):
    mr_ty = ir.MemRefType.get([elems], ir.BF16Type.get())
    return [
        memref.view(mr_ty, byte_buf, constant(i * elems * 2, index=True), [])
        for i in range(count)
    ]


def build_rms_broadcast_stage(cfg: FfnConfig, kb: KernelBank, mt_col: int, rms_tile: Tile):
    """Build RMSNorm core + 2-step 3D->4D MemTile A-layout broadcast."""
    row_ty = np.ndarray[(cfg.D,), np.dtype[bf16]]
    mat_i8_ty = np.ndarray[(cfg.M * cfg.D * 2,), np.dtype[np.int8]]

    tap_n_pass1 = TensorAccessPattern(
        (cfg.M * cfg.D * 2,), 0,
        sizes=[cfg.k_up, cfg.M, cfg.kt_up * 2],
        strides=[cfg.kt_up * 2, cfg.D * 2, 1],
    )
    tap_a_pass2 = TensorAccessPattern(
        (cfg.M * cfg.D * 2,), 0,
        sizes=[cfg.k_up * (cfg.M // 4), cfg.kt_up // 8, 4, 8 * 2],
        strides=[4 * cfg.kt_up * 2, 8 * 2, cfg.kt_up * 2, 1],
    )

    of_x_rms = ObjectFifo(row_ty, depth=2, name="x_rms")
    of_n_raw = ObjectFifo(mat_i8_ty, depth=1, name="n_raw", to_stream=tap_n_pass1)
    of_a_bcast = of_n_raw.cons(depth=1).forward(
        tile=Tile(mt_col, 1), obj_type=mat_i8_ty, depth=1, name="a_bcast",
        to_stream=tap_a_pass2,
    )

    def rms_body(o_in, o_out, func):
        n_bytes = o_out.acquire(1)
        nr_views = _make_views(n_bytes, cfg.M, cfg.D)
        for m in range(cfg.M):
            xr = o_in.acquire(1)
            func(xr, nr_views[m], cfg.D, 1e-6)
            o_in.release(1)
        o_out.release(1)

    rms_w = Worker(
        rms_body, [of_x_rms.cons(), of_n_raw.prod(), kb.rms],
        tile=rms_tile, stack_size=0x1000,
    )
    return of_x_rms, of_a_bcast, rms_w


def build_unified_lanes(cfg: FfnConfig, kb: KernelBank, of_a_bcast: ObjectFifo):
    """P shared cores at (p, 2): each owns up -> in-place gelu -> dn -> chain add."""
    b_ty = np.ndarray[(cfg.kt_up, cfg.nt_up), np.dtype[bf16]]
    c_ty = np.ndarray[(cfg.M * cfg.nt_dn,), np.dtype[bf16]]
    c_elems = cfg.M * cfg.nt_dn
    t_f, k_up, n_dn = cfg.n_up, cfg.k_up, cfg.n_dn

    w_fifos, z_chain, workers = [], [], []
    for p in range(cfg.lanes):
        of_w = ObjectFifo(b_ty, depth=4, name=f"w_{p}")
        mem_w = of_w.cons(depth=2).forward(
            tile=Tile(p, 1), depth=2, name=f"mw_{p}", to_stream=kb.up_mm.stream_dims.B,
        )
        w_fifos.append(of_w)
        of_z = ObjectFifo(c_ty, depth=n_dn if p == 0 else 2, name=f"z_{p}")
        z_chain.append(of_z)
        g_bufs = [Buffer(c_ty, name=f"g_{p}_{t}") for t in range(t_f)]

        if p == 0:
            def lane_first(o_a, o_w, o_z_out, rnd, zf, mf, gf, *gb):
                rnd()
                a_bytes = o_a.acquire(1)
                a_views = _make_views(a_bytes, k_up, cfg.M * cfg.kt_up)
                for t in range(t_f):
                    zf(gb[t])
                    for k in range(k_up):
                        b = o_w.acquire(1)
                        mf(a_views[k], b, gb[t])
                        o_w.release(1)
                    gf(c_elems, gb[t])
                o_a.release(1)

                z_tiles = o_z_out.acquire(n_dn)
                for n in range(n_dn):
                    zf(z_tiles[n])
                for t in range(t_f):
                    for n in range(n_dn):
                        b = o_w.acquire(1)
                        mf(gb[t], b, z_tiles[n])
                        o_w.release(1)
                o_z_out.release(n_dn)

            workers.append(Worker(
                lane_first,
                [of_a_bcast.cons(depth=1), mem_w.cons(), of_z.prod(),
                 kb.conv_even, kb.zero_up, kb.up_mm, kb.gelu_ip, *g_bufs],
                tile=Tile(p, 2), stack_size=0xC00,
            ))
        else:
            acc_bufs = [Buffer(c_ty, name=f"acc_{p}_{n}") for n in range(n_dn)]

            def lane_chain(o_a, o_w, o_z_in, o_z_out, rnd, zf, mf, gf, af, *bufs):
                rnd()
                gb = bufs[:t_f]
                ab = bufs[t_f:]
                a_bytes = o_a.acquire(1)
                a_views = _make_views(a_bytes, k_up, cfg.M * cfg.kt_up)
                for t in range(t_f):
                    zf(gb[t])
                    for k in range(k_up):
                        b = o_w.acquire(1)
                        mf(a_views[k], b, gb[t])
                        o_w.release(1)
                    gf(c_elems, gb[t])
                o_a.release(1)

                for n in range(n_dn):
                    zf(ab[n])
                for t in range(t_f):
                    for n in range(n_dn):
                        b = o_w.acquire(1)
                        mf(gb[t], b, ab[n])
                        o_w.release(1)
                for n in range(n_dn):
                    zi = o_z_in.acquire(1)
                    zo = o_z_out.acquire(1)
                    af(ab[n], zi, zo, c_elems)
                    o_z_in.release(1)
                    o_z_out.release(1)

            workers.append(Worker(
                lane_chain,
                [of_a_bcast.cons(depth=1), mem_w.cons(), z_chain[p - 1].cons(), of_z.prod(),
                 kb.conv_even, kb.zero_up, kb.up_mm, kb.gelu_ip, kb.add, *g_bufs, *acc_bufs],
                tile=Tile(p, 2), stack_size=0xC00,
            ))

    return w_fifos, z_chain[-1], workers


def build_spatial_lanes(cfg: FfnConfig, kb: KernelBank, of_a_bcast: ObjectFifo):
    """P spatial column pairs: up[p] at (p, 2) -> shared L1 -> dn[p] at (p, 3)."""
    b_up_ty = np.ndarray[(cfg.kt_up, cfg.nt_up), np.dtype[bf16]]
    b_dn_ty = np.ndarray[(cfg.kt_dn, cfg.nt_dn), np.dtype[bf16]]
    g_ty = np.ndarray[(cfg.M * cfg.nt_up,), np.dtype[bf16]]
    z_ty = np.ndarray[(cfg.M * cfg.nt_dn,), np.dtype[bf16]]
    g_elems = cfg.M * cfg.nt_up
    z_elems = cfg.M * cfg.nt_dn
    n_up, k_up, k_dn, n_dn = cfg.n_up, cfg.k_up, cfg.k_dn, cfg.n_dn

    wup_fifos, wdn_fifos, z_chain, workers = [], [], [], []
    for p in range(cfg.lanes):
        of_wup = ObjectFifo(b_up_ty, depth=4, name=f"wup_{p}")
        mem_wup = of_wup.cons(depth=2).forward(
            tile=Tile(p, 1), depth=2, name=f"mwup_{p}", to_stream=kb.up_mm.stream_dims.B,
        )
        of_wdn = ObjectFifo(b_dn_ty, depth=4, name=f"wdn_{p}")
        mem_wdn = of_wdn.cons(depth=2).forward(
            tile=Tile(p, 1), depth=2, name=f"mwdn_{p}", to_stream=kb.dn_mm.stream_dims.B,
        )
        wup_fifos.append(of_wup)
        wdn_fifos.append(of_wdn)

        of_g = ObjectFifo(g_ty, depth=2, name=f"g_{p}")
        of_z = ObjectFifo(z_ty, depth=n_dn if p == 0 else 2, name=f"z_{p}")
        z_chain.append(of_z)

        def up_body(o_a, o_b, o_g, rnd, zf, mf, gf):
            rnd()
            a_bytes = o_a.acquire(1)
            a_views = _make_views(a_bytes, k_up, cfg.M * cfg.kt_up)
            for _ in range_(n_up):
                gb = o_g.acquire(1)
                zf(gb)
                for k in range(k_up):
                    b = o_b.acquire(1)
                    mf(a_views[k], b, gb)
                    o_b.release(1)
                gf(g_elems, gb)
                o_g.release(1)
            o_a.release(1)

        workers.append(Worker(
            up_body,
            [of_a_bcast.cons(depth=1), mem_wup.cons(), of_g.prod(),
             kb.conv_even, kb.zero_up, kb.up_mm, kb.gelu_ip],
            tile=Tile(p, 2), stack_size=0xC00,
        ))

        if p == 0:
            def dn_first(o_g, o_b, o_z_out, rnd, zf, mf):
                rnd()
                z_tiles = o_z_out.acquire(n_dn)
                for n in range(n_dn):
                    zf(z_tiles[n])
                for _ in range_(k_dn):
                    g = o_g.acquire(1)
                    for n in range(n_dn):
                        b = o_b.acquire(1)
                        mf(g, b, z_tiles[n])
                        o_b.release(1)
                    o_g.release(1)
                o_z_out.release(n_dn)

            workers.append(Worker(
                dn_first,
                [of_g.cons(), mem_wdn.cons(), of_z.prod(),
                 kb.conv_even, kb.zero_dn, kb.dn_mm],
                tile=Tile(p, 3), stack_size=0xC00,
            ))
        else:
            acc_bufs = [Buffer(z_ty, name=f"acc_{p}_{n}") for n in range(n_dn)]

            def dn_chain(o_g, o_b, o_z_in, o_z_out, rnd, zf, mf, af, *ab):
                rnd()
                for n in range(n_dn):
                    zf(ab[n])
                for _ in range_(k_dn):
                    g = o_g.acquire(1)
                    for n in range(n_dn):
                        b = o_b.acquire(1)
                        mf(g, b, ab[n])
                        o_b.release(1)
                    o_g.release(1)
                for n in range(n_dn):
                    zi = o_z_in.acquire(1)
                    zo = o_z_out.acquire(1)
                    af(ab[n], zi, zo, z_elems)
                    o_z_in.release(1)
                    o_z_out.release(1)

            workers.append(Worker(
                dn_chain,
                [of_g.cons(), mem_wdn.cons(), z_chain[p - 1].cons(), of_z.prod(),
                 kb.conv_even, kb.zero_dn, kb.dn_mm, kb.add, *acc_bufs],
                tile=Tile(p, 3), stack_size=0xC00,
            ))

    return wup_fifos, wdn_fifos, z_chain[-1], workers


def build_residual_out_stage(
    cfg: FfnConfig, kb: KernelBank, of_z_last: ObjectFifo, mt_col: int, add_tile: Tile,
):
    """Un-tile micro-tiled Z once on MemTile, add residual X, and stream out."""
    c_ty = np.ndarray[(cfg.M * cfg.nt_dn,), np.dtype[bf16]]
    c_elems = cfg.M * cfg.nt_dn

    of_z_rowmaj = of_z_last.cons().forward(
        tile=Tile(mt_col, 1), name="z_untile", to_stream=kb.dn_mm.stream_dims.C,
    )
    of_x_res = ObjectFifo(c_ty, depth=2, name="x_res")
    of_out = ObjectFifo(c_ty, depth=2, name="out")

    def add_res_body(o_z, o_x, o_out, af):
        for _ in range_(cfg.n_dn):
            z = o_z.acquire(1)
            x = o_x.acquire(1)
            o = o_out.acquire(1)
            af(z, x, o, c_elems)
            o_z.release(1)
            o_x.release(1)
            o_out.release(1)

    add_w = Worker(
        add_res_body, [of_z_rowmaj.cons(), of_x_res.cons(), of_out.prod(), kb.add],
        tile=add_tile, stack_size=0xC00,
    )
    return of_x_res, of_out, add_w


def build_ffn_kernel(cfg: FfnConfig):
    """Compile-time factory returning a specialized @iron.jit ffn(X, Wup, Wdn, Out)."""
    kb = KernelBank(cfg)
    flat_x_ty = np.ndarray[(cfg.M * cfg.D,), np.dtype[bf16]]
    flat_wup_ty = np.ndarray[(cfg.D * cfg.F,), np.dtype[bf16]]
    flat_wdn_ty = np.ndarray[(cfg.F * cfg.D,), np.dtype[bf16]]

    @iron.jit
    def ffn(X_in: In, Wup_in: In, Wdn_in: In, Out_out: Out):
        if cfg.mode == "unified":
            rms_tile = Tile(0, 3)
            mt_rms_col = 0
            mt_out_col = cfg.lanes - 1
            add_tile = Tile(cfg.lanes - 1, 3)
            x_rms_shim = Tile(0, 0)
            x_res_shim = Tile(cfg.lanes - 1, 0)
            out_shim = Tile(cfg.lanes - 1, 0)
        else:
            aux_col = cfg.lanes
            rms_tile = Tile(aux_col, 2)
            mt_rms_col = aux_col
            mt_out_col = aux_col
            add_tile = Tile(aux_col, 3)
            x_rms_shim = Tile(aux_col, 0)
            x_res_shim = Tile(aux_col, 0)
            out_shim = Tile(aux_col, 0)

        of_x_rms, of_a_bcast, rms_w = build_rms_broadcast_stage(
            cfg, kb, mt_rms_col, rms_tile,
        )
        if cfg.mode == "unified":
            w_fifos, of_z_last, lane_workers = build_unified_lanes(cfg, kb, of_a_bcast)
        else:
            wup_fifos, wdn_fifos, of_z_last, lane_workers = build_spatial_lanes(
                cfg, kb, of_a_bcast,
            )

        of_x_res, of_out, add_w = build_residual_out_stage(
            cfg, kb, of_z_last, mt_out_col, add_tile,
        )

        x_row_tap = TensorAccessPattern.full((cfg.M, cfg.D)).tile((1, cfg.D))
        x_col_tap = TensorAccessPattern.full((cfg.M, cfg.D)).tile((cfg.M, cfg.nt_dn))
        wup_grid = TensorAccessPattern.full((cfg.D, cfg.F)).tile((cfg.kt_up, cfg.nt_up))
        wdn_grid = TensorAccessPattern.full((cfg.F, cfg.D)).tile((cfg.kt_dn, cfg.nt_dn))

        wup_taps = [
            wup_grid[:, p * cfg.n_up:(p + 1) * cfg.n_up, :, :].permute((1, 0, 2, 3))
            for p in range(cfg.lanes)
        ]
        wdn_taps = [
            wdn_grid[p * cfg.k_dn:(p + 1) * cfg.k_dn, :, :, :]
            for p in range(cfg.lanes)
        ]

        if cfg.mode == "unified":
            def sequence(Xh, Wuph, Wdnh, Outh, x_rms_p, x_res_p, out_c, *w_prods):
                tg = TaskGroup()
                x_rms_p.fill(Xh, tap=x_row_tap, group=tg)
                x_res_p.fill(Xh, tap=x_col_tap, group=tg)
                for p in range(cfg.lanes):
                    w_prods[p].fill(Wuph, tap=wup_taps[p], group=tg)
                    w_prods[p].fill(Wdnh, tap=wdn_taps[p], group=tg)
                out_c.drain(Outh, tap=x_col_tap, wait=True, group=tg)
                tg.finish()

            shim_handles = [f.prod(tile=Tile(p, 0)) for p, f in enumerate(w_fifos)]
        else:
            def sequence(Xh, Wuph, Wdnh, Outh, x_rms_p, x_res_p, out_c, *w_prods):
                wup_p, wdn_p = w_prods[:cfg.lanes], w_prods[cfg.lanes:]
                tg = TaskGroup()
                x_rms_p.fill(Xh, tap=x_row_tap, group=tg)
                x_res_p.fill(Xh, tap=x_col_tap, group=tg)
                for p in range(cfg.lanes):
                    wup_p[p].fill(Wuph, tap=wup_taps[p], group=tg)
                    wdn_p[p].fill(Wdnh, tap=wdn_taps[p], group=tg)
                out_c.drain(Outh, tap=x_col_tap, wait=True, group=tg)
                tg.finish()

            shim_handles = (
                [f.prod(tile=Tile(p, 0)) for p, f in enumerate(wup_fifos)]
                + [f.prod(tile=Tile(p, 0)) for p, f in enumerate(wdn_fifos)]
            )

        rt = Runtime(
            sequence,
            [
                flat_x_ty, flat_wup_ty, flat_wdn_ty, flat_x_ty,
                of_x_rms.prod(tile=x_rms_shim),
                of_x_res.prod(tile=x_res_shim),
                of_out.cons(tile=out_shim),
                *shim_handles,
            ],
        )
        prog = Program(DEV, rt, workers=[rms_w, *lane_workers, add_w])
        return prog.resolve_program()

    return ffn


def ref_ffn(X: np.ndarray, Wup: np.ndarray, Wdn: np.ndarray) -> np.ndarray:
    xf = X.astype(np.float32)
    n = xf / np.sqrt((xf * xf).mean(axis=1, keepdims=True) + 1e-6)
    n = n.astype(bf16).astype(np.float32)
    h = n @ Wup.astype(np.float32)
    g = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h**3)))
    z = g.astype(bf16).astype(np.float32) @ Wdn.astype(np.float32)
    return (z.astype(bf16).astype(np.float32) + xf).astype(bf16)


def _max_ulp_ge1(got_bf16: np.ndarray, ref_bf16: np.ndarray, ref_f32: np.ndarray) -> int:
    mask = np.abs(ref_f32) >= 1.0
    gb = got_bf16[mask].view(np.uint16).astype(np.int32)
    rb = ref_bf16[mask].view(np.uint16).astype(np.int32)
    go = np.where((gb & 0x8000) != 0, 0x8000 - (gb & 0x7FFF), 0x8000 + (gb & 0x7FFF))
    ro = np.where((rb & 0x8000) != 0, 0x8000 - (rb & 0x7FFF), 0x8000 + (rb & 0x7FFF))
    return int(np.abs(go - ro).max())


def run_benchmark(cfg: FfnConfig, iters: int = 100, warmup: int = 10):
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (cfg.M, cfg.D)).astype(bf16)
    Wup = rng.normal(0, 0.08, (cfg.D, cfg.F)).astype(bf16)
    Wdn = rng.normal(0, 0.08, (cfg.F, cfg.D)).astype(bf16)
    REF_BF16 = ref_ffn(X, Wup, Wdn)
    REF = REF_BF16.astype(np.float32)

    Xt = iron.tensor(X.reshape(-1), dtype=bf16, device="npu")
    Wupt = iron.tensor(Wup.reshape(-1), dtype=bf16, device="npu")
    Wdnt = iron.tensor(Wdn.reshape(-1), dtype=bf16, device="npu")
    Outt = iron.zeros(cfg.M * cfg.D, dtype=bf16, device="npu")

    ffn = build_ffn_kernel(cfg)
    t0 = time.perf_counter()
    hdl, _ = ffn(Xt, Wupt, Wdnt, Outt)
    compile_s = time.perf_counter() - t0

    got_bf16 = Outt.numpy().reshape(cfg.M, cfg.D)
    got = got_bf16.astype(np.float32)
    err = np.abs(got - REF)
    max_abs = float(err.max())
    mean_abs = float(err.mean())
    p99_abs = float(np.percentile(err, 99))
    cos = float(np.dot(got.ravel(), REF.ravel()) / (np.linalg.norm(got) * np.linalg.norm(REF)))
    max_ulp = _max_ulp_ge1(got_bf16, REF_BF16, REF)

    args = [Xt, Wupt, Wdnt, Outt]
    for _ in range(warmup):
        ffn(*args)

    call_us, hw_us = [], []
    for _ in range(iters):
        ta = time.perf_counter()
        _, k_res = ffn(*args)
        call_us.append((time.perf_counter() - ta) * 1e6)
        hw_us.append(k_res.npu_time / 1e3)

    def _bench_chain(n_layers: int, ch_iters: int) -> float:
        runs = [(hdl, args)] * n_layers
        for _ in range(3):
            DefaultNPURuntime.run_chain(runs)
        ts = [DefaultNPURuntime.run_chain(runs).npu_time / 1e3 / n_layers for _ in range(ch_iters)]
        return float(np.median(ts))

    med_call = float(np.median(call_us))
    med_hw = float(np.median(hw_us))
    med_ch8 = _bench_chain(8, 25)
    med_ch24 = _bench_chain(24, 15)
    unique_bytes = (cfg.M * cfg.D + cfg.D * cfg.F + cfg.F * cfg.D + cfg.M * cfg.D) * 2
    flops = 2 * cfg.M * cfg.D * cfg.F * 2
    bw_hw = unique_bytes / (med_hw * 1e-6) / 1e9
    bw_ch8 = unique_bytes / (med_ch8 * 1e-6) / 1e9
    gf_ch8 = flops / (med_ch8 * 1e-6) / 1e9
    cores = (cfg.lanes + 2) if cfg.mode == "unified" else (2 * cfg.lanes + 2)
    w_ch = cfg.lanes if cfg.mode == "unified" else 2 * cfg.lanes
    tile_str = (
        f"{cfg.kt_up}x{cfg.nt_up}"
        if (cfg.kt_up, cfg.nt_up) == (cfg.kt_dn, cfg.nt_dn)
        else f"{cfg.kt_up}x{cfg.nt_up}/{cfg.kt_dn}x{cfg.nt_dn}"
    )
    print(
        f"[{cfg.mode:7s} P={cfg.lanes} {tile_str:13s} c={cores:2d} wch={w_ch}] "
        f"call={med_call:5.1f}us hw={med_hw:5.1f}us({bw_hw:4.1f}GB/s) "
        f"ch8={med_ch8:5.1f}us/L({bw_ch8:4.1f}GB/s {gf_ch8:5.1f}GF) "
        f"ch24={med_ch24:5.1f}us/L({med_ch24*24/1e3:4.2f}ms) | "
        f"max={max_abs:.4f} mean={mean_abs:.5f} ulp={max_ulp} cos={cos:.7f} ({compile_s:.2f}s)",
        flush=True,
    )
    return {
        "mode": cfg.mode, "lanes": cfg.lanes, "tile": tile_str, "cores": cores, "w_ch": w_ch,
        "call_us": med_call, "hw_us": med_hw, "chain8_us": med_ch8, "chain24_us": med_ch24,
        "bw_hw": bw_hw, "bw_ch8": bw_ch8, "gf_ch8": gf_ch8,
        "max_abs": max_abs, "mean_abs": mean_abs, "p99_abs": p99_abs, "max_ulp": max_ulp, "cos": cos,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unified", "spatial"], default="unified")
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--kt", type=int, default=64)
    ap.add_argument("--nt", type=int, default=64)
    ap.add_argument("--kt-dn", type=int, default=None)
    ap.add_argument("--nt-dn", type=int, default=None)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--bench-all", action="store_true")
    args = ap.parse_args()

    if args.bench_all:
        configs = [
            FfnConfig(lanes=2, mode="spatial", KT=64, NT=64),
            FfnConfig(lanes=4, mode="unified", KT=64, NT=64),
            FfnConfig(lanes=4, mode="spatial", KT=64, NT=64),
            FfnConfig(lanes=8, mode="unified", KT=64, NT=64),
            FfnConfig(lanes=4, mode="spatial", KT=64, NT=128, KT_dn=128, NT_dn=64),
        ]
        for cfg in configs:
            run_benchmark(cfg, iters=args.iters)
    else:
        run_benchmark(
            FfnConfig(
                lanes=args.lanes, mode=args.mode,
                KT=args.kt, NT=args.nt, KT_dn=args.kt_dn, NT_dn=args.nt_dn,
            ),
            iters=args.iters,
        )


if __name__ == "__main__":
    main()
