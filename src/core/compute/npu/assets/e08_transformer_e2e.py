#!/usr/bin/env python3
"""EmbeddingGemma 2 Full 24-Layer Interleaved Transformer E2E on Ryzen AI NPU2.

Chains all 24 interleaved layers (Attention x GeGLU FFN) + final head on NPU2:
  - Layers 0..23:
      Attention (H_a -> H_b): qkv_pre -> qkv_norm_rope -> attn_gqa_core -> o_post
        * 20 sliding layers (not in {5,11,17,23}): d=256, H_q=4, H_kv=2, W=512, theta=1e4
        *  4 full layers     (    in {5,11,17,23}): d=512, H_q=4, H_kv=1, full, theta=1e6
      FFN (H_b -> H_a):
        * ffn_pre_mm(H_b, pre_ffn, W_gate -> Gate)   [20c Pre-RMSNorm + 2D 4x4 GEMM]
        * ffn_pre_mm(H_b, pre_ffn, W_up   -> Up)
        * ffn_geglu(Gate, Up -> Act)                 [8c GeLU_tanh(Gate) * Up]
        * o_post_full(Act, W_down, H_b, post_ffn -> H_a)  [reuses e07 20c K-split]
      (+ exact PLE block & layer_scalar when running real model weights)
  - Final Head:
      final_norm(H_a[:n_valid]) -> mean_pool -> out_proj(512 -> 768) -> L2 norm.

Total XCLBIN count = 10 (8 from AttnKernelBank + 2 FFN kernels <= 16 hwctx limit).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np

_ASSET_DIR = Path(__file__).resolve().parent
if str(_ASSET_DIR) not in sys.path:
    sys.path.insert(0, str(_ASSET_DIR))

import aie.iron as iron
from aie import ir
from aie.dialects import memref
from aie.extras.dialects.arith import constant
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import In, Out, kernels as K
from aie.iron.buffer import Buffer
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifo
from aie.iron.device import Tile
from aie.iron.program import Program
from aie.iron.runtime import Runtime, TaskGroup
from aie.iron.worker import Worker

from e07_attn import (
    AttnKernelBank,
    DEV,
    EPS_RMS,
    FULL_LAYER_INDICES,
    KT_MM,
    LAYER_TYPES_24,
    NT_MM,
    _f32,
    _i32,
    _npu_tensor,
    _npu_zeros,
    _rms_norm_f32,
    bf16,
    err_metrics,
    make_layer_config,
    make_rope_lut,
    make_swa_mask,
)


def _gelu_tanh_f32(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    u = 0.7978845608028654 * (x32 + 0.044715 * x32 * x32 * x32)
    return 0.5 * x32 * (1.0 + np.tanh(u))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    n = float(np.linalg.norm(x32))
    return (x32 / n) if n > 0.0 else x32


def build_ffn_pre_mm(S: int = 128, D: int = 512, N_dim: int = 2048, C: int = 4):
    """20-core Pre-RMSNorm + 2D 4xC row-major GEMM ([S, D] -> [S, N_dim])."""
    R, MT, KT, NT = 4, 32, KT_MM, NT_MM
    M_pass = R * MT
    if S % M_pass != 0 or D % KT != 0 or N_dim % (C * NT) != 0:
        raise ValueError(f"Unsupported shape for build_ffn_pre_mm: S={S}, D={D}, N={N_dim}, C={C}")
    n_passes = S // M_pass
    K_tiles = D // KT
    N_col = N_dim // C
    N_tiles_col = N_col // NT
    bands = MT // 4

    mm_k = K.mm(MT, KT, NT, bf16, bf16, b_col_maj=False)
    _, lay_b, _ = mm_k.contract.layouts
    z_k, ce_k = K.zero((MT * NT,), bf16), K.conv_even()
    rms_k, mul_k = K.rms_norm_eps(cols=D), K.mul_sized(D)

    row_ty = np.ndarray[(D,), np.dtype[bf16]]
    mat_i8 = np.ndarray[(MT * D * 2,), np.dtype[np.int8]]
    b_ty = np.ndarray[(KT * NT,), np.dtype[bf16]]
    c_ty = np.ndarray[(MT * NT,), np.dtype[bf16]]
    c2_ty = np.ndarray[(2 * MT * NT,), np.dtype[bf16]]

    tap_n1 = TensorAccessPattern([MT * D * 2], 0, [K_tiles, MT, KT * 2], [KT * 2, D * 2, 1])
    tap_a2 = TensorAccessPattern([MT * D * 2], 0, [K_tiles * bands, KT // 8, 4, 16], [4 * KT * 2, 16, KT * 2, 1])
    tap_c1 = TensorAccessPattern([MT * NT], 0, [bands, 4, 8, 8], [256, 8, 32, 1])
    a_cols = [C + r for r in range(R)]

    H_ty = np.ndarray[(S * D,), np.dtype[bf16]]
    NW_ty = np.ndarray[(D,), np.dtype[bf16]]
    W_ty = np.ndarray[(D * N_dim,), np.dtype[bf16]]
    Out_ty = np.ndarray[(S * N_dim,), np.dtype[bf16]]

    @iron.jit
    def ffn_pre_mm(H_in: In, NW_in: In, W_in: In, Out_out: Out):
        of_h = [ObjectFifo(row_ty, depth=2, name=f"fp_h_{r}") for r in range(R)]
        of_n_raw = [ObjectFifo(mat_i8, depth=1, to_stream=tap_n1, name=f"fp_nr_{r}") for r in range(R)]
        of_a_bc = [
            of_n_raw[r].cons(depth=1).forward(tile=Tile(a_cols[r], 1), obj_type=mat_i8, depth=1, to_stream=tap_a2, name=f"fp_ab_{r}")
            for r in range(R)
        ]
        pn_tmp = [Buffer(row_ty, name=f"fp_t_{r}") for r in range(R)]
        pn_wb = [Buffer(row_ty, name=f"fp_wb_{r}") for r in range(R)]

        of_b_in = [ObjectFifo(b_ty, depth=4, name=f"fp_bi_{c}") for c in range(C)]
        of_b_mem = [of_b_in[c].cons(depth=2).forward(tile=Tile(c, 1), depth=2, to_stream=lay_b.stream, name=f"fp_bm_{c}") for c in range(C)]

        of_c_out = [[ObjectFifo(c2_ty, depth=2, to_stream=tap_c1, name=f"fp_co_{c}_{pr}") for pr in range(2)] for c in range(C)]
        of_c_in = [[None] * R for _ in range(C)]
        for c in range(C):
            for pr in range(2):
                subs = of_c_out[c][pr].prod().join(
                    [0, MT * NT], tile=Tile(c, 1), depths=[2, 2], obj_types=[c_ty, c_ty],
                    names=[f"fp_ci_{c}_{pr*2}", f"fp_ci_{c}_{pr*2+1}"],
                )
                of_c_in[c][pr * 2], of_c_in[c][pr * 2 + 1] = subs[0], subs[1]

        def pn_body(oh, on_out, tmp, wbuf, rms_fn, mul_fn, ce_fn):
            ce_fn()
            c_d, c_eps = _i32(D), _f32(EPS_RMS)
            r_ty = ir.MemRefType.get([D], ir.BF16Type.get())
            w_in = oh.acquire(1)
            for i in range_(D):
                wbuf[i] = w_in[i]
            oh.release(1)
            for _ in range_(n_passes):
                nb = on_out.acquire(1)
                for m in range(MT):
                    xr = oh.acquire(1)
                    nv = memref.view(r_ty, nb, constant(m * D * 2, index=True), [])
                    rms_fn(xr, tmp, c_d, c_eps)
                    mul_fn(tmp, wbuf, nv, c_d)
                    oh.release(1)
                on_out.release(1)

        def gemm_body(oa, ob, oc, mm_fn, z_fn, ce_fn):
            ce_fn()
            a_vty = ir.MemRefType.get([MT * KT], ir.BF16Type.get())
            for _ in range_(n_passes):
                ab = oa.acquire(1)
                a_views = [memref.view(a_vty, ab, constant(kt * MT * KT * 2, index=True), []) for kt in range(K_tiles)]
                for _ in range_(N_tiles_col):
                    ct = oc.acquire(1)
                    z_fn(ct)
                    for kt in range(K_tiles):
                        w = ob.acquire(1)
                        mm_fn(a_views[kt], w, ct)
                        ob.release(1)
                    oc.release(1)
                oa.release(1)

        workers = [
            Worker(pn_body, fn_args=[of_h[r].cons(), of_n_raw[r].prod(), pn_tmp[r], pn_wb[r], rms_k, mul_k, ce_k], tile=Tile(C, 2 + r))
            for r in range(R)
        ] + [
            Worker(gemm_body, fn_args=[of_a_bc[r].cons(depth=1), of_b_mem[c].cons(), of_c_in[c][r].prod(), mm_k, z_k, ce_k], tile=Tile(c, 2 + r))
            for c in range(C)
            for r in range(R)
        ]

        h_hdls = [of_h[r].prod(tile=Tile(a_cols[r], 0)) for r in range(R)]
        b_hdls = [of_b_in[c].prod(tile=Tile(c, 0)) for c in range(C)]
        c_hdls = [of_c_out[c][pr].cons(tile=Tile(c, 0)) for c in range(C) for pr in range(2)]

        def seq(H_t, NW_t, W_t, Out_t, *hdls):
            hhs, bhs, chs = hdls[:R], hdls[R : R + C], hdls[R + C :]
            for p in range(n_passes):
                tg = TaskGroup()
                for r in range(R):
                    if p == 0:
                        hhs[r].fill(NW_t, group=tg)
                    hhs[r].fill(H_t, tap=TensorAccessPattern([S * D], (p * M_pass + r * MT) * D, [MT * D], [1]), group=tg)
                for c in range(C):
                    tap_w = TensorAccessPattern([D * N_dim], c * N_col, [N_tiles_col, K_tiles, KT, NT], [NT, KT * N_dim, N_dim, 1])
                    bhs[c].fill(W_t, tap=tap_w, group=tg)
                    for pr in range(2):
                        ch = chs[c * 2 + pr]
                        row_off = p * M_pass + pr * (2 * MT)
                        tap_c = TensorAccessPattern(
                            [S * N_dim], row_off * N_dim + c * N_col, [N_tiles_col, 2 * MT, NT], [NT, N_dim, 1]
                        )
                        ch.drain(Out_t, tap=tap_c, wait=True, group=tg)
                tg.finish()

        rt = Runtime(seq, [H_ty, NW_ty, W_ty, Out_ty, *h_hdls, *b_hdls, *c_hdls])
        return Program(DEV, rt, workers=workers).resolve_program()

    return ffn_pre_mm


def build_ffn_geglu(S: int = 128, F: int = 2048):
    """8-column vectorized GeGLU kernel: Out = gelu_tanh(Gate) * Up on BF16."""
    C = 8
    total = S * F
    if total % (C * 1024) != 0:
        raise ValueError(f"S*F={total} must divide by {C * 1024}")
    chunk = total // C
    n_vecs = chunk // 1024

    gelu_k = K.gelu(1024)
    mul_k = K.mul_sized(1024)
    ce_k = K.conv_even()

    vec_ty = np.ndarray[(1024,), np.dtype[bf16]]
    T_ty = np.ndarray[(total,), np.dtype[bf16]]

    @iron.jit
    def ffn_geglu(G_in: In, U_in: In, O_out: Out):
        of_g = [ObjectFifo(vec_ty, depth=2, name=f"gg_g_{c}") for c in range(C)]
        of_u = [ObjectFifo(vec_ty, depth=2, name=f"gg_u_{c}") for c in range(C)]
        of_o = [ObjectFifo(vec_ty, depth=2, name=f"gg_o_{c}") for c in range(C)]
        tmp_bufs = [Buffer(vec_ty, name=f"gg_t_{c}") for c in range(C)]

        def gg_body(og, ou, oo, tmp, g_fn, m_fn, ce_fn):
            ce_fn()
            c_sz = _i32(1024)
            for _ in range_(n_vecs):
                gv = og.acquire(1)
                uv = ou.acquire(1)
                ov = oo.acquire(1)
                g_fn(gv, tmp)
                m_fn(tmp, uv, ov, c_sz)
                og.release(1)
                ou.release(1)
                oo.release(1)

        workers = [
            Worker(gg_body, fn_args=[of_g[c].cons(), of_u[c].cons(), of_o[c].prod(), tmp_bufs[c], gelu_k, mul_k, ce_k], tile=Tile(c, 2))
            for c in range(C)
        ]

        g_hdls = [of_g[c].prod(tile=Tile(c, 0)) for c in range(C)]
        u_hdls = [of_u[c].prod(tile=Tile(c, 0)) for c in range(C)]
        o_hdls = [of_o[c].cons(tile=Tile(c, 0)) for c in range(C)]

        def seq(G_t, U_t, O_t, *hdls):
            ghs, uhs, ohs = hdls[:C], hdls[C : 2 * C], hdls[2 * C :]
            tg = TaskGroup()
            for c in range(C):
                tap = TensorAccessPattern([total], c * chunk, [chunk], [1])
                ghs[c].fill(G_t, tap=tap, group=tg)
                uhs[c].fill(U_t, tap=tap, group=tg)
                ohs[c].drain(O_t, tap=tap, wait=True, group=tg)
            tg.finish()

        rt = Runtime(seq, [T_ty, T_ty, T_ty, *g_hdls, *u_hdls, *o_hdls])
        return Program(DEV, rt, workers=workers).resolve_program()

    return ffn_geglu


class TransformerKernelBank:
    """10-XCLBIN full 24-layer EmbeddingGemma 2 NPU2 bank (Attention + FFN)."""

    def __init__(self, S: int = 128):
        self.S = S
        self.attn = AttnKernelBank(S)
        self.ffn_pre_mm = build_ffn_pre_mm(S=S, D=512, N_dim=2048, C=4)
        self.ffn_geglu = build_ffn_geglu(S=S, F=2048)

        self.Gate = _npu_zeros(S * 2048)
        self.Up = _npu_zeros(S * 2048)
        self.Act = _npu_zeros(S * 2048)

    def make_synthetic_layers(self, seed: int = 7) -> tuple[list[dict], np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        layers = []
        for li in range(24):
            is_full = li in FULL_LAYER_INDICES
            cfg = make_layer_config(self.S, is_full=is_full, window=512)
            w = self.attn.make_layer_weights_host(rng, cfg, scale=0.03)
            w["pre_ffn"] = rng.normal(1.0, 0.05, (512,)).astype(bf16)
            w["W_gate"] = rng.normal(0.0, 0.03, (512, 2048)).astype(bf16)
            w["W_up"] = rng.normal(0.0, 0.03, (512, 2048)).astype(bf16)
            w["W_down"] = rng.normal(0.0, 0.03, (2048, 512)).astype(bf16)
            w["post_ffn"] = rng.normal(0.0, 0.05, (512,)).astype(bf16)
            layers.append(w)
        final_norm = rng.normal(1.0, 0.05, (512,)).astype(bf16)
        out_proj = rng.normal(0.0, 0.03, (512, 768)).astype(bf16)
        return layers, final_norm, out_proj

    def upload_layer(self, w_host: dict[str, np.ndarray]) -> dict[str, object]:
        keys = (
            "in_norm", "W_qkv", "q_norm", "k_norm", "W_o", "post_attn",
            "pre_ffn", "W_gate", "W_up", "W_down", "post_ffn",
        )
        return {k: _npu_tensor(w_host[k]) for k in keys}

    def run_ffn_layer(self, H_in, w_dev: dict[str, object], H_out) -> float:
        t_us = 0.0
        _, r1 = self.ffn_pre_mm(H_in, w_dev["pre_ffn"], w_dev["W_gate"], self.Gate)
        t_us += r1.npu_time / 1e3
        _, r2 = self.ffn_pre_mm(H_in, w_dev["pre_ffn"], w_dev["W_up"], self.Up)
        t_us += r2.npu_time / 1e3
        _, r3 = self.ffn_geglu(self.Gate, self.Up, self.Act)
        t_us += r3.npu_time / 1e3
        _, r4 = self.attn.o_post_full(self.Act, w_dev["W_down"], H_in, w_dev["post_ffn"], H_out)
        t_us += r4.npu_time / 1e3
        return t_us

    def export_bundle(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_l = make_layer_config(self.S, is_full=False, window=512)
        cfg_f = make_layer_config(self.S, is_full=True, window=-1)
        rng = np.random.default_rng(0)
        wl = self.upload_layer(
            {**self.attn.make_layer_weights_host(rng, cfg_l),
             "pre_ffn": np.ones((512,), bf16), "W_gate": np.zeros((512, 2048), bf16),
             "W_up": np.zeros((512, 2048), bf16), "W_down": np.zeros((2048, 512), bf16),
             "post_ffn": np.ones((512,), bf16)}
        )
        wf = self.upload_layer(
            {**self.attn.make_layer_weights_host(rng, cfg_f),
             "pre_ffn": np.ones((512,), bf16), "W_gate": np.zeros((512, 2048), bf16),
             "W_up": np.zeros((512, 2048), bf16), "W_down": np.zeros((2048, 512), bf16),
             "post_ffn": np.ones((512,), bf16)}
        )
        lut_l = _npu_tensor(make_rope_lut(self.S, cfg_l.d, cfg_l.theta))
        lut_f = _npu_tensor(make_rope_lut(self.S, cfg_f.d, cfg_f.theta))
        mask_l = _npu_tensor(make_swa_mask(self.S, 512, is_full=False))
        mask_f = _npu_tensor(make_swa_mask(self.S, -1, is_full=True))

        specs = (
            self.attn.layer_runs_specs(False, self.attn.H_a, self.attn.H_b, wl, lut_l, mask_l)
            + self.attn.layer_runs_specs(True, self.attn.H_a, self.attn.H_b, wf, lut_f, mask_f)
            + [
                ("ffn_pre_mm", self.ffn_pre_mm, [self.attn.H_b, wl["pre_ffn"], wl["W_gate"], self.Gate]),
                ("ffn_geglu", self.ffn_geglu, [self.Gate, self.Up, self.Act]),
            ]
        )
        manifest = {"seq_len": self.S, "hidden": 512, "ffn": 2048, "kernels": {}}
        for name, fn, args in specs:
            hdl, _ = fn(*args)
            xcl_dst = out_dir / f"{name}.xclbin"
            inst_dst = out_dir / f"{name}.insts.bin"
            shutil.copyfile(hdl.xclbin_path, xcl_dst)
            if hdl.insts_path is None:
                raise RuntimeError(f"Kernel {name} did not produce insts_path")
            shutil.copyfile(hdl.insts_path, inst_dst)
            manifest["kernels"][name] = {
                "xclbin": xcl_dst.name,
                "insts": inst_dst.name,
                "kernel_name": hdl.kernel_name or "MLIR_AIE",
            }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[Export] Saved {len(specs)} IRON kernels to {out_dir}", flush=True)


def ref_transformer_forward(
    H0: np.ndarray,
    layers: list[dict],
    final_norm: np.ndarray,
    out_proj: np.ndarray,
    S: int,
    n_valid: int,
    luts: dict[bool, np.ndarray],
    masks: dict[bool, np.ndarray],
    ple_data: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """FP32 reference through all 24 interleaved layers + PLE + final head."""
    from e07_attn import ref_attn_layer

    H = H0.astype(np.float32)
    emb16_f = H0[:n_valid].astype(np.float32)
    ple_pre = 1.0 / math.sqrt(512.0)

    for li in range(24):
        is_full = li in FULL_LAYER_INDICES
        cfg = make_layer_config(S, is_full=is_full, window=512)
        w = layers[li]
        H_bf = ref_attn_layer(
            H.astype(bf16), w["in_norm"], w["W_qkv"], w["q_norm"], w["k_norm"],
            luts[is_full], masks[is_full], w["W_o"], w["post_attn"], cfg,
        )
        H_att = H_bf.astype(np.float32)

        x_f = (_rms_norm_f32(H_att) * w["pre_ffn"].astype(np.float32)).astype(bf16).astype(np.float32)
        gate = (x_f @ w["W_gate"].astype(np.float32)).astype(bf16).astype(np.float32)
        up = (x_f @ w["W_up"].astype(np.float32)).astype(bf16).astype(np.float32)
        act = (_gelu_tanh_f32(gate) * up).astype(bf16).astype(np.float32)
        y_ffn = (act @ w["W_down"].astype(np.float32)).astype(bf16)
        H = (_rms_norm_f32(y_ffn) * w["post_ffn"].astype(np.float32) + H_att)

        if ple_data is not None:
            P = 512
            ple_w = ple_data["ple_proj"][li * P : (li + 1) * P]  # [512, 512] [out, in]
            g = emb16_f @ ple_w.T
            pl = _rms_norm_f32(g * ple_pre) * ple_data["ple_norm"]
            h_v = H[:n_valid]
            act_p = _gelu_tanh_f32(h_v @ w["wpgate"].T) * pl
            y_p = act_p @ w["wpproj"].T
            h_v = (h_v + _rms_norm_f32(y_p) * w["ple_post"]) * w["scalar"]
            H = np.zeros((S, 512), dtype=np.float32)
            H[:n_valid] = h_v

    normed = _rms_norm_f32(H[:n_valid]) * final_norm.astype(np.float32)
    pooled = normed.mean(axis=0)
    emb = _l2_normalize(pooled @ out_proj.astype(np.float32))
    return H.astype(bf16), emb


def run_synthetic(bank: TransformerKernelBank, warmup: int = 1, iters: int = 3) -> None:
    S = bank.S
    print(f"\n=== EmbeddingGemma 2 Full 24L Interleaved Transformer (Synthetic, S={S}) ===", flush=True)
    layers, final_norm, out_proj = bank.make_synthetic_layers(seed=42)
    rng = np.random.default_rng(100)
    H0 = rng.normal(0.0, 0.5, (S, 512)).astype(bf16)

    cfg_l = make_layer_config(S, is_full=False, window=512)
    cfg_f = make_layer_config(S, is_full=True, window=-1)
    luts = {False: make_rope_lut(S, cfg_l.d, cfg_l.theta), True: make_rope_lut(S, cfg_f.d, cfg_f.theta)}
    masks = {False: make_swa_mask(S, 512, is_full=False), True: make_swa_mask(S, -1, is_full=True)}

    ref_H24, ref_emb = ref_transformer_forward(H0, layers, final_norm, out_proj, S, S, luts, masks, None)

    layers_dev = [bank.upload_layer(w) for w in layers]
    lut_dev = {k: _npu_tensor(v) for k, v in luts.items()}
    mask_dev = {k: _npu_tensor(v) for k, v in masks.items()}

    def run_once():
        npu_total_us = 0.0
        attn_us = 0.0
        ffn_us = 0.0
        with bank.attn.H_a.overwrite() as buf:
            buf[:] = H0.reshape(-1)
        for li in range(24):
            is_full = li in FULL_LAYER_INDICES
            st = bank.attn.run_layer(is_full, bank.attn.H_a, bank.attn.H_b, layers_dev[li], lut_dev[is_full], mask_dev[is_full], warmup=0, iters=0)
            a_t = sum(v[1] for v in st.values())
            f_t = bank.run_ffn_layer(bank.attn.H_b, layers_dev[li], bank.attn.H_a)
            attn_us += a_t
            ffn_us += f_t
            npu_total_us += a_t + f_t
        return bank.attn.H_a.numpy().reshape(S, 512).copy(), npu_total_us, attn_us, ffn_us

    # Warmup (triggers JIT compile on any first call + stabilizes clocks)
    for _ in range(warmup):
        got_H24, tot_us, a_us, f_us = run_once()
    t0 = time.perf_counter()
    best = None
    for _ in range(iters):
        got_H24, tot_us, a_us, f_us = run_once()
        best = (tot_us, a_us, f_us)
    wall_ms = (time.perf_counter() - t0) * 1000.0 / iters

    m_h = err_metrics(got_H24, ref_H24)
    normed = _rms_norm_f32(got_H24) * final_norm.astype(np.float32)
    got_emb = _l2_normalize(normed.mean(axis=0) @ out_proj.astype(np.float32))
    emb_cos = float(np.dot(got_emb, ref_emb) / (np.linalg.norm(got_emb) * np.linalg.norm(ref_emb)))

    print(
        f"[24L Synthetic] NPU_kernel_sum={best[0]/1000:.3f} ms "
        f"(attn={best[1]/1000:.3f} ms, ffn={best[2]/1000:.3f} ms, "
        f"{best[0]/24:.1f} us/L) | wall={wall_ms:.2f} ms",
        flush=True,
    )
    print(
        f"[24L Hidden Accuracy] max_abs={m_h['max_abs']:.4f} mean_abs={m_h['mean_abs']:.5f} cos={m_h['cos']:.7f}",
        flush=True,
    )
    print(f"[Final 768-dim Embedding] cos={emb_cos:.7f} l2_norm={np.linalg.norm(got_emb):.6f}", flush=True)
    passed = m_h["cos"] >= 0.995 and emb_cos >= 0.995
    print(f"SYNTHETIC_24L_RESULT={'PASS' if passed else 'FAIL'}", flush=True)
    if not passed:
        sys.exit(1)


def _load_safetensors(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        data_off = 8 + hlen
        out = {}
        for k, meta in hdr.items():
            if not k.startswith("language_model."):
                continue
            s0, s1 = meta["data_offsets"]
            f.seek(data_off + s0)
            raw = f.read(s1 - s0)
            dt = bf16 if meta["dtype"] == "BF16" else np.float32
            out[k] = np.frombuffer(raw, dtype=dt).reshape(meta["shape"]).copy()
        return out


def run_real_weights(bank: TransformerKernelBank, model_path: Path) -> None:
    S = bank.S
    print(f"\n=== EmbeddingGemma 2 Real-Weights 24L NPU Verification ({model_path.name}, S={S}) ===", flush=True)
    t_load = time.perf_counter()
    st = _load_safetensors(model_path)
    print(f"Loaded {len(st)} language_model tensors in {time.perf_counter() - t_load:.2f}s", flush=True)

    # Sample prompt token sequence (BOS + tokens + EOS, padded to S=128)
    tokens = [2, 236743, 669, 6170, 529, 10845, 236761, 1]
    n_valid = len(tokens)
    embed_scale = float(bf16(math.sqrt(512.0)))
    embed_table = st["language_model.embed_tokens.weight"]  # [vocab, 512]
    emb_valid = (embed_table[tokens].astype(np.float32) * embed_scale).astype(bf16)
    H0 = np.zeros((S, 512), dtype=bf16)
    H0[:n_valid] = emb_valid

    layers = []
    for li in range(24):
        is_full = li in FULL_LAYER_INDICES
        p = f"language_model.layers.{li}."
        wq = st[p + "self_attn.q_proj.weight"].T  # [512, qd]
        wk = st[p + "self_attn.k_proj.weight"].T  # [512, kd]
        wv = st[p + "self_attn.v_proj.weight"].T  # [512, kd]
        w = {
            "in_norm": st[p + "input_layernorm.weight"].astype(bf16),
            "W_qkv": np.ascontiguousarray(np.concatenate([wq, wk, wv], axis=1)).astype(bf16),
            "q_norm": st[p + "self_attn.q_norm.weight"].astype(bf16),
            "k_norm": st[p + "self_attn.k_norm.weight"].astype(bf16),
            "W_o": np.ascontiguousarray(st[p + "self_attn.o_proj.weight"].T).astype(bf16),
            "post_attn": st[p + "post_attention_layernorm.weight"].astype(bf16),
            "pre_ffn": st[p + "pre_feedforward_layernorm.weight"].astype(bf16),
            "W_gate": np.ascontiguousarray(st[p + "mlp.gate_proj.weight"].T).astype(bf16),
            "W_up": np.ascontiguousarray(st[p + "mlp.up_proj.weight"].T).astype(bf16),
            "W_down": np.ascontiguousarray(st[p + "mlp.down_proj.weight"].T).astype(bf16),
            "post_ffn": st[p + "post_feedforward_layernorm.weight"].astype(bf16),
            "wpgate": st[p + "ple_block.per_layer_input_gate.weight"].astype(np.float32),
            "wpproj": st[p + "ple_block.per_layer_projection.weight"].astype(np.float32),
            "ple_post": st[p + "ple_block.post_per_layer_input_norm.weight"].astype(np.float32),
            "scalar": float(st[p + "layer_scalar"].astype(np.float32).reshape(-1)[0]),
        }
        layers.append(w)

    ple_data = {
        "ple_proj": st["language_model.ple.per_layer_model_projection.weight"].astype(np.float32),
        "ple_norm": st["language_model.ple.per_layer_projection_norm.weight"].astype(np.float32),
    }
    final_norm = st["language_model.norm.weight"].astype(bf16)
    out_proj = np.ascontiguousarray(st["language_model.embedding_projection.weight"].T).astype(bf16)

    cfg_l = make_layer_config(S, is_full=False, window=512)
    cfg_f = make_layer_config(S, is_full=True, window=-1)
    luts = {False: make_rope_lut(S, cfg_l.d, cfg_l.theta), True: make_rope_lut(S, cfg_f.d, cfg_f.theta)}
    # Mask out padded columns >= n_valid so padding never leaks into attention softmax
    mask_l = np.full((S, S), -1.0e4, dtype=np.float32)
    mask_f = np.full((S, S), -1.0e4, dtype=np.float32)
    for i in range(n_valid):
        for j in range(n_valid):
            mask_f[i, j] = 0.0
            if abs(i - j) <= 512:
                mask_l[i, j] = 0.0
    masks = {False: mask_l.astype(bf16), True: mask_f.astype(bf16)}

    ref_H24, ref_emb = ref_transformer_forward(H0, layers, final_norm, out_proj, S, n_valid, luts, masks, ple_data)

    layers_dev = [bank.upload_layer(w) for w in layers]
    lut_dev = {k: _npu_tensor(v) for k, v in luts.items()}
    mask_dev = {k: _npu_tensor(v) for k, v in masks.items()}

    with bank.attn.H_a.overwrite() as buf:
        buf[:] = H0.reshape(-1)

    emb16_f = emb_valid.astype(np.float32)
    ple_pre = 1.0 / math.sqrt(512.0)
    npu_ms = 0.0
    for li in range(24):
        is_full = li in FULL_LAYER_INDICES
        st_l = bank.attn.run_layer(is_full, bank.attn.H_a, bank.attn.H_b, layers_dev[li], lut_dev[is_full], mask_dev[is_full], warmup=0, iters=0)
        npu_ms += sum(v[1] for v in st_l.values()) / 1000.0
        npu_ms += bank.run_ffn_layer(bank.attn.H_b, layers_dev[li], bank.attn.H_a) / 1000.0

        # Exact per-layer PLE block + layer_scalar between layers
        h_cur = bank.attn.H_a.numpy().reshape(S, 512)[:n_valid].astype(np.float32)
        w = layers[li]
        ple_w = ple_data["ple_proj"][li * 512 : (li + 1) * 512]
        pl = _rms_norm_f32((emb16_f @ ple_w.T) * ple_pre) * ple_data["ple_norm"]
        act_p = _gelu_tanh_f32(h_cur @ w["wpgate"].T) * pl
        h_next = (h_cur + _rms_norm_f32(act_p @ w["wpproj"].T) * w["ple_post"]) * w["scalar"]
        with bank.attn.H_a.overwrite() as buf:
            h_pad = buf.reshape(S, 512)
            h_pad.fill(bf16(0.0))
            h_pad[:n_valid] = h_next.astype(bf16)

    got_H24 = bank.attn.H_a.numpy().reshape(S, 512)[:n_valid].copy()
    m_h = err_metrics(got_H24, ref_H24[:n_valid])
    normed = _rms_norm_f32(got_H24) * final_norm.astype(np.float32)
    got_emb = _l2_normalize(normed.mean(axis=0) @ out_proj.astype(np.float32))
    emb_cos = float(np.dot(got_emb, ref_emb) / (np.linalg.norm(got_emb) * np.linalg.norm(ref_emb)))

    print(
        f"[Real 24L] NPU_kernel_time={npu_ms:.3f} ms | "
        f"H24 max_abs={m_h['max_abs']:.4f} cos={m_h['cos']:.7f} | "
        f"768d emb cos={emb_cos:.7f}",
        flush=True,
    )
    passed = emb_cos >= 0.995
    print(f"REAL_WEIGHTS_24L_RESULT={'PASS' if passed else 'FAIL'}", flush=True)
    if not passed:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--real-weights", type=Path, default=None)
    ap.add_argument("--export-bundle", type=Path, default=None)
    args = ap.parse_args()

    bank = TransformerKernelBank(S=args.seq_len)
    if args.export_bundle is not None:
        bank.export_bundle(args.export_bundle)
    if args.synthetic or (args.real_weights is None and args.export_bundle is None):
        run_synthetic(bank)
    if args.real_weights is not None:
        run_real_weights(bank, args.real_weights)


if __name__ == "__main__":
    main()
