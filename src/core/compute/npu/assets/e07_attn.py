#!/usr/bin/env python3
"""EmbeddingGemma 2 Attention (24 layers + 4 IRON sub-kernels) on Ryzen AI NPU2.

Verified ground truth (config.json & gufo-final2/.../embeddinggemma2/hip/session.hip):
  - hidden=512, H_q=4, rms_norm_eps=1e-6, sliding_window=512, bidirectional, scale=1.0
  - 20 sliding (local) layers (li not in {5,11,17,23}): H_kv=2, d=256, rope_theta=1e4
  -  4 full (global)   layers (li     in {5,11,17,23}): H_kv=1, d=512, rope_theta=1e6
  - 4-stage / 8-XCLBIN pipeline per layer:
      1. qkv_pre_fused (20c local / 28c full): 4-core Pre-RMSNorm + 2D 4xC GEMM,
         A pinned via e06_opt 3D->4D bridge, W broadcast vertically, 2-row join ->
         head-major [Q_raw(H_q,S,d), K_raw(H_kv,S,d), V_raw(H_kv,S,d)].
      2. qkv_norm_rope (4c local / 6c full): weighted Q/K RMSNorm + Two-Halves RoPE
         (hardware-broadcast LUT) + unweighted V RMSNorm.
      3. attn_gqa_core (12c): fused QK^T -> Mask+Softmax -> PV, GQA KV multicast,
         broadcast mask, zero S/P DDR spill, direct token-major Ctx(S, H_q*d).
      4. o_post_fused (20c): 2D 4x4 K-split O_proj (pure-bf16 Ctx split + shared-L1
         West->East reduce) -> MemTile untile -> 4-core Post-RMSNorm + residual add.
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
from aie.iron import In, Out, kernels as K
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

FULL_LAYER_INDICES = frozenset({5, 11, 17, 23})
LAYER_TYPES_24 = ["full" if i in FULL_LAYER_INDICES else "sliding" for i in range(24)]
EPS_RMS = 1e-6
KT_MM = 64
NT_MM = 64


@dataclass(frozen=True)
class AttnLayerConfig:
    S: int = 128
    D: int = 512
    H_q: int = 4
    H_kv: int = 2
    d: int = 256
    window: int = 512
    theta: float = 10000.0
    M_q: int | None = None
    is_full: bool = False

    def __post_init__(self):
        if self.M_q is None:
            mq = 32 if (self.S <= 128 and self.d == 256) else (16 if self.S <= 128 else 8)
            object.__setattr__(self, "M_q", mq)

    @property
    def qkv_cols(self) -> int:
        return 4 if self.H_kv == 2 else 6

    @property
    def qkv_out_dim(self) -> int:
        return (self.H_q + 2 * self.H_kv) * self.d


def make_layer_config(S: int, is_full: bool, window: int = 512) -> AttnLayerConfig:
    if is_full:
        return AttnLayerConfig(S=S, H_q=4, H_kv=1, d=512, window=-1, theta=1_000_000.0, is_full=True)
    return AttnLayerConfig(S=S, H_q=4, H_kv=2, d=256, window=window, theta=10_000.0, is_full=False)


def _i32(val: int):
    return constant(val, type=ir.IntegerType.get_signless(32))


def _f32(val: float):
    return constant(val, type=ir.F32Type.get())


def _npu_tensor(arr: np.ndarray):
    return iron.tensor(np.ascontiguousarray(arr, dtype=bf16).reshape(-1), dtype=bf16, device="npu")


def _npu_zeros(numel: int):
    return iron.zeros(numel, dtype=bf16, device="npu")


# ==============================================================================
# NumPy FP32 References & Error Metrics
# ==============================================================================
def make_rope_lut(S: int, d: int, theta: float) -> np.ndarray:
    half = d // 2
    pos = np.arange(S, dtype=np.float64)[:, None]
    freq_idx = np.arange(half, dtype=np.float64)[None, :]
    inv_freq = np.power(float(theta), -2.0 * freq_idx / float(d))
    ang = pos * inv_freq
    lut = np.empty((S, d), dtype=np.float32)
    lut[:, 0::2] = np.cos(ang).astype(np.float32)
    lut[:, 1::2] = np.sin(ang).astype(np.float32)
    return lut.astype(bf16)


def make_swa_mask(S: int, window: int, is_full: bool = False) -> np.ndarray:
    if is_full or window < 0:
        return np.zeros((S, S), dtype=bf16)
    pos = np.arange(S)
    allowed = np.abs(pos[:, None] - pos[None, :]) <= window
    return np.where(allowed, 0.0, -8192.0).astype(bf16)


def _rms_norm_f32(x: np.ndarray, eps: float = EPS_RMS) -> np.ndarray:
    xf = x.astype(np.float32)
    inv = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * inv).astype(bf16).astype(np.float32)


def ref_rope_two_halves(x: np.ndarray, lut: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32)
    lf = lut.astype(np.float32)
    c = lf[..., 0::2]
    s = lf[..., 1::2]
    half = xf.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    return np.concatenate([y1, y2], axis=-1).astype(bf16)


def ref_qkt(Q: np.ndarray, K_mat: np.ndarray) -> np.ndarray:
    H_q, S, _ = Q.shape
    H_kv = K_mat.shape[0]
    group = H_q // H_kv
    out = np.empty((H_q, S, S), dtype=bf16)
    qf, kf = Q.astype(np.float32), K_mat.astype(np.float32)
    for h in range(H_q):
        out[h] = (qf[h] @ kf[h // group].T).astype(bf16)
    return out


def ref_softmax_swa(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sf = scores.astype(np.float32) + mask.astype(np.float32)
    m = np.max(sf, axis=-1, keepdims=True)
    e = np.exp(sf - m)
    p = e / np.sum(e, axis=-1, keepdims=True)
    return p.astype(bf16)


def ref_pv(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    H_q, S, _ = P.shape
    H_kv, _, d = V.shape
    group = H_q // H_kv
    pf, vf = P.astype(np.float32), V.astype(np.float32)
    ctx = np.empty((S, H_q, d), dtype=bf16)
    for h in range(H_q):
        ctx[:, h, :] = (pf[h] @ vf[h // group]).astype(bf16)
    return ctx.reshape(S, H_q * d)


def ref_attn_layer(
    H: np.ndarray,
    in_norm: np.ndarray,
    W_qkv: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    lut: np.ndarray,
    mask: np.ndarray,
    W_o: np.ndarray,
    post_attn: np.ndarray,
    cfg: AttnLayerConfig,
) -> np.ndarray:
    S, H_q, H_kv, d = cfg.S, cfg.H_q, cfg.H_kv, cfg.d
    qd = H_q * d
    kd = H_kv * d
    x_n = (_rms_norm_f32(H) * in_norm.astype(np.float32)).astype(bf16)
    qkv = (x_n.astype(np.float32) @ W_qkv.astype(np.float32)).astype(bf16)

    Q_raw = qkv[:, :qd].reshape(S, H_q, d).transpose(1, 0, 2)
    K_raw = qkv[:, qd : qd + kd].reshape(S, H_kv, d).transpose(1, 0, 2)
    V_raw = qkv[:, qd + kd :].reshape(S, H_kv, d).transpose(1, 0, 2)

    qn = q_norm.astype(np.float32)
    kn = k_norm.astype(np.float32)
    Q = ref_rope_two_halves((_rms_norm_f32(Q_raw) * qn).astype(bf16), lut)
    K_m = ref_rope_two_halves((_rms_norm_f32(K_raw) * kn).astype(bf16), lut)
    V_m = _rms_norm_f32(V_raw).astype(bf16)

    S_sc = ref_qkt(Q, K_m)
    P = ref_softmax_swa(S_sc, mask)
    Ctx = ref_pv(P, V_m)
    Y = (Ctx.astype(np.float32) @ W_o.astype(np.float32)).astype(bf16)
    H_out = (_rms_norm_f32(Y) * post_attn.astype(np.float32) + H.astype(np.float32)).astype(bf16)
    return H_out


def _max_ulp_ge1(got_bf16: np.ndarray, ref_bf16: np.ndarray, ref_f32: np.ndarray) -> int:
    mask = np.abs(ref_f32) >= 1.0
    if not np.any(mask):
        return 0
    gb = got_bf16[mask].view(np.uint16).astype(np.int32)
    rb = ref_bf16[mask].view(np.uint16).astype(np.int32)
    go = np.where((gb & 0x8000) != 0, 0x8000 - (gb & 0x7FFF), 0x8000 + (gb & 0x7FFF))
    ro = np.where((rb & 0x8000) != 0, 0x8000 - (rb & 0x7FFF), 0x8000 + (rb & 0x7FFF))
    return int(np.abs(go - ro).max())


def err_metrics(got_bf16: np.ndarray, ref_bf16: np.ndarray) -> dict[str, float | int]:
    g = got_bf16.astype(np.float32).ravel()
    r = ref_bf16.astype(np.float32).ravel()
    diff = np.abs(g - r)
    denom = float(np.linalg.norm(g) * np.linalg.norm(r))
    cos = float(np.dot(g, r) / denom) if denom > 0.0 else 0.0
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.percentile(diff, 99)),
        "max_ulp": _max_ulp_ge1(got_bf16, ref_bf16, ref_bf16.astype(np.float32)),
        "cos": cos,
    }


def bench_kernel(fn, args: list, warmup: int = 3, iters: int = 20):
    t0 = time.perf_counter()
    hdl, _ = fn(*args)
    compile_s = time.perf_counter() - t0
    for _ in range(warmup):
        fn(*args)
    call_us, hw_us = [], []
    for _ in range(iters):
        ta = time.perf_counter()
        _, k_res = fn(*args)
        call_us.append((time.perf_counter() - ta) * 1e6)
        hw_us.append(k_res.npu_time / 1e3)
    return hdl, compile_s, float(np.median(call_us)), float(np.median(hw_us))


def bench_chain_same(hdl, args: list, count: int, warmup: int = 2, iters: int = 10) -> float:
    runs = [(hdl, args)] * count
    for _ in range(warmup):
        DefaultNPURuntime.run_chain(runs)
    ts = [DefaultNPURuntime.run_chain(runs).npu_time / 1e3 / count for _ in range(iters)]
    return float(np.median(ts))


# ==============================================================================
# Part A: 4 Standalone Decoupled Sub-Kernels (RoPE, QK^T, Softmax+SWA, PV)
# ==============================================================================
def build_rope_subkernel(S: int, d: int, n_heads: int):
    rope_k = K.rope(tile_size=d, two_halves=True)
    ce_k = K.conv_even()
    row_ty = np.ndarray[(d,), np.dtype[bf16]]
    X_ty = np.ndarray[(n_heads * S * d,), np.dtype[bf16]]
    LUT_ty = np.ndarray[(S * d,), np.dtype[bf16]]

    @iron.jit
    def rope_sub(X_in: In, LUT_in: In, Y_out: Out):
        of_x = [ObjectFifo(row_ty, depth=2, name=f"rp_x_{h}") for h in range(n_heads)]
        of_y = [ObjectFifo(row_ty, depth=2, name=f"rp_y_{h}") for h in range(n_heads)]
        of_lut_in = ObjectFifo(row_ty, depth=2, name="rp_lut_in")
        of_lut_mem = of_lut_in.cons().forward(tile=Tile(n_heads, 1), depth=2, name="rp_lut_m")

        def body(ox, olut, oy, rp_fn, ce_fn):
            ce_fn()
            c_d = _i32(d)
            for _ in range_(S):
                x, lut, y = ox.acquire(1), olut.acquire(1), oy.acquire(1)
                rp_fn(x, lut, y, c_d)
                ox.release(1)
                olut.release(1)
                oy.release(1)

        workers = [
            Worker(body, fn_args=[of_x[h].cons(), of_lut_mem.cons(), of_y[h].prod(), rope_k, ce_k], tile=Tile(h, 2))
            for h in range(n_heads)
        ]

        def seq(X_t, L_t, Y_t, lh, *hdls):
            xhs, yhs = hdls[:n_heads], hdls[n_heads:]
            tg = TaskGroup()
            lh.fill(L_t, group=tg)
            chunk = S * d
            for h in range(n_heads):
                tap = TensorAccessPattern([n_heads * S * d], h * chunk, [chunk], [1])
                xhs[h].fill(X_t, tap=tap, group=tg)
                yhs[h].drain(Y_t, tap=tap, wait=True, group=tg)
            tg.finish()

        x_hdls = [of_x[h].prod(tile=Tile(h, 0)) for h in range(n_heads)]
        y_hdls = [of_y[h].cons(tile=Tile(h, 0)) for h in range(n_heads)]
        rt = Runtime(seq, [X_ty, LUT_ty, X_ty, of_lut_in.prod(tile=Tile(n_heads, 0)), *x_hdls, *y_hdls])
        return Program(DEV, rt, workers=workers).resolve_program()

    return rope_sub


def build_qkt_subkernel(S: int, d: int, H_q: int, H_kv: int, M_q: int):
    KT, NT = KT_MM, NT_MM
    Q_strips, S_tiles, D_tiles = S // M_q, S // KT, d // KT
    group, bands = H_q // H_kv, M_q // 4

    mm_qkt = K.mm(M_q, KT, NT, bf16, bf16, b_col_maj=True)
    lay_q, lay_k, _ = mm_qkt.contract.layouts
    zero_s, ce_k = K.zero((M_q * NT,), bf16), K.conv_even()

    q_ty = np.ndarray[(M_q * KT,), np.dtype[bf16]]
    k_ty = np.ndarray[(NT * KT,), np.dtype[bf16]]
    s_ty = np.ndarray[(M_q * NT,), np.dtype[bf16]]
    tap_s_mem = TensorAccessPattern([M_q * NT], 0, [bands, 4, 8, 8], [256, 8, 32, 1])

    Q_ty = np.ndarray[(H_q * S * d,), np.dtype[bf16]]
    K_ty = np.ndarray[(H_kv * S * d,), np.dtype[bf16]]
    S_ty = np.ndarray[(H_q * S * S,), np.dtype[bf16]]

    @iron.jit
    def qkt_sub(Q_in: In, K_in: In, Scores_out: Out):
        of_q_in = [ObjectFifo(q_ty, depth=2, name=f"qkt_qi_{h}") for h in range(H_q)]
        of_q_mem = [
            of_q_in[h].cons().forward(tile=Tile(h, 1), depth=D_tiles, to_stream=lay_q.stream, name=f"qkt_qm_{h}")
            for h in range(H_q)
        ]
        of_k_in = [ObjectFifo(k_ty, depth=2, name=f"qkt_ki_{hk}") for hk in range(H_kv)]
        of_k_mem = [
            of_k_in[hk].cons().forward(tile=Tile(4 + hk, 1), depth=2, to_stream=lay_k.stream, name=f"qkt_km_{hk}")
            for hk in range(H_kv)
        ]
        of_s_in = [ObjectFifo(s_ty, depth=2, name=f"qkt_si_{h}") for h in range(H_q)]
        of_s_mem = [
            of_s_in[h].cons().forward(tile=Tile(h, 1), depth=2, to_stream=tap_s_mem, name=f"qkt_sm_{h}")
            for h in range(H_q)
        ]

        def body(oq, ok, os_out, mm_fn, z_fn, ce_fn):
            ce_fn()
            for _ in range_(Q_strips):
                q_tiles = oq.acquire(D_tiles)
                for _ in range_(S_tiles):
                    s_t = os_out.acquire(1)
                    z_fn(s_t)
                    for dt in range(D_tiles):
                        k_t = ok.acquire(1)
                        mm_fn(q_tiles[dt], k_t, s_t)
                        ok.release(1)
                    os_out.release(1)
                oq.release(D_tiles)

        workers = [
            Worker(
                body,
                fn_args=[of_q_mem[h].cons(), of_k_mem[h // group].cons(), of_s_in[h].prod(), mm_qkt, zero_s, ce_k],
                tile=Tile(h, 2),
            )
            for h in range(H_q)
        ]

        q_hdls = [of_q_in[h].prod(tile=Tile(h, 0)) for h in range(H_q)]
        s_hdls = [of_s_mem[h].cons(tile=Tile(h, 0)) for h in range(H_q)]
        k_hdls = [of_k_in[hk].prod(tile=Tile(4 + hk, 0)) for hk in range(H_kv)]

        def seq(Q_t, K_t, S_t, *hdls):
            qhs, shs, khs = hdls[:H_q], hdls[H_q : 2 * H_q], hdls[2 * H_q :]
            B = min(4, Q_strips)
            for g0 in range(0, Q_strips, B):
                cnt = min(B, Q_strips - g0)
                tg = TaskGroup()
                for h in range(H_q):
                    tap_q = TensorAccessPattern(
                        [H_q * S * d], h * S * d + g0 * M_q * d, [cnt, D_tiles, M_q, KT], [M_q * d, KT, d, 1]
                    )
                    qhs[h].fill(Q_t, tap=tap_q, group=tg)
                    tap_s = TensorAccessPattern(
                        [H_q * S * S], h * S * S + g0 * M_q * S, [cnt, S_tiles, M_q, NT], [M_q * S, NT, S, 1]
                    )
                    shs[h].drain(S_t, tap=tap_s, wait=True, group=tg)
                for hk in range(H_kv):
                    tap_k = TensorAccessPattern(
                        [H_kv * S * d], hk * S * d, [S_tiles, D_tiles, NT, KT], [NT * d, KT, d, 1]
                    )
                    for _ in range(cnt):
                        khs[hk].fill(K_t, tap=tap_k, group=tg)
                tg.finish()

        rt = Runtime(seq, [Q_ty, K_ty, S_ty, *q_hdls, *s_hdls, *k_hdls])
        return Program(DEV, rt, workers=workers).resolve_program()

    return qkt_sub


def build_softmax_swa_subkernel(S: int, H_q: int):
    add_k, sm_k, ce_k = K.add_sized(S), K.softmax(S), K.conv_even()
    row_ty = np.ndarray[(S,), np.dtype[bf16]]
    Sc_ty = np.ndarray[(H_q * S * S,), np.dtype[bf16]]
    M_ty = np.ndarray[(S * S,), np.dtype[bf16]]

    @iron.jit
    def softmax_swa_sub(Scores_in: In, Mask_in: In, Probs_out: Out):
        of_s = [ObjectFifo(row_ty, depth=2, name=f"sm_s_{h}") for h in range(H_q)]
        of_p = [ObjectFifo(row_ty, depth=2, name=f"sm_p_{h}") for h in range(H_q)]
        of_m_in = ObjectFifo(row_ty, depth=2, name="sm_m_in")
        of_m_mem = of_m_in.cons().forward(tile=Tile(H_q, 1), depth=2, name="sm_m_mem")
        tmps = [Buffer(row_ty, name=f"sm_sub_tmp_{h}") for h in range(H_q)]

        def body(os_in, om, op_out, tmp_buf, add_fn, sm_fn, ce_fn):
            ce_fn()
            c_s = _i32(S)
            for _ in range_(S):
                sr, mr, pr = os_in.acquire(1), om.acquire(1), op_out.acquire(1)
                add_fn(sr, mr, tmp_buf, c_s)
                sm_fn(tmp_buf, pr, c_s)
                os_in.release(1)
                om.release(1)
                op_out.release(1)

        workers = [
            Worker(
                body,
                fn_args=[of_s[h].cons(), of_m_mem.cons(), of_p[h].prod(), tmps[h], add_k, sm_k, ce_k],
                tile=Tile(h, 2),
            )
            for h in range(H_q)
        ]

        shs = [of_s[h].prod(tile=Tile(h, 0)) for h in range(H_q)]
        phs = [of_p[h].cons(tile=Tile(h, 0)) for h in range(H_q)]
        mh = of_m_in.prod(tile=Tile(H_q, 0))

        def seq(S_t, M_t, P_t, m_h, *hdls):
            s_hs, p_hs = hdls[:H_q], hdls[H_q:]
            tg = TaskGroup()
            m_h.fill(M_t, group=tg)
            chunk = S * S
            for h in range(H_q):
                tap = TensorAccessPattern([H_q * chunk], h * chunk, [chunk], [1])
                s_hs[h].fill(S_t, tap=tap, group=tg)
                p_hs[h].drain(P_t, tap=tap, wait=True, group=tg)
            tg.finish()

        rt = Runtime(seq, [Sc_ty, M_ty, Sc_ty, mh, *shs, *phs])
        return Program(DEV, rt, workers=workers).resolve_program()

    return softmax_swa_sub


def build_pv_subkernel(S: int, d: int, H_q: int, H_kv: int, M_q: int):
    KT, NT = KT_MM, NT_MM
    Q_strips, S_tiles, D_tiles = S // M_q, S // KT, d // NT
    group, bands = H_q // H_kv, M_q // 4

    mm_pv = K.mm(M_q, KT, NT, bf16, bf16, b_col_maj=False)
    lay_p, lay_v, _ = mm_pv.contract.layouts
    zero_c, ce_k = K.zero((M_q * NT,), bf16), K.conv_even()

    p_ty = np.ndarray[(M_q * KT,), np.dtype[bf16]]
    v_ty = np.ndarray[(KT * NT,), np.dtype[bf16]]
    c_ty = np.ndarray[(M_q * NT,), np.dtype[bf16]]
    tap_c_mem = TensorAccessPattern([M_q * NT], 0, [bands, 4, 8, 8], [256, 8, 32, 1])

    P_ty = np.ndarray[(H_q * S * S,), np.dtype[bf16]]
    V_ty = np.ndarray[(H_kv * S * d,), np.dtype[bf16]]
    C_ty = np.ndarray[(S * H_q * d,), np.dtype[bf16]]

    @iron.jit
    def pv_sub(Probs_in: In, V_in: In, Ctx_out: Out):
        of_p_in = [ObjectFifo(p_ty, depth=2, name=f"pv_pi_{h}") for h in range(H_q)]
        of_p_mem = [
            of_p_in[h].cons().forward(tile=Tile(h, 1), depth=2, to_stream=lay_p.stream, name=f"pv_pm_{h}")
            for h in range(H_q)
        ]
        of_v_in = [ObjectFifo(v_ty, depth=2, name=f"pv_vi_{hk}") for hk in range(H_kv)]
        of_v_mem = [
            of_v_in[hk].cons().forward(tile=Tile(4 + hk, 1), depth=2, to_stream=lay_v.stream, name=f"pv_vm_{hk}")
            for hk in range(H_kv)
        ]
        of_c_in = [ObjectFifo(c_ty, depth=D_tiles, name=f"pv_ci_{h}") for h in range(H_q)]
        of_c_mem = [
            of_c_in[h].cons().forward(tile=Tile(h, 1), depth=2, to_stream=tap_c_mem, name=f"pv_cm_{h}")
            for h in range(H_q)
        ]

        def body(op, ov, oc, mm_fn, z_fn, ce_fn):
            ce_fn()
            for _ in range_(Q_strips):
                c_tiles = oc.acquire(D_tiles)
                for dt in range(D_tiles):
                    z_fn(c_tiles[dt])
                for _ in range_(S_tiles):
                    p_t = op.acquire(1)
                    for dt in range(D_tiles):
                        v_t = ov.acquire(1)
                        mm_fn(p_t, v_t, c_tiles[dt])
                        ov.release(1)
                    op.release(1)
                oc.release(D_tiles)

        workers = [
            Worker(
                body,
                fn_args=[of_p_mem[h].cons(), of_v_mem[h // group].cons(), of_c_in[h].prod(), mm_pv, zero_c, ce_k],
                tile=Tile(h, 2),
            )
            for h in range(H_q)
        ]

        p_hdls = [of_p_in[h].prod(tile=Tile(h, 0)) for h in range(H_q)]
        c_hdls = [of_c_mem[h].cons(tile=Tile(h, 0)) for h in range(H_q)]
        v_hdls = [of_v_in[hk].prod(tile=Tile(4 + hk, 0)) for hk in range(H_kv)]

        def seq(P_t, V_t, C_t, *hdls):
            phs, chs, vhs = hdls[:H_q], hdls[H_q : 2 * H_q], hdls[2 * H_q :]
            B = min(4, Q_strips)
            for g0 in range(0, Q_strips, B):
                cnt = min(B, Q_strips - g0)
                tg = TaskGroup()
                for h in range(H_q):
                    tap_p = TensorAccessPattern(
                        [H_q * S * S], h * S * S + g0 * M_q * S, [cnt, S_tiles, M_q, KT], [M_q * S, KT, S, 1]
                    )
                    phs[h].fill(P_t, tap=tap_p, group=tg)
                    tap_c = TensorAccessPattern(
                        [S * H_q * d], h * d + g0 * M_q * H_q * d, [cnt, D_tiles, M_q, NT], [M_q * H_q * d, NT, H_q * d, 1]
                    )
                    chs[h].drain(C_t, tap=tap_c, wait=True, group=tg)
                for hk in range(H_kv):
                    tap_v = TensorAccessPattern(
                        [H_kv * S * d], hk * S * d, [S_tiles, D_tiles, KT, NT], [KT * d, NT, d, 1]
                    )
                    for _ in range(cnt):
                        vhs[hk].fill(V_t, tap=tap_v, group=tg)
                tg.finish()

        rt = Runtime(seq, [P_ty, V_ty, C_ty, *p_hdls, *c_hdls, *v_hdls])
        return Program(DEV, rt, workers=workers).resolve_program()

    return pv_sub


# ==============================================================================
# Part B: 4-Stage Fused Full-Attention Pipeline Builders
# ==============================================================================
def build_qkv_pre_fused(S: int, N_dim: int, C: int, head_d: int):
    """Fused 4-core Pre-RMSNorm + 2D 4xC QKV GEMM (20c local / 28c full)."""
    R, D, MT, KT, NT = 4, 512, 32, KT_MM, NT_MM
    M_pass = R * MT
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
    a_cols = [C + r for r in range(R)] if C <= 4 else [6, 6, 7, 7]

    H_ty = np.ndarray[(S * D,), np.dtype[bf16]]
    NW_ty = np.ndarray[(D,), np.dtype[bf16]]
    W_ty = np.ndarray[(D * N_dim,), np.dtype[bf16]]
    Out_ty = np.ndarray[(S * N_dim,), np.dtype[bf16]]

    @iron.jit
    def qkv_pre_fused(H_in: In, NW_in: In, W_in: In, Out_out: Out):
        of_h = [ObjectFifo(row_ty, depth=2, name=f"qf_h_{r}") for r in range(R)]
        of_n_raw = [ObjectFifo(mat_i8, depth=1, to_stream=tap_n1, name=f"qf_nr_{r}") for r in range(R)]
        of_a_bc = [
            of_n_raw[r].cons(depth=1).forward(tile=Tile(a_cols[r], 1), obj_type=mat_i8, depth=1, to_stream=tap_a2, name=f"qf_ab_{r}")
            for r in range(R)
        ]
        pn_tmp = [Buffer(row_ty, name=f"qf_t_{r}") for r in range(R)]
        pn_wb = [Buffer(row_ty, name=f"qf_wb_{r}") for r in range(R)]

        of_b_in = [ObjectFifo(b_ty, depth=4, name=f"qf_bi_{c}") for c in range(C)]
        of_b_mem = [of_b_in[c].cons(depth=2).forward(tile=Tile(c, 1), depth=2, to_stream=lay_b.stream, name=f"qf_bm_{c}") for c in range(C)]

        of_c_out = [[ObjectFifo(c2_ty, depth=2, to_stream=tap_c1, name=f"qf_co_{c}_{pr}") for pr in range(2)] for c in range(C)]
        of_c_in = [[None] * R for _ in range(C)]
        for c in range(C):
            for pr in range(2):
                subs = of_c_out[c][pr].prod().join(
                    [0, MT * NT], tile=Tile(c, 1), depths=[2, 2], obj_types=[c_ty, c_ty],
                    names=[f"qf_ci_{c}_{pr*2}", f"qf_ci_{c}_{pr*2+1}"],
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
            h_per_col = N_col // head_d
            dt_h = head_d // NT
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
                        for lh in range(h_per_col):
                            gh = c * h_per_col + lh
                            tap_c = TensorAccessPattern(
                                [S * N_dim], gh * S * head_d + row_off * head_d, [dt_h, 2 * MT, NT], [NT, head_d, 1]
                            )
                            ch.drain(Out_t, tap=tap_c, wait=True, group=tg)
                tg.finish()

        rt = Runtime(seq, [H_ty, NW_ty, W_ty, Out_ty, *h_hdls, *b_hdls, *c_hdls])
        return Program(DEV, rt, workers=workers).resolve_program()

    return qkv_pre_fused


def build_o_post_fused(S: int, K_dim: int):
    """Fused 20-core 2D 4x4 K-split O_proj + shared-L1 reduce + 4-core Post-RMSNorm + residual add."""
    R, C, D, MT, KT, NT = 4, 4, 512, 16, KT_MM, NT_MM
    M_pass = R * MT
    n_passes = S // M_pass
    K_col = K_dim // C
    K_tiles_col = K_col // KT
    N_tiles = D // NT
    bands = MT // 4

    mm_k = K.mm(MT, KT, NT, bf16, bf16, b_col_maj=False)
    _, lay_b, _ = mm_k.contract.layouts
    z_k, ce_k = K.zero((MT * NT,), bf16), K.conv_even()
    add_z = K.add_sized(MT * NT)
    rms_k, mul_k, add_d = K.rms_norm_eps(cols=D), K.mul_sized(D), K.add_sized(D)

    a_tile_bf = np.ndarray[(MT * KT,), np.dtype[bf16]]
    a_row_bf = np.ndarray[(MT * K_col,), np.dtype[bf16]]
    a_pack_bf = np.ndarray[(R * MT * K_col,), np.dtype[bf16]]
    b_ty = np.ndarray[(KT * NT,), np.dtype[bf16]]
    z_ty = np.ndarray[(MT * NT,), np.dtype[bf16]]
    y_i8 = np.ndarray[(MT * D * 2,), np.dtype[np.int8]]
    band_i8 = np.ndarray[(4 * D * 2,), np.dtype[np.int8]]
    row_ty = np.ndarray[(D,), np.dtype[bf16]]

    tap_a_4d = TensorAccessPattern([MT * K_col], 0, [K_tiles_col * bands, 8, 4, 8], [4 * KT, 8, KT, 1])
    tap_y_mem = TensorAccessPattern([MT * D * 2], 0, [bands, N_tiles, 512], [512, MT * 128, 1])
    tap_y_pna = TensorAccessPattern([4 * D * 2], 0, [N_tiles * 8, 4, 16], [16, D * 2, 1])

    Ctx_ty = np.ndarray[(S * K_dim,), np.dtype[bf16]]
    Wo_ty = np.ndarray[(K_dim * D,), np.dtype[bf16]]
    H_ty = np.ndarray[(S * D,), np.dtype[bf16]]
    W_ty = np.ndarray[(D,), np.dtype[bf16]]

    @iron.jit
    def o_post_fused(Ctx_in: In, Wo_in: In, H_in: In, NW_in: In, Out_out: Out):
        of_a_pack = [ObjectFifo(a_pack_bf, depth=1, name=f"op_ap_{c}") for c in range(C)]
        of_a_row = []
        for c in range(C):
            subs = of_a_pack[c].cons(depth=1).split(
                [r * MT * K_col for r in range(R)],
                tile=Tile(c, 1), depths=[1] * R, obj_types=[a_row_bf] * R,
                to_stream=[tap_a_4d] * R, names=[f"op_ar_{c}_{r}" for r in range(R)],
            )
            for s in subs:
                s._consumer_obj_type = a_tile_bf
            of_a_row.append(subs)

        of_b_in = [ObjectFifo(b_ty, depth=4, name=f"op_bi_{c}") for c in range(C)]
        of_b_mem = [of_b_in[c].cons(depth=2).forward(tile=Tile(c, 1), depth=2, to_stream=lay_b.stream, name=f"op_bm_{c}") for c in range(C)]
        of_z = [[ObjectFifo(z_ty, depth=2, name=f"op_z_{r}_{c}") for c in range(C - 1)] for r in range(R)]

        of_y_in = [ObjectFifo(y_i8, depth=1, name=f"op_yi_{r}") for r in range(R)]
        of_y_mem = []
        for r in range(R):
            f = of_y_in[r].cons(depth=1).forward(tile=Tile(4 + r, 1), depth=1, to_stream=tap_y_mem, name=f"op_ym_{r}")
            f._consumer_obj_type = band_i8
            f._from_stream_per_cons = tap_y_pna
            of_y_mem.append(f)

        of_r_in = [ObjectFifo(row_ty, depth=2, name=f"op_ri_{r}") for r in range(R)]
        of_h_out = [ObjectFifo(row_ty, depth=2, name=f"op_ho_{r}") for r in range(R)]
        acc_bufs = {(c, r): Buffer(z_ty, name=f"op_acc_{c}_{r}") for c in range(1, C) for r in range(R)}
        pna_t1 = [Buffer(row_ty, name=f"op_t1_{r}") for r in range(R)]
        pna_t2 = [Buffer(row_ty, name=f"op_t2_{r}") for r in range(R)]
        pna_wb = [Buffer(row_ty, name=f"op_wb_{r}") for r in range(R)]

        def body_first(oa, ob, oz_out, mm_fn, z_fn, ce_fn):
            ce_fn()
            for _ in range_(n_passes):
                av = oa.acquire(K_tiles_col)
                for _ in range_(N_tiles):
                    zo = oz_out.acquire(1)
                    z_fn(zo)
                    for kt in range(K_tiles_col):
                        w = ob.acquire(1)
                        mm_fn(av[kt], w, zo)
                        ob.release(1)
                    oz_out.release(1)
                oa.release(K_tiles_col)

        def body_mid(oa, ob, oz_in, oz_out, acc, mm_fn, z_fn, add_fn, ce_fn):
            ce_fn()
            c_el = _i32(MT * NT)
            for _ in range_(n_passes):
                av = oa.acquire(K_tiles_col)
                for _ in range_(N_tiles):
                    z_fn(acc)
                    for kt in range(K_tiles_col):
                        w = ob.acquire(1)
                        mm_fn(av[kt], w, acc)
                        ob.release(1)
                    zi, zo = oz_in.acquire(1), oz_out.acquire(1)
                    add_fn(acc, zi, zo, c_el)
                    oz_in.release(1)
                    oz_out.release(1)
                oa.release(K_tiles_col)

        def body_last(oa, ob, oz_in, oy_out, acc, mm_fn, z_fn, add_fn, ce_fn):
            ce_fn()
            y_vty = ir.MemRefType.get([MT * NT], ir.BF16Type.get())
            c_el = _i32(MT * NT)
            for _ in range_(n_passes):
                av = oa.acquire(K_tiles_col)
                yb = oy_out.acquire(1)
                for nt in range(N_tiles):
                    z_fn(acc)
                    for kt in range(K_tiles_col):
                        w = ob.acquire(1)
                        mm_fn(av[kt], w, acc)
                        ob.release(1)
                    zi = oz_in.acquire(1)
                    yv = memref.view(y_vty, yb, constant(nt * MT * NT * 2, index=True), [])
                    add_fn(acc, zi, yv, c_el)
                    oz_in.release(1)
                oy_out.release(1)
                oa.release(K_tiles_col)

        def pna_body(oy_in, or_in, oh_out, b1, b2, wbuf, rms_fn, mul_fn, add_fn, ce_fn):
            ce_fn()
            c_d, c_eps = _i32(D), _f32(EPS_RMS)
            r_vty = ir.MemRefType.get([D], ir.BF16Type.get())
            w_in = or_in.acquire(1)
            for i in range_(D):
                wbuf[i] = w_in[i]
            or_in.release(1)
            for _ in range_(n_passes):
                for _ in range_(bands):
                    yb = oy_in.acquire(1)
                    for rr in range(4):
                        yv = memref.view(r_vty, yb, constant(rr * D * 2, index=True), [])
                        hr, ho = or_in.acquire(1), oh_out.acquire(1)
                        rms_fn(yv, b1, c_d, c_eps)
                        mul_fn(b1, wbuf, b2, c_d)
                        add_fn(b2, hr, ho, c_d)
                        or_in.release(1)
                        oh_out.release(1)
                    oy_in.release(1)

        workers = []
        for r in range(R):
            workers.append(Worker(body_first, fn_args=[of_a_row[0][r].cons(depth=K_tiles_col), of_b_mem[0].cons(), of_z[r][0].prod(), mm_k, z_k, ce_k], tile=Tile(0, 2 + r)))
            for c in range(1, C - 1):
                workers.append(Worker(body_mid, fn_args=[of_a_row[c][r].cons(depth=K_tiles_col), of_b_mem[c].cons(), of_z[r][c - 1].cons(), of_z[r][c].prod(), acc_bufs[(c, r)], mm_k, z_k, add_z, ce_k], tile=Tile(c, 2 + r)))
            workers.append(Worker(body_last, fn_args=[of_a_row[C - 1][r].cons(depth=K_tiles_col), of_b_mem[C - 1].cons(), of_z[r][C - 2].cons(), of_y_in[r].prod(), acc_bufs[(C - 1, r)], mm_k, z_k, add_z, ce_k], tile=Tile(C - 1, 2 + r)))
            workers.append(Worker(pna_body, fn_args=[of_y_mem[r].cons(), of_r_in[r].cons(), of_h_out[r].prod(), pna_t1[r], pna_t2[r], pna_wb[r], rms_k, mul_k, add_d, ce_k], tile=Tile(4, 2 + r)))

        a_hdls = [of_a_pack[c].prod(tile=Tile(c, 0)) for c in range(C)]
        b_hdls = [of_b_in[c].prod(tile=Tile(c, 0)) for c in range(C)]
        r_hdls = [of_r_in[r].prod(tile=Tile(4 + r, 0)) for r in range(R)]
        o_hdls = [of_h_out[r].cons(tile=Tile(4 + r, 0)) for r in range(R)]

        def seq(Ctx_t, Wo_t, H_t, NW_t, Out_t, *hdls):
            ahs, bhs, rhs, ohs = hdls[:C], hdls[C : 2 * C], hdls[2 * C : 2 * C + R], hdls[2 * C + R :]
            for p in range(n_passes):
                tg = TaskGroup()
                for c in range(C):
                    tap_a = TensorAccessPattern([S * K_dim], p * M_pass * K_dim + c * K_col, [R, K_tiles_col, MT, KT], [MT * K_dim, KT, K_dim, 1])
                    ahs[c].fill(Ctx_t, tap=tap_a, group=tg)
                    tap_w = TensorAccessPattern([K_dim * D], c * K_col * D, [N_tiles, K_tiles_col, KT, NT], [NT, KT * D, D, 1])
                    bhs[c].fill(Wo_t, tap=tap_w, group=tg)
                for r in range(R):
                    if p == 0:
                        rhs[r].fill(NW_t, group=tg)
                    tap_h = TensorAccessPattern([S * D], (p * M_pass + r * MT) * D, [MT * D], [1])
                    rhs[r].fill(H_t, tap=tap_h, group=tg)
                    ohs[r].drain(Out_t, tap=tap_h, wait=True, group=tg)
                tg.finish()

        rt = Runtime(seq, [Ctx_ty, Wo_ty, H_ty, W_ty, H_ty, *a_hdls, *b_hdls, *r_hdls, *o_hdls])
        return Program(DEV, rt, workers=workers).resolve_program()

    return o_post_fused


def build_qkv_norm_rope(S: int, d: int, H_q: int, H_kv: int):
    """Per-head weighted Q/K RMSNorm + Two-Halves RoPE + unweighted V RMSNorm."""
    H_tot = H_q + 2 * H_kv
    C = 4 if H_kv == 2 else 6
    h_per_col = H_tot // C
    q_cols, k_cols = H_q // h_per_col, H_kv // h_per_col

    rms_k, mul_k, rope_k, ce_k = K.rms_norm_eps(cols=d), K.mul_sized(d), K.rope(tile_size=d, two_halves=True), K.conv_even()
    row_ty = np.ndarray[(d,), np.dtype[bf16]]

    Raw_ty = np.ndarray[(H_tot * S * d,), np.dtype[bf16]]
    Q_ty = np.ndarray[(H_q * S * d,), np.dtype[bf16]]
    K_ty = np.ndarray[(H_kv * S * d,), np.dtype[bf16]]
    V_ty = np.ndarray[(H_kv * S * d,), np.dtype[bf16]]
    W_ty = np.ndarray[(d,), np.dtype[bf16]]
    L_ty = np.ndarray[(S * d,), np.dtype[bf16]]

    @iron.jit
    def qkv_norm_rope(Raw_in: In, qn_in: In, kn_in: In, LUT_in: In, Q_out: Out, K_out: Out, V_out: Out):
        of_in = [ObjectFifo(row_ty, depth=2, name=f"qnr_i_{c}") for c in range(C)]
        of_out = [ObjectFifo(row_ty, depth=2, name=f"qnr_o_{c}") for c in range(C)]
        of_lut_in = ObjectFifo(row_ty, depth=2, name="qnr_li")
        of_lut_mem = of_lut_in.cons().forward(tile=Tile(C, 1), depth=2, name="qnr_lm")
        t1 = [Buffer(row_ty, name=f"qnr_t1_{c}") for c in range(q_cols + k_cols)]
        t2 = [Buffer(row_ty, name=f"qnr_t2_{c}") for c in range(q_cols + k_cols)]
        wb = [Buffer(row_ty, name=f"qnr_wb_{c}") for c in range(q_cols + k_cols)]

        def qk_body(ox, olut, oy, b1, b2, wbuf, rms_fn, mul_fn, rp_fn, ce_fn):
            ce_fn()
            c_d, c_eps = _i32(d), _f32(EPS_RMS)
            w_in = ox.acquire(1)
            for i in range_(d):
                wbuf[i] = w_in[i]
            ox.release(1)
            for _ in range_(h_per_col * S):
                x, lut, y = ox.acquire(1), olut.acquire(1), oy.acquire(1)
                rms_fn(x, b1, c_d, c_eps)
                mul_fn(b1, wbuf, b2, c_d)
                rp_fn(b2, lut, y, c_d)
                ox.release(1)
                olut.release(1)
                oy.release(1)

        def v_body(ox, oy, rms_fn, ce_fn):
            ce_fn()
            c_d, c_eps = _i32(d), _f32(EPS_RMS)
            for _ in range_(h_per_col * S):
                x, y = ox.acquire(1), oy.acquire(1)
                rms_fn(x, y, c_d, c_eps)
                ox.release(1)
                oy.release(1)

        workers = [
            Worker(
                qk_body,
                fn_args=[of_in[c].cons(), of_lut_mem.cons(), of_out[c].prod(), t1[c], t2[c], wb[c], rms_k, mul_k, rope_k, ce_k],
                tile=Tile(c, 2),
            )
            for c in range(q_cols + k_cols)
        ] + [
            Worker(v_body, fn_args=[of_in[c].cons(), of_out[c].prod(), rms_k, ce_k], tile=Tile(c, 2))
            for c in range(q_cols + k_cols, C)
        ]

        in_hdls = [of_in[c].prod(tile=Tile(c, 0)) for c in range(C)]
        out_hdls = [of_out[c].cons(tile=Tile(c, 0)) for c in range(C)]
        lut_hdl = of_lut_in.prod(tile=Tile(C, 0))

        def seq(Raw_t, qn_t, kn_t, lut_t, Q_t, K_t, V_t, *hdls):
            ihs, ohs, lh = hdls[:C], hdls[C : 2 * C], hdls[2 * C]
            tg = TaskGroup()
            for _ in range(h_per_col):
                lh.fill(lut_t, group=tg)
            chunk = h_per_col * S * d
            for c in range(C):
                if c < q_cols:
                    ihs[c].fill(qn_t, group=tg)
                elif c < q_cols + k_cols:
                    ihs[c].fill(kn_t, group=tg)
                ihs[c].fill(Raw_t, tap=TensorAccessPattern([H_tot * S * d], c * chunk, [chunk], [1]), group=tg)
                if c < q_cols:
                    ohs[c].drain(Q_t, tap=TensorAccessPattern([H_q * S * d], c * chunk, [chunk], [1]), wait=True, group=tg)
                elif c < q_cols + k_cols:
                    ohs[c].drain(K_t, tap=TensorAccessPattern([H_kv * S * d], (c - q_cols) * chunk, [chunk], [1]), wait=True, group=tg)
                else:
                    ohs[c].drain(
                        V_t, tap=TensorAccessPattern([H_kv * S * d], (c - q_cols - k_cols) * chunk, [chunk], [1]), wait=True, group=tg
                    )
            tg.finish()

        rt = Runtime(seq, [Raw_ty, W_ty, W_ty, L_ty, Q_ty, K_ty, V_ty, *in_hdls, *out_hdls, lut_hdl])
        return Program(DEV, rt, workers=workers).resolve_program()

    return qkv_norm_rope


def build_attn_gqa_core(S: int, d: int, H_q: int, H_kv: int, M_q: int):
    """12-core fused QK^T -> Mask+Softmax -> PV with GQA KV multicast & zero S/P DDR spill."""
    KT, NT = KT_MM, NT_MM
    Q_strips, S_tiles, D_tiles = S // M_q, S // KT, d // KT
    group, bands = H_q // H_kv, M_q // 4

    mm_qkt = K.mm(M_q, KT, NT, bf16, bf16, b_col_maj=True)
    mm_pv = K.mm(M_q, KT, NT, bf16, bf16, b_col_maj=False)
    lay_q, lay_k, _ = mm_qkt.contract.layouts
    _, lay_v, _ = mm_pv.contract.layouts
    zero_s, zero_c = K.zero((M_q * NT,), bf16), K.zero((M_q * NT,), bf16)
    add_s, sm_k, ce_k = K.add_sized(S), K.softmax(S), K.conv_even()

    q_ty = np.ndarray[(M_q * KT,), np.dtype[bf16]]
    k_ty = np.ndarray[(NT * KT,), np.dtype[bf16]]
    v_ty = np.ndarray[(KT * NT,), np.dtype[bf16]]
    c_ty = np.ndarray[(M_q * NT,), np.dtype[bf16]]
    s_i8 = np.ndarray[(M_q * S * 2,), np.dtype[np.int8]]
    b_i8 = np.ndarray[(4 * S * 2,), np.dtype[np.int8]]
    m_ty = np.ndarray[(S,), np.dtype[bf16]]

    tap_s_mem = TensorAccessPattern([M_q * S * 2], 0, [bands, S_tiles, 512], [512, M_q * 128, 1])
    tap_s_sm = TensorAccessPattern([4 * S * 2], 0, [S_tiles * 8, 4, 16], [16, S * 2, 1])
    tap_p_sm = TensorAccessPattern([M_q * S * 2], 0, [S_tiles, M_q, 128], [128, S * 2, 1])
    tap_p_mem = TensorAccessPattern([M_q * S * 2], 0, [S_tiles * bands, 8, 4, 16], [512, 16, 128, 1])
    tap_c_mem = TensorAccessPattern([M_q * NT], 0, [bands, 4, 8, 8], [256, 8, 32, 1])

    Q_ty = np.ndarray[(H_q * S * d,), np.dtype[bf16]]
    K_ty = np.ndarray[(H_kv * S * d,), np.dtype[bf16]]
    V_ty = np.ndarray[(H_kv * S * d,), np.dtype[bf16]]
    M_ty = np.ndarray[(S * S,), np.dtype[bf16]]
    C_ty = np.ndarray[(S * H_q * d,), np.dtype[bf16]]

    @iron.jit
    def attn_gqa_core(Q_in: In, K_in: In, V_in: In, Mask_in: In, Ctx_out: Out):
        sh_q = [Tile(h, 0) for h in range(H_q)]
        mem_q = [Tile(h, 1) for h in range(H_q)]
        sh_k = [Tile(4 + hk, 0) for hk in range(H_kv)]
        mem_k = [Tile(4 + hk, 1) for hk in range(H_kv)]
        sh_v = [Tile(6 + hk if H_kv == 2 else 5, 0) for hk in range(H_kv)]
        mem_v = [Tile(6 + hk if H_kv == 2 else 5, 1) for hk in range(H_kv)]
        sh_m = Tile(7 if H_kv == 1 else 4, 0)
        mem_m = Tile(7 if H_kv == 1 else 4, 1)

        of_q_in = [ObjectFifo(q_ty, depth=2, name=f"ag_qi_{h}") for h in range(H_q)]
        of_q_mem = [of_q_in[h].cons().forward(tile=mem_q[h], depth=D_tiles, to_stream=lay_q.stream, name=f"ag_qm_{h}") for h in range(H_q)]
        of_k_in = [ObjectFifo(k_ty, depth=2, name=f"ag_ki_{hk}") for hk in range(H_kv)]
        of_k_mem = [of_k_in[hk].cons().forward(tile=mem_k[hk], depth=2, to_stream=lay_k.stream, name=f"ag_km_{hk}") for hk in range(H_kv)]
        of_v_in = [ObjectFifo(v_ty, depth=2, name=f"ag_vi_{hk}") for hk in range(H_kv)]
        of_v_mem = [of_v_in[hk].cons().forward(tile=mem_v[hk], depth=2, to_stream=lay_v.stream, name=f"ag_vm_{hk}") for hk in range(H_kv)]
        of_m_in = ObjectFifo(m_ty, depth=2, name="ag_mi")
        of_m_mem = of_m_in.cons().forward(tile=mem_m, depth=2, name="ag_mm")

        s_depth = 1 if M_q * S * 2 >= 8192 else 2
        of_s_in = [ObjectFifo(s_i8, depth=s_depth, name=f"ag_si_{h}") for h in range(H_q)]
        of_s_mem = []
        for h in range(H_q):
            f = of_s_in[h].cons().forward(tile=mem_q[h], depth=s_depth, to_stream=tap_s_mem, name=f"ag_sm_{h}")
            f._consumer_obj_type = b_i8
            f._from_stream_per_cons = tap_s_sm
            of_s_mem.append(f)

        of_p_in = [ObjectFifo(s_i8, depth=s_depth, to_stream=tap_p_sm, name=f"ag_pi_{h}") for h in range(H_q)]
        of_p_mem = [of_p_in[h].cons().forward(tile=mem_q[h], depth=s_depth, to_stream=tap_p_mem, name=f"ag_pm_{h}") for h in range(H_q)]
        of_c_in = [ObjectFifo(c_ty, depth=D_tiles, name=f"ag_ci_{h}") for h in range(H_q)]
        of_c_mem = [of_c_in[h].cons().forward(tile=mem_q[h], depth=2, to_stream=tap_c_mem, name=f"ag_cm_{h}") for h in range(H_q)]
        sm_tmp = [Buffer(np.ndarray[(S,), np.dtype[bf16]], name=f"ag_tmp_{h}") for h in range(H_q)]

        def qk_body(oq, ok, os_out, mm_fn, z_fn, ce_fn):
            ce_fn()
            s_tile_ty = ir.MemRefType.get([M_q * NT], ir.BF16Type.get())
            for _ in range_(Q_strips):
                q_tiles = oq.acquire(D_tiles)
                s_buf = os_out.acquire(1)
                for st in range(S_tiles):
                    s_v = memref.view(s_tile_ty, s_buf, constant(st * M_q * NT * 2, index=True), [])
                    z_fn(s_v)
                    for dt in range(D_tiles):
                        k_t = ok.acquire(1)
                        mm_fn(q_tiles[dt], k_t, s_v)
                        ok.release(1)
                os_out.release(1)
                oq.release(D_tiles)

        def sm_body(os_in, om, op_out, tmp_buf, add_fn, sm_fn, ce_fn):
            ce_fn()
            row_ty = ir.MemRefType.get([S], ir.BF16Type.get())
            c_s = _i32(S)
            for _ in range_(Q_strips):
                p_buf = op_out.acquire(1)
                for b in range(bands):
                    s_band = os_in.acquire(1)
                    for r in range(4):
                        s_row = memref.view(row_ty, s_band, constant(r * S * 2, index=True), [])
                        p_row = memref.view(row_ty, p_buf, constant((b * 4 + r) * S * 2, index=True), [])
                        m_row = om.acquire(1)
                        add_fn(s_row, m_row, tmp_buf, c_s)
                        om.release(1)
                        sm_fn(tmp_buf, p_row, c_s)
                    os_in.release(1)
                op_out.release(1)

        def pv_body(op, ov, oc, mm_fn, z_fn, ce_fn):
            ce_fn()
            p_ty = ir.MemRefType.get([M_q * KT], ir.BF16Type.get())
            for _ in range_(Q_strips):
                c_tiles = oc.acquire(D_tiles)
                for dt in range(D_tiles):
                    z_fn(c_tiles[dt])
                p_buf = op.acquire(1)
                for st in range(S_tiles):
                    p_v = memref.view(p_ty, p_buf, constant(st * M_q * KT * 2, index=True), [])
                    for dt in range(D_tiles):
                        v_t = ov.acquire(1)
                        mm_fn(p_v, v_t, c_tiles[dt])
                        ov.release(1)
                op.release(1)
                oc.release(D_tiles)

        workers = []
        for h in range(H_q):
            hk = h // group
            workers.append(Worker(qk_body, fn_args=[of_q_mem[h].cons(), of_k_mem[hk].cons(), of_s_in[h].prod(), mm_qkt, zero_s, ce_k], tile=Tile(h, 2)))
            workers.append(Worker(sm_body, fn_args=[of_s_mem[h].cons(), of_m_mem.cons(), of_p_in[h].prod(), sm_tmp[h], add_s, sm_k, ce_k], tile=Tile(h, 3)))
            workers.append(Worker(pv_body, fn_args=[of_p_mem[h].cons(), of_v_mem[hk].cons(), of_c_in[h].prod(), mm_pv, zero_c, ce_k], tile=Tile(h, 4)))

        q_hdls = [of_q_in[h].prod(tile=sh_q[h]) for h in range(H_q)]
        c_hdls = [of_c_mem[h].cons(tile=sh_q[h]) for h in range(H_q)]
        k_hdls = [of_k_in[hk].prod(tile=sh_k[hk]) for hk in range(H_kv)]
        v_hdls = [of_v_in[hk].prod(tile=sh_v[hk]) for hk in range(H_kv)]
        m_hdl = of_m_in.prod(tile=sh_m)

        def seq(Q_t, K_t, V_t, M_t, C_t, *hdls):
            qhs, chs = hdls[:H_q], hdls[H_q : 2 * H_q]
            khs, vhs, mh = hdls[2 * H_q : 2 * H_q + H_kv], hdls[2 * H_q + H_kv : 2 * H_q + 2 * H_kv], hdls[-1]
            B = min(4, Q_strips)
            for g0 in range(0, Q_strips, B):
                cnt = min(B, Q_strips - g0)
                tg = TaskGroup()
                for h in range(H_q):
                    tap_q = TensorAccessPattern([H_q * S * d], h * S * d + g0 * M_q * d, [cnt, D_tiles, M_q, KT], [M_q * d, KT, d, 1])
                    qhs[h].fill(Q_t, tap=tap_q, group=tg)
                    tap_c = TensorAccessPattern([S * H_q * d], h * d + g0 * M_q * H_q * d, [cnt, D_tiles, M_q, NT], [M_q * H_q * d, NT, H_q * d, 1])
                    chs[h].drain(C_t, tap=tap_c, wait=True, group=tg)
                tap_m = TensorAccessPattern([S * S], g0 * M_q * S, [cnt * M_q * S], [1])
                mh.fill(M_t, tap=tap_m, group=tg)
                for hk in range(H_kv):
                    tap_k = TensorAccessPattern([H_kv * S * d], hk * S * d, [S_tiles, D_tiles, NT, KT], [NT * d, KT, d, 1])
                    tap_v = TensorAccessPattern([H_kv * S * d], hk * S * d, [S_tiles, D_tiles, KT, NT], [KT * d, NT, d, 1])
                    for _ in range(cnt):
                        khs[hk].fill(K_t, tap=tap_k, group=tg)
                        vhs[hk].fill(V_t, tap=tap_v, group=tg)
                tg.finish()

        rt = Runtime(seq, [Q_ty, K_ty, V_ty, M_ty, C_ty, *q_hdls, *c_hdls, *k_hdls, *v_hdls, m_hdl])
        return Program(DEV, rt, workers=workers).resolve_program()

    return attn_gqa_core


# ==============================================================================
# Part C: Compiled 8-XCLBIN Kernel Bank, Scratch, Weights & Runners
# ==============================================================================
class AttnKernelBank:
    """Caches the 8 fused XCLBIN kernels shared across all 24 layers at seq S."""

    def __init__(self, S: int):
        self.S = S
        cfg_l = make_layer_config(S, is_full=False)
        cfg_f = make_layer_config(S, is_full=True)
        self.cfg_l, self.cfg_f = cfg_l, cfg_f

        self.qkv_pre_local = build_qkv_pre_fused(S, cfg_l.qkv_out_dim, cfg_l.qkv_cols, head_d=cfg_l.d)
        self.qkv_nr_local = build_qkv_norm_rope(S, cfg_l.d, cfg_l.H_q, cfg_l.H_kv)
        self.attn_local = build_attn_gqa_core(S, cfg_l.d, cfg_l.H_q, cfg_l.H_kv, cfg_l.M_q)
        self.o_post_local = build_o_post_fused(S, cfg_l.H_q * cfg_l.d)

        self.qkv_pre_full = build_qkv_pre_fused(S, cfg_f.qkv_out_dim, cfg_f.qkv_cols, head_d=cfg_f.d)
        self.qkv_nr_full = build_qkv_norm_rope(S, cfg_f.d, cfg_f.H_q, cfg_f.H_kv)
        self.attn_full = build_attn_gqa_core(S, cfg_f.d, cfg_f.H_q, cfg_f.H_kv, cfg_f.M_q)
        self.o_post_full = build_o_post_fused(S, cfg_f.H_q * cfg_f.d)

        # Shared NPU scratch buffers (X_n & Y eliminated by fusion)
        self.H_a = _npu_zeros(S * 512)
        self.H_b = _npu_zeros(S * 512)
        self.QKV_raw_l = _npu_zeros(S * cfg_l.qkv_out_dim)
        self.QKV_raw_f = _npu_zeros(S * cfg_f.qkv_out_dim)
        self.Q_l = _npu_zeros(cfg_l.H_q * S * cfg_l.d)
        self.K_l = _npu_zeros(cfg_l.H_kv * S * cfg_l.d)
        self.V_l = _npu_zeros(cfg_l.H_kv * S * cfg_l.d)
        self.Q_f = _npu_zeros(cfg_f.H_q * S * cfg_f.d)
        self.K_f = _npu_zeros(cfg_f.H_kv * S * cfg_f.d)
        self.V_f = _npu_zeros(cfg_f.H_kv * S * cfg_f.d)
        self.Ctx_l = _npu_zeros(S * cfg_l.H_q * cfg_l.d)
        self.Ctx_f = _npu_zeros(S * cfg_f.H_q * cfg_f.d)

    def make_layer_weights_host(self, rng: np.random.Generator, cfg: AttnLayerConfig, scale: float = 0.04) -> dict[str, np.ndarray]:
        d = cfg.d
        return {
            "in_norm": rng.normal(1.0, 0.05, (cfg.D,)).astype(bf16),
            "W_qkv": rng.normal(0.0, scale, (cfg.D, cfg.qkv_out_dim)).astype(bf16),
            "q_norm": rng.normal(1.0, 0.05, (d,)).astype(bf16),
            "k_norm": rng.normal(1.0, 0.05, (d,)).astype(bf16),
            "W_o": rng.normal(0.0, scale, (cfg.H_q * d, cfg.D)).astype(bf16),
            "post_attn": rng.normal(0.0, scale, (cfg.D,)).astype(bf16),
        }

    def upload_weights(self, w_host: dict[str, np.ndarray]) -> dict[str, object]:
        return {k: _npu_tensor(v) for k, v in w_host.items()}

    def layer_runs_specs(self, is_full: bool, H_in, H_out, w_dev, lut_dev, mask_dev):
        if is_full:
            return [
                ("qkv_pre_full", self.qkv_pre_full, [H_in, w_dev["in_norm"], w_dev["W_qkv"], self.QKV_raw_f]),
                ("qnr_full", self.qkv_nr_full, [self.QKV_raw_f, w_dev["q_norm"], w_dev["k_norm"], lut_dev, self.Q_f, self.K_f, self.V_f]),
                ("attn_full", self.attn_full, [self.Q_f, self.K_f, self.V_f, mask_dev, self.Ctx_f]),
                ("o_post_full", self.o_post_full, [self.Ctx_f, w_dev["W_o"], H_in, w_dev["post_attn"], H_out]),
            ]
        return [
            ("qkv_pre_local", self.qkv_pre_local, [H_in, w_dev["in_norm"], w_dev["W_qkv"], self.QKV_raw_l]),
            ("qnr_local", self.qkv_nr_local, [self.QKV_raw_l, w_dev["q_norm"], w_dev["k_norm"], lut_dev, self.Q_l, self.K_l, self.V_l]),
            ("attn_local", self.attn_local, [self.Q_l, self.K_l, self.V_l, mask_dev, self.Ctx_l]),
            ("o_post_local", self.o_post_local, [self.Ctx_l, w_dev["W_o"], H_in, w_dev["post_attn"], H_out]),
        ]

    def run_layer(self, is_full: bool, H_in, H_out, w_dev, lut_dev, mask_dev, warmup: int = 2, iters: int = 8) -> dict[str, tuple[object, float]]:
        specs = self.layer_runs_specs(is_full, H_in, H_out, w_dev, lut_dev, mask_dev)
        out = {}
        for name, fn, args in specs:
            hdl, kres = fn(*args)
            out[name] = (hdl, kres.npu_time / 1e3)
        if iters > 0:
            for _ in range(warmup):
                for _, fn, args in specs:
                    fn(*args)
            timings = {name: [] for name, _, _ in specs}
            for _ in range(iters):
                for name, fn, args in specs:
                    _, kres = fn(*args)
                    timings[name].append(kres.npu_time / 1e3)
            out = {name: (out[name][0], float(np.median(timings[name]))) for name, _, _ in specs}
        return out


# ==============================================================================
# Verification & Benchmark Runners
# ==============================================================================
def run_subkernels(S: int = 128, iters: int = 20):
    print(f"\n=== Part 1: 4 Standalone Sub-Kernels Verification (S={S}) ===", flush=True)
    rng = np.random.default_rng(42)

    for d, nh, theta in [(256, 4, 10000.0), (512, 4, 1000000.0)]:
        X = rng.normal(0, 1.0, (nh, S, d)).astype(bf16)
        lut = make_rope_lut(S, d, theta)
        ref = ref_rope_two_halves(X, lut)
        Xt, Lt, Yt = _npu_tensor(X), _npu_tensor(lut), _npu_zeros(nh * S * d)
        fn = build_rope_subkernel(S, d, nh)
        hdl, cs, call_us, hw_us = bench_kernel(fn, [Xt, Lt, Yt], iters=iters)
        got = Yt.numpy().reshape(nh, S, d)
        m = err_metrics(got, ref)
        ch_us = bench_chain_same(hdl, [Xt, Lt, Yt], 16)
        print(
            f"[RoPE    d={d} H={nh}     ] hw={hw_us:6.1f}us call={call_us:6.1f}us ch16={ch_us:5.1f}us | "
            f"max={m['max_abs']:.4f} mean={m['mean_abs']:.5f} cos={m['cos']:.7f} ({cs:.1f}s)",
            flush=True,
        )

    for tag, d, H_kv, M_q in [("local", 256, 2, 32), ("full ", 512, 1, 16)]:
        Q = rng.normal(0, 0.25, (4, S, d)).astype(bf16)
        K_m = rng.normal(0, 0.25, (H_kv, S, d)).astype(bf16)
        ref = ref_qkt(Q, K_m)
        Qt, Kt, St = _npu_tensor(Q), _npu_tensor(K_m), _npu_zeros(4 * S * S)
        fn = build_qkt_subkernel(S, d, 4, H_kv, M_q)
        hdl, cs, call_us, hw_us = bench_kernel(fn, [Qt, Kt, St], iters=iters)
        got = St.numpy().reshape(4, S, S)
        m = err_metrics(got, ref)
        ch_us = bench_chain_same(hdl, [Qt, Kt, St], 16)
        print(
            f"[QKT     {tag} d={d} kv={H_kv}] hw={hw_us:6.1f}us call={call_us:6.1f}us ch16={ch_us:5.1f}us | "
            f"max={m['max_abs']:.4f} mean={m['mean_abs']:.5f} cos={m['cos']:.7f} ({cs:.1f}s)",
            flush=True,
        )

    Sc = rng.normal(0, 2.0, (4, S, S)).astype(bf16)
    fn_sm = build_softmax_swa_subkernel(S, 4)
    for w_label, w_val in [("W=512(all)", 512), ("W=32 (swa)", 32)]:
        mask = make_swa_mask(S, w_val, is_full=False)
        ref = ref_softmax_swa(Sc, mask)
        Sct, Mt, Pt = _npu_tensor(Sc), _npu_tensor(mask), _npu_zeros(4 * S * S)
        hdl, cs, call_us, hw_us = bench_kernel(fn_sm, [Sct, Mt, Pt], iters=iters)
        got = Pt.numpy().reshape(4, S, S)
        m = err_metrics(got, ref)
        pos = np.arange(S)
        masked_positions = np.abs(pos[:, None] - pos[None, :]) > w_val
        masked_max = float(np.abs(got[:, masked_positions].astype(np.float32)).max()) if np.any(masked_positions) else 0.0
        ch_us = bench_chain_same(hdl, [Sct, Mt, Pt], 16)
        print(
            f"[Softmax {w_label}     ] hw={hw_us:6.1f}us call={call_us:6.1f}us ch16={ch_us:5.1f}us | "
            f"max={m['max_abs']:.5f} mean={m['mean_abs']:.6f} masked_max={masked_max:.1e} cos={m['cos']:.7f} ({cs:.1f}s)",
            flush=True,
        )

    for tag, d, H_kv, M_q in [("local", 256, 2, 32), ("full ", 512, 1, 16)]:
        P = np.abs(rng.normal(0, 0.2, (4, S, S))).astype(np.float32)
        P = (P / P.sum(axis=-1, keepdims=True)).astype(bf16)
        V = rng.normal(0, 0.5, (H_kv, S, d)).astype(bf16)
        ref = ref_pv(P, V)
        Pt, Vt, Ct = _npu_tensor(P), _npu_tensor(V), _npu_zeros(S * 4 * d)
        fn = build_pv_subkernel(S, d, 4, H_kv, M_q)
        hdl, cs, call_us, hw_us = bench_kernel(fn, [Pt, Vt, Ct], iters=iters)
        got = Ct.numpy().reshape(S, 4 * d)
        m = err_metrics(got, ref)
        ch_us = bench_chain_same(hdl, [Pt, Vt, Ct], 16)
        print(
            f"[PV      {tag} d={d} kv={H_kv}] hw={hw_us:6.1f}us call={call_us:6.1f}us ch16={ch_us:5.1f}us | "
            f"max={m['max_abs']:.4f} mean={m['mean_abs']:.5f} cos={m['cos']:.7f} ({cs:.1f}s)",
            flush=True,
        )


def verify_single_layer(bank: AttnKernelBank, S: int, window_test: int = 32):
    print(f"\n=== Part 2: Complete Single-Layer Attention Verification (S={S}) ===", flush=True)
    rng = np.random.default_rng(123)
    H_in_host = rng.normal(0, 1.0, (S, 512)).astype(bf16)
    H_in_dev = _npu_tensor(H_in_host)

    results = {}
    cases = [
        ("local_unmasked", make_layer_config(S, is_full=False, window=512)),
        ("local_swa_act", make_layer_config(S, is_full=False, window=window_test)),
        ("full_global", make_layer_config(S, is_full=True, window=-1)),
    ]
    local_w_host = bank.make_layer_weights_host(rng, make_layer_config(S, is_full=False), scale=0.05)

    outs_np = {}
    for label, cfg in cases:
        w_host = local_w_host if not cfg.is_full else bank.make_layer_weights_host(rng, cfg, scale=0.05)
        w_dev = bank.upload_weights(w_host)
        lut_h = make_rope_lut(S, cfg.d, cfg.theta)
        mask_h = make_swa_mask(S, cfg.window, is_full=cfg.is_full)
        lut_d, mask_d = _npu_tensor(lut_h), _npu_tensor(mask_h)

        ref = ref_attn_layer(
            H_in_host, w_host["in_norm"], w_host["W_qkv"], w_host["q_norm"], w_host["k_norm"],
            lut_h, mask_h, w_host["W_o"], w_host["post_attn"], cfg,
        )
        stage_info = bank.run_layer(cfg.is_full, H_in_dev, bank.H_b, w_dev, lut_d, mask_d, warmup=2, iters=8)
        got = bank.H_b.numpy().reshape(S, 512)
        outs_np[label] = got.astype(np.float32)
        m = err_metrics(got, ref)
        hw_sum = sum(v[1] for v in stage_info.values())
        bkd = " ".join(f"{k}={v[1]:.1f}us" for k, v in stage_info.items())
        print(
            f"[{label:14s} d={cfg.d} kv={cfg.H_kv} W={cfg.window:3d}] "
            f"hw_sum={hw_sum:6.1f}us | max={m['max_abs']:.4f} mean={m['mean_abs']:.5f} cos={m['cos']:.7f}",
            flush=True,
        )
        print(f"   breakdown: {bkd}", flush=True)
        results[label] = (m, hw_sum, stage_info)

    swa_diff = float(np.abs(outs_np["local_unmasked"] - outs_np["local_swa_act"]).max())
    print(f"[SWA Mask Check] max_diff(W=512 vs W={window_test}) = {swa_diff:.4f} (confirms active window masking)", flush=True)
    return results


def _layer_bytes_and_flops(cfg: AttnLayerConfig) -> tuple[int, int]:
    S, D, H_q, H_kv, d = cfg.S, cfg.D, cfg.H_q, cfg.H_kv, cfg.d
    qkv_out = cfg.qkv_out_dim
    unique_bytes = 2 * (
        S * D + D + D * qkv_out + S * qkv_out
        + 2 * d + S * d + S * (H_q + 2 * H_kv) * d
        + (H_q + 2 * H_kv) * S * d + S * S + S * H_q * d
        + S * H_q * d + (H_q * d) * D + S * D + D + S * D
    )
    flops = (
        2 * S * D * qkv_out
        + 2 * H_q * S * d * S
        + 2 * H_q * S * S * d
        + 2 * S * (H_q * d) * D
    )
    return unique_bytes, flops


def run_chain_24(bank: AttnKernelBank, S: int, iters: int = 10):
    print(f"\n=== Part 3: 24-Layer Mixed Attention Chain (S={S}, 20 local + 4 full) ===", flush=True)
    rng = np.random.default_rng(2026)
    H0 = rng.normal(0, 1.0, (S, 512)).astype(bf16)

    lut_l = _npu_tensor(make_rope_lut(S, 256, 10_000.0))
    lut_f = _npu_tensor(make_rope_lut(S, 512, 1_000_000.0))
    mask_l = _npu_tensor(make_swa_mask(S, 512, is_full=False))
    mask_f = _npu_tensor(make_swa_mask(S, -1, is_full=True))

    lut_l_h = make_rope_lut(S, 256, 10_000.0)
    lut_f_h = make_rope_lut(S, 512, 1_000_000.0)
    mask_l_h = make_swa_mask(S, 512, is_full=False)
    mask_f_h = make_swa_mask(S, -1, is_full=True)

    layers_w_h, layers_w_d = [], []
    for lt in LAYER_TYPES_24:
        is_full = lt == "full"
        cfg = make_layer_config(S, is_full=is_full, window=512)
        wh = bank.make_layer_weights_host(rng, cfg, scale=0.04)
        layers_w_h.append((is_full, cfg, wh))
        layers_w_d.append((is_full, bank.upload_weights(wh)))

    href = H0.copy()
    for is_full, cfg, wh in layers_w_h:
        href = ref_attn_layer(
            href, wh["in_norm"], wh["W_qkv"], wh["q_norm"], wh["k_norm"],
            lut_f_h if is_full else lut_l_h, mask_f_h if is_full else mask_l_h,
            wh["W_o"], wh["post_attn"], cfg,
        )

    chain_runs = []
    stage_hdls = {}
    stage_sample_args = {}
    h_cur_src = _npu_tensor(H0)
    for li, (is_full, w_dev) in enumerate(layers_w_d):
        H_in = h_cur_src if li == 0 else (bank.H_a if li % 2 == 0 else bank.H_b)
        H_out = bank.H_b if li % 2 == 0 else bank.H_a
        specs = bank.layer_runs_specs(is_full, H_in, H_out, w_dev, lut_f if is_full else lut_l, mask_f if is_full else mask_l)
        for name, fn, args in specs:
            hdl, _ = fn(*args)
            chain_runs.append((hdl, args))
            stage_hdls[name] = hdl
            stage_sample_args[name] = args
    final_npu_buf = bank.H_a

    got_seq = final_npu_buf.numpy().reshape(S, 512)
    m_seq = err_metrics(got_seq, href)
    print(f"[24L Sequential Check] max={m_seq['max_abs']:.4f} mean={m_seq['mean_abs']:.5f} cos={m_seq['cos']:.7f}", flush=True)

    het_chain_ok = False
    het_npu_ms = None
    try:
        DefaultNPURuntime.run_chain(chain_runs)
        got_chain = final_npu_buf.numpy().reshape(S, 512)
        m_ch = err_metrics(got_chain, href)
        for _ in range(2):
            DefaultNPURuntime.run_chain(chain_runs)
        t_npu, t_wall = [], []
        for _ in range(iters):
            tw = time.perf_counter()
            res = DefaultNPURuntime.run_chain(chain_runs)
            t_wall.append((time.perf_counter() - tw) * 1e3)
            t_npu.append(res.npu_time / 1e6)
        het_npu_ms = float(np.median(t_npu))
        het_wall_ms = float(np.median(t_wall))
        het_chain_ok = True
        print(
            f"[24L HRX run_chain(96)] PASS npu_time={het_npu_ms:.3f}ms ({het_npu_ms*1000/24:.1f}us/L) "
            f"wall={het_wall_ms:.3f}ms | max={m_ch['max_abs']:.4f} mean={m_ch['mean_abs']:.5f} cos={m_ch['cos']:.7f}",
            flush=True,
        )
    except Exception as exc:
        print(f"[24L HRX run_chain(96)] heterogeneous run_chain unsupported ({type(exc).__name__}: {exc})", flush=True)

    stage_ch = {}
    for name, hdl in stage_hdls.items():
        n_rep = 4 if name.endswith("_full") else 20
        stage_ch[name] = bench_chain_same(hdl, stage_sample_args[name], n_rep, warmup=2, iters=10)

    local_chain_us = sum(stage_ch[n] for n in ("qkv_pre_local", "qnr_local", "attn_local", "o_post_local"))
    full_chain_us = sum(stage_ch[n] for n in ("qkv_pre_full", "qnr_full", "attn_full", "o_post_full"))
    est_chain24_ms = (20 * local_chain_us + 4 * full_chain_us) / 1e3

    seq_npu_ms, seq_wall_ms = [], []
    for _ in range(min(5, iters)):
        tw = time.perf_counter()
        acc_ns = 0
        for hdl, args in chain_runs:
            acc_ns += DefaultNPURuntime.run_chain([(hdl, args)]).npu_time
        seq_wall_ms.append((time.perf_counter() - tw) * 1e3)
        seq_npu_ms.append(acc_ns / 1e6)

    bl, fl = _layer_bytes_and_flops(bank.cfg_l)
    bf, ff = _layer_bytes_and_flops(bank.cfg_f)
    tot_bytes = 20 * bl + 4 * bf
    tot_flops = 20 * fl + 4 * ff
    ref_ms = het_npu_ms if het_chain_ok else est_chain24_ms
    bw_gbs = tot_bytes / (ref_ms * 1e-3) / 1e9
    gflops = tot_flops / (ref_ms * 1e-3) / 1e9

    print(
        f"[24L Stage Breakdown (chained)] local={local_chain_us:.1f}us/L full={full_chain_us:.1f}us/L "
        f"=> sum={est_chain24_ms:.3f}ms ({bw_gbs:.1f} GB/s, {gflops:.1f} GFLOPS)",
        flush=True,
    )
    print(
        "   " + " ".join(f"{k}={v:.1f}us" for k, v in stage_ch.items()),
        flush=True,
    )
    print(
        f"[24L Per-Kernel Seq] npu_time={np.median(seq_npu_ms):.3f}ms wall={np.median(seq_wall_ms):.3f}ms",
        flush=True,
    )
    return {
        "het_ok": het_chain_ok,
        "het_ms": het_npu_ms,
        "est_chain_ms": est_chain24_ms,
        "bw_gbs": bw_gbs,
        "gflops": gflops,
        "seq_cos": m_seq["cos"],
        "stage_ch": stage_ch,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subkernels", action="store_true")
    ap.add_argument("--layers", action="store_true")
    ap.add_argument("--chain24", action="store_true")
    ap.add_argument("--s512", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()
    if not any([args.subkernels, args.layers, args.chain24, args.s512, args.all]):
        args.all = True

    if args.all or args.subkernels:
        run_subkernels(S=128, iters=args.iters)

    if args.all or args.layers or args.chain24:
        bank_128 = AttnKernelBank(S=128)
        if args.all or args.layers:
            verify_single_layer(bank_128, S=128, window_test=32)
        if args.all or args.chain24:
            run_chain_24(bank_128, S=128, iters=args.iters)

    if args.all or args.s512:
        bank_512 = AttnKernelBank(S=512)
        verify_single_layer(bank_512, S=512, window_test=128)
        run_chain_24(bank_512, S=512, iters=min(5, args.iters))


if __name__ == "__main__":
    main()
