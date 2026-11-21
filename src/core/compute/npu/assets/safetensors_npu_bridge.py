#!/usr/bin/env python3
"""Safetensors -> NPU chunked weight bridge & shard planner (EmbeddingGemma 2).

Streams tensors in bounded chunks directly from `model.safetensors`, avoids
loading large host-only tables (`embed_tokens`, `vision_tower`, `audio_tower`),
and packs linear weights into `v8bfp16ebs8` / tiled `bf16` for XDNA2 IRON.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Iterator

import numpy as np
from aie.utils import bfp


DEFAULT_SAFETENSORS = Path(
    "/mnt/nvme-data/mbufo-models/embeddinggemma-2/model.safetensors"
)


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header, 8 + header_len


def read_tensor_chunk_bf16_u16(
    path: Path, header: dict, data_base: int, name: str, rel_byte_off: int, byte_len: int
) -> np.ndarray:
    meta = header[name]
    off0, off1 = meta["data_offsets"]
    if rel_byte_off < 0 or rel_byte_off + byte_len > (off1 - off0):
        raise ValueError(f"Chunk [{rel_byte_off}, +{byte_len}) out of range for {name}")
    with path.open("rb") as f:
        f.seek(data_base + off0 + rel_byte_off)
        raw = f.read(byte_len)
    return np.frombuffer(raw, dtype=np.uint16).copy()


def bf16_u16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def plan_shards(header: dict, max_shard_mb: int = 64) -> dict:
    max_bytes = max_shard_mb * (1 << 20)
    host_bytes = 0
    npu_packed_bytes = 0
    shards = []
    cur = {"tensors": [], "packed_bytes": 0}
    layers_seen = set()

    for name, meta in header.items():
        if name == "__metadata__":
            continue
        b0, b1 = meta["data_offsets"]
        raw = b1 - b0
        shape = meta["shape"]
        if (
            not name.startswith("language_model.")
            or name == "language_model.embed_tokens.weight"
        ):
            host_bytes += raw
            continue

        if name.startswith("language_model.layers."):
            layers_seen.add(int(name.split(".")[2]))

        elems = int(np.prod(shape)) if shape else 1
        is_2d = len(shape) == 2 and (elems % 8 == 0)
        packed = (elems * 9 // 8) if is_2d else raw
        if cur["tensors"] and cur["packed_bytes"] + packed > max_bytes:
            shards.append(cur)
            cur = {"tensors": [], "packed_bytes": 0}
        cur["tensors"].append((name, "bfp16ebs8" if is_2d else "bf16", packed))
        cur["packed_bytes"] += packed
        npu_packed_bytes += packed

    if cur["tensors"]:
        shards.append(cur)

    return {
        "layers": len(layers_seen),
        "host_only_mib": host_bytes / (1 << 20),
        "npu_packed_mib": npu_packed_bytes / (1 << 20),
        "num_shards": len(shards),
        "shard_mib": [round(s["packed_bytes"] / (1 << 20), 2) for s in shards],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_SAFETENSORS)
    ap.add_argument("--max-shard-mb", type=int, default=64)
    args = ap.parse_args()

    header, base = read_safetensors_header(args.model)
    plan = plan_shards(header, max_shard_mb=args.max_shard_mb)
    print("SAFETENSORS_SHARD_PLAN:", json.dumps(plan))

    # Smoke-test chunked read & bfp16ebs8 pack on layer 0 & layer 5 weights
    for tname in (
        "language_model.layers.0.self_attn.q_proj.weight",
        "language_model.layers.5.mlp.up_proj.weight",
    ):
        u16 = read_tensor_chunk_bf16_u16(args.model, header, base, tname, 0, 8192)
        f32 = bf16_u16_to_f32(u16)
        packed = bfp.encode(f32, rounding="floor")
        decoded = bfp.decode(packed)
        err = float(np.max(np.abs(f32 - decoded)))
        print(
            f"CHUNK_PACK_OK {tname}: bf16_bytes={u16.nbytes} bfp_bytes={packed.nbytes} max_abs_err={err:.6f}"
        )


if __name__ == "__main__":
    main()
