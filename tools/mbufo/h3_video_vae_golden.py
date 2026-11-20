#!/usr/bin/env python3
"""Capture an independent 256x256x22 MiniMax H3 VisualVAE oracle."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
from pathlib import Path

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != _SCRIPT_DIRECTORY
]

import numpy
import torch
import torch.nn.functional as functional
from safetensors import safe_open


MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
REFERENCE_REVISION = "8974cc055ea9c02fcd14cc27dfda3e1027c05153"
CHANNELS = 24
LATENT_TIME = 7
LATENT_HEIGHT = 16
LATENT_WIDTH = 16
FRAMES = 22
HIDDEN = 2048
BLOCKS = 36
HEADS = 32
HEAD_DIMENSION = 64
INNER = HEADS * HEAD_DIMENSION
FEED_FORWARD = 8192
REGISTERS = 4
SUFFIX = 5
ROPE_HALF = 24
OUTPUT_PATCH = 3 * 4 * 16 * 16
EPSILON = 1.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "float16"),
        default="float32",
        help=(
            "float32 preserves the frozen component oracle; float16 mirrors "
            "the released pipeline's autocast decode recipe"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_tensor(path: Path, tensor: torch.Tensor) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        tensor.detach()
        .contiguous()
        .float()
        .cpu()
        .numpy()
        .astype("<f4", copy=False)
    )
    path.write_bytes(memoryview(numpy.ascontiguousarray(array)).tobytes())
    return {
        "dtype": "F32",
        "shape": list(tensor.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_f32(path: Path, shape: tuple[int, ...], device: torch.device):
    array = numpy.fromfile(path, dtype="<f4")
    expected = math.prod(shape)
    if array.size != expected:
        raise RuntimeError(
            f"{path} has {array.size} F32 elements, expected {expected}"
        )
    return torch.from_numpy(array.copy()).reshape(shape).to(device)


def rms(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    values = input_tensor.float()
    inverse = torch.rsqrt(
        values.square().mean(dim=-1, keepdim=True) + EPSILON
    )
    return (values * inverse * weight.float()).to(input_tensor.dtype)


def rope(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    axes: list[tuple[float, float, float]] = []
    for time_index in range(LATENT_TIME):
        for y in range(LATENT_HEIGHT):
            for x in range(LATENT_WIDTH):
                axes.append(
                    (
                        2.0 * ((time_index + 0.5) / LATENT_TIME) - 1.0,
                        2.0 * ((y + 0.5) / LATENT_HEIGHT) - 1.0,
                        2.0 * ((x + 0.5) / LATENT_WIDTH) - 1.0,
                    )
                )
    axes.extend([(0.0, 0.0, 0.0)] * SUFFIX)
    positions = torch.tensor(axes, dtype=torch.float32, device=device)
    frequency = torch.arange(8, dtype=torch.float32, device=device)
    inverse = 1.0 / torch.pow(100.0, frequency * 0.125)
    angles = 2.0 * math.pi * positions[:, :, None] * inverse[None, None, :]
    cosine = torch.cos(angles).reshape(-1, ROPE_HALF)
    sine = torch.sin(angles).reshape(-1, ROPE_HALF)
    cosine[-SUFFIX:] = 1.0
    sine[-SUFFIX:] = 0.0
    return cosine, sine


def qkv_norm_rope(
    qkv: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped = qkv.view(qkv.shape[0], HEADS, 3, HEAD_DIMENSION)
    query = grouped[:, :, 0, :].float()
    key = grouped[:, :, 1, :].float()
    value = grouped[:, :, 2, :].contiguous()
    query = query * torch.rsqrt(
        query.square().mean(dim=-1, keepdim=True) + EPSILON
    )
    key = key * torch.rsqrt(
        key.square().mean(dim=-1, keepdim=True) + EPSILON
    )
    cosine = cosine[:, None, :]
    sine = sine[:, None, :]

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        first = tensor[:, :, :ROPE_HALF]
        second = tensor[:, :, ROPE_HALF : ROPE_HALF * 2]
        tail = tensor[:, :, ROPE_HALF * 2 :]
        return torch.cat(
            (
                first * cosine - second * sine,
                second * cosine + first * sine,
                tail,
            ),
            dim=-1,
        )

    return rotate(query).to(qkv.dtype), rotate(key).to(qkv.dtype), value


def full_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    query_heads = query.transpose(0, 1)
    key_heads = key.transpose(0, 1)
    value_heads = value.transpose(0, 1)
    scores = torch.matmul(query_heads, key_heads.transpose(-1, -2))
    scores *= 1.0 / math.sqrt(HEAD_DIMENSION)
    probabilities = torch.softmax(scores, dim=-1)
    return (
        torch.matmul(probabilities, value_heads)
        .transpose(0, 1)
        .contiguous()
        .view(query.shape[0], INNER)
    )


def unpack_frames(projected: torch.Tensor) -> torch.Tensor:
    patches = projected[: LATENT_TIME * LATENT_HEIGHT * LATENT_WIDTH].view(
        LATENT_TIME,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        3,
        4,
        16,
        16,
    )
    mean = torch.tensor(
        [0.485, 0.456, 0.406], dtype=torch.float32, device=projected.device
    )
    deviation = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32,
        device=projected.device,
    )
    frames = []
    for frame in range(FRAMES):
        decoded_time = frame + 3
        if frame >= 17:
            decoded_time += 3
        patch_time, within_time = divmod(decoded_time, 4)
        pixels = (
            patches[patch_time, :, :, :, within_time, :, :]
            .permute(0, 3, 1, 4, 2)
            .contiguous()
            .view(LATENT_HEIGHT * 16, LATENT_WIDTH * 16, 3)
        )
        frames.append(
            torch.clamp(pixels * deviation + mean, min=0.0, max=1.0)
        )
    return torch.stack(frames)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm torch device is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    config = json.loads(
        (args.model_root / "FL2VA" / "video_vae" / "config.json").read_text()
    )
    mean = torch.tensor(
        config["latents_mean"], dtype=torch.float32, device=device
    )
    deviation = torch.tensor(
        config["latents_std"], dtype=torch.float32, device=device
    )
    latent = read_f32(
        args.latent,
        (CHANNELS, LATENT_TIME, LATENT_HEIGHT, LATENT_WIDTH),
        device,
    )
    checkpoint_path = (
        args.model_root
        / "FL2VA"
        / "video_vae"
        / "source"
        / "model.safetensors"
    )
    started = time.monotonic()
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if args.compute_dtype == "float16"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast, safe_open(
        checkpoint_path, framework="pt", device=args.device
    ) as checkpoint:
        rows = (
            latent.permute(1, 2, 3, 0).contiguous().view(-1, CHANNELS)
            * deviation
            + mean
        )
        post_weight = checkpoint.get_tensor("post_quant_conv.weight").view(
            CHANNELS, CHANNELS
        )
        hidden = functional.linear(
            rows,
            post_weight,
            checkpoint.get_tensor("post_quant_conv.bias"),
        )
        hidden = functional.linear(
            hidden,
            checkpoint.get_tensor("decoder.x_embedder.weight"),
            checkpoint.get_tensor("decoder.x_embedder.bias"),
        )
        registers = checkpoint.get_tensor("decoder.register_tokens").view(
            REGISTERS, HIDDEN
        )
        hidden = torch.cat(
            (
                hidden,
                registers,
                torch.zeros((1, HIDDEN), dtype=torch.float32, device=device),
            )
        )
        cosine, sine = rope(device)
        for index in range(BLOCKS):
            prefix = f"decoder.transformer_blocks.{index}."
            normalized = rms(
                hidden, checkpoint.get_tensor(prefix + "norm1.weight")
            )
            qkv = functional.linear(
                normalized,
                checkpoint.get_tensor(prefix + "attn.to_qkv.weight"),
                checkpoint.get_tensor(prefix + "attn.to_qkv.bias"),
            )
            query, key, value = qkv_norm_rope(qkv, cosine, sine)
            heads = full_attention(query, key, value)
            branch = functional.linear(
                heads,
                checkpoint.get_tensor(prefix + "attn.to_out.weight"),
                checkpoint.get_tensor(prefix + "attn.to_out.bias"),
            )
            hidden = (
                hidden
                + branch * checkpoint.get_tensor(prefix + "scale1")
            )
            normalized = rms(
                hidden, checkpoint.get_tensor(prefix + "norm2.weight")
            )
            fused = functional.linear(
                normalized,
                checkpoint.get_tensor(prefix + "ff.w1.weight"),
                checkpoint.get_tensor(prefix + "ff.w1.bias"),
            )
            gate, up = fused.split(FEED_FORWARD, dim=-1)
            activated = functional.silu(gate) * up
            branch = functional.linear(
                activated,
                checkpoint.get_tensor(prefix + "ff.w2.weight"),
                checkpoint.get_tensor(prefix + "ff.w2.bias"),
            )
            hidden = (
                hidden
                + branch * checkpoint.get_tensor(prefix + "scale2")
            )
            print(f"VisualVAE teacher block {index + 1}/{BLOCKS}", flush=True)
        normalized = functional.layer_norm(
            hidden,
            (HIDDEN,),
            checkpoint.get_tensor("decoder.norm_out.weight"),
            checkpoint.get_tensor("decoder.norm_out.bias"),
            EPSILON,
        )
        projected = functional.linear(
            normalized,
            checkpoint.get_tensor("decoder.proj_out.weight"),
            checkpoint.get_tensor("decoder.proj_out.bias"),
        )
        frames = unpack_frames(projected)
        torch.cuda.synchronize(device)

    selected_indices = torch.tensor([0, 5, 11, 16, 21], device=device)
    selected = frames.index_select(0, selected_indices)
    tensors = {
        "latent.f32": latent,
        "frames.f32": frames,
        "selected_frames.f32": selected,
    }
    files = {
        name: write_tensor(args.output / name, tensor)
        for name, tensor in tensors.items()
    }
    metadata = {
        "schema": "mbufo.minimax-h3-video-vae-golden.v1",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "source_checkpoint": "FL2VA/video_vae/source/model.safetensors",
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_latent_sha256": sha256(args.latent),
        "teacher": {
            "framework": "torch-direct",
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "device": torch.cuda.get_device_name(device),
            "attention": "explicit-fp32-full-sdpa",
            "compute_dtype": args.compute_dtype,
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "geometry": {
            "latent": [CHANNELS, LATENT_TIME, LATENT_HEIGHT, LATENT_WIDTH],
            "frames": [FRAMES, LATENT_HEIGHT * 16, LATENT_WIDTH * 16, 3],
            "sequence": LATENT_TIME * LATENT_HEIGHT * LATENT_WIDTH + SUFFIX,
        },
        "selected_frame_indices": selected_indices.cpu().tolist(),
        "arithmetic": {
            "weights": "released F32",
            "activations": "F32",
            "attention_accumulation": "F32",
            "rgb": "ImageNet de-normalization and clamp to [0,1]",
        },
        "quality_bounds": {
            "relative_l2": 0.05,
            "relative_max": 0.05,
        },
        "files": files,
        "elapsed_seconds": time.monotonic() - started,
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
