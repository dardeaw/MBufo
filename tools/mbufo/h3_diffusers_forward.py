#!/usr/bin/env python3
"""Compare one pinned Diffusers MiniMax-H3 forward with a frozen teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

# Executing a script inside tools/mbufo places that directory first on
# sys.path, where the project's helper safetensors.py would shadow the
# third-party package imported by Diffusers.
_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != _SCRIPT_DIRECTORY
]

import numpy
import torch
from diffusers import MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted-root", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video-l2-bound", type=float, default=0.04)
    parser.add_argument("--video-max-bound", type=float, default=0.05)
    parser.add_argument("--audio-l2-bound", type=float, default=0.05)
    parser.add_argument("--audio-max-bound", type=float, default=0.05)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def read_f32(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    require_file(path)
    values = numpy.fromfile(path, dtype="<f4")
    if values.size != math.prod(shape):
        raise ValueError(f"{path} has {values.size} values, expected {shape}")
    return torch.from_numpy(values.copy()).reshape(shape)


def read_bf16(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    require_file(path)
    values = numpy.fromfile(path, dtype="<u2")
    if values.size != math.prod(shape):
        raise ValueError(f"{path} has {values.size} values, expected {shape}")
    return torch.from_numpy(values.copy()).view(torch.bfloat16).reshape(shape)


def unpatch_video(
    rows: torch.Tensor, channels: int, frames: int, height: int, width: int
) -> torch.Tensor:
    return (
        rows.reshape(1, frames, height // 2, width // 2, channels, 1, 2, 2)
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(channels, frames, height, width)
        .contiguous()
    )


def unpack_audio(rows: torch.Tensor, channels: int, frames: int) -> torch.Tensor:
    return (
        rows.reshape(2, frames, channels)
        .permute(2, 0, 1)
        .contiguous()
    )


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    delta = actual - expected
    relative_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(
        expected
    ).clamp_min(1.0e-12)
    relative_max = delta.abs().max() / expected.abs().max().clamp_min(1.0e-12)
    return {
        "relative_l2": float(relative_l2),
        "relative_max": float(relative_max),
    }


def sha256_f32(tensor: torch.Tensor) -> str:
    payload = tensor.detach().float().contiguous().cpu().numpy().astype(
        "<f4", copy=False
    )
    return hashlib.sha256(memoryview(payload).tobytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm torch device is unavailable")
    metadata = json.loads((args.golden / "metadata.json").read_text())
    if metadata.get("model_revision") != (
        "42ed227ee7df40d41602854ae760620d6eb651fe"
    ):
        raise ValueError("golden model revision is not the pinned MiniMax H3")
    if not metadata.get("forward_only") or metadata.get("executed_forwards") != 1:
        raise ValueError("golden must contain exactly one transformer forward")
    geometry = metadata["geometry"]
    text_rows = int(geometry["text_rows"])
    frames = int(geometry["video_latent_frames"])
    latent_height = int(geometry["latent_height"])
    latent_width = int(geometry["latent_width"])
    audio_frames = int(geometry["audio_latent_frames"])
    steps = int(metadata["steps"])
    device = torch.device(args.device)

    conditioning = read_bf16(
        args.golden / "conditioning.bf16", (text_rows, 5120)
    )
    video = read_f32(
        args.golden / "video_initial.f32",
        (24, frames, latent_height, latent_width),
    )
    audio = read_f32(
        args.golden / "audio_initial.f32", (32, 2, audio_frames)
    )
    expected_video = read_f32(
        args.golden / "video_velocity_step0.f32",
        (24, frames, latent_height, latent_width),
    )
    expected_audio = read_f32(
        args.golden / "audio_velocity_step0.f32", (32, 2, audio_frames)
    )

    model = MiniMaxH3Transformer3DModel.from_pretrained(
        args.converted_root / "transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    ).eval()
    dtypes = {
        "video_input": str(model.proj_in.weight.dtype),
        "audio_input": str(model.audio_proj_in.weight.dtype),
        "text_input": str(model.context_embedder.weight.dtype),
        "timestep": str(model.time_embedder.linear_1.weight.dtype),
        "block": str(model.transformer_blocks[0].norm1.weight.dtype),
        "video_output": str(model.proj_out.weight.dtype),
        "audio_output": str(model.audio_proj_out.weight.dtype),
    }
    expected_dtypes = {
        "video_input": "torch.float32",
        "audio_input": "torch.float32",
        "text_input": "torch.bfloat16",
        "timestep": "torch.float32",
        "block": "torch.bfloat16",
        "video_output": "torch.float32",
        "audio_output": "torch.float32",
    }
    if dtypes != expected_dtypes:
        raise RuntimeError(f"unexpected converted checkpoint dtypes: {dtypes}")

    (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        _,
        _,
    ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.ones(text_rows, dtype=torch.long),
        frames,
        latent_height,
        latent_width,
        audio_frames,
        (1, 2, 2),
        2,
        2,
        0,
    )
    video_rows = patchify_video_latents(video.unsqueeze(0), (1, 2, 2))
    audio_rows = audio.permute(1, 2, 0).reshape(2 * audio_frames, 32)

    video_scheduler = MiniMaxH3Scheduler(shift=12.0)
    audio_scheduler = MiniMaxH3Scheduler(shift=3.0)
    video_scheduler.set_timesteps(steps + 1, device="cpu")
    audio_scheduler.set_timesteps(steps + 1, device="cpu")
    timestep, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices,
        audio_indices,
        0,
        0,
        text_rows,
        float(video_scheduler.timesteps[0]),
        float(audio_scheduler.timesteps[0]),
        float(video_scheduler.timesteps[0]),
        1.0,
    )

    with torch.inference_mode():
        video_output, audio_output = model(
            hidden_states=video_rows.unsqueeze(0).to(device),
            audio_hidden_states=audio_rows.unsqueeze(0).to(device),
            encoder_hidden_states=conditioning.unsqueeze(0).to(device),
            timestep=timestep.to(device),
            timestep_indices=timestep_indices.to(device),
            token_tags=token_tags.to(device),
            position_ids=position_ids.to(device),
            video_indices=video_indices.to(device),
            audio_indices=audio_indices.to(device),
            text_indices=text_indices.to(device),
            return_dict=False,
        )
        torch.cuda.synchronize(device)

    actual_video = unpatch_video(
        video_output[0], 24, frames, latent_height, latent_width
    )
    actual_audio = unpack_audio(audio_output[0], 32, audio_frames)
    video_metrics = compare(actual_video, expected_video)
    audio_metrics = compare(actual_audio, expected_audio)
    report = {
        "schema": "mbufo.minimax-h3-diffusers-forward-parity.v1",
        "model_revision": metadata["model_revision"],
        "reference_revision": metadata["reference_revision"],
        "diffusers_revision": "fe15005a333d1270b490e4885a6ea13b66b1092a",
        "geometry": geometry,
        "dtypes": dtypes,
        "bounds": {
            "video_relative_l2": args.video_l2_bound,
            "video_relative_max": args.video_max_bound,
            "audio_relative_l2": args.audio_l2_bound,
            "audio_relative_max": args.audio_max_bound,
        },
        "video": {
            **video_metrics,
            "sha256": sha256_f32(actual_video),
        },
        "audio": {
            **audio_metrics,
            "sha256": sha256_f32(actual_audio),
        },
    }
    passed = (
        video_metrics["relative_l2"] < args.video_l2_bound
        and video_metrics["relative_max"] < args.video_max_bound
        and audio_metrics["relative_l2"] < args.audio_l2_bound
        and audio_metrics["relative_max"] < args.audio_max_bound
    )
    report["passed"] = passed
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
