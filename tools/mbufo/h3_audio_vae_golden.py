#!/usr/bin/env python3
"""Capture an independent MiniMax H3 AudioVAE waveform oracle."""

from __future__ import annotations

import argparse
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
CHANNELS = 32
STEREO = 2
SAMPLE_RATE = 32000
HOP_LENGTH = 800
UPSAMPLE_RATES = (5, 5, 2, 2, 2, 2, 2)
UPSAMPLE_KERNELS = (9, 9, 4, 4, 4, 4, 4)
RESIDUAL_KERNELS = (3, 7, 11)
RESIDUAL_DILATIONS = (1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
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


def read_latent(path: Path, device: torch.device) -> torch.Tensor:
    array = numpy.fromfile(path, dtype="<f4")
    plane = CHANNELS * STEREO
    if array.size == 0 or array.size % plane != 0:
        raise RuntimeError(
            f"{path} has {array.size} F32 elements, not [32,2,T]"
        )
    return (
        torch.from_numpy(array.copy())
        .reshape(CHANNELS, STEREO, array.size // plane)
        .to(device)
    )


def normalized_weight(
    checkpoint: safe_open, prefix: str
) -> torch.Tensor:
    vector = checkpoint.get_tensor(prefix + ".weight_v").float()
    magnitude = checkpoint.get_tensor(prefix + ".weight_g").float()
    rows = vector.reshape(vector.shape[0], -1)
    scale = magnitude.reshape(-1, 1) * torch.rsqrt(
        rows.square().sum(dim=1, keepdim=True)
    )
    return (rows * scale).reshape_as(vector)


def activation(
    checkpoint: safe_open, prefix: str, input_tensor: torch.Tensor
) -> torch.Tensor:
    channels = input_tensor.shape[1]
    up_filter = checkpoint.get_tensor(prefix + ".upsample.filter").float()
    down_filter = checkpoint.get_tensor(
        prefix + ".downsample.lowpass.filter"
    ).float()
    upsampled = functional.pad(input_tensor, (5, 5), mode="replicate")
    upsampled = 2.0 * functional.conv_transpose1d(
        upsampled,
        up_filter.expand(channels, -1, -1),
        stride=2,
        groups=channels,
    )
    upsampled = upsampled[..., 15:-15]
    alpha = checkpoint.get_tensor(prefix + ".act.alpha").float().exp()
    beta = checkpoint.get_tensor(prefix + ".act.beta").float().exp()
    sine = torch.sin(alpha[None, :, None] * upsampled)
    activated = upsampled + sine.square() / (
        beta[None, :, None] + 1.0e-9
    )
    activated = functional.pad(activated, (5, 6), mode="replicate")
    return functional.conv1d(
        activated,
        down_filter.expand(channels, -1, -1),
        stride=2,
        groups=channels,
    )


def conv(
    checkpoint: safe_open,
    prefix: str,
    input_tensor: torch.Tensor,
    *,
    padding: int,
    dilation: int = 1,
    bias: bool = True,
) -> torch.Tensor:
    return functional.conv1d(
        input_tensor,
        normalized_weight(checkpoint, prefix),
        checkpoint.get_tensor(prefix + ".bias").float() if bias else None,
        padding=padding,
        dilation=dilation,
    )


def residual_block(
    checkpoint: safe_open,
    global_index: int,
    kernel: int,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    hidden = input_tensor
    root = f"decoder.resblocks.{global_index}"
    for pair, dilation in enumerate(RESIDUAL_DILATIONS):
        branch = activation(
            checkpoint, f"{root}.activations.{pair * 2}", hidden
        )
        branch = conv(
            checkpoint,
            f"{root}.convs1.{pair}",
            branch,
            padding=dilation * (kernel - 1) // 2,
            dilation=dilation,
        )
        branch = activation(
            checkpoint, f"{root}.activations.{pair * 2 + 1}", branch
        )
        branch = conv(
            checkpoint,
            f"{root}.convs2.{pair}",
            branch,
            padding=(kernel - 1) // 2,
        )
        hidden = hidden + branch
    return hidden


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm torch device is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    component = args.model_root / "FL2VA" / "audio_vae"
    config = json.loads((component / "config.json").read_text())
    latent = read_latent(args.latent, device)
    latent_frames = latent.shape[-1]
    mean = torch.tensor(
        config["latents_mean"], dtype=torch.float32, device=device
    )
    deviation = torch.tensor(
        config["latents_std"], dtype=torch.float32, device=device
    )
    checkpoint_path = component / "model.safetensors"
    started = time.monotonic()
    with torch.inference_mode(), safe_open(
        checkpoint_path, framework="pt", device=args.device
    ) as checkpoint:
        hidden = latent.permute(1, 0, 2).contiguous()
        hidden = hidden * deviation[None, :, None] + mean[None, :, None]
        hidden = functional.conv1d(
            hidden,
            checkpoint.get_tensor("dec_in_proj.weight").float(),
            checkpoint.get_tensor("dec_in_proj.bias").float(),
        )
        hidden = conv(
            checkpoint,
            "decoder.conv_pre",
            hidden,
            padding=3,
        )
        for stage, (rate, kernel) in enumerate(
            zip(UPSAMPLE_RATES, UPSAMPLE_KERNELS)
        ):
            prefix = f"decoder.ups.{stage}.0"
            hidden = functional.conv_transpose1d(
                hidden,
                normalized_weight(checkpoint, prefix),
                checkpoint.get_tensor(prefix + ".bias").float(),
                stride=rate,
                padding=(kernel - rate) // 2,
            )
            branches = [
                residual_block(
                    checkpoint,
                    stage * len(RESIDUAL_KERNELS) + block,
                    residual_kernel,
                    hidden,
                )
                for block, residual_kernel in enumerate(RESIDUAL_KERNELS)
            ]
            hidden = torch.stack(branches).mean(dim=0)
            print(
                f"AudioVAE teacher stage {stage + 1}/{len(UPSAMPLE_RATES)} "
                f"shape={tuple(hidden.shape)}",
                flush=True,
            )
        hidden = activation(
            checkpoint, "decoder.activation_post", hidden
        )
        waveform = torch.clamp(
            conv(
                checkpoint,
                "decoder.conv_post",
                hidden,
                padding=3,
                bias=False,
            ),
            min=-1.0,
            max=1.0,
        )[:, 0, :]
    expected_samples = latent_frames * HOP_LENGTH
    if tuple(waveform.shape) != (STEREO, expected_samples):
        raise RuntimeError(
            f"AudioVAE returned {tuple(waveform.shape)}, expected "
            f"({STEREO}, {expected_samples})"
        )
    window = torch.hann_window(1024, dtype=torch.float32, device=device)
    spectrogram = torch.stft(
        waveform,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=window,
        center=True,
        return_complex=True,
    ).abs()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    latent_copy = output / "audio_latent.f32"
    waveform_path = output / "waveform.f32"
    spectrogram_path = output / "spectrogram.f32"
    latent_meta = write_tensor(latent_copy, latent)
    waveform_meta = write_tensor(waveform_path, waveform)
    spectrogram_meta = write_tensor(spectrogram_path, spectrogram)
    files = {
        latent_copy.name: latent_meta,
        waveform_path.name: waveform_meta,
        spectrogram_path.name: spectrogram_meta,
    }
    contract = {
        "schema": "mbufo.minimax-h3-audio-vae-golden.v1",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "source_checkpoint": "FL2VA/audio_vae/model.safetensors",
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_latent_sha256": sha256(args.latent),
        "latent_frames": latent_frames,
        "samples": expected_samples,
        "spectrogram_frames": spectrogram.shape[-1],
        "channels": STEREO,
        "sample_rate": SAMPLE_RATE,
        "quality_bounds": {
            "maximum_error": 1.0e-3,
            "relative_l2": 0.05,
            "spectrogram_relative_l2": 0.08,
            "spectrogram_relative_max": 0.1,
        },
        "files": files,
        "teacher": {
            "implementation": "direct torch functional equations",
            "elapsed_seconds": time.monotonic() - started,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "device": torch.cuda.get_device_name(device),
            "tf32": False,
            "deterministic_algorithms": True,
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"wrote AudioVAE oracle to {output} in "
        f"{contract['teacher']['elapsed_seconds']:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
