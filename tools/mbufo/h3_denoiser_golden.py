#!/usr/bin/env python3
"""Capture an independent MiniMax H3 BF16 denoiser oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from h3_rng import NormalRng
import h3_dit_golden as block


torch = block.torch
functional = block.functional
numpy = block.numpy

MODEL_REVISION = block.MODEL_REVISION
REFERENCE_REVISION = block.REFERENCE_REVISION
HIDDEN = block.HIDDEN
HEADS = block.HEADS
HEAD_DIMENSION = block.HEAD_DIMENSION
INNER = block.INNER
FEED_FORWARD = block.FEED_FORWARD
EPSILON = block.EPSILON
TIME_INPUT = 256
TIME_HIDDEN = 5376
TIME_OUTPUT = 2688
MODALITIES = 3
SLOTS = 6
VIDEO_PATCH = 96
AUDIO_WIDTH = 32
TEXT_WIDTH = 5120
BLOCKS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help=(
            "execute and retain only the first complete transformer forward; "
            "do not advance the denoising schedule"
        ),
    )
    parser.add_argument(
        "--text-rows",
        type=int,
        default=0,
        help="conditioning rows; zero infers them from the BF16 file",
    )
    parser.add_argument(
        "--noise-mode",
        choices=("synthetic", "seeded"),
        default="synthetic",
        help="synthetic preserves the component oracle; seeded matches serving",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_tensor(path: Path, tensor: torch.Tensor, dtype: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = tensor.detach().contiguous().cpu()
    if dtype == "BF16":
        array = (
            contiguous.to(torch.bfloat16)
            .view(torch.uint16)
            .numpy()
            .astype("<u2", copy=False)
        )
    elif dtype == "F32":
        array = contiguous.float().numpy().astype("<f4", copy=False)
    elif dtype == "U32":
        array = contiguous.to(torch.int64).numpy().astype("<u4", copy=False)
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
    path.write_bytes(memoryview(numpy.ascontiguousarray(array)).tobytes())
    return {
        "dtype": dtype,
        "shape": list(tensor.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_bf16(path: Path, shape: tuple[int, ...], device: torch.device):
    array = numpy.fromfile(path, dtype="<u2")
    expected = math.prod(shape)
    if array.size != expected:
        raise RuntimeError(
            f"{path} has {array.size} BF16 elements, expected {expected}"
        )
    tensor = torch.from_numpy(array.copy()).view(torch.bfloat16).reshape(shape)
    return tensor.to(device)


def rms(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    values = input_tensor.float()
    inverse = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + EPSILON)
    return (values * inverse * weight.float()).to(torch.bfloat16)


def qkv_norm_plain(
    qkv: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = qkv.shape[0]
    grouped = qkv.view(rows, HEADS, 3, HEAD_DIMENSION)
    query = grouped[:, :, 0, :].float()
    key = grouped[:, :, 1, :].float()
    value = grouped[:, :, 2, :].contiguous()
    query *= torch.rsqrt(
        query.square().mean(dim=-1, keepdim=True) + EPSILON
    )
    key *= torch.rsqrt(key.square().mean(dim=-1, keepdim=True) + EPSILON)
    return (
        (query * query_weight.float()).to(torch.bfloat16),
        (key * key_weight.float()).to(torch.bfloat16),
        value,
    )


def refine_text(
    checkpoint: block.Checkpoint, conditioning: torch.Tensor
) -> torch.Tensor:
    hidden = functional.linear(
        conditioning,
        checkpoint.tensor("condition_proj.weight"),
        checkpoint.tensor("condition_proj.bias"),
    )
    for index in range(2):
        prefix = f"token_refiner.blocks.{index}."
        normalized = rms(hidden, checkpoint.tensor(prefix + "norm1.weight"))
        qkv = functional.linear(
            normalized, checkpoint.tensor(prefix + "attn.qkv_proj.weight")
        )
        query, key, value = qkv_norm_plain(
            qkv,
            checkpoint.tensor(prefix + "attn.q_norm.weight"),
            checkpoint.tensor(prefix + "attn.k_norm.weight"),
        )
        heads = block.full_attention(query, key, value)
        branch = functional.linear(
            heads, checkpoint.tensor(prefix + "attn.out_proj.weight")
        )
        hidden = (hidden.float() + branch.float()).to(torch.bfloat16)
        normalized = rms(hidden, checkpoint.tensor(prefix + "norm2.weight"))
        fc1 = functional.linear(
            normalized, checkpoint.tensor(prefix + "mlp.fc1.weight")
        )
        gate, up = fc1.split(FEED_FORWARD, dim=-1)
        activated = (functional.silu(gate.float()) * up.float()).to(
            torch.bfloat16
        )
        branch = functional.linear(
            activated, checkpoint.tensor(prefix + "mlp.fc2.weight")
        )
        hidden = (hidden.float() + branch.float()).to(torch.bfloat16)
    return rms(hidden, checkpoint.tensor("token_refiner.final_norm.weight"))


def schedules(steps: int):
    if steps < 2:
        raise ValueError("steps must be at least two")
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    video = 12.0 * base / (1.0 + 11.0 * base)
    audio = 3.0 * base / (1.0 + 2.0 * base)
    video[-1] = 0.0
    audio[-1] = 0.0
    video_rows: list[int] = []
    audio_rows: list[int] = []
    count = 0
    for step in range(steps):
        video_time = float(1.0 - video[step])
        audio_time = float(1.0 - audio[step])
        if video_time == audio_time:
            video_rows.append(count)
            audio_rows.append(count)
            count += 1
        elif video_time < audio_time:
            video_rows.append(count)
            audio_rows.append(count + 1)
            count += 2
        else:
            audio_rows.append(count)
            video_rows.append(count + 1)
            count += 2
    times = torch.zeros(count, dtype=torch.float32)
    for step in range(steps):
        times[video_rows[step]] = 1.0 - video[step]
        times[audio_rows[step]] = 1.0 - audio[step]
    return video, audio, video_rows, audio_rows, times


def time_embedding(
    checkpoint: block.Checkpoint, times: torch.Tensor, device: torch.device
):
    indices = torch.arange(TIME_INPUT // 2, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10000.0) * indices / float(TIME_INPUT // 2)
    )
    angles = times[:, None] * frequencies[None, :]
    features = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1).to(
        device
    )
    hidden = functional.linear(
        features,
        checkpoint.tensor("time_embedder.proj_in.weight"),
        checkpoint.tensor("time_embedder.proj_in.bias"),
    )
    hidden = functional.silu(hidden)
    projected = functional.linear(
        hidden,
        checkpoint.tensor("time_embedder.proj_out.weight"),
        checkpoint.tensor("time_embedder.proj_out.bias"),
    )
    # The timestep MLP is F32. AdaLN applies SiLU before casting its input to
    # the BF16 modulation projection; casting first coherently biases every
    # block at every denoising evaluation.
    return functional.silu(projected).to(torch.bfloat16)


def modulation(
    checkpoint: block.Checkpoint,
    time: torch.Tensor,
    prefix: str,
    modalities: int,
):
    projected = functional.linear(
        time,
        checkpoint.tensor(prefix + ".linear.weight"),
        checkpoint.tensor(prefix + ".linear.bias"),
    )
    return projected.view(time.shape[0] * modalities, -1, HIDDEN)


class CachedCheckpoint:
    """Caches only the reusable DiT phase after streamed AdaLN precompute."""

    def __init__(self, checkpoint: block.Checkpoint):
        self.checkpoint = checkpoint
        self.enabled = False
        self.cache: dict[str, torch.Tensor] = {}

    def tensor(self, name: str) -> torch.Tensor:
        if not self.enabled:
            return self.checkpoint.tensor(name)
        if name not in self.cache:
            self.cache[name] = self.checkpoint.tensor(name)
        return self.cache[name]


def geometry(
    width: int, height: int, frames: int, text_rows: int = 6
) -> dict[str, int]:
    if (
        width < 32
        or height < 32
        or width % 32 != 0
        or height % 32 != 0
        or frames < 5
        or (frames - 5) % 17 != 0
        or text_rows < 1
    ):
        raise ValueError("invalid MiniMax H3 teacher geometry")
    video_latent_frames = ((frames - 5) // 17) * 5 + 2
    audio_latent_frames = math.floor(frames * 40.0 / 24.0 + 0.5)
    latent_height = height // 16
    latent_width = width // 16
    if latent_height % 2 != 0 or latent_width % 2 != 0:
        raise ValueError("MiniMax H3 teacher latent axes must be even")
    video_rows = (
        video_latent_frames * (latent_height // 2) * (latent_width // 2)
    )
    audio_rows = 2 * audio_latent_frames
    return {
        "width": width,
        "height": height,
        "frames": frames,
        "text_rows": text_rows,
        "video_latent_frames": video_latent_frames,
        "audio_latent_frames": audio_latent_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "video_rows": video_rows,
        "audio_rows": audio_rows,
        "rows": text_rows + audio_rows + video_rows,
    }


def production_positions(spec: dict[str, int]) -> torch.Tensor:
    frame_rows = spec["latent_height"] // 2
    frame_columns = spec["latent_width"] // 2
    positions: list[tuple[float, float, float]] = [
        (float(row), 0.0, 0.0) for row in range(spec["text_rows"])
    ]
    sqrt_area = math.sqrt(spec["latent_height"] * spec["latent_width"])
    height_ratio = spec["latent_height"] / sqrt_area
    width_ratio = spec["latent_width"] / sqrt_area
    height_left = (1.0 - height_ratio) / 2.0
    width_left = (1.0 - width_ratio) / 2.0
    spatial = [
        (
            (height_left + row * height_ratio / frame_rows) * 32.0,
            (width_left + column * width_ratio / frame_columns) * 32.0,
        )
        for row in range(frame_rows)
        for column in range(frame_columns)
    ]
    cursor = float(spec["text_rows"])
    low_width = spatial[0][1]
    high_width = spatial[frame_columns - 1][1]
    for index in range(spec["audio_latent_frames"]):
        positions.append((cursor + index, 0.0, low_width))
    for index in range(spec["audio_latent_frames"]):
        positions.append((cursor + index, 0.0, high_width))
    frames_per_token = (1, 4, 4, 4, 4)
    temporal = cursor
    for index in range(spec["video_latent_frames"]):
        for height, width in spatial:
            positions.append((temporal, height, width))
        temporal += (5.0 / 3.0) * frames_per_token[index % 5]
    result = torch.tensor(positions, dtype=torch.float32)
    if tuple(result.shape) != (spec["rows"], 3):
        raise RuntimeError(f"unexpected production layout: {result.shape}")
    return result


def row_map(
    step: int,
    video_time_rows: list[int],
    audio_time_rows: list[int],
    spec: dict[str, int],
    device: torch.device,
):
    text_end = spec["text_rows"]
    audio_end = text_end + spec["audio_rows"]
    result = torch.empty(spec["rows"], dtype=torch.int64, device=device)
    result[:text_end] = video_time_rows[step] * MODALITIES + 1
    result[text_end:audio_end] = audio_time_rows[step] * MODALITIES + 2
    result[audio_end:] = video_time_rows[step] * MODALITIES
    return result


def run_block(
    checkpoint: block.Checkpoint,
    index: int,
    hidden: torch.Tensor,
    current_modulation: torch.Tensor,
    current_row_map: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    prefix = f"blocks.{index}."
    normalized = block.rms_adaln(
        hidden,
        checkpoint.tensor(prefix + "norm1.weight"),
        current_modulation,
        current_row_map,
        0,
        1,
    )
    qkv = functional.linear(
        normalized, checkpoint.tensor(prefix + "attn.qkv_proj.weight")
    )
    query, key, value = block.qkv_norm_rope(
        qkv,
        checkpoint.tensor(prefix + "attn.q_norm.weight"),
        checkpoint.tensor(prefix + "attn.k_norm.weight"),
        rope_cos,
        rope_sin,
    )
    heads = block.full_attention(query, key, value)
    attention = functional.linear(
        heads, checkpoint.tensor(prefix + "attn.out_proj.weight")
    )
    hidden = block.gated_residual(
        hidden, attention, current_modulation, current_row_map, 2
    )
    normalized = block.rms_adaln(
        hidden,
        checkpoint.tensor(prefix + "norm2.weight"),
        current_modulation,
        current_row_map,
        3,
        4,
    )
    fc1 = functional.linear(
        normalized, checkpoint.tensor(prefix + "mlp.fc1.weight")
    )
    gate, up = fc1.split(FEED_FORWARD, dim=-1)
    activated = (functional.silu(gate.float()) * up.float()).to(
        torch.bfloat16
    )
    mlp = functional.linear(
        activated, checkpoint.tensor(prefix + "mlp.fc2.weight")
    )
    return block.gated_residual(
        hidden, mlp, current_modulation, current_row_map, 5
    )


def initial_latents(
    device: torch.device,
    spec: dict[str, int],
    *,
    mode: str,
    seed: int,
):
    video_count = (
        24
        * spec["video_latent_frames"]
        * spec["latent_height"]
        * spec["latent_width"]
    )
    audio_count = 32 * 2 * spec["audio_latent_frames"]
    if mode == "seeded":
        rng = NormalRng(seed)
        video = torch.from_numpy(
            numpy.asarray(rng.fill(video_count), dtype="<f4")
        ).to(device)
        audio_rows = torch.from_numpy(
            numpy.asarray(rng.fill(audio_count), dtype="<f4")
        ).to(device)
        audio = (
            audio_rows.view(2, spec["audio_latent_frames"], AUDIO_WIDTH)
            .permute(2, 0, 1)
            .contiguous()
            .view(-1)
        )
        return video, audio
    video_index = torch.arange(video_count, dtype=torch.float32, device=device)
    audio_index = torch.arange(audio_count, dtype=torch.float32, device=device)
    video = (
        torch.sin(video_index * 0.013) * 0.7
        + torch.cos(video_index * 0.0017) * 0.2
    )
    audio = (
        torch.cos(audio_index * 0.017) * 0.6
        + torch.sin(audio_index * 0.0023) * 0.15
    )
    return video, audio


def patch_video(video: torch.Tensor, spec: dict[str, int]) -> torch.Tensor:
    frame_rows = spec["latent_height"] // 2
    frame_columns = spec["latent_width"] // 2
    return (
        video.view(
            24,
            spec["video_latent_frames"],
            frame_rows,
            2,
            frame_columns,
            2,
        )
        .permute(1, 2, 4, 0, 3, 5)
        .contiguous()
        .view(spec["video_rows"], VIDEO_PATCH)
    )


def unpatch_video(rows: torch.Tensor, spec: dict[str, int]) -> torch.Tensor:
    frame_rows = spec["latent_height"] // 2
    frame_columns = spec["latent_width"] // 2
    return (
        rows.view(
            spec["video_latent_frames"],
            frame_rows,
            frame_columns,
            24,
            2,
            2,
        )
        .permute(3, 0, 1, 4, 2, 5)
        .contiguous()
        .view(-1)
    )


def pack_audio(audio: torch.Tensor, spec: dict[str, int]) -> torch.Tensor:
    return (
        audio.view(AUDIO_WIDTH, 2, spec["audio_latent_frames"])
        .permute(1, 2, 0)
        .contiguous()
        .view(spec["audio_rows"], AUDIO_WIDTH)
    )


def unpack_audio(rows: torch.Tensor, spec: dict[str, int]) -> torch.Tensor:
    return (
        rows.view(2, spec["audio_latent_frames"], AUDIO_WIDTH)
        .permute(2, 0, 1)
        .contiguous()
        .view(-1)
    )


def main() -> int:
    args = parse_args()
    if not args.forward_only:
        raise ValueError(
            "MiniMax H3 development teachers require --forward-only; "
            "multi-step model execution is intentionally disabled"
        )
    if args.steps < 2 or args.seed < 0 or args.seed > (1 << 64) - 1:
        raise ValueError("invalid MiniMax H3 teacher steps or seed")
    conditioning_bytes = args.conditioning.stat().st_size
    row_bytes = TEXT_WIDTH * 2
    if conditioning_bytes == 0 or conditioning_bytes % row_bytes != 0:
        raise ValueError("conditioning is not a non-empty [tokens,5120] BF16")
    inferred_rows = conditioning_bytes // row_bytes
    text_rows = args.text_rows or inferred_rows
    if text_rows != inferred_rows:
        raise ValueError("conditioning row count differs from --text-rows")
    spec = geometry(args.width, args.height, args.frames, text_rows)
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm torch device is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    checkpoint = CachedCheckpoint(
        block.Checkpoint(
            args.model_root / "FL2VA" / "transformer", args.device
        )
    )
    started = time.monotonic()
    conditioning = read_bf16(
        args.conditioning, (spec["text_rows"], TEXT_WIDTH), device
    )
    with torch.inference_mode():
        refined = refine_text(checkpoint, conditioning)
        positions = production_positions(spec).to(device)
        inverse = checkpoint.tensor("rope.inv_freq").float()
        angles = positions[:, :, None] * inverse[None, None, :]
        rope_cos = (
            torch.cos(angles).reshape(spec["rows"], 48).to(torch.bfloat16)
        )
        rope_sin = (
            torch.sin(angles).reshape(spec["rows"], 48).to(torch.bfloat16)
        )
        video_sigma, audio_sigma, video_time_rows, audio_time_rows, times = (
            schedules(args.steps)
        )
        embedded_time = time_embedding(checkpoint, times, device)
        block_modulations = []
        for index in range(BLOCKS):
            block_modulations.append(
                modulation(
                    checkpoint,
                    embedded_time,
                    f"blocks.{index}.adaln_proj",
                    MODALITIES,
                )
            )
        final_modulation = modulation(
            checkpoint,
            embedded_time,
            "final_layer.adaln_proj",
            1,
        )
        checkpoint.enabled = True

        video_initial, audio_initial = initial_latents(
            device, spec, mode=args.noise_mode, seed=args.seed
        )
        video_rows = patch_video(video_initial, spec)
        audio_rows = pack_audio(audio_initial, spec)
        first_video_velocity = None
        first_audio_velocity = None
        text_end = spec["text_rows"]
        audio_end = text_end + spec["audio_rows"]
        executed_forwards = 1
        for step in range(executed_forwards):
            step_started = time.monotonic()
            print(
                f"H3 denoiser teacher forward {step + 1}/{executed_forwards} "
                f"rows={spec['rows']}",
                flush=True,
            )
            hidden = torch.cat(
                (
                    refined,
                    functional.linear(
                        audio_rows,
                        checkpoint.tensor("audio_patch_proj.weight"),
                        checkpoint.tensor("audio_patch_proj.bias"),
                    ).to(torch.bfloat16),
                    functional.linear(
                        video_rows,
                        checkpoint.tensor("video_patch_proj.weight"),
                        checkpoint.tensor("video_patch_proj.bias"),
                    ).to(torch.bfloat16),
                ),
                dim=0,
            )
            current_map = row_map(
                step, video_time_rows, audio_time_rows, spec, device
            )
            for index in range(BLOCKS):
                hidden = run_block(
                    checkpoint,
                    index,
                    hidden,
                    block_modulations[index],
                    current_map,
                    rope_cos,
                    rope_sin,
                )
            final_weight = checkpoint.tensor("final_layer.norm.weight")
            audio_map = torch.full(
                (spec["audio_rows"],),
                audio_time_rows[step],
                dtype=torch.int64,
                device=device,
            )
            video_map = torch.full(
                (spec["video_rows"],),
                video_time_rows[step],
                dtype=torch.int64,
                device=device,
            )
            audio_norm = block.rms_adaln(
                hidden[text_end:audio_end],
                final_weight,
                final_modulation,
                audio_map,
                0,
                1,
            )
            video_norm = block.rms_adaln(
                hidden[audio_end:],
                final_weight,
                final_modulation,
                video_map,
                0,
                1,
            )
            audio_velocity = functional.linear(
                audio_norm.float(),
                checkpoint.tensor("final_layer.audio_out.weight"),
                checkpoint.tensor("final_layer.audio_out.bias"),
            )
            video_velocity = functional.linear(
                video_norm.float(),
                checkpoint.tensor("final_layer.video_out.weight"),
                checkpoint.tensor("final_layer.video_out.bias"),
            )
            if step == 0:
                first_video_velocity = video_velocity.clone()
                first_audio_velocity = audio_velocity.clone()
            print(
                f"H3 denoiser teacher forward {step + 1}/{executed_forwards} "
                f"completed in {time.monotonic() - step_started:.3f}s",
                flush=True,
            )
        torch.cuda.synchronize(device)

    tensors = {
        "conditioning.bf16": (conditioning, "BF16"),
        "refined_text.bf16": (refined, "BF16"),
        "block0_modulation.bf16": (block_modulations[0], "BF16"),
        "video_initial.f32": (video_initial, "F32"),
        "audio_initial.f32": (audio_initial, "F32"),
        "video_velocity_step0.f32": (
            unpatch_video(first_video_velocity, spec),
            "F32",
        ),
        "audio_velocity_step0.f32": (
            unpack_audio(first_audio_velocity, spec),
            "F32",
        ),
        "row_map_step0.u32": (
            row_map(0, video_time_rows, audio_time_rows, spec, device),
            "U32",
        ),
    }
    files = {
        name: write_tensor(args.output / name, tensor, dtype)
        for name, (tensor, dtype) in tensors.items()
    }
    metadata = {
        "schema": "mbufo.minimax-h3-denoiser-golden.v1",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "teacher": {
            "framework": "torch-direct",
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "device": torch.cuda.get_device_name(device),
            "attention": "explicit-fp32-full-sdpa",
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "geometry": {
            **spec,
        },
        "steps": args.steps,
        "executed_forwards": executed_forwards,
        "forward_only": True,
        "blocks": BLOCKS,
        "reuse_interval": 1,
        "spatial_rope_scale": 1.0,
        "seed": args.seed if args.noise_mode == "seeded" else None,
        "noise_mode": args.noise_mode,
        "time_rows": len(times),
        "arithmetic": {
            "weights": "released BF16/F32",
            "activations": "BF16 at transformer operation boundaries",
            "attention_accumulation": "FP32",
            "final_head_accumulation": "FP32",
            "euler_velocity": "FP32",
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
