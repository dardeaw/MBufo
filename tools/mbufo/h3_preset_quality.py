#!/usr/bin/env python3
"""Compare delivered MiniMax H3 MP4 presets against the BF16 exact route."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from mbufo.h3_quality import H3QualityError, audio_metrics, frame_metrics


SCHEMA = "mbufo.minimax-h3-preset-quality.v1"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
REFERENCE_REVISION = "8974cc055ea9c02fcd14cc27dfda3e1027c05153"
LEGACY_SCALAR_BINARY_SHA256 = (
    "35cb1e285e9d6263cee9c4d71afb9361afb4813208e023b42adad0c83a952211"
)
PRESET_CONTRACTS = {
    "exact-512": {
        "internal_width": 512,
        "internal_height": 512,
        "evaluations": 49,
        "active_blocks": 50,
        "reuse_interval": 1,
    },
    "fast-384": {
        "internal_width": 384,
        "internal_height": 384,
        "evaluations": 19,
        "active_blocks": 45,
        "reuse_interval": 2,
    },
    "aggressive-320": {
        "internal_width": 320,
        "internal_height": 320,
        "evaluations": 19,
        "active_blocks": 40,
        "reuse_interval": 3,
    },
}
ALEXNET_WEIGHTS_FILENAME = "alexnet-owt-7be5be79.pth"
ALEXNET_WEIGHTS_SHA256 = (
    "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
)
LPIPS_ALEX_V01_SHA256 = (
    "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_weight_file(
    path: Path, expected_sha256: str, description: str
) -> str:
    if path.is_symlink():
        resolved = path.resolve(strict=True)
    else:
        resolved = path
    if not resolved.is_file():
        raise H3QualityError(f"{description} is missing")
    digest = sha256_file(resolved)
    if digest != expected_sha256:
        raise H3QualityError(f"{description} hash differs")
    return digest


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(
        f"{path.name}.partial-{os.getpid()}-{time.monotonic_ns()}"
    )
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def run_bytes(command: list[str]) -> bytes:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace").strip()
        raise H3QualityError(
            f"media command failed with status {completed.returncode}: "
            f"{diagnostic[:1000]}"
        )
    return completed.stdout


def probe_media(path: Path, ffprobe: str) -> dict[str, Any]:
    encoded = run_bytes(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration,size:"
                "stream=index,codec_type,codec_name,width,height,r_frame_rate,"
                "sample_rate,channels,duration"
                ",start_time"
            ),
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        root = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise H3QualityError("FFprobe returned invalid JSON") from exc
    streams = root.get("streams")
    if not isinstance(streams, list):
        raise H3QualityError("FFprobe response lacks streams")
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video) != 1 or len(audio) != 1:
        raise H3QualityError(
            "quality input must contain one video and one audio stream"
        )
    return {
        "video": video[0],
        "audio": audio[0],
        "format": root.get("format", {}),
    }


def integer_field(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise H3QualityError(f"invalid {name}") from exc
    if parsed <= 0:
        raise H3QualityError(f"invalid {name}")
    return parsed


def float_field(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise H3QualityError(f"invalid {name}") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise H3QualityError(f"invalid {name}")
    return parsed


def signed_float_field(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise H3QualityError(f"invalid {name}") from exc
    if not np.isfinite(parsed):
        raise H3QualityError(f"invalid {name}")
    return parsed


def frame_rate(value: Any) -> Fraction:
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise H3QualityError("invalid video frame rate") from exc
    if rate <= 0:
        raise H3QualityError("invalid video frame rate")
    return rate


def sibling_parameters(path: Path) -> dict[str, Any] | None:
    parameters_path = path.with_name("parameters.json")
    if not parameters_path.is_file():
        return None
    try:
        document = json.loads(parameters_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise H3QualityError("sibling parameter report is invalid JSON") from exc
    if (
        not isinstance(document, dict)
        or document.get("schema") != "mbufo.minimax-h3-text-generation.v1"
    ):
        raise H3QualityError("sibling parameter report has an invalid schema")
    return {
        "sha256": sha256_file(parameters_path),
        "document": document,
    }


def sibling_profile(
    path: Path, parameters: dict[str, Any]
) -> dict[str, Any] | None:
    profile_path = path.with_name("profile.json")
    if not profile_path.is_file():
        return None
    try:
        document = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise H3QualityError("sibling profile is invalid JSON") from exc
    if (
        not isinstance(document, dict)
        or document.get("schema") != "mbufo.minimax-h3-profile-run.v1"
    ):
        raise H3QualityError("sibling profile has an invalid schema")
    output = document.get("output")
    telemetry = document.get("telemetry")
    binary_sha256 = document.get("binary_sha256")
    if (
        document.get("parameters") != parameters
        or not isinstance(output, dict)
        or output.get("sha256") != sha256_file(path)
    ):
        raise H3QualityError("sibling profile does not bind the delivered file")
    if (
        not isinstance(telemetry, dict)
        or not isinstance(binary_sha256, str)
        or len(binary_sha256) != 64
        or any(character not in "0123456789abcdef" for character in binary_sha256)
    ):
        raise H3QualityError("sibling profile provenance is invalid")
    kernel = telemetry.get("denoiser_attention_kernel")
    if not isinstance(kernel, str):
        kernel = parameters.get("attention_kernel")
    provenance = "profile"
    if not isinstance(kernel, str):
        if binary_sha256 != LEGACY_SCALAR_BINARY_SHA256:
            raise H3QualityError(
                "sibling profile lacks verifiable attention-kernel provenance"
            )
        kernel = "scalar"
        provenance = "legacy-scalar-binary"
    if kernel not in ("scalar", "row_parallel"):
        raise H3QualityError("sibling profile has invalid attention kernel")
    return {
        "sha256": sha256_file(profile_path),
        "binary_sha256": binary_sha256,
        "attention_kernel": kernel,
        "attention_kernel_provenance": provenance,
    }


def extract_frames(
    path: Path, probe: dict[str, Any], ffmpeg: str
) -> tuple[np.ndarray, Fraction]:
    video = probe["video"]
    width = integer_field(video.get("width"), "video width")
    height = integer_field(video.get("height"), "video height")
    rate = frame_rate(video.get("r_frame_rate"))
    payload = run_bytes(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    frame_bytes = width * height * 3
    if not payload or len(payload) % frame_bytes != 0:
        raise H3QualityError("decoded RGB payload has invalid geometry")
    frames = np.frombuffer(payload, dtype=np.uint8).reshape(
        (-1, height, width, 3)
    )
    return frames.astype(np.float32) / 255.0, rate


def extract_audio(
    path: Path, probe: dict[str, Any], ffmpeg: str
) -> tuple[np.ndarray, int]:
    audio = probe["audio"]
    sample_rate = integer_field(audio.get("sample_rate"), "audio sample rate")
    channels = integer_field(audio.get("channels"), "audio channels")
    payload = run_bytes(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "pipe:1",
        ]
    )
    values = np.frombuffer(payload, dtype="<f4")
    if not values.size or values.size % channels != 0:
        raise H3QualityError("decoded PCM payload has invalid geometry")
    return values.reshape((-1, channels)).T.copy(), sample_rate


def media_summary(
    probe: dict[str, Any],
    frames: np.ndarray,
    rate: Fraction,
    waveform: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    video_duration = frames.shape[0] / float(rate)
    audio_duration = waveform.shape[1] / sample_rate
    encoded_video_duration = float_field(
        probe["video"].get("duration"), "encoded video duration"
    )
    encoded_audio_duration = float_field(
        probe["audio"].get("duration"), "encoded audio duration"
    )
    video_start = signed_float_field(
        probe["video"].get("start_time"), "video start time"
    )
    audio_start = signed_float_field(
        probe["audio"].get("start_time"), "audio start time"
    )
    return {
        "video_codec": probe["video"].get("codec_name"),
        "audio_codec": probe["audio"].get("codec_name"),
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "frames": int(frames.shape[0]),
        "frame_rate": float(rate),
        "channels": int(waveform.shape[0]),
        "sample_rate": sample_rate,
        "video_duration_seconds": video_duration,
        "audio_duration_seconds": audio_duration,
        "av_duration_delta_seconds": abs(video_duration - audio_duration),
        "encoded_video_duration_seconds": encoded_video_duration,
        "encoded_audio_duration_seconds": encoded_audio_duration,
        "encoded_av_duration_delta_seconds": abs(
            encoded_video_duration - encoded_audio_duration
        ),
        "video_start_seconds": video_start,
        "audio_start_seconds": audio_start,
        "av_start_delta_seconds": abs(video_start - audio_start),
        "container_duration_seconds": float_field(
            probe["format"].get("duration"), "container duration"
        ),
    }


def validate_delivery_contract(media: dict[str, Any], label: str) -> None:
    exact = {
        "video_codec": "h264",
        "audio_codec": "aac",
        "width": 512,
        "height": 512,
        "frames": 22,
        "channels": 2,
        "sample_rate": 32000,
    }
    for name, expected in exact.items():
        if media.get(name) != expected:
            raise H3QualityError(
                f"{label} {name} differs: expected {expected!r}, "
                f"got {media.get(name)!r}"
            )
    if abs(float(media.get("frame_rate", 0.0)) - 24.0) > 1.0e-6:
        raise H3QualityError(f"{label} frame rate differs")
    expected_video_duration = 22.0 / 24.0
    expected_audio_duration = 37.0 * 800.0 / 32000.0
    if abs(
        media["video_duration_seconds"] - expected_video_duration
    ) > 1.0 / 24.0 or abs(
        media["encoded_video_duration_seconds"] - expected_video_duration
    ) > 1.0 / 24.0:
        raise H3QualityError(f"{label} video duration differs")
    if abs(
        media["audio_duration_seconds"] - expected_audio_duration
    ) > 2048.0 / 32000.0 or abs(
        media["encoded_audio_duration_seconds"] - expected_audio_duration
    ) > 2048.0 / 32000.0:
        raise H3QualityError(f"{label} audio duration differs")
    if media["av_start_delta_seconds"] > 1.0 / 32000.0:
        raise H3QualityError(f"{label} A/V start synchronization differs")
    if media["av_duration_delta_seconds"] > max(
        1.0 / 24.0, 2048.0 / 32000.0
    ) or media["encoded_av_duration_delta_seconds"] > 0.075:
        raise H3QualityError(f"{label} A/V duration synchronization differs")


def validate_parameter_contract(
    parameters: dict[str, Any],
    media: dict[str, Any],
    label: str,
    *,
    inferred_attention_kernel: str | None = None,
) -> None:
    preset = parameters.get("preset")
    contract = PRESET_CONTRACTS.get(preset)
    if contract is None:
        raise H3QualityError(f"{label} has an unsupported frozen preset")
    expected = {
        "schema": "mbufo.minimax-h3-text-generation.v1",
        "backend": "rocm-hip-gfx1151",
        "precision": "bf16-f32",
        "model_repository": "MiniMaxAI/MiniMax-H3",
        "model_revision": MODEL_REVISION,
        "reference_repository": "antirez/h3.c",
        "reference_revision": REFERENCE_REVISION,
        "output_width": 512,
        "output_height": 512,
        "fps": 24,
        "frames": 22,
        "token_reduction": False,
        "decode_audio": True,
        "mux": True,
        "selected_frames": [],
        **contract,
    }
    for name, value in expected.items():
        actual = parameters.get(name)
        if type(actual) is not type(value) or actual != value:
            raise H3QualityError(
                f"{label} parameter {name} differs from the frozen preset"
            )
    attention_kernel = parameters.get("attention_kernel")
    if attention_kernel is None:
        attention_kernel = inferred_attention_kernel
        if attention_kernel != "scalar":
            raise H3QualityError(
                f"{label} parameter provenance lacks an attention kernel"
            )
    elif (
        inferred_attention_kernel is not None
        and attention_kernel != inferred_attention_kernel
    ):
        raise H3QualityError(
            f"{label} parameter and profile attention kernels disagree"
        )
    if (
        not isinstance(parameters.get("seed"), int)
        or isinstance(parameters["seed"], bool)
        or parameters["seed"] < 0
        or parameters["seed"] > (1 << 64) - 1
        or not isinstance(parameters.get("prompt_sha256"), str)
        or len(parameters["prompt_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in parameters["prompt_sha256"]
        )
        or attention_kernel not in ("scalar", "row_parallel")
    ):
        raise H3QualityError(f"{label} parameter provenance is invalid")
    for parameter_name, media_name in (
        ("output_width", "width"),
        ("output_height", "height"),
        ("frames", "frames"),
        ("fps", "frame_rate"),
    ):
        if float(parameters[parameter_name]) != float(media[media_name]):
            raise H3QualityError(
                f"{label} parameters and delivered media disagree"
            )


def compare_parameter_provenance(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> None:
    for name in (
        "model_repository",
        "model_revision",
        "reference_repository",
        "reference_revision",
        "prompt_sha256",
        "seed",
        "output_width",
        "output_height",
        "fps",
        "frames",
        "decode_audio",
        "mux",
    ):
        if reference.get(name) != candidate.get(name):
            raise H3QualityError(
                f"reference and candidate parameter provenance differs for "
                f"{name}"
            )
    if reference.get("preset") != "exact-512":
        raise H3QualityError("reference parameter document is not exact-512")


def module_state_sha256(module: Any) -> str:
    """Hash a torch module's state without relying on archive serialization."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        value = tensor.detach().cpu().contiguous()
        encoded_dtype = str(value.dtype).encode("ascii")
        digest.update(len(encoded_dtype).to_bytes(8, "little"))
        digest.update(encoded_dtype)
        digest.update(len(value.shape).to_bytes(8, "little"))
        for dimension in value.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def compute_lpips(
    reference_frames: np.ndarray,
    candidate_frames: np.ndarray,
    *,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    """Compute pinned AlexNet LPIPS for delivered RGB frames."""

    if batch_size < 1:
        raise H3QualityError("LPIPS batch size must be positive")
    try:
        import lpips
        import torch
        import torchvision
    except ImportError as exc:
        raise H3QualityError(
            "LPIPS evaluation requires the pinned Nix development shell"
        ) from exc
    if device_name == "cuda" and not torch.cuda.is_available():
        raise H3QualityError("LPIPS CUDA evaluation requested without ROCm")
    torch_home = os.environ.get("TORCH_HOME")
    if not torch_home:
        raise H3QualityError(
            "LPIPS evaluation requires the pinned Nix TORCH_HOME"
        )
    alexnet_path = (
        Path(torch_home)
        / "hub"
        / "checkpoints"
        / ALEXNET_WEIGHTS_FILENAME
    )
    calibration_path = (
        Path(lpips.__file__).resolve().parent
        / "weights"
        / "v0.1"
        / "alex.pth"
    )
    alexnet_sha256 = require_weight_file(
        alexnet_path, ALEXNET_WEIGHTS_SHA256, "AlexNet weights"
    )
    calibration_sha256 = require_weight_file(
        calibration_path,
        LPIPS_ALEX_V01_SHA256,
        "LPIPS AlexNet calibration weights",
    )
    device = torch.device(device_name)
    model = lpips.LPIPS(
        net="alex",
        version="0.1",
        pretrained=True,
        pnet_rand=False,
        model_path=str(calibration_path),
        verbose=False,
    )
    model.eval()
    state_sha256 = module_state_sha256(model)
    model.to(device)
    scores: list[float] = []
    with torch.inference_mode():
        for begin in range(0, reference_frames.shape[0], batch_size):
            end = min(reference_frames.shape[0], begin + batch_size)
            reference = torch.from_numpy(
                np.ascontiguousarray(reference_frames[begin:end])
            )
            candidate = torch.from_numpy(
                np.ascontiguousarray(candidate_frames[begin:end])
            )
            reference = reference.permute(0, 3, 1, 2).to(device)
            candidate = candidate.permute(0, 3, 1, 2).to(device)
            values = model(reference, candidate, normalize=True)
            scores.extend(
                float(value)
                for value in values.detach().cpu().reshape(-1).tolist()
            )
    if len(scores) != reference_frames.shape[0] or not np.isfinite(scores).all():
        raise H3QualityError("LPIPS produced invalid frame scores")
    return {
        "per_frame": scores,
        "implementation": {
            "name": "richzhang-perceptual-similarity",
            "package": "lpips",
            "package_version": importlib.metadata.version("lpips"),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "network": "alex",
            "metric_version": "0.1",
            "alexnet_weights_filename": ALEXNET_WEIGHTS_FILENAME,
            "alexnet_weights_sha256": alexnet_sha256,
            "lpips_calibration_sha256": calibration_sha256,
            "state_sha256": state_sha256,
            "input_range": [0.0, 1.0],
            "device": device.type,
        },
    }


def lpips_summary(scores: list[float], indices: list[int]) -> dict[str, Any]:
    selected = [scores[index] for index in indices]
    return {
        "per_frame": [
            {"index": index, "value": scores[index]} for index in indices
        ],
        "mean": float(np.mean(selected)),
        "maximum": float(np.max(selected)),
    }


def compare_arrays(
    reference_frames: np.ndarray,
    candidate_frames: np.ndarray,
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    lpips_per_frame: list[float] | None = None,
) -> dict[str, Any]:
    if reference_frames.shape != candidate_frames.shape:
        raise H3QualityError("reference and candidate video geometry differs")
    if reference_audio.shape != candidate_audio.shape:
        raise H3QualityError("reference and candidate audio geometry differs")
    if reference_frames.shape[0] < 1:
        raise H3QualityError("quality comparison contains no frames")
    selected_indices = sorted(
        {0, reference_frames.shape[0] // 2, reference_frames.shape[0] - 1}
    )
    result = {
        "selected_frame_indices": selected_indices,
        "selected_frames": frame_metrics(
            reference_frames[selected_indices],
            candidate_frames[selected_indices],
            data_range=1.0,
        ),
        "full_clip": frame_metrics(
            reference_frames, candidate_frames, data_range=1.0
        ),
        "audio": audio_metrics(
            reference_audio, candidate_audio, sample_rate=sample_rate
        ),
    }
    if lpips_per_frame is not None:
        if len(lpips_per_frame) != reference_frames.shape[0] or not np.isfinite(
            lpips_per_frame
        ).all():
            raise H3QualityError("LPIPS frame scores have invalid geometry")
        result["selected_frames"]["lpips"] = lpips_summary(
            lpips_per_frame, selected_indices
        )
        result["full_clip"]["lpips"] = lpips_summary(
            lpips_per_frame, list(range(reference_frames.shape[0]))
        )
    return result


def build_report(
    reference_path: Path,
    candidate_path: Path,
    *,
    reference_label: str,
    candidate_label: str,
    ffmpeg: str,
    ffprobe: str,
    lpips_device: str,
    lpips_batch_size: int,
    skip_lpips: bool,
    require_parameters: bool,
) -> dict[str, Any]:
    reference_probe = probe_media(reference_path, ffprobe)
    candidate_probe = probe_media(candidate_path, ffprobe)
    reference_frames, reference_rate = extract_frames(
        reference_path, reference_probe, ffmpeg
    )
    candidate_frames, candidate_rate = extract_frames(
        candidate_path, candidate_probe, ffmpeg
    )
    reference_audio, reference_sample_rate = extract_audio(
        reference_path, reference_probe, ffmpeg
    )
    candidate_audio, candidate_sample_rate = extract_audio(
        candidate_path, candidate_probe, ffmpeg
    )
    if reference_rate != candidate_rate:
        raise H3QualityError("reference and candidate frame rates differ")
    if reference_sample_rate != candidate_sample_rate:
        raise H3QualityError("reference and candidate sample rates differ")
    reference_media = media_summary(
        reference_probe,
        reference_frames,
        reference_rate,
        reference_audio,
        reference_sample_rate,
    )
    candidate_media = media_summary(
        candidate_probe,
        candidate_frames,
        candidate_rate,
        candidate_audio,
        candidate_sample_rate,
    )
    validate_delivery_contract(reference_media, "reference")
    validate_delivery_contract(candidate_media, "candidate")
    reference = {
        "label": reference_label,
        "bytes": reference_path.stat().st_size,
        "sha256": sha256_file(reference_path),
        "media": reference_media,
    }
    candidate = {
        "label": candidate_label,
        "bytes": candidate_path.stat().st_size,
        "sha256": sha256_file(candidate_path),
        "media": candidate_media,
    }
    reference_parameters = sibling_parameters(reference_path)
    candidate_parameters = sibling_parameters(candidate_path)
    if require_parameters and (
        reference_parameters is None or candidate_parameters is None
    ):
        raise H3QualityError(
            "promotion evidence requires both parameter documents"
        )
    if reference_parameters is not None:
        reference_profile = sibling_profile(
            reference_path, reference_parameters["document"]
        )
        validate_parameter_contract(
            reference_parameters["document"],
            reference_media,
            "reference",
            inferred_attention_kernel=(
                None
                if reference_profile is None
                else reference_profile["attention_kernel"]
            ),
        )
        reference["parameters"] = reference_parameters
        if reference_profile is not None:
            reference["profile"] = reference_profile
    if candidate_parameters is not None:
        candidate_profile = sibling_profile(
            candidate_path, candidate_parameters["document"]
        )
        validate_parameter_contract(
            candidate_parameters["document"],
            candidate_media,
            "candidate",
            inferred_attention_kernel=(
                None
                if candidate_profile is None
                else candidate_profile["attention_kernel"]
            ),
        )
        candidate["parameters"] = candidate_parameters
        if candidate_profile is not None:
            candidate["profile"] = candidate_profile
    if reference_parameters is not None and candidate_parameters is not None:
        compare_parameter_provenance(
            reference_parameters["document"], candidate_parameters["document"]
        )
        parameter_status = "validated"
    else:
        parameter_status = "diagnostic-missing"
    perceptual = (
        None
        if skip_lpips
        else compute_lpips(
            reference_frames,
            candidate_frames,
            device_name=lpips_device,
            batch_size=lpips_batch_size,
        )
    )
    return {
        "schema": SCHEMA,
        "metric_implementations": {
            "windowed_ssim": {
                "name": "mbufo-gaussian-windowed-ssim",
                "window_size": 11,
                "sigma": 1.5,
                "covariance": "population",
                "numpy_version": np.__version__,
            },
            "audio_spectrogram": {
                "name": "torch-stft-compatible-magnitude",
                "n_fft": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "window": "periodic-hann",
                "center": True,
                "pad_mode": "reflect",
                "numpy_version": np.__version__,
            },
            "lpips": (
                perceptual["implementation"]
                if perceptual is not None
                else {"status": "diagnostic-skip"}
            ),
            "parameter_provenance": {"status": parameter_status},
        },
        "reference": reference,
        "candidate": candidate,
        "metrics": compare_arrays(
            reference_frames,
            candidate_frames,
            reference_audio,
            candidate_audio,
            reference_sample_rate,
            None if perceptual is None else perceptual["per_frame"],
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a MiniMax H3 preset MP4 with BF16 exact"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-label", default="exact")
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--lpips-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--lpips-batch-size", type=int, default=4)
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="diagnostic only; reports without LPIPS are not promotion evidence",
    )
    parser.add_argument(
        "--allow-missing-parameters",
        action="store_true",
        help="diagnostic only; promotion evidence requires both documents",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.reference = args.reference.resolve()
    args.candidate = args.candidate.resolve()
    args.output = args.output.resolve()
    return args


def main() -> int:
    args = parse_args()
    try:
        if not args.reference.is_file() or not args.candidate.is_file():
            raise H3QualityError("reference and candidate MP4s must exist")
        report = build_report(
            args.reference,
            args.candidate,
            reference_label=args.reference_label,
            candidate_label=args.candidate_label,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            lpips_device=args.lpips_device,
            lpips_batch_size=args.lpips_batch_size,
            skip_lpips=args.skip_lpips,
            require_parameters=not args.allow_missing_parameters,
        )
        atomic_json(args.output, report)
    except (H3QualityError, OSError, ValueError) as exc:
        print(f"MiniMax H3 preset quality error: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
