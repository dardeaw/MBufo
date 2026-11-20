#!/usr/bin/env python3

import importlib.util
import hashlib
import json
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "mbufo" / "h3_preset_quality.py"
sys.path.insert(0, str(MODULE_PATH.parents[1]))
SPEC = importlib.util.spec_from_file_location("h3_preset_quality", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


reference_frames = np.zeros((22, 2, 2, 3), dtype=np.float32)
candidate_frames = reference_frames.copy()
reference_audio = np.zeros((2, 2048), dtype=np.float32)
candidate_audio = reference_audio.copy()
metrics = MODULE.compare_arrays(
    reference_frames,
    candidate_frames,
    reference_audio,
    candidate_audio,
    32000,
    [0.0] * 22,
)
check(metrics["selected_frame_indices"] == [0, 11, 21], "selected frame set")
check(metrics["full_clip"]["numeric"]["relative_l2"] == 0.0, "exact video")
check(metrics["full_clip"]["ssim_windowed_mean"] == 1.0, "exact windowed SSIM")
check(metrics["full_clip"]["lpips"]["mean"] == 0.0, "exact LPIPS")
check(metrics["audio"]["waveform"]["relative_l2"] == 0.0, "exact audio")

candidate_frames[11, 0, 0, 0] = 1.0
candidate_audio[1, 10] = 0.5
metrics = MODULE.compare_arrays(
    reference_frames,
    candidate_frames,
    reference_audio,
    candidate_audio,
    32000,
)
check(metrics["full_clip"]["numeric"]["max_abs"] == 1.0, "video corruption")
check(metrics["audio"]["waveform"]["max_abs"] == 0.5, "audio corruption")

probe = {
    "video": {
        "codec_name": "h264",
        "start_time": "0",
        "duration": str(22.0 / 24.0),
    },
    "audio": {
        "codec_name": "aac",
        "start_time": "0",
        "duration": str(29600.0 / 32000.0),
    },
    "format": {"duration": "0.925"},
}
summary = MODULE.media_summary(
    probe,
    reference_frames,
    Fraction(24, 1),
    reference_audio,
    32000,
)
check(summary["frames"] == 22, "frame count")
check(summary["frame_rate"] == 24.0, "frame rate")
check(summary["channels"] == 2, "channel count")

captured_audio_command = []
original_run_bytes = MODULE.run_bytes


def capture_audio_command(command):
    captured_audio_command.append(command)
    return np.zeros(8, dtype="<f4").tobytes()


MODULE.run_bytes = capture_audio_command
try:
    decoded_audio, decoded_rate = MODULE.extract_audio(
        Path("test.mp4"),
        {"audio": {"sample_rate": "32000", "channels": "2"}},
        "ffmpeg",
    )
finally:
    MODULE.run_bytes = original_run_bytes
check(decoded_audio.shape == (2, 4), "decoded audio geometry")
check(decoded_rate == 32000, "decoded audio sample rate")
check(
    captured_audio_command[0].count("-ac") == 1,
    "FFmpeg audio channel option appears exactly once",
)
channel_option = captured_audio_command[0].index("-ac")
check(
    captured_audio_command[0][channel_option + 1] == "2",
    "FFmpeg audio channel option has the channel count",
)

delivery = {
    **summary,
    "width": 512,
    "height": 512,
    "frames": 22,
    "frame_rate": 24.0,
    "channels": 2,
    "sample_rate": 32000,
    "video_duration_seconds": 22.0 / 24.0,
    "audio_duration_seconds": 29600.0 / 32000.0,
    "av_duration_delta_seconds": abs(22.0 / 24.0 - 29600.0 / 32000.0),
    "av_start_delta_seconds": 0.0,
}
MODULE.validate_delivery_contract(delivery, "test")
bad_delivery = dict(delivery)
bad_delivery["video_codec"] = "vp9"
try:
    MODULE.validate_delivery_contract(bad_delivery, "test")
except MODULE.H3QualityError:
    pass
else:
    raise AssertionError("wrong delivery codec passed")
bad_delivery = dict(delivery)
bad_delivery["encoded_audio_duration_seconds"] = 0.5
try:
    MODULE.validate_delivery_contract(bad_delivery, "test")
except MODULE.H3QualityError:
    pass
else:
    raise AssertionError("wrong encoded duration passed")

exact_parameters = {
    "schema": "mbufo.minimax-h3-text-generation.v1",
    "backend": "rocm-hip-gfx1151",
    "precision": "bf16-f32",
    "model_repository": "MiniMaxAI/MiniMax-H3",
    "model_revision": MODULE.MODEL_REVISION,
    "reference_repository": "antirez/h3.c",
    "reference_revision": MODULE.REFERENCE_REVISION,
    "preset": "exact-512",
    "prompt_sha256": "a" * 64,
    "seed": 42,
    "internal_width": 512,
    "internal_height": 512,
    "output_width": 512,
    "output_height": 512,
    "fps": 24,
    "frames": 22,
    "evaluations": 49,
    "active_blocks": 50,
    "reuse_interval": 1,
    "attention_kernel": "scalar",
    "token_reduction": False,
    "decode_audio": True,
    "mux": True,
    "selected_frames": [],
}
fast_parameters = dict(exact_parameters)
fast_parameters.update(
    {
        "preset": "fast-384",
        "internal_width": 384,
        "internal_height": 384,
        "evaluations": 19,
        "active_blocks": 45,
        "reuse_interval": 2,
        "attention_kernel": "row_parallel",
    }
)
MODULE.validate_parameter_contract(exact_parameters, delivery, "reference")
MODULE.validate_parameter_contract(fast_parameters, delivery, "candidate")
MODULE.compare_parameter_provenance(exact_parameters, fast_parameters)
legacy_parameters = dict(exact_parameters)
del legacy_parameters["attention_kernel"]
try:
    MODULE.validate_parameter_contract(
        legacy_parameters, delivery, "legacy reference"
    )
except MODULE.H3QualityError:
    pass
else:
    raise AssertionError("missing attention-kernel provenance passed")
MODULE.validate_parameter_contract(
    legacy_parameters,
    delivery,
    "legacy reference",
    inferred_attention_kernel="scalar",
)
wrong_seed = dict(fast_parameters)
wrong_seed["seed"] = 43
try:
    MODULE.compare_parameter_provenance(exact_parameters, wrong_seed)
except MODULE.H3QualityError:
    pass
else:
    raise AssertionError("mismatched parameter provenance passed")
wrong_boolean_type = dict(fast_parameters)
wrong_boolean_type["token_reduction"] = 0
try:
    MODULE.validate_parameter_contract(
        wrong_boolean_type, delivery, "candidate"
    )
except MODULE.H3QualityError:
    pass
else:
    raise AssertionError("integer substituted for Boolean passed")

with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / "quality.json"
    value = {"schema": MODULE.SCHEMA, "private_path": False}
    MODULE.atomic_json(output, value)
    check(json.loads(output.read_text(encoding="utf-8")) == value, "atomic JSON")
    media = Path(directory) / "output.mp4"
    media.write_bytes(b"mp4")
    check(MODULE.sibling_parameters(media) is None, "optional parameters")
    parameters = {
        "schema": "mbufo.minimax-h3-text-generation.v1",
        "preset": "exact-512",
    }
    (Path(directory) / "parameters.json").write_text(
        json.dumps(parameters), encoding="utf-8"
    )
    linked = MODULE.sibling_parameters(media)
    check(linked is not None, "parameter report linked")
    check(linked["document"] == parameters, "parameter document retained")
    legacy_profile = {
        "schema": "mbufo.minimax-h3-profile-run.v1",
        "binary_sha256": MODULE.LEGACY_SCALAR_BINARY_SHA256,
        "parameters": parameters,
        "telemetry": {},
        "output": {"sha256": MODULE.sha256_file(media)},
    }
    (Path(directory) / "profile.json").write_text(
        json.dumps(legacy_profile), encoding="utf-8"
    )
    profile = MODULE.sibling_profile(media, parameters)
    check(profile is not None, "legacy profile linked")
    check(
        profile["attention_kernel"] == "scalar"
        and profile["attention_kernel_provenance"] == "legacy-scalar-binary",
        "legacy scalar kernel provenance",
    )
    legacy_profile["binary_sha256"] = "f" * 64
    (Path(directory) / "profile.json").write_text(
        json.dumps(legacy_profile), encoding="utf-8"
    )
    try:
        MODULE.sibling_profile(media, parameters)
    except MODULE.H3QualityError:
        pass
    else:
        raise AssertionError("unknown profile inferred scalar attention")
    weight = Path(directory) / "weight.pth"
    weight.write_bytes(b"pinned-weight")
    weight_hash = hashlib.sha256(weight.read_bytes()).hexdigest()
    check(
        MODULE.require_weight_file(weight, weight_hash, "test weight")
        == weight_hash,
        "pinned weight hash",
    )
    weight.write_bytes(b"corrupt")
    try:
        MODULE.require_weight_file(weight, weight_hash, "test weight")
    except MODULE.H3QualityError:
        pass
    else:
        raise AssertionError("corrupted pinned weight passed")

print("MiniMax H3 preset quality reporter tests passed.")
