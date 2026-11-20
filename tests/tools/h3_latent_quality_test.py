#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "mbufo" / "h3_latent_quality.py"
sys.path.insert(0, str(MODULE_PATH.parents[1]))
SPEC = importlib.util.spec_from_file_location("h3_latent_quality", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

direct = subprocess.run(
    [sys.executable, str(MODULE_PATH), "--help"],
    cwd=ROOT,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)
if direct.returncode != 0:
    raise AssertionError(
        "direct latent-quality CLI import failed: "
        + direct.stderr.decode("utf-8", errors="replace")
    )


def descriptor(path: Path, shape: list[int], dtype: str = "F32") -> dict:
    return {
        "dtype": dtype,
        "shape": shape,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    teacher = root / "teacher"
    native = root / "native"
    teacher.mkdir()
    native.mkdir()
    video = np.arange(24 * 7 * 2 * 2, dtype="<f4").reshape((24, 7, 2, 2))
    audio = np.arange(32 * 2 * 3, dtype="<f4").reshape((32, 2, 3))
    conditioning = np.arange(6 * 5120, dtype="<u2").reshape((6, 5120))
    for target in (teacher, native):
        video.tofile(target / "video_final.f32")
        audio.tofile(target / "audio_final.f32")
    conditioning.tofile(teacher / "conditioning.bf16")
    teacher_metadata = {
        "schema": "mbufo.minimax-h3-denoiser-golden.v1",
        "model_revision": MODULE.MODEL_REVISION,
        "reference_revision": MODULE.REFERENCE_REVISION,
        "geometry": {"width": 32, "height": 32, "frames": 22},
        "steps": 50,
        "blocks": 50,
        "reuse_interval": 1,
        "seed": 42,
        "noise_mode": "seeded",
        "files": {
            "video_final.f32": descriptor(
                teacher / "video_final.f32", list(video.shape)
            ),
            "audio_final.f32": descriptor(
                teacher / "audio_final.f32", list(audio.shape)
            ),
            "conditioning.bf16": descriptor(
                teacher / "conditioning.bf16",
                list(conditioning.shape),
                "BF16",
            ),
        },
    }
    (teacher / "metadata.json").write_text(json.dumps(teacher_metadata))
    native_manifest = {
        "schema": "mbufo.minimax-h3-final-latents.v1",
        "model_revision": MODULE.MODEL_REVISION,
        "reference_revision": MODULE.REFERENCE_REVISION,
        "seed": 42,
        "noise_mode": "seeded",
        "prompt_sha256": "a" * 64,
        "conditioning_sha256": teacher_metadata["files"][
            "conditioning.bf16"
        ]["sha256"],
        "geometry": {"width": 32, "height": 32, "frames": 22},
        "steps": 50,
        "blocks": 50,
        "reuse_interval": 1,
        "attention_kernel": "row_parallel",
        "video": descriptor(native / "video_final.f32", list(video.shape)),
        "audio": descriptor(native / "audio_final.f32", list(audio.shape)),
    }
    (native / "manifest.json").write_text(json.dumps(native_manifest))
    report = MODULE.build_report(teacher, native)
    if not report["passed"] or report["metrics"]["video"]["relative_l2"] != 0:
        raise AssertionError("exact latent comparison failed")
    corrupted = video.copy()
    corrupted.flat[0] = np.float32(1.0e9)
    corrupted.tofile(native / "video_final.f32")
    native_manifest["video"] = descriptor(
        native / "video_final.f32", list(video.shape)
    )
    (native / "manifest.json").write_text(json.dumps(native_manifest))
    report = MODULE.build_report(teacher, native)
    if report["passed"] or report["metrics"]["video"]["passed"]:
        raise AssertionError("latent corruption passed")

    native_manifest["seed"] = 43
    (native / "manifest.json").write_text(json.dumps(native_manifest))
    try:
        MODULE.build_report(teacher, native)
    except MODULE.H3QualityError:
        pass
    else:
        raise AssertionError("mismatched latent provenance passed")

print("MiniMax H3 latent quality reporter tests passed.")
