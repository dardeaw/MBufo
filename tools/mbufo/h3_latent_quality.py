#!/usr/bin/env python3
"""Compare native MiniMax H3 final latents with a direct teacher artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from mbufo.h3_quality import (
    H3QualityError,
    MODEL_REVISION,
    REFERENCE_REVISION,
    numeric_metrics,
)


SCHEMA = "mbufo.minimax-h3-latent-quality.v1"
MAX_RELATIVE_L2 = 0.10
MAX_RELATIVE_MAX = 0.15


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise H3QualityError(f"{path.name} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise H3QualityError(f"{path.name} must contain an object")
    return value


def validate_payload(
    root: Path,
    name: str,
    descriptor: dict[str, Any],
    *,
    expected_dtype: str = "F32",
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        descriptor.get("dtype") != expected_dtype
        or not isinstance(descriptor.get("shape"), list)
        or not descriptor["shape"]
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in descriptor["shape"]
        )
        or not isinstance(descriptor.get("bytes"), int)
        or not isinstance(descriptor.get("sha256"), str)
    ):
        raise H3QualityError(f"{name} descriptor is invalid")
    path = root / name
    expected_elements = math.prod(descriptor["shape"])
    item_bytes = 4 if expected_dtype == "F32" else 2
    numpy_dtype = "<f4" if expected_dtype == "F32" else "<u2"
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != descriptor["bytes"]
        or descriptor["bytes"] != expected_elements * item_bytes
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise H3QualityError(f"{name} payload identity differs")
    values = np.fromfile(path, dtype=numpy_dtype).reshape(descriptor["shape"])
    if expected_dtype == "F32" and not np.isfinite(values).all():
        raise H3QualityError(f"{name} contains non-finite values")
    return values, {
        "bytes": descriptor["bytes"],
        "sha256": descriptor["sha256"],
        "shape": descriptor["shape"],
        "dtype": expected_dtype,
    }


def teacher_payloads(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = load_json(root / "metadata.json")
    if metadata.get("schema") != "mbufo.minimax-h3-denoiser-golden.v1":
        raise H3QualityError("teacher metadata schema differs")
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise H3QualityError("teacher metadata lacks files")
    tensors: dict[str, np.ndarray] = {}
    identity: dict[str, Any] = {}
    for role, name in (
        ("video", "video_final.f32"),
        ("audio", "audio_final.f32"),
    ):
        descriptor = files.get(name)
        if not isinstance(descriptor, dict):
            raise H3QualityError(f"teacher lacks {name}")
        tensors[role], identity[role] = validate_payload(root, name, descriptor)
    identity["model_revision"] = metadata.get("model_revision")
    identity["reference_revision"] = metadata.get("reference_revision")
    identity["geometry"] = metadata.get("geometry")
    identity["steps"] = metadata.get("steps")
    identity["blocks"] = metadata.get("blocks")
    identity["reuse_interval"] = metadata.get("reuse_interval")
    identity["seed"] = metadata.get("seed")
    identity["noise_mode"] = metadata.get("noise_mode")
    conditioning = files.get("conditioning.bf16")
    if not isinstance(conditioning, dict):
        raise H3QualityError("teacher lacks conditioning.bf16")
    _, conditioning_identity = validate_payload(
        root, "conditioning.bf16", conditioning, expected_dtype="BF16"
    )
    identity["conditioning_sha256"] = conditioning_identity["sha256"]
    if identity["model_revision"] != MODEL_REVISION:
        raise H3QualityError("teacher model revision differs")
    if identity["reference_revision"] != REFERENCE_REVISION:
        raise H3QualityError("teacher reference revision differs")
    if identity["noise_mode"] != "seeded":
        raise H3QualityError("teacher noise mode differs")
    return tensors, identity


def native_payloads(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = load_json(root / "manifest.json")
    if manifest.get("schema") != "mbufo.minimax-h3-final-latents.v1":
        raise H3QualityError("native latent manifest schema differs")
    tensors: dict[str, np.ndarray] = {}
    identity: dict[str, Any] = {}
    for role, name in (
        ("video", "video_final.f32"),
        ("audio", "audio_final.f32"),
    ):
        descriptor = manifest.get(role)
        if not isinstance(descriptor, dict):
            raise H3QualityError(f"native manifest lacks {role}")
        tensors[role], identity[role] = validate_payload(root, name, descriptor)
    for name in (
        "model_revision",
        "reference_revision",
        "noise_mode",
        "attention_kernel",
    ):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise H3QualityError(f"native manifest lacks {name}")
        identity[name] = manifest[name]
    for name in ("conditioning_sha256", "prompt_sha256"):
        if not is_sha256(manifest.get(name)):
            raise H3QualityError(f"native manifest has invalid {name}")
        identity[name] = manifest[name]
    if identity["attention_kernel"] not in ("scalar", "row_parallel"):
        raise H3QualityError("native manifest has invalid attention_kernel")
    for name in ("seed", "steps", "blocks", "reuse_interval"):
        value = manifest.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise H3QualityError(f"native manifest lacks {name}")
        identity[name] = value
    geometry = manifest.get("geometry")
    if (
        not isinstance(geometry, dict)
        or any(
            isinstance(geometry.get(name), bool)
            or not isinstance(geometry.get(name), int)
            for name in ("width", "height", "frames")
        )
    ):
        raise H3QualityError("native manifest lacks geometry")
    identity["geometry"] = geometry
    if identity["model_revision"] != MODEL_REVISION:
        raise H3QualityError("native model revision differs")
    if identity["reference_revision"] != REFERENCE_REVISION:
        raise H3QualityError("native reference revision differs")
    if identity["noise_mode"] != "seeded":
        raise H3QualityError("native noise mode differs")
    return tensors, identity


def build_report(reference_root: Path, candidate_root: Path) -> dict[str, Any]:
    reference, reference_identity = teacher_payloads(reference_root)
    candidate, candidate_identity = native_payloads(candidate_root)
    for field in (
        "model_revision",
        "reference_revision",
        "seed",
        "noise_mode",
        "steps",
        "blocks",
        "reuse_interval",
        "conditioning_sha256",
    ):
        if reference_identity.get(field) != candidate_identity.get(field):
            raise H3QualityError(f"latent provenance differs for {field}")
    for field in ("width", "height", "frames"):
        if reference_identity["geometry"].get(field) != candidate_identity[
            "geometry"
        ].get(field):
            raise H3QualityError(f"latent geometry differs for {field}")
    metrics: dict[str, Any] = {}
    passed = True
    for role in ("video", "audio"):
        if reference[role].shape != candidate[role].shape:
            raise H3QualityError(f"{role} latent shape differs")
        role_metrics = numeric_metrics(reference[role], candidate[role])
        role_passed = (
            role_metrics["valid"]
            and role_metrics["relative_l2"] < MAX_RELATIVE_L2
            and role_metrics["relative_max"] < MAX_RELATIVE_MAX
        )
        role_metrics["passed"] = role_passed
        metrics[role] = role_metrics
        passed &= role_passed
    return {
        "schema": SCHEMA,
        "thresholds": {
            "relative_l2_maximum": MAX_RELATIVE_L2,
            "relative_max_maximum": MAX_RELATIVE_MAX,
            "source": "quality-contract-v1.semantic_latent",
        },
        "reference": reference_identity,
        "candidate": candidate_identity,
        "metrics": metrics,
        "passed": passed,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(
        f"{path.name}.partial-{os.getpid()}-{time.monotonic_ns()}"
    )
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare MiniMax H3 native and teacher final latents"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(
            args.reference.resolve(), args.candidate.resolve()
        )
        atomic_json(args.output.resolve(), report)
    except (H3QualityError, OSError, ValueError) as exc:
        print(f"MiniMax H3 latent quality error: {exc}", file=os.sys.stderr)
        return 1
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
