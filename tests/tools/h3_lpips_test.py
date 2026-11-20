#!/usr/bin/env python3

import importlib.util
import os
import sys
import tempfile
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


frames = np.zeros((2, 64, 64, 3), dtype=np.float32)
result = MODULE.compute_lpips(
    frames,
    frames.copy(),
    device_name="cpu",
    batch_size=1,
)
if len(result["per_frame"]) != 2 or max(
    abs(value) for value in result["per_frame"]
) > 1.0e-8:
    raise AssertionError("identical LPIPS frames are not numerically identical")
implementation = result["implementation"]
if (
    implementation["alexnet_weights_sha256"]
    != MODULE.ALEXNET_WEIGHTS_SHA256
    or implementation["lpips_calibration_sha256"]
    != MODULE.LPIPS_ALEX_V01_SHA256
    or len(implementation["state_sha256"]) != 64
):
    raise AssertionError("LPIPS implementation identity differs")

original_torch_home = os.environ.get("TORCH_HOME")
with tempfile.TemporaryDirectory() as directory:
    os.environ["TORCH_HOME"] = directory
    try:
        MODULE.compute_lpips(
            frames[:1],
            frames[:1],
            device_name="cpu",
            batch_size=1,
        )
    except MODULE.H3QualityError:
        pass
    else:
        raise AssertionError("missing pinned AlexNet weights passed")
if original_torch_home is None:
    os.environ.pop("TORCH_HOME", None)
else:
    os.environ["TORCH_HOME"] = original_torch_home

print("MiniMax H3 pinned offline LPIPS test passed.")
