#!/usr/bin/env python3

import hashlib
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "mbufo"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "h3_denoiser_golden", TOOLS / "h3_denoiser_golden.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


small = MODULE.geometry(256, 256, 22)
large = MODULE.geometry(512, 512, 22)
check(small["rows"] == 528, "256 layout rows")
check(small["video_rows"] == 448, "256 video rows")
check(large["rows"] == 1872, "512 layout rows")
check(large["video_rows"] == 1792, "512 video rows")
check(MODULE.geometry(512, 512, 22, 12)["rows"] == 1878, "variable text rows")
check(tuple(MODULE.production_positions(small).shape) == (528, 3), "256 RoPE")
check(tuple(MODULE.production_positions(large).shape) == (1872, 3), "512 RoPE")
check(
    MODULE.torch.equal(
        MODULE.production_positions(small),
        MODULE.block.production_positions(),
    ),
    "generalized 256 positions preserve the frozen teacher",
)

video, audio = MODULE.initial_latents(
    MODULE.torch.device("cpu"), small, mode="seeded", seed=42
)
payload = video.numpy().astype("<f4", copy=False).tobytes()
payload += audio.numpy().astype("<f4", copy=False).tobytes()
check(
    hashlib.sha256(payload).hexdigest()
    == "6318dbfea74c61415d470c12c019cda9df6a8491f86b075e403d1c2fc2403b4d",
    "serving noise identity",
)

source = MODULE.torch.arange(
    24
    * large["video_latent_frames"]
    * large["latent_height"]
    * large["latent_width"],
    dtype=MODULE.torch.float32,
)
rows = MODULE.patch_video(source, large)
check(tuple(rows.shape) == (1792, 96), "512 patch geometry")
check(MODULE.torch.equal(MODULE.unpatch_video(rows, large), source), "patch roundtrip")

print("MiniMax H3 generalized denoiser teacher tests passed.")
