#!/usr/bin/env python3

import hashlib
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mbufo"))

from h3_rng import NormalRng


expected_u32 = (
    0x8DAF78A1,
    0x78625B1A,
    0x843DAF20,
    0x3314B8DD,
    0xF33E7A58,
    0x7A10C688,
    0x77A0A5E8,
    0xDBA4433B,
)
rng = NormalRng(42)
actual_u32 = tuple(rng.next_u32() for _ in expected_u32)
if actual_u32 != expected_u32:
    raise AssertionError(f"PCG32 sequence differs: {actual_u32!r}")

video_elements = 24 * 7 * 16 * 16
audio_elements = 32 * 2 * 37
video = NormalRng(42).fill(video_elements)
audio = NormalRng(42).fill(audio_elements)
payload = b"".join(struct.pack("<f", value) for value in video + audio)
actual_hash = hashlib.sha256(payload).hexdigest()
expected_hash = "0b9e324f731605e8b6050b2c7cdc46a320b75609c426ea548adb35332204620d"
if actual_hash != expected_hash:
    raise AssertionError(f"normal-noise SHA-256 differs: {actual_hash}")
if video[:audio_elements] != audio:
    raise AssertionError("same-seed audio noise is not the video prefix")

print("MiniMax H3 seeded-noise oracle tests passed.")
