"""Dependency-free MiniMax H3 seeded normal-noise oracle."""

from __future__ import annotations

import math
import struct


_MASK64 = (1 << 64) - 1
_MASK32 = (1 << 32) - 1
_MULTIPLIER = 6364136223846793005
_MIX = 0x9E3779B97F4A7C15
_TAU = 2.0 * 3.14159265358979323846


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


class NormalRng:
    """PCG32 plus Box-Muller, byte-compatible with the native sampler."""

    def __init__(self, seed: int):
        if seed < 0 or seed > _MASK64:
            raise ValueError("seed must fit uint64")
        self.state = 0
        self.increment = ((seed << 1) | 1) & _MASK64
        self.spare = 0.0
        self.has_spare = False
        self.next_u32()
        self.state = (self.state + (seed ^ _MIX)) & _MASK64
        self.next_u32()

    def next_u32(self) -> int:
        old_state = self.state
        self.state = (
            old_state * _MULTIPLIER + self.increment
        ) & _MASK64
        shifted = (((old_state >> 18) ^ old_state) >> 27) & _MASK32
        rotation = (old_state >> 59) & 31
        return (
            (shifted >> rotation)
            | (shifted << ((-rotation) & 31))
        ) & _MASK32

    def next_normal(self) -> float:
        if self.has_spare:
            self.has_spare = False
            return self.spare
        uniform1 = (float(self.next_u32()) + 1.0) / 4294967297.0
        uniform2 = (float(self.next_u32()) + 0.5) / 4294967296.0
        radius = math.sqrt(-2.0 * math.log(uniform1))
        angle = _TAU * uniform2
        self.spare = _float32(radius * math.sin(angle))
        self.has_spare = True
        return _float32(radius * math.cos(angle))

    def fill(self, elements: int) -> list[float]:
        if elements < 0:
            raise ValueError("element count must be non-negative")
        return [self.next_normal() for _ in range(elements)]
