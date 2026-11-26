#!/usr/bin/env python3
"""CI guard for the shared blocked-KV codec (blocked_kv_codec.hpp).

The codec tags, nibble math and block-scale conversions have exactly one
definition. New copies -- however well-intentioned ("bit-compatible",
"dependency-free") -- silently fork numerics the next time one copy is
edited. This check fails the build on:

  A. a new `struct KvCodec(Q8|Fp8|Q4)` definition outside blocked_kv_codec.hpp;
  B. a new `struct CodecTraits` definition (the pre-unification shim) anywhere;
  C. direct block-scale bit-casts that bypass the trait helpers, i.e.
     `__half2float(*reinterpret_cast<const __half*>` reads or
     `= __float2half(sc)` scale writes outside blocked_kv_codec.hpp
     (output-value conversions such as `__float2half(q * sk)` are fine and
     are not matched).

Pre-existing private copies (other engines' kernels predate the trait) and
pattern collisions (weight-quant/GEMM/sampling code that merely shares the
syntax but is not a blocked-KV scale) are listed in GRANDFATHERED with a
reason. Anything not on that list fails. Do not extend GRANDFATHERED for
new code -- migrate it to the trait instead.

Usage: python3 tools/ci/check-kv-codec.py [--repo ROOT]
Exit status 1 on any non-grandfathered violation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ALLOW_HEADER = "src/core/compute/attention/blocked_kv_codec.hpp"

RULES = [
    ("codec-tag-duplicate",
     re.compile(r"struct\s+KvCodec(Q8|Fp8|Q4)\b"),
     "codec tag redefined; use blocked_kv::KvCodec* from blocked_kv_codec.hpp"),
    ("codectraits-shim",
     re.compile(r"struct\s+CodecTraits\b"),
     "CodecTraits shim reintroduced; call Codec::Decode via the unified tag"),
    ("scale-read-bypass",
     re.compile(r"__half2float\(\*reinterpret_cast<const __half\*>\("),
     "block-scale read bypasses LoadScaleF16 (use the trait helper)"),
    ("scale-write-bypass",
     re.compile(r"=\s*__float2half\(sc\)"),
     "block-scale write bypasses StoreScaleF16 (use the trait helper)"),
]

# file -> reason. "pending-migration": real blocked-KV forks to convert.
# "pattern-collision": same syntax, not a KV scale (weight quant / GEMM /
# sampling); kept out of the rule's reach permanently.
GRANDFATHERED = {
    "src/models/gemma4/kernels/kernels.hip.cpp": "pending-migration",
    "src/models/gemma4/kernels/attention_wmma.hip.cpp": "pending-migration",
    "src/models/gemma4/kernels/attention_wmma512.hip.cpp": "pending-migration",
    "src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp": "pending-migration",
    "src/models/llama/kernels/kernels.hip.cpp": "pending-migration",
    "src/models/gptoss/kernels/kernels.hip.cpp": "pending-migration",
    "src/models/glimmer/kernels/kernels.hip.cpp": "pending-migration",
    "src/core/compute/attention/flash_wmma_large.hip.cpp": "pending-migration",
    "src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_iq2_gate.hip.hpp": "pattern-collision",
    "src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_dspark_sampling.hip.hpp": "pattern-collision",
    "src/core/quant/row_dot_traits.hpp": "pattern-collision",
    "src/core/quant/quant_traits.hpp": "pattern-collision",
    "src/core/compute/gemm/moe_fused.hip": "pattern-collision",
    "src/core/compute/gemm/gemv_wave32_sdot4.hip": "pattern-collision",
    "src/core/compute/gemm/routed_moe.hip": "pattern-collision",
}

SKIP_DIRS = {".git", "build", "CMakeFiles"}
SKIP_SUFFIXES = (".bak_trait", ".bak_trait2", ".bak_trait3", ".bak_q8", ".bak_kv", ".bak_host",
                 ".bak_fix1")
SCAN_SUFFIXES = {".hpp", ".hip", ".cpp", ".h", ".cu"}


def iter_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in SCAN_SUFFIXES:
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        if any(d in p.parts for d in SKIP_DIRS):
            continue
        if rel.endswith(SKIP_SUFFIXES):
            continue
        yield p, rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    root = Path(args.repo)
    hits = 0
    suppressed = 0
    for path, rel in iter_files(root):
        if rel == ALLOW_HEADER:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name, pat, msg in RULES:
            for m in pat.finditer(text):
                if rel in GRANDFATHERED:
                    suppressed += 1
                    continue
                line = text.count("\n", 0, m.start()) + 1
                print(f"{rel}:{line}: [{name}] {msg}")
                hits += 1
    print(f"check-kv-codec: {hits} violation(s), {suppressed} grandfathered hit(s) suppressed")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
