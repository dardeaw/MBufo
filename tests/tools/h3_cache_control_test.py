#!/usr/bin/env python3

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "mbufo" / "h3_cache_control.py"
SPEC = importlib.util.spec_from_file_location("h3_cache_control", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "a.bin").write_bytes(b"a" * 4096)
        (root / "nested").mkdir()
        (root / "nested" / "b.bin").write_bytes(b"b" * 8192)
        (root / "ignored-link").symlink_to(root / "a.bin")
        report = MODULE.evict(root)
        if (
            not report["successful"]
            or report["files_attempted"] != 2
            or report["bytes_attempted"] != 12288
            or "model" in json.dumps(report)
        ):
            raise AssertionError(f"cache-control report differs: {report}")

print("MiniMax H3 cache-control tests passed.")
