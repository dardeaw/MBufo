#!/usr/bin/env python3
"""Request clean MiniMax H3 model pages be evicted from the Linux page cache."""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
from pathlib import Path
from typing import Any


SCHEMA = "mbufo.minimax-h3-cache-control.v1"


def memory_snapshot() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        name, separator, value = line.partition(":")
        if separator and name in {"Cached", "Buffers", "MemAvailable"}:
            fields = value.strip().split()
            if fields and fields[0].isdigit():
                result[f"{name.lower()}_kib"] = int(fields[0])
    return result


def model_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for directory, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(
            name
            for name in directories
            if not (Path(directory) / name).is_symlink()
        )
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink():
                result.append(path)
    return result


def evict(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError("model cache root must be a directory")
    if not hasattr(os, "posix_fadvise") or not hasattr(
        os, "POSIX_FADV_DONTNEED"
    ):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    started = time.monotonic()
    before = memory_snapshot()
    files = model_files(root)
    attempted_bytes = 0
    errors: list[dict[str, Any]] = []
    for index, path in enumerate(files):
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                status = os.fstat(descriptor)
                if not stat.S_ISREG(status.st_mode):
                    raise OSError("cache-control target is not a regular file")
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                attempted_bytes += status.st_size
            finally:
                os.close(descriptor)
        except OSError as exc:
            errors.append(
                {
                    "file_index": index,
                    "errno": exc.errno,
                }
            )
    return {
        "schema": SCHEMA,
        "operation": "posix_fadvise_dontneed",
        "files_attempted": len(files),
        "bytes_attempted": attempted_bytes,
        "errors": errors,
        "memory_before": before,
        "memory_after": memory_snapshot(),
        "elapsed_ms": (time.monotonic() - started) * 1000.0,
        "successful": not errors,
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
        description="Evict clean MiniMax H3 model pages before a cold profile"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evict(args.model.resolve())
        atomic_json(args.output.resolve(), report)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"MiniMax H3 cache-control error: {exc}", file=os.sys.stderr)
        return 1
    return 0 if report["successful"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
