#!/usr/bin/env python3
"""Seal, verify, or compare MiniMax H3 quality-oracle artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from mbufo.h3_quality import (
    H3QualityError,
    audio_metrics,
    frame_metrics,
    numeric_metrics,
    seal_artifact,
    verify_artifact,
)


def _write_result(value: dict, output: Path | None) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, "utf-8")
    else:
        print(encoded, end="")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage content-addressed MiniMax H3 quality artifacts"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--staging", type=Path, required=True)
    seal.add_argument("--template", type=Path, required=True)
    seal.add_argument("--store", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument(
        "--kind", choices=("tensor", "frames", "audio"), required=True
    )
    compare.add_argument("--data-range", type=float, default=1.0)
    compare.add_argument("--sample-rate", type=int, default=32000)
    compare.add_argument("--output", type=Path)

    args = parser.parse_args()
    try:
        if args.operation == "verify":
            result = verify_artifact(args.artifact)
        elif args.operation == "seal":
            destination = seal_artifact(
                args.staging, args.template, args.store
            )
            result = {
                "status": "ok",
                "artifact_id": destination.name,
                "path": str(destination),
            }
        else:
            reference = np.load(args.reference, allow_pickle=False)
            candidate = np.load(args.candidate, allow_pickle=False)
            if args.kind == "tensor":
                result = numeric_metrics(reference, candidate)
            elif args.kind == "frames":
                result = frame_metrics(
                    reference, candidate, data_range=args.data_range
                )
            else:
                result = audio_metrics(
                    reference, candidate, sample_rate=args.sample_rate
                )
            _write_result(result, args.output)
            return 0
    except (H3QualityError, OSError, ValueError) as exc:
        print(f"MiniMax H3 quality error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
