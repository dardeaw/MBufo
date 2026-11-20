#!/usr/bin/env python3
"""Process-isolated official Qwen3-TTS oracle used by MBufo validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", default="vivian")
    parser.add_argument("--language", default="english")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.reference_root))

    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = Qwen3TTSModel.from_pretrained(
        str(args.model),
        device_map="cpu",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.model.eval()

    captured: dict[str, torch.Tensor] = {}
    original_generate = model.model.generate

    def capture_generate(*positional, **keyword):
        codes, hidden = original_generate(*positional, **keyword)
        captured["codes"] = codes[0]
        return codes, hidden

    model.model.generate = capture_generate
    try:
        waveforms, sample_rate = model.generate_custom_voice(
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            subtalker_dosample=not args.greedy,
            seed=args.seed,
        )
    finally:
        model.model.generate = original_generate

    codes = np.asarray(captured["codes"].detach().cpu(), dtype="<i4")
    waveform = np.asarray(waveforms[0], dtype="<f4").reshape(-1)
    codes.tofile(args.out / "codes.i32")
    waveform.tofile(args.out / "waveform.f32")
    (args.out / "meta.json").write_text(
        json.dumps(
            {
                "sample_rate": int(sample_rate),
                "samples": int(waveform.size),
                "code_steps": int(codes.shape[0]),
                "code_groups": int(codes.shape[1]),
                "seed": args.seed,
                "greedy": args.greedy,
                "subtalker_greedy": args.greedy,
                "speaker": args.speaker,
                "language": args.language,
                "torch": torch.__version__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
