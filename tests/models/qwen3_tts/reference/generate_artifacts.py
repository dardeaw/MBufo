"""Run Qwen3-TTS CustomVoice reference baseline, dumping deterministic artifacts.

Emits:
  - talker_codes.npy : discrete codec token book (T x num_code_groups), the exact
    discrete output a mbufo engine must reproduce
  - waveform.npy     : decoded float32 waveform @ 24 kHz  (deterministic seed)
  - meta.json        : sample rate / shape metadata

Used for accuracy validation against mbufo.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

TEXT = (
    "The boy who lived. Mr. and Mrs. Dursley, of number four, Privet Drive, were "
    "proud to say that they were perfectly normal, thank you very much. They were "
    "the last people you'd expect to be involved in anything strange or mysterious, "
    "because they just didn't hold with such nonsense."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reference-root", default="/home/fbozzo/projects/Qwen3-TTS"
    )
    ap.add_argument(
        "--dependency-root",
        help=(
            "site-packages containing the official pinned Qwen dependencies; "
            "Torch and NumPy are imported from the selected interpreter first"
        ),
    )
    ap.add_argument("--model", default="/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--out", default="/home/fbozzo/projects/mbufo/artifacts/qwen3_tts")
    ap.add_argument("--speaker", default="vivian")
    ap.add_argument("--language", default="english")
    ap.add_argument(
        "--text-file",
        help="UTF-8 input text; defaults to the checked-in canonical sentence",
    )
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--device",
        default="cpu",
        help="official PyTorch device map, for example cpu or cuda:0 on ROCm",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        help=(
            "generation cap; defaults to 6 for the bounded greedy exactness "
            "contract and 2048 for sampled quality generation"
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--greedy", action="store_true",
                    help="use deterministic argmax sampling (do_sample=False)")
    args = ap.parse_args()
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 6 if args.greedy else 2048
    if max_new_tokens <= 0:
        ap.error("--max-new-tokens must be positive")
    text = TEXT
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as stream:
            text = stream.read().strip()
        if not text:
            ap.error("--text-file must contain non-empty UTF-8 text")

    compat_root = os.path.join(os.path.dirname(__file__), "compat")
    sys.path.insert(0, compat_root)
    next_path = 1
    if args.dependency_root:
        sys.path.insert(next_path, args.dependency_root)
        next_path += 1
    sys.path.insert(next_path, args.reference_root)
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    os.makedirs(args.out, exist_ok=True)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    load_start = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.model.eval()
    load_seconds = time.perf_counter() - load_start

    captured = {}

    def _capture_generate(*akw, **kw):
        codes, hidden = original_generate(*akw, **kw)
        captured["codes"] = codes
        return codes, hidden

    original_generate = model.model.generate
    model.model.generate = _capture_generate

    synthesis_start = time.perf_counter()
    try:
        wavs, fs = model.generate_custom_voice(
            text=text,
            speaker=args.speaker,
            language=args.language,
            max_new_tokens=max_new_tokens,
            do_sample=not args.greedy,
            subtalker_dosample=not args.greedy,
            seed=args.seed,
        )
    finally:
        model.model.generate = original_generate
    synthesis_seconds = time.perf_counter() - synthesis_start

    suffix = "_greedy" if args.greedy else ""
    codes = captured["codes"][0]  # (T, num_code_groups)
    code_np = codes.detach().cpu().numpy()
    np.save(os.path.join(args.out, f"talker_codes{suffix}.npy"), code_np)

    wav = wavs[0].astype(np.float32)
    np.save(os.path.join(args.out, f"waveform{suffix}.npy"), wav)
    mode = "greedy" if args.greedy else "sampled"
    reference_commit = subprocess.run(
        ["git", "-C", args.reference_root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    codes_path = os.path.join(args.out, f"talker_codes{suffix}.npy")
    waveform_path = os.path.join(args.out, f"waveform{suffix}.npy")
    with open(os.path.join(args.out, f"meta_{mode}.json"), "w") as f:
        json.dump(
            {
                "schema": "mbufo.qwen3-tts-reference.v1",
                "reference_commit": reference_commit,
                "text": text,
                "sample_rate": fs,
                "samples": int(wav.shape[0]),
                "sec": float(wav.shape[0] / fs),
                "code_groups": int(code_np.shape[1]),
                "code_steps": int(code_np.shape[0]),
                "seed": args.seed,
                "max_new_tokens": max_new_tokens,
                "dtype": args.dtype,
                "device": args.device,
                "load_seconds": load_seconds,
                "synthesis_seconds": synthesis_seconds,
                "speaker": args.speaker,
                "greedy": args.greedy,
                "subtalker_greedy": args.greedy,
                "talker_codes_sha256": sha256(codes_path),
                "waveform_sha256": sha256(waveform_path),
                "first_codes": code_np[:5, :6].tolist(),
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    print(f"wrote talker_codes{suffix}.npy shape={code_np.shape}")
    print(f"wrote waveform{suffix}.npy sr={fs} n={wav.shape[0]}")
    print(f"det:{code_np[:5, :6].tolist()}")


if __name__ == "__main__":
    main()
