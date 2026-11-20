#!/usr/bin/env python3
"""Capture official Qwen3-TTS 12 Hz speech-decoder boundaries."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-root", default="/home/fbozzo/projects/Qwen3-TTS"
    )
    parser.add_argument("--dependency-root")
    parser.add_argument(
        "--model",
        default="/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    )
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    args = parser.parse_args()

    compat_root = os.path.join(os.path.dirname(__file__), "compat")
    sys.path.insert(0, compat_root)
    next_path = 1
    if args.dependency_root:
        sys.path.insert(next_path, args.dependency_root)
        next_path += 1
    sys.path.insert(next_path, args.reference_root)

    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(Path(args.model) / "speech_tokenizer"),
        device_map=args.device,
        dtype=dtype,
        attn_implementation="eager",
    )
    decoder = tokenizer.model.decoder
    device = next(decoder.parameters()).device
    codes_np = np.load(args.codes, allow_pickle=False)
    if codes_np.ndim != 2 or codes_np.shape[1] != 16:
        raise ValueError(f"expected [frames,16] codes, got {codes_np.shape}")
    codes = torch.from_numpy(codes_np.astype(np.int64)).to(device)
    codes = codes.transpose(0, 1).unsqueeze(0)

    capture: dict[str, np.ndarray] = {}

    def dump(name: str, tensor: torch.Tensor) -> None:
        capture[name] = tensor.detach().cpu().to(torch.float32).numpy()

    original_quantizer_decode = decoder.quantizer.decode

    def quantizer_decode(value: torch.Tensor) -> torch.Tensor:
        output = original_quantizer_decode(value)
        dump("quantizer_output", output)
        return output

    decoder.quantizer.decode = quantizer_decode
    hooks = []
    hooks.append(
        decoder.pre_conv.register_forward_hook(
            lambda _module, _inputs, output: dump("pre_conv_output", output)
        )
    )
    hooks.append(
        decoder.pre_transformer.register_forward_hook(
            lambda _module, _inputs, output: dump(
                "transformer_output", output.last_hidden_state
            )
        )
    )
    for stage_index, stage in enumerate(decoder.upsample):
        for block_index, block in enumerate(stage):
            hooks.append(
                block.register_forward_hook(
                    lambda _module, _inputs, output, name=(
                        f"upsample_{stage_index}_{block_index}_output"
                    ): dump(name, output)
                )
            )
    for block_index, block in enumerate(decoder.decoder):
        hooks.append(
            block.register_forward_hook(
                lambda _module, _inputs, output,
                name=f"decoder_{block_index}_output": dump(name, output)
            )
        )

    try:
        with torch.inference_mode():
            waveform = decoder(codes)
        dump("waveform", waveform)
    finally:
        decoder.quantizer.decode = original_quantizer_decode
        for hook in hooks:
            hook.remove()

    args.out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    shapes = {}
    for name, value in capture.items():
        path = args.out / f"{name}.npy"
        np.save(path, value)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shapes[name] = list(value.shape)
        print(f"{name}: {value.shape}")

    reference_commit = subprocess.run(
        ["git", "-C", args.reference_root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "schema": "mbufo.qwen3-tts-speech-decoder-probe.v1",
        "reference_commit": reference_commit,
        "device": args.device,
        "dtype": args.dtype,
        "codes": str(args.codes),
        "code_steps": int(codes_np.shape[0]),
        "shapes": shapes,
        "sha256": hashes,
    }
    (args.out / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
