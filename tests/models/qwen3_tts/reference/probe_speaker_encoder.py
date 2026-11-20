"""Capture official Qwen3-TTS Base speaker-encoder embeddings and stages.

The retained speaker oracle was produced in bfloat16, which is what the official
`extract_speaker_embedding` casts its mel features to. That matters because the
ECAPA-TDNN pools statistics over every frame of the reference clip, so a
bfloat16 accumulation carries far more error than the rest of the port tolerates
elsewhere. Running the same input at float32 separates the official
implementation's own precision noise from a genuine native mismatch, and the
per-stage captures give a bisection target if a mismatch remains.
"""

import argparse
import collections
import os
import struct
import sys

import numpy as np
import torch


def read_wav(path):
    """Minimal RIFF reader. The retained reference clip is 32-bit float, which
    the standard library's `wave` module rejects."""
    with open(path, "rb") as stream:
        data = stream.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SystemExit(f"not a RIFF/WAVE file: {path}")
    offset = 12
    audio_format = channels = rate = bits = 0
    payload = None
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        (size,) = struct.unpack("<I", data[offset + 4 : offset + 8])
        body = data[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt ":
            audio_format, channels, rate = struct.unpack("<HHI", body[:8])
            (bits,) = struct.unpack("<H", body[14:16])
        elif chunk_id == b"data":
            payload = body
        offset += 8 + size + (size & 1)
    if payload is None:
        raise SystemExit(f"no data chunk in {path}")
    if audio_format == 3 and bits == 32:
        samples = np.frombuffer(payload, dtype=np.float32)
    elif audio_format == 1 and bits == 16:
        samples = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise SystemExit(f"unsupported WAV format {audio_format} at {bits} bits")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", default="/home/fbozzo/projects/Qwen3-TTS")
    parser.add_argument(
        "--dependency-root",
        help="site-packages containing the official pinned Qwen dependencies",
    )
    parser.add_argument(
        "--model",
        default=("/home/fbozzo/projects/audio.cpp/models/Qwen3-TTS-12Hz-1.7B-Base"),
    )
    parser.add_argument(
        "--audio",
        default=(
            "/home/fbozzo/projects/mbufo/"
            "artifacts/qwen3_tts/base/reference.wav"
        ),
    )
    parser.add_argument(
        "--out",
        default=(
            "/home/fbozzo/projects/mbufo/"
            "artifacts/qwen3_tts/base/speaker_encoder"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="float32"
    )
    args = parser.parse_args()

    compat_root = os.path.join(os.path.dirname(__file__), "compat")
    sys.path.insert(0, compat_root)
    next_path = 1
    if args.dependency_root:
        sys.path.insert(next_path, args.dependency_root)
        next_path += 1
    sys.path.insert(next_path, args.reference_root)

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.model.eval()

    audio, rate = read_wav(args.audio)
    print(f"audio: samples={audio.size} rate={rate} seconds={audio.size / rate:.4f}")

    captures = collections.OrderedDict()

    def capture(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captures[name] = np.ascontiguousarray(
                tensor.detach().cpu().to(torch.float32).numpy()
            )

        return hook

    encoder = model.model.speaker_encoder

    def capture_input(name):
        def hook(_module, inputs, _output):
            tensor = inputs[0]
            captures[name] = np.ascontiguousarray(
                tensor.detach().cpu().to(torch.float32).numpy()
            )

        return hook

    encoder = model.model.speaker_encoder
    # The mel features are the one stage with no weight shape to validate them,
    # so capture the tensor the first block actually receives.
    handles = [encoder.blocks[0].register_forward_pre_hook(
        lambda module, inputs: capture_input("mel")(module, inputs, None)
    )]
    handles.append(encoder.mfa.register_forward_hook(capture("mfa")))
    handles.append(encoder.asp.register_forward_hook(capture("asp")))
    handles.append(encoder.fc.register_forward_hook(capture("fc")))
    for index, block in enumerate(encoder.blocks):
        handles.append(block.register_forward_hook(capture(f"block{index}")))
    try:
        embedding = model.model.extract_speaker_embedding(audio, rate)
    finally:
        for handle in handles:
            handle.remove()

    values = np.ascontiguousarray(
        embedding.detach().cpu().to(torch.float32).numpy().reshape(-1)
    )
    os.makedirs(args.out, exist_ok=True)
    suffix = "" if args.dtype == "bfloat16" else f"_{args.dtype}"
    np.save(os.path.join(args.out, f"embedding{suffix}.npy"), values)
    print(
        f"embedding{suffix}: shape={values.shape} norm={np.linalg.norm(values):.8g} "
        f"mean={values.mean():.8g} std={values.std():.8g}"
    )
    for name, stage in captures.items():
        np.save(os.path.join(args.out, f"{name}{suffix}.npy"), stage)
        print(
            f"{name}{suffix}: shape={stage.shape} mean={stage.mean():.8g} "
            f"std={stage.std():.8g}"
        )


if __name__ == "__main__":
    main()
