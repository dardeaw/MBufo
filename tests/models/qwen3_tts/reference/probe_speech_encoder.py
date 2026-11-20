"""Capture official Qwen3-TTS 12 Hz speech-encoder intermediates.

The probe is intentionally separate from generation so the native encoder can
be compared stage by stage without running the talker or speech decoder.
"""

import argparse
import collections
import os
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-root", default="/home/fbozzo/projects/Qwen3-TTS"
    )
    parser.add_argument(
        "--dependency-root",
        help="site-packages containing the official pinned Qwen dependencies",
    )
    parser.add_argument(
        "--model",
        default=(
            "/home/fbozzo/projects/audio.cpp/models/"
            "Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer"
        ),
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
            "artifacts/qwen3_tts/base/speech_encoder"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
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

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
        attn_implementation="eager",
    )
    tokenizer.model.eval()

    captures = collections.OrderedDict()

    def capture(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if hasattr(tensor, "last_hidden_state"):
                tensor = tensor.last_hidden_state
            captures[name] = np.ascontiguousarray(
                tensor.detach().cpu().to(torch.float32).numpy()
            )

        return hook

    encoder = tokenizer.model.encoder
    handles = [
        encoder.encoder.register_forward_hook(capture("convolutional")),
    ]
    # The lumped "convolutional" stage covers six strided convolutions and four
    # residual units, which is too coarse to localise a mismatch. Capture each
    # SEANet layer so the native trace can be compared one layer at a time.
    for index, layer in enumerate(encoder.encoder.layers):
        handles.append(layer.register_forward_hook(capture(f"layer{index:02d}")))
    handles += [
        encoder.encoder_transformer.register_forward_hook(
            capture("transformer")
        ),
        encoder.downsample.register_forward_hook(capture("downsample")),
        encoder.quantizer.semantic_residual_vector_quantizer.input_proj
        .register_forward_hook(capture("semantic_projection")),
        encoder.quantizer.acoustic_residual_vector_quantizer.input_proj
        .register_forward_hook(capture("acoustic_projection")),
    ]
    try:
        encoded = tokenizer.encode(args.audio)
    finally:
        for handle in handles:
            handle.remove()

    os.makedirs(args.out, exist_ok=True)
    # Keep the bfloat16 capture on the retained filenames and suffix anything
    # else, so a float32 reference can sit beside it instead of replacing it.
    suffix = "" if args.dtype == "bfloat16" else f"_{args.dtype}"
    for name, values in captures.items():
        path = os.path.join(args.out, f"{name}{suffix}.npy")
        np.save(path, values)
        print(
            f"{name}: shape={values.shape} "
            f"mean={values.mean():.8g} std={values.std():.8g}"
        )
    codes = encoded.audio_codes[0].detach().cpu().numpy().astype(np.int32)
    np.save(os.path.join(args.out, f"codes{suffix}.npy"), codes)
    print(f"codes: shape={codes.shape}")


if __name__ == "__main__":
    main()
