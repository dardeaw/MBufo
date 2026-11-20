#!/usr/bin/env python3
"""Capture the external Qwen3-VL layer-50 tensor consumed by MiniMax H3.

This tool is intentionally outside the production package. It loads only the
text embedding and requested decoder layers from an operator-supplied pinned
H3 checkpoint, captures the hidden state before the text model's final norm,
and writes a raw little-endian BF16 oracle plus provenance metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

# Executing a script inside tools/mbufo places that directory first on
# sys.path, where the project's helper safetensors.py would shadow the
# third-party safetensors package used by this isolated teacher.
_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != _SCRIPT_DIRECTORY
]

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel,
    Qwen3VLTextRotaryEmbedding,
)


MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
REFERENCE_REVISION = "8974cc055ea9c02fcd14cc27dfda3e1027c05153"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt", default="A red fox walking through snow"
    )
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def set_parameter(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    parts = name.split(".")
    parent: torch.nn.Module = module
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf not in parent._parameters:
        raise KeyError(f"state entry is not a parameter: {name}")
    parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_text_model(
    model_root: Path, layers: int, device: str
) -> Qwen3VLTextModel:
    text_root = model_root / "FL2VA" / "text_encoder"
    config_document = json.loads((text_root / "config.json").read_text())
    config_data = dict(config_document["text_config"])
    if not 1 <= layers <= int(config_data["num_hidden_layers"]):
        raise ValueError("layers must be within the released text tower")
    config_data["num_hidden_layers"] = layers
    config = Qwen3VLTextConfig(**config_data)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3VLTextModel(config)
    model.rotary_emb = Qwen3VLTextRotaryEmbedding(config, device=device)

    index = json.loads(
        (text_root / "model.safetensors.index.json").read_text()
    )["weight_map"]
    requested = list(model.state_dict())
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for state_name in requested:
        checkpoint_name = f"model.language_model.{state_name}"
        shard = index.get(checkpoint_name)
        if shard is None:
            raise KeyError(f"checkpoint tensor is absent: {checkpoint_name}")
        by_shard.setdefault(shard, []).append((state_name, checkpoint_name))

    loaded = 0
    total = len(requested)
    for shard_name in sorted(by_shard):
        with safe_open(
            text_root / shard_name, framework="pt", device=device
        ) as shard:
            for state_name, checkpoint_name in by_shard[shard_name]:
                tensor = shard.get_tensor(checkpoint_name)
                if tensor.dtype != torch.bfloat16:
                    raise TypeError(
                        f"{checkpoint_name} is {tensor.dtype}, expected BF16"
                    )
                set_parameter(model, state_name, tensor)
                loaded += 1
        print(
            f"loaded {loaded}/{total} tensors through {shard_name}",
            file=sys.stderr,
            flush=True,
        )
    model.eval()
    return model


def write_bf16(path: Path, tensor: torch.Tensor) -> None:
    values = tensor.detach().contiguous().view(torch.uint16).cpu().flatten()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        for chunk in values.split(1 << 20):
            output.write(struct.pack(f"<{chunk.numel()}H", *chunk.tolist()))


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm torch device is unavailable")
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root / "FL2VA" / "tokenizer",
        local_files_only=True,
        trust_remote_code=False,
    )
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [151643]

    started = time.monotonic()
    model = load_text_model(args.model_root, args.layers, args.device)
    captured: dict[str, torch.Tensor] = {}

    def capture_before_final_norm(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        captured["hidden"] = inputs[0].detach()

    hook = model.norm.register_forward_pre_hook(capture_before_final_norm)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    hook.remove()
    hidden = captured["hidden"].squeeze(0)
    if hidden.dtype != torch.bfloat16:
        hidden = hidden.to(torch.bfloat16)
    if hidden.shape != (len(token_ids), 5120):
        raise RuntimeError(f"unexpected hidden-state shape: {tuple(hidden.shape)}")
    if not torch.isfinite(hidden.float()).all():
        raise RuntimeError("golden hidden state contains non-finite values")

    write_bf16(args.output, hidden)
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata = {
        "schema": "mbufo.minimax-h3-prompt-golden.v1",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "teacher": {
            "framework": "transformers",
            "version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "attention": "eager",
            "device": torch.cuda.get_device_name(),
        },
        "prompt": args.prompt,
        "token_ids": token_ids,
        "layers": args.layers,
        "boundary": "input-to-final-rmsnorm",
        "dtype": "BF16",
        "shape": list(hidden.shape),
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "elapsed_seconds": time.monotonic() - started,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
