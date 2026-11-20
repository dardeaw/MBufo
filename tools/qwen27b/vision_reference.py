#!/usr/bin/env python3
"""Check both Qwen3.8 vision encoders against pinned, unmodified HF operators.

The same BF16 GGUF weights are used on both sides. This validates operator
execution and binding, not the upstream GGUF conversion or language quantization.
No model or source download is implicit. Run inside nix develop.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvf

ROOT = Path(__file__).resolve().parents[2]
REVISION = "3713bd839e580d07e4b70f2c89e986cb3c0e8ddf"
SOURCE_HASHES = {
    "vision_utils.py": "bcecd5a92b3266b9926272a549d2b1a0f1fe7646c698c0fa19bd96f976085356",
    "models/qwen3_5/modeling_qwen3_5.py": "832e14a5c8d193f0904cf0f41f891a121eddeffd46201ccc044d0f86d527ce30",
    "models/qwen4_exp/modeling_qwen4_exp.py": "797a18fd6dd76c574d237a5643759acdeb1c4d0f1c2508693f8fdabce0a19057",
    "models/qwen2_vl/image_processing_qwen2_vl.py": "4e1da45f9e7e157ca08aa88243fcc1495ea6a333e40e0c5035ef17ef2747a97d",
}


class RemoveFrameworkDecorators(ast.NodeTransformer):
    """Remove documentation/dispatch hooks; leave every operator body intact."""

    def visit_FunctionDef(self, node):
        node.decorator_list = [
            decorator for decorator in node.decorator_list
            if isinstance(decorator, ast.Name)
            and decorator.id in ("staticmethod", "classmethod")]
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        node.decorator_list = []
        return self.generic_visit(node)


def execute(nodes, path, environment):
    tree = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
        *nodes], type_ignores=[])
    tree = ast.fix_missing_locations(RemoveFrameworkDecorators().visit(tree))
    exec(compile(tree, str(path), "exec"), environment)


def selected(path, predicate, environment):
    execute([node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and predicate(node.name)], path, environment)


class ModelBase(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config

    def post_init(self):
        pass  # Every parameter is loaded explicitly, with strict=True.


class AttentionRegistry:
    @staticmethod
    def get_interface(name, fallback):
        if name != "eager":
            raise ValueError("this operator reference requires eager attention")
        return fallback


def source_model(source, output_width):
    directory = source / "src/transformers"
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} differs from Transformers {REVISION}")
    prefix = "Qwen3_5" if output_width == 5120 else "Qwen4Exp"
    family = "qwen3_5" if output_width == 5120 else "qwen4_exp"
    environment = dict(
        torch=torch, nn=nn, F=F, GradientCheckpointingLayer=nn.Module,
        ACT2FN={"gelu_pytorch_tanh": lambda value: F.gelu(value, approximate="tanh")},
        ALL_ATTENTION_FUNCTIONS=AttentionRegistry,
        is_flash_attention_requested=lambda config: False,
        get_max_seqlen=lambda *a, **k: None,
        maybe_autocast=lambda *a, **k: nullcontext(),
        BaseModelOutputWithPooling=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    environment[prefix + "PreTrainedModel"] = ModelBase
    selected(directory / "vision_utils.py", lambda name: name in {
        "get_vision_position_ids", "get_vision_interpolation_indices_and_weights",
        "_interpolation_axis_taps_weights", "get_vision_cu_seqlens",
        "get_vision_attention_seqlens"}, environment)
    selected(directory / f"models/{family}/modeling_{family}.py",
             lambda name: name.startswith(prefix + "Vision") or name in {
                 "rotate_half", "repeat_kv", "eager_attention_forward",
                 "apply_rotary_pos_emb_vision"}, environment)
    processor = directory / "models/qwen2_vl/image_processing_qwen2_vl.py"
    definition = next(node for node in ast.parse(processor.read_text()).body
                      if isinstance(node, ast.ClassDef)
                      and node.name == "Qwen2VLImageProcessor")
    execute([node for node in definition.body
             if isinstance(node, ast.FunctionDef) and node.name == "patchify"],
            processor, environment)
    return environment, prefix


def load_weights(path, source):
    spec = importlib.util.spec_from_file_location("vision_gguf", ROOT / "tools/mbufo/gguf.py")
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    metadata = codec.parse_gguf(path)
    tensors = {tensor["name"]: tensor for tensor in metadata["tensors"]}
    width = metadata["user_kv"]["clip.vision.projection_dim"]
    if width not in (2560, 5120):
        raise ValueError("expected a Qwen3.8-27B or Flash-Next BF16 vision sidecar")
    environment, prefix = source_model(source, width)
    config = SimpleNamespace(
        hidden_size=1152, intermediate_size=4304, hidden_act="gelu_pytorch_tanh",
        patch_size=16, temporal_patch_size=2, in_channels=3, spatial_merge_size=2,
        num_heads=16, num_attention_heads=16, depth=27, num_position_embeddings=2304,
        out_hidden_size=width, rope_parameters={"rope_type": "axial", "rope_theta": 10000},
        _attn_implementation="eager",
    )
    with torch.device("meta"):
        model = environment[prefix + "VisionModel"](config)

    def read(name):
        tensor = tensors[name]
        dtype = np.float32 if tensor["type"] == 0 else np.uint16
        if tensor["type"] not in (0, 30):
            raise ValueError("the vision reference requires an unquantized BF16 sidecar")
        values = np.memmap(path, mode="r", dtype=dtype,
                           offset=metadata["data_offset"] + tensor["offset"],
                           shape=tuple(tensor["shape"]))
        value = torch.from_numpy(np.array(values))
        if dtype == np.uint16:
            value = value.view(torch.bfloat16)
        return value.to(device="cuda", dtype=torch.bfloat16)

    state = {}
    for name in model.state_dict():
        if name == "patch_embed.proj.weight":
            state[name] = torch.stack((read("v.patch_embd.weight"),
                                       read("v.patch_embd.weight.1")), dim=2)
            continue
        key = name.replace("pos_embed.", "v.position_embd.").replace(
            "patch_embed.proj.", "v.patch_embd.")
        if name.startswith("blocks."):
            _, layer, suffix = name.split(".", 2)
            for old, new in (("attn.qkv", "attn_qkv"), ("attn.proj", "attn_out"),
                             ("mlp.linear_fc1", "ffn_up"), ("mlp.linear_fc2", "ffn_down"),
                             ("norm1", "ln1"), ("norm2", "ln2")):
                suffix = suffix.replace(old, new)
            key = f"v.blk.{layer}.{suffix}"
        elif name.startswith("merger."):
            key = name.replace("merger.norm", "v.post_ln").replace(
                "merger.linear_fc1", "mm.0").replace("merger.linear_fc2", "mm.2")
        state[name] = read(key)
    model.load_state_dict(state, strict=True, assign=True)
    model.rotary_pos_emb = environment[prefix + "VisionRotaryEmbedding"](config, device="cuda")
    return model.eval(), environment


def difference(actual, reference):
    actual = np.asarray(actual, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ValueError("vision trace has a wrong shape or nonfinite values")
    norm = max(np.linalg.norm(reference), 1e-20)
    return {
        "max_abs": float(np.abs(actual - reference).max()),
        "relative_l2": float(np.linalg.norm(actual - reference) / norm),
        "cosine": float(np.dot(actual, reference) /
                        max(np.linalg.norm(actual) * norm, 1e-20)),
    }


@torch.inference_mode()
def compare(args):
    with args.mmproj.open("rb") as source:
        identity = hashlib.file_digest(source, "sha256").hexdigest()
    if (args.trace / "projector.sha256").read_text().strip() != identity:
        raise ValueError("projector fingerprint differs from SHA-256 of its contents")
    model, environment = load_weights(args.mmproj, args.source)
    width, height = map(int, (args.trace / "shape.txt").read_text().split())
    original = np.array(ImageOps.exif_transpose(Image.open(args.image)).convert("RGB"))
    reference_rgb = tvf.resize(
        torch.from_numpy(original).permute(2, 0, 1), [height, width],
        interpolation=InterpolationMode.BICUBIC, antialias=True).permute(1, 2, 0).numpy()
    actual_rgb = np.fromfile(args.trace / "resized.rgb", dtype=np.uint8).reshape(height, width, 3)
    if not np.array_equal(actual_rgb, reference_rgb):
        raise ValueError("decoded/resized pixels differ from Pillow/Torchvision")
    pixels = torch.from_numpy(reference_rgb.copy()).permute(2, 0, 1)[None].cuda().float()
    pixels = (pixels / 255 - 0.5) / 0.5
    patches, grid_h, grid_w = environment["patchify"](None, pixels, 16, 2, 2)
    patches = patches.flatten(0, 1)
    grid = torch.tensor([[1, grid_h, grid_w]], device="cuda")
    reports, handles, inputs = {}, [], {}

    def array(value):
        return value.float().cpu().numpy()

    def trace(name):
        return np.fromfile(args.trace / f"{name}.f32", dtype=np.float32)

    handles.append(model.blocks[0].register_forward_pre_hook(
        lambda module, args: reports.update(patch=difference(trace("patch"), array(args[0])))))
    for index, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(
            lambda module, args, output, i=index: reports.update(
                {f"layer{i}": difference(trace(f"layer{i}"), array(output))})))
        if index in (0, 8, 26):
            handles.append(block.register_forward_pre_hook(
                lambda module, args, kwargs, i=index: inputs.update({i: kwargs}),
                with_kwargs=True))
    bf16 = array(model(patches, grid).pooler_output)
    reports["embedding"] = difference(trace("embedding"), bf16)
    for handle in handles:
        handle.remove()
    isolated = {}
    for index, kwargs in inputs.items():
        prior = "patch" if index == 0 else f"layer{index - 1}"
        value = torch.from_numpy(trace(prior).reshape(-1, 1152)).cuda().bfloat16()
        isolated[f"layer{index}"] = difference(
            trace(f"layer{index}"), array(model.blocks[index](value, **kwargs)))
    # An FP32 run provides an independent numerical control. Compare MBufo's
    # distance from it with the original BF16 implementation's rounding error.
    model.float()
    fp32 = array(model(patches, grid).pooler_output)
    mbufo_fp32 = difference(trace("embedding"), fp32)
    bf16_fp32 = difference(bf16, fp32)
    passed = (
        max(stage["relative_l2"] for stage in reports.values()) < 0.05
        and max(stage["relative_l2"] for stage in isolated.values()) < 0.015
        and mbufo_fp32["relative_l2"] <= 1.25 * bf16_fp32["relative_l2"] + 0.001
    )
    return dict(passed=passed, source_revision=REVISION, pixel_exact=True,
                stages=reports, isolated_layers=isolated,
                mbufo_vs_fp32=mbufo_fp32, upstream_bf16_vs_fp32=bf16_fp32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="pinned Transformers checkout")
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = compare(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "passed", "pixel_exact", "isolated_layers", "mbufo_vs_fp32", "upstream_bf16_vs_fp32")}))
    if not report["passed"]:
        raise SystemExit("vision operator qualification failed")


if __name__ == "__main__":
    main()
