#!/usr/bin/env python3
"""Convert an EmbeddingGemma 2 checkpoint (HF safetensors) to a MBufo GGUF.

usage:
  embeddinggemma2_to_gguf.py <model_dir> <tokenizer_donor.gguf> <out.gguf>

<model_dir>            directory holding config.json + model.safetensors
<tokenizer_donor.gguf> any Gemma 4 GGUF; EmbeddingGemma 2 shares the Gemma 4
                       vocabulary, so its tokenizer.ggml.* keys are copied.

Only the text path is exported (language_model.* tensors). Projection
matrices and the embedding table stay BF16; norm weights and layer_scalar
are stored as F32. Tensor names are the HF names with the
"language_model." prefix removed.
"""
import json
import sys

import numpy as np
import torch
from gguf import GGUFReader, GGUFValueType, GGMLQuantizationType, GGUFWriter
from safetensors import safe_open

ARCH = "embedding_gemma2"


def main():
    model_dir, donor_path, out_path = sys.argv[1:4]
    cfg = json.load(open(model_dir + "/config.json"))["text_config"]
    per_layer = {int(k): v for k, v in cfg["per_layer_config"].items()}
    layer_types = cfg["layer_types"]
    n_layers = cfg["num_hidden_layers"]
    full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    local_head_dim = cfg["head_dim"]
    local_kv = cfg["num_key_value_heads"]
    full_head_dim = per_layer[full_layers[0]]["head_dim"]
    full_kv = per_layer[full_layers[0]]["num_key_value_heads"]
    for i in range(n_layers):
        pl = per_layer.get(i)
        if (i in full_layers) != (pl is not None):
            sys.exit("per_layer_config must cover exactly the full-attention layers")
        if pl and (pl["head_dim"] != full_head_dim or pl["num_key_value_heads"] != full_kv):
            sys.exit("non-uniform full-attention layer geometry is not supported")

    w = GGUFWriter(out_path, ARCH)
    w.add_name("EmbeddingGemma 2")
    w.add_uint32(f"{ARCH}.block_count", n_layers)
    w.add_uint32(f"{ARCH}.embedding_length", cfg["hidden_size"])
    w.add_uint32(f"{ARCH}.feed_forward_length", cfg["intermediate_size"])
    w.add_uint32(f"{ARCH}.attention.head_count", cfg["num_attention_heads"])
    w.add_uint32(f"{ARCH}.attention.head_dim_local", local_head_dim)
    w.add_uint32(f"{ARCH}.attention.head_dim_full", full_head_dim)
    w.add_uint32(f"{ARCH}.attention.head_count_kv_local", local_kv)
    w.add_uint32(f"{ARCH}.attention.head_count_kv_full", full_kv)
    w.add_uint32(f"{ARCH}.attention.sliding_window", cfg["sliding_window"])
    w.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
    w.add_float32(f"{ARCH}.rope.freq_base_local",
                  cfg["rope_parameters"]["sliding_attention"]["rope_theta"])
    w.add_float32(f"{ARCH}.rope.freq_base_full",
                  cfg["rope_parameters"]["full_attention"]["rope_theta"])
    w.add_uint32(f"{ARCH}.per_layer_input_dim", cfg["hidden_size_per_layer_input"])
    w.add_uint32(f"{ARCH}.output_embedding_length", cfg["embedding_dim"])
    w.add_uint32(f"{ARCH}.context_length", 8192)
    # Embedding scale is sqrt(hidden) rounded to bf16, as the bf16 reference does.
    scale = float(torch.tensor(cfg["hidden_size"] ** 0.5, dtype=torch.bfloat16).float())
    w.add_float32(f"{ARCH}.embed_scale", scale)
    period = full_layers[0] + 1
    if full_layers != [period * (j + 1) - 1 for j in range(len(full_layers))] or n_layers % period:
        sys.exit("full-attention layers must repeat with a fixed period")
    w.add_uint32(f"{ARCH}.attention.full_layer_period", period)
    w.add_uint32(f"{ARCH}.pooling_type", 1)  # mean

    # Tokenizer keys from the Gemma 4 donor.
    donor = GGUFReader(donor_path)
    for key, field in donor.fields.items():
        if not key.startswith("tokenizer.") or key == "tokenizer.chat_template":
            continue
        types = field.types
        if types[0] == GGUFValueType.ARRAY:
            w.add_key_value(key, field.contents(), GGUFValueType.ARRAY, sub_type=types[-1])
        else:
            w.add_key_value(key, field.contents(), types[0])

    with safe_open(model_dir + "/model.safetensors", framework="pt") as f:
        names = sorted(k for k in f.keys() if k.startswith("language_model."))
        for k in names:
            t = f.get_tensor(k)
            name = k[len("language_model."):]
            if t.dtype == torch.bfloat16 and t.ndim == 2:
                raw = t.contiguous().view(torch.int16).numpy().view(np.uint8)
                w.add_tensor(name, raw.reshape(t.shape[0], -1), raw_dtype=GGMLQuantizationType.BF16)
            else:
                w.add_tensor(name, t.float().contiguous().numpy().reshape(-1) if t.ndim == 0
                             else t.float().contiguous().numpy())
            print("tensor", name, tuple(t.shape), "BF16" if t.ndim == 2 and t.dtype == torch.bfloat16 else "F32")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", out_path)


if __name__ == "__main__":
    main()
