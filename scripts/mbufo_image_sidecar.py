#!/usr/bin/env python3
"""MBufo managed ROCm GPU image generation sidecar (Flux2-klein-4B / Z-Image-Turbo)."""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from diffusers import DiffusionPipeline, Flux2KleinPipeline, ZImagePipeline
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
import uvicorn


def parse_size(size_str: str, fallback: str = "1024x1024") -> tuple[int, int]:
    raw = (size_str or fallback or "1024x1024").strip().lower()
    if "x" not in raw:
        raw = fallback
    try:
        w_s, h_s = raw.split("x", 1)
        w = max(128, min(2048, int(w_s)))
        h = max(128, min(2048, int(h_s)))
    except Exception:
        return (1024, 1024)
    # Round to multiples of 16 for VAE / patchified DiT alignment
    w = max(128, (w // 16) * 16)
    h = max(128, (h // 16) * 16)
    return (w, h)


def resolve_dtype(name: str) -> torch.dtype:
    key = (name or "bf16").strip().lower()
    if key in ("fp16", "float16", "f16"):
        return torch.float16
    if key in ("fp32", "float32", "f32"):
        return torch.float32
    return torch.bfloat16


def load_pipeline(
    model_id: str,
    model_dir: Path,
    dtype_name: str,
    cpu_offload: str,
) -> tuple[Any, str]:
    dtype = resolve_dtype(dtype_name)
    class_name = ""
    mi = model_dir / "model_index.json"
    if mi.is_file():
        try:
            class_name = str(json.loads(mi.read_text(encoding="utf-8")).get("_class_name", ""))
        except Exception:
            class_name = ""

    cls: Any = DiffusionPipeline
    if class_name == "Flux2KleinPipeline" or "flux2-klein" in model_id.lower():
        cls = Flux2KleinPipeline
    elif class_name == "ZImagePipeline" or "z-image" in model_id.lower():
        cls = ZImagePipeline

    pipe = cls.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )

    offload_mode = (cpu_offload or "none").strip().lower()
    if offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload_mode == "model":
        pipe.enable_model_cpu_offload()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        offload_mode = "none"

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe, offload_mode


def decode_b64_or_data_url(raw: str) -> Image.Image:
    s = raw.strip()
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    data = base64.b64decode(s)
    return Image.open(io.BytesIO(data)).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser(description="MBufo GPU Diffusers Image Sidecar")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--default-size", default="1024x1024")
    ap.add_argument("--cpu-offload", default="none")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"[image-sidecar] model directory does not exist: {model_dir}", file=sys.stderr, flush=True)
        return 1

    is_flux = "flux2" in args.model_id.lower() or "flux2" in model_dir.name.lower()
    default_steps = args.steps if args.steps > 0 else (4 if is_flux else 9)
    default_guidance = 1.0 if is_flux else 0.0

    t0 = time.monotonic()
    print(
        f"[image-sidecar] loading model={args.model_id} dir={model_dir} "
        f"dtype={args.dtype} steps={default_steps} size={args.default_size} offload={args.cpu_offload}",
        flush=True,
    )
    pipe, active_offload = load_pipeline(args.model_id, model_dir, args.dtype, args.cpu_offload)
    print(f"[image-sidecar] ready in {time.monotonic() - t0:.2f}s (offload={active_offload})", flush=True)

    gpu_lock = threading.Lock()
    state = {"pipe": pipe, "closing": False}

    def release_resources() -> None:
        state["closing"] = True
        with gpu_lock:
            p = state.get("pipe")
            state["pipe"] = None
            if p is not None:
                try:
                    del p
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def handle_sigterm(_signum: int, _frame: Any) -> None:
        release_resources()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    app = FastAPI(title=f"MBufo Image Sidecar ({args.model_id})")

    @app.get("/health")
    def health() -> dict[str, Any]:
        if state["closing"] or state["pipe"] is None:
            raise HTTPException(status_code=503, detail="shutting down")
        return {
            "status": "ok",
            "ready": True,
            "model": args.model_id,
            "device": "gpu",
            "dtype": args.dtype,
            "steps": default_steps,
            "default_size": args.default_size,
            "cpu_offload": active_offload,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": args.model_id,
                    "object": "model",
                    "owned_by": "mbufo-image-sidecar",
                    "task": "image_generation",
                    "backend": "gpu",
                }
            ],
        }

    def run_generation(
        prompt: str,
        size_str: str,
        n: int,
        steps: int,
        guidance: float,
        seed: Optional[int],
        negative_prompt: Optional[str],
        init_images: Optional[list[Image.Image]] = None,
    ) -> dict[str, Any]:
        width, height = parse_size(size_str, args.default_size)
        n = max(1, min(4, int(n)))
        steps = max(1, min(100, int(steps)))

        with gpu_lock:
            active_pipe = state.get("pipe")
            if active_pipe is None:
                raise HTTPException(status_code=503, detail="pipeline unloaded")

            gen_dev = "cuda" if (torch.cuda.is_available() and active_offload == "none") else "cpu"
            generator = None
            if seed is not None:
                if n == 1:
                    generator = torch.Generator(device=gen_dev).manual_seed(int(seed))
                else:
                    generator = [
                        torch.Generator(device=gen_dev).manual_seed(int(seed) + i)
                        for i in range(n)
                    ]

            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "num_images_per_prompt": n,
            }
            if generator is not None:
                kwargs["generator"] = generator

            if isinstance(active_pipe, Flux2KleinPipeline):
                if init_images:
                    kwargs["image"] = init_images[0] if len(init_images) == 1 else init_images
            elif isinstance(active_pipe, ZImagePipeline):
                if init_images:
                    raise HTTPException(status_code=400, detail=f"{args.model_id} does not support image editing")
                if negative_prompt:
                    kwargs["negative_prompt"] = negative_prompt
            else:
                if negative_prompt:
                    kwargs["negative_prompt"] = negative_prompt

            with torch.inference_mode():
                out = active_pipe(**kwargs)
            images = out.images

            data_rows = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                data_rows.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})

        return {
            "created": int(time.time()),
            "model": args.model_id,
            "data": data_rows,
        }

    @app.post("/v1/images/generations")
    def generate(body: dict[str, Any]) -> dict[str, Any]:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")
        size_str = str(body.get("size") or args.default_size)
        n = int(body.get("n") or 1)
        steps = int(body.get("num_inference_steps") or body.get("steps") or default_steps)
        guidance = float(
            body["guidance_scale"] if "guidance_scale" in body and body["guidance_scale"] is not None else default_guidance
        )
        seed = body.get("seed")
        if seed is not None:
            seed = int(seed)
        negative_prompt = body.get("negative_prompt")
        init_images = None
        if isinstance(body.get("image"), str) and body["image"].strip():
            init_images = [decode_b64_or_data_url(body["image"])]
        return run_generation(prompt, size_str, n, steps, guidance, seed, negative_prompt, init_images)

    @app.post("/v1/images/edits")
    async def edit(request: Request) -> dict[str, Any]:
        ctype = (request.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            body = await request.json()
            prompt = str(body.get("prompt") or "")
            if not prompt.strip():
                raise HTTPException(status_code=400, detail="prompt is required")
            imgs: list[Image.Image] = []
            if isinstance(body.get("image"), str) and body["image"].strip():
                imgs.append(decode_b64_or_data_url(body["image"]))
            elif isinstance(body.get("images"), list):
                for item in body["images"]:
                    if isinstance(item, str) and item.strip():
                        imgs.append(decode_b64_or_data_url(item))
            if not imgs:
                raise HTTPException(status_code=400, detail="image is required for /v1/images/edits")
            return run_generation(
                prompt,
                str(body.get("size") or args.default_size),
                int(body.get("n") or 1),
                int(body.get("num_inference_steps") or body.get("steps") or default_steps),
                float(body.get("guidance_scale") if body.get("guidance_scale") is not None else default_guidance),
                int(body["seed"]) if body.get("seed") is not None else None,
                body.get("negative_prompt"),
                imgs,
            )

        form = await request.form()
        prompt = str(form.get("prompt") or "")
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")
        imgs = []
        for field in ("image", "image[]"):
            for val in form.getlist(field):
                if hasattr(val, "read"):
                    raw = await val.read()
                    if raw:
                        imgs.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        if not imgs:
            raise HTTPException(status_code=400, detail="image upload is required for /v1/images/edits")
        seed_raw = form.get("seed")
        steps_raw = form.get("num_inference_steps") or form.get("steps")
        guidance_raw = form.get("guidance_scale")
        return run_generation(
            prompt,
            str(form.get("size") or args.default_size),
            int(form.get("n") or 1),
            int(steps_raw) if steps_raw else default_steps,
            float(guidance_raw) if guidance_raw is not None else default_guidance,
            int(seed_raw) if seed_raw is not None else None,
            str(form.get("negative_prompt")) if form.get("negative_prompt") else None,
            imgs,
        )

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        release_resources()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())