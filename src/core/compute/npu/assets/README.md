# NPU2 IRON Kernel Assets (EmbeddingGemma 2)

Silicon-verified IRON builders for AMD Ryzen AI MAX+ 395 (`RyzenAI-npu5`, AIE2p, XDNA2):

## Files
- `e06_ffn.py`: EmbeddingGemma 2 24-layer FFN chain (`M=8, D=512, F=2048, bf16`, `rms -> up/gate -> gelu -> down -> add`), spatial `P=2 64x64`.
  - Silicon result: 24-layer chain `3.02 ms` (`125.7 us/L`), `ch8 = 122.3 us/L (34.4 GB/s, 274.3 GFLOPS)`, `max_abs = 0.1250`, `cos = 0.9999754`.
- `e07_attn.py`: EmbeddingGemma 2 24-layer Attention + 4 standalone sub-kernels (`RoPE`, `QK^T`, `Softmax+SWA`, `PV`) + fused 20/28-core 4-stage / 8-XCLBIN pipeline.
  - Architecture: 20 local sliding layers (`W=512, d=256, H_q=4, H_kv=2`) + 4 full global layers (0-based indices `{5, 11, 17, 23}`, `d=512, H_q=4, H_kv=1`), **bidirectional**, **unscaled (`scale=1.0`)**.
  - Why not `linalg.prefill_fv`: SDK `flash_attn_prefill.cc` and `mha.cc` hardcode causal masking and `1/sqrt(d)` scaling, whereas EmbeddingGemma 2 (`src/models/embeddinggemma2/hip/session.hip`) uses bidirectional sliding attention with `alpha = 1.0`.
  - Silicon results:
    - `S=128`: 24L chained stage sum `17.387 ms` (`max=0.1016, cos=0.9998865`); heterogeneous `run_chain(96)` `npu_time=133.217 ms`.
    - `S=512`: 24L `max=0.1406, cos=0.9998494`, chained sum `119.578 ms`, `run_chain(96)` `npu_time=234.644 ms`.
- `safetensors_npu_bridge.py`: Chunked reader & shard planner + `bfp16ebs8` / `bf16` prepacker for `/mnt/nvme-data/mbufo-models/embeddinggemma-2/model.safetensors` (`1.49 GB`).

## Usage
```bash
~/venv-npu/bin/python src/core/compute/npu/assets/e07_attn.py --subkernels
~/venv-npu/bin/python src/core/compute/npu/assets/e07_attn.py --chain24
```
Note: Requires `src/core/compute/npu/patches/aie_linalg_aie2p_conv_even.patch` applied in `~/venv-npu` so non-emulated `bf16` outputs use `conv_even` rounding instead of floor rounding.
