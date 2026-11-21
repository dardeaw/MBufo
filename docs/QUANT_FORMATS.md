# Quantization formats per engine

Which GGUF tensor formats each engine loads and how. A format outside an engine's list is refused at load time with
`unsupported ... type` (the name of the tensor and the format are in the message); nothing is decoded as another format.

| Engine | Dense / attention tensors | Routed experts | Notes |
| --- | --- | --- | --- |
| llama, glimmer, gemma4 (dense part) | F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K | gemma4: same list; Q4_0 takes the fast MoE path | Embedding gather accepts every format in the shared registry (below) |
| gpt-oss | `output.weight` Q8_0; attention Q8_0 / Q5_0 / Q4_K / ... via load-time conversion | MXFP4 (native, fused decode), Q4_0, Q8_0, Q5_1 (core lane-geometry MoE + routed GEMM); Q4_K / Q5_K only where the row width is a multiple of 256 | gpt-oss rows are 2880 wide: **K-quants cannot encode its experts at all** (2880 / 256 is not an integer), so a "Q4_K_M" gpt-oss keeps MXFP4 experts |
| Flash-Next | Q8_0, BF16, F16, F32, Q6_K, ROCmFP4, ROCmFP4_FAST, MXFP4 natively; **Q4_K, Q5_K, Q4_0, Q4_1, Q5_0, Q2_K, Q3_K, IQ4_XS, IQ4_NL** converted to Q8_0 at upload | Q4_K, Q5_K, Q5_1, Q8_0, ROCmFP4_FAST, MXFP4, **Q4_0, Q2_K, Q3_K, Q6_K** | Converted dense tensors use more memory than their source format (a Q4_K matrix becomes Q8_0); experts stay packed |
| Qwen 27B / qwen35moe | Q8_0, Q4_K, Q5_K, Q6_K, Q3_K, IQ4_NL, IQ4_XS, IQ3_S, ROCmFP4, ROCmFP4_FAST and F32 / F16 / BF16 (own in-kernel decoders) | the shared row-dot formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, MXFP4, ROCmFP4, ROCmFP4_FAST) | Q6_K expert down tensors are converted to Q8_0 at load |
| DeepSeek V4 Flash | the published mixed IQ2_XXS / Q2_K / Q8_0 recipe only | same | its kernels stay bit-identical to the ds4 reference |

## Shared registry

`core/quant/quant_traits.hpp` lists the formats every shared kernel understands (embedding gather, row dequantization,
the row-dot layer, the host dequantizer `core/compute/gemm/host_dequant.hpp`): F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K,
**Q3_K**, Q4_K, Q5_K, Q6_K, **IQ4_NL**, **IQ4_XS**, IQ2_XXS, MXFP4, ROCmFP4, ROCmFP4_FAST, ROCmFP3. Adding a format means one `Traits`
specialization checked element for element against the ggml oracle (`tools/bench/quant_traits_check`); every engine that
reaches the shared layer then gets it.

## How this was verified

* gpt-oss: Q4_0 experts (`gptoss-A-q4exp.gguf`) and Q8_0 experts (requantized from the Q4_K_M file), greedy tokens against llama.cpp.
* Flash-Next: dense Q4_K / Q5_K / Q4_0 / Q3_K / Q2_K / Q4_1 / Q5_0 / IQ4_NL mixed, and experts Q4_0 / Q3_K / Q2_K / Q6_K, each file
  requantized from the UD-Q4_K_XL build with an architecture-agnostic requantizer (llama-quantize does not know the arch), run through
  the engine, then deleted.
* Qwen 27B: an all-IQ4_XS file (4.51 BPW) against llama.cpp: 13.9 vs 14.0 tok/s.
