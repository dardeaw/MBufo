# MBufo

专为 AMD Strix Halo（`gfx1151`）打造的高并发推理引擎，面向生产环境，在公平调度下同时服务多种模型与多种量化格式，并以 FlashAttention、Split-K 解码注意力和高效的 KV 缓存压缩节省内存。

最初源自 [gufo](https://github.com/gufo-org/gufo)，之后已大幅重写。

> [English Version](README.md) · [繁體中文版](README.zh-Hant.md)
<img width="1429" height="829" alt="image" src="https://github.com/user-attachments/assets/562fcc09-a6dd-46eb-b143-6d702f5aa08c" />

**核心特点**

- **高并发**：单台 Strix Halo 节点在 16 路以上并发下保持满载，每路最长可达 256K 上下文（§1）。
- **公平调度**：预填充与解码统一混合批处理，配合防饥饿调度，高负载下不会有请求被饿死（§1）。
- **多模型单端口**：20 个模型家族并行服务，可通过内置 `/admin` 控制台在运行期加载与卸载（§3）。
- **多种量化格式**：K 系列、IQ4、MXFP4、ROCmFP4 等，各引擎的支持清单见[支持的量化格式](#支持的量化格式)（§3）。
- **FlashAttention 与 Split-K 注意力**：WMMA FlashAttention 预填充与 Split-K 解码注意力，让长上下文解码从短提示到 256K 都保持在内存带宽上限附近（§2）。
- **高效 KV 缓存**：分块 INT8 与 4-bit 紧凑布局，显著降低每个会话占用的内存（§2）。
- **低精度计算**：以量化权重降低内存带宽需求与成本（§2）。
- **投机解码**：支持 MTP、DFlash、DSpark（§1）。

**目录：** [核心特性](#mbufo-核心特性) ·
[§1 高并发架构](#1-高并发架构) ·
[§2 低精度计算与成本优化](#2-低精度计算与成本优化) ·
[§3 GGUF 全家族支持与多模型路由](#3-gguf-全家族支持与多模型路由) ·
[快速开始](#快速开始) ·
[性能基准](#性能基准-gfx1151-strix-halo) ·
[上游追踪](#上游追踪) ·
[从源码构建](#从源码构建) ·
[文档索引](#文档索引) ·
[兼容性与限制](#兼容性与限制)

---

## MBufo 核心特性

| 特性 | 指标 | 说明 |
| --- | --- | --- |
| **极速响应** | **短请求 TTFT 0.7 秒** | 具备防饥饿公平调度机制，短请求即便遭遇 25 万 token 超长 Prefill 并发，首字延迟仍稳定压在 0.7 秒内，保证输出位元精确无失真。 |
| **极致省容** | **KV 缓存显存减半** | Blocked-Q8 缓存架构让单一 262,144 (256K) token 请求的 KV 占用从 6.44 GB 骤降至 3.42 GB（4-bit 模式更可低至 1.81 GB）。 |
| **全能托管** | **20 个 GGUF 模型家族** | 单一 HTTP 端口统筹服务，涵盖 Flash-Next、27B、Gemma 4、MoE 模型、Embedding 与 Reranker，支持动态热加载与无重启卸载。 |

> §1 与 §2 中除 llama.cpp 对比以外的数据（混合批次、公平性、并发引擎对比、int8 WMMA）来自早期版本，记录于 `docs/MIXED_BATCH.md` 与 `docs/KV-V4.md`。llama.cpp 对比与「基准测试」一节已在本版本重新测量。

---

## 1. 高并发架构

在高并发服务场景中，吞吐量并非唯一指标。系统能否在满载下保持高可用，关键取决于两大核心要素：

- **ITL（Inter-Token Latency，Token 间延迟）**：请求进入流式生成后的输出间隔。若为批次吞吐而让流在 Prefill 阶段长时间冻结，将严重损害交互体验。
- **调度公平性（Fairness）**：严格保障不同请求、Session 与客户端之间的公平性，避免长 Prompt 独占硬件算力导致短对话饥饿中断。

上游调度具备分块 Prefill（`--prefill-chunk`）与优先 Decode 的基础架构，但在两者切换时，传统做法需要分别对模型权重进行两次扫描，造成调度空隙与延迟损耗。MBufo 通过深度融合解决了这个瓶颈。

### 统一混合批处理（Unified Mixed-Batch）

MBufo 将 Prefill Chunk 与 Decode 步骤合并至同一个 `ForwardBatch` 与注意力内核中执行，每轮计算在推进 Prefill 的同时立即输出到期的 Decode Token。

- **延迟缩短 2 倍**：流式请求在 5 批次下的 Token 间隔由 121 ms 缩短至 **57 ms**（仅需单次权重扫描）。
- **满载稳定**：16 路 sustained 并发下，平均耗时由 17.0 秒缩减至 **15.3 秒**，尾部延迟大幅收敛。
- **硬件高利用率**：16 路满载运行时 GPU 利用率维持在 99%，输出完全位元精确、零错误。
- **批次上限扩展**：支持高达 32 Sessions / 1024 Rows 批次规模。

> 混合批处理仅重排任务执行时机，不变更任何模型运算语义。默认已开启；亦可通过环境变量 `MBUFO_MIXED_BATCH=0` 切换回传统调度模式。设计细节与验证矩阵请参阅 [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md)。

### DecodeBatch 与 Split-K Decode 算子

在服务层将 Decode 步骤以预处理批次推进（`StepDecodeBatch` / `ExecutePreparedDecodeBatch`），底层注意力内核全面接入共用 Split-K 解码读取器（`blocked_splitk`）。在共用 Q8 下，Gemma 4 SWA 层解码性能提升，27B 注意力层直接读取开销极低，且 F16 输出维持全精度位元精确。

### 公平调度防饥饿架构

为杜绝超长 Prefill 挤压短请求首字延迟，MBufo 实现三层调度防护机制：

1. **SRTF（Shortest Remaining Time First）Prefill 调度**：剩余分块 ≤1 的短 Prefill 优先排入执行，避免短任务在队尾无谓等待。
2. **饥饿解码优先抢占（Starvation Preemption）**：当有请求等待 Session 槽位时，长 Prefill 会主动让路、优先排空常驻 Decode，以最快速度释放槽位（上限 32 轮，避免 Prefill 自身饥饿）。
3. **Near-Done 闸门机制**：仅在常驻解码器距离结束 ≤64 Tokens 时触发抢占，若常驻均为长篇输出则不盲目延后 Prefill，兼顾整体通畅度。

| 评测指标（gfx1151，FP4 + f16 + MTP + 4 sessions） | 实测表现 |
| --- | --- |
| Fairness-A 退化比 | **160 → 0.7** |
| Fairness-B 退化比 | **856 → 105** |
| 短请求 TTFT（首字延迟） | **3.4 秒 → 0.1 秒** |
| ITL（Token 间延迟） | 稳定不增加 |
| 贪婪输出正确性 | 位元一致（SHA 对比通过） |

### 负载下三引擎对比（启用 MTP）

在相同测试框架与充分预热环境下（并发 1 / 2 / 3 / 4 之总 TPS 与每 Session TPS）：

| 引擎 | 总 TPS | 每 Session TPS | Fairness A / B |
| --- | --- | --- | --- |
| A (Halogen) | 85 / 82 / 75 / 88 | 85 / 41 / 25 / 22 | 322 / 1140 |
| B (gufo) | 57 / 64 / 108 / 127 | 57 / 32 / 37 / 33 | 15 / 399 |
| **C (MBufo)** | **66 / 83 / 110 / 129** | **66 / 42 / 38 / 33** | **19 / 187** |

MBufo 在多路负载下展现出最优秀的线性扩展能力（总吞吐平稳攀升至 129 TPS，单路速率平缓），且具备最佳的公平性指标。

### 高效投机解码（Speculative Decoding）

通过统一 Draft Registry 整合多种模型专属的草稿架构：

- **Flash-Next MTP**：并发宽度 2 / 3 / 4 下接受率维持在 **88.7% / 91.6% / 89.5%**，打破随批次衰减的瓶颈，总吞吐达到 97 / 111 / 133 tok/s。
- **27B DFlash2**：原生整合 Blocked-Q8 KV 缓存，验证流程直接写入 Blocked Sidecar，保持精准一致。
- **Gemma 4 MTP**：采用异步 GPU 草稿与单 Forward 验证，长上下文解码速度达 **74–86 tok/s**（`d=3` 深度下每轮平均接受 1.74–1.78 tokens）。
- **Muse Glimmer DFlash**：端到端投机接受率达 **~89%**。

### Prefix RAM 缓存与生产级 Session 配置

- 重复 Prefix 直接从内存缓存读取（共用边界、可自定义 RAM 上限），避免重复计算。
- 多 Session 调度经全面校准，消除并发时的阶梯式延迟，支持多路并行加速。

| 部署目标 | 建议参数配置 | 资源占用概况 |
| --- | --- | --- |
| 16 路高并发服务 | `--sessions 16 -c 65536 --kv-quant q8` | 约 103 GB（保留约 20 GB 冗余） |
| 8 路超长对话服务 | `--sessions 8 -c 131072 --kv-quant q8` | 约 101 GB（保留约 21 GB 冗余） |
| 4 路极限 256K 对话 | `--sessions 4 -c 262144 --kv-quant q8` | 约 101 GB（保留约 21 GB 冗余） |
| 单路极限 256K 对话 | `--sessions 1 -c 262144 --kv-quant q8` | 轻量占用，极致吞吐 |
| 单路最高精度 Prefill | `--sessions 1 -c 262144 --kv-quant f16` | 全精度 KV，无量化损耗 |

---

## 2. 低精度计算与成本优化

MBufo 主张精度与性能之间的平衡需经严格量测。推荐基准配置为 `--kv-quant q8` 搭配 ROCmFP4 或 Q4_K 权重。

### KV 缓存：Blocked Q8 与 4-bit 紧凑布局

底层将 Blocked Q8 KV 抽象为标准化编解码器，支持 Key 与 Value 独立选择 4-bit（每 32 元素占 18 字节）或 8-bit（每 32 元素占 34 字节）精度。共用核心整合了 Blocked 写入器、直读者、WMMA Prefill 读取器与 Split-K 解码读取器。

| 模式 | 状态 | 每位置字节数 | 256K Session 显存占用 | 说明 |
| --- | --- | --- | --- | --- |
| `f16` | 默认基准 | 2048 B | 6.44 GB | 基准数值，无 KV 量化。 |
| `q8` | **推荐模式** | 1088 B | 3.42 GB | Blocked 8-bit，平衡性最佳。 |
| `q8` + `MBUFO_KV_V4=1` | 手动开启 (K8V4) | 832 B | 2.62 GB | Value 压缩为 4-bit，Key 维持 8-bit。 |
| `q8` + 两者开启 | 手动开启 (K4V4) | 576 B | 1.81 GB | Key / Value 均压缩为 4-bit，极致节省显存。 |

- **推荐使用 `q8`**：Blocked-Q8 在编解码层面表现优异，精度极高，显存直接减半。
- **4-bit 适用场景**：专为极限显存受限机型设计的环境变量开关，未启用时完全不影响基准数值。详情参阅 [docs/KV-V4.md](docs/KV-V4.md)。

### 权重精度架构

- **ROCmFP4 / ROCmFP4-FAST**：4.25 bpw 单尺度 FAST 专家解码借助 fp16 WMMA 矩阵核心硬件直驱，使 MoE Prefill 充分享受矩阵核心加速。在 gfx1151 上，FAST 模型在长文本 Prefill 下表现亮眼，解码更较传统方式大幅提升。
- **MXFP4**：支持 GPT-OSS 20B / 120B 原生 MXFP4 路由运算，Decode 端融合 MoE / QKV / 投影路径，120B 支持零拷贝流式上传。
- **Q4_K**：UD-Q4_K_XL 为 27B 与 Flash-Next 的参考优化架构；bge-m3 提供 Q4_K_M 加载器。
- **W8A8**：共用主机端 16-split varlen GEMM 核心，支持 Glimmer 与 Flash-Next Prefill。

### 注意力运算内核优化

- **int8 WMMA Direct Prefill**：单 Session Causal Prefill 直接读取 Blocked int8 缓存，跳过 F16 暂存反量化，14K Prefill 速度提升 **30%**（1085 → 1408 tok/s）。
- **Split-K Flash-Decoding**：结合融合解码 GEMV、融合 Norm / Residual / Epilogue 内核及设备端 Argmax。
- **Wide Prefill 全面覆盖**：针对所有宽于 Vector Batch 的 Prefill 请求启用融合投影与成对矩阵核心处理。

### 性能对比 llama.cpp（gfx1151 实测）

相同权重、贪婪解码、Release 构建，在同一台 Strix Halo 节点上对 llama.cpp `5d9e5ac30` 重新实测。MBufo 经由 HTTP 服务端测量，数字含分词与请求开销；llama.cpp 使用 `llama-bench` / `llama-server`，并取其最佳批大小。Prefill 为约 2K token 的冷提示；Decode 为约 2K token 上下文下的生成速度；均未开启投机解码。MBufo 使用默认的分块 Q8 KV 缓存，llama.cpp 使用 F16（Flash-Next 改用 F16 KV 缓存时 prefill 为 1662 tok/s，而非 1158；下文「性能基准」使用 F16）。正值代表 MBufo 更快。

| 模型（权重） | Prefill tok/s：MBufo / llama.cpp | 差距 | Decode tok/s：MBufo / llama.cpp | 差距 |
| --- | --- | --- | --- | --- |
| **Gemma 4 26B-A4B**（Q4_0） | 2689 / 1728 | **+56%** | 55.3 / 53.2 | +4% |
| **Qwen3.6-35B-A3B**（UD-Q4_K_XL） | 2130 / 1230 | **+73%** | 54.4 / 50.7 | +7% |
| **GPT-OSS 120B**（Q4_K_M） | 1226 / 1089 | **+13%** | 52.3 / 52.4 | 0% |
| **Muse Glimmer 30B**（UD-Q4_K_XL） | 516 / 398 | **+30%** | 12.6 / 13.2 | -5% |
| **DeepSeek V4 Flash**（IQ2_XXS 混合） | 432 / 63 | **+586%** | 15.8 / 14.2 | +12% |
| **Qwen3.8 Flash-Next**（ROCmFP4-FAST） | 1158 / 无 | 无 | 33.3 / 无 | 无 |

Embedding 与重排序（经由 HTTP 服务端的单次请求，长度单位为 token）：

| 模型（权重） | MBufo / llama.cpp | 差距 |
| --- | --- | --- |
| **bge-m3**（Q4_K_M），0.9K | 16651 / 5663 tok/s | **+194%** |
| **bge-m3**（Q4_K_M），1.7K | 14101 / 5032 tok/s | **+180%** |
| **bge-m3**（Q4_K_M），5K | 7426 / 3324 tok/s | **+123%** |
| **bge-m3**（FP16），1.7K | 13895 / 4962 tok/s | **+180%** |
| **Qwen3-Embedding-0.6B**（Q8_0），0.9K | 15481 / 7454 tok/s | **+108%** |
| **Qwen3-Embedding-0.6B**（Q8_0），1.7K | 13981 / 5301 tok/s | **+164%** |
| **Qwen3-Embedding-0.6B**（Q8_0），3.4K | 11356 / 4700 tok/s | **+142%** |
| **bge-reranker-v2-m3**（Q4_K_M），64 篇文档 | 76.9 / 38.0 docs/s | **+102%** |
| **bge-reranker-v2-m3**（FP16），64 篇文档 | 73.1 / 34.3 docs/s | **+113%** |
| **Qwen3-Reranker-0.6B**（Q8_0），64 篇文档 | 41.1 / 28.1 docs/s | **+46%** |

说明：
- llama.cpp 不支持 Flash-Next 架构，因此没有对照数字。
- DeepSeek V4 的对照为该 llama.cpp 构建自带的 `deepseek4` 支持，使用默认批大小 512；本次未使用 ds4 参考项目。
- llama.cpp 批大小：Gemma 4 与 Glimmer 为 2048，Qwen3.6 为 1024（2048 曾使测试机死机，其最佳成绩可能更高），GPT-OSS 经 `llama-server` 为 8192。
- bge-m3 单次输入可达模型原生的 8192 token。输入长度会按约 12% 的间隔分桶并做精确的键遮罩，不同长度的请求因此共用已调优的 GEMM 方案；调优结果缓存在磁盘（`MBUFO_TUNE_CACHE`，默认 `~/.cache/mbufo/bge_tune_v1.txt`），新形状的第一次请求仍需付出一次性调优时间。
- Qwen3-Reranker：MBufo 套用该模型要求的提示模板（每篇文档约多 27% token），并且每次请求只预填一次共用的系统指令与查询前缀，而不是每篇文档各算一次（得分完全相同）。该 llama.cpp 构建的 `/rerank` 未套用，在 8 篇相关与 40 篇无关文档的测试中，它的前 8 名里没有任何一篇相关文档，MBufo 则把 8 篇相关文档全排在最前。

- EmbeddingGemma 2（BF16，文本路径；`tools/convert/embeddinggemma2_to_gguf.py`）尚不受 llama.cpp 支持，因此没有 llama.cpp 对照。MBufo 的向量与独立的 PyTorch 实现一致（6 段 4 到 908 token 的文本，余弦相似度 1.000000）。经 HTTP 服务，15 到 23 token 的请求约 10 ms，908 token 为 76 ms（11.9K tok/s）；同一节点上以 32 个 CPU 线程（FP32）运行的 PyTorch 实现，13 token 需要 24.7 ms，908 token 需要 364 ms（这台机器上的 PyTorch ROCm 无法运行）。

---

## 3. GGUF 全家族支持与多模型路由

### 支持的模型家族

通过内置多模型路由器，单一 HTTP 端口即可统筹调度下列全部 20 个家族：

| 模型 | 参数量 | HTTP 服务 | 验证测试 | 投机草稿支持 | 特性与架构亮点 |
| --- | --- | --- | --- | --- | --- |
| **Qwen3.8 Flash-Next**（主力） | 30B-A3B MoE | 支持 | Pilot | MTP | ROCmFP4-FAST / UD-Q4_K_XL、Blocked-Q8 KV |
| **Qwen3.8 27B** | 27B Dense | 支持 | Pilot | DFlash2 (含 Q8) | 支持 Dense 与 MoE 路径，推荐 UD-Q4_K_XL |
| **Gemma 4 26B-A4B** | 26B-A4B | 支持 | 完整测试 | MTP (F16 镜像) | 具备原生 Tool 支持，SWA 共用 Q8 |
| **Gemma 4 E4B / E2B** | 4B / 2B 有效參數（逐層嵌入） | 支援 | 已驗證 | — | Q4_0 QAT；PLE（Q6_K gather）、共用 KV 層、WMMA 大 head prefill |
| **Gemma-4-12B** | 12B Dense | Bench | — | — | 支持 K-quant 预转换与 WMMA Q8 Prefill |
| **Qwen3.5-35B-A3B** | 35B-A3B MoE | Engine | — | — | 支持 qwen35moe 架构路径 |
| **Qwen3.6-35B-A3B** | 35B-A3B MoE | Engine | — | — | Q6_K→Q8_0 加载、Split-K MoE 解码 |
| **GPT-OSS 20B / 120B** | 20B / 117B MoE | 支持 | Pilot | — | 原生 MXFP4 矩阵核心、120B 支持零拷贝流式上传 |
| **Muse Glimmer 30B** | 30B | 支持 | Prompt | DFlash (~89%) | ATEM 工具调用、W8A8 / Q8_0 支持 |
| **Llama-3.1-8B** | 8B Dense | 支持 | Adapter | — | 标准 Dense GQA 支持 |
| **DeepSeek V4 Flash** | MoE（IQ2/Q2/Q8 混合 GGUF） | 支援 | 品質限制已記載 | DSpark | MLA decode、ds4 一致 kernel、約 81 GiB 常駐 |
| **Qwen3-Embedding-0.6B** | 0.6B | 支持 (`/v1/embeddings`) | — | — | 缓存池化 Single-Prefill 执行器 |
| **bge-m3** | 0.6B XLM-R | 支持 (`/v1/embeddings`) | Encoder | — | CLS 池化、Q4_K_M 加载器、hipBLASLt 调优 |
| **EmbeddingGemma 2** | 约 0.3B Gemma-4 编码器（文本路径） | 支持 (`/v1/embeddings`) | 与 PyTorch 一致（余弦 1.000000） | — | 双向注意力：20 层滑动窗口（±512）与 4 层全局层（head_dim 256 / 512）、逐层输入、平均池化、768 维输出 |
| **bge-reranker-v2-m3** | Cross-Encoder | 支持 (`/v1/reranking`) | ScoreBatch | — | BERT 分派、多序列 Padding 批次加速 |
| **Qwen3-Reranker** | Cross-Encoder | 支持 (`/v1/reranking`) | Scoring | — | Qwen 分派、共用 Rerank 评分头 |
| **Qwen3-ASR 1.7B** | 1.7B | 支援（語音轉文字） | — | — | BF16 safetensors、決定性 greedy 解碼 |
| **Qwen3-TTS 12Hz 1.7B** | 1.7B | 支援（語音合成） | — | — | BF16 safetensors、F32 波形解碼器 |
| **Qwen-Image-2.1** | Diffusion transformer | 支援（圖像生成／編輯） | — | — | BF16 safetensors、Qwen3-VL 提示編碼器、RGBA VAE |
| **MiniMax H3** | 50 區塊 BF16 transformer | 支援（影片，純文字 FL2VA） | — | — | HIP transformer、VisualVAE / AudioVAE、MP4 合成 |

### 支持的量化格式

各引擎均直接读取 GGUF；下表为各引擎接受的格式（不在清单内的格式会在加载时连同张量名称一并拒绝，绝不会被当成其他格式解码）。细节、限制与各组合的验证方式见 [docs/QUANT_FORMATS.md](docs/QUANT_FORMATS.md)。

| 引擎 | 稠密／注意力权重 | 路由专家 |
| --- | --- | --- |
| **Qwen 27B / Qwen3.5-3.6 MoE** | Q8_0、Q4_K、Q5_K、Q6_K、Q3_K、IQ4_NL、IQ4_XS、IQ3_S、ROCmFP4、F16 / BF16 / F32 | Q4_0、Q4_1、Q5_0、Q5_1、Q8_0、Q2_K、Q4_K、Q5_K、Q6_K、IQ2_XXS、MXFP4、ROCmFP4 |
| **Qwen3.8 Flash-Next** | 原生：Q8_0、Q6_K、ROCmFP4 / FAST、MXFP4、F16 / BF16 / F32；加载时转为 Q8_0：Q4_K、Q5_K、Q4_0、Q4_1、Q5_0、Q2_K、Q3_K、IQ4_XS、IQ4_NL | Q4_K、Q5_K、Q6_K、Q5_1、Q8_0、Q4_0、Q3_K、Q2_K、ROCmFP4_FAST、MXFP4 |
| **Gemma 4（26B-A4B、E4B、E2B、12B）** | Q4_0、Q8_0、Q4_K、Q5_K、Q6_K、F16 / BF16 / F32 | 同左（Q4_0 走快速 MoE 路径） |
| **GPT-OSS 20B / 120B** | Q8_0、Q5_0、Q4_K 及 F16 / BF16 / F32 注意力 | MXFP4（原生）、Q4_0、Q8_0、Q5_1（K 系列无法编码其 2880 宽的专家） |
| **Muse Glimmer、Llama-3.1** | Q4_0、Q8_0、Q4_K、Q5_K、Q6_K、F16 / BF16 / F32 | - |
| **DeepSeek V4 Flash** | 已发布的 IQ2_XXS / Q2_K / Q8_0 混合配方 | 同左 |
| **bge-m3、重排序、Qwen3-Embedding** | Q4_K_M 与 F16 加载器 | - |
| **EmbeddingGemma 2** | BF16（由 `tools/convert/embeddinggemma2_to_gguf.py` 生成 GGUF） | - |

嵌入查表、行反量化、行点积内核与主机端反量化器共用同一份格式注册表（`core/quant/quant_traits.hpp`）：F32、F16、BF16、Q4_0、Q4_1、Q5_0、Q5_1、Q8_0、Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、IQ4_NL、IQ4_XS、IQ2_XXS、MXFP4、ROCmFP4、ROCmFP3。KV 缓存：F16，或分块 INT8（Q8），内存约减半，并保持精确连续性的 split-K 注意力。

### 生产级异构多模型并驻（Multi-Model Co-Residency）与 `/admin` 控制台

执行 `serve llm --multi-model` 可将 MBufo 转为单一端口（如 `:8741` 或 `:8080`）的**异构多模型并驻服务器**。多颗不同架构的 GGUF 模型（MoE 主力模型 + Dense 次要模型 + Vision `mmproj` 视觉挂件 + `MTP`/`DFlash2` 投机解码草稿模型）可**同时常驻于 Strix Halo 统一内存（UMA）中并发服务**，无须反复卸载重载。

#### 核心能力与架构亮点
- **异构多模型同时并驻**：只要统一内存容量允许（例如 128 GB UMA 同时加载 94 GiB `Qwen3.8-Flash-Next` 与 13 GiB `Qwen3.8-27B`），多颗模型可同时在线并各自拥有独立的 Session Slots 与 KV 缓存池。客户端只需在 `/v1/chat/completions` 或 `/v1/messages` 的 JSON payload 指定 `"model"`（支持 ID 或自定义别名 `aliases`）即可精准路由。
- **内置 `/admin` Web UI 管理控制台**：浏览器直接打开 `http://<host>:<port>/admin` 即可使用：
  - **顶部固定式 Unified Memory & VRAM Gate**：实时显示 Strix Halo UMA 各模型常驻占用、System/KV 缓冲池与安全保留区；鼠标悬停在目录模型上时，以单行实时预览“可直接并驻”或“需触发 LRU 置换”的内存差额。
  - **已加载模型与实时 Slot 监控**：实时查看每颗模型的每个 Session Slot 状态（`BUSY` / `IDLE`、KV 缓存 Token 数、闲置时间），一键切换 Primary 默认模型或无损卸载。
  - **本地 GGUF 模型目录自动扫描（`/admin/catalog`）**：自动递归扫描 `MBUFO_MODELS_DIR`（或模型所在母目录），自动合并多分割文件（shards）并自动配对同目录下的 `mmproj` 视觉权重与 `MTP`/`DFlash2` 草稿模型。
  - **内置多模型流式 Playground**：直接在网页上选择任一并驻模型发送 OpenAI / Anthropic 流式请求，实时测量 TTFT 与 Decode tok/s。
- **零模型待机启动（Zero-Model Standby）**：支持不带 `--model` 直接执行 `mbufo serve llm --multi-model -p 8741`（仅占用约 86 MiB RSS），启动后直接从 `/admin` 网页点选加载所需模型。
- **安全内存闸门与无损排空**：`/admin/load` 自动保留安全边界（`MBUFO_MULTI_SAFETY_MIB`，默认 `2048` MiB）。空间不足时可自动对闲置的非默认模型执行 **LRU 置换卸载**（若设 `"allow_evict": false` 则返回 `409 insufficient_memory` 拒绝加载以保护现有并驻模型）；卸载模型时会等待进行中的请求生成完毕才释放内存。

#### 端点一览

| 端点 | 方法 | 功能说明 |
| --- | --- | --- |
| `/admin` | GET | 内置多模型并驻管理 Web UI 仪表盘与实时测试 Playground |
| `/v1/chat/completions` | POST | OpenAI 聊天对话接口（依 `"model"` ID/别名自动路由，未填则回退至默认模型） |
| `/v1/messages` | POST | Anthropic Messages 接口（依 `"model"` ID/别名自动路由，支持 Claude Code 等工具） |
| `/v1/embeddings` | POST | 向量嵌入接口（支持 bge-m3、Qwen3-Embedding、EmbeddingGemma 2） |
| `/v1/reranking` | POST | 文本重排序接口（支持 bge-reranker-v2-m3、Qwen3-Reranker） |
| `/admin/models` | GET | 查询当前并驻模型列表、各 Slot 状态与 UMA 内存遥测数据 |
| `/admin/catalog` | GET | 扫描本地 GGUF 模型库并自动配对 `mmproj` 与 `MTP`/`DFlash2` 附属权重 |
| `/admin/load` | POST | 动态热加载新模型并驻（支持自定义别名、Speculative、Vision 与 LRU 保护） |
| `/admin/default` | POST | 动态切换默认主模型（`{"id": "<model-id>"}`） |
| `/admin/eject` | POST | 在线无损卸载指定模型，待任务排空后平滑释放内存 |

---

## 快速开始

### 1. 启动主推理服务（以 Qwen3.8 Flash-Next 为例）

```sh
./build/release/mbufo serve llm \
  --model models/qwen3.8-rocmfp4/v2q8k/Qwen3.8-Flash-Next-ROCmFP4-FAST-v2q8k.gguf \
  --mmproj models/qwen3.8-flash-next/mmproj-BF16.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/agentionai-MTP-Q8_0.gguf \
  --speculative mtp --think off \
  --served-model-name mbufo-qwen3.8-flash-next \
  --sessions 2 -c 262144 --kv-quant q8 -p 8741
```

### 2. 多模型并驻模式（启用 `--multi-model` 与 `/admin` Web UI）

```sh
# 方式 A：带默认主力模型启动，并开启多模型并驻与 /admin 控制台
./build/release/mbufo serve llm --multi-model \
  --model models/qwen3.8-rocmfp4/v2q8k/Qwen3.8-Flash-Next-ROCmFP4-FAST-v2q8k.gguf \
  --mmproj models/qwen3.8-flash-next/mmproj-BF16.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/agentionai-MTP-Q8_0.gguf \
  --speculative mtp --think off \
  --served-model-name mbufo-qwen3.8-flash-next \
  --sessions 2 -c 262144 --kv-quant q8 -i 0.0.0.0 -p 8741

# 方式 B：零模型待机启动（不预载模型，直接由 http://localhost:8741/admin 网页菜单加载）
MBUFO_MODELS_DIR=/path/to/models ./build/release/mbufo serve llm --multi-model -i 0.0.0.0 -p 8741
```

启动后直接用浏览器打开 **`http://localhost:8741/admin`** 即可通过图形化界面管理多模型并驻。

### 3. 多模型并驻 API 调用示例

```sh
# 动态热加载第二颗模型（Qwen3.8-27B）与主力模型同时并驻（设置 allow_evict: false 确保不踢除现有模型）
curl http://localhost:8741/admin/load \
  -H "Content-Type: application/json" \
  -d '{
    "id": "qwen3.8-27b",
    "aliases": ["27b", "fast-dense"],
    "path": "models/qwen3.8-27b/Qwen3.8-27B-Q4_0_ROCMFP4_FAST.gguf",
    "max_context": 32768,
    "session_count": 1,
    "kv_cache_quant": "q8",
    "speculative": "off",
    "allow_evict": false
  }'

# 同一端口 (:8741) 同时对两颗并驻模型发送推理请求
curl http://localhost:8741/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mbufo-qwen3.8-flash-next", "messages": [{"role": "user", "content": "请用一句话自我介绍。"}]}'

curl http://localhost:8741/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "27b", "messages": [{"role": "user", "content": "请用一句话自我介绍。"}]}'

# 切换默认模型、查询并驻状态与内存占用、或卸载指定模型
curl http://localhost:8741/admin/default -H "Content-Type: application/json" -d '{"id": "qwen3.8-27b"}'
curl http://localhost:8741/admin/models
curl http://localhost:8741/admin/eject -H "Content-Type: application/json" -d '{"id": "qwen3.8-27b"}'
```

兼容于 OpenAI SDK、Open WebUI、VS Code 等前端工具，Base URL 设置为 `http://localhost:8080/v1` 即可。完整接口协议详见 [docs/SERVER.md](docs/SERVER.md)。

---

## 性能基准（gfx1151 Strix Halo）

测试条件（本版本重新测量，经 HTTP 服务）：冷启动 KV（每请求独立 Prompt）、关闭思考模式、F16 KV、单会话 `-j 1`、上下文 262,144。27B 采用 DFlash2 投机解码，Flash-Next 采用 MTP 投机解码。Prefill 为服务端回报的提示处理速率；Decode 为 128 至 256 token 生成三次的中位数，会随草稿接受率波动。

### Prefill 吞吐量（tok/s，越高越好）

| Token 规模 | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---:|---:|---:|---:|---:|
| **2K** | 533 | 534 | **1733** | 1719 |
| **8K** | 551 | 515 | **1708** | 1678 |
| **32K** | 525 | 512 | **1679** | 1672 |
| **64K** | 467 | 479 | **1643** | 1637 |
| **128K** | 382 | 392 | 1585 | **1587** |
| **250K** | — | — | 1512 | **1518** |

### Decode 速度（tok/s，开启投机解码）

| 测试场景 | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---|---:|---:|---:|---:|
| 散文生成 (temp 0.4) | 24.6 | 22.5 | **54.4** | 52.1 |
| 斐波那契数列生成 (temp 0.4) | 41.5 | 41.9 | **59.2** | 56.0 |
| JSON 结构生成 (temp 0) | 43.8 | 44.3 | 63.6 | **64.2** |
| 代码贪婪生成 /128 (temp 0) | 44.2 | 37.9 | 50.6 | **52.7** |
| 重复文本生成 /256 (temp 0) | 55.7 | 52.5 | 78.0 | **81.7** |

### Agentic 综合评测（agent_bench_v2 沙箱工具组）

_本版本未重新测量：此表来自早期版本。_

| 评测项目 | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---|---:|---:|---:|---:|
| 通过任务数 | **8/8** | 7/8 | 7/8 | 7/8 |
| 严重失误次数 | **0** | **0** | **0** | **0** |
| 生成 TPS 范围 | 9.1–31.1 | 9.3–21.5 | 16.7–56.6 | 16.8–55.1 |

### 各模型推荐配置

- **Qwen 3.8 27B**：推荐采用 **UD-Q4_K_XL** 作为参考版本；在上述测量中，两种构建的解码速度在上述测量中相差约 15% 以内。
- **Qwen 3.8 Flash-Next**：推荐采用 **ROCmFP4**，解码吞吐极高（散文 54 tok/s、斐波那契 59 tok/s），且仅需 ~94 GB 显存（对比 Q4_K_XL 节省约 17 GB）。

---

## 上游追踪

MBufo 持续追随并以标准 Git Merge 方式整合上游 `main` 分支，保持历史清晰与干净。

| 项目 | 内容说明 |
| --- | --- |
| 上游仓库 | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| 上游基准版本 | `v0.26.0`（持续追随稳定提交） |
| 合并策略 | 通过 `--no-ff` 合并上游最新提交，MBufo 特有提交保持独立清晰，永不篡改上游历史 |
| 标签命名规范 | MBufo 发布标签均采用 `mbufo-vX.Y.Z` 前缀，与上游原生标签完全隔离 |

---

## 从源码构建

目标环境为 Linux x86-64 上的 AMD Strix Halo（`gfx1151`）。构建流程与上游标准完全一致：

```sh
cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/mbufo diagnose
```

### 环境依赖需求
- **编译器与工具**：支持 C++20 之编译器（推荐 GCC 15.3）、CMake 3.21+、Ninja、pkg-config。
- **运算库**：ROCm 7.2.3 开发库（HIP、hipBLAS、hipBLASLt、rocBLAS、hipCUB、rocPRIM、rocWMMA）。亦支持 Nix 构建（`nix build`）。
- **系统依赖**：ICU、libcurl、OpenSSL、libpng、libjpeg、libwebp、FFmpeg。
- **权限**：执行账号需具备访问 `/dev/kfd` 与 `/dev/dri` 节点之权限。

完整构建说明可参考 [上游构建文档](https://github.com/gufo-org/gufo#build-from-source)。

---

## 文档索引

| 文档路径 | 说明内容 |
| --- | --- |
| [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md) | 混合批处理架构设计、核心不变量与实测矩阵 |
| [docs/KV-CACHE.md](docs/KV-CACHE.md) | 对话缓存与组件生命周期管理 |
| [docs/KV-V4.md](docs/KV-V4.md) | 4-bit KV 块布局规范与数值精确度度量 |
| [docs/SERVER.md](docs/SERVER.md) | HTTP API 接口规范与通讯协议 |
| [docs/CLI.md](docs/CLI.md) | 命令行参数完整指南 |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | 开发者指引、测试配置与构建 Preset |
| [docs/TESTING.md](docs/TESTING.md) | 自动化 CI 测试组与本地验证规范 |
| [docs/models/README.md](docs/models/README.md) | 各模型家族专属调优与配置指南 |
| [CHANGELOG.md](CHANGELOG.md) | 项目历程与详细更新日志 |

---

## 兼容性与限制

- **语义不变性**：混合批处理仅调度排程，绝不改变模型数学语义；若遇到不支持整批运算之算子，自动安全回退。
- **精度选择**：`--kv-quant q8` 以极微小的精度折损换取倍数级显存空间；若应用对数值一致性有绝对严格要求，请使用 `f16`。
- **快照隔离**：KV 状态快照严格按格式版本区隔，避免跨精度误用。
- **模型调度提醒**：Qwen 3.5 超过 64K 上下文时，建议将实际长度控制在 60K 内以确保驱动层级稳定；Gemma 4 之 Full-512 层维持 F16 运行。

---

## 授权与致谢

上游 gufo 采用 MIT 授权条款；第三方依赖库授权声明详见 [NOTICE](NOTICE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 `licenses/`。模型权重文件依其各原始发布者之许可协议规范。

MBufo 起步于 [gufo](https://github.com/gufo-org/gufo)。感谢 gufo-org 及其贡献者在 Q8 KV 缓存、int8 WMMA 与投机解码上的工作，为本项目奠定了基础。
