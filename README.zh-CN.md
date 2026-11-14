# MBufo

专为 AMD Strix Halo（`gfx1151`）架构深度打造的 [gufo](https://github.com/gufo-org/gufo) 分支，定位为**高并发、极致公平性与低成本的生产级推理引擎**。

> [English Version](README.md) · [繁體中文版](README.zh-Hant.md)

上游 gufo 专精于单流极限吞吐；MBufo 则聚焦于服务端实际服务的核心挑战：单台 Strix Halo 节点在面临十六路以上高并发请求时如何保持满载，且每路请求如何承载长达 256K 的极限上下文。MBufo 以三大核心支柱解答：**负载下的公平调度（§1）**、**极致节省显存与算力的低精度计算（§2）**，以及**单端口整合全家族的 GGUF 兼容支持（§3）**。底层引擎、算子内核与核心架构源自上游，在此诚挚致谢 gufo-org 及广大社区贡献者。

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
| **全能托管** | **13 个 GGUF 模型家族** | 单一 HTTP 端口统筹服务，涵盖 Flash-Next、27B、Gemma 4、MoE 模型、Embedding 与 Reranker，支持动态热加载与无重启卸载。 |

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

> 混合批处理仅重排任务执行时机，不变更任何模型运算语义。默认已开启；亦可通过环境变量 `GUFO_MIXED_BATCH=0` 切换回传统调度模式。设计细节与验证矩阵请参阅 [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md)。

### DecodeBatch 与 Split-K Decode 算子

在服务层将 Decode 步骤以预处理批次推进（`StepDecodeBatch` / `ExecutePreparedDecodeBatch`），底层注意力内核全面接入共用 Split-K 解码读取器（`blocked_splitk`）。在共用 Q8 下，Gemma 4 SWA 层解码性能提升，27B 注意力层直接读取开销极低，且 F16 输出维持全精度位元精确。

### 公平调度防饥饿架构

为杜绝超长 Prefill 挤压短请求首字延迟，MBufo 实现三层调度防护机制：

1. **SRTF（Shortest Remaining Time First）Prefill 调度**：剩余分块 ≤1 的短 Prefill 优先排入执行，避免短任务在队尾无谓等待。
2. **饥饿解码优先抢占（Starvation Preemption）**：当有请求等待 Session 槽位时，长 Prefill 会主动让路、优先排空常驻 Decode，以最快速度释放槽位（上限 32 轮，避免 Prefill 自身饥饿）。
3. **Near-Done 闸门机制**：仅在常驻解码器距离结束 ≤64 Tokens 时触发抢占，若常驻均为长篇输出则不盲目延后 Prefill，兼顾整体通畅度。

| 评测指标（gfx1151，FP4 + f16 + MTP + 4 sessions） | 实测表现 |
| --- | --- |
| Fairness-A 退化比 | **160 → 7.32** |
| Fairness-B 退化比 | **856 → 191** |
| 短请求 TTFT（首字延迟） | **3.4 秒 → 0.7 秒** |
| ITL（Token 间延迟） | 稳定不增加 |
| 贪婪输出正确性 | 位元一致（SHA 对比通过） |

### 负载下三引擎对比（启用 MTP）

在相同测试框架与充分预热环境下（并发 1 / 2 / 3 / 4 之总 TPS 与每 Session TPS）：

| 引擎 | 总 TPS | 每 Session TPS | Fairness A / B |
| --- | --- | --- | --- |
| A (Halogen) | 85 / 82 / 75 / 88 | 85 / 41 / 25 / 22 | 322 / 1140 |
| B (gufo 上游) | 57 / 64 / 108 / 127 | 57 / 32 / 37 / 33 | 15 / 399 |
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
| `q8` + `GUFO_KV_V4=1` | 手动开启 (K8V4) | 832 B | 2.62 GB | Value 压缩为 4-bit，Key 维持 8-bit。 |
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

相同权重、Release 构建、贪婪解码；正值代表 MBufo 领先幅度：

| 模型与任务 | Prefill 领先幅度 | Decode 表现 | 备注与测试条件 |
| --- | --- | --- | --- |
| **Flash-Next** | **+52%** | 持平或更优 | 冷启动 KV、开启 MTP |
| **Qwen3.6-35B-A3B** | **+52.5%** | 持平或更优 | Split-K Router 解码 |
| **bge-m3（长文本）** | **+35% ~ +50%** | — (纯 Prefill) | hipBLASLt 自动调优与缓存执行器 |
| **Qwen3-Embedding-0.6B** | **+41% ~ +50%** | — (纯 Prefill) | 单 Prefill 优化 |
| **Reranker（批次评分）** | — | **超额 +15%** | 多序列 Padding 批处理 |
| **GPT-OSS 120B** | **+62%** | -17% | 超大参数量，Prefill 显著领先 |

---

## 3. GGUF 全家族支持与多模型路由

### 支持的模型家族

通过内置多模型路由器，单一 HTTP 端口即可统筹调度下列全部 13 个家族：

| 模型 | 参数量 | HTTP 服务 | 验证测试 | 投机草稿支持 | 特性与架构亮点 |
| --- | --- | --- | --- | --- | --- |
| **Qwen3.8 Flash-Next**（主力） | 30B-A3B MoE | 支持 | Pilot | MTP | ROCmFP4-FAST / UD-Q4_K_XL、Blocked-Q8 KV |
| **Qwen3.8 27B** | 27B Dense | 支持 | Pilot | DFlash2 (含 Q8) | 支持 Dense 与 MoE 路径，推荐 UD-Q4_K_XL |
| **Gemma 4 26B-A4B** | 26B-A4B | 支持 | 完整测试 | MTP (F16 镜像) | 具备原生 Tool 支持，SWA 共用 Q8 |
| **Gemma-4-12B** | 12B Dense | Bench | — | — | 支持 K-quant 预转换与 WMMA Q8 Prefill |
| **Qwen3.5-35B-A3B** | 35B-A3B MoE | Engine | — | — | 支持 qwen35moe 架构路径 |
| **Qwen3.6-35B-A3B** | 35B-A3B MoE | Engine | — | — | Q6_K→Q8_0 加载、Split-K MoE 解码 |
| **GPT-OSS 20B / 120B** | 20B / 117B MoE | 支持 | Pilot | — | 原生 MXFP4 矩阵核心、120B 支持零拷贝流式上传 |
| **Muse Glimmer 30B** | 30B | 支持 | Prompt | DFlash (~89%) | ATEM 工具调用、W8A8 / Q8_0 支持 |
| **Llama-3.1-8B** | 8B Dense | 支持 | Adapter | — | 标准 Dense GQA 支持 |
| **Qwen3-Embedding-0.6B** | 0.6B | 支持 (`/v1/embeddings`) | — | — | 缓存池化 Single-Prefill 执行器 |
| **bge-m3** | 0.6B XLM-R | 支持 (`/v1/embeddings`) | Encoder | — | CLS 池化、Q4_K_M 加载器、hipBLASLt 调优 |
| **bge-reranker-v2-m3** | Cross-Encoder | 支持 (`/v1/reranking`) | ScoreBatch | — | BERT 分派、多序列 Padding 批次加速 |
| **Qwen3-Reranker** | Cross-Encoder | 支持 (`/v1/reranking`) | Scoring | — | Qwen 分派、共用 Rerank 评分头 |

### 生产级多模型 Router

执行 `serve llm --multi-model` 可将服务器转为模型注册中心：启动模型即为默认模型，后续请求根据 JSON 中的 `model` 字段自动分发；亦可通过 Admin API 在线动态增减模型，服务完全不断线。

#### 核心保障与容错机制
- **安全显存调度**：在调用 `/admin/load` 时自动评估可用显存，空间不足时自动对冷门的非默认模型执行 **LRU 卸载**（可设置 `"allow_evict": false` 拒载）；默认模型永不被卸载。
- **无损排空释放**：模型卸载时，显存会等待该模型正在进行的请求全部处理完成后才平滑释放，绝不中断在线对话。
- **默认自动回退**：若请求中 `model` 字段为空，自动路由至默认模型，不报错。

#### 端点一览

| 端点 | 方法 | 功能说明 |
| --- | --- | --- |
| `/v1/chat/completions` | POST | 聊天对话接口（支持文本、图像、Function Tools、结构化输出、SSE 流式） |
| `/v1/embeddings` | POST | 向量嵌入接口（支持 bge-m3、Qwen3-Embedding） |
| `/v1/reranking` | POST | 文本重排序接口（支持 bge-reranker-v2-m3、Qwen3-Reranker） |
| `/admin/models` | GET | 查询已加载的模型列表、上下文与访问状态 |
| `/admin/load` | POST | 动态加载并注册新模型（支持显存 LRU 置换保护） |
| `/admin/eject` | POST | 在线卸载模型，待任务排空后平滑释放显存 |

---

## 快速开始

### 1. 启动主推理服务（以 Qwen3.8 Flash-Next 为例）

```sh
./build/release/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mmproj models/qwen3.8-flash-next/mmproj-BF16.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/agentionai-MTP-Q8_0.gguf \
  --speculative mtp --think off \
  --served-model-name gufo-qwen3.8-flash-next \
  --sessions 4 -c 262144 --kv-quant q8
```

### 2. 多模型模式（启用 Router 与 Admin API）

```sh
./build/release/gufo serve llm --multi-model \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --served-model-name flash-next -c 262144 --kv-quant q8 --sessions 4
```

### 3. API 调用示例

```sh
# 对话生成
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "flash-next",
    "messages": [{"role": "user", "content": "你好，请自我介绍。"}]
  }'

# 动态加载新模型 (Gemma 4)
curl http://localhost:8080/admin/load \
  -H "Content-Type: application/json" \
  -d '{
    "id": "gemma4",
    "path": "models/gemma-4-26b/gemma-4-26B-A4B-it-Q4_0.gguf",
    "max_context": 262144, "session_count": 2, "kv_cache_quant": "q8"
  }'

# 查询当前已加载的模型
curl http://localhost:8080/admin/models

# 卸载指定模型
curl http://localhost:8080/admin/eject \
  -H "Content-Type: application/json" \
  -d '{"id": "gemma4"}'
```

兼容于 OpenAI SDK、Open WebUI、VS Code 等前端工具，Base URL 设置为 `http://localhost:8080/v1` 即可。完整接口协议详见 [docs/SERVER.md](docs/SERVER.md)。

---

## 性能基准（gfx1151 Strix Halo）

测试条件：冷启动 KV（每请求独立 Prompt）、关闭思考模式、F16 KV、单流 `-j 1`、上下文 262,144。27B 采用 DFlash2 投机解码，Flash-Next 采用 MTP 投机解码。

### Prefill 吞吐量（tok/s，越高越好）

| Token 规模 | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---:|---:|---:|---:|---:|
| **2K** | 605 | 455 | **1739** | 1301 |
| **8K** | 615 | 564 | **1701** | 1549 |
| **32K** | 567 | 551 | 1598 | **1618** |
| **64K** | 508 | 496 | **1695** | 1678 |
| **128K** | 409 | 407 | **1504** | 1507 |
| **250K** | — | — | **1438** | 1415 |

### Decode 速度（tok/s，开启投机解码）

| 测试场景 | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---|---:|---:|---:|---:|
| 散文生成 (temp 0.4) | 23.5 | 19.7 | 43.3 | **43.6** |
| 斐波那契数列生成 (temp 0.4) | 67.9 | 56.2 | **86.3** | 84.7 |
| JSON 结构生成 (temp 0) | **55.1** | 46.5 | 51.5 | 51.7 |
| 代码贪婪生成 /128 (temp 0) | 61.9 | 55.3 | 70.5 | **73.2** |
| 重复文本生成 /256 (temp 0) | 70.6 | 60.8 | 74.8 | **75.3** |

### Agentic 综合评测（agent_bench_v2 沙箱工具组）

| 评测项目 | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---|---:|---:|---:|---:|
| 通过任务数 | **8/8** | 7/8 | 7/8 | 7/8 |
| 严重失误次数 | **0** | **0** | **0** | **0** |
| 生成 TPS 范围 | 9.1–31.1 | 9.3–21.5 | 16.7–56.6 | 16.8–55.1 |

### 各模型推荐配置

- **Qwen 3.8 27B**：推荐采用 **UD-Q4_K_XL**，解码速度与草稿接受率皆达最佳状态。
- **Qwen 3.8 Flash-Next**：推荐采用 **ROCmFP4**，解码吞吐极高（散文 45 tok/s、斐波那契 87 tok/s），且仅需 ~94 GB 显存（对比 Q4_K_XL 节省约 17 GB）。

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
./build/release/gufo diagnose
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

MBufo 作为 [gufo](https://github.com/gufo-org/gufo)（gufo-org 及社区贡献者）的优化衍生版本，在此感谢开源社区奠定的深厚基础。
