# MBufo

专为 AMD Strix Halo 打造的 [gufo](https://github.com/gufo-org/gufo) 分支，定位
为**公平性＋高效低销的推理引擎**——排队不插队靠公平调度，花费少靠低精度计算。

> [English](README.md) · [繁體中文](README.zh-Hant.md)

上游 gufo 负责单路极限效能；MBufo 改问服务端的问题：一台 Strix Halo 节点同时
面对十六路请求时能跑多满，每一路又能吃多长的上下文？三根支柱回答它——负载下
公平的排程（§1）、花得少的低精度计算（§2）、一个 port 服务多颗模型的 GGUF
覆盖（§3）。底层引擎、核心与绝大部分程序码都来自上游，在此致谢 gufo-org 及
所有贡献者。

**目录：** [三个数字](#mbufo-三个数字) ·
[§1 高并发](#1-高并发架构差异) ·
[§2 低精度计算](#2-低精度计算成本差异) ·
[§3 GGUF 相容](#3-gguf-相容覆盖承诺) ·
[快速开始](#快速开始) ·
[效能基准](#效能基准mbufo-v0255gfx1151-strix-halo) ·
[上游追踪](#上游追踪) ·
[建置](#从源代码建置) · [文件](#文件索引) ·
[限制](#相容性与限制)

## MBufo 三个数字

| # | 数字 | 含义 |
| --- | --- | --- |
| 1 | **短请求 TTFT 0.7 秒** | 被 25 万 token prefill 卡住的短请求，原本要等 3.4 秒；公平排程后只要 0.7 秒（fairness-A 160 → 7.32、fairness-B 856 → 191，ITL 不变，贪婪输出位元一致；v0.25.3）。 |
| 2 | **KV 内存省一半** | blocked-Q8 快取让 262144 token 的 session 从 6.44 GB 降到 3.42 GB（K4V4 为 1.81 GB），代价是量出来的、不是假设的（v0.25.2）。 |
| 3 | **11 个 GGUF 家族** | 一个 HTTP port 全数服务——v0.28.0 新增 8 个可用家族，加上 Flash-Next、27B、Gemma 4（见[§3 模型总表](#支援的模型v0280)）。 |

## §1 高并发（架构差异）

并发不是一个数字。一台吃重的服务器能不能用，是两件事决定的，MBufo 两件都当成
主轴：

- **ITL（token 间延迟）**——请求已经在串流之后，它前后两个 token 之间的间隔。
  批次效率再高，只要会让一条串流冻住一整个 prefill 步骤，那就不叫服务，那叫
  排队加上额外手续。
- **公平性**——没有任何 session、客户端或 prompt 可以饿死别人。一段 25 万 token
  的 prompt 插进对话中途，不该因此换到比十六路短对话更大的装置占比。

上游的排程器骨架已经是对的：prefill chunk 有上限（`--prefill-chunk`）、有待跑
decoder 时先服务 decode 再做 prefill、admission 有每客户端上限。它缺的是“便宜
地同时兑现这三件事”所需的装置时间：一个有上限的 chunk 仍然吃掉一整个 tick，而
那个 tick 里**没有任何** decoder 产出 token——因为 chunk 与 decode 是两趟分别
扫过权重的计算。ITL 与公平性就是从这个 tick 漏掉的：每一轮，串流中的请求都要
付一次与它无关的 prefill 步骤。

MBufo 把两者合并：chunk 与 decode 列共用同一个 `ForwardBatch`、同一个注意力
核心，于是 prefill 的那个 tick 也顺便把该出的 token 全部产出。合并没有绕过公平
政策；公平政策正是让合并安全的前提。

### 统一 mixed-batch 界面

一次 `ForwardBatch` 把 decode 列与 prefill chunk 一起送进同一个注意力核心，
不再是两趟权重计算。自 v0.27.0 起，decode/prefill 路由走单一共用 dispatcher——
统一 varlen mixed-batch 注意力界面，底下是共用 blocked-KV 编解码器、WMMA
prefill reader 与 Split-K decode reader。27B 注意力层、Flash-Next decode 注意力、
Gemma 4 SWA 层全部经由它解析；F16 路径完全未动，接线后 F16 输出位元一致。

| 量测项 | 结果 |
| --- | --- |
| 串流请求的 token 间隔，5 列批次 | 两趟计算 121 ms → 合并后 **57 ms**（**2.1 倍**）：间隔从两趟权重计算缩成一趟 |
| 16 路 sustained 服务，3 轮 | **平均 15.3 秒 对 17.0 秒**（合并开 14/17/15，关 15/16/20），实测尾延迟改善 |
| 4 路混合 prompt A/B | **5.3 秒对 6.3 秒**，每轮都赢，输出全对 |
| 16 路并发 | GPU 使用率 99%，输出全数正确，零错误 |
| 批次封装 | 32 session / 1024 列（上游为 8 / 8，过程中修掉三个硬上限：GDN、MoE、logits） |

先讲一个但书：66 列时一个合并步骤要 316 ms，比两趟分开的 247 ms 慢，因为在那种
规模下 batch-MoE 掉到 memory-bound 的 vec 路径。上面的端到端数字在服务并发下仍然
胜出——每轮少扫一趟权重——但大批次在 FLOP 这一侧的问题另案追踪。

混合批次改变的是工作何时跑，不是模型算出什么。它默认开启；设定
`GUFO_MIXED_BATCH=0` 即可完全还原上游排程，tick-tock 也一并还原。设计铁律、
验证矩阵与逐阶段量测见 [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md)。

### DecodeBatch＋Split-K decode

serve 层以准备好的批次步进 decode（`src/cli/serve/` 的 `StepDecodeBatch` /
`ExecutePreparedDecodeBatch` / `TextModelRunner::DecodeBatch`），而其中的注意力
计算走共用 Split-K decode reader（`blocked_splitk`，v0.26.1），不再走 scalar 路径。
具体数字：Gemma 4 SWA 层在共用 Q8 下 decode 只比搬移前基准慢 6.4%（scalar reader
时期为 8.0%），27B 注意力层 decode 直接读取为 −1.0%，F16 输出与搬移前 binary 位元
一致。

### 公平性：排队不插队（SRTF＋饥饿 decode 优先抢占＋near-done 闸门）

长 prefill 曾把短请求的 TTFT 挤爆（5 进 4 反饥饿场景）。三个机制修掉它，全程保持
贪婪输出位元一致（sha 比对通过：24K prefill 单跑与并发跑出相同 sha）：

- **SRTF prefill 选取**——剩余 ≤1 个 chunk 的 near-done prefill 排在长 prefill
  续片前面（同级按 FIFO）。
- **饥饿 decode 优先抢占**——有请求在排队等 session 槽位时，长 prefill chunk 让路，
  先把常驻 decode 排空，好让等待者一有常驻完成就被接纳；上限 32 轮，prefill 不会
  被饿死。
- **near-done 闸门**——只有当某个常驻 decoder 距 `token_limit` ≤64 token（槽位即将
  空出）才触发抢占；若常驻全是长输出 decoder，就不做无意义的 prefill 延迟。光是这个
  闸门就把 fairness-A 从 76.5 拉回 7.32。

| 量测（gfx1151，FP4＋f16＋MTP＋sessions 4，同 harness） | 结果 |
| --- | --- |
| fairness-A 退化比 | **160 → 7.32** |
| fairness-B 退化比 | **856 → 191** |
| 短请求 TTFT | **3.4 秒 → 0.7 秒** |
| ITL | 不变 |
| 贪婪输出 | 位元一致（sha 通过） |

### 投机解码（MTP / DFlash）

投机是各模型草稿后端挂在统一 draft registry 之下（v0.27.0）：

- **Flash-Next MTP**——并发宽度 2/3/4 的接受率 **48.8/38.9/41.7% →
  88.7/91.6/89.5%**（v0.25.1 修复：含投机 verify 的批次不再走与草稿路径不一致的
  packed mixed WMMA 核心）。接受率不再随 batch 衰减——上游 95 → 82 → 68，本构建
  稳在 ~90%——各宽度总吞吐全面超过上游（97/111/133 对 70/105/112）。（注：旧笔记的
  flash 92% 是宽度 3 峰值 91.6% 的约数，以上三数为准。）
- **27B DFlash2**——支援 blocked-Q8 KV（verify 写进 blocked sidecar，continuation
  快取与 serve 磁盘身份位元级携带）；v0.24.1 verify 路由修复救回 **+248–340%**
  decode（short/prose/code/json 5.8/5.6/13.2/11.9 → 22.9/19.5/56.2/52.3 tok/s）。
- **Glimmer DFlash**——端到端投机草稿 **~89%**（E4 证伪为非 bug；v0.28.0 已知限制，
  照实陈述）。
- **Gemma 4 MTP**——async-GPU 草稿重写＋单 forward verify 之后，decode 达
  **74–86 tok/s**（长 KV **+50%**：p2k 56.9 → 86.2，p8k 50.2 → 75.7，`d=3`）。最佳
  草稿深度 `d=3`（净吞吐最高，**每轮接受 1.74–1.78**）；`d=4` 原始接受率最高
  （1.98）但 verify 成本稍高。（注：旧笔记 ~1.6/round 依 CHANGELOG v0.26.0 修正为
  1.74–1.78。）

### prefix RAM 快取＋多 sessions 生产配置

重复 prefix 走上游 RAM 快取（shared-prefix 边界、RAM 上限、coherent prefix 索引）
不再重算，continuation 快照让长对话可恢复。生产形状是 session 数 × 上下文；自
v0.25.4 `-j` 修复后 session 数是诚实的（`-j 8` 真跑 8 路：TTFT 楼梯消失，8 路
decode 总量 124.5 → 148.2／121.1 → 144.4），常用默认已写进 CLI（llm 路径
`--sessions` 1 → 4、`--kv-quant` f16 → q8；v0.25.2）。

| 目标 | 设定 |
| --- | --- |
| 16 路、高并发 | `--sessions 16 -c 65536 --kv-quant q8`（约 103 GB，余 20 GB） |
| 8 路、长对话 | `--sessions 8 -c 131072 --kv-quant q8`（约 101 GB，余 21 GB） |
| 单路、最大上下文 | `--sessions 1 -c 262144 --kv-quant q8` |
| 4 路、最大上下文 | `--sessions 4 -c 262144 --kv-quant q8`（约 101 GB，余 21 GB——总字节与 8×131072 相同） |
| 单路、最快最精准 prefill | `--sessions 1 -c 262144 --kv-quant f16`（全精度 KV；每位置最重，但单流装得下） |
| 还原上游排程 | 环境变数 `GUFO_MIXED_BATCH=0` |

载入器限制的是“session 数 × 上下文”，不是单一上下文。在 128 GB Strix Halo
上以 blocked Q8 快取＋4-bit value 列实测，8 路甚至可到 192K（`-c 196608`，
107 GB，只剩 16 GB 余裕）；16 路拉到 96K 则会被
`text runner request-state claim exceeds state capacity` 拒载。量测条件为
UD-Q4_K_XL ＋ MTP Q8_0 sidecar ＋ BF16 mmproj，旧的 32K 上限已不成立。

### 排程器保证什么

| 保证 | 机制 |
| --- | --- |
| 待跑 decoder 绝不排在 prefill 之后 | 只有在没有 decoder 待跑时（或正在凑第一批时）才进合并步骤，所以 prefill 只在空隙里跑，永远不会插队到串流请求前面。 |
| chunk 不能把 decoder 挤出轮替 | 每次合并或 prefill 步骤后，`MarkAllDecodersDue` 会把所有常驻 decoder 重新标记为待跑，下一步就全部服务。 |
| 任何 prompt 都无法独占装置 | chunk 是有上限的切片（`--prefill-chunk` 默认 512 token），25 万 token 的 prompt 每轮只喂一片。 |
| 同等的 session 轮流 | decode 伫列会轮替（`PopDecoder` 把服务过的请求移到队尾），不是固定队头顺序，所以没有哪个 session 能一直赢得前排。 |
| 单一客户端无法塞满伫列 | `--max-pending-per-client` 限制单一客户端的待处理请求（回 `client_quota`），出队也跨客户端轮流，一个客户端爆量不会饿死别人。 |
| 某一层拒绝也不损失 | 每一层都可独立降级；投机列被拒就退回原本路径：正确，只是没有加速。 |

### 三引擎负载对比（开 MTP）

沿用先前的并发服务对比（同 harness、皆暖机；并发 1／2／3／4 的“总／每 session”
decode TPS——A Halogen · B gufo-upstream · C MBufo；数字与英文版一致，旧中文版此表
为过时版本已同步）：

| 引擎 | 总 TPS | 每 session TPS | fairness A / B |
| --- | --- | --- | --- |
| A Halogen | 85 / 82 / 75 / 88 | 85 / 41 / 25 / 22 | 322 / 1140 |
| B upstream | 57 / 64 / 108 / 127 | 57 / 32 / 37 / 33 | 15 / 399 |
| C MBufo | 66 / 83 / 110 / 129 | 66 / 42 / 38 / 33 | **19 / 187** |

A 单流峰值最高，但只在单独执行时投机——一旦并发就关掉投机头，每 session TPS 掉到
~22。负载下 **C 扩展最佳**（线性到 129 TPS、每 session 最平），自 v0.25.3 起公平性
也最好。完整方法与其余面向（prefill、fairness、ITL）见 `高并发测速方法.md`。

### 多模型 router

一个 port、多颗模型。`serve llm --multi-model` 把服务器变成模型 registry：启动
模型成为**默认**，每个请求依 `model` 字段路由到对应模型。模型可在执行期增减
（admin API）——不重启、不断线。（[§3 模型总表](#支援的模型v0280)列出一个 port
能服务的全部对象。）

启动时指定默认模型并开启 router：

```sh
./build/release/gufo serve llm --multi-model \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --served-model-name flash-next -c 262144 --kv-quant q8 --sessions 4
```

依名称路由（不带 `model` 就打默认模型）：

```sh
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "flash-next", "messages": [{"role": "user", "content": "hi"}]}'
```

执行期载入另一颗模型（例如 Gemma 4），再路由过去：

```sh
curl http://localhost:8080/admin/load -H "Content-Type: application/json" \
  -d '{"id": "gemma4",
       "path": "models/gemma-4-26b/gemma-4-26B-A4B-it-Q4_0.gguf",
       "max_context": 262144, "session_count": 2, "kv_cache_quant": "q8"}'

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "gemma4", "messages": [{"role": "user", "content": "hi"}]}'
```

列出已载入的模型，用完就卸载：

```sh
curl http://localhost:8080/admin/models
curl http://localhost:8080/admin/eject -H "Content-Type: application/json" \
  -d '{"id": "gemma4"}'
```

**内存安全。** 每次 `/admin/load` 先估计模型占用、比对免费装置内存。塞不下
时，router 会 **LRU 卸载**最久未用的非默认模型腾出空间（在 `evicted` 回报踢了
哪些）；带 `"allow_evict": false` 则改回 `409 insufficient_memory` 失败。默认
模型永不被踢，registry 不会被清空。卸载模型时，其 GPU 内存要等 in-flight 请求
跑完才释放，不会打到一半断掉。

| Admin 端点 | 方法 | Body | 效果 |
| --- | --- | --- | --- |
| `/admin/models` | GET | — | 列出已载入模型、context、最后使用时间 |
| `/admin/load` | POST | `{id, path, max_context?, session_count?, kv_cache_quant?, vision_model_path?, enable_yarn?, allow_evict?}` | 载入并注册模型（内存吃紧时 LRU 卸载） |
| `/admin/eject` | POST | `{id}` | 卸载模型；内存等 in-flight 请求排空后才释放 |

router 是选择性的：不带 `--multi-model` 时，服务器就是纯粹的单模型版，位元级
不变。

### v0.28.0 router 新增

| 功能 | 行为 |
| --- | --- |
| 多模型托管 | L0/L1 EngineRegistry factory，27B、flash-next、gemma4、glimmer、llama 五个 adapter；执行期 `/admin/load`＋`/admin/eject`，启动模型为永不卸载的默认 |
| 自动分派 | 每个请求依 `model` 字段路由；`openai_chat` 空 `model` 改走默认，不再 `400 missing_model` |
| fallback 语义 | 未知模型 id 被拒绝且不影响已载入模型；载入期 OOM 就 LRU 卸载冷门非默认模型（`allow_evict:false` → `409 insufficient_memory`）；卸载等 in-flight 请求排空才释放 GPU 内存 |
| Embeddings | `POST /v1/embeddings` 经快取 single-prefill Embed（bge-m3、Qwen3-Embedding） |
| Reranking | `POST /v1/reranking`，BERT（bge-reranker-v2-m3）＋Qwen（Qwen3-Reranker）分派，padding 批次 |
| 工具呼叫 | ATEM／Gemma／Qwen 跨后端工具呼叫映射 |
| Qwen KV 旗标修正 | 显式 `--kv-quant` 在 Qwen 路径覆写 `QWEN_KV_QUANT` |

## §2 低精度计算（成本差异）

花费用数字论证。KV 精度、权重精度、注意力变体各自对基准量过；这里没有“越窄
越免费”这种事。一句话建议：`--kv-quant q8` 配 ROCmFP4 或 Q4_K 权重——再窄都是
明确的交易，代价写在下面。

### KV 快取：blocked Q8／Q4

上游的 blocked Q8 KV 路径被泛化成“格式加编解码器”抽象，且 key 与 value 两轴
都可再压成 4-bit 的 18-byte 打包区块（每 32 个元素 18 字节）。两轴精度独立
选择，因此 K8V8、K4V8、K8V4、K4V4 四种组合都可达成。自 v0.26.1 起编解码器收进
单一共用核心（`src/core/compute/attention/`：`gufo_core_blocked_kv`）——blocked
写入器、scalar 直读器（`blocked_kv`）、WMMA prefill reader（`blocked_wmma`）、
Split-K decode reader（`blocked_splitk`）。量化 KV 是共用能力；F16 默认路径各地
未动。

| 模式 | 状态 | 说明 |
| --- | --- | --- |
| `f16` | 默认 | 基准数值。KV 快取不做量化。 |
| `q8` | 支援 | Blocked 8-bit，每 32 个元素配一个 f16 尺度（34 字节）。 |
| `q8` + `GUFO_KV_V4=1` | 手动开启，K8V4 | 4-bit value 打包成 18 字节，key 维持 8-bit。 |
| `q8` + `GUFO_KV_Q4=1` | 手动开启，K4V8 | 4-bit key 打包成 18 字节，value 维持 8-bit。 |
| `q8` + 两者同时 | 手动开启，K4V4 | key 与 value 都是 4-bit。 |

**建议使用 `q8`。** blocked 模式在编解码层级是位元精确的；`f16` 仍是默认与基准。
4-bit 路径是给“KV 内存成为瓶颈”的机器手动开启的环境变数钩子，不是
`--kv-quant` 的格式选项，也不会改变任何默认值——未设定时引擎与不含这些路径的版本
位元相同（以 logit dump 杂凑比对验证）。完整说明见 [docs/KV-V4.md](docs/KV-V4.md)。

#### 占用

每个 KV 位置的字节数，以及 `-c 262144`、Qwen3.8 Flash-Next 12 个注意力层下
单一 session 的总量：

| 模式 | 每位置字节 | 262144 单一 session |
| --- | --- | --- |
| `f16` | 2048 | 6.44 GB |
| `q8` | 1088 | 3.42 GB |
| K8V4 / K4V8 | 832 | 2.62 GB |
| K4V4 | 576 | 1.81 GB |

#### 代价（量出来的）

以 `q8` 为基准的正确性：三个互不相关的 400-token 上下文、各 98 步
matched-token decode、对三个上下文取平均：

| 模式 | decode KL | decode 最大 KL | 贪婪 argmax 一致率 |
| --- | --- | --- | --- |
| K4V8 | 0.0214 | 0.258 | 90.5% |
| K8V4 | 0.0209 | 0.120 | 93.2% |
| K4V4 | 0.0335 | 0.411 | 89.8% |

启用比 `q8` 更窄的模式前，请先读懂这张表：

- **K4V8 与 K8V4 占用的内存完全相同。** key 是 512 个元素，4-bit key 配
  8-bit value 与 8-bit key 配 4-bit value 都落在 832 字节。把精度从一轴搬到
  另一轴换不到任何内存，却要多付约 2.7 个百分点的贪婪一致率，所以 K4V8 没有
  用途。
- **只有 K4V4 真的缩小占用**，代价是 decode KL 多 60%、尾巴大 3.4 倍，大约每
  十个贪婪决策就有一个改变。这是拿正确性换内存的明确交易，不是免费。
- **不要用短 prefill 探针决定 KV 精度。** 在 33-token 的 prompt 上，K4V4 量到
  的平均 KL 与 K8V4 相同（0.0274）。原因是两轴的逐位置误差几乎不相关，同时
  量化时会部分互相抵销，小样本会把伤害藏起来。只有多上下文、decode 阶段的数字
  能把这些模式排出正确顺序。
- 4-bit 区块就是 GGML `q4_0` 的布局（16 个打包 nibble 加一个 f16 尺度），K/V
  分轴选精度在上游也已是 `--cache-type-k` / `--cache-type-v` 的既有功能。这颗
  引擎特有的地方在于：两轴都贯通 blocked kernel、混合批次路径与 snapshot 身份，
  而且代价是量出来的、不是假设的。

`q8` 对照 CPU f16 基准：端到端偏差维持在 blocked-KV 设计范围内，MTP draft 命中
率维持在 57% 至 81% 区间，算术题与长文生成保持通顺。

必须直说的一点：这些 blocked 模式在贪婪解码下与 `f16` 的分歧，都比单纯用数值
误差能解释的程度来得大。这是既有 blocked-KV 设计的性质，不是 4-bit 区块造成
的，已另案追踪；而在它之上再叠 4-bit 区块，只会把差距放大，不会优雅地叠加。

### 权重精度：W8A8／FP4／MXFP4／Q4_K

- **W8A8**——共用 GEMM 核心（`src/core/compute/gemm`，v0.27.0）的 host-side
  16-split varlen GEMM；Flash-Next prefill W8A8 已改走共用核心且输出位元一致。
  Glimmer 跑 W8A8／Q8_0 路径。
- **ROCmFP4／ROCmFP4-FAST**——4.25 bpw 单尺度 FAST down 专家经 fp16 WMMA
  routed-expert GEMM 解码（perm codebook LUT 喂进既有 Q8_0 反量化），MoE prefill
  因此跑在矩阵核心上、不掉进通用 `mul_mat_q`。gfx1151 实测 FAST 模型 prefill 打平
  或胜过 UD-Q4_K_XL 基线（1.9K／5K token 为 1564／1368 对 1556／1324 tok/s），
  decode 速度与 64k 工具梯子正确性不变。大 dense 投影在 decode 期保留 Q4_1 拷贝
  （20 B/block，比 Q8_0 少 41% 字节）：decode **+17.4%**（40.63 → 47.68 tok/s，
  v0.20.5）。
- **MXFP4**——GPT-OSS 20B／120B 跑原生 MXFP4 routed GEMM，decode 端 MoE／QKV／`wo`／
  router 全融合；120B 以零拷贝串流上传＋分块 H2D 镜像载入。
- **Q4_K**——UD-Q4_K_XL 是 27B 推荐 build（decode 与草稿接受率胜 single-scale FP4），
  也是 Flash-Next 参照基线；bge-m3 带 Q4_K_M loader；Gemma 4 跑 Q4_0 权重。

### 注意力变体：WMMA／flash／Split-K

- **WMMA prefill reader**（`blocked_wmma`）——单 session causal prefill 直接读
  blocked int8 KV 快取，跳过 F16 staging 反量化，覆盖快取能持有的全部布局
  （K34/K18 × V34/V18，dense 与 sparse）。14K 单流 prefill 比 scalar 路径
  **快 30%**（1085 → 1408 tok/s）。
- **Split-K decode reader**（`blocked_splitk`）——v0.26.1 起共用；量测数字见
  [DecodeBatch＋Split-K](#decodebatchsplit-k-decode)。
- **flash-decoding**——Gemma 4 以 Split-K flash-decoding 注意力配融合 decode
  GEMV、融合 norm／residual／epilogue kernel 与装置端贪婪 argmax（p8k prefill
  1321 → 1446 tok/s，输出位元一致），WMMA 注意力另有 SWA 滑窗 KV 跳过。
- **scalar 直读器**（`blocked_kv`）——每种布局都保留的退路。
- **F16 路径未动**——以上全是量化路径；默认 F16 decode／prefill 从不改道。

### 逐模型精度表

| 模型 | 权重精度 | KV | F16 留守 |
| --- | --- | --- | --- |
| Qwen3.8 Flash-Next（主打） | ROCmFP4-FAST（4.25 bpw）或 UD-Q4_K_XL；decode Q4_1 拷贝（+17.4%） | blocked-Q8（`--kv-quant q8`）；MTP Q8_0 sidecar | KV 默认 `f16`；全部 F16 路径未动 |
| Qwen3.8 27B | UD-Q4_K_XL（建议）或 ROCmFP4-FAST | blocked-Q8（含 DFlash2；`QWEN_KV_QUANT`、`--kv-quant` 覆写） | 同上 |
| Gemma 4 26B-A4B | Q4_0；Q4→Q8 dense 一次性预转（每 chunk 省 210 次反量化启动） | SWA 层共用 Q8（prefill −2.8%、decode −6.4%） | **full-512 层留守 F16**；MTP 草稿取自每 session F16 镜像 |
| Qwen3.6-35B-A3B | Q6_K→Q8_0 载入；split-K router GEMV | 仅引擎／bench 路径 | 尚无 serve 后端 |
| GPT-OSS 20B／120B | 原生 MXFP4 routed GEMM | text 后端服务路径 | 无投机草稿 |
| Muse Glimmer 30B | W8A8／Q8_0；MQB-first 注意力 | serve 后端 `--kv-quant` 旗标 | DFlash 草稿，无 F16 例外 |
| Llama-3.1-8B | dense GQA（见模型卡） | text 后端服务路径 | 无草稿 |
| Qwen3-Embedding-0.6B | dense Qwen 注意力 | prefill-only single-prefill Embed | 无 ladder／草稿（评分类 N/A） |
| bge-m3 | Q4_K_M loader；逐形状 autotune | prefill-only `/v1/embeddings` | **F16 hipBLASLt 融合 QKV 路径** |
| bge-reranker-v2-m3 | 共用 rerank 评分头 | padding ScoreBatch（超目标 `+15%`） | 纯评分，无草稿 |
| Qwen3-Reranker | yes/no head，Qwen 分派 | 纯评分 | 纯评分，无草稿 |

### 核心层加速

以下改动同时提升单路与批次吞吐，且都不改变模型语意。（§2 数字的实作细节，
原样保留。）

- **ROCmFP4-FAST 专家走 fp16 WMMA 核心。** 4.25 bpw 的 `Q4_0_ROCMFP4_FAST` down
  专家现在与 Q4_K/Q8_0 专家共用同一条 fp16 WMMA routed-expert GEMM（perm codebook
  LUT 喂进既有 Q8_0 反量化），FAST 模型的 MoE prefill 因此跑在矩阵核心上、不再掉
  进通用 `mul_mat_q`。gfx1151 实测 FAST 模型 prefill 打平或胜过 UD-Q4_K_XL 基线
  （1.9K／5K token 为 1564／1368 对 1556／1324 tok/s），decode 速度与 64k 工具
  梯子正确性不变。
- **int8 WMMA 直读 prefill。** 单 session 的 causal prefill
  （`WmmaCausalAttention`）现在直接读 blocked int8 KV 快取，跳过 F16 staging
  反量化。它覆盖快取能持有的任何 KV 区块布局（K34/K18 x V34/V18，dense 与
  sparse），且由与快取写入端相同的格式/环境变数选取，因此任何布局都不会被误
  读。在 14K 单流 prefill 上比 scalar 路径**快 30%**（1085 → 1408 tok/s）。
- **每个 prefill pass 都走 wide 路径。** wide prefill 路径（融合投影、成对
  routed expert tile、WMMA attention）适用于所有宽于一个 vector batch 的
  prefill pass，不再只限 1024+ token。

### 对比 llama.cpp（v0.28.0，gfx1151）

同权重、release build、greedy；prefill＝冷 KV 单流 tok/s，decode＝开投机 tok/s
（另注除外）。正数＝MBufo 较快。六格数字皆为 CHANGELOG 量测到的 v0.28.0 开发期
数字原文重述。

| 负载 | Prefill 差值 | Decode 差值 | 条件／备注 |
| --- | --- | --- | --- |
| Flash-Next | **+52%** | 持平或更好 | 冷 KV、greedy、开 MTP |
| Qwen3.6-35B-A3B | **+52.5%** | 持平或更好 | split-K router GEMV decode；greedy |
| bge-m3（长文件） | **+35~50%** | n/a（纯 prefill） | 长文件形状、hipBLASLt autotune、快取执行器 |
| Qwen3-Embedding-0.6B（长文件） | **+41~50%** | n/a（纯 prefill） | 长文件形状、single-prefill Embed |
| Reranker（批次评分） | n/a | **超目标 +15%** | padding 多序列 ScoreBatch |
| GPT-OSS 120B | **+62%** | **-17%** | 结构性差距，照实陈述：此路径 decode 仍卡内存带宽 |

此次版本推进未跑模型服务效能测试；上表为量测到的 v0.28.0 开发期数字重述，
此处未重测。

## §3 GGUF 相容（覆盖承诺）

### 支援的模型（v0.28.0）

一个 port 经由[多模型 router](#多模型-router)服务以下全部。
`Serve`＝有 HTTP 后端；`Ladder`＝有 registry／pilot 或评分测试；`MTP/draft`＝有
投机草稿后端。共 11 列：v0.28.0 新增 8 个可用家族，加上既有的三个
（Flash-Next、27B、Gemma 4）。

| 模型 | 参数 | Serve | 梯子 | MTP/draft | 备注 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8 Flash-Next（主打） | 30B-A3B MoE | 有 | pilot | MTP | ROCmFP4-FAST preset、blocked-Q8 KV |
| Qwen3.8 27B | 27B | 有（`QWEN_KV_QUANT`、`--kv-quant` 覆写） | pilot | DFlash2（含 q8） | dense＋MoE 路径 |
| Gemma 4 | 26B-A4B | 有（HTTP 后端、原生工具） | ladder 15/15 @24k | MTP sidecar（q8 经 F16 镜像） | full-512 层留守 F16 |
| Qwen3.6-35B-A3B | 35B-A3B MoE | 仅引擎／bench | — | 无 | Q6_K→Q8_0 载入、split-K decode MoE；64K 以上 ctx 请压在 60K token 内（MODE2 重置风险） |
| GPT-OSS 20B／120B | 20B／117B MoE（MXFP4） | 有（text 后端） | pilot | 无 | 120B 零拷贝串流上传＋分块 H2D 镜像 |
| Muse Glimmer 30B | 30B | 有（ATEM 工具、kv-quant） | prompt 路径 | DFlash 端到端（~89%） | DFlash 89% 是量到的，E4 证伪为非 bug |
| Llama-3.1-8B | 8B dense GQA | 有（text 后端） | pilot adapter | 无 | — |
| Qwen3-Embedding-0.6B | 0.6B | 有（`/v1/embeddings`） | n/a（纯 prefill） | 无 | pooled single-prefill Embed |
| bge-m3 | 0.6B XLM-R | 有（`/v1/embeddings`） | encoder 测试 | 无 | CLS pooling、Q4_K_M loader、hipBLASLt autotune |
| bge-reranker-v2-m3 | cross-encoder | 有（`/v1/reranking`，BERT 分派） | ScoreBatch 测试 | 无 | padding 多序列批次 |
| Qwen3-Reranker | yes/no head | 有（`/v1/reranking`，Qwen 分派） | 评分测试 | 无 | 共用 rerank 评分头 |

DSpark 维持 legacy。模型权重不随附；各指南在 [docs/models](docs/models/README.md)
下列出所需档案。

### 端点＋工具映射

| 端点 | 方法 | 服务对象 | 备注 |
| --- | --- | --- | --- |
| `/v1/chat/completions` | POST | 文字对话（＋图像、工具、结构化输出、串流） | 空 `model` 改走默认，不再 `400 missing_model` |
| `/v1/embeddings` | POST | bge-m3、Qwen3-Embedding | 快取 single-prefill Embed 执行器 |
| `/v1/reranking` | POST | bge-reranker-v2-m3（BERT 分派）、Qwen3-Reranker（Qwen 分派） | padding 多序列 ScoreBatch |
| `/admin/models`、`/admin/load`、`/admin/eject` | GET／POST | router registry | 见[内存安全](#多模型-router) |

工具呼叫按后端家族映射——ATEM（Glimmer）、Gemma 原生（`<|tool|>` 模板、ladder
15/15 @24k）、Qwen——Qwen 路径显式 `--kv-quant` 现覆写 `QWEN_KV_QUANT`。Open WebUI、
VS Code、OpenAI SDK 与 Responses-API 客户端把 base URL 指到
`http://localhost:8080/v1` 即可；详见 [API 契约](docs/SERVER.md)。

### fallback 语义

- 未知模型 id 被拒绝，不影响已载入模型。
- 载入期 OOM 就 LRU 卸载冷门非默认模型（`allow_evict:false` →
  `409 insufficient_memory`）；默认永不卸载。
- 卸载等 in-flight 请求排空才释放 GPU 内存。
- 不带 `--multi-model` 时，服务器就是位元级一致的单模型版。
- 任何服务不了整批的注意力／GEMM 层都退回 legacy 路径——正确，只是没有加速。
- 快照以 KV 格式区分版本；某格式写出的快照不会被另一格式重用。

### 新模型接线（五步）

本树尚无独立的新模型手册（`GGUF通用化.md` 不存在，2026-11-09 已验证）；在手册
落地前，以下列清单＋[docs/models](docs/models/README.md) 下各模型指南为准。
每步附已走过的先例：

1. **EngineRegistry adapter**——在 L0／L1 factory 注册家族，按请求 `model` 分派
   （先例：27B、flash-next、gemma4、glimmer、llama）。
2. **serve 后端**——text 后端，或 embeddings single-prefill Embed，或 reranking
   BERT／Qwen 分派；把 `--kv-quant` 旗标带进去。
3. **ladder pilot 或评分测试**——registry pilot（27B、flash-next、gpt-oss、llama）、
   prompt 路径（glimmer）、encoder／评分测试（bge 家族）；纯 prefill 模型标 ladder
   N/A。
4. **草稿后端（或明确记无）**——MTP／DFlash2 挂 draft registry（gemma4 草稿取自每
   session F16 镜像以覆盖 4-bit 格）；记 `无` 要写理由（纯 prefill、尚无草稿）。
5. **serve 注册＋文件**——`--served-model-name`、free-memory 闸门估计、
   `docs/models/<家族>/` 指南（按 [docs/models](docs/models/README.md) 结构：
   README／BENCHMARKS／QUALITY／EXPERIMENTS），有已知限制就加 CHANGELOG
   Known-limits 一行。

## 快速开始

以建议的快取设定启动 Qwen3.8 Flash-Next：

```sh
./build/release/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mmproj models/qwen3.8-flash-next/mmproj-BF16.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/agentionai-MTP-Q8_0.gguf \
  --speculative mtp --think off \
  --served-model-name gufo-qwen3.8-flash-next \
  --sessions 2 -c 262144 --kv-quant q8
```

通过 OpenAI 相容 API 问它问题：

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gufo-qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "说点什么"}]
  }'
```

Open WebUI、VS Code、OpenAI SDK 与 Responses-API 的客户端，请把 base URL 指向
`http://localhost:8080/v1`。Chat Completions 与 Responses 支援文字、图片、
function tools、结构化输出与串流；详见[API 契约](docs/SERVER.md)。

## 效能基准（MBufo v0.25.5，gfx1151 Strix Halo）

冷 KV（每请求唯一 prompt，`cached_tokens≈0`）、think off、**F16 KV**、`-j 1`、
ctx 262144。四组设定：27B（UD-Q4_K_XL 对 ROCmFP4-FAST）＋ Flash-Next（ROCmFP4-FAST
对 UD-Q4_K_XL）。投机：27B＝DFlash2、Flash-Next＝MTP。（注：旧中文版此节为 v0.24.1
数字，已同步为与英文版一致的 v0.25.5。）

这些是单流数字，用来隔离模型与建置版本的选择；负载下改变的是同时能跑几条，那是
[§1 高并发](#1-高并发架构差异)的故事。

### Prefill（tok/s，越高越好）

| tokens | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---:|---:|---:|---:|---:|
| 2K | 605 | 455 | **1739** | 1301 |
| 8K | 615 | 564 | **1701** | 1549 |
| 32K | 567 | 551 | 1598 | **1618** |
| 64K | 508 | 496 | **1695** | 1678 |
| 128K | 409 | 407 | **1504** | 1507 |
| 250K | — | — | **1438** | 1415 |

> 27B ROCmFP4-**FAST** 的 prefill ≈ 非 FAST 的 STRIX 版——single-scale 只换 codebook，
> 不换 fp16-WMMA prefill 引擎（455/564/551/407 对 483/523/520/383）。FAST 真正的速度
> 来自 int8-MMQ 消费路径，而那条路径只喂给 Flash-Next 的 MoE experts，dense 27B 没接。

### Decode（tok/s，投机开）

标准：5 个冷 KV 情境（标明 temp）＋ 8 任务 ReAct 代理套件。

| 情境 | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---|---:|---:|---:|---:|
| 散文 (temp 0.4) | 23.5 | 19.7 | 43.3 | **43.6** |
| 费波那契 (temp 0.4) | 67.9 | 56.2 | **86.3** | 84.7 |
| JSON 约束 (temp 0) | **55.1** | 46.5 | 51.5 | 51.7 |
| 贪心程序码 /128 (temp 0) | 61.9 | 55.3 | 70.5 | **73.2** |
| 重复文本 /256 (temp 0) | 70.6 | 60.8 | 74.8 | **75.3** |

### 代理任务（agent_bench_v2，t1–t8，沙箱工具）

| | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---|---:|---:|---:|---:|
| 通过 | 8/8 | 7/8 | 7/8 | 7/8 |
| 核爆（须为 0） | 0 | 0 | 0 | 0 |
| 生成 tps 区间 | 9.1–31.1 | 9.3–21.5 | 16.7–56.6 | 16.8–55.1 |

### 各模型建议（目前最佳）

- **Qwen 3.8 27B → 用 UD-Q4_K_XL。** FP4-FAST 在 FP4 家族内全胜（5 个 decode 情境
  ＋代理 8/8，胜 STRIX 的 7/8），但 single-scale FP4 的 decode 与草稿接受率仍输
  Q4_K_XL，故 27B 以 Q4_K_XL 为最佳。
- **Qwen 3.8 Flash-Next → 用 ROCmFP4。** decode 最快（散文 45.0、费波那契 87.0）、
  代理 8/8，且 ~94 GB 比 Q4_K_XL 的 ~111 GB 更省，故 Flash-Next 以 ROCmFP4 为最佳。

> 27B ROCmFP4 decode 为 **v0.24.1 修复后**数值：修复前开 DFlash2 因 verify 走逐行
> GEMV 循环而崩速，修复后 FAST 追平并略胜非 FAST。

## 上游追踪

MBufo 追随上游 `main` 并以合并方式跟进，而不是分叉走自己的路。上游的 release
标签维持不可变；MBufo 的 release 标签一律加上 `mbufo-vX.Y.Z` 前缀，因此永远
不会与上游的 `vX.Y.Z` 撞名。（注：旧中文版此节停在 v0.20.0，已同步为英文版现状。）

| 项目 | 值 |
| --- | --- |
| 上游仓库 | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| 上游基准 release | `v0.26.0`（cherry-pick 安全修复；完整合并已 revert——见下） |
| 已合并的上游 `main` | cherry-pick 到 `v0.26.0`（2026-10-28）：#732 中性 sampler 字段、#736 可重复 `--served-model-name`。完整 0.21→0.26 合并**已 revert**：#758 chunked-DeltaNet prefill 让 gfx1151 FN prefill 衰退 30-46%（上游 #850 后来移除了它）；audio／Windows 与此 gfx1151 Linux 分支无关。 |
| 合并策略 | 上游 `main` 以 `--no-ff` 合并；MBufo 专属提交保持可辨识，上游历史永不改写 |

上游发布新版本时，“上游基准 release”这一列会在合并它的同一个提交里更新，
因此本 README 永远会标明这棵树确切包含到上游的哪个状态。

## 从源代码建置

支援的目标平台为 Linux x86-64 上的 AMD Strix Halo（`gfx1151`）。建置方式即
上游的建置方式，MBufo 未增加任何建置系统需求。

```sh
cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
```

请安装 C++20 编译器、CMake 3.21+、Ninja、pkg-config、ROCm 开发函式库
（HIP、hipBLAS、hipBLASLt、rocBLAS、hipCUB、rocPRIM、rocWMMA）以及一般系统
函式库（ICU、libcurl、OpenSSL、libpng、libjpeg、libwebp、FFmpeg）。目前验证
过的工具链为 GCC 15.3 搭配 ROCm 7.2.3。也提供 Nix 建置（`nix build`）。GPU
驱动必须允许你的使用者存取 `/dev/kfd` 与 `/dev/dri`；模型权重需另行取得。每个
模型指南（见 [docs/models](docs/models/README.md)）都列出所需档案与基准设定。
完整的相依性清单、ROCr queue-memory 补丁与容器相关说明，请见
[上游建置文件](https://github.com/gufo-org/gufo#build-from-source)。

## 文件索引

| 文件 | 内容 |
| --- | --- |
| [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md) | 混合批次设计：铁律、验证矩阵、量测数据 |
| [docs/KV-CACHE.md](docs/KV-CACHE.md) | 对话与元件快取（上游） |
| [docs/KV-V4.md](docs/KV-V4.md) | 4-bit K/V 区块：布局、正确性表 |
| [docs/SERVER.md](docs/SERVER.md) | HTTP API 契约 |
| [docs/CLI.md](docs/CLI.md) | 命令列参照 |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | 建置、测试、preset |
| [docs/TESTING.md](docs/TESTING.md) | 托管 CI 测试组与本机品质检查 |
| [docs/models/README.md](docs/models/README.md) | 各模型指南：README／BENCHMARKS／QUALITY／EXPERIMENTS |
| [CHANGELOG.md](CHANGELOG.md) | 上游发布历史与 MBufo release |

## 相容性与限制

- 混合批次改变的是排程，不是模型语意。任何无法服务整批的层都会退回旧路径。
- `--kv-quant q8` 是以“不再与基准完全一致”换取内存。
  当与基准的精确一致性比容量更重要时，请用 `f16`。
- 对话状态以 KV 格式区分版本；用某一种格式写出的快照不会被另一种格式重用。
- 上游的 release 标签不可变，也不会被当成 MBufo 的标签重复使用。
- 已知限制（v0.28.0，短句）：64K 以上 ctx 的 qwen35 请压在 60K token 内（驱动
  MODE2 重置风险）；DSpark 维持 legacy；gemma4 full-512 层留守 F16；Glimmer
  DFlash 约 89%（E4 证伪为非 bug）。

## 授权与致谢

上游 gufo 采 MIT 授权；改作程序码与相依套件各自保留其声明，收录于
[NOTICE](NOTICE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与
`licenses/`。MBufo 保留这些声明。模型权重不随附，依各发布者条款使用。

MBufo 是 [gufo](https://github.com/gufo-org/gufo)（by gufo-org 及贡献者）的
分支。低精度 KV 与性能调优的工作建立在上游的 Q8 KV cache、int8 WMMA、
persistence 与 history drafting 等改动之上。
