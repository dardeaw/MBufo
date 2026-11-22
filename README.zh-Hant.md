# MBufo

專為 AMD Strix Halo（`gfx1151`）打造的高併發推論引擎，面向生產環境，在公平排程下同時服務多種模型與多種量化格式，並以 FlashAttention、Split-K 解碼注意力和高效的 KV 快取壓縮節省記憶體。

最初源自 [gufo](https://github.com/gufo-org/gufo)，之後已大幅重寫。

> [English Version](README.md) · [简体中文版](README.zh-CN.md)
<img width="1429" height="829" alt="image" src="https://github.com/user-attachments/assets/fb3f41c3-5883-48c6-872e-fbf327dcf3f3" />

**核心特點**

- **高併發**：單台 Strix Halo 節點在 16 路以上併發下維持滿載，每路最長可達 256K 上下文（§1）。
- **公平排程**：預填與解碼統一混合批次，搭配防飢餓排程，高負載下不會有請求被餓死（§1）。
- **多模型單端口**：22 個 GPU + 9 個 NPU 模型並行服務，可透過內建 `/admin` 控制台在執行期載入與卸載（§3）。
- **多種量化格式**：K 系列、IQ4、MXFP4、ROCmFP4 等，各引擎的支援清單見[支援的量化格式](#支援的量化格式)（§3）。
- **FlashAttention 與 Split-K 注意力**：WMMA FlashAttention 預填與 Split-K 解碼注意力，讓長上下文解碼從短提示到 256K 都維持在記憶體頻寬上限附近（§2）。
- **高效 KV 快取**：分塊 INT8 與 4-bit 緊湊佈局，顯著降低每個會話佔用的記憶體（§2）。
- **低精度運算**：以量化權重降低記憶體頻寬需求與成本（§2）。
- **投機解碼**：支援 MTP、DFlash、DSpark（§1）。

**目錄：** [核心特性](#mbufo-核心特性) ·
[§1 高併發架構](#1-高併發架構) ·
[§2 低精度計算與成本優化](#2-低精度計算與成本優化) ·
[§3 GGUF 全家族支援與多模型路由](#3-gguf-全家族支援與多模型路由) ·
[快速開始](#快速開始) ·
[效能基準](#效能基準-gfx1151-strix-halo) ·
[從原始碼建置](#從原始碼建置) ·
[文件索引](#文件索引) ·
[相容性與限制](#相容性與限制)

---

## MBufo 核心特性

| 特性 | 指標 | 說明 |
| --- | --- | --- |
| **極速響應** | **短請求 TTFT 0.7 秒** | 具備防飢餓公平調度機制，短請求即便遭遇 25 萬 token 超長 Prefill 併發，首字延遲仍穩定壓在 0.7 秒內，保證輸出位元精確無失真。 |
| **極限省容** | **KV 快取記憶體減半** | Blocked-Q8 快取架構讓單一 262,144 (256K) token 請求的 KV 佔用從 6.44 GB 驟降至 3.42 GB（4-bit 模式更可低至 1.81 GB）。 |
| **全能託管** | **22 個 GGUF 模型家族** | 單一 HTTP 端口統一服務，涵蓋 Flash-Next、27B、Gemma 4、MoE 模型、Embedding 與 Reranker，支援動態熱載入與無重啟卸載。 |

> §1 與 §2 中除 llama.cpp 對比以外的數據（混合批次、公平性、並發引擎對比、int8 WMMA）來自早期版本，記錄於 `docs/MIXED_BATCH.md` 與 `docs/KV-V4.md`。llama.cpp 對比與「基準測試」一節已在本版本重新測量。

---

## 1. 高併發架構

在高併發服務場景中，吞吐量並非唯一指標。系統能否在滿載下保持高可用，關鍵取決於兩大核心要素：

- **ITL（Inter-Token Latency，Token 間延遲）**：請求進入串流生成後的生成間隔。若為批次吞吐而讓串流在 Prefill 階段長時間凍結，將嚴重損害互動體驗。
- **排程公平性（Fairness）**：嚴格保障不同請求、Session 與客戶端之間的公平性，避免長 Prompt 獨佔硬體算力導致短對話飢餓中斷。

上游排程具備分塊 Prefill（`--prefill-chunk`）與優先 Decode 的基礎架構，但在兩者切換時，傳統做法需要分別對模型權重進行兩次掃描，造成排程空隙與延遲損耗。MBufo 透過深度融合解決了這個瓶頸。

### 統一混合批次（Unified Mixed-Batch）

MBufo 將 Prefill Chunk 與 Decode 步驟合併至同一個 `ForwardBatch` 與注意力核心中執行，每輪計算在推進 Prefill 的同時立即輸出到期的 Decode Token。

- **延遲縮短 2 倍**：串流請求在 5 批次下的 Token 間隔由 121 ms 縮短至 **57 ms**（僅需一次權重掃描）。
- **滿載穩定**：16 路 sustained 併發下，平均耗時由 17.0 秒縮減至 **15.3 秒**，尾部延遲大幅收斂。
- **硬體高利用率**：16 路滿載運作時 GPU 利用率維持在 99%，輸出完全位元精確、零錯誤。
- **批次上限擴充**：支援高達 32 Sessions / 1024 Rows 批次規模。

> 混合批次僅重排任務執行時機，不變更任何模型運算語意。預設已開啟；亦可透過環境變數 `MBUFO_MIXED_BATCH=0` 切換回傳統排程模式。設計細節與驗證矩陣請參閱 [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md)。

### DecodeBatch 與 Split-K Decode 算子

在服務層將 Decode 步驟以預處理批次推進（`StepDecodeBatch` / `ExecutePreparedDecodeBatch`），底層注意力算子全面接入共用 Split-K 解碼讀取器（`blocked_splitk`）。在共用 Q8 下，Gemma 4 SWA 層解碼效能提升，27B 注意力層直接讀取開銷極低，且 F16 輸出維持全精度位元精確。

### 單流速度：Flash-Next 與 27B

單節點 Strix Halo、一次一個請求、冷啟動 KV、F16 KV、開啟投機解碼（tok/s，本版本經 HTTP 服務測量）。完整表格見「效能基準」一節。

| 模型 | Prefill 2K | Prefill 32K | Prefill 128K | Decode（散文） | Decode（重複文本） |
|---|---|---|---|---|---|
| Flash-Next ROCmFP4 | 1733 | 1679 | 1585 | 54.4 | 78.0 |
| Flash-Next Q4_K_XL | 1719 | 1672 | 1587 | 52.1 | 81.7 |
| 27B Q4_K_XL | 533 | 525 | 382 | 24.6 | 55.7 |
| 27B FP4-FAST | 534 | 512 | 392 | 22.5 | 52.5 |

### 負載下三引擎對比（開啟用 MTP）

在相同測試框架與充分預熱環境下（併發 1 / 2 / 3 / 4 之總 TPS 與每 Session TPS）：

| 引擎 | 總 TPS | 每 Session TPS | Fairness A / B |
| --- | --- | --- | --- |
| A (Halogen) | 85 / 82 / 75 / 88 | 85 / 41 / 25 / 22 | 322 / 1140 |
| B (gufo) | 57 / 64 / 108 / 127 | 57 / 32 / 37 / 33 | 15 / 399 |
| **C (MBufo)** | **66 / 83 / 110 / 129** | **66 / 42 / 38 / 33** | **19 / 187** |

MBufo 在多路負載下展現出最優異的線性擴展能力（總吞吐平穩攀升至 129 TPS，單路速率平緩），且具備最佳的公平性指標。

### 公平調度防飢餓架構

為杜絕超長 Prefill 擠壓短請求首字延遲，MBufo 實作三層排程防護機制：

1. **SRTF（Shortest Remaining Time First）Prefill 排程**：剩餘區塊 ≤1 的短 Prefill 優先排入執行，避免短任務在隊尾無謂等待。
2. **飢餓解碼優先搶佔（Starvation Preemption）**：當有請求等待 Session 槽位時，長 Prefill 會主動讓路、優先排空常駐 Decode，以最快速度釋放槽位（上限 32 輪，避免 Prefill 自身飢餓）。
3. **Near-Done 閘門機制**：僅在常駐解碼器距離結束 ≤64 Tokens 時觸發搶佔，若常駐均為長篇輸出則不盲目延後 Prefill，兼顧整體通暢度。

| 評測指標（gfx1151，FP4 + f16 + MTP + 4 sessions） | 實測表現 |
| --- | --- |
| Fairness-A 退化比 | **160 → 0.7** |
| Fairness-B 退化比 | **856 → 105** |
| 短請求 TTFT（首字延遲） | **3.4 秒 → 0.1 秒** |
| ITL（Token 間延遲） | 穩定不增加 |
| 貪婪輸出正確性 | 位元一致（SHA 比對通過） |

### 高效投機解碼（Speculative Decoding）

透過統一 Draft Registry 整合多種模型專屬的草稿架構：

- **Flash-Next MTP**：併發寬度 2 / 3 / 4 下接受率維持在 **88.7% / 91.6% / 89.5%**，打破隨批次衰減的瓶頸，總吞吐達到 97 / 111 / 133 tok/s。
- **27B DFlash2**：原生整合 Blocked-Q8 KV 快取，驗證流程直接寫入 Blocked Sidecar，保持精準一致。
- **Gemma 4 MTP**：採用異步 GPU 草稿與單 Forward 驗證，長上下文解碼速度達 **74–86 tok/s**（`d=3` 深度下每輪平均接受 1.74–1.78 tokens）。
- **Muse Glimmer DFlash**：端到端投機接受率達 **~89%**。

### Prefix RAM 快取與生產級 Session 配置

- 重複 Prefix 直接從記憶體快取讀取（共用邊界、可自訂 RAM 上限），避免重複計算。
- 多 Session 調度經全面校準，消除併發時的階梯式延遲，支援多路並行加速。

| 部署目標 | 建議參數設定 | 資源佔用概況 |
| --- | --- | --- |
| 16 路高併發服務 | `--sessions 16 -c 65536 --kv-quant q8` | 約 103 GB（保留約 20 GB 餘裕） |
| 8 路超長對話服務 | `--sessions 8 -c 131072 --kv-quant q8` | 約 101 GB（保留約 21 GB 餘裕） |
| 4 路極限 256K 對話 | `--sessions 4 -c 262144 --kv-quant q8` | 約 101 GB（保留約 21 GB 餘裕） |
| 單路極限 256K 對話 | `--sessions 1 -c 262144 --kv-quant q8` | 輕量佔用，極致吞吐 |
| 單路最高精度 Prefill | `--sessions 1 -c 262144 --kv-quant f16` | 全精度 KV，無量化損耗 |

---

## 2. 低精度計算與成本優化

MBufo 主張精度與效能之間的平衡需經嚴格量測。推薦基準配置為 `--kv-quant q8` 搭配 ROCmFP4 或 Q4_K 權重。

### KV 快取：Blocked Q8 與 4-bit 緊湊佈局

底層將 Blocked Q8 KV 抽象為標準化編解碼器，支援 Key 與 Value 獨立選擇 4-bit（每 32 元素佔 18 位元組）或 8-bit（每 32 元素佔 34 位元組）精度。共用核心整合了 Blocked 寫入器、直讀器、WMMA Prefill 讀取器與 Split-K 解碼讀取器。

| 模式 | 狀態 | 每位置位元組 | 256K Session 記憶體 | 說明 |
| --- | --- | --- | --- | --- |
| `f16` | 預設基準 | 2048 B | 6.44 GB | 基準數值，無 KV 量化。 |
| `q8` | **推薦模式** | 1088 B | 3.42 GB | Blocked 8-bit，平衡性最佳。 |
| `q8` + `MBUFO_KV_V4=1` | 手動開啟 (K8V4) | 832 B | 2.62 GB | Value 壓縮為 4-bit，Key 維持 8-bit。 |
| `q8` + 兩者開啟 | 手動開啟 (K4V4) | 576 B | 1.81 GB | Key / Value 均壓縮為 4-bit，極致節省記憶體。 |

- **推薦使用 `q8`**：Blocked-Q8 在編解碼層面表現優異，精度極高，記憶體直接省半。
- **4-bit 適用場景**：專為極限記憶體受限機型設計的環境變數開關，未啟用時完全不影響基準數值。詳情參閱 [docs/KV-V4.md](docs/KV-V4.md)。

### 權重精度架構

- **ROCmFP4 / ROCmFP4-FAST**：4.25 bpw 單尺度 FAST 專家解碼藉由 fp16 WMMA 矩陣核心硬體直驅，使 MoE Prefill 充分享受矩陣核心加速。在 gfx1151 上，FAST 模型在長文本 Prefill 下表現亮眼，解碼更較傳統方式大幅提升。
- **MXFP4**：支援 GPT-OSS 20B / 120B 原生 MXFP4 路由運算，Decode 端融合 MoE / QKV / 投影路徑，120B 支援零拷貝串流上傳。
- **Q4_K**：UD-Q4_K_XL 為 27B 與 Flash-Next 的參考優化架構；bge-m3 提供 Q4_K_M 載入器。
- **W8A8**：共用主機端 16-split varlen GEMM 核心，支援 Glimmer 與 Flash-Next Prefill。

### 注意力運算核心優化

- **int8 WMMA Direct Prefill**：單 Session Causal Prefill 直接讀取 Blocked int8 快取，跳過 F16 暫存反量化，14K Prefill 速度提升 **30%**（1085 → 1408 tok/s）。
- **Split-K Flash-Decoding**：結合融合解碼 GEMV、融合 Norm / Residual / Epilogue 核心及裝置端 Argmax。
- **Wide Prefill 全面覆蓋**：針對所有寬於 Vector Batch 的 Prefill 請求啟用融合投影與成對矩陣核心處理。

### 效能對比 llama.cpp（gfx1151 實測）

相同權重、貪婪解碼、Release 建置，在同一台 Strix Halo 節點上對 llama.cpp `5d9e5ac30` 重新實測。MBufo 經由 HTTP 服務端測量，數字含分詞與請求開銷；llama.cpp 使用 `llama-bench` / `llama-server`，並取其最佳批次大小。Prefill 為約 2K token 的冷提示；Decode 為約 2K token 上下文下的生成速度；均未開啟投機解碼。MBufo 使用預設的分塊 Q8 KV 快取，llama.cpp 使用 F16（下文「效能基準」使用 F16）。正值代表 MBufo 更快。

| 模型（權重） | Prefill tok/s：MBufo / llama.cpp | 差距 | Decode tok/s：MBufo / llama.cpp | 差距 |
| --- | --- | --- | --- | --- |
| **Gemma 4 26B-A4B**（Q4_0） | 2689 / 1728 | **+56%** | 55.3 / 53.2 | +4% |
| **Qwen3.6-35B-A3B**（UD-Q4_K_XL） | 2130 / 1230 | **+73%** | 54.4 / 50.7 | +7% |
| **GPT-OSS 120B**（Q4_K_M） | 1226 / 1089 | **+13%** | 52.3 / 52.4 | 0% |
| **Muse Glimmer 30B**（UD-Q4_K_XL） | 516 / 398 | **+30%** | 12.6 / 13.2 | -5% |
| **DeepSeek V4 Flash**（IQ2_XXS 混合） | 432 / 63 | **+586%** | 15.8 / 14.2 | +12% |
| **Laguna-XS-2.1**（Q4_K_M） | 1586 / 930 | **+71%** | 60.4 / 61.8 | -2% |
| **Laguna-S-2.1**（Q4_K_M） | 652 / 497 | **+31%** | 26.6 / 26.6 | 0% |

Embedding 與重排序（經由 HTTP 服務端的單次請求，長度單位為 token）：

| 模型（權重） | MBufo / llama.cpp | 差距 |
| --- | --- | --- |
| **bge-m3**（Q4_K_M），0.9K | 16651 / 5663 tok/s | **+194%** |
| **bge-m3**（Q4_K_M），1.7K | 14101 / 5032 tok/s | **+180%** |
| **bge-m3**（Q4_K_M），5K | 7426 / 3324 tok/s | **+123%** |
| **bge-m3**（FP16），1.7K | 13895 / 4962 tok/s | **+180%** |
| **Qwen3-Embedding-0.6B**（Q8_0），0.9K | 15481 / 7454 tok/s | **+108%** |
| **Qwen3-Embedding-0.6B**（Q8_0），1.7K | 13981 / 5301 tok/s | **+164%** |
| **Qwen3-Embedding-0.6B**（Q8_0），3.4K | 11356 / 4700 tok/s | **+142%** |
| **bge-reranker-v2-m3**（Q4_K_M），64 篇文件 | 76.9 / 38.0 docs/s | **+102%** |
| **bge-reranker-v2-m3**（FP16），64 篇文件 | 73.1 / 34.3 docs/s | **+113%** |
| **Qwen3-Reranker-0.6B**（Q8_0），64 篇文件 | 41.1 / 28.1 docs/s | **+46%** |

說明：
- DeepSeek V4 的對照為該 llama.cpp 建置自帶的 `deepseek4` 支援，使用預設批次大小 512；本次未使用 ds4 參考專案。
- llama.cpp 批次大小：Gemma 4 與 Glimmer 為 2048，Qwen3.6 為 1024（2048 曾使測試機當機，其最佳成績可能更高），GPT-OSS 經 `llama-server` 為 8192。
- bge-m3 單次輸入可達模型原生的 8192 token。輸入長度會按約 12% 的間隔分桶並做精確的鍵遮罩，不同長度的請求因此共用已調優的 GEMM 方案；調優結果快取在磁碟（`MBUFO_TUNE_CACHE`，默认 `~/.cache/mbufo/bge_tune_v1.txt`），新形狀的第一次請求仍需付出一次性調優時間。
- Qwen3-Reranker：MBufo 套用該模型要求的提示模板（每篇文件約多 27% token），並且每次請求只預填一次共用的系統指令與查詢前綴，而不是每篇文件各算一次（分數完全相同）。該 llama.cpp 建置的 `/rerank` 未套用，在 8 篇相關與 40 篇無關文件的測試中，它的前 8 名裡沒有任何一篇相關文件，MBufo 則把 8 篇相關文件全排在最前。

- EmbeddingGemma 2（BF16，文字路徑；`tools/convert/embeddinggemma2_to_gguf.py`）尚不受 llama.cpp 支援，因此沒有 llama.cpp 對照。MBufo 的向量與獨立的 PyTorch 實作一致（6 段 4 到 908 token 的文字，餘弦相似度 1.000000）。經 HTTP 服務，15 到 23 token 的請求約 10 ms，908 token 為 76 ms（11.9K tok/s）；同一節點上以 32 個 CPU 執行緒（FP32）執行的 PyTorch 實作，13 token 需要 24.7 ms，908 token 需要 364 ms（這台機器上的 PyTorch ROCm 無法執行）。

---

## 3. GGUF 全家族支援與多模型路由

### 支援的模型家族

透過內建多模型路由器，單一 HTTP 端口即可統一調度：上方 22 個 GPU 家族，下方分隔線起 9 個 NPU 項目：

| 模型 | 參數量 | 投機草稿支援 | 特性與架構亮點 |
| --- | --- | --- | --- |
| **Qwen3.8 Flash-Next**（主力） | 125B-A6B MoE | MTP | ROCmFP4-FAST / UD-Q4_K_XL、Blocked-Q8 KV |
| **Qwen3.8 27B** | 27B Dense | DFlash2 (含 Q8) | 支援 Dense 與 MoE 路徑，推薦 UD-Q4_K_XL |
| **Gemma 4 26B-A4B** | 26B-A4B | MTP (F16 鏡像) | 具備原生 Tool 支援，SWA 共用 Q8 |
| **Gemma 4 E4B / E2B** | 4B / 2B 有效參數（逐層嵌入） | — | Q4_0 QAT；PLE（Q6_K gather）、共用 KV 層、WMMA 大 head prefill |
| **Gemma-4-12B** | 12B Dense | — | 支援 K-quant 預轉換與 WMMA Q8 Prefill |
| **Qwen3.5-35B-A3B** | 35B-A3B MoE | — | 支援 qwen35moe 架構路徑 |
| **Qwen3.6-35B-A3B** | 35B-A3B MoE | — | Q6_K→Q8_0 載入、Split-K MoE 解碼 |
| **Ornith-1.5-35B-A3B** | 35B-A3B MoE | — | `qwen35moe` 混合 SSM/Attention、ROCmFP4 / Q4_K_M + BF16 視覺 (`width=2048`) |
| **Tiel-Coder / Cyber-Tiel-Coder-35B-A3B** | 35B-A3B MoE | — | 基於 Ornith-1.5 (`qwen35moe`) 之程式開發與去審查資安微調版；支援 Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 (`-GGUF-MTP`) |
| **GPT-OSS 20B / 120B** | 20B / 117B MoE | — | 原生 MXFP4 矩陣核心、120B 支援零拷貝串流上傳 |
| **Muse Glimmer 30B** | 30B | DFlash (~89%) | ATEM 工具呼叫、W8A8 / Q8_0 支援 |
| **Llama-3.1-8B** | 8B Dense | — | 標準 Dense GQA 支援 |
| **DeepSeek V4 Flash** | MoE（IQ2/Q2/Q8 混合 GGUF） | DSpark | MLA decode、ds4 一致 kernel、約 81 GiB 常駐 |
| **Laguna 2.1 (XS / S)** | 33B-A3B / 118B-A8B MoE | — | 1:3 全域（YaRN）/ SWA-512 雙 RoPE、per-head softplus gate、sigmoid+bias top-8/10 MoE + 共享專家 |
| **Qwen3-Embedding-0.6B** | 0.6B | — | 快取池化 Single-Prefill 執行器 |
| **bge-m3** | 0.6B XLM-R | — | CLS 池化、Q4_K_M 載入器、hipBLASLt 調優 |
| **EmbeddingGemma 2** | 約 0.3B Gemma-4 編碼器（文字路徑） | — | 雙向注意力：20 層滑動視窗（±512）與 4 層全域層（head_dim 256 / 512）、逐層輸入、平均池化、768 維輸出 |
| **bge-reranker-v2-m3** | Cross-Encoder | — | BERT 分派、多序列 Padding 批次加速 |
| **Qwen3-Reranker** | Cross-Encoder | — | Qwen 分派、共用 Rerank 評分頭 |
| **Qwen3-ASR 1.7B** | 1.7B | — | BF16 safetensors、決定性 greedy 解碼 |
| **Qwen3-TTS 12Hz 1.7B** | 1.7B | — | BF16 safetensors、F32 波形解碼器 |
| **Qwen-Image-2.1** | Diffusion transformer | — | BF16 safetensors、Qwen3-VL 提示編碼器、RGBA VAE |
| **MiniMax H3** | 50 區塊 BF16 transformer | — | HIP transformer、VisualVAE / AudioVAE、MP4 合成 |

---

### ⚡ NPU 模型（XDNA2 Strix Halo NPU）

| 模型 | 參數量 | 格式／後端 | HTTP 路由 | 狀態 |
| --- | --- | --- | --- | --- |
| **D3-Lite / Nano / Mini / Turbo / Ultra** | 0.8B / 2B / 4B / — / — | Q4NX（flm NPU2） | `POST /v1/systemone` | ✅ 已上線（router 代理＋admin 卡片；⚡ DECIDE 專用，qwen3.8 維持 CHAT） |
| **Whisper-V3-Turbo** | 0.8B | Q4NX NPU2（5 個 xclbin） | `POST /v1/audio/transcriptions` | ✅ 已上線（代理；1.4 秒音檔 → 2.3 秒已驗證） |
| **EmbeddingGemma 2** | 約 0.3B | Q8 bfp16ebs8 / Q4 q4nx | `/v1/embeddings`（NPU lane） | 🧪 執行期已驗證（24 層端到端 cos > 0.993；引擎已接縫，預設 kNone） |
| **bge-m3** | 0.6B XLM-R | Q8 / Q4 | `/v1/embeddings`（NPU lane） | 🧪 執行期已驗證（64×1024 GEMM cos 0.999988；編碼器 cos 0.999924） |
| **bge-reranker-v2-m3** | Cross-Encoder | Q8 / Q4 | `/v1/reranking`（NPU lane） | 🧪 執行期已驗證（評分差約 1e-5） |
| **Qwen3-ASR 1.7B** | 1.7B | Q8 / Q4 | `/v1/audio/transcriptions`（現由 GPU sidecar 提供） | 🧪 NPU 執行期已驗證（cos > 0.967，top-1 吻合） |
| **Qwen3-TTS 12Hz 1.7B** | 1.7B | Q8 / Q4 | `/v1/audio/speech`（現由 GPU sidecar 提供） | 🧪 NPU 執行期已驗證（已產出 wav，cos > 0.98） |
| **FLUX.2-klein-4B** | 4B DiT | Q8 / Q4 | 測試線束（輸出 PNG） | 🧪 執行期已驗證（cos > 0.997；尚無 HTTP 服務） |
| **Z-Image-Turbo** | DiT | Q8 / Q4 | 測試線束（輸出 PNG） | 🧪 執行期已驗證（cos > 0.997；尚無 HTTP 服務） |

### 支援的量化格式

各引擎皆直接讀取 GGUF；下表為各引擎接受的格式（不在清單內的格式會在載入時連同張量名稱一併拒絕，絕不會被當成其他格式解碼）。細節、限制與各組合的驗證方式見 [docs/QUANT_FORMATS.md](docs/QUANT_FORMATS.md)。

| 引擎 | 稠密／注意力權重 | 路由專家 |
| --- | --- | --- |
| **Qwen 27B / Qwen3.5-3.6 MoE** | Q8_0、Q4_K、Q5_K、Q6_K、Q3_K、IQ4_NL、IQ4_XS、IQ3_S、ROCmFP4、ROCmFP4_FAST、F16 / BF16 / F32 | Q4_0、Q4_1、Q5_0、Q5_1、Q8_0、Q2_K、Q4_K、Q5_K、Q6_K、IQ2_XXS、MXFP4、ROCmFP4、ROCmFP4_FAST |
| **Qwen3.8 Flash-Next** | 原生：Q8_0、Q6_K、ROCmFP4 / FAST、MXFP4、F16 / BF16 / F32；載入時轉為 Q8_0：Q4_K、Q5_K、Q4_0、Q4_1、Q5_0、Q2_K、Q3_K、IQ4_XS、IQ4_NL | Q4_K、Q5_K、Q6_K、Q5_1、Q8_0、Q4_0、Q3_K、Q2_K、ROCmFP4_FAST、MXFP4 |
| **Gemma 4（26B-A4B、E4B、E2B、12B）** | Q4_0、Q8_0、Q4_K、Q5_K、Q6_K、F16 / BF16 / F32 | 同左（Q4_0 走快速 MoE 路徑） |
| **GPT-OSS 20B / 120B** | Q8_0、Q5_0、Q4_K 及 F16 / BF16 / F32 注意力 | MXFP4（原生）、Q4_0、Q8_0、Q5_1（K 系列無法編碼其 2880 寬的專家） |
| **Muse Glimmer、Llama-3.1** | Q4_0、Q8_0、Q4_K、Q5_K、Q6_K、F16 / BF16 / F32 | - |
| **DeepSeek V4 Flash** | 已發佈的 IQ2_XXS / Q2_K / Q8_0 混合配方 | 同左 |
| **bge-m3、重排序、Qwen3-Embedding** | Q4_K_M 與 F16 載入器 | - |
| **EmbeddingGemma 2** | BF16（由 `tools/convert/embeddinggemma2_to_gguf.py` 產生 GGUF） | - |

嵌入查表、行反量化、行點積核心與主機端反量化器共用同一份格式註冊表（`core/quant/quant_traits.hpp`）：F32、F16、BF16、Q4_0、Q4_1、Q5_0、Q5_1、Q8_0、Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、IQ4_NL、IQ4_XS、IQ2_XXS、MXFP4、ROCmFP4、ROCmFP3。KV 快取：F16，或分塊 INT8（Q8），記憶體約減半，並保持精確連續性的 split-K 注意力。

### 生產級異質多模型並駐（Multi-Model Co-Residency）與 `/admin` 控制台

執行 `serve llm --multi-model` 可將 MBufo 轉為單一端口（如 `:8741` 或 `:8080`）的**異質多模型並駐伺服器**。多顆不同架構的 GGUF 模型（MoE 主力模型 + Dense 次要模型 + Vision `mmproj` 視覺掛件 + `MTP`/`DFlash2` 投機解碼草稿模型）可**同時常駐於 Strix Halo 統一記憶體（UMA）中並發服務**，無須反覆卸載重載。

#### 核心能力與架構亮點
- **異質多模型同時並駐**：只要統一記憶體容量允許（例如 128 GB UMA 同時載入 94 GiB `Qwen3.8-Flash-Next` 與 13 GiB `Qwen3.8-27B`），多顆模型可同時在線並各自擁有獨立的 Session Slots 與 KV 快取池。客户端只需在 `/v1/chat/completions` 或 `/v1/messages` 的 JSON payload 指定 `"model"`（支援 ID 或自訂別名 `aliases`）即可精準路由。
- **內建 `/admin` Web UI 管理控制台**：瀏覽器直接開啟 `http://<host>:<port>/admin` 即可使用：
  - **頂部固定式 Unified Memory & VRAM Gate**：即時顯示 Strix Halo UMA 各模型常駐佔用、System/KV 緩衝池與安全保留區；滑鼠懸停在目錄模型上時，以單行即時預覽「可直接並駐」或「需觸發 LRU 置換」的記憶體差額。
  - **已載入模型與即時 Slot 監控**：即時查看每顆模型的每個 Session Slot 狀態（`BUSY` / `IDLE`、KV 快取 Token 數、閒置時間），一鍵切換 Primary 預設模型或無損卸載。
  - **本地 GGUF 模型目錄自動掃描（`/admin/catalog`）**：自動遞迴掃描 `MBUFO_MODELS_DIR`（或模型所在母目錄），自動合併多分割檔（shards）並自動配對同目錄下的 `mmproj` 視覺權重與 `MTP`/`DFlash2` 草稿模型。
  - **內建多模型串流 Playground**：直接在網頁上選擇任一並駐模型發送 OpenAI / Anthropic 串流請求，即時量測 TTFT 與 Decode tok/s。
- **零模型待機啟動（Zero-Model Standby）**：支援不帶 `--model` 直接執行 `mbufo serve llm --multi-model -p 8741`（僅佔用約 86 MiB RSS），啟動後直接從 `/admin` 網頁點選載入所需模型。
- **安全記憶體閘門與無損排空**：`/admin/load` 自動保留安全邊界（`MBUFO_MULTI_SAFETY_MIB`，預設 `2048` MiB）。空間不足時可自動對閒置的非預設模型執行 **LRU 置換卸載**（若設 `"allow_evict": false` 則回傳 `409 insufficient_memory` 拒絕載入以保護現有並駐模型）；卸載模型時會等待進行中的請求生成完畢才釋放記憶體。

#### 端點一覽

| 端點 | 方法 | 功能說明 |
| --- | --- | --- |
| `/admin` | GET | 內建多模型並駐管理 Web UI 儀表板與即時測試 Playground |
| `/v1/chat/completions` | POST | OpenAI 聊天對話介面（依 `"model"` ID/別名自動路由，未填則回退至預設模型） |
| `/v1/messages` | POST | Anthropic Messages 介面（依 `"model"` ID/別名自動路由，支援 Claude Code 等工具） |
| `/v1/embeddings` | POST | 向量嵌入介面（支援 bge-m3、Qwen3-Embedding、EmbeddingGemma 2） |
| `/v1/reranking` | POST | 文本重排序介面（支援 bge-reranker-v2-m3、Qwen3-Reranker） |
| `/admin/models` | GET | 查詢目前並駐模型清單、各 Slot 狀態與 UMA 記憶體遙測數據 |
| `/admin/catalog` | GET | 掃描本地 GGUF 模型庫並自動配對 `mmproj` 與 `MTP`/`DFlash2` 附屬權重 |
| `/admin/load` | POST | 動態熱載入新模型並駐（支援自訂別名、Speculative、Vision 與 LRU 保護） |
| `/admin/default` | POST | 動態切換預設主模型（`{"id": "<model-id>"}`） |
| `/admin/eject` | POST | 線上無損卸載指定模型，待任務清空後平滑釋放記憶體 |

---

## 快速開始

### 1. 啟動主推論服務（以 Qwen3.8 Flash-Next 為例）

```sh
./build/release/mbufo serve llm \
  --model models/qwen3.8-rocmfp4/v2q8k/Qwen3.8-Flash-Next-ROCmFP4-FAST-v2q8k.gguf \
  --mmproj models/qwen3.8-flash-next/mmproj-BF16.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/agentionai-MTP-Q8_0.gguf \
  --speculative mtp --think off \
  --served-model-name mbufo-qwen3.8-flash-next \
  --sessions 2 -c 262144 --kv-quant q8 -p 8741
```

### 2. 多模型並駐模式（啟用 `--multi-model` 與 `/admin` Web UI）

```sh
# 方式 A：帶預設主力模型啟動，並開啟多模型並駐與 /admin 控制台
./build/release/mbufo serve llm --multi-model \
  --model models/qwen3.8-rocmfp4/v2q8k/Qwen3.8-Flash-Next-ROCmFP4-FAST-v2q8k.gguf \
  --mmproj models/qwen3.8-flash-next/mmproj-BF16.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/agentionai-MTP-Q8_0.gguf \
  --speculative mtp --think off \
  --served-model-name mbufo-qwen3.8-flash-next \
  --sessions 2 -c 262144 --kv-quant q8 -i 0.0.0.0 -p 8741

# 方式 B：零模型待機啟動（不預載模型，直接由 http://localhost:8741/admin 網頁選單載入）
MBUFO_MODELS_DIR=/path/to/models ./build/release/mbufo serve llm --multi-model -i 0.0.0.0 -p 8741
```

啟動後直接以瀏覽器開啟 **`http://localhost:8741/admin`** 即可透過圖形化介面管理多模型並駐。

### 3. 多模型並駐 API 呼叫示範

```sh
# 動態熱載入第二顆模型（Qwen3.8-27B）與主力模型同時並駐（設定 allow_evict: false 確保不踢除現有模型）
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

# 同一端口 (:8741) 同時對兩顆並駐模型發送推論請求
curl http://localhost:8741/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mbufo-qwen3.8-flash-next", "messages": [{"role": "user", "content": "請用一句話自我介紹。"}]}'

curl http://localhost:8741/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "27b", "messages": [{"role": "user", "content": "請用一句話自我介紹。"}]}'

# 切換預設模型、查詢並駐狀態與記憶體佔用、或卸載指定模型
curl http://localhost:8741/admin/default -H "Content-Type: application/json" -d '{"id": "qwen3.8-27b"}'
curl http://localhost:8741/admin/models
curl http://localhost:8741/admin/eject -H "Content-Type: application/json" -d '{"id": "qwen3.8-27b"}'
```

相容於 OpenAI SDK、Open WebUI、VS Code 等前端工具，Base URL 設定為 `http://localhost:8080/v1` 即可。完整介面協議詳見 [docs/SERVER.md](docs/SERVER.md)。

---

## 效能基準（gfx1151 Strix Halo）

測試條件（本版本重新測量，經 HTTP 服務）：冷啟動 KV（每請求獨立 Prompt）、關閉思考模式、F16 KV、單會話 `-j 1`、上下文 262,144。27B 採用 DFlash2 投機解碼，Flash-Next 採用 MTP 投機解碼。Prefill 為服務端回報的提示處理速率；Decode 為 128 至 256 token 生成三次的中位數，會隨草稿接受率波動。

### Prefill 吞吐量（tok/s，越高越好）

| Token 規模 | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---:|---:|---:|---:|---:|
| **2K** | 533 | 534 | **1733** | 1719 |
| **8K** | 551 | 515 | **1708** | 1678 |
| **32K** | 525 | 512 | **1679** | 1672 |
| **64K** | 467 | 479 | **1643** | 1637 |
| **128K** | 382 | 392 | 1585 | **1587** |
| **250K** | — | — | 1512 | **1518** |

### Decode 速度（tok/s，開啟投機解碼）

| 測試場景 | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---|---:|---:|---:|---:|
| 散文生成 (temp 0.4) | 24.6 | 22.5 | **54.4** | 52.1 |
| 費氏數列生成 (temp 0.4) | 41.5 | 41.9 | **59.2** | 56.0 |
| JSON 結構生成 (temp 0) | 43.8 | 44.3 | 63.6 | **64.2** |
| 程式碼貪婪生成 /128 (temp 0) | 44.2 | 37.9 | 50.6 | **52.7** |
| 重複文本生成 /256 (temp 0) | 55.7 | 52.5 | 78.0 | **81.7** |

### Agentic 綜合評測（agent_bench_v2 沙箱工具組）

_本版本未重新測量：此表來自早期版本。_

| 評測項目 | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---|---:|---:|---:|---:|
| 通過任務數 | **8/8** | 7/8 | 7/8 | 7/8 |
| 嚴重失誤次數 | **0** | **0** | **0** | **0** |
| 生成 TPS 範圍 | 9.1–31.1 | 9.3–21.5 | 16.7–56.6 | 16.8–55.1 |

### 各模型推薦配置

- **Qwen 3.8 27B**：推薦採用 **UD-Q4_K_XL** 作為參考版本；在上述測量中，兩種建置的解碼速度在上述測量中相差約 15% 以內。
- **Qwen 3.8 Flash-Next**：推薦採用 **ROCmFP4**，解碼吞吐極高（散文 54 tok/s、費氏數列 59 tok/s），且僅需 ~94 GB 記憶體（對比 Q4_K_XL 節省約 17 GB）。

---

## 從原始碼建置

目標環境為 Linux x86-64 上的 AMD Strix Halo（`gfx1151`）。編譯流程與上游標準完全一致：

```sh
cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/mbufo diagnose
```

### 環境相依性需求
- **編譯器與工具**：支援 C++20 之編譯器（推薦 GCC 15.3）、CMake 3.21+、Ninja、pkg-config。
- **運算庫**：ROCm 7.2.3 開發函式庫（HIP、hipBLAS、hipBLASLt、rocBLAS、hipCUB、rocPRIM、rocWMMA）。亦支援 Nix 建置（`nix build`）。
- **系統相依**：ICU、libcurl、OpenSSL、libpng、libjpeg、libwebp、FFmpeg。
- **權限**：執行帳號需具備存取 `/dev/kfd` 與 `/dev/dri` 節點之權限。

完整建置說明可參考 [上游建置文件](https://github.com/gufo-org/gufo#build-from-source)。

---

## 文件索引

| 文件路徑 | 說明內容 |
| --- | --- |
| [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md) | 混合批次架構設計、核心不變量與實測矩陣 |
| [docs/KV-CACHE.md](docs/KV-CACHE.md) | 對話快取與元件生命週期管理 |
| [docs/KV-V4.md](docs/KV-V4.md) | 4-bit KV 區塊佈局規格與數值精確度量測 |
| [docs/SERVER.md](docs/SERVER.md) | HTTP API 介面規格與通訊協議 |
| [docs/CLI.md](docs/CLI.md) | 命令列參數完整指南 |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | 開發者指引、測試配置與編譯 Preset |
| [docs/TESTING.md](docs/TESTING.md) | 自動化 CI 測試組與本機驗證規範 |
| [docs/models/README.md](docs/models/README.md) | 各模型家族專屬調優與配置指南 |
| [CHANGELOG.md](CHANGELOG.md) | 專案歷程與詳細更新日誌 |

---

## 相容性與限制

- **語意不變性**：混合批次僅調度排程，絕不改變模型數學語意；若遇到不支援整批運算之算子，自動安全回退。
- **精度選擇**：`--kv-quant q8` 以極微小的精度折損換取倍數級顯存空間；若應用對數值一致性有絕對嚴格要求，請使用 `f16`。
- **快照隔離**：KV 狀態快照嚴格按格式版本區隔，避免跨精度誤用。
- **模型調度提醒**：Qwen 3.5 超過 64K 上下文時，建議將實際長度控制在 60K 內以確保驅動層級穩定；Gemma 4 之 Full-512 層維持 F16 運行。

---

## 授權與致謝

上游 gufo 採用 MIT 授權條款；第三方函式庫授權聲明詳見 [NOTICE](NOTICE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 與 `licenses/`。模型權重檔案依其各原始發布者之許可協議規範。

MBufo 起步於 [gufo](https://github.com/gufo-org/gufo)。感謝 gufo-org 及其貢獻者在 Q8 KV 快取、int8 WMMA 與投機解碼上的工作，為本專案奠定了基礎。
