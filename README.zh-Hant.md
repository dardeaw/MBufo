# MBufo

專為 AMD Strix Halo（`gfx1151`）架構深度打造的 [gufo](https://github.com/gufo-org/gufo) 分支，定位為**高併發、極致公平性與低成本的生產級推論引擎**。

> [English Version](README.md) · [简体中文版](README.zh-CN.md)

上游 gufo 專精於單串流極限吞吐；MBufo 則聚焦於伺服端實際服務的核心挑戰：單台 Strix Halo 節點在面臨十六路以上高併發請求時如何保持滿載，且每路請求如何承載長達 256K 的極限上下文。MBufo 以三大核心支柱解答：**負載下的公平排程（§1）**、**極致省電節省記憶體的低精度計算（§2）**，以及**單端口整合全家族的 GGUF 相容支援（§3）**。底層引擎、運算核心與核心架構源自上游，在此誠摯致謝 gufo-org 及廣大社群貢獻者。

**目錄：** [核心特性](#mbufo-核心特性) ·
[§1 高併發架構](#1-高併發架構) ·
[§2 低精度計算與成本優化](#2-低精度計算與成本優化) ·
[§3 GGUF 全家族支援與多模型路由](#3-gguf-全家族支援與多模型路由) ·
[快速開始](#快速開始) ·
[效能基準](#效能基準-gfx1151-strix-halo) ·
[上游追蹤](#上游追蹤) ·
[從原始碼建置](#從原始碼建置) ·
[文件索引](#文件索引) ·
[相容性與限制](#相容性與限制)

---

## MBufo 核心特性

| 特性 | 指標 | 說明 |
| --- | --- | --- |
| **極速響應** | **短請求 TTFT 0.7 秒** | 具備防飢餓公平調度機制，短請求即便遭遇 25 萬 token 超長 Prefill 併發，首字延遲仍穩定壓在 0.7 秒內，保證輸出位元精確無失真。 |
| **極限省容** | **KV 快取記憶體減半** | Blocked-Q8 快取架構讓單一 262,144 (256K) token 請求的 KV 佔用從 6.44 GB 驟降至 3.42 GB（4-bit 模式更可低至 1.81 GB）。 |
| **全能託管** | **13 個 GGUF 模型家族** | 單一 HTTP 端口統一服務，涵蓋 Flash-Next、27B、Gemma 4、MoE 模型、Embedding 與 Reranker，支援動態熱載入與無重啟卸載。 |

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

> 混合批次僅重排任務執行時機，不變更任何模型運算語意。預設已開啟；亦可透過環境變數 `GUFO_MIXED_BATCH=0` 切換回傳統排程模式。設計細節與驗證矩陣請參閱 [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md)。

### DecodeBatch 與 Split-K Decode 算子

在服務層將 Decode 步驟以預處理批次推進（`StepDecodeBatch` / `ExecutePreparedDecodeBatch`），底層注意力算子全面接入共用 Split-K 解碼讀取器（`blocked_splitk`）。在共用 Q8 下，Gemma 4 SWA 層解碼效能提升，27B 注意力層直接讀取開銷極低，且 F16 輸出維持全精度位元精確。

### 公平調度防飢餓架構

為杜絕超長 Prefill 擠壓短請求首字延遲，MBufo 實作三層排程防護機制：

1. **SRTF（Shortest Remaining Time First）Prefill 排程**：剩餘區塊 ≤1 的短 Prefill 優先排入執行，避免短任務在隊尾無謂等待。
2. **飢餓解碼優先搶佔（Starvation Preemption）**：當有請求等待 Session 槽位時，長 Prefill 會主動讓路、優先排空常駐 Decode，以最快速度釋放槽位（上限 32 輪，避免 Prefill 自身飢餓）。
3. **Near-Done 閘門機制**：僅在常駐解碼器距離結束 ≤64 Tokens 時觸發搶佔，若常駐均為長篇輸出則不盲目延後 Prefill，兼顧整體通暢度。

| 評測指標（gfx1151，FP4 + f16 + MTP + 4 sessions） | 實測表現 |
| --- | --- |
| Fairness-A 退化比 | **160 → 7.32** |
| Fairness-B 退化比 | **856 → 191** |
| 短請求 TTFT（首字延遲） | **3.4 秒 → 0.7 秒** |
| ITL（Token 間延遲） | 穩定不增加 |
| 貪婪輸出正確性 | 位元一致（SHA 比對通過） |

### 負載下三引擎對比（開啟用 MTP）

在相同測試框架與充分預熱環境下（併發 1 / 2 / 3 / 4 之總 TPS 與每 Session TPS）：

| 引擎 | 總 TPS | 每 Session TPS | Fairness A / B |
| --- | --- | --- | --- |
| A (Halogen) | 85 / 82 / 75 / 88 | 85 / 41 / 25 / 22 | 322 / 1140 |
| B (gufo 上游) | 57 / 64 / 108 / 127 | 57 / 32 / 37 / 33 | 15 / 399 |
| **C (MBufo)** | **66 / 83 / 110 / 129** | **66 / 42 / 38 / 33** | **19 / 187** |

MBufo 在多路負載下展現出最優異的線性擴展能力（總吞吐平穩攀升至 129 TPS，單路速率平緩），且具備最佳的公平性指標。

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
| `q8` + `GUFO_KV_V4=1` | 手動開啟 (K8V4) | 832 B | 2.62 GB | Value 壓縮為 4-bit，Key 維持 8-bit。 |
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

相同權重、Release 建置、貪婪解碼；正值代表 MBufo 領先幅度：

| 模型與任務 | Prefill 領先幅度 | Decode 表現 | 備註與測試條件 |
| --- | --- | --- | --- |
| **Flash-Next** | **+52%** | 持平或更優 | 冷啟動 KV、開啟 MTP |
| **Qwen3.6-35B-A3B** | **+52.5%** | 持平或更優 | Split-K Router 解碼 |
| **bge-m3（長文本）** | **+35% ~ +50%** | — (純 Prefill) | hipBLASLt 自動調優與快取執行器 |
| **Qwen3-Embedding-0.6B** | **+41% ~ +50%** | — (純 Prefill) | 單 Prefill 優化 |
| **Reranker（批次評分）** | — | **超額 +15%** | 多序列 Padding 批次化 |
| **GPT-OSS 120B** | **+62%** | -17% | 超大參數量，Prefill 顯著領先 |

---

## 3. GGUF 全家族支援與多模型路由

### 支援的模型家族

透過內建多模型路由器，單一 HTTP 端口即可統一調度下列全部 13 個家族：

| 模型 | 參數量 | HTTP 服務 | 驗證測試 | 投機草稿支援 | 特性與架構亮點 |
| --- | --- | --- | --- | --- | --- |
| **Qwen3.8 Flash-Next**（主力） | 30B-A3B MoE | 支援 | Pilot | MTP | ROCmFP4-FAST / UD-Q4_K_XL、Blocked-Q8 KV |
| **Qwen3.8 27B** | 27B Dense | 支援 | Pilot | DFlash2 (含 Q8) | 支援 Dense 與 MoE 路徑，推薦 UD-Q4_K_XL |
| **Gemma 4 26B-A4B** | 26B-A4B | 支援 | 完整測試 | MTP (F16 鏡像) | 具備原生 Tool 支援，SWA 共用 Q8 |
| **Gemma-4-12B** | 12B Dense | Bench | — | — | 支援 K-quant 預轉換與 WMMA Q8 Prefill |
| **Qwen3.5-35B-A3B** | 35B-A3B MoE | Engine | — | — | 支援 qwen35moe 架構路徑 |
| **Qwen3.6-35B-A3B** | 35B-A3B MoE | Engine | — | — | Q6_K→Q8_0 載入、Split-K MoE 解碼 |
| **GPT-OSS 20B / 120B** | 20B / 117B MoE | 支援 | Pilot | — | 原生 MXFP4 矩陣核心、120B 支援零拷貝串流上傳 |
| **Muse Glimmer 30B** | 30B | 支援 | Prompt | DFlash (~89%) | ATEM 工具呼叫、W8A8 / Q8_0 支援 |
| **Llama-3.1-8B** | 8B Dense | 支援 | Adapter | — | 標準 Dense GQA 支援 |
| **Qwen3-Embedding-0.6B** | 0.6B | 支援 (`/v1/embeddings`) | — | — | 快取池化 Single-Prefill 執行器 |
| **bge-m3** | 0.6B XLM-R | 支援 (`/v1/embeddings`) | Encoder | — | CLS 池化、Q4_K_M 載入器、hipBLASLt 調優 |
| **bge-reranker-v2-m3** | Cross-Encoder | 支援 (`/v1/reranking`) | ScoreBatch | — | BERT 分派、多序列 Padding 批次加速 |
| **Qwen3-Reranker** | Cross-Encoder | 支援 (`/v1/reranking`) | Scoring | — | Qwen 分派、共用 Rerank 評分頭 |

### 生產級多模型 Router

執行 `serve llm --multi-model` 可將伺服器轉為模型註冊中心：啟動模型即為預設模型，後續請求根據 JSON 中的 `model` 欄位自動分發；亦可透過 Admin API 在線上動態增減模型，服務完全不斷線。

#### 核心保障與容錯機制
- **安全記憶體調度**：在呼叫 `/admin/load` 時自動評估可用顯存，空間不足時自動對冷門的非預設模型執行 **LRU 卸載**（可設定 `"allow_evict": false` 拒載）；預設模型永不被卸載。
- **無損排空釋放**：模型卸載時，顯存會等待該模型正在進行的請求全數處理完成後才平滑釋放，絕不中斷線上對話。
- **預設自動回退**：若請求中 `model` 欄位為空，自動路由至預設模型，不報錯。

#### 端點一覽

| 端點 | 方法 | 功能說明 |
| --- | --- | --- |
| `/v1/chat/completions` | POST | 聊天對話介面（支援文字、影像、Function Tools、結構化輸出、SSE 串流） |
| `/v1/embeddings` | POST | 向量嵌入介面（支援 bge-m3、Qwen3-Embedding） |
| `/v1/reranking` | POST | 文本重排序介面（支援 bge-reranker-v2-m3、Qwen3-Reranker） |
| `/admin/models` | GET | 查詢已載入的模型清單、上下文與存取狀態 |
| `/admin/load` | POST | 動態載入並註冊新模型（支援顯存 LRU 置換保護） |
| `/admin/eject` | POST | 線上卸載模型，待任務清空後平滑釋放顯存 |

---

## 快速開始

### 1. 啟動主推論服務（以 Qwen3.8 Flash-Next 為例）

```sh
./build/release/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mmproj models/qwen3.8-flash-next/mmproj-BF16.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/agentionai-MTP-Q8_0.gguf \
  --speculative mtp --think off \
  --served-model-name gufo-qwen3.8-flash-next \
  --sessions 4 -c 262144 --kv-quant q8
```

### 2. 多模型模式（啟用 Router 與 Admin API）

```sh
./build/release/gufo serve llm --multi-model \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --served-model-name flash-next -c 262144 --kv-quant q8 --sessions 4
```

### 3. API 呼叫示範

```sh
# 對話生成
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "flash-next",
    "messages": [{"role": "user", "content": "你好，請自我介紹。"}]
  }'

# 動態載入新模型 (Gemma 4)
curl http://localhost:8080/admin/load \
  -H "Content-Type: application/json" \
  -d '{
    "id": "gemma4",
    "path": "models/gemma-4-26b/gemma-4-26B-A4B-it-Q4_0.gguf",
    "max_context": 262144, "session_count": 2, "kv_cache_quant": "q8"
  }'

# 查詢目前已載入的模型
curl http://localhost:8080/admin/models

# 卸載指定模型
curl http://localhost:8080/admin/eject \
  -H "Content-Type: application/json" \
  -d '{"id": "gemma4"}'
```

相容於 OpenAI SDK、Open WebUI、VS Code 等前端工具，Base URL 設定為 `http://localhost:8080/v1` 即可。完整介面協議詳見 [docs/SERVER.md](docs/SERVER.md)。

---

## 效能基準（gfx1151 Strix Halo）

測試條件：冷啟動 KV（每請求獨立 Prompt）、關閉思考模式、F16 KV、單串流 `-j 1`、上下文 262,144。27B 採用 DFlash2 投機解碼，Flash-Next 採用 MTP 投機解碼。

### Prefill 吞吐量（tok/s，越高越好）

| Token 規模 | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---:|---:|---:|---:|---:|
| **2K** | 605 | 455 | **1739** | 1301 |
| **8K** | 615 | 564 | **1701** | 1549 |
| **32K** | 567 | 551 | 1598 | **1618** |
| **64K** | 508 | 496 | **1695** | 1678 |
| **128K** | 409 | 407 | **1504** | 1507 |
| **250K** | — | — | **1438** | 1415 |

### Decode 速度（tok/s，開啟投機解碼）

| 測試場景 | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---|---:|---:|---:|---:|
| 散文生成 (temp 0.4) | 23.5 | 19.7 | 43.3 | **43.6** |
| 費氏數列生成 (temp 0.4) | 67.9 | 56.2 | **86.3** | 84.7 |
| JSON 結構生成 (temp 0) | **55.1** | 46.5 | 51.5 | 51.7 |
| 程式碼貪婪生成 /128 (temp 0) | 61.9 | 55.3 | 70.5 | **73.2** |
| 重複文本生成 /256 (temp 0) | 70.6 | 60.8 | 74.8 | **75.3** |

### Agentic 綜合評測（agent_bench_v2 沙箱工具組）

| 評測項目 | 27B Q4_K_XL | 27B FP4-FAST | Flash-Next ROCmFP4 | Flash-Next Q4_K_XL |
|---|---:|---:|---:|---:|
| 通過任務數 | **8/8** | 7/8 | 7/8 | 7/8 |
| 嚴重失誤次數 | **0** | **0** | **0** | **0** |
| 生成 TPS 範圍 | 9.1–31.1 | 9.3–21.5 | 16.7–56.6 | 16.8–55.1 |

### 各模型推薦配置

- **Qwen 3.8 27B**：推薦採用 **UD-Q4_K_XL**，解碼速度與草稿接受率皆達最佳狀態。
- **Qwen 3.8 Flash-Next**：推薦採用 **ROCmFP4**，解碼吞吐極高（散文 45 tok/s、費氏數列 87 tok/s），且僅需 ~94 GB 記憶體（對比 Q4_K_XL 節省約 17 GB）。

---

## 上游追蹤

MBufo 持續追隨並以標準 Git Merge 方式整合上游 `main` 分支，保持歷史清晰與乾淨。

| 項目 | 內容說明 |
| --- | --- |
| 上游倉庫 | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| 上游基準版本 | `v0.26.0`（持續追隨穩定提交） |
| 合併策略 | 透過 `--no-ff` 合併上游最新提交，MBufo 特有提交保持獨立清晰，永不竄改上游歷史 |
| 標籤命名規範 | MBufo 發布標籤均採用 `mbufo-vX.Y.Z` 前綴，與上游原生標籤完全隔離 |

---

## 從原始碼建置

目標環境為 Linux x86-64 上的 AMD Strix Halo（`gfx1151`）。編譯流程與上游標準完全一致：

```sh
cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
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

MBufo 作為 [gufo](https://github.com/gufo-org/gufo)（gufo-org 及社群貢獻者）的優化衍生版本，在此感謝開源社群奠定的深厚基礎。
