# MBufo

專為 AMD Strix Halo 打造的 [gufo](https://github.com/gufo-org/gufo) 分支，定位
為**公平性＋高效低銷的推論引擎**——排隊不插隊靠公平調度，花費少靠低精度計算。

> [English version](README.md)

上游 gufo 負責單路極限效能；MBufo 改問服務端的問題：一台 Strix Halo 節點同時
面對十六路請求時能跑多滿，每一路又能吃多長的上下文？三根支柱回答它——負載下
公平的排程（§1）、花得少的低精度計算（§2）、一個 port 服務多顆模型的 GGUF
覆蓋（§3）。底層引擎、核心與絕大部分程式碼都來自上游，在此致謝 gufo-org 及
所有貢獻者。

**目錄：** [三個數字](#mbufo-三個數字) ·
[§1 高併發](#1-高併發架構差異) ·
[§2 低精度計算](#2-低精度計算成本差異) ·
[§3 GGUF 相容](#3-gguf-相容覆蓋承諾) ·
[快速開始](#快速開始) ·
[效能基準](#效能基準mbufo-v0255gfx1151-strix-halo) ·
[上游追蹤](#上游追蹤) ·
[建置](#從原始碼建置) · [文件](#文件索引) ·
[限制](#相容性與限制)

## MBufo 三個數字

| # | 數字 | 含義 |
| --- | --- | --- |
| 1 | **短請求 TTFT 0.7 秒** | 被 25 萬 token prefill 卡住的短請求，原本要等 3.4 秒；公平排程後只要 0.7 秒（fairness-A 160 → 7.32、fairness-B 856 → 191，ITL 不變，貪婪輸出位元一致；v0.25.3）。 |
| 2 | **KV 記憶體省一半** | blocked-Q8 快取讓 262144 token 的 session 從 6.44 GB 降到 3.42 GB（K4V4 為 1.81 GB），代價是量出來的、不是假設的（v0.25.2）。 |
| 3 | **11 個 GGUF 家族** | 一個 HTTP port 全數服務——v0.28.0 新增 8 個可用家族，加上 Flash-Next、27B、Gemma 4（見[§3 模型總表](#支援的模型v0280)）。 |

## §1 高併發（架構差異）

併發不是一個數字。一台吃重的伺服器能不能用，是兩件事決定的，MBufo 兩件都當成
主軸：

- **ITL（token 間延遲）**——請求已經在串流之後，它前後兩個 token 之間的間隔。
  批次效率再高，只要會讓一條串流凍住一整個 prefill 步驟，那就不叫服務，那叫
  排隊加上額外手續。
- **公平性**——沒有任何 session、客戶端或 prompt 可以餓死別人。一段 25 萬 token
  的 prompt 插進對話中途，不該因此換到比十六路短對話更大的裝置佔比。

上游的排程器骨架已經是對的：prefill chunk 有上限（`--prefill-chunk`）、有待跑
decoder 時先服務 decode 再做 prefill、admission 有每客戶端上限。它缺的是「便宜
地同時兌現這三件事」所需的裝置時間：一個有上限的 chunk 仍然吃掉一整個 tick，而
那個 tick 裡**沒有任何** decoder 產出 token——因為 chunk 與 decode 是兩趟分別
掃過權重的計算。ITL 與公平性就是從這個 tick 漏掉的：每一輪，串流中的請求都要
付一次與它無關的 prefill 步驟。

MBufo 把兩者合併：chunk 與 decode 列共用同一個 `ForwardBatch`、同一個注意力
核心，於是 prefill 的那個 tick 也順便把該出的 token 全部產出。合併沒有繞過公平
政策；公平政策正是讓合併安全的前提。

### 統一 mixed-batch 介面

一次 `ForwardBatch` 把 decode 列與 prefill chunk 一起送進同一個注意力核心，
不再是兩趟權重計算。自 v0.27.0 起，decode/prefill 路由走單一共用 dispatcher——
統一 varlen mixed-batch 注意力介面，底下是共用 blocked-KV 編解碼器、WMMA
prefill reader 與 Split-K decode reader。27B 注意力層、Flash-Next decode 注意力、
Gemma 4 SWA 層全部經由它解析；F16 路徑完全未動，接線後 F16 輸出位元一致。

| 量測項 | 結果 |
| --- | --- |
| 串流請求的 token 間隔，5 列批次 | 兩趟計算 121 ms → 合併後 **57 ms**（**2.1 倍**）：間隔從兩趟權重計算縮成一趟 |
| 16 路 sustained 服務，3 輪 | **平均 15.3 秒 對 17.0 秒**（合併開 14/17/15，關 15/16/20），實測尾延遲改善 |
| 4 路混合 prompt A/B | **5.3 秒對 6.3 秒**，每輪都贏，輸出全對 |
| 16 路併發 | GPU 使用率 99%，輸出全數正確，零錯誤 |
| 批次封裝 | 32 session / 1024 列（上游為 8 / 8，過程中修掉三個硬上限：GDN、MoE、logits） |

先講一個但書：66 列時一個合併步驟要 316 ms，比兩趟分開的 247 ms 慢，因為在那種
規模下 batch-MoE 掉到 memory-bound 的 vec 路徑。上面的端到端數字在服務併發下仍然
勝出——每輪少掃一趟權重——但大批次在 FLOP 這一側的問題另案追蹤。

混合批次改變的是工作何時跑，不是模型算出什麼。它預設開啟；設定
`GUFO_MIXED_BATCH=0` 即可完全還原上游排程，tick-tock 也一併還原。設計鐵律、
驗證矩陣與逐階段量測見 [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md)。

### DecodeBatch＋Split-K decode

serve 層以準備好的批次步進 decode（`src/cli/serve/` 的 `StepDecodeBatch` /
`ExecutePreparedDecodeBatch` / `TextModelRunner::DecodeBatch`），而其中的注意力
計算走共用 Split-K decode reader（`blocked_splitk`，v0.26.1），不再走 scalar 路徑。
具體數字：Gemma 4 SWA 層在共用 Q8 下 decode 只比搬移前基準慢 6.4%（scalar reader
時期為 8.0%），27B 注意力層 decode 直接讀取為 −1.0%，F16 輸出與搬移前 binary 位元
一致。

### 公平性：排隊不插隊（SRTF＋飢餓 decode 優先搶佔＋near-done 閘門）

長 prefill 曾把短請求的 TTFT 擠爆（5 進 4 反飢餓場景）。三個機制修掉它，全程保持
貪婪輸出位元一致（sha 比對通過：24K prefill 單跑與併發跑出相同 sha）：

- **SRTF prefill 選取**——剩餘 ≤1 個 chunk 的 near-done prefill 排在長 prefill
  續片前面（同級按 FIFO）。
- **飢餓 decode 優先搶佔**——有請求在排隊等 session 槽位時，長 prefill chunk 讓路，
  先把常駐 decode 排空，好讓等待者一有常駐完成就被接納；上限 32 輪，prefill 不會
  被餓死。
- **near-done 閘門**——只有當某個常駐 decoder 距 `token_limit` ≤64 token（槽位即將
  空出）才觸發搶佔；若常駐全是長輸出 decoder，就不做無意義的 prefill 延遲。光是這個
  閘門就把 fairness-A 從 76.5 拉回 7.32。

| 量測（gfx1151，FP4＋f16＋MTP＋sessions 4，同 harness） | 結果 |
| --- | --- |
| fairness-A 退化比 | **160 → 7.32** |
| fairness-B 退化比 | **856 → 191** |
| 短請求 TTFT | **3.4 秒 → 0.7 秒** |
| ITL | 不變 |
| 貪婪輸出 | 位元一致（sha 通過） |

### 投機解碼（MTP / DFlash）

投機是各模型草稿後端掛在統一 draft registry 之下（v0.27.0）：

- **Flash-Next MTP**——併發寬度 2/3/4 的接受率 **48.8/38.9/41.7% →
  88.7/91.6/89.5%**（v0.25.1 修復：含投機 verify 的批次不再走與草稿路徑不一致的
  packed mixed WMMA 核心）。接受率不再隨 batch 衰減——上游 95 → 82 → 68，本構建
  穩在 ~90%——各寬度總吞吐全面超過上游（97/111/133 對 70/105/112）。（註：舊筆記的
  flash 92% 是寬度 3 峰值 91.6% 的約數，以上三數為準。）
- **27B DFlash2**——支援 blocked-Q8 KV（verify 寫進 blocked sidecar，continuation
  快取與 serve 磁碟身分位元級攜帶）；v0.24.1 verify 路由修復救回 **+248–340%**
  decode（short/prose/code/json 5.8/5.6/13.2/11.9 → 22.9/19.5/56.2/52.3 tok/s）。
- **Glimmer DFlash**——端到端投機草稿 **~89%**（E4 證偽為非 bug；v0.28.0 已知限制，
  照實陳述）。
- **Gemma 4 MTP**——async-GPU 草稿重寫＋單 forward verify 之後，decode 達
  **74–86 tok/s**（長 KV **+50%**：p2k 56.9 → 86.2，p8k 50.2 → 75.7，`d=3`）。最佳
  草稿深度 `d=3`（淨吞吐最高，**每輪接受 1.74–1.78**）；`d=4` 原始接受率最高
  （1.98）但 verify 成本稍高。（註：舊筆記 ~1.6/round 依 CHANGELOG v0.26.0 修正為
  1.74–1.78。）

### prefix RAM 快取＋多 sessions 生產配置

重複 prefix 走上游 RAM 快取（shared-prefix 邊界、RAM 上限、coherent prefix 索引）
不再重算，continuation 快照讓長對話可恢復。生產形狀是 session 數 × 上下文；自
v0.25.4 `-j` 修復後 session 數是誠實的（`-j 8` 真跑 8 路：TTFT 樓梯消失，8 路
decode 總量 124.5 → 148.2／121.1 → 144.4），常用預設已寫進 CLI（llm 路徑
`--sessions` 1 → 4、`--kv-quant` f16 → q8；v0.25.2）。

| 目標 | 設定 |
| --- | --- |
| 16 路、高併發 | `--sessions 16 -c 65536 --kv-quant q8`（約 103 GB，餘 20 GB） |
| 8 路、長對話 | `--sessions 8 -c 131072 --kv-quant q8`（約 101 GB，餘 21 GB） |
| 單路、最大上下文 | `--sessions 1 -c 262144 --kv-quant q8` |
| 4 路、最大上下文 | `--sessions 4 -c 262144 --kv-quant q8`（約 101 GB，餘 21 GB——總位元組與 8×131072 相同） |
| 單路、最快最精準 prefill | `--sessions 1 -c 262144 --kv-quant f16`（全精度 KV；每位置最重，但單流裝得下） |
| 還原上游排程 | 環境變數 `GUFO_MIXED_BATCH=0` |

載入器限制的是「session 數 × 上下文」，不是單一上下文。在 128 GB Strix Halo
上以 blocked Q8 快取＋4-bit value 列實測，8 路甚至可到 192K（`-c 196608`，
107 GB，只剩 16 GB 餘裕）；16 路拉到 96K 則會被
`text runner request-state claim exceeds state capacity` 拒載。量測條件為
UD-Q4_K_XL ＋ MTP Q8_0 sidecar ＋ BF16 mmproj，舊的 32K 上限已不成立。

### 排程器保證什麼

| 保證 | 機制 |
| --- | --- |
| 待跑 decoder 絕不排在 prefill 之後 | 只有在沒有 decoder 待跑時（或正在湊第一批時）才進合併步驟，所以 prefill 只在空隙裡跑，永遠不會插隊到串流請求前面。 |
| chunk 不能把 decoder 擠出輪替 | 每次合併或 prefill 步驟後，`MarkAllDecodersDue` 會把所有常駐 decoder 重新標記為待跑，下一步就全部服務。 |
| 任何 prompt 都無法獨佔裝置 | chunk 是有上限的切片（`--prefill-chunk` 預設 512 token），25 萬 token 的 prompt 每輪只餵一片。 |
| 同等的 session 輪流 | decode 佇列會輪替（`PopDecoder` 把服務過的請求移到隊尾），不是固定隊頭順序，所以沒有哪個 session 能一直贏得前排。 |
| 單一客戶端無法塞滿佇列 | `--max-pending-per-client` 限制單一客戶端的待處理請求（回 `client_quota`），出隊也跨客戶端輪流，一個客戶端爆量不會餓死別人。 |
| 某一層拒絕也不損失 | 每一層都可獨立降級；投機列被拒就退回原本路徑：正確，只是沒有加速。 |

### 三引擎負載對比（開 MTP）

沿用先前的併發服務對比（同 harness、皆暖機；併發 1／2／3／4 的「總／每 session」
decode TPS——A Halogen · B gufo-upstream · C MBufo；數字與英文版一致，舊中文版此表
為過時版本已同步）：

| 引擎 | 總 TPS | 每 session TPS | fairness A / B |
| --- | --- | --- | --- |
| A Halogen | 85 / 82 / 75 / 88 | 85 / 41 / 25 / 22 | 322 / 1140 |
| B upstream | 57 / 64 / 108 / 127 | 57 / 32 / 37 / 33 | 15 / 399 |
| C MBufo | 66 / 83 / 110 / 129 | 66 / 42 / 38 / 33 | **19 / 187** |

A 單流峰值最高，但只在單獨執行時投機——一旦併發就關掉投機頭，每 session TPS 掉到
~22。負載下 **C 擴展最佳**（線性到 129 TPS、每 session 最平），自 v0.25.3 起公平性
也最好。完整方法與其餘面向（prefill、fairness、ITL）見 `高併發測速方法.md`。

### 多模型 router

一個 port、多顆模型。`serve llm --multi-model` 把伺服器變成模型 registry：啟動
模型成為**預設**，每個請求依 `model` 欄位路由到對應模型。模型可在執行期增減
（admin API）——不重啟、不斷線。（[§3 模型總表](#支援的模型v0280)列出一個 port
能服務的全部對象。）

啟動時指定預設模型並開啟 router：

```sh
./build/release/gufo serve llm --multi-model \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --served-model-name flash-next -c 262144 --kv-quant q8 --sessions 4
```

依名稱路由（不帶 `model` 就打預設模型）：

```sh
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "flash-next", "messages": [{"role": "user", "content": "hi"}]}'
```

執行期載入另一顆模型（例如 Gemma 4），再路由過去：

```sh
curl http://localhost:8080/admin/load -H "Content-Type: application/json" \
  -d '{"id": "gemma4",
       "path": "models/gemma-4-26b/gemma-4-26B-A4B-it-Q4_0.gguf",
       "max_context": 262144, "session_count": 2, "kv_cache_quant": "q8"}'

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "gemma4", "messages": [{"role": "user", "content": "hi"}]}'
```

列出已載入的模型，用完就卸載：

```sh
curl http://localhost:8080/admin/models
curl http://localhost:8080/admin/eject -H "Content-Type: application/json" \
  -d '{"id": "gemma4"}'
```

**記憶體安全。** 每次 `/admin/load` 先估計模型佔用、比對免費裝置記憶體。塞不下
時，router 會 **LRU 卸載**最久未用的非預設模型騰出空間（在 `evicted` 回報踢了
哪些）；帶 `"allow_evict": false` 則改回 `409 insufficient_memory` 失敗。預設
模型永不被踢，registry 不會被清空。卸載模型時，其 GPU 記憶體要等 in-flight 請求
跑完才釋放，不會打到一半斷掉。

| Admin 端點 | 方法 | Body | 效果 |
| --- | --- | --- | --- |
| `/admin/models` | GET | — | 列出已載入模型、context、最後使用時間 |
| `/admin/load` | POST | `{id, path, max_context?, session_count?, kv_cache_quant?, vision_model_path?, enable_yarn?, allow_evict?}` | 載入並註冊模型（記憶體吃緊時 LRU 卸載） |
| `/admin/eject` | POST | `{id}` | 卸載模型；記憶體等 in-flight 請求排空後才釋放 |

router 是選擇性的：不帶 `--multi-model` 時，伺服器就是純粹的單模型版，位元級
不變。

### v0.28.0 router 新增

| 功能 | 行為 |
| --- | --- |
| 多模型託管 | L0/L1 EngineRegistry factory，27B、flash-next、gemma4、glimmer、llama 五個 adapter；執行期 `/admin/load`＋`/admin/eject`，啟動模型為永不卸載的預設 |
| 自動分派 | 每個請求依 `model` 欄位路由；`openai_chat` 空 `model` 改走預設，不再 `400 missing_model` |
| fallback 語義 | 未知模型 id 被拒絕且不影響已載入模型；載入期 OOM 就 LRU 卸載冷門非預設模型（`allow_evict:false` → `409 insufficient_memory`）；卸載等 in-flight 請求排空才釋放 GPU 記憶體 |
| Embeddings | `POST /v1/embeddings` 經快取 single-prefill Embed（bge-m3、Qwen3-Embedding） |
| Reranking | `POST /v1/reranking`，BERT（bge-reranker-v2-m3）＋Qwen（Qwen3-Reranker）分派，padding 批次 |
| 工具呼叫 | ATEM／Gemma／Qwen 跨後端工具呼叫映射 |
| Qwen KV 旗標修正 | 顯式 `--kv-quant` 在 Qwen 路徑覆寫 `QWEN_KV_QUANT` |

## §2 低精度計算（成本差異）

花費用數字論證。KV 精度、權重精度、注意力變體各自對基準量過；這裡沒有「越窄
越免費」這種事。一句話建議：`--kv-quant q8` 配 ROCmFP4 或 Q4_K 權重——再窄都是
明確的交易，代價寫在下面。

### KV 快取：blocked Q8／Q4

上游的 blocked Q8 KV 路徑被泛化成「格式加編解碼器」抽象，且 key 與 value 兩軸
都可再壓成 4-bit 的 18-byte 打包區塊（每 32 個元素 18 位元組）。兩軸精度獨立
選擇，因此 K8V8、K4V8、K8V4、K4V4 四種組合都可達成。自 v0.26.1 起編解碼器收進
單一共用核心（`src/core/compute/attention/`：`gufo_core_blocked_kv`）——blocked
寫入器、scalar 直讀器（`blocked_kv`）、WMMA prefill reader（`blocked_wmma`）、
Split-K decode reader（`blocked_splitk`）。量化 KV 是共用能力；F16 預設路徑各地
未動。

| 模式 | 狀態 | 說明 |
| --- | --- | --- |
| `f16` | 預設 | 基準數值。KV 快取不做量化。 |
| `q8` | 支援 | Blocked 8-bit，每 32 個元素配一個 f16 尺度（34 位元組）。 |
| `q8` + `GUFO_KV_V4=1` | 手動開啟，K8V4 | 4-bit value 打包成 18 位元組，key 維持 8-bit。 |
| `q8` + `GUFO_KV_Q4=1` | 手動開啟，K4V8 | 4-bit key 打包成 18 位元組，value 維持 8-bit。 |
| `q8` + 兩者同時 | 手動開啟，K4V4 | key 與 value 都是 4-bit。 |

**建議使用 `q8`。** blocked 模式在編解碼層級是位元精確的；`f16` 仍是預設與基準。
4-bit 路徑是給「KV 記憶體成為瓶頸」的機器手動開啟的環境變數鉤子，不是
`--kv-quant` 的格式選項，也不會改變任何預設值——未設定時引擎與不含這些路徑的版本
位元相同（以 logit dump 雜湊比對驗證）。完整說明見 [docs/KV-V4.md](docs/KV-V4.md)。

#### 佔用

每個 KV 位置的位元組數，以及 `-c 262144`、Qwen3.8 Flash-Next 12 個注意力層下
單一 session 的總量：

| 模式 | 每位置位元組 | 262144 單一 session |
| --- | --- | --- |
| `f16` | 2048 | 6.44 GB |
| `q8` | 1088 | 3.42 GB |
| K8V4 / K4V8 | 832 | 2.62 GB |
| K4V4 | 576 | 1.81 GB |

#### 代價（量出來的）

以 `q8` 為基準的正確性：三個互不相關的 400-token 上下文、各 98 步
matched-token decode、對三個上下文取平均：

| 模式 | decode KL | decode 最大 KL | 貪婪 argmax 一致率 |
| --- | --- | --- | --- |
| K4V8 | 0.0214 | 0.258 | 90.5% |
| K8V4 | 0.0209 | 0.120 | 93.2% |
| K4V4 | 0.0335 | 0.411 | 89.8% |

啟用比 `q8` 更窄的模式前，請先讀懂這張表：

- **K4V8 與 K8V4 佔用的記憶體完全相同。** key 是 512 個元素，4-bit key 配
  8-bit value 與 8-bit key 配 4-bit value 都落在 832 位元組。把精度從一軸搬到
  另一軸換不到任何記憶體，卻要多付約 2.7 個百分點的貪婪一致率，所以 K4V8 沒有
  用途。
- **只有 K4V4 真的縮小佔用**，代價是 decode KL 多 60%、尾巴大 3.4 倍，大約每
  十個貪婪決策就有一個改變。這是拿正確性換記憶體的明確交易，不是免費。
- **不要用短 prefill 探針決定 KV 精度。** 在 33-token 的 prompt 上，K4V4 量到
  的平均 KL 與 K8V4 相同（0.0274）。原因是兩軸的逐位置誤差幾乎不相關，同時
  量化時會部分互相抵銷，小樣本會把傷害藏起來。只有多上下文、decode 階段的數字
  能把這些模式排出正確順序。
- 4-bit 區塊就是 GGML `q4_0` 的佈局（16 個打包 nibble 加一個 f16 尺度），K/V
  分軸選精度在上游也已是 `--cache-type-k` / `--cache-type-v` 的既有功能。這顆
  引擎特有的地方在於：兩軸都貫通 blocked kernel、混合批次路徑與 snapshot 身分，
  而且代價是量出來的、不是假設的。

`q8` 對照 CPU f16 基準：端到端偏差維持在 blocked-KV 設計範圍內，MTP draft 命中
率維持在 57% 至 81% 區間，算術題與長文生成保持通順。

必須直說的一點：這些 blocked 模式在貪婪解碼下與 `f16` 的分歧，都比單純用數值
誤差能解釋的程度來得大。這是既有 blocked-KV 設計的性質，不是 4-bit 區塊造成
的，已另案追蹤；而在它之上再疊 4-bit 區塊，只會把差距放大，不會優雅地疊加。

### 權重精度：W8A8／FP4／MXFP4／Q4_K

- **W8A8**——共用 GEMM 核心（`src/core/compute/gemm`，v0.27.0）的 host-side
  16-split varlen GEMM；Flash-Next prefill W8A8 已改走共用核心且輸出位元一致。
  Glimmer 跑 W8A8／Q8_0 路徑。
- **ROCmFP4／ROCmFP4-FAST**——4.25 bpw 單尺度 FAST down 專家經 fp16 WMMA
  routed-expert GEMM 解碼（perm codebook LUT 餵進既有 Q8_0 反量化），MoE prefill
  因此跑在矩陣核心上、不掉進通用 `mul_mat_q`。gfx1151 實測 FAST 模型 prefill 打平
  或勝過 UD-Q4_K_XL 基線（1.9K／5K token 為 1564／1368 對 1556／1324 tok/s），
  decode 速度與 64k 工具梯子正確性不變。大 dense 投影在 decode 期保留 Q4_1 拷貝
  （20 B/block，比 Q8_0 少 41% 位元組）：decode **+17.4%**（40.63 → 47.68 tok/s，
  v0.20.5）。
- **MXFP4**——GPT-OSS 20B／120B 跑原生 MXFP4 routed GEMM，decode 端 MoE／QKV／`wo`／
  router 全融合；120B 以零拷貝串流上傳＋分塊 H2D 鏡像載入。
- **Q4_K**——UD-Q4_K_XL 是 27B 推薦 build（decode 與草稿接受率勝 single-scale FP4），
  也是 Flash-Next 參照基線；bge-m3 帶 Q4_K_M loader；Gemma 4 跑 Q4_0 權重。

### 注意力變體：WMMA／flash／Split-K

- **WMMA prefill reader**（`blocked_wmma`）——單 session causal prefill 直接讀
  blocked int8 KV 快取，跳過 F16 staging 反量化，覆蓋快取能持有的全部佈局
  （K34/K18 × V34/V18，dense 與 sparse）。14K 單流 prefill 比 scalar 路徑
  **快 30%**（1085 → 1408 tok/s）。
- **Split-K decode reader**（`blocked_splitk`）——v0.26.1 起共用；量測數字見
  [DecodeBatch＋Split-K](#decodebatchsplit-k-decode)。
- **flash-decoding**——Gemma 4 以 Split-K flash-decoding 注意力配融合 decode
  GEMV、融合 norm／residual／epilogue kernel 與裝置端貪婪 argmax（p8k prefill
  1321 → 1446 tok/s，輸出位元一致），WMMA 注意力另有 SWA 滑窗 KV 跳過。
- **scalar 直讀器**（`blocked_kv`）——每種佈局都保留的退路。
- **F16 路徑未動**——以上全是量化路徑；預設 F16 decode／prefill 從不改道。

### 逐模型精度表

| 模型 | 權重精度 | KV | F16 留守 |
| --- | --- | --- | --- |
| Qwen3.8 Flash-Next（主打） | ROCmFP4-FAST（4.25 bpw）或 UD-Q4_K_XL；decode Q4_1 拷貝（+17.4%） | blocked-Q8（`--kv-quant q8`）；MTP Q8_0 sidecar | KV 預設 `f16`；全部 F16 路徑未動 |
| Qwen3.8 27B | UD-Q4_K_XL（建議）或 ROCmFP4-FAST | blocked-Q8（含 DFlash2；`QWEN_KV_QUANT`、`--kv-quant` 覆寫） | 同上 |
| Gemma 4 26B-A4B | Q4_0；Q4→Q8 dense 一次性預轉（每 chunk 省 210 次反量化啟動） | SWA 層共用 Q8（prefill −2.8%、decode −6.4%） | **full-512 層留守 F16**；MTP 草稿取自每 session F16 鏡像 |
| Qwen3.6-35B-A3B | Q6_K→Q8_0 載入；split-K router GEMV | 僅引擎／bench 路徑 | 尚無 serve 後端 |
| GPT-OSS 20B／120B | 原生 MXFP4 routed GEMM | text 後端服務路徑 | 無投機草稿 |
| Muse Glimmer 30B | W8A8／Q8_0；MQB-first 注意力 | serve 後端 `--kv-quant` 旗標 | DFlash 草稿，無 F16 例外 |
| Llama-3.1-8B | dense GQA（見模型卡） | text 後端服務路徑 | 無草稿 |
| Qwen3-Embedding-0.6B | dense Qwen 注意力 | prefill-only single-prefill Embed | 無 ladder／草稿（評分類 N/A） |
| bge-m3 | Q4_K_M loader；逐形狀 autotune | prefill-only `/v1/embeddings` | **F16 hipBLASLt 融合 QKV 路徑** |
| bge-reranker-v2-m3 | 共用 rerank 評分頭 | padding ScoreBatch（超目標 `+15%`） | 純評分，無草稿 |
| Qwen3-Reranker | yes/no head，Qwen 分派 | 純評分 | 純評分，無草稿 |

### 核心層加速

以下改動同時提升單路與批次吞吐，且都不改變模型語意。（§2 數字的實作細節，
原樣保留。）

- **ROCmFP4-FAST 專家走 fp16 WMMA 核心。** 4.25 bpw 的 `Q4_0_ROCMFP4_FAST` down
  專家現在與 Q4_K/Q8_0 專家共用同一條 fp16 WMMA routed-expert GEMM（perm codebook
  LUT 餵進既有 Q8_0 反量化），FAST 模型的 MoE prefill 因此跑在矩陣核心上、不再掉
  進通用 `mul_mat_q`。gfx1151 實測 FAST 模型 prefill 打平或勝過 UD-Q4_K_XL 基線
  （1.9K／5K token 為 1564／1368 對 1556／1324 tok/s），decode 速度與 64k 工具
  梯子正確性不變。
- **int8 WMMA 直讀 prefill。** 單 session 的 causal prefill
  （`WmmaCausalAttention`）現在直接讀 blocked int8 KV 快取，跳過 F16 staging
  反量化。它覆蓋快取能持有的任何 KV 區塊佈局（K34/K18 x V34/V18，dense 與
  sparse），且由與快取寫入端相同的格式/環境變數選取，因此任何佈局都不會被誤
  讀。在 14K 單流 prefill 上比 scalar 路徑**快 30%**（1085 → 1408 tok/s）。
- **每個 prefill pass 都走 wide 路徑。** wide prefill 路徑（融合投影、成對
  routed expert tile、WMMA attention）適用於所有寬於一個 vector batch 的
  prefill pass，不再只限 1024+ token。

### 對比 llama.cpp（v0.28.0，gfx1151）

同權重、release build、greedy；prefill＝冷 KV 單流 tok/s，decode＝開投機 tok/s
（另註除外）。正數＝MBufo 較快。六格數字皆為 CHANGELOG 量測到的 v0.28.0 開發期
數字原文重述。

| 負載 | Prefill 差值 | Decode 差值 | 條件／備註 |
| --- | --- | --- | --- |
| Flash-Next | **+52%** | 持平或更好 | 冷 KV、greedy、開 MTP |
| Qwen3.6-35B-A3B | **+52.5%** | 持平或更好 | split-K router GEMV decode；greedy |
| bge-m3（長文件） | **+35~50%** | n/a（純 prefill） | 長文件形狀、hipBLASLt autotune、快取執行器 |
| Qwen3-Embedding-0.6B（長文件） | **+41~50%** | n/a（純 prefill） | 長文件形狀、single-prefill Embed |
| Reranker（批次評分） | n/a | **超目標 +15%** | padding 多序列 ScoreBatch |
| GPT-OSS 120B | **+62%** | **-17%** | 結構性差距，照實陳述：此路徑 decode 仍卡記憶體頻寬 |

此次版本推進未跑模型服務效能測試；上表為量測到的 v0.28.0 開發期數字重述，
此處未重測。

## §3 GGUF 相容（覆蓋承諾）

### 支援的模型（v0.28.0）

一個 port 經由[多模型 router](#多模型-router)服務以下全部。
`Serve`＝有 HTTP 後端；`Ladder`＝有 registry／pilot 或評分測試；`MTP/draft`＝有
投機草稿後端。共 11 列：v0.28.0 新增 8 個可用家族，加上既有的三個
（Flash-Next、27B、Gemma 4）。

| 模型 | 參數 | Serve | 梯子 | MTP/draft | 備註 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8 Flash-Next（主打） | 30B-A3B MoE | 有 | pilot | MTP | ROCmFP4-FAST preset、blocked-Q8 KV |
| Qwen3.8 27B | 27B | 有（`QWEN_KV_QUANT`、`--kv-quant` 覆寫） | pilot | DFlash2（含 q8） | dense＋MoE 路徑 |
| Gemma 4 | 26B-A4B | 有（HTTP 後端、原生工具） | ladder 15/15 @24k | MTP sidecar（q8 經 F16 鏡像） | full-512 層留守 F16 |
| Qwen3.6-35B-A3B | 35B-A3B MoE | 僅引擎／bench | — | 無 | Q6_K→Q8_0 載入、split-K decode MoE；64K 以上 ctx 請壓在 60K token 內（MODE2 重置風險） |
| GPT-OSS 20B／120B | 20B／117B MoE（MXFP4） | 有（text 後端） | pilot | 無 | 120B 零拷貝串流上傳＋分塊 H2D 鏡像 |
| Muse Glimmer 30B | 30B | 有（ATEM 工具、kv-quant） | prompt 路徑 | DFlash 端到端（~89%） | DFlash 89% 是量到的，E4 證偽為非 bug |
| Llama-3.1-8B | 8B dense GQA | 有（text 後端） | pilot adapter | 無 | — |
| Qwen3-Embedding-0.6B | 0.6B | 有（`/v1/embeddings`） | n/a（純 prefill） | 無 | pooled single-prefill Embed |
| bge-m3 | 0.6B XLM-R | 有（`/v1/embeddings`） | encoder 測試 | 無 | CLS pooling、Q4_K_M loader、hipBLASLt autotune |
| bge-reranker-v2-m3 | cross-encoder | 有（`/v1/reranking`，BERT 分派） | ScoreBatch 測試 | 無 | padding 多序列批次 |
| Qwen3-Reranker | yes/no head | 有（`/v1/reranking`，Qwen 分派） | 評分測試 | 無 | 共用 rerank 評分頭 |

DSpark 維持 legacy。模型權重不隨附；各指南在 [docs/models](docs/models/README.md)
下列出所需檔案。

### 端點＋工具映射

| 端點 | 方法 | 服務對象 | 備註 |
| --- | --- | --- | --- |
| `/v1/chat/completions` | POST | 文字對話（＋圖像、工具、結構化輸出、串流） | 空 `model` 改走預設，不再 `400 missing_model` |
| `/v1/embeddings` | POST | bge-m3、Qwen3-Embedding | 快取 single-prefill Embed 執行器 |
| `/v1/reranking` | POST | bge-reranker-v2-m3（BERT 分派）、Qwen3-Reranker（Qwen 分派） | padding 多序列 ScoreBatch |
| `/admin/models`、`/admin/load`、`/admin/eject` | GET／POST | router registry | 見[記憶體安全](#多模型-router) |

工具呼叫按後端家族映射——ATEM（Glimmer）、Gemma 原生（`<|tool|>` 模板、ladder
15/15 @24k）、Qwen——Qwen 路徑顯式 `--kv-quant` 現覆寫 `QWEN_KV_QUANT`。Open WebUI、
VS Code、OpenAI SDK 與 Responses-API 客戶端把 base URL 指到
`http://localhost:8080/v1` 即可；詳見 [API 契約](docs/SERVER.md)。

### fallback 語義

- 未知模型 id 被拒絕，不影響已載入模型。
- 載入期 OOM 就 LRU 卸載冷門非預設模型（`allow_evict:false` →
  `409 insufficient_memory`）；預設永不卸載。
- 卸載等 in-flight 請求排空才釋放 GPU 記憶體。
- 不帶 `--multi-model` 時，伺服器就是位元級一致的單模型版。
- 任何服務不了整批的注意力／GEMM 層都退回 legacy 路徑——正確，只是沒有加速。
- 快照以 KV 格式區分版本；某格式寫出的快照不會被另一格式重用。

### 新模型接線（五步）

本樹尚無獨立的新模型手冊（`GGUF通用化.md` 不存在，2026-11-09 已驗證）；在手冊
落地前，以下列清單＋[docs/models](docs/models/README.md) 下各模型指南為準。
每步附已走過的先例：

1. **EngineRegistry adapter**——在 L0／L1 factory 註冊家族，按請求 `model` 分派
   （先例：27B、flash-next、gemma4、glimmer、llama）。
2. **serve 後端**——text 後端，或 embeddings single-prefill Embed，或 reranking
   BERT／Qwen 分派；把 `--kv-quant` 旗標帶進去。
3. **ladder pilot 或評分測試**——registry pilot（27B、flash-next、gpt-oss、llama）、
   prompt 路徑（glimmer）、encoder／評分測試（bge 家族）；純 prefill 模型標 ladder
   N/A。
4. **草稿後端（或明確記無）**——MTP／DFlash2 掛 draft registry（gemma4 草稿取自每
   session F16 鏡像以覆蓋 4-bit 格）；記 `無` 要寫理由（純 prefill、尚無草稿）。
5. **serve 註冊＋文件**——`--served-model-name`、free-memory 閘門估計、
   `docs/models/<家族>/` 指南（按 [docs/models](docs/models/README.md) 結構：
   README／BENCHMARKS／QUALITY／EXPERIMENTS），有已知限制就加 CHANGELOG
   Known-limits 一行。

## 快速開始

以建議的快取設定啟動 Qwen3.8 Flash-Next：

```sh
./build/release/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mmproj models/qwen3.8-flash-next/mmproj-BF16.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/agentionai-MTP-Q8_0.gguf \
  --speculative mtp --think off \
  --served-model-name gufo-qwen3.8-flash-next \
  --sessions 2 -c 262144 --kv-quant q8
```

透過 OpenAI 相容 API 問它問題：

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gufo-qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "說點什麼"}]
  }'
```

Open WebUI、VS Code、OpenAI SDK 與 Responses-API 的客戶端，請把 base URL 指向
`http://localhost:8080/v1`。Chat Completions 與 Responses 支援文字、圖片、
function tools、結構化輸出與串流；詳見[API 契約](docs/SERVER.md)。

## 效能基準（MBufo v0.25.5，gfx1151 Strix Halo）

冷 KV（每請求唯一 prompt，`cached_tokens≈0`）、think off、**F16 KV**、`-j 1`、
ctx 262144。四組設定：27B（UD-Q4_K_XL 對 ROCmFP4-FAST）＋ Flash-Next（ROCmFP4-FAST
對 UD-Q4_K_XL）。投機：27B＝DFlash2、Flash-Next＝MTP。（註：舊中文版此節為 v0.24.1
數字，已同步為與英文版一致的 v0.25.5。）

這些是單流數字，用來隔離模型與建置版本的選擇；負載下改變的是同時能跑幾條，那是
[§1 高併發](#1-高併發架構差異)的故事。

### Prefill（tok/s，越高越好）

| tokens | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---:|---:|---:|---:|---:|
| 2K | 605 | 455 | **1739** | 1301 |
| 8K | 615 | 564 | **1701** | 1549 |
| 32K | 567 | 551 | 1598 | **1618** |
| 64K | 508 | 496 | **1695** | 1678 |
| 128K | 409 | 407 | **1504** | 1507 |
| 250K | — | — | **1438** | 1415 |

> 27B ROCmFP4-**FAST** 的 prefill ≈ 非 FAST 的 STRIX 版——single-scale 只換 codebook，
> 不換 fp16-WMMA prefill 引擎（455/564/551/407 對 483/523/520/383）。FAST 真正的速度
> 來自 int8-MMQ 消費路徑，而那條路徑只餵給 Flash-Next 的 MoE experts，dense 27B 沒接。

### Decode（tok/s，投機開）

標準：5 個冷 KV 情境（標明 temp）＋ 8 任務 ReAct 代理套件。

| 情境 | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---|---:|---:|---:|---:|
| 散文 (temp 0.4) | 23.5 | 19.7 | 43.3 | **43.6** |
| 費波那契 (temp 0.4) | 67.9 | 56.2 | **86.3** | 84.7 |
| JSON 約束 (temp 0) | **55.1** | 46.5 | 51.5 | 51.7 |
| 貪心程式碼 /128 (temp 0) | 61.9 | 55.3 | 70.5 | **73.2** |
| 重複文本 /256 (temp 0) | 70.6 | 60.8 | 74.8 | **75.3** |

### 代理任務（agent_bench_v2，t1–t8，沙箱工具）

| | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---|---:|---:|---:|---:|
| 通過 | 8/8 | 7/8 | 7/8 | 7/8 |
| 核爆（須為 0） | 0 | 0 | 0 | 0 |
| 生成 tps 區間 | 9.1–31.1 | 9.3–21.5 | 16.7–56.6 | 16.8–55.1 |

### 各模型建議（目前最佳）

- **Qwen 3.8 27B → 用 UD-Q4_K_XL。** FP4-FAST 在 FP4 家族內全勝（5 個 decode 情境
  ＋代理 8/8，勝 STRIX 的 7/8），但 single-scale FP4 的 decode 與草稿接受率仍輸
  Q4_K_XL，故 27B 以 Q4_K_XL 為最佳。
- **Qwen 3.8 Flash-Next → 用 ROCmFP4。** decode 最快（散文 45.0、費波那契 87.0）、
  代理 8/8，且 ~94 GB 比 Q4_K_XL 的 ~111 GB 更省，故 Flash-Next 以 ROCmFP4 為最佳。

> 27B ROCmFP4 decode 為 **v0.24.1 修復後**數值：修復前開 DFlash2 因 verify 走逐行
> GEMV 迴圈而崩速，修復後 FAST 追平並略勝非 FAST。

## 上游追蹤

MBufo 追隨上游 `main` 並以合併方式跟進，而不是分叉走自己的路。上游的 release
標籤維持不可變；MBufo 的 release 標籤一律加上 `mbufo-vX.Y.Z` 前綴，因此永遠
不會與上游的 `vX.Y.Z` 撞名。（註：舊中文版此節停在 v0.20.0，已同步為英文版現狀。）

| 項目 | 值 |
| --- | --- |
| 上游倉庫 | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| 上游基準 release | `v0.26.0`（cherry-pick 安全修復；完整合併已 revert——見下） |
| 已合併的上游 `main` | cherry-pick 到 `v0.26.0`（2026-10-28）：#732 中性 sampler 欄位、#736 可重複 `--served-model-name`。完整 0.21→0.26 合併**已 revert**：#758 chunked-DeltaNet prefill 讓 gfx1151 FN prefill 衰退 30-46%（上游 #850 後來移除了它）；audio／Windows 與此 gfx1151 Linux 分支無關。 |
| 合併策略 | 上游 `main` 以 `--no-ff` 合併；MBufo 專屬提交保持可辨識，上游歷史永不改寫 |

上游發布新版本時，「上游基準 release」這一列會在合併它的同一個提交裡更新，
因此本 README 永遠會標明這棵樹確切包含到上游的哪個狀態。

## 從原始碼建置

支援的目標平台為 Linux x86-64 上的 AMD Strix Halo（`gfx1151`）。建置方式即
上游的建置方式，MBufo 未增加任何建置系統需求。

```sh
cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
```

請安裝 C++20 編譯器、CMake 3.21+、Ninja、pkg-config、ROCm 開發函式庫
（HIP、hipBLAS、hipBLASLt、rocBLAS、hipCUB、rocPRIM、rocWMMA）以及一般系統
函式庫（ICU、libcurl、OpenSSL、libpng、libjpeg、libwebp、FFmpeg）。目前驗證
過的工具鏈為 GCC 15.3 搭配 ROCm 7.2.3。也提供 Nix 建置（`nix build`）。GPU
驅動必須允許你的使用者存取 `/dev/kfd` 與 `/dev/dri`；模型權重需另行取得。每個
模型指南（見 [docs/models](docs/models/README.md)）都列出所需檔案與基準設定。
完整的相依性清單、ROCr queue-memory 補丁與容器相關說明，請見
[上游建置文件](https://github.com/gufo-org/gufo#build-from-source)。

## 文件索引

| 文件 | 內容 |
| --- | --- |
| [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md) | 混合批次設計：鐵律、驗證矩陣、量測數據 |
| [docs/KV-CACHE.md](docs/KV-CACHE.md) | 對話與元件快取（上游） |
| [docs/KV-V4.md](docs/KV-V4.md) | 4-bit K/V 區塊：佈局、正確性表 |
| [docs/SERVER.md](docs/SERVER.md) | HTTP API 契約 |
| [docs/CLI.md](docs/CLI.md) | 命令列參照 |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | 建置、測試、preset |
| [docs/TESTING.md](docs/TESTING.md) | 託管 CI 測試組與本機品質檢查 |
| [docs/models/README.md](docs/models/README.md) | 各模型指南：README／BENCHMARKS／QUALITY／EXPERIMENTS |
| [CHANGELOG.md](CHANGELOG.md) | 上游發布歷史與 MBufo release |

## 相容性與限制

- 混合批次改變的是排程，不是模型語意。任何無法服務整批的層都會退回舊路徑。
- `--kv-quant q8` 是以「不再與基準完全一致」換取記憶體。
  當與基準的精確一致性比容量更重要時，請用 `f16`。
- 對話狀態以 KV 格式區分版本；用某一種格式寫出的快照不會被另一種格式重用。
- 上游的 release 標籤不可變，也不會被當成 MBufo 的標籤重複使用。
- 已知限制（v0.28.0，短句）：64K 以上 ctx 的 qwen35 請壓在 60K token 內（驅動
  MODE2 重置風險）；DSpark 維持 legacy；gemma4 full-512 層留守 F16；Glimmer
  DFlash 約 89%（E4 證偽為非 bug）。

## 授權與致謝

上游 gufo 採 MIT 授權；改作程式碼與相依套件各自保留其聲明，收錄於
[NOTICE](NOTICE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 與
`licenses/`。MBufo 保留這些聲明。模型權重不隨附，依各發布者條款使用。

MBufo 是 [gufo](https://github.com/gufo-org/gufo)（by gufo-org 及貢獻者）的
分支。低精度 KV 與性能調優的工作建立在上游的 Q8 KV cache、int8 WMMA、
persistence 與 history drafting 等改動之上。
