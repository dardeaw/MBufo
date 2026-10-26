# MBufo

專為 AMD Strix Halo 打造的 [gufo](https://github.com/gufo-org/gufo) 分支，聚焦
兩件事：**高併發混合批次排程**與**低位元 KV 快取**。

> [English version](README.md)

上游 gufo 負責單路極限效能；MBufo 改問服務端的問題：一台 Strix Halo 節點同時
面對十六路請求時能跑多滿，每一路又能吃多長的上下文？本分支只動這兩個子系統。
底層引擎、核心與絕大部分程式碼都來自上游，在此致謝 gufo-org 及所有貢獻者。

| 主軸 | 改了什麼 | 換到什麼 |
| --- | --- | --- |
| **混合批次排程** | decode 列與 prefill chunk 共用同一個 `ForwardBatch` 與同一個注意力核心；批次封裝放大到 32 session / 1024 列。 | 負載下 ITL 穩定、服務公平：串流中的 token 只等一步而不是兩步（5 列批次每 token 121 ms → 57 ms）；16 路併發 GPU 使用率 99%，16 路 15.3 秒對 17.0 秒。 |
| **低位元 KV 快取** | 上游 blocked Q8 路徑泛化成「格式＋編解碼器」抽象，K、V 兩軸可各自壓成 4-bit 18-byte 區塊。 | 262144 token 的 session 從 6.44 GB 降到 3.42 GB，K4V4 為 1.81 GB——代價是量出來的，不是假設的。 |

混合批次預設開啟；KV 快取預設 `f16`，`q8` 是一個參數就能開。兩者都不改變模型
語意。

目標硬體：Ryzen AI MAX+ 395 / Radeon 8060S（`gfx1151`），最高 128 GiB 統一
記憶體。模型重心：Qwen3.8 Flash-Next 與 Qwen3.8 27B。

**目錄：** [快速開始](#快速開始) ·
[混合批次排程](#混合批次排程高併發) ·
[低位元-kv-快取](#低位元-kv-快取) ·
[核心層加速](#核心層加速) ·
[服務設定建議](#建議的服務設定) ·
[效能基準](#效能基準mbufo-v0241gfx1151-strix-halo) ·
[上游追蹤](#上游追蹤) ·
[建置](#從原始碼建置) · [文件](#文件索引) ·
[限制](#相容性與限制)

## 測速結果速覽

兩組結果，皆於單一台 Strix Halo（gfx1151）、冷 KV、greedy。

**單流（冷 KV）** — Qwen3.8 27B Q4_K_XL 與 Flash-Next ROCmFP4，一次一路：prefill
與投機 decode 吞吐。完整表見[效能基準](#效能基準mbufo-v0241gfx1151-strix-halo)。

**高併發三引擎（開 MTP）** — A Halogen · B gufo-upstream · C MBufo，1–4 路併發、
同 harness、皆暖機。併發 1 / 2 / 3 / 4 的「總／每 session」decode TPS：

| 引擎 | 總 TPS | 每 session TPS |
| --- | --- | --- |
| A Halogen | 85 / 84 / 71 / 88 | 85 / 43 / 24 / 22 |
| B upstream | 61 / 84 / 110 / 125 | 61 / 42 / 37 / 33 |
| C MBufo | 66 / 86 / 100 / 138 | 66 / 44 / 34 / 35 |

A 單流峰值最高，但只在單獨執行時投機——一旦併發就關掉投機頭，每 session TPS 掉到
~22。負載下 **C 擴展最佳**（線性到 138 TPS、每 session 最平）。完整方法與其餘面向
（prefill、fairness、ITL）見 `高併發測速方法.md`。

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

### 建議的服務設定

| 目標 | 設定 |
| --- | --- |
| 16 路、高併發 | `--sessions 16 -c 65536 --kv-quant q8`（約 103 GB，餘 20 GB） |
| 8 路、長對話 | `--sessions 8 -c 131072 --kv-quant q8`（約 101 GB，餘 21 GB） |
| 單路、最大上下文 | `--sessions 1 -c 262144 --kv-quant q8` |
| 還原上游排程 | 環境變數 `GUFO_MIXED_BATCH=0` |

載入器限制的是「session 數 × 上下文」，不是單一上下文。在 128 GB Strix Halo
上以 blocked Q8 快取＋4-bit value 列實測，8 路甚至可到 192K（`-c 196608`，
107 GB，只剩 16 GB 餘裕）；16 路拉到 96K 則會被
`text runner request-state claim exceeds state capacity` 拒載。量測條件為
UD-Q4_K_XL ＋ MTP Q8_0 sidecar ＋ BF16 mmproj，舊的 32K 上限已不成立。

## 混合批次排程（高併發）

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

### 排程器保證什麼

| 保證 | 機制 |
| --- | --- |
| 待跑 decoder 絕不排在 prefill 之後 | 只有在沒有 decoder 待跑時（或正在湊第一批時）才進合併步驟，所以 prefill 只在空隙裡跑，永遠不會插隊到串流請求前面。 |
| chunk 不能把 decoder 擠出輪替 | 每次合併或 prefill 步驟後，`MarkAllDecodersDue` 會把所有常駐 decoder 重新標記為待跑，下一步就全部服務。 |
| 任何 prompt 都無法獨佔裝置 | chunk 是有上限的切片（`--prefill-chunk` 預設 512 token），25 萬 token 的 prompt 每輪只餵一片。 |
| 同等的 session 輪流 | decode 佇列會輪替（`PopDecoder` 把服務過的請求移到隊尾），不是固定隊頭順序，所以沒有哪個 session 能一直贏得前排。 |
| 單一客戶端無法塞滿佇列 | `--max-pending-per-client` 限制單一客戶端的待處理請求（回 `client_quota`），出隊也跨客戶端輪流，一個客戶端爆量不會餓死別人。 |
| 某一層拒絕也不損失 | 每一層都可獨立降級；投機列被拒就退回原本路徑：正確，只是沒有加速。 |

### 換到什麼

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

## 低位元 KV 快取

上游的 blocked Q8 KV 路徑被泛化成「格式加編解碼器」抽象，且 key 與 value 兩軸
都可再壓成 4-bit 的 18-byte 打包區塊（每 32 個元素 18 位元組）。兩軸精度獨立
選擇，因此 K8V8、K4V8、K8V4、K4V4 四種組合都可達成。

| 模式 | 狀態 | 說明 |
| --- | --- | --- |
| `f16` | 預設 | 基準數值。KV 快取不做量化。 |
| `q8` | 支援 | Blocked 8-bit，每 32 個元素配一個 f16 尺度（34 位元組）。 |
| `q8` + `GUFO_KV_V4=1` | 手動開啟，K8V4 | 4-bit value 打包成 18 位元組，key 維持 8-bit。 |
| `q8` + `GUFO_KV_Q4=1` | 手動開啟，K4V8 | 4-bit key 打包成 18 位元組，value 維持 8-bit。 |
| `q8` + 兩者同時 | 手動開啟，K4V4 | key 與 value 都是 4-bit。 |

**建議使用 `q8`。** blocked 模式在編解碼層級是位元精確的，實務上可互換；
`f16` 仍是預設與基準。4-bit 路徑是給「KV 記憶體成為瓶頸」的機器手動開啟的環境
變數鉤子，不是 `--kv-quant` 的格式選項，也不會改變任何預設值——未設定時引擎與
不含這些路徑的版本位元相同（以 logit dump 雜湊比對驗證）。完整說明見
[docs/KV-V4.md](docs/KV-V4.md)。

### 佔用

每個 KV 位置的位元組數，以及 `-c 262144`、Qwen3.8 Flash-Next 12 個注意力層下
單一 session 的總量：

| 模式 | 每位置位元組 | 262144 單一 session |
| --- | --- | --- |
| `f16` | 2048 | 6.44 GB |
| `q8` | 1088 | 3.42 GB |
| K8V4 / K4V8 | 832 | 2.62 GB |
| K4V4 | 576 | 1.81 GB |

### 代價（量出來的）

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

## 核心層加速

以下改動同時提升單路與批次吞吐，且都不改變模型語意。

- **ROCmFP4-FAST 專家走 fp16 WMMA 核心。** 4.25 bpw 的 `Q4_0_ROCMFP4_FAST` down
  專家現在與 Q4_K/Q8_0 專家共用同一條 fp16 WMMA routed-expert GEMM（perm codebook
  LUT 餵進既有 Q8_0 反量化），FAST 模型的 MoE prefill 因此跑在矩陣核心上、不再掉
  進通用 `mul_mat_q`。gfx1151 實測 FAST 模型 prefill 打平或勝過 UD-Q4_K_XL 基線
  （1.9K／5K token 為 1564／1368 對 1556／1324 tok/s），decode 速度與 64k 工具
  梯子正確性不變。
- **int8 WMMA 直讀 prefill。** 單 session 的 causal prefill
  （`WmmaCausalAttention`）現在直接讀 blocked int8 KV 快取，跳過 F16 staging
  反量化。它覆蓋快取能持有的所有 KV 區塊佈局（K34/K18 x V34/V18，dense 與
  sparse），且由與快取寫入端相同的格式/環境變數選取，因此任何佈局都不會被誤
  讀。在 14K 單流 prefill 上比 scalar 路徑**快 30%**（1085 → 1408 tok/s）。
- **每個 prefill pass 都走 wide 路徑。** wide prefill 路徑（融合投影、成對
  routed expert tile、WMMA attention）適用於所有寬於一個 vector batch 的
  prefill pass，不再只限 1024+ token。

## 效能基準（MBufo v0.24.1，gfx1151 Strix Halo）

冷 KV（每請求唯一 prompt，`cached_tokens≈0`）、think off、**F16 KV**、`-j 1`、
ctx 262144。四組設定：27B（UD-Q4_K_XL 對 ROCmFP4-FAST）＋ Flash-Next（ROCmFP4-FAST
對 UD-Q4_K_XL）。投機：27B＝DFlash2，Flash-Next＝MTP。

這些是單流數字，用來隔離模型與建置版本的選擇；負載下改變的是同時能跑幾條，
那是[混合批次](#混合批次排程高併發)的故事。

### Prefill（tok/s，越高越好）

| tokens | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---:|---:|---:|---:|---:|
| 2K | 544 | 432 | **1739** | 1301 |
| 8K | 568 | 534 | **1701** | 1549 |
| 32K | 526 | 517 | 1598 | **1618** |
| 128K | 385 | 382 | **1504** | 1507 |
| 250K | — | — | **1438** | 1415 |

> 27B ROCmFP4-**FAST** 的 prefill ≈ 非 FAST 的 STRIX 版——single-scale 只換 codebook，
> 不換 fp16-WMMA prefill 引擎（432/534/517/382 對 483/523/520/383）。FAST 真正的速度
> 來自 int8-MMQ 消費路徑，而那條路徑只餵給 Flash-Next 的 MoE experts，dense 27B 沒接。

### Decode（tok/s，投機開）

標準：5 個冷 KV 情境（標明 temp）＋ 8 任務 ReAct 代理套件。

| 情境 | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---|---:|---:|---:|---:|
| 散文 (temp 0.4) | 21.8 | 21.4 | **45.0** | 43.6 |
| 費波那契 (temp 0.4) | 68.7 | 60.6 | **87.0** | 75.5 |
| JSON 約束 (temp 0) | 55.7 | 48.1 | **55.8** | 51.6 |
| 貪心程式碼 /128 (temp 0) | 62.4 | 57.4 | 69.9 | **71.8** |
| 重複文本 /256 (temp 0) | 71.1 | 62.0 | **74.9** | 74.5 |

### 代理任務（agent_bench_v2，t1–t8，沙箱工具）

| | 27B Q4_K_XL | 27B FP4-FAST | FN ROCmFP4 | FN Q4_K_XL |
|---|---:|---:|---:|---:|
| 通過 | 8/8 | 8/8 | 8/8 | 7/8 |
| 核爆（須為 0） | 0 | 0 | 0 | 0 |
| 生成 tps 區間 | 9.4–32.4 | 10.7–21.0 | 21.4–57.4 | 31.8–53.5 |

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
不會與上游的 `vX.Y.Z` 撞名。

| 項目 | 值 |
| --- | --- |
| 上游倉庫 | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| 上游基準 release | `v0.20.0` |
| 已合併的上游 `main` | `0c03748a` — *chore(main): release 0.20.0*（2026-10-21） |
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
| [CHANGELOG.md](CHANGELOG.md) | 上游發布歷史與 MBufo release |

## 相容性與限制

- 混合批次改變的是排程，不是模型語意。任何無法服務整批的層都會退回舊路徑。
- `--kv-quant q8` 是以「不再與基準完全一致」換取記憶體。
  當與基準的精確一致性比容量更重要時，請用 `f16`。
- 對話狀態以 KV 格式區分版本；用某一種格式寫出的快照不會被另一種格式重用。
- 上游的 release 標籤不可變，也不會被當成 MBufo 的標籤重複使用。

## 授權與致謝

上游 gufo 採 MIT 授權；改作程式碼與相依套件各自保留其聲明，收錄於
[NOTICE](NOTICE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 與
`licenses/`。MBufo 保留這些聲明。模型權重不隨附，依各發布者條款使用。

MBufo 是 [gufo](https://github.com/gufo-org/gufo)（by gufo-org 及貢獻者）的
分支。低精度 KV 與性能調優的工作建立在上游的 Q8 KV cache、int8 WMMA、
persistence 與 history drafting 等改動之上。
