# MBufo

專為 AMD Strix Halo 打造的高併發 [gufo](https://github.com/gufo-org/gufo) 分支。

> [English version](README.md)

MBufo 保留上游引擎原貌，只改動「同時有多個請求在飛時，晶片到底有多忙」這一段。
上游 gufo 負責單路的極限效能，MBufo 負責在多路負載下把硬體打滿，且不改變模型輸出的語意。
底層模型、核心與絕大部分程式碼都來自上游開源專案，在此致謝 gufo-org 及所有貢獻者。

目標硬體：Ryzen AI MAX+ 395 / Radeon 8060S（`gfx1151`），最高 128 GiB 統一記憶體。
模型重心：Qwen3.8 Flash-Next 與 Qwen3.8 27B。

## 上游追蹤

MBufo 追隨上游 `main` 並以合併方式跟進，而不是分叉走自己的路。上游的 release 標籤維持不可變；
MBufo 的 release 標籤一律加上 `mbufo-vX.Y.Z` 前綴，因此永遠不會與上游的 `vX.Y.Z` 撞名。

| 項目 | 值 |
| --- | --- |
| 上游倉庫 | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| 上游基準 release | `v0.17.1` |
| 已合併的上游 `main` | `e7730942` — *feat(serve): prepare the structured-output vocabulary before serving (#638)*（2026-10-18） |
| 領先上游 release 的提交數 | 51 |
| 合併策略 | 上游 `main` 以 `--no-ff` 合併；MBufo 專屬提交保持可辨識，上游歷史永不改寫 |

上游發布新版本時，「上游基準 release」這一列會在合併它的同一個提交裡更新，
因此本 README 永遠會標明這棵樹確切包含到上游的哪個狀態。

## MBufo 帶來了什麼

- **混合批次注意力（mixed-batch attention）。** 把 1 個 decode 列與 N 個 prefill-chunk 列
  裝進同一個 `ForwardBatch`，用單一混合注意力核心取代逐項注意力。這消除了低併發時
  decode / prefill 輪流空轉的現象：小批次 2.1 倍，服務層實測 5.3 秒對比 6.3 秒。
- **32 路封裝（32-session envelope）。** `ForwardBatch` 從 8 session / 8 列放大到
  32 session / 1024 列，過程中一併修掉三個硬上限（GDN、MoE、logits 緩衝）。
  16 路併發實測 GPU 使用率穩定在 99%，輸出全數正確、零錯誤。
- **批次化工作的 MTP 對等。** 進入混合批次的 chunk 會自動補上 predictor-KV
  （frontier 檢查加上 tail），行為與單次 forward 一致。若某一層無法服務投機列，
  該列會降級走原本的路徑：正確，只是沒有加速。
- **Serve 併單。** 排程器把 prefill chunk 與一般 decode 合成一批
  （`TryStepMixedBatch`）。每一層都可獨立降級，任何一步拒絕就退回舊的雙路徑。
- **KV 快取量化。** 上游的 blocked Q8 KV 路徑被泛化成「格式加編解碼器」抽象，
  並在其旁加入了可選用的 FP8 快取。

混合批次預設開啟。設定環境變數 `GUFO_MIXED_BATCH=0` 即可完全還原上游行為。

## KV 快取選擇（`--kv-quant`）

| 模式 | 狀態 | 說明 |
| --- | --- | --- |
| `f16` | 預設 | 基準數值。KV 快取不做量化。 |
| `q8` | 支援 | Blocked Q8，每 32 個元素配一個 f16 尺度（每 32 元素 34 位元組）。 |
| `fp8` | 支援，建議使用 | 軟體實作 E4M3，每 32 個元素配一個 f16 尺度，區塊佈局與 Q8 完全相同（34 位元組）。 |
| `q8` + `GUFO_KV_V4=1` | 支援，需手動開啟 | 8-bit key、4-bit value，18-byte block：KV 縮小 23.5%，平均 KL 0.027（對 `q8`）。見 [docs/KV-V4.md](docs/KV-V4.md)。 |

**建議使用 `--kv-quant fp8`。** 它在編解碼層級是位元精確的，端到端表現與 `q8` 相當、
甚至略優，而記憶體佔用與 `q8` 相同、只有 `f16` 的一半。

以下為對照 CPU f16 基準的實測結果：

- E4M3 編碼器重現了 OCP 規格表全部 21 個項目（含邊界值與次正規數），
  且每個可表示的碼都精確往返（encode 後 decode 再 encode 完全一致）。
- 端到端偏差與 `q8` 相當：第一個位置的 KL 為 `fp8` 0.11752、`q8` 0.11700；
  後段位置 `fp8` 反而更接近基準。在 50 個位置的探測中，貪婪解碼的 argmax
  與 f16 一致率為 `fp8` 38/50、`q8` 33/50。
- 服務層不受影響：MTP draft 命中率維持在 57% 至 81% 區間，
  算術題與長文生成的回答都保持通順連貫。
- FP8 的量化誤差在平滑資料上比 INT8 大，但在離群值偏重的資料上並不較差；
  而 post-RoPE 的 key 正是以離群值為主。這就是它端到端能與 `q8` 打平、
  而非落後的原因。

必須直說的一點：兩種 blocked 模式（`q8` 與 `fp8`）在貪婪解碼下與 `f16` 的分歧，
都比單純用數值誤差能解釋的程度來得大。這是既有 blocked-KV 設計的性質，
不是 FP8 造成的，已另案追蹤。

## 快速開始

以建議的快取設定啟動 Qwen3.8 Flash-Next：

```sh
./build/release/gufo serve llm \
  --model models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mmproj models/qwen3.8-flash-next/mmproj-BF16.gguf \
  --mtp-model models/qwen3.8-flash-next/MTP/agentionai-MTP-Q8_0.gguf \
  --speculative mtp --think off \
  --served-model-name gufo-qwen3.8-flash-next \
  --sessions 2 -c 262144 --kv-quant fp8
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
| 單路、最大上下文 | `--sessions 1 -c 262144 --kv-quant fp8` |
| 高併發 | `--sessions 16 -c 32768`（約佔 99 GB；32K 用來限制 KV 大小） |
| 還原上游排程 | 環境變數 `GUFO_MIXED_BATCH=0` |

## 從原始碼建置

支援的目標平台為 Linux x86-64 上的 AMD Strix Halo（`gfx1151`）。
建置方式即上游的建置方式，MBufo 未增加任何建置系統需求。

### 使用 Nix

```sh
nix build
./result/bin/gufo diagnose
./result/bin/gufo serve llm --model /path/to/model.gguf
```

### 不使用 Nix

請安裝 C++20 編譯器、CMake 3.21+、Ninja、pkg-config、ROCm 開發函式庫
（HIP、hipBLAS、hipBLASLt、rocBLAS、hipCUB、rocPRIM、rocWMMA）
以及一般系統函式庫（ICU、libcurl、OpenSSL、libpng、libjpeg、libwebp、FFmpeg）。
目前驗證過的工具鏈為 GCC 15.3 搭配 ROCm 7.2.3。

```sh
cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
```

GPU 驅動必須允許你的使用者存取 `/dev/kfd` 與 `/dev/dri`。模型權重需另行取得；
每個模型指南（見 [docs/models](docs/models/README.md)）都列出所需檔案與基準設定。
完整的相依性清單、ROCr queue-memory 補丁與容器相關說明，請見
[上游建置文件](https://github.com/gufo-org/gufo#build-from-source)。

## 文件索引

| 文件 | 內容 |
| --- | --- |
| [docs/MIXED_BATCH.md](docs/MIXED_BATCH.md) | 混合批次設計：鐵律、驗證矩陣、量測數據 |
| [docs/KV-CACHE.md](docs/KV-CACHE.md) | 對話與元件快取（上游） |
| [docs/SERVER.md](docs/SERVER.md) | HTTP API 契約 |
| [docs/CLI.md](docs/CLI.md) | 命令列參照 |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | 建置、測試、preset |
| [docs/TESTING.md](docs/TESTING.md) | 託管 CI 測試組與本機品質檢查 |
| [CHANGELOG.md](CHANGELOG.md) | 上游發布歷史與 MBufo release |

## 相容性與限制

- 混合批次改變的是排程，不是模型語意。任何無法服務整批的層都會退回舊路徑。
- `--kv-quant q8` 與 `--kv-quant fp8` 是以「不再與基準完全一致」換取記憶體。
  當與基準的精確一致性比容量更重要時，請用 `f16`。
- 對話狀態以 KV 格式區分版本；用某一種格式寫出的快照不會被另一種格式重用。
- 上游的 release 標籤不可變，也不會被當成 MBufo 的標籤重複使用。

## 授權與致謝

上游 gufo 採 MIT 授權；改作程式碼與相依套件各自保留其聲明，收錄於
[NOTICE](NOTICE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 與 `licenses/`。
MBufo 保留這些聲明。模型權重不隨附，依各發布者條款使用。

MBufo 是 [gufo](https://github.com/gufo-org/gufo)（by gufo-org 及貢獻者）的分支。
混合批次的工作建立在上游的 Q8 KV cache、int8 WMMA、persistence 與
history drafting 等改動之上。
