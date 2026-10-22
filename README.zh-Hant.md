# MBufo

專為 AMD Strix Halo 打造的 [gufo](https://github.com/gufo-org/gufo) 分支，
焦點是**低精度 KV 快取**與**性能調優**。

> [English version](README.md)

MBufo 保留上游引擎原貌，只改動兩件事：KV 快取用多少精度，以及多路併發時晶片
到底有多忙。上游 gufo 負責單路的極限效能；MBufo 加入低精度 KV 格式——代價是
**量出來**的、不是假設的——以及讓 GPU 在多路負載下持續打滿、且不改變模型輸出
語意的調優。底層引擎、核心與絕大部分程式碼都來自上游開源專案，在此致謝
gufo-org 及所有貢獻者。

目標硬體：Ryzen AI MAX+ 395 / Radeon 8060S（`gfx1151`），最高 128 GiB 統一
記憶體。模型重心：Qwen3.8 Flash-Next 與 Qwen3.8 27B。

## MBufo 帶來了什麼

### 低精度 KV 快取

上游的 blocked Q8 KV 路徑被泛化成「格式加編解碼器」抽象，
且 key 與 value 兩軸都可再壓成 4-bit 的 18-byte 打包區塊（每 32 個元素 18
位元組）。兩軸精度獨立選擇，因此 K8V8、K4V8、K8V4、K4V4 四種組合都可達成。

| 模式 | 狀態 | 說明 |
| --- | --- | --- |
| `f16` | 預設 | 基準數值。KV 快取不做量化。 |
| `q8` | 支援 | Blocked 8-bit，每 32 個元素配一個 f16 尺度（34 位元組）。 |
| `q8` + `GUFO_KV_V4=1` | 手動開啟，K8V4 | 4-bit value 打包成 18 位元組，key 維持 8-bit。 |
| `q8` + `GUFO_KV_Q4=1` | 手動開啟，K4V8 | 4-bit key 打包成 18 位元組，value 維持 8-bit。 |
| `q8` + 兩者同時 | 手動開啟，K4V4 | key 與 value 都是 4-bit。 |

**建議使用 `q8`。** blocked 模式在編解碼層級是位元精確的，
實務上可互換；`f16` 仍是預設與基準。4-bit 路徑是給「KV 記憶體成為瓶頸」的
機器手動開啟的環境變數鉤子，不是 `--kv-quant` 的格式選項，也不會改變任何預設
值——未設定時引擎與不含這些路徑的版本位元相同（以 logit dump 雜湊比對驗證）。
完整說明見 [docs/KV-V4.md](docs/KV-V4.md)。

#### 正確性優先

每個 KV 位置的位元組數，以及 `-c 262144`、Qwen3.8 Flash-Next 12 個注意力層下
單一 session 的總量：

| 模式 | 每位置位元組 | 262144 單一 session |
| --- | --- | --- |
| `f16` | 2048 | 6.44 GB |
| `q8` | 1088 | 3.42 GB |
| K8V4 / K4V8 | 832 | 2.62 GB |
| K4V4 | 576 | 1.81 GB |

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

`q8` 對照 CPU f16 基準：端到端偏差維持在 blocked-KV 設計範圍內，MTP draft 命中率維持在 57% 至 81% 區間，算術題與長文生成保持通順。

必須直說的一點：這些 blocked 模式在貪婪解碼下與 `f16` 的分歧，都比單純用數值
誤差能解釋的程度來得大。這是既有 blocked-KV 設計的性質，不是 4-bit
區塊造成的，已另案追蹤；而在它之上再疊 4-bit 區塊，只會把差距放大，不會優雅
地疊加。

### 性能調優

- **int8 WMMA 直讀 prefill。** 單 session 的 causal prefill
  （`WmmaCausalAttention`）現在直接讀 blocked int8 KV 快取，跳過 F16 staging
  反量化。它覆蓋快取能持有的所有 KV 區塊佈局（K34/K18 x V34/V18，dense 與
  sparse），且由與快取寫入端相同的格式/環境變數選取，因此任何佈局都不會被誤
  讀。在 14K 單流 prefill 上比 scalar 路徑**快 30%**（1085 → 1408 tok/s）。

- **每個 prefill pass 都走 wide 路徑。** wide prefill 路徑（融合投影、成對
  routed expert tile、WMMA attention）適用於所有寬於一個 vector batch 的
  prefill pass，不再只限 1024+ token。
- **混合批次注意力（mixed-batch attention）。** 把 1 個 decode 列與 N 個
  prefill-chunk 列裝進同一個 `ForwardBatch`，用單一混合注意力核心取代逐項注意
  力。這消除了低併發時 decode / prefill 輪流空轉的現象：小批次 2.1 倍，服務層
  實測 5.3 秒對比 6.3 秒。
- **32 路封裝（32-session envelope）。** `ForwardBatch` 從 8 session / 8 列放大
  到 32 session / 1024 列，過程中一併修掉三個硬上限（GDN、MoE、logits 緩衝）。
  16 路併發實測 GPU 使用率穩定在 99%，輸出全數正確、零錯誤。
- **批次化工作的 MTP 對等。** 進入混合批次的 chunk 會自動補上 predictor-KV
  （frontier 檢查加上 tail），行為與單次 forward 一致。若某一層無法服務投機列，
  該列會降級走原本的路徑：正確，只是沒有加速。MTP 預設 draft 深度為 5（loader
  對 Flash-Next 上限 7）；深度 5 在 decode 上與 7 持平，在 greedy 各 workload
  上領先 5–6 tok/s。
- **Serve 併單。** 排程器把 prefill chunk 與一般 decode 合成一批
  （`TryStepMixedBatch`）。每一層都可獨立降級，任何一步拒絕就退回舊的雙路徑。

混合批次預設開啟。設定環境變數 `GUFO_MIXED_BATCH=0` 即可完全還原上游行為。

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
| 單路、最大上下文 | `--sessions 1 -c 262144 --kv-quant q8` |
| 高併發 | `--sessions 16 -c 65536 --kv-quant q8`（約 103 GB，餘 20 GB） |
| 多個小 session | `--sessions 32 -c 32768 --kv-quant q8`（約 107 GB，餘 16 GB） |
| 還原上游排程 | 環境變數 `GUFO_MIXED_BATCH=0` |

載入器限制的是「session 數 × 上下文」，不是單一上下文。在 128 GB Strix Halo
上以 blocked Q8 快取＋4-bit value 列實測，可通過的組合上限是 100 萬個 token
slot：`16 × 65536` 與 `32 × 32768` 都能載入，`16 × 98304`、`24 × 65536`、
`32 × 65536` 會以 `text runner request-state claim exceeds state capacity`
拒載。量測條件為 UD-Q4_K_XL ＋ MTP Q8_0 sidecar ＋ BF16 mmproj，因此舊的
32K 上限已不成立；客戶端不多但對話長時可直接用 64K。

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
