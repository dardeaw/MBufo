# MBufo Decode & Serving 優化交接清單（參考 Halogen Flash Server 0.17～0.21.0）

依 **「跨模型共用度（含 `qwen38_flash_next`、`inkling`+`dspark`、`gemma4`、`laguna`、`deepseek_v4_flash`）× ROI」** 排序，並依檔案邊界切分以便多模型平行開發。

## 共用前置規範
- **主機與路徑**：`192.168.68.55`，Repo `/home/dardeaw/git/gufo-final2`，建置 `ninja -C /home/dardeaw/git/gufo-final2/build/release mbufo`。
- **測試與壓測**：基準統一對比 `~/llama.cpp`（或同機前後對照），嚴禁編造未實測數據；改完若需測 `.55` GPU，須先停 `mbufo.service`，測完務必重啟 `mbufo.service`（`:8741`）。
- **檔案鎖定提醒**：`src/cli/serve/inference_backend.cpp` 與 `src/cli/serve/text_generation_scheduler.cpp` 屬於高衝突區，**同一時間只派給一個任務修改**，其餘任務改各自獨立模組。

---

## Task 1｜全模型防呆與開機 Warm-up（EOT in `<think>` / 未閉合 Tool Call + GPU Warm-up）
- **適用範圍**：全 GPU LLM（含 `qwen38_flash_next`、`inkling`+`dspark`、`gemma4`、`laguna`、`qwen`）。
- **建議模型／難度**：輕量～中階模型｜★☆☆☆☆｜**可最先單獨開工**
- **目標檔案**：
  - `src/cli/serve/http_server.cpp`、`src/cli/serve/inference_backend.cpp`（參考現有 `src/cli/serve/npu_warmup.hpp` 與 `model_admin_api.cpp` 的 `warmup_ms` 欄位）
- **實作內容**：
  1. **EOT 防呆**：串流與非串流 parser 若偵測模型在 `<think>` 未閉合或 tool call 尚未閉合時提前輸出 turn-stop / EOT（如 `<|im_end|>`、`〈|EOS|〉`），立刻在該點終止輸出（或補閉合標籤），絕不讓後續解碼繼續吐出幻覺 `<|im_start|>user` / `assistant` 假對話；可於 `usage` 回報原因旗標。
  2. **GPU Warm-up**：模型載入完成、`listening` 就緒前，跑 1 筆極短（~4～8 tokens）warm-up 請求，把首次 HIP Graph capture / lazy buffer init 吃掉，避免第一個真實請求多等首字延遲。
- **驗收標準**：啟動 log 印出 GPU `warmup_ms`；構造「`<think>` 內強塞 EOT」測試案例，確認不會吐出偽造 `user` turn。

---

## Task 2｜智慧 Prompt Lookup（去行號 + JSON Un-escape）
- **適用範圍**：所有投機解碼模型（`qwen38_flash_next`、`gemma4`、`qwen` DFlash2、`deepseek_v4_flash` DSpark、`inkling` DSpark）。
- **建議模型／難度**：中階模型｜★★☆☆☆｜**獨立度高，可與 Task 1 平行（先做核心庫不碰 `inference_backend.cpp`）**
- **目標檔案**：
  - `src/core/speculative/prompt_lookup.hpp`（全模型共用版）
  - `src/models/qwen38_flash_next/lookup_draft.hpp`（`LookupIndex`）與 `src/models/qwen38_flash_next/draft_backend.cpp`
  - 視整合方式對接 `src/models/inkling/dspark.hip` / `src/models/deepseek_v4_flash/`
- **實作內容**：
  1. 目前 `prompt_lookup.hpp` / `LookupIndex` 只對原始 commit token ID 做 4-gram hash，遇到 coding agent 讀檔帶行號（`123: ` / `L123:`）或 tool output 裡的 JSON 跳脫字元（`\n`、`\"`、`\\`、`\t`）會完全 miss。
  2. 在 prefill / committed history 建索引階段（**不可卡在 decode 熱路徑**），偵測 tool/user 區段並建立輔助「去行號＋un-escape」後的乾淨 token index（須透過模型 tokenizer 做輕量分段轉換，注意 BPE 邊界）。
  3. Decode 提案時若主索引 miss，查詢清洗後的輔助索引；`sampled`（`temp > 0`）仍維持較長最小匹配門檻（`kSampledMinMatch`），避免常見語法誤觸。
- **驗收標準**：模擬「先給帶 `1: ...` 行號或 JSON escaped 的 1,500-token 程式碼 tool result，再要求整檔重寫/Edit」，量測 `lookup` 命中率與 decode tok/s（參考值：整檔重寫 +30%～40%、Edit +20%）。

---

## Task 3｜Grammar / Schema-Forced Drafter（嚴格 Tool Call 跳過草稿模型）
- **適用範圍**：所有支援 `ConstraintVocabulary` 與 `multi_token_decode` 的投機模型（含 `qwen38_flash_next`、`inkling`+`dspark`、`deepseek_v4_flash`、`gemma4`）。
- **建議模型／難度**：強模型（推理/架構能力佳）｜★★★☆☆｜**建議等 Task 1 改完 `serve/` 後接手**
- **目標檔案**：
  - `src/cli/serve/text_generation_scheduler.cpp`、`src/cli/serve/inference_backend.cpp`
  - `src/core/speculative/speculative_verifier.hpp` / `.cpp`
  - `src/models/qwen38_flash_next/mtp_policy.hpp` & `src/models/qwen38_flash_next/draft_backend.cpp`
- **實作內容**：
  1. 當請求啟用 JSON schema / `strict: true` tool，且 host 端 constraint 狀態機判定接下來 `1..8` 個 token 為**唯一確定字元**（僅 1 個合法候選，例如 `{"name": "`、key 名、冒號引號、結尾括號）時，直接由 host 產出確定 token 序列送 target model 單步驗證，**跳過 MTP / DSpark draft pass**。
  2. **關鍵防衝突**：Grammar-forced token **不可計入 `MtpPolicy::Observe(accepted, drafted)`**，否則會汙染 MTP/DSpark 的動態猜測深度統計；`timings` 可獨立拆出 `grammar` 統計欄位。
  3. Sampled（`temp > 0`）下因機率為 1.0，跳過機率檢驗在數學上仍嚴格等價。
- **驗收標準**：對含 `strict: true` tool schema 的多輪 tool call 測試，確認 JSON 合法性 100%、`MtpPolicy` 統計不受汙染、tool-call 區段 decode tok/s 提升（參考值：+15%～20%）。

---

## Task 4｜回覆結尾自動快取＋長對話抗驅逐策略
- **適用範圍**：所有支援 `TextRunnerState` snapshot / `component-cache-v1` 的模型。
- **建議模型／難度**：中階模型｜★★★☆☆｜**獨立模組，可與 Task 2/3 平行**
- **目標檔案**：
  - `src/cli/serve/component_text_cache.cpp`、`src/cli/serve/component_cache_metrics.hpp`
- **實作內容**：
  1. **回覆結尾自動快取**：每輪生成結束時，將結尾狀態以「可隨時淘汰（non-pinned）」形式留一份在 RAM cache（設單槽上限，避免擠爆多模型或 `--cache-ram-bytes` 預算），讓下一輪 client 原封不動帶回上輪回覆時直接從結尾接續。
  2. **長對話抗驅逐**：修正純 LRU 在「插入數十個短請求」時把 100K+ 長對話洗掉的問題；驅逐評分改考慮「自上次命中後累積處理的 token/工作量 vs 該條目自身長度」，長對話須經過相當於自身長度的其他工作量才允許淘汰。
- **驗收標準**：(a) 兩輪連續對話量測續聊 TTFT 下降；(b) 插入 30 筆短請求後再回到 64K+ 長對話，確認 cache hit 未被洗掉。

---

## Task 5｜併發批次解碼與多路投機排程（Concurrency Mode）
- **適用範圍**：所有實作 `DecodeBatch` 的模型（`qwen38_flash_next`、`deepseek_v4_flash`、`glimmer`，及未來 `inkling`）。
- **建議模型／難度**：強模型（GPU/排程熟悉）｜★★★★☆｜**不可與 Task 3 同時改 `text_generation_scheduler.cpp`**
- **目標檔案**：
  - `src/cli/serve/text_generation_scheduler.cpp`
  - `src/models/qwen38_flash_next/kernels/rocm/batch.cpp` & `src/models/qwen38_flash_next/kernels/rocm/executor.hpp`（`graphs_`）
  - `src/core/compute/attention/mixed_batch_attn.hip.cpp`
- **已知衝突與實作重點**：
  1. 檢查 `.55` 開 `--sessions 2`～`8` 時，`qwen38_flash_next` 與 `deepseek_v4_flash`（`DsparkBatch`）是否真的共用一次 verify pass，還是中途退回 `serial-fallback`。
  2. **解決與 HIP Graph (`graphs_`) 的衝突**：對常見 `(batch_width, draft_width)` 組合做 bucketed graph capture，或確認非 capture 的 `batch.cpp` 路徑不會拖慢單請求 `n=1`。
  3. 確保多請求併發時 `temp=0` reduction order 固定（不隨併發數或請求到達順序改變輸出）。
- **驗收標準**：2 / 4 / 8 併發壓測總吞吐較單線排隊提升；`temp=0` 單請求與併發輸出一致；單請求 decode 不掉速。

---

## Task 6｜MTP 深度 7 ＋ Gate 調優（僅 MTP 模型快速調參）
- **適用範圍**：`qwen38_flash_next`、`gemma4` MTP（**Inkling DSpark 不適用**）。
- **建議模型／難度**：輕量模型｜★☆☆☆☆｜**隨時可測、不改核心邏輯**
- **目標檔案**：
  - `/etc/systemd/system/mbufo.service`（`-d` 參數）、`src/models/qwen38_flash_next/mtp_policy.hpp` & `src/models/qwen38_flash_next/draft_backend.cpp`
- **實作內容**：
  1. `.55` 目前 `mbufo.service` 未傳 `-d`，預設為 `5`（雖然 `kMaxMtpDraftTokens = 7`）。實測 `-d 7` 搭配信心度門檻（參考 `0.90` early-stop）在程式碼、對話、tool call 上的淨 decode tok/s。
  2. 量測 `draft` 迴圈與 `verify` 之間是否還有多餘的 host sync 或 `draft_subset` 開銷。
- **驗收標準**：提出 `-d 5` vs `-d 7` 在程式碼／一般對話的實測 `decode_tps` 與 `draft_accepted` 對比，若淨收益為正則更新 service 參數。

---

## Task 7｜Vision Encoder 4x 加速
- **適用範圍**：各自獨立——`src/models/qwen/vision/encoder.hip`（Qwen 全家族）或 `src/models/inkling/vision.hip`（Inkling 專用）。
- **建議模型／難度**：強模型（HIP/WMMA 熟悉）｜★★★★☆｜**完全獨立、可隨時平行派工**
- **實作內容**：
  1. 先分別實測 `1280×1280` 圖片在 Qwen vision 與 Inkling vision 的編碼耗時（對比 Halogen ~0.5s）。
  2. 針對慢的 ViT patch attention / MLP 接上 WMMA FlashAttention 或調整 `hipBLASLt` / BF16 GEMM tile 參數。
- **驗收標準**：`1280×1280` 圖片前處理耗時顯著下降，且多模態問答/OCR 輸出與原版一致。

---

## Task 8｜MTP Draft Head 輕量化（僅 MTP 模型）
- **適用範圍**：`qwen38_flash_next`（目前 `.55` 跑 `agentionai-MTP-Q8_0.gguf`）與 `src/models/qwen/hip/mtp/`。
- **建議模型／難度**：中～強模型｜★★★☆☆｜**可與其他任務平行**
- **實作內容**：檢查 MTP loader 與 GEMV kernel 支援的量化格式（如 `Q4_K` / `Q5_K` / `ROCmFP4`），將 MTP head 壓小以減少每輪 draft 的記憶體頻寬，實測「接受率微降 vs 單步變快」後的淨 decode tok/s 是否為正。

---

## Task 9｜DeltaNet Recurrent State 降精度（僅 `qwen38_flash_next`）
- **適用範圍**：僅 `qwen38_flash_next`（Linear Attention / SSM 層；**Laguna、Inkling、Gemma 4、DS4 均不適用**）。
- **建議模型／難度**：強模型｜★★★☆☆｜**最後做，且有破壞性相容注意**
- **目標檔案**：`src/models/qwen38_flash_next/continuation_layout.cpp`（目前 `kConvolution` 與 `kRecurrent` 皆為 `sizeof(float)`）、對應 ROCm kernels。
- **強制注意**：改為 FP16/BF16 **會使 `--cache-disk /mnt/nvme-data/mbufo-cache` 舊快取全部不相容**，必須同步升級 `state_abi` / cache schema 版本號讓舊快取自動重建，並補長上下文（128K+）品質與 PPL 回歸驗證。

---

## 建議平行派工組合（互不打架）
- **Model A（核心/排程強模型）**：先做 **Task 1** → 做完後接 **Task 3（Grammar-Forced Drafter）** → 再接 **Task 5（併發批次）**。
- **Model B（中階模型）**：**Task 2（智慧 Prompt Lookup）**（專攻 `src/core/speculative/` + `lookup_draft.hpp`）。
- **Model C（中階模型）**：**Task 4（`component_text_cache.cpp` 結尾快取與抗驅逐）**。
- **Model D（HIP/Kernel 強模型）**：**Task 7（Vision Encoder 加速）** 或 **Task 6（MTP `-d 7` 實測調優）**。
