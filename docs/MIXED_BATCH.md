# Mixed-Batch Attention（decode + prefill-chunk 同一批）

目標：把「1 個 decode 行 + N 個 prefill-chunk 行」裝進**一次** `ForwardBatch`，
用一個 `WmmaMixedAttention` 核代替逐 item 的 attention core，消除
decode/prefill tick-tock（decode 空等整個 prefill chunk 跑完）。

狀態：attention core 已驗證正確（`r151-mixed`，gate：`GUFO_MIXED_BATCH=1`）。
Scheduler 接線（Phase 4）未做——production 預設 env 未設，行為零變化。

## 架構

```
ForwardBatch([{decode x1}, {chunk x4}])
  ├─ per-item deferred projections（Q norm/RoPE/KV append，跳過 core+gate）
  ├─ WmmaMixedAttention：packed Q 行，一個 launch，per-(session, kv-head-group) block
  │    MixedBlk{q_off, k_ptr, v_ptr, kv_len, q_pos_base, n_rows}
  └─ 後續層共享 packed 表示（MoE/GDN/ Roadout 等不動）
```

- 入口：`Executor::ForwardBatch`（`kernels/rocm/batch.cpp`），條件
  `!l.linear && UseMixedBatch() && !kv_cache_int8 && items>=2`，
  且每項 plain（無 speculative、無 sparse、非空）。
- 核：`WmmaMixedAttentionKernel`（`kernels.hip.cpp`），`WmmaCausalAttention`
  的 dense-only varlen 雙胞胎；`grid(n_blks, 12)`，每 block 服務一個 session。
- 計數器：`Executor::mixed_batches()`，audit 用它確認 mixed 真 fired。

## 三條鐵律（debug 血淚）

1. **Gate 只做一次。** 核的 epilogue 已經 `value *= SigmoidF(gate[offset])`
   （跟 `WmmaCausalAttention` 一樣 output-gate-included），tail **不得**
   再調 `SigmoidMul`——否則 gate 被平方，輸出壓到 ~0.25x。
   （2026-10-16 真實踩坑：`2d38aeee` 修掉。）
2. **Batch 內永遠 `prefill_phase=false`。** `PrefillPhase` 是 thread-local，
   只有 `Forward()` 設，`ForwardBatch` 不設。所以：
   - `PrepareAttention` 在 batch 內永遠走 `<1>` 核（n<32），
     `Forward(kPrefill)` 走 `<4>` 核——同輸入不同求和順序，logits 差 units 級是預期的。
   - `MatrixRows(n) = phase || n>8` 在 batch 內對小 batch 恆 false——
     legacy-batch 的 chunk 行走 scalar narrow 核，不是 WMMA。
3. **跨家族比較沒有意義。** Batch 內 decode 行 legacy 走 narrow、mixed 走 WMMA，
   這是兩種算法（scalar vs tensor-core），不是同一個核的捨入差。
   正確性只看**同家族**比較（見驗證）。

## 驗證（都在 `tests/models/qwen38_flash_next/`）

| 測試 | 比什麼 | 標準 | 現狀 |
|---|---|---|---|
| `CompareMixed`（attention_ops_test） | mixed 核 vs host FP64，同形狀 | < 2e-2 | **1.5e-4**（in-situ 形狀 kv=35/52 已測） |
| `--mixed-audit`（gpu_probe） | mixed-batch vs 單獨 Forward | report-only | 差 units 級（phase+家族差，預期內）；只卡 NaN/counter-stuck |
| `--mixed-audit` + `MX_TEXT=1` | teacher-forcing top-1 吻合率 | 參考標尺 | mixed-vs-narrow 17/30、15/30 |
| 標尺：同條件 Q8-vs-F16 | 已上線量化噪聲 | — | 11/30、1/30（mixed 比它好） |

診斷環境變數（test binary only）：`MX_DUMP`（dump batch logits）、`MX_TEXT=1`
（teacher-forcing 文本比對）。`[mixed] il=.. rows=.. blks=.. ok=..` 是 lib 內
per-layer diag，只有 `GUFO_MIXED_BATCH=1` 才印。

## Phase 4 設計（scheduler 接線，未開工）

現狀測繪（2026-10-17）：

- `StepPrefill`：單 request 調 `runner_request.Prefill(budget)` → runner
  `Prefill` → `session.Sync(整個 prefix[0..next_pos))`（內部增量，只算新行）。
  chunk 不是獨立的 token 段，合併必須進 Sync 語義。
- `StepDecodeBatch`：decodes 經 `AdvanceBatch` → `Session::EvaluateBatch`
 （plain advance）或 `DecodeBatch`（MTP speculative）→ 已走 `exec.ForwardBatch`。
- `Session::MixedBatchImpl`（engine.cpp:1473）已寫好、未被調用：
  MTP draft dance + chunk-last 非採樣 item + bookkeeping；MTP speculative
  item 進 tail 會自動降級 legacy（use_mixed 要求全 plain），正確但不加速。

合併選項：

- A（小）：plain-advance decodes + chunk 共用一個 `ForwardBatch`
  （仿 `EvaluateBatchImpl` 25 行模式加 chunk-last）。MTP speculative 不進，
  自動降級。需新 engine 方法 + 測具（engine 級 harness 目前沒有）。
- B（大）：`MixedBatchImpl` 直連 scheduler（`StepPrefill`+`StepDecodeBatch`
  二合一）。動 preview token、sampler RNG、snapshot/checkpoint、MTP verify、
  scheduler bookkeeping（prefill_tokens/chunks/decode_ready 全要等價重做）。
  MTP session 的 chunk 還要處理 `mtp_enabled_ && prefill_phase` 檢查在 batch
  內被跳過的問題（batch 永遠 phase=false）。

建議順序：A（engine 方法 + 合成驗證）→ scheduler 試點（env + serve flag 雙 gate，
只合併 plain-advance + 無 snapshot 邊界的 chunk，其餘照舊）→ Q8/MTP 覆蓋 →
needle/TTFT 驗收 → 上線。B 的 serve 手術開工前需要單獨 review。

# P0/P1/P2 狀態追加（2026-10-17）

## P0：batch 包絡 32 session / 1024 行（已驗）

- `kBatchSessions` 8→32，單 item/總行數上限 1024，`batch_logits_` 按總行數配
  （622MB；MTP-heads 尾巴 staging 照舊裝得下）。
- 附帶修了三個擋路的硬上限（都是「小形狀能跑、大形狀直接拒/炸」的）：
  GDN（按 n 切分大小兩路：小路徑原樣，大路徑 y-tiling 無 history 核 +
  host-side 3 行 history 拷貝；speculative 大 item 直接拒收保安全）、
  MoE-Q4（vec fast path 保留到 32 行，大行數走 64-tile vec，逐行獨立故 tiling
  精確；WMMA 全行分支因 batch 數值未定暫時不用）、logits。
- 驗：264 行/9 session，decode 行 **bit-exact（0.00）**，chunk 續跑在已知
  prepare 家族差內（1.49）。

## P1：MTP parity（已驗）

- 非 speculative 多行 item（=chunk）：入口 frontier-check（乾淨拒絕，對標
  single-Forward 報錯文案）+ drain 後 `MtpForward kv_only` 餵 predictor。
  target_hidden staging 只拷裝得下的行數（chunk 靠 tail，不靠它）。
- 驗：chunk `MtpPosition` 跟 legacy tail **一字不差**（都落后 trunk 一拍，
  靠 engine `DraftCatchUp` 補齊——那是 P2 的活），mixed 12/12 fire，
  非 MTP 路徑數字完全不變。

## P2：EvaluateBatchWithChunk（已驗，engine 級）

- 拼裝：Feed 的 catch-up + EvaluateBatchImpl 的 advances 處理 + Feed 的
  bookkeeping；失敗回退（advances 逐個 Evaluate，chunk 全前綴 Sync）。
- 驗（`engine_mixed_test`）：chunk tokens/position **精確一致**（含第二 chunk，
  覆蓋 catch-up 協議）、decode logits 小包絡、bench。

## 教訓：全局 phase 不要碰

- 試過整批 `prefill_phase=true` 想讓 chunk 對齊 single-prefill（Prepare<4>、
  wide core、raw bucket），結果 gap 惡化 ~10x（0.60→7.51），子系統交互沒
  隔離乾淨。已 revert，batch 保持 phase-false（驗證過的包絡）。
- per-item `Prepare<4>` 透傳是可能的微優化，但不是正確性項，先不做。

## 性能現狀（engine 級，2 decode + 64 chunk）

- 小 batch（5 行）：混合 **57ms** vs 分開 121ms = **2.1x**。
- 大 batch（66 行）：混合 316ms vs 分開 247ms——**暫輸**，主因 batch-MoE 走
  vec（memory-bound）而單路走 matrix（compute-bound）。WMMA-MoE 修好前，
  混合贏在延遲/併發，不贏單 prefill FLOP。

## P2 驗收（2026-10-17，engine 級 `engine_mixed_test`）

- `EvaluateBatchWithChunk`：plain advances + chunk-last，含 Feed 式 catch-up、
  epoch-safe fallback（照抄 `RunIsolatedBatch`：乾淨重試/髒 poison，絕不原地
  重算髒 session）。
- 數字（2 decode + 64 chunk，plain 與 MTP 一致）：
  decode logits 0.60、continuation 1.57、chunk2 tokens+position 精確、
  teacher-forcing 在通用文本上 0/10（分布太平，top-1 全翻；(1,1) 形狀 17/30，
  Q8 標尺 11/30——包絡內）。
- 速度：batch 316ms vs 分開 247ms（vec-MoE 拖累；WMMA-MoE 修好前先記賬）。

## MTP 調試血淚（按時間順序，別再踩）

1. `target_hidden` 只有 `max_speculative` 行，chunk 行數直接寫爆堆——
   batch 內跳過（tail  cover），已修。
2. `MX_MTP` 初版期望寫錯：legacy tail 本來就只走 `tokens[1:]`，mp 落后一拍
   是設計（engine catch-up 補），不是 bug。
3. `full-rows WMMA MoE` 在 batch 里差 10.07：counts 對過是對的，懷疑 epilogue/
   combine 或 tile-map 的 batch 上下文問題；已回退 vec-64-tiles（0.60，
   tiling 精確），WMMA 另案。
4. 全局 phase 方案回退（0.60→7.51，子系統交互未隔離）。

## P3：serve 併單（2026-10-17，已上線驗證）

- 新鏈路：scheduler 1898 分支（`TryStepMixedBatch`）→ pool
  `PrefillWithCoworkers` → backend `AdvanceBatchWithPrefillChunk` →
  engine `EvaluateBatchWithChunk`。每層可獨立降級，任何一步拒絕都走 legacy
  pair（tick-tock 一輪），行為與今天一致。
- Gate：`GUFO_MIXED_BATCH=1`（scheduler + engine 共用；關掉整條鏈路靜默，
  已驗 mixed 觸發 120 vs 0）。
- Serve A/B（4 session，3 短 + 1 長隨機 prompt，3 輪）：
  開 5.3s vs 關 6.3s（每輪都贏，~16%），輸出全對（精確回覆、數學、摘要），
  零錯誤。幅度不如 engine 級 2.1x（serve 有 MTP/調度/cache 開銷，且 session
  只有 4，上限在併發數不在單批）。
- 安全設計：prepared decode 絕不重採樣；completed 的 advance 絕不重跑；
  脏 session poison 大聲報錯；chunk Sidebar via Sync 冪等自愈。
