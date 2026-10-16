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

## Phase 4 TODO（scheduler 接線）

- 合併點：`StepPrefill` 的 chunk + `StepDecodeBatch` 的 decodes 合成一個
  `ForwardBatch`。注意 MTP 草稿/verify、preview token、sampler 狀態、
  snapshot/checkpoint 都綁在單 request 路徑上， blast radius 大。
- 前置實驗（零風險）：engine 級 benchmark——1 個混合 batch vs
  decode-forward + chunk-forward 兩次調用的 wall time，量化獎品再動 serve 路徑。
- Q8 KV + MTP verify 行在 mixed 下尚未驗證（`c37b1d5e` 只做了指針/staging 框架）。
- 上線門檻：needle 51207 + 50K prefill A/B + TTFT（目標 27s→3s 量級）實測。
