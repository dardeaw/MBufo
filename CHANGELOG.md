# MBufo Changelog（Qwen3.8-Flash-Next 推論分支）

以 `gufo-org/gufo` 上游為基底，專注低精度 KV、decode 頻寬與 MTP 推測解碼。
格式：`version.txt` 即版號；只推 fork（`dardeaw/MBufo`），不上游。

## v0.20.8（2026-10-23）
- 修正 blocked-WMMA 路徑的 **double-gate**：`WmmaCausalAttentionKernel` 已在 kernel
  內施加 `SigmoidF(gate)`（kernel 5583 行註明 epilogue apply sigmoid output gate），
  但 blocked-WMMA 跑完又掉到 `executor.cpp:3088` 的 `SigmoidMul` 再 gate 一次，
  輸出剩 ~1/4 大小、corr 0.88，64k 工具呼叫 22/30（27% 失敗）。
  修法：WMMA 走過（`wmma_used`）就跳過該 `SigmoidMul`（fused `AttentionKv`
  簽名無 gate 參數，只靠它 gate 一次）。f16 early-return 在 `SigmoidMul` 前返回
  （kernel 內單次，正確），decode（`rocm::Attention` 無 gate）不受影響。
- 修正 K4V8（`GUFO_KV_Q4` 不帶 V4，K18/V34）的 GPU page fault（signal 6）：
  `AttentionQ8Gemm` 融合投影路徑寫死 34-byte Q8_0 K 塊，K4（18-byte K 列）誤入
  `fused_projection` → 每 token 溢寫 256 B → 30k tok 左右越界 abort。
  修法：`fused_projection` 開關加 `!KvKeyIs4Bit(...)`（鏡像 V4 排除），K4V8 改走
  `Dense` + `PrepareAttentionKv`（獨立 K/V block sizes，正確處理混排）。
- 驗證（64k 工具梯子 N=10，`GUFO_KV_WMMA=1 GUFO_KV_I8=1`，修後 binary）：
  K8V8 / K8V4 / K4V4 / K4V8 皆 **30/30**；生產實例（`+GUFO_KV_V4=1`）9/9 通過。
  prefill 保留 WMMA 速度（64k 全 prefill ~1350 tok/s；fused 路徑約 990 tok/s）。


## v0.20.7（2026-10-22）
- 文件：更新高併發建議。舊的 `--sessions 16 -c 32768` 已非上限；q8 KV＋4-bit
  value 列下實測 `8 × 131072` 載入 101.0 GB（餘 21 GB）、`16 × 65536`
  102.9 GB（餘 20 GB）、`8 × 196608` 106.8 GB（餘 16 GB）。
  `16 × 98304` 會以 `text runner request-state claim exceeds state capacity`
  拒載。README 中英同步，只保留 8 路與 16 路兩種情境。

## v0.20.6（2026-10-22）
- 修正 `GUFO_KV_V4` 下連續性快照把 **value 列誤用 key 列寬度**（544 B）的問題：
  `ContinuationLayout` 現在分別取 key/value 列寬（V4 為 288 B），capture/restore
  不再越界讀取 value 快取。
- 症狀：prompt ≥ ~139k tokens（超過 `max_context × 288 / 544`）的請求回 200 後
  服務 SIGABRT，日誌 `Memory access fault ... Page not present`；關閉 decode 副本
  仍重現。根因是 value 快照每列多讀 256 B，154k 時越過 72 MiB 的 V 快取配置。
- 修正後 154k prompt 可完整 capture（快照 3.11 GB → 2.60 GB）並正常 restore
  （turn2 0.32 s，答案正確）。`kSnapshotPayloadVersion` 18 → 19，舊快照自動失效。
- 驗證：154k prefill + 兩輪對話、154k 快照 restore、16K 快取測試全過；
  `build/gpu-test` 僅 `kernel_resources_test`（WMMA scratch 基線）失敗，與本修正無關。

## v0.20.5（2026-10-22）
- Decode 大型 dense 投影改讀 Q4_1 副本（`ssm_in/out`、`attn_qkv/out`，20 B/block，
  較 Q8_0 少 41% 位元組；prefill 仍走 Q8_0）。
  `GUFO_DECODE_Q4=0` 切 Q5_1 副本，`GUFO_DECODE_Q5=0` 關閉副本（純 Q8_0）。
- 實測（UD-Q4_K_XL，MTP 開，純 decode 256 tok×5 中位）：Q4_1 **47.68 tok/s**
  vs Q8_0 40.63 tok/s（**+17.4%**）；4 題精確答案無變；副本 1.56 GiB。

## v0.20.4（2026-10-21）
- Prefill chunk tile 改為 `GUFO_PREFILL_TILE` 可調（預設 4096，覆寫免重編）。
  驗證 8192 在 24K/48K 提示下與 4096 差異在噪聲內（±2%），故維持預設 4096。

## v0.20.3（2026-10-21）
- 合併上游 `origin/main` v0.20.0（含 #699 MTP 取樣開銷、#702 bench 中斷修正）。
  解決 `gpu_probe.cpp` argparse 衝突；release + gpu-test 全綠。

## v0.20.2（2026-10-21）
- 修正小 q8 prefill 誤入 fused 投影導致 config C 崩潰：
  加 `q8_fused_ok` 閘門（`kv_format==kQ8` 需 `n_tokens>=1024`）。
  `attention_ops` 33s 通過，28-token prefill 輸出正確。

## v0.20.1（2026-10-20）
- Decode 自適應 key-tile splits（窄 attention batch 切分，`use_splits`）。
  隔夜重啟 A/B +6.9%（41.7→44.6 tok/s）；config A decode 均值 43.65。

## v0.20.0（上游基底）
- 對齊上游 v0.20.0；MTP 預設 draft depth 5；#690 寬 prefill 全開；
  F16 GEMM 放寬至 `n_tokens==0`。

## v0.19.0 以前（fork 主軸）
- **int8 WMMA**：blocked-KV causal prefill 直走 int8 WMMA（`GUFO_KV_WMMA`）。
- **Q8 KV**：per-32-block scale（K34/V34 8-bit+scale@32，K18/V18 4-bit+scale@16）。
- **4-bit KV**：`GUFO_KV_Q4`（4-bit key）、`GUFO_KV_V4`（4-bit value pack 18B）。
- **Mixed batch**：P0–P3（envelope 32 sessions/1024 rows、MTP parity、mixed 預設開）。
- **MTP**：draft depth、ngram 預取、取樣開銷優化。
- KV 唯一推薦 `q8`（`--kv-quant fp8` 已移除；gfx1151 無 fp8 WMMA，且 Q8 + Q4_1 副本已達 +17% decode）。

## 環境開關速查（config A 預設全開）
`GUFO_KV_WMMA=1`、`GUFO_KV_I8=1`、`GUFO_KV_Q4=1`、`GUFO_KV_V4=1`；
`GUFO_DECODE_Q4/Q5` 控制 decode 副本；`GUFO_PREFILL_TILE` 控制 prefill 切塊；
`GUFO_MIXED_BATCH=0` 關 mixed；MTP `-d 5`（實測 `-d 7` 無益，loader 拒 `-d 9`）。
