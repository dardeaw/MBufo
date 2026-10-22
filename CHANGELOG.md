# MBufo Changelog（Qwen3.8-Flash-Next 推論分支）

以 `gufo-org/gufo` 上游為基底，專注低精度 KV、decode 頻寬與 MTP 推測解碼。
格式：`version.txt` 即版號；只推 fork（`dardeaw/MBufo`），不上游。

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
- **FP8 KV**：`--kv-quant fp8` opt-in（gfx1151 無 fp8 WMMA，僅快取格式）。

## 環境開關速查（config A 預設全開）
`GUFO_KV_WMMA=1`、`GUFO_KV_I8=1`、`GUFO_KV_Q4=1`、`GUFO_KV_V4=1`；
`GUFO_DECODE_Q4/Q5` 控制 decode 副本；`GUFO_PREFILL_TILE` 控制 prefill 切塊；
`GUFO_MIXED_BATCH=0` 關 mixed；MTP `-d 5`（實測 `-d 7` 無益，loader 拒 `-d 9`）。
