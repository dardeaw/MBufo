# MBufo：為 Strix Halo 而生的高併發推論引擎

> MBufo 是 [gufo](https://github.com/gufo-org/gufo) 的一個專注高併發方向的分支（branch `r151-mixed`），目標是第一款真正為 AMD Strix Halo（Ryzen AI MAX+ 395 / Radeon 8060S / 128GB 統一記憶體）調校的高併發本地推論引擎。上游 gufo 負責單路極限效能，MBufo 負責多路併發打滿算力。底層模型、核心與絕大部分程式碼均來自上游開源專案，致謝 gufo-org 及所有貢獻者。
>
> [English version](README.md)

## MBufo 帶來了什麼

- **混合批次注意力（mixed-batch attention）**：把 1 個 decode 列和 N 個 prefill-chunk 列裝進一次 `ForwardBatch`，用一個 `WmmaMixedAttention` 核心代替逐項 attention 核心，消除 decode/prefill 輪流空轉（小包 2.1 倍，服務層實測開 5.3 秒 vs 關 6.3 秒）。
- **32 路包絡**：`ForwardBatch` 從 8 session/8 列放大到 32 session/1024 列（附帶修了 GDN、MoE、logits 緩衝三個硬上限），16 路併發實測 GPU 使用率 **99%**，輸出全對、零錯誤。
- **MTP 對等**：chunk 進入 batch 自動補 predictor-KV（frontier 檢查 + tail），MTP session 行為與單路一致；speculative 列自動降級走舊核心（正確但不加速）。
- **Serve 併單（P3）**：排程器把 prefill chunk 和一般 decode 合成一批（`TryStepMixedBatch`）。每一層都可獨立降級，任何一步拒絕都走今天的老路。

## 快速開始（相對上游的增量）

```bash
# mixed 預設開啟；關掉回到上游行為：
GUFO_MIXED_BATCH=0 gufo serve llm ... &
# 高併發建議（16 路約佔 99GB；32K context 控制 KV 大小）：
gufo serve llm --sessions 16 --context 32768 ... &
```

- 設計文件（含三條鐵律、驗證矩陣、效能數據）：[docs/MIXED_BATCH.md](docs/MIXED_BATCH.md)
- 上游相關 PR（Q8 KV cache #561、int8 WMMA #566、persist #579、history drafting #608）：見上游倉庫，本分支已包含對應實作。

---

*English 為主文件，見 [README.md](README.md)。*
