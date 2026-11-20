# MBUFO NPU 五路 — 代碼框 v0（搭好未接線，一小時後一次測）

> 原則：只加新檔、不改舊檔、不改 CMake、不編譯、不跑服務。
> 主機佔用中，所以現在只搭框；接線 + 測試等主機空出來一次做。

## 框在哪（.55 上）

| 框 | 路徑 | 說明 |
|---|---|---|
| 中央開關 | `src/core/compute/npu/npu_lane.hpp` | 五路開關 + 後端枚舉，預設全關 |
| embedding 縫 | `src/models/bge_m3/npu_embed.hpp` | `DecideNpuEmbed()` 樁，永遠 kNone |
| reranker 縫 | `src/models/bge_reranker/npu_score.hpp` | `DecideNpuScore()` 樁，永遠 kNone |
| decide 定義 | `src/models/d3_decide/d3_config.hpp` | D3 家族 + NPU2 對照（修正：decide=D3） |
| sidecar 對照 | `src/cli/serve/npu_sidecar_routes.hpp` | decide/asr/tts 預設埠 + 路徑 |

## 一小時後測試（一次跑完，不要分段打擾主機）

1. `git status --short` 確認只有 6 個新檔（含本 md），舊檔零修改。
2. 輕量語法：`g++ -std=c++20 -fsyntax-only` 逐個過 header（不用整體 build）。
3. decide：`flm pull` D3-Lite-NPU2（~776MB）→ `flm serve --decision d3:lite --port 8766` → `POST /v1/systemone` 抓包對 `DecideRequest` 欄位。
4. asr/tts 回放：沿用錨點（`Front_Center.wav` / voice-clone）驗 sidecar 仍活著。
5. embedding/reranker：仍走 HIP（樁回 kNone），只確認行為零變化。

## 下一步（接線時才做，不在本框）

- `EngineSession::Sync` 前查 `DecideNpuEmbed`；`core/compute` dispatcher 加 `kNpu`。
- router 加三條 upstream（8766/8764/8765），WebUI 加後端選項。
- decide 接線後，qwen3.8-flash-next 的 `task=decide` 標籤要改名，避免再混淆。
