#pragma once
// NPU lane 中央抽象（代碼框 v1：六路，in-process servable 描述子）。
//
// 六路目標：embedding / reranker / decide / ASR / TTS / image。
// 策略：
//   - embedding + reranker：in-process，`src/core/compute` dispatcher 多一條
//     kNpu route（仿 gemm_dispatcher.hpp DecidePrefillRoute），EngineSession::Sync
//     之前查一次。NPU kernel（IRON 自建）還沒好之前一律回 kNone = 走現有 HIP。
//   - decide / ASR / TTS：sidecar + router upstream（flm / gufo serve tts/asr
//     獨立進程），見 npu_sidecar_routes.hpp。D3 decide 模型見
//     src/models/d3_decide/。
//   - ASR / TTS / image in-process servable（本分支
//     npu/unified-servable）：ThinkerNpuDecoder / TalkerNpuDecoder +
//     PredictorNpuDecoder + TtsNpuFrameLoop / TeNpuDecoder / FluxDitNpu /
//     ZImageDitNpu，全部 NPU-GEMM + host 數值。router seam 見
//     models/{qwen3_asr,qwen3_tts}/npu/router.hpp 與 models/image_npu/router.hpp。
//     預設全關，按需 on-demand 載入，絕不進 boot autoload。
//
// 一小時後測試步驟見 docs/NPU_LANES.md。

#include <cstdint>
#include <string>

namespace mbufo::npu {

/// NPU 可用性來源。flm sidecar（whisper/D3-NPU2）與 in-process NPU 是兩條獨立路。
enum class NpuBackend : std::uint8_t {
  kNone = 0,    ///< 不走 NPU（預設，行為與今天完全一致）
  kFlmSidecar,  ///< flm 獨立進程（decide-D3 / ASR-whisper），HTTP 轉發
  kGufoSidecar, ///< gufo serve tts/asr 獨立進程，HTTP 轉發
  kIron,        ///< in-process IRON/AIE（embedding/reranker 未來式）
  kNpuInProcess,///< in-process NPU-GEMM servable（本分支：ASR/TTS/image）
};

/// 六路開關：預設全關。router 按需打開，不開機自啟。
struct NpuLaneConfig {
  bool embedding = false;
  bool reranker = false;
  bool decide = false;
  bool asr = false;
  bool tts = false;
  bool image = false;
  // sidecar 位址（router upstream 用，預設埠沿用手測值，可改）
  std::string decide_url;  // e.g. http://127.0.0.1:8766/v1/systemone
  std::string asr_url;     // e.g. http://127.0.0.1:8764/v1/audio/transcriptions
  std::string tts_url;     // e.g. http://127.0.0.1:8765/v1/audio/speech
  // in-process servable 權重根目錄（空 = 走 sidecar 路，不動現行行為）
  std::string asr_model_root;    // e.g. /mnt/nvme-data/mbufo-models/qwen3-asr-1.7b
  std::string tts_model_root;    // e.g. /mnt/nvme-data/mbufo-models/qwen3-tts-1.7b-base
  std::string image_model_root;  // e.g. .../flux2-klein-4b (TE+DiT 同根)
};

/// 輕量探針：只做 HTTP /models 或端口可達檢查，不載模型、不佔 NPU。
/// 實作時放在 .cpp（需要 curl/HTTP client），此處僅宣告。
[[nodiscard]] bool ProbeSidecar(const std::string& base_url) noexcept;

}  // namespace mbufo::npu
