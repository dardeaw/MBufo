#pragma once
// sidecar + router upstream 對照（代碼框 v0，唯讀、無副作用）。
//
// 為什麼 sidecar：D3 decide / Whisper ASR / Qwen3-TTS 都不是 in-process GGUF
// 卡（`:8741` admin catalog 只收 GGUF），分別是 flm 進程與 gufo serve tts/asr
// 進程。router 側加 upstream 轉發即可，不動 catalog 機制。不開機自啟，router
// 按需載入（on-demand 拉起，用完釋放）。
//
// 預設埠（沿用手測值，可改）：
//   decide(D3)  http://127.0.0.1:8766  POST /v1/systemone        (flm --decision)
//   asr        http://127.0.0.1:8764  POST /v1/audio/transcriptions (flm whisper)
//   tts        http://127.0.0.1:8765  POST /v1/audio/speech      (gufo serve tts)
//   voices     http://127.0.0.1:8765  GET  /v1/audio/voices
//
// 實測錨點（已驗證，回放用）：
//   ASR: 1.42s wav -> "Front, center." (~2.3s)
//   TTS: voice-clone + reference_audio(b64) + reference_text -> 24kHz wav，
//        synthesis ~31s / 7.04s 音訊；ASR 回讀中英對照幾乎全對。
//   TTS 本體 RSS ~4.2GB（118GB 是整機共享池快照，不是 TTS 體積）。

#include <string_view>

namespace mbufo::server::npu {

[[nodiscard]] inline constexpr std::string_view DecideSidecarDefault() noexcept {
  return "http://127.0.0.1:8766";
}
[[nodiscard]] inline constexpr std::string_view AsrSidecarDefault() noexcept {
  return "http://127.0.0.1:8764";
}
[[nodiscard]] inline constexpr std::string_view TtsSidecarDefault() noexcept {
  return "http://127.0.0.1:8765";
}
[[nodiscard]] inline constexpr std::string_view SystemOnePath() noexcept {
  return "/v1/systemone";
}
[[nodiscard]] inline constexpr std::string_view TranscriptionsPath() noexcept {
  return "/v1/audio/transcriptions";
}
[[nodiscard]] inline constexpr std::string_view SpeechPath() noexcept {
  return "/v1/audio/speech";
}
[[nodiscard]] inline constexpr std::string_view VoicesPath() noexcept {
  return "/v1/audio/voices";
}
[[nodiscard]] inline constexpr std::string_view Flux2ImageSidecarDefault() noexcept {
  return "http://127.0.0.1:8768";
}
[[nodiscard]] inline constexpr std::string_view ZImageSidecarDefault() noexcept {
  return "http://127.0.0.1:8769";
}
[[nodiscard]] inline constexpr std::string_view ImagesGenerationsPath() noexcept {
  return "/v1/images/generations";
}
[[nodiscard]] inline constexpr std::string_view ImagesEditsPath() noexcept {
  return "/v1/images/edits";
}

}  // namespace mbufo::server::npu
