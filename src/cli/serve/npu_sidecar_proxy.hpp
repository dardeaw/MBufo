// HTTP reverse proxy forwarding to NPU/GPU sidecar processes
// (/v1/systemone -> D3, /v1/audio/transcriptions -> Whisper ASR,
//  /v1/audio/speech & /v1/audio/voices -> Qwen3-TTS).
#ifndef MBUFO_SERVER_NPU_SIDECAR_PROXY_HPP_
#define MBUFO_SERVER_NPU_SIDECAR_PROXY_HPP_

#include <string>
#include <string_view>

#include "src/cli/serve/npu_sidecar_routes.hpp"

namespace mbufo::server {

struct HttpRequest;
struct HttpResponse;

struct SidecarProxyConfig {
  bool enabled{false};
  std::string decide_base_url{std::string(npu::DecideSidecarDefault())};
  std::string asr_base_url{std::string(npu::AsrSidecarDefault())};
  std::string tts_base_url{std::string(npu::TtsSidecarDefault())};
  long connect_timeout_ms{2000};
  long request_timeout_ms{120000};

  /// Reads `MBUFO_SIDECAR_PROXY`, `MBUFO_DECIDE_UPSTREAM`,
  /// `MBUFO_ASR_UPSTREAM`, and `MBUFO_TTS_UPSTREAM`.
  [[nodiscard]] static SidecarProxyConfig FromEnvironment(
      bool default_enabled = false);
};

/// Forwards `req` synchronously to `upstream_base_url` preserving method,
/// query string, safe headers, raw request bytes, response status, content-type,
/// and binary/JSON response payload. Returns 502 Bad Gateway on transport
/// failure or 400 on invalid upstream URL scheme.
[[nodiscard]] HttpResponse ForwardSidecarRequest(
    const HttpRequest& req, std::string_view upstream_base_url,
    long connect_timeout_ms = 2000, long request_timeout_ms = 120000);

}  // namespace mbufo::server

#endif  // MBUFO_SERVER_NPU_SIDECAR_PROXY_HPP_
