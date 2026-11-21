// HTTP reverse proxy forwarding to NPU/GPU sidecar processes
// (/v1/systemone -> D3, /v1/audio/transcriptions -> Whisper ASR,
//  /v1/audio/speech & /v1/audio/voices -> Qwen3-TTS).
#ifndef MBUFO_SERVER_NPU_SIDECAR_PROXY_HPP_
#define MBUFO_SERVER_NPU_SIDECAR_PROXY_HPP_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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

// ---------------------------------------------------------------------------
// Sidecar lifecycle: spawn / probe / kill sidecar processes from /admin.
// A sidecar is NOT a GGUF in the registry — it is an independent OS process
// (flm, or gufo itself for TTS) managed here. The admin UI renders the same
// Load / Eject buttons as GPU models; Load spawns, Eject SIGTERMs.
// Never auto-started at boot; on-demand only, like everything else here.
// ---------------------------------------------------------------------------

/// Static launch descriptor. argv is executed directly (no shell).
struct SidecarLaunchSpec {
  std::string id;                     // catalog id, e.g. "whisper-turbo-npu"
  std::string display_name;         // card title, e.g. "D3-Lite-NPU2 (flm sidecar)"
  std::string backend;              // "npu" or "gpu" (UI badge)
  std::string task;                 // "decide" | "asr" | "tts"
  std::string capability;           // badge: "DECIDE" | "AUDIO"
  int port{0};                      // loopback port
  std::string base_url;             // e.g. http://127.0.0.1:8764
  std::string health_path;          // readiness probe path
  std::string endpoint;             // main API path (UI hint)
  std::size_t estimated_resident_mib{0};  // memory bar + UI size
  // argv builders need runtime env (flm path, TTS model dir), so the
  // table stores a kind tag and argv is materialized per Start() call.
  enum class Kind { kFlmDecision, kFlmAsr, kGufoTts } kind;
  std::string flm_model;  // e.g. "llama3.2:1b" (decision chat host) — see below
  std::string flm_flag;   // e.g. "d3:lite" / "whisper-v3:turbo"
};

/// The 7 known sidecars. Single source of truth for the admin catalog
/// entries, the loader, and the UI.
[[nodiscard]] const std::vector<SidecarLaunchSpec>& KnownSidecars() noexcept;
[[nodiscard]] const SidecarLaunchSpec* FindSidecarSpec(
    std::string_view id) noexcept;

/// Human-readable launch command for display/copy (same argv Start uses).
[[nodiscard]] std::string SidecarLaunchDisplay(const SidecarLaunchSpec& spec);

class SidecarManager {
 public:
  struct Status {
    bool running{false};
    bool external{false};  // alive but not our child (adopted or untracked)
    long pid{-1};          // -1 when unknown
    std::string base_url;
  };

  SidecarManager() = default;
  SidecarManager(const SidecarManager&) = delete;
  SidecarManager& operator=(const SidecarManager&) = delete;

  // Spawn (or adopt, if already listening) the sidecar. Blocks until the
  // health endpoint answers or timeout_ms elapses. Idempotent.
  [[nodiscard]] bool Start(const std::string& id, std::string* error,
                           long timeout_ms = 180000);
  // SIGTERM, then SIGKILL after a grace period. Refuses untracked
  // externals (no pid) with an explanatory error.
  [[nodiscard]] bool Stop(const std::string& id, std::string* error);
  [[nodiscard]] bool IsRunning(const std::string& id);
  [[nodiscard]] std::vector<Status> ListRunning();
  [[nodiscard]] Status GetStatus(const std::string& id);

 private:
  struct Entry {
    long pid{-1};
    bool external{false};
    std::string base_url;
  };
  std::mutex mu_;
  std::unordered_map<std::string, Entry> table_;
};

/// Process-wide manager used by the admin handlers.
SidecarManager& GlobalSidecarManager();

}  // namespace mbufo::server

#endif  // MBUFO_SERVER_NPU_SIDECAR_PROXY_HPP_
