#include "src/cli/serve/npu_sidecar_proxy.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/core/compute/npu/npu_lane.hpp"
#include "src/core/json.hpp"

namespace mbufo::server {
namespace {

std::string ToLowerAscii(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

std::string TrimOws(std::string_view s) {
  while (!s.empty() &&
         (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' ||
          s.front() == '\n')) {
    s.remove_prefix(1);
  }
  while (!s.empty() &&
         (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' ||
          s.back() == '\n')) {
    s.remove_suffix(1);
  }
  return std::string(s);
}

void EnsureCurlGlobalInit() {
  static std::once_flag once;
  std::call_once(once, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

struct CurlDeleter {
  void operator()(CURL* h) const noexcept {
    if (h != nullptr) curl_easy_cleanup(h);
  }
};
using CurlPtr = std::unique_ptr<CURL, CurlDeleter>;

struct SlistDeleter {
  void operator()(curl_slist* l) const noexcept {
    if (l != nullptr) curl_slist_free_all(l);
  }
};
using SlistPtr = std::unique_ptr<curl_slist, SlistDeleter>;

bool IsValidHttpBase(std::string_view url) {
  return url.starts_with("http://") || url.starts_with("https://");
}

std::string BuildUpstreamUrl(std::string_view base, std::string_view path,
                             std::string_view query) {
  std::string out(base);
  while (!out.empty() && out.back() == '/') out.pop_back();
  if (!path.empty() && path.front() != '/') out.push_back('/');
  out.append(path);
  if (!query.empty()) {
    out.push_back('?');
    out.append(query);
  }
  return out;
}

size_t WriteBodyCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
  const size_t total = size * nmemb;
  auto* out = static_cast<std::string*>(userdata);
  out->append(ptr, total);
  return total;
}

struct HeaderCapture {
  std::string reason{"OK"};
  std::string content_type;
};

size_t WriteHeaderCallback(char* ptr, size_t size, size_t nmemb,
                           void* userdata) {
  const size_t total = size * nmemb;
  auto* cap = static_cast<HeaderCapture*>(userdata);
  const std::string_view line(ptr, total);
  if (line.starts_with("HTTP/")) {
    const size_t sp1 = line.find(' ');
    if (sp1 != std::string_view::npos) {
      const size_t sp2 = line.find(' ', sp1 + 1);
      if (sp2 != std::string_view::npos) {
        const std::string r = TrimOws(line.substr(sp2 + 1));
        if (!r.empty()) cap->reason = r;
      }
    }
    return total;
  }
  const size_t colon = line.find(':');
  if (colon != std::string_view::npos) {
    const std::string key = ToLowerAscii(TrimOws(line.substr(0, colon)));
    const std::string val = TrimOws(line.substr(colon + 1));
    if (key == "content-type" && !val.empty()) {
      cap->content_type = val;
    }
  }
  return total;
}

HttpResponse ProxyError(int status, const std::string& reason,
                        const std::string& message, const std::string& code) {
  json::Value err_obj = json::Value::object();
  err_obj["message"] = message;
  err_obj["type"] = "server_error";
  err_obj["code"] = code;
  json::Value root = json::Value::object();
  root["error"] = std::move(err_obj);
  return HttpResponse{
      .status = status,
      .reason = reason,
      .body = root.dump(),
      .headers = {{"Content-Type", "application/json"}},
  };
}

}  // namespace

SidecarProxyConfig SidecarProxyConfig::FromEnvironment(bool default_enabled) {
  SidecarProxyConfig cfg;
  cfg.enabled = default_enabled;
  if (const char* v = std::getenv("MBUFO_SIDECAR_PROXY")) {
    const std::string lv = ToLowerAscii(TrimOws(v));
    if (lv == "0" || lv == "false" || lv == "off" || lv == "no") {
      cfg.enabled = false;
    } else if (lv == "1" || lv == "true" || lv == "on" || lv == "yes") {
      cfg.enabled = true;
    }
  }
  if (const char* u = std::getenv("MBUFO_DECIDE_UPSTREAM");
      u != nullptr && *u != '\0') {
    cfg.decide_base_url = u;
  }
  if (const char* u = std::getenv("MBUFO_ASR_UPSTREAM");
      u != nullptr && *u != '\0') {
    cfg.asr_base_url = u;
  }
  if (const char* u = std::getenv("MBUFO_TTS_UPSTREAM");
      u != nullptr && *u != '\0') {
    cfg.tts_base_url = u;
  }
  return cfg;
}

HttpResponse ForwardSidecarRequest(const HttpRequest& req,
                                   std::string_view upstream_base_url,
                                   long connect_timeout_ms,
                                   long request_timeout_ms) {
  if (!IsValidHttpBase(upstream_base_url)) {
    return ProxyError(503, "Service Unavailable",
                      "sidecar upstream URL must start with http:// or https://",
                      "sidecar_invalid_upstream");
  }
  EnsureCurlGlobalInit();
  const CurlPtr curl(curl_easy_init());
  if (!curl) {
    return ProxyError(502, "Bad Gateway", "failed to initialize libcurl",
                      "sidecar_curl_init_failed");
  }

  const std::string target_url =
      BuildUpstreamUrl(upstream_base_url, req.path, req.query);

  std::string resp_body;
  HeaderCapture hdr_cap;

  curl_easy_setopt(curl.get(), CURLOPT_URL, target_url.c_str());
  curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl.get(), CURLOPT_PROXY, "");
  curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "http,https");
  curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
  curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS, connect_timeout_ms);
  curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS, request_timeout_ms);
  curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, WriteBodyCallback);
  curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &resp_body);
  curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, WriteHeaderCallback);
  curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, &hdr_cap);

  if (req.method == "GET") {
    curl_easy_setopt(curl.get(), CURLOPT_HTTPGET, 1L);
  } else if (req.method == "POST") {
    curl_easy_setopt(curl.get(), CURLOPT_POST, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS, req.body.data());
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE_LARGE,
                     static_cast<curl_off_t>(req.body.size()));
  } else {
    curl_easy_setopt(curl.get(), CURLOPT_CUSTOMREQUEST, req.method.c_str());
    if (!req.body.empty()) {
      curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS, req.body.data());
      curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE_LARGE,
                       static_cast<curl_off_t>(req.body.size()));
    }
  }

  curl_slist* raw_headers = nullptr;
  // Disable 100-continue stalls for local sidecars.
  raw_headers = curl_slist_append(raw_headers, "Expect:");
  for (const auto& [name, value] : req.headers) {
    const std::string lower = ToLowerAscii(name);
    if (lower == "content-type" || lower == "accept" ||
        lower == "x-request-id") {
      const std::string line = name + ": " + value;
      raw_headers = curl_slist_append(raw_headers, line.c_str());
    }
  }
  const SlistPtr headers_guard(raw_headers);
  if (headers_guard) {
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers_guard.get());
  }

  const CURLcode rc = curl_easy_perform(curl.get());
  if (rc != CURLE_OK) {
    return ProxyError(
        502, "Bad Gateway",
        std::string("sidecar upstream unreachable (") + target_url +
            "): " + curl_easy_strerror(rc),
        "sidecar_upstream_unavailable");
  }

  long http_code = 0;
  curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);
  HttpResponse resp;
  resp.status = (http_code > 0) ? static_cast<int>(http_code) : 200;
  resp.reason = hdr_cap.reason;
  resp.body = std::move(resp_body);
  if (!hdr_cap.content_type.empty()) {
    resp.headers.emplace_back("Content-Type", hdr_cap.content_type);
  }
  resp.headers.emplace_back("X-Sidecar-Upstream", target_url);
  return resp;
}

namespace {

// ---- SidecarManager internals (POSIX only; header-only API above) ----

bool ProbeUrl(const std::string& base_url, const std::string& path,
              long timeout_ms) {
  try {
    if (!base_url.starts_with("http://") &&
        !base_url.starts_with("https://")) {
      return false;
    }
    EnsureCurlGlobalInit();
    const CurlPtr curl(curl_easy_init());
    if (!curl) return false;
    std::string url = base_url;
    while (!url.empty() && url.back() == '/') url.pop_back();
    url += path;
    std::string sink;
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_PROXY, "");
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS, 500L);
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS, timeout_ms);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, WriteBodyCallback);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &sink);
    if (curl_easy_perform(curl.get()) != CURLE_OK) return false;
    long code = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &code);
    return code >= 200 && code < 500;
  } catch (...) {
    return false;
  }
}

bool PidAlive(long pid) {
  if (pid <= 0) return false;
  if (::kill(static_cast<pid_t>(pid), 0) == 0) return true;
  return errno != ESRCH;
}

// Best-effort lookup of a sidecar pid by matching its argv tokens
// (e.g. {"flm", "whisper-v3:turbo", "--port", "8764"}). Returns -1.
long FindPidByArgv(const std::vector<std::string>& needles) {
  namespace fs = std::filesystem;
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator("/proc", ec)) {
    if (ec) break;
    const std::string name = entry.path().filename().string();
    if (name.empty() ||
        !std::all_of(name.begin(), name.end(),
                     [](unsigned char c) { return std::isdigit(c); })) {
      continue;
    }
    std::ifstream cmd("/proc/" + name + "/cmdline", std::ios::binary);
    if (!cmd) continue;
    std::string blob((std::istreambuf_iterator<char>(cmd)),
                     std::istreambuf_iterator<char>());
    if (blob.empty()) continue;
    std::string flat = blob;
    for (char& c : flat) {
      if (c == '\0') c = ' ';
    }
    bool ok = true;
    for (const auto& n : needles) {
      if (n.empty()) continue;
      if (flat.find(n) == std::string::npos) {
        ok = false;
        break;
      }
    }
    if (ok) {
      try {
        return std::stol(name);
      } catch (...) {
      }
    }
  }
  return -1;
}

std::string ResolveFlmBinary() {
  if (const char* e = std::getenv("MBUFO_FLM_BIN");
      e != nullptr && *e != '\0') {
    return e;
  }
  // PATH lookup: walk PATH manually so behavior is explicit and logged.
  const char* path = std::getenv("PATH");
  const std::string dirs = (path != nullptr) ? path : "/usr/bin:/bin";
  std::string cur;
  for (std::size_t i = 0; i <= dirs.size(); ++i) {
    if (i == dirs.size() || dirs[i] == ':') {
      if (!cur.empty()) {
        const std::string cand = cur + "/flm";
        if (::access(cand.c_str(), X_OK) == 0) return cand;
        cur.clear();
      }
    } else {
      cur.push_back(dirs[i]);
    }
  }
  return "flm";  // last resort: let execvp report the error into the log
}

std::string SelfExePath() {
  char buf[4096] = {0};
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n > 0) return std::string(buf, static_cast<std::size_t>(n));
  return "mbufo";
}

std::string TtsModelDir() {
  if (const char* e = std::getenv("MBUFO_TTS_MODEL_DIR");
      e != nullptr && *e != '\0') {
    return e;
  }
  return "/mnt/nvme-data/mbufo-models/qwen3-tts-1.7b-base";
}

std::string SidecarLogPath(const std::string& id) {
  const char* d = std::getenv("MBUFO_SIDECAR_LOG_DIR");
  const std::string dir =
      (d != nullptr && *d != '\0') ? d : "/tmp/mbufo-sidecars";
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  return dir + "/" + id + ".log";
}

// Build argv (no shell anywhere). Returns false when misconfigured.
bool BuildArgv(const SidecarLaunchSpec& spec, std::vector<std::string>* out,
               std::string* error) {
  out->clear();
  const std::string port = std::to_string(spec.port);
  switch (spec.kind) {
    case SidecarLaunchSpec::Kind::kFlmDecision:
      *out = {ResolveFlmBinary(), "serve", spec.flm_model, "--decision",
              spec.flm_flag, "--port", port};
      break;
    case SidecarLaunchSpec::Kind::kFlmAsr:
      *out = {ResolveFlmBinary(), "serve", spec.flm_flag, "--asr", "1",
              "--port", port};
      break;
    case SidecarLaunchSpec::Kind::kGufoTts:
      *out = {SelfExePath(), "serve", "tts", "--model", TtsModelDir(),
              "--port", port};
      break;
  }
  if (out->empty() || (*out)[0].empty()) {
    if (error != nullptr) *error = "cannot resolve launcher binary";
    return false;
  }
  return true;
}

}  // namespace

const std::vector<SidecarLaunchSpec>& KnownSidecars() noexcept {
  static const std::vector<SidecarLaunchSpec> kSpecs = {
      {"d3-lite-npu", "D3-Lite-NPU2 (flm sidecar)", "npu", "decide", "DECIDE", 8766,
       "http://127.0.0.1:8766", "/v1/models", "/v1/systemone", 900,
       SidecarLaunchSpec::Kind::kFlmDecision, "llama3.2:1b", "d3:lite"},
      {"d3-nano-npu", "D3-Nano-NPU2 (flm sidecar)", "npu", "decide", "DECIDE", 8766,
       "http://127.0.0.1:8766", "/v1/models", "/v1/systemone", 1800,
       SidecarLaunchSpec::Kind::kFlmDecision, "llama3.2:1b", "d3:nano"},
      {"d3-mini-npu", "D3-Mini-NPU2 (flm sidecar)", "npu", "decide", "DECIDE", 8766,
       "http://127.0.0.1:8766", "/v1/models", "/v1/systemone", 3200,
       SidecarLaunchSpec::Kind::kFlmDecision, "llama3.2:1b", "d3:mini"},
      {"d3-turbo-npu", "D3-Turbo-NPU2 (flm sidecar)", "npu", "decide", "DECIDE", 8766,
       "http://127.0.0.1:8766", "/v1/models", "/v1/systemone", 6400,
       SidecarLaunchSpec::Kind::kFlmDecision, "llama3.2:1b", "d3:turbo"},
      {"d3-ultra-npu", "D3-Ultra-NPU2 (flm sidecar)", "npu", "decide", "DECIDE", 8766,
       "http://127.0.0.1:8766", "/v1/models", "/v1/systemone", 12000,
       SidecarLaunchSpec::Kind::kFlmDecision, "llama3.2:1b", "d3:ultra"},
      {"whisper-turbo-npu", "Whisper-V3-Turbo-NPU2 (flm sidecar)", "npu", "asr", "AUDIO", 8764,
       "http://127.0.0.1:8764", "/v1/models", "/v1/audio/transcriptions", 700,
       SidecarLaunchSpec::Kind::kFlmAsr, "", "whisper-v3:turbo"},
      {"qwen3-tts-1.7b", "Qwen3-TTS-12Hz-1.7B-Base (gufo sidecar)", "gpu", "tts", "AUDIO", 8765,
       "http://127.0.0.1:8765", "/v1/audio/voices", "/v1/audio/speech", 4700,
       SidecarLaunchSpec::Kind::kGufoTts, "", ""},
  };
  return kSpecs;
}

const SidecarLaunchSpec* FindSidecarSpec(std::string_view id) noexcept {
  for (const auto& s : KnownSidecars()) {
    if (s.id == id) return &s;
  }
  return nullptr;
}

std::string SidecarLaunchDisplay(const SidecarLaunchSpec& spec) {
  // Same argv Start() would exec (env-resolved); joined for display only.
  std::vector<std::string> args;
  if (!BuildArgv(spec, &args, nullptr)) return std::string{};
  std::string out;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (i != 0) out.push_back(' ');
    const bool sp = args[i].find(' ') != std::string::npos;
    if (sp) out.push_back('"');
    out += args[i];
    if (sp) out.push_back('"');
  }
  return out;
}

SidecarManager& GlobalSidecarManager() {
  static SidecarManager mgr;
  return mgr;
}

bool SidecarManager::IsRunning(const std::string& id) {
  const SidecarLaunchSpec* spec = FindSidecarSpec(id);
  if (spec == nullptr) return false;
  std::lock_guard<std::mutex> lock(mu_);
  auto it = table_.find(id);
  if (it != table_.end() && it->second.pid > 0) {
    if (PidAlive(it->second.pid)) return true;
    // Reap our own dead child to avoid zombies, then drop the record.
    int status = 0;
    ::waitpid(static_cast<pid_t>(it->second.pid), &status, WNOHANG);
    table_.erase(it);
  }
  return false;
}

SidecarManager::Status SidecarManager::GetStatus(const std::string& id) {
  Status st;
  const SidecarLaunchSpec* spec = FindSidecarSpec(id);
  if (spec == nullptr) return st;
  st.base_url = spec->base_url;
  std::lock_guard<std::mutex> lock(mu_);
  auto it = table_.find(id);
  if (it != table_.end() && it->second.pid > 0 && PidAlive(it->second.pid)) {
    st.running = true;
    st.external = it->second.external;
    st.pid = it->second.pid;
  }
  return st;
}

std::vector<SidecarManager::Status> SidecarManager::ListRunning() {
  std::vector<Status> out;
  for (const auto& spec : KnownSidecars()) {
    Status st = GetStatus(spec.id);
    if (st.running) out.push_back(st);
  }
  return out;
}

bool SidecarManager::Start(const std::string& id, std::string* error,
                           long timeout_ms) {
  const SidecarLaunchSpec* spec = FindSidecarSpec(id);
  if (spec == nullptr) {
    if (error != nullptr) *error = "unknown sidecar id: " + id;
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(id);
    if (it != table_.end() && it->second.pid > 0 &&
        PidAlive(it->second.pid)) {
      return true;  // idempotent: already ours
    }
    if (it != table_.end()) table_.erase(it);
  }
  // Someone else (manual launch or env-overridden upstream) may already
  // serve this port: adopt instead of double-binding.
  if (ProbeUrl(spec->base_url, spec->health_path, 1500)) {
    std::vector<std::string> needles;
    switch (spec->kind) {
      case SidecarLaunchSpec::Kind::kFlmDecision:
        needles = {"flm", spec->flm_flag, "--port",
                   std::to_string(spec->port)};
        break;
      case SidecarLaunchSpec::Kind::kFlmAsr:
        needles = {"flm", spec->flm_flag, "--port",
                   std::to_string(spec->port)};
        break;
      case SidecarLaunchSpec::Kind::kGufoTts:
        needles = {"serve", "tts", "--port", std::to_string(spec->port)};
        break;
    }
    const long pid = FindPidByArgv(needles);
    std::lock_guard<std::mutex> lock(mu_);
    table_[id] = Entry{pid, true, spec->base_url};
    return true;
  }
  std::vector<std::string> args;
  if (!BuildArgv(*spec, &args, error)) return false;
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);

  const std::string log_path = SidecarLogPath(id);
  const int log_fd =
      ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (log_fd < 0) {
    if (error != nullptr) {
      *error = "cannot open sidecar log " + log_path + ": " +
               std::strerror(errno);
    }
    return false;
  }
  const pid_t child = ::fork();
  if (child < 0) {
    const int e = errno;
    ::close(log_fd);
    if (error != nullptr) {
      *error = std::string("fork failed: ") + std::strerror(e);
    }
    return false;
  }
  if (child == 0) {
    // Child: async-signal-safe calls only until exec.
    ::setsid();
    ::signal(SIGPIPE, SIG_DFL);
    ::signal(SIGTERM, SIG_DFL);
    if (::dup2(log_fd, STDOUT_FILENO) < 0) ::_exit(127);
    if (::dup2(log_fd, STDERR_FILENO) < 0) ::_exit(127);
    for (int fd = 3; fd < 4096; ++fd) {
      if (fd != log_fd) ::close(fd);
    }
    ::execvp(argv[0], argv.data());
    ::_exit(127);  // exec failed; reason is in the log file
  }
  ::close(log_fd);
  // Health-gate: wait until the sidecar answers, else kill the child so a
  // bad launch (bad binary, busy port) never lingers headless.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  bool ready = false;
  while (std::chrono::steady_clock::now() < deadline) {
    int status = 0;
    const pid_t r =
        ::waitpid(child, &status, WNOHANG);  // NOLINT: reap on early exit
    if (r == child) {
      if (error != nullptr) {
        *error = "sidecar '" + id + "' exited during startup (code " +
                 std::to_string(status) + "); see " + log_path;
      }
      return false;
    }
    if (ProbeUrl(spec->base_url, spec->health_path, 2000)) {
      ready = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  if (!ready) {
    ::kill(child, SIGTERM);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    int status = 0;
    if (::waitpid(child, &status, WNOHANG) == 0) {
      ::kill(child, SIGKILL);
      ::waitpid(child, &status, 0);
    }
    if (error != nullptr) {
      *error = "sidecar '" + id + "' did not answer " + spec->base_url +
               spec->health_path + " in time; see " + log_path;
    }
    return false;
  }
  std::lock_guard<std::mutex> lock(mu_);
  table_[id] = Entry{static_cast<long>(child), false, spec->base_url};
  return true;
}

bool SidecarManager::Stop(const std::string& id, std::string* error) {
  const SidecarLaunchSpec* spec = FindSidecarSpec(id);
  if (spec == nullptr) {
    if (error != nullptr) *error = "unknown sidecar id: " + id;
    return false;
  }
  long pid = -1;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(id);
    if (it == table_.end() || it->second.pid <= 0 ||
        !PidAlive(it->second.pid)) {
      if (it != table_.end()) {
        int status = 0;
        if (it->second.pid > 0) {
          ::waitpid(static_cast<pid_t>(it->second.pid), &status, WNOHANG);
        }
        table_.erase(it);
      }
      if (error != nullptr) *error = "sidecar '" + id + "' is not running";
      return false;
    }
    pid = it->second.pid;
  }
  ::kill(static_cast<pid_t>(pid), SIGTERM);
  bool gone = false;
  for (int i = 0; i < 150; ++i) {  // ~15 s grace
    int status = 0;
    if (::waitpid(static_cast<pid_t>(pid), &status, WNOHANG) ==
        static_cast<pid_t>(pid)) {
      gone = true;
      break;
    }
    if (!PidAlive(pid)) {
      // Not our child (adopted): no waitpid possible; gone is gone.
      gone = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (!gone) ::kill(static_cast<pid_t>(pid), SIGKILL);
  {
    int status = 0;
    ::waitpid(static_cast<pid_t>(pid), &status, WNOHANG);
    std::lock_guard<std::mutex> lock(mu_);
    table_.erase(id);
  }
  return true;
}

}  // namespace mbufo::server

namespace mbufo::npu {

bool ProbeSidecar(const std::string& base_url) noexcept {
  try {
    if (!base_url.starts_with("http://") &&
        !base_url.starts_with("https://")) {
      return false;
    }
    mbufo::server::EnsureCurlGlobalInit();
    const mbufo::server::CurlPtr curl(curl_easy_init());
    if (!curl) return false;
    std::string url = base_url;
    while (!url.empty() && url.back() == '/') url.pop_back();
    url += "/v1/models";
    std::string sink;
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_PROXY, "");
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS, 500L);
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS, 1000L);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION,
                     mbufo::server::WriteBodyCallback);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &sink);
    if (curl_easy_perform(curl.get()) != CURLE_OK) return false;
    long code = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &code);
    return code >= 200 && code < 500;
  } catch (...) {
    return false;
  }
}

}  // namespace mbufo::npu
