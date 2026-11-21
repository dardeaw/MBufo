#include "src/cli/serve/npu_sidecar_proxy.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

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
