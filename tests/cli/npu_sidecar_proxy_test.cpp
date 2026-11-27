#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/model_admin_api.hpp"
#include "src/cli/serve/model_registry.hpp"
#include "src/cli/serve/npu_sidecar_proxy.hpp"
#include "src/core/compute/npu/npu_lane.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/net.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <iostream>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace mbufo::server;

std::string SendRawHttp(int port, std::string_view req) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  assert(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1);
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  assert(fd >= 0);
  assert(mbufo::net::SetSocketTimeouts(fd, 5000));
  assert(::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) ==
         0);
  while (!req.empty()) {
    const auto n = mbufo::net::SendNoSignal(fd, req.data(), req.size());
    assert(n > 0);
    req.remove_prefix(static_cast<std::size_t>(n));
  }
  std::string resp;
  char buf[4096];
  for (;;) {
    const auto n = ::read(fd, buf, sizeof(buf));
    if (n <= 0) break;
    resp.append(buf, static_cast<std::size_t>(n));
  }
  mbufo::net::CloseSocket(fd);
  return resp;
}

class MockUpstreamServer {
 public:
  MockUpstreamServer() {
    mbufo::net::EnsureSocketRuntimeStarted();
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(listen_fd_ >= 0);
    int yes = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    assert(::bind(listen_fd_, reinterpret_cast<const sockaddr*>(&addr),
                  sizeof(addr)) == 0);
    socklen_t len = sizeof(addr);
    assert(::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
                         &len) == 0);
    port_ = ntohs(addr.sin_port);
    assert(::listen(listen_fd_, 16) == 0);
    thread_ = std::jthread([this](std::stop_token) { Run(); });
  }

  ~MockUpstreamServer() {
    stopped_.store(true);
    if (listen_fd_ >= 0) {
      ::shutdown(listen_fd_, SHUT_RDWR);
      mbufo::net::CloseSocket(listen_fd_);
      listen_fd_ = -1;
    }
    if (thread_.joinable()) thread_.join();
  }

  int port() const noexcept { return port_; }
  std::string base_url() const {
    return "http://127.0.0.1:" + std::to_string(port_);
  }

 private:
  void Run() {
    while (!stopped_.load()) {
      sockaddr_in peer{};
      socklen_t plen = sizeof(peer);
      const int cfd =
          ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer), &plen);
      if (cfd < 0) break;
      HandleClient(cfd);
      mbufo::net::CloseSocket(cfd);
    }
  }

  static void HandleClient(int cfd) {
    std::string raw;
    char buf[4096];
    std::size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
      const ssize_t n = ::read(cfd, buf, sizeof(buf));
      if (n <= 0) return;
      raw.append(buf, static_cast<std::size_t>(n));
      header_end = raw.find("\r\n\r\n");
    }
    const std::string hdrs = raw.substr(0, header_end);
    std::size_t content_len = 0;
    {
      const auto pos = hdrs.find("Content-Length:");
      const auto pos2 = hdrs.find("content-length:");
      const auto p = (pos != std::string::npos) ? pos : pos2;
      if (p != std::string::npos) {
        content_len = std::stoul(hdrs.substr(p + 15));
      }
    }
    const std::size_t body_start = header_end + 4;
    while (raw.size() - body_start < content_len) {
      const ssize_t n = ::read(cfd, buf, sizeof(buf));
      if (n <= 0) break;
      raw.append(buf, static_cast<std::size_t>(n));
    }
    const std::string req_body = raw.substr(body_start, content_len);
    const std::size_t sp1 = hdrs.find(' ');
    const std::size_t sp2 = hdrs.find(' ', sp1 + 1);
    const std::string method = hdrs.substr(0, sp1);
    const std::string target = hdrs.substr(sp1 + 1, sp2 - sp1 - 1);

    std::string ctype = "application/json";
    std::string resp_body;
    if (target == "/v1/audio/speech") {
      ctype = "audio/wav";
      resp_body = std::string("RIFF\x24\x00\x00\x00WAVEfmt ", 16);
    } else if (target == "/v1/models") {
      resp_body = R"({"object":"list","data":[]})";
    } else {
      mbufo::json::Value obj = mbufo::json::Value::object();
      obj["method"] = method;
      obj["target"] = target;
      obj["body"] = req_body;
      resp_body = obj.dump();
    }

    const std::string wire =
        "HTTP/1.1 200 OK\r\nContent-Type: " + ctype +
        "\r\nContent-Length: " + std::to_string(resp_body.size()) +
        "\r\nConnection: close\r\n\r\n" + resp_body;
    std::size_t sent = 0;
    while (sent < wire.size()) {
      const auto n = mbufo::net::SendNoSignal(cfd, wire.data() + sent,
                                              wire.size() - sent);
      if (n <= 0) break;
      sent += static_cast<std::size_t>(n);
    }
  }

  int listen_fd_{-1};
  int port_{0};
  std::atomic<bool> stopped_{false};
  std::jthread thread_;
};

std::string ExtractBody(const std::string& http_resp) {
  const auto p = http_resp.find("\r\n\r\n");
  assert(p != std::string::npos);
  return http_resp.substr(p + 4);
}

void TestSidecarProxyAndProbe() {
  MockUpstreamServer upstream;
  assert(mbufo::npu::ProbeSidecar(upstream.base_url()));
  assert(!mbufo::npu::ProbeSidecar("http://127.0.0.1:1"));

  HttpServerOptions opts;
  opts.sidecar_proxy.enabled = true;
  opts.sidecar_proxy.decide_base_url = upstream.base_url();
  opts.sidecar_proxy.asr_base_url = upstream.base_url();
  opts.sidecar_proxy.tts_base_url = upstream.base_url();

  HttpServer server("127.0.0.1", 0, nullptr, nullptr, nullptr, nullptr, opts);
  std::string err;
  assert(server.start(&err));
  std::jthread srv_thread([&] { server.run(false); });

  // 1. D3 /v1/systemone
  {
    const std::string body = R"({"query":"route task"})";
    const std::string r = SendRawHttp(
        server.port(),
        "POST /v1/systemone HTTP/1.1\r\nContent-Type: application/json\r\n"
        "Content-Length: " +
            std::to_string(body.size()) + "\r\n\r\n" + body);
    assert(r.starts_with("HTTP/1.1 200 OK"));
    const auto j = mbufo::json::parse(ExtractBody(r));
    assert(j.member_str("method") == "POST");
    assert(j.member_str("target") == "/v1/systemone");
    assert(j.member_str("body") == body);
  }

  // 2. Whisper /v1/audio/transcriptions
  {
    const std::string body = "--B\r\nFAKEWAV\x00\x01\r\n--B--";
    const std::string r = SendRawHttp(
        server.port(),
        "POST /v1/audio/transcriptions?language=en HTTP/1.1\r\nContent-Type: "
        "multipart/form-data; boundary=B\r\nContent-Length: " +
            std::to_string(body.size()) + "\r\n\r\n" + body);
    assert(r.starts_with("HTTP/1.1 200 OK"));
    const auto j = mbufo::json::parse(ExtractBody(r));
    assert(j.member_str("target") == "/v1/audio/transcriptions?language=en");
    assert(j.member_str("body") == body);
  }

  // 3. Qwen3-TTS /v1/audio/speech (binary WAV preserved) & /v1/audio/voices
  {
    const std::string body = R"({"input":"hello","voice":"demo"})";
    const std::string r = SendRawHttp(
        server.port(),
        "POST /v1/audio/speech HTTP/1.1\r\nContent-Type: application/json\r\n"
        "Content-Length: " +
            std::to_string(body.size()) + "\r\n\r\n" + body);
    assert(r.starts_with("HTTP/1.1 200 OK"));
    assert(r.find("Content-Type: audio/wav") != std::string::npos);
    assert(ExtractBody(r) == std::string("RIFF\x24\x00\x00\x00WAVEfmt ", 16));

    const std::string v = SendRawHttp(
        server.port(), "GET /v1/audio/voices HTTP/1.1\r\n\r\n");
    assert(v.starts_with("HTTP/1.1 200 OK"));
    assert(mbufo::json::parse(ExtractBody(v)).member_str("method") == "GET");
  }

  server.stop();
  srv_thread.join();

  // Unreachable upstream returns 502 Bad Gateway
  HttpServerOptions down_opts;
  down_opts.sidecar_proxy.enabled = true;
  down_opts.sidecar_proxy.decide_base_url = "http://127.0.0.1:1";
  down_opts.sidecar_proxy.connect_timeout_ms = 200;
  HttpServer down_srv("127.0.0.1", 0, nullptr, nullptr, nullptr, nullptr,
                      down_opts);
  assert(down_srv.start(&err));
  std::jthread down_thread([&] { down_srv.run(false); });
  const std::string r502 = SendRawHttp(
      down_srv.port(),
      "POST /v1/systemone HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}");
  assert(r502.starts_with("HTTP/1.1 502 "));
  down_srv.stop();
  down_thread.join();
}

void TestModelCapabilityClassification() {
  const auto qwen = ClassifyModelCapabilities(
      "qwen3.8-flash-next", "qwen3.8-flash-next",
      "/mnt/nvme-data/mbufo-models/qwen3.8-flash-next/model.gguf", true, "mtp");
  assert(qwen.task == "chat");
  assert(qwen.has_vision);
  assert(qwen.has_tools);
  assert(std::find(qwen.badges.begin(), qwen.badges.end(), "CHAT") !=
         qwen.badges.end());
  assert(std::find(qwen.badges.begin(), qwen.badges.end(), "DECIDE") ==
         qwen.badges.end());
  assert(std::find(qwen.badges.begin(), qwen.badges.end(), "TOOLS") !=
         qwen.badges.end());

  const auto d3 = ClassifyModelCapabilities("d3-lite-npu", "sidecar",
                                            "http://127.0.0.1:8766", false, "");
  assert(d3.task == "decide");
  assert(!d3.has_tools);
  assert(std::find(d3.badges.begin(), d3.badges.end(), "DECIDE") !=
         d3.badges.end());

  auto reg = std::make_shared<ModelRegistry>();
  ModelAdminService admin(reg, nullptr);
  HttpRequest cat_req{.method = "GET", .path = "/admin/catalog"};
  const auto cat_resp = admin.Handle(cat_req);
  assert(cat_resp.status == 200);
  const auto doc = mbufo::json::parse(cat_resp.body);
  bool saw_d3 = false;
  for (const auto& m : doc.find("models")->items()) {
    if (m.member_str("id") == "d3-lite-npu") {
      saw_d3 = true;
      assert(m.member_str("task") == "decide");
    }
  }
  assert(saw_d3);
}

// One-shot listener on the sidecar's fixed port: captures the warmup request
// and answers 200. Returns false when the port is busy (test then skips).
bool CaptureWarmup(const std::string& id, std::string* captured, bool* warmed) {
  const SidecarLaunchSpec* spec = FindSidecarSpec(id);
  assert(spec != nullptr);
  mbufo::net::EnsureSocketRuntimeStarted();
  const int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
  int yes = 1;
  ::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(spec->port));
  if (::bind(lfd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0 ||
      ::listen(lfd, 4) != 0) {
    mbufo::net::CloseSocket(lfd);
    return false;
  }
  std::jthread server([&](std::stop_token) {
    const int cfd = ::accept(lfd, nullptr, nullptr);
    if (cfd < 0) return;
    mbufo::net::SetSocketTimeouts(cfd, 3000);
    std::string raw;
    char buf[4096];
    for (;;) {
      const ssize_t n = ::read(cfd, buf, sizeof(buf));
      if (n <= 0) break;
      raw.append(buf, static_cast<std::size_t>(n));
      const auto he = raw.find("\r\n\r\n");
      if (he == std::string::npos) continue;
      const auto cl = raw.find("Content-Length: ");
      if (cl == std::string::npos) break;
      const std::size_t want =
          static_cast<std::size_t>(std::stoul(raw.substr(cl + 16)));
      if (raw.size() >= he + 4 + want) break;
    }
    *captured = raw;
    const std::string resp =
        "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
    (void)mbufo::net::SendNoSignal(cfd, resp.data(), resp.size());
    mbufo::net::CloseSocket(cfd);
  });
  SidecarManager mgr;
  std::string err;
  *warmed = mgr.Warmup(id, &err);
  ::shutdown(lfd, SHUT_RDWR);
  mbufo::net::CloseSocket(lfd);
  return true;
}

void TestSidecarWarmup() {
  // Payload shapes per kind; image sidecars send nothing.
  for (const auto& s : KnownSidecars()) {
    const auto r = WarmupRequestFor(s);
    if (s.task == "image_generation") {
      assert(r.path.empty());
    } else {
      assert(r.path == s.endpoint);
      assert(!r.body.empty());
    }
    if (s.task == "asr") {
      assert(r.content_type.starts_with("multipart/form-data; boundary="));
      assert(r.body.find("RIFF") != std::string::npos);
    }
    if (s.task == "decide") assert(r.body.find("questions") != std::string::npos);
  }
  // Wire check: the request really reaches the sidecar port.
  for (const char* id : {"d3-lite-npu"}) {
    std::string captured;
    bool warmed = false;
    if (!CaptureWarmup(id, &captured, &warmed)) {
      std::cout << "SKIP warmup wire check (port busy): " << id << "\n";
      continue;
    }
    assert(warmed);
    assert(captured.starts_with("POST /v1/systemone "));
    assert(captured.find("application/json") != std::string::npos);
  }
}
}  // namespace

int main() {
  TestSidecarProxyAndProbe();
  TestModelCapabilityClassification();
  TestSidecarWarmup();
  std::cout << "NPU_SIDECAR_PROXY_AND_ADMIN_TESTS_PASS\n";
  return 0;
}
