#include "src/cli/serve/admin_ops.hpp"

#include <curl/curl.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "mbufo_build_version.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace mbufo::server::admin_ops {
namespace {

namespace fs = std::filesystem;

constexpr const char* kGithubTagsUrl =
    "https://api.github.com/repos/dardeaw/MBufo/tags?per_page=100";
constexpr std::uint64_t kSafetyDiskMarginBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;

HttpResponse OkJson(json::Value body) {
  return {.status = 200,
          .reason = "OK",
          .body = body.dump(),
          .headers = {{"Cache-Control", "no-store, no-cache, must-revalidate"},
                      {"Content-Type", "application/json; charset=utf-8"}}};
}

HttpResponse OkRawJson(std::string body) {
  return {.status = 200,
          .reason = "OK",
          .body = std::move(body),
          .headers = {{"Cache-Control", "no-store, no-cache, must-revalidate"},
                      {"Content-Type", "application/json; charset=utf-8"}}};
}

HttpResponse ErrJson(int status, const std::string& reason,
                     const std::string& message, const std::string& code) {
  json::Value obj = json::Value::object();
  obj["message"] = message;
  obj["type"] = "admin_error";
  obj["code"] = code;
  json::Value e = json::Value::object();
  e["error"] = std::move(obj);
  return {.status = status,
          .reason = reason,
          .body = e.dump(),
          .headers = {{"Content-Type", "application/json; charset=utf-8"}}};
}

std::string ToLower(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

std::uint64_t NowEpochSec() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

std::string NowIsoUtc() {
  const std::time_t t = std::time(nullptr);
  std::tm tm_utc{};
  gmtime_r(&t, &tm_utc);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
  return std::string(buf);
}

std::string ReadWholeFile(const fs::path& p) {
  std::ifstream ifs(p, std::ios::binary);
  if (!ifs) return {};
  return std::string((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());
}

bool WriteFileAtomic(const fs::path& p, const std::string& content,
                     std::string* error = nullptr) {
  std::error_code ec;
  if (p.has_parent_path()) {
    fs::create_directories(p.parent_path(), ec);
    if (ec) {
      if (error != nullptr) *error = "mkdir failed: " + ec.message();
      return false;
    }
  }
  const fs::path tmp = p.string() + ".tmp";
  {
    std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
    if (!ofs) {
      if (error != nullptr) *error = "cannot write temp file: " + tmp.string();
      return false;
    }
    ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!ofs) {
      if (error != nullptr) *error = "write failed: " + tmp.string();
      return false;
    }
  }
  fs::rename(tmp, p, ec);
  if (ec) {
    if (error != nullptr) *error = "rename failed: " + ec.message();
    return false;
  }
  return true;
}

std::string ReadTailLines(const fs::path& p, std::size_t max_lines = 30) {
  const std::string raw = ReadWholeFile(p);
  if (raw.empty()) return {};
  std::vector<std::string> lines;
  std::istringstream iss(raw);
  std::string line;
  while (std::getline(iss, line)) {
    // Strip trailing CR from progress bars cleanly; keep last line segment after \r
    const auto cr = line.rfind('\r');
    if (cr != std::string::npos) {
      line = line.substr(cr + 1);
    }
    if (!line.empty()) lines.push_back(std::move(line));
  }
  if (lines.size() > max_lines) {
    lines.erase(lines.begin(),
                lines.begin() + static_cast<std::ptrdiff_t>(lines.size() - max_lines));
  }
  std::string out;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (i > 0) out.push_back('\n');
    out += lines[i];
  }
  return out;
}

fs::path ResolveRepoDir() {
  if (const char* e = std::getenv("MBUFO_REPO_DIR"); e != nullptr && *e != '\0') {
    return fs::path(e);
  }
  if (fs::exists("/home/dardeaw/git/gufo-final2")) {
    return fs::path("/home/dardeaw/git/gufo-final2");
  }
  return fs::current_path();
}

fs::path ResolveModelsRoot() {
  if (const char* e = std::getenv("MBUFO_MODELS_DIR");
      e != nullptr && *e != '\0') {
    return fs::path(e);
  }
  if (fs::exists("/mnt/nvme-data/mbufo-models")) {
    return fs::path("/mnt/nvme-data/mbufo-models");
  }
  return fs::path("models");
}

fs::path ResolveCacheRoot() {
  if (fs::exists("/mnt/nvme-data/mbufo-cache")) {
    return fs::path("/mnt/nvme-data/mbufo-cache");
  }
  if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
    return fs::path(home) / ".cache/mbufo";
  }
  return fs::path("/tmp/mbufo-cache");
}

std::string ResolveHfBinary() {
  if (const char* e = std::getenv("MBUFO_HF_BIN"); e != nullptr && *e != '\0') {
    return e;
  }
  if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
    for (const char* rel : {"/.local/bin/hf", "/venv-npu/bin/hf"}) {
      const fs::path cand = fs::path(home) / (rel + 1);
      if (::access(cand.c_str(), X_OK) == 0) return cand.string();
    }
  }
  for (const char* fixed : {"/home/dardeaw/.local/bin/hf",
                            "/home/dardeaw/venv-npu/bin/hf",
                            "/usr/local/bin/hf", "/usr/bin/hf"}) {
    if (::access(fixed, X_OK) == 0) return fixed;
  }
  return "hf";
}

void QueryDiskSpace(const fs::path& dir, std::uint64_t* out_free,
                    std::uint64_t* out_total) {
  std::error_code ec;
  fs::path probe = dir;
  if (!fs::exists(probe, ec) && probe.has_parent_path()) {
    probe = probe.parent_path();
  }
  const auto sp = fs::space(probe, ec);
  if (!ec) {
    *out_free = static_cast<std::uint64_t>(sp.available);
    *out_total = static_cast<std::uint64_t>(sp.capacity);
  } else {
    *out_free = 0;
    *out_total = 0;
  }
}

std::size_t CurlWriteCallback(void* ptr, std::size_t size, std::size_t nmemb,
                              void* userdata) {
  const std::size_t total = size * nmemb;
  auto* out = static_cast<std::string*>(userdata);
  out->append(static_cast<const char*>(ptr), total);
  return total;
}

bool HttpGet(const std::string& url,
             const std::vector<std::string>& extra_headers, long timeout_sec,
             long* out_http_status, std::string* out_body,
             std::string* out_err) {
  static std::once_flag s_curl_init;
  std::call_once(s_curl_init, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });

  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    if (out_err != nullptr) *out_err = "curl_easy_init failed";
    return false;
  }

  std::string buf;
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "User-Agent: mbufo-admin/1.0");
  headers = curl_slist_append(headers, "Accept: application/json");
  for (const auto& h : extra_headers) {
    headers = curl_slist_append(headers, h.c_str());
  }

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);

  const CURLcode rc = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (out_http_status != nullptr) *out_http_status = http_code;
  if (rc != CURLE_OK) {
    if (out_err != nullptr) *out_err = curl_easy_strerror(rc);
    return false;
  }
  if (out_body != nullptr) *out_body = std::move(buf);
  return true;
}

bool IsValidRepoSlug(std::string_view repo) {
  const auto slash = repo.find('/');
  if (slash == std::string_view::npos || slash == 0 ||
      slash + 1 >= repo.size()) {
    return false;
  }
  if (repo.find('/', slash + 1) != std::string_view::npos) return false;
  if (repo.find("..") != std::string_view::npos) return false;
  for (char c : repo) {
    const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '/' ||
                    c == '.' || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

bool SanitizeSubdir(std::string_view raw, std::string* out) {
  std::string s(raw);
  while (!s.empty() && (s.front() == '/' || s.front() == ' ')) s.erase(0, 1);
  while (!s.empty() && (s.back() == '/' || s.back() == ' ')) s.pop_back();
  if (s.empty() || s.find("..") != std::string::npos ||
      s.find('\\') != std::string::npos) {
    return false;
  }
  for (char& c : s) {
    const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '.' ||
                    c == '_' || c == '-' || c == '/';
    if (!ok) c = '-';
  }
  *out = std::move(s);
  return true;
}

bool IsValidRepoFilePath(std::string_view fp) {
  if (fp.empty() || fp.front() == '/' || fp.find("..") != std::string::npos ||
      fp.find('\\') != std::string::npos) {
    return false;
  }
  return true;
}

void CloseInheritedFdsAndExec(const std::vector<std::string>& args,
                              const std::string& log_path,
                              const std::string& extra_hf_token,
                              pid_t* out_pid, std::string* error) {
  std::error_code ec;
  fs::create_directories(fs::path(log_path).parent_path(), ec);
  const int log_fd =
      ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (log_fd < 0) {
    if (error != nullptr) {
      *error = "cannot open log " + log_path + ": " + std::strerror(errno);
    }
    *out_pid = -1;
    return;
  }

  std::vector<std::string> args_copy = args;
  std::vector<char*> argv;
  argv.reserve(args_copy.size() + 1);
  for (auto& a : args_copy) argv.push_back(a.data());
  argv.push_back(nullptr);

  long max_fd = ::sysconf(_SC_OPEN_MAX);
  if (max_fd < 256 || max_fd > 65536) max_fd = 4096;

  const pid_t child = ::fork();
  if (child < 0) {
    const int e = errno;
    ::close(log_fd);
    if (error != nullptr) {
      *error = std::string("fork failed: ") + std::strerror(e);
    }
    *out_pid = -1;
    return;
  }
  if (child == 0) {
    ::setsid();
    ::signal(SIGPIPE, SIG_DFL);
    ::signal(SIGTERM, SIG_DFL);
    if (::dup2(log_fd, STDOUT_FILENO) < 0) ::_exit(127);
    if (::dup2(log_fd, STDERR_FILENO) < 0) ::_exit(127);
    for (int fd = 3; fd < static_cast<int>(max_fd); ++fd) {
      if (fd != log_fd) ::close(fd);
    }
    if (log_fd >= 3) ::close(log_fd);
    const int dev_null = ::open("/dev/null", O_RDONLY);
    if (dev_null >= 0) {
      if (dev_null != STDIN_FILENO) {
        ::dup2(dev_null, STDIN_FILENO);
        ::close(dev_null);
      }
    }
    if (!extra_hf_token.empty()) {
      ::setenv("HF_TOKEN", extra_hf_token.c_str(), 1);
    }
    ::execvp(argv[0], argv.data());
    ::_exit(127);
  }
  ::close(log_fd);
  *out_pid = child;
}

// ---- Quantization & shard grouping for Option B HF Browser ----

struct ParsedGgufInfo {
  std::string quant;
  std::string role;   // "main", "mmproj", "draft"
  std::string group_key;
  int shard_idx{1};
  int shard_total{1};
};

std::string DetectQuantFromFilename(const std::string& path_str) {
  static const std::regex kQuantRe(
      "(ROCmFP4-FAST|ROCmFP4|Q4_0_ROCMFP4_FAST|ROCmFP3|MXFP4|BF16|F16|F32|"
      "UD-Q8_K_XL|UD-Q6_K_XL|UD-Q5_K_XL|UD-Q4_K_XL|UD-Q3_K_XL|UD-Q2_K_XL|"
      "UD-IQ4_XS|UD-IQ4_NL|UD-IQ3_XXS|UD-IQ3_S|UD-IQ3_M|UD-IQ2_XXS|UD-IQ2_XS|UD-IQ2_S|UD-IQ2_M|UD-IQ1_M|UD-IQ1_S|"
      "IQ4_NL|IQ4_XS|IQ3_S|IQ3_XS|IQ3_M|IQ3_XXS|IQ2_XXS|IQ2_XS|IQ2_S|IQ2_M|"
      "IQ1_S|IQ1_M|Q8_0|Q6_K_L|Q6_K|Q5_K_M|Q5_K_S|Q5_K_L|Q5_0|Q5_1|"
      "Q4_K_M|Q4_K_S|Q4_K_L|Q4_0|Q4_1|Q3_K_L|Q3_K_M|Q3_K_S|Q2_K_L|Q2_K|"
      "v2q8k|W8A8|W4A16)",
      std::regex::icase);

  // Check basename first, then parent directory name
  const fs::path p(path_str);
  const std::string base = p.filename().string();
  std::smatch m;
  if (std::regex_search(base, m, kQuantRe)) {
    return m.str(1);
  }
  const std::string parent = p.parent_path().filename().string();
  if (!parent.empty() && std::regex_search(parent, m, kQuantRe)) {
    return m.str(1);
  }
  return "GGUF";
}

ParsedGgufInfo ParseGgufPath(const std::string& rel_path) {
  ParsedGgufInfo info;
  const std::string lower = ToLower(rel_path);
  if (lower.find("imatrix") != std::string::npos) {
    info.role = "imatrix";
  } else if (lower.find("mmproj") != std::string::npos) {
    info.role = "mmproj";
  } else if (lower.find("mtp") != std::string::npos ||
             lower.find("dflash") != std::string::npos ||
             lower.find("dspark") != std::string::npos ||
             lower.find("draft") != std::string::npos) {
    info.role = "draft";
  } else {
    info.role = "main";
  }
  info.quant = DetectQuantFromFilename(rel_path);

  static const std::regex kShardRe(
      "^(.*)-([0-9]{5})-of-([0-9]{5})\\.gguf$", std::regex::icase);
  std::smatch sm;
  if (std::regex_match(rel_path, sm, kShardRe)) {
    info.group_key = sm.str(1) + ".gguf";
    info.shard_idx = std::atoi(sm.str(2).c_str());
    info.shard_total = std::atoi(sm.str(3).c_str());
  } else {
    info.group_key = rel_path;
    info.shard_idx = 1;
    info.shard_total = 1;
  }
  return info;
}

bool AllFilesExistUnder(const fs::path& dir,
                        const std::vector<std::string>& files) {
  if (files.empty()) return false;
  std::error_code ec;
  if (!fs::exists(dir, ec)) return false;
  for (const auto& f : files) {
    if (!fs::exists(dir / f, ec) &&
        !fs::exists(dir / fs::path(f).filename(), ec)) {
      return false;
    }
  }
  return true;
}

// ---- Background Download Job Registry ----

struct DownloadJob {
  std::string id;
  std::string repo;
  std::string target_subdir;
  std::string target_dir;
  std::string log_path;
  std::vector<std::string> files;
  bool full_repo{false};
  std::uint64_t expected_bytes{0};
  std::uint64_t started_epoch{0};
  std::uint64_t finished_epoch{0};
  pid_t pid{-1};
  std::string status{"running"};  // running, completed, failed, cancelled
  int exit_code{0};
  std::string error;
};

std::mutex g_dl_mutex;
std::vector<DownloadJob> g_dl_jobs;
std::uint64_t g_dl_seq = 0;

std::uint64_t SumDirectoryBytes(const fs::path& dir) {
  std::error_code ec;
  if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return 0;
  std::uint64_t total = 0;
  fs::recursive_directory_iterator it(
      dir, fs::directory_options::skip_permission_denied, ec);
  fs::recursive_directory_iterator end;
  for (; !ec && it != end; it.increment(ec)) {
    std::error_code fec;
    if (!it->is_regular_file(fec)) continue;
    const std::string fname = it->path().filename().string();
    // Skip lock/metadata tiny files if desired, count weights + .incomplete blobs
    if (fname.size() >= 5 && fname.substr(fname.size() - 5) == ".lock") continue;
    const auto sz = it->file_size(fec);
    if (!fec) total += static_cast<std::uint64_t>(sz);
  }
  return total;
}

void ReapDownloadJobsLocked() {
  for (auto& job : g_dl_jobs) {
    if (job.status != "running" || job.pid <= 0) continue;
    int st = 0;
    const pid_t r = ::waitpid(job.pid, &st, WNOHANG);
    if (r == job.pid) {
      job.finished_epoch = NowEpochSec();
      if (WIFEXITED(st)) {
        job.exit_code = WEXITSTATUS(st);
        job.status = (job.exit_code == 0) ? "completed" : "failed";
        if (job.status == "failed") {
          job.error = "hf exited with code " + std::to_string(job.exit_code);
        }
      } else if (WIFSIGNALED(st)) {
        job.exit_code = 128 + WTERMSIG(st);
        job.status = "cancelled";
      } else {
        job.status = "failed";
      }
      job.pid = -1;
    } else if (r < 0 && errno == ECHILD) {
      job.finished_epoch = NowEpochSec();
      job.status = "completed";
      job.pid = -1;
    }
  }
}

// ---- Version Check Cache ----

struct UpdateCheckCache {
  std::mutex mu;
  std::uint64_t fetched_epoch{0};
  std::string remote_sha;
  std::string remote_short;
  std::string remote_tag;
  std::string check_error;
};

UpdateCheckCache g_upd_cache;

struct SemverTag {
  int major{0};
  int minor{0};
  int patch{0};
  bool mbufo_prefixed{false};
  bool valid{false};
};

SemverTag ParseSemverTag(std::string_view tag) {
  SemverTag res;
  if (tag.rfind("mbufo-v", 0) == 0) {
    res.mbufo_prefixed = true;
    tag.remove_prefix(7);
  } else if (!tag.empty() && tag.front() == 'v') {
    tag.remove_prefix(1);
  } else {
    return {};
  }
  auto parse_num = [&](std::string_view sv, int* out) -> bool {
    if (sv.empty()) return false;
    long v = 0;
    for (char c : sv) {
      if (!std::isdigit(static_cast<unsigned char>(c))) return false;
      v = v * 10 + (c - '0');
      if (v > 999999) return false;
    }
    *out = static_cast<int>(v);
    return true;
  };
  const std::size_t d1 = tag.find('.');
  if (d1 == std::string_view::npos) return {};
  const std::size_t d2 = tag.find('.', d1 + 1);
  if (d2 == std::string_view::npos) return {};
  std::string_view tail = tag.substr(d2 + 1);
  if (const auto dash = tail.find('-'); dash != std::string_view::npos) {
    tail = tail.substr(0, dash);
  }
  if (!parse_num(tag.substr(0, d1), &res.major) ||
      !parse_num(tag.substr(d1 + 1, d2 - d1 - 1), &res.minor) ||
      !parse_num(tail, &res.patch)) {
    return {};
  }
  res.valid = true;
  return res;
}

bool SemverGreater(const SemverTag& a, const SemverTag& b) {
  return std::tie(a.major, a.minor, a.patch) >
         std::tie(b.major, b.minor, b.patch);
}

std::string RandomHex(std::size_t num_bytes) {
  std::vector<std::uint8_t> buf(num_bytes, 0);
  std::ifstream urand("/dev/urandom", std::ios::binary);
  if (urand) {
    urand.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(buf.size()));
  }
  if (!urand || urand.gcount() < static_cast<std::streamsize>(buf.size())) {
    const std::uint64_t seed =
        NowEpochSec() ^ static_cast<std::uint64_t>(::getpid());
    for (std::size_t i = 0; i < buf.size(); ++i) {
      buf[i] = static_cast<std::uint8_t>((seed >> ((i % 8) * 8)) ^ (i * 37U));
    }
  }
  constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(buf.size() * 2);
  for (std::uint8_t b : buf) {
    out.push_back(kHex[b >> 4]);
    out.push_back(kHex[b & 15]);
  }
  return out;
}

std::string HashSecret(std::string_view salt, std::string_view secret) {
  const std::string material =
      std::string(salt) + ":" + std::string(secret);
  return crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(material.data()),
      material.size()));
}

std::string ComputeWebSessionToken(const ServerConfig& cfg) {
  if (cfg.web_password_hash.empty()) return {};
  return HashSecret(cfg.web_password_salt + ":session", cfg.web_password_hash);
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

std::string ExtractCookie(std::string_view cookie_hdr, std::string_view key) {
  while (!cookie_hdr.empty()) {
    const std::size_t semi = cookie_hdr.find(';');
    const std::string_view part =
        semi == std::string_view::npos ? cookie_hdr
                                       : cookie_hdr.substr(0, semi);
    const std::size_t eq = part.find('=');
    if (eq != std::string_view::npos) {
      if (TrimOws(part.substr(0, eq)) == key) {
        return TrimOws(part.substr(eq + 1));
      }
    }
    if (semi == std::string_view::npos) break;
    cookie_hdr.remove_prefix(semi + 1);
  }
  return {};
}

std::string ExtractBearerToken(std::string_view auth_hdr) {
  const std::string trimmed = TrimOws(auth_hdr);
  const std::size_t sp = trimmed.find(' ');
  if (sp == std::string::npos) return {};
  if (ToLower(trimmed.substr(0, sp)) != "bearer") return {};
  return TrimOws(trimmed.substr(sp + 1));
}

bool WritePrivateFileAtomic(const fs::path& p, const std::string& content,
                            std::string* error = nullptr) {
  std::error_code ec;
  if (p.has_parent_path()) {
    fs::create_directories(p.parent_path(), ec);
    if (ec) {
      if (error != nullptr) *error = "mkdir failed: " + ec.message();
      return false;
    }
  }
  const fs::path tmp = p.string() + ".tmp";
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) {
    if (error != nullptr) *error = "cannot open temp file: " + tmp.string();
    return false;
  }
  std::size_t off = 0;
  while (off < content.size()) {
    const ssize_t w = ::write(fd, content.data() + off, content.size() - off);
    if (w <= 0) {
      ::close(fd);
      ::unlink(tmp.c_str());
      if (error != nullptr) *error = "write failed: " + tmp.string();
      return false;
    }
    off += static_cast<std::size_t>(w);
  }
  ::fchmod(fd, 0600);
  ::fsync(fd);
  ::close(fd);
  fs::rename(tmp, p, ec);
  if (ec) {
    ::unlink(tmp.c_str());
    if (error != nullptr) *error = "rename failed: " + ec.message();
    return false;
  }
  return true;
}

json::Value DefaultModelToJson(const DefaultModelConfig& d) {
  json::Value obj = json::Value::object();
  obj["configured"] = d.configured;
  obj["id"] = d.id;
  json::Value aliases = json::Value::array();
  for (const auto& a : d.aliases) {
    if (!a.empty()) aliases.push_back(a);
  }
  obj["aliases"] = std::move(aliases);
  obj["path"] = d.path;
  obj["max_context"] = static_cast<std::size_t>(d.max_context);
  obj["session_count"] = d.session_count;
  obj["kv_cache_quant"] = d.kv_cache_quant;
  obj["device"] = d.device;
  obj["npu_quant"] = d.npu_quant;
  obj["speculative"] = d.speculative;
  obj["draft_model_path"] = d.draft_model_path;
  obj["vision_model_path"] = d.vision_model_path;
  obj["enable_yarn"] = d.enable_yarn;
  obj["enable_disk_cache"] = d.enable_disk_cache;
  return obj;
}

json::Value ServerConfigToJson(const ServerConfig& cfg) {
  json::Value obj = json::Value::object();
  obj["port"] = static_cast<std::size_t>(cfg.port);
  obj["web_auth_enabled"] = cfg.web_auth_enabled;
  obj["web_password_salt"] = cfg.web_password_salt;
  obj["web_password_hash"] = cfg.web_password_hash;
  obj["api_auth_enabled"] = cfg.api_auth_enabled;
  obj["api_auth_share_web_password"] = cfg.api_auth_share_web_password;
  obj["api_key_salt"] = cfg.api_key_salt;
  obj["api_key_hash"] = cfg.api_key_hash;
  obj["default_model"] = DefaultModelToJson(cfg.default_model);
  return obj;
}

bool ParseDefaultModelJson(const json::Value& v, DefaultModelConfig* out) {
  if (!v.is_object()) return false;
  DefaultModelConfig d;
  if (const auto* f = v.find("configured"); f != nullptr && f->is_bool())
    d.configured = f->as_bool(false);
  d.id = v.member_str("id");
  if (const auto* al = v.find("aliases"); al != nullptr && al->is_array()) {
    for (const auto& x : al->items()) {
      if (x.is_string() && !x.str().empty()) d.aliases.push_back(x.str());
    }
  }
  d.path = v.member_str("path");
  d.max_context = static_cast<std::uint32_t>(
      v.member_size("max_context", v.member_size("context_length", 0)));
  d.session_count = std::max<std::size_t>(
      1, v.member_size("session_count", v.member_size("sessions", 1)));
  d.kv_cache_quant = v.member_str("kv_cache_quant", "q8");
  d.device = v.member_str("device", "gpu");
  if (d.device != "gpu" && d.device != "npu") d.device = "gpu";
  d.npu_quant = v.member_str("npu_quant");
  d.speculative = v.member_str("speculative", "off");
  d.draft_model_path = v.member_str("draft_model_path");
  d.vision_model_path = v.member_str("vision_model_path");
  if (const auto* f = v.find("enable_yarn"); f != nullptr && f->is_bool())
    d.enable_yarn = f->as_bool(false);
  if (const auto* f = v.find("enable_disk_cache"); f != nullptr && f->is_bool())
    d.enable_disk_cache = f->as_bool(false);
  if (!d.path.empty() && v.find("configured") == nullptr) d.configured = true;
  *out = std::move(d);
  return true;
}

struct RuntimeConfigState {
  std::mutex mu;
  fs::path config_path;
  int running_port{8741};
  fs::path cache_disk_dir;
  std::size_t cache_disk_bytes{0};
  bool cli_api_key_active{false};
  ServerConfig cfg;
};

RuntimeConfigState g_rt;

fs::path ActiveConfigPathLocked() {
  if (!g_rt.config_path.empty()) return g_rt.config_path;
  if (const char* e = std::getenv("MBUFO_CONFIG_FILE");
      e != nullptr && *e != '\0') {
    return fs::path(e);
  }
  return ResolveCacheRoot() / "server_config.json";
}

struct OfflineManifestMeta {
  std::string path;
  std::string filename;
  std::uint64_t checkpoint_id{0};
  std::uint64_t lineage_id{0};
  std::string compatibility_hex;
  std::size_t token_count{0};
  std::size_t chunk_count{0};
  std::uint64_t chunk_bytes{0};
  std::uint64_t private_bytes{0};
  std::uint64_t manifest_bytes{0};
  std::int64_t mtime_epoch{0};
  std::string purpose;
  std::vector<std::string> chunk_filenames;
  std::vector<std::string> private_filenames;
};

std::string DiskIdToFilename(const std::uint8_t* bytes16) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(36);
  for (int i = 0; i < 16; ++i) {
    out.push_back(kHex[bytes16[i] >> 4]);
    out.push_back(kHex[bytes16[i] & 15]);
  }
  out += ".bin";
  return out;
}

std::uint32_t ReadLe32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t ReadLe64(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= (static_cast<std::uint64_t>(p[i]) << (8 * i));
  }
  return v;
}

bool ParseOfflineManifest(const fs::path& file_path, OfflineManifestMeta* out) {
  std::ifstream ifs(file_path, std::ios::binary);
  if (!ifs) return false;
  std::vector<std::uint8_t> b(
      (std::istreambuf_iterator<char>(ifs)),
      std::istreambuf_iterator<char>());
  if (b.size() < 8 + 4 + 8 + 8 + 32 + 1 + 4 + 4 + 4 + 4 + 8) return false;
  if (std::memcmp(b.data(), "GUFOMNF2", 8) != 0) return false;
  std::size_t pos = 8;
  auto need = [&](std::size_t n) { return pos + n <= b.size(); };
  if (!need(4)) return false;
  const std::uint32_t ver = ReadLe32(b.data() + pos);
  pos += 4;
  if (ver != 2) return false;
  if (!need(16)) return false;
  const std::uint64_t ckpt = ReadLe64(b.data() + pos);
  const std::uint64_t lin = ReadLe64(b.data() + pos + 8);
  pos += 16;
  if (!need(32)) return false;
  constexpr char kHex[] = "0123456789abcdef";
  std::string compat_hex;
  compat_hex.reserve(64);
  for (std::size_t i = 0; i < 32; ++i) {
    compat_hex.push_back(kHex[b[pos + i] >> 4]);
    compat_hex.push_back(kHex[b[pos + i] & 15]);
  }
  pos += 32;
  if (!need(1 + 4 + 4)) return false;
  const std::uint8_t purpose = b[pos];
  pos += 1;
  pos += 4;
  const std::uint32_t in_len = ReadLe32(b.data() + pos);
  pos += 4;
  if (!need(in_len + 4)) return false;
  pos += in_len;
  const std::uint32_t tok_cnt = ReadLe32(b.data() + pos);
  pos += 4;
  if (static_cast<std::uint64_t>(tok_cnt) * 4ULL + 4ULL > b.size() - pos)
    return false;
  pos += static_cast<std::size_t>(tok_cnt) * 4;
  if (!need(4)) return false;
  const std::uint32_t comp_cnt = ReadLe32(b.data() + pos);
  pos += 4;

  std::set<std::string> seen_chunks;
  std::set<std::string> seen_priv;
  std::uint64_t chunk_bytes = 0;
  std::uint64_t priv_bytes = 0;

  for (std::uint32_t ci = 0; ci < comp_cnt; ++ci) {
    if (!need(41 + 4)) return false;
    pos += 41;
    const std::uint32_t cc = ReadLe32(b.data() + pos);
    pos += 4;
    for (std::uint32_t j = 0; j < cc; ++j) {
      if (!need(32)) return false;
      std::string fname = DiskIdToFilename(b.data() + pos);
      const std::uint64_t sz = ReadLe64(b.data() + pos + 16);
      pos += 32;
      if (seen_chunks.insert(fname).second) {
        chunk_bytes += sz;
      }
    }
    if (!need(1)) return false;
    const std::uint8_t has_tail = b[pos++];
    if (has_tail) {
      if (!need(32)) return false;
      std::string fname = DiskIdToFilename(b.data() + pos);
      const std::uint64_t sz = ReadLe64(b.data() + pos + 16);
      pos += 32;
      if (seen_priv.insert(fname).second) priv_bytes += sz;
    }
    if (!need(1)) return false;
    const std::uint8_t has_ps = b[pos++];
    if (has_ps) {
      if (!need(32)) return false;
      std::string fname = DiskIdToFilename(b.data() + pos);
      const std::uint64_t sz = ReadLe64(b.data() + pos + 16);
      pos += 32;
      if (seen_priv.insert(fname).second) priv_bytes += sz;
    }
  }

  OfflineManifestMeta meta;
  meta.path = file_path.string();
  meta.filename = file_path.filename().string();
  meta.checkpoint_id = ckpt;
  meta.lineage_id = lin;
  meta.compatibility_hex = std::move(compat_hex);
  meta.token_count = tok_cnt;
  meta.chunk_count = seen_chunks.size();
  meta.chunk_bytes = chunk_bytes;
  meta.private_bytes = priv_bytes;
  meta.manifest_bytes = b.size();
  meta.purpose = purpose == 1 ? "learned" : "prompt";
  meta.chunk_filenames.assign(seen_chunks.begin(), seen_chunks.end());
  meta.private_filenames.assign(seen_priv.begin(), seen_priv.end());

  struct stat st{};
  if (::stat(file_path.c_str(), &st) == 0) {
    meta.mtime_epoch = static_cast<std::int64_t>(st.st_mtime);
  }
  *out = std::move(meta);
  return true;
}

std::vector<OfflineManifestMeta> ScanOfflineManifests(const fs::path& dir) {
  std::vector<OfflineManifestMeta> list;
  const fs::path mf_dir = dir / "v2" / "manifests";
  std::error_code ec;
  if (!fs::exists(mf_dir, ec) || !fs::is_directory(mf_dir, ec)) return list;
  for (const auto& entry : fs::directory_iterator(mf_dir, ec)) {
    if (ec) break;
    if (!entry.is_regular_file(ec)) continue;
    if (entry.path().extension() != ".bin") continue;
    OfflineManifestMeta meta;
    if (ParseOfflineManifest(entry.path(), &meta)) {
      list.push_back(std::move(meta));
    }
  }
  return list;
}

json::Value ReadPersistentUpdateStatus() {
  const fs::path sf = ResolveCacheRoot() / "update_status.json";
  const std::string raw = ReadWholeFile(sf);
  if (raw.empty()) {
    json::Value obj = json::Value::object();
    obj["phase"] = "idle";
    obj["active"] = false;
    return obj;
  }
  try {
    auto v = json::parse(raw);
    if (v.is_object()) return v;
  } catch (...) {
  }
  json::Value obj = json::Value::object();
  obj["phase"] = "idle";
  obj["active"] = false;
  return obj;
}

}  // namespace

// ---- Local repository model_mapping.json ----

HttpResponse HandleGetModelMapping(const HttpRequest&) {
  const fs::path repo_dir = ResolveRepoDir();
  const fs::path cache_file = ResolveCacheRoot() / "model_mapping.json";
  for (const fs::path& cand :
       {repo_dir / "model_mapping.json", fs::path("model_mapping.json"),
        cache_file}) {
    std::string content = ReadWholeFile(cand);
    if (!content.empty()) {
      try {
        auto parsed = json::parse(content);
        if (parsed.is_object()) return OkRawJson(std::move(content));
      } catch (...) {
      }
    }
  }
  return ErrJson(404, "Not Found",
                 "model_mapping.json not found in local repository",
                 "mapping_not_found");
}

HttpResponse HandleSyncModelMapping(const HttpRequest& req) {
  return HandleGetModelMapping(req);
}

// ---- Option B: Dynamic HF Repo File Browser ----

HttpResponse HandleHfListFiles(const HttpRequest& req) {
  const std::string repo = req.query_param("repo");
  const std::string subdir_q = req.query_param("subdir");
  if (!IsValidRepoSlug(repo)) {
    return ErrJson(400, "Bad Request",
                   "Invalid HuggingFace repo slug (expected org/model)",
                   "invalid_repo");
  }

  std::vector<std::string> hdrs;
  std::string hf_token = req.header("X-HF-Token");
  if (hf_token.empty()) {
    if (const char* env_tok = std::getenv("HF_TOKEN");
        env_tok != nullptr && *env_tok != '\0') {
      hf_token = env_tok;
    }
  }
  if (!hf_token.empty()) {
    hdrs.push_back("Authorization: Bearer " + hf_token);
  }

  const std::string api_url =
      "https://huggingface.co/api/models/" + repo + "/tree/main?recursive=true";
  long http_code = 0;
  std::string body;
  std::string err;
  if (!HttpGet(api_url, hdrs, 20, &http_code, &body, &err)) {
    return ErrJson(502, "Bad Gateway",
                   "HuggingFace API request failed: " + err,
                   "hf_api_failed");
  }
  if (http_code == 401 || http_code == 403) {
    return ErrJson(static_cast<int>(http_code), "Forbidden",
                   "HuggingFace access denied (gated/private repo? Provide HF_TOKEN).",
                   "hf_auth_required");
  }
  if (http_code != 200) {
    return ErrJson(502, "Bad Gateway",
                   "HuggingFace API returned HTTP " + std::to_string(http_code) +
                       " for " + repo,
                   "hf_http_error");
  }

  json::Value tree;
  try {
    tree = json::parse(body);
  } catch (const std::exception& e) {
    return ErrJson(502, "Bad Gateway",
                   std::string("Failed parsing HF tree response: ") + e.what(),
                   "hf_parse_error");
  }
  if (!tree.is_array()) {
    return ErrJson(502, "Bad Gateway", "HF tree response is not an array",
                   "hf_unexpected_response");
  }

  struct GroupEntry {
    std::string id;
    std::string quant;
    std::string role;
    std::string display_name;
    std::vector<std::string> files;
    std::uint64_t size_bytes{0};
    int shards{0};
    bool full_repo{false};
  };

  std::map<std::string, GroupEntry> gguf_groups;
  std::uint64_t safetensors_total_bytes = 0;
  std::uint64_t safetensors_weight_bytes = 0;
  int safetensors_count = 0;
  std::vector<std::pair<std::string, std::uint64_t>> individual_st_files;

  for (const auto& item : tree.items()) {
    if (!item.is_object()) continue;
    if (item.member_str("type") != "file") continue;
    const std::string path_str = item.member_str("path");
    if (path_str.empty()) continue;

    std::uint64_t sz = 0;
    if (const auto* lfs = item.find("lfs"); lfs != nullptr && lfs->is_object()) {
      if (const auto* lsz = lfs->find("size");
          lsz != nullptr && lsz->is_number()) {
        sz = static_cast<std::uint64_t>(lsz->as_double(0.0));
      }
    }
    if (sz == 0) {
      if (const auto* rsz = item.find("size");
          rsz != nullptr && rsz->is_number()) {
        sz = static_cast<std::uint64_t>(rsz->as_double(0.0));
      }
    }

    const std::string low = ToLower(path_str);
    if (low.size() >= 5 && low.substr(low.size() - 5) == ".gguf") {
      const ParsedGgufInfo pi = ParseGgufPath(path_str);
      auto& g = gguf_groups[pi.group_key];
      if (g.files.empty()) {
        g.id = pi.group_key;
        g.quant = pi.quant;
        g.role = pi.role;
        g.display_name = pi.group_key;
      }
      g.files.push_back(path_str);
      g.size_bytes += sz;
      g.shards = std::max(g.shards + 1, pi.shard_total);
    } else if (low.size() >= 12 &&
               low.substr(low.size() - 12) == ".safetensors") {
      ++safetensors_count;
      safetensors_weight_bytes += sz;
      safetensors_total_bytes += sz;
      individual_st_files.emplace_back(path_str, sz);
    } else {
      // Skip large media sample images/videos in README when estimating safetensors repo size
      const bool is_media =
          (low.size() >= 4 && (low.substr(low.size() - 4) == ".png" ||
                               low.substr(low.size() - 4) == ".jpg" ||
                               low.substr(low.size() - 4) == ".mp4" ||
                               low.substr(low.size() - 4) == ".wav")) ||
          (low.size() >= 5 && low.substr(low.size() - 5) == ".webp");
      if (!is_media) safetensors_total_bytes += sz;
    }
  }

  std::string default_subdir = subdir_q;
  if (default_subdir.empty()) {
    const auto slash = repo.find('/');
    default_subdir = ToLower(repo.substr(slash + 1));
  }
  std::string clean_subdir;
  if (!SanitizeSubdir(default_subdir, &clean_subdir)) clean_subdir = "model";
  const fs::path models_root = ResolveModelsRoot();
  const fs::path check_dir = models_root / clean_subdir;

  std::vector<GroupEntry> entries;
  if (safetensors_count > 0) {
    GroupEntry bundle;
    bundle.id = "full-safetensors-repo";
    bundle.quant = "BF16 / Safetensors";
    bundle.role = "safetensors_bundle";
    bundle.display_name =
        "Full Safetensors Model (" + std::to_string(safetensors_count) + " weight files + configs)";
    bundle.files = {"*"};
    bundle.size_bytes = safetensors_total_bytes;
    bundle.shards = safetensors_count;
    bundle.full_repo = true;
    entries.push_back(std::move(bundle));
  }

  for (auto& [k, g] : gguf_groups) {
    g.shards = static_cast<int>(g.files.size());
    entries.push_back(std::move(g));
  }

  std::sort(entries.begin(), entries.end(),
            [](const GroupEntry& a, const GroupEntry& b) {
              auto rank = [](const GroupEntry& e) {
                if (e.full_repo) return 0;
                if (e.role == "main") return 1;
                if (e.role == "mmproj") return 2;
                return 3;
              };
              const int ra = rank(a), rb = rank(b);
              if (ra != rb) return ra < rb;
              return a.size_bytes < b.size_bytes;
            });

  // Also include individual safetensors if no GGUFs exist and <= 20 files
  if (gguf_groups.empty() && individual_st_files.size() > 1 &&
      individual_st_files.size() <= 30) {
    for (const auto& [fp, sz] : individual_st_files) {
      GroupEntry g;
      g.id = fp;
      g.quant = "Safetensors";
      g.role = "shard";
      g.display_name = fp;
      g.files = {fp};
      g.size_bytes = sz;
      g.shards = 1;
      entries.push_back(std::move(g));
    }
  }

  json::Value out_files = json::Value::array();
  for (const auto& g : entries) {
    json::Value item = json::Value::object();
    item["id"] = g.id;
    item["name"] = g.display_name;
    item["quant"] = g.quant;
    item["role"] = g.role;
    item["shards"] = g.shards;
    item["full_repo"] = g.full_repo;
    item["size_bytes"] = static_cast<double>(g.size_bytes);
    std::error_code on_ec;
    const bool on_disk = g.full_repo
                             ? (fs::exists(check_dir / "config.json", on_ec) ||
                                fs::exists(check_dir / "model_index.json", on_ec))
                             : AllFilesExistUnder(check_dir, g.files);
    item["on_disk"] = on_disk;
    json::Value farr = json::Value::array();
    for (const auto& f : g.files) farr.push_back(f);
    item["files"] = std::move(farr);
    out_files.push_back(std::move(item));
  }

  std::uint64_t disk_free = 0, disk_total = 0;
  QueryDiskSpace(models_root, &disk_free, &disk_total);

  json::Value resp = json::Value::object();
  resp["repo"] = repo;
  resp["suggested_subdir"] = clean_subdir;
  resp["models_dir"] = models_root.string();
  resp["disk_free_bytes"] = static_cast<double>(disk_free);
  resp["disk_total_bytes"] = static_cast<double>(disk_total);
  resp["files"] = std::move(out_files);
  return OkJson(std::move(resp));
}

// ---- Server-side `hf download` Manager ----

HttpResponse HandleStartDownload(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return ErrJson(400, "Bad Request", e.what(), "parse_error");
  }
  if (!body.is_object()) {
    return ErrJson(400, "Bad Request", "request body must be an object",
                   "invalid_request");
  }

  const std::string repo = body.member_str("repo");
  if (!IsValidRepoSlug(repo)) {
    return ErrJson(400, "Bad Request", "invalid HuggingFace repo slug",
                   "invalid_repo");
  }

  std::string raw_subdir = body.member_str("target_subdir");
  if (raw_subdir.empty()) {
    raw_subdir = ToLower(repo.substr(repo.find('/') + 1));
  }
  std::string clean_subdir;
  if (!SanitizeSubdir(raw_subdir, &clean_subdir)) {
    return ErrJson(400, "Bad Request", "invalid target_subdir",
                   "invalid_subdir");
  }

  const bool full_repo =
      body.find("full_repo") != nullptr && body.find("full_repo")->as_bool(false);
  std::vector<std::string> files;
  if (const auto* farr = body.find("files"); farr != nullptr && farr->is_array()) {
    for (const auto& v : farr->items()) {
      if (!v.is_string()) continue;
      const std::string fp = v.str();
      if (fp == "*") continue;
      if (!IsValidRepoFilePath(fp)) {
        return ErrJson(400, "Bad Request", "invalid file path: " + fp,
                       "invalid_file");
      }
      files.push_back(fp);
    }
  }
  if (!full_repo && files.empty()) {
    return ErrJson(400, "Bad Request", "no files selected for download",
                   "no_files");
  }

  std::uint64_t expected_bytes = 0;
  if (const auto* tb = body.find("total_bytes");
      tb != nullptr && tb->is_number()) {
    expected_bytes = static_cast<std::uint64_t>(tb->as_double(0.0));
  }

  const fs::path models_root = ResolveModelsRoot();
  const fs::path target_dir = models_root / clean_subdir;
  std::error_code ec;
  fs::create_directories(target_dir, ec);
  if (ec) {
    return ErrJson(500, "Internal Server Error",
                   "cannot create target dir: " + ec.message(),
                   "mkdir_failed");
  }

  std::uint64_t disk_free = 0, disk_total = 0;
  QueryDiskSpace(models_root, &disk_free, &disk_total);
  if (expected_bytes > 0 && disk_free > 0 &&
      expected_bytes + kSafetyDiskMarginBytes > disk_free) {
    return ErrJson(409, "Conflict",
                   "Insufficient disk space on " + models_root.string() +
                       " (need ~" +
                       std::to_string(expected_bytes >> 30) + " GiB + 2 GiB safety, available " +
                       std::to_string(disk_free >> 30) + " GiB)",
                   "insufficient_disk");
  }

  const std::string hf_token = body.member_str("hf_token");
  const std::string hf_bin = ResolveHfBinary();

  std::lock_guard<std::mutex> lock(g_dl_mutex);
  ReapDownloadJobsLocked();

  // Prevent duplicate active download to same repo + target_subdir
  for (const auto& j : g_dl_jobs) {
    if (j.status == "running" && j.repo == repo &&
        j.target_subdir == clean_subdir) {
      return ErrJson(409, "Conflict",
                     "A download for " + repo + " into " + clean_subdir +
                         " is already running (job " + j.id + ")",
                     "already_downloading");
    }
  }

  const std::string job_id =
      "dl-" + std::to_string(NowEpochSec()) + "-" + std::to_string(++g_dl_seq);
  const fs::path log_path =
      ResolveCacheRoot() / "downloads" / (job_id + ".log");

  std::vector<std::string> args = {hf_bin, "download", repo};
  if (full_repo) {
    for (const char* pat : {"*.png", "*.jpg", "*.jpeg", "*.webp", "*.mp4", "*.gif"}) {
      args.emplace_back("--exclude");
      args.emplace_back(pat);
    }
  } else {
    for (const auto& f : files) {
      args.push_back(f);
    }
  }
  args.emplace_back("--local-dir");
  args.push_back(target_dir.string());

  pid_t child = -1;
  std::string err;
  CloseInheritedFdsAndExec(args, log_path.string(), hf_token, &child, &err);
  if (child < 0) {
    return ErrJson(500, "Internal Server Error",
                   "failed launching hf download: " + err,
                   "download_launch_failed");
  }

  DownloadJob job;
  job.id = job_id;
  job.repo = repo;
  job.target_subdir = clean_subdir;
  job.target_dir = target_dir.string();
  job.log_path = log_path.string();
  job.files = std::move(files);
  job.full_repo = full_repo;
  job.expected_bytes = expected_bytes;
  job.started_epoch = NowEpochSec();
  job.pid = child;
  job.status = "running";
  g_dl_jobs.push_back(std::move(job));

  json::Value resp = json::Value::object();
  resp["id"] = job_id;
  resp["repo"] = repo;
  resp["target_dir"] = target_dir.string();
  resp["status"] = "running";
  resp["pid"] = static_cast<long long>(child);
  return OkJson(std::move(resp));
}

HttpResponse HandleListDownloads(const HttpRequest&) {
  std::lock_guard<std::mutex> lock(g_dl_mutex);
  ReapDownloadJobsLocked();

  json::Value arr = json::Value::array();
  for (auto it = g_dl_jobs.rbegin(); it != g_dl_jobs.rend(); ++it) {
    const auto& j = *it;
    const std::uint64_t downloaded_bytes = SumDirectoryBytes(j.target_dir);
    json::Value row = json::Value::object();
    row["id"] = j.id;
    row["repo"] = j.repo;
    row["target_subdir"] = j.target_subdir;
    row["target_dir"] = j.target_dir;
    row["full_repo"] = j.full_repo;
    row["file_count"] = static_cast<std::size_t>(j.files.size());
    row["status"] = j.status;
    row["pid"] = static_cast<long long>(j.pid);
    row["expected_bytes"] = static_cast<double>(j.expected_bytes);
    row["downloaded_bytes"] = static_cast<double>(downloaded_bytes);
    row["started_epoch"] = static_cast<double>(j.started_epoch);
    row["finished_epoch"] = static_cast<double>(j.finished_epoch);
    row["exit_code"] = j.exit_code;
    row["error"] = j.error;
    row["log_tail"] = ReadTailLines(j.log_path, 8);
    arr.push_back(std::move(row));
  }

  const fs::path models_root = ResolveModelsRoot();
  std::uint64_t disk_free = 0, disk_total = 0;
  QueryDiskSpace(models_root, &disk_free, &disk_total);

  json::Value resp = json::Value::object();
  resp["models_dir"] = models_root.string();
  resp["disk_free_bytes"] = static_cast<double>(disk_free);
  resp["disk_total_bytes"] = static_cast<double>(disk_total);
  resp["jobs"] = std::move(arr);
  return OkJson(std::move(resp));
}

HttpResponse HandleCancelDownload(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return ErrJson(400, "Bad Request", e.what(), "parse_error");
  }
  const std::string id = body.is_object() ? body.member_str("id") : "";
  if (id.empty()) {
    return ErrJson(400, "Bad Request", "'id' is required", "missing_id");
  }

  std::lock_guard<std::mutex> lock(g_dl_mutex);
  ReapDownloadJobsLocked();
  for (auto& j : g_dl_jobs) {
    if (j.id == id) {
      if (j.status == "running" && j.pid > 0) {
        // Negative PID sends signal to the child's process group (setsid)
        ::kill(-j.pid, SIGTERM);
        int st = 0;
        for (int i = 0; i < 10; ++i) {
          if (::waitpid(j.pid, &st, WNOHANG) == j.pid) break;
          ::usleep(100000);
        }
        if (::waitpid(j.pid, &st, WNOHANG) == 0) {
          ::kill(-j.pid, SIGKILL);
          ::waitpid(j.pid, &st, 0);
        }
        j.pid = -1;
        j.status = "cancelled";
        j.finished_epoch = NowEpochSec();
      }
      json::Value resp = json::Value::object();
      resp["id"] = j.id;
      resp["status"] = j.status;
      return OkJson(std::move(resp));
    }
  }
  return ErrJson(404, "Not Found", "unknown download job id: " + id,
                 "job_not_found");
}

// ---- Real-Time GitHub Tag-Based Self-Update Check ----

HttpResponse HandleUpdateCheck(const HttpRequest& req) {
  const bool force = (req.query_param("force") == "1" ||
                      req.query_param("force") == "true");
  const std::string running_sha = mbufo::build::kGitCommit;
  const std::string running_short = mbufo::build::kGitShort;
  const std::string running_tag = mbufo::build::kGitTag;
  const std::string build_time = mbufo::build::kBuildTimeUtc;

  std::lock_guard<std::mutex> lock(g_upd_cache.mu);
  const std::uint64_t now = NowEpochSec();
  if (force || g_upd_cache.fetched_epoch == 0 ||
      now - g_upd_cache.fetched_epoch >= 60) {
    std::vector<std::string> hdrs;
    if (const char* ght = std::getenv("GITHUB_TOKEN");
        ght != nullptr && *ght != '\0') {
      hdrs.push_back("Authorization: Bearer " + std::string(ght));
    }

    long tcode = 0;
    std::string tbody;
    std::string terr;
    g_upd_cache.check_error.clear();

    if (HttpGet(kGithubTagsUrl, hdrs, 10, &tcode, &tbody, &terr) &&
        tcode == 200) {
      try {
        auto tags = json::parse(tbody);
        if (tags.is_array()) {
          std::string best_mbufo_name, best_mbufo_sha;
          SemverTag best_mbufo_ver;
          std::string best_any_name, best_any_sha;
          SemverTag best_any_ver;

          for (const auto& t : tags.items()) {
            if (!t.is_object()) continue;
            const std::string name = t.member_str("name");
            const SemverTag sv = ParseSemverTag(name);
            if (!sv.valid) continue;
            std::string sha;
            if (const auto* co = t.find("commit");
                co != nullptr && co->is_object()) {
              sha = co->member_str("sha");
            }
            if (sv.mbufo_prefixed) {
              if (!best_mbufo_ver.valid || SemverGreater(sv, best_mbufo_ver)) {
                best_mbufo_ver = sv;
                best_mbufo_name = name;
                best_mbufo_sha = sha;
              }
            } else {
              if (!best_any_ver.valid || SemverGreater(sv, best_any_ver)) {
                best_any_ver = sv;
                best_any_name = name;
                best_any_sha = sha;
              }
            }
          }
          if (best_mbufo_ver.valid) {
            g_upd_cache.remote_tag = best_mbufo_name;
            g_upd_cache.remote_sha = best_mbufo_sha;
          } else if (best_any_ver.valid) {
            g_upd_cache.remote_tag = best_any_name;
            g_upd_cache.remote_sha = best_any_sha;
          } else {
            g_upd_cache.check_error = "No semantic release tags found on GitHub";
          }
          g_upd_cache.remote_short =
              g_upd_cache.remote_sha.size() >= 8
                  ? g_upd_cache.remote_sha.substr(0, 8)
                  : g_upd_cache.remote_sha;
        }
      } catch (const std::exception& e) {
        g_upd_cache.check_error =
            std::string("GitHub tags parse error: ") + e.what();
      }
    } else {
      g_upd_cache.check_error =
          terr.empty() ? ("GitHub HTTP " + std::to_string(tcode)) : terr;
    }
    g_upd_cache.fetched_epoch = now;
  }

  const SemverTag local_sv = ParseSemverTag(running_tag);
  const SemverTag remote_sv = ParseSemverTag(g_upd_cache.remote_tag);
  const bool update_available =
      local_sv.valid && remote_sv.valid && SemverGreater(remote_sv, local_sv);

  json::Value resp = json::Value::object();
  resp["compare_mode"] = "tag";
  resp["running_sha"] = running_sha;
  resp["running_short"] = running_short;
  resp["running_tag"] = running_tag;
  resp["build_time_utc"] = build_time;
  resp["remote_sha"] = g_upd_cache.remote_sha;
  resp["remote_short"] = g_upd_cache.remote_short;
  resp["remote_tag"] = g_upd_cache.remote_tag;
  resp["remote_message"] =
      g_upd_cache.remote_tag.empty()
          ? ""
          : ("Release tag " + g_upd_cache.remote_tag);
  resp["update_available"] = update_available;
  resp["check_error"] = g_upd_cache.check_error;
  resp["checked_at_utc"] = NowIsoUtc();
  resp["repo_dir"] = ResolveRepoDir().string();
  resp["status"] = ReadPersistentUpdateStatus();
  return OkJson(std::move(resp));
}

HttpResponse HandleUpdateApply(const HttpRequest&) {
  const auto cur = ReadPersistentUpdateStatus();
  if (const auto* act = cur.find("active"); act != nullptr && act->as_bool(false)) {
    return ErrJson(409, "Conflict", "A server self-update is already in progress",
                   "update_in_progress");
  }

  const fs::path repo_dir = ResolveRepoDir();
  const fs::path script = repo_dir / "scripts/mbufo_self_update.sh";
  if (!fs::exists(script)) {
    return ErrJson(500, "Internal Server Error",
                   "Missing updater script: " + script.string(),
                   "missing_updater");
  }

  const fs::path cache_dir = ResolveCacheRoot();
  const fs::path status_file = cache_dir / "update_status.json";
  const fs::path log_file = cache_dir / "update.log";
  const fs::path mapping_cache = cache_dir / "model_mapping.json";

  // Pre-seed status file synchronously so immediate UI polls see active=true
  json::Value init_st = json::Value::object();
  init_st["phase"] = "fetching";
  init_st["active"] = true;
  init_st["running_before"] = mbufo::build::kGitCommit;
  init_st["target_sha"] = "";
  init_st["message"] = "Launching self-update worker...";
  init_st["error"] = "";
  init_st["started_at"] = NowIsoUtc();
  init_st["updated_at"] = NowIsoUtc();
  WriteFileAtomic(status_file, init_st.dump());

  const std::string pid_str = std::to_string(static_cast<long long>(::getpid()));
  const std::vector<std::string> args = {
      "/bin/bash",
      script.string(),
      pid_str,
      repo_dir.string(),
      status_file.string(),
      log_file.string(),
      mapping_cache.string(),
  };

  pid_t child = -1;
  std::string err;
  CloseInheritedFdsAndExec(args, log_file.string(), "", &child, &err);
  if (child < 0) {
    init_st["phase"] = "failed";
    init_st["active"] = false;
    init_st["error"] = err;
    WriteFileAtomic(status_file, init_st.dump());
    return ErrJson(500, "Internal Server Error",
                   "Failed launching self-update worker: " + err,
                   "updater_launch_failed");
  }

  json::Value resp = json::Value::object();
  resp["started"] = true;
  resp["worker_pid"] = static_cast<long long>(child);
  resp["running_sha"] = mbufo::build::kGitCommit;
  resp["status_file"] = status_file.string();
  resp["log_file"] = log_file.string();
  return OkJson(std::move(resp));
}

HttpResponse HandleUpdateStatus(const HttpRequest&) {
  const fs::path cache_dir = ResolveCacheRoot();
  json::Value resp = ReadPersistentUpdateStatus();
  resp["running_sha"] = mbufo::build::kGitCommit;
  resp["running_short"] = mbufo::build::kGitShort;
  resp["running_tag"] = mbufo::build::kGitTag;
  resp["build_time_utc"] = mbufo::build::kBuildTimeUtc;
  resp["log_tail"] = ReadTailLines(cache_dir / "update.log", 25);
  return OkJson(std::move(resp));
}

// ---- Persistent ServerConfig, Dual-Layer Auth & Offline Disk Cache ----

bool LoadServerConfigFile(const fs::path& path, ServerConfig* out,
                          std::string* error) {
  if (out == nullptr) return false;
  const std::string raw = ReadWholeFile(path);
  if (raw.empty()) {
    if (error != nullptr) *error = "config file empty or missing";
    return false;
  }
  try {
    const auto v = json::parse(raw);
    if (!v.is_object()) {
      if (error != nullptr) *error = "config JSON must be an object";
      return false;
    }
    ServerConfig cfg;
    const int p = static_cast<int>(v.member_size("port", 8741));
    cfg.port = (p >= 1 && p <= 65535) ? p : 8741;
    if (const auto* f = v.find("web_auth_enabled"); f != nullptr && f->is_bool())
      cfg.web_auth_enabled = f->as_bool(false);
    cfg.web_password_salt = v.member_str("web_password_salt");
    cfg.web_password_hash = v.member_str("web_password_hash");
    if (const auto* f = v.find("api_auth_enabled"); f != nullptr && f->is_bool())
      cfg.api_auth_enabled = f->as_bool(false);
    if (const auto* f = v.find("api_auth_share_web_password");
        f != nullptr && f->is_bool())
      cfg.api_auth_share_web_password = f->as_bool(false);
    cfg.api_key_salt = v.member_str("api_key_salt");
    cfg.api_key_hash = v.member_str("api_key_hash");
    if (const auto* dm = v.find("default_model"); dm != nullptr) {
      ParseDefaultModelJson(*dm, &cfg.default_model);
    }
    *out = std::move(cfg);
    return true;
  } catch (const std::exception& e) {
    if (error != nullptr) *error = e.what();
    return false;
  }
}

bool SeedServerConfigIfMissing(const fs::path& path, const ServerConfig& seed,
                               std::string* error) {
  std::error_code ec;
  if (fs::exists(path, ec)) return true;
  return WritePrivateFileAtomic(path, ServerConfigToJson(seed).dump(), error);
}

void InitServerRuntime(const fs::path& config_path, int running_port,
                       const fs::path& cache_disk_dir,
                       std::size_t cache_disk_bytes,
                       bool cli_api_key_active) {
  std::lock_guard<std::mutex> lock(g_rt.mu);
  g_rt.config_path = config_path;
  g_rt.running_port = running_port > 0 ? running_port : 8741;
  g_rt.cache_disk_dir = cache_disk_dir;
  g_rt.cache_disk_bytes = cache_disk_bytes;
  g_rt.cli_api_key_active = cli_api_key_active;
  g_rt.cfg = ServerConfig{};
  g_rt.cfg.port = g_rt.running_port;
  if (!config_path.empty()) {
    ServerConfig loaded;
    if (LoadServerConfigFile(config_path, &loaded)) {
      g_rt.cfg = std::move(loaded);
    }
  }
}

int GetRunningPort() {
  std::lock_guard<std::mutex> lock(g_rt.mu);
  return g_rt.running_port;
}

fs::path GetConfiguredCacheDiskDir() {
  std::lock_guard<std::mutex> lock(g_rt.mu);
  if (!g_rt.cache_disk_dir.empty()) return g_rt.cache_disk_dir;
  return ResolveCacheRoot();
}

std::size_t GetConfiguredCacheDiskBytes() {
  std::lock_guard<std::mutex> lock(g_rt.mu);
  return g_rt.cache_disk_bytes;
}

bool IsWebAuthEnabled() {
  std::lock_guard<std::mutex> lock(g_rt.mu);
  return g_rt.cfg.web_auth_enabled && !g_rt.cfg.web_password_hash.empty();
}

bool CheckWebAuth(const HttpRequest& req) {
  ServerConfig cfg;
  {
    std::lock_guard<std::mutex> lock(g_rt.mu);
    if (!g_rt.cfg.web_auth_enabled || g_rt.cfg.web_password_hash.empty())
      return true;
    cfg = g_rt.cfg;
  }
  const std::string expected_cookie = ComputeWebSessionToken(cfg);
  const std::string cookie_tok =
      ExtractCookie(req.header("Cookie"), "mbufo_admin_session");
  if (!cookie_tok.empty() && cookie_tok == expected_cookie) return true;

  const std::string bearer = ExtractBearerToken(req.header("Authorization"));
  if (!bearer.empty() &&
      HashSecret(cfg.web_password_salt, bearer) == cfg.web_password_hash) {
    return true;
  }
  const std::string hdr_pw = req.header("X-Admin-Password");
  if (!hdr_pw.empty() &&
      HashSecret(cfg.web_password_salt, hdr_pw) == cfg.web_password_hash) {
    return true;
  }
  return false;
}

bool IsApiAuthEnabled() {
  std::lock_guard<std::mutex> lock(g_rt.mu);
  if (!g_rt.cfg.api_auth_enabled) return false;
  if (g_rt.cfg.api_auth_share_web_password)
    return !g_rt.cfg.web_password_hash.empty();
  return !g_rt.cfg.api_key_hash.empty();
}

bool CheckApiAuth(const HttpRequest& req) {
  ServerConfig cfg;
  {
    std::lock_guard<std::mutex> lock(g_rt.mu);
    if (!g_rt.cfg.api_auth_enabled) return true;
    cfg = g_rt.cfg;
  }
  std::string supplied = ExtractBearerToken(req.header("Authorization"));
  if (supplied.empty()) supplied = TrimOws(req.header("x-api-key"));
  if (supplied.empty()) return false;

  if (cfg.api_auth_share_web_password) {
    return !cfg.web_password_hash.empty() &&
           HashSecret(cfg.web_password_salt, supplied) == cfg.web_password_hash;
  }
  return !cfg.api_key_hash.empty() &&
         HashSecret(cfg.api_key_salt, supplied) == cfg.api_key_hash;
}

HttpResponse HandleAuthStatus(const HttpRequest& req) {
  bool web_enabled = false;
  bool api_enabled = false;
  bool web_set = false;
  {
    std::lock_guard<std::mutex> lock(g_rt.mu);
    web_set = !g_rt.cfg.web_password_hash.empty();
    web_enabled = g_rt.cfg.web_auth_enabled && web_set;
    api_enabled = g_rt.cfg.api_auth_enabled &&
                  (g_rt.cfg.api_auth_share_web_password
                       ? web_set
                       : !g_rt.cfg.api_key_hash.empty());
  }
  const bool authenticated = !web_enabled || CheckWebAuth(req);
  json::Value resp = json::Value::object();
  resp["web_auth_enabled"] = web_enabled;
  resp["api_auth_enabled"] = api_enabled;
  resp["authenticated"] = authenticated;
  return OkJson(std::move(resp));
}

HttpResponse HandleAuthLogin(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return ErrJson(400, "Bad Request", e.what(), "parse_error");
  }
  const std::string password =
      body.is_object() ? body.member_str("password") : "";
  ServerConfig cfg;
  {
    std::lock_guard<std::mutex> lock(g_rt.mu);
    cfg = g_rt.cfg;
  }
  if (!cfg.web_auth_enabled || cfg.web_password_hash.empty()) {
    json::Value resp = json::Value::object();
    resp["authenticated"] = true;
    resp["web_auth_enabled"] = false;
    return OkJson(std::move(resp));
  }
  if (HashSecret(cfg.web_password_salt, password) != cfg.web_password_hash) {
    return ErrJson(401, "Unauthorized", "Invalid Web UI password",
                   "invalid_password");
  }
  const std::string tok = ComputeWebSessionToken(cfg);
  json::Value resp = json::Value::object();
  resp["authenticated"] = true;
  HttpResponse r = OkJson(std::move(resp));
  r.headers.emplace_back(
      "Set-Cookie",
      "mbufo_admin_session=" + tok +
          "; Path=/; HttpOnly; SameSite=Lax; Max-Age=604800");
  return r;
}

HttpResponse HandleAuthLogout(const HttpRequest&) {
  json::Value resp = json::Value::object();
  resp["authenticated"] = false;
  HttpResponse r = OkJson(std::move(resp));
  r.headers.emplace_back(
      "Set-Cookie",
      "mbufo_admin_session=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0");
  return r;
}

HttpResponse HandleGetSettings(const HttpRequest&,
                               const std::string& live_default_id) {
  ServerConfig cfg;
  int running_port = 8741;
  bool cli_api = false;
  fs::path cfg_path;
  fs::path cache_dir;
  std::size_t cache_bytes = 0;
  {
    std::lock_guard<std::mutex> lock(g_rt.mu);
    cfg = g_rt.cfg;
    running_port = g_rt.running_port;
    cli_api = g_rt.cli_api_key_active;
    cfg_path = ActiveConfigPathLocked();
    cache_dir =
        g_rt.cache_disk_dir.empty() ? ResolveCacheRoot() : g_rt.cache_disk_dir;
    cache_bytes = g_rt.cache_disk_bytes;
  }
  json::Value resp = json::Value::object();
  resp["config_path"] = cfg_path.string();
  resp["running_port"] = static_cast<std::size_t>(running_port);
  resp["configured_port"] = static_cast<std::size_t>(cfg.port);
  resp["restart_required"] = (cfg.port != running_port);
  resp["web_auth_enabled"] = cfg.web_auth_enabled;
  resp["web_password_set"] = !cfg.web_password_hash.empty();
  resp["api_auth_enabled"] = cfg.api_auth_enabled;
  resp["api_auth_share_web_password"] = cfg.api_auth_share_web_password;
  resp["api_key_set"] = !cfg.api_key_hash.empty();
  resp["cli_api_key_active"] = cli_api;
  resp["live_default_id"] = live_default_id;
  resp["default_model"] = DefaultModelToJson(cfg.default_model);
  resp["cache_disk_dir"] = cache_dir.string();
  resp["cache_disk_bytes"] = static_cast<std::size_t>(cache_bytes);
  return OkJson(std::move(resp));
}

HttpResponse HandleSaveSettings(const HttpRequest& req) {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return ErrJson(400, "Bad Request", e.what(), "parse_error");
  }
  if (!body.is_object()) {
    return ErrJson(400, "Bad Request", "request body must be an object",
                   "invalid_request");
  }

  std::lock_guard<std::mutex> lock(g_rt.mu);
  ServerConfig next = g_rt.cfg;

  if (const auto* pf = body.find("port"); pf != nullptr && pf->is_number()) {
    const int p = static_cast<int>(pf->as_size(0));
    if (p < 1 || p > 65535) {
      return ErrJson(400, "Bad Request", "port must be in 1..65535",
                     "invalid_port");
    }
    next.port = p;
  }

  if (const auto* f = body.find("clear_web_password");
      f != nullptr && f->as_bool(false)) {
    next.web_password_salt.clear();
    next.web_password_hash.clear();
    next.web_auth_enabled = false;
  }
  if (const std::string new_pw = body.member_str("web_password");
      !new_pw.empty()) {
    next.web_password_salt = RandomHex(16);
    next.web_password_hash = HashSecret(next.web_password_salt, new_pw);
  }
  if (const auto* f = body.find("web_auth_enabled");
      f != nullptr && f->is_bool()) {
    next.web_auth_enabled = f->as_bool(false);
  }
  if (next.web_auth_enabled && next.web_password_hash.empty()) {
    return ErrJson(400, "Bad Request",
                   "Set a Web UI password before enabling Web UI authentication",
                   "missing_web_password");
  }

  if (const auto* f = body.find("clear_api_key");
      f != nullptr && f->as_bool(false)) {
    next.api_key_salt.clear();
    next.api_key_hash.clear();
    if (!next.api_auth_share_web_password) next.api_auth_enabled = false;
  }
  if (const std::string new_key = body.member_str("api_key");
      !new_key.empty()) {
    next.api_key_salt = RandomHex(16);
    next.api_key_hash = HashSecret(next.api_key_salt, new_key);
  }
  if (const auto* f = body.find("api_auth_share_web_password");
      f != nullptr && f->is_bool()) {
    next.api_auth_share_web_password = f->as_bool(false);
  }
  if (const auto* f = body.find("api_auth_enabled");
      f != nullptr && f->is_bool()) {
    next.api_auth_enabled = f->as_bool(false);
  }
  if (next.api_auth_enabled) {
    if (next.api_auth_share_web_password) {
      if (next.web_password_hash.empty()) {
        return ErrJson(
            400, "Bad Request",
            "Set a Web UI password before sharing it with /v1/* API auth",
            "missing_web_password");
      }
    } else if (next.api_key_hash.empty()) {
      return ErrJson(400, "Bad Request",
                     "Set an API Key before enabling /v1/* API authentication",
                     "missing_api_key");
    }
  }

  if (const auto* dm = body.find("default_model"); dm != nullptr) {
    DefaultModelConfig parsed_dm;
    if (!ParseDefaultModelJson(*dm, &parsed_dm)) {
      return ErrJson(400, "Bad Request", "default_model must be an object",
                     "invalid_default_model");
    }
    if (parsed_dm.configured && parsed_dm.path.empty()) {
      return ErrJson(400, "Bad Request", "default_model.path is required",
                     "missing_default_path");
    }
    next.default_model = std::move(parsed_dm);
  }

  const fs::path cfg_path = ActiveConfigPathLocked();
  std::string err;
  if (!WritePrivateFileAtomic(cfg_path, ServerConfigToJson(next).dump(), &err)) {
    return ErrJson(500, "Internal Server Error",
                   "Failed writing server_config.json: " + err,
                   "config_write_failed");
  }
  g_rt.cfg = next;

  json::Value resp = json::Value::object();
  resp["saved"] = true;
  resp["config_path"] = cfg_path.string();
  resp["running_port"] = static_cast<std::size_t>(g_rt.running_port);
  resp["configured_port"] = static_cast<std::size_t>(next.port);
  resp["restart_required"] = (next.port != g_rt.running_port);
  resp["web_auth_enabled"] = next.web_auth_enabled;
  resp["web_password_set"] = !next.web_password_hash.empty();
  resp["api_auth_enabled"] = next.api_auth_enabled;
  resp["api_auth_share_web_password"] = next.api_auth_share_web_password;
  resp["api_key_set"] = !next.api_key_hash.empty();
  resp["default_model"] = DefaultModelToJson(next.default_model);

  HttpResponse r = OkJson(std::move(resp));
  if (next.web_auth_enabled && !next.web_password_hash.empty()) {
    const std::string tok = ComputeWebSessionToken(next);
    r.headers.emplace_back(
        "Set-Cookie",
        "mbufo_admin_session=" + tok +
            "; Path=/; HttpOnly; SameSite=Lax; Max-Age=604800");
  }
  return r;
}

HttpResponse HandleServerRestart(const HttpRequest&) {
  int target_port = 8741;
  {
    std::lock_guard<std::mutex> lock(g_rt.mu);
    target_port = g_rt.cfg.port;
  }
  json::Value resp = json::Value::object();
  resp["status"] = "restarting";
  resp["target_port"] = static_cast<std::size_t>(target_port);
  std::thread([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(350));
    ::kill(::getpid(), SIGTERM);
  }).detach();
  return OkJson(std::move(resp));
}

DiskCacheSnapshot ListDiskCacheOffline(const fs::path& directory,
                                       std::uint64_t capacity_bytes) {
  DiskCacheSnapshot snap;
  snap.directory = directory.string();
  snap.capacity_bytes = capacity_bytes;
  if (directory.empty()) return snap;
  const fs::path v2_dir = directory / "v2";
  std::error_code ec;
  snap.enabled = fs::exists(v2_dir, ec);
  if (!snap.enabled) return snap;

  snap.managed_bytes =
      SumDirectoryBytes(v2_dir / "manifests") +
      SumDirectoryBytes(v2_dir / "chunks") +
      SumDirectoryBytes(v2_dir / "private");

  auto metas = ScanOfflineManifests(directory);
  std::sort(metas.begin(), metas.end(),
            [](const OfflineManifestMeta& a, const OfflineManifestMeta& b) {
              if (a.mtime_epoch != b.mtime_epoch)
                return a.mtime_epoch > b.mtime_epoch;
              return a.checkpoint_id > b.checkpoint_id;
            });
  snap.entries.reserve(metas.size());
  for (auto& m : metas) {
    DiskCacheEntryInfo row;
    row.checkpoint_id = m.checkpoint_id;
    row.lineage_id = m.lineage_id;
    row.manifest_file = std::move(m.filename);
    row.compatibility_hex = std::move(m.compatibility_hex);
    row.token_count = m.token_count;
    row.chunk_count = m.chunk_count;
    row.chunk_bytes = m.chunk_bytes;
    row.private_bytes = m.private_bytes;
    row.manifest_bytes = m.manifest_bytes;
    row.mtime_epoch = m.mtime_epoch;
    row.purpose = std::move(m.purpose);
    row.compatible = false;
    row.invalid = false;
    snap.entries.push_back(std::move(row));
  }
  return snap;
}

DiskCacheDeleteResult DeleteDiskCacheOffline(
    const fs::path& directory, const std::vector<std::uint64_t>& ids,
    bool delete_all, bool delete_incompatible,
    std::string_view active_compatibility) {
  DiskCacheDeleteResult res;
  if (directory.empty()) return res;
  auto metas = ScanOfflineManifests(directory);

  std::set<std::uint64_t> target_ids(ids.begin(), ids.end());
  auto is_target = [&](const OfflineManifestMeta& m) {
    if (delete_all) return true;
    if (delete_incompatible &&
        (active_compatibility.empty() ||
         m.compatibility_hex != active_compatibility)) {
      return true;
    }
    return target_ids.count(m.checkpoint_id) != 0;
  };

  std::map<std::string, int> chunk_refs;
  std::map<std::string, int> priv_refs;
  for (const auto& m : metas) {
    for (const auto& f : m.chunk_filenames) ++chunk_refs[f];
    for (const auto& f : m.private_filenames) ++priv_refs[f];
  }

  std::vector<OfflineManifestMeta> removed;
  std::set<std::uint64_t> found_ids;
  for (const auto& m : metas) {
    if (is_target(m)) {
      found_ids.insert(m.checkpoint_id);
      removed.push_back(m);
    }
  }
  if (!delete_all && !delete_incompatible) {
    for (const auto id : ids) {
      if (found_ids.count(id) == 0) ++res.not_found;
    }
  }

  const fs::path mf_dir = directory / "v2" / "manifests";
  const fs::path ch_dir = directory / "v2" / "chunks";
  const fs::path pr_dir = directory / "v2" / "private";

  std::error_code ec;
  for (const auto& m : removed) {
    if (fs::remove(mf_dir / m.filename, ec) || !fs::exists(mf_dir / m.filename)) {
      ++res.deleted;
      for (const auto& f : m.chunk_filenames) {
        if (--chunk_refs[f] <= 0) fs::remove(ch_dir / f, ec);
      }
      for (const auto& f : m.private_filenames) {
        if (--priv_refs[f] <= 0) fs::remove(pr_dir / f, ec);
      }
    }
  }

  res.remaining_managed_bytes =
      SumDirectoryBytes(mf_dir) + SumDirectoryBytes(ch_dir) +
      SumDirectoryBytes(pr_dir);
  return res;
}

}  // namespace mbufo::server::admin_ops
