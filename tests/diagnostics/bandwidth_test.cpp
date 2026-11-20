#include "src/core/diagnostics/bandwidth.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "src/core/diagnostics/artifact_validator.h"
#include "src/core/diagnostics/fingerprint.h"
#include "src/core/diagnostics/system_inventory.h"
#include "src/core/json.hpp"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

std::filesystem::path FindFixturesRoot() {
#ifdef TEST_FIXTURES_DIR
  if (std::filesystem::exists(TEST_FIXTURES_DIR)) {
    return TEST_FIXTURES_DIR;
  }
#endif
  const std::array<std::filesystem::path, 3> paths = {
      "tests/fixtures/diagnostics",
      "../tests/fixtures/diagnostics",
      "../../tests/fixtures/diagnostics",
  };
  for (const auto& path : paths) {
    if (std::filesystem::exists(path)) {
      return path;
    }
  }
  return "tests/fixtures/diagnostics";
}

void TestCpuBandwidthCorrectness() {
  mbufo::diagnostics::BandwidthOptions opts;
  opts.backends = {"cpu"};
  opts.warmup = 1;
  opts.repetitions = 3;
  opts.duration_ms = 0;
  opts.working_set_bytes = 16ULL * 1024ULL * 1024ULL;  // 16 MiB for fast test

  const auto results = mbufo::diagnostics::MeasureCpuBandwidth(opts);
  Expect(results.size() == 3,
         "Expected 3 CPU bandwidth paths (copy, read, write)");

  for (const auto& r : results) {
    Expect(r.backend == "cpu", "Backend is cpu");
    Expect(r.sentinel_verified, "Sentinel bytes verified");
    Expect(r.status == "completed", "Status is completed");
    Expect(r.raw_repetitions_gbps.size() == 3, "Recorded 3 repetitions");
    Expect(r.median_gbps > 0.0, "Median GB/s > 0");
    Expect(r.min_gbps <= r.median_gbps, "min <= median");
    Expect(r.median_gbps <= r.max_gbps, "median <= max");
    Expect(r.p95_gbps >= r.median_gbps, "p95 >= median");
  }
}

void TestBandwidthReportJsonStructure() {
  const auto inv = mbufo::diagnostics::CollectSystemInventory();
  const auto fp = mbufo::diagnostics::GenerateMachineFingerprint(inv);

  mbufo::diagnostics::BandwidthOptions opts;
  opts.backends = {"cpu"};
  opts.warmup = 1;
  opts.repetitions = 2;
  opts.duration_ms = 0;
  opts.working_set_bytes = 8ULL * 1024ULL * 1024ULL;

  const auto report = mbufo::diagnostics::RunBandwidthBenchmark(opts, fp);
  const std::string json = report.ToJson();

  Expect(json.find("\"schemaVersion\": \"1.0.0\"") != std::string::npos,
         "Has schemaVersion");
  Expect(json.find("\"fingerprintId\": \"" + fp.fingerprint_id + "\"") !=
             std::string::npos,
         "Has matching fingerprintId");
  Expect(json.find("\"paths\":") != std::string::npos, "Has paths array");
  Expect(json.find("\"cpu_copy\"") != std::string::npos, "Has cpu_copy path");
  Expect(json.find("\"rawRepetitionsGbps\":") != std::string::npos,
         "Has rawRepetitionsGbps");

  // Validate artifact
  const auto val_res = mbufo::diagnostics::ValidateArtifactContent(json);
  Expect(val_res.is_valid, "BandwidthReport JSON is valid artifact");
}

void TestBandwidthDuration() {
  mbufo::diagnostics::BandwidthOptions opts;
  opts.backends = {"cpu"};
  opts.working_set_bytes = 1 << 20;
  opts.warmup = 1;
  opts.repetitions = 2;
  for (const auto duration_ms : {1U, 30U}) {
    opts.duration_ms = duration_ms;
    for (const auto& path : mbufo::diagnostics::MeasureCpuBandwidth(opts)) {
      Expect(path.elapsed_ms >= duration_ms, "requested duration is measured");
      Expect(path.repetitions >= opts.repetitions &&
                 path.raw_repetitions_gbps.size() == path.repetitions,
             "artifact reports actual iteration count");
      Expect(path.sentinel_verified && path.median_gbps > 0,
             "sustained transfers retain correctness");
    }
  }
}

void TestArtifactValidation() {
  using mbufo::diagnostics::ValidateArtifactContent;
  mbufo::diagnostics::MachineFingerprint fingerprint;
  fingerprint.canonical.cpu_model = "a \"quoted\" CPU";
  fingerprint.fingerprint_id = fingerprint.canonical.ComputeFingerprintId();
  const auto valid = fingerprint.ToJson();
  Expect(ValidateArtifactContent(valid).is_valid,
         "escaped canonical fields validate");
  for (const auto& text : {valid + "garbage", valid.substr(0, valid.size() - 3),
                           std::string("[]"), std::string("null")})
    Expect(!ValidateArtifactContent(text).is_valid,
           "malformed artifact rejected");
  auto root = mbufo::json::parse(valid);
  root["fingerprintId"] = std::string(64, 'z');
  Expect(!ValidateArtifactContent(root.dump()).is_valid,
         "non-hex SHA rejected");
  root = mbufo::json::parse(valid);
  root["schemaVersion"] = 1;
  Expect(!ValidateArtifactContent(root.dump()).is_valid, "wrong type rejected");
  root = mbufo::json::parse(valid);
  root["canonical"]["cpuLogicalCores"] = -1;
  Expect(!ValidateArtifactContent(root.dump()).is_valid,
         "negative core count rejected");
  root = mbufo::json::parse(valid);
  root["canonical"]["cpuLogicalCores"] = 4294967296.0;
  Expect(!ValidateArtifactContent(root.dump()).is_valid,
         "overflowing core count rejected");
  root = mbufo::json::parse(valid);
  root["cpuModel"] = "spoofed value outside canonical";
  Expect(ValidateArtifactContent(root.dump()).is_valid,
         "only canonical fields are hashed");
  root["canonical"]["cpuModel"] = "tampered";
  Expect(!ValidateArtifactContent(root.dump()).is_valid,
         "tampered canonical field rejected");
  root = mbufo::json::Value::object();
  root["nested"] = mbufo::json::parse(valid);
  Expect(!ValidateArtifactContent(root.dump()).is_valid,
         "nested required fields rejected");
  root = mbufo::json::parse(valid);
  root.append_member("fingerprintId", fingerprint.fingerprint_id);
  Expect(!ValidateArtifactContent(root.dump()).is_valid,
         "duplicate key rejected");
  root = mbufo::json::parse(valid);
  root["canonical"] = "not an object";
  Expect(!ValidateArtifactContent(root.dump()).is_valid,
         "canonical must be an object");
  root = mbufo::json::Value::object();
  root["schemaVersion"] = "1.0.0";
  root["fingerprintId"] = fingerprint.fingerprint_id;
  Expect(!ValidateArtifactContent(root.dump()).is_valid,
         "a schema and hash alone are not an artifact");
  root = mbufo::json::parse(valid);
  root["artifactType"] = "unknown";
  Expect(!ValidateArtifactContent(root.dump()).is_valid,
         "unknown artifact type rejected");
}

void TestSchemaFileExists() {
  const auto schema_path = FindFixturesRoot() / "bandwidth_schema.json";
  Expect(std::filesystem::exists(schema_path),
         "bandwidth_schema.json fixture exists");
  std::ifstream f(schema_path);
  Expect(f.is_open(), "Can open bandwidth_schema.json");
  std::string line;
  std::getline(f, line);
  Expect(!line.empty(), "bandwidth_schema.json is non-empty");
}

}  // namespace

int main() {
  std::cout << "Running memory bandwidth benchmark test suite...\n";

  TestCpuBandwidthCorrectness();
  TestBandwidthDuration();
  TestBandwidthReportJsonStructure();
  TestSchemaFileExists();
  TestArtifactValidation();

  std::cout << "All memory bandwidth tests passed successfully.\n";
  return 0;
}
