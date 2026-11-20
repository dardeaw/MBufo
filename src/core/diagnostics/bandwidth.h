#ifndef MBUFO_CORE_DIAGNOSTICS_BANDWIDTH_H_
#define MBUFO_CORE_DIAGNOSTICS_BANDWIDTH_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/diagnostics/fingerprint.h"

namespace mbufo::diagnostics {

struct BandwidthOptions {
  std::vector<std::string> backends{"cpu", "hip"};
  std::uint32_t warmup{3};
  std::uint32_t repetitions{10};  // Minimum measured iterations per path.
  std::uint32_t duration_ms{2000};
  std::size_t working_set_bytes{256ULL * 1024ULL * 1024ULL};  // 256 MiB
};

struct BandwidthPathResult {
  std::string backend;          // "cpu", "hip"
  std::string path_name;        // e.g. "cpu_copy", "hip_h2d"
  std::string allocation_type;  // e.g. "host_pageable", "hip_device_memory"
  std::size_t working_set_bytes{0};
  std::uint32_t warmup_runs{0};
  std::uint64_t repetitions{0};  // Actual measured iterations.
  double elapsed_ms{0.0};        // Measured loop wall time, excluding warmup.
  double median_gbps{0.0};
  double p95_gbps{0.0};
  double min_gbps{0.0};
  double max_gbps{0.0};
  std::vector<double> raw_repetitions_gbps;
  bool sentinel_verified{true};
  std::string status{"completed"};  // "completed", "unsupported", "unavailable"
  std::string reason;
};

struct BandwidthReport {
  std::string schema_version{"1.0.0"};
  std::string fingerprint_id;
  std::string engine_revision{"0.1.0"};
  std::string timestamp;
  BandwidthOptions options;
  std::vector<BandwidthPathResult> paths;

  [[nodiscard]] std::string ToJson() const;
  [[nodiscard]] std::string ToHuman() const;
};

[[nodiscard]] std::vector<BandwidthPathResult> MeasureCpuBandwidth(
    const BandwidthOptions& options);

[[nodiscard]] std::vector<BandwidthPathResult> MeasureHipBandwidth(
    const BandwidthOptions& options);

[[nodiscard]] BandwidthReport RunBandwidthBenchmark(
    const BandwidthOptions& options, const MachineFingerprint& fingerprint);

}  // namespace mbufo::diagnostics

#endif  // MBUFO_CORE_DIAGNOSTICS_BANDWIDTH_H_
