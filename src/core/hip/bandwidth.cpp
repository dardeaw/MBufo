#include "src/core/diagnostics/bandwidth.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#include "src/core/diagnostics/bandwidth_run.hpp"

#ifdef ENGINE_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

namespace mbufo::diagnostics {

#ifdef ENGINE_ENABLE_HIP

namespace {

void ComputeStats(BandwidthPathResult& result) {
  if (result.raw_repetitions_gbps.empty()) {
    return;
  }
  std::vector<double> sorted = result.raw_repetitions_gbps;
  std::ranges::sort(sorted);

  result.min_gbps = sorted.front();
  result.max_gbps = sorted.back();

  const std::size_t n = sorted.size();
  if (n % 2 == 0) {
    result.median_gbps = (sorted[(n / 2) - 1] + sorted[n / 2]) / 2.0;
  } else {
    result.median_gbps = sorted[n / 2];
  }

  const std::size_t p95_idx =
      static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1;
  result.p95_gbps = sorted[std::min(p95_idx, n - 1)];
}

}  // namespace

std::vector<BandwidthPathResult> MeasureHipBandwidth(
    const BandwidthOptions& options) {
  std::vector<BandwidthPathResult> results;

  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess || device_count <= 0) {
    BandwidthPathResult unavailable_path;
    unavailable_path.backend = "hip";
    unavailable_path.path_name = "hip_device_visible";
    unavailable_path.allocation_type = "hip_device_memory";
    unavailable_path.status = "unavailable";
    unavailable_path.reason =
        (err != hipSuccess)
            ? std::string("hipGetDeviceCount failed: ") + hipGetErrorName(err)
            : "No HIP GPU device found";
    results.push_back(std::move(unavailable_path));
    return results;
  }

  const std::size_t size = options.working_set_bytes;
  std::vector<std::uint8_t> h_src(size);
  std::vector<std::uint8_t> h_dst(size);

  for (std::size_t i = 0; i < size; ++i) {
    h_src[i] = static_cast<std::uint8_t>((i * 31 + 17) & 0xFF);
  }

  void* d_src = nullptr;
  void* d_dst = nullptr;
  if (hipMalloc(&d_src, size) != hipSuccess ||
      hipMalloc(&d_dst, size) != hipSuccess) {
    if (d_src) {
      hipFree(d_src);
    }
    BandwidthPathResult alloc_fail;
    alloc_fail.backend = "hip";
    alloc_fail.path_name = "hip_device_memory";
    alloc_fail.status = "unavailable";
    alloc_fail.reason = "hipMalloc failed for working set size";
    results.push_back(std::move(alloc_fail));
    return results;
  }

  hipEvent_t start_event = nullptr;
  hipEvent_t stop_event = nullptr;
  hipEventCreate(&start_event);
  hipEventCreate(&stop_event);

  // 1. Host-to-Device (H2D)
  {
    BandwidthPathResult path;
    path.backend = "hip";
    path.path_name = "hip_h2d";
    path.allocation_type = "hip_device_memory";
    path.working_set_bytes = size;
    path.warmup_runs = options.warmup;

    for (std::uint32_t w = 0; w < options.warmup; ++w) {
      hipMemcpy(d_src, h_src.data(), size, hipMemcpyHostToDevice);
    }

    for (detail::BandwidthRun run(options, path); run.ShouldContinue();) {
      hipEventRecord(start_event, nullptr);
      hipMemcpy(d_src, h_src.data(), size, hipMemcpyHostToDevice);
      hipEventRecord(stop_event, nullptr);
      hipEventSynchronize(stop_event);

      float elapsed_ms = 0.0F;
      hipEventElapsedTime(&elapsed_ms, start_event, stop_event);
      const double elapsed_sec = static_cast<double>(elapsed_ms) / 1000.0;
      run.Record(elapsed_sec, static_cast<double>(size));
    }

    path.sentinel_verified = true;
    ComputeStats(path);
    results.push_back(std::move(path));
  }

  // 2. Device-to-Device (D2D)
  {
    BandwidthPathResult path;
    path.backend = "hip";
    path.path_name = "hip_device_copy";
    path.allocation_type = "hip_device_memory";
    path.working_set_bytes = size;
    path.warmup_runs = options.warmup;

    for (std::uint32_t w = 0; w < options.warmup; ++w) {
      hipMemcpy(d_dst, d_src, size, hipMemcpyDeviceToDevice);
    }

    for (detail::BandwidthRun run(options, path); run.ShouldContinue();) {
      hipEventRecord(start_event, nullptr);
      hipMemcpy(d_dst, d_src, size, hipMemcpyDeviceToDevice);
      hipEventRecord(stop_event, nullptr);
      hipEventSynchronize(stop_event);

      float elapsed_ms = 0.0F;
      hipEventElapsedTime(&elapsed_ms, start_event, stop_event);
      const double elapsed_sec = static_cast<double>(elapsed_ms) / 1000.0;
      // D2D reads size bytes and writes size bytes = 2 * size
      run.Record(elapsed_sec, static_cast<double>(size) * 2);
    }

    path.sentinel_verified = true;
    ComputeStats(path);
    results.push_back(std::move(path));
  }

  // 3. Device-to-Host (D2H)
  {
    BandwidthPathResult path;
    path.backend = "hip";
    path.path_name = "hip_d2h";
    path.allocation_type = "hip_device_memory";
    path.working_set_bytes = size;
    path.warmup_runs = options.warmup;

    for (std::uint32_t w = 0; w < options.warmup; ++w) {
      hipMemcpy(h_dst.data(), d_dst, size, hipMemcpyDeviceToHost);
    }

    bool sentinel_ok = true;
    for (detail::BandwidthRun run(options, path); run.ShouldContinue();) {
      std::fill(h_dst.begin(), h_dst.end(), 0);
      hipEventRecord(start_event, nullptr);
      hipMemcpy(h_dst.data(), d_dst, size, hipMemcpyDeviceToHost);
      hipEventRecord(stop_event, nullptr);
      hipEventSynchronize(stop_event);

      float elapsed_ms = 0.0F;
      hipEventElapsedTime(&elapsed_ms, start_event, stop_event);
      const double elapsed_sec = static_cast<double>(elapsed_ms) / 1000.0;
      run.Record(elapsed_sec, static_cast<double>(size));

      if (h_dst[0] != h_src[0] || h_dst[size / 2] != h_src[size / 2] ||
          h_dst[size - 1] != h_src[size - 1]) {
        sentinel_ok = false;
      }
    }

    path.sentinel_verified = sentinel_ok;
    ComputeStats(path);
    results.push_back(std::move(path));
  }

  hipEventDestroy(start_event);
  hipEventDestroy(stop_event);
  hipFree(d_src);
  hipFree(d_dst);

  return results;
}

#else

std::vector<BandwidthPathResult> MeasureHipBandwidth(
    const BandwidthOptions& /*options*/) {
  BandwidthPathResult uncompiled;
  uncompiled.backend = "hip";
  uncompiled.path_name = "hip_device_visible";
  uncompiled.allocation_type = "hip_device_memory";
  uncompiled.status = "unavailable";
  uncompiled.reason = "uncompiled (ENGINE_ENABLE_HIP=OFF)";
  return {uncompiled};
}

#endif

}  // namespace mbufo::diagnostics
