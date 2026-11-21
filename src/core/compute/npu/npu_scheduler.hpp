#pragma once
// NPU run-to-completion scheduler (scaffold v0, header-only, not wired).
//
// Work shape: one-shot Sync prefill — embedding / reranker / D3 decide.
// No decode, no tokens, no KV, no sampling. Phase machine is therefore
// Queued -> Admitted -> Running -> Terminal (no Prefilling/Decoding split,
// unlike TextGenerationScheduler).
//
// Ownership: submitters never touch the device. The future NPU device thread
// pumps AdmitNext() -> MarkRunning() (holding the single NPU device lock) ->
// executes work -> Finish(). Until the device lock + kernels land, Submit()
// returns kDisabled and runs nothing, so behavior is unchanged.
#include <cstdint>
#include <functional>
#include <string>

#include "src/core/compute/npu/npu_admission.hpp"
#include "src/core/compute/npu/npu_lane.hpp"

namespace mbufo::core::npu {

enum class NpuWorkLane : std::uint8_t {
  kEmbedding = 0,
  kReranker,
  kDecide,
};

/// Opaque work descriptor. `bytes_estimate` feeds the admin memory gate later.
struct NpuWorkDesc {
  NpuWorkLane lane{NpuWorkLane::kEmbedding};
  std::size_t seq_len{0};
  std::size_t bytes_estimate{0};
  std::string client{"anonymous"};
};

struct NpuWorkResult {
  bool ok{false};
  /// "npu-disabled" until the device thread + kernels are wired.
  std::string error{"npu-disabled"};
};

/// The executable payload. Runs on the owner (device) thread, never on the
/// submitting thread. Scaffold stage: never invoked.
using NpuWork = std::function<NpuWorkResult()>;

class NpuScheduler {
 public:
  explicit NpuScheduler(mbufo::npu::NpuLaneConfig config = {},
                        NpuAdmissionPolicy policy = {})
      : config_(config), queue_(policy) {}

  NpuScheduler(const NpuScheduler&) = delete;
  NpuScheduler& operator=(const NpuScheduler&) = delete;

  /// Disabled until wired: admits nothing, runs nothing.
  [[nodiscard]] bool enabled() const noexcept {
    return config_.embedding || config_.reranker || config_.decide;
  }

  /// Scaffold: validates lane flag, returns kDisabled without running `work`.
  /// Wiring step will: Submit ticket -> Wait -> owner runs under device lock.
  [[nodiscard]] NpuWorkResult Submit(NpuWorkDesc desc, NpuWork /*work*/) {
    if (!LaneOn(desc.lane)) return NpuWorkResult{};
    // Not wired yet: queue exists so the API is fixed, but nothing pumps it.
    return NpuWorkResult{};
  }

  // ---- future owner-thread pump hooks (declared, unused for now) ----
  [[nodiscard]] std::uint64_t AdmitNext() { return queue_.AdmitNext(); }
  void MarkRunning(std::uint64_t ticket) noexcept {
    queue_.MarkRunning(ticket);
  }
  void Finish(std::uint64_t ticket) noexcept { queue_.Finish(ticket); }
  void SetDeviceLost() noexcept { queue_.SetDeviceLost(); }
  [[nodiscard]] std::size_t pending() const { return queue_.pending(); }

 private:
  [[nodiscard]] bool LaneOn(NpuWorkLane lane) const noexcept {
    switch (lane) {
      case NpuWorkLane::kEmbedding:
        return config_.embedding;
      case NpuWorkLane::kReranker:
        return config_.reranker;
      case NpuWorkLane::kDecide:
        return config_.decide;
    }
    return false;
  }

  mbufo::npu::NpuLaneConfig config_;
  NpuAdmissionQueue queue_;
};

}  // namespace mbufo::core::npu
