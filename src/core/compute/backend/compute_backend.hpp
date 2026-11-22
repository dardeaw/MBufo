#pragma once
// P0 shared decoupled compute backend (header-only, zero wiring).
//
// Goal: model code (ASR/TTS/DiT/TE) programs against this interface, never
// against HIP or NPU directly. Future devices add a new impl, models stay
// untouched.
//
//   - HipBackend  -> wraps existing HIP GEMM/attention kernels (GPU path).
//   - NpuBackend  -> wraps NpuQuantGemmEngine + ApplyGqaRopeAttn (NPU path).
//
// Both impls live next to this header later; today this header only fixes
// the seam so P1-P5 can be built against it.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mbufo::core::backend {

enum class DeviceKind : std::uint8_t { kHipGpu = 0, kNpu = 1, kCpuRef = 2 };

struct GemmSpec {
  std::size_t M{0};  ///< rows of A / C (M>=1; M==1 is token decode)
  std::size_t N{0};  ///< cols of weight / C
  std::size_t K{0};  ///< reduction dim
};

/// Minimal GEMM + norm + attention surface shared by all four NPU targets.
/// Weights are opaque handles owned by the backend (packed NPU blocks on NPU,
/// raw pointers on HIP); models only pass ids.
class ComputeBackend {
 public:
  virtual ~ComputeBackend() = default;
  [[nodiscard]] virtual DeviceKind device() const noexcept = 0;
  [[nodiscard]] virtual bool ready(std::string* error = nullptr) const noexcept = 0;
  /// C[M,N] = A[M,K] @ W + bias. Must accept M>=1 (see NpuQuantGemmEngine).
  [[nodiscard]] virtual bool Gemm(std::span<const float> A, const GemmSpec& spec,
                                  std::uint64_t weight_id, std::span<const float> bias,
                                  std::span<float> C, double* ms_accum = nullptr,
                                  std::string* error = nullptr) = 0;
  virtual void RmsNorm(std::span<const float> x, std::span<const float> w, std::size_t rows,
                       std::size_t dim, float eps, std::span<float> y) = 0;
  virtual void LayerNorm(std::span<const float> x, std::span<const float> w,
                         std::span<const float> b, std::size_t rows, std::size_t dim,
                         float eps, std::span<float> y) = 0;
};

/// Router key shared by every model seam: (lane, seq/shape, residency).
struct RouteAsk {
  std::size_t seq_len{0};
  bool weights_resident{false};
  bool device_ready{false};
};

}  // namespace mbufo::core::backend
