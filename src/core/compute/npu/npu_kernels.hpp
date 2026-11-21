#pragma once
// NPU kernel registry (v0: descriptors + constraint validation + CPU refs).
//
// Split of responsibilities (this is the "kernels" half of the NPU core):
//   - C++ side (this file): WHAT can run — op descriptors, shape/dtype
//     validation against the hardware rules discovered in E0/E0.5, CPU
//     reference math for numerics goldens, and a registry of IRON builder
//     assets. Zero hardware dependency: fully unit-testable.
//   - Python side (assets/*.py, future): HOW it runs — the actual IRON
//     @iron.jit designs compiled by aiecc into xclbins, registered in
//     NpuXclbinTable by name. Builders are referenced here, not embedded.
//
// Verified-vs-probed status is explicit per op so nobody mistakes a probed
// signature for a ran-on-silicon result.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mbufo::core::npu {

/// Compute ops the NPU lane can express.
enum class NpuOp : std::uint8_t {
  kLayerNorm = 0,
  kGemm,
  kGelu,
  kAdd,
  kFusedMmGelu,  ///< GEMM with activation epilogue (fused_mm)
  kMha,
};

/// Evidence level. kVerified ran E2E on this XDNA2; kSdkOnly was probed from
/// SDK headers/examples but never executed here; kPlanned is interface-only.
enum class NpuOpStatus : std::uint8_t { kVerified = 0, kSdkOnly, kPlanned };

enum class NpuDType : std::uint8_t { kBf16 = 0, kInt16, kFp16 };

struct NpuOpInfo {
  NpuOp op;
  const char* name;
  NpuOpStatus status;
  /// IRON builder asset (python design file), relative to npu assets dir.
  const char* builder;
  const char* note;
};

[[nodiscard]] inline std::vector<NpuOpInfo> AvailableOps() {
  return {
      {NpuOp::kGemm, "mm", NpuOpStatus::kVerified, "single_core.py",
       "E0: int16 64x64x64, 3.34 GFLOPS"},
      {NpuOp::kLayerNorm, "layer_norm", NpuOpStatus::kVerified, "e05_chain.py",
       "E0.5: bf16 rows, max_abs_err 0.013 vs CPU"},
      {NpuOp::kGelu, "gelu_sized", NpuOpStatus::kVerified, "e05_chain.py",
       "E0.5: 64-elem chunks chained after LN"},
      {NpuOp::kAdd, "add_sized", NpuOpStatus::kSdkOnly, nullptr,
       "roles (In,In,Out,Param); needs E2E run"},
      {NpuOp::kFusedMmGelu, "fused_mm", NpuOpStatus::kSdkOnly, nullptr,
       "epilogue=gelu, bfp16ebs8 prepacked-B (flm embedding format)"},
      {NpuOp::kMha, "mha", NpuOpStatus::kSdkOnly, nullptr,
       "mac (4,8,8); flash_attn_prefill exists in SDK"},
  };
}

/// Hardware rules (E0/E0.5 verified unless noted). Empty string = valid.
[[nodiscard]] inline std::string ValidateGemm(std::size_t dim_m,
                                              std::size_t dim_k,
                                              std::size_t dim_n,
                                              NpuDType dtype) {
  if (dtype != NpuDType::kBf16 && dtype != NpuDType::kInt16) {
    return "gemm dtype must be bf16 (E0.5) or int16 (E0)";
  }
  if (dim_m == 0 || dim_k == 0 || dim_n == 0) {
    return "gemm dims must be nonzero";
  }
  if (dim_m % 4 != 0) {
    return "dim_m must be divisible by 4 (mac_dims row tile)";
  }
  return {};
}

/// Elementwise row ops (layer_norm 1D tiles, gelu/add sized chunks).
/// E0.5 verified: LN tile 8 over (8,64) bf16 rows; gelu/add chunks of 64.
[[nodiscard]] inline std::string ValidateVectorOp(NpuOp op,
                                                  std::size_t elems) {
  if (op != NpuOp::kLayerNorm && op != NpuOp::kGelu && op != NpuOp::kAdd) {
    return "not a vector op";
  }
  if (elems == 0) return "vector op needs nonzero elems";
  if (elems % 8 != 0) {
    return "vector tile must be a multiple of 8 elems (E0.5: 8 and 64)";
  }
  return {};
}

// ---- CPU reference math (numerics goldens; float, not bit-exact) ----

/// Row-wise layer norm, gamma=1 beta=0 eps=1e-5 (matches IRON layer_norm).
inline void RefLayerNormRows(const std::vector<float>& x, std::size_t rows,
                             std::size_t cols, std::vector<float>& out) {
  out.resize(x.size());
  for (std::size_t r = 0; r < rows; ++r) {
    double mean = 0;
    for (std::size_t c = 0; c < cols; ++c) mean += x[r * cols + c];
    mean /= cols;
    double var = 0;
    for (std::size_t c = 0; c < cols; ++c) {
      const double d = x[r * cols + c] - mean;
      var += d * d;
    }
    var /= cols;
    const double inv = 1.0 / std::sqrt(var + 1e-5);
    for (std::size_t c = 0; c < cols; ++c) {
      out[r * cols + c] =
          static_cast<float>((x[r * cols + c] - mean) * inv);
    }
  }
}

/// gelu tanh approximation (matches IRON gelu kernel family).
inline void RefGeluTanh(const std::vector<float>& x, std::vector<float>& out) {
  out.resize(x.size());
  const double k = std::sqrt(2.0 / 3.141592653589793);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double v = x[i];
    out[i] = static_cast<float>(
        0.5 * v * (1.0 + std::tanh(k * (v + 0.044715 * v * v * v))));
  }
}

inline void RefAdd(const std::vector<float>& a, const std::vector<float>& b,
                   std::vector<float>& out) {
  out.resize(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
}

}  // namespace mbufo::core::npu
