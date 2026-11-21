#pragma once
// NPU kernel registry (descriptors + constraint validation + CPU refs).
//
// Split of responsibilities (this is the "kernels" half of the NPU core):
//   - C++ side (this file + npu_kernels_attn.hpp): WHAT can run — op
//     descriptors, shape/dtype/L1/DMA validation against the hardware rules
//     established in E0..E0.7, CPU reference math for numerics goldens, and a
//     registry of IRON builder assets. Zero hardware dependency: unit-testable.
//   - Python side (assets/*.py): HOW it runs — the actual IRON @iron.jit
//     designs compiled by aiecc into xclbins, registered in NpuXclbinTable.
//
// Verified-vs-probed status is explicit per op so nobody mistakes a probed
// signature for a ran-on-silicon result.
//
// Note on aie2p bf16 matmul rounding:
//   patches/aie_linalg_aie2p_conv_even.patch fixes aie/iron/kernels/linalg.py
//   where non-emulated bf16 output mm() on aie2p skipped `setup=conv_even`
//   and inherited floor rounding (-0.158 negative bias).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
  kMha,          ///< Multi-head / GQA attention pipeline (e07_attn.py)
  kRmsNorm,      ///< rms_norm_eps bf16 row norm
  kSoftmax,      ///< Row softmax + optional sliding window mask
  kMhaSoftmax,   ///< SDK linalg.mha_softmax (partial_softmax 64x64)
  kRope,         ///< datamovement.rope (interleaved or rope_two_halves)
  kQ4nxDequant,  ///< quant.q4nx_dequant (q4nx -> bfp16ebs8 B-operand)
};

/// Evidence level. kVerified ran E2E on this XDNA2; kSdkOnly was probed from
/// SDK headers/examples but never executed here; kPlanned is interface-only.
enum class NpuOpStatus : std::uint8_t { kVerified = 0, kSdkOnly, kPlanned };

enum class NpuDType : std::uint8_t {
  kBf16 = 0,
  kInt16,
  kFp16,
  kBfp16Ebs8,
  kQ4Nx,
};

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
      {NpuOp::kGemm, "mm", NpuOpStatus::kVerified, "e06_ffn.py",
       "E0 int16 64x64x64 3.34 GF; E0.6 bf16 (conv_even aie2p patched)"},
      {NpuOp::kLayerNorm, "layer_norm", NpuOpStatus::kVerified, "e05_chain.py",
       "E0.5: bf16 rows, max_abs_err 0.013 vs CPU"},
      {NpuOp::kRmsNorm, "rms_norm_eps", NpuOpStatus::kVerified, "e06_ffn.py",
       "E0.6/E0.7: bf16 rows eps=1e-6, fused in 24L FFN/Attn"},
      {NpuOp::kGelu, "gelu_sized", NpuOpStatus::kVerified, "e05_chain.py",
       "E0.5/E0.6: 64-elem tanh GELU, 24L FFN 3.02ms max_abs=0.125"},
      {NpuOp::kAdd, "add_sized", NpuOpStatus::kVerified, "e06_ffn.py",
       "E0.6/E0.7: roles (In,In,Out,Param); residual & split-K reduction"},
      {NpuOp::kSoftmax, "softmax", NpuOpStatus::kVerified, "e07_attn.py",
       "E0.7: bf16 row softmax + bidirectional SWA mask, cos=0.99985"},
      {NpuOp::kMha, "mha", NpuOpStatus::kVerified, "e07_attn.py",
       "E0.7: bidirectional unscaled GQA (4/2 d=256 W=512, 4/1 d=512)"},
      {NpuOp::kRope, "datamovement.rope", NpuOpStatus::kVerified, "e07_attn.py",
       "E0.7: rope_two_halves d=256/512 bf16, max_abs=0.0000 on NPU2"},
      {NpuOp::kMhaSoftmax, "linalg.mha_softmax", NpuOpStatus::kSdkOnly, nullptr,
       "SDK causal online partial_softmax 64x64 (roles In,Out,InOut,Param*6)"},
      {NpuOp::kQ4nxDequant, "quant.q4nx_dequant", NpuOpStatus::kSdkOnly,
       nullptr, "AIE2P-only q4nx -> bfp16ebs8 dequant (32K & 64K m_tile)"},
      {NpuOp::kFusedMmGelu, "fused_mm", NpuOpStatus::kSdkOnly, nullptr,
       "epilogue=gelu, bfp16ebs8 prepacked-B (flm embedding format)"},
  };
}

/// Hardware rules (E0/E0.5/E0.6/E0.7 verified unless noted). Empty string = valid.
[[nodiscard]] inline std::string ValidateGemm(std::size_t dim_m,
                                              std::size_t dim_k,
                                              std::size_t dim_n,
                                              NpuDType dtype) {
  if (dtype != NpuDType::kBf16 && dtype != NpuDType::kInt16 &&
      dtype != NpuDType::kBfp16Ebs8) {
    return "gemm dtype must be bf16 (E0.5), bfp16ebs8 (E0.6/flm), or int16 (E0)";
  }
  if (dim_m == 0 || dim_k == 0 || dim_n == 0) {
    return "gemm dims must be nonzero";
  }
  if (dim_m % 4 != 0) {
    return "dim_m must be divisible by 4 (mac_dims row tile)";
  }
  if (dtype == NpuDType::kBfp16Ebs8 && ((dim_k % 8 != 0) || (dim_n % 8 != 0))) {
    return "bfp16ebs8 gemm requires dim_k and dim_n divisible by 8";
  }
  return {};
}

/// Elementwise row ops (layer_norm, rms_norm_eps, gelu_sized, add_sized, softmax, rope).
[[nodiscard]] inline std::string ValidateVectorOp(NpuOp op,
                                                  std::size_t elems) {
  if (op != NpuOp::kLayerNorm && op != NpuOp::kRmsNorm && op != NpuOp::kGelu &&
      op != NpuOp::kAdd && op != NpuOp::kSoftmax && op != NpuOp::kRope) {
    return "not a vector op";
  }
  if (elems == 0) return "vector op needs nonzero elems";
  if (elems % 8 != 0) {
    return "vector tile must be a multiple of 8 elems (E0.5: 8 and 64)";
  }
  if ((op == NpuOp::kRmsNorm || op == NpuOp::kSoftmax) && (elems % 16 != 0)) {
    return "rms_norm_eps / softmax tile must be a multiple of 16 elems";
  }
  if (op == NpuOp::kRope && (elems % 16 != 0)) {
    return "rope row width must be a multiple of 16 elems (32 for two_halves)";
  }
  return {};
}

/// Validates datamovement.rope row alignment per layout (two_halves vs interleaved).
[[nodiscard]] inline std::string ValidateRope(std::size_t head_dim,
                                              bool two_halves) {
  if (head_dim == 0) return "rope head_dim must be nonzero";
  const std::size_t align = two_halves ? 32u : 16u;
  if (head_dim % align != 0) {
    return two_halves
               ? "rope_two_halves requires head_dim divisible by 32 (32B-aligned halves)"
               : "rope requires head_dim divisible by 16";
  }
  return {};
}

/// Byte sizes for one AIE2P q4nx_dequant tile.
struct Q4nxPackLayout {
  std::size_t scales_count{0};
  std::size_t input_bytes{0};
  std::size_t output_bytes{0};
  std::size_t total_l1_bytes{0};
};

[[nodiscard]] inline Q4nxPackLayout ComputeQ4nxPackLayout(
    std::size_t m_tile, std::size_t k_tile, std::size_t group) noexcept {
  Q4nxPackLayout layout{};
  if (group == 0) return layout;
  layout.scales_count = (m_tile * k_tile) / group;
  layout.input_bytes = (m_tile * k_tile) / 2u + 4u * layout.scales_count;
  layout.output_bytes = (m_tile * k_tile * 9u) / 8u;
  layout.total_l1_bytes = layout.input_bytes + layout.output_bytes;
  return layout;
}

/// Validates quant.q4nx_dequant tile geometry (supports standard m_tile=32 AND
/// upgraded 64K-capable m_tile=64 tiles within CoreTile L1 SRAM budget).
/// Mirrors aie/iron/kernels/quant.py::_geometry + AIE2P L1 budget check.
[[nodiscard]] inline std::string ValidateQ4nxPack(
    std::size_t m_tile, std::size_t k_tile, std::size_t group = 32,
    std::size_t ct_k = 128, std::size_t s = 8, std::size_t t = 8,
    std::size_t l1_budget_bytes = 65536,
    Q4nxPackLayout* out_layout = nullptr) {
  if (m_tile == 0 || k_tile == 0 || group == 0 || ct_k == 0 || s == 0 ||
      t == 0) {
    return "q4nx_dequant: all geometry parameters must be positive integers";
  }
  if (s != 8 || t != 8) {
    return "q4nx_dequant: s and t must both be 8";
  }
  if (m_tile % 16 != 0) {
    return "q4nx_dequant: m_tile must be a multiple of 16 (supports 32, 64, ...)";
  }
  if (ct_k % s != 0) {
    return "q4nx_dequant: ct_k must be a multiple of s (8)";
  }
  if (group % s != 0) {
    return "q4nx_dequant: group must be a multiple of s (8)";
  }
  if (k_tile % ct_k != 0) {
    return "q4nx_dequant: k_tile must be a multiple of ct_k";
  }
  if (k_tile % group != 0) {
    return "q4nx_dequant: k_tile must be a multiple of group";
  }
  if (m_tile > std::numeric_limits<std::size_t>::max() / k_tile ||
      (m_tile * k_tile) > std::numeric_limits<std::size_t>::max() / 9u) {
    return "q4nx_dequant: tile size overflow";
  }
  const Q4nxPackLayout layout = ComputeQ4nxPackLayout(m_tile, k_tile, group);
  if (layout.output_bytes >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return "q4nx_dequant: geometry exceeds kernel 32-bit int offsets";
  }
  if (l1_budget_bytes > 0 && layout.total_l1_bytes > l1_budget_bytes) {
    return "q4nx_dequant: input+output tile (" +
           std::to_string(layout.total_l1_bytes) + " B) exceeds L1 budget (" +
           std::to_string(l1_budget_bytes) + " B)";
  }
  if (out_layout != nullptr) *out_layout = layout;
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

/// Row-wise RMSNorm (matches Gemma / EmbeddingGemma2 rms_norm_eps, eps=1e-6).
inline void RefRmsNormRows(const std::vector<float>& x,
                           const std::vector<float>* weight, std::size_t rows,
                           std::size_t cols, double eps,
                           std::vector<float>& out) {
  out.resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    double sq_sum = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      const double v = x[r * cols + c];
      sq_sum += v * v;
    }
    const double inv = 1.0 / std::sqrt(sq_sum / static_cast<double>(cols) + eps);
    for (std::size_t c = 0; c < cols; ++c) {
      const double w =
          (weight != nullptr && c < weight->size()) ? (*weight)[c] : 1.0;
      out[r * cols + c] = static_cast<float>(x[r * cols + c] * inv * w);
    }
  }
}

inline void RefRmsNormRows(const std::vector<float>& x, std::size_t rows,
                           std::size_t cols, double eps,
                           std::vector<float>& out) {
  RefRmsNormRows(x, nullptr, rows, cols, eps, out);
}

/// Numerically stable row-wise softmax (matches IRON softmax reference).
inline void RefSoftmaxRows(const std::vector<float>& x, std::size_t rows,
                           std::size_t cols, std::vector<float>& out) {
  out.resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    float max_v = x[r * cols];
    for (std::size_t c = 1; c < cols; ++c) {
      max_v = std::max(max_v, x[r * cols + c]);
    }
    double sum = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      const float v = x[r * cols + c];
      const double e = (v <= -1.0e8f) ? 0.0 : std::exp(static_cast<double>(v - max_v));
      out[r * cols + c] = static_cast<float>(e);
      sum += e;
    }
    const double inv = (sum > 0.0) ? (1.0 / sum) : 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      out[r * cols + c] = static_cast<float>(out[r * cols + c] * inv);
    }
  }
}

/// HuggingFace-style two-halves RoPE using interleaved (cos, sin) LUT rows
/// (exact match to aie.iron.kernels.datamovement.rope_ref(..., two_halves=True)).
inline void RefRopeTwoHalves(const std::vector<float>& x,
                             const std::vector<float>& cos_sin_lut,
                             std::size_t rows, std::size_t head_dim,
                             std::vector<float>& out) {
  out.resize(rows * head_dim);
  const std::size_t half = head_dim / 2u;
  for (std::size_t r = 0; r < rows; ++r) {
    const std::size_t base = r * head_dim;
    for (std::size_t h = 0; h < half; ++h) {
      const float x1 = x[base + h];
      const float x2 = x[base + half + h];
      const float c = cos_sin_lut[base + 2u * h];
      const float s = cos_sin_lut[base + 2u * h + 1u];
      out[base + h] = x1 * c - x2 * s;
      out[base + half + h] = x2 * c + x1 * s;
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
