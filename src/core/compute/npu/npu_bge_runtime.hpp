#pragma once
// Real NPU2 runtime for BGE-M3 and bge-reranker-v2-m3 (24L XLM-RoBERTa).
// Executes all Q/K/V/Out/Up/Down 1024x1024 projections on XDNA2 via
// gemm_1024x1024 IRON kernel (4x4 cores), keeping exact host LayerNorm/Attn/GELU.
// Compiled only when MBUFO_ENABLE_NPU=ON.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_device_hrx.hpp"
#include "src/models/bge_m3/forward.hpp"

namespace mbufo::core::npu {

class NpuBgeRuntime {
 public:
  static constexpr std::size_t kChunkRows = 64;
  static constexpr std::size_t kHidden = 1024;
  static constexpr std::size_t kInter = 4096;

  NpuBgeRuntime() = default;
  ~NpuBgeRuntime() = default;
  NpuBgeRuntime(const NpuBgeRuntime&) = delete;
  NpuBgeRuntime& operator=(const NpuBgeRuntime&) = delete;

  /// Opens HRX session, loads `gemm_1024x1024` from `kernel_dir`, and uploads
  /// all 24 layers' transposed BF16 projection weights into resident BOs.
  [[nodiscard]] bool Init(const models::bge_m3::BgeM3F32Weights& weights,
                          const std::string& kernel_dir, std::string* error);

  /// Runs full bidirectional 24-layer encoder forward for `1 <= ids <= 2048`
  /// on NPU2 silicon, writing row 0 (CLS hidden state, 1024 floats) to `cls_out`.
  [[nodiscard]] bool ForwardCls(std::span<const std::uint32_t> ids,
                                const models::bge_m3::BgeM3F32Weights& weights,
                                std::vector<float>* cls_out,
                                double* prefill_ms, std::string* error);

  [[nodiscard]] bool resident() const noexcept { return resident_; }
  [[nodiscard]] const std::string& kernel_dir() const noexcept {
    return kernel_dir_;
  }

 private:
  struct LayerResident {
    std::unique_ptr<hrx::ext::bo> q_bo;
    std::unique_ptr<hrx::ext::bo> k_bo;
    std::unique_ptr<hrx::ext::bo> v_bo;
    std::unique_ptr<hrx::ext::bo> out_bo;
    std::array<std::unique_ptr<hrx::ext::bo>, 4> up_bo;
    std::array<std::unique_ptr<hrx::ext::bo>, 4> down_bo;
  };

  bool resident_{false};
  std::string kernel_dir_;
  std::unique_ptr<HrxSession> session_;
  std::vector<LayerResident> layers_;
  std::array<std::unique_ptr<hrx::ext::bo>, 4> a_bo_;
  std::array<std::unique_ptr<hrx::ext::bo>, 4> c_bo_;
};

}  // namespace mbufo::core::npu
