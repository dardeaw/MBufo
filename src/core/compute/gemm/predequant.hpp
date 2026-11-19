#ifndef GUFO_CORE_COMPUTE_GEMM_PREDEQUANT_HPP_
#define GUFO_CORE_COMPUTE_GEMM_PREDEQUANT_HPP_

// Layer-major prefill support shared by the engines: the weights of the layer being prefilled are dequantized ONCE
// into the tiled F16 layout (gemm.hpp, kTiledRows) and every chunk of the prompt then runs through that layer with
// TiledF16Gemm (chunk-major prefill re-dequantizes the whole model for every chunk). Engines look weights up by
// device pointer; `count == 0` means "not in layer-major mode".
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>

#include "src/core/compute/gemm/gemm.hpp"

namespace gufo::core::compute::gemm {

struct PreDequant {
  static constexpr int kMax = 8;
  const void* dev[kMax]{};
  __half* tiled[kMax]{};
  int count{0};
  [[nodiscard]] __half* Find(const void* d) const noexcept {
    for (int i = 0; i < count; ++i)
      if (dev[i] == d) return tiled[i];
    return nullptr;
  }
};

/// True when a [rows][cols] weight can live in the tiled layout.
[[nodiscard]] inline bool PreDequantEligible(std::size_t rows, std::size_t cols) noexcept {
  return rows != 0 && rows % kTiledRows == 0 && cols != 0 && cols % 32 == 0;
}

/// Dequantizes one weight into `cursor` (advanced by rows * cols elements) and records it. Returns false (nothing
/// recorded) for ineligible shapes, unsupported formats or a full table; the engine then keeps its normal route.
inline bool PreDequantAdd(PreDequant& pd, __half*& cursor, const void* dev, int type, std::size_t rows,
                          std::size_t cols, hipStream_t stream) {
  if (pd.count >= PreDequant::kMax || dev == nullptr || cursor == nullptr || !PreDequantEligible(rows, cols))
    return false;
  if (!DequantRowsF16Tiled(dev, type, cursor, rows, cols, stream)) return false;
  pd.dev[pd.count] = dev;
  pd.tiled[pd.count] = cursor;
  ++pd.count;
  cursor += rows * cols;
  return true;
}

}  // namespace gufo::core::compute::gemm

#endif  // GUFO_CORE_COMPUTE_GEMM_PREDEQUANT_HPP_
