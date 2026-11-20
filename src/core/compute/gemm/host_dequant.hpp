#ifndef MBUFO_CORE_COMPUTE_GEMM_HOST_DEQUANT_HPP_
#define MBUFO_CORE_COMPUTE_GEMM_HOST_DEQUANT_HPP_

// Host-side row dequantization for every format a loader may meet: the formats the row-dot registry knows (Q4_0, Q4_1, Q5_0,
// Q5_1, Q2_K, IQ2_XXS, MXFP4, ROCmFP4 ... decoded through the same Traits every kernel is checked against) plus the host
// dequantizers of quant::Dequantize (Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, Q8_0, IQ4_NL, IQ4_XS, IQ3_S). One call serves any engine that
// converts a packed matrix at load time instead of carrying a kernel for it.

#include <cstddef>

#include "src/core/gguf_reader.hpp"

namespace mbufo::core::compute::gemm {

/// True when HostDequantRow can decode this type.
[[nodiscard]] bool HostDequantSupported(::mbufo::core::GgmlType type) noexcept;

/// Decodes one packed row of `k` elements (k a multiple of the format's block) into F32. Aborts on an unsupported type
/// (check HostDequantSupported first).
void HostDequantRow(::mbufo::core::GgmlType type, const void* src, float* dst, std::size_t k);

}  // namespace mbufo::core::compute::gemm

#endif  // MBUFO_CORE_COMPUTE_GEMM_HOST_DEQUANT_HPP_
