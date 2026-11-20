#if defined(ENGINE_ENABLE_HIP)

#include "src/core/compute/gemm/host_dequant.hpp"

#include <cstdint>
#include <cstdlib>

#include "src/core/quant/ggml_gemm.hpp"
#include "src/core/quant/row_dot_traits.hpp"

namespace mbufo::core::compute::gemm {

bool HostDequantSupported(::mbufo::core::GgmlType type) noexcept {
  return ::mbufo::quant::IsSupported(type) ||
         ::mbufo::core::quant::AllFormats::Dispatch(static_cast<int>(type), [](auto) {});
}

void HostDequantRow(::mbufo::core::GgmlType type, const void* src, float* dst, std::size_t k) {
  if (::mbufo::quant::IsSupported(type)) {
    ::mbufo::quant::Dequantize(type, src, dst, k);
    return;
  }
  const bool done = ::mbufo::core::quant::AllFormats::Dispatch(static_cast<int>(type), [&](auto tr) {
    using Tr = ::mbufo::core::quant::Traits<decltype(tr)::kId>;
    const auto* row = static_cast<const std::uint8_t*>(src);
    for (std::size_t i = 0; i < k; ++i) dst[i] = Tr::Element(row, static_cast<std::uint32_t>(i));
  });
  if (!done) std::abort();
}

}  // namespace mbufo::core::compute::gemm

#endif  // defined(ENGINE_ENABLE_HIP)
