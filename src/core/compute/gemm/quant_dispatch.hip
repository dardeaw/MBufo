// Type-dispatched GEMV (decode) and GEMM (prefill) entry points over core::quant::AllFormats.
//
// Models call these instead of choosing between GemvBatched / LaunchQ8KBlockGEMV / DenseF16Gemm /
// UnquantizedF16Gemm themselves. Each format declares how it is served (Traits::kDecode,
// Traits::kPrefill); the switches below have no default, so a new DecodePath / PrefillPath
// enumerator does not compile until handled here, and a format code that is not listed returns
// Status::kUnsupportedType instead of being read as another format.
#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstdlib>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/quant/quant_traits.hpp"

namespace gufo::core::compute::gemm {
namespace q = ::gufo::core::quant;

q::Status GemvDecode(const void* w, int type, const float* x, float* y, std::size_t n_tokens, std::size_t m,
                     std::size_t k, void* preq_scratch, std::size_t preq_bytes, hipStream_t stream) {
  if (w == nullptr || x == nullptr || y == nullptr || m == 0 || k == 0 || n_tokens == 0 || n_tokens > kQuantGemvGroupedMaxTokens)
    return q::Status::kBadShape;
  q::Status st = q::Status::kUnsupportedType;
  const bool known = q::AllFormats::Dispatch(type, [&](auto fm) {
    using F = decltype(fm);
    if (!F::ColsOk(k)) {
      st = q::Status::kBadShape;
      return;
    }
    // Only the K-quant block path has a multi-token kernel (up to kQuantGemvMultiMaxTokens); the other paths serve at
    // most 8 tokens and report kUnsupportedType above that so the caller takes its dense batched route.
    if (n_tokens > 8 && F::kDecode != q::DecodePath::kBlockGemv) {
      st = q::Status::kUnsupportedType;
      return;
    }
    switch (F::kDecode) {  // no default: every DecodePath must be handled
      case q::DecodePath::kNarrowQ8:
        if (DecideNarrowDecodePrequantRoute(/*weight_q8_0=*/true, /*gated=*/false, n_tokens, k, m) ==
                NarrowDecodeRoute::kNarrowQ80Prequant &&
            preq_scratch != nullptr &&
            LaunchNarrowQ80Prequant(w, x, y, k, m, n_tokens, preq_scratch, preq_bytes, stream)) {
          st = q::Status::kOk;
          break;
        }
        GemvBatched(w, type, F::RowBytes(k), k, x, y, static_cast<std::uint32_t>(n_tokens),
                    static_cast<std::uint32_t>(m), 0.0F, stream);
        st = q::Status::kOk;
        break;
      case q::DecodePath::kBlockGemv:
        // One token: rows share a single load of each activation slice (GUFO_NO_SINGLE_GEMV=1 keeps the block GEMV).
        if (n_tokens == 1 && std::getenv("GUFO_NO_SINGLE_GEMV") == nullptr && QuantGemvSingle(w, type, x, y, m, k, stream)) {
          st = q::Status::kOk;
          break;
        }
        // 2..192 tokens: read every weight once per group of <= 32 tokens (the loop below re-reads the matrix per token).
        // GUFO_NO_MULTI_GEMV=1 keeps the per-token loop for A/B runs.
        if (n_tokens >= 2 && std::getenv("GUFO_NO_MULTI_GEMV") == nullptr) {
          const std::size_t groups = (n_tokens + kQuantGemvMultiMaxTokens - 1) / kQuantGemvMultiMaxTokens;
          const std::size_t per = (n_tokens + groups - 1) / groups;
          std::size_t done = 0;
          bool served = true;
          while (done < n_tokens) {
            const std::size_t cnt = std::min(per, n_tokens - done);
            if (!QuantGemvMulti(w, type, x + done * k, y + done * m, cnt, m, k, stream)) {
              served = false;
              break;
            }
            done += cnt;
          }
          if (served) {
            st = q::Status::kOk;
            break;
          }
          if (done != 0) {  // a later group was refused after earlier ones ran: cannot happen for one weight shape
            st = q::Status::kBadShape;
            break;
          }
        }
        if (n_tokens > 8) {
          st = q::Status::kUnsupportedType;
          break;
        }
        for (std::size_t t = 0; t < n_tokens; ++t)
          LaunchQ8KBlockGEMV(w, static_cast<::gufo::core::GgmlType>(type), x + t * k, y + t * m, m, k, stream);
        st = q::Status::kOk;
        break;
      case q::DecodePath::kLegacyGemv:
        GemvBatched(w, type, F::RowBytes(k), k, x, y, static_cast<std::uint32_t>(n_tokens),
                    static_cast<std::uint32_t>(m), 0.0F, stream);
        st = q::Status::kOk;
        break;
      case q::DecodePath::kNone:
        st = q::Status::kUnsupportedType;  // caller dequantizes: QuantGemmPrefill works for any batch
        break;
    }
  });
  return known ? st : q::Status::kUnsupportedType;
}

q::Status QuantGemmPrefill(const void* w, int type, const __half* x16, float* y, std::size_t batch,
                           std::size_t m, std::size_t k, __half* wdq_scratch, std::size_t wdq_bytes,
                           hipStream_t stream) {
  if (w == nullptr || x16 == nullptr || y == nullptr || batch == 0 || m == 0 || k == 0)
    return q::Status::kBadShape;
  q::Status st = q::Status::kUnsupportedType;
  const bool known = q::AllFormats::Dispatch(type, [&](auto fm) {
    using F = decltype(fm);
    if (!F::ColsOk(k)) {
      st = q::Status::kBadShape;
      return;
    }
    switch (F::kPrefill) {  // no default: every PrefillPath must be handled
      case q::PrefillPath::kDenseQ8:
        st = DenseF16Gemm(w, x16, y, batch, m, k, stream) ? q::Status::kOk : q::Status::kBadShape;
        break;
      case q::PrefillPath::kPassthroughF16:
        st = UnquantizedF16Gemm(w, x16, y, batch, m, k, stream) ? q::Status::kOk : q::Status::kBadShape;
        break;
      case q::PrefillPath::kDequantF16:
        if (wdq_scratch == nullptr || m * k * sizeof(__half) > wdq_bytes) {
          st = q::Status::kScratchTooSmall;
          break;
        }
        // Wide batches: dequantize into the tiled layout and run the shared WMMA GEMM (3x the row-major kernel's
        // memory efficiency). GUFO_NO_TILED_GEMM=1 keeps the previous route for A/B runs.
        if (batch >= 256 && m % kTiledRows == 0 && k % 32 == 0 && std::getenv("GUFO_NO_TILED_GEMM") == nullptr &&
            DequantRowsF16Tiled(w, type, wdq_scratch, m, k, stream)) {
          st = TiledF16Gemm(wdq_scratch, x16, y, batch, m, k, stream) ? q::Status::kOk : q::Status::kBadShape;
          break;
        }
        if (!DequantRowsF16(w, type, wdq_scratch, m, k, stream)) {
          st = q::Status::kBadShape;
          break;
        }
        st = UnquantizedF16Gemm(wdq_scratch, x16, y, batch, m, k, stream) ? q::Status::kOk
                                                                           : q::Status::kBadShape;
        break;
    }
  });
  return known ? st : q::Status::kUnsupportedType;
}

}  // namespace gufo::core::compute::gemm

#endif  // ENGINE_ENABLE_HIP
