#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

// Quant block layouts (Q8_0 / Q5_K / Q6_K / Q8_K) and the quant helpers are
// shared via src/models/qwen/hip/quant_ops.hpp.
void LaunchBatchedEmbeddingLookup(const void* table, core::GgmlType type,
                                  const std::uint32_t* token_ids,
                                  float* out_hidden, std::size_t batch_size,
                                  std::size_t hidden_size, hipStream_t stream) {
  // Shared core gather (one decoder per format in core::quant); an unlisted format aborts.
  if (!core::compute::gemm::EmbedGather(table, static_cast<int>(type),
                                        reinterpret_cast<const std::int32_t*>(token_ids), out_hidden, batch_size,
                                        hidden_size, stream)) {
    std::fprintf(stderr, "[qwen] unsupported token_embd type %d (hidden %zu)\n", static_cast<int>(type), hidden_size);
    std::abort();
  }
}
}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
