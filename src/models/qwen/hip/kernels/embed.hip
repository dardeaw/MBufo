#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "src/core/gguf_reader.hpp"
#include "src/core/compute/gemm/gemm.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

// All three lookups ride the shared core gather (src/core/compute/gemm/dequant_rows.hip): one decoder per
// format in core::quant, and an unlisted format aborts instead of being decoded as another one.
static void LookupOrDie(bool ok, core::GgmlType type, std::size_t hidden_size) {
  if (!ok) {
    std::fprintf(stderr, "[qwen] unsupported token_embd type %d (hidden %zu)\n", static_cast<int>(type), hidden_size);
    std::abort();
  }
}

void LaunchEmbeddingLookup(const void* table, core::GgmlType type,
                           std::uint32_t token_id, float* out_hidden,
                           std::size_t hidden_size, hipStream_t stream) {
  LookupOrDie(core::compute::gemm::EmbedGatherOne(table, static_cast<int>(type), token_id, out_hidden, hidden_size,
                                                  stream),
              type, hidden_size);
}

void LaunchEmbeddingLookup(const void* table, core::GgmlType type,
                           const std::uint32_t* d_token_id, float* out_hidden,
                           std::size_t hidden_size, hipStream_t stream) {
  LookupOrDie(core::compute::gemm::EmbedGather(table, static_cast<int>(type),
                                               reinterpret_cast<const std::int32_t*>(d_token_id), out_hidden, 1,
                                               hidden_size, stream),
              type, hidden_size);
}

}  // namespace gufo::hip
#endif
