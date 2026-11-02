// Unified varlen mixed-batch attention interface (GGUF plan §4.1).
//
// One launch covers any mix of decode rows (n_rows == 1) and prefill-chunk
// rows (n_rows == 64) from independent sessions. F16 batches fuse in one
// generic varlen kernel (per-(block, head, row) FP32 online softmax with
// optional sliding window and fused sigmoid output gate); blocked-Q8/Q4
// rows dispatch per block to the shared split-K (narrow) / WMMA (wide)
// kernels. A batch with mixed F16/blocked rows is refused explicitly (the
// caller falls back to legacy per-item cores).
//
// §4.2 compliance: the launcher is stateless. All device scratch (device
// copy of the descriptor array, split-K partials) comes from the caller
// `scratch` arena. The implementation keeps no `static` buffers.
//
// Gate iron rule (docs/MIXED_BATCH.md): when `gate != nullptr` the kernel
// applies `sigmoid(gate)` in its epilogue and the caller MUST NOT run a
// second SigmoidMul over the output.
#ifndef GUFO_CORE_COMPUTE_ATTENTION_MIXED_BATCH_ATTN_HPP_
#define GUFO_CORE_COMPUTE_ATTENTION_MIXED_BATCH_ATTN_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::core::compute::mixed_batch {

// Per-side KV element codec. kQ4 selects the 18-byte packed-nibble grid
// (GUFO_KV_Q4/V4); it rides the same blocked kernels as kQ8 with
// key/value_block_bytes == 18. Blocked Fp8 (E4M3) has no kernel and is
// refused by the launcher.
enum class MixedKvCodec : std::uint32_t { kF16 = 0, kQ8 = 1, kQ4 = 2 };

// One session's rows inside a mixed batch. All pointers are device pointers;
// the descriptor array itself lives on the host and is uploaded to the
// caller scratch by the launcher.
struct MixedBlk {
  const float* q{nullptr};  // [n_rows][q_stride] post-norm/RoPE query rows
  const void* k{nullptr};   // KV cache base (F16: __half rows; blocked: codec blocks)
  const void* v{nullptr};
  const float* gate{nullptr};  // [n_rows][gate_stride] fused output gate, or
                               // nullptr when the layer has none
  float* out{nullptr};         // [n_rows][out_stride] attention output rows
  std::uint32_t kv_len{0};     // visible key count for this session
  std::uint32_t q_pos_base{0};  // absolute position of row 0
  std::uint32_t n_rows{0};      // decode == 1, prefill_chunk == 64
  MixedKvCodec k_codec{MixedKvCodec::kF16};
  MixedKvCodec v_codec{MixedKvCodec::kF16};
  std::uint32_t head_dim{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  float score_scale{0.0F};  // 0 selects the legacy 1/sqrt(head_dim)
  std::uint32_t q_stride{0};     // float elems between q rows (0 = packed)
  std::uint32_t k_stride{0};     // F16: __half elems between k rows
                                 // (0 = kv_heads*head_dim packed);
                                 // blocked: bytes between rows (0 = packed)
  std::uint32_t v_stride{0};     // same convention as k_stride
  std::uint32_t k_head_stride{0};  // F16 elems between KV heads at the same
                                   // position (0 = head_dim packed within
                                   // the row: covers [pos][kvdim] caches and
                                   // [layer][head][pos][d] planes alike)
  std::uint32_t v_head_stride{0};  // same convention as k_head_stride
  std::uint32_t gate_stride{0};  // float elems between gate rows
                                 // (0 = heads*head_dim packed)
  std::uint32_t out_stride{0};   // float elems between out rows
                                 // (0 = heads*head_dim packed)
  std::uint32_t window{0};  // 0 = dense causal; else sliding window (keys
                            // older than qpos+1-window are masked out)
  std::uint32_t key_block_bytes{34};    // blocked sides: 34 (Q8) or 18 (Q4)
  std::uint32_t value_block_bytes{34};
};

// Upper bound of rows per block (decode and prefill-chunk shapes).
inline constexpr std::uint32_t kMaxMixedRows = 1024;
// Upper bound of sessions per launch (matches the flash-next batch envelope).
inline constexpr std::uint32_t kMaxMixedBlks = 32;

// Worst-case device scratch for a launch over `n_blks` descriptors: the
// device descriptor array, one device position scalar per block, and the
// split-K partials. F16-only callers may use MixedBatchF16ScratchBytes.
std::size_t MixedBatchScratchBytes(std::size_t n_blks);
inline std::size_t MixedBatchF16ScratchBytes(std::size_t n_blks) {
  return n_blks * (sizeof(MixedBlk) + sizeof(std::uint32_t));
}

// Varlen dispatch over `blks` on `stream` (see header contract). Returns
// true when the batch ran; returns false — launching nothing — on any
// refusal (mixed codecs in one call, geometry mismatch across blocks,
// unsupported shape, null/short scratch). Refusal is not an error: the
// caller runs its legacy per-item cores instead.
bool LaunchMixedBatchAttention(const MixedBlk* blks, std::size_t n_blks,
                               void* scratch, std::size_t scratch_bytes,
                               hipStream_t stream);

}  // namespace gufo::core::compute::mixed_batch

#endif  // GUFO_CORE_COMPUTE_ATTENTION_MIXED_BATCH_ATTN_HPP_
