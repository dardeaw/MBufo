// Unified varlen mixed-batch attention (GGUF plan §4.1/§4.2).
//
// Dispatch (per call, single codec only — mixed F16/blocked refuses):
//   F16        -> MixedBatchF16Kernel (generic varlen causal kernel below:
//                 one block per (block, head, row), FP32 online softmax,
//                 optional sliding window, optional fused sigmoid gate).
//   blocked Q8/Q4 (SWA geometry) -> per-block dispatch to the shared
//                 blocked_splitk (n_rows <= 8, caller scratch partials) or
//                 blocked_wmma kernels. Cross-session fusion of blocked rows
//                 is follow-up work; the entry point is already unified.
//   anything else -> refuse (return false, launch nothing).
//
// Scratch layout: [device MixedBlk array][device start_pos array]
//   [+ split-K partials]. Sized by MixedBatchScratchBytes().
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "src/core/compute/attention/blocked_splitk.hpp"
#include "src/core/compute/attention/blocked_wmma.hpp"
#include "src/core/compute/attention/mixed_batch_attn.hpp"

namespace gufo::core::compute::mixed_batch {
namespace {

__device__ __forceinline__ float Sigmoid(float x) {
  // __expf like the rest of the family (SigmoidF, SplitK, blocked kernels):
  // bit-consistency with the production paths matters more than the last ulp.
  return 1.0F / (1.0F + __expf(-x));
}

// Block-wide sum over 256 threads; every thread receives the total.
__device__ __forceinline__ float BlockSum256(float v) {
  __shared__ float red[256];
  const unsigned tid = threadIdx.x;
  red[tid] = v;
  __syncthreads();
  for (unsigned s = 128; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  return red[0];
}

// Generic F16 varlen kernel: block (b, h, r) owns row r of head h in block
// b. Q is float, K/V are __half rows; math is FP32 online softmax, matching
// the scalar reference (same summation family as the per-item narrow
// kernels). Gate (when non-null) is fused here — the caller must not run a
// second SigmoidMul (gate iron rule).
template <unsigned kMaxHeadDim>
__global__ void MixedBatchF16Kernel(const MixedBlk* __restrict__ blks,
                                    std::uint32_t n_blks) {
  const std::uint32_t b = blockIdx.x;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t r = blockIdx.z;
  if (b >= n_blks) return;
  const MixedBlk blk = blks[b];
  if (h >= blk.heads || r >= blk.n_rows) return;
  const std::uint32_t d = blk.head_dim;
  const std::uint32_t group = blk.heads / blk.kv_heads;
  const std::uint32_t kh = h / group;
  const unsigned tid = threadIdx.x;

  const std::uint32_t q_stride =
      blk.q_stride != 0 ? blk.q_stride : blk.heads * d;
  const std::uint32_t k_stride =
      blk.k_stride != 0 ? blk.k_stride : blk.kv_heads * d;
  const std::uint32_t v_stride =
      blk.v_stride != 0 ? blk.v_stride : blk.kv_heads * d;
  const std::uint32_t k_head_stride =
      blk.k_head_stride != 0 ? blk.k_head_stride : d;
  const std::uint32_t v_head_stride =
      blk.v_head_stride != 0 ? blk.v_head_stride : d;
  const std::uint32_t gate_stride =
      blk.gate_stride != 0 ? blk.gate_stride : blk.heads * d;
  const std::uint32_t out_stride =
      blk.out_stride != 0 ? blk.out_stride : blk.heads * d;
  const float scale = (blk.score_scale != 0.0F)
                          ? blk.score_scale
                          : (1.0F / sqrtf(static_cast<float>(d)));

  __shared__ float qv[512];
  static_assert(kMaxHeadDim <= 512, "shared Q row bound");
  const float* qrow = blk.q + static_cast<std::size_t>(r) * q_stride +
                      static_cast<std::size_t>(h) * d;
  for (std::uint32_t i = tid; i < d; i += blockDim.x) qv[i] = qrow[i];
  __syncthreads();

  float m = -INFINITY;
  float l = 0.0F;
  float acc[2] = {0.0F, 0.0F};
  const std::uint32_t qpos = blk.q_pos_base + r;
  std::uint32_t k_lo = 0;
  if (blk.window != 0 && qpos + 1 > blk.window) k_lo = qpos + 1 - blk.window;
  const std::uint32_t k_hi = qpos < blk.kv_len ? qpos : blk.kv_len - 1;
  const __half* k_base = static_cast<const __half*>(blk.k);
  const __half* v_base = static_cast<const __half*>(blk.v);
  for (std::uint32_t p = k_lo; p <= k_hi; ++p) {
    const __half* krow = k_base +
                         static_cast<std::size_t>(kh) * k_head_stride +
                         static_cast<std::size_t>(p) * k_stride;
    float dot = 0.0F;
    for (std::uint32_t i = tid; i < d; i += blockDim.x)
      dot += qv[i] * __half2float(krow[i]);
    dot = BlockSum256(dot) * scale;
    const float m_new = fmaxf(m, dot);
    const float a = __expf(m - m_new);
    const float w = __expf(dot - m_new);
    l = l * a + w;
    const __half* vrow = v_base +
                         static_cast<std::size_t>(kh) * v_head_stride +
                         static_cast<std::size_t>(p) * v_stride;
    unsigned k = 0;
    for (std::uint32_t i = tid; i < d; i += blockDim.x, ++k)
      acc[k] = acc[k] * a + w * __half2float(vrow[i]);
    m = m_new;
  }
  const float inv = (l > 0.0F) ? (1.0F / l) : 0.0F;
  float* orow = blk.out + static_cast<std::size_t>(r) * out_stride +
                static_cast<std::size_t>(h) * d;
  unsigned k = 0;
  for (std::uint32_t i = tid; i < d; i += blockDim.x, ++k) {
    float v = acc[k] * inv;
    if (blk.gate != nullptr) {
      const float g =
          blk.gate[static_cast<std::size_t>(r) * gate_stride +
                   static_cast<std::size_t>(h) * d + i];
      v *= Sigmoid(g);
    }
    orow[i] = v;
  }
}

bool IsBlockedCodec(MixedKvCodec c) {
  return c == MixedKvCodec::kQ8 || c == MixedKvCodec::kQ4;
}

std::uint32_t BlockBytesFor(MixedKvCodec c) {
  return c == MixedKvCodec::kQ4 ? 18u : 34u;
}

}  // namespace

std::size_t MixedBatchScratchBytes(std::size_t n_blks) {
  // Device descriptors + device position scalars + worst-case split-K
  // partials (shared across blocks; launches on one stream are serial).
  return n_blks * (sizeof(MixedBlk) + sizeof(std::uint32_t)) +
         blocked_splitk::kBlockedSplitKMaxTokens *
             16ull * blocked_splitk::kBlockedSplitKMaxSplits * (256ull + 2ull) *
             sizeof(float);
}

bool LaunchMixedBatchAttention(const MixedBlk* blks, std::size_t n_blks,
                               void* scratch, std::size_t scratch_bytes,
                               hipStream_t stream) {
  if (blks == nullptr || n_blks == 0 || n_blks > kMaxMixedBlks) {
    std::fprintf(stderr, "[mixed_batch] refuse: bad batch (%zu blks)\n",
                 n_blks);
    return false;
  }
  if (scratch == nullptr ||
      scratch_bytes < MixedBatchF16ScratchBytes(n_blks)) {
    std::fprintf(stderr, "[mixed_batch] refuse: no caller scratch\n");
    return false;
  }
  const MixedBlk& b0 = blks[0];
  for (std::size_t i = 0; i < n_blks; ++i) {
    const MixedBlk& b = blks[i];
    if (b.q == nullptr || b.k == nullptr || b.v == nullptr ||
        b.out == nullptr || b.n_rows == 0 || b.n_rows > kMaxMixedRows ||
        b.kv_len == 0 || b.head_dim == 0 || b.head_dim > 512 ||
        b.heads == 0 || b.kv_heads == 0 || b.heads % b.kv_heads != 0 ||
        b.heads != b0.heads || b.kv_heads != b0.kv_heads ||
        b.head_dim != b0.head_dim) {
      std::fprintf(stderr, "[mixed_batch] refuse: blk %zu geometry\n", i);
      return false;
    }
    if ((b.k_codec == MixedKvCodec::kF16) !=
            (b0.k_codec == MixedKvCodec::kF16) ||
        (b.v_codec == MixedKvCodec::kF16) !=
            (b0.v_codec == MixedKvCodec::kF16)) {
      // Mixed F16/blocked rows in one call: explicit refusal (§4.1).
      std::fprintf(stderr, "[mixed_batch] refuse: mixed codecs in one call\n");
      return false;
    }
  }

  // Upload descriptors + position scalars into the caller arena. The
  // descriptor upload is synchronous: the caller's host array may be a stack
  // buffer that dies on return, and the per-element position scalars below
  // must not outlive their loop iteration either.
  auto* dev_blks = static_cast<MixedBlk*>(scratch);
  auto* dev_pos = reinterpret_cast<std::uint32_t*>(dev_blks + n_blks);
  if (hipMemcpy(dev_blks, blks, n_blks * sizeof(MixedBlk),
                hipMemcpyHostToDevice) != hipSuccess) {
    return false;
  }
  {
    std::uint32_t host_pos[kMaxMixedBlks];
    for (std::size_t i = 0; i < n_blks; ++i) host_pos[i] = blks[i].q_pos_base;
    if (hipMemcpy(dev_pos, host_pos, n_blks * sizeof(std::uint32_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
      return false;
    }
  }
  // Both uploads above are synchronous, so the device copies are complete
  // before any kernel below reads them, on any stream.

  const bool f16 =
      b0.k_codec == MixedKvCodec::kF16 && b0.v_codec == MixedKvCodec::kF16;
  if (f16) {
    std::uint32_t max_heads = 0, max_rows = 0;
    for (std::size_t i = 0; i < n_blks; ++i) {
      if (blks[i].heads > max_heads) max_heads = blks[i].heads;
      if (blks[i].n_rows > max_rows) max_rows = blks[i].n_rows;
    }
    hipLaunchKernelGGL((MixedBatchF16Kernel<512>),
                       dim3(static_cast<unsigned>(n_blks), max_heads, max_rows),
                       dim3(256), 0, stream, dev_blks,
                       static_cast<std::uint32_t>(n_blks));
    return true;
  }

  // Blocked path: per-block dispatch (varlen: small -> split-K scan, large
  // -> WMMA tiling). Requires dense-causal inputs — SWA callers pre-shift
  // the pointers and rebase q_pos_base, exactly like the scalar
  // AttentionBlocked path — and packed row layout.
  if (!IsBlockedCodec(b0.k_codec) || !IsBlockedCodec(b0.v_codec)) {
    std::fprintf(stderr, "[mixed_batch] refuse: unsupported codec\n");
    return false;
  }
  if (BlockBytesFor(b0.k_codec) != b0.key_block_bytes ||
      BlockBytesFor(b0.v_codec) != b0.value_block_bytes) {
    std::fprintf(stderr, "[mixed_batch] refuse: codec/block-bytes mismatch\n");
    return false;
  }
  for (std::size_t i = 0; i < n_blks; ++i) {
    const MixedBlk& b = blks[i];
    if (b.window != 0 || b.gate != nullptr ||
        b.k_codec != b0.k_codec || b.v_codec != b0.v_codec ||
        b.key_block_bytes != b0.key_block_bytes ||
        b.value_block_bytes != b0.value_block_bytes ||
        (b.q_stride != 0 && b.q_stride != b.heads * b.head_dim) ||
        (b.out_stride != 0 && b.out_stride != b.heads * b.head_dim) ||
        (b.k_stride != 0 || b.v_stride != 0)) {
      std::fprintf(stderr, "[mixed_batch] refuse: blk %zu blocked layout\n",
                   i);
      return false;
    }
  }
  auto* partials = reinterpret_cast<float*>(dev_pos + n_blks);
  const std::size_t partials_avail =
      (scratch_bytes > MixedBatchF16ScratchBytes(n_blks))
          ? scratch_bytes - MixedBatchF16ScratchBytes(n_blks)
          : 0;
  for (std::size_t i = 0; i < n_blks; ++i) {
    const MixedBlk& b = blks[i];
    const float scale =
        (b.score_scale != 0.0F) ? b.score_scale : 1.0F;  // blocked kernels
                                                        // default to 1.0
    bool ok = false;
    if (b.n_rows <= blocked_splitk::kBlockedSplitKMaxTokens) {
      ok = blocked_splitk::AttentionBlockedSplitK<
          blocked_splitk::KvCodecQ8, 256>(
          b.q, static_cast<const std::uint8_t*>(b.k),
          static_cast<const std::uint8_t*>(b.v), b.out, b.n_rows, dev_pos + i,
          b.kv_len, b.heads, b.kv_heads, b.head_dim, stream, scale,
          b.key_block_bytes, b.value_block_bytes, partials, partials_avail);
    } else {
      ok = blocked_wmma::AttentionBlockedWmma<blocked_wmma::KvCodecQ8, 256>(
          b.q, static_cast<const std::uint8_t*>(b.k),
          static_cast<const std::uint8_t*>(b.v), b.out, b.n_rows,
          b.q_pos_base, b.heads, b.kv_heads, /*window=*/0, stream, scale,
          b.key_block_bytes, b.value_block_bytes);
    }
    if (!ok) {
      std::fprintf(stderr,
                   "[mixed_batch] refuse: blk %zu blocked dispatch (n=%u)\n", i,
                   b.n_rows);
      return false;
    }
  }
  return true;
}

}  // namespace gufo::core::compute::mixed_batch
#endif  // defined(ENGINE_ENABLE_HIP)
