// Multi-query blocking F16 causal attention for muse-glimmer (gfx1151).
//
// Specialized for d == 128 (Glimmer 30B: 32Q/2KV GQA): one 128-thread block
// serves a tile of 4 query rows of a single head (query tiling). Thread wave
// w (32 threads) owns query qb+w with the exact 32-lane, 4-dims-per-lane
// decomposition and online-softmax update order of AttentionF16Kernel, so
// per-query outputs are bit-identical to the legacy kernel. The 4 queries
// share K/V traffic: each key step stages one K row + one V row (256 B each
// at d == 128) once per block in LDS and all four waves read from LDS; the
// 4 Q rows (2 KiB) are staged once per block the same way. SWA `window`
// semantics match AttentionF16 (0 = dense causal: query t sees keys
// [0, t+start]; nonzero: keys at or below start+t+1-window are masked out).
//
// Entry convention (dispatcher-adjacent): returns true when the call ran;
// returns false -- launching nothing -- for unsupported shapes (non-d128,
// degenerate geometry, tiny calls that stay on the legacy scalar path).
// False is not an error: the caller falls back to AttentionF16.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::glimmer::kernels {
namespace {

constexpr std::uint32_t kMqbQueriesPerBlock = 4;
constexpr std::uint32_t kMqbHeadDim = 128;
constexpr std::uint32_t kMqbMinTokens = 16;

__global__ void AttentionF16MQBKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, float* __restrict__ out,
    std::uint32_t n_tokens, std::uint32_t start_pos, std::uint32_t heads,
    std::uint32_t kv_heads, float score_scale, std::uint32_t window) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t b = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t wave = tid >> 5;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t qb = b * kMqbQueriesPerBlock;
  const std::uint32_t t = qb + wave;
  const bool valid = t < n_tokens;
  // Clamped row for staging/loads on partial tail tiles; the final store
  // below is predicated on `valid`, so padding computes are discarded.
  const std::uint32_t tc = valid ? t : n_tokens - 1u;
  const std::uint32_t group = heads / kv_heads;
  const std::uint32_t kvh = h / group;

  __shared__ float qTile[kMqbQueriesPerBlock * kMqbHeadDim];
  __shared__ __half kRow[kMqbHeadDim];
  __shared__ __half vRow[kMqbHeadDim];

  // Stage the tile's Q rows once per block (512 floats, 4 per thread).
  for (std::uint32_t k = 0; k < 4u; ++k) {
    const std::uint32_t e = tid * 4u + k;
    const std::uint32_t row = e / kMqbHeadDim;
    const std::uint32_t col = e % kMqbHeadDim;
    const std::uint32_t tt =
        (qb + row < n_tokens) ? qb + row : n_tokens - 1u;
    qTile[e] =
        q[(static_cast<std::size_t>(tt) * heads + h) * kMqbHeadDim + col];
  }
  __syncthreads();

  // Per-thread Q shard in registers (legacy dpl == 4 lane mapping).
  const std::size_t qbase =
      static_cast<std::size_t>(wave) * kMqbHeadDim + lane * 4u;
  const float q0 = qTile[qbase + 0u];
  const float q1 = qTile[qbase + 1u];
  const float q2 = qTile[qbase + 2u];
  const float q3 = qTile[qbase + 3u];

  const std::uint32_t pos = start_pos + tc;
  const std::uint32_t n_kv = pos + 1u;
  const std::uint32_t k_beg =
      (window != 0u && n_kv > window) ? n_kv - window : 0u;

  // Uniform trip count per block (waves own different query rows, hence
  // different key extents): iterate the tile-wide span and predicate each
  // wave's own window. Guarded iterations change no state, so the per-query
  // update order matches the legacy kernel exactly. Uniform bounds also keep
  // every thread on the same __syncthreads sequence (no divergence hang).
  const std::uint32_t t_last =
      (qb + kMqbQueriesPerBlock - 1u < n_tokens)
          ? qb + kMqbQueriesPerBlock - 1u
          : n_tokens - 1u;
  const std::uint32_t n_kv_first = start_pos + qb + 1u;
  const std::uint32_t j_lo =
      (window != 0u && n_kv_first > window) ? n_kv_first - window : 0u;
  const std::uint32_t j_hi = start_pos + t_last + 1u;

  float m = -1e30f;
  float l = 0.0f;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (std::uint32_t j = j_lo; j < j_hi; ++j) {
    // One K row + one V row per block per key (one half per thread each);
    // all four waves then read the shared LDS copies.
    const std::size_t base =
        (static_cast<std::size_t>(j) * kv_heads + kvh) * kMqbHeadDim;
    kRow[tid] = k_cache[base + tid];
    vRow[tid] = v_cache[base + tid];
    __syncthreads();
    // Wave-uniform predicate (same j window for all 32 lanes of the wave),
    // so the shuffle reductions below always run wave-converged.
    if (j >= k_beg && j < n_kv) {
      float dot = q0 * __half2float(kRow[lane * 4u + 0u]);
      dot += q1 * __half2float(kRow[lane * 4u + 1u]);
      dot += q2 * __half2float(kRow[lane * 4u + 2u]);
      dot += q3 * __half2float(kRow[lane * 4u + 3u]);
      dot += __shfl_down_sync(0xffffffffull, dot, 16);
      dot += __shfl_down_sync(0xffffffffull, dot, 8);
      dot += __shfl_down_sync(0xffffffffull, dot, 4);
      dot += __shfl_down_sync(0xffffffffull, dot, 2);
      dot += __shfl_down_sync(0xffffffffull, dot, 1);
      const float full = __shfl_sync(0xffffffffull, dot, 0);
      const float score = full * score_scale;
      float m_new = m;
      if (lane == 0u) {
        m_new = m > score ? m : score;
      }
      const float msh = __shfl_sync(0xffffffffull, m_new, 0);
      const float corr = (lane == 0u && m == -1e30f) ? 0.0f : expf(m - msh);
      const float wgt = expf(score - msh);
      acc0 = acc0 * corr + wgt * __half2float(vRow[lane * 4u + 0u]);
      acc1 = acc1 * corr + wgt * __half2float(vRow[lane * 4u + 1u]);
      acc2 = acc2 * corr + wgt * __half2float(vRow[lane * 4u + 2u]);
      acc3 = acc3 * corr + wgt * __half2float(vRow[lane * 4u + 3u]);
      m = msh;
      if (lane == 0u) {
        l = l * corr + wgt;
      }
    }
    __syncthreads();
  }
  const float mf = __shfl_sync(0xffffffffull, m, 0);
  const float lf = __shfl_sync(0xffffffffull, l, 0);
  const float inv = 1.0f / lf;
  const float fix = expf(m - mf) * inv;
  if (valid) {
    float* orow =
        out + (static_cast<std::size_t>(t) * heads + h) * kMqbHeadDim;
    orow[lane * 4u + 0u] = acc0 * fix;
    orow[lane * 4u + 1u] = acc1 * fix;
    orow[lane * 4u + 2u] = acc2 * fix;
    orow[lane * 4u + 3u] = acc3 * fix;
  }
}

}  // namespace

bool AttentionF16MQB(const float* q, const __half* k_cache,
                     const __half* v_cache, float* out,
                     std::uint32_t n_tokens, std::uint32_t start_pos,
                     std::uint32_t heads, std::uint32_t kv_heads,
                     std::uint32_t d, hipStream_t stream, float score_scale,
                     std::uint32_t window) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      out == nullptr) {
    return false;
  }
  if (d != kMqbHeadDim || heads == 0u || kv_heads == 0u ||
      (heads % kv_heads) != 0u) {
    return false;
  }
  if (n_tokens == 0u) {
    return true;
  }
  // Tiny calls (decode rows, speculative verify batches) stay on the legacy
  // scalar kernel: a 4-query tile cannot pay off below one full tile, and
  // decode is latency-shaped rather than KV-traffic-shaped.
  if (n_tokens < kMqbMinTokens) {
    return false;
  }
  const std::uint32_t n_tiles =
      (n_tokens + kMqbQueriesPerBlock - 1u) / kMqbQueriesPerBlock;
  if (n_tiles > 65535u) {
    return false;
  }
  hipLaunchKernelGGL(AttentionF16MQBKernel, dim3(heads, n_tiles), dim3(128),
                     0, stream, q, k_cache, v_cache, out, n_tokens, start_pos,
                     heads, kv_heads, score_scale, window);
  return true;
}

}  // namespace gufo::models::glimmer::kernels
