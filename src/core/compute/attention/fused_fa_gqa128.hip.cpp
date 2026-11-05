// Fused GQA flash-attention prefill for (16Q, 8KV, d=128), causal, F16 KV.
//
// Data flow mirrors the proven core flash_f16_dense kernel 1:1 (LDS-staged Q,
// K/V streaming, LoadFrag fragments, LDS-mediated S/P exchange, F32 online
// softmax, F16 P, F32 O accumulate), retiled: d=64->128 (K-steps 4->8,
// split across four s_kh groups instead of two), MHA->GQA (one block per KV
// head serves its query pair, halving KV traffic vs one block per head),
// bidirectional->causal (per-row key mask against the global query position).
//
// gfx1151 notes: wave32 __shfl_xor reductions; LDS strides keep every
// LoadFrag 16B-aligned (q/k stride 136 halves, v/p stride 40 halves);
// K/V staging is thread-coarsened x8 with row-major reads. No prefetch
// double-buffering in v1 (correctness first); the sweep is still
// scores-traffic-free, which is the whole point on GTT-backed memory.
//
// Math contract (matches the LaunchBatchedAttention baseline this replaces
// for this geometry): QK scale 1/sqrt(d) folded into staged Q; FP32
// accumulate in both WMMA passes; online softmax renormalization identical
// to a full-row pass up to summation order; optional SiLU gate multiply in
// the epilogue (dense decoders pass gate == nullptr and skip it).
#if defined(ENGINE_ENABLE_HIP)
#include "src/core/compute/attention/fused_fa_gqa128.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gufo::core::compute::fused_fa_gqa128 {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// A fragment is 16 contiguous halves = 32 bytes. __half and _Float16 are
// bit-identical, so two 16-byte loads plus a bitcast replace 16 scalar
// loads. Every fragment base here is 16-byte aligned by construction.
__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kRows = 16;
constexpr std::uint32_t kKeys = 32;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kQueryHeads = 16;
constexpr std::uint32_t kKvHeads = 8;
constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
constexpr std::uint32_t kKSteps = kHeadDim / 16;
constexpr std::uint32_t kKStepsPerWave = 2;
constexpr std::uint32_t kKGroups = kKSteps / kKStepsPerWave;
constexpr std::uint32_t kKeyBlocks = 2;  // 32 keys = 2 x 16
constexpr std::uint32_t kSoftmaxLanes = 16;
constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
constexpr std::uint32_t kQStride = kHeadDim + 8;
constexpr std::uint32_t kKStride = kHeadDim + 8;
constexpr std::uint32_t kVtStride = kKeys + 8;
constexpr std::uint32_t kScoreStride = 20;
constexpr std::uint32_t kPStride = kKeys + 8;

static_assert(kGqa == 2, "one block serves exactly one KV head's query pair");
static_assert(kKSteps % kKStepsPerWave == 0, "k-steps split evenly");
static_assert(kKGroups == 4, "four s_kh groups cover d=128");
static_assert(kPerLane == 2, "two key columns per softmax lane");

// KV-cache write for the F32/F16 planes, same math as the model-side
// WriteBatchedKVCacheKernel this prefill chain otherwise uses: one
// (kv_head, token) block each, null planes skipped.
__global__ void FusedFaPackKvKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim) {
  const std::size_t b = blockIdx.y;
  const std::uint32_t kv_head = blockIdx.x;
  if (b >= batch_size || kv_head >= num_kv_heads)
    return;

  const std::size_t tid = threadIdx.x;
  const std::uint32_t pos = start_pos + static_cast<std::uint32_t>(b);
  const std::size_t k_src_offset =
      (b * num_kv_heads * head_dim) +
      (static_cast<std::size_t>(kv_head) * head_dim);

  const std::size_t cache_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
            max_context +
        pos) *
       head_dim);

  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t f16_cache_offset =
      ((static_cast<std::size_t>(layer_idx) * max_context + pos) * kv_width) +
      (static_cast<std::size_t>(kv_head) * head_dim);

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float key = k[k_src_offset + i];
    const float val = v[k_src_offset + i];
    if (k_cache != nullptr) {
      k_cache[cache_offset + i] = key;
    }
    if (v_cache != nullptr) {
      v_cache[cache_offset + i] = val;
    }
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_cache_offset + i] = __float2half_rn(key);
    }
    if (v_cache_f16 != nullptr) {
      v_cache_f16[f16_cache_offset + i] = __float2half_rn(val);
    }
  }
}

// Prefix re-sync (FP32 -> F16) for continued sessions, same math as the
// model-side SyncTiledAttentionKvPrefixKernel. Single-chunk embeddings run
// with start_pos == 0 and never reach here.
__global__ void FusedFaSyncKvPrefixKernel(
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t prefix_length,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim) {
  const std::size_t pos = blockIdx.y;
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  if (pos >= prefix_length || index >= kv_width) {
    return;
  }
  const std::size_t kv_head = index / head_dim;
  const std::size_t dimension = index % head_dim;
  const std::size_t f32_cache_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
            max_context +
        pos) *
       head_dim) +
      dimension;
  const std::size_t f16_cache_offset =
      ((static_cast<std::size_t>(layer_idx) * max_context + pos) * kv_width) +
      index;
  k_cache_f16[f16_cache_offset] = __float2half_rn(k_cache[f32_cache_offset]);
  v_cache_f16[f16_cache_offset] = __float2half_rn(v_cache[f32_cache_offset]);
}

__launch_bounds__(256, 2) __global__ void FusedFaGqa128Kernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    float* __restrict__ out, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::uint32_t batch_size, std::uint32_t max_context, float scale) {
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;
  const std::uint32_t kv_head = blockIdx.y;
  const std::uint32_t query_start = blockIdx.x * kRows;

  const std::size_t kv_width =
      static_cast<std::size_t>(kKvHeads) * kHeadDim;
  const __half* layer_k = k_cache +
      static_cast<std::size_t>(layer_idx) * max_context * kv_width;
  const __half* layer_v = v_cache +
      static_cast<std::size_t>(layer_idx) * max_context * kv_width;

  __shared__ __half q_lds[kGqa][kRows][kQStride];
  __shared__ __half k_lds[kKeys][kKStride];
  __shared__ __half v_lds[kHeadDim][kVtStride];
  __shared__ float s_lds[kKGroups][kKeyBlocks][kRows][kScoreStride];
  __shared__ __half p_lds[kRows][kPStride];
  __shared__ float row_sum[kGqa][kRows];
  __shared__ float row_scale[kRows];

  // Stage Q once per GQA head (F32 -> F16, scale folded).
  for (std::uint32_t i = tid; i < kGqa * kRows * kHeadDim; i += 256u) {
    const std::uint32_t j = i / (kRows * kHeadDim);
    const std::uint32_t r = (i / kHeadDim) % kRows;
    const std::uint32_t c = i % kHeadDim;
    const std::uint32_t qr = query_start + r;
    const std::uint32_t h = kv_head * kGqa + j;
    const float val =
        (qr < batch_size)
            ? q[(static_cast<std::size_t>(qr) * kQueryHeads + h) * kHeadDim +
                c] *
                  scale
            : 0.0F;
    q_lds[j][r][c] = __half(val);
  }
  __syncthreads();

  const std::uint32_t s_kh = wave / kKeyBlocks;
  const std::uint32_t s_tile = wave % kKeyBlocks;

  v16h q_frag[kGqa][kKStepsPerWave];
#pragma unroll
  for (std::uint32_t j = 0; j < kGqa; ++j) {
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = (s_kh * kKStepsPerWave + ks) * 16u;
      q_frag[j][ks] = LoadFrag(&q_lds[j][sub][d0]);
    }
  }

  // One O tile per wave: dim-tile bijection over 8 waves; the GQA head and
  // the second dim half are sequenced in the loops below.
  const std::uint32_t dim_o = wave & 3u;
  constexpr std::uint32_t kDimHalves = 2;
  v8f o_acc[kGqa][kDimHalves] = {};
  float running_max[kGqa] = {-INFINITY, -INFINITY};
  float running_sum[kGqa] = {0.0F, 0.0F};

  const std::uint32_t n_keys = start_pos + batch_size;
  const std::uint32_t n_key_tiles = (n_keys + kKeys - 1u) / kKeys;
  for (std::uint32_t kb = 0; kb < n_key_tiles; ++kb) {
    const std::uint32_t key_start = kb * kKeys;
    // Stage K tile.
    for (std::uint32_t i = tid; i < kKeys * kHeadDim; i += 256u) {
      const std::uint32_t r = i / kHeadDim;
      const std::uint32_t c = i % kHeadDim;
      const std::uint32_t kr = key_start + r;
      const __half val =
          (kr < n_keys)
              ? layer_k[(static_cast<std::size_t>(kr) * kv_width) +
                        static_cast<std::size_t>(kv_head) * kHeadDim + c]
              : __half(0.0F);
      k_lds[r][c] = val;
    }
    // Stage V tile transposed (lane owns key tid%32, dims (tid/32)*16+i).
    {
      const std::uint32_t v_key = tid % kKeys;
      const std::uint32_t v_slice = (tid / kKeys) * 16u;
      const std::uint32_t kr = key_start + v_key;
#pragma unroll
      for (std::uint32_t i = 0; i < 16u; ++i) {
        const __half val =
            (kr < n_keys)
                ? layer_v[(static_cast<std::size_t>(kr) * kv_width) +
                          static_cast<std::size_t>(kv_head) * kHeadDim +
                          v_slice + i]
                : __half(0.0F);
        v_lds[v_slice + i][v_key] = val;
      }
    }
    __syncthreads();

    for (std::uint32_t j = 0; j < kGqa; ++j) {
      // S = Q K^T for this head and wave's key block.
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = (s_kh * kKStepsPerWave + ks) * 16u;
        const v16h k_frag = LoadFrag(&k_lds[(s_tile * 16u + sub)][d0]);
        s_acc = Wmma(q_frag[j][ks], k_frag, s_acc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8u; ++i) {
        s_lds[s_kh][s_tile][(2u * i) + half_id][sub] = s_acc[i];
      }
      __syncthreads();

      // Online softmax over the full 32-key tile, causal: key kc is live
      // for global query qr_g iff kc <= qr_g.
      {
        const std::uint32_t rg = tid / kSoftmaxLanes;
        const std::uint32_t seg = tid % kSoftmaxLanes;
        const std::uint32_t qr = query_start + rg;
        const bool row_live = qr < batch_size;
        const std::uint32_t qr_global = start_pos + qr;
        const std::uint32_t row = rg;
        float part_max = -INFINITY;
        float vals[kPerLane];
#pragma unroll
        for (std::uint32_t m = 0; m < kPerLane; ++m) {
          const std::uint32_t col = seg * kPerLane + m;
          const std::uint32_t kc = key_start + col;
          const bool valid = row_live && kc < n_keys && kc <= qr_global;
          const std::uint32_t tile = col / 16u;
          float s = valid ? s_lds[0][tile][row][col % 16u] : -INFINITY;
          s += valid ? s_lds[1][tile][row][col % 16u] : 0.0F;
          s += valid ? s_lds[2][tile][row][col % 16u] : 0.0F;
          s += valid ? s_lds[3][tile][row][col % 16u] : 0.0F;
          vals[m] = s;
          part_max = fmaxf(part_max, vals[m]);
        }
#pragma unroll
        for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
          part_max = fmaxf(part_max, __shfl_xor(part_max, off));
        }
        const float prev_max = running_max[j];
        const float next_max = fmaxf(prev_max, part_max);
        const float prior_scale =
            isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
        float part_sum = 0.0F;
#pragma unroll
        for (std::uint32_t m = 0; m < kPerLane; ++m) {
          const float w =
              isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
          part_sum += w;
          p_lds[rg][seg * kPerLane + m] = __half(w);
        }
#pragma unroll
        for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
          part_sum += __shfl_xor(part_sum, off);
        }
        running_max[j] = next_max;
        running_sum[j] = running_sum[j] * prior_scale + part_sum;
        if (seg == 0) row_scale[rg] = prior_scale;
      }
      __syncthreads();

      // Rescale O accumulator.
      {
        float sc[kDimHalves][8];
#pragma unroll
        for (std::uint32_t dh = 0; dh < kDimHalves; ++dh) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8u; ++i) {
            sc[dh][i] = row_scale[(2u * i) + half_id];
          }
        }
#pragma unroll
        for (std::uint32_t dh = 0; dh < kDimHalves; ++dh) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8u; ++i) {
            o_acc[j][dh][i] *= sc[dh][i];
          }
        }
      }

      // O += P V, both dim halves sequenced.
#pragma unroll
      for (std::uint32_t dh = 0; dh < kDimHalves; ++dh) {
        const std::uint32_t dim_base = (dh * 4u + dim_o) * 16u;
#pragma unroll
        for (std::uint32_t kbb = 0; kbb < kKeyBlocks; ++kbb) {
          const v16h p_frag = LoadFrag(&p_lds[sub][kbb * 16u]);
          const v16h v_frag = LoadFrag(&v_lds[dim_base + sub][kbb * 16u]);
          o_acc[j][dh] = Wmma(p_frag, v_frag, o_acc[j][dh]);
        }
      }
      // Separate this head's PV/V reads from the next head's S writes (all
      // cross-thread LDS traffic needs the barrier on both sides).
      __syncthreads();
    }
  }

  if (tid % kSoftmaxLanes == 0) {
    const std::uint32_t rg = tid / kSoftmaxLanes;
    row_sum[0][rg] = running_sum[0];
    row_sum[1][rg] = running_sum[1];
  }
  __syncthreads();

  // Epilogue: normalize + write each head's two 16x16 O tiles (dim halves).
  const float* gh_base = gate;
#pragma unroll
  for (std::uint32_t j = 0; j < kGqa; ++j) {
#pragma unroll
    for (std::uint32_t dh = 0; dh < kDimHalves; ++dh) {
      const std::uint32_t dim_base = (dh * 4u + dim_o) * 16u;
#pragma unroll
      for (std::uint32_t i = 0; i < 8u; ++i) {
        const std::uint32_t row = 2u * i + half_id;
        const std::uint32_t qr = query_start + row;
        if (qr >= batch_size) continue;
        const float denominator = row_sum[j][row];
        const std::uint32_t h = kv_head * kGqa + j;
        const std::size_t off =
            (static_cast<std::size_t>(qr) * kQueryHeads + h) * kHeadDim +
            dim_base + sub;
        float val =
            (denominator > 0.0F) ? (o_acc[j][dh][i] / denominator) : 0.0F;
        if (gh_base != nullptr) {
          const float g = gh_base[off];
          val *= 1.0F / (1.0F + __expf(-g));
        }
        out[off] = val;
      }
    }
  }
}

}  // namespace

bool LaunchFusedFaGqa128(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream, bool skip_kv_write) {
  if (q == nullptr || k == nullptr || v == nullptr || out_context == nullptr ||
      k_cache_f16 == nullptr || v_cache_f16 == nullptr || batch_size == 0 ||
      !IsFusedFaGqa128Supported(num_heads, num_kv_heads, head_dim,
                                static_cast<std::uint32_t>(batch_size),
                                start_pos +
                                    static_cast<std::uint32_t>(batch_size)) ||
      start_pos + batch_size > max_context) {
    return false;
  }
  constexpr std::uint32_t kBlockSize = 256;
  if (!skip_kv_write) {
    if (start_pos > 0 && k_cache != nullptr && v_cache != nullptr) {
      const dim3 sync_grid(
          static_cast<unsigned int>((kKvHeads * kHeadDim + kBlockSize - 1) /
                                    kBlockSize),
          start_pos);
      FusedFaSyncKvPrefixKernel<<<sync_grid, kBlockSize, 0, stream>>>(
          k_cache, v_cache, static_cast<__half*>(k_cache_f16),
          static_cast<__half*>(v_cache_f16), layer_idx, start_pos, max_context,
          kKvHeads, kHeadDim);
    }
    const dim3 pack_grid(kKvHeads,
                         static_cast<unsigned int>(batch_size));
    FusedFaPackKvKernel<<<pack_grid, kBlockSize, 0, stream>>>(
        k, v, k_cache, v_cache, static_cast<__half*>(k_cache_f16),
        static_cast<__half*>(v_cache_f16), layer_idx, start_pos, batch_size,
        max_context, kKvHeads, kHeadDim);
  }

  const dim3 grid(static_cast<unsigned int>((batch_size + kRows - 1) / kRows),
                  kKvHeads);
  const float scale = 1.0F / sqrtf(static_cast<float>(kHeadDim));
  hipLaunchKernelGGL(FusedFaGqa128Kernel, grid, dim3(256), 0, stream, q, gate,
                     static_cast<const __half*>(k_cache_f16),
                     static_cast<const __half*>(v_cache_f16), out_context,
                     layer_idx, start_pos,
                     static_cast<std::uint32_t>(batch_size), max_context,
                     scale);
  return true;
}

}  // namespace gufo::core::compute::fused_fa_gqa128

#endif  // defined(ENGINE_ENABLE_HIP)
