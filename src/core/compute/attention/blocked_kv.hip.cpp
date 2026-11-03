#include "src/core/compute/attention/blocked_kv.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/models/qwen/vision/rope.hpp"

// Shared blocked-KV codecs and kernels, extracted verbatim from the
// flash-next ROCm backend (src/models/qwen38_flash_next/kernels/rocm/
// kernels.hip.cpp). Only the enclosing namespace changed, so launches stay
// bitwise identical.

namespace gufo::core::compute::blocked_kv {
namespace {

constexpr unsigned kThreads = 256;

/// Lane i receives lane (i ^ offset)'s value, as __shfl_xor(v, offset) does
/// across a wave32. On gfx1151 it is a DPP row_xmask (offsets below 16) or
/// v_permlanex16 (16): a register move instead of an LDS crossbar round trip
/// (ds_bpermute). Only the transport differs, so reductions stay bitwise;
/// every lane must be active, as the shuffle's callers already require.
__device__ __forceinline__ int XorLane(int v, int offset) {
#if defined(__gfx1151__)
  switch (offset) {
    case 1:
      return __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, false);
    case 2:
      return __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, false);
    case 4:
      return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, false);
    case 8:
      return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, false);
    case 16:
      return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false,
                                          false);
  }
#endif
  return __shfl_xor(v, offset);
}

__device__ __forceinline__ unsigned XorLane(unsigned v, int offset) {
  return static_cast<unsigned>(XorLane(static_cast<int>(v), offset));
}

__device__ __forceinline__ float XorLane(float v, int offset) {
  return __builtin_bit_cast(float, XorLane(__builtin_bit_cast(int, v), offset));
}

/// Sum over one wave; every lane receives the total. The wave-per-row
/// kernels split rows across lanes (gfx1151 runs wave32).
__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += XorLane(v, offset);
  }
  return v;
}

/// Block-wide sum over kThreads threads; every thread receives the total.
__device__ float BlockSum(float v, float* shared) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += XorLane(v, offset);
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total += shared[w];
  }
  return total;
}

__device__ float BlockMax(float v, float* shared) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, XorLane(v, offset));
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = -INFINITY;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total = fmaxf(total, shared[w]);
  }
  return total;
}

// Element codecs for the blocked KV layouts. Q8 and FP8 share one byte layout
// -- 32 elements followed by an f16 block scale, 34 bytes -- so the blocked
// kernels are templated on the codec and only the element encode/decode and
// the quantization range differ.
//
// Q8_0 is a symmetric INT8 grid over amax/127. E4M3 is an OCP FP8 grid (bias 7,
// three mantissa bits, max 448) over amax/448: its exponent field absorbs the
// outliers that a per-block INT8 scale would clip, at the cost of mantissa
// bits.
struct KvCodecQ8 {
  using Elem = std::uint8_t;  // the symmetric INT8 grid, stored as bytes
  static constexpr float kRange = 127.0f;
  static __device__ __forceinline__ Elem Encode(float v) {
    v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
    return static_cast<Elem>(static_cast<std::int8_t>(rintf(v)));
  }
  static __device__ __forceinline__ float Decode(Elem e) {
    return static_cast<float>(static_cast<std::int8_t>(e));
  }
};

// OCP E4M3 (no inf; 0x7F and 0xFF are NaN). Emulated in software: gfx1151 has
// no fp8 conversion instruction and the mmq helper is CDNA3/Blackwell only.
// Values round to nearest even; the block scale keeps in-range blocks below the
// format maximum, and anything past it saturates.
struct KvCodecFp8 {
  using Elem = std::uint8_t;
  static constexpr float kRange = 448.0f;
  static __device__ __forceinline__ Elem Encode(float x) {
    const std::uint32_t bits = __float_as_uint(x);
    const std::uint32_t sign = (bits >> 24) & 0x80u;
    const float a = fabsf(x);
    if (a != a)  // NaN keeps the format's NaN encoding
      return static_cast<Elem>(sign | 0x7Fu);
    if (a >= 448.0f)  // saturate; also catches inf
      return static_cast<Elem>(sign | 0x7Eu);
    if (a < 0.015625f) {  // 2^-6: subnormal grid of step 2^-9
      const std::uint32_t m = static_cast<std::uint32_t>(rintf(a * 512.0f));
      return static_cast<Elem>(sign | m);
    }
    const std::uint32_t m23 = bits & 0x7FFFFFu;
    const std::uint32_t r =
        m23 + 0x7FFFFu + ((m23 >> 20) & 1u);  // round to nearest even, 3 bits
    std::uint32_t m3 = r >> 20u;
    std::uint32_t e = ((bits >> 23) & 0xFFu) - 120u;  // unbiased + E4M3 bias
    if (m3 > 7u) {                                    // mantissa carry
      m3 = 0u;
      ++e;
    }
    if (e > 15u)  // unreachable below 448; kept so a bug cannot alias the sign
      return static_cast<Elem>(sign | 0x7Eu);
    return static_cast<Elem>(sign | (e << 3u) | m3);
  }
  static __device__ __forceinline__ float Decode(Elem e) {
    const std::uint32_t sign = static_cast<std::uint32_t>(e & 0x80u) << 24;
    const std::uint32_t f = static_cast<std::uint32_t>(e & 0x7Fu);
    const std::uint32_t exp = f >> 3u;
    if (exp == 0u) {  // subnormal grid: mant * 2^-9, exact in fp32
      const float v = static_cast<float>(f & 0x7u) * (1.0f / 512.0f);
      return sign == 0u ? v : -v;
    }
    return __uint_as_float(sign | ((exp + 120u) << 23u) | ((f & 0x7u) << 20u));
  }
};

// Blocked twin of PrepareAttentionKernel: identical RMS/RoPE/q/gate math, but
// K and V rows go to `Codec` blocks with a real per-32-block scale (a per-head
// scale is too coarse for post-RoPE K, where outliers collapse the small dims
// to zero). q/gate are unchanged.
template<std::uint32_t kHeadsPerBlock, typename Codec>
__global__ void PrepareAttentionKvKernel(
    const float* __restrict__ packed, std::uint32_t stride,
    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
    float* __restrict__ q, float* __restrict__ gate,
    typename Codec::Elem* __restrict__ k_cache,
    typename Codec::Elem* __restrict__ v_cache, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t d, std::uint32_t rotary,
    const std::uint32_t* start_pos, float theta, float eps, float k_range,
    float v_range, std::uint32_t k_block_bytes, std::uint32_t v_block_bytes,
    const gufo::models::qwen::vision::DeviceRope* rope) {
  __shared__ float partial[kHeadsPerBlock][8];
  __shared__ float norm[kHeadsPerBlock][256];
  const std::uint32_t t = blockIdx.x, first = blockIdx.y * kHeadsPerBlock,
                      tid = threadIdx.x;
  const std::uint32_t lane = tid % 32, wave = tid / 32;
  const std::uint32_t live = min(kHeadsPerBlock, heads + kv_heads - first);
  const std::size_t width = std::size_t(heads) * d,
                    kv_width = std::size_t(kv_heads) * d;
  const std::uint32_t ppos = *start_pos + t;
  const float* row = packed + std::size_t(t) * stride;
  float values[kHeadsPerBlock];
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    const std::uint32_t h = first + j;
    const bool query = h < heads;
    const std::uint32_t head = query ? h : h - heads;
    values[j] = j < live && tid < d
                    ? row[(query ? head * 2 * d : 2 * width + head * d) + tid]
                    : 0.0f;
    float ss = values[j] * values[j];
    // Match the separate RMS kernel's rounded square before its reduction.
    asm volatile("" : "+v"(ss));
    ss = WaveSum(ss);
    if (lane == 0)
      partial[j][wave] = ss;
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    if (j >= live)
      continue;
    const std::uint32_t h = first + j;
    const bool query = h < heads;
    const std::uint32_t head = query ? h : h - heads;
    float total = 0.0f;
    for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w)
      total += partial[j][w];
    const float scale = rsqrtf(total / static_cast<float>(d) + eps);
    const float* gamma = query ? q_gamma : k_gamma;
    if (tid < d) {
      norm[j][tid] = values[j] * scale * (gamma != nullptr ? gamma[tid] : 1.0f);
      if (query)
        gate[std::size_t(t) * width + head * d + tid] =
            row[head * 2 * d + d + tid];
      // NOTE (q8): K/V cache stores deferred to the quantize phases below.
    }
  }
  __syncthreads();
  const std::uint32_t half = rotary / 2;
  float sine = 0.0f, cosine = 0.0f;
  if (tid < half) {
    const float freq = powf(
        theta, -2.0f * static_cast<float>(tid) / static_cast<float>(rotary));
    sincosf(gufo::models::qwen::vision::RopePosition(rope, *start_pos + t, tid) * freq, &sine,
            &cosine);
  }
// Phase 1: rotate K rows, stash back into norm.
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    if (j >= live)
      continue;
    const std::uint32_t h = first + j;
    const bool query = h < heads;
    const std::uint32_t head = query ? h : h - heads;
    if (tid < half) {
      const float a = norm[j][tid], b = norm[j][tid + half];
      const float lo = __fmaf_rn(a, cosine, -__fmul_rn(b, sine)),
                  hi = __fmaf_rn(a, sine, __fmul_rn(b, cosine));
      if (query) {
        q[std::size_t(t) * width + head * d + tid] = lo;
        q[std::size_t(t) * width + head * d + tid + half] = hi;
      } else {
        norm[j][tid] = lo;
        norm[j][tid + half] = hi;
      }
    }
    if (tid >= rotary && tid < d) {
      if (query)
        q[std::size_t(t) * width + head * d + tid] = norm[j][tid];
      // else: tail already staged in norm.
    }
  }
  __syncthreads();
// Phase 2: per-block scale + element store.
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    if (j >= live)
      continue;
    const std::uint32_t h = first + j;
    if (h < heads)
      continue;
    const std::uint32_t head = h - heads;
    // Blocked layout with a real per-32-block scale. One lane owns a whole
    // block so the packed path writes both nibbles of every payload byte.
    const std::uint32_t head_blocks = (d + 31) / 32;
    const std::uint32_t row_blocks =
        static_cast<std::uint32_t>((kv_width + 31) / 32);
    typename Codec::Elem* out = k_cache +
                                std::size_t(ppos) * row_blocks * k_block_bytes +
                                std::size_t(head) * head_blocks * k_block_bytes;
    for (std::uint32_t b = tid; b < head_blocks; b += blockDim.x) {
      const std::uint32_t base = b * 32;
      const std::uint32_t n = base + 32 <= d ? 32 : (base < d ? d - base : 0);
      float mx = 0.0f;
      for (std::uint32_t k = 0; k < n; ++k)
        mx = fmaxf(mx, fabsf(norm[j][base + k]));
      const float sc = mx > 0.0f ? mx / k_range : 1.0f;
      typename Codec::Elem* blk = out + std::size_t(b) * k_block_bytes;
      if (k_block_bytes == 18) {
        // Packed 4-bit keys: element 2e in the low nibble, 2e+1 in the high.
        const auto nibble = [&](std::uint32_t e) -> std::uint32_t {
          if (e >= n)
            return 0u;
          int q = static_cast<int>(rintf(norm[j][base + e] / sc));
          q = q > 7 ? 7 : (q < -7 ? -7 : q);
          return static_cast<std::uint32_t>(q) & 0xFu;
        };
        for (std::uint32_t e = 0; e < 16; ++e) {
          blk[e] = static_cast<std::uint8_t>(nibble(e * 2) |
                                             (nibble(e * 2 + 1) << 4));
        }
        *reinterpret_cast<__half*>(blk + 16) = __float2half(sc);
      } else {
        for (std::uint32_t k = 0; k < n; ++k) {
          blk[k] = Codec::Encode(norm[j][base + k] / sc);
        }
        *reinterpret_cast<__half*>(blk + 32) = __float2half(sc);
      }
    }
  }
  __syncthreads();
// Phase 3: V rows (raw packed values, no norm/RoPE) -> blocked.
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    if (j >= live)
      continue;
    const std::uint32_t h = first + j;
    if (h < heads)
      continue;
    const std::uint32_t head = h - heads;
    const float* vrow = row + 2 * width + kv_width + std::size_t(head) * d;
    const std::uint32_t head_blocks = (d + 31) / 32;
    const std::uint32_t row_blocks =
        static_cast<std::uint32_t>((kv_width + 31) / 32);
    typename Codec::Elem* out = v_cache +
                                std::size_t(ppos) * row_blocks * v_block_bytes +
                                std::size_t(head) * head_blocks * v_block_bytes;
    if (v_block_bytes == 18) {
      // Packed 4-bit values: one lane owns a whole 32-element block so both
      // nibbles of a payload byte are computed and stored by the same lane.
      for (std::uint32_t b = tid; b < head_blocks; b += blockDim.x) {
        const std::uint32_t base = b * 32;
        const std::uint32_t n = base + 32 <= d ? 32 : (base < d ? d - base : 0);
        float mx = 0.0f;
        for (std::uint32_t k = 0; k < n; ++k)
          mx = fmaxf(mx, fabsf(vrow[base + k]));
        const float sc = mx > 0.0f ? mx / v_range : 1.0f;
        std::uint8_t* blk = out + std::size_t(b) * 18;
        for (std::uint32_t j = 0; j < 16; ++j) {
          const auto nibble = [&](std::uint32_t e) -> std::uint32_t {
            if (e >= n)
              return 0u;
            int q = static_cast<int>(rintf(vrow[base + e] / sc));
            q = q > 7 ? 7 : (q < -7 ? -7 : q);
            return static_cast<std::uint32_t>(q) & 0xFu;
          };
          blk[j] = static_cast<std::uint8_t>(nibble(j * 2) |
                                             (nibble(j * 2 + 1) << 4));
        }
        *reinterpret_cast<__half*>(blk + 16) = __float2half(sc);
      }
    } else {
      for (std::uint32_t i = tid; i < d; i += blockDim.x) {
        const std::uint32_t b = i / 32;
        const std::uint32_t base = b * 32;
        const std::uint32_t n = base + 32 <= d ? 32 : (base < d ? d - base : 0);
        float mx = 0.0f;
        for (std::uint32_t k = 0; k < n; ++k)
          mx = fmaxf(mx, fabsf(vrow[base + k]));
        // Value rows quantize against `v_range`, not the codec range: a K8V4
        // study passes 7 so V lands on a 4-bit grid while keys stay 8-bit.
        const float sc = mx > 0.0f ? mx / v_range : 1.0f;
        out[b * v_block_bytes + (i % 32)] = Codec::Encode(vrow[i] / sc);
        if (i % 32 == 0) {
          *reinterpret_cast<__half*>(out + std::size_t(b) * v_block_bytes +
                                     32) = __float2half(sc);
        }
      }
    }
  }
}

// Per-(token,head) symmetric quantize of float rows into `Codec` blocks.
// src row (t,h) = src + t*src_tok_stride + h*src_head_stride, length d.
// dst row (start_pos+t) holds (kv_width+31)/32 blocks of `block_bytes`;
// head h starts at block h*(d+31)/32 and its block b trails an f16 scale
// (8-bit: 32 codes + scale = 34 bytes; 4-bit: 16 packed bytes + scale = 18).
template<typename Codec>
__global__ void QuantizeKvRowsKernel(
    const float* __restrict__ src, std::uint32_t src_tok_stride,
    std::uint32_t src_head_stride, typename Codec::Elem* __restrict__ dst,
    std::uint32_t kv_width, std::uint32_t kv_heads, std::uint32_t d,
    std::uint32_t n_tokens, const std::uint32_t* start_pos, float range,
    std::uint32_t block_bytes) {
  const std::uint32_t t = blockIdx.x, h = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;
  const float* row =
      src + std::size_t(t) * src_tok_stride + std::size_t(h) * src_head_stride;
  const std::uint32_t ppos = *start_pos + t;
  // Blocked layout with a real per-32-block scale. A per-head scale is
  // too coarse for post-RoPE K (outliers collapse the small dims to 0).
  const std::uint32_t row_blocks = (kv_width + 31) / 32;
  const std::uint32_t head_blocks = (d + 31) / 32;
  typename Codec::Elem* out = dst +
                              std::size_t(ppos) * row_blocks * block_bytes +
                              std::size_t(h) * head_blocks * block_bytes;
  for (std::uint32_t b = tid; b < head_blocks; b += blockDim.x) {
    const std::uint32_t base = b * 32;
    const std::uint32_t n = base + 32 <= d ? 32 : (base < d ? d - base : 0);
    float mx = 0.0f;
    for (std::uint32_t k = 0; k < n; ++k)
      mx = fmaxf(mx, fabsf(row[base + k]));
    const float sc = mx > 0.0f ? mx / range : 1.0f;
    typename Codec::Elem* blk = out + std::size_t(b) * block_bytes;
    if (block_bytes == 18) {
      // One lane owns the whole block, so both nibbles of each payload byte
      // are packed here without sharing a byte across lanes.
      const auto nibble = [&](std::uint32_t e) -> std::uint32_t {
        if (e >= n)
          return 0u;
        int q = static_cast<int>(rintf(row[base + e] / sc));
        q = q > 7 ? 7 : (q < -7 ? -7 : q);
        return static_cast<std::uint32_t>(q) & 0xFu;
      };
      for (std::uint32_t j = 0; j < 16; ++j) {
        blk[j] =
            static_cast<std::uint8_t>(nibble(j * 2) | (nibble(j * 2 + 1) << 4));
      }
      *reinterpret_cast<__half*>(blk + 16) = __float2half(sc);
    } else {
      for (std::uint32_t k = 0; k < n; ++k) {
        blk[k] = Codec::Encode(row[base + k] / sc);
      }
      *reinterpret_cast<__half*>(blk + 32) = __float2half(sc);
    }
  }
}

// Dual-buffer dequant (one launch for K+V staging; halves launches).
template<typename Codec>
__global__ void DequantizeKvRowsKernel(
    const typename Codec::Elem* __restrict__ src_k, __half* __restrict__ dst_k,
    const typename Codec::Elem* __restrict__ src_v, __half* __restrict__ dst_v,
    std::uint32_t row_blocks, std::uint32_t d, std::uint32_t n_rows,
    std::uint32_t k_block_bytes, std::uint32_t v_block_bytes) {
  const std::uint32_t r = blockIdx.x;
  const std::uint32_t b = blockIdx.y;
  const std::uint32_t l = threadIdx.x;
  if (r >= n_rows)
    return;
  const std::uint32_t base = b * 32;
  const std::uint32_t n = base + 32 <= d ? 32 : (base < d ? d - base : 0);
  if (l >= n)
    return;
  const std::size_t blk_off = (std::size_t(r) * row_blocks + b) * k_block_bytes;
  const typename Codec::Elem* kb = src_k + blk_off;
  const std::size_t out_off = std::size_t(r) * d + base + l;
  if (k_block_bytes == 18) {
    const auto* kbp = reinterpret_cast<const std::uint8_t*>(kb);
    const float sk = __half2float(*reinterpret_cast<const __half*>(kbp + 16));
    const std::uint32_t byte = kbp[l / 2];
    const std::uint32_t nib = (byte >> (4 * (l & 1u))) & 0xFu;
    const int q = static_cast<int>(nib) - (nib & 8u ? 16 : 0);
    dst_k[out_off] = __float2half(static_cast<float>(q) * sk);
  } else {
    const float sk = __half2float(*reinterpret_cast<const __half*>(kb + 32));
    dst_k[out_off] = __float2half(Codec::Decode(kb[l]) * sk);
  }
  const std::size_t v_blk_off =
      (std::size_t(r) * row_blocks + b) * v_block_bytes;
  const auto* vb = reinterpret_cast<const std::uint8_t*>(src_v + v_blk_off);
  if (v_block_bytes == 18) {
    // Two lanes share each payload byte; both only read it.
    const float sv = __half2float(*reinterpret_cast<const __half*>(vb + 16));
    const std::uint32_t byte = vb[l / 2];
    const std::uint32_t nib = (byte >> (4 * (l & 1u))) & 0xFu;
    const int q = static_cast<int>(nib) - (nib & 8u ? 16 : 0);
    dst_v[out_off] = __float2half(static_cast<float>(q) * sv);
  } else {
    const float sv = __half2float(*reinterpret_cast<const __half*>(vb + 32));
    dst_v[out_off] = __float2half(Codec::Decode(vb[l]) * sv);
  }
}

template<typename Codec>
__global__ void AttentionKernelKv(
    const float* q, const typename Codec::Elem* k_cache,
    const typename Codec::Elem* v_cache, const std::uint32_t* mask,
    std::uint32_t mask_words, float* out, float* partials,
    const std::uint32_t* start_pos, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t d, std::uint32_t ratio, std::uint32_t k_block_bytes,
    std::uint32_t v_block_bytes, float score_scale) {
  constexpr std::uint32_t kTile = 256;
  constexpr std::uint32_t kWindow = 1024;  // blocks compacted per pass
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t splits = gridDim.z;
  __shared__ float qs[256];
  __shared__ float p[kTile];
  __shared__ float shared[32];
  __shared__ std::uint32_t keys[kTile];
  // Room for one carried partial tile at the model's ratio of 4 (under a
  // tile's worth of blocks). A whole tile of keys would push the kernel past
  // 16 KiB of LDS and cost a resident block per CU on multi-row calls; below
  // ratio 4 a partial tile too big to carry is attended in its window.
  constexpr std::uint32_t kCarry = kTile / 4;
  __shared__ std::uint32_t list[kWindow + kCarry];
  __shared__ std::uint32_t wave_total[8];
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t pos = *start_pos + t;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const std::uint32_t n_kv = pos + 1;
  const std::uint32_t tail_start = (n_kv / ratio) * ratio;
  const std::uint32_t* words =
      mask == nullptr ? nullptr
                      : mask + static_cast<std::size_t>(t) * mask_words;
  const float scale =
      score_scale > 0.0F ? score_scale : rsqrtf(static_cast<float>(d));
  const std::uint32_t i = threadIdx.x;
  const std::uint32_t lane = i & 31u;
  const std::uint32_t wave = i >> 5u;
  qs[i] = i < d ? q[(static_cast<std::size_t>(t) * heads + h) * d + i] : 0.0f;
  __syncthreads();
  float m = -INFINITY;
  float l = 0.0f;
  // Every wave accumulates its own 32 keys of each tile into a partial
  // context row (eight dims per lane); the partials are summed at the end.
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  // Blocked rows: (kv_heads*d/32) blocks. Key rows use `k_block_bytes` and
  // value rows `v_block_bytes` (34 = 8-bit codes, 18 = packed nibbles).
  const std::size_t kv_stride =
      static_cast<std::size_t>(kv_heads) * d / 32 * k_block_bytes;
  const std::size_t v_stride =
      static_cast<std::size_t>(kv_heads) * d / 32 * v_block_bytes;
  const typename Codec::Elem* k_head = k_cache + kvh * (d / 32 * k_block_bytes);
  const typename Codec::Elem* v_head = v_cache + kvh * (d / 32 * v_block_bytes);

  // Rows are read eight contiguous halves per lane, one 16-byte load, so a
  // wave reads a whole 256-wide row at once. A wave issues the rows of
  // `kBatch` consecutive keys of its 32 before using any, so their loads
  // overlap instead of each waiting out its own latency. Empty lanes read
  // the last key's row and are never used; a batch with no key at all is
  // skipped (returns false).
  constexpr std::uint32_t kBatch = 8;
  const auto load_batch_blocked =
      [&](const typename Codec::Elem* head, std::size_t stride,
          std::uint32_t block_bytes, std::uint32_t base, uint64_t* rows,
          float* sc8) {
        bool any = false;
#pragma unroll
        for (std::uint32_t b = 0; b < kBatch; ++b) {
          any = any || keys[base + b] < n_kv;
        }
        if (!any) {
          return false;
        }
#pragma unroll
        for (std::uint32_t b = 0; b < kBatch; ++b) {
          const std::uint32_t j = min(keys[base + b], n_kv - 1);
          // Block holding this lane’s 8 elements; fp16 scale trails it.
          // The 8-byte chunk may straddle 8-alignment (34-byte blocks), so
          // memcpy lets the compiler emit safe loads; the scale is 2-aligned.
          const typename Codec::Elem* blk =
              head + (static_cast<std::size_t>(j) * stride) +
              (lane / 4) * block_bytes;
          __builtin_memcpy(&rows[b], blk + ((lane * 8) % 32),
                           sizeof(std::uint64_t));
          sc8[b] = static_cast<float>(
              *reinterpret_cast<const __half*>(blk + block_bytes - 2));
        }
        return true;
      };
  const auto unpack_blocked = [](const uint64_t& packed, float sc,
                                 float* out8) {
    const auto* elems = reinterpret_cast<const typename Codec::Elem*>(&packed);
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j) {
      out8[j] = Codec::Decode(elems[j]) * sc;
    }
  };
  // Packed 4-bit values: a lane owns eight consecutive elements = four
  // payload bytes starting at ((lane*8)%32)/2; the f16 scale follows the
  // 16-byte payload.
  const auto load_batch_v4 = [&](const typename Codec::Elem* head,
                                 std::uint32_t base, uint32_t* rows,
                                 float* sc8) {
    bool any = false;
#pragma unroll
    for (std::uint32_t b = 0; b < kBatch; ++b) {
      any = any || keys[base + b] < n_kv;
    }
    if (!any) {
      return false;
    }
#pragma unroll
    for (std::uint32_t b = 0; b < kBatch; ++b) {
      const std::uint32_t j = min(keys[base + b], n_kv - 1);
      const typename Codec::Elem* blk =
          head + (static_cast<std::size_t>(j) * v_stride) + (lane / 4) * 18;
      const typename Codec::Elem* payload = blk + ((lane * 8) % 32) / 2;
      __builtin_memcpy(&rows[b], payload, sizeof(std::uint32_t));
      sc8[b] = static_cast<float>(*reinterpret_cast<const __half*>(blk + 16));
    }
    return true;
  };
  const auto unpack_v4 = [](const uint32_t& packed, float sc, float* out8) {
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j) {
      const std::uint32_t nib = (packed >> (4 * j)) & 0xFu;
      const int q = static_cast<int>(nib ^ 8u) - 8;  // sign-extend 4 bits
      out8[j] = static_cast<float>(q) * sc;
    }
  };
  // Packed 4-bit keys: a lane owns eight consecutive elements = four payload
  // bytes; identical layout to values, with the key row stride.
  const auto load_batch_k4 = [&](std::uint32_t base, uint32_t* rows,
                                 float* sc8) {
    bool any = false;
#pragma unroll
    for (std::uint32_t b = 0; b < kBatch; ++b) {
      any = any || keys[base + b] < n_kv;
    }
    if (!any) {
      return false;
    }
#pragma unroll
    for (std::uint32_t b = 0; b < kBatch; ++b) {
      const std::uint32_t j = min(keys[base + b], n_kv - 1);
      const typename Codec::Elem* blk =
          k_head + (static_cast<std::size_t>(j) * kv_stride) +
          (lane / 4) * k_block_bytes;
      const typename Codec::Elem* payload = blk + ((lane * 8) % 32) / 2;
      __builtin_memcpy(&rows[b], payload, sizeof(std::uint32_t));
      sc8[b] = static_cast<float>(*reinterpret_cast<const __half*>(blk + 16));
    }
    return true;
  };

  // One tile: lane i holds key position keys[i] (n_kv marks an empty lane).
  // Scores: wave w takes keys 32w..32w+31, a wave-wide dot product per key.
  // Values: the same split, eight dims per lane.
  const auto tile_step = [&]() {
    float qv[8];
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j) {
      qv[j] = qs[(lane * 8) + j];
    }
    float s_mine = -INFINITY;
    if (k_block_bytes == 18) {
      for (std::uint32_t k0 = 0; k0 < 32; k0 += kBatch) {
        uint32_t rows[kBatch];
        float ksc[kBatch];
        if (!load_batch_k4((wave * 32) + k0, rows, ksc)) {
          continue;
        }
#pragma unroll
        for (std::uint32_t b = 0; b < kBatch; ++b) {
          const std::uint32_t ju = keys[(wave * 32) + k0 + b];
          float kv[8];
          unpack_v4(rows[b], ksc[b], kv);
          float dot = qv[7] * kv[7];
          dot = fmaf(qv[3], kv[3], dot);
          dot = fmaf(qv[5], kv[5], dot);
          dot = fmaf(qv[1], kv[1], dot);
          dot = fmaf(qv[6], kv[6], dot);
          dot = fmaf(qv[2], kv[2], dot);
          dot = fmaf(qv[4], kv[4], dot);
          dot = fmaf(qv[0], kv[0], dot);
          dot = WaveSum(dot);
          if (lane == k0 + b) {
            s_mine = ju < n_kv ? dot * scale : -INFINITY;
          }
        }
      }
    } else {
      for (std::uint32_t k0 = 0; k0 < 32; k0 += kBatch) {
        uint64_t rows[kBatch];
        float ksc[kBatch];
        if (!load_batch_blocked(k_head, kv_stride, k_block_bytes,
                                (wave * 32) + k0, rows, ksc)) {
          continue;
        }
#pragma unroll
        for (std::uint32_t b = 0; b < kBatch; ++b) {
          const std::uint32_t ju = keys[(wave * 32) + k0 + b];
          const std::uint32_t j = min(ju, n_kv - 1);
          float kv[8];
          unpack_blocked(rows[b], ksc[b], kv);
          // The lane's eight products in the order fast-math has always
          // summed them, spelled out so the scores stay bitwise.
          float dot = qv[7] * kv[7];
          dot = fmaf(qv[3], kv[3], dot);
          dot = fmaf(qv[5], kv[5], dot);
          dot = fmaf(qv[1], kv[1], dot);
          dot = fmaf(qv[6], kv[6], dot);
          dot = fmaf(qv[2], kv[2], dot);
          dot = fmaf(qv[4], kv[4], dot);
          dot = fmaf(qv[0], kv[0], dot);
          dot = WaveSum(dot);
          if (lane == k0 + b) {
            s_mine = ju < n_kv ? dot * scale : -INFINITY;
          }
        }
      }
    }
    const float tile_max = BlockMax(s_mine, shared);
    const float m_new = fmaxf(m, tile_max);
    const float rescale = m == -INFINITY ? 0.0f : __expf(m - m_new);
    const float pj = s_mine == -INFINITY ? 0.0f : __expf(s_mine - m_new);
    p[i] = pj;
    const float tile_sum = BlockSum(pj, shared);
    l = l * rescale + tile_sum;
#pragma unroll
    for (std::uint32_t x = 0; x < 8; ++x) {
      acc[x] *= rescale;
    }
    m = m_new;
    __syncthreads();
    if (v_block_bytes == 18) {
      for (std::uint32_t k0 = 0; k0 < 32; k0 += kBatch) {
        uint32_t rows[kBatch];
        float vsc[kBatch];
        if (!load_batch_v4(v_head, (wave * 32) + k0, rows, vsc)) {
          continue;
        }
#pragma unroll
        for (std::uint32_t b = 0; b < kBatch; ++b) {
          const float w = p[(wave * 32) + k0 + b];
          if (w != 0.0f) {
            float vv[8];
            unpack_v4(rows[b], vsc[b], vv);
#pragma unroll
            for (std::uint32_t x = 0; x < 8; ++x) {
              acc[x] = fmaf(w, vv[x], acc[x]);
            }
          }
        }
      }
    } else {
      for (std::uint32_t k0 = 0; k0 < 32; k0 += kBatch) {
        uint64_t rows[kBatch];
        float vsc[kBatch];
        if (!load_batch_blocked(v_head, v_stride, v_block_bytes,
                                (wave * 32) + k0, rows, vsc)) {
          continue;
        }
#pragma unroll
        for (std::uint32_t b = 0; b < kBatch; ++b) {
          const float w = p[(wave * 32) + k0 + b];
          if (w != 0.0f) {
            const std::uint32_t jv = min(keys[(wave * 32) + k0 + b], n_kv - 1);
            (void)jv;
            float vv[8];
            unpack_blocked(rows[b], vsc[b], vv);
#pragma unroll
            for (std::uint32_t x = 0; x < 8; ++x) {
              acc[x] = fmaf(w, vv[x], acc[x]);
            }
          }
        }
      }
    }
    __syncthreads();
  };

  if (words == nullptr) {
    for (std::uint32_t tile = split * kTile; tile < n_kv;
         tile += kTile * splits) {
      keys[i] = tile + i < n_kv ? tile + i : n_kv;
      __syncthreads();
      tile_step();
    }
  } else {
    const std::uint32_t n_complete = tail_start / ratio;
    const std::uint32_t lane = i & 31u;
    const std::uint32_t wave = i >> 5u;
    // Tiles are cut from the running compacted list, not per window: the
    // budget's ~512 selected blocks make the same eight full tiles at any
    // depth, dealt round-robin over the splits. Cut per window, a deep
    // context scattered them into dozens of mostly empty tiles, and every
    // window holding fewer than one tile's blocks landed on split 0 alone,
    // so decode attention grew with the context instead of the budget.
    const std::uint32_t per_tile = kTile / ratio;
    std::uint32_t carry = 0;    // compacted blocks not yet cut into a tile
    std::uint32_t tile_no = 0;  // tiles cut so far (the same in every split)
    for (std::uint32_t w0 = 0; w0 < n_complete; w0 += kWindow) {
      // Thread i owns blocks w0 + 4i .. +3 of the window: flag the selected
      // ones and compact their indices with a block-wide exclusive scan,
      // appending after the carried-over blocks.
      // The four blocks share one mask word: read it once.
      std::uint32_t flags = 0;
      std::uint32_t count = 0;
      const std::uint32_t b0 = w0 + (i * 4);
      const std::uint32_t word = b0 < n_complete ? words[b0 / 32] : 0u;
      for (std::uint32_t k = 0; k < 4; ++k) {
        const std::uint32_t b = b0 + k;
        const bool set = b < n_complete && ((word >> (b % 32)) & 1u);
        flags |= (set ? 1u : 0u) << k;
        count += set ? 1u : 0u;
      }
      std::uint32_t incl = count;
      for (std::uint32_t off = 1; off < 32; off <<= 1) {
        const std::uint32_t v = __shfl_up(incl, off);
        incl += lane >= off ? v : 0u;
      }
      if (lane == 31) {
        wave_total[wave] = incl;
      }
      __syncthreads();
      std::uint32_t base = 0;
      for (std::uint32_t w = 0; w < wave; ++w) {
        base += wave_total[w];
      }
      std::uint32_t total = 0;
      for (std::uint32_t w = 0; w < 8; ++w) {
        total += wave_total[w];
      }
      std::uint32_t slot = carry + base + incl - count;
      for (std::uint32_t k = 0; k < 4; ++k) {
        if ((flags >> k) & 1u) {
          list[slot++] = w0 + (i * 4) + k;
        }
      }
      __syncthreads();
      const std::uint32_t n = carry + total;
      std::uint32_t t0 = 0;
      for (; t0 < n; t0 += per_tile, ++tile_no) {
        const std::uint32_t len = min(per_tile, n - t0);
        if (len < per_tile && len <= kCarry) {
          break;  // carried into the next window
        }
        if (tile_no % splits != split) {
          continue;
        }
        const std::uint32_t entry = i / ratio;
        keys[i] = entry < len ? (list[t0 + entry] * ratio) + (i % ratio) : n_kv;
        __syncthreads();
        tile_step();
      }
      // Move the partial tile's blocks to the front for the next window.
      carry = t0 < n ? n - t0 : 0;
      if (t0 > 0) {
        const std::uint32_t moved = i < carry ? list[t0 + i] : 0u;
        __syncthreads();
        if (i < carry) {
          list[i] = moved;
        }
      }
      __syncthreads();
    }
    if (carry > 0 && tile_no % splits == split) {
      const std::uint32_t entry = i / ratio;
      keys[i] = entry < carry ? (list[entry] * ratio) + (i % ratio) : n_kv;
      __syncthreads();
      tile_step();
    }
    // The incomplete tail block is always visible.
    if (tail_start < n_kv && split == 0) {
      keys[i] = tail_start + i < n_kv ? tail_start + i : n_kv;
      __syncthreads();
      tile_step();
    }
  }
  // Sum the eight wave partials: wave w's lane holds dims 8*lane..+7.
  __shared__ float red[8][256];
#pragma unroll
  for (std::uint32_t x = 0; x < 8; ++x) {
    red[wave][(lane * 8) + x] = acc[x];
  }
  __syncthreads();
  float total = 0.0f;
  for (std::uint32_t w = 0; w < 8; ++w) {
    total += red[w][i];
  }
  if (splits == 1) {
    if (i < d) {
      out[(static_cast<std::size_t>(t) * heads + h) * d + i] = total / l;
    }
    return;
  }
  float* part =
      partials +
      ((static_cast<std::size_t>(t) * heads + h) * splits + split) * (d + 2);
  if (i < d) {
    part[2 + i] = total;
  }
  if (i == 0) {
    part[0] = m;
    part[1] = l;
  }
}

/// grid (heads, queries): merges the split partials of one row with the
/// usual log-sum-exp rescaling.
__global__ void AttentionMergeKernel(const float* partials, float* out,
                                     std::uint32_t heads, std::uint32_t d,
                                     std::uint32_t splits) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t i = threadIdx.x;
  const float* base =
      partials + (static_cast<std::size_t>(t) * heads + h) * splits * (d + 2);
  float m = -INFINITY;
  for (std::uint32_t s = 0; s < splits; ++s) {
    m = fmaxf(m, base[s * (d + 2)]);
  }
  float l = 0.0f;
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float* part = base + s * (d + 2);
    const float ms = part[0];
    const float scale = ms == -INFINITY ? 0.0f : __expf(ms - m);
    l += part[1] * scale;
    if (i < d) {
      acc += part[2 + i] * scale;
    }
  }
  if (i < d) {
    out[(static_cast<std::size_t>(t) * heads + h) * d + i] = acc / l;
  }
}

}  // namespace

// Blocked host entries share one element codec per call. The codec is a
// runtime value while the kernels are templated on it, so each entry builds
// its launch arguments once inside `fn` and `DispatchCodec` picks the tag.
template<typename Fn>
void DispatchCodec(KvCodec codec, Fn&& fn) {
  if (codec == KvCodec::kFp8)
    fn(KvCodecFp8{});
  else
    fn(KvCodecQ8{});
}

bool PrepareAttentionKv(const float* packed, std::uint32_t stride,
                        const float* q_gamma, const float* k_gamma, float* q,
                        float* gate, std::uint8_t* k_cache,
                        std::uint8_t* v_cache, KvCodec codec, float k_range,
                        float v_range, std::uint32_t key_block_bytes,
                        std::uint32_t value_block_bytes, std::uint32_t n_tokens,
                        std::uint32_t heads, std::uint32_t kv_heads,
                        std::uint32_t d, std::uint32_t rotary_dim,
                        const std::uint32_t* start_pos, float theta, float eps,
                        hipStream_t stream,
                        const gufo::models::qwen::vision::DeviceRope* rope, bool prefill) {
  if (d == 0 || d > 256 || rotary_dim == 0 || rotary_dim > d ||
      rotary_dim % 2 != 0 || kv_heads == 0 ||
      stride < static_cast<std::size_t>(2) * (heads + kv_heads) * d) {
    return false;
  }
  if (n_tokens == 0)
    return true;
  const bool single = !prefill && n_tokens < 32;
  const dim3 grid = single ? dim3(n_tokens, heads + kv_heads)
                           : dim3(n_tokens, (heads + kv_heads + 3) / 4);
  DispatchCodec(codec, [&](auto tag) {
    using Codec = decltype(tag);
    if (single)
      hipLaunchKernelGGL((PrepareAttentionKvKernel<1, Codec>), grid,
                         dim3(kThreads), 0, stream, packed, stride, q_gamma,
                         k_gamma, q, gate, k_cache, v_cache, heads, kv_heads, d,
                         rotary_dim, start_pos, theta, eps, k_range, v_range,
                         key_block_bytes, value_block_bytes, rope);
    else
      hipLaunchKernelGGL((PrepareAttentionKvKernel<4, Codec>), grid,
                         dim3(kThreads), 0, stream, packed, stride, q_gamma,
                         k_gamma, q, gate, k_cache, v_cache, heads, kv_heads, d,
                         rotary_dim, start_pos, theta, eps, k_range, v_range,
                         key_block_bytes, value_block_bytes, rope);
  });
  if (const char* dump = std::getenv("GUFO_KV_DUMP")) {
    // Offline audit hook: append the raw blocked K/V rows written by this call
    // so the cache can be compared against a CPU emulation of the same
    // quantization. Layout: u32 n_tokens, kv_heads, d, k_row_bytes,
    // v_row_bytes, start_pos; then k_row_bytes*n_tokens of K followed by
    // v_row_bytes*n_tokens of V.
    if (hipStreamSynchronize(stream) == hipSuccess) {
      const std::size_t kv_width = static_cast<std::size_t>(kv_heads) * d;
      const std::size_t k_row_bytes = (kv_width + 31) / 32 * key_block_bytes;
      const std::size_t v_row_bytes = (kv_width + 31) / 32 * value_block_bytes;
      const std::size_t k_bytes = k_row_bytes * n_tokens;
      const std::size_t v_bytes = v_row_bytes * n_tokens;
      std::vector<std::uint8_t> host_k(k_bytes), host_v(v_bytes);
      if (hipMemcpy(host_k.data(), k_cache, k_bytes, hipMemcpyDeviceToHost) ==
              hipSuccess &&
          hipMemcpy(host_v.data(), v_cache, v_bytes, hipMemcpyDeviceToHost) ==
              hipSuccess) {
        if (std::FILE* f = std::fopen(dump, "ab")) {
          const std::uint32_t header[6] = {
              n_tokens,
              kv_heads,
              d,
              static_cast<std::uint32_t>(k_row_bytes),
              static_cast<std::uint32_t>(v_row_bytes),
              start_pos != nullptr ? *start_pos : 0u};
          std::fwrite(header, sizeof(header), 1, f);
          std::fwrite(host_k.data(), 1, k_bytes, f);
          std::fwrite(host_v.data(), 1, v_bytes, f);
          std::fclose(f);
        }
      }
    }
  }
  return true;
}

bool DequantizeKvRows(const std::uint8_t* src_k, __half* dst_k,
                      const std::uint8_t* src_v, __half* dst_v,
                      std::uint32_t n_rows, std::uint32_t row_elems,
                      std::uint32_t key_block_bytes,
                      std::uint32_t value_block_bytes, KvCodec codec,
                      hipStream_t stream) {
  if (src_k == nullptr || dst_k == nullptr || src_v == nullptr ||
      dst_v == nullptr || n_rows == 0 || row_elems == 0)
    return false;
  const std::uint32_t blocks = (row_elems + 31) / 32;
  if (blocks == 0)
    return false;
  DispatchCodec(codec, [&](auto tag) {
    using Codec = decltype(tag);
    hipLaunchKernelGGL((DequantizeKvRowsKernel<Codec>), dim3(n_rows, blocks),
                       dim3(32), 0, stream, src_k, dst_k, src_v, dst_v, blocks,
                       row_elems, n_rows, key_block_bytes, value_block_bytes);
  });
  return true;
}

void QuantizeKvRows(const float* src, std::uint32_t src_tok_stride,
                    std::uint32_t src_head_stride, std::uint8_t* dst,
                    std::uint32_t kv_width, std::uint32_t kv_heads,
                    std::uint32_t d, std::uint32_t n_tokens, KvCodec codec,
                    float range, std::uint32_t block_bytes,
                    const std::uint32_t* start_pos, hipStream_t stream) {
  if (n_tokens == 0 || kv_heads == 0 || d == 0)
    return;
  DispatchCodec(codec, [&](auto tag) {
    using Codec = decltype(tag);
    hipLaunchKernelGGL((QuantizeKvRowsKernel<Codec>), dim3(n_tokens, kv_heads),
                       dim3(kThreads), 0, stream, src, src_tok_stride,
                       src_head_stride, dst, kv_width, kv_heads, d, n_tokens,
                       start_pos, range, block_bytes);
  });
}

void AttentionKv(const float* q, const std::uint8_t* k_cache,
                 const std::uint8_t* v_cache, const std::uint32_t* mask,
                 std::uint32_t mask_words, float* out, float* partials,
                 std::uint32_t splits, std::uint32_t n_tokens,
                 const std::uint32_t* start_pos, std::uint32_t heads,
                 std::uint32_t kv_heads, std::uint32_t d, std::uint32_t ratio,
                 std::uint32_t key_block_bytes, std::uint32_t value_block_bytes,
                 KvCodec codec, hipStream_t stream, float score_scale) {
  // Scalar direct read serves d == 256 and d == 128 (Phase A <Q8,128> gap:
  // the kernel below is d-generic up to 256 -- every out/partial write is
  // guarded by `i < d` and block addressing only needs 32-elem blocks).
  // Anything else refuses silently so the caller falls back.
  if (d != 256 && d != 128) {
    return;
  }
  const std::uint32_t z = partials != nullptr ? std::max(splits, 1u) : 1u;
  DispatchCodec(codec, [&](auto tag) {
    using Codec = decltype(tag);
    hipLaunchKernelGGL((AttentionKernelKv<Codec>), dim3(heads, n_tokens, z),
                       dim3(kThreads), 0, stream, q, k_cache, v_cache, mask,
                       mask_words, out, partials, start_pos, heads, kv_heads, d,
                       ratio, key_block_bytes, value_block_bytes, score_scale);
  });
  if (z > 1) {
    hipLaunchKernelGGL(AttentionMergeKernel, dim3(heads, n_tokens),
                       dim3(kThreads), 0, stream, partials, out, heads, d, z);
  }
}

}  // namespace gufo::core::compute::blocked_kv
