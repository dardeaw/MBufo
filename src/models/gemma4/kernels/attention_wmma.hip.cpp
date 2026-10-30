// Gemma 4 prefill attention on the WMMA matrix cores.
//
// Ported from the validated Qwen 27B WMMA prefill kernel. Only the
// compile-time geometry is retargeted to Gemma 4's sliding-window layers
// (16 query heads, 8 KV heads, head_dim 256 => GQA ratio 2). The fused
// gate path and the sparse-mask path stay compiled; the launcher passes
// nullptrs for both (Gemma 4 has no output gate and uses a window, not a
// selection mask).
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gufo::models::gemma4::kernels::wmma {

// --- helpers carried over from the Qwen kernel ---
namespace {

constexpr unsigned kThreads = 256;

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float SoftplusF(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}
/// One block_q8_1 (mmq's decode activation format): d and the row sum, then
/// the 32 codes.
}

// F16 WMMA fragments (wave32): sixteen halves per lane, eight F32
// accumulators per lane.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// BF16 fragments carry the same sixteen 16-bit lanes.
using v16bf = __attribute__((ext_vector_type(16))) __bf16;

__device__ __forceinline__ v8f WmmaBf16(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
      __builtin_bit_cast(v16bf, a), __builtin_bit_cast(v16bf, b), c);
}

// Keep indexer queries in F32: narrowing them before ranking can swap blocks
// at the selection boundary. A thread scores one key, reusing it across all
// four heads. Preserve the original wave32 F32 accumulation and reduction
// tree: even tiny changes can exchange blocks at the top-k boundary.
constexpr std::uint32_t kSelectHeads = 4;
constexpr std::uint32_t kSelectDim = 128;

// F16 fragments are used by attention and projection kernels below.
__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

// --- XorLane / SyncLds helpers ---
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

/// mmq's quantize_q8_1 for element `i` of contiguous rows (row width a
// A barrier for kernels whose waves exchange data through LDS only.
// __syncthreads also waits for every outstanding global load and drops the
// L0 cache, which serializes register prefetches behind the barrier. The
// global inputs are read-only here, so completing LDS traffic is enough.
__device__ __forceinline__ __attribute__((convergent)) void SyncLds() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
  asm volatile("" ::: "memory");
}

// Masked prefill attention on the WMMA matrix cores, ported from the Qwen

// --- kernel (geometry retargeted to Gemma 4 SWA) ---
constexpr std::uint32_t kWmmaHeadDim = 256;
constexpr std::uint32_t kWmmaQueryHeads = 16;
constexpr std::uint32_t kWmmaKvHeads = 8;
constexpr std::uint32_t kWmmaGqa = kWmmaQueryHeads / kWmmaKvHeads;
constexpr std::uint32_t kWmmaAttnWidth = kWmmaQueryHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaKvWidth = kWmmaKvHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaHeads = 2;  // query heads per block, divides GQA
// Sixteen queries keep the causal-tail accumulator in registers.
constexpr std::uint32_t kWmmaQueryRows = 16;
constexpr std::uint32_t kWmmaKeys = 16;
constexpr std::uint32_t kWmmaRatio = 4;
constexpr std::uint32_t kWmmaKSteps = kWmmaHeadDim / 16;
// Row padding keeps a 16-byte-per-lane fragment read off a single bank group.
constexpr std::uint32_t kWmmaKStride = kWmmaHeadDim + 8;
// Union-mask capacity of the live visible extent: one bit per 4-token block,
// 262,144 visible tokens. Rows may be strided wider.
constexpr std::uint32_t kWmmaMaxMaskWords = 2048;

/// Two row layouts share the kernel. The dense window packs 16 queries x 2
/// heads (a row block is 16 queries of one head, grid.y over head pairs).
/// The sparse window packs four queries x twelve heads into three row
/// blocks without padding, with adjacent blocks covering both KV heads: the key
/// tiles are gathered from the union of the block's query selections and
/// four selections overlap far less than 32 (measured at 16k depth: 846
/// versus 2,478 selected blocks against 512 per query).
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

__device__ __forceinline__ int32x8_t WmmaI8(int32x4_t a, int32x4_t b,
                                            int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}

template<std::uint32_t kQueryRows, std::uint32_t kKeys, bool kPackHeads,
         bool kI8Mma>
__launch_bounds__(256, 2) __global__ void WmmaCausalAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    const std::uint32_t* __restrict__ mask, std::uint32_t mask_words,
    float* __restrict__ out, std::uint32_t start_pos, std::uint32_t n_tokens,
    std::uint32_t first_query_group = 0, std::uint32_t k_block_bytes = 0,
    std::uint32_t v_block_bytes = 0, std::uint32_t window = 0) {
  // The launcher accepts only four-token selection blocks. Keep that
  // geometry constant throughout the key sweep.
  constexpr std::uint32_t ratio = kWmmaRatio;
  constexpr std::uint32_t kHeadDim = kWmmaHeadDim;
  constexpr std::uint32_t kRowBlocks = kPackHeads
                                           ? (kQueryRows * kWmmaGqa + 15) / 16
                                           : (kQueryRows / 16) * kWmmaHeads;
  static_assert(!kPackHeads || kQueryRows * kWmmaGqa == 48,
                "four queries pack twelve heads into three tiles");
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / 2;
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kOTilesPerWave = (kRowBlocks * kWmmaKSteps) / 8;
  constexpr std::uint32_t kSoftmaxLanes = 4;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  // Output dim tiles owned per wave: 8 waves x this = kHeadDim/16.
  constexpr std::uint32_t kDimTilesPerWave = kHeadDim / 128;
  static_assert(kSTiles == (kPackHeads ? 3 : 2), "two waves per S tile");
  static_assert(kOTilesPerWave == 2 * kRowBlocks, "O tiles per wave");
  static_assert(kDimTilesPerWave * 8 * 16 == kHeadDim, "dim tiles cover head");
  static_assert(kKeys % 16 == 0 && (kPackHeads || kQueryRows % 16 == 0),
                "16-row WMMA tiles");
  static_assert(kKStepsPerWave * 2 == kWmmaKSteps, "k split");
  static_assert(kWmmaKStride % 8 == 0 && kVtStride % 8 == 0,
                "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  // Keep the two KV heads of nearby queries together in the launch order.
  // This improves cache reuse without changing any query's key sweep.
  const std::uint32_t linear = blockIdx.y * gridDim.x + blockIdx.x;
  const std::uint32_t query_group =
      first_query_group + (kPackHeads ? linear / kWmmaKvHeads : blockIdx.x);
  const std::uint32_t query_start = query_group * kQueryRows;
  const std::uint32_t kv_head =
      kPackHeads ? linear % kWmmaKvHeads : blockIdx.y / (kWmmaGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      kPackHeads ? kv_head * kWmmaGqa
                 : (kv_head * kWmmaGqa) +
                       ((blockIdx.y % (kWmmaGqa / kWmmaHeads)) * kWmmaHeads);
  // Gemma 4 sets f_attention_scale = 1.0 (llama.cpp gemma4: no 1/sqrt(d)).
  constexpr float attention_scale = 1.0F;

  // Row (row block rb, row r) -> (local query, query head, live).
  const auto row_query = [&](std::uint32_t rb, std::uint32_t r) {
    return kPackHeads ? query_start + (rb * 16 + r) / kWmmaGqa
                      : query_start + ((rb / kWmmaHeads) * 16) + r;
  };
  const auto row_head = [&](std::uint32_t rb, std::uint32_t r) {
    return kPackHeads ? first_query_head + (rb * 16 + r) % kWmmaGqa
                      : first_query_head + (rb % kWmmaHeads);
  };
  const auto row_live = [&](std::uint32_t rb, std::uint32_t r) {
    return row_query(rb, r) < n_tokens &&
           (!kPackHeads || rb * 16 + r < kQueryRows * kWmmaGqa);
  };

  // K and V^T have their own buffers so the next tile's S overlaps this
  // tile's PV (see the key sweep below).
  __shared__ __half k_lds[kKeys * kWmmaKStride];
  __shared__ __attribute__((aligned(16))) __half v_lds[kHeadDim * kVtStride];
  // int8 path staging (only touched when kI8Mma; f16 buffers stay).
  __shared__ std::uint8_t k_i8[2 * kKeys * 272];
  __shared__ float k_scl[2 * kKeys * 8];
  __shared__ float q0_scl[kRows * 16];
  __shared__ float q1_scl[kRows * 8];
  // Pad score rows for the four-lane softmax reads and probability rows
  // for the WMMA fragments. Both transposes otherwise repeat LDS banks.
  // Dense score rows are padded to 20 floats: the WMMA scatter stays nearly
  // conflict-free and each softmax lane reads its four scores as a float4.
  // The sparse window keeps 17-float rows and scalar reads (faster there).
  constexpr bool kVectorSoftmax = !kPackHeads;
  constexpr std::uint32_t kScoreStride = kVectorSoftmax ? 20 : 17;
  static_assert(kKeys / kSoftmaxLanes == 4, "one float4 of scores per lane");
  __shared__
      __attribute__((aligned(16))) float s_lds[2][kSTiles][16][kScoreStride];
  __shared__ __attribute__((aligned(16))) __half p_lds[kRows][kKeys + 8];
  __shared__ float row_sum[kRows];
  // Rescale factors grouped as [row block][row parity][row / 2]: a lane of
  // the PV phase needs the eight rows of its parity, read as two float4.
  __shared__ __attribute__((aligned(16))) float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kI8Mma ? 1 : kKStepsPerWave];
  // int8 Q codes: 8 blocks x 8 int32 (32 codes each, 4 per int32).
  int q0w[kI8Mma ? 32 : 1];
  int q1w[kI8Mma ? 32 : 1];
  if (wave < 2 * kSTiles) {
    const std::uint32_t local_query = row_query(s_rb, sub);
    const bool live = row_live(s_rb, sub);
    const float* q_row =
        q +
        (static_cast<std::size_t>(live ? local_query : 0) * kWmmaAttnWidth) +
        (static_cast<std::size_t>(live ? row_head(s_rb, sub) : 0) * kHeadDim);
    if constexpr (kI8Mma) {
      (void)q_frag;
      // Two-level Q: Q0 captures range (incl. outliers), R1 the
      // remainder. Per-wave-half blocks (bb); pure scales published (the
      // attention scale folds once at rescale). All loop bounds are
      // compile-time; scalars only, no extra register arrays.
      const std::uint32_t my_row = (s_rb * 16) + sub;
#pragma unroll
      for (std::uint32_t bb = 0; bb < 4; ++bb) {
        const std::uint32_t b = (s_kh * 4) + bb;
        float m0[2] = {0.0F, 0.0F};
#pragma unroll
        for (std::uint32_t h = 0; h < 2; ++h) {
#pragma unroll
          for (std::uint32_t v = 0; v < 4; ++v) {
            const auto* fp = reinterpret_cast<const float4*>(
                q_row + (b * 32) + ((h * 16) + (v * 4)));
            const float4 f = live ? *fp : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            m0[h] = fmaxf(m0[h], fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)),
                                       fmaxf(fabsf(f.z), fabsf(f.w))));
          }
        }
        const float d0[2] = {live ? (m0[0] / 127.0F) : 0.0F,
                             live ? (m0[1] / 127.0F) : 0.0F};
        const float id0[2] = {(live && m0[0] > 0.0F) ? (127.0F / m0[0]) : 0.0F,
                              (live && m0[1] > 0.0F) ? (127.0F / m0[1]) : 0.0F};
        q0_scl[(my_row * 16) + (b * 2)] = d0[0];
        q0_scl[(my_row * 16) + (b * 2) + 1] = d0[1];
#pragma unroll
        for (std::uint32_t v = 0; v < 8; ++v) {
          const float idh = id0[v / 4];
          const auto* fp =
              reinterpret_cast<const float4*>(q_row + (b * 32) + (v * 4));
          const float4 f = live ? *fp : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
          const std::uint32_t c0 = static_cast<std::uint8_t>(
              static_cast<std::int8_t>(roundf(f.x * idh)));
          const std::uint32_t c1 = static_cast<std::uint8_t>(
              static_cast<std::int8_t>(roundf(f.y * idh)));
          const std::uint32_t c2 = static_cast<std::uint8_t>(
              static_cast<std::int8_t>(roundf(f.z * idh)));
          const std::uint32_t c3 = static_cast<std::uint8_t>(
              static_cast<std::int8_t>(roundf(f.w * idh)));
          q0w[(bb * 8) + v] =
              static_cast<int>(c0 | (c1 << 8) | (c2 << 16) | (c3 << 24));
        }
        float m1 = 0.0F;
#pragma unroll
        for (std::uint32_t v = 0; v < 8; ++v) {
          const auto* fp =
              reinterpret_cast<const float4*>(q_row + (b * 32) + (v * 4));
          const float4 f = live ? *fp : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
          const int w = q0w[(bb * 8) + v];
          const float dh = q0_scl[(my_row * 16) + (b * 2) + (v / 4)];
          const float e0 =
              f.x - static_cast<float>(static_cast<std::int8_t>(w & 0xFF)) * dh;
          const float e1 =
              f.y -
              static_cast<float>(static_cast<std::int8_t>((w >> 8) & 0xFF)) *
                  dh;
          const float e2 =
              f.z -
              static_cast<float>(static_cast<std::int8_t>((w >> 16) & 0xFF)) *
                  dh;
          const float e3 =
              f.w -
              static_cast<float>(static_cast<std::int8_t>((w >> 24) & 0xFF)) *
                  dh;
          m1 = fmaxf(m1, fmaxf(fmaxf(fabsf(e0), fabsf(e1)),
                               fmaxf(fabsf(e2), fabsf(e3))));
        }
        const float d1 = live ? (m1 / 127.0F) : 0.0F;
        const float id1 = (live && m1 > 0.0F) ? (127.0F / m1) : 0.0F;
        q1_scl[(my_row * 8) + b] = d1;
#pragma unroll
        for (std::uint32_t v = 0; v < 8; ++v) {
          const auto* fp =
              reinterpret_cast<const float4*>(q_row + (b * 32) + (v * 4));
          const float4 f = live ? *fp : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
          const int w = q0w[(bb * 8) + v];
          const float dh = q0_scl[(my_row * 16) + (b * 2) + (v / 4)];
          const float e0 =
              f.x - static_cast<float>(static_cast<std::int8_t>(w & 0xFF)) * dh;
          const float e1 =
              f.y -
              static_cast<float>(static_cast<std::int8_t>((w >> 8) & 0xFF)) *
                  dh;
          const float e2 =
              f.z -
              static_cast<float>(static_cast<std::int8_t>((w >> 16) & 0xFF)) *
                  dh;
          const float e3 =
              f.w -
              static_cast<float>(static_cast<std::int8_t>((w >> 24) & 0xFF)) *
                  dh;
          const std::uint32_t c0 = static_cast<std::uint8_t>(
              static_cast<std::int8_t>(roundf(e0 * id1)));
          const std::uint32_t c1 = static_cast<std::uint8_t>(
              static_cast<std::int8_t>(roundf(e1 * id1)));
          const std::uint32_t c2 = static_cast<std::uint8_t>(
              static_cast<std::int8_t>(roundf(e2 * id1)));
          const std::uint32_t c3 = static_cast<std::uint8_t>(
              static_cast<std::int8_t>(roundf(e3 * id1)));
          q1w[(bb * 8) + v] =
              static_cast<int>(c0 | (c1 << 8) | (c2 << 16) | (c3 << 24));
        }
      }
    } else {
      (void)q0w;
      (void)q1w;
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
        for (std::uint32_t v = 0; v < 4; ++v) {
          const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
          q_frag[ks][(v * 4) + 0] =
              static_cast<_Float16>(f.x * attention_scale);
          q_frag[ks][(v * 4) + 1] =
              static_cast<_Float16>(f.y * attention_scale);
          q_frag[ks][(v * 4) + 2] =
              static_cast<_Float16>(f.z * attention_scale);
          q_frag[ks][(v * 4) + 3] =
              static_cast<_Float16>(f.w * attention_scale);
        }
      }
    }
  }

  // This wave owns dim tiles `wave` and `wave + 8` for every row block.
  v8f o_acc[kRowBlocks][kDimTilesPerWave] = {};
  // The same four threads own each softmax row throughout the key sweep.
  // Keep its running statistics in registers; only the final denominator
  // and each tile's rescale factor need to cross waves.
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t context_end = start_pos + n_tokens;
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  // Key tiles are gathered from `kBlocksPerTile` selection blocks (ratio
  // keys each). In the sparse window the blocks come from the union of this
  // block's query masks (every row still applies its own mask below), then
  // every block from the earliest row's tail on; the dense window walks the
  // blocks in order. The sweep therefore costs the union of the selected
  // windows, not the context.
  constexpr std::uint32_t kBlocksPerTile = kKeys / 4;
  static_assert(kBlocksPerTile == 4, "a tile is four selection blocks");
  const std::uint32_t n_blocks = (max_visible + ratio - 1) / ratio;
  const std::uint32_t tail_block =
      ((start_pos + query_start + 1) / ratio);  // first always-visible block
  constexpr bool sparse = kPackHeads;
  // Four 512-block selections plus their incomplete tail. The same LDS
  // stores a mask for arbitrary wider selections used by operator callers.
  constexpr unsigned kListCapacity = 4 * 512 + 4;
  // A compact entry also carries which of the packed queries selected the
  // block, so the softmax reads its mask bits from the tile, not global.
  constexpr unsigned kMaskRows = kPackHeads ? kQueryRows : 1;
  constexpr unsigned kMemberShift = 28;
  static_assert(kMaskRows <= 32 - kMemberShift, "membership bits fit");
  __shared__ unsigned union_words[kListCapacity];
  __shared__ unsigned wave_counts[8];
  bool compact = false;
  unsigned selected = 0;
  if (sparse) {
    // Consecutive word ranges per thread make the prefix scan preserve
    // increasing block order, including ties and the always-visible tail.
    const unsigned live_rows = min(kQueryRows, n_tokens - query_start);
    const unsigned word_count = (n_blocks + 31) / 32;
    const unsigned words_per_thread = (word_count + 255) / 256;
    unsigned local_words[8];
    unsigned row_words[8][kMaskRows];
    unsigned count = 0;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      const unsigned w = tid * words_per_thread + j;
      unsigned bits = 0;
#pragma unroll
      for (unsigned r = 0; r < kMaskRows; ++r)
        row_words[j][r] = 0;
      if (j < words_per_thread && w < word_count) {
#pragma unroll
        for (unsigned r = 0; r < kMaskRows; ++r) {
          if (r < live_rows) {
            row_words[j][r] = mask[size_t(query_start + r) * mask_words + w];
            bits |= row_words[j][r];
          }
        }
        if (w * 32 >= tail_block)
          bits = ~0u;
        else if ((w + 1) * 32 > tail_block)
          bits |= ~0u << (tail_block % 32);
        if ((w + 1) * 32 > n_blocks)
          bits &= (1u << (n_blocks % 32)) - 1u;
      }
      local_words[j] = bits;
      count += __popc(bits);
    }
    unsigned scan = count;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset *= 2) {
      unsigned before = __shfl_up(scan, offset, 32);
      if (lane >= offset)
        scan += before;
    }
    if (lane == 31)
      wave_counts[wave] = scan;
    __syncthreads();
    unsigned prefix = scan - count;
#pragma unroll
    for (unsigned w = 0; w < 8; ++w) {
      if (w < wave)
        prefix += wave_counts[w];
      selected += wave_counts[w];
    }
    compact = selected <= kListCapacity;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      const unsigned word = tid * words_per_thread + j;
      unsigned bits = local_words[j];
      if (compact) {
        while (bits) {
          unsigned bit = __builtin_ctz(bits);
          unsigned members = 0;
#pragma unroll
          for (unsigned r = 0; r < kMaskRows; ++r)
            members |= ((row_words[j][r] >> bit) & 1u) << r;
          union_words[prefix++] = (word * 32 + bit) | (members << kMemberShift);
          bits &= bits - 1;
        }
      } else if (j < words_per_thread && word < word_count) {
        union_words[word] = bits;
      }
    }
    __syncthreads();
  }
  // First visible block at or after `b` (n_blocks when none). Uniform across
  // the block: every thread walks the same words.
  const auto next_block = [&](std::uint32_t b) -> std::uint32_t {
    if (!sparse) {
      return b;
    }
    while (b < tail_block && b < n_blocks) {
      const std::uint32_t bits = union_words[b / 32] >> (b % 32);
      // Never jump past the tail: from tail_block on every block is visible.
      if (bits != 0) {
        return min(b + static_cast<std::uint32_t>(__builtin_ctz(bits)),
                   tail_block);
      }
      b = min(((b / 32) + 1) * 32u, tail_block);
    }
    return b;
  };
  struct Tile {
    std::uint32_t block[kBlocksPerTile];
    std::uint32_t count;
    std::uint32_t members;  // kMaskRows bits per block, compact lists only
  };
  // Gathers the next tile starting the search at block `b`; returns the
  // block to continue from.
  const auto gather_tile = [&](std::uint32_t b, Tile& tile) {
    if constexpr (!sparse) {
      // The dense window's tiles are four consecutive blocks.
      tile.count = min(kBlocksPerTile, n_blocks - min(b, n_blocks));
      tile.members = 0;
#pragma unroll
      for (std::uint32_t i = 0; i < kBlocksPerTile; ++i)
        tile.block[i] = b + i < n_blocks ? b + i : n_blocks;
      return b + tile.count;
    }
    if (compact) {
      tile.count = min(kBlocksPerTile, selected - min(b, selected));
      tile.members = 0;
#pragma unroll
      for (unsigned i = 0; i < kBlocksPerTile; ++i) {
        const unsigned entry = b + i < selected ? union_words[b + i] : 0u;
        tile.block[i] =
            b + i < selected ? entry & ((1u << kMemberShift) - 1u) : n_blocks;
        tile.members |= (entry >> kMemberShift) << (i * kMaskRows);
      }
      return b + tile.count;
    }
    tile.count = 0;
    tile.members = 0;
#pragma unroll
    for (std::uint32_t i = 0; i < kBlocksPerTile; ++i) {
      b = next_block(b);
      const bool have = b < n_blocks;
      tile.block[i] = have ? b : n_blocks;
      tile.count += have ? 1u : 0u;
      b = have ? b + 1 : b;
    }
    return b;
  };
  // Key position of tile row `r` (0..15), or context_end past the tile.
  const auto tile_key = [&](const Tile& tile, std::uint32_t r) {
    if constexpr (!sparse) {
      const std::uint32_t blk = tile.block[0] + (r / ratio);
      return blk < n_blocks ? (blk * ratio) + (r % ratio) : context_end;
    }
    const std::uint32_t i = r / ratio;
    std::uint32_t blk = n_blocks;
#pragma unroll
    for (std::uint32_t j = 0; j < kBlocksPerTile; ++j) {
      if (i == j) {
        blk = tile.block[j];
      }
    }
    return blk < n_blocks ? (blk * ratio) + (r % ratio) : context_end;
  };

  // One key per lane for V, a 16-dim slice per thread: a strided global read
  // in exchange for conflict-free transpose writes.
  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  static_assert(kKRegs * 256 == kKeys * (kHeadDim / 8), "K stages evenly");
  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  const auto* v_base =
      v_cache + (static_cast<std::size_t>(kv_head) * kHeadDim) + v_slice;
  const auto* k_base = k_cache + (static_cast<std::size_t>(kv_head) * kHeadDim);

  const auto load_v = [&](const Tile& tile, uint4* dst) {
    if constexpr (kI8Mma) {
      if (v_block_bytes == 18 || v_block_bytes == 34) {
        const std::uint32_t key_position = tile_key(tile, v_key);
        const bool live = key_position < context_end;
        const std::size_t head_bytes =
            (static_cast<std::size_t>(kHeadDim) / 32) * v_block_bytes;
        const std::size_t row_bytes = head_bytes * kWmmaKvHeads;
        const std::uint8_t* vrow =
            reinterpret_cast<const std::uint8_t*>(v_cache) +
            (static_cast<std::size_t>(live ? key_position : 0) * row_bytes) +
            (static_cast<std::size_t>(kv_head) * head_bytes);
#pragma unroll
        for (std::uint32_t j = 0; j < kVRegs; ++j) {
          uint4 reg = make_uint4(0u, 0u, 0u, 0u);
          if (live) {
            const std::uint32_t dim = v_slice + (j * 8);
            const std::uint8_t* vent = vrow + ((dim / 32) * v_block_bytes);
            auto* hp = reinterpret_cast<__half*>(&reg);
            if (v_block_bytes == 34) {
              const float sc =
                  __half2float(*reinterpret_cast<const __half*>(vent + 32));
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                const int q = static_cast<int>(
                    static_cast<std::int8_t>(vent[(dim % 32) + i]));
                hp[i] = __float2half(static_cast<float>(q) * sc);
              }
            } else {
              const float sc =
                  __half2float(*reinterpret_cast<const __half*>(vent + 16));
              std::uint32_t packed = 0;
              __builtin_memcpy(&packed, vent + ((dim % 32) / 2),
                               sizeof(std::uint32_t));
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                const int q = static_cast<int>((packed >> (4 * i)) & 0xFu);
                const int val = q - (q & 8 ? 16 : 0);
                hp[i] = __float2half(static_cast<float>(val) * sc);
              }
            }
          }
          dst[j] = reg;
        }
        return;
      }
    }
    const std::uint32_t key_position = tile_key(tile, v_key);
    const bool live = key_position < context_end;
    const auto* src =
        v_base +
        (static_cast<std::size_t>(live ? key_position : 0) * kWmmaKvWidth);
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      dst[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                    : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  // Coalesced: a wave reads one key row's 512 contiguous bytes.
  const auto load_k = [&](const Tile& tile, uint4* dst, int kbank) {
    if constexpr (kI8Mma) {
      (void)dst;
      (void)kbank;
      // One thread per (key, 16-dim slice): 256 threads cover 16 keys x
      // 16 slices. Data lands directly in the int8 LDS (no staging), so
      // this doubles as the stage step; the f16 stage_k is skipped below.
      const std::uint32_t key = tid % kKeys;
      const std::uint32_t slice = (tid / kKeys) * 16;
      const std::uint32_t key_position = tile_key(tile, key);
      const bool live = key_position < context_end;
      const std::size_t head_bytes =
          (static_cast<std::size_t>(kHeadDim) / 32) * k_block_bytes;
      const std::size_t row_bytes = head_bytes * kWmmaKvHeads;
      const std::uint8_t* k_blk_base =
          reinterpret_cast<const std::uint8_t*>(k_cache);
      const std::uint8_t* krow =
          k_blk_base +
          (static_cast<std::size_t>(live ? key_position : 0) * row_bytes) +
          (static_cast<std::size_t>(kv_head) * head_bytes) +
          ((slice / 32) * k_block_bytes);
      std::uint8_t* out = &k_i8[(kbank * kKeys * 272) + (key * 272) + slice];
      float sc = 0.0F;
      if (live) {
        if (k_block_bytes == 34) {
          sc = __half2float(*reinterpret_cast<const __half*>(
              krow + 32));  // prod: codes-first
          __builtin_memcpy(out, krow + (slice % 32), 16);
        } else {
          // 18-byte 4-bit block: 16 nibble bytes + f16 scale at byte 16.
          // Nibble code semantics match the f16 unpack path below
          // (q - (q & 8 ? 16 : 0)); only the narrowing is skipped.
          sc = __half2float(*reinterpret_cast<const __half*>(krow + 16));
          const std::uint8_t* nib = krow + ((slice % 32) / 2);
#pragma unroll
          for (std::uint32_t i = 0; i < 16; ++i) {
            const std::uint8_t byte = nib[i / 2];
            const int q = (i % 2 == 0) ? (byte & 0xFu) : (byte >> 4);
            out[i] = static_cast<std::uint8_t>(q - (q & 8 ? 16 : 0));
          }
        }
      } else {
        const std::uint8_t z[16] = {};
        __builtin_memcpy(out, z, 16);
      }
      if ((slice % 32) == 0) {
        k_scl[(kbank * kKeys * 8) + (key * 8) + (slice / 32)] = sc;
      }
      return;
    }
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position = tile_key(tile, idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      const bool live = key_position < context_end;
      dst[n] =
          live ? *reinterpret_cast<const uint4*>(
                     k_base +
                     (static_cast<std::size_t>(key_position) * kWmmaKvWidth) +
                     d8)
               : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  const auto stage_k = [&](const uint4* regs) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&k_lds[(key_row * kWmmaKStride) + d8]) =
          regs[n];
    }
  };
  const auto stage_v = [&](const uint4* regs) {
    // Neighbouring lanes hold neighbouring keys of the same dims. Swapping
    // one half with the partner lets each lane store a key pair of one dim
    // as a word: the even lane the even dim, the odd lane the odd dim.
    const bool odd = (v_key & 1u) != 0u;
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      const std::uint32_t words[4] = {regs[j].x, regs[j].y, regs[j].z,
                                      regs[j].w};
#pragma unroll
      for (std::uint32_t w = 0; w < 4; ++w) {
        const std::uint32_t own = words[w];
        const auto partner =
            static_cast<std::uint32_t>(__builtin_amdgcn_mov_dpp(
                static_cast<int>(own), 0xB1, 0xF, 0xF, false));
        const std::uint32_t pair =
            odd ? __builtin_amdgcn_perm(partner, own, 0x03020706U)
                : __builtin_amdgcn_perm(partner, own, 0x05040100U);
        const std::uint32_t dim = v_slice + (j * 8) + (2 * w) + (odd ? 1 : 0);
        *reinterpret_cast<std::uint32_t*>(
            &v_lds[(dim * kVtStride) + (v_key & ~1u)]) = pair;
      }
    }
  };

  // --- S = Q K^T of the tile in k_lds. Each half writes its own slot; the
  // reader sums them.
  const auto scores = [&](int sbank) {
    if constexpr (kI8Mma) {
      if (wave < 2 * kSTiles) {
        // Per-32-dim-block int32 accumulate, immediately rescaled by
        // q_scale[row][blk] * k_scale[col][blk] (fp32). Row/col mapping is
        // the validated oracle layout: acc[l] = (row 2*l+half, col sub).
        float s_acc[8] = {};
#pragma unroll
        for (std::uint32_t bb = 0; bb < 4; ++bb) {
          const std::uint32_t blk = (s_kh * 4) + bb;
          int32x8_t c0l = {0, 0, 0, 0, 0, 0, 0, 0};
          int32x8_t c0h = {0, 0, 0, 0, 0, 0, 0, 0};
          int32x8_t c1 = {0, 0, 0, 0, 0, 0, 0, 0};
          int32x4_t a0, a1, r0, r1, b0, b1;
          __builtin_memcpy(&a0, &q0w[(bb * 8)], 16);
          __builtin_memcpy(&a1, &q0w[(bb * 8) + 4], 16);
          __builtin_memcpy(&r0, &q1w[(bb * 8)], 16);
          __builtin_memcpy(&r1, &q1w[(bb * 8) + 4], 16);
          __builtin_memcpy(
              &b0, &k_i8[(sbank * kKeys * 272) + (sub * 272) + (blk * 32)], 16);
          __builtin_memcpy(
              &b1, &k_i8[(sbank * kKeys * 272) + (sub * 272) + (blk * 32) + 16],
              16);
          c0l = WmmaI8(a0, b0, c0l);
          c0h = WmmaI8(a1, b1, c0h);
          c1 = WmmaI8(r0, b0, c1);
          c1 = WmmaI8(r1, b1, c1);
          const float kscale = k_scl[(sbank * kKeys * 8) + (sub * 8) + blk];
#pragma unroll
          for (std::uint32_t l = 0; l < 8; ++l) {
            const std::uint32_t row = (s_rb * 16) + ((2 * l) + half_id);
            s_acc[l] +=
                attention_scale * kscale *
                (q0_scl[(row * 16) + (blk * 2)] * static_cast<float>(c0l[l]) +
                 q0_scl[(row * 16) + (blk * 2) + 1] *
                     static_cast<float>(c0h[l]) +
                 q1_scl[(row * 8) + blk] * static_cast<float>(c1[l]));
          }
        }
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
        }
      }
    }

    else if (wave < 2 * kSTiles) {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&k_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = Wmma(q_frag[ks], k_frag, s_acc);
        if constexpr (kPackHeads) {
          // Bound temporary K fragments without changing the dot-product
          // sequence. Hoisting all eight LDS fragments raises register use.
          if ((ks + 1) % 2 == 0)
            __builtin_amdgcn_sched_barrier(0);
        }
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
  };

  // --- online softmax of the scores in s_lds: kSoftmaxLanes threads per
  // query row ---
  const auto softmax = [&](const Tile& tile) {
    const std::uint32_t rg = tid / kSoftmaxLanes;
    if (rg < kRows) {
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query = row_query(rb, row);
      const bool live_row = row_live(rb, row);
      const std::uint32_t absolute_query = start_pos + local_query;
      // Keys in the query's own incomplete block are always visible; earlier
      // blocks follow the selection mask.
      const std::uint32_t tail_start = ((absolute_query + 1) / ratio) * ratio;
      // The dense window runs without a selection mask.
      const std::uint32_t* words =
          !sparse || mask == nullptr || !live_row
              ? nullptr
              : mask + static_cast<std::size_t>(local_query) * mask_words;
      float part_max = -INFINITY;
      float vals[kPerLane];
      float s0v[kPerLane];
      float s1v[kPerLane];
      // A lane's four keys form one selection block, so the sparse window
      // decides its mask bit once per lane instead of once per key.
      static_assert(kPerLane == ratio, "one selection block per lane");
      bool block_selected = true;
      std::uint32_t blk = 0;
      if constexpr (sparse) {
        blk = n_blocks;
#pragma unroll
        for (std::uint32_t j = 0; j < kBlocksPerTile; ++j) {
          if (seg == j) {
            blk = tile.block[j];
          }
        }
        if (words != nullptr && blk < n_blocks && blk * ratio < tail_start) {
          const std::uint32_t member =
              (seg * kMaskRows) + (local_query - query_start);
          block_selected = compact
                               ? ((tile.members >> member) & 1u) != 0u
                               : ((words[blk / 32] >> (blk % 32)) & 1u) != 0u;
        }
      }
      if constexpr (kVectorSoftmax) {
        const float4 s0 = *reinterpret_cast<const float4*>(
            &s_lds[0][rb][row][seg * kPerLane]);
        const float4 s1 = *reinterpret_cast<const float4*>(
            &s_lds[1][rb][row][seg * kPerLane]);
        s0v[0] = s0.x, s0v[1] = s0.y, s0v[2] = s0.z, s0v[3] = s0.w;
        s1v[0] = s1.x, s1v[1] = s1.y, s1v[2] = s1.z, s1v[3] = s1.w;
      } else {
        // Only lanes with a visible block read their scores, which saves
        // LDS bandwidth, but under one condition so the reads share a wait.
#pragma unroll
        for (std::uint32_t m = 0; m < kPerLane; ++m)
          s0v[m] = s1v[m] = 0.0F;
        if (live_row && block_selected && blk < n_blocks) {
#pragma unroll
          for (std::uint32_t m = 0; m < kPerLane; ++m) {
            const std::uint32_t col = (seg * kPerLane) + m;
            const std::uint32_t s_index = ((col / 16) * kRowBlocks) + rb;
            s0v[m] = s_lds[0][s_index][row][col % 16];
            s1v[m] = s_lds[1][s_index][row][col % 16];
          }
        }
      }
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = tile_key(tile, col);
        // Gemma 4 SWA: a token only sees the last `window` positions
        // (llama.cpp clamps lo = pos+1-window). window==0 means full causal.
        const bool in_window =
            window == 0 || key_position + window > absolute_query;
        const bool valid = live_row && key_position <= absolute_query &&
                           key_position < context_end && in_window &&
                           block_selected;
        vals[m] = valid ? (s0v[m] + s1v[m]) : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, XorLane(part_max, off));
      }
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
      __half p[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p[m] = static_cast<__half>(w);
        if constexpr (!kVectorSoftmax)
          p_lds[rg][(seg * kPerLane) + m] = p[m];
      }
      if constexpr (kVectorSoftmax)
        __builtin_memcpy(&p_lds[rg][seg * kPerLane], p, sizeof(p));
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += XorLane(part_sum, off);
      }
      running_max = next_max;
      // The completed partial sum, then one fused update: written out so
      // fast-math cannot reassociate the reduction into the update.
      running_sum = __fmaf_rn(prior_scale, running_sum, part_sum);
      if (seg == 0) {
        row_scale[kVectorSoftmax ? (rb * 16) + ((row % 2) * 8) + (row / 2)
                                 : rg] = prior_scale;
      }
    }
  };

  // --- rescale the running O by the tile's new maximum, then O += P V. A
  // lane touches only rows 2i + half_id, so the factors are read once per
  // row block.
  const auto accumulate = [&](const Tile& tile) {
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale[8];
      if constexpr (kVectorSoftmax) {
        const auto* factors = reinterpret_cast<const float4*>(
            &row_scale[(rb * 16) + (half_id * 8)]);
        const float4 f0 = factors[0];
        const float4 f1 = factors[1];
        scale[0] = f0.x, scale[1] = f0.y, scale[2] = f0.z, scale[3] = f0.w;
        scale[4] = f1.x, scale[5] = f1.y, scale[6] = f1.z, scale[7] = f1.w;
      } else {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i)
          scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }
#pragma unroll
    for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
      v16h v_frag[kKeyBlocks];
#pragma unroll
      for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
        v_frag[kb] =
            LoadFrag(&v_lds[(((dim_tile * 16) + sub) * kVtStride) + (kb * 16)]);
      }
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          v8f next = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
          if constexpr (!kPackHeads) {
            const std::uint32_t first_key = tile.block[0] * ratio;
            // WMMA's dot accumulation can change its last bits when zero
            // probabilities multiply populated future values instead of the
            // zero padding at a chunk boundary. Accumulate each query's final
            // partial tile over its visible keys only. Earlier complete tiles
            // retain the matrix-core path.
            if (first_key + kKeys - 1 > start_pos + query_start) {
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                const std::uint32_t row = (2 * i) + half_id;
                const std::uint32_t absolute_query =
                    start_pos + row_query(rb, row);
                if (absolute_query < first_key + kKeys - 1) {
                  float acc = o_acc[rb][t][i];
                  for (std::uint32_t key = 0;
                       key < kKeys && first_key + key <= absolute_query;
                       ++key) {
                    acc = fmaf(
                        __half2float(p_lds[(rb * 16) + row][key]),
                        __half2float(
                            v_lds[((dim_tile * 16) + sub) * kVtStride + key]),
                        acc);
                  }
                  next[i] = acc;
                }
              }
            }
          }
          o_acc[rb][t] = next;
        }
      }
    }
  };

  // Software-pipelined key sweep. Phase A of step j accumulates tile j's PV
  // while the score waves compute tile j + 1's S; phase B runs tile j + 1's
  // softmax while staging K for tile j + 2 and V for tile j + 1, whose
  // registers were loaded one step earlier. Every row sees the same S, P,
  // rescale and accumulation sequence as a sequential sweep.
  uint4 k_regs[kKRegs] = {};
  uint4 v_regs[kVRegs] = {};
  // int8 K banks: loads run one tile ahead of scores; the tile being
  // scored and the tile being loaded must live in different banks.
  int bcur = 0, bnext = 1, bfree = 0, bk_tile = 0;
  Tile cur;     // tile j: P in p_lds, V in v_lds
  Tile next;    // tile j + 1: K in k_lds
  Tile k_tile;  // tile j + 2: K in k_regs
  Tile v_tile;  // tile j + 1 V in v_regs
  std::uint32_t cursor = gather_tile(0, cur);
  if (cur.count != 0) {
    load_k(cur, k_regs, bcur);
    load_v(cur, v_regs);
  }
  if constexpr (!kI8Mma)
    stage_k(k_regs);
  stage_v(v_regs);
  cursor = gather_tile(cursor, next);
  if (next.count != 0) {
    load_k(next, k_regs, bnext);
    load_v(next, v_regs);
  }
  SyncLds();
  if (cur.count != 0)
    scores(bcur);
  SyncLds();
  if (cur.count != 0)
    softmax(cur);
  if constexpr (!kI8Mma)
    stage_k(k_regs);
  v_tile = next;
  cursor = gather_tile(cursor, k_tile);
  if (k_tile.count != 0) {
    load_k(k_tile, k_regs, bfree);
    bk_tile = bfree;
    bfree ^= 1;
  }
  SyncLds();

  while (cur.count != 0) {
    // Phase A: k_lds = K(j + 1), v_lds = V(j), p_lds = P(j).
    if (next.count != 0)
      scores(bnext);
    accumulate(cur);
    // Finish PV before the barrier. Left alone, the compiler sinks these
    // WMMAs into phase B behind the softmax, which is the critical path.
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb)
#pragma unroll
      for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t)
        asm volatile("" : "+v"(o_acc[rb][t]));
    SyncLds();
    // Phase B.
    if (next.count != 0)
      softmax(next);
    stage_v(v_regs);
    if constexpr (!kI8Mma)
      stage_k(k_regs);
    cur = next;
    next = k_tile;
    bcur = bnext;
    bnext = bk_tile;
    v_tile = next;
    cursor = gather_tile(cursor, k_tile);
    if (k_tile.count != 0) {
      load_k(k_tile, k_regs, bfree);
      bk_tile = bfree;
      bfree ^= 1;
    }
    if (v_tile.count != 0)
      load_v(v_tile, v_regs);
    SyncLds();
  }
  if (tid < kRows * kSoftmaxLanes && tid % kSoftmaxLanes == 0)
    row_sum[tid / kSoftmaxLanes] = running_sum;
  SyncLds();

  // --- epilogue: normalize and apply the sigmoid output gate ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
    for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        if (!row_live(rb, row)) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(row_query(rb, row)) * kWmmaAttnWidth) +
            (static_cast<std::size_t>(row_head(rb, row)) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr) {
          value *= SigmoidF(gate[offset]);
        }
        out[offset] = value;
      }
    }
  }
}

// W8A8 int8 WMMA GEMM over the GGUF Q8_0 weights, ported from the Qwen 27B
// route (src/models/qwen/hip/kernels/prefill_quant_gemm.hip,
// opt-c163-blocked-w8a8 with the opt-c179 addressing). Weights stay in their
// row-major 34-byte block_q8_0 layout; activations are quantized per 32-wide
// block into WMMA B-fragment order:
//
//   tile(tt, kb) = [b0: 16 tokens x 16 bytes]   offset   0
//                  [b1: 16 tokens x 16 bytes]   offset 256
//                  [16 fp32 token scales    ]   offset 512
//
// with tt = token / 16 and kb the 32-element K block, 576 bytes per tile.
// Launch the WMMA prefill attention for Gemma 4's sliding-window geometry.
// Returns false when the geometry is unsupported so the caller falls back to
// the scalar kernel. `window` is currently ignored (dense causal) -- only
// valid while the visible context is <= the sliding window, which the caller
// must guarantee; wired behind GEMMA4_ATTN_WMMA for validation.
bool AttentionWmma(const float* q, const __half* k_cache, const __half* v_cache,
                   float* out, std::uint32_t n_tokens, std::uint32_t start_pos,
                   std::uint32_t d, std::uint32_t kv_stride,
                   std::uint32_t window, hipStream_t stream) {
  if (d != kWmmaHeadDim) return false;
  if (kv_stride != kWmmaKvWidth) return false;
  if (n_tokens == 0) return true;
  const dim3 grid((n_tokens + kWmmaQueryRows - 1) / kWmmaQueryRows,
                  kWmmaKvHeads * (kWmmaGqa / kWmmaHeads));
  hipLaunchKernelGGL(
      (WmmaCausalAttentionKernel<kWmmaQueryRows, kWmmaKeys, false, false>),
      grid, dim3(256), 0, stream, q, nullptr, k_cache, v_cache, nullptr, 0u,
      out, start_pos, n_tokens, 0u, 0u, 0u, window);
  return true;
}

}  // namespace gufo::models::gemma4::kernels::wmma
#endif  // ENGINE_ENABLE_HIP
