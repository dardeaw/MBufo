#include "src/models/inkling/dspark.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/compute/gemm/gemm.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/inkling/kernels/kernels.hpp"

namespace mbufo::models::inkling {
namespace {

inline bool CheckHip(hipError_t err, std::string* msg, const char* what) {
  if (err == hipSuccess) return true;
  if (msg != nullptr) {
    *msg = std::string("inkling-dspark: ") + what + ": " +
           hipGetErrorString(err);
  }
  return false;
}

__global__ void Mix4CtxKernel(const float* __restrict__ h0,
                              const float* __restrict__ h1,
                              const float* __restrict__ h2,
                              const float* __restrict__ h3,
                              const float* __restrict__ mix,
                              float* __restrict__ out, std::size_t total) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < total) {
    out[i] = mix[0] * h0[i] + mix[1] * h1[i] + mix[2] * h2[i] + mix[3] * h3[i];
  }
}

__global__ void HeadRmsNormRopeF16StoreKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ q_norm,
    const float* __restrict__ k_norm, __half* __restrict__ k_cache,
    __half* __restrict__ v_cache, float* __restrict__ q_out,
    std::uint32_t start_pos, std::uint32_t q_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, float eps, float rope_theta) {
  // grid = (n_tokens, q_heads + kv_heads), block = 64 (pairs of half_dim=64)
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t blk_h = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;  // 0..63 -> indices tid and tid + 64
  const std::uint32_t pos = start_pos + tok;

  if (blk_h < q_heads) {
    const std::uint32_t h = blk_h;
    const float* src =
        q + (static_cast<std::size_t>(tok) * q_heads + h) * head_dim;
    float* dst =
        q_out + (static_cast<std::size_t>(tok) * q_heads + h) * head_dim;
    const std::uint32_t i0 = tid;
    const std::uint32_t i1 = tid + (head_dim >> 1);
    const float x0 = src[i0];
    const float x1 = src[i1];
    float ss = x0 * x0 + x1 * x1;
    for (int off = 16; off > 0; off /= 2) ss += __shfl_xor(ss, off);
    __shared__ float sm[2];
    if ((tid & 31U) == 0U) sm[tid >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sm[0] + sm[1]) / static_cast<float>(head_dim) + eps);
    const float y0 = x0 * inv * q_norm[i0];
    const float y1 = x1 * inv * q_norm[i1];
    const float freq =
        powf(rope_theta, -2.0F * static_cast<float>(tid) /
                             static_cast<float>(head_dim));
    const float ang = static_cast<float>(pos) * freq;
    float s, c;
    __sincosf(ang, &s, &c);
    dst[i0] = y0 * c - y1 * s;
    dst[i1] = y0 * s + y1 * c;
  } else {
    const std::uint32_t h = blk_h - q_heads;
    const float* ksrc =
        k + (static_cast<std::size_t>(tok) * kv_heads + h) * head_dim;
    const float* vsrc =
        v + (static_cast<std::size_t>(tok) * kv_heads + h) * head_dim;
    const std::size_t kv_off =
        (static_cast<std::size_t>(tok) * kv_heads + h) * head_dim;
    const std::uint32_t i0 = tid;
    const std::uint32_t i1 = tid + (head_dim >> 1);
    const float x0 = ksrc[i0];
    const float x1 = ksrc[i1];
    float ss = x0 * x0 + x1 * x1;
    for (int off = 16; off > 0; off /= 2) ss += __shfl_xor(ss, off);
    __shared__ float sm[2];
    if ((tid & 31U) == 0U) sm[tid >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((sm[0] + sm[1]) / static_cast<float>(head_dim) + eps);
    const float y0 = x0 * inv * k_norm[i0];
    const float y1 = x1 * inv * k_norm[i1];
    const float freq =
        powf(rope_theta, -2.0F * static_cast<float>(tid) /
                             static_cast<float>(head_dim));
    const float ang = static_cast<float>(pos) * freq;
    float s, c;
    __sincosf(ang, &s, &c);
    k_cache[kv_off + i0] = __float2half(y0 * c - y1 * s);
    k_cache[kv_off + i1] = __float2half(y0 * s + y1 * c);
    v_cache[kv_off + i0] = __float2half(vsrc[i0]);
    v_cache[kv_off + i1] = __float2half(vsrc[i1]);
  }
}

__global__ void StoreCtxKvF16Kernel(
    const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ k_norm, const float* __restrict__ v_norm,
    __half* __restrict__ k_cache, __half* __restrict__ v_cache,
    std::uint32_t start_pos, std::uint32_t kv_heads, std::uint32_t head_dim,
    float eps, float rope_theta) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;  // 0..63
  const std::uint32_t pos = start_pos + tok;

  const float* ksrc =
      k + (static_cast<std::size_t>(tok) * kv_heads + h) * head_dim;
  const float* vsrc =
      v + (static_cast<std::size_t>(tok) * kv_heads + h) * head_dim;
  const std::size_t kv_off =
      (static_cast<std::size_t>(pos) * kv_heads + h) * head_dim;

  // K norm + RoPE (interleaved)
  const std::uint32_t i0 = tid;
  const std::uint32_t i1 = tid + (head_dim >> 1);
  const float k0 = ksrc[i0];
  const float k1 = ksrc[i1];
  float sk = k0 * k0 + k1 * k1;
  for (int off = 16; off > 0; off /= 2) sk += __shfl_xor(sk, off);
  __shared__ float sm[2];
  if ((tid & 31U) == 0U) sm[tid >> 5] = sk;
  __syncthreads();
  const float invk = rsqrtf((sm[0] + sm[1]) / static_cast<float>(head_dim) + eps);
  const float ky0 = k0 * invk * k_norm[i0];
  const float ky1 = k1 * invk * k_norm[i1];
  const float freq =
      powf(rope_theta, -2.0F * static_cast<float>(tid) /
                           static_cast<float>(head_dim));
  const float ang = static_cast<float>(pos) * freq;
  float s, c;
  __sincosf(ang, &s, &c);
  k_cache[kv_off + i0] = __float2half(ky0 * c - ky1 * s);
  k_cache[kv_off + i1] = __float2half(ky0 * s + ky1 * c);

  // V norm (if v_norm != nullptr)
  const float v0 = vsrc[i0];
  const float v1 = vsrc[i1];
  if (v_norm != nullptr) {
    float sv = v0 * v0 + v1 * v1;
    for (int off = 16; off > 0; off /= 2) sv += __shfl_xor(sv, off);
    if ((tid & 31U) == 0U) sm[tid >> 5] = sv;
    __syncthreads();
    const float invv = rsqrtf((sm[0] + sm[1]) / static_cast<float>(head_dim) + eps);
    v_cache[kv_off + i0] = __float2half(v0 * invv * v_norm[i0]);
    v_cache[kv_off + i1] =
        __float2half(v1 * invv * v_norm[i1]);
  } else {
    v_cache[kv_off + i0] = __float2half(v0);
    v_cache[kv_off + i1] = __float2half(v1);
  }
}

__global__ void DsparkBlockAttnKernel(
    const float* __restrict__ q, const __half* __restrict__ k_ctx,
    const __half* __restrict__ v_ctx, const __half* __restrict__ k_self,
    const __half* __restrict__ v_self, float* __restrict__ out,
    std::uint32_t anchor_pos, std::uint32_t n_block, std::uint32_t window,
    std::uint32_t q_heads, std::uint32_t kv_heads, float scale) {
  // Non-causal attention over [ctx(lo..anchor_pos-1), self(0..n_block-1)].
  // grid = (n_block, q_heads), block = 32 (4 elems per lane, head_dim=128)
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t qh = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t kvh = qh / (q_heads / kv_heads);
  const float4 q4 = *reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(tok) * q_heads + qh) * 128U + lane * 4U);
  const std::size_t kv_stride = static_cast<std::size_t>(kv_heads) * 128U;
  const std::size_t kv_off_h = static_cast<std::size_t>(kvh) * 128U + lane * 4U;

  const std::uint32_t ctx_end = anchor_pos;
  const std::uint32_t ctx_lo =
      (window > 0 && anchor_pos + tok + 1U > window)
          ? (anchor_pos + tok + 1U - window)
          : 0U;

  float m = -1e30F;
  float l = 0.0F;
  float4 acc = make_float4(0.0F, 0.0F, 0.0F, 0.0F);

  auto attend = [&](const __half* kc, const __half* vc, std::uint32_t p) {
    const std::size_t off = static_cast<std::size_t>(p) * kv_stride + kv_off_h;
    const uint2 kr = *reinterpret_cast<const uint2*>(kc + off);
    const __half2* kh = reinterpret_cast<const __half2*>(&kr);
    const float2 k0 = __half22float2(kh[0]);
    const float2 k1 = __half22float2(kh[1]);
    float dot = q4.x * k0.x + q4.y * k0.y + q4.z * k1.x + q4.w * k1.y;
    for (int o = 16; o > 0; o /= 2) dot += __shfl_xor(dot, o);
    const float sc = dot * scale;
    const float m_new = fmaxf(m, sc);
    const float alpha = __expf(m - m_new);
    const float w = __expf(sc - m_new);
    const uint2 vr = *reinterpret_cast<const uint2*>(vc + off);
    const __half2* vh = reinterpret_cast<const __half2*>(&vr);
    const float2 v0 = __half22float2(vh[0]);
    const float2 v1 = __half22float2(vh[1]);
    acc.x = acc.x * alpha + w * v0.x;
    acc.y = acc.y * alpha + w * v0.y;
    acc.z = acc.z * alpha + w * v1.x;
    acc.w = acc.w * alpha + w * v1.y;
    l = l * alpha + w;
    m = m_new;
  };

  for (std::uint32_t p = ctx_lo; p < ctx_end; ++p) attend(k_ctx, v_ctx, p);
  for (std::uint32_t p = 0; p < n_block; ++p) attend(k_self, v_self, p);

  const float inv = (l > 0.0F) ? (1.0F / l) : 0.0F;
  *reinterpret_cast<float4*>(
      out + (static_cast<std::size_t>(tok) * q_heads + qh) * 128U + lane * 4U) =
      make_float4(acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv);
}

__global__ void SwiGluMulKernel(const float* __restrict__ gate,
                                const float* __restrict__ up,
                                float* __restrict__ out, std::size_t total) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < total) {
    const float g = gate[i];
    out[i] = (g / (1.0F + __expf(-g))) * up[i];
  }
}

__global__ void AddResidualKernel(float* __restrict__ x,
                                  const float* __restrict__ y,
                                  std::size_t total) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < total) x[i] += y[i];
}

__global__ void CopyMaskEmbdKernel(const float* __restrict__ mask_embd,
                                   float* __restrict__ dst,
                                   std::uint32_t n_draft, std::uint32_t h) {
  const std::uint32_t r = blockIdx.x;  // 0..n_draft-1 -> row r+1 of dst
  const std::uint32_t c = blockIdx.y * blockDim.x + threadIdx.x;
  if (r < n_draft && c < h) {
    dst[static_cast<std::size_t>(r + 1U) * h + c] =
        mask_embd[static_cast<std::size_t>(r) * h + c];
  }
}

__device__ __forceinline__ float Bf16ToF32(std::uint16_t u) {
  return __uint_as_float(static_cast<std::uint32_t>(u) << 16);
}

__global__ void MarkovBuildZKernel(
    const float* __restrict__ state, const std::uint16_t* __restrict__ w1,
    const std::int32_t* __restrict__ d_prev_tok, const float* __restrict__ h_k,
    const float* __restrict__ mk_ctx, float* __restrict__ z,
    std::uint32_t rank, std::uint32_t hidden, std::uint32_t vocab) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t total = 2U * rank + 2U * hidden;
  if (i >= total) return;
  if (i < rank) {
    z[i] = state[i];
  } else if (i < 2U * rank) {
    const std::int32_t raw = *d_prev_tok;
    const std::uint32_t tok =
        (raw >= 0 && static_cast<std::uint32_t>(raw) < vocab)
            ? static_cast<std::uint32_t>(raw)
            : 0U;
    z[i] = (raw >= 0 && static_cast<std::uint32_t>(raw) < vocab)
               ? Bf16ToF32(w1[static_cast<std::size_t>(tok) * rank + (i - rank)])
               : 0.0F;
  } else if (i < 2U * rank + hidden) {
    z[i] = h_k[i - 2U * rank];
  } else {
    z[i] = mk_ctx[i - 2U * rank - hidden];
  }
}

__global__ void JointProjBf16Kernel(const std::uint16_t* __restrict__ w,
                                    const float* __restrict__ bias,
                                    const float* __restrict__ z,
                                    float* __restrict__ out, std::uint32_t rows,
                                    std::uint32_t cols) {
  // grid = rows, block = 128 (4 wave32s)
  const std::uint32_t r = blockIdx.x;
  if (r >= rows) return;
  const std::uint32_t tid = threadIdx.x;
  const std::uint16_t* wrow = w + static_cast<std::size_t>(r) * cols;
  float sum = 0.0F;
  for (std::uint32_t c = tid; c < cols; c += 128U) {
    sum += Bf16ToF32(wrow[c]) * z[c];
  }
  for (int off = 16; off > 0; off /= 2) sum += __shfl_xor(sum, off);
  __shared__ float sm[4];
  if ((tid & 31U) == 0U) sm[tid >> 5] = sum;
  __syncthreads();
  if (tid == 0U) {
    out[r] = sm[0] + sm[1] + sm[2] + sm[3] + bias[r];
  }
}

__global__ void GruStepKernel(const float* __restrict__ proj,
                              float* __restrict__ state,
                              float* __restrict__ out, std::uint32_t rank) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rank) return;
  const float g = 1.0F / (1.0F + __expf(-proj[i]));
  const float c = tanhf(proj[rank + i]);
  state[i] = g * state[i] + (1.0F - g) * c;
  out[i] = tanhf(proj[2U * rank + i]);
}

__global__ void MarkovW2AddKernel(const std::uint16_t* __restrict__ w2,
                                  const float* __restrict__ bias,
                                  const float* __restrict__ out_vec,
                                  float* __restrict__ logits,
                                  std::uint32_t vocab, std::uint32_t rank) {
  // 8 threads per vocab row -> 32 rows per 256-thread block
  __shared__ float s_out[256];
  const std::uint32_t tid = threadIdx.x;
  if (tid < rank) s_out[tid] = out_vec[tid];
  __syncthreads();

  const std::uint32_t row_in_blk = tid >> 3;
  const std::uint32_t lane8 = tid & 7U;
  const std::uint32_t v = blockIdx.x * 32U + row_in_blk;
  if (v >= vocab) return;

  const std::uint16_t* wrow = w2 + static_cast<std::size_t>(v) * rank;
  const std::uint32_t k0 = lane8 * 32U;
  float sum = 0.0F;
#pragma unroll
  for (std::uint32_t j = 0; j < 32U; j += 4U) {
    const uint2 raw = *reinterpret_cast<const uint2*>(wrow + k0 + j);
    const std::uint16_t* u = reinterpret_cast<const std::uint16_t*>(&raw);
    sum += Bf16ToF32(u[0]) * s_out[k0 + j] +
           Bf16ToF32(u[1]) * s_out[k0 + j + 1U] +
           Bf16ToF32(u[2]) * s_out[k0 + j + 2U] +
           Bf16ToF32(u[3]) * s_out[k0 + j + 3U];
  }
  for (int off = 4; off > 0; off /= 2) sum += __shfl_xor(sum, off);
  if (lane8 == 0U) {
    logits[v] += sum + bias[v];
  }
}

}  // namespace

struct DsparkModel::Impl {
  struct Layer {
    const float* ctx_mix{nullptr};
    const float* ctx_norm{nullptr};
    const float* attn_norm{nullptr};
    const float* ffn_norm{nullptr};
    const float* attn_q_norm{nullptr};
    const float* attn_k_norm{nullptr};
    const float* attn_k_ctx_norm{nullptr};
    const float* attn_v_ctx_norm{nullptr};
    const void* attn_q{nullptr};
    const void* attn_k{nullptr};
    const void* attn_v{nullptr};
    const void* attn_k_ctx{nullptr};
    const void* attn_v_ctx{nullptr};
    const void* attn_output{nullptr};
    const void* ffn_gate{nullptr};
    const void* ffn_up{nullptr};
    const void* ffn_down{nullptr};
    bool sliding{true};
  };

  std::unique_ptr<core::GgufReader> reader;
  std::vector<void*> device_allocs;

  std::uint32_t hidden{2304};
  std::uint32_t ff{9216};
  std::uint32_t q_heads{24};
  std::uint32_t kv_heads{8};
  std::uint32_t head_dim{128};
  std::uint32_t window{1025};
  float rope_theta{1000000.0F};
  float rms_eps{1e-6F};
  float softcap{30.0F};

  const float* mask_embd{nullptr};
  const float* token_embd_norm{nullptr};
  const float* output_norm{nullptr};
  bool has_markov{false};
  std::uint32_t markov_rank{256};
  std::uint32_t markov_vocab{0};
  const float* markov_ctx_mix{nullptr};
  const float* markov_ctx_norm{nullptr};
  const std::uint16_t* joint_proj_w{nullptr};
  const float* joint_proj_b{nullptr};
  const std::uint16_t* markov_w1{nullptr};
  const std::uint16_t* markov_w2{nullptr};
  const float* markov_w2_b{nullptr};
  std::array<Layer, kNumLayers> layers;

  ~Impl() {
    for (void* p : device_allocs) {
      if (p != nullptr) (void)hipFree(p);
    }
  }
};

DsparkModel::~DsparkModel() = default;

std::unique_ptr<DsparkModel> DsparkModel::Load(const std::string& dspark_path,
                                               std::uint32_t hidden_size,
                                               std::uint32_t vocab_size,
                                               std::string* error_msg) {
  (void)vocab_size;
  auto reader = core::GgufReader::OpenFile(dspark_path, error_msg);
  if (!reader) return nullptr;
  if (reader->GetMetadataString("general.architecture") != "dflash") {
    if (error_msg != nullptr) *error_msg = "dspark: expected dflash gguf";
    return nullptr;
  }
  const std::uint32_t h =
      reader->GetMetadataUint32("dflash.embedding_length").value_or(0);
  if (h != hidden_size) {
    if (error_msg != nullptr) *error_msg = "dspark: hidden_size mismatch";
    return nullptr;
  }

  auto m = std::unique_ptr<DsparkModel>(new DsparkModel());
  m->impl_ = std::make_unique<Impl>();
  auto& im = *m->impl_;
  im.reader = std::move(reader);
  im.hidden = h;
  im.ff = im.reader->GetMetadataUint32("dflash.feed_forward_length").value_or(9216);
  im.q_heads = im.reader->GetMetadataUint32("dflash.attention.head_count").value_or(24);
  im.kv_heads = im.reader->GetMetadataUint32("dflash.attention.head_count_kv").value_or(8);
  im.head_dim = im.reader->GetMetadataUint32("dflash.attention.key_length").value_or(128);
  im.window = im.reader->GetMetadataUint32("dflash.attention.sliding_window").value_or(1025);
  im.rope_theta = im.reader->GetMetadataFloat32("dflash.rope.freq_base").value_or(1000000.0F);
  im.rms_eps = im.reader->GetMetadataFloat32("dflash.attention.layer_norm_rms_epsilon").value_or(1e-6F);
  im.softcap = im.reader->GetMetadataFloat32("dflash.final_logit_softcapping").value_or(30.0F);

  auto upload = [&](const std::string& name, const void** out_ptr,
                    core::GgmlType expect_type) -> bool {
    const auto* t = im.reader->FindTensor(name);
    if (t == nullptr || t->type != expect_type) {
      if (error_msg != nullptr) {
        *error_msg = "dspark: missing/bad tensor " + name;
      }
      return false;
    }
    const std::size_t bytes =
        expect_type == core::GgmlType::kF32
            ? t->ElementCount() * sizeof(float)
            : (expect_type == core::GgmlType::kBF16
                   ? t->ElementCount() * sizeof(std::uint16_t)
                   : (t->ElementCount() / 32ULL) * 34ULL);
    void* dev = nullptr;
    if (!CheckHip(hipMalloc(&dev, bytes), error_msg, name.c_str()) ||
        !CheckHip(hipMemcpy(dev, t->data, bytes, hipMemcpyHostToDevice),
                  error_msg, name.c_str())) {
      if (dev != nullptr) (void)hipFree(dev);
      return false;
    }
    im.device_allocs.push_back(dev);
    m->resident_bytes_ += bytes;
    *out_ptr = dev;
    return true;
  };
  auto up_f32 = [&](const std::string& n, const float** p) {
    return upload(n, reinterpret_cast<const void**>(p), core::GgmlType::kF32);
  };
  auto up_q8 = [&](const std::string& n, const void** p) {
    return upload(n, p, core::GgmlType::kQ8_0);
  };
  auto up_bf16 = [&](const std::string& n, const std::uint16_t** p) {
    return upload(n, reinterpret_cast<const void**>(p), core::GgmlType::kBF16);
  };

  if (!up_f32("mask_embd.weight", &im.mask_embd) ||
      !up_f32("token_embd_norm.weight", &im.token_embd_norm) ||
      !up_f32("output_norm.weight", &im.output_norm)) {
    return nullptr;
  }

  for (std::uint32_t i = 0; i < kNumLayers; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    auto& l = im.layers[i];
    l.sliding = (i != 1U);  // [True, False, True]
    if (!up_f32(p + "ctx_mix.weight", &l.ctx_mix) ||
        !up_f32(p + "ctx_norm.weight", &l.ctx_norm) ||
        !up_f32(p + "attn_norm.weight", &l.attn_norm) ||
        !up_f32(p + "ffn_norm.weight", &l.ffn_norm) ||
        !up_f32(p + "attn_q_norm.weight", &l.attn_q_norm) ||
        !up_f32(p + "attn_k_norm.weight", &l.attn_k_norm) ||
        !up_f32(p + "attn_k_ctx_norm.weight", &l.attn_k_ctx_norm) ||
        !up_q8(p + "attn_q.weight", &l.attn_q) ||
        !up_q8(p + "attn_k.weight", &l.attn_k) ||
        !up_q8(p + "attn_v.weight", &l.attn_v) ||
        !up_q8(p + "attn_k_ctx.weight", &l.attn_k_ctx) ||
        !up_q8(p + "attn_v_ctx.weight", &l.attn_v_ctx) ||
        !up_q8(p + "attn_output.weight", &l.attn_output) ||
        !up_q8(p + "ffn_gate.weight", &l.ffn_gate) ||
        !up_q8(p + "ffn_up.weight", &l.ffn_up) ||
        !up_q8(p + "ffn_down.weight", &l.ffn_down)) {
      return nullptr;
    }
    if (im.reader->FindTensor(p + "attn_v_ctx_norm.weight") != nullptr &&
        !up_f32(p + "attn_v_ctx_norm.weight", &l.attn_v_ctx_norm)) {
      return nullptr;
    }
  }
  if (const auto* w1t = im.reader->FindTensor("markov_w1.weight")) {
    im.has_markov = true;
    im.markov_rank = static_cast<std::uint32_t>(w1t->dimensions[0]);
    im.markov_vocab = static_cast<std::uint32_t>(w1t->dimensions[1]);
    if (!up_f32("markov_ctx_mix.weight", &im.markov_ctx_mix) ||
        !up_f32("markov_ctx_norm.weight", &im.markov_ctx_norm) ||
        !up_bf16("joint_proj.weight", &im.joint_proj_w) ||
        !up_f32("joint_proj.bias", &im.joint_proj_b) ||
        !up_bf16("markov_w1.weight", &im.markov_w1) ||
        !up_bf16("markov_w2.weight", &im.markov_w2) ||
        !up_f32("markov_w2.bias", &im.markov_w2_b)) {
      return nullptr;
    }
  }
  return m;
}

struct DsparkSession::Impl {
  const DsparkModel* model{nullptr};
  hipStream_t stream{nullptr};
  std::uint32_t max_context{0};
  std::uint32_t vocab_rows{0};
  std::vector<void*> allocs;

  std::array<__half*, DsparkModel::kNumLayers> k_ctx_cache{};
  std::array<__half*, DsparkModel::kNumLayers> v_ctx_cache{};
  __half* k_self{nullptr};
  __half* v_self{nullptr};

  float* d_ctx_buf{nullptr};  // [max_ctx, hidden]
  float* d_x{nullptr};        // [8, hidden]
  float* d_xn{nullptr};       // [8, hidden]
  float* d_q{nullptr};        // [8, q_dim]
  float* d_k{nullptr};        // [8, kv_dim]
  float* d_v{nullptr};        // [8, kv_dim]
  float* d_attn{nullptr};     // [8, q_dim]
  float* d_gate{nullptr};     // [8, ff]
  float* d_up{nullptr};       // [8, ff]
  float* d_proj{nullptr};     // [8, hidden]
  float* d_logits{nullptr};   // [8, vocab_rows]
  std::int32_t* d_tok{nullptr};
  std::int32_t* d_arg{nullptr};
  void* d_arg_scratch{nullptr};
  void* d_preq{nullptr};
  __half* d_x16{nullptr};
  __half* d_x16_ctx{nullptr};
  float* d_k_ctx_tmp{nullptr};
  float* d_v_ctx_tmp{nullptr};
  __half* d_wdq{nullptr};
  float* d_markov_ctx{nullptr};
  float* d_mk_state{nullptr};
  float* d_mk_z{nullptr};
  float* d_mk_proj{nullptr};
  float* d_mk_out{nullptr};

  ~Impl() {
    for (void* p : allocs) {
      if (p != nullptr) (void)hipFree(p);
    }
  }
};

DsparkSession::~DsparkSession() = default;

std::unique_ptr<DsparkSession> DsparkSession::Create(
    const DsparkModel& model, std::uint32_t max_context,
    std::uint32_t vocab_rows, hipStream_t stream, std::string* error_msg) {
  auto s = std::unique_ptr<DsparkSession>(new DsparkSession());
  s->impl_ = std::make_unique<Impl>();
  auto& im = *s->impl_;
  im.model = &model;
  im.stream = stream;
  im.max_context = max_context;
  im.vocab_rows = vocab_rows;
  const auto& mi = model.impl();
  const std::uint32_t q_dim = mi.q_heads * mi.head_dim;
  const std::uint32_t kv_dim = mi.kv_heads * mi.head_dim;

  auto alloc = [&](auto** ptr, std::size_t bytes, const char* what) -> bool {
    void* dev = nullptr;
    if (!CheckHip(hipMalloc(&dev, bytes), error_msg, what)) return false;
    im.allocs.push_back(dev);
    *ptr = reinterpret_cast<std::remove_reference_t<decltype(*ptr)>>(dev);
    return true;
  };

  for (std::uint32_t i = 0; i < DsparkModel::kNumLayers; ++i) {
    if (!alloc(&im.k_ctx_cache[i],
               static_cast<std::size_t>(max_context) * kv_dim * sizeof(__half),
               "k_ctx") ||
        !alloc(&im.v_ctx_cache[i],
               static_cast<std::size_t>(max_context) * kv_dim * sizeof(__half),
               "v_ctx")) {
      return nullptr;
    }
  }
  if (!alloc(&im.k_self, 8ULL * kv_dim * sizeof(__half), "k_self") ||
      !alloc(&im.v_self, 8ULL * kv_dim * sizeof(__half), "v_self") ||
      !alloc(&im.d_ctx_buf, static_cast<std::size_t>(max_context) * mi.hidden * sizeof(float), "ctx_buf") ||
      !alloc(&im.d_x, 8ULL * mi.hidden * sizeof(float), "x") ||
      !alloc(&im.d_xn, 8ULL * mi.hidden * sizeof(float), "xn") ||
      !alloc(&im.d_q, 8ULL * q_dim * sizeof(float), "q") ||
      !alloc(&im.d_k, 8ULL * kv_dim * sizeof(float), "k") ||
      !alloc(&im.d_v, 8ULL * kv_dim * sizeof(float), "v") ||
      !alloc(&im.d_attn, 8ULL * q_dim * sizeof(float), "attn") ||
      !alloc(&im.d_gate, 8ULL * mi.ff * sizeof(float), "gate") ||
      !alloc(&im.d_up, 8ULL * mi.ff * sizeof(float), "up") ||
      !alloc(&im.d_proj, 8ULL * mi.hidden * sizeof(float), "proj") ||
      !alloc(&im.d_logits, 8ULL * vocab_rows * sizeof(float), "logits") ||
      !alloc(&im.d_tok, 8ULL * sizeof(std::int32_t), "tok") ||
      !alloc(&im.d_arg, 8ULL * sizeof(std::int32_t), "arg") ||
      !alloc(&im.d_arg_scratch,
             8ULL * 64ULL * sizeof(core::compute::gemm::ArgmaxCandidate), "arg_s") ||
      !alloc(&im.d_preq, 256ULL * 1024ULL, "preq") ||
      !alloc(&im.d_x16, 8ULL * mi.ff * sizeof(__half), "x16") ||
      !alloc(&im.d_x16_ctx, 512ULL * mi.hidden * sizeof(__half), "x16_ctx") ||
      !alloc(&im.d_k_ctx_tmp, 512ULL * kv_dim * sizeof(float), "k_ctx_tmp") ||
      !alloc(&im.d_v_ctx_tmp, 512ULL * kv_dim * sizeof(float), "v_ctx_tmp") ||
      !alloc(&im.d_wdq, static_cast<std::size_t>(mi.ff) * mi.hidden * sizeof(__half), "wdq")) {
    return nullptr;
  }
  if (mi.has_markov) {
    if (!alloc(&im.d_markov_ctx,
               static_cast<std::size_t>(max_context) * mi.hidden * sizeof(float),
               "mk_ctx") ||
        !alloc(&im.d_mk_state, mi.markov_rank * sizeof(float), "mk_state") ||
        !alloc(&im.d_mk_z,
               (2ULL * mi.markov_rank + 2ULL * mi.hidden) * sizeof(float),
               "mk_z") ||
        !alloc(&im.d_mk_proj, 3ULL * mi.markov_rank * sizeof(float), "mk_proj") ||
        !alloc(&im.d_mk_out, mi.markov_rank * sizeof(float), "mk_out")) {
      return nullptr;
    }
  }
  return s;
}

namespace {
void DsparkGemm(const void* w, const float* x, __half* x16, __half* wdq,
                std::size_t wdq_bytes, void* preq, std::size_t preq_bytes,
                float* y, std::uint32_t n, std::uint32_t m, std::uint32_t k,
                hipStream_t stream) {
  namespace gemm = core::compute::gemm;
  if (n == 1 &&
      gemm::GemvDecode(w, static_cast<int>(core::GgmlType::kQ8_0), x, y, n, m,
                       k, preq, preq_bytes, stream) ==
          ::mbufo::core::quant::Status::kOk) {
    return;
  }
  gemm::ConvertF32ToF16(x, x16, static_cast<std::size_t>(n) * k, stream);
  (void)gemm::QuantGemmPrefill(w, static_cast<int>(core::GgmlType::kQ8_0), x16,
                               y, n, m, k, wdq, wdq_bytes, stream);
}
}  // namespace

bool DsparkSession::CaptureContext(
    const std::array<const float*, DsparkModel::kNumTargetLayers>& target_h,
    std::uint32_t start_pos, std::uint32_t n_tokens, std::string* error_msg) {
  if (n_tokens == 0) return true;
  auto& im = *impl_;
  const auto& mi = im.model->impl();
  const std::uint32_t kv_dim = mi.kv_heads * mi.head_dim;
  const std::size_t elems = static_cast<std::size_t>(n_tokens) * mi.hidden;
  const std::size_t wdq_bytes = static_cast<std::size_t>(mi.ff) * mi.hidden * 2ULL;

  for (std::uint32_t b = 0; b < DsparkModel::kNumLayers; ++b) {
    const auto& l = mi.layers[b];
    hipLaunchKernelGGL(Mix4CtxKernel, dim3((elems + 255ULL) / 256ULL),
                       dim3(256), 0, im.stream, target_h[0], target_h[1],
                       target_h[2], target_h[3], l.ctx_mix, im.d_ctx_buf,
                       elems);
    kernels::RmsNormGamma(im.d_ctx_buf, l.ctx_norm, im.d_ctx_buf, n_tokens,
                          mi.hidden, mi.rms_eps, im.stream);
    // Process in chunks of 512 -> k_ctx, v_ctx (avoids 256x per-8-token Q8_0 dequant!)
    for (std::uint32_t off = 0; off < n_tokens; off += 512U) {
      const std::uint32_t cnt = std::min(512U, n_tokens - off);
      DsparkGemm(l.attn_k_ctx, im.d_ctx_buf + static_cast<std::size_t>(off) * mi.hidden,
                 im.d_x16_ctx, im.d_wdq, wdq_bytes, im.d_preq, 256ULL * 1024ULL,
                 im.d_k_ctx_tmp, cnt, kv_dim, mi.hidden, im.stream);
      DsparkGemm(l.attn_v_ctx, im.d_ctx_buf + static_cast<std::size_t>(off) * mi.hidden,
                 im.d_x16_ctx, im.d_wdq, wdq_bytes, im.d_preq, 256ULL * 1024ULL,
                 im.d_v_ctx_tmp, cnt, kv_dim, mi.hidden, im.stream);
      hipLaunchKernelGGL(StoreCtxKvF16Kernel, dim3(cnt, mi.kv_heads),
                         dim3(mi.head_dim >> 1), 0, im.stream, im.d_k_ctx_tmp,
                 im.d_v_ctx_tmp, l.attn_k_ctx_norm, l.attn_v_ctx_norm,
                         im.k_ctx_cache[b], im.v_ctx_cache[b], start_pos + off,
                         mi.kv_heads, mi.head_dim, mi.rms_eps, mi.rope_theta);
    }
  }
  if (mi.has_markov) {
    float* mk_dst =
        im.d_markov_ctx + static_cast<std::size_t>(start_pos) * mi.hidden;
    hipLaunchKernelGGL(Mix4CtxKernel, dim3((elems + 255ULL) / 256ULL),
                       dim3(256), 0, im.stream, target_h[0], target_h[1],
                       target_h[2], target_h[3], mi.markov_ctx_mix, mk_dst,
                       elems);
    kernels::RmsNormGamma(mk_dst, mi.markov_ctx_norm, mk_dst, n_tokens,
                          mi.hidden, mi.rms_eps, im.stream);
  }
  (void)error_msg;
  return true;
}

bool DsparkSession::DraftBlock(
    std::int32_t anchor_token, std::uint32_t anchor_pos,
    std::uint32_t n_draft, const TensorRef& target_token_embd,
    const TensorRef& target_output, std::vector<std::int32_t>* out_drafts,
    std::string* error_msg) {
  if (n_draft == 0 || n_draft > DsparkModel::kMaxDraft) {
    if (error_msg != nullptr) *error_msg = "dspark: invalid n_draft";
    return false;
  }
  auto& im = *impl_;
  const auto& mi = im.model->impl();
  const std::uint32_t n_blk = 1U + n_draft;
  const std::uint32_t q_dim = mi.q_heads * mi.head_dim;
  const std::uint32_t kv_dim = mi.kv_heads * mi.head_dim;
  const std::size_t wdq_bytes = static_cast<std::size_t>(mi.ff) * mi.hidden * 2ULL;

  if (!CheckHip(hipMemcpyAsync(im.d_tok, &anchor_token, sizeof(std::int32_t),
                               hipMemcpyHostToDevice, im.stream),
                error_msg, "copy anchor")) {
    return false;
  }
  if (!core::compute::gemm::EmbedGather(
          target_token_embd.device, static_cast<int>(target_token_embd.type),
          im.d_tok, im.d_x, 1, mi.hidden, im.stream)) {
    if (error_msg != nullptr) *error_msg = "dspark: EmbedGather failed";
    return false;
  }
  kernels::RmsNormGamma(im.d_x, mi.token_embd_norm, im.d_x, 1, mi.hidden,
                        mi.rms_eps, im.stream);
  hipLaunchKernelGGL(CopyMaskEmbdKernel,
                     dim3(n_draft, (mi.hidden + 255U) / 256U), dim3(256), 0,
                     im.stream, mi.mask_embd, im.d_x, n_draft, mi.hidden);

  const float attn_scale = 1.0F / std::sqrt(static_cast<float>(mi.head_dim));
  for (std::uint32_t b = 0; b < DsparkModel::kNumLayers; ++b) {
    const auto& l = mi.layers[b];
    kernels::RmsNormGamma(im.d_x, l.attn_norm, im.d_xn, n_blk, mi.hidden,
                          mi.rms_eps, im.stream);
    DsparkGemm(l.attn_q, im.d_xn, im.d_x16, im.d_wdq, wdq_bytes, im.d_preq,
               256ULL * 1024ULL, im.d_q, n_blk, q_dim, mi.hidden, im.stream);
    DsparkGemm(l.attn_k, im.d_xn, im.d_x16, im.d_wdq, wdq_bytes, im.d_preq,
               256ULL * 1024ULL, im.d_k, n_blk, kv_dim, mi.hidden, im.stream);
    DsparkGemm(l.attn_v, im.d_xn, im.d_x16, im.d_wdq, wdq_bytes, im.d_preq,
               256ULL * 1024ULL, im.d_v, n_blk, kv_dim, mi.hidden, im.stream);
    hipLaunchKernelGGL(HeadRmsNormRopeF16StoreKernel,
                       dim3(n_blk, mi.q_heads + mi.kv_heads),
                       dim3(mi.head_dim >> 1), 0, im.stream, im.d_q, im.d_k,
                       im.d_v, l.attn_q_norm, l.attn_k_norm, im.k_self,
                       im.v_self, im.d_q, anchor_pos, mi.q_heads, mi.kv_heads,
                       mi.head_dim, mi.rms_eps, mi.rope_theta);
    // Note: HeadRmsNormRopeF16StoreKernel wrote k_self/v_self at offset `anchor_pos`!
    // Pass k_self/v_self shifted by `-anchor_pos`? Easier: pass `0` as start_pos for k_self/v_self?
    // Wait: RoPE needs `anchor_pos + tok`, while k_self/v_self index is `tok`!
    // Wait: in HeadRmsNormRopeF16StoreKernel, kv_off used `pos = start_pos + tok`.
    // For k_self/v_self (sized 8), `kv_off` MUST be `(tok * kv_heads + h) * head_dim`!
    // Let's fix that by passing `k_self - anchor_pos * kv_dim`? No, separate param or `0`!
    // (We'll fix HeadRmsNormRopeF16StoreKernel to use `tok` for k_cache/v_cache below.)
    hipLaunchKernelGGL(DsparkBlockAttnKernel, dim3(n_blk, mi.q_heads), dim3(32),
                       0, im.stream, im.d_q, im.k_ctx_cache[b],
                       im.v_ctx_cache[b], im.k_self, im.v_self, im.d_attn,
                       anchor_pos, n_blk, l.sliding ? mi.window : 0U,
                       mi.q_heads, mi.kv_heads, attn_scale);
    DsparkGemm(l.attn_output, im.d_attn, im.d_x16, im.d_wdq, wdq_bytes,
               im.d_preq, 256ULL * 1024ULL, im.d_proj, n_blk, mi.hidden, q_dim,
               im.stream);
    hipLaunchKernelGGL(AddResidualKernel,
                       dim3((n_blk * mi.hidden + 255U) / 256U), dim3(256), 0,
                       im.stream, im.d_x, im.d_proj,
                       static_cast<std::size_t>(n_blk) * mi.hidden);

    kernels::RmsNormGamma(im.d_x, l.ffn_norm, im.d_xn, n_blk, mi.hidden,
                          mi.rms_eps, im.stream);
    DsparkGemm(l.ffn_gate, im.d_xn, im.d_x16, im.d_wdq, wdq_bytes, im.d_preq,
               256ULL * 1024ULL, im.d_gate, n_blk, mi.ff, mi.hidden, im.stream);
    DsparkGemm(l.ffn_up, im.d_xn, im.d_x16, im.d_wdq, wdq_bytes, im.d_preq,
               256ULL * 1024ULL, im.d_up, n_blk, mi.ff, mi.hidden, im.stream);
    hipLaunchKernelGGL(SwiGluMulKernel, dim3((n_blk * mi.ff + 255U) / 256U),
                       dim3(256), 0, im.stream, im.d_gate, im.d_up, im.d_gate,
                       static_cast<std::size_t>(n_blk) * mi.ff);
    DsparkGemm(l.ffn_down, im.d_gate, im.d_x16, im.d_wdq, wdq_bytes, im.d_preq,
               256ULL * 1024ULL, im.d_proj, n_blk, mi.hidden, mi.ff, im.stream);
    hipLaunchKernelGGL(AddResidualKernel,
                       dim3((n_blk * mi.hidden + 255U) / 256U), dim3(256), 0,
                       im.stream, im.d_x, im.d_proj,
                       static_cast<std::size_t>(n_blk) * mi.hidden);
  }

  kernels::RmsNormGamma(im.d_x, mi.output_norm, im.d_xn, n_blk, mi.hidden,
                        mi.rms_eps, im.stream);
  // Project positions 1..n_draft via target lm_head
  const void* out_dev =
      target_output.device != nullptr ? target_output.device : target_output.data;
  core::compute::gemm::ConvertF32ToF16(
      im.d_xn + mi.hidden, im.d_x16,
      static_cast<std::size_t>(n_draft) * mi.hidden, im.stream);
  if (core::compute::gemm::QuantGemmPrefill(
          out_dev, static_cast<int>(target_output.type), im.d_x16, im.d_logits,
          n_draft, im.vocab_rows, mi.hidden, im.d_wdq, wdq_bytes, im.stream) !=
      ::mbufo::core::quant::Status::kOk) {
    if (error_msg != nullptr) *error_msg = "dspark: lm_head QuantGemmPrefill failed";
    return false;
  }
  if (mi.has_markov) {
    if (!CheckHip(hipMemsetAsync(im.d_mk_state, 0,
                                 mi.markov_rank * sizeof(float), im.stream),
                  error_msg, "zero mk_state")) {
      return false;
    }
    const float* mk_ctx =
        im.d_markov_ctx +
        static_cast<std::size_t>(anchor_pos > 0 ? anchor_pos - 1U : 0U) *
            mi.hidden;
    const std::uint32_t z_dim = 2U * mi.markov_rank + 2U * mi.hidden;
    const std::uint32_t proj_dim = 3U * mi.markov_rank;
    auto* scratch =
        static_cast<core::compute::gemm::ArgmaxCandidate*>(im.d_arg_scratch);
    for (std::uint32_t k = 0; k < n_draft; ++k) {
      const std::int32_t* d_prev = (k == 0) ? im.d_tok : (im.d_arg + k - 1U);
      const float* h_k = im.d_xn + static_cast<std::size_t>(1U + k) * mi.hidden;
      float* lg_k = im.d_logits + static_cast<std::size_t>(k) * im.vocab_rows;
      hipLaunchKernelGGL(MarkovBuildZKernel, dim3((z_dim + 255U) / 256U),
                         dim3(256), 0, im.stream, im.d_mk_state, mi.markov_w1,
                         d_prev, h_k, mk_ctx, im.d_mk_z, mi.markov_rank,
                         mi.hidden, mi.markov_vocab);
      hipLaunchKernelGGL(JointProjBf16Kernel, dim3(proj_dim), dim3(128), 0,
                         im.stream, mi.joint_proj_w, mi.joint_proj_b, im.d_mk_z,
                         im.d_mk_proj, proj_dim, z_dim);
      hipLaunchKernelGGL(GruStepKernel, dim3((mi.markov_rank + 255U) / 256U),
                         dim3(256), 0, im.stream, im.d_mk_proj, im.d_mk_state,
                         im.d_mk_out, mi.markov_rank);
      hipLaunchKernelGGL(MarkovW2AddKernel,
                         dim3((mi.markov_vocab + 31U) / 32U), dim3(256), 0,
                         im.stream, mi.markov_w2, mi.markov_w2_b, im.d_mk_out,
                         lg_k, mi.markov_vocab, mi.markov_rank);
      core::compute::gemm::SoftCap(lg_k, mi.softcap, mi.markov_vocab, im.stream);
      core::compute::gemm::Argmax(lg_k, scratch, im.d_arg + k, 1,
                                  mi.markov_vocab, im.stream);
    }
  } else {
    core::compute::gemm::Argmax(
        im.d_logits, static_cast<core::compute::gemm::ArgmaxCandidate*>(im.d_arg_scratch),
        im.d_arg, n_draft, im.vocab_rows, im.stream);
  }
  out_drafts->resize(n_draft);
  if (!CheckHip(hipMemcpyAsync(out_drafts->data(), im.d_arg,
                               n_draft * sizeof(std::int32_t),
                               hipMemcpyDeviceToHost, im.stream),
                error_msg, "copy drafts") ||
      !CheckHip(hipStreamSynchronize(im.stream), error_msg, "sync drafts")) {
    return false;
  }
  return true;
}

}  // namespace mbufo::models::inkling
