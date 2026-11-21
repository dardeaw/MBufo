#include "src/models/laguna/kernels/kernels.hpp"

#include <cmath>

namespace mbufo::models::laguna::kernels {
namespace {

__global__ void ConvertBf16ToF16Kernel(const std::uint16_t* in, __half* out,
                                       std::size_t count) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  const std::uint32_t bits = static_cast<std::uint32_t>(in[idx]) << 16U;
  float f = 0.0F;
  memcpy(&f, &bits, sizeof(f));
  out[idx] = __float2half(f);
}

__global__ void RmsNormKernel(const float* x, const float* w, float* out,
                              std::uint32_t hidden, float eps) {
  const std::uint32_t tok = blockIdx.x;
  const float* row = x + static_cast<std::size_t>(tok) * hidden;
  float* dst = out + static_cast<std::size_t>(tok) * hidden;

  __shared__ float s_sum[8];
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = row[i];
    sum += v * v;
  }
  for (int off = 16; off > 0; off /= 2) {
    sum += __shfl_down(sum, off);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) s_sum[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float total = (lane < static_cast<int>(blockDim.x >> 5)) ? s_sum[lane] : 0.0F;
    for (int off = 16; off > 0; off /= 2) {
      total += __shfl_down(total, off);
    }
    if (lane == 0) s_sum[0] = rsqrtf(total / static_cast<float>(hidden) + eps);
  }
  __syncthreads();
  const float inv = s_sum[0];
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst[i] = row[i] * inv * w[i];
  }
}

__global__ void HeadRmsNormKernel(float* x, const float* w,
                                  std::uint32_t n_heads,
                                  std::uint32_t head_dim, float eps) {
  const std::uint32_t row_idx = blockIdx.x;
  float* row = x + static_cast<std::size_t>(row_idx) * head_dim;
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    const float v = row[i];
    sum += v * v;
  }
  for (int off = 16; off > 0; off /= 2) {
    sum += __shfl_down(sum, off);
  }
  __shared__ float s_inv;
  if (threadIdx.x == 0) {
    s_inv = rsqrtf(sum / static_cast<float>(head_dim) + eps);
  }
  __syncthreads();
  const float inv = s_inv;
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    row[i] = row[i] * inv * w[i];
  }
  (void)n_heads;
}

__device__ __forceinline__ float YarnFindCorrectionDim(float num_rotations,
                                                       std::uint32_t dim,
                                                       float base,
                                                       std::uint32_t max_pos) {
  constexpr float kTwoPi = 6.28318530717958647692F;
  return (static_cast<float>(dim) *
          logf(static_cast<float>(max_pos) / (num_rotations * kTwoPi))) /
         (2.0F * logf(base));
}

__global__ void ApplyDualRopeKernel(
    float* q, float* k, std::uint32_t start_pos, std::uint32_t n_q_heads,
    std::uint32_t n_kv_heads, std::uint32_t head_dim, std::uint32_t rotary_dim,
    float rope_theta, bool is_swa, float yarn_factor,
    std::uint32_t yarn_orig_ctx, float yarn_beta_fast, float yarn_beta_slow,
    float yarn_attn_factor) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  const std::uint32_t half_rot = rotary_dim / 2;
  if (d >= half_rot) return;

  const float pos = static_cast<float>(start_pos + tok);
  const float exp_val = static_cast<float>(2U * d) / static_cast<float>(rotary_dim);
  const float inv_freq_extrap = 1.0F / powf(rope_theta, exp_val);
  float inv_freq = inv_freq_extrap;
  float attn_scale = 1.0F;

  if (!is_swa && yarn_factor > 1.0F) {
    const float low = fmaxf(
        floorf(YarnFindCorrectionDim(yarn_beta_fast, rotary_dim, rope_theta,
                                     yarn_orig_ctx)),
        0.0F);
    const float high = fminf(
        ceilf(YarnFindCorrectionDim(yarn_beta_slow, rotary_dim, rope_theta,
                                    yarn_orig_ctx)),
        static_cast<float>(rotary_dim - 1));
    const float denom = (fabsf(high - low) < 1e-3F) ? 1e-3F : (high - low);
    const float ramp = fminf(fmaxf((static_cast<float>(d) - low) / denom, 0.0F), 1.0F);
    const float inv_freq_interp = inv_freq_extrap / yarn_factor;
    inv_freq = inv_freq_interp * ramp + inv_freq_extrap * (1.0F - ramp);
    attn_scale = yarn_attn_factor;
  }

  const float angle = pos * inv_freq;
  const float c = cosf(angle) * attn_scale;
  const float s = sinf(angle) * attn_scale;

  if (head < n_q_heads) {
    float* qh = q + (static_cast<std::size_t>(tok) * n_q_heads + head) * head_dim;
    const float x0 = qh[d];
    const float x1 = qh[d + half_rot];
    qh[d] = x0 * c - x1 * s;
    qh[d + half_rot] = x1 * c + x0 * s;
  }
  if (head < n_kv_heads) {
    float* kh = k + (static_cast<std::size_t>(tok) * n_kv_heads + head) * head_dim;
    const float x0 = kh[d];
    const float x1 = kh[d + half_rot];
    kh[d] = x0 * c - x1 * s;
    kh[d + half_rot] = x1 * c + x0 * s;
  }
}

__global__ void SoftplusPerHeadGateMulKernel(float* attn_out,
                                             const float* gate_logits,
                                             std::uint32_t n_heads,
                                             std::uint32_t head_dim) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  if (d >= head_dim) return;
  const float g = gate_logits[static_cast<std::size_t>(tok) * n_heads + head];
  // Numerically stable softplus(g) = log(1 + exp(g))
  const float sp = (g > 20.0F) ? g : log1pf(expf(g));
  const std::size_t idx =
      (static_cast<std::size_t>(tok) * n_heads + head) * head_dim + d;
  attn_out[idx] *= sp;
}

__global__ void SigmoidBiasTopKRouterKernel(
    const float* logits, const float* bias, std::int32_t* ids, float* weights,
    std::uint32_t n_experts, std::uint32_t top_k, bool norm_topk, float scale) {
  const std::uint32_t tok = blockIdx.x;
  if (threadIdx.x != 0) return;

  const float* row = logits + static_cast<std::size_t>(tok) * n_experts;
  std::int32_t* out_ids = ids + static_cast<std::size_t>(tok) * top_k;
  float* out_w = weights + static_cast<std::size_t>(tok) * top_k;

  constexpr int kMaxK = 16;
  float best_sel[kMaxK];
  float best_prob[kMaxK];
  int best_idx[kMaxK];
  for (std::uint32_t k = 0; k < top_k; ++k) {
    best_sel[k] = -1e30F;
    best_prob[k] = 0.0F;
    best_idx[k] = -1;
  }

  for (std::uint32_t e = 0; e < n_experts; ++e) {
    const float p = 1.0F / (1.0F + expf(-row[e]));
    const float b = (bias != nullptr) ? bias[e] : 0.0F;
    const float sel = p + b;
    if (sel > best_sel[top_k - 1]) {
      best_sel[top_k - 1] = sel;
      best_prob[top_k - 1] = p;
      best_idx[top_k - 1] = static_cast<int>(e);
      for (int j = static_cast<int>(top_k) - 1; j > 0; --j) {
        if (best_sel[j] > best_sel[j - 1]) {
          float ts = best_sel[j - 1]; best_sel[j - 1] = best_sel[j]; best_sel[j] = ts;
          float tp = best_prob[j - 1]; best_prob[j - 1] = best_prob[j]; best_prob[j] = tp;
          int ti = best_idx[j - 1]; best_idx[j - 1] = best_idx[j]; best_idx[j] = ti;
        } else {
          break;
        }
      }
    }
  }

  float sum = 0.0F;
  for (std::uint32_t k = 0; k < top_k; ++k) sum += best_prob[k];
  const float inv = (norm_topk && sum > 0.0F) ? (1.0F / sum) : 1.0F;
  for (std::uint32_t k = 0; k < top_k; ++k) {
    out_ids[k] = best_idx[k];
    out_w[k] = best_prob[k] * inv * scale;
  }
}

__global__ void SiluMulKernel(const float* gate, const float* up, float* out,
                              std::size_t count) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  const float g = gate[idx];
  const float s = g / (1.0F + expf(-g));
  out[idx] = s * up[idx];
}

__global__ void AddResidualKernel(float* a, const float* b, std::size_t count) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  a[idx] += b[idx];
}

__global__ void StoreKvF16Kernel(const float* k_in, const float* v_in,
                                 __half* k_cache, __half* v_cache,
                                 std::uint32_t start_pos,
                                 std::uint32_t n_kv_heads,
                                 std::uint32_t head_dim) {
  const std::uint32_t tok = blockIdx.x;
  const std::size_t kv_stride = static_cast<std::size_t>(n_kv_heads) * head_dim;
  const std::size_t src_off = static_cast<std::size_t>(tok) * kv_stride;
  const std::size_t dst_off =
      static_cast<std::size_t>(start_pos + tok) * kv_stride;
  for (std::size_t i = threadIdx.x; i < kv_stride; i += blockDim.x) {
    k_cache[dst_off + i] = __float2half(k_in[src_off + i]);
    v_cache[dst_off + i] = __float2half(v_in[src_off + i]);
  }
}

__global__ void F16AttentionKernel(const float* q, const __half* k_cache,
                                   const __half* v_cache, float* out,
                                   std::uint32_t n_tokens,
                                   std::uint32_t start_pos,
                                   std::uint32_t sliding_window,
                                   std::uint32_t n_q_heads,
                                   std::uint32_t n_kv_heads,
                                   std::uint32_t head_dim, float scale) {
  const std::uint32_t tok = blockIdx.x;
  const std::uint32_t qh = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  if (d >= head_dim) return;

  const std::uint32_t kv_group = n_q_heads / n_kv_heads;
  const std::uint32_t kvh = qh / kv_group;
  const std::uint32_t cur_pos = start_pos + tok;
  const std::uint32_t min_pos =
      (sliding_window > 0 && cur_pos >= sliding_window)
          ? (cur_pos - sliding_window + 1)
          : 0U;

  const float* q_vec =
      q + (static_cast<std::size_t>(tok) * n_q_heads + qh) * head_dim;
  const float q_val = q_vec[d];
  const std::size_t kv_stride = static_cast<std::size_t>(n_kv_heads) * head_dim;

  // Online softmax over [min_pos, cur_pos]
  float m = -1e30F;
  float l = 0.0F;
  float acc = 0.0F;

  for (std::uint32_t p = min_pos; p <= cur_pos; ++p) {
    const std::size_t kv_off =
        static_cast<std::size_t>(p) * kv_stride +
        static_cast<std::size_t>(kvh) * head_dim;
    float dot = q_val * __half2float(k_cache[kv_off + d]);
    // Warp/block reduction across head_dim (128 threads = 4 waves of 32)
    __shared__ float s_dot[4];
    for (int off = 16; off > 0; off /= 2) dot += __shfl_down(dot, off);
    const int lane = d & 31;
    const int warp = d >> 5;
    if (lane == 0) s_dot[warp] = dot;
    __syncthreads();
    if (warp == 0) {
      float t = (lane < 4) ? s_dot[lane] : 0.0F;
      for (int off = 16; off > 0; off /= 2) t += __shfl_down(t, off);
      if (lane == 0) s_dot[0] = t * scale;
    }
    __syncthreads();
    const float score = s_dot[0];
    __syncthreads();
    const float m_new = fmaxf(m, score);
    const float alpha = expf(m - m_new);
    const float w = expf(score - m_new);
    acc = acc * alpha + w * __half2float(v_cache[kv_off + d]);
    l = l * alpha + w;
    m = m_new;
  }

  out[(static_cast<std::size_t>(tok) * n_q_heads + qh) * head_dim + d] =
      (l > 0.0F) ? (acc / l) : 0.0F;
  (void)n_tokens;
}

__global__ void ArgmaxKernel(const float* logits, std::uint32_t vocab_size,
                             std::int32_t* d_out) {
  __shared__ float s_val[8];
  __shared__ std::int32_t s_idx[8];
  float best_v = -1e30F;
  std::int32_t best_i = 0;
  for (std::uint32_t i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    const float v = logits[i];
    if (v > best_v) {
      best_v = v;
      best_i = static_cast<std::int32_t>(i);
    }
  }
  for (int off = 16; off > 0; off /= 2) {
    const float ov = __shfl_down(best_v, off);
    const std::int32_t oi = __shfl_down(best_i, off);
    if (ov > best_v) {
      best_v = ov;
      best_i = oi;
    }
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    s_val[warp] = best_v;
    s_idx[warp] = best_i;
  }
  __syncthreads();
  if (warp == 0) {
    float bv = (lane < 8) ? s_val[lane] : -1e30F;
    std::int32_t bi = (lane < 8) ? s_idx[lane] : 0;
    for (int off = 4; off > 0; off /= 2) {
      const float ov = __shfl_down(bv, off);
      const std::int32_t oi = __shfl_down(bi, off);
      if (ov > bv) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0) *d_out = bi;
  }
}

}  // namespace

void ConvertBf16ToF16(const std::uint16_t* in, __half* out, std::size_t count,
                      hipStream_t stream) {
  if (count == 0) return;
  const std::size_t blocks = (count + 255) / 256;
  hipLaunchKernelGGL(ConvertBf16ToF16Kernel, dim3(blocks), dim3(256), 0,
                     stream, in, out, count);
}

void RmsNorm(const float* x, const float* weight, float* out,
             std::uint32_t n_tokens, std::uint32_t hidden, float eps,
             hipStream_t stream) {
  if (n_tokens == 0) return;
  hipLaunchKernelGGL(RmsNormKernel, dim3(n_tokens), dim3(256), 0, stream, x,
                     weight, out, hidden, eps);
}

void HeadRmsNorm(float* x, const float* weight, std::uint32_t n_tokens,
                 std::uint32_t n_heads, std::uint32_t head_dim, float eps,
                 hipStream_t stream) {
  if (n_tokens == 0) return;
  hipLaunchKernelGGL(HeadRmsNormKernel, dim3(n_tokens * n_heads), dim3(32), 0,
                     stream, x, weight, n_heads, head_dim, eps);
}

void ApplyDualRope(float* q, float* k, std::uint32_t n_tokens,
                   std::uint32_t start_pos, std::uint32_t n_q_heads,
                   std::uint32_t n_kv_heads, std::uint32_t head_dim,
                   std::uint32_t rotary_dim, float rope_theta, bool is_swa,
                   float yarn_factor, std::uint32_t yarn_orig_ctx,
                   float yarn_beta_fast, float yarn_beta_slow,
                   float yarn_attn_factor, hipStream_t stream) {
  if (n_tokens == 0) return;
  const std::uint32_t max_heads =
      n_q_heads > n_kv_heads ? n_q_heads : n_kv_heads;
  hipLaunchKernelGGL(ApplyDualRopeKernel, dim3(n_tokens, max_heads),
                     dim3(rotary_dim / 2), 0, stream, q, k, start_pos,
                     n_q_heads, n_kv_heads, head_dim, rotary_dim, rope_theta,
                     is_swa, yarn_factor, yarn_orig_ctx, yarn_beta_fast,
                     yarn_beta_slow, yarn_attn_factor);
}

void SoftplusPerHeadGateMul(float* attn_out, const float* gate_logits,
                            std::uint32_t n_tokens, std::uint32_t n_heads,
                            std::uint32_t head_dim, hipStream_t stream) {
  if (n_tokens == 0) return;
  hipLaunchKernelGGL(SoftplusPerHeadGateMulKernel, dim3(n_tokens, n_heads),
                     dim3(head_dim), 0, stream, attn_out, gate_logits,
                     n_heads, head_dim);
}

void SigmoidBiasTopKRouter(const float* logits, const float* bias,
                           std::int32_t* ids, float* weights,
                           std::uint32_t n_tokens, std::uint32_t n_experts,
                           std::uint32_t top_k, bool norm_topk, float scale,
                           hipStream_t stream) {
  if (n_tokens == 0) return;
  hipLaunchKernelGGL(SigmoidBiasTopKRouterKernel, dim3(n_tokens), dim3(1), 0,
                     stream, logits, bias, ids, weights, n_experts, top_k,
                     norm_topk, scale);
}

void SiluMul(const float* gate, const float* up, float* out,
             std::size_t count, hipStream_t stream) {
  if (count == 0) return;
  const std::size_t blocks = (count + 255) / 256;
  hipLaunchKernelGGL(SiluMulKernel, dim3(blocks), dim3(256), 0, stream, gate,
                     up, out, count);
}

void AddResidual(float* a, const float* b, std::size_t count,
                 hipStream_t stream) {
  if (count == 0) return;
  const std::size_t blocks = (count + 255) / 256;
  hipLaunchKernelGGL(AddResidualKernel, dim3(blocks), dim3(256), 0, stream, a,
                     b, count);
}

void StoreKvF16(const float* k_in, const float* v_in, __half* k_cache,
                __half* v_cache, std::uint32_t n_tokens,
                std::uint32_t start_pos, std::uint32_t n_kv_heads,
                std::uint32_t head_dim, hipStream_t stream) {
  if (n_tokens == 0) return;
  hipLaunchKernelGGL(StoreKvF16Kernel, dim3(n_tokens), dim3(256), 0, stream,
                     k_in, v_in, k_cache, v_cache, start_pos, n_kv_heads,
                     head_dim);
}

void F16AttentionDecode(const float* q, const __half* k_cache,
                        const __half* v_cache, float* out,
                        std::uint32_t seq_len, std::uint32_t sliding_window,
                        std::uint32_t n_q_heads, std::uint32_t n_kv_heads,
                        std::uint32_t head_dim, float scale,
                        hipStream_t stream) {
  hipLaunchKernelGGL(F16AttentionKernel, dim3(1, n_q_heads), dim3(head_dim), 0,
                     stream, q, k_cache, v_cache, out, 1, seq_len - 1,
                     sliding_window, n_q_heads, n_kv_heads, head_dim, scale);
}

void F16AttentionPrefill(const float* q, const __half* k_cache,
                         const __half* v_cache, float* out,
                         std::uint32_t n_tokens, std::uint32_t start_pos,
                         std::uint32_t sliding_window, std::uint32_t n_q_heads,
                         std::uint32_t n_kv_heads, std::uint32_t head_dim,
                         float scale, hipStream_t stream) {
  if (n_tokens == 0) return;
  hipLaunchKernelGGL(F16AttentionKernel, dim3(n_tokens, n_q_heads),
                     dim3(head_dim), 0, stream, q, k_cache, v_cache, out,
                     n_tokens, start_pos, sliding_window, n_q_heads,
                     n_kv_heads, head_dim, scale);
}

std::int32_t Argmax(const float* logits, std::uint32_t vocab_size,
                    std::int32_t* d_out, hipStream_t stream) {
  hipLaunchKernelGGL(ArgmaxKernel, dim3(1), dim3(256), 0, stream, logits,
                     vocab_size, d_out);
  std::int32_t host_out = 0;
  (void)hipMemcpyAsync(&host_out, d_out, sizeof(host_out),
                       hipMemcpyDeviceToHost, stream);
  (void)hipStreamSynchronize(stream);
  return host_out;
}


__global__ void ConvertQ6KToQ80Kernel(const std::uint8_t* __restrict__ src,
                                      std::uint8_t* __restrict__ dst) {
  const std::size_t b = blockIdx.x;
  const std::uint32_t i = threadIdx.x;  // 0..255
  const std::uint8_t* blk = src + b * 210ULL;

  const std::uint32_t half = i / 128U;
  const std::uint32_t within_half = i % 128U;
  const std::uint32_t segment = within_half / 32U;
  const std::uint32_t lane32 = within_half % 32U;
  const std::uint32_t ql_base = half * 64U;
  const std::uint8_t qh = blk[128U + half * 32U + lane32];

  std::uint8_t low = 0;
  std::uint8_t high = 0;
  if (segment == 0U) {
    low = blk[ql_base + lane32] & 0x0FU;
    high = qh & 0x03U;
  } else if (segment == 1U) {
    low = blk[ql_base + 32U + lane32] & 0x0FU;
    high = (qh >> 2U) & 0x03U;
  } else if (segment == 2U) {
    low = blk[ql_base + lane32] >> 4U;
    high = (qh >> 4U) & 0x03U;
  } else {
    low = blk[ql_base + 32U + lane32] >> 4U;
    high = (qh >> 6U) & 0x03U;
  }

  const std::uint32_t scale_index = (half * 8U) + (lane32 / 16U) + (segment * 2U);
  const std::int8_t sc = static_cast<std::int8_t>(blk[192U + scale_index]);
  const std::int32_t quant = static_cast<std::int32_t>((high << 4U) | low) - 32;
  const std::uint16_t d_bits =
      static_cast<std::uint16_t>(blk[208U]) |
      (static_cast<std::uint16_t>(blk[209U]) << 8U);
  const float d_block = __half2float(__ushort_as_half(d_bits));
  const float val = d_block * static_cast<float>(sc) * static_cast<float>(quant);

  float amax = fabsf(val);
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
  }
  const float d_out = amax / 127.0F;
  const float id_out = (d_out != 0.0F) ? (1.0F / d_out) : 0.0F;
  long q = lrintf(val * id_out);
  if (q < -127L) q = -127L;
  if (q > 127L) q = 127L;

  const std::size_t q80_blk_idx = b * 8ULL + (i / 32U);
  const std::uint32_t q80_lane = i % 32U;
  std::uint8_t* out_blk = dst + q80_blk_idx * 34ULL;
  if (q80_lane == 0U) {
    const std::uint16_t out_bits = __half_as_ushort(__float2half(d_out));
    out_blk[0] = static_cast<std::uint8_t>(out_bits & 0xFFU);
    out_blk[1] = static_cast<std::uint8_t>(out_bits >> 8U);
  }
  out_blk[2U + q80_lane] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
}

void ConvertQ6KToQ80(const void* q6k_dev, void* q80_dev, std::size_t n_elems,
                    hipStream_t stream) {
  const std::size_t n_blocks = n_elems / 256ULL;
  if (n_blocks == 0) return;
  hipLaunchKernelGGL(ConvertQ6KToQ80Kernel, dim3(static_cast<unsigned>(n_blocks)),
                     dim3(256), 0, stream,
                     static_cast<const std::uint8_t*>(q6k_dev),
                     static_cast<std::uint8_t*>(q80_dev));
}

__global__ void FillSlotTilesKernel(const std::int32_t* __restrict__ ids,
                                    std::int32_t* __restrict__ tiles,
                                    std::size_t slots) {
  const std::size_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < slots) {
    tiles[2 * s] = ids[s];
    tiles[2 * s + 1] = ids[s] | (1 << 16);
  }
}

void FillSlotTiles(const std::int32_t* ids, std::int32_t* tiles,
                   std::size_t slots, hipStream_t stream) {
  if (slots == 0) return;
  const unsigned blocks = static_cast<unsigned>((slots + 255) / 256);
  hipLaunchKernelGGL(FillSlotTilesKernel, dim3(blocks), dim3(256), 0, stream,
                     ids, tiles, slots);
}

__global__ void BuildRoutedTilesKernel(const std::uint32_t* __restrict__ counts,
                                       std::uint32_t n_experts,
                                       std::uint32_t tile_rows,
                                       std::int32_t* __restrict__ tiles,
                                       std::uint32_t* __restrict__ n_tiles) {
  const std::uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_experts) return;
  const std::uint32_t c = counts[e];
  const std::uint32_t chunks = (c + tile_rows - 1) / tile_rows;
  for (std::uint32_t j = 0; j < chunks; ++j) {
    const std::uint32_t idx = atomicAdd(n_tiles, 1u);
    tiles[idx] = static_cast<std::int32_t>(e | (j << 16));
  }
}

void BuildRoutedTiles(const std::uint32_t* counts, std::uint32_t n_experts,
                      std::uint32_t tile_rows, std::int32_t* tiles,
                      std::uint32_t* n_tiles, hipStream_t stream) {
  if (n_experts == 0) return;
  const unsigned blocks = static_cast<unsigned>((n_experts + 255) / 256);
  hipLaunchKernelGGL(BuildRoutedTilesKernel, dim3(blocks), dim3(256), 0, stream,
                     counts, n_experts, tile_rows, tiles, n_tiles);
}

}  // namespace mbufo::models::laguna::kernels
