// EmbeddingGemma 2 device forward.
//
// One bidirectional pass over the whole sequence:
//   embed gather (scaled) -> per layer {pre-norm, fused QKV GEMM, q/k/v
//   norm + RoPE, attention (full or +-window), o-proj, post-norm residual,
//   gated GELU MLP with post-norm residual, per-layer-input block, layer
//   scalar} -> final norm -> mean pool -> output projection -> L2.
//
// Weights are uploaded once as F16 (converted from the BF16 GGUF; every
// value is range-checked). The residual stream is F32; every GEMM input is
// a freshly normalized F16 activation with F32 accumulation and F32 output.
// Attention keeps the score tile materialized per 1024-row query chunk
// (strided-batched hipBLAS GEMMs for QK^T and PV, a masked softmax between
// them); a local layer only touches the key range inside its window.
// There is no per-shape autotune, so every length costs the same cold or warm.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/embeddinggemma2/engine.hpp"

namespace mbufo::models::embeddinggemma2 {
namespace {

constexpr int kThreads = 256;
constexpr int kQueryChunk = 1024;
// Sequences one batched forward accepts (also bounds the pooled-output scratch).
constexpr std::size_t kMaxSeqs = 64;

#define EG_HIP(call)                                                         \
  do {                                                                       \
    const hipError_t eg_err = (call);                                        \
    if (eg_err != hipSuccess) {                                              \
      if (error != nullptr)                                                  \
        *error = std::string("hip error: ") + hipGetErrorString(eg_err) +    \
                 " at " #call;                                               \
      return false;                                                          \
    }                                                                        \
  } while (0)

__device__ inline float BlockSum(float v, float* shared) {
  for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_down(v, o);
  const int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
  if (lane == 0) shared[wave] = v;
  __syncthreads();
  float r = 0.0F;
  for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; ++i) r += shared[i];
  __syncthreads();
  return r;
}

__device__ inline float BlockMax(float v, float* shared) {
  for (int o = warpSize / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
  const int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
  if (lane == 0) shared[wave] = v;
  __syncthreads();
  float r = -INFINITY;
  for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; ++i) r = fmaxf(r, shared[i]);
  __syncthreads();
  return r;
}

__device__ inline float GeluTanh(float x) {
  const float u = 0.7978845608028654F * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + tanhf(u));
}

// h[t] = table[id] * scale (F32 residual) and the F16 copy for the PLE GEMM.
__global__ void EmbedGatherKernel(const std::uint32_t* ids, const __half* table,
                                  float scale, int hidden, float* h,
                                  __half* emb16) {
  const int t = blockIdx.x;
  const __half* row = table + static_cast<std::size_t>(ids[t]) * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = __half2float(row[i]) * scale;
    h[static_cast<std::size_t>(t) * hidden + i] = v;
    emb16[static_cast<std::size_t>(t) * hidden + i] = __float2half(v);
  }
}

// out16[t] = rms(h[t]) * w  (F16 GEMM input).
__global__ void RmsNormF16Kernel(const float* h, const float* w, int hidden,
                                 float eps, __half* out) {
  __shared__ float sh[8];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * hidden;
  float ss = 0.0F;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = h[base + i];
    ss += v * v;
  }
  const float inv = rsqrtf(BlockSum(ss, sh) / hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x)
    out[base + i] = __float2half(h[base + i] * inv * w[i]);
}

// out[t] = rms(h[t]) * w  (F32, final norm).
__global__ void RmsNormF32Kernel(const float* h, const float* w, int hidden,
                                 float eps, float* out) {
  __shared__ float sh[8];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * hidden;
  float ss = 0.0F;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = h[base + i];
    ss += v * v;
  }
  const float inv = rsqrtf(BlockSum(ss, sh) / hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x)
    out[base + i] = h[base + i] * inv * w[i];
}

// h = (h + rms(y) * w) * post_scale
__global__ void RmsAddKernel(float* h, const float* y, const float* w,
                             int hidden, float eps, float post_scale) {
  __shared__ float sh[8];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * hidden;
  float ss = 0.0F;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = y[base + i];
    ss += v * v;
  }
  const float inv = rsqrtf(BlockSum(ss, sh) / hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x)
    h[base + i] = (h[base + i] + y[base + i] * inv * w[i]) * post_scale;
}

// PLE side input for one layer: pl = rms(raw * hidden^-0.5) * w, in place.
__global__ void PleNormKernel(float* raw, const float* w, int width,
                              float pre_scale, float eps) {
  __shared__ float sh[8];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * width;
  float ss = 0.0F;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float v = raw[base + i] * pre_scale;
    ss += v * v;
  }
  const float inv = rsqrtf(BlockSum(ss, sh) / width + eps);
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    raw[base + i] = raw[base + i] * pre_scale * inv * w[i];
}

__global__ void HToF16Kernel(const float* h, std::size_t count, __half* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) out[i] = __float2half(h[i]);
}

// out16 = gelu(g) * pl
__global__ void PleGateKernel(const float* g, const float* pl, std::size_t count,
                              __half* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) out[i] = __float2half(GeluTanh(g[i]) * pl[i]);
}

// out16[t][j] = gelu(gu[t][j]) * gu[t][F + j]
__global__ void GegluKernel(const float* gu, int ffn, std::size_t rows,
                            __half* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * ffn) return;
  const std::size_t t = i / ffn;
  const int j = static_cast<int>(i % ffn);
  const float* r = gu + t * 2 * ffn;
  out[i] = __float2half(GeluTanh(r[j]) * r[ffn + j]);
}

// Splits the fused QKV GEMM output; q/k get a weighted RMS norm + RoPE, v an
// unweighted RMS norm. One block per (token, head-slot).
__global__ void QkvPostKernel(const float* qkv, const float* qn, const float* kn,
                              int heads, int kvh, int d, float theta, float eps,
                              __half* q_out, __half* k_out, __half* v_out,
                              const std::uint32_t* pos) {
  __shared__ float sh[8];
  __shared__ float vals[512];
  const int t = blockIdx.x;
  const int slot = blockIdx.y;
  const int qd = heads * d, kd = kvh * d;
  const std::size_t row = static_cast<std::size_t>(t) * (qd + 2 * kd);
  int kind;  // 0 q, 1 k, 2 v
  int head;
  if (slot < heads) { kind = 0; head = slot; }
  else if (slot < heads + kvh) { kind = 1; head = slot - heads; }
  else { kind = 2; head = slot - heads - kvh; }
  const float* src = qkv + row + (kind == 0 ? 0 : (kind == 1 ? qd : qd + kd)) + head * d;
  float ss = 0.0F;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    const float v = src[i];
    vals[i] = v;
    ss += v * v;
  }
  const float inv = rsqrtf(BlockSum(ss, sh) / d + eps);
  const float* w = kind == 0 ? qn : kn;
  for (int i = threadIdx.x; i < d; i += blockDim.x)
    vals[i] = vals[i] * inv * (kind == 2 ? 1.0F : w[i]);
  __syncthreads();
  if (kind == 2) {
    __half* dst = v_out + static_cast<std::size_t>(t) * kd + head * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) dst[i] = __float2half(vals[i]);
    return;
  }
  __half* dst = (kind == 0 ? q_out + static_cast<std::size_t>(t) * qd
                           : k_out + static_cast<std::size_t>(t) * kd) + head * d;
  const int half = d / 2;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    const float ang = static_cast<float>(pos[t]) * powf(theta, -2.0F * i / d);
    float s, c;
    sincosf(ang, &s, &c);
    const float a = vals[i], b = vals[i + half];
    dst[i] = __float2half(a * c - b * s);
    dst[i + half] = __float2half(b * c + a * s);
  }
}

// Row softmax over one query chunk's scores. scores: [heads][rows][nk] F32 for
// keys k0..k0+nk-1; query row r is absolute position q0+r. window < 0: full.
__global__ void SoftmaxKernel(const float* s, int rows, int nk, int q0, int k0,
                              int window, __half* p) {
  __shared__ float sh[8];
  const int r = blockIdx.x, head = blockIdx.y;
  const std::size_t base = (static_cast<std::size_t>(head) * rows + r) * nk;
  const int qpos = q0 + r;
  float m = -INFINITY;
  for (int c = threadIdx.x; c < nk; c += blockDim.x) {
    if (window >= 0 && abs(qpos - (k0 + c)) > window) continue;
    m = fmaxf(m, s[base + c]);
  }
  m = BlockMax(m, sh);
  float sum = 0.0F;
  for (int c = threadIdx.x; c < nk; c += blockDim.x) {
    if (window >= 0 && abs(qpos - (k0 + c)) > window) continue;
    sum += __expf(s[base + c] - m);
  }
  const float inv = 1.0F / BlockSum(sum, sh);
  for (int c = threadIdx.x; c < nk; c += blockDim.x) {
    const bool ok = window < 0 || abs(qpos - (k0 + c)) <= window;
    p[base + c] = __float2half(ok ? __expf(s[base + c] - m) * inv : 0.0F);
  }
}

__global__ void ColMeanKernel(const float* x, int rows, int cols, float* out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  float acc = 0.0F;
  for (int r = 0; r < rows; ++r) acc += x[static_cast<std::size_t>(r) * cols + c];
  out[c] = acc / rows;
}

// out[o] = sum_k mean[k] * W[o][k]; one block per output.
__global__ void MatVecKernel(const float* mean, const __half* w, int k, float* out) {
  __shared__ float sh[8];
  float acc = 0.0F;
  for (int i = threadIdx.x; i < k; i += blockDim.x)
    acc += mean[i] * __half2float(w[static_cast<std::size_t>(blockIdx.x) * k + i]);
  const float total = BlockSum(acc, sh);
  if (threadIdx.x == 0) out[blockIdx.x] = total;
}

int Blocks(std::size_t n) { return static_cast<int>((n + kThreads - 1) / kThreads); }

float Bf16ToFloat(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

}  // namespace

struct EngineModel::HipState {
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  struct Layer {
    __half *wqkv{}, *wo{}, *wgu{}, *wdown{}, *wpgate{}, *wpproj{};
    float *in_norm{}, *q_norm{}, *k_norm{}, *post_attn{}, *pre_ffn{},
        *post_ffn{}, *ple_post{};
    float scalar{1.0F};
  };
  std::vector<Layer> layers;
  __half *embed{}, *ple_proj{}, *out_proj{};
  float *ple_norm{}, *final_norm{};
  // Scratch sized for kMaxTokens.
  std::uint32_t* ids{};
  std::uint32_t* pos_dev{};
  float *h{}, *qkv{}, *gu{}, *y{}, *g{}, *normed{}, *scores{}, *mean{}, *out{};
  __half *emb16{}, *x16{}, *q16{}, *k16{}, *v16{}, *ctx16{}, *act16{}, *probs{};
  std::vector<void*> allocations;

  ~HipState() {
    for (void* p : allocations) (void)hipFree(p);
    if (blas != nullptr) (void)hipblasDestroy(blas);
    if (stream != nullptr) (void)hipStreamDestroy(stream);
  }

  template <typename T>
  bool Alloc(T** ptr, std::size_t count, std::string* error) {
    void* raw = nullptr;
    if (hipMalloc(&raw, count * sizeof(T)) != hipSuccess) {
      if (error != nullptr) *error = "device allocation failed";
      return false;
    }
    allocations.push_back(raw);
    *ptr = static_cast<T*>(raw);
    return true;
  }

  bool UploadF32(const core::GgufReader& reader, const std::string& name,
                 std::size_t expect, float** dst, std::string* error) {
    const auto* t = reader.FindTensor(name);
    if (t == nullptr || t->type != core::GgmlType::kF32 ||
        t->ElementCount() != expect) {
      if (error != nullptr) *error = "missing or mis-shaped F32 tensor " + name;
      return false;
    }
    if (!Alloc(dst, expect, error)) return false;
    EG_HIP(hipMemcpy(*dst, t->data, expect * sizeof(float), hipMemcpyHostToDevice));
    return true;
  }

  // Appends a BF16 tensor to `host` as F16, range-checked.
  static bool AppendBf16(const core::GgufReader& reader, const std::string& name,
                         std::size_t expect, std::vector<__half>* host,
                         std::string* error) {
    const auto* t = reader.FindTensor(name);
    if (t == nullptr || t->type != core::GgmlType::kBF16 ||
        t->ElementCount() != expect) {
      if (error != nullptr) *error = "missing or mis-shaped BF16 tensor " + name;
      return false;
    }
    const auto* src = static_cast<const std::uint16_t*>(t->data);
    const std::size_t at = host->size();
    host->resize(at + expect);
    for (std::size_t i = 0; i < expect; ++i) {
      const float f = Bf16ToFloat(src[i]);
      if (!(std::fabs(f) < 65000.0F)) {
        if (error != nullptr) *error = "weight outside F16 range in " + name;
        return false;
      }
      (*host)[at + i] = __float2half(f);
    }
    return true;
  }

  bool UploadHost(const std::vector<__half>& host, __half** dst, std::string* error) {
    if (!Alloc(dst, host.size(), error)) return false;
    EG_HIP(hipMemcpy(*dst, host.data(), host.size() * sizeof(__half),
                     hipMemcpyHostToDevice));
    return true;
  }

  bool UploadBf16(const core::GgufReader& reader, const std::string& name,
                  std::size_t expect, __half** dst, std::string* error) {
    std::vector<__half> host;
    if (!AppendBf16(reader, name, expect, &host, error)) return false;
    return UploadHost(host, dst, error);
  }

  bool Init(const core::GgufReader& reader, const Config& c, std::string* error) {
    EG_HIP(hipStreamCreate(&stream));
    if (hipblasCreate(&blas) != HIPBLAS_STATUS_SUCCESS) {
      if (error != nullptr) *error = "hipblasCreate failed";
      return false;
    }
    hipblasSetStream(blas, stream);
    const std::size_t H = c.hidden, F = c.ffn, P = c.ple_dim;
    const auto* emb = reader.FindTensor("embed_tokens.weight");
    if (emb == nullptr) { if (error) *error = "embed_tokens.weight missing"; return false; }
    const std::size_t rows = emb->ElementCount() / H;
    if (!UploadBf16(reader, "embed_tokens.weight", rows * H, &embed, error)) return false;
    if (!UploadBf16(reader, "ple.per_layer_model_projection.weight", c.layers * P * H,
                    &ple_proj, error)) return false;
    if (!UploadF32(reader, "ple.per_layer_projection_norm.weight", P, &ple_norm, error)) return false;
    if (!UploadBf16(reader, "embedding_projection.weight", c.out_dim * H, &out_proj, error)) return false;
    if (!UploadF32(reader, "norm.weight", H, &final_norm, error)) return false;
    layers.resize(c.layers);
    std::size_t max_qkvd = 0, max_qd = 0, max_kd = 0;
    for (std::uint32_t i = 0; i < c.layers; ++i) {
      const bool full = c.layer_is_full[i] != 0;
      const std::size_t d = full ? c.head_dim_full : c.head_dim_local;
      const std::size_t kvh = full ? c.kv_full : c.kv_local;
      const std::size_t qd = c.heads * d, kd = kvh * d;
      max_qkvd = std::max(max_qkvd, qd + 2 * kd);
      max_qd = std::max(max_qd, qd);
      max_kd = std::max(max_kd, kd);
      const std::string p = "layers." + std::to_string(i) + ".";
      Layer& L = layers[i];
      std::vector<__half> host;
      if (!AppendBf16(reader, p + "self_attn.q_proj.weight", qd * H, &host, error) ||
          !AppendBf16(reader, p + "self_attn.k_proj.weight", kd * H, &host, error) ||
          !AppendBf16(reader, p + "self_attn.v_proj.weight", kd * H, &host, error) ||
          !UploadHost(host, &L.wqkv, error)) return false;
      if (!UploadBf16(reader, p + "self_attn.o_proj.weight", H * qd, &L.wo, error)) return false;
      host.clear();
      if (!AppendBf16(reader, p + "mlp.gate_proj.weight", F * H, &host, error) ||
          !AppendBf16(reader, p + "mlp.up_proj.weight", F * H, &host, error) ||
          !UploadHost(host, &L.wgu, error)) return false;
      if (!UploadBf16(reader, p + "mlp.down_proj.weight", H * F, &L.wdown, error)) return false;
      if (!UploadBf16(reader, p + "ple_block.per_layer_input_gate.weight", P * H, &L.wpgate, error)) return false;
      if (!UploadBf16(reader, p + "ple_block.per_layer_projection.weight", H * P, &L.wpproj, error)) return false;
      if (!UploadF32(reader, p + "input_layernorm.weight", H, &L.in_norm, error) ||
          !UploadF32(reader, p + "self_attn.q_norm.weight", d, &L.q_norm, error) ||
          !UploadF32(reader, p + "self_attn.k_norm.weight", d, &L.k_norm, error) ||
          !UploadF32(reader, p + "post_attention_layernorm.weight", H, &L.post_attn, error) ||
          !UploadF32(reader, p + "pre_feedforward_layernorm.weight", H, &L.pre_ffn, error) ||
          !UploadF32(reader, p + "post_feedforward_layernorm.weight", H, &L.post_ffn, error) ||
          !UploadF32(reader, p + "ple_block.post_per_layer_input_norm.weight", H, &L.ple_post, error))
        return false;
      const auto* sc = reader.FindTensor(p + "layer_scalar");
      if (sc == nullptr || sc->type != core::GgmlType::kF32 || sc->ElementCount() != 1) {
        if (error != nullptr) *error = "layer_scalar missing";
        return false;
      }
      L.scalar = *static_cast<const float*>(sc->data);
    }
    const std::size_t N = kMaxTokens;
    if (!Alloc(&ids, N, error) || !Alloc(&pos_dev, N, error) || !Alloc(&h, N * H, error) ||
        !Alloc(&qkv, N * max_qkvd, error) || !Alloc(&gu, N * 2 * F, error) ||
        !Alloc(&y, N * H, error) || !Alloc(&g, N * std::max<std::size_t>(P, H), error) ||
        !Alloc(&normed, N * H, error) ||
        !Alloc(&scores, static_cast<std::size_t>(c.heads) * kQueryChunk * N, error) ||
        !Alloc(&probs, static_cast<std::size_t>(c.heads) * kQueryChunk * N, error) ||
        !Alloc(&mean, H * kMaxSeqs, error) || !Alloc(&out, c.out_dim * kMaxSeqs, error) ||
        !Alloc(&emb16, N * H, error) || !Alloc(&x16, N * H, error) ||
        !Alloc(&q16, N * max_qd, error) || !Alloc(&k16, N * max_kd, error) ||
        !Alloc(&v16, N * max_kd, error) || !Alloc(&ctx16, N * max_qd, error) ||
        !Alloc(&act16, N * std::max<std::size_t>(F, std::max<std::size_t>(P, H)), error))
      return false;
    return true;
  }

  // Y[n][m] = X[n][k] . W[m][k]^T  (F16 in, F32 or F16 out).
  bool Gemm(const __half* w, const __half* x, void* y, hipDataType out_type,
            int m, int n, int k) {
    const float alpha = 1.0F, beta = 0.0F;
    return hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha, w,
                         HIPBLAS_R_16F, k, x, HIPBLAS_R_16F, k, &beta, y,
                         out_type, m,
                         HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT) ==
           HIPBLAS_STATUS_SUCCESS;
  }

  bool Attention(const Config& c, bool full, int n, int d, int kvh, int row0, std::string* error) {
    const int heads = c.heads, group = heads / kvh;
    const int qd = heads * d, kd = kvh * d;
    const __half* q16r = q16 + static_cast<std::size_t>(row0) * qd;
    const __half* k16r = k16 + static_cast<std::size_t>(row0) * kd;
    const __half* v16r = v16 + static_cast<std::size_t>(row0) * kd;
    __half* ctx16r = ctx16 + static_cast<std::size_t>(row0) * qd;
    const int window = full ? -1 : static_cast<int>(c.sliding_window);
    for (int q0 = 0; q0 < n; q0 += kQueryChunk) {
      const int rows = std::min(kQueryChunk, n - q0);
      int k0 = 0, k1 = n;
      if (!full) {
        k0 = std::max(0, q0 - window);
        k1 = std::min(n, q0 + rows + window);
      }
      const int nk = k1 - k0;
      const float alpha = 1.0F, beta = 0.0F;
      for (int hk = 0; hk < kvh; ++hk) {
        const int h0 = hk * group;
        const __half* a = k16r + static_cast<std::size_t>(k0) * kd + hk * d;
        const __half* b = q16r + static_cast<std::size_t>(q0) * qd + h0 * d;
        float* sc = scores + static_cast<std::size_t>(h0) * rows * nk;
        if (hipblasGemmStridedBatchedEx(
                blas, HIPBLAS_OP_T, HIPBLAS_OP_N, nk, rows, d, &alpha, a,
                HIPBLAS_R_16F, kd, 0, b, HIPBLAS_R_16F, qd, d, &beta, sc,
                HIPBLAS_R_32F, nk, static_cast<long long>(rows) * nk, group,
                HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT) != HIPBLAS_STATUS_SUCCESS) {
          if (error != nullptr) *error = "QK gemm failed";
          return false;
        }
      }
      hipLaunchKernelGGL(SoftmaxKernel, dim3(rows, heads), dim3(kThreads), 0, stream,
                         scores, rows, nk, q0, k0, window, probs);
      for (int hk = 0; hk < kvh; ++hk) {
        const int h0 = hk * group;
        const __half* a = v16r + static_cast<std::size_t>(k0) * kd + hk * d;
        const __half* b = probs + static_cast<std::size_t>(h0) * rows * nk;
        __half* o = ctx16r + static_cast<std::size_t>(q0) * qd + h0 * d;
        if (hipblasGemmStridedBatchedEx(
                blas, HIPBLAS_OP_N, HIPBLAS_OP_N, d, rows, nk, &alpha, a,
                HIPBLAS_R_16F, kd, 0, b, HIPBLAS_R_16F, nk,
                static_cast<long long>(rows) * nk, &beta, o, HIPBLAS_R_16F, qd, d,
                group, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT) != HIPBLAS_STATUS_SUCCESS) {
          if (error != nullptr) *error = "PV gemm failed";
          return false;
        }
      }
    }
    return true;
  }

  // One forward over `lens.size()` sequences concatenated in `tokens`. Every
  // token-wise stage (norms, projections, MLP, per-layer inputs) runs once
  // over all rows, so the weights are read once per layer for the whole
  // batch; attention is per sequence (bidirectional, never across sequences)
  // and RoPE positions restart at each sequence start. `results` receives one
  // pooled 768-d (pre-L2) vector per sequence.
  bool Forward(const Config& c, std::span<const std::uint32_t> tokens,
               std::span<const std::uint32_t> lens,
               std::vector<std::vector<float>>* results, std::string* error) {
    const int n = static_cast<int>(tokens.size());
    if (lens.empty() || lens.size() > kMaxSeqs || tokens.size() > kMaxTokens) {
      if (error != nullptr) *error = "batch shape outside the device envelope";
      return false;
    }
    {
      std::vector<std::uint32_t> pos(tokens.size());
      std::size_t at = 0, total = 0;
      for (const std::uint32_t len : lens) {
        for (std::uint32_t i = 0; i < len; ++i) pos[at++] = i;
        total += len;
      }
      if (total != tokens.size()) {
        if (error != nullptr) *error = "sequence lengths do not cover the batch";
        return false;
      }
      EG_HIP(hipMemcpyAsync(pos_dev, pos.data(), pos.size() * sizeof(std::uint32_t),
                            hipMemcpyHostToDevice, stream));
    }
    const int H = static_cast<int>(c.hidden), F = static_cast<int>(c.ffn);
    const int P = static_cast<int>(c.ple_dim);
    EG_HIP(hipMemcpyAsync(ids, tokens.data(), tokens.size() * sizeof(std::uint32_t),
                          hipMemcpyHostToDevice, stream));
    hipLaunchKernelGGL(EmbedGatherKernel, dim3(n), dim3(kThreads), 0, stream, ids,
                       embed, c.embed_scale, H, h, emb16);
    const float ple_pre = 1.0F / std::sqrt(static_cast<float>(H));
    for (std::uint32_t li = 0; li < c.layers; ++li) {
      const Layer& L = layers[li];
      const bool full = c.layer_is_full[li] != 0;
      const int d = full ? c.head_dim_full : c.head_dim_local;
      const int kvh = full ? c.kv_full : c.kv_local;
      const int qd = c.heads * d, kd = kvh * d;
      hipLaunchKernelGGL(RmsNormF16Kernel, dim3(n), dim3(kThreads), 0, stream, h,
                         L.in_norm, H, c.rms_eps, x16);
      if (!Gemm(L.wqkv, x16, qkv, HIPBLAS_R_32F, qd + 2 * kd, n, H)) {
        if (error != nullptr) *error = "qkv gemm failed";
        return false;
      }
      hipLaunchKernelGGL(QkvPostKernel, dim3(n, c.heads + 2 * kvh), dim3(kThreads), 0,
                         stream, qkv, L.q_norm, L.k_norm, static_cast<int>(c.heads),
                         kvh, d, full ? c.rope_full : c.rope_local, c.rms_eps, q16,
                         k16, v16, pos_dev);
      int row0 = 0;
      for (const std::uint32_t len : lens) {
        if (!Attention(c, full, static_cast<int>(len), d, kvh, row0, error)) return false;
        row0 += static_cast<int>(len);
      }
      if (!Gemm(L.wo, ctx16, y, HIPBLAS_R_32F, H, n, qd)) {
        if (error != nullptr) *error = "o_proj gemm failed";
        return false;
      }
      hipLaunchKernelGGL(RmsAddKernel, dim3(n), dim3(kThreads), 0, stream, h, y,
                         L.post_attn, H, c.rms_eps, 1.0F);
      hipLaunchKernelGGL(RmsNormF16Kernel, dim3(n), dim3(kThreads), 0, stream, h,
                         L.pre_ffn, H, c.rms_eps, x16);
      if (!Gemm(L.wgu, x16, gu, HIPBLAS_R_32F, 2 * F, n, H)) {
        if (error != nullptr) *error = "gate/up gemm failed";
        return false;
      }
      hipLaunchKernelGGL(GegluKernel, dim3(Blocks(static_cast<std::size_t>(n) * F)),
                         dim3(kThreads), 0, stream, gu, F, static_cast<std::size_t>(n),
                         act16);
      if (!Gemm(L.wdown, act16, y, HIPBLAS_R_32F, H, n, F)) {
        if (error != nullptr) *error = "down gemm failed";
        return false;
      }
      hipLaunchKernelGGL(RmsAddKernel, dim3(n), dim3(kThreads), 0, stream, h, y,
                         L.post_ffn, H, c.rms_eps, 1.0F);
      // Per-layer input block.
      if (!Gemm(ple_proj + static_cast<std::size_t>(li) * P * H, emb16, g, HIPBLAS_R_32F,
                P, n, H)) {
        if (error != nullptr) *error = "ple projection gemm failed";
        return false;
      }
      hipLaunchKernelGGL(PleNormKernel, dim3(n), dim3(kThreads), 0, stream, g,
                         ple_norm, P, ple_pre, c.rms_eps);
      // Gate GEMM (h -> P) overwrites y (n x P <= n x H scratch).
      hipLaunchKernelGGL(HToF16Kernel, dim3(Blocks(static_cast<std::size_t>(n) * H)),
                         dim3(kThreads), 0, stream, h, static_cast<std::size_t>(n) * H,
                         x16);
      if (!Gemm(L.wpgate, x16, y, HIPBLAS_R_32F, P, n, H)) {
        if (error != nullptr) *error = "ple gate gemm failed";
        return false;
      }
      hipLaunchKernelGGL(PleGateKernel, dim3(Blocks(static_cast<std::size_t>(n) * P)),
                         dim3(kThreads), 0, stream, y, g, static_cast<std::size_t>(n) * P,
                         act16);
      if (!Gemm(L.wpproj, act16, y, HIPBLAS_R_32F, H, n, P)) {
        if (error != nullptr) *error = "ple projection gemm failed";
        return false;
      }
      hipLaunchKernelGGL(RmsAddKernel, dim3(n), dim3(kThreads), 0, stream, h, y,
                         L.ple_post, H, c.rms_eps, L.scalar);
    }
    hipLaunchKernelGGL(RmsNormF32Kernel, dim3(n), dim3(kThreads), 0, stream, h,
                       final_norm, H, c.rms_eps, normed);
    {
      std::size_t row0 = 0;
      for (std::size_t si = 0; si < lens.size(); ++si) {
        hipLaunchKernelGGL(ColMeanKernel, dim3((H + kThreads - 1) / kThreads),
                           dim3(kThreads), 0, stream,
                           normed + row0 * static_cast<std::size_t>(H),
                           static_cast<int>(lens[si]), H, mean + si * H);
        hipLaunchKernelGGL(MatVecKernel, dim3(c.out_dim), dim3(kThreads), 0, stream,
                           mean + si * H, out_proj, H, out + si * c.out_dim);
        row0 += lens[si];
      }
    }
    std::vector<float> flat(lens.size() * c.out_dim);
    EG_HIP(hipMemcpyAsync(flat.data(), out, flat.size() * sizeof(float),
                          hipMemcpyDeviceToHost, stream));
    EG_HIP(hipStreamSynchronize(stream));
    results->assign(lens.size(), std::vector<float>());
    for (std::size_t si = 0; si < lens.size(); ++si)
      (*results)[si].assign(flat.begin() + si * c.out_dim,
                            flat.begin() + (si + 1) * c.out_dim);
    EG_HIP(hipGetLastError());
    return true;
  }
};

EngineModel::EngineModel() = default;
EngineModel::~EngineModel() = default;

std::optional<std::vector<EmbedResult>> EngineModel::EmbedBatch(
    std::span<const std::vector<std::uint32_t>> batch, std::string* error_msg) {
  auto refuse = [&](const std::string& m) {
    if (error_msg != nullptr) *error_msg = m;
    return std::optional<std::vector<EmbedResult>>{};
  };
  if (batch.empty()) return refuse("embeddinggemma2: empty batch");
  for (const auto& ids : batch) {
    if (ids.empty()) return refuse("embeddinggemma2: empty id sequence");
    if (ids.size() > kMaxTokens)
      return refuse("embeddinggemma2: " + std::to_string(ids.size()) +
                    " tokens exceeds the " + std::to_string(kMaxTokens) + " window");
  }
  std::vector<EmbedResult> results(batch.size());
  std::vector<std::size_t> pending;
  for (std::size_t i = 0; i < batch.size(); ++i) {
    std::optional<EmbedResult> npu_out;
    if (TryNpuEmbed(batch[i], &npu_out, error_msg)) {
      if (!npu_out.has_value()) return std::nullopt;
      results[i] = std::move(*npu_out);
    } else {
      pending.push_back(i);
    }
  }
  if (!pending.empty() && hip_ == nullptr) {
    auto state = std::make_unique<HipState>();
    std::string error;
    if (!state->Init(*reader_, config_, &error))
      return refuse("embeddinggemma2 device init failed: " + error);
    hip_ = std::move(state);
  }
  // Greedy groups in input order bounded by the device envelope.
  for (std::size_t at = 0; at < pending.size();) {
    std::vector<std::uint32_t> tokens, lens;
    std::size_t end = at;
    while (end < pending.size() && lens.size() < kMaxSeqs &&
           tokens.size() + batch[pending[end]].size() <= kMaxTokens) {
      const auto& ids = batch[pending[end]];
      tokens.insert(tokens.end(), ids.begin(), ids.end());
      lens.push_back(static_cast<std::uint32_t>(ids.size()));
      ++end;
    }
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> pooled;
    std::string error;
    if (!hip_->Forward(config_, tokens, lens, &pooled, &error))
      return refuse("embeddinggemma2: " + error);
    const double group_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start).count();
    for (std::size_t k = 0; k < lens.size(); ++k) {
      auto& values = pooled[k];
      double norm = 0.0;
      for (float v : values) norm += static_cast<double>(v) * v;
      norm = std::sqrt(norm);
      if (!(norm > 0.0)) return refuse("embeddinggemma2: zero-norm embedding");
      for (float& v : values) v = static_cast<float>(v / norm);
      EmbedResult& r = results[pending[at + k]];
      r.values = std::move(values);
      r.prefill_ms = group_ms * static_cast<double>(lens[k]) /
                     static_cast<double>(tokens.size());
    }
    at = end;
  }
  return results;
}

std::optional<EmbedResult> EngineModel::Embed(std::span<const std::uint32_t> ids,
                                              std::string* error_msg) {
  const std::vector<std::uint32_t> one(ids.begin(), ids.end());
  auto batch = EmbedBatch(std::span<const std::vector<std::uint32_t>>(&one, 1), error_msg);
  if (!batch.has_value()) return std::nullopt;
  return std::move(batch->front());
}

}  // namespace mbufo::models::embeddinggemma2
