// bge-m3 HIP prefill (leg-2 step 2): the same full-sequence encoder math as
// CpuForward, F16 projections via hipBLASLt (F32 accumulate/output) plus
// F32 strided-batched scores and fused attention elementwise.
//
// What changed from leg-1 (F32 SGEMM throughout):
//   * step 1: the projection tables per layer stay F16 on the device
//     (raw GGUF bytes, one upload, no F32 mirror on device); each layer
//     narrows its F32 activation once per distinct input into a shared F16
//     staging buffer; hipBLASLt runs the GEMM with an exact-(m,n,k) plan
//     cache (workspace-free heuristic selection, warmed once at load).
//   * step 2: the three QKV projections share one narrowed activation and
//     a single fused split+bias kernel fans them out to per-head buffers
//     (bias folded into the split; bit-exact vs split-then-add). A single
//     [3h, h] GEMM was tried first and reverted (its small-n kernel loses
//     to three m=1024 GEMMs); the projections stay three launches over the
//     proven shapes. Scores stay a strided-batched SGEMM (it streams V once
//     per layer; a fused per-row context would re-read V per query and
//     multiply traffic by n), followed by the in-place softmax, a
//     strided-batched context SGEMM, and the merge. Per layer this cuts
//     25 launches to 20 with identical buffer traffic.
// Attention scores, LayerNorm, GELU all stay F32. Precision gate:
// HIP-vs-CPU cosine >= 0.999 (leg-1 was ~1.0).
//
// Layout note for the remaining hipBLAS calls: column-major, issued as
//   hipblasSgemm(OP_T, OP_N, N, M, K, &alpha, B, K, A, K, &beta, C, N)
// over row-major buffers. hipBLASLt takes row-major semantics directly:
// out[n][m] = in[n][k] x W[m][k]^T with weights layout (k, m), input
// layout (k, n), output layout (m, n), transA=T, transB=N.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "src/core/compute/gelu/gelu.hpp"
#include "src/models/bge_m3/engine.hpp"

namespace gufo::models::bge_m3 {
namespace {

// ---------------------------------------------------------------- kernels

// out[row] = LayerNorm(word[row] + pos[row] + ttype) for row in [0, n).
// word/pos are [n, hidden] staging; ttype is the single row-0 vector.
__global__ void EmbedKernel(const float* __restrict__ word,
                            const float* __restrict__ pos,
                            const float* __restrict__ ttype,
                            const float* __restrict__ norm_w,
                            const float* __restrict__ norm_b, float eps,
                            float* __restrict__ out, int n, int hidden) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= n) return;
  float* orow = out + static_cast<std::size_t>(row) * hidden;
  const float* wrow = word + static_cast<std::size_t>(row) * hidden;
  const float* prow = pos + static_cast<std::size_t>(row) * hidden;
  double sum = 0.0, sumsq = 0.0;
  for (int d = static_cast<int>(threadIdx.x); d < hidden;
       d += static_cast<int>(blockDim.x)) {
    const double v =
        static_cast<double>(wrow[d] + prow[d] + ttype[d]);
    orow[d] = static_cast<float>(v);
    sum += v;
    sumsq += v * v;
  }
  __shared__ double sh_sum[256];
  __shared__ double sh_sq[256];
  const int tid = static_cast<int>(threadIdx.x);
  sh_sum[tid] = sum;
  sh_sq[tid] = sumsq;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sh_sum[tid] += sh_sum[tid + stride];
      sh_sq[tid] += sh_sq[tid + stride];
    }
    __syncthreads();
  }
  const double mean = sh_sum[0] / hidden;
  double var = sh_sq[0] / hidden - mean * mean;
  if (var < 0 && var > -1e-12) var = 0;
  const float inv =
      static_cast<float>(1.0 / sqrt(var + static_cast<double>(eps)));
  const float mean_f = static_cast<float>(mean);
  for (int d = tid; d < hidden; d += static_cast<int>(blockDim.x)) {
    orow[d] = (orow[d] - mean_f) * inv * norm_w[d] + norm_b[d];
  }
}

// Fused QKV split + bias (leg-2 step 2): one launch fans the three [n, h]
// projection outputs out to the per-head q/k/v buffers while adding each
// part's bias. Bit-exact vs the old split-then-BiasAdd order (one addition
// per element, same order): out[(hh, i), d] = in_part[i, hh*hd + d]
// + bias_part[hh*hd + d]. (A single [3h, h] GEMM was tried first and
// reverted: the m=3072 hipBLASLt kernel loses to three m=1024 GEMMs at
// small n, so the projections stay three launches over the proven shapes
// and only the split/bias fan-out fuses.)
__global__ void QkvSplitBiasKernel(const float* __restrict__ q,
                                   const float* __restrict__ k,
                                   const float* __restrict__ v,
                                   const float* __restrict__ q_b,
                                   const float* __restrict__ k_b,
                                   const float* __restrict__ v_b,
                                   float* __restrict__ qh,
                                   float* __restrict__ kh,
                                   float* __restrict__ vh, int n, int heads,
                                   int hd) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int h = heads * hd;
  const std::size_t plane = static_cast<std::size_t>(n) * h;
  if (idx >= plane * 3) return;
  const int part = static_cast<int>(idx / plane);
  const std::size_t rem = idx % plane;
  const int i = static_cast<int>(rem / static_cast<std::size_t>(h));
  const int dh = static_cast<int>(rem % static_cast<std::size_t>(h));
  const int hh = dh / hd;
  const int d = dh % hd;
  const float* in = (part == 0 ? q : (part == 1 ? k : v));
  const float bias = (part == 0 ? q_b : (part == 1 ? k_b : v_b))[dh];
  float* out = (part == 0 ? qh : (part == 1 ? kh : vh));
  out[(static_cast<std::size_t>(hh) * n + i) * hd + d] =
      in[static_cast<std::size_t>(i) * h + dh] + bias;
}

// In-place rowwise softmax over [rows, cols]. (Kept as its own kernel:
// folding context into the same pass would re-stream V once per query
// row, multiplying V traffic by n; the strided-batched context SGEMM
// below streams V once per layer instead.)
__global__ void SoftmaxRowsKernel(float* __restrict__ scores, int rows,
                                  int cols) {
  const int r = static_cast<int>(blockIdx.x);
  if (r >= rows) return;
  float* srow = scores + static_cast<std::size_t>(r) * cols;
  const int bd = static_cast<int>(blockDim.x);
  const int tid = static_cast<int>(threadIdx.x);
  float local_max = -3.402823466e+38F;
  for (int j = tid; j < cols; j += bd) {
    if (srow[j] > local_max) local_max = srow[j];
  }
  __shared__ float sh_max[256];
  sh_max[tid] = local_max;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride && sh_max[tid + stride] > sh_max[tid]) {
      sh_max[tid] = sh_max[tid + stride];
    }
    __syncthreads();
  }
  const float mx = sh_max[0];
  double sum = 0.0;
  for (int j = tid; j < cols; j += bd) {
    const double e = exp(static_cast<double>(srow[j] - mx));
    srow[j] = static_cast<float>(e);
    sum += e;
  }
  __shared__ double sh_sum[256];
  sh_sum[tid] = sum;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) sh_sum[tid] += sh_sum[tid + stride];
    __syncthreads();
  }
  const float inv = static_cast<float>(1.0 / sh_sum[0]);
  for (int j = tid; j < cols; j += bd) srow[j] *= inv;
}

__global__ void MergeHeadsKernel(const float* __restrict__ in,
                                 float* __restrict__ out, int seq, int heads,
                                 int head_dim) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total =
      static_cast<std::size_t>(seq) * heads * head_dim;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % static_cast<std::size_t>(head_dim));
  const std::size_t tmp = idx / static_cast<std::size_t>(head_dim);
  const int i = static_cast<int>(tmp % static_cast<std::size_t>(seq));
  const int hh = static_cast<int>(tmp / static_cast<std::size_t>(seq));
  const int hidden = heads * head_dim;
  out[static_cast<std::size_t>(i) * hidden +
      static_cast<std::size_t>(hh) * head_dim + d] =
      in[(static_cast<std::size_t>(hh) * seq + i) * head_dim + d];
}

// out[row] = bias[row % cols] added in place (x == out safe).
__global__ void BiasAddKernel(float* __restrict__ x, const float* __restrict__ bias,
                              int rows, int cols) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  if (idx >= total) return;
  x[idx] += bias[idx % static_cast<std::size_t>(cols)];
}

// out[row] = LayerNorm(x[row] + y[row]).
__global__ void ResidualLnKernel(const float* __restrict__ x,
                                 const float* __restrict__ y,
                                 const float* __restrict__ norm_w,
                                 const float* __restrict__ norm_b, float eps,
                                 float* __restrict__ out, int n, int hidden) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= n) return;
  const float* xr = x + static_cast<std::size_t>(row) * hidden;
  const float* yr = y + static_cast<std::size_t>(row) * hidden;
  float* orow = out + static_cast<std::size_t>(row) * hidden;
  const int bd = static_cast<int>(blockDim.x);
  const int tid = static_cast<int>(threadIdx.x);
  double sum = 0.0, sumsq = 0.0;
  for (int d = tid; d < hidden; d += bd) {
    const double v = static_cast<double>(xr[d]) + static_cast<double>(yr[d]);
    orow[d] = static_cast<float>(v);
    sum += v;
    sumsq += v * v;
  }
  __shared__ double sh_sum[256];
  __shared__ double sh_sq[256];
  sh_sum[tid] = sum;
  sh_sq[tid] = sumsq;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sh_sum[tid] += sh_sum[tid + stride];
      sh_sq[tid] += sh_sq[tid + stride];
    }
    __syncthreads();
  }
  const double mean = sh_sum[0] / hidden;
  double var = sh_sq[0] / hidden - mean * mean;
  if (var < 0 && var > -1e-12) var = 0;
  const float inv =
      static_cast<float>(1.0 / sqrt(var + static_cast<double>(eps)));
  const float mean_f = static_cast<float>(mean);
  for (int d = tid; d < hidden; d += bd) {
    orow[d] = (orow[d] - mean_f) * inv * norm_w[d] + norm_b[d];
  }
}

// out[i] = GELU-exact(x[i] + bias[i % cols]).
__device__ inline float GeluExactDevice(float x);

__global__ void BiasGeluKernel(const float* __restrict__ x,
                               const float* __restrict__ bias,
                               float* __restrict__ out, int rows, int cols) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  if (idx >= total) return;
  out[idx] = GeluExactDevice(
      x[idx] + bias[idx % static_cast<std::size_t>(cols)]);
}

// Device twin of core::gelu::GeluExact: the core helper is host-only
// (std::erf), while HIP device code provides its own erf.
__device__ inline float GeluExactDevice(float x) {
  constexpr double kSqrtHalf = 0.70710678118654752440;  // 1/sqrt(2)
  const double xd = static_cast<double>(x);
  return static_cast<float>(0.5 * xd * (1.0 + erf(xd * kSqrtHalf)));
}

// F32 activations narrowed to F16 for the hipBLASLt projections (step 1;
// the GEMMs accumulate in F32, so this single rounding is the only new
// error source on the projection path).
__global__ void NarrowHalfKernel(const float* __restrict__ in,
                                 __half* __restrict__ out,
                                 std::size_t count) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[idx] = __half(in[idx]);
  }
}

// ------------------------------------------------- hipBLASLt plan cache
// Minimal per-state runner mirroring the shared HipBlasLtRunner selection
// (exact (m, n, k) keys, workspace-free heuristic pick, no trial GEMMs on
// model data). Only F16-in/F32-out dense projections are needed here, so
// there are no own-kernel hooks: every shape goes through the library.
struct BlasLtPlan {
  hipblasLtMatmulDesc_t op{nullptr};
  hipblasLtMatrixLayout_t weights{nullptr};
  hipblasLtMatrixLayout_t input{nullptr};
  hipblasLtMatrixLayout_t output{nullptr};
  hipblasLtMatmulAlgo_t algo{};

  ~BlasLtPlan() {
    if (output != nullptr) hipblasLtMatrixLayoutDestroy(output);
    if (input != nullptr) hipblasLtMatrixLayoutDestroy(input);
    if (weights != nullptr) hipblasLtMatrixLayoutDestroy(weights);
    if (op != nullptr) hipblasLtMatmulDescDestroy(op);
  }
};

bool BlasLtMakePlan(hipblasLtHandle_t handle, int m, int n, int k,
                     BlasLtPlan& plan) {
  const hipblasOperation_t transpose = HIPBLAS_OP_T;
  const hipblasOperation_t normal = HIPBLAS_OP_N;
  if (hipblasLtMatmulDescCreate(&plan.op, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(plan.op, HIPBLASLT_MATMUL_DESC_TRANSA,
                                      &transpose, sizeof(transpose)) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(plan.op, HIPBLASLT_MATMUL_DESC_TRANSB,
                                      &normal, sizeof(normal)) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&plan.weights, HIP_R_16F, k, m, k) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&plan.input, HIP_R_16F, k, n, k) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&plan.output, HIP_R_32F, m, n, m) !=
          HIPBLAS_STATUS_SUCCESS) {
    return false;
  }
  const float one = 1.0F;
  const float zero = 0.0F;
  const auto usable = [&](hipblasLtMatmulAlgo_t& algo) {
    std::size_t workspace = 0;
    return hipblaslt_ext::matmulIsAlgoSupported(
               handle, plan.op, &one, plan.weights, plan.input, &zero,
               plan.output, plan.output, algo,
               workspace) == HIPBLAS_STATUS_SUCCESS &&
           workspace == 0;
  };
  hipblasLtMatmulPreference_t preference = nullptr;
  if (hipblasLtMatmulPreferenceCreate(&preference) != HIPBLAS_STATUS_SUCCESS) {
    return false;
  }
  constexpr std::size_t kWorkspace = 0;
  (void)hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspace,
      sizeof(kWorkspace));
  std::array<hipblasLtMatmulHeuristicResult_t, 16> candidates{};
  int count = 0;
  const auto status = hipblasLtMatmulAlgoGetHeuristic(
      handle, plan.op, plan.weights, plan.input, plan.output, plan.output,
      preference, candidates.size(), candidates.data(), &count);
  (void)hipblasLtMatmulPreferenceDestroy(preference);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return false;
  }
  for (int i = 0; i < count; ++i) {
    if (candidates[i].state == HIPBLAS_STATUS_SUCCESS &&
        candidates[i].workspaceSize == 0 && usable(candidates[i].algo)) {
      plan.algo = candidates[i].algo;
      return true;
    }
  }
  return false;
}

using BlasLtPlanCache =
    std::map<std::array<int, 3>, std::unique_ptr<BlasLtPlan>>;

// Row-major out[n][m] = input[n][k] x weights[m][k]^T, F16 in, F32 out.
bool BlasLtGemm(hipblasLtHandle_t handle, BlasLtPlanCache& plans,
                const void* weights, const __half* input, float* out, int m,
                int n, int k, hipStream_t stream) {
  if (handle == nullptr || weights == nullptr || input == nullptr ||
      out == nullptr || m <= 0 || n <= 0 || k <= 0) {
    return false;
  }
  const std::array<int, 3> key{m, n, k};
  auto it = plans.find(key);
  if (it == plans.end()) {
    auto plan = std::make_unique<BlasLtPlan>();
    if (!BlasLtMakePlan(handle, m, n, k, *plan)) {
      return false;
    }
    it = plans.emplace(key, std::move(plan)).first;
  }
  const BlasLtPlan& plan = *it->second;
  const float one = 1.0F;
  const float zero = 0.0F;
  return hipblasLtMatmul(handle, plan.op, &one, weights, plan.weights, input,
                         plan.input, &zero, out, plan.output, out,
                         plan.output, &plan.algo, nullptr, 0,
                         stream) == HIPBLAS_STATUS_SUCCESS;
}

}  // namespace

// ------------------------------------------------------------ device state
// (bge_m3 scope: HipState is EngineModel's nested type and InitHipState
// / HipSync are its friends, so they cannot live in the anonymous
// namespace above.)

struct HipWeights {
  std::vector<void*> owned;  // every hipMalloc, freed with the state
  const BgeM3F32Weights* f32{nullptr};
  // F32 tables still consumed on device (embed positions/types, all biases
  // and LayerNorm parameters):
  const float* position_embd{nullptr};
  const float* token_types{nullptr};
  const float* emb_norm_w{nullptr};
  const float* emb_norm_b{nullptr};
  struct HipLayer {
    // Projection tables stay F16 on device (raw GGUF bytes, no F32 mirror):
    const void *q_w, *k_w, *v_w;
    const float *q_b, *k_b, *v_b;
    const void* out_w;
    const float *out_b, *attn_norm_w, *attn_norm_b;
    const void *up_w, *down_w;
    const float *up_b, *down_b, *out_norm_w, *out_norm_b;
  };
  std::vector<HipLayer> layers;
};

struct EngineModel::HipState {
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  hipblasLtHandle_t blaslt{nullptr};
  std::mutex mutex;
  HipWeights weights;
  BlasLtPlanCache plans;  // destroyed before blaslt (declared after it)
  float *hidden{nullptr}, *normed{nullptr};
  float *qkv{nullptr};
  float *split{nullptr};
  float *scores{nullptr};
  float *context{nullptr};
  float *proj{nullptr};
  float *ffn_inter{nullptr};
  float *ffn_out{nullptr};
  __half* half_buf{nullptr};  // shared F16 activation staging
  bool ok{false};
  std::string error;
  ~HipState() {
    plans.clear();  // descriptors reference the handle; drop them first
    if (blaslt != nullptr) hipblasLtDestroy(blaslt);
    // Buffers are named individually for the prefill below; every one of
    // them is also tracked in weights.owned for release.
    for (void* p : weights.owned) hipFree(p);
    weights.owned.clear();
    if (blas != nullptr) hipblasDestroy(blas);
    if (stream != nullptr) hipStreamDestroy(stream);
  }
};

const float* UploadVec(const std::vector<float>& host,
                       std::vector<void*>& owned, hipStream_t stream,
                       bool& ok) {
  if (host.empty()) {
    ok = false;
    return nullptr;
  }
  float* dev = nullptr;
  if (hipMalloc(&dev, host.size() * sizeof(float)) != hipSuccess) {
    ok = false;
    return nullptr;
  }
  owned.push_back(dev);
  if (hipMemcpyAsync(dev, host.data(), host.size() * sizeof(float),
                     hipMemcpyHostToDevice, stream) != hipSuccess) {
    ok = false;
    return nullptr;
  }
  return dev;
}

// Raw GGUF bytes (F16 projection tables): uploaded as-is, never upcast.
const void* UploadRaw(const void* host, std::size_t bytes,
                      std::vector<void*>& owned, hipStream_t stream,
                      bool& ok) {
  if (host == nullptr || bytes == 0) {
    ok = false;
    return nullptr;
  }
  void* dev = nullptr;
  if (hipMalloc(&dev, bytes) != hipSuccess) {
    ok = false;
    return nullptr;
  }
  owned.push_back(dev);
  if (hipMemcpyAsync(dev, host, bytes, hipMemcpyHostToDevice, stream) !=
      hipSuccess) {
    ok = false;
    return nullptr;
  }
  return dev;
}

float* AllocScratch(std::size_t elements, std::vector<void*>& owned) {
  float* dev = nullptr;
  if (hipMalloc(&dev, elements * sizeof(float)) != hipSuccess) {
    return nullptr;
  }
  owned.push_back(dev);
  return dev;
}

__half* AllocHalf(std::size_t elements, std::vector<void*>& owned) {
  __half* dev = nullptr;
  if (hipMalloc(&dev, elements * sizeof(__half)) != hipSuccess) {
    return nullptr;
  }
  owned.push_back(dev);
  return dev;
}

void InitHipState(EngineModel& model) {
  auto state = std::make_unique<EngineModel::HipState>();
  bool ok = true;
  if (hipStreamCreate(&state->stream) != hipSuccess) {
    state->error = "bge-m3 HIP: hipStreamCreate failed";
    model.hip_ = std::move(state);
    return;
  }
  if (hipblasCreate(&state->blas) != hipSuccess) {
    state->error = "bge-m3 HIP: hipblasCreate failed";
    model.hip_ = std::move(state);
    return;
  }
  hipblasSetStream(state->blas, state->stream);
  if (hipblasLtCreate(&state->blaslt) != HIPBLAS_STATUS_SUCCESS) {
    state->error = "bge-m3 HIP: hipblasLtCreate failed";
    model.hip_ = std::move(state);
    return;
  }
  const BgeM3F32Weights& f32 = model.f32_;
  const BgeM3Weights& raw = model.weights_;
  HipWeights& w = state->weights;
  w.f32 = &f32;
  std::vector<void*> owned;
  w.position_embd = UploadVec(f32.position_embd, owned, state->stream, ok);
  w.token_types = UploadVec(f32.token_types, owned, state->stream, ok);
  w.emb_norm_w = UploadVec(f32.emb_norm_w, owned, state->stream, ok);
  w.emb_norm_b = UploadVec(f32.emb_norm_b, owned, state->stream, ok);
  // Leg-2 step 1 requires raw F16 projection tables; anything else refuses
  // the F16 path explicitly (never a silent upcast back to F32).
  const auto upload_f16 = [&](const BgeM3TensorRef& ref, const void*& dst) {
    if (ref.type != core::GgmlType::kF16) {
      ok = false;
      return;
    }
    dst = UploadRaw(ref.data, ref.num_elements * sizeof(std::uint16_t), owned,
                    state->stream, ok);
  };
  w.layers.resize(f32.layers.size());
  for (std::size_t i = 0; i < f32.layers.size() && ok; ++i) {
    const auto& src = f32.layers[i];
    const auto& raw_layer = raw.layers[i];
    auto& dst = w.layers[i];
    upload_f16(raw_layer.attn_q_weight, dst.q_w);
    upload_f16(raw_layer.attn_k_weight, dst.k_w);
    upload_f16(raw_layer.attn_v_weight, dst.v_w);
    upload_f16(raw_layer.attn_output_weight, dst.out_w);
    upload_f16(raw_layer.ffn_up_weight, dst.up_w);
    upload_f16(raw_layer.ffn_down_weight, dst.down_w);
    dst.q_b = UploadVec(src.q_b, owned, state->stream, ok);
    dst.k_b = UploadVec(src.k_b, owned, state->stream, ok);
    dst.v_b = UploadVec(src.v_b, owned, state->stream, ok);
    dst.out_b = UploadVec(src.out_b, owned, state->stream, ok);
    dst.attn_norm_w = UploadVec(src.attn_norm_w, owned, state->stream, ok);
    dst.attn_norm_b = UploadVec(src.attn_norm_b, owned, state->stream, ok);
    dst.up_b = UploadVec(src.up_b, owned, state->stream, ok);
    dst.down_b = UploadVec(src.down_b, owned, state->stream, ok);
    dst.out_norm_w = UploadVec(src.out_norm_w, owned, state->stream, ok);
    dst.out_norm_b = UploadVec(src.out_norm_b, owned, state->stream, ok);
  }
  if (!ok && state->error.empty()) {
    state->error =
        "bge-m3 HIP: F16 projection upload failed (non-F16 storage?)";
  }
  const std::size_t max_seq = kBgeM3Leg1MaxTokens;
  const std::size_t h = f32.config.hidden_size;
  const std::size_t heads = f32.config.num_attention_heads;
  const std::size_t hd = h / heads;
  const std::size_t inter = f32.config.intermediate_size;
  state->hidden = AllocScratch(max_seq * h, owned);
  state->normed = AllocScratch(max_seq * h, owned);
  state->qkv = AllocScratch(3 * max_seq * h, owned);
  state->split = AllocScratch(3 * heads * max_seq * hd, owned);
  state->scores = AllocScratch(heads * max_seq * max_seq, owned);
  state->context = AllocScratch(heads * max_seq * hd, owned);
  state->proj = AllocScratch(max_seq * h, owned);
  state->ffn_inter = AllocScratch(max_seq * inter, owned);
  state->ffn_out = AllocScratch(max_seq * h, owned);
  state->half_buf = AllocHalf(max_seq * inter, owned);
  ok = ok && state->hidden != nullptr && state->normed != nullptr &&
       state->qkv != nullptr && state->split != nullptr &&
       state->scores != nullptr && state->context != nullptr &&
       state->proj != nullptr && state->ffn_inter != nullptr &&
       state->ffn_out != nullptr && state->half_buf != nullptr;
  if (!ok && state->error.empty()) {
    state->error = "bge-m3 HIP: device allocation or upload failed";
  }
  if (hipStreamSynchronize(state->stream) != hipSuccess) {
    ok = false;
    if (state->error.empty()) {
      state->error = "bge-m3 HIP: initial upload sync failed";
    }
  }
  state->ok = ok;
  if (!ok) {
    for (void* p : owned) hipFree(p);
    model.hip_ = std::make_unique<EngineModel::HipState>();
    model.hip_->error = state->error;
    return;
  }
  // Warm the hipBLASLt solution library at load time (same category as
  // weight uploads: once per process, outside the prefill timer), so the
  // first prefill only pays per-shape heuristic picks (~ms each), not the
  // library init (~100 ms on first use). A later prefill with n == 128
  // reuses this plan exactly; any other n builds its own on first use.
  {
    const int warm_m = static_cast<int>(h);
    auto warm = std::make_unique<BlasLtPlan>();
    if (BlasLtMakePlan(state->blaslt, warm_m, 128, warm_m, *warm)) {
      state->plans.emplace(std::array<int, 3>{warm_m, 128, warm_m},
                           std::move(warm));
    }
  }
  state->weights.owned = std::move(owned);
  model.hip_ = std::move(state);
}

// Out-of-line EngineModel destructor: HipState is complete only here.
EngineModel::EngineModel() = default;
EngineModel::~EngineModel() = default;

std::optional<EngineSession::SyncResult> HipSync(
    const EngineModel& model, std::span<const std::uint32_t> ids,
    std::string* error_msg) {
  auto refuse = [&](const std::string& message) {
    if (error_msg != nullptr) *error_msg = message;
    return std::optional<EngineSession::SyncResult>{};
  };
  EngineModel& mutable_model = const_cast<EngineModel&>(model);
  if (mutable_model.hip_ == nullptr) {
    InitHipState(mutable_model);
  }
  EngineModel::HipState& st = *mutable_model.hip_;
  const std::lock_guard<std::mutex> lock(st.mutex);
  if (!st.ok) {
    return refuse(st.error.empty() ? "bge-m3 HIP: device state unavailable"
                                    : st.error);
  }
  const BgeM3Config& config = model.Config();
  const std::size_t n = ids.size();
  const std::size_t h = config.hidden_size;
  const std::size_t heads = config.num_attention_heads;
  const std::size_t hd = h / heads;
  const std::size_t inter = config.intermediate_size;
  if (n == 0 || n > kBgeM3Leg1MaxTokens) {
    return refuse("bge-m3 HIP Sync: sequence length out of leg-1 window");
  }
  const float kOne = 1.0F;
  const float kZero = 0.0F;
  const float kScale =
      static_cast<float>(1.0 / sqrt(static_cast<double>(hd)));
  const auto start = std::chrono::steady_clock::now();
  auto fail = [&](const std::string& what) {
    return refuse("bge-m3 HIP prefill failed at " + what);
  };
  const HipWeights& w = st.weights;
  const int nn = static_cast<int>(n);
  const int hh = static_cast<int>(heads);
  const int hdd = static_cast<int>(hd);
  const int hdim = static_cast<int>(h);
  const int ninter = static_cast<int>(inter);
  constexpr unsigned kBlock = 256;
  const auto blocks = [&](std::size_t total) {
    return static_cast<unsigned>((total + kBlock - 1) / kBlock);
  };
  const auto narrow = [&](const float* src, std::size_t count) {
    NarrowHalfKernel<<<blocks(count), kBlock, 0, st.stream>>>(
        src, st.half_buf, count);
  };

  // Embed: word rows gathered on the host (ids are host data), uploaded to
  // qkv staging, then pos/type added and normalized into hidden. The upload
  // is synchronous: `staged` is a stack local, so an async copy could read
  // it after it dies.
  {
    std::vector<float> staged(n * h);
    for (std::size_t t = 0; t < n; ++t) {
      if (ids[t] >= config.vocab_size) {
        return refuse("bge-m3 HIP Sync: token id out of vocabulary");
      }
      const float* src = w.f32->token_embd.data() +
                         static_cast<std::size_t>(ids[t]) * h;
      std::copy(src, src + h, staged.data() + t * h);
    }
    if (hipMemcpy(st.qkv, staged.data(), staged.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) {
      return fail("embed upload");
    }
  }
  EmbedKernel<<<static_cast<unsigned>(n), kBlock, 0, st.stream>>>(
      st.qkv, w.position_embd, w.token_types, w.emb_norm_w, w.emb_norm_b,
      config.layer_norm_eps, st.hidden, nn, hdim);

  float* merged = st.qkv;  // [3*n*h] QKV output, then merged [n,h] reuse
  float* qh = st.split;
  float* kh = st.split + heads * n * hd;
  float* vh = st.split + 2 * heads * n * hd;
  const long long stride_qkv = static_cast<long long>(n) * hdd;
  const long long stride_scores = static_cast<long long>(n) * n;

  float* q = st.qkv;
  float* k = st.qkv + n * h;
  float* v = st.qkv + 2 * n * h;

  for (const auto& layer : w.layers) {
    // QKV: three F16 hipBLASLt projections over one shared narrowed
    // activation, then a single fused split+bias kernel fans all three
    // parts out to the per-head buffers (bias folded into the split).
    narrow(st.hidden, n * h);
    if (!BlasLtGemm(st.blaslt, st.plans, layer.q_w, st.half_buf, q, hdim, nn,
                    hdim, st.stream)) {
      return fail("attn_q gemm");
    }
    if (!BlasLtGemm(st.blaslt, st.plans, layer.k_w, st.half_buf, k, hdim, nn,
                    hdim, st.stream)) {
      return fail("attn_k gemm");
    }
    if (!BlasLtGemm(st.blaslt, st.plans, layer.v_w, st.half_buf, v, hdim, nn,
                    hdim, st.stream)) {
      return fail("attn_v gemm");
    }
    QkvSplitBiasKernel<<<blocks(3 * n * h), kBlock, 0, st.stream>>>(
        q, k, v, layer.q_b, layer.k_b, layer.v_b, qh, kh, vh, nn, hh, hdd);
    // Batched scores = QK^T, in-place softmax, strided-batched context
    // (streams V once per layer), merge back to [n,h] into the now-dead
    // QKV staging.
    if (hipblasSgemmStridedBatched(
            st.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, nn, nn, hdd, &kScale, kh,
            hdd, stride_qkv, qh, hdd, stride_qkv, &kZero, st.scores, nn,
            stride_scores, hh) != HIPBLAS_STATUS_SUCCESS) {
      return fail("attention scores gemm");
    }
    SoftmaxRowsKernel<<<hh * static_cast<unsigned>(nn), kBlock, 0,
                        st.stream>>>(st.scores, hh * nn, nn);
    if (hipblasSgemmStridedBatched(
            st.blas, HIPBLAS_OP_N, HIPBLAS_OP_N, hdd, nn, nn, &kOne, vh, hdd,
            stride_qkv, st.scores, nn, stride_scores, &kZero, st.context,
            hdd, stride_qkv, hh) != HIPBLAS_STATUS_SUCCESS) {
      return fail("attention context gemm");
    }
    MergeHeadsKernel<<<blocks(heads * n * hd), kBlock, 0, st.stream>>>(
        st.context, merged, nn, hh, hdd);
    // Output projection + bias, residual + post-attention LayerNorm.
    narrow(merged, n * h);
    if (!BlasLtGemm(st.blaslt, st.plans, layer.out_w, st.half_buf, st.proj,
                    hdim, nn, hdim, st.stream)) {
      return fail("attn_output gemm");
    }
    BiasAddKernel<<<blocks(n * h), kBlock, 0, st.stream>>>(st.proj,
                                                           layer.out_b, nn,
                                                           hdim);
    ResidualLnKernel<<<static_cast<unsigned>(n), kBlock, 0, st.stream>>>(
        st.hidden, st.proj, layer.attn_norm_w, layer.attn_norm_b,
        config.layer_norm_eps, st.normed, nn, hdim);
    // FFN: up + bias + exact GELU, down + bias, residual + LayerNorm.
    narrow(st.normed, n * h);
    if (!BlasLtGemm(st.blaslt, st.plans, layer.up_w, st.half_buf,
                    st.ffn_inter, ninter, nn, hdim, st.stream)) {
      return fail("ffn_up gemm");
    }
    BiasGeluKernel<<<blocks(n * inter), kBlock, 0, st.stream>>>(
        st.ffn_inter, layer.up_b, st.ffn_inter, nn, ninter);
    narrow(st.ffn_inter, n * inter);
    if (!BlasLtGemm(st.blaslt, st.plans, layer.down_w, st.half_buf,
                    st.ffn_out, hdim, nn, ninter, st.stream)) {
      return fail("ffn_down gemm");
    }
    BiasAddKernel<<<blocks(n * h), kBlock, 0, st.stream>>>(st.ffn_out,
                                                           layer.down_b, nn,
                                                           hdim);
    ResidualLnKernel<<<static_cast<unsigned>(n), kBlock, 0, st.stream>>>(
        st.normed, st.ffn_out, layer.out_norm_w, layer.out_norm_b,
        config.layer_norm_eps, st.hidden, nn, hdim);
  }

  if (hipStreamSynchronize(st.stream) != hipSuccess) {
    return fail("stream sync");
  }
  const double prefill_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count();
  // CLS row (row 0) to host; pooling + L2 stay on the shared core path.
  std::vector<float> cls(h);
  if (hipMemcpy(cls.data(), st.hidden, h * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return fail("cls download");
  }
  EngineSession::SyncResult result;
  result.values = std::move(cls);
  result.prefill_ms = prefill_ms;
  {
    // L2Normalize reports the pre-norm; a zero vector is refused, never
    // returned silently (same contract as the CPU Sync).
    double norm_sq = 0.0;
    for (const float x : result.values) {
      norm_sq += static_cast<double>(x) * static_cast<double>(x);
    }
    result.pre_norm = std::sqrt(norm_sq);
    if (!(result.pre_norm > 0.0)) {
      return refuse("bge-m3 HIP Sync: zero-norm embedding refused");
    }
    const float inv = static_cast<float>(1.0 / result.pre_norm);
    for (float& x : result.values) x *= inv;
  }
  return result;
}

}  // namespace gufo::models::bge_m3
