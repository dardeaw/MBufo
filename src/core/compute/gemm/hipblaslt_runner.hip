// Shared hipBLASLt plan-cache runner (Phase B-3, staged). See
// hipblaslt_runner.hpp for provenance: verbatim BlasLt algorithm, own-kernel
// fallbacks injected as model hooks.
#if defined(ENGINE_ENABLE_HIP)

#include "src/core/compute/gemm/hipblaslt_runner.hpp"

#include <hipblaslt/hipblaslt-ext.hpp>

namespace gufo::core::compute::gemm {
namespace {

void AssignError(std::string* error, const char* message) {
  if (error != nullptr) {
    *error = message;
  }
}

bool OwnF16(hipDataType type, int m, int k) {
  return type == HIP_R_16F && k == 2560 && (m == 96 || m == 513);
}

bool OwnBf16(hipDataType type, int m, int k) {
  // The indexer query and key projections, and a TP2 rank's halves.
  return type == HIP_R_16BF && k == 2560 &&
         (m == 512 || m == 256 || m == 128 || m == 64);
}

}  // namespace

struct HipBlasLtRunner::Plan {
  hipblasLtMatmulDesc_t operation{nullptr};
  hipblasLtMatrixLayout_t weights{nullptr};
  hipblasLtMatrixLayout_t input{nullptr};
  hipblasLtMatrixLayout_t output{nullptr};
  hipblasLtMatmulAlgo_t algorithm{};

  ~Plan() {
    if (output != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(output);
    if (input != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(input);
    if (weights != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(weights);
    if (operation != nullptr)
      (void)hipblasLtMatmulDescDestroy(operation);
  }
};

HipBlasLtRunner::~HipBlasLtRunner() {
  plans_.clear();
  if (handle_ != nullptr) {
    (void)hipblasLtDestroy(handle_);
  }
}

std::unique_ptr<HipBlasLtRunner> HipBlasLtRunner::Create(
    hipStream_t stream, std::string* error_msg, HipBlasLtOwnHook own_f16,
    HipBlasLtOwnHook own_bf16) {
  std::unique_ptr<HipBlasLtRunner> blas(new HipBlasLtRunner());
  blas->stream_ = stream;
  blas->own_f16_ = own_f16;
  blas->own_bf16_ = own_bf16;
  if (hipblasLtCreate(&blas->handle_) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error_msg, "hipblasLtCreate failed");
    return nullptr;
  }
  return blas;
}

std::unique_ptr<HipBlasLtRunner::Plan> HipBlasLtRunner::MakePlan(
    hipDataType type, int m, int n, int k, std::string* error_msg) const {
  auto p = std::make_unique<Plan>();
  const hipblasOperation_t transpose = HIPBLAS_OP_T;
  const hipblasOperation_t normal = HIPBLAS_OP_N;
  if (hipblasLtMatmulDescCreate(&p->operation, HIPBLAS_COMPUTE_32F,
                                HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(
          p->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose,
          sizeof(transpose)) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(
          p->operation, HIPBLASLT_MATMUL_DESC_TRANSB, &normal,
          sizeof(normal)) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&p->weights, type, k, m, k) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&p->input, type, k, n, k) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&p->output, HIP_R_32F, m, n, m) !=
          HIPBLAS_STATUS_SUCCESS) {
    AssignError(error_msg, "hipBLASLt descriptor creation failed");
    return nullptr;
  }

  const float one = 1.0F;
  const float zero = 0.0F;
  const auto usable = [&](hipblasLtMatmulAlgo_t& algorithm) {
    std::size_t workspace = 0;
    return hipblaslt_ext::matmulIsAlgoSupported(
               handle_, p->operation, &one, p->weights, p->input, &zero,
               p->output, p->output, algorithm,
               workspace) == HIPBLAS_STATUS_SUCCESS &&
           workspace == 0;
  };

  // Initialize the solution library and keep a deterministic fallback for
  // other tensor geometries. No trial GEMMs run on model activations.
  hipblasLtMatmulPreference_t preference = nullptr;
  if (hipblasLtMatmulPreferenceCreate(&preference) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error_msg, "hipBLASLt preference creation failed");
    return nullptr;
  }
  constexpr std::size_t workspace = 0;
  (void)hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
      sizeof(workspace));
  std::array<hipblasLtMatmulHeuristicResult_t, 16> candidates{};
  int count = 0;
  const auto status = hipblasLtMatmulAlgoGetHeuristic(
      handle_, p->operation, p->weights, p->input, p->output, p->output,
      preference, candidates.size(), candidates.data(), &count);
  (void)hipblasLtMatmulPreferenceDestroy(preference);

  if (status == HIPBLAS_STATUS_SUCCESS) {
    for (int i = 0; i < count; ++i) {
      if (candidates[i].state == HIPBLAS_STATUS_SUCCESS &&
          candidates[i].workspaceSize == 0 && usable(candidates[i].algo)) {
        p->algorithm = candidates[i].algo;
        return p;
      }
    }
  }
  AssignError(error_msg, "no workspace-free hipBLASLt kernel for the GEMM shape");
  return nullptr;
}

bool HipBlasLtRunner::Gemm(const void* weights, const void* input, float* out,
                           hipDataType type, int m, int n, int k,
                           std::string* error_msg) {
  if (m <= 0 || n <= 0 || k <= 0) {
    AssignError(error_msg, "hipBLASLt dimensions must be positive");
    return false;
  }
  // Library edge tiles change the accumulation order when the same token
  // moves within a batch. Router and recurrent-gate errors then amplify
  // across layers. Keep one K reduction for every row and prefill chunk
  // size. The model-owned kernel runs here when its hook is wired;
  // otherwise decline so the caller keeps its legacy path.
  if (OwnF16(type, m, k)) {
    if (own_f16_ != nullptr) {
      return own_f16_(weights, input, out, m, n, k, stream_, error_msg);
    }
    AssignError(error_msg, "F16 own-kernel shape needs a model hook");
    return false;
  }
  // The library's MT32x32x64 BF16 kernel starts each 32-token tile's K loop
  // at a block rotated by the tile's index in the batch, so a token's
  // indexer score depended on its position modulo 128 tokens within the
  // prefill chunk, and a top-k near-tie could select another key with the
  // chunking.
  if (OwnBf16(type, m, k)) {
    if (own_bf16_ != nullptr) {
      return own_bf16_(weights, input, out, m, n, k, stream_, error_msg);
    }
    AssignError(error_msg, "BF16 own-kernel shape needs a model hook");
    return false;
  }
  Plan* p = FindPlan(type, m, n, k, error_msg);
  return p != nullptr && Run(*p, weights, input, out, error_msg);
}

bool HipBlasLtRunner::Prepare(hipDataType type, int m, int n, int k,
                              std::string* error_msg) {
  if (m <= 0 || n <= 0 || k <= 0) {
    AssignError(error_msg, "hipBLASLt dimensions must be positive");
    return false;
  }
  if (OwnF16(type, m, k)) {
    if (own_f16_ == nullptr) {
      AssignError(error_msg, "F16 own-kernel shape needs a model hook");
      return false;
    }
    return true;
  }
  if (OwnBf16(type, m, k)) {
    if (own_bf16_ == nullptr) {
      AssignError(error_msg, "BF16 own-kernel shape needs a model hook");
      return false;
    }
    return true;
  }
  return FindPlan(type, m, n, k, error_msg) != nullptr;
}

HipBlasLtRunner::Plan* HipBlasLtRunner::FindPlan(hipDataType type, int m, int n,
                                                 int k, std::string* error_msg) {
  // Exact dimensions prevent the first ragged request from determining
  // which kernel later requests in the same size bucket receive.
  const std::array<int, 4> key{static_cast<int>(type), m, n, k};
  auto& p = plans_[key];
  if (!p)
    p = MakePlan(type, m, n, k, error_msg);
  return p.get();
}

bool HipBlasLtRunner::UsesOwnKernel(hipDataType type, int m, int k) {
  return OwnF16(type, m, k) || OwnBf16(type, m, k);
}

bool HipBlasLtRunner::LibraryGemm(const void* weights, const void* input,
                                  float* out, hipDataType type, int m, int n,
                                  int k, std::string* error_msg) {
  Plan* p = FindPlan(type, m, n, k, error_msg);
  return p != nullptr && Run(*p, weights, input, out, error_msg);
}

bool HipBlasLtRunner::Run(Plan& p, const void* weights, const void* input,
                          float* out, std::string* error_msg) {
  const float one = 1.0F;
  const float zero = 0.0F;
  if (hipblasLtMatmul(handle_, p.operation, &one, weights, p.weights, input,
                      p.input, &zero, out, p.output, out, p.output,
                      &p.algorithm, nullptr, 0,
                      stream_) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error_msg, "hipBLASLt GEMM failed");
    return false;
  }
  return true;
}

}  // namespace gufo::core::compute::gemm

#endif  // defined(ENGINE_ENABLE_HIP)
