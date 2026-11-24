#include "src/models/inkling/vision.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "src/core/compute/gemm/hipblaslt_runner.hpp"
#include "src/core/gguf_reader.hpp"

namespace mbufo::models::inkling {
namespace {

inline bool CheckHip(hipError_t err, std::string* msg, const char* what) {
  if (err == hipSuccess) return true;
  if (msg != nullptr) {
    *msg = std::string("inkling-vision: ") + what + ": " +
           hipGetErrorString(err);
  }
  return false;
}

__global__ void PatchEmbedPrependClsKernel(
    const float* __restrict__ img, const float* __restrict__ patch_w,
    const float* __restrict__ cls_emb, const float* __restrict__ cls_pos,
    const float* __restrict__ pos_emb, float* __restrict__ out) {
  // blockIdx.x = token (0..1024), threadIdx.x = channel (0..255, stride 256 -> 1024)
  const std::uint32_t tok = blockIdx.x;
  if (tok == 0) {
    for (std::uint32_t c = threadIdx.x; c < VisionEncoder::kHidden;
         c += blockDim.x) {
      out[c] = cls_emb[c] + cls_pos[c];
    }
    return;
  }
  const std::uint32_t p = tok - 1U;
  const std::uint32_t py = p / VisionEncoder::kGridSide;
  const std::uint32_t px = p % VisionEncoder::kGridSide;
  const std::uint32_t y0 = py * VisionEncoder::kPatchSize;
  const std::uint32_t x0 = px * VisionEncoder::kPatchSize;
  for (std::uint32_t c = threadIdx.x; c < VisionEncoder::kHidden;
       c += blockDim.x) {
    float acc = pos_emb[static_cast<std::size_t>(p) * VisionEncoder::kHidden + c];
    // patch_w GGUF dims=(20, 20, 3, 1024) -> [c][ch][ky][kx] or [ky][kx][ch][c]?
    // In GGML conv2d: dims=(KW=20, KH=20, IC=3, OC=1024), stride order: kx + 20*(ky + 20*(ch + 3*c))
    const float* wc =
        patch_w + static_cast<std::size_t>(c) * (3U * 20U * 20U);
    for (std::uint32_t ch = 0; ch < 3U; ++ch) {
      const float* wch = wc + ch * (20U * 20U);
      for (std::uint32_t ky = 0; ky < 20U; ++ky) {
        const float* img_row =
            img + (static_cast<std::size_t>(y0 + ky) * VisionEncoder::kImageSize +
                   x0) *
                      3U;
        for (std::uint32_t kx = 0; kx < 20U; ++kx) {
          acc += img_row[kx * 3U + ch] * wch[ky * 20U + kx];
        }
      }
    }
    out[static_cast<std::size_t>(tok) * VisionEncoder::kHidden + c] = acc;
  }
}

template <typename OutT>
__global__ void LayerNormKernel(const float* __restrict__ x,
                                const float* __restrict__ w,
                                const float* __restrict__ b,
                                OutT* __restrict__ out,
                                std::uint32_t dim, float eps) {
  const std::uint32_t row = blockIdx.x;
  const float* xr = x + static_cast<std::size_t>(row) * dim;
  OutT* yr = out + static_cast<std::size_t>(row) * dim;
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) sum += xr[i];
  for (int off = 16; off > 0; off /= 2) sum += __shfl_xor(sum, off);
  __shared__ float sm[8];
  const std::uint32_t wave = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31U;
  if (lane == 0U) sm[wave] = sum;
  __syncthreads();
  if (threadIdx.x == 0U) {
    float t = 0.0F;
    for (int i = 0; i < 8; ++i) t += sm[i];
    sm[0] = t / static_cast<float>(dim);
  }
  __syncthreads();
  const float mean = sm[0];
  float var = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = xr[i] - mean;
    var += d * d;
  }
  for (int off = 16; off > 0; off /= 2) var += __shfl_xor(var, off);
  if (lane == 0U) sm[wave] = var;
  __syncthreads();
  if (threadIdx.x == 0U) {
    float t = 0.0F;
    for (int i = 0; i < 8; ++i) t += sm[i];
    sm[0] = rsqrtf(t / static_cast<float>(dim) + eps);
  }
  __syncthreads();
  const float inv = sm[0];
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = (xr[i] - mean) * inv * w[i] + b[i];
    if constexpr (std::is_same_v<OutT, __hip_bfloat16>) {
      yr[i] = __float2bfloat16(v);
    } else {
      yr[i] = v;
    }
  }
}

__global__ void AddBiasKernel(float* __restrict__ x,
                              const float* __restrict__ b, std::uint32_t rows,
                              std::uint32_t dim) {
  const std::uint32_t r = blockIdx.x;
  const std::uint32_t c = blockIdx.y * blockDim.x + threadIdx.x;
  if (r < rows && c < dim) x[static_cast<std::size_t>(r) * dim + c] += b[c];
}

__global__ void ViTAttnKernel(const float* __restrict__ q,
                              const float* __restrict__ k,
                              const float* __restrict__ v,
                              __hip_bfloat16* __restrict__ out,
                              std::uint32_t seq_len, std::uint32_t heads,
                              std::uint32_t head_dim, float scale) {
  // grid = (seq_len, heads), block = 64 (1 thread per head_dim element)
  const std::uint32_t qt = blockIdx.x;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  const std::uint32_t dim = heads * head_dim;
  const float qv = q[static_cast<std::size_t>(qt) * dim + h * head_dim + d];

  float m = -1e30F;
  float l = 0.0F;
  float acc = 0.0F;
  for (std::uint32_t kt = 0; kt < seq_len; ++kt) {
    const std::size_t koff =
        static_cast<std::size_t>(kt) * dim + h * head_dim + d;
    float dot = qv * k[koff];
    // reduce across 64 threads (2 waves)
    for (int off = 16; off > 0; off /= 2) dot += __shfl_xor(dot, off);
    __shared__ float s_dot[2];
    if ((d & 31U) == 0U) s_dot[d >> 5] = dot;
    __syncthreads();
    const float sc = (s_dot[0] + s_dot[1]) * scale;
    const float m_new = fmaxf(m, sc);
    const float alpha = __expf(m - m_new);
    const float w = __expf(sc - m_new);
    acc = acc * alpha + w * v[koff];
    l = l * alpha + w;
    m = m_new;
    __syncthreads();
  }
  out[static_cast<std::size_t>(qt) * dim + h * head_dim + d] =
      __float2bfloat16(acc / l);
}

__global__ void LayerScaleAddKernel(float* __restrict__ x,
                                    const float* __restrict__ y,
                                    const float* __restrict__ ls,
                                    std::uint32_t rows, std::uint32_t dim) {
  const std::uint32_t r = blockIdx.x;
  const std::uint32_t c = blockIdx.y * blockDim.x + threadIdx.x;
  if (r < rows && c < dim) {
    const std::size_t off = static_cast<std::size_t>(r) * dim + c;
    x[off] += y[off] * ls[c];
  }
}

__global__ void GeluF32ToBf16Kernel(const float* __restrict__ x,
                                    __hip_bfloat16* __restrict__ out,
                                    std::size_t total) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < total) {
    const float v = x[i];
    const float g = 0.5F * v * (1.0F + erff(v * 0.7071067811865475F));
    out[i] = __float2bfloat16(g);
  }
}

__global__ void SpatialMerge2x2ToBf16Kernel(const float* __restrict__ x,
                                            __hip_bfloat16* __restrict__ out) {
  // x: [1025, 1024], drop token 0 -> [32, 32, 1024] -> [16, 16, 4096]
  const std::uint32_t m_idx = blockIdx.x;  // 0..255
  const std::uint32_t my = m_idx / VisionEncoder::kMergeSide;
  const std::uint32_t mx = m_idx % VisionEncoder::kMergeSide;
  for (std::uint32_t c = threadIdx.x; c < VisionEncoder::kMergeDim;
       c += blockDim.x) {
    const std::uint32_t sub = c / VisionEncoder::kHidden;  // 0..3
    const std::uint32_t ch = c % VisionEncoder::kHidden;
    const std::uint32_t sy = my * 2U + (sub >> 1);
    const std::uint32_t sx = mx * 2U + (sub & 1U);
    const std::uint32_t src_tok = 1U + sy * VisionEncoder::kGridSide + sx;
    out[static_cast<std::size_t>(m_idx) * VisionEncoder::kMergeDim + c] =
        __float2bfloat16(
            x[static_cast<std::size_t>(src_tok) * VisionEncoder::kHidden + ch]);
  }
}

}  // namespace

struct VisionEncoder::Impl {
  struct Block {
    const float* ln1_w{nullptr};
    const float* ln1_b{nullptr};
    const float* ln2_w{nullptr};
    const float* ln2_b{nullptr};
    const float* ls1_w{nullptr};
    const float* ls2_w{nullptr};
    const void* attn_q_w{nullptr};
    const float* attn_q_b{nullptr};
    const void* attn_k_w{nullptr};
    const float* attn_k_b{nullptr};
    const void* attn_v_w{nullptr};
    const float* attn_v_b{nullptr};
    const void* attn_out_w{nullptr};
    const float* attn_out_b{nullptr};
    const void* ffn_up_w{nullptr};
    const float* ffn_up_b{nullptr};
    const void* ffn_down_w{nullptr};
    const float* ffn_down_b{nullptr};
  };

  std::unique_ptr<core::GgufReader> reader;
  hipStream_t stream{nullptr};
  std::unique_ptr<core::compute::gemm::HipBlasLtRunner> blaslt;
  std::vector<void*> device_allocs;
  std::mutex mu;

  const float* patch_w{nullptr};
  const float* cls_emb{nullptr};
  const float* cls_pos{nullptr};
  const float* pos_emb{nullptr};
  const float* pre_ln_w{nullptr};
  const float* pre_ln_b{nullptr};
  const void* mm1_w{nullptr};
  const void* mm2_w{nullptr};
  std::vector<Block> blocks;
  float ln_eps{1e-5F};

  // Scratch buffers
  float* d_img{nullptr};
  float* d_x{nullptr};       // [1025, 1024]
  __hip_bfloat16* d_h16{nullptr};  // [1025, 4096]
  float* d_q{nullptr};       // [1025, 1024]
  float* d_k{nullptr};       // [1025, 1024]
  float* d_v{nullptr};       // [1025, 1024]
  float* d_tmp{nullptr};     // [1025, 4096]
  float* d_out{nullptr};     // [256, proj_dim]

  ~Impl() {
    for (void* p : device_allocs) {
      if (p != nullptr) (void)hipFree(p);
    }
  }
};

VisionEncoder::~VisionEncoder() = default;

std::unique_ptr<VisionEncoder> VisionEncoder::Load(
    const std::string& mmproj_path, std::uint32_t proj_dim,
    hipStream_t stream, std::string* error_msg) {
  auto reader = core::GgufReader::OpenFile(mmproj_path, error_msg);
  if (!reader) return nullptr;
  if (reader->GetMetadataString("clip.vision.projector_type") != "inkling2v") {
    if (error_msg != nullptr) {
      *error_msg = "inkling-vision: unsupported clip.vision.projector_type";
    }
    return nullptr;
  }
  const std::uint32_t file_proj =
      reader->GetMetadataUint32("clip.vision.projection_dim").value_or(0);
  if (file_proj != proj_dim) {
    if (error_msg != nullptr) {
      *error_msg = "inkling-vision: projection_dim mismatch";
    }
    return nullptr;
  }

  auto enc = std::unique_ptr<VisionEncoder>(new VisionEncoder());
  enc->proj_dim_ = proj_dim;
  enc->impl_ = std::make_unique<Impl>();
  auto& im = *enc->impl_;
  im.reader = std::move(reader);
  im.stream = stream;
  im.ln_eps = im.reader->GetMetadataFloat32("clip.vision.attention.layer_norm_epsilon")
                  .value_or(1e-5F);
  im.blaslt = core::compute::gemm::HipBlasLtRunner::Create(stream, error_msg);
  if (!im.blaslt) return nullptr;

  auto upload = [&](const std::string& name, const void** out_ptr,
                    core::GgmlType expect_type) -> bool {
    const auto* t = im.reader->FindTensor(name);
    if (t == nullptr || t->type != expect_type) {
      if (error_msg != nullptr) {
        *error_msg = "inkling-vision: missing/bad tensor " + name;
      }
      return false;
    }
    const std::size_t bytes =
        t->ElementCount() * (expect_type == core::GgmlType::kF32 ? 4ULL : 2ULL);
    void* dev = nullptr;
    if (!CheckHip(hipMalloc(&dev, bytes), error_msg, name.c_str()) ||
        !CheckHip(hipMemcpy(dev, t->data, bytes, hipMemcpyHostToDevice),
                  error_msg, name.c_str())) {
      if (dev != nullptr) (void)hipFree(dev);
      return false;
    }
    im.device_allocs.push_back(dev);
    enc->resident_bytes_ += bytes;
    *out_ptr = dev;
    return true;
  };
  auto upload_f32 = [&](const std::string& n, const float** p) {
    return upload(n, reinterpret_cast<const void**>(p), core::GgmlType::kF32);
  };
  auto upload_bf16 = [&](const std::string& n, const void** p) {
    return upload(n, p, core::GgmlType::kBF16);
  };

  if (!upload_f32("v.patch_embd.weight", &im.patch_w) ||
      !upload_f32("v.class_embd", &im.cls_emb) ||
      !upload_f32("v.class_position_embd.weight", &im.cls_pos) ||
      !upload_f32("v.position_embd.weight", &im.pos_emb) ||
      !upload_f32("v.pre_ln.weight", &im.pre_ln_w) ||
      !upload_f32("v.pre_ln.bias", &im.pre_ln_b) ||
      !upload_bf16("mm.1.weight", &im.mm1_w) ||
      !upload_bf16("mm.2.weight", &im.mm2_w)) {
    return nullptr;
  }

  im.blocks.resize(kBlocks);
  for (std::uint32_t i = 0; i < kBlocks; ++i) {
    const std::string p = "v.blk." + std::to_string(i) + ".";
    auto& b = im.blocks[i];
    if (!upload_f32(p + "ln1.weight", &b.ln1_w) ||
        !upload_f32(p + "ln1.bias", &b.ln1_b) ||
        !upload_f32(p + "ln2.weight", &b.ln2_w) ||
        !upload_f32(p + "ln2.bias", &b.ln2_b) ||
        !upload_f32(p + "ls1.weight", &b.ls1_w) ||
        !upload_f32(p + "ls2.weight", &b.ls2_w) ||
        !upload_bf16(p + "attn_q.weight", &b.attn_q_w) ||
        !upload_f32(p + "attn_q.bias", &b.attn_q_b) ||
        !upload_bf16(p + "attn_k.weight", &b.attn_k_w) ||
        !upload_f32(p + "attn_k.bias", &b.attn_k_b) ||
        !upload_bf16(p + "attn_v.weight", &b.attn_v_w) ||
        !upload_f32(p + "attn_v.bias", &b.attn_v_b) ||
        !upload_bf16(p + "attn_out.weight", &b.attn_out_w) ||
        !upload_f32(p + "attn_out.bias", &b.attn_out_b) ||
        !upload_bf16(p + "ffn_up.weight", &b.ffn_up_w) ||
        !upload_f32(p + "ffn_up.bias", &b.ffn_up_b) ||
        !upload_bf16(p + "ffn_down.weight", &b.ffn_down_w) ||
        !upload_f32(p + "ffn_down.bias", &b.ffn_down_b)) {
      return nullptr;
    }
  }

  auto alloc = [&](auto** ptr, std::size_t bytes, const char* what) -> bool {
    void* dev = nullptr;
    if (!CheckHip(hipMalloc(&dev, bytes), error_msg, what)) return false;
    im.device_allocs.push_back(dev);
    *ptr = reinterpret_cast<std::remove_reference_t<decltype(*ptr)>>(dev);
    return true;
  };
  if (!alloc(&im.d_img, kImageSize * kImageSize * 3ULL * sizeof(float), "img") ||
      !alloc(&im.d_x, kSeqLen * kHidden * sizeof(float), "x") ||
      !alloc(&im.d_h16, kSeqLen * kFfn * sizeof(__hip_bfloat16), "h16") ||
      !alloc(&im.d_q, kSeqLen * kHidden * sizeof(float), "q") ||
      !alloc(&im.d_k, kSeqLen * kHidden * sizeof(float), "k") ||
      !alloc(&im.d_v, kSeqLen * kHidden * sizeof(float), "v") ||
      !alloc(&im.d_tmp, kSeqLen * kFfn * sizeof(float), "tmp") ||
      !alloc(&im.d_out, kMergedTokens * proj_dim * sizeof(float), "out")) {
    return nullptr;
  }
  return enc;
}

bool VisionEncoder::Encode(const core::Image& image,
                           std::vector<float>* out_host,
                           std::string* error_msg) {
  if (image.width == 0 || image.height == 0 ||
      image.pixels.size() !=
          static_cast<std::size_t>(image.width) * image.height * 3U) {
    if (error_msg != nullptr) *error_msg = "inkling-vision: invalid input image";
    return false;
  }
  // Bilinear resize to 640x640 and normalize (p/255 - 0.5)/0.5 = p/127.5 - 1.0
  std::vector<float> host_img(kImageSize * kImageSize * 3ULL);
  const float scale_y = static_cast<float>(image.height) / kImageSize;
  const float scale_x = static_cast<float>(image.width) / kImageSize;
  for (std::uint32_t y = 0; y < kImageSize; ++y) {
    const float sy = std::clamp((y + 0.5F) * scale_y - 0.5F, 0.0F,
                                static_cast<float>(image.height - 1U));
    const std::uint32_t y0 = static_cast<std::uint32_t>(sy);
    const std::uint32_t y1 = std::min(y0 + 1U, image.height - 1U);
    const float fy = sy - y0;
    for (std::uint32_t x = 0; x < kImageSize; ++x) {
      const float sx = std::clamp((x + 0.5F) * scale_x - 0.5F, 0.0F,
                                  static_cast<float>(image.width - 1U));
      const std::uint32_t x0 = static_cast<std::uint32_t>(sx);
      const std::uint32_t x1 = std::min(x0 + 1U, image.width - 1U);
      const float fx = sx - x0;
      for (std::uint32_t c = 0; c < 3U; ++c) {
        const float p00 =
            image.pixels[(static_cast<std::size_t>(y0) * image.width + x0) * 3U + c];
        const float p01 =
            image.pixels[(static_cast<std::size_t>(y0) * image.width + x1) * 3U + c];
        const float p10 =
            image.pixels[(static_cast<std::size_t>(y1) * image.width + x0) * 3U + c];
        const float p11 =
            image.pixels[(static_cast<std::size_t>(y1) * image.width + x1) * 3U + c];
        const float v = (p00 * (1.0F - fx) + p01 * fx) * (1.0F - fy) +
                        (p10 * (1.0F - fx) + p11 * fx) * fy;
        host_img[(static_cast<std::size_t>(y) * kImageSize + x) * 3U + c] =
            v * (1.0F / 127.5F) - 1.0F;
      }
    }
  }

  std::lock_guard<std::mutex> lock(impl_->mu);
  auto& im = *impl_;
  if (!CheckHip(hipMemcpyAsync(im.d_img, host_img.data(),
                               host_img.size() * sizeof(float),
                               hipMemcpyHostToDevice, im.stream),
                error_msg, "copy img")) {
    return false;
  }

  hipLaunchKernelGGL(PatchEmbedPrependClsKernel, dim3(kSeqLen), dim3(256), 0,
                     im.stream, im.d_img, im.patch_w, im.cls_emb, im.cls_pos,
                     im.pos_emb, im.d_x);
  hipLaunchKernelGGL((LayerNormKernel<float>), dim3(kSeqLen), dim3(256), 0,
                     im.stream, im.d_x, im.pre_ln_w, im.pre_ln_b, im.d_x,
                     kHidden, im.ln_eps);

  for (const auto& b : im.blocks) {
    hipLaunchKernelGGL((LayerNormKernel<__hip_bfloat16>), dim3(kSeqLen), dim3(256), 0,
                       im.stream, im.d_x, b.ln1_w, b.ln1_b, im.d_h16, kHidden,
                       im.ln_eps);
    if (!im.blaslt->LibraryGemm(b.attn_q_w, im.d_h16, im.d_q, HIP_R_16BF,
                                kHidden, kSeqLen, kHidden, error_msg) ||
        !im.blaslt->LibraryGemm(b.attn_k_w, im.d_h16, im.d_k, HIP_R_16BF,
                                kHidden, kSeqLen, kHidden, error_msg) ||
        !im.blaslt->LibraryGemm(b.attn_v_w, im.d_h16, im.d_v, HIP_R_16BF,
                                kHidden, kSeqLen, kHidden, error_msg)) {
      return false;
    }
    hipLaunchKernelGGL(AddBiasKernel, dim3(kSeqLen, kHidden / 256U), dim3(256),
                       0, im.stream, im.d_q, b.attn_q_b, kSeqLen, kHidden);
    hipLaunchKernelGGL(AddBiasKernel, dim3(kSeqLen, kHidden / 256U), dim3(256),
                       0, im.stream, im.d_k, b.attn_k_b, kSeqLen, kHidden);
    hipLaunchKernelGGL(AddBiasKernel, dim3(kSeqLen, kHidden / 256U), dim3(256),
                       0, im.stream, im.d_v, b.attn_v_b, kSeqLen, kHidden);
    hipLaunchKernelGGL(ViTAttnKernel, dim3(kSeqLen, kHeads), dim3(kHeadDim), 0,
                       im.stream, im.d_q, im.d_k, im.d_v, im.d_h16, kSeqLen,
                       kHeads, kHeadDim, 0.125F);
    if (!im.blaslt->LibraryGemm(b.attn_out_w, im.d_h16, im.d_tmp, HIP_R_16BF,
                                kHidden, kSeqLen, kHidden, error_msg)) {
      return false;
    }
    hipLaunchKernelGGL(AddBiasKernel, dim3(kSeqLen, kHidden / 256U), dim3(256),
                       0, im.stream, im.d_tmp, b.attn_out_b, kSeqLen, kHidden);
    hipLaunchKernelGGL(LayerScaleAddKernel, dim3(kSeqLen, kHidden / 256U),
                       dim3(256), 0, im.stream, im.d_x, im.d_tmp, b.ls1_w,
                       kSeqLen, kHidden);

    hipLaunchKernelGGL((LayerNormKernel<__hip_bfloat16>), dim3(kSeqLen), dim3(256), 0,
                       im.stream, im.d_x, b.ln2_w, b.ln2_b, im.d_h16, kHidden,
                       im.ln_eps);
    if (!im.blaslt->LibraryGemm(b.ffn_up_w, im.d_h16, im.d_tmp, HIP_R_16BF,
                                kFfn, kSeqLen, kHidden, error_msg)) {
      return false;
    }
    hipLaunchKernelGGL(AddBiasKernel, dim3(kSeqLen, kFfn / 256U), dim3(256), 0,
                       im.stream, im.d_tmp, b.ffn_up_b, kSeqLen, kFfn);
    hipLaunchKernelGGL(GeluF32ToBf16Kernel,
                       dim3((kSeqLen * kFfn + 255U) / 256U), dim3(256), 0,
                       im.stream, im.d_tmp, im.d_h16,
                       static_cast<std::size_t>(kSeqLen) * kFfn);
    if (!im.blaslt->LibraryGemm(b.ffn_down_w, im.d_h16, im.d_tmp, HIP_R_16BF,
                                kHidden, kSeqLen, kFfn, error_msg)) {
      return false;
    }
    hipLaunchKernelGGL(AddBiasKernel, dim3(kSeqLen, kHidden / 256U), dim3(256),
                       0, im.stream, im.d_tmp, b.ffn_down_b, kSeqLen, kHidden);
    hipLaunchKernelGGL(LayerScaleAddKernel, dim3(kSeqLen, kHidden / 256U),
                       dim3(256), 0, im.stream, im.d_x, im.d_tmp, b.ls2_w,
                       kSeqLen, kHidden);
  }

  hipLaunchKernelGGL(SpatialMerge2x2ToBf16Kernel, dim3(kMergedTokens),
                     dim3(256), 0, im.stream, im.d_x, im.d_h16);
  if (!im.blaslt->LibraryGemm(im.mm1_w, im.d_h16, im.d_tmp, HIP_R_16BF,
                              kMergeDim, kMergedTokens, kMergeDim, error_msg)) {
    return false;
  }
  hipLaunchKernelGGL(GeluF32ToBf16Kernel,
                     dim3((kMergedTokens * kMergeDim + 255U) / 256U), dim3(256),
                     0, im.stream, im.d_tmp, im.d_h16,
                     static_cast<std::size_t>(kMergedTokens) * kMergeDim);
  if (!im.blaslt->LibraryGemm(im.mm2_w, im.d_h16, im.d_out, HIP_R_16BF,
                              proj_dim_, kMergedTokens, kMergeDim, error_msg)) {
    return false;
  }

  out_host->resize(static_cast<std::size_t>(kMergedTokens) * proj_dim_);
  if (!CheckHip(hipMemcpyAsync(out_host->data(), im.d_out,
                               out_host->size() * sizeof(float),
                               hipMemcpyDeviceToHost, im.stream),
                error_msg, "copy out") ||
      !CheckHip(hipStreamSynchronize(im.stream), error_msg, "sync vision")) {
    return false;
  }
  return true;
}

}  // namespace mbufo::models::inkling
