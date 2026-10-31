#include "src/models/gemma4/mtp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "src/models/gemma4/kernels/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace gufo::models::gemma4 {
namespace {

#if defined(ENGINE_ENABLE_HIP)

namespace q38 = ::gufo::models::qwen38_flash_next::rocm;

constexpr std::uint32_t kHidden = 1024;
constexpr std::uint32_t kFf = 8192;
constexpr std::uint32_t kLayers = 4;
constexpr std::uint32_t kHeads = 16;
constexpr std::uint32_t kHeadDimSwa = 256;
constexpr std::uint32_t kHeadDimFull = 512;

/// Q4_0 row dot product against an F16 activation row: logits[row] for one
/// row of a [rows][k] Q4_0 matrix. Same algebra as the engine's GemvQKernel.
__global__ void MtpQ4GemvKernel(const void* __restrict__ w,
                               std::uint64_t row_bytes, const __half* x,
                               int32_t* __restrict__ out_argmax, std::uint32_t k,
                               std::uint32_t rows) {
  const std::uint32_t row = blockIdx.x;
  const auto* base = static_cast<const std::uint8_t*>(w) +
                     static_cast<std::size_t>(row) * row_bytes;
  float sum = 0.0F;
  const std::uint32_t nblk = k / 32;
  for (std::uint32_t b = threadIdx.x; b < nblk; b += blockDim.x) {
    const auto* blk = base + static_cast<std::size_t>(b) * 18;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const std::uint8_t* qs = blk + 2;
    const __half* xv = x + static_cast<std::size_t>(b) * 32;
#pragma unroll
    for (int q = 0; q < 16; ++q) {
      const float lo = d * (static_cast<float>(qs[q] & 0xF) - 8.0F);
      const float hi = d * (static_cast<float>(qs[q] >> 4) - 8.0F);
      sum += lo * __half2float(xv[q]) + hi * __half2float(xv[q + 16]);
    }
  }
  __shared__ float red[32];
  for (std::uint32_t off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffffull, sum, off);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0F;
    for (std::uint32_t off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffull, v, off);
    if (threadIdx.x == 0) {
      // One argmax slot per row; the host reduces over rows.
      out_argmax[row] = __float_as_int(v);
    }
  }
  (void)rows;
}

/// Host-side Q4_0 -> F16 dequant. Weights are (rows x cols) row-major Q4_0.
std::vector<__half> DequantQ40F16(const std::uint8_t* src, std::uint64_t rows,
                                  std::uint64_t cols) {
  std::vector<__half> dst(static_cast<std::size_t>(rows * cols));
  const std::uint64_t blocks = cols / 32;
  for (std::uint64_t r = 0; r < rows; ++r) {
    const std::uint8_t* row = src + r * blocks * 18;
    __half* out = dst.data() + r * cols;
    for (std::uint64_t b = 0; b < blocks; ++b) {
      const std::uint8_t* blk = row + b * 18;
      const float d = [&] {
        const std::uint16_t h = static_cast<std::uint16_t>(blk[0]) |
                                (static_cast<std::uint16_t>(blk[1]) << 8);
        const std::uint32_t sign = (h >> 15) & 1u;
        const std::uint32_t exp = (h >> 10) & 0x1Fu;
        const std::uint32_t man = h & 0x3FFu;
        float v;
        if (exp == 0) {
          v = std::ldexp(static_cast<float>(man), -24);
        } else if (exp == 31) {
          v = man == 0 ? INFINITY : NAN;
        } else {
          v = std::ldexp(static_cast<float>(man | 0x400u),
                         static_cast<int>(exp) - 25);
        }
        return sign != 0 ? -v : v;
      }();
      const std::uint8_t* qs = blk + 2;
      __half* o = out + b * 32;
      for (int j = 0; j < 16; ++j) {
        o[j] = __float2half_rn(d * (static_cast<float>(qs[j] & 0xF) - 8.0F));
        o[j + 16] =
            __float2half_rn(d * (static_cast<float>(qs[j] >> 4) - 8.0F));
      }
    }
  }
  return dst;
}

#endif  // ENGINE_ENABLE_HIP

}  // namespace

struct MtpDrafter::Impl {
#if defined(ENGINE_ENABLE_HIP)
  std::uint32_t vocab{0};
  std::uint32_t backbone{0};
  int swa_layer{-1};
  int full_layer{-1};

  struct L {
    const float* attn_norm{nullptr};
    const float* post_attn_norm{nullptr};
    const float* post_ffw_norm{nullptr};
    const __half* wq{nullptr};
    const float* wq_norm{nullptr};
    const __half* wo{nullptr};
    const float* ffn_norm{nullptr};
    const __half* w_gate{nullptr};
    const __half* w_up{nullptr};
    const __half* w_down{nullptr};
    const float* lscale{nullptr};
    std::uint32_t hd{0};
  } layer[kLayers];
  const __half* pre{nullptr};
  const __half* post{nullptr};
  const float* out_norm{nullptr};
  const void* tok_embd{nullptr};   // Q4_0 [vocab][1024], device
  std::uint64_t tok_embd_row_bytes{0};

  std::vector<void*> allocs;
  hipStream_t stream{nullptr};

  // scratch (device)
  float* d_pre_in{nullptr};      // [2*backbone] F32
  __half* d_pre_in_h{nullptr};   // [2*backbone] F16
  float* d_g{nullptr};           // [hidden] pre_projection output (inp_g)
  float* d_h{nullptr};           // [hidden]
  float* d_norm{nullptr};        // [hidden]
  __half* d_norm_h{nullptr};     // [hidden]
  float* d_gnorm{nullptr};       // [hidden]
  __half* d_gnorm_h{nullptr};    // [hidden]
  __half* d_cat_h{nullptr};      // [2*hidden]
  __half* d_q_h{nullptr};        // [heads*512] generic f16 staging
  float* d_q{nullptr};           // [heads*512]
  float* d_attn{nullptr};        // [heads*512]
  __half* d_attn_h{nullptr};     // [heads*512]
  float* d_proj{nullptr};        // [hidden]
  __half* d_ffn_h{nullptr};      // [ff]
  float* d_ffn{nullptr};         // [ff]
  float* d_logits_arg{nullptr};  // [vocab] float-as-int
  int32_t* d_argmax{nullptr};

  void Alloc(std::uint32_t backbone_hidden) {
    const auto up = [&](auto*& p, std::size_t bytes) {
      void* raw = nullptr;
      if (hipMalloc(&raw, bytes) != hipSuccess) {
        throw std::runtime_error("MTP allocation failed");
      }
      allocs.push_back(raw);
      p = static_cast<std::decay_t<decltype(p)>>(raw);
    };
    up(d_pre_in, 2 * backbone_hidden * sizeof(float));
    up(d_pre_in_h, 2 * backbone_hidden * sizeof(__half));
    up(d_g, kHidden * sizeof(float));
    up(d_h, kHidden * sizeof(float));
    up(d_norm, kHidden * sizeof(float));
    up(d_norm_h, kHidden * sizeof(__half));
    up(d_gnorm, kHidden * sizeof(float));
    up(d_gnorm_h, kHidden * sizeof(__half));
    up(d_cat_h, 2 * kHidden * sizeof(__half));
    up(d_q_h, kHeads * 512 * sizeof(__half));
    up(d_q, kHeads * 512 * sizeof(float));
    up(d_attn, kHeads * 512 * sizeof(float));
    up(d_attn_h, kHeads * 512 * sizeof(__half));
    up(d_proj, kHidden * sizeof(float));
    up(d_ffn_h, kFf * sizeof(__half));
    up(d_ffn, 2 * kFf * sizeof(float));  // gate and up live side by side
    up(d_logits_arg, vocab * sizeof(float));
    up(d_argmax, sizeof(int32_t));
    if (hipStreamCreate(&stream) != hipSuccess) {
      throw std::runtime_error("MTP stream creation failed");
    }
  }

  [[nodiscard]] const __half* Upload(const std::uint8_t* src, std::uint64_t rows,
                                     std::uint64_t cols) {
    auto f16 = DequantQ40F16(src, rows, cols);
    void* dev = nullptr;
    const std::size_t bytes = f16.size() * sizeof(__half);
    if (hipMalloc(&dev, bytes) != hipSuccess ||
        hipMemcpy(dev, f16.data(), bytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
      throw std::runtime_error("MTP weight upload failed");
    }
    allocs.push_back(dev);
    return static_cast<const __half*>(dev);
  }
#endif
};

MtpDrafter::MtpDrafter() : impl_(std::make_unique<Impl>()) {}

MtpDrafter::~MtpDrafter() {
#if defined(ENGINE_ENABLE_HIP)
  if (impl_) {
    if (impl_->stream != nullptr) {
      (void)hipStreamSynchronize(impl_->stream);
      (void)hipStreamDestroy(impl_->stream);
    }
    for (void* p : impl_->allocs) {
      (void)hipFree(p);
    }
  }
#endif
}

std::unique_ptr<MtpDrafter> MtpDrafter::Load(const std::string& path,
                                             std::uint32_t backbone_hidden,
                                             std::string* error_msg) {
#if defined(ENGINE_ENABLE_HIP)
  try {
    std::string err;
    auto reader = core::GgufReader::OpenFile(path, &err);
    if (reader == nullptr) {
      if (error_msg != nullptr) *error_msg = "MTP open failed: " + err;
      return nullptr;
    }
    auto drafter = std::unique_ptr<MtpDrafter>(new MtpDrafter());
    auto& im = *drafter->impl_;
    im.backbone = backbone_hidden;

    const auto need = [&](const std::string& name, std::uint64_t cols,
                          std::uint64_t rows) -> const std::uint8_t* {
      const auto* t = reader->FindTensor(name);
      if (t == nullptr) {
        throw std::runtime_error("MTP tensor missing: " + name);
      }
      if (t->dimensions.size() < 2 || t->dimensions[0] != cols ||
          t->dimensions[1] != rows) {
        throw std::runtime_error("MTP tensor " + name + " has unexpected dims");
      }
      if (t->type != core::GgmlType::kQ4_0 && t->type != core::GgmlType::kF32 &&
          t->type != core::GgmlType::kF16) {
        throw std::runtime_error("MTP tensor " + name + " unsupported type");
      }
      return static_cast<const std::uint8_t*>(t->data);
    };
    const auto is_f32 = [&](const std::string& name) {
      const auto* t = reader->FindTensor(name);
      return t != nullptr && t->type == core::GgmlType::kF32;
    };
    const auto load_vec = [&](const std::string& name, std::uint64_t n)
        -> const float* {
      const auto* t = reader->FindTensor(name);
      if (t == nullptr) throw std::runtime_error("MTP tensor missing: " + name);
      std::vector<float> f(n);
      if (t->type == core::GgmlType::kF32) {
        const float* src = static_cast<const float*>(t->data);
        for (std::uint64_t i = 0; i < n; ++i) f[i] = src[i];
      } else if (t->type == core::GgmlType::kF16) {
        const __half* src = static_cast<const __half*>(t->data);
        for (std::uint64_t i = 0; i < n; ++i) f[i] = __half2float(src[i]);
      } else {
        return nullptr;
      }
      void* dev = nullptr;
      if (hipMalloc(&dev, n * sizeof(float)) != hipSuccess) {
        throw std::runtime_error("MTP alloc failed");
      }
      (void)hipMemcpy(dev, f.data(), n * sizeof(float), hipMemcpyHostToDevice);
      im.allocs.push_back(dev);
      return static_cast<const float*>(dev);
    };
    const auto load_mat = [&](const std::string& name, std::uint64_t cols,
                              std::uint64_t rows) -> const __half* {
      const auto* t = reader->FindTensor(name);
      if (t == nullptr) throw std::runtime_error("MTP tensor missing: " + name);
      if (t->dimensions.size() < 2 || t->dimensions[0] != cols ||
          t->dimensions[1] != rows) {
        throw std::runtime_error("MTP tensor " + name + " dims mismatch");
      }
      if (t->type == core::GgmlType::kF32 || t->type == core::GgmlType::kF16) {
        const bool f32 = t->type == core::GgmlType::kF32;
        std::vector<__half> h(static_cast<std::size_t>(rows * cols));
        for (std::size_t i = 0; i < h.size(); ++i) {
          h[i] = f32 ? __float2half_rn(static_cast<const float*>(t->data)[i])
                     : static_cast<const __half*>(t->data)[i];
        }
        void* dev = nullptr;
        if (hipMalloc(&dev, h.size() * sizeof(__half)) != hipSuccess) {
          throw std::runtime_error("MTP alloc failed");
        }
        (void)hipMemcpy(dev, h.data(), h.size() * sizeof(__half),
                        hipMemcpyHostToDevice);
        im.allocs.push_back(dev);
        return static_cast<const __half*>(dev);
      }
      return im.Upload(static_cast<const std::uint8_t*>(t->data), rows, cols);
    };

    // Vocab from the draft embedding table (rows = vocab, cols = kHidden).
    const auto* emb = reader->FindTensor("token_embd.weight");
    if (emb == nullptr || emb->dimensions.size() < 2 ||
        emb->dimensions[0] != kHidden) {
      throw std::runtime_error("MTP token_embd has unexpected shape");
    }
    im.vocab = static_cast<std::uint32_t>(emb->dimensions[1]);
    im.tok_embd_row_bytes = (kHidden / 32) * 18;
    {
      // The GGUF reader's mapping goes away when Load returns, so the tied
      // embedding/LM-head rows must live in device memory.
      const std::size_t bytes =
          static_cast<std::size_t>(im.vocab) * im.tok_embd_row_bytes;
      void* dev = nullptr;
      if (hipMalloc(&dev, bytes) != hipSuccess ||
          hipMemcpy(dev, emb->data, bytes, hipMemcpyHostToDevice) !=
              hipSuccess) {
        throw std::runtime_error("MTP token_embd upload failed");
      }
      im.allocs.push_back(dev);
      im.tok_embd = dev;
    }

    im.pre = load_mat("nextn.pre_projection.weight", 2 * backbone_hidden,
                      kHidden);
    im.post = load_mat("nextn.post_projection.weight", kHidden, backbone_hidden);
    im.out_norm = load_vec("output_norm.weight", kHidden);
    if (im.out_norm == nullptr) {
      throw std::runtime_error("MTP output_norm must be F32");
    }
    for (std::uint32_t il = 0; il < kLayers; ++il) {
      const auto p = "blk." + std::to_string(il) + ".";
      auto& L = im.layer[il];
      const auto* q = reader->FindTensor(p + "attn_q.weight");
      if (q == nullptr || q->dimensions.size() < 2 ||
          q->dimensions[0] != kHidden) {
        throw std::runtime_error("MTP attn_q shape unexpected at layer " +
                                 std::to_string(il));
      }
      L.hd = static_cast<std::uint32_t>(q->dimensions[1] / kHeads);
      if (L.hd != kHeadDimSwa && L.hd != kHeadDimFull) {
        throw std::runtime_error("MTP attn_q head dim unexpected");
      }
      L.attn_norm = load_vec(p + "attn_norm.weight", kHidden);
      L.post_attn_norm = load_vec(p + "post_attention_norm.weight", kHidden);
      L.post_ffw_norm = load_vec(p + "post_ffw_norm.weight", kHidden);
      L.wq = load_mat(p + "attn_q.weight", kHidden, kHeads * L.hd);
      L.wq_norm = load_vec(p + "attn_q_norm.weight", L.hd);
      L.wo = load_mat(p + "attn_output.weight", kHeads * L.hd, kHidden);
      L.ffn_norm = load_vec(p + "ffn_norm.weight", kHidden);
      L.w_gate = load_mat(p + "ffn_gate.weight", kHidden, kFf);
      L.w_up = load_mat(p + "ffn_up.weight", kHidden, kFf);
      L.w_down = load_mat(p + "ffn_down.weight", kFf, kHidden);
      L.lscale = load_vec(p + "layer_output_scale.weight", 1);
      if (L.attn_norm == nullptr || L.post_attn_norm == nullptr ||
          L.post_ffw_norm == nullptr || L.wq_norm == nullptr ||
          L.ffn_norm == nullptr || L.lscale == nullptr) {
        throw std::runtime_error("MTP norm tensors must be F32");
      }
    }
    im.Alloc(backbone_hidden);
    (void)is_f32;
    return drafter;
  } catch (const std::exception& e) {
    if (error_msg != nullptr) *error_msg = e.what();
    return nullptr;
  }
#else
  (void)path;
  (void)backbone_hidden;
  if (error_msg != nullptr) *error_msg = "MTP requires the HIP build";
  return nullptr;
#endif
}

std::uint32_t MtpDrafter::VocabSize() const noexcept { return impl_->vocab; }
std::uint32_t MtpDrafter::HiddenSize() const noexcept { return kHidden; }
std::uint32_t MtpDrafter::BackboneHidden() const noexcept {
  return impl_->backbone;
}
int MtpDrafter::SwaLayer() const noexcept { return impl_->swa_layer; }
int MtpDrafter::FullLayer() const noexcept { return impl_->full_layer; }

bool MtpDrafter::Step(int32_t token, const float* tgt_emb,
                      const float* tgt_hidden, const void* swa_k,
                      const void* swa_v, std::uint32_t swa_stride,
                      std::uint32_t swa_kv_heads, const void* full_k,
                      const void* full_v, std::uint32_t full_stride,
                      std::uint32_t full_kv_heads, std::uint32_t pos, float rope_theta_swa, float rope_theta_full, std::uint32_t window_swa,
                      float* feedback_hidden, int32_t* out_token,
                      std::string* error_msg) {
#if defined(ENGINE_ENABLE_HIP)
  (void)token;
  try {
    auto& im = *impl_;
    hipStream_t s = im.stream;
    const std::uint32_t bb = im.backbone;
    // pre_projection input: concat(target embedding, target hidden).
    if (hipMemcpyAsync(im.d_pre_in, tgt_emb, bb * sizeof(float),
                       hipMemcpyDeviceToDevice, s) != hipSuccess ||
        hipMemcpyAsync(im.d_pre_in + bb, tgt_hidden, bb * sizeof(float),
                       hipMemcpyDeviceToDevice, s) != hipSuccess) {
      throw std::runtime_error("MTP input copy failed");
    }
    kernels::F32ToF16(im.d_pre_in, im.d_pre_in_h, 2 * bb, s);
    q38::UnquantizedF16Gemm(im.pre, im.d_pre_in_h, im.d_g, 1, kHidden, 2 * bb,
                            s);
    if (hipMemcpyAsync(im.d_h, im.d_g, kHidden * sizeof(float),
                       hipMemcpyDeviceToDevice, s) != hipSuccess) {
      throw std::runtime_error("MTP residual copy failed");
    }
    for (std::uint32_t il = 0; il < kLayers; ++il) {
      const auto& L = im.layer[il];
      const __half* k = static_cast<const __half*>(L.hd == kHeadDimSwa ? swa_k : full_k);
      const __half* v = static_cast<const __half*>(L.hd == kHeadDimSwa ? swa_v : full_v);
      const std::uint32_t stride = L.hd == kHeadDimSwa ? swa_stride : full_stride;
      const std::uint32_t kvh = L.hd == kHeadDimSwa ? swa_kv_heads : full_kv_heads;
      if (k == nullptr || v == nullptr || kvh == 0) {
        throw std::runtime_error("MTP cross-attention KV missing");
      }
      // Standard Gemma 4 block on 1024-wide draft hidden; Q only (the draft
      // has no K/V projections and reuses the target's KV cache), and no
      // RoPE because the draft position is constant (identity rotation).
      kernels::RmsNorm(im.d_h, L.attn_norm, im.d_norm, 1, kHidden, 1e-6F, s);
      kernels::F32ToF16(im.d_norm, im.d_norm_h, kHidden, s);
      q38::UnquantizedF16Gemm(L.wq, im.d_norm_h, im.d_q, 1, kHeads * L.hd,
                              kHidden, s);
      kernels::RmsNorm(im.d_q, L.wq_norm, im.d_q, kHeads, L.hd, 1e-6F, s);
      // NOTE: llama.cpp rotates the draft query with a constant position; with
      // the engine's causal Attention() the identity rotation is the closest
      // match measured so far. The remaining gap is the ISWA bidirectional
      // mask, which the engine kernel does not implement yet.
      (void)rope_theta_swa;
      (void)rope_theta_full;
      (void)window_swa;
      kernels::Attention(im.d_q, k, v, im.d_attn, 1, pos, kHeads, kvh, L.hd,
                         stride, 0, s);
      kernels::F32ToF16(im.d_attn, im.d_attn_h, kHeads * L.hd, s);
      q38::UnquantizedF16Gemm(L.wo, im.d_attn_h, im.d_proj, 1, kHidden,
                              kHeads * L.hd, s);
      kernels::RmsNorm(im.d_proj, L.post_attn_norm, im.d_norm, 1, kHidden,
                       1e-6F, s);
      kernels::AddInplace(im.d_h, im.d_norm, kHidden, s);
      kernels::RmsNorm(im.d_h, L.ffn_norm, im.d_norm, 1, kHidden, 1e-6F, s);
      kernels::F32ToF16(im.d_norm, im.d_norm_h, kHidden, s);
      q38::UnquantizedF16Gemm(L.w_gate, im.d_norm_h, im.d_ffn, 1, kFf, kHidden,
                              s);
      kernels::F32ToF16(im.d_ffn, im.d_ffn_h, kFf, s);
      q38::UnquantizedF16Gemm(L.w_up, im.d_norm_h, im.d_ffn + kFf, 1, kFf,
                              kHidden, s);
      kernels::GeluTanhSwiglu(im.d_ffn, im.d_ffn + kFf, kFf, s);
      kernels::F32ToF16(im.d_ffn, im.d_ffn_h, kFf, s);
      q38::UnquantizedF16Gemm(L.w_down, im.d_ffn_h, im.d_proj, 1, kHidden, kFf,
                              s);
      kernels::RmsNorm(im.d_proj, L.post_ffw_norm, im.d_norm, 1, kHidden,
                       1e-6F, s);
      kernels::AddInplace(im.d_h, im.d_norm, kHidden, s);
      float lscale = 1.0F;
      kernels::DownloadMirror(&lscale, L.lscale, sizeof(float));
      kernels::LayerScale(im.d_h, lscale, kHidden, s);
    }
    // post_projection -> feedback hidden (backbone width).
    kernels::F32ToF16(im.d_h, im.d_norm_h, kHidden, s);
    q38::UnquantizedF16Gemm(im.post, im.d_norm_h, feedback_hidden, 1, bb,
                            kHidden, s);
    // Tied LM head: draft output_norm then tok_embd rows.
    kernels::RmsNorm(im.d_h, im.out_norm, im.d_norm, 1, kHidden, 1e-6F, s);
    kernels::F32ToF16(im.d_norm, im.d_norm_h, kHidden, s);
    hipLaunchKernelGGL(MtpQ4GemvKernel, dim3(im.vocab), dim3(256), 0, s,
                       im.tok_embd, im.tok_embd_row_bytes, im.d_norm_h,
                       reinterpret_cast<int32_t*>(im.d_logits_arg), kHidden,
                       im.vocab);
    if (hipStreamSynchronize(s) != hipSuccess) {
      throw std::runtime_error("MTP stream failed");
    }
    std::vector<float> host(im.vocab);
    if (hipMemcpy(host.data(), im.d_logits_arg, im.vocab * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      throw std::runtime_error("MTP logits download failed");
    }
    std::uint32_t best = 0;
    float best_v = -INFINITY;
    for (std::uint32_t i = 0; i < im.vocab; ++i) {
      if (host[i] > best_v) {
        best_v = host[i];
        best = i;
      }
    }
    *out_token = static_cast<int32_t>(best);
    return true;
  } catch (const std::exception& e) {
    if (error_msg != nullptr) *error_msg = e.what();
    return false;
  }
#else
  (void)token; (void)tgt_emb; (void)tgt_hidden; (void)swa_k; (void)swa_v;
  (void)swa_stride; (void)swa_kv_heads; (void)full_k; (void)full_v;
  (void)full_stride; (void)full_kv_heads; (void)pos; (void)feedback_hidden;
  (void)out_token;
  if (error_msg != nullptr) *error_msg = "MTP requires the HIP build";
  return false;
#endif
}

}  // namespace gufo::models::gemma4
