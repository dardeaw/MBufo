#include "src/models/gemma4/mtp.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
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

__global__ void ConcatTwoKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ out, std::uint32_t dim) {
  const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim) {
    out[idx] = a[idx];
  } else if (idx < 2u * dim) {
    out[idx] = b[idx - dim];
  }
}

struct MatRef {
  const void* dev{nullptr};
  int type{0};
  std::uint64_t row_bytes{0};
};

#endif  // ENGINE_ENABLE_HIP

}  // namespace

struct MtpDrafter::Impl {
#if defined(ENGINE_ENABLE_HIP)
  std::uint32_t vocab{0};
  std::uint32_t backbone{0};
  float rms_eps{1e-6F};
  int swa_layer{-1};
  int full_layer{-1};

  struct L {
    const float* attn_norm{nullptr};
    const float* post_attn_norm{nullptr};
    const float* post_ffw_norm{nullptr};
    MatRef wq;
    const float* wq_norm{nullptr};
    MatRef wo;
    const float* ffn_norm{nullptr};
    MatRef w_gate;
    MatRef w_up;
    MatRef w_down;
    float lscale_val{1.0F};
    std::uint32_t hd{0};
  } layer[kLayers];
  MatRef pre;
  MatRef post;
  const float* out_norm{nullptr};
  MatRef tok_embd;

  std::vector<void*> allocs;

  // scratch (device)
  float* d_pre_in{nullptr};   // [2*backbone] F32
  float* d_h{nullptr};        // [hidden]
  float* d_norm{nullptr};     // [hidden]
  float* d_q{nullptr};        // [heads*512]
  float* d_attn{nullptr};     // [heads*512]
  float* d_proj{nullptr};     // [hidden]
  float* d_ffn{nullptr};      // [2*ff]
  float* d_logits{nullptr};   // [vocab] F32
  float* d_tgt_hn{nullptr};   // [backbone] post-final-norm target hidden
  q38::ArgmaxCandidate* d_argmax_scratch{nullptr};

  void Alloc(std::uint32_t backbone_hidden) {
    const auto up = [&](auto*& p, std::size_t bytes) {
      void* raw = nullptr;
      if (hipMalloc(&raw, bytes) != hipSuccess) {
        throw std::runtime_error("MTP allocation failed");
      }
      allocs.push_back(raw);
      p = static_cast<std::decay_t<decltype(p)>>(raw);
    };
    up(d_pre_in, 2ull * backbone_hidden * sizeof(float));
    up(d_h, kHidden * sizeof(float));
    up(d_norm, kHidden * sizeof(float));
    up(d_q, kHeads * 512ull * sizeof(float));
    up(d_attn, kHeads * 512ull * sizeof(float));
    up(d_proj, kHidden * sizeof(float));
    up(d_ffn, 2ull * kFf * sizeof(float));
    up(d_logits, static_cast<std::size_t>(vocab) * sizeof(float));
    up(d_tgt_hn, backbone_hidden * sizeof(float));
    up(d_argmax_scratch, q38::kArgmaxParts * sizeof(q38::ArgmaxCandidate));
  }
#endif
};

MtpDrafter::MtpDrafter() : impl_(std::make_unique<Impl>()) {}

MtpDrafter::~MtpDrafter() {
#if defined(ENGINE_ENABLE_HIP)
  if (impl_) {
    (void)hipStreamSynchronize(0);
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
    if (auto eps = reader->GetMetadataFloat32(
            "gemma4-assistant.attention.layer_norm_rms_epsilon")) {
      im.rms_eps = *eps;
    } else if (auto eps2 = reader->GetMetadataFloat32(
                   "gemma4.attention.layer_norm_rms_epsilon")) {
      im.rms_eps = *eps2;
    }

    const auto read_host_f32 = [&](const core::GgufTensorInfo* t,
                                   std::uint64_t n) -> std::vector<float> {
      std::vector<float> f(n, 0.0F);
      if (t->type == core::GgmlType::kF32) {
        std::memcpy(f.data(), t->data, n * sizeof(float));
      } else if (t->type == core::GgmlType::kF16) {
        const auto* src = static_cast<const __half*>(t->data);
        for (std::uint64_t i = 0; i < n; ++i) f[i] = __half2float(src[i]);
      } else if (t->type == core::GgmlType::kBF16) {
        const auto* src = static_cast<const std::uint16_t*>(t->data);
        for (std::uint64_t i = 0; i < n; ++i) {
          f[i] = std::bit_cast<float>(static_cast<std::uint32_t>(src[i]) << 16);
        }
      } else {
        throw std::runtime_error("MTP vector tensor must be F32/F16/BF16");
      }
      return f;
    };

    const auto load_vec = [&](const std::string& name,
                              std::uint64_t n) -> const float* {
      const auto* t = reader->FindTensor(name);
      if (t == nullptr) throw std::runtime_error("MTP tensor missing: " + name);
      auto f = read_host_f32(t, n);
      void* dev = nullptr;
      if (hipMalloc(&dev, n * sizeof(float)) != hipSuccess ||
          hipMemcpy(dev, f.data(), n * sizeof(float), hipMemcpyHostToDevice) !=
              hipSuccess) {
        if (dev != nullptr) (void)hipFree(dev);
        throw std::runtime_error("MTP vec alloc/upload failed: " + name);
      }
      im.allocs.push_back(dev);
      return static_cast<const float*>(dev);
    };

    const auto load_mat = [&](const std::string& name, std::uint64_t cols,
                              std::uint64_t rows) -> MatRef {
      const auto* t = reader->FindTensor(name);
      if (t == nullptr) throw std::runtime_error("MTP tensor missing: " + name);
      if (t->dimensions.size() < 2 || t->dimensions[0] != cols ||
          t->dimensions[1] != rows) {
        throw std::runtime_error("MTP tensor " + name + " dims mismatch");
      }
      TensorRef tr;
      tr.data = t->data;
      tr.type = t->type;
      tr.cols = cols;
      tr.rows = rows;
      const int tp = static_cast<int>(t->type);
      if (tp != 0 && tp != 1 && tp != 2 && tp != 8 && tp != 30) {
        throw std::runtime_error("MTP unsupported quant type on " + name);
      }
      const std::uint64_t rb = tr.RowBytes();
      const std::size_t bytes = static_cast<std::size_t>(rb) * rows;
      void* dev = nullptr;
      if (hipMalloc(&dev, bytes) != hipSuccess) {
        throw std::runtime_error("MTP mat alloc failed: " + name);
      }
      if (std::getenv("GEMMA4_MTP_NODEQUANT") != nullptr) {
        (void)hipMemset(dev, 0, bytes);
      } else if (hipMemcpy(dev, t->data, bytes, hipMemcpyHostToDevice) !=
                 hipSuccess) {
        (void)hipFree(dev);
        throw std::runtime_error("MTP mat upload failed: " + name);
      }
      im.allocs.push_back(dev);
      return MatRef{dev, tp, rb};
    };

    const auto* emb = reader->FindTensor("token_embd.weight");
    if (emb == nullptr || emb->dimensions.size() < 2 ||
        emb->dimensions[0] != kHidden) {
      throw std::runtime_error("MTP token_embd has unexpected shape");
    }
    im.vocab = static_cast<std::uint32_t>(emb->dimensions[1]);
    im.tok_embd = load_mat("token_embd.weight", kHidden, im.vocab);
    im.pre = load_mat("nextn.pre_projection.weight", 2ull * backbone_hidden,
                      kHidden);
    im.post =
        load_mat("nextn.post_projection.weight", kHidden, backbone_hidden);
    im.out_norm = load_vec("output_norm.weight", kHidden);

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
      const auto* ls_t = reader->FindTensor(p + "layer_output_scale.weight");
      if (ls_t == nullptr) {
        throw std::runtime_error("MTP missing layer_output_scale at layer " +
                                 std::to_string(il));
      }
      L.lscale_val = read_host_f32(ls_t, 1)[0];
    }
    im.Alloc(backbone_hidden);
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

bool MtpDrafter::SetTargetNorm(const float* raw_hidden, const float* norm_w,
                               float eps) {
#if defined(ENGINE_ENABLE_HIP)
  auto& im = *impl_;
  kernels::RmsNorm(raw_hidden, norm_w, im.d_tgt_hn, 1, im.backbone, eps, 0);
  return true;
#else
  (void)raw_hidden;
  (void)norm_w;
  (void)eps;
  return false;
#endif
}

const float* MtpDrafter::TargetNorm() const {
#if defined(ENGINE_ENABLE_HIP)
  return impl_->d_tgt_hn;
#else
  return nullptr;
#endif
}

bool MtpDrafter::Step(const float* tgt_emb, const float* tgt_hidden,
                      const void* swa_k, const void* swa_v,
                      std::uint32_t swa_stride, std::uint32_t swa_kv_heads,
                      const void* full_k, const void* full_v,
                      std::uint32_t full_stride, std::uint32_t full_kv_heads,
                      std::uint32_t kv_len, std::uint32_t rope_pos,
                      std::uint32_t sliding_window, float rope_theta_swa,
                      float rope_theta_full, const float* rope_freqs_full,
                      float* feedback_hidden, std::int32_t* d_out_token,
                      std::string* error_msg) {
#if defined(ENGINE_ENABLE_HIP)
  try {
    auto& im = *impl_;
    hipStream_t s = 0;
    const std::uint32_t bb = im.backbone;
    const float eps = im.rms_eps;

    // pre_projection(concat(tgt_emb, tgt_hidden)) -> im.d_h
    hipLaunchKernelGGL(ConcatTwoKernel, dim3((2u * bb + 255u) / 256u),
                       dim3(256), 0, s, tgt_emb, tgt_hidden, im.d_pre_in, bb);
    kernels::GemvBatched(im.pre.dev, im.pre.type, im.pre.row_bytes, 2ull * bb,
                         im.d_pre_in, im.d_h, 1u, kHidden, 0.0F, s);

    const bool bypass_attn = std::getenv("GEMMA4_MTP_NOATTN") != nullptr;
    const bool bypass_ffn = std::getenv("GEMMA4_MTP_NOFFN") != nullptr;
    for (std::uint32_t il = 0; il < kLayers; ++il) {
      const auto& L = im.layer[il];
      const bool is_swa = (L.hd == kHeadDimSwa);
      const __half* k = static_cast<const __half*>(is_swa ? swa_k : full_k);
      const __half* v = static_cast<const __half*>(is_swa ? swa_v : full_v);
      const std::uint32_t stride = is_swa ? swa_stride : full_stride;
      const std::uint32_t kvh = is_swa ? swa_kv_heads : full_kv_heads;
      if (k == nullptr || v == nullptr || kvh == 0) {
        throw std::runtime_error("MTP cross-attention KV missing");
      }

      kernels::RmsNorm(im.d_h, L.attn_norm, im.d_norm, 1u, kHidden, eps, s);
      kernels::GemvBatched(L.wq.dev, L.wq.type, L.wq.row_bytes, kHidden,
                           im.d_norm, im.d_q, 1u, kHeads * L.hd, 0.0F, s);
      kernels::PrepAttnQkv(im.d_q, nullptr, nullptr, L.wq_norm, nullptr,
                           nullptr, nullptr, nullptr, 1u, kHeads, 0u, false,
                           false, L.hd, L.hd, rope_pos,
                           is_swa ? rope_theta_swa : rope_theta_full,
                           (!is_swa ? rope_freqs_full : nullptr), eps, s);

      if (bypass_attn) {
        (void)hipMemsetAsync(
            im.d_attn, 0, static_cast<std::size_t>(kHeads) * L.hd * sizeof(float),
            s);
      } else {
        kernels::AttentionCross(im.d_q, k, v, im.d_attn, 1u, rope_pos, kv_len,
                                kHeads, kvh, L.hd, stride,
                                is_swa ? sliding_window : 0u, s);
      }

      kernels::GemvBatched(L.wo.dev, L.wo.type, L.wo.row_bytes, kHeads * L.hd,
                           im.d_attn, im.d_proj, 1u, kHidden, 0.0F, s);
      kernels::RmsNormAdd(im.d_proj, L.post_attn_norm, im.d_h, 1u, kHidden, eps,
                          1.0F, s);

      if (!bypass_ffn) {
        kernels::RmsNorm(im.d_h, L.ffn_norm, im.d_norm, 1u, kHidden, eps, s);
        if (L.w_gate.type == L.w_up.type &&
            std::getenv("GEMMA4_NO_WAVE32_GEMV") == nullptr) {
          kernels::GemvGateUpSwiglu(L.w_gate.dev, L.w_up.dev, L.w_gate.type,
                                    L.w_gate.row_bytes, kHidden, im.d_norm,
                                    im.d_ffn, 1u, kFf, s);
        } else {
          kernels::GemvBatched(L.w_gate.dev, L.w_gate.type, L.w_gate.row_bytes,
                               kHidden, im.d_norm, im.d_ffn, 1u, kFf, 0.0F, s);
          kernels::GemvBatched(L.w_up.dev, L.w_up.type, L.w_up.row_bytes,
                               kHidden, im.d_norm, im.d_ffn + kFf, 1u, kFf,
                               0.0F, s);
          kernels::GeluTanhSwiglu(im.d_ffn, im.d_ffn + kFf, kFf, s);
        }
        kernels::GemvBatched(L.w_down.dev, L.w_down.type, L.w_down.row_bytes,
                             kFf, im.d_ffn, im.d_proj, 1u, kHidden, 0.0F, s);
        kernels::RmsNormAdd(im.d_proj, L.post_ffw_norm, im.d_h, 1u, kHidden,
                            eps, L.lscale_val, s);
      } else if (L.lscale_val != 1.0F) {
        kernels::LayerScale(im.d_h, L.lscale_val, kHidden, s);
      }
    }

    // CRITICAL (gemma4-assistant.cpp parity): output_norm runs BEFORE both
    // output (tied LM head) AND nextn_proj_post (feedback_hidden)!
    kernels::RmsNorm(im.d_h, im.out_norm, im.d_norm, 1u, kHidden, eps, s);
    if (feedback_hidden != nullptr) {
      kernels::GemvQkv(im.post.dev, im.post.type, im.post.row_bytes, bb,
                       im.tok_embd.dev, im.tok_embd.type, im.tok_embd.row_bytes,
                       nullptr, 0, 0, im.vocab, kHidden, im.d_norm,
                       feedback_hidden, im.d_logits, nullptr, 1u, s);
    } else {
      kernels::GemvBatched(im.tok_embd.dev, im.tok_embd.type,
                           im.tok_embd.row_bytes, kHidden, im.d_norm,
                           im.d_logits, 1u, im.vocab, 0.0F, s);
    }
    q38::Argmax(im.d_logits, im.d_argmax_scratch, d_out_token, 1u, im.vocab, s);
    return true;
  } catch (const std::exception& e) {
    if (error_msg != nullptr) *error_msg = e.what();
    return false;
  }
#else
  (void)tgt_emb;
  (void)tgt_hidden;
  (void)swa_k;
  (void)swa_v;
  (void)swa_stride;
  (void)swa_kv_heads;
  (void)full_k;
  (void)full_v;
  (void)full_stride;
  (void)full_kv_heads;
  (void)kv_len;
  (void)rope_pos;
  (void)sliding_window;
  (void)rope_theta_swa;
  (void)rope_theta_full;
  (void)rope_freqs_full;
  (void)feedback_hidden;
  (void)d_out_token;
  if (error_msg != nullptr) *error_msg = "MTP requires the HIP build";
  return false;
#endif
}

}  // namespace gufo::models::gemma4
