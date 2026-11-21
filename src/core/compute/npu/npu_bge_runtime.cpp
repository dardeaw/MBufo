#include "src/core/compute/npu/npu_bge_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <utility>

#include "src/core/compute/gelu/gelu.hpp"
#include "src/core/compute/layernorm/layernorm.hpp"
#include "src/core/compute/npu/npu_pooling.hpp"

namespace mbufo::core::npu {
namespace {

// Upload a transposed [in_k, out_n] BF16 BO from row-major F32 [out_n, in_k].
std::unique_ptr<hrx::ext::bo> UploadTransposed1024(
    HrxSession& sess, const float* w, std::size_t out_row_start,
    std::size_t in_col_start, std::size_t total_in_k, std::string* error) {
  constexpr std::size_t K = 1024;
  constexpr std::size_t N = 1024;
  std::vector<std::uint16_t> buf(K * N);
  for (std::size_t n = 0; n < N; ++n) {
    const float* row = w + (out_row_start + n) * total_in_k + in_col_start;
    for (std::size_t k = 0; k < K; ++k) {
      buf[k * N + n] = FloatToBf16Rne(row[k]);
    }
  }
  auto bo = sess.AllocBo(buf.size() * sizeof(std::uint16_t), error);
  if (!bo) return nullptr;
  std::memcpy(bo->map<void*>(), buf.data(),
              buf.size() * sizeof(std::uint16_t));
  bo->flush();
  return bo;
}

void PackChunkBf16(const float* src, std::size_t valid_rows,
                   std::size_t src_stride, std::size_t src_col_offset,
                   hrx::ext::bo& dst_bo) {
  constexpr std::size_t M = NpuBgeRuntime::kChunkRows;
  constexpr std::size_t K = NpuBgeRuntime::kHidden;
  auto* dst = dst_bo.map<std::uint16_t*>();
  std::memset(dst, 0, M * K * sizeof(std::uint16_t));
  for (std::size_t r = 0; r < valid_rows; ++r) {
    const float* s = src + r * src_stride + src_col_offset;
    std::uint16_t* d = dst + r * K;
    for (std::size_t c = 0; c < K; ++c) {
      d[c] = FloatToBf16Rne(s[c]);
    }
  }
  dst_bo.flush();
}

void UnpackChunkAddBias(hrx::ext::bo& src_bo, std::size_t valid_rows,
                        const float* bias, float* dst,
                        std::size_t dst_stride) {
  src_bo.invalidate();
  const auto* src = src_bo.map<const std::uint16_t*>();
  constexpr std::size_t N = NpuBgeRuntime::kHidden;
  for (std::size_t r = 0; r < valid_rows; ++r) {
    const std::uint16_t* s = src + r * N;
    float* d = dst + r * dst_stride;
    for (std::size_t c = 0; c < N; ++c) {
      d[c] = Bf16BitsToFloat(s[c]) + (bias != nullptr ? bias[c] : 0.0F);
    }
  }
}

}  // namespace

bool NpuBgeRuntime::Init(const models::bge_m3::BgeM3F32Weights& weights,
                         const std::string& kernel_dir, std::string* error) {
  resident_ = false;
  kernel_dir_ = kernel_dir;
  const auto& cfg = weights.config;
  if (cfg.hidden_size != kHidden || cfg.intermediate_size != kInter ||
      cfg.num_layers == 0 || weights.layers.size() != cfg.num_layers) {
    if (error != nullptr) {
      *error = "NpuBgeRuntime: unsupported BGE geometry (expects H=1024, F=4096)";
    }
    return false;
  }

  session_ = HrxSession::Open(0, error);
  if (!session_) return false;

  const std::string xcl =
      (std::filesystem::path(kernel_dir) / "gemm_1024x1024.xclbin").string();
  const std::string inst =
      (std::filesystem::path(kernel_dir) / "gemm_1024x1024.insts.bin").string();
  if (!session_->LoadIronKernel("gemm_1024x1024", xcl, inst, error)) {
    return false;
  }

  layers_.clear();
  layers_.resize(cfg.num_layers);
  for (std::size_t li = 0; li < cfg.num_layers; ++li) {
    const auto& src = weights.layers[li];
    LayerResident& dst = layers_[li];
    dst.q_bo = UploadTransposed1024(*session_, src.q_w.data(), 0, 0, kHidden, error);
    dst.k_bo = UploadTransposed1024(*session_, src.k_w.data(), 0, 0, kHidden, error);
    dst.v_bo = UploadTransposed1024(*session_, src.v_w.data(), 0, 0, kHidden, error);
    dst.out_bo = UploadTransposed1024(*session_, src.out_w.data(), 0, 0, kHidden, error);
    if (!dst.q_bo || !dst.k_bo || !dst.v_bo || !dst.out_bo) return false;
    for (std::size_t s = 0; s < 4; ++s) {
      dst.up_bo[s] = UploadTransposed1024(*session_, src.up_w.data(),
                                          s * kHidden, 0, kHidden, error);
      dst.down_bo[s] = UploadTransposed1024(*session_, src.down_w.data(),
                                            0, s * kHidden, kInter, error);
      if (!dst.up_bo[s] || !dst.down_bo[s]) return false;
    }
  }

  for (std::size_t i = 0; i < 4; ++i) {
    a_bo_[i] = session_->AllocBo(kChunkRows * kHidden * sizeof(std::uint16_t), error);
    c_bo_[i] = session_->AllocBo(kChunkRows * kHidden * sizeof(std::uint16_t), error);
    if (!a_bo_[i] || !c_bo_[i]) return false;
  }

  resident_ = true;
  return true;
}

bool NpuBgeRuntime::ForwardCls(std::span<const std::uint32_t> ids,
                               const models::bge_m3::BgeM3F32Weights& weights,
                               std::vector<float>* cls_out,
                               double* prefill_ms, std::string* error) {
  if (!resident_ || !session_) {
    if (error != nullptr) *error = "NpuBgeRuntime: not initialized";
    return false;
  }
  const std::size_t n = ids.size();
  const auto& cfg = weights.config;
  const std::size_t h = cfg.hidden_size;
  const std::size_t heads = cfg.num_attention_heads;
  const std::size_t inter = cfg.intermediate_size;
  const std::size_t hd = heads == 0 ? 0 : h / heads;
  if (n == 0 || n > 2048 || h != kHidden || inter != kInter || heads == 0 ||
      hd == 0 || weights.position_rows < n) {
    if (error != nullptr) *error = "NpuBgeRuntime: invalid input size/weights";
    return false;
  }

  const auto start = std::chrono::steady_clock::now();
  std::vector<float> hidden(n * h, 0.0F);
  std::vector<float> normed(n * h, 0.0F);
  for (std::size_t t = 0; t < n; ++t) {
    const std::size_t id = ids[t];
    if (id >= cfg.vocab_size) {
      if (error != nullptr) *error = "NpuBgeRuntime: token id out of vocab";
      return false;
    }
    const float* wrow = weights.token_embd.data() + id * h;
    const float* prow = weights.position_embd.data() + t * h;
    const float* trow = weights.token_types.data();
    float* hrow = hidden.data() + t * h;
    for (std::size_t d = 0; d < h; ++d) {
      hrow[d] = wrow[d] + prow[d] + trow[d];
    }
    if (!core::ln::LayerNorm(
            std::span<const float>(hrow, h),
            std::span<const float>(weights.emb_norm_w.data(), h),
            std::span<const float>(weights.emb_norm_b.data(), h),
            cfg.layer_norm_eps, std::span<float>(normed.data() + t * h, h))) {
      if (error != nullptr) *error = "NpuBgeRuntime: emb LayerNorm failed";
      return false;
    }
  }
  hidden.swap(normed);

  std::vector<float> q(n * h), k(n * h), v(n * h);
  std::vector<float> scores(heads * n * n);
  std::vector<float> ctx(n * h), proj(n * h);
  std::vector<float> ffn_inter(n * inter), ffn_out(n * h);
  const double scale = 1.0 / std::sqrt(static_cast<double>(hd));

  for (std::size_t li = 0; li < weights.layers.size(); ++li) {
    const auto& L_w = weights.layers[li];
    const auto& L_n = layers_[li];

    // Q, K, V projections via 3-step NPU kernel chain per <=64-row chunk.
    for (std::size_t r0 = 0; r0 < n; r0 += kChunkRows) {
      const std::size_t cnt = std::min(kChunkRows, n - r0);
      PackChunkBf16(hidden.data() + r0 * h, cnt, h, 0, *a_bo_[0]);
      std::vector<std::pair<std::string, std::vector<const hrx::bo*>>> qkv_steps;
      qkv_steps.emplace_back("gemm_1024x1024",
                             std::vector<const hrx::bo*>{
                                 a_bo_[0].get(), L_n.q_bo.get(), c_bo_[0].get()});
      qkv_steps.emplace_back("gemm_1024x1024",
                             std::vector<const hrx::bo*>{
                                 a_bo_[0].get(), L_n.k_bo.get(), c_bo_[1].get()});
      qkv_steps.emplace_back("gemm_1024x1024",
                             std::vector<const hrx::bo*>{
                                 a_bo_[0].get(), L_n.v_bo.get(), c_bo_[2].get()});
      if (!session_->RunKernelChain(qkv_steps, error)) return false;
      UnpackChunkAddBias(*c_bo_[0], cnt, L_w.q_b.data(), q.data() + r0 * h, h);
      UnpackChunkAddBias(*c_bo_[1], cnt, L_w.k_b.data(), k.data() + r0 * h, h);
      UnpackChunkAddBias(*c_bo_[2], cnt, L_w.v_b.data(), v.data() + r0 * h, h);
    }

    // Exact bidirectional multi-head attention across heads.
#pragma omp parallel for schedule(static)
    for (std::size_t hh = 0; hh < heads; ++hh) {
      float* sc = scores.data() + hh * n * n;
      for (std::size_t i = 0; i < n; ++i) {
        const float* qi = q.data() + i * h + hh * hd;
        double row_max = -1e300;
        for (std::size_t j = 0; j < n; ++j) {
          const float* kj = k.data() + j * h + hh * hd;
          double dot = 0.0;
          for (std::size_t d = 0; d < hd; ++d) {
            dot += static_cast<double>(qi[d]) * static_cast<double>(kj[d]);
          }
          const double s = dot * scale;
          sc[i * n + j] = static_cast<float>(s);
          if (s > row_max) row_max = s;
        }
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
          const double e = std::exp(sc[i * n + j] - row_max);
          sc[i * n + j] = static_cast<float>(e);
          sum += e;
        }
        const float inv = static_cast<float>(1.0 / sum);
        for (std::size_t j = 0; j < n; ++j) sc[i * n + j] *= inv;
        float* ci = ctx.data() + i * h + hh * hd;
        std::fill(ci, ci + hd, 0.0F);
        for (std::size_t j = 0; j < n; ++j) {
          const float w = sc[i * n + j];
          const float* vj = v.data() + j * h + hh * hd;
          for (std::size_t d = 0; d < hd; ++d) ci[d] += w * vj[d];
        }
      }
    }

    // Attention O projection on NPU2.
    for (std::size_t r0 = 0; r0 < n; r0 += kChunkRows) {
      const std::size_t cnt = std::min(kChunkRows, n - r0);
      PackChunkBf16(ctx.data() + r0 * h, cnt, h, 0, *a_bo_[0]);
      if (!session_->RunKernelSync("gemm_1024x1024",
                                   {a_bo_[0].get(), L_n.out_bo.get(),
                                    c_bo_[0].get()},
                                   error)) {
        return false;
      }
      UnpackChunkAddBias(*c_bo_[0], cnt, L_w.out_b.data(),
                         proj.data() + r0 * h, h);
    }

    for (std::size_t t = 0; t < n; ++t) {
      float* hrow = hidden.data() + t * h;
      const float* prow = proj.data() + t * h;
      for (std::size_t d = 0; d < h; ++d) hrow[d] += prow[d];
      if (!core::ln::LayerNorm(
              std::span<const float>(hrow, h),
              std::span<const float>(L_w.attn_norm_w.data(), h),
              std::span<const float>(L_w.attn_norm_b.data(), h),
              cfg.layer_norm_eps,
              std::span<float>(normed.data() + t * h, h))) {
        if (error != nullptr) *error = "NpuBgeRuntime: attn LayerNorm failed";
        return false;
      }
    }

    // FFN Up (4 slices of 1024) on NPU2 + bias + exact GELU.
    for (std::size_t r0 = 0; r0 < n; r0 += kChunkRows) {
      const std::size_t cnt = std::min(kChunkRows, n - r0);
      PackChunkBf16(normed.data() + r0 * h, cnt, h, 0, *a_bo_[0]);
      std::vector<std::pair<std::string, std::vector<const hrx::bo*>>> up_steps;
      for (std::size_t s = 0; s < 4; ++s) {
        up_steps.emplace_back("gemm_1024x1024",
                              std::vector<const hrx::bo*>{
                                  a_bo_[0].get(), L_n.up_bo[s].get(),
                                  c_bo_[s].get()});
      }
      if (!session_->RunKernelChain(up_steps, error)) return false;
      for (std::size_t s = 0; s < 4; ++s) {
        UnpackChunkAddBias(*c_bo_[s], cnt, L_w.up_b.data() + s * h,
                           ffn_inter.data() + r0 * inter + s * h, inter);
      }
    }
    if (!core::gelu::ApplyGelu(
            std::span<const float>(ffn_inter.data(), n * inter),
            std::span<float>(ffn_inter.data(), n * inter),
            core::gelu::GeluMode::kExact)) {
      if (error != nullptr) *error = "NpuBgeRuntime: ApplyGelu failed";
      return false;
    }

    // FFN Down (4 K-split slices accumulated) on NPU2 + bias + residual + norm.
    for (std::size_t r0 = 0; r0 < n; r0 += kChunkRows) {
      const std::size_t cnt = std::min(kChunkRows, n - r0);
      std::vector<std::pair<std::string, std::vector<const hrx::bo*>>> dn_steps;
      for (std::size_t s = 0; s < 4; ++s) {
        PackChunkBf16(ffn_inter.data() + r0 * inter, cnt, inter, s * h, *a_bo_[s]);
        dn_steps.emplace_back("gemm_1024x1024",
                              std::vector<const hrx::bo*>{
                                  a_bo_[s].get(), L_n.down_bo[s].get(),
                                  c_bo_[s].get()});
      }
      if (!session_->RunKernelChain(dn_steps, error)) return false;
      for (auto& cb : c_bo_) cb->invalidate();
      for (std::size_t r = 0; r < cnt; ++r) {
        float* fout = ffn_out.data() + (r0 + r) * h;
        const std::uint16_t* p0 = c_bo_[0]->map<const std::uint16_t*>() + r * h;
        const std::uint16_t* p1 = c_bo_[1]->map<const std::uint16_t*>() + r * h;
        const std::uint16_t* p2 = c_bo_[2]->map<const std::uint16_t*>() + r * h;
        const std::uint16_t* p3 = c_bo_[3]->map<const std::uint16_t*>() + r * h;
        for (std::size_t c = 0; c < h; ++c) {
          fout[c] = Bf16BitsToFloat(p0[c]) + Bf16BitsToFloat(p1[c]) +
                    Bf16BitsToFloat(p2[c]) + Bf16BitsToFloat(p3[c]) +
                    L_w.down_b[c];
        }
      }
    }

    for (std::size_t t = 0; t < n; ++t) {
      float* nrow = normed.data() + t * h;
      const float* frow = ffn_out.data() + t * h;
      for (std::size_t d = 0; d < h; ++d) nrow[d] += frow[d];
      if (!core::ln::LayerNorm(
              std::span<const float>(nrow, h),
              std::span<const float>(L_w.out_norm_w.data(), h),
              std::span<const float>(L_w.out_norm_b.data(), h),
              cfg.layer_norm_eps,
              std::span<float>(hidden.data() + t * h, h))) {
        if (error != nullptr) *error = "NpuBgeRuntime: out LayerNorm failed";
        return false;
      }
    }
  }

  cls_out->assign(hidden.begin(), hidden.begin() + h);
  if (prefill_ms != nullptr) {
    *prefill_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  }
  return true;
}

}  // namespace mbufo::core::npu
