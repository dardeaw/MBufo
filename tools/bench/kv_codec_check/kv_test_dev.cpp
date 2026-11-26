// kv_codec_check (device): shared blocked-KV kernels through the unified trait.
//
// QuantizeKvRows -> DequantizeKvRows roundtrip (34/34 and 34/18) plus tiny
// AttentionKv / AttentionBlockedSplitK cases against a host FP32 reference.
// Any refactor that changes shared-kernel numerics fails here.
//
// Built by tools/bench/kv_codec_check/kv_ab.sh (needs one ROCm GPU to run).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/core/compute/attention/blocked_kv.hpp"
#include "src/core/compute/attention/blocked_kv_codec.hpp"
#include "src/core/compute/attention/blocked_splitk.hpp"

namespace bk = mbufo::core::compute::blocked_kv;

static int g_bad = 0;
#define CHECK(cond, ...)                                \
  do {                                                  \
    if (!(cond)) {                                      \
      ++g_bad;                                          \
      std::printf("FAIL " __VA_ARGS__);                 \
      std::printf("\n");                                \
    }                                                   \
  } while (0)

#define HCHECK(call)                                                          \
  do {                                                                        \
    hipError_t _e = (call);                                                   \
    if (_e != hipSuccess) {                                                   \
      std::printf("HIP FAIL %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
      return 1;                                                               \
    }                                                                         \
  } while (0)

static void* DevAlloc(std::size_t bytes) {
  void* p = nullptr;
  if (hipMalloc(&p, bytes) != hipSuccess || p == nullptr) {
    std::printf("HIP FAIL: hipMalloc(%zu)\n", bytes);
    std::exit(1);
  }
  return p;
}

// host FP32 causal attention reference (no mask/bias/sink): query row qrow
// attends keys [0, kv_len).
static void RefAttnRow(const float* q, std::uint32_t qrow, const float* k,
                       const float* v, std::uint32_t kv_len, std::uint32_t h,
                       std::uint32_t n_heads, std::uint32_t kv_heads, std::uint32_t d, float scale,
                       float* out) {
  const std::uint32_t kvh = h / (n_heads / kv_heads);
  std::vector<float> scores(kv_len);
  float m = -1e30f;
  for (std::uint32_t p = 0; p < kv_len; ++p) {
    double dot = 0;
    for (std::uint32_t i = 0; i < d; ++i)
      dot += static_cast<double>(q[(qrow * n_heads + h) * d + i]) *
             static_cast<double>(k[(p * kv_heads + kvh) * d + i]);
    scores[p] = static_cast<float>(dot) * scale;
    if (scores[p] > m)
      m = scores[p];
  }
  double l = 0;
  for (auto& s : scores) {
    s = static_cast<float>(std::exp(static_cast<double>(s) - m));
    l += s;
  }
  for (std::uint32_t i = 0; i < d; ++i) {
    double acc = 0;
    for (std::uint32_t p = 0; p < kv_len; ++p)
      acc += scores[p] * v[(p * kv_heads + kvh) * d + i];
    out[i] = static_cast<float>(acc / l);
  }
}

int main() {
  std::printf("kv_codec_check (device):\n");
  hipStream_t stream = nullptr;
  HCHECK(hipStreamCreate(&stream));
  std::mt19937 rng(20240501);

  const std::uint32_t kv_heads = 2, d = 128, kv_dim = kv_heads * d;
  const std::uint32_t n_rows = 64;

  auto rand_rows = [&](std::uint32_t rows, float mag) {
    std::vector<float> v(static_cast<std::size_t>(rows) * kv_dim);
    for (auto& x : v)
      x = static_cast<float>(static_cast<int>(rng() % 200000) - 100000) * 1e-5f * mag;
    return v;
  };

  // ---- 1. Quantize -> Dequantize roundtrip, 34/34 and 34/18 ----
  for (int cfg = 0; cfg < 2; ++cfg) {
    const std::uint32_t vbb = cfg == 0 ? 34u : 18u;
    std::vector<float> hk = rand_rows(n_rows, 3.0f);
    std::vector<float> hv = rand_rows(n_rows, 1.0f);
    float* d_k = static_cast<float*>(DevAlloc(hk.size() * sizeof(float)));
    float* d_v = static_cast<float*>(DevAlloc(hv.size() * sizeof(float)));
    const std::size_t row_b = bk::KvBlockedRowBytes(kv_dim, 34u);
    const std::size_t vrow_b = bk::KvBlockedRowBytes(kv_dim, vbb);
    std::uint8_t* d_bk = static_cast<std::uint8_t*>(DevAlloc(n_rows * row_b));
    std::uint8_t* d_bv = static_cast<std::uint8_t*>(DevAlloc(n_rows * vrow_b));
    std::uint32_t* d_pos = static_cast<std::uint32_t*>(DevAlloc(sizeof(std::uint32_t)));
    HCHECK(hipMemcpyAsync(d_k, hk.data(), hk.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HCHECK(hipMemcpyAsync(d_v, hv.data(), hv.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HCHECK(hipMemsetAsync(d_pos, 0, sizeof(std::uint32_t), stream));
    bk::QuantizeKvRows(d_k, kv_dim, d, d_bk, kv_dim, kv_heads, d, n_rows, bk::KvCodec::kQ8, 127.0F,
                       34u, d_pos, stream);
    bk::QuantizeKvRows(d_v, kv_dim, d, d_bv, kv_dim, kv_heads, d, n_rows, bk::KvCodec::kQ8,
                       vbb == 18u ? 7.0F : 127.0F,
                       vbb, d_pos, stream);
    __half* d_dk = static_cast<__half*>(DevAlloc(hk.size() * sizeof(__half)));
    __half* d_dv = static_cast<__half*>(DevAlloc(hv.size() * sizeof(__half)));
    if (!bk::DequantizeKvRows(d_bk, d_dk, d_bv, d_dv, n_rows, kv_dim, 34u, vbb, bk::KvCodec::kQ8,
                              stream)) {
      std::printf("  roundtrip kbb=34 vbb=%u: DequantizeKvRows refused FAIL\n", vbb);
      ++g_bad;
      continue;
    }
    std::vector<__half> hk2(hk.size()), hv2(hv.size());
    HCHECK(hipMemcpyAsync(hk2.data(), d_dk, hk.size() * sizeof(__half), hipMemcpyDeviceToHost, stream));
    HCHECK(hipMemcpyAsync(hv2.data(), d_dv, hv.size() * sizeof(__half), hipMemcpyDeviceToHost, stream));
    HCHECK(hipStreamSynchronize(stream));
    auto to_f = [](__half h) {
      std::uint16_t b = 0;
      std::memcpy(&b, &h, 2);
      return bk::HalfBitsToFloat(b);
    };
    double worst_k = 0, worst_v = 0, amax = 0;
    for (std::size_t i = 0; i < hk.size(); ++i) {
      amax = std::fmax(amax, std::fabs(static_cast<double>(hk[i])));
      worst_k = std::fmax(worst_k, std::fabs(static_cast<double>(to_f(hk2[i])) - hk[i]));
    }
    for (std::size_t i = 0; i < hv.size(); ++i)
      worst_v = std::fmax(worst_v, std::fabs(static_cast<double>(to_f(hv2[i])) - hv[i]));
    const double bound = amax / (vbb == 18u ? 14.0 : 254.0) + 1e-3;
    bool ok = worst_k <= bound && worst_v <= bound;
    std::printf("  roundtrip kbb=34 vbb=%u: worst_k=%.5f worst_v=%.5f bound=%.5f %s\n", vbb, worst_k,
                worst_v, bound, ok ? "ok" : "FAIL");
    if (!ok)
      ++g_bad;
    hipFree(d_k);
    hipFree(d_v);
    hipFree(d_bk);
    hipFree(d_bv);
    hipFree(d_dk);
    hipFree(d_dv);
    hipFree(d_pos);
  }

  // ---- 2. AttentionKv smoke vs host FP32 reference ----
  {
    const std::uint32_t heads = 2, ntok = 2, krows = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> hq(static_cast<std::size_t>(ntok) * heads * d);
    for (auto& x : hq)
      x = static_cast<float>(static_cast<int>(rng() % 200000) - 100000) * 1e-5f;
    std::vector<float> hk = rand_rows(krows, 2.0f);
    std::vector<float> hv = rand_rows(krows, 1.0f);
    float* d_q = static_cast<float*>(DevAlloc(hq.size() * sizeof(float)));
    float* d_k = static_cast<float*>(DevAlloc(hk.size() * sizeof(float)));
    float* d_v = static_cast<float*>(DevAlloc(hv.size() * sizeof(float)));
    const std::size_t row_b = bk::KvBlockedRowBytes(kv_dim, 34u);
    std::uint8_t* d_bk = static_cast<std::uint8_t*>(DevAlloc(krows * row_b));
    std::uint8_t* d_bv = static_cast<std::uint8_t*>(DevAlloc(krows * row_b));
    float* d_out = static_cast<float*>(DevAlloc(hq.size() * sizeof(float)));
    std::uint32_t* d_pos = static_cast<std::uint32_t*>(DevAlloc(sizeof(std::uint32_t)));
    HCHECK(hipMemcpyAsync(d_q, hq.data(), hq.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HCHECK(hipMemcpyAsync(d_k, hk.data(), hk.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HCHECK(hipMemcpyAsync(d_v, hv.data(), hv.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HCHECK(hipMemsetAsync(d_pos, 0, sizeof(std::uint32_t), stream));
    HCHECK(hipMemsetAsync(d_out, 0, hq.size() * sizeof(float), stream));
    bk::QuantizeKvRows(d_k, kv_dim, d, d_bk, kv_dim, kv_heads, d, krows, bk::KvCodec::kQ8, 127.0F,
                       34u, d_pos, stream);
    bk::QuantizeKvRows(d_v, kv_dim, d, d_bv, kv_dim, kv_heads, d, krows, bk::KvCodec::kQ8, 127.0F,
                       34u, d_pos, stream);
    bk::AttentionKv(d_q, d_bk, d_bv, nullptr, 0, d_out, nullptr, 1, ntok, d_pos, heads, kv_heads, d,
                    1, 34u, 34u, bk::KvCodec::kQ8, stream, scale);
    std::vector<float> hout(hq.size(), 0);
    HCHECK(hipMemcpyAsync(hout.data(), d_out, hout.size() * sizeof(float), hipMemcpyDeviceToHost,
                          stream));
    HCHECK(hipStreamSynchronize(stream));
    double worst = 0, wmean = 0, rmean = 0;
    std::vector<float> ref(d);
    for (std::uint32_t t = 0; t < ntok; ++t)
      for (std::uint32_t h = 0; h < heads; ++h) {
        RefAttnRow(hq.data(), t, hk.data(), hv.data(), t + 1, h, heads, kv_heads, d, scale,
                   ref.data());
        for (std::uint32_t i = 0; i < d; ++i) {
          double e = std::fabs(static_cast<double>(hout[(t * heads + h) * d + i] - ref[i]));
          worst = std::fmax(worst, e);
          wmean += e;
          rmean += std::fabs(ref[i]);
        }
      }
    wmean /= hout.size();
    rmean /= hout.size();
    bool ok = worst < 0.08 * (rmean + 1e-3) * 10 && wmean < 0.02;
    std::printf("  AttentionKv smoke: worst=%.5f mean=%.5f refmean=%.4f %s\n", worst, wmean, rmean,
                ok ? "ok" : "FAIL");
    if (!ok)
      ++g_bad;
    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_v);
    hipFree(d_bk);
    hipFree(d_bv);
    hipFree(d_out);
    hipFree(d_pos);
  }

  // ---- 3. AttentionBlockedSplitK smoke vs host FP32 reference ----
  {
    const std::uint32_t heads = 2, ntok = 1;
    const std::uint32_t kv_len = 512;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> hq(static_cast<std::size_t>(ntok) * heads * d);
    for (auto& x : hq)
      x = static_cast<float>(static_cast<int>(rng() % 200000) - 100000) * 1e-5f;
    std::vector<float> hk = rand_rows(kv_len, 2.0f);
    std::vector<float> hv = rand_rows(kv_len, 1.0f);
    float* d_q = static_cast<float*>(DevAlloc(hq.size() * sizeof(float)));
    float* d_k = static_cast<float*>(DevAlloc(hk.size() * sizeof(float)));
    float* d_v = static_cast<float*>(DevAlloc(hv.size() * sizeof(float)));
    const std::size_t row_b = bk::KvBlockedRowBytes(kv_dim, 34u);
    std::uint8_t* d_bk = static_cast<std::uint8_t*>(DevAlloc(kv_len * row_b));
    std::uint8_t* d_bv = static_cast<std::uint8_t*>(DevAlloc(kv_len * row_b));
    float* d_out = static_cast<float*>(DevAlloc(hq.size() * sizeof(float)));
    std::uint32_t* d_qpos = static_cast<std::uint32_t*>(DevAlloc(sizeof(std::uint32_t)));
    std::uint32_t* d_spos = static_cast<std::uint32_t*>(DevAlloc(sizeof(std::uint32_t)));
    const std::size_t part_b = mbufo::core::compute::blocked_splitk::AttentionBlockedSplitKScratchBytes();
    float* d_part = static_cast<float*>(DevAlloc(part_b));
    HCHECK(hipMemcpyAsync(d_q, hq.data(), hq.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HCHECK(hipMemcpyAsync(d_k, hk.data(), hk.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HCHECK(hipMemcpyAsync(d_v, hv.data(), hv.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    HCHECK(hipMemsetAsync(d_qpos, 0, sizeof(std::uint32_t), stream));
    const std::uint32_t spos = kv_len - 1;
    HCHECK(hipMemcpyAsync(d_spos, &spos, sizeof(std::uint32_t), hipMemcpyHostToDevice, stream));
    HCHECK(hipMemsetAsync(d_out, 0, hq.size() * sizeof(float), stream));
    bk::QuantizeKvRows(d_k, kv_dim, d, d_bk, kv_dim, kv_heads, d, kv_len, bk::KvCodec::kQ8, 127.0F,
                       34u, d_qpos, stream);
    bk::QuantizeKvRows(d_v, kv_dim, d, d_bv, kv_dim, kv_heads, d, kv_len, bk::KvCodec::kQ8, 127.0F,
                       34u, d_qpos, stream);
    bool launched = mbufo::core::compute::blocked_splitk::AttentionBlockedSplitK<bk::KvCodecQ8, 128>(
        d_q, d_bk, d_bv, d_out, ntok, d_spos, kv_len, heads, kv_heads, d, stream, scale, 34u, 34u,
        d_part, part_b);
    std::vector<float> hout(hq.size(), 0);
    if (launched) {
      HCHECK(hipMemcpyAsync(hout.data(), d_out, hout.size() * sizeof(float), hipMemcpyDeviceToHost,
                            stream));
      HCHECK(hipStreamSynchronize(stream));
    }
    double worst = 0, wmean = 0, rmean = 0;
    std::vector<float> ref(d);
    for (std::uint32_t h = 0; h < heads; ++h) {
      // single decode row at the end attends [0, kv_len)
      RefAttnRow(hq.data(), 0, hk.data(), hv.data(), kv_len, h, heads, kv_heads, d, scale,
                 ref.data());
      for (std::uint32_t i = 0; i < d; ++i) {
        double e = std::fabs(static_cast<double>(hout[h * d + i] - ref[i]));
        worst = std::fmax(worst, e);
        wmean += e;
        rmean += std::fabs(ref[i]);
      }
    }
    wmean /= hout.size();
    rmean /= hout.size();
    // NOTE: rows beyond the causal extent hold zeros; the reference uses the
    // same stored rows, and start_pos puts the single row at the end.
    bool ok = launched && worst < 0.08 * (rmean + 1e-3) * 10 && wmean < 0.02;
    std::printf("  SplitK smoke: launched=%d worst=%.5f mean=%.5f refmean=%.4f %s\n", launched ? 1 : 0,
                worst, wmean, rmean, ok ? "ok" : "FAIL");
    if (!ok)
      ++g_bad;
    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_v);
    hipFree(d_bk);
    hipFree(d_bv);
    hipFree(d_out);
    hipFree(d_qpos);
    hipFree(d_spos);
    hipFree(d_part);
  }

  HCHECK(hipStreamDestroy(stream));
  if (g_bad) {
    std::printf("DEVICE FAILED (%d)\n", g_bad);
    return 1;
  }
  std::printf("device OK\n");
  return 0;
}
