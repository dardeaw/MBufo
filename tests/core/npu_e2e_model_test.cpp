#include "src/core/compute/npu/npu_device.hpp"
#include "src/core/compute/npu/npu_pooling.hpp"
#include "src/core/compute/npu/npu_scheduler.hpp"
#include "src/core/pooling_norm.hpp"
#include "src/models/bge_m3/engine.hpp"
#include "src/models/bge_reranker/engine.hpp"
#include "src/models/embeddinggemma2/engine.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using mbufo::core::npu::Bf16BitsToFloat;
using mbufo::core::npu::NpuEmbGemma2PostProcess;
using mbufo::core::npu::NpuScheduler;

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  assert(a.size() == b.size() && !a.empty());
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

std::vector<float> ReadF32Tensor(const mbufo::core::GgufReader& reader,
                                 const std::string& name, std::size_t count) {
  const auto* t = reader.FindTensor(name);
  assert(t != nullptr && t->ElementCount() == count);
  std::vector<float> out(count);
  if (t->type == mbufo::core::GgmlType::kF32) {
    std::memcpy(out.data(), t->data, count * sizeof(float));
  } else {
    assert(t->type == mbufo::core::GgmlType::kBF16);
    const auto* p = static_cast<const std::uint16_t*>(t->data);
    for (std::size_t i = 0; i < count; ++i) out[i] = Bf16BitsToFloat(p[i]);
  }
  return out;
}

void RefMatMulT(const std::vector<float>& a, const std::vector<float>& bt,
                std::size_t m, std::size_t n, std::size_t k,
                std::vector<float>* out) {
  out->assign(m * n, 0.0F);
#pragma omp parallel for schedule(static) if (m * n >= 1024)
  for (std::size_t r = 0; r < m; ++r) {
    for (std::size_t c = 0; c < n; ++c) {
      double acc = 0.0;
      for (std::size_t i = 0; i < k; ++i) {
        acc += static_cast<double>(a[r * k + i]) *
               static_cast<double>(bt[c * k + i]);
      }
      (*out)[r * n + c] = static_cast<float>(acc);
    }
  }
}

void RefRms(const std::vector<float>& x, const std::vector<float>& w,
            std::size_t rows, std::size_t cols, float pre_scale, float eps,
            std::vector<float>* out) {
  out->resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      const double v = static_cast<double>(x[r * cols + c]) * pre_scale;
      ss += v * v;
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / cols + eps));
    for (std::size_t c = 0; c < cols; ++c) {
      (*out)[r * cols + c] =
          x[r * cols + c] * pre_scale * inv * (w.empty() ? 1.0F : w[c]);
    }
  }
}

float GeluTanh(float x) {
  const float u = 0.7978845608028654F * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + std::tanh(u));
}

std::vector<float> RefEmbGemma2Forward(
    const mbufo::models::embeddinggemma2::EngineModel& model,
    std::span<const std::uint32_t> ids) {
  const auto& cfg = model.GetConfig();
  const auto& reader = model.GetReader();
  const std::size_t n = ids.size();
  const std::size_t H = cfg.hidden;
  const std::size_t F = cfg.ffn;
  const std::size_t P = cfg.ple_dim;

  const auto* emb_t = reader.FindTensor("embed_tokens.weight");
  assert(emb_t != nullptr && emb_t->type == mbufo::core::GgmlType::kBF16);
  const auto* emb_data = static_cast<const std::uint16_t*>(emb_t->data);

  std::vector<float> emb_f32(n * H);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t c = 0; c < H; ++c) {
      emb_f32[t * H + c] = Bf16BitsToFloat(emb_data[ids[t] * H + c]) * cfg.embed_scale;
    }
  }
  std::vector<float> h = emb_f32;
  const auto ple_all =
      ReadF32Tensor(reader, "ple.per_layer_model_projection.weight",
                    cfg.layers * P * H);
  const auto ple_norm =
      ReadF32Tensor(reader, "ple.per_layer_projection_norm.weight", P);
  const float ple_pre = 1.0F / std::sqrt(static_cast<float>(H));

  for (std::uint32_t li = 0; li < cfg.layers; ++li) {
    const bool full = cfg.layer_is_full[li] != 0;
    const std::size_t d = full ? cfg.head_dim_full : cfg.head_dim_local;
    const std::size_t kvh = full ? cfg.kv_full : cfg.kv_local;
    const std::size_t qd = cfg.heads * d;
    const std::size_t kd = kvh * d;
    const std::size_t group = cfg.heads / kvh;
    const float theta = full ? cfg.rope_full : cfg.rope_local;
    const int window = full ? -1 : static_cast<int>(cfg.sliding_window);
    const std::string p = "layers." + std::to_string(li) + ".";

    const auto in_norm = ReadF32Tensor(reader, p + "input_layernorm.weight", H);
    const auto wq = ReadF32Tensor(reader, p + "self_attn.q_proj.weight", qd * H);
    const auto wk = ReadF32Tensor(reader, p + "self_attn.k_proj.weight", kd * H);
    const auto wv = ReadF32Tensor(reader, p + "self_attn.v_proj.weight", kd * H);
    const auto qn = ReadF32Tensor(reader, p + "self_attn.q_norm.weight", d);
    const auto kn = ReadF32Tensor(reader, p + "self_attn.k_norm.weight", d);
    const auto wo = ReadF32Tensor(reader, p + "self_attn.o_proj.weight", H * qd);
    const auto post_attn =
        ReadF32Tensor(reader, p + "post_attention_layernorm.weight", H);

    std::vector<float> xn, q_m, k_m, v_m;
    RefRms(h, in_norm, n, H, 1.0F, cfg.rms_eps, &xn);
    RefMatMulT(xn, wq, n, qd, H, &q_m);
    RefMatMulT(xn, wk, n, kd, H, &k_m);
    RefMatMulT(xn, wv, n, kd, H, &v_m);
    RefRms(q_m, qn, n * cfg.heads, d, 1.0F, cfg.rms_eps, &q_m);
    RefRms(k_m, kn, n * kvh, d, 1.0F, cfg.rms_eps, &k_m);
    RefRms(v_m, {}, n * kvh, d, 1.0F, cfg.rms_eps, &v_m);

    auto apply_rope = [&](std::vector<float>& m, std::size_t nh) {
      const std::size_t half = d / 2;
      for (std::size_t t = 0; t < n; ++t) {
        for (std::size_t hh = 0; hh < nh; ++hh) {
          float* vec = m.data() + (t * nh + hh) * d;
          for (std::size_t i = 0; i < half; ++i) {
            const double ang =
                static_cast<double>(t) *
                std::pow(static_cast<double>(theta), -2.0 * i / d);
            const float c = static_cast<float>(std::cos(ang));
            const float s = static_cast<float>(std::sin(ang));
            const float a = vec[i], b = vec[i + half];
            vec[i] = a * c - b * s;
            vec[i + half] = b * c + a * s;
          }
        }
      }
    };
    apply_rope(q_m, cfg.heads);
    apply_rope(k_m, kvh);

    std::vector<float> ctx(n * qd, 0.0F);
    for (std::size_t hq = 0; hq < cfg.heads; ++hq) {
      const std::size_t hk = hq / group;
      for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> sc(n, -1e30);
        double mx = -1e30;
        for (std::size_t j = 0; j < n; ++j) {
          if (window >= 0 &&
              std::abs(static_cast<int>(i) - static_cast<int>(j)) > window) {
            continue;
          }
          double dot = 0.0;
          for (std::size_t x = 0; x < d; ++x) {
            dot += static_cast<double>(q_m[(i * cfg.heads + hq) * d + x]) *
                   static_cast<double>(k_m[(j * kvh + hk) * d + x]);
          }
          sc[j] = dot;
          if (dot > mx) mx = dot;
        }
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
          if (window >= 0 &&
              std::abs(static_cast<int>(i) - static_cast<int>(j)) > window) {
            sc[j] = 0.0;
          } else {
            sc[j] = std::exp(sc[j] - mx);
            sum += sc[j];
          }
        }
        for (std::size_t j = 0; j < n; ++j) {
          const double w = sc[j] / sum;
          for (std::size_t x = 0; x < d; ++x) {
            ctx[(i * cfg.heads + hq) * d + x] +=
                static_cast<float>(w * v_m[(j * kvh + hk) * d + x]);
          }
        }
      }
    }

    std::vector<float> y, yn;
    RefMatMulT(ctx, wo, n, H, qd, &y);
    RefRms(y, post_attn, n, H, 1.0F, cfg.rms_eps, &yn);
    for (std::size_t i = 0; i < n * H; ++i) h[i] += yn[i];

    const auto pre_ffn =
        ReadF32Tensor(reader, p + "pre_feedforward_layernorm.weight", H);
    const auto wgate = ReadF32Tensor(reader, p + "mlp.gate_proj.weight", F * H);
    const auto wup = ReadF32Tensor(reader, p + "mlp.up_proj.weight", F * H);
    const auto wdn = ReadF32Tensor(reader, p + "mlp.down_proj.weight", H * F);
    const auto post_ffn =
        ReadF32Tensor(reader, p + "post_feedforward_layernorm.weight", H);
    std::vector<float> xf, gate, up, act(n * F), y_ffn, yn_ffn;
    RefRms(h, pre_ffn, n, H, 1.0F, cfg.rms_eps, &xf);
    RefMatMulT(xf, wgate, n, F, H, &gate);
    RefMatMulT(xf, wup, n, F, H, &up);
    for (std::size_t i = 0; i < n * F; ++i) act[i] = GeluTanh(gate[i]) * up[i];
    RefMatMulT(act, wdn, n, H, F, &y_ffn);
    RefRms(y_ffn, post_ffn, n, H, 1.0F, cfg.rms_eps, &yn_ffn);
    for (std::size_t i = 0; i < n * H; ++i) h[i] += yn_ffn[i];

    std::vector<float> ple_slice(ple_all.begin() + li * P * H,
                                 ple_all.begin() + (li + 1) * P * H);
    const auto wpgate =
        ReadF32Tensor(reader, p + "ple_block.per_layer_input_gate.weight", P * H);
    const auto wpproj =
        ReadF32Tensor(reader, p + "ple_block.per_layer_projection.weight", H * P);
    const auto ple_post =
        ReadF32Tensor(reader, p + "ple_block.post_per_layer_input_norm.weight", H);
    const auto scalar = ReadF32Tensor(reader, p + "layer_scalar", 1)[0];

    std::vector<float> g, pl, pg, actp(n * P), yp, ypn;
    RefMatMulT(emb_f32, ple_slice, n, P, H, &g);
    RefRms(g, ple_norm, n, P, ple_pre, cfg.rms_eps, &pl);
    RefMatMulT(h, wpgate, n, P, H, &pg);
    for (std::size_t i = 0; i < n * P; ++i) actp[i] = GeluTanh(pg[i]) * pl[i];
    RefMatMulT(actp, wpproj, n, H, P, &yp);
    RefRms(yp, ple_post, n, H, 1.0F, cfg.rms_eps, &ypn);
    for (std::size_t i = 0; i < n * H; ++i) h[i] = (h[i] + ypn[i]) * scalar;
  }

  const auto final_norm = ReadF32Tensor(reader, "norm.weight", H);
  const auto out_proj =
      ReadF32Tensor(reader, "embedding_projection.weight", cfg.out_dim * H);
  std::vector<float> out_emb;
  double pre_norm = 0.0;
  assert(NpuEmbGemma2PostProcess(h, n, H, final_norm, cfg.rms_eps, out_proj,
                                 cfg.out_dim, out_emb, &pre_norm));
  return out_emb;
}

void TestEmbeddingGemma2NpuE2E() {
  const std::string path =
      "/mnt/nvme-data/mbufo-models/embeddinggemma-2-gguf/embeddinggemma-2-bf16.gguf";
  if (!std::filesystem::exists(path)) {
    std::cout << "[SKIP] EmbeddingGemma 2 GGUF missing\n";
    return;
  }
  std::string err;
  auto model = mbufo::models::embeddinggemma2::EngineModel::Create(path, &err);
  assert(model != nullptr);
  assert(model->LoadNpuWeights("", &err));
  assert(model->npu_weights_resident());

  mbufo::npu::NpuLaneConfig cfg;
  cfg.embedding = true;
  NpuScheduler sched(cfg);
  model->SetNpuLaneConfig(cfg);
  model->SetNpuScheduler(&sched);

  const auto ids1 = model->Encode("What is the capital of France?");
  const auto ids2 = model->Encode("Paris is the capital of France.");
  const auto ids3 = model->Encode("Database vacuum freeze tuning on SSD.");
  assert(!ids1.empty() && ids1.size() <= 128);

  const auto npu1 = model->Embed(ids1, &err);
  const auto npu2 = model->Embed(ids2, &err);
  const auto npu3 = model->Embed(ids3, &err);
  assert(npu1.has_value() && npu2.has_value() && npu3.has_value());
  assert(npu1->values.size() == 768);
  const double norm1 = mbufo::core::embedding::VectorNorm(npu1->values);
  assert(std::fabs(norm1 - 1.0) < 1e-4);

  const auto ref1 = RefEmbGemma2Forward(*model, ids1);
  const double cos_ref = Cosine(npu1->values, ref1);
  const double cos_sim = Cosine(npu1->values, npu2->values);
  const double cos_diff = Cosine(npu1->values, npu3->values);

  std::cout << "[EmbGemma2 NPU E2E] seq=" << ids1.size()
            << " ms=" << npu1->prefill_ms << " cos_vs_fp32_ref=" << cos_ref
            << " sim_cos=" << cos_sim << " diff_cos=" << cos_diff << "\n";
  assert(cos_ref >= 0.995);
  assert(cos_sim > cos_diff);
}

void TestBgeM3NpuE2E() {
  const std::string path =
      "/mnt/nvme-data/mbufo-models/bge-m3/bge-m3-FP16.gguf";
  if (!std::filesystem::exists(path)) {
    std::cout << "[SKIP] bge-m3 GGUF missing\n";
    return;
  }
  std::string err;
  auto model = mbufo::models::bge_m3::EngineModel::Create(path, &err);
  assert(model != nullptr);
  assert(model->LoadNpuWeights("", &err));
  assert(model->npu_weights_resident());

  auto ids = model->Tokenizer().Encode(
      "AMD Ryzen AI XDNA2 NPU matrix multiplication accelerator.",
      mbufo::models::bge_m3::kBgeM3Leg1MaxTokens);
  assert(ids.has_value());

  mbufo::models::bge_m3::EngineSession cpu_sess(model.get());
  const auto cpu_res = cpu_sess.Sync(*ids, false, &err);
  assert(cpu_res.has_value());

  mbufo::npu::NpuLaneConfig cfg;
  cfg.embedding = true;
  NpuScheduler sched(cfg);
  mbufo::models::bge_m3::EngineSession npu_sess(model.get());
  npu_sess.SetNpuLaneConfig(cfg);
  npu_sess.SetNpuScheduler(&sched);

  const auto npu_res = npu_sess.Sync(*ids, false, &err);
  assert(npu_res.has_value());
  const double cos = Cosine(cpu_res->values, npu_res->values);
  std::cout << "[BGE-M3 NPU E2E] seq=" << ids->size()
            << " cpu_ms=" << cpu_res->prefill_ms
            << " npu_ms=" << npu_res->prefill_ms << " cos_vs_cpu=" << cos
            << "\n";
  assert(cos >= 0.995);
}

void TestBgeRerankerNpuE2E() {
  const std::string path =
      "/mnt/nvme-data/mbufo-models/bge-reranker-v2-m3/bge-reranker-v2-m3-FP16.gguf";
  if (!std::filesystem::exists(path)) {
    std::cout << "[SKIP] bge-reranker GGUF missing\n";
    return;
  }
  std::string err;
  auto model = mbufo::models::bge_reranker::EngineModel::Create(path, &err);
  assert(model != nullptr);
  assert(model->LoadNpuWeights("", &err));
  assert(model->npu_weights_resident());

  const std::string q = "What is the capital of France?";
  const std::string doc_rel = "Paris is the capital and most populous city of France.";
  const std::string doc_irrel = "Bananas are elongated edible berries produced by Musa plants.";

  mbufo::models::bge_reranker::EngineSession cpu_sess(model.get());
  const auto cpu_rel = cpu_sess.ScorePair(q, doc_rel, false, &err);
  const auto cpu_irr = cpu_sess.ScorePair(q, doc_irrel, false, &err);
  assert(cpu_rel.has_value() && cpu_irr.has_value());

  mbufo::npu::NpuLaneConfig cfg;
  cfg.reranker = true;
  NpuScheduler sched(cfg);
  mbufo::models::bge_reranker::EngineSession npu_sess(model.get());
  npu_sess.SetNpuLaneConfig(cfg);
  npu_sess.SetNpuScheduler(&sched);

  const auto npu_rel = npu_sess.ScorePair(q, doc_rel, false, &err);
  const auto npu_irr = npu_sess.ScorePair(q, doc_irrel, false, &err);
  assert(npu_rel.has_value() && npu_irr.has_value());

  const double diff_rel = std::fabs(npu_rel->score - cpu_rel->score);
  const double diff_irr = std::fabs(npu_irr->score - cpu_irr->score);
  std::cout << "[BGE-Reranker NPU E2E] rel cpu=" << cpu_rel->score
            << " npu=" << npu_rel->score << " (diff=" << diff_rel << ")"
            << " | irr cpu=" << cpu_irr->score << " npu=" << npu_irr->score
            << " (diff=" << diff_irr << ") ms=" << npu_rel->prefill_ms << "\n";
  assert(diff_rel <= 0.02);
  assert(diff_irr <= 0.02);
  assert(npu_rel->score > npu_irr->score);
}

}  // namespace

int main() {
  TestEmbeddingGemma2NpuE2E();
  TestBgeM3NpuE2E();
  TestBgeRerankerNpuE2E();
  std::cout << "ALL_NPU_E2E_MODEL_TESTS_PASS\n";
  return 0;
}
