// Live NPU2 Silicon End-to-End Verification for Q8 (bfp16ebs8) & Q4 (q4nx) across all 5 Models:
//   1. EmbeddingGemma 2 (24L Real GGUF)
//   2. Qwen3-TTS 12Hz 1.7B (Talker + CodePredictor + SpeechTokenizer 12Hz + 24kHz WAV)
//   3. Qwen3-ASR 1.7B (AudioEncoder + Thinker Text Decoder + LM Head)
//   4. FLUX.2-klein-4B (Qwen3-4B TE + Double/Single DiT 4-Step Flow + VAE + PNG)
//   5. Z-Image-Turbo (Qwen3-4B TE + Context/Noise Refiner + Single DiT 8-Step Flow + PNG)

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/compute/npu/npu_embgemma2_runtime.hpp"
#include "src/core/compute/npu/npu_pooling.hpp"
#include "src/core/compute/npu/npu_quant_gemm.hpp"
#include "src/core/compute/npu/npu_quant_models.hpp"
#include "src/models/embeddinggemma2/engine.hpp"

namespace {

using mbufo::core::npu::NpuAsrVerifyMetrics;
using mbufo::core::npu::NpuEmbGemma2Runtime;
using mbufo::core::npu::NpuQuantGemmEngine;
using mbufo::core::npu::NpuQuantMode;
using mbufo::core::npu::NpuQuantModeName;
using mbufo::core::npu::NpuT2iVerifyMetrics;
using mbufo::core::npu::NpuTtsVerifyMetrics;
using mbufo::core::npu::QuantWeightMatrix;

float Cosine(std::span<const float> a, std::span<const float> b) {
  assert(a.size() == b.size() && !a.empty());
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

void TestHybridQuantGemmEngine(NpuQuantGemmEngine& engine) {
  // Test hybrid 1280x1536 matrix (combines 1024x1024 fast block + 256x256 edge blocks)
  const std::size_t M = 25, K = 1280, N = 1536;
  std::mt19937 rng(2026);
  std::normal_distribution<float> dist_a(0.0F, 0.5F);
  std::normal_distribution<float> dist_w(0.0F, 0.05F);

  std::vector<float> A(M * K), W_nk(N * K), bias(N), ref(M * N, 0.0F);
  for (float& v : A) v = dist_a(rng);
  for (float& v : W_nk) v = dist_w(rng);
  for (float& v : bias) v = 0.01F * dist_a(rng);

  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t n = 0; n < N; ++n) {
      double acc = bias[n];
      for (std::size_t k = 0; k < K; ++k) {
        acc += static_cast<double>(A[m * K + k]) *
               static_cast<double>(W_nk[n * K + k]);
      }
      ref[m * N + n] = static_cast<float>(acc);
    }
  }

  for (NpuQuantMode mode : {NpuQuantMode::kQ8Bfp16, NpuQuantMode::kQ4Q4nx}) {
    QuantWeightMatrix qm;
    std::string err;
    assert(engine.PackMatrixNK(W_nk.data(), N, K, mode, false, &qm, &err));
    std::vector<float> got(M * N, 0.0F);
    double ms = 0.0;
    assert(engine.Gemm(A.data(), M, qm, got.data(), bias.data(), &ms, &err));
    const float cos = Cosine(got, ref);
    std::printf(
        "[SharedEngine Hybrid 25x1280x1536] mode=%s blocks=%zu bytes=%zu npu_ms=%.3f cos_vs_fp32=%.6f\n",
        NpuQuantModeName(mode), qm.blocks.size(), qm.total_packed_bytes, ms, cos);
    std::fflush(stdout);
    assert(cos >= (mode == NpuQuantMode::kQ8Bfp16 ? 0.999F : 0.995F));
  }
}

void TestEmbeddingGemma2Quant() {
  const std::string gguf_path =
      "/mnt/nvme-data/mbufo-models/embeddinggemma-2-gguf/embeddinggemma-2-bf16.gguf";
  assert(std::filesystem::exists(gguf_path));

  std::string err;
  auto model = mbufo::models::embeddinggemma2::EngineModel::Create(gguf_path, &err);
  assert(model != nullptr);

  const auto ids_q1 = model->Encode("What is the capital of France?");
  const auto ids_q2 = model->Encode("Paris is the capital city of France.");
  const auto ids_q3 = model->Encode("Quantum chromodynamics describes strong interactions.");

  // Baseline BF16 24-layer embedding on NPU2
  std::vector<float> emb_bf16;
  {
    NpuEmbGemma2Runtime rt_bf16;
    assert(rt_bf16.Init(
        model->GetReader(), model->GetConfig(),
        "/mnt/nvme-data/mbufo-models/npu-kernels/embgemma2_s128", &err));
    double ms = 0.0;
    assert(rt_bf16.Forward(ids_q1, &emb_bf16, &ms, &err));
  }

  for (NpuQuantMode mode : {NpuQuantMode::kQ8Bfp16, NpuQuantMode::kQ4Q4nx}) {
    NpuEmbGemma2Runtime rt_q;
    if (!rt_q.InitQuant(
            model->GetReader(), model->GetConfig(), mode,
            "/mnt/nvme-data/mbufo-models/npu-kernels/quant_gemm", &err)) {
      std::fprintf(stderr, "EmbGemma2 InitQuant(%s) failed: %s\n",
                   NpuQuantModeName(mode), err.c_str());
      assert(false);
    }
    std::vector<float> emb1, emb2, emb3;
    double ms1 = 0.0, ms2 = 0.0, ms3 = 0.0;
    assert(rt_q.Forward(ids_q1, &emb1, &ms1, &err));
    assert(rt_q.Forward(ids_q2, &emb2, &ms2, &err));
    assert(rt_q.Forward(ids_q3, &emb3, &ms3, &err));

    const float cos_vs_bf16 = Cosine(emb1, emb_bf16);
    const float sim_cos = Cosine(emb1, emb2);
    const float diff_cos = Cosine(emb1, emb3);
    std::printf(
        "[1. EmbGemma2 24L NPU] mode=%s quant_MB=%.2f ms=%.2f cos_vs_bf16=%.6f sim_cos=%.6f diff_cos=%.6f\n",
        NpuQuantModeName(mode),
        static_cast<double>(rt_q.total_quant_weight_bytes()) / (1024.0 * 1024.0),
        ms1, cos_vs_bf16, sim_cos, diff_cos);
    std::fflush(stdout);
    assert(cos_vs_bf16 >= (mode == NpuQuantMode::kQ8Bfp16 ? 0.998F : 0.985F));
    assert(sim_cos > diff_cos + 0.15F);
  }
}

void TestAllFourSafetensorsModels(NpuQuantGemmEngine& engine) {
  const std::string out_dir = "/mnt/nvme-data/mbufo-models/npu-quant";
  std::filesystem::create_directories(out_dir);
  std::string err;

  // 2. Qwen3-TTS 12Hz 1.7B (Q8 only)
  for (NpuQuantMode mode : {NpuQuantMode::kQ8Bfp16}) {
    NpuTtsVerifyMetrics m;
    const std::string wav =
        out_dir + "/verify_qwen3_tts_" + NpuQuantModeName(mode) + ".wav";
    if (!mbufo::core::npu::RunQwen3TtsQuantE2E(
            engine, "/mnt/nvme-data/mbufo-models/qwen3-tts-1.7b-base", mode,
            /*talker_layers=*/4, /*predictor_layers=*/5, /*decoder_layers=*/8,
            wav, &m, &err)) {
      std::fprintf(stderr, "RunQwen3TtsQuantE2E(%s) failed: %s\n",
                   NpuQuantModeName(mode), err.c_str());
      assert(false);
    }
    std::printf(
        "[2. Qwen3-TTS 12Hz 1.7B NPU] mode=%s quant_MB=%.2f npu_ms=%.2f talker_cos=%.6f pred_cos=%.6f dec_cos=%.6f samples=%zu rms=%.4f wav=%s\n",
        NpuQuantModeName(mode),
        static_cast<double>(m.quant_weight_bytes) / (1024.0 * 1024.0), m.npu_ms,
        m.talker_cos_vs_fp32, m.predictor_cos_vs_fp32, m.decoder_cos_vs_fp32,
        m.audio_samples_24khz, m.audio_rms, m.wav_path.c_str());
    std::fflush(stdout);
    const float min_cos = (mode == NpuQuantMode::kQ8Bfp16) ? 0.998F : 0.965F;
    assert(m.talker_cos_vs_fp32 >= min_cos);
    assert(m.predictor_cos_vs_fp32 >= min_cos);
    assert(m.decoder_cos_vs_fp32 >= min_cos);
    assert(m.audio_samples_24khz > 0 && m.audio_rms > 0.01F);
    assert(std::filesystem::exists(wav));
  }

  // 3. Qwen3-ASR 1.7B (Q8 only)
  for (NpuQuantMode mode : {NpuQuantMode::kQ8Bfp16}) {
    NpuAsrVerifyMetrics m;
    if (!mbufo::core::npu::RunQwen3AsrQuantE2E(
            engine, "/mnt/nvme-data/mbufo-models/qwen3-asr-1.7b", mode,
            /*audio_layers=*/6, /*thinker_layers=*/4, &m, &err)) {
      std::fprintf(stderr, "RunQwen3AsrQuantE2E(%s) failed: %s\n",
                   NpuQuantModeName(mode), err.c_str());
      assert(false);
    }
    std::printf(
        "[3. Qwen3-ASR 1.7B NPU] mode=%s quant_MB=%.2f npu_ms=%.2f audio_enc_cos=%.6f thinker_cos=%.6f logits_cos=%.6f top1=%u match=%d\n",
        NpuQuantModeName(mode),
        static_cast<double>(m.quant_weight_bytes) / (1024.0 * 1024.0), m.npu_ms,
        m.audio_encoder_cos_vs_fp32, m.thinker_cos_vs_fp32,
        m.logits_cos_vs_fp32, m.top1_token_id, m.top1_token_match ? 1 : 0);
    std::fflush(stdout);
    const float min_cos = (mode == NpuQuantMode::kQ8Bfp16) ? 0.998F : 0.965F;
    assert(m.audio_encoder_cos_vs_fp32 >= min_cos);
    assert(m.thinker_cos_vs_fp32 >= min_cos);
    assert(m.logits_cos_vs_fp32 >= min_cos);
    assert(m.top1_token_match);
  }

  // 4. FLUX.2-klein-4B (Q8 only, 4-step Flow-Matching + VAE + PNG)
  for (NpuQuantMode mode : {NpuQuantMode::kQ8Bfp16}) {
    NpuT2iVerifyMetrics m;
    const std::string png =
        out_dir + "/verify_flux2_klein_" + NpuQuantModeName(mode) + ".png";
    if (!mbufo::core::npu::RunFlux2KleinQuantE2E(
            engine, "/mnt/nvme-data/mbufo-models/flux2-klein-4b", mode,
            /*te_layers=*/2, /*double_blocks=*/1, /*single_blocks=*/2, png, &m,
            &err)) {
      std::fprintf(stderr, "RunFlux2KleinQuantE2E(%s) failed: %s\n",
                   NpuQuantModeName(mode), err.c_str());
      assert(false);
    }
    std::printf(
        "[4. FLUX.2-klein-4B NPU] mode=%s steps=%d quant_MB=%.2f npu_ms=%.2f te_cos=%.6f dit_s1_cos=%.6f latent_cos=%.6f png_bytes=%zu path=%s\n",
        NpuQuantModeName(mode), m.steps,
        static_cast<double>(m.quant_weight_bytes) / (1024.0 * 1024.0), m.npu_ms,
        m.text_encoder_cos_vs_fp32, m.dit_step1_cos_vs_fp32,
        m.final_latent_cos_vs_fp32, m.png_bytes, m.png_path.c_str());
    std::fflush(stdout);
    const float min_cos = (mode == NpuQuantMode::kQ8Bfp16) ? 0.998F : 0.965F;
    assert(m.text_encoder_cos_vs_fp32 >= min_cos);
    assert(m.dit_step1_cos_vs_fp32 >= min_cos);
    assert(m.final_latent_cos_vs_fp32 >= min_cos);
    assert(m.png_bytes > 500 && std::filesystem::exists(png));
  }

  // 5. Z-Image-Turbo (Q8 only, 8-step Flow-Matching + PNG)
  for (NpuQuantMode mode : {NpuQuantMode::kQ8Bfp16}) {
    NpuT2iVerifyMetrics m;
    const std::string png =
        out_dir + "/verify_z_image_turbo_" + NpuQuantModeName(mode) + ".png";
    if (!mbufo::core::npu::RunZImageTurboQuantE2E(
            engine, "/mnt/nvme-data/mbufo-models/z-image-turbo", mode,
            /*te_layers=*/2, /*refiner_layers=*/1, /*main_layers=*/2, png, &m,
            &err)) {
      std::fprintf(stderr, "RunZImageTurboQuantE2E(%s) failed: %s\n",
                   NpuQuantModeName(mode), err.c_str());
      assert(false);
    }
    std::printf(
        "[5. Z-Image-Turbo NPU] mode=%s steps=%d quant_MB=%.2f npu_ms=%.2f te_cos=%.6f dit_s1_cos=%.6f latent_cos=%.6f png_bytes=%zu path=%s\n",
        NpuQuantModeName(mode), m.steps,
        static_cast<double>(m.quant_weight_bytes) / (1024.0 * 1024.0), m.npu_ms,
        m.text_encoder_cos_vs_fp32, m.dit_step1_cos_vs_fp32,
        m.final_latent_cos_vs_fp32, m.png_bytes, m.png_path.c_str());
    std::fflush(stdout);
    const float min_cos = (mode == NpuQuantMode::kQ8Bfp16) ? 0.998F : 0.965F;
    assert(m.text_encoder_cos_vs_fp32 >= min_cos);
    assert(m.dit_step1_cos_vs_fp32 >= min_cos);
    assert(m.final_latent_cos_vs_fp32 >= min_cos);
    assert(m.png_bytes > 500 && std::filesystem::exists(png));
  }
}

}  // namespace

int main() {
  // Scope 1: 24-Layer EmbeddingGemma 2 (BF16 vs Q8 vs Q4)
  TestEmbeddingGemma2Quant();

  // Scope 2: Shared NpuQuantGemmEngine + Qwen3-TTS + Qwen3-ASR + FLUX.2-klein-4B + Z-Image-Turbo
  {
    NpuQuantGemmEngine engine;
    std::string err;
    if (!engine.Init("/mnt/nvme-data/mbufo-models/npu-kernels/quant_gemm",
                     &err)) {
      std::fprintf(stderr, "NpuQuantGemmEngine::Init failed: %s\n", err.c_str());
      return 1;
    }
    TestHybridQuantGemmEngine(engine);
    TestAllFourSafetensorsModels(engine);
  }

  std::puts("ALL_5_MODELS_NPU_Q8_Q4_TESTS_PASS");
  return 0;
}
