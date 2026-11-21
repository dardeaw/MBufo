#include "src/core/compute/npu/npu_admission.hpp"
#include "src/core/compute/npu/npu_device.hpp"
#include "src/core/compute/npu/npu_kernels.hpp"
#include "src/core/compute/npu/npu_kernels_attn.hpp"
#include "src/core/compute/npu/npu_pooling.hpp"
#include "src/core/compute/npu/npu_safetensors_bridge.hpp"
#include "src/core/compute/npu/npu_scheduler.hpp"
#include "src/models/bge_m3/npu_embed.hpp"
#include "src/models/bge_reranker/npu_score.hpp"
#include "src/models/d3_decide/d3_config.hpp"
#include "src/models/embeddinggemma2/npu_embed.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace mbufo::core::npu;

bool Near(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) <= eps;
}

void TestAvailableOpsAndValidators() {
  const auto ops = AvailableOps();
  assert(ops.size() == 11);

  auto find_op = [&](const char* name) -> const NpuOpInfo* {
    for (const auto& info : ops) {
      if (std::string(info.name) == name) return &info;
    }
    return nullptr;
  };

  assert(find_op("mm") != nullptr &&
         find_op("mm")->status == NpuOpStatus::kVerified);
  assert(find_op("rms_norm_eps") != nullptr &&
         find_op("rms_norm_eps")->status == NpuOpStatus::kVerified);
  assert(find_op("add_sized") != nullptr &&
         find_op("add_sized")->status == NpuOpStatus::kVerified);
  assert(find_op("softmax") != nullptr &&
         find_op("softmax")->status == NpuOpStatus::kVerified);
  assert(find_op("mha") != nullptr &&
         find_op("mha")->status == NpuOpStatus::kVerified);
  assert(find_op("datamovement.rope") != nullptr &&
         find_op("datamovement.rope")->status == NpuOpStatus::kVerified);
  assert(find_op("linalg.mha_softmax") != nullptr &&
         find_op("linalg.mha_softmax")->status == NpuOpStatus::kSdkOnly);
  assert(find_op("quant.q4nx_dequant") != nullptr &&
         find_op("quant.q4nx_dequant")->status == NpuOpStatus::kSdkOnly);

  assert(ValidateGemm(32, 256, 64, NpuDType::kBf16).empty());
  assert(ValidateGemm(32, 256, 64, NpuDType::kBfp16Ebs8).empty());
  assert(!ValidateGemm(32, 250, 64, NpuDType::kBfp16Ebs8).empty());
  assert(ValidateVectorOp(NpuOp::kRmsNorm, 64).empty());
  assert(!ValidateVectorOp(NpuOp::kRmsNorm, 24).empty());
  assert(ValidateRope(256, true).empty());
  assert(ValidateRope(512, true).empty());
  assert(!ValidateRope(16, true).empty());
  assert(ValidateRope(16, false).empty());

  // Q4NX pack validation: 32K standard AND upgraded 64K tile!
  Q4nxPackLayout l32{};
  assert(ValidateQ4nxPack(32, 256, 32, 128, 8, 8, 65536, &l32).empty());
  assert(l32.input_bytes == 5120 && l32.output_bytes == 9216);

  Q4nxPackLayout l64{};
  assert(ValidateQ4nxPack(64, 256, 32, 128, 8, 8, 65536, &l64).empty());
  assert(l64.total_l1_bytes == 28672);
  assert(ValidateQ4nxPack(64, 512, 32, 128, 8, 8, 65536).empty());
  // Exceeds 64 KiB CoreTile L1 when both input & output live together:
  assert(!ValidateQ4nxPack(64, 1024, 32, 128, 8, 8, 65536).empty());
  assert(!ValidateQ4nxPack(64, 256, 32, 128, 4, 8).empty());
}

void TestAttentionGuardsAndRefs() {
  for (std::size_t li = 0; li < 24; ++li) {
    const bool expect = (li == 5 || li == 11 || li == 17 || li == 23);
    assert(EmbGemma2LayerIsFull(li) == expect);
  }

  AttnHardwareBudget lb{};
  assert(ValidateSlidingAttn(EmbGemma2LocalSpec(128), &lb).empty());
  assert(lb.gqa_ratio == 2 && lb.q_tiles == 4 && lb.kv_tiles == 2);
  assert(lb.qkt_l1_bytes <= kNpu2CoreTileL1Bytes);
  assert(ValidateSlidingAttn(EmbGemma2LocalSpec(512)).empty());

  AttnHardwareBudget fb{};
  assert(ValidateSlidingAttn(EmbGemma2FullSpec(128), &fb).empty());
  assert(fb.gqa_ratio == 4 && fb.k_subtiles == 8);
  assert(ValidateSlidingAttn(EmbGemma2FullSpec(512)).empty());

  // BD / Shim / GQA / L1 / DMA guards catch unsafe configurations:
  {
    auto bad = EmbGemma2LocalSpec(128);
    bad.shim_batch_limit = 8;
    assert(!ValidateSlidingAttn(bad).empty());
  }
  {
    auto bad = EmbGemma2LocalSpec(130);  // unaligned seq_len
    assert(!ValidateSlidingAttn(bad).empty());
  }
  {
    auto bad = EmbGemma2LocalSpec(128);
    bad.num_kv_heads = 3;  // invalid GQA ratio
    assert(!ValidateSlidingAttn(bad).empty());
  }
  {
    auto bad = EmbGemma2LocalSpec(128);
    bad.core_in_channels = 3;  // >2 CoreTile DMA in channels
    assert(!ValidateSlidingAttn(bad).empty());
  }
  {
    auto bad = EmbGemma2LocalSpec(128);
    bad.m_q_tile = 128;
    bad.k_tile = 128;  // L1 overflow + seq alignment
    assert(!ValidateSlidingAttn(bad).empty());
  }

  assert(!ValidatePrefillFv(256, false).empty());
  assert(ValidatePrefillFv(512, true).empty());

  // Reference math smoke checks
  std::vector<float> rms_out;
  RefRmsNormRows({3.0f, 4.0f, 3.0f, 4.0f}, 2, 2, 1e-6, rms_out);
  const float expected_rms = 3.0f / std::sqrt(12.5f + 1e-6f);
  assert(Near(rms_out[0], expected_rms));

  std::vector<float> sm_out;
  RefSoftmaxRows({1.0f, 2.0f, 3.0f, 4.0f}, 1, 4, sm_out);
  float sm_sum = 0.0f;
  for (float v : sm_out) sm_sum += v;
  assert(Near(sm_sum, 1.0f, 1e-5f));

  // Two-halves RoPE identity + 90 deg rotation check
  std::vector<float> rope_out;
  const std::vector<float> rx{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> lut{0.0f, 1.0f, 0.0f, 1.0f};  // cos=0, sin=1
  RefRopeTwoHalves(rx, lut, 1, 4, rope_out);
  assert(Near(rope_out[0], -3.0f) && Near(rope_out[1], -4.0f) &&
         Near(rope_out[2], 1.0f) && Near(rope_out[3], 2.0f));

  // Bidirectional sliding-window masking changes output vs unmasked
  AttnLayerSpec swa;
  swa.seq_len = 4;
  swa.num_q_heads = 2;
  swa.num_kv_heads = 1;
  swa.head_dim = 4;
  swa.sliding_window = 1;
  swa.is_full = false;
  swa.qk_scale = 1.0f;
  std::vector<float> q(2 * 4 * 4, 1.0f);
  std::vector<float> k(1 * 4 * 4, 1.0f);
  std::vector<float> v(1 * 4 * 4, 0.0f);
  for (std::size_t t = 0; t < 4; ++t) {
    for (std::size_t d = 0; d < 4; ++d) v[t * 4 + d] = static_cast<float>(t);
  }
  std::vector<float> out_swa, out_full;
  RefBidirectionalGqaAttention(swa, q, k, v, out_swa);
  swa.is_full = true;
  RefBidirectionalGqaAttention(swa, q, k, v, out_full);
  // Row 0 in window=1 averages tokens {0,1} -> 0.5; full averages {0,1,2,3} -> 1.5
  assert(Near(out_swa[0], 0.5f) && Near(out_full[0], 1.5f));
}

void TestPoolingAndRouteSeams() {
  // CLS pooling + L2 normalize in bf16
  const std::vector<float> raw{3.0f, 4.0f, 99.0f, 99.0f};
  std::vector<std::uint16_t> raw_bf16(4);
  for (std::size_t i = 0; i < 4; ++i) raw_bf16[i] = FloatToBf16Rne(raw[i]);
  std::vector<float> cls_out;
  double pre_norm = 0.0;
  assert(NpuPoolAndNormalizeBf16(
      raw_bf16, 2, 2, mbufo::core::embedding::PoolingStrategy::kCls, cls_out,
      &pre_norm));
  assert(Near(static_cast<float>(pre_norm), 5.0f));
  assert(Near(cls_out[0], 0.6f) && Near(cls_out[1], 0.8f));

  // Mean pooling
  std::vector<float> mean_out;
  assert(NpuPoolAndNormalizeBf16(
      raw_bf16, 2, 2, mbufo::core::embedding::PoolingStrategy::kMean,
      mean_out));
  assert(Near(static_cast<float>(mbufo::core::embedding::VectorNorm(mean_out)),
              1.0f, 1e-5f));

  // EmbeddingGemma 2 post-process: RMSNorm -> Mean -> 512->768 proj -> L2
  const std::size_t S = 4, H = 512, O = 768;
  std::vector<std::uint16_t> h_bf16(S * H), nw_bf16(H), pw_bf16(O * H);
  for (std::size_t i = 0; i < h_bf16.size(); ++i) {
    h_bf16[i] = FloatToBf16Rne(static_cast<float>((i % 7) - 3) * 0.25f);
  }
  for (std::size_t i = 0; i < H; ++i) nw_bf16[i] = FloatToBf16Rne(1.0f);
  for (std::size_t i = 0; i < pw_bf16.size(); ++i) {
    pw_bf16[i] = FloatToBf16Rne(((i % 13) - 6) * 0.01f);
  }
  std::vector<float> eg_out;
  double eg_pre = 0.0;
  assert(NpuEmbGemma2PostProcessBf16(h_bf16, S, H, nw_bf16, 1e-6f, pw_bf16, O,
                                     eg_out, &eg_pre));
  assert(eg_out.size() == O && eg_pre > 0.0);
  assert(Near(static_cast<float>(mbufo::core::embedding::VectorNorm(eg_out)),
              1.0f, 1e-5f));

  // Route gating: safe kNone by default
  mbufo::npu::NpuLaneConfig cfg;
  assert(mbufo::models::bge_m3::npu::DecideNpuEmbed(cfg, 16) ==
         mbufo::models::bge_m3::npu::EmbedNpuRoute::kNone);
  assert(mbufo::models::embeddinggemma2::npu::DecideNpuEmbed(cfg, 128) ==
         mbufo::models::embeddinggemma2::npu::EmbedNpuRoute::kNone);
  cfg.embedding = true;
  assert(mbufo::models::bge_m3::npu::DecideNpuEmbed(cfg, 16, false, true) ==
         mbufo::models::bge_m3::npu::EmbedNpuRoute::kNone);
  assert(mbufo::models::bge_m3::npu::DecideNpuEmbed(cfg, 16, true, true) ==
         mbufo::models::bge_m3::npu::EmbedNpuRoute::kIron);
  assert(mbufo::models::embeddinggemma2::npu::DecideNpuEmbed(cfg, 128, true,
                                                            true) ==
         mbufo::models::embeddinggemma2::npu::EmbedNpuRoute::kIron);
}

void TestBfpAndSafetensorsBridge() {
  // Exact byte-for-byte parity vs aie.utils.bfp.encode(..., rounding="floor")
  const std::vector<float> v1{0.0f,  -0.0f,  1.0f,   -1.0f,
                              1e-8f, -1e-8f, 0.125f, -0.125f};
  const std::vector<std::uint8_t> g1{127, 0, 255, 64, 192, 0, 255, 8, 248};
  std::vector<std::uint8_t> enc1;
  assert(PackFloatToBfp16Ebs8(v1, enc1, false));
  assert(enc1 == g1);

  const std::vector<float> v2{-3.5f,   0.25f,  -0.001f, 100.0f,
                              -100.0f, 0.0f,   1e-20f,  -1e-20f};
  const std::vector<std::uint8_t> g2{133, 252, 0, 255, 100, 156, 0, 0, 255};
  std::vector<std::uint8_t> enc2;
  assert(PackFloatToBfp16Ebs8(v2, enc2, false));
  assert(enc2 == g2);

  std::vector<float> dec2;
  assert(UnpackBfp16Ebs8ToFloat(enc2, dec2));
  assert(Near(dec2[3], 100.0f, 1e-3f) && Near(dec2[4], -100.0f, 1e-3f));

  // Tiled transpose pack check
  const std::vector<std::uint16_t> mat{1, 2, 3, 4, 5, 6, 7, 8};  // 2x4 -> 4x2
  std::vector<std::uint16_t> packed;
  assert(PrepackBf16MatrixTiled(mat, 2, 4, 2, 2, true, packed));
  // Transposed 4x2 = [1,5, 2,6 | 3,7, 4,8]
  assert(packed == (std::vector<std::uint16_t>{1, 5, 2, 6, 3, 7, 4, 8}));

  // Live 1.49 GB EmbeddingGemma 2 safetensors integration check (if present on host)
  const std::string st_path =
      "/mnt/nvme-data/mbufo-models/embeddinggemma-2/model.safetensors";
  if (std::filesystem::exists(st_path)) {
    std::string err;
    auto idx = SafetensorsIndex::Open(st_path, &err);
    assert(idx.has_value());
    const auto plan = PlanEmbGemma2NpuShards(*idx, true, 64ull << 20);
    assert(plan.layer_count == 24);
    assert(plan.shards.size() == 3);
    assert(plan.host_only_bytes > (1024ull << 20));
    const auto* q0 =
        idx->Find("language_model.layers.0.self_attn.q_proj.weight");
    const auto* u5 = idx->Find("language_model.layers.5.mlp.up_proj.weight");
    assert(q0 != nullptr && u5 != nullptr);
    std::vector<std::uint8_t> chunk;
    assert(idx->ReadChunk(*q0, 0, 8192, chunk, &err));
    std::vector<std::uint16_t> u16(4096);
    std::memcpy(u16.data(), chunk.data(), 8192);
    std::vector<std::uint8_t> bfp_chunk;
    assert(PackBf16ToBfp16Ebs8(u16, bfp_chunk));
    assert(bfp_chunk.size() == 4608);
  }
}

void TestDeviceLockAndHrxSeam() {
  NpuDeviceLock dev;
  assert(dev.xclbins().Register("ffn", "/tmp/ffn.xclbin", nullptr));
  assert(dev.xclbins().Has("ffn"));
  assert(dev.xclbins().Resolve("ffn") == "/tmp/ffn.xclbin");

  mbufo::core::NpuAdmissionQueue queue;
  const auto t = queue.Submit("clientA", std::chrono::milliseconds{100});
  assert(t > 0);
  std::uint64_t ran_ticket = 0;
  assert(dev.PumpOnce(queue, [&](std::uint64_t id) { ran_ticket = id; }) == t);
  assert(ran_ticket == t);

#ifndef MBUFO_ENABLE_NPU
  std::string bind_err;
  assert(!HrxBind(&bind_err));
  assert(bind_err.find("kHrxUnbound") != std::string::npos);
#endif
}

void TestSharedAdmissionQueue() {
  using mbufo::core::AdmissionPolicy;
  using mbufo::core::AdmissionQueue;
  using mbufo::core::AdmissionRejectReason;

  AdmissionQueue<std::string> q(AdmissionPolicy{
      .max_pending = 4,
      .max_pending_per_client = 2,
      .request_timeout = std::chrono::milliseconds{0},
  });
  std::atomic<std::int64_t> gauge{0};
  q.SetQueuedDeltaCallback([&](std::int64_t delta) { gauge.fetch_add(delta); });

  const auto r1 = q.SubmitPayload("c1", "c1-a");
  const auto r2 = q.SubmitPayload("c1", "c1-b");
  const auto r3 = q.SubmitPayload("c1", "c1-c");
  assert(r1.ticket > 0 && r2.ticket > 0);
  assert(r3.ticket == 0 && r3.reject == AdmissionRejectReason::kClientQueueFull);

  const auto r4 = q.SubmitPayload("c2", "c2-a");
  const auto r5 = q.SubmitPayload("c3", "c3-a");
  const auto r6 = q.SubmitPayload("c4", "c4-a");
  assert(r4.ticket > 0 && r5.ticket > 0);
  assert(r6.ticket == 0 && r6.reject == AdmissionRejectReason::kQueueFull);
  assert(q.queued() == 4 && gauge.load() == 4);

  // Round-robin fairness across client buckets: c1-a, c2-a, c3-a, c1-b.
  std::vector<std::string> order;
  while (auto pop = q.AdmitNextPayload()) {
    assert(q.active() == 1);
    q.MarkRunning(pop->first);
    order.push_back(pop->second);
    q.Finish(pop->first);
    assert(q.active() == 0);
  }
  assert((order == std::vector<std::string>{"c1-a", "c2-a", "c3-a", "c1-b"}));
  assert(q.queued() == 0 && q.pending() == 0 && gauge.load() == 0);

  // Single-ticket Cancel + Wait, and CancelAll drain.
  const auto tc1 = q.SubmitPayload("c1", "x1");
  const auto tc2 = q.SubmitPayload("c2", "x2");
  assert(q.Cancel(tc1.ticket, false));
  const auto w1 = q.Wait(tc1.ticket);
  assert(!w1.admitted && w1.reason == AdmissionRejectReason::kCancelled);
  q.Finish(tc1.ticket);
  auto drained = q.CancelAll();
  assert(drained.size() == 1 && drained[0].first == tc2.ticket);
  assert(q.queued() == 0 && gauge.load() == 0);

  // Per-request timeout.
  const auto t_exp = q.SubmitWithTimeout("c1", "expire",
                                         std::chrono::milliseconds{15});
  assert(t_exp.ticket > 0);
  const auto w_exp = q.Wait(t_exp.ticket);
  assert(!w_exp.admitted && w_exp.reason == AdmissionRejectReason::kTimeout);
  q.Finish(t_exp.ticket);
  assert(q.queued() == 0 && gauge.load() == 0);

  // Sticky device-lost circuit breaker.
  q.SetDeviceLost();
  assert(q.device_lost());
  const auto r_lost = q.SubmitPayload("c1", "blocked");
  assert(r_lost.ticket == 0 && r_lost.reject == AdmissionRejectReason::kDeviceLost);
}

void TestNpuSchedulerPumpAndCoalesce() {
  using mbufo::core::AdmissionRejectReason;

  // Disabled shortcut never spawns or accepts work.
  {
    NpuScheduler disabled_sched(mbufo::npu::NpuLaneConfig{});
    const auto res = disabled_sched.Submit(
        NpuWorkDesc{.lane = NpuWorkLane::kEmbedding, .seq_len = 8},
        []() { return NpuWorkResult{.ok = true}; });
    assert(!res.ok && res.ticket_id == 0 && res.error == "npu-disabled");
    assert(disabled_sched.pending() == 0);
  }

  // Pump + external NpuDeviceLock contention + cancel/timeout + coalescing.
  {
    NpuDeviceLock dev_lock;
    mbufo::npu::NpuLaneConfig cfg;
    cfg.embedding = true;
    cfg.reranker = true;
    cfg.decide = true;
    mbufo::core::NpuAdmissionPolicy policy{
        .max_pending = 5,
        .max_pending_per_client = 3,
        .request_timeout = std::chrono::milliseconds{0},
    };
    NpuSchedulerOptions opts{
        .max_coalesce_batch = 4,
        .lock_acquire_timeout = std::chrono::milliseconds{5},
        .idle_poll_interval = std::chrono::milliseconds{5},
    };
    std::atomic<int> coalesce_calls{0};
    NpuScheduler sched(
        cfg, policy, &dev_lock, opts,
        [&](std::vector<NpuBatchEntry>& entries) {
          // Pump must own NpuDeviceLock while running the coalesced batch
          // (checked from a helper thread to avoid recursive timed_mutex UB).
          std::thread probe([&] {
            NpuDeviceLock::Hold h(&dev_lock, std::chrono::milliseconds{0});
            assert(!h.holds());
          });
          probe.join();
          coalesce_calls.fetch_add(1);
          for (std::size_t i = 0; i < entries.size(); ++i) {
            entries[i].result.ok = true;
            entries[i].result.error =
                "coalesced:" + std::to_string(entries.size()) + ":" +
                entries[i].desc.client;
          }
        });

    // External holder blocks admission until released.
    NpuDeviceLock::Hold ext_hold(&dev_lock, std::chrono::milliseconds{50});
    assert(ext_hold.holds());

    auto e1 = sched.SubmitAsync(
        NpuWorkDesc{.lane = NpuWorkLane::kEmbedding,
                    .seq_len = 64,
                    .client = "emb_client",
                    .coalesce_key = "emb-64"},
        []() { return NpuWorkResult{.ok = true}; });
    auto e2 = sched.SubmitAsync(
        NpuWorkDesc{.lane = NpuWorkLane::kEmbedding,
                    .seq_len = 64,
                    .client = "emb_client",
                    .coalesce_key = "emb-64"},
        []() { return NpuWorkResult{.ok = true}; });
    auto e3 = sched.SubmitAsync(
        NpuWorkDesc{.lane = NpuWorkLane::kEmbedding,
                    .seq_len = 64,
                    .client = "emb_client",
                    .coalesce_key = "emb-64"},
        []() { return NpuWorkResult{.ok = true}; });
    // Per-client cap check while blocked on device lock.
    auto e_over = sched.Submit(
        NpuWorkDesc{.lane = NpuWorkLane::kEmbedding,
                    .seq_len = 64,
                    .client = "emb_client",
                    .coalesce_key = "emb-64"},
        []() { return NpuWorkResult{.ok = true}; });
    assert(!e_over.ok &&
           e_over.reject_reason == AdmissionRejectReason::kClientQueueFull);

    // Enqueue 1 cancel + 1 short timeout -> hits max_pending=5.
    auto h_cancel = sched.SubmitAsync(
        NpuWorkDesc{.lane = NpuWorkLane::kEmbedding,
                    .seq_len = 64,
                    .client = "cancel_client",
                    .coalesce_key = "emb-64"},
        []() { return NpuWorkResult{.ok = true}; });
    auto h_timeout = sched.SubmitAsync(
        NpuWorkDesc{.lane = NpuWorkLane::kEmbedding,
                    .seq_len = 64,
                    .client = "timeout_client",
                    .coalesce_key = "emb-64",
                    .timeout = std::chrono::milliseconds{15}},
        []() { return NpuWorkResult{.ok = true}; });
    assert(e1.valid() && e2.valid() && e3.valid() && h_cancel.valid() &&
           h_timeout.valid());

    // Queue full (5 items pending).
    auto q_full = sched.Submit(
        NpuWorkDesc{.lane = NpuWorkLane::kReranker,
                    .seq_len = 16,
                    .client = "full_client"},
        []() { return NpuWorkResult{.ok = true}; });
    assert(!q_full.ok && q_full.reject_reason == AdmissionRejectReason::kQueueFull);

    // Pump should not admit while external lock is held.
    assert(sched.active() == 0);
    assert(h_cancel.Cancel());
    std::this_thread::sleep_for(std::chrono::milliseconds{35});
    const auto r_cancel = h_cancel.Wait();
    const auto r_timeout = h_timeout.Wait();
    assert(!r_cancel.ok && r_cancel.reject_reason == AdmissionRejectReason::kCancelled);
    assert(!r_timeout.ok && r_timeout.reject_reason == AdmissionRejectReason::kTimeout);

    // Different lane must NOT coalesce with kEmbedding items.
    auto rr = sched.SubmitAsync(
        NpuWorkDesc{.lane = NpuWorkLane::kReranker,
                    .seq_len = 16,
                    .client = "rerank_client",
                    .coalesce_key = "rerank-16"},
        []() { return NpuWorkResult{.ok = true, .error = "single-rerank"}; });
    assert(rr.valid());

    // Release external device lock: pump admits 3-item coalesced batch + 1 rerank.
    ext_hold.Reset();
    const auto re1 = e1.Wait();
    const auto re2 = e2.Wait();
    const auto re3 = e3.Wait();
    const auto r_rr = rr.Wait();
    assert(re1.ok && re2.ok && re3.ok);
    assert(re1.coalesced_batch_size == 3 && re2.coalesced_batch_size == 3 &&
           re3.coalesced_batch_size == 3);
    assert(coalesce_calls.load() == 1);
    assert(r_rr.ok && r_rr.coalesced_batch_size == 1 &&
           r_rr.error == "single-rerank");
    assert(sched.pending() == 0);
  }

  // Device-lost propagation breaks queued + new requests immediately.
  {
    NpuDeviceLock dev_lock;
    NpuDeviceLock::Hold ext_hold(&dev_lock, std::chrono::milliseconds{50});
    mbufo::npu::NpuLaneConfig dec_cfg;
    dec_cfg.decide = true;
    NpuScheduler sched(dec_cfg, {}, &dev_lock);
    auto h = sched.SubmitAsync(
        NpuWorkDesc{.lane = NpuWorkLane::kDecide, .seq_len = 1},
        []() { return NpuWorkResult{.ok = true}; });
    assert(h.valid());
    sched.SetDeviceLost();
    assert(sched.device_lost());
    const auto res_wait = h.Wait();
    assert(!res_wait.ok && res_wait.reject_reason == AdmissionRejectReason::kDeviceLost);
    const auto res_after = sched.Submit(
        NpuWorkDesc{.lane = NpuWorkLane::kDecide, .seq_len = 1},
        []() { return NpuWorkResult{.ok = true}; });
    assert(!res_after.ok &&
           res_after.reject_reason == AdmissionRejectReason::kDeviceLost);
  }

  // Reranker + Decide seam checks.
  {
    mbufo::npu::NpuLaneConfig off_cfg;
    assert(mbufo::models::bge_reranker::npu::DecideNpuScore(off_cfg, 16, true,
                                                             true) ==
           mbufo::models::bge_reranker::npu::ScoreNpuRoute::kNone);
    mbufo::npu::NpuLaneConfig rr_cfg;
    rr_cfg.reranker = true;
    assert(mbufo::models::bge_reranker::npu::DecideNpuScore(rr_cfg, 0, true,
                                                             true) ==
           mbufo::models::bge_reranker::npu::ScoreNpuRoute::kNone);
    assert(mbufo::models::bge_reranker::npu::DecideNpuScore(rr_cfg, 2049, true,
                                                             true) ==
           mbufo::models::bge_reranker::npu::ScoreNpuRoute::kNone);
    assert(mbufo::models::bge_reranker::npu::DecideNpuScore(rr_cfg, 16, false,
                                                             true) ==
           mbufo::models::bge_reranker::npu::ScoreNpuRoute::kNone);
    assert(mbufo::models::bge_reranker::npu::DecideNpuScore(rr_cfg, 14, true,
                                                             true) ==
           mbufo::models::bge_reranker::npu::ScoreNpuRoute::kIron);

    mbufo::models::d3_decide::DecideRequest d3_req;
    mbufo::models::d3_decide::DecideQuestion d3_q;
    d3_q.type = "noul";
    d3_q.criteria = {{"false", "no"}, {"true", "yes"}};
    d3_req.questions["q1"] = std::move(d3_q);
    mbufo::npu::NpuLaneConfig d3_cfg;
    d3_cfg.decide = true;
    NpuScheduler sched(d3_cfg);
    const auto off_d3 =
        mbufo::models::d3_decide::RunDecideSync(sched, d3_req, false);
    assert(!off_d3.ok && off_d3.error == "npu-disabled-or-unready");
    const auto on_d3 = mbufo::models::d3_decide::RunDecideSync(
        sched, d3_req, true, []() {
          return NpuWorkResult{.ok = true, .error = "d3-ran"};
        });
    assert(on_d3.ok && on_d3.error == "d3-ran");
  }
}

}  // namespace

int main() {
  TestAvailableOpsAndValidators();
  TestAttentionGuardsAndRefs();
  TestPoolingAndRouteSeams();
  TestBfpAndSafetensorsBridge();
  TestDeviceLockAndHrxSeam();
  TestSharedAdmissionQueue();
  TestNpuSchedulerPumpAndCoalesce();
  std::cout << "ALL_NPU_CORE_TESTS_PASS\n";
  return 0;
}
