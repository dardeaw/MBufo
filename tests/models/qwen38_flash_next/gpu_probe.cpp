// Dumps trunk prefill logits and optionally compares them with the F32 oracle.
// Use the production CLI for generation and the session test for replay.
//
// gpu_probe --model FIRST_SHARD.gguf --prompt TEXT [--reference]
//           [--batch T] [--context N] [--dump logits.bin]
// Independent predictor oracle: --mtp-model MTP.gguf --mtp-audit
// MTP cost calibration: --mtp-model MTP.gguf --cost-audit C (0 = all)
//                      [--depth N] (default: 0, 4096, 32768)
// Matched-token decode: --prompt-file P --decode-file C --dump rows.bin
//                       [--decode-ids ids.bin] (prefill P, then feed C's
//                       tokens one at a time; row i predicts the token
//                       after ids[i], so score row i against ids[i + 1])
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next;

void AuditMtp(q::rocm::Executor& executor, const q::rocm::DeviceModel& device,
              const q::ModelWeights& weights, const q::MtpWeights& mtp,
              const gufo::tokenization::QwenTokenizer& tokenizer,
              const std::filesystem::path& path);
void AuditMtpCosts(q::rocm::Executor& executor,
                   const gufo::tokenization::QwenTokenizer& tokenizer,
                   std::span<const std::uint32_t> depths,
                   std::uint32_t concurrency);

namespace {

struct Stats {
  std::uint32_t argmax_a;
  std::uint32_t argmax_b;
  double kl;
  double max_abs;
};

Stats Compare(const float* a, const float* b, std::size_t n) {
  std::vector<double> pa(n);
  std::vector<double> pb(n);
  const float ma = *std::max_element(a, a + n);
  const float mb = *std::max_element(b, b + n);
  double sa = 0.0;
  double sb = 0.0;
  double max_abs = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    pa[i] = std::exp(static_cast<double>(a[i] - ma));
    pb[i] = std::exp(static_cast<double>(b[i] - mb));
    sa += pa[i];
    sb += pb[i];
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(a[i] - b[i])));
  }
  double kl = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    pa[i] /= sa;
    pb[i] /= sb;
    kl += pa[i] * (std::log(pa[i] + 1e-30) - std::log(pb[i] + 1e-30));
  }
  return {static_cast<std::uint32_t>(std::max_element(a, a + n) - a),
          static_cast<std::uint32_t>(std::max_element(b, b + n) - b), kl,
          max_abs};
}

}  // namespace

namespace {
// Mixed-batch audit: session A decodes one token while session B prefills a
// 4-token chunk, first in one ForwardBatch (mixed attention path), then on
// fresh sessions via separate Forwards. Compares all 5 logit rows and
// requires the mixed counter to advance (no vacuous pass).
// Text-level mixed-batch check: two decode sessions stepped together through
// ForwardBatch (mixed path) vs separate Forwards (legacy narrow). Compares
// greedy token sequences; family-sized logit deltas are fine as long as the
// argmax text agrees.
void TextMixed(q::rocm::Executor& executor,
               gufo::tokenization::QwenTokenizer& tokenizer,
               const q::ModelWeights& weights) {
  const auto& c = weights.config;
  const std::uint32_t context = 256;
  const auto idsA = tokenizer.Encode("The capital of France is Paris. And");
  const auto idsB = tokenizer.Encode("Quantum mechanics describes nature.");
  std::string error;
  auto make = [&]() {
    auto s = executor.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                    context, &error);
    if (!s)
      throw std::runtime_error(std::string("text session failed: ") + error);
    return s;
  };
  auto prefill = [&](auto& session, const std::vector<std::int32_t>& toks) {
    std::vector<float> out(toks.size() * c.vocab_size);
    if (!executor.Forward(
            *session, std::span<const std::int32_t>(toks.data(), toks.size()),
            static_cast<std::uint32_t>(toks.size()), out.data(),
            q::rocm::Executor::ForwardMode::kPrefill, &error, nullptr,
            std::span<const std::int32_t>()))
      throw std::runtime_error("text prefill failed: " + error);
  };
  auto argmax = [&](const float* logits) {
    std::uint32_t best = 0;
    for (std::uint32_t i = 1; i < c.vocab_size; ++i)
      if (logits[i] > logits[best])
        best = i;
    return static_cast<std::int32_t>(best);
  };
  auto sA = make();
  auto sB = make();
  auto sAr = make();
  auto sBr = make();
  const std::vector<std::int32_t> pA(idsA.begin(), idsA.end());
  const std::vector<std::int32_t> pB(idsB.begin(), idsB.end());
  prefill(sA, pA);
  prefill(sAr, pA);
  prefill(sB, pB);
  prefill(sBr, pB);
  std::int32_t tA = pA.back(), tB = pB.back(), tAr = pA.back(), tBr = pB.back();
  std::vector<std::int32_t> seqA, seqB, seqAr, seqBr;
  const std::uint64_t before = executor.mixed_batches();
  for (int step = 0; step < 30; ++step) {
    std::vector<q::rocm::Executor::BatchItem> items;
    items.push_back({sA.get(), std::span<const std::int32_t>(&tA, 1), false});
    items.push_back({sB.get(), std::span<const std::int32_t>(&tB, 1), false});
    if (!executor.ForwardBatch(items, &error))
      throw std::runtime_error("text ForwardBatch failed: " + error);
    std::vector<float> mA(c.vocab_size), mB(c.vocab_size);
    if (!executor.SelectBatchLogits(0, 1, mA.data(), &error) ||
        !executor.SelectBatchLogits(1, 1, mB.data(), &error))
      throw std::runtime_error("text logits select failed: " + error);
    tA = argmax(mA.data());
    tB = argmax(mB.data());
    seqA.push_back(tA);
    seqB.push_back(tB);
    std::vector<float> rA(c.vocab_size), rB(c.vocab_size);
    if (!executor.Forward(*sAr, std::span<const std::int32_t>(&tAr, 1), 1,
                          rA.data(), q::rocm::Executor::ForwardMode::kDecode,
                          &error))
      throw std::runtime_error("text ref decode failed: " + error);
    if (!executor.Forward(*sBr, std::span<const std::int32_t>(&tBr, 1), 1,
                          rB.data(), q::rocm::Executor::ForwardMode::kDecode,
                          &error))
      throw std::runtime_error("text ref decode failed: " + error);
    tAr = argmax(rA.data());
    tBr = argmax(rB.data());
    seqAr.push_back(tAr);
    seqBr.push_back(tBr);
    // Teacher-forcing: identical histories every step, so agreement counts
    // per-step top-1 matches (family noise) instead of chaotic divergence.
    tA = tAr;
    tB = tBr;
  }
  const std::uint64_t after = executor.mixed_batches();
  auto match = [](const std::vector<std::int32_t>& a,
                  const std::vector<std::int32_t>& b) {
    int n = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
      n += (a[i] == b[i]);
    return n;
  };
  std::cout << "  text mixed batches served: " << (after - before) << "\n";
  std::cout << "  text agree A " << match(seqA, seqAr) << "/30 B "
            << match(seqB, seqBr) << "/30\n";
  std::cout << "  text mixedA:";
  for (auto id : seqA)
    std::cout << " " << id;
  std::cout << "\n  text refA:  ";
  for (auto id : seqAr)
    std::cout << " " << id;
  std::cout << "\n  text mixedB:";
  for (auto id : seqB)
    std::cout << " " << id;
  std::cout << "\n  text refB:  ";
  for (auto id : seqBr)
    std::cout << " " << id;
  std::cout << "\n";
  if (after == before)
    throw std::runtime_error("text mixed path did not fire");
}

// Big-batch envelope check (P0 caps): N decode sessions plus one chunk ride
// a single ForwardBatch. The 256-row chunk exceeds the mixed kernel width so
// the tail must take the legacy per-item path (counter stays); the 8-row
// chunk must fire mixed. Both compare report-only (phase/family gaps apply).
void BigBatch(q::rocm::Executor& executor,
              gufo::tokenization::QwenTokenizer& tokenizer,
              const q::ModelWeights& weights, std::uint32_t chunk_rows,
              bool expect_mixed) {
  const auto& c = weights.config;
  const std::uint32_t context = 1024;
  std::string big;
  for (int i = 0; i < 40; ++i)
    big += "The quick brown fox jumps over the lazy dog. ";
  const auto ids = tokenizer.Encode(big);
  if (ids.size() < chunk_rows + 8)
    throw std::runtime_error("big-batch prompt too short");
  std::string error;
  auto make = [&]() {
    auto s = executor.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                    context, &error);
    if (!s)
      throw std::runtime_error(std::string("big session failed: ") + error);
    return s;
  };
  constexpr int kDecodes = 8;
  std::vector<decltype(make())> ds, dr;
  for (int i = 0; i < kDecodes; ++i) {
    ds.push_back(make());
    dr.push_back(make());
  }
  auto sC = make();
  auto sCr = make();
  // tiny decode histories
  std::vector<std::int32_t> seed(ids.begin(), ids.begin() + 8);
  auto runrows = [&](auto& s, const std::vector<std::int32_t>& toks, bool pre) {
    std::vector<float> o(toks.size() * c.vocab_size);
    if (!executor.Forward(
            *s, std::span<const std::int32_t>(toks.data(), toks.size()),
            static_cast<std::uint32_t>(toks.size()), o.data(),
            pre ? q::rocm::Executor::ForwardMode::kPrefill
                : q::rocm::Executor::ForwardMode::kDecode,
            &error, nullptr, std::span<const std::int32_t>()))
      throw std::runtime_error("big-batch setup failed: " + error);
  };
  for (int i = 0; i < kDecodes; ++i) {
    runrows(ds[i], seed, true);
    runrows(dr[i], seed, true);
  }
  std::vector<std::int32_t> chunk(ids.begin(), ids.begin() + chunk_rows);
  // chunk sessions start empty; the batch appends the whole chunk at pos 0.
  // references: twin prefill of the same chunk.
  const std::uint64_t before = executor.mixed_batches();
  const auto bt0 = std::chrono::steady_clock::now();
  {
    std::vector<q::rocm::Executor::BatchItem> items;
    for (int i = 0; i < kDecodes; ++i)
      items.push_back({ds[i].get(),
                       std::span<const std::int32_t>(seed.data() + 7, 1),
                       false});
    items.push_back({sC.get(),
                     std::span<const std::int32_t>(chunk.data(), chunk.size()),
                     false});
    if (!executor.ForwardBatch(items, &error))
      throw std::runtime_error("big ForwardBatch failed: " + error);
    std::vector<float> tmp(c.vocab_size);
    for (int i = 0; i < kDecodes; ++i)
      if (!executor.SelectBatchLogits(i, 1, tmp.data(), &error))
        throw std::runtime_error("big logits select failed: " + error);
  }
  const auto bt1 = std::chrono::steady_clock::now();
  const std::uint64_t after = executor.mixed_batches();
  std::cout << "  big chunk=" << chunk_rows
            << " rows=" << (kDecodes + chunk_rows)
            << " sessions=" << (kDecodes + 1)
            << " mixed_fired=" << (after - before) << " batch_ms="
            << std::chrono::duration<double, std::milli>(bt1 - bt0).count()
            << "\n";
  // Correctness: reference twin prefill of the chunk, then both sides take
  // one decode step with the same token (catches bad GDN history / MoE / KV).
  // Decode rows compare directly against separate forwards (same plumbing).
  {
    std::vector<float> o(c.vocab_size);
    if (!executor.Forward(
            *sCr, std::span<const std::int32_t>(chunk.data(), chunk.size()), 1,
            o.data(), q::rocm::Executor::ForwardMode::kPrefill, &error, nullptr,
            std::span<const std::int32_t>()))
      throw std::runtime_error("big reference prefill failed: " + error);
    double worst_d = 0.0;
    for (int i = 0; i < kDecodes; ++i) {
      std::vector<float> m(1 * c.vocab_size), r(1 * c.vocab_size);
      if (!executor.SelectBatchLogits(i, 1, m.data(), &error))
        throw std::runtime_error("big re-select failed: " + error);
      if (!executor.Forward(
              *dr[i], std::span<const std::int32_t>(seed.data() + 7, 1), 1,
              r.data(), q::rocm::Executor::ForwardMode::kDecode, &error))
        throw std::runtime_error("big ref decode failed: " + error);
      for (std::size_t j = 0; j < m.size(); ++j)
        worst_d = std::max(worst_d, std::abs(double(m[j]) - double(r[j])));
    }
    const std::int32_t nt = chunk.back();
    std::vector<float> m1(c.vocab_size), r1(c.vocab_size);
    if (!executor.Forward(*sC, std::span<const std::int32_t>(&nt, 1), 1,
                          m1.data(), q::rocm::Executor::ForwardMode::kDecode,
                          &error))
      throw std::runtime_error("big continuation failed: " + error);
    if (!executor.Forward(*sCr, std::span<const std::int32_t>(&nt, 1), 1,
                          r1.data(), q::rocm::Executor::ForwardMode::kDecode,
                          &error))
      throw std::runtime_error("big ref continuation failed: " + error);
    double worst_c = 0.0;
    for (std::size_t j = 0; j < m1.size(); ++j)
      worst_c = std::max(worst_c, std::abs(double(m1[j]) - double(r1[j])));
    std::cout << "  big decode_rows worst " << worst_d << " continuation worst "
              << worst_c << "\n";
    bool finite = std::isfinite(worst_d) && std::isfinite(worst_c);
    if (!finite)
      throw std::runtime_error("big-batch non-finite output");
  }
  if (expect_mixed && after == before)
    throw std::runtime_error("big-batch mixed path did not fire");
  if (!expect_mixed && after != before)
    throw std::runtime_error("big-batch mixed fired unexpectedly");
}

// MTP edition (P1): speculative sessions prove the in-batch predictor-KV
// tail. MtpPosition must track the tracked trunk position exactly like legacy
// (tail ran), for the chunk and (unchanged) for decodes. Needs --mtp-model.
void MtpBatch(q::rocm::Executor& executor,
              gufo::tokenization::QwenTokenizer& tokenizer,
              const q::ModelWeights& weights, std::uint32_t kChunk) {
  if (!executor.has_mtp())
    throw std::runtime_error("MX_MTP needs --mtp-model");
  const auto& c = weights.config;
  const std::uint32_t context = 512;
  constexpr int kDecodes = 2;
  std::string big;
  for (int i = 0; i < 10; ++i)
    big += "The quick brown fox jumps over the lazy dog. ";
  const auto ids = tokenizer.Encode(big);
  if (ids.size() < kChunk + 8)
    throw std::runtime_error("mtp prompt too short");
  std::string error;
  auto make = [&]() {
    auto s = executor.CreateSession(gufo::core::SessionMode::kSpeculative,
                                    context, &error);
    if (!s)
      throw std::runtime_error(std::string("mtp session failed: ") + error);
    return s;
  };
  std::vector<std::int32_t> seed(ids.begin(), ids.begin() + 8);
  std::vector<std::int32_t> chunk(ids.begin(), ids.begin() + kChunk);
  auto runrows = [&](auto& s, const std::vector<std::int32_t>& toks, bool pre) {
    std::vector<float> o((pre ? 1 : 1) * c.vocab_size);
    if (!executor.Forward(
            *s, std::span<const std::int32_t>(toks.data(), toks.size()), 1,
            o.data(),
            pre ? q::rocm::Executor::ForwardMode::kPrefill
                : q::rocm::Executor::ForwardMode::kDecode,
            &error, nullptr, std::span<const std::int32_t>()))
      throw std::runtime_error("mtp setup failed: " + error);
  };
  std::vector<decltype(make())> ds, dr;
  for (int i = 0; i < kDecodes; ++i) {
    ds.push_back(make());
    dr.push_back(make());
    runrows(ds[i], seed, true);
    runrows(dr[i], seed, true);
  }
  auto sC = make();
  auto sCr = make();
  // tracked positions: decodes at 8, chunks at 0
  const std::uint64_t before = executor.mixed_batches();
  const auto bt0 = std::chrono::steady_clock::now();
  {
    const bool alone = std::getenv("MX_MTP_ALONE") != nullptr;
    const bool no_chunk = std::getenv("MX_MTP_NOCHUNK") != nullptr;
    std::vector<q::rocm::Executor::BatchItem> items;
    if (!alone) {
      for (int i = 0; i < kDecodes; ++i) {
        items.push_back({ds[i].get(),
                         std::span<const std::int32_t>(seed.data() + 7, 1),
                         false});
      }
    }
    if (!no_chunk || alone) {
      items.push_back(
          {sC.get(), std::span<const std::int32_t>(chunk.data(), chunk.size()),
           false});
    }
    if (!executor.ForwardBatch(items, &error))
      throw std::runtime_error(std::string("mtp ForwardBatch failed: ") +
                               error);
    if (no_chunk)
      return;
  }
  const auto bt1 = std::chrono::steady_clock::now();
  const std::uint64_t after = executor.mixed_batches();
  // predictor invariant: chunk MtpPosition == 0 + 32; decodes unchanged at 8.
  const auto mpC = executor.MtpPosition(*sC);
  std::cout << "  mtp chunk mtp_pos=" << mpC << " expect=" << kChunk
            << " mixed_fired=" << (after - before) << " batch_ms="
            << std::chrono::duration<double, std::milli>(bt1 - bt0).count()
            << "\n";
  // Parity with the legacy single-Forward tail (MtpForward over tokens[1:]):
  // mp trails trunk position by one; the engine-level DraftCatchUp (Feed
  // protocol, P2) consumes the frontier on the next call. P1 contract ends
  // here.
  if (mpC + 1 != kChunk)
    throw std::runtime_error("mtp predictor tail did not run for the chunk");
  for (int i = 0; i < kDecodes; ++i) {
    const auto mp = executor.MtpPosition(*ds[i]);
    if (mp + 1 != 8)
      throw std::runtime_error("mtp decode predictor position moved");
  }
  // reference twin + continuation compare (report-only, prepare gap applies)
  {
    std::vector<float> o(c.vocab_size);
    if (!executor.Forward(
            *sCr, std::span<const std::int32_t>(chunk.data(), chunk.size()), 1,
            o.data(), q::rocm::Executor::ForwardMode::kPrefill, &error, nullptr,
            std::span<const std::int32_t>()))
      throw std::runtime_error("mtp reference prefill failed: " + error);
    if (executor.MtpPosition(*sCr) + 1 != kChunk)
      throw std::runtime_error("reference predictor position unexpected");
    const std::int32_t nt = chunk.back();
    std::vector<float> m1(c.vocab_size), r1(c.vocab_size);
    if (!executor.Forward(*sC, std::span<const std::int32_t>(&nt, 1), 1,
                          m1.data(), q::rocm::Executor::ForwardMode::kDecode,
                          &error))
      throw std::runtime_error("mtp continuation failed: " + error);
    if (!executor.Forward(*sCr, std::span<const std::int32_t>(&nt, 1), 1,
                          r1.data(), q::rocm::Executor::ForwardMode::kDecode,
                          &error))
      throw std::runtime_error("mtp ref continuation failed: " + error);
    double worst_c = 0.0;
    for (std::size_t j = 0; j < m1.size(); ++j)
      worst_c = std::max(worst_c, std::abs(double(m1[j]) - double(r1[j])));
    std::cout << "  mtp continuation worst " << worst_c << "\n";
    if (!std::isfinite(worst_c))
      throw std::runtime_error("mtp non-finite output");
  }
}

void AuditMixed(q::rocm::Executor& executor,
                gufo::tokenization::QwenTokenizer& tokenizer,
                const q::ModelWeights& weights) {
  if (::setenv("GUFO_MIXED_BATCH",
               std::getenv("GUFO_MIXED_BATCH") ? std::getenv("GUFO_MIXED_BATCH")
                                               : "1",
               1) != 0)
    throw std::runtime_error("cannot set GUFO_MIXED_BATCH");
  const auto& c = weights.config;
  const std::uint32_t context = 256;
  auto stretch = [&](const std::string& text, std::size_t n) {
    const auto ids = tokenizer.Encode(text);
    if (ids.empty())
      throw std::runtime_error("empty prompt encoding");
    std::vector<std::int32_t> out;
    while (out.size() < n)
      for (const auto id : ids) {
        out.push_back(static_cast<std::int32_t>(id));
        if (out.size() == n)
          break;
      }
    return out;
  };
  // A: 32 prefill + 1 decode; B: 48 prefill + 4-token chunk.
  const auto tokA = stretch("The capital of France is Paris.", 33);
  const auto tokB =
      stretch("Quantum mechanics describes nature at small scales.", 52);
  std::string error;
  auto make = [&]() {
    auto s = executor.CreateSession(gufo::core::SessionMode::kAutoregressive,
                                    context, &error);
    if (!s)
      throw std::runtime_error("mixed audit session failed: " + error);
    return s;
  };
  auto prefill = [&](auto& session, const std::int32_t* toks, std::size_t n,
                     std::uint32_t rows) {
    std::vector<float> out(static_cast<std::size_t>(rows) * c.vocab_size);
    if (!executor.Forward(*session, std::span<const std::int32_t>(toks, n),
                          rows, out.data(),
                          q::rocm::Executor::ForwardMode::kPrefill, &error,
                          nullptr, std::span<const std::int32_t>())) {
      throw std::runtime_error("mixed audit prefill failed: " + error);
    }
    return out;
  };
  auto decode_one = [&](auto& session, std::int32_t tok) {
    std::vector<float> row(c.vocab_size);
    if (!executor.Forward(*session, std::span<const std::int32_t>(&tok, 1), 1,
                          row.data(), q::rocm::Executor::ForwardMode::kDecode,
                          &error)) {
      throw std::runtime_error("mixed audit decode failed: " + error);
    }
    return row;
  };
  auto sA = make();
  auto sB = make();
  auto sAr = make();
  auto sBr = make();
  prefill(sA, tokA.data(), 32, 32);
  prefill(sAr, tokA.data(), 32, 32);
  decode_one(sA, tokA[32]);
  decode_one(sAr, tokA[32]);
  prefill(sB, tokB.data(), 48, 48);
  prefill(sBr, tokB.data(), 48, 48);
  // Mixed: A decodes tokA[32]... (already consumed above for history only).
  // Fresh single-token decode row + B's chunk in one batch:
  const std::int32_t tAx = tokA[32];
  (void)decode_one(sA, tAx);
  (void)decode_one(sAr, tAx);
  // NOTE: sA/sAr now sit at 34; rewind-free comparison uses the NEXT token.
  // Simpler deterministic setup: compare only B's chunk rows plus a second
  // decode pair taken from identical histories below.
  const std::uint64_t before = executor.mixed_batches();
  std::vector<q::rocm::Executor::BatchItem> items;
  items.push_back({sA.get(), std::span<const std::int32_t>(&tokA[32], 1),
                   /*speculative=*/false});
  items.push_back({sB.get(), std::span<const std::int32_t>(tokB.data() + 48, 4),
                   /*speculative=*/false});
  const auto bt0 = std::chrono::steady_clock::now();
  if (!executor.ForwardBatch(items, &error))
    throw std::runtime_error("mixed ForwardBatch failed: " + error);
  const auto bt1 = std::chrono::steady_clock::now();
  const std::uint64_t after = executor.mixed_batches();
  std::vector<float> mA(c.vocab_size), mB(4 * c.vocab_size);
  if (!executor.SelectBatchLogits(0, 1, mA.data(), &error) ||
      !executor.SelectBatchLogits(1, 4, mB.data(), &error))
    throw std::runtime_error("mixed logits select failed: " + error);
  // Reference on the twin sessions (identical histories).
  const char* dump = std::getenv("MX_DUMP");
  if (dump) {
    FILE* f = std::fopen(dump, "wb");
    if (f) {
      std::fwrite(mA.data(), 4, mA.size(), f);
      std::fwrite(mB.data(), 4, mB.size(), f);
      std::fclose(f);
    }
  }
  std::vector<float> rA(c.vocab_size), rB(4 * c.vocab_size);
  const auto bt2 = std::chrono::steady_clock::now();
  if (!executor.Forward(*sAr, std::span<const std::int32_t>(&tokA[32], 1), 1,
                        rA.data(), q::rocm::Executor::ForwardMode::kDecode,
                        &error))
    throw std::runtime_error("reference decode failed: " + error);
  const auto bt3 = std::chrono::steady_clock::now();
  const auto bt4 = std::chrono::steady_clock::now();
  if (!executor.Forward(*sBr,
                        std::span<const std::int32_t>(tokB.data() + 48, 4), 4,
                        rB.data(), q::rocm::Executor::ForwardMode::kPrefill,
                        &error, nullptr, std::span<const std::int32_t>()))
    throw std::runtime_error("reference prefill failed: " + error);
  const auto bt5 = std::chrono::steady_clock::now();
  std::cout << "  bench mixed-batch ms "
            << std::chrono::duration<double, std::milli>(bt1 - bt0).count()
            << " ref-decode ms "
            << std::chrono::duration<double, std::milli>(bt3 - bt2).count()
            << " ref-prefill ms "
            << std::chrono::duration<double, std::milli>(bt5 - bt4).count()
            << "\n";
  double worst = 0.0, worstA = 0.0, worstB = 0.0;
  for (std::size_t i = 0; i < mA.size(); ++i) {
    worstA = std::max(worstA, std::abs(static_cast<double>(mA[i]) - rA[i]));
    worst = std::max(worst, std::abs(static_cast<double>(mA[i]) - rA[i]));
  }
  for (std::size_t i = 0; i < mB.size(); ++i) {
    worstB = std::max(worstB, std::abs(static_cast<double>(mB[i]) - rB[i]));
    worst = std::max(worst, std::abs(static_cast<double>(mB[i]) - rB[i]));
  }
  std::cout << "  mixed row-split worstA(decode) " << worstA
            << " worstB(chunk) " << worstB << "\n";
  for (std::uint32_t r = 0; r < 4; ++r) {
    double wr = 0.0;
    for (std::size_t j = 0; j < mA.size(); ++j)
      wr = std::max(wr, std::abs(static_cast<double>(mB[r * mA.size() + j]) -
                                 rB[r * mA.size() + j]));
    std::cout << "  mixed Brow[" << r << "]=" << wr << "\n";
  }

  std::cout << "  mixed attention worst absolute error " << worst << '\n';
  std::cout << "  mixed batches served: " << (after - before) << '\n';
  if (std::getenv("MX_TEXT") != nullptr)
    TextMixed(executor, tokenizer, weights);
  const char* mtp = std::getenv("MX_MTP");
  if (mtp != nullptr) {
    const unsigned long mr = std::strtoul(mtp, nullptr, 10);
    if (mr > 0)
      MtpBatch(executor, tokenizer, weights, static_cast<std::uint32_t>(mr));
  }
  const char* big = std::getenv("MX_BIG");
  if (big != nullptr) {
    const unsigned long br = std::strtoul(big, nullptr, 10);
    if (br > 0)
      BigBatch(executor, tokenizer, weights, static_cast<std::uint32_t>(br),
               br <= 16);
  }
  if (after == before)
    throw std::runtime_error("mixed path did not fire (counter stuck)");
  if (false) {
    FILE* f = std::fopen(dump, "wb");
    if (f) {
      std::fwrite(mA.data(), 4, mA.size(), f);
      std::fwrite(mB.data(), 4, mB.size(), f);
      std::fclose(f);
    }
  }
  // NOTE: the batch path always runs with prefill_phase=false while the
  // reference prefill runs phase=true (Prepare<1> vs <4>), and decode rows
  // move from the scalar narrow core to WMMA: expect kernel-family sized
  // deltas here. Same-family correctness is covered by CompareMixed (FP64).
  // Fail only on non-finite output or a dead mixed path.
  bool finite = true;
  for (float v : mA)
    finite = finite && std::isfinite(v);
  for (float v : mB)
    finite = finite && std::isfinite(v);
  if (!finite)
    throw std::runtime_error("mixed batch produced non-finite logits");
}
}  // namespace
int main(int argc, char** argv) {
  std::string model_path;
  std::string prompt = "The capital of France is";
  std::string dump_path;
  std::string mtp_path;
  std::string decode_text;
  std::string decode_ids_path;
  bool decode = false;
  bool mtp_audit = false;
  bool mixed_audit = false;
  std::optional<q::rocm::KvFormat> kv_format;
  bool reference = false;
  bool cost_audit = false;
  std::uint32_t cost_concurrency = 0;
  std::optional<std::uint32_t> cost_depth;
  std::uint32_t batch = 512;
  std::uint32_t context = 4096;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--mtp-model") {
      mtp_path = next();
    } else if (arg == "--mtp-audit") {
      mtp_audit = true;
    } else if (arg == "--mixed-audit") {
      mixed_audit = true;
    } else if (arg == "--kv-q8") {
      kv_format = q::rocm::KvFormat::kQ8;
    } else if (arg == "--kv-fp8") {
      kv_format = q::rocm::KvFormat::kFp8;
    } else if (arg == "--cost-audit") {
      cost_audit = true;
      const auto value = next();
      const auto [end, ec] = std::from_chars(
          value.data(), value.data() + value.size(), cost_concurrency);
      if (ec != std::errc{} || end != value.data() + value.size() ||
          (cost_concurrency != 0 && cost_concurrency != 1 &&
           cost_concurrency != 2 && cost_concurrency != 4 &&
           cost_concurrency != 6 && cost_concurrency != 8)) {
        std::fprintf(stderr, "--cost-audit requires C=0/1/2/4/6/8 (0 = all)\n");
        return 2;
      }
    } else if (arg == "--depth") {
      const auto value = next();
      std::uint32_t depth = 0;
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), depth);
      if (ec != std::errc{} || end != value.data() + value.size()) {
        std::fprintf(stderr, "--depth requires a nonnegative integer\n");
        return 2;
      }
      cost_depth = depth;
    } else if (arg == "--prompt") {
      prompt = next();
    } else if (arg == "--prompt-file" || arg == "--decode-file") {
      const auto path = next();
      std::ifstream in(path, std::ios::binary);
      if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return 2;
      }
      std::string text((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
      if (in.bad()) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        return 2;
      }
      (arg == "--prompt-file" ? prompt : decode_text) = std::move(text);
      decode |= arg == "--decode-file";
    } else if (arg == "--decode-ids") {
      decode_ids_path = next();
    } else if (arg == "--reference") {
      reference = true;
    } else if (arg == "--batch" || arg == "--context") {
      const auto value = next();
      auto& count = arg == "--batch" ? batch : context;
      const auto [end, ec] =
          std::from_chars(value.data(), value.data() + value.size(), count);
      if (ec != std::errc{} || end != value.data() + value.size() ||
          count == 0) {
        std::fprintf(stderr, "%s requires a positive integer\n", arg.c_str());
        return 2;
      }
    } else if (arg == "--dump") {
      dump_path = next();
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "--model is required\n");
    return 2;
  }
  if ((decode &&
       (decode_text.empty() || reference || mtp_audit || cost_audit)) ||
      (!decode && !decode_ids_path.empty())) {
    std::fprintf(stderr,
                 "--decode-file requires a nonempty continuation and cannot "
                 "be combined with other probes; --decode-ids requires it\n");
    return 2;
  }
  if ((mtp_audit || cost_audit) && mtp_path.empty()) {
    std::fprintf(stderr, "MTP audits require --mtp-model\n");
    return 2;
  }
  if (cost_audit && (mtp_audit || reference || !dump_path.empty())) {
    std::fprintf(stderr, "--cost-audit cannot be combined with other probes\n");
    return 2;
  }
  if (cost_depth && !cost_audit) {
    std::fprintf(stderr, "--depth requires --cost-audit\n");
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  auto weights = q::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::fprintf(stderr, "bind failed: %s\n", error.c_str());
    return 1;
  }
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
    return 1;
  }
  const auto& c = weights->config;
  if (cost_depth &&
      (c.context_length < 96 || *cost_depth > c.context_length - 96)) {
    std::fprintf(stderr, "--depth leaves no room for the cost-audit suffix\n");
    return 2;
  }
  std::unique_ptr<q::NgramTable> ngram;
  if (c.ple_layer >= 0) {
    const auto& t = weights->ple_table;
    ngram = q::NgramTable::Open(
        reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, c.ple_head_dim, t.type, &error);
    if (!ngram) {
      std::fprintf(stderr, "n-gram table failed: %s\n", error.c_str());
      return 1;
    }
  }

  std::shared_ptr<gufo::core::GgufReader> mtp_reader;
  std::optional<q::MtpWeights> mtp_weights;
  if (!mtp_path.empty()) {
    mtp_reader = gufo::core::GgufReader::OpenFile(mtp_path, &error);
    if (mtp_reader)
      mtp_weights = q::MtpWeights::Bind(*mtp_reader, c, &error);
    if (!mtp_weights) {
      std::fprintf(stderr, "MTP bind failed: %s\n", error.c_str());
      return 1;
    }
  }
  auto device = q::rocm::DeviceModel::Upload(
      *weights, *reader, mtp_weights ? &*mtp_weights : nullptr,
      mtp_reader.get(), &error);
  if (!device) {
    std::fprintf(stderr, "upload failed: %s\n", error.c_str());
    return 1;
  }
  q::rocm::Executor::Options options;
  options.max_batch = batch;
  options.max_logit_rows = std::min<std::uint32_t>(batch, 64);
  if (mtp_audit || cost_audit)
    options.max_speculative = 8;
  if (kv_format)
    options.kv_format = *kv_format;

  if (cost_audit) {
    options.max_batch = 2048;
    options.max_logit_rows = 8;
  }
  auto executor =
      q::rocm::Executor::Create(*device, ngram.get(), options, &error);
  if (!executor) {
    std::fprintf(stderr, "executor failed: %s\n", error.c_str());
    return 1;
  }
  if (cost_audit) {
    try {
      const std::array<std::uint32_t, 3> depths{0, 4096, 32768};
      std::span<const std::uint32_t> selected_depths = depths;
      if (cost_depth)
        selected_depths = std::span(&*cost_depth, 1);
      AuditMtpCosts(*executor, *tokenizer, selected_depths, cost_concurrency);
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "cost audit failed: %s\n", e.what());
      return 1;
    }
  }
  if (mixed_audit) {
    try {
      AuditMixed(*executor, *tokenizer, *weights);
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "mixed audit failed: %s\n", e.what());
      return 1;
    }
  }
  if (mtp_audit) {
    try {
      AuditMtp(*executor, *device, *weights, *mtp_weights, *tokenizer,
               model_path);
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "MTP oracle failed: %s\n", e.what());
      return 1;
    }
  }
  auto session = executor->CreateSession(
      gufo::core::SessionMode::kAutoregressive, context, &error);
  if (!session) {
    std::fprintf(stderr, "session failed: %s\n", error.c_str());
    return 1;
  }

  std::vector<std::int32_t> tokens;
  for (auto id : tokenizer->Encode(prompt)) {
    tokens.push_back(static_cast<std::int32_t>(id));
  }
  if (tokens.empty() || tokens.size() > context) {
    std::fprintf(stderr, "prompt must contain 1..%u tokens\n", context);
    return 2;
  }
  std::vector<std::int32_t> continuation;
  if (decode) {
    const auto ids = tokenizer->Encode(decode_text);
    continuation.assign(ids.begin(), ids.end());
    if (continuation.empty() || continuation.size() > context - tokens.size()) {
      std::fprintf(stderr, "complete continuation must fit in the context\n");
      return 2;
    }
  }
  std::printf("prompt tokens (%zu)\n", tokens.size());
  std::ofstream dump;
  if (!dump_path.empty()) {
    dump.open(dump_path, std::ios::binary);
    if (!dump) {
      std::fprintf(stderr, "cannot open logit dump %s\n", dump_path.c_str());
      return 1;
    }
  }

  std::vector<float> gpu_logits;
  std::size_t logit_rows = 0;
  for (std::size_t off = 0; off < tokens.size(); off += batch) {
    const std::size_t n = std::min<std::size_t>(batch, tokens.size() - off);
    const auto rows = static_cast<std::uint32_t>(
        std::min<std::size_t>(n, options.max_logit_rows));
    std::vector<float> out(static_cast<std::size_t>(rows) * c.vocab_size);
    // Name the following chunk as Session::Feed does, so its n-gram read
    // overlaps this one.
    const std::size_t next =
        std::min<std::size_t>(batch, tokens.size() - off - n);
    if (!executor->Forward(
            *session, std::span<const std::int32_t>(tokens.data() + off, n),
            rows, out.data(), q::rocm::Executor::ForwardMode::kPrefill, &error,
            nullptr,
            std::span<const std::int32_t>(tokens.data() + off + n, next))) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
    const bool last = off + n == tokens.size();
    if (last) {
      gpu_logits = std::move(out);
      logit_rows = rows;
    }
  }
  if (decode) {
    // Matched-token decode: feed a fixed continuation one token at a time
    // through the decode path and dump each step's logits.
    std::vector<float> row(c.vocab_size);
    std::size_t steps = 0;
    for (const auto t : continuation) {
      if (!executor->Forward(*session, std::span<const std::int32_t>(&t, 1), 1,
                             row.data(),
                             q::rocm::Executor::ForwardMode::kDecode, &error)) {
        std::fprintf(stderr, "decode failed: %s\n", error.c_str());
        return 1;
      }
      if (dump.is_open())
        dump.write(reinterpret_cast<const char*>(row.data()),
                   static_cast<std::streamsize>(row.size() * sizeof(float)));
      ++steps;
    }
    if (dump.is_open()) {
      dump.close();
      if (!dump) {
        std::fprintf(stderr, "cannot write logit dump %s\n", dump_path.c_str());
        return 1;
      }
    }
    if (!decode_ids_path.empty()) {
      std::ofstream ids(decode_ids_path, std::ios::binary);
      ids.write(reinterpret_cast<const char*>(continuation.data()),
                static_cast<std::streamsize>(continuation.size() *
                                             sizeof(continuation[0])));
      ids.close();
      if (!ids) {
        std::fprintf(stderr, "cannot write token IDs %s\n",
                     decode_ids_path.c_str());
        return 1;
      }
    }
    std::printf("decode steps %zu after %zu prompt tokens\n", steps,
                tokens.size());
    return 0;
  }
  if (dump.is_open()) {
    dump.write(reinterpret_cast<const char*>(gpu_logits.data()),
               static_cast<std::streamsize>(gpu_logits.size() * sizeof(float)));
    dump.flush();
    if (!dump) {
      std::fprintf(stderr, "cannot write logit dump %s\n", dump_path.c_str());
      return 1;
    }
  }

  if (reference) {
    q::ReferenceModel ref(*weights, ngram.get(),
                          static_cast<std::uint32_t>(tokens.size()));
    std::vector<float> ref_logits(c.vocab_size);
    const std::size_t first = tokens.size() - logit_rows;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (!ref.Step(tokens[i], ref_logits, {}, &error)) {
        std::fprintf(stderr, "reference failed: %s\n", error.c_str());
        return 1;
      }
      if (i < first) {
        continue;
      }
      const Stats st =
          Compare(ref_logits.data(),
                  gpu_logits.data() + (i - first) * c.vocab_size, c.vocab_size);
      std::printf("pos %3zu ref %6u gpu %6u %s  KL %.5f  max|d| %.3f\n", i,
                  st.argmax_a, st.argmax_b,
                  st.argmax_a == st.argmax_b ? "OK  " : "DIFF", st.kl,
                  st.max_abs);
    }
  }

  return 0;
}
