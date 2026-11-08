// bge_rerank_score: test driver for the bge-reranker leg.
//
// Loads a `bert` + classifier-head GGUF artifact through the model registry
// ("bge-reranker"), pair-encodes --query with each --doc ([CLS] q [SEP] d
// [SEP]), and prints one JSON summary with sigmoid scores in input order
// plus the top_n ranking. --show-logits also prints raw head logits (the
// llama.cpp relevance scale).
//
// Exit codes: 0 ok, 2 usage, 3 load/tokenize/Score failure.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/rerank_score.hpp"
#include "src/models/bge_m3/engine.hpp"
#include "src/models/bge_m3/forward.hpp"
#include "src/models/bge_reranker/engine.hpp"

namespace {

void Usage() {
  std::cerr
      << "usage: bge_rerank_score <model.gguf> --query \"...\" --doc \"...\" "
         "[--doc \"...\"] [--top-n K] [--cpu] [--show-logits] "
         "[--dump-cls-hidden f] [--batch]\n"
      << "  --batch scores through EngineSession::ScoreBatch (the padded\n"
      << "  multi-sequence fast path) instead of the per-pair Score loop.\n";
}

std::string JsonEscape(std::string_view text) {
  std::string out;
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(c);
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out.push_back(c);
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    Usage();
    return 2;
  }
  const std::string model_path = argv[1];
  std::string query;
  std::vector<std::string> documents;
  std::size_t top_n = 0;
  bool use_hip = true;
  bool show_logits = false;
  bool use_batch = false;
  std::string dump_cls;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--query" && i + 1 < argc) {
      query = argv[++i];
    } else if (arg == "--doc" && i + 1 < argc) {
      documents.emplace_back(argv[++i]);
    } else if (arg == "--top-n" && i + 1 < argc) {
      top_n = static_cast<std::size_t>(std::stoul(argv[++i]));
    } else if (arg == "--cpu") {
      use_hip = false;
    } else if (arg == "--show-logits") {
      show_logits = true;
    } else if (arg == "--dump-cls-hidden" && i + 1 < argc) {
      dump_cls = argv[++i];
    } else if (arg == "--batch") {
      use_batch = true;
    } else {
      Usage();
      return 2;
    }
  }
  if (query.empty() || documents.empty()) {
    Usage();
    return 2;
  }

  std::string error;
  auto model = gufo::models::bge_reranker::ModelRegistry::Instance().Create(
      "bge-reranker", model_path, &error);
  if (model == nullptr) {
    std::cerr << "bge_rerank_score: load failed: " << error << "\n";
    return 3;
  }
  gufo::models::bge_reranker::EngineSession session(model.get());
  std::vector<float> logits;
  logits.reserve(documents.size());
  std::vector<float> scores;
  scores.reserve(documents.size());
  double total_ms = 0.0;
  std::size_t prompt_tokens = 0;
  if (use_batch) {
    // Verification path: the ScoreBatch under test (padded multi-sequence
    // fast path, or the legacy loop under GUFO_BGE_BATCH=0 / over capacity).
    auto batched =
        session.ScoreBatch(query, documents, use_hip, &total_ms,
                           &prompt_tokens, &error);
    if (!batched.has_value()) {
      std::cerr << "bge_rerank_score: ScoreBatch failed: " << error << "\n";
      return 3;
    }
    scores = std::move(*batched);
    std::cerr << "bge_rerank_score: batched=" << scores.size() << "\n";
  } else {
    for (const auto& document : documents) {
      // Encode once for usage accounting, then score the same ids.
      auto ids = session.EncodePair(query, document, &error);
      if (!ids.has_value()) {
        std::cerr << "bge_rerank_score: encode failed: " << error << "\n";
        return 3;
      }
      prompt_tokens += ids->size();
      auto result = session.Score(*ids, use_hip, &error);
      if (!result.has_value()) {
        std::cerr << "bge_rerank_score: Score failed: " << error << "\n";
        return 3;
      }
      logits.push_back(result->logit);
      total_ms += result->prefill_ms;
    }
  }
  if (!dump_cls.empty()) {
    // Precision probe: dump the first pair's raw CLS hidden row (F32) for
    // offline cosine comparison against torch. No math changes. Combined
    // with --batch it dumps the batch path's row 0 instead, so the
    // batch-vs-single CLS cosine pins the padding mask precision.
    std::string dump_error;
    std::vector<float> cls_hidden;
    std::vector<std::vector<std::uint32_t>> dump_batch;
    std::size_t dump_lmax = 0;
    std::vector<std::uint32_t> dump_first;
    if (use_batch) {
      for (const auto& document : documents) {
        auto ids = session.EncodePair(query, document, &dump_error);
        if (!ids.has_value()) {
          std::cerr << "bge_rerank_score: dump EncodePair failed: "
                    << dump_error << "\n";
          return 3;
        }
        dump_lmax = std::max(dump_lmax, ids->size());
        dump_batch.push_back(std::move(*ids));
      }
      if (dump_lmax == 0) {
        std::cerr << "bge_rerank_score: dump batch is empty\n";
        return 3;
      }
    } else {
      auto first_ids =
          session.EncodePair(query, documents.front(), &dump_error);
      if (!first_ids.has_value()) {
        std::cerr << "bge_rerank_score: dump EncodePair failed: "
                  << dump_error << "\n";
        return 3;
      }
      dump_first = std::move(*first_ids);
    }
    if (use_hip) {
#ifdef ENGINE_ENABLE_HIP
      if (use_batch) {
        auto cls = gufo::models::bge_m3::HipClsHiddenBatch(
            model->Encoder(), dump_batch, model->Config().pad_token_id,
            &dump_error);
        if (!cls.has_value() || cls->cls.empty()) {
          std::cerr << "bge_rerank_score: dump HipClsHiddenBatch failed: "
                    << dump_error << "\n";
          return 3;
        }
        cls_hidden = cls->cls.front();
      } else {
        auto cls = gufo::models::bge_m3::HipClsHidden(model->Encoder(),
                                                      dump_first, &dump_error);
        if (!cls.has_value()) {
          std::cerr << "bge_rerank_score: dump HipClsHidden failed: "
                    << dump_error << "\n";
          return 3;
        }
        cls_hidden = std::move(cls->cls);
      }
#else
      std::cerr << "bge_rerank_score: --dump-cls-hidden needs a HIP build\n";
      return 3;
#endif
    } else {
      const std::size_t h = model->Encoder().Config().hidden_size;
      auto scratch = gufo::models::bge_m3::BgeM3Scratch::Create(
          model->Encoder().Config(),
          gufo::models::bge_reranker::kBgeRerankerMaxTokens);
      if (use_batch) {
        std::vector<float> last_hidden(dump_batch.size() * dump_lmax * h);
        if (!gufo::models::bge_m3::CpuForwardBatch(
                model->Encoder().F32(), dump_batch,
                model->Config().pad_token_id, scratch, last_hidden)) {
          std::cerr << "bge_rerank_score: dump CpuForwardBatch failed\n";
          return 3;
        }
        cls_hidden.assign(last_hidden.begin(), last_hidden.begin() + h);
      } else {
        std::vector<float> last_hidden(dump_first.size() * h);
        if (!gufo::models::bge_m3::CpuForward(
                model->Encoder().F32(), dump_first, scratch, last_hidden)) {
          std::cerr << "bge_rerank_score: dump CpuForward failed\n";
          return 3;
        }
        cls_hidden.assign(last_hidden.begin(), last_hidden.begin() + h);
      }
    }
    std::ofstream dump_out(dump_cls, std::ios::binary);
    if (!dump_out) {
      std::cerr << "bge_rerank_score: cannot open --dump-cls-hidden: " +
                       dump_cls + "\n";
      return 3;
    }
    dump_out.write(reinterpret_cast<const char*>(cls_hidden.data()),
                   static_cast<std::streamsize>(cls_hidden.size() *
                                                sizeof(float)));
    const std::size_t dump_tokens =
        use_batch ? dump_lmax : dump_first.size();
    std::cerr << "bge_rerank_score: dumped cls_hidden dim=" << cls_hidden.size()
              << " tokens=" << dump_tokens << " to " << dump_cls << "\n";
  }
  if (!use_batch) {
    scores.reserve(logits.size());
    for (const float logit : logits) {
      scores.push_back(gufo::core::rerank::Sigmoid(logit));
    }
  }
  std::cout << "{\"model\":\"" << JsonEscape(model_path) << "\",\"scores\":[";
  for (std::size_t i = 0; i < scores.size(); ++i) {
    if (i > 0)
      std::cout << ",";
    std::cout << "{\"index\":" << i << ",\"relevance_score\":" << scores[i];
    if (show_logits && !use_batch && i < logits.size())
      std::cout << ",\"logit\":" << logits[i];
    std::cout << "}";
  }
  std::cout << "],\"prompt_tokens\":" << prompt_tokens
            << ",\"prefill_ms\":" << total_ms;
  const auto order = gufo::core::rerank::RankIndices(scores, top_n);
  std::cout << ",\"ranking\":[";
  for (std::size_t i = 0; i < order.size(); ++i) {
    if (i > 0)
      std::cout << ",";
    std::cout << order[i];
  }
  std::cout << "]}\n";
  return 0;
}
