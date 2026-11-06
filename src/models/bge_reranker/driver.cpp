// bge_rerank_score: test driver for the bge-reranker leg.
//
// Loads a `bert` + classifier-head GGUF artifact through the model registry
// ("bge-reranker"), pair-encodes --query with each --doc ([CLS] q [SEP] d
// [SEP]), and prints one JSON summary with sigmoid scores in input order
// plus the top_n ranking. --show-logits also prints raw head logits (the
// llama.cpp relevance scale).
//
// Exit codes: 0 ok, 2 usage, 3 load/tokenize/Score failure.
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
#include "src/models/bge_reranker/engine.hpp"

namespace {

void Usage() {
  std::cerr
      << "usage: bge_rerank_score <model.gguf> --query \"...\" --doc \"...\" "
         "[--doc \"...\"] [--top-n K] [--cpu] [--show-logits]\n";
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
  double total_ms = 0.0;
  std::size_t prompt_tokens = 0;
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
  std::vector<float> scores;
  scores.reserve(logits.size());
  for (const float logit : logits) {
    scores.push_back(gufo::core::rerank::Sigmoid(logit));
  }
  std::cout << "{\"model\":\"" << JsonEscape(model_path) << "\",\"scores\":[";
  for (std::size_t i = 0; i < scores.size(); ++i) {
    if (i > 0)
      std::cout << ",";
    std::cout << "{\"index\":" << i << ",\"relevance_score\":" << scores[i];
    if (show_logits)
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
