// qwen_rerank_score: test driver for the qwen-reranker leg.
//
// Loads a `qwen3` GGUF artifact through the model registry
// ("qwen-reranker"), renders each (--query, --doc) pair through the
// Qwen3-Reranker chat template, and prints one JSON summary with P(yes)
// scores in input order plus the top_n ranking. --instruction overrides
// the default retrieval instruction.
//
// Exit codes: 0 ok, 2 usage, 3 load/tokenize/Score failure.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/rerank_score.hpp"
#include "src/models/qwen_reranker/engine.hpp"

namespace {

void Usage() {
  std::cerr
      << "usage: qwen_rerank_score <model.gguf> --query \"...\" --doc \"...\" "
         "[--doc \"...\"] [--instruction \"...\"] [--top-n K]\n";
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
  std::string instruction;
  std::size_t top_n = 0;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--query" && i + 1 < argc) {
      query = argv[++i];
    } else if (arg == "--doc" && i + 1 < argc) {
      documents.emplace_back(argv[++i]);
    } else if (arg == "--instruction" && i + 1 < argc) {
      instruction = argv[++i];
    } else if (arg == "--top-n" && i + 1 < argc) {
      top_n = static_cast<std::size_t>(std::stoul(argv[++i]));
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
  auto model = gufo::models::qwen_reranker::ModelRegistry::Instance().Create(
      "qwen-reranker", model_path, &error);
  if (model == nullptr) {
    std::cerr << "qwen_rerank_score: load failed: " << error << "\n";
    return 3;
  }
  gufo::models::qwen_reranker::EngineSession session(model.get());
  std::vector<float> scores;
  scores.reserve(documents.size());
  double total_ms = 0.0;
  std::size_t prompt_tokens = 0;
  for (const auto& document : documents) {
    auto ids = gufo::models::qwen_reranker::EngineModel::BuildRerankIds(
        model->Qwen().GetTokenizer(), instruction, query, document, &error);
    if (!ids.has_value()) {
      std::cerr << "qwen_rerank_score: encode failed: " << error << "\n";
      return 3;
    }
    prompt_tokens += ids->size();
    auto result = session.Score(*ids, &error);
    if (!result.has_value()) {
      std::cerr << "qwen_rerank_score: Score failed: " << error << "\n";
      return 3;
    }
    scores.push_back(result->score);
    total_ms += result->prefill_ms;
    if (scores.size() == 1) {
      const auto yes_id = model->YesId();
      const auto no_id = model->NoId();
      const gufo::tokenization::TokenId yes_arr[1] = {yes_id};
      const gufo::tokenization::TokenId no_arr[1] = {no_id};
      std::cout << "{\"debug_first_pair\":{\"yes_id\":" << yes_id
                << ",\"no_id\":" << no_id << ",\"yes_token\":\""
                << JsonEscape(model->Qwen().GetTokenizer().Decode(yes_arr))
                << "\",\"no_token\":\""
                << JsonEscape(model->Qwen().GetTokenizer().Decode(no_arr))
                << "\",\"logit_yes\":" << result->logit_yes
                << ",\"logit_no\":" << result->logit_no << "},";
    }
  }
  // The debug block above opens the JSON object; scores continue it.
  std::cout << "\"model\":\"" << JsonEscape(model_path) << "\",\"scores\":[";
  for (std::size_t i = 0; i < scores.size(); ++i) {
    if (i > 0)
      std::cout << ",";
    std::cout << "{\"index\":" << i << ",\"relevance_score\":" << scores[i]
              << "}";
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
