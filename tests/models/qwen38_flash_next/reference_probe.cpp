// Runs the float32 reference over a prompt and reports the next-token
// distribution at every position. Used to pin the operator semantics against
// llama.cpp logits and, later, against the ROCm runtime.
//
//   reference_probe --model FIRST_SHARD.gguf [--prompt TEXT | --tokens 1,2,3]
//                   [--generate N] [--dump logits.bin] [--dump-k PREFIX]
//                   [--kv-q8] [--kv-q8p16] [--top 5]
//
// --dump writes float32 logits for every prompt position (and generated
// position) back to back; the row width is the vocabulary size.
// --dump-k writes the accumulated post-RoPE key/value history and the raw
// indexer keys of every attention layer under PREFIX (see
// ReferenceModel::DumpAttentionState).
// --kv-q8 emulates the blocked int8 KV cache (per-32-block symmetric levels
// on cached K/V) instead of float32 formulas, for offline drift studies.
// --kv-q8p16 is the same with per-16-block granularity.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next;

namespace {

std::vector<std::int32_t> ParseTokens(const std::string& text) {
  std::vector<std::int32_t> out;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t end = text.find(',', start);
    out.push_back(std::atoi(text.substr(start, end - start).c_str()));
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return out;
}

void PrintTop(const std::vector<float>& logits,
              const gufo::tokenization::QwenTokenizer& tokenizer,
              std::uint32_t top) {
  std::vector<std::uint32_t> order(logits.size());
  std::iota(order.begin(), order.end(), 0U);
  std::partial_sort(
      order.begin(), order.begin() + top, order.end(),
      [&](std::uint32_t a, std::uint32_t b) { return logits[a] > logits[b]; });
  const float max_logit = logits[order[0]];
  double denom = 0.0;
  for (float v : logits) {
    denom += std::exp(static_cast<double>(v - max_logit));
  }
  for (std::uint32_t i = 0; i < top; ++i) {
    const std::uint32_t id = order[i];
    const double p =
        std::exp(static_cast<double>(logits[id] - max_logit)) / denom;
    std::string text(tokenizer.DecodeToken(id));
    for (char& ch : text) {
      if (ch == '\n') {
        ch = ' ';
      }
    }
    std::printf("    %6u  %8.4f  %6.3f  %s\n", id, logits[id], p, text.c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string prompt;
  std::string tokens_arg;
  std::string dump_path;
  std::string dump_key_prefix;
  bool kv_q8 = false;
  bool kv_q8p16 = false;
  int generate = 0;
  std::uint32_t top = 5;
  bool tokenize_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--prompt") {
      prompt = next();
    } else if (arg == "--tokens") {
      tokens_arg = next();
    } else if (arg == "--dump") {
      dump_path = next();
    } else if (arg == "--dump-k") {
      dump_key_prefix = next();
    } else if (arg == "--kv-q8") {
      kv_q8 = true;
    } else if (arg == "--kv-q8p16") {
      kv_q8p16 = true;
    } else if (arg == "--generate") {
      generate = std::atoi(next().c_str());
    } else if (arg == "--top") {
      top = static_cast<std::uint32_t>(std::atoi(next().c_str()));
    } else if (arg == "--tokenize-only") {
      tokenize_only = true;
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "--model is required\n");
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
  std::printf(
      "model: %u layers, hidden %u, vocab %u, experts %u/%u, ple layer %d\n",
      c.num_layers, c.hidden_size, c.vocab_size, c.num_experts_used,
      c.num_experts, c.ple_layer);

  std::vector<std::int32_t> tokens;
  if (!tokens_arg.empty()) {
    tokens = ParseTokens(tokens_arg);
  } else {
    for (auto id : tokenizer->Encode(prompt)) {
      tokens.push_back(static_cast<std::int32_t>(id));
    }
  }
  std::printf("prompt tokens (%zu):", tokens.size());
  for (auto t : tokens) {
    std::printf(" %d", t);
  }
  std::printf("\n");
  if (tokenize_only) {
    return 0;
  }

  std::ofstream dump;
  if (!dump_path.empty()) {
    dump.open(dump_path, std::ios::binary);
  }

  q::ReferenceModel model(*weights, ngram.get(),
                          static_cast<std::uint32_t>(tokens.size()) + generate,
                          kv_q8p16 ? q::ReferenceModel::Storage::kQ8p16
                          : kv_q8  ? q::ReferenceModel::Storage::kQ8
                                   : q::ReferenceModel::Storage::kFloat32);
  std::vector<float> logits(c.vocab_size);
  const std::size_t total = tokens.size() + static_cast<std::size_t>(generate);
  for (std::size_t i = 0; i < total; ++i) {
    std::int32_t token = 0;
    if (i < tokens.size()) {
      token = tokens[i];
    } else {
      token = static_cast<std::int32_t>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      tokens.push_back(token);
    }
    const auto start = std::chrono::steady_clock::now();
    if (!model.Step(token, logits, {}, &error)) {
      std::fprintf(stderr, "step %zu failed: %s\n", i, error.c_str());
      return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    std::string text(tokenizer->DecodeToken(static_cast<std::uint32_t>(token)));
    std::printf("pos %zu token %d '%s' (%.0f ms)%s\n", i, token, text.c_str(),
                ms, i >= tokens.size() - (generate > 0 ? 0 : 1) ? "" : "");
    PrintTop(logits, *tokenizer, top);
    if (dump.is_open()) {
      dump.write(reinterpret_cast<const char*>(logits.data()),
                 static_cast<std::streamsize>(logits.size() * sizeof(float)));
    }
  }
  if (!dump_key_prefix.empty()) {
    if (!model.DumpAttentionState(dump_key_prefix, &error)) {
      std::fprintf(stderr, "key dump failed: %s\n", error.c_str());
      return 1;
    }
    std::printf("key dump written under %s\n", dump_key_prefix.c_str());
  }
  if (generate > 0) {
    std::vector<std::uint32_t> generated(
        tokens.begin() + static_cast<std::ptrdiff_t>(total - generate),
        tokens.end());
    std::printf("generated: %s\n", tokenizer->Decode(generated).c_str());
  }
  return 0;
}
