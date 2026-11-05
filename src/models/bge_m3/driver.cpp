// bge_m3_embed: test driver for the bge-m3 encoder leg.
//
// Loads a `bert` GGUF artifact through the model registry ("bge-m3"),
// tokenizes --text (with [CLS]/[SEP] wrap) or reads --ids-file (whitespace
// separated HF reference ids, no wrap), runs one Sync, and prints a JSON
// summary. --out writes the raw F32 vector for offline cosine comparison
// against sentence-transformers (torch CPU).
//
// Exit codes: 0 ok, 2 usage, 3 load/tokenize/Sync failure.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/bge_m3/engine.hpp"

namespace {

void Usage() {
  std::cerr
      << "usage: bge_m3_embed <model.gguf> [--text \"...\"] [--ids-file f]\n"
         "                       [--ids 0,1,2,...] [--cpu] [--out vec.f32]\n";
}

bool ParseIdsFile(const std::string& path, std::vector<std::uint32_t>& ids,
                  std::string& error) {
  std::ifstream in(path);
  if (!in) {
    error = "cannot open --ids-file: " + path;
    return false;
  }
  std::uint32_t id = 0;
  while (in >> id) ids.push_back(id);
  if (ids.empty()) {
    error = "--ids-file holds no ids: " + path;
    return false;
  }
  return true;
}

bool ParseIdsList(const std::string& text, std::vector<std::uint32_t>& ids,
                  std::string& error) {
  std::string token;
  std::istringstream in(text);
  while (std::getline(in, token, ',')) {
    try {
      ids.push_back(static_cast<std::uint32_t>(std::stoul(token)));
    } catch (...) {
      error = "bad --ids entry: " + token;
      return false;
    }
  }
  if (ids.empty()) {
    error = "--ids holds no ids";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    Usage();
    return 2;
  }
  const std::string model_path = argv[1];
  std::string text;
  std::string ids_file;
  std::string ids_list;
  std::string out_path;
  bool use_hip = true;
  bool dump_ids = false;
  bool has_text = false;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--text" && i + 1 < argc) {
      text = argv[++i];
      has_text = true;
    } else if (arg == "--ids-file" && i + 1 < argc) {
      ids_file = argv[++i];
    } else if (arg == "--ids" && i + 1 < argc) {
      ids_list = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--dump-ids") {
      dump_ids = true;
    } else if (arg == "--cpu") {
      use_hip = false;
    } else {
      Usage();
      return 2;
    }
  }
  // Exactly one input source (--text "" is a legal empty prompt).
  const int sources =
      (has_text ? 1 : 0) + (!ids_file.empty() ? 1 : 0) + (!ids_list.empty() ? 1 : 0);
  if (sources != 1) {
    if (sources == 0) {
      Usage();
    } else {
      std::cerr << "bge_m3_embed: pick one of --text/--ids-file/--ids\n";
    }
    return 2;
  }

  std::string error;
  auto model = gufo::models::bge_m3::ModelRegistry::Instance().Create(
      "bge-m3", model_path, &error);
  if (model == nullptr) {
    std::cerr << "bge_m3_embed: load failed: " << error << "\n";
    return 3;
  }
  std::vector<std::uint32_t> ids;
  if (has_text) {
    auto encoded = model->Tokenizer().Encode(
        text, gufo::models::bge_m3::kBgeM3Leg1MaxTokens);
    if (!encoded.has_value()) {
      std::cerr << "bge_m3_embed: tokenize failed (over leg-1 window?)\n";
      return 3;
    }
    ids = std::move(*encoded);
  } else if (!ids_file.empty()) {
    if (!ParseIdsFile(ids_file, ids, error)) {
      std::cerr << "bge_m3_embed: " << error << "\n";
      return 3;
    }
  } else {
    if (!ParseIdsList(ids_list, ids, error)) {
      std::cerr << "bge_m3_embed: " << error << "\n";
      return 3;
    }
  }

  gufo::models::bge_m3::EngineSession session(model.get());
  if (dump_ids) {
    std::cout << "ids[" << ids.size() << "]:";
    for (const auto id : ids) std::cout << " " << id;
    std::cout << "\n";
    return 0;
  }
  auto result = session.Sync(ids, use_hip, &error);
  if (!result.has_value()) {
    std::cerr << "bge_m3_embed: Sync failed: " << error << "\n";
    return 3;
  }
  if (!out_path.empty()) {
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
      std::cerr << "bge_m3_embed: cannot open --out: " + out_path + "\n";
      return 3;
    }
    out.write(reinterpret_cast<const char*>(result->values.data()),
              static_cast<std::streamsize>(result->values.size() *
                                           sizeof(float)));
  }
  std::cout << "{\"dim\":" << result->values.size()
            << ",\"tokens\":" << ids.size()
            << ",\"pre_norm\":" << result->pre_norm
            << ",\"prefill_ms\":" << result->prefill_ms << ",\"first8\":[";
  for (std::size_t i = 0; i < result->values.size() && i < 8; ++i) {
    if (i > 0) std::cout << ",";
    std::cout << result->values[i];
  }
  std::cout << "]}\n";
  return 0;
}
