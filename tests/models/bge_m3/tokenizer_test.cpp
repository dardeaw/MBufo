// bge-m3 Unigram tokenizer tests.
//
// Part A (always runs): a hand-built minimal SentencePiece-Unigram artifact
// exercises added-token splitting, NFKC/whitespace normalization, Viterbi
// lattice decoding, single-unk runs, the [CLS]/[SEP] wrap, and the window
// refusal -- with no external files.
// Part B (GUFO_BGE_MODEL set, else skipped): exact HF parity over the 343
// committed golden vectors (tests/models/bge_m3/testdata/*.json, generated
// from sentence-transformers + transformers). Any NFKC-version drift between
// ICU and the HF Precompiled table shows up here first.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/bge_m3/engine.hpp"
#include "src/models/bge_m3/tokenizer.hpp"

namespace {

// --- minimal GGUF v3 writer (just enough KV for the Unigram loader) ---

void AppendU32(std::string& buf, std::uint32_t v) {
  buf.append(reinterpret_cast<const char*>(&v), sizeof(v));
}
void AppendU64(std::string& buf, std::uint64_t v) {
  buf.append(reinterpret_cast<const char*>(&v), sizeof(v));
}
void AppendI32(std::string& buf, std::int32_t v) {
  buf.append(reinterpret_cast<const char*>(&v), sizeof(v));
}
void AppendF32(std::string& buf, float v) {
  buf.append(reinterpret_cast<const char*>(&v), sizeof(v));
}
void AppendStr(std::string& buf, std::string_view s) {
  AppendU64(buf, s.size());
  buf.append(s.data(), s.size());
}

struct MiniGguf {
  std::string buf;
  std::uint64_t kv_count{0};
  MiniGguf() {
    buf.append("GGUF", 4);
    AppendU32(buf, 3);       // version
    AppendU64(buf, 0);       // tensor count (patched never: stays 0)
    AppendU64(buf, 0);       // kv count (patched in Build)
    kv_pos_ = buf.size() - 8;
  }
  void AddString(std::string_view key, std::string_view val) {
    AppendStr(buf, key);
    AppendU32(buf, 8);
    AppendStr(buf, val);
    ++kv_count;
  }
  void AddU32(std::string_view key, std::uint32_t val) {
    AppendStr(buf, key);
    AppendU32(buf, 4);
    AppendU32(buf, val);
    ++kv_count;
  }
  void AddStringArray(std::string_view key,
                      const std::vector<std::string>& vals) {
    AppendStr(buf, key);
    AppendU32(buf, 9);
    AppendU32(buf, 8);
    AppendU64(buf, vals.size());
    for (const auto& v : vals) AppendStr(buf, v);
    ++kv_count;
  }
  void AddF32Array(std::string_view key, const std::vector<float>& vals) {
    AppendStr(buf, key);
    AppendU32(buf, 9);
    AppendU32(buf, 6);
    AppendU64(buf, vals.size());
    for (const float v : vals) AppendF32(buf, v);
    ++kv_count;
  }
  void AddI32Array(std::string_view key, const std::vector<std::int32_t>& vals) {
    AppendStr(buf, key);
    AppendU32(buf, 9);
    AppendU32(buf, 5);
    AppendU64(buf, vals.size());
    for (const std::int32_t v : vals) AppendI32(buf, v);
    ++kv_count;
  }
  std::string Build() {
    std::memcpy(buf.data() + kv_pos_, &kv_count, sizeof(kv_count));
    // Tensor data base must land inside the buffer (GGUF alignment rule);
    // pad well past the default 32-byte alignment.
    buf.append(64, '\0');
    return buf;
  }
  std::size_t kv_pos_{0};
};

std::unique_ptr<gufo::models::bge_m3::SpmUnigramTokenizer> MakeTinyTokenizer(
    std::string& error) {
  // Pieces: 0 <s>, 1 </s>, 2 <unk>, 3 "▁a", 4 "▁b", 5 "▁ab" (cheap),
  // 6 "▁c", 7 <pad>, 8 "b" + 9 "c" (continuation pieces without the marker,
  // mirroring real SPM vocabs where 'b' follows an unk run as in 'a㋿b').
  MiniGguf builder;
  builder.AddString("general.architecture", "bert");
  builder.AddU32("tokenizer.ggml.bos_token_id", 0);
  builder.AddU32("tokenizer.ggml.eos_token_id", 1);
  builder.AddU32("tokenizer.ggml.unknown_token_id", 2);
  builder.AddU32("tokenizer.ggml.padding_token_id", 7);
  builder.AddStringArray("tokenizer.ggml.tokens",
                         {"<s>", "</s>", "<unk>", "\xe2\x96\x81"
                                                  "a",
                          "\xe2\x96\x81"
                          "b",
                          "\xe2\x96\x81"
                          "ab",
                          "\xe2\x96\x81"
                          "c",
                          "<pad>", "b", "c"});
  builder.AddF32Array("tokenizer.ggml.scores",
                      {0.0F, 0.0F, 0.0F, -3.0F, -3.0F, -1.5F, -10.0F, 0.0F,
                       -2.0F, -4.0F});
  builder.AddI32Array("tokenizer.ggml.token_type",
                      {3, 3, 2, 1, 1, 1, 1, 3, 1, 1});
  const std::string blob = builder.Build();
  auto reader = gufo::core::GgufReader::OpenMemory(blob.data(), blob.size(),
                                                  &error);
  if (reader == nullptr) {
    std::cerr << "tiny gguf open failed: " << error << "\n";
    return nullptr;
  }
  auto tok = gufo::models::bge_m3::SpmUnigramTokenizer::CreateFromGguf(
      *reader, &error);
  if (tok == nullptr) {
    std::cerr << "tiny tokenizer create failed: " << error << "\n";
  }
  return tok;
}

int TestTinyTokenizer() {
  using gufo::models::bge_m3::SpmUnigramTokenizer;
  std::string error;
  auto tok = MakeTinyTokenizer(error);
  if (tok == nullptr) {
    std::cerr << "tiny tokenizer setup failed: " << error << "\n";
    return 1;
  }

  // "ab" as one word: Viterbi prefers cheap "▁ab" (-1.5) over
  // "▁a"+"b" (-3 + -2).
  auto ids = tok->Encode("ab", 16);
  assert(ids.has_value());
  // [CLS]=0, "▁ab"=5, [SEP]=1.
  assert(ids->size() == 3 && (*ids)[0] == 0 && (*ids)[1] == 5 &&
         (*ids)[2] == 1);

  // Unknown run collapses to one <unk>; the trailing "b" still matches its
  // continuation piece (mirrors HF 'a㋿b' -> [▁a, <unk>, b]).
  ids = tok->Encode("a\xF0\x9F\x98\x80"
                    "\xF0\x9F\x98\x81"
                    "b",
                    16);
  assert(ids.has_value());
  assert(ids->size() == 5 && (*ids)[0] == 0 && (*ids)[1] == 3 &&
         (*ids)[2] == 2 && (*ids)[3] == 8 && (*ids)[4] == 1);

  // Added-token literal splits before normalization.
  ids = tok->Encode("a<unk>b", 16);
  assert(ids.has_value());
  assert(ids->size() == 5 && (*ids)[0] == 0 && (*ids)[1] == 3 &&
         (*ids)[2] == 2 && (*ids)[3] == 4 && (*ids)[4] == 1);

  // Whitespace collapse + empty body wrap.
  ids = tok->Encode("  a   b ", 16);
  assert(ids.has_value());
  assert(ids->size() == 4);
  ids = tok->Encode("", 16);
  assert(ids.has_value() && ids->size() == 2 && (*ids)[0] == 0 &&
         (*ids)[1] == 1);

  // Window refusal.
  assert(!tok->Encode("a a a a a", 2).has_value());
  return 0;
}

void TestTinyRefusals() {  MiniGguf builder;
  builder.AddString("general.architecture", "bert");
  const std::string blob = builder.Build();
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(blob.data(), blob.size(), &error);
  assert(reader != nullptr);
  auto tok =
      gufo::models::bge_m3::SpmUnigramTokenizer::CreateFromGguf(*reader,
                                                                &error);
  assert(tok == nullptr && !error.empty());
}

// --- Part B: HF golden parity (env-gated, SKIP_RETURN_CODE 77) ---

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Minimal JSON reader for the golden shape:
// {"probes":[{"text":"...","ids":[...]}, ...]} with \uXXXX escapes.
bool ParseGoldens(const std::string& json,
                  std::vector<std::pair<std::string, std::vector<std::uint32_t>>>& out,
                  std::string& error) {
  std::size_t pos = 0;
  const auto skip_ws = [&]() {
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' ||
            json[pos] == '\t')) {
      ++pos;
    }
  };
  // Find each {"text":"...","ids":[...]} object by scanning for "text".
  while (true) {
    pos = json.find("\"text\"", pos);
    if (pos == std::string::npos) return true;
    pos += 6;
    skip_ws();
    if (pos >= json.size() || json[pos] != ':') {
      error = "golden: expected ':'";
      return false;
    }
    ++pos;
    skip_ws();
    if (pos >= json.size() || json[pos] != '"') {
      error = "golden: expected text string";
      return false;
    }
    ++pos;
    std::string text16;  // UTF-16 code units as parsed
    std::string text8;
    while (pos < json.size() && json[pos] != '"') {
      if (json[pos] == '\\' && pos + 1 < json.size()) {
        const char e = json[pos + 1];
        if (e == 'u' && pos + 5 < json.size()) {
          const std::string hex = json.substr(pos + 2, 4);
          const auto cp = static_cast<char32_t>(std::stoul(hex, nullptr, 16));
          pos += 6;
          // Surrogate pair?
          if (cp >= 0xD800 && cp <= 0xDBFF && pos + 5 < json.size() &&
              json[pos] == '\\' && json[pos + 1] == 'u') {
            const auto lo = static_cast<char32_t>(
                std::stoul(json.substr(pos + 2, 4), nullptr, 16));
            const char32_t full =
                0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            pos += 6;
            // encode full to UTF-8
            text8.push_back(static_cast<char>(0xF0 | (full >> 18)));
            text8.push_back(static_cast<char>(0x80 | ((full >> 12) & 0x3F)));
            text8.push_back(static_cast<char>(0x80 | ((full >> 6) & 0x3F)));
            text8.push_back(static_cast<char>(0x80 | (full & 0x3F)));
          } else if (cp < 0x80) {
            text8.push_back(static_cast<char>(cp));
          } else if (cp < 0x800) {
            text8.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            text8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
          } else {
            text8.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            text8.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            text8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
          }
          (void)text16;
        } else {
          if (e == 'n') {
            text8.push_back('\n');
          } else if (e == 't') {
            text8.push_back('\t');
          } else if (e == 'r') {
            text8.push_back('\r');
          } else if (e == '0') {
            text8.push_back('\0');
          } else {
            text8.push_back(e);
          }
          pos += 2;
        }
      } else {
        text8.push_back(json[pos++]);
      }
    }
    if (pos >= json.size()) {
      error = "golden: unterminated text";
      return false;
    }
    ++pos;  // closing quote
    pos = json.find("\"ids\"", pos);
    if (pos == std::string::npos) {
      error = "golden: missing ids";
      return false;
    }
    pos = json.find('[', pos);
    if (pos == std::string::npos) {
      error = "golden: missing ids array";
      return false;
    }
    ++pos;
    std::vector<std::uint32_t> ids;
    while (true) {
      skip_ws();
      if (pos < json.size() && json[pos] == ']') {
        ++pos;
        break;
      }
      std::size_t end = pos;
      while (end < json.size() && json[end] != ',' && json[end] != ']') ++end;
      ids.push_back(static_cast<std::uint32_t>(
          std::stoul(json.substr(pos, end - pos))));
      pos = end;
      skip_ws();
      if (pos < json.size() && json[pos] == ',') ++pos;
    }
    out.emplace_back(text8, ids);
  }
}

int TestGoldens(const std::string& model_path, const std::string& data_dir) {
  std::string error;
  auto model = gufo::models::bge_m3::ModelRegistry::Instance().Create(
      "bge-m3", model_path, &error);
  if (model == nullptr) {
    std::cerr << "golden: model load failed: " << error << "\n";
    return 1;
  }
  // NOTE: the registry lives in engine.cpp; this test links gufo_core, so
  // the self-registration runs. Include engine.hpp for ModelRegistry.
  std::size_t checked = 0;
  std::size_t mismatched = 0;
  // bge_tok_golden.json holds HF outputs with specials (compare Encode);
  // bge_tok_golden2.json holds bare bodies (compare EncodeBody).
  const std::pair<const char*, bool> files[] = {
      {"bge_tok_golden.json", true}, {"bge_tok_golden2.json", false}};
  for (const auto [file, is_wrapped] : files) {
    const std::string json = ReadFile(data_dir + "/" + file);
    if (json.empty()) {
      std::cerr << "golden: cannot read " << file << "\n";
      return 1;
    }
    std::vector<std::pair<std::string, std::vector<std::uint32_t>>> probes;
    if (!ParseGoldens(json, probes, error)) {
      std::cerr << "golden: " << file << ": " << error << "\n";
      return 1;
    }
    std::size_t file_mismatched = 0;
    for (const auto& [text, want] : probes) {
      // Wrapped file: compare Encode (adds [CLS]/[SEP]). Body file:
      // compare EncodeBody directly.
      auto got = is_wrapped ? model->Tokenizer().Encode(text, 8192)
                            : model->Tokenizer().EncodeBody(text, 8192);
      ++checked;
      if (!got.has_value() || *got != want) {
        if (file_mismatched < 5) {
          std::cerr << "golden mismatch in " << file << " for text:";
          for (const unsigned char c : text) {
            std::cerr << (c >= 32 && c < 127 ? static_cast<char>(c) : '?');
          }
          std::cerr << "\n  want[" << want.size() << "]:";
          for (std::size_t i = 0; i < want.size() && i < 12; ++i) {
            std::cerr << " " << want[i];
          }
          std::cerr << "\n  got ";
          if (!got.has_value()) {
            std::cerr << "<refused>";
          } else {
            std::cerr << "[" << got->size() << "]:";
            for (std::size_t i = 0; i < got->size() && i < 12; ++i) {
              std::cerr << " " << (*got)[i];
            }
          }
          std::cerr << "\n";
        }
        ++mismatched;
        ++file_mismatched;
      }
    }
    std::cout << "tokenizer golden " << file << ": " << probes.size()
              << " vectors, " << file_mismatched << " mismatches\n";
  }
  std::cout << "tokenizer golden: " << checked << " vectors, " << mismatched
            << " mismatches\n";
  return mismatched == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (TestTinyTokenizer() != 0) return 1;
  TestTinyRefusals();
  std::cout << "tokenizer synthetic: all assertions passed\n";
  const char* model_path = std::getenv("GUFO_BGE_MODEL");
  if (model_path == nullptr || model_path[0] == '\0') {
    std::cout << "tokenizer golden: skipped (GUFO_BGE_MODEL unset)\n";
    return 77;  // CTest SKIP_RETURN_CODE: external-model gate
  }
  std::string data_dir =
      argc > 1 ? argv[1] : std::string(GUFO_BGE_TESTDATA_DIR);
  return TestGoldens(model_path, data_dir);
}
