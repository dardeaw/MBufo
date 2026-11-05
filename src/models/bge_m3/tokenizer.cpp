// SpmUnigramTokenizer implementation (see tokenizer.hpp for the contract).
#include "src/models/bge_m3/tokenizer.hpp"

#include <unicode/normalizer2.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace gufo::models::bge_m3 {
namespace {

// HF-Precompiled whitespace table for XLM-R (bge_norm_tables.json):
// every codepoint whose 'a'+ch+'b' probe tokenizes identically to 'a b'.
constexpr std::uint32_t kSpaceRanges[][2] = {
    {9, 10}, {12, 13}, {32, 32}, {133, 133}, {160, 160}, {5760, 5760},
    {8192, 8207}, {8232, 8233}, {8239, 8239}, {8287, 8287}, {9601, 9601},
    {12288, 12288}, {65279, 65279}, {65533, 65533},
};
// Codepoints deleted outright ('a'+ch+'b' tokenizes as 'ab').
constexpr std::uint32_t kDeleteRanges[][2] = {
    {1, 8}, {11, 11}, {14, 31}, {127, 127}, {143, 143}, {159, 159},
};

constexpr char32_t kReplacementChar = 0xFFFDu;
constexpr char32_t kAsciiSpace = 0x20u;

// Codepoints the HF Precompiled normalizer keeps verbatim (or expands
// post-compose) where ICU NFKC would rewrite them. Derived empirically:
// every codepoint 0..0x10FFFF was run through both the HuggingFace
// `tokenizers` Precompiled normalizer (BAAI/bge-m3 tokenizer.json) and NFKC,
// plus starter+combining composition contexts and Jamo pairs. Three classes:
//   KEEP    (172 chars incl. U+32FF, U+A7F2..U+A7F4, U+AB69, U+FF5E, U+10781..,
//            U+1CCD6.., U+1E030.., U+1F16C, U+1F23B, U+1FBF0..): Precompiled
//            keeps them; NFKC would fold them. All are ccc=0, so shielding
//            them from composition/reordering is exact.
//   JAMO    (U+1100..U+1112, U+1161..U+1175, U+11A8..U+11C2): Precompiled
//            never composes conjoining Jamo; NFKC would.
//   EXPAND  (U+0344 only): Precompiled emits U+0308 U+0301 *after* the
//            composition phase, so a preceding starter must not compose
//            with it (HF "a+U+0344" -> a + 0308 + 0301, while ICU NFKC
//            gives a-with-diaeresis + 0301).
// Mechanism: shielded codepoints are substituted with plane-15 PUA
// placeholders before NFKC (PUA is stable under NFKC and blocks composition
// across the boundary, matching Precompiled), then restored verbatim --
// except U+0344, which is restored as its two-codepoint expansion.
constexpr std::uint32_t kKeepRanges[][2] = {
    {0x32FF, 0x32FF}, {0xA7F2, 0xA7F4}, {0xAB69, 0xAB69}, {0xFF5E, 0xFF5E},
    {0x10781, 0x10785}, {0x10787, 0x107B0}, {0x107B2, 0x107BA},
    {0x1100, 0x1112}, {0x1161, 0x1175}, {0x11A8, 0x11C2},
    {0x1CCD6, 0x1CCF9}, {0x1E030, 0x1E06D}, {0x1F16C, 0x1F16C},
    {0x1F23B, 0x1F23B}, {0x1FBF0, 0x1FBF9},
};
constexpr std::uint32_t kExpand0344 = 0x0344;

[[nodiscard]] bool IsShielded(std::uint32_t cp) {
  if (cp == kExpand0344) return true;
  for (const auto& r : kKeepRanges) {
    if (cp >= r[0] && cp <= r[1]) return true;
  }
  return false;
}

[[nodiscard]] bool IsSpaceCp(std::uint32_t cp) {
  for (const auto& r : kSpaceRanges) {
    if (cp >= r[0] && cp <= r[1]) return true;
  }
  return false;
}

[[nodiscard]] bool IsDeleteCp(std::uint32_t cp) {
  for (const auto& r : kDeleteRanges) {
    if (cp >= r[0] && cp <= r[1]) return true;
  }
  return false;
}


// Decodes UTF-8 into codepoints; any malformed sequence (truncated, overlong,
// surrogate halves, > U+10FFFF) yields exactly one U+FFFD, matching the HF
// behavior that funnels invalid bytes to the space-mapped replacement char.
void DecodeUtf8(std::string_view text, std::vector<char32_t>& out) {
  out.clear();
  out.reserve(text.size());
  std::size_t i = 0;
  const auto bytes = reinterpret_cast<const unsigned char*>(text.data());
  const std::size_t n = text.size();
  while (i < n) {
    const unsigned char b0 = bytes[i];
    if (b0 < 0x80) {
      out.push_back(b0);
      ++i;
    } else if ((b0 & 0xE0) == 0xC0) {
      if (i + 1 < n && (bytes[i + 1] & 0xC0) == 0x80) {
        const char32_t cp = ((b0 & 0x1F) << 6) | (bytes[i + 1] & 0x3F);
        out.push_back(cp < 0x80 ? kReplacementChar : cp);  // overlong
        i += 2;
      } else {
        out.push_back(kReplacementChar);
        ++i;
      }
    } else if ((b0 & 0xF0) == 0xE0) {
      if (i + 2 < n && (bytes[i + 1] & 0xC0) == 0x80 &&
          (bytes[i + 2] & 0xC0) == 0x80) {
        const char32_t cp = ((b0 & 0x0F) << 12) |
                            ((bytes[i + 1] & 0x3F) << 6) |
                            (bytes[i + 2] & 0x3F);
        out.push_back((cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))
                          ? kReplacementChar
                          : cp);
        i += 3;
      } else {
        out.push_back(kReplacementChar);
        ++i;
      }
    } else if ((b0 & 0xF8) == 0xF0) {
      if (i + 3 < n && (bytes[i + 1] & 0xC0) == 0x80 &&
          (bytes[i + 2] & 0xC0) == 0x80 && (bytes[i + 3] & 0xC0) == 0x80) {
        const char32_t cp = ((b0 & 0x07) << 18) |
                            ((bytes[i + 1] & 0x3F) << 12) |
                            ((bytes[i + 2] & 0x3F) << 6) |
                            (bytes[i + 3] & 0x3F);
        out.push_back((cp < 0x10000 || cp > 0x10FFFF) ? kReplacementChar : cp);
        i += 4;
      } else {
        out.push_back(kReplacementChar);
        ++i;
      }
    } else {
      out.push_back(kReplacementChar);
      ++i;
    }
  }
}

void AppendUtf8(std::string& out, char32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Added-token literal split (surface forms are fixed HF added_tokens;
// ids resolve per instance in EncodeBody from the GGUF header, except
// <mask>, which HF maps to 250001 -- see tokenizer.hpp).
struct AddedToken {
  std::string_view text;
  std::uint32_t id;
};

}  // namespace

void SpmUnigramTokenizer::InsertPiece(std::size_t piece_index) {
  std::uint32_t node = 0;
  const std::string& piece = pieces_[piece_index];
  for (const unsigned char b :
       std::string_view(piece.data(), piece.size())) {
    auto& children = trie_[node].next;
    auto it = std::lower_bound(
        children.begin(), children.end(), b,
        [](const std::pair<unsigned char, std::uint32_t>& e, unsigned char v) {
          return e.first < v;
        });
    if (it != children.end() && it->first == b) {
      node = it->second;
    } else {
      const std::uint32_t fresh = static_cast<std::uint32_t>(trie_.size());
      trie_.emplace_back();
      // `children` may have reallocated via trie_.emplace_back; re-fetch.
      auto& children2 = trie_[node].next;
      auto it2 = std::lower_bound(
          children2.begin(), children2.end(), b,
          [](const std::pair<unsigned char, std::uint32_t>& e,
             unsigned char v) { return e.first < v; });
      children2.insert(it2, {b, fresh});
      node = fresh;
    }
  }
  trie_[node].piece_id = static_cast<std::int32_t>(piece_index);
}

std::unique_ptr<SpmUnigramTokenizer> SpmUnigramTokenizer::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  auto refuse = [&](const std::string& message) {
    if (error_msg != nullptr) *error_msg = message;
    return std::unique_ptr<SpmUnigramTokenizer>{};
  };
  const auto token_views =
      reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  if (token_views.empty()) {
    return refuse("GGUF metadata missing 'tokenizer.ggml.tokens'");
  }
  const auto* scores_meta = reader.FindMetadata("tokenizer.ggml.scores");
  const auto* type_meta = reader.FindMetadata("tokenizer.ggml.token_type");
  if (scores_meta == nullptr || type_meta == nullptr) {
    return refuse("GGUF metadata missing Unigram scores/token_type arrays");
  }
  const auto* scores_vec =
      std::get_if<std::vector<double>>(&scores_meta->value);
  const auto* type_i64 = std::get_if<std::vector<std::int64_t>>(&type_meta->value);
  const auto* type_u64 =
      std::get_if<std::vector<std::uint64_t>>(&type_meta->value);
  if (scores_vec == nullptr || (type_i64 == nullptr && type_u64 == nullptr) ||
      scores_vec->size() != token_views.size() ||
      (type_i64 != nullptr ? type_i64->size()
                           : type_u64->size()) != token_views.size()) {
    return refuse("Unigram scores/token_type arrays are missing or ragged");
  }
  auto tokenizer = std::unique_ptr<SpmUnigramTokenizer>(
      new SpmUnigramTokenizer());
  tokenizer->pieces_.reserve(token_views.size());
  tokenizer->scores_.reserve(token_views.size());
  for (const auto tv : token_views) {
    tokenizer->pieces_.emplace_back(tv);
  }
  for (const double s : *scores_vec) {
    tokenizer->scores_.push_back(static_cast<float>(s));
  }
  // Lattice membership: token_type 1 (normal) participates in Viterbi;
  // 2 = unknown, 3 = control, 5 = unused never match. Non-normal pieces are
  // still addressable by id (specials) but excluded from the trie.
  tokenizer->trie_.emplace_back();  // root
  for (std::size_t i = 0; i < token_views.size(); ++i) {
    const std::int64_t t =
        type_i64 != nullptr
            ? (*type_i64)[i]
            : static_cast<std::int64_t>((*type_u64)[i]);
    if (t != 1) continue;
    if (tokenizer->pieces_[i].empty()) continue;
    tokenizer->InsertPiece(i);
    tokenizer->max_piece_bytes_ =
        std::max(tokenizer->max_piece_bytes_, tokenizer->pieces_[i].size());
  }
  if (auto v = reader.GetMetadataUint32("tokenizer.ggml.bos_token_id")) {
    tokenizer->cls_id_ = *v;
  }
  if (auto v = reader.GetMetadataUint32("tokenizer.ggml.eos_token_id")) {
    tokenizer->sep_id_ = *v;
  }
  if (auto v = reader.GetMetadataUint32("tokenizer.ggml.unknown_token_id")) {
    tokenizer->unk_id_ = *v;
  }
  if (auto v = reader.GetMetadataUint32("tokenizer.ggml.padding_token_id")) {
    tokenizer->pad_id_ = *v;
  }
  if (tokenizer->trie_.size() <= 1) {
    return refuse("Unigram lattice is empty");
  }
  return tokenizer;
}

std::vector<std::uint32_t> SpmUnigramTokenizer::ViterbiWord(
    const std::string& word_bytes) const {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  const std::size_t n = word_bytes.size();
  const auto* bytes =
      reinterpret_cast<const unsigned char*>(word_bytes.data());
  std::vector<double> best(n + 1, kInf);
  std::vector<std::size_t> prev(n + 1, 0);
  // back_id: piece index, or kUnkRun for a maximal unmatchable run.
  constexpr std::size_t kUnkRun = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> back_id(n + 1, kUnkRun);
  best[0] = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(best[i])) continue;
    // Trie walk from i: every allowed piece starting here relaxes its end.
    std::uint32_t node = 0;
    bool matched = false;
    const std::size_t limit =
        std::min(n, i + max_piece_bytes_);
    for (std::size_t j = i; j < limit; ++j) {
      const auto& children = trie_[node].next;
      auto it = std::lower_bound(
          children.begin(), children.end(), bytes[j],
          [](const std::pair<unsigned char, std::uint32_t>& e,
             unsigned char v) { return e.first < v; });
      if (it == children.end() || it->first != bytes[j]) break;
      node = it->second;
      if (trie_[node].piece_id < 0) continue;
      matched = true;
      const std::size_t piece =
          static_cast<std::size_t>(trie_[node].piece_id);
      const double cost =
          best[i] - static_cast<double>(scores_[piece]);
      if (cost < best[j + 1]) {
        best[j + 1] = cost;
        prev[j + 1] = i;
        back_id[j + 1] = piece;
      }
    }
    if (matched) continue;
    // No allowed piece starts at i: consume the maximal run with no piece
    // start inside it as a single <unk> (HF behavior for OOV spans).
    std::size_t j = i + 1;
    while (j < n) {
      std::uint32_t probe = 0;
      bool starts = false;
      const std::size_t probe_limit =
          std::min(n, j + max_piece_bytes_);
      for (std::size_t k = j; k < probe_limit; ++k) {
        const auto& children = trie_[probe].next;
        auto it = std::lower_bound(
            children.begin(), children.end(), bytes[k],
            [](const std::pair<unsigned char, std::uint32_t>& e,
               unsigned char v) { return e.first < v; });
        if (it == children.end() || it->first != bytes[k]) break;
        probe = it->second;
        if (trie_[probe].piece_id >= 0) {
          starts = true;
          break;
        }
      }
      if (starts) break;
      ++j;
    }
    if (best[i] < best[j]) {
      best[j] = best[i];  // unk cost 0: forced gap, never a choice
      prev[j] = i;
      back_id[j] = kUnkRun;
    }
  }
  std::vector<std::uint32_t> ids;
  std::size_t pos = n;
  while (pos > 0) {
    if (back_id[pos] == kUnkRun) {
      // No finite path recorded: still emit one unk per remaining byte run
      // rather than fail (unreachable when the loop above ran; defensive).
      ids.push_back(unk_id_);
      pos = (prev[pos] < pos) ? prev[pos] : pos - 1;
    } else {
      ids.push_back(static_cast<std::uint32_t>(back_id[pos]));
      pos = prev[pos];
    }
  }
  std::reverse(ids.begin(), ids.end());
  return ids;
}

std::optional<std::vector<std::uint32_t>> SpmUnigramTokenizer::EncodeBody(
    std::string_view text, std::size_t max_body_tokens) const {
  // Added-token literals split before normalization.
  std::vector<std::uint32_t> ids;
  std::size_t offset = 0;
  const auto emit_segment = [&](std::string_view segment) -> bool {
    if (segment.empty()) return true;
    std::vector<char32_t> cps;
    DecodeUtf8(segment, cps);
    // Space/delete mapping first (whitespace definition is fixed).
    std::vector<char32_t> mapped;
    mapped.reserve(cps.size());
    for (const char32_t cp : cps) {
      if (IsDeleteCp(cp)) continue;
      mapped.push_back(IsSpaceCp(cp) ? kAsciiSpace : cp);
    }
    // ICU NFKC over the mapped sequence. Exempt codepoints (kNfkcExempt)
    // are substituted with private-use placeholders for the normalization
    // pass and restored verbatim afterwards: they are spacing starters, so
    // placeholder substitution neither merges nor composes across them, and
    // NFKC is stable on private-use characters. (Placeholders live in
    // plane 15; an adversarial input already containing them would alias --
    // documented, not guarded: embedding inputs, and the goldens pin the
    // behavior.)
    constexpr char32_t kPuaBase = 0xF0000u;
    std::vector<char32_t> subst_cps;  // PUA index -> original codepoint
    std::string pre;
    pre.reserve(segment.size() + 4);
    for (const char32_t cp : mapped) {
      if (IsShielded(cp)) {
        std::size_t idx = 0;
        for (; idx < subst_cps.size(); ++idx) {
          if (subst_cps[idx] == cp) break;
        }
        if (idx == subst_cps.size()) subst_cps.push_back(cp);
        AppendUtf8(pre, kPuaBase + static_cast<char32_t>(idx));
      } else {
        AppendUtf8(pre, cp);
      }
    }
    UErrorCode status = U_ZERO_ERROR;
    const icu::Normalizer2* nfkc = icu::Normalizer2::getNFKCInstance(status);
    if (nfkc == nullptr || U_FAILURE(status)) return false;
    icu::UnicodeString src =
        icu::UnicodeString::fromUTF8(icu::StringPiece(pre.data(), pre.size()));
    icu::UnicodeString dst;
    nfkc->normalize(src, dst, status);
    if (U_FAILURE(status)) return false;
    std::string norm;
    dst.toUTF8String(norm);
    // Restore exempt codepoints verbatim from their PUA placeholders.
    std::string clean;
    clean.reserve(norm.size() + subst_cps.size() * 3);
    {
      std::vector<char32_t> norm_cps;
      DecodeUtf8(norm, norm_cps);
      for (const char32_t cp : norm_cps) {
        if (cp >= kPuaBase && cp < kPuaBase + subst_cps.size()) {
          const char32_t orig = subst_cps[cp - kPuaBase];
          if (orig == kExpand0344) {
            // Post-compose expansion: the PUA placeholder blocked composition
            // above, and the expansion itself must not compose below.
            AppendUtf8(clean, 0x0308u);
            AppendUtf8(clean, 0x0301u);
          } else {
            AppendUtf8(clean, orig);
          }
        } else {
          AppendUtf8(clean, cp);
        }
      }
    }
    // Split on ASCII space (only U+0020 survives: every other whitespace
    // was mapped in the first pass; NFKC introduces no new spaces).
    std::size_t w = 0;
    static constexpr std::string_view kMarker = "\xe2\x96\x81";  // U+2581
    while (w < clean.size()) {
      while (w < clean.size() && clean[w] == ' ') ++w;
      if (w >= clean.size()) break;
      std::size_t e = w;
      while (e < clean.size() && clean[e] != ' ') ++e;
      std::string word;
      word.reserve(kMarker.size() + (e - w));
      word.append(kMarker);
      word.append(clean.substr(w, e - w));
      auto word_ids = ViterbiWord(word);
      ids.insert(ids.end(), word_ids.begin(), word_ids.end());
      if (ids.size() > max_body_tokens) return false;
      w = e;
    }
    return true;
  };
  const AddedToken added[] = {
      {"<s>", cls_id_},   {"<pad>", pad_id_}, {"</s>", sep_id_},
      {"<unk>", unk_id_}, {"<mask>", 250001},
  };
  while (offset < text.size()) {
    std::size_t hit_pos = std::string_view::npos;
    const AddedToken* hit = nullptr;
    for (const auto& cand : added) {
      const std::size_t p = text.find(cand.text, offset);
      if (p != std::string_view::npos &&
          (hit_pos == std::string_view::npos || p < hit_pos ||
           (p == hit_pos && cand.text.size() > hit->text.size()))) {
        hit_pos = p;
        hit = &cand;
      }
    }
    if (hit == nullptr) {
      if (!emit_segment(text.substr(offset))) return std::nullopt;
      break;
    }
    if (!emit_segment(text.substr(offset, hit_pos - offset))) {
      return std::nullopt;
    }
    ids.push_back(hit->id);
    if (ids.size() > max_body_tokens) return std::nullopt;
    offset = hit_pos + hit->text.size();
  }
  return ids;
}

std::optional<std::vector<std::uint32_t>> SpmUnigramTokenizer::Encode(
    std::string_view text, std::size_t max_body_tokens) const {
  auto body = EncodeBody(text, max_body_tokens);
  if (!body.has_value()) return std::nullopt;
  std::vector<std::uint32_t> ids;
  ids.reserve(body->size() + 2);
  ids.push_back(cls_id_);
  ids.insert(ids.end(), body->begin(), body->end());
  ids.push_back(sep_id_);
  return ids;
}

}  // namespace gufo::models::bge_m3
