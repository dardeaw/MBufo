// SentencePiece-Unigram tokenizer for bge-m3 (XLM-RoBERTa).
//
// bge-m3's vocabulary is NOT BPE: HF tokenizer.json declares
// model.type "Unigram" with zero merges, and the GGUF carries the Unigram
// lattice data instead of merges: `tokenizer.ggml.tokens` (250002 strings),
// `tokenizer.ggml.scores` (Unigram log-probs; specials carry 0.0), and
// `tokenizer.ggml.token_type` (1 = normal lattice piece, 2 = unknown,
// 3 = control, 5 = unused). The Qwen/Gemma BPE engine cannot read this, so
// bge-m3 ships its own Viterbi port:
//
//   normalize  = HF-Precompiled behavior, reproduced observably:
//                invalid UTF-8 -> U+FFFD -> space; explicit space/delete
//                codepoint tables (see kSpaceRanges/kDeleteRanges, derived
//                from bge_norm_tables.json); ICU NFKC for the rest, with a
//                minimal exemption map for codepoints the HF table keeps
//                verbatim that NFKC would rewrite (currently U+FF5E).
//   pre-split  = ASCII-space words after collapsing/stripping, each word
//                prefixed with U+2581 (Metaspace, prepend always).
//   lattice    = Viterbi shortest path over normal pieces (token_type 1);
//                maximal unmatchable byte runs emit a single <unk>.
//   specials   = literal <s> </s> <pad> <unk> <mask> split before
//                normalization and mapped to the GGUF bos/eos/padding
//                /unknown ids (<mask> is HF 250001, the unused row).
//                NOTE: id 250001 is `[PAD250000]`/unused in the GGUF vocab
//                (llama.cpp replaced the HF `<mask>`), but HF emits 250001
//                for a literal "<mask>", so the port mirrors that.
//   wrap       = single-sequence [CLS] body [SEP] ([0] ... [2]).
//
// Golden parity: tests/models/bge_m3/testdata/*.json (343 HF-generated
// vectors) must match exactly; any NFKC-version drift shows up there first.
#ifndef GUFO_MODELS_BGE_M3_TOKENIZER_HPP_
#define GUFO_MODELS_BGE_M3_TOKENIZER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::bge_m3 {

class SpmUnigramTokenizer {
 public:
  SpmUnigramTokenizer(const SpmUnigramTokenizer&) = delete;
  SpmUnigramTokenizer& operator=(const SpmUnigramTokenizer&) = delete;
  SpmUnigramTokenizer(SpmUnigramTokenizer&&) noexcept = default;
  SpmUnigramTokenizer& operator=(SpmUnigramTokenizer&&) noexcept = default;

  /// Builds the tokenizer from GGUF `tokenizer.ggml.*` arrays. Refuses when
  /// tokens/scores/token_type are missing, ragged, or empty.
  [[nodiscard]] static std::unique_ptr<SpmUnigramTokenizer> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Single-sequence encode with [CLS]/[SEP] wrap. Returns nullopt only when
  /// the normalized body exceeds `max_body_tokens` (callers size that bound;
  /// the embedding session passes the leg-1 window).
  [[nodiscard]] std::optional<std::vector<std::uint32_t>> Encode(
      std::string_view text, std::size_t max_body_tokens) const;

  /// Raw body encode without the [CLS]/[SEP] wrap (tests, drivers).
  [[nodiscard]] std::optional<std::vector<std::uint32_t>> EncodeBody(
      std::string_view text, std::size_t max_body_tokens) const;

  [[nodiscard]] std::size_t VocabSize() const noexcept { return pieces_.size(); }
  [[nodiscard]] std::uint32_t ClsId() const noexcept { return cls_id_; }
  [[nodiscard]] std::uint32_t SepId() const noexcept { return sep_id_; }
  [[nodiscard]] std::uint32_t UnkId() const noexcept { return unk_id_; }

 private:
  SpmUnigramTokenizer() = default;

  struct TrieNode {
    std::int32_t piece_id{-1};  // allowed-piece index, or -1
    // (byte, child) sorted by byte for binary search.
    std::vector<std::pair<unsigned char, std::uint32_t>> next;
  };

  [[nodiscard]] std::vector<std::uint32_t> ViterbiWord(
      const std::string& word_bytes) const;
  void InsertPiece(std::size_t piece_index);

  std::vector<std::string> pieces_;
  std::vector<float> scores_;
  std::vector<TrieNode> trie_;
  std::size_t max_piece_bytes_{0};
  std::uint32_t cls_id_{0};
  std::uint32_t sep_id_{2};
  std::uint32_t pad_id_{1};
  std::uint32_t unk_id_{3};
};

}  // namespace gufo::models::bge_m3

#endif  // GUFO_MODELS_BGE_M3_TOKENIZER_HPP_
