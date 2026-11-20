#ifndef MBUFO_MODELS_MINIMAX_H3_SHA256_HPP_
#define MBUFO_MODELS_MINIMAX_H3_SHA256_HPP_

#include <filesystem>
#include <span>
#include <string>

namespace mbufo::minimax_h3 {

[[nodiscard]] std::string Sha256(std::span<const unsigned char> bytes);
[[nodiscard]] std::string Sha256File(const std::filesystem::path& path);

}  // namespace mbufo::minimax_h3

#endif  // MBUFO_MODELS_MINIMAX_H3_SHA256_HPP_
