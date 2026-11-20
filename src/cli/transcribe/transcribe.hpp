#ifndef MBUFO_CLI_TRANSCRIBE_HPP_
#define MBUFO_CLI_TRANSCRIBE_HPP_

#include <span>
#include <string_view>

namespace mbufo::cli {

void PrintTranscribeHelp(std::string_view program_name);
[[nodiscard]] int RunTranscribe(std::span<const char* const> args);

}  // namespace mbufo::cli

#endif  // MBUFO_CLI_TRANSCRIBE_HPP_
