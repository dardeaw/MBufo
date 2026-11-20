#ifndef MBUFO_CLI_EVAL_EVAL_HPP_
#define MBUFO_CLI_EVAL_EVAL_HPP_

#include <span>
#include <string_view>

namespace mbufo::cli {

void PrintEvalHelp(std::string_view program_name);
int RunEval(std::span<const char* const> args);

}  // namespace mbufo::cli

#endif  // MBUFO_CLI_EVAL_EVAL_HPP_
