#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/bench/bench.hpp"
#include "src/cli/prompt/prompt.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void CheckBackend(std::initializer_list<const char*> arguments,
                  std::string_view backend) {
  const std::span<const char* const> args(arguments.begin(), arguments.size());
  std::string error;
  const auto prompt = mbufo::cli::ParsePromptOptions(args, &error);
  Expect(prompt.has_value(), error);
  const auto bench = mbufo::cli::ParseBenchOptions(args, &error);
  Expect(bench.has_value(), error);
  Expect(prompt->speculative_backend == backend &&
             bench->speculative_backend == backend,
         "prompt and benchmark select the same DSpark mode");
  Expect(prompt->dspark_model_path == "support.gguf" &&
             bench->dspark_model_path == "support.gguf",
         "both commands retain the support artifact");
}

int CheckSampledChat() {
  const char* model = std::getenv("MBUFO_DEEPSEEK_V4_FLASH_MODEL");
  const char* support = std::getenv("MBUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL");
  if (!model || !*model || !support || !*support) {
    std::cout << "SKIP: target and DSpark artifacts are required\n";
    return 77;
  }
  constexpr const char* prompt =
      "Continue this pattern for twenty more terms: red, blue, blue, red, "
      "blue, blue,";
  const std::string input = std::string(prompt) +
                            "\nExplain the rule in one short sentence.\nRepeat "
                            "the rule again.\nexit\n";
  const auto execute = [&](bool chat, bool dspark) {
    std::vector<const char*> args{
        "--model",          model,
        "--system",         "You are a concise assistant.",
        "--max-tokens",     "32",
        "--temperature",    "0.6",
        "--top-p",          "0.95",
        "--seed",           "7",
        "--repeat-penalty", "1.1",
        "--verbose"};
    if (dspark)
      args.insert(args.end(),
                  {"--dspark-model", support, "--speculative", "dspark"});
    if (!chat)
      args.insert(args.end(), {"--prompt", prompt, "--no-display-prompt"});
    std::istringstream incoming(input);
    std::ostringstream output, diagnostics;
    struct RestoreStreams {
      std::streambuf *in, *out, *err;
      ~RestoreStreams() {
        std::cin.rdbuf(in);
        std::cout.rdbuf(out);
        std::cerr.rdbuf(err);
      }
    } restore{std::cin.rdbuf(incoming.rdbuf()), std::cout.rdbuf(output.rdbuf()),
              std::cerr.rdbuf(diagnostics.rdbuf())};
    const int status =
        chat ? mbufo::cli::RunChat(args) : mbufo::cli::RunPrompt(args);
    if (status != 0)
      throw std::runtime_error("CLI generation failed: " + diagnostics.str());
    std::vector<std::string> traces;
    std::istringstream lines(diagnostics.str());
    for (std::string line; std::getline(lines, line);)
      if (line.starts_with("[TokenTrace]:"))
        traces.push_back(line);
    if (dspark && diagnostics.str().find("[Speculative]: acceptance=") ==
                      std::string::npos)
      throw std::runtime_error("sampled CLI did not execute DSpark");
    return traces;
  };
  const auto autoregressive = execute(true, false);
  const auto speculative = execute(true, true);
  Expect(autoregressive.size() == 3 && speculative == autoregressive,
         "sampled chat retains AR token identity after length and EOS stops");
  const auto single_prompt = execute(false, true);
  Expect(single_prompt.size() == 1 &&
             single_prompt.front() == autoregressive.front(),
         "sampled prompt and first chat turn share the same token trace");
  std::cout
      << "Sampled prompt/chat: three turns match AR, with DSpark active\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--sampled-chat")
    return CheckSampledChat();
  CheckBackend({"--dspark-model", "support.gguf"}, "dspark");
  CheckBackend({"--speculative", "dspark", "--dspark-model", "support.gguf"},
               "dspark");
  CheckBackend({"--speculative", "off", "--dspark-model", "support.gguf"}, "");
  CheckBackend({"--dspark-model", "support.gguf", "--speculative", "off"}, "");

  const char* args[] = {"--dspark-model", "support.gguf", "--draft-tokens",
                        "3"};
  const auto prompt = mbufo::cli::ParsePromptOptions(args);
  const auto bench = mbufo::cli::ParseBenchOptions(args);
  Expect(
      prompt && bench && prompt->draft_tokens == 3 && bench->draft_tokens == 3,
      "both commands retain the requested draft budget");
  const char* sweep_args[] = {
      "--concurrency",          "1,2,4,6,8", "-p", "2048", "-n", "128", "-d",
      "0,4096,8192,12288,16384"};
  const auto sweep = mbufo::cli::ParseBenchOptions(sweep_args);
  Expect(sweep &&
             sweep->concurrency == std::vector<std::size_t>{1, 2, 4, 6, 8} &&
             sweep->n_prompts == std::vector<std::size_t>{2048} &&
             sweep->n_gens == std::vector<std::size_t>{128} &&
             sweep->n_depths ==
                 std::vector<std::size_t>{0, 4096, 8192, 12288, 16384},
         "DS4 benchmark retains exact concurrency, prompt, generation, and "
         "depth workload");
  for (const char* invalid : {"0", "9", "2,0", "-1", ""}) {
    const char* invalid_args[] = {"--concurrency", invalid};
    Expect(!mbufo::cli::ParseBenchOptions(invalid_args),
           "invalid DS4 concurrency is rejected");
  }
  std::cout << "DS4 CLI option contract passed\n";
}
