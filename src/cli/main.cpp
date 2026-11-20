#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <span>
#include <string_view>

#include "src/cli/banner.hpp"
#include "src/cli/bench/bench.hpp"
#include "src/cli/diagnose/diagnose.h"
#include "src/cli/eval/eval.hpp"
#include "src/cli/prompt/prompt.hpp"
#include "src/cli/serve/serve.hpp"
#include "src/cli/transcribe/transcribe.hpp"
#include "src/cli/video/video.hpp"

#ifndef MBUFO_RELEASE_VERSION
#define MBUFO_RELEASE_VERSION "development"
#endif
#ifndef MBUFO_REVISION
#define MBUFO_REVISION "unknown"
#endif

constexpr std::string_view kMBufoReleaseVersion = MBUFO_RELEASE_VERSION;
constexpr std::string_view kMBufoRevision = MBUFO_REVISION;

namespace {

void print_version() {
  std::cout << "mbufo version " << kMBufoReleaseVersion << " (" << kMBufoRevision
            << ")\n";
}

void print_help(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name << " <COMMAND> [OPTIONS]\n\n"
      << "Fast LLM & DiT video inference engine for AMD Strix Halo "
         "(gfx1151).\n\n"
      << "Commands:\n"
      << "  serve          Start the OpenAI-compatible HTTP server\n"
      << "  prompt         Execute one prompt request and exit\n"
      << "  chat           Start an interactive terminal conversation\n"
      << "  bench          Benchmark prompt processing and token generation\n"
      << "  eval           Evaluate an OpenAI-compatible text server\n"
      << "  video          Generate MiniMax H3 text-to-video\n"
      << "  transcribe     Transcribe WAV audio with Qwen3-ASR-1.7B\n"
      << "  diagnose       Inspect system hardware, memory, and drivers\n"
      << "  help           Print help for a specific command\n\n"
      << "Run '" << program_name << " help <COMMAND>' or '" << program_name
      << " <COMMAND> --help' for details on a command\n\n"
      << "Options:\n"
      << "  -h, --help     Print help\n"
      << "  -v, --version  Print version\n";
}

int run(std::span<const char* const> args) {
  if (args.empty()) {
    std::cerr << "Error: missing command or option\n";
    print_help("mbufo");
    return 2;
  }

  const std::string_view program_name = args[0];
  const auto options = args.subspan(1);

  if (options.empty()) {
    print_help(program_name);
    return 0;
  }

  const std::string_view first_arg = options[0];
  if (first_arg == "--version" || first_arg == "-v") {
    print_version();
    return 0;
  }

  if (first_arg == "--help" || first_arg == "-h") {
    print_help(program_name);
    return 0;
  }

  if (first_arg == "help") {
    if (options.size() <= 1) {
      print_help(program_name);
      return 0;
    }
    const std::string_view sub = options[1];
    const std::array<const char*, 1> help_flag = {"--help"};
    if (sub == "serve") {
      if (options.size() > 2) {
        const std::array<const char*, 2> serve_help_flags = {options[2],
                                                             "--help"};
        return mbufo::cli::RunServe(serve_help_flags);
      }
      return mbufo::cli::RunServe(help_flag);
    }
    if (sub == "prompt") {
      return mbufo::cli::RunPrompt(help_flag);
    }
    if (sub == "bench") {
      return mbufo::cli::RunBench(help_flag);
    }
    if (sub == "eval") {
      return mbufo::cli::RunEval(help_flag);
    }
    if (sub == "video") {
      return mbufo::cli::RunVideo(help_flag);
    }
    if (sub == "transcribe") {
      return mbufo::cli::RunTranscribe(help_flag);
    }
    if (sub == "chat") {
      return mbufo::cli::RunChat(help_flag);
    }
    if (sub == "probe") {
      return mbufo::cli::RunProbe(help_flag);
    }
    if (sub == "diagnose") {
      return mbufo::cli::RunDiagnose(help_flag);
    }
    std::cerr << "Error: unknown help topic '" << sub << "'\n";
    print_help(program_name);
    return 2;
  }

  mbufo::cli::PrintStartupBanner(kMBufoReleaseVersion, options);

  if (first_arg == "probe") {
    return mbufo::cli::RunProbe(options.subspan(1));
  }

  if (first_arg == "diagnose") {
    return mbufo::cli::RunDiagnose(options);
  }

  if (first_arg == "bench") {
    return mbufo::cli::RunBench(options.subspan(1));
  }

  if (first_arg == "eval") {
    return mbufo::cli::RunEval(options.subspan(1));
  }

  if (first_arg == "prompt") {
    return mbufo::cli::RunPrompt(options.subspan(1));
  }

  if (first_arg == "video") {
    return mbufo::cli::RunVideo(options.subspan(1));
  }

  if (first_arg == "transcribe") {
    return mbufo::cli::RunTranscribe(options.subspan(1));
  }

  if (first_arg == "chat") {
    return mbufo::cli::RunChat(options.subspan(1));
  }

  if (first_arg == "serve") {
    return mbufo::cli::RunServe(options.subspan(1));
  }

  // If invoked as legacy binary name or flag passed directly, route to serve
  if (program_name.ends_with("mbufo-server") ||
      program_name.ends_with("/mbufo-server")) {
    return mbufo::cli::RunServe(options);
  }

  std::cerr << "Error: unknown command '" << first_arg << "'\n";
  print_help(program_name);
  return 2;
}

}  // namespace

// Compatibility with the pre-rename name: a legacy GUFO_<X> variable (service units, scripts) is honored as MBUFO_<X> when
// the new name is not set.
extern char** environ;
static void ImportLegacyEnvironment() {
  std::vector<std::pair<std::string, std::string>> legacy;
  for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
    if (std::strncmp(*e, "GUFO_", 5) != 0) continue;
    const char* eq = std::strchr(*e, '=');
    if (eq == nullptr) continue;
    legacy.emplace_back(std::string("MBUFO_") + std::string(*e + 5, static_cast<std::size_t>(eq - *e - 5)), std::string(eq + 1));
  }
  for (const auto& [name, value] : legacy) {
    if (std::getenv(name.c_str()) == nullptr) ::setenv(name.c_str(), value.c_str(), 0);
  }
}

int main(int argc, char* argv[]) {
  ImportLegacyEnvironment();
  const std::span<const char* const> args(argv, static_cast<std::size_t>(argc));
  return run(args);
}
