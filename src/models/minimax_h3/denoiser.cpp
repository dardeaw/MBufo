#include "src/models/minimax_h3/denoiser.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include "src/models/minimax_h3/dit.hpp"

namespace mbufo::minimax_h3 {
namespace {

constexpr std::size_t kGateModalities = 3;
constexpr std::size_t kGateSlots = 6;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

}  // namespace

std::optional<double> DenoiserGateScore(
    std::span<const std::uint16_t> modulation, std::uint32_t time_rows,
    std::string* error) {
  constexpr std::size_t kValuesPerRow =
      kGateModalities * kGateSlots * kDitHiddenSize;
  if (time_rows == 0 ||
      static_cast<std::size_t>(time_rows) >
          std::numeric_limits<std::size_t>::max() / kValuesPerRow ||
      modulation.size() !=
          static_cast<std::size_t>(time_rows) * kValuesPerRow) {
    SetError(error, "invalid MiniMax H3 block modulation for gate scoring");
    return std::nullopt;
  }
  double total = 0.0;
  std::size_t samples = 0;
  for (std::size_t row = 0; row < time_rows; ++row) {
    for (std::size_t modality = 0; modality < kGateModalities; ++modality) {
      for (const std::size_t slot : {std::size_t{2}, std::size_t{5}}) {
        const std::size_t base = ((row * kGateModalities * kGateSlots +
                                   modality * kGateSlots + slot) *
                                  kDitHiddenSize);
        for (std::size_t column = 0; column < kDitHiddenSize; ++column) {
          const std::uint32_t bits =
              static_cast<std::uint32_t>(modulation[base + column]) << 16U;
          const float value = std::bit_cast<float>(bits);
          if (!std::isfinite(value)) {
            SetError(error,
                     "non-finite MiniMax H3 AdaLN gate during block ranking");
            return std::nullopt;
          }
          total += std::abs(static_cast<double>(value));
        }
        samples += kDitHiddenSize;
      }
    }
  }
  return total / static_cast<double>(samples);
}

std::optional<std::vector<std::size_t>> SelectGateRankedDenoiserBlocks(
    std::span<const double> gate_scores, int active_blocks,
    std::string* error) {
  if (gate_scores.size() != kDitBlocks || active_blocks < 3 ||
      active_blocks > static_cast<int>(kDitBlocks)) {
    SetError(error, "invalid MiniMax H3 gate-ranked block selection");
    return std::nullopt;
  }
  struct BlockScore {
    std::size_t block;
    double score;
  };
  std::vector<BlockScore> ranked;
  ranked.reserve(kDitBlocks - 3);
  for (std::size_t block = 2; block + 1 < kDitBlocks; ++block) {
    if (!std::isfinite(gate_scores[block]) || gate_scores[block] < 0.0) {
      SetError(error, "invalid MiniMax H3 AdaLN gate score");
      return std::nullopt;
    }
    ranked.push_back({block, gate_scores[block]});
  }
  // Reproduce h3.c's strict-less-than selection sort, including its tie
  // behavior, rather than substituting a stable or implementation-defined
  // library sort.
  for (std::size_t left = 0; left < ranked.size(); ++left) {
    std::size_t least = left;
    for (std::size_t right = left + 1; right < ranked.size(); ++right) {
      if (ranked[right].score < ranked[least].score) {
        least = right;
      }
    }
    std::swap(ranked[left], ranked[least]);
  }
  std::vector<bool> active(kDitBlocks, true);
  const std::size_t skipped =
      kDitBlocks - static_cast<std::size_t>(active_blocks);
  for (std::size_t index = 0; index < skipped; ++index) {
    active[ranked[index].block] = false;
  }
  std::vector<std::size_t> selected;
  selected.reserve(static_cast<std::size_t>(active_blocks));
  for (std::size_t block = 0; block < kDitBlocks; ++block) {
    if (active[block]) {
      selected.push_back(block);
    }
  }
  return selected;
}

#if !defined(ENGINE_ENABLE_HIP)

struct DenoiserSession::Impl {};

DenoiserSession::DenoiserSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DenoiserSession::~DenoiserSession() = default;
DenoiserSession::DenoiserSession(DenoiserSession&&) noexcept = default;
DenoiserSession& DenoiserSession::operator=(DenoiserSession&&) noexcept =
    default;

std::unique_ptr<DenoiserSession> DenoiserSession::Create(
    const ModelInventory&, const DenoiserOptions&, const DenoiserConditioning&,
    const CancellationToken*, DenoiserTelemetry*, std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 denoising requires a HIP-enabled build";
  }
  return nullptr;
}

bool DenoiserSession::Forward(int, std::span<const float>,
                              std::span<const float>, const CancellationToken*,
                              DenoiserVelocity*, DenoiserTelemetry*,
                              std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 denoising requires a HIP-enabled build";
  }
  return false;
}

bool DenoiserSession::Denoise(std::span<float>, std::span<float>,
                              const CancellationToken*, DenoiserTelemetry*,
                              std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 denoising requires a HIP-enabled build";
  }
  return false;
}

bool DenoiserSession::DenoiseWithProgress(std::span<float>, std::span<float>,
                                          const CancellationToken*,
                                          DenoiserProgress, void*,
                                          DenoiserTelemetry*,
                                          std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 denoising requires a HIP-enabled build";
  }
  return false;
}

const GenerationGeometry& DenoiserSession::geometry() const noexcept {
  static const GenerationGeometry empty;
  return empty;
}

const PackedLayout& DenoiserSession::layout() const noexcept {
  static const PackedLayout empty;
  return empty;
}

const SigmaSchedule& DenoiserSession::schedule() const noexcept {
  static const SigmaSchedule empty;
  return empty;
}

std::span<const std::uint16_t> DenoiserSession::block0_modulation()
    const noexcept {
  return {};
}

std::span<const std::uint16_t> DenoiserSession::refined_text() const noexcept {
  return {};
}

#endif

}  // namespace mbufo::minimax_h3
