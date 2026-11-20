#include "src/models/minimax_h3/dit.hpp"

#include <utility>

namespace mbufo::minimax_h3 {

#if !defined(ENGINE_ENABLE_HIP)

struct DitBlockSession::Impl {};

DitBlockSession::DitBlockSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DitBlockSession::~DitBlockSession() = default;
DitBlockSession::DitBlockSession(DitBlockSession&&) noexcept = default;
DitBlockSession& DitBlockSession::operator=(DitBlockSession&&) noexcept =
    default;

std::unique_ptr<DitBlockSession> DitBlockSession::Create(
    const ModelInventory&, const DitBlockOptions&, const CancellationToken*,
    std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 DiT requires a HIP-enabled build";
  }
  return nullptr;
}

bool DitBlockSession::Run(const DitBlockInput&, const CancellationToken*,
                          DitBlockRetained*, DitBlockTelemetry*,
                          std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 DiT requires a HIP-enabled build";
  }
  return false;
}

bool DitBlockSession::PrepareConstants(std::span<const std::uint16_t>,
                                       std::span<const std::uint16_t>,
                                       std::span<const std::uint16_t>,
                                       const CancellationToken*,
                                       std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 DiT requires a HIP-enabled build";
  }
  return false;
}

bool DitBlockSession::RunPrepared(const DitPreparedInput&,
                                  const CancellationToken*,
                                  std::span<std::uint16_t>, DitBlockTelemetry*,
                                  std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 DiT requires a HIP-enabled build";
  }
  return false;
}

std::size_t DitBlockSession::rows() const noexcept {
  return 0;
}
std::size_t DitBlockSession::block_index() const noexcept {
  return 0;
}
std::uintptr_t DitBlockSession::scratch_address() const noexcept {
  return 0;
}
std::uint64_t DitBlockSession::weight_bytes() const noexcept {
  return 0;
}
std::uint64_t DitBlockSession::activation_bytes() const noexcept {
  return 0;
}
std::uint64_t DitBlockSession::gemm_plan_validations() const noexcept {
  return 0;
}
double DitBlockSession::load_ms() const noexcept {
  return 0.0;
}

#endif

}  // namespace mbufo::minimax_h3
