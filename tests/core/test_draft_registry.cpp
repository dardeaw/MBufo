// CPU unit test for the unified speculative draft registry.
//
// Covers Register/Create/List/Contains and the MockDraftBackend smoke path.
// GPU-backed factories (dflash2, qwen-mtp, flash-next-mtp, gemma4-mtp) live
// in HIP translation units and are exercised by the GPU correctness runs,
// not here.

#include "src/core/speculative/draft_registry.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

using gufo::speculative::CreateDraftBackend;
using gufo::speculative::DraftBackendInfo;
using gufo::speculative::DraftBackendRegistry;
using gufo::speculative::DraftBackendRequest;
using gufo::speculative::IDraftBackend;
using gufo::speculative::ListDraftBackends;
using gufo::speculative::MockDraftBackend;
using gufo::speculative::RegisterDraftBackend;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

std::unique_ptr<IDraftBackend> MakeMock(const DraftBackendRequest&,
                                        std::string*) {
  return std::make_unique<MockDraftBackend>(
      std::vector<gufo::tokenization::TokenId>{11, 22, 33});
}

}  // namespace

int main() {
  RegisterDraftBackend(
      DraftBackendInfo{.name = "test-mock",
                       .description = "registry unit-test backend",
                       .requires_target_model = false,
                       .requires_session = false},
      MakeMock);
  Expect(DraftBackendRegistry::Instance().Contains("test-mock"),
         "registry contains test-mock after Register");

  // Unknown backend: nullptr + error.
  std::string error;
  DraftBackendRequest req;
  auto missing = CreateDraftBackend("no-such-backend", req, &error);
  Expect(missing == nullptr, "unknown backend returns nullptr");
  Expect(!error.empty(), "unknown backend sets error_msg");

  // Known backend: propose/accept/reset smoke.
  error.clear();
  auto backend = CreateDraftBackend("test-mock", req, &error);
  Expect(backend != nullptr, "test-mock creates a backend");
  Expect(error.empty(), "successful create leaves error empty");
  Expect(backend->Name() == "MockDraftBackend", "mock backend name");
  auto proposal = backend->Propose({}, 0, 2);
  Expect(proposal.tokens.size() == 2, "mock proposes capped tokens");
  Expect(proposal.tokens[0] == 11 && proposal.tokens[1] == 22,
         "mock proposes pool tokens in order");
  Expect(!backend->SupportsSampledProposals(),
         "mock does not support sampled proposals");
  backend->AcceptFeedback({proposal.tokens.data(), 1}, 7);
  backend->Reset();

  // List exposes the registration.
  bool found = false;
  for (const auto& info : ListDraftBackends()) {
    if (info.name == "test-mock") {
      found = true;
      Expect(!info.requires_session && !info.requires_target_model,
             "mock metadata flags");
    }
  }
  Expect(found, "ListDraftBackends contains test-mock");

  // Re-registering replaces the entry.
  RegisterDraftBackend(
      DraftBackendInfo{.name = "test-mock",
                       .description = "replacement",
                       .requires_target_model = true,
                       .requires_session = false},
      MakeMock);
  found = false;
  for (const auto& info : ListDraftBackends()) {
    if (info.name == "test-mock") {
      found = true;
      Expect(info.requires_target_model, "re-registration replaces metadata");
    }
  }
  Expect(found, "test-mock still listed after re-register");

  // Session-bound factory rejects a request without a session handle.
  RegisterDraftBackend(
      DraftBackendInfo{.name = "test-session-bound",
                       .description = "requires session",
                       .requires_session = true},
      [](const DraftBackendRequest& r, std::string* err)
          -> std::unique_ptr<IDraftBackend> {
        if (r.session_kind != "test" || r.session_handle == nullptr) {
          if (err != nullptr) *err = "need test session";
          return nullptr;
        }
        return MakeMock(r, err);
      });
  DraftBackendRequest bad;
  bad.session_kind = "wrong";
  auto rejected = CreateDraftBackend("test-session-bound", bad, &error);
  Expect(rejected == nullptr, "session-bound factory rejects bad handle");
  Expect(!error.empty(), "session-bound factory sets error_msg");

  std::cout << "draft_registry_test: all checks passed\n";
  return 0;
}
