// Minimal core-library anchor for the L0/L1 engine abstraction.
//
// All registry logic is inline in engine_batch.hpp (mirroring
// speculative::DraftBackendRegistry); this TU only provides the
// process-wide EngineRegistry::Instance definition so the symbol lives in
// gufo_core and header-only consumers link against exactly one instance.

#include "src/core/engine/engine_batch.hpp"

namespace gufo::core::engine {

EngineRegistry& EngineRegistry::Instance() {
  static EngineRegistry instance;
  return instance;
}

}  // namespace gufo::core::engine
