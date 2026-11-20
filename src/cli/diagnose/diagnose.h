#ifndef MBUFO_CLI_DIAGNOSE_H_
#define MBUFO_CLI_DIAGNOSE_H_

#include <span>
#include <string_view>

#include "src/core/diagnostics/linux_sysfs.h"
#include "src/core/diagnostics/report.h"

namespace mbufo::cli {

[[nodiscard]] diagnostics::DiagnosticReport CollectDiagnostics(
    const diagnostics::LinuxSysfs& sysfs = diagnostics::LinuxSysfs(),
    std::string_view section = "all");

int RunDiagnose(std::span<const char* const> args);
int RunProbe(std::span<const char* const> args);

}  // namespace mbufo::cli

#endif  // MBUFO_CLI_DIAGNOSE_H_
