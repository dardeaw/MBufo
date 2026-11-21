#pragma once
// NPU admission-queue compatibility aliases backed by shared
// src/core/admission_queue.hpp. Header-only; zero hardware dependency.

#include "src/core/admission_queue.hpp"

namespace mbufo::core {

using NpuRequestPhase = AdmissionPhase;
using NpuAdmissionPolicy = AdmissionPolicy;
using NpuAdmissionQueue = AdmissionQueue<>;

}  // namespace mbufo::core
