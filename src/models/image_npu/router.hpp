#pragma once
// Image (FLUX.2-klein / Z-Image-Turbo) NPU router seam.
// Both models share the Qwen3-4B text encoder on NPU; DiT stays GPU until
// P4/P5 land, so the seam exposes two independent bits: TE ready? DiT ready?

#include <cstddef>
#include <cstdint>

namespace mbufo::models::image_npu {

enum class ImageNpuRoute : std::uint8_t { kNone = 0, kTeOnly = 1, kFullDit = 2 };

[[nodiscard]] inline ImageNpuRoute DecideNpuImage(bool te_resident,
                                                 bool dit_resident,
                                                 bool device_ready) noexcept {
  if (!device_ready) return ImageNpuRoute::kNone;
  if (te_resident && dit_resident) return ImageNpuRoute::kFullDit;
  if (te_resident) return ImageNpuRoute::kTeOnly;
  return ImageNpuRoute::kNone;
}

}  // namespace mbufo::models::image_npu
