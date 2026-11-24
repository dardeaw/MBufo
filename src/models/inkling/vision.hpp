#ifndef MBUFO_MODELS_INKLING_VISION_HPP_
#define MBUFO_MODELS_INKLING_VISION_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/core/image.hpp"

namespace mbufo::models::inkling {

/// Vision encoder for Inkling-2-Mini (mmproj-BF16.gguf, projector_type=inkling2v).
/// Encodes an RGB8 image (resized to 640x640) into [256, 2304] F32 embeddings on GPU.
class VisionEncoder {
 public:
  static constexpr std::uint32_t kImageSize = 640;
  static constexpr std::uint32_t kPatchSize = 20;
  static constexpr std::uint32_t kGridSide = kImageSize / kPatchSize;  // 32
  static constexpr std::uint32_t kNumPatches = kGridSide * kGridSide;  // 1024
  static constexpr std::uint32_t kSeqLen = kNumPatches + 1;            // 1025
  static constexpr std::uint32_t kHidden = 1024;
  static constexpr std::uint32_t kFfn = 4096;
  static constexpr std::uint32_t kHeads = 16;
  static constexpr std::uint32_t kHeadDim = 64;
  static constexpr std::uint32_t kBlocks = 23;
  static constexpr std::uint32_t kMergeSide = kGridSide / 2;           // 16
  static constexpr std::uint32_t kMergedTokens = kMergeSide * kMergeSide;  // 256
  static constexpr std::uint32_t kMergeDim = kHidden * 4;              // 4096

  ~VisionEncoder();
  VisionEncoder(const VisionEncoder&) = delete;
  VisionEncoder& operator=(const VisionEncoder&) = delete;

  [[nodiscard]] static std::unique_ptr<VisionEncoder> Load(
      const std::string& mmproj_path, std::uint32_t proj_dim,
      hipStream_t stream, std::string* error_msg);

  /// Encodes `image` into `out_host` ([256 * proj_dim] floats).
  [[nodiscard]] bool Encode(const core::Image& image,
                            std::vector<float>* out_host,
                            std::string* error_msg);

  [[nodiscard]] std::uint32_t ProjDim() const noexcept { return proj_dim_; }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept {
    return resident_bytes_;
  }

 private:
  VisionEncoder() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::uint32_t proj_dim_{0};
  std::size_t resident_bytes_{0};
};

}  // namespace mbufo::models::inkling

#endif  // MBUFO_MODELS_INKLING_VISION_HPP_
