#ifndef MBUFO_CORE_QUANT_QUANT_STATUS_HPP_
#define MBUFO_CORE_QUANT_QUANT_STATUS_HPP_

namespace mbufo::core::quant {

/// Result of a type-dispatched core operation. Callers must look at it: an unsupported format or
/// shape is reported, never decoded as some other format.
enum class Status : int {
  kOk,
  kUnsupportedType,  ///< format code not in core::quant::AllFormats, or no kernel for this operation
  kBadShape,         ///< null pointer, zero size, or columns not a whole number of blocks
  kScratchTooSmall,  ///< caller scratch smaller than the operation needs
};

}  // namespace mbufo::core::quant

#endif  // MBUFO_CORE_QUANT_QUANT_STATUS_HPP_
