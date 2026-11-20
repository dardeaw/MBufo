#ifndef MBUFO_CORE_SESSION_MODE_HPP_
#define MBUFO_CORE_SESSION_MODE_HPP_

namespace mbufo::core {

/// Per-session execution choice, independent of resident model capabilities.
enum class SessionMode { kAutoregressive, kSpeculative };

}  // namespace mbufo::core

#endif  // MBUFO_CORE_SESSION_MODE_HPP_
