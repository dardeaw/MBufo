#ifndef MBUFO_SERVER_AUDIO_ASR_API_HPP_
#define MBUFO_SERVER_AUDIO_ASR_API_HPP_

#include <string_view>

#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/http_server.hpp"

namespace mbufo::server {

inline constexpr std::string_view kAudioAsrApiSchema = "mbufo.audio-asr-api.v1";

[[nodiscard]] bool IsAudioAsrApiPath(std::string_view path) noexcept;
[[nodiscard]] HttpResponse HandleAudioAsrApiRequest(const HttpRequest& request,
                                                    AsrService& service);

}  // namespace mbufo::server

#endif  // MBUFO_SERVER_AUDIO_ASR_API_HPP_
