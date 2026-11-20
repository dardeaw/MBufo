#ifndef MBUFO_SERVER_AUDIO_TTS_API_HPP_
#define MBUFO_SERVER_AUDIO_TTS_API_HPP_

#include <string_view>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/tts_service.hpp"

namespace mbufo::server {

inline constexpr std::string_view kAudioTtsApiSchema = "mbufo.audio-tts-api.v1";

[[nodiscard]] bool IsAudioTtsApiPath(std::string_view path) noexcept;
[[nodiscard]] HttpResponse HandleAudioTtsApiRequest(const HttpRequest& request,
                                                    TtsService& service);
[[nodiscard]] HttpResponse ValidateAudioTtsRequest(const HttpRequest& request,
                                                   TtsService& service);

}  // namespace mbufo::server

#endif  // MBUFO_SERVER_AUDIO_TTS_API_HPP_
