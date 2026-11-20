#ifndef MBUFO_SERVER_AUDIO_WEBSOCKET_HPP_
#define MBUFO_SERVER_AUDIO_WEBSOCKET_HPP_

#include "src/cli/serve/http_server.hpp"

namespace mbufo::server {
HttpResponse HandleTtsWebSocket(const HttpRequest& request,
                                TtsService& service);
HttpResponse HandleAsrWebSocket(const HttpRequest& request,
                                AsrService& service);
}  // namespace mbufo::server

#endif  // MBUFO_SERVER_AUDIO_WEBSOCKET_HPP_
