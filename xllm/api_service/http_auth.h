/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <brpc/controller.h>

#include <boost/algorithm/string.hpp>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "api_service/public_error.h"
#include "api_service/request_id.h"

namespace xllm::api_service {

// Only public HTTP endpoints use the Bearer key. Internal typed RPCs retain
// their existing transport and must be secured by the deployment boundary.
class HttpBearerAuthenticator final {
 public:
  explicit HttpBearerAuthenticator(std::string api_key)
      : api_key_(std::move(api_key)) {}

  // Initialize the response ID and authenticate before parsing or opening SSE.
  bool prepare_request(brpc::Controller* controller,
                       bool anthropic = false) const {
    ensure_http_x_request_id(controller);
    if (api_key_.empty()) {
      return true;
    }

    const std::string* authorization =
        controller->http_request().GetHeader("Authorization");
    if (authorization != nullptr &&
        boost::istarts_with(*authorization, "Bearer ") &&
        key_matches(std::string_view(*authorization).substr(7))) {
      return true;
    }

    write_http_error(controller,
                     Status(StatusCode::UNKNOWN,
                            "Invalid API key provided.",
                            FailureReason::UNAUTHENTICATED),
                     anthropic);
    return false;
  }

 private:
  bool key_matches(std::string_view presented) const {
    if (presented.size() != api_key_.size()) {
      return false;
    }
    uint8_t mismatch = 0;
    for (size_t i = 0; i < presented.size(); ++i) {
      mismatch |= static_cast<uint8_t>(presented[i] ^ api_key_[i]);
    }
    return mismatch == 0;
  }

  const std::string api_key_;
};

}  // namespace xllm::api_service
