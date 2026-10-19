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

#include <nlohmann/json.hpp>
#include <string>
#include <utility>

#include "core/common/public_error.h"
#include "core/common/types.h"

namespace xllm::api_service {

using ::xllm::public_error_from_status;
using ::xllm::PublicHttpError;

inline nlohmann::json openai_error_payload(const PublicHttpError& error) {
  const nlohmann::json param = error.param.has_value()
                                   ? nlohmann::json(*error.param)
                                   : nlohmann::json(nullptr);
  return {{"error",
           {{"message", error.message},
            {"type", error.type},
            {"param", param},
            {"code", error.code}}}};
}

inline nlohmann::json anthropic_error_payload(const PublicHttpError& error) {
  // Anthropic has a different type vocabulary. This is protocol rendering,
  // not another mapping from internal Status; preserve the public message.
  std::string type = "api_error";
  if (error.status_code == 401) {
    type = "authentication_error";
  } else if (error.status_code == 403) {
    type = "permission_error";
  } else if (error.status_code == 400 || error.status_code == 404) {
    type = "invalid_request_error";
  } else if (error.status_code == 429) {
    type = "rate_limit_error";
  } else if (error.status_code == 503) {
    type = "overloaded_error";
  }
  return {{"type", "error"},
          {"error", {{"type", type}, {"message", error.message}}}};
}

inline std::string openai_error_sse_event(const PublicHttpError& error) {
  return "data: " + openai_error_payload(error).dump() + "\n\ndata: [DONE]\n\n";
}

inline std::string anthropic_error_sse_event(const PublicHttpError& error) {
  return "event: error\ndata: " + anthropic_error_payload(error).dump() +
         "\n\n";
}

inline void write_http_error(brpc::Controller* controller,
                             const PublicHttpError& error,
                             bool anthropic = false) {
  controller->http_response().set_status_code(error.status_code);
  controller->http_response().set_content_type(
      "application/json; charset=utf-8");
  controller->http_response().SetHeader("X-Xllm-Error-Code", error.code);
  controller->http_response().SetHeader("X-Xllm-Retryable",
                                        error.retryable ? "true" : "false");
  if (error.status_code == 401) {
    controller->http_response().SetHeader("WWW-Authenticate", "Bearer");
  }
  if (error.retryable) {
    controller->http_response().SetHeader("Retry-After", "1");
  }
  controller->response_attachment().clear();
  controller->response_attachment().append(
      (anthropic ? anthropic_error_payload(error) : openai_error_payload(error))
          .dump());
}

inline void write_http_error(brpc::Controller* controller,
                             const Status& status,
                             bool anthropic = false) {
  write_http_error(controller, public_error_from_status(status), anthropic);
}

// Parsing can fail before a Call exists. Keep those HTTP responses identical
// to the errors produced by a Call, without changing typed RPC failures.
inline void write_bad_request(brpc::Controller* controller,
                              std::string message,
                              bool anthropic = false) {
  write_http_error(controller,
                   Status(StatusCode::INVALID_ARGUMENT, std::move(message)),
                   anthropic);
}

}  // namespace xllm::api_service
