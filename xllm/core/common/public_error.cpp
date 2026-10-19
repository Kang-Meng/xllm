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

#include "core/common/public_error.h"

#include <utility>

#include "core/common/types.h"

namespace xllm {

PublicHttpError public_error_from_status(const Status& status) {
  const auto make_error = [&status](int32_t http_status,
                                    const char* code,
                                    const char* type,
                                    std::string message,
                                    bool retryable = false) {
    return PublicHttpError{
        http_status, code, type, std::move(message), retryable, status.param()};
  };
  switch (status.reason()) {
    case FailureReason::UNAUTHENTICATED:
      return make_error(401,
                        "invalid_api_key",
                        "authentication_error",
                        "Invalid API key provided.");
    case FailureReason::PERMISSION_DENIED:
      return make_error(403,
                        "permission_denied",
                        "permission_error",
                        "You do not have permission to access this resource.");
    case FailureReason::INVALID_PARAMETER:
      return make_error(
          400, "invalid_parameter", "invalid_request_error", status.message());
    case FailureReason::MODEL_NOT_FOUND:
      return make_error(404,
                        "model_not_found",
                        "invalid_request_error",
                        "The requested model does not exist.");
    case FailureReason::PROMPT_TOO_LONG:
      return make_error(
          400,
          "context_length_exceeded",
          "invalid_request_error",
          "The prompt exceeds the maximum length supported by the "
          "model or serving configuration.");
    case FailureReason::TEMPORARY_OVERLOAD:
      return make_error(503,
                        "server_is_overloaded",
                        "service_unavailable_error",
                        "The requested model is temporarily overloaded.",
                        /*retryable=*/true);
    case FailureReason::CONCURRENCY_LIMIT_REACHED:
      return make_error(
          429,
          "rate_limit_exceeded",
          "rate_limit_error",
          "The maximum number of concurrent requests has been reached.",
          /*retryable=*/true);
    case FailureReason::UNSPECIFIED:
      break;
  }
  switch (status.code()) {
    case StatusCode::INVALID_ARGUMENT:
      return make_error(
          400, "invalid_request", "invalid_request_error", status.message());
    case StatusCode::DEADLINE_EXCEEDED:
      return make_error(
          504, "request_timeout", "server_error", "The request timed out.");
    case StatusCode::UNAVAILABLE:
      return make_error(503,
                        "service_unavailable",
                        "service_unavailable_error",
                        "The requested model is temporarily unavailable.");
    // Resource exhaustion/cancellation alone does not identify a client error.
    // Classified failures use reason() above; never expose internal messages.
    case StatusCode::RESOURCE_EXHAUSTED:
    case StatusCode::CANCELLED:
    case StatusCode::UNKNOWN:
    case StatusCode::OK:
      break;
  }
  PublicHttpError error;
  error.param = status.param();
  return error;
}

}  // namespace xllm
