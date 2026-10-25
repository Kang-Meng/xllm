/* Copyright 2026 The xLLM Authors.

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

#include "api_service/rpc_request_metrics.h"

#include <string>

#include "api_service/request_id.h"
#include "core/util/verbose_trace_logger.h"

namespace xllm {
namespace {

const char* outcome_from_status(const Status& status) {
  switch (status.reason()) {
    case FailureReason::UNAUTHENTICATED:
    case FailureReason::PERMISSION_DENIED:
      return "authentication";
    case FailureReason::MODEL_NOT_FOUND:
      return "model_not_found";
    case FailureReason::CONCURRENCY_LIMIT_REACHED:
      return "rate_limited";
    case FailureReason::TEMPORARY_OVERLOAD:
      return "unavailable";
    case FailureReason::INVALID_PARAMETER:
    case FailureReason::PROMPT_TOO_LONG:
      return "invalid_request";
    case FailureReason::UNSPECIFIED:
      break;
    default:
      // An unknown reason must not inherit a status-code classification.
      return "internal_error";
  }
  // Bare RESOURCE_EXHAUSTED has no proven overload reason and maps to HTTP 500.
  switch (status.code()) {
    case StatusCode::CANCELLED:
      return "cancelled";
    case StatusCode::DEADLINE_EXCEEDED:
      return "timeout";
    case StatusCode::INVALID_ARGUMENT:
      return "invalid_request";
    case StatusCode::UNAVAILABLE:
      return "unavailable";
    case StatusCode::OK:
    case StatusCode::RESOURCE_EXHAUSTED:
    case StatusCode::UNKNOWN:
      return "internal_error";
  }
  return "internal_error";
}

const char* outcome_from_response(const brpc::Controller& controller) {
  if (controller.ErrorCode() == brpc::ELIMIT) {
    return "rate_limited";
  }
  // Keep these published codes in sync with public_error_from_status;
  // unknown codes still fall back to the HTTP status below.
  const std::string* code =
      controller.http_response().GetHeader("X-Xllm-Error-Code");
  if (code != nullptr) {
    if (*code == "invalid_api_key" || *code == "permission_denied") {
      return "authentication";
    }
    if (*code == "model_not_found") {
      return "model_not_found";
    }
    if (*code == "rate_limit_exceeded") {
      return "rate_limited";
    }
    if (*code == "request_timeout") {
      return "timeout";
    }
    if (*code == "invalid_request" || *code == "invalid_parameter" ||
        *code == "context_length_exceeded") {
      return "invalid_request";
    }
    if (*code == "server_is_overloaded" || *code == "service_unavailable") {
      return "unavailable";
    }
  }
  // Unknown public codes retain the HTTP status as their coarse outcome.
  switch (controller.http_response().status_code()) {
    case 400:
    case 404:
      return "invalid_request";
    case 401:
    case 403:
      return "authentication";
    case 429:
      return "rate_limited";
    case 503:
      return "unavailable";
    case 504:
      return "timeout";
    default:
      return "internal_error";
  }
}

}  // namespace

void record_api_request_terminal(brpc::Controller* controller,
                                 bool response_complete,
                                 const Status* error_status,
                                 bool stream_disconnected,
                                 const std::string& x_request_id,
                                 bool controller_released) {
  // Call resolves the ID before releasing its controller; this is defensive.
  if (controller_released ? x_request_id.empty() : controller == nullptr) {
    return;
  }

  // A disconnected stream or explicit error wins over teardown state; after
  // controller release only the completion state remains available.
  const char* outcome = "success";
  if (stream_disconnected) {
    outcome = "cancelled";
  } else if (!response_complete && error_status != nullptr &&
             !error_status->ok()) {
    outcome = outcome_from_status(*error_status);
  } else if (controller_released) {
    outcome = response_complete ? "success" : "internal_error";
  } else if (controller->http_response().status_code() >= 400) {
    // A committed HTTP error wins over cancellation observed during teardown.
    outcome = outcome_from_response(*controller);
  } else if (controller->IsCanceled()) {
    outcome = "cancelled";
  } else if (controller->Failed()) {
    outcome = outcome_from_response(*controller);
  } else if (!response_complete) {
    outcome = "internal_error";
  }

  // Early HTTP errors may have no ID yet; use the same response ID that a
  // later Call would have used. No model, path, or user input becomes a label.
  const std::string id =
      (controller_released || !x_request_id.empty())
          ? x_request_id
          : api_service::ensure_http_x_request_id(controller);
  MULTI_COUNTER_ADD(api_request_terminal_total, outcome, 1);
  XLLM_VERBOSE_TRACE() << "x-request-id: " << id
                       << ", record_type: api_request_terminal, outcome: "
                       << outcome;
}

}  // namespace xllm
