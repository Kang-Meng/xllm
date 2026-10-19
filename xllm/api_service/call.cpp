/* Copyright 2025-2026 The xLLM Authors.

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

#include "api_service/call.h"

#include "api_service/http_request_body.h"
#include "api_service/request_id.h"
#include "core/common/constants.h"
#include "core/common/metrics.h"
#include "core/util/verbose_trace_logger.h"

namespace xllm {
void Call::record_error(const Status& status) {
  response_complete_ = false;
  if (status.reason() == FailureReason::CONCURRENCY_LIMIT_REACHED) {
    COUNTER_INC(server_request_total_limit);
  }
}

void Call::record_result(bool connected) const {
  if (response_complete_ && connected) {
    COUNTER_INC(server_request_total_ok);
  } else {
    COUNTER_INC(server_request_total_fail);
  }
}

Call::Call(brpc::Controller* controller,
           std::string body_x_request_id,
           bool is_http_request)
    : controller_(controller) {
  init(std::move(body_x_request_id), is_http_request);
}

void Call::init(std::string body_x_request_id, bool is_http_request) {
  is_http_request_ = is_http_request;
  if (controller_->http_request().GetHeader("x-request-time")) {
    x_request_time_ = *controller_->http_request().GetHeader("x-request-time");
  } else if (controller_->http_request().GetHeader("x-request-timems")) {
    x_request_time_ =
        *controller_->http_request().GetHeader("x-request-timems");
  }

  x_request_id_ =
      api_service::resolve_x_request_id(controller_, body_x_request_id);
  if (is_http_request) {
    controller_->http_response().SetHeader("x-request-id", x_request_id_);
  }

  XLLM_VERBOSE_TRACE() << "event=request_received x-request-id="
                       << x_request_id_
                       << " path=" << controller_->http_request().uri().path();

  init_request_payload();
}

void Call::init_request_payload() {
  request_payload_.clear();
  auto [status, json_size] = api_service::http_json_body_length(*controller_);
  request_payload_status_ = std::move(status);
  if (!request_payload_status_.ok()) {
    return;
  }
  const size_t payload_size =
      controller_->request_attachment().size() - json_size;
  const size_t appended = controller_->request_attachment().append_to(
      &request_payload_, payload_size, json_size);
  if (appended != payload_size) {
    request_payload_.clear();
    request_payload_status_ = Status::invalid_parameter(
        "Failed to read the binary request payload.", kInferContentLength);
  }
}

}  // namespace xllm
