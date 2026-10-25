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

#pragma once

#include <brpc/server.h>
#include <google/protobuf/service.h>

#include <string>

#include "core/common/metrics.h"
#include "core/common/types.h"

namespace xllm {

// Called exactly once by the owner of an inference API entry: the entry guard
// before handoff, or Call at the end of its lifetime (not at SSE first write).
// A released SSE controller may be nullptr; its Call must supply x_request_id.
void record_api_request_terminal(brpc::Controller* controller,
                                 bool response_complete,
                                 const Status* error_status = nullptr,
                                 bool stream_disconnected = false,
                                 const std::string& x_request_id = "",
                                 bool controller_released = false);

inline void request_in_metric(void* /*context*/) {
  COUNTER_INC(server_request_in_total);
}

inline void request_out_metric(void* context) {
  auto* ctrl = static_cast<brpc::Controller*>(context);
  if (ctrl == nullptr) {
    LOG(ERROR) << "ctrl is nullptr";
    return;
  }

  const bool request_ok = !ctrl->Failed() && !ctrl->IsCanceled() &&
                          ctrl->http_response().status_code() < 400;
  if (request_ok) {
    COUNTER_INC(server_request_total_ok);
  } else {
    COUNTER_INC(server_request_total_fail);
    if (ctrl->ErrorCode() == brpc::ELIMIT ||
        ctrl->http_response().status_code() == 429) {
      COUNTER_INC(server_request_total_limit);
    }
  }
  record_api_request_terminal(ctrl, request_ok);
}

}  // namespace xllm
