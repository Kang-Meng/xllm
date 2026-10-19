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

#include <charconv>
#include <cstddef>
#include <string>
#include <utility>

#include "core/common/types.h"

namespace xllm::api_service {

// Validate the JSON/binary boundary before parsing JSON or opening SSE.
inline std::pair<Status, size_t> http_json_body_length(
    const brpc::Controller& controller) {
  const size_t body_size = controller.request_attachment().size();
  const std::string* total =
      controller.http_request().GetHeader("Content-Length");
  const std::string* json =
      controller.http_request().GetHeader("Infer-Content-Length");
  const auto invalid = [](std::string message, std::string param) {
    return std::make_pair(
        Status::invalid_parameter(std::move(message), std::move(param)),
        size_t{0});
  };
  const auto parse = [](const std::string& text, size_t* value) {
    const std::from_chars_result result =
        std::from_chars(text.data(), text.data() + text.size(), *value);
    return result.ec == std::errc() && result.ptr == text.data() + text.size();
  };
  if (total == nullptr && json == nullptr) {
    return {Status(), body_size};
  }
  if (total == nullptr) {
    return invalid(
        "Content-Length is required when Infer-Content-Length is present.",
        "Content-Length");
  }
  size_t total_size = 0;
  if (!parse(*total, &total_size)) {
    return invalid("Content-Length must be a non-negative integer.",
                   "Content-Length");
  }
  if (total_size != body_size) {
    return invalid("Content-Length does not match the request body length.",
                   "Content-Length");
  }
  if (json == nullptr) {
    return {Status(), body_size};
  }
  size_t json_size = 0;
  if (!parse(*json, &json_size)) {
    return invalid("Infer-Content-Length must be a non-negative integer.",
                   "Infer-Content-Length");
  }
  if (json_size > total_size) {
    return invalid("Infer-Content-Length exceeds Content-Length.",
                   "Infer-Content-Length");
  }
  return {Status(), json_size};
}

}  // namespace xllm::api_service
