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

#include <butil/iobuf.h>
#include <google/protobuf/message.h>
#include <json2pb/pb_to_json.h>

#include <exception>
#include <nlohmann/json.hpp>
#include <string>

#include "anthropic.pb.h"
#include "core/common/types.h"

namespace xllm {
namespace api_service {

inline Status validate_anthropic_request(
    const proto::AnthropicMessagesRequest& request) {
  if (request.max_tokens() < 0) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Anthropic max_tokens must be non-negative.");
  }
  if (request.messages_size() == 0) {
    return Status(StatusCode::INVALID_ARGUMENT, "Messages is empty!");
  }
  return Status();
}

namespace detail {

inline bool has_only_key(const nlohmann::json& value, const char* key) {
  return value.is_object() && value.size() == 1 && value.contains(key);
}

inline nlohmann::json normalize_json(const nlohmann::json& value);

inline nlohmann::json normalize_struct(const nlohmann::json& value) {
  nlohmann::json result = nlohmann::json::object();
  if (!value.is_object() || !value.contains("fields") ||
      !value["fields"].is_object()) {
    return result;
  }

  for (auto it = value["fields"].begin(); it != value["fields"].end(); ++it) {
    result[it.key()] = normalize_json(it.value());
  }
  return result;
}

inline nlohmann::json normalize_list(const nlohmann::json& value) {
  nlohmann::json result = nlohmann::json::array();
  if (!value.is_object() || !value.contains("values") ||
      !value["values"].is_array()) {
    return result;
  }

  for (const nlohmann::json& item : value["values"]) {
    result.push_back(normalize_json(item));
  }
  return result;
}

inline bool is_value_wrapper(const nlohmann::json& value) {
  return has_only_key(value, "null_value") ||
         has_only_key(value, "number_value") ||
         has_only_key(value, "string_value") ||
         has_only_key(value, "bool_value") ||
         has_only_key(value, "struct_value") ||
         has_only_key(value, "list_value");
}

inline nlohmann::json normalize_value(const nlohmann::json& value) {
  if (has_only_key(value, "null_value")) {
    return nullptr;
  }
  if (has_only_key(value, "number_value")) {
    return value["number_value"];
  }
  if (has_only_key(value, "string_value")) {
    return value["string_value"];
  }
  if (has_only_key(value, "bool_value")) {
    return value["bool_value"];
  }
  if (has_only_key(value, "struct_value")) {
    return normalize_struct(value["struct_value"]);
  }
  if (has_only_key(value, "list_value")) {
    return normalize_list(value["list_value"]);
  }
  return value;
}

inline nlohmann::json normalize_json(const nlohmann::json& value) {
  if (has_only_key(value, "fields")) {
    return normalize_struct(value);
  }
  if (is_value_wrapper(value)) {
    return normalize_value(value);
  }
  if (value.is_array()) {
    nlohmann::json result = nlohmann::json::array();
    for (const nlohmann::json& item : value) {
      result.push_back(normalize_json(item));
    }
    return result;
  }
  if (value.is_object()) {
    nlohmann::json result = nlohmann::json::object();
    for (auto it = value.begin(); it != value.end(); ++it) {
      result[it.key()] = normalize_json(it.value());
    }
    return result;
  }
  return value;
}

}  // namespace detail

inline bool proto_to_anthropic_json(const google::protobuf::Message& message,
                                    const json2pb::Pb2JsonOptions& options,
                                    std::string* json,
                                    std::string* err_msg) {
  butil::IOBuf raw_buf;
  butil::IOBufAsZeroCopyOutputStream json_output(&raw_buf);
  if (!json2pb::ProtoMessageToJson(message, &json_output, options, err_msg)) {
    return false;
  }

  try {
    nlohmann::json parsed = nlohmann::json::parse(raw_buf.to_string());
    parsed = detail::normalize_json(parsed);
    const auto* event =
        dynamic_cast<const proto::AnthropicStreamEvent*>(&message);
    if (event != nullptr) {
      if (event->type() == "message_start" && event->has_message()) {
        nlohmann::json& start = parsed["message"];
        if (!event->message().has_stop_reason()) {
          start["stop_reason"] = nullptr;
        }
        if (!event->message().has_stop_sequence()) {
          start["stop_sequence"] = nullptr;
        }
        start["usage"]["input_tokens"] =
            event->message().usage().input_tokens();
        start["usage"]["output_tokens"] =
            event->message().usage().output_tokens();
      } else if (event->type() == "message_delta" && event->has_delta()) {
        if (!event->delta().has_stop_sequence()) {
          parsed["delta"]["stop_sequence"] = nullptr;
        }
      }
    }
    *json = parsed.dump();
  } catch (const std::exception& e) {
    *err_msg = e.what();
    return false;
  }
  return true;
}

}  // namespace api_service
}  // namespace xllm
