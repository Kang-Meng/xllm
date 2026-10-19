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

#include <cstddef>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

#include "core/common/types.h"

namespace xllm {

// Keep this contract in sync with Service's
// chat_template/tool_call_arguments.h. API arguments remain strings. Templates
// receive objects with recursively lexicographically sorted keys
// (nlohmann::json), never a silent {} fallback. Duplicate keys use nlohmann's
// last-value-wins policy; arrays retain order.
inline nlohmann::json parse_tool_call_arguments(const std::string& arguments) {
  return nlohmann::json::parse(arguments, nullptr, false);
}

inline std::string tool_call_arguments_param(size_t message_index,
                                             size_t tool_index) {
  return "messages[" + std::to_string(message_index) + "].tool_calls[" +
         std::to_string(tool_index) + "].function.arguments";
}

inline Status invalid_tool_call_arguments(std::string param) {
  return Status::invalid_parameter(
      "Tool-call arguments must be a JSON string encoding an object.",
      std::move(param));
}

template <typename Messages>
Status validate_tool_call_arguments(const Messages& messages) {
  for (size_t message_index = 0; message_index < messages.size();
       ++message_index) {
    const auto& calls = messages[message_index].tool_calls;
    if (!calls.has_value()) {
      continue;
    }
    for (size_t tool_index = 0; tool_index < calls->size(); ++tool_index) {
      if (!parse_tool_call_arguments((*calls)[tool_index].function.arguments)
               .is_object()) {
        return invalid_tool_call_arguments(
            tool_call_arguments_param(message_index, tool_index));
      }
    }
  }
  return Status();
}

// Share traversal between wire validation and the mutable template copy.
template <typename Messages, typename Visitor>
Status visit_tool_call_arguments(Messages& messages, Visitor visitor) {
  for (size_t message_index = 0; message_index < messages.size();
       ++message_index) {
    auto& message = messages[message_index];
    if (!message.is_object() || !message.contains("tool_calls") ||
        !message["tool_calls"].is_array()) {
      continue;
    }
    auto& calls = message["tool_calls"];
    for (size_t tool_index = 0; tool_index < calls.size(); ++tool_index) {
      auto& call = calls[tool_index];
      if (!call.is_object() || !call.contains("function") ||
          !call["function"].is_object() ||
          !call["function"].contains("arguments") ||
          !visitor(call["function"]["arguments"])) {
        return invalid_tool_call_arguments(
            tool_call_arguments_param(message_index, tool_index));
      }
    }
  }
  return Status();
}

// Validate the wire value without replacing its string representation.
inline Status validate_wire_tool_call_arguments(const nlohmann::json& json) {
  if (!json.contains("messages") || !json["messages"].is_array()) {
    return Status();
  }
  return visit_tool_call_arguments(json["messages"], [](const auto& arguments) {
    return arguments.is_string() &&
           parse_tool_call_arguments(
               arguments.template get_ref<const std::string&>())
               .is_object();
  });
}

// Only the render copy accepts objects and canonicalizes their key order.
inline bool normalize_template_tool_call_arguments(
    nlohmann::ordered_json* messages) {
  return visit_tool_call_arguments(
             *messages,
             [](auto& arguments) {
               nlohmann::json parsed =
                   arguments.is_string()
                       ? parse_tool_call_arguments(
                             arguments.template get_ref<const std::string&>())
                       : nlohmann::json(arguments);
               if (!parsed.is_object()) {
                 return false;
               }
               arguments = std::move(parsed);
               return true;
             })
      .ok();
}
}  // namespace xllm
