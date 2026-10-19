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

#include "api_service/completion_json_parser.h"

#include <glog/logging.h>

#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>

namespace xllm {

std::pair<Status, std::string> preprocess_completion_prompt(
    std::string json_str) {
  try {
    auto json = nlohmann::json::parse(json_str);
    const auto prompt_it = json.find("prompt");
    const bool has_prompt = prompt_it != json.end();
    if (has_prompt && !prompt_it->is_string() && !prompt_it->is_array() &&
        !prompt_it->is_null()) {
      return {
          Status::invalid_parameter(
              "prompt must be a string or an array of token ids.", "prompt"),
          ""};
    }
    if (has_prompt && prompt_it->is_string() &&
        prompt_it->get_ref<const std::string&>().empty() &&
        (!json.contains("token_ids") || json["token_ids"].empty())) {
      return {Status::invalid_parameter("Prompt is empty!", "prompt"), ""};
    }
    if (!has_prompt || !prompt_it->is_array()) {
      // string / scalar / missing prompt: leave bytes untouched.
      return {Status(), std::move(json_str)};
    }

    if (json.contains("token_ids")) {
      return {Status::invalid_parameter(
                  "specify prompt as a token array or token_ids, not both",
                  "prompt"),
              ""};
    }

    const auto& prompt = *prompt_it;
    if (prompt.empty()) {
      return {Status::invalid_parameter("prompt array is empty", "prompt"), ""};
    }

    const auto& first = prompt.front();
    if (first.is_string()) {
      return {
          Status::invalid_parameter(
              "batch prompts (array of strings) are not supported", "prompt"),
          ""};
    }
    if (first.is_array()) {
      return {Status::invalid_parameter(
                  "batch prompts (nested arrays) are not supported", "prompt"),
              ""};
    }

    for (const auto& token : prompt) {
      if (!token.is_number_integer()) {
        return {
            Status::invalid_parameter(
                "prompt array must contain only integer token ids", "prompt"),
            ""};
      }
      bool in_range = false;
      if (token.is_number_unsigned()) {
        in_range = token.get<uint64_t>() <=
                   static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
      } else {
        const int64_t value = token.get<int64_t>();
        in_range = value >= std::numeric_limits<int32_t>::min() &&
                   value <= std::numeric_limits<int32_t>::max();
      }
      if (!in_range) {
        return {
            Status::invalid_parameter("token id exceeds int32 range", "prompt"),
            ""};
      }
    }
    json["token_ids"] = std::move(json["prompt"]);
    json["prompt"] = "";
    return {Status(), json.dump()};
  } catch (const nlohmann::json::exception& e) {
    return {Status(StatusCode::INVALID_ARGUMENT,
                   "Invalid JSON format: " + std::string(e.what())),
            ""};
  } catch (const std::exception& e) {
    LOG(ERROR) << "Exception during completion JSON preprocessing: "
               << e.what();
    return {Status(StatusCode::UNKNOWN,
                   "Internal server error during JSON processing."),
            ""};
  }
}

}  // namespace xllm
