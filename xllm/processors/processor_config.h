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

#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace xllm {

class ProcessorConfig final {
 public:
  explicit ProcessorConfig(nlohmann::json overrides = nlohmann::json::object(),
                           std::string name = "processor")
      : overrides_(std::move(overrides)), name_(std::move(name)) {
    CHECK(overrides_.is_object()) << name_ << " must be an object";
  }

  bool contains(const std::string& path) const {
    return overrides_.contains(pointer(path));
  }

  bool empty(const std::string& path = "") const {
    if (path.empty()) {
      return overrides_.empty();
    }
    return !contains(path) || overrides_.at(pointer(path)).empty();
  }

  template <typename Value>
  Value value(const std::string& path, Value default_value) const {
    if (!contains(path)) {
      return default_value;
    }
    consumed_.insert(pointer(path).to_string());
    return decode<Value>(overrides_.at(pointer(path)), name_ + "." + path);
  }

  void validate(const std::string& path = "") const {
    if (empty(path)) {
      return;
    }
    const auto& fields =
        path.empty() ? overrides_ : overrides_.at(pointer(path));
    const auto parent =
        path.empty() ? nlohmann::json::json_pointer() : pointer(path);
    for (const auto& [key, value] : fields.items()) {
      validate_field(value, parent / key);
    }
    LOG(INFO) << name_ << (path.empty() ? "" : "." + path)
              << " overrides: " << fields.dump();
  }

 private:
  static nlohmann::json::json_pointer pointer(std::string path) {
    std::replace(path.begin(), path.end(), '.', '/');
    return nlohmann::json::json_pointer("/" + path);
  }

  template <typename Value>
  static Value decode(const nlohmann::json& value, const std::string& path) {
    if constexpr (std::is_same_v<Value, bool>) {
      CHECK(value.is_boolean()) << path << " must be a boolean";
    } else if constexpr (std::is_integral_v<Value>) {
      CHECK(value.is_number_integer()) << path << " must be an integer";
      const long double number = value.get<long double>();
      CHECK(number >= std::numeric_limits<Value>::lowest() &&
            number <= std::numeric_limits<Value>::max())
          << path << " integer is out of range";
    } else if constexpr (std::is_floating_point_v<Value>) {
      CHECK(value.is_number()) << path << " must be a number";
    } else if constexpr (std::is_same_v<Value, std::string>) {
      CHECK(value.is_string()) << path << " must be a string";
    } else {
      CHECK(value.is_array()) << path << " must be an array";
      Value result;
      result.reserve(value.size());
      for (const auto& element : value) {
        result.emplace_back(decode<typename Value::value_type>(element, path));
      }
      return result;
    }
    return value.get<Value>();
  }

  void validate_field(const nlohmann::json& value,
                      const nlohmann::json::json_pointer& path) const {
    if (consumed_.contains(path.to_string())) {
      return;
    }
    std::string field_path = path.to_string();
    std::replace(field_path.begin(), field_path.end(), '/', '.');
    CHECK(value.is_object() && !value.empty())
        << "Unsupported processor parameter: " << name_ << field_path;
    for (const auto& [key, child] : value.items()) {
      validate_field(child, path / key);
    }
  }

  nlohmann::json overrides_;
  std::string name_;
  mutable std::unordered_set<std::string> consumed_;
};

}  // namespace xllm
