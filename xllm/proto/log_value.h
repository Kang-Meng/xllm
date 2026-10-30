/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>

namespace xllm::observability {

inline constexpr size_t kMaxLogValueBytes = 256;

inline std::string escape_log_value(
    const std::string& value,
    size_t max_input_bytes = kMaxLogValueBytes) {
  constexpr char kHex[] = "0123456789ABCDEF";
  const size_t limit = std::min(value.size(), max_input_bytes);
  std::string escaped;
  escaped.reserve(limit);
  for (size_t index = 0; index < limit; ++index) {
    const unsigned char character = static_cast<unsigned char>(value[index]);
    const bool unreserved = (character >= 'a' && character <= 'z') ||
                            (character >= 'A' && character <= 'Z') ||
                            (character >= '0' && character <= '9') ||
                            character == '-' || character == '_' ||
                            character == '.' || character == ':';
    if (unreserved) {
      escaped.push_back(static_cast<char>(character));
      continue;
    }
    escaped.push_back('%');
    escaped.push_back(kHex[character >> 4]);
    escaped.push_back(kHex[character & 0x0f]);
  }
  if (value.size() > limit) {
    escaped.append("...");
  }
  return escaped;
}

}  // namespace xllm::observability
