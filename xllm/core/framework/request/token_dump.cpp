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

#include "core/framework/request/token_dump.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <system_error>

namespace xllm {

void dump_request_tokens(const std::string& role,
                         const std::string& request_id,
                         size_t sequence_index,
                         Slice<int32_t> tokens,
                         size_t num_prompt_tokens) {
  const char* dump_dir = std::getenv("XLLM_TOKEN_DUMP_DIR");
  if (dump_dir == nullptr || dump_dir[0] == '\0') {
    return;
  }
  CHECK_LE(num_prompt_tokens, tokens.size());

  std::error_code error;
  std::filesystem::create_directories(dump_dir, error);
  if (error) {
    LOG(ERROR) << "[TokenDump] Cannot create directory " << dump_dir << ": "
               << error.message();
    return;
  }

  std::string safe_request_id = request_id.substr(0, 120);
  for (char& character : safe_request_id) {
    const bool is_safe = (character >= 'a' && character <= 'z') ||
                         (character >= 'A' && character <= 'Z') ||
                         (character >= '0' && character <= '9') ||
                         character == '-' || character == '_';
    if (!is_safe) {
      character = '_';
    }
  }
  const absl::Time now = absl::Now();
  const std::string timestamp =
      absl::FormatTime("%Y%m%d_%H%M%S", now, absl::LocalTimeZone()) + "_" +
      std::to_string(absl::ToUnixNanos(now));
  static std::atomic<uint64_t> dump_index{0};
  const std::string filename =
      role + "_" + timestamp + "_" + safe_request_id + "_seq" +
      std::to_string(sequence_index) + "_pid" + std::to_string(getpid()) + "_" +
      std::to_string(dump_index.fetch_add(1, std::memory_order_relaxed)) +
      ".json";
  const std::filesystem::path path = std::filesystem::path(dump_dir) / filename;
  const std::filesystem::path temporary_path = path.string() + ".tmp";

  nlohmann::ordered_json data;
  data["role"] = role;
  data["event"] = role == "prefill" ? "request_arrival" : "decode_finish";
  data["timestamp"] = timestamp;
  data["request_id"] = request_id;
  data["sequence_index"] = sequence_index;
  data["prompt_count"] = num_prompt_tokens;
  data["generated_count"] = tokens.size() - num_prompt_tokens;
  data["count"] = tokens.size();
  data["token_ids"] = nlohmann::ordered_json::array();
  for (const int32_t token : tokens) {
    data["token_ids"].push_back(token);
  }
  std::ofstream output(temporary_path, std::ios::out | std::ios::trunc);
  output << data.dump() << '\n';
  output.close();
  if (!output) {
    LOG(ERROR) << "[TokenDump] Failed to write " << temporary_path;
    std::filesystem::remove(temporary_path, error);
    return;
  }
  // Publish only complete JSON files so readers cannot see partial token lists.
  std::filesystem::rename(temporary_path, path, error);
  if (error) {
    LOG(ERROR) << "[TokenDump] Failed to publish " << path << ": "
               << error.message();
    return;
  }
  LOG(INFO) << "[TokenDump] role=" << role << ", request_id=" << request_id
            << ", sequence=" << sequence_index << ", count=" << tokens.size()
            << ", prompt_count=" << num_prompt_tokens
            << ", generated_count=" << tokens.size() - num_prompt_tokens
            << ", file=" << path.string();
}

}  // namespace xllm
