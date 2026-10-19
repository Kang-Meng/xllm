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

#include <cstdint>
#include <optional>
#include <string>

namespace xllm {

class Status;

// Protocol-neutral public error shared by HTTP and generation callbacks.
struct PublicHttpError {
  int32_t status_code = 500;
  std::string code = "internal_error";
  std::string type = "server_error";
  std::string message =
      "The server encountered an error processing your request.";
  bool retryable = false;
  std::optional<std::string> param;
};

PublicHttpError public_error_from_status(const Status& status);

}  // namespace xllm
