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
#include <cstdint>
#include <string>

#include "core/util/slice.h"

namespace xllm {

// Enabled only when XLLM_TOKEN_DUMP_DIR names a diagnostic output directory.
void dump_request_tokens(const std::string& role,
                         const std::string& request_id,
                         size_t sequence_index,
                         Slice<int32_t> tokens,
                         size_t num_prompt_tokens);

}  // namespace xllm
