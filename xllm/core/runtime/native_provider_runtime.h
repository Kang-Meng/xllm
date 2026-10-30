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

#include <optional>
#include <string>

#include "core/runtime/native_provider_contract.h"

namespace xllm {

class BlockManagerPool;
class Engine;
struct InstanceInfo;

// Collects immutable Native V2 inputs from the loaded runtime. A null result
// with an empty error means the registration uses a legacy-compatible mode not
// opened by V2; a nonempty error means an eligible V2 runtime is malformed.
std::optional<NativeProviderDescriptorConfig>
collect_native_provider_descriptor_config(
    const InstanceInfo& instance_info,
    const Engine& engine,
    const BlockManagerPool& block_manager_pool,
    std::string* error);

}  // namespace xllm
