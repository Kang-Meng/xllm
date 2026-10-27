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

#include <cstdint>

namespace xllm::npu {

// Prepares an embedded interpreter and the torch_npu runtime for a native NPU
// process that xLLM starts itself, so that the process can reach the
// accelerator at all: torch_npu resolves part of its initialization through
// Python (PyGILState_Ensure inside its pybind layer) and cannot run without a
// live interpreter, which is created here on the main thread.
//
// Only the processes xLLM owns call this, namely the serving binary and the
// spawned offline workers. Offline inference driven from Python already has an
// interpreter, so it brings the accelerator up from Python instead
// (xllm/pybind/utils.py). Idempotent per process.
void initialize_python_runtime(int32_t device_index);

}  // namespace xllm::npu
