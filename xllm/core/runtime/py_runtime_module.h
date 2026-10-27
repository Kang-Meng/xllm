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

namespace xllm {

// Register the runtime bridge in the current, initialized Python interpreter.
// Self-managing on the GIL: safe to call with or without the GIL held, from
// the PYBIND11_MODULE body (bind.cpp) and from plain worker threads
// (worker_impl.cpp) alike. Registration happens at most once per process;
// later calls are cheap no-ops.
void ensure_xllm_runtime_module();

}  // namespace xllm
