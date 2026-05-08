/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "runtime/options.h"

namespace xllm {

inline bool should_install_worker_exit_signal_handler(
    const runtime::Options& options) {
  // In service-routing mode, the master process owns XServiceClient and needs
  // to keep the graceful shutdown path so it can actively close the failover
  // session before the process exits.
  return !options.enable_service_routing();
}

}  // namespace xllm
