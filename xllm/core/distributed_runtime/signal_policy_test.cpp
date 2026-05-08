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

#include "distributed_runtime/signal_policy.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(SignalPolicyTest, InstallsWorkerExitHandlerWithoutServiceRouting) {
  runtime::Options options;
  options.enable_service_routing(false);

  EXPECT_TRUE(should_install_worker_exit_signal_handler(options));
}

TEST(SignalPolicyTest, SkipsWorkerExitHandlerWithServiceRouting) {
  runtime::Options options;
  options.enable_service_routing(true);

  EXPECT_FALSE(should_install_worker_exit_signal_handler(options));
}

}  // namespace
}  // namespace xllm
