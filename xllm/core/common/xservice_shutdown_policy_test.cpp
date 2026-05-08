/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "xservice_shutdown_policy.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(XServiceShutdownPolicyTest, DisabledWithoutServiceRoutingOrDisaggPd) {
  EXPECT_FALSE(should_enable_xservice_shutdown(false, false));
}

TEST(XServiceShutdownPolicyTest, EnabledWithServiceRouting) {
  EXPECT_TRUE(should_enable_xservice_shutdown(true, false));
}

TEST(XServiceShutdownPolicyTest, EnabledWithDisaggPdOnly) {
  EXPECT_TRUE(should_enable_xservice_shutdown(false, true));
}

}  // namespace
}  // namespace xllm
