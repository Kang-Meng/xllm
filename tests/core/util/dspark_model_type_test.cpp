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

#include <gtest/gtest.h>

#include "core/util/utils.h"

namespace xllm::util {

// The native executor resolves the C++ TORCH model under "deepseek_v4_dspark";
// the Python path registers the V4.1 draft under "deepseek_v41_dspark"
// (xllm/python/registry.py) and PyCausalLM looks it up by that exact string.
// Every downstream draft decision (own head/embedding, DSpark SAS mode, KV
// cache layout) keys on these predicates, so both names must classify
// identically.
TEST(DsparkModelTypeTest, BothDraftNamesAreDeepseekV4Dspark) {
  EXPECT_TRUE(is_deepseek_v4_dspark_model_type("deepseek_v4_dspark"));
  EXPECT_TRUE(is_deepseek_v4_dspark_model_type("deepseek_v41_dspark"));
}

TEST(DsparkModelTypeTest, BothDraftNamesStayInTheDeepseekV4Family) {
  EXPECT_TRUE(is_deepseek_v4_model_type("deepseek_v4_dspark"));
  EXPECT_TRUE(is_deepseek_v4_model_type("deepseek_v41_dspark"));
  // The V4.1 and V4 targets keep the same classification.
  EXPECT_TRUE(is_deepseek_v4_model_type("deepseek_v41"));
  EXPECT_TRUE(is_deepseek_v4_model_type("deepseek_v4"));
}

TEST(DsparkModelTypeTest, TargetsAreNotDraftNames) {
  EXPECT_FALSE(is_deepseek_v4_dspark_model_type("deepseek_v41"));
  EXPECT_FALSE(is_deepseek_v4_dspark_model_type("deepseek_v4"));
  EXPECT_FALSE(is_deepseek_v4_dspark_model_type(""));
}

// Locks the constants that the worker draft routing and the Python registry
// must agree on.
TEST(DsparkModelTypeTest, ConstantsMatchTheDraftNames) {
  EXPECT_EQ(kDeepseekV4DSparkModelType, "deepseek_v4_dspark");
  EXPECT_EQ(kDeepseekV41DSparkModelType, "deepseek_v41_dspark");
}

}  // namespace xllm::util
