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

#include "models/model_registry.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/framework/request/stopping_checker.h"
#include "core/util/json_reader.h"

namespace xllm {
namespace {

TEST(ModelRegistryTest, ImageAlphaPolicyDefaultsToFalse) {
  EXPECT_FALSE(ModelMeta{}.preserve_image_alpha);
  EXPECT_FALSE(ModelRegistry::get_preserve_image_alpha("unknown_alpha_policy"));
  EXPECT_FALSE(
      ModelRegistry::get_preserve_image_alpha("QwenImageEditPlusPipeline"));
}

TEST(ModelRegistryTest, QwenImage21RegistersImageAlphaPolicy) {
  ASSERT_TRUE(ModelRegistry::has_dit_model_factory("QwenImage21Pipeline"));
  EXPECT_TRUE(ModelRegistry::get_preserve_image_alpha("QwenImage21Pipeline"));
}

TEST(ModelRegistryTest, DeepseekV4DSparkUsesTorchBackend) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("deepseek_v4_dspark",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v4_dspark");
  EXPECT_TRUE(error_message.empty());

  EXPECT_TRUE(resolve_model_registration("deepseek_v4_dspark",
                                         "TORCH",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v4_dspark");

  EXPECT_FALSE(resolve_model_registration("deepseek_v4_dspark",
                                          "ATB",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message,
            "Model type deepseek_v4_dspark only supports "
            "--npu_kernel_backend=TORCH.");
}

// The Python-path V4.1 DSpark draft is registered under its own model_type (see
// xllm/python/registry.py) and must resolve as a TORCH-only model type like its
// C++ counterpart, so a stray ATB request fails with a clear message instead of
// silently selecting the ATB kernels.
TEST(ModelRegistryTest, DeepseekV41DSparkUsesTorchBackend) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("deepseek_v41_dspark",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v41_dspark");
  EXPECT_TRUE(error_message.empty());

  EXPECT_TRUE(resolve_model_registration("deepseek_v41_dspark",
                                         "TORCH",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v41_dspark");

  EXPECT_FALSE(resolve_model_registration("deepseek_v41_dspark",
                                          "ATB",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message,
            "Model type deepseek_v41_dspark only supports "
            "--npu_kernel_backend=TORCH.");
}

TEST(ModelRegistryTest, DeepseekV41UsesTorchBackend) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("deepseek_v41",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v41");
  EXPECT_TRUE(error_message.empty());

  EXPECT_TRUE(resolve_model_registration("deepseek_v41",
                                         "TORCH",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v41");

  EXPECT_FALSE(resolve_model_registration("deepseek_v41",
                                          "ATB",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message,
            "Model type deepseek_v41 only supports "
            "--npu_kernel_backend=TORCH.");
}

TEST(ModelRegistryTest, DeepseekV41RegistersArgsLoaderAndLlmBackend) {
  EXPECT_TRUE(
      static_cast<bool>(ModelRegistry::get_model_args_loader("deepseek_v41")));
  EXPECT_EQ(ModelRegistry::get_model_backend("deepseek_v41"), "llm");
}

TEST(ModelRegistryTest, KimiK3StopsOnEndMarkersButNotOrdinaryTokenZero) {
  ModelArgsLoader loader = ModelRegistry::get_model_args_loader("kimi_k3");
  ASSERT_TRUE(static_cast<bool>(loader));
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"({"model_type":"kimi_k3"})"));
  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));

  EXPECT_EQ(args.stop_token_ids().count(0), 0u);
  StoppingChecker checker(/*max_generated_tokens=*/32,
                          /*max_context_len=*/128,
                          args.eos_token_id(),
                          /*ignore_eos=*/false,
                          args.stop_token_ids(),
                          /*stop_sequences=*/{});
  const std::vector<int32_t> punctuation{163584, 0};
  EXPECT_EQ(checker.check(punctuation, /*num_prompt_tokens=*/1),
            FinishReason::NONE);
  const std::vector<int32_t> continued{163584, 0, 100};
  EXPECT_EQ(checker.check(continued, /*num_prompt_tokens=*/1),
            FinishReason::NONE);
  for (const int32_t end_marker : {163585, 163586}) {
    const std::vector<int32_t> completed{163584, 0, 100, end_marker};
    EXPECT_EQ(checker.check(completed, /*num_prompt_tokens=*/1),
              FinishReason::STOP);
  }
}

}  // namespace
}  // namespace xllm
