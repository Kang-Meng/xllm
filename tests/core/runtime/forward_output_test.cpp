/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "runtime/forward_params.h"

namespace xllm {
namespace {

TEST(ForwardInputTest, SamplingBufferPlanIncludesQueriedTokenIds) {
  ForwardInput source;
  source.sampling_params.logprob_token_ids =
      torch::tensor({{7, -1}, {11, 13}}, torch::kInt32);
  source.decoder_sampling_params.logprob_token_ids =
      torch::tensor({{17, -1}}, torch::kInt32);
  ForwardInput target;
  detail::ForwardInputBufferPlan plan;

  ASSERT_TRUE(detail::add_sampling_to_plan(
      source.sampling_params, target.sampling_params, plan));
  ASSERT_TRUE(detail::add_sampling_to_plan(
      source.decoder_sampling_params, target.decoder_sampling_params, plan));
  ASSERT_EQ(plan.entries.size(), 2u);
  EXPECT_EQ(plan.entries[0].target, &target.sampling_params.logprob_token_ids);
  EXPECT_EQ(plan.entries[1].target,
            &target.decoder_sampling_params.logprob_token_ids);
  EXPECT_TRUE(torch::equal(plan.entries[0].host_tensor,
                           source.sampling_params.logprob_token_ids));
  EXPECT_TRUE(torch::equal(plan.entries[1].host_tensor,
                           source.decoder_sampling_params.logprob_token_ids));
  EXPECT_GT(plan.prepare_layout(), 0u);
  EXPECT_LT(plan.entries[0].offset, plan.entries[1].offset);
}

TEST(ForwardInputTest, SamplingBufferPlanSkipsUndefinedQueriedTokenIds) {
  SamplingParameters source;
  SamplingParameters target;
  detail::ForwardInputBufferPlan plan;
  ASSERT_TRUE(detail::add_sampling_to_plan(source, target, plan));
  for (const auto& entry : plan.entries) {
    EXPECT_NE(entry.target, &target.logprob_token_ids);
  }
  EXPECT_FALSE(target.logprob_token_ids.defined());
}

TEST(ForwardOutputTest, NonDriverFailureIsReturnedWhenEplbIsDisabled) {
  std::optional<ForwardOutput> output = make_non_driver_forward_output(
      ForwardOutput{}, {"request-a"}, /*eplb_enabled=*/false);

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->failed_request_ids,
            (std::vector<std::string>{"request-a"}));
}

TEST(ForwardOutputTest, NonDriverSuccessRemainsEmptyWhenEplbIsDisabled) {
  std::optional<ForwardOutput> output = make_non_driver_forward_output(
      ForwardOutput{}, {}, /*eplb_enabled=*/false);

  EXPECT_FALSE(output.has_value());
}

TEST(ForwardOutputTest, NonDriverEplbOutputIsRetainedWithoutFailures) {
  std::optional<ForwardOutput> output = make_non_driver_forward_output(
      ForwardOutput{}, {}, /*eplb_enabled=*/true);

  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(output->failed_request_ids.empty());
}

}  // namespace
}  // namespace xllm
