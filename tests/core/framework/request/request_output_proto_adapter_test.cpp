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

#include "core/framework/request/request_output_proto_adapter.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace xllm {
namespace {

TEST(RequestOutputProtoAdapterTest, PreservesCanonicalFirstEventExactly) {
  RequestOutput output;
  output.request_id = "engine-request-1";
  output.service_request_id = "01234567-89ab-7cde-bf01-23456789abcd";
  output.finished_on_prefill_instance = true;
  output.usage = Usage{.num_prompt_tokens = 8,
                       .num_generated_tokens = 1,
                       .num_total_tokens = 9,
                       .num_cached_tokens = 6};

  SequenceOutput sequence;
  sequence.index = 0;
  sequence.text = "精确首事件";
  sequence.token_ids = {101};
  sequence.finish_reason = "stop";
  LogProb logprob;
  logprob.token = "精";
  logprob.token_id = 101;
  logprob.logprob = -0.25f;
  logprob.finished_token = true;
  LogProbData top_logprob;
  top_logprob.token = "准";
  top_logprob.token_id = 102;
  top_logprob.logprob = -0.5f;
  top_logprob.finished_token = false;
  logprob.top_logprobs = std::vector<LogProbData>{top_logprob};
  sequence.logprobs = std::vector<LogProb>{logprob};
  output.outputs.emplace_back(std::move(sequence));

  proto::DisaggStreamGeneration generation;
  ASSERT_TRUE(request_output_to_proto(output, "prefill-1", &generation));

  EXPECT_EQ(generation.req_id(), output.request_id);
  EXPECT_EQ(generation.service_req_id(), output.service_request_id);
  EXPECT_EQ(generation.sender_engine_uid(), "prefill-1");
  EXPECT_TRUE(generation.finished_on_prefill_instance());
  ASSERT_EQ(generation.outputs_size(), 1);
  EXPECT_EQ(generation.outputs(0).text(), "精确首事件");
  EXPECT_EQ(generation.outputs(0).token_ids(0), 101);
  EXPECT_EQ(generation.outputs(0).finish_reason(), "stop");
  ASSERT_EQ(generation.outputs(0).logprobs_size(), 1);
  EXPECT_EQ(generation.outputs(0).logprobs(0).log_prob_data().token(), "精");
  EXPECT_FLOAT_EQ(generation.outputs(0).logprobs(0).log_prob_data().logprob(),
                  -0.25f);
  ASSERT_EQ(generation.outputs(0).logprobs(0).top_logprobs_size(), 1);
  EXPECT_EQ(generation.outputs(0).logprobs(0).top_logprobs(0).token(), "准");
  EXPECT_EQ(generation.usage().num_cached_tokens(), 6);
}

TEST(RequestOutputProtoAdapterTest, RejectsInvalidIdentityAndUsage) {
  RequestOutput output;
  output.usage = Usage{.num_prompt_tokens = -1};
  proto::DisaggStreamGeneration generation;

  EXPECT_FALSE(request_output_to_proto(output, "prefill-1", &generation));
  output.usage = Usage{.num_prompt_tokens = 8,
                       .num_generated_tokens = 0,
                       .num_total_tokens = 8,
                       .num_cached_tokens = 0};
  EXPECT_TRUE(request_output_to_proto(output, "prefill-1", &generation));
  output.usage.reset();
  EXPECT_FALSE(request_output_to_proto(output, "", &generation));
  EXPECT_FALSE(request_output_to_proto(output, "prefill-1", nullptr));
}

}  // namespace
}  // namespace xllm
