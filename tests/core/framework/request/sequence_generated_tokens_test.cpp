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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "framework/request/incremental_decoder.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"

namespace xllm {
namespace {

Sequence make_decode_ready_sequence(bool enable_schedule_overlap) {
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = 8;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = enable_schedule_overlap;
  params.rec_type = RecType::kNone;
  params.bos_token_id = 0;
  params.request_id = "generated_tokens_req";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;

  std::vector<int32_t> prompt_token_ids = {1, 2, 3};
  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/prompt_token_ids.size(),
      /*echo=*/params.echo,
      /*skip_special_tokens=*/params.skip_special_tokens);
  Sequence sequence(/*index=*/0,
                    prompt_token_ids,
                    /*input_embedding=*/torch::Tensor(),
                    /*mm_data=*/MMData(),
                    decoder,
                    params);
  // Move the sequence out of the prefill stage so append_token / the decode
  // path treats new tokens as generated decode tokens.
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  return sequence;
}

class GeneratedTokenTokenizer final : public Tokenizer {
 public:
  std::string decode(const Slice<int32_t>& ids, bool) const override {
    std::string text;
    for (const int32_t token_id : ids) {
      text.push_back(static_cast<char>(token_id));
    }
    return text;
  }
};

class RequestGeneratedTokensTest
    : public ::testing::TestWithParam<std::tuple<bool, size_t, size_t>> {};

TEST_P(RequestGeneratedTokensTest, UsageCountsOnlyCommittedTokens) {
  const auto [enable_overlap, committed_tokens, placeholder_tokens] =
      GetParam();
  const std::vector<int32_t> prompt_tokens = {'P'};
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  RequestState state(
      "P",
      prompt_tokens,
      sampling_param,
      SchedulerParam{},
      stopping_checker,
      32,
      1,
      1,
      false,
      false,
      false,
      true,
      enable_overlap,
      [](const RequestOutput&) { return true; },
      OutputsFunc{});
  Request request("generated-token-usage", "", "", std::move(state));
  Sequence& sequence = *request.sequences()[0];
  sequence.kv_state().set_kv_cache_tokens_num(prompt_tokens.size());
  for (size_t index = 0; index < committed_tokens; ++index) {
    if (enable_overlap) {
      sequence.append_token(Token(-1));
      sequence.update_last_step_token(Token('A'), 0);
    } else {
      sequence.append_token(Token('A'));
    }
  }
  for (size_t index = 0; index < placeholder_tokens; ++index) {
    sequence.append_token(Token(-1));
  }
  GeneratedTokenTokenizer tokenizer;
  RequestOutput output = request.generate_output(tokenizer);
  ASSERT_TRUE(output.usage.has_value());
  EXPECT_EQ(output.usage->num_prompt_tokens, prompt_tokens.size());
  EXPECT_EQ(output.usage->num_generated_tokens, committed_tokens);
  EXPECT_EQ(output.usage->num_total_tokens,
            prompt_tokens.size() + committed_tokens);
}

INSTANTIATE_TEST_SUITE_P(OverlapPadding,
                         RequestGeneratedTokensTest,
                         ::testing::Values(std::make_tuple(false, 0u, 0u),
                                           std::make_tuple(false, 3u, 0u),
                                           std::make_tuple(true, 0u, 0u),
                                           std::make_tuple(true, 0u, 1u),
                                           std::make_tuple(true, 0u, 2u),
                                           std::make_tuple(true, 3u, 0u),
                                           std::make_tuple(true, 3u, 1u),
                                           std::make_tuple(true, 3u, 2u)));

}  // namespace

TEST(SequenceGeneratedTokensTest, CountsRealAppendedTokens) {
  Sequence sequence = make_decode_ready_sequence(
      /*enable_schedule_overlap=*/false);
  EXPECT_EQ(sequence.generated_tokens_since_latency(), 0u);

  sequence.append_token(Token(10));
  sequence.append_token(Token(11));

  EXPECT_EQ(sequence.generated_tokens_since_latency(), 2u);
}

TEST(SequenceGeneratedTokensTest, TbtResetsGeneratedTokenCount) {
  Sequence sequence = make_decode_ready_sequence(
      /*enable_schedule_overlap=*/false);
  sequence.append_token(Token(10));
  sequence.append_token(Token(11));
  ASSERT_EQ(sequence.generated_tokens_since_latency(), 2u);

  sequence.tbt(absl::Now());
  EXPECT_EQ(sequence.generated_tokens_since_latency(), 0u);

  sequence.append_token(Token(12));
  EXPECT_EQ(sequence.generated_tokens_since_latency(), 1u);
}

TEST(SequenceGeneratedTokensTest, IgnoresOverlapFakeTokens) {
  Sequence sequence = make_decode_ready_sequence(
      /*enable_schedule_overlap=*/true);
  // Under schedule overlap a placeholder token id (< 0) is appended and the
  // real token is committed later via update_last_step_token. Fake tokens must
  // not inflate the committed-token count used for amortized TPOT.
  sequence.append_token(Token(-1));
  sequence.append_token(Token(-1));

  EXPECT_EQ(sequence.generated_tokens_since_latency(), 0u);
}

TEST(SequenceGeneratedTokensTest, CountsOverlapCommittedTokens) {
  Sequence sequence = make_decode_ready_sequence(
      /*enable_schedule_overlap=*/true);
  EXPECT_EQ(sequence.generated_tokens_since_latency(), 0u);

  sequence.update_last_step_token(Token(10), /*token_offset=*/0);
  sequence.update_last_step_token(Token(11), /*token_offset=*/0);

  EXPECT_EQ(sequence.generated_tokens_since_latency(), 2u);
}

}  // namespace xllm
