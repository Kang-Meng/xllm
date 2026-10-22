/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <memory>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/incremental_decoder.h"
#include "request.h"
#include "request_state.h"

namespace xllm {
namespace {

template <typename T>
class ScopedValue final {
 public:
  ScopedValue(T* target, T value) : target_(target), old_(*target) {
    *target_ = value;
  }
  ~ScopedValue() { *target_ = old_; }

  ScopedValue(const ScopedValue&) = delete;
  ScopedValue& operator=(const ScopedValue&) = delete;

 private:
  T* target_;
  T old_;
};

Sequence MakeSequence(size_t index, const std::vector<int32_t>& prompt_tokens) {
  RequestSamplingParam sampling_param;
  sampling_param.beam_width = 0;
  sampling_param.is_embeddings = false;

  StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = prompt_tokens.size() + 8;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = false;
  params.rec_type = RecType::kNone;
  params.bos_token_id = 0;
  params.request_id = "cached_tokens_test";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;

  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/prompt_tokens.size(),
      /*echo=*/params.echo,
      /*skip_special_tokens=*/params.skip_special_tokens);

  return Sequence(index,
                  prompt_tokens,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  params);
}

std::shared_ptr<Request> MakeRequest(
    const std::vector<int32_t>& prompt_tokens) {
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  RequestState state(
      "prompt",
      prompt_tokens,
      sampling_param,
      SchedulerParam{},
      stopping_checker,
      /*seq_capacity=*/prompt_tokens.size() + 8,
      /*n=*/1,
      /*best_of=*/1,
      /*logprobs=*/false,
      /*stream=*/false,
      /*echo=*/false,
      /*skip_special_tokens=*/true,
      /*enable_schedule_overlap=*/false,
      [](const RequestOutput&) { return true; },
      OutputsFunc{});

  return std::make_shared<Request>(
      "cached-tokens-test", "", "", std::move(state));
}

void EvictPrefixCache(BlockManagerPool& pool,
                      const BlockManagerPool::Options& options) {
  int32_t dp_rank = 0;
  pool.allocate((options.num_blocks() - 1) * options.block_size(), dp_rank);
}

TEST(RequestCachedTokensTest, RecordsCachedTokensFromPrefixCache) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.max_seqs_per_batch(1024);
  BlockManagerPool pool(options, /*dp_size=*/1);

  {
    // Populate prefix cache via a standalone sequence.
    Sequence seq1 =
        MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9});
    ASSERT_TRUE(pool.allocate(&seq1));
    seq1.kv_state().set_kv_cache_tokens_num(seq1.num_prompt_tokens());
    pool.cache(&seq1);

    // Create a Request whose prompt overlaps with the cached prefix.
    auto request = MakeRequest({1, 2, 3, 4, 5, 6, 7, 8, 10});
    auto* seq = request->sequences()[0].get();
    ASSERT_TRUE(pool.allocate(seq, seq->num_prompt_tokens()));

    EXPECT_EQ(seq->num_prefix_cache_tokens(), 8u);

    request->record_num_prefix_cache_tokens();
    EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);
  }

  EvictPrefixCache(pool, options);
}

TEST(RequestCachedTokensTest, CachedTokensSurvivesKVBlockRelease) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.max_seqs_per_batch(1024);
  BlockManagerPool pool(options, /*dp_size=*/1);

  std::shared_ptr<Request> request;
  {
    Sequence seq1 =
        MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9});
    ASSERT_TRUE(pool.allocate(&seq1));
    seq1.kv_state().set_kv_cache_tokens_num(seq1.num_prompt_tokens());
    pool.cache(&seq1);

    request = MakeRequest({1, 2, 3, 4, 5, 6, 7, 8, 10});
    auto* seq = request->sequences()[0].get();
    ASSERT_TRUE(pool.allocate(seq, seq->num_prompt_tokens()));

    request->record_num_prefix_cache_tokens();
    EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);

    // Simulate KV block release (preemption or finish).
    pool.deallocate(seq);
    EXPECT_EQ(seq->num_prefix_cache_tokens(), 0u);

    // The recorded value on Request must survive the release.
    EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);
  }

  EvictPrefixCache(pool, options);
}

TEST(RequestCachedTokensTest, RestoredSequenceReportsAtMostPromptTokens) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.max_seqs_per_batch(1024);
  BlockManagerPool pool(options, /*dp_size=*/1);

  {
    // Cache the 8-token prompt plus 5 generated tokens; full blocks cover 12.
    Sequence cached = MakeSequence(
        0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    ASSERT_TRUE(pool.allocate(&cached));
    cached.kv_state().set_kv_cache_tokens_num(cached.num_prompt_tokens());
    pool.cache(&cached);

    // A preempted request re-matches its prompt and generated tokens.
    auto request = MakeRequest({1, 2, 3, 4, 5, 6, 7, 8});
    auto* seq = request->sequences()[0].get();
    seq->kv_state().set_kv_cache_tokens_num(seq->num_prompt_tokens());
    for (int32_t token_id : {9, 10, 11, 12, 13}) {
      seq->append_token(Token(token_id));
    }
    seq->reset();
    ASSERT_TRUE(pool.allocate(seq, seq->num_tokens()));
    ASSERT_EQ(seq->kv_state().shared_tokens_num(), 12u);

    EXPECT_EQ(seq->num_prefix_cache_tokens(), 8u);
    request->record_num_prefix_cache_tokens();
    EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);
    pool.deallocate(seq);
  }

  EvictPrefixCache(pool, options);
}

TEST(RequestCachedTokensTest, ZeroWhenPrefixCacheDisabled) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      false);
  BlockManagerPool pool(options, /*dp_size=*/1);

  auto request = MakeRequest({1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto* seq = request->sequences()[0].get();
  ASSERT_TRUE(pool.allocate(seq, seq->num_prompt_tokens()));

  request->record_num_prefix_cache_tokens();
  EXPECT_EQ(request->num_prefix_cache_tokens(), 0u);
}

TEST(RequestCachedTokensTest, RecordIsIdempotentWithMaxSemantics) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.max_seqs_per_batch(1024);
  BlockManagerPool pool(options, /*dp_size=*/1);

  std::shared_ptr<Request> request;
  {
    Sequence seq1 =
        MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9});
    ASSERT_TRUE(pool.allocate(&seq1));
    seq1.kv_state().set_kv_cache_tokens_num(seq1.num_prompt_tokens());
    pool.cache(&seq1);

    request = MakeRequest({1, 2, 3, 4, 5, 6, 7, 8, 10});
    auto* seq = request->sequences()[0].get();
    ASSERT_TRUE(pool.allocate(seq, seq->num_prompt_tokens()));

    request->record_num_prefix_cache_tokens();
    EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);

    // Release KV blocks, then record again — max semantics preserves the value.
    pool.deallocate(seq);
    request->record_num_prefix_cache_tokens();
    EXPECT_EQ(request->num_prefix_cache_tokens(), 8u);
  }

  EvictPrefixCache(pool, options);
}

TEST(RequestCachedTokensTest, AdmissionRecordsFinalMaximumOnlyOnce) {
  auto request = MakeRequest({1, 2, 3, 4, 5, 6, 7, 8, 9});
  const double prompt_before = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_before = COUNTER_VALUE(prefix_cache_hit_tokens_total);

  request->record_num_prefix_cache_tokens(/*num_prefix_cache_tokens=*/6);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_prompt_tokens_total),
                   prompt_before);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_hit_tokens_total), hit_before);

  request->record_num_prefix_cache_tokens();
  request->record_num_prefix_cache_tokens();

  EXPECT_EQ(request->num_prefix_cache_tokens(), 6U);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_prompt_tokens_total),
                   prompt_before + 9.0);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_hit_tokens_total),
                   hit_before + 6.0);
}

TEST(RequestCachedTokensTest, CancelledOutputPreservesUsageAndStatus) {
  for (const int32_t generated_tokens : {0, 1}) {
    SCOPED_TRACE(generated_tokens);
    auto request = MakeRequest({1, 2, 3, 4, 5, 6, 7, 8, 9});
    request->record_num_prefix_cache_tokens(/*num_prefix_cache_tokens=*/8);
    auto& sequence = *request->sequences().front();
    sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
    if (generated_tokens != 0) {
      sequence.append_token(Token(10));
    }
    request->set_cancel();

    const Tokenizer tokenizer;
    const RequestOutput output = request->generate_output(tokenizer);
    EXPECT_TRUE(output.cancelled);
    ASSERT_TRUE(output.status.has_value());
    EXPECT_EQ(output.status->code(), StatusCode::CANCELLED);
    EXPECT_TRUE(output.outputs.empty());
    ASSERT_TRUE(output.usage.has_value());
    EXPECT_EQ(output.usage->num_prompt_tokens, 9);
    EXPECT_EQ(output.usage->num_generated_tokens, generated_tokens);
    EXPECT_EQ(output.usage->num_cached_tokens, 8);
    EXPECT_EQ(output.usage->num_total_tokens, 9 + generated_tokens);

    const double cancelled_before =
        COUNTER_VALUE(request_status_total_cancelled);
    const double ok_before = COUNTER_VALUE(request_status_total_ok);
    output.log_request_status();
    EXPECT_DOUBLE_EQ(COUNTER_VALUE(request_status_total_cancelled),
                     cancelled_before + 1.0);
    EXPECT_DOUBLE_EQ(COUNTER_VALUE(request_status_total_ok), ok_before);
  }
}

TEST(RequestCachedTokensTest, RequestErrorTakesPrecedenceOverCancellation) {
  auto request = MakeRequest({1, 2, 3});
  request->record_num_prefix_cache_tokens(/*num_prefix_cache_tokens=*/2);
  request->sequences().front()->fail(Status(StatusCode::INVALID_ARGUMENT));
  request->set_cancel();

  const Tokenizer tokenizer;
  const RequestOutput output = request->generate_output(tokenizer);
  EXPECT_TRUE(output.cancelled);
  ASSERT_TRUE(output.status.has_value());
  EXPECT_EQ(output.status->code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_TRUE(output.outputs.empty());
  ASSERT_TRUE(output.usage.has_value());
  EXPECT_EQ(output.usage->num_cached_tokens, 2);
}

TEST(RequestCachedTokensTest, NonCancelledOutputKeepsOkStatus) {
  auto request = MakeRequest({1, 2, 3});
  const Tokenizer tokenizer;
  const RequestOutput output = request->generate_output(tokenizer);
  EXPECT_FALSE(output.cancelled);
  ASSERT_TRUE(output.status.has_value());
  EXPECT_EQ(output.status->code(), StatusCode::OK);
}

}  // namespace
}  // namespace xllm
