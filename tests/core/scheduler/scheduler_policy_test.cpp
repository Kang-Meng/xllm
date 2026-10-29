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

#include "scheduler/scheduler_policy.h"

#include <absl/time/clock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "continuous_scheduler.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "distributed_runtime/engine.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/speculative/spec_input_builder.h"
#include "runtime/forward_params.h"
#include "util/utils.h"

namespace xllm {

namespace {

class FakeTokenizer : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens) const {
    NOT_IMPLEMENTED();
  }
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const {
    NOT_IMPLEMENTED();
  }
  std::optional<int32_t> token_to_id(const std::string_view& token) const {
    NOT_IMPLEMENTED();
  }
  std::string id_to_token(int32_t id) const { NOT_IMPLEMENTED(); }
  size_t vocab_size() const { NOT_IMPLEMENTED(); }
  std::unique_ptr<Tokenizer> clone() const {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine : public Engine {
 public:
  FakeEngine(int32_t num_blocks,
             int32_t block_size,
             bool enable_prefix_cache = false,
             bool enable_linear_attention = false,
             int32_t speculative_tokens = 0) {
    BlockManagerPool::Options opt;
    opt.num_blocks_ = num_blocks;
    opt.block_size_ = block_size;
    opt.max_seqs_per_batch_ = 1024;
    opt.enable_prefix_cache_ = enable_prefix_cache;
    opt.enable_linear_state_ = enable_linear_attention;
    opt.num_speculative_tokens_ = speculative_tokens;
    opt.num_embedding_blocks_ =
        speculative_tokens > 0 ? opt.max_seqs_per_batch_ : 0;
    if (enable_linear_attention) {
      opt.linear_state_num_slots_ = 64;
      model_args_.layer_types({"linear_attention"});
    }
    fake_tokenizer_ = std::make_unique<FakeTokenizer>();
    fake_block_manager_ = std::make_unique<BlockManagerPool>(opt, 1);
  }
  ForwardOutput step(std::vector<Batch>& batch) { return {}; }
  void update_last_step_result(std::vector<Batch>& batch) { NOT_IMPLEMENTED(); }
  const Tokenizer* tokenizer() const { return fake_tokenizer_.get(); }
  BlockManagerPool* block_manager_pool() const {
    return fake_block_manager_.get();
  }
  const ModelArgs& model_args() const override { return model_args_; }
  const TokenizerArgs& tokenizer_args() const { NOT_IMPLEMENTED(); }
  std::vector<int64_t> get_active_activation_memory() const { return {0}; }
  bool init() override { return true; }

 private:
  std::unique_ptr<Tokenizer> fake_tokenizer_;
  std::unique_ptr<BlockManagerPool> fake_block_manager_;
  ModelArgs model_args_;
};

class TrackingBlockManagerPool final : public BlockManagerPool {
 public:
  TrackingBlockManagerPool()
      : BlockManagerPool(make_options(), /*dp_size=*/1) {}

  void allocate_shared(Sequence* sequence) override {
    ++allocate_shared_calls_;
    sequence->kv_state().set_prefix_cache_matched();
  }

  int32_t allocate_shared_calls() const { return allocate_shared_calls_; }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 16;
    options.block_size_ = 128;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = true;
    return options;
  }

  int32_t allocate_shared_calls_ = 0;
};

class RetryRestoreBlockManagerPool final : public BlockManagerPool {
 public:
  RetryRestoreBlockManagerPool()
      : BlockManagerPool(make_options(), /*dp_size=*/1) {}

  bool allocate(Sequence* sequence, size_t num_tokens) override {
    allocate_targets_.emplace_back(num_tokens);
    if (allocate_targets_.size() == 1) {
      sequence->host_kv_state().reset();
      sequence->clear_host_cache_match();
      return false;
    }
    return true;
  }

  void allocate_shared(Sequence* sequence) override {
    ++allocate_shared_calls_;
    sequence->host_kv_state().set_prefix_cache_matched();
    sequence->set_host_cache_match(/*restore_tokens=*/32768,
                                   /*copy_units=*/2);
  }

  const std::vector<size_t>& allocate_targets() const {
    return allocate_targets_;
  }

  bool needs_shared_reprobe(Sequence* sequence) const override {
    return !sequence->host_kv_state().prefix_cache_matched();
  }

  bool supports_host_cache_restore() const override { return true; }

  int32_t allocate_shared_calls() const { return allocate_shared_calls_; }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 512;
    options.block_size_ = 128;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = true;
    return options;
  }

  std::vector<size_t> allocate_targets_;
  int32_t allocate_shared_calls_ = 0;
};

class PendingReleaseBlockManagerPool final : public BlockManagerPool {
 public:
  PendingReleaseBlockManagerPool()
      : BlockManagerPool(make_options(), /*dp_size=*/1) {}

  bool allocate(Sequence* /*sequence*/, size_t /*num_tokens*/) override {
    ++allocate_calls_;
    return false;
  }

  void allocate_shared(Sequence* sequence) override {
    sequence->kv_state().set_prefix_cache_matched();
  }

  bool has_pending_async_block_release() const override { return true; }

  int32_t allocate_calls() const { return allocate_calls_; }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 256;
    options.block_size_ = 128;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = true;
    return options;
  }

  int32_t allocate_calls_ = 0;
};

class RestoreWaitingBlockManagerPool final : public BlockManagerPool {
 public:
  RestoreWaitingBlockManagerPool()
      : BlockManagerPool(make_options(), /*dp_size=*/1) {}

  bool allocate(Sequence* /*sequence*/, size_t /*num_tokens*/) override {
    ++allocate_calls_;
    return !pending_async_release_;
  }

  void allocate_shared(Sequence* sequence) override {
    ++allocate_shared_calls_;
    sequence->kv_state().set_prefix_cache_matched();
  }

  bool has_pending_async_block_release() const override {
    return pending_async_release_;
  }

  void set_pending_async_release(bool pending) {
    pending_async_release_ = pending;
  }

  int32_t allocate_calls() const { return allocate_calls_; }

  int32_t allocate_shared_calls() const { return allocate_shared_calls_; }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 256;
    options.block_size_ = 128;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = true;
    return options;
  }

  bool pending_async_release_ = true;
  int32_t allocate_calls_ = 0;
  int32_t allocate_shared_calls_ = 0;
};

class DecodeVictimBlockManagerPool final : public BlockManagerPool {
 public:
  DecodeVictimBlockManagerPool()
      : BlockManagerPool(make_options(), /*dp_size=*/1) {}

  bool allocate(Sequence* /*sequence*/, size_t /*num_tokens*/) override {
    return false;
  }

  bool supports_host_cache_restore() const override { return true; }

  void deallocate(Request* request) override {
    BlockManagerPool::deallocate(request);
    pending_async_release_ = true;
  }

  bool has_pending_async_block_release() const override {
    return pending_async_release_;
  }

  void complete_async_release() { pending_async_release_ = false; }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 256;
    options.block_size_ = 128;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = true;
    return options;
  }

  bool pending_async_release_ = false;
};

class RestorePriorityBlockManagerPool final : public BlockManagerPool {
 public:
  RestorePriorityBlockManagerPool()
      : BlockManagerPool(make_options(), /*dp_size=*/1) {}

  void set_sequences(Sequence* blocked, Sequence* victim) {
    blocked_ = blocked;
    victim_ = victim;
  }

  bool allocate(Sequence* sequence, size_t /*num_tokens*/) override {
    if (sequence == blocked_) {
      if (free_blocks_ < kBlockedGrowthBlocks) {
        return false;
      }
      free_blocks_ -= kBlockedGrowthBlocks;
      return true;
    }
    if (sequence == victim_) {
      if (free_blocks_ < kVictimRestoreBlocks) {
        return false;
      }
      free_blocks_ -= kVictimRestoreBlocks;
      return true;
    }
    return false;
  }

  void allocate_shared(Sequence* sequence) override {
    if (sequence != victim_) {
      return;
    }
    CHECK_GT(sequence->num_tokens(), 0u);
    sequence->kv_state().set_prefix_cache_matched();
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_tokens() - 1);
  }

  bool supports_host_cache_restore() const override { return true; }

  void deallocate(Request* request) override {
    BlockManagerPool::deallocate(request);
    pending_async_release_ = true;
  }

  bool has_pending_async_block_release() const override {
    return pending_async_release_;
  }

  void complete_release() {
    pending_async_release_ = false;
    free_blocks_ = kVictimRestoreBlocks;
  }

 private:
  static BlockManagerPool::Options make_options() {
    BlockManagerPool::Options options;
    options.num_blocks_ = 10;
    options.block_size_ = 128;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = true;
    return options;
  }

  static constexpr size_t kBlockedGrowthBlocks = 1;
  static constexpr size_t kVictimRestoreBlocks = 3;
  Sequence* blocked_ = nullptr;
  Sequence* victim_ = nullptr;
  size_t free_blocks_ = 0;
  bool pending_async_release_ = false;
};

class TestUnifiedPolicy final : public UnifiedPolicy {
 public:
  using SchedulerPolicy::allocate_for_prefill;
  using UnifiedPolicy::UnifiedPolicy;

  void allocate_shared_blocks_for_test(Sequence* sequence,
                                       SchedulerState& state) {
    allocate_shared_blocks_for(sequence, state);
  }
};

template <typename T>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(T& value, T new_value) : value_(value), old_(value) {
    value_ = new_value;
  }

  ~ScopedConfigValue() { value_ = old_; }

 private:
  T& value_;
  T old_;
};

ContinuousScheduler::Options create_scheduler_options(
    int32_t max_tokens_per_batch,
    int32_t max_seqs_per_batch,
    int32_t num_speculative_tokens,
    int32_t max_tokens_per_chunk_for_prefill,
    int32_t dp_size,
    const std::string& priority_strategy = "fcfs",
    bool enable_profile_kv_blocks = true,
    bool enable_latency_aware_schedule = false,
    int32_t max_global_ttft_ms = std::numeric_limits<int32_t>::max(),
    int32_t max_global_tpot_ms = std::numeric_limits<int32_t>::max()) {
  ContinuousScheduler::Options opt;
  opt.num_speculative_tokens_ = num_speculative_tokens;
  opt.max_tokens_per_chunk_for_prefill_ = max_tokens_per_chunk_for_prefill;
  opt.max_tokens_per_batch_ = max_tokens_per_batch;
  opt.max_seqs_per_batch_ = max_seqs_per_batch;
  opt.dp_size_ = dp_size;
  opt.priority_strategy_ = priority_strategy;
  opt.enable_profile_kv_blocks_ = enable_profile_kv_blocks;
  opt.enable_latency_aware_schedule_ = enable_latency_aware_schedule;
  opt.max_global_ttft_ms_ = max_global_ttft_ms;
  opt.max_global_tpot_ms_ = max_global_tpot_ms;
  return opt;
}

std::vector<std::shared_ptr<Request>> generate_request(
    const std::vector<int32_t>& prompt_lens,
    const std::vector<int32_t>& max_tokens,
    std::optional<std::vector<bool>> offlines,
    std::optional<std::vector<int32_t>> priorities,
    int32_t max_context_len) {
  std::vector<std::shared_ptr<Request>> requests;
  EXPECT_TRUE(prompt_lens.size() == max_tokens.size());

  size_t batch_size = prompt_lens.size();
  std::vector<bool> offline_vec;
  std::vector<int32_t> priority_vec;
  if (offlines.has_value()) {
    offline_vec = *offlines;
  } else {
    offline_vec = std::vector<bool>(batch_size, false);
  }

  if (priorities.has_value()) {
    priority_vec = priorities.value();
  } else {
    priority_vec = std::vector<int32_t>(
        batch_size, static_cast<int32_t>(RequestPriority::NORMAL));
  }

  for (size_t i = 0; i < batch_size; ++i) {
    std::vector<int32_t> prompt_token_ids;
    prompt_token_ids.resize(prompt_lens[i]);
    RequestSamplingParam sampling_param;
    SchedulerParam scheduler_param;
    scheduler_param.offline = offline_vec[i];
    scheduler_param.priority = static_cast<RequestPriority>(priority_vec[i]);

    StoppingChecker stopping_checker;
    stopping_checker.set_max_generated_tokens(max_tokens[i]);
    stopping_checker.set_max_context_len(max_context_len);
    stopping_checker.set_ignore_eos(true);

    RequestState req_state("x",
                           prompt_token_ids,
                           sampling_param,
                           scheduler_param,
                           stopping_checker,
                           prompt_lens[i] + 30000,
                           1,
                           1,
                           false,
                           false,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr);

    auto request =
        std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
    requests.emplace_back(request);
  }

  return requests;
}

// dont not consider speculative decoding.
void update_requests(std::vector<std::shared_ptr<Request>> requests) {
  for (auto req : requests) {
    for (auto& seq : req->sequences()) {
      if (seq->kv_state().kv_cache_tokens_num() == 0) {
        seq->kv_state().incr_kv_cache_tokens_num(seq->num_prompt_tokens());
      } else {
        seq->kv_state().incr_kv_cache_tokens_num(1);
      }
      Token token(1);
      seq->append_token(token);
    }
  }
}

}  // namespace

// TEST-1:
// Three independent prefill requests, according to the configs,
// verify how many tokens are processed in one scheduling.
TEST(SchedulerPolicyTest, AddNewRequestBase) {
  std::vector<int32_t> prompt_len{10, 1024, 2048};
  std::vector<int32_t> num_blocks{16, 128, 128};
  std::vector<int32_t> block_size{16, 16, 16};
  std::vector<int32_t> validate_allowed_max_tokens{10, 1024, 1024};
  for (size_t idx = 0; idx < prompt_len.size(); ++idx) {
    ContinuousScheduler::Options opt =
        create_scheduler_options(10000, 256, 0, 1024, 1);
    auto engine =
        std::make_unique<FakeEngine>(num_blocks[idx], block_size[idx]);
    auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
    EXPECT_TRUE(scheduler != nullptr);

    // create requests
    auto requests = generate_request(
        {prompt_len[idx]}, {10}, std::nullopt, std::nullopt, 10000);
    for (auto req : requests) {
      scheduler->add_request(req);
    }

    auto batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    const std::vector<uint32_t>& allowed_max_tokens =
        batch[0].get_allowed_max_tokens();
    for (size_t i = 0; i < batch[0].size(); ++i) {
      auto seq = batch[0][i];
      EXPECT_TRUE(allowed_max_tokens[i] == validate_allowed_max_tokens[idx]);
    }
  }
}

TEST(SchedulerPolicyTest, UnifiedDrainRejectsBeforeQueueingRequest) {
  ContinuousScheduler::Options options =
      create_scheduler_options(1024, 16, 0, 256, 1, "multi_slo_and_prio");
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "multi_slo_and_prio",
  };
  std::unique_ptr<SchedulerPolicy> policy =
      create_scheduler_policy(mode, options);
  FakeEngine engine(16, 128);
  auto requests =
      generate_request({16, 16}, {1, 1}, std::nullopt, std::nullopt, 10000);
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  int32_t reject_calls = 0;
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = engine.block_manager_pool(),
      .profile_manager = nullptr,
      .response_processor = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = false,
      .reject_queued_request =
          [&](const std::shared_ptr<Request>& request) {
            ++reject_calls;
            return request == requests[0];
          },
  };
  folly::MPMCQueue<std::shared_ptr<Request>> request_queue(2);
  ASSERT_TRUE(request_queue.write(requests[0]));
  ASSERT_TRUE(request_queue.write(requests[1]));

  policy->drain_request_queue(state, request_queue);

  EXPECT_EQ(reject_calls, 2);
  ASSERT_EQ(unified_queue.size(), 1u);
  EXPECT_EQ(unified_queue.front(), requests[1]);
}

TEST(SchedulerPolicyTest, UnifiedPrefixHitIncludesScheduledSuffixCapacity) {
  ScopedConfigValue<bool> mix_batch_guard(
      SchedulerConfig::get_instance().enable_mix_batch(), true);
  constexpr int32_t kBlockSize = 128;
  constexpr int32_t kPrefixTokens = 256;
  constexpr int32_t kPromptTokens = 2313;
  constexpr int32_t kChunkTokens = 256;

  ContinuousScheduler::Options opt = create_scheduler_options(
      /*max_tokens_per_batch=*/kChunkTokens,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/kChunkTokens,
      /*dp_size=*/1,
      /*priority_strategy=*/"multi_slo_and_prio");
  auto engine = std::make_unique<FakeEngine>(
      /*num_blocks=*/64, kBlockSize, /*enable_prefix_cache=*/true);

  auto cached_requests =
      generate_request({kPrefixTokens}, {1}, std::nullopt, std::nullopt, 10000);
  Sequence* cached_sequence = cached_requests[0]->sequences()[0].get();
  ASSERT_TRUE(engine->block_manager_pool()->allocate(cached_sequence));
  cached_sequence->kv_state().set_kv_cache_tokens_num(kPrefixTokens);
  engine->block_manager_pool()->cache(cached_sequence);
  engine->block_manager_pool()->deallocate(cached_sequence);

  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  auto requests =
      generate_request({kPromptTokens}, {1}, std::nullopt, std::nullopt, 10000);
  scheduler->add_request(requests[0]);

  std::vector<Batch> batches = scheduler->prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches[0].size(), 1u);
  Sequence* hit_sequence = requests[0]->sequences()[0].get();
  EXPECT_EQ(hit_sequence->kv_cache_tokens_num(), kPrefixTokens);
  EXPECT_EQ(batches[0].get_allowed_max_tokens()[0], kChunkTokens);
  EXPECT_GE(hit_sequence->kv_state().current_max_tokens_capacity(),
            kPrefixTokens + kChunkTokens);
}

TEST(SchedulerPolicyTest,
     PrefillPlanningAcceptsPartialAndRedistributedBudgets) {
  ContinuousScheduler::Options options =
      create_scheduler_options(1024, 16, 0, 256, 1);
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  TestUnifiedPolicy policy(mode, options);
  FakeEngine engine(64, 128);
  auto requests =
      generate_request({777}, {1}, std::nullopt, std::nullopt, 10000);
  Sequence* sequence = requests.front()->sequences().front().get();
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  DecodeAdmission decode_admission(engine.block_manager_pool(),
                                   DecodeAdmission::Options{});
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = engine.block_manager_pool(),
      .profile_manager = nullptr,
      .response_processor = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = false,
      .decode_admission = &decode_admission,
  };
  for (const bool enable_prefix_cache : {false, true}) {
    state.enable_prefix_cache = enable_prefix_cache;
    size_t actual_tokens = 0;
    ASSERT_TRUE(
        policy.allocate_for_prefill(sequence, 128, &actual_tokens, state));
    EXPECT_EQ(actual_tokens, 128u);
    ASSERT_TRUE(
        policy.allocate_for_prefill(sequence, 256, &actual_tokens, state));
    EXPECT_EQ(actual_tokens, 256u);
    ASSERT_TRUE(
        policy.allocate_for_prefill(sequence, 1024, &actual_tokens, state));
    EXPECT_EQ(actual_tokens, 777u);
  }
  engine.block_manager_pool()->deallocate(sequence);
}

TEST(SchedulerPolicyTest, LinearSchedulingPreservesReadSourceAcrossPolicies) {
  ScopedConfigValue<bool> chunked_guard(
      SchedulerConfig::get_instance().enable_chunked_prefill(), true);
  ScopedConfigValue<int32_t> stride_guard(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 256);
  for (const bool prefix_cache : {false, true}) {
    ScopedConfigValue<bool> cache_guard(
        KVCacheConfig::get_instance().enable_prefix_cache(), prefix_cache);
    for (const std::string strategy : {"fcfs", "multi_slo_and_prio"}) {
      for (const bool mix_batch : {false, true}) {
        ScopedConfigValue<bool> mix_guard(
            SchedulerConfig::get_instance().enable_mix_batch(), mix_batch);
        ContinuousScheduler::Options options =
            create_scheduler_options(256, 16, 0, 256, 1, strategy);
        FakeEngine engine(128, 128, prefix_cache, true);
        ContinuousScheduler scheduler(&engine, options);
        auto requests =
            generate_request({777}, {1}, std::nullopt, std::nullopt, 10000);
        Sequence* sequence = requests.front()->sequences().front().get();
        scheduler.add_request(requests.front());
        size_t cached_tokens = 0;
        int32_t previous_output_id = -1;
        for (const size_t expected_tokens : {256, 256, 256, 9}) {
          auto batches = scheduler.prepare_batch_test();
          ASSERT_EQ(batches.size(), 1u);
          ASSERT_EQ(batches.front().size(), 1u);
          EXPECT_EQ(batches.front().get_allowed_max_tokens().front(),
                    expected_tokens);
          const Slice<Block> linear_blocks =
              sequence->kv_state().blocks(BlockType::LINEAR);
          ASSERT_EQ(linear_blocks.size(), cached_tokens == 0 ? 1u : 2u);
          if (cached_tokens > 0) {
            EXPECT_EQ(linear_blocks.front().id(), previous_output_id);
          }
          EXPECT_NE(linear_blocks.back().id(), previous_output_id);
          previous_output_id = linear_blocks.back().id();
          cached_tokens += expected_tokens;
          sequence->kv_state().set_kv_cache_tokens_num(cached_tokens);
        }
      }
    }
  }
}

TEST(SchedulerPolicyTest, LinearTailPrefillPreservesKvSplitBoundaries) {
  constexpr int32_t kPhysicalBlockSize = 128;
  constexpr int32_t kKvSplitSize = 2;
  constexpr int32_t kEffectiveBlockSize = kPhysicalBlockSize * kKvSplitSize;
  constexpr int32_t kChunkSize = 8192;
  ScopedConfigValue<bool> chunked_guard(
      SchedulerConfig::get_instance().enable_chunked_prefill(), true);
  ScopedConfigValue<bool> prefix_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  ScopedConfigValue<bool> mix_guard(
      SchedulerConfig::get_instance().enable_mix_batch(), false);
  ScopedConfigValue<int32_t> chunk_guard(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      kChunkSize);
  ScopedConfigValue<int32_t> cp_guard(ParallelConfig::get_instance().cp_size(),
                                      kKvSplitSize);
  ScopedConfigValue<int32_t> split_guard(
      ParallelConfig::get_instance().kv_split_size(), kKvSplitSize);

  const std::vector<std::tuple<int32_t, std::vector<size_t>>> test_cases = {
      {8191, {7936, 255}}, {16383, {8192, 7936, 255}}};
  for (const std::string strategy : {"fcfs", "multi_slo_and_prio"}) {
    SCOPED_TRACE(strategy);
    for (const auto& [prompt_tokens, expected_chunks] : test_cases) {
      SCOPED_TRACE(prompt_tokens);
      ContinuousScheduler::Options options =
          create_scheduler_options(kChunkSize, 16, 0, kChunkSize, 1, strategy);
      // LLM initialization passes a logical block size that already includes
      // KV split. The tail cap and LINEAR hash chain must use that same size.
      FakeEngine engine(/*num_blocks=*/128,
                        /*block_size=*/kEffectiveBlockSize,
                        /*enable_prefix_cache=*/true,
                        /*enable_linear_attention=*/true);
      ContinuousScheduler scheduler(&engine, options);
      auto requests = generate_request(
          {prompt_tokens}, {1}, std::nullopt, std::nullopt, 32768);
      Sequence* sequence = requests.front()->sequences().front().get();
      scheduler.add_request(requests.front());

      size_t cached_tokens = 0;
      for (const size_t expected_tokens : expected_chunks) {
        auto batches = scheduler.prepare_batch_test();
        ASSERT_EQ(batches.size(), 1u);
        ASSERT_EQ(batches.front().size(), 1u);
        ASSERT_EQ(batches.front().get_allowed_max_tokens().front(),
                  expected_tokens);
        if (expected_tokens == 255) {
          const Slice<Block> linear_blocks =
              sequence->kv_state().blocks(BlockType::LINEAR);
          ASSERT_EQ(linear_blocks.size(), 2u);
          sequence->update_block_hashes(kEffectiveBlockSize,
                                        BlockHasherType::TEXT);
          const size_t hash_index = cached_tokens / kEffectiveBlockSize - 1;
          const Slice<XXH3Key> hashes =
              sequence->block_hashes(kEffectiveBlockSize);
          ASSERT_LT(hash_index, hashes.size());
          EXPECT_TRUE(
              XXH3Key(linear_blocks.front().get_immutable_hash_value()) ==
              hashes[hash_index]);
        }
        cached_tokens += expected_tokens;
        if (cached_tokens < static_cast<size_t>(prompt_tokens)) {
          EXPECT_EQ(cached_tokens % kEffectiveBlockSize, 0u);
          EXPECT_EQ((cached_tokens / kKvSplitSize) % kPhysicalBlockSize, 0u);
        }
        sequence->kv_state().set_kv_cache_tokens_num(cached_tokens);
      }
      EXPECT_EQ(cached_tokens, static_cast<size_t>(prompt_tokens));
    }
  }
}

TEST(SchedulerPolicyTest, LinearWarmPrefixPrefillPreservesTailCheckpoint) {
  constexpr int32_t kBlockSize = 128;
  constexpr int32_t kChunkSize = 8192;
  constexpr int32_t kPromptTokens = 9314;
  constexpr size_t kTailCheckpoint = 9216;
  ScopedConfigValue<bool> chunked_guard(
      SchedulerConfig::get_instance().enable_chunked_prefill(), true);
  ScopedConfigValue<bool> prefix_guard(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  ScopedConfigValue<int32_t> chunk_guard(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      kChunkSize);

  for (const std::string strategy : {"fcfs", "multi_slo_and_prio"}) {
    SCOPED_TRACE(strategy);
    for (const bool mix_batch : {false, true}) {
      SCOPED_TRACE(mix_batch);
      ScopedConfigValue<bool> mix_guard(
          SchedulerConfig::get_instance().enable_mix_batch(), mix_batch);
      ContinuousScheduler::Options options =
          create_scheduler_options(kChunkSize, 16, 0, kChunkSize, 1, strategy);
      FakeEngine engine(/*num_blocks=*/128,
                        kBlockSize,
                        /*enable_prefix_cache=*/true,
                        /*enable_linear_attention=*/true);

      auto cached_requests = generate_request(
          {kChunkSize}, {1}, std::nullopt, std::nullopt, 32768);
      Sequence* cached_sequence =
          cached_requests.front()->sequences().front().get();
      ASSERT_TRUE(engine.block_manager_pool()->allocate(cached_sequence));
      cached_sequence->kv_state().set_kv_cache_tokens_num(kChunkSize);
      engine.block_manager_pool()->deallocate(cached_sequence);

      ContinuousScheduler scheduler(&engine, options);
      auto requests = generate_request(
          {kPromptTokens}, {1}, std::nullopt, std::nullopt, 32768);
      Sequence* sequence = requests.front()->sequences().front().get();
      ASSERT_EQ(sequence->kv_cache_tokens_num(), 0u);
      scheduler.add_request(requests.front());

      // A deallocated full-stride checkpoint is not restored before the first
      // chunk budget. That forward takes the whole stride. The next chunk
      // must still stop at the tail checkpoint, leaving the final 98 tokens
      // for a separate forward.
      auto batches = scheduler.prepare_batch_test();
      ASSERT_EQ(batches.size(), 1u);
      ASSERT_EQ(batches.front().size(), 1u);
      ASSERT_EQ(sequence->kv_cache_tokens_num(), 0u);
      ASSERT_EQ(batches.front().get_allowed_max_tokens().front(),
                static_cast<size_t>(kChunkSize));
      sequence->kv_state().set_kv_cache_tokens_num(kChunkSize);

      batches = scheduler.prepare_batch_test();
      ASSERT_EQ(batches.size(), 1u);
      ASSERT_EQ(batches.front().size(), 1u);
      ASSERT_EQ(batches.front().get_allowed_max_tokens().front(), 1024u);
      ASSERT_EQ(sequence->kv_state().current_max_tokens_capacity(),
                kTailCheckpoint);
      sequence->kv_state().set_kv_cache_tokens_num(kTailCheckpoint);

      batches = scheduler.prepare_batch_test();
      ASSERT_EQ(batches.size(), 1u);
      ASSERT_EQ(batches.front().size(), 1u);
      ASSERT_EQ(batches.front().get_allowed_max_tokens().front(), 98u);
      sequence->kv_state().set_kv_cache_tokens_num(kPromptTokens);
      engine.block_manager_pool()->deallocate(sequence);
    }
  }
}

TEST(SchedulerPolicyTest, DecodeFirstSkipsRedistributionForLinearModels) {
  ScopedConfigValue<bool> chunked_guard(
      SchedulerConfig::get_instance().enable_chunked_prefill(), true);
  ScopedConfigValue<bool> mix_guard(
      SchedulerConfig::get_instance().enable_mix_batch(), true);
  ScopedConfigValue<int32_t> stride_guard(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  for (const bool prefix_cache : {false, true}) {
    ScopedConfigValue<bool> cache_guard(
        KVCacheConfig::get_instance().enable_prefix_cache(), prefix_cache);
    for (const bool overlap : {false, true}) {
      auto options = create_scheduler_options(6, 8, 0, 4, 1);
      options.enable_schedule_overlap(overlap);
      FakeEngine engine(64, 2, prefix_cache, true);
      ContinuousScheduler scheduler(&engine, options);
      auto requests =
          generate_request({13}, {16}, std::nullopt, std::nullopt, 128);
      Sequence* sequence = requests.front()->sequences().front().get();
      int32_t previous_write_id = -1;
      scheduler.add_request(requests.front());
      size_t cached_tokens = 0;
      for (const size_t expected_tokens : {4u, 4u, 4u, 1u}) {
        auto batches = scheduler.prepare_batch_test();
        ASSERT_EQ(batches.size(), 1u);
        ASSERT_EQ(batches.front().size(), 1u);
        EXPECT_EQ(batches.front().get_allowed_max_tokens().front(),
                  expected_tokens);
        const Slice<Block> blocks =
            sequence->kv_state().blocks(BlockType::LINEAR);
        ASSERT_EQ(blocks.size(), cached_tokens == 0 ? 1u : 2u);
        if (cached_tokens > 0) {
          EXPECT_EQ(blocks.front().id(), previous_write_id);
        }
        EXPECT_NE(blocks.back().id(), previous_write_id);
        previous_write_id = blocks.back().id();
        cached_tokens += expected_tokens;
        sequence->kv_state().set_kv_cache_tokens_num(cached_tokens);
      }
    }
  }
}

TEST(SchedulerPolicyTest, MtpOverlapReservationControlsAdmissionAndPages) {
  constexpr int32_t kDraftTokens = 3;
  constexpr int32_t kPageSize = 128;
  // The old 2N lookahead ends at page 16; the bonus requires page 17.
  constexpr int32_t kScheduledTokens = 16 * kPageSize - 2 * kDraftTokens;
  // Admission includes the current token in addition to the 2N+1 lookahead.
  constexpr int32_t kDecodeBudget = 2 * kDraftTokens + 2;
  class NoTextTokenizer final : public Tokenizer {
   public:
    std::unique_ptr<Tokenizer> clone() const override {
      return std::make_unique<NoTextTokenizer>();
    }
  };
  class ResourceTestEngine final : public FakeEngine {
   public:
    ResourceTestEngine(int32_t num_blocks, int32_t block_size)
        : FakeEngine(num_blocks, block_size) {}
    const Tokenizer* tokenizer() const override { return &tokenizer_; }

   private:
    NoTextTokenizer tokenizer_;
  };
  struct TestCase {
    int32_t token_budget;
    int32_t request_count;
  };
  const TestCase cases[] = {{kDecodeBudget - 1, 1},
                            {kDecodeBudget, 1},
                            {2 * kDecodeBudget - 1, 2},
                            {2 * kDecodeBudget, 2}};
  ScopedConfigValue<std::string> algorithm(
      SpeculativeConfig::get_instance().speculative_algorithm(), "MTP");
  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.token_budget);
    ResourceTestEngine engine(/*num_blocks=*/64, kPageSize);
    ContinuousScheduler::Options options =
        create_scheduler_options(test_case.token_budget,
                                 test_case.request_count,
                                 kDraftTokens,
                                 /*max_tokens_per_chunk_for_prefill=*/16,
                                 /*dp_size=*/1);
    options.enable_schedule_overlap(true);
    ContinuousScheduler scheduler(&engine, options);
    auto requests = generate_request(
        std::vector<int32_t>(test_case.request_count, kScheduledTokens - 1),
        std::vector<int32_t>(test_case.request_count, 1024),
        std::nullopt,
        std::nullopt,
        8192);
    for (size_t index = 0; index < requests.size(); ++index) {
      const std::string id = std::to_string(index);
      requests[index] = std::make_shared<Request>(
          id, id, "", requests[index]->state().clone(), "");
      requests[index]->state().output_func =
          [](const RequestOutput& /*output*/) { return true; };
      Sequence* sequence = requests[index]->sequences().front().get();
      ASSERT_TRUE(
          engine.block_manager_pool()->allocate(sequence, kScheduledTokens));
      sequence->kv_state().set_kv_cache_tokens_num(kScheduledTokens - 1);
      sequence->append_token(1);
      ASSERT_EQ(sequence->num_tokens(), kScheduledTokens);
      ASSERT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 16u);
      ASSERT_TRUE(scheduler.add_request(requests[index]));
    }
    auto batches = scheduler.prepare_batch_test();
    ASSERT_EQ(batches.size(), 1u);
    const size_t expected_sequences =
        std::min(requests.size(),
                 static_cast<size_t>(test_case.token_budget / kDecodeBudget));
    ASSERT_EQ(batches.front().size(), expected_sequences);
    for (const auto& request : requests) {
      Sequence* sequence = request->sequences().front().get();
      const bool scheduled =
          (expected_sequences > 0 && batches.front()[0] == sequence) ||
          (expected_sequences == 2 && batches.front()[1] == sequence);
      EXPECT_EQ(sequence->kv_state().num_blocks(BlockType::KV),
                scheduled ? 17u : 16u);
      EXPECT_EQ(sequence->num_need_compute_tokens(), 1u);
    }
    // Bridge allocation and lookup with real physical block IDs, without
    // treating this Host-only check as full Batch/Worker table propagation.
    for (size_t index = 0; index < batches.front().size(); ++index) {
      Sequence* sequence = batches.front()[index];
      const auto& blocks = sequence->kv_state().blocks(BlockType::KV);
      std::vector<int32_t> block_ids;
      block_ids.reserve(blocks.size());
      for (const auto& block : blocks) {
        block_ids.emplace_back(block.id());
      }
      ForwardInput metadata_template;
      metadata_template.input_params.meta.num_sequences = 1;
      metadata_template.input_params.embedding.request_ids = {"boundary"};
      metadata_template.token_ids_host = torch::tensor({-1}, torch::kInt32);
      // The unresolved Host update can advance N before the N+1 template.
      const int32_t host_kv_len = kScheduledTokens + kDraftTokens;
      metadata_template.positions_host =
          torch::tensor({host_kv_len - 1}, torch::kInt32);
      specBuilder::append_seq_len_by_layout(
          metadata_template.input_params.attention.host.kv_seq_lens,
          host_kv_len);
      const auto states = specBuilder::advance_first_draft_template(
          metadata_template, torch::Tensor(), kDraftTokens);
      ASSERT_EQ(states.size(), 1u);
      const int32_t position =
          metadata_template.positions_host[0].item<int32_t>();
      ASSERT_EQ(position, 16 * kPageSize);
      EXPECT_EQ(
          specBuilder::calc_kv_len(
              metadata_template.input_params.attention.host.kv_seq_lens, 0, 0),
          position + 1);
      ASSERT_EQ(block_ids.size(), 17u);
      EXPECT_EQ(specBuilder::calc_slot_id(
                    position, {block_ids.data(), block_ids.size()}, kPageSize),
                blocks.back().id() * kPageSize);
    }
    for (const uint32_t allowed : batches.front().get_allowed_max_tokens()) {
      EXPECT_EQ(allowed, static_cast<uint32_t>(kDecodeBudget));
    }
    if (requests.size() == 2 && expected_sequences == 1) {
      const size_t scheduled_index =
          requests[0]->sequences().front().get() == batches.front()[0] ? 0 : 1;
      requests[scheduled_index]->set_cancel();
      auto next_batches = scheduler.prepare_batch_test();
      ASSERT_EQ(next_batches.size(), 1u);
      ASSERT_EQ(next_batches.front().size(), 1u);
      Sequence* pending =
          requests[1 - scheduled_index]->sequences().front().get();
      EXPECT_EQ(next_batches.front()[0], pending);
      EXPECT_EQ(pending->kv_state().num_blocks(BlockType::KV), 17u);
    }
    for (const auto& request : requests) {
      engine.block_manager_pool()->deallocate(request.get());
    }
  }
}

TEST(SchedulerPolicyTest, DecodeQueuePlansColdBestOfCandidatesAsPrefill) {
  ScopedConfigValue<bool> chunked_guard(
      SchedulerConfig::get_instance().enable_chunked_prefill(), true);
  ScopedConfigValue<int32_t> stride_guard(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  for (const int32_t token_budget : {5, 16}) {
    for (const bool prefix_cache : {false, true}) {
      ScopedConfigValue<bool> cache_guard(
          KVCacheConfig::get_instance().enable_prefix_cache(), prefix_cache);
      for (const bool mix_batch : {false, true}) {
        ScopedConfigValue<bool> mix_guard(
            SchedulerConfig::get_instance().enable_mix_batch(), mix_batch);
        for (const int32_t speculative_tokens : {0, 2}) {
          SCOPED_TRACE(::testing::Message()
                       << "token_budget=" << token_budget << ", prefix_cache="
                       << prefix_cache << ", mix_batch=" << mix_batch
                       << ", speculative_tokens=" << speculative_tokens);
          FakeEngine engine(64, 2, prefix_cache, false, speculative_tokens);
          auto options = create_scheduler_options(
              token_budget, 8, speculative_tokens, 4, 1);
          options.enable_schedule_overlap(false);
          ContinuousScheduler scheduler(&engine, options);
          auto requests =
              generate_request({7}, {16}, std::nullopt, std::nullopt, 128);
          RequestState request_state = requests.front()->state().clone();
          request_state.best_of = 2;
          auto request = std::make_shared<Request>(
              "best-of", "best-of", "", std::move(request_state), "");
          Sequence* first = request->sequences().front().get();
          ASSERT_TRUE(engine.block_manager_pool()->allocate(first, 7));
          std::vector<Batch> prefill(1);
          prefill.front().add(first, 7);
          first->kv_state().set_kv_cache_tokens_num(7);
          first->append_token(1);
          scheduler.add_request(request);
          auto batches = scheduler.prepare_batch_test();
          ASSERT_EQ(request->sequences().size(), 2u);
          ASSERT_EQ(batches.size(), 1u);
          ASSERT_EQ(batches.front().size(), 2u);
          Sequence* candidate = request->sequences().back().get();
          EXPECT_TRUE(candidate->is_prefill_stage());
          const size_t expected_cached_tokens = prefix_cache ? 6 : 0;
          EXPECT_EQ(candidate->kv_cache_tokens_num(), expected_cached_tokens);
          const size_t remaining_tokens = 7 - expected_cached_tokens;
          const size_t expected_prefill_tokens = std::min(
              static_cast<size_t>(token_budget - 1 - speculative_tokens),
              mix_batch && speculative_tokens == 0
                  ? remaining_tokens
                  : std::min(remaining_tokens, size_t{4}));
          EXPECT_EQ(
              candidate->kv_state().current_max_tokens_capacity(),
              (expected_cached_tokens + expected_prefill_tokens + 1) / 2 * 2);
          EXPECT_EQ(batches.front().get_allowed_max_tokens().back(),
                    expected_prefill_tokens);
          EXPECT_EQ(batches.front().get_allowed_max_tokens().front(),
                    static_cast<size_t>(1 + speculative_tokens));
          engine.block_manager_pool()->deallocate(request.get());
        }
      }
    }
  }
}

TEST(SchedulerPolicyTest, LinearLatencySchedulingRespectsRemainingTokenBudget) {
  ScopedConfigValue<bool> chunked_guard(
      SchedulerConfig::get_instance().enable_chunked_prefill(), true);
  ScopedConfigValue<int32_t> stride_guard(
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 256);
  for (const int32_t token_budget : {288, 384}) {
    for (const bool prefix_cache : {false, true}) {
      ScopedConfigValue<bool> cache_guard(
          KVCacheConfig::get_instance().enable_prefix_cache(), prefix_cache);
      for (const bool mix_batch : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "token_budget=" << token_budget
                                          << ", prefix_cache=" << prefix_cache
                                          << ", mix_batch=" << mix_batch);
        ScopedConfigValue<bool> mix_guard(
            SchedulerConfig::get_instance().enable_mix_batch(), mix_batch);
        ContinuousScheduler::Options options =
            create_scheduler_options(token_budget,
                                     /*max_seqs_per_batch=*/16,
                                     /*num_speculative_tokens=*/0,
                                     /*max_tokens_per_chunk_for_prefill=*/256,
                                     /*dp_size=*/1,
                                     /*priority_strategy=*/"multi_slo_and_prio",
                                     /*enable_profile_kv_blocks=*/false,
                                     /*enable_latency_aware_schedule=*/true,
                                     /*max_global_ttft_ms=*/1000000,
                                     /*max_global_tpot_ms=*/1000000);
        FakeEngine engine(128, 128, prefix_cache, true);
        ContinuousScheduler scheduler(&engine, options);
        scheduler.get_profile_manager()->train_prefill_time_predictor(
            std::vector<std::pair<int32_t, double>>{
                {64, 8}, {128, 24}, {256, 80}, {512, 288}});
        auto requests = generate_request(
            {777, 777}, {1, 1}, std::nullopt, std::nullopt, 10000);
        for (auto& request : requests) {
          scheduler.add_request(request);
        }

        auto batches = scheduler.prepare_batch_test();

        ASSERT_EQ(batches.size(), 1u);
        ASSERT_EQ(batches.front().size(), 2u);
        EXPECT_EQ(batches.front().get_allowed_max_tokens(),
                  (std::vector<uint32_t>{
                      256, static_cast<uint32_t>(token_budget - 256)}));
        EXPECT_EQ(scheduler.get_running_requests().size(), 2u);
        for (size_t sequence_index = 0; sequence_index < batches.front().size();
             ++sequence_index) {
          batches.front()
              .get_sequences()[sequence_index]
              ->kv_state()
              .set_kv_cache_tokens_num(
                  batches.front().get_allowed_max_tokens()[sequence_index]);
        }
        batches = scheduler.prepare_batch_test();
        ASSERT_EQ(batches.size(), 1u);
        ASSERT_EQ(batches.front().size(), 2u);
        const std::vector<uint32_t> expected_second_step =
            prefix_cache
                ? (token_budget == 288 ? std::vector<uint32_t>{224, 64}
                                       : std::vector<uint32_t>{128, 256})
                : std::vector<uint32_t>{
                      256, static_cast<uint32_t>(token_budget - 256)};
        EXPECT_EQ(batches.front().get_allowed_max_tokens(),
                  expected_second_step);
      }
    }
  }
}

TEST(SchedulerPolicyTest, KvlessCompositeReprobesAfterPartialAllocation) {
  ContinuousScheduler::Options options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1,
      /*priority_strategy=*/"multi_slo_and_prio");
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "multi_slo_and_prio",
  };
  TestUnifiedPolicy policy(mode, options);
  TrackingBlockManagerPool block_manager_pool;

  std::vector<std::shared_ptr<Request>> requests = generate_request(
      {32769}, {1}, std::nullopt, std::nullopt, /*max_context_len=*/40000);
  Sequence* sequence = requests.front()->sequences().front().get();
  sequence->kv_state().add_blocks(BlockType::C4, std::vector<Block>(1));
  sequence->kv_state().set_prefix_cache_matched();
  ASSERT_TRUE(sequence->kv_state().has_any_blocks());
  ASSERT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 0u);

  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  DecodeAdmission decode_admission(&block_manager_pool,
                                   DecodeAdmission::Options{});
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .response_processor = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
      .decode_admission = &decode_admission,
  };

  policy.allocate_shared_blocks_for_test(sequence, state);

  EXPECT_EQ(block_manager_pool.allocate_shared_calls(), 1);
}

TEST(SchedulerPolicyTest, UnifiedRetryRefreshesHostRestoreBeforeChunkSizing) {
  constexpr size_t kPromptTokens = 34025;
  constexpr size_t kRestoreTokens = 32768;
  ContinuousScheduler::Options options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1,
      /*priority_strategy=*/"multi_slo_and_prio");
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "multi_slo_and_prio",
  };
  UnifiedPolicy policy(mode, options);
  RetryRestoreBlockManagerPool block_manager_pool;
  auto profile_engine = std::make_unique<FakeEngine>(512, 128);
  ProfileManager::Options profile_options;
  profile_options.max_tokens_per_batch(16384).max_seqs_per_batch(16);
  ProfileManager profile_manager(profile_engine.get(), profile_options);

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({static_cast<int32_t>(kPromptTokens),
                        static_cast<int32_t>(kPromptTokens)},
                       {1, 1},
                       std::nullopt,
                       std::vector<int32_t>{1, 3},
                       /*max_context_len=*/40000);
  requests[1]->sequences().front()->kv_state().set_kv_cache_tokens_num(1);
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue{requests[0], requests[1]};
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  DecodeAdmission decode_admission(&block_manager_pool,
                                   DecodeAdmission::Options{});
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = &profile_manager,
      .response_processor = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
      .decode_admission = &decode_admission,
  };
  ScheduleBudget budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, budget, finished);

  ASSERT_EQ(block_manager_pool.allocate_targets().size(), 2u);
  EXPECT_EQ(block_manager_pool.allocate_targets()[0], kPromptTokens);
  EXPECT_EQ(block_manager_pool.allocate_targets()[1], kPromptTokens);
  EXPECT_GE(block_manager_pool.allocate_shared_calls(), 4);
  ASSERT_EQ(running_sequences.size(), 1u);
  EXPECT_EQ(running_sequences.front()->kv_cache_tokens_num(), kRestoreTokens);
  EXPECT_EQ(running_sequence_budgets,
            (std::vector<size_t>{kPromptTokens - kRestoreTokens}));
  EXPECT_EQ(budget.num_preempted_requests, 1u);
  EXPECT_TRUE(finished.empty());
}

TEST(SchedulerPolicyTest, DecodeRestoreRetrySizesChunkAfterHostMatch) {
  ScopedConfigValue<int32_t> match_frequency(
      SchedulerConfig::get_instance().chunked_match_frequency(), 2);
  ScopedConfigValue<bool> in_batch_cache(
      KVCacheConfig::get_instance().enable_in_batch_prefix_cache(), false);
  constexpr size_t kHbmTokens = 8192;
  constexpr size_t kRestoreTokens = 32768;
  constexpr size_t kChunkTokens = 4096;
  ContinuousScheduler::Options options = create_scheduler_options(
      /*max_tokens_per_batch=*/kChunkTokens,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/kChunkTokens,
      /*dp_size=*/1);
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  RetryRestoreBlockManagerPool block_manager_pool;
  // Declared after the pool: the policy keeps restored requests alive.
  DecodeFirstPolicy policy(mode, options);
  std::vector<std::shared_ptr<Request>> requests = generate_request(
      {40001}, {1}, std::nullopt, std::nullopt, /*max_context_len=*/50000);
  Sequence* sequence = requests.front()->sequences().front().get();
  ASSERT_TRUE(
      block_manager_pool.BlockManagerPool::allocate(sequence, kHbmTokens));
  Batch hbm_batch(sequence);
  sequence->kv_state().set_kv_cache_tokens_num(kHbmTokens);
  sequence->kv_state().set_prefix_cache_matched();
  ASSERT_GT(sequence->kv_state().num_blocks(BlockType::KV), 0u);
  block_manager_pool.allocate_shared(sequence);

  // The retained HBM prefix is at chunk 2, outside the regular match interval
  // of 5 chunks. A released Host match must still be refreshed on retry.
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting{
      DecodeRestoreEntry{requests.front(), absl::Now()}};
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  DecodeAdmission decode_admission(&block_manager_pool,
                                   DecodeAdmission::Options{});
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .response_processor = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
      .decode_admission = &decode_admission,
  };
  ScheduleBudget budget{
      .remaining_token_budget = kChunkTokens,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, budget, finished);

  ASSERT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_TRUE(running_sequences.empty());
  EXPECT_EQ(sequence->kv_cache_tokens_num(), kHbmTokens);
  EXPECT_FALSE(sequence->host_kv_state().prefix_cache_matched());
  EXPECT_EQ(budget.remaining_token_budget, kChunkTokens);

  policy.schedule(state, budget, finished);

  EXPECT_EQ(block_manager_pool.allocate_targets(),
            (std::vector<size_t>{kRestoreTokens + kChunkTokens,
                                 kRestoreTokens + kChunkTokens}));
  EXPECT_TRUE(decode_restore_waiting.empty());
  ASSERT_EQ(running_sequences.size(), 1u);
  EXPECT_EQ(running_sequences.front(), sequence);
  EXPECT_EQ(sequence->kv_cache_tokens_num(), kRestoreTokens);
  EXPECT_EQ(running_sequence_budgets, (std::vector<size_t>{kChunkTokens}));
  EXPECT_EQ(budget.remaining_token_budget, 0u);
  EXPECT_TRUE(finished.empty());
}

class AsyncBlockReleasePolicyTest
    : public ::testing::TestWithParam<std::tuple<int32_t, bool, int32_t>> {};

TEST_P(AsyncBlockReleasePolicyTest, WaitsWithoutPreemptionThenRetries) {
  const auto [policy_kind, prefill, request_count] = GetParam();
  const std::string strategy = policy_kind == 2 ? "multi_slo_and_prio" : "fcfs";
  ContinuousScheduler::Options options =
      create_scheduler_options(1024, 16, 0, 512, 1, strategy);
  options.enable_schedule_overlap(true).enable_chunked_prefill(true);
  BatchMode mode{
      .enable_mix_batch = policy_kind != 0,
      .enable_chunked_prefill = true,
      .priority_strategy = strategy,
  };
  auto policy = create_scheduler_policy(mode, options);
  RestoreWaitingBlockManagerPool block_manager_pool;
  FakeEngine profile_engine(256, 128);
  ProfileManager::Options profile_options;
  profile_options.max_tokens_per_batch(1024).max_seqs_per_batch(16);
  ProfileManager profile_manager(&profile_engine, profile_options);
  auto requests = generate_request(std::vector<int32_t>(request_count, 128),
                                   std::vector<int32_t>(request_count, 16),
                                   std::nullopt,
                                   std::nullopt,
                                   1024);
  if (!prefill) {
    for (const auto& request : requests) {
      Sequence* sequence = request->sequences().front().get();
      sequence->kv_state().set_kv_cache_tokens_num(
          sequence->num_prompt_tokens());
      sequence->append_token(1);
    }
  }
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue;
  for (const auto& request : requests) {
    if (policy_kind == 2) {
      unified_queue.emplace_back(request);
    } else {
      (prefill ? prefill_queue : decode_queue).push(request);
    }
  }
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  model_args.layer_types({"linear_attention"});
  DecodeAdmission decode_admission(&block_manager_pool,
                                   DecodeAdmission::Options{});
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = &profile_manager,
      .response_processor = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = false,
      .decode_admission = &decode_admission,
  };
  ScheduleBudget budget{
      .remaining_token_budget = 1024,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy->schedule(state, budget, finished);

  EXPECT_TRUE(running_sequences.empty());
  EXPECT_TRUE(finished.empty());
  EXPECT_EQ(budget.num_preempted_requests, 0u);
  EXPECT_EQ(prefill_queue.size() + decode_queue.size() + unified_queue.size(),
            static_cast<size_t>(request_count));
  for (const auto& request : requests) {
    EXPECT_FALSE(request->preempted());
  }

  block_manager_pool.set_pending_async_release(false);
  policy->schedule(state, budget, finished);

  EXPECT_EQ(running_sequences.size(), static_cast<size_t>(request_count));
  EXPECT_EQ(budget.num_preempted_requests, 0u);
  EXPECT_TRUE(finished.empty());
}

INSTANTIATE_TEST_SUITE_P(AllPolicies,
                         AsyncBlockReleasePolicyTest,
                         ::testing::Combine(::testing::Values(0, 1, 2),
                                            ::testing::Bool(),
                                            ::testing::Values(1, 2)));

TEST(SchedulerPolicyTest, DefersWhileAsyncBlockReleaseIsPending) {
  ContinuousScheduler::Options options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1,
      /*priority_strategy=*/"multi_slo_and_prio");
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "multi_slo_and_prio",
  };
  UnifiedPolicy policy(mode, options);
  PendingReleaseBlockManagerPool block_manager_pool;
  auto profile_engine = std::make_unique<FakeEngine>(256, 128);
  ProfileManager::Options profile_options;
  profile_options.max_tokens_per_batch(16384).max_seqs_per_batch(16);
  ProfileManager profile_manager(profile_engine.get(), profile_options);

  std::vector<std::shared_ptr<Request>> requests = generate_request(
      {34025}, {2}, std::nullopt, std::nullopt, /*max_context_len=*/40000);
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue{requests.front()};
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  DecodeAdmission decode_admission(&block_manager_pool,
                                   DecodeAdmission::Options{});
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = &profile_manager,
      .response_processor = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
      .decode_admission = &decode_admission,
  };
  ScheduleBudget budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, budget, finished);

  EXPECT_EQ(block_manager_pool.allocate_calls(), 1);
  EXPECT_EQ(unified_queue.size(), 1u);
  EXPECT_TRUE(running_sequences.empty());
  EXPECT_TRUE(finished.empty());
}

TEST(SchedulerPolicyTest,
     RestoreWaitingDefersUntilAsyncReleaseThenSchedulesPrefill) {
  constexpr int32_t kPromptTokens = 128;
  constexpr int32_t kHostRestoreTokens = 256;
  ContinuousScheduler::Options options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1);
  BatchMode mode{
      .enable_mix_batch = false,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  RestoreWaitingBlockManagerPool block_manager_pool;
  BlockManagerPool::Options prefix_cache_options;
  prefix_cache_options.num_blocks_ = 4;
  prefix_cache_options.block_size_ = 128;
  prefix_cache_options.max_seqs_per_batch_ = 16;
  BlockManagerPool prefix_cache_pool(prefix_cache_options, /*dp_size=*/1);
  PrefillFirstPolicy policy(mode, options);

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({kPromptTokens},
                       {kPromptTokens + 1},
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/40000);
  std::vector<std::shared_ptr<Request>> cached_requests =
      generate_request({kHostRestoreTokens},
                       {1},
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/40000);
  Sequence* sequence = requests.front()->sequences().front().get();
  Sequence* cached_sequence =
      cached_requests.front()->sequences().front().get();
  sequence->kv_state().set_kv_cache_tokens_num(kPromptTokens);
  for (int32_t token_id = 0; token_id <= kPromptTokens; ++token_id) {
    sequence->append_token(token_id);
  }
  sequence->kv_state().set_kv_cache_tokens_num(0);
  ASSERT_TRUE(prefix_cache_pool.allocate(cached_sequence));
  sequence->host_kv_state().mount_composite_shared(
      BlockType::KV, cached_sequence->kv_state().take_blocks(BlockType::KV));
  sequence->host_kv_state().set_kv_cache_tokens_num(kHostRestoreTokens);
  sequence->host_kv_state().set_prefix_cache_matched();
  sequence->set_host_cache_match(/*restore_tokens=*/kHostRestoreTokens,
                                 /*copy_units=*/2);
  EXPECT_EQ(requests.front()->num_prefix_cache_tokens(), 0u);
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting{
      {requests.front(), absl::Now()}};
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  DecodeAdmission decode_admission(&block_manager_pool,
                                   DecodeAdmission::Options{});
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .response_processor = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
      .decode_admission = &decode_admission,
  };
  ScheduleBudget budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, budget, finished);

  EXPECT_EQ(block_manager_pool.allocate_calls(), 0);
  EXPECT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_TRUE(running_sequences.empty());

  block_manager_pool.set_pending_async_release(false);
  policy.schedule(state, budget, finished);

  EXPECT_EQ(block_manager_pool.allocate_shared_calls(), 1);
  EXPECT_EQ(block_manager_pool.allocate_calls(), 1);
  EXPECT_TRUE(decode_restore_waiting.empty());
  ASSERT_EQ(running_sequences.size(), 1u);
  EXPECT_EQ(running_sequences.front(),
            requests.front()->sequences().front().get());
  EXPECT_EQ(requests.front()->num_prefix_cache_tokens(), 0u);
  EXPECT_LE(requests.front()->num_prefix_cache_tokens(),
            sequence->num_prompt_tokens());
  EXPECT_TRUE(finished.empty());
}

TEST(SchedulerPolicyTest, PendingDecodeReleaseStopsFurtherPreemption) {
  ScopedConfigValue<double> host_blocks_factor(
      KVCacheStoreConfig::get_instance().host_blocks_factor(), 2.0);
  ContinuousScheduler::Options options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1);
  options.enable_disagg_pd() = true;
  options.enable_pd_ooc() = false;
  options.instance_role() = InstanceRole::DECODE;
  options.enable_schedule_overlap() = false;
  BatchMode mode{
      .enable_mix_batch = false,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  PrefillFirstPolicy policy(mode, options);
  DecodeVictimBlockManagerPool block_manager_pool;

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({128, 128, 128},
                       {2, 2, 2},
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/40000);
  for (const std::shared_ptr<Request>& request : requests) {
    Sequence* sequence = request->sequences().front().get();
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
    sequence->append_token(Token(1));
  }
  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  decode_queue.push(requests[0], /*if_back=*/true);
  decode_queue.push(requests[1], /*if_back=*/true);
  decode_queue.push(requests[2], /*if_back=*/true);
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  DecodeAdmission decode_admission(&block_manager_pool,
                                   DecodeAdmission::Options{});
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .response_processor = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
      .decode_admission = &decode_admission,
  };
  ScheduleBudget budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, budget, finished);

  EXPECT_EQ(decode_restore_waiting.size(), 1u);
  ASSERT_FALSE(decode_restore_waiting.empty());
  EXPECT_EQ(decode_restore_waiting.front().request.get(), requests[2].get());
  EXPECT_TRUE(requests[2]->preempted());
  EXPECT_TRUE(requests[2]->sequences().front()->is_prefill_stage());
  EXPECT_FALSE(requests[1]->preempted());
  EXPECT_EQ(decode_queue.size(), 2u);
  ASSERT_FALSE(decode_queue.empty());
  EXPECT_EQ(decode_queue.top().get(), requests[0].get());
  EXPECT_EQ(decode_queue.back().get(), requests[1].get());
  EXPECT_EQ(budget.num_preempted_requests, 1u);
  EXPECT_TRUE(prefill_queue.empty());
  EXPECT_TRUE(running_sequences.empty());
  EXPECT_TRUE(finished.empty());

  ScheduleBudget pending_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  policy.schedule(state, pending_budget, finished);

  EXPECT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_FALSE(requests[1]->preempted());
  EXPECT_EQ(decode_queue.size(), 2u);
  EXPECT_EQ(decode_queue.back().get(), requests[1].get());
  EXPECT_EQ(pending_budget.num_preempted_requests, 0u);

  block_manager_pool.complete_async_release();
  ScheduleBudget released_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  policy.schedule(state, released_budget, finished);

  EXPECT_EQ(decode_restore_waiting.size(), 2u);
  EXPECT_TRUE(requests[1]->preempted());
  EXPECT_EQ(decode_queue.size(), 1u);
  EXPECT_EQ(released_budget.num_preempted_requests, 1u);
}

TEST(SchedulerPolicyTest, RetriesBlockedDecodeBeforeRestoringVictim) {
  ScopedConfigValue<double> host_blocks_factor(
      KVCacheStoreConfig::get_instance().host_blocks_factor(), 2.0);
  ContinuousScheduler::Options options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1);
  options.enable_disagg_pd() = true;
  options.enable_pd_ooc() = false;
  options.instance_role() = InstanceRole::DECODE;
  options.enable_schedule_overlap() = false;
  BatchMode mode{
      .enable_mix_batch = false,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  PrefillFirstPolicy policy(mode, options);
  RestorePriorityBlockManagerPool block_manager_pool;

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({128, 128},
                       {2, 2},
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/40000);
  Sequence* blocked = requests[0]->sequences().front().get();
  Sequence* victim = requests[1]->sequences().front().get();
  for (const std::shared_ptr<Request>& request : requests) {
    Sequence* sequence = request->sequences().front().get();
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
    sequence->append_token(Token(1));
  }
  block_manager_pool.set_sequences(blocked, victim);

  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  decode_queue.push(requests[0], /*if_back=*/true);
  decode_queue.push(requests[1], /*if_back=*/true);
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  DecodeAdmission decode_admission(&block_manager_pool,
                                   DecodeAdmission::Options{});
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .response_processor = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
      .decode_admission = &decode_admission,
  };
  ScheduleBudget initial_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, initial_budget, finished);

  ASSERT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_EQ(decode_restore_waiting.front().request.get(), requests[1].get());
  EXPECT_EQ(decode_queue.size(), 1u);
  EXPECT_EQ(decode_queue.top().get(), requests[0].get());
  EXPECT_TRUE(running_sequences.empty());

  block_manager_pool.complete_release();
  ScheduleBudget released_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };

  policy.schedule(state, released_budget, finished);

  ASSERT_EQ(running_sequences.size(), 1u);
  EXPECT_EQ(running_sequences.front(), blocked);
  ASSERT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_EQ(decode_restore_waiting.front().request.get(), requests[1].get());
  EXPECT_TRUE(decode_queue.empty());
}

TEST(SchedulerPolicyTest,
     DecodeFirstRetriesBlockedDecodeBeforeRestoringVictim) {
  ScopedConfigValue<double> host_blocks_factor(
      KVCacheStoreConfig::get_instance().host_blocks_factor(), 2.0);
  ContinuousScheduler::Options options = create_scheduler_options(
      /*max_tokens_per_batch=*/16384,
      /*max_seqs_per_batch=*/16,
      /*num_speculative_tokens=*/0,
      /*max_tokens_per_chunk_for_prefill=*/16384,
      /*dp_size=*/1);
  options.enable_disagg_pd() = true;
  options.enable_pd_ooc() = false;
  options.instance_role() = InstanceRole::DECODE;
  options.enable_schedule_overlap() = false;
  BatchMode mode{
      .enable_mix_batch = true,
      .enable_chunked_prefill = true,
      .priority_strategy = "fcfs",
  };
  DecodeFirstPolicy policy(mode, options);
  RestorePriorityBlockManagerPool block_manager_pool;

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({128, 128},
                       {2, 2},
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/40000);
  Sequence* blocked = requests[0]->sequences().front().get();
  Sequence* victim = requests[1]->sequences().front().get();
  for (const std::shared_ptr<Request>& request : requests) {
    Sequence* sequence = request->sequences().front().get();
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
    sequence->append_token(Token(1));
  }
  block_manager_pool.set_sequences(blocked, victim);

  DequeQueue prefill_queue;
  DequeQueue chunk_queue;
  DequeQueue decode_queue;
  decode_queue.push(requests[0], /*if_back=*/true);
  decode_queue.push(requests[1], /*if_back=*/true);
  std::list<std::shared_ptr<Request>> unified_queue;
  std::deque<DecodeRestoreEntry> decode_restore_waiting;
  std::vector<std::shared_ptr<Request>> running_requests;
  std::vector<Sequence*> running_sequences;
  std::vector<size_t> running_sequence_budgets;
  bool last_step_prefill = false;
  ModelArgs model_args;
  DecodeAdmission decode_admission(&block_manager_pool,
                                   DecodeAdmission::Options{});
  SchedulerState state{
      .prefill_queue = prefill_queue,
      .chunk_queue = chunk_queue,
      .decode_queue = decode_queue,
      .unified_queue = unified_queue,
      .decode_restore_waiting = decode_restore_waiting,
      .running_requests = running_requests,
      .running_sequences = running_sequences,
      .running_sequences_budgets = running_sequence_budgets,
      .kv_cache_manager = &block_manager_pool,
      .profile_manager = nullptr,
      .response_processor = nullptr,
      .model_args = model_args,
      .last_step_prefill = last_step_prefill,
      .options = options,
      .min_speculative_tokens_required = 0,
      .enable_prefix_cache = true,
      .decode_admission = &decode_admission,
  };
  ScheduleBudget initial_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };
  std::vector<std::shared_ptr<Request>> finished;

  policy.schedule(state, initial_budget, finished);

  ASSERT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_EQ(decode_restore_waiting.front().request.get(), requests[1].get());
  EXPECT_EQ(decode_queue.size(), 1u);
  EXPECT_EQ(decode_queue.top().get(), requests[0].get());
  EXPECT_TRUE(running_sequences.empty());

  block_manager_pool.complete_release();
  ScheduleBudget released_budget{
      .remaining_token_budget = 16384,
      .remaining_seq_budget = 16,
      .latency_budget = std::numeric_limits<double>::max(),
      .estimate_latency = 0,
      .num_preempted_requests = 0,
  };

  policy.schedule(state, released_budget, finished);

  ASSERT_EQ(running_sequences.size(), 1u);
  EXPECT_EQ(running_sequences.front(), blocked);
  ASSERT_EQ(decode_restore_waiting.size(), 1u);
  EXPECT_EQ(decode_restore_waiting.front().request.get(), requests[1].get());
  EXPECT_TRUE(decode_queue.empty());
}

// TEST-2:
// memory or budget not enough
TEST(SchedulerPolicyTest, ResourceNotEnough) {
  // case1: max tokens budget not enough
  {
    // max token budget: 1
    ContinuousScheduler::Options opt =
        create_scheduler_options(1, 256, 0, 1024, 1);
    auto engine = std::make_unique<FakeEngine>(16, 16);
    auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
    EXPECT_TRUE(scheduler != nullptr);

    // request prompt len: 100
    auto requests =
        generate_request({1, 100}, {1, 10}, std::nullopt, std::nullopt, 10000);
    scheduler->add_request(requests[0]);
    scheduler->add_request(requests[1]);

    auto batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 1);  // can not schedule the second.
  }

  // case2: blocks memory not enough
  {
    ContinuousScheduler::Options opt =
        create_scheduler_options(1000, 256, 0, 1024, 1);
    // free block slot: 1
    auto engine = std::make_unique<FakeEngine>(2, 8);
    auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
    EXPECT_TRUE(scheduler != nullptr);

    // request prompt len: 1000
    auto requests = generate_request(
        {1, 1000}, {1, 100}, std::nullopt, std::nullopt, 10000);
    scheduler->add_request(requests[0]);
    scheduler->add_request(requests[1]);

    auto batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 1);  // can not schedule the second.
  }
}

// TEST-3:
// schedule decoding requests + some prefill requests
TEST(SchedulerPolicyTest, NormalSchedule) {
  // set max free blocks: 1024, support 1024*32=32768 tokens
  // (needs to be large enough to accommodate full-footprint reservation of
  // multiple in-flight chunked prefills plus decode blocks with margin)
  int block_num = 1024;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  // 1. schedule some new prefill requests
  auto requests = generate_request(
      {100, 200, 300}, {10, 20, 30}, std::nullopt, std::nullopt, 30000);
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto total_blocks =
      util::max(engine->block_manager_pool()->num_free_blocks());
  auto batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 3);
  auto seq_use_blocks =
      (100 / block_size + 1) + (200 / block_size + 1) + (300 / block_size + 1);
  EXPECT_TRUE(util::max(engine->block_manager_pool()->num_free_blocks()) ==
              (total_blocks - seq_use_blocks));
  update_requests(requests);

  // 2. schedule decoding requests
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 3);
  update_requests(requests);

  // 3. add new prefill requests
  auto requests1 =
      generate_request({400, 500}, {40, 50}, std::nullopt, std::nullopt, 30000);
  for (auto req : requests1) {
    scheduler->add_request(req);
  }
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 5);
  update_requests(requests);
  update_requests(requests1);

  // 4. add long new prefill requests,
  // memory is enough, can handled tokens from
  // max_tokens_per_chunk_for_prefill(1024) -> remain_max_tokens_budget
  auto requests2 =
      generate_request({10000}, {10}, std::nullopt, std::nullopt, 30000);
  for (auto req : requests2) {
    scheduler->add_request(req);
  }
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 6);
  const std::vector<uint32_t>& allowed_max_tokens =
      batch[0].get_allowed_max_tokens();
  // memory is enough, can handled tokens from 1024 -> remain_max_tokens_budget
  EXPECT_TRUE(allowed_max_tokens[5] == 10000 - 5);
  update_requests(requests);
  update_requests(requests1);
  update_requests(requests2);

  // 5. add long new prefill requests,
  // memory is not enough, only handled `max_tokens_per_chunk_for_prefill`(1024)
  // tokens
  auto requests3 =
      generate_request({10000}, {10}, std::nullopt, std::nullopt, 30000);
  for (auto req : requests3) {
    scheduler->add_request(req);
  }
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 7);
  const std::vector<uint32_t>& allowed_max_tokens1 =
      batch[0].get_allowed_max_tokens();
  // only can handle max_tokens_per_chunk_for_prefill tokens (block allocation
  // limits actual allocation even though admission gate passes).
  EXPECT_TRUE(allowed_max_tokens1[6] >= 1024);
}

// TEST-4:
// test preempt
TEST(SchedulerPolicyTest, PreemptSchedule) {
  // set max free blocks: 9, support 9*32=288 tokens
  // actually only 8 free blocks , because default 1 block is for padding
  int block_num = 9;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1. schedule some new prefill requests
  // request-1 has higher priority than request-2
  auto requests =
      generate_request({127, 127}, {10, 10}, std::nullopt, std::nullopt, 30000);
  running_requests = requests;
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  update_requests(running_requests);

  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  update_requests(running_requests);

  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  int free_blocks_before_preempt =
      util::max(block_manager_pool->num_free_blocks());
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 1);
  int free_blocks_after_preempt =
      util::max(block_manager_pool->num_free_blocks());
  EXPECT_TRUE(free_blocks_after_preempt > free_blocks_before_preempt);
  EXPECT_TRUE(scheduler->get_waiting_requests_num() == 1);
  // append a new block
  block_manager_pool->allocate(batch[0][0]);
  // remove preempted request from running_requests
  running_requests.pop_back();
  update_requests(running_requests);

  // 2.The preempted request was saved in waiting_priority_queue_,
  // it just like a new prefill request.
  // Continue to run, the new prefill request will not preempt
  // request in running_queue.
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 1);
}

// TEST-5:
// test priority schedule
TEST(SchedulerPolicyTest, PrioritySchedule) {
  // set max free blocks: 12
  // actually only 11 free blocks , because default 1 block is for padding
  int block_num = 12;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1, "priority");
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1: HIGH, 2: NORMAL, 3: LOW
  auto requests = generate_request({127, 127, 127},
                                   {10, 10, 10},
                                   std::vector<bool>{false, false, false},
                                   std::vector<int32_t>{3, 3, 2},
                                   30000);
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->priority() ==
              RequestPriority::NORMAL /*NORMAL*/);
  EXPECT_TRUE(scheduler->get_running_requests()[1]->priority() ==
              RequestPriority::LOW /*LOW*/);

  running_requests = scheduler->get_running_requests();
  update_requests(running_requests);

  // new HIGH priority request arrives, its prefill starts
  auto new_requests = generate_request({32},
                                       {10},
                                       std::vector<bool>{false},
                                       std::vector<int32_t>{1},
                                       30000);  // use 1 blocks
  scheduler->add_request(new_requests[0]);
  batch = scheduler->prepare_batch_test();
  // check there are 3 running requests owing to decode-maximal
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 3);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 3);
  running_requests.push_back(new_requests[0]);
  update_requests(running_requests);
  // preemption happens, only HIGH and NORMAL decode requests
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->priority() ==
              RequestPriority::HIGH /*HIGH*/);
  EXPECT_TRUE(scheduler->get_running_requests()[1]->priority() ==
              RequestPriority::NORMAL /*NORMAL*/);
}

// TEST-6:
// test latency budget
TEST(SchedulerPolicyTest, LatencySchedule) {
  int block_num = 12;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 4;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000,
                               256,
                               0,
                               max_tokens_per_chunk_for_prefill,
                               1,
                               "fcfs",
                               false,
                               true,
                               350,
                               150);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  // manually created profile data for y=0.5x^2+10x
  std::vector<std::pair<int32_t, double>> created_profile_data = {
      {2, 22}, {4, 48}, {6, 78}, {8, 112}};
  auto profile_manager = scheduler->get_profile_manager();
  // fit y=0.5x^2+10x
  profile_manager->train_prefill_time_predictor(created_profile_data);

  auto requests = generate_request(
      {10, 10, 10}, {10, 10, 10}, std::nullopt, std::nullopt, 30000);
  // check if time equation fits well
  EXPECT_TRUE(
      static_cast<int32_t>(std::round(profile_manager->predict_step_time(
          requests[0]->sequences()[0].get(), true, true))) == 150);
  EXPECT_TRUE(static_cast<int32_t>(std::round(
                  profile_manager->predict_step_time(2, 0, true, true))) == 22);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1. three requests enter prefill
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 3);
  EXPECT_EQ(scheduler->get_running_requests().size(), 3);
  running_requests = scheduler->get_running_requests();
  update_requests(running_requests);

  // 2. three requests enter decode, one new request chunked prefill
  auto new_requests = generate_request(
      {10}, {10}, std::nullopt, std::nullopt, 30000);  // use 1 blocks
  scheduler->add_request(new_requests[0]);
  batch = scheduler->prepare_batch_test();
  auto running_sequences_budgets = scheduler->get_running_sequences_budgets();

  EXPECT_EQ(batch.size(), 1);
  // tpot = 150 > 3 * 10 , all requests enter decode. But not enough for extra
  // whole prefill, then do chunked
  EXPECT_EQ(batch[0].size(), 4);
  EXPECT_EQ(scheduler->get_running_requests().size(), 4);
  EXPECT_EQ(running_sequences_budgets.size(), 4);
  EXPECT_EQ(running_sequences_budgets[3], max_tokens_per_chunk_for_prefill);
}

// TEST: Full-footprint admission gate
// When chunked prefill is enabled, a new request whose full footprint exceeds
// the available blocks (total - used) should NOT be admitted.
TEST(SchedulerPolicyTest, FullFootprintAdmissionGate) {
  // total_blocks = 8, block_size = 4.
  // Request A: 8 tokens → footprint = 2 blocks. Small, easily fits.
  // Request B: 32 tokens → footprint = 8 blocks. Needs all blocks.
  // max_tokens_per_chunk = 4 → each chunk = 1 block.
  //
  // A admitted, allocates 1 chunk (1 block used).
  // B check: available = 8 - 1 = 7, B footprint = 8 → 7 < 8 → blocked.
  const int32_t block_num = 9;  // 8 + 1 reserved
  const int32_t block_size = 4;
  const int32_t max_tokens_per_chunk = 4;

  ContinuousScheduler::Options opt = create_scheduler_options(
      /*max_tokens_per_batch=*/100,
      /*max_seqs=*/256,
      /*spec_tokens=*/0,
      max_tokens_per_chunk,
      /*dp_size=*/1);
  opt.enable_chunked_prefill() = true;
  opt.enable_disagg_pd() = true;
  opt.instance_role() = InstanceRole::PREFILL;

  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  ASSERT_NE(scheduler.get(), nullptr);

  // A (8 tokens = 2 blocks footprint), B (32 tokens = 8 blocks footprint).
  auto requests =
      generate_request({8, 32}, {10, 10}, std::nullopt, std::nullopt, 10000);
  scheduler->add_request(requests[0]);
  scheduler->add_request(requests[1]);

  auto batch = scheduler->prepare_batch_test();
  ASSERT_EQ(batch.size(), 1);
  // Only A admitted. B blocked: after A uses 1 block, available=8-1=7 < 8.
  EXPECT_EQ(batch[0].size(), 1);
}

// TEST: Full-footprint gate admits both when capacity is sufficient.
TEST(SchedulerPolicyTest, FullFootprintAdmitsBothWhenFits) {
  // total_blocks = 9, block_size = 4.
  // Request A: 8 tokens → footprint = 2 blocks.
  // Request B: 8 tokens → footprint = 2 blocks.
  // After A: reserved=2, B check: 2+2=4 <= 9 → admitted.
  const int32_t block_num = 9;
  const int32_t block_size = 4;
  const int32_t max_tokens_per_chunk = 4;

  ContinuousScheduler::Options opt = create_scheduler_options(
      /*max_tokens_per_batch=*/100,
      /*max_seqs=*/256,
      /*spec_tokens=*/0,
      max_tokens_per_chunk,
      /*dp_size=*/1);
  opt.enable_chunked_prefill() = true;
  opt.enable_disagg_pd() = true;
  opt.instance_role() = InstanceRole::PREFILL;

  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  ASSERT_NE(scheduler.get(), nullptr);

  auto requests =
      generate_request({8, 8}, {10, 10}, std::nullopt, std::nullopt, 10000);
  scheduler->add_request(requests[0]);
  scheduler->add_request(requests[1]);

  auto batch = scheduler->prepare_batch_test();
  ASSERT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 2);  // Both admitted.
}

namespace {

// Real block accounting on a backend that permits Host restore. Device prefix
// caching is off so requests with identical test tokens never share blocks.
class HostRestoreBlockPool final : public BlockManagerPool {
 public:
  HostRestoreBlockPool(int32_t num_blocks, int32_t dp_size)
      : BlockManagerPool(make_options(num_blocks), dp_size) {}

  bool supports_host_cache_restore() const override { return true; }

  using BlockManagerPool::cache;
  void cache(Sequence* sequence) override {
    ++cache_calls_;
    BlockManagerPool::cache(sequence);
  }

  size_t cache_calls() const { return cache_calls_; }

 private:
  static BlockManagerPool::Options make_options(int32_t num_blocks) {
    BlockManagerPool::Options options;
    options.num_blocks_ = num_blocks;
    options.block_size_ = 16;
    options.max_seqs_per_batch_ = 16;
    options.enable_prefix_cache_ = false;
    return options;
  }

  size_t cache_calls_ = 0;
};

ContinuousScheduler::Options make_restore_options(int32_t token_budget,
                                                  int32_t dp_size) {
  ContinuousScheduler::Options options =
      create_scheduler_options(token_budget,
                               /*max_seqs_per_batch=*/16,
                               /*num_speculative_tokens=*/0,
                               token_budget,
                               dp_size);
  options.enable_disagg_pd() = true;
  options.enable_pd_ooc() = false;
  options.instance_role() = InstanceRole::DECODE;
  options.enable_schedule_overlap() = false;
  return options;
}

// Owns the queues a policy schedules against across steps.
class RestoreHarness final {
 public:
  RestoreHarness(BlockManagerPool* block_manager_pool,
                 const ContinuousScheduler::Options& options)
      : options_(options),
        admission_(block_manager_pool,
                   DecodeAdmission::Options{
                       .dp_size = options.dp_size(),
                       .reserved_tokens_per_request =
                           SchedulerConfig::get_instance()
                               .decode_reserved_tokens_per_request(),
                   }),
        state_{
            .prefill_queue = prefill_queue_,
            .chunk_queue = chunk_queue_,
            .decode_queue = decode_queue_,
            .unified_queue = unified_queue_,
            .decode_restore_waiting = waiting_,
            .running_requests = running_requests_,
            .running_sequences = running_sequences_,
            .running_sequences_budgets = running_budgets_,
            .kv_cache_manager = block_manager_pool,
            .profile_manager = nullptr,
            .response_processor = nullptr,
            .model_args = model_args_,
            .last_step_prefill = last_step_prefill_,
            .options = options_,
            .min_speculative_tokens_required = 0,
            .enable_prefix_cache = true,
            .decode_admission = &admission_,
        } {}

  void schedule(SchedulerPolicy& policy,
                size_t token_budget,
                size_t seq_budget = 16) {
    ScheduleBudget budget{
        .remaining_token_budget = token_budget,
        .remaining_seq_budget = seq_budget,
        .latency_budget = std::numeric_limits<double>::max(),
        .estimate_latency = 0,
        .num_preempted_requests = 0,
    };
    std::vector<std::shared_ptr<Request>> finished;
    policy.schedule(state_, budget, finished);
  }

  // Applies the scheduled step: each sequence caches its budget of tokens.
  void advance() {
    for (size_t i = 0; i < running_sequences_.size(); ++i) {
      running_sequences_[i]->kv_state().incr_kv_cache_tokens_num(
          running_budgets_[i]);
    }
  }

  void add_waiter(std::shared_ptr<Request> request) {
    waiting_.emplace_back(DecodeRestoreEntry{.request = std::move(request),
                                             .started_at = absl::Now()});
  }

  SchedulerState& state() { return state_; }
  DecodeAdmission& admission() { return admission_; }
  DequeQueue& decode_queue() { return decode_queue_; }
  DequeQueue& chunk_queue() { return chunk_queue_; }
  std::deque<DecodeRestoreEntry>& waiting() { return waiting_; }
  const std::vector<Sequence*>& running_sequences() const {
    return running_sequences_;
  }

 private:
  ContinuousScheduler::Options options_;
  DecodeAdmission admission_;
  DequeQueue prefill_queue_;
  DequeQueue chunk_queue_;
  DequeQueue decode_queue_;
  std::list<std::shared_ptr<Request>> unified_queue_;
  std::deque<DecodeRestoreEntry> waiting_;
  std::vector<std::shared_ptr<Request>> running_requests_;
  std::vector<Sequence*> running_sequences_;
  std::vector<size_t> running_budgets_;
  bool last_step_prefill_ = false;
  ModelArgs model_args_;
  SchedulerState state_;
};

// A preempted decode: one generated token, no blocks, waiting for restore.
std::shared_ptr<Request> make_restore_waiter(int32_t prompt_len,
                                             int32_t max_tokens,
                                             int32_t dp_rank) {
  std::shared_ptr<Request> request = generate_request({prompt_len},
                                                      {max_tokens},
                                                      std::nullopt,
                                                      std::nullopt,
                                                      /*max_context_len=*/4096)
                                         .front();
  Sequence* sequence = request->sequences().front().get();
  sequence->kv_state().set_kv_cache_tokens_num(prompt_len);
  sequence->append_token(Token(1));
  sequence->reset();
  sequence->set_dp_rank(dp_rank);
  request->set_preempted();
  return request;
}

// A decode-stage request holding device blocks for its prompt.
std::shared_ptr<Request> make_running_decode(BlockManagerPool& pool,
                                             int32_t prompt_len,
                                             int32_t max_tokens,
                                             int32_t dp_rank) {
  std::shared_ptr<Request> request = generate_request({prompt_len},
                                                      {max_tokens},
                                                      std::nullopt,
                                                      std::nullopt,
                                                      /*max_context_len=*/4096)
                                         .front();
  Sequence* sequence = request->sequences().front().get();
  sequence->set_dp_rank(dp_rank);
  EXPECT_TRUE(pool.BlockManagerPool::allocate(sequence, prompt_len));
  sequence->kv_state().set_kv_cache_tokens_num(prompt_len);
  sequence->append_token(Token(1));
  return request;
}

// Holds `num_blocks` device blocks on `dp_rank` outside the scheduler, like a
// PD reservation whose KV is still in transfer.
std::shared_ptr<Request> hold_blocks(BlockManagerPool& pool,
                                     int32_t num_blocks,
                                     int32_t dp_rank) {
  std::shared_ptr<Request> request = generate_request({num_blocks * 16},
                                                      {1},
                                                      std::nullopt,
                                                      std::nullopt,
                                                      /*max_context_len=*/4096)
                                         .front();
  Sequence* sequence = request->sequences().front().get();
  sequence->set_dp_rank(dp_rank);
  EXPECT_TRUE(pool.BlockManagerPool::allocate(sequence, num_blocks * 16));
  return request;
}

BatchMode decode_first_mode() {
  return BatchMode{.enable_mix_batch = true,
                   .enable_chunked_prefill = true,
                   .priority_strategy = "fcfs"};
}

class HostRestoreConfig final {
 private:
  ScopedConfigValue<double> host_blocks_factor_{
      KVCacheStoreConfig::get_instance().host_blocks_factor(),
      2.0};
  ScopedConfigValue<int32_t> reserved_tokens_{
      SchedulerConfig::get_instance().decode_reserved_tokens_per_request(),
      0};
};

std::shared_ptr<Request> make_pd_decode(BlockManagerPool& pool,
                                        int32_t num_outputs = 1) {
  StoppingChecker stopping;
  stopping.set_max_generated_tokens(64);
  stopping.set_max_context_len(4096);
  stopping.set_ignore_eos(true);
  RequestState state("prompt",
                     std::vector<int32_t>(16, 0),
                     RequestSamplingParam(),
                     SchedulerParam(),
                     stopping,
                     /*seq_capacity=*/4096,
                     /*n=*/num_outputs,
                     /*best_of=*/num_outputs,
                     /*logprobs=*/false,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*skip_special_tokens=*/false,
                     /*enable_schedule_overlap=*/false,
                     nullptr,
                     nullptr);
  auto request = std::make_shared<Request>("decode", "", "", std::move(state));
  Sequence* sequence = request->sequences().front().get();
  sequence->set_dp_rank(0);
  EXPECT_TRUE(pool.BlockManagerPool::allocate(sequence, /*num_tokens=*/16));
  sequence->kv_state().set_kv_cache_tokens_num(16);
  sequence->append_token(Token(1));
  return request;
}

class PdDecodeFcfsTest : public testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  std::unique_ptr<SchedulerPolicy> make_policy(
      const ContinuousScheduler::Options& options) const {
    BatchMode mode{.enable_mix_batch = std::get<0>(GetParam()),
                   .enable_chunked_prefill = true,
                   .priority_strategy = "fcfs"};
    return create_scheduler_policy(mode, options);
  }

  void set_requeue_state(SchedulerState& state) const {
    state.last_step_prefill = std::get<1>(GetParam());
  }
};

}  // namespace

TEST_P(PdDecodeFcfsTest, OlderWaiterIsNotStarvedByRepeatedPdArrivals) {
  HostRestoreConfig config;
  // Exclude memory pressure while arrivals continue until natural completion.
  HostRestoreBlockPool pool(/*num_blocks=*/128, /*dp_size=*/1);
  auto options = make_restore_options(/*token_budget=*/2, /*dp_size=*/1);
  options.max_seqs_per_batch() = 2;
  auto policy = make_policy(options);
  RestoreHarness harness(&pool, options);
  set_requeue_state(harness.state());
  auto active = make_pd_decode(pool);
  auto old_waiter = make_pd_decode(pool);
  harness.state().running_requests.emplace_back(active);
  harness.decode_queue().push(old_waiter);
  folly::MPMCQueue<std::shared_ptr<Request>> incoming(2);

  for (int32_t round = 0; round < 64 && !old_waiter->finished(); ++round) {
    SCOPED_TRACE(round);
    ASSERT_TRUE(incoming.write(make_pd_decode(pool)));
    policy->drain_request_queue(harness.state(), incoming);
    harness.schedule(*policy, /*token_budget=*/2, /*seq_budget=*/2);
    ASSERT_EQ(harness.state().running_requests.size(), 2u);
    EXPECT_NE(std::find(harness.state().running_requests.begin(),
                        harness.state().running_requests.end(),
                        old_waiter),
              harness.state().running_requests.end());
    EXPECT_NE(std::find(harness.state().running_requests.begin(),
                        harness.state().running_requests.end(),
                        active),
              harness.state().running_requests.end());
    EXPECT_EQ(harness.decode_queue().size(), static_cast<size_t>(round + 1));
    update_requests(harness.state().running_requests);
  }

  ASSERT_TRUE(old_waiter->finished());
  ASSERT_TRUE(active->finished());
  EXPECT_EQ(policy->collect_finished(harness.state()).size(), 2u);
  EXPECT_FALSE(old_waiter->sequences().front()->has_any_blocks());
  EXPECT_FALSE(active->sequences().front()->has_any_blocks());
  harness.schedule(*policy, /*token_budget=*/2, /*seq_budget=*/2);
  EXPECT_EQ(std::find(harness.state().running_requests.begin(),
                      harness.state().running_requests.end(),
                      old_waiter),
            harness.state().running_requests.end());
}

TEST_P(PdDecodeFcfsTest, TerminalHandoffsReleaseBlocksAtEveryAdmissionStage) {
  for (bool enable_prefix_cache : {false, true}) {
    // Terminate before drain, while staged, or at the decode queue's tail.
    for (int32_t stage : {0, 1, 2}) {
      SCOPED_TRACE(enable_prefix_cache);
      SCOPED_TRACE(stage);
      HostRestoreBlockPool pool(/*num_blocks=*/16, /*dp_size=*/1);
      auto options = make_restore_options(/*token_budget=*/2, /*dp_size=*/1);
      auto policy = make_policy(options);
      RestoreHarness harness(&pool, options);
      set_requeue_state(harness.state());
      harness.state().enable_prefix_cache = enable_prefix_cache;
      auto live = make_pd_decode(pool);
      auto finished = make_pd_decode(pool);
      auto cancelled = make_pd_decode(pool, /*num_outputs=*/2);
      harness.decode_queue().push(live);
      auto terminate = [&]() {
        finished->sequences().front()->finish();
        cancelled->set_cancel();
      };
      if (stage == 0) {
        terminate();
      }
      folly::MPMCQueue<std::shared_ptr<Request>> incoming(2);
      ASSERT_TRUE(incoming.write(finished));
      ASSERT_TRUE(incoming.write(cancelled));
      policy->drain_request_queue(harness.state(), incoming);
      if (stage == 2) {
        harness.schedule(*policy, /*token_budget=*/0, /*seq_budget=*/0);
        ASSERT_EQ(harness.decode_queue().size(), 3u);
      }
      const size_t sequence_count = cancelled->sequences().size();
      const size_t free_before = pool.num_free_blocks()[0];
      terminate();
      size_t release_calls = 0;
      harness.state().release_failed_request =
          [&release_calls](const std::shared_ptr<Request>& /*request*/) {
            ++release_calls;
          };

      EXPECT_EQ(policy->collect_finished(harness.state()).size(), 2u);
      EXPECT_EQ(release_calls, 2u);
      EXPECT_EQ(pool.num_free_blocks()[0], free_before + 2);
      EXPECT_FALSE(finished->sequences()[0]->has_any_blocks());
      for (const auto& sequence : cancelled->sequences()) {
        EXPECT_FALSE(sequence->has_any_blocks());
      }
      EXPECT_EQ(cancelled->sequences().size(), sequence_count);
      EXPECT_TRUE(policy->collect_finished(harness.state()).empty());
      EXPECT_EQ(release_calls, 2u);
      harness.schedule(*policy, /*token_budget=*/0, /*seq_budget=*/0);
      EXPECT_TRUE(harness.state().running_requests.empty());
      EXPECT_TRUE(harness.state().prefill_queue.empty());
      ASSERT_EQ(harness.decode_queue().size(), 1u);
      EXPECT_EQ(harness.decode_queue().top(), live);
      EXPECT_TRUE(live->sequences()[0]->has_any_blocks());
    }
  }
}

TEST_P(PdDecodeFcfsTest, PartiallyFinishedHandoffPreservesLiveBlocks) {
  for (bool enable_prefix_cache : {false, true}) {
    SCOPED_TRACE(enable_prefix_cache);
    HostRestoreBlockPool pool(/*num_blocks=*/16, /*dp_size=*/1);
    auto options = make_restore_options(/*token_budget=*/1, /*dp_size=*/1);
    auto policy = make_policy(options);
    RestoreHarness harness(&pool, options);
    set_requeue_state(harness.state());
    harness.state().enable_prefix_cache = enable_prefix_cache;
    auto request = make_pd_decode(pool, /*num_outputs=*/2);
    ASSERT_TRUE(request->expand_sequences());
    Sequence* finished = request->sequences()[0].get();
    Sequence* live = request->sequences()[1].get();
    live->set_dp_rank(0);
    ASSERT_TRUE(pool.BlockManagerPool::allocate(live, /*num_tokens=*/16));
    live->kv_state().set_kv_cache_tokens_num(16);
    live->append_token(Token(1));
    const int32_t live_block_id =
        live->kv_state().blocks(BlockType::KV)[0].id();
    const size_t free_before = pool.num_free_blocks()[0];
    finished->finish();
    folly::MPMCQueue<std::shared_ptr<Request>> incoming(2);
    ASSERT_TRUE(incoming.write(request));

    policy->drain_request_queue(harness.state(), incoming);
    EXPECT_TRUE(finished->has_any_blocks());
    EXPECT_TRUE(policy->collect_finished(harness.state()).empty());
    harness.schedule(*policy, /*token_budget=*/0, /*seq_budget=*/0);
    EXPECT_FALSE(finished->has_any_blocks());
    ASSERT_TRUE(live->has_any_blocks());
    EXPECT_EQ(live->kv_state().blocks(BlockType::KV)[0].id(), live_block_id);
    EXPECT_EQ(pool.num_free_blocks()[0], free_before + 1);
    harness.schedule(*policy, /*token_budget=*/1, /*seq_budget=*/1);
    EXPECT_EQ(harness.running_sequences(), (std::vector<Sequence*>{live}));
    live->finish();
    EXPECT_EQ(policy->collect_finished(harness.state()),
              (std::vector<std::shared_ptr<Request>>{request}));
    EXPECT_FALSE(live->has_any_blocks());
  }
}

INSTANTIATE_TEST_SUITE_P(AllBatchModes,
                         PdDecodeFcfsTest,
                         testing::Combine(testing::Bool(), testing::Bool()));

TEST(SchedulerPolicyTest, PdHandoffsExpandOnceAndPreserveNonFcfsAdmission) {
  for (bool mix_batch : {false, true}) {
    for (const std::string strategy : {"fcfs", "priority", "deadline"}) {
      for (bool enable_prefix_cache : {false, true}) {
        SCOPED_TRACE(mix_batch);
        SCOPED_TRACE(strategy);
        SCOPED_TRACE(enable_prefix_cache);
        HostRestoreBlockPool pool(/*num_blocks=*/16, /*dp_size=*/1);
        auto options = make_restore_options(/*token_budget=*/2, /*dp_size=*/1);
        BatchMode mode{.enable_mix_batch = mix_batch,
                       .enable_chunked_prefill = true,
                       .priority_strategy = strategy};
        auto policy = create_scheduler_policy(mode, options);
        RestoreHarness harness(&pool, options);
        harness.state().enable_prefix_cache = enable_prefix_cache;
        auto request = make_pd_decode(pool, /*num_outputs=*/2);
        folly::MPMCQueue<std::shared_ptr<Request>> incoming(2);
        ASSERT_TRUE(incoming.write(request));

        policy->drain_request_queue(harness.state(), incoming);
        EXPECT_TRUE(harness.decode_queue().empty());
        EXPECT_EQ(harness.state().running_requests.size(),
                  strategy == "fcfs" ? 0u : 1u);
        EXPECT_EQ(request->sequences().size(), enable_prefix_cache ? 1u : 2u);
        EXPECT_EQ(pool.cache_calls(), 0u);
        harness.schedule(*policy, /*token_budget=*/0, /*seq_budget=*/0);
        EXPECT_EQ(request->sequences().size(), 2u);
        EXPECT_EQ(pool.cache_calls(), enable_prefix_cache ? 1u : 0u);
        EXPECT_FALSE(request->expand_sequences());
        ASSERT_EQ(harness.decode_queue().size(), 1u);
        EXPECT_EQ(harness.decode_queue().top(), request);
        EXPECT_EQ(request->queue_to_batch_latency_ms(), -1.0);
        harness.schedule(*policy, /*token_budget=*/1, /*seq_budget=*/1);
        const double first_batch_latency = request->queue_to_batch_latency_ms();
        EXPECT_GE(first_batch_latency, 0.0);
        harness.schedule(*policy, /*token_budget=*/2, /*seq_budget=*/2);
        EXPECT_EQ(request->queue_to_batch_latency_ms(), first_batch_latency);
      }
    }
  }
}

TEST(SchedulerPolicyTest, TerminalRequestsBypassExpansionAndAdmission) {
  for (bool mix_batch : {false, true}) {
    for (const std::string strategy :
         {"fcfs", "priority", "deadline", "multi_slo_and_prio"}) {
      for (bool enable_prefix_cache : {false, true}) {
        SCOPED_TRACE(mix_batch);
        SCOPED_TRACE(strategy);
        SCOPED_TRACE(enable_prefix_cache);
        HostRestoreBlockPool pool(/*num_blocks=*/16, /*dp_size=*/1);
        const size_t free_initial = pool.num_free_blocks()[0];
        auto options = make_restore_options(/*token_budget=*/2, /*dp_size=*/1);
        BatchMode mode{.enable_mix_batch = mix_batch,
                       .enable_chunked_prefill = true,
                       .priority_strategy = strategy};
        auto policy = create_scheduler_policy(mode, options);
        RestoreHarness harness(&pool, options);
        harness.state().enable_prefix_cache = enable_prefix_cache;
        auto finished = make_pd_decode(pool);
        auto cancelled = make_pd_decode(pool, /*num_outputs=*/2);
        finished->sequences()[0]->kv_state().set_kv_cache_tokens_num(0);
        cancelled->sequences()[0]->kv_state().set_kv_cache_tokens_num(0);
        finished->sequences()[0]->finish();
        cancelled->set_cancel();
        folly::MPMCQueue<std::shared_ptr<Request>> incoming(2);
        ASSERT_TRUE(incoming.write(finished));
        ASSERT_TRUE(incoming.write(cancelled));
        size_t release_calls = 0;
        harness.state().release_failed_request =
            [&release_calls](const std::shared_ptr<Request>& /*request*/) {
              ++release_calls;
            };

        policy->drain_request_queue(harness.state(), incoming);

        EXPECT_TRUE(harness.state().prefill_queue.empty());
        EXPECT_TRUE(harness.decode_queue().empty());
        EXPECT_TRUE(harness.state().unified_queue.empty());
        EXPECT_EQ(cancelled->sequences().size(), 1u);
        EXPECT_EQ(pool.cache_calls(), 0u);
        EXPECT_EQ(policy->collect_finished(harness.state()).size(), 2u);
        EXPECT_EQ(release_calls, 2u);
        EXPECT_EQ(pool.num_free_blocks()[0], free_initial);
        EXPECT_TRUE(policy->collect_finished(harness.state()).empty());
        EXPECT_EQ(release_calls, 2u);
      }
    }
  }
}

TEST(SchedulerPolicyTest, ChunkedAdmissionPreservesPrefetchDpRank) {
  for (int32_t dp_rank : {0, 1}) {
    SCOPED_TRACE(dp_rank);
    HostRestoreBlockPool pool(/*num_blocks=*/16, /*dp_size=*/2);
    HostRestoreBlockPool host_pool(/*num_blocks=*/16, /*dp_size=*/2);
    ContinuousScheduler::Options options = create_scheduler_options(
        /*max_tokens_per_batch=*/16,
        /*max_seqs_per_batch=*/16,
        /*num_speculative_tokens=*/0,
        /*max_tokens_per_chunk_for_prefill=*/16,
        /*dp_size=*/2);
    options.enable_chunked_prefill() = true;
    options.enable_disagg_pd() = true;
    options.instance_role() = InstanceRole::PREFILL;
    BatchMode mode{.enable_mix_batch = false,
                   .enable_chunked_prefill = true,
                   .priority_strategy = "fcfs"};
    PrefillFirstPolicy policy(mode, options);
    RestoreHarness harness(&pool, options);
    auto held = hold_blocks(pool, /*num_blocks=*/1, dp_rank);
    ASSERT_EQ(pool.select_dp_rank(), 1 - dp_rank);
    auto request =
        generate_request({32}, {1}, std::nullopt, std::nullopt, 4096).front();
    Sequence* sequence = request->sequences().front().get();
    sequence->set_dp_rank(dp_rank);
    auto prefetched = hold_blocks(host_pool, /*num_blocks=*/1, dp_rank);
    const auto host_blocks =
        prefetched->sequences().front()->kv_state().blocks(BlockType::KV);
    sequence->host_kv_state().add_blocks(
        BlockType::KV,
        std::vector<Block>(host_blocks.begin(), host_blocks.end()));
    ASSERT_TRUE(sequence->host_kv_state().has_any_blocks());
    ASSERT_FALSE(sequence->kv_state().has_any_blocks());
    ASSERT_EQ(sequence->kv_state().kv_cache_tokens_num(), 0u);
    harness.state().prefill_queue.push(request);

    harness.schedule(policy, /*token_budget=*/16);

    ASSERT_EQ(harness.running_sequences(), (std::vector<Sequence*>{sequence}));
    EXPECT_EQ(sequence->dp_rank(), dp_rank);
    EXPECT_EQ(pool.num_used_blocks()[dp_rank], 2u);
    EXPECT_EQ(pool.num_used_blocks()[1 - dp_rank], 0u);
    pool.deallocate(held.get());
    pool.deallocate(request.get());
  }
}

TEST(DecodeRestoreTest, BatchesRestoresAlongsideRunningDecodes) {
  HostRestoreConfig config;
  HostRestoreBlockPool pool(/*num_blocks=*/64, /*dp_size=*/1);
  ContinuousScheduler::Options options =
      make_restore_options(/*token_budget=*/256, /*dp_size=*/1);
  DecodeFirstPolicy policy(decode_first_mode(), options);
  RestoreHarness harness(&pool, options);
  auto decode = make_running_decode(pool, 32, 64, /*dp_rank=*/0);
  harness.decode_queue().push(decode);
  std::vector<std::shared_ptr<Request>> waiters;
  for (int32_t i = 0; i < 3; ++i) {
    waiters.emplace_back(make_restore_waiter(32, 8, /*dp_rank=*/0));
    harness.add_waiter(waiters.back());
  }

  harness.schedule(policy, 256);

  EXPECT_EQ(harness.running_sequences(),
            (std::vector<Sequence*>{decode->sequences()[0].get(),
                                    waiters[0]->sequences()[0].get(),
                                    waiters[1]->sequences()[0].get(),
                                    waiters[2]->sequences()[0].get()}));
  EXPECT_TRUE(harness.waiting().empty());
  // Restores are still in prefill stage, so admission stays closed.
  EXPECT_TRUE(harness.admission().paused());

  harness.advance();
  harness.schedule(policy, 256);

  EXPECT_EQ(harness.running_sequences().size(), 4u);
  EXPECT_FALSE(harness.admission().paused());
}

TEST(DecodeRestoreTest, PerRankFifoDoesNotBlockOtherRank) {
  HostRestoreConfig config;
  HostRestoreBlockPool pool(/*num_blocks=*/6, /*dp_size=*/2);
  ContinuousScheduler::Options options =
      make_restore_options(/*token_budget=*/256, /*dp_size=*/2);
  DecodeFirstPolicy policy(decode_first_mode(), options);
  RestoreHarness harness(&pool, options);
  auto decode = make_running_decode(pool, 48, 64, /*dp_rank=*/0);
  harness.decode_queue().push(decode);
  auto blocked = make_restore_waiter(32, 8, /*dp_rank=*/0);
  auto other = make_restore_waiter(32, 8, /*dp_rank=*/1);
  harness.add_waiter(blocked);
  harness.add_waiter(other);

  harness.schedule(policy, 256);

  // Rank 0 has one free block left after the decode grows; the rank-1 waiter
  // behind the blocked head still restores in the same step.
  EXPECT_EQ(harness.running_sequences(),
            (std::vector<Sequence*>{decode->sequences()[0].get(),
                                    other->sequences()[0].get()}));
  ASSERT_EQ(harness.waiting().size(), 1u);
  EXPECT_EQ(harness.waiting().front().request, blocked);
  EXPECT_FALSE(blocked->sequences()[0]->kv_state().has_any_blocks());
}

TEST(DecodeRestoreTest, RestoreKeepsDecodeReserve) {
  for (int32_t reserved_tokens : {0, 96}) {
    HostRestoreConfig config;
    ScopedConfigValue<int32_t> reserve(
        SchedulerConfig::get_instance().decode_reserved_tokens_per_request(),
        reserved_tokens);
    HostRestoreBlockPool pool(/*num_blocks=*/11, /*dp_size=*/1);
    ContinuousScheduler::Options options =
        make_restore_options(/*token_budget=*/256, /*dp_size=*/1);
    DecodeFirstPolicy policy(decode_first_mode(), options);
    RestoreHarness harness(&pool, options);
    auto decode = make_running_decode(pool, 32, 64, /*dp_rank=*/0);
    harness.decode_queue().push(decode);
    auto waiter = make_restore_waiter(32, 8, /*dp_rank=*/0);
    harness.add_waiter(waiter);

    harness.schedule(policy, 256);

    // 7 free blocks: the waiter needs 3 for its tokens, plus 1 of its own
    // reserve and 4 for the running decode when the reserve is enabled.
    EXPECT_EQ(harness.waiting().empty(), reserved_tokens == 0)
        << "reserved_tokens=" << reserved_tokens;
  }
}

TEST(DecodeRestoreTest, PreemptedInflightRestoreReturnsToQueueFront) {
  HostRestoreConfig config;
  HostRestoreBlockPool pool(/*num_blocks=*/8, /*dp_size=*/1);
  ContinuousScheduler::Options options =
      make_restore_options(/*token_budget=*/17, /*dp_size=*/1);
  options.max_tokens_per_chunk_for_prefill() = 16;
  DecodeFirstPolicy policy(decode_first_mode(), options);
  RestoreHarness harness(&pool, options);
  auto decode = make_running_decode(pool, 47, 64, /*dp_rank=*/0);
  harness.decode_queue().push(decode);
  auto inflight = make_restore_waiter(48, 8, /*dp_rank=*/0);
  harness.add_waiter(inflight);

  harness.schedule(policy, 17);
  ASSERT_EQ(harness.running_sequences().size(), 2u);
  ASSERT_EQ(harness.running_sequences()[1], inflight->sequences()[0].get());
  harness.advance();
  ASSERT_TRUE(inflight->sequences()[0]->is_chunked_prefill_stage());

  // The decode crosses into a new block while every free block is held.
  decode->sequences()[0]->append_token(Token(2));
  const int32_t free_blocks = static_cast<int32_t>(pool.num_free_blocks()[0]);
  ASSERT_GT(free_blocks, 0);
  auto held = hold_blocks(pool, free_blocks, /*dp_rank=*/0);
  auto later = make_restore_waiter(32, 8, /*dp_rank=*/0);
  harness.add_waiter(later);
  harness.schedule(policy, 17);

  EXPECT_EQ(harness.running_sequences(),
            (std::vector<Sequence*>{decode->sequences()[0].get()}));
  ASSERT_EQ(harness.waiting().size(), 2u);
  EXPECT_EQ(harness.waiting()[0].request, inflight);
  EXPECT_EQ(harness.waiting()[1].request, later);
  EXPECT_TRUE(inflight->preempted());
  EXPECT_FALSE(inflight->sequences()[0]->kv_state().has_any_blocks());
  EXPECT_TRUE(harness.chunk_queue().empty());
}

TEST(DecodeRestoreTest, PrefillFirstContinuesInflightRestoreFirst) {
  HostRestoreConfig config;
  HostRestoreBlockPool pool(/*num_blocks=*/32, /*dp_size=*/1);
  ContinuousScheduler::Options options =
      make_restore_options(/*token_budget=*/16, /*dp_size=*/1);
  options.max_tokens_per_chunk_for_prefill() = 16;
  BatchMode mode{.enable_mix_batch = false,
                 .enable_chunked_prefill = true,
                 .priority_strategy = "fcfs"};
  PrefillFirstPolicy policy(mode, options);
  RestoreHarness harness(&pool, options);
  auto first = make_restore_waiter(48, 8, /*dp_rank=*/0);
  harness.add_waiter(first);
  harness.schedule(policy, 16);
  ASSERT_EQ(harness.running_sequences(),
            (std::vector<Sequence*>{first->sequences()[0].get()}));
  harness.advance();
  auto next = make_restore_waiter(32, 8, /*dp_rank=*/0);
  harness.add_waiter(next);

  harness.schedule(policy, 16);

  EXPECT_EQ(harness.running_sequences(),
            (std::vector<Sequence*>{first->sequences()[0].get()}));
  ASSERT_EQ(harness.waiting().size(), 1u);
  EXPECT_EQ(harness.waiting().front().request, next);
}

TEST(DecodeRestoreTest, ServedBlockedDecodeNoLongerHoldsRestore) {
  HostRestoreConfig config;
  HostRestoreBlockPool pool(/*num_blocks=*/8, /*dp_size=*/1);
  ContinuousScheduler::Options options =
      make_restore_options(/*token_budget=*/256, /*dp_size=*/1);
  BatchMode mode{.enable_mix_batch = false,
                 .enable_chunked_prefill = true,
                 .priority_strategy = "fcfs"};
  PrefillFirstPolicy policy(mode, options);
  RestoreHarness harness(&pool, options);
  // `first` crosses into a third block; `victim` still fits in two.
  auto first = make_running_decode(pool, 32, 64, /*dp_rank=*/0);
  auto victim = make_running_decode(pool, 31, 64, /*dp_rank=*/0);
  // push() inserts at the head: `first` is scheduled first, `victim` is the
  // preemption candidate at the back.
  harness.decode_queue().push(victim);
  harness.decode_queue().push(first);
  const int32_t free_blocks = static_cast<int32_t>(pool.num_free_blocks()[0]);
  auto held = hold_blocks(pool, free_blocks, /*dp_rank=*/0);

  harness.schedule(policy, 256);
  ASSERT_EQ(harness.running_sequences(),
            (std::vector<Sequence*>{first->sequences()[0].get()}));
  ASSERT_EQ(harness.waiting().size(), 1u);
  harness.advance();

  // `first` got its block by preempting `victim`, so once room frees up the
  // restore no longer waits behind it.
  pool.deallocate(held.get());
  harness.schedule(policy, 256);

  EXPECT_EQ(harness.running_sequences(),
            (std::vector<Sequence*>{victim->sequences()[0].get()}));
  EXPECT_TRUE(harness.waiting().empty());
}

TEST(DecodeRestoreTest, ComputesDecodeRankLoad) {
  HostRestoreBlockPool pool(/*num_blocks=*/32, /*dp_size=*/2);
  ContinuousScheduler::Options options =
      make_restore_options(/*token_budget=*/256, /*dp_size=*/2);
  RestoreHarness harness(&pool, options);
  // 63 tokens left, capped at 20: 2 blocks.
  harness.decode_queue().push(make_running_decode(pool, 32, 64, 0));
  // 1 token left: 1 block.
  harness.decode_queue().push(make_running_decode(pool, 32, 2, 1));
  // In-flight restore on rank 1 holding 1 of its 4 blocks: 3 blocks to grow
  // plus min(20, 7) reserved tokens in 1 block.
  auto inflight = make_restore_waiter(48, 8, /*dp_rank=*/1);
  ASSERT_TRUE(
      pool.BlockManagerPool::allocate(inflight->sequences()[0].get(), 16));
  inflight->sequences()[0]->kv_state().set_kv_cache_tokens_num(16);
  harness.chunk_queue().push(inflight);

  const DecodeRankLoad load = compute_decode_rank_load(harness.state(), 20);
  EXPECT_EQ(load.reserved_blocks, (std::vector<size_t>{2, 5}));
  EXPECT_EQ(load.num_requests, (std::vector<size_t>{1, 2}));
  EXPECT_EQ(load.requests.size(), 3u);
  EXPECT_EQ(compute_decode_rank_load(harness.state(), 0).reserved_blocks,
            (std::vector<size_t>{0, 3}));
}

TEST(DecodeRestoreTest, PrefillHandoffKeepsGrowthReserve) {
  ScopedConfigValue<int32_t> reserve_guard(
      SchedulerConfig::get_instance().decode_reserved_tokens_per_request(), 16);
  HostRestoreBlockPool pool(/*num_blocks=*/8, /*dp_size=*/1);
  RestoreHarness harness(
      &pool, make_restore_options(/*token_budget=*/256, /*dp_size=*/1));
  auto request = generate_request({32},
                                  {32},
                                  std::nullopt,
                                  std::nullopt,
                                  /*max_context_len=*/4096)
                     .front();
  Sequence* sequence = request->sequences().front().get();
  sequence->set_dp_rank(0);
  ASSERT_EQ(harness.admission().admit(request),
            DecodeAdmissionResult::ADMITTED);

  // The request retains its prompt allocation while it needs local prefill.
  sequence->kv_state().set_kv_cache_tokens_num(16);
  ASSERT_TRUE(sequence->is_prefill_stage());
  ASSERT_FALSE(request->preempted());
  for (int32_t phase = 0; phase < 3; ++phase) {
    SCOPED_TRACE(phase);
    if (phase == 1) {
      harness.chunk_queue().push(request);
    }
    if (phase == 2) {
      harness.chunk_queue().pop_top();
      sequence->kv_state().set_kv_cache_tokens_num(32);
      sequence->append_token(Token(/*token_id=*/1));
      ASSERT_FALSE(sequence->is_prefill_stage());
      harness.decode_queue().push(request);
    }
    const DecodeRankLoad load = compute_decode_rank_load(harness.state(), 16);
    harness.admission().publish(load);
    EXPECT_EQ(load.reserved_blocks[0], phase == 0 ? 0u : 1u);
    EXPECT_EQ(harness.admission().unscheduled_reserve_blocks(load)[0],
              phase == 0 ? 1u : 0u);

    // Seven usable blocks: two held, one reserved. A four-block arrival
    // with its own reserve cannot fit; a three-block arrival fits exactly.
    for (int32_t prompt_tokens : {64, 48}) {
      auto probe = generate_request({prompt_tokens},
                                    {32},
                                    std::nullopt,
                                    std::nullopt,
                                    /*max_context_len=*/4096)
                       .front();
      probe->sequences().front()->set_dp_rank(0);
      const DecodeAdmissionResult result = harness.admission().admit(probe);
      EXPECT_EQ(result,
                prompt_tokens == 64 ? DecodeAdmissionResult::RETRY_LATER
                                    : DecodeAdmissionResult::ADMITTED);
      probe->sequences().front()->reset();
      probe.reset();
      harness.admission().publish(load);
      EXPECT_EQ(pool.num_used_blocks()[0], 2u);
    }
  }
  sequence->reset();
}

}  // namespace xllm
