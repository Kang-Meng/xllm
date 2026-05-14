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

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <vector>

#define protected public
#include "scheduler/disagg_pd_chunked_prefill_scheduler.h"
#undef protected

#include "distributed_runtime/engine.h"

namespace xllm {

namespace {

class FakeTokenizer final : public Tokenizer {
 public:
  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

BlockManagerPool::Options make_counting_block_manager_options() {
  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  return options;
}

class CountingBlockManagerPool final : public BlockManagerPool {
 public:
  CountingBlockManagerPool()
      : BlockManagerPool(make_counting_block_manager_options(),
                         /*dp_size=*/1) {}

  void transfer_blocks(std::vector<Batch>& batches) override {
    transfer_blocks_count_.fetch_add(1, std::memory_order_relaxed);
    for (const Batch& batch : batches) {
      if (!batch.empty()) {
        non_empty_transfer_blocks_count_.fetch_add(1,
                                                   std::memory_order_relaxed);
        break;
      }
    }
  }

  int32_t non_empty_transfer_blocks_count() const {
    return non_empty_transfer_blocks_count_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<int32_t> transfer_blocks_count_{0};
  std::atomic<int32_t> non_empty_transfer_blocks_count_{0};
};

class CountingEngine final : public Engine {
 public:
  CountingEngine() {
    fake_tokenizer_ = std::make_unique<FakeTokenizer>();
    fake_block_manager_ = std::make_unique<CountingBlockManagerPool>();
  }

  ForwardOutput step(std::vector<Batch>& batch) override { NOT_IMPLEMENTED(); }

  void update_last_step_result(std::vector<Batch>& batch) override {
    NOT_IMPLEMENTED();
  }

  const Tokenizer* tokenizer() const override { return fake_tokenizer_.get(); }

  BlockManagerPool* block_manager_pool() const override {
    return fake_block_manager_.get();
  }

  std::vector<int64_t> get_active_activation_memory() const override {
    NOT_IMPLEMENTED();
  }

  CountingBlockManagerPool* counting_block_manager() const {
    return fake_block_manager_.get();
  }

 private:
  std::unique_ptr<Tokenizer> fake_tokenizer_;
  std::unique_ptr<CountingBlockManagerPool> fake_block_manager_;
};

ContinuousScheduler::Options make_prefill_scheduler_options() {
  ContinuousScheduler::Options options;
  options.max_tokens_per_batch(16)
      .max_seqs_per_batch(1)
      .max_tokens_per_chunk_for_prefill(8)
      .dp_size(1)
      .enable_disagg_pd(true)
      .enable_pd_ooc(true)
      .instance_role(InstanceRole::PREFILL)
      .enable_schedule_overlap(false);
  return options;
}

std::shared_ptr<Request> make_request(const size_t prompt_len) {
  std::vector<int32_t> prompt_token_ids(prompt_len, 1);
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);
  stopping_checker.set_max_context_len(1024);
  stopping_checker.set_ignore_eos(true);

  RequestState request_state("x",
                             prompt_token_ids,
                             sampling_param,
                             scheduler_param,
                             stopping_checker,
                             prompt_len + 8,
                             1,
                             1,
                             false,
                             false,
                             false,
                             false,
                             false,
                             nullptr,
                             nullptr);

  return std::make_shared<Request>(
      "request-1", "x-request-1", "0", std::move(request_state), "service-1");
}

}  // namespace

TEST(DisaggPDChunkedPrefillSchedulerTest, PicksCurrentChunkBudget) {
  const PDChunkBudget budget = pick_pd_chunk_budget(32, 96, 40, 64);
  EXPECT_EQ(budget.next_tokens, 40);
  EXPECT_EQ(budget.max_tokens, 72);
}

TEST(DisaggPDChunkedPrefillSchedulerTest, LastPromptChunkStopsAtPromptEnd) {
  const PDChunkBudget budget = pick_pd_chunk_budget(80, 96, 40, 64);
  EXPECT_EQ(budget.next_tokens, 16);
  EXPECT_EQ(budget.max_tokens, 96);
}

TEST(DisaggPDChunkedPrefillSchedulerTest, EmptyBudgetRejectsSchedule) {
  const PDChunkBudget budget = pick_pd_chunk_budget(32, 96, 40, 0);
  EXPECT_EQ(budget.next_tokens, 0);
  EXPECT_EQ(budget.max_tokens, 32);
}

TEST(DisaggPDChunkedPrefillSchedulerTest,
     PrefillPrepareBatchTransfersNonEmptyBatches) {
  CountingEngine engine;
  DisaggPDChunkedPrefillScheduler scheduler(&engine,
                                            make_prefill_scheduler_options());
  std::shared_ptr<Request> request = make_request(/*prompt_len=*/16);

  ASSERT_TRUE(scheduler.request_queue_.write(request));
  std::vector<Batch> batches = scheduler.prepare_batch_test();

  ASSERT_EQ(batches.size(), 1u);
  ASSERT_FALSE(batches[0].empty());
  EXPECT_EQ(engine.counting_block_manager()->non_empty_transfer_blocks_count(),
            1);
}

}  // namespace xllm
