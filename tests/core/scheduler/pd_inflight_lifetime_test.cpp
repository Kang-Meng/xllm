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

#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "distributed_runtime/engine.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request.h"
#include "scheduler/disagg_pd_scheduler.h"

namespace xllm {
namespace {

template <typename Value>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(Value& value, Value replacement)
      : value_(value), previous_(value) {
    value_ = replacement;
  }

  ~ScopedConfigValue() { value_ = previous_; }

 private:
  Value& value_;
  Value previous_;
};

class FakeTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view&,
              std::vector<int32_t>*,
              bool) const override {
    return false;
  }
  std::string decode(const Slice<int32_t>&, bool) const override { return ""; }
  std::optional<int32_t> token_to_id(const std::string_view&) const override {
    return std::nullopt;
  }
  std::string id_to_token(int32_t) const override { return ""; }
  size_t vocab_size() const override { return 1024; }
  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

class DeferredWriteEngine final : public Engine {
 public:
  explicit DeferredWriteEngine(int32_t num_blocks, bool prefix_cache = false)
      : memory_(static_cast<size_t>(num_blocks), 0) {
    BlockManagerPool::Options options;
    options.num_blocks(num_blocks)
        .block_size(128)
        .enable_disagg_pd(true)
        .instance_is_decode(true)
        .enable_prefix_cache(prefix_cache);
    kv_cache_manager_ = std::make_unique<BlockManagerPool>(options, 1);
    tokenizer_ = std::make_unique<FakeTokenizer>();
  }

  ForwardOutput step(std::vector<Batch>& batches) override {
    ++launches_;
    std::vector<int32_t> writable_ids;
    for (Batch& batch : batches) {
      for (Sequence* sequence : batch.get_sequences()) {
        const size_t position = sequence->kv_cache_tokens_num();
        writable_ids.emplace_back(
            sequence->kv_state().blocks(BlockType::KV)[position / 128].id());
        sequence->kv_state().incr_kv_cache_tokens_num(1);
        sequence->append_token(Token(-1));
      }
    }
    pending_writes_.emplace_back(std::move(writable_ids));
    return {};
  }

  void update_last_step_result(std::vector<Batch>& batches) override {
    ++collections_;
    ASSERT_FALSE(pending_writes_.empty());
    if (before_collect) {
      before_collect();
    }
    for (const int32_t block_id : pending_writes_.front()) {
      memory(block_id) = 1;
    }
    pending_writes_.pop_front();
    for (Batch& batch : batches) {
      for (Sequence* sequence : batch.get_sequences()) {
        if (!sequence->finished() && !sequence->cancelled()) {
          sequence->update_last_step_token(Token(5), 0);
        }
      }
    }
  }

  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }

  size_t launches() const { return launches_; }
  size_t collections() const { return collections_; }
  int32_t pending_block_id() const { return pending_writes_.front().front(); }
  int32_t& memory(int32_t block_id) {
    return memory_.at(static_cast<size_t>(block_id));
  }

  std::function<void()> before_collect;

 private:
  std::vector<int32_t> memory_;
  std::deque<std::vector<int32_t>> pending_writes_;
  size_t launches_ = 0;
  size_t collections_ = 0;
};

class TestPDScheduler final : public DisaggPDScheduler {
 public:
  explicit TestPDScheduler(Engine* engine)
      : DisaggPDScheduler(engine, make_options(), false) {}

  void wait_for_responses() { response_processor_->wait_completion(); }

 private:
  static Options make_options() {
    Options options;
    options.enable_disagg_pd(true)
        .instance_role(InstanceRole::DECODE)
        .enable_schedule_overlap(true)
        .max_tokens_per_batch(512)
        .max_seqs_per_batch(4)
        .max_tokens_per_chunk_for_prefill(512)
        .dp_size(1);
    return options;
  }
};

std::shared_ptr<Request> make_request(const std::string& request_id,
                                      std::vector<int32_t> prompt) {
  StoppingChecker stopping;
  stopping.set_max_generated_tokens(2);
  stopping.set_max_context_len(1024);
  stopping.set_ignore_eos(true);
  RequestState state(
      "prompt",
      std::move(prompt),
      RequestSamplingParam(),
      SchedulerParam(),
      stopping,
      1024,
      1,
      1,
      false,
      false,
      false,
      false,
      true,
      [](const RequestOutput&) { return true; },
      nullptr);
  return std::make_shared<Request>(
      request_id, request_id, request_id, std::move(state), request_id);
}

void receive_prefill(TestPDScheduler& scheduler,
                     std::shared_ptr<Request>& request) {
  ASSERT_EQ(scheduler.try_admit(request), DecodeAdmissionResult::ADMITTED);
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));
  ASSERT_TRUE(scheduler.decode_recv_first_generation(request->request_id(),
                                                     42,
                                                     false,
                                                     0.0f,
                                                     0.1,
                                                     0.0,
                                                     {},
                                                     {},
                                                     "PUSH",
                                                     {},
                                                     {},
                                                     {},
                                                     1,
                                                     0));
}

enum class PendingRequestExit : int8_t { FINISHED, CANCELLED, FAILED };

class PDInflightLifetimeTest : public ::testing::Test {
 protected:
  PDInflightLifetimeTest()
      : overlap_(SchedulerConfig::get_instance().enable_schedule_overlap(),
                 true),
        reserve_(SchedulerConfig::get_instance()
                     .decode_reserved_tokens_per_request(),
                 0),
        utilization_(SchedulerConfig::get_instance()
                         .prefill_scheduling_memory_usage_threshold(),
                     2.0) {}

 private:
  ScopedConfigValue<bool> overlap_;
  ScopedConfigValue<int32_t> reserve_;
  ScopedConfigValue<double> utilization_;
};

TEST_F(PDInflightLifetimeTest, DestinationCannotReusePendingWrite) {
  for (const bool prefix_cache : {false, true}) {
    for (const int32_t num_blocks : {2, 4}) {
      for (const PendingRequestExit exit : {PendingRequestExit::FINISHED,
                                            PendingRequestExit::CANCELLED,
                                            PendingRequestExit::FAILED}) {
        SCOPED_TRACE(prefix_cache);
        SCOPED_TRACE(num_blocks);
        SCOPED_TRACE(static_cast<int32_t>(exit));
        ScopedConfigValue<bool> prefix(
            KVCacheConfig::get_instance().enable_prefix_cache(), prefix_cache);
        DeferredWriteEngine engine(num_blocks, prefix_cache);
        TestPDScheduler scheduler(&engine);
        BlockManagerPool* pool = engine.block_manager_pool();
        const size_t initial_free = pool->num_free_blocks().front();
        auto ending = make_request("ending", {1, 2, 3, 4});
        receive_prefill(scheduler, ending);
        scheduler.step(absl::ZeroDuration());
        if (exit == PendingRequestExit::FINISHED) {
          scheduler.step(absl::ZeroDuration());
          ASSERT_TRUE(ending->finished());
        } else if (exit == PendingRequestExit::CANCELLED) {
          ending->set_cancel();
        } else {
          ending->sequences()[0]->fail(
              Status(StatusCode::UNKNOWN, "injected request failure"));
        }
        const size_t expected_launches =
            exit == PendingRequestExit::FINISHED ? 2 : 1;
        ASSERT_EQ(engine.launches(), expected_launches);
        ASSERT_EQ(engine.collections(), expected_launches - 1);
        const int32_t old_write_id = engine.pending_block_id();
        auto destination =
            make_request("destination", std::vector<int32_t>(128, 9));
        engine.before_collect = [&] {
          EXPECT_FALSE(ending->sequences()[0]->has_any_blocks());
          EXPECT_EQ(pool->num_used_blocks().front(), 1u);
          auto allocation = std::async(std::launch::async, [&] {
            return scheduler.try_admit(destination);
          });
          EXPECT_EQ(allocation.get(),
                    num_blocks > 2 ? DecodeAdmissionResult::ADMITTED
                                   : DecodeAdmissionResult::REJECTED);
          for (const Block& block :
               destination->sequences()[0]->kv_state().blocks(BlockType::KV)) {
            EXPECT_NE(block.id(), old_write_id);
            engine.memory(block.id()) = 2;
          }
        };
        scheduler.step(absl::ZeroDuration());
        engine.before_collect = nullptr;
        EXPECT_EQ(engine.launches(), expected_launches);
        EXPECT_EQ(engine.collections(), expected_launches);
        for (const Block& block :
             destination->sequences()[0]->kv_state().blocks(BlockType::KV)) {
          EXPECT_EQ(engine.memory(block.id()), 2);
        }
        if (num_blocks == 2) {
          ASSERT_EQ(scheduler.try_admit(destination),
                    DecodeAdmissionResult::ADMITTED);
          EXPECT_EQ(destination->sequences()[0]
                        ->kv_state()
                        .blocks(BlockType::KV)[0]
                        .id(),
                    old_write_id);
        }
        pool->deallocate(destination.get());
        pool->reset_prefix_cache();
        scheduler.wait_for_responses();
        EXPECT_EQ(pool->num_used_blocks().front(), 0u);
        EXPECT_EQ(pool->num_free_blocks().front(), initial_free);
      }
    }
  }
}

TEST_F(PDInflightLifetimeTest, PreparedBatchAlreadyOwnsWritableBlocks) {
  ScopedConfigValue<bool> prefix(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  DeferredWriteEngine engine(3);
  TestPDScheduler scheduler(&engine);
  auto request = make_request("prepared", {1, 2, 3, 4});
  receive_prefill(scheduler, request);
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_EQ(sequence->kv_state().blocks(BlockType::KV).front().ref_count(), 1u);
  std::vector<Batch> batches = scheduler.prepare_batch_test();
  ASSERT_FALSE(batches.front().empty());
  EXPECT_EQ(engine.launches(), 0u);
  EXPECT_EQ(sequence->kv_state().blocks(BlockType::KV).front().ref_count(), 2u);
  BlockManagerPool* pool = engine.block_manager_pool();
  pool->deallocate(request.get());
  EXPECT_EQ(pool->num_used_blocks().front(), 1u);
  batches.clear();
  EXPECT_EQ(pool->num_used_blocks().front(), 0u);
}

TEST_F(PDInflightLifetimeTest, DestructionCollectsBeforeDroppingBatch) {
  ScopedConfigValue<bool> prefix(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  DeferredWriteEngine engine(3);
  BlockManagerPool* pool = engine.block_manager_pool();
  const size_t initial_free = pool->num_free_blocks().front();
  auto request = make_request("pending", {1, 2, 3, 4});
  {
    TestPDScheduler scheduler(&engine);
    receive_prefill(scheduler, request);
    scheduler.step(absl::ZeroDuration());
    ASSERT_EQ(engine.launches(), 1u);
    ASSERT_EQ(engine.collections(), 0u);
    engine.before_collect = [&] {
      EXPECT_TRUE(request->sequences()[0]->has_any_blocks());
      EXPECT_GT(pool->num_used_blocks().front(), 0u);
    };
  }
  engine.before_collect = nullptr;
  EXPECT_EQ(engine.collections(), 1u);
  request.reset();
  EXPECT_EQ(pool->num_used_blocks().front(), 0u);
  EXPECT_EQ(pool->num_free_blocks().front(), initial_free);
}

}  // namespace
}  // namespace xllm
