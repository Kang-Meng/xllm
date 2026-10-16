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

#include <absl/time/time.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "core/distributed_runtime/engine.h"
#include "core/framework/batch/batch.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/request.h"
#include "core/framework/request/request_state.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/scheduler/disagg_pd_scheduler.h"

namespace xllm {
namespace {

class FakeTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& /*text*/,
              std::vector<int32_t>* /*ids*/,
              bool /*add_special_tokens*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string decode(const Slice<int32_t>& /*ids*/,
                     bool /*skip_special_tokens*/) const override {
    return "";
  }

  std::optional<int32_t> token_to_id(
      const std::string_view& /*token*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string id_to_token(int32_t /*id*/) const override { NOT_IMPLEMENTED(); }

  size_t vocab_size() const override { NOT_IMPLEMENTED(); }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

// Keep real allocation and eviction; model a backend that permits Host restore.
class RestoreBlockManager final : public BlockManagerPool {
 public:
  RestoreBlockManager(const Options& options, int32_t dp_size)
      : BlockManagerPool(options, dp_size) {}
  bool supports_host_cache_restore() const override { return true; }
  bool has_pending_async_block_release() const override {
    return pending_release_.load();
  }
  void set_pending_release(bool pending) { pending_release_.store(pending); }
  bool allocate(Sequence* sequence, size_t num_tokens) override {
    if (sequence == fail_allocate_for_) {
      return false;
    }
    return BlockManagerPool::allocate(sequence, num_tokens);
  }
  void fail_allocate_for(Sequence* sequence) { fail_allocate_for_ = sequence; }

 private:
  std::atomic<bool> pending_release_{false};
  Sequence* fail_allocate_for_ = nullptr;
};

// Stop inside allocation so admission overlaps a scheduler operation while
// retaining real block allocation and rollback behavior.
class GatedAdmissionPool final : public BlockManagerPool {
 public:
  explicit GatedAdmissionPool(const Options& options)
      : BlockManagerPool(options, /*dp_size=*/1),
        release_(release_promise_.get_future().share()) {}

  bool try_allocate(Sequence* sequence) override {
    if (allocation_calls_.fetch_add(1) == 0) {
      entered_.set_value();
      release_.wait();
    }
    return BlockManagerPool::try_allocate(sequence);
  }

  std::future<void> entered() { return entered_.get_future(); }
  void release() { release_promise_.set_value(); }
  int32_t allocation_calls() const { return allocation_calls_.load(); }

 private:
  std::atomic<int32_t> allocation_calls_{0};
  std::promise<void> entered_;
  std::promise<void> release_promise_;
  std::shared_future<void> release_;
};

class FakeEngine final : public Engine {
 public:
  FakeEngine(int32_t num_blocks,
             int32_t block_size,
             int32_t speculative_tokens,
             int32_t dp_size) {
    BlockManagerPool::Options options;
    options.num_blocks(num_blocks)
        .block_size(block_size)
        .enable_prefix_cache(true)
        .enable_disagg_pd(true)
        .num_speculative_tokens(speculative_tokens)
        .num_embedding_blocks(num_blocks);
    block_manager_ = std::make_unique<RestoreBlockManager>(options, dp_size);
    model_args_.num_speculative_tokens(speculative_tokens);
  }
  ForwardOutput step(std::vector<Batch>& /*batch*/) override {
    NOT_IMPLEMENTED();
  }
  void update_last_step_result(std::vector<Batch>& /*batch*/) override {
    NOT_IMPLEMENTED();
  }
  const Tokenizer* tokenizer() const override { return &tokenizer_; }
  BlockManagerPool* block_manager_pool() const override {
    return block_manager_.get();
  }
  const ModelArgs& model_args() const override { return model_args_; }
  const TokenizerArgs& tokenizer_args() const override { NOT_IMPLEMENTED(); }
  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }
  bool init() override { return true; }
  bool pull_kv_blocks(
      int32_t /*src_dp_size*/,
      int32_t /*src_dp_rank*/,
      const std::vector<uint64_t>& /*src_cluster_ids*/,
      const std::vector<std::string>& /*src_addrs*/,
      int32_t /*dst_dp_rank*/,
      const std::vector<KVTransferMapping>& /*mappings*/) override {
    return pull ? pull() : true;
  }
  std::function<bool()> pull;

 private:
  FakeTokenizer tokenizer_;
  std::unique_ptr<RestoreBlockManager> block_manager_;
  ModelArgs model_args_;
};

class TestDisaggPDScheduler final : public DisaggPDScheduler {
 public:
  TestDisaggPDScheduler(Engine* engine, const Options& options)
      : DisaggPDScheduler(engine, options, /*start_services=*/false) {}
};

DisaggPDScheduler::Options make_decode_options() {
  DisaggPDScheduler::Options options;
  options.enable_pd_ooc(false)
      .enable_disagg_pd(true)
      .enable_schedule_overlap(false)
      .instance_role(InstanceRole::DECODE)
      .max_seqs_per_batch(4);
  return options;
}

std::shared_ptr<Request> make_request(
    const std::vector<int32_t>& prompt_token_ids,
    const std::string& req_id = "req") {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  stopping_checker.set_max_context_len(64);
  stopping_checker.set_ignore_eos(true);

  RequestState state("prompt",
                     prompt_token_ids,
                     sampling_param,
                     scheduler_param,
                     stopping_checker,
                     prompt_token_ids.size() + 8,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*logprobs=*/false,
                     /*skip_special_tokens=*/false,
                     /*include_usage=*/false,
                     /*mm_data=*/nullptr,
                     /*service_request_id=*/nullptr);

  return std::make_shared<Request>(req_id,
                                   "x-request-id",
                                   "x-request-time",
                                   std::move(state),
                                   "service-req");
}

class DecodeAdmissionConcurrencyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    BlockManagerPool::Options options;
    options.num_blocks(7).block_size(2).enable_prefix_cache(false);
    pool_ = std::make_unique<GatedAdmissionPool>(options);
    admission_ = std::make_unique<DecodeAdmission>(
        pool_.get(),
        DecodeAdmission::Options{.dp_size = 1,
                                 .reserved_tokens_per_request = 4});
  }

  std::unique_ptr<GatedAdmissionPool> pool_;
  std::unique_ptr<DecodeAdmission> admission_;
};

TEST_F(DecodeAdmissionConcurrencyTest, PublishKeepsConcurrentAdmissionReserve) {
  auto request = make_request({1, 2});
  request->sequences()[0]->set_dp_rank(0);
  auto entered = pool_->entered();
  auto admit = std::async(std::launch::async,
                          [&] { return admission_->admit(request); });
  const std::future_status entered_status =
      entered.wait_for(std::chrono::seconds(5));
  EXPECT_EQ(entered_status, std::future_status::ready);
  DecodeRankLoad load{.reserved_blocks = {0}, .num_requests = {0}};
  std::promise<void> publishing;
  auto publishing_started = publishing.get_future();
  auto publish = std::async(std::launch::async, [&] {
    publishing.set_value();
    admission_->publish(load);
  });
  publishing_started.wait();
  EXPECT_EQ(publish.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  pool_->release();
  EXPECT_EQ(admit.get(), DecodeAdmissionResult::ADMITTED);
  publish.get();

  // A stale scheduler snapshot must retain the concurrent pending reserve.
  auto too_large = make_request({11, 12, 13, 14, 15, 16});
  too_large->sequences()[0]->set_dp_rank(0);
  EXPECT_EQ(admission_->admit(too_large), DecodeAdmissionResult::RETRY_LATER);
  pool_->deallocate_without_cache(too_large->sequences()[0].get());
  EXPECT_EQ(pool_->num_used_blocks()[0], 1u);

  // Scheduler ownership replaces the pending reserve instead of adding it.
  load.reserved_blocks[0] = 1;
  load.num_requests[0] = 1;
  load.requests.insert(request.get());
  admission_->publish(load);
  auto fits = make_request({21, 22, 23, 24});
  fits->sequences()[0]->set_dp_rank(0);
  EXPECT_EQ(admission_->admit(fits), DecodeAdmissionResult::ADMITTED);
  pool_->deallocate_without_cache(fits->sequences()[0].get());
  pool_->deallocate_without_cache(request->sequences()[0].get());
}

TEST_F(DecodeAdmissionConcurrencyTest,
       PauseWaitsForAllocationAndBlocksArrivals) {
  auto request = make_request({1, 2});
  request->sequences()[0]->set_dp_rank(0);
  auto entered = pool_->entered();
  auto admit = std::async(std::launch::async,
                          [&] { return admission_->admit(request); });
  const std::future_status entered_status =
      entered.wait_for(std::chrono::seconds(5));
  EXPECT_EQ(entered_status, std::future_status::ready);
  std::promise<void> pausing;
  auto pausing_started = pausing.get_future();
  auto pause = std::async(std::launch::async, [&] {
    pausing.set_value();
    admission_->pause();
  });
  pausing_started.wait();
  EXPECT_EQ(pause.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  pool_->release();
  EXPECT_EQ(admit.get(), DecodeAdmissionResult::ADMITTED);
  pause.get();

  auto probe = make_request({11, 12});
  probe->sequences()[0]->set_dp_rank(0);
  EXPECT_EQ(admission_->admit(probe), DecodeAdmissionResult::RETRY_LATER);
  EXPECT_EQ(pool_->allocation_calls(), 1);
  EXPECT_EQ(pool_->num_used_blocks()[0], 1u);
  pool_->deallocate_without_cache(probe->sequences()[0].get());
  pool_->deallocate_without_cache(request->sequences()[0].get());
  admission_->reopen();
  EXPECT_EQ(admission_->admit(probe), DecodeAdmissionResult::ADMITTED);
  pool_->deallocate_without_cache(probe->sequences()[0].get());
}

template <typename T>
class RestoreConfigGuard final {
 public:
  RestoreConfigGuard(T& value, T replacement) : value_(value), saved_(value) {
    value_ = std::move(replacement);
  }
  ~RestoreConfigGuard() { value_ = std::move(saved_); }

 private:
  T& value_;
  T saved_;
};

class HostRestoreSchedulerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    init_scheduler(/*block_size=*/2, /*speculative_tokens=*/0);
  }

  void init_scheduler(int32_t block_size,
                      int32_t speculative_tokens,
                      int32_t num_blocks = 5,
                      const std::string& strategy = "fcfs") {
    scheduler_.reset();
    engine_.reset();
    speculative_tokens_ = speculative_tokens;
    engine_ = std::make_unique<FakeEngine>(num_blocks,
                                           block_size,
                                           speculative_tokens,
                                           /*dp_size=*/2);
    auto options = make_decode_options();
    options.enable_pd_ooc(false)
        .priority_strategy(strategy)
        .enable_chunked_prefill(true)
        .dp_size(2)
        .num_speculative_tokens(speculative_tokens)
        .max_tokens_per_batch(std::max(block_size, speculative_tokens + 1))
        .max_tokens_per_chunk_for_prefill(block_size);
    scheduler_ =
        std::make_unique<TestDisaggPDScheduler>(engine_.get(), options);
  }

  std::shared_ptr<Request> reserve(const std::string& id,
                                   int32_t rank,
                                   const std::vector<int32_t>& tokens) {
    auto request = make_request(tokens, id);
    request->state().pd_reservation_id = id + "-reservation";
    request->state().output_func = [](const RequestOutput&) { return true; };
    request->sequences()[0]->set_dp_rank(rank);
    EXPECT_EQ(scheduler_->try_admit(request), DecodeAdmissionResult::ADMITTED);
    EXPECT_TRUE(scheduler_->decode_schedule(request, "prefill"));
    return request;
  }

  bool deliver(const std::shared_ptr<Request>& request,
               const std::string& mode = "PUSH") {
    return scheduler_->decode_recv_first_generation(
        request->request_id(),
        /*token_id=*/9,
        /*has_logprob=*/false,
        /*logprob=*/0.0f,
        /*time_to_first_token_latency_seconds=*/0.0,
        /*upstream_elapsed_seconds=*/0.0,
        /*top_tokens=*/{},
        /*top_logprobs=*/{},
        /*kv_cache_transfer_mode=*/mode,
        /*src_cluster_ids=*/{},
        /*src_addrs=*/{},
        /*source_mappings=*/{},
        /*src_dp_size=*/2,
        /*src_dp_rank=*/0,
        speculative_tokens_ > 0 ? torch::zeros({1, 2}) : torch::Tensor());
  }

  void advance(std::vector<Batch>& batches) {
    for (Batch& batch : batches) {
      for (size_t i = 0; i < batch.size(); ++i) {
        Sequence* sequence = batch[i];
        sequence->kv_state().incr_kv_cache_tokens_num(
            batch.get_allowed_max_tokens()[i]);
        if (sequence->is_prefill_stage()) {
          continue;
        }
        sequence->append_token(Token(/*token_id=*/42));
      }
    }
  }

  std::array<std::shared_ptr<Request>, 3> preempt_pair() {
    init_scheduler(/*block_size=*/2,
                   /*speculative_tokens=*/0,
                   /*num_blocks=*/7,
                   /*strategy=*/"priority");
    auto first = reserve("first", /*rank=*/0, {1, 2, 3, 4});
    auto later = reserve("later", /*rank=*/0, {5, 6, 7, 8});
    auto earlier = reserve("earlier", /*rank=*/0, {11, 12, 13, 14});
    auto reservation = reserve("reservation", /*rank=*/1, {31, 32});
    first->state().scheduler_param.priority = RequestPriority::HIGH;
    later->state().scheduler_param.priority = RequestPriority::NORMAL;
    earlier->state().scheduler_param.priority = RequestPriority::LOW;
    EXPECT_TRUE(deliver(first));
    EXPECT_TRUE(deliver(later));
    EXPECT_TRUE(deliver(earlier));
    for (int32_t step = 0; step < 2; ++step) {
      auto batches = scheduler_->prepare_batch_test();
      EXPECT_EQ(batches[0].size(), 2u);
      advance(batches);
    }
    EXPECT_EQ(scheduler_->prepare_batch_test()[0].size(), 1u);
    EXPECT_TRUE(earlier->preempted());
    EXPECT_TRUE(later->preempted());
    first->set_cancel();
    EXPECT_TRUE(scheduler_->release_reservation(
        reservation->request_id(), reservation->state().pd_reservation_id));
    return {first, earlier, later};
  }

  int32_t speculative_tokens_ = 0;
  RestoreConfigGuard<std::string> algorithm_{
      SpeculativeConfig::get_instance().speculative_algorithm(),
      "MTP"};
  RestoreConfigGuard<bool> prefix_{
      KVCacheConfig::get_instance().enable_prefix_cache(),
      true};
  RestoreConfigGuard<bool> mix_{
      SchedulerConfig::get_instance().enable_mix_batch(),
      false};
  RestoreConfigGuard<double> host_{
      KVCacheStoreConfig::get_instance().host_blocks_factor(),
      2.0};
  // Restore mechanics with a handful of blocks; reserve tests set it.
  RestoreConfigGuard<int32_t> reserve_{
      SchedulerConfig::get_instance().decode_reserved_tokens_per_request(),
      0};
  std::unique_ptr<FakeEngine> engine_;
  std::unique_ptr<TestDisaggPDScheduler> scheduler_;
};

TEST_F(HostRestoreSchedulerTest, PreemptionClosesAdmissionOnEveryRank) {
  auto first = reserve("first", /*rank=*/0, {1, 2, 3, 4});
  auto victim = reserve("victim", /*rank=*/0, {5, 6, 7, 8});
  auto accepted = reserve("accepted", /*rank=*/1, {11, 12});
  ASSERT_TRUE(deliver(first));
  ASSERT_TRUE(deliver(victim));
  auto batches = scheduler_->prepare_batch_test();
  ASSERT_EQ(batches[0].size(), 1u);
  ASSERT_EQ(batches[0][0], first->sequences()[0].get());
  ASSERT_TRUE(victim->preempted());
  auto incoming = make_request({21, 22}, "incoming");
  incoming->sequences()[0]->set_dp_rank(1);
  const auto used = engine_->block_manager_pool()->num_used_blocks();
  for (RequestPriority priority :
       {RequestPriority::NORMAL, RequestPriority::HIGH}) {
    incoming->state().scheduler_param.priority = priority;
    EXPECT_EQ(scheduler_->try_admit(incoming),
              DecodeAdmissionResult::RETRY_LATER);
    EXPECT_EQ(engine_->block_manager_pool()->num_used_blocks(), used);
  }
  EXPECT_TRUE(deliver(accepted));
}

TEST_F(HostRestoreSchedulerTest, RestoreDoesNotWaitForOtherRankReservations) {
  auto first = reserve("first", /*rank=*/0, {1, 2, 3, 4});
  auto victim = reserve("victim", /*rank=*/0, {5, 6, 7, 8});
  auto accepted = reserve("accepted", /*rank=*/1, {11, 12});
  auto allocated = make_request({13, 14}, "allocated");
  allocated->sequences()[0]->set_dp_rank(1);
  ASSERT_EQ(scheduler_->try_admit(allocated), DecodeAdmissionResult::ADMITTED);
  ASSERT_TRUE(deliver(first));
  ASSERT_TRUE(deliver(victim));
  ASSERT_EQ(scheduler_->prepare_batch_test()[0].size(), 1u);
  ASSERT_TRUE(victim->preempted());
  first->set_cancel();

  // Rank-1 reservations hold their own blocks and do not delay the restore.
  auto batches = scheduler_->prepare_batch_test();
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0], victim->sequences()[0].get());
  EXPECT_TRUE(batches[1].empty());
  EXPECT_TRUE(scheduler_->release_reservation(
      accepted->request_id(), accepted->state().pd_reservation_id));
  engine_->block_manager_pool()->deallocate_without_cache(
      allocated->sequences()[0].get());
}

TEST_F(HostRestoreSchedulerTest, WaitsWhileRankHoldsForeignBlocks) {
  auto first = reserve("first", /*rank=*/0, {1, 2, 3, 4});
  auto victim = reserve("victim", /*rank=*/0, {5, 6, 7, 8});
  auto error = std::make_shared<std::promise<Status>>();
  auto result = error->get_future();
  victim->state().output_func = [error](const RequestOutput& output) {
    error->set_value(output.status.value());
    return true;
  };
  ASSERT_TRUE(deliver(first));
  ASSERT_TRUE(deliver(victim));
  ASSERT_EQ(scheduler_->prepare_batch_test()[0].size(), 1u);
  ASSERT_TRUE(victim->preempted());
  auto* pool =
      dynamic_cast<RestoreBlockManager*>(engine_->block_manager_pool());
  ASSERT_NE(pool, nullptr);
  pool->set_pending_release(true);
  first->set_cancel();
  EXPECT_TRUE(scheduler_->prepare_batch_test()[0].empty());
  // A transfer still holds 2 of the 4 rank-0 blocks; the victim needs 3.
  auto retained = make_request({30, 30, 30, 30}, "retained");
  retained->sequences()[0]->set_dp_rank(0);
  ASSERT_TRUE(
      engine_->block_manager_pool()->allocate(retained->sequences()[0].get()));
  pool->set_pending_release(false);

  EXPECT_TRUE(scheduler_->prepare_batch_test()[0].empty());
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  EXPECT_TRUE(victim->preempted());

  engine_->block_manager_pool()->deallocate(retained.get());
  auto batches = scheduler_->prepare_batch_test();
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0], victim->sequences()[0].get());
}

TEST_F(HostRestoreSchedulerTest, ReturnsFromStepWhileRestoreWaits) {
  auto first = reserve("first", /*rank=*/0, {1, 2, 3, 4});
  auto victim = reserve("victim", /*rank=*/0, {5, 6, 7, 8});
  ASSERT_TRUE(deliver(first));
  ASSERT_TRUE(deliver(victim));
  ASSERT_EQ(scheduler_->prepare_batch_test()[0].size(), 1u);
  ASSERT_TRUE(victim->preempted());
  auto* pool =
      dynamic_cast<RestoreBlockManager*>(engine_->block_manager_pool());
  ASSERT_NE(pool, nullptr);
  pool->set_pending_release(true);
  first->set_cancel();
  auto step = std::async(std::launch::async,
                         [this] { scheduler_->step(absl::Milliseconds(5)); });
  const std::future_status status = step.wait_for(std::chrono::seconds(1));
  // Unblock the call even when the deadline assertion fails.
  victim->set_cancel();
  pool->set_pending_release(false);
  step.get();
  EXPECT_EQ(status, std::future_status::ready);
}

TEST_F(HostRestoreSchedulerTest, RestoresFifoDespiteBusinessPriority) {
  auto [first, earlier, later] = preempt_pair();
  ASSERT_TRUE(earlier->preempted());
  ASSERT_TRUE(later->preempted());
  auto probe = make_request({21, 22}, "probe");
  probe->sequences()[0]->set_dp_rank(1);
  for (int32_t step = 0;
       step < 8 && earlier->sequences()[0]->is_prefill_stage();
       ++step) {
    auto batches = scheduler_->prepare_batch_test();
    ASSERT_EQ(batches[0].size(), 1u);
    EXPECT_EQ(batches[0][0], earlier->sequences()[0].get());
    EXPECT_TRUE(batches[1].empty());
    EXPECT_EQ(scheduler_->try_admit(probe), DecodeAdmissionResult::RETRY_LATER);
    advance(batches);
  }
  ASSERT_FALSE(earlier->sequences()[0]->is_prefill_stage());
  // `later` needs 4 of the 6 blocks while `earlier` holds 3, so it restores
  // as soon as `earlier` releases them.
  bool later_restored = false;
  for (int32_t step = 0; step < 8 && !later_restored; ++step) {
    auto batches = scheduler_->prepare_batch_test();
    ASSERT_EQ(batches[0].size(), 1u);
    later_restored = batches[0][0] == later->sequences()[0].get();
    EXPECT_TRUE(later_restored || !earlier->finished());
    advance(batches);
  }
  EXPECT_TRUE(later_restored);
  EXPECT_TRUE(earlier->finished());
}

TEST_F(HostRestoreSchedulerTest,
       OpensWhenLastWaiterIsCancelledAfterCurrentReady) {
  auto [first, earlier, later] = preempt_pair();
  ASSERT_TRUE(earlier->preempted());
  ASSERT_TRUE(later->preempted());
  for (int32_t step = 0;
       step < 8 && earlier->sequences()[0]->is_prefill_stage();
       ++step) {
    auto batches = scheduler_->prepare_batch_test();
    ASSERT_EQ(batches[0].size(), 1u);
    ASSERT_EQ(batches[0][0], earlier->sequences()[0].get());
    advance(batches);
  }
  ASSERT_FALSE(earlier->sequences()[0]->is_prefill_stage());
  ASSERT_FALSE(earlier->finished());
  later->set_cancel();
  auto batches = scheduler_->prepare_batch_test();
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0], earlier->sequences()[0].get());
  auto probe = reserve("probe", /*rank=*/1, {21, 22});
  EXPECT_TRUE(scheduler_->release_reservation(
      probe->request_id(), probe->state().pd_reservation_id));
}

TEST_F(HostRestoreSchedulerTest, FailsRestoreHeadOnIdleRankThenContinuesFifo) {
  auto [first, earlier, later] = preempt_pair();
  ASSERT_TRUE(earlier->preempted());
  ASSERT_TRUE(later->preempted());
  auto error = std::make_shared<std::promise<Status>>();
  auto result = error->get_future();
  earlier->state().output_func = [error](const RequestOutput& output) {
    error->set_value(output.status.value());
    return true;
  };
  auto* pool =
      dynamic_cast<RestoreBlockManager*>(engine_->block_manager_pool());
  ASSERT_NE(pool, nullptr);
  // Nothing else holds rank-0 blocks, so a failed restore cannot wait.
  pool->fail_allocate_for(earlier->sequences()[0].get());
  EXPECT_TRUE(scheduler_->prepare_batch_test()[0].empty());
  ASSERT_EQ(result.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_EQ(result.get().code(), StatusCode::RESOURCE_EXHAUSTED);
  EXPECT_FALSE(earlier->sequences()[0]->kv_state().has_any_blocks());
  auto batches = scheduler_->prepare_batch_test();
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0], later->sequences()[0].get());
}

TEST_F(HostRestoreSchedulerTest, RequeuesBlockedRestoreContinuation) {
  auto first = reserve("first", /*rank=*/0, {1, 2, 3, 4});
  auto victim = reserve("victim", /*rank=*/0, {5, 6, 7, 8});
  auto error = std::make_shared<std::promise<Status>>();
  auto result = error->get_future();
  victim->state().output_func = [error](const RequestOutput& output) {
    error->set_value(output.status.value());
    return true;
  };
  ASSERT_TRUE(deliver(first));
  ASSERT_TRUE(deliver(victim));
  ASSERT_EQ(scheduler_->prepare_batch_test()[0].size(), 1u);
  first->set_cancel();
  engine_->block_manager_pool()->reset_prefix_cache();
  {
    auto batches = scheduler_->prepare_batch_test();
    ASSERT_EQ(batches[0].size(), 1u);
    advance(batches);
  }
  ASSERT_TRUE(victim->sequences()[0]->is_prefill_stage());
  auto retained = make_request(std::vector<int32_t>(6, 30), "retained");
  retained->sequences()[0]->set_dp_rank(0);
  ASSERT_TRUE(
      engine_->block_manager_pool()->allocate(retained->sequences()[0].get()));

  // The continuation goes back to the restore queue instead of failing.
  EXPECT_TRUE(scheduler_->prepare_batch_test()[0].empty());
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  EXPECT_TRUE(victim->preempted());
  EXPECT_FALSE(victim->sequences()[0]->kv_state().has_any_blocks());

  engine_->block_manager_pool()->deallocate(retained.get());
  auto batches = scheduler_->prepare_batch_test();
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0], victim->sequences()[0].get());
}

TEST_F(HostRestoreSchedulerTest, WaitsForAsyncReleaseAfterAllRequestsCancel) {
  auto first = reserve("first", /*rank=*/0, {1, 2, 3, 4});
  auto victim = reserve("victim", /*rank=*/0, {5, 6, 7, 8});
  ASSERT_TRUE(deliver(first));
  ASSERT_TRUE(deliver(victim));
  ASSERT_EQ(scheduler_->prepare_batch_test()[0].size(), 1u);
  auto* pool =
      dynamic_cast<RestoreBlockManager*>(engine_->block_manager_pool());
  ASSERT_NE(pool, nullptr);
  pool->set_pending_release(true);
  first->set_cancel();
  victim->set_cancel();
  EXPECT_TRUE(scheduler_->prepare_batch_test()[0].empty());
  auto probe = make_request({21, 22}, "probe");
  probe->state().pd_reservation_id = "probe-reservation";
  probe->sequences()[0]->set_dp_rank(1);
  EXPECT_EQ(scheduler_->try_admit(probe), DecodeAdmissionResult::RETRY_LATER);
  pool->set_pending_release(false);
  EXPECT_TRUE(scheduler_->prepare_batch_test()[0].empty());
  ASSERT_EQ(scheduler_->try_admit(probe), DecodeAdmissionResult::ADMITTED);
  ASSERT_TRUE(scheduler_->decode_schedule(probe, "prefill"));
  EXPECT_TRUE(scheduler_->release_reservation("probe", "probe-reservation"));
}

TEST_F(HostRestoreSchedulerTest, RepausesAdmissionOnNextPreemption) {
  auto first = reserve("first", /*rank=*/0, {1, 2, 3, 4});
  auto victim = reserve("victim", /*rank=*/0, {5, 6, 7, 8});
  ASSERT_TRUE(deliver(first));
  ASSERT_TRUE(deliver(victim));
  ASSERT_EQ(scheduler_->prepare_batch_test()[0].size(), 1u);
  first->set_cancel();
  engine_->block_manager_pool()->reset_prefix_cache();
  for (int32_t step = 0; step < 8 && victim->sequences()[0]->is_prefill_stage();
       ++step) {
    auto batches = scheduler_->prepare_batch_test();
    ASSERT_EQ(batches[0].size(), 1u);
    advance(batches);
  }
  ASSERT_FALSE(victim->sequences()[0]->is_prefill_stage());
  {
    auto batches = scheduler_->prepare_batch_test();
    ASSERT_EQ(batches[0].size(), 1u);
    advance(batches);
  }
  ASSERT_FALSE(victim->finished());
  auto next = reserve("next", /*rank=*/0, {31, 32});
  ASSERT_TRUE(deliver(next));
  auto batches = scheduler_->prepare_batch_test();
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0], victim->sequences()[0].get());
  ASSERT_TRUE(next->preempted());
  auto probe = make_request({21, 22}, "probe");
  probe->sequences()[0]->set_dp_rank(1);
  EXPECT_EQ(scheduler_->try_admit(probe), DecodeAdmissionResult::RETRY_LATER);
}

class CancelRestoreTest : public HostRestoreSchedulerTest,
                          public ::testing::WithParamInterface<bool> {};

TEST_P(CancelRestoreTest, CancelsRestoredRequestAndContinuesFifo) {
  auto [first, earlier, later] = preempt_pair();
  ASSERT_TRUE(earlier->preempted());
  ASSERT_TRUE(later->preempted());
  engine_->block_manager_pool()->reset_prefix_cache();
  {
    auto batches = scheduler_->prepare_batch_test();
    ASSERT_EQ(batches[0].size(), 1u);
    ASSERT_EQ(batches[0][0], earlier->sequences()[0].get());
    advance(batches);
  }
  ASSERT_TRUE(earlier->sequences()[0]->is_prefill_stage());
  if (GetParam()) {
    for (int32_t step = 0;
         step < 8 && earlier->sequences()[0]->is_prefill_stage();
         ++step) {
      auto batches = scheduler_->prepare_batch_test();
      ASSERT_EQ(batches[0].size(), 1u);
      ASSERT_EQ(batches[0][0], earlier->sequences()[0].get());
      advance(batches);
    }
    ASSERT_FALSE(earlier->sequences()[0]->is_prefill_stage());
    ASSERT_FALSE(earlier->finished());
  }
  earlier->set_cancel();
  auto batches = scheduler_->prepare_batch_test();
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0], later->sequences()[0].get());
  EXPECT_FALSE(earlier->sequences()[0]->kv_state().has_any_blocks());
}

INSTANTIATE_TEST_SUITE_P(RestoreOrGeneration,
                         CancelRestoreTest,
                         ::testing::Bool());

TEST_F(HostRestoreSchedulerTest, FailedAdmissionHandoffDoesNotBlockRestore) {
  auto first = reserve("first", /*rank=*/0, {1, 2, 3, 4});
  auto victim = reserve("victim", /*rank=*/0, {5, 6, 7, 8});
  auto reservation = reserve("reservation", /*rank=*/1, {11, 12});
  auto duplicate = make_request({21, 22}, "reservation");
  duplicate->sequences()[0]->set_dp_rank(1);
  ASSERT_EQ(scheduler_->try_admit(duplicate), DecodeAdmissionResult::ADMITTED);
  ASSERT_TRUE(deliver(first));
  ASSERT_TRUE(deliver(victim));
  ASSERT_EQ(scheduler_->prepare_batch_test()[0].size(), 1u);
  ASSERT_TRUE(victim->preempted());
  EXPECT_FALSE(scheduler_->decode_schedule(duplicate, "prefill"));
  EXPECT_EQ(engine_->block_manager_pool()->num_used_blocks()[1], 1u);
  first->set_cancel();
  auto batches = scheduler_->prepare_batch_test();
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0], victim->sequences()[0].get());
  ASSERT_TRUE(scheduler_->release_reservation(
      reservation->request_id(), reservation->state().pd_reservation_id));
}

TEST_F(HostRestoreSchedulerTest, DecodeReserveGatesAdmission) {
  for (int32_t reserved_tokens : {0, 8}) {
    RestoreConfigGuard<int32_t> reserve_guard(
        SchedulerConfig::get_instance().decode_reserved_tokens_per_request(),
        reserved_tokens);
    init_scheduler(/*block_size=*/2, /*speculative_tokens=*/0);
    auto first = reserve("first", /*rank=*/0, {1, 2, 3, 4});
    ASSERT_TRUE(deliver(first));
    ASSERT_EQ(scheduler_->prepare_batch_test()[0].size(), 1u);
    const auto used = engine_->block_manager_pool()->num_used_blocks();

    // Rank 0 has 1 free block. With the reserve, `first` keeps 2 blocks to
    // finish its 3 remaining tokens, so a 1-block arrival must retry later.
    auto probe = make_request({21, 22}, "probe");
    probe->sequences()[0]->set_dp_rank(0);
    const DecodeAdmissionResult admission = scheduler_->try_admit(probe);
    EXPECT_EQ(admission,
              reserved_tokens == 0 ? DecodeAdmissionResult::ADMITTED
                                   : DecodeAdmissionResult::RETRY_LATER)
        << "reserved_tokens=" << reserved_tokens;
    if (admission == DecodeAdmissionResult::ADMITTED) {
      engine_->block_manager_pool()->deallocate_without_cache(
          probe->sequences()[0].get());
    }
    EXPECT_EQ(engine_->block_manager_pool()->num_used_blocks(), used);
    first->set_cancel();
    scheduler_->prepare_batch_test();
  }
}

TEST_F(HostRestoreSchedulerTest, ReservationsCountTowardDecodeReserve) {
  RestoreConfigGuard<int32_t> reserve_guard(
      SchedulerConfig::get_instance().decode_reserved_tokens_per_request(), 8);
  init_scheduler(/*block_size=*/2, /*speculative_tokens=*/0);
  // Registered but not yet delivered: holds 1 block and, as the first request
  // on its rank, keeps 1 more for its next decode step.
  auto accepted = reserve("accepted", /*rank=*/1, {11, 12});
  scheduler_->prepare_batch_test();
  auto probe = make_request({21, 22}, "probe");
  probe->sequences()[0]->set_dp_rank(1);
  EXPECT_EQ(scheduler_->try_admit(probe), DecodeAdmissionResult::RETRY_LATER);
  EXPECT_EQ(engine_->block_manager_pool()->num_used_blocks()[1], 1u);
  EXPECT_TRUE(scheduler_->release_reservation(
      accepted->request_id(), accepted->state().pd_reservation_id));
}

TEST_F(HostRestoreSchedulerTest, AdmittedRequestKeepsReserveUntilScheduled) {
  RestoreConfigGuard<int32_t> reserve_guard(
      SchedulerConfig::get_instance().decode_reserved_tokens_per_request(), 8);
  init_scheduler(/*block_size=*/2, /*speculative_tokens=*/0);
  // Admitted but not yet handed to the scheduler: a publish in between must
  // not drop its reserve.
  auto admitted = make_request({11, 12}, "admitted");
  admitted->sequences()[0]->set_dp_rank(1);
  ASSERT_EQ(scheduler_->try_admit(admitted), DecodeAdmissionResult::ADMITTED);
  scheduler_->prepare_batch_test();
  auto probe = make_request({21, 22}, "probe");
  probe->sequences()[0]->set_dp_rank(1);
  EXPECT_EQ(scheduler_->try_admit(probe), DecodeAdmissionResult::RETRY_LATER);

  // Once the admitted request is dropped, its reserve goes with it.
  engine_->block_manager_pool()->deallocate_without_cache(
      admitted->sequences()[0].get());
  admitted.reset();
  scheduler_->prepare_batch_test();
  EXPECT_EQ(scheduler_->try_admit(probe), DecodeAdmissionResult::ADMITTED);
  engine_->block_manager_pool()->deallocate_without_cache(
      probe->sequences()[0].get());
}

TEST_F(HostRestoreSchedulerTest, IdleRankKeepsRoomForNextMtpStep) {
  RestoreConfigGuard<int32_t> reserve_guard(
      SchedulerConfig::get_instance().decode_reserved_tokens_per_request(), 8);
  init_scheduler(/*block_size=*/2, /*speculative_tokens=*/3);
  // 8 prompt tokens fill the 4 usable blocks, leaving none for the sampled
  // token and the 3 speculative tokens of the first decode step.
  auto full = make_request(std::vector<int32_t>(8, 1), "full");
  full->sequences()[0]->set_dp_rank(0);
  EXPECT_EQ(scheduler_->try_admit(full), DecodeAdmissionResult::RETRY_LATER);
  EXPECT_EQ(engine_->block_manager_pool()->num_used_blocks()[0], 0u);

  auto fits = make_request(std::vector<int32_t>(4, 1), "fits");
  fits->sequences()[0]->set_dp_rank(0);
  EXPECT_EQ(scheduler_->try_admit(fits), DecodeAdmissionResult::ADMITTED);
  engine_->block_manager_pool()->deallocate_without_cache(
      fits->sequences()[0].get());
}

class HostRestoreReadyTest
    : public HostRestoreSchedulerTest,
      public ::testing::WithParamInterface<std::tuple<int32_t, int32_t>> {
 protected:
  void SetUp() override {
    split_ = std::make_unique<RestoreConfigGuard<int32_t>>(
        ParallelConfig::get_instance().kv_split_size(),
        std::get<1>(GetParam()));
    init_scheduler(/*block_size=*/16, std::get<0>(GetParam()));
  }
  std::unique_ptr<RestoreConfigGuard<int32_t>> split_;
};

TEST_P(HostRestoreReadyTest, OpensAdmissionAfterFinalChunkCompletes) {
  auto first = reserve("first", /*rank=*/0, std::vector<int32_t>(32, 1));
  auto victim = reserve("victim", /*rank=*/0, std::vector<int32_t>(32, 2));
  ASSERT_TRUE(deliver(first));
  ASSERT_TRUE(deliver(victim));
  ASSERT_EQ(scheduler_->prepare_batch_test()[0].size(), 1u);
  ASSERT_TRUE(victim->preempted());
  first->set_cancel();
  engine_->block_manager_pool()->reset_prefix_cache();
  auto incoming = make_request({21, 22}, "incoming");
  incoming->state().pd_reservation_id = "probe-reservation";
  incoming->sequences()[0]->set_dp_rank(1);

  for (uint32_t chunk_tokens : {16u, 16u, 1u}) {
    auto batches = scheduler_->prepare_batch_test();
    ASSERT_EQ(batches[0].size(), 1u);
    EXPECT_TRUE(batches[1].empty());
    EXPECT_EQ(batches[0][0], victim->sequences()[0].get());
    EXPECT_EQ(batches[0].get_allowed_max_tokens(),
              (std::vector<uint32_t>{chunk_tokens}));
    EXPECT_EQ(scheduler_->try_admit(incoming),
              DecodeAdmissionResult::RETRY_LATER);
    victim->sequences()[0]->kv_state().incr_kv_cache_tokens_num(chunk_tokens);
  }
  ASSERT_FALSE(victim->sequences()[0]->is_prefill_stage());
  victim->sequences()[0]->append_token(Token(10));
  auto batches = scheduler_->prepare_batch_test();
  ASSERT_EQ(batches[0].size(), 1u);
  EXPECT_EQ(batches[0][0], victim->sequences()[0].get());
  EXPECT_FALSE(victim->finished());
  ASSERT_EQ(scheduler_->try_admit(incoming), DecodeAdmissionResult::ADMITTED);
  ASSERT_TRUE(scheduler_->decode_schedule(incoming, "prefill"));
  EXPECT_TRUE(scheduler_->release_reservation("incoming", "probe-reservation"));
}

INSTANTIATE_TEST_SUITE_P(MtpAndKvSplit,
                         HostRestoreReadyTest,
                         ::testing::Combine(::testing::Values(0, 3),
                                            ::testing::Values(1, 8)));

}  // namespace
}  // namespace xllm
