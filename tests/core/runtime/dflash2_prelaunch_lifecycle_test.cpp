/* Copyright 2026 The xLLM Authors.

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
#include <torch/torch.h>

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/framework/batch/batch.h"
#include "core/framework/kv_cache_transfer/kv_transfer_completion.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/speculative/dflash_async_input_builder.h"
#include "framework/block/block_manager_impl.h"
#include "framework/request/stopping_checker.h"
#include "layers/common/attention_metadata.h"
#include "platform/device.h"
#include "runtime/dflash/dflash2_worker_impl.h"
#include "runtime/dspark_worker_impl.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"

namespace xllm {

class DFlash2WorkerImplTestPeer final {
 public:
  static void install_pending(DFlash2WorkerImpl& worker,
                              const torch::Tensor& accepted,
                              StreamEventPtr ready,
                              bool graph_enabled) {
    worker.prelaunch_graph_enabled_ = graph_enabled;
    worker.pending_draft_.emplace();
    worker.pending_draft_->accepted_host = accepted;
    worker.pending_draft_->ready = std::move(ready);
  }

  static void finish(DFlash2WorkerImpl& worker, const ForwardInput& input) {
    worker.finish_draft_prelaunch(input);
  }

  static void install_identity(DFlash2WorkerImpl& worker,
                               ForwardInput identity,
                               const torch::Tensor& accepted,
                               StreamEventPtr ready) {
    install_pending(
        worker, accepted, std::move(ready), /*graph_enabled=*/false);
    worker.prelaunch_enabled_ = true;
    worker.pending_draft_->identity = std::move(identity);
  }

  static bool matches(const DFlash2WorkerImpl& worker,
                      const ForwardInput& input) {
    return worker.pending_draft_matches(input);
  }

  static ForwardInput consume_target(DFlash2WorkerImpl& worker,
                                     const ForwardInput& input,
                                     ForwardInput target) {
    worker.pending_draft_->target_input = std::move(target);
    ForwardInput refreshed;
    worker.run_decode_draft(input, refreshed);
    CHECK_EQ(worker.prepare_stream_->synchronize(), 0);
    return refreshed;
  }
};

class DSparkWorkerImplTestPeer final {
 public:
  static void install_pending(DSparkWorkerImpl& worker,
                              ForwardInput identity,
                              StreamEventPtr ready) {
    worker.async_output_enabled_ = true;
    worker.draft_prelaunch_enabled_ = true;
    worker.pending_context_.emplace();
    worker.pending_context_->identity = std::move(identity);
    worker.pending_context_->d2h_ready = ready;
    worker.pending_context_->context_ready = std::move(ready);
  }

  static bool can_use_step_local(const DSparkWorkerImpl& worker,
                                 const ForwardInput& input) {
    return worker.can_prelaunch_draft(input);
  }

  static std::vector<int64_t> pending_rows(const DSparkWorkerImpl& worker,
                                           const ForwardInput& input) {
    return worker.build_batch_to_pending_row_mapping(input);
  }

  static void install_target_context(DSparkWorkerImpl& worker,
                                     ForwardInput identity,
                                     const torch::Tensor& accepted_device,
                                     const torch::Tensor& accepted_host,
                                     const torch::Tensor& embeddings,
                                     const torch::Tensor& positions,
                                     StreamEventPtr ready) {
    install_pending(worker, std::move(identity), std::move(ready));
    worker.pending_context_->accepted_device = accepted_device;
    worker.pending_context_->accepted_host = accepted_host;
    worker.pending_context_->embeddings = embeddings;
    worker.pending_context_->base_positions = positions;
  }

  static void install_embedding_cache(DSparkWorkerImpl& worker,
                                      std::shared_ptr<EmbeddingCache> cache) {
    worker.embedding_cache_ = std::move(cache);
  }

  static void commit(DSparkWorkerImpl& worker) {
    worker.commit_pending_context();
  }

  static bool has_pending(const DSparkWorkerImpl& worker) {
    return worker.pending_context_.has_value();
  }

  static size_t consumed_tensor_count(const DSparkWorkerImpl& worker) {
    return worker.consumed_context_tensors_.size();
  }
};

namespace {

class OverlapResultWorker final : public WorkerImpl {
 public:
  OverlapResultWorker(const ParallelArgs& parallel_args,
                      const torch::Device& device,
                      const runtime::Options& options)
      : WorkerImpl(parallel_args, device, options) {
    auto guard = compute_stream_->set_stream_guard();
    const torch::TensorOptions tensor_options =
        torch::TensorOptions().device(device).dtype(torch::kBFloat16);
    lhs_ = torch::ones({4096, 4096}, tensor_options);
    rhs_ = torch::ones_like(lhs_);
    scratch_ = torch::empty_like(lhs_);
    compute_stream_->synchronize();
  }

  bool init_model(ModelContext& context) override { return false; }

  void prepare_work_before_execute(const ForwardInput& input,
                                   ForwardInput& prepared) override {
    prepared = input;
  }

  std::optional<ForwardOutput> step(const ForwardInput& input) override {
    ADD_FAILURE() << "synchronous execution is outside this fixture";
    return std::nullopt;
  }

  bool wait_for_submission(int64_t step_id) const {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (submitted_step_.load(std::memory_order_acquire) != step_id) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
  }

  bool has_stored_output() const { return last_step_output_valid_; }

  bool has_recorded_output() const { return is_recorded_; }

 protected:
  std::optional<ForwardOutput> step_for_schedule_overlap(
      ForwardInput& input) override {
    if (input.input_params.meta.batch_forward_type.is_empty()) {
      return std::nullopt;
    }
    const int64_t step_id = input.positions_host.item<int64_t>();
    auto guard = compute_stream_->set_stream_guard();
    const torch::TensorOptions token_options =
        torch::TensorOptions().device(device_.unwrap()).dtype(torch::kLong);
    torch::Tensor tokens = torch::full({1, 1}, -777, token_options);
    torch::Tensor host_tokens = torch::full(
        {1, 1},
        -777,
        torch::TensorOptions().dtype(torch::kLong).pinned_memory(true));
    auto retained = std::make_shared<ForwardInput>(input);
    retained->token_ids = tokens;
    retained->token_ids_host = host_tokens;
    if (!driver_ || step_id % 2 == 0) {
      for (int32_t iteration = 0; iteration < 128; ++iteration) {
        torch::mm_out(scratch_, lhs_, rhs_);
      }
    }
    tokens.fill_(step_id);
    host_tokens.copy_(tokens, true);
    ForwardOutput output;
    output.sample_output.next_tokens = std::move(tokens);
    output.retained_inputs.push_back(std::move(retained));
    output.ready_event = compute_stream_->record_event();
    output.requires_host_retirement = true;
    submitted_step_.store(step_id, std::memory_order_release);
    return output;
  }

 private:
  torch::Tensor lhs_;
  torch::Tensor rhs_;
  torch::Tensor scratch_;
  std::atomic<int64_t> submitted_step_{-1};
};

ForwardInput make_overlap_producer_input(int64_t step_id) {
  ForwardInput input;
  input.token_ids = torch::tensor({step_id}, torch::kLong);
  input.token_ids_host = input.token_ids;
  input.positions = torch::zeros({1}, torch::kLong);
  input.positions_host = torch::tensor({step_id}, torch::kLong);
  input.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  input.input_params.meta.num_sequences = 1;
  input.input_params.embedding.embedding_ids = {0};
  input.input_params.embedding.request_ids = {"request-" +
                                              std::to_string(step_id)};
  input.sample_sequence_ids = {"sequence-" + std::to_string(step_id)};
  return input;
}

ForwardInput make_overlap_empty_input() {
  ForwardInput input;
  input.token_ids = torch::empty({0}, torch::kLong);
  input.input_params.meta.batch_forward_type = BatchForwardType::EMPTY;
  return input;
}

class WorkerOverlapResultSlotTest : public ::testing::TestWithParam<int32_t> {};

TEST_P(WorkerOverlapResultSlotTest,
       FullSlotKeepsCurrentStepUntilConsumerDequeues) {
  Device device(GetParam());
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 2, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .enable_speculative_decode(true)
      .num_speculative_tokens(15);
  OverlapResultWorker worker(parallel_args, device.unwrap(), options);
  int32_t full_slot_windows = 0;
  int32_t dequeued_not_ready_windows = 0;
  for (int64_t iteration = 0; iteration < 1000; ++iteration) {
    const int64_t step_id = iteration * 2;
    auto current = worker.step_async(make_overlap_producer_input(step_id));
    ASSERT_TRUE(std::move(current).get().has_value());
    auto next = worker.step_async(make_overlap_producer_input(step_id + 1));
    EXPECT_TRUE(worker.wait_for_submission(step_id + 1));
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!next.isReady()) {
      ++full_slot_windows;
    }
    EXPECT_FALSE(next.isReady());
    ForwardOutput result = worker.get_last_step_result();
    ASSERT_NE(result.ready_event, nullptr);
    ASSERT_EQ(result.retained_inputs.size(), 1);
    aclrtEventRecordedStatus status;
    ASSERT_EQ(aclrtQueryEventStatus(result.ready_event->npu_event(), &status),
              ACL_SUCCESS);
    if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
      ++dequeued_not_ready_windows;
    }
    EXPECT_EQ(
        result.retained_inputs.front()->input_params.embedding.request_ids,
        std::vector<std::string>{"request-" + std::to_string(step_id)});
    EXPECT_EQ(result.retained_inputs.front()->sample_sequence_ids,
              std::vector<std::string>{"sequence-" + std::to_string(step_id)});
    EXPECT_TRUE(result.ready_event->synchronize());
    EXPECT_EQ(result.retained_inputs.front()->token_ids_host.item<int64_t>(),
              step_id);
    ASSERT_TRUE(std::move(next).get().has_value());
    ForwardOutput next_result = worker.get_last_step_result();
    ASSERT_NE(next_result.ready_event, nullptr);
    EXPECT_TRUE(next_result.ready_event->synchronize());
    EXPECT_EQ(
        next_result.retained_inputs.front()->token_ids_host.item<int64_t>(),
        step_id + 1);
  }
  EXPECT_EQ(full_slot_windows, 1000);
  EXPECT_EQ(dequeued_not_ready_windows, 1000);
  ::testing::Test::RecordProperty("full_slot_windows", full_slot_windows);
  ::testing::Test::RecordProperty("dequeued_not_ready_windows",
                                  dequeued_not_ready_windows);
  ::testing::Test::RecordProperty("logical_rank", 0);
  ::testing::Test::RecordProperty("local_device_index", GetParam());
  ::testing::Test::RecordProperty("scope",
                                  "real WorkerImpl publication/dequeue; "
                                  "bounded producer, no Service RPC or model");
}

TEST_P(WorkerOverlapResultSlotTest,
       NonDriverOverwritesAndClearsAfterOwnDeviceTail) {
  Device device(GetParam());
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(1, 2, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .enable_speculative_decode(true)
      .num_speculative_tokens(15);
  OverlapResultWorker worker(parallel_args, device.unwrap(), options);
  auto previous =
      std::move(worker.step_async(make_overlap_producer_input(0))).get();
  ASSERT_TRUE(previous.has_value());
  int32_t overwrite_not_ready_windows = 0;
  int32_t clear_not_ready_windows = 0;
  for (int64_t step_id = 0; step_id < 1000; ++step_id) {
    const StreamEventPtr previous_ready = previous->ready_event;
    const std::weak_ptr<ForwardInput> previous_owner =
        previous->retained_inputs.front();
    const torch::Tensor previous_host =
        previous->retained_inputs.front()->token_ids_host;
    previous.reset();
    EXPECT_FALSE(previous_owner.expired());
    aclrtEventRecordedStatus status;
    ASSERT_EQ(aclrtQueryEventStatus(previous_ready->npu_event(), &status),
              ACL_SUCCESS);
    if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
      ++overwrite_not_ready_windows;
    }
    previous =
        std::move(worker.step_async(make_overlap_producer_input(step_id + 1)))
            .get();
    ASSERT_TRUE(previous.has_value());
    ASSERT_EQ(aclrtQueryEventStatus(previous_ready->npu_event(), &status),
              ACL_SUCCESS);
    EXPECT_EQ(status, ACL_EVENT_RECORDED_STATUS_COMPLETE);
    EXPECT_TRUE(previous_owner.expired());
    EXPECT_EQ(previous_host.item<int64_t>(), step_id);
    EXPECT_FALSE(worker.has_recorded_output());
  }
  for (int64_t step_id = 1000; step_id < 2000; ++step_id) {
    const StreamEventPtr previous_ready = previous->ready_event;
    const std::weak_ptr<ForwardInput> previous_owner =
        previous->retained_inputs.front();
    const torch::Tensor previous_host =
        previous->retained_inputs.front()->token_ids_host;
    previous.reset();
    EXPECT_FALSE(previous_owner.expired());
    aclrtEventRecordedStatus status;
    ASSERT_EQ(aclrtQueryEventStatus(previous_ready->npu_event(), &status),
              ACL_SUCCESS);
    if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
      ++clear_not_ready_windows;
    }
    const auto cleared =
        std::move(worker.step_async(make_overlap_empty_input())).get();
    EXPECT_FALSE(cleared.has_value());
    ASSERT_EQ(aclrtQueryEventStatus(previous_ready->npu_event(), &status),
              ACL_SUCCESS);
    EXPECT_EQ(status, ACL_EVENT_RECORDED_STATUS_COMPLETE);
    EXPECT_TRUE(previous_owner.expired());
    EXPECT_EQ(previous_host.item<int64_t>(), step_id);
    EXPECT_FALSE(worker.has_stored_output());
    EXPECT_FALSE(worker.has_recorded_output());
    if (step_id < 1999) {
      previous =
          std::move(worker.step_async(make_overlap_producer_input(step_id + 1)))
              .get();
      ASSERT_TRUE(previous.has_value());
    }
  }
  EXPECT_EQ(overwrite_not_ready_windows, 1000);
  EXPECT_EQ(clear_not_ready_windows, 1000);
  ::testing::Test::RecordProperty("overwrite_not_ready_windows",
                                  overwrite_not_ready_windows);
  ::testing::Test::RecordProperty("clear_not_ready_windows",
                                  clear_not_ready_windows);
  ::testing::Test::RecordProperty("logical_rank", 1);
  ::testing::Test::RecordProperty("local_device_index", GetParam());
  ::testing::Test::RecordProperty(
      "scope",
      "real WorkerImpl non-driver replacement/empty retirement; no HCCL or "
      "model");
}

TEST_P(WorkerOverlapResultSlotTest,
       NonDriverIdleTailKeepsOwnershipWithoutAnotherModelStep) {
  Device device(GetParam());
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(1, 2, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .enable_speculative_decode(true)
      .num_speculative_tokens(15);
  OverlapResultWorker worker(parallel_args, device.unwrap(), options);
  int32_t idle_not_ready_windows = 0;
  for (int64_t step_id = 0; step_id < 100; ++step_id) {
    auto output =
        std::move(worker.step_async(make_overlap_producer_input(step_id)))
            .get();
    ASSERT_TRUE(output.has_value());
    const StreamEventPtr ready = output->ready_event;
    const std::weak_ptr<ForwardInput> owner = output->retained_inputs.front();
    const torch::Tensor host = output->retained_inputs.front()->token_ids_host;
    output.reset();
    EXPECT_FALSE(owner.expired());
    EXPECT_TRUE(worker.has_stored_output());
    aclrtEventRecordedStatus status;
    ASSERT_EQ(aclrtQueryEventStatus(ready->npu_event(), &status), ACL_SUCCESS);
    if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
      ++idle_not_ready_windows;
    }
    EXPECT_TRUE(ready->synchronize());
    EXPECT_EQ(host.item<int64_t>(), step_id);
    EXPECT_FALSE(owner.expired());
    const auto cleared =
        std::move(worker.step_async(make_overlap_empty_input())).get();
    EXPECT_FALSE(cleared.has_value());
    EXPECT_TRUE(owner.expired());
    EXPECT_FALSE(worker.has_stored_output());
  }
  EXPECT_EQ(idle_not_ready_windows, 100);
  ::testing::Test::RecordProperty("idle_not_ready_windows",
                                  idle_not_ready_windows);
  ::testing::Test::RecordProperty("logical_rank", 1);
  ::testing::Test::RecordProperty("local_device_index", GetParam());
  ::testing::Test::RecordProperty(
      "scope",
      "idle tail completes without a next model step; retirement uses real "
      "empty control call");
}

INSTANTIATE_TEST_SUITE_P(BothLocalDevices,
                         WorkerOverlapResultSlotTest,
                         ::testing::Values(0, 1));

TEST(DSparkPrelaunchDeviceInputTest, EagerUsesCurrentDeviceAcceptanceAndPages) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  torch::Tensor accepted_host = torch::full({2, 16}, -1, torch::kLong);
  accepted_host[0][0] = 21;
  accepted_host[1].copy_(torch::arange(30, 46, torch::kLong));
  torch::Tensor base_host = torch::tensor({113, 112}, torch::kLong);
  torch::Tensor table_host = torch::tensor({{3, 1}, {0, 2}}, torch::kInt32);
  torch::Tensor accepted = accepted_host.to(device.unwrap());
  torch::Tensor base = base_host.to(device.unwrap());
  torch::Tensor table = table_host.to(device.unwrap());
  auto stream = device.get_stream_from_pool();
  stream->wait_stream(*device.current_stream());
  auto guard = stream->set_stream_guard();
  const auto initial_result =
      dflash_async::prepare_next_draft(accepted, base, table, 999, 128, 15);
  accepted_host[0].copy_(torch::arange(50, 66, torch::kLong));
  accepted_host[1].fill_(-1);
  accepted_host[1][0] = 71;
  base_host = torch::tensor({112, 113}, torch::kLong);
  table_host = torch::tensor({{2, 0}, {1, 3}}, torch::kInt32);
  accepted.copy_(accepted_host.to(device.unwrap()));
  base.copy_(base_host.to(device.unwrap()));
  table.copy_(table_host.to(device.unwrap()));
  const auto result =
      dflash_async::prepare_next_draft(accepted, base, table, 999, 128, 15);
  ASSERT_EQ(stream->synchronize(), 0);
  EXPECT_TRUE(torch::equal(initial_result.anchor_tokens.cpu(),
                           torch::tensor({21, 45}, torch::kLong)));
  const auto expected = dflash_async::prepare_next_draft(
      accepted_host, base_host, table_host, 999, 128, 15);
  EXPECT_TRUE(
      torch::equal(result.accepted_counts.cpu(), expected.accepted_counts));
  EXPECT_TRUE(torch::equal(result.anchor_tokens.cpu(), expected.anchor_tokens));
  EXPECT_TRUE(torch::equal(result.query_tokens.cpu(), expected.query_tokens));
  EXPECT_TRUE(
      torch::equal(result.query_positions.cpu(), expected.query_positions));
  EXPECT_TRUE(torch::equal(result.query_slots.cpu(), expected.query_slots));
  EXPECT_TRUE(torch::equal(result.kv_lengths.cpu(), expected.kv_lengths));
  EXPECT_TRUE(torch::equal(result.context_slots.cpu(), expected.context_slots));
}

Sequence make_overlap_sequence(int32_t prompt_token,
                               RequestSamplingParam& sampling,
                               StoppingChecker& stopping) {
  SequenceParams params;
  params.seq_capacity = 128;
  params.sampling_param = &sampling;
  params.stopping_checker = &stopping;
  params.enable_schedule_overlap = true;
  IncrementalDecoder decoder(/*prompt=*/"",
                             /*num_prompt_tokens=*/1,
                             /*echo=*/false,
                             /*skip_special_tokens=*/true);
  return Sequence(/*index=*/0,
                  {prompt_token},
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  std::move(decoder),
                  params);
}

TEST(DFlash2PrelaunchTransferTest, NonSendingPushCompletionIsGraphCapturable) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  ProcessGroup tp_group(/*rank=*/0, /*world_size=*/2, device.unwrap());
  ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/2, /*process_group=*/nullptr);
  parallel_args.tp_group_ = &tp_group;
  auto marker = torch::zeros(
      {1}, torch::TensorOptions().device(device.unwrap()).dtype(torch::kInt));
  auto stream = device.get_stream_from_pool();
  stream->wait_stream(*device.current_stream());
  auto guard = stream->set_stream_guard();
  marker.add_(1);
  stream->synchronize();
  for (const InstanceRole role :
       {InstanceRole::DEFAULT, InstanceRole::DECODE}) {
    KVTransferCompletion completion;
    c10_npu::NPUGraph graph;
    graph.capture_begin(/*pool=*/{0, 0},
                        ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    const auto failures = finalize_kv_push_failures(
        completion, {}, "PUSH", role, parallel_args, device);
    marker.add_(1);
    graph.capture_end();
    EXPECT_TRUE(failures.empty());
    marker.zero_();
    graph.replay();
    stream->synchronize();
    EXPECT_EQ(marker.cpu().item<int32_t>(), 1);
  }
}

class DFlash2PrelaunchLifecycleTest : public ::testing::TestWithParam<bool> {};

TEST(DFlash2PrelaunchIdentityTest, ReusesTargetAfterLinearSlotRotation) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(7).block_size(
      128);
  DFlash2WorkerImpl worker(parallel_args, device.unwrap(), options);
  ForwardInput previous;
  previous.token_ids_host = torch::tensor({11}, torch::kInt32);
  previous.positions_host = torch::tensor({2048}, torch::kInt32);
  previous.input_params.meta.num_sequences = 1;
  previous.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  previous.input_params.embedding.embedding_ids = {1};
  previous.input_params.embedding.request_ids = {"continuing-request"};
  previous.input_params.embedding.linear_state_ids = {2};
  previous.input_params.embedding.linear_state_read_ids = {1};
  previous.input_params.attention.host.kv_seq_lens = {2049};
  previous.input_params.attention.host.block_tables =
      torch::arange(32, torch::kInt32).view({1, 32});
  const torch::Tensor accepted =
      torch::tensor({{31, -1, -1, -1, -1, -1, -1, -1}}, torch::kLong);
  DFlash2WorkerImplTestPeer::install_identity(
      worker,
      previous,
      accepted,
      device.current_stream()->record_event_or_sync());
  ForwardInput current = previous;
  current.token_ids_host = torch::tensor({31}, torch::kInt32);
  current.positions_host = torch::tensor({2049}, torch::kInt32);
  current.input_params.attention.host.kv_seq_lens = {2050};
  ASSERT_TRUE(DFlash2WorkerImplTestPeer::matches(worker, current));
  ForwardInput target = previous;
  target.input_params.embedding.linear_state_indices = torch::tensor(
      {2}, torch::TensorOptions().dtype(torch::kInt32).device(device.unwrap()));
  target.input_params.embedding.linear_state_read_indices = torch::tensor(
      {1}, target.input_params.embedding.linear_state_indices.options());
  target.input_params.attn_metadata =
      std::make_shared<layer::AttentionMetadata>();
  target.input_params.attn_metadata->linear_state_indices =
      target.input_params.embedding.linear_state_indices;
  target.metadata_ready_event = device.current_stream()->record_event_or_sync();
  const auto metadata = target.input_params.attn_metadata;
  const torch::Tensor token_ids = target.token_ids_host;
  // Read-only checkpoint rotation and write-slot rotation both preserve the
  // proposal. Only the confirmed target's state addressing needs refreshing.
  current.input_params.embedding.linear_state_read_ids = {2};
  EXPECT_TRUE(DFlash2WorkerImplTestPeer::matches(worker, current));
  current.input_params.embedding.linear_state_read_ids = {3};
  current.input_params.embedding.linear_state_ids = {4};
  ASSERT_TRUE(DFlash2WorkerImplTestPeer::matches(worker, current));
  ForwardInput refreshed = DFlash2WorkerImplTestPeer::consume_target(
      worker, current, std::move(target));
  EXPECT_EQ(refreshed.input_params.embedding.linear_state_ids,
            std::vector<int32_t>({4}));
  EXPECT_EQ(refreshed.input_params.embedding.linear_state_read_ids,
            std::vector<int32_t>({3}));
  EXPECT_EQ(refreshed.input_params.attn_metadata, metadata);
  EXPECT_EQ(refreshed.token_ids_host.data_ptr(), token_ids.data_ptr());
  EXPECT_EQ(
      refreshed.input_params.embedding.linear_state_indices.item<int32_t>(), 4);
  EXPECT_EQ(refreshed.input_params.embedding.linear_state_read_indices
                .item<int32_t>(),
            3);
  EXPECT_EQ(metadata->linear_state_indices.item<int32_t>(), 4);
}

TEST_P(DFlash2PrelaunchLifecycleTest,
       MixedEosBatchRetiresWritesBeforeBlockReuse) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(7);
  DFlash2WorkerImpl worker(parallel_args, device.unwrap(), options);
  constexpr std::array<BlockType, 6> kBlockTypes = {BlockType::KV,
                                                    BlockType::SWA,
                                                    BlockType::C4,
                                                    BlockType::C128,
                                                    BlockType::LINEAR,
                                                    BlockType::EMBEDDING};
  BlockManager::Options block_options;
  block_options.num_blocks(13).block_size(128).enable_prefix_cache(false);
  BlockManagerImpl manager(block_options);
  RequestSamplingParam sampling;
  StoppingChecker stopping;
  stopping.set_eos_token(2);
  auto ending = make_overlap_sequence(11, sampling, stopping);
  auto continuing = make_overlap_sequence(12, sampling, stopping);
  auto replacement = make_overlap_sequence(13, sampling, stopping);
  std::array<int32_t, kBlockTypes.size()> ending_ids;
  std::array<int32_t, kBlockTypes.size()> continuing_ids;
  for (size_t i = 0; i < kBlockTypes.size(); ++i) {
    auto ending_blocks = manager.allocate(1);
    auto continuing_blocks = manager.allocate(1);
    ASSERT_EQ(ending_blocks.size(), 1u);
    ASSERT_EQ(continuing_blocks.size(), 1u);
    ending_ids[i] = ending_blocks.front().id();
    continuing_ids[i] = continuing_blocks.front().id();
    ending.kv_state().add_blocks(kBlockTypes[i], std::move(ending_blocks));
    continuing.kv_state().add_blocks(kBlockTypes[i],
                                     std::move(continuing_blocks));
  }
  ASSERT_EQ(manager.num_free_blocks(), 0u);
  // Both the result batch and the next dispatched batch still reference these
  // sequences. Neither batch owns aliases that can delay block reuse.
  Batch result_batch(std::vector<Sequence*>{&ending, &continuing});
  Batch inflight_batch(std::vector<Sequence*>{&ending, &continuing});
  for (Sequence* sequence : result_batch.get_sequences()) {
    sequence->kv_state().set_kv_cache_tokens_num(1);
    sequence->append_token(Token(-1));
    sequence->append_token(Token(-1));
  }
  const auto tensor_options =
      torch::TensorOptions().device(device.unwrap()).dtype(torch::kBFloat16);
  auto cache = torch::zeros({13, 128}, tensor_options);
  auto lhs = torch::ones({4096, 4096}, tensor_options);
  auto rhs = torch::ones_like(lhs);
  auto scratch = torch::empty_like(lhs);
  auto compute = device.current_stream();
  compute->synchronize();
  StreamEventPtr ready;
  {
    auto guard = compute->set_stream_guard();
    // Keep real NPU work in flight so removing the completion fence fails
    // the event-state assertion, rather than depending on a lucky CPU race.
    for (int32_t i = 0; i < 128; ++i) {
      torch::mm_out(scratch, lhs, rhs);
    }
    for (size_t i = 0; i < kBlockTypes.size(); ++i) {
      cache[ending_ids[i]].fill_(7);
      cache[continuing_ids[i]].fill_(9);
    }
    ready = compute->record_event();
  }
  const auto accepted = torch::tensor(
      {{2, -1, -1, -1, -1, -1, -1, -1}, {31, 32, -1, -1, -1, -1, -1, -1}},
      torch::kLong);
  DFlash2WorkerImplTestPeer::install_pending(
      worker, accepted, ready, GetParam());
  ForwardInput input;
  // Graph mode can also skip next-target preparation (e.g. token filters).
  // That early return must not bypass retirement of the pending KV writes.
  input.sampling_params.filter_mask = torch::ones({2, 1}, torch::kBool);
  aclrtEventRecordedStatus before;
  ASSERT_EQ(aclrtQueryEventStatus(ready->npu_event(), &before), ACL_SUCCESS);
  ASSERT_EQ(before, ACL_EVENT_RECORDED_STATUS_NOT_READY);
  DFlash2WorkerImplTestPeer::finish(worker, input);
  aclrtEventRecordedStatus after;
  ASSERT_EQ(aclrtQueryEventStatus(ready->npu_event(), &after), ACL_SUCCESS);
  EXPECT_EQ(after, ACL_EVENT_RECORDED_STATUS_COMPLETE);

  const auto accepted_tokens = accepted.accessor<int64_t, 2>();
  ending.update_last_step_token(Token(accepted_tokens[0][0]));
  continuing.update_last_step_token(Token(accepted_tokens[1][0]));
  continuing.update_last_step_token(Token(accepted_tokens[1][1]),
                                    /*token_offset=*/1);
  ASSERT_TRUE(ending.finished());
  ASSERT_FALSE(continuing.finished());
  for (size_t i = 0; i < kBlockTypes.size(); ++i) {
    SCOPED_TRACE(static_cast<int32_t>(kBlockTypes[i]));
    (void)ending.kv_state().take_blocks(kBlockTypes[i]);
    auto blocks = manager.allocate(1);
    ASSERT_EQ(blocks.size(), 1u);
    ASSERT_EQ(blocks.front().id(), ending_ids[i]);
    replacement.kv_state().add_blocks(kBlockTypes[i], std::move(blocks));
    EXPECT_EQ(continuing.kv_state().blocks(kBlockTypes[i]).front().ref_count(),
              1u);
  }
  EXPECT_FALSE(inflight_batch.get_sequences().front()->has_any_blocks());
  auto reuse_stream = device.get_stream_from_pool();
  {
    auto guard = reuse_stream->set_stream_guard();
    for (const BlockType type : kBlockTypes) {
      cache[replacement.kv_state().blocks(type).front().id()].fill_(23);
    }
  }
  reuse_stream->synchronize();
  compute->synchronize();
  const auto host_cache = cache.to(torch::kCPU);
  for (size_t i = 0; i < kBlockTypes.size(); ++i) {
    SCOPED_TRACE(static_cast<int32_t>(kBlockTypes[i]));
    EXPECT_TRUE(host_cache[ending_ids[i]].eq(23).all().item<bool>());
    EXPECT_TRUE(host_cache[continuing_ids[i]].eq(9).all().item<bool>());
    (void)continuing.kv_state().take_blocks(kBlockTypes[i]);
    (void)replacement.kv_state().take_blocks(kBlockTypes[i]);
  }
  EXPECT_EQ(manager.num_free_blocks(), 12u);
}

INSTANTIATE_TEST_SUITE_P(EagerAndGraph,
                         DFlash2PrelaunchLifecycleTest,
                         ::testing::Bool());

TEST(DSparkStepLocalLifecycleTest, DestructionRetiresPendingContextWrites) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  auto worker = std::make_unique<DSparkWorkerImpl>(
      parallel_args, device.unwrap(), options);
  const torch::TensorOptions tensor_options =
      torch::TensorOptions().device(device.unwrap()).dtype(torch::kBFloat16);
  torch::Tensor lhs = torch::ones({4096, 4096}, tensor_options);
  torch::Tensor rhs = torch::ones_like(lhs);
  torch::Tensor scratch = torch::empty_like(lhs);
  torch::Tensor marker = torch::zeros({1}, tensor_options);
  auto compute = device.current_stream();
  ASSERT_EQ(compute->synchronize(), 0);
  StreamEventPtr ready;
  {
    auto guard = compute->set_stream_guard();
    for (int32_t iteration = 0; iteration < 128; ++iteration) {
      torch::mm_out(scratch, lhs, rhs);
    }
    marker.fill_(7);
    ready = compute->record_event();
  }
  DSparkWorkerImplTestPeer::install_pending(*worker, {}, ready);
  aclrtEventRecordedStatus before;
  ASSERT_EQ(aclrtQueryEventStatus(ready->npu_event(), &before), ACL_SUCCESS);
  ASSERT_EQ(before, ACL_EVENT_RECORDED_STATUS_NOT_READY);
  worker.reset();
  aclrtEventRecordedStatus after;
  ASSERT_EQ(aclrtQueryEventStatus(ready->npu_event(), &after), ACL_SUCCESS);
  EXPECT_EQ(after, ACL_EVENT_RECORDED_STATUS_COMPLETE);
  EXPECT_EQ(marker.cpu().item<int32_t>(), 7);
}

TEST(DSparkStepLocalLifecycleTest, PendingRowsFollowTheActualBatch) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  ForwardInput previous;
  previous.input_params.meta.num_sequences = 3;
  previous.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  previous.input_params.embedding.embedding_ids = {4, 5, 6};
  previous.input_params.embedding.request_ids = {"first", "second", "third"};
  previous.sample_sequence_ids = {"sequence-0", "sequence-1", "sequence-2"};
  previous.sampling_params.all_greedy_sample = true;
  DSparkWorkerImplTestPeer::install_pending(
      worker, previous, device.current_stream()->record_event());
  ForwardInput current = previous;
  current.input_params.meta.num_sequences = 2;
  current.input_params.embedding.embedding_ids = {6, 4};
  current.input_params.embedding.request_ids = {"third", "first"};
  current.sample_sequence_ids = {"sequence-2", "sequence-0"};
  current.input_params.attention.host.block_tables =
      torch::tensor({{7, 3}, {1, 8}}, torch::kInt32);
  EXPECT_TRUE(DSparkWorkerImplTestPeer::can_use_step_local(worker, current));
  EXPECT_EQ(DSparkWorkerImplTestPeer::pending_rows(worker, current),
            (std::vector<int64_t>{2, 0}));
  current.input_params.embedding.embedding_ids[0] = 9;
  EXPECT_TRUE(DSparkWorkerImplTestPeer::pending_rows(worker, current).empty());
  current.input_params.embedding.embedding_ids[0] = 6;
  current.sample_sequence_ids[0] = "replacement-sequence";
  EXPECT_TRUE(DSparkWorkerImplTestPeer::pending_rows(worker, current).empty());
  current.sample_sequence_ids.clear();
  EXPECT_TRUE(DSparkWorkerImplTestPeer::pending_rows(worker, current).empty());
}

TEST(DSparkStepLocalLifecycleTest, UnsupportedInputsDoNotUseTheAsyncPath) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  ForwardInput input;
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.embedding.embedding_ids = {4};
  input.input_params.embedding.request_ids = {"request"};
  input.sampling_params.all_greedy_sample = true;
  EXPECT_TRUE(DSparkWorkerImplTestPeer::pending_rows(worker, input).empty());
  DSparkWorkerImplTestPeer::install_pending(
      worker, input, device.current_stream()->record_event());
  EXPECT_TRUE(DSparkWorkerImplTestPeer::can_use_step_local(worker, input));
  ForwardInput changed = input;
  changed.sampling_params.repetition_penalties = torch::tensor({1.1});
  EXPECT_FALSE(DSparkWorkerImplTestPeer::can_use_step_local(worker, changed));
  changed = input;
  changed.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  EXPECT_FALSE(DSparkWorkerImplTestPeer::can_use_step_local(worker, changed));
  changed = input;
  changed.input_params.meta.num_sequences = 0;
  EXPECT_FALSE(DSparkWorkerImplTestPeer::can_use_step_local(worker, changed));
  // Host KV restore is synchronized inside this batch's forward. It does not
  // remove the pending row mapping, so pre-launch stays eligible.
  changed = input;
  changed.input_params.meta.requires_host_restore = true;
  EXPECT_TRUE(DSparkWorkerImplTestPeer::can_use_step_local(worker, changed));
}

TEST(DSparkStepLocalLifecycleTest,
     CommitWaitsForDeviceCopyAndSharesReadOnlyAcceptance) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  auto cache = std::make_shared<EmbeddingCache>(4, true);
  DSparkWorkerImplTestPeer::install_embedding_cache(worker, cache);
  ForwardInput identity;
  identity.input_params.embedding.embedding_ids = {0, 1, 2, 3};
  identity.input_params.embedding.request_ids = {
      "one", "two", "fifteen", "sixteen"};
  const std::vector<int64_t> lengths = {1, 2, 15, 16};
  torch::Tensor expected = torch::full({4, 16}, -1, torch::kLong);
  for (int64_t row = 0; row < 4; ++row) {
    expected[row]
        .narrow(0, 0, lengths[row])
        .copy_(torch::arange(
            100 + row * 100, 100 + row * 100 + lengths[row], torch::kLong));
  }
  const torch::TensorOptions host_options =
      torch::TensorOptions().dtype(torch::kLong).pinned_memory(true);
  torch::Tensor accepted_host = torch::full({4, 16}, -99, host_options);
  torch::Tensor accepted_device = expected.to(device.unwrap());
  torch::Tensor embeddings = torch::arange(4 * 16 * 8, torch::kFloat)
                                 .reshape({4, 16, 8})
                                 .to(device.unwrap());
  torch::Tensor positions =
      torch::tensor({113, 112, 129, 255}, torch::kLong).to(device.unwrap());
  const torch::TensorOptions device_options =
      torch::TensorOptions().device(device.unwrap()).dtype(torch::kBFloat16);
  torch::Tensor lhs = torch::ones({4096, 4096}, device_options);
  torch::Tensor rhs = torch::ones_like(lhs);
  torch::Tensor scratch = torch::empty_like(lhs);
  auto compute = device.current_stream();
  ASSERT_EQ(compute->synchronize(), 0);
  StreamEventPtr ready;
  {
    auto guard = compute->set_stream_guard();
    for (int32_t iteration = 0; iteration < 128; ++iteration) {
      torch::mm_out(scratch, lhs, rhs);
    }
    accepted_host.copy_(accepted_device, true);
    ready = compute->record_event();
  }
  DSparkWorkerImplTestPeer::install_target_context(worker,
                                                   identity,
                                                   accepted_device,
                                                   accepted_host,
                                                   embeddings,
                                                   positions,
                                                   ready);
  aclrtEventRecordedStatus before;
  ASSERT_EQ(aclrtQueryEventStatus(ready->npu_event(), &before), ACL_SUCCESS);
  ASSERT_EQ(before, ACL_EVENT_RECORDED_STATUS_NOT_READY);
  auto consumer =
      std::async(std::launch::async, [&device, ready, accepted_host]() {
        device.set_device();
        auto service_stream = device.get_stream_from_pool();
        if (!service_stream->wait_event(ready) ||
            service_stream->synchronize() != 0) {
          return torch::Tensor();
        }
        return accepted_host.clone();
      });
  DSparkWorkerImplTestPeer::commit(worker);
  const torch::Tensor service_tokens = consumer.get();
  EXPECT_TRUE(torch::equal(service_tokens, expected));
  EXPECT_TRUE(torch::equal(accepted_host, expected));
  EXPECT_FALSE(DSparkWorkerImplTestPeer::has_pending(worker));
  EXPECT_EQ(DSparkWorkerImplTestPeer::consumed_tensor_count(worker), 3);
  const auto states =
      cache->read_decode_states(identity.input_params.embedding.embedding_ids,
                                identity.input_params.embedding.request_ids);
  ASSERT_EQ(states.size(), 4);
  for (int64_t row = 0; row < 4; ++row) {
    EXPECT_EQ(states[row].token_id, 100 + row * 100 + lengths[row] - 1);
    EXPECT_EQ(states[row].position_offset, lengths[row] - 1);
    EXPECT_EQ(states[row].replay_token_ids.size(), lengths[row]);
    EXPECT_EQ(states[row].all_draft_accepted, lengths[row] == 16);
    EXPECT_TRUE(torch::equal(states[row].embedding.cpu(),
                             embeddings[row][lengths[row] - 1].cpu()));
  }
  DSparkWorkerImplTestPeer::commit(worker);
  EXPECT_EQ(DSparkWorkerImplTestPeer::consumed_tensor_count(worker), 3);
}

TEST(DSparkStepLocalLifecycleTest,
     DestructionRetiresDeviceWorkAfterContextOwnershipTransfer) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  auto worker = std::make_unique<DSparkWorkerImpl>(
      parallel_args, device.unwrap(), options);
  auto cache = std::make_shared<EmbeddingCache>(1, true);
  DSparkWorkerImplTestPeer::install_embedding_cache(*worker, cache);
  ForwardInput identity;
  identity.input_params.embedding.embedding_ids = {0};
  identity.input_params.embedding.request_ids = {"retained"};
  torch::Tensor accepted = torch::full({1, 16}, -1, torch::kLong);
  accepted[0][0] = 7;
  const torch::TensorOptions device_options =
      torch::TensorOptions().device(device.unwrap()).dtype(torch::kBFloat16);
  torch::Tensor embeddings = torch::ones({1, 16, 8}, device_options);
  auto compute = device.current_stream();
  DSparkWorkerImplTestPeer::install_target_context(
      *worker,
      identity,
      accepted.to(device.unwrap()),
      accepted,
      embeddings,
      torch::zeros(
          {1},
          torch::TensorOptions().device(device.unwrap()).dtype(torch::kLong)),
      compute->record_event());
  DSparkWorkerImplTestPeer::commit(*worker);
  ASSERT_FALSE(DSparkWorkerImplTestPeer::has_pending(*worker));
  ASSERT_EQ(DSparkWorkerImplTestPeer::consumed_tensor_count(*worker), 3);
  torch::Tensor lhs = torch::ones({4096, 4096}, device_options);
  torch::Tensor rhs = torch::ones_like(lhs);
  torch::Tensor scratch = torch::empty_like(lhs);
  torch::Tensor marker = torch::zeros({1}, device_options);
  ASSERT_EQ(compute->synchronize(), 0);
  StreamEventPtr ready;
  {
    auto guard = compute->set_stream_guard();
    for (int32_t iteration = 0; iteration < 128; ++iteration) {
      torch::mm_out(scratch, lhs, rhs);
    }
    marker.fill_(11);
    ready = compute->record_event();
  }
  aclrtEventRecordedStatus before;
  ASSERT_EQ(aclrtQueryEventStatus(ready->npu_event(), &before), ACL_SUCCESS);
  ASSERT_EQ(before, ACL_EVENT_RECORDED_STATUS_NOT_READY);
  worker.reset();
  aclrtEventRecordedStatus after;
  ASSERT_EQ(aclrtQueryEventStatus(ready->npu_event(), &after), ACL_SUCCESS);
  EXPECT_EQ(after, ACL_EVENT_RECORDED_STATUS_COMPLETE);
  EXPECT_EQ(marker.cpu().item<int32_t>(), 11);
}

}  // namespace
}  // namespace xllm
