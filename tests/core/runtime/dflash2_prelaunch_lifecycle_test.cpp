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
#include <torch/torch.h>

#include <array>

#include "core/framework/batch/batch.h"
#include "core/framework/kv_cache_transfer/kv_transfer_completion.h"
#include "core/framework/parallel_state/process_group.h"
#include "framework/block/block_manager_impl.h"
#include "framework/request/stopping_checker.h"
#include "layers/common/attention_metadata.h"
#include "platform/device.h"
#include "runtime/dflash2_worker_impl.h"
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

namespace {

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
    manager.deallocate(ending.kv_state().take_blocks(kBlockTypes[i]));
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
    manager.deallocate(continuing.kv_state().take_blocks(kBlockTypes[i]));
    manager.deallocate(replacement.kv_state().take_blocks(kBlockTypes[i]));
  }
  EXPECT_EQ(manager.num_free_blocks(), 12u);
}

INSTANTIATE_TEST_SUITE_P(EagerAndGraph,
                         DFlash2PrelaunchLifecycleTest,
                         ::testing::Bool());

}  // namespace
}  // namespace xllm
