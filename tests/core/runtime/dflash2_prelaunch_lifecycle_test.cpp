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

#include "framework/block/block_manager_impl.h"
#include "framework/request/stopping_checker.h"
#include "platform/device.h"
#include "runtime/dflash2_worker_impl.h"

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
};

namespace {

class DFlash2PrelaunchLifecycleTest : public ::testing::TestWithParam<bool> {};

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
  BlockManager::Options block_options;
  block_options.num_blocks(4).block_size(128);
  BlockManagerImpl manager(block_options);
  auto ending_blocks = manager.allocate(1);
  auto continuing_blocks = manager.allocate(1);
  const int32_t ending_id = ending_blocks.front().id();
  const int32_t continuing_id = continuing_blocks.front().id();
  const auto tensor_options =
      torch::TensorOptions().device(device.unwrap()).dtype(torch::kBFloat16);
  auto cache = torch::zeros({4, 128}, tensor_options);
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
    cache[ending_id].fill_(7);
    cache[continuing_id].fill_(9);
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

  StoppingChecker stopping;
  stopping.set_eos_token(2);
  const std::vector<int32_t> ending_tokens = {11, 2};
  const std::vector<int32_t> continuing_tokens = {12, 31, 32};
  EXPECT_EQ(stopping.check({ending_tokens.data(), ending_tokens.size()}, 1),
            FinishReason::STOP);
  EXPECT_EQ(
      stopping.check({continuing_tokens.data(), continuing_tokens.size()}, 1),
      FinishReason::NONE);
  ending_blocks.clear();
  auto replacement = manager.allocate(1);
  ASSERT_EQ(replacement.front().id(), ending_id);
  auto reuse_stream = device.get_stream_from_pool();
  {
    auto guard = reuse_stream->set_stream_guard();
    cache[replacement.front().id()].fill_(23);
  }
  reuse_stream->synchronize();
  compute->synchronize();
  const auto host_cache = cache.to(torch::kCPU);
  EXPECT_TRUE(host_cache[ending_id].eq(23).all().item<bool>());
  EXPECT_TRUE(host_cache[continuing_id].eq(9).all().item<bool>());
}

INSTANTIATE_TEST_SUITE_P(EagerAndGraph,
                         DFlash2PrelaunchLifecycleTest,
                         ::testing::Bool());

}  // namespace
}  // namespace xllm
