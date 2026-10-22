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

#include "scheduler/dit_scheduler.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

#include "common/metrics.h"
#include "framework/request/dit_request.h"
#include "framework/request/dit_request_state.h"

namespace xllm {
namespace {

DiTScheduler::Options make_options(int64_t server_idx) {
  DiTScheduler::Options options;
  options.server_idx(server_idx).max_request_per_batch(2);
  return options;
}

class TestDiTScheduler final : public DiTDynamicBatchScheduler {
 public:
  explicit TestDiTScheduler(int64_t server_idx)
      : DiTDynamicBatchScheduler(make_options(server_idx)) {}

  void report_queues() { report_request_queue_metrics(); }

  std::vector<DiTBatch> prepare() { return prepare_batch(); }

  void clear_running_requests() { running_requests_.clear(); }
};

std::shared_ptr<DiTRequest> make_request(const std::string& request_id,
                                         int32_t width = 512) {
  DiTGenerationParams generation_params;
  generation_params.width = width;
  DiTRequestState state(
      DiTInputParams{},
      std::move(generation_params),
      [](const DiTRequestOutput&) { return true; },
      DiTOutputsFunc{},
      DiTRequestKind::kText);
  return std::make_shared<DiTRequest>(
      request_id, "x-request", "time", std::move(state));
}

TEST(DiTSchedulerTest, WaitingTracksQueuesSelectionCancellationAndDestruction) {
  TestDiTScheduler other(436);
  auto other_request = make_request("other");
  ASSERT_TRUE(other.add_request(other_request));
  other.report_queues();
  auto* other_waiting =
      MULTI_GAUGE_scheduler_waiting_requests.get_stats({"436"});
  auto* ingress =
      MULTI_GAUGE_scheduler_queue_requests.get_stats({"ingress", "433"});
  auto* deferred =
      MULTI_GAUGE_scheduler_queue_requests.get_stats({"dit_deferred", "433"});
  auto* waiting = MULTI_GAUGE_scheduler_waiting_requests.get_stats({"433"});
  ASSERT_NE(other_waiting, nullptr);
  ASSERT_NE(ingress, nullptr);
  ASSERT_NE(deferred, nullptr);
  ASSERT_NE(waiting, nullptr);
  {
    TestDiTScheduler scheduler(433);
    auto first = make_request("first");
    auto second = make_request("second", 256);
    auto third = make_request("third", 128);
    ASSERT_TRUE(scheduler.add_request(first));
    ASSERT_TRUE(scheduler.add_request(second));
    ASSERT_TRUE(scheduler.add_request(third));
    scheduler.report_queues();
    EXPECT_DOUBLE_EQ(ingress->get_value(), 3.0);
    EXPECT_DOUBLE_EQ(waiting->get_value(), 3.0);
    scheduler.prepare();
    EXPECT_DOUBLE_EQ(ingress->get_value(), 1.0);
    EXPECT_DOUBLE_EQ(deferred->get_value(), 1.0);
    EXPECT_DOUBLE_EQ(waiting->get_value(), 2.0);
    scheduler.clear_running_requests();
    scheduler.prepare();
    ASSERT_EQ(scheduler.get_running_requests().front(), second);
    EXPECT_DOUBLE_EQ(ingress->get_value(), 0.0);
    EXPECT_DOUBLE_EQ(deferred->get_value(), 1.0);
    third->set_cancel();
    scheduler.clear_running_requests();
    EXPECT_TRUE(scheduler.prepare().front().empty());
    EXPECT_DOUBLE_EQ(deferred->get_value(), 0.0);
    EXPECT_DOUBLE_EQ(waiting->get_value(), 0.0);
    auto remaining = make_request("remaining");
    ASSERT_TRUE(scheduler.add_request(remaining));
    scheduler.report_queues();
    EXPECT_DOUBLE_EQ(waiting->get_value(), 1.0);
  }
  EXPECT_DOUBLE_EQ(ingress->get_value(), 0.0);
  EXPECT_DOUBLE_EQ(waiting->get_value(), 0.0);
  EXPECT_DOUBLE_EQ(other_waiting->get_value(), 1.0);
}

}  // namespace
}  // namespace xllm
