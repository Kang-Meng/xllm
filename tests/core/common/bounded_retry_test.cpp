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

#include "common/bounded_retry.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <string>

#include "common/bounded_batch.h"

namespace xllm {
namespace {

using namespace std::chrono_literals;

TEST(BoundedRetryBackoffTest, ExponentialDelayHonorsHintAndMaximum) {
  BoundedRetryBackoff backoff(
      {.initial_delay = 100ms, .maximum_delay = 1000ms});
  ASSERT_TRUE(backoff.valid());
  EXPECT_EQ(backoff.delay(1), 100ms);
  EXPECT_EQ(backoff.delay(2), 200ms);
  EXPECT_EQ(backoff.delay(3, 500ms), 500ms);
  EXPECT_EQ(backoff.delay(5), 1000ms);
  EXPECT_EQ(backoff.delay(std::numeric_limits<uint32_t>::max()), 1000ms);
  EXPECT_FALSE(backoff.delay(0).has_value());
  EXPECT_EQ(backoff.delay(1, 1001ms), 1000ms);
  EXPECT_FALSE(backoff.delay(1, -1ms).has_value());
}

TEST(FairRetryPreferenceTest, AlternatesUnderSustainedDualQueuePressure) {
  FairRetryPreference preference;
  uint32_t fresh_taken = 0;
  uint32_t retry_taken = 0;
  for (uint32_t iteration = 0; iteration < 100000; ++iteration) {
    if (preference.prefer_retry()) {
      ++retry_taken;
      preference.record_retry_taken();
    } else {
      ++fresh_taken;
      preference.record_fresh_taken();
    }
  }
  EXPECT_EQ(fresh_taken, 50000u);
  EXPECT_EQ(retry_taken, 50000u);
}

TEST(KeyedBoundedBatchPlannerTest, ChunksEachServiceWithoutSerializingKeys) {
  KeyedBoundedBatchPlanner<std::string> planner(2);
  EXPECT_EQ(planner.assign("service-a"), 0u);
  EXPECT_EQ(planner.assign("service-b"), 1u);
  EXPECT_EQ(planner.assign("service-a"), 0u);
  EXPECT_EQ(planner.assign("service-a"), 2u);
  EXPECT_EQ(planner.assign("service-b"), 1u);
  EXPECT_EQ(planner.assign("service-b"), 3u);

  EXPECT_EQ(planner.batch_count(), 4u);
  EXPECT_EQ(planner.batch_size(0), 2u);
  EXPECT_EQ(planner.batch_size(1), 2u);
  EXPECT_EQ(planner.batch_size(2), 1u);
  EXPECT_EQ(planner.batch_size(3), 1u);
  EXPECT_EQ(planner.batch_size(4), 0u);
}

TEST(KeyedBoundedBatchPlannerTest, RejectsZeroCapacity) {
  KeyedBoundedBatchPlanner<std::string> planner(0);
  EXPECT_FALSE(planner.assign("service-a").has_value());
  EXPECT_EQ(planner.batch_count(), 0u);
}

TEST(KeyedBoundedBatchPlannerTest, StressNeverExceedsPerServiceRpcBound) {
  constexpr size_t kServices = 17;
  constexpr size_t kItems = 100000;
  constexpr size_t kMaximumBatch = 256;
  KeyedBoundedBatchPlanner<std::string> planner(kMaximumBatch);
  for (size_t index = 0; index < kItems; ++index) {
    ASSERT_TRUE(planner.assign("service-" + std::to_string(index % kServices))
                    .has_value());
  }
  size_t assigned = 0;
  for (size_t batch = 0; batch < planner.batch_count(); ++batch) {
    EXPECT_GT(planner.batch_size(batch), 0u);
    EXPECT_LE(planner.batch_size(batch), kMaximumBatch);
    assigned += planner.batch_size(batch);
  }
  EXPECT_EQ(assigned, kItems);
}

TEST(FailureEpisodeTrackerTest, ReportsFirstSummaryAndSingleRecovery) {
  FailureEpisodeTracker tracker(
      {.backoff = {.initial_delay = 100ms, .maximum_delay = 1000ms},
       .summary_interval = 1s});
  ASSERT_TRUE(tracker.valid());
  const FailureEpisodeTracker::TimePoint start{};

  FailureObservation first = tracker.record_failure("rpc_failed", start);
  EXPECT_EQ(first.action, FailureReportAction::kFirstFailure);
  EXPECT_EQ(first.retry_delay, 100ms);
  EXPECT_EQ(first.consecutive_failures, 1u);

  FailureObservation repeated =
      tracker.record_failure("rpc_failed", start + 100ms);
  EXPECT_EQ(repeated.action, FailureReportAction::kSuppressed);
  EXPECT_EQ(repeated.retry_delay, 200ms);
  EXPECT_EQ(repeated.total_suppressed, 1u);

  FailureObservation summary =
      tracker.record_failure("remote_rejected", start + 1s);
  EXPECT_EQ(summary.action, FailureReportAction::kPeriodicSummary);
  EXPECT_EQ(summary.suppressed_since_last_report, 1u);
  EXPECT_EQ(summary.total_suppressed, 1u);
  EXPECT_EQ(summary.first_reason, "rpc_failed");
  EXPECT_EQ(summary.last_reason, "remote_rejected");

  const std::optional<FailureRecovery> recovery =
      tracker.record_success(start + 1500ms);
  ASSERT_TRUE(recovery.has_value());
  EXPECT_EQ(recovery->consecutive_failures, 3u);
  EXPECT_EQ(recovery->total_suppressed, 1u);
  EXPECT_EQ(recovery->duration_ms, 1500u);
  EXPECT_EQ(recovery->last_reason, "remote_rejected");
  EXPECT_FALSE(tracker.record_success(start + 2s).has_value());
}

TEST(FailureEpisodeTrackerTest, StressSaturatesBackoffWithoutOverflow) {
  FailureEpisodeTracker tracker(
      {.backoff = {.initial_delay = 1ms, .maximum_delay = 1s},
       .summary_interval = 1h});
  const FailureEpisodeTracker::TimePoint start{};
  FailureObservation observation;
  for (uint32_t i = 0; i < 100000; ++i) {
    observation = tracker.record_failure("unavailable", start + 1ms * i);
  }
  EXPECT_EQ(observation.retry_delay, 1s);
  EXPECT_EQ(observation.consecutive_failures, 100000u);
  EXPECT_EQ(observation.total_suppressed, 99999u);

  const std::optional<FailureRecovery> recovery =
      tracker.record_success(start + 100s);
  ASSERT_TRUE(recovery.has_value());
  EXPECT_EQ(recovery->consecutive_failures, 100000u);
}

TEST(FailureEpisodeTrackerTest, OversizedRemoteHintCannotCreateHotLoop) {
  FailureEpisodeTracker tracker(
      {.backoff = {.initial_delay = 50ms, .maximum_delay = 1s},
       .summary_interval = 30s});
  ASSERT_TRUE(tracker.valid());
  const FailureObservation observation =
      tracker.record_failure("remote_overload",
                             FailureEpisodeTracker::TimePoint{},
                             std::chrono::hours(1));
  EXPECT_EQ(observation.retry_delay, 1s);
}

}  // namespace
}  // namespace xllm
