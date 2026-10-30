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

#include "scheduler/decode_admission_retry_policy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace xllm {
namespace {

TEST(DecodeAdmissionRetryPolicyTest,
     SustainedV2ShortageCannotBecomeAnAdmissionHotLoop) {
  constexpr uint32_t kFailures = 220;
  constexpr uint64_t kDeadlineMs = 300'000;
  std::chrono::milliseconds virtual_elapsed(0);
  for (uint32_t attempt = 1; attempt <= kFailures; ++attempt) {
    const DecodeAdmissionRetryDecision decision =
        decide_decode_admission_retry({
            .attempts = attempt,
            .is_v2_attempt = true,
            .has_request_deadline = true,
            .remaining_request_deadline_ms =
                kDeadlineMs - static_cast<uint64_t>(virtual_elapsed.count()),
        });
    ASSERT_EQ(decision.status, DecodeAdmissionRetryStatus::RETRY);
    const uint32_t exponent = std::min<uint32_t>(attempt - 1, 5);
    const std::chrono::milliseconds expected_delay(
        std::min<uint64_t>(1000, uint64_t{50} << exponent));
    EXPECT_EQ(decision.retry_delay, expected_delay);
    virtual_elapsed += decision.retry_delay;
  }

  // The online failure attempted admission 1,156,819 times in about 207
  // seconds. The production policy permits only 220 attempts over a longer
  // equivalent interval, before RPC latency is even included.
  EXPECT_GE(virtual_elapsed, std::chrono::seconds(207));
}

TEST(DecodeAdmissionRetryPolicyTest,
     MissingDeadlineUsesTheLocalAttemptBoundForEveryCaller) {
  for (const bool is_v2_attempt : {false, true}) {
    EXPECT_EQ(
        decide_decode_admission_retry(
            {.attempts = kMaximumDecodeAdmissionAttemptsWithoutDeadline - 1,
             .is_v2_attempt = is_v2_attempt})
            .status,
        DecodeAdmissionRetryStatus::RETRY);
    EXPECT_EQ(decide_decode_admission_retry(
                  {.attempts = kMaximumDecodeAdmissionAttemptsWithoutDeadline,
                   .is_v2_attempt = is_v2_attempt})
                  .status,
              DecodeAdmissionRetryStatus::ATTEMPT_BUDGET_EXHAUSTED);
  }
}

TEST(DecodeAdmissionRetryPolicyTest, DeadlineAndServerDelayFailClosed) {
  const DecodeAdmissionRetryDecision allowed = decide_decode_admission_retry({
      .attempts = 1,
      .is_v2_attempt = true,
      .has_request_deadline = true,
      .remaining_request_deadline_ms = 501,
      .retry_after = std::chrono::milliseconds(500),
  });
  EXPECT_EQ(allowed.status, DecodeAdmissionRetryStatus::RETRY);
  EXPECT_EQ(allowed.retry_delay, std::chrono::milliseconds(500));

  EXPECT_EQ(decide_decode_admission_retry(
                {
                    .attempts = 1,
                    .is_v2_attempt = true,
                    .has_request_deadline = true,
                    .remaining_request_deadline_ms = 500,
                    .retry_after = std::chrono::milliseconds(500),
                })
                .status,
            DecodeAdmissionRetryStatus::DEADLINE_BUDGET_INSUFFICIENT);
}

TEST(DecodeAdmissionRetryPolicyTest,
     GenerationCommitFailureIsRetryableUnavailable) {
  const Status status = decode_generation_commit_unavailable_status();
  EXPECT_EQ(status.code(), StatusCode::UNAVAILABLE);
  EXPECT_EQ(status.message(), "DECODE_GENERATION_COMMIT_UNAVAILABLE");
}

}  // namespace
}  // namespace xllm
