/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <chrono>
#include <cstdint>
#include <limits>

#include "core/framework/request/request_timing_proto.h"

namespace xllm {
namespace {

RequestTiming::TimePoint at_ms(int64_t milliseconds) {
  return RequestTiming::TimePoint(std::chrono::milliseconds(milliseconds));
}

void record_prefill(RequestTiming& timing) {
  timing.record_arrival(at_ms(0));
  timing.record_queued(at_ms(10));
  timing.record_scheduled(at_ms(20));
  timing.update_from_output(at_ms(50));
}

void record_pd_segment(RequestTiming& timing, uint64_t tokens, int64_t start) {
  timing.note_generated_token();
  timing.record_pd_decode_start(tokens, at_ms(start));
  EXPECT_FALSE(timing.record_core_output(at_ms(start + 5)));
  timing.record_queued(at_ms(start + 10));
  timing.record_scheduled(at_ms(start + 30));
  timing.note_generated_token();
  EXPECT_TRUE(timing.record_core_output(at_ms(start + 100)));
}

TEST(RequestTimingTest, LocalStagesUseDistinctAndFrozenBoundaries) {
  RequestTiming timing;
  record_prefill(timing);
  timing.record_arrival(at_ms(35));
  timing.record_queued(at_ms(60));
  timing.record_scheduled(at_ms(100));
  timing.update_from_output(at_ms(40));
  timing.record_first_token_observed(at_ms(55));
  timing.update_from_output(at_ms(110));
  timing.update_from_finished_request(/*generation_complete=*/true, at_ms(150));
  timing.update_from_output(at_ms(200));
  timing.update_from_finished_request(/*generation_complete=*/false,
                                      at_ms(300));
  const auto sample = timing.snapshot(at_ms(1000));
  EXPECT_EQ(sample.server_ttft.milliseconds, 55.0);
  EXPECT_EQ(sample.queued_time.milliseconds, 10.0);
  EXPECT_EQ(sample.prefill_time.milliseconds, 20.0);
  EXPECT_EQ(sample.decode_time.milliseconds, 70.0);
  for (const auto& stage : {sample.server_ttft,
                            sample.queued_time,
                            sample.prefill_time,
                            sample.decode_time}) {
    EXPECT_TRUE(stage.valid);
    EXPECT_FALSE(stage.partial);
  }
  EXPECT_EQ(request_tpot_sample(sample, 8).milliseconds, 10.0);
}

TEST(RequestTimingTest, MissingInterruptedAndSingleTokenAreDistinct) {
  for (int32_t scenario = 0; scenario < 3; ++scenario) {
    SCOPED_TRACE(scenario);
    RequestTiming timing;
    timing.record_arrival(at_ms(0));
    if (scenario != 0) {
      timing.update_from_output(at_ms(30));
    }
    timing.update_from_finished_request(/*generation_complete=*/scenario == 2,
                                        at_ms(100));
    const auto sample = timing.snapshot(at_ms(1000));
    EXPECT_TRUE(sample.terminal);
    EXPECT_EQ(sample.decode_time.valid, scenario == 2);
    EXPECT_EQ(sample.decode_time.milliseconds.has_value(), scenario != 0);
    EXPECT_FALSE(sample.server_ttft.valid);
    EXPECT_TRUE(sample.server_ttft.partial);
    EXPECT_EQ(sample.server_ttft.milliseconds, 100.0);
    if (scenario == 2) {
      EXPECT_EQ(sample.decode_time.milliseconds, 0.0);
      timing.record_first_token_observed(at_ms(120));
      EXPECT_TRUE(timing.snapshot().server_ttft.valid);
    }
  }
}

TEST(RequestTimingTest, SharedOutputSignalsAndPdRestoreAreIdempotent) {
  auto ingress = RequestTiming::start_ingress(/*timing=*/nullptr, at_ms(0));
  EXPECT_EQ(RequestTiming::start_ingress(ingress, at_ms(30)), ingress);
  ingress->update_from_output(at_ms(80));
  ingress->record_first_token_observed(at_ms(82));
  EXPECT_EQ(ingress->snapshot().server_ttft.milliseconds, 82.0);
  RequestTiming timing;
  EXPECT_FALSE(timing.has_scheduled());
  timing.record_first_token_observed(at_ms(5));
  EXPECT_FALSE(timing.has_observed_token());
  record_prefill(timing);
  timing.note_generated_token();
  timing.note_generated_token();
  EXPECT_TRUE(timing.record_core_output(at_ms(70)));
  EXPECT_FALSE(timing.record_core_output(at_ms(80)));
  EXPECT_EQ(timing.snapshot().decode_time.milliseconds, 20.0);
  timing.mark_pd_local();
  timing.mark_pd_handoff_accepted();
  EXPECT_TRUE(timing.snapshot().pd_handoff_accepted);
  timing.record_first_token_observed(at_ms(80));
  timing.note_generated_token();
  timing.restore_pd(RequestTimingSnapshot{});
  EXPECT_TRUE(timing.has_scheduled());
  EXPECT_TRUE(timing.needs_pd_scheduling());
  EXPECT_TRUE(timing.has_observed_token());
  EXPECT_FALSE(timing.has_generated_token());
  EXPECT_FALSE(timing.snapshot().pd_handoff_accepted);
  // OOC keeps the ingress observation but cannot claim a complete PD TTFT
  // on an imported role visit or join its old and new generation spans.
  EXPECT_EQ(timing.snapshot().server_ttft.milliseconds, 80.0);
  EXPECT_FALSE(timing.snapshot().server_ttft.valid);
  EXPECT_FALSE(timing.snapshot().decode_time.milliseconds.has_value());
  EXPECT_FALSE(timing.record_core_output(at_ms(90)));
}

TEST(RequestTimingTest, ReversedQueueCannotBecomeComplete) {
  RequestTiming timing;
  timing.record_queued(at_ms(20));
  timing.record_scheduled(at_ms(10));
  timing.update_from_output(at_ms(40));
  timing.update_from_output(at_ms(30));
  timing.update_from_output(at_ms(80));
  timing.update_from_finished_request(/*generation_complete=*/true);
  const auto sample = timing.snapshot();
  EXPECT_FALSE(sample.queued_time.milliseconds.has_value());
  EXPECT_EQ(sample.queued_time.reason, RequestTimingReason::UNKNOWN_BOUNDARY);
  EXPECT_EQ(sample.prefill_time.milliseconds, 20.0);
  EXPECT_EQ(sample.decode_time.milliseconds, 50.0);
}

TEST(RequestTimingTest, PdAckRequiresLiveOriginalIngress) {
  for (int32_t scenario = 0; scenario < 4; ++scenario) {
    RequestTiming timing;
    timing.record_arrival(at_ms(0));
    timing.mark_pd_local();
    if (scenario != 0) {
      timing.update_from_output(at_ms(10));
    }
    if (scenario == 1) {
      timing.restore_pd(RequestTimingSnapshot{});
      timing.update_from_output(at_ms(15));
    } else if (scenario == 2) {
      timing.update_from_finished_request(/*generation_complete=*/true,
                                          at_ms(15));
    }
    timing.mark_pd_handoff_accepted();
    timing.record_first_token_observed(at_ms(20));
    EXPECT_EQ(timing.snapshot().pd_handoff_accepted, scenario == 3);
    EXPECT_EQ(timing.snapshot().server_ttft.valid, scenario == 3);
  }
}

TEST(RequestTimingTest, PdLocalSegmentsExcludeImportsAndResetOnReturn) {
  RequestTiming ingress;
  record_prefill(ingress);
  ingress.mark_pd_local();
  const auto upstream = ingress.snapshot();
  for (const bool returning : {false, true}) {
    RequestTiming timing;
    if (returning) {
      record_prefill(timing);
    }
    timing.restore_pd(upstream);
    if (returning) {
      record_pd_segment(timing, /*tokens=*/1, /*start=*/10000);
      timing.restore_pd(timing.snapshot());
    }
    EXPECT_TRUE(timing.needs_pd_scheduling());
    EXPECT_FALSE(timing.snapshot().pd_queued_time.valid);
    record_pd_segment(timing, /*tokens=*/3, /*start=*/20000);
    timing.record_pd_decode_start(/*num_generation_tokens=*/4, at_ms(20150));
    timing.update_from_finished_request(/*generation_complete=*/true,
                                        at_ms(20200));
    const auto sample = timing.snapshot();
    EXPECT_EQ(sample.queued_time.milliseconds, 10.0);
    EXPECT_EQ(sample.prefill_time.milliseconds, 30.0);
    EXPECT_EQ(sample.pd_queued_time.milliseconds, 20.0);
    EXPECT_EQ(sample.pd_decode_time.milliseconds, 100.0);
    EXPECT_TRUE(sample.pd_queued_time.valid);
    EXPECT_TRUE(sample.pd_decode_time.valid);
    EXPECT_FALSE(sample.decode_time.valid);
    EXPECT_EQ(request_pd_tpot_sample(sample, 7).milliseconds, 25.0);
    EXPECT_FALSE(request_pd_tpot_sample(sample, 3).valid);
    EXPECT_FALSE(request_tpot_sample(sample, 7).valid);
    timing.restore_pd(upstream);
    timing.record_pd_decode_start(/*num_generation_tokens=*/7, at_ms(30000));
    EXPECT_EQ(timing.snapshot().pd_decode_tokens_at_start, 3u);
  }
}

TEST(RequestTimingTest, PdInterruptedAndUnknownDecodeRemainInvalid) {
  for (const bool interrupted : {false, true}) {
    RequestTiming timing;
    timing.restore_pd(RequestTimingSnapshot{});
    record_pd_segment(timing, /*tokens=*/1, /*start=*/10);
    if (!interrupted) {
      timing.mark_generation_events_incomplete();
    }
    timing.update_from_finished_request(/*generation_complete=*/!interrupted);
    const auto sample = request_pd_tpot_sample(timing.snapshot(), 2);
    EXPECT_FALSE(sample.valid);
    EXPECT_EQ(sample.reason,
              interrupted ? RequestTimingReason::INTERRUPTED
                          : RequestTimingReason::UNKNOWN_BOUNDARY);
  }
  RequestTiming timing;
  timing.record_pd_decode_start(/*num_generation_tokens=*/1);
  EXPECT_FALSE(timing.has_generated_token());
}

TEST(RequestTimingTest, CoreTpotPreservesValidityAndOutputScope) {
  RequestTimingSnapshot sample;
  sample.decode_time = {70.0, true, false, RequestTimingReason::COMPLETE};
  EXPECT_FALSE(request_tpot_sample(sample, std::nullopt).valid);
  EXPECT_FALSE(request_tpot_sample(sample, 0).valid);
  EXPECT_EQ(request_tpot_sample(sample, 1).milliseconds, 0.0);
  EXPECT_EQ(request_tpot_sample(sample, 8).milliseconds, 10.0);
  sample.pd_local = true;
  EXPECT_FALSE(request_tpot_sample(sample, 8).valid);
  sample.pd_local = false;
  sample.decode_time = {70.0, false, true, RequestTimingReason::INTERRUPTED};
  EXPECT_FALSE(request_tpot_sample(sample, 1).valid);
  EXPECT_TRUE(request_tpot_sample(sample, 1).partial);
}

TEST(RequestTimingProtoTest, RoundTripPreservesStagesButNotIngressAck) {
  RequestTimingSnapshot sample;
  sample.queued_time = {15.0, true, false, RequestTimingReason::COMPLETE};
  sample.prefill_time.reason = RequestTimingReason::UNKNOWN_BOUNDARY;
  sample.decode_time = {7.0, false, true, RequestTimingReason::INTERRUPTED};
  sample.pd_handoff_accepted = true;
  proto::RequestPhaseTiming wire;
  request_timing_to_proto(sample, &wire);
  const auto restored = request_timing_from_proto(&wire);
  EXPECT_EQ(restored.queued_time.milliseconds, 15.0);
  EXPECT_TRUE(restored.queued_time.valid);
  EXPECT_FALSE(restored.prefill_time.milliseconds.has_value());
  EXPECT_EQ(restored.decode_time.milliseconds, 7.0);
  EXPECT_TRUE(restored.decode_time.partial);
  EXPECT_FALSE(restored.decode_time.valid);
  EXPECT_FALSE(restored.pd_handoff_accepted);
  EXPECT_FALSE(request_timing_from_proto(nullptr).queued_time.valid);
}

TEST(RequestTimingProtoTest, MalformedWireCannotClaimCompleteDuration) {
  for (const uint32_t reason :
       {static_cast<uint32_t>(RequestTimingReason::UNSUPPORTED_OBJECT),
        std::numeric_limits<uint32_t>::max()}) {
    proto::RequestPhaseTiming wire;
    auto* sample = wire.mutable_queued_time();
    sample->set_reason(reason);
    sample->set_milliseconds(7.0);
    sample->set_valid(true);
    const auto restored = request_timing_from_proto(&wire).queued_time;
    EXPECT_EQ(restored.reason, RequestTimingReason::UNKNOWN_BOUNDARY);
    EXPECT_FALSE(restored.valid);
    EXPECT_TRUE(restored.partial);
  }
  const double values[] = {0.0,
                           7.0,
                           0.0,
                           -1.0,
                           std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN(),
                           7.0};
  for (int32_t scenario = 0; scenario < 7; ++scenario) {
    proto::RequestPhaseTiming wire;
    auto* sample = wire.mutable_decode_time();
    sample->set_valid(scenario != 6);
    sample->set_partial(scenario == 1);
    if (scenario != 0) {
      sample->set_milliseconds(values[scenario]);
      sample->set_reason(
          scenario == 2 ? 100
                        : static_cast<uint32_t>(RequestTimingReason::COMPLETE));
    }
    const auto restored = request_timing_from_proto(&wire).decode_time;
    EXPECT_FALSE(restored.valid);
    if (scenario == 6) {
      EXPECT_TRUE(restored.partial);
      EXPECT_EQ(restored.reason, RequestTimingReason::UNKNOWN_BOUNDARY);
    }
    EXPECT_EQ(restored.milliseconds.has_value(),
              scenario == 1 || scenario == 2 || scenario == 6);
  }
}

}  // namespace
}  // namespace xllm
