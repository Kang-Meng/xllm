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

#include <butil/iobuf.h>
#include <butil/strings/string_piece.h>

// This BRPC builtin header requires butil declarations first.
#include <brpc/builtin/prometheus_metrics_service.h>
#include <brpc/channel.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/common/metrics.h"
#include "core/framework/batch/batch.h"
#include "core/framework/request/request.h"
#include "core/scheduler/async_response_processor.h"

namespace xllm {
namespace {

using MetricCounts = std::array<int64_t, 5>;

RequestTiming::TimePoint at_ms(int64_t milliseconds) {
  return RequestTiming::TimePoint(std::chrono::milliseconds(milliseconds));
}

MetricCounts local_counts() {
  return {HISTOGRAM_xllm_request_ttft_microseconds.count(),
          HISTOGRAM_xllm_request_queue_microseconds.count(),
          HISTOGRAM_xllm_request_prefill_microseconds.count(),
          HISTOGRAM_xllm_request_decode_microseconds.count(),
          HISTOGRAM_xllm_request_tpot_microseconds.count()};
}

MetricCounts pd_counts() {
  const auto count = [](bvar::MultiDimension<bvar::LatencyRecorder>& recorder,
                        const char* role) {
    return recorder.get_stats({role})->count();
  };
  return {
      count(MULTI_HISTOGRAM_xllm_pd_request_queue_microseconds, "prefill"),
      count(MULTI_HISTOGRAM_xllm_pd_request_prefill_microseconds, "prefill"),
      count(MULTI_HISTOGRAM_xllm_pd_request_queue_microseconds, "decode"),
      count(MULTI_HISTOGRAM_xllm_pd_request_decode_microseconds, "decode"),
      count(MULTI_HISTOGRAM_xllm_pd_request_tpot_microseconds, "decode")};
}

void expect_delta(const MetricCounts& before,
                  const MetricCounts& after,
                  const MetricCounts& delta) {
  for (size_t index = 0; index < before.size(); ++index) {
    EXPECT_EQ(after[index], before[index] + delta[index]) << "metric " << index;
  }
}

bvar::Adder<int64_t>& skipped_count(const char* phase, const char* reason) {
  return *MULTI_COUNTER_xllm_request_phase_skipped_total.get_stats(
      {phase, reason});
}

class PhaseTokenizer final : public Tokenizer {
 public:
  std::string decode(const Slice<int32_t>& ids,
                     bool /*skip_special_tokens*/) const override {
    return std::string(ids.size(), 'A');
  }
  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<PhaseTokenizer>();
  }
};

std::shared_ptr<Request> make_request(int32_t max_tokens = 2,
                                      bool stream = false,
                                      bool overlap = false) {
  RequestState state;
  state.prompt = "A";
  state.prompt_tokens = {1};
  state.seq_capacity = 8;
  state.n = state.best_of = 1;
  state.stream = stream;
  state.enable_schedule_overlap = overlap;
  state.stopping_checker.set_max_generated_tokens(max_tokens);
  state.output_func = [](const RequestOutput&) { return true; };
  auto request =
      std::make_shared<Request>("phase-output", "", "", std::move(state));
  request->state().timing =
      RequestTiming::start_ingress(request->state().timing, at_ms(0));
  request->state().timing->record_queued(at_ms(10));
  request->mark_first_scheduled(at_ms(20));
  request->sequences().front()->kv_state().set_kv_cache_tokens_num(1);
  return request;
}

void commit_token(Request& request, int32_t token, int64_t milliseconds) {
  request.sequences().front()->append_token(Token(token));
  request.state().timing->record_core_output(at_ms(milliseconds));
}

void publish_concurrently(const std::shared_ptr<Request>& request,
                          bool terminal) {
  std::vector<std::thread> callbacks;
  callbacks.reserve(4);
  for (int32_t index = 0; index < 4; ++index) {
    callbacks.emplace_back([request, terminal] {
      if (terminal) {
        request->record_server_first_token(at_ms(40));
      }
      request->update_timing_after_execution(terminal);
    });
  }
  for (auto& callback : callbacks) {
    callback.join();
  }
}

TEST(RequestPhaseOutputTest, LocalMetricsPrecedeResponseAndPublishOnce) {
  for (const bool stream : {false, true}) {
    for (const int32_t tokens : {1, 2}) {
      SCOPED_TRACE(std::to_string(tokens) + ":" + std::to_string(stream));
      PhaseTokenizer tokenizer;
      AsyncResponseProcessor processor(&tokenizer,
                                       std::nullopt,
                                       /*enable_service_routing=*/false,
                                       /*disable_log_stats=*/stream,
                                       [](std::shared_ptr<Request>) {});
      auto request = make_request(tokens, stream, /*overlap=*/true);
      std::atomic<int32_t> responses{0};
      request->state().output_func = [&responses](const RequestOutput&) {
        ++responses;
        return true;
      };
      const auto before = local_counts();
      auto& decode_skipped = skipped_count("decode", "single_token");
      auto& tpot_skipped = skipped_count("tpot", "single_token");
      const int64_t decode_before = decode_skipped.get_value();
      const int64_t tpot_before = tpot_skipped.get_value();
      commit_token(*request, /*token=*/65, /*milliseconds=*/30);
      request->record_server_first_token(at_ms(40));
      expect_delta(before, local_counts(), {1, 0, 0, 0, 0});
      if (tokens == 2) {
        commit_token(*request, /*token=*/66, /*milliseconds=*/50);
      }
      request->sequences().front()->append_token(Token(-1));
      publish_concurrently(request, /*terminal=*/true);
      const auto completed = local_counts();
      expect_delta(before, completed, {1, 1, 1, tokens - 1, tokens - 1});
      EXPECT_EQ(decode_skipped.get_value(), decode_before + (tokens == 1));
      EXPECT_EQ(tpot_skipped.get_value(), tpot_before + (tokens == 1));
      EXPECT_EQ(request_tpot_sample(request->state().timing->snapshot(), tokens)
                    .milliseconds,
                tokens == 1 ? 0.0 : 20.0);
      EXPECT_EQ(responses.load(), 0);
      processor.process_completed_request(request);
      processor.process_completed_request(request);
      processor.wait_completion();
      EXPECT_EQ(responses.load(), 1);
      EXPECT_EQ(local_counts(), completed);
      EXPECT_EQ(decode_skipped.get_value(), decode_before + (tokens == 1));
      EXPECT_EQ(tpot_skipped.get_value(), tpot_before + (tokens == 1));
    }
  }
  for (const bool stream : {false, true}) {
    auto first = make_request(/*max_tokens=*/2, stream);
    auto second = make_request(/*max_tokens=*/2, stream);
    Batch first_batch(first->sequences().front().get());
    Batch second_batch(second->sequences().front().get());
    const auto observer = [first, second] {
      const RequestTiming::TimePoint now = RequestTiming::Clock::now();
      first->record_server_first_token(now);
      second->record_server_first_token(now);
    };
    first_batch.set_output_observer(observer);
    second_batch.set_output_observer(observer);
    // Pending tokens in another batch must not advance that request's TTFT.
    first->sequences().front()->append_token(Token(65));
    second->sequences().front()->append_token(Token(65));
    first_batch.record_core_output(at_ms(30));
    EXPECT_TRUE(first->state().timing->has_observed_token());
    EXPECT_FALSE(second->state().timing->has_observed_token());
    second_batch.record_core_output(at_ms(50));
    EXPECT_TRUE(second->state().timing->has_observed_token());
  }
}

TEST(RequestPhaseOutputTest, PdStagesPublishOnCorrectRoleOnce) {
  const auto local_before = local_counts();
  auto prefill = make_request();
  prefill->state().timing->mark_pd_local();
  commit_token(*prefill, /*token=*/65, /*milliseconds=*/30);
  const auto before_prefill = pd_counts();
  publish_concurrently(prefill, /*terminal=*/false);
  expect_delta(before_prefill, pd_counts(), {1, 1, 0, 0, 0});
  EXPECT_FALSE(prefill->state().timing->snapshot().terminal);

  auto decode =
      make_request(/*max_tokens=*/3, /*stream=*/true, /*overlap=*/true);
  auto& timing = *decode->state().timing;
  timing.restore_pd(prefill->state().timing->snapshot());
  decode->sequences().front()->append_token(Token(65));
  timing.record_pd_decode_start(/*num_generation_tokens=*/1, at_ms(10000));
  timing.record_queued(at_ms(10010));
  decode->mark_first_scheduled(at_ms(10030));
  commit_token(*decode, /*token=*/66, /*milliseconds=*/10050);
  commit_token(*decode, /*token=*/67, /*milliseconds=*/10100);
  decode->sequences().front()->append_token(Token(-1));
  const auto before_decode = pd_counts();
  publish_concurrently(decode, /*terminal=*/true);
  expect_delta(before_decode, pd_counts(), {0, 0, 1, 1, 1});
  const auto snapshot = timing.snapshot();
  EXPECT_EQ(snapshot.pd_queued_time.milliseconds, 20.0);
  EXPECT_EQ(snapshot.pd_decode_time.milliseconds, 100.0);
  EXPECT_EQ(request_pd_tpot_sample(
                snapshot,
                decode->sequences().front()->num_committed_generated_tokens())
                .milliseconds,
            50.0);
  EXPECT_FALSE(snapshot.decode_time.valid);
  const auto after = pd_counts();
  prefill->set_cancel();
  prefill->update_timing_after_execution(/*generation_terminal=*/false);
  decode->update_timing_after_execution(/*generation_terminal=*/true);
  EXPECT_EQ(pd_counts(), after);
  EXPECT_EQ(local_counts(), local_before);
}

TEST(RequestPhaseOutputTest, ExcludedOutputsDoNotPublish) {
  for (int32_t scenario = 0; scenario < 11; ++scenario) {
    SCOPED_TRACE(scenario);
    auto request = make_request();
    auto& timing = *request->state().timing;
    if (scenario < 4) {
      timing.restore_pd(RequestTimingSnapshot{});
      request->sequences().front()->append_token(Token(65));
      timing.record_pd_decode_start(/*num_generation_tokens=*/1, at_ms(100));
      timing.record_queued(at_ms(110));
      request->mark_first_scheduled(at_ms(120));
    }
    const auto before = local_counts();
    const auto pd_before = pd_counts();
    const char* phase = scenario < 4     ? "pd_decode"
                        : scenario == 10 ? "pd_prefill"
                                         : "decode";
    const char* reason = scenario == 0   ? "no_local_tokens"
                         : scenario == 2 ? "unknown_boundary"
                         : scenario == 3 ? "failed"
                         : scenario == 1 || scenario == 4 || scenario == 10
                             ? "cancelled"
                             : "unsupported_output";
    auto& skipped = skipped_count(phase, reason);
    const int64_t skipped_before = skipped.get_value();
    if (scenario != 0) {
      commit_token(*request, /*token=*/66, /*milliseconds=*/150);
    }
    switch (scenario) {
      case 1:
      case 4:
        request->set_cancel();
        break;
      case 2:
        timing.mark_generation_events_incomplete();
        break;
      case 3:
        request->sequences().front()->fail(
            Status(StatusCode::UNKNOWN, "failure"));
        break;
      case 5:
        request->state().best_of = 2;
        break;
      case 6:
        request->state().sampling_param.is_embeddings = true;
        break;
      case 7:
        request->state().is_graph_warmup = true;
        break;
      case 8:
        request->state().sample_slots.resize(1);
        break;
      case 9:
        request->state().n = 2;
        break;
      case 10:
        timing.mark_pd_local();
        request->set_cancel();
        break;
    }
    if (scenario == 1 || scenario == 4 || scenario == 10) {
      // Cancellation alone must publish, even if no later callback runs.
      EXPECT_EQ(skipped.get_value(), skipped_before + 1);
      request->set_cancel();
      request->record_timing_failure(Status(StatusCode::CANCELLED));
      EXPECT_EQ(skipped.get_value(), skipped_before + 1);
    }
    request->record_server_first_token(at_ms(160));
    request->sequences().front()->finish();
    request->update_timing_after_execution(/*generation_terminal=*/scenario !=
                                           10);
    EXPECT_EQ(local_counts(), before);
    EXPECT_EQ(pd_counts(), pd_before);
    EXPECT_EQ(skipped.get_value(), skipped_before + (scenario == 7 ? 0 : 1));
    publish_concurrently(request, /*terminal=*/scenario != 10);
    request->set_cancel();
    EXPECT_EQ(skipped.get_value(), skipped_before + (scenario == 7 ? 0 : 1));
  }
  // Rejected admission need not finish a Sequence to count a failed sample.
  auto rejected = make_request();
  auto& failed = skipped_count("queue", "failed");
  const int64_t before_failure = failed.get_value();
  rejected->record_timing_failure(Status(StatusCode::RESOURCE_EXHAUSTED));
  rejected->record_timing_failure(Status(StatusCode::RESOURCE_EXHAUSTED));
  EXPECT_EQ(failed.get_value(), before_failure + 1);
}

TEST(RequestPhaseOutputTest, RecorderRejectsIncompleteAndMalformedSamples) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  const std::array<RequestTimingSample, 9> samples{
      {{1.0, true, false, RequestTimingReason::COMPLETE},
       {1.0, true, true, RequestTimingReason::COMPLETE},
       {1.0, true, false, RequestTimingReason::INTERRUPTED},
       {1.0, true, false, RequestTimingReason::UNKNOWN_BOUNDARY},
       {std::nullopt, true, false, RequestTimingReason::COMPLETE},
       {nan, true, false, RequestTimingReason::COMPLETE},
       {infinity, true, false, RequestTimingReason::COMPLETE},
       {-1.0, true, false, RequestTimingReason::COMPLETE},
       {static_cast<double>(std::numeric_limits<int64_t>::max()),
        true,
        false,
        RequestTimingReason::COMPLETE}}};
  for (const auto& sample : samples) {
    RequestTimingSnapshot snapshot;
    snapshot.terminal = true;
    snapshot.server_ttft = snapshot.queued_time = snapshot.prefill_time =
        snapshot.decode_time = snapshot.pd_queued_time =
            snapshot.pd_decode_time = sample;
    snapshot.pd_decode_tokens_at_start = 1;
    const auto before = local_counts();
    const auto pd_before = pd_counts();
    const char* reason =
        sample.reason == RequestTimingReason::INTERRUPTED ? "interrupted"
        : sample.reason == RequestTimingReason::UNKNOWN_BOUNDARY
            ? "unknown_boundary"
            : "invalid_sample";
    auto& p_skipped = skipped_count("pd_prefill_queue", reason);
    auto& d_skipped = skipped_count("pd_decode_queue", reason);
    const int64_t p_before = p_skipped.get_value();
    const int64_t d_before = d_skipped.get_value();
    observe_request_ttft_metric(snapshot, /*single_output=*/true);
    observe_request_phase_metrics(
        snapshot, /*single_output=*/true, /*num_generation_tokens=*/2);
    snapshot.pd_local = true;
    // A complete queue must not bypass an invalid PD execution segment.
    snapshot.queued_time = snapshot.pd_queued_time = {
        1.0, true, false, RequestTimingReason::COMPLETE};
    observe_request_pd_prefill_metrics(snapshot, /*single_output=*/true);
    snapshot.pd_imported = true;
    observe_request_phase_metrics(
        snapshot, /*single_output=*/true, /*num_generation_tokens=*/1000000);
    const int64_t expected =
        static_cast<int64_t>(sample.valid && !sample.partial &&
                             sample.reason == RequestTimingReason::COMPLETE &&
                             sample.milliseconds == 1.0);
    const MetricCounts delta{expected, expected, expected, expected, expected};
    expect_delta(before, local_counts(), delta);
    expect_delta(pd_before, pd_counts(), delta);
    EXPECT_EQ(p_skipped.get_value(), p_before + 1 - expected);
    EXPECT_EQ(d_skipped.get_value(), d_before + 1 - expected);
  }
}

TEST(RequestPhaseOutputTest, MissingQueueDoesNotDiscardKnownLocalDecode) {
  RequestTimingSnapshot snapshot;
  snapshot.terminal = true;
  snapshot.decode_time = {20.0, true, false, RequestTimingReason::COMPLETE};
  const auto before = local_counts();
  auto& skipped = skipped_count("queue", "missing_start");
  const int64_t skipped_before = skipped.get_value();
  observe_request_phase_metrics(
      snapshot, /*single_output=*/true, /*num_generation_tokens=*/2);
  expect_delta(before, local_counts(), {0, 0, 0, 1, 1});
  EXPECT_EQ(skipped.get_value(), skipped_before + 1);
}

TEST(RequestPhaseOutputTest,
     HttpScrapesPreserveLatencyRecordersAndLegacyMetrics) {
  auto& failed = skipped_count("queue", "failed");
  // Default double formatting loses unit increments at million-scale counts.
  failed << 1000001;
  const int64_t expected_failed = failed.get_value();
  bvar::LatencyRecorder summary("rpc_server_t3_export_summary");
  summary << 10 << 20;
  bvar::MultiDimension<bvar::Status<double>> gauge("t3_export_labeled_gauge",
                                                   {"role"});
  gauge.get_stats({"prefill"})->set_value(1.0);
  gauge.get_stats({"decode"})->set_value(2.0);
  brpc::Server server;
  ASSERT_EQ(server.Start("127.0.0.1:0", /*options=*/nullptr), 0);
  brpc::ChannelOptions options;
  options.protocol = "http";
  brpc::Channel channel;
  ASSERT_EQ(channel.Init(server.listen_address(), &options), 0);
  const auto metrics = local_counts();
  const auto pd_metrics = pd_counts();
  const std::array<std::string, 5> phases{
      "ttft", "queue", "prefill", "decode", "tpot"};
  const std::array<std::string, 5> pd_phases{
      "queue", "prefill", "queue", "decode", "tpot"};
  for (int32_t scrape = 0; scrape < 3; ++scrape) {
    brpc::Controller controller;
    controller.http_request().uri() = "/brpc_metrics";
    channel.CallMethod(/*method=*/nullptr,
                       &controller,
                       /*request=*/nullptr,
                       /*response=*/nullptr,
                       /*done=*/nullptr);
    ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
    ASSERT_EQ(controller.http_response().status_code(), 200);
    const std::string text = controller.response_attachment().to_string();
    const auto contains = [&text](const std::string& line) {
      EXPECT_NE(text.find(line), std::string::npos) << line;
    };
    contains("xllm_request_phase_skipped_total_phase_queue_reason_failed " +
             std::to_string(expected_failed) + "\n");
    for (size_t index = 0; index < phases.size(); ++index) {
      const std::string local =
          "xllm_request_" + phases[index] + "_microseconds";
      const std::string pd =
          "xllm_pd_request_" + pd_phases[index] + "_microseconds";
      const std::string role = index < 2 ? "_role_prefill" : "_role_decode";
      for (const std::string& suffix : {"_latency", "_count"}) {
        contains("# TYPE " + local + suffix + " gauge\n");
        contains("# TYPE " + pd + suffix + role + " gauge\n");
      }
      contains("# TYPE " + local + "_latency_99 gauge\n");
      contains("# TYPE " + pd + "_latency" + role + "_quantile_99 gauge\n");
      contains(local + "_count " + std::to_string(metrics[index]) + "\n");
      contains(pd + "_count" + role + " " + std::to_string(pd_metrics[index]) +
               "\n");
      EXPECT_EQ(text.find(local + "_bucket"), std::string::npos);
      EXPECT_EQ(text.find(pd + "_bucket"), std::string::npos);
    }
    contains("# TYPE rpc_server_t3_export_summary summary\n");
    contains("rpc_server_t3_export_summary_count 2\n");
    const std::string metadata =
        "# TYPE t3_export_labeled_gauge_role_decode gauge\n";
    contains(metadata);
    EXPECT_EQ(text.find(metadata), text.rfind(metadata));
    contains("t3_export_labeled_gauge_role_decode 2");
    EXPECT_EQ(local_counts(), metrics);
    EXPECT_EQ(pd_counts(), pd_metrics);
  }
  server.Stop(/*timeout_ms=*/0);
  server.Join();
}

}  // namespace
}  // namespace xllm
