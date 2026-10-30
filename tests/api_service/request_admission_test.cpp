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

#include "api_service/request_admission.h"

#include <absl/container/flat_hash_set.h>
#include <brpc/controller.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "api_service/api_service.h"
#include "api_service/http_auth.h"
#include "api_service/non_stream_call.h"
#include "api_service/rpc_request_metrics.h"
#include "api_service/stream_call.h"
#include "completion.pb.h"
#include "core/common/metrics.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/service_config.h"
#include "core/util/closure_guard.h"

namespace xllm {
namespace {

class CountingClosure final : public google::protobuf::Closure {
 public:
  void Run() override { ++count_; }

  int32_t count() const { return count_; }

 private:
  int32_t count_ = 0;
};

class ReleasedStreamCall final
    : public StreamCall<proto::CompletionRequest, proto::CompletionResponse> {
 public:
  using StreamCall::StreamCall;

  void simulate_handoff_and_finish(bool disconnected) {
    stream_started_ = true;
    response_complete_ = !disconnected;
    connection_status_ = disconnected ? -1 : 0;
    controller_ = nullptr;
  }
};

class CanceledController final : public brpc::Controller {
 public:
  bool IsCanceled() const override { return true; }
};

class ConnectedController final : public brpc::Controller {
 public:
  // A bare brpc Controller has no peer socket and reports IsCanceled().
  // This fixture models the live connection owned by a normal server call.
  bool IsCanceled() const override { return false; }
};

class EmbeddingsEntryMetricsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_node_rank_ = DistributedConfig::get_instance().node_rank();
    previous_api_key_ = ServiceConfig::get_instance().api_key();
    // A non-master rank skips backend construction for the no-service path.
    DistributedConfig::get_instance().node_rank(1);
  }

  void TearDown() override {
    DistributedConfig::get_instance().node_rank(previous_node_rank_);
    ServiceConfig::get_instance().api_key(previous_api_key_);
  }

 private:
  int32_t previous_node_rank_ = 0;
  std::string previous_api_key_;
};

TEST_F(EmbeddingsEntryMetricsTest, NoServiceEarlyExitsCountOnce) {
  for (bool authentication : {false, true}) {
    SCOPED_TRACE(authentication);
    ServiceConfig::get_instance().api_key(authentication ? "test-key" : "");
    APIService service(/*master=*/nullptr,
                       /*model_names=*/{"test"},
                       /*model_repository_names=*/{"test"},
                       /*model_versions=*/{"1"});
    ConnectedController controller;
    controller.http_request().uri() = "/v1/embeddings";
    proto::HttpRequest request;
    proto::HttpResponse response;
    CountingClosure done;
    const char* outcome = authentication ? "authentication" : "invalid_request";
    const double incoming = COUNTER_VALUE(server_request_in_total);
    const double successes = COUNTER_VALUE(server_request_total_ok);
    const double failures = COUNTER_VALUE(server_request_total_fail);
    const double classified =
        MULTI_COUNTER_VALUE(api_request_terminal_total, outcome);

    service.EmbeddingsHttp(&controller, &request, &response, &done);

    EXPECT_EQ(controller.http_response().status_code(),
              authentication ? 401 : 400);
    EXPECT_EQ(done.count(), 1);
    EXPECT_EQ(COUNTER_VALUE(server_request_in_total), incoming + 1);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), successes);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), failures + 1);
    EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, outcome),
              classified + 1);

    ::testing::internal::CaptureStderr();
    service.EmbeddingsHttp(nullptr, &request, &response, &done);
    const std::string error_log = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(error_log.find("brpc controller is null"), std::string::npos);
    EXPECT_EQ(error_log.find("ctrl is nullptr"), std::string::npos);
    EXPECT_EQ(done.count(), 2);
    EXPECT_EQ(COUNTER_VALUE(server_request_in_total), incoming + 2);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), successes);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), failures + 1);
    EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, outcome),
              classified + 1);
  }
}

TEST(ApiTerminalMetricsTest, SuccessfulHandoffCountsOnceAfterGuardDestruction) {
  ConnectedController controller;
  ASSERT_FALSE(controller.Failed());
  ASSERT_FALSE(controller.IsCanceled());
  proto::CompletionRequest request;
  proto::CompletionResponse response;
  CountingClosure done;
  std::shared_ptr<RequestTiming> timing;
  const double successes = COUNTER_VALUE(server_request_total_ok);
  const double failures = COUNTER_VALUE(server_request_total_fail);
  const double terminal_successes =
      MULTI_COUNTER_VALUE(api_request_terminal_total, "success");
  {
    ClosureGuard guard(
        &done,
        [](void* context) { request_in_metric(context); },
        [&controller](void* /*unused*/) { request_out_metric(&controller); });
    {
      NonStreamCall<proto::CompletionRequest, proto::CompletionResponse> call(
          &controller,
          guard.release(),
          &request,
          &response,
          /*use_arena=*/true,
          /*is_http_request=*/true);
      timing = call.request_timing();
      timing->mark_pd_local();
      timing->update_from_output();
      ASSERT_TRUE(call.write_and_finish(response));
      EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, "success"),
                terminal_successes);
    }
    EXPECT_EQ(done.count(), 1);
  }
  EXPECT_EQ(done.count(), 1);
  EXPECT_FALSE(timing->snapshot().terminal);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), successes + 1);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), failures);
  EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, "success"),
            terminal_successes + 1);
}

TEST(ApiTerminalMetricsTest, DisconnectedStreamRetainsTransportOutcome) {
  const Status status(StatusCode::DEADLINE_EXCEEDED);
  const double cancelled =
      MULTI_COUNTER_VALUE(api_request_terminal_total, "cancelled");
  const double timeouts =
      MULTI_COUNTER_VALUE(api_request_terminal_total, "timeout");
  record_api_request_terminal(nullptr,
                              /*response_complete=*/false,
                              &status,
                              /*stream_disconnected=*/true,
                              "disconnected-error-stream",
                              /*controller_released=*/true);
  EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, "cancelled"),
            cancelled + 1);
  EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, "timeout"),
            timeouts);
}

TEST(ApiTerminalMetricsTest, ClassifiesExplicitFailuresWithoutHttpState) {
  struct FailureCase {
    Status status;
    const char* outcome;
  };
  const FailureCase cases[] = {
      {Status(StatusCode::UNKNOWN, "", FailureReason::UNAUTHENTICATED),
       "authentication"},
      {Status(StatusCode::UNKNOWN, "", FailureReason::PERMISSION_DENIED),
       "authentication"},
      {Status(StatusCode::UNKNOWN, "", FailureReason::MODEL_NOT_FOUND),
       "model_not_found"},
      {Status(StatusCode::RESOURCE_EXHAUSTED,
              "",
              FailureReason::CONCURRENCY_LIMIT_REACHED),
       "rate_limited"},
      {Status(StatusCode::RESOURCE_EXHAUSTED,
              "",
              FailureReason::TEMPORARY_OVERLOAD),
       "unavailable"},
      {Status::invalid_parameter("invalid value", "temperature"),
       "invalid_request"},
      {Status(StatusCode::INVALID_ARGUMENT, "", FailureReason::PROMPT_TOO_LONG),
       "invalid_request"},
      {Status(StatusCode::CANCELLED), "cancelled"},
      {Status(StatusCode::DEADLINE_EXCEEDED), "timeout"},
      {Status(StatusCode::INVALID_ARGUMENT), "invalid_request"},
      {Status(StatusCode::UNAVAILABLE), "unavailable"},
      {Status(StatusCode::RESOURCE_EXHAUSTED), "internal_error"},
      {Status(StatusCode::UNKNOWN), "internal_error"},
      {Status(
           StatusCode::INVALID_ARGUMENT, "", static_cast<FailureReason>(999)),
       "internal_error"}};
  for (const FailureCase& test_case : cases) {
    SCOPED_TRACE(test_case.outcome);
    const double before =
        MULTI_COUNTER_VALUE(api_request_terminal_total, test_case.outcome);
    // After SSE handoff the controller may already have been destroyed.
    record_api_request_terminal(nullptr,
                                /*response_complete=*/false,
                                &test_case.status,
                                /*stream_disconnected=*/false,
                                "released-error-stream",
                                /*controller_released=*/true);
    EXPECT_EQ(
        MULTI_COUNTER_VALUE(api_request_terminal_total, test_case.outcome),
        before + 1);
  }
}

TEST(ApiTerminalMetricsTest, UnknownPublicCodeRetainsHttpClassification) {
  struct ResponseCase {
    int32_t status_code;
    const char* outcome;
  };
  const ResponseCase cases[] = {{400, "invalid_request"},
                                {404, "invalid_request"},
                                {401, "authentication"},
                                {403, "authentication"},
                                {429, "rate_limited"},
                                {503, "unavailable"},
                                {504, "timeout"},
                                {500, "internal_error"}};
  for (const ResponseCase& test_case : cases) {
    SCOPED_TRACE(test_case.status_code);
    CanceledController controller;
    controller.http_response().set_status_code(test_case.status_code);
    controller.http_response().SetHeader("X-Xllm-Error-Code", "future_code");
    const double before =
        MULTI_COUNTER_VALUE(api_request_terminal_total, test_case.outcome);
    // A committed HTTP error must win over cancellation at teardown.
    request_out_metric(&controller);
    EXPECT_EQ(
        MULTI_COUNTER_VALUE(api_request_terminal_total, test_case.outcome),
        before + 1);
  }
}

TEST(ApiTerminalMetricsTest, UnclassifiedResourceExhaustionIsInternalError) {
  const double before =
      MULTI_COUNTER_VALUE(api_request_terminal_total, "internal_error");
  brpc::Controller controller;
  proto::CompletionRequest request;
  proto::CompletionResponse response;
  CountingClosure done;
  {
    NonStreamCall<proto::CompletionRequest, proto::CompletionResponse> call(
        &controller,
        &done,
        &request,
        &response,
        /*use_arena=*/true,
        /*is_http_request=*/true);
    call.request_timing()->record_arrival();
    EXPECT_TRUE(call.finish_with_error(
        Status(StatusCode::RESOURCE_EXHAUSTED, "capacity exhausted")));
    EXPECT_EQ(controller.http_response().status_code(), 500);
    const auto sample = call.request_timing()->snapshot();
    EXPECT_TRUE(sample.terminal);
    EXPECT_FALSE(sample.server_ttft.valid);
    EXPECT_TRUE(sample.server_ttft.partial);
    EXPECT_FALSE(sample.queued_time.milliseconds.has_value());
  }
  EXPECT_EQ(done.count(), 1);
  EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, "internal_error"),
            before + 1);
}

TEST(ApiTerminalMetricsTest, NonStreamCancellationWinsOverBrpcFailure) {
  const double before =
      MULTI_COUNTER_VALUE(api_request_terminal_total, "cancelled");
  CanceledController controller;
  controller.SetFailed("client disconnected");
  ASSERT_TRUE(controller.Failed());
  ASSERT_TRUE(controller.IsCanceled());
  ASSERT_LT(controller.http_response().status_code(), 400);
  proto::CompletionRequest request;
  proto::CompletionResponse response;
  CountingClosure done;
  std::shared_ptr<RequestTiming> timing;
  {
    NonStreamCall<proto::CompletionRequest, proto::CompletionResponse> call(
        &controller,
        &done,
        &request,
        &response,
        /*use_arena=*/true,
        /*is_http_request=*/true);
    timing = call.request_timing();
    timing->record_queued();
  }
  EXPECT_TRUE(timing->snapshot().terminal);
  EXPECT_TRUE(timing->snapshot().queued_time.partial);
  EXPECT_FALSE(timing->snapshot().decode_time.milliseconds.has_value());
  EXPECT_EQ(done.count(), 1);
  EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, "cancelled"),
            before + 1);
}

TEST(ApiTerminalMetricsTest, StreamExitDoesNotReadReleasedController) {
  for (bool disconnected : {false, true}) {
    const char* outcome = disconnected ? "cancelled" : "success";
    const double before =
        MULTI_COUNTER_VALUE(api_request_terminal_total, outcome);
    brpc::Controller controller;
    controller.http_request().SetHeader("x-request-id", "released-stream");
    proto::CompletionRequest request;
    request.set_stream(true);
    proto::CompletionResponse response;
    CountingClosure done;
    std::shared_ptr<RequestTiming> timing;
    {
      ReleasedStreamCall call(&controller,
                              &done,
                              &request,
                              &response,
                              /*use_arena=*/true,
                              /*is_http_request=*/true);
      timing = call.request_timing();
      timing->update_from_output();
      if (!disconnected) {
        timing->update_from_finished_request(/*generation_complete=*/true);
      }
      done.Run();  // Opening SSE hands the RPC controller back to brpc.
      call.simulate_handoff_and_finish(disconnected);
    }
    EXPECT_TRUE(timing->snapshot().terminal);
    EXPECT_EQ(timing->snapshot().decode_time.valid, !disconnected);
    EXPECT_EQ(done.count(), 1);
    EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, outcome),
              before + 1);
  }
}

class RateLimitMetricsTest : public testing::Test {
 protected:
  void SetUp() override {
    previous_limit_ = ServiceConfig::get_instance().max_concurrent_requests();
    ServiceConfig::get_instance().max_concurrent_requests(1);
  }

  void TearDown() override {
    ServiceConfig::get_instance().max_concurrent_requests(previous_limit_);
  }

  template <typename CallType>
  void finish_rejection(const Status& status, bool streaming, bool http) {
    brpc::Controller controller;
    controller.http_request().SetHeader("x-request-id", "limit-metrics-test");
    proto::CompletionRequest request;
    request.set_stream(streaming);
    proto::CompletionResponse response;
    CountingClosure done;
    const double limited_before = COUNTER_VALUE(server_request_total_limit);
    const double failed_before = COUNTER_VALUE(server_request_total_fail);
    const double ok_before = COUNTER_VALUE(server_request_total_ok);
    const double terminal_before =
        MULTI_COUNTER_VALUE(api_request_terminal_total, "rate_limited");
    ClosureGuard guard(
        &done,
        [](void* context) { request_in_metric(context); },
        [&controller](void* /*unused*/) { request_out_metric(&controller); });
    {
      CallType call(&controller,
                    guard.release(),
                    &request,
                    &response,
                    /*use_arena=*/true,
                    http);
      EXPECT_TRUE(call.finish_with_error(status));
      EXPECT_EQ(COUNTER_VALUE(server_request_total_limit), limited_before);
      EXPECT_EQ(call.finish_with_error(status), !streaming);
      EXPECT_EQ(COUNTER_VALUE(server_request_total_limit), limited_before);
      if (http) {
        EXPECT_FALSE(controller.Failed());
        EXPECT_EQ(controller.http_response().status_code(),
                  api_service::public_error_from_status(status).status_code);
      } else {
        EXPECT_TRUE(controller.Failed());
      }
    }
    EXPECT_EQ(done.count(), 1);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_limit), limited_before);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), failed_before + 1);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), ok_before);
    if (status.reason() == FailureReason::CONCURRENCY_LIMIT_REACHED) {
      EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, "rate_limited"),
                terminal_before + 1);
    }
  }

 private:
  int32_t previous_limit_ = 0;
};

TEST_F(RateLimitMetricsTest, NonStreamConcurrencyRejectionCountsOnce) {
  using CompletionCall =
      NonStreamCall<proto::CompletionRequest, proto::CompletionResponse>;
  RateLimiter rate_limiter;
  const absl::flat_hash_set<std::string> models = {"supported-model"};
  const double limited_before = COUNTER_VALUE(server_request_total_limit);
  ASSERT_TRUE(api_service_internal::admit_rpc_request(
                  "supported-model", models, &rate_limiter)
                  .ok());
  EXPECT_EQ(COUNTER_VALUE(server_request_total_limit), limited_before);
  for (bool http : {false, true}) {
    const double before_rejection = COUNTER_VALUE(server_request_total_limit);
    const Status status = api_service_internal::admit_rpc_request(
        "supported-model", models, &rate_limiter);
    ASSERT_EQ(status.reason(), FailureReason::CONCURRENCY_LIMIT_REACHED);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_limit), before_rejection + 1);
    finish_rejection<CompletionCall>(status, /*streaming=*/false, http);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_limit), before_rejection + 1);
    EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);
  }
  rate_limiter.decrease_one_request();
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
}

TEST_F(RateLimitMetricsTest, StreamCallConcurrencyRejectionCountsOnce) {
  using CompletionCall =
      StreamCall<proto::CompletionRequest, proto::CompletionResponse>;
  RateLimiter rate_limiter;
  const absl::flat_hash_set<std::string> models = {"supported-model"};
  ASSERT_TRUE(api_service_internal::admit_rpc_request(
                  "supported-model", models, &rate_limiter)
                  .ok());
  for (bool streaming : {false, true}) {
    const double before_rejection = COUNTER_VALUE(server_request_total_limit);
    const Status status = api_service_internal::admit_rpc_request(
        "supported-model", models, &rate_limiter);
    ASSERT_EQ(status.reason(), FailureReason::CONCURRENCY_LIMIT_REACHED);
    finish_rejection<CompletionCall>(status, streaming, /*http=*/true);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_limit), before_rejection + 1);
    EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);
  }
  rate_limiter.decrease_one_request();
}

TEST_F(RateLimitMetricsTest, SleepingModelDoesNotCountAsConcurrencyLimit) {
  using CompletionCall =
      NonStreamCall<proto::CompletionRequest, proto::CompletionResponse>;
  RateLimiter rate_limiter;
  const absl::flat_hash_set<std::string> models = {"supported-model"};
  ASSERT_TRUE(rate_limiter.try_set_sleeping());
  const double limited_before = COUNTER_VALUE(server_request_total_limit);
  const Status status = api_service_internal::admit_rpc_request(
      "supported-model", models, &rate_limiter);
  ASSERT_EQ(status.code(), StatusCode::UNAVAILABLE);
  finish_rejection<CompletionCall>(status, /*streaming=*/false, /*http=*/true);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_limit), limited_before);
  EXPECT_TRUE(rate_limiter.is_sleeping());
  EXPECT_TRUE(rate_limiter.try_wakeup());
}

TEST_F(RateLimitMetricsTest, UnsupportedModelDoesNotCountAsConcurrencyLimit) {
  using CompletionCall =
      NonStreamCall<proto::CompletionRequest, proto::CompletionResponse>;
  RateLimiter rate_limiter;
  const absl::flat_hash_set<std::string> models = {"supported-model"};
  const double limited_before = COUNTER_VALUE(server_request_total_limit);
  const Status status = api_service_internal::admit_rpc_request(
      "unsupported-model", models, &rate_limiter);
  ASSERT_EQ(status.reason(), FailureReason::MODEL_NOT_FOUND);
  finish_rejection<CompletionCall>(status, /*streaming=*/false, /*http=*/true);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_limit), limited_before);
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
}

TEST_F(RateLimitMetricsTest, EarlyHttp429WithoutCallCountsOnce) {
  brpc::Controller controller;
  CountingClosure done;
  const double limited_before = COUNTER_VALUE(server_request_total_limit);
  const double failed_before = COUNTER_VALUE(server_request_total_fail);
  const double terminal_before =
      MULTI_COUNTER_VALUE(api_request_terminal_total, "rate_limited");
  {
    ClosureGuard guard(
        &done,
        [](void* context) { request_in_metric(context); },
        [&controller](void* /*unused*/) { request_out_metric(&controller); });
    api_service::write_http_error(
        &controller,
        Status(StatusCode::RESOURCE_EXHAUSTED,
               "concurrency limit reached",
               FailureReason::CONCURRENCY_LIMIT_REACHED));
  }
  EXPECT_EQ(done.count(), 1);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_limit), limited_before + 1);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), failed_before + 1);
  EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, "rate_limited"),
            terminal_before + 1);
}

TEST(RequestAdmissionTest, EarlyHttpErrorsHaveOneClassifiedApiTerminal) {
  const api_service::HttpBearerAuthenticator authenticator("expected-key");
  const Status model_not_found(
      StatusCode::UNKNOWN, "missing model", FailureReason::MODEL_NOT_FOUND);
  for (const std::string& outcome :
       {"authentication", "invalid_request", "model_not_found"}) {
    brpc::Controller controller;
    CountingClosure done;
    const double before =
        MULTI_COUNTER_VALUE(api_request_terminal_total, outcome);
    {
      ClosureGuard guard(
          &done,
          [](void* context) { request_in_metric(context); },
          [&controller](void* /*unused*/) { request_out_metric(&controller); });
      if (outcome == "authentication") {
        EXPECT_FALSE(authenticator.prepare_request(&controller));
      } else if (outcome == "invalid_request") {
        api_service::write_bad_request(&controller, "malformed JSON");
      } else {
        api_service::write_http_error(&controller, model_not_found);
      }
    }
    EXPECT_EQ(done.count(), 1);
    EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, outcome),
              before + 1);
    EXPECT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  }
}

TEST(RequestAdmissionTest, UnsupportedModelDoesNotConsumeConcurrencySlot) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;
  const absl::flat_hash_set<std::string> models = {"supported-model"};

  for (int32_t attempt = 0; attempt < 2; ++attempt) {
    const Status status = api_service_internal::admit_rpc_request(
        "unsupported-model", models, &rate_limiter);

    EXPECT_EQ(status.code(), StatusCode::UNKNOWN);
    EXPECT_EQ(status.message(), "Model not supported");
    EXPECT_EQ(status.reason(), FailureReason::MODEL_NOT_FOUND);
    EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
  }

  const Status status = api_service_internal::admit_rpc_request(
      "supported-model", models, &rate_limiter);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);
  rate_limiter.decrease_one_request();
}

TEST(RequestAdmissionTest, FullConcurrencyLimitPreserves429Cause) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;
  const absl::flat_hash_set<std::string> models = {"supported-model"};

  EXPECT_TRUE(api_service_internal::admit_rpc_request(
                  "supported-model", models, &rate_limiter)
                  .ok());
  const Status rejected = api_service_internal::admit_rpc_request(
      "supported-model", models, &rate_limiter);
  EXPECT_EQ(rejected.code(), StatusCode::RESOURCE_EXHAUSTED);
  EXPECT_EQ(rejected.reason(), FailureReason::CONCURRENCY_LIMIT_REACHED);
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);
  rate_limiter.decrease_one_request();
}

TEST(RequestAdmissionTest, SleepingModelIsNotAConcurrencyLimit) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;
  const absl::flat_hash_set<std::string> models = {"supported-model"};

  ASSERT_TRUE(rate_limiter.try_set_sleeping());
  const Status rejected = api_service_internal::admit_rpc_request(
      "supported-model", models, &rate_limiter);
  EXPECT_EQ(rejected.code(), StatusCode::UNAVAILABLE);
  EXPECT_NE(rejected.reason(), FailureReason::CONCURRENCY_LIMIT_REACHED);
  EXPECT_TRUE(rate_limiter.is_sleeping());
  EXPECT_TRUE(rate_limiter.try_wakeup());
}

}  // namespace
}  // namespace xllm

namespace xllm::api_service {
namespace {
TEST(PublicErrorTest, MapsAllPublicErrorCodes) {
  struct ErrorCase {
    StatusCode status;
    FailureReason reason;
    int32_t http_status;
    const char* code;
  };
  const ErrorCase cases[] = {{StatusCode::INVALID_ARGUMENT,
                              FailureReason::UNSPECIFIED,
                              400,
                              "invalid_request"},
                             {StatusCode::INVALID_ARGUMENT,
                              FailureReason::INVALID_PARAMETER,
                              400,
                              "invalid_parameter"},
                             {StatusCode::RESOURCE_EXHAUSTED,
                              FailureReason::PROMPT_TOO_LONG,
                              400,
                              "context_length_exceeded"},
                             {StatusCode::UNKNOWN,
                              FailureReason::UNAUTHENTICATED,
                              401,
                              "invalid_api_key"},
                             {StatusCode::UNKNOWN,
                              FailureReason::PERMISSION_DENIED,
                              403,
                              "permission_denied"},
                             {StatusCode::UNKNOWN,
                              FailureReason::MODEL_NOT_FOUND,
                              404,
                              "model_not_found"},
                             {StatusCode::RESOURCE_EXHAUSTED,
                              FailureReason::CONCURRENCY_LIMIT_REACHED,
                              429,
                              "rate_limit_exceeded"},
                             {StatusCode::RESOURCE_EXHAUSTED,
                              FailureReason::TEMPORARY_OVERLOAD,
                              503,
                              "server_is_overloaded"},
                             {StatusCode::UNAVAILABLE,
                              FailureReason::UNSPECIFIED,
                              503,
                              "service_unavailable"},
                             {StatusCode::DEADLINE_EXCEEDED,
                              FailureReason::UNSPECIFIED,
                              504,
                              "request_timeout"},
                             {StatusCode::RESOURCE_EXHAUSTED,
                              FailureReason::UNSPECIFIED,
                              500,
                              "internal_error"}};
  for (const ErrorCase& row : cases) {
    SCOPED_TRACE(row.code);
    brpc::Controller controller;
    write_http_error(&controller,
                     Status(row.status, "test failure", row.reason));
    EXPECT_FALSE(controller.Failed());
    EXPECT_EQ(controller.http_response().status_code(), row.http_status);
    const auto body =
        nlohmann::json::parse(controller.response_attachment().to_string());
    EXPECT_EQ(body["error"]["code"], row.code);
    EXPECT_TRUE(body["error"]["param"].is_null());
  }
}

TEST(PublicErrorTest, RejectsMissingOrIncorrectBearerKeyBeforeStreaming) {
  const HttpBearerAuthenticator auth("test-key");

  brpc::Controller missing;
  EXPECT_FALSE(auth.prepare_request(&missing));
  EXPECT_NE(missing.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_FALSE(missing.Failed());
  EXPECT_EQ(missing.http_response().status_code(), 401);
  ASSERT_NE(missing.http_response().GetHeader("WWW-Authenticate"), nullptr);
  EXPECT_EQ(*missing.http_response().GetHeader("WWW-Authenticate"), "Bearer");
  const nlohmann::json missing_payload =
      nlohmann::json::parse(missing.response_attachment().to_string());
  EXPECT_EQ(missing_payload["error"]["type"], "authentication_error");
  EXPECT_EQ(missing_payload["error"]["code"], "invalid_api_key");
  EXPECT_TRUE(missing_payload["error"]["param"].is_null());
  EXPECT_EQ(missing.http_response().GetHeader("Retry-After"), nullptr);

  for (const std::string& header : {"Bearer wrong-key",
                                    "Basic test-key",
                                    "Bearer test-key-extra",
                                    "Bearer"}) {
    brpc::Controller invalid;
    invalid.http_request().SetHeader("Authorization", header);
    EXPECT_FALSE(auth.prepare_request(&invalid)) << header;
    EXPECT_EQ(invalid.http_response().status_code(), 401) << header;
    EXPECT_EQ(nlohmann::json::parse(
                  invalid.response_attachment().to_string())["error"]["code"],
              "invalid_api_key");
  }
}

TEST(PublicErrorTest, AllowsValidBearerKeyAndDisabledAuthentication) {
  const HttpBearerAuthenticator auth("test-key");
  brpc::Controller valid;
  valid.http_request().SetHeader("Authorization", "bEaReR test-key");
  EXPECT_TRUE(auth.prepare_request(&valid));
  EXPECT_NE(valid.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_TRUE(valid.response_attachment().empty());

  const HttpBearerAuthenticator disabled("");
  brpc::Controller no_key;
  EXPECT_TRUE(disabled.prepare_request(&no_key));
  EXPECT_NE(no_key.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_TRUE(no_key.response_attachment().empty());
}

TEST(PublicErrorTest, FormatsAnthropicAuthenticationError) {
  const HttpBearerAuthenticator auth("test-key");
  brpc::Controller anthropic;
  EXPECT_FALSE(auth.prepare_request(&anthropic, /*anthropic=*/true));
  EXPECT_EQ(anthropic.http_response().status_code(), 401);
  const nlohmann::json auth_payload =
      nlohmann::json::parse(anthropic.response_attachment().to_string());
  EXPECT_EQ(auth_payload["type"], "error");
  EXPECT_EQ(auth_payload["error"]["type"], "authentication_error");
  EXPECT_EQ(auth_payload["error"].size(), 2);
}

TEST(PublicErrorTest, RedactsUnclassifiedInternalFailuresForBothProtocols) {
  for (StatusCode code : {StatusCode::RESOURCE_EXHAUSTED,
                          StatusCode::CANCELLED,
                          StatusCode::UNKNOWN}) {
    const PublicHttpError error =
        public_error_from_status(Status(code, "private device/path detail"));
    EXPECT_EQ(error.status_code, 500);
    EXPECT_EQ(error.code, "internal_error");
    EXPECT_EQ(openai_error_payload(error)["error"]["type"], "server_error");
    EXPECT_EQ(anthropic_error_payload(error)["error"]["type"], "api_error");
    EXPECT_EQ(error.message.find("private"), std::string::npos);
  }
}

TEST(PublicErrorTest, RendersPublicOverloadUsingEachProtocolVocabulary) {
  const PublicHttpError error =
      public_error_from_status(Status(StatusCode::RESOURCE_EXHAUSTED,
                                      "private overload detail",
                                      FailureReason::TEMPORARY_OVERLOAD));
  EXPECT_EQ(error.status_code, 503);
  EXPECT_EQ(openai_error_payload(error)["error"]["type"],
            "service_unavailable_error");
  EXPECT_EQ(anthropic_error_payload(error)["error"]["type"],
            "overloaded_error");
  EXPECT_EQ(openai_error_payload(error)["error"]["message"],
            anthropic_error_payload(error)["error"]["message"]);
}
}  // namespace
}  // namespace xllm::api_service
