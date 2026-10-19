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
#include <nlohmann/json.hpp>
#include <string>

#include "api_service/http_auth.h"
#include "api_service/non_stream_call.h"
#include "api_service/rpc_request_metrics.h"
#include "api_service/stream_call.h"
#include "completion.pb.h"
#include "core/common/metrics.h"
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
