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
#include "core/framework/config/service_config.h"

namespace xllm {
namespace {

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
