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

#include "api_service/request_id.h"

#include <brpc/channel.h>
#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "anthropic.pb.h"
#include "api_service/api_service_impl.h"
#include "api_service/non_stream_call.h"
#include "api_service/rpc_request_metrics.h"
#include "api_service/stream_call.h"
#include "chat.pb.h"
#include "completion.pb.h"
#include "core/runtime/xservice_client.h"
#include "core/util/closure_guard.h"
#include "xllm_service.pb.h"
#include "xservice.pb.h"

namespace xllm {
namespace {

class NoopClosure final : public google::protobuf::Closure {
 public:
  void Run() override {}
};

using CompletionCallForTest =
    NonStreamCall<proto::CompletionRequest, proto::CompletionResponse>;

class CountingClosure final : public google::protobuf::Closure {
 public:
  explicit CountingClosure(int32_t* count) : count_(count) {}

  void Run() override { ++*count_; }

 private:
  int32_t* count_;
};

class GenerationReceiver final : public xllm_service::proto::XllmRpcService {
 public:
  void Generations(google::protobuf::RpcController* /*unused*/,
                   const proto::DisaggStreamGenerations* request,
                   proto::StatusSet* response,
                   google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    received = *request;
    for (int32_t index = 0; index < request->gens_size(); ++index) {
      response->add_all_status()->set_ok(true);
    }
  }

  proto::DisaggStreamGenerations received;
};

TEST(PublicErrorWireTest, SenderUsesSamePublicErrorAsDirectHttp) {
  GenerationReceiver receiver;
  brpc::Server server;
  ASSERT_EQ(server.AddService(&receiver, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  const std::string address =
      "127.0.0.1:" + std::to_string(server.listen_address().port);
  const std::vector<Status> statuses = {
      Status::invalid_parameter("invalid value", "temperature"),
      Status::invalid_parameter("", "temperature"),
      Status(StatusCode::INVALID_ARGUMENT),
      Status(StatusCode::INVALID_ARGUMENT, "invalid request"),
      Status(StatusCode::UNKNOWN, "private model detail"),
      Status(StatusCode::UNAVAILABLE),
      Status(StatusCode::DEADLINE_EXCEEDED),
      Status(StatusCode::RESOURCE_EXHAUSTED,
             "",
             FailureReason::TEMPORARY_OVERLOAD),
      Status(StatusCode::RESOURCE_EXHAUSTED,
             "",
             FailureReason::CONCURRENCY_LIMIT_REACHED),
      Status(StatusCode::INVALID_ARGUMENT, "", FailureReason::PROMPT_TOO_LONG),
      Status(StatusCode::INVALID_ARGUMENT, "", FailureReason::MODEL_NOT_FOUND),
      Status(StatusCode::UNKNOWN, "", FailureReason::UNAUTHENTICATED),
      Status(StatusCode::UNKNOWN, "", FailureReason::PERMISSION_DENIED),
      Status(StatusCode::CANCELLED, "", FailureReason::UNSPECIFIED, ""),
      Status()};
  std::vector<RequestOutput> outputs;
  outputs.reserve(statuses.size() + 1);
  for (const Status& status : statuses) {
    RequestOutput output;
    output.target_xservice_addr = address;
    output.status = status;
    outputs.emplace_back(std::move(output));
  }
  RequestOutput token;
  token.target_xservice_addr = address;
  outputs.emplace_back(std::move(token));
  EXPECT_EQ(XServiceClient::get_instance()->generations(outputs),
            std::vector<bool>(outputs.size(), true));
  ASSERT_EQ(receiver.received.gens_size(), outputs.size());
  for (size_t index = 0; index < statuses.size(); ++index) {
    const Status& status = statuses[index];
    const auto& wire = receiver.received.gens(index).gen_status();
    EXPECT_EQ(wire.status_code(), static_cast<int32_t>(status.code()));
    EXPECT_EQ(wire.status_msg(), status.message());
    EXPECT_EQ(wire.has_failure_reason(),
              status.reason() != FailureReason::UNSPECIFIED);
    EXPECT_EQ(wire.failure_reason(), static_cast<int32_t>(status.reason()));
    EXPECT_EQ(wire.has_param(), status.param().has_value());
    EXPECT_EQ(wire.param(), status.param().value_or(""));
    ASSERT_EQ(wire.has_error(), !status.ok());
    if (status.ok()) {
      continue;
    }
    const PublicHttpError error = public_error_from_status(status);
    EXPECT_EQ(wire.error().http_status_code(), error.status_code);
    EXPECT_EQ(wire.error().type(), error.type);
    EXPECT_EQ(wire.error().code(), error.code);
    EXPECT_EQ(wire.error().message(), error.message);
    EXPECT_EQ(wire.error().retryable(), error.retryable);
    EXPECT_EQ(wire.error().has_param(), error.param.has_value());
    EXPECT_EQ(wire.error().param(), error.param.value_or(""));
    brpc::Controller controller;
    api_service::write_http_error(&controller, status);
    EXPECT_EQ(controller.http_response().status_code(),
              wire.error().http_status_code());
    EXPECT_EQ(
        nlohmann::json::parse(controller.response_attachment().to_string()),
        api_service::openai_error_payload(error));
  }
  EXPECT_FALSE(receiver.received.gens(statuses.size()).has_gen_status());
  server.Stop(0);
  server.Join();
}

template <typename CallType>
class PayloadService final : public APIServiceImpl<CallType> {
 public:
  PayloadService() : APIServiceImpl<CallType>({"test"}) {}

  void process_async_impl(std::shared_ptr<CallType> /*call*/) override {
    ++processed_;
  }

  int32_t processed() const { return processed_; }

 private:
  int32_t processed_ = 0;
};

TEST(RequestPayloadTest, InvalidLengthHeadersReturnHttp400WithoutDispatch) {
  struct InvalidHeaderCase {
    std::string infer_length;
    std::string content_length;
    std::string param;
  };
  const std::vector<InvalidHeaderCase> cases = {
      {"nonnumeric", "6", "Infer-Content-Length"},
      {"2x", "6", "Infer-Content-Length"},
      {"-1", "6", "Infer-Content-Length"},
      {"184467440737095516160", "6", "Infer-Content-Length"},
      {"7", "6", "Infer-Content-Length"},
      {"2", "7", "Content-Length"},
      {"2", "bad", "Content-Length"}};
  for (const InvalidHeaderCase& test_case : cases) {
    SCOPED_TRACE(test_case.infer_length + "/" + test_case.content_length);
    brpc::Controller controller;
    controller.http_request().SetHeader("Infer-Content-Length",
                                        test_case.infer_length);
    controller.http_request().SetHeader("Content-Length",
                                        test_case.content_length);
    controller.request_attachment().append("ab1234");
    proto::CompletionRequest request;
    proto::CompletionResponse response;
    int32_t done_count = 0;
    CountingClosure done(&done_count);
    PayloadService<CompletionCallForTest> service;

    {
      std::shared_ptr<CompletionCallForTest> call =
          std::make_shared<CompletionCallForTest>(&controller,
                                                  &done,
                                                  &request,
                                                  &response,
                                                  /*use_arena=*/true,
                                                  /*is_http_request=*/true);
      service.process_async(call);
      EXPECT_EQ(service.processed(), 0);
      EXPECT_EQ(controller.http_response().status_code(), 400);
      const nlohmann::json payload =
          nlohmann::json::parse(controller.response_attachment().to_string());
      EXPECT_EQ(payload["error"]["type"], "invalid_request_error");
      EXPECT_EQ(payload["error"]["code"], "invalid_parameter");
      EXPECT_EQ(payload["error"]["param"], test_case.param);
      EXPECT_EQ(done_count, 0);
    }
    EXPECT_EQ(done_count, 1);
  }
}

TEST(RequestPayloadTest, HttpAndTypedRpcPreserveBinaryPayload) {
  for (const bool is_http : {false, true}) {
    brpc::Controller controller;
    controller.http_request().SetHeader("Infer-Content-Length", "2");
    controller.http_request().SetHeader("Content-Length", "6");
    controller.request_attachment().append("ab1234");
    proto::CompletionRequest request;
    proto::CompletionResponse response;
    NoopClosure done;

    CompletionCallForTest call(&controller,
                               &done,
                               &request,
                               &response,
                               /*use_arena=*/true,
                               is_http);
    EXPECT_TRUE(call.request_payload_status().ok());
    EXPECT_EQ(call.take_request_iobuf().to_string(), "1234");
  }
}

TEST(RequestIdTest, HeaderTakesPrecedenceOverBody) {
  brpc::Controller controller;
  controller.http_request().SetHeader("x-request-id", "header-id");
  proto::CompletionRequest request;
  request.set_x_request_id("body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id(), "header-id");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"), "header-id");
}

TEST(RequestIdTest, BodyIsUsedWhenHeaderIsMissing) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  request.set_x_request_id("body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id(), "body-id");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"), "body-id");
}

TEST(RequestIdTest, InvalidPrimaryHeaderFallsBackToSecondaryHeader) {
  brpc::Controller controller;
  controller.http_request().SetHeader("x-request-id", "invalid\r\nid");
  controller.http_request().SetHeader("x-ms-client-request-id", "fallback-id");
  proto::CompletionRequest request;
  request.set_x_request_id("body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id(), "fallback-id");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"),
            "fallback-id");
}

TEST(RequestIdTest, InvalidHeadersAreReplacedBeforeResponseHeader) {
  brpc::Controller controller;
  const std::string invalid_primary_id = "invalid\r\nprimary";
  const std::string invalid_secondary_id = "invalid\nsecondary";
  controller.http_request().SetHeader("x-request-id", invalid_primary_id);
  controller.http_request().SetHeader("x-ms-client-request-id",
                                      invalid_secondary_id);
  proto::CompletionRequest request;
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id().find("req-"), 0);
  EXPECT_NE(call.get_x_request_id(), invalid_primary_id);
  EXPECT_NE(call.get_x_request_id(), invalid_secondary_id);
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"),
            call.get_x_request_id());
}

TEST(RequestIdTest, BodyOverridesIdGeneratedBeforeParsing) {
  brpc::Controller controller;
  const std::string early_x_request_id =
      api_service::ensure_http_x_request_id(&controller);
  proto::CompletionRequest request;
  request.set_x_request_id("body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_NE(early_x_request_id, "body-id");
  EXPECT_EQ(call.get_x_request_id(), "body-id");
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"), "body-id");
}

TEST(RequestIdTest, MissingIdIsGeneratedAndReturned) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id().find("req-"), 0);
  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"),
            call.get_x_request_id());
}

TEST(RequestIdTest, InvalidBodyIdIsReplacedBeforeResponseHeader) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  request.set_x_request_id("body-id\r\nx-injected: true");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true,
                             /*is_http_request=*/true);

  EXPECT_EQ(call.get_x_request_id().find("req-"), 0);
  EXPECT_EQ(call.get_x_request_id().find('\r'), std::string::npos);
  EXPECT_EQ(call.get_x_request_id().find('\n'), std::string::npos);
}

TEST(RequestIdTest, EnforcesLengthLimit) {
  EXPECT_TRUE(api_service::is_valid_x_request_id(std::string(256, 'a')));
  EXPECT_FALSE(api_service::is_valid_x_request_id(std::string(257, 'a')));
}

TEST(RequestIdTest, TypedCallUsesBodyIdWithoutHttpResponseHeader) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  request.set_x_request_id("typed-body-id");
  proto::CompletionResponse response;
  NoopClosure done;

  CompletionCallForTest call(&controller,
                             &done,
                             &request,
                             &response,
                             /*use_arena=*/true);

  EXPECT_EQ(call.get_x_request_id(), "typed-body-id");
  EXPECT_EQ(controller.http_response().GetHeader("x-request-id"), nullptr);
}

TEST(RequestIdTest, EarlyHttpIdSurvivesSetFailed) {
  brpc::Controller controller;
  const std::string x_request_id =
      api_service::ensure_http_x_request_id(&controller);

  controller.SetFailed("request failed before parsing");

  ASSERT_NE(controller.http_response().GetHeader("x-request-id"), nullptr);
  EXPECT_EQ(*controller.http_response().GetHeader("x-request-id"),
            x_request_id);
}

TEST(RequestIdTest, ExtractsIdFromSupportedRequestBodies) {
  proto::CompletionRequest completion;
  completion.set_x_request_id("completion-id");
  EXPECT_EQ(request_body_x_request_id(&completion), "completion-id");

  proto::ChatRequest chat;
  chat.set_x_request_id("chat-id");
  EXPECT_EQ(request_body_x_request_id(&chat), "chat-id");

  proto::AnthropicMessagesRequest anthropic;
  anthropic.set_x_request_id("anthropic-id");
  EXPECT_EQ(request_body_x_request_id(&anthropic), "anthropic-id");
}

using CompletionStreamCall =
    StreamCall<proto::CompletionRequest, proto::CompletionResponse>;

class TestableCompletionStreamCall final : public CompletionStreamCall {
 public:
  using CompletionStreamCall::CompletionStreamCall;
  using CompletionStreamCall::start_stream;
};

class CallbackClosure final : public google::protobuf::Closure {
 public:
  explicit CallbackClosure(std::function<void()> callback)
      : callback_(std::move(callback)) {}

  void Run() override { callback_(); }

 private:
  std::function<void()> callback_;
};

class BlockingCancelController final : public brpc::Controller {
 public:
  bool IsCanceled() const override {
    std::unique_lock<std::mutex> lock(mutex_);
    checking_ = true;
    condition_.notify_all();
    condition_.wait(lock, [this] { return released_; });
    return false;
  }

  bool wait_for_check() {
    std::unique_lock<std::mutex> lock(mutex_);
    return condition_.wait_for(
        lock, std::chrono::seconds(5), [this] { return checking_; });
  }

  void release_check() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable condition_;
  mutable bool checking_ = false;
  bool released_ = false;
};

TEST(StreamCallTest, StreamStartWaitsForInFlightDisconnectCheck) {
  BlockingCancelController controller;
  proto::CompletionRequest request;
  request.set_stream(true);
  proto::CompletionResponse response;
  std::promise<void> done_called;
  auto done_future = done_called.get_future();
  CallbackClosure done([&done_called] { done_called.set_value(); });
  TestableCompletionStreamCall call(&controller,
                                    &done,
                                    &request,
                                    &response,
                                    /*use_arena=*/true,
                                    /*is_http_request=*/true);

  bool disconnected = true;
  std::thread checker([&] { disconnected = call.is_disconnected(); });
  const bool checking = controller.wait_for_check();
  if (!checking) {
    controller.release_check();
    checker.join();
    FAIL() << "Disconnect check did not enter IsCanceled";
  }
  std::promise<void> starting;
  auto starting_future = starting.get_future();
  std::thread writer([&] {
    starting.set_value();
    // No SSE writes are needed here: a standalone Controller has no socket.
    call.start_stream();
  });
  EXPECT_EQ(starting_future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_EQ(done_future.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  controller.release_check();
  checker.join();
  writer.join();

  EXPECT_FALSE(disconnected);
  EXPECT_EQ(done_future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_FALSE(call.is_disconnected());
}

TEST(StreamCallTest, DoneCanReleaseControllerAndReenterDisconnectCheck) {
  std::unique_ptr<brpc::Controller> controller =
      std::make_unique<brpc::Controller>();
  proto::CompletionRequest request;
  request.set_stream(true);
  proto::CompletionResponse response;
  TestableCompletionStreamCall* active_call = nullptr;
  int32_t done_count = 0;
  CallbackClosure done([&] {
    controller.reset();
    ++done_count;
    EXPECT_FALSE(active_call->is_disconnected());
  });
  {
    TestableCompletionStreamCall call(controller.get(),
                                      &done,
                                      &request,
                                      &response,
                                      /*use_arena=*/true,
                                      /*is_http_request=*/true);
    active_call = &call;
    call.start_stream();
    EXPECT_EQ(controller, nullptr);
    EXPECT_FALSE(call.is_disconnected());
    call.start_stream();
  }
  EXPECT_EQ(done_count, 1);
}

TEST(StreamCallTest, InvalidPayloadLengthRejectsBeforeOpeningSse) {
  brpc::Controller controller;
  controller.http_request().SetHeader("Infer-Content-Length", "not-a-number");
  controller.http_request().SetHeader("Content-Length", "2");
  controller.request_attachment().append("{}");
  int32_t done_count = 0;
  CountingClosure done(&done_count);
  proto::CompletionRequest request;
  request.set_stream(true);
  proto::CompletionResponse response;
  PayloadService<CompletionStreamCall> service;

  {
    std::shared_ptr<CompletionStreamCall> call =
        std::make_shared<CompletionStreamCall>(&controller,
                                               &done,
                                               &request,
                                               &response,
                                               /*use_arena=*/true,
                                               /*is_http_request=*/true);
    service.process_async(call);
    EXPECT_EQ(service.processed(), 0);
    EXPECT_EQ(controller.http_response().status_code(), 400);
    EXPECT_EQ(controller.http_response().content_type(),
              "application/json; charset=utf-8");
    const nlohmann::json payload =
        nlohmann::json::parse(controller.response_attachment().to_string());
    EXPECT_EQ(payload["error"]["param"], "Infer-Content-Length");
    EXPECT_EQ(done_count, 0);
  }

  EXPECT_EQ(done_count, 1);
}

class StreamTestService final : public proto::XllmAPIService {
 public:
  void CompletionsHttp(google::protobuf::RpcController* controller,
                       const proto::HttpRequest* /*request*/,
                       proto::HttpResponse* /*response*/,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    auto* http_controller = static_cast<brpc::Controller*>(controller);
    std::unique_ptr<proto::CompletionRequest> request =
        std::make_unique<proto::CompletionRequest>();
    request->set_stream(true);
    std::unique_ptr<proto::CompletionResponse> response =
        std::make_unique<proto::CompletionResponse>();
    StreamCall<proto::CompletionRequest, proto::CompletionResponse> call(
        http_controller,
        done_guard.release(),
        request.release(),
        response.release(),
        /*use_arena=*/false,
        /*is_http_request=*/true);

    const std::string mode = http_controller->request_attachment().to_string();
    const Status rejected =
        Status::invalid_parameter("invalid value", "temperature");
    if (mode == "before") {
      call.finish_with_error(rejected);
      return;
    }
    if (mode == "concurrent") {
      std::atomic<bool> polling{true};
      bool disconnected = false;
      std::promise<void> checked;
      auto checked_future = checked.get_future();
      std::thread checker([&] {
        disconnected = call.is_disconnected();
        checked.set_value();
        while (polling.load(std::memory_order_acquire)) {
          disconnected = call.is_disconnected() || disconnected;
        }
      });
      EXPECT_EQ(checked_future.wait_for(std::chrono::seconds(5)),
                std::future_status::ready);
      proto::CompletionResponse chunk;
      chunk.set_id("concurrent-chunk");
      for (int32_t index = 0; index < 128; ++index) {
        EXPECT_TRUE(call.write(chunk));
      }
      EXPECT_TRUE(call.finish());
      polling.store(false, std::memory_order_release);
      checker.join();
      EXPECT_FALSE(disconnected);
      return;
    }
    proto::CompletionResponse chunk;
    chunk.set_id("test-chunk");
    if (call.write(chunk)) {
      if (mode == "after") {
        call.finish_with_error(rejected);
        EXPECT_FALSE(call.write(chunk));
        EXPECT_FALSE(call.finish_with_error(rejected));
        call.finish();
      } else {
        call.finish();
        call.finish();
        EXPECT_FALSE(call.write(chunk));
        EXPECT_FALSE(call.finish_with_error(rejected));
      }
    }
  }
};

class StreamHttpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_,
                                 brpc::SERVER_DOESNT_OWN_SERVICE,
                                 "v1/completions => CompletionsHttp"),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
    brpc::ChannelOptions options;
    options.protocol = brpc::PROTOCOL_HTTP;
    options.timeout_ms = 5000;
    options.max_retry = 0;
    const std::string address =
        "127.0.0.1:" + std::to_string(server_.listen_address().port);
    ASSERT_EQ(channel_.Init(address.c_str(), &options), 0);
  }

  void TearDown() override {
    server_.Stop(0);
    server_.Join();
  }

  void send(brpc::Controller* controller,
            const std::string& path = "/v1/completions") {
    controller->http_request().uri() = path;
    controller->http_request().set_method(brpc::HTTP_METHOD_POST);
    channel_.CallMethod(nullptr, controller, nullptr, nullptr, nullptr);
  }

 private:
  StreamTestService service_;
  brpc::Server server_;
  brpc::Channel channel_;
};

TEST_F(StreamHttpTest, SuccessfulStreamStillEndsWithDone) {
  const double successes = COUNTER_VALUE(server_request_total_ok);
  const double failures = COUNTER_VALUE(server_request_total_fail);
  const double terminal_successes =
      MULTI_COUNTER_VALUE(api_request_terminal_total, "success");
  brpc::Controller controller;
  send(&controller);

  EXPECT_EQ(controller.http_response().status_code(), 200);
  EXPECT_EQ(controller.http_response().content_type(),
            "text/event-stream; charset=utf-8");
  const std::string body = controller.response_attachment().to_string();
  EXPECT_NE(body.find("\"test-chunk\""), std::string::npos);
  EXPECT_NE(body.find("data: [DONE]\n\n"), std::string::npos);
  EXPECT_EQ(body.find("data: [DONE]\n\n"), body.rfind("data: [DONE]\n\n"));
  EXPECT_EQ(body.find("rate_limit_exceeded"), std::string::npos);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), successes + 1);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), failures);
  EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, "success"),
            terminal_successes + 1);
}

TEST_F(StreamHttpTest, DisconnectPollingCanOverlapStreamWrites) {
  brpc::Controller controller;
  controller.request_attachment().append("concurrent");
  send(&controller);

  EXPECT_FALSE(controller.Failed());
  EXPECT_EQ(controller.http_response().status_code(), 200);
  EXPECT_EQ(controller.http_response().content_type(),
            "text/event-stream; charset=utf-8");
  const std::string body = controller.response_attachment().to_string();
  EXPECT_NE(body.find("\"concurrent-chunk\""), std::string::npos);
  EXPECT_NE(body.find("data: [DONE]\n\n"), std::string::npos);
}

TEST_F(StreamHttpTest, ErrorPreservesResponseCommitState) {
  for (const std::string& mode : {"before", "after"}) {
    const double successes = COUNTER_VALUE(server_request_total_ok);
    const double failures = COUNTER_VALUE(server_request_total_fail);
    const double invalid_requests =
        MULTI_COUNTER_VALUE(api_request_terminal_total, "invalid_request");
    brpc::Controller controller;
    controller.request_attachment().append(mode);
    send(&controller);
    const bool started = mode == "after";
    EXPECT_EQ(controller.http_response().status_code(), started ? 200 : 400);
    const std::string body = controller.response_attachment().to_string();
    EXPECT_NE(body.find("invalid_parameter"), std::string::npos);
    EXPECT_EQ(body.find("invalid_parameter"), body.rfind("invalid_parameter"));
    EXPECT_NE(body.find("temperature"), std::string::npos);
    EXPECT_EQ(body.find("data: [DONE]") != std::string::npos, started);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), successes);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), failures + 1);
    EXPECT_EQ(
        MULTI_COUNTER_VALUE(api_request_terminal_total, "invalid_request"),
        invalid_requests + 1);
  }
}

TEST(RequestMetricsTest, HandoffDoesNotCountAsCompletion) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  proto::CompletionResponse response;
  NoopClosure done;
  const double successes = COUNTER_VALUE(server_request_total_ok);
  const double failures = COUNTER_VALUE(server_request_total_fail);
  const double unavailable =
      MULTI_COUNTER_VALUE(api_request_terminal_total, "unavailable");
  {
    ClosureGuard guard(
        &done,
        [](void* context) { request_in_metric(context); },
        [&controller](void*) { request_out_metric(&controller); });
    CompletionCallForTest call(&controller,
                               guard.release(),
                               &request,
                               &response,
                               /*use_arena=*/true,
                               /*is_http_request=*/true);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), successes);
    call.finish_with_error(StatusCode::UNAVAILABLE, "backend unavailable");
  }
  EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), successes);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), failures + 1);
  EXPECT_EQ(MULTI_COUNTER_VALUE(api_request_terminal_total, "unavailable"),
            unavailable + 1);
}

TEST(RequestMetricsTest, EarlyHttpErrorIsNotSuccessful) {
  brpc::Controller controller;
  NoopClosure done;
  const double successes = COUNTER_VALUE(server_request_total_ok);
  const double failures = COUNTER_VALUE(server_request_total_fail);
  {
    ClosureGuard guard(
        &done,
        [](void* context) { request_in_metric(context); },
        [&controller](void*) { request_out_metric(&controller); });
    api_service::write_bad_request(&controller, "invalid request");
  }
  EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), successes);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), failures + 1);
}

TEST(RequestMetricsTest, UnfinishedResponseIsNotSuccessful) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  proto::CompletionResponse response;
  NoopClosure done;
  const double successes = COUNTER_VALUE(server_request_total_ok);
  const double failures = COUNTER_VALUE(server_request_total_fail);
  {
    CompletionCallForTest call(&controller,
                               &done,
                               &request,
                               &response,
                               /*use_arena=*/true,
                               /*is_http_request=*/true);
  }
  EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), successes);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), failures + 1);
}

}  // namespace
}  // namespace xllm
