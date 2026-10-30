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

#include "core/distributed_runtime/worker_rpc_result.h"

#include <brpc/channel.h>
#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <bthread/bthread.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>

namespace xllm {
namespace {
class SlowWorker final : public proto::DistributeWorker {
 public:
  void ExecuteModel(google::protobuf::RpcController*,
                    const proto::ForwardInput*,
                    proto::ForwardOutput*,
                    google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    ++executions;
    bthread_usleep(200000);
  }
  void GetLastStepResult(google::protobuf::RpcController*,
                         const proto::Empty*,
                         proto::ForwardOutput*,
                         google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    ++collections;
    bthread_usleep(200000);
  }
  std::atomic<int32_t> executions{0};
  std::atomic<int32_t> collections{0};
};

TEST(WorkerRpcResultTest,
     RealTransportTimesOutWithoutReplayingModelOperations) {
  SlowWorker worker;
  brpc::Server server;
  ASSERT_EQ(server.AddService(&worker, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  brpc::ServerOptions server_options;
  ASSERT_EQ(server.Start("127.0.0.1:0", &server_options), 0);
  brpc::Channel channel;
  brpc::ChannelOptions channel_options;
  channel_options.timeout_ms = -1;
  channel_options.max_retry = 3;
  ASSERT_EQ(channel.Init(server.listen_address(), &channel_options), 0);
  proto::DistributeWorker_Stub stub(&channel);
  for (const char* method : {"ExecuteModel", "GetLastStepResult"}) {
    brpc::Controller controller;
    prepare_worker_model_rpc(controller, 50);
    proto::ForwardOutput output;
    const auto started = std::chrono::steady_clock::now();
    if (std::string(method) == "ExecuteModel") {
      proto::ForwardInput input;
      stub.ExecuteModel(&controller, &input, &output, nullptr);
    } else {
      proto::Empty input;
      stub.GetLastStepResult(&controller, &input, &output, nullptr);
    }
    EXPECT_TRUE(controller.Failed());
    EXPECT_LT(std::chrono::steady_clock::now() - started,
              std::chrono::seconds(1));
    folly::Promise<std::optional<RawForwardOutput>> promise;
    auto future = promise.getSemiFuture();
    EXPECT_FALSE(
        complete_worker_rpc_result(method, controller, output, promise));
    EXPECT_FALSE(std::move(future).get().has_value());
  }
  server.Stop(0);
  server.Join();
  EXPECT_EQ(worker.executions, 1);
  EXPECT_EQ(worker.collections, 1);
}

TEST(WorkerRpcResultTest, FailureCompletesPromiseWithoutParsingPartialOutput) {
  for (const char* method : {"ExecuteModel", "GetLastStepResult"}) {
    brpc::Controller controller;
    controller.SetFailed(ETIMEDOUT, "injected timeout after model submission");
    proto::ForwardOutput output;
    output.add_outputs()->add_tokens()->set_id(123);
    folly::Promise<std::optional<RawForwardOutput>> promise;
    auto future = promise.getSemiFuture();
    EXPECT_FALSE(
        complete_worker_rpc_result(method, controller, output, promise));
    ASSERT_TRUE(future.isReady());
    EXPECT_FALSE(std::move(future).get().has_value());
  }
}

TEST(WorkerRpcResultTest, SuccessPreservesModelOutput) {
  brpc::Controller controller;
  proto::ForwardOutput output;
  output.add_outputs()->add_tokens()->set_id(42);
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  ASSERT_TRUE(
      complete_worker_rpc_result("ExecuteModel", controller, output, promise));
  const auto result = std::move(future).get();
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->outputs.size(), 1);
  ASSERT_EQ(result->outputs[0].tokens.size(), 1);
  EXPECT_EQ(result->outputs[0].tokens[0].id, 42);
}
}  // namespace
}  // namespace xllm
