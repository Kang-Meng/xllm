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

#include <brpc/details/controller_private_accessor.h>
#include <brpc/policy/baidu_rpc_protocol.h>
#include <brpc/span.h>
#include <butil/time.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>

#include "core/common/health_check_manager.h"
#include "core/distributed_runtime/comm_channel.h"
#include "core/distributed_runtime/shm_channel.h"

namespace xllm {
namespace {

TEST(CommChannelFailureTest,
     RpcFailureResolvesWithFailureInsteadOfEmptyOutput) {
  auto* closure = new ExecuteModelClosure();
  auto future = closure->promise.getSemiFuture();
  closure->cntl.SetFailed("injected worker disconnect");
  closure->Run();
  EXPECT_FALSE(std::move(future).get().has_value());
  EXPECT_FALSE(HealthCheckManager::instance().is_healthy());
  HealthCheckManager::instance().set_healthy();
}

TEST(CommChannelFailureTest, RpcCapacityIsCapturedAtChannelInitialization) {
  gflags::FlagSaver restore_flags;
  ASSERT_FALSE(gflags::SetCommandLineOption("max_body_size", "4096").empty());
  CommChannel channel;
  ASSERT_TRUE(channel.init_brpc("127.0.0.1:1"));
  const uint64_t initial_limit = channel.forward_input_limit();
  EXPECT_GT(initial_limit, 0);
  EXPECT_LT(initial_limit, 4096);
  ASSERT_FALSE(gflags::SetCommandLineOption("max_body_size", "8192").empty());
  EXPECT_EQ(channel.forward_input_limit(), initial_limit);
}

TEST(CommChannelFailureTest, RpcCapacityCoversTracedWireRequests) {
  gflags::FlagSaver restore_flags;
  const std::array<uint64_t, 4> body_limits = {256, 16384, 65536, 1048576};
  for (uint64_t body_limit : body_limits) {
    SCOPED_TRACE(body_limit);
    ASSERT_FALSE(gflags::SetCommandLineOption(
                     "max_body_size", std::to_string(body_limit).c_str())
                     .empty());
    CommChannel channel;
    ASSERT_TRUE(channel.init_brpc("127.0.0.1:1"));
    const uint64_t shm_overhead = sizeof(ControlMetadata) + sizeof(uint64_t);
    ASSERT_GT(channel.forward_input_limit(), shm_overhead);

    // Fill the advertised raw payload budget, then let BRPC produce the real
    // envelope with maximum-width trace and correlation IDs.
    proto::ForwardInput input;
    input.mutable_packed_input()->mutable_payload()->resize(
        channel.forward_input_limit() - shm_overhead);
    butil::IOBuf request_body;
    request_body.append(input.SerializeAsString());
    brpc::Controller controller;
    const uint64_t max_id = std::numeric_limits<uint64_t>::max();
    auto* span = brpc::Span::CreateServerSpan("ExecuteModel",
                                              /*trace_id=*/max_id,
                                              /*span_id=*/max_id,
                                              /*parent_span_id=*/max_id,
                                              butil::gettimeofday_us());
    ASSERT_NE(span, nullptr);
    brpc::ControllerPrivateAccessor(&controller).set_span(span);
    butil::IOBuf wire_request;
    const auto* method =
        proto::DistributeWorker::descriptor()->FindMethodByName("ExecuteModel");
    brpc::policy::PackRpcRequest(&wire_request,
                                 /*socket_message=*/nullptr,
                                 /*correlation_id=*/max_id,
                                 method,
                                 &controller,
                                 request_body,
                                 /*auth=*/nullptr);
    ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
    // PRPC's fixed header is outside the body-size limit.
    constexpr uint64_t kPrpcHeaderSize = 12;
    ASSERT_GE(wire_request.size(), kPrpcHeaderSize);
    EXPECT_LE(wire_request.size() - kPrpcHeaderSize, body_limit);
  }
}

TEST(CommChannelFailureTest, RpcCapacityIsZeroWhenEnvelopeCannotFit) {
  gflags::FlagSaver restore_flags;
  ASSERT_FALSE(gflags::SetCommandLineOption("max_body_size", "1").empty());
  CommChannel channel;
  ASSERT_TRUE(channel.init_brpc("127.0.0.1:1"));
  EXPECT_EQ(channel.forward_input_limit(), 0);
}

TEST(CommChannelFailureTest, ShmWriteFailureCancelsGroupWithoutRpcFallback) {
  runtime::Options options;
  options.enable_shm(true).input_shm_size(4096).output_shm_size(4096);
  options.master_node_addr("127.0.0.1:" + std::to_string(getpid()));
  ShmChannel channel(/*dp_group=*/0, /*rank=*/0, /*is_driver=*/true, options);
  const std::string name = ForwardSharedMemoryManager::create_unique_name(
      "xllm_" + std::to_string(getpid()),
      /*dp_group=*/0,
      ForwardType::RAW_INPUT,
      /*rank=*/0);
  bool creator = false;
  ForwardSharedMemoryManager peer(name, 4096, creator, ForwardType::RAW_INPUT);
  ForwardInput input;
  input.token_ids = torch::zeros({8192}, torch::kInt32);
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  // No RPC stub is initialized: this must finish locally, not try a fallback.
  channel.execute_model_async(input, promise);
  EXPECT_FALSE(std::move(future).get().has_value());
  EXPECT_TRUE(peer.cancelled());
  EXPECT_FALSE(HealthCheckManager::instance().is_healthy());
  HealthCheckManager::instance().set_healthy();
}

}  // namespace
}  // namespace xllm
