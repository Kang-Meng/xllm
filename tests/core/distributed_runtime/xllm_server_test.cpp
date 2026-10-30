/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "server/xllm_server.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include "core/distributed_runtime/collective_service.h"
#include "core/distributed_runtime/disagg_pd_service.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/service_config.h"

namespace xllm {
namespace {

class ScopedServerConfig final {
 public:
  ScopedServerConfig()
      : old_host_(ServiceConfig::get_instance().host()),
        old_port_(DisaggPDConfig::get_instance().disagg_pd_port()) {
    // Empty host uses IP_ANY and port zero asks brpc for an unused local port.
    ServiceConfig::get_instance().host("");
    DisaggPDConfig::get_instance().disagg_pd_port(0);
  }

  ~ScopedServerConfig() {
    ServiceConfig::get_instance().host(old_host_);
    DisaggPDConfig::get_instance().disagg_pd_port(old_port_);
  }

 private:
  std::string old_host_;
  int32_t old_port_;
};

TEST(XllmServerTest, RequestStopWakesThreadBlockedInServerJoin) {
  ScopedServerConfig config;
  XllmServer server;
  std::atomic<bool> start_returned{false};
  bool start_succeeded = false;
  std::thread join_thread([&] {
    start_succeeded = server.start(std::make_unique<DisaggPDService>());
    start_returned.store(true, std::memory_order_release);
  });

  const auto initialization_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!server.has_initialized() &&
         !start_returned.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < initialization_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  const bool initialized = server.has_initialized();
  const auto stop_started = std::chrono::steady_clock::now();
  server.request_stop();
  join_thread.join();

  EXPECT_TRUE(initialized);
  EXPECT_TRUE(start_succeeded);
  EXPECT_LT(std::chrono::steady_clock::now() - stop_started,
            std::chrono::seconds(2));
}

TEST(XllmServerTest, RequestStopCancelsCollectiveRendezvousWaiter) {
  XllmServer server;
  auto service = std::make_shared<CollectiveService>(/*total_num=*/2);
  ASSERT_TRUE(server.start(service, "127.0.0.1:0", "CollectiveTest"));
  auto waiting = std::async(std::launch::async, [service]() {
    return service->wait(std::chrono::seconds(10));
  });

  server.request_stop();

  EXPECT_EQ(waiting.wait_for(std::chrono::milliseconds(200)),
            std::future_status::ready);
  EXPECT_FALSE(waiting.get().has_value());
  server.stop();
}

TEST(XllmServerTest, StopRequestedBeforeCollectiveStartFencesStartup) {
  XllmServer server;
  server.request_stop();
  auto service = std::make_shared<CollectiveService>(/*total_num=*/2);

  EXPECT_FALSE(server.start(service, "127.0.0.1:0", "CollectiveTest"));
  EXPECT_FALSE(service->wait(std::chrono::milliseconds(20)).has_value());
}

}  // namespace
}  // namespace xllm
