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

#include "core/distributed_runtime/collective_service.h"

#include <brpc/controller.h>
#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <string>
#include <thread>
#include <vector>

namespace xllm {
namespace {

void register_rank(CollectiveService* service,
                   int32_t rank,
                   const std::string& address,
                   proto::CommUniqueIdList* response) {
  proto::AddressInfo request;
  request.set_global_rank(rank);
  request.set_address(address);
  service->Sync(nullptr, &request, response, nullptr);
}

TEST(CollectiveServiceTest, RegistersAddressesWithoutCommunicationIds) {
  CollectiveService service(/*total_num=*/2);
  proto::CommUniqueIdList first_response;
  proto::CommUniqueIdList second_response;
  first_response.add_comm_unique_ids()->set_comm_unique_id("stale-id");
  register_rank(&service, 0, "127.0.0.1:10000", &first_response);
  register_rank(&service, 1, "127.0.0.1:10001", &second_response);

  const auto addresses = service.wait(std::chrono::milliseconds(20));
  ASSERT_TRUE(addresses.has_value());
  EXPECT_EQ(addresses->at(0), "127.0.0.1:10000");
  EXPECT_EQ(addresses->at(1), "127.0.0.1:10001");
  EXPECT_TRUE(first_response.comm_unique_ids().empty());
  EXPECT_TRUE(second_response.comm_unique_ids().empty());
}

TEST(CollectiveServiceTest, ConcurrentDuplicateRegistrationsConverge) {
  constexpr int32_t kRanks = 128;
  CollectiveService service(kRanks);
  std::vector<std::thread> threads;
  threads.reserve(kRanks);
  for (int32_t rank = 0; rank < kRanks; ++rank) {
    threads.emplace_back([&service, rank]() {
      proto::CommUniqueIdList response;
      const std::string address = "10.0.0." + std::to_string(rank + 1) + ":" +
                                  std::to_string(10000 + rank);
      register_rank(&service, rank, address, &response);
      register_rank(&service, rank, address, &response);
      EXPECT_TRUE(response.comm_unique_ids().empty());
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  const auto addresses = service.wait(std::chrono::milliseconds(200));
  ASSERT_TRUE(addresses.has_value());
  EXPECT_EQ(addresses->size(), static_cast<size_t>(kRanks));
  EXPECT_EQ(addresses->at(0), "10.0.0.1:10000");
  EXPECT_EQ(addresses->at(kRanks - 1), "10.0.0.128:10127");
}

TEST(CollectiveServiceTest, MissingRankTimesOutWithinBound) {
  CollectiveService service(/*total_num=*/2);
  proto::CommUniqueIdList response;
  register_rank(&service, 0, "127.0.0.1:10000", &response);

  const auto started = std::chrono::steady_clock::now();
  EXPECT_FALSE(service.wait(std::chrono::milliseconds(20)).has_value());
  EXPECT_LT(std::chrono::steady_clock::now() - started,
            std::chrono::milliseconds(200));
}

TEST(CollectiveServiceTest, FencesInvalidAndConflictingRegistrations) {
  CollectiveService service(/*total_num=*/2);
  proto::AddressInfo request;
  proto::CommUniqueIdList response;
  brpc::Controller controller;
  request.set_global_rank(2);
  request.set_address("127.0.0.1:10002");
  service.Sync(&controller, &request, &response, nullptr);
  EXPECT_TRUE(controller.Failed());

  controller.Reset();
  request.set_global_rank(0);
  request.set_address("127.0.0.1:10000");
  service.Sync(&controller, &request, &response, nullptr);
  EXPECT_FALSE(controller.Failed());
  controller.Reset();
  request.set_address("127.0.0.1:20000");
  service.Sync(&controller, &request, &response, nullptr);
  EXPECT_TRUE(controller.Failed());

  register_rank(&service, 1, "127.0.0.1:10001", &response);
  const auto addresses = service.wait(std::chrono::milliseconds(20));
  ASSERT_TRUE(addresses.has_value());
  EXPECT_EQ(addresses->at(0), "127.0.0.1:10000");
}

TEST(CollectiveServiceTest, CancellationWakesWaiter) {
  CollectiveService service(/*total_num=*/2);
  auto waiting = std::async(std::launch::async, [&service]() {
    return service.wait(std::chrono::seconds(10));
  });
  service.cancel();

  EXPECT_EQ(waiting.wait_for(std::chrono::milliseconds(200)),
            std::future_status::ready);
  EXPECT_FALSE(waiting.get().has_value());

  proto::AddressInfo request;
  request.set_global_rank(0);
  request.set_address("127.0.0.1:10000");
  proto::CommUniqueIdList response;
  brpc::Controller controller;
  service.Sync(&controller, &request, &response, nullptr);
  EXPECT_TRUE(controller.Failed());
}

}  // namespace
}  // namespace xllm
