/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/common/etcd_schema.h"

#include <gtest/gtest.h>

#include <string>

namespace xllm::etcd_schema {
namespace {

TEST(EtcdSchemaTest, ControlPlaneShardKeyContractIsBoundedAndCompatible) {
  EXPECT_TRUE(valid_control_plane_shard_id("default"));
  EXPECT_TRUE(valid_control_plane_shard_id("model-pool_17.prod"));
  EXPECT_FALSE(valid_control_plane_shard_id(""));
  EXPECT_FALSE(valid_control_plane_shard_id("pool/escape"));
  EXPECT_FALSE(valid_control_plane_shard_id(std::string(65, 'a')));
  EXPECT_FALSE(valid_control_plane_shard_id(std::string("bad\xFF", 4)));

  EXPECT_EQ(master_service_key_for_shard(kDefaultControlPlaneShardId),
            kDefaultMasterServiceKey);
  EXPECT_EQ(router_member_prefix_for_shard(kDefaultControlPlaneShardId),
            kDefaultRouterMemberPrefix);
  EXPECT_EQ(master_service_key_for_shard("pool-a"),
            "XLLM:CONTROL:SHARD:pool-a:MASTER");
  EXPECT_EQ(router_member_prefix_for_shard("pool-a"),
            "XLLM:CONTROL:SHARD:pool-a:ROUTER:");
  EXPECT_NE(master_service_key_for_shard("pool-a"),
            master_service_key_for_shard("pool-b"));
}

}  // namespace
}  // namespace xllm::etcd_schema
