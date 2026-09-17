/* Copyright 2026 The xLLM Authors.

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

#include <gtest/gtest.h>

#include "core/framework/parallel_state/collective_communicator.h"

namespace xllm {
namespace {

TEST(CollectiveCommunicatorPolicyTest, ReusesOnlyEquivalentTpGroup) {
  EXPECT_TRUE(can_reuse_tp_group_for_moe(
      /*dp_size=*/1, /*tp_size=*/4, /*moe_tp_size=*/4));
  EXPECT_FALSE(can_reuse_tp_group_for_moe(
      /*dp_size=*/1, /*tp_size=*/2, /*moe_tp_size=*/4));
  EXPECT_FALSE(can_reuse_tp_group_for_moe(
      /*dp_size=*/2, /*tp_size=*/2, /*moe_tp_size=*/4));
}

TEST(CollectiveCommunicatorPolicyTest, MoETpGroupIdentityFollowsReusePolicy) {
  // Bare ProcessGroup instances carry no backend communicator: construction
  // and destruction are safe while the pg_ backend handle stays null.
  ProcessGroup tp_group(
      /*rank=*/0, /*world_size=*/4, torch::Device(torch::kCPU));
  ProcessGroup world_group(
      /*rank=*/0, /*world_size=*/4, torch::Device(torch::kCPU));
  ASSERT_NE(&tp_group, &world_group);

  // DP1/EP1 with an equivalent TP rank set: dense and MoE collectives share
  // the very same TP communicator object.
  EXPECT_EQ(select_moe_tp_group(&tp_group,
                                &world_group,
                                /*dp_size=*/1,
                                /*tp_size=*/4,
                                /*moe_tp_size=*/4),
            &tp_group);

  // DP>1 or a non-equivalent MoE TP size keeps the world process group.
  EXPECT_EQ(select_moe_tp_group(&tp_group,
                                &world_group,
                                /*dp_size=*/2,
                                /*tp_size=*/2,
                                /*moe_tp_size=*/4),
            &world_group);
  EXPECT_EQ(select_moe_tp_group(&tp_group,
                                &world_group,
                                /*dp_size=*/1,
                                /*tp_size=*/2,
                                /*moe_tp_size=*/4),
            &world_group);
}

}  // namespace
}  // namespace xllm
