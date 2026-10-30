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

#include <gtest/gtest.h>

#include <string>

#include "core/common/types.h"

namespace xllm {
namespace {

// Data flow B: the top-level InstanceInfo carries super_node_id into the etcd
// registration JSON, mirroring descriptor.topology.super_node_id from the same
// DisaggPDConfig source. A non-empty id is emitted; an empty id (wildcard, the
// engine opts out of cross super-node isolation) is omitted so older readers
// and opted-out instances see no key at all.
InstanceInfo make_prefill_info() {
  InstanceInfo info;
  info.name = "prefill-0";
  info.rpc_address = "127.0.0.1:19000";
  info.type = "PREFILL";
  return info;
}

TEST(InstanceInfoSerializeTest, EmitsSuperNodeIdWhenSet) {
  InstanceInfo info = make_prefill_info();
  info.super_node_id = "pod-A";

  const nlohmann::json registration = info.serialize_to_json();

  ASSERT_TRUE(registration.contains("super_node_id"));
  EXPECT_EQ(registration.at("super_node_id").get<std::string>(), "pod-A");
}

TEST(InstanceInfoSerializeTest, OmitsSuperNodeIdWhenWildcard) {
  InstanceInfo info = make_prefill_info();
  // Default-constructed super_node_id is empty (wildcard).

  const nlohmann::json registration = info.serialize_to_json();

  EXPECT_FALSE(registration.contains("super_node_id"));
}

}  // namespace
}  // namespace xllm
