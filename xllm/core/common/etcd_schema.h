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

#pragma once

#include <string>
#include <string_view>

namespace xllm::etcd_schema {

// These literals and the derivation contract mirror xllm-service. The default
// remains wire compatible with existing deployments; named shards elect
// independent Coordinator leaders inside the same etcd namespace.
inline constexpr char kDefaultControlPlaneShardId[] = "default";
inline constexpr char kDefaultMasterServiceKey[] = "XLLM:SERVICE:MASTER";
inline constexpr char kDefaultMasterServiceName[] = "MASTER";
inline constexpr char kDefaultRouterMemberPrefix[] = "XLLM:SERVICE:";
inline constexpr char kControlPlaneShardPrefix[] = "XLLM:CONTROL:SHARD:";

inline bool valid_control_plane_shard_id(std::string_view shard_id) {
  if (shard_id.empty() || shard_id.size() > 64) {
    return false;
  }
  for (const unsigned char value : shard_id) {
    const bool ascii_alphanumeric = (value >= 'a' && value <= 'z') ||
                                    (value >= 'A' && value <= 'Z') ||
                                    (value >= '0' && value <= '9');
    if (!ascii_alphanumeric && value != '-' && value != '_' && value != '.') {
      return false;
    }
  }
  return true;
}

inline std::string master_service_key_for_shard(std::string_view shard_id) {
  if (shard_id == kDefaultControlPlaneShardId) {
    return kDefaultMasterServiceKey;
  }
  return std::string(kControlPlaneShardPrefix) + std::string(shard_id) +
         ":MASTER";
}

inline std::string router_member_prefix_for_shard(std::string_view shard_id) {
  if (shard_id == kDefaultControlPlaneShardId) {
    return kDefaultRouterMemberPrefix;
  }
  return std::string(kControlPlaneShardPrefix) + std::string(shard_id) +
         ":ROUTER:";
}

}  // namespace xllm::etcd_schema
