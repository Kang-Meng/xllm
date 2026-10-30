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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "provider.pb.h"

namespace xllm {

inline constexpr uint32_t kNativeProviderContractVersion = 1;

// Immutable, observed inputs used to advertise one Native Engine profile.
// Callers must derive these fields from the loaded model and normalized runtime
// configuration; the builder deliberately has no environment fallbacks.
struct NativeProviderDescriptorConfig {
  std::string engine_uid;
  std::string api_address;
  proto::EngineRole role = proto::ENGINE_ROLE_UNSPECIFIED;
  std::string runtime_version;
  std::string hardware_runtime_version;
  std::string model_revision;
  std::string tokenizer_revision;
  std::string chat_template_contract;
  std::string renderer_contract;
  std::string quantization;
  std::string soc;
  uint32_t device_count = 0;
  uint32_t tp = 0;
  uint32_t dp = 0;
  uint32_t pp = 0;
  uint32_t ep = 0;
  uint32_t cp = 0;
  // Physical Ascend super-pod id. Empty is a wildcard: the engine opts out of
  // cross super-node isolation and stays compatible with every super-node.
  std::string super_node_id;
  std::string kv_layout_contract;
  // Model/cache semantics before local block granularity or rank sharding.
  // Empty preserves descriptors produced by older callers during migration.
  std::string kv_transfer_layout_contract;
  std::string kv_cache_dtype;
  uint32_t block_size = 0;
  uint32_t hash_version = 1;
  uint64_t hash_seed = 1024;
  // Zero omits the optional field. A nonzero value must be derived from the
  // allocated runtime layout, never from an environment default.
  uint64_t logical_kv_bytes_per_token = 0;
  std::string cache_isolation_domain = "single-tenant-default";
  std::vector<std::string> cache_groups;
  std::string head_shard_mapping_contract;
  std::string connector;
  std::string connector_version;
  std::string scheduler_class;
  uint32_t max_num_seqs = 0;
  uint64_t max_num_batched_tokens = 0;
  std::string scheduler_policy_contract;
  bool supports_local_prefill_decode = false;
  bool supports_prefill_only = false;
  std::vector<std::string> api_features;
};

struct NativeEngineStateSnapshot {
  // num_available_blocks(): physically free blocks plus evictable prefix
  // cache residents. Service uses this as the dispatch free count.
  std::vector<uint64_t> kv_free_blocks;
  std::vector<uint64_t> kv_used_blocks;
  std::vector<uint64_t> kv_total_blocks;
  // Heartbeats publish READY. Service records DRAINING in etcd.
  proto::EngineLifecycle lifecycle = proto::ENGINE_LIFECYCLE_READY;
};

// Builds a fail-closed Native V2 descriptor. Optional local modes are only
// published when the observed runtime can enforce their admission contract.
std::optional<proto::ProviderDescriptor> build_native_provider_descriptor(
    const NativeProviderDescriptorConfig& config,
    std::string* error);

// Publish the API surface this Engine actually opens, mirroring the
// ServiceImplFactory registry per serving mode. The returned vector is empty
// when the backend cannot be resolved to a known EngineType; callers must
// fail closed in that case instead of publishing an empty api_features list.
std::vector<std::string> api_features_for_backend(const std::string& backend);

// Builds one EngineState from an immutable descriptor and a capacity snapshot.
// Service records receive order; this builder does not assign a sequence.
class NativeEngineStateBuilder final {
 public:
  explicit NativeEngineStateBuilder(proto::ProviderDescriptor descriptor);

  std::optional<proto::EngineState> build(
      const NativeEngineStateSnapshot& snapshot,
      std::string* error);

 private:
  proto::ProviderDescriptor descriptor_;
  bool descriptor_valid_ = false;
};

}  // namespace xllm
