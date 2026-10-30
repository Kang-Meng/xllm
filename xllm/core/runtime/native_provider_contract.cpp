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

#include "core/runtime/native_provider_contract.h"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "core/common/types.h"

namespace xllm {
namespace {

constexpr std::string_view kDescriptorDigestDomain =
    "xllm-native-provider-descriptor-v1";
constexpr std::string_view kChatTemplateDigestDomain =
    "xllm-native-chat-template-v1";
constexpr std::string_view kRendererDigestDomain = "xllm-native-renderer-v1";
constexpr std::string_view kKVLayoutDigestDomain = "xllm-native-kv-layout-v1";
constexpr std::string_view kKVNamespaceDigestDomain =
    "xllm-canonical-kv-namespace-v1";
constexpr std::string_view kHeadShardDigestDomain = "xllm-native-head-shard-v1";
constexpr std::string_view kSchedulerDigestDomain = "xllm-native-scheduler-v1";

void set_error(std::string message, std::string* error) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::optional<std::string> sha256_hex(std::string_view domain,
                                      std::string_view value) {
  EVP_MD_CTX* context = EVP_MD_CTX_new();
  if (context == nullptr) {
    return std::nullopt;
  }

  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  const bool ok =
      EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
      EVP_DigestUpdate(context, domain.data(), domain.size()) == 1 &&
      EVP_DigestUpdate(context, value.data(), value.size()) == 1 &&
      EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
  EVP_MD_CTX_free(context);
  if (!ok) {
    return std::nullopt;
  }

  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned int index = 0; index < digest_size; ++index) {
    output << std::setw(2) << static_cast<uint32_t>(digest[index]);
  }
  return output.str();
}

bool has_empty_required_string(const NativeProviderDescriptorConfig& config) {
  return config.engine_uid.empty() || config.api_address.empty() ||
         config.runtime_version.empty() ||
         config.hardware_runtime_version.empty() ||
         config.model_revision.empty() || config.tokenizer_revision.empty() ||
         config.renderer_contract.empty() || config.quantization.empty() ||
         config.soc.empty() || config.kv_layout_contract.empty() ||
         config.kv_cache_dtype.empty() ||
         config.head_shard_mapping_contract.empty() ||
         config.connector.empty() || config.connector_version.empty() ||
         config.scheduler_class.empty() ||
         config.scheduler_policy_contract.empty();
}

bool has_unique_nonempty_strings(const std::vector<std::string>& values) {
  std::unordered_set<std::string> unique;
  unique.reserve(values.size());
  for (const std::string& value : values) {
    if (value.empty() || !unique.insert(value).second) {
      return false;
    }
  }
  return true;
}

bool has_capability(const proto::ProviderDescriptor& descriptor,
                    proto::ProviderCapability capability) {
  return std::find(descriptor.capabilities().begin(),
                   descriptor.capabilities().end(),
                   capability) != descriptor.capabilities().end();
}

std::optional<std::string> required_digest(std::string_view domain,
                                           std::string_view contract,
                                           const std::string& field,
                                           std::string* error) {
  std::optional<std::string> digest = sha256_hex(domain, contract);
  if (!digest.has_value()) {
    set_error("Failed to compute " + field + " digest", error);
  }
  return digest;
}

void append_namespace_value(std::string_view name,
                            std::string_view value,
                            std::string* contract) {
  contract->append(std::to_string(name.size()));
  contract->push_back(':');
  contract->append(name);
  contract->append(std::to_string(value.size()));
  contract->push_back(':');
  contract->append(value);
}

// Preserve the local hash identity byte-for-byte. Transfer identity uses the
// same isolation/model inputs but excludes the local logical block granularity.
std::string cache_contract(const NativeProviderDescriptorConfig& config,
                           const std::string& chat_template_digest,
                           const std::string& renderer_digest,
                           bool include_block_size) {
  std::string namespace_contract;
  append_namespace_value(
      "model_revision", config.model_revision, &namespace_contract);
  append_namespace_value(
      "tokenizer_revision", config.tokenizer_revision, &namespace_contract);
  append_namespace_value(
      "chat_template_digest", chat_template_digest, &namespace_contract);
  append_namespace_value(
      "renderer_digest", renderer_digest, &namespace_contract);
  if (include_block_size) {
    append_namespace_value(
        "block_size", std::to_string(config.block_size), &namespace_contract);
  }
  append_namespace_value(
      "hash_version", std::to_string(config.hash_version), &namespace_contract);
  append_namespace_value(
      "hash_seed", std::to_string(config.hash_seed), &namespace_contract);
  append_namespace_value(
      "quantization", config.quantization, &namespace_contract);
  append_namespace_value(
      "cache_dtype", config.kv_cache_dtype, &namespace_contract);
  append_namespace_value(
      "runtime_version", config.runtime_version, &namespace_contract);
  append_namespace_value("hardware_runtime_version",
                         config.hardware_runtime_version,
                         &namespace_contract);
  append_namespace_value("cache_isolation_domain",
                         config.cache_isolation_domain,
                         &namespace_contract);
  for (const std::string& cache_group : config.cache_groups) {
    append_namespace_value("cache_group", cache_group, &namespace_contract);
  }
  return namespace_contract;
}

}  // namespace

// Publish the API surface this Engine actually opens, mirroring the
// ServiceImplFactory registry per serving mode. The previous hardcoded
// {"chat_completions", "completions"} advertised text-only endpoints on VLM
// Engines, so any multimodal CanonicalRequest carrying
// required_api_features=["chat_completions"] failed the Service-side
// descriptor fail-closed check and the VLM Engine could never be selected
// for multimodal traffic. Speculative variants (VLMSSM via the "vlm"
// backend) keep the same API surface as their base mode, matching
// to_serving_mode() in api_service/serving_mode.h.
// VLM modes additionally publish "multimodal_chat_completions": plain
// "chat_completions" is also published by LLM and REC Engines, so it cannot
// alone route multimodal traffic. Requiring the extra feature word keeps
// multimodal requests on multimodal Engines.
// NOTE: text-only chat traffic does NOT carry required_api_features, so the
// subset check never excludes a VLM Engine for it - a sole VLM instance
// serving text requests is a supported topology (the brpc entry routes
// them through MMChatServiceImpl, same as the VLM HTTP handler). Do not
// read this feature list as a text-traffic exclusion contract.
std::vector<std::string> api_features_for_backend(const std::string& backend) {
  const EngineType engine_type(backend);
  switch (static_cast<EngineType::Value>(engine_type)) {
    case EngineType::VLM:
    case EngineType::VLMSSM:
      return {"chat_completions", "multimodal_chat_completions"};
    case EngineType::DIT:
      return {"image_generation",
              "audio_generation",
              "text_generation",
              "video_generation"};
    case EngineType::REC:
      return {"completions", "chat_completions"};
    case EngineType::LLM:
    case EngineType::SSM:
      return {"chat_completions", "completions"};
    default:
      return {};
  }
}

std::optional<proto::ProviderDescriptor> build_native_provider_descriptor(
    const NativeProviderDescriptorConfig& config,
    std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  if (has_empty_required_string(config)) {
    set_error("Native Provider Descriptor has an empty required field", error);
    return std::nullopt;
  }
  if (config.role != proto::ENGINE_ROLE_PREFILL &&
      config.role != proto::ENGINE_ROLE_DECODE &&
      config.role != proto::ENGINE_ROLE_AGGREGATED) {
    set_error("Native V2 role is unsupported", error);
    return std::nullopt;
  }
  if (config.device_count == 0 || config.tp == 0 || config.dp == 0 ||
      config.pp == 0 || config.ep == 0 || config.cp == 0 ||
      config.block_size == 0 || config.hash_version == 0 ||
      config.hash_seed == 0 || config.cache_isolation_domain.empty() ||
      config.max_num_seqs == 0 || config.max_num_batched_tokens == 0) {
    set_error("Native Provider Descriptor has an invalid zero capacity", error);
    return std::nullopt;
  }
  if (config.cache_groups.empty() || config.api_features.empty() ||
      !has_unique_nonempty_strings(config.cache_groups) ||
      !has_unique_nonempty_strings(config.api_features)) {
    set_error("Native Provider Descriptor has invalid repeated fields", error);
    return std::nullopt;
  }

  const std::optional<std::string> chat_template_digest =
      required_digest(kChatTemplateDigestDomain,
                      config.chat_template_contract,
                      "chat template",
                      error);
  const std::optional<std::string> renderer_digest = required_digest(
      kRendererDigestDomain, config.renderer_contract, "renderer", error);
  const std::optional<std::string> kv_layout_digest = required_digest(
      kKVLayoutDigestDomain, config.kv_layout_contract, "KV layout", error);
  const std::optional<std::string> head_shard_digest =
      required_digest(kHeadShardDigestDomain,
                      config.head_shard_mapping_contract,
                      "head shard mapping",
                      error);
  const std::optional<std::string> scheduler_digest =
      required_digest(kSchedulerDigestDomain,
                      config.scheduler_policy_contract,
                      "scheduler policy",
                      error);
  if (!chat_template_digest.has_value() || !renderer_digest.has_value() ||
      !kv_layout_digest.has_value() || !head_shard_digest.has_value() ||
      !scheduler_digest.has_value()) {
    return std::nullopt;
  }

  proto::ProviderDescriptor descriptor;
  descriptor.set_contract_version(kNativeProviderContractVersion);
  proto::ProviderIdentity* identity = descriptor.mutable_identity();
  identity->set_engine_uid(config.engine_uid);
  identity->set_provider_id(proto::PROVIDER_ID_XLLM_NATIVE);
  identity->set_runtime_family("xllm");
  identity->set_runtime_version(config.runtime_version);
  identity->set_plugin_version("builtin-" + config.runtime_version);
  identity->set_hardware_runtime_version(config.hardware_runtime_version);
  identity->set_protocol_version(kNativeProviderContractVersion);

  proto::ProviderEndpoint* endpoint = descriptor.mutable_endpoint();
  endpoint->set_control_transport("brpc");
  endpoint->set_data_transport(config.connector);
  endpoint->set_address(config.api_address);

  proto::ServingDescriptor* serving = descriptor.mutable_serving();
  serving->set_role(config.role);
  proto::ExecutionModeSpec* mode = serving->add_execution_modes();
  if (config.role == proto::ENGINE_ROLE_AGGREGATED) {
    mode->set_mode(proto::EXECUTION_MODE_AGGREGATED);
    mode->set_transfer_mode(proto::TRANSFER_MODE_NONE);
    mode->set_selection_order(proto::SELECTION_ORDER_SINGLE);
    mode->set_binding_stage(proto::BINDING_STAGE_AT_SUBMIT);
  } else {
    mode->set_mode(proto::EXECUTION_MODE_REMOTE_PD);
    mode->set_transfer_mode(proto::TRANSFER_MODE_LAYERWISE_PUSH);
    mode->set_selection_order(proto::SELECTION_ORDER_P_FIRST);
    mode->set_binding_stage(proto::BINDING_STAGE_BEFORE_PREFILL);
  }
  mode->set_p_selection_delegated(false);
  if (config.supports_local_prefill_decode) {
    if (config.role != proto::ENGINE_ROLE_DECODE) {
      set_error("Only a DECODE Engine can publish local Prefill/Decode", error);
      return std::nullopt;
    }
    proto::ExecutionModeSpec* local = serving->add_execution_modes();
    local->set_mode(proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE);
    local->set_transfer_mode(proto::TRANSFER_MODE_NONE);
    local->set_selection_order(proto::SELECTION_ORDER_D_ONLY);
    local->set_binding_stage(proto::BINDING_STAGE_AT_SUBMIT);
    local->set_p_selection_delegated(false);
  }
  if (config.supports_prefill_only) {
    if (config.role != proto::ENGINE_ROLE_PREFILL) {
      set_error("Only a PREFILL Engine can publish Prefill-only", error);
      return std::nullopt;
    }
    proto::ExecutionModeSpec* prefill_only = serving->add_execution_modes();
    prefill_only->set_mode(proto::EXECUTION_MODE_PREFILL_ONLY);
    prefill_only->set_transfer_mode(proto::TRANSFER_MODE_NONE);
    prefill_only->set_selection_order(proto::SELECTION_ORDER_P_ONLY);
    prefill_only->set_binding_stage(proto::BINDING_STAGE_AT_SUBMIT);
    prefill_only->set_p_selection_delegated(false);
  }
  for (const std::string& feature : config.api_features) {
    serving->add_api_features(feature);
  }

  proto::ModelDescriptor* model = descriptor.mutable_model();
  model->set_model_revision(config.model_revision);
  model->set_tokenizer_revision(config.tokenizer_revision);
  model->set_chat_template_digest(*chat_template_digest);
  model->set_quantization(config.quantization);
  model->set_renderer_digest(*renderer_digest);

  proto::TopologyDescriptor* topology = descriptor.mutable_topology();
  topology->set_soc(config.soc);
  topology->set_device_count(config.device_count);
  topology->set_tp(config.tp);
  topology->set_dp(config.dp);
  topology->set_pp(config.pp);
  topology->set_ep(config.ep);
  topology->set_cp(config.cp);
  topology->set_super_node_id(config.super_node_id);

  proto::KVDescriptor* kv = descriptor.mutable_kv();
  kv->set_kv_layout_digest(*kv_layout_digest);
  if (!config.kv_transfer_layout_contract.empty()) {
    std::string transfer_contract =
        cache_contract(config,
                       *chat_template_digest,
                       *renderer_digest,
                       /*include_block_size=*/false);
    append_namespace_value(
        "kv_semantics", config.kv_transfer_layout_contract, &transfer_contract);
    const std::optional<std::string> transfer_digest = required_digest(
        "xllm-native-kv-transfer-v1", transfer_contract, "KV transfer", error);
    if (!transfer_digest.has_value()) {
      return std::nullopt;
    }
    kv->set_transfer_compatibility_digest(*transfer_digest);
  }
  kv->set_cache_dtype(config.kv_cache_dtype);
  kv->set_block_size(config.block_size);
  kv->set_hash_version(config.hash_version);
  kv->set_hash_seed(config.hash_seed);
  if (config.logical_kv_bytes_per_token > 0) {
    kv->set_logical_bytes_per_token(config.logical_kv_bytes_per_token);
  }
  for (const std::string& cache_group : config.cache_groups) {
    kv->add_cache_groups(cache_group);
  }
  kv->set_head_shard_mapping_digest(*head_shard_digest);
  kv->set_connector(config.connector);
  kv->set_connector_version(config.connector_version);
  kv->add_transfer_modes(proto::TRANSFER_MODE_LAYERWISE_PUSH);
  if (config.role == proto::ENGINE_ROLE_AGGREGATED ||
      config.supports_local_prefill_decode || config.supports_prefill_only) {
    kv->add_transfer_modes(proto::TRANSFER_MODE_NONE);
  }

  proto::SchedulerDescriptor* scheduler = descriptor.mutable_scheduler();
  scheduler->set_scheduler_class(config.scheduler_class);
  scheduler->set_max_num_seqs(config.max_num_seqs);
  scheduler->set_max_num_batched_tokens(config.max_num_batched_tokens);
  scheduler->set_scheduler_policy_digest(*scheduler_digest);

  if (config.role == proto::ENGINE_ROLE_AGGREGATED) {
    descriptor.add_capabilities(proto::PROVIDER_CAPABILITY_AGGREGATED);
  } else {
    descriptor.add_capabilities(
        proto::PROVIDER_CAPABILITY_REMOTE_PD_LAYERWISE_PUSH);
  }
  descriptor.add_capabilities(proto::PROVIDER_CAPABILITY_NATIVE_RESERVATION);
  descriptor.add_capabilities(proto::PROVIDER_CAPABILITY_ATTEMPT_QUERY);
  descriptor.add_capabilities(proto::PROVIDER_CAPABILITY_CANCEL_FENCE);
  descriptor.add_capabilities(proto::PROVIDER_CAPABILITY_ENGINE_LOCAL_DEADLINE);
  descriptor.add_capabilities(proto::PROVIDER_CAPABILITY_PER_DP_STATE);
  descriptor.add_capabilities(proto::PROVIDER_CAPABILITY_SELF_FENCING);
  descriptor.add_capabilities(proto::PROVIDER_CAPABILITY_DRAIN);
  if (config.supports_local_prefill_decode) {
    descriptor.add_capabilities(
        proto::PROVIDER_CAPABILITY_LOCAL_PREFILL_DECODE);
    descriptor.add_capabilities(
        proto::PROVIDER_CAPABILITY_MIXED_PREFILL_DECODE_ACCOUNTING);
    descriptor.add_capabilities(
        proto::PROVIDER_CAPABILITY_STRUCTURED_ADMISSION);
  }
  if (config.supports_prefill_only) {
    descriptor.add_capabilities(proto::PROVIDER_CAPABILITY_PREFILL_ONLY);
    descriptor.add_capabilities(
        proto::PROVIDER_CAPABILITY_STRUCTURED_ADMISSION);
  }

  proto::ProviderDescriptor profile = descriptor;
  profile.mutable_identity()->clear_engine_uid();
  profile.mutable_endpoint()->clear_address();
  const std::optional<std::string> profile_digest =
      sha256_hex(kDescriptorDigestDomain, profile.SerializeAsString());
  if (!profile_digest.has_value()) {
    set_error("Failed to compute Native Provider profile digest", error);
    return std::nullopt;
  }
  descriptor.set_profile_digest(*profile_digest);
  return descriptor;
}

NativeEngineStateBuilder::NativeEngineStateBuilder(
    proto::ProviderDescriptor descriptor)
    : descriptor_(std::move(descriptor)) {
  descriptor_valid_ =
      descriptor_.contract_version() == kNativeProviderContractVersion &&
      descriptor_.identity().provider_id() == proto::PROVIDER_ID_XLLM_NATIVE &&
      !descriptor_.identity().engine_uid().empty() &&
      !descriptor_.profile_digest().empty() &&
      !descriptor_.model().model_revision().empty() &&
      descriptor_.topology().dp() > 0 &&
      has_capability(descriptor_, proto::PROVIDER_CAPABILITY_PER_DP_STATE);
}

std::optional<proto::EngineState> NativeEngineStateBuilder::build(
    const NativeEngineStateSnapshot& snapshot,
    std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  if (!descriptor_valid_) {
    set_error("Native EngineState builder has an invalid descriptor", error);
    return std::nullopt;
  }
  const size_t expected_dp = static_cast<size_t>(descriptor_.topology().dp());
  if (snapshot.kv_free_blocks.size() != expected_dp ||
      snapshot.kv_used_blocks.size() != expected_dp ||
      snapshot.kv_total_blocks.size() != expected_dp ||
      snapshot.lifecycle != proto::ENGINE_LIFECYCLE_READY) {
    set_error("Native EngineState snapshot is incomplete", error);
    return std::nullopt;
  }
  for (size_t dp_rank = 0; dp_rank < expected_dp; ++dp_rank) {
    const uint64_t free = snapshot.kv_free_blocks[dp_rank];
    const uint64_t used = snapshot.kv_used_blocks[dp_rank];
    const uint64_t total = snapshot.kv_total_blocks[dp_rank];
    if (free > total || used > total) {
      set_error("Native EngineState KV capacity is invalid", error);
      return std::nullopt;
    }
  }

  proto::EngineState state;
  state.set_engine_uid(descriptor_.identity().engine_uid());
  state.set_lifecycle(snapshot.lifecycle);
  state.set_ownership(proto::ENGINE_OWNERSHIP_OWNED);
  state.set_shallow_health(proto::HEALTH_STATUS_HEALTHY);
  state.set_deep_health(proto::HEALTH_STATUS_UNKNOWN);
  state.set_state_quality(proto::STATE_QUALITY_FULL);
  state.set_provider_id(descriptor_.identity().provider_id());
  state.set_profile_digest(descriptor_.profile_digest());
  state.set_model_revision(descriptor_.model().model_revision());
  state.set_heartbeat_age_ms_at_publish(0);
  state.set_state_age_ms_at_publish(0);

  for (size_t dp_rank = 0; dp_rank < expected_dp; ++dp_rank) {
    proto::PerDpEngineState* per_dp = state.add_per_dp();
    per_dp->set_dp_rank(static_cast<uint32_t>(dp_rank));
    per_dp->set_kv_free_blocks(snapshot.kv_free_blocks[dp_rank]);
    per_dp->set_kv_total_blocks(snapshot.kv_total_blocks[dp_rank]);
    per_dp->set_admission_credit(snapshot.lifecycle ==
                                         proto::ENGINE_LIFECYCLE_READY
                                     ? snapshot.kv_free_blocks[dp_rank]
                                     : 0);
    const double used = static_cast<double>(snapshot.kv_used_blocks[dp_rank]);
    const double total = static_cast<double>(snapshot.kv_total_blocks[dp_rank]);
    per_dp->set_kv_used_ratio(total == 0.0 ? 0.0 : used / total);
  }
  return state;
}

}  // namespace xllm
