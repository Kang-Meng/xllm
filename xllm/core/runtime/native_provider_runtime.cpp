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

#include "core/runtime/native_provider_runtime.h"

#include <torch/version.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include "core/common/types.h"
#include "core/common/xllm_build_info.h"
#include "core/distributed_runtime/engine.h"
#include "core/framework/block/block.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/platform/platform.h"

namespace xllm {
namespace {

void append_contract_value(std::string_view value, std::string* contract) {
  contract->append(std::to_string(value.size()));
  contract->push_back(':');
  contract->append(value);
}

template <typename Value>
void append_numeric_contract_value(Value value, std::string* contract) {
  append_contract_value(std::to_string(value), contract);
}

std::string lowercase_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](char character) {
    return static_cast<char>(
        std::tolower(static_cast<unsigned char>(character)));
  });
  return value;
}

std::optional<proto::EngineRole> native_engine_role(
    const InstanceInfo& instance_info) {
  const InstanceRole role(instance_info.type);
  if (role == InstanceRole::PREFILL) {
    return proto::ENGINE_ROLE_PREFILL;
  }
  if (role == InstanceRole::DECODE) {
    return proto::ENGINE_ROLE_DECODE;
  }
  if (role == InstanceRole::DEFAULT) {
    return proto::ENGINE_ROLE_AGGREGATED;
  }
  return std::nullopt;
}

std::string tokenizer_contract(const TokenizerArgs& args) {
  std::string contract;
  append_contract_value(args.tokenizer_type(), &contract);
  append_contract_value(args.vocab_file(), &contract);
  append_contract_value(args.pattern(), &contract);
  append_contract_value(args.chat_template(), &contract);
  append_contract_value(args.tokenizer_class(), &contract);
  append_contract_value(args.bos_token(), &contract);
  append_contract_value(args.eos_token(), &contract);
  append_contract_value(args.pad_token(), &contract);
  append_numeric_contract_value(args.add_bos_token(), &contract);
  append_numeric_contract_value(args.add_eos_token(), &contract);
  for (const SpecialToken& token : args.special_tokens()) {
    append_contract_value(token.first, &contract);
    append_numeric_contract_value(token.second, &contract);
  }
  for (const std::string& token : args.visible_special_tokens()) {
    append_contract_value(token, &contract);
  }
  for (const std::string& token : args.prefix_tokens()) {
    append_contract_value(token, &contract);
  }
  return contract;
}

std::string renderer_contract(const TokenizerArgs& tokenizer,
                              const ModelConfig& model) {
  std::string contract = tokenizer_contract(tokenizer);
  append_contract_value(model.backend(), &contract);
  append_contract_value(model.task(), &contract);
  append_contract_value(model.model_impl(), &contract);
  append_contract_value(model.reasoning_parser(), &contract);
  append_contract_value(model.tool_call_parser(), &contract);
  append_numeric_contract_value(model.use_cpp_chat_template(), &contract);
  return contract;
}

std::string kv_layout_contract(const ModelArgs& args,
                               const BlockManagerPool::Options& options,
                               bool include_block_size) {
  std::string contract;
  append_contract_value(args.model_type(), &contract);
  append_contract_value(args.dtype(), &contract);
  append_numeric_contract_value(args.n_layers(), &contract);
  append_numeric_contract_value(args.n_heads(), &contract);
  append_numeric_contract_value(args.n_kv_heads().value_or(args.n_heads()),
                                &contract);
  append_numeric_contract_value(args.head_dim(), &contract);
  append_numeric_contract_value(args.enable_mla(), &contract);
  append_numeric_contract_value(args.kv_lora_rank(), &contract);
  append_numeric_contract_value(args.window_size(), &contract);
  if (include_block_size) {
    append_numeric_contract_value(options.block_size(), &contract);
  }
  for (uint32_t manager_type : options.manager_types()) {
    append_numeric_contract_value(manager_type, &contract);
  }
  for (uint32_t ratio : options.compress_ratios()) {
    append_numeric_contract_value(ratio, &contract);
  }
  return contract;
}

std::vector<std::string> cache_groups(
    const BlockManagerPool::Options& options) {
  if (options.manager_types().empty()) {
    return {"group:" + std::to_string(cache_group_id(BlockType::KV))};
  }
  std::vector<std::string> groups;
  groups.reserve(options.manager_types().size());
  for (size_t index = 0; index < options.manager_types().size(); ++index) {
    const BlockType block_type =
        static_cast<BlockType>(options.manager_types()[index]);
    groups.emplace_back("group:" + std::to_string(cache_group_id(block_type)));
  }
  return groups;
}

std::string head_shard_mapping_contract(const ModelArgs& args,
                                        uint32_t tp,
                                        uint32_t dp,
                                        uint32_t cp,
                                        uint32_t kv_split_size) {
  std::string contract;
  append_numeric_contract_value(args.n_heads(), &contract);
  append_numeric_contract_value(args.n_kv_heads().value_or(args.n_heads()),
                                &contract);
  append_numeric_contract_value(args.head_dim(), &contract);
  append_numeric_contract_value(tp, &contract);
  append_numeric_contract_value(dp, &contract);
  append_numeric_contract_value(cp, &contract);
  append_numeric_contract_value(kv_split_size, &contract);
  return contract;
}

std::string scheduler_policy_contract(const SchedulerConfig& scheduler) {
  std::string contract;
  append_contract_value(scheduler.priority_strategy(), &contract);
  append_numeric_contract_value(scheduler.enable_schedule_overlap(), &contract);
  append_numeric_contract_value(scheduler.enable_chunked_prefill(), &contract);
  append_numeric_contract_value(scheduler.max_tokens_per_chunk_for_prefill(),
                                &contract);
  append_numeric_contract_value(scheduler.enable_mix_batch(), &contract);
  append_numeric_contract_value(scheduler.enable_online_preempt_offline(),
                                &contract);
  return contract;
}

}  // namespace

std::optional<NativeProviderDescriptorConfig>
collect_native_provider_descriptor_config(
    const InstanceInfo& instance_info,
    const Engine& engine,
    const BlockManagerPool& block_manager_pool,
    std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  const std::optional<proto::EngineRole> role =
      native_engine_role(instance_info);
  if (!role.has_value()) {
    return std::nullopt;
  }
  const DisaggPDConfig& disagg = DisaggPDConfig::get_instance();
  const bool aggregated = *role == proto::ENGINE_ROLE_AGGREGATED;
  if (aggregated) {
    if (disagg.enable_disagg_pd()) {
      return std::nullopt;
    }
  } else if (!disagg.enable_disagg_pd() ||
             disagg.kv_cache_transfer_mode() != "PUSH") {
    return std::nullopt;
  }

  const ParallelConfig& parallel = ParallelConfig::get_instance();
  if (parallel.tp_size() <= 0 || parallel.dp_size() <= 0 ||
      parallel.ep_size() <= 0 || parallel.cp_size() <= 0 ||
      parallel.tp_size() >
          static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    if (error != nullptr) {
      *error = "Native V2 parallel topology is invalid";
    }
    return std::nullopt;
  }
  if (instance_info.dp_size != parallel.dp_size()) {
    if (error != nullptr) {
      *error = "Native V2 runtime and registered DP topology disagree";
    }
    return std::nullopt;
  }
  const uint64_t dp_cp_size = static_cast<uint64_t>(parallel.dp_size()) *
                              static_cast<uint64_t>(parallel.cp_size());
  const uint64_t configured_device_count =
      static_cast<uint64_t>(parallel.tp_size()) * dp_cp_size;
  const size_t device_count = *role == proto::ENGINE_ROLE_AGGREGATED
                                  ? static_cast<size_t>(configured_device_count)
                                  : instance_info.cluster_ids.size();
  if (device_count == 0 ||
      device_count > std::numeric_limits<uint32_t>::max() ||
      device_count % dp_cp_size != 0) {
    if (error != nullptr) {
      *error = "Native V2 observed worker topology is invalid";
    }
    return std::nullopt;
  }
  const uint32_t tp = static_cast<uint32_t>(device_count / dp_cp_size);
  const uint32_t dp = static_cast<uint32_t>(parallel.dp_size());
  const uint32_t cp = static_cast<uint32_t>(parallel.cp_size());
  const int32_t kv_split_size = parallel.kv_split_size_effective();
  if (tp == 0 || kv_split_size <= 0 ||
      parallel.cp_size() % kv_split_size != 0) {
    if (error != nullptr) {
      *error = "Native V2 effective TP or KV split topology is invalid";
    }
    return std::nullopt;
  }

  const ModelConfig& model_config = ModelConfig::get_instance();
  const std::string model_revision = model_config.model_id().empty()
                                         ? model_config.model()
                                         : model_config.model_id();
  const ModelArgs& model_args = engine.model_args();
  const TokenizerArgs& tokenizer_args = engine.tokenizer_args();
  const SchedulerConfig& scheduler = SchedulerConfig::get_instance();
  const KVCacheConfig& kv_cache = KVCacheConfig::get_instance();
  if (block_manager_pool.block_size() <= 0 ||
      scheduler.max_seqs_per_batch() <= 0 ||
      scheduler.max_tokens_per_batch() <= 0) {
    if (error != nullptr) {
      *error = "Native V2 cache or scheduler capacity is invalid";
    }
    return std::nullopt;
  }
  const std::string connector =
      lowercase_ascii(disagg.kv_cache_transfer_mode());
  const std::string runtime_version = XLLM_BUILD_VERSION;
  const std::string hardware_runtime_version =
      Platform::type_str() + ";torch=" + std::to_string(TORCH_VERSION_MAJOR) +
      "." + std::to_string(TORCH_VERSION_MINOR) + "." +
      std::to_string(TORCH_VERSION_PATCH);
  const std::string model_dtype =
      model_args.dtype().empty() ? "unspecified" : model_args.dtype();
  const std::vector<std::string> api_features =
      api_features_for_backend(model_config.backend());
  if (api_features.empty()) {
    if (error != nullptr) {
      *error = "Native V2 serving backend is unknown for API features";
    }
    return std::nullopt;
  }

  return NativeProviderDescriptorConfig{
      .engine_uid = instance_info.name,
      .api_address = instance_info.name,
      .role = *role,
      .runtime_version = runtime_version,
      .hardware_runtime_version = hardware_runtime_version,
      .model_revision = model_revision,
      .tokenizer_revision = model_revision,
      .chat_template_contract = tokenizer_args.chat_template(),
      .renderer_contract = renderer_contract(tokenizer_args, model_config),
      .quantization = "model-dtype=" + model_dtype,
      .soc = Platform::type_str(),
      .device_count = static_cast<uint32_t>(device_count),
      .tp = tp,
      .dp = dp,
      .pp = 1,
      .ep = static_cast<uint32_t>(parallel.ep_size()),
      .cp = cp,
      .super_node_id = disagg.super_node_id(),
      .kv_layout_contract = kv_layout_contract(model_args,
                                               block_manager_pool.options(),
                                               /*include_block_size=*/true),
      .kv_transfer_layout_contract =
          kv_layout_contract(model_args,
                             block_manager_pool.options(),
                             /*include_block_size=*/false),
      .kv_cache_dtype = kv_cache.kv_cache_dtype(),
      .block_size = static_cast<uint32_t>(block_manager_pool.block_size()),
      .hash_seed = kv_cache.xxh3_128bits_seed(),
      .logical_kv_bytes_per_token =
          block_manager_pool.options().logical_kv_bytes_per_token(),
      .cache_groups = cache_groups(block_manager_pool.options()),
      .head_shard_mapping_contract = head_shard_mapping_contract(
          model_args, tp, dp, cp, static_cast<uint32_t>(kv_split_size)),
      .connector = connector,
      .connector_version = "builtin-" + runtime_version,
      .scheduler_class = aggregated               ? "continuous"
                         : disagg.enable_pd_ooc() ? "pd_ooc"
                                                  : "disagg_pd",
      .max_num_seqs = static_cast<uint32_t>(scheduler.max_seqs_per_batch()),
      .max_num_batched_tokens =
          static_cast<uint64_t>(scheduler.max_tokens_per_batch()),
      .scheduler_policy_contract = scheduler_policy_contract(scheduler),
      .supports_local_prefill_decode = *role == proto::ENGINE_ROLE_DECODE &&
                                       scheduler.enable_chunked_prefill() &&
                                       scheduler.enable_mix_batch(),
      .supports_prefill_only = *role == proto::ENGINE_ROLE_PREFILL,
      .api_features = api_features,
  };
}

}  // namespace xllm
