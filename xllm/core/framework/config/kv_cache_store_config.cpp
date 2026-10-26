/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/config/kv_cache_store_config.h"

#include <glog/logging.h>

#include <algorithm>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"
#include "core/framework/config/execution_config.h"

DEFINE_uint32(
    prefetch_timeout,
    120000,
    "Stop KV cache Store prefetch after this timeout in milliseconds; "
    "stalled prefetch streams are closed.");

DEFINE_uint32(prefetch_batch_size,
              2,
              "Prefetch from kvcache store copy batch size.");

DEFINE_uint32(prefetch_valid_blocks_per_batch,
              1,
              "Valid LINEAR blocks placed in one prefetch batch. At startup "
              "this is clamped to max(min(prefetch_batch_size, value), 1) "
              "and then to the largest divisor of prefetch_batch_size.");

DEFINE_uint32(layers_wise_copy_batchs,
              4,
              "Layer wise H2D copy batches. Graph mode waits for every copy "
              "before replay and initializes this to 1.");

DEFINE_double(host_blocks_factor,
              0.0,
              "Host block factor, e.g. host block num = host_blocks_factor * "
              "hbm block num.");

DEFINE_bool(enable_kvcache_store, false, "Whether to use kvcache store.");

DEFINE_string(store_protocol,
              "tcp",
              "KV cache store protocol(e.g. tcp, rdma).");

DEFINE_string(store_rdma_devices,
              "",
              "Comma-separated RDMA HCAs for the embedded Store client.");

DEFINE_string(store_master_server_address,
              "",
              "The Store master address: IP:Port for standalone mode or "
              "etcd://IP:Port;IP:Port;... for high availability mode.");

DEFINE_string(store_metadata_server,
              "",
              "The address of the kv cache store metadata service.");

DEFINE_string(store_local_hostname,
              "",
              "The local host name of the kv cache store client.");

DEFINE_bool(enable_control_h2d_block_num,
            false,
            "Whether to control h2d copy block num.");

namespace xllm {
namespace {

// prefetch_batch_size is max(1, value). prefetch_valid_blocks_per_batch is
// max(min(batch, requested), 1). A value that already divides the batch stays;
// otherwise it walks down to the nearest divisor.
void normalize_prefetch_batch_shape() {
  const uint32_t requested_batch = FLAGS_prefetch_batch_size;
  const uint32_t batch_size = std::max(requested_batch, 1u);
  if (batch_size != requested_batch) {
    LOG(WARNING) << "prefetch_batch_size=" << requested_batch
                 << " is below 1; using max(1, prefetch_batch_size)="
                 << batch_size << ".";
    FLAGS_prefetch_batch_size = batch_size;
  }

  const uint32_t requested_valid = FLAGS_prefetch_valid_blocks_per_batch;
  uint32_t valid_per_batch =
      std::max(std::min(batch_size, requested_valid), 1u);
  while (valid_per_batch > 1 && batch_size % valid_per_batch != 0) {
    --valid_per_batch;
  }
  if (valid_per_batch != requested_valid) {
    LOG(WARNING) << "prefetch_valid_blocks_per_batch=" << requested_valid
                 << " with prefetch_batch_size=" << batch_size
                 << " does not match max(min(prefetch_batch_size, "
                    "prefetch_valid_blocks_per_batch), 1) rounded down to a "
                    "divisor of prefetch_batch_size; using "
                 << valid_per_batch << ".";
    FLAGS_prefetch_valid_blocks_per_batch = valid_per_batch;
  }
}

// Graph replay cannot wait between layers. One event covers the whole H2D
// copy, and both the Python and C++ graph paths wait for it before replay.
void normalize_layers_wise_copy_batchs() {
  if (!ExecutionConfig::get_instance().enable_graph()) {
    return;
  }
  const uint32_t requested = FLAGS_layers_wise_copy_batchs;
  if (requested == 1) {
    return;
  }
  LOG(WARNING) << "layers_wise_copy_batchs=" << requested
               << " is not compatible with enable_graph; using 1 so the "
                  "whole Host load finishes before graph replay.";
  FLAGS_layers_wise_copy_batchs = 1;
}

}  // namespace

void KVCacheStoreConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(prefetch_timeout);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(prefetch_batch_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(prefetch_valid_blocks_per_batch);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(layers_wise_copy_batchs);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(host_blocks_factor);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_kvcache_store);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(store_protocol);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(store_rdma_devices);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(store_master_server_address);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(store_metadata_server);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(store_local_hostname);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_control_h2d_block_num);
}

void KVCacheStoreConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(prefetch_timeout);
  XLLM_CONFIG_ASSIGN_FROM_JSON(prefetch_batch_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(prefetch_valid_blocks_per_batch);
  XLLM_CONFIG_ASSIGN_FROM_JSON(layers_wise_copy_batchs);
  XLLM_CONFIG_ASSIGN_FROM_JSON(host_blocks_factor);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_kvcache_store);
  XLLM_CONFIG_ASSIGN_FROM_JSON(store_protocol);
  XLLM_CONFIG_ASSIGN_FROM_JSON(store_rdma_devices);
  XLLM_CONFIG_ASSIGN_FROM_JSON(store_master_server_address);
  XLLM_CONFIG_ASSIGN_FROM_JSON(store_metadata_server);
  XLLM_CONFIG_ASSIGN_FROM_JSON(store_local_hostname);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_control_h2d_block_num);
}

void KVCacheStoreConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const KVCacheStoreConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, prefetch_timeout);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, prefetch_batch_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, prefetch_valid_blocks_per_batch);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, layers_wise_copy_batchs);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, host_blocks_factor);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_kvcache_store);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_protocol);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_rdma_devices);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_master_server_address);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_metadata_server);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_local_hostname);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_control_h2d_block_num);
}

KVCacheStoreConfig& KVCacheStoreConfig::get_instance() {
  static KVCacheStoreConfig config;
  return config;
}

void KVCacheStoreConfig::initialize() {
  normalize_prefetch_batch_shape();
  normalize_layers_wise_copy_batchs();
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
