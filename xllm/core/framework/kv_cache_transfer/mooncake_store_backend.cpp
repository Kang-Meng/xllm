/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache_transfer/mooncake_store_backend.h"

#include <glog/logging.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <utility>

namespace xllm {

MooncakeStoreBackend::~MooncakeStoreBackend() { unregister_ranges(); }

bool MooncakeStoreBackend::init(
    const MooncakeStoreBackendConfig& config,
    const std::vector<StoreRegisteredRange>& ranges) {
  const std::optional<std::vector<StoreRegisteredRange>> unique =
      unique_ranges(ranges);
  if (!unique.has_value()) {
    LOG(ERROR) << "Invalid Mooncake Host tensor registration ranges.";
    return false;
  }

  try {
    client_ptr_ = mooncake::RealClient::create();
    const int setup_result =
        client_ptr_->setup_real(config.localhost_name,
                                config.metadata_server,
                                /*global_segment_size=*/0,
                                /*local_buffer_size=*/0,
                                config.protocol,
                                config.rdma_devices,
                                config.master_server_address,
                                /*transfer_engine=*/nullptr,
                                /*ipc_socket_path=*/"",
                                /*enable_ssd_offload=*/false,
                                /*ssd_offload_path=*/"",
                                /*tenant_id=*/"default");
    if (setup_result != 0) {
      LOG(ERROR) << "Failed to initialize Mooncake RealClient: result="
                 << setup_result;
      client_ptr_.reset();
      return false;
    }
    rep_config_.replica_num = config.replica_num;
    registered_ranges_.reserve(unique->size());
    for (const StoreRegisteredRange& range : *unique) {
      const int register_result =
          client_ptr_->register_buffer(range.address, range.bytes);
      if (register_result != 0) {
        LOG(ERROR) << "Failed to register Mooncake Host tensor: result="
                   << register_result;
        unregister_ranges();
        client_ptr_.reset();
        return false;
      }
      registered_ranges_.emplace_back(range);
    }
  } catch (const std::exception& error) {
    LOG(ERROR) << "Mooncake RealClient initialization failed: " << error.what();
    unregister_ranges();
    client_ptr_.reset();
    return false;
  } catch (...) {
    LOG(ERROR) << "Mooncake RealClient initialization failed unexpectedly.";
    unregister_ranges();
    client_ptr_.reset();
    return false;
  }
  return true;
}

std::vector<uint8_t> MooncakeStoreBackend::batch_put(
    const std::vector<std::string>& keys,
    const std::vector<MultiBuffer>& buffers) {
  std::vector<uint8_t> statuses(keys.size(), /*value=*/0);
  if (client_ptr_ == nullptr || keys.size() != buffers.size()) {
    LOG(ERROR) << "Invalid Mooncake multi-buffer Put request.";
    return statuses;
  }

  std::vector<std::vector<void*>> all_buffers;
  std::vector<std::vector<size_t>> all_sizes;
  all_buffers.reserve(buffers.size());
  all_sizes.reserve(buffers.size());
  for (const MultiBuffer& buffer : buffers) {
    all_buffers.emplace_back(buffer.addresses);
    all_sizes.emplace_back(buffer.sizes);
  }

  try {
    const std::vector<int> results = client_ptr_->batch_put_from_multi_buffers(
        keys, all_buffers, all_sizes, rep_config_);
    if (results.size() > keys.size()) {
      LOG(ERROR) << "Mooncake Put returned extra result items.";
    }
    const size_t result_count = std::min(keys.size(), results.size());
    for (size_t result_index = 0; result_index < result_count; ++result_index) {
      statuses[result_index] = put_succeeded(results[result_index]) ? 1 : 0;
    }
  } catch (const std::exception& error) {
    LOG(ERROR) << "Mooncake Put failed: " << error.what();
  } catch (...) {
    LOG(ERROR) << "Mooncake Put failed unexpectedly.";
  }
  return statuses;
}

std::vector<uint8_t> MooncakeStoreBackend::batch_get(
    const std::vector<std::string>& keys,
    const std::vector<MultiBuffer>& buffers) {
  std::vector<uint8_t> statuses(keys.size(), /*value=*/0);
  if (keys.size() != buffers.size()) {
    LOG(ERROR) << "Mismatched Mooncake batch Get keys and buffers: keys="
               << keys.size() << " buffers=" << buffers.size();
    return statuses;
  }
  if (keys.empty()) {
    return statuses;
  }
  if (client_ptr_ == nullptr) {
    LOG(ERROR) << "Mooncake batch Get client is not initialized: batch_size="
               << keys.size() << " first_key=" << keys.front();
    return statuses;
  }

  std::vector<std::string> get_keys;
  std::vector<std::vector<void*>> all_buffers;
  std::vector<std::vector<size_t>> all_sizes;
  std::vector<int32_t> expected_bytes;
  std::vector<size_t> request_indices;
  get_keys.reserve(keys.size());
  all_buffers.reserve(buffers.size());
  all_sizes.reserve(buffers.size());
  expected_bytes.reserve(buffers.size());
  request_indices.reserve(keys.size());
  for (size_t index = 0; index < buffers.size(); ++index) {
    const MultiBuffer& buffer = buffers[index];
    const std::optional<int32_t> bytes = get_bytes(buffer);
    if (!bytes.has_value()) {
      LOG(ERROR) << "Invalid Mooncake Get buffer: index=" << index
                 << " key=" << keys[index] << " batch_size=" << keys.size()
                 << " addresses=" << buffer.addresses.size()
                 << " sizes=" << buffer.sizes.size()
                 << "; requires matching counts and total bytes <= INT32_MAX.";
      continue;
    }
    get_keys.emplace_back(keys[index]);
    expected_bytes.emplace_back(*bytes);
    all_buffers.emplace_back(buffer.addresses);
    all_sizes.emplace_back(buffer.sizes);
    request_indices.emplace_back(index);
  }
  if (get_keys.empty()) {
    return statuses;
  }

  // Error codes are per object. Unexpected exceptions cannot identify which
  // writes completed, so they propagate instead of being converted to misses.
  const std::vector<int> results = client_ptr_->batch_get_into_multi_buffers(
      get_keys, all_buffers, all_sizes, /*prefer_same_node=*/false);
  if (results.size() != get_keys.size()) {
    LOG(ERROR) << "Mooncake batch Get result count mismatch: batch_size="
               << keys.size() << " submitted=" << get_keys.size()
               << " results=" << results.size()
               << " first_key=" << get_keys.front();
  }
  const std::vector<uint8_t> get_results =
      get_statuses(expected_bytes, results);
  for (size_t index = 0; index < request_indices.size(); ++index) {
    statuses[request_indices[index]] = get_results[index];
  }
  return statuses;
}

std::vector<MooncakeReplicaTier> MooncakeStoreBackend::batch_query_tiers(
    const std::vector<std::string>& keys) {
  std::vector<MooncakeReplicaTier> tiers(keys.size(),
                                         MooncakeReplicaTier::MISSING);
  if (client_ptr_ == nullptr || keys.empty()) {
    return tiers;
  }
  try {
    const std::vector<tl::expected<mooncake::QueryResult, mooncake::ErrorCode>>
        results = client_ptr_->batch_query(keys);
    const size_t result_count = std::min(keys.size(), results.size());
    for (size_t index = 0; index < result_count; ++index) {
      if (results[index].has_value()) {
        tiers[index] = replica_tier(results[index].value().replicas);
      }
    }
  } catch (const std::exception& error) {
    LOG(WARNING) << "Mooncake BatchQuery failed: " << error.what();
  } catch (...) {
    LOG(WARNING) << "Mooncake BatchQuery failed unexpectedly.";
  }
  return tiers;
}

MooncakeReplicaTier MooncakeStoreBackend::replica_tier(
    const std::vector<mooncake::Replica::Descriptor>& replicas) {
  MooncakeReplicaTier tier = MooncakeReplicaTier::MISSING;
  for (const mooncake::Replica::Descriptor& replica : replicas) {
    // Mooncake reads only complete replicas, see SelectBestReplica.
    if (replica.status != mooncake::ReplicaStatus::COMPLETE) {
      continue;
    }
    if (replica.is_memory_replica()) {
      return MooncakeReplicaTier::MEMORY;
    }
    if (replica.is_nof_replica() || replica.is_local_disk_replica() ||
        replica.is_disk_replica()) {
      tier = MooncakeReplicaTier::DISK;
    }
  }
  return tier;
}

std::optional<std::vector<StoreRegisteredRange>>
MooncakeStoreBackend::unique_ranges(
    const std::vector<StoreRegisteredRange>& ranges) {
  std::vector<StoreRegisteredRange> unique;
  unique.reserve(ranges.size());
  for (const StoreRegisteredRange& range : ranges) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(range.address);
    if (range.address == nullptr || range.bytes == 0 ||
        range.bytes > std::numeric_limits<uintptr_t>::max() - address) {
      return std::nullopt;
    }
    const auto same_address = std::find_if(
        unique.begin(),
        unique.end(),
        [address](const StoreRegisteredRange& item) {
          return reinterpret_cast<uintptr_t>(item.address) == address;
        });
    if (same_address == unique.end()) {
      unique.emplace_back(range);
    } else if (same_address->bytes != range.bytes) {
      return std::nullopt;
    }
  }

  std::vector<StoreRegisteredRange> ordered = unique;
  std::sort(
      ordered.begin(),
      ordered.end(),
      [](const StoreRegisteredRange& lhs, const StoreRegisteredRange& rhs) {
        return reinterpret_cast<uintptr_t>(lhs.address) <
               reinterpret_cast<uintptr_t>(rhs.address);
      });
  for (size_t range_index = 1; range_index < ordered.size(); ++range_index) {
    const StoreRegisteredRange& previous = ordered[range_index - 1];
    const uintptr_t previous_end =
        reinterpret_cast<uintptr_t>(previous.address) + previous.bytes;
    if (reinterpret_cast<uintptr_t>(ordered[range_index].address) <
        previous_end) {
      return std::nullopt;
    }
  }
  return unique;
}

std::optional<int32_t> MooncakeStoreBackend::get_bytes(
    const MultiBuffer& buffer) {
  if (buffer.addresses.size() != buffer.sizes.size()) {
    return std::nullopt;
  }
  size_t bytes = 0;
  const size_t max_bytes =
      static_cast<size_t>(std::numeric_limits<int32_t>::max());
  for (const size_t size : buffer.sizes) {
    if (size > max_bytes - bytes) {
      return std::nullopt;
    }
    bytes += size;
  }
  return static_cast<int32_t>(bytes);
}

std::vector<uint8_t> MooncakeStoreBackend::get_statuses(
    const std::vector<int32_t>& expected_bytes,
    const std::vector<int>& results) {
  std::vector<uint8_t> statuses(expected_bytes.size(), /*value=*/0);
  const size_t result_count = std::min(expected_bytes.size(), results.size());
  for (size_t index = 0; index < result_count; ++index) {
    statuses[index] =
        results[index] >= 0 && results[index] == expected_bytes[index] ? 1 : 0;
  }
  return statuses;
}

bool MooncakeStoreBackend::put_succeeded(int result) {
  return result == 0 ||
         result == static_cast<int>(mooncake::ErrorCode::OBJECT_ALREADY_EXISTS);
}

void MooncakeStoreBackend::unregister_ranges() {
  if (client_ptr_ == nullptr) {
    return;
  }
  for (auto range_it = registered_ranges_.rbegin();
       range_it != registered_ranges_.rend();
       ++range_it) {
    try {
      const int unregister_result =
          client_ptr_->unregister_buffer(range_it->address);
      if (unregister_result != 0) {
        LOG(WARNING) << "Failed to unregister Mooncake Host tensor: result="
                     << unregister_result;
      }
    } catch (const std::exception& error) {
      LOG(WARNING) << "Failed to unregister Mooncake Host tensor: "
                   << error.what();
    } catch (...) {
      LOG(WARNING) << "Failed to unregister Mooncake Host tensor unexpectedly.";
    }
  }
  registered_ranges_.clear();
}

}  // namespace xllm
