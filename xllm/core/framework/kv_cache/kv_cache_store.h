#pragma once
#include <glog/logging.h>

#include <string>

#include "common/macros.h"
#include "framework/model/model_input_params.h"
#include "kv_cache.h"

namespace xllm {

struct StoreConfig {
  std::string store_type = "memcache";
  std::string localhost_name = "127.0.0.1";
  std::string protocol = "tcp";
  std::string metadata_connstring = "";
  std::string master_server_entry = "";
  int replica_num = 1;
  uint32_t tp_rank = 0;
  uint32_t device_idx = 0;
  std::vector<xllm::KVCache>* host_kv_caches;
  std::vector<xllm::KVCache>* device_kv_caches;
};

class KVCacheStore {
 public:
  KVCacheStore(StoreConfig& config);
  virtual ~KVCacheStore() = default;

  virtual uint64_t batch_put(
      const std::vector<CacheBlockInfo>& cache_block_info) = 0;

  virtual uint64_t batch_get(
      const std::vector<CacheBlockInfo>& cache_block_info) = 0;

  virtual uint64_t batch_remove(
      const std::vector<CacheBlockInfo>& cache_block_info) = 0;

  static std::shared_ptr<KVCacheStore> CreateKVCacheStore(StoreConfig& config);

 protected:
  StoreConfig config_;
  uint64_t key_cache_size_per_layer_;
  uint64_t value_cache_size_per_layer_;
};

}  // namespace xllm
