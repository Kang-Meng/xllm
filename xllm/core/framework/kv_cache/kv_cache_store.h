#pragma once

#include <Mooncake/mooncake-store/include/client.h>
#include <glog/logging.h>

#include <string>

#include "common/macros.h"
#include "framework/model/model_input_params.h"
#include "kv_cache.h"

namespace xllm {

struct StoreConfig {
  std::string localhost_name = "127.0.0.1";
  std::string protocol = "tcp";
  std::string metadata_connstring = "";
  std::string master_server_entry = "";
  int replica_num = 1;
  uint32_t tp_rank = 0;
};

class KVCacheStore {
 public:
  KVCacheStore(const StoreConfig& config,
               std::vector<xllm::KVCache>* host_kv_caches);
  ~KVCacheStore();

  uint64_t batch_put(const std::vector<CacheBlockInfo>& cache_block_info);

  uint64_t batch_get(const std::vector<CacheBlockInfo>& cache_block_info);

  uint64_t batch_remove(const std::vector<CacheBlockInfo>& cache_block_info);

 private:
  StoreConfig config_;
  mooncake::ReplicateConfig rep_config_;

  void** args_ = nullptr;

  std::vector<xllm::KVCache>* host_kv_caches_;

  uint64_t key_cache_size_per_layer_;
  uint64_t value_cache_size_per_layer_;

  std::shared_ptr<mooncake::Client> client_ptr_;
};

}  // namespace xllm
