#pragma once
#include <memcache/cpp/mmcache.h>

#include "kv_cache_store.h"

namespace xllm {

class MemcacheStore final : public KVCacheStore {
 public:
  MemcacheStore(StoreConfig& config);
  ~MemcacheStore() = default;

  uint64_t batch_put(
      const std::vector<CacheBlockInfo>& cache_block_info) override;

  uint64_t batch_get(
      const std::vector<CacheBlockInfo>& cache_block_info) override;

  uint64_t batch_remove(
      const std::vector<CacheBlockInfo>& cache_block_info) override;

 private:
  std::shared_ptr<ock::mmc::ObjectStore> obj_store_;
};

}  // namespace xllm
