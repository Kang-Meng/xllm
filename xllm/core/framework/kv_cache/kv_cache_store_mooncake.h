#pragma once

#include <Mooncake/mooncake-store/include/client.h>

#include "kv_cache_store.h"

namespace xllm {

class MooncakeStore final : public KVCacheStore {
 public:
  MooncakeStore(StoreConfig& config);
  ~MooncakeStore();

  uint64_t batch_put(
      const std::vector<CacheBlockInfo>& cache_block_info) override;

  uint64_t batch_get(
      const std::vector<CacheBlockInfo>& cache_block_info) override;

  uint64_t batch_remove(
      const std::vector<CacheBlockInfo>& cache_block_info) override;

 private:
  mooncake::ReplicateConfig rep_config_;
  void** args_ = nullptr;
  std::shared_ptr<mooncake::Client> client_ptr_;
};

}  // namespace xllm
