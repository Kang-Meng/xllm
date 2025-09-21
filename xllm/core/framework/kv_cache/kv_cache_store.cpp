
#include "kv_cache_store.h"

#include <glog/logging.h>

#include "kv_cache_store_memcache.h"
#include "kv_cache_store_mooncake.h"

namespace xllm {

KVCacheStore::KVCacheStore(StoreConfig& config) : config_(std::move(config)) {
  auto key_tensor_one_layer = config_.host_kv_caches->at(0).get_k_cache();
  auto value_tensor_one_layer = config_.host_kv_caches->at(0).get_v_cache();

  key_cache_size_per_layer_ =
      key_tensor_one_layer[0].numel() * key_tensor_one_layer[0].element_size();
  value_cache_size_per_layer_ = value_tensor_one_layer[0].numel() *
                                value_tensor_one_layer[0].element_size();

  LOG(INFO) << "key_cache_size_per_layer: " << key_cache_size_per_layer_;
  LOG(INFO) << "value_cache_size_per_layer: " << value_cache_size_per_layer_;
}

std::shared_ptr<KVCacheStore> KVCacheStore::CreateKVCacheStore(
    StoreConfig& config) {
  if (config.store_type == "memcache") {
    return std::make_shared<MemcacheStore>(config);
  } else if (config.store_type == "mooncake") {
    return std::make_shared<MooncakeStore>(config);
  } else {
    LOG(ERROR) << "unrecognized store type : " << config.store_type;
    return nullptr;
  }
}

}  // namespace xllm
