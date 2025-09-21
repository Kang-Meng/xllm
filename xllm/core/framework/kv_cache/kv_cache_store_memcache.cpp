
#include "kv_cache_store_memcache.h"

#include "util/hash_util.h"

namespace xllm {
typedef enum {
  SMEMB_COPY_L2G = 0, /* copy data from local space to global space */
  SMEMB_COPY_G2L = 1, /* copy data from global space to local space */
  SMEMB_COPY_G2H = 2, /* copy data from global space to host memory */
  SMEMB_COPY_H2G = 3, /* copy data from host memory to global space */
  SMEMB_COPY_G2G = 4, /* copy data from global space to global space */
  /* add here */
  SMEMB_COPY_BUTT
} smem_bm_copy_type;

MemcacheStore::MemcacheStore(StoreConfig& config) : KVCacheStore(config) {
  obj_store_ = ock::mmc::ObjectStore::CreateObjectStore();
  obj_store_->Init(config_.device_idx);

  auto key_tensor_one_layer = config_.device_kv_caches->at(0).get_k_cache();
  auto value_tensor_one_layer = config_.device_kv_caches->at(0).get_v_cache();

  auto key_cache_size =
      key_tensor_one_layer.numel() * key_tensor_one_layer.element_size();
  auto value_cache_size =
      value_tensor_one_layer.numel() * value_tensor_one_layer.element_size();

  for (int layer = 0; layer < config_.device_kv_caches->size(); layer++) {
    void* key_cache = static_cast<char*>(
        config_.device_kv_caches->at(layer).get_k_cache().data_ptr());

    auto register_k_result =
        obj_store_->RegisterBuffer(key_cache, key_cache_size);

    if (register_k_result != 0) {
      LOG(ERROR)
          << "Failed to register device memory for key cache, error code: "
          << register_k_result;
      return;
    }

    void* value_cache = static_cast<char*>(
        config_.device_kv_caches->at(layer).get_v_cache().data_ptr());

    auto register_v_result =
        obj_store_->RegisterBuffer(value_cache, value_cache_size);

    if (register_v_result != 0) {
      LOG(ERROR)
          << "Failed to register device memory for value cache, error code: "
          << register_v_result;
      return;
    }
  }
}

uint64_t MemcacheStore::batch_put(
    const std::vector<CacheBlockInfo>& cache_block_info) {
  std::vector<std::string> str_keys;
  std::vector<std::vector<void*>> buffers;
  std::vector<std::vector<size_t>> sizes;

  str_keys.reserve(cache_block_info.size());
  buffers.reserve(cache_block_info.size());
  sizes.reserve(cache_block_info.size());
  for (auto block_info : cache_block_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        MURMUR_HASH3_VALUE_LEN);

    str_key.append(std::to_string(config_.tp_rank));

    if (obj_store_->IsExist(str_key) == 0) {
      continue;
    }
    str_keys.emplace_back(std::move(str_key));

    std::vector<void*> buffer;
    std::vector<size_t> size;
    buffer.reserve(config_.device_kv_caches->size() * 2);
    size.reserve(config_.device_kv_caches->size() * 2);

    for (int layer = 0; layer < config_.device_kv_caches->size(); layer++) {
      void* key_cache =
          static_cast<char*>(
              config_.device_kv_caches->at(layer).get_k_cache().data_ptr()) +
          block_info.device_block_id * key_cache_size_per_layer_;
      buffer.emplace_back(key_cache);
      size.emplace_back(key_cache_size_per_layer_);

      void* value_cache =
          static_cast<char*>(
              config_.device_kv_caches->at(layer).get_v_cache().data_ptr()) +
          block_info.device_block_id * value_cache_size_per_layer_;
      buffer.emplace_back(value_cache);
      size.emplace_back(value_cache_size_per_layer_);
    }
  }

  if (str_keys.size() == 0) {
    return cache_block_info.size();
  }

  uint64_t success_cnt = str_keys.size();
  auto results =
      obj_store_->BatchPutFromLayers(str_keys, buffers, sizes, SMEMB_COPY_L2G);

  for (int i = 0; i < str_keys.size(); i++) {
    if (results[i] != 0) {
      success_cnt = i;
      DLOG(ERROR) << "success_cnt: " << success_cnt
                  << ", failed to BatchGetIntoLayers, error code: "
                  << results[i];
      break;
    }
  }
  return success_cnt;
}

uint64_t MemcacheStore::batch_get(
    const std::vector<CacheBlockInfo>& cache_block_info) {
  std::vector<std::string> str_keys;
  std::vector<std::vector<void*>> buffers;
  std::vector<std::vector<size_t>> sizes;

  str_keys.reserve(cache_block_info.size());
  buffers.reserve(cache_block_info.size());
  sizes.reserve(cache_block_info.size());
  for (auto block_info : cache_block_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        MURMUR_HASH3_VALUE_LEN);

    str_key.append(std::to_string(config_.tp_rank));

    if (obj_store_->IsExist(str_key) == 0) {
      break;
    }
    str_keys.emplace_back(std::move(str_key));

    std::vector<void*> buffer;
    std::vector<size_t> size;
    buffer.reserve(config_.device_kv_caches->size() * 2);
    size.reserve(config_.device_kv_caches->size() * 2);

    for (int layer = 0; layer < config_.device_kv_caches->size(); layer++) {
      void* key_cache =
          static_cast<char*>(
              config_.device_kv_caches->at(layer).get_k_cache().data_ptr()) +
          block_info.device_block_id * key_cache_size_per_layer_;
      buffer.emplace_back(key_cache);
      size.emplace_back(key_cache_size_per_layer_);

      void* value_cache =
          static_cast<char*>(
              config_.device_kv_caches->at(layer).get_v_cache().data_ptr()) +
          block_info.device_block_id * value_cache_size_per_layer_;
      buffer.emplace_back(value_cache);
      size.emplace_back(value_cache_size_per_layer_);
    }
  }

  if (str_keys.size() == 0) {
    return 0;
  }

  uint64_t success_cnt = str_keys.size();

  auto results =
      obj_store_->BatchGetIntoLayers(str_keys, buffers, sizes, SMEMB_COPY_G2L);
  for (int i = 0; i < str_keys.size(); i++) {
    if (results[i] != 0) {
      success_cnt = i;
      DLOG(ERROR) << "success_cnt: " << success_cnt
                  << ", failed to BatchGetIntoLayers, error code: "
                  << results[i];
      break;
    }
  }
  return success_cnt;
}

uint64_t MemcacheStore::batch_remove(
    const std::vector<CacheBlockInfo>& cache_block_info) {
  uint64_t success_cnt = 0;
  for (auto block_info : cache_block_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        MURMUR_HASH3_VALUE_LEN);
    str_key.append(std::to_string(config_.tp_rank));

    auto result = obj_store_->Remove(str_key);

    if (result == 0) {
      success_cnt++;
    }
  }
  return success_cnt;
}

}  // namespace xllm
