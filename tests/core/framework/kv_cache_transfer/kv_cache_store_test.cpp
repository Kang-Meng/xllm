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

#include "framework/kv_cache_transfer/kv_cache_store.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "framework/kv_cache_transfer/mooncake_store_backend.h"

namespace xllm {

class KVCacheStoreTestPeer final {
 public:
  static void attach_transport(
      KVCacheStore* store,
      std::unique_ptr<KVCacheStoreTransport> transport) {
    store->transport_ = std::move(transport);
    store->is_initialized_ = true;
  }
  static std::vector<uint8_t> read_buffers(
      KVCacheStore* store,
      const std::vector<std::string>& keys,
      const std::vector<MultiBuffer>& buffers) {
    return store->transport_->batch_get(keys, buffers);
  }

  static void initialize_index(KVCacheStore* store,
                               const KVCacheStoreInitConfig& config,
                               HostCacheStoreIndex store_index) {
    store->config_ = config;
    store->initialize_store_index(std::move(store_index));
  }

  static std::vector<std::pair<std::string, std::string>> build_keys(
      const KVCacheStore& store,
      std::vector<BlockTransferInfo> block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    const std::vector<KVCacheStore::PhysicalRequest> requests =
        store.build_requests(slice);
    std::vector<std::pair<std::string, std::string>> keys;
    keys.reserve(requests.size());
    for (const KVCacheStore::PhysicalRequest& request : requests) {
      keys.emplace_back(request.entry->key_component, request.key);
    }
    return keys;
  }

  static std::vector<uint8_t> aggregate(
      const KVCacheStore& store,
      std::vector<BlockTransferInfo> block_transfer_info,
      const std::vector<uint8_t>& physical_results) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    const std::vector<KVCacheStore::PhysicalRequest> requests =
        store.build_requests(slice);
    return KVCacheStore::aggregate_results(
        block_transfer_info.size(), requests, physical_results);
  }

  static size_t physical_request_count(
      const KVCacheStore& store,
      std::vector<BlockTransferInfo> block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    return store.build_requests(slice).size();
  }

  static size_t unique_key_count(
      const KVCacheStore& store,
      std::vector<BlockTransferInfo> block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    const std::vector<KVCacheStore::PhysicalRequest> requests =
        store.build_requests(slice);
    return KVCacheStore::group_requests(requests).size();
  }

  static std::vector<uint8_t> present_objects(
      const std::vector<MooncakeReplicaTier>& tiers) {
    return KVCacheStore::present_objects(tiers);
  }

  static void record_get(MooncakeReplicaTier tier,
                         uint64_t bytes,
                         StoreGetStats* stats) {
    KVCacheStore::record_get(tier, bytes, stats);
  }

  static void mark_initialized(KVCacheStore* store) {
    store->is_initialized_ = true;
  }

  static std::optional<std::string> store_device_names(
      const KVCacheStoreInitConfig& config) {
    return KVCacheStore::get_store_device_names(config);
  }

  static std::optional<MultiBuffer> multi_buffer(
      const KVCacheStore& store,
      std::vector<BlockTransferInfo> block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    const std::vector<KVCacheStore::PhysicalRequest> requests =
        store.build_requests(slice);
    if (requests.empty()) {
      return std::nullopt;
    }
    const KVCacheStore::PhysicalRequest& request = requests.front();
    return store.build_multi_buffer(
        *request.entry,
        block_transfer_info[request.logical_index].dst_block_id);
  }
};

class MooncakeStoreBackendTestPeer final {
 public:
  static std::optional<std::vector<StoreRegisteredRange>> unique_ranges(
      const std::vector<StoreRegisteredRange>& ranges) {
    return MooncakeStoreBackend::unique_ranges(ranges);
  }

  static std::optional<int32_t> get_bytes(const MultiBuffer& buffer) {
    return MooncakeStoreBackend::get_bytes(buffer);
  }

  static std::vector<uint8_t> get_statuses(
      const std::vector<int32_t>& expected_bytes,
      const std::vector<int>& results) {
    return MooncakeStoreBackend::get_statuses(expected_bytes, results);
  }

  static bool put_succeeded(int result) {
    return MooncakeStoreBackend::put_succeeded(result);
  }

  static MooncakeReplicaTier replica_tier(
      const std::vector<mooncake::Replica::Descriptor>& replicas) {
    return MooncakeStoreBackend::replica_tier(replicas);
  }
};

namespace {

using StoreObjects = std::unordered_map<std::string, std::vector<uint8_t>>;
class MemoryStoreTransport final : public KVCacheStoreTransport {
 public:
  explicit MemoryStoreTransport(std::shared_ptr<StoreObjects> objects)
      : objects_(std::move(objects)) {}
  std::vector<uint8_t> batch_put(
      const std::vector<std::string>& keys,
      const std::vector<MultiBuffer>& buffers) override {
    std::vector<uint8_t> results(keys.size(), /*value=*/0);
    for (size_t index = 0; index < keys.size(); ++index) {
      if (objects_->count(keys[index]) != 0) {
        results[index] = 1;
        continue;
      }
      if (failed_puts_.count(keys[index]) != 0) {
        continue;
      }
      const auto& buffer = buffers[index];
      size_t bytes = 0;
      for (size_t size : buffer.sizes) {
        bytes += size;
      }
      auto& object = (*objects_)[keys[index]];
      object.resize(bytes);
      size_t offset = 0;
      for (size_t part = 0; part < buffer.sizes.size(); ++part) {
        std::memcpy(
            object.data() + offset, buffer.addresses[part], buffer.sizes[part]);
        offset += buffer.sizes[part];
      }
      results[index] = 1;
    }
    if (result_count_.has_value()) {
      results.resize(*result_count_);
    }
    return results;
  }
  std::vector<uint8_t> batch_get(
      const std::vector<std::string>& keys,
      const std::vector<MultiBuffer>& buffers) override {
    ++batch_get_calls_;
    last_get_size_ = keys.size();
    std::vector<uint8_t> results(keys.size(), /*value=*/0);
    for (size_t index = 0; index < keys.size(); ++index) {
      if (failed_gets_.count(keys[index]) != 0) {
        continue;
      }
      const auto object = objects_->find(keys[index]);
      if (object == objects_->end()) {
        continue;
      }
      const auto& buffer = buffers[index];
      size_t bytes = 0;
      for (size_t size : buffer.sizes) {
        bytes += size;
      }
      if (object->second.size() != bytes) {
        continue;
      }
      size_t offset = 0;
      for (size_t part = 0; part < buffer.sizes.size(); ++part) {
        std::memcpy(buffer.addresses[part],
                    object->second.data() + offset,
                    buffer.sizes[part]);
        offset += buffer.sizes[part];
      }
      results[index] = 1;
    }
    if (result_count_.has_value()) {
      results.resize(*result_count_);
    }
    return results;
  }
  std::vector<MooncakeReplicaTier> batch_query_tiers(
      const std::vector<std::string>& keys) override {
    std::vector<MooncakeReplicaTier> tiers;
    tiers.reserve(keys.size());
    for (const std::string& key : keys) {
      tiers.emplace_back(objects_->count(key) != 0
                             ? MooncakeReplicaTier::MEMORY
                             : MooncakeReplicaTier::MISSING);
    }
    return tiers;
  }
  std::optional<size_t> result_count_;
  std::unordered_set<std::string> failed_gets_;
  std::unordered_set<std::string> failed_puts_;
  int32_t batch_get_calls_ = 0;
  size_t last_get_size_ = 0;

 private:
  std::shared_ptr<StoreObjects> objects_;
};

class ScopedEnvVar final {
 public:
  explicit ScopedEnvVar(const std::string& name) : name_(name) {
    const char* value = std::getenv(name_.c_str());
    if (value != nullptr) {
      old_value_ = value;
    }
  }

  ~ScopedEnvVar() {
    if (old_value_.has_value()) {
      setenv(name_.c_str(), old_value_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  bool set(const std::string& value) {
    return setenv(name_.c_str(), value.c_str(), /*overwrite=*/1) == 0;
  }

 private:
  std::string name_;
  std::optional<std::string> old_value_;
};

KVCache make_attention_cache(int64_t host_blocks, int64_t width) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  return KVCache(
      KVCacheTensors{torch::zeros({host_blocks, 2, width}, options),
                     torch::zeros({host_blocks, 2, width}, options)});
}

enum class ReplicaKind : uint8_t {
  MEMORY = 0,
  NOF = 1,
  DISK = 2,
  LOCAL_DISK = 3,
};

KVCache make_linear_cache(int64_t host_blocks, int64_t width) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  return KVCache(LinearAttentionKVCacheTensors{
      torch::zeros({host_blocks, width}, options),
      torch::zeros({host_blocks, width}, options)});
}

KVCache make_compressed_kpool_cache(int64_t host_blocks,
                                    int64_t index_width,
                                    int64_t tail_blocks) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  IndexedKVCacheTensors tensors;
  tensors.kv_cache_tensors.key_cache =
      torch::zeros({host_blocks, 2, 8}, options);
  tensors.index_cache = torch::zeros({host_blocks, 2, 1, index_width}, options);
  tensors.kpool_tail = torch::zeros({tail_blocks, 2, 4, index_width}, options);
  return KVCache(tensors);
}

BlockTransferInfo make_block_info(uint8_t hash_value,
                                  BlockType block_type = BlockType::KV,
                                  int32_t destination_block_id = 0) {
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> hash_key;
  hash_key.fill(hash_value);
  return BlockTransferInfo(/*src_id=*/0,
                           destination_block_id,
                           hash_key.data(),
                           TransferType::G2H,
                           block_type);
}

KVCacheStoreInitConfig make_store_config(
    const std::string& model_id = "target-model",
    uint32_t tp_rank = 1,
    uint32_t tp_size = 2,
    bool enable_mla = false,
    int32_t cp_index_write_mode = kCpIndexWriteModeReplicated) {
  KVCacheStoreInitConfig config;
  config.model_id = model_id;
  config.tp_rank = tp_rank;
  config.tp_size = tp_size;
  config.kv_split_full_domain_size = static_cast<int32_t>(tp_size);
  config.enable_mla = enable_mla;
  config.cp_index_write_mode = cp_index_write_mode;
  return config;
}

std::string key_for_component(
    const std::vector<std::pair<std::string, std::string>>& keys,
    const std::string& component) {
  const auto key =
      std::find_if(keys.begin(), keys.end(), [&component](const auto& entry) {
        return entry.first == component;
      });
  return key == keys.end() ? "" : key->second;
}

class KVCacheStoreIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* master = std::getenv("XLLM_TEST_STORE_MASTER");
    if (master == nullptr) {
      GTEST_SKIP() << "Set XLLM_TEST_STORE_MASTER to a Mooncake master with "
                      "an active TCP storage node.";
    }
    config_ = make_store_config("split-store-test",
                                /*tp_rank=*/0,
                                /*tp_size=*/8,
                                /*enable_mla=*/true);
    config_.model_id +=
        ":" + std::to_string(getpid()) + ":" +
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    config_.master_server_address = master;
    config_.metadata_server = "P2PHANDSHAKE";
  }

  bool init_store(KVCacheStore& store,
                  KVCache& cache,
                  const KVCacheStoreInitConfig& config) {
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                          "main",
                                                          &cache,
                                                          std::nullopt,
                                                          /*is_target=*/true});
    return store.init(config, std::move(index));
  }

  KVCacheStoreInitConfig config_;
};

TEST_F(KVCacheStoreIntegrationTest,
       RestoresEachSplitLayoutWithoutKeyCollisions) {
  const std::vector<BlockTransferInfo> infos = {make_block_info(91)};
  for (int32_t size : {1, 2, 4, 8}) {
    for (int32_t rank = 0; rank < size; ++rank) {
      KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
      cache.get_k_cache().fill_(size * 10 + rank);
      cache.get_v_cache().fill_(size * 10 + rank + 100);
      config_.cp_size = size;
      config_.cp_rank = rank;
      config_.kv_split_full_domain_size =
          size * static_cast<int32_t>(config_.tp_size);
      config_.kv_split_size = size;
      config_.kv_split_rank = rank;
      KVCacheStore store;
      ASSERT_TRUE(init_store(store, cache, config_));
      ASSERT_EQ(store.batch_put(infos), 1U);
    }
  }
  for (int32_t size : {1, 2, 4, 8}) {
    for (int32_t rank = 0; rank < size; ++rank) {
      SCOPED_TRACE(::testing::Message() << "size=" << size << " rank=" << rank);
      KVCache cache = make_attention_cache(/*host_blocks=*/3, /*width=*/8);
      config_.cp_size = size;
      config_.cp_rank = rank;
      config_.kv_split_full_domain_size =
          size * static_cast<int32_t>(config_.tp_size);
      config_.kv_split_size = size;
      config_.kv_split_rank = rank;
      KVCacheStore store;
      ASSERT_TRUE(init_store(store, cache, config_));
      ASSERT_EQ(store.batch_get(infos), 1U);
      EXPECT_TRUE(torch::equal(
          cache.get_k_cache()[0],
          torch::full_like(cache.get_k_cache()[0], size * 10 + rank)));
      EXPECT_TRUE(torch::equal(
          cache.get_v_cache()[0],
          torch::full_like(cache.get_v_cache()[0], size * 10 + rank + 100)));
    }
  }
}

TEST_F(KVCacheStoreIntegrationTest, MlaReplicaReadsTpZeroPublishedSplit) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  cache.get_k_cache().fill_(23);
  cache.get_v_cache().fill_(47);
  config_.cp_size = 4;
  config_.cp_rank = 2;
  config_.kv_split_full_domain_size = 4 * static_cast<int32_t>(config_.tp_size);
  config_.kv_split_size = 4;
  config_.kv_split_rank = 2;
  config_.tp_rank = 0;
  const std::vector<BlockTransferInfo> infos = {make_block_info(92)};
  {
    KVCacheStore writer;
    ASSERT_TRUE(init_store(writer, cache, config_));
    ASSERT_EQ(writer.batch_put(infos), 1U);
  }
  cache.get_k_cache().zero_();
  cache.get_v_cache().zero_();
  config_.tp_rank = 4;
  KVCacheStore replica;
  ASSERT_TRUE(init_store(replica, cache, config_));
  ASSERT_EQ(replica.batch_get(infos), 1U);
  EXPECT_TRUE(torch::equal(cache.get_k_cache()[0],
                           torch::full_like(cache.get_k_cache()[0], 23)));
  EXPECT_TRUE(torch::equal(cache.get_v_cache()[0],
                           torch::full_like(cache.get_v_cache()[0], 47)));
  EXPECT_EQ(replica.batch_put(infos), 1U);
}

TEST_F(KVCacheStoreIntegrationTest, NonzeroCpRankPublishesItsSplit) {
  for (bool enable_mla : {false, true}) {
    for (int32_t split_size : {2, 4}) {
      SCOPED_TRACE(::testing::Message() << "enable_mla=" << enable_mla
                                        << " split_size=" << split_size);
      config_.enable_mla = enable_mla;
      config_.cp_size = split_size;
      config_.cp_rank = split_size - 1;
      config_.tp_size = 2;
      config_.tp_rank = enable_mla ? 0 : 1;
      config_.kv_split_full_domain_size =
          split_size * static_cast<int32_t>(config_.tp_size);
      config_.kv_split_size = split_size;
      config_.kv_split_rank = split_size - 1;
      const std::vector<BlockTransferInfo> infos = {make_block_info(93)};
      {
        KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
        cache.get_k_cache().fill_(23 + split_size);
        cache.get_v_cache().fill_(47 + split_size);
        KVCacheStore writer;
        ASSERT_TRUE(init_store(writer, cache, config_));
        ASSERT_EQ(writer.batch_put(infos), 1U);
      }

      KVCache cache = make_attention_cache(/*host_blocks=*/3, /*width=*/8);
      KVCacheStore reader;
      ASSERT_TRUE(init_store(reader, cache, config_));
      ASSERT_EQ(reader.batch_get(infos), 1U);
      EXPECT_TRUE(torch::equal(
          cache.get_k_cache()[0],
          torch::full_like(cache.get_k_cache()[0], 23 + split_size)));
      EXPECT_TRUE(torch::equal(
          cache.get_v_cache()[0],
          torch::full_like(cache.get_v_cache()[0], 47 + split_size)));
    }
  }
}

TEST(KVCacheStoreTest, SplitPutElectsOneWriterPerStoreKey) {
  struct SplitWriterCase {
    int32_t cp_size;
    int32_t cp_rank;
    uint32_t tp_rank;
    uint32_t tp_size;
    int32_t full_domain_size;
    int32_t split_size;
    int32_t split_rank;
    bool sharded_owner;
    bool replicated_owner;
  };
  const std::vector<SplitWriterCase> writers = {
      // kv_split_size < cp_size: elect one CP replica per split and TP shard.
      {4, 0, 0, 2, 8, 2, 0, true, true},
      {4, 1, 0, 2, 8, 2, 0, false, false},
      {4, 2, 0, 2, 8, 2, 1, true, true},
      {4, 3, 0, 2, 8, 2, 1, false, false},
      {4, 0, 1, 2, 8, 2, 0, true, false},
      {4, 2, 1, 2, 8, 2, 1, true, false},
      // kv_split_size == cp_size: CP ranks are distinct, MLA TP ranks
      // replicate.
      {2, 1, 0, 2, 4, 2, 1, true, true},
      {2, 1, 1, 2, 4, 2, 1, true, false},
      // Full-domain DCP: split rank already distinguishes every CP/TP worker.
      {2, 0, 0, 2, 4, 4, 0, true, true},
      {2, 0, 1, 2, 4, 4, 1, true, true},
      {2, 1, 0, 2, 4, 4, 2, true, true},
      {2, 1, 1, 2, 4, 4, 3, true, true},
      // Legacy Store TP coordinates spanning CP still select the first TP
      // group per split, rather than publishing every unique sharded key.
      {4, 0, 0, 8, 8, 2, 0, true, true},
      {4, 0, 1, 8, 8, 2, 0, true, false},
      {4, 1, 2, 8, 8, 2, 0, false, false},
      {4, 2, 4, 8, 8, 2, 1, true, true},
      {4, 3, 6, 8, 8, 2, 1, false, false},
      // Full-domain remains unique when it equals CP and Store TP spans CP.
      {4, 3, 3, 4, 4, 4, 3, true, true},
  };
  for (bool enable_mla : {false, true}) {
    for (BlockType block_type : {BlockType::KV, BlockType::LINEAR}) {
      KVCache cache = block_type == BlockType::KV
                          ? make_attention_cache(/*host_blocks=*/2, /*width=*/8)
                          : make_linear_cache(/*host_blocks=*/2, /*width=*/8);
      for (const SplitWriterCase& writer : writers) {
        SCOPED_TRACE(
            ::testing::Message()
            << "enable_mla=" << enable_mla
            << " block_type=" << static_cast<int32_t>(block_type)
            << " cp_size=" << writer.cp_size << " cp_rank=" << writer.cp_rank
            << " tp_rank=" << writer.tp_rank << " tp_size=" << writer.tp_size
            << " full_domain_size=" << writer.full_domain_size << " split_size="
            << writer.split_size << " split_rank=" << writer.split_rank);
        KVCacheStoreInitConfig config = make_store_config(
            "split-writer-test", writer.tp_rank, writer.tp_size, enable_mla);
        config.cp_size = writer.cp_size;
        config.cp_rank = writer.cp_rank;
        config.kv_split_full_domain_size = writer.full_domain_size;
        config.kv_split_size = writer.split_size;
        config.kv_split_rank = writer.split_rank;
        HostCacheStoreIndex index;
        index[block_type].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                           "main",
                                                           &cache,
                                                           std::nullopt,
                                                           /*is_target=*/true});
        KVCacheStore store;
        KVCacheStoreTestPeer::initialize_index(
            &store, config, std::move(index));
        KVCacheStoreTestPeer::mark_initialized(&store);

        const std::vector<BlockTransferInfo> infos = {
            make_block_info(94, block_type)};
        ASSERT_TRUE(
            KVCacheStoreTestPeer::multi_buffer(store, infos).has_value());
        const bool tp_replicated = enable_mla && block_type == BlockType::KV;
        const bool linear_owner =
            writer.cp_rank == 0 &&
            (enable_mla || writer.split_size <= writer.cp_size ||
             writer.tp_rank == 0);
        const bool is_owner = block_type == BlockType::LINEAR
                                  ? linear_owner
                                  : (tp_replicated ? writer.replicated_owner
                                                   : writer.sharded_owner);
        // An owner reaches the missing backend and fails; a replica skips the
        // physical put and reports the logical block as already published.
        EXPECT_EQ(store.batch_put(infos), is_owner ? 0U : 1U);
      }
    }
  }
}

TEST(KVCacheStoreTest, PublishesTargetDraftAndLinearFromTheirOwnRankGroups) {
  struct OwnerCase {
    int32_t cp_size;
    uint32_t store_tp_size;
    int32_t domain_size;
    int32_t split_size;
    bool enable_mla;
    std::vector<int32_t> target_kv_owners;
    std::vector<int32_t> draft_linear_owners;
  };
  const std::vector<OwnerCase> cases = {
      {4, 2, 8, 2, true, {0, 4}, {0, 1}},
      {4, 2, 8, 2, false, {0, 1, 4, 5}, {0, 1}},
      {1, 2, 2, 2, true, {0, 1}, {0, 1}},
      // Non-MLA decode: two splits, each containing a complete TP2 group.
      {1, 4, 4, 2, false, {0, 1, 2, 3}, {0, 1}},
      {1, 4, 4, 1, true, {0}, {0, 1, 2, 3}},
      {1, 4, 4, 1, false, {0, 1, 2, 3}, {0, 1, 2, 3}},
  };
  for (const OwnerCase& owner_case : cases) {
    for (int32_t rank = 0; rank < owner_case.domain_size; ++rank) {
      SCOPED_TRACE(::testing::Message()
                   << "cp=" << owner_case.cp_size
                   << " split=" << owner_case.split_size
                   << " mla=" << owner_case.enable_mla << " rank=" << rank);
      KVCache target_kv = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
      KVCache draft_kv = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
      KVCache target_linear = make_linear_cache(/*host_blocks=*/2, /*width=*/8);
      KVCache draft_linear = make_linear_cache(/*host_blocks=*/2, /*width=*/4);
      HostCacheStoreIndex index;
      index[BlockType::KV] = {
          {0, "main", &target_kv, owner_case.enable_mla, true},
          {1, "draft", &draft_kv, false, false}};
      index[BlockType::LINEAR] = {
          {0, "main", &target_linear, owner_case.enable_mla, true},
          {1, "draft", &draft_linear, false, false}};
      const int32_t workers_per_cp =
          owner_case.domain_size / owner_case.cp_size;
      KVCacheStoreInitConfig config =
          make_store_config("owner-test",
                            rank % owner_case.store_tp_size,
                            owner_case.store_tp_size,
                            owner_case.enable_mla);
      config.cp_size = owner_case.cp_size;
      config.cp_rank = rank / workers_per_cp;
      config.kv_split_full_domain_size = owner_case.domain_size;
      config.kv_split_size = owner_case.split_size;
      config.kv_split_rank =
          rank / (owner_case.domain_size / owner_case.split_size);
      KVCacheStore store;
      KVCacheStoreTestPeer::initialize_index(&store, config, std::move(index));
      auto objects = std::make_shared<StoreObjects>();
      KVCacheStoreTestPeer::attach_transport(
          &store, std::make_unique<MemoryStoreTransport>(objects));
      const std::vector<BlockTransferInfo> infos = {
          make_block_info(96, BlockType::KV),
          make_block_info(97, BlockType::LINEAR)};
      ASSERT_EQ(store.batch_put(infos), 2U);
      const bool target_owner =
          std::find(owner_case.target_kv_owners.begin(),
                    owner_case.target_kv_owners.end(),
                    rank) != owner_case.target_kv_owners.end();
      const bool draft_linear_owner =
          std::find(owner_case.draft_linear_owners.begin(),
                    owner_case.draft_linear_owners.end(),
                    rank) != owner_case.draft_linear_owners.end();
      for (const BlockTransferInfo& info : infos) {
        const auto keys = KVCacheStoreTestPeer::build_keys(store, {info});
        EXPECT_EQ(objects->count(key_for_component(keys, "main")),
                  info.block_type == BlockType::KV ? target_owner
                                                   : draft_linear_owner);
        EXPECT_EQ(objects->count(key_for_component(keys, "draft")),
                  draft_linear_owner);
      }
    }
  }
}

TEST(KVCacheStoreTest, UnsplitPutSkipsOnlySameKeyReplicas) {
  for (bool enable_mla : {false, true}) {
    for (BlockType block_type : {BlockType::KV, BlockType::LINEAR}) {
      KVCache cache = block_type == BlockType::KV
                          ? make_attention_cache(/*host_blocks=*/2, /*width=*/8)
                          : make_linear_cache(/*host_blocks=*/2, /*width=*/8);
      for (int32_t cp_rank : {0, 1}) {
        for (uint32_t tp_rank : {0U, 1U}) {
          SCOPED_TRACE(::testing::Message()
                       << "enable_mla=" << enable_mla
                       << " block_type=" << static_cast<int32_t>(block_type)
                       << " cp_rank=" << cp_rank << " tp_rank=" << tp_rank);
          KVCacheStoreInitConfig config = make_store_config(
              "unsplit-writer-test", tp_rank, /*tp_size=*/2, enable_mla);
          config.cp_size = 2;
          config.cp_rank = cp_rank;
          config.kv_split_full_domain_size = 4;
          HostCacheStoreIndex index;
          index[block_type].emplace_back(
              HostCacheStoreEntry{/*cache_handle=*/0,
                                  "main",
                                  &cache,
                                  std::nullopt,
                                  /*is_target=*/true});
          KVCacheStore store;
          KVCacheStoreTestPeer::initialize_index(
              &store, config, std::move(index));
          KVCacheStoreTestPeer::mark_initialized(&store);
          const bool same_key_replica =
              cp_rank != 0 ||
              (enable_mla && block_type == BlockType::KV && tp_rank != 0);
          EXPECT_EQ(store.batch_put({make_block_info(95, block_type)}),
                    same_key_replica ? 1U : 0U);
        }
      }
    }
  }
}

TEST_F(KVCacheStoreIntegrationTest, RestoresBatchBlocksWithIndependentMisses) {
  KVCache cache = make_attention_cache(/*host_blocks=*/5, /*width=*/8);
  cache.get_k_cache()[0].fill_(7);
  cache.get_v_cache()[0].fill_(70);
  cache.get_k_cache()[1].fill_(9);
  cache.get_v_cache()[1].fill_(90);
  KVCacheStore store;
  ASSERT_TRUE(init_store(store, cache, config_));
  const std::vector<BlockTransferInfo> sources = {
      make_block_info(/*hash_value=*/94),
      make_block_info(
          /*hash_value=*/95, BlockType::KV, /*destination_block_id=*/1)};
  ASSERT_EQ(store.batch_put(sources), 2U);

  std::vector<BlockTransferInfo> destinations = {
      make_block_info(
          /*hash_value=*/97, BlockType::KV, /*destination_block_id=*/99),
      make_block_info(
          /*hash_value=*/94, BlockType::KV, /*destination_block_id=*/2),
      make_block_info(
          /*hash_value=*/96, BlockType::KV, /*destination_block_id=*/3),
      make_block_info(
          /*hash_value=*/95, BlockType::KV, /*destination_block_id=*/4)};
  Slice<BlockTransferInfo> destination_slice(destinations);
  EXPECT_EQ(store.batch_get_with_status(destination_slice),
            std::vector<uint8_t>({0, 1, 0, 1}));
  EXPECT_TRUE(torch::equal(cache.get_k_cache()[2], cache.get_k_cache()[0]));
  EXPECT_TRUE(torch::equal(cache.get_v_cache()[2], cache.get_v_cache()[0]));
  EXPECT_TRUE(torch::equal(cache.get_k_cache()[4], cache.get_k_cache()[1]));
  EXPECT_TRUE(torch::equal(cache.get_v_cache()[4], cache.get_v_cache()[1]));
  EXPECT_EQ(cache.get_k_cache()[3].count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(cache.get_v_cache()[3].count_nonzero().item<int64_t>(), 0);
}

TEST_F(KVCacheStoreIntegrationTest, ReadsValidBuffersDespiteInvalidNeighbors) {
  KVCache cache = make_attention_cache(/*host_blocks=*/3, /*width=*/8);
  cache.get_k_cache()[0].fill_(7);
  cache.get_v_cache()[0].fill_(70);
  KVCacheStore store;
  ASSERT_TRUE(init_store(store, cache, config_));
  const std::vector<BlockTransferInfo> sources = {make_block_info(101)};
  ASSERT_EQ(store.batch_put(sources), 1U);
  const std::string key =
      KVCacheStoreTestPeer::build_keys(store, sources).front().second;
  const auto buffer = KVCacheStoreTestPeer::multi_buffer(
      store, {make_block_info(101, BlockType::KV, /*destination_block_id=*/2)});
  ASSERT_TRUE(buffer.has_value());
  MultiBuffer mismatched = *buffer;
  mismatched.sizes.clear();
  MultiBuffer oversized = *buffer;
  oversized.sizes.front() =
      static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1;
  EXPECT_EQ(KVCacheStoreTestPeer::read_buffers(
                &store, {key, key, key}, {mismatched, *buffer, oversized}),
            std::vector<uint8_t>({0, 1, 0}));
  EXPECT_TRUE(torch::equal(cache.get_k_cache()[0], cache.get_k_cache()[2]));
  EXPECT_TRUE(torch::equal(cache.get_v_cache()[0], cache.get_v_cache()[2]));
  EXPECT_EQ(KVCacheStoreTestPeer::read_buffers(&store, {key}, {mismatched}),
            std::vector<uint8_t>({0}));
  EXPECT_TRUE(KVCacheStoreTestPeer::read_buffers(&store, {}, {}).empty());
}

TEST(KVCacheStoreDeathTest, RejectsTransportResultCountMismatch) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  KVCacheStoreTestPeer::initialize_index(
      &store, make_store_config(), std::move(index));
  auto objects = std::make_shared<StoreObjects>();
  auto transport = std::make_unique<MemoryStoreTransport>(objects);
  MemoryStoreTransport* transport_ptr = transport.get();
  KVCacheStoreTestPeer::attach_transport(&store, std::move(transport));
  std::vector<BlockTransferInfo> infos = {make_block_info(102)};
  Slice<BlockTransferInfo> slice(infos);
  for (size_t result_count : {0U, 2U}) {
    transport_ptr->result_count_ = result_count;
    EXPECT_DEATH(store.batch_put(slice), "results.size\\(\\)");
    EXPECT_DEATH(store.batch_get_with_status(slice), "results.size\\(\\)");
  }
}

TEST(KVCacheStoreDeathTest, RejectsInvalidSplitTopologyBeforeStoreSetup) {
  for (const auto& [size, rank] : std::vector<std::pair<int32_t, int32_t>>{
           {0, 0}, {-1, 0}, {4, -1}, {4, 4}}) {
    KVCacheStoreInitConfig config = make_store_config();
    config.kv_split_size = size;
    config.kv_split_rank = rank;
    EXPECT_DEATH(
        {
          KVCacheStore store;
          store.init(config, {});
        },
        "kv_split");
  }
  for (const auto& [size, rank] : std::vector<std::pair<int32_t, int32_t>>{
           {0, 0}, {-1, 0}, {2, -1}, {2, 2}}) {
    KVCacheStoreInitConfig config = make_store_config();
    config.cp_size = size;
    config.cp_rank = rank;
    EXPECT_DEATH(
        {
          KVCacheStore store;
          store.init(config, {});
        },
        "cp_");
  }
  KVCacheStoreInitConfig nonfactor = make_store_config();
  nonfactor.cp_size = 4;
  nonfactor.kv_split_full_domain_size = 8;
  nonfactor.kv_split_size = 3;
  EXPECT_DEATH(
      {
        KVCacheStore store;
        store.init(nonfactor, {});
      },
      "kv_split_size must divide cp_size");

  KVCacheStoreInitConfig overlapping_tp = make_store_config(
      "target-model", /*tp_rank=*/7, /*tp_size=*/8, /*enable_mla=*/true);
  overlapping_tp.cp_size = 4;
  overlapping_tp.cp_rank = 3;
  overlapping_tp.kv_split_full_domain_size = 8;
  overlapping_tp.kv_split_size = 8;
  overlapping_tp.kv_split_rank = 7;
  EXPECT_DEATH(
      {
        KVCacheStore store;
        store.init(overlapping_tp, {});
      },
      "requires Host caches");

  KVCacheStoreInitConfig invalid_tp_domain = make_store_config(
      "target-model", /*tp_rank=*/0, /*tp_size=*/4, /*enable_mla=*/true);
  invalid_tp_domain.cp_size = 4;
  invalid_tp_domain.kv_split_full_domain_size = 8;
  EXPECT_DEATH(
      {
        KVCacheStore store;
        store.init(invalid_tp_domain, {});
      },
      "Store TP ranks");
}

TEST(KVCacheStoreTest, TargetKeyDoesNotDependOnDraftRegistration) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);

  KVCacheStore target_only_store;
  HostCacheStoreIndex target_only_index;
  target_only_index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0,
                          "main",
                          &target_cache,
                          std::nullopt,
                          /*is_target=*/true});
  KVCacheStoreTestPeer::initialize_index(
      &target_only_store, make_store_config(), std::move(target_only_index));

  KVCacheStore speculative_store;
  HostCacheStoreIndex speculative_index;
  speculative_index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0,
                          "main",
                          &target_cache,
                          std::nullopt,
                          /*is_target=*/true});
  speculative_index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/1,
                          "spec_draft::mtp::draft-model",
                          &draft_cache,
                          std::nullopt,
                          /*is_target=*/false});
  KVCacheStoreTestPeer::initialize_index(
      &speculative_store, make_store_config(), std::move(speculative_index));

  const std::vector<BlockTransferInfo> block_info = {make_block_info(3)};
  const auto target_only_keys =
      KVCacheStoreTestPeer::build_keys(target_only_store, block_info);
  const auto speculative_keys =
      KVCacheStoreTestPeer::build_keys(speculative_store, block_info);

  ASSERT_EQ(target_only_keys.size(), 1U);
  ASSERT_EQ(speculative_keys.size(), 2U);
  EXPECT_EQ(target_only_keys.front().second,
            key_for_component(speculative_keys, "main"));
  EXPECT_NE(
      key_for_component(speculative_keys, "main"),
      key_for_component(speculative_keys, "spec_draft::mtp::draft-model"));
  EXPECT_EQ(target_only_keys.front().second.find("xllm-kv-v3:"), 0U);
}

TEST(KVCacheStoreTest, UsesPerCacheMlaLayoutForTargetAndDraft) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &target_cache,
                                                        /*enable_mla=*/true,
                                                        /*is_target=*/true});
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/1,
                          "spec_draft::dflash2::draft-model",
                          &draft_cache,
                          /*enable_mla=*/false,
                          /*is_target=*/false});
  KVCacheStoreTestPeer::initialize_index(&store,
                                         make_store_config("target-model",
                                                           /*tp_rank=*/1,
                                                           /*tp_size=*/8,
                                                           /*enable_mla=*/true),
                                         std::move(index));

  const auto keys = KVCacheStoreTestPeer::build_keys(
      store, {make_block_info(7, BlockType::KV)});
  const std::string target_key = key_for_component(keys, "main");
  const std::string draft_key =
      key_for_component(keys, "spec_draft::dflash2::draft-model");
  EXPECT_NE(target_key.find("mla:"), std::string::npos);
  EXPECT_NE(draft_key.find("8:1:"), std::string::npos);
  EXPECT_EQ(draft_key.find("mla:"), std::string::npos);
}

TEST(KVCacheStoreTest, DraftKeyDependsOnTargetDraftAndAlgorithm) {
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  const std::vector<BlockTransferInfo> block_info = {make_block_info(5)};

  const auto build_draft_key = [&draft_cache, &block_info](
                                   const std::string& target_model_id,
                                   const std::string& component) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/1,
                                                          component,
                                                          &draft_cache,
                                                          std::nullopt,
                                                          /*is_target=*/false});
    KVCacheStoreTestPeer::initialize_index(
        &store, make_store_config(target_model_id), std::move(index));
    return KVCacheStoreTestPeer::build_keys(store, block_info).front().second;
  };

  const std::string baseline =
      build_draft_key("target-a", "spec_draft::mtp::draft-a");
  EXPECT_NE(baseline, build_draft_key("target-b", "spec_draft::mtp::draft-a"));
  EXPECT_NE(baseline, build_draft_key("target-a", "spec_draft::mtp::draft-b"));
  EXPECT_NE(baseline,
            build_draft_key("target-a", "spec_draft::dspark::draft-a"));
  EXPECT_EQ(baseline, build_draft_key("target-a", "spec_draft::mtp::draft-a"));
}

// The CP index write mode changes which physical pages of an INDEX resource
// hold valid data while leaving every tensor shape identical, so the object
// keys must separate the modes: a replicated reader scoring a sharded
// writer's stale peer pages as valid is exactly the silent cross-mode hit
// the schema field exists to prevent.
TEST(KVCacheStoreTest, SchemaSeparatesCpIndexWriteModes) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  const std::vector<BlockTransferInfo> block_info = {make_block_info(9)};

  const auto build_key = [&cache, &block_info](int32_t cp_index_write_mode) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                          "main",
                                                          &cache,
                                                          std::nullopt,
                                                          /*is_target=*/true});
    KVCacheStoreTestPeer::initialize_index(
        &store,
        make_store_config(/*model_id=*/"target-model",
                          /*tp_rank=*/1,
                          /*tp_size=*/2,
                          /*enable_mla=*/false,
                          cp_index_write_mode),
        std::move(index));
    return KVCacheStoreTestPeer::build_keys(store, block_info).front().second;
  };

  const std::string replicated = build_key(kCpIndexWriteModeReplicated);
  EXPECT_NE(replicated, build_key(kCpIndexWriteModeSharded));
  EXPECT_EQ(replicated, build_key(kCpIndexWriteModeReplicated));
}

TEST(KVCacheStoreTest, SchemaExcludesHostCapacity) {
  KVCache small_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache large_cache = make_attention_cache(/*host_blocks=*/5, /*width=*/8);
  KVCache different_cache =
      make_attention_cache(/*host_blocks=*/2, /*width=*/16);
  const std::vector<BlockTransferInfo> block_info = {make_block_info(7)};

  const auto build_target_key = [&block_info](KVCache* cache) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{
        /*cache_handle=*/0, "main", cache, std::nullopt, /*is_target=*/true});
    KVCacheStoreTestPeer::initialize_index(
        &store, make_store_config(), std::move(index));
    return KVCacheStoreTestPeer::build_keys(store, block_info).front().second;
  };

  EXPECT_EQ(build_target_key(&small_cache), build_target_key(&large_cache));
  EXPECT_NE(build_target_key(&small_cache), build_target_key(&different_cache));
}

TEST(KVCacheStoreTest, CompressedKPoolRegistersIndexAndTailLayouts) {
  KVCache cache = make_compressed_kpool_cache(
      /*host_blocks=*/2, /*index_width=*/8, /*tail_blocks=*/3);
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  index[BlockType::LINEAR].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0,
                          "main",
                          &cache,
                          std::nullopt,
                          /*is_target=*/true});
  KVCacheStore store;
  KVCacheStoreTestPeer::initialize_index(&store,
                                         make_store_config("compressed-kpool",
                                                           /*tp_rank=*/0,
                                                           /*tp_size=*/2,
                                                           /*enable_mla=*/true),
                                         std::move(index));

  const std::vector<BlockTransferInfo> kv_info = {
      make_block_info(81, BlockType::KV)};
  const std::vector<BlockTransferInfo> tail_info = {
      make_block_info(81, BlockType::LINEAR)};
  const std::optional<MultiBuffer> kv_buffer =
      KVCacheStoreTestPeer::multi_buffer(store, kv_info);
  const std::optional<MultiBuffer> tail_buffer =
      KVCacheStoreTestPeer::multi_buffer(store, tail_info);
  ASSERT_TRUE(kv_buffer.has_value());
  ASSERT_TRUE(tail_buffer.has_value());
  EXPECT_EQ(kv_buffer->sizes.size(), 2U);
  EXPECT_EQ(tail_buffer->sizes.size(), 1U);
  EXPECT_NE(KVCacheStoreTestPeer::build_keys(store, kv_info).front().second,
            KVCacheStoreTestPeer::build_keys(store, tail_info).front().second);

  KVCache different_index = make_compressed_kpool_cache(
      /*host_blocks=*/2, /*index_width=*/16, /*tail_blocks=*/3);
  HostCacheStoreIndex different_index_layout;
  different_index_layout[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0,
                          "main",
                          &different_index,
                          std::nullopt,
                          /*is_target=*/true});
  KVCacheStore different_store;
  KVCacheStoreTestPeer::initialize_index(&different_store,
                                         make_store_config("compressed-kpool",
                                                           /*tp_rank=*/0,
                                                           /*tp_size=*/2,
                                                           /*enable_mla=*/true),
                                         std::move(different_index_layout));
  EXPECT_NE(KVCacheStoreTestPeer::build_keys(store, kv_info).front().second,
            KVCacheStoreTestPeer::build_keys(different_store, kv_info)
                .front()
                .second);
}

TEST(KVCacheStoreTest, MlaKeyExcludesTensorParallelTopology) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  const std::vector<BlockTransferInfo> block_info = {make_block_info(8)};

  const auto build_key = [&cache, &block_info](uint32_t tp_rank,
                                               uint32_t tp_size,
                                               bool enable_mla,
                                               int32_t split_size = 1,
                                               int32_t split_rank = 0) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                          "main",
                                                          &cache,
                                                          std::nullopt,
                                                          /*is_target=*/true});
    KVCacheStoreInitConfig config =
        make_store_config("target-model", tp_rank, tp_size, enable_mla);
    config.cp_size = split_size;
    config.cp_rank = split_rank;
    config.kv_split_full_domain_size =
        split_size * static_cast<int32_t>(tp_size);
    config.kv_split_size = split_size;
    config.kv_split_rank = split_rank;
    KVCacheStoreTestPeer::initialize_index(&store, config, std::move(index));
    return KVCacheStoreTestPeer::build_keys(store, block_info).front().second;
  };

  const std::string mla_key = build_key(/*tp_rank=*/0,
                                        /*tp_size=*/1,
                                        /*enable_mla=*/true);
  EXPECT_EQ(mla_key.find("xllm-kv-v3:12:target-model:4:main:mla:1:0:0:"), 0U);
  for (bool enable_mla : {false, true}) {
    const std::string split_key = build_key(/*tp_rank=*/0,
                                            /*tp_size=*/8,
                                            enable_mla,
                                            /*split_size=*/4,
                                            /*split_rank=*/0);
    EXPECT_NE(split_key,
              build_key(/*tp_rank=*/0,
                        /*tp_size=*/8,
                        enable_mla,
                        /*split_size=*/4,
                        /*split_rank=*/1));
    EXPECT_NE(split_key,
              build_key(/*tp_rank=*/0,
                        /*tp_size=*/8,
                        enable_mla,
                        /*split_size=*/8,
                        /*split_rank=*/0));
  }
  EXPECT_EQ(build_key(/*tp_rank=*/0,
                      /*tp_size=*/8,
                      /*enable_mla=*/true,
                      /*split_size=*/4,
                      /*split_rank=*/2),
            build_key(/*tp_rank=*/7,
                      /*tp_size=*/8,
                      /*enable_mla=*/true,
                      /*split_size=*/4,
                      /*split_rank=*/2));
  EXPECT_EQ(mla_key,
            build_key(/*tp_rank=*/7,
                      /*tp_size=*/8,
                      /*enable_mla=*/true));
  EXPECT_NE(mla_key,
            build_key(/*tp_rank=*/0,
                      /*tp_size=*/1,
                      /*enable_mla=*/false));
  EXPECT_NE(build_key(/*tp_rank=*/0,
                      /*tp_size=*/8,
                      /*enable_mla=*/false),
            build_key(/*tp_rank=*/7,
                      /*tp_size=*/8,
                      /*enable_mla=*/false));
}

TEST(KVCacheStoreTest, NonSplitCachesUseCanonicalSplitZeroKeys) {
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  KVCache linear_cache = make_linear_cache(/*host_blocks=*/2, /*width=*/4);
  const std::vector<BlockTransferInfo> draft_info = {
      make_block_info(11, BlockType::KV)};
  const std::vector<BlockTransferInfo> linear_info = {
      make_block_info(11, BlockType::LINEAR)};

  const auto build_key = [&](BlockType block_type,
                             bool is_target,
                             uint32_t tp_rank,
                             int32_t split_rank) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    KVCache* cache = block_type == BlockType::KV ? &draft_cache : &linear_cache;
    index[block_type].emplace_back(
        HostCacheStoreEntry{/*cache_handle=*/0,
                            block_type == BlockType::KV ? "draft" : "main",
                            cache,
                            /*enable_mla=*/false,
                            is_target});
    KVCacheStoreInitConfig config = make_store_config("non-split",
                                                      tp_rank,
                                                      /*tp_size=*/2,
                                                      /*enable_mla=*/false);
    config.cp_size = 1;
    config.cp_rank = 0;
    config.kv_split_full_domain_size = 4;
    config.kv_split_size = 2;
    config.kv_split_rank = split_rank;
    KVCacheStoreTestPeer::initialize_index(&store, config, std::move(index));
    return KVCacheStoreTestPeer::build_keys(
               store, block_type == BlockType::KV ? draft_info : linear_info)
        .front()
        .second;
  };

  EXPECT_EQ(build_key(BlockType::KV, /*is_target=*/false, /*tp_rank=*/0, 0),
            build_key(BlockType::KV, /*is_target=*/false, /*tp_rank=*/0, 1));
  EXPECT_NE(build_key(BlockType::KV, /*is_target=*/false, /*tp_rank=*/0, 0),
            build_key(BlockType::KV, /*is_target=*/false, /*tp_rank=*/1, 0));
  EXPECT_EQ(build_key(BlockType::LINEAR, /*is_target=*/true, /*tp_rank=*/0, 0),
            build_key(BlockType::LINEAR, /*is_target=*/true, /*tp_rank=*/0, 1));
  EXPECT_NE(build_key(BlockType::LINEAR, /*is_target=*/true, /*tp_rank=*/0, 0),
            build_key(BlockType::LINEAR, /*is_target=*/true, /*tp_rank=*/1, 0));
}

TEST(KVCacheStoreTest, MlaNonzeroRankSkipsRemotePut) {
  KVCacheStore store;
  HostCacheStoreIndex index;
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  KVCacheStoreTestPeer::initialize_index(&store,
                                         make_store_config("target-model",
                                                           /*tp_rank=*/3,
                                                           /*tp_size=*/8,
                                                           /*enable_mla=*/true),
                                         std::move(index));
  KVCacheStoreTestPeer::mark_initialized(&store);
  const std::vector<BlockTransferInfo> block_info = {
      make_block_info(10, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(11, BlockType::KV, /*destination_block_id=*/1)};

  // A null client makes any accidental remote Store access fail loudly.
  EXPECT_EQ(store.batch_put(block_info), block_info.size());
}

TEST(KVCacheStoreTest, SelectsOnlyExplicitRdmaDevices) {
  KVCacheStoreInitConfig config;
  config.protocol = "rdma";
  config.rdma_devices = "mlx5_0,mlx5_1";
  EXPECT_EQ(KVCacheStoreTestPeer::store_device_names(config),
            std::optional<std::string>("mlx5_0,mlx5_1"));

  config.rdma_devices.clear();
  ScopedEnvVar device_names("DEVICE_NAMES");
  ASSERT_TRUE(device_names.set("legacy_hca"));
  EXPECT_EQ(KVCacheStoreTestPeer::store_device_names(config), std::nullopt);
  EXPECT_EQ(config.protocol, "rdma");
}

TEST(KVCacheStoreTest, TcpIgnoresRdmaDevices) {
  KVCacheStoreInitConfig config;
  config.protocol = "tcp";
  config.rdma_devices = "mlx5_0";

  EXPECT_EQ(KVCacheStoreTestPeer::store_device_names(config), std::nullopt);
}

TEST(KVCacheStoreDeathTest, RejectsDuplicateKeyComponentsPerBlockType) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);

  EXPECT_DEATH(
      {
        KVCacheStore store;
        HostCacheStoreIndex index;
        index[BlockType::KV].emplace_back(
            HostCacheStoreEntry{/*cache_handle=*/0,
                                "main",
                                &target_cache,
                                std::nullopt,
                                /*is_target=*/true});
        index[BlockType::KV].emplace_back(
            HostCacheStoreEntry{/*cache_handle=*/1,
                                "main",
                                &draft_cache,
                                std::nullopt,
                                /*is_target=*/false});
        KVCacheStoreTestPeer::initialize_index(
            &store, make_store_config(), std::move(index));
      },
      "Duplicate KVCacheStore key component");
}

TEST(KVCacheStoreTest, AggregatesPhysicalResultsPerLogicalBlock) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &target_cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/1,
                          "spec_draft::mtp::draft-model",
                          &draft_cache,
                          std::nullopt,
                          /*is_target=*/false});
  KVCacheStoreTestPeer::initialize_index(
      &store, make_store_config(), std::move(index));

  const std::vector<BlockTransferInfo> block_info = {
      make_block_info(9, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(10, BlockType::KV, /*destination_block_id=*/1)};
  EXPECT_EQ(KVCacheStoreTestPeer::physical_request_count(store, block_info),
            4U);
  EXPECT_EQ(KVCacheStoreTestPeer::aggregate(
                store, block_info, std::vector<uint8_t>{1, 1, 1, 0}),
            std::vector<uint8_t>({1, 0}));
}

TEST(KVCacheStoreTest, ObjectPresenceRequiresEveryComponentObject) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &target_cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/1,
                          "spec_draft::mtp::draft-model",
                          &draft_cache,
                          std::nullopt,
                          /*is_target=*/false});
  KVCacheStoreTestPeer::initialize_index(
      &store, make_store_config(), std::move(index));

  const std::vector<BlockTransferInfo> block_info = {
      make_block_info(11, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(12, BlockType::KV, /*destination_block_id=*/1),
      make_block_info(13, BlockType::KV, /*destination_block_id=*/0)};
  // Physical objects are ordered per logical block, then per component.
  const std::vector<MooncakeReplicaTier> tiers = {MooncakeReplicaTier::MEMORY,
                                                  MooncakeReplicaTier::DISK,
                                                  MooncakeReplicaTier::MEMORY,
                                                  MooncakeReplicaTier::MISSING,
                                                  MooncakeReplicaTier::MISSING,
                                                  MooncakeReplicaTier::MISSING};
  const std::vector<uint8_t> present =
      KVCacheStoreTestPeer::present_objects(tiers);

  EXPECT_EQ(present, std::vector<uint8_t>({1, 1, 1, 0, 0, 0}));
  EXPECT_EQ(KVCacheStoreTestPeer::aggregate(store, block_info, present),
            std::vector<uint8_t>({1, 0, 0}));
}

TEST(KVCacheStoreTest, RecordsGetVolumePerReplicaTier) {
  StoreGetStats stats;
  KVCacheStoreTestPeer::record_get(
      MooncakeReplicaTier::MEMORY, /*bytes=*/100, &stats);
  KVCacheStoreTestPeer::record_get(
      MooncakeReplicaTier::MEMORY, /*bytes=*/20, &stats);
  KVCacheStoreTestPeer::record_get(
      MooncakeReplicaTier::DISK, /*bytes=*/300, &stats);
  // An unresolved tier still counts towards the total read volume.
  KVCacheStoreTestPeer::record_get(
      MooncakeReplicaTier::MISSING, /*bytes=*/7, &stats);

  EXPECT_EQ(stats.read_bytes, 427U);
  EXPECT_EQ(stats.memory_objects, 2U);
  EXPECT_EQ(stats.memory_bytes, 120U);
  EXPECT_EQ(stats.disk_objects, 1U);
  EXPECT_EQ(stats.disk_bytes, 300U);
  EXPECT_EQ(stats.tier_query_us, 0U);
}

TEST(KVCacheStoreTest, DeduplicatesPhysicalKeysAndRespectsBlockTypeEntries) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  KVCache linear_cache = make_linear_cache(/*host_blocks=*/2, /*width=*/6);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &target_cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/1,
                          "spec_draft::mtp::draft-model",
                          &draft_cache,
                          std::nullopt,
                          /*is_target=*/false});
  index[BlockType::LINEAR].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0,
                          "main",
                          &linear_cache,
                          std::nullopt,
                          /*is_target=*/true});
  KVCacheStoreTestPeer::initialize_index(
      &store, make_store_config(), std::move(index));

  const std::vector<BlockTransferInfo> duplicate_kv = {
      make_block_info(11, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(11, BlockType::KV, /*destination_block_id=*/1)};
  EXPECT_EQ(KVCacheStoreTestPeer::physical_request_count(store, duplicate_kv),
            4U);
  EXPECT_EQ(KVCacheStoreTestPeer::unique_key_count(store, duplicate_kv), 2U);

  const std::vector<BlockTransferInfo> linear = {
      make_block_info(12, BlockType::LINEAR)};
  EXPECT_EQ(KVCacheStoreTestPeer::physical_request_count(store, linear), 1U);
}

TEST(KVCacheStoreTest, BuildsMultiBufferForRequestedHostBlock) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/3);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  KVCacheStoreTestPeer::initialize_index(
      &store, make_store_config(), std::move(index));

  const std::vector<BlockTransferInfo> block_info = {
      make_block_info(12, BlockType::KV, /*destination_block_id=*/1)};
  const std::optional<MultiBuffer> buffer =
      KVCacheStoreTestPeer::multi_buffer(store, block_info);
  ASSERT_TRUE(buffer.has_value());
  ASSERT_EQ(buffer->addresses.size(), 2U);
  ASSERT_EQ(buffer->sizes.size(), 2U);
  EXPECT_EQ(buffer->sizes[0], 6U * sizeof(float));
  EXPECT_EQ(buffer->sizes[1], 6U * sizeof(float));

  const BlockTypeTensorMap tensors =
      cache.get_block_type_tensors(BlockType::KV);
  auto tensor_it = tensors.begin();
  EXPECT_EQ(buffer->addresses[0], tensor_it->second[1].data_ptr());
  ++tensor_it;
  EXPECT_EQ(buffer->addresses[1], tensor_it->second[1].data_ptr());
}

TEST(KVCacheStoreTest, FailsDuplicateGetKeysWithoutStoreAccess) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  KVCacheStoreTestPeer::initialize_index(
      &store, make_store_config(), std::move(index));
  KVCacheStoreTestPeer::mark_initialized(&store);
  std::vector<BlockTransferInfo> block_info = {
      make_block_info(13, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(13, BlockType::KV, /*destination_block_id=*/1)};
  Slice<BlockTransferInfo> slice(block_info);

  EXPECT_EQ(store.batch_get_with_status(slice), std::vector<uint8_t>({0, 0}));
}

TEST(MooncakeStoreBackendTest, ValidatesGetBufferCountsAndByteLimits) {
  int32_t value = 0;
  const size_t max_bytes =
      static_cast<size_t>(std::numeric_limits<int32_t>::max());
  EXPECT_EQ(MooncakeStoreBackendTestPeer::get_bytes({{}, {}}), 0);
  EXPECT_FALSE(
      MooncakeStoreBackendTestPeer::get_bytes({{&value}, {}}).has_value());
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::get_bytes({{}, {4}}).has_value());
  EXPECT_EQ(MooncakeStoreBackendTestPeer::get_bytes({{&value}, {max_bytes}}),
            std::numeric_limits<int32_t>::max());
  EXPECT_FALSE(
      MooncakeStoreBackendTestPeer::get_bytes({{&value}, {max_bytes + 1}})
          .has_value());
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::get_bytes(
                   {{&value, &value}, {max_bytes, 1}})
                   .has_value());
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::get_bytes(
                   {{&value}, {std::numeric_limits<size_t>::max()}})
                   .has_value());
  EXPECT_EQ(MooncakeStoreBackendTestPeer::get_bytes({{&value, &value}, {4, 8}}),
            12);
}

TEST(MooncakeStoreBackendTest, MapsExactGetBytesInKeyOrder) {
  EXPECT_TRUE(MooncakeStoreBackendTestPeer::get_statuses({}, {}).empty());
  EXPECT_EQ(MooncakeStoreBackendTestPeer::get_statuses({32}, {32}),
            std::vector<uint8_t>({1}));
  EXPECT_EQ(MooncakeStoreBackendTestPeer::get_statuses({32}, {}),
            std::vector<uint8_t>({0}));
  EXPECT_EQ(MooncakeStoreBackendTestPeer::get_statuses({32}, {32, 32}),
            std::vector<uint8_t>({1}));
  EXPECT_EQ(MooncakeStoreBackendTestPeer::get_statuses({32, 32, 32, 32, 0},
                                                       {32, -1, 31, 33, 0}),
            std::vector<uint8_t>({1, 0, 0, 0, 1}));
  EXPECT_EQ(MooncakeStoreBackendTestPeer::get_statuses({32, 64}, {32}),
            std::vector<uint8_t>({1, 0}));
  EXPECT_EQ(MooncakeStoreBackendTestPeer::get_statuses(
                {std::numeric_limits<int32_t>::max()},
                {std::numeric_limits<int32_t>::max()}),
            std::vector<uint8_t>({1}));
}

mooncake::Replica::Descriptor make_replica(ReplicaKind kind,
                                           mooncake::ReplicaStatus status) {
  mooncake::Replica::Descriptor replica;
  replica.id = 0;
  replica.status = status;
  switch (kind) {
    case ReplicaKind::MEMORY:
      replica.descriptor_variant = mooncake::MemoryDescriptor{};
      break;
    case ReplicaKind::NOF:
      replica.descriptor_variant = mooncake::NoFDescriptor{};
      break;
    case ReplicaKind::DISK:
      replica.descriptor_variant = mooncake::DiskDescriptor{};
      break;
    case ReplicaKind::LOCAL_DISK:
      replica.descriptor_variant = mooncake::LocalDiskDescriptor{};
      break;
  }
  return replica;
}

TEST(MooncakeStoreBackendTest, ReplicaTierFollowsCompleteReadPreference) {
  constexpr mooncake::ReplicaStatus kComplete =
      mooncake::ReplicaStatus::COMPLETE;
  constexpr mooncake::ReplicaStatus kProcessing =
      mooncake::ReplicaStatus::PROCESSING;
  const struct {
    const char* name;
    std::vector<mooncake::Replica::Descriptor> replicas;
    MooncakeReplicaTier expected;
  } cases[] = {
      {"no replica", {}, MooncakeReplicaTier::MISSING},
      {"memory",
       {make_replica(ReplicaKind::MEMORY, kComplete)},
       MooncakeReplicaTier::MEMORY},
      {"nof",
       {make_replica(ReplicaKind::NOF, kComplete)},
       MooncakeReplicaTier::DISK},
      {"disk",
       {make_replica(ReplicaKind::DISK, kComplete)},
       MooncakeReplicaTier::DISK},
      {"local disk",
       {make_replica(ReplicaKind::LOCAL_DISK, kComplete)},
       MooncakeReplicaTier::DISK},
      {"memory listed after disk",
       {make_replica(ReplicaKind::DISK, kComplete),
        make_replica(ReplicaKind::MEMORY, kComplete)},
       MooncakeReplicaTier::MEMORY},
      {"memory still being written",
       {make_replica(ReplicaKind::MEMORY, kProcessing),
        make_replica(ReplicaKind::DISK, kComplete)},
       MooncakeReplicaTier::DISK},
      {"only incomplete replicas",
       {make_replica(ReplicaKind::MEMORY, kProcessing),
        make_replica(ReplicaKind::DISK, mooncake::ReplicaStatus::FAILED)},
       MooncakeReplicaTier::MISSING},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    EXPECT_EQ(MooncakeStoreBackendTestPeer::replica_tier(test_case.replicas),
              test_case.expected);
  }
}

TEST(MooncakeStoreBackendTest, TreatsObjectAlreadyExistsAsPutSuccess) {
  EXPECT_TRUE(MooncakeStoreBackendTestPeer::put_succeeded(/*result=*/0));
  EXPECT_TRUE(MooncakeStoreBackendTestPeer::put_succeeded(
      static_cast<int>(mooncake::ErrorCode::OBJECT_ALREADY_EXISTS)));
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::put_succeeded(/*result=*/-1));
}

TEST(MooncakeStoreBackendTest, ValidatesRegistrationRangesBeforeSetup) {
  const auto valid_ranges = MooncakeStoreBackendTestPeer::unique_ranges(
      {{reinterpret_cast<void*>(0x1000), 64},
       {reinterpret_cast<void*>(0x1000), 64},
       {reinterpret_cast<void*>(0x2000), 32}});
  ASSERT_TRUE(valid_ranges.has_value());
  ASSERT_EQ(valid_ranges->size(), 2U);
  EXPECT_EQ((*valid_ranges)[0].address, reinterpret_cast<void*>(0x1000));
  EXPECT_EQ((*valid_ranges)[1].address, reinterpret_cast<void*>(0x2000));

  EXPECT_FALSE(MooncakeStoreBackendTestPeer::unique_ranges(
                   {{reinterpret_cast<void*>(0x1000), 64},
                    {reinterpret_cast<void*>(0x1000), 32}})
                   .has_value());
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::unique_ranges(
                   {{reinterpret_cast<void*>(0x1000), 64},
                    {reinterpret_cast<void*>(0x1020), 64}})
                   .has_value());
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::unique_ranges(
                   {{reinterpret_cast<void*>(UINTPTR_MAX - 7), 8}})
                   .has_value());
}

TEST(KVCacheStoreTest, RestoresRoleBytesAndRequiresEveryRegisteredShard) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  KVCache target(IndexedKVCacheTensors{
      KVCacheTensors{torch::zeros({3, 2, 8}, options), torch::Tensor()},
      torch::zeros({3, 2, 4}, options.dtype(torch::kInt8)),
      torch::zeros({3, 2}, options)});
  KVCache draft = make_attention_cache(/*host_blocks=*/3, /*width=*/4);
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{
      /*cache_handle=*/0, "main", &target, std::nullopt, /*is_target=*/true});
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/1,
                                                        "draft",
                                                        &draft,
                                                        std::nullopt,
                                                        /*is_target=*/false});
  KVCacheStore store;
  KVCacheStoreTestPeer::initialize_index(
      &store, make_store_config(), std::move(index));
  auto objects = std::make_shared<StoreObjects>();
  auto transport = std::make_unique<MemoryStoreTransport>(objects);
  MemoryStoreTransport* transport_ptr = transport.get();
  KVCacheStoreTestPeer::attach_transport(&store, std::move(transport));

  const std::vector<KVCache*> caches = {&target, &draft};
  for (size_t cache_index = 0; cache_index < caches.size(); ++cache_index) {
    for (const auto& [role, tensor] :
         caches[cache_index]->get_block_type_tensors(BlockType::KV)) {
      tensor[0].fill_(1 + cache_index * 10 + static_cast<size_t>(role));
    }
  }
  const std::vector<BlockTransferInfo> put_info = {
      make_block_info(/*hash_value=*/44)};
  EXPECT_EQ(store.batch_put(put_info), 1U);
  EXPECT_EQ(objects->size(), 2U);
  std::vector<BlockTransferInfo> get_info = {make_block_info(
      /*hash_value=*/44, BlockType::KV, /*destination_block_id=*/2)};
  Slice<BlockTransferInfo> get_slice(get_info);
  EXPECT_EQ(store.batch_get_with_status(get_slice), std::vector<uint8_t>({1}));
  EXPECT_EQ(store.batch_exist(get_slice), std::vector<uint8_t>({1}));
  StoreGetStats stats;
  EXPECT_EQ(store.batch_get_with_status(get_slice, &stats),
            std::vector<uint8_t>({1}));
  EXPECT_EQ(stats.memory_objects, 2U);
  EXPECT_GT(stats.read_bytes, 0U);
  EXPECT_EQ(transport_ptr->batch_get_calls_, 2);
  EXPECT_EQ(transport_ptr->last_get_size_, 2u);
  for (KVCache* cache : caches) {
    for (const auto& [role, tensor] :
         cache->get_block_type_tensors(BlockType::KV)) {
      EXPECT_TRUE(torch::equal(tensor[0], tensor[2]));
      tensor[2].zero_();
    }
  }

  const auto keys = KVCacheStoreTestPeer::build_keys(store, put_info);
  const std::string draft_key = key_for_component(keys, "draft");
  transport_ptr->failed_gets_.emplace(draft_key);
  // Both objects exist, but a failed actual GET must remain a logical miss.
  EXPECT_EQ(store.batch_get_with_status(get_slice), std::vector<uint8_t>({0}));
  transport_ptr->failed_gets_.clear();
  objects->erase(draft_key);
  target.get_k_cache()[2].zero_();
  EXPECT_EQ(store.batch_get_with_status(get_slice), std::vector<uint8_t>({0}));
  // A partial backend GET may write one shard, but the logical block remains
  // a miss and cannot be published until all registered shards succeed.
  EXPECT_TRUE(torch::equal(target.get_k_cache()[0], target.get_k_cache()[2]));
  EXPECT_EQ(draft.get_k_cache()[2].count_nonzero().item<int64_t>(), 0);
  transport_ptr->failed_puts_.emplace(draft_key);
  EXPECT_EQ(store.batch_put(put_info), 0U);
  EXPECT_EQ(store.batch_exist(get_slice), std::vector<uint8_t>({0}));
  auto replacement = std::make_unique<MemoryStoreTransport>(objects);
  KVCacheStoreTestPeer::attach_transport(&store, std::move(replacement));
  EXPECT_EQ(store.batch_exist(get_slice), std::vector<uint8_t>({0}));
  EXPECT_EQ(store.batch_put(put_info), 1U);
  EXPECT_EQ(store.batch_exist(get_slice), std::vector<uint8_t>({1}));
  EXPECT_EQ(store.batch_get_with_status(get_slice), std::vector<uint8_t>({1}));
}

}  // namespace
}  // namespace xllm
