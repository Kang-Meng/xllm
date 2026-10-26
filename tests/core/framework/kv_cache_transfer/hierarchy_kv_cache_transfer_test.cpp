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

#include "framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/config/scheduler_config.h"
#include "framework/kv_cache/kv_cache_capacity.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/model/model_args.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace {

class V41HierarchyTransferTest : public ::testing::TestWithParam<bool> {};

TEST_P(V41HierarchyTransferTest, TokenSourceLayersRoundTripAllRoles) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "A healthy accelerator is required for hierarchy KV transfer";
  }
  const char* master = std::getenv("XLLM_TEST_STORE_MASTER");
  if (GetParam() && master == nullptr) {
    GTEST_SKIP()
        << "Set XLLM_TEST_STORE_MASTER to a Mooncake master with an active "
           "TCP storage node; host-only coverage is the /0 case";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  KVCacheCapacity capacity;
  capacity.block_size(128).swa_count(4).c2_count(4).c1_count(8);
  ModelArgs args;
  args.model_type("deepseek_v41");
  const KVCacheShape shape(capacity, args, /*world_size=*/1);
  KVCacheCreateOptions options;
  options.device(device.unwrap())
      .dtype(torch::kBFloat16)
      .num_layers(6)
      .model_type("deepseek_v41")
      .block_size(128)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({0, 0, 2, 2, 1, 1})
      .kv_source_layer_ids({2, 4});
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);
  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.layers(6)
      .tp_rank(0)
      .tp_size(1)
      .host_blocks_factor(2.0)
      .layers_wise_copy_batchs(2);
  if (GetParam()) {
    transfer_options.enable_kvcache_store(true)
        .store_protocol("tcp")
        .store_master_server_address(master)
        .store_metadata_server("P2PHANDSHAKE")
        .store_namespace(
            "v41-hierarchy-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
  }
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  HierarchyKVCacheTransfer transfer(transfer_options,
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &caches,
                                    shape,
                                    options);
  ASSERT_TRUE(transfer.supports_block_type(BlockType::C1));
  ASSERT_TRUE(transfer.supports_block_type(BlockType::C2));
  EXPECT_FALSE(transfer.supports_block_type(BlockType::C4));
  for (const BlockType type : {BlockType::SWA, BlockType::C2, BlockType::C1}) {
    for (KVCache& cache : caches) {
      for (const auto& [role, tensor] : cache.get_block_type_tensors(type)) {
        tensor[1].fill_(static_cast<int32_t>(role) + 1);
        tensor[2].zero_();
      }
    }
    ASSERT_EQ(device.synchronize_default_stream(), 0);
    std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> hash{};
    hash.fill(static_cast<uint8_t>(type));
    BlockTransferInfo offload(
        /*src_id=*/1, /*dst_id=*/0, hash.data(), TransferType::D2H2G, type);
    ASSERT_EQ(transfer.transfer_kv_blocks(/*batch_id=*/81, {offload}), 1u);
    if (GetParam()) {
      std::vector<BlockTransferInfo> infos;
      infos.emplace_back(-1, 1, hash.data(), TransferType::G2H, type);
      Slice<BlockTransferInfo> slice(infos);
      // Read into a different Host block; H2D cannot reuse the D2H buffer.
      EXPECT_EQ(transfer.prefetch_kv_blocks(slice, /*stats=*/nullptr),
                (std::vector<uint8_t>{1}));
    }
    BlockTransferInfo load(/*src_id=*/GetParam() ? 1 : 0,
                           /*dst_id=*/2,
                           hash.data(),
                           TransferType::H2D,
                           type);
    ASSERT_EQ(transfer.transfer_kv_blocks(/*batch_id=*/81, {load}), 1u);
    ModelInputParams params;
    params.meta.batch_id = 81;
    params.meta.requires_host_restore = true;
    transfer.set_layer_synchronizer(params);
    for (int32_t layer = 0; layer < 6; ++layer) {
      ASSERT_TRUE(params.synchronize_layer(layer));
    }
    for (KVCache& cache : caches) {
      for (const auto& [role, tensor] : cache.get_block_type_tensors(type)) {
        EXPECT_TRUE(torch::equal(tensor[1], tensor[2]))
            << static_cast<int32_t>(role);
      }
    }
  }
}

TEST_P(V41HierarchyTransferTest,
       DSparkRestoresTargetAndDraftFromOneCheckpoint) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator is required for hierarchy KV transfer";
  }
  const char* master = std::getenv("XLLM_TEST_STORE_MASTER");
  if (GetParam() && master == nullptr) {
    GTEST_SKIP()
        << "Set XLLM_TEST_STORE_MASTER to an active TCP Mooncake Store";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  for (const int64_t window : {128, 512}) {
    for (const bool has_c1 : {false, true}) {
      SCOPED_TRACE(window);
      SCOPED_TRACE(has_c1);
      KVCacheCapacity capacity;
      capacity.block_size(128).swa_count(4).c2_count(4).c1_count(has_c1 ? 8
                                                                        : 0);
      ModelArgs args;
      args.model_type("deepseek_v41");
      const KVCacheShape shape(capacity, args, /*world_size=*/1);
      KVCacheCreateOptions target_options;
      target_options.device(device.unwrap())
          .dtype(torch::kBFloat16)
          .num_layers(6)
          .model_type("deepseek_v41")
          .block_size(128)
          .head_dim(16)
          .index_head_dim(8)
          .window_size(window)
          .compress_ratios(has_c1 ? std::vector<int32_t>{0, 0, 2, 2, 1, 1}
                                  : std::vector<int32_t>{0, 0, 2, 2, 0, 0})
          .kv_source_layer_ids(has_c1 ? std::vector<int32_t>{2, 4}
                                      : std::vector<int32_t>{2});
      KVCacheCreateOptions draft_options = target_options;
      draft_options.model_type("deepseek_v41_dspark")
          .num_layers(3)
          .compress_ratios({0, 0, 0})
          .kv_source_layer_ids({});
      std::vector<KVCache> target_caches;
      std::vector<KVCache> draft_caches;
      allocate_kv_caches(target_caches, shape, target_options);
      allocate_kv_caches(draft_caches, shape, draft_options);
      HierarchyKVCacheTransfer::Options transfer_options;
      transfer_options.layers(6)
          .tp_rank(0)
          .tp_size(1)
          .host_blocks_factor(2.0)
          .layers_wise_copy_batchs(2);
      if (GetParam()) {
        transfer_options.enable_kvcache_store(true)
            .store_protocol("tcp")
            .store_master_server_address(master)
            .store_metadata_server("P2PHANDSHAKE")
            .store_namespace("v41-dspark-" +
                             std::to_string(std::chrono::steady_clock::now()
                                                .time_since_epoch()
                                                .count()));
      }
      auto make_transfer = [&](const std::string& draft_component) {
        auto transfer = std::make_unique<HierarchyKVCacheTransfer>(
            transfer_options, device.unwrap());
        HierarchyKVCacheTransfer::CacheRegistration registration;
        registration.role = HierarchyKVCacheTransfer::CacheRole::DRAFT;
        registration.device_kv_caches = &draft_caches;
        registration.kv_cache_shape = shape;
        registration.create_options = draft_options;
        registration.producer_stream = compute_stream.get();
        registration.store_key_component = draft_component;
        transfer->register_cache(registration);
        registration.role = HierarchyKVCacheTransfer::CacheRole::TARGET;
        registration.device_kv_caches = &target_caches;
        registration.create_options = target_options;
        registration.store_key_component = "main";
        transfer->register_cache(registration);
        CHECK(transfer->finalize_registration());
        return transfer;
      };
      auto transfer = make_transfer("spec_draft::dspark::v41-test");
      EXPECT_TRUE(transfer->supports_block_type(
          HierarchyKVCacheTransfer::CacheRole::DRAFT, BlockType::SWA));
      EXPECT_FALSE(transfer->supports_block_type(
          HierarchyKVCacheTransfer::CacheRole::DRAFT, BlockType::C2));
      EXPECT_FALSE(transfer->supports_block_type(
          HierarchyKVCacheTransfer::CacheRole::DRAFT, BlockType::C1));
      std::vector<BlockType> block_types = {BlockType::SWA, BlockType::C2};
      if (has_c1) {
        block_types.emplace_back(BlockType::C1);
      }
      std::vector<BlockTransferInfo> offloads;
      std::vector<BlockTransferInfo> gets;
      std::vector<BlockTransferInfo> loads;
      for (const BlockType type : block_types) {
        for (size_t layer = 0; layer < target_caches.size(); ++layer) {
          for (const auto& [role, tensor] :
               target_caches[layer].get_block_type_tensors(type)) {
            tensor[1].fill_(static_cast<int32_t>(role) + layer + 1);
            tensor[2].zero_();
          }
        }
        for (size_t layer = 0; layer < draft_caches.size(); ++layer) {
          for (const auto& [role, tensor] :
               draft_caches[layer].get_block_type_tensors(type)) {
            tensor[1].fill_(static_cast<int32_t>(role) + layer + 31);
            tensor[2].zero_();
          }
        }
        std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> hash{};
        hash.fill(static_cast<uint8_t>(type));
        const int64_t host_row =
            type == BlockType::SWA ? 8 * std::min<int64_t>(window / 128, 2) - 1
                                   : 7;
        offloads.emplace_back(
            1, host_row, hash.data(), TransferType::D2H2G, type);
        gets.emplace_back(-1, 0, hash.data(), TransferType::G2H, type);
        loads.emplace_back(
            GetParam() ? 0 : host_row, 2, hash.data(), TransferType::H2D, type);
      }
      ASSERT_EQ(device.synchronize_default_stream(), 0);
      ASSERT_EQ(transfer->transfer_kv_blocks(91, offloads), block_types.size());
      if (GetParam()) {
        transfer.reset();
        transfer = make_transfer("spec_draft::dspark::different-weights");
        Slice<BlockTransferInfo> missing_draft(gets);
        std::vector<uint8_t> expected(block_types.size(), 1);
        expected[0] = 0;
        EXPECT_EQ(transfer->prefetch_kv_blocks(missing_draft, nullptr),
                  expected);
        transfer.reset();
        transfer = make_transfer("spec_draft::dspark::v41-test");
        Slice<BlockTransferInfo> slice(gets);
        ASSERT_EQ(transfer->prefetch_kv_blocks(slice, nullptr),
                  std::vector<uint8_t>(block_types.size(), 1));
      }
      ASSERT_EQ(transfer->transfer_kv_blocks(92, loads), block_types.size());
      ModelInputParams params;
      params.meta.batch_id = 92;
      params.meta.requires_host_restore = true;
      transfer->set_layer_synchronizer(params);
      for (int32_t layer = 0; layer < 6; ++layer) {
        ASSERT_TRUE(params.synchronize_layer(layer));
      }
      ASSERT_TRUE(params.synchronize_draft_layer());
      for (const BlockType type : block_types) {
        for (const auto* caches : {&target_caches, &draft_caches}) {
          for (const KVCache& cache : *caches) {
            for (const auto& [role, tensor] :
                 cache.get_block_type_tensors(type)) {
              EXPECT_TRUE(torch::equal(tensor[1], tensor[2]))
                  << static_cast<int32_t>(type) << ":"
                  << static_cast<int32_t>(role);
            }
          }
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Transport,
                         V41HierarchyTransferTest,
                         ::testing::Bool());

TEST(HierarchyKVCacheTransferTest,
     RoundTripPublishesStrategyLayerReadyGranularity) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }

  constexpr int64_t kBlockCount = 2;
  constexpr int64_t kBlockSize = 4;
  constexpr int64_t kLayerCount = 4;
  constexpr int64_t kSourceBlockId = 0;
  constexpr int64_t kDestinationBlockId = 1;
  constexpr uint64_t kBatchId = 7;

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount).block_size(kBlockSize);
  ModelArgs model_args;
  model_args.model_type("test_model")
      .n_layers(kLayerCount)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(8);
  const KVCacheShape cache_shape(capacity, model_args, /*world_size=*/1);

  KVCacheCreateOptions create_options;
  create_options.device(device.unwrap())
      .dtype(torch::kFloat32)
      .num_layers(kLayerCount)
      .model_type("test_model");
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, cache_shape, create_options);
  ASSERT_EQ(caches.size(), static_cast<size_t>(kLayerCount));

  for (size_t layer_index = 0; layer_index < caches.size(); ++layer_index) {
    const double layer_value = static_cast<double>(layer_index);
    caches[layer_index].get_k_cache()[kSourceBlockId].fill_(3.0 + layer_value);
    caches[layer_index].get_v_cache()[kSourceBlockId].fill_(7.0 + layer_value);
    caches[layer_index].get_k_cache()[kDestinationBlockId].zero_();
    caches[layer_index].get_v_cache()[kDestinationBlockId].zero_();
  }
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.layers(kLayerCount)
      .host_blocks_factor(2.0)
      .layers_wise_copy_batchs(2);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  HierarchyKVCacheTransfer transfer(transfer_options,
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &caches,
                                    cache_shape,
                                    create_options);

  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ModelInputParams missing;
  missing.meta.batch_id = kBatchId;
  missing.meta.requires_host_restore = true;
  EXPECT_DEATH(transfer.set_layer_synchronizer(missing),
               "Missing Host restore handle");

  BlockTransferInfo offload_info(kSourceBlockId, /*dst_block_id=*/0);
  offload_info.block_type = BlockType::KV;
  offload_info.transfer_type = TransferType::D2H2G;
  EXPECT_EQ(transfer.transfer_kv_blocks(kBatchId, {offload_info}), 1U);

  BlockTransferInfo load_info(/*src_block_id=*/0, kDestinationBlockId);
  load_info.block_type = BlockType::KV;
  load_info.transfer_type = TransferType::H2D;
  EXPECT_EQ(transfer.transfer_kv_blocks(kBatchId, {load_info}), 1U);

  missing.meta.batch_id = kBatchId + 1;
  EXPECT_DEATH(transfer.set_layer_synchronizer(missing),
               "Missing Host restore handle");

  ModelInputParams input_params;
  input_params.meta.batch_id = kBatchId;
  input_params.meta.requires_host_restore = true;
  transfer.set_layer_synchronizer(input_params);
  ASSERT_NE(input_params.parallel.layer_wise_load_synchronizer, nullptr);
  EXPECT_FALSE(input_params.meta.requires_host_restore);
  const auto synchronizer = input_params.parallel.layer_wise_load_synchronizer;
  transfer.set_layer_synchronizer(input_params);
  EXPECT_EQ(input_params.parallel.layer_wise_load_synchronizer, synchronizer);
  EXPECT_FALSE(input_params.meta.requires_host_restore);
  missing.meta.batch_id = kBatchId;
  EXPECT_DEATH(transfer.set_layer_synchronizer(missing),
               "Missing Host restore handle");
  ModelInputParams next;
  next.meta.batch_id = kBatchId + 1;
  transfer.set_layer_synchronizer(next);
  EXPECT_EQ(next.parallel.layer_wise_load_synchronizer, nullptr);
  EXPECT_FALSE(next.meta.requires_host_restore);
  EXPECT_EQ(input_params.parallel.layer_wise_load_synchronizer->size(), 2U);
  EXPECT_EQ(input_params.parallel.layers_per_event, 2U);
  for (uint32_t layer_index = 0; layer_index < kLayerCount; ++layer_index) {
    ASSERT_TRUE(input_params.synchronize_layer(layer_index));
  }

  for (KVCache& cache : caches) {
    EXPECT_TRUE(torch::equal(cache.get_k_cache()[kSourceBlockId],
                             cache.get_k_cache()[kDestinationBlockId]));
    EXPECT_TRUE(torch::equal(cache.get_v_cache()[kSourceBlockId],
                             cache.get_v_cache()[kDestinationBlockId]));
  }
}

void verify_committed_linear_round_trip(const char* model_type) {
  constexpr int64_t kSlotCount = 3;
  constexpr int64_t kSourceSlot = 1;
  constexpr int64_t kDestinationSlot = 2;
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  KVCacheCapacity capacity;
  capacity.n_blocks(2)
      .block_size(4)
      .num_linear_state_blocks(kSlotCount)
      .linear_conv_state_len(5)
      .linear_ssm_checkpoint_stride(3);
  ModelArgs model_args;
  model_args.model_type(model_type)
      .n_layers(2)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(4)
      .full_attention_interval(2)
      .linear_conv_kernel_dim(4)
      .linear_num_key_heads(1)
      .linear_num_value_heads(1)
      .linear_key_head_dim(2)
      .linear_value_head_dim(2);
  const KVCacheShape cache_shape(capacity, model_args, /*world_size=*/1);
  KVCacheCreateOptions create_options;
  create_options.device(device.unwrap())
      .dtype(torch::kFloat32)
      .ssm_dtype(torch::kFloat32)
      .block_size(4)
      .num_layers(2)
      .full_attention_interval(2)
      .enable_linear_attention(true)
      .model_type(model_type);
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, cache_shape, create_options);
  torch::Tensor conv = caches[0].get_conv_cache();
  torch::Tensor ssm = caches[0].get_ssm_cache();
  conv[kSourceSlot].copy_(
      torch::arange(conv[kSourceSlot].numel(), conv.options())
          .view_as(conv[kSourceSlot]));
  ssm.narrow(0, kSourceSlot * 3, 3)
      .copy_(torch::arange(ssm[0].numel() * 3, ssm.options())
                 .view_as(ssm.narrow(0, kSourceSlot * 3, 3)));
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.layers(2).host_blocks_factor(2.0).layers_wise_copy_batchs(1);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  const int32_t old_chunk_size =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(8);
  HierarchyKVCacheTransfer transfer(transfer_options,
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &caches,
                                    cache_shape,
                                    create_options);
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(
      old_chunk_size);
  BlockTransferInfo offload(kSourceSlot, /*dst_block_id=*/0);
  offload.block_type = BlockType::LINEAR;
  offload.transfer_type = TransferType::D2H2G;
  ASSERT_EQ(transfer.transfer_kv_blocks(/*batch_id=*/9, {offload}), 1U);

  conv[kDestinationSlot].fill_(-1.0);
  ssm.narrow(0, kDestinationSlot * 3, 3).fill_(-1.0);
  ASSERT_EQ(device.synchronize_default_stream(), 0);
  BlockTransferInfo load(/*src_block_id=*/0, kDestinationSlot);
  load.block_type = BlockType::LINEAR;
  load.transfer_type = TransferType::H2D;
  ASSERT_EQ(transfer.transfer_kv_blocks(/*batch_id=*/9, {load}), 1U);
  ModelInputParams input_params;
  input_params.meta.batch_id = 9;
  input_params.meta.requires_host_restore = true;
  transfer.set_layer_synchronizer(input_params);
  ASSERT_TRUE(input_params.synchronize_all_layers());
  EXPECT_FALSE(transfer.take_load_handle(/*batch_id=*/9).has_value());

#if defined(USE_MUSA)
  constexpr int64_t kConvHistoryAxis = 1;
#else
  constexpr int64_t kConvHistoryAxis = 0;
#endif
  const torch::Tensor destination_conv = conv[kDestinationSlot];
  const torch::Tensor source_conv = conv[kSourceSlot];
  EXPECT_TRUE(torch::equal(destination_conv.narrow(kConvHistoryAxis, 0, 3),
                           source_conv.narrow(kConvHistoryAxis, 0, 3)));
  EXPECT_TRUE(torch::all(destination_conv.narrow(kConvHistoryAxis, 3, 2) == -1)
                  .item<bool>());
  EXPECT_TRUE(torch::equal(ssm[kDestinationSlot * 3], ssm[kSourceSlot * 3]));
  EXPECT_TRUE(torch::all(ssm.narrow(0, kDestinationSlot * 3 + 1, 2) == -1)
                  .item<bool>());
}

TEST(HierarchyKVCacheTransferTest,
     LinearRoundTripCopiesOnlyCommittedCheckpoint) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }

  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 4;
  for (const char* model_type : {"qwen3_5_text", "glm5_next"}) {
    SCOPED_TRACE(model_type);
    verify_committed_linear_round_trip(model_type);
  }
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      original_chunk_stride;
}

TEST(HierarchyKVCacheTransferTest, RejectsMixedOffloadBatchBeforeSubmission) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  KVCacheCapacity capacity;
  capacity.n_blocks(2).block_size(4);
  ModelArgs model_args;
  model_args.model_type("test_model")
      .n_layers(1)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(8);
  const KVCacheShape cache_shape(capacity, model_args, /*world_size=*/1);
  KVCacheCreateOptions create_options;
  create_options.device(device.unwrap())
      .dtype(torch::kFloat32)
      .num_layers(1)
      .model_type("test_model");
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, cache_shape, create_options);

  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.layers(1).host_blocks_factor(2.0).layers_wise_copy_batchs(1);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  HierarchyKVCacheTransfer transfer(transfer_options,
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &caches,
                                    cache_shape,
                                    create_options);

  BlockTransferInfo d2h_info(/*src_block_id=*/0, /*dst_block_id=*/0);
  d2h_info.block_type = BlockType::KV;
  d2h_info.transfer_type = TransferType::D2H2G;
  BlockTransferInfo h2d_info(/*src_block_id=*/0, /*dst_block_id=*/1);
  h2d_info.block_type = BlockType::KV;
  h2d_info.transfer_type = TransferType::H2D;

  EXPECT_DEATH(
      {
        (void)transfer.transfer_kv_blocks(/*batch_id=*/7, {d2h_info, h2d_info});
      },
      "mixed transfer types");
}

}  // namespace
}  // namespace xllm
