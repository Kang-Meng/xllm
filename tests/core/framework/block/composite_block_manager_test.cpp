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

#include "composite_block_manager.h"

#include <gtest/gtest.h>

#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/batch/batch.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/block/concurrent_block_manager_impl.h"
#include "core/framework/block/embedding_block_manager.h"
#include "framework/block/block_manager_pool_test_peer.h"
#include "framework/block/block_utils.h"
#include "framework/block/linear_state_block_manager.h"
#include "framework/block/sliding_window_block_manager.h"
#include "framework/config/scheduler_config.h"
#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

namespace {

constexpr uint32_t kManagerTypeBlockManagerImpl = 0;
constexpr uint32_t kManagerTypeSlidingWindowBlockManager = 1;
constexpr uint32_t kMaxTokensPerBatch = 1280;
constexpr uint32_t kCompressedBlockTokenSize = kDsv4CompressedBlockTokenSize;

// C4 and C128 cover the same number of original tokens per logical block, so
// a base token capacity maps to equal typed block counts.
BlockManager::Options MakeCompositeOptions(uint32_t base_num_blocks,
                                           uint32_t block_size,
                                           uint32_t window_size,
                                           uint32_t max_seqs_per_batch) {
  const uint32_t swa_blocks_per_seq =
      static_cast<uint32_t>(get_swa_blocks_per_seq(window_size, block_size));
  const uint32_t burst_blocks =
      (kMaxTokensPerBatch + block_size - 1) / block_size;
  const uint32_t swa_num_blocks = swa_blocks_per_seq * max_seqs_per_batch +
                                  burst_blocks + max_seqs_per_batch + 2;
  const uint32_t compressed_num_blocks =
      base_num_blocks * block_size / kCompressedBlockTokenSize;
  BlockManager::Options opts;
  opts.num_blocks(base_num_blocks)
      .block_size(block_size)
      .sliding_window_size(window_size)
      .swa_blocks_per_seq(swa_blocks_per_seq)
      .swa_num_blocks(swa_num_blocks)
      .c4_num_blocks(compressed_num_blocks)
      .c128_num_blocks(compressed_num_blocks)
      .max_tokens_per_batch(kMaxTokensPerBatch)
      .max_seqs_per_batch(max_seqs_per_batch)
      .manager_types({kManagerTypeSlidingWindowBlockManager,
                      kManagerTypeBlockManagerImpl,
                      kManagerTypeBlockManagerImpl})
      .compress_ratios({0, 4, 128});
  return opts;
}

TEST(CompositeBlockManagerTest, BlockUtilizationUsesIndividualLeafCapacity) {
  BlockManager::Options options = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      /*block_size=*/128,
      /*window_size=*/256,
      /*max_seqs_per_batch=*/2);
  options.enable_prefix_cache(false);
  CompositeBlockManager manager(build_composite_leaves(options), options);

  const std::unordered_map<BlockType, double> initial =
      manager.block_utilization_by_type();
  EXPECT_EQ(initial.size(), 3u);
  EXPECT_DOUBLE_EQ(initial.at(BlockType::SWA), 0.0);
  EXPECT_DOUBLE_EQ(initial.at(BlockType::C4), 0.0);
  EXPECT_DOUBLE_EQ(initial.at(BlockType::C128), 0.0);

  std::vector<Block> allocated =
      manager.allocate_blocks(BlockType::C4, /*num_blocks=*/1);
  ASSERT_EQ(allocated.size(), 1u);
  const std::unordered_map<BlockType, double> utilization =
      manager.block_utilization_by_type();
  EXPECT_GT(utilization.at(BlockType::C4), 0.0);
  EXPECT_DOUBLE_EQ(utilization.at(BlockType::SWA), 0.0);
  EXPECT_DOUBLE_EQ(utilization.at(BlockType::C128), 0.0);
  allocated.clear();

  options.model_type("deepseek_v41");
  options.compress_ratios({0, 2, 1});
  CompositeBlockManager v41_manager(build_composite_leaves(options), options);
  allocated = v41_manager.allocate_blocks(BlockType::C2, /*num_blocks=*/1);
  ASSERT_EQ(allocated.size(), 1u);
  const std::unordered_map<BlockType, double> v41_utilization =
      v41_manager.block_utilization_by_type();
  EXPECT_EQ(v41_utilization.size(), 3u);
  EXPECT_DOUBLE_EQ(v41_utilization.at(BlockType::SWA), 0.0);
  EXPECT_GT(v41_utilization.at(BlockType::C2), 0.0);
  EXPECT_DOUBLE_EQ(v41_utilization.at(BlockType::C1), 0.0);
  allocated.clear();
}

TEST(CompositeBlockManagerTest, ZeroCapacityLeafStillReportsItsType) {
  BlockManager::Options options = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      /*block_size=*/128,
      /*window_size=*/256,
      /*max_seqs_per_batch=*/2);
  options.enable_prefix_cache(false);
  auto leaves = build_composite_leaves(options);
  BlockManager::Options c4_options = leaves.at(BlockType::C4).leaf->options();
  // Block 0 is reserved for padding; one physical block has zero capacity.
  c4_options.num_blocks(1);
  auto empty_c4 = std::make_unique<BlockManagerImpl>(c4_options);
  ASSERT_EQ(empty_c4->num_total_blocks(), 0u);
  leaves.at(BlockType::C4).leaf = std::move(empty_c4);

  CompositeBlockManager manager(std::move(leaves), options);
  const std::unordered_map<BlockType, double> utilization =
      manager.block_utilization_by_type();
  ASSERT_EQ(utilization.size(), 3u);
  EXPECT_DOUBLE_EQ(utilization.at(BlockType::C4), 0.0);
  EXPECT_DOUBLE_EQ(utilization.at(BlockType::SWA), 0.0);
  EXPECT_DOUBLE_EQ(utilization.at(BlockType::C128), 0.0);
}

void set_swa_capacity_for_token_budget(BlockManager::Options* options,
                                       uint32_t max_tokens_per_batch) {
  ASSERT_NE(options, nullptr);
  const uint32_t block_size = options->block_size();
  ASSERT_GT(block_size, 0u);
  const uint32_t burst_blocks =
      (max_tokens_per_batch + block_size - 1) / block_size;
  const uint32_t max_seqs = std::max(options->max_seqs_per_batch(), 1u);
  const uint32_t swa_num_blocks =
      options->swa_blocks_per_seq() * max_seqs + burst_blocks + max_seqs + 2;
  options->max_tokens_per_batch(max_tokens_per_batch)
      .swa_num_blocks(swa_num_blocks);
}

constexpr uint32_t kBaseBlockSize = 128;
constexpr uint32_t kCompressedUnitTokens = kCompressedBlockTokenSize;

// Shortest repeated prompt, no shorter than `min_units` compressed units, whose
// exact-repeat hit still spans one SWA window after dropping the last unit.
inline size_t repeat_hit_prompt_tokens(size_t min_units, uint32_t window_size) {
  return std::max<size_t>(min_units * kCompressedUnitTokens,
                          window_size + kCompressedUnitTokens);
}

inline size_t CeilBlocks(size_t num_tokens, size_t block_size) {
  return (num_tokens + block_size - 1) / block_size;
}

inline size_t ExpectedSwaLogicalBlocks(size_t num_tokens) {
  return CeilBlocks(num_tokens, kBaseBlockSize);
}

// Creates a minimal Sequence for testing (same pattern as batch_test.cpp).
Sequence MakeTestSequence(
    size_t index,
    const std::vector<int32_t>& prompt_token_ids,
    const std::vector<SampleSlot>* sample_slots = nullptr) {
  torch::Device device(Platform::type_torch(), 0);
  // Sequence keeps non-owning pointers to these parameters after this helper
  // returns, so they must outlive every sequence created here.
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker = [] {
    StoppingChecker checker;
    checker.set_max_generated_tokens(256);
    return checker;
  }();
  SequenceParams seq_params;
  // Large enough to hold DSV4-scale prompts (>= a full compressed block = 2048
  // tokens). Individual tests can still use short prompts; sequence capacity
  // just has to bound `num_tokens + max_generated_tokens`.
  seq_params.seq_capacity = 65536;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;
  seq_params.sample_slots = sample_slots;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  return Sequence(index,
                  prompt_token_ids,
                  input_embedding,
                  mm_data,
                  std::move(decoder),
                  seq_params);
}

TEST(CompositeBlockManagerTest, SampleSelectorsRetainRequiredHiddenPositions) {
  for (const size_t position : {0u, 1u, 2048u, 2049u, 4096u, 4097u}) {
    SCOPED_TRACE(position);
    const BlockManager::Options options = MakeCompositeOptions(
        /*base_num_blocks=*/128,
        /*block_size=*/kBaseBlockSize,
        /*window_size=*/128,
        /*max_seqs_per_batch=*/32);
    CompositeBlockManager manager(build_composite_leaves(options), options);
    const std::vector<int32_t> prompt(4099, /*value=*/7);
    Sequence cold = MakeTestSequence(/*index=*/0, prompt);
    ASSERT_TRUE(manager.allocate_sequence(&cold, prompt.size()));
    cold.kv_state().incr_kv_cache_tokens_num(prompt.size());
    manager.deallocate_for_sequence(&cold);
    cold.reset();

    // The earlier selector is deliberately second, so all slots must be read.
    const std::vector<SampleSlot> slots = {{"sample", 0, 4097},
                                           {"sample", 1, position}};
    Sequence selected = MakeTestSequence(/*index=*/1, prompt, &slots);
    manager.allocate_shared_for_sequence(&selected);
    const size_t source_position = position == 0 ? 0 : position - 1;
    const size_t expected =
        source_position / kCompressedBlockTokenSize * kCompressedBlockTokenSize;
    EXPECT_EQ(selected.kv_cache_tokens_num(), expected);
    EXPECT_LE(selected.kv_cache_tokens_num(), source_position);
    EXPECT_EQ(selected.kv_state().shared_blocks_num(BlockType::C128),
              expected / kCompressedBlockTokenSize);
    EXPECT_EQ(selected.kv_state().shared_blocks_num(BlockType::C4),
              expected / kCompressedBlockTokenSize);
    manager.deallocate_for_sequence(&selected);

    Sequence normal = MakeTestSequence(/*index=*/2, prompt);
    manager.allocate_shared_for_sequence(&normal);
    EXPECT_EQ(normal.kv_cache_tokens_num(), 4096u);
    manager.deallocate_for_sequence(&normal);
  }
}

// Blocks are keyed by BlockType in the sequence KVCacheState, not by
// sub-manager index. These helpers read the three DSV4 groups by type so the
// tests stay independent of compress_ratios ordering.
std::vector<Block> SwaBlocks(Sequence& seq) {
  const Slice<Block> s = seq.kv_state().blocks(BlockType::SWA);
  return std::vector<Block>(s.begin(), s.end());
}
std::vector<Block> C4Blocks(Sequence& seq) {
  const Slice<Block> s = seq.kv_state().blocks(BlockType::C4);
  return std::vector<Block>(s.begin(), s.end());
}
std::vector<Block> C128Blocks(Sequence& seq) {
  const Slice<Block> s = seq.kv_state().blocks(BlockType::C128);
  return std::vector<Block>(s.begin(), s.end());
}

}  // namespace

TEST(CompositeBlockManagerTest, ExplicitHostAllocationNeverPublishesProgress) {
  BlockManager::Options options;
  options.num_blocks(16).block_size(4).enable_host_offload(true);
  CompositeBlockManager device(build_composite_leaves(options), options);
  auto host_leaves = build_composite_leaves(options);
  host_leaves.at(BlockType::KV).participates_in_admission = false;
  CompositeBlockManager host(
      std::move(host_leaves),
      options,
      CompositeBlockManager::PrefixCachePublishMode::EXPLICIT);
  Sequence sequence = MakeTestSequence(0, std::vector<int32_t>(9, 7));
  ASSERT_TRUE(device.allocate_sequence(&sequence, 9));
  sequence.kv_state().set_kv_cache_tokens_num(8);
  const int32_t device_id = sequence.kv_state().blocks(BlockType::KV)[0].id();

  ASSERT_TRUE(host.allocate_sequence(&sequence, sequence.host_kv_state(), 9));
  EXPECT_EQ(host.num_total_blocks(), 0u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::KV), 3u);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::KV), 0u);
  for (const Block& block : sequence.host_kv_state().blocks(BlockType::KV)) {
    EXPECT_EQ(block.ref_count(), 1u);
  }
  host.cache_for_sequence(&sequence);
  host.cache_for_sequence(&sequence, 8);
  host.cache_full_blocks_for_sequence(&sequence);
  EXPECT_EQ(host.num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(
      host.leaf_entries().at(BlockType::KV).leaf->num_blocks_in_prefix_cache(),
      0u);
  EXPECT_EQ(device.num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 0u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::KV), 0u);
  for (const Block& block : sequence.kv_state().blocks(BlockType::KV)) {
    EXPECT_EQ(block.ref_count(), 1u);
  }
  for (const Block& block : sequence.host_kv_state().blocks(BlockType::KV)) {
    EXPECT_EQ(block.ref_count(), 1u);
  }

  host.deallocate_for_sequence(&sequence, sequence.host_kv_state());
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::KV), 0u);
  sequence.host_kv_state().reset();
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::KV)[0].id(), device_id);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 8u);
  EXPECT_EQ(host.num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(host.leaf_entries().at(BlockType::KV).leaf->num_used_blocks(), 0u);
  device.deallocate_for_sequence(&sequence);
  sequence.reset();
  EXPECT_EQ(device.num_blocks_in_prefix_cache(), 2u);
}

TEST(CompositeBlockManagerTest, ExplicitPublicationEnablesHostOnlySharedMatch) {
  BlockManager::Options options;
  options.num_blocks(16).block_size(4).enable_host_offload(true);
  CompositeBlockManager host(
      build_composite_leaves(options),
      options,
      CompositeBlockManager::PrefixCachePublishMode::EXPLICIT);
  Sequence source = MakeTestSequence(0, std::vector<int32_t>(9, 7));
  ASSERT_TRUE(host.allocate_sequence(&source, source.host_kv_state(), 8));
  source.update_block_hashes(4, BlockHasherType::TEXT);
  auto& blocks = *source.host_kv_state().mutable_blocks(BlockType::KV);
  for (size_t index = 0; index < blocks.size(); ++index) {
    blocks[index].set_hash_value(source.block_hashes(4)[index].data);
  }
  host.cache_blocks(BlockType::KV, blocks);
  EXPECT_EQ(host.num_blocks_in_prefix_cache(), 2u);
  EXPECT_EQ(source.host_kv_state().num_cached_blocks(BlockType::KV), 0u);
  EXPECT_EQ(source.host_kv_state().kv_cache_tokens_num(), 0u);
  for (const Block& block : blocks) {
    EXPECT_EQ(block.ref_count(), 2u);
  }

  Sequence hit = MakeTestSequence(1, std::vector<int32_t>(9, 7));
  host.allocate_shared_for_sequence(&hit, hit.host_kv_state());
  EXPECT_EQ(hit.host_kv_state().kv_cache_tokens_num(), 8u);
  EXPECT_EQ(hit.host_kv_state().num_cached_blocks(BlockType::KV), 2u);
  EXPECT_TRUE(hit.host_kv_state().prefix_cache_matched());
  EXPECT_FALSE(hit.kv_state().has_any_blocks());
  EXPECT_FALSE(hit.kv_state().prefix_cache_matched());
  host.deallocate_for_sequence(&hit, hit.host_kv_state());
  hit.host_kv_state().reset();

  Sequence exact = MakeTestSequence(2, std::vector<int32_t>(8, 7));
  host.allocate_shared_for_sequence(&exact, exact.host_kv_state());
  EXPECT_EQ(exact.host_kv_state().kv_cache_tokens_num(), 4u);
  EXPECT_EQ(exact.host_kv_state().num_cached_blocks(BlockType::KV), 1u);
  host.deallocate_for_sequence(&exact, exact.host_kv_state());
  exact.host_kv_state().reset();
  host.deallocate_for_sequence(&source, source.host_kv_state());
  source.host_kv_state().reset();
  EXPECT_EQ(host.leaf_entries().at(BlockType::KV).leaf->num_used_blocks(), 0u);
}

TEST(CompositeBlockManagerTest, FailedHostGrowthRollsBackAllStagedLeaves) {
  BlockManager::Options options = MakeCompositeOptions(4096, 128, 128, 4);
  options.enable_prefix_cache(true).enable_host_offload(true);
  CompositeBlockManager device(build_composite_leaves(options), options);
  CompositeBlockManager host(
      build_composite_leaves(options),
      options,
      CompositeBlockManager::PrefixCachePublishMode::EXPLICIT);
  Sequence sequence = MakeTestSequence(0, std::vector<int32_t>(129, 7));
  ASSERT_TRUE(device.allocate_sequence(&sequence, 128));
  const int32_t device_id = sequence.kv_state().blocks(BlockType::C4)[0].id();
  BlockManager* c128 = host.leaf_entries().at(BlockType::C128).leaf.get();
  auto held_blocks = c128->allocate(c128->num_free_blocks());
  std::map<BlockType, size_t> used_before;
  for (const auto& [type, entry] : host.leaf_entries()) {
    used_before.emplace(type, entry.leaf->num_used_blocks());
  }

  EXPECT_FALSE(
      host.allocate_sequence(&sequence, sequence.host_kv_state(), 128));
  EXPECT_FALSE(sequence.host_kv_state().has_any_blocks());
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::C4)[0].id(), device_id);
  for (const auto& [type, entry] : host.leaf_entries()) {
    EXPECT_EQ(entry.leaf->num_used_blocks(), used_before.at(type));
  }
  host.deallocate(std::move(held_blocks));
  held_blocks.clear();
  ASSERT_TRUE(host.allocate_sequence(&sequence, sequence.host_kv_state(), 128));
  for (const auto& [type, entry] : host.leaf_entries()) {
    EXPECT_EQ(sequence.host_kv_state().blocks(type)[0].ref_count(), 1u);
  }
  host.deallocate_for_sequence(&sequence, sequence.host_kv_state());
  sequence.host_kv_state().reset();
  device.deallocate_for_sequence(&sequence);
  sequence.reset();
}

TEST(CompositeBlockManagerTest, EmptyHostGrowthCannotPassCapacityValidation) {
  class EmptyHostGrowthBlockManager final : public BlockManagerImpl {
   public:
    explicit EmptyHostGrowthBlockManager(const BlockManager::Options& options)
        : BlockManagerImpl(options) {}

    std::optional<std::vector<Block>> allocate_for_sequence(Sequence*,
                                                            KVCacheState&,
                                                            size_t) override {
      return std::vector<Block>{};
    }
  };

  BlockManager::Options options;
  options.num_blocks(16).block_size(4);
  auto leaves = build_composite_leaves(options);
  leaves.at(BlockType::KV).leaf =
      std::make_unique<EmptyHostGrowthBlockManager>(options);
  CompositeBlockManager host(
      std::move(leaves),
      options,
      CompositeBlockManager::PrefixCachePublishMode::EXPLICIT);
  Sequence sequence = MakeTestSequence(0, std::vector<int32_t>(9, 7));
  EXPECT_FALSE(host.allocate_sequence(&sequence, sequence.host_kv_state(), 9));
  EXPECT_FALSE(sequence.host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence.kv_state().has_any_blocks());
}

TEST(CompositeBlockManagerTest, ExplicitDeviceStatePreservesEmbeddingDispatch) {
  BlockManager::Options options;
  options.num_blocks(16).block_size(128);
  auto leaves = build_composite_leaves(options);
  leaves.emplace(
      BlockType::EMBEDDING,
      CompositeBlockManager::LeafEntry{
          std::make_unique<EmbeddingBlockManager>(4, "test"), false, false});
  CompositeBlockManager manager(std::move(leaves), options);
  Sequence sequence = MakeTestSequence(0, std::vector<int32_t>(1025, 7));
  ASSERT_TRUE(manager.allocate_sequence(&sequence, sequence.kv_state(), 128));
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::EMBEDDING), 1u);
  const int32_t embedding_id =
      sequence.kv_state().blocks(BlockType::EMBEDDING)[0].id();
  ASSERT_TRUE(manager.allocate_sequence(&sequence, sequence.kv_state(), 1024));
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::EMBEDDING), 1u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::EMBEDDING)[0].id(),
            embedding_id);
  manager.deallocate_for_sequence(&sequence);
  sequence.reset();
}

TEST(CompositeBlockManagerTest, EmbeddingLeafUsesExplicitState) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(concurrent);
    std::unique_ptr<BlockManager> manager =
        std::make_unique<EmbeddingBlockManager>(3, "test");
    if (concurrent) {
      manager =
          std::make_unique<ConcurrentBlockManagerImpl>(std::move(manager));
    }
    Sequence sequence = MakeTestSequence(0, std::vector<int32_t>(65, 7));
    sequence.kv_state().add_blocks(BlockType::EMBEDDING, manager->allocate(1));
    const int32_t sequence_block_id =
        sequence.kv_state().blocks(BlockType::EMBEDDING)[0].id();
    KVCacheState kv_state;

    auto allocated = manager->allocate_for_sequence(&sequence, kv_state, 64);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    EXPECT_NE(allocated->front().id(), sequence_block_id);
    kv_state.add_blocks(BlockType::EMBEDDING, std::move(*allocated));
    allocated->clear();

    allocated = manager->allocate_for_sequence(&sequence, kv_state, 128);
    ASSERT_TRUE(allocated.has_value());
    EXPECT_TRUE(allocated->empty());
    EXPECT_EQ(kv_state.num_blocks(BlockType::EMBEDDING), 1u);
    EXPECT_EQ(sequence.kv_state().blocks(BlockType::EMBEDDING)[0].id(),
              sequence_block_id);
    manager->deallocate(kv_state.take_blocks(BlockType::EMBEDDING));
    kv_state.reset();
    manager->deallocate(sequence.kv_state().take_blocks(BlockType::EMBEDDING));
    sequence.reset();
    EXPECT_EQ(manager->num_used_blocks(), 0u);
    EXPECT_EQ(manager->num_free_blocks(), manager->num_total_blocks());
  }
}

TEST(CompositeBlockManagerTest, SlidingWindowLeafAcceptsNullSequence) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(concurrent);
    BlockManager::Options options = MakeCompositeOptions(512, 128, 256, 1);
    options.enable_disagg_pd(concurrent).enable_prefix_cache(false);
    CompositeBlockManager manager(build_composite_leaves(options), options);
    BlockManager* leaf = manager.leaf_entries().at(BlockType::SWA).leaf.get();
    KVCacheState kv_state;

    EXPECT_FALSE(
        leaf->allocate_for_sequence(nullptr, kv_state, 128).has_value());
    EXPECT_EQ(kv_state.num_blocks(BlockType::SWA), 0u);
    EXPECT_EQ(leaf->num_used_blocks(), 0u);
  }
}

TEST(CompositeBlockManagerTest, AllocateForSequence_SingleSeq) {
  const uint32_t base_block_size = kBaseBlockSize;
  const uint32_t base_num_blocks = 4096;  // C4/C128: 256 blocks each.
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, base_block_size, window_size, max_seqs_per_batch);

  CompositeBlockManager manager(build_composite_leaves(opts), opts);
  EXPECT_TRUE(manager.is_composite());
  EXPECT_EQ(manager.num_sub_managers(), 3u);

  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(1024, 1));
  const size_t num_tokens = 1024;

  EXPECT_TRUE(manager.allocate_sequence(&seq, num_tokens));

  const std::vector<Block> swa = SwaBlocks(seq);
  const std::vector<Block> c4 = C4Blocks(seq);
  const std::vector<Block> c128 = C128Blocks(seq);

  // SlidingWindow group: logical block count follows sequence length.
  EXPECT_EQ(swa.size(), ExpectedSwaLogicalBlocks(num_tokens));
  for (const auto& b : swa) {
    EXPECT_TRUE(b.is_valid());
    EXPECT_EQ(b.size(), base_block_size);
  }

  // C4 logical block size is 2048 original tokens.
  const size_t expected_blocks_1 =
      CeilBlocks(num_tokens, kCompressedUnitTokens);
  EXPECT_EQ(c4.size(), expected_blocks_1);
  for (const auto& b : c4) {
    EXPECT_TRUE(b.is_valid());
    EXPECT_EQ(b.size(), kCompressedUnitTokens);
  }

  // C128 logical block size is also 2048 original tokens.
  const size_t expected_blocks_2 =
      CeilBlocks(num_tokens, kCompressedUnitTokens);
  EXPECT_EQ(c128.size(), expected_blocks_2);
  for (const auto& b : c128) {
    EXPECT_TRUE(b.is_valid());
    EXPECT_EQ(b.size(), kCompressedUnitTokens);
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, AllocateForSequence_DifferentBatchSeqs) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;
  const uint32_t sliding_window_blocks_per_sequence = static_cast<uint32_t>(
      get_swa_blocks_per_seq(window_size, kBaseBlockSize));

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  // Seq1: 1024 tokens. Both compressed groups need one logical block.
  Sequence seq1 = MakeTestSequence(0, std::vector<int32_t>(1024, 1));
  EXPECT_TRUE(manager.allocate_sequence(&seq1, 1024));
  const std::vector<Block> s1_swa = SwaBlocks(seq1);
  const std::vector<Block> s1_c4 = C4Blocks(seq1);
  const std::vector<Block> s1_c128 = C128Blocks(seq1);
  EXPECT_EQ(s1_swa.size(), ExpectedSwaLogicalBlocks(1024));
  EXPECT_EQ(s1_c4.size(), CeilBlocks(1024, kCompressedUnitTokens));
  EXPECT_EQ(s1_c128.size(), CeilBlocks(1024, kCompressedUnitTokens));

  // Seq2: 1400 tokens. Both compressed groups need one 2048-token logical
  // block. Keep total SWA logical blocks within the dynamic pool budget
  // derived from max_tokens_per_batch.
  Sequence seq2 = MakeTestSequence(1, std::vector<int32_t>(1400, 1));
  EXPECT_TRUE(manager.allocate_sequence(&seq2, 1400));
  const std::vector<Block> s2_swa = SwaBlocks(seq2);
  const std::vector<Block> s2_c4 = C4Blocks(seq2);
  const std::vector<Block> s2_c128 = C128Blocks(seq2);
  EXPECT_EQ(s2_swa.size(), ExpectedSwaLogicalBlocks(1400));
  EXPECT_EQ(s2_c4.size(), CeilBlocks(1400, kCompressedUnitTokens));
  EXPECT_EQ(s2_c128.size(), CeilBlocks(1400, kCompressedUnitTokens));

  // Blocks allocated to different seqs must not overlap (distinct block ids).
  std::set<int32_t> ids1_0, ids1_1, ids1_2, ids2_0, ids2_1, ids2_2;
  for (const auto& b : s1_swa) ids1_0.insert(b.id());
  for (const auto& b : s1_c4) ids1_1.insert(b.id());
  for (const auto& b : s1_c128) ids1_2.insert(b.id());
  for (const auto& b : s2_swa) ids2_0.insert(b.id());
  for (const auto& b : s2_c4) ids2_1.insert(b.id());
  for (const auto& b : s2_c128) ids2_2.insert(b.id());
  for (int32_t id : ids1_0) EXPECT_EQ(ids2_0.count(id), 0u);
  for (int32_t id : ids1_1) EXPECT_EQ(ids2_1.count(id), 0u);
  for (int32_t id : ids1_2) EXPECT_EQ(ids2_2.count(id), 0u);
  const int32_t max_swa_block_id =
      sliding_window_blocks_per_sequence * max_seqs_per_batch +
      CeilBlocks(kMaxTokensPerBatch, kBaseBlockSize) + max_seqs_per_batch + 1;
  for (int32_t id : ids1_0) EXPECT_LE(id, max_swa_block_id);
  for (int32_t id : ids2_0) EXPECT_LE(id, max_swa_block_id);

  manager.deallocate_for_sequence(&seq1);
  manager.deallocate_for_sequence(&seq2);
}

TEST(CompositeBlockManagerTest, AllocateForSequence_GrowSameSeq) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1, 2, 3});
  // Both compressed leaves use one common 2048-token logical unit.
  EXPECT_TRUE(manager.allocate_sequence(&seq, 600));
  EXPECT_EQ(SwaBlocks(seq).size(), ExpectedSwaLogicalBlocks(600));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(600, kCompressedUnitTokens));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(600, kCompressedUnitTokens));

  // Growing within the same compressed unit must not allocate another block.
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1200));
  EXPECT_EQ(SwaBlocks(seq).size(), ExpectedSwaLogicalBlocks(1200));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(1200, kCompressedUnitTokens));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(1200, kCompressedUnitTokens));

  // No growth: still 1200 tokens, block counts unchanged.
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1200));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(1200, kCompressedUnitTokens));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(1200, kCompressedUnitTokens));

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, AllocateContinuesAfterSatisfiedTokenManager) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  opts.compress_ratios({0, 128, 4});
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1024));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(1024, kCompressedUnitTokens));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(1024, kCompressedUnitTokens));

  EXPECT_TRUE(manager.allocate_sequence(&seq, 1500));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(1500, kCompressedUnitTokens));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(1500, kCompressedUnitTokens));

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, AllocateForSequence_NullSeqReturnsFalse) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);
  EXPECT_FALSE(manager.allocate_sequence(nullptr, 10));
}

TEST(CompositeBlockManagerTest, FailedGrowthKeepsCommittedBlocks) {
  BlockManager::Options opts = MakeCompositeOptions(/*base_num_blocks=*/256,
                                                    kBaseBlockSize,
                                                    /*window_size=*/12,
                                                    /*max_seqs_per_batch=*/4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  ASSERT_TRUE(manager.allocate_sequence(&seq, 1024));
  const std::vector<Block> before_swa = SwaBlocks(seq);
  const size_t c4_blocks_before = C4Blocks(seq).size();
  const size_t c128_blocks_before = C128Blocks(seq).size();
  std::vector<int32_t> swa_ids_before;
  swa_ids_before.reserve(before_swa.size());
  for (const auto& block : before_swa) {
    ASSERT_TRUE(block.is_valid());
    swa_ids_before.push_back(block.id());
  }
  seq.kv_state().incr_kv_cache_tokens_num(1024);

  // 1024 committed tokens and a 12-token window release every block before
  // the block that contains token 1012. C4/C128 have no window, so a failed
  // growth keeps the blocks those leaves already committed.
  const size_t released_blocks =
      (1024 - opts.sliding_window_size()) / kBaseBlockSize;
  ASSERT_GT(before_swa.size(), released_blocks);

  EXPECT_FALSE(manager.allocate_sequence(&seq, 4096));
  EXPECT_EQ(C4Blocks(seq).size(), c4_blocks_before);
  EXPECT_EQ(C128Blocks(seq).size(), c128_blocks_before);

  const std::vector<Block> after_swa = SwaBlocks(seq);
  ASSERT_EQ(after_swa.size(), before_swa.size());
  for (size_t i = 0; i < released_blocks; ++i) {
    EXPECT_FALSE(after_swa[i].is_valid());
  }
  for (size_t i = released_blocks; i < after_swa.size(); ++i) {
    EXPECT_TRUE(after_swa[i].is_valid());
    EXPECT_EQ(after_swa[i].id(), swa_ids_before[i]);
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, DeallocateToleratesRolledBackEmptySequence) {
  BlockManager::Options opts = MakeCompositeOptions(/*base_num_blocks=*/128,
                                                    kBaseBlockSize,
                                                    /*window_size=*/12,
                                                    /*max_seqs_per_batch=*/4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});

  EXPECT_FALSE(manager.allocate_sequence(&seq, 4096));
  EXPECT_NO_FATAL_FAILURE(manager.deallocate_for_sequence(&seq));
  EXPECT_FALSE(seq.kv_state().has_multi_block_export());
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

// Verifies that when seq token count increases, CompositeBlockManager correctly
// adds blocks (only appends new blocks; existing block ids are preserved).
TEST(CompositeBlockManagerTest, TokenIncrease_AddsBlocksIncrementally) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  std::vector<size_t> token_steps = {100, 600, 1200, 2000, 2400};

  std::vector<std::set<int32_t>> prev_ids_1;
  std::vector<std::set<int32_t>> prev_ids_2;

  for (size_t num_tokens : token_steps) {
    EXPECT_TRUE(manager.allocate_sequence(&seq, num_tokens));

    const std::vector<Block> c4 = C4Blocks(seq);
    const std::vector<Block> c128 = C128Blocks(seq);

    const size_t expect_1 = CeilBlocks(num_tokens, kCompressedUnitTokens);
    const size_t expect_2 = CeilBlocks(num_tokens, kCompressedUnitTokens);
    EXPECT_EQ(c4.size(), expect_1)
        << "num_tokens=" << num_tokens << " C4 block count";
    EXPECT_EQ(c128.size(), expect_2)
        << "num_tokens=" << num_tokens << " C128 block count";

    // Check that previously allocated block ids are still present (only
    // append).
    std::set<int32_t> ids_1, ids_2;
    for (const auto& b : c4) ids_1.insert(b.id());
    for (const auto& b : c128) ids_2.insert(b.id());
    for (const auto& prev : prev_ids_1) {
      for (int32_t id : prev) EXPECT_GT(ids_1.count(id), 0u);
    }
    for (const auto& prev : prev_ids_2) {
      for (int32_t id : prev) EXPECT_GT(ids_2.count(id), 0u);
    }
    prev_ids_1.push_back(ids_1);
    prev_ids_2.push_back(ids_2);
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, Dsv4CompressedAllocationUsesPlatformUnits) {
  BlockManager::Options opts = MakeCompositeOptions(/*base_num_blocks=*/4096,
                                                    kBaseBlockSize,
                                                    /*window_size=*/128,
                                                    /*max_seqs_per_batch=*/4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  for (const auto& [num_tokens, expected_blocks] :
       std::vector<std::pair<size_t, size_t>>{{kCompressedUnitTokens - 1, 1},
                                              {kCompressedUnitTokens, 1},
                                              {kCompressedUnitTokens + 1, 2}}) {
    SCOPED_TRACE(num_tokens);
    ASSERT_TRUE(manager.allocate_sequence(&seq, num_tokens));
    EXPECT_EQ(C4Blocks(seq).size(), expected_blocks);
    EXPECT_EQ(C128Blocks(seq).size(), expected_blocks);
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, SlidingWindowReleasesSkippedPhysicalBlocks) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t sliding_window_blocks_per_sequence = 3;
  const uint32_t window_size =
      sliding_window_blocks_per_sequence * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  const size_t window_tokens =
      static_cast<size_t>(sliding_window_blocks_per_sequence) * kBaseBlockSize;

  EXPECT_TRUE(manager.allocate_sequence(&seq, window_tokens));
  const std::vector<Block> initial = SwaBlocks(seq);
  ASSERT_EQ(initial.size(), sliding_window_blocks_per_sequence);
  seq.kv_state().incr_kv_cache_tokens_num(window_tokens);

  std::vector<int32_t> initial_ids;
  initial_ids.reserve(initial.size());
  for (const auto& block : initial) {
    initial_ids.push_back(block.id());
  }

  EXPECT_TRUE(
      manager.allocate_sequence(&seq, window_tokens + 2 * kBaseBlockSize));
  const std::vector<Block> boundary = SwaBlocks(seq);
  ASSERT_EQ(boundary.size(), sliding_window_blocks_per_sequence + 2);
  for (size_t i = 0; i < initial_ids.size(); ++i) {
    EXPECT_EQ(boundary[i].id(), initial_ids[i]);
  }
  seq.kv_state().incr_kv_cache_tokens_num(2 * kBaseBlockSize);

  EXPECT_TRUE(
      manager.allocate_sequence(&seq, window_tokens + 3 * kBaseBlockSize));
  const std::vector<Block> exceeded = SwaBlocks(seq);
  ASSERT_EQ(exceeded.size(), sliding_window_blocks_per_sequence + 3);
  // cached=5 blocks, window=3 blocks: every block before the block containing
  // the window start is released.
  EXPECT_FALSE(exceeded[0].is_valid());
  EXPECT_EQ(exceeded[0].id(), -1);
  EXPECT_FALSE(exceeded[1].is_valid());
  EXPECT_EQ(exceeded[1].id(), -1);
  for (size_t i = 2; i < initial_ids.size(); ++i) {
    EXPECT_EQ(exceeded[i].id(), initial_ids[i]);
  }
  for (size_t i = initial_ids.size(); i < exceeded.size(); ++i) {
    EXPECT_TRUE(exceeded[i].is_valid());
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest,
     SlidingWindowHostCursorLagRetryPreservesSpeculativeWindow) {
  BlockManager::Options device_options;
  device_options.num_blocks(8)
      .block_size(kBaseBlockSize)
      .block_type(BlockType::SWA)
      .sliding_window_size(kBaseBlockSize)
      .swa_blocks_per_seq(1)
      .num_speculative_tokens(2)
      .enable_prefix_cache(false);
  BlockManager::Options host_options = device_options;
  host_options.num_blocks(5);

  SlidingWindowBlockManager device(device_options);
  SlidingWindowBlockManager host(host_options);
  Sequence sequence = MakeTestSequence(0, std::vector<int32_t>(16, 7));

  auto allocate_and_mount = [&sequence](BlockManager& manager,
                                        KVCacheState& state,
                                        size_t num_tokens) {
    auto blocks = manager.allocate_for_sequence(&sequence, state, num_tokens);
    EXPECT_TRUE(blocks.has_value());
    if (blocks.has_value()) {
      state.add_blocks(BlockType::SWA, std::move(*blocks));
    }
  };

  // Establish Host's older one-block cursor first. HBM then advances beyond
  // it, so the sequence-level cursor is the HBM cursor for later allocations.
  allocate_and_mount(host, sequence.host_kv_state(), kBaseBlockSize);
  sequence.host_kv_state().set_kv_cache_tokens_num(kBaseBlockSize);

  // HBM is ahead of Host by one block.
  allocate_and_mount(device, sequence.kv_state(), 2 * kBaseBlockSize);
  sequence.kv_state().set_kv_cache_tokens_num(2 * kBaseBlockSize);

  allocate_and_mount(device, sequence.kv_state(), 3 * kBaseBlockSize);
  sequence.kv_state().set_kv_cache_tokens_num(3 * kBaseBlockSize);

  // The 16-token prompt is already behind the cache cursor, so this growth is
  // decode and covers num_tokens plus the speculative width. The sequence
  // cursor also releases the first HBM block before the new block is appended.
  auto hbm_growth = device.allocate_for_sequence(
      &sequence, sequence.kv_state(), 4 * kBaseBlockSize);
  ASSERT_TRUE(hbm_growth.has_value());
  sequence.kv_state().add_blocks(BlockType::SWA, std::move(*hbm_growth));
  // The speculative span is covered by the retained tail. The window pads
  // two invalid blocks and keeps two live blocks.
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::SWA), 4u);
  EXPECT_FALSE(sequence.kv_state().blocks(BlockType::SWA)[0].is_valid());
  EXPECT_FALSE(sequence.kv_state().blocks(BlockType::SWA)[1].is_valid());

  // Leave no free Host blocks. Host first releases its committed block against
  // the HBM cursor and extends the logical vector, then the physical allocate
  // fails.
  std::vector<Block> exhausted = host.allocate(host.num_free_blocks());
  ASSERT_EQ(host.num_free_blocks(), 0u);
  auto host_growth = host.allocate_for_sequence(
      &sequence, sequence.host_kv_state(), 4 * kBaseBlockSize);
  EXPECT_FALSE(host_growth.has_value());
  ASSERT_EQ(sequence.host_kv_state().num_blocks(BlockType::SWA), 2u);
  EXPECT_FALSE(sequence.host_kv_state().blocks(BlockType::SWA)[0].is_valid());
  EXPECT_FALSE(sequence.host_kv_state().blocks(BlockType::SWA)[1].is_valid());

  // Once Host capacity returns, the same target fills the logical tail. The
  // released prefix stays an invalid placeholder, and the retained window is
  // the same four blocks HBM holds.
  host.deallocate(std::move(exhausted));
  host_growth = host.allocate_for_sequence(
      &sequence, sequence.host_kv_state(), 4 * kBaseBlockSize);
  ASSERT_TRUE(host_growth.has_value());
  sequence.host_kv_state().add_blocks(BlockType::SWA, std::move(*host_growth));
  ASSERT_EQ(sequence.host_kv_state().num_blocks(BlockType::SWA), 4u);
  EXPECT_FALSE(sequence.host_kv_state().blocks(BlockType::SWA)[0].is_valid());
  EXPECT_FALSE(sequence.host_kv_state().blocks(BlockType::SWA)[1].is_valid());
  EXPECT_TRUE(sequence.host_kv_state().blocks(BlockType::SWA)[2].is_valid());

  device.deallocate(std::move(sequence.kv_state().take_blocks(BlockType::SWA)));
  host.deallocate(
      std::move(sequence.host_kv_state().take_blocks(BlockType::SWA)));
}

TEST(CompositeBlockManagerTest, DeallocateConsumesBlocksFromOwnerManagers) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  opts.enable_prefix_cache(false);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1500));
  EXPECT_GT(manager.num_used_blocks(), 0u);

  std::vector<Block> flat_blocks;
  flat_blocks.reserve(seq.kv_state().num_blocks(BlockType::SWA) +
                      seq.kv_state().num_blocks(BlockType::C4) +
                      seq.kv_state().num_blocks(BlockType::C128) + 1);
  flat_blocks.emplace_back();
  for (const BlockType type :
       {BlockType::SWA, BlockType::C4, BlockType::C128}) {
    std::vector<Block> manager_blocks = seq.kv_state().take_blocks(type);
    for (Block& block : manager_blocks) {
      flat_blocks.emplace_back(std::move(block));
    }
  }

  manager.deallocate(std::move(flat_blocks));
  EXPECT_TRUE(flat_blocks.empty());
  EXPECT_EQ(manager.num_used_blocks(), 0u);
  for (const auto& [type, entry] : manager.leaf_entries()) {
    EXPECT_EQ(entry.leaf->num_free_blocks(), entry.leaf->num_total_blocks());
  }

  seq.reset();
}

TEST(CompositeBlockManagerTest, DeallocateDispatchesWithoutInflatingRefCount) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1500));
  EXPECT_GT(manager.num_used_blocks(), 0u);

  std::vector<Block> owned_blocks;
  for (const BlockType type :
       {BlockType::SWA, BlockType::C4, BlockType::C128}) {
    std::vector<Block> manager_blocks = seq.kv_state().take_blocks(type);
    for (auto& block : manager_blocks) {
      owned_blocks.emplace_back(std::move(block));
    }
  }

  for (const Block& block : owned_blocks) {
    EXPECT_EQ(block.ref_count(), 1u);
  }

  manager.deallocate(std::move(owned_blocks));
  EXPECT_TRUE(owned_blocks.empty());
  EXPECT_EQ(manager.num_used_blocks(), 0u);

  seq.reset();
}

TEST(CompositeBlockManagerTest, CapacityStatsUseBaseBlockUnits) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t total = manager.num_total_blocks();
  const size_t base_blocks_per_compressed_unit =
      kCompressedUnitTokens / kBaseBlockSize;
  EXPECT_EQ(total,
            (opts.c4_num_blocks() - 1u) * base_blocks_per_compressed_unit);

  // Free (no sequence yet) equals total; used is 0.
  EXPECT_EQ(manager.num_free_blocks(), total);
  EXPECT_EQ(manager.num_used_blocks(), 0u);

  // After allocating one sequence, used remains normalized to base blocks.
  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(1024, 1));
  ASSERT_TRUE(manager.allocate_sequence(&seq, 1024));
  const size_t c4_used = C4Blocks(seq).size();  // capacity leaf's used count
  EXPECT_EQ(manager.num_used_blocks(),
            c4_used * base_blocks_per_compressed_unit);
  EXPECT_EQ(manager.num_free_blocks(),
            total - c4_used * base_blocks_per_compressed_unit);

  manager.deallocate_for_sequence(&seq);
}

// DSV4 prefix cache: a fresh sequence with the same prompt as a previously
// released sequence should mount shared blocks from all three prefix-cache
// leaves (SWA / C4 / C128). The composite takes the cross-leaf min hit length
// clamped to a common compressed block, so the prompt has to span at least
// 2048 tokens for the hit to be non-trivial. Uses a two-unit
// prompt so we get a meaningful hit even after the exact-repeat pop.
TEST(CompositeBlockManagerTest, Dsv4PrefixCacheHitOnRepeatedPrefix) {
  const uint32_t base_num_blocks = 4096;
  // Sliding window covers a couple compressed units worth of tokens so the SWA
  // tail-continuity check has room to succeed even after the exact-repeat pop.
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  // max_tokens_per_batch has to accommodate the prompt so allocate_sequence
  // does not exceed the SWA burst budget.
  const size_t num_tokens =
      repeat_hit_prompt_tokens(/*min_units=*/2, window_size);
  set_swa_capacity_for_token_budget(
      &opts, static_cast<uint32_t>(num_tokens + kCompressedUnitTokens));
  ASSERT_TRUE(opts.enable_prefix_cache());
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  // First sequence: a 2*C128 prompt (32768 tokens) so both C4 and C128 have
  // multiple full blocks worth of cacheable prefix. Mark all tokens as
  // forwarded so the pre-grow hook (fired during the NEXT allocate) has data
  // to insert -- and we drive one extra allocate to trigger it explicitly.
  const std::vector<int32_t> prompt(num_tokens, 7);
  Sequence seq_first = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq_first, num_tokens));
  seq_first.kv_state().incr_kv_cache_tokens_num(num_tokens);
  // deallocate_for_sequence runs the pre-grow hook once more (via
  // cache_for_sequence's fan-out), flushing the residual full blocks.
  manager.deallocate_for_sequence(&seq_first);
  seq_first.reset();

  // The prefix cache under SWA / C4 / C128 should now hold the prompt's full
  // blocks. On a second sequence with the same prompt, admission mounts from
  // all three leaves and pins kv_cache_tokens_num to the safe min hit.
  Sequence seq_hit = MakeTestSequence(1, prompt);
  manager.allocate_shared_for_sequence(&seq_hit);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::SWA), 0u);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::C128), 0u);
  const size_t kv_tokens = seq_hit.kv_state().kv_cache_tokens_num();
  EXPECT_GT(kv_tokens, 0u);
  EXPECT_LT(kv_tokens, num_tokens);  // exact-repeat pop kept at least one c128
  EXPECT_EQ(kv_tokens % kCompressedUnitTokens, 0u)
      << "safe_hit_tokens must be a multiple of the compressed unit";
  // SWA vector length matches safe_hit / base and the tail carries valid
  // blocks.
  const std::vector<Block> hit_swa = SwaBlocks(seq_hit);
  EXPECT_EQ(hit_swa.size(), kv_tokens / kBaseBlockSize);
  EXPECT_TRUE(hit_swa.back().is_valid());

  manager.deallocate_for_sequence(&seq_hit);
}

TEST(CompositeBlockManagerTest,
     Dsv4PrefixCacheMatchesAtCompressedUnitBoundary) {
  const size_t unit_tokens = kCompressedUnitTokens;
  const std::vector<std::pair<size_t, size_t>> cases = {
      {unit_tokens - 1, 0},
      {unit_tokens, 0},
      {unit_tokens + 1, unit_tokens},
      {2 * unit_tokens, unit_tokens},
      {2 * unit_tokens + 1, 2 * unit_tokens},
  };
  for (const auto& [prompt_tokens, expected_hit_tokens] : cases) {
    SCOPED_TRACE(prompt_tokens);
    const uint32_t base_num_blocks = 4096;
    const uint32_t window_size = kBaseBlockSize;
    const uint32_t max_seqs_per_batch = 4;
    BlockManager::Options options = MakeCompositeOptions(
        base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
    set_swa_capacity_for_token_budget(&options, 3 * kCompressedUnitTokens);
    CompositeBlockManager manager(build_composite_leaves(options), options);

    const std::vector<int32_t> cold_prompt(prompt_tokens + 1, 7);
    Sequence cold = MakeTestSequence(0, cold_prompt);
    ASSERT_TRUE(manager.allocate_sequence(&cold, cold_prompt.size()));
    cold.kv_state().incr_kv_cache_tokens_num(cold_prompt.size());
    manager.deallocate_for_sequence(&cold);
    cold.reset();

    // The cold sequence processes prompt_tokens plus one extra token.
    const size_t published_units = (prompt_tokens + 1) / unit_tokens;
    EXPECT_EQ(manager.leaf_entries()
                  .at(BlockType::C4)
                  .leaf->num_blocks_in_prefix_cache(),
              published_units);
    EXPECT_EQ(manager.leaf_entries()
                  .at(BlockType::C128)
                  .leaf->num_blocks_in_prefix_cache(),
              published_units);

    Sequence hot = MakeTestSequence(1, std::vector<int32_t>(prompt_tokens, 7));
    manager.allocate_shared_for_sequence(&hot);
    EXPECT_EQ(hot.kv_state().kv_cache_tokens_num(), expected_hit_tokens);
    EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C4),
              expected_hit_tokens / unit_tokens);
    EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C128),
              expected_hit_tokens / unit_tokens);
    manager.deallocate_for_sequence(&hot);
  }
}

// DSV4 prefix cache: a fresh sequence with a completely different prompt should
// miss cleanly (no shared mount, kv_cache_tokens_num unchanged).
TEST(CompositeBlockManagerTest, Dsv4PrefixCacheMissCleanly) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  set_swa_capacity_for_token_budget(&opts, 3 * kCompressedUnitTokens);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t num_tokens = 2 * kCompressedUnitTokens;
  const std::vector<int32_t> prompt_a(num_tokens, 7);
  Sequence seq_a = MakeTestSequence(0, prompt_a);
  ASSERT_TRUE(manager.allocate_sequence(&seq_a, num_tokens));
  seq_a.kv_state().incr_kv_cache_tokens_num(num_tokens);
  manager.deallocate_for_sequence(&seq_a);
  seq_a.reset();

  // A sequence with a totally different prompt should not share any block.
  std::vector<int32_t> prompt_b(num_tokens, 0);
  for (size_t i = 0; i < prompt_b.size(); ++i) {
    prompt_b[i] = static_cast<int32_t>(i + 100);
  }
  Sequence seq_b = MakeTestSequence(1, prompt_b);
  manager.allocate_shared_for_sequence(&seq_b);
  EXPECT_EQ(seq_b.kv_state().shared_blocks_num(BlockType::SWA), 0u);
  EXPECT_EQ(seq_b.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_EQ(seq_b.kv_state().shared_blocks_num(BlockType::C128), 0u);
  EXPECT_EQ(seq_b.kv_state().kv_cache_tokens_num(), 0u);

  manager.deallocate_for_sequence(&seq_b);
}

TEST(CompositeBlockManagerTest, Dsv4PrefixCacheEvictsAtC128Capacity) {
  // C128 has four physical blocks and each prompt consumes two. The third
  // distinct prompt must evict the first prompt from both compressed leaves.
  const uint32_t base_num_blocks =
      4 * kCompressedBlockTokenSize / kBaseBlockSize;
  const uint32_t window_size = 4 * kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, /*max_seqs_per_batch=*/1);
  set_swa_capacity_for_token_budget(&opts, 2 * kCompressedUnitTokens);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t num_tokens = 2 * kCompressedUnitTokens;
  for (int32_t value : {11, 22, 33}) {
    Sequence seq =
        MakeTestSequence(value, std::vector<int32_t>(num_tokens, value));
    ASSERT_TRUE(manager.allocate_sequence(&seq, num_tokens));
    seq.kv_state().incr_kv_cache_tokens_num(num_tokens);
    manager.deallocate_for_sequence(&seq);
    seq.reset();
  }

  Sequence first = MakeTestSequence(99, std::vector<int32_t>(num_tokens, 11));
  manager.allocate_shared_for_sequence(&first);
  EXPECT_EQ(first.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_EQ(first.kv_state().shared_blocks_num(BlockType::C128), 0u);
  EXPECT_EQ(first.kv_state().kv_cache_tokens_num(), 0u);
  manager.deallocate_for_sequence(&first);
}

// SWA slid-out blocks should enter the prefix cache via the pre-grow hook (v2b:
// hook fires at every allocate_sequence, insertion is incremental and cached
// blocks survive slid-out release through the ref<=2u path). A follow-up
// sequence with the same prompt should hit those cached blocks.
TEST(CompositeBlockManagerTest, SlidingWindowSlidOutBlocksEnterPrefixCache) {
  const uint32_t base_num_blocks = 4096;
  // Window covers a small number of base blocks so slide-out happens quickly.
  const uint32_t sliding_window_blocks_per_sequence = 3;
  const uint32_t window_size =
      sliding_window_blocks_per_sequence * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  const size_t num_tokens =
      repeat_hit_prompt_tokens(/*min_units=*/2, window_size);
  set_swa_capacity_for_token_budget(
      &opts, static_cast<uint32_t>(num_tokens + kCompressedUnitTokens));
  ASSERT_TRUE(opts.enable_prefix_cache());
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  // Prompt spans two full compressed units so the min-across-leaves gate can
  // survive AND the exact-repeat pop (one compressed unit) still leaves shared
  // blocks behind. A single-c128 prompt would end up entirely popped.
  const std::vector<int32_t> prompt(num_tokens, 7);
  Sequence seq_first = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq_first, num_tokens));
  seq_first.kv_state().incr_kv_cache_tokens_num(num_tokens);
  // The SWA leaf should have released everything but the last window-sized
  // slab back to the pool (as invalid placeholders), and those blocks landed
  // in the prefix cache via the pre-grow hook before release.
  manager.deallocate_for_sequence(&seq_first);
  seq_first.reset();

  // A second sequence with the same prompt should hit the cached SWA tail.
  Sequence seq_hit = MakeTestSequence(1, prompt);
  manager.allocate_shared_for_sequence(&seq_hit);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::SWA), 0u);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::C128), 0u);

  manager.deallocate_for_sequence(&seq_hit);
}

TEST(CompositeBlockManagerTest,
     SlidingWindowReclaimsUncachedBlockBeforeFullCacheUnit) {
#if defined(USE_MLU)
  GTEST_SKIP() << "Requires a compressed unit larger than the base block";
#endif
  const uint32_t window_size = 2 * kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      kBaseBlockSize,
      window_size,
      /*max_seqs_per_batch=*/1);
  opts.enable_prefix_cache(true).swa_num_blocks(
      /*three live blocks plus padding=*/4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t first_chunk_tokens = 3 * kBaseBlockSize;
  const size_t second_chunk_tokens = 4 * kBaseBlockSize;
  Sequence seq =
      MakeTestSequence(0, std::vector<int32_t>(second_chunk_tokens, 7));

  ASSERT_TRUE(manager.allocate_sequence(&seq, first_chunk_tokens));
  BlockManager* swa_leaf = manager.leaf_entries().at(BlockType::SWA).leaf.get();
  ASSERT_NE(swa_leaf, nullptr);
  EXPECT_EQ(swa_leaf->num_free_blocks(), 0u);

  seq.kv_state().incr_kv_cache_tokens_num(first_chunk_tokens);

  // Three committed blocks and a two-block window release the leading block.
  // The freed slot covers the next block, so growth succeeds.
  EXPECT_TRUE(manager.allocate_sequence(&seq, second_chunk_tokens));
  const std::vector<Block> swa_blocks = SwaBlocks(seq);
  ASSERT_EQ(swa_blocks.size(), 4u);
  EXPECT_FALSE(swa_blocks.front().is_valid());
  EXPECT_TRUE(swa_blocks.back().is_valid());
  EXPECT_EQ(swa_leaf->num_free_blocks(), 0u);
  EXPECT_EQ(swa_leaf->num_blocks_in_prefix_cache(), 0u);

  manager.deallocate_for_sequence(&seq);
}

// The post-grow hook advances KVCacheState::num_cached_blocks incrementally.
// Newly allocated blocks are present by then, but the token cursor limits the
// published range to blocks completed by the preceding forward.
TEST(CompositeBlockManagerTest, Dsv4PrefixCachePostGrowCursorAdvances) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  set_swa_capacity_for_token_budget(&opts, 4 * kCompressedUnitTokens);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  // Two-chunk prompt. Chunk 1 is one compressed unit wide.
  const size_t chunk = kCompressedUnitTokens;
  const size_t total = 2 * chunk;
  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(total, 7));

  // Chunk 1: allocate + forward.
  ASSERT_TRUE(manager.allocate_sequence(&seq, chunk));
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::SWA), 0u)
      << "post-grow hook has nothing to cache on the first allocate: kv=0";
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C4), 0u);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C128), 0u);
  seq.kv_state().incr_kv_cache_tokens_num(chunk);

  // Chunk 2 allocation triggers the post-grow hook, which now sees kv=chunk
  // worth of forwarded tokens and inserts them.
  ASSERT_TRUE(manager.allocate_sequence(&seq, total));
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::SWA),
            chunk / kBaseBlockSize);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C4),
            chunk / kCompressedUnitTokens);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C128),
            chunk / kCompressedUnitTokens);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::SWA)
                .leaf->num_blocks_in_prefix_cache(),
            std::min<size_t>(window_size, chunk) / kBaseBlockSize);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::C4)
                .leaf->num_blocks_in_prefix_cache(),
            1u);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::C128)
                .leaf->num_blocks_in_prefix_cache(),
            1u);

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, Dsv4PrefixCacheSkipsPartialCacheUnitTail) {
#if defined(USE_MLU)
  GTEST_SKIP() << "Requires a compressed unit larger than the base block";
#endif
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  set_swa_capacity_for_token_budget(&opts, 2 * kCompressedUnitTokens);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t completed_tokens = kCompressedUnitTokens + kBaseBlockSize;
  Sequence seq =
      MakeTestSequence(0, std::vector<int32_t>(completed_tokens, 17));
  ASSERT_TRUE(manager.allocate_sequence(&seq, completed_tokens));
  seq.kv_state().incr_kv_cache_tokens_num(completed_tokens);
  manager.cache_for_sequence(&seq);

  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::SWA)
                .leaf->num_blocks_in_prefix_cache(),
            1u);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::C4)
                .leaf->num_blocks_in_prefix_cache(),
            1u);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::C128)
                .leaf->num_blocks_in_prefix_cache(),
            1u);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::SWA),
            kCompressedUnitTokens / kBaseBlockSize);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C4),
            kCompressedUnitTokens / kCompressedUnitTokens);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C128), 1u);

  manager.deallocate_for_sequence(&seq);
}

// v2b: exact-repeat safe-hit hits the pop path -- when a fresh sequence has a
// prompt whose entire length is cached, safe_hit_tokens is popped by one
// c128 block worth of tokens so the forward has data to process. The test
// verifies kv_cache_tokens_num after mount is strictly less than the prompt.
TEST(CompositeBlockManagerTest, Dsv4PrefixCacheExactRepeatPopsOneC128) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  const size_t num_tokens =
      repeat_hit_prompt_tokens(/*min_units=*/3, window_size);
  set_swa_capacity_for_token_budget(
      &opts, static_cast<uint32_t>(num_tokens + kCompressedUnitTokens));
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const std::vector<int32_t> prompt(num_tokens, 42);
  Sequence seq_first = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq_first, num_tokens));
  seq_first.kv_state().incr_kv_cache_tokens_num(num_tokens);
  manager.deallocate_for_sequence(&seq_first);
  seq_first.reset();

  Sequence seq_hit = MakeTestSequence(1, prompt);
  manager.allocate_shared_for_sequence(&seq_hit);
  // Full-length hit → pop one c128 block. Expect kv = num_tokens - c128.
  EXPECT_EQ(seq_hit.kv_state().kv_cache_tokens_num(),
            num_tokens - kCompressedUnitTokens);

  manager.deallocate_for_sequence(&seq_hit);
}

TEST(CompositeBlockManagerTest,
     Dsv4ExactRepeatRejectsEvictedPreviousSwaWindow) {
  const uint32_t window_size = 4 * kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      kBaseBlockSize,
      window_size,
      /*max_seqs_per_batch=*/1);
  set_swa_capacity_for_token_budget(&opts, 2 * kCompressedUnitTokens);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t prompt_tokens = 2 * kCompressedUnitTokens;
  const std::vector<int32_t> prompt_a(prompt_tokens, 42);
  Sequence source = MakeTestSequence(/*index=*/0, prompt_a);
  ASSERT_TRUE(manager.allocate_sequence(&source, prompt_tokens));
  source.kv_state().incr_kv_cache_tokens_num(prompt_tokens);
  manager.deallocate_for_sequence(&source);
  source.reset();

  // Filling the SWA pool with a different active sequence evicts the oldest
  // cached SWA blocks from A, including A's window ending at 16384, while its
  // newest window ending at 32768 remains. The larger C4/C128 pools retain A's
  // complete compressed prefix.
  const std::vector<int32_t> prompt_b(prompt_tokens, 43);
  Sequence pressure = MakeTestSequence(/*index=*/1, prompt_b);
  ASSERT_TRUE(manager.allocate_sequence(&pressure, prompt_tokens));

  Sequence probe = MakeTestSequence(/*index=*/2, prompt_a);
  manager.allocate_shared_for_sequence(&probe);
  EXPECT_EQ(probe.kv_state().kv_cache_tokens_num(), 0u);

  manager.deallocate_for_sequence(&probe);
  manager.deallocate_for_sequence(&pressure);
}

// DSV4 D-side (instance_is_decode=true) should skip the SWA leaf's prefix
// cache entirely: no shared blocks on SWA even when the same prompt was
// previously seeded. C4 / C128 continue to hit because the role predicate
// only masks SWA / LINEAR / EMBEDDING on the DECODE side. This is the mirror of
// Dsv4PrefixCacheHitOnRepeatedPrefix under the DECODE role.
TEST(CompositeBlockManagerTest, DecodeRoleSkipsSwaPrefixCache) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  const size_t num_tokens =
      repeat_hit_prompt_tokens(/*min_units=*/2, window_size);
  set_swa_capacity_for_token_budget(
      &opts, static_cast<uint32_t>(num_tokens + kCompressedUnitTokens));
  // First seed the cache under a PREFILL role so all three leaves publish
  // their blocks -- then swap in a DECODE-role composite that shares the
  // hash space via the same leaf construction path.
  ASSERT_TRUE(opts.enable_prefix_cache());
  {
    CompositeBlockManager prefill_manager(build_composite_leaves(opts), opts);
    const std::vector<int32_t> prompt(num_tokens, 7);
    Sequence seq_seed = MakeTestSequence(0, prompt);
    ASSERT_TRUE(prefill_manager.allocate_sequence(&seq_seed, num_tokens));
    seq_seed.kv_state().incr_kv_cache_tokens_num(num_tokens);
    prefill_manager.deallocate_for_sequence(&seq_seed);
    seq_seed.reset();

    // Under the PREFILL role, SWA/C4/C128 all hit.
    Sequence seq_p_hit = MakeTestSequence(1, prompt);
    prefill_manager.allocate_shared_for_sequence(&seq_p_hit);
    EXPECT_GT(seq_p_hit.kv_state().shared_blocks_num(BlockType::SWA), 0u);
    EXPECT_GT(seq_p_hit.kv_state().shared_blocks_num(BlockType::C4), 0u);
    EXPECT_GT(seq_p_hit.kv_state().shared_blocks_num(BlockType::C128), 0u);
    prefill_manager.deallocate_for_sequence(&seq_p_hit);
  }

  // Under DECODE role: separate manager (its own SWA leaf ⇒ own cache), so a
  // fresh prompt lookup must return zero SWA / zero C4 / zero C128. The
  // point of this test is that CONSTRUCTION does not FATAL and the SWA leaf
  // is skipped at classify + probe time; the composite still classifies as
  // SWA_COMPRESSED (C4+C128 remain prefix-cache-on).
  BlockManager::Options decode_opts = opts;
  decode_opts.instance_is_decode(true);
  CompositeBlockManager decode_manager(build_composite_leaves(decode_opts),
                                       decode_opts);
  const std::vector<int32_t> prompt(num_tokens, 7);
  Sequence seq_d = MakeTestSequence(2, prompt);
  decode_manager.allocate_shared_for_sequence(&seq_d);
  // SWA prefix cache is off on DECODE ⇒ never hits.
  EXPECT_EQ(seq_d.kv_state().shared_blocks_num(BlockType::SWA), 0u);
  // C4/C128 caches are fresh (per-manager) so also zero here, but the intent
  // is: the participate predicate keeps them enabled so future hits work.
  EXPECT_EQ(seq_d.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_EQ(seq_d.kv_state().shared_blocks_num(BlockType::C128), 0u);
  // No safe_hit_tokens change either (nothing mounted).
  EXPECT_EQ(seq_d.kv_state().kv_cache_tokens_num(), 0u);

  decode_manager.deallocate_for_sequence(&seq_d);
}

TEST(CompositeBlockManagerTest, DecodeInitialSwaAllocationKeepsOnlyWindowTail) {
  const uint32_t window_size = 2 * kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      kBaseBlockSize,
      window_size,
      /*max_seqs_per_batch=*/1);
  opts.instance_is_decode(true).enable_prefix_cache(false).swa_num_blocks(
      /*two windows plus padding=*/5);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t logical_blocks = 10;
  const size_t num_tokens = logical_blocks * kBaseBlockSize;
  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(num_tokens, 7));

  ASSERT_TRUE(manager.allocate_sequence(&seq, num_tokens));
  const std::vector<Block> swa = SwaBlocks(seq);
  ASSERT_EQ(swa.size(), logical_blocks);
  for (size_t i = 0; i < logical_blocks - 2; ++i) {
    EXPECT_FALSE(swa[i].is_valid());
  }
  EXPECT_TRUE(swa[logical_blocks - 2].is_valid());
  EXPECT_TRUE(swa[logical_blocks - 1].is_valid());

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest,
     DecodeInitialSwaAllocationCoversNonAlignedWindow) {
  const uint32_t window_size = kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      kBaseBlockSize,
      window_size,
      /*max_seqs_per_batch=*/1);
  opts.instance_is_decode(true)
      .enable_prefix_cache(false)
      .num_speculative_tokens(5)
      .swa_num_blocks(/*two windows plus padding=*/5);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t num_tokens = 10000;
  const size_t logical_blocks =
      (num_tokens + kBaseBlockSize - 1) / kBaseBlockSize;
  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(num_tokens, 7));

  ASSERT_TRUE(manager.allocate_sequence(&seq, num_tokens));
  const std::vector<Block> swa = SwaBlocks(seq);
  ASSERT_EQ(swa.size(), logical_blocks);
  for (size_t i = 0; i < logical_blocks - 2; ++i) {
    EXPECT_FALSE(swa[i].is_valid());
  }
  EXPECT_TRUE(swa[logical_blocks - 2].is_valid());
  EXPECT_TRUE(swa[logical_blocks - 1].is_valid());

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest,
     DecodeInitialSwaAllocationKeepsBlockContainingWindowStart) {
  const uint32_t window_size = kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      kBaseBlockSize,
      window_size,
      /*max_seqs_per_batch=*/1);
  opts.instance_is_decode(true)
      .enable_prefix_cache(false)
      .num_speculative_tokens(0)
      .swa_num_blocks(/*two windows plus padding=*/5);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t num_tokens = 2 * kBaseBlockSize - 1;
  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(num_tokens, 7));

  ASSERT_TRUE(manager.allocate_sequence(&seq, num_tokens));
  const std::vector<Block> swa = SwaBlocks(seq);
  ASSERT_EQ(swa.size(), 2u);
  EXPECT_TRUE(swa[0].is_valid());
  EXPECT_TRUE(swa[1].is_valid());

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest,
     DecodeLinearWithoutPrefixCacheKeepsReceivedState) {
  // Prefix cache is off, so the linear stride is max(block size, chunk).
  // Pin the chunk at the KV block size and restore it after.
  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 128;

  const uint32_t block_size = 128;
  const uint32_t num_blocks = 128;
  BlockManager::Options opts;
  opts.num_blocks(num_blocks)
      .block_size(block_size)
      .enable_linear_state(true)
      .linear_state_num_slots(64)
      .enable_prefix_cache(false)
      .instance_is_decode(true);
  // No manager_types → flat KV shape. LINEAR is added on top by
  // build_composite_leaves.
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  EXPECT_FALSE(manager.leaf_entries().at(BlockType::KV).supports_prefix_cache);
  EXPECT_FALSE(
      manager.leaf_entries().at(BlockType::LINEAR).supports_prefix_cache);
  EXPECT_EQ(manager.leaf_entries().at(BlockType::LINEAR).leaf->block_size(),
            block_size);

  const std::vector<int32_t> prompt(4 * block_size, 5);
  Sequence seq = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq, prompt.size()));
  const int32_t received_id = seq.get_linear_state_slot_id();
  seq.kv_state().set_kv_cache_tokens_num(prompt.size());
  seq.host_kv_state().set_kv_cache_tokens_num(prompt.size());
  seq.append_token(8);
  ASSERT_TRUE(manager.allocate_sequence(&seq, seq.num_tokens()));
  EXPECT_EQ(seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(seq.kv_state().blocks(BlockType::LINEAR).front().id(), received_id);
  ASSERT_TRUE(manager.allocate_sequence(&seq, seq.num_tokens()));
  ASSERT_EQ(seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(seq.kv_state().blocks(BlockType::LINEAR).front().id(), received_id);
  seq.kv_state().incr_kv_cache_tokens_num(1);
  manager.deallocate_for_sequence(&seq);

  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      original_chunk_stride;
}

TEST(CompositeBlockManagerTest, DecodeRoleLinearDefaultStrideReusesSlot) {
  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  // Emulate --enable_chunked_prefill=false: stride stays at the -1 default.
  // Prefix cache is off, so the unset chunk falls back to the KV block size.
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = -1;

  const uint32_t block_size = 128;
  const uint32_t num_blocks = 128;
  BlockManager::Options opts;
  opts.num_blocks(num_blocks)
      .block_size(block_size)
      .enable_linear_state(true)
      .linear_state_num_slots(64)
      .enable_prefix_cache(false)
      .instance_is_decode(true);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);
  EXPECT_EQ(manager.leaf_entries().at(BlockType::LINEAR).leaf->block_size(),
            block_size);
  const std::vector<int32_t> prompt(4 * block_size, 5);
  Sequence seq = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq, prompt.size()));
  const int32_t received_id = seq.get_linear_state_slot_id();
  seq.kv_state().set_kv_cache_tokens_num(prompt.size());
  seq.append_token(8);
  ASSERT_TRUE(manager.allocate_sequence(&seq, seq.num_tokens()));
  // One generated token stays inside the KV block, so the received slot is
  // reused and no second slot is opened.
  EXPECT_EQ(seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(seq.kv_state().blocks(BlockType::LINEAR).front().id(), received_id);
  EXPECT_EQ(seq.get_linear_state_slot_id(), received_id);
  manager.deallocate_for_sequence(&seq);

  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      original_chunk_stride;
}

namespace {

BlockManager::Options linear_options(uint32_t num_slots,
                                     int32_t block_size,
                                     bool enable_prefix_cache,
                                     bool instance_is_decode = false,
                                     uint32_t num_speculative_tokens = 0,
                                     bool preserve_decode_checkpoint = false) {
  BlockManager::Options options;
  options.num_blocks(num_slots)
      .block_size(block_size)
      .enable_prefix_cache(enable_prefix_cache)
      .instance_is_decode(instance_is_decode)
      .num_speculative_tokens(num_speculative_tokens)
      .preserve_decode_checkpoint(preserve_decode_checkpoint)
      .block_type(BlockType::LINEAR);
  return options;
}

class LinearStateWindowTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    original_stride_ =
        SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 4;
    stopping_checker_.set_max_generated_tokens(256);
  }

  void TearDown() override {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        original_stride_;
  }

  std::unique_ptr<CompositeBlockManager> make_manager(
      uint32_t slots = 16,
      uint32_t kv_blocks = 32,
      uint32_t speculative_tokens = 0,
      bool concurrent = false,
      uint32_t block_size = 2) {
    BlockManager::Options options;
    options.num_blocks(kv_blocks)
        .block_size(block_size)
        .enable_linear_state(true)
        .linear_state_num_slots(slots)
        .num_speculative_tokens(speculative_tokens)
        .enable_disagg_pd(concurrent)
        .enable_prefix_cache(GetParam());
    return std::make_unique<CompositeBlockManager>(
        build_composite_leaves(options), options);
  }

  Sequence make_sequence(size_t index,
                         const std::vector<int32_t>& prompt_token_ids,
                         bool enable_schedule_overlap = false) {
    SequenceParams sequence_params;
    sequence_params.seq_capacity = 8192;
    sequence_params.stopping_checker = &stopping_checker_;
    sequence_params.sampling_param = &sampling_param_;
    sequence_params.skip_special_tokens = true;
    sequence_params.echo = false;
    sequence_params.logprobs = false;
    sequence_params.enable_schedule_overlap = enable_schedule_overlap;
    IncrementalDecoder decoder("", 1, false, false);
    return Sequence(index,
                    prompt_token_ids,
                    torch::Tensor(),
                    MMData(),
                    std::move(decoder),
                    sequence_params);
  }

  void finish_input(Sequence& sequence, size_t cached_tokens) {
    sequence.kv_state().set_kv_cache_tokens_num(cached_tokens);
  }

 private:
  int32_t original_stride_ = 0;
  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
};

}  // namespace

TEST_P(LinearStateWindowTest, PrefillLeafUsesExplicitState) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(concurrent);
    auto manager = make_manager(8, 32, 0, concurrent);
    Sequence sequence = make_sequence(0, std::vector<int32_t>(12, 7));
    ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
    finish_input(sequence, 4);
    const int32_t sequence_source_id = sequence.get_linear_state_slot_id();
    BlockManager* leaf =
        manager->leaf_entries().at(BlockType::LINEAR).leaf.get();
    KVCacheState kv_state;
    kv_state.add_blocks(BlockType::LINEAR, leaf->allocate(1));

    auto allocated = leaf->allocate_for_sequence(&sequence, kv_state, 8);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 2u);
    EXPECT_TRUE(kv_state.blocks(BlockType::LINEAR).empty());
    EXPECT_EQ(kv_state.num_cached_blocks(BlockType::LINEAR), 0u);
    // Released physical IDs may be reused, but the live sequence must keep
    // its own source and the newly allocated window must own fresh references.
    EXPECT_NE(allocated->front().id(), sequence_source_id);
    EXPECT_NE(allocated->back().id(), sequence_source_id);
    EXPECT_NE(allocated->front().id(), allocated->back().id());
    EXPECT_EQ(allocated->front().ref_count(), 1u);
    EXPECT_EQ(allocated->back().ref_count(), 1u);
    EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::LINEAR), 0u);
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), sequence_source_id);
    allocated.reset();
    leaf->deallocate(kv_state.take_blocks(BlockType::LINEAR));
    kv_state.reset();
    manager->deallocate_for_sequence(&sequence);
    sequence.reset();
    EXPECT_EQ(leaf->num_used_blocks(), 0u);
  }
}

TEST_P(LinearStateWindowTest, DecodeReceiverUsesExplicitState) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(concurrent);
    std::unique_ptr<BlockManager> manager =
        std::make_unique<LinearStateBlockManager>(
            linear_options(/*num_slots=*/4,
                           /*block_size=*/4,
                           GetParam(),
                           /*instance_is_decode=*/true));
    if (concurrent) {
      manager =
          std::make_unique<ConcurrentBlockManagerImpl>(std::move(manager));
    }
    Sequence sequence = make_sequence(0, std::vector<int32_t>(12, 7));
    sequence.kv_state().add_blocks(BlockType::LINEAR, manager->allocate(1));
    const int32_t sequence_source_id = sequence.get_linear_state_slot_id();
    KVCacheState kv_state;

    auto allocated = manager->allocate_for_sequence(&sequence, kv_state, 12);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    kv_state.add_blocks(BlockType::LINEAR, std::move(*allocated));
    allocated->clear();
    const int32_t retained_source_id =
        kv_state.blocks(BlockType::LINEAR).back().id();

    sequence.kv_state().set_kv_cache_tokens_num(12);
    allocated = manager->allocate_for_sequence(&sequence, kv_state, 13);
    ASSERT_TRUE(allocated.has_value());
    EXPECT_TRUE(allocated->empty());
    ASSERT_EQ(kv_state.num_blocks(BlockType::LINEAR), 1u);
    EXPECT_EQ(kv_state.blocks(BlockType::LINEAR).front().id(),
              retained_source_id);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), sequence_source_id);
    EXPECT_EQ(manager->num_used_blocks(), 2u);
    manager->deallocate(kv_state.take_blocks(BlockType::LINEAR));
    kv_state.reset();
    manager->deallocate(sequence.kv_state().take_blocks(BlockType::LINEAR));
    sequence.reset();
    EXPECT_EQ(manager->num_used_blocks(), 0u);
    EXPECT_EQ(manager->num_free_blocks(), manager->num_total_blocks());
  }
}

TEST_P(LinearStateWindowTest,
       HostDecodeCheckpointDoesNotAdvanceSequenceHashes) {
  LinearStateBlockManager host_manager(linear_options(
      /*num_slots=*/8,
      /*block_size=*/4,
      GetParam(),
      /*instance_is_decode=*/true,
      /*num_speculative_tokens=*/0,
      /*preserve_decode_checkpoint=*/true));
  Sequence sequence = make_sequence(0, {7, 7, 7});
  sequence.kv_state().set_kv_cache_tokens_num(3);
  sequence.append_token(8);
  sequence.host_kv_state().add_blocks(BlockType::LINEAR,
                                      host_manager.allocate(2));
  const int32_t checkpoint_id =
      sequence.host_kv_state().blocks(BlockType::LINEAR).front().id();
  const int32_t live_id =
      sequence.host_kv_state().blocks(BlockType::LINEAR).back().id();

  std::optional<std::vector<Block>> allocated =
      host_manager.allocate_for_sequence(
          &sequence, sequence.host_kv_state(), sequence.num_tokens());

  ASSERT_TRUE(allocated.has_value());
  EXPECT_TRUE(allocated->empty());
  EXPECT_TRUE(sequence.block_hashes(4).empty());
  // The step has not confirmed a checkpoint, so the two slots swap. The
  // previous live slot is the read source and the checkpoint slot is live.
  ASSERT_EQ(sequence.host_kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.host_kv_state().blocks(BlockType::LINEAR).front().id(),
            live_id);
  EXPECT_EQ(sequence.host_kv_state().blocks(BlockType::LINEAR).back().id(),
            checkpoint_id);

  host_manager.deallocate(
      sequence.host_kv_state().take_blocks(BlockType::LINEAR));
  sequence.host_kv_state().reset();
  EXPECT_EQ(host_manager.num_used_blocks(), 0u);
}

TEST_P(LinearStateWindowTest,
       DecodeCheckpointPreserveRebuildsWindowAfterCrossingWithOneBlock) {
  LinearStateBlockManager manager(linear_options(
      /*num_slots=*/8,
      /*block_size=*/4,
      GetParam(),
      /*instance_is_decode=*/true,
      /*num_speculative_tokens=*/0,
      /*preserve_decode_checkpoint=*/true));
  Sequence sequence = make_sequence(0, std::vector<int32_t>(12, 7));
  sequence.kv_state().set_kv_cache_tokens_num(12);
  sequence.host_kv_state().set_kv_cache_tokens_num(12);
  sequence.kv_state().add_blocks(BlockType::LINEAR, manager.allocate(1));
  sequence.host_kv_state().add_blocks(BlockType::LINEAR, manager.allocate(1));
  sequence.append_token(8);
  sequence.set_accepted_length(1);

  auto hbm_allocated = manager.allocate_for_sequence(
      &sequence, sequence.kv_state(), sequence.num_tokens());
  auto host_allocated = manager.allocate_for_sequence(
      &sequence, sequence.host_kv_state(), sequence.num_tokens());

  ASSERT_TRUE(hbm_allocated.has_value());
  ASSERT_TRUE(host_allocated.has_value());
  // A one-block window looks ahead to the next step. HBM cached tokens is
  // 12 after append_token writes the token index, and a stride of 4 shares a
  // checkpoint bucket with predicted=13, so neither tier
  // asks for a second block and the caller keeps the original window.
  EXPECT_TRUE(hbm_allocated->empty());
  EXPECT_TRUE(host_allocated->empty());
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_TRUE(sequence.kv_state().blocks(BlockType::LINEAR).front().is_valid());
  EXPECT_TRUE(
      sequence.host_kv_state().blocks(BlockType::LINEAR).front().is_valid());

  manager.deallocate(sequence.kv_state().take_blocks(BlockType::LINEAR));
  manager.deallocate(sequence.host_kv_state().take_blocks(BlockType::LINEAR));
  sequence.reset();
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

TEST_P(LinearStateWindowTest, DecodeReceiverAllocationIsOwnedByLeaf) {
  for (const size_t prompt_length : {12u, 13u}) {
    LinearStateBlockManager manager(
        linear_options(/*num_slots=*/3,
                       /*block_size=*/4,
                       GetParam(),
                       /*instance_is_decode=*/true));
    Sequence sequence =
        make_sequence(0, std::vector<int32_t>(prompt_length, 7));
    auto allocated = manager.allocate_for_sequence(
        &sequence, sequence.kv_state(), prompt_length);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    EXPECT_TRUE(allocated->back().is_valid());
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 0u);
    allocated.reset();
    EXPECT_EQ(manager.num_used_blocks(), 0u);

    allocated = manager.allocate_for_sequence(
        &sequence, sequence.kv_state(), prompt_length);
    ASSERT_TRUE(allocated.has_value());
    sequence.kv_state().add_blocks(BlockType::LINEAR, std::move(*allocated));
    allocated->clear();
    const int32_t received_id = sequence.get_linear_state_slot_id();

    finish_input(sequence, prompt_length);
    sequence.append_token(8);
    allocated = manager.allocate_for_sequence(
        &sequence, sequence.kv_state(), prompt_length + 1);
    ASSERT_TRUE(allocated.has_value());
    EXPECT_TRUE(allocated->empty());
    EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), prompt_length);
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
    EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
              received_id);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), received_id);
    EXPECT_EQ(manager.num_used_blocks(), 1u);
    manager.deallocate(sequence.kv_state().take_blocks(BlockType::LINEAR));
  }
}

TEST_P(LinearStateWindowTest, DecodeReceiverNeedsOnlyOneAvailableSlot) {
  BlockManager::Options options;
  options.num_blocks(32)
      .block_size(2)
      .enable_linear_state(true)
      .linear_state_num_slots(3)
      .enable_prefix_cache(false)
      .instance_is_decode(true);
  CompositeBlockManager manager(build_composite_leaves(options), options);
  BlockManager* linear_leaf =
      manager.leaf_entries().at(BlockType::LINEAR).leaf.get();
  auto held_slots = linear_leaf->allocate(2);
  ASSERT_EQ(held_slots.size(), 2u);
  Sequence sequence = make_sequence(0, std::vector<int32_t>(12, 7));
  // KV grows. With two slots held, the linear leaf does not mount a block.
  EXPECT_TRUE(manager.allocate_sequence(&sequence, 12));
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::KV), 6u);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 0u);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 0u);

  held_slots.pop_back();
  EXPECT_EQ(linear_leaf->num_free_blocks(), 1u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), -1);
  finish_input(sequence, 12);
  sequence.append_token(8);
  // The freed slot becomes the live decode slot.
  EXPECT_TRUE(manager.allocate_sequence(&sequence, 13));
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_GE(sequence.get_linear_state_slot_id(), 0);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 12u);
  manager.deallocate_for_sequence(&sequence);
  EXPECT_EQ(linear_leaf->num_used_blocks(), 1u);
  held_slots.clear();
}

TEST_P(LinearStateWindowTest,
       PrefillUsesTwoBlocksAndPublishesLogicalCheckpoints) {
  auto manager = make_manager();
  Sequence sequence = make_sequence(0, std::vector<int32_t>(13, 7));
  int32_t previous_id = -1;
  for (size_t output_index = 0; output_index < 4; ++output_index) {
    const size_t target = std::min((output_index + 1) * 4, size_t{13});
    ASSERT_TRUE(manager->allocate_sequence(&sequence, target));
    const Slice<Block> blocks = sequence.kv_state().blocks(BlockType::LINEAR);
    ASSERT_EQ(blocks.size(), output_index == 0 ? 1u : 2u);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), blocks.back().id());
    EXPECT_NE(blocks.back().id(), previous_id);
    if (output_index > 0) {
      EXPECT_EQ(blocks.front().id(), previous_id);
      if (GetParam()) {
        EXPECT_EQ(XXH3Key(blocks.front().get_immutable_hash_value()),
                  sequence.block_hashes(2)[output_index * 2 - 1]);
      }
    }
    const int32_t output_id = blocks.back().id();
    finish_input(sequence, target);
    manager->cache_for_sequence(&sequence);
    previous_id = output_id;
  }
  sequence.append_token(8);
  ASSERT_TRUE(manager->allocate_sequence(&sequence, sequence.num_tokens()));
  const Slice<Block> blocks = sequence.kv_state().blocks(BlockType::LINEAR);
  // Decode keeps the two-slot window. The previous write slot stays in front.
  ASSERT_EQ(blocks.size(), 2u);
  EXPECT_EQ(blocks.front().id(), previous_id);
  EXPECT_TRUE(blocks.front().is_valid());
  manager->deallocate_for_sequence(&sequence);
  EXPECT_EQ(
      manager->leaf_entries().at(BlockType::LINEAR).leaf->num_used_blocks(),
      0u);
}

TEST_P(LinearStateWindowTest, ExhaustionPreservesSourceAndRollsBackKvGrowth) {
  auto manager = make_manager(3);
  Sequence sequence = make_sequence(0, std::vector<int32_t>(13, 7));
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
  finish_input(sequence, 4);
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 8));
  finish_input(sequence, 8);
  const int32_t source_id = sequence.get_linear_state_slot_id();
  Block retained_old_read =
      sequence.kv_state().blocks(BlockType::LINEAR).front();
  const size_t kv_used =
      manager->leaf_entries().at(BlockType::KV).leaf->num_used_blocks();
  for (size_t retry = 0; retry < 2; ++retry) {
    // Three slots cover the two-block prefill window, so the next chunk
    // admits and KV grows with it. LINEAR stays one block because this
    // allocation is still prefill (HBM cache is behind the prompt).
    EXPECT_TRUE(manager->allocate_sequence(&sequence, 12));
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), source_id);
    EXPECT_EQ(sequence.kv_cache_tokens_num(), 8u);
    // LINEAR does not record a prefix-cache insertion cursor.
    EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::LINEAR), 0u);
    EXPECT_EQ(manager->leaf_entries().at(BlockType::KV).leaf->num_used_blocks(),
              kv_used + 2);
  }
  retained_old_read = Block();
  // Repeating the same target rolls the window: the previous live slot moves
  // to the front and a new write slot is allocated.
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 12));
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
            source_id);
  manager->deallocate_for_sequence(&sequence);
  EXPECT_EQ(
      manager->leaf_entries().at(BlockType::LINEAR).leaf->num_used_blocks(),
      0u);
}

TEST_P(LinearStateWindowTest, DecodeRollingExhaustionFallsBackToCurrentSource) {
  auto manager = make_manager(3, 32);
  Sequence sequence = make_sequence(0, {7, 7, 7});
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 3));
  finish_input(sequence, 3);
  sequence.append_token(8);
  const int32_t source_id = sequence.get_linear_state_slot_id();
  BlockManager* linear_leaf =
      manager->leaf_entries().at(BlockType::LINEAR).leaf.get();
  std::vector<Block> held = linear_leaf->allocate(1);
  ASSERT_EQ(held.size(), 1u);

  for (size_t retry = 0; retry < 2; ++retry) {
    EXPECT_TRUE(manager->allocate_sequence(&sequence, 4));
    ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), source_id);
  }

  held.clear();
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
            source_id);
  manager->deallocate_for_sequence(&sequence);
}

TEST_P(LinearStateWindowTest, KvFailureDoesNotChangeLinearWindow) {
  auto manager = make_manager(16, 3);
  Sequence sequence = make_sequence(0, std::vector<int32_t>(9, 7));
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
  finish_input(sequence, 4);
  const int32_t source_id = sequence.get_linear_state_slot_id();
  EXPECT_FALSE(manager->allocate_sequence(&sequence, 8));
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), source_id);
  EXPECT_EQ(
      manager->leaf_entries().at(BlockType::LINEAR).leaf->num_used_blocks(),
      1u);
  manager->deallocate_for_sequence(&sequence);
}

TEST_P(LinearStateWindowTest, IncompleteKvGrowthRollsBackLinearAllocation) {
  class EmptyGrowthBlockManager final : public BlockManagerImpl {
   public:
    explicit EmptyGrowthBlockManager(const BlockManager::Options& options)
        : BlockManagerImpl(options) {}

    std::optional<std::vector<Block>> allocate_for_sequence(Sequence*,
                                                            KVCacheState&,
                                                            size_t) override {
      return std::vector<Block>{};
    }
  };

  BlockManager::Options options;
  options.num_blocks(32)
      .block_size(2)
      .enable_linear_state(true)
      .linear_state_num_slots(3)
      .enable_prefix_cache(GetParam());
  auto leaves = build_composite_leaves(options);
  leaves.at(BlockType::KV).leaf =
      std::make_unique<EmptyGrowthBlockManager>(options);
  CompositeBlockManager manager(std::move(leaves), options);
  Sequence sequence = make_sequence(0, std::vector<int32_t>(3, 7));
  sequence.kv_state().add_blocks(
      BlockType::KV,
      manager.leaf_entries().at(BlockType::KV).leaf->allocate(2));
  ASSERT_TRUE(manager.allocate_sequence(&sequence, 3));
  finish_input(sequence, 3);
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.append_token(8);
  ASSERT_TRUE(manager.allocate_sequence(&sequence, 4));
  const int32_t source_id = sequence.get_linear_state_slot_id();

  // Empty KV growth does not open another linear slot. The mounted source
  // stays, and the sequence is not admitted.
  EXPECT_FALSE(manager.allocate_sequence(&sequence, 5));
  // The linear leaf still opens the write slot. KV growth stays empty, so
  // admission fails, and the new slot is the live id.
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_NE(sequence.get_linear_state_slot_id(), source_id);
  EXPECT_EQ(
      manager.leaf_entries().at(BlockType::LINEAR).leaf->num_used_blocks(), 2u);
  sequence.kv_state().add_blocks(
      BlockType::KV,
      manager.leaf_entries().at(BlockType::KV).leaf->allocate(1));
  ASSERT_TRUE(manager.allocate_sequence(&sequence, 5));
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), source_id);
  manager.deallocate_for_sequence(&sequence);
}

TEST_P(LinearStateWindowTest, PrefillCapacityIsIndependentOfCheckpointStride) {
  auto manager = make_manager();
  Sequence sequence = make_sequence(0, std::vector<int32_t>(13, 7));
  for (const size_t target : {3u, 5u, 8u, 13u}) {
    ASSERT_TRUE(manager->allocate_sequence(&sequence, target));
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR),
              target == 3 ? 1u : 2u);
    finish_input(sequence, target);
  }
  manager->deallocate_for_sequence(&sequence);

  Sequence full_prefill = make_sequence(1, std::vector<int32_t>(13, 8));
  ASSERT_TRUE(manager->allocate_sequence(&full_prefill, 16));
  EXPECT_EQ(full_prefill.kv_state().num_blocks(BlockType::LINEAR), 1u);
  manager->deallocate_for_sequence(&full_prefill);
}

TEST_P(LinearStateWindowTest, PrefillReusesTwoSlotsAcrossArbitraryChunks) {
  auto manager = make_manager(3);
  Sequence sequence = make_sequence(0, std::vector<int32_t>(13, 7));
  int32_t previous_write_id = -1;
  for (const size_t target : {3u, 5u, 8u, 12u, 13u}) {
    ASSERT_TRUE(manager->allocate_sequence(&sequence, target));
    const Slice<Block> blocks = sequence.kv_state().blocks(BlockType::LINEAR);
    ASSERT_EQ(blocks.size(), target == 3 ? 1u : 2u);
    if (previous_write_id >= 0) {
      EXPECT_NE(blocks.front().id(), blocks.back().id());
      EXPECT_EQ(blocks.front().id(), previous_write_id);
    }
    previous_write_id = blocks.back().id();
    finish_input(sequence, target);
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR),
              target == 3 ? 1u : 2u);
  }
  manager->deallocate_for_sequence(&sequence);
}

TEST_P(LinearStateWindowTest, ColdPrefillNeedsOnlyOneSlot) {
  auto manager = make_manager(2);
  Sequence sequence = make_sequence(0, std::vector<int32_t>(9, 7));
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
  const int32_t source_id = sequence.get_linear_state_slot_id();
  finish_input(sequence, 4);
  // Prefill reuses the live slot and allocates the next write into the
  // second free slot, so two slots are enough for the next chunk.
  EXPECT_TRUE(manager->allocate_sequence(&sequence, 8));
  EXPECT_EQ(sequence.get_linear_state_slot_id(), source_id);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
  Sequence consumer = make_sequence(1, std::vector<int32_t>(9, 7));
  manager->allocate_shared_for_sequence(&consumer);
  // The producer has not published a checkpoint yet, so the consumer misses.
  EXPECT_EQ(consumer.kv_cache_tokens_num(), 0u);
  EXPECT_EQ(consumer.get_linear_state_slot_id(), -1);
  manager->deallocate_for_sequence(&consumer);
  manager->deallocate_for_sequence(&sequence);
}

TEST_P(LinearStateWindowTest, NonOverlapDecodePredictsAndRetiresCheckpoint) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(concurrent);
    auto manager = make_manager(3, 256, 0, concurrent);
    Sequence sequence = make_sequence(0, {7, 7, 7});
    ASSERT_TRUE(manager->allocate_sequence(&sequence, 3));
    const int32_t initial_id = sequence.get_linear_state_slot_id();
    finish_input(sequence, 3);
    sequence.append_token(8);
    EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 3u);

    ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
    ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
    EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
              initial_id);
    const int32_t rolled_id = sequence.get_linear_state_slot_id();
    EXPECT_NE(rolled_id, initial_id);

    sequence.append_token(9);
    EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 4u);
    ASSERT_TRUE(manager->allocate_sequence(&sequence, 5));
    ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), rolled_id);

    BlockManager* leaf =
        manager->leaf_entries().at(BlockType::LINEAR).leaf.get();
    EXPECT_EQ(leaf->num_blocks_in_prefix_cache(), 0u);
    manager->deallocate_for_sequence(&sequence);
    EXPECT_EQ(leaf->num_used_blocks(), 0u);
  }
}

TEST_P(LinearStateWindowTest, OverlapDecodePredictsAndResolvesCheckpoint) {
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 8;
  for (const bool checkpoint_hit : {false, true}) {
    SCOPED_TRACE(checkpoint_hit);
    // Prefix cache off sizes the linear slot as max(block size, chunk). Keep
    // the KV block at the chunk so both modes share this checkpoint grid.
    auto manager = make_manager(4, 256, 3, false, /*block_size=*/8);
    Sequence sequence = make_sequence(0, std::vector<int32_t>(6, 7), true);
    ASSERT_TRUE(manager->allocate_sequence(&sequence, 6));
    finish_input(sequence, 6);

    ASSERT_TRUE(manager->allocate_sequence(&sequence, 10));
    EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 6u);
    ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
    const int32_t initial_id = sequence.get_linear_state_slot_id();
    const int32_t historical_id =
        sequence.kv_state().blocks(BlockType::LINEAR).front().id();
    EXPECT_NE(initial_id, historical_id);

    sequence.set_accepted_length(checkpoint_hit ? 1 : 0);
    sequence.kv_state().set_kv_cache_tokens_num(checkpoint_hit ? 8 : 7);
    ASSERT_TRUE(manager->allocate_sequence(&sequence, 12));
    if (checkpoint_hit) {
      // Cache 8 is the KV boundary, so the historical slot is retired.
      // The next speculative span (8 + 3 + 1) stays inside that block, so
      // the live slot is not paired with a new write slot.
      ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
      EXPECT_EQ(sequence.get_linear_state_slot_id(), initial_id);
    } else {
      ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
      EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
                initial_id);
      EXPECT_EQ(sequence.get_linear_state_slot_id(), historical_id);
      EXPECT_EQ(
          manager->leaf_entries().at(BlockType::LINEAR).leaf->num_used_blocks(),
          2u);
    }
    manager->deallocate_for_sequence(&sequence);
  }
}

TEST_P(LinearStateWindowTest, NullSequenceDoesNotAllocate) {
  LinearStateBlockManager manager(linear_options(/*num_slots=*/2,
                                                 /*block_size=*/2,
                                                 GetParam()));
  KVCacheState kv_state;
  EXPECT_FALSE(manager.allocate_for_sequence(nullptr, kv_state, 8).has_value());
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

TEST_P(LinearStateWindowTest, AllowsSpeculativeSpanAboveHalfStride) {
  LinearStateBlockManager manager(linear_options(
      /*num_slots=*/3,
      /*block_size=*/4,
      /*enable_prefix_cache=*/GetParam(),
      /*instance_is_decode=*/false,
      /*num_speculative_tokens=*/2));
  EXPECT_EQ(manager.options().num_speculative_tokens(), 2u);
}

TEST_P(LinearStateWindowTest, FlatPrefixMountRestoresCachedTokens) {
  BlockManager::Options options;
  options.num_blocks(32).block_size(2).enable_prefix_cache(GetParam());
  CompositeBlockManager manager(build_composite_leaves(options), options);
  Sequence producer = make_sequence(0, std::vector<int32_t>(8, 7));
  ASSERT_TRUE(manager.allocate_sequence(&producer, 8));
  finish_input(producer, 8);
  manager.deallocate_for_sequence(&producer);
  Sequence consumer = make_sequence(1, std::vector<int32_t>(8, 7));
  manager.allocate_shared_for_sequence(&consumer);
  EXPECT_EQ(consumer.kv_cache_tokens_num(), GetParam() ? 6u : 0u);
  manager.deallocate_for_sequence(&consumer);
}

TEST_P(LinearStateWindowTest, PrefixMountIsSparseAndMatchedOnlyOnce) {
  auto manager = make_manager();
  Sequence producer = make_sequence(0, std::vector<int32_t>(13, 7));
  for (const size_t target : {4, 8, 12, 13}) {
    ASSERT_TRUE(manager->allocate_sequence(&producer, target));
    finish_input(producer, target);
  }
  manager->deallocate_for_sequence(&producer);
  Sequence consumer = make_sequence(1, std::vector<int32_t>(13, 7));
  manager->allocate_shared_for_sequence(&consumer);
  EXPECT_EQ(consumer.kv_cache_tokens_num(), GetParam() ? 12u : 0u);
  EXPECT_EQ(consumer.kv_state().kv_cache_tokens_num(), GetParam() ? 12u : 0u);
  // A linear mount keeps one restore source, the deepest checkpoint.
  EXPECT_EQ(consumer.kv_state().num_blocks(BlockType::LINEAR),
            GetParam() ? 1u : 0u);
  const int32_t source_id = consumer.get_linear_state_slot_id();
  if (GetParam()) {
    const Slice<Block> blocks = consumer.kv_state().blocks(BlockType::LINEAR);
    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_TRUE(blocks[0].is_valid());
  }
  manager->allocate_shared_for_sequence(&consumer);
  EXPECT_EQ(consumer.get_linear_state_slot_id(), source_id);
  ASSERT_TRUE(manager->allocate_sequence(&consumer, GetParam() ? 13 : 4));
  manager->allocate_shared_for_sequence(&consumer);
  EXPECT_EQ(consumer.kv_state().num_blocks(BlockType::LINEAR),
            GetParam() ? 2u : 1u);
  if (GetParam()) {
    EXPECT_EQ(consumer.kv_state().blocks(BlockType::LINEAR).front().id(),
              source_id);
  }
  finish_input(consumer, GetParam() ? 13 : 4);
  EXPECT_EQ(consumer.kv_state().num_blocks(BlockType::LINEAR),
            GetParam() ? 2u : 1u);
  manager->deallocate_for_sequence(&consumer);
  consumer.reset();
  manager->allocate_shared_for_sequence(&consumer);
  EXPECT_EQ(consumer.kv_cache_tokens_num(), GetParam() ? 12u : 0u);
  EXPECT_EQ(consumer.kv_state().kv_cache_tokens_num(), GetParam() ? 12u : 0u);
  manager->deallocate_for_sequence(&consumer);
}

TEST_P(LinearStateWindowTest, DuplicateHashesStillAllocateDistinctOutputs) {
  auto manager = make_manager();
  Sequence first = make_sequence(0, std::vector<int32_t>(9, 7));
  Sequence second = make_sequence(1, std::vector<int32_t>(9, 7));
  manager->allocate_shared_for_sequence(&first);
  manager->allocate_shared_for_sequence(&second);
  ASSERT_TRUE(manager->allocate_sequence(&first, 4));
  ASSERT_TRUE(manager->allocate_sequence(&second, 4));
  const int32_t second_source_id = second.get_linear_state_slot_id();
  finish_input(first, 4);
  finish_input(second, 4);
  ASSERT_TRUE(manager->allocate_sequence(&first, 8));
  ASSERT_TRUE(manager->allocate_sequence(&second, 8));
  EXPECT_NE(second.get_linear_state_slot_id(), second_source_id);
  EXPECT_EQ(second.kv_state().blocks(BlockType::LINEAR)[0].id(),
            second_source_id);
  EXPECT_EQ(manager->leaf_entries()
                .at(BlockType::LINEAR)
                .leaf->num_blocks_in_prefix_cache(),
            GetParam() ? 1u : 0u);
  manager->deallocate_for_sequence(&first);
  manager->deallocate_for_sequence(&second);
}

TEST_P(LinearStateWindowTest, PrefixMissStaysMatchedUntilReset) {
  auto manager = make_manager();
  Sequence waiting = make_sequence(0, std::vector<int32_t>(9, 7));
  manager->allocate_shared_for_sequence(&waiting);
  EXPECT_EQ(waiting.kv_cache_tokens_num(), 0u);
  Sequence producer = make_sequence(1, std::vector<int32_t>(9, 7));
  for (const size_t target : {4, 8, 9}) {
    ASSERT_TRUE(manager->allocate_sequence(&producer, target));
    finish_input(producer, target);
  }
  manager->deallocate_for_sequence(&producer);
  manager->allocate_shared_for_sequence(&waiting);
  EXPECT_EQ(waiting.kv_cache_tokens_num(), 0u);
  ASSERT_TRUE(manager->allocate_sequence(&waiting, 4));
  EXPECT_EQ(waiting.kv_state().num_blocks(BlockType::LINEAR), 1u);
  manager->deallocate_for_sequence(&waiting);
  waiting.reset();
  manager->allocate_shared_for_sequence(&waiting);
  EXPECT_EQ(waiting.kv_cache_tokens_num(), GetParam() ? 8u : 0u);
  manager->deallocate_for_sequence(&waiting);
}

TEST_P(LinearStateWindowTest,
       OverlapPrefillKeepsTwoBlocksWithoutCompletionHooks) {
  auto manager = make_manager();
  Sequence sequence = make_sequence(0, std::vector<int32_t>(13, 7));
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
  finish_input(sequence, 4);
  const int32_t first_id = sequence.get_linear_state_slot_id();
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 8));
  const int32_t second_id = sequence.get_linear_state_slot_id();
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
            first_id);
  Batch second_batch(&sequence);
  sequence.kv_state().set_kv_cache_tokens_num(8);
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), second_id);
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 12));
  const int32_t third_id = sequence.get_linear_state_slot_id();
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
            second_id);
  EXPECT_NE(second_id, third_id);
  Batch third_batch(&sequence);
  sequence.kv_state().set_kv_cache_tokens_num(12);
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), third_id);
  if (!GetParam()) {
    for (const Block& block : sequence.kv_state().blocks(BlockType::LINEAR)) {
      EXPECT_EQ(block.ref_count(), 1u);
    }
  }
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), third_id);
  manager->deallocate_for_sequence(&sequence);
}

TEST_P(LinearStateWindowTest, PoolReleasesCancelledSequenceBeforeResults) {
  for (const bool discard : {false, true}) {
    BlockManagerPool::Options options;
    options.num_blocks(32)
        .block_size(2)
        .enable_linear_state(true)
        .linear_state_num_slots(8)
        .enable_prefix_cache(GetParam());
    BlockManagerPool pool(options, 1);
    Sequence sequence = make_sequence(0, std::vector<int32_t>(13, 7));
    ASSERT_TRUE(pool.allocate(&sequence, 4));
    std::vector<Batch> first_batch(1);
    first_batch.front().add(&sequence, 4);
    sequence.kv_state().set_kv_cache_tokens_num(4);
    ASSERT_TRUE(pool.allocate(&sequence, 8));
    std::vector<Batch> second_batch(1);
    second_batch.front().add(&sequence, 4);
    sequence.kv_state().set_kv_cache_tokens_num(8);
    sequence.set_cancel();
    if (discard) {
      pool.deallocate_without_cache(&sequence);
    } else {
      pool.deallocate(&sequence);
    }
    EXPECT_FALSE(sequence.has_any_blocks());
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 0u);
    EXPECT_EQ(sequence.kv_cache_tokens_num(), 0u);
    Sequence replacement = make_sequence(1, {9, 9, 9, 9});
    EXPECT_TRUE(pool.allocate(&replacement, 4));
    pool.deallocate_without_cache(&replacement);
  }
}

INSTANTIATE_TEST_SUITE_P(PrefixCacheModes,
                         LinearStateWindowTest,
                         ::testing::Bool());

namespace {

BlockManager::Options make_v41_prefix_options(uint32_t window_size = 128) {
  BlockManager::Options options = MakeCompositeOptions(
      /*base_num_blocks=*/128,
      /*block_size=*/kBaseBlockSize,
      window_size,
      /*max_seqs_per_batch=*/4);
  options.model_type("deepseek_v41").compress_ratios({0, 2, 1});
  return options;
}

BlockManagerPool::Options make_v41_prefix_pool_options() {
  const BlockManager::Options source = make_v41_prefix_options();
  BlockManagerPool::Options options;
  options.num_blocks(source.num_blocks())
      .block_size(static_cast<int32_t>(source.block_size()))
      .model_type(source.model_type())
      .sliding_window_size(source.sliding_window_size())
      .swa_blocks_per_seq(source.swa_blocks_per_seq())
      .swa_num_blocks(source.swa_num_blocks())
      .max_tokens_per_batch(source.max_tokens_per_batch())
      .max_seqs_per_batch(source.max_seqs_per_batch())
      .manager_types(source.manager_types())
      .compress_ratios(source.compress_ratios());
  return options;
}

using V41PrefixCacheTest = ::testing::Test;

void publish_v41_prompt(CompositeBlockManager& manager,
                        const std::vector<int32_t>& tokens) {
  Sequence cold = MakeTestSequence(/*index=*/0, tokens);
  ASSERT_TRUE(manager.allocate_sequence(&cold, tokens.size()));
  cold.kv_state().incr_kv_cache_tokens_num(tokens.size());
  manager.deallocate_for_sequence(&cold);
  cold.reset();
}

}  // namespace

TEST_F(V41PrefixCacheTest, V41PrefixCacheDefaultsOnLikeV4) {
  const BlockManager::Options options = make_v41_prefix_options();
  ASSERT_TRUE(options.enable_prefix_cache());
  CompositeBlockManager manager(build_composite_leaves(options), options);
  EXPECT_EQ(manager.leaf_combination(),
            CompositeBlockManager::LeafCombination::SWA_COMPRESSED);
  for (const BlockType type : {BlockType::SWA, BlockType::C1, BlockType::C2}) {
    EXPECT_TRUE(manager.leaf_entries().at(type).supports_prefix_cache);
    EXPECT_TRUE(
        manager.leaf_entries().at(type).leaf->options().enable_prefix_cache());
  }
  publish_v41_prompt(manager, std::vector<int32_t>(769, /*value=*/7));
  Sequence hot =
      MakeTestSequence(/*index=*/1, std::vector<int32_t>(769, /*value=*/7));
  manager.allocate_shared_for_sequence(&hot);
  EXPECT_EQ(hot.kv_cache_tokens_num(), 768u);
  manager.deallocate_for_sequence(&hot);
}

TEST_F(V41PrefixCacheTest, V41PrefixCacheFollowsV4EnableSwitch) {
  for (const bool enabled : {false, true}) {
    SCOPED_TRACE(enabled);
    BlockManager::Options options = make_v41_prefix_options();
    options.enable_prefix_cache(enabled);
    CompositeBlockManager manager(build_composite_leaves(options), options);
    for (const BlockType type :
         {BlockType::SWA, BlockType::C1, BlockType::C2}) {
      EXPECT_EQ(manager.leaf_entries().at(type).supports_prefix_cache, enabled);
      EXPECT_EQ(
          manager.leaf_entries().at(type).leaf->options().enable_prefix_cache(),
          enabled);
    }

    options.model_type("deepseek_v4").compress_ratios({0, 4, 128});
    CompositeBlockManager v4_manager(build_composite_leaves(options), options);
    EXPECT_EQ(manager.leaf_combination(), v4_manager.leaf_combination());
    for (const BlockType type :
         {BlockType::SWA, BlockType::C4, BlockType::C128}) {
      EXPECT_EQ(v4_manager.leaf_entries().at(type).supports_prefix_cache,
                enabled);
    }
  }
}

TEST_F(V41PrefixCacheTest, V41PrefixCacheKeepsDeviceOnlyRolePolicy) {
  BlockManager::Options options = make_v41_prefix_options();
  for (const bool is_decode : {false, true}) {
    SCOPED_TRACE(is_decode);
    options.instance_is_decode(is_decode);
    CompositeBlockManager manager(build_composite_leaves(options), options);
    for (const BlockType type :
         {BlockType::SWA, BlockType::C2, BlockType::C1}) {
      const BlockManager* leaf =
          BlockManagerPoolTestPeer::leaf_of(manager, type);
      ASSERT_NE(leaf, nullptr);
      EXPECT_EQ(leaf->options().enable_prefix_cache(), !is_decode);
    }
    EXPECT_EQ(manager.leaf_combination(),
              is_decode
                  ? CompositeBlockManager::LeafCombination::UNSUPPORTED
                  : CompositeBlockManager::LeafCombination::SWA_COMPRESSED);
  }
}

TEST_F(V41PrefixCacheTest, V41PrefixCacheRejectsMissingSwaWindow) {
  const BlockManager::Options options = make_v41_prefix_options();
  CompositeBlockManager::LeafMap leaves = build_composite_leaves(options);
  ASSERT_EQ(leaves.erase(BlockType::SWA), 1u);
  EXPECT_DEATH(
      { CompositeBlockManager manager(std::move(leaves), options); },
      "V4.1 prefix cache requires the SWA restore window.*C1=1 C2=1.*"
      "disable prefix cache");
}

TEST_F(V41PrefixCacheTest, V41PrefixCacheRejectsMissingHistoryLeaves) {
  const BlockManager::Options options = make_v41_prefix_options();
  CompositeBlockManager::LeafMap leaves = build_composite_leaves(options);
  ASSERT_EQ(leaves.erase(BlockType::C1), 1u);
  ASSERT_EQ(leaves.erase(BlockType::C2), 1u);
  EXPECT_DEATH(
      { CompositeBlockManager manager(std::move(leaves), options); },
      "SWA prefix cache requires a full-history TOKEN \\(C1/C2\\) or "
      "compressed \\(C4/C128\\) leaf, but none were found");
}

TEST_F(V41PrefixCacheTest, V41SingleTokenGroupUsesItsOwnRestoreUnit) {
  for (const uint32_t ratio : {1u, 2u}) {
    SCOPED_TRACE(ratio);
    BlockManager::Options options = make_v41_prefix_options();
    options
        .manager_types({kManagerTypeSlidingWindowBlockManager,
                        kManagerTypeBlockManagerImpl})
        .compress_ratios({0u, ratio});
    CompositeBlockManager manager(build_composite_leaves(options), options);
    ASSERT_EQ(manager.leaf_combination(),
              CompositeBlockManager::LeafCombination::SWA_COMPRESSED);
    const BlockType type = ratio == 1u ? BlockType::C1 : BlockType::C2;
    const size_t unit = kBaseBlockSize * ratio;
    const size_t prompt_tokens = 4 * unit;
    const std::vector<int32_t> tokens(prompt_tokens, /*value=*/7);
    publish_v41_prompt(manager, tokens);

    Sequence repeat = MakeTestSequence(/*index=*/1, tokens);
    manager.allocate_shared_for_sequence(&repeat);
    EXPECT_EQ(repeat.kv_cache_tokens_num(), prompt_tokens - unit);
    EXPECT_EQ(repeat.kv_state().shared_blocks_num(type), 3u);
    EXPECT_EQ(repeat.kv_state().shared_blocks_num(BlockType::SWA),
              (prompt_tokens - unit) / kBaseBlockSize);
    manager.deallocate_for_sequence(&repeat);

    std::vector<int32_t> extended = tokens;
    extended.emplace_back(11);
    Sequence suffix = MakeTestSequence(/*index=*/2, std::move(extended));
    manager.allocate_shared_for_sequence(&suffix);
    EXPECT_EQ(suffix.kv_cache_tokens_num(), prompt_tokens);
    EXPECT_EQ(suffix.kv_state().shared_blocks_num(type), 4u);
    manager.deallocate_for_sequence(&suffix);
  }
}

TEST_F(V41PrefixCacheTest, V41SampleSelectorsRetainRequiredHiddenPositions) {
  for (const size_t position : {0u, 1u, 256u, 257u, 512u, 513u}) {
    SCOPED_TRACE(position);
    const BlockManager::Options options = make_v41_prefix_options();
    CompositeBlockManager manager(build_composite_leaves(options), options);
    const std::vector<int32_t> prompt(515, /*value=*/7);
    publish_v41_prompt(manager, prompt);
    const std::vector<SampleSlot> slots = {{"sample", 0, 513},
                                           {"sample", 1, position}};
    Sequence selected = MakeTestSequence(/*index=*/1, prompt, &slots);
    manager.allocate_shared_for_sequence(&selected);
    const size_t source_position = position == 0 ? 0 : position - 1;
    const size_t expected = source_position / 256 * 256;
    EXPECT_EQ(selected.kv_cache_tokens_num(), expected);
    EXPECT_LE(selected.kv_cache_tokens_num(), source_position);
    EXPECT_EQ(selected.kv_state().shared_blocks_num(BlockType::C2),
              expected / 256);
    EXPECT_EQ(selected.kv_state().shared_blocks_num(BlockType::C1),
              expected / kBaseBlockSize);
    manager.deallocate_for_sequence(&selected);

    Sequence normal = MakeTestSequence(/*index=*/2, prompt);
    manager.allocate_shared_for_sequence(&normal);
    EXPECT_EQ(normal.kv_cache_tokens_num(), 512u);
    manager.deallocate_for_sequence(&normal);
  }
}

TEST_F(V41PrefixCacheTest, V41DisabledPrefixCacheDoesNotPublishOrMount) {
  BlockManager::Options options = make_v41_prefix_options();
  options.enable_prefix_cache(false);
  CompositeBlockManager manager(build_composite_leaves(options), options);
  publish_v41_prompt(manager, std::vector<int32_t>(/*count=*/769, /*value=*/7));
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);
  Sequence hot = MakeTestSequence(/*index=*/1, std::vector<int32_t>(769, 7));
  manager.allocate_shared_for_sequence(&hot);
  EXPECT_EQ(hot.kv_state().kv_cache_tokens_num(), 0u);
  manager.deallocate_for_sequence(&hot);
}

TEST_F(V41PrefixCacheTest, V41PrefixCacheUsesCommonTokenBoundary) {
  const std::vector<std::pair<size_t, size_t>> cases = {{127, 0},
                                                        {128, 0},
                                                        {255, 0},
                                                        {256, 0},
                                                        {257, 256},
                                                        {511, 256},
                                                        {512, 256},
                                                        {513, 512},
                                                        {768, 512},
                                                        {769, 768}};
  for (const auto& [prompt_tokens, expected_hit] : cases) {
    SCOPED_TRACE(prompt_tokens);
    const BlockManager::Options options = make_v41_prefix_options();
    CompositeBlockManager manager(build_composite_leaves(options), options);
    publish_v41_prompt(manager,
                       std::vector<int32_t>(prompt_tokens + 1, /*value=*/7));
    Sequence hot = MakeTestSequence(
        /*index=*/1, std::vector<int32_t>(prompt_tokens, /*value=*/7));
    manager.allocate_shared_for_sequence(&hot);
    EXPECT_EQ(hot.kv_state().kv_cache_tokens_num(), expected_hit);
    EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C1),
              expected_hit / kBaseBlockSize);
    EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C2),
              expected_hit / (2 * kBaseBlockSize));
    EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::SWA),
              expected_hit / kBaseBlockSize);
    manager.deallocate_for_sequence(&hot);
  }
}

TEST_F(V41PrefixCacheTest, V41PrefixCachePublishesOnlyCompletedUnits) {
  const BlockManager::Options options = make_v41_prefix_options();
  CompositeBlockManager manager(build_composite_leaves(options), options);
  Sequence cold = MakeTestSequence(/*index=*/0, std::vector<int32_t>(769, 7));
  ASSERT_TRUE(manager.allocate_sequence(&cold, /*num_tokens=*/769));
  cold.kv_state().incr_kv_cache_tokens_num(/*size=*/255);
  manager.cache_full_blocks_for_sequence(&cold);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);
  cold.kv_state().incr_kv_cache_tokens_num(/*size=*/1);
  manager.cache_full_blocks_for_sequence(&cold);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::C1)
                .leaf->num_blocks_in_prefix_cache(),
            2u);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::C2)
                .leaf->num_blocks_in_prefix_cache(),
            1u);
  Sequence hot = MakeTestSequence(/*index=*/1, std::vector<int32_t>(769, 7));
  manager.allocate_shared_for_sequence(&hot);
  EXPECT_EQ(hot.kv_state().kv_cache_tokens_num(), 256u);
  manager.deallocate_for_sequence(&hot);
  manager.deallocate_for_sequence(&cold);
}

TEST_F(V41PrefixCacheTest, V41ChunkedPrefillSharesPrefixWithStandardSwitch) {
  for (const bool enabled : {false, true}) {
    for (const size_t chunk_tokens : {128u, 384u}) {
      for (const uint32_t window_tokens : {128u, 512u}) {
        SCOPED_TRACE(enabled);
        SCOPED_TRACE(chunk_tokens);
        SCOPED_TRACE(window_tokens);
        BlockManager::Options options = make_v41_prefix_options(window_tokens);
        options.enable_prefix_cache(enabled);
        CompositeBlockManager manager(build_composite_leaves(options), options);
        const std::vector<int32_t> tokens(1025, /*value=*/7);
        Sequence cold = MakeTestSequence(/*index=*/0, tokens);
        for (size_t completed = 0; completed < tokens.size();) {
          const size_t target =
              std::min(completed + chunk_tokens, tokens.size());
          ASSERT_TRUE(manager.allocate_sequence(&cold, target));
          const size_t published_tokens = enabled ? completed / 256 * 256 : 0;
          EXPECT_EQ(cold.kv_state().num_cached_blocks(BlockType::C1),
                    published_tokens / 128);
          EXPECT_EQ(cold.kv_state().num_cached_blocks(BlockType::C2),
                    published_tokens / 256);
          cold.kv_state().set_kv_cache_tokens_num(target);
          completed = target;
        }
        manager.deallocate_for_sequence(&cold);
        cold.reset();

        const std::vector<int32_t> extended(1537, /*value=*/7);
        Sequence hot = MakeTestSequence(/*index=*/1, extended);
        manager.allocate_shared_for_sequence(&hot);
        const size_t expected_hit = enabled ? 1024 : 0;
        ASSERT_EQ(hot.kv_cache_tokens_num(), expected_hit);
        EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C1),
                  expected_hit / 128);
        EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C2),
                  expected_hit / 256);
        for (size_t completed = expected_hit; completed < extended.size();) {
          const size_t target =
              std::min(completed + chunk_tokens, extended.size());
          ASSERT_TRUE(manager.allocate_sequence(&hot, target));
          EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C1),
                    expected_hit / 128);
          EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C2),
                    expected_hit / 256);
          hot.kv_state().set_kv_cache_tokens_num(target);
          completed = target;
        }
        manager.deallocate_for_sequence(&hot);
      }
    }
  }
}

TEST_F(V41PrefixCacheTest, V41PrefixCacheUsesLargestParticipatingBlock) {
  const BlockManager::Options options = make_v41_prefix_options();
  CompositeBlockManager::LeafMap leaves = build_composite_leaves(options);
  const size_t unit_tokens = 4 * kBaseBlockSize;
  BlockManager::Options c1_options = leaves.at(BlockType::C1).leaf->options();
  c1_options.block_size(static_cast<uint32_t>(unit_tokens));
  leaves.at(BlockType::C1).leaf =
      std::make_unique<BlockManagerImpl>(c1_options);
  CompositeBlockManager manager(std::move(leaves), options);
  const std::vector<int32_t> tokens(2 * unit_tokens + 1, /*value=*/7);
  Sequence cold = MakeTestSequence(/*index=*/0, tokens);
  ASSERT_TRUE(manager.allocate_sequence(&cold, tokens.size()));
  cold.kv_state().incr_kv_cache_tokens_num(unit_tokens - 1);
  manager.cache_full_blocks_for_sequence(&cold);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);

  cold.kv_state().incr_kv_cache_tokens_num(/*size=*/1);
  manager.cache_full_blocks_for_sequence(&cold);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::C1)
                .leaf->num_blocks_in_prefix_cache(),
            1u);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::C2)
                .leaf->num_blocks_in_prefix_cache(),
            2u);

  Sequence hot = MakeTestSequence(/*index=*/1, tokens);
  manager.allocate_shared_for_sequence(&hot);
  EXPECT_EQ(hot.kv_cache_tokens_num(), unit_tokens);
  EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C1), 1u);
  EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C2), 2u);
  EXPECT_EQ(hot.kv_state().num_blocks(BlockType::SWA),
            unit_tokens / kBaseBlockSize);
  EXPECT_TRUE(hot.kv_state().blocks(BlockType::SWA).back().is_valid());
  manager.deallocate_for_sequence(&hot);
  manager.deallocate_for_sequence(&cold);
}

TEST_F(V41PrefixCacheTest, V41PrefixCacheMissesIfAnyRequiredGroupIsGone) {
  for (const BlockType missing :
       {BlockType::SWA, BlockType::C2, BlockType::C1}) {
    SCOPED_TRACE(static_cast<int32_t>(missing));
    const BlockManager::Options options = make_v41_prefix_options();
    CompositeBlockManager manager(build_composite_leaves(options), options);
    publish_v41_prompt(manager, std::vector<int32_t>(769, 7));
    BlockManagerPoolTestPeer::leaf_of(manager, missing)->reset_prefix_cache();
    Sequence hot = MakeTestSequence(/*index=*/1, std::vector<int32_t>(769, 7));
    manager.allocate_shared_for_sequence(&hot);
    EXPECT_EQ(hot.kv_state().kv_cache_tokens_num(), 0u);
    for (const BlockType type :
         {BlockType::SWA, BlockType::C2, BlockType::C1}) {
      EXPECT_EQ(hot.kv_state().shared_blocks_num(type), 0u);
    }
    manager.deallocate_for_sequence(&hot);
  }
}

TEST_F(V41PrefixCacheTest, V41PrefixCachePreservesSharedBlocksOnContinuation) {
  const BlockManager::Options options = make_v41_prefix_options();
  CompositeBlockManager manager(build_composite_leaves(options), options);
  publish_v41_prompt(manager, std::vector<int32_t>(769, 7));
  Sequence first = MakeTestSequence(/*index=*/1, std::vector<int32_t>(769, 7));
  Sequence second = MakeTestSequence(/*index=*/2, std::vector<int32_t>(770, 7));
  manager.allocate_shared_for_sequence(&first);
  manager.allocate_shared_for_sequence(&second);
  ASSERT_EQ(first.kv_state().kv_cache_tokens_num(), 768u);
  ASSERT_EQ(second.kv_state().kv_cache_tokens_num(), 768u);
  const int32_t shared_swa_id =
      first.kv_state().blocks(BlockType::SWA).back().id();
  EXPECT_EQ(second.kv_state().blocks(BlockType::SWA).back().id(),
            shared_swa_id);
  ASSERT_TRUE(manager.allocate_sequence(&first, /*num_tokens=*/769));
  ASSERT_TRUE(manager.allocate_sequence(&second, /*num_tokens=*/770));
  EXPECT_NE(first.kv_state().blocks(BlockType::SWA).back().id(), shared_swa_id);
  EXPECT_NE(second.kv_state().blocks(BlockType::SWA).back().id(),
            shared_swa_id);
  EXPECT_NE(first.kv_state().blocks(BlockType::SWA).back().id(),
            second.kv_state().blocks(BlockType::SWA).back().id());
  for (const BlockType type : {BlockType::C1, BlockType::C2}) {
    const Slice<Block> a = first.kv_state().blocks(type);
    const Slice<Block> b = second.kv_state().blocks(type);
    const size_t shared = first.kv_state().shared_blocks_num(type);
    for (size_t i = 0; i < shared; ++i) {
      EXPECT_EQ(a[i].id(), b[i].id());
    }
    EXPECT_NE(a.back().id(), b.back().id());
  }
  manager.allocate_shared_for_sequence(&first);
  EXPECT_EQ(first.kv_state().kv_cache_tokens_num(), 768u);
  first.kv_state().incr_kv_cache_tokens_num(/*size=*/1);
  manager.allocate_shared_for_sequence(&first);
  EXPECT_EQ(first.kv_state().kv_cache_tokens_num(), 769u);
  manager.deallocate_for_sequence(&first);
  manager.deallocate_for_sequence(&second);
}

TEST_F(V41PrefixCacheTest, V41PrefixCacheRequiresCompleteSwaWindow) {
  for (const uint32_t window_size : {128u, 256u, 512u}) {
    SCOPED_TRACE(window_size);
    const BlockManager::Options options = make_v41_prefix_options(window_size);
    CompositeBlockManager manager(build_composite_leaves(options), options);
    publish_v41_prompt(manager, std::vector<int32_t>(769, 7));
    Sequence hot = MakeTestSequence(/*index=*/1, std::vector<int32_t>(513, 7));
    manager.allocate_shared_for_sequence(&hot);
    ASSERT_EQ(hot.kv_state().kv_cache_tokens_num(), 512u);
    const Slice<Block> swa = hot.kv_state().blocks(BlockType::SWA);
    ASSERT_GE(swa.size(), options.swa_blocks_per_seq());
    for (size_t i = swa.size() - options.swa_blocks_per_seq(); i < swa.size();
         ++i) {
      EXPECT_TRUE(swa[i].is_valid());
    }
    manager.deallocate_for_sequence(&hot);
  }
}

TEST_F(V41PrefixCacheTest, V41ShortPrefixRestoresAvailableSwaHistory) {
  for (const bool c1_only : {false, true}) {
    SCOPED_TRACE(c1_only);
    const uint32_t window_size = c1_only ? 256u : 512u;
    const size_t prefix_tokens = c1_only ? 128u : 256u;
    BlockManager::Options options = make_v41_prefix_options(window_size);
    if (c1_only) {
      options
          .manager_types({kManagerTypeSlidingWindowBlockManager,
                          kManagerTypeBlockManagerImpl})
          .compress_ratios({0u, 1u});
    }
    CompositeBlockManager manager(build_composite_leaves(options), options);
    publish_v41_prompt(manager, std::vector<int32_t>(prefix_tokens, 7));
    Sequence hot = MakeTestSequence(
        /*index=*/1, std::vector<int32_t>(prefix_tokens + 1, 7));
    manager.allocate_shared_for_sequence(&hot);
    EXPECT_EQ(hot.kv_cache_tokens_num(), prefix_tokens);
    const Slice<Block> swa = hot.kv_state().blocks(BlockType::SWA);
    ASSERT_EQ(swa.size(), prefix_tokens / kBaseBlockSize);
    for (const Block& block : swa) {
      EXPECT_TRUE(block.is_valid());
    }
    EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C1),
              prefix_tokens / kBaseBlockSize);
    if (!c1_only) {
      EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C2), 1u);
    }
    ASSERT_TRUE(manager.allocate_sequence(&hot, prefix_tokens + 1));
    manager.deallocate_for_sequence(&hot);
  }
}

TEST_F(V41PrefixCacheTest, V41PrefixCacheRejectsHoleInsideRequiredSwaTail) {
  const BlockManager::Options options = make_v41_prefix_options(512);
  CompositeBlockManager manager(build_composite_leaves(options), options);
  const std::vector<int32_t> tokens(769, 7);
  Sequence producer = MakeTestSequence(/*index=*/0, tokens);
  ASSERT_TRUE(manager.allocate_sequence(&producer, tokens.size()));
  producer.kv_state().set_kv_cache_tokens_num(tokens.size());
  manager.cache_full_blocks_for_sequence(&producer);
  const Slice<Block> swa = producer.kv_state().blocks(BlockType::SWA);
  ASSERT_GE(swa.size(), 6u);
  std::vector<Block> retained;
  for (const size_t index : {0u, 1u, 2u, 4u, 5u}) {
    retained.push_back(swa[index]);
  }
  manager.deallocate_for_sequence(&producer);
  BlockManager* swa_leaf =
      BlockManagerPoolTestPeer::leaf_of(manager, BlockType::SWA);
  ASSERT_NE(swa_leaf, nullptr);
  swa_leaf->reset_prefix_cache();
  swa_leaf->cache(retained);
  swa_leaf->deallocate(std::move(retained));

  Sequence hot = MakeTestSequence(/*index=*/1, tokens);
  manager.allocate_shared_for_sequence(&hot);
  // Both the 768- and 512-token windows need the missing fourth SWA block.
  // The first 256 tokens have complete history and remain recoverable.
  EXPECT_EQ(hot.kv_cache_tokens_num(), 256u);
  EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::SWA), 2u);
  EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C1), 2u);
  EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C2), 1u);
  manager.deallocate_for_sequence(&hot);
}

TEST_F(V41PrefixCacheTest, V41PrefixCacheMissesForDifferentTokenPrefix) {
  const BlockManager::Options options = make_v41_prefix_options();
  CompositeBlockManager manager(build_composite_leaves(options), options);
  publish_v41_prompt(manager, std::vector<int32_t>(769, /*value=*/7));
  Sequence unrelated = MakeTestSequence(
      /*index=*/1, std::vector<int32_t>(769, /*value=*/11));
  manager.allocate_shared_for_sequence(&unrelated);
  EXPECT_EQ(unrelated.kv_state().kv_cache_tokens_num(), 0u);
  for (const BlockType type : {BlockType::SWA, BlockType::C1, BlockType::C2}) {
    EXPECT_EQ(unrelated.kv_state().shared_blocks_num(type), 0u);
  }
  manager.deallocate_for_sequence(&unrelated);
}

TEST_F(V41PrefixCacheTest, V41PrefixCacheRecomputesDivergentSuffix) {
  const BlockManager::Options options = make_v41_prefix_options();
  CompositeBlockManager manager(build_composite_leaves(options), options);
  publish_v41_prompt(manager, std::vector<int32_t>(769, /*value=*/7));
  std::vector<int32_t> tokens(769, /*value=*/7);
  tokens[600] = 11;
  Sequence branch = MakeTestSequence(/*index=*/1, tokens);
  manager.allocate_shared_for_sequence(&branch);
  EXPECT_EQ(branch.kv_state().kv_cache_tokens_num(), 512u);
  ASSERT_TRUE(manager.allocate_sequence(&branch, tokens.size()));
  EXPECT_EQ(branch.kv_state().shared_blocks_num(BlockType::C1), 4u);
  EXPECT_EQ(branch.kv_state().shared_blocks_num(BlockType::C2), 2u);
  manager.deallocate_for_sequence(&branch);
}

TEST_F(V41PrefixCacheTest, V41CancelledSharedAliasReleasesAllLeafUsage) {
  for (const bool discard : {false, true}) {
    for (const bool reverse_order : {false, true}) {
      SCOPED_TRACE(discard);
      SCOPED_TRACE(reverse_order);
      BlockManagerPool pool(make_v41_prefix_pool_options(), /*dp_size=*/1);
      const auto leaf_usage = [&pool]() {
        std::vector<size_t> usage;
        for (const BlockType type :
             {BlockType::SWA, BlockType::C1, BlockType::C2}) {
          usage.push_back(
              BlockManagerPoolTestPeer::leaf_of(pool, /*dp_rank=*/0, type)
                  ->num_used_blocks());
        }
        return usage;
      };
      const std::vector<size_t> initial_usage = leaf_usage();
      const std::vector<int32_t> tokens(769, /*value=*/7);
      Sequence producer = MakeTestSequence(/*index=*/0, tokens);
      ASSERT_TRUE(pool.allocate(&producer, tokens.size()));
      producer.kv_state().set_kv_cache_tokens_num(tokens.size());
      pool.deallocate(&producer);
      EXPECT_EQ(leaf_usage(), initial_usage);

      Sequence first = MakeTestSequence(/*index=*/1, tokens);
      Sequence second = MakeTestSequence(/*index=*/2, tokens);
      ASSERT_TRUE(pool.allocate(&first, tokens.size()));
      ASSERT_TRUE(pool.allocate(&second, tokens.size()));
      ASSERT_EQ(first.kv_cache_tokens_num(), 768u);
      ASSERT_EQ(second.kv_cache_tokens_num(), 768u);
      std::vector<int32_t> shared_ids;
      for (const BlockType type :
           {BlockType::SWA, BlockType::C1, BlockType::C2}) {
        const Slice<Block> a = first.kv_state().blocks(type);
        const Slice<Block> b = second.kv_state().blocks(type);
        const size_t shared = first.kv_state().shared_blocks_num(type);
        ASSERT_GT(shared, 0u);
        ASSERT_GE(b.size(), shared);
        EXPECT_EQ(a[shared - 1].id(), b[shared - 1].id());
        shared_ids.push_back(a[shared - 1].id());
      }
      const auto release = [&pool, discard](Sequence* sequence) {
        sequence->set_cancel();
        if (discard) {
          pool.deallocate_without_cache(sequence);
        } else {
          pool.deallocate(sequence);
        }
        EXPECT_FALSE(sequence->has_any_blocks());
        EXPECT_EQ(sequence->kv_cache_tokens_num(), 0u);
      };
      Sequence* released = reverse_order ? &second : &first;
      Sequence* survivor = reverse_order ? &first : &second;
      release(released);
      size_t group = 0;
      for (const BlockType type :
           {BlockType::SWA, BlockType::C1, BlockType::C2}) {
        const Slice<Block> blocks = survivor->kv_state().blocks(type);
        const size_t shared = survivor->kv_state().shared_blocks_num(type);
        ASSERT_GT(shared, 0u);
        EXPECT_TRUE(blocks[shared - 1].is_valid());
        EXPECT_EQ(blocks[shared - 1].id(), shared_ids[group]);
        EXPECT_GT(leaf_usage()[group], initial_usage[group]);
        ++group;
      }
      release(survivor);
      EXPECT_EQ(leaf_usage(), initial_usage);

      Sequence revisit = MakeTestSequence(/*index=*/3, tokens);
      ASSERT_TRUE(pool.allocate(&revisit, tokens.size()));
      EXPECT_EQ(revisit.kv_cache_tokens_num(), 768u);
      pool.deallocate(&revisit);
      EXPECT_EQ(leaf_usage(), initial_usage);
      pool.reset_prefix_cache();
      EXPECT_EQ(leaf_usage(), initial_usage);
    }
  }
}

TEST_F(V41PrefixCacheTest,
       V41CancelledPartialForwardDoesNotPublishUnfinishedUnit) {
  BlockManagerPool pool(make_v41_prefix_pool_options(), /*dp_size=*/1);
  const std::vector<size_t> initial_usage = pool.num_used_blocks();
  const std::vector<int32_t> tokens(769, /*value=*/7);
  Sequence cancelled = MakeTestSequence(/*index=*/0, tokens);
  ASSERT_TRUE(pool.allocate(&cancelled, /*num_tokens=*/384));
  cancelled.kv_state().set_kv_cache_tokens_num(257);
  cancelled.set_cancel();
  pool.deallocate(&cancelled);
  EXPECT_EQ(pool.num_used_blocks(), initial_usage);

  Sequence revisit = MakeTestSequence(/*index=*/1, tokens);
  pool.allocate_shared(&revisit);
  EXPECT_EQ(revisit.kv_cache_tokens_num(), 256u);
  EXPECT_EQ(revisit.kv_state().shared_blocks_num(BlockType::C1), 2u);
  EXPECT_EQ(revisit.kv_state().shared_blocks_num(BlockType::C2), 1u);
  pool.deallocate(&revisit);
  pool.reset_prefix_cache();
  EXPECT_EQ(pool.num_used_blocks(), initial_usage);
}

TEST_F(V41PrefixCacheTest, V41PrefixPoolsDoNotSharePublishedEntries) {
  BlockManagerPool pool(make_v41_prefix_pool_options(), /*dp_size=*/2);
  const std::vector<int32_t> tokens(769, /*value=*/7);
  Sequence producer = MakeTestSequence(/*index=*/0, tokens);
  producer.set_dp_rank(0);
  ASSERT_TRUE(pool.allocate(&producer, tokens.size()));
  producer.kv_state().set_kv_cache_tokens_num(tokens.size());
  pool.deallocate(&producer);

  Sequence other_pool = MakeTestSequence(/*index=*/1, tokens);
  other_pool.set_dp_rank(1);
  pool.allocate_shared(&other_pool);
  EXPECT_EQ(other_pool.kv_cache_tokens_num(), 0u);
  pool.deallocate(&other_pool);

  Sequence same_pool = MakeTestSequence(/*index=*/2, tokens);
  same_pool.set_dp_rank(0);
  pool.allocate_shared(&same_pool);
  EXPECT_EQ(same_pool.kv_cache_tokens_num(), 768u);
  pool.deallocate(&same_pool);

  Sequence second_producer = MakeTestSequence(/*index=*/3, tokens);
  second_producer.set_dp_rank(1);
  ASSERT_TRUE(pool.allocate(&second_producer, tokens.size()));
  second_producer.kv_state().set_kv_cache_tokens_num(tokens.size());
  pool.deallocate(&second_producer);
  for (const int32_t dp_rank : {0, 1}) {
    Sequence warmed = MakeTestSequence(/*index=*/4, tokens);
    warmed.set_dp_rank(dp_rank);
    pool.allocate_shared(&warmed);
    EXPECT_EQ(warmed.kv_cache_tokens_num(), 768u);
    pool.deallocate(&warmed);
  }
}

TEST_F(V41PrefixCacheTest, V41PressureEvictsPrefixAndAllowsRepublication) {
  BlockManager::Options options = make_v41_prefix_options();
  options.num_blocks(10);
  CompositeBlockManager manager(build_composite_leaves(options), options);
  const size_t initial_usage = manager.num_used_blocks();
  const std::vector<int32_t> first_tokens(769, /*value=*/7);
  publish_v41_prompt(manager, first_tokens);
  Sequence first_hit = MakeTestSequence(/*index=*/1, first_tokens);
  manager.allocate_shared_for_sequence(&first_hit);
  ASSERT_EQ(first_hit.kv_cache_tokens_num(), 768u);
  manager.deallocate_for_sequence(&first_hit);

  for (const int32_t token : {11, 13, 17, 19}) {
    SCOPED_TRACE(token);
    publish_v41_prompt(manager, std::vector<int32_t>(769, token));
    EXPECT_EQ(manager.num_used_blocks(), initial_usage);
  }
  Sequence evicted = MakeTestSequence(/*index=*/2, first_tokens);
  manager.allocate_shared_for_sequence(&evicted);
  EXPECT_EQ(evicted.kv_cache_tokens_num(), 0u);
  manager.deallocate_for_sequence(&evicted);
  publish_v41_prompt(manager, first_tokens);
  Sequence restored = MakeTestSequence(/*index=*/3, first_tokens);
  manager.allocate_shared_for_sequence(&restored);
  EXPECT_EQ(restored.kv_cache_tokens_num(), 768u);
  manager.deallocate_for_sequence(&restored);
  manager.reset_prefix_cache();
  EXPECT_EQ(manager.num_used_blocks(), initial_usage);
}

TEST_F(V41PrefixCacheTest, V41PrefixCacheRejectsMultimodalHasher) {
  BlockManager::Options options = make_v41_prefix_options();
  options.hasher_type(BlockHasherType::MM);
  EXPECT_DEATH(build_composite_leaves(options), "text prompts only");
}

TEST_F(V41PrefixCacheTest, TokenRatiosDoNotImplyV41Identity) {
  BlockManager::Options options = make_v41_prefix_options();
  options.model_type("other_token_model");
  CompositeBlockManager manager(build_composite_leaves(options), options);
  EXPECT_EQ(manager.leaf_combination(),
            CompositeBlockManager::LeafCombination::SWA_COMPRESSED);
  for (const BlockType type : {BlockType::SWA, BlockType::C1, BlockType::C2}) {
    EXPECT_TRUE(manager.leaf_entries().at(type).supports_prefix_cache);
  }
}

namespace {

using V41SpecPrefixCacheTest = V41PrefixCacheTest;

}  // namespace

TEST_F(V41SpecPrefixCacheTest, SpecKeepsAllPrefixLeavesEnabled) {
  BlockManager::Options options = make_v41_prefix_options();
  options.num_speculative_tokens(5);
  CompositeBlockManager manager(build_composite_leaves(options), options);
  EXPECT_EQ(manager.leaf_combination(),
            CompositeBlockManager::LeafCombination::SWA_COMPRESSED);
  for (const BlockType type : {BlockType::SWA, BlockType::C1, BlockType::C2}) {
    EXPECT_TRUE(BlockManagerPoolTestPeer::leaf_of(manager, type)
                    ->options()
                    .enable_prefix_cache());
  }
}

TEST_F(V41SpecPrefixCacheTest, SpecRejectsNonTextHasher) {
  BlockManager::Options options = make_v41_prefix_options();
  options.num_speculative_tokens(5).hasher_type(BlockHasherType::MTP_TEXT);
  EXPECT_DEATH(build_composite_leaves(options), "text prompts only");
  options.hasher_type(BlockHasherType::MM);
  EXPECT_DEATH(build_composite_leaves(options), "text prompts only");
}

TEST_F(V41SpecPrefixCacheTest, EnableSwitchDisablesSpecPrefix) {
  BlockManager::Options options = make_v41_prefix_options();
  options.num_speculative_tokens(5).enable_prefix_cache(false);
  CompositeBlockManager manager(build_composite_leaves(options), options);
  EXPECT_EQ(manager.leaf_combination(),
            CompositeBlockManager::LeafCombination::UNSUPPORTED);
}

TEST_F(V41SpecPrefixCacheTest, VerifyAllocationPublishesOnlyAcceptedUnits) {
  for (const size_t accepted : {1u, 2u, 5u}) {
    SCOPED_TRACE(accepted);
    BlockManager::Options options = make_v41_prefix_options();
    options.num_speculative_tokens(5);
    CompositeBlockManager manager(build_composite_leaves(options), options);
    const size_t initial_usage = manager.num_used_blocks();
    std::vector<int32_t> prompt(1021, 7);
    publish_v41_prompt(manager, prompt);
    Sequence sequence = MakeTestSequence(1, prompt);
    manager.allocate_shared_for_sequence(&sequence);
    ASSERT_EQ(sequence.kv_cache_tokens_num(), 768u);
    ASSERT_TRUE(manager.allocate_sequence(&sequence, prompt.size()));
    sequence.kv_state().set_kv_cache_tokens_num(prompt.size());
    sequence.append_token(11);  // Anchor at 1021 is not yet computed.
    std::vector<int32_t> shared_ids;
    for (const BlockType type : {BlockType::C1, BlockType::C2}) {
      shared_ids.push_back(sequence.kv_state().blocks(type).front().id());
    }
    // V4's admission already reserves anchor + drafts. Merely allocating
    // across 1024 must not publish the rejected/uncomputed final C2 unit.
    ASSERT_TRUE(manager.allocate_sequence(&sequence, 1027));
    EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::C1), 6u);
    EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::C2), 3u);
    for (size_t index = 0; index < accepted; ++index) {
      sequence.append_token(static_cast<int64_t>(20 + index));
    }
    // append_token advances the committed cursor, as the shared V4
    // acceptance path does; none of the rejected proposal IDs enter tokens().
    manager.cache_for_sequence(&sequence);
    const size_t safe_tokens = (sequence.kv_cache_tokens_num() / 256) * 256;
    EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::C1),
              safe_tokens / 128);
    EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::C2),
              safe_tokens / 256);
    size_t leaf_index = 0;
    for (const BlockType type : {BlockType::C1, BlockType::C2}) {
      EXPECT_EQ(sequence.kv_state().blocks(type).front().id(),
                shared_ids[leaf_index++]);
    }
    const Slice<int32_t> confirmed = sequence.tokens();
    std::vector<int32_t> continuation(confirmed.begin(), confirmed.end());
    continuation.push_back(63);
    manager.deallocate_for_sequence(&sequence);
    Sequence repeat = MakeTestSequence(2, continuation);
    manager.allocate_shared_for_sequence(&repeat);
    EXPECT_EQ(repeat.kv_cache_tokens_num(), safe_tokens);
    manager.deallocate_for_sequence(&repeat);
    manager.reset_prefix_cache();
    EXPECT_EQ(manager.num_used_blocks(), initial_usage);
  }
}

TEST_F(V41SpecPrefixCacheTest,
       CancelledVerifyDoesNotPublishUnacceptedBoundary) {
  BlockManagerPool::Options options = make_v41_prefix_pool_options();
  options.num_speculative_tokens(5).num_embedding_blocks(8);
  for (const bool discard : {false, true}) {
    SCOPED_TRACE(discard);
    BlockManagerPool pool(options, 1);
    Sequence source = MakeTestSequence(0, std::vector<int32_t>(1021, 7));
    ASSERT_TRUE(pool.allocate(&source, 1021));
    source.kv_state().set_kv_cache_tokens_num(1021);
    pool.deallocate(&source);
    Sequence cancelled = MakeTestSequence(1, std::vector<int32_t>(1021, 7));
    ASSERT_TRUE(pool.allocate(&cancelled, 1021));
    ASSERT_EQ(cancelled.kv_cache_tokens_num(), 768u);
    cancelled.kv_state().set_kv_cache_tokens_num(1021);
    cancelled.append_token(11);
    ASSERT_TRUE(pool.allocate(&cancelled, 1027));
    cancelled.set_cancel();
    if (discard) {
      pool.deallocate_without_cache(&cancelled);
    } else {
      pool.deallocate(&cancelled);
    }
    EXPECT_FALSE(cancelled.has_any_blocks());
    Sequence repeat = MakeTestSequence(2, std::vector<int32_t>(1027, 7));
    ASSERT_TRUE(pool.allocate(&repeat, 1027));
    EXPECT_EQ(repeat.kv_cache_tokens_num(), 768u);
    pool.deallocate_without_cache(&repeat);
  }
}

TEST_F(V41SpecPrefixCacheTest, SharedSpecRequestsKeepIndependentEmbeddingRows) {
  BlockManagerPool::Options options = make_v41_prefix_pool_options();
  options.num_speculative_tokens(5).num_embedding_blocks(8);
  BlockManagerPool pool(options, 1);
  const std::vector<int32_t> tokens(769, 7);
  Sequence source = MakeTestSequence(0, tokens);
  ASSERT_TRUE(pool.allocate(&source, 769));
  source.kv_state().set_kv_cache_tokens_num(769);
  pool.deallocate(&source);
  Sequence first = MakeTestSequence(1, tokens);
  Sequence second = MakeTestSequence(2, tokens);
  ASSERT_TRUE(pool.allocate(&first, 769));
  ASSERT_TRUE(pool.allocate(&second, 769));
  ASSERT_EQ(first.kv_cache_tokens_num(), 768u);
  ASSERT_EQ(second.kv_cache_tokens_num(), 768u);
  EXPECT_NE(first.kv_state().blocks(BlockType::EMBEDDING)[0].id(),
            second.kv_state().blocks(BlockType::EMBEDDING)[0].id());
  for (const BlockType type : {BlockType::C1, BlockType::C2}) {
    EXPECT_EQ(first.kv_state().blocks(type)[0].id(),
              second.kv_state().blocks(type)[0].id());
    EXPECT_NE(first.kv_state().blocks(type).back().id(),
              second.kv_state().blocks(type).back().id());
  }
  first.set_cancel();
  pool.deallocate_without_cache(&first);
  EXPECT_FALSE(first.has_any_blocks());
  EXPECT_EQ(second.kv_cache_tokens_num(), 768u);
  ASSERT_TRUE(pool.allocate(&second, 775));
  pool.deallocate_without_cache(&second);
  EXPECT_FALSE(second.has_any_blocks());
}

TEST(CompositeBlockManagerTest, V4DefaultPrefixCacheKeepsRolePolicy) {
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      /*block_size=*/kBaseBlockSize,
      /*window_size=*/4 * kBaseBlockSize,
      /*max_seqs_per_batch=*/4);
  ASSERT_TRUE(opts.enable_prefix_cache());
  for (const bool is_decode : {false, true}) {
    SCOPED_TRACE(is_decode);
    opts.instance_is_decode(is_decode);
    CompositeBlockManager manager(build_composite_leaves(opts), opts);
    const BlockManager* swa =
        BlockManagerPoolTestPeer::leaf_of(manager, BlockType::SWA);
    ASSERT_NE(swa, nullptr);
    EXPECT_EQ(swa->options().enable_prefix_cache(), !is_decode);
    for (const BlockType type : {BlockType::C4, BlockType::C128}) {
      const BlockManager* leaf =
          BlockManagerPoolTestPeer::leaf_of(manager, type);
      ASSERT_NE(leaf, nullptr);
      EXPECT_TRUE(leaf->options().enable_prefix_cache());
    }
  }
}

}  // namespace xllm
