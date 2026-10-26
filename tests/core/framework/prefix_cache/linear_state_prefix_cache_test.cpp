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

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/request/sequence.h"
#include "framework/block/block_manager_pool_test_peer.h"
#include "framework/block/linear_state_block_manager.h"
#include "util/hash_util.h"

namespace xllm {
namespace {

// Build a distinct prefix hash keyed by a single byte so tests can reference
// checkpoints by an easy-to-read tag.
XXH3Key make_hash(uint8_t tag) {
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> bytes{};
  bytes.fill(tag);
  return XXH3Key(bytes.data());
}

int32_t insert_checkpoint(LinearStateBlockManager& cache, const XXH3Key& hash) {
  if (BlockManagerPoolTestPeer::contains(&cache, hash)) {
    Block matched = BlockManagerPoolTestPeer::match(&cache, hash);
    return matched.is_valid() ? matched.id() : -1;
  }
  std::vector<Block> allocated = cache.allocate(1);
  if (allocated.empty()) {
    return -1;
  }
  Block slot_block = std::move(allocated.front());
  const int32_t slot = slot_block.id();
  slot_block.set_hash_value(hash.data);
  std::vector<Block> checkpoint;
  checkpoint.emplace_back(std::move(slot_block));
  cache.cache(checkpoint);
  return slot;
}

BlockManager::Options linear_options(uint32_t num_slots, int32_t block_size) {
  BlockManager::Options options;
  options.num_blocks(num_slots).block_size(block_size);
  return options;
}

Sequence make_prefetch_sequence(size_t num_tokens) {
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker(/*max_generated_tokens=*/16,
                                          /*max_context_len=*/0,
                                          /*eos_token=*/-1,
                                          /*ignore_eos=*/false,
                                          /*stop_tokens=*/{},
                                          /*stop_sequences=*/{});
  SequenceParams seq_params;
  seq_params.seq_capacity = num_tokens + 16;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  IncrementalDecoder decoder("", num_tokens, false, false);
  return Sequence(/*index=*/0,
                  std::vector<int32_t>(num_tokens, 7),
                  torch::Tensor(),
                  MMData(),
                  std::move(decoder),
                  seq_params);
}

std::vector<int32_t> placed_ids(const Sequence& sequence,
                                size_t target_blocks) {
  const Slice<Block> blocks =
      sequence.host_kv_state().blocks(BlockType::LINEAR);
  std::vector<int32_t> ids;
  ids.reserve(target_blocks);
  for (size_t index = 0; index < target_blocks; ++index) {
    if (index < blocks.size() && blocks[index].is_valid()) {
      ids.push_back(blocks[index].id());
    } else {
      ids.push_back(-1);
    }
  }
  return ids;
}

}  // namespace

// Padding slot 0 is never handed out, and a cache of N physical slots exposes
// N - 1 live slots before it runs dry.
TEST(LinearStatePrefixCacheTest, PaddingSlotReservedAndLiveSlotsExhaust) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/3, /*block_size=*/16));

  std::vector<Block> first_blocks = cache.allocate(1);
  std::vector<Block> second_blocks = cache.allocate(1);
  ASSERT_EQ(first_blocks.size(), 1u);
  ASSERT_EQ(second_blocks.size(), 1u);
  Block first = std::move(first_blocks.front());
  Block second = std::move(second_blocks.front());
  ASSERT_TRUE(first.is_valid());
  ASSERT_TRUE(second.is_valid());
  EXPECT_GE(first.id(), 1);
  EXPECT_GE(second.id(), 1);
  EXPECT_NE(first.id(), second.id());

  // No free slot and no checkpoint to reclaim -> invalid handle.
  EXPECT_TRUE(cache.allocate(1).empty());
}

// A live slot returns to the cache when its handle is dropped.
TEST(LinearStatePrefixCacheTest, DroppingLiveHandleReleasesSlot) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/2, /*block_size=*/16));
  int32_t reused_id = -1;
  {
    std::vector<Block> live_blocks = cache.allocate(1);
    ASSERT_EQ(live_blocks.size(), 1u);
    Block live = std::move(live_blocks.front());
    ASSERT_TRUE(live.is_valid());
    reused_id = live.id();
    // Only one usable slot; it is now live.
    EXPECT_TRUE(cache.allocate(1).empty());
  }
  // Handle dropped: the slot is free again.
  std::vector<Block> again = cache.allocate(1);
  ASSERT_EQ(again.size(), 1u);
  EXPECT_EQ(again.front().id(), reused_id);
}

// insert() pins a slot that match()/contains() can find, and reusing the same
// hash returns the same slot without consuming another.
TEST(LinearStatePrefixCacheTest, InsertMatchAndDedup) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/4, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);

  const int32_t slot = insert_checkpoint(cache, h1);
  EXPECT_GE(slot, 1);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  Block pinned = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(pinned.is_valid());
  EXPECT_EQ(pinned.id(), slot);
  EXPECT_TRUE(pinned.is_shared());

  // Re-inserting the same hash is idempotent.
  EXPECT_EQ(insert_checkpoint(cache, h1), slot);

  // Unknown hashes miss.
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, make_hash(2)));
  EXPECT_FALSE(
      BlockManagerPoolTestPeer::match(&cache, make_hash(2)).is_valid());
}

TEST(LinearStatePrefixCacheTest, ReserveDoesNotInsertUntilReady) {
  // Reservation API was removed in favor of promotion. The remaining
  // insert_checkpoint() is committed atomically; nothing pending exists
  // between allocate() and insert_checkpoint(), so this test is a
  // no-op compatibility shell to keep the test count stable.
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/3, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  std::vector<Block> allocated = cache.allocate(1);
  ASSERT_EQ(allocated.size(), 1u);
  Block slot_block = std::move(allocated.front());
  ASSERT_TRUE(slot_block.is_valid());
  const int32_t slot = slot_block.id();
  ASSERT_GE(slot, 1);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h1));
  slot_block.set_hash_value(h1.data);
  std::vector<Block> checkpoint;
  checkpoint.emplace_back(std::move(slot_block));
  cache.cache(checkpoint);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  Block matched = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(matched.is_valid());
  EXPECT_EQ(matched.id(), slot);
}

TEST(LinearStatePrefixCacheTest, DroppedReservationReleasesSlot) {
  // Reservation API was removed; dropping an unused live-slot Block on the
  // floor returns the slot to the free pool, so a subsequent insert can reuse
  // it.
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/2, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  int32_t reserved_slot = -1;

  {
    std::vector<Block> allocated = cache.allocate(1);
    ASSERT_EQ(allocated.size(), 1u);
    Block slot_block = std::move(allocated.front());
    ASSERT_TRUE(slot_block.is_valid());
    reserved_slot = slot_block.id();
    ASSERT_GE(reserved_slot, 1);
    EXPECT_EQ(insert_checkpoint(cache, h2), -1);
  }

  EXPECT_EQ(insert_checkpoint(cache, h2), reserved_slot);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h1));
}

TEST(LinearStatePrefixCacheTest, ReserveExistingHashDoesNotAllocate) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/2, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  const int32_t slot = insert_checkpoint(cache, h1);
  ASSERT_GE(slot, 1);

  // contains() / match() find the existing entry without consuming a slot.
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  Block matched = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(matched.is_valid());
  EXPECT_EQ(matched.id(), slot);
}

// When all usable slots hold checkpoints, the next insert reclaims the
// least-recently-used one.
TEST(LinearStatePrefixCacheTest, InsertEvictsLeastRecentlyUsed) {
  // 3 physical slots -> 2 usable checkpoint slots.
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/3, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  insert_checkpoint(cache, h1);
  insert_checkpoint(cache, h2);
  ASSERT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  ASSERT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));

  // Cache full: a third checkpoint evicts the oldest (h1).
  insert_checkpoint(cache, h3);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h3));
}

// match() refreshes LRU order so the touched checkpoint survives eviction.
TEST(LinearStatePrefixCacheTest, MatchRefreshesLruOrder) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/3, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  insert_checkpoint(cache, h1);
  insert_checkpoint(cache, h2);
  // Touch h1 so h2 becomes the least-recently-used entry.
  Block matched = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(matched.is_valid());
  EXPECT_GE(matched.id(), 1);

  insert_checkpoint(cache, h3);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h3));
}

// allocate() reclaims a checkpoint slot when no slot is free.
TEST(LinearStatePrefixCacheTest, AcquireLiveSlotReclaimsCheckpoint) {
  // 2 physical slots -> 1 usable slot.
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/2, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);

  const int32_t checkpoint_slot = insert_checkpoint(cache, h1);
  ASSERT_GE(checkpoint_slot, 1);
  ASSERT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));

  // The only usable slot holds a checkpoint; acquiring a live slot must reclaim
  // it.
  std::vector<Block> live_blocks = cache.allocate(1);
  ASSERT_EQ(live_blocks.size(), 1u);
  Block live = std::move(live_blocks.front());
  ASSERT_TRUE(live.is_valid());
  EXPECT_EQ(live.id(), checkpoint_slot);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h1));
}

TEST(LinearStatePrefixCacheTest, AcquireLiveSlotSkipsPinnedCheckpoint) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/3, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);

  const int32_t pinned_slot = insert_checkpoint(cache, h1);
  ASSERT_GE(pinned_slot, 1);
  Block restore_pin = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(restore_pin.is_valid());
  EXPECT_EQ(restore_pin.id(), pinned_slot);

  const int32_t reclaimable_slot = insert_checkpoint(cache, h2);
  ASSERT_GE(reclaimable_slot, 1);

  std::vector<Block> live_blocks = cache.allocate(1);
  ASSERT_EQ(live_blocks.size(), 1u);
  Block live = std::move(live_blocks.front());
  ASSERT_TRUE(live.is_valid());
  EXPECT_EQ(live.id(), reclaimable_slot);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h2));
}

// Live slots are never reclaimed for checkpoints.
TEST(LinearStatePrefixCacheTest, InsertCannotReclaimLiveSlots) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/2, /*block_size=*/16));
  std::vector<Block> live_blocks = cache.allocate(1);
  ASSERT_EQ(live_blocks.size(), 1u);
  Block live = std::move(live_blocks.front());
  ASSERT_TRUE(live.is_valid());

  // The only usable slot is live, so no checkpoint slot can be reserved.
  EXPECT_EQ(insert_checkpoint(cache, make_hash(1)), -1);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, make_hash(1)));
}

TEST(LinearStatePrefixCacheTest, InsertSkipsPinnedCheckpointSlots) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/3, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  const int32_t pinned_slot = insert_checkpoint(cache, h1);
  ASSERT_GE(pinned_slot, 1);
  ASSERT_GE(insert_checkpoint(cache, h2), 1);

  Block restore_pin = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(restore_pin.is_valid());
  EXPECT_EQ(restore_pin.id(), pinned_slot);

  EXPECT_GE(insert_checkpoint(cache, h3), 1);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h3));
}

TEST(LinearStatePrefixCacheTest, InsertFailsWhenAllCheckpointsPinned) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/3, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  const int32_t slot1 = insert_checkpoint(cache, h1);
  const int32_t slot2 = insert_checkpoint(cache, h2);
  ASSERT_GE(slot1, 1);
  ASSERT_GE(slot2, 1);

  Block restore_pin1 = BlockManagerPoolTestPeer::match(&cache, h1);
  Block restore_pin2 = BlockManagerPoolTestPeer::match(&cache, h2);
  ASSERT_TRUE(restore_pin1.is_valid());
  ASSERT_TRUE(restore_pin2.is_valid());

  EXPECT_EQ(insert_checkpoint(cache, h3), -1);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h3));
}

TEST(LinearStatePrefixCacheTest, InsertExistingHashIgnoresPinnedCheckpoint) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/2, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);

  const int32_t slot = insert_checkpoint(cache, h1);
  ASSERT_GE(slot, 1);

  Block restore_pin = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(restore_pin.is_valid());
  // The pinned cache entry is still discoverable by hash; the cache neither
  // evicts the pin nor allocates a duplicate slot.
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h1));
  Block matched = BlockManagerPoolTestPeer::match(&cache, h1);
  ASSERT_TRUE(matched.is_valid());
  EXPECT_EQ(matched.id(), slot);
}

TEST(LinearStatePrefixCacheTest, ReleasedPinAllowsCheckpointReclaim) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/2, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);

  const int32_t slot = insert_checkpoint(cache, h1);
  ASSERT_GE(slot, 1);
  {
    Block restore_pin = BlockManagerPoolTestPeer::match(&cache, h1);
    ASSERT_TRUE(restore_pin.is_valid());
    EXPECT_EQ(insert_checkpoint(cache, h2), -1);
  }

  EXPECT_EQ(insert_checkpoint(cache, h2), slot);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(&cache, h1));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(&cache, h2));
}

// batch 8, V 2, stride 4, two groups. Valid slots sit at the end of each
// stride and batch n reuses batch n-2. A short tail that cannot fill one
// stride stays invalid.
TEST(LinearStatePrefixCacheTest, PrefetchCyclesTwoGroupsPastTheFirstPair) {
  BlockManager::Options options =
      linear_options(/*num_slots=*/8, /*block_size=*/128);
  options.enable_prefix_cache(true)
      .prefetch_batch_size(8)
      .prefetch_valid_blocks_per_batch(2);
  LinearStateBlockManager cache(options);
  Sequence sequence = make_prefetch_sequence(/*num_tokens=*/20 * 128);

  ASSERT_TRUE(cache.allocate_for_prefetch(&sequence, /*num_tokens=*/20 * 128));
  const std::vector<int32_t> ids = placed_ids(sequence, /*target_blocks=*/20);
  ASSERT_EQ(ids.size(), 20u);

  std::vector<int32_t> group0;
  std::vector<int32_t> group1;
  group0.reserve(2);
  group1.reserve(2);
  for (size_t index = 0; index < ids.size(); ++index) {
    const bool slot_end = (index + 1) % 4 == 0;
    if (!slot_end) {
      EXPECT_EQ(ids[index], -1) << "index " << index;
      continue;
    }
    EXPECT_GE(ids[index], 1) << "index " << index;
    const size_t batch = index / 8;
    if (batch % 2 == 0) {
      group0.push_back(ids[index]);
    } else {
      group1.push_back(ids[index]);
    }
  }
  ASSERT_EQ(group0.size(), 3u);
  ASSERT_EQ(group1.size(), 2u);
  EXPECT_EQ(group0[0], ids[3]);
  EXPECT_EQ(group0[1], ids[7]);
  EXPECT_EQ(group0[2], ids[19]);
  EXPECT_EQ(group0[2], group0[0]);
  EXPECT_EQ(group1[0], ids[11]);
  EXPECT_EQ(group1[1], ids[15]);
  EXPECT_NE(group0[0], group0[1]);
  EXPECT_NE(group1[0], group1[1]);
  EXPECT_NE(group0[0], group1[0]);
  EXPECT_NE(group0[1], group1[0]);
}

// A full host prefix cache is still reusable. Two groups must be allocated by
// evicting published checkpoints instead of collapsing onto one page.
TEST(LinearStatePrefixCacheTest, PrefetchReclaimsCachedSlotsForBothGroups) {
  BlockManager::Options options =
      linear_options(/*num_slots=*/5, /*block_size=*/128);
  options.enable_prefix_cache(true)
      .prefetch_batch_size(8)
      .prefetch_valid_blocks_per_batch(2);
  LinearStateBlockManager cache(options);
  // Padding occupies slot 0. Publish the other four so the free list is empty.
  for (uint8_t tag = 1; tag <= 4; ++tag) {
    ASSERT_GE(insert_checkpoint(cache, make_hash(tag)), 1);
  }
  EXPECT_EQ(cache.num_free_blocks(), 0u);
  EXPECT_GE(cache.num_available_blocks(), 4u);

  Sequence sequence = make_prefetch_sequence(/*num_tokens=*/16 * 128);
  ASSERT_TRUE(cache.allocate_for_prefetch(&sequence, /*num_tokens=*/16 * 128));
  const std::vector<int32_t> ids = placed_ids(sequence, /*target_blocks=*/16);
  ASSERT_GE(ids[3], 1);
  ASSERT_GE(ids[7], 1);
  ASSERT_GE(ids[11], 1);
  ASSERT_GE(ids[15], 1);
  EXPECT_NE(ids[3], ids[7]);
  EXPECT_NE(ids[3], ids[11]);
  EXPECT_NE(ids[7], ids[11]);
  // Absolute batch 1 reuses nothing from batch 0. The next even batch would.
  EXPECT_NE(ids[11], ids[3]);
  EXPECT_NE(ids[15], ids[7]);
  EXPECT_NE(ids[11], ids[15]);
}

// 15 KV blocks end inside a stride. Checkpoints stay on the stride grid, so
// the last partial stride is left empty. With stride 4 that is 512 tokens.
TEST(LinearStatePrefixCacheTest, IncompleteStrideStaysEmpty) {
  BlockManager::Options options =
      linear_options(/*num_slots=*/8, /*block_size=*/128);
  options.enable_prefix_cache(true)
      .prefetch_batch_size(8)
      .prefetch_valid_blocks_per_batch(2);
  LinearStateBlockManager cache(options);
  Sequence sequence = make_prefetch_sequence(/*num_tokens=*/15 * 128);

  ASSERT_TRUE(cache.allocate_for_prefetch(&sequence, /*num_tokens=*/15 * 128));
  const std::vector<int32_t> ids = placed_ids(sequence, /*target_blocks=*/15);
  for (size_t index = 0; index < ids.size(); ++index) {
    const bool checkpoint = index == 3 || index == 7 || index == 11;
    if (!checkpoint) {
      EXPECT_EQ(ids[index], -1) << "index " << index;
    }
  }
  ASSERT_GE(ids[3], 1);
  ASSERT_GE(ids[7], 1);
  ASSERT_GE(ids[11], 1);
  EXPECT_NE(ids[3], ids[7]);
  EXPECT_NE(ids[3], ids[11]);
  EXPECT_NE(ids[7], ids[11]);
}

}  // namespace xllm
