/* Copyright 2026 The xLLM Authors.

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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "framework/batch/batch.h"
#include "framework/batch/batch_factory.h"
#include "framework/block/block_manager_impl.h"

namespace xllm {
namespace {

class BatchInflightBlocksTest : public ::testing::Test {
 protected:
  Sequence make_sequence(size_t prompt_tokens) {
    SequenceParams params;
    params.seq_capacity = prompt_tokens + 64;
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;
    return Sequence(0,
                    std::vector<int32_t>(prompt_tokens, 1),
                    torch::Tensor(),
                    MMData(),
                    IncrementalDecoder("", prompt_tokens, false, true),
                    params);
  }

  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
};

TEST_F(BatchInflightBlocksTest, CopiesAndMovesKeepBlocksUntilLastBatchRetires) {
  for (const bool prefix_cache : {false, true}) {
    SCOPED_TRACE(prefix_cache);
    BlockManager::Options options;
    options.num_blocks(8).block_size(4).enable_prefix_cache(prefix_cache);
    BlockManagerImpl manager(options);
    Sequence sequence = make_sequence(8);
    sequence.kv_state().add_blocks(BlockType::KV, manager.allocate(3));
    sequence.kv_state().set_kv_cache_tokens_num(4);
    Batch original;
    original.add(&sequence, 1);
    const Slice<Block> blocks = sequence.kv_state().blocks(BlockType::KV);
    const std::vector<int32_t> held_ids = {
        blocks[0].id(), blocks[1].id(), blocks[2].id()};
    Batch copied = original;
    Batch moved = std::move(copied);
    manager.cache(sequence.tokens(),
                  *sequence.kv_state().mutable_blocks(BlockType::KV));
    sequence.reset();
    EXPECT_EQ(manager.num_used_blocks(), held_ids.size());
    original = Batch();
    EXPECT_EQ(manager.num_used_blocks(), held_ids.size());
    manager.reset_prefix_cache();
    EXPECT_EQ(manager.num_free_blocks(),
              manager.num_total_blocks() - held_ids.size());
    std::vector<Block> destinations =
        manager.allocate(manager.num_free_blocks());
    for (const Block& block : destinations) {
      EXPECT_EQ(std::count(held_ids.begin(), held_ids.end(), block.id()), 0);
    }
    EXPECT_TRUE(manager.allocate(1).empty());
    moved = Batch();
    EXPECT_EQ(manager.num_used_blocks(), destinations.size());
    EXPECT_EQ(manager.num_available_blocks(), held_ids.size());
    std::vector<Block> reused = manager.allocate(held_ids.size());
    ASSERT_EQ(reused.size(), held_ids.size());
    for (const Block& block : reused) {
      EXPECT_EQ(std::count(held_ids.begin(), held_ids.end(), block.id()), 1);
    }
    reused.clear();
    destinations.clear();
    manager.reset_prefix_cache();
    EXPECT_EQ(manager.num_used_blocks(), 0u);
    EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks());
  }
}

TEST_F(BatchInflightBlocksTest, LinearKeepaliveDoesNotCreateLogicalSharing) {
  BlockManager::Options options;
  options.num_blocks(4).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);
  Sequence sequence = make_sequence(4);
  sequence.kv_state().add_blocks(BlockType::LINEAR, manager.allocate(2));
  Batch batch(&sequence);
  Batch copied = batch;
  const Slice<Block> blocks = sequence.kv_state().blocks(BlockType::LINEAR);
  for (const Block& block : blocks) {
    EXPECT_EQ(block.ref_count(), 3u);
    EXPECT_EQ(block.owner_count(), 1u);
    EXPECT_FALSE(block.has_multiple_owners());
  }

  Block shared_source = blocks.front();
  EXPECT_TRUE(blocks.front().has_multiple_owners());
  EXPECT_FALSE(blocks.back().has_multiple_owners());
  shared_source = Block();
  EXPECT_FALSE(blocks.front().has_multiple_owners());

  sequence.reset();
  EXPECT_EQ(manager.num_used_blocks(), 2u);
  batch = Batch();
  EXPECT_EQ(manager.num_used_blocks(), 2u);
  copied = Batch();
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

TEST_F(BatchInflightBlocksTest, IndependentBatchesReleaseTheirOwnReferences) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);
  Sequence first = make_sequence(4);
  Sequence second = make_sequence(4);
  first.kv_state().add_blocks(BlockType::KV, manager.allocate(1));
  second.kv_state().add_blocks(BlockType::KV, manager.allocate(1));
  Batch first_batch(&first);
  Batch second_batch(&second);
  first.reset();
  second.reset();
  EXPECT_EQ(manager.num_used_blocks(), 2u);
  first_batch = Batch();
  EXPECT_EQ(manager.num_used_blocks(), 1u);
  second_batch = Batch();
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

TEST_F(BatchInflightBlocksTest, KeepsBlocksForSequenceGroups) {
  BlockManager::Options options;
  options.num_blocks(4).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);
  const std::string prompt = "prompt";
  const std::vector<int32_t> prompt_tokens(4, 1);
  const torch::Tensor embedding;
  const MMData mm_data;
  SequenceParams params;
  params.seq_capacity = 32;
  params.sampling_param = &sampling_param_;
  params.stopping_checker = &stopping_checker_;
  SequencesGroup group(prompt, prompt_tokens, embedding, mm_data, params);
  Sequence* sequence = group.sequences().front().get();
  sequence->kv_state().add_blocks(BlockType::KV, manager.allocate(1));
  Batch batch;
  batch.add(&group);
  EXPECT_EQ(sequence->kv_state().blocks(BlockType::KV).front().ref_count(), 2u);
  Batch ordinary(sequence);
  ordinary.add(&group);
  EXPECT_EQ(sequence->kv_state().blocks(BlockType::KV).front().ref_count(), 3u);
  sequence->reset();
  ordinary = Batch();
  EXPECT_EQ(manager.num_used_blocks(), 1u);
  batch = Batch();
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

TEST_F(BatchInflightBlocksTest, HoldsSwaTailForEachSequence) {
  BlockManager::Options options;
  options.num_blocks(12)
      .block_size(4)
      .block_type(BlockType::SWA)
      .enable_prefix_cache(false);
  BlockManagerImpl manager(options);
  Sequence first = make_sequence(64);
  Sequence second = make_sequence(64);
  first.kv_state().add_blocks(BlockType::SWA, manager.allocate(3));
  second.kv_state().add_blocks(BlockType::SWA, manager.allocate(5));
  first.kv_state().set_kv_cache_tokens_num(12);
  second.kv_state().set_kv_cache_tokens_num(20);
  Batch batch;
  batch.add(&first, 1);
  batch.add(&second, 1);
  for (const Sequence* sequence : {&first, &second}) {
    const Slice<Block> blocks = sequence->kv_state().blocks(BlockType::SWA);
    for (size_t index = 0; index < blocks.size(); ++index) {
      EXPECT_EQ(blocks[index].ref_count(),
                index + 1 == blocks.size() ? 2u : 1u);
    }
  }
  first.reset();
  second.reset();
  EXPECT_EQ(manager.num_used_blocks(), 2u);
  batch = Batch();
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

TEST_F(BatchInflightBlocksTest, FactoryConstructsEachRankWithItsOwnBlocks) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);
  Sequence first = make_sequence(4);
  Sequence second = make_sequence(4);
  first.set_dp_rank(0);
  second.set_dp_rank(1);
  first.kv_state().add_blocks(BlockType::KV, manager.allocate(1));
  second.kv_state().add_blocks(BlockType::KV, manager.allocate(1));
  std::vector<Batch> batches = BatchFactory::get_instance(2)->create_batches(
      {}, {&first, &second}, {1, 1});
  ASSERT_EQ(batches.size(), 2u);
  EXPECT_EQ(batches[0].get_sequences().front(), &first);
  EXPECT_EQ(batches[1].get_sequences().front(), &second);
  first.reset();
  second.reset();
  EXPECT_EQ(manager.num_used_blocks(), 2u);
  batches[0] = Batch();
  EXPECT_EQ(manager.num_used_blocks(), 1u);
  batches[1] = Batch();
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

}  // namespace
}  // namespace xllm
