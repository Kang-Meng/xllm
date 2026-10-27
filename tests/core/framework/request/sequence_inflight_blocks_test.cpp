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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/config/speculative_config.h"
#include "framework/block/block_manager_impl.h"
#include "framework/request/sequence.h"

namespace xllm {
namespace {

class SequenceInflightBlocksTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& config = SpeculativeConfig::get_instance();
    previous_speculative_tokens_ = config.num_speculative_tokens();
    previous_algorithm_ = config.speculative_algorithm();
    config.num_speculative_tokens() = 0;
    config.speculative_algorithm() = "DSpark";
  }

  void TearDown() override {
    auto& config = SpeculativeConfig::get_instance();
    config.num_speculative_tokens() = previous_speculative_tokens_;
    config.speculative_algorithm() = previous_algorithm_;
  }

  Sequence make_sequence(size_t prompt_tokens) {
    SequenceParams params;
    params.seq_capacity = prompt_tokens + 512;
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;
    params.enable_schedule_overlap = true;
    return Sequence(0,
                    std::vector<int32_t>(prompt_tokens, 1),
                    torch::Tensor(),
                    MMData(),
                    IncrementalDecoder("", prompt_tokens, false, true),
                    params);
  }

  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
  int32_t previous_speculative_tokens_ = 0;
  std::string previous_algorithm_;
};

std::vector<int32_t> block_ids(const std::vector<Block>& blocks) {
  std::vector<int32_t> ids;
  ids.reserve(blocks.size());
  for (const Block& block : blocks) {
    ids.emplace_back(block.id());
  }
  return ids;
}

TEST_F(SequenceInflightBlocksTest, EmptyAndNonOwningTypesReturnNoBlocks) {
  Sequence sequence = make_sequence(8);
  EXPECT_TRUE(sequence.get_inflight_blocks().empty());
  sequence.kv_state().add_blocks(BlockType::KV, {Block(), Block(7, nullptr)});
  sequence.kv_state().add_blocks(BlockType::LINEAR,
                                 {Block(), Block(6, nullptr)});
  sequence.kv_state().add_blocks(BlockType::EMBEDDING, {Block(5, nullptr)});
  EXPECT_TRUE(sequence.get_inflight_blocks().empty());
}

TEST_F(SequenceInflightBlocksTest, MixedTypesRetainTheirOwnRangesAndSlots) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl kv(options);
  options.block_size(16).block_type(BlockType::C4);
  BlockManagerImpl compressed(options);
  options.block_size(1).block_type(BlockType::LINEAR);
  BlockManagerImpl linear(options);
  options.block_type(BlockType::EMBEDDING);
  BlockManagerImpl embedding(options);
  Sequence sequence = make_sequence(64);
  sequence.kv_state().add_blocks(BlockType::EMBEDDING, embedding.allocate(1));
  sequence.kv_state().add_blocks(BlockType::C4, compressed.allocate(4));
  sequence.kv_state().add_blocks(BlockType::LINEAR, linear.allocate(2));
  sequence.kv_state().add_blocks(BlockType::KV, kv.allocate(6));
  sequence.kv_state().set_kv_cache_tokens_num(18);

  std::vector<Block> retained = sequence.get_inflight_blocks();
  ASSERT_EQ(retained.size(), 8u);
  const Slice<Block> kv_blocks = sequence.kv_state().blocks(BlockType::KV);
  for (size_t index = 0; index < kv_blocks.size(); ++index) {
    EXPECT_EQ(kv_blocks[index].ref_count(), index >= 4 ? 2u : 1u);
  }
  const Slice<Block> compressed_blocks =
      sequence.kv_state().blocks(BlockType::C4);
  for (size_t index = 0; index < compressed_blocks.size(); ++index) {
    EXPECT_EQ(compressed_blocks[index].ref_count(), index >= 1 ? 2u : 1u);
  }
  for (const Block& block : sequence.kv_state().blocks(BlockType::LINEAR)) {
    EXPECT_EQ(block.ref_count(), 2u);
  }
  EXPECT_EQ(
      sequence.kv_state().blocks(BlockType::EMBEDDING).front().ref_count(), 2u);

  sequence.reset();
  EXPECT_EQ(kv.num_used_blocks(), 2u);
  EXPECT_EQ(compressed.num_used_blocks(), 3u);
  EXPECT_EQ(linear.num_used_blocks(), 2u);
  EXPECT_EQ(embedding.num_used_blocks(), 1u);
  retained.clear();
  EXPECT_EQ(kv.num_used_blocks(), 0u);
  EXPECT_EQ(compressed.num_used_blocks(), 0u);
  EXPECT_EQ(linear.num_used_blocks(), 0u);
  EXPECT_EQ(embedding.num_used_blocks(), 0u);
}

TEST_F(SequenceInflightBlocksTest, PrefillRetainsTailFromCachedTokens) {
  SpeculativeConfig::get_instance().num_speculative_tokens() = 64;
  SpeculativeConfig::get_instance().speculative_algorithm() = "MTP";
  struct CacheType {
    BlockType type;
    size_t ratio;
  };
  const std::vector<CacheType> types = {{BlockType::KV, 1},
                                        {BlockType::SWA, 1},
                                        {BlockType::C1, 1},
                                        {BlockType::C2, 2},
                                        {BlockType::C4, 4},
                                        {BlockType::C128, 128}};
  for (const CacheType& cache : types) {
    const size_t block_size = 4 * cache.ratio;
    BlockManager::Options options;
    options.num_blocks(8)
        .block_size(static_cast<int32_t>(block_size))
        .block_type(cache.type)
        .enable_prefix_cache(false);
    BlockManagerImpl manager(options);
    Sequence sequence = make_sequence(6 * block_size);
    sequence.kv_state().add_blocks(cache.type, manager.allocate(5));
    const Slice<Block> blocks = sequence.kv_state().blocks(cache.type);
    for (const size_t cached_tokens : {size_t{0},
                                       size_t{1},
                                       size_t{2},
                                       cache.ratio - 1,
                                       block_size - 1,
                                       block_size,
                                       5 * block_size}) {
      sequence.kv_state().set_kv_cache_tokens_num(cached_tokens);
      SCOPED_TRACE(static_cast<int32_t>(cache.type));
      SCOPED_TRACE(cached_tokens);
      std::vector<int32_t> expected;
      expected.reserve(blocks.size());
      const size_t token_begin = cached_tokens > 2 ? cached_tokens - 2 : 0;
      for (size_t block_index = token_begin / block_size;
           block_index < blocks.size();
           ++block_index) {
        expected.emplace_back(blocks[block_index].id());
      }
      EXPECT_EQ(block_ids(sequence.get_inflight_blocks()), expected);
    }
  }
}

TEST_F(SequenceInflightBlocksTest, DecodeRetainsConservativeTail) {
  for (const size_t speculative_tokens : {0u, 1u, 15u}) {
    for (const bool needs_previous_token : {false, true}) {
      if (speculative_tokens == 0 && needs_previous_token) {
        continue;
      }
      SpeculativeConfig::get_instance().num_speculative_tokens() =
          static_cast<int32_t>(speculative_tokens);
      SpeculativeConfig::get_instance().speculative_algorithm() =
          needs_previous_token ? "MTP" : "DSpark";
      for (const int32_t block_size : {1, 4, 16, 128}) {
        SCOPED_TRACE(speculative_tokens);
        SCOPED_TRACE(needs_previous_token);
        SCOPED_TRACE(block_size);
        const size_t prompt_tokens = 17;
        const size_t token_begin = prompt_tokens - 1;
        const size_t speculative_budget =
            2 * speculative_tokens + needs_previous_token;
        const size_t token_end = prompt_tokens + 2 + speculative_budget;
        const size_t block_count =
            (token_end + block_size - 1) / block_size + 2;
        BlockManager::Options options;
        options.num_blocks(static_cast<uint32_t>(block_count + 1))
            .block_size(block_size)
            .enable_prefix_cache(false);
        BlockManagerImpl manager(options);
        Sequence sequence = make_sequence(prompt_tokens);
        sequence.kv_state().add_blocks(BlockType::KV,
                                       manager.allocate(block_count));
        sequence.kv_state().set_kv_cache_tokens_num(prompt_tokens);
        sequence.append_token(2);
        sequence.append_token(-1);
        ASSERT_EQ(sequence.num_valid_tokens(), prompt_tokens + 1);
        ASSERT_EQ(sequence.kv_state().kv_cache_tokens_num(), prompt_tokens + 1);
        std::vector<Block> retained = sequence.get_inflight_blocks();
        const Slice<Block> blocks = sequence.kv_state().blocks(BlockType::KV);
        const size_t first_block = token_begin / block_size;
        ASSERT_EQ(retained.size(), blocks.size() - first_block);
        for (size_t index = 0; index < blocks.size(); ++index) {
          EXPECT_EQ(blocks[index].ref_count(), index >= first_block ? 2u : 1u);
        }
        for (size_t accepted = 0; accepted <= speculative_tokens; ++accepted) {
          const size_t query_position = prompt_tokens + accepted + 1;
          const size_t write_begin = query_position - needs_previous_token;
          const size_t write_end =
              query_position + speculative_tokens + 1 + needs_previous_token;
          for (size_t position = write_begin; position < write_end;
               ++position) {
            EXPECT_EQ(blocks[position / block_size].ref_count(), 2u);
          }
        }
        sequence.reset();
        EXPECT_EQ(manager.num_used_blocks(), retained.size());
        retained.clear();
        EXPECT_EQ(manager.num_used_blocks(), 0u);
        EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks());
      }
    }
  }
}

TEST_F(SequenceInflightBlocksTest, RetainsReservedTailWithoutPredictingEnd) {
  BlockManager::Options options;
  options.num_blocks(80).block_size(1).enable_prefix_cache(false);
  BlockManagerImpl manager(options);
  Sequence sequence = make_sequence(8);
  sequence.kv_state().add_blocks(BlockType::KV, manager.allocate(72));
  sequence.kv_state().set_kv_cache_tokens_num(8);
  sequence.append_token(2);
  sequence.append_token(-1);
  std::vector<Block> retained = sequence.get_inflight_blocks();
  ASSERT_EQ(retained.size(), 65u);
  const Slice<Block> blocks = sequence.kv_state().blocks(BlockType::KV);
  EXPECT_EQ(retained.front().id(), blocks[7].id());
  EXPECT_EQ(retained.back().id(), blocks.back().id());
  EXPECT_EQ(blocks[6].ref_count(), 1u);
  EXPECT_EQ(blocks[56].ref_count(), 2u);
}

TEST_F(SequenceInflightBlocksTest, SwaSelectsTailUsingAbsoluteBlockIndices) {
  BlockManager::Options options;
  options.num_blocks(8)
      .block_size(4)
      .block_type(BlockType::SWA)
      .enable_prefix_cache(false);
  BlockManagerImpl manager(options);
  Sequence sequence = make_sequence(64);
  sequence.kv_state().add_blocks(BlockType::SWA, manager.allocate(6));
  std::vector<Block>* swa_blocks =
      sequence.kv_state().mutable_blocks(BlockType::SWA);
  (*swa_blocks)[0] = Block();
  (*swa_blocks)[1] = Block();
  const Slice<Block> blocks = sequence.kv_state().blocks(BlockType::SWA);
  sequence.kv_state().set_kv_cache_tokens_num(9);
  EXPECT_EQ(
      block_ids(sequence.get_inflight_blocks()),
      (std::vector<int32_t>{
          blocks[2].id(), blocks[3].id(), blocks[4].id(), blocks[5].id()}));
  sequence.kv_state().set_kv_cache_tokens_num(14);
  EXPECT_EQ(
      block_ids(sequence.get_inflight_blocks()),
      (std::vector<int32_t>{blocks[3].id(), blocks[4].id(), blocks[5].id()}));
  sequence.kv_state().set_kv_cache_tokens_num(24);
  EXPECT_EQ(block_ids(sequence.get_inflight_blocks()),
            (std::vector<int32_t>{blocks[5].id()}));
}

TEST_F(SequenceInflightBlocksTest, SlotsRetainAllHbmButNoHostBlocks) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl linear(options);
  BlockManagerImpl embedding(options);
  BlockManagerImpl host(options);
  Sequence sequence = make_sequence(8);
  sequence.kv_state().add_blocks(BlockType::LINEAR, linear.allocate(2));
  sequence.kv_state().add_blocks(BlockType::EMBEDDING, embedding.allocate(1));
  sequence.host_kv_state().add_blocks(BlockType::KV, host.allocate(3));
  sequence.kv_state().set_kv_cache_tokens_num(8);
  std::vector<Block> retained = sequence.get_inflight_blocks();
  ASSERT_EQ(retained.size(), 3u);
  const Slice<Block> linear_blocks =
      sequence.kv_state().blocks(BlockType::LINEAR);
  const Slice<Block> embedding_blocks =
      sequence.kv_state().blocks(BlockType::EMBEDDING);
  EXPECT_EQ(linear_blocks.front().ref_count(), 2u);
  EXPECT_EQ(linear_blocks.back().ref_count(), 2u);
  EXPECT_EQ(embedding_blocks.front().ref_count(), 2u);
  sequence.reset();
  EXPECT_EQ(linear.num_used_blocks(), 2u);
  EXPECT_EQ(embedding.num_used_blocks(), 1u);
  EXPECT_EQ(host.num_used_blocks(), 0u);
}

TEST_F(SequenceInflightBlocksTest, SkipsHolesAndNonOwningIds) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);
  Sequence sequence = make_sequence(32);
  std::vector<Block> blocks(5);
  std::vector<Block> allocated = manager.allocate(2);
  blocks[0] = Block(7, nullptr);
  blocks[1] = std::move(allocated[0]);
  blocks[3] = std::move(allocated[1]);
  sequence.kv_state().add_blocks(BlockType::KV, std::move(blocks));
  std::vector<Block> retained = sequence.get_inflight_blocks();
  ASSERT_EQ(retained.size(), 2u);
  EXPECT_EQ(retained[0].manager(), &manager);
  EXPECT_EQ(retained[1].manager(), &manager);
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_tokens());
  EXPECT_TRUE(sequence.get_inflight_blocks().empty());
}

TEST_F(SequenceInflightBlocksTest, FullyCachedTokensStillRetainAllocatedTail) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);
  Sequence sequence = make_sequence(8);
  sequence.kv_state().add_blocks(BlockType::KV, manager.allocate(4));
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_tokens());
  const Slice<Block> blocks = sequence.kv_state().blocks(BlockType::KV);
  EXPECT_EQ(
      block_ids(sequence.get_inflight_blocks()),
      (std::vector<int32_t>{blocks[1].id(), blocks[2].id(), blocks[3].id()}));
}

}  // namespace
}  // namespace xllm
