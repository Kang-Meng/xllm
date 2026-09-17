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

#include "core/framework/block/linear_state_block_manager.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "core/framework/request/sequence.h"

namespace xllm {

namespace {

BlockManager::Options make_linear_state_options(uint32_t num_slots,
                                                int32_t chunk_stride,
                                                bool enable_prefix_cache,
                                                bool instance_is_decode) {
  BlockManager::Options options;
  options.num_blocks(num_slots);
  options.block_size(chunk_stride);
  options.enable_prefix_cache(enable_prefix_cache);
  options.enable_disagg_pd(false);
  options.block_type(BlockType::LINEAR);
  options.instance_is_decode(instance_is_decode);
  return options;
}

}  // namespace

LinearStateBlockManager::LinearStateBlockManager(uint32_t num_slots,
                                                 int32_t chunk_stride,
                                                 bool enable_prefix_cache,
                                                 bool instance_is_decode)
    : BlockManagerImpl(make_linear_state_options(num_slots,
                                                 chunk_stride,
                                                 enable_prefix_cache,
                                                 instance_is_decode)) {
  CHECK_GT(num_slots, 1u)
      << "linear-state leaf needs at least one usable slot (plus padding)";
  CHECK_GT(chunk_stride, 0)
      << "linear-state leaf needs a positive chunk stride";
}

std::optional<std::vector<Block>>
LinearStateBlockManager::allocate_for_sequence(Sequence* seq,
                                               size_t num_tokens) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  const Slice<Block> blocks = seq->kv_state().blocks(BlockType::LINEAR);
  if (options_.instance_is_decode()) {
    if (!blocks.empty()) {
      CHECK_EQ(blocks.size(), 1u);
      CHECK(blocks.back().is_valid());
      return std::vector<Block>{};
    }
  } else if (!seq->is_prefill_stage()) {
    CHECK(!blocks.empty());
    CHECK(blocks.back().is_valid());
    return std::vector<Block>{};
  } else {
    const size_t cached_tokens = seq->kv_cache_tokens_num();
    const size_t chunk_stride = block_size();
    CHECK_GT(num_tokens, cached_tokens);
    CHECK_LE(num_tokens, seq->num_tokens());
    CHECK_EQ(cached_tokens % chunk_stride, 0u);
    CHECK_LE(num_tokens - cached_tokens, chunk_stride);
    CHECK(num_tokens == seq->num_tokens() || num_tokens % chunk_stride == 0);
    const size_t output_index = cached_tokens / chunk_stride;
    CHECK(blocks.size() == output_index || blocks.size() == output_index + 1);
    if (output_index > 0) {
      CHECK(blocks[output_index - 1].is_valid());
    }
    if (blocks.size() == output_index + 1) {
      CHECK(blocks.back().is_valid());
      return std::vector<Block>{};
    }
  }
  Block slot = allocate();
  if (!slot.is_valid()) {
    return std::nullopt;
  }
  std::vector<Block> allocated;
  allocated.reserve(1);
  allocated.emplace_back(std::move(slot));
  return allocated;
}

void LinearStateBlockManager::release_out_of_window(Sequence* seq) {
  if (seq == nullptr) {
    return;
  }
  std::vector<Block>* blocks =
      seq->kv_state().mutable_blocks(BlockType::LINEAR);
  const size_t keep =
      !options_.instance_is_decode() && seq->is_prefill_stage() ? 2 : 1;
  if (blocks == nullptr || blocks->size() <= keep) {
    return;
  }
  Block released = std::move((*blocks)[blocks->size() - keep - 1]);
  deallocate(Slice<Block>(&released, 1));
}

Block LinearStateBlockManager::allocate() {
  std::vector<Block> blocks = BlockManagerImpl::allocate(1);
  if (blocks.empty()) {
    return Block();
  }
  return std::move(blocks[0]);
}

std::vector<Block> LinearStateBlockManager::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& block_hashes) {
  if (token_ids.size() == 0 || prefix_cache_ == nullptr) {
    return {};
  }
  std::vector<Block> blocks = prefix_cache_->match(
      token_ids.slice(0, token_ids.size() - 1), {}, MMData(), block_hashes);
  for (const Block& block : blocks) {
    if (block.is_valid() && mark_used(&usage_accounted_ids_, block.id())) {
      num_used_blocks_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return blocks;
}

void LinearStateBlockManager::cache(const Slice<int32_t>& token_ids,
                                    std::vector<Block>& blocks,
                                    size_t existed_shared_blocks_num,
                                    const MMData& mm_data,
                                    const Slice<XXH3Key>& block_hashes) {
  if (blocks.empty() || prefix_cache_ == nullptr) {
    return;
  }
  const size_t publish_end =
      std::min(token_ids.size() / block_size(), blocks.size() - 1);
  if (publish_end <= existed_shared_blocks_num) {
    return;
  }
  CHECK_GE(block_hashes.size(), publish_end);
  BlockManagerImpl::cache(token_ids.slice(0, publish_end * block_size()),
                          blocks,
                          existed_shared_blocks_num,
                          mm_data,
                          block_hashes);
}

}  // namespace xllm
