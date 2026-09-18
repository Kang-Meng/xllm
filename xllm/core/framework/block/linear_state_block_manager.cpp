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

BlockManager::Options make_linear_state_options(
    uint32_t num_slots,
    int32_t chunk_stride,
    bool enable_prefix_cache,
    bool instance_is_decode,
    uint32_t num_speculative_tokens) {
  BlockManager::Options options;
  options.num_blocks(num_slots);
  options.block_size(chunk_stride);
  options.enable_prefix_cache(enable_prefix_cache);
  options.enable_disagg_pd(false);
  options.block_type(BlockType::LINEAR);
  options.instance_is_decode(instance_is_decode);
  options.num_speculative_tokens(num_speculative_tokens);
  return options;
}

}  // namespace

LinearStateBlockManager::LinearStateBlockManager(
    uint32_t num_slots,
    int32_t chunk_stride,
    bool enable_prefix_cache,
    bool instance_is_decode,
    uint32_t num_speculative_tokens)
    : BlockManagerImpl(make_linear_state_options(num_slots,
                                                 chunk_stride,
                                                 enable_prefix_cache,
                                                 instance_is_decode,
                                                 num_speculative_tokens)) {
  CHECK_GT(num_slots, 1u)
      << "linear-state leaf needs at least one usable slot (plus padding)";
  CHECK_GT(chunk_stride, 0)
      << "linear-state leaf needs a positive chunk stride";
  CHECK_LE(static_cast<uint64_t>(num_speculative_tokens) + 1,
           static_cast<uint64_t>(chunk_stride))
      << "linear-state checkpoint stride must cover one speculative step";
}

std::optional<std::vector<Block>>
LinearStateBlockManager::allocate_for_sequence(Sequence* seq,
                                               size_t num_tokens) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  KVCacheState& kv_state = seq->kv_state();
  const bool decode_window =
      options_.instance_is_decode() || !seq->is_prefill_stage();
  const size_t chunk_stride = block_size();
  if (!decode_window) {
    const size_t cached_tokens = seq->kv_cache_tokens_num();
    CHECK_GT(num_tokens, cached_tokens);
    CHECK_LE(num_tokens, seq->num_tokens());
    CHECK_EQ(cached_tokens % chunk_stride, 0u);
    CHECK_LE(num_tokens - cached_tokens, chunk_stride);
    CHECK(num_tokens == seq->num_tokens() || num_tokens % chunk_stride == 0);
  }
  const size_t current = kv_state.last_confirmed_cached_tokens();
  const size_t max_step_tokens =
      static_cast<size_t>(options_.num_speculative_tokens()) + 1;
  const bool rotate = current % chunk_stride + max_step_tokens >= chunk_stride;
  trim_window(*seq, decode_window, current);
  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::LINEAR);
  if (!blocks->empty() && decode_window && !rotate) {
    CHECK(blocks->back().is_valid());
    return std::vector<Block>{};
  }
  Block slot = allocate();
  if (!slot.is_valid()) {
    return std::nullopt;
  }
  std::vector<Block> allocated(1);
  allocated.back() = std::move(slot);
  if (decode_window || seq->is_graph_warmup() || prefix_cache_ == nullptr ||
      blocks->empty()) {
    return allocated;
  }
  CHECK_EQ(blocks->size(), 1u);
  const size_t cached_tokens = seq->kv_cache_tokens_num();
  const size_t checkpoint_index = cached_tokens / chunk_stride;
  if (checkpoint_index <= kv_state.num_cached_blocks(BlockType::LINEAR)) {
    return allocated;
  }
  CHECK(blocks->front().is_valid());
  seq->update_linear_state_hashes(block_size());
  const Slice<XXH3Key> hashes = seq->linear_state_hashes();
  CHECK_GE(hashes.size(), checkpoint_index);
  blocks->front().set_hash_value(hashes[checkpoint_index - 1].data);
  Slice<Block> checkpoint = Slice<Block>(*blocks).slice(0, 1);
  prefix_cache_->insert(checkpoint);
  kv_state.set_num_cached_blocks(BlockType::LINEAR, checkpoint_index);
  return allocated;
}

void LinearStateBlockManager::trim_window(Sequence& seq,
                                          bool decode_window,
                                          size_t cached_tokens) {
  std::vector<Block>* blocks = seq.kv_state().mutable_blocks(BlockType::LINEAR);
  if (blocks->empty()) {
    return;
  }
  const size_t max_step_tokens =
      static_cast<size_t>(options_.num_speculative_tokens()) + 1;
  if (blocks->size() == 2 && decode_window && cached_tokens >= block_size() &&
      cached_tokens % block_size() < max_step_tokens) {
    // TODO: Publish the decode checkpoint before releasing the first block.
  }
  deallocate(Slice<Block>(*blocks).slice(0, blocks->size() - 1));
  blocks->erase(blocks->begin(), blocks->end() - 1);
  CHECK(blocks->back().is_valid());
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
