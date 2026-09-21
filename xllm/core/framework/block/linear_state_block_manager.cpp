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
                                               KVCacheState& kv_state,
                                               size_t /*num_tokens*/) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  if (options_.instance_is_decode() || !seq->is_prefill_stage()) {
    retain_read_source(kv_state);
    return allocate_decode(kv_state);
  }
  return allocate_prefill(seq, kv_state);
}

void LinearStateBlockManager::retain_read_source(KVCacheState& kv_state) {
  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::LINEAR);
  if (blocks->empty()) {
    return;
  }
  CHECK(blocks->back().is_valid());
  const bool shared_source =
      kv_state.shared_blocks_num(BlockType::LINEAR) == blocks->size();
  const size_t keep_begin = blocks->size() - 1;
  deallocate(Slice<Block>(*blocks).slice(0, keep_begin));
  blocks->erase(blocks->begin(), blocks->begin() + keep_begin);
  kv_state.set_shared_blocks_num(BlockType::LINEAR, shared_source ? 1 : 0);
}

void LinearStateBlockManager::cache_read_source(Sequence* seq,
                                                KVCacheState& kv_state) {
  const Slice<Block> blocks = kv_state.blocks(BlockType::LINEAR);
  if (seq->is_graph_warmup() || prefix_cache_ == nullptr || blocks.empty()) {
    return;
  }
  const size_t chunk_stride = block_size();
  const size_t cached_tokens = seq->kv_cache_tokens_num();
  if (cached_tokens % chunk_stride != 0) {
    return;
  }
  const size_t checkpoint_index = cached_tokens / chunk_stride;
  if (checkpoint_index <= kv_state.num_cached_blocks(BlockType::LINEAR)) {
    return;
  }
  seq->update_linear_state_hashes(block_size());
  const Slice<XXH3Key> hashes = seq->linear_state_hashes();
  CHECK_GE(hashes.size(), checkpoint_index);
  auto* source = kv_state.mutable_blocks(BlockType::LINEAR);
  CHECK(source->back().is_valid());
  source->back().set_hash_value(hashes[checkpoint_index - 1].data);
  prefix_cache_->insert(
      Slice<Block>(*source).slice(source->size() - 1, source->size()));
  kv_state.set_num_cached_blocks(BlockType::LINEAR, checkpoint_index);
}

std::optional<std::vector<Block>> LinearStateBlockManager::allocate_prefill(
    Sequence* seq,
    KVCacheState& kv_state) {
  cache_read_source(seq, kv_state);
  retain_read_source(kv_state);
  std::vector<Block> allocated = BlockManagerImpl::allocate(1);
  if (allocated.empty()) {
    return std::nullopt;
  }
  return allocated;
}

std::optional<std::vector<Block>> LinearStateBlockManager::allocate_decode(
    const KVCacheState& kv_state) {
  if (kv_state.num_blocks(BlockType::LINEAR) > 0) {
    return std::vector<Block>{};
  }
  std::vector<Block> allocated = BlockManagerImpl::allocate(1);
  if (allocated.empty()) {
    return std::nullopt;
  }
  return allocated;
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
