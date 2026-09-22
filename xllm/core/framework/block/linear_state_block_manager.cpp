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
#include <limits>
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

bool crosses_checkpoint(size_t begin, size_t end, size_t checkpoint_stride) {
  CHECK_LE(begin, end);
  CHECK_GT(checkpoint_stride, 0u);
  return begin / checkpoint_stride != end / checkpoint_stride;
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
}

std::optional<std::vector<Block>>
LinearStateBlockManager::allocate_for_sequence(Sequence* seq,
                                               KVCacheState& kv_state,
                                               size_t /*num_tokens*/) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  if (seq->is_prefill_stage()) {
    if (!options_.instance_is_decode()) {
      return allocate_prefill(seq, kv_state);
    }
    retain_read_source(kv_state);
    return allocate_decode(seq, kv_state);
  }
  return allocate_decode(seq, kv_state);
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
    Sequence* seq,
    KVCacheState& kv_state) {
  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::LINEAR);
  CHECK_LE(blocks->size(), 2u);
  if (seq->is_prefill_stage()) {
    if (!blocks->empty()) {
      return std::vector<Block>{};
    }
    std::vector<Block> allocated = BlockManagerImpl::allocate(1);
    return allocated.empty()
               ? std::nullopt
               : std::optional<std::vector<Block>>(std::move(allocated));
  }

  const size_t confirmed_tokens = seq->last_confirmed_cached_tokens_num();
  const size_t tokens_per_step =
      static_cast<size_t>(options_.num_speculative_tokens()) + 1;
  const size_t checkpoint_stride = block_size();

  // Prefill may leave its read source in front of the newest write block.
  // Decode starts from the newest state, so discard that historical prefill
  // source before applying the decode rolling rules.  In particular, do not
  // treat the prefill pair as a pending decode window.
  if (blocks->size() == 2 &&
      kv_state.kv_cache_tokens_num() <= seq->num_prompt_tokens()) {
    retain_read_source(kv_state);
  }

  if (blocks->size() == 2) {
    const size_t previous_begin = confirmed_tokens > tokens_per_step
                                      ? confirmed_tokens - tokens_per_step
                                      : 0;
    if (crosses_checkpoint(
            previous_begin, confirmed_tokens, checkpoint_stride)) {
      // TODO: Offload the first block before releasing it once LINEAR
      // hierarchy cache can preserve the confirmed checkpoint state.
      deallocate(Slice<Block>(*blocks).slice(0, 1));
      blocks->erase(blocks->begin());
      kv_state.set_shared_blocks_num(BlockType::LINEAR, 0);
    } else {
      // The previous decode step has not yet confirmed a checkpoint. Keep
      // both slots alive and alternate their roles so the next out-of-place
      // step reads the slot written by the preceding step. This is also
      // what makes overlap allocation safe: the old read slot cannot be
      // returned to the global pool while its in-flight step may still use it.
      std::iter_swap(blocks->begin(), blocks->begin() + 1);
      return std::vector<Block>{};
    }
  }

  if (blocks->empty()) {
    std::vector<Block> allocated = BlockManagerImpl::allocate(1);
    return allocated.empty()
               ? std::nullopt
               : std::optional<std::vector<Block>>(std::move(allocated));
  }

  // A stride of one is the legacy fallback when chunked prefill is disabled.
  // It does not describe a persistent LINEAR checkpoint window to roll.
  if (checkpoint_stride == 1) {
    return std::vector<Block>{};
  }

  CHECK_LE(tokens_per_step,
           std::numeric_limits<size_t>::max() - confirmed_tokens)
      << "linear-state confirmed-token cursor overflow";
  const size_t predicted_tokens = confirmed_tokens + tokens_per_step;
  if (!crosses_checkpoint(
          confirmed_tokens, predicted_tokens, checkpoint_stride)) {
    return std::vector<Block>{};
  }

  std::vector<Block> allocated = BlockManagerImpl::allocate(1);
  if (allocated.empty()) {
    // The extra slot only preserves a checkpoint for the future offload path;
    // recurrent decode itself supports reading and writing the live slot in
    // place. Under pressure, keep decoding instead of turning an optional
    // checkpoint copy into a sequence preemption.
    return std::vector<Block>{};
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
